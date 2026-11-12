"""CAL-1: port GoldWorm v6 from its tensor layout, reproduce the published eval, and close the residual frame.

Usage: python cal1.py <model.safetensors> <TinyStories-valid.txt> <out.json> [--gelu erf|tanh]
Frozen design: PREREG_CAL1.md (sha256 in PREREG_CAL1.sha256). No author code is imported or executed.
"""

import hashlib
import json
import math
import platform
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

OUTER = (17764697, 19447282)
W_CONTEXT, STRIDE, PER_WINDOW, CTX = 8, 13, 16, 256
TARGET_BPB = 1.007349040026128
N_HEAD, D = 4, 192


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Port:
    """GoldWorm v6 forward, with every residual write exposed and optionally zeroed."""

    def __init__(self, sd, gelu="erf"):
        self.sd = sd
        self.n_layer = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("blocks."))
        self.approx = "none" if gelu == "erf" else "tanh"

    def ln(self, x, name):
        return F.layer_norm(x, (D,), self.sd[f"{name}.weight"], self.sd[f"{name}.bias"], 1e-5)

    def forward(self, idx, zero=None):
        """idx [B,T] long. Returns logits, writes dict name->[B,T,D], final pre-ln_f residual."""
        sd = self.sd
        B, T = idx.shape
        emb = sd["wte.weight"][idx] + sd["wpe.weight"][torch.arange(T)][None]
        writes = {"embed": emb}
        x = emb
        mask = torch.ones(T, T, dtype=torch.bool).tril()
        for i in range(self.n_layer):
            h = self.ln(x, f"blocks.{i}.ln1")
            qkv = h @ sd[f"blocks.{i}.attn.qkv.weight"].T
            q, k, v = qkv.split(D, dim=2)
            q, k, v = (t.view(B, T, N_HEAD, D // N_HEAD).transpose(1, 2) for t in (q, k, v))
            att = (q @ k.transpose(-2, -1)) / math.sqrt(D // N_HEAD)
            att = att.masked_fill(~mask, float("-inf")).softmax(-1)
            y = (att @ v).transpose(1, 2).reshape(B, T, D) @ sd[f"blocks.{i}.attn.proj.weight"].T
            if zero == f"attn{i}":
                y = torch.zeros_like(y)
            writes[f"attn{i}"] = y
            x = x + y
            h = self.ln(x, f"blocks.{i}.ln2")
            m = F.gelu(h @ sd[f"blocks.{i}.mlp.fc.weight"].T, approximate=self.approx) @ sd[f"blocks.{i}.mlp.proj.weight"].T
            if zero == f"mlp{i}":
                m = torch.zeros_like(m)
            writes[f"mlp{i}"] = m
            x = x + m
        logits = self.ln(x, "ln_f") @ sd["head.weight"].T
        return logits, writes, x


def eval_positions():
    return np.array([OUTER[0] + W_CONTEXT + k * STRIDE for k in range((OUTER[1] - 1 - OUTER[0] - W_CONTEXT) // STRIDE + 1)])


@torch.no_grad()
def windowed_bpb(port, data, pos, batch=64):
    bits = np.empty(len(pos))
    windows = []
    for s in range(0, len(pos), PER_WINDOW):
        blk = pos[s:s + PER_WINDOW]
        last = int(blk[-1])
        w0 = max(OUTER[0], last + 1 - CTX)
        windows.append((s, blk, w0, last + 1))
    for b in range(0, len(windows), batch):
        group = windows[b:b + batch]
        L = max(w1 - w0 for _, _, w0, w1 in group)
        idx = torch.zeros(len(group), L, dtype=torch.long)
        for j, (_, _, w0, w1) in enumerate(group):
            idx[j, : w1 - w0] = torch.from_numpy(data[w0:w1].astype(np.int64))
        logp = port.forward(idx)[0].log_softmax(-1)
        for j, (s, blk, w0, _) in enumerate(group):
            off = blk - w0 - 1
            tgt = torch.from_numpy(data[blk].astype(np.int64))
            bits[s:s + len(blk)] = (-logp[j, off, tgt] / math.log(2)).double().numpy()
    return bits


@torch.no_grad()
def frame_checks(port, data, starts):
    idx = torch.stack([torch.from_numpy(data[s:s + CTX].astype(np.int64)) for s in starts])
    logits, writes, xf = port.forward(idx)
    summed = sum(writes.values())
    c3 = ((summed - xf).abs().max() / xf.abs().max()).item()
    mu, sd_ = xf.mean(-1, keepdim=True), xf.var(-1, unbiased=False, keepdim=True).add(1e-5).sqrt()
    g, b, W = port.sd["ln_f.weight"], port.sd["ln_f.bias"], port.sd["head.weight"]
    contrib = {n: ((w - w.mean(-1, keepdim=True)) / sd_ * g) @ W.T for n, w in writes.items()}
    recon = sum(contrib.values()) + b @ W.T
    c4 = (recon - logits).abs().max().item()
    return c3, c4, idx, logits, contrib


@torch.no_grad()
def r1(port, data, rng):
    """Linear (C4) prediction vs realised change in the true byte's log-prob when one block write is zeroed."""
    starts = rng.integers(OUTER[0], OUTER[1] - CTX - 1, size=2000)
    out = {}
    for c0 in range(0, 2000, 100):
        st = starts[c0:c0 + 100]
        _, _, idx, logits, contrib = frame_checks(port, data, st)
        tgt = torch.stack([torch.from_numpy(data[s + 1:s + CTX + 1].astype(np.int64)) for s in st])[:, -1]
        base = logits[:, -1].log_softmax(-1).gather(1, tgt[:, None])[:, 0]
        for name in [f"{p}{i}" for i in range(port.n_layer) for p in ("attn", "mlp")]:
            lin = (logits[:, -1] - contrib[name][:, -1]).log_softmax(-1).gather(1, tgt[:, None])[:, 0] - base
            real = port.forward(idx, zero=name)[0][:, -1].log_softmax(-1).gather(1, tgt[:, None])[:, 0] - base
            d = out.setdefault(name, {"lin": [], "real": []})
            d["lin"] += lin.tolist(); d["real"] += real.tolist()
    table = {}
    for name, d in out.items():
        a, r = np.array(d["lin"]), np.array(d["real"])
        table[name] = {"corr": float(np.corrcoef(a, r)[0, 1]), "mean_signed_error": float((a - r).mean()),
                       "sign_agreement": float((np.sign(a) == np.sign(r)).mean()),
                       "mean_abs_real": float(np.abs(r).mean()), "mean_abs_lin": float(np.abs(a).mean())}
    return table


def main(model_path, corpus_path, out_path, gelu="erf"):
    torch.set_num_threads(8)
    t0 = time.time()
    sd = {k: v.float() for k, v in load_file(model_path).items()}
    port = Port(sd, gelu)
    data = np.frombuffer(open(corpus_path, "rb").read(), dtype=np.uint8)
    pos = eval_positions()
    c2 = {"count": int(len(pos)), "first": int(pos[0]), "last": int(pos[-1])}
    bits1 = windowed_bpb(port, data, pos)
    bits2 = windowed_bpb(port, data, pos)
    bpb = float(bits1.mean())
    rng = np.random.default_rng(20261112)
    c3, c4, *_ = frame_checks(port, data, rng.integers(OUTER[0], OUTER[1] - CTX, size=32))
    table = r1(port, data, np.random.default_rng(20261112))
    res = {
        "prereg_sha256": sha256(__file__.replace("cal1.py", "PREREG_CAL1.md")),
        "model_sha256": sha256(model_path), "corpus_sha256": sha256(corpus_path), "gelu": gelu,
        "C1": {"bpb": bpb, "target": TARGET_BPB, "abs_diff": abs(bpb - TARGET_BPB), "pass": abs(bpb - TARGET_BPB) <= 1e-3},
        "C2": {**c2, "pass": c2 == {"count": 129429, "first": 17764705, "last": 19447269}},
        "C3": {"max_rel_err": c3, "pass": c3 <= 1e-5},
        "C4": {"max_abs_err_logits": c4, "pass": c4 <= 1e-3},
        "C5": {"identical": bool(np.array_equal(bits1, bits2)), "pass": bool(np.array_equal(bits1, bits2))},
        "R1": table,
        "env": {"torch": torch.__version__, "numpy": np.__version__, "python": platform.python_version(),
                "platform": platform.platform(), "threads": torch.get_num_threads()},
        "wall_sec": time.time() - t0,
    }
    res["PASS"] = all(res[c]["pass"] for c in ("C1", "C2", "C3", "C4", "C5"))
    json.dump(res, open(out_path, "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("C1", "C2", "C3", "C4", "C5", "PASS", "wall_sec")}, indent=1))


if __name__ == "__main__":
    a = sys.argv[1:]
    gelu = a[a.index("--gelu") + 1] if "--gelu" in a else "erf"
    main(a[0], a[1], a[2], gelu)
