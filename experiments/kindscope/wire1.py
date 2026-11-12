"""WIRE-1: does mlp0's final-bytes lookup run on attn0/h3? Frozen design: PREREG_WIRE1.md.
Usage: python wire1.py <TinyStories-valid.txt> <fresh slice> <out.json>"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import json, math, sys, time  # noqa: E401,E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlp0_1 as M  # noqa: E402
import vocab1 as V  # noqa: E402
from ks import engine as E  # noqa: E402

FRESH_SHA = "3232df12d6bdd8f83071f0d8196ad10128624168e228aaf9a23f6a1f8431ee8a"
D, H = 192, 4
ARMS = ["none", "h0", "h1", "h2", "h3"]


@torch.no_grad()
def mlp0_write(port, idx, zero_head=None):
    sd = port.sd
    B, T = idx.shape
    x = sd["wte.weight"][idx] + sd["wpe.weight"][torch.arange(T, device=E.DEV)][None]
    mask = torch.ones(T, T, dtype=torch.bool, device=E.DEV).tril()
    h = port.ln(x, "blocks.0.ln1")
    q, k, v = (h @ sd["blocks.0.attn.qkv.weight"].T).split(D, dim=2)
    q, k, v = (t.view(B, T, H, D // H).transpose(1, 2) for t in (q, k, v))
    att = ((q @ k.transpose(-2, -1)) / math.sqrt(D // H)).masked_fill(~mask, float("-inf")).softmax(-1)
    o = att @ v                                    # [B, H, T, 48]
    if zero_head is not None:
        o = o.clone(); o[:, zero_head] = 0
    x = x + o.transpose(1, 2).reshape(B, T, D) @ sd["blocks.0.attn.proj.weight"].T
    hh = port.ln(x, "blocks.0.ln2")
    return F.gelu(hh @ sd["blocks.0.mlp.fc.weight"].T, approximate=port.approx) @ sd["blocks.0.mlp.proj.weight"].T


def main(valid, fresh, out_path):
    torch.use_deterministic_algorithms(True)
    t0 = time.time()
    assert V.sha256(fresh) == FRESH_SHA
    cache = E.Cache(os.path.join(os.environ.get("KS_SCRATCH", "/tmp/claude-1000/-home-emoore-RATCHET/"
                                                "4fdbd195-6bf1-45c9-8ffc-931540da4e4d/scratchpad"), "ks_cache"))
    port = cache.port()
    vset = set(p.strip() for p in open(valid, "rb").read().split(b"<|endoftext|>") if p.strip())
    parts = open(fresh, "rb").read().split(b"<|endoftext|>")[1:-1]
    kept = [p.strip() for p in parts if p.strip() and all(c < 128 for c in p.strip()) and p.strip() not in vset]
    stories = [p.decode("ascii") for p in kept[9000:12000]]
    windows, toks = V.tag(stories)
    forms = np.array([bytes(windows[t[0]][t[1]:t[2] + 1]).decode("ascii") for t in toks])
    vc = {f: c for f, c in zip(*np.unique(forms, return_counts=True))}
    keep = np.array([vc[f] >= 10 and len(f) >= 4 for f in forms])
    sel = np.where(keep)[0]
    story = np.array([t[0] for t in toks])[sel]; last = np.array([t[2] for t in toks])[sel]; fk = forms[sel]
    by = {}
    for r, s in enumerate(story):
        by.setdefault(int(s), []).append(r)
    sids = sorted(by)

    class C:
        pass
    cv = C(); cv.windows = windows
    Wr = {a: np.zeros((len(sel), D)) for a in ARMS}
    base_again = np.zeros((len(sel), D))
    for s0 in range(0, len(sids), 64):
        ss = sids[s0:s0 + 64]
        idx = M.to_idx(cv, ss)
        rows, cols, ks = [], [], []
        for r, s in enumerate(ss):
            for k in by[s]:
                rows.append(r); cols.append(int(last[k])); ks.append(k)
        rt, ct = torch.tensor(rows, device=E.DEV), torch.tensor(cols, device=E.DEV)
        _, writes = M.forward(port, idx)                      # the reference model's own mlp0 write
        base_again[ks] = writes["mlp0"][rt, ct].double().cpu().numpy()
        for a in ARMS:
            w = mlp0_write(port, idx, None if a == "none" else int(a[1]))
            Wr[a][ks] = w[rt, ct].double().cpu().numpy()
    res = {"counts": {"stories": len(stories), "sites": int(len(sel)), "forms": int(len(set(fk.tolist())))},
           "plant_null_ablation": {"bit_identical": bool(np.array_equal(Wr["none"], base_again))}}
    res["plant_null_ablation"]["pass"] = res["plant_null_ablation"]["bit_identical"]
    rng = np.random.default_rng(20261120)
    groups = {"S": np.array([f[-3:] for f in fk]), "F": np.array([f[:3] for f in fk]), "form": fk}
    null_means = {g: float(np.mean([M.eta2(Wr["none"], rng.permutation(v)) for _ in range(50)])) for g, v in groups.items()}

    def eadj(W, g, ii=None):
        v = groups[g] if ii is None else groups[g][ii]
        return M.eta2(W if ii is None else W[ii], v) - null_means[g]
    res["eta2_adj"] = {a: {g: float(eadj(Wr[a], g)) for g in groups} for a in ARMS}
    print("eta2_adj:", json.dumps(res["eta2_adj"]), f"[{time.time()-t0:.0f}s]", flush=True)
    us = np.unique(story); w1, w2 = [], []
    for _ in range(1000):
        ii = np.concatenate([np.where(story == s)[0] for s in rng.choice(us, len(us))])
        e = {a: {g: eadj(Wr[a], g, ii) for g in ("S", "F")} for a in ARMS}
        drop = {a: {g: e["none"][g] - e[a][g] for g in ("S", "F")} for a in ARMS if a != "none"}
        w1.append(drop["h3"]["S"] - max(drop[h]["S"] for h in ("h0", "h1", "h2")))
        w2.append(drop["h3"]["S"] - drop["h3"]["F"])
    E0 = res["eta2_adj"]
    drop = {a: {g: E0["none"][g] - E0[a][g] for g in ("S", "F", "form")} for a in ARMS if a != "none"}
    res["drops"] = drop
    res["W1"] = {"mean": drop["h3"]["S"] - max(drop[h]["S"] for h in ("h0", "h1", "h2")),
                 "ci95": [float(np.percentile(w1, 2.5)), float(np.percentile(w1, 97.5))]}
    res["W2"] = {"mean": drop["h3"]["S"] - drop["h3"]["F"], "ci95": [float(np.percentile(w2, 2.5)), float(np.percentile(w2, 97.5))]}
    for k in ("W1", "W2"):
        res[k]["pass"] = bool(res["plant_null_ablation"]["pass"] and res[k]["ci95"][0] > 0)
    res["H_WIRE"] = bool(res["W1"]["pass"] and res["W2"]["pass"])
    print(json.dumps({k: res[k] for k in ("plant_null_ablation", "drops", "W1", "W2", "H_WIRE")}), flush=True)
    res.update({"prereg_sha256": V.sha256(os.path.join(os.path.dirname(os.path.abspath(__file__)), "PREREG_WIRE1.md")),
                "script_sha256": V.sha256(os.path.abspath(__file__)), "env": E.device_class(), "wall_sec": time.time() - t0})
    json.dump(res, open(out_path, "w"), indent=1, default=float)


if __name__ == "__main__":
    main(*sys.argv[1:4])
