"""Attention 'wiring' for the 8 trace items: for each attention layer and each position, the top-5 source positions by
attention weight averaged over the 4 heads, plus the per-head top-1. Same items and windows as make_traces.py.
Usage: python make_attn.py <TinyStories-valid.txt> <fresh slice> <traces dir>"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import json, math, sys  # noqa: E401,E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import edit1 as X  # noqa: E402
import vocab1 as V  # noqa: E402
from ks import engine as E  # noqa: E402

D, H = 192, 4


@torch.no_grad()
def attention(port, idx):
    sd = port.sd
    B, T = idx.shape
    x = sd["wte.weight"][idx] + sd["wpe.weight"][torch.arange(T, device=E.DEV)][None]
    mask = torch.ones(T, T, dtype=torch.bool, device=E.DEV).tril()
    out = []
    for i in range(port.n_layer):
        h = port.ln(x, f"blocks.{i}.ln1")
        q, k, v = (h @ sd[f"blocks.{i}.attn.qkv.weight"].T).split(D, dim=2)
        q, k, v = (t.view(B, T, H, D // H).transpose(1, 2) for t in (q, k, v))
        att = ((q @ k.transpose(-2, -1)) / math.sqrt(D // H)).masked_fill(~mask, float("-inf")).softmax(-1)
        out.append(att[0].cpu().numpy())                       # [H, T, T]
        x = x + (att @ v).transpose(1, 2).reshape(B, T, D) @ sd[f"blocks.{i}.attn.proj.weight"].T
        hh = port.ln(x, f"blocks.{i}.ln2")
        x = x + torch.nn.functional.gelu(hh @ sd[f"blocks.{i}.mlp.fc.weight"].T, approximate=port.approx) \
            @ sd[f"blocks.{i}.mlp.proj.weight"].T
    return out


def main(valid, fresh, tdir):
    cache = E.Cache(os.path.join(X.SCRATCH, "ks_cache")); port = cache.port()
    vset = set(p.strip() for p in open(valid, "rb").read().split(b"<|endoftext|>") if p.strip())
    parts = open(fresh, "rb").read().split(b"<|endoftext|>")[1:-1]
    kept = [p.strip() for p in parts if p.strip() and all(c < 128 for c in p.strip()) and p.strip() not in vset]
    for k, story in enumerate(kept[20000:20008]):
        w = (b"<|endoftext|>\n\n" + story)[:256]
        side = json.load(open(os.path.join(tdir, f"gw_item{k}.readings.json")))
        assert side["bytes"] == list(w), "item bytes differ from the trace sidecar"
        att = attention(port, torch.tensor(list(w), dtype=torch.long, device=E.DEV)[None])
        layers = {}
        for L, A in enumerate(att):
            m = A.mean(0)
            top = [[[int(j), round(float(m[p, j]), 4)] for j in np.argsort(-m[p])[:5] if m[p, j] > 0] for p in range(len(w))]
            head1 = [[[int(A[h, p].argmax()), round(float(A[h, p].max()), 4)] for h in range(H)] for p in range(len(w))]
            layers[f"attn{L}"] = {"top5_mean_over_heads": top, "per_head_top1": head1}
        json.dump({"item": k, "readme": "per attention layer and position p: top-5 source positions j <= p by attention "
                   "weight averaged over the 4 heads, as [j, weight]; per_head_top1: each head's single strongest "
                   "source. Attention weight says where a stage READS FROM, not how much that matters downstream.",
                   "layers": layers}, open(os.path.join(tdir, f"gw_item{k}.attention.json"), "w"))
        print(k, "ok", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
