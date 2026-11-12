"""Exploration (CAL only): what each parallax axis is 'about'. For mlp0..mlp5: per next-word form, mean z along the
probe axis and along the writing axis; and the cosine between the writing direction and a 'byte forecast' direction:
the frequency-weighted, centred unembedding rows of the first bytes of the adjectives that follow a determiner."""
import os, sys, json, math, collections, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import edit1 as X, edit2 as Y
from ks import engine as E
cache = E.Cache(os.path.join(X.SCRATCH, "ks_cache")); sd0, _ = X.load_params(cache.info["model_path"])
stories, windows, toks = X.tag_train_core(os.path.join(X.SCRATCH, "gw_data/TinyStories-valid.txt"))
rng = np.random.default_rng(20261115); order = rng.permutation(len(stories)); cal = sorted(order[: len(stories)//10].tolist())
cal_set = set(cal); prs = X.pairs(windows, toks)
cal_prs = [p for p in prs if p[0] in cal_set and p[3] == X.DET]
sub = cal[:800]; sub_set = set(sub)
key = {}
for s, a, b, y0, y1, f in cal_prs:
    if s in sub_set: key.setdefault(s, []).append((a, b, y1, f))
adj_first = collections.Counter(windows[s][a] for s, a, b, y0, y1, f in cal_prs if s in sub_set and y1 == X.ADJ)
W = sd0["wte.weight"].double(); Wc = W - W.mean(0)
D = 192; out = {"adj_first_bytes": {chr(k): v for k, v in adj_first.most_common(12)}, "layers": {}}
def cos(a, b): return float(a @ b / (a.norm() * b.norm()))
for L in range(6):
    g = torch.zeros(D, device=E.DEV, dtype=torch.float64); Ss, ys, fs, fb = [], [], [], []
    for ss, idx in X.batches(windows, sub, bs=32):
        add = torch.zeros(idx.shape[0], idx.shape[1], D, device=E.DEV, requires_grad=True)
        logits, writes, states, _ = Y.fwd_full(sd0, idx, {("mlp", L): add}); lp = logits.log_softmax(-1)
        J, rows, cols = 0, [], []
        for r, s in enumerate(ss):
            for a, b, y1, f in key.get(s, []):
                rows.append(r); cols.append(a - 1); fs.append(f); ys.append(int(y1 == X.ADJ)); fb.append(int(windows[s][a]))
                if y1 == X.ADJ:
                    p = torch.arange(a - 1, b, device=E.DEV); J = J + lp[r, p, idx[r, p + 1]].sum()
        if not rows: continue
        rt, ct = torch.tensor(rows, device=E.DEV), torch.tensor(cols, device=E.DEV)
        if torch.is_tensor(J): J.backward(); g += add.grad[rt, ct].double().sum(0)
        with torch.no_grad(): Ss.append(states[L][rt, ct].double())
    S = torch.cat(Ss); y = torch.tensor(ys, device=E.DEV); mu = S.mean(0); Sc = S - mu
    sd = Sc.std(0).clamp_min(1e-12)
    th, _, _, _ = E.newton((Sc / sd).to(E.DT), y.cpu()[None]); r = th[0, :-1].to(E.DEV) / sd
    zr = (Sc @ r); zr = (zr - zr.mean()) / zr.std()
    zw = (Sc @ g); zw = (zw - zw.mean()) / zw.std()
    # byte-forecast direction: frequency-weighted centred unembedding of adjective first bytes (scaled by ln_f gain)
    fdir = sum(c * Wc[b] for b, c in adj_first.items()) * sd0["ln_f.weight"].double()
    # per-form means (forms with >= 15 occurrences)
    cnt = collections.Counter(fs); rows_ = []
    for f, c in cnt.items():
        if c >= 15:
            m = torch.tensor([x == f for x in fs], device=E.DEV)
            rows_.append({"form": f, "n": c, "adj": bool(y[m].float().mean() > 0.5), "z_probe": float(zr[m].mean()),
                          "z_write": float(zw[m].mean())})
    rows_.sort(key=lambda d: -d["z_write"])
    out["layers"][L] = {"cos_write_byteforecast": cos(g, fdir), "cos_probe_byteforecast": cos(r, fdir),
                        "corr_probe_write_coords": float(torch.corrcoef(torch.stack([zr, zw]))[0, 1]),
                        "words": rows_}
    print(f"mlp{L}: cos(write, byte-forecast) {cos(g, fdir):+.3f}  cos(probe, byte-forecast) {cos(r, fdir):+.3f}  "
          f"corr(z_probe, z_write) {out['layers'][L]['corr_probe_write_coords']:+.2f}")
    print("   top of writing axis:", [(d['form'], round(d['z_write'], 2), 'A' if d['adj'] else '-') for d in rows_[:8]])
    print("   top of probe axis:  ", [(d['form'], round(d['z_probe'], 2), 'A' if d['adj'] else '-')
                                       for d in sorted(rows_, key=lambda d: -d['z_probe'])[:8]])
print("adjective first bytes:", out["adj_first_bytes"])
json.dump(out, open("explore_parallax3.json", "w"), indent=1)
