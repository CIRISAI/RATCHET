"""Exploration (CAL only): decompose the parallax. Per MLP layer at positions before a word that follows a determiner:
probe (decoder) r, pattern a = Sigma r (Haufe et al. 2014), class mean difference dmu, and the pulled-back writing
direction w (gradient of the ADJ-word target through the full network). Angles, push effects, and the 2-D plane."""
import os, sys, json, math, numpy as np, torch
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
    if s in sub_set: key.setdefault(s, []).append((a, b, y1))
adj = [p for p in cal_prs if p[0] in sub_set and p[4] == X.ADJ]
base = float(np.nanmean(X.evaluate(sd0, windows, sub, adj)[0]))
D = 192; out = {"base": base, "layers": {}}
def ang(a, b): return math.degrees(math.acos(max(-1, min(1, float(a @ b / (a.norm() * b.norm()))))))
for L in range(6):
    g = torch.zeros(D, device=E.DEV, dtype=torch.float64); Ss, ys, wn = [], [], []
    for ss, idx in X.batches(windows, sub, bs=32):
        add = torch.zeros(idx.shape[0], idx.shape[1], D, device=E.DEV, requires_grad=True)
        logits, writes, states, _ = Y.fwd_full(sd0, idx, {("mlp", L): add}); lp = logits.log_softmax(-1)
        J, rows, cols = 0, [], []
        for r, s in enumerate(ss):
            for a, b, y1 in key.get(s, []):
                rows.append(r); cols.append(a - 1)
                if y1 == X.ADJ:
                    p = torch.arange(a - 1, b, device=E.DEV); J = J + lp[r, p, idx[r, p + 1]].sum()
        if not rows: continue
        rt, ct = torch.tensor(rows, device=E.DEV), torch.tensor(cols, device=E.DEV)
        if torch.is_tensor(J): J.backward(); g += add.grad[rt, ct].double().sum(0)
        with torch.no_grad():
            Ss.append(states[L][rt, ct].double()); wn.append(writes[("mlp", L)][rt, ct].norm(dim=1).double())
            ys += [int(y1 == X.ADJ) for r, s in enumerate(ss) for a, b, y1 in key.get(s, [])]
    S = torch.cat(Ss); y = torch.tensor(ys, device=E.DEV); wn = float(torch.cat(wn).mean())
    mu = S.mean(0); Sc = S - mu; Sig = Sc.T @ Sc / len(S)
    sd = Sc.std(0).clamp_min(1e-12)
    th, _, _, why = E.newton(((S - mu) / sd).to(E.DT), y.cpu()[None])
    r = th[0, :-1].to(E.DEV) / sd                     # decoder, raw space
    a = Sig @ r                                        # Haufe pattern
    dmu = S[y == 1].mean(0) - S[y == 0].mean(0)
    w = g
    # top principal direction of the states (where the variance is)
    U, Sv, Vt = torch.linalg.svd(Sc, full_matrices=False); pc1 = Vt[0]
    rec = {"angles": {"probe-write": ang(r, w), "pattern-write": ang(a, w), "dmu-write": ang(dmu, w),
                      "probe-pattern": ang(r, a), "pattern-dmu": ang(a, dmu), "write-pc1": min(ang(w, pc1), 180 - ang(w, pc1)),
                      "probe-pc1": min(ang(r, pc1), 180 - ang(r, pc1))},
           "var_frac_along": {n: float(((Sc @ (v / v.norm())) ** 2).mean() / (Sc ** 2).sum(1).mean())
                              for n, v in (("probe", r), ("pattern", a), ("dmu", dmu), ("write", w))},
           "probe_exit": why[0], "write_norm": wn}
    curves = {}
    for n, v in (("probe", r), ("pattern", a), ("dmu", dmu), ("write", w)):
        d = (v / v.norm()).float(); cur = []
        for sz in (0.25, 0.5, 1.0):
            tot, cnt = 0.0, 0
            with torch.no_grad():
                for ss, idx in X.batches(windows, sub, bs=64):
                    add = torch.zeros(idx.shape[0], idx.shape[1], D, device=E.DEV)
                    for rr, s in enumerate(ss):
                        for aa, bb, y1 in key.get(s, []): add[rr, aa - 1] += sz * wn * d
                    lp = Y.fwd_full(sd0, idx, {("mlp", L): add})[0].log_softmax(-1)
                    for rr, s in enumerate(ss):
                        for aa, bb, y1 in key.get(s, []):
                            if y1 == X.ADJ:
                                p = torch.arange(aa - 1, bb, device=E.DEV); tot += float(lp[rr, p, idx[rr, p + 1]].sum()); cnt += 1
            cur.append([sz, tot / cnt - base])
        curves[n] = cur
    rec["curves"] = curves
    # 2-D plane spanned by the probe and the writing direction (orthonormalised), states centred
    e1 = r / r.norm(); e2 = w - (w @ e1) * e1; e2 = e2 / e2.norm()
    P = torch.stack([Sc @ e1, Sc @ e2], 1).cpu().numpy()
    idx_s = np.random.default_rng(L).choice(len(P), size=min(1500, len(P)), replace=False)
    rec["plane"] = {"xy": P[idx_s].tolist(), "y": y.cpu().numpy()[idx_s].tolist(),
                    "arrows": {n: [float((v / v.norm()) @ e1), float((v / v.norm()) @ e2)]
                               for n, v in (("probe", r), ("pattern", a), ("dmu", dmu), ("write", w))}}
    out["layers"][L] = rec
    print(L, json.dumps(rec["angles"]), {k: round(v, 3) for k, v in rec["var_frac_along"].items()},
          {k: max(c[1] for c in v) for k, v in curves.items()}, flush=True)
json.dump(out, open("explore_parallax2.json", "w"))
