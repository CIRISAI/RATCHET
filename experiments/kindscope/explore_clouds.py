"""Exploration (CAL only): (1) what organises the state cloud before a word that follows a determiner — null-adjusted
eta^2 of the state by next POS, next first byte, the determiner's form, and position bucket; (2) the byte basis: per
next-first-byte writing directions w_c (gradient of log p(first byte = c) at that position w.r.t. the mlp_L write), how
much of the ADJ writing direction they explain (R^2 of w_ADJ on span{w_c}), and what the residual does when pushed."""
import os, sys, json, collections, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import edit1 as X, edit2 as Y, mlp0_1 as M
from ks import engine as E
cache = E.Cache(os.path.join(X.SCRATCH, "ks_cache")); sd0, _ = X.load_params(cache.info["model_path"])
stories, windows, toks = X.tag_train_core(os.path.join(X.SCRATCH, "gw_data/TinyStories-valid.txt"))
rng = np.random.default_rng(20261115); order = rng.permutation(len(stories)); cal = sorted(order[: len(stories)//10].tolist())
cal_set = set(cal); prs = X.pairs(windows, toks)
# need the determiner form too: rebuild pairs with previous form
prev_form = {}
for i in range(1, len(toks)):
    prev_form[(int(toks[i][0]), int(toks[i][1]))] = str(toks[i - 1][4])
cal_prs = [p for p in prs if p[0] in cal_set and p[3] == X.DET]
sub = cal[:800]; sub_set = set(sub)
key = {}
for s, a, b, y0, y1, f in cal_prs:
    if s in sub_set: key.setdefault(s, []).append((a, b, y1, f))
first = collections.Counter(int(windows[s][a]) for s, a, b, y0, y1, f in cal_prs if s in sub_set)
TOPB = [c for c, _ in first.most_common(16)]
D = 192; out = {"top_first_bytes": [chr(c) for c in TOPB], "layers": {}}
rs = np.random.default_rng(3)
def adj_eta(W, g):
    return float(M.eta2(W, g) - np.mean([M.eta2(W, rs.permutation(g)) for _ in range(30)]))
for L in range(6):
    S, lab = [], {"next_pos": [], "next_first_byte": [], "det": [], "pos_bucket": []}
    gADJ = torch.zeros(D, device=E.DEV, dtype=torch.float64)
    gB = {c: torch.zeros(D, device=E.DEV, dtype=torch.float64) for c in TOPB}
    for ss, idx in X.batches(windows, sub, bs=32):
        add = torch.zeros(idx.shape[0], idx.shape[1], D, device=E.DEV, requires_grad=True)
        logits, writes, states, _ = Y.fwd_full(sd0, idx, {("mlp", L): add}); lp = logits.log_softmax(-1)
        rows, cols, Jadj = [], [], 0
        Jb = {c: 0 for c in TOPB}
        for r, s in enumerate(ss):
            for a, b, y1, f in key.get(s, []):
                rows.append(r); cols.append(a - 1)
                lab["next_pos"].append(int(y1)); lab["next_first_byte"].append(int(windows[s][a]))
                lab["det"].append(prev_form.get((s, a), "?")); lab["pos_bucket"].append((a - 1) // 32)
                if y1 == X.ADJ:
                    p = torch.arange(a - 1, b, device=E.DEV); Jadj = Jadj + lp[r, p, idx[r, p + 1]].sum()
                for c in TOPB:
                    Jb[c] = Jb[c] + lp[r, a - 1, c]
        if not rows: continue
        rt, ct = torch.tensor(rows, device=E.DEV), torch.tensor(cols, device=E.DEV)
        tot = Jadj
        grads = torch.autograd.grad([Jadj] + [Jb[c] for c in TOPB], add, retain_graph=True, allow_unused=True) if False else None
        # separate backward passes (same graph): ADJ target, then each byte
        g = torch.autograd.grad(Jadj, add, retain_graph=True)[0]; gADJ += g[rt, ct].double().sum(0)
        for c in TOPB:
            g = torch.autograd.grad(Jb[c], add, retain_graph=True)[0]; gB[c] += g[rt, ct].double().sum(0)
        with torch.no_grad(): S.append(states[L][rt, ct].double().cpu().numpy())
    S = np.concatenate(S)
    eta = {k: adj_eta(S, np.array(v)) for k, v in lab.items()}
    Bm = torch.stack([gB[c] / gB[c].norm() for c in TOPB], 1)            # D x 16
    w = gADJ / gADJ.norm()
    coef = torch.linalg.lstsq(Bm, w[:, None]).solution[:, 0]
    proj = Bm @ coef; resid = w - proj
    r2 = float(proj @ proj)                                              # w unit: explained share of its squared norm
    top2 = torch.linalg.lstsq(Bm[:, :2], w[:, None]).solution[:, 0]; r2_top2 = float((Bm[:, :2] @ top2).norm() ** 2)
    rnd = torch.randn(D, 16, device=E.DEV, dtype=torch.float64, generator=torch.Generator(device=E.DEV).manual_seed(L))
    r2_null = float((rnd @ torch.linalg.lstsq(rnd, w[:, None]).solution).norm() ** 2)
    gram = (Bm.T @ Bm).cpu().numpy()
    out["layers"][L] = {"eta2_adjusted": eta, "R2_byte_basis16": r2, "R2_top2_bytes": r2_top2,
                        "R2_random16_null": r2_null, "coef": {chr(c): float(v) for c, v in zip(TOPB, coef)},
                        "byte_basis_gram": gram.tolist(), "resid_dir": resid.cpu().numpy().tolist()}
    print(f"mlp{L}: eta2 {json.dumps({k: round(v, 3) for k, v in eta.items()})} | R2(w_ADJ on 16 byte dirs) {r2:.2f} "
          f"(top-2 bytes {r2_top2:.2f}; random-16 null {r2_null:.2f}) | coef " +
          " ".join(f"{chr(c)}:{float(v):+.2f}" for c, v in sorted(zip(TOPB, coef), key=lambda t: -abs(float(t[1])))[:6]),
          flush=True)
json.dump(out, open("explore_clouds.json", "w"))
