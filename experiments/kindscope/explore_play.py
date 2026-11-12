"""Exploration (CAL only): play with the two ends. (1) mlp5: push the pure next-byte='l' direction. (2) mlp0-2: push only
the residual of the ADJ writing direction (the part not explained by 16 first-byte forecast directions). Report the
change in word log-prob for groups (ADJ by first byte l/b/other; NOUN by first byte l/b/other) and the top movers."""
import os, sys, json, collections, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import edit1 as X, edit2 as Y
from ks import engine as E
cache = E.Cache(os.path.join(X.SCRATCH, "ks_cache")); sd0, _ = X.load_params(cache.info["model_path"])
stories, windows, toks = X.tag_train_core(os.path.join(X.SCRATCH, "gw_data/TinyStories-valid.txt"))
rng = np.random.default_rng(20261115); order = rng.permutation(len(stories)); cal = sorted(order[: len(stories)//10].tolist())
cal_set = set(cal); prs = X.pairs(windows, toks)
cal_prs = [p for p in prs if p[0] in cal_set and p[3] == X.DET]
sub = cal[800:1600]; sub_set = set(sub)          # a DIFFERENT 800 CAL stories from those the directions came from
dirsub = cal[:800]; dir_set = set(dirsub)
pp = [p for p in cal_prs if p[0] in sub_set and p[4] in (X.ADJ, X.NOUN)]
keyp = {}
for s, a, b, y0, y1, f in pp: keyp.setdefault(s, []).append(a - 1)
D = 192
clouds = json.load(open("explore_clouds.json"))
def grad_dir(L, target):
    g = torch.zeros(D, device=E.DEV, dtype=torch.float64); wn = []
    key = {}
    for s, a, b, y0, y1, f in cal_prs:
        if s in dir_set: key.setdefault(s, []).append((a, b, y1))
    for ss, idx in X.batches(windows, dirsub, bs=32):
        add = torch.zeros(idx.shape[0], idx.shape[1], D, device=E.DEV, requires_grad=True)
        logits, writes, _, _ = Y.fwd_full(sd0, idx, {("mlp", L): add}); lp = logits.log_softmax(-1)
        J, rows, cols = 0, [], []
        for r, s in enumerate(ss):
            for a, b, y1 in key.get(s, []):
                rows.append(r); cols.append(a - 1); J = J + lp[r, a - 1, target]
        if not rows: continue
        rt, ct = torch.tensor(rows, device=E.DEV), torch.tensor(cols, device=E.DEV)
        J.backward(); g += add.grad[rt, ct].double().sum(0); wn.append(writes[("mlp", L)][rt, ct].norm(dim=1).detach().double().mean().item())
    return (g / g.norm()).float(), float(np.mean(wn))
@torch.no_grad()
def wordlp(L, vec):
    by = {}
    for k, (s, a, b, _, _, _) in enumerate(pp): by.setdefault(s, []).append((k, a, b))
    out = np.full(len(pp), np.nan)
    for ss, idx in X.batches(windows, sub, bs=64):
        add = torch.zeros(idx.shape[0], idx.shape[1], D, device=E.DEV)
        for r, s in enumerate(ss):
            for p in keyp.get(s, []): add[r, p] += vec
        lp = Y.fwd_full(sd0, idx, {("mlp", L): add})[0].log_softmax(-1)
        for r, s in enumerate(ss):
            for k, a, b in by.get(s, []):
                p = torch.arange(a - 1, b, device=E.DEV); out[k] = float(lp[r, p, idx[r, p + 1]].double().sum())
    return out
def report(name, d):
    grp = collections.defaultdict(list); form = collections.defaultdict(list)
    for (s, a, b, y0, y1, f), v in zip(pp, d):
        fb = chr(windows[s][a]); key = ("ADJ" if y1 == X.ADJ else "NOUN") + "_" + (fb if fb in "lb" else "other")
        grp[key].append(v); form[f].append(v)
    g = {k: (round(float(np.mean(v)), 3), len(v)) for k, v in sorted(grp.items())}
    movers = sorted(((f, float(np.mean(v)), len(v)) for f, v in form.items() if len(v) >= 12), key=lambda t: -t[1])
    print(f"{name}: {g}\n   up:   {[(f, round(m, 2)) for f, m, n in movers[:8]]}\n   down: {[(f, round(m, 2)) for f, m, n in movers[-6:]]}", flush=True)
    return {"groups": g, "up": movers[:10], "down": movers[-10:]}
out = {}
w_l, wn5 = grad_dir(5, ord("l"))
base5 = wordlp(5, torch.zeros(D, device=E.DEV))
out["mlp5_l_quarter"] = report("mlp5 push 'l' (1/4 write norm)", wordlp(5, 0.25 * wn5 * w_l) - base5)
for L in (0, 1, 2):
    res = torch.tensor(clouds["layers"][str(L)]["resid_dir"], device=E.DEV).float(); res = res / res.norm()
    _, wnL = grad_dir(L, ord("l"))
    base = wordlp(L, torch.zeros(D, device=E.DEV))
    out[f"mlp{L}_residual_half"] = report(f"mlp{L} push RESIDUAL (1/2 write norm)", wordlp(L, 0.5 * wnL * res) - base)
json.dump(out, open("explore_play.json", "w"), indent=1)
