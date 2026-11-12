"""Exploration (CAL stories 1600-2399, unused so far): is the early non-byte residual 'adjective next' or 'modifier
next'? Under the mlp0-2 residual push, compare nouns after a determiner that are followed by another noun (used as a
modifier: 'the water bottle') with nouns that are not."""
import os, sys, json, collections, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import edit1 as X, edit2 as Y
from ks import engine as E
cache = E.Cache(os.path.join(X.SCRATCH, "ks_cache")); sd0, _ = X.load_params(cache.info["model_path"])
stories, windows, toks = X.tag_train_core(os.path.join(X.SCRATCH, "gw_data/TinyStories-valid.txt"))
rng = np.random.default_rng(20261115); order = rng.permutation(len(stories)); cal = sorted(order[: len(stories)//10].tolist())
sub = cal[1600:2400]; sub_set = set(sub)
# triples DET, X, next-word (adjacent, one space each)
trip = []
for i in range(1, len(toks) - 1):
    s0, a0, b0, y0, _ = toks[i - 1]; s1, a1, b1, y1, f1 = toks[i]; s2, a2, b2, y2, f2 = toks[i + 1]
    if s0 == s1 and s1 in sub_set and y0 == X.DET and a1 == b0 + 2 and windows[s1][b0 + 1] == 32:
        follow = int(y2) if (s2 == s1 and a2 == b1 + 2 and windows[s1][b1 + 1] == 32) else -1
        trip.append((int(s1), int(a1), int(b1), int(y1), str(f1), follow))
def cls(t):
    s, a, b, y, f, nxt = t
    if y == X.ADJ: return "ADJ"
    if y == X.NOUN: return "NOUN→NOUN (modifier)" if nxt == X.NOUN else "NOUN→other (head)"
    return None
trip = [t for t in trip if cls(t)]
key = {}
for s, a, b, y, f, n in trip: key.setdefault(s, []).append(a - 1)
clouds = json.load(open("explore_clouds.json")); D = 192
@torch.no_grad()
def wordlp(L, vec):
    by = {}
    for k, (s, a, b, *_ ) in enumerate(trip): by.setdefault(s, []).append((k, a, b))
    out = np.full(len(trip), np.nan)
    for ss, idx in X.batches(windows, sub, bs=64):
        add = torch.zeros(idx.shape[0], idx.shape[1], D, device=E.DEV)
        for r, s in enumerate(ss):
            for p in key.get(s, []): add[r, p] += vec
        lp = Y.fwd_full(sd0, idx, {("mlp", L): add})[0].log_softmax(-1)
        for r, s in enumerate(ss):
            for k, a, b in by.get(s, []):
                p = torch.arange(a - 1, b, device=E.DEV); out[k] = float(lp[r, p, idx[r, p + 1]].double().sum())
    return out
wn = {0: 5.50, 1: 3.33, 2: 3.91}  # mean mlp_L write norms at these positions (EDIT-2 log)
groups = np.array([cls(t) for t in trip])
out = {"counts": dict(collections.Counter(groups.tolist()))}
print(out["counts"])
for L in (0, 1, 2):
    rd = torch.tensor(clouds["layers"][str(L)]["resid_dir"], device=E.DEV).float(); rd /= rd.norm()
    d = wordlp(L, 0.5 * wn[L] * rd) - wordlp(L, torch.zeros(D, device=E.DEV))
    g = {k: (round(float(np.nanmean(d[groups == k])), 3), int((groups == k).sum())) for k in sorted(set(groups.tolist()))}
    mods = collections.defaultdict(list)
    for t, v in zip(trip, d):
        if cls(t).startswith("NOUN→NOUN"): mods[t[4]].append(v)
    out[L] = {"groups": g, "modifier_nouns": sorted(((f, round(float(np.mean(v)), 2), len(v)) for f, v in mods.items()
                                                    if len(v) >= 3), key=lambda x: -x[1])[:10]}
    print(f"mlp{L}: {g}\n   modifier nouns: {out[L]['modifier_nouns']}", flush=True)
json.dump(out, open("explore_modifier.json", "w"), indent=1)
