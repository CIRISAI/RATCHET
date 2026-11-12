"""The kindscope engine: feature cache, the probe verb, its plant (one level below), power check, nulls, control task and
surface baseline.

Declared rules (fixed here, before any loop stake is read; changes to them are amendments and change engine_sha):
  - Probe (binary, one-vs-rest): logistic regression, objective sum log-loss + 0.5*|w|^2 (scikit-learn's binary L2 at
    C=1), bias unpenalised; fitted by batched damped Newton to grad norm <= NEWTON_TOL, at most NEWTON_MAX iterations;
    every fit records iterations and exit reason (converged | cap | stalled). A real fit that does not converge VOIDs
    the stake (M-EXIT-DISCRIMINATOR).
  - Splits: VOCAB-1's 80/20 story split; the 20% is halved by story (seed SEED+3) into EXPLORE (proposer stakes) and
    CONFIRM (diagnosis tests only). Novel-type = test token whose form never occurs in train.
  - V1: acc > 95th pct of N_NULL label-shuffled fits. V2: acc > max of N_CTRL control-task fits (type-consistent random
    labels at the class prior). V3: 95% story-bootstrap CI of (state - surface) correctness on novel-type tokens:
    true if lo > 0, false if hi < 0, undecided otherwise; VOID if fewer than MIN_NOVEL_EACH novel positives or
    negatives. carried: true if V1 and V2 and V3; false if any of them is false; undecided otherwise.
  - Plant (M-PLANT-OBS, M-JOINT-PASS-REGION, holon `plant_not_by_construction`): a type-consistent random label r at
    the stake's class prior is injected at the token's site position one state BELOW the read (state j-1), along one
    random direction per class at PLANT_AMPLITUDE x that state's projected std, and the model is run forward to state
    j. The full pipeline must return carried = true for r. Not available at `embed` (nothing below it).
  - Power (holon `power`): the same pipeline for r on the UNINJECTED features must return carried = false. Plant true
    and null-plant false shows this cell can produce both outcomes; otherwise the stake is VOID (underpowered).
"""

import json
import math
import pathlib
import sys
import time
import zlib

import numpy as np
import torch
import torch.nn.functional as F

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import vocab1 as V  # noqa: E402  (frozen VOCAB-1 data pipeline: stories, tagging, state collection, bootstrap)
from ks.lang import POS, SITES, STATES  # noqa: E402

NEWTON_TOL, NEWTON_MAX = 1e-6, 60
# Declared tier floor (holon expansion-floor pattern): with a float32 Hessian the attainable gradient norm is limited;
# the instrument-only timing test (random features, before any loop stake) stalled at <= 1.6e-4. A fit that stops
# (stalled or cap) at or below NEWTON_FLOOR has exhausted its tier -> exit "floor", usable; above it -> VOID.
NEWTON_FLOOR = 1e-3
N_NULL, N_CTRL = 200, 20
PLANT_AMPLITUDE = 1.0
MIN_NOVEL_EACH = 10
SEED = 20261113
import os  # noqa: E402
if V.DEV.type != "cuda" and os.environ.get("KS_DEVICE") != "cpu":
    raise SystemExit("ks: no CUDA device — refusing (no silent CPU fallback; set KS_DEVICE=cpu to choose CPU as a "
                     "declared device class)")
DEV = V.DEV
DT = torch.float64


def capabilities(stake):
    caps = {"plant", "plant_joint", "power", "base_rate", "null", "control_task", "exit_reason", "device_class",
            "cite", "split"}
    if stake["verb"] == "probe" and stake["layer"] >= 1:
        caps.add("plant_not_by_construction")
    return caps if stake["verb"] == "probe" else set()


def device_class():
    return {"device": str(DEV), "gpu": torch.cuda.get_device_name(0) if DEV.type == "cuda" else None,
            "fit_dtype": str(DT), "torch": torch.__version__}


# ------------------------------------------------------------------ cache

def build_cache(model_path, corpus_path, cache_dir):
    from safetensors.torch import load_file
    from cal1 import Port
    cache = pathlib.Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    port = Port({k: v.float().to(DEV) for k, v in load_file(model_path).items()}, "tanh")
    stories = V.load_stories(corpus_path)
    windows, toks = V.tag(stories)
    feats, _, _ = V.collect(port, windows, toks)
    for s in SITES:
        np.save(cache / f"feats_{s}.npy", feats[s])
    rng = np.random.default_rng(SEED)  # identical split to VOCAB-1
    perm = rng.permutation(len(stories))
    train_s = set(perm[: int(0.8 * len(stories))].tolist())
    story_of = np.array([t[0] for t in toks])
    forms = np.array([t[4] for t in toks])
    tr = np.array([s in train_s for s in story_of])
    train_forms = set(forms[tr].tolist())
    novel = ~tr & np.array([f not in train_forms for f in forms])
    lens = np.array([len(w) for w in windows])
    np.savez(cache / "meta.npz", story_of=story_of, y=np.array([t[3] for t in toks]), forms=forms, tr=tr, novel=novel,
             a=np.array([t[1] for t in toks]), b=np.array([t[2] for t in toks]),
             windows=np.concatenate(windows), lens=lens)
    json.dump({"model_path": str(model_path), "model_sha256": V.sha256(model_path), "corpus_sha256": V.sha256(corpus_path),
               "vocab1_sha256": V.sha256(V.__file__), "tokens": len(toks), "stories": len(stories),
               **device_class()}, open(cache / "cache.json", "w"), indent=1)


class Cache:
    def __init__(self, cache_dir):
        c = pathlib.Path(cache_dir)
        self.info = json.load(open(c / "cache.json"))
        # identity check (holon M-CACHE-KIND): the cache must have been built by THIS pipeline from THESE inputs
        if self.info["vocab1_sha256"] != V.sha256(V.__file__):
            raise SystemExit("ks: cache was built by a different vocab1.py — rebuild it")
        if self.info.get("model_path") and V.sha256(self.info["model_path"]) != self.info["model_sha256"]:
            raise SystemExit("ks: model file no longer matches the cache — rebuild it")
        m = np.load(c / "meta.npz")
        self.story_of, self.y, self.forms, self.tr, self.novel = m["story_of"], m["y"], m["forms"], m["tr"], m["novel"]
        self.a, self.b = m["a"], m["b"]
        offs = np.concatenate([[0], np.cumsum(m["lens"])])
        flat = m["windows"]
        self.windows = [flat[offs[i]:offs[i + 1]] for i in range(len(m["lens"]))]
        self.toks = list(zip(self.story_of, self.a, self.b, self.y, self.forms))
        self.feats = {s: np.load(c / f"feats_{s}.npy", mmap_mode="r") for s in SITES}
        test_stories = np.unique(self.story_of[~self.tr])
        half = np.random.default_rng(SEED + 3).permutation(test_stories)[: len(test_stories) // 2]
        self.split = {"confirm": ~self.tr & np.isin(self.story_of, half),
                      "explore": ~self.tr & ~np.isin(self.story_of, half)}
        self._surface, self._port = {}, None

    def surface(self, site):
        if site not in self._surface:
            k = {"last": 4, "first": 5, "pre": 4}[site]
            self._surface[site] = V.onehot_bytes(self.windows, self.toks, site, k)
        return self._surface[site]

    def port(self):
        if self._port is None:
            from safetensors.torch import load_file
            from cal1 import Port
            self._port = Port({k: v.float().to(DEV) for k, v in load_file(self.info["model_path"]).items()}, "tanh")
        return self._port

    def positions(self, site):
        return {"last": self.b, "first": self.a, "pre": self.a - 1}[site]


# ------------------------------------------------------------------ model forward with an injection one state below

@torch.no_grad()
def planted_states(cache, site, j, vecs):
    """Run every story with vecs[n] added at token n's site position after state j-1; return state j at those
    positions [N, D] (float32 numpy)."""
    port = cache.port()
    sd = port.sd
    pos = cache.positions(site)
    by_story = {}
    for n, s in enumerate(cache.story_of):
        by_story.setdefault(int(s), []).append(n)
    out = np.zeros((len(cache.story_of), 192), dtype=np.float32)
    vt = torch.from_numpy(vecs).float().to(DEV)
    ids_all = list(range(len(cache.windows)))
    for s0 in range(0, len(ids_all), 64):
        ids = ids_all[s0:s0 + 64]
        L = max(len(cache.windows[i]) for i in ids)
        idx = torch.zeros(len(ids), L, dtype=torch.long)
        add = torch.zeros(len(ids), L, 192, device=DEV)
        rows, cols, toks = [], [], []
        for r, si in enumerate(ids):
            idx[r, :len(cache.windows[si])] = torch.from_numpy(cache.windows[si].astype(np.int64))
            for n in by_story.get(si, []):
                rows.append(r); cols.append(int(pos[n])); toks.append(n)
        if toks:
            add.index_put_((torch.tensor(rows, device=DEV), torch.tensor(cols, device=DEV)), vt[toks], accumulate=True)
        idx = idx.to(DEV)
        B, T = idx.shape
        x = sd["wte.weight"][idx] + sd["wpe.weight"][torch.arange(T, device=DEV)][None]
        states = [x]
        mask = torch.ones(T, T, dtype=torch.bool, device=DEV).tril()
        D = 192
        for i in range(port.n_layer):
            for part in ("attn", "mlp"):
                if len(states) - 1 == j - 1:
                    states[-1] = states[-1] + add
                x = states[-1]
                if part == "attn":
                    h = port.ln(x, f"blocks.{i}.ln1")
                    q, k, v = (h @ sd[f"blocks.{i}.attn.qkv.weight"].T).split(D, dim=2)
                    q, k, v = (t.view(B, T, 4, D // 4).transpose(1, 2) for t in (q, k, v))
                    att = ((q @ k.transpose(-2, -1)) / math.sqrt(D // 4)).masked_fill(~mask, float("-inf")).softmax(-1)
                    x = x + (att @ v).transpose(1, 2).reshape(B, T, D) @ sd[f"blocks.{i}.attn.proj.weight"].T
                else:
                    h = port.ln(x, f"blocks.{i}.ln2")
                    x = x + F.gelu(h @ sd[f"blocks.{i}.mlp.fc.weight"].T, approximate=port.approx) \
                        @ sd[f"blocks.{i}.mlp.proj.weight"].T
                states.append(x)
                if len(states) - 1 == j:
                    break
            if len(states) - 1 == j:
                break
        S = states[j]
        if toks:
            out[toks] = S[torch.tensor(rows, device=DEV), torch.tensor(cols, device=DEV)].cpu().numpy()
    return out


# ------------------------------------------------------------------ fitting: batched damped Newton, binary

def hessians(Xa, w, budget=1.5e8):
    """Batched Hessians sum_n w[b,n] x_n x_n^T, float32, as one GEMM per row-chunk over the upper triangle of the
    pairwise feature products (shared by every fit; cost nearly flat in the number of fits). Returns [B, d, d]."""
    d = Xa.shape[1]
    iu = torch.triu_indices(d, d, device=DEV)
    Xs, ws = Xa.float(), w.float()
    out = torch.zeros(ws.shape[0], iu.shape[1], device=DEV)
    nc = max(64, int(budget // iu.shape[1]))  # row chunk sized so the pairwise-product block stays ~0.6 GB
    for s in range(0, Xs.shape[0], nc):
        Xc = Xs[s:s + nc]
        out.addmm_(ws[:, s:s + nc], Xc[:, iu[0]] * Xc[:, iu[1]])
    H = torch.zeros(ws.shape[0], d, d, device=DEV)
    H[:, iu[0], iu[1]] = out
    H[:, iu[1], iu[0]] = out
    return H.double()


def newton(X, Y, chunk=256):
    """X [n,d] (standardised, on DEV), Y [B,n] in {0,1}. Returns theta [B,d+1], grad norms, iters, exit reasons."""
    n, d = X.shape
    Xa = torch.cat([X, torch.ones(n, 1, dtype=DT, device=DEV)], 1)
    lam = torch.ones(d + 1, dtype=DT, device=DEV); lam[-1] = 0.0
    thetas, gns, its, whys = [], [], [], []
    for s in range(0, Y.shape[0], chunk):
        y = Y[s:s + chunk].to(DEV, DT)
        B = y.shape[0]
        th = torch.zeros(B, d + 1, dtype=DT, device=DEV)

        def obj(t):
            z = t @ Xa.T
            return (F.softplus(z) - y * z).sum(1) + 0.5 * (lam * t * t).sum(1)

        f = obj(th)
        it = torch.zeros(B, dtype=torch.long)
        why = ["cap"] * B
        active = torch.ones(B, dtype=torch.bool, device=DEV)
        for k in range(NEWTON_MAX):
            z = th @ Xa.T
            p = torch.sigmoid(z)
            g = (p - y) @ Xa + lam * th
            gn = g.norm(dim=1)
            done = gn <= NEWTON_TOL
            for b in torch.nonzero(done & active).flatten().tolist():
                why[b] = "converged"
            active &= ~done
            if not active.any():
                break
            w = p * (1 - p)
            # Hessian built in float32 (64x faster on this GPU); gradient, objective and the convergence test stay
            # float64, so the fixed point and the stopping rule are unchanged (an approximate-Hessian Newton step)
            H = hessians(Xa, w) + torch.diag(lam)
            step = torch.linalg.solve(H, g.unsqueeze(-1)).squeeze(-1)
            t_ = torch.ones(B, dtype=DT, device=DEV)
            new = th - t_[:, None] * step
            fn = obj(new)
            for _ in range(30):
                bad = (fn > f - 1e-4 * t_ * (g * step).sum(1)) & active
                if not bad.any():
                    break
                t_ = torch.where(bad, t_ / 2, t_)
                new = th - t_[:, None] * step
                fn = obj(new)
            stalled = bad & active  # 30 halvings without sufficient decrease
            for b in torch.nonzero(stalled).flatten().tolist():
                why[b] = "stalled"
            active &= ~stalled
            upd = active[:, None]
            th = torch.where(upd, new, th)
            f = torch.where(active, fn, f)
            it += active.cpu().long()
        z = th @ Xa.T
        g = (torch.sigmoid(z) - y) @ Xa + lam * th
        gfin = g.norm(dim=1).cpu()
        why = ["floor" if (w in ("stalled", "cap") and float(gfin[i]) <= NEWTON_FLOOR) else w for i, w in enumerate(why)]
        thetas.append(th); gns.append(gfin); its.append(it); whys += why
    return torch.cat(thetas), torch.cat(gns), torch.cat(its), whys


def predict(X, th):
    Xa = torch.cat([X, torch.ones(X.shape[0], 1, dtype=DT, device=DEV)], 1)
    return (th @ Xa.T > 0).long().cpu().numpy()


def standardise(Xtr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    f = lambda a: torch.from_numpy((np.asarray(a, dtype=np.float64) - mu) / sd).to(DEV, DT)  # noqa: E731
    return f(Xtr), f(Xte)


def fits_summary(gn, it, why):
    return {"n": len(why), "max_iter": int(it.max()), "exits": {w: why.count(w) for w in sorted(set(why))},
            "grad_norm_max": float(gn.max())}


# ------------------------------------------------------------------ the probe pipeline

def pipeline(X, y, cache, te, rng, s_corr):
    """X [N,d] all tokens; y [N] binary labels; te = evaluation mask (explore or confirm split)."""
    tr = cache.tr
    Xtr, Xte = standardise(X[tr], X[te])
    ytr, yte = y[tr], y[te]
    nov = cache.novel[te]
    th, gn, it, why = newton(Xtr, torch.from_numpy(ytr)[None])
    corr = predict(Xte, th)[0] == yte
    acc = float(corr.mean())
    acc_nov = float(corr[nov].mean()) if nov.any() else float("nan")
    null_Y = torch.stack([torch.from_numpy(ytr[rng.permutation(len(ytr))]) for _ in range(N_NULL)])
    thn, gnn, itn, whyn = newton(Xtr, null_Y)
    null_acc = (predict(Xte, thn) == yte[None]).mean(1)
    prior = ytr.mean()
    types = sorted(set(cache.forms.tolist()))
    ctr, cte = [], []
    for _ in range(N_CTRL):
        lab = dict(zip(types, (rng.random(len(types)) < prior).astype(np.int64)))
        allc = np.array([lab[f] for f in cache.forms])
        ctr.append(allc[tr]); cte.append(allc[te])
    thc, gnc, itc, whyc = newton(Xtr, torch.from_numpy(np.stack(ctr)))
    ctrl_acc = (predict(Xte, thc) == np.stack(cte)).mean(1)
    yn = yte[nov]
    v3_void = int((yn == 1).sum()) < MIN_NOVEL_EACH or int((yn == 0).sum()) < MIN_NOVEL_EACH
    if v3_void:
        ci, v3 = [None, None], None
    else:
        diff = corr.astype(float) - s_corr.astype(float)
        ci = V.boot_ci(diff[nov], cache.story_of[te][nov], rng)
        v3 = True if ci[0] > 0 else (False if ci[1] < 0 else None)
    v1 = bool(acc > float(np.percentile(null_acc, 95)))
    v2 = bool(acc > float(ctrl_acc.max()))
    carried = True if (v1 and v2 and v3) else (False if (not v1 or not v2 or v3 is False) else None)
    return {"acc": acc, "acc_novel": acc_nov, "base_rate": float(max(yte.mean(), 1 - yte.mean())),
            "null_p95": float(np.percentile(null_acc, 95)), "ctrl_max": float(ctrl_acc.max()),
            "diff_novel": None if v3_void else float(acc_nov - s_corr[nov].mean()), "ci95": ci,
            "V1": v1, "V2": v2, "V3": v3, "V3_void": v3_void, "carried": carried,
            "n_novel": int(nov.sum()), "n_novel_pos": int((yn == 1).sum()),
            "fits": {"real": {"iters": int(it[0]), "exit": why[0], "grad_norm": float(gn[0])},
                     "null": fits_summary(gnn, itn, whyn), "ctrl": fits_summary(gnc, itc, whyc)}}


def run_probe(stake, cache):
    t0 = time.time()
    rng = np.random.default_rng(SEED + zlib.crc32(stake["id"].encode()) % 100000)
    site, j = stake["site"], stake["layer"]
    cls = stake["noun"].split(":", 1)[1]
    y = (cache.y == POS.index(cls)).astype(np.int64)
    te = cache.split[stake.get("split", "explore")]
    X = np.asarray(cache.feats[site][j], dtype=np.float64)
    tr = cache.tr
    # surface baseline on the real labels
    S = cache.surface(site)
    Str, Ste = standardise(S[tr], S[te])
    ths, _, _, why_s = newton(Str, torch.from_numpy(y[tr])[None])
    s_corr = predict(Ste, ths)[0] == y[te]
    rec = {"split": stake.get("split", "explore"), "surface": {"acc_novel": float(s_corr[cache.novel[te]].mean()),
                                                              "fit_exit": why_s[0], "features": {
        "last": "last 4 bytes of w", "first": "first byte of w + 4 before", "pre": "4 bytes before w"}[site]}}
    # plant label r: type-consistent random at the class prior
    prior = y[tr].mean()
    types = sorted(set(cache.forms.tolist()))
    lab = dict(zip(types, (np.random.default_rng(SEED + 11 + j).random(len(types)) < prior).astype(np.int64)))
    r = np.array([lab[f] for f in cache.forms])
    Sr = predict(Ste, newton(Str, torch.from_numpy(r[tr])[None])[0])[0] == r[te]
    E, _ = np.linalg.qr(np.random.default_rng(SEED + 7).standard_normal((192, 2)))
    below = np.asarray(cache.feats[site][j - 1], dtype=np.float64)
    amp = (below[tr] @ E).std(0)
    vecs = (PLANT_AMPLITUDE * amp[r] * E[:, r]).T
    Xp = planted_states(cache, site, j, vecs).astype(np.float64)
    plant = pipeline(Xp, r, cache, te, rng, Sr)
    nullp = pipeline(X, r, cache, te, rng, Sr)
    rec["plant"] = {k: plant[k] for k in ("acc", "acc_novel", "V1", "V2", "V3", "carried", "fits")}
    rec["null_plant"] = {k: nullp[k] for k in ("acc", "acc_novel", "V1", "V2", "V3", "carried", "fits")}
    rec["plant_ok"] = plant["carried"] is True
    rec["power_ok"] = nullp["carried"] is False
    if not rec["plant_ok"]:
        rec["void"] = "plant (one state below) did not return carried=true (M-PLANT-OBS, M-JOINT-PASS-REGION)"
    elif not rec["power_ok"]:
        rec["void"] = "null plant did not return carried=false: this cell cannot produce both outcomes (power)"
    else:
        real = pipeline(X, y, cache, te, rng, s_corr)
        rec.update(real)
        if real["fits"]["real"]["exit"] not in ("converged", "floor"):
            rec["void"] = f"real fit did not converge ({real['fits']['real']['exit']}) (M-EXIT-DISCRIMINATOR)"
    rec.update({"env": device_class(), "wall_sec": time.time() - t0})
    return rec
