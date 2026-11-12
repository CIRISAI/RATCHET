"""FOLD-1: compress by use, not by spread. Frozen design: PREREG_FOLD1.md.
Usage: python fold1.py <TinyStories-valid.txt> <fresh slice> <out.json>"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import json, sys, time  # noqa: E401,E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import edit1 as X  # noqa: E402
import edit2 as Y  # noqa: E402
import vocab1 as V  # noqa: E402
from ks import engine as E  # noqa: E402

FRESH_SHA = "3232df12d6bdd8f83071f0d8196ad10128624168e228aaf9a23f6a1f8431ee8a"
D, H, P0 = 192, 4, 15
SITES = [(p, i) for i in range(6) for p in ("attn", "mlp")]
RANKS_A, RANKS_B = (8, 16, 32, 64), (16, 32)


def site_states(sd, idx, writes, states):
    """State after each write (12 sites), from fwd_full's writes and post-mlp states."""
    T = idx.shape[1]
    x0 = sd["wte.weight"][idx] + sd["wpe.weight"][torch.arange(T, device=E.DEV)][None]
    out = {}
    for i in range(6):
        prev = x0 if i == 0 else states[i - 1]
        out[("attn", i)] = prev + writes[("attn", i)]
        out[("mlp", i)] = states[i]
    return out


def main(valid, fresh, out_path):
    torch.use_deterministic_algorithms(True)
    t0 = time.time()
    log = lambda m: print(f"{m}  [{time.time()-t0:.0f}s]", flush=True)  # noqa: E731
    assert V.sha256(fresh) == FRESH_SHA
    cache = E.Cache(os.path.join(X.SCRATCH, "ks_cache")); sd0, _ = X.load_params(cache.info["model_path"])
    stories, windows, toks = X.tag_train_core(valid)
    rng = np.random.default_rng(20261115); order = rng.permutation(len(stories))
    cal = sorted(order[: len(stories) // 10].tolist())[:800]
    vset = set(p.strip() for p in open(valid, "rb").read().split(b"<|endoftext|>") if p.strip())
    parts = open(fresh, "rb").read().split(b"<|endoftext|>")[1:-1]
    kept = [p.strip() for p in parts if p.strip() and all(c < 128 for c in p.strip()) and p.strip() not in vset]
    ev = [np.frombuffer((b"<|endoftext|>\n\n" + s)[:256], dtype=np.uint8).copy() for s in kept[8100:8900]]
    ev_ids = list(range(len(ev)))
    # ---------------- fit on CAL: covariance of states and writes, mean, gradient second moment per site
    S1 = {s: torch.zeros(D, device=E.DEV, dtype=torch.float64) for s in SITES}
    S2 = {s: torch.zeros(D, D, device=E.DEV, dtype=torch.float64) for s in SITES}
    W1 = {i: torch.zeros(D, device=E.DEV, dtype=torch.float64) for i in range(6)}
    W2 = {i: torch.zeros(D, D, device=E.DEV, dtype=torch.float64) for i in range(6)}
    G2 = {s: torch.zeros(D, D, device=E.DEV, dtype=torch.float64) for s in SITES}
    n = 0
    for ss, idx in X.batches(windows, cal, bs=32):
        adds = {s: torch.zeros(idx.shape[0], idx.shape[1], D, device=E.DEV, requires_grad=True) for s in SITES}
        logits, writes, states, _ = Y.fwd_full(sd0, idx, adds)
        mask = torch.zeros(idx.shape, dtype=torch.bool, device=E.DEV)
        for r, s in enumerate(ss):
            mask[r, P0:len(windows[s])] = True
        lp = logits.log_softmax(-1)
        tgt_mask = mask.clone(); tgt_mask[:, P0 - 1] = True
        for r, s in enumerate(ss):
            tgt_mask[r, len(windows[s]) - 1:] = False
        nll = -(lp[:, :-1].gather(-1, idx[:, 1:, None])[..., 0] * tgt_mask[:, :-1]).sum()
        grads = torch.autograd.grad(nll, [adds[s] for s in SITES])
        with torch.no_grad():
            st = site_states(sd0, idx, writes, states)
            m = mask
            for s, g in zip(SITES, grads):
                xs = st[s][m].double(); gs = g[m].double()
                S1[s] += xs.sum(0); S2[s] += xs.T @ xs; G2[s] += gs.T @ gs
            for i in range(6):
                w = writes[("attn", i)][m].double(); W1[i] += w.sum(0); W2[i] += w.T @ w
            n += int(m.sum())
    fit = {}
    for s in SITES:
        mu = S1[s] / n; Sig = S2[s] / n - torch.outer(mu, mu); F = G2[s] / n
        ev_, evec = torch.linalg.eigh(Sig)
        Sh = evec @ torch.diag(ev_.clamp_min(0).sqrt()) @ evec.T
        a, b = torch.linalg.eigh(Sh @ F @ Sh)
        fit[s] = {"mu": mu, "var_basis": evec.flip(1), "use_vecs": (Sh @ b.flip(1))}
    wfit = {}
    for i in range(6):
        mu = W1[i] / n; Sig = W2[i] / n - torch.outer(mu, mu); F = G2[("attn", i)] / n
        ev_, evec = torch.linalg.eigh(Sig)
        Sh = evec @ torch.diag(ev_.clamp_min(0).sqrt()) @ evec.T
        a, b = torch.linalg.eigh(Sh @ F @ Sh)
        qkv, proj = sd0[f"blocks.{i}.attn.qkv.weight"].double(), sd0[f"blocks.{i}.attn.proj.weight"].double()
        ov = torch.cat([proj[:, h * 48:(h + 1) * 48] @ qkv[2 * D + h * 48: 2 * D + (h + 1) * 48] for h in range(H)], 1)
        U, _, _ = torch.linalg.svd(ov, full_matrices=False)
        wfit[i] = {"mu": mu, "pca": evec.flip(1), "use_vecs": Sh @ b.flip(1), "svd": U}
    wpe = sd0["wpe.weight"].double(); wc = wpe - wpe.mean(0)
    clock = torch.linalg.svd(wc, full_matrices=False)[2][:2].T             # D x 2
    log(f"fitted on {n} CAL positions")

    def orth(M, r):
        if r == 0:
            return torch.zeros(D, 0, device=E.DEV, dtype=torch.float64)
        return torch.linalg.qr(M[:, :r])[0]

    # ---------------- evaluation
    @torch.no_grad()
    def per_story_nll(make_add=None):
        out = np.zeros((len(ev), 2))
        for s0 in range(0, len(ev), 64):
            ss = ev_ids[s0:s0 + 64]
            L = max(len(ev[s]) for s in ss)
            idx = torch.zeros(len(ss), L, dtype=torch.long)
            for r, s in enumerate(ss):
                idx[r, :len(ev[s])] = torch.from_numpy(ev[s].astype(np.int64))
            idx = idx.to(E.DEV)
            adds = None
            if make_add is not None:
                _, writes, states, _ = Y.fwd_full(sd0, idx)
                adds = make_add(writes, site_states(sd0, idx, writes, states))
            lp = Y.fwd_full(sd0, idx, adds)[0].log_softmax(-1)
            for r, s in enumerate(ss):
                t = torch.arange(P0 - 1, len(ev[s]) - 1, device=E.DEV)
                v = -lp[r, t, idx[r, t + 1]].double()
                out[s0 + r] = (float(v.sum()), len(t))
        return out

    base = per_story_nll()
    base_mean = base[:, 0].sum() / base[:, 1].sum()
    brng = np.random.default_rng(20261121)
    boots = [brng.integers(0, len(ev), len(ev)) for _ in range(1000)]

    def cost(arm):
        return float(arm[:, 0].sum() / arm[:, 1].sum() - base_mean)

    def diff_ci(a, b):  # (cost a) - (cost b), paired by story
        vals = [(a[ii, 0].sum() - b[ii, 0].sum()) / a[ii, 1].sum() for ii in boots]
        return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]

    def stream_arm(site, Q):
        f = fit[site]; mu = f["mu"].float(); Qf = Q.float()

        def mk(writes, st):
            x = st[site]; dlt = x - mu
            return {site: -dlt + (dlt @ Qf) @ Qf.T}
        return mk

    def write_arm(i, U=None, mean_only=False):
        mu = wfit[i]["mu"].float(); Uf = None if U is None else U.float()

        def mk(writes, st):
            w = writes[("attn", i)]
            new = mu.expand_as(w) if mean_only else mu + ((w - mu) @ Uf) @ Uf.T
            return {("attn", i): new - w}
        return mk

    res = {"base_nll": base_mean, "fit_positions": n, "eval_stories": len(ev), "a": {}, "b": {}, "c": {}, "plants": {}}
    plant_ok = True
    for s in SITES:
        key = f"{s[0]}{s[1]}"
        full = per_story_nll(stream_arm(s, orth(fit[s]["var_basis"], D)))
        zero = per_story_nll(stream_arm(s, orth(fit[s]["var_basis"], 0)))
        rec = {"full_rank_cost": cost(full), "rank0_cost": cost(zero)}
        for r in RANKS_A:
            av = per_story_nll(stream_arm(s, orth(fit[s]["var_basis"], r)))
            au = per_story_nll(stream_arm(s, orth(fit[s]["use_vecs"], r)))
            rec[r] = {"variance": cost(av), "use": cost(au), "var_minus_use_ci": diff_ci(av, au)}
            rec[r]["use_wins"] = bool(rec[r]["var_minus_use_ci"][0] > 0)
        rec["plant_full_rank_ok"] = abs(rec["full_rank_cost"]) <= 1e-4
        rec["plant_rank0_ok"] = all(rec["rank0_cost"] > max(rec[r]["variance"], rec[r]["use"]) for r in RANKS_A)
        plant_ok &= rec["plant_full_rank_ok"] and rec["plant_rank0_ok"]
        res["a"][key] = rec
        log(f"{key}: full {rec['full_rank_cost']:+.5f} zero {rec['rank0_cost']:+.3f} | " +
            " ".join(f"r{r} var {rec[r]['variance']:+.3f} use {rec[r]['use']:+.3f}{'*' if rec[r]['use_wins'] else ''}"
                     for r in RANKS_A))
    for i in range(6):
        rec = {}
        for r in RANKS_B:
            arms = {nm: per_story_nll(write_arm(i, orth(wfit[i][k], r))) for nm, k in
                    (("weight_svd", "svd"), ("data_pca", "pca"), ("use", "use_vecs"))}
            rec[r] = {nm: cost(a) for nm, a in arms.items()}
            rec[r]["pca_minus_use_ci"] = diff_ci(arms["data_pca"], arms["use"])
            rec[r]["use_wins"] = bool(rec[r]["pca_minus_use_ci"][0] > 0)
        res["b"][f"attn{i}"] = rec
        res["c"][f"attn{i}"] = cost(per_story_nll(write_arm(i, mean_only=True)))
        log(f"attn{i} write: " + " ".join(f"r{r} svd {rec[r]['weight_svd']:+.3f} pca {rec[r]['data_pca']:+.3f} "
                                          f"use {rec[r]['use']:+.3f}{'*' if rec[r]['use_wins'] else ''}" for r in RANKS_B) +
            f" | mean-only {res['c'][f'attn{i}']:+.3f}")
    cl = fit[("attn", 0)]["mu"].float(); Cf = clock.float()

    def clock_arm(writes, st):
        dlt = st[("attn", 0)] - cl
        return {("attn", 0): -(dlt @ Cf) @ Cf.T}
    res["d_clock_cost"] = cost(per_story_nll(clock_arm))
    res["d_bar_0.3"] = bool(res["d_clock_cost"] > 0.3)
    res["plants"] = {"all_ok": bool(plant_ok)}
    res["U1"] = {"cells": sum(1 for s in res["a"].values() for r in (8, 16) if s[r]["use_wins"]), "of": 24}
    res["U1"]["holds"] = bool(plant_ok and res["U1"]["cells"] == 24)
    res["U2"] = {"layers": sum(1 for v in res["b"].values() if v[16]["use_wins"]), "of": 6}
    res["U2"]["holds"] = bool(plant_ok and res["U2"]["layers"] == 6)
    log(f"clock cost {res['d_clock_cost']:+.3f}; plants {plant_ok}; U1 {res['U1']}; U2 {res['U2']}")
    res.update({"prereg_sha256": V.sha256(os.path.join(os.path.dirname(os.path.abspath(__file__)), "PREREG_FOLD1.md")),
                "script_sha256": V.sha256(os.path.abspath(__file__)), "env": E.device_class(), "wall_sec": time.time() - t0})
    json.dump(res, open(out_path, "w"), indent=1, default=float)


if __name__ == "__main__":
    main(*sys.argv[1:4])
