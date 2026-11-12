"""PARALLAX-2: category early, bytes late — on FRESH-1. Frozen design: PREREG_PARALLAX2.md.
Usage: python parallax2.py <TinyStories-valid.txt> <TinyStories-train.head20MiB.txt> <out.json>
"""
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

D = 192
FRESH_SHA = "3232df12d6bdd8f83071f0d8196ad10128624168e228aaf9a23f6a1f8431ee8a"


def main(valid, fresh, out_path):
    torch.use_deterministic_algorithms(True)
    t0 = time.time()
    assert V.sha256(fresh) == FRESH_SHA, "FRESH-1 slice hash mismatch"
    cache = E.Cache(os.path.join(X.SCRATCH, "ks_cache")); sd0, _ = X.load_params(cache.info["model_path"])
    # --- FRESH-1 stories
    vtext = open(valid, "rb").read()
    vset = set(p.strip() for p in vtext.split(b"<|endoftext|>") if p.strip())
    parts = open(fresh, "rb").read().split(b"<|endoftext|>")[1:-1]
    kept = [p.strip() for p in parts if p.strip() and all(c < 128 for c in p.strip()) and p.strip() not in vset][:4000]
    fstories = [p.decode("ascii") for p in kept]
    fwin, ftoks = V.tag(fstories)
    fprs = [p for p in X.pairs(fwin, ftoks) if p[3] == X.DET and p[4] in (X.ADJ, X.NOUN)]
    key = {}
    for s, a, b, y0, y1, f in fprs:
        key.setdefault(s, []).append(a - 1)
    sids = sorted(key)
    # --- directions (fixed from CAL)
    stories, windows, toks = X.tag_train_core(valid)
    rng0 = np.random.default_rng(20261115); order = rng0.permutation(len(stories))
    cal = sorted(order[: len(stories) // 10].tolist()); cal_set = set(cal)
    cal_prs = [p for p in X.pairs(windows, toks) if p[0] in cal_set and p[3] == X.DET]
    dirsub = cal[:800]; dir_set = set(dirsub)
    ckey = {}
    for s, a, b, y0, y1, f in cal_prs:
        if s in dir_set:
            ckey.setdefault(s, []).append((a, b, y1))

    def grad_dir(L, target):  # identical to explore_play.grad_dir
        g = torch.zeros(D, device=E.DEV, dtype=torch.float64); wn = []
        for ss, idx in X.batches(windows, dirsub, bs=32):
            add = torch.zeros(idx.shape[0], idx.shape[1], D, device=E.DEV, requires_grad=True)
            logits, writes, _, _ = Y.fwd_full(sd0, idx, {("mlp", L): add}); lp = logits.log_softmax(-1)
            J, rows, cols = 0, [], []
            for r, s in enumerate(ss):
                for a, b, y1 in ckey.get(s, []):
                    rows.append(r); cols.append(a - 1); J = J + lp[r, a - 1, target]
            if not rows:
                continue
            rt, ct = torch.tensor(rows, device=E.DEV), torch.tensor(cols, device=E.DEV)
            J.backward(); g += add.grad[rt, ct].double().sum(0)
            wn.append(writes[("mlp", L)][rt, ct].norm(dim=1).detach().double().mean().item())
        return (g / g.norm()).float(), float(np.mean(wn))

    clouds = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "explore_clouds.json")))

    @torch.no_grad()
    def wordlp(L, vec):
        by = {}
        for k, (s, a, b, _, _, _) in enumerate(fprs):
            by.setdefault(s, []).append((k, a, b))
        out = np.full(len(fprs), np.nan)
        for ss, idx in X.batches(fwin, sids, bs=64):  # identical layout for every arm
            add = torch.zeros(idx.shape[0], idx.shape[1], D, device=E.DEV)
            for r, s in enumerate(ss):
                for p in key[s]:
                    add[r, p] += vec
            lp = Y.fwd_full(sd0, idx, {("mlp", L): add})[0].log_softmax(-1)
            for r, s in enumerate(ss):
                for k, a, b in by.get(s, []):
                    p = torch.arange(a - 1, b, device=E.DEV)
                    out[k] = float(lp[r, p, idx[r, p + 1]].double().sum())
        return out

    st = np.array([p[0] for p in fprs])
    grp = np.array([("ADJ" if p[4] == X.ADJ else "NOUN") + "_" + (chr(fwin[p[0]][p[1]]) if chr(fwin[p[0]][p[1]]) in "lb"
                                                                 else "other") for p in fprs])
    rng = np.random.default_rng(20261117)
    us = np.unique(st); pos_of = np.searchsorted(us, st)

    def mean_diff(d, g1, g2):
        m1, m2 = grp == g1, grp == g2
        if m1.sum() < 30 or m2.sum() < 30:
            return {"void": f"n {int(m1.sum())}/{int(m2.sum())} < 30"}
        bs = []
        for _ in range(1000):
            wts = np.bincount(np.searchsorted(us, rng.choice(us, len(us))), minlength=len(us))[pos_of]
            bs.append((d[m1] * wts[m1]).sum() / max(wts[m1].sum(), 1) - (d[m2] * wts[m2]).sum() / max(wts[m2].sum(), 1))
        return {"mean": float(d[m1].mean() - d[m2].mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                "n": [int(m1.sum()), int(m2.sum())]}

    res = {"counts": {"fresh_stories_kept": len(fstories), "pairs": len(fprs),
                      "groups": {g: int((grp == g).sum()) for g in sorted(set(grp.tolist()))}}, "P1": {}, "floor": {}}
    for L in (0, 1, 2):
        rdir = torch.tensor(clouds["layers"][str(L)]["resid_dir"], device=E.DEV).float(); rdir = rdir / rdir.norm()
        _, wn = grad_dir(L, ord("l"))
        base = wordlp(L, torch.zeros(D, device=E.DEV)); zero = wordlp(L, 0.0 * rdir)
        res["floor"][L] = float(np.nanmax(np.abs(zero - base)))
        d = wordlp(L, 0.5 * wn * rdir) - base
        cells = {x: mean_diff(d, f"ADJ_{x}", f"NOUN_{x}") for x in ("b", "l", "other")}
        for c in cells.values():
            if "ci95" in c:
                c["pass"] = bool(c["ci95"][0] > 0)
        res["P1"][L] = {"cells": cells, "holds": bool(all(c.get("pass", False) for c in cells.values())),
                        "group_means": {g: float(d[grp == g].mean()) for g in sorted(set(grp.tolist()))}}
        print(f"L{L}: floor {res['floor'][L]:.1e} " + " ".join(f"{x}:{c.get('mean', c.get('void'))}"
              f"{c.get('ci95', '')}" for x, c in cells.items()) + f" holds={res['P1'][L]['holds']}  [{time.time()-t0:.0f}s]",
              flush=True)
    res["P1"]["holds"] = bool(all(res["P1"][L]["holds"] for L in (0, 1, 2)) and all(v <= 1e-6 for v in res["floor"].values()))
    w_l, wn5 = grad_dir(5, ord("l"))
    base = wordlp(5, torch.zeros(D, device=E.DEV)); zero = wordlp(5, 0.0 * w_l)
    res["floor"][5] = float(np.nanmax(np.abs(zero - base)))
    d = wordlp(5, 0.25 * wn5 * w_l) - base
    nl = X.boot(d[grp == "NOUN_l"], st[grp == "NOUN_l"], rng); ab = X.boot(d[grp == "ADJ_b"], st[grp == "ADJ_b"], rng)
    diff = mean_diff(d, "NOUN_l", "ADJ_other")
    res["P2"] = {"NOUN_l": nl, "ADJ_b": ab, "NOUN_l_minus_ADJ_other": diff,
                 "group_means": {g: float(d[grp == g].mean()) for g in sorted(set(grp.tolist()))},
                 "holds": bool(nl["ci95"][0] > 0 and ab["ci95"][1] < 0 and diff.get("ci95", [-1])[0] > 0
                               and res["floor"][5] <= 1e-6)}
    print(f"L5: floor {res['floor'][5]:.1e} NOUN_l {nl} ADJ_b {ab} diff {diff} P2={res['P2']['holds']}", flush=True)
    res.update({"prereg_sha256": V.sha256(os.path.join(os.path.dirname(os.path.abspath(__file__)), "PREREG_PARALLAX2.md")),
                "script_sha256": V.sha256(os.path.abspath(__file__)), "fresh_sha256": FRESH_SHA, "env": E.device_class()})
    json.dump(res, open(out_path, "w"), indent=1, default=float)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
