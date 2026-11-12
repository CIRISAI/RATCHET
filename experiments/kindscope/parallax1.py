"""PARALLAX-1: does the model write at byte grain? Frozen design: PREREG_PARALLAX1.md. CONFIRM split (second read).
Usage: python parallax1.py <TinyStories-valid.txt> <out.json>
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import json, sys, time  # noqa: E401,E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import edit1 as X  # noqa: E402
import edit2 as Y  # noqa: E402
from ks import engine as E  # noqa: E402

LAYERS = (3, 4)
B = (ord("l"), ord("b"), ord("s"))
D = 192


def main(corpus, out_path):
    torch.use_deterministic_algorithms(True)
    t0 = time.time()
    cache = E.Cache(os.path.join(X.SCRATCH, "ks_cache")); sd0, _ = X.load_params(cache.info["model_path"])
    stories, windows, toks = X.tag_train_core(corpus)
    rng = np.random.default_rng(20261115); order = rng.permutation(len(stories))
    cal = sorted(order[: len(stories) // 10].tolist()); cal_set = set(cal)
    cal_prs = [p for p in X.pairs(windows, toks) if p[0] in cal_set and p[3] == X.DET]
    key = {}
    for s, a, b, y0, y1, f in cal_prs:
        key.setdefault(s, []).append((a, b, y1))
    cf_stories = sorted(set(cache.story_of[cache.split["confirm"]].tolist())); cf_set = set(cf_stories)
    ex_toks = [(int(s), int(a), int(b), int(y), f) for s, a, b, y, f in cache.toks]
    cf_prs = [p for p in X.pairs(cache.windows, ex_toks) if p[0] in cf_set and p[3] == X.DET]
    groups = {"ADJ": [p for p in cf_prs if p[4] == X.ADJ],
              "NOUN_in_B": [p for p in cf_prs if p[4] == X.NOUN and cache.windows[p[0]][p[1]] in B],
              "NOUN_out": [p for p in cf_prs if p[4] == X.NOUN and cache.windows[p[0]][p[1]] not in B]}
    allp = groups["ADJ"] + groups["NOUN_in_B"] + groups["NOUN_out"]
    res = {"counts": {k: len(v) for k, v in groups.items()}, "B": [chr(b) for b in B], "layers": {}}
    keycf = {}
    for s, a, b, y0, y1, f in cf_prs:
        keycf.setdefault(s, []).append(a - 1)

    @torch.no_grad()
    def wordlp(L, vec):
        """Word log-probs for allp with vec added to the mlp_L write at every CONFIRM key position (vec may be 0)."""
        by = {}
        for k, (s, a, b, _, _, _) in enumerate(allp):
            by.setdefault(s, []).append((k, a, b))
        out = np.full(len(allp), np.nan)
        for ss, idx in X.batches(cache.windows, cf_stories, bs=64):  # identical layout for every arm
            add = torch.zeros(idx.shape[0], idx.shape[1], D, device=E.DEV)
            for r, s in enumerate(ss):
                for p in keycf.get(s, []):
                    add[r, p] += vec
            lp = Y.fwd_full(sd0, idx, {("mlp", L): add})[0].log_softmax(-1)
            for r, s in enumerate(ss):
                for k, a, b in by.get(s, []):
                    p = torch.arange(a - 1, b, device=E.DEV)
                    out[k] = float(lp[r, p, idx[r, p + 1]].double().sum())
        return out

    sts = np.array([p[0] for p in allp])
    sl = {"ADJ": slice(0, len(groups["ADJ"])),
          "NOUN_in_B": slice(len(groups["ADJ"]), len(groups["ADJ"]) + len(groups["NOUN_in_B"])),
          "NOUN_out": slice(len(groups["ADJ"]) + len(groups["NOUN_in_B"]), len(allp))}
    for L in LAYERS:
        g = torch.zeros(D, device=E.DEV, dtype=torch.float64); wn = []
        for ss, idx in X.batches(windows, cal, bs=32):
            add = torch.zeros(idx.shape[0], idx.shape[1], D, device=E.DEV, requires_grad=True)
            logits, writes, _, _ = Y.fwd_full(sd0, idx, {("mlp", L): add}); lp = logits.log_softmax(-1)
            J, rows, cols = 0, [], []
            for r, s in enumerate(ss):
                for a, b, y1 in key.get(s, []):
                    rows.append(r); cols.append(a - 1)
                    if y1 == X.ADJ:
                        p = torch.arange(a - 1, b, device=E.DEV); J = J + lp[r, p, idx[r, p + 1]].sum()
            if not rows:
                continue
            rt, ct = torch.tensor(rows, device=E.DEV), torch.tensor(cols, device=E.DEV)
            if torch.is_tensor(J):
                J.backward(); g += add.grad[rt, ct].double().sum(0)
            wn.append(writes[("mlp", L)][rt, ct].norm(dim=1).detach().double().cpu().numpy())
        size = 0.5 * float(np.concatenate(wn).mean())
        w = (g / g.norm()).float()
        base = wordlp(L, torch.zeros(D, device=E.DEV))
        zero = wordlp(L, 0.0 * w)
        pushed = wordlp(L, size * w)
        floor = float(np.nanmax(np.abs(zero - base)))
        d = pushed - base
        rec = {"size": size, "floor": floor, "plant_self_ok": floor <= 1e-6}
        for k, s_ in sl.items():
            rec[k] = X.boot(d[s_], sts[s_], rng)
        # P2: difference of means, stories resampled jointly
        us = np.unique(sts); diffs = []
        for _ in range(1000):
            pick = rng.choice(us, len(us)); w_ = np.bincount(np.searchsorted(us, pick), minlength=len(us))
            wt = w_[np.searchsorted(us, sts)]
            m_in = np.nansum(d[sl["NOUN_in_B"]] * wt[sl["NOUN_in_B"]]) / max(wt[sl["NOUN_in_B"]].sum(), 1)
            m_out = np.nansum(d[sl["NOUN_out"]] * wt[sl["NOUN_out"]]) / max(wt[sl["NOUN_out"]].sum(), 1)
            diffs.append(m_in - m_out)
        rec["P2"] = {"mean": float(np.nanmean(d[sl["NOUN_in_B"]]) - np.nanmean(d[sl["NOUN_out"]])),
                     "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]}
        rec["P_TARGET"] = bool(rec["ADJ"]["ci95"][0] > 0)
        rec["P1"] = bool(rec["P_TARGET"] and rec["plant_self_ok"] and rec["NOUN_in_B"]["ci95"][0] > 0)
        rec["P2"]["pass"] = bool(rec["P_TARGET"] and rec["plant_self_ok"] and rec["P2"]["ci95"][0] > 0)
        byb = {}
        for k in range(sl["NOUN_in_B"].start, len(allp)):
            byb.setdefault(chr(cache.windows[allp[k][0]][allp[k][1]]), []).append(d[k])
        rec["noun_by_first_byte"] = {c: {"n": len(v), "mean": float(np.nanmean(v))} for c, v in byb.items() if len(v) >= 30}
        res["layers"][L] = rec
        print(f"L{L}: floor {floor:.1e} ADJ {rec['ADJ']['mean']:+.3f} {rec['ADJ']['ci95']} | NOUN_in_B "
              f"{rec['NOUN_in_B']['mean']:+.3f} {rec['NOUN_in_B']['ci95']} | NOUN_out {rec['NOUN_out']['mean']:+.3f} "
              f"{rec['NOUN_out']['ci95']} | P2 {rec['P2']} | P1 {rec['P1']}  [{time.time()-t0:.0f}s]", flush=True)
    res.update({"prereg_sha256": X.V.sha256(os.path.join(os.path.dirname(os.path.abspath(__file__)), "PREREG_PARALLAX1.md")),
                "script_sha256": X.V.sha256(os.path.abspath(__file__)), "env": E.device_class()})
    json.dump(res, open(out_path, "w"), indent=1, default=float)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
