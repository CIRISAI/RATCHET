"""MLP0-2: is mlp0's lookup keyed on the word's final bytes? Frozen design: PREREG_MLP0_2.md. CONFIRM split only.
Usage: python mlp0_2.py <out.json>
"""

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlp0_1 as M  # noqa: E402  (forward, to_idx, nll_after, boot, eta2 — the MLP0-1 instrument, unchanged)
from ks import engine as E  # noqa: E402

SEED = 20261116
WRITES = ["wte", "mlp0", "mlp1"]


def main(out_path):
    torch.use_deterministic_algorithms(True)
    t0 = time.time()
    log = lambda m: print(f"{m}  [{time.time()-t0:.0f}s]", flush=True)  # noqa: E731
    cache = E.Cache(os.environ.get("KS_CACHE", "/tmp/claude-1000/-home-emoore-RATCHET/"
                                   "4fdbd195-6bf1-45c9-8ffc-931540da4e4d/scratchpad/ks_cache"))
    port = cache.port()
    cf = np.where(cache.split["confirm"])[0]
    pos, story, forms = cache.b, cache.story_of, cache.forms
    sids = sorted(set(story[cf].tolist()))
    lens = {s: len(cache.windows[s]) for s in sids}
    Wr = {w: np.zeros((len(cf), M.D)) for w in WRITES}
    base = np.zeros(len(cf))
    row_of = {n: k for k, n in enumerate(cf)}
    by_story = {}
    for n in cf:
        by_story.setdefault(int(story[n]), []).append(n)
    for s0 in range(0, len(sids), 64):
        ss = sids[s0:s0 + 64]
        idx = M.to_idx(cache, ss)
        logits, writes = M.forward(port, idx)
        rows, cols, ns = [], [], []
        for r, s in enumerate(ss):
            for n in by_story[s]:
                rows.append(r); cols.append(int(pos[n])); ns.append(n)
        for w in WRITES:
            v = writes[w][torch.tensor(rows, device=E.DEV), torch.tensor(cols, device=E.DEV)].double().cpu().numpy()
            Wr[w][[row_of[n] for n in ns]] = v
        base[[row_of[n] for n in ns]] = M.nll_after(logits, idx, rows, cols, [lens[ss[r]] for r in rows])
    rng = np.random.default_rng(SEED)
    fc = forms[cf]
    res = {"counts": {"confirm_tokens": int(len(cf)), "confirm_stories": len(sids)}}

    # ---- M1
    vc = {f: c for f, c in zip(*np.unique(fc, return_counts=True))}
    keep = np.array([vc[f] >= 10 and len(f) >= 4 for f in fc])
    Wk, st_k, fk = Wr["mlp0"][keep], story[cf][keep], fc[keep]
    groupings = {"form": fk}
    for k in (1, 2, 3):
        groupings[f"last{k}"] = np.array([f[-k:] for f in fk])
        groupings[f"first{k}"] = np.array([f[:k] for f in fk])

    def adj(W, g):
        return M.eta2(W, g) - np.mean([M.eta2(W, rng.permutation(g)) for _ in range(200)])

    m1 = {"n_tokens": int(keep.sum()), "n_forms": int(len(set(fk.tolist())))}
    for name, g in groupings.items():
        m1[name] = {"eta2": float(M.eta2(Wk, g)), "adjusted": float(adj(Wk, g)), "n_groups": int(len(set(g.tolist())))}
    us = np.unique(st_k)
    for k in (1, 2, 3):
        diffs = []
        for _ in range(1000):
            ii = np.concatenate([np.where(st_k == s)[0] for s in rng.choice(us, len(us))])
            # bootstrap the raw-eta2 difference; the null means are group-structure corrections fixed from the full set
            corr = (m1[f"last{k}"]["eta2"] - m1[f"last{k}"]["adjusted"]) - (m1[f"first{k}"]["eta2"] - m1[f"first{k}"]["adjusted"])
            diffs.append(M.eta2(Wk[ii], groupings[f"last{k}"][ii]) - M.eta2(Wk[ii], groupings[f"first{k}"][ii]) - corr)
        m1[f"diff_k{k}"] = {"mean": float(m1[f"last{k}"]["adjusted"] - m1[f"first{k}"]["adjusted"]),
                            "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]}
    m1["P1"] = bool(all(m1[f"diff_k{k}"]["ci95"][0] > 0 for k in (2, 3)))
    res["M1"] = m1
    log("M1: " + " ".join(f"{n}={m1[n]['adjusted']:.3f}" for n in groupings) +
        f" | diff k2 {m1['diff_k2']} k3 {m1['diff_k3']} P1={m1['P1']}")

    # ---- M2
    occ, suf, pre = {}, {}, {}
    for k, n in enumerate(cf):
        f = forms[n]
        occ.setdefault(f, []).append(k)
        if len(f) >= 4:
            suf.setdefault(f[-3:], []).append(k)
            pre.setdefault(f[:3], []).append(k)
    jobs = []
    for k, n in enumerate(cf):
        f = forms[n]
        if len(f) < 4:
            continue
        same = [j for j in occ[f] if story[cf[j]] != story[n]]
        sd = [j for j in suf[f[-3:]] if forms[cf[j]] != f and story[cf[j]] != story[n]]
        pd = [j for j in pre[f[:3]] if forms[cf[j]] != f and story[cf[j]] != story[n]]
        if same and sd and pd:
            jobs.append((k, same[rng.integers(len(same))], sd[rng.integers(len(sd))], pd[rng.integers(len(pd))]))
    res["M2"] = {"n_eligible": len(jobs)}
    deltas = {}
    for w in WRITES:
        for ci, cond in enumerate(("self", "same", "suffix3", "prefix3")):
            out = np.zeros(len(jobs))
            for b0 in range(0, len(jobs), 128):
                jb = jobs[b0:b0 + 128]
                ss = [int(story[cf[j[0]]]) for j in jb]
                idx = M.to_idx(cache, ss)
                rows = list(range(len(jb)))
                cols = [int(pos[cf[j[0]]]) for j in jb]
                donor = [j[0] if cond == "self" else j[ci] for j in jb]
                vecs = torch.from_numpy(Wr[w][donor]).float().to(E.DEV)
                logits, _ = M.forward(port, idx, (w, torch.tensor(rows, device=E.DEV), torch.tensor(cols, device=E.DEV),
                                                  vecs))
                out[b0:b0 + len(jb)] = M.nll_after(logits, idx, rows, cols, [lens[s] for s in ss]) - base[[j[0] for j in jb]]
            deltas[(w, cond)] = out
            log(f"M2 {w}/{cond}: mean Δ {np.nanmean(out):+.5f}")
    js = np.array([int(story[cf[j[0]]]) for j in jobs])
    floor = max(float(np.nanmax(np.abs(np.concatenate([deltas[(w, 'self')] for w in WRITES])))), 1e-6)
    m2 = {"floor": floor}

    def _b(v):
        ok = ~np.isnan(v)
        return {"mean": float(v[ok].mean()), "ci95": M.boot(v[ok], js[ok], rng)}

    for (w, c), v in deltas.items():
        m2[f"{w}/{c}"] = _b(v)
    m2["plant_suffix_exact_zero"] = bool(np.nanmax(np.abs(deltas[("wte", "suffix3")])) == 0.0)
    m2["plant_prefix"] = _b(deltas[("wte", "prefix3")])
    m2["plant_ok"] = m2["plant_suffix_exact_zero"] and m2["plant_prefix"]["ci95"][0] > 10 * floor
    m2["P2"] = _b(deltas[("mlp0", "prefix3")] - deltas[("mlp0", "suffix3")])
    m2["P2"]["pass"] = bool(m2["plant_ok"] and m2["P2"]["ci95"][0] > 0 and abs(m2["P2"]["mean"]) > 10 * floor)
    for w in ("mlp0", "mlp1"):
        m2[f"{w}_suffix_over_same"] = float(np.nanmean(deltas[(w, "suffix3")]) / np.nanmean(deltas[(w, "same")]))
        m2[f"{w}_prefix_over_same"] = float(np.nanmean(deltas[(w, "prefix3")]) / np.nanmean(deltas[(w, "same")]))
    res["M2"].update(m2)
    log(f"M2: plant_ok {m2['plant_ok']} P2 {m2['P2']} ratios mlp0 suf/same {m2['mlp0_suffix_over_same']:.2f} "
        f"pre/same {m2['mlp0_prefix_over_same']:.2f}; mlp1 suf/same {m2['mlp1_suffix_over_same']:.2f} "
        f"pre/same {m2['mlp1_prefix_over_same']:.2f}")
    res.update({"prereg_sha256": E.V.sha256(os.path.join(os.path.dirname(os.path.abspath(__file__)), "PREREG_MLP0_2.md")),
                "script_sha256": E.V.sha256(os.path.abspath(__file__)), "env": E.device_class(),
                "wall_sec": time.time() - t0})
    json.dump(res, open(out_path, "w"), indent=1, default=float)


if __name__ == "__main__":
    main(sys.argv[1])
