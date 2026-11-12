"""MLP0-4: MLP0-3 with donors matched on RAW bytes (KS-CASEFOLD-DONOR), FRESH-1 stories 16000-19999.
Frozen design: PREREG_MLP0_4.md.  Usage: python mlp0_3.py <TinyStories-valid.txt> <fresh slice> <out.json>"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import json, sys, time  # noqa: E401,E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlp0_1 as M  # noqa: E402
import vocab1 as V  # noqa: E402
from ks import engine as E  # noqa: E402

FRESH_SHA = "3232df12d6bdd8f83071f0d8196ad10128624168e228aaf9a23f6a1f8431ee8a"
WRITES = ["wte", "mlp0", "mlp1"]


def main(valid, fresh, out_path):
    torch.use_deterministic_algorithms(True)
    t0 = time.time()
    assert V.sha256(fresh) == FRESH_SHA
    cache = E.Cache(os.path.join(os.environ.get("KS_SCRATCH", "/tmp/claude-1000/-home-emoore-RATCHET/"
                                                "4fdbd195-6bf1-45c9-8ffc-931540da4e4d/scratchpad"), "ks_cache"))
    port = cache.port()
    vset = set(p.strip() for p in open(valid, "rb").read().split(b"<|endoftext|>") if p.strip())
    parts = open(fresh, "rb").read().split(b"<|endoftext|>")[1:-1]
    kept = [p.strip() for p in parts if p.strip() and all(c < 128 for c in p.strip()) and p.strip() not in vset]
    stories = [p.decode("ascii") for p in kept[16000:20000]]
    windows, toks = V.tag(stories)

    class C:  # minimal cache-like view for M.to_idx
        pass
    cv = C(); cv.windows = windows
    story = np.array([t[0] for t in toks]); a_ = np.array([t[1] for t in toks]); b_ = np.array([t[2] for t in toks])
    forms = np.array([bytes(windows[t[0]][t[1]:t[2] + 1]).decode("ascii") for t in toks])  # RAW bytes, no case-folding
    lens = {s: len(windows[s]) for s in range(len(windows))}
    # writes at every token's last byte (collected once; donors read from here)
    Wr = {w: np.zeros((len(toks), M.D)) for w in WRITES}
    by = {}
    for n in range(len(toks)):
        by.setdefault(int(story[n]), []).append(n)
    sids = sorted(by)
    for s0 in range(0, len(sids), 64):
        ss = sids[s0:s0 + 64]
        idx = M.to_idx(cv, ss)
        _, writes = M.forward(port, idx)
        rows, cols, ns = [], [], []
        for r, s in enumerate(ss):
            for n in by[s]:
                rows.append(r); cols.append(int(b_[n])); ns.append(n)
        for w in WRITES:
            Wr[w][ns] = writes[w][torch.tensor(rows, device=E.DEV), torch.tensor(cols, device=E.DEV)].double().cpu().numpy()
    rng = np.random.default_rng(20261119)
    occ, suf, pre = {}, {}, {}
    for n, f in enumerate(forms):
        occ.setdefault(f, []).append(n)
        if len(f) >= 4:
            suf.setdefault(f[-3:], []).append(n); pre.setdefault(f[:3], []).append(n)
    jobs = []
    for n, f in enumerate(forms):
        if len(f) < 4:
            continue
        same = [j for j in occ[f] if story[j] != story[n]]
        sd = [j for j in suf[f[-3:]] if forms[j] != f and story[j] != story[n]]
        pd = [j for j in pre[f[:3]] if forms[j] != f and story[j] != story[n]]
        if same and sd and pd:
            jobs.append((n, same[rng.integers(len(same))], sd[rng.integers(len(sd))], pd[rng.integers(len(pd))]))
    lb = lambda j: int(windows[story[j]][b_[j]])  # noqa: E731
    exact = all(lb(n) == lb(sd) == lb(sm) for n, sm, sd, pd in jobs)  # checked before anything is read
    assert exact, "plant precondition violated: a same-word or suffix donor has a different raw last byte"
    res = {"counts": {"stories": len(stories), "tokens": len(toks), "jobs": len(jobs)}, "plant_precondition_checked": exact}
    deltas = {}
    for w in WRITES:
        for ci, cond in enumerate(("self", "same", "suffix3", "prefix3")):
            deltas[(w, cond)] = np.zeros(len(jobs))
    for b0 in range(0, len(jobs), 128):
        jb = jobs[b0:b0 + 128]
        ss = [int(story[j[0]]) for j in jb]
        idx = M.to_idx(cv, ss)
        rows = list(range(len(jb))); cols = [int(b_[j[0]]) for j in jb]
        ln = [lens[s] for s in ss]
        logits0, _ = M.forward(port, idx)  # unpatched, SAME batch and layout
        base = M.nll_after(logits0, idx, rows, cols, ln)
        rt, ct = torch.tensor(rows, device=E.DEV), torch.tensor(cols, device=E.DEV)
        for w in WRITES:
            for ci, cond in enumerate(("self", "same", "suffix3", "prefix3")):
                donor = [j[0] if cond == "self" else j[ci] for j in jb]
                vecs = torch.from_numpy(Wr[w][donor]).float().to(E.DEV)
                logits, _ = M.forward(port, idx, (w, rt, ct, vecs))
                deltas[(w, cond)][b0:b0 + len(jb)] = M.nll_after(logits, idx, rows, cols, ln) - base
    js = np.array([int(story[j[0]]) for j in jobs])
    floor = max(float(np.nanmax(np.abs(np.concatenate([deltas[(w, "self")] for w in WRITES])))), 1e-6)

    def bt(v):
        ok = ~np.isnan(v)
        return {"mean": float(v[ok].mean()), "ci95": M.boot(v[ok], js[ok], rng)}

    m2 = {"floor": floor, "self_max_abs": float(np.nanmax(np.abs(np.concatenate([deltas[(w, 'self')] for w in WRITES]))))}
    for (w, c), v in deltas.items():
        m2[f"{w}/{c}"] = bt(v)
    m2["plant_suffix_within_floor"] = bool(np.nanmax(np.abs(deltas[("wte", "suffix3")])) <= floor)
    m2["plant_prefix"] = bt(deltas[("wte", "prefix3")])
    m2["plant_ok"] = m2["plant_suffix_within_floor"] and m2["plant_prefix"]["ci95"][0] > 10 * floor
    m2["P2"] = bt(deltas[("mlp0", "prefix3")] - deltas[("mlp0", "suffix3")])
    m2["P2"]["pass"] = bool(m2["plant_ok"] and m2["P2"]["ci95"][0] > 0 and abs(m2["P2"]["mean"]) > 10 * floor)
    for w in ("mlp0", "mlp1"):
        m2[f"{w}_suffix_over_same"] = float(np.nanmean(deltas[(w, "suffix3")]) / np.nanmean(deltas[(w, "same")]))
    res["M2"] = m2
    print(json.dumps({k: m2[k] for k in ("floor", "self_max_abs", "plant_ok", "P2", "mlp0_suffix_over_same",
                                          "mlp1_suffix_over_same", "mlp0/same", "mlp0/suffix3", "mlp0/prefix3")},
                     default=float), flush=True)
    res.update({"prereg_sha256": V.sha256(os.path.join(os.path.dirname(os.path.abspath(__file__)), "PREREG_MLP0_4.md")),
                "script_sha256": V.sha256(os.path.abspath(__file__)), "env": E.device_class(), "wall_sec": time.time() - t0})
    json.dump(res, open(out_path, "w"), indent=1, default=float)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
