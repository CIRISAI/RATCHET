"""PARALLAX-H: Holon's removability gate on the parallax dictionary, plus realised removal costs.
Frozen design: PREREG_PARALLAXH.md.  Usage: python parallaxh.py <TinyStories-valid.txt> <admit_file binary> <out.json>
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import collections, hashlib, json, subprocess, sys, time  # noqa: E401,E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import edit1 as X  # noqa: E402
import edit2 as Y  # noqa: E402
from ks import engine as E  # noqa: E402

D = 192


def main(valid, gate_bin, out_path):
    torch.use_deterministic_algorithms(True)
    t0 = time.time()
    log = lambda m: print(f"{m}  [{time.time()-t0:.0f}s]", flush=True)  # noqa: E731
    cache = E.Cache(os.path.join(X.SCRATCH, "ks_cache")); sd0, _ = X.load_params(cache.info["model_path"])
    stories, windows, toks = X.tag_train_core(valid)
    rng = np.random.default_rng(20261115); order = rng.permutation(len(stories))
    cal = sorted(order[: len(stories) // 10].tolist())
    dir_set, gate_stories = set(cal[:800]), cal[2000:2400]
    gate_set = set(gate_stories)
    prev_form = {(int(toks[i][0]), int(toks[i][1])): str(toks[i - 1][4]) for i in range(1, len(toks))}
    prs = [p for p in X.pairs(windows, toks) if p[3] == X.DET]
    dkey = collections.defaultdict(list); gkey = collections.defaultdict(list)
    for s, a, b, y0, y1, f in prs:
        if s in dir_set:
            dkey[s].append((a, b, y1))
        if s in gate_set:
            gkey[s].append((a, b, y1))
    first = collections.Counter(int(windows[s][a]) for s in dkey for a, b, y1 in dkey[s])
    TOPB = [c for c, _ in first.most_common(16)]
    clouds = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "explore_clouds.json")))
    assert [chr(c) for c in TOPB] == clouds["top_first_bytes"], "byte basis differs from the exploration's"
    dets = [f for f, _ in collections.Counter(prev_form.get((s, a), "?") for s in gkey for a, b, y in gkey[s]).most_common(5)]
    res = {"top_first_bytes": [chr(c) for c in TOPB], "determiners": dets, "layers": {}}
    # ---- gate rows: base forward on gate stories, states after every mlp_L and target log-probs
    rows_meta, states, target = [], {L: [] for L in range(6)}, []
    with torch.no_grad():
        for ss, idx in X.batches(windows, gate_stories, bs=64):
            logits, writes, st, _ = Y.fwd_full(sd0, idx)
            lp = logits.log_softmax(-1)
            for r, s in enumerate(ss):
                for a, b, y1 in gkey.get(s, []):
                    rows_meta.append((s, a - 1, prev_form.get((s, a), "?"), int(windows[s][a])))
                    for L in range(6):
                        states[L].append(st[L][r, a - 1].double().cpu().numpy())
                    target.append(lp[r, a - 1].double().cpu().numpy())
    S = {L: np.array(v) for L, v in states.items()}; T = np.array(target)
    kept = np.array([[1.0, p / 256] + [1.0 * (d == q) for q in dets] + [1.0 * (d not in dets)] for _, p, d, _ in rows_meta])
    unit = np.array([s % 5 for s, _, _, _ in rows_meta])
    log(f"gate rows {len(rows_meta)}; units {np.bincount(unit).tolist()}")
    for L in range(6):
        # ---- directions from CAL 0-799 (as explore_clouds / explore_play)
        gA = torch.zeros(D, device=E.DEV, dtype=torch.float64)
        gB = {c: torch.zeros(D, device=E.DEV, dtype=torch.float64) for c in TOPB}
        Xs, ys = [], []
        for ss, idx in X.batches(windows, cal[:800], bs=32):
            add = torch.zeros(idx.shape[0], idx.shape[1], D, device=E.DEV, requires_grad=True)
            logits, _, st, _ = Y.fwd_full(sd0, idx, {("mlp", L): add}); lp = logits.log_softmax(-1)
            rows, cols, Ja = [], [], 0
            Jb = {c: 0 for c in TOPB}
            for r, s in enumerate(ss):
                for a, b, y1 in dkey.get(s, []):
                    rows.append(r); cols.append(a - 1); ys.append(int(y1 == X.ADJ))
                    if y1 == X.ADJ:
                        p = torch.arange(a - 1, b, device=E.DEV); Ja = Ja + lp[r, p, idx[r, p + 1]].sum()
                    for c in TOPB:
                        Jb[c] = Jb[c] + lp[r, a - 1, c]
            if not rows:
                continue
            rt, ct = torch.tensor(rows, device=E.DEV), torch.tensor(cols, device=E.DEV)
            gA += torch.autograd.grad(Ja, add, retain_graph=True)[0][rt, ct].double().sum(0)
            for c in TOPB:
                gB[c] += torch.autograd.grad(Jb[c], add, retain_graph=True)[0][rt, ct].double().sum(0)
            Xs.append(st[L][rt, ct].detach().double().cpu().numpy())
        Xa = np.concatenate(Xs); yv = np.array(ys)
        mu, sd = Xa.mean(0), np.where(Xa.std(0) > 0, Xa.std(0), 1)
        Z = torch.from_numpy((Xa - mu) / sd).to(E.DEV, E.DT)
        th, _, _, why = E.newton(Z, torch.from_numpy(yv)[None])
        dirs = {"probe": [th[0, :-1].cpu().numpy() / sd]}
        for k in (1, 2, 3):
            ths, _, _, _ = E.newton(Z, torch.from_numpy(np.random.default_rng(k).permutation(yv))[None])
            dirs[f"shuf{k}"] = [ths[0, :-1].cpu().numpy() / sd]
        dirs["write"] = [gA.cpu().numpy()]
        dirs["resid"] = [np.array(clouds["layers"][str(L)]["resid_dir"])]
        dirs["bytes16"] = [gB[c].cpu().numpy() for c in TOPB]
        dirs = {k: [v / np.linalg.norm(v) for v in vs] for k, vs in dirs.items()}
        # ---- SPEC
        lines = ["budget 0.02", "folds units 5", "lag 0", "horizon future", "ridge order1", "null permute 7", "units 5"]
        fmt = lambda M: f"{M.shape[0]} {M.shape[1]} " + " ".join(f"{v:.7g}" for v in M.ravel())  # noqa: E731
        for u in range(5):
            m = unit == u
            lines.append(f"unit kept {fmt(kept[m])} target {fmt(T[m])}")
        for name, vs in dirs.items():
            Q = np.stack(vs, 1)
            lines.append(f"candidate {name} " + " ".join(f"block {fmt(S[L][unit == u] @ Q)}" for u in range(5)))
        spec = "\n".join(lines) + "\n"
        spath = os.path.join(X.SCRATCH, f"parallaxh_L{L}.spec")
        open(spath, "w").write(spec)
        out = subprocess.run([gate_bin, spath], capture_output=True, text=True)
        gate_json = [json.loads(l) for l in out.stdout.splitlines() if l.startswith("{")]
        # ---- realisation: project each candidate's directions out of the state after mlp_L at the gate positions
        mu_g = torch.from_numpy(S[L].mean(0)).to(E.DEV).float()
        bypos = collections.defaultdict(list)
        for k, (s, p, _, nb) in enumerate(rows_meta):
            bypos[s].append((k, p, nb))
        realised = {}
        for name, vs in [("none", None)] + list(dirs.items()):
            nll = np.zeros(len(rows_meta))
            Q = None if vs is None else torch.from_numpy(np.linalg.qr(np.stack(vs, 1))[0]).to(E.DEV).float()
            with torch.no_grad():
                for ss, idx in X.batches(windows, gate_stories, bs=64):
                    add = torch.zeros(idx.shape[0], idx.shape[1], D, device=E.DEV)
                    if Q is not None:
                        _, _, st, _ = Y.fwd_full(sd0, idx)
                        for r, s in enumerate(ss):
                            for k, p, nb in bypos.get(s, []):
                                x = st[L][r, p]
                                add[r, p] = -((x - mu_g) @ Q) @ Q.T
                    lp = Y.fwd_full(sd0, idx, {("mlp", L): add})[0].log_softmax(-1)
                    for r, s in enumerate(ss):
                        for k, p, nb in bypos.get(s, []):
                            nll[k] = -float(lp[r, p, nb])
            realised[name] = nll
        cost = {n: float((v - realised["none"]).mean()) for n, v in realised.items() if n != "none"}
        res["layers"][L] = {"spec_sha256": hashlib.sha256(spec.encode()).hexdigest(), "probe_exit": why[0],
                            "gate_stdout_tail": out.stdout[-6000:], "gate_stderr": out.stderr[-2000:],
                            "gate_rc": out.returncode, "gate_json": gate_json, "realised_cost": cost}
        log(f"mlp{L}: gate rc {out.returncode}, {len(gate_json)} json lines; realised cost " +
            " ".join(f"{n}:{c:+.4f}" for n, c in cost.items()))
        for j in gate_json:
            print("   ", json.dumps(j)[:300], flush=True)
    res.update({"prereg_sha256": hashlib.sha256(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                  "PREREG_PARALLAXH.md"), "rb").read()).hexdigest(),
                "script_sha256": hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest(),
                "gate_commit": "028228b", "env": E.device_class(), "wall_sec": time.time() - t0})
    json.dump(res, open(out_path, "w"), indent=1, default=float)


if __name__ == "__main__":
    main(*sys.argv[1:4])
