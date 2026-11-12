"""EDIT-1: surgical rank-one edit vs fine-tune vs readout-only fix for "more adjectives after determiners".
Frozen design: PREREG_EDIT1.md.  Usage: python edit1.py <corpus TinyStories-valid.txt> <out.json>
"""

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import json  # noqa: E402
import math  # noqa: E402
import pickle  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ks import engine as E  # noqa: E402
from ks.lang import POS  # noqa: E402
import vocab1 as V  # noqa: E402

TRAIN_CORE = (0, 17502553)
PREFIX = b"<|endoftext|>\n\n"
P = len(PREFIX)
CTX, D, NL = 256, 192, 6
SEEDS = (1, 2, 3)
STEPS, BATCH, LR, LR_READOUT = 400, 32, 1e-4, 1e-3
DET, ADJ, NOUN = POS.index("DET"), POS.index("ADJ"), POS.index("NOUN")
SCRATCH = os.environ.get("KS_SCRATCH", "/tmp/claude-1000/-home-emoore-RATCHET/4fdbd195-6bf1-45c9-8ffc-931540da4e4d/"
                                       "scratchpad")


# ------------------------------------------------------------------ data

def tag_train_core(corpus):
    cache_p = os.path.join(SCRATCH, "edit1_traincore_tags.pkl")
    key = V.sha256(corpus) + V.sha256(V.__file__)
    if os.path.exists(cache_p):
        d = pickle.load(open(cache_p, "rb"))
        if d["key"] == key:
            return d["stories"], d["windows"], d["toks"]
    raw = open(corpus, "rb").read()[TRAIN_CORE[0]:TRAIN_CORE[1]]
    parts = raw.split(b"<|endoftext|>")[:-1]  # last fragment cut by the boundary
    stories = [p.strip().decode("ascii") for p in parts if p.strip() and all(b < 128 for b in p.strip())]
    windows, toks = V.tag(stories)
    pickle.dump({"key": key, "stories": stories, "windows": windows, "toks": toks}, open(cache_p, "wb"))
    return stories, windows, toks


def pairs(windows, toks):
    """Adjacent kept-token pairs (one space between) in the same story: (story, a, b, y_prev, y_cur, form_cur)."""
    out = []
    for i in range(1, len(toks)):
        s0, a0, b0, y0, _ = toks[i - 1]
        s1, a1, b1, y1, f1 = toks[i]
        if s0 == s1 and a1 == b0 + 2 and windows[s1][b0 + 1] == 32:
            out.append((int(s1), int(a1), int(b1), int(y0), int(y1), str(f1)))
    return out


# ------------------------------------------------------------------ model

def load_params(path):
    from safetensors.torch import load_file
    sd = {k: v.float().to(E.DEV) for k, v in load_file(path).items()}
    tied = torch.equal(sd["wte.weight"], sd["head.weight"])
    if tied:
        del sd["head.weight"]
    return sd, tied


def fwd(sd, idx, readout=None, want_hidden=None):
    """Differentiable forward. readout = (M, c) applied to logits. want_hidden: layer L -> also return post-GELU hidden."""
    B, T = idx.shape
    x = sd["wte.weight"][idx] + sd["wpe.weight"][torch.arange(T, device=E.DEV)][None]
    mask = torch.ones(T, T, dtype=torch.bool, device=E.DEV).tril()
    hid, states = None, []
    ln = lambda x, n: F.layer_norm(x, (D,), sd[f"{n}.weight"], sd[f"{n}.bias"], 1e-5)  # noqa: E731
    for i in range(NL):
        h = ln(x, f"blocks.{i}.ln1")
        q, k, v = (h @ sd[f"blocks.{i}.attn.qkv.weight"].T).split(D, dim=2)
        q, k, v = (t.view(B, T, 4, D // 4).transpose(1, 2) for t in (q, k, v))
        att = ((q @ k.transpose(-2, -1)) / math.sqrt(D // 4)).masked_fill(~mask, float("-inf")).softmax(-1)
        x = x + (att @ v).transpose(1, 2).reshape(B, T, D) @ sd[f"blocks.{i}.attn.proj.weight"].T
        hh = F.gelu(ln(x, f"blocks.{i}.ln2") @ sd[f"blocks.{i}.mlp.fc.weight"].T, approximate="tanh")
        if want_hidden == i:
            hid = hh
        x = x + hh @ sd[f"blocks.{i}.mlp.proj.weight"].T
        states.append(x)
    head = sd["head.weight"] if "head.weight" in sd else sd["wte.weight"]
    logits = ln(x, "ln_f") @ head.T
    if readout is not None:
        logits = logits @ readout[0] + readout[1]
    return logits, hid, states


def batches(windows, sids, bs=64):
    for s0 in range(0, len(sids), bs):
        ss = sids[s0:s0 + bs]
        L = max(len(windows[s]) for s in ss)
        idx = torch.zeros(len(ss), L, dtype=torch.long)
        for r, s in enumerate(ss):
            idx[r, :len(windows[s])] = torch.from_numpy(windows[s].astype(np.int64))
        yield ss, idx.to(E.DEV)


@torch.no_grad()
def evaluate(sd, windows, sids, prs, readout=None):
    """Returns per-pair word log-prob [len(prs)] and per-story (sum NLL, count) over positions after the prefix."""
    by = {}
    for k, (s, a, b, _, _, _) in enumerate(prs):
        by.setdefault(s, []).append((k, a, b))
    wl = np.full(len(prs), np.nan)
    nll = {}
    for ss, idx in batches(windows, sids):
        lp = fwd(sd, idx, readout)[0].log_softmax(-1)
        for r, s in enumerate(ss):
            L = len(windows[s])
            t = torch.arange(P - 1, L - 1, device=E.DEV)
            v = -lp[r, t, idx[r, t + 1]].double()
            nll[s] = (float(v.sum()), int(len(t)))
            for k, a, b in by.get(s, []):
                p = torch.arange(a - 1, b, device=E.DEV)
                wl[k] = float(lp[r, p, idx[r, p + 1]].double().sum())
    return wl, nll


# ------------------------------------------------------------------ arms

def stream(windows, sids):
    return np.concatenate([windows[s] for s in sids])


def finetune(sd0, data, seed, readout_only=False, base_readout=None):
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    sd = {k: v.clone().requires_grad_(not readout_only) for k, v in sd0.items()}
    ro = None
    if readout_only:
        ro = (torch.eye(256, device=E.DEV).requires_grad_(True), torch.zeros(256, device=E.DEV).requires_grad_(True))
        params, lr = list(ro), LR_READOUT
    else:
        params, lr = list(sd.values()), LR
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), weight_decay=0.0)
    dt = torch.from_numpy(data.astype(np.int64))
    losses, traj = [], []
    for step in range(STEPS):
        st = torch.randint(0, len(dt) - CTX - 1, (BATCH,), generator=g)
        xb = torch.stack([dt[s:s + CTX + 1] for s in st.tolist()]).to(E.DEV)
        logits = fwd(sd, xb[:, :-1], ro)[0]
        loss = F.cross_entropy(logits.reshape(-1, 256), xb[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))
        if not readout_only and (step + 1) % 50 == 0:  # progression: per-tensor relative change every 50 steps
            with torch.no_grad():
                traj.append({"step": step + 1, "loss": float(loss),
                             "rel": {k: float((sd[k] - sd0[k]).norm() / sd0[k].norm()) for k in sd}})
    out = {k: v.detach() for k, v in sd.items()}
    return out, (tuple(t.detach() for t in ro) if ro else None), {"loss_first": losses[0], "loss_last": losses[-1],
                                                                   "loss_mean_last50": float(np.mean(losses[-50:])),
                                                                   "trajectory": traj}


def boot(vals, stories, rng, n=1000):
    ok = ~np.isnan(vals)
    vals, stories = vals[ok], stories[ok]
    us = np.unique(stories)
    by = {s: np.where(stories == s)[0] for s in us}
    tot = np.array([vals[by[s]].sum() for s in us]); cnt = np.array([len(by[s]) for s in us])
    bs = [tot[r].sum() / cnt[r].sum() for r in (rng.integers(0, len(us), len(us)) for _ in range(n))]
    return {"mean": float(vals.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "n": int(len(vals))}


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


# ------------------------------------------------------------------ main

def main(corpus, out_path):
    torch.use_deterministic_algorithms(True)
    t0 = time.time()
    log = lambda m: print(f"{m}  [{time.time()-t0:.0f}s]", flush=True)  # noqa: E731
    cache = E.Cache(os.path.join(SCRATCH, "ks_cache"))
    sd0, tied = load_params(cache.info["model_path"])
    # --- train_core: tag, partition, select
    stories, windows, toks = tag_train_core(corpus)
    rng = np.random.default_rng(20261115)
    order = rng.permutation(len(stories))
    cal = sorted(order[: len(stories) // 10].tolist())
    pool = order[len(stories) // 10:]
    prs = pairs(windows, toks)
    det_next, det_adj = np.zeros(len(stories)), np.zeros(len(stories))
    for s, a, b, y0, y1, _ in prs:
        if y0 == DET:
            det_next[s] += 1
            det_adj[s] += y1 == ADJ
    elig = [s for s in pool if det_next[s] >= 5]
    rate = {s: det_adj[s] / det_next[s] for s in elig}
    n_sel = len(pool) // 5
    adj_rich = sorted(sorted(elig, key=lambda s: -rate[s])[:n_sel])
    control = sorted(rng.choice(pool, size=len(adj_rich), replace=False).tolist())
    res = {"counts": {"stories": len(stories), "cal": len(cal), "pool": len(pool), "eligible": len(elig),
                      "adj_rich": len(adj_rich), "control": len(control)},
           "rates": {"adj_rich_mean": float(np.mean([rate[s] for s in adj_rich])),
                     "pool_mean": float(det_adj[pool].sum() / max(det_next[pool].sum(), 1)),
                     "control_mean": float(det_adj[control].sum() / max(det_next[control].sum(), 1))},
           "tied_embeddings": tied}
    log(f"data {res['counts']} rates {res['rates']}")
    # --- evaluation sets
    ex_stories = sorted(set(cache.story_of[cache.split["explore"]].tolist()))
    ex_set = set(ex_stories)
    ex_toks = [(int(s), int(a), int(b), int(y), f) for s, a, b, y, f in cache.toks]
    ex_prs = [p for p in pairs(cache.windows, ex_toks) if p[0] in ex_set and p[3] == DET]
    ev = {"adj": [p for p in ex_prs if p[4] == ADJ], "noun": [p for p in ex_prs if p[4] == NOUN]}
    cal_set = set(cal)
    cal_prs = [p for p in prs if p[0] in cal_set and p[3] == DET]
    cal_adj = [p for p in cal_prs if p[4] == ADJ]
    res["counts"].update({"explore_det_adj": len(ev["adj"]), "explore_det_noun": len(ev["noun"]),
                          "cal_det_adj": len(cal_adj), "cal_det_next": len(cal_prs)})

    def measure(sd, readout=None):
        wa, nll = evaluate(sd, cache.windows, ex_stories, ev["adj"] + ev["noun"], readout)
        return {"adj": wa[:len(ev["adj"])], "noun": wa[len(ev["adj"]):],
                "G": sum(v[0] for v in nll.values()) / sum(v[1] for v in nll.values()),
                "nll_story": nll}

    def cal_target(sd, readout=None):
        wa, _ = evaluate(sd, windows, cal, cal_adj, readout)
        return float(np.nanmean(wa))

    base = measure(sd0)
    base_cal = cal_target(sd0)
    # P-IDENTITY plant: zero-size arms reproduce the base exactly
    ident = measure(sd0, (torch.eye(256, device=E.DEV), torch.zeros(256, device=E.DEV)))
    floor = float(max(np.nanmax(np.abs(ident["adj"] - base["adj"])), abs(ident["G"] - base["G"])))
    res["plant_identity"] = {"max_abs_delta": floor, "exact": floor == 0.0}
    log(f"base G {base['G']:.5f}; identity-readout floor {floor:.2e}")

    st_adj = np.array([p[0] for p in ev["adj"]]); st_noun = np.array([p[0] for p in ev["noun"]])

    def deltas(m):
        return {"T_adj": boot(m["adj"] - base["adj"], st_adj, rng), "T_noun": boot(m["noun"] - base["noun"], st_noun, rng),
                "G": m["G"] - base["G"]}

    arms = {}
    # --- fine-tunes and readout maps
    adj_stream, ctl_stream = stream(windows, adj_rich), stream(windows, control)
    ft = {}
    for seed in SEEDS:
        sdA2, _, l2 = finetune(sd0, adj_stream, seed)
        sdA3, _, l3 = finetune(sd0, ctl_stream, seed)
        _, roA1, l1 = finetune(sd0, adj_stream, seed, readout_only=True)
        _, roA12, l12 = finetune(sdA2, adj_stream, seed, readout_only=True)
        ft[seed] = {"A2": sdA2, "A3": sdA3, "A1": roA1, "A12": roA12}
        mA2, mA3, mA1, mA12 = measure(sdA2), measure(sdA3), measure(sd0, roA1), measure(sdA2, roA12)
        for name, m, l in (("A2", mA2, l2), ("A3", mA3, l3), ("A1", mA1, l1), ("A1+2", mA12, l12)):
            arms.setdefault(name, {"per_seed": {}, "raw_adj": [], "raw_noun": []})
            arms[name]["per_seed"][seed] = {**deltas(m), "train_loss": l}
            arms[name]["raw_adj"].append(m["adj"]); arms[name]["raw_noun"].append(m["noun"])
        log(f"seed {seed}: " + " ".join(f"{n} T_adj {arms[n]['per_seed'][seed]['T_adj']['mean']:+.3f} "
                                       f"G {arms[n]['per_seed'][seed]['G']:+.5f}" for n in ("A1", "A2", "A3", "A1+2")))
    for name, a in arms.items():
        a["seed_mean"] = {"T_adj": boot(np.mean(a["raw_adj"], 0) - base["adj"], st_adj, rng),
                          "T_noun": boot(np.mean(a["raw_noun"], 0) - base["noun"], st_noun, rng),
                          "G": float(np.mean([v["G"] for v in a["per_seed"].values()]))}
    # P1: A2 - A1, seed-matched, paired per word
    p1 = boot(np.mean([a2 - a1 for a2, a1 in zip(arms["A2"]["raw_adj"], arms["A1"]["raw_adj"])], 0), st_adj, rng)
    res["P1"] = {**p1, "prediction": "CI includes 0", "holds": bool(p1["ci95"][0] <= 0 <= p1["ci95"][1])}
    log(f"P1 A2-A1 on T_adj: {p1}")
    # --- surgical edits
    tau = float(np.mean([cal_target(ft[s]["A2"]) for s in SEEDS])) - base_cal
    res["tau_cal"] = tau
    key_pos = {}
    for k, (s, a, b, y0, y1, f1) in enumerate(cal_prs):
        key_pos.setdefault(s, []).append((k, a - 1, y1, f1))
    surg, sel = {}, {}
    for L in range(NL):
        with torch.no_grad():
            C = torch.zeros(768, 768, device=E.DEV, dtype=torch.float64)
            ncnt = 0
            ksum = torch.zeros(768, device=E.DEV, dtype=torch.float64)
            Xs, ys, ss_of, wn, fs = [], [], [], [], []
            for ss, idx in batches(windows, cal):
                _, hid, states = fwd(sd0, idx, want_hidden=L)
                for r, s in enumerate(ss):
                    Lw = len(windows[s])
                    hh = hid[r, P - 1:Lw - 1].double()
                    C += hh.T @ hh; ncnt += hh.shape[0]
                    for k, p, y1, f1 in key_pos.get(s, []):
                        fs.append(f1)
                        ksum += hid[r, p].double()
                        Xs.append(states[L][r, p].double().cpu().numpy()); ys.append(int(y1 == ADJ)); ss_of.append(s)
                        wn.append(float((hid[r, p] @ sd0[f"blocks.{L}.mlp.proj.weight"].T).norm()))
            C = C / ncnt
            C += 1e-3 * torch.trace(C) / 768 * torch.eye(768, device=E.DEV, dtype=torch.float64)
            kv = ksum / len(Xs)
            u = torch.linalg.solve(C, kv)
        X, yv, so = np.array(Xs), np.array(ys), np.array(ss_of)
        us = np.unique(so); tr_s = set(np.random.default_rng(L).permutation(us)[: int(0.8 * len(us))].tolist())
        trm = np.array([s in tr_s for s in so])
        mu, sdv = X[trm].mean(0), np.where(X[trm].std(0) > 0, X[trm].std(0), 1)
        Xt = lambda m: torch.from_numpy((X[m] - mu) / sdv).to(E.DEV, E.DT)  # noqa: E731
        th, gn, it, why = E.newton(Xt(trm), torch.from_numpy(yv[trm])[None])
        acc = float((E.predict(Xt(~trm), th)[0] == yv[~trm]).mean())
        ctrl = []
        for c in range(20):  # control task: one random label per NEXT-WORD FORM (type-consistent), at the class prior
            crng = np.random.default_rng(1000 + c)
            lab = {}
            yl = np.array([lab.setdefault(fs[i], int(crng.random() < yv[trm].mean())) for i in range(len(yv))])
            thc, _, _, _ = E.newton(Xt(trm), torch.from_numpy(yl[trm])[None])
            ctrl.append(float((E.predict(Xt(~trm), thc)[0] == yl[~trm]).mean()))
        sel[L] = {"acc": acc, "ctrl_mean": float(np.mean(ctrl)), "selectivity": acc - float(np.mean(ctrl)),
                  "base_rate": float(max(yv[~trm].mean(), 1 - yv[~trm].mean())), "fit_exit": why[0]}
        w = th[0, :-1].cpu().double().numpy() / sdv
        d = torch.from_numpy(w / np.linalg.norm(w)).to(E.DEV)
        dW = (d[:, None] * u[None, :] / float(u @ kv)).float()  # [192, 768]

        def edited(alpha):
            sd = dict(sd0)
            sd[f"blocks.{L}.mlp.proj.weight"] = sd0[f"blocks.{L}.mlp.proj.weight"] + alpha * dW
            return sd

        cap = 200 * float(np.mean(wn))
        lo, hi, why_b = 0.0, 0.1 * float(np.mean(wn)), "matched"
        while cal_target(edited(hi)) - base_cal < tau:
            lo, hi = hi, hi * 2
            if hi > cap:
                why_b = "unreachable"
                break
        if why_b == "matched":
            for _ in range(40):
                mid = (lo + hi) / 2
                if cal_target(edited(mid)) - base_cal < tau:
                    lo = mid
                else:
                    hi = mid
                if (hi - lo) <= 1e-4 * hi:
                    break
            alpha = hi
            m = measure(edited(alpha))
            dl = deltas(m)
            surg[L] = {"alpha": alpha, "alpha_over_write_norm": alpha / float(np.mean(wn)), "exit": why_b, **dl,
                       "efficiency": dl["T_adj"]["mean"] / dl["G"] if dl["G"] > 0 else float("inf")}
        else:
            surg[L] = {"exit": why_b, "alpha_cap": cap}
        log(f"S_{L}: sel {sel[L]['selectivity']:+.3f} | " + (f"alpha {surg[L]['alpha']:.3g} T_adj "
            f"{surg[L]['T_adj']['mean']:+.3f} T_noun {surg[L]['T_noun']['mean']:+.3f} G {surg[L]['G']:+.5f}"
            if surg[L]["exit"] == "matched" else "UNREACHABLE"))
        surg[L]["dW"] = dW
    reach = [L for L in range(NL) if surg[L]["exit"] == "matched" and surg[L]["efficiency"] is not None]
    rho = spearman([sel[L]["selectivity"] for L in reach], [surg[L]["efficiency"] for L in reach]) if len(reach) >= 3 else None
    crit = {3: 1.0, 4: 1.0, 5: 0.9, 6: 0.829}.get(len(reach))
    res["P2"] = {"layers": reach, "rho": rho, "critical": crit,
                 "holds": None if rho is None else bool(rho < crit)}
    log(f"P2: rho {rho} over {reach} (critical {crit})")
    # --- weights
    wts = {}
    for k in sd0:
        dA2 = torch.stack([ft[s]["A2"][k] - sd0[k] for s in SEEDS]).mean(0)
        dA3 = torch.stack([ft[s]["A3"][k] - sd0[k] for s in SEEDS]).mean(0)
        n0 = float(sd0[k].norm())
        wts[k] = {"A2": float(dA2.norm()) / n0, "A3": float(dA3.norm()) / n0, "A2-A3": float((dA2 - dA3).norm()) / n0}
        if k.endswith("mlp.proj.weight"):
            L = int(k.split(".")[1])
            spec = (dA2 - dA3).flatten()
            wts[k]["cos_with_S"] = float(spec @ surg[L]["dW"].flatten() / (spec.norm() * surg[L]["dW"].norm() + 1e-30))
    res["weights"] = wts
    for L in surg:
        surg[L].pop("dW", None)
    res.update({"arms": {n: {"per_seed": a["per_seed"], "seed_mean": a["seed_mean"]} for n, a in arms.items()},
                "surgical": surg, "selectivity": sel, "base": {"G": base["G"], "cal_target": base_cal},
                "prereg_sha256": V.sha256(os.path.join(os.path.dirname(os.path.abspath(__file__)), "PREREG_EDIT1.md")),
                "script_sha256": V.sha256(os.path.abspath(__file__)), "env": E.device_class(),
                "wall_sec": time.time() - t0})
    json.dump(res, open(out_path, "w"), indent=1, default=float)
    log("done")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
