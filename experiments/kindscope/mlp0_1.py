"""MLP0-1: is GoldWorm's mlp0 a lexical lookup at a word's last byte? Frozen design: PREREG_MLP0_1.md.

Usage: python mlp0_1.py <out.json>   (uses the loop cache at $KS_CACHE)
"""

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # before CUDA initialises
import json
import math
import platform
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ks import engine as E  # noqa: E402
from ks.lang import POS  # noqa: E402

SEED = 20261114
WRITES = ["wte", "attn0", "mlp0", "attn1", "mlp1"]
HORIZON = 8
D = 192


@torch.no_grad()
def forward(port, idx, patch=None):
    """idx [B,T] on DEV. patch: (name, rows, cols, vecs) replaces that write at (row, col). Returns logits, writes."""
    sd = port.sd
    B, T = idx.shape
    wte = sd["wte.weight"][idx]
    writes = {}

    def rep(name, w):
        if patch is not None and patch[0] == name:
            w = w.clone()
            w[patch[1], patch[2]] = patch[3]
        writes[name] = w
        return w

    x = rep("wte", wte) + sd["wpe.weight"][torch.arange(T, device=E.DEV)][None]
    mask = torch.ones(T, T, dtype=torch.bool, device=E.DEV).tril()
    for i in range(port.n_layer):
        h = port.ln(x, f"blocks.{i}.ln1")
        q, k, v = (h @ sd[f"blocks.{i}.attn.qkv.weight"].T).split(D, dim=2)
        q, k, v = (t.view(B, T, 4, D // 4).transpose(1, 2) for t in (q, k, v))
        att = ((q @ k.transpose(-2, -1)) / math.sqrt(D // 4)).masked_fill(~mask, float("-inf")).softmax(-1)
        x = x + rep(f"attn{i}", (att @ v).transpose(1, 2).reshape(B, T, D) @ sd[f"blocks.{i}.attn.proj.weight"].T)
        h = port.ln(x, f"blocks.{i}.ln2")
        x = x + rep(f"mlp{i}", F.gelu(h @ sd[f"blocks.{i}.mlp.fc.weight"].T, approximate=port.approx)
                    @ sd[f"blocks.{i}.mlp.proj.weight"].T)
    return port.ln(x, "ln_f") @ sd["head.weight"].T, writes


def to_idx(cache, sids):
    L = max(len(cache.windows[s]) for s in sids)
    idx = torch.zeros(len(sids), L, dtype=torch.long)
    for r, s in enumerate(sids):
        idx[r, :len(cache.windows[s])] = torch.from_numpy(cache.windows[s].astype(np.int64))
    return idx.to(E.DEV)


def nll_after(logits, idx, rows, cols, lens):
    """Mean NLL over positions t..t+HORIZON-1 (targets t+1..), truncated at each window's end. Returns [n] numpy."""
    lp = logits.log_softmax(-1)
    out = []
    for r, t, L in zip(rows, cols, lens):
        ps = torch.arange(t, min(t + HORIZON, L - 1), device=E.DEV)
        out.append(float(-lp[r, ps, idx[r, ps + 1]].double().mean()) if len(ps) else float("nan"))
    return np.array(out)


def boot(vals, stories, rng, n=1000):
    us = np.unique(stories)
    by = {s: np.where(stories == s)[0] for s in us}
    tot = np.array([vals[by[s]].sum() for s in us]); cnt = np.array([len(by[s]) for s in us])
    bs = []
    for _ in range(n):
        r = rng.integers(0, len(us), len(us))
        bs.append(tot[r].sum() / cnt[r].sum())
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def eta2(W, groups):
    mu = W.mean(0)
    tot = ((W - mu) ** 2).sum()
    btw = 0.0
    for g in np.unique(groups):
        m = groups == g
        btw += m.sum() * ((W[m].mean(0) - mu) ** 2).sum()
    return btw / tot


def main(out_path):
    torch.use_deterministic_algorithms(True)
    t0 = time.time()
    cache = E.Cache(os.environ.get("KS_CACHE", "/tmp/claude-1000/-home-emoore-RATCHET/"
                                   "4fdbd195-6bf1-45c9-8ffc-931540da4e4d/scratchpad/ks_cache"))
    port = cache.port()
    ex = np.where(cache.split["explore"])[0]
    pos = cache.b  # last byte
    story = cache.story_of
    forms, y = cache.forms, cache.y
    sids = sorted(set(story[ex].tolist()))
    lens = {s: len(cache.windows[s]) for s in sids}
    # ---- collect writes at last-byte positions, and base NLL
    Wr = {w: np.zeros((len(ex), D), dtype=np.float64) for w in WRITES}
    base = np.zeros(len(ex))
    row_of = {n: k for k, n in enumerate(ex)}
    by_story = {}
    for n in ex:
        by_story.setdefault(int(story[n]), []).append(n)
    for s0 in range(0, len(sids), 64):
        ss = sids[s0:s0 + 64]
        idx = to_idx(cache, ss)
        logits, writes = forward(port, idx)
        rows, cols, ns = [], [], []
        for r, s in enumerate(ss):
            for n in by_story[s]:
                rows.append(r); cols.append(int(pos[n])); ns.append(n)
        for w in WRITES:
            v = writes[w][torch.tensor(rows, device=E.DEV), torch.tensor(cols, device=E.DEV)].double().cpu().numpy()
            for k, n in enumerate(ns):
                Wr[w][row_of[n]] = v[k]
        b = nll_after(logits, idx, rows, cols, [lens[ss[r]] for r in rows])
        for k, n in enumerate(ns):
            base[row_of[n]] = b[k]
    rng = np.random.default_rng(SEED)
    res = {"counts": {"explore_tokens": int(len(ex)), "explore_stories": len(sids)}}

    # ---- M1: eta^2 by word form (forms with >= 20 explore occurrences)
    fe = forms[ex]
    vc = {f: c for f, c in zip(*np.unique(fe, return_counts=True))}
    keep = np.array([vc[f] >= 20 for f in fe])
    groups = fe[keep]
    m1 = {"n_tokens": int(keep.sum()), "n_forms": int(len(set(groups.tolist())))}
    for w in WRITES:
        Wk = Wr[w][keep]
        real = eta2(Wk, groups)
        null = [eta2(Wk, rng.permutation(groups)) for _ in range(200)]
        m1[w] = {"eta2": float(real), "null_p95": float(np.percentile(null, 95)), "null_mean": float(np.mean(null)),
                 "above_null": bool(real > np.percentile(null, 95))}
    # bootstrap by story of eta2(mlp0) - eta2(attn0)
    st_k = story[ex][keep]
    us = np.unique(st_k)
    diffs = []
    for _ in range(1000):
        pick = rng.choice(us, len(us))
        ii = np.concatenate([np.where(st_k == s)[0] for s in pick])
        diffs.append(eta2(Wr["mlp0"][keep][ii], groups[ii]) - eta2(Wr["attn0"][keep][ii], groups[ii]))
    m1["diff_mlp0_minus_attn0"] = float(m1["mlp0"]["eta2"] - m1["attn0"]["eta2"])
    m1["ci95"] = [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]
    m1["P1"] = bool(m1["ci95"][0] > 0 and m1["mlp0"]["above_null"] and m1["attn0"]["above_null"])
    res["M1"] = m1
    print(f"M1: eta2 " + " ".join(f"{w}={m1[w]['eta2']:.3f}(null95 {m1[w]['null_p95']:.3f})" for w in WRITES)
          + f"  mlp0-attn0 {m1['diff_mlp0_minus_attn0']:+.3f} ci {m1['ci95']} P1={m1['P1']}  [{time.time()-t0:.0f}s]",
          flush=True)

    # ---- M2: patching
    occ = {}
    for k, n in enumerate(ex):
        occ.setdefault(forms[n], []).append(k)
    by_pl = {}
    for k, n in enumerate(ex):
        by_pl.setdefault((int(y[n]), len(forms[n])), []).append(k)
    jobs = []  # (k, same_donor_k, other_donor_k)
    for k, n in enumerate(ex):
        same = [j for j in occ[forms[n]] if story[ex[j]] != story[n]]
        other = [j for j in by_pl[(int(y[n]), len(forms[n]))] if forms[ex[j]] != forms[n] and story[ex[j]] != story[n]]
        if same and other:
            jobs.append((k, same[rng.integers(len(same))], other[rng.integers(len(other))]))
    res["M2"] = {"n_eligible": len(jobs)}
    deltas = {}
    for w in ("wte", "attn0", "mlp0"):
        for cond in ("self", "same", "other"):
            out = np.zeros(len(jobs))
            for b0 in range(0, len(jobs), 128):
                jb = jobs[b0:b0 + 128]
                ss = [int(story[ex[k]]) for k, _, _ in jb]
                idx = to_idx(cache, ss)
                rows = list(range(len(jb)))
                cols = [int(pos[ex[k]]) for k, _, _ in jb]
                donor = [{"self": k, "same": s, "other": o}[cond] for k, s, o in jb]
                vecs = torch.from_numpy(Wr[w][donor]).float().to(E.DEV)
                logits, _ = forward(port, idx, (w, torch.tensor(rows, device=E.DEV), torch.tensor(cols, device=E.DEV),
                                                vecs))
                out[b0:b0 + len(jb)] = nll_after(logits, idx, rows, cols, [lens[s] for s in ss]) - base[[k for k, _, _ in jb]]
            deltas[(w, cond)] = out
            print(f"M2 {w}/{cond}: mean Δ {np.nanmean(out):+.5f} nats  [{time.time()-t0:.0f}s]", flush=True)
    js = np.array([int(story[ex[k]]) for k, _, _ in jobs])
    floor = max(float(np.nanmax(np.abs(np.concatenate([deltas[(w, 'self')] for w in ('wte', 'attn0', 'mlp0')])))), 1e-6)
    m2 = {"floor": floor, "void_below": 10 * floor}
    for (w, c), v in deltas.items():
        ok = ~np.isnan(v)
        m2[f"{w}/{c}"] = {"mean": float(v[ok].mean()), "ci95": boot(v[ok], js[ok], rng)}
    def paired(a, b):
        v = a - b; ok = ~np.isnan(v)
        return {"mean": float(v[ok].mean()), "ci95": boot(v[ok], js[ok], rng)}
    m2["plant_wte_same_exact_zero"] = bool(np.nanmax(np.abs(deltas[("wte", "same")])) == 0.0)
    m2["plant_wte_other"] = paired(deltas[("wte", "other")], deltas[("wte", "same")])
    m2["plant_ok"] = m2["plant_wte_same_exact_zero"] and m2["plant_wte_other"]["ci95"][0] > m2["void_below"]
    m2["P2"] = paired(deltas[("attn0", "same")], deltas[("mlp0", "same")])
    m2["P3"] = paired(deltas[("mlp0", "other")], deltas[("mlp0", "same")])
    for p in ("P2", "P3"):
        m2[p]["pass"] = bool(m2["plant_ok"] and m2[p]["ci95"][0] > 0 and abs(m2[p]["mean"]) > m2["void_below"])
    res["M2"].update(m2)
    print(f"M2: floor {floor:.2e} plant_ok {m2['plant_ok']} P2 {m2['P2']} P3 {m2['P3']}", flush=True)

    # ---- M3: POS re-encoding (descriptive)
    tr = cache.tr
    m3 = {}
    A = np.asarray(cache.feats["last"][1], dtype=np.float64)  # attn0 state
    M = np.asarray(cache.feats["last"][2], dtype=np.float64)  # mlp0 state
    te = cache.split["explore"]
    for ci, c in enumerate(POS):
        yy = (y == ci).astype(np.int64)
        mu, sd = A[tr].mean(0), np.where(A[tr].std(0) > 0, A[tr].std(0), 1)
        Atr = torch.from_numpy((A[tr] - mu) / sd).to(E.DEV, E.DT)
        th, gn, it, why = E.newton(Atr, torch.from_numpy(yy[tr])[None])
        Mte_a = torch.from_numpy((M[te] - mu) / sd).to(E.DEV, E.DT)
        transfer = float((E.predict(Mte_a, th)[0] == yy[te]).mean())
        own = float((E.predict(torch.from_numpy((A[te] - mu) / sd).to(E.DEV, E.DT), th)[0] == yy[te]).mean())
        mu2, sd2 = M[tr].mean(0), np.where(M[tr].std(0) > 0, M[tr].std(0), 1)
        th2, gn2, it2, why2 = E.newton(torch.from_numpy((M[tr] - mu2) / sd2).to(E.DEV, E.DT), torch.from_numpy(yy[tr])[None])
        native = float((E.predict(torch.from_numpy((M[te] - mu2) / sd2).to(E.DEV, E.DT), th2)[0] == yy[te]).mean())
        m3[c] = {"attn0_probe_on_attn0": own, "attn0_probe_on_mlp0": transfer, "native_mlp0": native,
                 "base_rate": float(max(yy[te].mean(), 1 - yy[te].mean())), "exits": [why[0], why2[0]]}
    res["M3"] = m3
    print("M3: " + " ".join(f"{c}:{v['attn0_probe_on_attn0']:.3f}->{v['attn0_probe_on_mlp0']:.3f}|{v['native_mlp0']:.3f}"
                           for c, v in m3.items()), flush=True)
    res.update({"prereg_sha256": E.V.sha256(os.path.join(os.path.dirname(os.path.abspath(__file__)), "PREREG_MLP0_1.md")),
                "script_sha256": E.V.sha256(os.path.abspath(__file__)), "cache": cache.info, "env": E.device_class(),
                "python": platform.python_version(), "wall_sec": time.time() - t0})
    json.dump(res, open(out_path, "w"), indent=1, default=float)


if __name__ == "__main__":
    main(sys.argv[1])
