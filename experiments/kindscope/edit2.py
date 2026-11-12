"""EDIT-2: parallax (read vs pulled-back write directions), torque (angle to the weights' own pull), progression and
withdrawal. Frozen design: PREREG_EDIT2.md.  Usage: python edit2.py <TinyStories-valid.txt> <out.json>
"""

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import edit1 as X  # noqa: E402  (EDIT-1's data, fine-tune recipe and evaluation, unchanged)
from ks import engine as E  # noqa: E402

D, NL = 192, 6
ALPHAS = (0.25, 0.5, 1.0)


def fwd_full(sd, idx, adds=None):
    """Forward returning logits and every write; adds: {("mlp"|"attn", L): tensor [B,T,D]} added to that write."""
    B, T = idx.shape
    x = sd["wte.weight"][idx] + sd["wpe.weight"][torch.arange(T, device=E.DEV)][None]
    mask = torch.ones(T, T, dtype=torch.bool, device=E.DEV).tril()
    ln = lambda x, n: F.layer_norm(x, (D,), sd[f"{n}.weight"], sd[f"{n}.bias"], 1e-5)  # noqa: E731
    writes, states, hidden = {}, {}, {}
    for i in range(NL):
        h = ln(x, f"blocks.{i}.ln1")
        q, k, v = (h @ sd[f"blocks.{i}.attn.qkv.weight"].T).split(D, dim=2)
        q, k, v = (t.view(B, T, 4, D // 4).transpose(1, 2) for t in (q, k, v))
        att = ((q @ k.transpose(-2, -1)) / math.sqrt(D // 4)).masked_fill(~mask, float("-inf")).softmax(-1)
        w = (att @ v).transpose(1, 2).reshape(B, T, D) @ sd[f"blocks.{i}.attn.proj.weight"].T
        if adds and ("attn", i) in adds:
            w = w + adds[("attn", i)]
        writes[("attn", i)] = w
        x = x + w
        hh = F.gelu(ln(x, f"blocks.{i}.ln2") @ sd[f"blocks.{i}.mlp.fc.weight"].T, approximate="tanh")
        hidden[i] = hh
        w = hh @ sd[f"blocks.{i}.mlp.proj.weight"].T
        if adds and ("mlp", i) in adds:
            w = w + adds[("mlp", i)]
        writes[("mlp", i)] = w
        x = x + w
        states[i] = x
    return ln(x, "ln_f") @ sd["wte.weight"].T, writes, states, hidden


def train(sd_start, data, seed, steps, every=None, cb=None):
    """EDIT-1's recipe (AdamW 1e-4, batch 32 x 256, fp32) from any starting weights, with a callback every `every`."""
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    sd = {k: v.clone().requires_grad_(True) for k, v in sd_start.items()}
    opt = torch.optim.AdamW(list(sd.values()), lr=X.LR, betas=(0.9, 0.999), weight_decay=0.0)
    dt = torch.from_numpy(data.astype(np.int64))
    log = []
    if cb and every:
        log.append(cb(0, {k: v.detach() for k, v in sd.items()}))
    for step in range(steps):
        st = torch.randint(0, len(dt) - X.CTX - 1, (X.BATCH,), generator=g)
        xb = torch.stack([dt[s:s + X.CTX + 1] for s in st.tolist()]).to(E.DEV)
        loss = F.cross_entropy(X.fwd(sd, xb[:, :-1])[0].reshape(-1, 256), xb[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if cb and every and (step + 1) % every == 0:
            log.append(cb(step + 1, {k: v.detach() for k, v in sd.items()}))
    return {k: v.detach() for k, v in sd.items()}, log


def main(corpus, out_path):
    torch.use_deterministic_algorithms(True)
    t0 = time.time()
    log = lambda m: print(f"{m}  [{time.time()-t0:.0f}s]", flush=True)  # noqa: E731
    cache = E.Cache(os.path.join(X.SCRATCH, "ks_cache"))
    sd0, _ = X.load_params(cache.info["model_path"])
    stories, windows, toks = X.tag_train_core(corpus)
    rng = np.random.default_rng(20261115)  # EDIT-1's partition, reproduced exactly
    order = rng.permutation(len(stories))
    cal = sorted(order[: len(stories) // 10].tolist())
    pool = order[len(stories) // 10:]
    prs = X.pairs(windows, toks)
    dn, da = np.zeros(len(stories)), np.zeros(len(stories))
    for s, a, b, y0, y1, _ in prs:
        if y0 == X.DET:
            dn[s] += 1; da[s] += y1 == X.ADJ
    elig = [s for s in pool if dn[s] >= 5]
    rate = {s: da[s] / dn[s] for s in elig}
    adj_rich = sorted(sorted(elig, key=lambda s: -rate[s])[: len(pool) // 5])
    control = sorted(rng.choice(pool, size=len(adj_rich), replace=False).tolist())
    adj_stream, ctl_stream = X.stream(windows, adj_rich), X.stream(windows, control)
    ex_stories = sorted(set(cache.story_of[cache.split["explore"]].tolist()))
    ex_set = set(ex_stories)
    ex_toks = [(int(s), int(a), int(b), int(y), f) for s, a, b, y, f in cache.toks]
    ex_prs = [p for p in X.pairs(cache.windows, ex_toks) if p[0] in ex_set and p[3] == X.DET]
    ev_adj = [p for p in ex_prs if p[4] == X.ADJ]
    ev_noun = [p for p in ex_prs if p[4] == X.NOUN]
    st_adj = np.array([p[0] for p in ev_adj]); st_noun = np.array([p[0] for p in ev_noun])
    sub80 = ex_stories[:80]; sub_set = set(sub80)
    adj80 = [p for p in ev_adj if p[0] in sub_set]
    cal_set = set(cal)
    cal_prs = [p for p in prs if p[0] in cal_set and p[3] == X.DET]
    res = {"counts": {"explore_det_adj": len(ev_adj), "explore_det_noun": len(ev_noun), "cal_key": len(cal_prs),
                      "progression_subset_adj": len(adj80)}}

    def measure(sd):
        wa, nll = X.evaluate(sd, cache.windows, ex_stories, ev_adj + ev_noun)
        return wa[:len(ev_adj)], wa[len(ev_adj):], sum(v[0] for v in nll.values()) / sum(v[1] for v in nll.values())

    b_adj, b_noun, b_G = measure(sd0)

    def deltas(sd):
        a, n, G = measure(sd)
        return {"T_adj": X.boot(a - b_adj, st_adj, rng), "T_noun": X.boot(n - b_noun, st_noun, rng), "G": G - b_G}, a

    # ---------------- directions per layer (CAL)
    key = {}
    for s, a, b, y0, y1, f in cal_prs:
        key.setdefault(s, []).append((a, b, y1))
    dirs = {}
    for L in range(NL):
        g_sum = torch.zeros(D, device=E.DEV, dtype=torch.float64)
        C = torch.zeros(768, 768, device=E.DEV, dtype=torch.float64); ncnt = 0
        ksum = torch.zeros(768, device=E.DEV, dtype=torch.float64); nk = 0
        Xs, ys, wn = [], [], []
        for ss, idx in X.batches(windows, cal, bs=32):
            add = torch.zeros(idx.shape[0], idx.shape[1], D, device=E.DEV, requires_grad=True)
            logits, writes, states, hidden = fwd_full(sd0, idx, {("mlp", L): add})
            lp = logits.log_softmax(-1)
            J, rows, cols = 0, [], []
            for r, s in enumerate(ss):
                for a, b, y1 in key.get(s, []):
                    rows.append(r); cols.append(a - 1)
                    if y1 == X.ADJ:
                        p = torch.arange(a - 1, b, device=E.DEV)
                        J = J + lp[r, p, idx[r, p + 1]].sum()
            if not rows:
                continue
            if torch.is_tensor(J):
                J.backward()
                rt, ct = torch.tensor(rows, device=E.DEV), torch.tensor(cols, device=E.DEV)
                g_sum += add.grad[rt, ct].double().sum(0)
            with torch.no_grad():
                rt, ct = torch.tensor(rows, device=E.DEV), torch.tensor(cols, device=E.DEV)
                for r, s in enumerate(ss):
                    hh = hidden[L][r, X.P - 1:len(windows[s]) - 1].double()
                    C += hh.T @ hh; ncnt += hh.shape[0]
                ksum += hidden[L][rt, ct].double().sum(0); nk += len(rows)
                Xs.append(states[L][rt, ct].double().cpu().numpy())
                ys += [int(y1 == X.ADJ) for r, s in enumerate(ss) for a, b, y1 in key.get(s, [])]
                wn.append(writes[("mlp", L)][rt, ct].norm(dim=1).double().cpu().numpy())
        C = C / ncnt + 1e-3 * torch.trace(C / ncnt) / 768 * torch.eye(768, device=E.DEV, dtype=torch.float64)
        kv = ksum / nk
        u = torch.linalg.solve(C, kv)
        Xa = np.concatenate(Xs); yv = np.array(ys)
        mu, sdv = Xa.mean(0), np.where(Xa.std(0) > 0, Xa.std(0), 1)
        th, _, _, why = E.newton(torch.from_numpy((Xa - mu) / sdv).to(E.DEV, E.DT), torch.from_numpy(yv)[None])
        d_read = torch.from_numpy(th[0, :-1].cpu().numpy() / sdv).to(E.DEV); d_read = d_read / d_read.norm()
        d_write = g_sum / g_sum.norm()
        dirs[L] = {"read": d_read, "write": d_write, "u": u, "k": kv, "wn": float(np.concatenate(wn).mean()),
                   "probe_exit": why[0], "parallax_deg": math.degrees(math.acos(float(d_read @ d_write)))}
        log(f"L{L}: parallax {dirs[L]['parallax_deg']:.1f} deg, write norm {dirs[L]['wn']:.2f}, probe {why[0]}")

    def edited(spec):
        sd = dict(sd0)
        for L, name, a in spec:
            dd = dirs[L]
            dW = (dd[name][:, None] * dd["u"][None, :] / float(dd["u"] @ dd["k"])).float()
            sd[f"blocks.{L}.mlp.proj.weight"] = sd[f"blocks.{L}.mlp.proj.weight"] + float(a * dd["wn"]) * dW
        return sd

    # P-IDENTITY
    z_adj, z_noun, z_G = measure(edited([(0, "write", 0.0)]))
    floor = float(max(np.nanmax(np.abs(z_adj - b_adj)), abs(z_G - b_G)))
    res["plant_identity"] = {"max_abs": floor, "ok": floor <= 1e-6}
    # ---------------- P1: read vs write per layer and size
    edits, raw = {}, {}
    for L in range(NL):
        for name in ("read", "write"):
            for a in ALPHAS:
                d, ra = deltas(edited([(L, name, a)]))
                edits[f"{L}/{name}/{a}"] = d
                raw[(L, name, a)] = ra
        log(f"L{L} @1/2: read T_adj {edits[f'{L}/read/0.5']['T_adj']['mean']:+.3f} write "
            f"{edits[f'{L}/write/0.5']['T_adj']['mean']:+.3f} G {edits[f'{L}/write/0.5']['G']:+.4f}")
    p1 = {}
    for L in range(NL):
        p1[L] = X.boot(raw[(L, "write", 0.5)] - raw[(L, "read", 0.5)], st_adj, rng)
        p1[L]["pass"] = bool(p1[L]["ci95"][0] > 0)
    res["P1"] = {"per_layer": p1, "holds": bool(all(v["pass"] for v in p1.values()))}
    res["edits"] = edits
    res["parallax_deg"] = {L: dirs[L]["parallax_deg"] for L in range(NL)}
    d, _ = deltas(edited([(L, "write", 0.25) for L in range(NL)]))
    res["distributed_write_quarter"] = d
    log(f"P1 holds={res['P1']['holds']}; distributed T_adj {d['T_adj']['mean']:+.3f} G {d['G']:+.4f}")

    # ---------------- fine-tunes with progression, and torque
    key_ex = {}
    for s, a, b, y0, y1, f in ex_prs:
        if s in sub_set:
            key_ex.setdefault(s, []).append(a - 1)
    with torch.no_grad():
        base_w = {}
        for ss, idx in X.batches(cache.windows, sub80):
            _, writes, _, _ = fwd_full(sd0, idx)
            for r, s in enumerate(ss):
                for p in key_ex.get(s, []):
                    for kk, w in writes.items():
                        base_w.setdefault(kk, []).append(w[r, p].double())
        base_w = {k: torch.stack(v) for k, v in base_w.items()}
        base80 = np.nanmean(X.evaluate(sd0, cache.windows, sub80, adj80)[0])

    def prog(step, sd):
        with torch.no_grad():
            T = float(np.nanmean(X.evaluate(sd, cache.windows, sub80, adj80)[0]) - base80)
            cur = {}
            for ss, idx in X.batches(cache.windows, sub80):
                _, writes, _, _ = fwd_full(sd, idx)
                for r, s in enumerate(ss):
                    for p in key_ex.get(s, []):
                        for kk, w in writes.items():
                            cur.setdefault(kk, []).append(w[r, p].double())
            ch = {f"{kk[0]}{kk[1]}": float((torch.stack(v) - base_w[kk]).norm(dim=1).mean() / base_w[kk].norm(dim=1).mean())
                  for kk, v in cur.items()}
        return {"step": step, "T_adj_sub80": T, "write_change": ch}

    A2, A3, progression = {}, {}, {}
    for seed in X.SEEDS:
        A2[seed], progression[seed] = train(sd0, adj_stream, seed, X.STEPS, every=50, cb=prog)
        A3[seed], _ = train(sd0, ctl_stream, seed, X.STEPS)
        log(f"seed {seed}: A2 progression end T_adj(sub80) {progression[seed][-1]['T_adj_sub80']:+.3f}")
    res["progression"] = progression
    sdg = {k: v.clone().requires_grad_(True) for k, v in sd0.items()}
    gacc = {k: torch.zeros_like(v) for k, v in sd0.items()}
    dt = torch.from_numpy(ctl_stream.astype(np.int64)); gg = torch.Generator().manual_seed(7)
    for _ in range(64):
        st = torch.randint(0, len(dt) - X.CTX - 1, (X.BATCH,), generator=gg)
        xb = torch.stack([dt[s:s + X.CTX + 1] for s in st.tolist()]).to(E.DEV)
        loss = F.cross_entropy(X.fwd(sdg, xb[:, :-1])[0].reshape(-1, 256), xb[:, 1:].reshape(-1))
        grads = torch.autograd.grad(loss, list(sdg.values()))
        for k, gv in zip(sdg.keys(), grads):
            gacc[k] += gv.detach()
    torque = {}
    for k in sd0:
        dlt = torch.stack([A2[s][k] - A3[s][k] for s in X.SEEDS]).mean(0).flatten()
        pull = -gacc[k].flatten()
        c = float(dlt @ pull / (dlt.norm() * pull.norm() + 1e-30))
        torque[k] = {"deg": math.degrees(math.acos(max(-1.0, min(1.0, c)))), "cos": c,
                     "rel_change": float(dlt.norm() / sd0[k].norm())}
    res["torque"] = torque
    log("torque (deg): " + " ".join(f"{k.replace('blocks.', 'b').replace('.weight', '')}={v['deg']:.0f}"
                                    for k, v in torque.items() if "proj" in k or "fc" in k))

    # ---------------- withdrawal
    def tadj_cb(step, sd):
        a, _, G = measure(sd)
        return {"step": step, "adj": a, "G": G}

    wd, kept = {}, {}
    for seed in X.SEEDS:
        _, wd[seed] = train(A2[seed], ctl_stream, 100 + seed, X.STEPS, every=100, cb=tadj_cb)
        _, kept[seed] = train(A2[seed], adj_stream, 100 + seed, X.STEPS, every=100, cb=tadj_cb)
    def path(runs):
        return [{"step": runs[X.SEEDS[0]][i]["step"],
                 "T_adj_mean": float(np.mean([np.nanmean(runs[s][i]["adj"] - b_adj) for s in X.SEEDS])),
                 "G_mean": float(np.mean([runs[s][i]["G"] - b_G for s in X.SEEDS]))} for i in range(len(runs[X.SEEDS[0]]))]
    res["withdrawal"] = {"withdrawn": path(wd), "kept": path(kept)}
    ret = np.mean([wd[s][-1]["adj"] - wd[s][0]["adj"] for s in X.SEEDS], 0)
    ret_kept = np.mean([kept[s][-1]["adj"] - kept[s][0]["adj"] for s in X.SEEDS], 0)
    p2 = X.boot(ret, st_adj, rng)
    res["P2"] = {**p2, "kept_change": X.boot(ret_kept, st_adj, rng),
                 "plant_force_ok": bool(np.nanmean(ret_kept) >= np.nanmean(ret)),
                 "holds": bool(np.nanmean(ret_kept) >= np.nanmean(ret) and p2["ci95"][1] < 0)}
    log(f"P2 withdrawn change {p2}; kept {res['P2']['kept_change']}")
    bestL = max(range(NL), key=lambda L: edits[f"{L}/write/0.5"]["T_adj"]["mean"] / max(edits[f"{L}/write/0.5"]["G"], 1e-9))
    sdS = edited([(bestL, "write", 0.5)])
    _, swd = train(sdS, ctl_stream, 200, X.STEPS, every=100, cb=tadj_cb)
    res["surgical_withdrawal"] = {"layer": bestL, "path": [{"step": r["step"], "T_adj": float(np.nanmean(r["adj"] - b_adj)),
                                                            "G": r["G"] - b_G} for r in swd]}
    log(f"surgical L{bestL} withdrawal path {res['surgical_withdrawal']['path']}")
    res.update({"prereg_sha256": X.V.sha256(os.path.join(os.path.dirname(os.path.abspath(__file__)), "PREREG_EDIT2.md")),
                "script_sha256": X.V.sha256(os.path.abspath(__file__)), "env": E.device_class(),
                "wall_sec": time.time() - t0})
    json.dump(res, open(out_path, "w"), indent=1, default=float)
    log("done")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
