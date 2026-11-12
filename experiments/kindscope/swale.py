"""Swale: move the agreement curve en masse by one coordinated weight change that runs along the contour of ordinary
behaviour (GoldWorm v6 base; EXPLORATORY, nothing frozen).

FIT (stories 0-11,999 of the FRESH-1 kept list): agreement occurrences where the previous word's training statistics
favour the WRONG form (vocab share < 0.5) or were never seen, "there" excluded (its rule depends on the next noun),
at most 5 per distinct (previous word, form) pair. For each, the gradient over ALL weights of
log P(correct form) - log P(partner).
  drive    : spectrum of those per-item gradients -- how few shared directions move the items together; the mean
             gradient g; how many items g moves the right way; which tensors hold it.
  contour  : per-window gradients of the ordinary loss on 200 story windows; their span is the set of first-order
             changes to ordinary behaviour on those windows. swale = g with that span projected out.
ARMS at matched first-order gain on the fit items (1, 2, 4 nats): swale, straight (g unprojected), random (a random
direction at the swale's norm).
READ (stories 12,000-15,999, agreement not read before): the whole curve by vocab-share bin, per distinct pair;
"there were"; ordinary bits per byte on stories 20,000 on.
Usage: python swale.py <valid> <fresh> <out.json>"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import collections, json, math, re, sys
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import edit1 as X
from ks import engine as E
PRE = b"<|endoftext|>\n\n"; WORD = re.compile(r"[A-Za-z']+")
PARTNER = {"was": "were", "were": "was", "is": "are", "are": "is", "has": "have", "have": "has"}
GAINS = [1.0, 2.0, 4.0]; CAP = 5; NKEEP = 200; NFIT = 400

def main(valid, fresh, out):
    raw = open(valid, "rb").read(); train = raw[:17502553].decode("ascii", "ignore")
    big = collections.Counter(); prev = None
    for m in WORD.finditer(train):
        t = m.group(0)
        if t in PARTNER and prev is not None: big[(prev, t)] += 1
        prev = t.lower()
    vset = set(p.strip() for p in raw.split(b"<|endoftext|>") if p.strip())
    parts = open(fresh, "rb").read().split(b"<|endoftext|>")[1:-1]
    kept = [p.strip().decode("ascii") for p in parts if p.strip() and all(c < 128 for c in p.strip()) and p.strip() not in vset]
    cache = E.Cache(os.path.join(X.SCRATCH, "ks_cache")); sd0, _ = X.load_params(cache.info["model_path"])
    keys = list(sd0.keys()); sizes = [sd0[k].numel() for k in keys]; P = sum(sizes)
    flat = lambda d: torch.cat([d[k].reshape(-1) for k in keys])
    def unflat(v):
        o, i = {}, 0
        for k, n in zip(keys, sizes): o[k] = v[i:i + n].reshape(sd0[k].shape); i += n
        return o
    th0 = flat(sd0)

    def occurrences(lo, hi):
        rows = []
        for si in range(lo, hi):
            w = (PRE + kept[si].encode("ascii"))[:256]; text = w.decode("ascii"); prev = None
            for m in WORD.finditer(text):
                t = m.group(0); p, e = m.start(), m.end()
                if t in PARTNER and prev is not None and p > len(PRE) + 20 and e + 3 < len(w) and text[p - 1] == " ":
                    alt = PARTNER[t]; c, ca = big[(prev, t)], big[(prev, alt)]
                    rows.append({"story": si, "prev": prev, "form": t, "p": p, "true": w[:e + 1], "alt": w[:p] + alt.encode() + w[e:e + 1],
                                 "nt": len(t) + 1, "na": len(alt) + 1, "vocab": (c + 0.5) / (c + ca + 1.0) if c + ca >= 5 else None})
                prev = t.lower()
        return rows
    def score(lp, r, seq, n):       # lp [T,256] for this row
        ps = torch.arange(r["p"] - 1, r["p"] - 1 + n, device=E.DEV)
        return lp[ps, torch.tensor(list(seq[r["p"]:r["p"] + n]), device=E.DEV)].sum()
    def logodds_one(sd, r):
        L = max(len(r["true"]), len(r["alt"])); idx = torch.zeros(2, L, dtype=torch.long, device=E.DEV)
        idx[0, :len(r["true"])] = torch.tensor(list(r["true"])); idx[1, :len(r["alt"])] = torch.tensor(list(r["alt"]))
        lp = X.fwd(sd, idx)[0].log_softmax(-1)
        return score(lp[0], r, r["true"], r["nt"]) - score(lp[1], r, r["alt"], r["na"])
    @torch.no_grad()
    def shares(sd, rows):
        seqs = [s for r in rows for s in (r["true"], r["alt"])]; out = []
        for b in range(0, len(seqs), 256):
            ch = seqs[b:b + 256]; idx = torch.zeros(len(ch), 256, dtype=torch.long, device=E.DEV)
            for i, s in enumerate(ch): idx[i, :len(s)] = torch.tensor(list(s))
            lp = X.fwd(sd, idx)[0].log_softmax(-1).double()
            for i, s in enumerate(ch):
                r = rows[(b + i) // 2]; out.append(float(score(lp[i], r, s, r["nt"] if (b + i) % 2 == 0 else r["na"])))
        o = np.array(out).reshape(-1, 2)
        return 1.0 / (1.0 + np.exp(o[:, 1] - o[:, 0]))
    def windows(lo, hi):
        ws = [(PRE + kept[i].encode("ascii"))[:256] for i in range(lo, hi)]
        return [w for w in ws if len(w) == 256]
    @torch.no_grad()
    def bpb(sd, ws):
        tot, n = 0.0, 0
        for b in range(0, len(ws), 128):
            idx = torch.tensor([list(w) for w in ws[b:b + 128]], dtype=torch.long, device=E.DEV)
            lg = X.fwd(sd, idx[:, :-1])[0]
            tot += float(F.cross_entropy(lg.reshape(-1, 256).double(), idx[:, 1:].reshape(-1), reduction="sum")); n += idx[:, 1:].numel()
        return tot / n / math.log(2)

    # ---- fit items
    allfit = occurrences(0, 12000); per = collections.Counter(); fit = []
    rng = np.random.default_rng(11)
    for i in rng.permutation(len(allfit)):
        r = allfit[i]
        if r["prev"] == "there" or (r["vocab"] is not None and r["vocab"] >= 0.5): continue
        if per[(r["prev"], r["form"])] >= CAP: continue
        per[(r["prev"], r["form"])] += 1; fit.append(r)
        if len(fit) == NFIT: break
    print("fit items", len(fit), "distinct pairs", len(per), "params", P, flush=True)
    sdg = {k: v.clone().requires_grad_(True) for k, v in sd0.items()}
    G = torch.zeros(len(fit), P); base_lo = []
    for i, r in enumerate(fit):
        lo = logodds_one(sdg, r); base_lo.append(float(lo))
        G[i] = torch.cat([g.reshape(-1) for g in torch.autograd.grad(lo, list(sdg.values()))]).cpu()
    kw = windows(0, 400)[:NKEEP]; K = torch.zeros(len(kw), P)
    for i, w in enumerate(kw):
        idx = torch.tensor(list(w), dtype=torch.long, device=E.DEV)[None]
        loss = F.cross_entropy(X.fwd(sdg, idx[:, :-1])[0].reshape(-1, 256), idx[:, 1:].reshape(-1))
        K[i] = torch.cat([g.reshape(-1) for g in torch.autograd.grad(loss, list(sdg.values()))]).cpu()
    print("gradients done", flush=True)
    # ---- drive
    Gd = G.double(); gram = (Gd @ Gd.T).numpy(); ev, U = np.linalg.eigh(gram); ev, U = ev[::-1].clip(0), U[:, ::-1]
    g = Gd.mean(0); along = (Gd @ g).numpy()
    v1 = (torch.from_numpy(U[:, 0].copy()) @ Gd) / math.sqrt(ev[0])
    gn2 = float(g @ g); tens, i = {}, 0
    for k, n in zip(keys, sizes): tens[k] = float(g[i:i + n] @ g[i:i + n]) / gn2; i += n
    grp = collections.defaultdict(float)
    for k, v in tens.items():
        m = re.match(r"blocks\.(\d)\.(attn|mlp)\.", k); grp[f"{m.group(2)}{m.group(1)}" if m else ("norms" if ".ln" in k or k.startswith("ln_f") else k.split(".")[0])] += v
    drive = {"energy_top": {str(k): float(ev[:k].sum() / ev.sum()) for k in (1, 2, 5, 10, 20, 50)},
             "participation_ratio": float(ev.sum() ** 2 / (ev ** 2).sum()),
             "items_mean_gradient_moves_right_way": float((along > 0).mean()), "cos_mean_top1": abs(float(g @ v1) / math.sqrt(gn2)),
             "mean_gradient_share_by_block": dict(grp), "base_logodds_mean": float(np.mean(base_lo))}
    print("drive", {k: (v if not isinstance(v, dict) else {a: round(b, 3) for a, b in v.items()}) for k, v in drive.items()}, flush=True)
    # ---- contour
    Kd = K.double(); kg = (Kd @ Kd.T).numpy(); kev, KU = np.linalg.eigh(kg); keep = kev > kev.max() * 1e-8
    Q = (torch.from_numpy(KU[:, keep] / np.sqrt(kev[keep])).T @ Kd)                   # orthonormal rows spanning the keep gradients
    sw = g - Q.T @ (Q @ g)
    del G, K, Kd
    rd = torch.from_numpy(np.random.default_rng(5).standard_normal(P)); 
    contour = {"keep_windows": len(kw), "keep_rank": int(keep.sum()), "share_of_mean_gradient_inside_keep_span": 1 - float(sw @ sw) / gn2}
    print("contour", contour, flush=True)
    # ---- read
    ev_rows = occurrences(12000, 16000); bw = windows(20000, len(kept))
    edges = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1.0001]
    def curve(sh):
        o = {"bins": []}
        def pp(sel):
            by = collections.defaultdict(list)
            for i in sel: by[(ev_rows[i]["prev"], ev_rows[i]["form"])].append(sh[i])
            return {"n": len(sel), "pairs": len(by), "per_pair": float(np.mean([np.mean(v) for v in by.values()])) if by else None,
                    "by_pair": {f"{a} {b}": float(np.mean(v)) for (a, b), v in by.items()}}
        for lo, hi in zip(edges[:-1], edges[1:]):
            o["bins"].append({"range": [lo, min(hi, 1.0)], **pp([i for i, r in enumerate(ev_rows) if r["vocab"] is not None and lo <= r["vocab"] < hi and r["prev"] != "there"])})
        o["against_vocab"] = pp([i for i, r in enumerate(ev_rows) if r["vocab"] is not None and r["vocab"] < 0.5 and r["prev"] != "there"])
        o["unseen_prev"] = pp([i for i, r in enumerate(ev_rows) if r["vocab"] is None])
        o["with_vocab"] = pp([i for i, r in enumerate(ev_rows) if r["vocab"] is not None and r["vocab"] >= 0.9 and r["prev"] != "there"])
        o["there_were"] = pp([i for i, r in enumerate(ev_rows) if r["prev"] == "there" and r["form"] == "were"])
        o["there_was"] = pp([i for i, r in enumerate(ev_rows) if r["prev"] == "there" and r["form"] == "was"])
        return o
    def read(delta):
        sd = unflat((th0.double().cpu() + delta).float().to(E.DEV)) if delta is not None else sd0
        c = curve(shares(sd, ev_rows)); c["bits_per_byte"] = bpb(sd, bw); c["fit_share"] = float(shares(sd, fit).mean()); return c
    base = read(None); res = {"base": base}
    def line(tag, c):
        print(tag.ljust(22), "bpb", round(c["bits_per_byte"], 4), "| fit", round(c["fit_share"], 3), "| against-vocab", round(c["against_vocab"]["per_pair"], 3),
              "unseen", round(c["unseen_prev"]["per_pair"], 3), "with-vocab", round(c["with_vocab"]["per_pair"], 3),
              "| there were", round(c["there_were"]["per_pair"], 3), "there was", round(c["there_was"]["per_pair"], 3),
              "| bins", [round(b["per_pair"], 2) for b in c["bins"]], flush=True)
    line("base", base)
    brng = np.random.default_rng(3)
    for gain in GAINS:
        a_sw = gain / float(g @ sw); a_st = gain / gn2
        arms = {"swale": a_sw * sw, "straight": a_st * g, "random": rd * (a_sw * float(sw.norm()) / float(rd.norm()))}
        for name, dl in arms.items():
            c = read(dl); c["delta_norm"] = float(dl.norm()); c["delta_norm_over_weights_norm"] = float(dl.norm() / th0.double().norm())
            b0, b1 = base["against_vocab"]["by_pair"], c["against_vocab"]["by_pair"]; d = np.array([b1[k] - b0[k] for k in b0])
            bs = [d[brng.integers(0, len(d), len(d))].mean() for _ in range(1000)]
            c["against_vocab_change"] = {"mean": float(d.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "pairs": len(d)}
            res[f"{name}@{gain}"] = c; line(f"{name}@{gain}", c)
            print("   against-vocab change", round(d.mean(), 3), [round(x, 3) for x in c["against_vocab_change"]["ci95"]], "norm/|theta|", round(c["delta_norm_over_weights_norm"], 5), flush=True)
    # ---- drive, by which form the rule asks for
    plural = torch.tensor([r["form"] in ("were", "are", "have") for r in fit])
    gp, gs = (plural.double() / plural.sum()) @ Gd, ((~plural).double() / (~plural).sum()) @ Gd
    del Gd
    drive["fit_items_asking_plural_form"] = int(plural.sum()); drive["cos_plural_vs_singular_mean_gradient"] = float(gp @ gs / (gp.norm() * gs.norm()))
    print("plural-asking", int(plural.sum()), "of", len(fit), "cos(plural, singular) mean gradients", round(drive["cos_plural_vs_singular_mean_gradient"], 3), flush=True)
    # ---- path: many small steps, the direction re-taken at each step (the curve is followed, not extrapolated)
    per2 = collections.Counter(); fitall = []
    for r in allfit:
        if r["prev"] == "there" or (r["vocab"] is not None and r["vocab"] >= 0.5) or per2[(r["prev"], r["form"])] >= CAP: continue
        per2[(r["prev"], r["form"])] += 1; fitall.append(r)
    def grad_at(th):
        v = th.float().to(E.DEV).requires_grad_(True); sd = unflat(v); acc = torch.zeros(P, dtype=torch.float64)
        for b in range(0, len(fitall), 96):
            ch = fitall[b:b + 96]; idx = torch.zeros(2 * len(ch), 256, dtype=torch.long, device=E.DEV)
            for i, r in enumerate(ch):
                idx[2 * i, :len(r["true"])] = torch.tensor(list(r["true"])); idx[2 * i + 1, :len(r["alt"])] = torch.tensor(list(r["alt"]))
            lp = X.fwd(sd, idx)[0].log_softmax(-1)
            lo = torch.stack([score(lp[2 * i], r, r["true"], r["nt"]) - score(lp[2 * i + 1], r, r["alt"], r["na"]) for i, r in enumerate(ch)])
            acc += torch.autograd.grad(F.logsigmoid(lo).sum(), v)[0].double().cpu()
        return acc / len(fitall)
    STEP, NSTEP, READ_AT = 0.0002, 40, (5, 10, 20, 40); tn = float(th0.double().norm()); path = {}
    print("path fit items", len(fitall), "pairs", len(per2), flush=True)
    for arm in ("swale", "straight"):
        th = th0.double().cpu().clone()
        for st in range(1, NSTEP + 1):
            gv = grad_at(th)
            if arm == "swale": gv = gv - Q.T @ (Q @ gv)
            th = th + STEP * tn * gv / gv.norm()
            if st in READ_AT:
                dl = th - th0.double().cpu(); c = read(dl); c["delta_norm_over_weights_norm"] = float(dl.norm()) / tn
                b0, b1 = base["against_vocab"]["by_pair"], c["against_vocab"]["by_pair"]; d = np.array([b1[k] - b0[k] for k in b0])
                bs = [d[brng.integers(0, len(d), len(d))].mean() for _ in range(1000)]
                c["against_vocab_change"] = {"mean": float(d.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "pairs": len(d)}
                c["fitall_share"] = float(shares(unflat(th.float().to(E.DEV)), fitall).mean())
                path[f"{arm}@step{st}"] = c; line(f"path {arm}@{st}", c)
                print("   against-vocab change", round(d.mean(), 3), [round(x, 3) for x in c["against_vocab_change"]["ci95"]], "net move/|theta|", round(c["delta_norm_over_weights_norm"], 5), "fit(all)", round(c["fitall_share"], 3), flush=True)
    res.update({"path " + k: v for k, v in path.items()})
    for c in res.values():
        for k in ("against_vocab", "unseen_prev", "with_vocab", "there_were", "there_was"): c[k].pop("by_pair", None)
        for b in c["bins"]: b.pop("by_pair", None)
    json.dump({"status": "EXPLORATORY", "model": "GoldWorm v6 base", "device": str(E.DEV), "fit_items": len(fit), "fit_pairs": len(per),
               "fit_stories": [0, 12000], "read_stories": [12000, 16000], "bpb_stories": [20000, len(kept)], "bpb_windows": len(bw),
               "params": P, "drive": drive, "contour": contour, "gains_nats_first_order": GAINS, "path": {"step_over_weights_norm": 0.0002, "steps": 40, "objective": "mean log share of the correct form on the fit items, direction re-taken each step; keep span fixed at the base weights", "fit_items": len(fitall)}, "read": res}, open(out, "w"))
main(*sys.argv[1:4])
