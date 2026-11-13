"""Coupling between sources of the agreement dynamics (GoldWorm v6 base; EXPLORATORY).

GROUPS: each form (was, were, is, are, has, have) x whether the previous word's training statistics are against it
(< 0.5) or with it (>= 0.9), "there" cases kept apart (there was, there were), and previous word unseen.
For each group with >= 30 fit items (stories 0-11,999, <= 5 per distinct pair, <= 600 items):
  first order : cosine between groups' mean gradients of log share at the base weights
  measured    : 10 re-taken steps of 0.0002 x |weights| raising THAT group only, then the per-distinct-pair change of
                EVERY group's share on stories 12,000-15,999, and ordinary bits per byte on stories 20,000 on.
Every word pair's movement is saved (not only the groups'), so off-target effects can be read.
Usage: python coupling.py <valid> <fresh> <out.json>"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import collections, json, math, re, sys
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import edit1 as X
from ks import engine as E
PRE = b"<|endoftext|>\n\n"; WORD = re.compile(r"[A-Za-z']+")
PARTNER = {"was": "were", "were": "was", "is": "are", "are": "is", "has": "have", "have": "has"}
CAP, MAXFIT, MINFIT, STEP, NSTEP = 5, 600, 30, 0.0002, 10

def group(r):
    if r["prev"] == "there": return "there " + r["form"]
    if r["vocab"] is None: return "unseen"
    if r["vocab"] < 0.5: return r["form"] + " against"
    if r["vocab"] >= 0.9: return r["form"] + " with"
    return None

def main(valid, fresh, out, mode="table"):
    raw = open(valid, "rb").read(); train = raw[:17502553].decode("ascii", "ignore")
    big = collections.Counter(); prev = None
    for m in WORD.finditer(train):
        t = m.group(0)
        if t in PARTNER and prev is not None: big[(prev, t)] += 1
        prev = t.lower()
    del train
    vset = set(p.strip() for p in raw.split(b"<|endoftext|>") if p.strip())
    parts = open(fresh, "rb").read().split(b"<|endoftext|>")[1:-1]
    kept = [p.strip().decode("ascii") for p in parts if p.strip() and all(c < 128 for c in p.strip()) and p.strip() not in vset]
    cache = E.Cache(os.path.join(X.SCRATCH, "ks_cache")); sd0, _ = X.load_params(cache.info["model_path"])
    keys = list(sd0.keys()); sizes = [sd0[k].numel() for k in keys]; P = sum(sizes)
    def unflat(v):
        o, i = {}, 0
        for k, n in zip(keys, sizes): o[k] = v[i:i + n].reshape(sd0[k].shape); i += n
        return o
    th0 = torch.cat([sd0[k].reshape(-1) for k in keys]).double().cpu(); tn = float(th0.norm())
    def occurrences(lo, hi):
        rows = []
        for si in range(lo, hi):
            w = (PRE + kept[si].encode("ascii"))[:256]; text = w.decode("ascii"); prev = None
            for m in WORD.finditer(text):
                t = m.group(0); p, e = m.start(), m.end()
                if t in PARTNER and prev is not None and p > len(PRE) + 20 and e + 3 < len(w) and text[p - 1] == " ":
                    alt = PARTNER[t]; c, ca = big[(prev, t)], big[(prev, alt)]
                    r = {"prev": prev, "form": t, "p": p, "true": w[:e + 1], "alt": w[:p] + alt.encode() + w[e:e + 1],
                         "nt": len(t) + 1, "na": len(alt) + 1, "vocab": (c + 0.5) / (c + ca + 1.0) if c + ca >= 5 else None}
                    r["group"] = group(r); rows.append(r)
                prev = t.lower()
        return rows
    def score(lp, r, seq, n):
        ps = torch.arange(r["p"] - 1, r["p"] - 1 + n, device=E.DEV)
        return lp[ps, torch.tensor(list(seq[r["p"]:r["p"] + n]), device=E.DEV)].sum()
    def logodds(sd, ch):
        idx = torch.zeros(2 * len(ch), 256, dtype=torch.long, device=E.DEV)
        for i, r in enumerate(ch):
            idx[2 * i, :len(r["true"])] = torch.tensor(list(r["true"])); idx[2 * i + 1, :len(r["alt"])] = torch.tensor(list(r["alt"]))
        lp = X.fwd(sd, idx)[0].log_softmax(-1)
        return torch.stack([score(lp[2 * i], r, r["true"], r["nt"]) - score(lp[2 * i + 1], r, r["alt"], r["na"]) for i, r in enumerate(ch)])
    @torch.no_grad()
    def shares(sd, rows):
        return np.concatenate([torch.sigmoid(logodds(sd, rows[b:b + 128]).double()).cpu().numpy() for b in range(0, len(rows), 128)])
    @torch.no_grad()
    def bpb(sd, ws):
        tot, n = 0.0, 0
        for b in range(0, len(ws), 128):
            idx = torch.tensor([list(w) for w in ws[b:b + 128]], dtype=torch.long, device=E.DEV)
            tot += float(F.cross_entropy(X.fwd(sd, idx[:, :-1])[0].reshape(-1, 256).double(), idx[:, 1:].reshape(-1), reduction="sum")); n += idx[:, 1:].numel()
        return tot / n / math.log(2)
    def grad_at(th, rows):
        v = th.float().to(E.DEV).requires_grad_(True); sd = unflat(v); acc = torch.zeros(P, dtype=torch.float64)
        for b in range(0, len(rows), 96):
            acc += torch.autograd.grad(F.logsigmoid(logodds(sd, rows[b:b + 96])).sum(), v)[0].double().cpu()
        return acc / len(rows)
    per = collections.Counter(); fit = collections.defaultdict(list)
    for r in occurrences(0, 12000):
        g = r["group"]
        if g is None or per[(r["prev"], r["form"])] >= CAP or len(fit[g]) >= MAXFIT: continue
        per[(r["prev"], r["form"])] += 1; fit[g].append(r)
    ev = occurrences(12000, 16000); bw = [w for w in ((PRE + kept[i].encode("ascii"))[:256] for i in range(20000, len(kept))) if len(w) == 256]
    names = sorted(fit); drivers = [g for g in names if len(fit[g]) >= MINFIT]
    pairs = sorted({(r["prev"], r["form"]) for r in ev}); pid = {p: i for i, p in enumerate(pairs)}
    pgroup = {}
    for r in ev: pgroup[(r["prev"], r["form"])] = r["group"]
    def by_pair(sh):
        s, n = np.zeros(len(pairs)), np.zeros(len(pairs))
        for x, r in zip(sh, ev): s[pid[(r["prev"], r["form"])]] += x; n[pid[(r["prev"], r["form"])]] += 1
        return s / n
    def by_group(bp):
        return {g: float(np.mean([bp[i] for i, p in enumerate(pairs) if pgroup[p] == g])) for g in names if any(pgroup[p] == g for p in pairs)}
    base_bp = by_pair(shares(sd0, ev)); base_g = by_group(base_bp); base_bpb = bpb(sd0, bw)
    print("groups (fit items / read pairs / base share):", {g: (len(fit[g]), sum(pgroup[p] == g for p in pairs), round(base_g.get(g, float("nan")), 3)) for g in names}, flush=True)
    if mode == "seesaw":
        # Decouple: raise BOTH against-vocab was and were together; arm "minus seesaw" removes from every step the direction
        # that trades one form for the other (gradient of all was cases minus gradient of all were cases, re-taken each step).
        rng = np.random.default_rng(3); cols = [g for g in names if g in base_g]; res = {}
        print("fit items", {g: len(fit[g]) for g in ("was against", "were against", "was with", "were with")})
        print(" " * 24 + " ".join(c[:11].rjust(11) for c in cols) + "         bpb")
        for arm in ("balanced", "balanced minus seesaw"):
            th = th0.clone()
            for st in range(1, 21):
                gv = 0.5 * (grad_at(th, fit["was against"]) + grad_at(th, fit["were against"]))
                if arm != "balanced":
                    s = grad_at(th, fit["was against"] + fit["was with"]) - grad_at(th, fit["were against"] + fit["were with"])
                    gv = gv - s * float(gv @ s) / float(s @ s)
                th = th + STEP * tn * gv / gv.norm()
                if st in (10, 20):
                    sd = unflat(th.float().to(E.DEV)); d = by_pair(shares(sd, ev)) - base_bp; b1 = bpb(sd, bw); ch = {}
                    for g in cols:
                        x = np.array([d[i] for i, p in enumerate(pairs) if pgroup[p] == g]); bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(500)]
                        ch[g] = {"mean": float(x.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "pairs": len(x)}
                    order = np.argsort(-np.abs(d))[:25]
                    res[f"{arm}@{st}"] = {"change_by_group": ch, "bits_per_byte": b1, "pair_change": {f"{p[0]} {p[1]}": round(float(d[i]), 4) for i, p in enumerate(pairs)},
                                          "largest_moves": [[f"{pairs[i][0]} {pairs[i][1]}", pgroup[pairs[i]], round(float(base_bp[i]), 3), round(float(d[i]), 3)] for i in order]}
                    print(f"{arm}@{st}".ljust(24) + " ".join(f"{ch[g]['mean']:+11.3f}" for g in cols) + f"   {b1 - base_bpb:+.4f}", flush=True)
                    print("   was against", [round(x, 3) for x in ch["was against"]["ci95"]], "were against", [round(x, 3) for x in ch["were against"]["ci95"]], flush=True)
        json.dump({"status": "EXPLORATORY", "model": "GoldWorm v6 base", "base_share_by_group": base_g, "base_bits_per_byte": base_bpb,
                   "steps": 20, "step_over_weights_norm": STEP, "read": res}, open(out, "w"))
        return
    G0 = {g: grad_at(th0, fit[g]) for g in drivers}
    cos = {a: {b: float(G0[a] @ G0[b] / (G0[a].norm() * G0[b].norm())) for b in drivers} for a in drivers}
    print("first-order cosines"); print(" " * 15 + " ".join(b[:11].rjust(11) for b in drivers))
    for a in drivers: print(a.ljust(15) + " ".join(f"{cos[a][b]:+11.2f}" for b in drivers), flush=True)
    del G0
    rng = np.random.default_rng(3); measured = {}
    print("measured: raise the ROW group for 10 steps, change in each COLUMN group's share (per distinct pair), then bits per byte")
    cols = [g for g in names if g in base_g]; print(" " * 15 + " ".join(b[:11].rjust(11) for b in cols) + "         bpb")
    for a in drivers:
        th = th0.clone()
        for _ in range(NSTEP):
            gv = grad_at(th, fit[a]); th = th + STEP * tn * gv / gv.norm()
        sd = unflat(th.float().to(E.DEV)); d = by_pair(shares(sd, ev)) - base_bp; b1 = bpb(sd, bw)
        ch = {}
        for g in cols:
            x = np.array([d[i] for i, p in enumerate(pairs) if pgroup[p] == g]); bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(500)]
            ch[g] = {"mean": float(x.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "pairs": len(x)}
        order = np.argsort(-np.abs(d))[:25]
        measured[a] = {"change_by_group": ch, "bits_per_byte": b1, "fit_items": len(fit[a]),
                       "pair_change": {f"{p[0]} {p[1]}": round(float(d[i]), 4) for i, p in enumerate(pairs)},
                       "largest_moves": [[f"{pairs[i][0]} {pairs[i][1]}", pgroup[pairs[i]], round(float(base_bp[i]), 3), round(float(d[i]), 3)] for i in order]}
        print(a.ljust(15) + " ".join(f"{ch[g]['mean']:+11.3f}" for g in cols) + f"   {b1 - base_bpb:+.4f}", flush=True)
    json.dump({"status": "EXPLORATORY", "model": "GoldWorm v6 base", "device": str(E.DEV), "groups": names, "drivers": drivers,
               "fit_items": {g: len(fit[g]) for g in names}, "base_share_by_group": base_g, "base_bits_per_byte": base_bpb,
               "base_share_by_pair": {f"{p[0]} {p[1]}": round(float(base_bp[i]), 4) for i, p in enumerate(pairs)},
               "pair_group": {f"{p[0]} {p[1]}": pgroup[p] for p in pairs},
               "first_order_cosine": cos, "steps": NSTEP, "step_over_weights_norm": STEP, "measured": measured}, open(out, "w"))
main(*sys.argv[1:5])
