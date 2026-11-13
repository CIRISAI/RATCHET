"""Swale, followed step by step (GoldWorm v6 base; EXPLORATORY). Light-memory companion to swale.py (same fit rule,
same read slices; the spectrum is in swale_step.json and is not recomputed).

FIT: agreement occurrences in stories 0-11,999 where the previous word's training statistics favour the wrong form
(< 0.5) or were never seen, "there" excluded, at most 5 per distinct (previous word, form) pair.
PATH: 40 steps of 0.0002 x |weights|, each along the gradient of the mean log share of the correct form on the fit
items, re-taken at the current weights. Arm "swale": the step has the span of 200 story windows' ordinary-loss
gradients (taken at the base weights) projected out. Arm "straight": no projection.
READ at steps 5, 10, 20, 40 on stories 12,000-15,999: the curve by vocab-share bin (per distinct pair), against-vocab
split by which form the rule asks for, "there were"/"there was", and ordinary bits per byte on stories 20,000 on.
Usage: python swale_path.py <valid> <fresh> <out.json>"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import collections, json, math, re, sys
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import edit1 as X
from ks import engine as E
PRE = b"<|endoftext|>\n\n"; WORD = re.compile(r"[A-Za-z']+")
PARTNER = {"was": "were", "were": "was", "is": "are", "are": "is", "has": "have", "have": "has"}
PLURAL = ("were", "are", "have"); CAP = 5; NKEEP = 200; STEP, NSTEP, READ_AT = 0.0002, 40, (5, 10, 20, 40)

def main(valid, fresh, out):
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
                    rows.append({"prev": prev, "form": t, "p": p, "true": w[:e + 1], "alt": w[:p] + alt.encode() + w[e:e + 1],
                                 "nt": len(t) + 1, "na": len(alt) + 1, "vocab": (c + 0.5) / (c + ca + 1.0) if c + ca >= 5 else None})
                prev = t.lower()
        return rows
    def score(lp, r, seq, n):
        ps = torch.arange(r["p"] - 1, r["p"] - 1 + n, device=E.DEV)
        return lp[ps, torch.tensor(list(seq[r["p"]:r["p"] + n]), device=E.DEV)].sum()
    def pack(ch):
        idx = torch.zeros(2 * len(ch), 256, dtype=torch.long, device=E.DEV)
        for i, r in enumerate(ch):
            idx[2 * i, :len(r["true"])] = torch.tensor(list(r["true"])); idx[2 * i + 1, :len(r["alt"])] = torch.tensor(list(r["alt"]))
        return idx
    def logodds(sd, ch):
        lp = X.fwd(sd, pack(ch))[0].log_softmax(-1)
        return torch.stack([score(lp[2 * i], r, r["true"], r["nt"]) - score(lp[2 * i + 1], r, r["alt"], r["na"]) for i, r in enumerate(ch)])
    @torch.no_grad()
    def shares(sd, rows):
        return np.concatenate([torch.sigmoid(logodds(sd, rows[b:b + 128]).double()).cpu().numpy() for b in range(0, len(rows), 128)])
    def windows(lo, hi):
        return [w for w in ((PRE + kept[i].encode("ascii"))[:256] for i in range(lo, hi)) if len(w) == 256]
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

    per = collections.Counter(); fit = []
    for r in occurrences(0, 12000):
        if r["prev"] == "there" or (r["vocab"] is not None and r["vocab"] >= 0.5) or per[(r["prev"], r["form"])] >= CAP: continue
        per[(r["prev"], r["form"])] += 1; fit.append(r)
    fp, fs = [r for r in fit if r["form"] in PLURAL], [r for r in fit if r["form"] not in PLURAL]
    gp, gs = grad_at(th0, fp), grad_at(th0, fs); cos_ps = float(gp @ gs / (gp.norm() * gs.norm()))
    print("fit", len(fit), "pairs", len(per), "| asking plural", len(fp), "singular", len(fs), "| cos of their mean gradients", round(cos_ps, 3), flush=True)
    # contour: orthonormal basis of 200 windows' ordinary-loss gradients at the base weights (float32, QR)
    kw = windows(0, 400)[:NKEEP]; K = torch.zeros(P, len(kw)); sdg = {k: v.clone().requires_grad_(True) for k, v in sd0.items()}
    for i, w in enumerate(kw):
        idx = torch.tensor(list(w), dtype=torch.long, device=E.DEV)[None]
        loss = F.cross_entropy(X.fwd(sdg, idx[:, :-1])[0].reshape(-1, 256), idx[:, 1:].reshape(-1))
        K[:, i] = torch.cat([g.reshape(-1) for g in torch.autograd.grad(loss, list(sdg.values()))]).cpu()
    del sdg; Q = torch.linalg.qr(K)[0]; del K
    proj = lambda v: v - (Q @ (Q.T @ v.float())).double()
    print("contour built", flush=True)
    ev_rows = occurrences(12000, 16000); bw = windows(20000, len(kept)); edges = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1.0001]
    seen = lambda r: r["vocab"] is not None and r["prev"] != "there"
    groups = {"against_vocab": lambda r: seen(r) and r["vocab"] < 0.5,
              "against_vocab_asking_plural": lambda r: seen(r) and r["vocab"] < 0.5 and r["form"] in PLURAL,
              "against_vocab_asking_singular": lambda r: seen(r) and r["vocab"] < 0.5 and r["form"] not in PLURAL,
              "unseen_prev": lambda r: r["vocab"] is None, "with_vocab": lambda r: seen(r) and r["vocab"] >= 0.9,
              "there_were": lambda r: r["prev"] == "there" and r["form"] == "were", "there_was": lambda r: r["prev"] == "there" and r["form"] == "was"}
    def pp(sh, f):
        by = collections.defaultdict(list)
        for i, r in enumerate(ev_rows):
            if f(r): by[(r["prev"], r["form"])].append(sh[i])
        return {"n": sum(len(v) for v in by.values()), "pairs": len(by), "per_pair": float(np.mean([np.mean(v) for v in by.values()])),
                "by_pair": {f"{a} {b}": float(np.mean(v)) for (a, b), v in by.items()}}
    def read(th):
        sd = unflat(th.float().to(E.DEV)); sh = shares(sd, ev_rows)
        c = {k: pp(sh, f) for k, f in groups.items()}
        c["bins"] = [{"range": [lo, min(hi, 1.0)], **pp(sh, lambda r, lo=lo, hi=hi: seen(r) and lo <= r["vocab"] < hi)} for lo, hi in zip(edges[:-1], edges[1:])]
        c["bits_per_byte"] = bpb(sd, bw); c["fit_share"] = float(shares(sd, fit).mean()); return c
    rng = np.random.default_rng(3)
    def change(c, base, k):
        d = np.array([c[k]["by_pair"][p] - base[k]["by_pair"][p] for p in base[k]["by_pair"]])
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(1000)]
        return {"mean": float(d.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "pairs": len(d)}
    def line(tag, c):
        print(tag.ljust(18), "bpb", round(c["bits_per_byte"], 4), "| fit", round(c["fit_share"], 3), "| against", round(c["against_vocab"]["per_pair"], 3),
              "(plural", round(c["against_vocab_asking_plural"]["per_pair"], 3), "singular", round(c["against_vocab_asking_singular"]["per_pair"], 3), ")",
              "unseen", round(c["unseen_prev"]["per_pair"], 3), "with", round(c["with_vocab"]["per_pair"], 3),
              "| there were", round(c["there_were"]["per_pair"], 3), "was", round(c["there_was"]["per_pair"], 3),
              "| bins", [round(b["per_pair"], 2) for b in c["bins"]], flush=True)
    base = read(th0); line("base", base); res = {"base": base}
    for arm in ("swale", "straight"):
        th = th0.clone()
        for st in range(1, NSTEP + 1):
            gv = grad_at(th, fit)
            if arm == "swale": gv = proj(gv)
            th = th + STEP * tn * gv / gv.norm()
            if st in READ_AT:
                c = read(th); c["net_move_over_weights_norm"] = float((th - th0).norm()) / tn
                c["change"] = {k: change(c, base, k) for k in ("against_vocab", "against_vocab_asking_plural", "against_vocab_asking_singular", "unseen_prev", "with_vocab")}
                res[f"{arm}@{st}"] = c; line(f"{arm}@{st}", c)
                print("   change against", {k: (round(v["mean"], 3), [round(x, 3) for x in v["ci95"]]) for k, v in c["change"].items() if k.startswith("against")},
                      "with", round(c["change"]["with_vocab"]["mean"], 3), "net move", round(c["net_move_over_weights_norm"], 5), flush=True)
    for c in res.values():
        for k in groups: c[k].pop("by_pair", None)
        for b in c["bins"]: b.pop("by_pair", None)
    json.dump({"status": "EXPLORATORY", "model": "GoldWorm v6 base", "device": str(E.DEV), "fit_items": len(fit), "fit_pairs": len(per),
               "fit_asking_plural": len(fp), "fit_asking_singular": len(fs), "cos_plural_vs_singular_mean_gradient": cos_ps,
               "fit_stories": [0, 12000], "read_stories": [12000, 16000], "bpb_stories": [20000, len(kept)], "bpb_windows": len(bw),
               "keep_windows": len(kw), "step_over_weights_norm": STEP, "steps": NSTEP, "read": res}, open(out, "w"))
main(*sys.argv[1:4])
