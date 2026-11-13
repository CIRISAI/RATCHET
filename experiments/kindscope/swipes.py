"""Big swipes: many two-way word choices at once, and how raising each moves all the others
(GoldWorm v6 base; EXPLORATORY). Each FAMILY is a set of word swaps; at every real occurrence of a member the model's
share for the word the text has, against its partner, is read through the whole network.
  base     : share per family and per form, per distinct (previous word, form) pair, on stories 12,000-15,999
  swipe    : 10 re-taken steps of 0.0002 x |weights| raising ONE family (its forms weighted equally, so a plain
             trade of one form for the other cancels to first order), then every family read again, with ordinary
             bits per byte on stories 20,000 on. Every pair's movement is saved.
Fit: stories 0-11,999, <= 5 per distinct (previous word, form), <= 300 per form.  Real text only.
Usage: python swipes.py <valid> <fresh> <out.json>"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import collections, json, math, re, sys
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import edit1 as X
from ks import engine as E
PRE = b"<|endoftext|>\n\n"; WORD = re.compile(r"[A-Za-z']+")
CAP, MAXFORM, MINFORM, STEP, NSTEP, MAXREAD = 5, 300, 15, 0.0002, 10, 1500
BI = {"agreement": ["was were", "is are", "has have"], "tense": ["is was", "are were", "can could", "will would"],
      "gender": ["he she", "He She"], "pronoun number": ["it they", "It They", "he they", "she they"],
      "negation": ["did didn't", "could couldn't", "was wasn't", "can can't", "do don't"],
      "this/these": ["this these"], "this/that": ["this that"], "and/but": ["and but"], "so/because": ["so because"],
      "feeling": ["happy sad", "good bad"], "size": ["big small", "big little"], "in/on": ["in on"], "up/down": ["up down"],
      "said/asked": ["said asked"], "who": ["girl boy", "mom dad", "Mom Dad", "mommy daddy"], "a/the": ["a the"], "a/an": ["a an"],
      "day/night": ["day night"], "to/for": ["to for"], "his/her": ["his her"], "one/two": ["one two"]}
SWAP = collections.defaultdict(list)
for fam, prs in BI.items():
    for pr in prs:
        a, b = pr.split(); SWAP[a].append((fam, b))
        if not (fam == "his/her") and not (fam == "pronoun number" and a in ("he", "she")): SWAP[b].append((fam, a))


def main(valid, fresh, out):
    raw = open(valid, "rb").read(); vset = set(p.strip() for p in raw.split(b"<|endoftext|>") if p.strip()); del raw
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
                if t in SWAP and prev is not None and p > len(PRE) + 20 and e < len(w) and text[p - 1] in ' \n"':
                    for fam, alt in SWAP[t]:
                        if p + len(alt) + 1 > 256: continue
                        rows.append({"fam": fam, "prev": prev, "form": t, "p": p, "true": w[:e + 1], "alt": w[:p] + alt.encode() + w[e:e + 1],
                                     "nt": len(t) + 1, "na": len(alt) + 1})
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

    rs = np.random.default_rng(11); allfit = occurrences(0, 12000); per = collections.Counter()
    fit = collections.defaultdict(lambda: collections.defaultdict(list))
    for i in rs.permutation(len(allfit)):
        r = allfit[i]; k = (r["fam"], r["prev"], r["form"])
        if per[k] >= CAP or len(fit[r["fam"]][r["form"]]) >= MAXFORM: continue
        per[k] += 1; fit[r["fam"]][r["form"]].append(r)
    del allfit
    allev = occurrences(12000, 16000); byfam = collections.defaultdict(list)
    for r in allev: byfam[r["fam"]].append(r)
    ev = []
    for fam, rr in byfam.items(): ev += [rr[i] for i in rs.permutation(len(rr))[:MAXREAD]]
    bw = [w for w in ((PRE + kept[i].encode("ascii"))[:256] for i in range(20000, len(kept))) if len(w) == 256]
    key = lambda r: (r["fam"], r["form"], r["prev"]); pairs = sorted({key(r) for r in ev}); pid = {p: i for i, p in enumerate(pairs)}
    fams = sorted(BI); ff = sorted({(p[0], p[1]) for p in pairs})

    def by_pair(sh):
        s, n = np.zeros(len(pairs)), np.zeros(len(pairs))
        for x, r in zip(sh, ev): s[pid[key(r)]] += x; n[pid[key(r)]] += 1
        return s / n
    fam_idx = {f: [i for i, p in enumerate(pairs) if p[0] == f] for f in fams}
    ff_idx = {q: [i for i, p in enumerate(pairs) if (p[0], p[1]) == q] for q in ff}
    base_bp = by_pair(shares(sd0, ev)); base_bpb = bpb(sd0, bw)
    base_ff = {f"{a}: {b}": {"share": float(base_bp[ix].mean()), "pairs": len(ix), "fit_items": len(fit[a][b])} for (a, b), ix in ff_idx.items()}
    print("read items", len(ev), "pairs", len(pairs), "| base bits per byte", round(base_bpb, 4))
    print("base share per family: form share(pairs)")
    for f in fams:
        print("  ", f.ljust(15), round(float(base_bp[fam_idx[f]].mean()), 3), "|",
              "  ".join(f"{b} {base_bp[ix].mean():.2f}({len(ix)})" for (a, b), ix in ff_idx.items() if a == f), flush=True)
    sh = lambda f: f[:7].rjust(7)
    print("swipe: raise the ROW family for 10 steps; change in each COLUMN family's share (per distinct pair); bits per byte")
    print(" " * 15 + " ".join(sh(f) for f in fams) + "     bpb")
    rng = np.random.default_rng(3); measured = {}
    for a in fams:
        forms = [fit[a][f] for f in fit[a] if len(fit[a][f]) >= MINFORM]
        if not forms: continue
        th = th0.clone()
        for _ in range(NSTEP):
            gv = sum(grad_at(th, x) for x in forms) / len(forms); th = th + STEP * tn * gv / gv.norm()
        sd = unflat(th.float().to(E.DEV)); d = by_pair(shares(sd, ev)) - base_bp; b1 = bpb(sd, bw); ch = {}
        for g in fams:
            x = d[fam_idx[g]]; bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(300)]
            ch[g] = {"mean": float(x.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "pairs": len(x)}
        order = np.argsort(-np.abs(d))[:40]
        measured[a] = {"change_by_family": ch, "change_by_form": {f"{q[0]}: {q[1]}": float(d[ix].mean()) for q, ix in ff_idx.items()},
                       "bits_per_byte": b1, "fit_share_after": {x[0]["form"]: float(shares(sd, x).mean()) for x in forms},
                       "pair_change": {" / ".join(p): round(float(d[i]), 4) for i, p in enumerate(pairs) if abs(d[i]) >= 0.02},
                       "largest_moves": [[" / ".join(pairs[i]), round(float(base_bp[i]), 3), round(float(d[i]), 3)] for i in order]}
        print(a.ljust(15) + " ".join(f"{ch[g]['mean']:+7.2f}" for g in fams) + f"  {b1 - base_bpb:+.4f}", flush=True)
    json.dump({"status": "EXPLORATORY", "model": "GoldWorm v6 base", "device": str(E.DEV), "families": BI, "read_items": len(ev),
               "read_pairs": len(pairs), "base_bits_per_byte": base_bpb,
               "base_share_by_family": {f: float(base_bp[fam_idx[f]].mean()) for f in fams}, "base_share_by_form": base_ff,
               "base_share_by_pair": {" / ".join(p): round(float(base_bp[i]), 4) for i, p in enumerate(pairs)},
               "steps": NSTEP, "step_over_weights_norm": STEP, "measured": measured}, open(out, "w"))


main(*sys.argv[1:4])
