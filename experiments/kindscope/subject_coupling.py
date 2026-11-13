"""Coupling table regrouped by WHERE THE SUBJECT SITS (GoldWorm v6 base; EXPLORATORY).

Each was/were occurrence is placed by one dependency parser (spaCy en_core_web_sm; a single parser's reading, with
its errors) into:
  adjacent        the subject's head word is the word just before the verb                ("The kids were")
  back            the subject's head is further back                                       ("The toys in the box were")
  back attractor  as back, and the word just before the verb is a noun of the OTHER number ("One of the animals was")
  coordinated     the subject is a coordination                                            ("Tom and Jane were")
  relative        the subject is that / who / which                                        ("the dog that was")
  shared          the verb is a second conjunct and takes the first verb's subject         ("She ran and was")
  there           expletive there; the noun the verb agrees with comes after               ("There were birds")
  after           the subject follows the verb (questions)
  none            no subject found by the parser
crossed with the form the text has. is / are / has / have are read as four lumped columns to show off-block movement.
For each was/were group with >= 30 fit items (stories 0-11,999; <= 5 per distinct (previous word, form); <= 600):
first-order cosines of the groups' gradients, then 10 re-taken steps of 0.0002 x |weights| raising THAT group alone and
the per-distinct-pair change of EVERY group on stories 12,000-15,999, with bits per byte on stories 20,000 on.
Usage: python subject_coupling.py <valid> <fresh> <out.json>"""
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
SUBJ = ("nsubj", "nsubjpass", "csubj", "csubjpass")

def place(tok):
    head = tok.head if tok.dep_ in ("aux", "auxpass") else tok
    kids = list(head.children)
    if any(c.dep_ == "expl" for c in kids): return "there"
    subs = [c for c in kids if c.dep_ in SUBJ]; shared = False
    if not subs and head.dep_ == "conj":
        subs = [c for c in head.head.children if c.dep_ in SUBJ]; shared = True
    if not subs: return "none"
    s = subs[0]
    if s.i > tok.i: return "after"
    if shared: return "shared"
    if s.lower_ in ("that", "who", "which"): return "relative"
    if any(c.dep_ == "conj" for c in s.children): return "coordinated"
    if s.i == tok.i - 1: return "adjacent"
    p = tok.doc[tok.i - 1]; plural_form = tok.lower_ in ("were", "are", "have")
    if (p.tag_ in ("NNS", "NNPS") and not plural_form) or (p.tag_ in ("NN", "NNP") and plural_form): return "back attractor"
    return "back"

def main(valid, fresh, out):
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
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
        rows = []; body = [kept[si][:256 - len(PRE)] for si in range(lo, hi)]
        for si, doc in zip(range(lo, hi), nlp.pipe(body, batch_size=256)):
            w = (PRE + kept[si].encode("ascii"))[:256]; text = w.decode("ascii"); prev = None; at = {t.idx + len(PRE): t for t in doc}
            for m in WORD.finditer(text):
                t = m.group(0); p, e = m.start(), m.end()
                if t in PARTNER and prev is not None and p > len(PRE) + 20 and e + 3 < len(w) and text[p - 1] == " " and p in at:
                    alt = PARTNER[t]
                    g = f"{place(at[p])} | {t}" if t in ("was", "were") else t
                    rows.append({"prev": prev, "form": t, "p": p, "true": w[:e + 1], "alt": w[:p] + alt.encode() + w[e:e + 1],
                                 "nt": len(t) + 1, "na": len(alt) + 1, "group": g, "snip": text[max(len(PRE), p - 60):e]})
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
    per = collections.Counter(); fit = collections.defaultdict(list); allfit = occurrences(0, 12000)
    for i in np.random.default_rng(11).permutation(len(allfit)):
        r = allfit[i]; g = r["group"]
        if per[(g, r["prev"], r["form"])] >= CAP or len(fit[g]) >= MAXFIT: continue
        per[(g, r["prev"], r["form"])] += 1; fit[g].append(r)
    total = collections.Counter(r["group"] for r in allfit)
    ev = occurrences(12000, 16000); bw = [w for w in ((PRE + kept[i].encode("ascii"))[:256] for i in range(20000, len(kept))) if len(w) == 256]
    key = lambda r: (r["group"], r["prev"], r["form"])
    pairs = sorted({key(r) for r in ev}); pid = {p: i for i, p in enumerate(pairs)}
    npairs = collections.Counter(p[0] for p in pairs)
    cols = sorted(g for g in npairs if npairs[g] >= 3); drivers = [g for g in sorted(fit) if "|" in g and len(fit[g]) >= MINFIT and g in cols]
    def by_pair(sh):
        s, n = np.zeros(len(pairs)), np.zeros(len(pairs))
        for x, r in zip(sh, ev): s[pid[key(r)]] += x; n[pid[key(r)]] += 1
        return s / n
    base_bp = by_pair(shares(sd0, ev)); base_bpb = bpb(sd0, bw)
    base_g = {g: float(np.mean([base_bp[i] for i, p in enumerate(pairs) if p[0] == g])) for g in cols}
    print("group: occurrences in fit stories / fit items used / read pairs / base share")
    for g in cols: print("  ", g.ljust(24), total[g], len(fit[g]), npairs[g], round(base_g[g], 3), "| e.g.", " // ".join(r["snip"][-45:] for r in fit[g][:2]), flush=True)
    short = lambda g: g.replace("back attractor", "attr").replace("coordinated", "coord").replace("adjacent", "adj").replace("relative", "rel").replace(" | ", ":")[:10].rjust(10)
    G0 = {g: grad_at(th0, fit[g]) for g in drivers}
    cos = {a: {b: float(G0[a] @ G0[b] / (G0[a].norm() * G0[b].norm())) for b in drivers} for a in drivers}; del G0
    print("first-order cosines"); print(" " * 24 + " ".join(short(b) for b in drivers))
    for a in drivers: print(a.ljust(24) + " ".join(f"{cos[a][b]:+10.2f}" for b in drivers), flush=True)
    rng = np.random.default_rng(3); measured = {}
    print("measured: raise the ROW group for 10 steps; change in each COLUMN group's share (per distinct pair); bits per byte")
    print(" " * 24 + " ".join(short(b) for b in cols) + "      bpb")
    for a in drivers:
        th = th0.clone()
        for _ in range(NSTEP):
            gv = grad_at(th, fit[a]); th = th + STEP * tn * gv / gv.norm()
        sd = unflat(th.float().to(E.DEV)); d = by_pair(shares(sd, ev)) - base_bp; b1 = bpb(sd, bw); ch = {}
        for g in cols:
            x = np.array([d[i] for i, p in enumerate(pairs) if p[0] == g]); bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(500)]
            ch[g] = {"mean": float(x.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "pairs": len(x)}
        order = np.argsort(-np.abs(d))[:25]
        measured[a] = {"change_by_group": ch, "bits_per_byte": b1, "fit_items": len(fit[a]),
                       "pair_change": {" / ".join(p): round(float(d[i]), 4) for i, p in enumerate(pairs)},
                       "largest_moves": [[" / ".join(pairs[i]), round(float(base_bp[i]), 3), round(float(d[i]), 3)] for i in order]}
        print(a.ljust(24) + " ".join(f"{ch[g]['mean']:+10.3f}" for g in cols) + f"  {b1 - base_bpb:+.4f}", flush=True)
    json.dump({"status": "EXPLORATORY", "model": "GoldWorm v6 base", "device": str(E.DEV), "parser": "spaCy en_core_web_sm (single parser)",
               "columns": cols, "drivers": drivers, "occurrences_in_fit_stories": dict(total), "fit_items": {g: len(fit[g]) for g in fit},
               "read_pairs": dict(npairs), "base_share_by_group": base_g, "base_bits_per_byte": base_bpb,
               "base_share_by_pair": {" / ".join(p): round(float(base_bp[i]), 4) for i, p in enumerate(pairs)},
               "examples": {g: [r["snip"] for r in fit[g][:8]] for g in fit},
               "first_order_cosine": cos, "steps": NSTEP, "step_over_weights_norm": STEP, "measured": measured}, open(out, "w"))
main(*sys.argv[1:4])
