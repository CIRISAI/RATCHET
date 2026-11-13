"""Harvest: let the model's own behaviour name the word choices, and a cheap test for which kind each is
(GoldWorm v6 base; EXPLORATORY).

PART 1, opposition (a heuristic for "trade" against "moves as a body", no steps taken): for every two-way swap in
swipes.py, the cosine between the weight gradient that raises word a where the text has a and the gradient that raises
word b where the text has b. Near -1: the two are one lever pulled opposite ways. Also the size of the common part
against the trade part, and which blocks hold each (the weight-side map).

PART 2, harvest (deterministic, no hand-picked families): at every word start in 600 unread stories the model's own
strongest OTHER word is generated (top two first bytes, each completed greedily to a word end), and the share the model
gives the text's word against it is recorded. Word pairs are ranked by how often they are contested (share 0.1-0.9).
Usage: python harvest.py <valid> <fresh> <out.json>"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import collections, json, math, re, sys
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import edit1 as X
from ks import engine as E
PRE = b"<|endoftext|>\n\n"; WORD = re.compile(r"[A-Za-z']+")
CAP, MAXFORM, MINFORM = 5, 300, 15
BI = {"agreement": ["was were", "is are", "has have"], "tense": ["is was", "are were", "can could", "will would"],
      "gender": ["he she", "He She"], "pronoun number": ["it they", "It They"],
      "negation": ["did didn't", "could couldn't", "was wasn't", "can can't", "do don't"],
      "this/these": ["this these"], "this/that": ["this that"], "and/but": ["and but"], "so/because": ["so because"],
      "feeling": ["happy sad", "good bad"], "size": ["big small", "big little"], "in/on": ["in on"], "up/down": ["up down"],
      "said/asked": ["said asked"], "who": ["girl boy", "mom dad", "Mom Dad", "mommy daddy"], "a/the": ["a the"], "a/an": ["a an"],
      "day/night": ["day night"], "to/for": ["to for"], "one/two": ["one two"]}
SWAP = collections.defaultdict(list)
for fam, prs in BI.items():
    for pr in prs:
        a, b = pr.split(); SWAP[a].append((fam, b)); SWAP[b].append((fam, a))
LETTER = set(range(65, 91)) | set(range(97, 123)) | {39}


def main(valid, fresh, out):
    raw = open(valid, "rb").read(); vset = set(p.strip() for p in raw.split(b"<|endoftext|>") if p.strip()); del raw
    parts = open(fresh, "rb").read().split(b"<|endoftext|>")[1:-1]
    kept = [p.strip().decode("ascii") for p in parts if p.strip() and all(c < 128 for c in p.strip()) and p.strip() not in vset]
    cache = E.Cache(os.path.join(X.SCRATCH, "ks_cache")); sd0, _ = X.load_params(cache.info["model_path"])
    keys = list(sd0.keys()); sizes = [sd0[k].numel() for k in keys]; P = sum(sizes)
    block = []
    for k, n in zip(keys, sizes):
        m = re.match(r"blocks\.(\d)\.(attn|mlp)\.", k)
        block += [(f"{m.group(2)}{m.group(1)}" if m else ("norms" if ".ln" in k or k.startswith("ln_f") else k.split(".")[0]))] * 1
    bnames = sorted(set(block))

    def by_block(v):
        o, i = collections.defaultdict(float), 0
        for b, n in zip(block, sizes): o[b] += float(v[i:i + n] @ v[i:i + n]); i += n
        t = sum(o.values()); return {b: round(o[b] / t, 3) for b in bnames}

    def score(lp, p, seq, n):
        ps = torch.arange(p - 1, p - 1 + n, device=E.DEV)
        return lp[ps, torch.tensor(list(seq[p:p + n]), device=E.DEV)].sum()

    def logodds(sd, ch):
        idx = torch.zeros(2 * len(ch), 256, dtype=torch.long, device=E.DEV)
        for i, r in enumerate(ch):
            idx[2 * i, :len(r["true"])] = torch.tensor(list(r["true"])); idx[2 * i + 1, :len(r["alt"])] = torch.tensor(list(r["alt"]))
        lp = X.fwd(sd, idx)[0].log_softmax(-1)
        return torch.stack([score(lp[2 * i], r["p"], r["true"], r["nt"]) - score(lp[2 * i + 1], r["p"], r["alt"], r["na"]) for i, r in enumerate(ch)])

    def grad(rows):
        sd = {k: v.clone().requires_grad_(True) for k, v in sd0.items()}; acc = torch.zeros(P, dtype=torch.float64)
        for b in range(0, len(rows), 96):
            g = torch.autograd.grad(F.logsigmoid(logodds(sd, rows[b:b + 96])).sum(), list(sd.values()))
            acc += torch.cat([x.reshape(-1) for x in g]).double().cpu()
        return acc / len(rows)

    # ---------------- PART 1
    rs = np.random.default_rng(11); rows = []
    for si in range(0, 12000):
        w = (PRE + kept[si].encode("ascii"))[:256]; text = w.decode("ascii"); prev = None
        for m in WORD.finditer(text):
            t = m.group(0); p, e = m.start(), m.end()
            if t in SWAP and prev is not None and p > len(PRE) + 20 and e < len(w) and text[p - 1] in ' \n"':
                for fam, alt in SWAP[t]:
                    if p + len(alt) + 1 <= 256:
                        rows.append({"k": (fam, t, alt), "prev": prev, "p": p, "true": w[:e + 1], "alt": w[:p] + alt.encode() + w[e:e + 1], "nt": len(t) + 1, "na": len(alt) + 1})
            prev = t.lower()
    per = collections.Counter(); fit = collections.defaultdict(list)
    for i in rs.permutation(len(rows)):
        r = rows[i]
        if per[(r["k"], r["prev"])] >= CAP or len(fit[r["k"]]) >= MAXFORM: continue
        per[(r["k"], r["prev"])] += 1; fit[r["k"]].append(r)
    del rows
    opposition = {}
    print("PART 1  swap: cosine of the two words' gradients | common part / trade part (norms) | where the common part sits | where the trade part sits")
    for fam, prs in BI.items():
        for pr in prs:
            a, b = pr.split(); A, B = fit[(fam, a, b)], fit[(fam, b, a)]
            if len(A) < MINFORM or len(B) < MINFORM: continue
            ga, gb = grad(A), grad(B); cos = float(ga @ gb / (ga.norm() * gb.norm())); com, tr = (ga + gb) / 2, (ga - gb) / 2
            top = lambda d: " ".join(f"{k}:{v:.2f}" for k, v in sorted(d.items(), key=lambda kv: -kv[1])[:3])
            opposition[f"{fam}: {a}/{b}"] = {"cosine": cos, "items": [len(A), len(B)], "common_over_trade_norm": float(com.norm() / tr.norm()),
                                             "common_by_block": by_block(com), "trade_by_block": by_block(tr)}
            print(f"  {fam}: {a}/{b}".ljust(30), f"{cos:+.2f}", f"| {float(com.norm() / tr.norm()):.2f}", "|", top(by_block(com)), "|", top(by_block(tr)), flush=True)

    # ---------------- PART 2
    torch.set_grad_enabled(False)

    def last_lp(seqs):
        out = []
        for b in range(0, len(seqs), 512):
            ch = seqs[b:b + 512]; L = max(len(s) for s in ch); idx = torch.zeros(len(ch), L, dtype=torch.long, device=E.DEV)
            for i, s in enumerate(ch): idx[i, :len(s)] = torch.tensor(list(s))
            lg = X.fwd(sd0, idx)[0]
            out.append(lg[torch.arange(len(ch)), torch.tensor([len(s) - 1 for s in ch], device=E.DEV)].log_softmax(-1).double().cpu())
        return torch.cat(out)
    letters = torch.tensor(sorted(LETTER)); cands = []; n_words = 0
    for si in range(16000, 16600):
        w = (PRE + kept[si].encode("ascii"))[:256]; text = w.decode("ascii")
        lp = X.fwd(sd0, torch.tensor(list(w), dtype=torch.long, device=E.DEV)[None])[0][0].log_softmax(-1).double().cpu()
        for m in WORD.finditer(text):
            t = m.group(0); p, e = m.start(), m.end()
            if p <= len(PRE) + 20 or e >= len(w) or e + 14 > 255 or text[p - 1] not in ' \n"': continue
            n_words += 1; lt = float(sum(lp[p - 1 + j, w[p + j]] for j in range(len(t) + 1)))
            for fb in letters[lp[p - 1, letters].topk(2).indices].tolist():
                cands.append({"word": t, "lt": lt, "ctx": w[:p], "alt": bytes([fb]), "la": float(lp[p - 1, fb]), "done": False, "first": True})
    for _ in range(14):
        live = [c for c in cands if not c["done"]]
        if not live: break
        L = last_lp([c["ctx"] + c["alt"] for c in live])
        for c, l in zip(live, L):
            nb = int(l.argmax()); c["la"] += float(l[nb])
            if nb in LETTER and len(c["alt"]) < 13: c["alt"] += bytes([nb])
            else: c["done"] = True
    best = {}
    for i, c in enumerate(cands):
        if not c["done"]: continue
        a = c["alt"].decode("ascii")
        if a == c["word"]: continue
        key = i // 2
        if key not in best or c["la"] > best[key]["la"]: best[key] = c
    table = collections.defaultdict(lambda: {"n": 0, "contested": 0, "share": []})
    for c in best.values():
        sh = 1 / (1 + math.exp(min(50.0, c["la"] - c["lt"]))); k = (c["word"], c["alt"].decode("ascii")); t = table[k]
        t["n"] += 1; t["contested"] += 0.1 < sh < 0.9; t["share"].append(sh)
    un = collections.defaultdict(lambda: {"contested": 0, "n": 0, "dirs": {}})
    for (a, b), t in table.items():
        u = un[tuple(sorted((a, b), key=str.lower))]; u["contested"] += t["contested"]; u["n"] += t["n"]
        u["dirs"][f"text {a}, model's other {b}"] = {"n": t["n"], "mean_share_for_text": round(float(np.mean(t["share"])), 3)}
    ranked = sorted(un.items(), key=lambda kv: -kv[1]["contested"]); tot_c = sum(u["contested"] for u in un.values())
    hand = {tuple(sorted(pr.split(), key=str.lower)) for prs in BI.values() for pr in prs}
    rank_of = {k: i + 1 for i, (k, _) in enumerate(ranked)}
    print(f"PART 2  word starts {n_words}, with another word found {len(best)}, contested (share 0.1-0.9) {tot_c}; distinct word pairs {len(un)}; "
          f"top 50 pairs hold {sum(u['contested'] for _, u in ranked[:50]) / tot_c:.2f} of contested, top 200 {sum(u['contested'] for _, u in ranked[:200]) / tot_c:.2f}")
    print("  hand-picked pairs' rank in the harvest:", {"/".join(k): rank_of.get(k) for k in sorted(hand)})
    print("  top 60 harvested pairs: contested count (both directions) | each direction: n, mean share for the text's word")
    for k, u in ranked[:60]:
        print("   ", "/".join(k).ljust(22), str(u["contested"]).rjust(4), "|", " ; ".join(f"{d}: n {v['n']} share {v['mean_share_for_text']}" for d, v in u["dirs"].items()), flush=True)
    json.dump({"status": "EXPLORATORY", "model": "GoldWorm v6 base", "device": str(E.DEV), "opposition": opposition,
               "harvest": {"stories": [16000, 16600], "word_starts": n_words, "with_other_word": len(best), "contested": int(tot_c), "distinct_pairs": len(un),
                           "hand_pair_rank": {"/".join(k): rank_of.get(k) for k in sorted(hand)},
                           "pairs": [{"pair": "/".join(k), "contested": int(u["contested"]), "n": u["n"], "directions": u["dirs"]} for k, u in ranked[:1000]]}}, open(out, "w"))


main(*sys.argv[1:4])
