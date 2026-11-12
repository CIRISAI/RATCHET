"""Vocab against grammar, one picture (GoldWorm v6 base; EXPLORATORY).

Case: verb agreement. At each real occurrence of was/were, is/are, has/have in unread stories:
  grammar    : the form the text has (1.0). Assumes the story is grammatical (TinyStories; not checked by hand).
  vocab      : share of that form, against its partner, after the SAME previous word in the model's training text
               (train_core) -- what local word statistics alone give. Also the unconditional share.
  weights    : the model's share for that form against its partner, P(form+next byte) / (P(form+..)+P(partner+..)),
               each scored through the whole network from the real context.
If the weights only carried local word statistics the points lie on the diagonal; if they carry the rule they lie at 1.
Reported per bin of the vocab share, with distinct (previous word, form) pairs and a per-distinct-pair mean
(KS-ONE-WORD-PAIR). Real stories only: FRESH-1 kept stories 16,000 on.
Usage: python torque_agree.py <valid> <fresh> <out.json>"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import collections, json, math, re, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import edit1 as X, mlp0_1 as M
from ks import engine as E
PRE = b"<|endoftext|>\n\n"; WORD = re.compile(r"[A-Za-z']+")
PARTNER = {"was": "were", "were": "was", "is": "are", "are": "is", "has": "have", "have": "has"}
def main(valid, fresh, out):
    torch.set_grad_enabled(False)
    raw = open(valid, "rb").read()
    train = raw[:17502553].decode("ascii", "ignore")
    big = collections.Counter(); uni = collections.Counter(); prev = None
    for m in WORD.finditer(train):
        t = m.group(0)
        if t in PARTNER and prev is not None: big[(prev, t)] += 1; uni[t] += 1
        prev = t.lower()
    vset = set(p.strip() for p in raw.split(b"<|endoftext|>") if p.strip())
    parts = open(fresh, "rb").read().split(b"<|endoftext|>")[1:-1]
    kept = [p.strip().decode("ascii") for p in parts if p.strip() and all(c < 128 for c in p.strip()) and p.strip() not in vset]
    port = E.Cache(os.path.join(X.SCRATCH, "ks_cache")).port()
    def lps(b):
        return M.forward(port, torch.tensor(list(b), dtype=torch.long, device=E.DEV)[None])[0][0].log_softmax(-1).double()
    rows = []; hi_story = min(len(kept), 20000)
    for si in range(16000, hi_story):
        w = (PRE + kept[si].encode("ascii"))[:256]; text = w.decode("ascii"); lp = None; prev = None
        for m in WORD.finditer(text):
            t = m.group(0); p, e = m.start(), m.end()
            if t in PARTNER and prev is not None and p > len(PRE) + 20 and e + 3 < len(w) and text[p - 1] == " ":
                alt = PARTNER[t]
                if lp is None: lp = lps(w)
                lt = sum(float(lp[p - 1 + j, w[p + j]]) for j in range(len(t) + 1))
                wa = w[:p] + alt.encode() + w[e:e + 1]; la_ = lps(wa)
                la = sum(float(la_[p - 1 + j, wa[p + j]]) for j in range(len(alt) + 1))
                c, ca = big[(prev, t)], big[(prev, alt)]
                rows.append({"story": si, "prev": prev, "form": t, "partner": alt, "n_prev_form": c, "n_prev_partner": ca,
                             "vocab_share_local": (c + 0.5) / (c + ca + 1.0) if c + ca >= 5 else None,
                             "vocab_share_global": uni[t] / (uni[t] + uni[alt]),
                             "weights_share": 1.0 / (1.0 + math.exp(la - lt))})
            prev = t.lower()
    def summarise(sel):
        by = collections.defaultdict(list)
        for r in sel: by[(r["prev"], r["form"])].append(r)
        return {"occurrences": len(sel), "distinct_pairs": len(by),
                "vocab_share": float(np.mean([r["vocab_share_local"] for r in sel])) if sel and all(r["vocab_share_local"] is not None for r in sel) else None,
                "weights_share": float(np.mean([r["weights_share"] for r in sel])),
                "weights_share_per_distinct_pair": float(np.mean([np.mean([r["weights_share"] for r in v]) for v in by.values()])),
                "weights_pick_rate": float(np.mean([r["weights_share"] > 0.5 for r in sel])),
                "largest_pair": max(((k, len(v)) for k, v in by.items()), key=lambda kv: kv[1])}
    seen = [r for r in rows if r["vocab_share_local"] is not None]
    edges = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1.0001]; bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [r for r in seen if lo <= r["vocab_share_local"] < hi]
        if sel: bins.append({"vocab_share_range": [lo, min(hi, 1.0)], **summarise(sel)})
    unseen = summarise([r for r in rows if r["vocab_share_local"] is None]) if len(seen) < len(rows) else None
    json.dump({"status": "EXPLORATORY", "model": "GoldWorm v6 base", "device": str(E.DEV), "stories": [16000, hi_story],
               "n": len(rows), "unigram": dict(uni), "bins": bins, "previous_word_unseen_with_either_form": unseen,
               "by_form": {f: summarise([r for r in rows if r["form"] == f]) | {"vocab_share_global": uni[f] / (uni[f] + uni[PARTNER[f]])} for f in PARTNER},
               "rows": rows}, open(out, "w"))
    print("kept stories", len(kept), "occurrences", len(rows), "with local statistics", len(seen))
    for b in bins: print(b["vocab_share_range"], "n", b["occurrences"], "pairs", b["distinct_pairs"], "| vocab", round(b["vocab_share"], 3),
                         "| weights", round(b["weights_share"], 3), "per-pair", round(b["weights_share_per_distinct_pair"], 3),
                         "pick", round(b["weights_pick_rate"], 3), "| largest", b["largest_pair"])
    if unseen: print("unseen prev:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in unseen.items()})
    for f in PARTNER:
        s = summarise([r for r in rows if r["form"] == f]); print(f, "n", s["occurrences"], "global vocab", round(uni[f] / (uni[f] + uni[PARTNER[f]]), 3), "weights", round(s["weights_share"], 3))
main(*sys.argv[1:4])
