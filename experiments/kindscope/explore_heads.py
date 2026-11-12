"""Exploration (the 8 trace items, unseen stories): what each attention head's single strongest source is.
Not a test: top-1 only; categories overlap (the previous character can be the word start); attention weight says where
a head reads from, not how much that matters (KS-DIRECT-ONLY / KS-PRICE-NOT-COST)."""
import json, numpy as np
P0 = len(b"<|endoftext|>\n\n")
agg = {}
for k in range(8):
    b = json.load(open(f"traces/gw_item{k}.readings.json"))["bytes"]
    att = json.load(open(f"traces/gw_item{k}.attention.json"))["layers"]
    ws, last = [], 0
    for p, c in enumerate(b):
        if c in (32, 10): last = p + 1
        ws.append(last)
    for L in range(6):
        for h in range(4):
            a = agg.setdefault(f"attn{L}/h{h}", {"self": 0, "prev": 0, "word_start": 0, "space_before_word": 0, "prefix": 0, "n": 0, "w": 0.0})
            for p in range(P0, len(b)):
                j, wt = att[f"attn{L}"]["per_head_top1"][p][h]
                a["n"] += 1; a["w"] += wt
                a["self"] += j == p; a["prev"] += j == p - 1; a["prefix"] += j < P0
                a["word_start"] += (j == ws[p] and j != p); a["space_before_word"] += (j == ws[p] - 1 and j != p)
out = {k: {**{m: round(v[m] / v["n"], 3) for m in ("self", "prev", "word_start", "space_before_word", "prefix")},
           "mean_top1_weight": round(v["w"] / v["n"], 3), "positions": v["n"]} for k, v in agg.items()}
names = {"attn0/h3": "previous character (90%)", "attn2/h3": "first letter of the current word (39%, weight 0.68)",
         "attn3/h1": "the space just before the current word (61%)", "attn1/h2": "the story's opening prefix (100%, diffuse weight 0.07: a resting place)"}
json.dump({"readme": __doc__, "heads": out, "nameable_exploratory": names,
           "suspected_not_tested": "attn0/h3 (previous character) feeds mlp0's lookup keyed on a word's final bytes (MLP0-4); test: ablate attn0/h3 and measure mlp0's suffix lookup"},
          open("explore_heads.json", "w"), indent=1)
print(json.dumps(names, indent=1))
