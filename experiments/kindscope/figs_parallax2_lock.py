import json, sys, os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figs_mlp0_1 import CAT, SURF, INK, INK2, GRID, frame
r = json.load(open(sys.argv[1]))
fig, axs = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.2, 1], "wspace": 0.3})
ax = axs[0]; xs = np.arange(3); w = 0.26
for k, x in enumerate(("b", "l", "other")):
    for j, L in enumerate(("0", "1", "2")):
        c = r["P1"][L]["cells"][x]
        ax.bar(j + (k - 1) * w, c["mean"], width=w - 0.04, color=CAT[k], label=f"first letter {x}" if j == 0 else None)
        ax.plot([j + (k - 1) * w] * 2, c["ci95"], color=INK, lw=1.2)
ax.axhline(0, color=INK2, lw=0.8); ax.set_xticks(xs, ["mlp0", "mlp1", "mlp2"]); ax.legend(frameon=False, fontsize=8)
ax.set_ylabel("adjective gain − noun gain (nats)")
ax.set_title("Early: the residual writes the category — adjectives over nouns\nwithin every first-letter group (9/9)",
             loc="left", fontsize=10.5)
ax2 = axs[1]; gm = r["P2"]["group_means"]; order = ["NOUN_l", "ADJ_l", "NOUN_other", "ADJ_other", "NOUN_b", "ADJ_b"]
cols = [CAT[1] if g.endswith("_l") else (CAT[0] if g.endswith("_b") else CAT[2]) for g in order]
ax2.barh(range(len(order)), [gm[g] for g in order], color=cols)
ax2.set_yticks(range(len(order)), [g.replace("_", " · first letter ") for g in order]); ax2.invert_yaxis()
ax2.axvline(0, color=INK2, lw=0.8); ax2.set_xlabel("Δ log-prob of the next word (nats)")
ax2.set_title("Late: the 'l' direction writes a byte — l-words up\nwhatever their class, everything else down",
              loc="left", fontsize=10.5)
fig.suptitle("PARALLAX-2, locked on unseen text: category early, bytes late", x=0.01, y=1.04, ha="left", fontsize=12)
fig.subplots_adjust(bottom=0.2)
frame(fig, f"GoldWorm v6 base · FRESH-1 (TinyStories-train slice, never seen by the model; {r['counts']['fresh_stories_kept']} "
           f"stories, {r['counts']['pairs']} determiner–word pairs) · directions fixed from CAL before freezing · 95% story-bootstrap CIs")
fig.savefig(sys.argv[2], dpi=140, facecolor=SURF, bbox_inches="tight")
