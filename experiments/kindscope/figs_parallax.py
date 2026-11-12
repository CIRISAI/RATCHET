"""Forest view of the EDIT-1 parallax exploration (CAL calibration stories only; exploratory, not a frozen test)."""
import json, math, sys, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figs_mlp0_1 import CAT, SURF, INK, INK2, GRID, frame  # shared style
r = json.load(open(sys.argv[1])); L = sorted(r["layers"], key=int); Ls = [int(x) for x in L]
deg = lambda c: math.degrees(math.acos(max(-1, min(1, c))))
fig = plt.figure(figsize=(12, 7.2))
gs = fig.add_gridspec(2, 6, height_ratios=[1.15, 1], hspace=0.55, wspace=0.55)
ax = fig.add_subplot(gs[0, :3])
for k, (key, lab, col) in enumerate((("cos_read_write", "reading (probe) ↔ writing (gradient)", CAT[0]),
                                     ("cos_ft_write", "fine-tune's change ↔ writing", CAT[1]),
                                     ("cos_ft_read", "fine-tune's change ↔ reading", CAT[2]))):
    v = [deg(r["layers"][l][key]) for l in L]
    ax.plot(Ls, v, color=col, lw=2, marker="o", ms=7, mec=SURF, mew=1.5, label=lab)
ax.axhline(90, color=INK2, lw=1, ls="--"); ax.text(5, 91, "orthogonal", color=INK2, fontsize=8, ha="right")
ax.set_ylim(70, 95); ax.set_xticks(Ls, [f"mlp{l}" for l in Ls]); ax.set_ylabel("angle between directions (°)")
ax.set_title("Parallax: where a category is readable is not where to push it", loc="left", fontsize=11)
ax.legend(loc="lower left", frameon=False, fontsize=8)
ax2 = fig.add_subplot(gs[0, 3:])
w = 0.26
for k, (name, lab, col) in enumerate((("read", "push along reading dir.", CAT[0]), ("write", "push along writing dir.", CAT[1]),
                                      ("ft", "push along fine-tune's change", CAT[2]))):
    best = [max(g for _, g in r["layers"][l][f"curve_{name}"]) for l in L]
    ax2.bar(np.array(Ls) + (k - 1) * w, best, width=w - 0.04, color=col, label=lab)
ax2.axhline(r["gainA2_cal_subset"], color=INK, lw=1.2, ls="--")
ax2.text(5.45, r["gainA2_cal_subset"] + 0.008, f"full fine-tune +{r['gainA2_cal_subset']:.2f}", ha="right", fontsize=8)
ax2.set_xticks(Ls, [f"mlp{l}" for l in Ls]); ax2.set_ylabel("best gain on adjective words (nats)")
ax2.set_title("One push at one layer buys at most a third of the fine-tune", loc="left", fontsize=11)
ax2.legend(loc="upper left", frameon=False, fontsize=8, bbox_to_anchor=(0, 0.93))
for i, l in enumerate(L):
    a = fig.add_subplot(gs[1, i])
    for name, col in (("read", CAT[0]), ("write", CAT[1]), ("ft", CAT[2])):
        xs, ys = zip(*r["layers"][l][f"curve_{name}"]); a.plot(xs, ys, color=col, lw=2, marker="o", ms=4)
    a.axhline(0, color=INK2, lw=0.8); a.set_xscale("log", base=2); a.set_xticks([0.25, 0.5, 1, 2], ["¼", "½", "1", "2"])
    a.set_ylim(-1.0, 0.2); a.set_title(f"mlp{l}", fontsize=9, loc="left")
    if i == 0: a.set_ylabel("gain (nats)")
    a.set_xlabel("push size (× write norm)", fontsize=7)
fig.text(0.01, 0.43, "Torque signature: each push helps a little, then breaks the model as it is driven harder (clipped at −1)",
         fontsize=10, color=INK)
frame(fig, "GoldWorm v6 base · additive push at positions before a word that follows a determiner · target: log-prob of "
           "adjective words · CAL calibration stories (training side) · EXPLORATORY, one fine-tune seed")
fig.savefig(sys.argv[2], dpi=150, bbox_inches="tight", facecolor=SURF)
