"""Forest view of what the two parallax axes are about, across depth (CAL exploration)."""
import json, sys, os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figs_mlp0_1 import CAT, SURF, INK, INK2, GRID, frame
r = json.load(open(sys.argv[1])); r2 = json.load(open(sys.argv[2]))
L = list(range(6)); lay = [r["layers"][str(l)] for l in L]
fig = plt.figure(figsize=(13.5, 5.6)); gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 1], wspace=0.35)
ax = fig.add_subplot(gs[0])
ax.plot(L, [d["corr_probe_write_coords"] for d in lay], color=CAT[0], lw=2.4, marker="o", ms=7, mec=SURF, mew=1.5,
        label="agreement: correlation of the two\ncoordinates across contexts")
ax.plot(L, [d["cos_write_byteforecast"] for d in lay], color=CAT[1], lw=2.4, marker="o", ms=7, mec=SURF, mew=1.5,
        label="writing axis ↔ forecast of the\nadjectives' first bytes (cosine)")
ax.plot(L, [d["cos_probe_byteforecast"] for d in lay], color=CAT[2], lw=2.4, marker="o", ms=7, mec=SURF, mew=1.5,
        label="probe axis ↔ the same forecast")
ax.plot(L, [np.cos(np.radians(r2["layers"][str(l)]["angles"]["probe-write"])) for l in L], color=INK2, lw=1.5, ls="--",
        marker="o", ms=5, label="geometry: cosine of the angle\nprobe ↔ writing direction")
ax.axhline(0, color=INK2, lw=0.8); ax.set_ylim(-0.25, 0.8)
ax.set_xticks(L, [f"mlp{l}" for l in L]); ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.set_title("The views converge in what they rank, not in geometry", loc="left", fontsize=11)
for k, l in enumerate((0, 5)):
    a = fig.add_subplot(gs[1 + k]); words = r["layers"][str(l)]["words"]
    top_w = sorted(words, key=lambda d: -d["z_write"])[:9]; top_p = sorted(words, key=lambda d: -d["z_probe"])[:9]
    for j, (lst, key, x) in enumerate(((top_p, "z_probe", 0), (top_w, "z_write", 1))):
        for i, d in enumerate(lst):
            a.text(x, -i, d["form"], ha="center", va="center", fontsize=10,
                   color=CAT[1] if d["adj"] else INK2, fontweight="bold" if d["adj"] else "normal")
    a.set_xlim(-0.6, 1.6); a.set_ylim(-9, 1); a.axis("off")
    a.text(0, 0.8, "top of probe axis", ha="center", fontsize=9, color=INK); a.text(1, 0.8, "top of writing axis",
                                                                                  ha="center", fontsize=9, color=INK)
    a.set_title(f"after mlp{l}: next words ranked", loc="left", fontsize=11)
fig.text(0.40, 0.035, "orange bold = the next word is an adjective · gray = it is not", fontsize=9, color=INK2)
fig.suptitle("Parallax: the probe sees a category; the model writes a forecast of what comes next", x=0.01, ha="left",
             fontsize=12)
frame(fig, "GoldWorm v6 base · states before a word that follows a determiner · CAL calibration stories (800) · forms with "
           "≥15 occurrences · EXPLORATORY")
fig.savefig(sys.argv[3], dpi=140, facecolor=SURF, bbox_inches="tight")
