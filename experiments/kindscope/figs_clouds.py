"""Forest view: what organises the state cloud across depth, and how much of the writing direction is a byte forecast."""
import json, sys, os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figs_mlp0_1 import CAT, SURF, INK, INK2, GRID, frame
r = json.load(open(sys.argv[1])); Ls = list(range(6)); lay = [r["layers"][str(l)] for l in Ls]
fig = plt.figure(figsize=(14, 5.4)); gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1, 1.25], wspace=0.5)
ax = fig.add_subplot(gs[0])
for k, (key, lab) in enumerate((("pos_bucket", "position in the window"), ("det", "the determiner just read"),
                                ("next_first_byte", "next word's first byte"), ("next_pos", "next word's part of speech"))):
    ax.plot(Ls, [d["eta2_adjusted"][key] for d in lay], color=CAT[k], lw=2.4, marker="o", ms=7, mec=SURF, mew=1.5)
    ax.annotate(lab, (5, lay[5]["eta2_adjusted"][key]), xytext=(6, {"pos_bucket": 6, "next_first_byte": -6}.get(key, 0)),
                textcoords="offset points", fontsize=8,
                color=INK, va="center")
ax.set_xticks(Ls, [f"mlp{l}" for l in Ls]); ax.set_xlim(-0.3, 7.3); ax.set_ylim(0, 0.62)
ax.set_ylabel("share of state variance (η², null-adjusted)")
ax.set_title("What organises the cloud", loc="left", fontsize=10.5)
ax2 = fig.add_subplot(gs[1])
ax2.plot(Ls, [d["R2_byte_basis16"] for d in lay], color=CAT[1], lw=2.4, marker="o", ms=7, mec=SURF, mew=1.5,
         label="16 per-byte forecast directions")
ax2.plot(Ls, [d["R2_top2_bytes"] for d in lay], color=CAT[3], lw=2, marker="o", ms=6, label="only 'l' and 'b'")
ax2.plot(Ls, [d["R2_random16_null"] for d in lay], color=INK2, lw=1.4, ls="--", marker="o", ms=5,
         label="16 random directions (null)")
ax2.fill_between(Ls, [d["R2_byte_basis16"] for d in lay], 1, color=GRID, alpha=0.7)
ax2.text(0.1, 0.9, "residual: not a first-byte forecast", fontsize=8, color=INK2)
ax2.set_xticks(Ls, [f"mlp{l}" for l in Ls]); ax2.set_ylim(0, 1.02); ax2.legend(frameon=False, fontsize=8, loc="center right")
ax2.set_ylabel("share of adjective writing dir. explained")
ax2.set_title("Sharp late, residual early", loc="left", fontsize=10.5)
ax3 = fig.add_subplot(gs[2])
bytes_ = r["top_first_bytes"]; C = np.array([[d["coef"][b] for b in bytes_] for d in lay]).T
v = np.abs(C).max()
im = ax3.imshow(C, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("div", ["#1baf7a", "#f0efec", "#eb6834"]),
                vmin=-v, vmax=v, aspect="auto")
ax3.set_yticks(range(len(bytes_)), [f"'{b}'" for b in bytes_]); ax3.set_xticks(Ls, [f"mlp{l}" for l in Ls]); ax3.grid(False)
for i in range(C.shape[0]):
    for j in range(C.shape[1]):
        if abs(C[i, j]) > 0.15: ax3.text(j, i, f"{C[i,j]:+.2f}", ha="center", va="center", fontsize=7, color=INK)
cb = fig.colorbar(im, ax=ax3, pad=0.02); cb.set_label("coefficient (orange = push up, green = push down)")
ax3.set_title("The adjective direction spelled in next-byte forecasts", loc="left", fontsize=10.5)
frame(fig, "GoldWorm v6 base · states before a word that follows a determiner · CAL calibration stories (800) · "
           "EXPLORATORY · first bytes ordered by frequency")
fig.savefig(sys.argv[2], dpi=140, facecolor=SURF, bbox_inches="tight")
