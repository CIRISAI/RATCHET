"""The parallax plane per layer, each axis scaled to its own spread (z-scores), with per-class marginals."""
import json, sys, os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figs_mlp0_1 import CAT, SURF, INK, INK2, GRID, frame
r = json.load(open(sys.argv[1]))
GRAY = "#a9a8a2"
fig = plt.figure(figsize=(14, 9))
outer = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.28)
for k, L in enumerate(sorted(r["layers"], key=int)):
    p = r["layers"][L]["plane"]; xy = np.array(p["xy"]); y = np.array(p["y"])
    z = (xy - xy.mean(0)) / xy.std(0)
    g = outer[k // 3, k % 3].subgridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4], hspace=0.05, wspace=0.05)
    ax = fig.add_subplot(g[1, 0]); hx = fig.add_subplot(g[0, 0], sharex=ax); hy = fig.add_subplot(g[1, 1], sharey=ax)
    for cls, col in ((0, GRAY), (1, CAT[1])):
        m = y == cls
        ax.scatter(z[m, 0], z[m, 1], s=5, color=col, alpha=0.45, linewidths=0)
        hx.hist(z[m, 0], bins=40, range=(-4, 4), density=True, histtype="step", color=col, lw=1.6)
        hy.hist(z[m, 1], bins=40, range=(-4, 4), density=True, histtype="step", color=col, lw=1.6, orientation="horizontal")
    def sep(v):
        a, b = v[y == 1], v[y == 0]; return (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)
    ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
    ax.set_xlabel(f"probe axis (z) · class separation d′={sep(z[:,0]):.2f}", fontsize=8)
    ax.set_ylabel(f"writing axis ⊥ probe (z) · d′={sep(z[:,1]):.2f}", fontsize=8)
    for a in (hx, hy): a.axis("off")
    an = r["layers"][L]["angles"]; vf = r["layers"][L]["var_frac_along"]
    hx.set_title(f"after mlp{L} · angle {an['probe-write']:.0f}° · variance: probe {100*vf['probe']:.1f}%, "
                 f"write {100*vf['write']:.1f}%", loc="left", fontsize=9)
fig.suptitle("Both axes separate the classes; the probe reads a near-silent direction, the model listens on a "
             "louder one", x=0.01, ha="left", fontsize=12)
fig.text(0.01, 0.955, "orange = next word is an adjective · gray = another word · d′ = mean gap / pooled spread",
         fontsize=9, color=INK2)
frame(fig, "GoldWorm v6 base · states before a word that follows a determiner · CAL calibration stories · axes "
           "z-scored separately · writing axis orthogonalised to the probe · EXPLORATORY")
fig.savefig(sys.argv[2], dpi=140, facecolor=SURF, bbox_inches="tight")
