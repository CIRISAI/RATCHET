"""Figures for MLP0-1 (descriptive; frame stated on each figure). Usage: python figs_mlp0_1.py mlp0_1_result.json figs/"""

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import json  # noqa: E402
import sys  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlp0_1 as M  # noqa: E402
from ks import engine as E  # noqa: E402
from ks.lang import POS  # noqa: E402

CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.axisbelow": True})


def frame(fig, text):
    fig.text(0.01, 0.005, text, fontsize=7.5, color=INK2, ha="left", va="bottom")


def fig_eta(res, out):
    m1 = res["M1"]
    ws = ["wte", "attn0", "mlp0", "attn1", "mlp1"]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    v = [m1[w]["eta2"] for w in ws]
    ax.plot(range(5), v, color=CAT[0], lw=2, marker="o", ms=8, mec=SURF, mew=2)
    for i, (w, x) in enumerate(zip(ws, v)):
        ax.annotate(f"{x:.2f}", (i, x), textcoords="offset points", xytext=(0, 9), ha="center", color=INK)
    ax.axhline(m1["mlp0"]["null_p95"], color=INK2, lw=1, ls="--")
    ax.text(4.1, m1["mlp0"]["null_p95"] + 0.02, "shuffled-form null (95th pct)", color=INK2, fontsize=8, ha="right")
    ax.set_xticks(range(5), ["byte embedding", "attn0 write", "mlp0 write", "attn1 write", "mlp1 write"])
    ax.set_ylim(0, 1.08); ax.set_ylabel("variance explained by word form (η²)")
    ax.set_title("Attention mixes context in; each MLP writes word identity back", loc="left", fontsize=11)
    frame(fig, f"GoldWorm v6 base · write at word's last byte · EXPLORE · {m1['n_tokens']} tokens of "
               f"{m1['n_forms']} forms (≥20 each) · MLP0-1 M1")
    fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(out, dpi=160); plt.close(fig)


def fig_patch(res, out):
    m2 = res["M2"]
    ws = ["wte", "attn0", "mlp0"]
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for j, (cond, lab, col) in enumerate((("same", "donor: same word, other story", CAT[0]),
                                          ("other", "donor: other word, same POS and length", CAT[1]))):
        for i, w in enumerate(ws):
            d = m2[f"{w}/{cond}"]
            y = i + (0.12 if j else -0.12)
            mean = max(d["mean"], 1e-7)
            lo, hi = max(d["ci95"][0], 1e-7), max(d["ci95"][1], 1e-7)
            ax.plot([lo, hi], [y, y], color=col, lw=2)
            ax.plot(mean, y, "o", color=col, ms=8, mec=SURF, mew=2, label=lab if i == 0 else None)
            ax.annotate("exactly 0" if d["mean"] == 0 else f"{d['mean']:.4f}", (mean, y), textcoords="offset points",
                        xytext=(8, -3), color=INK, fontsize=8)
    ax.set_xscale("log"); ax.set_xlim(5e-8, 2)
    ax.set_yticks(range(3), ["byte embedding (plant)", "attn0 write", "mlp0 write"]); ax.invert_yaxis()
    ax.set_xlabel("Δ next-8-byte loss after patching the write (nats, log scale; 0 drawn at 1e-7)")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.2), frameon=False, fontsize=8, ncol=2)
    ax.set_title("mlp0's write depends on the word, almost not on the story", loc="left", fontsize=11)
    frame(fig, f"GoldWorm v6 base · patch at word's last byte, full forward · EXPLORE · {m2['n_eligible']} words · "
               "95% story-bootstrap CI · MLP0-1 M2")
    fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(out, dpi=160); plt.close(fig)


def fig_scatter(out):
    """PCA (fit per panel) of attn0 and mlp0 writes for the 8 most frequent explore-split word forms."""
    cache = E.Cache(os.path.join(os.environ.get("KS_SCRATCH", "/tmp/claude-1000/-home-emoore-RATCHET/"
                                                "4fdbd195-6bf1-45c9-8ffc-931540da4e4d/scratchpad"), "ks_cache"))
    port = cache.port()
    ex = np.where(cache.split["explore"])[0]
    fe = cache.forms[ex]
    u, c = np.unique(fe, return_counts=True)
    top = [f for f in u[np.argsort(-c)] if f.isalpha()][:8]
    sel = ex[np.isin(fe, top)]
    by = {}
    for n in sel:
        by.setdefault(int(cache.story_of[n]), []).append(n)
    W = {"attn0": [], "mlp0": []}
    lab = []
    with torch.no_grad():
        for ss, idx in ((s, M.to_idx(cache, s)) for s in [sorted(by)[i:i + 64] for i in range(0, len(by), 64)]):
            _, writes = M.forward(port, idx)
            for r, s in enumerate(ss):
                for n in by[s]:
                    for w in W:
                        W[w].append(writes[w][r, int(cache.b[n])].cpu().numpy())
                    lab.append(cache.forms[n])
    lab = np.array(lab)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.6))
    for ax, w in zip(axs, ("attn0", "mlp0")):
        X = np.array(W[w]); X = X - X.mean(0)
        _, S, Vt = np.linalg.svd(X, full_matrices=False)
        P = X @ Vt[:2].T
        for k, f in enumerate(top):
            m = lab == f
            ax.scatter(P[m, 0], P[m, 1], s=9, color=CAT[k], alpha=0.55, linewidths=0)
            cx, cy = np.median(P[m, 0]), np.median(P[m, 1])
            ax.annotate(f, (cx, cy), color=INK, fontsize=9, fontweight="bold", ha="center",
                        bbox=dict(boxstyle="round,pad=0.15", fc=SURF, ec=CAT[k], lw=1))
        ev = S[:2] ** 2 / (S ** 2).sum()
        ax.set_title(f"{w} write  (PC1 {ev[0]:.0%}, PC2 {ev[1]:.0%} of variance)", loc="left", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("The same word, many stories: attn0 smears it, mlp0 snaps it to one point", x=0.01, ha="left",
                 fontsize=11)
    frame(fig, f"GoldWorm v6 base · write at the word's last byte · EXPLORE · 8 most frequent alphabetic forms, "
               f"{len(lab)} occurrences · PCA fit separately per panel (axes not comparable across panels)")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95)); fig.savefig(out, dpi=160); plt.close(fig)


def fig_pos(res, out):
    m3 = res["M3"]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    series = (("attn0_probe_on_attn0", "attn0 probe, read at attn0", CAT[0]),
              ("attn0_probe_on_mlp0", "same attn0 probe, read after mlp0", CAT[1]),
              ("native_mlp0", "probe trained after mlp0", CAT[2]))
    for j, (k, labl, col) in enumerate(series):
        xs = np.arange(len(POS)) + (j - 1) * 0.22
        ax.scatter(xs, [m3[c][k] for c in POS], color=col, s=40, edgecolor=SURF, linewidth=1.5, label=labl, zorder=3)
    ax.scatter(np.arange(len(POS)), [m3[c]["base_rate"] for c in POS], marker="_", s=300, color=INK2, label="base rate",
               zorder=2)
    ax.set_xticks(range(len(POS)), POS); ax.set_ylim(0.45, 1.02); ax.set_ylabel("one-vs-rest accuracy")
    ax.legend(loc="lower left", frameon=False, fontsize=8, ncol=2)
    ax.set_title("After mlp0 part of speech is still readable, but not where attn0 kept it", loc="left", fontsize=11)
    frame(fig, "GoldWorm v6 base · word's last byte · probes trained on the train split, read on EXPLORE · MLP0-1 M3")
    fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(out, dpi=160); plt.close(fig)


if __name__ == "__main__":
    res = json.load(open(sys.argv[1]))
    od = sys.argv[2]
    os.makedirs(od, exist_ok=True)
    fig_eta(res, os.path.join(od, "mlp0_1_eta.png"))
    fig_patch(res, os.path.join(od, "mlp0_1_patch.png"))
    fig_pos(res, os.path.join(od, "mlp0_1_pos.png"))
    fig_scatter(os.path.join(od, "mlp0_1_scatter.png"))
    print("wrote 4 figures to", od)
