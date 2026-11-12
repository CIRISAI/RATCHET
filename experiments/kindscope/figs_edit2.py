"""EDIT-2 figures: the fine-tune shaping the model (progression), torque per tensor, withdrawal, parallax per layer.
Usage: python figs_edit2.py edit2_result.json figs/"""
import json, math, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figs_mlp0_1 import CAT, SURF, INK, INK2, GRID, frame  # noqa: E402

SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
FOOT = "GoldWorm v6 base · fine-tune toward adjectives after determiners (EDIT-1 recipe) · EXPLORE split · EDIT-2"


def fig_progression(r, out):
    seeds = sorted(r["progression"], key=int)
    p0 = r["progression"][seeds[0]]
    names = [f"{k}{i}" for i in range(6) for k in ("attn", "mlp")]
    steps = [p["step"] for p in p0]
    M = np.mean([[[p["write_change"][n] for n in names] for p in r["progression"][s]] for s in seeds], 0).T
    fig = plt.figure(figsize=(11, 6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.6], hspace=0.35)
    ax = fig.add_subplot(gs[0])
    for k, s in enumerate(seeds):
        ax.plot(steps, [p["T_adj_sub80"] for p in r["progression"][s]], color=CAT[k], lw=2, marker="o", ms=4,
                label=f"seed {s}")
    ax.set_ylabel("Δ log-prob of adjectives\nafter a determiner (nats)"); ax.legend(frameon=False, fontsize=8, ncol=3)
    ax.set_title("Watching the fine-tune shape the model: the behaviour, and where the writes change", loc="left",
                 fontsize=11)
    ax2 = fig.add_subplot(gs[1])
    im = ax2.imshow(M, aspect="auto", cmap=matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQ),
                    extent=(steps[0] - 25, steps[-1] + 25, len(names) - 0.5, -0.5))
    ax2.set_yticks(range(len(names)), names); ax2.set_xlabel("fine-tune step"); ax2.grid(False)
    cb = fig.colorbar(im, ax=ax2, pad=0.01); cb.set_label("write change / write size\n(before a word after DET)")
    frame(fig, FOOT + " · top: first 80 EXPLORE stories · heatmap: mean over 3 seeds")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURF); plt.close(fig)


def fig_torque(r, out):
    t = r["torque"]
    rows = [k for k in t if k.startswith("blocks.")]
    kinds = ["attn.qkv", "attn.proj", "mlp.fc", "mlp.proj"]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for j, kind in enumerate(kinds):
        ks = [f"blocks.{i}.{kind}.weight" for i in range(6)]
        axs[0].plot(range(6), [t[k]["deg"] for k in ks], color=CAT[j], lw=2, marker="o", ms=6, label=kind)
        axs[1].plot(range(6), [t[k]["rel_change"] for k in ks], color=CAT[j], lw=2, marker="o", ms=6, label=kind)
    axs[0].axhline(90, color=INK2, lw=1, ls="--")
    axs[0].set_ylabel("torque: angle to the weights' own pull (°)"); axs[1].set_ylabel("‖A2 − A3‖ / ‖W‖")
    for a in axs:
        a.set_xticks(range(6), [f"block {i}" for i in range(6)]); a.legend(frameon=False, fontsize=8)
    axs[0].set_title("Torque per tensor (90° = orthogonal to ordinary training)", loc="left", fontsize=10)
    axs[1].set_title("How much each tensor moved for the shift", loc="left", fontsize=10)
    emb = {k: v for k, v in t.items() if not k.startswith("blocks.")}
    frame(fig, FOOT + " · shift = A2 − A3 seed mean · pull = −∇ base loss on CONTROL · other tensors: " +
          ", ".join(f"{k.replace('.weight','')} {v['deg']:.0f}°" for k, v in emb.items() if "ln" not in k))
    fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(out, dpi=150, facecolor=SURF); plt.close(fig)


def fig_withdrawal(r, out):
    w = r["withdrawal"]
    fig, ax = plt.subplots(figsize=(8, 4))
    for k, (name, lab) in enumerate((("kept", "force kept (more adjective-rich training)"),
                                     ("withdrawn", "force withdrawn (ordinary stories)"))):
        ax.plot([p["step"] for p in w[name]], [p["T_adj_mean"] for p in w[name]], color=CAT[k], lw=2, marker="o",
                label=lab + " — fine-tune, 3 seeds")
    sp = r["surgical_withdrawal"]["path"]
    ax.plot([p["step"] for p in sp], [p["T_adj"] for p in sp], color=CAT[2], lw=2, marker="s",
            label=f"surgical edit at mlp{r['surgical_withdrawal']['layer']}, force withdrawn")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlabel("steps after the change"); ax.set_ylabel("Δ log-prob of adjectives after DET (nats)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Torque is a force you keep applying: what stays when it stops", loc="left", fontsize=11)
    frame(fig, FOOT + f" · P2 {'holds' if r['P2']['holds'] else 'does not hold'}")
    fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(out, dpi=150, facecolor=SURF); plt.close(fig)


def fig_parallax(r, out):
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    L = range(6)
    axs[0].plot(L, [r["parallax_deg"][str(l)] for l in L], color=CAT[0], lw=2, marker="o", ms=7)
    axs[0].axhline(90, color=INK2, lw=1, ls="--"); axs[0].set_ylim(60, 95)
    axs[0].set_ylabel("angle read ↔ write (°)"); axs[0].set_title("Parallax per layer (directions from CAL)", loc="left",
                                                                   fontsize=10)
    for k, name in enumerate(("read", "write")):
        for j, a in enumerate((0.25, 0.5, 1.0)):
            axs[1].plot(L, [r["edits"][f"{l}/{name}/{a}"]["T_adj"]["mean"] for l in L], color=CAT[k],
                        lw=1 + j, alpha=0.5 + 0.25 * j, marker="o", ms=4,
                        label=f"{name} dir., size {a}" if True else None)
    axs[1].axhline(0, color=INK2, lw=0.8); axs[1].set_ylabel("T_adj on EXPLORE (nats)")
    axs[1].legend(frameon=False, fontsize=7, ncol=2); axs[1].set_title("Edit effect: reading vs writing direction",
                                                                       loc="left", fontsize=10)
    for a in axs:
        a.set_xticks(L, [f"mlp{l}" for l in L])
    frame(fig, FOOT + f" · P1 (writing beats reading at size ½ at every layer) {'holds' if r['P1']['holds'] else 'does not hold'}")
    fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(out, dpi=150, facecolor=SURF); plt.close(fig)


if __name__ == "__main__":
    r = json.load(open(sys.argv[1])); od = sys.argv[2]; os.makedirs(od, exist_ok=True)
    fig_progression(r, os.path.join(od, "edit2_progression.png")); fig_torque(r, os.path.join(od, "edit2_torque.png"))
    fig_withdrawal(r, os.path.join(od, "edit2_withdrawal.png")); fig_parallax(r, os.path.join(od, "edit2_parallax.png"))
    print("wrote 4 figures")
