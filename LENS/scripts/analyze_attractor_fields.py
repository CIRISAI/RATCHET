#!/usr/bin/env python3
"""
Generate path-overlay figures for CIRIS trace trajectories.

The intent is practical rather than metaphysical: identify repeated regions in
score space that correlate with deferral, rejection, completion, or active
execution so operators can treat them as behavioral attractors.
"""

from __future__ import annotations

import argparse
import math
import subprocess
from collections import Counter, defaultdict
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SSH_KEY = "~/Desktop/ciris_transfer/.ciris_bridge_keys/cirisbridge_ed25519"
HOST = "root@108.61.242.236"

QUERY = """COPY (
SELECT
    trace_id,
    task_id,
    agent_name,
    thought_start_at,
    started_at,
    csdma_plausibility_score AS p,
    dsdma_domain_alignment AS a,
    coherence_level AS c,
    selected_action AS verb,
    action_success,
    action_was_overridden,
    has_execution_error,
    conscience_passed,
    idma_fragility_flag
FROM cirislens.accord_traces
WHERE signature_verified = true
  AND agent_name IS NOT NULL
  AND task_id IS NOT NULL
  AND csdma_plausibility_score IS NOT NULL
  AND dsdma_domain_alignment IS NOT NULL
  AND coherence_level IS NOT NULL
ORDER BY agent_name, task_id, COALESCE(thought_start_at, started_at)
) TO STDOUT WITH CSV HEADER"""

CLASS_COLORS = {
    "complete": "#1f9d55",
    "defer": "#d97706",
    "reject": "#c81e1e",
    "override/error": "#7c3aed",
    "active": "#2563eb",
    "other": "#6b7280",
}
PONDER_COLOR = "#d97706"


def load_df() -> pd.DataFrame:
    cmd = (
        f"ssh -i {SSH_KEY} {HOST} "
        f"'docker exec cirislens-db psql -U cirislens -d cirislens -c \"{QUERY}\"'"
    )
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(result.stderr)

    df = pd.read_csv(StringIO(result.stdout))
    df["thought_start_at"] = pd.to_datetime(df["thought_start_at"], errors="coerce")
    df["started_at"] = pd.to_datetime(df["started_at"], errors="coerce")
    df["sort_time"] = df["thought_start_at"].fillna(df["started_at"])
    for col in [
        "action_success",
        "action_was_overridden",
        "has_execution_error",
        "conscience_passed",
        "idma_fragility_flag",
    ]:
        if col in df.columns:
            df[col] = df[col].map(parse_bool)
    return df


def parse_bool(value: object) -> bool:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    return str(value).strip().lower() in {"true", "t", "1", "yes", "y"}


def terminal_class(row: pd.Series) -> str:
    verb = str(row.get("verb") or "").upper()
    success = bool(row.get("action_success"))
    overridden = bool(row.get("action_was_overridden"))
    execution_error = bool(row.get("has_execution_error"))

    if overridden or execution_error:
        return "override/error"
    if verb in {"REJECT"}:
        return "reject"
    if verb in {"DEFER", "PONDER"}:
        return "defer"
    if verb == "TASK_COMPLETE":
        return "complete"
    if verb in {"SPEAK", "TOOL", "OBSERVE", "MEMORIZE", "RECALL"}:
        return "active"
    if success:
        return "active"
    return "other"


def build_paths(df: pd.DataFrame, agent: str) -> list[dict]:
    agent_df = df[df["agent_name"] == agent].copy()
    paths = []
    for task_id, group in agent_df.groupby("task_id"):
        group = group.sort_values("sort_time")
        coords = group[["p", "a", "c"]].to_numpy(float)
        if len(coords) < 2:
            continue
        final = group.iloc[-1]
        paths.append(
            {
                "task_id": task_id,
                "coords": coords,
                "verbs": tuple(group["verb"].fillna("NA").astype(str)),
                "terminal_verb": str(final["verb"]),
                "terminal_class": terminal_class(final),
                "terminal_point": coords[-1],
                "ponder_points": coords[
                    group["verb"].fillna("").astype(str).str.upper().isin({"PONDER", "DEFER"}).to_numpy()
                ],
                "fragile": bool(final.get("idma_fragility_flag")),
                "conscience": bool(final.get("conscience_passed")),
                "overridden": bool(final.get("action_was_overridden")),
                "error": bool(final.get("has_execution_error")),
            }
        )
    return paths


def top_terminal_bins(paths: list[dict], x_idx: int, y_idx: int, bins: int = 8) -> dict[str, list[tuple]]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    counts: dict[str, Counter] = defaultdict(Counter)
    for path in paths:
        x = min(max(path["terminal_point"][x_idx], 0.0), 0.999999)
        y = min(max(path["terminal_point"][y_idx], 0.0), 0.999999)
        xi = min(np.digitize(x, edges) - 1, bins - 1)
        yi = min(np.digitize(y, edges) - 1, bins - 1)
        counts[path["terminal_class"]][(xi, yi)] += 1
    return {k: v.most_common(4) for k, v in counts.items()}


def format_bin_label(item: tuple, bins: int = 8) -> str:
    (xi, yi), count = item
    step = 1.0 / bins
    return f"[{xi*step:.2f}-{(xi+1)*step:.2f}] x [{yi*step:.2f}-{(yi+1)*step:.2f}] : {count}"


def draw_projection(ax, paths: list[dict], x_idx: int, y_idx: int, x_label: str, y_label: str, title: str) -> None:
    all_points = np.concatenate([path["coords"][:, [x_idx, y_idx]] for path in paths], axis=0)
    ax.hexbin(
        all_points[:, 0],
        all_points[:, 1],
        gridsize=18,
        extent=(0, 1, 0, 1),
        cmap="Greys",
        bins="log",
        mincnt=1,
        alpha=0.30,
        linewidths=0,
    )

    for path in paths:
        pts = path["coords"][:, [x_idx, y_idx]]
        ax.plot(pts[:, 0], pts[:, 1], color="#0f172a", alpha=0.10, linewidth=0.9, zorder=1)

    ponder_pts = np.concatenate(
        [
            path["ponder_points"][:, [x_idx, y_idx]]
            for path in paths
            if len(path["ponder_points"]) > 0
        ],
        axis=0,
    ) if any(len(path["ponder_points"]) > 0 for path in paths) else np.empty((0, 2))
    if len(ponder_pts) > 0:
        ax.scatter(
            ponder_pts[:, 0],
            ponder_pts[:, 1],
            s=28,
            marker="x",
            color=PONDER_COLOR,
            alpha=0.55,
            linewidths=0.9,
            label="ponder/defer step",
            zorder=2,
        )

    for label, color in CLASS_COLORS.items():
        pts = np.array(
            [path["terminal_point"][[x_idx, y_idx]] for path in paths if path["terminal_class"] == label]
        )
        if len(pts) == 0:
            continue
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=36,
            color=color,
            alpha=0.92,
            edgecolors="white",
            linewidths=0.6,
            label=label,
            zorder=3,
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title, fontsize=12, pad=10)
    ax.grid(True, alpha=0.18, linewidth=0.5)


def summarize_paths(paths: list[dict]) -> list[str]:
    counts = Counter(path["terminal_class"] for path in paths)
    lengths = np.array([len(path["coords"]) for path in paths], dtype=float)
    lines = [
        f"paths: {len(paths)}",
        f"median steps: {np.median(lengths):.1f}",
        f"mean steps: {lengths.mean():.2f}",
        "",
        "terminal classes:",
    ]
    for label in ["complete", "defer", "reject", "override/error", "active", "other"]:
        if counts[label]:
            lines.append(f"  {label}: {counts[label]}")
    return lines


def add_summary_panel(ax, agent: str, paths: list[dict]) -> None:
    ax.axis("off")
    lines = [f"{agent} behavioral attractors", ""] + summarize_paths(paths)
    pa_bins = top_terminal_bins(paths, 0, 1)
    ponder_counter: Counter = Counter()
    bins = 8
    edges = np.linspace(0.0, 1.0, bins + 1)
    for path in paths:
        for point in path["ponder_points"]:
            x = min(max(point[0], 0.0), 0.999999)
            y = min(max(point[1], 0.0), 0.999999)
            xi = min(np.digitize(x, edges) - 1, bins - 1)
            yi = min(np.digitize(y, edges) - 1, bins - 1)
            ponder_counter[(xi, yi)] += 1
    lines += ["", "top P/A terminal bins:"]
    for label in ["complete", "defer", "reject", "override/error", "active", "other"]:
        items = pa_bins.get(label) or []
        if not items:
            continue
        lines.append(f"  {label}:")
        for item in items[:2]:
            lines.append(f"    {format_bin_label(item)}")
    if ponder_counter:
        lines += ["", "top P/A ponder bins:"]
        for item in ponder_counter.most_common(3):
            lines.append(f"  {format_bin_label(item)}")
    ax.text(
        0.0,
        1.0,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=10.5,
        family="monospace",
    )


def plot_agent(paths: list[dict], agent: str, outpath: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 13), constrained_layout=True)
    fig.patch.set_facecolor("white")
    draw_projection(axes[0, 0], paths, 0, 1, "Plausibility", "Alignment", "All paths with terminal classes")
    draw_projection(axes[0, 1], paths, 0, 2, "Plausibility", "Coherence", "Coherence corridor vs terminal exits")
    draw_projection(axes[1, 0], paths, 1, 2, "Alignment", "Coherence", "High-alignment/high-coherence basins")
    add_summary_panel(axes[1, 1], agent, paths)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        dedup = dict(zip(labels, handles))
        fig.legend(
            dedup.values(),
            dedup.keys(),
            loc="upper center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=6,
            frameon=False,
        )
    fig.suptitle(
        f"{agent} path overlay in CIRIS score space",
        fontsize=18,
        y=1.03,
    )
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_stats(paths: list[dict], agent: str, outpath: Path) -> None:
    counts = Counter(path["terminal_class"] for path in paths)
    lines = [f"agent={agent}", f"paths={len(paths)}"]
    lengths = np.array([len(path["coords"]) for path in paths], dtype=float)
    lines += [
        f"mean_steps={lengths.mean():.4f}",
        f"median_steps={np.median(lengths):.4f}",
        f"max_steps={int(lengths.max())}",
    ]
    for label in sorted(counts):
        lines.append(f"{label}={counts[label]}")
    outpath.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate path-attractor overlays from accord traces.")
    parser.add_argument("--agent", default="Ally", help="Agent name to render.")
    parser.add_argument(
        "--outdir",
        default="screenshots/attractor_fields",
        help="Directory for PNG outputs.",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_df()
    paths = build_paths(df, args.agent)
    if not paths:
        raise SystemExit(f"No multi-step paths found for agent {args.agent!r}")

    stem = args.agent.lower().replace(" ", "_")
    plot_agent(paths, args.agent, outdir / f"{stem}_overlay.png")
    write_stats(paths, args.agent, outdir / f"{stem}_overlay_stats.txt")


if __name__ == "__main__":
    main()
