#!/usr/bin/env python3
"""
measure_n_eff_rolling — N_eff time series over the corpus lifetime.

Pulls every trace's 17-dim constraint vector ordered by timestamp,
slides a fixed-size window through them, and emits the participation
ratio and entropy-perplexity N_eff at each window center.

Same constraint vector and methodology as `measure_n_eff.py` (see
FSD/PROOF_OF_BENEFIT_FEDERATION.md §2.4); this is the rolling
companion for tracking PoB independence drift over time.

Usage
-----
    scripts/measure_n_eff_rolling.py                   # 500-trace window, step 100
    scripts/measure_n_eff_rolling.py --window 1000 --step 200
    scripts/measure_n_eff_rolling.py --filter-qa       # organic-only
    scripts/measure_n_eff_rolling.py --since 2026-03-22T00:00 --csv > neff.csv
    scripts/measure_n_eff_rolling.py --plot neff.png   # write a PNG

Default output is a fixed-width text table; pass `--csv` for machine
parsing or `--plot <path>` for a quick PNG.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from io import StringIO


DEFAULT_PSQL = (
    "ssh -i ~/Desktop/ciris_transfer/.ciris_bridge_keys/cirisbridge_ed25519 "
    "root@108.61.242.236 "
    '"docker exec -i cirislens-db psql -U cirislens -d cirislens -q"'
)

FEATURE_COLUMNS = [
    "csdma_plausibility_score",
    "dsdma_domain_alignment",
    "coherence_level",
    "entropy_level",
    "idma_k_eff",
    "idma_correlation_risk",
    "COALESCE(entropy_score, 0)        AS entropy_score",
    "COALESCE(coherence_score, 0)      AS coherence_score",
    "COALESCE(optimization_veto_entropy_ratio, 0) AS opt_veto_entropy_ratio",
    "COALESCE(epistemic_humility_certainty, 0)    AS epistemic_certainty",
    "CASE WHEN idma_fragility_flag THEN 1 ELSE 0 END               AS fragile",
    "CASE WHEN entropy_passed THEN 1 WHEN entropy_passed IS FALSE THEN 0 END "
    "                                                              AS entropy_pass",
    "CASE WHEN coherence_passed THEN 1 WHEN coherence_passed IS FALSE THEN 0 END "
    "                                                              AS coherence_pass",
    "CASE WHEN optimization_veto_passed THEN 1 WHEN optimization_veto_passed "
    "IS FALSE THEN 0 END                                            AS opt_veto_pass",
    "CASE WHEN epistemic_humility_passed THEN 1 WHEN epistemic_humility_passed "
    "IS FALSE THEN 0 END                                            AS epistemic_pass",
    "CASE WHEN conscience_passed THEN 1 ELSE 0 END                  AS conscience_pass",
    "CASE WHEN action_was_overridden THEN 1 ELSE 0 END              AS overridden",
]


def build_query(args: argparse.Namespace) -> str:
    cols = ",\n  ".join(FEATURE_COLUMNS)
    where: list[str] = []
    if args.since:
        where.append(f"timestamp >= '{args.since}'")
    if args.until:
        where.append(f"timestamp < '{args.until}'")
    if args.filter_qa:
        where.append(
            "(task_class IS NULL OR task_class NOT IN "
            "('qa_eval','wakeup_ritual','other'))"
        )
    if args.agent:
        where.append(f"agent_name = '{args.agent}'")
    where_clause = ("WHERE " + " AND ".join(where)) if where else ""

    return (
        f"COPY ("
        f"SELECT timestamp, {cols} "
        f"FROM cirislens.trace_context {where_clause} "
        f"ORDER BY timestamp ASC"
        f") TO STDOUT WITH CSV HEADER"
    )


def run_query(psql_cmd: str, sql: str) -> str:
    wrapped = "\\set ON_ERROR_STOP on\n" + sql + ";\n"
    r = subprocess.run(
        psql_cmd,
        shell=True,
        input=wrapped,
        capture_output=True,
        text=True,
        check=False,
    )
    if r.returncode != 0:
        sys.stderr.write(r.stderr)
        sys.exit(r.returncode)
    return r.stdout


def n_eff_from_window(Z, eps: float = 1e-9) -> tuple[float, float]:
    import numpy as np

    # Drop columns that are constant in this window (zero variance).
    sigma = Z.std(axis=0, ddof=1)
    keep = sigma > eps
    if keep.sum() < 2:
        return float("nan"), float("nan")
    W = Z[:, keep]
    W = (W - W.mean(axis=0)) / W.std(axis=0, ddof=1)

    C = np.cov(W, rowvar=False)
    eigvals = np.linalg.eigvalsh(C)
    eigvals = np.clip(eigvals, 0, None)
    total = eigvals.sum()
    if total <= 0:
        return float("nan"), float("nan")
    pr = (total ** 2) / (eigvals ** 2).sum()
    p = eigvals / total
    h = float(np.exp(-(p * np.log(p + 1e-30)).sum()))
    return float(pr), h


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--window", type=int, default=500, help="rolling window size in traces")
    p.add_argument("--step", type=int, default=100, help="advance per measurement")
    p.add_argument("--since", help="ISO timestamp lower bound")
    p.add_argument("--until", help="ISO timestamp upper bound")
    p.add_argument("--filter-qa", action="store_true",
                   help="exclude qa_eval / wakeup_ritual / other")
    p.add_argument("--agent", help="restrict to one agent_name")
    p.add_argument("--csv", action="store_true", help="emit CSV instead of table")
    p.add_argument("--plot", metavar="PATH", help="write a PNG plot to PATH")
    args = p.parse_args()

    import numpy as np
    import pandas as pd

    psql_cmd = os.environ.get("CIRISLENS_PSQL", DEFAULT_PSQL)
    sql = build_query(args)
    sys.stderr.write(f"[fetch] querying corpus lifetime constraint vectors...\n")
    csv_text = run_query(psql_cmd, sql)
    df = pd.read_csv(StringIO(csv_text))
    sys.stderr.write(f"[fetch] {len(df)} rows\n")

    # Drop rows missing essentials; impute the rest with column means.
    df = df.dropna(subset=["csdma_plausibility_score", "idma_k_eff"]).reset_index(drop=True)
    feature_cols = [c for c in df.columns if c != "timestamp"]
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].mean(numeric_only=True))
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    sys.stderr.write(f"[clean] {len(df)} rows after null-filter\n")

    if len(df) < args.window:
        sys.stderr.write(
            f"only {len(df)} rows; need at least {args.window} for one window\n"
        )
        return 1

    # Z-score the WHOLE corpus once. Each window then re-standardizes
    # internally (so a window where one feature is constant doesn't
    # poison its eigenvalue spectrum), but we use the global means as
    # a sane starting point.
    X = df[feature_cols].to_numpy(dtype=float)
    mu = X.mean(axis=0)
    sigma = X.std(axis=0, ddof=1)
    sigma = np.where(sigma > 1e-9, sigma, 1.0)
    Z_full = (X - mu) / sigma

    # Slide the window.
    starts = list(range(0, len(df) - args.window + 1, args.step))
    rows: list[tuple] = []
    for i, s in enumerate(starts):
        e = s + args.window
        Z = Z_full[s:e]
        pr, h = n_eff_from_window(Z)
        center_ts = df["timestamp"].iloc[(s + e) // 2]
        rows.append((center_ts, s, e, pr, h))
        if not args.csv and i % 10 == 0:
            sys.stderr.write(
                f"[roll] {i}/{len(starts)} center={center_ts} "
                f"PR={pr:.2f} H={h:.2f}\n"
            )

    out = pd.DataFrame(rows, columns=["center_ts", "start", "end", "n_eff_pr", "n_eff_h"])

    if args.csv:
        out.to_csv(sys.stdout, index=False)
    else:
        bar = "─" * 70
        print(bar)
        print(f"  N_eff rolling — window={args.window}, step={args.step}, "
              f"corpus_rows={len(df)}, measurements={len(out)}")
        if args.filter_qa:
            print("  filter:    organic-only")
        if args.agent:
            print(f"  agent:     {args.agent}")
        print(bar)
        print(f"  {'center':<25} {'rows':>10}  {'N_eff_PR':>9}  {'N_eff_H':>9}")
        for _, row in out.iterrows():
            print(f"  {row.center_ts!s:<25} {row.start:>5}-{row.end:<5}  "
                  f"{row.n_eff_pr:9.2f}  {row.n_eff_h:9.2f}")
        print(bar)
        print(f"  Lifetime stats:")
        print(f"    mean PR={out.n_eff_pr.mean():.2f}  median={out.n_eff_pr.median():.2f}  "
              f"min={out.n_eff_pr.min():.2f}  max={out.n_eff_pr.max():.2f}")
        print(f"    mean H ={out.n_eff_h.mean():.2f}  median={out.n_eff_h.median():.2f}  "
              f"min={out.n_eff_h.min():.2f}  max={out.n_eff_h.max():.2f}")
        print(f"    PoB high water mark: 9.2")

    if args.plot:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            sys.stderr.write("matplotlib not installed; skipping --plot\n")
            return 0
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(out["center_ts"], out["n_eff_pr"], label="N_eff (participation ratio)",
                linewidth=2)
        ax.plot(out["center_ts"], out["n_eff_h"], label="N_eff (entropy perplexity)",
                linewidth=2)
        ax.axhline(9.2, color="gray", linestyle="--", alpha=0.5,
                   label="PoB high water (9.2)")
        ax.set_xlabel("trace center timestamp")
        ax.set_ylabel("effective independent constraints")
        ax.set_title(
            f"N_eff rolling — window={args.window} traces, step={args.step}"
            + (" (organic)" if args.filter_qa else " (mixed)")
        )
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(args.plot, dpi=120)
        sys.stderr.write(f"[plot] wrote {args.plot}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
