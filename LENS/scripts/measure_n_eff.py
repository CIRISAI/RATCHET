#!/usr/bin/env python3
"""
measure_n_eff — effective independent participatory constraints over the
trace corpus.

Computes two N_eff measures from the eigenvalue spectrum of the
z-scored 17-dim constraint vector:

    participation_ratio:  N_eff_PR = (Σ λ_i)² / Σ λ_i²
    entropy_perplexity:   N_eff_H  = exp(-Σ p_i · log p_i)
                                    where p_i = λ_i / Σ λ_i

Both are valid; reporting both bounds the answer. PR penalizes variance
concentration more aggressively (squares amplify dominant eigenvalues);
entropy-perplexity weights the eigenvalue tail more gently.

This is the load-bearing measurement for the Proof-of-Benefit
independence claim — see FSD/PROOF_OF_BENEFIT_FEDERATION.md §2.4.

Methodology discipline
----------------------
QA / wakeup_ritual traffic deterministically stresses the same gates
in the same shape (the conscience faculty fails-and-overrides on
identical inputs). When the same constraint is observed N times under
identical conditions, it contributes to one effective dimension, not
N — N_eff correctly compresses. Filter QA traffic before claiming
federation-primitive independence; the mixed-corpus reading is
informative for measurement-system health, not for the anti-Sybil
claim.

Connection
----------
Reuses the `CIRISLENS_PSQL` env var convention from `corpus_shape.py`.
Default goes against the production DB over SSH. Override for dev.

Usage
-----
    scripts/measure_n_eff.py                       # last 500, mixed
    scripts/measure_n_eff.py --n 1000              # bigger window
    scripts/measure_n_eff.py --filter-qa           # organic only
    scripts/measure_n_eff.py --window 24h          # by time
    scripts/measure_n_eff.py --agent Ally
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

# Drawn from `cirislens.trace_context`. Stay in sync with the FSD §2.4
# constraint vector definition; if you add or remove a column here,
# update the FSD.
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


def parse_window(window: str) -> str:
    m = re.fullmatch(r"(\d+)\s*([smhd])", window.strip().lower())
    if not m:
        raise ValueError(f"window must match Nm/Nh/Nd (got {window!r})")
    n, unit = int(m.group(1)), m.group(2)
    unit_full = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days"}[unit]
    return f"{n} {unit_full}"


def build_query(args: argparse.Namespace) -> str:
    cols = ",\n  ".join(FEATURE_COLUMNS)
    where: list[str] = []
    if args.window:
        where.append(f"timestamp > NOW() - INTERVAL '{parse_window(args.window)}'")
    if args.filter_qa:
        # Real-traffic only. Excludes QA harness + wakeup ritual + status
        # collectors; keeps real_user_web and any task_class we haven't
        # tagged (most likely production traffic from registered agents).
        where.append("(task_class IS NULL OR task_class NOT IN ('qa_eval','wakeup_ritual','other'))")
    if args.agent:
        where.append(f"agent_name = '{args.agent}'")
    where_clause = ("WHERE " + " AND ".join(where)) if where else ""

    limit_clause = f"LIMIT {args.n}" if args.n else ""

    return (
        f"COPY ("
        f"SELECT {cols} "
        f"FROM cirislens.trace_context {where_clause} "
        f"ORDER BY timestamp DESC {limit_clause}"
        f") TO STDOUT WITH CSV HEADER"
    )


def run_query(psql_cmd: str, sql: str) -> str:
    """Pipe SQL to psql via stdin (avoids -c quoting hell with parens
    and escapes), capture CSV-formatted COPY output from stdout. Same
    pattern as `corpus_shape.py`."""
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


def compute_n_eff(csv_text: str) -> dict:
    import numpy as np
    import pandas as pd

    df = pd.read_csv(StringIO(csv_text))
    raw_rows = len(df)

    # Drop rows where the most-essential features are null; impute the rest
    # with column means so a single missing optional gate doesn't poison
    # the whole row.
    df = df.dropna(subset=["csdma_plausibility_score", "idma_k_eff"])
    df = df.fillna(df.mean(numeric_only=True))
    n = len(df)
    if n < 20:
        raise SystemExit(
            f"only {n} rows survive null-filter (raw={raw_rows}); "
            "widen the window or relax filters"
        )

    X = df.values.astype(float)
    mu, sigma = X.mean(axis=0), X.std(axis=0, ddof=1)
    keep = sigma > 1e-9
    X = X[:, keep]
    labels = [c for c, k in zip(df.columns, keep) if k]
    mu, sigma = mu[keep], sigma[keep]
    Z = (X - mu) / sigma

    C = np.cov(Z, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(C)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.clip(eigvals[order], 0, None)
    eigvecs = eigvecs[:, order]

    total = eigvals.sum()
    n_eff_pr = (total ** 2) / (eigvals ** 2).sum()
    p = eigvals / total
    n_eff_h = float(np.exp(-(p * np.log(p + 1e-30)).sum()))

    pc1 = sorted(zip(labels, eigvecs[:, 0]), key=lambda x: -abs(x[1]))

    return {
        "rows": n,
        "raw_rows": raw_rows,
        "feature_dim": len(eigvals),
        "n_eff_pr": float(n_eff_pr),
        "n_eff_h": n_eff_h,
        "eigvals": eigvals.tolist(),
        "labels": labels,
        "pc1_top": pc1[:8],
    }


def render(result: dict, args: argparse.Namespace) -> None:
    bar = "─" * 60
    print(bar)
    print("EFFECTIVE INDEPENDENT PARTICIPATORY CONSTRAINTS")
    print(bar)
    print(f"  rows analyzed:    {result['rows']}  (raw {result['raw_rows']})")
    print(f"  feature dim:      {result['feature_dim']}")
    if args.window:
        print(f"  window:           {args.window}")
    if args.n:
        print(f"  cap:              last {args.n}")
    if args.filter_qa:
        print(f"  filter:           organic-only (qa_eval/wakeup_ritual/other excluded)")
    if args.agent:
        print(f"  agent:            {args.agent}")
    print()
    print(f"  N_eff (participation ratio):   {result['n_eff_pr']:.2f}")
    print(f"  N_eff (entropy perplexity):    {result['n_eff_h']:.2f}")
    print(f"  PoB high water mark:           9.2")
    print()
    print("  EIGENVALUE SPECTRUM")
    eigs = result["eigvals"]
    total = sum(eigs)
    cum = 0.0
    for i, e in enumerate(eigs):
        cum += e
        if cum / total > 0.99 and e / total < 0.005:
            print(f"    PC{i+1:2d}  …(remaining {len(eigs)-i} carry <{0.5}% each)")
            break
        print(f"    PC{i+1:2d}  λ={e:6.3f}  ({100*e/total:5.1f}%, cum {100*cum/total:5.1f}%)")
    print()
    print("  PC1 TOP LOADINGS  (the dominant constraint mode)")
    for name, load in result["pc1_top"]:
        sign = "+" if load >= 0 else "−"
        print(f"    {sign}{abs(load):.3f}  {name}")
    print()
    if args.filter_qa:
        print("  (organic-only — federation-primitive independence reading)")
    else:
        print("  (mixed corpus — for measurement-system health; rerun")
        print("   with --filter-qa for the PoB independence claim)")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=500, help="row cap (default 500)")
    p.add_argument("--window", help="time window (Nm/Nh/Nd)")
    p.add_argument("--filter-qa", action="store_true",
                   help="exclude qa_eval / wakeup_ritual / other (organic-only)")
    p.add_argument("--agent", help="restrict to one agent_name")
    args = p.parse_args()

    psql_cmd = os.environ.get("CIRISLENS_PSQL", DEFAULT_PSQL)
    sql = build_query(args)
    csv_text = run_query(psql_cmd, sql)
    result = compute_n_eff(csv_text)
    render(result, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
