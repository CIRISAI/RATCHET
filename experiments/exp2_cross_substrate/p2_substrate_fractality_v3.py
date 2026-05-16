#!/usr/bin/env python3
"""
Exp 2 P2 v3.0 — Substrate Fractality on ρ_t (cross-constituent coordination).

v2.0 RETIREMENT: Across 8 substrates (battery, alphafold, allen, biotime, ciris,
institutional, vdem, wgi), v2.0's excess|φ|(σ_t-Kish-residual) gave Spearman
ρ(rung, excess|φ|) = -0.012 — essentially zero. The metric was dominated by
*temporal smoothness of σ_t*, which is determined by substrate-specific data
type (categorical Polity5 → jumpy; composite V-Dem → smooth; z-score WGI →
noisy), aggregation choices, and substrate-native physics. Agency was buried.

v3.0 DESIGN: bypass σ_t entirely. The framework's claim is
"CONSTITUENTS COORDINATE" — ρ_t encodes exactly that (mean pairwise correlation
across constituents at time t). The same statistic computed the same way for
every substrate. Its TEMPORAL STRUCTURE is the agency signal.

For each substrate's trajectory:
  ρ_t time series: ρ_1, ρ_2, …, ρ_N   (already computed per window in v2.0)

  Metric A (raw): mean|φ| of ρ_t at lags 1..min(10, N/3)
                  minus median of 200-shuffle null   → excess_ρ_raw

  Metric B (AR1): fit ρ_t = a + b·ρ_{t-1} + ε_t (AR(1))
                  mean|φ| of ε_t at lags 1..min(10, N/3)
                  minus median of 200-shuffle null   → excess_ρ_AR1

Metric A captures total temporal structure (including smoothness baseline).
Metric B captures structure BEYOND simple AR(1) smoothness — closer to the
framework's "coordination beyond what trivial dynamics predict."

Substrate decision: Spearman ρ(rung, excess_ρ_X) for each X ∈ {raw, AR1}.
Same partition as v2.0 / Lake decideP2:
  ≥ +0.7  STRONG_PASS
  ≥ +0.3  WEAK_PASS
  ∈ (-0.3, +0.3) INCONCLUSIVE
  ≤ -0.3  WEAK_FAIL
  ≤ -0.7  STRONG_FAIL

Substrates: same 8 as v2.0+WGI run (battery alphafold allen biotime ciris
institutional vdem wgi). Trajectory extractors reused unchanged from v2.0.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from analysis.omega.kish_fit import autocorr_decay_profile  # noqa: E402

from experiments.exp2_cross_substrate.p2_substrate_fractality_v2 import (  # noqa: E402
    SUBSTRATE_RUNGS,
    TRAJECTORY_GETTERS,
    Trajectory,
    MIN_TRAJ_LEN,
    MAX_LAG,
    N_NULL_SHUFFLES,
    N_BOOTSTRAP,
)


def ar1_residuals(x: np.ndarray) -> np.ndarray:
    """Fit AR(1) x_t = a + b·x_{t-1} + ε_t and return ε_t (length n-1).

    Uses ordinary least squares on the lagged regression.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 3:
        return np.array([])
    y = x[1:]
    z = x[:-1]
    # OLS for y = a + b·z
    A = np.vstack([np.ones_like(z), z]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    return y - (a + b * z)


def phi_with_null(x: np.ndarray, n_shuffles: int, seed: int) -> Tuple[float, float]:
    """Return (mean|φ|_ordered, mean|φ|_null_median) for x at lags 1..MAX_LAG."""
    if len(x) < MIN_TRAJ_LEN:
        return float("nan"), float("nan")
    _, _, phi_ord, _ = autocorr_decay_profile(x, max_lag=MAX_LAG)
    rng = np.random.default_rng(seed)
    nulls = []
    for _ in range(n_shuffles):
        nulls.append(autocorr_decay_profile(x[rng.permutation(len(x))], max_lag=MAX_LAG)[2])
    return float(phi_ord), float(np.median(nulls))


def trajectory_rho_metrics(traj: Trajectory, n_null: int, seed: int) -> Optional[dict]:
    """Compute v3.0 metrics on ρ_t (and AR(1)-residual of ρ_t)."""
    k, rho, sigma = traj
    if len(rho) < MIN_TRAJ_LEN:
        return None
    phi_raw, null_raw = phi_with_null(rho, n_null, seed)
    eps = ar1_residuals(rho)
    if len(eps) < MIN_TRAJ_LEN - 1:
        phi_ar1, null_ar1 = float("nan"), float("nan")
    else:
        phi_ar1, null_ar1 = phi_with_null(eps, n_null, seed + 1)
    return {
        "n": int(len(rho)),
        "phi_rho_raw_ordered": phi_raw,
        "phi_rho_raw_null":    null_raw,
        "phi_rho_AR1_ordered": phi_ar1,
        "phi_rho_AR1_null":    null_ar1,
        "rho_mean":  float(np.mean(rho)),
        "rho_std":   float(np.std(rho)),
    }


def compute_substrate_v3(name: str, seed: int = 42) -> dict:
    getter = TRAJECTORY_GETTERS[name]
    trajectories = getter(seed=seed)
    if not trajectories:
        return {"substrate": name, "status": "no_data"}
    per_traj = []
    for i, traj in enumerate(trajectories):
        r = trajectory_rho_metrics(traj, n_null=N_NULL_SHUFFLES, seed=seed + i)
        if r is not None:
            per_traj.append(r)
    if len(per_traj) < 1:
        return {"substrate": name, "status": "no_valid_trajectories"}

    arr_raw_ord = np.array([r["phi_rho_raw_ordered"] for r in per_traj])
    arr_raw_null = np.array([r["phi_rho_raw_null"] for r in per_traj])
    arr_ar1_ord = np.array([r["phi_rho_AR1_ordered"] for r in per_traj])
    arr_ar1_null = np.array([r["phi_rho_AR1_null"] for r in per_traj])
    excess_raw = arr_raw_ord - arr_raw_null
    excess_ar1 = arr_ar1_ord - arr_ar1_null

    # Bootstrap CI on the two excess metrics
    rng = np.random.default_rng(seed + 31337)
    n_traj = len(per_traj)
    boot_raw, boot_ar1 = [], []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n_traj, n_traj)
        boot_raw.append(float((arr_raw_ord[idx] - arr_raw_null[idx]).mean()))
        ar1_sample = excess_ar1[idx]
        ar1_sample = ar1_sample[~np.isnan(ar1_sample)]
        if len(ar1_sample) > 0:
            boot_ar1.append(float(ar1_sample.mean()))

    rho_mean_all = float(np.mean([r["rho_mean"] for r in per_traj]))
    rho_std_all = float(np.mean([r["rho_std"] for r in per_traj]))

    return {
        "substrate": name,
        "rung": SUBSTRATE_RUNGS[name],
        "status": "ok",
        "n_trajectories": n_traj,
        "mean_traj_length": float(np.mean([r["n"] for r in per_traj])),
        "rho_mean_traj_mean": rho_mean_all,
        "rho_std_traj_mean":  rho_std_all,
        "phi_rho_raw_ordered": float(arr_raw_ord.mean()),
        "phi_rho_raw_null":    float(arr_raw_null.mean()),
        "excess_rho_raw":      float(excess_raw.mean()),
        "excess_rho_raw_ci":   [float(np.percentile(boot_raw, 2.5)),
                                float(np.percentile(boot_raw, 97.5))],
        "phi_rho_AR1_ordered": float(np.nanmean(arr_ar1_ord)),
        "phi_rho_AR1_null":    float(np.nanmean(arr_ar1_null)),
        "excess_rho_AR1":      float(np.nanmean(excess_ar1)),
        "excess_rho_AR1_ci":   ([float(np.percentile(boot_ar1, 2.5)),
                                 float(np.percentile(boot_ar1, 97.5))]
                                if len(boot_ar1) > 0 else [float("nan"), float("nan")]),
    }


def verdict_from_spearman(rho: float) -> str:
    if np.isnan(rho):
        return "INDETERMINATE_NaN"
    if rho >= 0.7:
        return "STRONG_PASS"
    if rho >= 0.3:
        return "WEAK_PASS"
    if rho > -0.3:
        return "INCONCLUSIVE"
    if rho > -0.7:
        return "WEAK_FAIL"
    return "STRONG_FAIL"


def run_p2_v3(seed: int = 42) -> dict:
    print("Exp 2 P2 v3.0 — Cross-constituent coordination ρ_t structure")
    print("=" * 76)
    results = {}
    for name in TRAJECTORY_GETTERS:
        print(f"\n[{name}] rung A{SUBSTRATE_RUNGS[name]}")
        r = compute_substrate_v3(name, seed=seed)
        if r.get("status") == "ok":
            print(f"  n_traj={r['n_trajectories']}  mean_len={r['mean_traj_length']:.1f}  "
                  f"<ρ>={r['rho_mean_traj_mean']:.3f}  σ(ρ)={r['rho_std_traj_mean']:.3f}")
            print(f"  RAW |φ|(ρ_t):     ord={r['phi_rho_raw_ordered']:.4f}  "
                  f"null={r['phi_rho_raw_null']:.4f}  "
                  f"excess={r['excess_rho_raw']:+.4f}  "
                  f"CI {r['excess_rho_raw_ci']}")
            print(f"  AR(1)res |φ|:    ord={r['phi_rho_AR1_ordered']:.4f}  "
                  f"null={r['phi_rho_AR1_null']:.4f}  "
                  f"excess={r['excess_rho_AR1']:+.4f}  "
                  f"CI {r['excess_rho_AR1_ci']}")
        else:
            print(f"  {r}")
        results[name] = r

    valid = [r for r in results.values() if r.get("status") == "ok"]
    print()
    print("=" * 76)
    print(f"VALID SUBSTRATES: {len(valid)} / {len(TRAJECTORY_GETTERS)}")

    summary = {"version": "v3.0",
               "n_valid_substrates": len(valid),
               "per_substrate": results}

    if len(valid) < 4:
        summary["verdict"] = "INDETERMINATE"
        summary["spearman"] = {}
        print("  → INDETERMINATE (need ≥ 4 valid substrates)")
        return summary

    from scipy.stats import spearmanr
    rungs = [r["rung"] for r in valid]

    for metric_name, key in [("RAW ρ_t",       "excess_rho_raw"),
                              ("AR(1)-residual ρ_t", "excess_rho_AR1")]:
        excs = [r[key] for r in valid]
        # Drop NaNs (AR(1) can fail on very short trajectories)
        pairs = [(ru, ex) for ru, ex in zip(rungs, excs) if not np.isnan(ex)]
        if len(pairs) < 4:
            print(f"\n  [{metric_name}] insufficient valid for Spearman")
            summary["spearman"] = summary.get("spearman", {})
            summary["spearman"][metric_name] = None
            continue
        ru_a = [p[0] for p in pairs]
        ex_a = [p[1] for p in pairs]
        rho, p = spearmanr(ru_a, ex_a)
        verdict = verdict_from_spearman(rho)
        print(f"\n  [{metric_name}]  n={len(pairs)}  "
              f"Spearman ρ(rung, excess) = {rho:+.4f}  p={p:.4g}  → {verdict}")
        for r in sorted(valid, key=lambda x: (x["rung"], x["substrate"])):
            if np.isnan(r[key]):
                continue
            print(f"    A{r['rung']} {r['substrate']:<14}  excess = {r[key]:+.4f}")
        summary["spearman"] = summary.get("spearman", {})
        summary["spearman"][metric_name] = {
            "spearman_rho": float(rho),
            "spearman_p":   float(p),
            "verdict":      verdict,
            "n_substrates": len(pairs),
        }

    return summary


if __name__ == "__main__":
    out = run_p2_v3(seed=42)
    out_dir = REPO_ROOT / "experiments/exp2_cross_substrate/data"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "p2_substrate_fractality_v3_results.json").write_text(
        json.dumps(out, indent=2, default=str))
    print()
    print(f"Wrote {out_dir / 'p2_substrate_fractality_v3_results.json'}")
