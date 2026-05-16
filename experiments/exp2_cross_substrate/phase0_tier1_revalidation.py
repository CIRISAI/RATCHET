#!/usr/bin/env python3
"""
Exp 2 Phase 0 — Tier-1 re-validation through engine-aware omega.

Re-runs the 3 Tier-1 substrates (battery A0, microbiome A1, V-Dem A4)
through the per-sample (k, ρ, σ) → Kish-regression → omega → null-test
chain on master and reports:

  P1 baseline:  per-substrate R² from σ = α + β · k_eff OLS fit
  P2 baseline:  Ljung-Box p-value of ω = σ_obs - σ_pred residual
  P2 direction: Spearman ρ(rung, ljung_box_p) — must trend ≤ 0 for PASS

v0.4 update: uses the engine-aware predictor (Kish regression, NOT
predictor='mean') per REGIME.md v0.4 §"Phase 0 first-run finding".
Each substrate provides per-sample (k, ρ, σ) triples; the Kish formula
predicts σ from k_eff; the residual is the framework-honest ω.

Output: data/phase0_tier1_results.json + console summary.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from analysis.omega.kish_fit import (
    compute_omega_from_kish_fit, fit_kish_regression, compute_k_eff,
)
from analysis.omega.residuals import DomainType
from analysis.omega.null_test import (
    test_autocorrelation as nh_test_autocorrelation,
    test_mean_zero as nh_test_mean_zero,
    test_normality as nh_test_normality,
)

# Pre-registered agency rung per Core.AgencyRung intrinsic operationalization.
AGENCY_RUNG = {
    "battery": 0,        # A0 — inert
    "microbiome": 1,     # A1 — low (cellular signaling)
    "vdem": 4,           # A4 — high (full human agency)
}


def collect_battery_samples(
    rng_seed: int = 42, n_samples: int = 40
) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray, list]]:
    """Build per-sample (k, ρ, σ) triples for battery.

    Strategy: bootstrap-sample subsets of NASA cells. For each draw,
      k = subset size, ρ = mean pairwise SOH correlation, σ = mean final SOH.
    """
    try:
        from ratchet.data.battery_loader import load_nasa_battery_data
    except ImportError:
        return None
    try:
        dataset = load_nasa_battery_data(
            data_dir=str(REPO_ROOT / "data" / "battery" / "5. Battery Data Set"),
            high_quality_only=True,
        )
    except Exception:
        return None

    cells = list(dataset.cells.values())
    if len(cells) < 3:
        return None

    # Per-cell SOH trajectory truncated to common length
    soh_arrays = [np.asarray(c.soh_values, dtype=float) for c in cells if hasattr(c, 'soh_values')]
    soh_arrays = [s for s in soh_arrays if len(s) >= 20]
    if len(soh_arrays) < 3:
        return None
    common_len = min(len(s) for s in soh_arrays)
    soh_mat = np.array([s[:common_len] for s in soh_arrays])  # (n_cells, n_cycles)

    rng = np.random.default_rng(rng_seed)
    n_cells = soh_mat.shape[0]
    k_list, rho_list, sigma_list, sid_list = [], [], [], []
    for i in range(n_samples):
        # Random subset size in [3, n_cells]
        k = int(rng.integers(3, n_cells + 1))
        idx = rng.choice(n_cells, size=k, replace=False)
        subset = soh_mat[idx]
        # Mean pairwise correlation across cell trajectories
        corr_mat = np.corrcoef(subset)
        # Off-diagonal mean
        off_diag = corr_mat[np.triu_indices(k, k=1)]
        rho = float(np.mean(off_diag)) if len(off_diag) > 0 else 0.0
        rho = float(np.clip(rho, 0.0, 1.0))  # Kish convention
        # σ = mean final-window SOH
        sigma = float(np.mean(subset[:, -10:]))
        sigma = float(np.clip(sigma, 0.0, 1.0))
        k_list.append(k)
        rho_list.append(rho)
        sigma_list.append(sigma)
        sid_list.append(f"battery_bootstrap_{i:03d}")

    return np.asarray(k_list), np.asarray(rho_list), np.asarray(sigma_list), sid_list


def collect_microbiome_samples(
    rng_seed: int = 42, n_samples: int = 300
) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray, list]]:
    """Per-sample (k, ρ, σ) for microbiome via SyntheticMicrobiomeGenerator.

    Each sample's k = detected species count, ρ = within-sample mean
    abundance autocorrelation, σ = normalized Shannon diversity. The
    generator's internal randomness varies all three across samples,
    giving the cross-sample regression real spread.
    """
    try:
        from ratchet.data.microbiome_loader import SyntheticMicrobiomeGenerator
    except ImportError:
        return None
    gen = SyntheticMicrobiomeGenerator(seed=rng_seed)

    k_list, rho_list, sigma_list, sid_list = [], [], [], []
    for i in range(n_samples):
        sample = gen.generate_healthy_adult()
        k = int(getattr(sample, 'k', 0) or 0)
        rho = float(getattr(sample, 'rho', 0.0) or 0.0)
        sigma = float(getattr(sample, 'sigma', 0.0) or 0.0)
        if k < 3 or sigma <= 0:
            continue
        rho = float(np.clip(rho, 0.0, 1.0))
        k_list.append(k)
        rho_list.append(rho)
        sigma_list.append(sigma)
        sid_list.append(f"microbiome_synth_{i:04d}")

    if len(k_list) < 3:
        return None
    return np.asarray(k_list), np.asarray(rho_list), np.asarray(sigma_list), sid_list


def collect_vdem_samples() -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray, list]]:
    """Per-country-year (k, ρ, σ) for V-Dem / Polity.

    Not yet vendored on master. Returns None; Phase 0 continues with the
    available substrates.
    """
    return None


def positive_control_samples(
    rung: int,
    n_samples: int = 200,
    rng_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """Positive control: synthesize data where σ = 0.5 + 0.05·k_eff with
    rung-conditioned residual structure.

    By construction:
      A0 (rung=0): residual is WHITE (uncorrelated Gaussian noise)
      A1 (rung=1): residual has mild AR(1) (φ = 0.20)
      A2 (rung=2): residual has stronger AR(1) (φ = 0.45)
      A3 (rung=3): residual has heavier AR(1) (φ = 0.70)
      A4 (rung=4): residual is HIGHLY structured (φ = 0.85)

    This validates that the pipeline correctly distinguishes white from
    structured residuals across the agency ladder. If the positive control
    reports the predicted direction (Spearman ρ ≤ -0.7), the omega +
    Kish-fit pipeline is working. If it doesn't, the pipeline has a bug.
    Either result is informative.
    """
    rng = np.random.default_rng(rng_seed + rung)
    # Draw k uniform in [10, 100]; ρ in [0.05, 0.6] uniformly
    k = rng.integers(10, 101, n_samples).astype(float)
    rho = rng.uniform(0.05, 0.60, n_samples)
    k_eff = compute_k_eff(k, rho)
    sigma_true = 0.5 + 0.05 * k_eff  # ground-truth Kish relationship

    # Rung-conditioned residual: AR(1) noise with phi = 0.20 * rung
    phi_table = {0: 0.0, 1: 0.20, 2: 0.45, 3: 0.70, 4: 0.85}
    phi = phi_table.get(rung, 0.20 * rung)
    eps_std = 0.05
    eps = np.zeros(n_samples)
    eps[0] = rng.normal(0, eps_std)
    for t in range(1, n_samples):
        eps[t] = phi * eps[t - 1] + np.sqrt(max(0.0, 1 - phi ** 2)) * rng.normal(0, eps_std)

    sigma = sigma_true + eps
    sigma = np.clip(sigma, 0.01, 0.99)
    sids = [f"pos_ctrl_A{rung}_{i:04d}" for i in range(n_samples)]
    return k, rho, sigma, sids


def _score_substrate(name: str, sample, domain, rung: int, results: dict, verbose: bool = True) -> None:
    """Run P1 (Kish fit R²) + P2 (Ljung-Box on residual) on one substrate's samples."""
    if sample is None:
        if verbose:
            print(f"  SKIP: data not available")
        results[name] = {"status": "skipped_no_data", "rung": rung}
        return
    k_arr, rho_arr, sigma_arr, sids = sample
    n = len(sigma_arr)
    if verbose:
        print(f"  Samples: {n}  (k {int(k_arr.min())}–{int(k_arr.max())}, "
              f"ρ {rho_arr.min():.3f}–{rho_arr.max():.3f}, "
              f"σ {sigma_arr.min():.3f}–{sigma_arr.max():.3f})")
    fit = fit_kish_regression(k_arr, rho_arr, sigma_arr, fit_intercept=True)
    if verbose:
        print(f"  P1 — σ = {fit.alpha:.4f} + {fit.beta:.4f}·k_eff   R² = {fit.r_squared:.4f}")
    if len(fit.omega) < 20:
        results[name] = {"status": "too_short", "rung": rung}
        return
    lb = nh_test_autocorrelation(fit.omega, lags=[10])
    mz = nh_test_mean_zero(fit.omega)
    nm = nh_test_normality(fit.omega)
    if verbose:
        print(f"  P2 — Ljung-Box p (lag 10): {lb.p_value:.4g}  "
              f"({'reject (structured)' if lb.reject_null else 'fail-to-reject (white)'})")
        print(f"       Mean-zero p: {mz.p_value:.4g}    Normality p: {nm.p_value:.4g}")
    results[name] = {
        "status": "ok",
        "rung": rung,
        "n_samples": n,
        "p1_r_squared": float(fit.r_squared),
        "p1_alpha": fit.alpha,
        "p1_beta": fit.beta,
        "p1_pass_at_0_7": bool(fit.r_squared > 0.7),
        "k_range": [int(k_arr.min()), int(k_arr.max())],
        "rho_range": [float(rho_arr.min()), float(rho_arr.max())],
        "sigma_range": [float(sigma_arr.min()), float(sigma_arr.max())],
        "omega_mean": float(np.mean(fit.omega)),
        "omega_std": float(np.std(fit.omega)),
        "p2_ljung_box_p_lag10": float(lb.p_value),
        "p2_rejects_white": bool(lb.reject_null),
        "mean_zero_p": float(mz.p_value),
        "normality_p": float(nm.p_value),
    }


def _spearman_direction(label: str, results: dict, key_prefix: str = "") -> dict:
    """Compute Spearman ρ(rung, ljung-box p) across substrates with status=ok."""
    points = [(r["rung"], r["p2_ljung_box_p_lag10"], name)
              for name, r in results.items()
              if r.get("status") == "ok" and not name.startswith("_")]
    print()
    print(f"=== {label} — Spearman ρ(rung, ljung_box_p) across substrates ===")
    if len(points) < 2:
        print(f"  insufficient substrates: {len(points)} (need ≥ 2)")
        return {"verdict": "INSUFFICIENT_DATA", "n": len(points)}
    try:
        from scipy.stats import spearmanr
    except ImportError:
        print("  (scipy unavailable)")
        return {"verdict": "NO_SCIPY"}
    rungs = [p[0] for p in points]
    pvals = [p[1] for p in points]
    rho, p_val = spearmanr(rungs, pvals)
    for r, pv, nm in points:
        print(f"    A{r} {nm}: p = {pv:.4g}")
    print(f"  Spearman ρ = {rho:.3f}  (significance p = {p_val:.4g})")
    print(f"  Prediction (P2): ρ ≤ -0.7")
    if np.isnan(rho):
        verdict = "INDETERMINATE_NaN"
    elif rho <= -0.7:
        verdict = "STRONG_PASS"
    elif rho <= -0.3:
        verdict = "WEAK_PASS"
    else:
        verdict = "FAIL_DIRECTION"
    print(f"  → {verdict}")
    return {
        "verdict": verdict,
        "n": len(points),
        "spearman_rho": (None if np.isnan(rho) else float(rho)),
        "spearman_p": (None if np.isnan(p_val) else float(p_val)),
    }


def main() -> int:
    out_dir = Path(__file__).parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Exp 2 Phase 0 — Tier-1 re-validation (engine-aware predictor)")
    print("=" * 70)

    # ─── Positive control ───────────────────────────────────────────────
    print("\n--- POSITIVE CONTROL (synthetic, rung-conditioned AR(1) noise) ---")
    pos_ctrl: dict[str, dict] = {}
    for rung in (0, 1, 2, 3, 4):
        name = f"posctrl_A{rung}"
        print(f"\n[{name}]")
        sample = positive_control_samples(rung=rung, n_samples=200)
        _score_substrate(name, sample, DomainType.GENERIC, rung, pos_ctrl, verbose=True)
    pos_verdict = _spearman_direction("POSITIVE CONTROL", pos_ctrl)
    pos_ctrl["_p2_direction"] = pos_verdict

    # ─── Real Tier-1 substrates ─────────────────────────────────────────
    print("\n\n--- REAL TIER-1 SUBSTRATES ---")
    collectors = {
        "battery":    (collect_battery_samples,    DomainType.BATTERY),
        "microbiome": (collect_microbiome_samples, DomainType.MICROBIOME),
        "vdem":       (collect_vdem_samples,       DomainType.INSTITUTIONAL),
    }

    results: dict[str, dict] = {}

    for name, (fn, domain) in collectors.items():
        print(f"\n[{name}] rung A{AGENCY_RUNG[name]}")
        try:
            sample = fn()
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            results[name] = {"status": "error", "error": str(e), "rung": AGENCY_RUNG[name]}
            continue
        _score_substrate(name, sample, domain, AGENCY_RUNG[name], results, verbose=True)

    real_verdict = _spearman_direction("REAL TIER-1", results)
    results["_p2_direction"] = real_verdict

    # ─── Combined output ────────────────────────────────────────────────
    combined = {
        "positive_control": pos_ctrl,
        "real_tier1": results,
    }
    out_path = out_dir / "phase0_tier1_results.json"
    out_path.write_text(json.dumps(combined, indent=2))
    print(f"\nWrote {out_path.relative_to(REPO_ROOT)}")

    # ─── Final interpretation banner ────────────────────────────────────
    print()
    print("=" * 70)
    print("PHASE 0 GATE INTERPRETATION")
    print("=" * 70)
    pos_v = pos_verdict.get("verdict")
    real_v = real_verdict.get("verdict")
    print(f"  positive control: {pos_v}")
    print(f"  real Tier-1:      {real_v}")
    if pos_v in ("STRONG_PASS", "WEAK_PASS") and real_v in ("STRONG_PASS", "WEAK_PASS"):
        print(f"  → PIPELINE WORKS + DATA SUPPORTS P2  (gate passes)")
    elif pos_v in ("STRONG_PASS", "WEAK_PASS"):
        print(f"  → PIPELINE WORKS but Tier-1 data does NOT show P2 direction")
        print(f"    Implication: sampling design or rung mapping needs refinement, OR")
        print(f"    P2 prediction needs revision before pre-registration.")
    elif real_v in ("STRONG_PASS", "WEAK_PASS"):
        print(f"  → POS CTRL FAILS but real data passes — investigate pipeline bug")
    else:
        print(f"  → NEITHER direction holds — investigate pipeline bug or revise prediction")
    return 0


if __name__ == "__main__":
    sys.exit(main())
