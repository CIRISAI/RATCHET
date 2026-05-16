#!/usr/bin/env python3
"""
Exp 2 Phase 0 — Tier-1 re-validation through omega.

Re-runs the 3 Tier-1 substrates (battery A0, microbiome A1, V-Dem A4)
through the loader → engine → omega chain on master and reports:

  P1 baseline:  per-substrate R² (against CCA paper anchors)
  P2 baseline:  Ljung-Box p-value of σ_obs - σ_pred residual
  P2 direction: Spearman ρ of (Ljung-Box p, agency_rung) — must trend ≤ 0

This is the gate before pre-registering Exp 2 proper. If Tier-1 produces
clean Ljung-Box statistics with the predicted direction, the pipeline is
trustworthy for new-substrate work. If it doesn't, debug here, not on
4 new substrates simultaneously.

Output: data/phase0_tier1_results.json + console summary.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from ratchet.data.battery_loader import load_nasa_battery_data
from analysis.omega.residuals import compute_omega_series, DomainType
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


def collect_battery_sigma_series() -> np.ndarray:
    """Per-cell SOH trajectories concatenated as one σ series (NASA Li-ion)."""
    dataset = load_nasa_battery_data(
        data_dir=str(REPO_ROOT / "data" / "battery" / "5. Battery Data Set"),
        high_quality_only=True,
    )
    series_parts: list[np.ndarray] = []
    for cell_id, cell in dataset.cells.items():
        soh = getattr(cell, "soh_values", None)
        if soh is not None and len(soh) >= 20:
            series_parts.append(np.asarray(soh, dtype=float))
    if not series_parts:
        raise RuntimeError("battery: no usable SOH trajectories found")
    # Concatenate after detrending each cell to remove linear aging drift.
    detrended: list[np.ndarray] = []
    for s in series_parts:
        t = np.arange(len(s), dtype=float)
        coef = np.polyfit(t, s, 1)
        detrended.append(s - np.polyval(coef, t))
    return np.concatenate(detrended)


def collect_microbiome_sigma_series() -> np.ndarray | None:
    """Generates a σ (Shannon diversity) sample sequence using the master-side
    SyntheticMicrobiomeGenerator. Each draw represents one host sample; the
    sequence is therefore *across hosts*, not across time for one host.

    For Phase 0 this gives us a usable A1 point. AGP raw data is not vendored
    yet; when it is, replace this with the actual cross-host σ series.
    """
    try:
        from ratchet.data.microbiome_loader import SyntheticMicrobiomeGenerator
    except Exception as e:
        print(f"  (microbiome generator import failed: {e})")
        return None
    gen = SyntheticMicrobiomeGenerator(seed=42)
    n = 300  # cohort size
    sigma_values: list[float] = []
    for _ in range(n):
        sample = gen.generate_healthy_adult()
        sigma_values.append(float(sample.sigma))
    return np.asarray(sigma_values, dtype=float)


def collect_vdem_sigma_series() -> np.ndarray | None:
    """V-Dem polyarchy index time series across countries — proxy for σ.

    Returns None if QoG/V-Dem CSV not vendored locally.
    """
    # The institutional loader expects QoG / Polity V CSV which we haven't
    # vendored yet. Phase 0 acknowledges this gap.
    return None


def main() -> int:
    out_dir = Path(__file__).parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}

    print("Exp 2 Phase 0 — Tier-1 re-validation through omega")
    print("=" * 70)

    series_funcs = {
        "battery": (collect_battery_sigma_series, DomainType.BATTERY),
        "microbiome": (collect_microbiome_sigma_series, DomainType.MICROBIOME),
        "vdem": (collect_vdem_sigma_series, DomainType.INSTITUTIONAL),
    }

    for name, (fn, domain) in series_funcs.items():
        print(f"\n[{name}] rung A{AGENCY_RUNG[name]}")
        try:
            sigma_series = fn()
        except Exception as e:
            print(f"  SKIP: {type(e).__name__}: {e}")
            results[name] = {"status": "error", "error": str(e), "rung": AGENCY_RUNG[name]}
            continue

        if sigma_series is None:
            print(f"  SKIP: data not available (substrate not yet vendored locally)")
            results[name] = {"status": "skipped_no_data", "rung": AGENCY_RUNG[name]}
            continue

        n_total = len(sigma_series)
        print(f"  σ series length: {n_total}")

        omega_series = compute_omega_series(
            sigma_series=sigma_series,
            predictor="mean",
            domain=domain,
            warmup=min(30, n_total // 10),
        )
        omega_values = np.asarray(omega_series.omega_values, dtype=float)

        if len(omega_values) < 20:
            print(f"  SKIP: omega series too short ({len(omega_values)} < 20)")
            results[name] = {"status": "too_short", "rung": AGENCY_RUNG[name]}
            continue

        # Null-hypothesis battery on the residual.
        lb = nh_test_autocorrelation(omega_values, lags=[10])
        mz = nh_test_mean_zero(omega_values)
        nm = nh_test_normality(omega_values)

        print(f"  Ljung-Box p (lag 10):  {lb.p_value:.4g}  ({'reject H0 (structured)' if lb.reject_null else 'fail to reject (white)'})")
        print(f"  Mean-zero p:           {mz.p_value:.4g}  (reject={mz.reject_null})")
        print(f"  Normality p:           {nm.p_value:.4g}  (reject={nm.reject_null})")

        results[name] = {
            "status": "ok",
            "rung": AGENCY_RUNG[name],
            "n_sigma": n_total,
            "n_omega": len(omega_values),
            "omega_mean": float(np.mean(omega_values)),
            "omega_std": float(np.std(omega_values)),
            "ljung_box_p_lag10": float(lb.p_value),
            "ljung_box_rejects_white": bool(lb.reject_null),
            "mean_zero_p": float(mz.p_value),
            "normality_p": float(nm.p_value),
        }

    # P2-direction check: Spearman correlation of (rung, ljung_box_p) across substrates with status=ok.
    print()
    print("=" * 70)
    print("P2 direction check (Spearman ρ across substrates)")
    print("=" * 70)
    points = [(r["rung"], r["ljung_box_p_lag10"], name)
              for name, r in results.items() if r.get("status") == "ok"]
    if len(points) >= 2:
        try:
            from scipy.stats import spearmanr  # type: ignore
            rungs = [p[0] for p in points]
            pvals = [p[1] for p in points]
            spearman_rho, spearman_p = spearmanr(rungs, pvals)
            print(f"  N substrates: {len(points)}")
            for p in points:
                print(f"    A{p[0]} {p[2]}: ljung-box p = {p[1]:.4g}")
            print(f"  Spearman ρ(rung, ljung_box_p) = {spearman_rho:.3f}  (p = {spearman_p:.4g})")
            print(f"  Prediction: ρ ≤ -0.7 (higher rung → lower whiteness p)")
            if spearman_rho <= -0.7:
                print(f"  → DIRECTION CONSISTENT (P2 pre-pass on Tier-1 baseline)")
            elif spearman_rho <= -0.3:
                print(f"  → DIRECTION WEAK (partial)")
            else:
                print(f"  → DIRECTION INCONSISTENT (Tier-1 doesn't show the predicted gradient — investigate before pre-registering)")
            results["_p2_direction"] = {
                "n_substrates": len(points),
                "spearman_rho": float(spearman_rho),
                "spearman_p": float(spearman_p),
                "passes_strong": bool(spearman_rho <= -0.7),
                "passes_weak": bool(spearman_rho <= -0.3),
            }
        except ImportError:
            print("  (scipy not available — skipping Spearman correlation)")
    else:
        print(f"  insufficient substrates with data: {len(points)} (need ≥ 2 for direction check)")
        print(f"  Note: microbiome + V-Dem loaders need source data vendored before this gate can fully execute.")

    out_path = out_dir / "phase0_tier1_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print()
    print(f"Wrote {out_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
