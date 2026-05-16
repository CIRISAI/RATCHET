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
    ar1_coefficient,
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
    "polity": 4,         # A4 — high (full human agency, Polity5 country-decades)
}


def collect_battery_samples(
    rng_seed: int = 42, window: int = 5, stride: int = 5
) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray, list]]:
    """v0.6 trajectory-window sampling for NASA battery (A0).

    Each sample is a (cycle-window, k=cells-alive, ρ=cross-cell SOH
    correlation in window, σ=mean SOH at window end). As windows slide
    forward in cycle space, σ falls (cells degrade) while ρ rises (cells
    increasingly synchronized in their decay), giving genuine σ vs k_eff
    dependence — the framework's actual setup. This replaces the v0.5
    bootstrap-of-arbitrary-cell-subsets, which collapsed σ variability
    away from the cycle-time axis and yielded R²=0.04.
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
    soh_arrays = [np.asarray(c.soh_values, dtype=float)
                  for c in cells if hasattr(c, "soh_values")]
    soh_arrays = [s for s in soh_arrays if len(s) >= window + 1]
    if len(soh_arrays) < 3:
        return None

    common_len = min(len(s) for s in soh_arrays)
    soh_mat = np.array([s[:common_len] for s in soh_arrays])
    n_cells, n_cycles = soh_mat.shape

    k_list, rho_list, sigma_list, sid_list = [], [], [], []
    for end in range(window, n_cycles + 1, stride):
        win = soh_mat[:, end - window: end]                          # (cells, window)
        # σ = mean SOH at window end across cells
        sigma = float(np.mean(win[:, -1]))
        sigma = float(np.clip(sigma, 0.01, 0.99))
        # ρ = mean pairwise correlation of cell SOH series within window
        corr_mat = np.corrcoef(win)
        off_diag = corr_mat[np.triu_indices(n_cells, k=1)]
        if len(off_diag) == 0:
            continue
        rho = float(np.nanmean(off_diag))
        if np.isnan(rho):
            continue
        # Kish convention: ρ in [0, 1]. Pre-collapse ρ can be negative for
        # battery (cells drift apart); clip but record sign in context.
        rho = float(np.clip(rho, 0.0, 1.0))
        # k = number of cells contributing (constant here; window-invariant)
        k = int(n_cells)
        k_list.append(k)
        rho_list.append(rho)
        sigma_list.append(sigma)
        sid_list.append(f"battery_w{window}_end{end:04d}")

    if len(k_list) < 4:
        return None
    return np.asarray(k_list), np.asarray(rho_list), np.asarray(sigma_list), sid_list


def collect_polity_samples(
    rng_seed: int = 42, decade_step: int = 5
) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray, list]]:
    """A4 — Polity5 country-window sampling.

    Each sample is a (country, decade-window) observation. Within the window:
      k = number of distinct institutional dimensions tracked (executive
          constraints + competition + openness etc.; capped at the count
          of non-null indicator columns)
      ρ = within-country correlation of institutional indicators over the window
      σ = mean Polity2 score normalized to [0, 1] = (polity2 + 10) / 20

    Polity2 scale is [-10, +10] where higher = more democratic-stable.
    """
    polity_path = REPO_ROOT / "data" / "institutional" / "polity5.xls"
    if not polity_path.exists():
        return None
    try:
        import pandas as pd
        df = pd.read_excel(polity_path)
    except Exception:
        return None

    cols_indicator = ["xconst", "xrcomp", "xropen", "xrreg", "exrec", "exconst"]
    cols_present = [c for c in cols_indicator if c in df.columns]
    if "polity2" not in df.columns or "country" not in df.columns or "year" not in df.columns:
        return None
    if len(cols_present) < 3:
        return None

    df = df[df["polity2"].notna()].copy()
    df["polity2"] = pd.to_numeric(df["polity2"], errors="coerce")
    df = df[(df["polity2"] >= -10) & (df["polity2"] <= 10)].copy()

    k_list, rho_list, sigma_list, sid_list = [], [], [], []
    for country, grp in df.groupby("country"):
        grp = grp.sort_values("year")
        years = grp["year"].values
        if len(years) < decade_step + 1:
            continue
        for start in range(0, len(years) - decade_step, decade_step):
            window_df = grp.iloc[start: start + decade_step]
            # σ = normalized mean polity2 over window
            sigma = (float(window_df["polity2"].mean()) + 10.0) / 20.0
            sigma = float(np.clip(sigma, 0.01, 0.99))
            # k = number of indicator columns with valid data in window
            non_null_cols = [c for c in cols_present if window_df[c].notna().sum() >= 3]
            k = len(non_null_cols)
            if k < 3:
                continue
            # ρ = mean pairwise correlation of these indicators across years in window
            sub = window_df[non_null_cols].apply(pd.to_numeric, errors="coerce").dropna()
            if len(sub) < 3:
                continue
            corr = sub.corr().values
            off = corr[np.triu_indices(corr.shape[0], k=1)]
            if len(off) == 0 or np.all(np.isnan(off)):
                continue
            rho = float(np.nanmean(off))
            if np.isnan(rho):
                continue
            rho = float(np.clip(abs(rho), 0.0, 1.0))  # use magnitude — Kish convention
            k_list.append(int(k))
            rho_list.append(rho)
            sigma_list.append(sigma)
            sid_list.append(f"polity_{country[:12].replace(' ', '_')}_y{int(window_df['year'].iloc[0])}")

    if len(k_list) < 20:
        return None
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
    """Run P1 (Kish fit R²) + P2 (Ljung-Box p + AR(1) |φ|) on one substrate's samples."""
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
    if len(fit.omega) < 4:
        results[name] = {"status": "too_short", "rung": rung}
        return
    # AR(1) — sample-size invariant, always computable for n ≥ 2
    ar1 = ar1_coefficient(fit.omega)
    # Adaptive Ljung-Box lag (needs lag < (n-1)/2 for reasonable test)
    n = len(fit.omega)
    lb_lag = max(1, min(10, (n - 1) // 2))
    try:
        lb = nh_test_autocorrelation(fit.omega, lags=[lb_lag])
        lb_p = float(lb.p_value)
        lb_reject = bool(lb.reject_null)
    except Exception as e:
        lb_p = float("nan")
        lb_reject = False
        lb_lag = None
    try:
        mz = nh_test_mean_zero(fit.omega)
        mz_p = float(mz.p_value)
        mz_reject = bool(mz.reject_null)
    except Exception:
        mz_p = float("nan"); mz_reject = False
    try:
        nm = nh_test_normality(fit.omega)
        nm_p = float(nm.p_value)
        nm_reject = bool(nm.reject_null)
    except Exception:
        nm_p = float("nan"); nm_reject = False
    if verbose:
        print(f"  P2 — AR(1) |φ| (v0.6 PRIMARY): {ar1:.4f}")
        if lb_lag is not None and not np.isnan(lb_p):
            print(f"       Ljung-Box p (lag {lb_lag}):     {lb_p:.4g}  "
                  f"({'reject (structured)' if lb_reject else 'fail-to-reject (white)'})")
        else:
            print(f"       Ljung-Box: n too small for valid lag-{lb_lag} test")
        print(f"       Mean-zero p: {mz_p:.4g}    Normality p: {nm_p:.4g}")
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
        "p2_ar1_abs_phi": float(ar1),       # v0.6 PRIMARY metric (sample-size invariant)
        "p2_ljung_box_p": lb_p,             # secondary (sample-size sensitive)
        "p2_ljung_box_lag": lb_lag,
        "p2_rejects_white": lb_reject,
        "mean_zero_p": mz_p,
        "normality_p": nm_p,
    }


def _spearman_direction(label: str, results: dict, key_prefix: str = "") -> dict:
    """v0.6 — Spearman ρ(rung, AR(1)|φ|) across substrates. Predicted ρ ≥ +0.7.

    AR(1) magnitude is sample-size invariant, so different-n substrates
    can be compared directly. Higher |φ| = more residual structure =
    higher predicted agency rung.
    """
    points = [(r["rung"], r["p2_ar1_abs_phi"], r["p2_ljung_box_p"], name)
              for name, r in results.items()
              if r.get("status") == "ok" and not name.startswith("_")]
    print()
    print(f"=== {label} — Spearman ρ(rung, AR(1) |φ|)  +  ρ(rung, Ljung-Box p) ===")
    if len(points) < 2:
        print(f"  insufficient substrates: {len(points)} (need ≥ 2)")
        return {"verdict": "INSUFFICIENT_DATA", "n": len(points)}
    try:
        from scipy.stats import spearmanr
    except ImportError:
        print("  (scipy unavailable)")
        return {"verdict": "NO_SCIPY"}
    rungs = [p[0] for p in points]
    ar1s  = [p[1] for p in points]
    pvals = [p[2] for p in points]
    rho_ar1, p_ar1 = spearmanr(rungs, ar1s)
    rho_lb,  p_lb  = spearmanr(rungs, pvals)
    for r, ar, lbp, nm in points:
        print(f"    A{r} {nm}: |φ|={ar:.4f}   ljung-box p={lbp:.4g}")
    print(f"  Spearman ρ(rung, |φ|)         = {rho_ar1:.3f}   significance p={p_ar1:.4g}")
    print(f"  Spearman ρ(rung, ljung_box_p) = {rho_lb:.3f}   significance p={p_lb:.4g}")
    print(f"  Prediction (v0.6 PRIMARY): ρ(rung, |φ|) ≥ +0.7")
    if np.isnan(rho_ar1):
        verdict = "INDETERMINATE_NaN"
    elif rho_ar1 >= 0.7:
        verdict = "STRONG_PASS"
    elif rho_ar1 >= 0.3:
        verdict = "WEAK_PASS"
    else:
        verdict = "FAIL_DIRECTION"
    print(f"  → {verdict}")
    return {
        "verdict": verdict,
        "n": len(points),
        "spearman_rho_ar1": (None if np.isnan(rho_ar1) else float(rho_ar1)),
        "spearman_p_ar1": (None if np.isnan(p_ar1) else float(p_ar1)),
        "spearman_rho_ljungbox": (None if np.isnan(rho_lb) else float(rho_lb)),
        "spearman_p_ljungbox": (None if np.isnan(p_lb) else float(p_lb)),
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
        "polity":     (collect_polity_samples,     DomainType.INSTITUTIONAL),
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
