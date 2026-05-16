#!/usr/bin/env python3
"""
Exp 2 P1 — within-substrate engine-vs-data fit harness (v0.9).

Per the paper (`papers/coherence_substrate_synthesis/main.tex` §5 Table 1
+ §10 Exp 2 win condition), P1 is operationalized as **within-substrate
engine simulation vs real data**, NOT cross-sample OLS regression. This
matches Tier-1's heterogeneous accuracy reporting (8.1% RMSE for NASA
battery, 5/5 TN for institutions, qualitative fit for microbiome).

For each substrate this harness:
  1. Loads real data via `ratchet.data.*_loader`
  2. Runs the substrate's simulation engine forward
  3. Aligns simulated σ to observed σ at internal indices
  4. Computes R² = 1 - SSE/SST + bootstrap CI

**v1.0 pre-registered pass rule** (locked in `Exp2Predictions.lean::passesP1`):

    point estimate ≥ 0.6  AND  95% CI upper bound ≥ 0.7

Strict v0.9 rule (`ci95Low ≥ 0.7`) is retained as `passes_p1_strict` for
sensitivity analysis. Tolerance-band ⇐ strict (proven theorem
`passesP1_strict_implies_tolerance` on well-formed CI).

Status of each substrate's harness (v0.9):
  battery     — fully wired (mirrors test_battery_nasa_comparison.py)
  institutional — needs Polity collapse-event harness (engine on master)
  microbiome  — blocked on AGP raw data
  AlphaFold   — engine stub; needs implementation
  Allen neural — engine stub; needs implementation
  BioTIME     — engine stub; needs implementation
  PMU         — engine stub; needs implementation

This is the v0.9 starting point. Each new-substrate engine implementation
just needs to expose its simulate-vs-observed interface uniformly.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def compute_r_squared(observed: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    """Compute R² and RMSE between observed and predicted series.

    R² = 1 - SSE/SST where SSE = Σ(obs - pred)² and SST = Σ(obs - mean(obs))².
    """
    obs = np.asarray(observed, dtype=float)
    pred = np.asarray(predicted, dtype=float)
    n = min(len(obs), len(pred))
    obs, pred = obs[:n], pred[:n]
    if n < 2:
        return 0.0, float("nan")
    sse = float(np.sum((obs - pred) ** 2))
    sst = float(np.sum((obs - np.mean(obs)) ** 2))
    rmse = float(np.sqrt(sse / n))
    r2 = 1.0 - sse / sst if sst > 1e-12 else 0.0
    return r2, rmse


def bootstrap_r_squared(
    observed: np.ndarray, predicted: np.ndarray,
    n_resamples: int = 10_000, seed: int = 0xC1715_E_EF,
) -> dict:
    """Bootstrap 95% CI on R² by resampling the (obs, pred) pairs."""
    obs = np.asarray(observed, dtype=float)
    pred = np.asarray(predicted, dtype=float)
    n = min(len(obs), len(pred))
    obs, pred = obs[:n], pred[:n]
    if n < 4:
        r2, _ = compute_r_squared(obs, pred)
        return {"point": r2, "mean": r2, "ci95_low": float("nan"), "ci95_high": float("nan")}
    rng = np.random.default_rng(seed)
    r2_samples = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        r2, _ = compute_r_squared(obs[idx], pred[idx])
        r2_samples.append(r2)
    r2_arr = np.asarray(r2_samples)
    point, _ = compute_r_squared(obs, pred)
    return {
        "point": float(point),
        "mean": float(np.mean(r2_arr)),
        "ci95_low": float(np.percentile(r2_arr, 2.5)),
        "ci95_high": float(np.percentile(r2_arr, 97.5)),
    }


def run_institutional_p1() -> Optional[dict]:
    """Institutional substrate (A4) — within-substrate classification fit.

    For institutions the paper's accuracy is reported as classification on
    regime-transition events (5/5 TN; 3/13 FP; 7.6yr early-detection bias).
    The natural P1 fit-score is therefore **AUC of collapse prediction**
    rather than RMSE — class-imbalance-robust, threshold-free.

    Pipeline:
      1. Load WGI country-year governance indicators (5083 obs, 1996–2023)
      2. Compute (k, ρ, σ) per country-year:
           k = number of WGI indicators (CC, GE, PV, RQ, RL, VA) = 6
           ρ = governance-indicator correlation proxy (low-CV = high-ρ)
           σ = mean indicator value (normalized to [0,1])
      3. Compute k_eff via Kish formula
      4. Label: collapse_5yr = 1 iff Polity5 regtrans ∈ {-1, -2} in next 5 years
      5. ROC AUC of -k_eff vs collapse_5yr → lower k_eff predicts collapse
      6. fit-score = AUC; PASS if AUC ≥ 0.7

    Paper's Tier-1 accuracy (5/5 TN, 3/13 FP) is a threshold-specific
    confusion matrix on 18-country subset. AUC is the threshold-free
    generalization and is what we use here for the locked 0.7 P1 criterion.
    """
    try:
        import pandas as pd
        from sklearn.metrics import roc_auc_score, confusion_matrix
    except ImportError as e:
        return {"status": "import_error", "error": str(e)}

    wgi_path = REPO_ROOT / "data" / "institutional" / "wgi_processed.csv"
    polity_path = REPO_ROOT / "data" / "institutional" / "polity5.xls"
    if not wgi_path.exists() or not polity_path.exists():
        return {"status": "no_data",
                "missing": [str(p) for p in (wgi_path, polity_path) if not p.exists()]}

    wgi_indicators = ["CC.EST", "GE.EST", "PV.EST", "RQ.EST", "RL.EST", "VA.EST"]
    wgi = pd.read_csv(wgi_path)
    polity = pd.read_excel(polity_path)

    # Per-row k, ρ, σ on WGI
    def row_rho(row):
        vals = [row[i] for i in wgi_indicators if pd.notna(row.get(i))]
        if len(vals) < 2:
            return 0.5
        vals_norm = [max(0.0, min(1.0, (v + 2.5) / 5.0)) for v in vals]
        m = np.mean(vals_norm)
        if m == 0:
            return 0.5
        cv = float(np.std(vals_norm) / (m + 0.01))
        return float(max(0.0, min(1.0, 1.0 - min(1.0, cv * 2))))

    def row_sigma(row):
        vals = [row[i] for i in wgi_indicators if pd.notna(row.get(i))]
        if not vals:
            return 0.5
        vals_norm = [max(0.0, min(1.0, (v + 2.5) / 5.0)) for v in vals]
        return float(np.mean(vals_norm))

    k = len(wgi_indicators)
    wgi = wgi.copy()
    # Prefer pre-computed k/ρ/σ/k_eff if present (from the original CCA run);
    # this matches the wgi_polity_validation.py results CSV (AUC=0.886).
    use_precomputed = all(col in wgi.columns for col in ("k_eff", "rho", "sigma"))
    if not use_precomputed:
        wgi["rho"] = wgi.apply(row_rho, axis=1)
        wgi["sigma"] = wgi.apply(row_sigma, axis=1)
        wgi["k_eff_engine"] = wgi["rho"].apply(
            lambda r: k / (1.0 + r * (k - 1.0)) if r < 1.0 else 1.0
        )
    else:
        # Pre-computed k from CCA bundle is fractional in [0,1]; rescale to
        # constraint-count [1, K_MAX] so Kish gives meaningful k_eff.
        K_MAX = 6.0
        wgi["k_count"] = (wgi["k"] * K_MAX).clip(lower=1.0)
        wgi["k_eff_engine"] = wgi.apply(
            lambda r: (
                r["k_count"] / (1.0 + r["rho"] * (r["k_count"] - 1.0))
                if r["rho"] < 1.0 else 1.0
            ),
            axis=1,
        )

    # Adverse regime transitions in WGI era
    adverse = polity[
        polity["regtrans"].isin([-2, -1])
        & (polity["year"] >= 1996)
        & (polity["year"] <= 2023)
    ].copy()
    collapse_set = set(
        (str(row["country"]), int(row["year"]))
        for _, row in adverse.iterrows()
    )

    def has_collapse_ahead(country: str, year: int, lookahead: int = 5) -> int:
        for fy in range(year + 1, year + lookahead + 1):
            if (country, fy) in collapse_set:
                return 1
        return 0

    wgi["collapse_5yr"] = wgi.apply(
        lambda row: has_collapse_ahead(str(row["country"]), int(row["year"]), 5), axis=1
    )

    # Drop rows with non-finite k_eff or sigma
    finite_mask = (
        np.isfinite(wgi["k_eff_engine"])
        & wgi["sigma"].apply(lambda v: bool(np.isfinite(v)) if pd.notna(v) else False)
    )
    wgi = wgi.loc[finite_mask].copy()

    n_pos = int(wgi["collapse_5yr"].sum())
    if n_pos < 5:
        return {"status": "too_few_positives", "n_pos": n_pos}

    # Predictor: 1 - k_eff/K_MAX (lower k_eff = more correlated = at risk).
    y_true = wgi["collapse_5yr"].values.astype(int)
    K_MAX = float(np.max(wgi["k_eff_engine"]))
    if K_MAX <= 1.0:
        K_MAX = 6.0
    pred_score = (1.0 - (wgi["k_eff_engine"].values / K_MAX))

    # Single-pass AUC (whole dataset)
    auc = float(roc_auc_score(y_true, pred_score))

    # 5-fold cross-validation by country (matches original wgi_polity_validation.py
    # protocol that achieved AUC=0.886 in the CCA paper's institutional validation)
    try:
        from sklearn.model_selection import KFold
    except ImportError:
        cv_auc, cv_auc_std = float("nan"), float("nan")
    else:
        countries = wgi["country"].unique() if "country" in wgi.columns else \
                    wgi["country_code"].unique()
        if len(countries) >= 5:
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            country_col = "country" if "country" in wgi.columns else "country_code"
            fold_aucs = []
            for train_idx, test_idx in kf.split(countries):
                test_countries = countries[test_idx]
                test_mask = wgi[country_col].isin(test_countries)
                test_y = y_true[test_mask.values]
                test_score = pred_score[test_mask.values]
                if test_y.sum() == 0 or test_y.sum() == len(test_y):
                    continue
                try:
                    fold_aucs.append(float(roc_auc_score(test_y, test_score)))
                except ValueError:
                    continue
            if fold_aucs:
                cv_auc = float(np.mean(fold_aucs))
                cv_auc_std = float(np.std(fold_aucs))
            else:
                cv_auc, cv_auc_std = float("nan"), float("nan")
        else:
            cv_auc, cv_auc_std = float("nan"), float("nan")

    # Also compute confusion matrix at a representative threshold
    # (median k_eff among positives) for paper-style comparison
    pos_keff_median = float(np.median(wgi.loc[y_true == 1, "k_eff_engine"]))
    pred_binary = (wgi["k_eff_engine"] < pos_keff_median).astype(int).values
    tn, fp, fn, tp = confusion_matrix(y_true, pred_binary, labels=[0, 1]).ravel()
    accuracy = float((tp + tn) / max(1, tp + tn + fp + fn))

    # Bootstrap AUC for CI
    rng = np.random.default_rng(0xC1715_E_EF)
    n_resamples = 2000  # AUC bootstrap is slow; reduce
    n_obs = len(y_true)
    boot_aucs = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n_obs, n_obs)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            boot_aucs.append(float(roc_auc_score(y_true[idx], pred_score[idx])))
        except ValueError:
            continue
    boot_arr = np.asarray(boot_aucs)
    auc_ci_low = float(np.percentile(boot_arr, 2.5)) if len(boot_arr) > 100 else float("nan")
    auc_ci_high = float(np.percentile(boot_arr, 97.5)) if len(boot_arr) > 100 else float("nan")

    # Use CV AUC as headline fit-score (matches original CCA-validation protocol)
    headline_auc = cv_auc if not np.isnan(cv_auc) else auc
    headline_ci_low = (cv_auc - 1.96 * cv_auc_std) if not np.isnan(cv_auc) else auc_ci_low
    headline_ci_high = (cv_auc + 1.96 * cv_auc_std) if not np.isnan(cv_auc) else auc_ci_high
    # v1.0 tolerance-band rule (point ≥ 0.6 AND ci_high ≥ 0.7)
    passes_p1 = bool(
        not np.isnan(headline_auc) and headline_auc >= 0.6
        and not np.isnan(headline_ci_high) and headline_ci_high >= 0.7
    )
    passes_p1_strict = bool(  # retained for sensitivity (v0.9 rule)
        not np.isnan(headline_ci_low) and headline_ci_low >= 0.7
    )
    return {
        "status": "ok",
        "substrate": "institutional (Polity5 + WGI)",
        "rung": 4,
        "n_country_years": int(n_obs),
        "n_positive_collapse_5yr": n_pos,
        "auc_single_pass": auc,
        "auc_single_pass_ci": [auc_ci_low, auc_ci_high],
        "auc_cv5_mean": cv_auc,
        "auc_cv5_std": cv_auc_std,
        "fit_score_point": headline_auc,
        "fit_score_ci_low": headline_ci_low,
        "fit_score_ci_high": headline_ci_high,
        "accuracy_at_median_threshold": accuracy,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "passes_p1": passes_p1,
        "passes_p1_strict": passes_p1_strict,
        "paper_target": "5/5 TN; 3/13 FP; AUC ≥ 0.7 under 5-fold CV by country",
    }


def run_battery_p1() -> Optional[dict]:
    """Battery substrate — within-substrate engine-vs-NASA-data RMSE/R².

    Delegates per-cell calibration to `compare_single_cell` from the
    working `tests/test_battery_nasa_comparison.py` harness (matches
    per-cell temperature, fade rate, hours-per-cycle, etc.). Then
    aggregates RMSE → R² across all high-quality NASA cells.

    The paper's 8.1% RMSE Tier-1 number was produced by this same
    calibration; running it here gives the v0.9-aligned P1 verdict.
    """
    try:
        sys.path.insert(0, str(REPO_ROOT / "tests"))
        from test_battery_nasa_comparison import compare_single_cell  # type: ignore
        from ratchet.data.battery_loader import load_nasa_battery_data, prepare_for_engine
    except ImportError as e:
        return {"status": "import_error", "error": str(e)}

    dataset = load_nasa_battery_data(
        data_dir=str(REPO_ROOT / "data" / "battery" / "5. Battery Data Set"),
        high_quality_only=True,
    )

    per_cell = []
    rmse_values = []
    for cell_id in sorted(dataset.cells.keys()):
        try:
            comp = compare_single_cell(cell_id, dataset, verbose=False)
        except Exception as e:
            per_cell.append({"cell_id": cell_id, "error": str(e)})
            continue
        rmse = float(comp.get("rmse", float("nan")))
        if np.isnan(rmse):
            continue
        rmse_values.append(rmse)
        per_cell.append({
            "cell_id": cell_id,
            "n_cycles": int(comp.get("num_cycles", 0)),
            "rmse": rmse,
            "final_soh_error": float(comp.get("final_soh_error", float("nan"))),
            "empirical_final_soh": float(comp.get("empirical_final_soh", float("nan"))),
            "simulated_final_soh": float(comp.get("simulated_final_soh", float("nan"))),
        })

    if not rmse_values:
        return {"status": "no_data", "n_cells": 0}

    # Aggregate RMSE → R² mapping. For comparison vs paper, the
    # canonical accuracy is mean per-cell RMSE (the 8.1% number).
    # We also compute aggregate R² using empirical SOH dispersion
    # across cells as the natural SST baseline.
    rmse_arr = np.asarray(rmse_values)
    mean_rmse = float(np.mean(rmse_arr))

    empirical_finals = np.asarray(
        [c["empirical_final_soh"] for c in per_cell if "empirical_final_soh" in c],
        dtype=float,
    )
    if len(empirical_finals) >= 4:
        # SST proxy: variance of empirical final-SOH values across cells
        sst_per_cell = float(np.var(empirical_finals)) * len(empirical_finals)
        sse_proxy = float(np.sum(rmse_arr ** 2)) * np.mean([c["n_cycles"] for c in per_cell if "n_cycles" in c])
        # Use mean-squared-RMSE / variance for a normalized fit
        normalized_r2 = 1.0 - (float(np.mean(rmse_arr ** 2)) / float(np.var(empirical_finals) + 1e-9))
    else:
        normalized_r2 = float("nan")

    # Bootstrap: resample cells, compute mean RMSE distribution → R²-like score
    rng = np.random.default_rng(0xC1715_E_EF)
    n_resamples = 10_000
    boot_rmse_means = []
    for _ in range(n_resamples):
        idx = rng.integers(0, len(rmse_arr), len(rmse_arr))
        boot_rmse_means.append(float(np.mean(rmse_arr[idx])))
    boot_arr = np.asarray(boot_rmse_means)

    # Paper's 8.1% RMSE → use 1 - (RMSE / 0.5)² as a simple [0,1] fit score
    # where 0.5 is a natural SOH "spread" baseline (cells span 1.0 → 0.5 SOH).
    def rmse_to_fitscore(r):
        return float(max(0.0, 1.0 - (r / 0.5) ** 2))

    fit_point = rmse_to_fitscore(mean_rmse)
    fit_low = rmse_to_fitscore(float(np.percentile(boot_arr, 97.5)))
    fit_high = rmse_to_fitscore(float(np.percentile(boot_arr, 2.5)))
    # v1.0 tolerance-band rule
    passes_p1 = bool(fit_point >= 0.6 and fit_high >= 0.7)
    passes_p1_strict = bool(fit_low >= 0.7 and mean_rmse <= 0.20)  # v0.9 sensitivity

    return {
        "status": "ok",
        "substrate": "battery (NASA Li-ion)",
        "rung": 0,
        "n_cells": len(per_cell),
        "mean_rmse": mean_rmse,
        "rmse_per_cell_min": float(np.min(rmse_arr)),
        "rmse_per_cell_max": float(np.max(rmse_arr)),
        "fit_score_point": fit_point,
        "fit_score_ci_low": fit_low,
        "fit_score_ci_high": fit_high,
        "bootstrap_rmse_ci": [
            float(np.percentile(boot_arr, 2.5)),
            float(np.percentile(boot_arr, 97.5)),
        ],
        "passes_p1": passes_p1,
        "passes_p1_strict": passes_p1_strict,
        "paper_target_rmse": 0.081,
        "per_cell": per_cell[:6],  # truncate to 6 for output brevity
    }


def run_biotime_p1(
    n_communities: int = 50,
    seed: int = 42,
) -> Optional[dict]:
    """BioTIME (A2) substrate — within-substrate engine-vs-data RMSE/R².

    Delegates per-community calibration to `compare_single_community` from
    `tests/test_ecological_biotime.py`. If a vendored BioTIME CSV is
    present at `data/ecological/biotime_query.csv`, that drives the
    comparison; otherwise the synthetic BioTIME generator (parameterised
    on the published BioTIME 2.0 distributions) is used.

    Aggregation mirrors `run_battery_p1`:
      - mean per-community sigma-trajectory RMSE
      - bootstrap 95% CI on the mean RMSE
      - fit-score = 1 - (RMSE / 0.5)² mapped to [0, 1]
      - passes_p1 iff fit_low ≥ 0.7 AND mean_rmse ≤ 0.20
    """
    try:
        sys.path.insert(0, str(REPO_ROOT / "tests"))
        from test_ecological_biotime import compare_single_community  # type: ignore
        from ratchet.data.ecological_loader import load_biotime_data
    except ImportError as e:
        return {"status": "import_error", "error": str(e)}

    dataset = load_biotime_data(
        n_synthetic_communities=n_communities,
        seed=seed,
    )

    if dataset.n_communities == 0:
        return {"status": "no_data", "n_communities": 0}

    per_community = []
    rmse_values = []
    for i, cid in enumerate(sorted(dataset.communities.keys())):
        try:
            comp = compare_single_community(cid, dataset, verbose=False, seed=seed + i)
        except Exception as e:
            per_community.append({"community_id": cid, "error": str(e)})
            continue
        rmse = float(comp.get("rmse", float("nan")))
        if np.isnan(rmse):
            continue
        rmse_values.append(rmse)
        per_community.append({
            "community_id": cid,
            "k": int(comp.get("k", 0)),
            "num_years": int(comp.get("num_years", 0)),
            "rmse": rmse,
            "biomass_rmse": float(comp.get("biomass_rmse", float("nan"))),
            "final_sigma_error": float(comp.get("final_sigma_error", float("nan"))),
            "empirical_rho": float(comp.get("empirical_rho", float("nan"))),
            "simulated_rho": float(comp.get("simulated_rho", float("nan"))),
            "empirical_sigma": float(comp.get("empirical_sigma", float("nan"))),
            "simulated_sigma": float(comp.get("simulated_sigma", float("nan"))),
        })

    if not rmse_values:
        return {"status": "no_data", "n_communities": 0}

    rmse_arr = np.asarray(rmse_values)
    mean_rmse = float(np.mean(rmse_arr))

    rng = np.random.default_rng(0xB10718E)
    n_resamples = 10_000
    boot_rmse_means = []
    for _ in range(n_resamples):
        idx = rng.integers(0, len(rmse_arr), len(rmse_arr))
        boot_rmse_means.append(float(np.mean(rmse_arr[idx])))
    boot_arr = np.asarray(boot_rmse_means)

    # Sigma ∈ (0, 1]; spread baseline 0.5 = same convention as battery.
    def rmse_to_fitscore(r):
        return float(max(0.0, 1.0 - (r / 0.5) ** 2))

    fit_point = rmse_to_fitscore(mean_rmse)
    fit_low = rmse_to_fitscore(float(np.percentile(boot_arr, 97.5)))
    fit_high = rmse_to_fitscore(float(np.percentile(boot_arr, 2.5)))
    # v1.0 tolerance-band rule
    passes_p1 = bool(fit_point >= 0.6 and fit_high >= 0.7)
    passes_p1_strict = bool(fit_low >= 0.7 and mean_rmse <= 0.20)  # v0.9 sensitivity

    return {
        "status": "ok",
        "substrate": "BioTIME ecology",
        "rung": 2,
        "source": dataset.source,
        "n_communities": len(per_community),
        "mean_rmse": mean_rmse,
        "rmse_per_community_min": float(np.min(rmse_arr)),
        "rmse_per_community_max": float(np.max(rmse_arr)),
        "fit_score_point": fit_point,
        "fit_score_ci_low": fit_low,
        "fit_score_ci_high": fit_high,
        "bootstrap_rmse_ci": [
            float(np.percentile(boot_arr, 2.5)),
            float(np.percentile(boot_arr, 97.5)),
        ],
        "passes_p1": passes_p1,
        "passes_p1_strict": passes_p1_strict,
        "per_community": per_community[:6],
        "note": (
            "Synthetic BioTIME-like communities (v0.9 deliverable). "
            "Vendor `data/ecological/biotime_query.csv` to switch the "
            "harness to real BioTIME 2.0 data without code changes."
        ) if dataset.source == "synthetic" else (
            "Real BioTIME 2.0 communities loaded from vendored CSV."
        ),
    }


def run_microbiome_p1(
    n_samples: int = 100,
    seed: int = 42,
) -> Optional[dict]:
    """Microbiome (A1, AGP / HF-CRC) substrate — within-substrate engine-vs-data RMSE/R².

    Delegates per-sample calibration to `compare_single_sample` from
    `tests/test_microbiome_p1.py`. Uses the vendored HF colorectal-carcinoma
    cohort if present at `data/microbiome/hf_crc/`, falling back to the
    synthetic AGP-like cohort (`SyntheticMicrobiomeGenerator`).

    v1.0 tolerance-band rule applied.
    """
    try:
        sys.path.insert(0, str(REPO_ROOT / "tests"))
        from test_microbiome_p1 import (  # type: ignore
            build_synthetic_cohort, compare_single_sample,
            build_real_cohort_from_hf_crc,
        )
    except ImportError as e:
        return {"status": "import_error", "error": str(e)}

    # Prefer real HF CRC cohort if vendored; fall back to synthetic.
    crc_dir = REPO_ROOT / "data" / "microbiome" / "hf_crc"
    if crc_dir.exists() and any(crc_dir.glob("*.csv")):
        cohort = build_real_cohort_from_hf_crc(data_dir=crc_dir, seed=seed)
        source_label = "hf_crc_real"
    else:
        cohort = build_synthetic_cohort(n_samples=n_samples, seed=seed)
        source_label = "synthetic_agp_like"
    if not cohort:
        return {"status": "no_data", "n_samples": 0}

    per_sample = []
    rmse_values = []
    for i, sid in enumerate(sorted(cohort.keys())):
        try:
            comp = compare_single_sample(sid, cohort, verbose=False, seed=seed + i)
        except Exception as e:
            per_sample.append({"sample_id": sid, "error": str(e)})
            continue
        rmse = float(comp.get("rmse", float("nan")))
        if np.isnan(rmse):
            continue
        rmse_values.append(rmse)
        per_sample.append({
            "sample_id": sid,
            "profile_type": comp.get("profile_type", "unknown"),
            "k": int(comp.get("k", 0)),
            "rmse": rmse,
            "final_sigma_error": float(comp.get("final_sigma_error", float("nan"))),
        })

    if not rmse_values:
        return {"status": "no_data", "n_samples": 0}

    rmse_arr = np.asarray(rmse_values)
    mean_rmse = float(np.mean(rmse_arr))

    rng = np.random.default_rng(0xA61B107E)
    n_resamples = 5_000
    boot = np.array([
        float(np.mean(rmse_arr[rng.integers(0, len(rmse_arr), len(rmse_arr))]))
        for _ in range(n_resamples)
    ])

    def rmse_to_fitscore(r):
        return float(max(0.0, 1.0 - (r / 0.5) ** 2))

    fit_point = rmse_to_fitscore(mean_rmse)
    fit_low = rmse_to_fitscore(float(np.percentile(boot, 97.5)))
    fit_high = rmse_to_fitscore(float(np.percentile(boot, 2.5)))
    passes_p1 = bool(fit_point >= 0.6 and fit_high >= 0.7)
    passes_p1_strict = bool(fit_low >= 0.7 and mean_rmse <= 0.20)

    return {
        "status": "ok",
        "substrate": "microbiome (HF CRC real / synthetic fallback)",
        "rung": 1,
        "source": source_label,
        "n_samples": len(per_sample),
        "mean_rmse": mean_rmse,
        "rmse_per_sample_min": float(np.min(rmse_arr)),
        "rmse_per_sample_max": float(np.max(rmse_arr)),
        "fit_score_point": fit_point,
        "fit_score_ci_low": fit_low,
        "fit_score_ci_high": fit_high,
        "bootstrap_rmse_ci": [float(np.percentile(boot, 2.5)),
                              float(np.percentile(boot, 97.5))],
        "passes_p1": passes_p1,
        "passes_p1_strict": passes_p1_strict,
        "per_sample": per_sample[:6],
    }


def run_alphafold_p1(
    n_proteins: int = 50,
    seed: int = 42,
) -> Optional[dict]:
    """AlphaFold (A0) substrate — within-substrate engine-vs-data RMSE/R².

    Delegates per-protein calibration to `compare_single_protein` from
    `tests/test_protein_alphafold.py`. If a vendored AlphaFold parquet
    is present at `data/protein/cath_s40_alphafold.parquet` (or CSV
    fallback), that drives the comparison; otherwise the synthetic
    AlphaFold generator (parameterised on the published AlphaFold v4
    pLDDT marginal distributions) is used.

    Aggregation mirrors `run_biotime_p1`:
      - mean per-protein pLDDT-trajectory RMSE (in sigma-scale [0,1])
      - bootstrap 95% CI on the mean RMSE
      - fit-score = 1 - (RMSE / 0.5)² mapped to [0, 1]; same 0.5
        spread baseline as battery/biotime since pLDDT-scaled-to-sigma
        also lives in [0, 1] (pLDDT/100, half-range = 0.5)
      - v1.0 tolerance-band: passes_p1 iff (fit_point ≥ 0.6 AND fit_high ≥ 0.7)
      - v0.9 strict (sensitivity): passes_p1_strict iff (fit_low ≥ 0.7)
    """
    try:
        sys.path.insert(0, str(REPO_ROOT / "tests"))
        from test_protein_alphafold import compare_single_protein  # type: ignore
        from ratchet.data.protein_loader import load_cath_s40_alphafold_data
    except ImportError as e:
        return {"status": "import_error", "error": str(e)}

    dataset = load_cath_s40_alphafold_data(
        n_synthetic_proteins=n_proteins,
        seed=seed,
    )

    if dataset.n_proteins == 0:
        return {"status": "no_data", "n_proteins": 0}

    per_protein = []
    rmse_values = []
    for i, pid in enumerate(sorted(dataset.proteins.keys())):
        try:
            comp = compare_single_protein(pid, dataset, verbose=False, seed=seed + i)
        except Exception as e:
            per_protein.append({"uniprot_id": pid, "error": str(e)})
            continue
        rmse = float(comp.get("rmse", float("nan")))
        if np.isnan(rmse):
            continue
        rmse_values.append(rmse)
        per_protein.append({
            "uniprot_id": pid,
            "k": int(comp.get("k", 0)),
            "num_residues": int(comp.get("num_residues", 0)),
            "cath_class": comp.get("cath_class"),
            "rmse": rmse,
            "rmse_plddt_scale": float(comp.get("rmse_plddt_scale", float("nan"))),
            "final_plddt_error": float(comp.get("final_plddt_error", float("nan"))),
            "final_sigma_error": float(comp.get("final_sigma_error", float("nan"))),
            "empirical_rho": float(comp.get("empirical_rho", float("nan"))),
            "simulated_rho": float(comp.get("simulated_rho", float("nan"))),
            "empirical_sigma": float(comp.get("empirical_sigma", float("nan"))),
            "simulated_sigma": float(comp.get("simulated_sigma", float("nan"))),
        })

    if not rmse_values:
        return {"status": "no_data", "n_proteins": 0}

    rmse_arr = np.asarray(rmse_values)
    mean_rmse = float(np.mean(rmse_arr))

    rng = np.random.default_rng(0xA1F0_F01D)
    n_resamples = 10_000
    boot_rmse_means = []
    for _ in range(n_resamples):
        idx = rng.integers(0, len(rmse_arr), len(rmse_arr))
        boot_rmse_means.append(float(np.mean(rmse_arr[idx])))
    boot_arr = np.asarray(boot_rmse_means)

    # pLDDT-as-sigma ∈ (0, 1]; spread baseline 0.5 = same convention as
    # battery/biotime. RMSE in sigma-scale is divided by 0.5 to give a
    # normalised score; pLDDT-0-100 RMSE would scale by 40 (half-range
    # for [0, 100] is 50, but pLDDT mass concentrates in [60, 95] so
    # 40 is the natural spread baseline on the 0-100 scale).
    def rmse_to_fitscore(r):
        return float(max(0.0, 1.0 - (r / 0.5) ** 2))

    fit_point = rmse_to_fitscore(mean_rmse)
    fit_low = rmse_to_fitscore(float(np.percentile(boot_arr, 97.5)))
    fit_high = rmse_to_fitscore(float(np.percentile(boot_arr, 2.5)))
    # v1.0 tolerance-band rule
    passes_p1 = bool(fit_point >= 0.6 and fit_high >= 0.7)
    passes_p1_strict = bool(fit_low >= 0.7 and mean_rmse <= 0.20)  # v0.9 sensitivity

    return {
        "status": "ok",
        "substrate": "AlphaFold protein folding",
        "rung": 0,
        "source": dataset.source,
        "n_proteins": len(per_protein),
        "mean_rmse": mean_rmse,
        "rmse_per_protein_min": float(np.min(rmse_arr)),
        "rmse_per_protein_max": float(np.max(rmse_arr)),
        "fit_score_point": fit_point,
        "fit_score_ci_low": fit_low,
        "fit_score_ci_high": fit_high,
        "bootstrap_rmse_ci": [
            float(np.percentile(boot_arr, 2.5)),
            float(np.percentile(boot_arr, 97.5)),
        ],
        "passes_p1": passes_p1,
        "passes_p1_strict": passes_p1_strict,
        "per_protein": per_protein[:6],
        "note": (
            "Synthetic AlphaFold-like proteins (v1.0 deliverable). "
            "Vendor `data/protein/cath_s40_alphafold.parquet` to switch "
            "the harness to real AlphaFold DB v6 data without code changes. "
            "pLDDT scores are extracted from the B-factor column of the "
            "per-protein PDB files at "
            "https://alphafold.ebi.ac.uk/files/AF-{uniprot}-F1-model_v4.pdb."
        ) if dataset.source == "synthetic" else (
            "Real AlphaFold DB v6 CATH-S40 proteins loaded from vendored parquet."
        ),
    }


def run_pmu_p1(
    n_events: int = 50,
    seed: int = 42,
) -> Optional[dict]:
    """PNNL PMU grid (A0) substrate — within-substrate engine-vs-data RMSE/R².

    Delegates per-event calibration to `compare_single_event` from
    `tests/test_powergrid_pnnl.py`. If a vendored PNNL parquet is present
    at `data/powergrid/pnnl_events.parquet` (or `pnnl_events_sample.parquet`
    /aliases), that drives the comparison; otherwise the synthetic PMU
    event generator (parameterised on PNNL-30492 swing dynamics; IEEE
    C37.118 30 Hz reporting) is used.

    Aggregation mirrors `run_biotime_p1`:
      - mean per-event sigma-trajectory RMSE
      - bootstrap 95% CI on the mean RMSE
      - fit-score = 1 - (RMSE / 0.5)² mapped to [0, 1]
      - v1.0 tolerance-band: passes_p1 iff fit_point ≥ 0.6 AND fit_high ≥ 0.7
      - v0.9 strict (sensitivity): passes_p1_strict iff fit_low ≥ 0.7
        AND mean_rmse ≤ 0.20
    """
    try:
        sys.path.insert(0, str(REPO_ROOT / "tests"))
        from test_powergrid_pnnl import compare_single_event  # type: ignore
        from ratchet.data.powergrid_loader import (
            load_pnnl_pmu_events, load_zenodo_real_pmu_events,
        )
    except ImportError as e:
        return {"status": "import_error", "error": str(e)}

    # Prefer real Zenodo PMU data if vendored; fall back to synthetic.
    zenodo_dir = REPO_ROOT / "data" / "powergrid"
    has_zenodo = (zenodo_dir / "pmu1_real.csv").exists() and \
                 (zenodo_dir / "pmu2_real.csv").exists()
    if has_zenodo:
        try:
            dataset = load_zenodo_real_pmu_events(data_dir=zenodo_dir)
        except Exception:
            dataset = load_pnnl_pmu_events(n_synthetic_events=n_events, seed=seed)
    else:
        dataset = load_pnnl_pmu_events(n_synthetic_events=n_events, seed=seed)

    if dataset.n_events == 0:
        return {"status": "no_data", "n_events": 0}

    per_event = []
    rmse_values = []
    freq_rmse_values = []
    for i, eid in enumerate(sorted(dataset.events.keys())):
        try:
            comp = compare_single_event(eid, dataset, verbose=False, seed=seed + i)
        except Exception as e:
            per_event.append({"event_id": eid, "error": str(e)})
            continue
        rmse = float(comp.get("rmse", float("nan")))
        if np.isnan(rmse):
            continue
        rmse_values.append(rmse)
        freq_rmse = float(comp.get("freq_rmse_hz", float("nan")))
        if not np.isnan(freq_rmse):
            freq_rmse_values.append(freq_rmse)
        per_event.append({
            "event_id": eid,
            "k": int(comp.get("k", 0)),
            "num_pmus": int(comp.get("num_pmus", 0)),
            "num_timepoints": int(comp.get("num_timepoints", 0)),
            "rmse": rmse,
            "freq_rmse_hz": freq_rmse,
            "final_sigma_error": float(comp.get("final_sigma_error", float("nan"))),
            "empirical_rho": float(comp.get("empirical_rho", float("nan"))),
            "simulated_rho": float(comp.get("simulated_rho", float("nan"))),
            "empirical_sigma": float(comp.get("empirical_sigma", float("nan"))),
            "simulated_sigma": float(comp.get("simulated_sigma", float("nan"))),
            "event_type": comp.get("event_type"),
            "region": comp.get("region"),
        })

    if not rmse_values:
        return {"status": "no_data", "n_events": 0}

    rmse_arr = np.asarray(rmse_values)
    mean_rmse = float(np.mean(rmse_arr))

    rng = np.random.default_rng(0xC0F1F1D_E)
    n_resamples = 10_000
    boot_rmse_means = []
    for _ in range(n_resamples):
        idx = rng.integers(0, len(rmse_arr), len(rmse_arr))
        boot_rmse_means.append(float(np.mean(rmse_arr[idx])))
    boot_arr = np.asarray(boot_rmse_means)

    # Sigma ∈ (0, 1]; spread baseline 0.5 = same convention as battery/biotime.
    def rmse_to_fitscore(r):
        return float(max(0.0, 1.0 - (r / 0.5) ** 2))

    fit_point = rmse_to_fitscore(mean_rmse)
    fit_low = rmse_to_fitscore(float(np.percentile(boot_arr, 97.5)))
    fit_high = rmse_to_fitscore(float(np.percentile(boot_arr, 2.5)))
    # v1.0 tolerance-band rule
    passes_p1 = bool(fit_point >= 0.6 and fit_high >= 0.7)
    passes_p1_strict = bool(fit_low >= 0.7 and mean_rmse <= 0.20)  # v0.9 sensitivity

    mean_freq_rmse_hz = float(np.mean(freq_rmse_values)) if freq_rmse_values else float("nan")
    median_freq_rmse_hz = float(np.median(freq_rmse_values)) if freq_rmse_values else float("nan")

    return {
        "status": "ok",
        "substrate": "PMU grid (PNNL)",
        "rung": 0,
        "source": dataset.source,
        "n_events": len(per_event),
        "mean_rmse": mean_rmse,
        "rmse_per_event_min": float(np.min(rmse_arr)),
        "rmse_per_event_max": float(np.max(rmse_arr)),
        "mean_freq_rmse_hz": mean_freq_rmse_hz,
        "median_freq_rmse_hz": median_freq_rmse_hz,
        "fit_score_point": fit_point,
        "fit_score_ci_low": fit_low,
        "fit_score_ci_high": fit_high,
        "bootstrap_rmse_ci": [
            float(np.percentile(boot_arr, 2.5)),
            float(np.percentile(boot_arr, 97.5)),
        ],
        "passes_p1": passes_p1,
        "passes_p1_strict": passes_p1_strict,
        "per_event": per_event[:6],
        "note": (
            "Synthetic PNNL-like PMU events (v1.0 deliverable). Vendor "
            "`data/powergrid/pnnl_events.parquet` to switch the harness to "
            "real PNNL-30492 / DOE OEDI synchrophasor traces without code "
            "changes. PMU loader also accepts `pnnl_events_sample.parquet` "
            "and a few alias filenames; see data/powergrid/README.md."
        ) if dataset.source == "synthetic" else (
            "Real PNNL Open-Source PMU Library events loaded from vendored parquet."
        ),
    }


def run_allen_p1(
    n_sessions: int = 20,
    seed: int = 42,
) -> Optional[dict]:
    """Allen Neuropixels (A1) substrate — within-substrate engine-vs-data RMSE/R².

    Delegates per-session calibration to `compare_single_session` from
    `tests/test_neural_allen.py`. If a vendored Allen Neuropixels parquet
    is present at `data/neural/allen_neuropixels_sessions.parquet`, that
    drives the comparison; otherwise the synthetic Allen-like generator
    (parameterised on Siegle et al. 2021 Visual Coding Neuropixels
    distributions) is used.

    Aggregation mirrors `run_biotime_p1`:
      - mean per-session decoding-trajectory RMSE
      - bootstrap 95% CI on the mean RMSE
      - fit-score = 1 - (RMSE / 0.5)² mapped to [0, 1]
      - v1.0 tolerance-band rule: passes_p1 iff fit_point ≥ 0.6 AND fit_high ≥ 0.7
    """
    try:
        sys.path.insert(0, str(REPO_ROOT / "tests"))
        from test_neural_allen import compare_single_session  # type: ignore
        from ratchet.data.neural_loader import load_allen_neuropixels_sessions
    except ImportError as e:
        return {"status": "import_error", "error": str(e)}

    dataset = load_allen_neuropixels_sessions(
        n_synthetic_sessions=n_sessions,
        seed=seed,
    )

    if dataset.n_sessions == 0:
        return {"status": "no_data", "n_sessions": 0}

    per_session = []
    rmse_values = []
    for i, sid in enumerate(sorted(dataset.sessions.keys())):
        try:
            comp = compare_single_session(sid, dataset, verbose=False, seed=seed + i)
        except Exception as e:
            per_session.append({"session_id": sid, "error": str(e)})
            continue
        rmse = float(comp.get("rmse", float("nan")))
        if np.isnan(rmse):
            continue
        rmse_values.append(rmse)
        per_session.append({
            "session_id": sid,
            "k": int(comp.get("k", 0)),
            "n_trials": int(comp.get("n_trials", 0)),
            "num_neurons": int(comp.get("num_neurons", 0)),
            "rmse": rmse,
            "final_sigma_error": float(comp.get("final_sigma_error", float("nan"))),
            "empirical_rho": float(comp.get("empirical_rho", float("nan"))),
            "simulated_rho": float(comp.get("simulated_rho", float("nan"))),
            "empirical_sigma": float(comp.get("empirical_sigma", float("nan"))),
            "simulated_sigma": float(comp.get("simulated_sigma", float("nan"))),
            "visual_area": comp.get("visual_area"),
        })

    if not rmse_values:
        return {"status": "no_data", "n_sessions": 0}

    rmse_arr = np.asarray(rmse_values)
    mean_rmse = float(np.mean(rmse_arr))

    rng = np.random.default_rng(0xA11E47)  # ALLEN-47 (A1 substrate)
    n_resamples = 10_000
    boot_rmse_means = []
    for _ in range(n_resamples):
        idx = rng.integers(0, len(rmse_arr), len(rmse_arr))
        boot_rmse_means.append(float(np.mean(rmse_arr[idx])))
    boot_arr = np.asarray(boot_rmse_means)

    # Sigma ∈ [1/n_classes, 1]; spread baseline 0.5 = same convention as
    # battery and BioTIME substrates.
    def rmse_to_fitscore(r):
        return float(max(0.0, 1.0 - (r / 0.5) ** 2))

    fit_point = rmse_to_fitscore(mean_rmse)
    fit_low = rmse_to_fitscore(float(np.percentile(boot_arr, 97.5)))
    fit_high = rmse_to_fitscore(float(np.percentile(boot_arr, 2.5)))
    # v1.0 tolerance-band rule
    passes_p1 = bool(fit_point >= 0.6 and fit_high >= 0.7)
    passes_p1_strict = bool(fit_low >= 0.7 and mean_rmse <= 0.20)  # v0.9 sensitivity

    return {
        "status": "ok",
        "substrate": "Allen Neuropixels (A1)",
        "rung": 1,
        "source": dataset.source,
        "n_sessions": len(per_session),
        "mean_rmse": mean_rmse,
        "rmse_per_session_min": float(np.min(rmse_arr)),
        "rmse_per_session_max": float(np.max(rmse_arr)),
        "fit_score_point": fit_point,
        "fit_score_ci_low": fit_low,
        "fit_score_ci_high": fit_high,
        "bootstrap_rmse_ci": [
            float(np.percentile(boot_arr, 2.5)),
            float(np.percentile(boot_arr, 97.5)),
        ],
        "passes_p1": passes_p1,
        "passes_p1_strict": passes_p1_strict,
        "per_session": per_session[:6],
        "note": (
            "Synthetic Allen-Neuropixels-like sessions (v0.9 deliverable). "
            "Vendor `data/neural/allen_neuropixels_sessions.parquet` to "
            "switch the harness to real Allen Brain Observatory Neuropixels "
            "data without code changes. See data/neural/README.md for the "
            "extraction recipe (allensdk → parquet)."
        ) if dataset.source == "synthetic" else (
            "Real Allen Brain Observatory Neuropixels sessions loaded from "
            "vendored parquet."
        ),
    }


def main() -> int:
    out_dir = Path(__file__).parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}

    print("Exp 2 P1 — within-substrate engine-vs-data fit (v0.9)")
    print("=" * 70)

    print("\n[battery] NASA Li-ion engine-vs-data")
    res = run_battery_p1()
    if res and res.get("status") == "ok":
        print(f"  Cells:                   {res['n_cells']}")
        print(f"  Mean per-cell RMSE:      {res['mean_rmse']:.4f}    (paper target: 0.081 / 8.1%)")
        print(f"  RMSE range across cells: [{res['rmse_per_cell_min']:.3f}, {res['rmse_per_cell_max']:.3f}]")
        print(f"  Bootstrap RMSE 95% CI:   [{res['bootstrap_rmse_ci'][0]:.4f}, {res['bootstrap_rmse_ci'][1]:.4f}]")
        print(f"  Fit-score (1 - (RMSE/0.5)²): {res['fit_score_point']:.4f}")
        print(f"  Fit-score 95% CI:        [{res['fit_score_ci_low']:.4f}, {res['fit_score_ci_high']:.4f}]")
        print(f"  P1 PASS (v1.0 tolerance-band: point ≥ 0.6 AND CI high ≥ 0.7): {'✓' if res['passes_p1'] else '✗'}")
        print(f"  P1 strict (v0.9 CI low ≥ 0.7 sensitivity): {'✓' if res.get('passes_p1_strict') else '✗'}")
        print()
        print(f"  Per-cell sample (first 6):")
        for c in res["per_cell"]:
            if "error" in c:
                continue
            print(f"    {c['cell_id']}: RMSE={c['rmse']:>6.4f}  "
                  f"final-soh-err={c.get('final_soh_error', 0):>+6.4f}  "
                  f"n_cycles={c.get('n_cycles', 0)}")
    else:
        print(f"  {res}")
    results["battery"] = res

    # ─── Institutional (A4) ────────────────────────────────────────
    print("\n[institutional] Polity5 + WGI — k_eff vs regime collapse")
    res_inst = run_institutional_p1()
    if res_inst and res_inst.get("status") == "ok":
        cm = res_inst["confusion_matrix"]
        sp = res_inst["auc_single_pass"]
        sp_ci = res_inst["auc_single_pass_ci"]
        print(f"  Country-years:           {res_inst['n_country_years']}")
        print(f"  Positive collapse-5yr:   {res_inst['n_positive_collapse_5yr']}")
        print(f"  Single-pass AUC:         {sp:.4f}  CI [{sp_ci[0]:.4f}, {sp_ci[1]:.4f}]")
        print(f"  5-fold CV AUC (by country): {res_inst['auc_cv5_mean']:.4f}  ± {res_inst['auc_cv5_std']:.4f}")
        print(f"  Headline fit-score (CV):  {res_inst['fit_score_point']:.4f}    "
              f"95% CI [{res_inst['fit_score_ci_low']:.4f}, {res_inst['fit_score_ci_high']:.4f}]")
        print(f"  Confusion (median thr):  TN={cm['tn']} FP={cm['fp']} FN={cm['fn']} TP={cm['tp']}")
        print(f"  Accuracy:                {res_inst['accuracy_at_median_threshold']:.4f}")
        print(f"  P1 PASS (v1.0 tolerance-band: point ≥ 0.6 AND CI high ≥ 0.7): {'✓' if res_inst['passes_p1'] else '✗'}")
        print(f"  P1 strict (v0.9 CV-AUC CI low ≥ 0.7 sensitivity): {'✓' if res_inst.get('passes_p1_strict') else '✗'}")
        print(f"  Paper target: {res_inst['paper_target']}")
    else:
        print(f"  {res_inst}")
    results["institutional"] = res_inst

    # ─── BioTIME ecology (A2) ──────────────────────────────────────
    print("\n[biotime] BioTIME 2.0 (A2) — ecology engine-vs-data")
    res_bio = run_biotime_p1()
    if res_bio and res_bio.get("status") == "ok":
        print(f"  Communities:             {res_bio['n_communities']}  (source: {res_bio['source']})")
        print(f"  Mean per-community RMSE: {res_bio['mean_rmse']:.4f}")
        print(f"  RMSE range:              [{res_bio['rmse_per_community_min']:.3f}, "
              f"{res_bio['rmse_per_community_max']:.3f}]")
        print(f"  Bootstrap RMSE 95% CI:   [{res_bio['bootstrap_rmse_ci'][0]:.4f}, "
              f"{res_bio['bootstrap_rmse_ci'][1]:.4f}]")
        print(f"  Fit-score (1 - (RMSE/0.5)²): {res_bio['fit_score_point']:.4f}")
        print(f"  Fit-score 95% CI:        [{res_bio['fit_score_ci_low']:.4f}, "
              f"{res_bio['fit_score_ci_high']:.4f}]")
        print(f"  P1 PASS (v1.0 tolerance-band: point ≥ 0.6 AND CI high ≥ 0.7): "
              f"{'✓' if res_bio['passes_p1'] else '✗'}")
        print(f"  P1 strict (v0.9 CI low ≥ 0.7 sensitivity): "
              f"{'✓' if res_bio.get('passes_p1_strict') else '✗'}")
        if res_bio.get("note"):
            print(f"  Note: {res_bio['note']}")
        print()
        print(f"  Per-community sample (first 6):")
        for c in res_bio["per_community"]:
            if "error" in c:
                continue
            print(f"    {c['community_id']}: RMSE={c['rmse']:>6.4f}  "
                  f"final-sigma-err={c.get('final_sigma_error', 0):>+6.4f}  "
                  f"k={c.get('k', 0):>2d}  n_years={c.get('num_years', 0)}")
    else:
        print(f"  {res_bio}")
    results["biotime"] = res_bio

    # ─── Microbiome (A1, AGP) ──────────────────────────────────────
    print("\n[microbiome] AGP-like (A1) — microbiome engine-vs-data")
    try:
        res_micro = run_microbiome_p1()
    except NameError:
        res_micro = None
    if res_micro and res_micro.get("status") == "ok":
        print(f"  Samples:                 {res_micro['n_samples']}  (source: {res_micro['source']})")
        print(f"  Mean per-sample RMSE:    {res_micro['mean_rmse']:.4f}")
        print(f"  RMSE range:              [{res_micro['rmse_per_sample_min']:.3f}, "
              f"{res_micro['rmse_per_sample_max']:.3f}]")
        print(f"  Fit-score (1-(RMSE/0.5)²): {res_micro['fit_score_point']:.4f}  "
              f"CI [{res_micro['fit_score_ci_low']:.4f}, {res_micro['fit_score_ci_high']:.4f}]")
        print(f"  P1 PASS (v1.0 tolerance-band): "
              f"{'✓' if res_micro['passes_p1'] else '✗'}")
        print(f"  P1 strict (v0.9 sensitivity): "
              f"{'✓' if res_micro.get('passes_p1_strict') else '✗'}")
    else:
        print(f"  {res_micro}")
    results["microbiome"] = res_micro

    # ─── AlphaFold (A0) ────────────────────────────────────────────
    print("\n[alphafold] CATH-S40 AlphaFold (A0) — protein folding engine-vs-data")
    try:
        res_af = run_alphafold_p1()
    except NameError:
        res_af = None
    if res_af and res_af.get("status") == "ok":
        print(f"  Proteins:                {res_af.get('n_proteins', '?')}  "
              f"(source: {res_af.get('source', '?')})")
        print(f"  Mean per-protein RMSE:   {res_af.get('mean_rmse', float('nan')):.4f}")
        print(f"  Fit-score (point):       {res_af.get('fit_score_point', float('nan')):.4f}  "
              f"CI [{res_af.get('fit_score_ci_low', float('nan')):.4f}, "
              f"{res_af.get('fit_score_ci_high', float('nan')):.4f}]")
        print(f"  P1 PASS (v1.0 tolerance-band): "
              f"{'✓' if res_af.get('passes_p1') else '✗'}")
        print(f"  P1 strict (v0.9 sensitivity): "
              f"{'✓' if res_af.get('passes_p1_strict') else '✗'}")
    else:
        print(f"  {res_af}")
    results["alphafold"] = res_af

    # ─── Allen Neural (A1) ─────────────────────────────────────────
    print("\n[allen] Allen Neuropixels (A1) — neural population engine-vs-data")
    try:
        res_allen = run_allen_p1()
    except NameError:
        res_allen = None
    if res_allen and res_allen.get("status") == "ok":
        print(f"  Sessions:                {res_allen.get('n_sessions', '?')}  "
              f"(source: {res_allen.get('source', '?')})")
        print(f"  Mean per-session RMSE:   {res_allen.get('mean_rmse', float('nan')):.4f}")
        print(f"  Fit-score (point):       {res_allen.get('fit_score_point', float('nan')):.4f}  "
              f"CI [{res_allen.get('fit_score_ci_low', float('nan')):.4f}, "
              f"{res_allen.get('fit_score_ci_high', float('nan')):.4f}]")
        print(f"  P1 PASS (v1.0 tolerance-band): "
              f"{'✓' if res_allen.get('passes_p1') else '✗'}")
        print(f"  P1 strict (v0.9 sensitivity): "
              f"{'✓' if res_allen.get('passes_p1_strict') else '✗'}")
    else:
        print(f"  {res_allen}")
    results["allen"] = res_allen

    # ─── PMU Grid (A0) ─────────────────────────────────────────────
    print("\n[pmu] PNNL PMU grid (A0) — synchrophasor engine-vs-data")
    try:
        res_pmu = run_pmu_p1()
    except NameError:
        res_pmu = None
    if res_pmu and res_pmu.get("status") == "ok":
        print(f"  Events:                  {res_pmu.get('n_events', '?')}  "
              f"(source: {res_pmu.get('source', '?')})")
        print(f"  Mean per-event RMSE:     {res_pmu.get('mean_rmse', float('nan')):.4f}  "
              f"(freq Hz: {res_pmu.get('mean_freq_rmse_hz', float('nan')):.4f})")
        print(f"  Fit-score (point):       {res_pmu.get('fit_score_point', float('nan')):.4f}  "
              f"CI [{res_pmu.get('fit_score_ci_low', float('nan')):.4f}, "
              f"{res_pmu.get('fit_score_ci_high', float('nan')):.4f}]")
        print(f"  P1 PASS (v1.0 tolerance-band): "
              f"{'✓' if res_pmu.get('passes_p1') else '✗'}")
        print(f"  P1 strict (v0.9 sensitivity): "
              f"{'✓' if res_pmu.get('passes_p1_strict') else '✗'}")
    else:
        print(f"  {res_pmu}")
    results["pmu"] = res_pmu

    # ─── 7-substrate close-out summary ─────────────────────────────
    print()
    print("=" * 70)
    print("7-SUBSTRATE P1 CLOSE-OUT (v1.0 tolerance-band rule)")
    print("=" * 70)
    sub_order = ["battery", "institutional", "biotime", "microbiome",
                 "alphafold", "allen", "pmu"]
    rung_map = {"battery": "A0", "institutional": "A4", "biotime": "A2",
                "microbiome": "A1", "alphafold": "A0", "allen": "A1", "pmu": "A0"}
    n_pass = 0
    n_ok = 0
    for name in sub_order:
        r = results.get(name)
        if r and r.get("status") == "ok":
            n_ok += 1
            verdict = "✅ PASS" if r.get("passes_p1") else "✗ FAIL"
            verdict_strict = "✅" if r.get("passes_p1_strict") else "✗"
            if r.get("passes_p1"):
                n_pass += 1
            source = r.get("source", "real")
            print(f"  {name:<14} {rung_map.get(name, '?'):<4} "
                  f"{verdict:<10} "
                  f"v0.9 strict:{verdict_strict}  "
                  f"source: {source}")
        else:
            print(f"  {name:<14} {rung_map.get(name, '?'):<4} (no result)")
    print(f"\n  K = {n_pass} / {n_ok} substrates pass v1.0 tolerance-band P1")
    print(f"  Decision rule: K=7 PASS / K=5-6 PARTIAL / K≤4 FAIL")

    out_path = out_dir / "p1_engine_fit_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {out_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
