"""
Kish-formula regression and engine-aware omega computation.

The standard `compute_omega_series` in residuals.py uses *time-series*
prediction: σ(t) predicted from σ(0..t-1). That captures within-substrate
temporal residual structure.

This module implements *cross-sample* Kish regression: across many samples
(each with its own (k, ρ, σ) triple), fit σ ≈ α + β · k_eff via OLS, then
take residuals. This is the framework-correct predictor for substrate-
fractality (Exp 2 P2):

    σ_pred,i = α + β · k_eff,i  where  k_eff,i = k_i / (1 + ρ_i (k_i - 1))
    ω_i = σ_obs,i - σ_pred,i

P2 then asks whether the distribution of ω across substrates of different
agency rungs shows structure correlated with rung (specifically: lower
Ljung-Box p-value at higher rung).

This is the "engine-aware predictor" referenced in REGIME.md v0.4 §"Phase 0
first-run finding": the residual we want is against the FRAMEWORK'S
prediction (Kish formula via OLS), not against a trivial mean baseline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .residuals import (
    DomainType, OmegaObservation, OmegaTimeSeries, PredictorType,
)


@dataclass
class KishFitResult:
    """Result of fitting σ = α + β · k_eff via OLS over a sample set."""
    alpha: float
    beta: float
    sigma_predicted: np.ndarray  # per-sample prediction
    omega: np.ndarray            # per-sample residual ω = σ_obs - σ_pred
    r_squared: float
    n_samples: int
    k_eff: np.ndarray
    # Diagnostic fields
    sigma_observed: np.ndarray
    k: np.ndarray
    rho: np.ndarray
    fit_intercept: bool

    def to_dict(self) -> dict:
        return {
            'alpha': self.alpha,
            'beta': self.beta,
            'r_squared': self.r_squared,
            'n_samples': self.n_samples,
            'fit_intercept': self.fit_intercept,
            'k_eff_min': float(np.min(self.k_eff)),
            'k_eff_max': float(np.max(self.k_eff)),
            'omega_mean': float(np.mean(self.omega)),
            'omega_std': float(np.std(self.omega)),
        }


def compute_k_eff(k: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Kish formula: k_eff = k / (1 + ρ (k - 1)).

    Vectorized; safe for k=1 (returns 1.0 regardless of rho).
    """
    k = np.asarray(k, dtype=float)
    rho = np.asarray(rho, dtype=float)
    denom = 1.0 + rho * (k - 1.0)
    # Guard against degenerate k=1 (denom = 1, k_eff = 1) and division-near-zero
    safe_denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    return k / safe_denom


def fit_kish_regression(
    k: np.ndarray,
    rho: np.ndarray,
    sigma: np.ndarray,
    *,
    fit_intercept: bool = True,
) -> KishFitResult:
    """Fit σ = α + β · k_eff via OLS over an array of samples.

    Args:
        k:      Array of constraint counts per sample (e.g., # cells, # neurons, # species)
        rho:    Array of within-sample correlations per sample (in [0, 1])
        sigma:  Array of observed sustainability values per sample (in [0, 1] typically)
        fit_intercept: If True, fit α; if False, force through origin (α=0)

    Returns:
        KishFitResult with per-sample predictions, residuals (omega), R², and fit coefficients.

    The framework's load-bearing prediction (P2) is that the residual ω has
    cross-substrate structure correlated with constituent agency rung.
    """
    k = np.asarray(k, dtype=float)
    rho = np.asarray(rho, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if not (len(k) == len(rho) == len(sigma)):
        raise ValueError(f"Length mismatch: k={len(k)}, rho={len(rho)}, sigma={len(sigma)}")
    n = len(sigma)
    if n < 3:
        raise ValueError(f"Need ≥ 3 samples for OLS; got {n}")

    k_eff = compute_k_eff(k, rho)

    if fit_intercept:
        X = np.column_stack([np.ones(n), k_eff])
    else:
        X = k_eff.reshape(-1, 1)

    coef, _, _, _ = np.linalg.lstsq(X, sigma, rcond=None)
    sigma_pred = X @ coef
    omega = sigma - sigma_pred

    ss_res = float(np.sum(omega ** 2))
    ss_tot = float(np.sum((sigma - np.mean(sigma)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return KishFitResult(
        alpha=float(coef[0]) if fit_intercept else 0.0,
        beta=float(coef[1]) if fit_intercept else float(coef[0]),
        sigma_predicted=sigma_pred,
        omega=omega,
        r_squared=r_squared,
        n_samples=n,
        k_eff=k_eff,
        sigma_observed=sigma,
        k=k,
        rho=rho,
        fit_intercept=fit_intercept,
    )


def compute_omega_from_kish_fit(
    k: np.ndarray,
    rho: np.ndarray,
    sigma: np.ndarray,
    *,
    domain: DomainType = DomainType.GENERIC,
    fit_intercept: bool = True,
    sample_ids: Optional[list] = None,
) -> OmegaTimeSeries:
    """Run the cross-sample Kish regression and wrap the residual as an OmegaTimeSeries.

    Each per-sample residual becomes one OmegaObservation, indexed by sample
    rather than by time (timestamp = sample index). The OmegaTimeSeries can
    then be passed to `analysis.omega.null_test` for whiteness/normality tests
    exactly as if it were a time series.
    """
    fit = fit_kish_regression(k, rho, sigma, fit_intercept=fit_intercept)
    series = OmegaTimeSeries(domain=domain)
    series.metadata['predictor'] = 'kish_regression'
    series.metadata['kish_alpha'] = fit.alpha
    series.metadata['kish_beta'] = fit.beta
    series.metadata['kish_r_squared'] = fit.r_squared
    series.metadata['fit_intercept'] = fit_intercept

    for i in range(fit.n_samples):
        obs = OmegaObservation(
            omega=float(fit.omega[i]),
            sigma_observed=float(fit.sigma_observed[i]),
            sigma_predicted=float(fit.sigma_predicted[i]),
            timestamp=float(i),  # sample index, not time
            domain=domain,
            predictor_type=PredictorType.LEARNED,
            k=int(fit.k[i]) if fit.k[i] >= 1 else None,
            k_eff=float(fit.k_eff[i]),
            context={
                'rho': float(fit.rho[i]),
                'sample_id': sample_ids[i] if sample_ids is not None and i < len(sample_ids) else None,
            },
        )
        series.append(obs)
    return series
