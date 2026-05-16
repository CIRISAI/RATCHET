"""
RATCHET Omega Module - Engine Validation

Validates omega computations against existing engines to ensure
predictions and residuals are computed correctly.

Validation includes:
    - Cross-validation of sigma predictions
    - Comparison of omega across domains
    - Consistency checks
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union, Callable

from .residuals import (
    OmegaTimeSeries, OmegaObservation, DomainType,
    compute_omega_series, compute_omega_from_engine, EngineProtocol,
)
from .null_test import run_null_hypothesis_battery, NullHypothesisBattery
from .distribution import compute_distribution_stats, DistributionStats


@dataclass
class ValidationResult:
    """
    Result of validating omega computation against an engine.

    Attributes:
        valid: Whether validation passed
        domain: Domain type
        n_observations: Number of observations
        mean_omega: Mean omega value
        std_omega: Standard deviation of omega
        rmse: Root mean squared error of predictions
        mae: Mean absolute error of predictions
        r_squared: R-squared of predictions
        null_tests: Results from null hypothesis tests
        distribution_stats: Distribution statistics
        issues: List of validation issues found
        details: Additional details
    """
    valid: bool = True
    domain: Optional[DomainType] = None
    n_observations: int = 0
    mean_omega: float = 0.0
    std_omega: float = 0.0
    rmse: float = 0.0
    mae: float = 0.0
    r_squared: float = 0.0
    null_tests: Optional[NullHypothesisBattery] = None
    distribution_stats: Optional[DistributionStats] = None
    issues: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Check for issues."""
        if self.n_observations < 10:
            self.issues.append('Insufficient observations (n < 10)')
            self.valid = False

        if abs(self.mean_omega) > 0.5:
            self.issues.append(f'Large mean omega bias: {self.mean_omega:.3f}')

        if self.rmse > 0.5:
            self.issues.append(f'High prediction RMSE: {self.rmse:.3f}')

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'valid': self.valid,
            'domain': self.domain.value if self.domain else None,
            'n_observations': self.n_observations,
            'mean_omega': self.mean_omega,
            'std_omega': self.std_omega,
            'rmse': self.rmse,
            'mae': self.mae,
            'r_squared': self.r_squared,
            'issues': self.issues,
            'null_tests': self.null_tests.to_dict() if self.null_tests else None,
            'distribution_stats': self.distribution_stats.to_dict() if self.distribution_stats else None,
            **self.details,
        }


@dataclass
class CrossValidationResult:
    """
    Result of cross-validating sigma predictions.

    Attributes:
        n_folds: Number of CV folds
        fold_results: Results for each fold
        mean_rmse: Mean RMSE across folds
        std_rmse: Standard deviation of RMSE
        mean_r_squared: Mean R-squared across folds
        overall_valid: Whether all folds passed validation
    """
    n_folds: int = 0
    fold_results: List[Dict[str, Any]] = field(default_factory=list)
    mean_rmse: float = 0.0
    std_rmse: float = 0.0
    mean_r_squared: float = 0.0
    overall_valid: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'n_folds': self.n_folds,
            'fold_results': self.fold_results,
            'mean_rmse': self.mean_rmse,
            'std_rmse': self.std_rmse,
            'mean_r_squared': self.mean_r_squared,
            'overall_valid': self.overall_valid,
        }


def validate_against_engine(
    engine: EngineProtocol,
    duration: float,
    dt: float = 1.0,
    predictor: str = 'mean',
    run_null_tests: bool = True,
) -> ValidationResult:
    """
    Validate omega computation against a specific engine.

    Args:
        engine: Engine instance to validate against
        duration: Simulation duration
        dt: Time step
        predictor: Prediction method to use
        run_null_tests: Whether to run null hypothesis tests

    Returns:
        ValidationResult object
    """
    # Compute omega series from engine
    omega_series = compute_omega_from_engine(
        engine=engine,
        duration=duration,
        dt=dt,
        predictor=predictor,
    )

    n = len(omega_series)

    if n < 5:
        return ValidationResult(
            valid=False,
            n_observations=n,
            issues=['Insufficient data from engine simulation'],
        )

    omega_values = omega_series.omega_values
    sigma_obs = omega_series.sigma_observed_values
    sigma_pred = omega_series.sigma_predicted_values

    # Compute metrics
    mean_omega = float(np.mean(omega_values))
    std_omega = float(np.std(omega_values, ddof=1))

    # RMSE and MAE
    rmse = float(np.sqrt(np.mean(omega_values ** 2)))
    mae = float(np.mean(np.abs(omega_values)))

    # R-squared
    ss_res = np.sum(omega_values ** 2)
    ss_tot = np.sum((sigma_obs - np.mean(sigma_obs)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Run null hypothesis tests
    null_tests = None
    if run_null_tests and n >= 10:
        null_tests = run_null_hypothesis_battery(omega_series)

    # Distribution stats
    dist_stats = compute_distribution_stats(omega_series)

    return ValidationResult(
        valid=True,
        domain=omega_series.domain,
        n_observations=n,
        mean_omega=mean_omega,
        std_omega=std_omega,
        rmse=rmse,
        mae=mae,
        r_squared=float(r_squared),
        null_tests=null_tests,
        distribution_stats=dist_stats,
        details={
            'predictor': predictor,
            'duration': duration,
            'dt': dt,
            'sigma_range': (float(np.min(sigma_obs)), float(np.max(sigma_obs))),
        },
    )


def cross_validate_predictions(
    sigma_series: np.ndarray,
    n_folds: int = 5,
    predictor: str = 'mean',
) -> CrossValidationResult:
    """
    Cross-validate sigma predictions.

    Args:
        sigma_series: Array of sigma values
        n_folds: Number of cross-validation folds
        predictor: Prediction method

    Returns:
        CrossValidationResult object
    """
    n = len(sigma_series)

    if n < n_folds * 5:
        return CrossValidationResult(
            n_folds=0,
            overall_valid=False,
            fold_results=[{'error': 'Insufficient data for cross-validation'}],
        )

    fold_size = n // n_folds
    fold_results = []
    rmse_values = []
    r_squared_values = []

    for fold in range(n_folds):
        # Define test set
        test_start = fold * fold_size
        test_end = (fold + 1) * fold_size if fold < n_folds - 1 else n

        # Train on everything except test set
        train_indices = list(range(0, test_start)) + list(range(test_end, n))
        test_indices = list(range(test_start, test_end))

        if len(train_indices) < 5 or len(test_indices) < 2:
            continue

        train_data = sigma_series[train_indices]
        test_data = sigma_series[test_indices]

        # Compute predictions
        if predictor == 'mean':
            predictions = np.full(len(test_data), np.mean(train_data))
        elif predictor == 'last':
            predictions = np.full(len(test_data), train_data[-1])
        else:
            predictions = np.full(len(test_data), np.mean(train_data))

        # Compute metrics
        omega = test_data - predictions
        rmse = float(np.sqrt(np.mean(omega ** 2)))
        mae = float(np.mean(np.abs(omega)))

        ss_res = np.sum(omega ** 2)
        ss_tot = np.sum((test_data - np.mean(test_data)) ** 2)
        r_sq = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        rmse_values.append(rmse)
        r_squared_values.append(r_sq)

        fold_results.append({
            'fold': fold,
            'train_size': len(train_indices),
            'test_size': len(test_indices),
            'rmse': rmse,
            'mae': mae,
            'r_squared': float(r_sq),
        })

    if len(rmse_values) == 0:
        return CrossValidationResult(
            n_folds=0,
            overall_valid=False,
        )

    return CrossValidationResult(
        n_folds=len(fold_results),
        fold_results=fold_results,
        mean_rmse=float(np.mean(rmse_values)),
        std_rmse=float(np.std(rmse_values)),
        mean_r_squared=float(np.mean(r_squared_values)),
        overall_valid=True,
    )


def validate_omega_consistency(
    omega_series: OmegaTimeSeries,
) -> Dict[str, Any]:
    """
    Validate internal consistency of omega series.

    Checks:
        - omega = sigma_obs - sigma_pred for all observations
        - No NaN or Inf values
        - Timestamps are monotonic
        - Values are in reasonable range

    Args:
        omega_series: OmegaTimeSeries to validate

    Returns:
        Dictionary with validation results
    """
    issues = []
    warnings = []

    if len(omega_series) == 0:
        return {
            'valid': False,
            'issues': ['Empty omega series'],
            'warnings': [],
        }

    omega_values = omega_series.omega_values
    sigma_obs = omega_series.sigma_observed_values
    sigma_pred = omega_series.sigma_predicted_values
    timestamps = omega_series.timestamps

    # Check omega computation
    expected_omega = sigma_obs - sigma_pred
    if not np.allclose(omega_values, expected_omega, rtol=1e-6):
        issues.append('Omega values do not match sigma_obs - sigma_pred')

    # Check for NaN/Inf
    if np.any(np.isnan(omega_values)):
        issues.append(f'{np.sum(np.isnan(omega_values))} NaN values in omega')

    if np.any(np.isinf(omega_values)):
        issues.append(f'{np.sum(np.isinf(omega_values))} Inf values in omega')

    # Check timestamps are monotonic
    if len(timestamps) > 1:
        if not np.all(np.diff(timestamps) >= 0):
            warnings.append('Timestamps are not monotonically increasing')

    # Check sigma values are in [0, 1] range
    if np.any(sigma_obs < 0) or np.any(sigma_obs > 1):
        warnings.append('Some sigma_observed values outside [0, 1] range')

    # Check omega magnitude
    if np.any(np.abs(omega_values) > 1):
        warnings.append(f'{np.sum(np.abs(omega_values) > 1)} omega values with |omega| > 1')

    return {
        'valid': len(issues) == 0,
        'n_observations': len(omega_series),
        'issues': issues,
        'warnings': warnings,
        'statistics': {
            'mean_omega': float(np.mean(omega_values)),
            'std_omega': float(np.std(omega_values, ddof=1)) if len(omega_values) > 1 else 0,
            'min_omega': float(np.min(omega_values)),
            'max_omega': float(np.max(omega_values)),
        },
    }


def compare_predictors(
    sigma_series: np.ndarray,
    predictors: Optional[List[str]] = None,
    warmup: int = 10,
) -> pd.DataFrame:
    """
    Compare different prediction methods on sigma series.

    Args:
        sigma_series: Array of sigma values
        predictors: List of predictor names to compare
        warmup: Number of observations to use for warmup

    Returns:
        DataFrame comparing predictor performance
    """
    if predictors is None:
        predictors = ['mean', 'median', 'last', 'exp_smooth']

    results = []

    for predictor in predictors:
        omega_series = compute_omega_series(
            sigma_series=sigma_series,
            predictor=predictor,
            warmup=warmup,
        )

        if len(omega_series) < 5:
            continue

        omega_values = omega_series.omega_values

        results.append({
            'predictor': predictor,
            'n': len(omega_series),
            'mean_omega': float(np.mean(omega_values)),
            'std_omega': float(np.std(omega_values, ddof=1)),
            'rmse': float(np.sqrt(np.mean(omega_values ** 2))),
            'mae': float(np.mean(np.abs(omega_values))),
            'max_error': float(np.max(np.abs(omega_values))),
        })

    return pd.DataFrame(results)


__all__ = [
    'ValidationResult',
    'CrossValidationResult',
    'validate_against_engine',
    'cross_validate_predictions',
    'validate_omega_consistency',
    'compare_predictors',
]
