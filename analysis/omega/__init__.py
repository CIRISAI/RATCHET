"""
RATCHET Omega Module - Sustainability Residual Analysis

This module provides tools for analyzing omega (w) residuals, defined as:
    w = sigma_observed - sigma_predicted

The Omega analysis tests whether CCA (Constraint-Collapse Analysis) captures
real structural dynamics or is merely curve-fitting.

Null Hypothesis (H0): If CCA is just curve-fitting, mean(w) ~ 0 with high variance
Alternative (H1): If CCA captures real dynamics, w shows detectable structure

Key Components:
    - residuals: OmegaObservation dataclass and compute_omega functions
    - null_test: Statistical tests for H0 (t-test, normality, autocorrelation)
    - correlations: Cross-domain omega correlation analysis
    - distribution: Distribution analysis (skew, kurtosis)
    - outliers: Anomalous omega value detection
    - validity: Validation against existing engines
    - report: Generate comprehensive Omega analysis reports
"""

from .residuals import (
    OmegaObservation,
    OmegaTimeSeries,
    compute_omega,
    compute_omega_series,
    sigma_predictor_baseline,
)

from .null_test import (
    NullTestResult,
    test_mean_zero,
    test_normality,
    test_autocorrelation,
    run_null_hypothesis_battery,
)

from .correlations import (
    CorrelationResult,
    compute_cross_domain_correlation,
    compute_granger_causality,
    correlation_matrix,
)

from .distribution import (
    DistributionStats,
    compute_distribution_stats,
    fit_distribution,
    compare_to_null_distribution,
)

from .outliers import (
    OutlierResult,
    detect_outliers_zscore,
    detect_outliers_iqr,
    detect_changepoints,
)

from .validity import (
    ValidationResult,
    validate_against_engine,
    cross_validate_predictions,
)

from .report import (
    OmegaReport,
    generate_omega_report,
)

__all__ = [
    # residuals
    'OmegaObservation',
    'OmegaTimeSeries',
    'compute_omega',
    'compute_omega_series',
    'sigma_predictor_baseline',
    # null_test
    'NullTestResult',
    'test_mean_zero',
    'test_normality',
    'test_autocorrelation',
    'run_null_hypothesis_battery',
    # correlations
    'CorrelationResult',
    'compute_cross_domain_correlation',
    'compute_granger_causality',
    'correlation_matrix',
    # distribution
    'DistributionStats',
    'compute_distribution_stats',
    'fit_distribution',
    'compare_to_null_distribution',
    # outliers
    'OutlierResult',
    'detect_outliers_zscore',
    'detect_outliers_iqr',
    'detect_changepoints',
    # validity
    'ValidationResult',
    'validate_against_engine',
    'cross_validate_predictions',
    # report
    'OmegaReport',
    'generate_omega_report',
]
