"""
RATCHET Omega Module - Distribution Analysis

Analyzes the distribution of omega residuals to detect non-random structure.

Key metrics:
    - Skewness: asymmetry of the distribution
    - Kurtosis: tail heaviness (excess kurtosis)
    - Distribution fitting: fit known distributions to omega
    - Comparison to null: compare to expected noise distribution
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from scipy import stats
from scipy.stats import (
    norm, t as t_dist, laplace, cauchy,
    skew, kurtosis, kstest, cramervonmises,
)

from .residuals import OmegaTimeSeries


@dataclass
class DistributionStats:
    """
    Distributional statistics for omega series.

    Attributes:
        n: Number of observations
        mean: Mean of omega
        std: Standard deviation
        median: Median
        skewness: Skewness (Fisher's definition)
        kurtosis: Excess kurtosis
        min: Minimum value
        max: Maximum value
        q1: 25th percentile
        q3: 75th percentile
        iqr: Interquartile range
        normality_p: p-value from normality test
    """
    n: int = 0
    mean: float = 0.0
    std: float = 0.0
    median: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    min: float = 0.0
    max: float = 0.0
    q1: float = 0.0
    q3: float = 0.0
    iqr: float = 0.0
    normality_p: float = 1.0

    @property
    def skewness_interpretation(self) -> str:
        """Interpret skewness value."""
        if abs(self.skewness) < 0.5:
            return 'approximately symmetric'
        elif self.skewness > 0:
            return 'right-skewed (positive tail)'
        else:
            return 'left-skewed (negative tail)'

    @property
    def kurtosis_interpretation(self) -> str:
        """Interpret excess kurtosis value."""
        if abs(self.kurtosis) < 0.5:
            return 'mesokurtic (normal-like tails)'
        elif self.kurtosis > 0:
            return 'leptokurtic (heavy tails)'
        else:
            return 'platykurtic (light tails)'

    @property
    def is_approximately_normal(self) -> bool:
        """Check if distribution is approximately normal."""
        return (
            abs(self.skewness) < 0.5 and
            abs(self.kurtosis) < 1.0 and
            self.normality_p > 0.05
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'n': self.n,
            'mean': self.mean,
            'std': self.std,
            'median': self.median,
            'skewness': self.skewness,
            'kurtosis': self.kurtosis,
            'min': self.min,
            'max': self.max,
            'q1': self.q1,
            'q3': self.q3,
            'iqr': self.iqr,
            'normality_p': self.normality_p,
            'skewness_interpretation': self.skewness_interpretation,
            'kurtosis_interpretation': self.kurtosis_interpretation,
            'is_approximately_normal': self.is_approximately_normal,
        }


@dataclass
class FittedDistribution:
    """
    Result of fitting a distribution to omega data.

    Attributes:
        distribution: Name of the distribution
        parameters: Fitted parameters
        ks_statistic: Kolmogorov-Smirnov statistic
        ks_p_value: p-value from KS test
        aic: Akaike Information Criterion
        bic: Bayesian Information Criterion
        good_fit: Whether the fit is acceptable (p > 0.05)
    """
    distribution: str
    parameters: Dict[str, float] = field(default_factory=dict)
    ks_statistic: float = 0.0
    ks_p_value: float = 0.0
    aic: float = float('inf')
    bic: float = float('inf')
    good_fit: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'distribution': self.distribution,
            'parameters': self.parameters,
            'ks_statistic': self.ks_statistic,
            'ks_p_value': self.ks_p_value,
            'aic': self.aic,
            'bic': self.bic,
            'good_fit': self.good_fit,
        }


def compute_distribution_stats(
    omega: Union[np.ndarray, OmegaTimeSeries],
) -> DistributionStats:
    """
    Compute comprehensive distribution statistics for omega.

    Args:
        omega: Omega values (array or OmegaTimeSeries)

    Returns:
        DistributionStats object
    """
    if isinstance(omega, OmegaTimeSeries):
        values = omega.omega_values
    else:
        values = np.asarray(omega)

    n = len(values)

    if n == 0:
        return DistributionStats()

    if n < 3:
        return DistributionStats(
            n=n,
            mean=float(np.mean(values)),
            std=0.0,
            median=float(np.median(values)),
            min=float(np.min(values)),
            max=float(np.max(values)),
        )

    # Basic stats
    mean_val = float(np.mean(values))
    std_val = float(np.std(values, ddof=1))
    median_val = float(np.median(values))

    # Percentiles
    q1 = float(np.percentile(values, 25))
    q3 = float(np.percentile(values, 75))
    iqr = q3 - q1

    # Higher moments
    skewness_val = float(skew(values))
    kurtosis_val = float(kurtosis(values))  # Fisher's definition (excess kurtosis)

    # Normality test
    if n >= 8:
        try:
            _, normality_p = stats.shapiro(values[:5000])  # Limit for Shapiro
        except Exception:
            normality_p = 1.0
    else:
        normality_p = 1.0

    return DistributionStats(
        n=n,
        mean=mean_val,
        std=std_val,
        median=median_val,
        skewness=skewness_val,
        kurtosis=kurtosis_val,
        min=float(np.min(values)),
        max=float(np.max(values)),
        q1=q1,
        q3=q3,
        iqr=iqr,
        normality_p=float(normality_p),
    )


def fit_distribution(
    omega: Union[np.ndarray, OmegaTimeSeries],
    distribution: str = 'norm',
) -> FittedDistribution:
    """
    Fit a specified distribution to omega data.

    Args:
        omega: Omega values (array or OmegaTimeSeries)
        distribution: Distribution name ('norm', 't', 'laplace', 'cauchy')

    Returns:
        FittedDistribution object
    """
    if isinstance(omega, OmegaTimeSeries):
        values = omega.omega_values
    else:
        values = np.asarray(omega)

    n = len(values)

    if n < 5:
        return FittedDistribution(
            distribution=distribution,
            parameters={'error': 'Insufficient data'},
        )

    # Map distribution names to scipy distributions
    dist_map = {
        'norm': norm,
        'normal': norm,
        't': t_dist,
        'student_t': t_dist,
        'laplace': laplace,
        'cauchy': cauchy,
    }

    if distribution not in dist_map:
        return FittedDistribution(
            distribution=distribution,
            parameters={'error': f'Unknown distribution: {distribution}'},
        )

    dist = dist_map[distribution]

    try:
        # Fit the distribution
        params = dist.fit(values)

        # Create parameter dictionary
        if distribution in ('norm', 'normal'):
            param_dict = {'loc': params[0], 'scale': params[1]}
        elif distribution in ('t', 'student_t'):
            param_dict = {'df': params[0], 'loc': params[1], 'scale': params[2]}
        elif distribution == 'laplace':
            param_dict = {'loc': params[0], 'scale': params[1]}
        elif distribution == 'cauchy':
            param_dict = {'loc': params[0], 'scale': params[1]}
        else:
            param_dict = {f'param_{i}': p for i, p in enumerate(params)}

        # KS test for goodness of fit
        ks_stat, ks_p = kstest(values, distribution, args=params)

        # Log-likelihood for AIC/BIC
        log_lik = np.sum(dist.logpdf(values, *params))
        k = len(params)  # Number of parameters
        aic = 2 * k - 2 * log_lik
        bic = k * np.log(n) - 2 * log_lik

        return FittedDistribution(
            distribution=distribution,
            parameters=param_dict,
            ks_statistic=float(ks_stat),
            ks_p_value=float(ks_p),
            aic=float(aic),
            bic=float(bic),
            good_fit=ks_p > 0.05,
        )

    except Exception as e:
        return FittedDistribution(
            distribution=distribution,
            parameters={'error': str(e)},
        )


def fit_best_distribution(
    omega: Union[np.ndarray, OmegaTimeSeries],
    candidates: Optional[List[str]] = None,
) -> Tuple[FittedDistribution, List[FittedDistribution]]:
    """
    Fit multiple distributions and select the best one.

    Args:
        omega: Omega values (array or OmegaTimeSeries)
        candidates: List of distribution names to try

    Returns:
        Tuple of (best_fit, all_fits)
    """
    if candidates is None:
        candidates = ['norm', 't', 'laplace']

    fits = []
    for dist in candidates:
        fit = fit_distribution(omega, dist)
        fits.append(fit)

    # Sort by BIC (lower is better)
    fits.sort(key=lambda f: f.bic if f.bic != float('inf') else 1e10)

    return fits[0], fits


def compare_to_null_distribution(
    omega: Union[np.ndarray, OmegaTimeSeries],
    null_mean: float = 0.0,
    null_std: Optional[float] = None,
    n_simulations: int = 1000,
) -> Dict[str, Any]:
    """
    Compare omega distribution to expected null distribution.

    Under H0 (CCA is curve-fitting), omega should be:
    - Centered at zero (mean ~ 0)
    - Normally distributed
    - Have no temporal structure

    Args:
        omega: Omega values (array or OmegaTimeSeries)
        null_mean: Expected mean under null (usually 0)
        null_std: Expected std under null (estimated from data if None)
        n_simulations: Number of null simulations for p-value estimation

    Returns:
        Dictionary with comparison results
    """
    if isinstance(omega, OmegaTimeSeries):
        values = omega.omega_values
    else:
        values = np.asarray(omega)

    n = len(values)

    if n < 5:
        return {'error': 'Insufficient data'}

    # Observed statistics
    obs_mean = np.mean(values)
    obs_std = np.std(values, ddof=1)
    obs_skew = skew(values)
    obs_kurt = kurtosis(values)

    # Use observed std if null_std not specified
    if null_std is None:
        null_std = obs_std

    # Simulate null distribution
    np.random.seed(42)  # For reproducibility
    null_means = []
    null_skews = []
    null_kurts = []

    for _ in range(n_simulations):
        null_sample = np.random.normal(null_mean, null_std, n)
        null_means.append(np.mean(null_sample))
        null_skews.append(skew(null_sample))
        null_kurts.append(kurtosis(null_sample))

    # Compute empirical p-values
    p_mean = np.mean(np.abs(np.array(null_means)) >= np.abs(obs_mean))
    p_skew = np.mean(np.abs(np.array(null_skews)) >= np.abs(obs_skew))
    p_kurt = np.mean(np.abs(np.array(null_kurts)) >= np.abs(obs_kurt))

    # KS test against null normal
    z_scores = (values - null_mean) / null_std
    ks_stat, ks_p = kstest(z_scores, 'norm')

    # Overall assessment
    tests_failed = sum([
        p_mean < 0.05,
        p_skew < 0.05,
        p_kurt < 0.05,
        ks_p < 0.05,
    ])

    return {
        'n': n,
        'observed': {
            'mean': float(obs_mean),
            'std': float(obs_std),
            'skewness': float(obs_skew),
            'kurtosis': float(obs_kurt),
        },
        'null': {
            'mean': null_mean,
            'std': null_std,
        },
        'p_values': {
            'mean': float(p_mean),
            'skewness': float(p_skew),
            'kurtosis': float(p_kurt),
            'ks_test': float(ks_p),
        },
        'ks_statistic': float(ks_stat),
        'tests_failed': tests_failed,
        'consistent_with_null': tests_failed == 0,
        'interpretation': (
            'Omega is consistent with null distribution (random noise)'
            if tests_failed == 0
            else f'Omega deviates from null in {tests_failed}/4 tests'
        ),
    }


def compute_tail_statistics(
    omega: Union[np.ndarray, OmegaTimeSeries],
    percentile_low: float = 5.0,
    percentile_high: float = 95.0,
) -> Dict[str, Any]:
    """
    Analyze the tails of the omega distribution.

    Args:
        omega: Omega values (array or OmegaTimeSeries)
        percentile_low: Lower percentile for tail analysis
        percentile_high: Upper percentile for tail analysis

    Returns:
        Dictionary with tail statistics
    """
    if isinstance(omega, OmegaTimeSeries):
        values = omega.omega_values
    else:
        values = np.asarray(omega)

    n = len(values)

    if n < 10:
        return {'error': 'Insufficient data for tail analysis'}

    # Percentiles
    p_low = np.percentile(values, percentile_low)
    p_high = np.percentile(values, percentile_high)

    # Values in tails
    left_tail = values[values <= p_low]
    right_tail = values[values >= p_high]

    # Expected count under normal
    expected_tail_fraction = (percentile_low + (100 - percentile_high)) / 100
    expected_tail_count = n * expected_tail_fraction / 2

    # Tail ratio (observed / expected)
    left_tail_ratio = len(left_tail) / expected_tail_count if expected_tail_count > 0 else 1.0
    right_tail_ratio = len(right_tail) / expected_tail_count if expected_tail_count > 0 else 1.0

    # Hill estimator for tail index (if enough extreme values)
    extreme_values = np.abs(values)[np.abs(values) > np.percentile(np.abs(values), 90)]
    if len(extreme_values) >= 10:
        extreme_values = np.sort(extreme_values)[::-1]
        k = len(extreme_values) // 5  # Use top 20% of extremes
        if k >= 2:
            hill_estimate = k / np.sum(np.log(extreme_values[:k] / extreme_values[k]))
        else:
            hill_estimate = None
    else:
        hill_estimate = None

    return {
        'percentile_low': percentile_low,
        'percentile_high': percentile_high,
        'threshold_low': float(p_low),
        'threshold_high': float(p_high),
        'left_tail_count': len(left_tail),
        'right_tail_count': len(right_tail),
        'left_tail_mean': float(np.mean(left_tail)) if len(left_tail) > 0 else None,
        'right_tail_mean': float(np.mean(right_tail)) if len(right_tail) > 0 else None,
        'left_tail_ratio': float(left_tail_ratio),
        'right_tail_ratio': float(right_tail_ratio),
        'hill_tail_index': float(hill_estimate) if hill_estimate else None,
        'heavy_tails': (left_tail_ratio > 1.5 or right_tail_ratio > 1.5),
    }


__all__ = [
    'DistributionStats',
    'FittedDistribution',
    'compute_distribution_stats',
    'fit_distribution',
    'fit_best_distribution',
    'compare_to_null_distribution',
    'compute_tail_statistics',
]
