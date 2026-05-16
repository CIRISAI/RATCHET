"""
RATCHET Omega Module - Null Hypothesis Testing

Statistical tests for the null hypothesis:
    H0: If CCA is just curve-fitting, mean(omega) ~ 0 with high variance

Tests include:
    - t-test: Test if mean(omega) = 0
    - Shapiro-Wilk / Anderson-Darling: Test for normality
    - Ljung-Box: Test for autocorrelation in omega series
    - Runs test: Test for randomness

If H0 is rejected, this suggests omega has detectable structure,
supporting the alternative hypothesis that CCA captures real dynamics.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum

from scipy import stats
from scipy.stats import (
    ttest_1samp,
    shapiro,
    anderson,
    normaltest,
    jarque_bera,
    kstest,
)

try:
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.stats.stattools import durbin_watson
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

from .residuals import OmegaTimeSeries


class TestType(Enum):
    """Types of null hypothesis tests."""
    MEAN_ZERO = "mean_zero"
    NORMALITY_SHAPIRO = "normality_shapiro"
    NORMALITY_ANDERSON = "normality_anderson"
    NORMALITY_DAGOSTINO = "normality_dagostino"
    NORMALITY_JARQUE_BERA = "normality_jarque_bera"
    AUTOCORRELATION_LJUNG_BOX = "autocorrelation_ljung_box"
    AUTOCORRELATION_DURBIN_WATSON = "autocorrelation_durbin_watson"
    RUNS_TEST = "runs_test"


@dataclass
class NullTestResult:
    """
    Result of a null hypothesis test on omega series.

    Attributes:
        test_type: Type of test performed
        statistic: Test statistic value
        p_value: p-value of the test
        reject_null: Whether to reject H0 at given alpha
        alpha: Significance level
        interpretation: Human-readable interpretation
        details: Additional test-specific details
    """
    test_type: TestType
    statistic: float
    p_value: float
    reject_null: bool
    alpha: float = 0.05
    interpretation: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Set interpretation if not provided."""
        if not self.interpretation:
            self.interpretation = self._generate_interpretation()

    def _generate_interpretation(self) -> str:
        """Generate human-readable interpretation."""
        reject_str = "REJECTED" if self.reject_null else "NOT REJECTED"

        if self.test_type == TestType.MEAN_ZERO:
            return (
                f"H0 (mean omega = 0) {reject_str} at alpha={self.alpha}. "
                f"t-statistic={self.statistic:.3f}, p-value={self.p_value:.4f}. "
                f"{'Omega shows systematic bias.' if self.reject_null else 'Omega is centered around zero.'}"
            )
        elif self.test_type in (TestType.NORMALITY_SHAPIRO, TestType.NORMALITY_ANDERSON,
                                 TestType.NORMALITY_DAGOSTINO, TestType.NORMALITY_JARQUE_BERA):
            return (
                f"H0 (omega is normal) {reject_str} at alpha={self.alpha}. "
                f"statistic={self.statistic:.3f}, p-value={self.p_value:.4f}. "
                f"{'Omega deviates from normality.' if self.reject_null else 'Omega is approximately normal.'}"
            )
        elif self.test_type in (TestType.AUTOCORRELATION_LJUNG_BOX, TestType.AUTOCORRELATION_DURBIN_WATSON):
            return (
                f"H0 (no autocorrelation) {reject_str} at alpha={self.alpha}. "
                f"statistic={self.statistic:.3f}, p-value={self.p_value:.4f}. "
                f"{'Omega shows temporal structure.' if self.reject_null else 'Omega appears uncorrelated.'}"
            )
        elif self.test_type == TestType.RUNS_TEST:
            return (
                f"H0 (omega is random) {reject_str} at alpha={self.alpha}. "
                f"Z-statistic={self.statistic:.3f}, p-value={self.p_value:.4f}. "
                f"{'Omega shows non-random pattern.' if self.reject_null else 'Omega appears random.'}"
            )
        else:
            return f"Test {self.test_type.value}: statistic={self.statistic:.3f}, p={self.p_value:.4f}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'test_type': self.test_type.value,
            'statistic': self.statistic,
            'p_value': self.p_value,
            'reject_null': self.reject_null,
            'alpha': self.alpha,
            'interpretation': self.interpretation,
            **self.details,
        }


# =============================================================================
# MEAN TESTS
# =============================================================================

def test_mean_zero(
    omega: Union[np.ndarray, OmegaTimeSeries],
    alpha: float = 0.05,
    alternative: str = 'two-sided',
) -> NullTestResult:
    """
    Test H0: mean(omega) = 0 using one-sample t-test.

    Args:
        omega: Omega values (array or OmegaTimeSeries)
        alpha: Significance level
        alternative: 'two-sided', 'less', or 'greater'

    Returns:
        NullTestResult object
    """
    if isinstance(omega, OmegaTimeSeries):
        omega_values = omega.omega_values
    else:
        omega_values = np.asarray(omega)

    if len(omega_values) < 2:
        return NullTestResult(
            test_type=TestType.MEAN_ZERO,
            statistic=0.0,
            p_value=1.0,
            reject_null=False,
            alpha=alpha,
            details={'error': 'Insufficient data (n < 2)'},
        )

    # Use try/except for scipy version compatibility
    try:
        t_stat, p_value = ttest_1samp(omega_values, 0.0, alternative=alternative)
    except TypeError:
        # Older scipy version without 'alternative' parameter
        t_stat, p_value = ttest_1samp(omega_values, 0.0)

    return NullTestResult(
        test_type=TestType.MEAN_ZERO,
        statistic=float(t_stat),
        p_value=float(p_value),
        reject_null=p_value < alpha,
        alpha=alpha,
        details={
            'n': len(omega_values),
            'mean_omega': float(np.mean(omega_values)),
            'std_omega': float(np.std(omega_values, ddof=1)),
            'se_mean': float(np.std(omega_values, ddof=1) / np.sqrt(len(omega_values))),
            'alternative': alternative,
        },
    )


# =============================================================================
# NORMALITY TESTS
# =============================================================================

def test_normality(
    omega: Union[np.ndarray, OmegaTimeSeries],
    method: str = 'shapiro',
    alpha: float = 0.05,
) -> NullTestResult:
    """
    Test H0: omega follows a normal distribution.

    Args:
        omega: Omega values (array or OmegaTimeSeries)
        method: 'shapiro', 'anderson', 'dagostino', 'jarque_bera', 'ks'
        alpha: Significance level

    Returns:
        NullTestResult object
    """
    if isinstance(omega, OmegaTimeSeries):
        omega_values = omega.omega_values
    else:
        omega_values = np.asarray(omega)

    n = len(omega_values)

    if n < 3:
        return NullTestResult(
            test_type=TestType.NORMALITY_SHAPIRO,
            statistic=0.0,
            p_value=1.0,
            reject_null=False,
            alpha=alpha,
            details={'error': 'Insufficient data (n < 3)'},
        )

    if method == 'shapiro':
        # Shapiro-Wilk test (best for n < 5000)
        if n > 5000:
            omega_values = omega_values[:5000]  # Limit for Shapiro
        stat, p_value = shapiro(omega_values)
        test_type = TestType.NORMALITY_SHAPIRO

    elif method == 'anderson':
        # Anderson-Darling test
        result = anderson(omega_values, dist='norm')
        stat = result.statistic
        # Get critical value for alpha
        # Anderson critical values are for [15%, 10%, 5%, 2.5%, 1%]
        alpha_map = {0.15: 0, 0.10: 1, 0.05: 2, 0.025: 3, 0.01: 4}
        idx = alpha_map.get(alpha, 2)  # Default to 5%
        critical = result.critical_values[idx]
        # Approximate p-value (Anderson doesn't give exact p-value)
        p_value = 0.0 if stat > critical else 1.0
        test_type = TestType.NORMALITY_ANDERSON

    elif method == 'dagostino':
        # D'Agostino-Pearson test (requires n >= 20)
        if n < 20:
            return NullTestResult(
                test_type=TestType.NORMALITY_DAGOSTINO,
                statistic=0.0,
                p_value=1.0,
                reject_null=False,
                alpha=alpha,
                details={'error': 'Insufficient data for D\'Agostino (n < 20)'},
            )
        stat, p_value = normaltest(omega_values)
        test_type = TestType.NORMALITY_DAGOSTINO

    elif method == 'jarque_bera':
        # Jarque-Bera test
        stat, p_value = jarque_bera(omega_values)
        test_type = TestType.NORMALITY_JARQUE_BERA

    elif method == 'ks':
        # Kolmogorov-Smirnov test against normal
        # Standardize data
        z = (omega_values - np.mean(omega_values)) / np.std(omega_values, ddof=1)
        stat, p_value = kstest(z, 'norm')
        test_type = TestType.NORMALITY_SHAPIRO  # Use SHAPIRO as generic

    else:
        raise ValueError(f"Unknown normality test method: {method}")

    # Compute additional statistics
    skewness = float(stats.skew(omega_values))
    kurtosis = float(stats.kurtosis(omega_values))

    return NullTestResult(
        test_type=test_type,
        statistic=float(stat),
        p_value=float(p_value),
        reject_null=p_value < alpha,
        alpha=alpha,
        details={
            'n': n,
            'method': method,
            'skewness': skewness,
            'kurtosis': kurtosis,
        },
    )


# =============================================================================
# AUTOCORRELATION TESTS
# =============================================================================

def test_autocorrelation(
    omega: Union[np.ndarray, OmegaTimeSeries],
    method: str = 'ljung_box',
    lags: Optional[int] = None,
    alpha: float = 0.05,
) -> NullTestResult:
    """
    Test H0: No autocorrelation in omega series.

    Args:
        omega: Omega values (array or OmegaTimeSeries)
        method: 'ljung_box' or 'durbin_watson'
        lags: Number of lags for Ljung-Box (default: min(10, n//5))
        alpha: Significance level

    Returns:
        NullTestResult object
    """
    if isinstance(omega, OmegaTimeSeries):
        omega_values = omega.omega_values
    else:
        omega_values = np.asarray(omega)

    n = len(omega_values)

    if n < 5:
        return NullTestResult(
            test_type=TestType.AUTOCORRELATION_LJUNG_BOX,
            statistic=0.0,
            p_value=1.0,
            reject_null=False,
            alpha=alpha,
            details={'error': 'Insufficient data (n < 5)'},
        )

    if method == 'ljung_box':
        if not HAS_STATSMODELS:
            return NullTestResult(
                test_type=TestType.AUTOCORRELATION_LJUNG_BOX,
                statistic=0.0,
                p_value=1.0,
                reject_null=False,
                alpha=alpha,
                details={'error': 'statsmodels not available'},
            )

        if lags is None:
            lags = min(10, n // 5)
            lags = max(1, lags)

        try:
            result = acorr_ljungbox(omega_values, lags=[lags], return_df=True)
            # New API returns DataFrame
            stat = float(result['lb_stat'].iloc[0])
            p_value = float(result['lb_pvalue'].iloc[0])
        except (TypeError, KeyError, AttributeError):
            try:
                # Try older API
                result = acorr_ljungbox(omega_values, lags=[lags], return_df=False)
                stat = float(result[0][0]) if hasattr(result[0], '__getitem__') else float(result[0])
                p_value = float(result[1][0]) if hasattr(result[1], '__getitem__') else float(result[1])
            except Exception:
                stat = 0.0
                p_value = 1.0

        return NullTestResult(
            test_type=TestType.AUTOCORRELATION_LJUNG_BOX,
            statistic=stat,
            p_value=p_value,
            reject_null=p_value < alpha,
            alpha=alpha,
            details={
                'n': n,
                'lags': lags,
                'method': 'ljung_box',
            },
        )

    elif method == 'durbin_watson':
        if not HAS_STATSMODELS:
            return NullTestResult(
                test_type=TestType.AUTOCORRELATION_DURBIN_WATSON,
                statistic=0.0,
                p_value=1.0,
                reject_null=False,
                alpha=alpha,
                details={'error': 'statsmodels not available'},
            )

        dw_stat = durbin_watson(omega_values)
        # DW statistic ranges from 0 to 4
        # DW ~ 2: no autocorrelation
        # DW < 2: positive autocorrelation
        # DW > 2: negative autocorrelation

        # Approximate p-value (rough heuristic)
        # More precise would require critical value tables
        deviation = abs(dw_stat - 2)
        if deviation < 0.5:
            p_value = 0.5  # Likely no autocorrelation
        elif deviation < 1.0:
            p_value = 0.1
        else:
            p_value = 0.01

        return NullTestResult(
            test_type=TestType.AUTOCORRELATION_DURBIN_WATSON,
            statistic=float(dw_stat),
            p_value=p_value,
            reject_null=p_value < alpha,
            alpha=alpha,
            details={
                'n': n,
                'method': 'durbin_watson',
                'interpretation': 'DW ~ 2 means no autocorrelation',
            },
        )

    else:
        raise ValueError(f"Unknown autocorrelation test method: {method}")


# =============================================================================
# RUNS TEST
# =============================================================================

def test_runs(
    omega: Union[np.ndarray, OmegaTimeSeries],
    alpha: float = 0.05,
) -> NullTestResult:
    """
    Wald-Wolfowitz runs test for randomness.

    Tests whether the sequence of positive/negative omega values is random.

    Args:
        omega: Omega values (array or OmegaTimeSeries)
        alpha: Significance level

    Returns:
        NullTestResult object
    """
    if isinstance(omega, OmegaTimeSeries):
        omega_values = omega.omega_values
    else:
        omega_values = np.asarray(omega)

    n = len(omega_values)

    if n < 10:
        return NullTestResult(
            test_type=TestType.RUNS_TEST,
            statistic=0.0,
            p_value=1.0,
            reject_null=False,
            alpha=alpha,
            details={'error': 'Insufficient data (n < 10)'},
        )

    # Convert to binary: above/below median
    median = np.median(omega_values)
    binary = (omega_values > median).astype(int)

    # Count runs
    runs = 1
    for i in range(1, n):
        if binary[i] != binary[i - 1]:
            runs += 1

    # Number of positive and negative values
    n_pos = np.sum(binary)
    n_neg = n - n_pos

    if n_pos == 0 or n_neg == 0:
        return NullTestResult(
            test_type=TestType.RUNS_TEST,
            statistic=0.0,
            p_value=1.0,
            reject_null=False,
            alpha=alpha,
            details={'error': 'All values on same side of median'},
        )

    # Expected number of runs and variance under H0
    expected_runs = (2 * n_pos * n_neg) / n + 1
    var_runs = (2 * n_pos * n_neg * (2 * n_pos * n_neg - n)) / (n ** 2 * (n - 1))

    if var_runs <= 0:
        return NullTestResult(
            test_type=TestType.RUNS_TEST,
            statistic=0.0,
            p_value=1.0,
            reject_null=False,
            alpha=alpha,
            details={'error': 'Zero variance in runs test'},
        )

    # Z-statistic
    z_stat = (runs - expected_runs) / np.sqrt(var_runs)
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    return NullTestResult(
        test_type=TestType.RUNS_TEST,
        statistic=float(z_stat),
        p_value=float(p_value),
        reject_null=p_value < alpha,
        alpha=alpha,
        details={
            'n': n,
            'runs': runs,
            'expected_runs': expected_runs,
            'n_positive': n_pos,
            'n_negative': n_neg,
        },
    )


# =============================================================================
# COMPREHENSIVE TEST BATTERY
# =============================================================================

@dataclass
class NullHypothesisBattery:
    """
    Results from running a battery of null hypothesis tests.

    Attributes:
        results: Dictionary of test results
        summary: Summary statistics
        overall_reject: Whether to reject H0 based on any test
        overall_interpretation: Combined interpretation
    """
    results: Dict[str, NullTestResult] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)
    overall_reject: bool = False
    overall_interpretation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'results': {k: v.to_dict() for k, v in self.results.items()},
            'summary': self.summary,
            'overall_reject': self.overall_reject,
            'overall_interpretation': self.overall_interpretation,
        }


def run_null_hypothesis_battery(
    omega: Union[np.ndarray, OmegaTimeSeries],
    alpha: float = 0.05,
    tests: Optional[List[str]] = None,
) -> NullHypothesisBattery:
    """
    Run a battery of null hypothesis tests on omega series.

    Args:
        omega: Omega values (array or OmegaTimeSeries)
        alpha: Significance level
        tests: List of tests to run. If None, runs all available.
               Options: 'mean_zero', 'normality', 'autocorrelation', 'runs'

    Returns:
        NullHypothesisBattery object with all test results
    """
    if tests is None:
        tests = ['mean_zero', 'normality', 'autocorrelation', 'runs']

    if isinstance(omega, OmegaTimeSeries):
        omega_values = omega.omega_values
        n = len(omega)
    else:
        omega_values = np.asarray(omega)
        n = len(omega_values)

    results = {}
    rejections = []

    # Mean test
    if 'mean_zero' in tests:
        result = test_mean_zero(omega_values, alpha=alpha)
        results['mean_zero'] = result
        rejections.append(('mean_zero', result.reject_null))

    # Normality tests
    if 'normality' in tests:
        # Choose appropriate test based on sample size
        if n < 50:
            result = test_normality(omega_values, method='shapiro', alpha=alpha)
        elif n < 5000:
            result = test_normality(omega_values, method='dagostino', alpha=alpha)
        else:
            result = test_normality(omega_values, method='jarque_bera', alpha=alpha)
        results['normality'] = result
        rejections.append(('normality', result.reject_null))

    # Autocorrelation test
    if 'autocorrelation' in tests and HAS_STATSMODELS:
        result = test_autocorrelation(omega_values, method='ljung_box', alpha=alpha)
        results['autocorrelation'] = result
        rejections.append(('autocorrelation', result.reject_null))

    # Runs test
    if 'runs' in tests:
        result = test_runs(omega_values, alpha=alpha)
        results['runs'] = result
        rejections.append(('runs', result.reject_null))

    # Compile summary
    n_tests = len(rejections)
    n_rejected = sum(1 for _, r in rejections if r)
    any_rejected = any(r for _, r in rejections)

    summary = {
        'n_observations': n,
        'n_tests': n_tests,
        'n_rejected': n_rejected,
        'fraction_rejected': n_rejected / n_tests if n_tests > 0 else 0,
        'alpha': alpha,
        'mean_omega': float(np.mean(omega_values)),
        'std_omega': float(np.std(omega_values, ddof=1)) if n > 1 else 0,
        'rejected_tests': [name for name, r in rejections if r],
    }

    # Generate overall interpretation
    if n_rejected == 0:
        interpretation = (
            f"All {n_tests} tests PASSED (H0 not rejected at alpha={alpha}). "
            "Omega residuals are consistent with random noise around zero. "
            "This is consistent with the null hypothesis that CCA is curve-fitting."
        )
    elif n_rejected == n_tests:
        interpretation = (
            f"All {n_tests} tests FAILED (H0 rejected at alpha={alpha}). "
            "Omega residuals show significant structure. "
            "This suggests CCA captures real dynamics beyond curve-fitting."
        )
    else:
        rejected_names = ', '.join(summary['rejected_tests'])
        interpretation = (
            f"{n_rejected}/{n_tests} tests rejected H0 at alpha={alpha} ({rejected_names}). "
            "Omega shows some structure, but results are mixed. "
            "Further investigation recommended."
        )

    return NullHypothesisBattery(
        results=results,
        summary=summary,
        overall_reject=any_rejected,
        overall_interpretation=interpretation,
    )


__all__ = [
    'TestType',
    'NullTestResult',
    'NullHypothesisBattery',
    'test_mean_zero',
    'test_normality',
    'test_autocorrelation',
    'test_runs',
    'run_null_hypothesis_battery',
]
