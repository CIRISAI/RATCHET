"""
RATCHET Omega Module - Cross-Domain Correlations

Analyzes correlations between omega series from different domains
to detect shared structure or causal relationships.

Key analyses:
    - Pearson/Spearman correlations between omega series
    - Granger causality testing
    - Cross-correlation functions
    - Correlation matrices across domains
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from scipy import stats
from scipy.stats import pearsonr, spearmanr

try:
    from statsmodels.tsa.stattools import grangercausalitytests, ccf
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

from .residuals import OmegaTimeSeries, DomainType


@dataclass
class CorrelationResult:
    """
    Result of a correlation analysis between omega series.

    Attributes:
        correlation: Correlation coefficient
        p_value: p-value for testing H0: correlation = 0
        method: Correlation method used ('pearson' or 'spearman')
        domain_1: First domain
        domain_2: Second domain
        n_observations: Number of paired observations
        significant: Whether correlation is significant at alpha=0.05
        details: Additional details
    """
    correlation: float
    p_value: float
    method: str = 'pearson'
    domain_1: Optional[DomainType] = None
    domain_2: Optional[DomainType] = None
    n_observations: int = 0
    significant: bool = False
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Set significance flag."""
        self.significant = self.p_value < 0.05

    @property
    def strength(self) -> str:
        """Interpret correlation strength."""
        r = abs(self.correlation)
        if r < 0.1:
            return 'negligible'
        elif r < 0.3:
            return 'weak'
        elif r < 0.5:
            return 'moderate'
        elif r < 0.7:
            return 'strong'
        else:
            return 'very_strong'

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'correlation': self.correlation,
            'p_value': self.p_value,
            'method': self.method,
            'domain_1': self.domain_1.value if self.domain_1 else None,
            'domain_2': self.domain_2.value if self.domain_2 else None,
            'n_observations': self.n_observations,
            'significant': self.significant,
            'strength': self.strength,
            **self.details,
        }


@dataclass
class GrangerResult:
    """
    Result of Granger causality test.

    Attributes:
        causes: Whether series_1 Granger-causes series_2
        f_statistic: F-statistic from the test
        p_value: p-value of the test
        lag: Lag used in the test
        domain_cause: Domain of potential cause
        domain_effect: Domain of potential effect
        details: Additional test details
    """
    causes: bool
    f_statistic: float
    p_value: float
    lag: int = 1
    domain_cause: Optional[DomainType] = None
    domain_effect: Optional[DomainType] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'causes': self.causes,
            'f_statistic': self.f_statistic,
            'p_value': self.p_value,
            'lag': self.lag,
            'domain_cause': self.domain_cause.value if self.domain_cause else None,
            'domain_effect': self.domain_effect.value if self.domain_effect else None,
            **self.details,
        }


def compute_cross_domain_correlation(
    omega_1: Union[np.ndarray, OmegaTimeSeries],
    omega_2: Union[np.ndarray, OmegaTimeSeries],
    method: str = 'pearson',
    align: bool = True,
) -> CorrelationResult:
    """
    Compute correlation between two omega series.

    Args:
        omega_1: First omega series
        omega_2: Second omega series
        method: 'pearson' or 'spearman'
        align: If True and lengths differ, truncate to shorter

    Returns:
        CorrelationResult object
    """
    # Extract values and domains
    if isinstance(omega_1, OmegaTimeSeries):
        values_1 = omega_1.omega_values
        domain_1 = omega_1.domain
    else:
        values_1 = np.asarray(omega_1)
        domain_1 = None

    if isinstance(omega_2, OmegaTimeSeries):
        values_2 = omega_2.omega_values
        domain_2 = omega_2.domain
    else:
        values_2 = np.asarray(omega_2)
        domain_2 = None

    # Align lengths if necessary
    n1, n2 = len(values_1), len(values_2)
    if n1 != n2:
        if align:
            n = min(n1, n2)
            values_1 = values_1[:n]
            values_2 = values_2[:n]
        else:
            raise ValueError(f"Series lengths differ ({n1} vs {n2}) and align=False")

    n = len(values_1)

    if n < 3:
        return CorrelationResult(
            correlation=0.0,
            p_value=1.0,
            method=method,
            domain_1=domain_1,
            domain_2=domain_2,
            n_observations=n,
            details={'error': 'Insufficient data (n < 3)'},
        )

    # Compute correlation
    if method == 'pearson':
        corr, p_value = pearsonr(values_1, values_2)
    elif method == 'spearman':
        corr, p_value = spearmanr(values_1, values_2)
    else:
        raise ValueError(f"Unknown correlation method: {method}")

    return CorrelationResult(
        correlation=float(corr),
        p_value=float(p_value),
        method=method,
        domain_1=domain_1,
        domain_2=domain_2,
        n_observations=n,
        details={
            'mean_1': float(np.mean(values_1)),
            'mean_2': float(np.mean(values_2)),
            'std_1': float(np.std(values_1, ddof=1)),
            'std_2': float(np.std(values_2, ddof=1)),
        },
    )


def compute_cross_correlation(
    omega_1: Union[np.ndarray, OmegaTimeSeries],
    omega_2: Union[np.ndarray, OmegaTimeSeries],
    max_lag: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute cross-correlation function between two omega series.

    Args:
        omega_1: First omega series
        omega_2: Second omega series
        max_lag: Maximum lag to compute (default: n//4)

    Returns:
        Tuple of (lags, cross_correlation_values)
    """
    if isinstance(omega_1, OmegaTimeSeries):
        values_1 = omega_1.omega_values
    else:
        values_1 = np.asarray(omega_1)

    if isinstance(omega_2, OmegaTimeSeries):
        values_2 = omega_2.omega_values
    else:
        values_2 = np.asarray(omega_2)

    # Align lengths
    n = min(len(values_1), len(values_2))
    values_1 = values_1[:n]
    values_2 = values_2[:n]

    if max_lag is None:
        max_lag = n // 4

    # Standardize
    values_1 = (values_1 - np.mean(values_1)) / np.std(values_1)
    values_2 = (values_2 - np.mean(values_2)) / np.std(values_2)

    # Compute cross-correlation for each lag
    lags = np.arange(-max_lag, max_lag + 1)
    ccf_values = np.zeros(len(lags))

    for i, lag in enumerate(lags):
        if lag < 0:
            # omega_1 leads omega_2
            cc = np.correlate(values_1[-lag:], values_2[:n + lag]) / (n - abs(lag))
        elif lag > 0:
            # omega_2 leads omega_1
            cc = np.correlate(values_1[:n - lag], values_2[lag:]) / (n - lag)
        else:
            cc = np.correlate(values_1, values_2) / n
        ccf_values[i] = cc[0] if len(cc) > 0 else 0

    return lags, ccf_values


def compute_granger_causality(
    omega_cause: Union[np.ndarray, OmegaTimeSeries],
    omega_effect: Union[np.ndarray, OmegaTimeSeries],
    max_lag: int = 5,
    alpha: float = 0.05,
) -> GrangerResult:
    """
    Test if omega_cause Granger-causes omega_effect.

    Granger causality tests whether past values of omega_cause help
    predict omega_effect beyond what omega_effect's own past predicts.

    Args:
        omega_cause: Potential causal omega series
        omega_effect: Potential effect omega series
        max_lag: Maximum lag to test
        alpha: Significance level

    Returns:
        GrangerResult object
    """
    if not HAS_STATSMODELS:
        return GrangerResult(
            causes=False,
            f_statistic=0.0,
            p_value=1.0,
            lag=1,
            details={'error': 'statsmodels not available'},
        )

    # Extract values and domains
    if isinstance(omega_cause, OmegaTimeSeries):
        values_cause = omega_cause.omega_values
        domain_cause = omega_cause.domain
    else:
        values_cause = np.asarray(omega_cause)
        domain_cause = None

    if isinstance(omega_effect, OmegaTimeSeries):
        values_effect = omega_effect.omega_values
        domain_effect = omega_effect.domain
    else:
        values_effect = np.asarray(omega_effect)
        domain_effect = None

    # Align lengths
    n = min(len(values_cause), len(values_effect))
    values_cause = values_cause[:n]
    values_effect = values_effect[:n]

    if n < max_lag * 3:
        return GrangerResult(
            causes=False,
            f_statistic=0.0,
            p_value=1.0,
            lag=max_lag,
            domain_cause=domain_cause,
            domain_effect=domain_effect,
            details={'error': f'Insufficient data (n={n} < 3*max_lag={3*max_lag})'},
        )

    # Prepare data for Granger test (columns: [effect, cause])
    data = np.column_stack([values_effect, values_cause])

    try:
        results = grangercausalitytests(data, maxlag=max_lag, verbose=False)

        # Find best lag based on minimum p-value
        best_lag = 1
        best_p = 1.0
        best_f = 0.0

        for lag in range(1, max_lag + 1):
            lag_result = results[lag]
            # Use F-test result
            f_test = lag_result[0]['ssr_ftest']
            f_stat, p_value = f_test[0], f_test[1]

            if p_value < best_p:
                best_p = p_value
                best_f = f_stat
                best_lag = lag

        causes = best_p < alpha

        return GrangerResult(
            causes=causes,
            f_statistic=float(best_f),
            p_value=float(best_p),
            lag=best_lag,
            domain_cause=domain_cause,
            domain_effect=domain_effect,
            details={
                'max_lag_tested': max_lag,
                'alpha': alpha,
                'n_observations': n,
            },
        )

    except Exception as e:
        return GrangerResult(
            causes=False,
            f_statistic=0.0,
            p_value=1.0,
            lag=1,
            domain_cause=domain_cause,
            domain_effect=domain_effect,
            details={'error': str(e)},
        )


def correlation_matrix(
    omega_series_list: List[OmegaTimeSeries],
    method: str = 'pearson',
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute correlation matrix across multiple omega series.

    Args:
        omega_series_list: List of OmegaTimeSeries from different domains
        method: 'pearson' or 'spearman'

    Returns:
        Tuple of (correlation_matrix, p_value_matrix) as DataFrames
    """
    n = len(omega_series_list)

    if n == 0:
        return pd.DataFrame(), pd.DataFrame()

    # Get domain labels
    labels = []
    for series in omega_series_list:
        label = series.domain.value if series.domain else f'series_{len(labels)}'
        # Handle duplicates
        count = labels.count(label)
        if count > 0:
            label = f'{label}_{count + 1}'
        labels.append(label)

    # Initialize matrices
    corr_matrix = np.eye(n)
    p_matrix = np.zeros((n, n))

    # Compute pairwise correlations
    for i in range(n):
        for j in range(i + 1, n):
            result = compute_cross_domain_correlation(
                omega_series_list[i],
                omega_series_list[j],
                method=method,
            )
            corr_matrix[i, j] = result.correlation
            corr_matrix[j, i] = result.correlation
            p_matrix[i, j] = result.p_value
            p_matrix[j, i] = result.p_value

    # Convert to DataFrames
    corr_df = pd.DataFrame(corr_matrix, index=labels, columns=labels)
    p_df = pd.DataFrame(p_matrix, index=labels, columns=labels)

    return corr_df, p_df


def bidirectional_granger_test(
    omega_1: Union[np.ndarray, OmegaTimeSeries],
    omega_2: Union[np.ndarray, OmegaTimeSeries],
    max_lag: int = 5,
    alpha: float = 0.05,
) -> Dict[str, GrangerResult]:
    """
    Test Granger causality in both directions.

    Args:
        omega_1: First omega series
        omega_2: Second omega series
        max_lag: Maximum lag to test
        alpha: Significance level

    Returns:
        Dictionary with '1_causes_2' and '2_causes_1' results
    """
    result_1_to_2 = compute_granger_causality(omega_1, omega_2, max_lag, alpha)
    result_2_to_1 = compute_granger_causality(omega_2, omega_1, max_lag, alpha)

    return {
        '1_causes_2': result_1_to_2,
        '2_causes_1': result_2_to_1,
    }


__all__ = [
    'CorrelationResult',
    'GrangerResult',
    'compute_cross_domain_correlation',
    'compute_cross_correlation',
    'compute_granger_causality',
    'correlation_matrix',
    'bidirectional_granger_test',
]
