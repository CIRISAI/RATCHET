"""
RATCHET Omega Module - Outlier Detection

Detects anomalous omega values that may indicate:
    - Model failures
    - Regime changes
    - Genuine structural anomalies

Methods:
    - Z-score based detection
    - IQR-based detection
    - Changepoint detection
    - Isolation Forest (if sklearn available)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union

from .residuals import OmegaTimeSeries, OmegaObservation

try:
    from sklearn.ensemble import IsolationForest
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


@dataclass
class OutlierResult:
    """
    Result of outlier detection on omega series.

    Attributes:
        n_outliers: Number of outliers detected
        outlier_indices: Indices of outlier observations
        outlier_values: Omega values of outliers
        outlier_timestamps: Timestamps of outliers
        threshold: Threshold used for detection
        method: Detection method used
        fraction_outliers: Fraction of observations that are outliers
        details: Additional method-specific details
    """
    n_outliers: int = 0
    outlier_indices: List[int] = field(default_factory=list)
    outlier_values: List[float] = field(default_factory=list)
    outlier_timestamps: List[float] = field(default_factory=list)
    threshold: float = 0.0
    method: str = ''
    fraction_outliers: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_outliers(self) -> bool:
        """Check if any outliers were detected."""
        return self.n_outliers > 0

    @property
    def severity(self) -> str:
        """Assess outlier severity."""
        if self.fraction_outliers == 0:
            return 'none'
        elif self.fraction_outliers < 0.01:
            return 'minimal'
        elif self.fraction_outliers < 0.05:
            return 'moderate'
        elif self.fraction_outliers < 0.10:
            return 'high'
        else:
            return 'extreme'

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'n_outliers': self.n_outliers,
            'outlier_indices': self.outlier_indices,
            'outlier_values': self.outlier_values,
            'outlier_timestamps': self.outlier_timestamps,
            'threshold': self.threshold,
            'method': self.method,
            'fraction_outliers': self.fraction_outliers,
            'severity': self.severity,
            **self.details,
        }


@dataclass
class ChangepointResult:
    """
    Result of changepoint detection.

    Attributes:
        n_changepoints: Number of changepoints detected
        changepoint_indices: Indices where changes occur
        changepoint_timestamps: Timestamps of changepoints
        segment_means: Mean omega in each segment
        segment_stds: Std of omega in each segment
        method: Detection method used
        details: Additional details
    """
    n_changepoints: int = 0
    changepoint_indices: List[int] = field(default_factory=list)
    changepoint_timestamps: List[float] = field(default_factory=list)
    segment_means: List[float] = field(default_factory=list)
    segment_stds: List[float] = field(default_factory=list)
    method: str = ''
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_changepoints(self) -> bool:
        """Check if any changepoints were detected."""
        return self.n_changepoints > 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'n_changepoints': self.n_changepoints,
            'changepoint_indices': self.changepoint_indices,
            'changepoint_timestamps': self.changepoint_timestamps,
            'segment_means': self.segment_means,
            'segment_stds': self.segment_stds,
            'method': self.method,
            **self.details,
        }


def detect_outliers_zscore(
    omega: Union[np.ndarray, OmegaTimeSeries],
    threshold: float = 3.0,
    use_mad: bool = False,
) -> OutlierResult:
    """
    Detect outliers using z-score method.

    Args:
        omega: Omega values (array or OmegaTimeSeries)
        threshold: Number of standard deviations for outlier cutoff
        use_mad: If True, use Median Absolute Deviation instead of std

    Returns:
        OutlierResult object
    """
    if isinstance(omega, OmegaTimeSeries):
        values = omega.omega_values
        timestamps = omega.timestamps
    else:
        values = np.asarray(omega)
        timestamps = np.arange(len(values), dtype=float)

    n = len(values)

    if n < 3:
        return OutlierResult(
            method='zscore',
            threshold=threshold,
            details={'error': 'Insufficient data'},
        )

    if use_mad:
        # Median Absolute Deviation (more robust)
        median = np.median(values)
        mad = np.median(np.abs(values - median))
        # MAD to std conversion factor for normal distribution
        scale = 1.4826 * mad if mad > 0 else 1.0
        z_scores = (values - median) / scale
    else:
        # Standard z-score
        mean = np.mean(values)
        std = np.std(values, ddof=1)
        if std == 0:
            z_scores = np.zeros(n)
        else:
            z_scores = (values - mean) / std

    # Find outliers
    outlier_mask = np.abs(z_scores) > threshold
    outlier_indices = np.where(outlier_mask)[0].tolist()
    outlier_values = values[outlier_mask].tolist()
    outlier_timestamps = timestamps[outlier_mask].tolist()

    return OutlierResult(
        n_outliers=len(outlier_indices),
        outlier_indices=outlier_indices,
        outlier_values=outlier_values,
        outlier_timestamps=outlier_timestamps,
        threshold=threshold,
        method='zscore' if not use_mad else 'mad_zscore',
        fraction_outliers=len(outlier_indices) / n,
        details={
            'use_mad': use_mad,
            'center': float(np.median(values) if use_mad else np.mean(values)),
            'scale': float(scale if use_mad else np.std(values, ddof=1)),
        },
    )


def detect_outliers_iqr(
    omega: Union[np.ndarray, OmegaTimeSeries],
    multiplier: float = 1.5,
) -> OutlierResult:
    """
    Detect outliers using IQR (Interquartile Range) method.

    Outliers are defined as values below Q1 - multiplier*IQR or
    above Q3 + multiplier*IQR.

    Args:
        omega: Omega values (array or OmegaTimeSeries)
        multiplier: IQR multiplier (1.5 for outliers, 3.0 for extreme outliers)

    Returns:
        OutlierResult object
    """
    if isinstance(omega, OmegaTimeSeries):
        values = omega.omega_values
        timestamps = omega.timestamps
    else:
        values = np.asarray(omega)
        timestamps = np.arange(len(values), dtype=float)

    n = len(values)

    if n < 4:
        return OutlierResult(
            method='iqr',
            threshold=multiplier,
            details={'error': 'Insufficient data'},
        )

    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1

    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr

    # Find outliers
    outlier_mask = (values < lower_bound) | (values > upper_bound)
    outlier_indices = np.where(outlier_mask)[0].tolist()
    outlier_values = values[outlier_mask].tolist()
    outlier_timestamps = timestamps[outlier_mask].tolist()

    return OutlierResult(
        n_outliers=len(outlier_indices),
        outlier_indices=outlier_indices,
        outlier_values=outlier_values,
        outlier_timestamps=outlier_timestamps,
        threshold=multiplier,
        method='iqr',
        fraction_outliers=len(outlier_indices) / n,
        details={
            'q1': float(q1),
            'q3': float(q3),
            'iqr': float(iqr),
            'lower_bound': float(lower_bound),
            'upper_bound': float(upper_bound),
        },
    )


def detect_outliers_isolation_forest(
    omega: Union[np.ndarray, OmegaTimeSeries],
    contamination: float = 0.05,
    random_state: int = 42,
) -> OutlierResult:
    """
    Detect outliers using Isolation Forest algorithm.

    Args:
        omega: Omega values (array or OmegaTimeSeries)
        contamination: Expected fraction of outliers
        random_state: Random seed

    Returns:
        OutlierResult object
    """
    if not HAS_SKLEARN:
        return OutlierResult(
            method='isolation_forest',
            details={'error': 'sklearn not available'},
        )

    if isinstance(omega, OmegaTimeSeries):
        values = omega.omega_values
        timestamps = omega.timestamps
    else:
        values = np.asarray(omega)
        timestamps = np.arange(len(values), dtype=float)

    n = len(values)

    if n < 10:
        return OutlierResult(
            method='isolation_forest',
            details={'error': 'Insufficient data (n < 10)'},
        )

    # Reshape for sklearn
    X = values.reshape(-1, 1)

    # Fit Isolation Forest
    clf = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=100,
    )
    predictions = clf.fit_predict(X)

    # -1 indicates outliers, 1 indicates inliers
    outlier_mask = predictions == -1
    outlier_indices = np.where(outlier_mask)[0].tolist()
    outlier_values = values[outlier_mask].tolist()
    outlier_timestamps = timestamps[outlier_mask].tolist()

    # Get anomaly scores
    scores = clf.decision_function(X)

    return OutlierResult(
        n_outliers=len(outlier_indices),
        outlier_indices=outlier_indices,
        outlier_values=outlier_values,
        outlier_timestamps=outlier_timestamps,
        threshold=contamination,
        method='isolation_forest',
        fraction_outliers=len(outlier_indices) / n,
        details={
            'contamination': contamination,
            'min_score': float(np.min(scores)),
            'max_score': float(np.max(scores)),
            'mean_outlier_score': float(np.mean(scores[outlier_mask])) if len(outlier_indices) > 0 else None,
        },
    )


def detect_changepoints(
    omega: Union[np.ndarray, OmegaTimeSeries],
    method: str = 'cusum',
    threshold: Optional[float] = None,
    min_segment_length: int = 10,
) -> ChangepointResult:
    """
    Detect changepoints in omega series.

    Args:
        omega: Omega values (array or OmegaTimeSeries)
        method: 'cusum' (cumulative sum) or 'binary_segmentation'
        threshold: Detection threshold (auto-computed if None)
        min_segment_length: Minimum segment length between changepoints

    Returns:
        ChangepointResult object
    """
    if isinstance(omega, OmegaTimeSeries):
        values = omega.omega_values
        timestamps = omega.timestamps
    else:
        values = np.asarray(omega)
        timestamps = np.arange(len(values), dtype=float)

    n = len(values)

    if n < min_segment_length * 2:
        return ChangepointResult(
            method=method,
            details={'error': f'Insufficient data (n < {min_segment_length * 2})'},
        )

    if method == 'cusum':
        changepoints = _detect_cusum_changepoints(
            values, threshold, min_segment_length
        )
    elif method == 'binary_segmentation':
        changepoints = _detect_binary_segmentation(
            values, min_segment_length
        )
    else:
        return ChangepointResult(
            method=method,
            details={'error': f'Unknown method: {method}'},
        )

    # Compute segment statistics
    segment_means = []
    segment_stds = []
    boundaries = [0] + list(changepoints) + [n]

    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        segment = values[start:end]
        segment_means.append(float(np.mean(segment)))
        segment_stds.append(float(np.std(segment, ddof=1)) if len(segment) > 1 else 0.0)

    return ChangepointResult(
        n_changepoints=len(changepoints),
        changepoint_indices=changepoints,
        changepoint_timestamps=[float(timestamps[i]) for i in changepoints],
        segment_means=segment_means,
        segment_stds=segment_stds,
        method=method,
        details={
            'min_segment_length': min_segment_length,
            'n_segments': len(segment_means),
        },
    )


def _detect_cusum_changepoints(
    values: np.ndarray,
    threshold: Optional[float],
    min_segment_length: int,
) -> List[int]:
    """Detect changepoints using CUSUM algorithm."""
    n = len(values)

    # Center the series
    mean_val = np.mean(values)
    centered = values - mean_val

    # Compute cumulative sum
    cusum = np.cumsum(centered)

    # Auto-compute threshold if not provided
    if threshold is None:
        # Use 2 standard deviations of the cumsum
        threshold = 2 * np.std(cusum)

    changepoints = []

    # Find peaks in absolute cusum
    for i in range(min_segment_length, n - min_segment_length):
        # Check if this is a local maximum in |cusum|
        window = slice(max(0, i - 5), min(n, i + 6))
        if abs(cusum[i]) == np.max(np.abs(cusum[window])):
            if abs(cusum[i]) > threshold:
                # Check minimum distance from previous changepoint
                if len(changepoints) == 0 or i - changepoints[-1] >= min_segment_length:
                    changepoints.append(i)

    return changepoints


def _detect_binary_segmentation(
    values: np.ndarray,
    min_segment_length: int,
    max_changepoints: int = 10,
) -> List[int]:
    """Detect changepoints using binary segmentation."""
    n = len(values)
    changepoints = []

    def find_best_split(start: int, end: int) -> Tuple[int, float]:
        """Find best split point in a segment."""
        if end - start < 2 * min_segment_length:
            return -1, 0.0

        best_idx = -1
        best_gain = 0.0

        segment = values[start:end]
        total_var = np.var(segment) * len(segment)

        for i in range(min_segment_length, len(segment) - min_segment_length):
            left = segment[:i]
            right = segment[i:]

            left_var = np.var(left) * len(left)
            right_var = np.var(right) * len(right)

            gain = total_var - (left_var + right_var)

            if gain > best_gain:
                best_gain = gain
                best_idx = start + i

        return best_idx, best_gain

    # Iteratively find changepoints
    segments = [(0, n)]

    while len(changepoints) < max_changepoints and segments:
        # Find the segment with the best split
        best_segment = None
        best_split = -1
        best_segment_gain = 0.0

        for start, end in segments:
            split_idx, gain = find_best_split(start, end)
            if gain > best_segment_gain:
                best_segment = (start, end)
                best_split = split_idx
                best_segment_gain = gain

        if best_split == -1 or best_segment_gain < np.var(values) * 0.1:
            break

        # Add changepoint
        changepoints.append(best_split)

        # Update segments
        segments.remove(best_segment)
        segments.append((best_segment[0], best_split))
        segments.append((best_split, best_segment[1]))

    return sorted(changepoints)


def detect_all_outliers(
    omega: Union[np.ndarray, OmegaTimeSeries],
    methods: Optional[List[str]] = None,
) -> Dict[str, OutlierResult]:
    """
    Run multiple outlier detection methods.

    Args:
        omega: Omega values (array or OmegaTimeSeries)
        methods: List of methods to run. Default: ['zscore', 'iqr', 'mad']

    Returns:
        Dictionary mapping method names to OutlierResult objects
    """
    if methods is None:
        methods = ['zscore', 'iqr', 'mad']

    results = {}

    for method in methods:
        if method == 'zscore':
            results['zscore'] = detect_outliers_zscore(omega, threshold=3.0)
        elif method == 'iqr':
            results['iqr'] = detect_outliers_iqr(omega, multiplier=1.5)
        elif method == 'mad':
            results['mad'] = detect_outliers_zscore(omega, threshold=3.0, use_mad=True)
        elif method == 'isolation_forest':
            results['isolation_forest'] = detect_outliers_isolation_forest(omega)

    return results


__all__ = [
    'OutlierResult',
    'ChangepointResult',
    'detect_outliers_zscore',
    'detect_outliers_iqr',
    'detect_outliers_isolation_forest',
    'detect_changepoints',
    'detect_all_outliers',
]
