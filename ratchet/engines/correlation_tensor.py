"""
RATCHET Extended Correlation Model

Addresses reviewer concern 2.3: "A single scalar ρ may not adequately capture
constraint topology. Two systems with identical ρ can have radically different
resilience."

This module provides:
1. Full correlation matrix analysis (not just mean)
2. Spectral properties (eigenvalues, spectral gap, participation ratio)
3. Higher-order correlations (triplet dependencies)
4. Time-delayed coupling analysis
5. Backward compatibility with scalar ρ

Key insight: Two systems with same scalar ρ but different spectral gap have
different resilience. Block-diagonal structure creates "fault lines" invisible
to scalar ρ.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Union
from enum import Enum
import warnings


class CorrelationStructure(Enum):
    """Classification of correlation structure types."""
    UNIFORM = "uniform"           # All pairs have similar ρ
    BLOCK_DIAGONAL = "block"      # Clusters with high internal, low external ρ
    HIERARCHICAL = "hierarchical" # Nested cluster structure
    SPARSE = "sparse"             # Most pairs uncorrelated, few highly correlated
    UNKNOWN = "unknown"           # Cannot classify


@dataclass
class SpectralProperties:
    """Spectral analysis of the correlation matrix."""

    eigenvalues: np.ndarray
    """Eigenvalues in descending order."""

    spectral_gap: float
    """Gap between largest and second-largest eigenvalue.
    Large gap indicates dominant mode; small gap indicates distributed structure."""

    participation_ratio: float
    """Effective number of independent dimensions.
    PR = (Σλ)² / Σλ². Ranges from 1 (one dominant mode) to k (uniform)."""

    condition_number: float
    """Ratio of largest to smallest eigenvalue. Indicates sensitivity."""

    effective_rank: float
    """Number of eigenvalues needed to capture 95% of variance."""

    @property
    def is_well_conditioned(self) -> bool:
        """True if condition number is reasonable (< 1000)."""
        return self.condition_number < 1000

    @property
    def has_dominant_mode(self) -> bool:
        """True if spectral gap indicates one dominant correlation mode."""
        if len(self.eigenvalues) < 2:
            return True
        return self.spectral_gap > 0.5 * self.eigenvalues[0]


@dataclass
class HigherOrderCorrelations:
    """Higher-order dependency statistics beyond pairwise."""

    triplet_correlation: float
    """Average third-order correlation: E[(X-μ)(Y-μ)(Z-μ)] normalized.
    High values indicate three-way dependencies not captured by pairwise ρ."""

    max_triplet: float
    """Maximum triplet correlation (identifies strongest 3-way dependency)."""

    triplet_sparsity: float
    """Fraction of triplets with |correlation| < 0.1. High = mostly pairwise."""

    n_triplets_sampled: int
    """Number of triplets sampled (exact computation is O(k³))."""


@dataclass
class TimeLagCorrelations:
    """Time-delayed correlation structure."""

    lag_correlations: Dict[int, float]
    """Map from lag (in time steps) to average correlation at that lag."""

    decay_rate: float
    """Exponential decay rate of correlation with lag."""

    memory_length: int
    """Number of lags until correlation drops below 0.1."""

    has_oscillation: bool
    """True if correlation oscillates (indicates periodic coupling)."""

    granger_causality_pairs: List[Tuple[int, int, float]]
    """Pairs (i, j, strength) where i Granger-causes j."""


@dataclass
class ExtendedCorrelationModel:
    """
    Extended correlation model capturing structure beyond scalar ρ.

    This is the core data structure addressing the reviewer's concern that
    "a single scalar ρ may not adequately capture constraint topology."

    Attributes:
        rho_pairwise: Backward-compatible scalar ρ (mean pairwise correlation)
        rho_matrix: Full k×k correlation matrix (optional for large k)
        spectral: Spectral properties of correlation matrix
        higher_order: Triplet and higher correlations
        time_lag: Time-delayed correlation structure
        structure_type: Classified correlation structure
        k: Number of constraints
    """

    rho_pairwise: float
    """Scalar mean pairwise correlation [0, 1]. Backward compatible."""

    k: int
    """Number of constraints."""

    rho_matrix: Optional[np.ndarray] = None
    """Full correlation matrix. Shape (k, k). None if k too large or unavailable."""

    spectral: Optional[SpectralProperties] = None
    """Spectral analysis of correlation matrix."""

    higher_order: Optional[HigherOrderCorrelations] = None
    """Higher-order correlation statistics."""

    time_lag: Optional[TimeLagCorrelations] = None
    """Time-delayed correlations."""

    structure_type: CorrelationStructure = CorrelationStructure.UNKNOWN
    """Classified structure type."""

    asymmetry_index: float = 0.0
    """How asymmetric is the correlation structure? 0 = symmetric, 1 = maximally asymmetric."""

    # --- Backward Compatibility ---

    def to_scalar(self) -> float:
        """
        Return scalar ρ for backward compatibility.

        All existing RATCHET code that uses ρ can call this method.
        """
        return self.rho_pairwise

    def __float__(self) -> float:
        """Allow implicit conversion to float."""
        return self.rho_pairwise

    # --- Derived Properties ---

    @property
    def effective_dimension(self) -> float:
        """
        Effective number of independent constraint dimensions.

        This is the KEY insight: two systems with same scalar ρ but different
        effective dimension have different resilience.

        Uses participation ratio from spectral analysis if available,
        otherwise falls back to scalar estimate.
        """
        if self.spectral is not None:
            return self.spectral.participation_ratio

        # Fallback: estimate from scalar ρ assuming uniform structure
        # For uniform correlation, PR ≈ 1 + (k-1)(1-ρ)²/(1+(k-1)ρ)
        if self.k <= 1:
            return 1.0

        rho = self.rho_pairwise
        # Simplified approximation
        return self.k / (1 + rho * (self.k - 1))

    @property
    def resilience_index(self) -> float:
        """
        Composite resilience indicator [0, 1].

        High resilience = diverse, well-distributed correlation structure.
        Low resilience = concentrated, block-diagonal, or near-singular.
        """
        score = 0.0
        weight_sum = 0.0

        # Factor 1: Effective dimension (higher = more resilient)
        ed = self.effective_dimension
        ed_score = min(1.0, ed / max(1, self.k * 0.5))
        score += 0.4 * ed_score
        weight_sum += 0.4

        # Factor 2: Spectral gap (smaller = more resilient for multi-mode)
        if self.spectral is not None:
            # Invert: small gap is good (no dominant mode)
            gap_ratio = self.spectral.spectral_gap / max(0.01, self.spectral.eigenvalues[0])
            gap_score = 1.0 - min(1.0, gap_ratio)
            score += 0.3 * gap_score
            weight_sum += 0.3

        # Factor 3: Structure type
        structure_scores = {
            CorrelationStructure.UNIFORM: 0.8,
            CorrelationStructure.SPARSE: 0.6,
            CorrelationStructure.HIERARCHICAL: 0.4,
            CorrelationStructure.BLOCK_DIAGONAL: 0.2,
            CorrelationStructure.UNKNOWN: 0.5,
        }
        score += 0.3 * structure_scores.get(self.structure_type, 0.5)
        weight_sum += 0.3

        return score / weight_sum if weight_sum > 0 else 0.5

    @property
    def has_fault_lines(self) -> bool:
        """
        True if correlation structure has "fault lines" - boundaries between
        weakly-coupled clusters that could fail independently.
        """
        if self.structure_type == CorrelationStructure.BLOCK_DIAGONAL:
            return True

        if self.spectral is not None:
            # Multiple large eigenvalues suggest distinct clusters
            if len(self.spectral.eigenvalues) >= 2:
                ratio = self.spectral.eigenvalues[1] / self.spectral.eigenvalues[0]
                return ratio > 0.3  # Second mode is significant

        return False

    # --- k_eff Computation ---

    def compute_k_eff(self) -> float:
        """
        Compute effective constraint count using extended correlation model.

        This is an IMPROVED version of the scalar formula:
            k_eff = k / (1 + ρ(k-1))

        The extended version accounts for non-uniform correlation structure.
        """
        if self.spectral is not None:
            # Use participation ratio directly - it IS the effective dimension
            return self.spectral.participation_ratio

        # Fallback to scalar formula
        return self.k / (1 + self.rho_pairwise * (self.k - 1))

    # --- Reporting ---

    def summary(self) -> str:
        """Human-readable summary of correlation structure."""
        lines = [
            f"Extended Correlation Model (k={self.k})",
            f"  Scalar ρ: {self.rho_pairwise:.3f}",
            f"  Structure: {self.structure_type.value}",
            f"  Effective dimension: {self.effective_dimension:.1f}",
            f"  Resilience index: {self.resilience_index:.2f}",
            f"  Has fault lines: {self.has_fault_lines}",
        ]

        if self.spectral is not None:
            lines.extend([
                f"  Spectral gap: {self.spectral.spectral_gap:.3f}",
                f"  Condition number: {self.spectral.condition_number:.1f}",
            ])

        if self.higher_order is not None:
            lines.append(f"  Triplet correlation: {self.higher_order.triplet_correlation:.3f}")

        return "\n".join(lines)


# =============================================================================
# Computation Functions
# =============================================================================

def compute_correlation_matrix(
    signals: np.ndarray,
    method: str = "pearson"
) -> np.ndarray:
    """
    Compute correlation matrix from constraint signals.

    Args:
        signals: Shape (n_samples, k) matrix of constraint signals.
        method: "pearson" (default) or "spearman".

    Returns:
        Correlation matrix of shape (k, k).
    """
    if signals.ndim != 2:
        raise ValueError(f"signals must be 2D, got shape {signals.shape}")

    n_samples, k = signals.shape

    if n_samples < 2:
        warnings.warn("Need at least 2 samples for correlation; returning identity")
        return np.eye(k)

    if method == "pearson":
        # Standardize
        centered = signals - signals.mean(axis=0)
        std = signals.std(axis=0)
        std[std < 1e-10] = 1.0  # Avoid division by zero
        standardized = centered / std

        # Correlation = covariance of standardized
        corr = np.dot(standardized.T, standardized) / (n_samples - 1)

    elif method == "spearman":
        # Rank-based correlation
        from scipy.stats import spearmanr
        corr, _ = spearmanr(signals)
        if k == 1:
            corr = np.array([[1.0]])
    else:
        raise ValueError(f"Unknown method: {method}")

    # Ensure diagonal is 1 and matrix is symmetric
    np.fill_diagonal(corr, 1.0)
    corr = (corr + corr.T) / 2

    # Clip to valid range
    corr = np.clip(corr, -1.0, 1.0)

    return corr


def compute_spectral_properties(corr_matrix: np.ndarray) -> SpectralProperties:
    """
    Compute spectral properties of correlation matrix.

    Args:
        corr_matrix: Symmetric correlation matrix of shape (k, k).

    Returns:
        SpectralProperties with eigenvalues, spectral gap, participation ratio, etc.
    """
    k = corr_matrix.shape[0]

    # Eigenvalue decomposition (symmetric matrix)
    eigenvalues = np.linalg.eigvalsh(corr_matrix)
    eigenvalues = np.sort(eigenvalues)[::-1]  # Descending order

    # Spectral gap
    if k >= 2:
        spectral_gap = eigenvalues[0] - eigenvalues[1]
    else:
        spectral_gap = eigenvalues[0]

    # Participation ratio: (Σλ)² / Σλ²
    # For correlation matrix, Σλ = k (trace)
    sum_lambda = np.sum(eigenvalues)
    sum_lambda_sq = np.sum(eigenvalues ** 2)

    if sum_lambda_sq > 1e-10:
        participation_ratio = (sum_lambda ** 2) / sum_lambda_sq
    else:
        participation_ratio = 1.0

    # Condition number
    min_eigenvalue = eigenvalues[-1]
    if min_eigenvalue > 1e-10:
        condition_number = eigenvalues[0] / min_eigenvalue
    else:
        condition_number = float('inf')

    # Effective rank (eigenvalues needed for 95% variance)
    cumsum = np.cumsum(eigenvalues) / sum_lambda
    effective_rank = np.searchsorted(cumsum, 0.95) + 1

    return SpectralProperties(
        eigenvalues=eigenvalues,
        spectral_gap=spectral_gap,
        participation_ratio=participation_ratio,
        condition_number=condition_number,
        effective_rank=float(effective_rank),
    )


def compute_triplet_correlations(
    signals: np.ndarray,
    n_samples: int = 1000
) -> HigherOrderCorrelations:
    """
    Compute third-order correlations (triplet dependencies).

    Exact computation is O(k³), so we sample triplets for large k.

    Args:
        signals: Shape (n_samples, k) matrix.
        n_samples: Number of triplets to sample.

    Returns:
        HigherOrderCorrelations with triplet statistics.
    """
    n, k = signals.shape

    if k < 3:
        return HigherOrderCorrelations(
            triplet_correlation=0.0,
            max_triplet=0.0,
            triplet_sparsity=1.0,
            n_triplets_sampled=0,
        )

    # Standardize signals
    centered = signals - signals.mean(axis=0)
    std = signals.std(axis=0)
    std[std < 1e-10] = 1.0
    standardized = centered / std

    # Total possible triplets
    total_triplets = k * (k - 1) * (k - 2) // 6
    n_to_sample = min(n_samples, total_triplets)

    triplet_corrs = []

    if total_triplets <= n_samples:
        # Enumerate all triplets
        for i in range(k):
            for j in range(i + 1, k):
                for m in range(j + 1, k):
                    # Third-order cumulant
                    c3 = np.mean(standardized[:, i] * standardized[:, j] * standardized[:, m])
                    triplet_corrs.append(abs(c3))
    else:
        # Sample triplets randomly
        rng = np.random.default_rng(42)
        for _ in range(n_to_sample):
            indices = rng.choice(k, size=3, replace=False)
            c3 = np.mean(standardized[:, indices[0]] *
                        standardized[:, indices[1]] *
                        standardized[:, indices[2]])
            triplet_corrs.append(abs(c3))

    triplet_corrs = np.array(triplet_corrs)

    return HigherOrderCorrelations(
        triplet_correlation=float(np.mean(triplet_corrs)),
        max_triplet=float(np.max(triplet_corrs)) if len(triplet_corrs) > 0 else 0.0,
        triplet_sparsity=float(np.mean(triplet_corrs < 0.1)),
        n_triplets_sampled=len(triplet_corrs),
    )


def compute_time_lag_correlations(
    signals: np.ndarray,
    max_lag: int = 10
) -> TimeLagCorrelations:
    """
    Compute time-delayed correlations.

    Args:
        signals: Shape (n_samples, k) time series matrix.
        max_lag: Maximum lag to compute.

    Returns:
        TimeLagCorrelations with lag structure.
    """
    n, k = signals.shape

    if n <= max_lag + 1:
        return TimeLagCorrelations(
            lag_correlations={0: 1.0},
            decay_rate=0.0,
            memory_length=0,
            has_oscillation=False,
            granger_causality_pairs=[],
        )

    # Compute average correlation at each lag
    lag_corrs = {}
    for lag in range(max_lag + 1):
        if lag == 0:
            lag_corrs[0] = 1.0
            continue

        corrs = []
        for i in range(k):
            for j in range(k):
                if i == j and lag == 0:
                    continue
                # Correlation between signal i at time t and signal j at time t+lag
                x = signals[:-lag, i]
                y = signals[lag:, j]
                if len(x) > 1:
                    corr = np.corrcoef(x, y)[0, 1]
                    if not np.isnan(corr):
                        corrs.append(abs(corr))

        lag_corrs[lag] = float(np.mean(corrs)) if corrs else 0.0

    # Estimate decay rate (exponential fit)
    lags = np.array(list(lag_corrs.keys()))
    corrs = np.array(list(lag_corrs.values()))

    # Avoid log of zero
    corrs_clipped = np.clip(corrs, 1e-10, 1.0)

    if len(lags) > 1 and lags[-1] > 0:
        # Linear regression on log(corr) vs lag
        log_corrs = np.log(corrs_clipped[1:])  # Skip lag 0
        slope, _ = np.polyfit(lags[1:], log_corrs, 1)
        decay_rate = -slope
    else:
        decay_rate = 0.0

    # Memory length (lags until correlation < 0.1)
    memory_length = 0
    for lag in range(1, max_lag + 1):
        if lag_corrs.get(lag, 0) < 0.1:
            memory_length = lag - 1
            break
    else:
        memory_length = max_lag

    # Check for oscillation (correlation increases then decreases)
    corr_values = [lag_corrs.get(i, 0) for i in range(1, max_lag + 1)]
    has_oscillation = False
    if len(corr_values) >= 3:
        diffs = np.diff(corr_values)
        # Oscillation if sign changes
        sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
        has_oscillation = sign_changes >= 2

    return TimeLagCorrelations(
        lag_correlations=lag_corrs,
        decay_rate=decay_rate,
        memory_length=memory_length,
        has_oscillation=has_oscillation,
        granger_causality_pairs=[],  # Expensive to compute; skip for now
    )


def classify_structure(
    corr_matrix: np.ndarray,
    spectral: SpectralProperties
) -> CorrelationStructure:
    """
    Classify the correlation structure type.

    Args:
        corr_matrix: Correlation matrix.
        spectral: Spectral properties.

    Returns:
        CorrelationStructure enum value.
    """
    k = corr_matrix.shape[0]

    if k <= 2:
        return CorrelationStructure.UNIFORM

    # Get off-diagonal correlations
    off_diag = corr_matrix[np.triu_indices(k, k=1)]

    if len(off_diag) == 0:
        return CorrelationStructure.UNIFORM

    mean_corr = np.mean(np.abs(off_diag))
    std_corr = np.std(off_diag)

    # Sparse: most correlations near zero
    sparsity = np.mean(np.abs(off_diag) < 0.1)
    if sparsity > 0.8:
        return CorrelationStructure.SPARSE

    # Uniform: low variance in correlations
    cv = std_corr / max(mean_corr, 0.01)  # Coefficient of variation
    if cv < 0.3:
        return CorrelationStructure.UNIFORM

    # Block diagonal: check if spectral gap suggests multiple clusters
    if spectral.participation_ratio < k * 0.3:
        # Few effective dimensions suggests block structure
        if not spectral.has_dominant_mode:
            return CorrelationStructure.BLOCK_DIAGONAL

    # Hierarchical: check for nested structure (multiple scales)
    # Simplified heuristic: if effective rank is intermediate
    if k * 0.3 < spectral.effective_rank < k * 0.7:
        return CorrelationStructure.HIERARCHICAL

    return CorrelationStructure.UNKNOWN


def compute_extended_correlation(
    signals: np.ndarray,
    max_lag: int = 10,
    triplet_samples: int = 1000,
    compute_full_matrix: bool = True
) -> ExtendedCorrelationModel:
    """
    Compute the full extended correlation model from constraint signals.

    This is the main entry point for converting raw signal data into the
    extended correlation model.

    Args:
        signals: Shape (n_samples, k) matrix where each column is a constraint signal.
        max_lag: Maximum lag for time correlation analysis.
        triplet_samples: Number of triplets to sample for higher-order analysis.
        compute_full_matrix: If False, only compute scalar statistics (for large k).

    Returns:
        ExtendedCorrelationModel with full analysis.
    """
    if signals.ndim != 2:
        raise ValueError(f"signals must be 2D, got shape {signals.shape}")

    n_samples, k = signals.shape

    # Compute correlation matrix
    if compute_full_matrix and k <= 1000:  # Limit for memory
        corr_matrix = compute_correlation_matrix(signals)
        rho_pairwise = np.mean(np.abs(corr_matrix[np.triu_indices(k, k=1)]))
        spectral = compute_spectral_properties(corr_matrix)
        structure = classify_structure(corr_matrix, spectral)
    else:
        # Large k: sample-based estimation
        corr_matrix = None
        # Estimate mean pairwise correlation from samples
        n_pairs = min(1000, k * (k - 1) // 2)
        rng = np.random.default_rng(42)
        pair_corrs = []
        for _ in range(n_pairs):
            i, j = rng.choice(k, size=2, replace=False)
            corr = np.corrcoef(signals[:, i], signals[:, j])[0, 1]
            if not np.isnan(corr):
                pair_corrs.append(abs(corr))
        rho_pairwise = float(np.mean(pair_corrs)) if pair_corrs else 0.0
        spectral = None
        structure = CorrelationStructure.UNKNOWN

    # Higher-order correlations
    if k >= 3:
        higher_order = compute_triplet_correlations(signals, triplet_samples)
    else:
        higher_order = None

    # Time-lag correlations
    if n_samples >= 20:
        time_lag = compute_time_lag_correlations(signals, max_lag)
    else:
        time_lag = None

    # Asymmetry index
    if corr_matrix is not None:
        # For correlation matrix, asymmetry = 0 by construction
        # But we can measure "directionality" via time-lag
        asymmetry = 0.0
        if time_lag is not None and 1 in time_lag.lag_correlations:
            # Asymmetry if lag-1 correlation differs from lag-(-1)
            asymmetry = abs(time_lag.lag_correlations.get(1, 0) -
                          time_lag.lag_correlations.get(-1, 0)) if -1 in time_lag.lag_correlations else 0.0
    else:
        asymmetry = 0.0

    return ExtendedCorrelationModel(
        rho_pairwise=rho_pairwise,
        k=k,
        rho_matrix=corr_matrix,
        spectral=spectral,
        higher_order=higher_order,
        time_lag=time_lag,
        structure_type=structure,
        asymmetry_index=asymmetry,
    )


# =============================================================================
# Convenience Functions for Integration
# =============================================================================

def scalar_to_extended(rho: float, k: int) -> ExtendedCorrelationModel:
    """
    Create ExtendedCorrelationModel from scalar ρ (backward compatibility).

    Use when only scalar ρ is available (e.g., from existing measurements).
    """
    return ExtendedCorrelationModel(
        rho_pairwise=rho,
        k=k,
        structure_type=CorrelationStructure.UNKNOWN,
    )


def compare_correlation_models(
    model1: ExtendedCorrelationModel,
    model2: ExtendedCorrelationModel
) -> Dict[str, float]:
    """
    Compare two correlation models.

    Returns dict of differences for each metric.
    """
    return {
        "rho_diff": abs(model1.rho_pairwise - model2.rho_pairwise),
        "effective_dim_diff": abs(model1.effective_dimension - model2.effective_dimension),
        "resilience_diff": abs(model1.resilience_index - model2.resilience_index),
        "same_structure": float(model1.structure_type == model2.structure_type),
    }
