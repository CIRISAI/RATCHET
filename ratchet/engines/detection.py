"""
RATCHET Detection Engine

Implements statistical deception detection using Likelihood Ratio Tests (LRT),
Mahalanobis distance computation, and power analysis with finite-sample corrections.

Mathematical Foundation:
- LRT Statistic: Lambda = -2 * log(L(H0) / L(H1))
- Mahalanobis Distance: D = sqrt((mu1 - mu0)^T * Sigma^{-1} * (mu1 - mu0))
- Power Analysis: n >= (z_alpha + z_beta)^2 / (D^2 * p)
- Berry-Esseen Correction: error <= C_BE * rho / (sigma^3 * sqrt(n))

Preconditions (U-01):
- mahalanobis_distance >= 0.5 (otherwise detection is weak)
- deception_rate >= 0.001 (otherwise sample size infeasible)
- For 30 <= n < 100: Apply Berry-Esseen finite-sample correction
- For n < 30: Warn about insufficient sample size

References:
- T-DET-01: Mahalanobis distance non-negativity
- T-DET-02: Deception rate domain
- T-DET-03: Alpha/beta probability constraints
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
from scipy import stats
from scipy.linalg import inv, cholesky

from schemas.types import (
    MahalanobisDistance,
    Probability,
    SampleSize,
    PowerAnalysisResult,
    DetectionMethod,
)


# =============================================================================
# Constants
# =============================================================================

# Berry-Esseen constant (best known bound as of 2012, Shevtsova)
C_BE: float = 0.4748

# Minimum sample size for asymptotic validity
MIN_ASYMPTOTIC_N: int = 100

# Minimum sample size for Berry-Esseen correction applicability
MIN_BERRY_ESSEEN_N: int = 30

# Precondition thresholds (from U-01)
MIN_MAHALANOBIS_DISTANCE: float = 0.5
MIN_DECEPTION_RATE: float = 0.001


# =============================================================================
# Data Classes for Internal Use
# =============================================================================

@dataclass
class LRTResult:
    """Result of Likelihood Ratio Test computation."""
    statistic: float
    p_value: float
    reject_null: bool
    log_likelihood_null: float
    log_likelihood_alt: float


@dataclass
class DistributionParams:
    """Parameters defining a multivariate normal distribution."""
    mean: np.ndarray
    covariance: np.ndarray

    @property
    def dimension(self) -> int:
        return len(self.mean)

    def validate(self) -> None:
        """Validate distribution parameters."""
        if self.mean.ndim != 1:
            raise ValueError("Mean must be a 1D array")
        if self.covariance.ndim != 2:
            raise ValueError("Covariance must be a 2D array")
        if self.covariance.shape[0] != self.covariance.shape[1]:
            raise ValueError("Covariance must be square")
        if self.covariance.shape[0] != len(self.mean):
            raise ValueError("Covariance dimension must match mean dimension")
        # Check positive semi-definiteness
        eigenvalues = np.linalg.eigvalsh(self.covariance)
        if np.any(eigenvalues < -1e-10):
            raise ValueError("Covariance matrix must be positive semi-definite")


# =============================================================================
# Detection Engine Class
# =============================================================================

class DetectionEngine:
    """
    Statistical detection engine for deception detection in agent traces.

    Implements:
    - Likelihood Ratio Test (LRT) for hypothesis testing
    - Mahalanobis distance computation between distributions
    - Power analysis with finite-sample corrections (Berry-Esseen)
    - Required sample size computation for given power

    Usage:
        engine = DetectionEngine()

        # Compute Mahalanobis distance
        D = engine.mahalanobis_distance(mean_honest, mean_deceptive, covariance)

        # Perform power analysis
        result = engine.power_analysis(
            mahalanobis_distance=D,
            deception_rate=0.01,
            alpha=0.05,
            beta=0.05
        )

        # Compute LRT statistic
        lrt_result = engine.likelihood_ratio_test(data, null_dist, alt_dist)
    """

    def __init__(self, method: DetectionMethod = DetectionMethod.LRT):
        """
        Initialize the detection engine.

        Args:
            method: Detection method to use (default: LRT)
        """
        self.method = method

    # =========================================================================
    # Mahalanobis Distance Computation
    # =========================================================================

    def mahalanobis_distance(
        self,
        mean_0: np.ndarray,
        mean_1: np.ndarray,
        covariance: np.ndarray,
    ) -> float:
        """
        Compute Mahalanobis distance between two distributions.

        The Mahalanobis distance measures the separation between two multivariate
        normal distributions with the same covariance matrix:

            D = sqrt((mu_1 - mu_0)^T * Sigma^{-1} * (mu_1 - mu_0))

        Args:
            mean_0: Mean vector of null distribution (honest behavior)
            mean_1: Mean vector of alternative distribution (deceptive behavior)
            covariance: Common covariance matrix

        Returns:
            Mahalanobis distance D >= 0

        Raises:
            ValueError: If inputs have incompatible dimensions
            np.linalg.LinAlgError: If covariance matrix is singular
        """
        mean_0 = np.asarray(mean_0, dtype=np.float64)
        mean_1 = np.asarray(mean_1, dtype=np.float64)
        covariance = np.asarray(covariance, dtype=np.float64)

        # Validate dimensions
        if mean_0.shape != mean_1.shape:
            raise ValueError(
                f"Mean vectors must have same shape: {mean_0.shape} vs {mean_1.shape}"
            )
        if covariance.shape[0] != len(mean_0):
            raise ValueError(
                f"Covariance dimension {covariance.shape[0]} must match "
                f"mean dimension {len(mean_0)}"
            )

        # Compute difference vector
        diff = mean_1 - mean_0

        # Compute inverse covariance (precision matrix)
        try:
            cov_inv = inv(covariance)
        except np.linalg.LinAlgError as e:
            raise np.linalg.LinAlgError(
                f"Covariance matrix is singular: {e}"
            ) from e

        # Compute D^2 = diff^T @ cov_inv @ diff
        d_squared = diff @ cov_inv @ diff

        # Handle numerical issues (D^2 should be non-negative)
        if d_squared < 0:
            if d_squared > -1e-10:
                d_squared = 0.0
            else:
                raise ValueError(
                    f"Negative Mahalanobis distance squared: {d_squared}. "
                    "This indicates numerical instability."
                )

        return math.sqrt(d_squared)

    def mahalanobis_distance_pooled(
        self,
        mean_0: np.ndarray,
        mean_1: np.ndarray,
        cov_0: np.ndarray,
        cov_1: np.ndarray,
        n_0: int,
        n_1: int,
    ) -> float:
        """
        Compute Mahalanobis distance with pooled covariance estimate.

        When the two distributions have different sample covariance matrices,
        we use the pooled covariance:

            Sigma_pooled = ((n_0 - 1) * Sigma_0 + (n_1 - 1) * Sigma_1) / (n_0 + n_1 - 2)

        Args:
            mean_0: Mean of distribution 0
            mean_1: Mean of distribution 1
            cov_0: Covariance of distribution 0
            cov_1: Covariance of distribution 1
            n_0: Sample size for distribution 0
            n_1: Sample size for distribution 1

        Returns:
            Mahalanobis distance using pooled covariance
        """
        cov_0 = np.asarray(cov_0, dtype=np.float64)
        cov_1 = np.asarray(cov_1, dtype=np.float64)

        # Compute pooled covariance
        denom = n_0 + n_1 - 2
        if denom <= 0:
            raise ValueError(f"Insufficient samples: n_0={n_0}, n_1={n_1}")

        cov_pooled = ((n_0 - 1) * cov_0 + (n_1 - 1) * cov_1) / denom

        return self.mahalanobis_distance(mean_0, mean_1, cov_pooled)

    # =========================================================================
    # Likelihood Ratio Test
    # =========================================================================

    def likelihood_ratio_test(
        self,
        data: np.ndarray,
        null_dist: DistributionParams,
        alt_dist: DistributionParams,
        alpha: float = 0.05,
    ) -> LRTResult:
        """
        Perform Likelihood Ratio Test for deception detection.

        The LRT statistic is:
            Lambda = -2 * log(L(H0) / L(H1))
                   = -2 * (log L(H0) - log L(H1))

        Under H0, Lambda approximately follows chi-squared distribution.

        Args:
            data: Observed data points (n x d array)
            null_dist: Parameters of null hypothesis distribution (honest)
            alt_dist: Parameters of alternative hypothesis distribution (deceptive)
            alpha: Significance level for the test

        Returns:
            LRTResult with test statistic, p-value, and decision
        """
        data = np.asarray(data, dtype=np.float64)
        null_dist.validate()
        alt_dist.validate()

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        n_samples, n_features = data.shape

        if n_features != null_dist.dimension:
            raise ValueError(
                f"Data dimension {n_features} must match distribution "
                f"dimension {null_dist.dimension}"
            )

        # Compute log-likelihoods
        log_l_null = self._log_likelihood_mvn(data, null_dist.mean, null_dist.covariance)
        log_l_alt = self._log_likelihood_mvn(data, alt_dist.mean, alt_dist.covariance)

        # LRT statistic: -2 * log(L0 / L1) = -2 * (log L0 - log L1)
        lrt_statistic = -2.0 * (log_l_null - log_l_alt)

        # Degrees of freedom: difference in number of parameters
        # For multivariate normal with known covariance, df = dimension (means only)
        df = null_dist.dimension

        # P-value from chi-squared distribution
        p_value = 1.0 - stats.chi2.cdf(lrt_statistic, df)

        # Decision
        reject_null = p_value < alpha

        return LRTResult(
            statistic=lrt_statistic,
            p_value=p_value,
            reject_null=reject_null,
            log_likelihood_null=log_l_null,
            log_likelihood_alt=log_l_alt,
        )

    def _log_likelihood_mvn(
        self,
        data: np.ndarray,
        mean: np.ndarray,
        covariance: np.ndarray,
    ) -> float:
        """
        Compute log-likelihood for multivariate normal distribution.

        log L = sum_i log p(x_i | mu, Sigma)
              = -n/2 * log(2*pi) - n/2 * log|Sigma| - 1/2 * sum_i (x_i - mu)^T Sigma^{-1} (x_i - mu)

        Args:
            data: Data points (n x d)
            mean: Mean vector
            covariance: Covariance matrix

        Returns:
            Total log-likelihood
        """
        n_samples, n_features = data.shape

        # Use scipy's multivariate normal for numerical stability
        try:
            rv = stats.multivariate_normal(mean=mean, cov=covariance, allow_singular=True)
            log_pdf = rv.logpdf(data)
            return np.sum(log_pdf)
        except np.linalg.LinAlgError:
            # Fallback for singular matrices
            warnings.warn("Covariance matrix is nearly singular, using pseudo-inverse")
            cov_inv = np.linalg.pinv(covariance)
            sign, log_det = np.linalg.slogdet(covariance)
            if sign <= 0:
                log_det = -np.inf

            diff = data - mean
            mahal = np.sum(diff @ cov_inv * diff, axis=1)

            const = -0.5 * n_features * np.log(2 * np.pi)
            log_pdf = const - 0.5 * log_det - 0.5 * mahal
            return np.sum(log_pdf)

    def compute_lrt_statistic(
        self,
        data: np.ndarray,
        mean_null: np.ndarray,
        mean_alt: np.ndarray,
        covariance: np.ndarray,
    ) -> float:
        """
        Compute LRT statistic for simple hypothesis testing.

        Convenience method when both distributions share the same covariance.

        Args:
            data: Observed data
            mean_null: Mean under null hypothesis
            mean_alt: Mean under alternative hypothesis
            covariance: Common covariance matrix

        Returns:
            LRT statistic value
        """
        null_dist = DistributionParams(mean=mean_null, covariance=covariance)
        alt_dist = DistributionParams(mean=mean_alt, covariance=covariance)
        result = self.likelihood_ratio_test(data, null_dist, alt_dist)
        return result.statistic

    # =========================================================================
    # Power Analysis
    # =========================================================================

    def power_analysis(
        self,
        mahalanobis_distance: float,
        deception_rate: float,
        alpha: float = 0.05,
        beta: float = 0.05,
        sample_size: Optional[int] = None,
    ) -> PowerAnalysisResult:
        """
        Perform power analysis for deception detection.

        Computes required sample size for given power, or power for given sample size.

        Formula:
            n >= (z_alpha + z_beta)^2 / (D^2 * p)

        Preconditions (U-01):
            - D >= 0.5 (warns if violated)
            - p >= 0.001 (warns if violated)
            - For 30 <= n < 100: Apply Berry-Esseen correction
            - For n < 30: Warn about insufficient sample size

        Args:
            mahalanobis_distance: D, the Mahalanobis distance between distributions
            deception_rate: p, the prior probability of deception
            alpha: Type I error rate (false positive rate)
            beta: Type II error rate (false negative rate)
            sample_size: If provided, compute achieved power for this n

        Returns:
            PowerAnalysisResult with required_n, power, finite_sample_correction
        """
        # Validate preconditions (U-01)
        self._validate_preconditions(mahalanobis_distance, deception_rate)

        # Validate probability bounds
        if not (0 < alpha < 1):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if not (0 < beta < 1):
            raise ValueError(f"beta must be in (0, 1), got {beta}")

        # Compute z-scores using inverse standard normal (Phi_inv)
        z_alpha = stats.norm.ppf(1 - alpha)
        z_beta = stats.norm.ppf(1 - beta)

        # Compute required sample size
        D = mahalanobis_distance
        p = deception_rate

        numerator = (z_alpha + z_beta) ** 2
        denominator = D ** 2 * p

        if denominator < 1e-15:
            raise ValueError(
                f"Sample size computation failed: D={D}, p={p} gives "
                "near-zero denominator. Increase D or p."
            )

        n_asymptotic = numerator / denominator
        required_n = int(math.ceil(n_asymptotic))

        # Determine which n to use for power computation
        n_for_power = sample_size if sample_size is not None else required_n

        # Compute finite-sample correction if needed
        finite_sample_correction = 0.0
        if n_for_power < MIN_ASYMPTOTIC_N:
            if n_for_power >= MIN_BERRY_ESSEEN_N:
                # Apply Berry-Esseen correction
                finite_sample_correction = self._berry_esseen_correction(n_for_power)
            else:
                warnings.warn(
                    f"Sample size n={n_for_power} < {MIN_BERRY_ESSEEN_N}: "
                    "Results may be unreliable. Consider increasing sample size.",
                    UserWarning
                )
                # Still compute a correction estimate
                finite_sample_correction = self._berry_esseen_correction(n_for_power)

        # Compute achieved power
        if sample_size is not None:
            power = self._compute_power(
                sample_size, D, p, alpha, finite_sample_correction
            )
        else:
            power = 1 - beta  # Target power

        return PowerAnalysisResult(
            required_sample_size=max(1, required_n),
            achieved_power=min(max(power, 0.001), 0.999),  # Clamp to valid range
            mahalanobis_distance=D,
            alpha=alpha,
            beta=beta,
            finite_sample_correction=finite_sample_correction,
        )

    def _validate_preconditions(
        self,
        mahalanobis_distance: float,
        deception_rate: float,
    ) -> None:
        """
        Validate preconditions from U-01.

        Issues warnings (not errors) when preconditions are violated,
        as the computation can still proceed.
        """
        if mahalanobis_distance < 0:
            raise ValueError(
                f"Mahalanobis distance must be non-negative, got {mahalanobis_distance}"
            )

        if mahalanobis_distance < MIN_MAHALANOBIS_DISTANCE:
            warnings.warn(
                f"U-01 Precondition: mahalanobis_distance={mahalanobis_distance} < "
                f"{MIN_MAHALANOBIS_DISTANCE}. Detection power may be weak and "
                "required sample size may be infeasibly large.",
                UserWarning
            )

        if deception_rate < MIN_DECEPTION_RATE:
            warnings.warn(
                f"U-01 Precondition: deception_rate={deception_rate} < "
                f"{MIN_DECEPTION_RATE}. Required sample size may be infeasibly large.",
                UserWarning
            )

    def _berry_esseen_correction(self, n: int, rho: float = 1.0, sigma: float = 1.0) -> float:
        """
        Compute Berry-Esseen finite-sample correction.

        The Berry-Esseen theorem provides an upper bound on the error of the
        normal approximation to the CDF of a standardized sum:

            |F_n(x) - Phi(x)| <= C_BE * rho / (sigma^3 * sqrt(n))

        where:
            - C_BE = 0.4748 (best known constant)
            - rho = E[|X - mu|^3] (third absolute moment)
            - sigma = standard deviation
            - n = sample size

        For standard normal: rho = 2 * sqrt(2/pi) approx 1.596

        Args:
            n: Sample size
            rho: Third absolute moment (default: 1.0 for unit normalization)
            sigma: Standard deviation (default: 1.0)

        Returns:
            Upper bound on CDF approximation error
        """
        if n <= 0:
            return float('inf')

        # For standard normal, rho = E[|Z|^3] = 2 * sqrt(2/pi) approx 1.596
        # But we allow custom rho for non-normal distributions

        sigma_cubed = sigma ** 3
        if sigma_cubed < 1e-15:
            return float('inf')

        correction = C_BE * rho / (sigma_cubed * math.sqrt(n))
        return correction

    def _compute_power(
        self,
        n: int,
        D: float,
        p: float,
        alpha: float,
        correction: float = 0.0,
    ) -> float:
        """
        Compute achieved statistical power for given sample size.

        Power = P(reject H0 | H1 true)
              = Phi(D * sqrt(n * p) - z_alpha)

        With finite-sample correction, we adjust the result.

        Args:
            n: Sample size
            D: Mahalanobis distance
            p: Deception rate
            alpha: Type I error rate
            correction: Berry-Esseen correction to apply

        Returns:
            Statistical power (probability of detecting deception)
        """
        if n <= 0 or D <= 0 or p <= 0:
            return 0.0

        z_alpha = stats.norm.ppf(1 - alpha)

        # Non-centrality parameter
        ncp = D * math.sqrt(n * p)

        # Power = Phi(ncp - z_alpha)
        power = stats.norm.cdf(ncp - z_alpha)

        # Apply finite-sample correction (reduce power estimate)
        power_corrected = max(0.0, power - correction)

        return power_corrected

    # =========================================================================
    # Required Sample Size Computation
    # =========================================================================

    def compute_required_sample_size(
        self,
        mahalanobis_distance: float,
        deception_rate: float,
        alpha: float = 0.05,
        beta: float = 0.05,
        apply_correction: bool = True,
    ) -> int:
        """
        Compute required sample size for given detection power.

        This is a convenience wrapper around power_analysis that returns
        just the required sample size.

        Formula:
            n >= (z_alpha + z_beta)^2 / (D^2 * p)

        If apply_correction is True and the initial n < 100, we iteratively
        increase n to account for finite-sample effects.

        Args:
            mahalanobis_distance: D, separation between distributions
            deception_rate: p, prior deception probability
            alpha: Type I error rate
            beta: Type II error rate
            apply_correction: Whether to inflate n for finite-sample effects

        Returns:
            Required sample size (integer)
        """
        result = self.power_analysis(
            mahalanobis_distance=mahalanobis_distance,
            deception_rate=deception_rate,
            alpha=alpha,
            beta=beta,
        )

        n = result.required_sample_size

        if apply_correction and n < MIN_ASYMPTOTIC_N:
            # Iteratively increase n to achieve target power after correction
            target_power = 1 - beta
            max_iterations = 1000

            for _ in range(max_iterations):
                result = self.power_analysis(
                    mahalanobis_distance=mahalanobis_distance,
                    deception_rate=deception_rate,
                    alpha=alpha,
                    beta=beta,
                    sample_size=n,
                )

                if result.achieved_power >= target_power - 0.001:
                    break
                n += 1

        return n

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def compute_effect_size(
        self,
        mean_diff: np.ndarray,
        pooled_std: np.ndarray,
    ) -> float:
        """
        Compute Cohen's d effect size (multivariate generalization).

        For univariate: d = (mu1 - mu0) / sigma_pooled
        For multivariate: Use Mahalanobis distance

        Args:
            mean_diff: Difference in mean vectors
            pooled_std: Pooled standard deviation (or diagonal of covariance)

        Returns:
            Effect size
        """
        mean_diff = np.asarray(mean_diff)
        pooled_std = np.asarray(pooled_std)

        if mean_diff.ndim == 0 or len(mean_diff) == 1:
            # Univariate case
            return float(np.abs(mean_diff).item() / pooled_std.item())
        else:
            # Multivariate: use diagonal covariance approximation
            cov = np.diag(pooled_std ** 2)
            return self.mahalanobis_distance(
                np.zeros_like(mean_diff),
                mean_diff,
                cov,
            )

    def critical_value(
        self,
        alpha: float,
        df: int = 1,
        test_type: str = "chi2",
    ) -> float:
        """
        Compute critical value for hypothesis testing.

        Args:
            alpha: Significance level
            df: Degrees of freedom
            test_type: "chi2" for chi-squared, "z" for normal

        Returns:
            Critical value for rejection
        """
        if test_type == "chi2":
            return stats.chi2.ppf(1 - alpha, df)
        elif test_type == "z":
            return stats.norm.ppf(1 - alpha)
        else:
            raise ValueError(f"Unknown test type: {test_type}")


# =============================================================================
# Module-level convenience functions
# =============================================================================

def mahalanobis_distance(
    mean_0: np.ndarray,
    mean_1: np.ndarray,
    covariance: np.ndarray,
) -> float:
    """
    Compute Mahalanobis distance between two distributions.

    Module-level convenience function.

    Args:
        mean_0: Mean of first distribution
        mean_1: Mean of second distribution
        covariance: Common covariance matrix

    Returns:
        Mahalanobis distance
    """
    engine = DetectionEngine()
    return engine.mahalanobis_distance(mean_0, mean_1, covariance)


def power_analysis(
    mahalanobis_distance: float,
    deception_rate: float,
    alpha: float = 0.05,
    beta: float = 0.05,
    sample_size: Optional[int] = None,
) -> PowerAnalysisResult:
    """
    Perform power analysis for deception detection.

    Module-level convenience function.

    Args:
        mahalanobis_distance: D, separation between distributions
        deception_rate: p, prior deception probability
        alpha: Type I error rate
        beta: Type II error rate
        sample_size: If provided, compute power for this n

    Returns:
        PowerAnalysisResult
    """
    engine = DetectionEngine()
    return engine.power_analysis(
        mahalanobis_distance=mahalanobis_distance,
        deception_rate=deception_rate,
        alpha=alpha,
        beta=beta,
        sample_size=sample_size,
    )


def compute_required_sample_size(
    mahalanobis_distance: float,
    deception_rate: float,
    alpha: float = 0.05,
    beta: float = 0.05,
) -> int:
    """
    Compute required sample size for given detection power.

    Module-level convenience function.

    Args:
        mahalanobis_distance: D, separation between distributions
        deception_rate: p, prior deception probability
        alpha: Type I error rate
        beta: Type II error rate

    Returns:
        Required sample size
    """
    engine = DetectionEngine()
    return engine.compute_required_sample_size(
        mahalanobis_distance=mahalanobis_distance,
        deception_rate=deception_rate,
        alpha=alpha,
        beta=beta,
    )


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Constants
    "C_BE",
    "MIN_ASYMPTOTIC_N",
    "MIN_BERRY_ESSEEN_N",
    "MIN_MAHALANOBIS_DISTANCE",
    "MIN_DECEPTION_RATE",
    # Data classes
    "LRTResult",
    "DistributionParams",
    # Main class
    "DetectionEngine",
    # Convenience functions
    "mahalanobis_distance",
    "power_analysis",
    "compute_required_sample_size",
]
