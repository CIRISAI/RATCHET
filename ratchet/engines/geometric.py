"""
RATCHET Geometric Engine

Monte Carlo volume estimation for deceptive region intersection with hyperplane constraints.

This engine implements the core geometric analysis of the RATCHET framework:
- Generate random hyperplanes (orthonormal or correlated mode)
- Compute Monte Carlo volume estimation for deceptive region intersection
- Apply k_eff correlation adjustment: k_eff = k / (1 + rho * (k-1))
- Return VolumeEstimate with confidence intervals

Mathematical Foundation:
    Volume of D-dimensional unit ball: V_D = pi^(D/2) / Gamma(D/2 + 1)

    For k hyperplane constraints with correlation rho:
        Effective constraints: k_eff = k / (1 + rho * (k-1))

    Volume decay (exponential in k_eff):
        V(k) ~ V(0) * exp(-lambda * k_eff)

References:
    - T-GEO-01: Dimension bounds (D > 0)
    - T-GEO-02: Radius bounds (0 < r < 0.5)
    - T-GEO-03: Correlation bounds (-1 <= rho <= 1)
"""

from typing import List, Optional, Tuple
import numpy as np
from scipy.special import gamma
from scipy.stats import ortho_group, norm

from schemas.types import (
    Dimension,
    Radius,
    Correlation,
    SampleSize,
    NumConstraints,
    SamplingMode,
    Hyperplane,
    VolumeEstimate,
    AdversarialStrategy,
    compute_effective_rank,
)
from schemas.simulation import GeometricParams


class GeometricEngine:
    """
    Monte Carlo volume estimation engine for deceptive region analysis.

    This engine estimates the volume of the intersection between a deceptive
    region (ball of radius r) and multiple hyperplane constraints.

    Key insight: As k hyperplanes are added, the feasible volume shrinks
    exponentially, making consistent deception increasingly difficult.

    Attributes:
        params: GeometricParams configuration
        rng: NumPy random number generator
        hyperplanes: List of generated hyperplane constraints
    """

    def __init__(
        self,
        params: Optional[GeometricParams] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize the Geometric Engine.

        Args:
            params: GeometricParams configuration. If None, defaults will be used.
            seed: Random seed for reproducibility.
        """
        self.params = params
        self.rng = np.random.default_rng(seed)
        self.hyperplanes: List[np.ndarray] = []
        self._hyperplane_offsets: List[float] = []

    def generate_hyperplanes(
        self,
        dimension: Dimension,
        num_constraints: NumConstraints,
        sampling_mode: SamplingMode = SamplingMode.ORTHONORMAL,
        correlation: Correlation = 0.0,
    ) -> List[Hyperplane]:
        """
        Generate random hyperplanes in D-dimensional space.

        Args:
            dimension: Dimension D of the ambient space.
            num_constraints: Number k of hyperplanes to generate.
            sampling_mode: ORTHONORMAL for independent, CORRELATED for correlated normals.
            correlation: Target pairwise correlation rho for CORRELATED mode.

        Returns:
            List of Hyperplane objects with unit normal vectors and offsets.

        Raises:
            ValueError: If correlation is invalid for given k (must have rho > -1/(k-1)).
        """
        self.hyperplanes = []
        self._hyperplane_offsets = []

        if sampling_mode == SamplingMode.ORTHONORMAL:
            normals = self._generate_orthonormal_normals(dimension, num_constraints)
        elif sampling_mode == SamplingMode.CORRELATED:
            normals = self._generate_correlated_normals(dimension, num_constraints, correlation)
        elif sampling_mode == SamplingMode.ADVERSARIAL:
            normals = self._generate_adversarial_normals(dimension, num_constraints)
        else:
            raise ValueError(f"Unknown sampling mode: {sampling_mode}")

        hyperplane_list = []
        for i in range(num_constraints):
            normal = normals[i]
            # Normalize to unit length
            normal = normal / np.linalg.norm(normal)

            # Random offset in [0.2, 0.8] to ensure hyperplanes intersect unit ball
            offset = self.rng.uniform(0.2, 0.8)

            self.hyperplanes.append(normal)
            self._hyperplane_offsets.append(offset)

            hyperplane_list.append(Hyperplane(
                normal=normal.tolist(),
                offset=offset,
            ))

        return hyperplane_list

    def _generate_orthonormal_normals(
        self,
        dimension: Dimension,
        num_constraints: NumConstraints,
    ) -> np.ndarray:
        """
        Generate orthonormal hyperplane normals using random rotation matrix.

        For k <= D, generates k orthogonal unit vectors.
        For k > D, generates random independent unit vectors (not orthogonal).
        """
        if num_constraints <= dimension:
            # Use orthonormal basis from random rotation matrix
            try:
                Q = ortho_group.rvs(dimension, random_state=self.rng)
                return Q[:num_constraints, :]
            except ValueError:
                # Fallback for dimension=1
                return self.rng.standard_normal((num_constraints, dimension))
        else:
            # More constraints than dimensions: use independent random normals
            normals = self.rng.standard_normal((num_constraints, dimension))
            return normals

    def _generate_correlated_normals(
        self,
        dimension: Dimension,
        num_constraints: NumConstraints,
        correlation: Correlation,
    ) -> np.ndarray:
        """
        Generate correlated hyperplane normals.

        Uses a factor model: n_i = sqrt(1-rho) * z_i + sqrt(rho) * z_common
        where z_i are independent standard normal vectors.

        This achieves pairwise correlation of rho between any two normals.

        Raises:
            ValueError: If correlation would make covariance matrix non-positive-definite.
        """
        # Validate correlation for positive-definite covariance
        # For equicorrelated matrix, need rho > -1/(k-1) for k > 1
        if num_constraints > 1:
            min_rho = -1.0 / (num_constraints - 1)
            if correlation <= min_rho:
                raise ValueError(
                    f"Correlation rho={correlation} <= {min_rho:.6f} is invalid for k={num_constraints}. "
                    f"Need rho > -1/(k-1) for positive definite correlation matrix."
                )

        if abs(correlation) < 1e-10:
            # Zero correlation: just use orthonormal
            return self._generate_orthonormal_normals(dimension, num_constraints)

        # Common factor for correlation
        z_common = self.rng.standard_normal(dimension)
        z_common = z_common / np.linalg.norm(z_common)

        # Compute factor loadings
        # Var(n_i) = (1-rho) + rho = 1
        # Cov(n_i, n_j) = rho * ||z_common||^2 = rho
        rho_clamped = np.clip(correlation, 0, 1)  # Factor model requires non-negative rho

        if correlation < 0:
            # For negative correlation, use rejection sampling or alternative method
            return self._generate_negative_correlated_normals(
                dimension, num_constraints, correlation
            )

        sqrt_rho = np.sqrt(rho_clamped)
        sqrt_one_minus_rho = np.sqrt(1.0 - rho_clamped)

        normals = np.zeros((num_constraints, dimension))
        for i in range(num_constraints):
            z_i = self.rng.standard_normal(dimension)
            z_i = z_i / np.linalg.norm(z_i)
            normals[i] = sqrt_one_minus_rho * z_i + sqrt_rho * z_common

        return normals

    def _generate_negative_correlated_normals(
        self,
        dimension: Dimension,
        num_constraints: NumConstraints,
        correlation: Correlation,
    ) -> np.ndarray:
        """
        Generate negatively correlated normals using Cholesky decomposition.

        For negative correlation, we construct the correlation matrix directly
        and sample from the multivariate normal with that structure.
        """
        # Build equicorrelated covariance matrix for each component
        # Sigma[i,j] = rho if i != j, 1 if i == j
        Sigma = np.full((num_constraints, num_constraints), correlation)
        np.fill_diagonal(Sigma, 1.0)

        # Cholesky decomposition
        try:
            L = np.linalg.cholesky(Sigma)
        except np.linalg.LinAlgError:
            raise ValueError(
                f"Correlation matrix not positive definite for rho={correlation}, k={num_constraints}"
            )

        # Generate normals component-wise with correlation structure
        normals = np.zeros((num_constraints, dimension))
        for d in range(dimension):
            # Independent standard normals
            z = self.rng.standard_normal(num_constraints)
            # Apply correlation structure
            correlated = L @ z
            normals[:, d] = correlated

        return normals

    def _generate_adversarial_normals(
        self,
        dimension: Dimension,
        num_constraints: NumConstraints,
    ) -> np.ndarray:
        """
        Generate adversarial hyperplane normals.

        Adversarial mode: normals are aligned to maximize volume reduction
        by pointing toward the deceptive region center.
        """
        # Simple adversarial: all normals point toward origin (center of unit ball)
        # with small random perturbations
        base_direction = np.ones(dimension) / np.sqrt(dimension)

        normals = np.zeros((num_constraints, dimension))
        for i in range(num_constraints):
            # Add random perturbation
            perturbation = self.rng.standard_normal(dimension) * 0.1
            normals[i] = base_direction + perturbation
            normals[i] = normals[i] / np.linalg.norm(normals[i])

        return normals

    def sample_unit_ball(
        self,
        dimension: Dimension,
        num_samples: SampleSize,
    ) -> np.ndarray:
        """
        Sample points uniformly from the D-dimensional unit ball.

        Uses the method of Muller (1959): generate Gaussian vector and normalize,
        then scale by uniform random radius.

        Args:
            dimension: Dimension D of the ball.
            num_samples: Number of points to sample.

        Returns:
            Array of shape (num_samples, dimension) with points in unit ball.
        """
        # Generate direction: normalize Gaussian vectors
        directions = self.rng.standard_normal((num_samples, dimension))
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        directions = directions / norms

        # Generate radius: U^(1/D) where U ~ Uniform(0,1)
        u = self.rng.uniform(0, 1, num_samples)
        radii = u ** (1.0 / dimension)

        # Scale directions by radii
        points = directions * radii[:, np.newaxis]

        return points

    def check_hyperplane_constraints(
        self,
        points: np.ndarray,
        hyperplanes: Optional[List[np.ndarray]] = None,
        offsets: Optional[List[float]] = None,
    ) -> np.ndarray:
        """
        Check which points satisfy all hyperplane constraints.

        A point x satisfies hyperplane (n, d) if n . x <= d.
        This defines a half-space constraint.

        Args:
            points: Array of shape (n_points, dimension).
            hyperplanes: List of normal vectors. If None, uses self.hyperplanes.
            offsets: List of offsets. If None, uses self._hyperplane_offsets.

        Returns:
            Boolean array of shape (n_points,) indicating constraint satisfaction.
        """
        if hyperplanes is None:
            hyperplanes = self.hyperplanes
        if offsets is None:
            offsets = self._hyperplane_offsets

        if len(hyperplanes) == 0:
            return np.ones(len(points), dtype=bool)

        # Check all constraints: n_i . x <= d_i
        satisfies_all = np.ones(len(points), dtype=bool)

        for normal, offset in zip(hyperplanes, offsets):
            # Dot product of each point with normal
            dots = points @ np.array(normal)
            # Point satisfies constraint if dot product <= offset
            satisfies = dots <= offset
            satisfies_all = satisfies_all & satisfies

        return satisfies_all

    def compute_ball_volume(self, dimension: Dimension, radius: float = 1.0) -> float:
        """
        Compute the volume of a D-dimensional ball.

        Formula: V_D(r) = (pi^(D/2) / Gamma(D/2 + 1)) * r^D

        Args:
            dimension: Dimension D.
            radius: Ball radius r.

        Returns:
            Volume of the D-ball.
        """
        return (np.pi ** (dimension / 2) / gamma(dimension / 2 + 1)) * (radius ** dimension)

    def estimate_volume(
        self,
        dimension: Dimension,
        num_constraints: NumConstraints,
        deceptive_radius: Radius,
        constraint_correlation: Correlation = 0.0,
        sampling_mode: SamplingMode = SamplingMode.ORTHONORMAL,
        num_samples: SampleSize = 100_000,
        adversary: Optional[AdversarialStrategy] = None,
    ) -> VolumeEstimate:
        """
        Estimate volume of deceptive region intersection with hyperplane constraints.

        Procedure:
        1. Generate k hyperplanes (orthonormal or correlated mode)
        2. Sample n points uniformly from unit ball (scaled by deceptive_radius)
        3. Count points satisfying all hyperplane constraints
        4. Volume = (points_inside / total_points) * ball_volume
        5. Compute confidence intervals using normal approximation
        6. Apply k_eff correlation adjustment

        Args:
            dimension: Dimension D of the space.
            num_constraints: Number k of hyperplane constraints.
            deceptive_radius: Radius r of the deceptive region (0 < r < 0.5).
            constraint_correlation: Pairwise correlation rho between constraints.
            sampling_mode: How to generate hyperplane normals.
            num_samples: Number of Monte Carlo samples.
            adversary: Optional adversarial strategy configuration.

        Returns:
            VolumeEstimate with volume, confidence intervals, and metadata.
        """
        # Use adversarial mode if specified
        if adversary is not None:
            sampling_mode = SamplingMode.ADVERSARIAL

        # Generate hyperplanes
        self.generate_hyperplanes(
            dimension=dimension,
            num_constraints=num_constraints,
            sampling_mode=sampling_mode,
            correlation=constraint_correlation,
        )

        # Sample from unit ball and scale by deceptive radius
        points = self.sample_unit_ball(dimension, num_samples)
        points = points * deceptive_radius

        # Check which points satisfy all hyperplane constraints
        inside = self.check_hyperplane_constraints(points)

        # Compute volume fraction
        fraction_inside = np.mean(inside)

        # Compute ball volume
        ball_volume = self.compute_ball_volume(dimension, deceptive_radius)

        # Estimated volume = fraction * ball_volume
        volume = fraction_inside * ball_volume

        # Compute confidence intervals (Wilson score interval for proportion)
        n = num_samples
        p_hat = fraction_inside

        # Standard error for proportion
        if p_hat > 0 and p_hat < 1:
            se = np.sqrt(p_hat * (1 - p_hat) / n)
        else:
            se = 1.0 / np.sqrt(n)  # Conservative estimate

        # 95% confidence interval using normal approximation
        z = 1.96
        ci_lower_frac = max(0, p_hat - z * se)
        ci_upper_frac = min(1, p_hat + z * se)

        ci_lower = ci_lower_frac * ball_volume
        ci_upper = ci_upper_frac * ball_volume

        # Compute effective rank with correlation adjustment
        # Handle edge case: when num_constraints=0, k_eff would be 0 but VolumeEstimate
        # requires effective_rank > 0. For k=0, we set it to None.
        if num_constraints > 0:
            k_eff = compute_effective_rank(num_constraints, constraint_correlation)
        else:
            k_eff = None

        # Estimate decay constant (lambda)
        # From theory: V(k) ~ V(0) * exp(-lambda * k_eff)
        # So lambda = -log(V(k)/V(0)) / k_eff
        if k_eff is not None and k_eff > 0 and fraction_inside > 0 and fraction_inside < 1:
            decay_constant = -np.log(fraction_inside) / k_eff
        else:
            decay_constant = None

        return VolumeEstimate(
            volume=fraction_inside,  # Volume fraction, not absolute volume
            ci_lower=ci_lower_frac,
            ci_upper=ci_upper_frac,
            num_samples=num_samples,
            decay_constant=decay_constant,
            effective_rank=k_eff,
        )

    def estimate_volume_from_params(
        self,
        params: Optional[GeometricParams] = None,
    ) -> VolumeEstimate:
        """
        Estimate volume using GeometricParams configuration.

        Args:
            params: GeometricParams object. If None, uses self.params.

        Returns:
            VolumeEstimate result.

        Raises:
            ValueError: If no params provided and self.params is None.
        """
        if params is None:
            params = self.params
        if params is None:
            raise ValueError("No GeometricParams provided")

        return self.estimate_volume(
            dimension=params.dimension,
            num_constraints=params.num_constraints,
            deceptive_radius=params.deceptive_radius,
            constraint_correlation=params.constraint_correlation,
            sampling_mode=params.sampling_mode,
            num_samples=params.num_samples,
            adversary=params.adversary,
        )

    def compute_volume_decay_curve(
        self,
        dimension: Dimension,
        max_constraints: NumConstraints,
        deceptive_radius: Radius,
        constraint_correlation: Correlation = 0.0,
        sampling_mode: SamplingMode = SamplingMode.ORTHONORMAL,
        num_samples: SampleSize = 10_000,
        step: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Compute volume decay curve as constraints increase.

        Args:
            dimension: Dimension D of the space.
            max_constraints: Maximum number of constraints to test.
            deceptive_radius: Radius r of deceptive region.
            constraint_correlation: Pairwise correlation rho.
            sampling_mode: How to generate hyperplane normals.
            num_samples: Monte Carlo samples per estimate.
            step: Step size for k values.

        Returns:
            Tuple of (k_values, volumes, ci_widths, fitted_decay_rate).
        """
        k_values = np.arange(0, max_constraints + 1, step)
        volumes = []
        ci_widths = []

        for k in k_values:
            if k == 0:
                # No constraints: full ball volume
                ball_vol = self.compute_ball_volume(dimension, deceptive_radius)
                volumes.append(ball_vol)
                ci_widths.append(0.0)
            else:
                result = self.estimate_volume(
                    dimension=dimension,
                    num_constraints=k,
                    deceptive_radius=deceptive_radius,
                    constraint_correlation=constraint_correlation,
                    sampling_mode=sampling_mode,
                    num_samples=num_samples,
                )
                volumes.append(result.volume)
                ci_widths.append(result.ci_upper - result.ci_lower)

        volumes = np.array(volumes)
        ci_widths = np.array(ci_widths)

        # Fit exponential decay: V(k) = V(0) * exp(-lambda * k)
        # Apply k_eff correction for fitting
        k_eff_values = np.array([
            compute_effective_rank(k, constraint_correlation) if k > 0 else 0
            for k in k_values
        ])

        # Linear regression on log(V) vs k_eff
        valid_mask = volumes > 1e-100
        if np.sum(valid_mask) > 1:
            log_volumes = np.log(volumes[valid_mask])
            k_eff_valid = k_eff_values[valid_mask]

            # Fit: log(V) = log(V0) - lambda * k_eff
            coeffs = np.polyfit(k_eff_valid, log_volumes, 1)
            decay_rate = -coeffs[0]
        else:
            decay_rate = np.nan

        return k_values, volumes, ci_widths, decay_rate

    def apply_correlation_adjustment(
        self,
        num_constraints: NumConstraints,
        correlation: Correlation,
    ) -> float:
        """
        Apply the k_eff correlation adjustment formula.

        Formula: k_eff = k / (1 + rho * (k - 1))

        Interpretation:
        - rho = 0: k_eff = k (independent constraints)
        - rho = 1: k_eff = 1 (perfectly correlated, equivalent to single constraint)
        - rho < 0: k_eff > k (anti-correlated constraints are more restrictive)

        Args:
            num_constraints: Number k of constraints.
            correlation: Pairwise correlation rho.

        Returns:
            Effective number of independent constraints k_eff.

        Raises:
            ValueError: If rho <= -1/(k-1) (invalid for positive-definite matrix).
        """
        return compute_effective_rank(num_constraints, correlation)


def create_geometric_engine(
    params: Optional[GeometricParams] = None,
    seed: Optional[int] = None,
) -> GeometricEngine:
    """
    Factory function to create a GeometricEngine instance.

    Args:
        params: GeometricParams configuration.
        seed: Random seed for reproducibility.

    Returns:
        Configured GeometricEngine instance.
    """
    return GeometricEngine(params=params, seed=seed)


__all__ = [
    'GeometricEngine',
    'create_geometric_engine',
]
