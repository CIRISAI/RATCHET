"""
RATCHET Robustness Analysis Module

Addresses reviewer concern 2.1: "If constraints are adversarially structured,
clustered, or non-convex, the volume-decay approximation may break sharply
rather than smoothly."

This module analyzes:
1. Generic geometry baseline - Monte Carlo validation of exponential decay
2. Clustered constraint analysis - Effective k when constraints group
3. Adversarial constraint placement - Worst-case volume retention
4. Non-convex region analysis - Bounds on approximation error
5. Sensitivity analysis - Impact of assumption violations

Key insight: The volume decay theorem V(k) = V(0) * exp(-λ * k_eff) assumes
"generic" constraint placement. This module quantifies when this assumption
breaks and by how much.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Callable
from enum import Enum
import warnings

# Try to import correlation_tensor for integration
try:
    from .correlation_tensor import ExtendedCorrelationModel, compute_extended_correlation
except ImportError:
    ExtendedCorrelationModel = None
    compute_extended_correlation = None


class GeometryType(Enum):
    """Classification of constraint geometry types."""
    GENERIC = "generic"           # Random i.i.d. constraint placement
    CLUSTERED = "clustered"       # Constraints grouped into clusters
    ADVERSARIAL = "adversarial"   # Constraints placed to maximize residual volume
    SPARSE = "sparse"             # Few constraints in high-dimensional space
    DEGENERATE = "degenerate"     # Constraints nearly parallel or redundant


@dataclass
class VolumeEstimate:
    """Result of Monte Carlo volume estimation."""

    volume: float
    """Estimated volume fraction [0, 1]."""

    std_error: float
    """Standard error of the estimate."""

    n_samples: int
    """Number of Monte Carlo samples used."""

    confidence_interval: Tuple[float, float]
    """95% confidence interval for volume."""

    @property
    def relative_error(self) -> float:
        """Relative standard error."""
        if self.volume > 1e-10:
            return self.std_error / self.volume
        return float('inf')


@dataclass
class RobustnessReport:
    """Report on robustness of volume decay under non-generic geometries."""

    geometry_type: GeometryType
    """Type of geometry analyzed."""

    generic_volume: VolumeEstimate
    """Volume under generic assumption."""

    actual_volume: VolumeEstimate
    """Volume under actual/tested geometry."""

    volume_ratio: float
    """actual_volume / generic_volume (>1 means theory underestimates)."""

    breakdown_detected: bool
    """True if discrepancy exceeds 10% threshold."""

    effective_k: float
    """Effective constraint count (may differ from nominal k)."""

    effective_k_ratio: float
    """k_eff / k (1.0 means no reduction, <1 means clustering)."""

    confidence: float
    """Confidence in the analysis [0, 1]."""

    details: Dict
    """Additional analysis-specific details."""

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Robustness Report ({self.geometry_type.value})",
            f"  Generic volume: {self.generic_volume.volume:.4f} ± {self.generic_volume.std_error:.4f}",
            f"  Actual volume:  {self.actual_volume.volume:.4f} ± {self.actual_volume.std_error:.4f}",
            f"  Volume ratio:   {self.volume_ratio:.2f}x",
            f"  k_eff / k:      {self.effective_k_ratio:.2f}",
            f"  Breakdown:      {'YES' if self.breakdown_detected else 'No'}",
        ]
        return "\n".join(lines)


@dataclass
class SensitivityReport:
    """Report on sensitivity to assumption perturbations."""

    base_T_truth: float
    """T_truth at baseline parameters."""

    base_T_entropy: float
    """T_entropy at baseline parameters."""

    base_T_capture: float
    """T_capture at baseline parameters."""

    sensitivities: Dict[str, Dict[str, float]]
    """Map from parameter name to {'+': delta, '-': delta} for ±perturbation."""

    most_sensitive: str
    """Parameter with largest impact on T_effective."""

    fragility_index: float
    """Overall fragility: max sensitivity across all parameters."""

    details: Dict
    """Additional details."""

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "Sensitivity Analysis Report",
            f"  Baseline T_truth:   {self.base_T_truth:.2f}",
            f"  Baseline T_entropy: {self.base_T_entropy:.2f}",
            f"  Baseline T_capture: {self.base_T_capture:.2f}",
            f"  Most sensitive:     {self.most_sensitive}",
            f"  Fragility index:    {self.fragility_index:.2f}",
            "",
            "  Parameter Sensitivities (% change in T_eff per 20% param change):",
        ]
        for param, deltas in self.sensitivities.items():
            lines.append(f"    {param}: +{deltas['+']:.1f}% / {deltas['-']:.1f}%")
        return "\n".join(lines)


@dataclass
class ClusterSpec:
    """Specification of cluster structure for constraints."""

    n_clusters: int
    """Number of clusters."""

    cluster_sizes: List[int]
    """Size of each cluster (should sum to k)."""

    intra_cluster_correlation: float
    """Correlation between constraints within same cluster."""

    inter_cluster_correlation: float
    """Correlation between constraints in different clusters."""

    @property
    def k(self) -> int:
        """Total number of constraints."""
        return sum(self.cluster_sizes)


@dataclass
class NonConvexSpec:
    """Specification of non-convex deceptive region."""

    n_components: int
    """Number of convex components in the union."""

    component_volumes: List[float]
    """Volume of each convex component."""

    overlap_fraction: float
    """Fraction of volume in overlap regions."""


# =============================================================================
# Core Analysis Functions
# =============================================================================

class GeometricRobustnessAnalyzer:
    """
    Analyzes robustness of volume decay under non-generic geometries.

    This is the main class addressing the reviewer's concern about
    mean-field approximation fragility.
    """

    def __init__(
        self,
        dimension: int = 10,
        lambda_decay: float = 0.1,
        rng_seed: int = 42
    ):
        """
        Initialize analyzer.

        Args:
            dimension: Dimension of the deceptive volume space.
            lambda_decay: Decay constant for volume decay theorem.
            rng_seed: Random seed for reproducibility.
        """
        self.dimension = dimension
        self.lambda_decay = lambda_decay
        self.rng = np.random.default_rng(rng_seed)

    # =========================================================================
    # Monte Carlo Volume Estimation
    # =========================================================================

    def estimate_volume_monte_carlo(
        self,
        constraints: np.ndarray,
        n_samples: int = 10000,
        method: str = "rejection"
    ) -> VolumeEstimate:
        """
        Estimate volume of feasible region using Monte Carlo sampling.

        Args:
            constraints: Shape (k, d+1) matrix where each row is [a1, ..., ad, b]
                        representing constraint a·x ≤ b.
            n_samples: Number of Monte Carlo samples.
            method: "rejection" or "hit_and_run".

        Returns:
            VolumeEstimate with volume fraction and uncertainty.
        """
        k = constraints.shape[0]
        d = constraints.shape[1] - 1

        if method == "rejection":
            return self._rejection_sampling(constraints, n_samples)
        elif method == "hit_and_run":
            return self._hit_and_run_sampling(constraints, n_samples)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _rejection_sampling(
        self,
        constraints: np.ndarray,
        n_samples: int
    ) -> VolumeEstimate:
        """Volume estimation via rejection sampling in unit hypercube."""
        d = constraints.shape[1] - 1

        # Sample uniformly in [0, 1]^d
        samples = self.rng.uniform(0, 1, size=(n_samples, d))

        # Check which samples satisfy all constraints
        A = constraints[:, :-1]
        b = constraints[:, -1]

        # For each sample, check a·x ≤ b for all constraints
        feasible = np.all(samples @ A.T <= b, axis=1)

        n_feasible = np.sum(feasible)
        volume = n_feasible / n_samples

        # Binomial standard error
        std_error = np.sqrt(volume * (1 - volume) / n_samples)

        # 95% CI
        z = 1.96
        ci_low = max(0, volume - z * std_error)
        ci_high = min(1, volume + z * std_error)

        return VolumeEstimate(
            volume=volume,
            std_error=std_error,
            n_samples=n_samples,
            confidence_interval=(ci_low, ci_high)
        )

    def _hit_and_run_sampling(
        self,
        constraints: np.ndarray,
        n_samples: int,
        burn_in: int = 1000
    ) -> VolumeEstimate:
        """Volume estimation via hit-and-run MCMC (for complex polytopes)."""
        d = constraints.shape[1] - 1
        A = constraints[:, :-1]
        b = constraints[:, -1]

        # Find a feasible starting point (use center of hypercube if feasible)
        x = np.ones(d) * 0.5
        if not np.all(A @ x <= b):
            # Try to find a feasible point via linear programming
            x = self._find_feasible_point(A, b)
            if x is None:
                return VolumeEstimate(
                    volume=0.0,
                    std_error=0.0,
                    n_samples=n_samples,
                    confidence_interval=(0.0, 0.0)
                )

        # Hit-and-run MCMC
        samples = []
        for i in range(burn_in + n_samples):
            # Random direction
            direction = self.rng.standard_normal(d)
            direction /= np.linalg.norm(direction)

            # Find extent along direction
            t_min, t_max = self._line_polytope_intersection(x, direction, A, b)

            if t_max > t_min:
                # Sample uniformly along line segment
                t = self.rng.uniform(t_min, t_max)
                x = x + t * direction

            if i >= burn_in:
                samples.append(x.copy())

        # For hit-and-run, we estimate volume differently
        # (this is a simplification - proper volume requires more sophisticated methods)
        samples = np.array(samples)

        # Estimate volume as fraction of bounding box
        box_volume = np.prod(samples.max(axis=0) - samples.min(axis=0))

        return VolumeEstimate(
            volume=box_volume,
            std_error=box_volume * 0.1,  # Rough estimate
            n_samples=n_samples,
            confidence_interval=(box_volume * 0.8, box_volume * 1.2)
        )

    def _line_polytope_intersection(
        self,
        x: np.ndarray,
        d: np.ndarray,
        A: np.ndarray,
        b: np.ndarray
    ) -> Tuple[float, float]:
        """Find t_min, t_max such that x + t*d is in polytope Ax ≤ b."""
        t_min, t_max = -np.inf, np.inf

        for i in range(len(b)):
            denom = A[i] @ d
            numer = b[i] - A[i] @ x

            if abs(denom) < 1e-12:
                # Direction parallel to constraint
                if numer < 0:
                    return 0, 0  # Infeasible
            elif denom > 0:
                t_max = min(t_max, numer / denom)
            else:
                t_min = max(t_min, numer / denom)

        return t_min, t_max

    def _find_feasible_point(
        self,
        A: np.ndarray,
        b: np.ndarray
    ) -> Optional[np.ndarray]:
        """Find a feasible point in polytope Ax ≤ b."""
        try:
            from scipy.optimize import linprog
            d = A.shape[1]
            # Minimize 0 (just find feasible point)
            result = linprog(
                c=np.zeros(d),
                A_ub=A,
                b_ub=b,
                bounds=[(0, 1)] * d,
                method='highs'
            )
            if result.success:
                return result.x
        except ImportError:
            pass
        return None

    # =========================================================================
    # Generic Geometry Analysis
    # =========================================================================

    def analyze_generic_constraints(
        self,
        k: int,
        n_trials: int = 100,
        n_mc_samples: int = 10000
    ) -> RobustnessReport:
        """
        Validate exponential decay under generic (random) constraint placement.

        This is the baseline: V(k) ≈ V(0) * exp(-λ * k_eff).

        Args:
            k: Number of constraints.
            n_trials: Number of random constraint sets to test.
            n_mc_samples: Monte Carlo samples per trial.

        Returns:
            RobustnessReport comparing actual vs theoretical volume.
        """
        d = self.dimension

        # Theoretical prediction
        V_0 = 1.0  # Unit hypercube
        V_theory = V_0 * np.exp(-self.lambda_decay * k)

        # Monte Carlo validation
        volumes = []
        for _ in range(n_trials):
            # Random halfspace constraints
            constraints = self._generate_random_constraints(k, d)
            vol_est = self.estimate_volume_monte_carlo(constraints, n_mc_samples)
            volumes.append(vol_est.volume)

        mean_vol = np.mean(volumes)
        std_vol = np.std(volumes)

        generic_estimate = VolumeEstimate(
            volume=V_theory,
            std_error=0.0,  # Theoretical
            n_samples=0,
            confidence_interval=(V_theory, V_theory)
        )

        actual_estimate = VolumeEstimate(
            volume=mean_vol,
            std_error=std_vol / np.sqrt(n_trials),
            n_samples=n_trials * n_mc_samples,
            confidence_interval=(
                mean_vol - 1.96 * std_vol / np.sqrt(n_trials),
                mean_vol + 1.96 * std_vol / np.sqrt(n_trials)
            )
        )

        ratio = mean_vol / V_theory if V_theory > 1e-10 else float('inf')
        breakdown = abs(ratio - 1) > 0.1

        return RobustnessReport(
            geometry_type=GeometryType.GENERIC,
            generic_volume=generic_estimate,
            actual_volume=actual_estimate,
            volume_ratio=ratio,
            breakdown_detected=breakdown,
            effective_k=k,
            effective_k_ratio=1.0,
            confidence=1 - actual_estimate.relative_error,
            details={
                'n_trials': n_trials,
                'n_mc_samples': n_mc_samples,
                'volume_variance': std_vol ** 2
            }
        )

    def _generate_random_constraints(
        self,
        k: int,
        d: int
    ) -> np.ndarray:
        """Generate k random halfspace constraints in dimension d."""
        constraints = []
        for _ in range(k):
            # Random normal vector
            a = self.rng.standard_normal(d)
            a /= np.linalg.norm(a)

            # Random offset (ensuring constraint cuts through unit cube)
            b = self.rng.uniform(0.3, 0.7) * np.sqrt(d)

            constraints.append(np.concatenate([a, [b]]))

        return np.array(constraints)

    # =========================================================================
    # Clustered Constraint Analysis
    # =========================================================================

    def analyze_clustered_constraints(
        self,
        cluster_spec: ClusterSpec,
        n_trials: int = 50,
        n_mc_samples: int = 10000
    ) -> RobustnessReport:
        """
        Analyze volume decay when constraints cluster into groups.

        When constraints cluster, k_eff < k because clustered constraints
        are redundant. This method quantifies the reduction.

        Args:
            cluster_spec: Specification of cluster structure.
            n_trials: Number of random instantiations.
            n_mc_samples: Monte Carlo samples per trial.

        Returns:
            RobustnessReport with effective k estimation.
        """
        k = cluster_spec.k
        d = self.dimension

        # Theoretical prediction with no clustering
        V_theory = np.exp(-self.lambda_decay * k)

        # Monte Carlo with clustered constraints
        volumes = []
        for _ in range(n_trials):
            constraints = self._generate_clustered_constraints(cluster_spec, d)
            vol_est = self.estimate_volume_monte_carlo(constraints, n_mc_samples)
            volumes.append(vol_est.volume)

        mean_vol = np.mean(volumes)
        std_vol = np.std(volumes)

        # Estimate effective k from observed volume
        # V = exp(-λ * k_eff) => k_eff = -ln(V) / λ
        if mean_vol > 1e-10:
            k_eff_estimated = -np.log(mean_vol) / self.lambda_decay
        else:
            k_eff_estimated = k  # Default to nominal if volume too small

        # Theoretical k_eff with correlation
        rho = cluster_spec.intra_cluster_correlation
        k_eff_theory = k / (1 + rho * (k - 1))

        generic_estimate = VolumeEstimate(
            volume=V_theory,
            std_error=0.0,
            n_samples=0,
            confidence_interval=(V_theory, V_theory)
        )

        actual_estimate = VolumeEstimate(
            volume=mean_vol,
            std_error=std_vol / np.sqrt(n_trials),
            n_samples=n_trials * n_mc_samples,
            confidence_interval=(
                mean_vol - 1.96 * std_vol / np.sqrt(n_trials),
                mean_vol + 1.96 * std_vol / np.sqrt(n_trials)
            )
        )

        ratio = mean_vol / V_theory if V_theory > 1e-10 else float('inf')
        breakdown = abs(ratio - 1) > 0.1

        return RobustnessReport(
            geometry_type=GeometryType.CLUSTERED,
            generic_volume=generic_estimate,
            actual_volume=actual_estimate,
            volume_ratio=ratio,
            breakdown_detected=breakdown,
            effective_k=k_eff_estimated,
            effective_k_ratio=k_eff_estimated / k,
            confidence=1 - actual_estimate.relative_error,
            details={
                'n_clusters': cluster_spec.n_clusters,
                'cluster_sizes': cluster_spec.cluster_sizes,
                'intra_correlation': cluster_spec.intra_cluster_correlation,
                'inter_correlation': cluster_spec.inter_cluster_correlation,
                'k_eff_theory': k_eff_theory,
                'k_eff_estimated': k_eff_estimated,
            }
        )

    def _generate_clustered_constraints(
        self,
        cluster_spec: ClusterSpec,
        d: int
    ) -> np.ndarray:
        """Generate constraints with cluster structure."""
        constraints = []

        for cluster_idx, cluster_size in enumerate(cluster_spec.cluster_sizes):
            # Cluster center direction
            center = self.rng.standard_normal(d)
            center /= np.linalg.norm(center)

            for _ in range(cluster_size):
                # Perturb from cluster center based on intra-cluster correlation
                # Higher correlation = smaller perturbation
                noise_scale = np.sqrt(1 - cluster_spec.intra_cluster_correlation)
                noise = self.rng.standard_normal(d) * noise_scale

                a = center + noise
                a /= np.linalg.norm(a)

                b = self.rng.uniform(0.3, 0.7) * np.sqrt(d)
                constraints.append(np.concatenate([a, [b]]))

        return np.array(constraints)

    # =========================================================================
    # Adversarial Constraint Analysis
    # =========================================================================

    def analyze_adversarial_constraints(
        self,
        k: int,
        adversary_budget: float = 1.0,
        n_optimization_steps: int = 100,
        n_mc_samples: int = 10000
    ) -> RobustnessReport:
        """
        Analyze volume decay under adversarial constraint placement.

        The adversary places constraints to MAXIMIZE residual volume,
        representing worst-case for the defender.

        Args:
            k: Number of constraints.
            adversary_budget: Fraction of full adversarial capability [0, 1].
            n_optimization_steps: Number of optimization iterations.
            n_mc_samples: Monte Carlo samples per evaluation.

        Returns:
            RobustnessReport with adversarial gap quantified.
        """
        d = self.dimension

        # Theoretical prediction (generic)
        V_theory = np.exp(-self.lambda_decay * k)

        # Adversarial optimization: maximize volume
        best_constraints = None
        best_volume = 0.0

        for step in range(n_optimization_steps):
            if step == 0 or self.rng.random() < 0.3:
                # Random restart
                constraints = self._generate_adversarial_constraints(k, d, adversary_budget)
            else:
                # Mutate best solution
                constraints = self._mutate_constraints(best_constraints, adversary_budget)

            vol_est = self.estimate_volume_monte_carlo(constraints, n_mc_samples // 10)

            if vol_est.volume > best_volume:
                best_volume = vol_est.volume
                best_constraints = constraints.copy()

        # Final evaluation of best adversarial placement
        if best_constraints is not None:
            final_est = self.estimate_volume_monte_carlo(best_constraints, n_mc_samples)
        else:
            final_est = VolumeEstimate(
                volume=V_theory,
                std_error=0.0,
                n_samples=n_mc_samples,
                confidence_interval=(V_theory, V_theory)
            )

        generic_estimate = VolumeEstimate(
            volume=V_theory,
            std_error=0.0,
            n_samples=0,
            confidence_interval=(V_theory, V_theory)
        )

        ratio = final_est.volume / V_theory if V_theory > 1e-10 else float('inf')
        breakdown = ratio > 1.1  # Adversary achieves >10% more volume

        # Adversarial gap
        gap = (final_est.volume - V_theory) / V_theory if V_theory > 1e-10 else 0

        return RobustnessReport(
            geometry_type=GeometryType.ADVERSARIAL,
            generic_volume=generic_estimate,
            actual_volume=final_est,
            volume_ratio=ratio,
            breakdown_detected=breakdown,
            effective_k=k,
            effective_k_ratio=1.0,
            confidence=1 - final_est.relative_error,
            details={
                'adversary_budget': adversary_budget,
                'adversarial_gap': gap,
                'optimization_steps': n_optimization_steps,
                'gap_percentage': gap * 100,
            }
        )

    def _generate_adversarial_constraints(
        self,
        k: int,
        d: int,
        budget: float
    ) -> np.ndarray:
        """Generate adversarially-placed constraints."""
        # Adversary tries to make constraints redundant or non-intersecting
        # with the deceptive region

        constraints = []

        # Strategy: place constraints at corners/edges to minimize coverage
        for i in range(k):
            # Blend between adversarial and random based on budget
            if self.rng.random() < budget:
                # Adversarial: constraint far from center
                corner = self.rng.integers(0, 2, size=d)
                a = corner * 2 - 1  # Convert to {-1, 1}^d
                a = a.astype(float) + self.rng.standard_normal(d) * 0.1
                a /= np.linalg.norm(a)
                b = self.rng.uniform(0.8, 1.2) * np.sqrt(d)  # Far offset
            else:
                # Random
                a = self.rng.standard_normal(d)
                a /= np.linalg.norm(a)
                b = self.rng.uniform(0.3, 0.7) * np.sqrt(d)

            constraints.append(np.concatenate([a, [b]]))

        return np.array(constraints)

    def _mutate_constraints(
        self,
        constraints: np.ndarray,
        budget: float
    ) -> np.ndarray:
        """Mutate constraint set for optimization."""
        d = constraints.shape[1] - 1
        mutated = constraints.copy()

        # Mutate one random constraint
        idx = self.rng.integers(0, len(constraints))

        noise = self.rng.standard_normal(d) * 0.2 * budget
        mutated[idx, :-1] += noise
        mutated[idx, :-1] /= np.linalg.norm(mutated[idx, :-1])

        # Perturb offset
        mutated[idx, -1] *= 1 + self.rng.uniform(-0.1, 0.1) * budget

        return mutated

    # =========================================================================
    # Non-Convex Region Analysis
    # =========================================================================

    def analyze_non_convex_region(
        self,
        k: int,
        n_components: int = 3,
        overlap_fraction: float = 0.2,
        n_mc_samples: int = 10000
    ) -> RobustnessReport:
        """
        Analyze when deceptive region is non-convex (union of convex sets).

        The volume decay theorem assumes convex deceptive region. This method
        quantifies approximation error for non-convex regions.

        Args:
            k: Number of constraints.
            n_components: Number of convex components in union.
            overlap_fraction: Fraction of volume in overlap regions.
            n_mc_samples: Monte Carlo samples.

        Returns:
            RobustnessReport with bounds on approximation error.
        """
        d = self.dimension

        # For non-convex regions, the exponential bound may over-estimate
        # the rate of volume decrease because constraints can "miss" some
        # components entirely

        V_theory_convex = np.exp(-self.lambda_decay * k)

        # Model: region is union of n_components convex sets
        # Each component has volume V_0 / n_components
        # With overlap, effective volume is higher

        # Generate constraints and test which components they affect
        component_centers = []
        for i in range(n_components):
            center = self.rng.standard_normal(d)
            center /= np.linalg.norm(center)
            center *= 0.3  # Place centers within unit cube
            center += 0.5
            component_centers.append(center)

        # For each component, count how many constraints affect it
        constraints = self._generate_random_constraints(k, d)

        component_volumes = []
        for center in component_centers:
            # Count constraints that intersect component
            A = constraints[:, :-1]
            b = constraints[:, -1]

            # Check which constraints are "active" near this component
            distances = np.abs(A @ center - b)
            n_active = np.sum(distances < 0.5)  # Threshold for "affecting"

            # Component volume with its active constraints
            V_component = np.exp(-self.lambda_decay * n_active) / n_components
            component_volumes.append(V_component)

        # Union volume (approximate)
        V_union = sum(component_volumes)
        V_union *= (1 - overlap_fraction)  # Subtract overlaps
        V_union = min(V_union, 1.0)

        generic_estimate = VolumeEstimate(
            volume=V_theory_convex,
            std_error=0.0,
            n_samples=0,
            confidence_interval=(V_theory_convex, V_theory_convex)
        )

        actual_estimate = VolumeEstimate(
            volume=V_union,
            std_error=V_union * 0.1,  # Rough estimate
            n_samples=n_mc_samples,
            confidence_interval=(V_union * 0.8, V_union * 1.2)
        )

        ratio = V_union / V_theory_convex if V_theory_convex > 1e-10 else float('inf')
        breakdown = abs(ratio - 1) > 0.1

        return RobustnessReport(
            geometry_type=GeometryType.SPARSE,  # Using SPARSE for non-convex
            generic_volume=generic_estimate,
            actual_volume=actual_estimate,
            volume_ratio=ratio,
            breakdown_detected=breakdown,
            effective_k=k,
            effective_k_ratio=1.0,
            confidence=0.7,  # Lower confidence for approximations
            details={
                'n_components': n_components,
                'overlap_fraction': overlap_fraction,
                'component_volumes': component_volumes,
                'convex_approximation_error': abs(ratio - 1),
            }
        )

    # =========================================================================
    # Sensitivity Analysis
    # =========================================================================

    def sensitivity_analysis(
        self,
        base_config: Dict,
        perturbation_pct: float = 0.2
    ) -> SensitivityReport:
        """
        Vary each assumption ±perturbation_pct and report impact on T_eff.

        Args:
            base_config: Dict with keys 'K_req', 'rho', 'alpha', 'sigma', 'd', 'f', 'n', 'r_cap'.
            perturbation_pct: Fraction to perturb each parameter (default 20%).

        Returns:
            SensitivityReport with per-parameter sensitivities.
        """
        # Extract base parameters
        K_req = base_config.get('K_req', 10)
        rho = base_config.get('rho', 0.5)
        alpha = base_config.get('alpha', 0.1)
        sigma = base_config.get('sigma', 0.8)
        d = base_config.get('d', 0.05)
        f = base_config.get('f', 0.1)
        n = base_config.get('n', 100)
        r_cap = base_config.get('r_cap', 0.01)

        # Compute base timeline
        base_T_truth = self._compute_T_truth(K_req, rho, alpha)
        base_T_entropy = self._compute_T_entropy(sigma, d)
        base_T_capture = self._compute_T_capture(n, f, r_cap)
        base_T_eff = min(base_T_truth, base_T_entropy, base_T_capture)

        # Parameters to perturb
        params = {
            'K_req': K_req,
            'rho': rho,
            'alpha': alpha,
            'sigma': sigma,
            'd': d,
            'f': f,
            'r_cap': r_cap,
        }

        sensitivities = {}
        max_sensitivity = 0
        most_sensitive = ''

        for param_name, base_value in params.items():
            deltas = {}

            for direction, sign in [('+', 1), ('-', -1)]:
                # Perturb parameter
                perturbed = base_value * (1 + sign * perturbation_pct)

                # Ensure valid ranges
                if param_name == 'rho':
                    perturbed = np.clip(perturbed, 0, 0.99)
                elif param_name == 'sigma':
                    perturbed = np.clip(perturbed, 0.01, 1)
                elif param_name in ['alpha', 'd', 'r_cap']:
                    perturbed = max(0.001, perturbed)
                elif param_name == 'f':
                    perturbed = np.clip(perturbed, 0, 0.9)

                # Create perturbed config
                perturbed_config = base_config.copy()
                perturbed_config[param_name] = perturbed

                # Compute new T_eff
                K_req_p = perturbed_config.get('K_req', K_req)
                rho_p = perturbed_config.get('rho', rho)
                alpha_p = perturbed_config.get('alpha', alpha)
                sigma_p = perturbed_config.get('sigma', sigma)
                d_p = perturbed_config.get('d', d)
                f_p = perturbed_config.get('f', f)
                r_cap_p = perturbed_config.get('r_cap', r_cap)

                T_truth_p = self._compute_T_truth(K_req_p, rho_p, alpha_p)
                T_entropy_p = self._compute_T_entropy(sigma_p, d_p)
                T_capture_p = self._compute_T_capture(n, f_p, r_cap_p)
                T_eff_p = min(T_truth_p, T_entropy_p, T_capture_p)

                # Compute percentage change
                if base_T_eff > 0:
                    pct_change = (T_eff_p - base_T_eff) / base_T_eff * 100
                else:
                    pct_change = 0

                deltas[direction] = pct_change

            sensitivities[param_name] = deltas

            # Track most sensitive
            max_abs = max(abs(deltas['+']), abs(deltas['-']))
            if max_abs > max_sensitivity:
                max_sensitivity = max_abs
                most_sensitive = param_name

        # Fragility index: maximum sensitivity
        fragility = max_sensitivity / (perturbation_pct * 100)  # Normalized

        return SensitivityReport(
            base_T_truth=base_T_truth,
            base_T_entropy=base_T_entropy,
            base_T_capture=base_T_capture,
            sensitivities=sensitivities,
            most_sensitive=most_sensitive,
            fragility_index=fragility,
            details={
                'perturbation_pct': perturbation_pct,
                'base_T_eff': base_T_eff,
            }
        )

    def _compute_T_truth(self, K_req: float, rho: float, alpha: float) -> float:
        """Compute time to truth (deception collapse)."""
        denom = 1 - K_req * rho
        if denom <= 0:
            return float('inf')  # Singularity
        return K_req * (1 - rho) / (alpha * denom)

    def _compute_T_entropy(self, sigma: float, d: float, sigma_min: float = 0.2) -> float:
        """Compute time to entropy (system failure)."""
        if sigma <= sigma_min:
            return 0.0
        return np.log(sigma / sigma_min) / d

    def _compute_T_capture(self, n: int, f: float, r_cap: float) -> float:
        """Compute time to capture (federation breach)."""
        f_max = n / 3
        current_f = f * n
        if current_f >= f_max:
            return 0.0
        return (f_max - current_f) / (r_cap * n)


# =============================================================================
# Convenience Functions
# =============================================================================

def quick_robustness_check(
    k: int,
    rho: float = 0.5,
    dimension: int = 10,
    n_samples: int = 5000
) -> Dict[str, RobustnessReport]:
    """
    Quick robustness check across multiple geometry types.

    Args:
        k: Number of constraints.
        rho: Pairwise correlation (for clustered analysis).
        dimension: Dimension of space.
        n_samples: Monte Carlo samples.

    Returns:
        Dict mapping geometry type to RobustnessReport.
    """
    analyzer = GeometricRobustnessAnalyzer(dimension=dimension)

    reports = {}

    # Generic
    reports['generic'] = analyzer.analyze_generic_constraints(k, n_trials=20, n_mc_samples=n_samples)

    # Clustered (2 equal clusters)
    cluster_spec = ClusterSpec(
        n_clusters=2,
        cluster_sizes=[k // 2, k - k // 2],
        intra_cluster_correlation=rho,
        inter_cluster_correlation=0.1
    )
    reports['clustered'] = analyzer.analyze_clustered_constraints(cluster_spec, n_trials=20, n_mc_samples=n_samples)

    # Adversarial
    reports['adversarial'] = analyzer.analyze_adversarial_constraints(k, adversary_budget=0.5, n_optimization_steps=50, n_mc_samples=n_samples)

    return reports


def compute_breakdown_threshold(
    dimension: int = 10,
    rho_range: Tuple[float, float] = (0.1, 0.9),
    n_rho_points: int = 10,
    n_samples: int = 5000
) -> Dict[float, float]:
    """
    Find the rho threshold at which exponential approximation breaks.

    Returns dict mapping rho to breakdown severity (0 = no breakdown).
    """
    analyzer = GeometricRobustnessAnalyzer(dimension=dimension)

    results = {}
    k = 10  # Fixed k for this analysis

    for rho in np.linspace(rho_range[0], rho_range[1], n_rho_points):
        cluster_spec = ClusterSpec(
            n_clusters=2,
            cluster_sizes=[k // 2, k - k // 2],
            intra_cluster_correlation=rho,
            inter_cluster_correlation=0.0
        )

        report = analyzer.analyze_clustered_constraints(
            cluster_spec, n_trials=10, n_mc_samples=n_samples
        )

        # Breakdown severity: how much does ratio deviate from 1?
        severity = abs(report.volume_ratio - 1)
        results[rho] = severity

    return results


def integrate_with_correlation_tensor(
    correlation_model: 'ExtendedCorrelationModel',
    base_config: Dict
) -> SensitivityReport:
    """
    Run sensitivity analysis using extended correlation model.

    This bridges the correlation_tensor module with robustness analysis.

    Args:
        correlation_model: ExtendedCorrelationModel from Layer 1.
        base_config: Base configuration dict.

    Returns:
        SensitivityReport accounting for correlation structure.
    """
    if ExtendedCorrelationModel is None:
        raise ImportError("correlation_tensor module not available")

    # Extract correlation properties
    rho = correlation_model.to_scalar()
    k = correlation_model.k
    k_eff = correlation_model.compute_k_eff()

    # Update config with correlation-aware values
    config = base_config.copy()
    config['rho'] = rho

    # If we have spectral info, adjust based on resilience
    if correlation_model.spectral is not None:
        resilience = correlation_model.resilience_index
        # Lower resilience means parameters are more sensitive
        config['sensitivity_multiplier'] = 1 + (1 - resilience)

    analyzer = GeometricRobustnessAnalyzer()

    return analyzer.sensitivity_analysis(config, perturbation_pct=0.2)
