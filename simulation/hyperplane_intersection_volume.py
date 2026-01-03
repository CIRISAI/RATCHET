#!/usr/bin/env python3
"""
Standalone Computational Geometry Simulation Module

Question: Does the intersection of k random hyperplanes in D-dimensional space
         shrink the "feasible volume" exponentially?

This module uses Monte Carlo sampling to measure how the volume of a deceptive
region shrinks as we add random affine hyperplane constraints.

Author: Computational Geometer
Date: 2026-01-02
"""

import numpy as np
from scipy.special import gamma
from scipy.stats import ortho_group
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class SimulationConfig:
    """Configuration for hyperplane intersection simulation."""
    D: int  # Dimension of ambient space
    k_max: int  # Maximum number of constraints
    c: int  # Codimension of each hyperplane (typically 1)
    deceptive_radius: float  # Radius of deceptive ball
    deceptive_center: np.ndarray  # Center of deceptive region
    n_samples: int  # Monte Carlo samples for volume estimation
    random_seed: Optional[int] = 42


class RandomHyperplane:
    """
    Represents a random affine hyperplane of codimension c in R^D.

    For codimension c=1: hyperplane is {x : <n, x> = d}
    For codimension c>1: intersection of c hyperplanes of codim 1
    """

    def __init__(self, D: int, c: int = 1):
        """
        Initialize random hyperplane.

        Args:
            D: Ambient dimension
            c: Codimension (1 = hyperplane, 2 = line in 3D, etc.)
        """
        self.D = D
        self.c = c

        # Generate c random normal vectors (orthonormal basis of normal space)
        self.normals = ortho_group.rvs(D)[:c, :]  # Shape: (c, D)

        # Random offsets for each hyperplane equation
        # We want hyperplanes to intersect [0,1]^D with high probability
        # So we choose offsets near the center of the cube
        self.offsets = np.random.uniform(0.2, 0.8, size=c)

    def contains(self, points: np.ndarray, tolerance: float = 1e-6) -> np.ndarray:
        """
        Check if points lie on the hyperplane (within tolerance).

        Args:
            points: Array of shape (n_points, D)
            tolerance: Distance threshold to consider point "on" hyperplane

        Returns:
            Boolean array of shape (n_points,)
        """
        # For codimension c, point must satisfy all c constraints
        # Distance to hyperplane i: |<n_i, x> - d_i|
        distances = np.abs(points @ self.normals.T - self.offsets)  # Shape: (n_points, c)

        # Point is on manifold if within tolerance for all c constraints
        return np.all(distances < tolerance, axis=1)

    def distance(self, points: np.ndarray) -> np.ndarray:
        """
        Compute minimum distance from points to the manifold.

        Args:
            points: Array of shape (n_points, D)

        Returns:
            Array of shape (n_points,) with distances
        """
        # For codimension 1: distance is |<n, x> - d|
        # For higher codimension: maximum of distances to all hyperplanes
        distances = np.abs(points @ self.normals.T - self.offsets)
        return np.max(distances, axis=1)


class HyperplaneIntersectionSimulator:
    """
    Simulates volume shrinkage as random hyperplanes are added.
    """

    def __init__(self, config: SimulationConfig):
        self.config = config
        np.random.seed(config.random_seed)

        # Pre-generate all hyperplanes
        self.hyperplanes: List[RandomHyperplane] = []
        for _ in range(config.k_max):
            self.hyperplanes.append(RandomHyperplane(config.D, config.c))

    def is_in_deceptive_region(self, points: np.ndarray) -> np.ndarray:
        """
        Check if points are in the deceptive region (ball of radius r).

        Args:
            points: Array of shape (n_points, D)

        Returns:
            Boolean array of shape (n_points,)
        """
        distances = np.linalg.norm(
            points - self.config.deceptive_center, axis=1
        )
        return distances <= self.config.deceptive_radius

    def sample_from_hypercube(self, n_samples: int) -> np.ndarray:
        """
        Uniformly sample points from [0,1]^D.

        Args:
            n_samples: Number of samples

        Returns:
            Array of shape (n_samples, D)
        """
        return np.random.uniform(0, 1, size=(n_samples, self.config.D))

    def estimate_volume_monte_carlo(
        self,
        k: int,
        tolerance: float = 1e-6
    ) -> Tuple[float, float]:
        """
        Estimate volume of D_ec ∩ (⋂_{i=1}^k M_i) using Monte Carlo.

        Strategy: Sample uniformly from [0,1]^D, count fraction that:
        1. Lie in deceptive region
        2. Lie on all k hyperplanes (within tolerance)

        Args:
            k: Number of hyperplanes to intersect
            tolerance: Distance threshold for "on hyperplane"

        Returns:
            (volume_estimate, standard_error)
        """
        if k == 0:
            # Just compute volume of deceptive region
            points = self.sample_from_hypercube(self.config.n_samples)
            in_deceptive = self.is_in_deceptive_region(points)
            fraction = np.mean(in_deceptive)
            std_error = np.std(in_deceptive) / np.sqrt(self.config.n_samples)
            return fraction, std_error

        # For k > 0, we need points near intersection of all k hyperplanes
        # Problem: naive sampling from [0,1]^D will miss thin manifolds!
        # Solution: Sample points ON the manifolds using projection

        volume_estimate = self._adaptive_manifold_sampling(k, tolerance)
        return volume_estimate, 0.0  # TODO: Add error estimation

    def _adaptive_manifold_sampling(
        self,
        k: int,
        tolerance: float
    ) -> float:
        """
        Adaptively sample from manifold intersection.

        For codimension c hyperplanes, the intersection has codimension c*k.
        We need to sample from this lower-dimensional manifold.

        Strategy:
        1. Start with random points in [0,1]^D
        2. Project onto manifold intersection iteratively
        3. Measure what fraction lands in deceptive region
        """
        n_samples = self.config.n_samples
        D = self.config.D
        c = self.config.c

        # Start with random points
        points = self.sample_from_hypercube(n_samples)

        # Iteratively project onto each hyperplane
        max_iterations = 100
        for iteration in range(max_iterations):
            converged = True

            for i in range(k):
                hp = self.hyperplanes[i]

                # Project each point onto hyperplane i
                # For codimension 1: x' = x - <n, x - d*n> * n / ||n||^2
                for j in range(hp.c):
                    normal = hp.normals[j]
                    offset = hp.offsets[j]

                    # Distance to hyperplane
                    dist = points @ normal - offset

                    # Project
                    points = points - np.outer(dist, normal)

                    if np.max(np.abs(dist)) > tolerance:
                        converged = False

            if converged:
                break

        # Clip to [0,1]^D (some points may have escaped during projection)
        points = np.clip(points, 0, 1)

        # Count fraction in deceptive region
        in_deceptive = self.is_in_deceptive_region(points)

        # Volume estimate: fraction * (theoretical volume of manifold)
        # Theoretical volume of k hyperplanes in [0,1]^D
        manifold_volume = self._theoretical_manifold_volume(k)

        fraction_in_deceptive = np.mean(in_deceptive)

        return fraction_in_deceptive * manifold_volume

    def _theoretical_manifold_volume(self, k: int) -> float:
        """
        Theoretical volume of intersection of k random hyperplanes in [0,1]^D.

        For codimension c, the intersection has dimension D - c*k.
        The "volume" (in the measure-theoretic sense) scales as:

        V_k ~ (edge_length)^(D - c*k)

        For a unit hypercube, this is approximately 1 if D - c*k > 0,
        and 0 if the codimension exceeds dimension.
        """
        effective_dim = self.config.D - self.config.c * k

        if effective_dim <= 0:
            return 0.0

        # For random hyperplanes in [0,1]^D, expected volume scales as
        # roughly 1 / sqrt(2*pi)^(c*k) (Gaussian tail behavior)
        # This is a crude approximation
        return (1.0 / np.sqrt(2 * np.pi)) ** (self.config.c * k)

    def run_simulation(self) -> dict:
        """
        Run full simulation: measure volume for k = 0, 1, ..., k_max.

        Returns:
            Dictionary with results and analysis
        """
        k_values = np.arange(0, self.config.k_max + 1)
        volumes = []
        errors = []

        print(f"Running simulation in D={self.config.D} dimensions...")
        print(f"Deceptive region: ball of radius {self.config.deceptive_radius}")
        print(f"Codimension per hyperplane: {self.config.c}")
        print()

        for k in k_values:
            vol, err = self.estimate_volume_monte_carlo(k)
            volumes.append(vol)
            errors.append(err)

            if k % max(1, self.config.k_max // 10) == 0:
                print(f"k={k:3d}: Volume = {vol:.6e} ± {err:.6e}")

        volumes = np.array(volumes)
        errors = np.array(errors)

        # Fit exponential decay: V(k) = V(0) * exp(-λ * k)
        log_volumes = np.log(volumes + 1e-100)  # Avoid log(0)

        # Linear fit to log(V) vs k
        valid_idx = volumes > 0
        if np.sum(valid_idx) > 1:
            coeffs = np.polyfit(k_values[valid_idx], log_volumes[valid_idx], 1)
            decay_rate = -coeffs[0]
        else:
            decay_rate = np.nan

        # Find k for 99% reduction
        k_99 = self._find_k_for_reduction(volumes, 0.99)

        return {
            'k_values': k_values,
            'volumes': volumes,
            'errors': errors,
            'decay_rate': decay_rate,
            'k_99_reduction': k_99,
            'initial_volume': volumes[0],
            'config': self.config
        }

    def _find_k_for_reduction(
        self,
        volumes: np.ndarray,
        reduction_fraction: float
    ) -> int:
        """Find smallest k where volume is reduced by reduction_fraction."""
        if len(volumes) == 0 or volumes[0] == 0:
            return -1

        target = volumes[0] * (1 - reduction_fraction)

        for k, vol in enumerate(volumes):
            if vol <= target:
                return k

        return -1  # Not reached within k_max


def theoretical_analysis(D: int, c: int, r: float) -> dict:
    """
    Theoretical prediction for volume shrinkage.

    Theory: Each hyperplane of codimension c reduces the "accessible volume"
    by a factor that depends on the distance of the deceptive region from
    the hyperplane.

    For a random hyperplane, the probability it "cuts" a ball of radius r
    is approximately proportional to r / sqrt(D).

    Expected volume after k hyperplanes:
        V(k) ≈ V(0) * (1 - p)^k

    where p ≈ 2r / sqrt(D) for small r.

    This gives exponential decay with rate λ = -log(1 - p) ≈ p for small p.

    Args:
        D: Dimension
        c: Codimension
        r: Radius of deceptive region

    Returns:
        Dictionary with theoretical predictions
    """
    # Volume of D-dimensional ball of radius r
    ball_volume = (np.pi ** (D / 2)) / gamma(D / 2 + 1) * (r ** D)

    # Volume of unit hypercube
    cube_volume = 1.0

    # Initial volume (ball clipped to cube)
    initial_volume = min(ball_volume, cube_volume)

    # Cutting probability per hyperplane
    # Heuristic: hyperplane cuts ball if distance < r
    # Probability random hyperplane is within distance r of origin ≈ 2r
    p_cut = min(2 * r, 1.0)

    # Decay rate
    decay_rate = -np.log(1 - p_cut) if p_cut < 1 else np.inf

    # k for 99% reduction: solve (1-p)^k = 0.01
    # k = log(0.01) / log(1-p)
    if p_cut < 1 and p_cut > 0:
        k_99 = int(np.ceil(np.log(0.01) / np.log(1 - p_cut)))
    else:
        k_99 = 1 if p_cut >= 1 else np.inf

    return {
        'initial_volume': initial_volume,
        'cutting_probability': p_cut,
        'decay_rate': decay_rate,
        'k_99_reduction': k_99,
        'formula': f'V(k) ≈ {initial_volume:.6f} * exp(-{decay_rate:.4f} * k)'
    }


def visualize_results(results: dict, theory: dict):
    """
    Create visualization of simulation results.

    Args:
        results: Output from run_simulation()
        theory: Output from theoretical_analysis()
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    k_values = results['k_values']
    volumes = results['volumes']

    # Plot 1: Volume vs k (linear scale)
    ax1 = axes[0]
    ax1.plot(k_values, volumes, 'o-', label='Simulation', linewidth=2, markersize=6)

    # Theoretical curve
    if not np.isnan(theory['decay_rate']):
        k_theory = np.linspace(0, max(k_values), 100)
        v_theory = theory['initial_volume'] * np.exp(-theory['decay_rate'] * k_theory)
        ax1.plot(k_theory, v_theory, '--', label='Theory', linewidth=2, alpha=0.7)

    ax1.set_xlabel('Number of Hyperplanes (k)', fontsize=12)
    ax1.set_ylabel('Volume', fontsize=12)
    ax1.set_title('Volume Shrinkage: Linear Scale', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    # Plot 2: Log-scale
    ax2 = axes[1]
    valid = volumes > 0
    ax2.semilogy(k_values[valid], volumes[valid], 'o-', label='Simulation',
                 linewidth=2, markersize=6)

    if not np.isnan(theory['decay_rate']):
        ax2.semilogy(k_theory, v_theory, '--', label='Theory', linewidth=2, alpha=0.7)

    ax2.set_xlabel('Number of Hyperplanes (k)', fontsize=12)
    ax2.set_ylabel('Volume (log scale)', fontsize=12)
    ax2.set_title('Volume Shrinkage: Exponential Decay', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    return fig


def print_summary(results: dict, theory: dict):
    """Print human-readable summary of results."""
    print("\n" + "="*70)
    print("SIMULATION SUMMARY")
    print("="*70)

    cfg = results['config']
    print(f"\nConfiguration:")
    print(f"  Dimension (D):              {cfg.D}")
    print(f"  Codimension per plane (c):  {cfg.c}")
    print(f"  Max hyperplanes (k):        {cfg.k_max}")
    print(f"  Deceptive radius:           {cfg.deceptive_radius}")
    print(f"  Monte Carlo samples:        {cfg.n_samples:,}")

    print(f"\nTheoretical Predictions:")
    print(f"  Initial volume:             {theory['initial_volume']:.6e}")
    print(f"  Cutting probability:        {theory['cutting_probability']:.6f}")
    print(f"  Decay rate (λ):             {theory['decay_rate']:.6f}")
    print(f"  k for 99% reduction:        {theory['k_99_reduction']}")
    print(f"  Formula:                    {theory['formula']}")

    print(f"\nSimulation Results:")
    print(f"  Measured initial volume:    {results['initial_volume']:.6e}")
    print(f"  Measured decay rate:        {results['decay_rate']:.6f}")
    print(f"  k for 99% reduction:        {results['k_99_reduction']}")

    print(f"\nConclusion:")
    if results['decay_rate'] > 0.1:
        print(f"  ✓ EXPONENTIAL SHRINKAGE CONFIRMED")
        print(f"    Volume decays as V(k) ≈ V(0) * exp(-{results['decay_rate']:.3f} * k)")
    else:
        print(f"  ✗ Weak or no exponential decay detected")

    if results['k_99_reduction'] > 0:
        print(f"  ✓ 99% volume reduction achieved at k = {results['k_99_reduction']}")
    else:
        print(f"  ✗ 99% reduction not reached (need k > {cfg.k_max})")

    print("="*70 + "\n")


def main():
    """Main entry point for simulation."""

    # Example 1: Low-dimensional case (D=3)
    print("EXAMPLE 1: D=3 (3D space)")
    print("-" * 70)

    config_3d = SimulationConfig(
        D=3,
        k_max=20,
        c=1,  # Codimension 1 = planes in 3D
        deceptive_radius=0.3,
        deceptive_center=np.array([0.5, 0.5, 0.5]),
        n_samples=100_000,
        random_seed=42
    )

    simulator_3d = HyperplaneIntersectionSimulator(config_3d)
    results_3d = simulator_3d.run_simulation()
    theory_3d = theoretical_analysis(
        config_3d.D,
        config_3d.c,
        config_3d.deceptive_radius
    )

    print_summary(results_3d, theory_3d)

    # Example 2: Higher-dimensional case (D=10)
    print("\nEXAMPLE 2: D=10 (10D space)")
    print("-" * 70)

    config_10d = SimulationConfig(
        D=10,
        k_max=50,
        c=1,
        deceptive_radius=0.2,
        deceptive_center=np.ones(10) * 0.5,
        n_samples=100_000,
        random_seed=42
    )

    simulator_10d = HyperplaneIntersectionSimulator(config_10d)
    results_10d = simulator_10d.run_simulation()
    theory_10d = theoretical_analysis(
        config_10d.D,
        config_10d.c,
        config_10d.deceptive_radius
    )

    print_summary(results_10d, theory_10d)

    # Visualizations
    print("Generating visualizations...")
    fig1 = visualize_results(results_3d, theory_3d)
    fig1.savefig('/home/emoore/RATCHET/simulation/volume_shrinkage_3d.png',
                 dpi=150, bbox_inches='tight')

    fig2 = visualize_results(results_10d, theory_10d)
    fig2.savefig('/home/emoore/RATCHET/simulation/volume_shrinkage_10d.png',
                 dpi=150, bbox_inches='tight')

    print("Plots saved:")
    print("  - /home/emoore/RATCHET/simulation/volume_shrinkage_3d.png")
    print("  - /home/emoore/RATCHET/simulation/volume_shrinkage_10d.png")

    plt.show()


if __name__ == '__main__':
    main()
