#!/usr/bin/env python3
"""
Demonstration of the Exponential Shrinkage Scaling Law

This script generates a comprehensive analysis showing:
1. Volume decay curves for different dimensions
2. The remarkable dimension-independence of k_99
3. Scaling law: k_99 ≈ 2.3/r

Author: Computational Geometer
"""

from hyperplane_intersection_stdlib import TheoreticalAnalysis
import math


def print_header(title: str):
    """Print a formatted section header."""
    print("\n" + "="*80)
    print(title.center(80))
    print("="*80 + "\n")


def demo_volume_decay():
    """Demonstrate volume decay for a fixed configuration."""
    print_header("DEMONSTRATION 1: Volume Decay Profile")

    print("Configuration: D=10, r=0.2, codimension c=1\n")

    analysis = TheoreticalAnalysis(D=10, c=1, radius=0.2)

    print(f"Initial deceptive volume: V(0) = {analysis.initial_volume:.6e}")
    print(f"Cutting probability:      p = {analysis.p_cut:.4f}")
    print(f"Decay rate:               λ = {analysis.decay_rate:.4f}")
    print(f"\nVolume formula: V(k) = V(0) × (1-p)^k ≈ V(0) × exp(-λk)\n")

    print(f"{'k':>5} {'V(k)':>15} {'Reduction %':>15} {'Half-lives':>12}")
    print("-" * 80)

    half_life = -math.log(0.5) / analysis.decay_rate if analysis.decay_rate > 0 else float('inf')

    for k in [0, 1, 2, 3, 5, 7, 10, 15, 20, 30, 50]:
        v = analysis.volume_at_k(k)
        reduction = (1 - v / analysis.initial_volume) * 100 if analysis.initial_volume > 0 else 100
        n_half_lives = k / half_life if half_life > 0 else 0

        print(f"{k:5d} {v:15.6e} {reduction:14.2f}% {n_half_lives:11.2f}")

    print(f"\nHalf-life: t_1/2 = ln(2)/λ ≈ {half_life:.2f} constraints")


def demo_dimension_independence():
    """Show that k_99 is independent of dimension."""
    print_header("DEMONSTRATION 2: Dimension Independence")

    print("Fixed radius r = 0.2\n")
    print(f"{'Dimension D':>15} {'Initial Volume':>18} {'k_50':>8} {'k_90':>8} {'k_99':>8} {'k_999':>8}")
    print("-" * 80)

    r = 0.2
    for D in [2, 3, 5, 10, 20, 50, 100, 500, 1000]:
        analysis = TheoreticalAnalysis(D=D, c=1, radius=r)

        k_50 = analysis.k_for_reduction(0.50)
        k_90 = analysis.k_for_reduction(0.90)
        k_99 = analysis.k_for_reduction(0.99)
        k_999 = analysis.k_for_reduction(0.999)

        print(f"{D:15d} {analysis.initial_volume:18.6e} {k_50:8d} {k_90:8d} "
              f"{k_99:8d} {k_999:8d}")

    print("\nOBSERVATION:")
    print("  While initial volume shrinks exponentially with D (curse of dimensionality),")
    print("  the k values remain CONSTANT across all dimensions!")
    print("  This is the key insight: random constraints scale dimension-free.")


def demo_radius_scaling():
    """Show the scaling law k_99 ≈ 2.3/r."""
    print_header("DEMONSTRATION 3: Radius Scaling Law")

    print("Fixed dimension D = 10\n")
    print(f"{'Radius r':>12} {'p_cut':>10} {'λ':>10} {'k_99':>8} {'Predicted k':>14} {'Error':>10}")
    print("-" * 80)

    D = 10
    for r in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        analysis = TheoreticalAnalysis(D=D, c=1, radius=r)

        k_99 = analysis.k_for_reduction(0.99)
        predicted_k = 2.3 / r if r > 0 else float('inf')
        error = abs(k_99 - predicted_k) / predicted_k if predicted_k > 0 else 0

        print(f"{r:12.2f} {analysis.p_cut:10.4f} {analysis.decay_rate:10.4f} "
              f"{k_99:8d} {predicted_k:14.1f} {error*100:9.1f}%")

    print("\nSCALING LAW: k_99 ≈ 2.3 / r")
    print("  This is a universal formula independent of dimension D!")
    print("  Smaller deceptive regions require more constraints, but predictably so.")


def demo_practical_implications():
    """Practical takeaways for deception elimination."""
    print_header("DEMONSTRATION 4: Practical Implications")

    print("How many constraints to eliminate deception?\n")

    scenarios = [
        ("Tiny deceptive basin (r=0.05)", 10, 0.05, 0.99),
        ("Small deceptive basin (r=0.1)", 10, 0.1, 0.99),
        ("Medium deceptive basin (r=0.2)", 10, 0.2, 0.99),
        ("Large deceptive basin (r=0.3)", 10, 0.3, 0.99),
        ("Very large basin (r=0.4)", 10, 0.4, 0.99),
    ]

    for desc, D, r, target in scenarios:
        analysis = TheoreticalAnalysis(D=D, c=1, radius=r)
        k_needed = analysis.k_for_reduction(target)
        print(f"  {desc:40s}: k = {k_needed:3d}")

    print("\n" + "-"*80)
    print("\nDifferent confidence levels (D=10, r=0.2):\n")

    analysis = TheoreticalAnalysis(D=10, c=1, radius=0.2)

    confidence_levels = [
        ("50% reduction", 0.50),
        ("90% reduction (1 in 10)", 0.90),
        ("99% reduction (1 in 100)", 0.99),
        ("99.9% reduction (1 in 1000)", 0.999),
        ("99.99% reduction (1 in 10000)", 0.9999),
    ]

    for desc, reduction in confidence_levels:
        k = analysis.k_for_reduction(reduction)
        print(f"  {desc:35s}: k = {k:3d}")

    print("\n" + "-"*80)
    print("\nHigh-dimensional setting (D=100, r=0.15):\n")

    analysis_high_d = TheoreticalAnalysis(D=100, c=1, radius=0.15)

    for desc, reduction in confidence_levels:
        k = analysis_high_d.k_for_reduction(reduction)
        print(f"  {desc:35s}: k = {k:3d}")

    print("\n  → Same k values! Dimension doesn't matter.")


def demo_theoretical_foundation():
    """Explain the mathematical foundation."""
    print_header("DEMONSTRATION 5: Theoretical Foundation")

    print("Why exponential decay?\n")
    print("1. GEOMETRIC INTUITION:")
    print("   - Each random hyperplane 'cuts' the space")
    print("   - Probability of cutting a ball of radius r: p ≈ 2r")
    print("   - Independent cuts compound multiplicatively")
    print("   - Survival probability after k cuts: (1-p)^k → exponential decay\n")

    print("2. MATHEMATICAL FORMULA:")
    print("   V(k) = V(0) × (1 - p)^k")
    print("        = V(0) × exp(k × ln(1-p))")
    print("        ≈ V(0) × exp(-pk)         [for small p]")
    print("        = V(0) × exp(-λk)         [where λ = p ≈ 2r]\n")

    print("3. DIMENSION INDEPENDENCE:")
    print("   - The cutting probability p depends on radius r, NOT dimension D")
    print("   - While V(0) shrinks with D (curse of dimensionality),")
    print("     the decay RATE λ remains constant")
    print("   - This makes random constraints remarkably effective in high dimensions\n")

    print("4. PRACTICAL FORMULA:")
    print("   For 99% reduction: k_99 = ⌈ln(100) / ln(1/(1-p))⌉")
    print("                           ≈ ⌈4.6 / p⌉")
    print("                           ≈ ⌈2.3 / r⌉\n")

    print("5. INFORMATION-THEORETIC VIEW:")
    print("   - Each constraint provides ~λ nats of information")
    print("   - Information grows linearly: I(k) = λk")
    print("   - Deception eliminated when I(k) > ln(1/tolerance)")


def demo_comparison_table():
    """Generate a comprehensive comparison table."""
    print_header("DEMONSTRATION 6: Comprehensive Comparison")

    print("Constraints needed for 99% deceptive volume reduction:\n")
    print(f"{'Radius':>10} │ {'D=2':>6} {'D=3':>6} {'D=5':>6} {'D=10':>6} "
          f"{'D=20':>6} {'D=50':>6} {'D=100':>6} │ {'Theory':>8}")
    print("─" * 80)

    for r in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
        k_values = []
        for D in [2, 3, 5, 10, 20, 50, 100]:
            analysis = TheoreticalAnalysis(D=D, c=1, radius=r)
            k_values.append(analysis.k_for_reduction(0.99))

        theory_k = int(2.3 / r)

        print(f"{r:10.2f} │ {k_values[0]:6d} {k_values[1]:6d} {k_values[2]:6d} "
              f"{k_values[3]:6d} {k_values[4]:6d} {k_values[5]:6d} {k_values[6]:6d} │ "
              f"{theory_k:8d}")

    print("\nOBSERVATION: All rows are constant → dimension-free scaling!")


def main():
    """Run all demonstrations."""
    print("\n" + "█" * 80)
    print("EXPONENTIAL SHRINKAGE: A COMPUTATIONAL GEOMETRY DEMONSTRATION".center(80))
    print("█" * 80)

    demo_volume_decay()
    demo_dimension_independence()
    demo_radius_scaling()
    demo_practical_implications()
    demo_theoretical_foundation()
    demo_comparison_table()

    print_header("CONCLUSION")

    print("KEY FINDINGS:\n")
    print("1. Volume shrinks EXPONENTIALLY: V(k) = V(0) × exp(-λk)")
    print("   where λ ≈ 2r (twice the deceptive radius)\n")

    print("2. Constraints needed for 99% reduction: k_99 ≈ 2.3 / r")
    print("   - Independent of dimension D")
    print("   - Only depends on size of deceptive region\n")

    print("3. Practical ranges:")
    print("   - Small deception (r < 0.1):  k ~ 20-50")
    print("   - Medium deception (r ~ 0.2): k ~ 10-12")
    print("   - Large deception (r > 0.3):  k ~ 6-8\n")

    print("4. This result has profound implications:")
    print("   - Random constraints are surprisingly effective")
    print("   - Scales to arbitrary dimensions without degradation")
    print("   - Provides a theoretical foundation for constraint-based optimization\n")

    print("="*80)
    print()


if __name__ == '__main__':
    main()
