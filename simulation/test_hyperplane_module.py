#!/usr/bin/env python3
"""
Test suite for hyperplane intersection volume module.

Verifies theoretical predictions and basic functionality.
"""

from hyperplane_intersection_stdlib import TheoreticalAnalysis, ball_volume
import math


def test_ball_volume():
    """Test D-dimensional ball volume calculation."""
    print("Testing ball_volume()...")

    # Test cases with known values
    tests = [
        (2, 1.0, math.pi),  # Circle: πr²
        (3, 1.0, 4*math.pi/3),  # Sphere: 4πr³/3
        (2, 0.5, math.pi/4),  # Circle with r=0.5
    ]

    for D, r, expected in tests:
        result = ball_volume(D, r)
        error = abs(result - expected) / expected
        status = "PASS" if error < 0.01 else "FAIL"
        print(f"  D={D}, r={r}: {result:.6f} (expected {expected:.6f}) [{status}]")

    print()


def test_exponential_decay():
    """Verify exponential decay formula."""
    print("Testing exponential decay law...")

    analysis = TheoreticalAnalysis(D=10, c=1, radius=0.2)

    # Check that V(k) follows exponential decay
    v0 = analysis.volume_at_k(0)
    v1 = analysis.volume_at_k(1)
    v2 = analysis.volume_at_k(2)

    # Ratio should be constant
    ratio1 = v1 / v0
    ratio2 = v2 / v1
    expected_ratio = 1 - analysis.p_cut

    error1 = abs(ratio1 - expected_ratio) / expected_ratio
    error2 = abs(ratio2 - expected_ratio) / expected_ratio

    print(f"  V(1)/V(0) = {ratio1:.6f} (expected {expected_ratio:.6f})")
    print(f"  V(2)/V(1) = {ratio2:.6f} (expected {expected_ratio:.6f})")
    print(f"  Exponential decay: {'PASS' if error1 < 0.01 and error2 < 0.01 else 'FAIL'}")
    print()


def test_k_99_formula():
    """Test the k_99 ≈ 2.3/r scaling law."""
    print("Testing k_99 scaling law...")

    test_cases = [
        (0.1, 23),   # r=0.1 → k≈23
        (0.2, 12),   # r=0.2 → k≈12
        (0.3, 8),    # r=0.3 → k≈8
        (0.4, 6),    # r=0.4 → k≈6
    ]

    for r, expected_k in test_cases:
        analysis = TheoreticalAnalysis(D=10, c=1, radius=r)
        k_99 = analysis.k_for_reduction(0.99)

        # Allow ±2 tolerance
        error = abs(k_99 - expected_k)
        status = "PASS" if error <= 2 else "FAIL"

        print(f"  r={r}: k_99={k_99} (expected ≈{expected_k}) [{status}]")

    print()


def test_dimension_independence():
    """Verify that k_99 is independent of dimension D."""
    print("Testing dimension independence...")

    r = 0.2
    k_values = []

    for D in [2, 5, 10, 20, 50, 100]:
        analysis = TheoreticalAnalysis(D=D, c=1, radius=r)
        k_99 = analysis.k_for_reduction(0.99)
        k_values.append(k_99)

    # All k_99 should be the same
    all_same = all(k == k_values[0] for k in k_values)
    status = "PASS" if all_same else "FAIL"

    print(f"  r={r}, k_99 values across dimensions: {k_values}")
    print(f"  All equal: {status}")
    print()


def test_edge_cases():
    """Test edge cases and boundary conditions."""
    print("Testing edge cases...")

    # Very small radius
    analysis_small = TheoreticalAnalysis(D=10, c=1, radius=0.01)
    k_99_small = analysis_small.k_for_reduction(0.99)
    print(f"  Very small radius (r=0.01): k_99={k_99_small}")

    # Large radius
    analysis_large = TheoreticalAnalysis(D=3, c=1, radius=0.5)
    k_99_large = analysis_large.k_for_reduction(0.99)
    print(f"  Large radius (r=0.5): k_99={k_99_large}")

    # High dimension
    analysis_high_d = TheoreticalAnalysis(D=1000, c=1, radius=0.2)
    k_99_high_d = analysis_high_d.k_for_reduction(0.99)
    print(f"  Very high dimension (D=1000): k_99={k_99_high_d}")

    print()


def test_reduction_fractions():
    """Test different reduction fractions."""
    print("Testing various reduction fractions...")

    analysis = TheoreticalAnalysis(D=10, c=1, radius=0.2)

    reductions = [0.5, 0.9, 0.99, 0.999, 0.9999]

    for reduction in reductions:
        k = analysis.k_for_reduction(reduction)
        actual_volume = analysis.volume_at_k(k)
        actual_reduction = 1 - actual_volume / analysis.initial_volume

        print(f"  Target {reduction*100:.1f}% reduction: k={k}, "
              f"actual={actual_reduction*100:.2f}%")

    print()


def run_all_tests():
    """Run complete test suite."""
    print("="*70)
    print("HYPERPLANE INTERSECTION MODULE TEST SUITE")
    print("="*70)
    print()

    test_ball_volume()
    test_exponential_decay()
    test_k_99_formula()
    test_dimension_independence()
    test_edge_cases()
    test_reduction_fractions()

    print("="*70)
    print("ALL TESTS COMPLETED")
    print("="*70)


if __name__ == '__main__':
    run_all_tests()
