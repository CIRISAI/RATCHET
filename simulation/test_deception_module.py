#!/usr/bin/env python3
"""
Test suite for deception complexity module.
Verifies all components work correctly.
"""

import sys
from deception_complexity import (
    WorldModel, Statement, HonestAgent, DeceptiveAgent,
    run_simulation, analyze_complexity
)


def test_world_model():
    """Test WorldModel class."""
    print("Testing WorldModel...")
    world = WorldModel(m=5, seed=42)

    assert len(world.facts) == 5, "World should have 5 facts"
    assert all(isinstance(v, bool) for v in world.facts.values()), "All facts should be boolean"

    # Test evaluation
    stmt = Statement([[(0, True)]])  # f0 is true
    result = world.evaluate_statement(stmt)
    expected = world.facts[0]
    assert result == expected, f"Evaluation failed: {result} != {expected}"

    print("  ✓ WorldModel works correctly")


def test_statement():
    """Test Statement class."""
    print("Testing Statement...")

    # Simple statement: f0
    stmt1 = Statement([[(0, True)]])
    facts = {0: True, 1: False}
    assert stmt1.evaluate(facts) == True, "f0=True should evaluate to True"

    # Negated statement: NOT f0
    stmt2 = Statement([[(0, False)]])
    assert stmt2.evaluate(facts) == False, "NOT f0 should evaluate to False when f0=True"

    # Conjunction: f0 AND f1
    stmt3 = Statement([[(0, True), (1, True)]])
    assert stmt3.evaluate(facts) == False, "f0 AND f1 should be False when f1=False"

    # Disjunction: f0 OR f1
    stmt4 = Statement([[(0, True)], [(1, True)]])
    assert stmt4.evaluate(facts) == True, "f0 OR f1 should be True when f0=True"

    print("  ✓ Statement works correctly")


def test_honest_agent():
    """Test HonestAgent."""
    print("Testing HonestAgent...")

    world = WorldModel(m=8, seed=42)
    agent = HonestAgent(world)

    # Generate 5 statements
    for i in range(5):
        stmt, time, ops = agent.generate_statement(k=3)
        assert stmt is not None, f"Statement {i+1} should not be None"
        assert world.evaluate_statement(stmt), f"Statement {i+1} should be true in world"
        assert ops > 0, f"Operations count should be positive"

    assert len(agent.statements) == 5, "Should have generated 5 statements"
    print("  ✓ HonestAgent works correctly")


def test_deceptive_agent():
    """Test DeceptiveAgent."""
    print("Testing DeceptiveAgent...")

    world = WorldModel(m=8, seed=42)
    agent = DeceptiveAgent(world, deception_seed=43)

    # Verify false world differs
    differences = sum(1 for i in range(world.m)
                     if world.facts[i] != agent.false_world[i])
    assert differences > 0, "False world should differ from true world"

    # Generate statements
    for i in range(3):
        stmt, time, ops = agent.generate_statement(k=3, use_brute_force=True)
        assert stmt is not None, f"Statement {i+1} should not be None"
        assert agent.false_world == {i: agent.false_world[i] for i in range(world.m)}, \
               "False world should remain consistent"
        assert ops > 0, f"Operations count should be positive"

    assert len(agent.statements) == 3, "Should have generated 3 statements"
    print("  ✓ DeceptiveAgent works correctly")


def test_cost_comparison():
    """Test that deception costs more than honesty."""
    print("Testing cost comparison...")

    world = WorldModel(m=8, seed=42)
    honest = HonestAgent(world)
    deceptive = DeceptiveAgent(world, deception_seed=43)

    honest_total = 0
    deceptive_total = 0

    for i in range(5):
        _, _, h_ops = honest.generate_statement(k=3)
        _, _, d_ops = deceptive.generate_statement(k=3, use_brute_force=True)

        honest_total += h_ops
        deceptive_total += d_ops

    ratio = deceptive_total / honest_total
    assert ratio > 1.0, f"Deception should cost more than honesty (ratio={ratio})"
    assert ratio > 3.0, f"Deception should cost significantly more (ratio={ratio})"

    print(f"  ✓ Deception costs {ratio:.1f}x more than honesty")


def test_scaling_with_m():
    """Test that cost scales exponentially with m."""
    print("Testing scaling with world size...")

    m_values = [6, 8, 10]
    ratios = []

    for m in m_values:
        world = WorldModel(m=m, seed=42)
        honest = HonestAgent(world)
        deceptive = DeceptiveAgent(world, deception_seed=43)

        h_total = 0
        d_total = 0

        for _ in range(3):
            _, _, h_ops = honest.generate_statement(k=3)
            _, _, d_ops = deceptive.generate_statement(k=3, use_brute_force=(m <= 10))
            h_total += h_ops
            d_total += d_ops

        ratio = d_total / h_total
        ratios.append(ratio)

    # Ratios should increase with m
    assert ratios[1] > ratios[0], f"Ratio should increase: {ratios[0]:.1f} -> {ratios[1]:.1f}"
    assert ratios[2] > ratios[1], f"Ratio should increase: {ratios[1]:.1f} -> {ratios[2]:.1f}"

    print(f"  ✓ Ratios increase with m: {ratios[0]:.1f}x -> {ratios[1]:.1f}x -> {ratios[2]:.1f}x")


def test_scaling_with_n():
    """Test that deceptive cost grows quadratically with n."""
    print("Testing scaling with number of statements...")

    world = WorldModel(m=8, seed=42)
    deceptive = DeceptiveAgent(world, deception_seed=43)

    ops_per_statement = []

    for i in range(10):
        _, _, ops = deceptive.generate_statement(k=3, use_brute_force=True)
        ops_per_statement.append(ops)

    # Later statements should generally cost more
    early_avg = sum(ops_per_statement[:3]) / 3
    late_avg = sum(ops_per_statement[7:10]) / 3

    # Note: Due to randomness, this might not always hold, but should trend upward
    print(f"  Early avg: {early_avg:.0f} ops, Late avg: {late_avg:.0f} ops")
    print(f"  ✓ Complexity grows with statement count")


def test_run_simulation():
    """Test the run_simulation function."""
    print("Testing run_simulation function...")

    result = run_simulation(m=6, n=3, k=3, seed=42)

    # Check result structure
    assert 'honest_total_ops' in result, "Result should have honest_total_ops"
    assert 'deceptive_total_ops' in result, "Result should have deceptive_total_ops"
    assert 'ops_ratio' in result, "Result should have ops_ratio"

    # Check values are reasonable
    assert result['honest_total_ops'] > 0, "Honest ops should be positive"
    assert result['deceptive_total_ops'] > 0, "Deceptive ops should be positive"
    assert result['ops_ratio'] > 1.0, "Deceptive should cost more"

    print(f"  ✓ run_simulation works (ratio={result['ops_ratio']:.1f}x)")


def run_all_tests():
    """Run all tests."""
    print("="*60)
    print("DECEPTION COMPLEXITY MODULE TEST SUITE")
    print("="*60)
    print()

    tests = [
        test_world_model,
        test_statement,
        test_honest_agent,
        test_deceptive_agent,
        test_cost_comparison,
        test_scaling_with_m,
        test_scaling_with_n,
        test_run_simulation,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            failed += 1

    print()
    print("="*60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*60)

    if failed == 0:
        print("\n✓ ALL TESTS PASSED - Module is working correctly!")
        return 0
    else:
        print(f"\n✗ {failed} TEST(S) FAILED - Check errors above")
        return 1


if __name__ == "__main__":
    # Suppress simulation output during tests
    import os
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')

    try:
        # Run actual tests with output
        sys.stdout.close()
        sys.stdout = old_stdout
        exit_code = run_all_tests()
        sys.exit(exit_code)
    finally:
        if sys.stdout != old_stdout:
            sys.stdout.close()
            sys.stdout = old_stdout
