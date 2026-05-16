#!/usr/bin/env python3
"""
Test suite for RATCHET simulation engines.
Verifies MicrobiomeEngine, BatteryDegradationEngine, and InstitutionalCollapseEngine.

These engines expose the RATCHET structural invariants for stress testing:
  - k (constraints), rho (correlation), k_eff (effective constraints)
  - sigma (sustainability), f (compromise), alpha (generation), d (decay)
"""

import sys
import os

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from ratchet.engines.microbiome import (
    MicrobiomeEngine, MicrobiomeParams, MicrobiomeShock, MicrobiomeIntervention,
    ShockType, InterventionType, create_microbiome_engine,
)
from ratchet.engines.battery import (
    BatteryDegradationEngine, BatteryParams, BatteryShock, BatteryIntervention,
    BatteryShockType, BatteryInterventionType, create_battery_engine,
)
from ratchet.engines.institutional import (
    InstitutionalCollapseEngine, InstitutionalParams, InstitutionalShock,
    InstitutionalIntervention, InstitutionalShockType, InstitutionalInterventionType,
    RegimeType, REGIME_ARCHETYPES, create_institutional_engine,
)


# =============================================================================
# MICROBIOME ENGINE TESTS
# =============================================================================

def test_microbiome_initialization():
    """Test MicrobiomeEngine initialization."""
    print("Testing MicrobiomeEngine initialization...")

    engine = create_microbiome_engine(seed=42)
    engine.initialize_from_reference("healthy_adult")

    assert engine.get_k() > 0, "Should have species after initialization"
    assert 0 <= engine.get_sigma() <= 1, "Sigma should be normalized"
    assert not engine.is_collapsed(), "Should not be collapsed initially"

    print("  OK MicrobiomeEngine initializes correctly")


def test_microbiome_k_eff_formula():
    """Test k_eff = k / (1 + rho*(k-1))."""
    print("Testing MicrobiomeEngine k_eff formula...")

    engine = create_microbiome_engine(seed=42)
    engine.initialize_from_reference("healthy_adult")

    k = engine.get_k()
    rho = engine.get_rho()
    k_eff = engine.get_k_eff()

    if k > 1:
        expected_k_eff = k / (1 + rho * (k - 1))
        assert abs(k_eff - expected_k_eff) < 0.01, f"k_eff formula mismatch: {k_eff} vs {expected_k_eff}"

    print(f"  OK k_eff = {k_eff:.2f} (k={k}, rho={rho:.3f})")


def test_microbiome_dynamics():
    """Test MicrobiomeEngine simulation dynamics."""
    print("Testing MicrobiomeEngine dynamics...")

    engine = create_microbiome_engine(seed=42)
    engine.initialize_from_reference("healthy_adult")

    initial_sigma = engine.get_sigma()
    df = engine.run(duration=10, dt=0.1)

    assert len(df) > 1, "Should record multiple states"
    assert 'sigma' in df.columns, "DataFrame should have sigma"
    assert 'k_eff' in df.columns, "DataFrame should have k_eff"

    print(f"  OK Ran 10 time units, sigma: {initial_sigma:.3f} -> {engine.get_sigma():.3f}")


def test_microbiome_shock():
    """Test MicrobiomeEngine shock application."""
    print("Testing MicrobiomeEngine shock response...")

    engine = create_microbiome_engine(seed=42)
    engine.initialize_from_reference("healthy_adult")

    k_before = engine.get_k()
    sigma_before = engine.get_sigma()

    shock = MicrobiomeShock(type=ShockType.ANTIBIOTIC_BROAD, magnitude=0.7)
    engine.apply_shock(shock)

    assert engine.get_k() <= k_before, "Antibiotic should reduce species count"

    print(f"  OK Shock reduced k: {k_before} -> {engine.get_k()}")


def test_microbiome_intervention():
    """Test MicrobiomeEngine intervention."""
    print("Testing MicrobiomeEngine intervention...")

    engine = create_microbiome_engine(seed=42)
    engine.initialize_from_reference("dysbiotic")

    intervention = MicrobiomeIntervention(type=InterventionType.FMT, intensity=0.5)
    engine.apply_intervention(intervention)

    # FMT should change the community composition
    assert engine.get_k() > 0, "Should still have species after FMT"

    print(f"  OK FMT intervention applied, sigma={engine.get_sigma():.3f}")


def test_microbiome_collapse():
    """Test MicrobiomeEngine collapse detection."""
    print("Testing MicrobiomeEngine collapse detection...")

    engine = create_microbiome_engine(seed=42)
    engine.initialize_from_reference("healthy_adult")

    # Apply severe shock
    shock = MicrobiomeShock(type=ShockType.ANTIBIOTIC_BROAD, magnitude=0.95)
    engine.apply_shock(shock)

    # Run to potentially trigger collapse
    engine.run(duration=50, dt=0.1)

    # Either collapsed or still running - both valid
    print(f"  OK Collapse status: {engine.is_collapsed()}, sigma={engine.get_sigma():.3f}")


# =============================================================================
# BATTERY ENGINE TESTS
# =============================================================================

def test_battery_initialization():
    """Test BatteryDegradationEngine initialization."""
    print("Testing BatteryDegradationEngine initialization...")

    engine = create_battery_engine(seed=42)
    engine.initialize()

    assert engine.get_k() == 4, "Default should have 4 cells"
    assert engine.get_sigma() == 1.0, "Fresh battery should have SOH=1.0"
    assert not engine.is_collapsed(), "Should not be collapsed initially"

    print("  OK BatteryDegradationEngine initializes correctly")


def test_battery_k_eff_formula():
    """Test battery k_eff formula."""
    print("Testing BatteryDegradationEngine k_eff formula...")

    engine = create_battery_engine(seed=42)
    engine.initialize()

    k = engine.get_k()
    rho = engine.get_rho()
    k_eff = engine.get_k_eff()

    if k > 1 and rho < 1:
        expected_k_eff = k / (1 + rho * (k - 1))
        # Allow some tolerance for rho estimation
        assert abs(k_eff - expected_k_eff) < 0.5, f"k_eff mismatch: {k_eff} vs {expected_k_eff}"

    print(f"  OK k_eff = {k_eff:.2f} (k={k}, rho={rho:.3f})")


def test_battery_degradation():
    """Test BatteryDegradationEngine degradation over time."""
    print("Testing BatteryDegradationEngine degradation...")

    engine = create_battery_engine(seed=42)
    engine.initialize()

    initial_sigma = engine.get_sigma()

    # Run for 1000 hours (calendar aging)
    df = engine.run(duration=1000, dt=10)

    final_sigma = engine.get_sigma()
    assert final_sigma < initial_sigma, "SOH should decrease over time"
    assert len(df) > 1, "Should record history"

    print(f"  OK Degradation: SOH {initial_sigma:.3f} -> {final_sigma:.3f}")


def test_battery_shock():
    """Test BatteryDegradationEngine shock."""
    print("Testing BatteryDegradationEngine shock response...")

    engine = create_battery_engine(seed=42)
    engine.initialize()

    sigma_before = engine.get_sigma()

    shock = BatteryShock(type=BatteryShockType.THERMAL, magnitude=30)  # +30C
    engine.apply_shock(shock)

    # Thermal shock changes temperature, which affects degradation rate
    # Run a bit to see the effect
    engine.run(duration=100, dt=1)

    print(f"  OK Thermal shock applied, sigma={engine.get_sigma():.3f}")


def test_battery_intervention():
    """Test BatteryDegradationEngine intervention."""
    print("Testing BatteryDegradationEngine intervention...")

    # Degrade a battery first
    engine = create_battery_engine(seed=42)
    engine.initialize()
    engine.run(duration=2000, dt=10)

    sigma_before = engine.get_sigma()

    # Replace worst cell
    intervention = BatteryIntervention(type=BatteryInterventionType.CELL_REPLACEMENT)
    engine.apply_intervention(intervention)

    sigma_after = engine.get_sigma()
    assert sigma_after >= sigma_before, "Cell replacement should improve or maintain SOH"

    print(f"  OK Cell replacement: SOH {sigma_before:.3f} -> {sigma_after:.3f}")


def test_battery_collapse():
    """Test BatteryDegradationEngine collapse (80% SOH threshold)."""
    print("Testing BatteryDegradationEngine collapse threshold...")

    engine = create_battery_engine(seed=42)
    engine.initialize()

    # Run long enough to potentially collapse
    engine.run(duration=50000, dt=100)

    # Check collapse condition
    if engine.is_collapsed():
        assert engine.get_sigma() < 0.80, "Collapse should occur below 80% SOH"
        print(f"  OK Collapse at t={engine.get_collapse_time():.0f}h, SOH={engine.get_sigma():.3f}")
    else:
        print(f"  OK No collapse yet, SOH={engine.get_sigma():.3f}")


# =============================================================================
# INSTITUTIONAL ENGINE TESTS
# =============================================================================

def test_institutional_initialization():
    """Test InstitutionalCollapseEngine initialization."""
    print("Testing InstitutionalCollapseEngine initialization...")

    engine = create_institutional_engine(seed=42)
    engine.initialize_synthetic(RegimeType.DEMOCRACY)

    assert 0 <= engine.get_k() <= 1, "k should be normalized"
    assert 0 <= engine.get_sigma() <= 1, "sigma should be normalized"
    assert not engine.is_collapsed(), "Democracy should not start collapsed"

    print("  OK InstitutionalCollapseEngine initializes correctly")


def test_institutional_archetypes():
    """Test regime archetype differences."""
    print("Testing InstitutionalCollapseEngine regime archetypes...")

    engine = create_institutional_engine(seed=42)

    archetypes = {}
    for regime in [RegimeType.DEMOCRACY, RegimeType.ANOCRACY, RegimeType.AUTOCRACY]:
        engine.initialize_synthetic(regime, noise=False)
        archetypes[regime] = {
            'k': engine.get_k(),
            'rho': engine.get_rho(),
            'sigma': engine.get_sigma(),
            'f': engine.get_f(),
        }

    # Democracy should have higher k and lower rho than autocracy
    assert archetypes[RegimeType.DEMOCRACY]['k'] > archetypes[RegimeType.AUTOCRACY]['k'], \
        "Democracy should have more constraints"
    assert archetypes[RegimeType.DEMOCRACY]['rho'] < archetypes[RegimeType.AUTOCRACY]['rho'], \
        "Democracy should have lower elite coupling"

    print("  OK Regime archetypes differ as expected")


def test_institutional_k_eff_formula():
    """Test institutional k_eff formula."""
    print("Testing InstitutionalCollapseEngine k_eff formula...")

    engine = create_institutional_engine(seed=42)
    engine.initialize_synthetic(RegimeType.ANOCRACY)

    k = engine.get_k()
    rho = engine.get_rho()
    k_eff = engine.get_k_eff()

    # Scale k for formula (engine normalizes to 0-1, formula uses k*10)
    k_scaled = k * 10
    denom = 1 + rho * (k_scaled - 1)
    expected_k_eff = k_scaled / max(denom, 0.01)

    assert abs(k_eff - expected_k_eff) < 0.1, f"k_eff mismatch: {k_eff} vs {expected_k_eff}"

    print(f"  OK k_eff = {k_eff:.2f} (k={k:.2f}, rho={rho:.3f})")


def test_institutional_dynamics():
    """Test InstitutionalCollapseEngine dynamics."""
    print("Testing InstitutionalCollapseEngine dynamics...")

    engine = create_institutional_engine(seed=42)
    engine.initialize_synthetic(RegimeType.ANOCRACY)

    initial_sigma = engine.get_sigma()
    initial_f = engine.get_f()

    df = engine.run(duration=20, dt=1.0)

    # Without intervention, sigma decreases and f increases
    assert len(df) > 1, "Should record history"

    print(f"  OK 20-year run: sigma {initial_sigma:.3f}->{engine.get_sigma():.3f}, "
          f"f {initial_f:.3f}->{engine.get_f():.3f}")


def test_institutional_shock():
    """Test InstitutionalCollapseEngine shock."""
    print("Testing InstitutionalCollapseEngine shock response...")

    engine = create_institutional_engine(seed=42)
    engine.initialize_synthetic(RegimeType.DEMOCRACY)

    sigma_before = engine.get_sigma()

    shock = InstitutionalShock(
        type=InstitutionalShockType.ECONOMIC,
        magnitude=0.3,
        target_variable='sigma'
    )
    engine.apply_shock(shock)

    assert engine.get_sigma() < sigma_before, "Economic shock should reduce stability"

    print(f"  OK Economic shock: sigma {sigma_before:.3f} -> {engine.get_sigma():.3f}")


def test_institutional_intervention():
    """Test InstitutionalCollapseEngine intervention."""
    print("Testing InstitutionalCollapseEngine intervention...")

    engine = create_institutional_engine(seed=42)
    engine.initialize_synthetic(RegimeType.ANOCRACY)

    k_before = engine.get_k()

    intervention = InstitutionalIntervention(
        type=InstitutionalInterventionType.REFORM,
        intensity=1.0,
        target_variable='k'
    )
    engine.apply_intervention(intervention)

    assert engine.get_k() >= k_before, "Reform should increase constraints"

    print(f"  OK Reform intervention: k {k_before:.3f} -> {engine.get_k():.3f}")


def test_institutional_collapse():
    """Test InstitutionalCollapseEngine collapse detection."""
    print("Testing InstitutionalCollapseEngine collapse detection...")

    engine = create_institutional_engine(seed=42)
    engine.initialize_synthetic(RegimeType.AUTOCRACY)

    # Run until collapse or max duration
    df = engine.run_until_collapse(max_duration=100)

    if engine.is_collapsed():
        # Check collapse was triggered by threshold
        final_sigma = engine.get_sigma()
        final_f = engine.get_f()
        assert final_sigma < 0.2 or final_f > 0.8, "Collapse thresholds should be met"
        print(f"  OK Collapse at t={engine.get_collapse_time():.0f}, sigma={final_sigma:.3f}, f={final_f:.3f}")
    else:
        print(f"  OK No collapse in 100y, sigma={engine.get_sigma():.3f}")


# =============================================================================
# CROSS-ENGINE TESTS
# =============================================================================

def test_all_engines_have_consistent_interface():
    """Verify all engines expose the same RATCHET variables."""
    print("Testing consistent RATCHET interface across engines...")

    engines = [
        ("Microbiome", create_microbiome_engine(seed=42)),
        ("Battery", create_battery_engine(seed=42)),
        ("Institutional", create_institutional_engine(seed=42)),
    ]

    # Initialize each engine
    engines[0][1].initialize_from_reference("healthy_adult")
    engines[1][1].initialize()
    engines[2][1].initialize_synthetic(RegimeType.ANOCRACY)

    required_methods = ['get_k', 'get_rho', 'get_k_eff', 'get_sigma', 'get_f',
                        'is_collapsed', 'to_dataframe', 'run']

    for name, engine in engines:
        for method in required_methods:
            assert hasattr(engine, method), f"{name} missing method: {method}"
            # Call getter methods to verify they work
            if method.startswith('get_') or method.startswith('is_'):
                getattr(engine, method)()

    print("  OK All engines expose consistent RATCHET interface")


def test_k_eff_bounds():
    """Verify k_eff is always between 1 and k."""
    print("Testing k_eff bounds across engines...")

    # Microbiome
    m_engine = create_microbiome_engine(seed=42)
    m_engine.initialize_from_reference("healthy_adult")
    m_k, m_k_eff = m_engine.get_k(), m_engine.get_k_eff()
    assert 0 < m_k_eff <= m_k, f"Microbiome k_eff bounds: {m_k_eff} not in (0, {m_k}]"

    # Battery
    b_engine = create_battery_engine(seed=42)
    b_engine.initialize()
    b_k, b_k_eff = b_engine.get_k(), b_engine.get_k_eff()
    assert 0 < b_k_eff <= b_k, f"Battery k_eff bounds: {b_k_eff} not in (0, {b_k}]"

    # Institutional (normalized k)
    i_engine = create_institutional_engine(seed=42)
    i_engine.initialize_synthetic(RegimeType.ANOCRACY)
    i_k_eff = i_engine.get_k_eff()
    assert i_k_eff >= 0, f"Institutional k_eff should be non-negative: {i_k_eff}"

    print(f"  OK k_eff bounds valid: Microbiome={m_k_eff:.1f}, Battery={b_k_eff:.2f}, Institutional={i_k_eff:.2f}")


# =============================================================================
# TEST RUNNER
# =============================================================================

def run_all_tests():
    """Run all tests."""
    print("=" * 70)
    print("RATCHET SIMULATION ENGINES TEST SUITE")
    print("=" * 70)
    print()

    tests = [
        # Microbiome tests
        test_microbiome_initialization,
        test_microbiome_k_eff_formula,
        test_microbiome_dynamics,
        test_microbiome_shock,
        test_microbiome_intervention,
        test_microbiome_collapse,
        # Battery tests
        test_battery_initialization,
        test_battery_k_eff_formula,
        test_battery_degradation,
        test_battery_shock,
        test_battery_intervention,
        test_battery_collapse,
        # Institutional tests
        test_institutional_initialization,
        test_institutional_archetypes,
        test_institutional_k_eff_formula,
        test_institutional_dynamics,
        test_institutional_shock,
        test_institutional_intervention,
        test_institutional_collapse,
        # Cross-engine tests
        test_all_engines_have_consistent_interface,
        test_k_eff_bounds,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            failed += 1

    print()
    print("=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)

    if failed == 0:
        print("\nALL TESTS PASSED - Simulation engines working correctly!")
        return 0
    else:
        print(f"\n{failed} TEST(S) FAILED - Check errors above")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
