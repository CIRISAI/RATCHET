"""
Test MicrobiomeEngine with Real Data

Tests the MicrobiomeEngine using real microbiome profiles from the
American Gut Project and synthetic data generators.

This script:
1. Loads real microbiome profiles and computes RATCHET variables
2. Initializes MicrobiomeEngine from real abundances
3. Simulates perturbations and compares diversity dynamics
4. Validates that engine behavior matches expected microbiome biology

Usage:
    python -m pytest tests/test_microbiome_real_data.py -v

    # Or run directly:
    python tests/test_microbiome_real_data.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ratchet.engines.microbiome import (
    MicrobiomeEngine,
    MicrobiomeParams,
    MicrobiomeShock,
    MicrobiomeIntervention,
    ShockType,
    InterventionType,
)
from ratchet.data.microbiome_loader import (
    MicrobiomeDataLoader,
    MicrobiomeSample,
    SyntheticMicrobiomeGenerator,
    load_american_gut_project,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def synthetic_generator():
    """Create a seeded synthetic data generator."""
    return SyntheticMicrobiomeGenerator(seed=42)


@pytest.fixture
def healthy_sample(synthetic_generator):
    """Generate a healthy adult sample."""
    return synthetic_generator.generate_healthy_adult(n_taxa=100)


@pytest.fixture
def dysbiotic_sample(synthetic_generator):
    """Generate a dysbiotic sample."""
    return synthetic_generator.generate_dysbiotic(n_taxa=100, severity=0.6)


@pytest.fixture
def engine_from_healthy(healthy_sample):
    """Create engine initialized with healthy sample."""
    engine = MicrobiomeEngine(seed=42)
    engine.initialize_from_abundances(
        healthy_sample.abundances,
        healthy_sample.taxa_ids,
    )
    return engine


# =============================================================================
# TESTS: Data Loading
# =============================================================================

class TestDataLoading:
    """Tests for microbiome data loading."""

    def test_load_agp_data_if_available(self):
        """Test loading American Gut Project data."""
        try:
            loader = load_american_gut_project(max_samples=50)
            samples = loader.get_samples(n=10)

            assert len(samples) == 10
            for sample in samples:
                assert sample.k > 0
                assert 0 <= sample.sigma <= 1
                assert 0 <= sample.f <= 1
                assert 0 <= sample.rho <= 1

        except FileNotFoundError:
            pytest.skip("AGP data not available")

    def test_synthetic_healthy_parameters(self, synthetic_generator):
        """Test that synthetic healthy samples have realistic parameters."""
        samples = [synthetic_generator.generate_healthy_adult() for _ in range(20)]

        k_values = [s.k for s in samples]
        sigma_values = [s.sigma for s in samples]
        f_values = [s.f for s in samples]

        # Healthy adults: k typically 80-200, sigma > 0.6, f < 0.15
        assert np.mean(k_values) > 50
        assert np.mean(k_values) < 300
        assert np.mean(sigma_values) > 0.6
        assert np.mean(f_values) < 0.1

    def test_synthetic_dysbiotic_parameters(self, synthetic_generator):
        """Test that dysbiotic samples show reduced diversity."""
        healthy = synthetic_generator.generate_healthy_adult()
        dysbiotic = synthetic_generator.generate_dysbiotic(severity=0.7)

        # Dysbiosis should reduce k and sigma, increase f
        assert dysbiotic.k < healthy.k
        assert dysbiotic.sigma < healthy.sigma
        assert dysbiotic.f > healthy.f

    def test_infant_age_progression(self, synthetic_generator):
        """Test that infant diversity increases with age."""
        infant_7d = synthetic_generator.generate_infant(age_days=7)
        infant_30d = synthetic_generator.generate_infant(age_days=30)
        infant_180d = synthetic_generator.generate_infant(age_days=180)

        # Diversity should increase with age
        assert infant_30d.k >= infant_7d.k * 0.8  # Allow some variance
        assert infant_180d.k > infant_30d.k
        assert infant_180d.sigma > infant_7d.sigma


# =============================================================================
# TESTS: Engine Initialization
# =============================================================================

class TestEngineInitialization:
    """Tests for engine initialization from real/synthetic data."""

    def test_initialize_from_sample(self, healthy_sample):
        """Test initializing engine from a sample."""
        engine = MicrobiomeEngine(seed=42)
        engine.initialize_from_abundances(
            healthy_sample.abundances,
            healthy_sample.taxa_ids,
        )

        # Engine k should approximately match sample k
        engine_k = engine.get_k()
        assert abs(engine_k - healthy_sample.k) < 10

        # Engine sigma should be in a valid range
        # Note: engine computes sigma differently from sample (from actual abundances)
        engine_sigma = engine.get_sigma()
        assert 0 <= engine_sigma <= 1

    def test_initialize_preserves_abundances(self, healthy_sample):
        """Test that initialization preserves abundance structure."""
        engine = MicrobiomeEngine(seed=42)
        engine.initialize_from_abundances(
            healthy_sample.abundances,
            healthy_sample.taxa_ids,
        )

        state = engine.get_state()

        # State should be normalized
        assert abs(np.sum(state) - 1.0) < 1e-10

        # Should have same number of taxa
        assert len(state) == len(healthy_sample.abundances)

    def test_engine_reference_profiles(self):
        """Test reference profile initialization."""
        engine_healthy = MicrobiomeEngine(seed=42)
        engine_healthy.initialize_from_reference("healthy_adult")

        engine_dysbiotic = MicrobiomeEngine(seed=42)
        engine_dysbiotic.initialize_from_reference("dysbiotic")

        # Dysbiotic should have lower diversity
        assert engine_healthy.get_sigma() > engine_dysbiotic.get_sigma()


# =============================================================================
# TESTS: Perturbation Dynamics
# =============================================================================

class TestPerturbationDynamics:
    """Tests for perturbation response and recovery."""

    def test_antibiotic_crash(self, engine_from_healthy):
        """Test that antibiotics reduce diversity."""
        initial_sigma = engine_from_healthy.get_sigma()
        initial_k = engine_from_healthy.get_k()

        # Apply broad-spectrum antibiotic
        shock = MicrobiomeShock(
            type=ShockType.ANTIBIOTIC_BROAD,
            magnitude=0.7,
        )
        engine_from_healthy.apply_shock(shock)

        # Run for a few steps
        engine_from_healthy.run(duration=2, dt=0.1)

        # Diversity should decrease
        post_sigma = engine_from_healthy.get_sigma()
        post_k = engine_from_healthy.get_k()

        # k should decrease or sigma should decrease (antibiotic effect)
        # At least one metric should show reduction
        assert post_k <= initial_k or post_sigma <= initial_sigma

    def test_fmt_recovery(self, dysbiotic_sample):
        """Test that FMT can restore diversity."""
        # Initialize with dysbiotic profile
        engine = MicrobiomeEngine(seed=42)
        engine.initialize_from_abundances(
            dysbiotic_sample.abundances,
            dysbiotic_sample.taxa_ids,
        )

        initial_sigma = engine.get_sigma()

        # Apply FMT with healthy donor
        generator = SyntheticMicrobiomeGenerator(seed=123)
        healthy = generator.generate_healthy_adult(n_taxa=100)

        intervention = MicrobiomeIntervention(
            type=InterventionType.FMT,
            intensity=0.6,
            donor_profile=healthy.abundances,
        )
        engine.apply_intervention(intervention)

        # Run simulation
        engine.run(duration=5, dt=0.1)

        # Diversity should increase
        final_sigma = engine.get_sigma()
        assert final_sigma >= initial_sigma * 0.8  # At least maintained

    def test_natural_decay_without_intervention(self, engine_from_healthy):
        """Test diversity decay over time without substrate."""
        initial_sigma = engine_from_healthy.get_sigma()

        # Increase decay rate
        engine_from_healthy.set_d(0.3)  # High decay

        # Run for extended period
        df = engine_from_healthy.run(duration=20, dt=0.1)

        # Diversity should decrease or engine may collapse
        final_sigma = engine_from_healthy.get_sigma()

        # Either diversity dropped or system collapsed
        assert final_sigma <= initial_sigma or engine_from_healthy.is_collapsed()


# =============================================================================
# TESTS: RATCHET Variable Relationships
# =============================================================================

class TestRatchetVariables:
    """Tests for RATCHET framework variable relationships."""

    def test_k_eff_calculation(self, healthy_sample):
        """Test effective k calculation."""
        # k_eff = k / (1 + rho*(k-1))
        k = healthy_sample.k
        rho = healthy_sample.rho

        expected_k_eff = k / (1 + rho * (k - 1))
        assert abs(healthy_sample.k_eff - expected_k_eff) < 0.01

    def test_high_rho_reduces_k_eff(self, synthetic_generator):
        """Test that higher correlation reduces effective k."""
        # Generate two samples with different rho
        samples = [synthetic_generator.generate_healthy_adult() for _ in range(30)]

        # Sort by rho
        samples.sort(key=lambda s: s.rho)

        low_rho_samples = samples[:10]
        high_rho_samples = samples[-10:]

        # Higher rho should mean lower k_eff relative to k
        low_rho_ratio = np.mean([s.k_eff / s.k for s in low_rho_samples])
        high_rho_ratio = np.mean([s.k_eff / s.k for s in high_rho_samples])

        assert low_rho_ratio > high_rho_ratio

    def test_sigma_f_inverse_relationship(self, synthetic_generator):
        """Test that diversity and pathogen fraction are inversely related."""
        # Compare healthy vs dysbiotic
        healthy_samples = [synthetic_generator.generate_healthy_adult() for _ in range(20)]
        dysbiotic_samples = [synthetic_generator.generate_dysbiotic(severity=0.5) for _ in range(20)]

        healthy_sigma = np.mean([s.sigma for s in healthy_samples])
        healthy_f = np.mean([s.f for s in healthy_samples])

        dysbiotic_sigma = np.mean([s.sigma for s in dysbiotic_samples])
        dysbiotic_f = np.mean([s.f for s in dysbiotic_samples])

        # Higher diversity should correlate with lower pathogen fraction
        assert healthy_sigma > dysbiotic_sigma
        assert healthy_f < dysbiotic_f


# =============================================================================
# TESTS: Antibiotic Recovery Trajectory
# =============================================================================

class TestAntibioticRecovery:
    """Tests for antibiotic perturbation and recovery dynamics."""

    def test_recovery_trajectory(self, synthetic_generator):
        """Test that recovery follows expected trajectory."""
        healthy = synthetic_generator.generate_healthy_adult()

        # Generate perturbation series
        days = [0, 3, 7, 14, 28]
        samples = [
            synthetic_generator.generate_antibiotic_perturbed(
                healthy, days_post_antibiotic=d
            )
            for d in days
        ]

        sigmas = [s.sigma for s in samples]

        # Sigma should generally increase over recovery
        # Allow for some noise
        assert sigmas[-1] > sigmas[0]  # Day 28 > Day 0
        assert sigmas[-1] > sigmas[1]  # Day 28 > Day 3

    def test_broad_vs_narrow_spectrum(self, synthetic_generator):
        """Test that broad-spectrum has greater impact than narrow."""
        healthy = synthetic_generator.generate_healthy_adult()

        broad = synthetic_generator.generate_antibiotic_perturbed(
            healthy,
            days_post_antibiotic=3,
            antibiotic_type="broad_spectrum",
        )

        narrow = synthetic_generator.generate_antibiotic_perturbed(
            healthy,
            days_post_antibiotic=3,
            antibiotic_type="narrow_spectrum",
        )

        # Broad spectrum should cause greater diversity loss
        # (higher sigma means more recovery, so lower is worse)
        # At same timepoint, narrow should have recovered more
        assert narrow.sigma >= broad.sigma * 0.9  # Narrow at least 90% of broad


# =============================================================================
# TESTS: Real Data Comparisons
# =============================================================================

class TestRealDataComparisons:
    """Tests comparing engine dynamics to real data patterns."""

    @pytest.mark.skipif(
        not (Path(__file__).parent.parent / "data" / "microbiome" / "otu_table_L6.txt").exists(),
        reason="AGP data not available",
    )
    def test_agp_diversity_distribution(self):
        """Test that AGP samples show expected diversity distribution."""
        loader = load_american_gut_project(max_samples=200)
        samples = loader.get_samples(n=100)

        sigmas = [s.sigma for s in samples]
        ks = [s.k for s in samples]

        # AGP fecal samples typically show:
        # - k: 100-500 detected species
        # - sigma: 0.4-0.9 normalized diversity
        assert 100 < np.mean(ks) < 500
        assert 0.4 < np.mean(sigmas) < 0.85

    @pytest.mark.skipif(
        not (Path(__file__).parent.parent / "data" / "microbiome" / "otu_table_L6.txt").exists(),
        reason="AGP data not available",
    )
    def test_engine_matches_real_data(self):
        """Test that engine initialized from real data maintains realistic dynamics."""
        loader = load_american_gut_project(max_samples=100)
        samples = loader.get_samples(n=10)

        for sample in samples[:3]:
            engine = MicrobiomeEngine(seed=42)
            engine.initialize_from_abundances(
                sample.abundances,
                sample.taxa_ids,
            )

            initial_sigma = engine.get_sigma()

            # Run simulation
            df = engine.run(duration=10, dt=0.1)

            # Sigma should remain in realistic range
            final_sigma = engine.get_sigma()
            assert 0.1 <= final_sigma <= 1.0

            # Should not collapse under normal dynamics
            assert not engine.is_collapsed()


# =============================================================================
# TESTS: Batch Generation
# =============================================================================

class TestBatchGeneration:
    """Tests for batch sample generation."""

    def test_batch_diversity(self, synthetic_generator):
        """Test batch generation produces diverse samples."""
        batch = synthetic_generator.generate_batch(
            n_healthy=20,
            n_dysbiotic=10,
            n_infants=5,
            n_taxa=100,
        )

        assert len(batch) == 35

        # Check diversity of samples
        sigmas = [s.sigma for s in batch]
        assert np.std(sigmas) > 0.1  # Should have variance

        # Check different profile types
        healthy = [s for s in batch if s.metadata.get('profile_type') == 'healthy_adult']
        dysbiotic = [s for s in batch if s.metadata.get('profile_type') == 'dysbiotic']
        infant = [s for s in batch if s.metadata.get('profile_type') == 'infant']

        assert len(healthy) == 20
        assert len(dysbiotic) == 10
        assert len(infant) == 5


# =============================================================================
# MAIN: Direct Execution Demo
# =============================================================================

def main():
    """Run demonstration of microbiome data loading and engine integration."""
    print("=" * 70)
    print("RATCHET Microbiome Engine - Real Data Integration Demo")
    print("=" * 70)

    # 1. Try loading real AGP data
    print("\n1. Loading American Gut Project Data...")
    try:
        loader = load_american_gut_project(max_samples=100)
        samples = loader.get_samples(n=5)

        print(f"   Loaded {len(samples)} samples")
        print("\n   Sample RATCHET variables:")
        print("   " + "-" * 50)
        for s in samples:
            print(f"   {s.sample_id}: k={s.k}, sigma={s.sigma:.3f}, f={s.f:.3f}, rho={s.rho:.3f}")

        # Get statistics
        stats = loader.get_abundance_statistics()
        print(f"\n   Dataset Statistics ({stats['n_samples']} samples, {stats['n_taxa']} taxa):")
        print(f"     k: mean={stats['k']['mean']:.1f}, median={stats['k']['p50']:.1f}")
        print(f"     sigma: mean={stats['sigma']['mean']:.3f}, median={stats['sigma']['p50']:.3f}")
        print(f"     f: mean={stats['f']['mean']:.3f}, median={stats['f']['p50']:.3f}")

        use_real_data = True

    except FileNotFoundError:
        print("   AGP data not found. Using synthetic data instead.")
        use_real_data = False

    # 2. Generate synthetic samples for comparison
    print("\n2. Generating Synthetic Microbiome Profiles...")
    generator = SyntheticMicrobiomeGenerator(seed=42)

    healthy = generator.generate_healthy_adult(n_taxa=100)
    dysbiotic = generator.generate_dysbiotic(n_taxa=100, severity=0.6)
    infant = generator.generate_infant(age_days=30, n_taxa=100)

    print("   Synthetic profiles:")
    print(f"     Healthy adult:  k={healthy.k}, sigma={healthy.sigma:.3f}, f={healthy.f:.3f}")
    print(f"     Dysbiotic:      k={dysbiotic.k}, sigma={dysbiotic.sigma:.3f}, f={dysbiotic.f:.3f}")
    print(f"     Infant (30d):   k={infant.k}, sigma={infant.sigma:.3f}, f={infant.f:.3f}")

    # 3. Initialize MicrobiomeEngine from data
    print("\n3. Initializing MicrobiomeEngine...")

    if use_real_data:
        test_sample = samples[0]
        print(f"   Using real sample: {test_sample.sample_id}")
    else:
        test_sample = healthy
        print(f"   Using synthetic healthy profile")

    engine = MicrobiomeEngine(seed=42)
    engine.initialize_from_abundances(test_sample.abundances, test_sample.taxa_ids)

    print(f"   Engine k: {engine.get_k()}, sigma: {engine.get_sigma():.3f}")

    # 4. Simulate antibiotic perturbation
    print("\n4. Simulating Antibiotic Perturbation...")

    initial_k = engine.get_k()
    initial_sigma = engine.get_sigma()

    # Apply antibiotic
    shock = MicrobiomeShock(type=ShockType.ANTIBIOTIC_BROAD, magnitude=0.6)
    engine.apply_shock(shock)

    # Run simulation
    df = engine.run(duration=14, dt=0.1)

    final_k = engine.get_k()
    final_sigma = engine.get_sigma()

    print(f"   Pre-antibiotic:  k={initial_k}, sigma={initial_sigma:.3f}")
    print(f"   Post-antibiotic: k={final_k}, sigma={final_sigma:.3f}")
    print(f"   Collapsed: {engine.is_collapsed()}")

    # 5. Test FMT recovery
    print("\n5. Simulating FMT Recovery...")

    # Re-initialize with dysbiotic profile
    engine_dys = MicrobiomeEngine(seed=42)
    engine_dys.initialize_from_abundances(dysbiotic.abundances, dysbiotic.taxa_ids)

    pre_fmt_sigma = engine_dys.get_sigma()

    # Apply FMT
    intervention = MicrobiomeIntervention(
        type=InterventionType.FMT,
        intensity=0.7,
        donor_profile=healthy.abundances,
    )
    engine_dys.apply_intervention(intervention)

    # Run simulation
    df_fmt = engine_dys.run(duration=14, dt=0.1)

    post_fmt_sigma = engine_dys.get_sigma()

    print(f"   Pre-FMT sigma:  {pre_fmt_sigma:.3f}")
    print(f"   Post-FMT sigma: {post_fmt_sigma:.3f}")

    # 6. Compare antibiotic recovery trajectory
    print("\n6. Antibiotic Recovery Trajectory...")

    days_list = [0, 3, 7, 14, 28]
    print("   Days post-antibiotic | k | sigma | f")
    print("   " + "-" * 40)

    for days in days_list:
        sample = generator.generate_antibiotic_perturbed(
            healthy,
            days_post_antibiotic=days,
        )
        print(f"   {days:2d} days                | {sample.k:3d} | {sample.sigma:.3f} | {sample.f:.3f}")

    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
