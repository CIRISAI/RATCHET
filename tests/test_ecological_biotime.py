"""
Test EcologicalCommunityEngine against BioTIME-style community time series.

Mirrors the shape of tests/test_battery_nasa_comparison.py:
  - `compare_single_community` runs the engine to match one community's
     observed biomass trajectory and reports RMSE / MAE / final-sigma error.
  - `run_full_comparison` aggregates per-community results across the dataset.

If a vendored BioTIME CSV is present at data/ecological/biotime_query.csv,
that drives the comparison. Otherwise the synthetic generator (which is
parameterised on BioTIME 2.0 marginal distributions) provides the
ground-truth communities. In the synthetic-vs-engine case the comparison
is still meaningful: the engine and generator share the *same* Kish
dynamics, so a tight per-community fit is the expected behaviour and is
the v0.9 P1 deliverable.

Usage:
    python3 tests/test_ecological_biotime.py
    python3 tests/test_ecological_biotime.py --community synth_0000_ter
    python3 tests/test_ecological_biotime.py --n-communities 50 --quiet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from ratchet.engines.ecological import (
    EcologicalCommunityEngine,
    EcologicalParams,
    create_ecological_engine,
)
from ratchet.data.ecological_loader import (
    BioTIMECommunityDataset,
    EcologicalSample,
    load_biotime_data,
    prepare_for_engine,
    compute_biomass_stability,
    compute_abundance_correlation,
)


# ─────────────────────────────────────────────────────────────────────────
# Metric helpers (same shape as battery test)
# ─────────────────────────────────────────────────────────────────────────


def calculate_rmse(empirical: np.ndarray, simulated: np.ndarray) -> float:
    min_len = min(len(empirical), len(simulated))
    if min_len == 0:
        return float("nan")
    return float(np.sqrt(np.mean((empirical[:min_len] - simulated[:min_len]) ** 2)))


def calculate_mae(empirical: np.ndarray, simulated: np.ndarray) -> float:
    min_len = min(len(empirical), len(simulated))
    if min_len == 0:
        return float("nan")
    return float(np.mean(np.abs(empirical[:min_len] - simulated[:min_len])))


def normalise_biomass(b: np.ndarray) -> np.ndarray:
    """Scale biomass to its mean (so cross-community RMSE is comparable)."""
    mu = float(np.mean(b)) if len(b) > 0 else 1.0
    if mu <= 1e-10:
        return b
    return b / mu


# ─────────────────────────────────────────────────────────────────────────
# Engine calibration to a single observed community
# ─────────────────────────────────────────────────────────────────────────


def _fit_engine_to_community(
    sample: EcologicalSample,
    seed: int = 42,
) -> EcologicalCommunityEngine:
    """Configure engine parameters from the community's observed traits.

    We match:
      - n_species : exact match to k
      - carrying_capacity_mean : log-mean of observed final-year abundances
      - coupling_strength : derived from observed rho (higher rho → higher cs)
      - env_forcing_amp : derived from observed sigma (lower sigma → higher amp)
      - intrinsic_growth_mean : fixed baseline (0.4)
      - obs_noise_frac : derived from observed sigma

    This is a coarse calibration — perfect for a P1 within-substrate fit
    test since the engine and the synthetic generator share dynamics.
    """
    n_species = int(max(2, sample.k))

    # Estimate carrying capacities from observed final abundances
    if sample.species_abundances.size > 0:
        # Use mean of last 5 years to dampen end-of-series noise
        n_t = sample.species_abundances.shape[1]
        tail = sample.species_abundances[:, max(0, n_t - 5):]
        final_abund = np.mean(tail, axis=1)
        K_mean = float(np.mean(final_abund))
        K_mean = max(5.0, min(200.0, K_mean))
    else:
        K_mean = 33.0

    # Higher observed rho → higher coupling strength.
    # Empirically the synthetic generator with cs ∈ [0, 0.4] yields rho ∈ [0, 0.95].
    # A monotone calibration: cs = 0.05 + 0.30 * rho
    coupling = float(np.clip(0.05 + 0.30 * sample.rho, 0.0, 0.45))

    # Lower observed sigma → higher env forcing.
    # sigma = 1/(1+CV); CV → ∞ as sigma → 0. We invert that to env amp ∈ [0.05, 0.35]
    env_amp = float(np.clip(0.40 * (1.0 - sample.sigma), 0.05, 0.35))

    # Noise is small contribution to CV
    noise_frac = float(np.clip(0.02 + 0.05 * (1.0 - sample.sigma), 0.02, 0.10))

    params = EcologicalParams(
        n_species=n_species,
        carrying_capacity_mean=K_mean,
        carrying_capacity_std=0.5,
        coupling_strength=coupling,
        env_forcing_amp=env_amp,
        obs_noise_frac=noise_frac,
        seed=seed,
    )
    engine = create_ecological_engine(params=params, seed=seed)
    engine.initialize()
    return engine


def compare_single_community(
    community_id: str,
    dataset: BioTIMECommunityDataset,
    verbose: bool = True,
    seed: int = 42,
) -> Dict:
    """Compare EcologicalCommunityEngine to a single BioTIME community.

    Returns dict with the keys the v0.9 P1 harness consumes:
        community_id, num_years, k,
        empirical_rho, simulated_rho, empirical_sigma, simulated_sigma,
        rmse, mae, final_sigma_error, collapsed
    """
    obs = prepare_for_engine(dataset, community_id)

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Comparing engine to community {community_id}")
        print(f"{'=' * 60}")
        print(f"Empirical:")
        print(f"  k (species)     : {obs['k']}")
        print(f"  rho             : {obs['rho']:.4f}")
        print(f"  sigma (final)   : {obs['sigma_final']:.4f}")
        print(f"  years           : {obs['num_years']}")
        print(f"  realm           : {obs.get('realm', 'unknown')}")

    sample = dataset.communities[community_id]
    engine = _fit_engine_to_community(sample, seed=seed)

    # Run engine for matching duration
    df = engine.run(duration=float(obs["num_years"]), dt=1.0)

    sim_biomass = engine.get_biomass_trajectory()
    sim_sigma_traj = np.zeros_like(sim_biomass, dtype=float)
    # Compute rolling-sigma the same way prepare_for_engine does
    window = max(3, len(sim_biomass) // 5)
    for t in range(len(sim_biomass)):
        lo = max(0, t - window + 1)
        hi = t + 1
        if (hi - lo) >= 2:
            sim_sigma_traj[t] = compute_biomass_stability(sim_biomass[lo:hi])
        else:
            sim_sigma_traj[t] = 1.0

    emp_biomass = obs["empirical_biomass"]
    emp_sigma_traj = obs["empirical_sigma_trajectory"]

    # Normalise biomass to mean (raw biomass scales differ wildly across communities)
    emp_b_norm = normalise_biomass(emp_biomass)
    sim_b_norm = normalise_biomass(sim_biomass)

    biomass_rmse = calculate_rmse(emp_b_norm, sim_b_norm)
    biomass_mae = calculate_mae(emp_b_norm, sim_b_norm)

    sigma_rmse = calculate_rmse(emp_sigma_traj, sim_sigma_traj)
    sigma_mae = calculate_mae(emp_sigma_traj, sim_sigma_traj)

    sim_sigma_final = float(engine.get_sigma())
    emp_sigma_final = float(obs["sigma_final"])
    final_sigma_error = abs(sim_sigma_final - emp_sigma_final)

    sim_rho = float(engine.get_rho())
    emp_rho = float(obs["rho"])

    if verbose:
        print(f"\nSimulated:")
        print(f"  k (species)     : {engine.get_k()}")
        print(f"  rho             : {sim_rho:.4f}")
        print(f"  sigma (final)   : {sim_sigma_final:.4f}")
        print(f"  collapsed       : {engine.is_collapsed()}")
        print(f"\nComparison:")
        print(f"  biomass RMSE (norm)  : {biomass_rmse:.4f}")
        print(f"  biomass MAE  (norm)  : {biomass_mae:.4f}")
        print(f"  sigma-traj RMSE      : {sigma_rmse:.4f}")
        print(f"  sigma-traj MAE       : {sigma_mae:.4f}")
        print(f"  final sigma error    : {final_sigma_error:.4f}")
        print(f"  rho error            : {abs(sim_rho - emp_rho):.4f}")

    return {
        "community_id": community_id,
        "num_years": int(obs["num_years"]),
        "k": int(obs["k"]),
        "empirical_rho": emp_rho,
        "simulated_rho": sim_rho,
        "rho_error": abs(sim_rho - emp_rho),
        "empirical_sigma": emp_sigma_final,
        "simulated_sigma": sim_sigma_final,
        "final_sigma_error": final_sigma_error,
        "rmse": sigma_rmse,
        "mae": sigma_mae,
        "biomass_rmse": biomass_rmse,
        "biomass_mae": biomass_mae,
        "collapsed": bool(engine.is_collapsed()),
    }


# ─────────────────────────────────────────────────────────────────────────
# Full-comparison driver
# ─────────────────────────────────────────────────────────────────────────


def run_full_comparison(
    n_communities: int = 50,
    verbose: bool = True,
    sample_to_print: int = 4,
    seed: int = 42,
) -> Dict:
    """Run engine-vs-data comparison across many BioTIME communities.

    If real BioTIME data is on disk it's used; otherwise a synthetic
    BioTIME-like dataset is generated (this is the v0.9 deliverable).
    """
    print("=" * 70)
    print("BioTIME / Ecology Engine vs Data Comparison")
    print("=" * 70)

    dataset = load_biotime_data(
        n_synthetic_communities=n_communities,
        seed=seed,
    )
    print(f"\nLoaded {dataset.n_communities} communities from source={dataset.source}")
    if dataset.n_communities == 0:
        print("No communities — cannot run comparison.")
        return {"status": "no_data"}

    per_community: List[Dict] = []
    for i, cid in enumerate(list(dataset.communities.keys())):
        if i < sample_to_print and verbose:
            comp = compare_single_community(cid, dataset, verbose=True, seed=seed + i)
        else:
            comp = compare_single_community(cid, dataset, verbose=False, seed=seed + i)
        per_community.append(comp)

    # Aggregate
    rmses = np.array([c["rmse"] for c in per_community if not np.isnan(c.get("rmse", np.nan))])
    biomass_rmses = np.array([c["biomass_rmse"] for c in per_community
                              if not np.isnan(c.get("biomass_rmse", np.nan))])
    final_errs = np.array([c["final_sigma_error"] for c in per_community])
    rho_errs = np.array([c["rho_error"] for c in per_community])

    summary = {
        "source": dataset.source,
        "n_communities": dataset.n_communities,
        "mean_sigma_rmse": float(np.mean(rmses)) if len(rmses) else float("nan"),
        "median_sigma_rmse": float(np.median(rmses)) if len(rmses) else float("nan"),
        "mean_biomass_rmse": float(np.mean(biomass_rmses)) if len(biomass_rmses) else float("nan"),
        "mean_final_sigma_error": float(np.mean(final_errs)),
        "mean_rho_error": float(np.mean(rho_errs)),
        "per_community": per_community,
    }

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Communities                : {summary['n_communities']}  (source: {summary['source']})")
    print(f"  Mean sigma-trajectory RMSE : {summary['mean_sigma_rmse']:.4f}")
    print(f"  Median sigma-traj   RMSE   : {summary['median_sigma_rmse']:.4f}")
    print(f"  Mean biomass RMSE (norm)   : {summary['mean_biomass_rmse']:.4f}")
    print(f"  Mean final-sigma error     : {summary['mean_final_sigma_error']:.4f}")
    print(f"  Mean rho error             : {summary['mean_rho_error']:.4f}")

    return summary


def test_ratchet_variable_mapping(
    dataset: BioTIMECommunityDataset,
    verbose: bool = True,
) -> Dict:
    """Validate (k, rho, sigma, k_eff) accessors on EcologicalSample."""
    if not dataset.communities:
        return {"all_valid": False, "reason": "empty dataset"}

    if verbose:
        print(f"\n{'=' * 60}")
        print("RATCHET Variable Mapping Validation (Ecological)")
        print(f"{'=' * 60}")

    bad = []
    for cid, c in list(dataset.communities.items())[:5]:
        k = c.get_k()
        rho = c.get_rho()
        sigma = c.get_sigma()
        f = c.get_f()
        k_eff = c.get_k_eff()
        expected_k_eff = k / (1.0 + rho * (k - 1)) if k > 1 else float(k)
        valid = all([
            k >= 0,
            0.0 <= rho <= 1.0,
            0.0 <= sigma <= 1.0,
            abs(f - (1.0 - sigma)) < 1e-9,
            abs(k_eff - expected_k_eff) < 1e-6,
        ])
        if verbose:
            print(f"  {cid}: k={k:>2d}  rho={rho:.3f}  sigma={sigma:.3f}  "
                  f"k_eff={k_eff:.2f}  valid={valid}")
        if not valid:
            bad.append(cid)

    return {"all_valid": len(bad) == 0, "invalid_communities": bad}


# ─────────────────────────────────────────────────────────────────────────
# CLI entry
# ─────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare EcologicalCommunityEngine to BioTIME communities"
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Reduce output verbosity"
    )
    parser.add_argument(
        "--community", type=str, default=None,
        help="Compare a specific community by ID"
    )
    parser.add_argument(
        "--n-communities", type=int, default=50,
        help="Number of communities to sample (default 50)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="RNG seed for synthetic data + engine init"
    )
    args = parser.parse_args()

    if args.community:
        ds = load_biotime_data(
            n_synthetic_communities=max(args.n_communities, 10),
            seed=args.seed,
        )
        if args.community in ds.communities:
            compare_single_community(args.community, ds, verbose=not args.quiet, seed=args.seed)
        else:
            print(f"Community {args.community!r} not found.")
            print(f"Available (first 20): {list(ds.communities.keys())[:20]}")
    else:
        ds = load_biotime_data(
            n_synthetic_communities=args.n_communities,
            seed=args.seed,
        )
        mapping = test_ratchet_variable_mapping(ds, verbose=not args.quiet)
        print(f"\nVariable mapping: {'PASSED' if mapping['all_valid'] else 'FAILED'}")
        run_full_comparison(
            n_communities=args.n_communities,
            verbose=not args.quiet,
            seed=args.seed,
        )
