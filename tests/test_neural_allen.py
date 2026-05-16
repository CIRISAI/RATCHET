"""
Test NeuralPopulationEngine against Allen-Neuropixels-style sessions.

Mirrors the shape of tests/test_battery_nasa_comparison.py and
tests/test_ecological_biotime.py:
  - `compare_single_session` runs the engine to match one session's
     observed decoding-accuracy trajectory and reports RMSE / MAE /
     final-σ error / ρ error.
  - `run_full_comparison` aggregates per-session results across the dataset.

If a vendored Allen Neuropixels parquet is present at
`data/neural/allen_neuropixels_sessions.parquet`, that drives the
comparison. Otherwise the synthetic generator (parameterised on
published Allen Neuropixels session distributions) provides the
ground-truth sessions. In the synthetic-vs-engine case the comparison
is still meaningful: the engine and generator share the *same*
Poisson + common-input + tuning dynamics, so a tight per-session fit
is the expected behaviour and is the v0.9 P1 deliverable.

Usage:
    python3 tests/test_neural_allen.py
    python3 tests/test_neural_allen.py --session synth_0000_VISp
    python3 tests/test_neural_allen.py --n-sessions 20 --quiet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from ratchet.engines.neural import (
    NeuralPopulationEngine,
    NeuralParams,
    create_neural_engine,
)
from ratchet.data.neural_loader import (
    AllenNeuropixelsDataset,
    NeuralSession,
    load_allen_neuropixels_sessions,
    prepare_for_engine,
    decode_population_drifting_gratings,
    compute_pairwise_spike_correlation,
    N_DRIFTING_ORIENTATIONS,
)


# ─────────────────────────────────────────────────────────────────────────
# Metric helpers (same shape as battery / ecological tests)
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


# ─────────────────────────────────────────────────────────────────────────
# Engine calibration to a single observed session
# ─────────────────────────────────────────────────────────────────────────


def _fit_engine_to_session(
    sample: NeuralSession,
    seed: int = 42,
) -> NeuralPopulationEngine:
    """Configure engine parameters from the session's observed traits.

    We match:
      - n_neurons : exact match to k
      - n_reps_per_orientation : recover from n_trials // N_orientations
      - common_input_coupling : derived from observed rho (higher rho → higher cs)
      - tune_strength : derived from observed sigma (higher sigma → higher tune)
      - baseline_rate_mean_hz : estimated from observed mean spike rate
      - trial_duration_ms : from session bin_ms × bins-per-trial

    This is a coarse calibration — perfect for a P1 within-substrate fit
    test since the engine and the synthetic generator share dynamics.
    """
    n_neurons = int(max(2, sample.k))

    # Recover reps-per-orientation from trial count
    n_trials = sample.n_trials
    n_reps = max(1, n_trials // N_DRIFTING_ORIENTATIONS)

    # Trial duration in ms from bin edges
    edges = sample.trial_bin_edges
    if len(edges) >= 2:
        bins_per_trial = int(edges[1] - edges[0])
    else:
        bins_per_trial = 2000
    trial_duration_ms = float(bins_per_trial * sample.bin_ms)

    # Higher observed rho → higher common-input coupling.
    # Empirically the synthetic generator with cs ∈ [0, 0.8] yields rho ∈ [0.01, 0.30].
    # So a monotone calibration: cs ≈ 2.5 · rho, clipped to [0.05, 0.8].
    coupling = float(np.clip(2.5 * sample.rho, 0.05, 0.8))

    # Higher observed sigma → higher tune_strength.
    # With chance = 1/8 = 0.125, sigma ∈ [0.125, 1.0]. Map to tune ∈ [0.05, 1.5].
    # Real Allen neurons have noisier tuning than a simple cosine model, so a
    # straight linear mapping over-decodes high-σ sessions and under-decodes
    # low-σ sessions. A piecewise calibration with a softer slope below 0.5
    # and a steeper slope above tracks the real data better; empirically
    # this lands simulator σ within ~0.1 of real σ across [0.2, 0.9].
    chance = 1.0 / N_DRIFTING_ORIENTATIONS
    sigma_norm = float(np.clip((sample.sigma - chance) / (1.0 - chance), 0.0, 1.0))
    if sigma_norm <= 0.5:
        tune = 0.05 + 1.0 * sigma_norm  # 0.05..0.55 for σ_norm ∈ [0, 0.5]
    else:
        tune = 0.55 + 1.5 * (sigma_norm - 0.5)  # 0.55..1.30 for σ_norm ∈ [0.5, 1.0]
    tune = float(np.clip(tune, 0.05, 1.5))

    # Estimate baseline firing rate from observed spike totals
    if sample.spike_train_matrix.size > 0 and sample.n_time_bins > 0:
        total_spikes = float(sample.spike_train_matrix.sum())
        recording_s = (sample.n_time_bins * sample.bin_ms) / 1000.0
        mean_rate_hz = total_spikes / (max(1, sample.n_neurons) * max(recording_s, 1e-6))
        mean_rate_hz = float(np.clip(mean_rate_hz, 0.5, 30.0))
    else:
        mean_rate_hz = 5.0

    params = NeuralParams(
        n_neurons=n_neurons,
        n_orientations=N_DRIFTING_ORIENTATIONS,
        n_reps_per_orientation=n_reps,
        trial_duration_ms=trial_duration_ms,
        bin_ms=float(sample.bin_ms),
        baseline_rate_mean_hz=mean_rate_hz,
        common_input_coupling=coupling,
        tune_strength=tune,
        seed=seed,
    )
    engine = create_neural_engine(params=params, seed=seed)
    engine.initialize()
    return engine


def compare_single_session(
    session_id: str,
    dataset: AllenNeuropixelsDataset,
    verbose: bool = True,
    seed: int = 42,
) -> Dict:
    """Compare NeuralPopulationEngine to a single Allen Neuropixels session.

    Returns dict with the keys the v0.9 P1 harness consumes:
        session_id, n_trials, k,
        empirical_rho, simulated_rho, empirical_sigma, simulated_sigma,
        rmse, mae, final_sigma_error, num_neurons, collapsed
    """
    obs = prepare_for_engine(dataset, session_id)

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Comparing engine to session {session_id}")
        print(f"{'=' * 60}")
        print(f"Empirical:")
        print(f"  k (neurons)        : {obs['k']}")
        print(f"  rho                : {obs['rho']:.4f}")
        print(f"  sigma (decoding)   : {obs['sigma_final']:.4f}")
        print(f"  n_trials           : {obs['n_trials']}")
        print(f"  visual_area        : {obs.get('visual_area', 'unknown')}")

    sample = dataset.sessions[session_id]
    engine = _fit_engine_to_session(sample, seed=seed)

    # Run engine through the full stimulus block
    df = engine.run()

    # Build simulated decoding-trajectory at the same grid as empirical
    sim_spike = engine.get_spike_train_matrix()
    sim_labels = engine.get_stimulus_labels()
    sim_edges = engine.get_trial_bin_edges()

    grid = obs["trajectory_x"]
    sim_sigma_traj = np.zeros(len(grid), dtype=float)
    n_classes = N_DRIFTING_ORIENTATIONS
    for i, n_use in enumerate(grid):
        n_use = int(n_use)
        if n_use < n_classes * 2 or n_use > len(sim_labels):
            sim_sigma_traj[i] = 1.0 / n_classes
            continue
        edges_sub = sim_edges[: n_use + 1]
        bin_max = int(edges_sub[-1])
        truncated_spike = sim_spike[:, :bin_max]
        truncated_labels = sim_labels[:n_use]
        if len(np.unique(truncated_labels)) < 2:
            sim_sigma_traj[i] = 1.0 / n_classes
            continue
        sim_sigma_traj[i] = decode_population_drifting_gratings(
            truncated_spike, truncated_labels, edges_sub,
            n_folds=min(5, n_use // 4),
            seed=(seed * 31 + i),
        )

    emp_sigma_traj = obs["empirical_sigma_trajectory"]

    sigma_rmse = calculate_rmse(emp_sigma_traj, sim_sigma_traj)
    sigma_mae = calculate_mae(emp_sigma_traj, sim_sigma_traj)

    sim_sigma_final = float(engine.get_sigma())
    emp_sigma_final = float(obs["sigma_final"])
    final_sigma_error = abs(sim_sigma_final - emp_sigma_final)

    sim_rho = float(engine.get_rho())
    emp_rho = float(obs["rho"])

    if verbose:
        print(f"\nSimulated:")
        print(f"  k (neurons)        : {engine.get_k()}")
        print(f"  rho                : {sim_rho:.4f}")
        print(f"  sigma (decoding)   : {sim_sigma_final:.4f}")
        print(f"  collapsed          : {engine.is_collapsed()}")
        print(f"\nComparison:")
        print(f"  sigma-traj RMSE    : {sigma_rmse:.4f}")
        print(f"  sigma-traj MAE     : {sigma_mae:.4f}")
        print(f"  final sigma error  : {final_sigma_error:.4f}")
        print(f"  rho error          : {abs(sim_rho - emp_rho):.4f}")

    return {
        "session_id": session_id,
        "n_trials": int(obs["n_trials"]),
        "k": int(obs["k"]),
        "num_neurons": int(obs["k"]),
        "empirical_rho": emp_rho,
        "simulated_rho": sim_rho,
        "rho_error": abs(sim_rho - emp_rho),
        "empirical_sigma": emp_sigma_final,
        "simulated_sigma": sim_sigma_final,
        "final_sigma_error": final_sigma_error,
        "rmse": sigma_rmse,
        "mae": sigma_mae,
        "collapsed": bool(engine.is_collapsed()),
        "visual_area": obs.get("visual_area"),
    }


# ─────────────────────────────────────────────────────────────────────────
# Full-comparison driver
# ─────────────────────────────────────────────────────────────────────────


def run_full_comparison(
    n_sessions: int = 20,
    verbose: bool = True,
    sample_to_print: int = 3,
    seed: int = 42,
) -> Dict:
    """Run engine-vs-data comparison across many Allen Neuropixels sessions.

    If real Allen Neuropixels data is on disk it's used; otherwise a
    synthetic Allen-like dataset is generated (this is the v0.9 deliverable).
    """
    print("=" * 70)
    print("Allen Neuropixels / Neural Engine vs Data Comparison")
    print("=" * 70)

    dataset = load_allen_neuropixels_sessions(
        n_synthetic_sessions=n_sessions,
        seed=seed,
    )
    print(f"\nLoaded {dataset.n_sessions} sessions from source={dataset.source}")
    if dataset.n_sessions == 0:
        print("No sessions — cannot run comparison.")
        return {"status": "no_data"}

    per_session: List[Dict] = []
    for i, sid in enumerate(list(dataset.sessions.keys())):
        if i < sample_to_print and verbose:
            comp = compare_single_session(sid, dataset, verbose=True, seed=seed + i)
        else:
            comp = compare_single_session(sid, dataset, verbose=False, seed=seed + i)
        per_session.append(comp)

    # Aggregate
    rmses = np.array([c["rmse"] for c in per_session if not np.isnan(c.get("rmse", np.nan))])
    final_errs = np.array([c["final_sigma_error"] for c in per_session])
    rho_errs = np.array([c["rho_error"] for c in per_session])
    n_neurons_arr = np.array([c["num_neurons"] for c in per_session])

    summary = {
        "source": dataset.source,
        "n_sessions": dataset.n_sessions,
        "mean_sigma_rmse": float(np.mean(rmses)) if len(rmses) else float("nan"),
        "median_sigma_rmse": float(np.median(rmses)) if len(rmses) else float("nan"),
        "mean_final_sigma_error": float(np.mean(final_errs)),
        "mean_rho_error": float(np.mean(rho_errs)),
        "mean_num_neurons": float(np.mean(n_neurons_arr)),
        "per_session": per_session,
    }

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Sessions                  : {summary['n_sessions']}  (source: {summary['source']})")
    print(f"  Mean neurons per session  : {summary['mean_num_neurons']:.1f}")
    print(f"  Mean sigma-traj RMSE      : {summary['mean_sigma_rmse']:.4f}")
    print(f"  Median sigma-traj RMSE    : {summary['median_sigma_rmse']:.4f}")
    print(f"  Mean final-sigma error    : {summary['mean_final_sigma_error']:.4f}")
    print(f"  Mean rho error            : {summary['mean_rho_error']:.4f}")

    return summary


def test_ratchet_variable_mapping(
    dataset: AllenNeuropixelsDataset,
    verbose: bool = True,
) -> Dict:
    """Validate (k, rho, sigma, k_eff) accessors on NeuralSession."""
    if not dataset.sessions:
        return {"all_valid": False, "reason": "empty dataset"}

    if verbose:
        print(f"\n{'=' * 60}")
        print("RATCHET Variable Mapping Validation (Neural)")
        print(f"{'=' * 60}")

    bad = []
    for sid, s in list(dataset.sessions.items())[:5]:
        k = s.get_k()
        rho = s.get_rho()
        sigma = s.get_sigma()
        f = s.get_f()
        k_eff = s.get_k_eff()
        expected_k_eff = k / (1.0 + rho * (k - 1)) if k > 1 else float(k)
        valid = all([
            k >= 0,
            0.0 <= rho <= 1.0,
            0.0 <= sigma <= 1.0,
            abs(f - (1.0 - sigma)) < 1e-9,
            abs(k_eff - expected_k_eff) < 1e-6,
        ])
        if verbose:
            print(f"  {sid}: k={k:>3d}  rho={rho:.3f}  sigma={sigma:.3f}  "
                  f"k_eff={k_eff:.2f}  valid={valid}")
        if not valid:
            bad.append(sid)

    return {"all_valid": len(bad) == 0, "invalid_sessions": bad}


# ─────────────────────────────────────────────────────────────────────────
# CLI entry
# ─────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare NeuralPopulationEngine to Allen Neuropixels sessions"
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Reduce output verbosity"
    )
    parser.add_argument(
        "--session", type=str, default=None,
        help="Compare a specific session by ID"
    )
    parser.add_argument(
        "--n-sessions", type=int, default=20,
        help="Number of sessions to sample (default 20)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="RNG seed for synthetic data + engine init"
    )
    args = parser.parse_args()

    if args.session:
        ds = load_allen_neuropixels_sessions(
            n_synthetic_sessions=max(args.n_sessions, 10),
            seed=args.seed,
        )
        if args.session in ds.sessions:
            compare_single_session(args.session, ds, verbose=not args.quiet, seed=args.seed)
        else:
            print(f"Session {args.session!r} not found.")
            print(f"Available (first 20): {list(ds.sessions.keys())[:20]}")
    else:
        ds = load_allen_neuropixels_sessions(
            n_synthetic_sessions=args.n_sessions,
            seed=args.seed,
        )
        mapping = test_ratchet_variable_mapping(ds, verbose=not args.quiet)
        print(f"\nVariable mapping: {'PASSED' if mapping['all_valid'] else 'FAILED'}")
        run_full_comparison(
            n_sessions=args.n_sessions,
            verbose=not args.quiet,
            seed=args.seed,
        )
