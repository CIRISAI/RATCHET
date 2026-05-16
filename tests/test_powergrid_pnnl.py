"""
Test PMUGridEngine against PNNL-style PMU event traces.

Mirrors the shape of tests/test_battery_nasa_comparison.py and
tests/test_ecological_biotime.py:
  - `compare_single_event` calibrates the engine to one event's
    (k, ρ, σ) triple, runs the swing-equation forward, and reports
    frequency-trajectory RMSE / final-σ error / num_pmus.
  - `run_full_comparison` aggregates per-event results across the
    dataset.

If a vendored PNNL parquet is present at data/powergrid/pnnl_events.parquet
(or pnnl_events_sample.parquet), that drives the comparison. Otherwise
the synthetic generator (parameterised on PNNL-30492 distributions)
provides the ground-truth events. In the synthetic-vs-engine case the
comparison is still meaningful: the engine and generator share the same
swing dynamics, so a tight per-event fit is the expected behaviour and is
the v0.9 P1 deliverable.

Usage:
    python3 tests/test_powergrid_pnnl.py
    python3 tests/test_powergrid_pnnl.py --event synth_0000_lin_WECC
    python3 tests/test_powergrid_pnnl.py --n-events 50 --quiet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from ratchet.engines.powergrid import (  # noqa: E402
    PMUGridEngine,
    PMUGridParams,
    create_pmu_grid_engine,
    NOMINAL_FREQUENCY_HZ,
)
from ratchet.data.powergrid_loader import (  # noqa: E402
    PNNLPMUDataset,
    PMUEvent,
    load_pnnl_pmu_events,
    prepare_for_engine,
    compute_settling_times,
    compute_settling_sigma,
    DEFAULT_SAMPLE_RATE_HZ,
)


# ─────────────────────────────────────────────────────────────────────────
# Metric helpers (same shape as battery / biotime tests)
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


def normalise_frequency(f: np.ndarray, nominal: float = NOMINAL_FREQUENCY_HZ) -> np.ndarray:
    """Express frequency as deviation from nominal (Hz)."""
    return f - nominal


# ─────────────────────────────────────────────────────────────────────────
# Engine calibration to a single observed event
# ─────────────────────────────────────────────────────────────────────────


def _fit_engine_to_event(
    event: PMUEvent,
    seed: int = 42,
) -> PMUGridEngine:
    """Configure engine parameters from the event's observed traits.

    Calibration matches the engine's free parameters to the event so that
    the simulated trajectory is in the same regime as the observed one:
      - n_pmus           : exact match to k
      - duration_s / sample_rate_hz : match the trace
      - pre_event_s      : match event_time_idx / sample_rate
      - base_coupling    : derived from observed rho (higher rho → higher coupling)
      - disturbance_magnitude_hz : peak |freq - nominal| in post-event window
      - disturbance_sign : sign of mean post-event deviation
      - pre_event_noise_sd_hz : std of pre-event PMU frequencies (median)
    """
    n_pmus = int(max(2, event.k))
    sample_rate_hz = float(event.sample_rate_hz) or DEFAULT_SAMPLE_RATE_HZ
    duration_s = float(event.duration_s) or 60.0
    pre_event_s = float(event.timestamps[event.event_time_idx]
                        - event.timestamps[0]) if event.timestamps.size else 30.0

    # Derive coupling strength from observed rho.
    # Synthetic generator with base_coupling ∈ [0.1, 0.8] yields rho ∈ [0.05, 0.95].
    # Monotone calibration: cs ≈ 0.1 + 0.7 * rho.
    base_coupling = float(np.clip(0.10 + 0.70 * event.rho, 0.05, 1.0))

    # Pre-event noise: median per-PMU std over pre-event window
    if event.event_time_idx >= 2:
        pre = event.frequency_matrix[:, : event.event_time_idx]
        per_pmu_std = np.std(pre, axis=1)
        noise_sd = float(np.clip(np.median(per_pmu_std), 0.001, 0.05))
    else:
        noise_sd = 0.01

    # Disturbance magnitude: peak |dev| in post-event mean trace
    if event.event_time_idx < event.frequency_matrix.shape[1]:
        post = event.frequency_matrix[:, event.event_time_idx:]
        mean_post = np.mean(post, axis=0)
        peak_dev = float(np.max(np.abs(mean_post - NOMINAL_FREQUENCY_HZ)))
        signed_mean_dev = float(np.mean(mean_post - NOMINAL_FREQUENCY_HZ))
        sign = -1.0 if signed_mean_dev <= 0 else +1.0
    else:
        peak_dev = 0.1
        sign = -1.0
    disturbance_mag = float(np.clip(peak_dev, 0.02, 1.0))

    params = PMUGridParams(
        n_pmus=n_pmus,
        duration_s=duration_s,
        pre_event_s=pre_event_s,
        sample_rate_hz=sample_rate_hz,
        base_coupling=base_coupling,
        pre_event_noise_sd_hz=noise_sd,
        disturbance_magnitude_hz=disturbance_mag,
        disturbance_sign=sign,
        seed=seed,
    )
    engine = create_pmu_grid_engine(params=params, seed=seed)
    engine.initialize()
    return engine


def compare_single_event(
    event_id: str,
    dataset: PNNLPMUDataset,
    verbose: bool = True,
    seed: int = 42,
) -> Dict:
    """Compare PMUGridEngine to a single PNNL-style event.

    Returns dict with keys the v1.0 P1 harness consumes:
        event_id, num_pmus, num_timepoints, k,
        empirical_rho, simulated_rho, empirical_sigma, simulated_sigma,
        rmse, mae, final_sigma_error, collapsed
    """
    obs = prepare_for_engine(dataset, event_id)
    event = dataset.events[event_id]

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Comparing engine to event {event_id}")
        print(f"{'=' * 60}")
        print(f"Empirical:")
        print(f"  k (PMUs)           : {obs['k']}")
        print(f"  rho                : {obs['rho']:.4f}")
        print(f"  sigma (final)      : {obs['sigma_final']:.4f}")
        print(f"  num_timepoints     : {obs['num_timepoints']}")
        print(f"  sample_rate_hz     : {obs['sample_rate_hz']}")
        print(f"  duration_s         : {obs['duration_s']:.2f}")
        print(f"  event_type         : {obs.get('event_type', 'unknown')}")
        print(f"  region             : {obs.get('region', 'unknown')}")

    engine = _fit_engine_to_event(event, seed=seed)
    engine.run(duration_s=obs["duration_s"])

    sim_mean_freq = engine.get_mean_frequency_trajectory()
    emp_mean_freq = obs["empirical_mean_frequency"]
    emp_sigma_traj = obs["empirical_sigma_trajectory"]

    # Build simulated sigma trajectory the same way prepare_for_engine
    # does (rolling-CV of per-PMU deviations).
    sim_F = engine.get_frequency_matrix()
    sim_ts = engine.get_timestamps()
    n_sim = sim_F.shape[1]
    sim_sigma_traj = np.ones(n_sim)
    sim_event_idx = engine.get_event_time_idx()
    if sim_event_idx >= 0:
        window = max(int(2.0 * engine.params.sample_rate_hz), 8)
        for t in range(sim_event_idx, n_sim):
            lo = max(sim_event_idx, t - window + 1)
            hi = t + 1
            sub = sim_F[:, lo:hi]
            if sub.shape[1] < 2:
                sim_sigma_traj[t] = 1.0
                continue
            dev = np.mean(np.abs(sub - NOMINAL_FREQUENCY_HZ), axis=1)
            mu = float(np.mean(dev))
            if mu <= 1e-9:
                sim_sigma_traj[t] = 1.0
            else:
                sd = float(np.std(dev))
                cv = sd / mu
                sim_sigma_traj[t] = 1.0 / (1.0 + cv)

    # Frequency-trajectory RMSE (in Hz deviation from nominal)
    emp_dev = normalise_frequency(emp_mean_freq)
    sim_dev = normalise_frequency(sim_mean_freq)

    freq_rmse = calculate_rmse(emp_dev, sim_dev)
    freq_mae = calculate_mae(emp_dev, sim_dev)

    sigma_rmse = calculate_rmse(emp_sigma_traj, sim_sigma_traj)
    sigma_mae = calculate_mae(emp_sigma_traj, sim_sigma_traj)

    sim_sigma_final = float(engine.get_sigma())
    emp_sigma_final = float(obs["sigma_final"])
    final_sigma_error = abs(sim_sigma_final - emp_sigma_final)

    sim_rho = float(engine.get_rho())
    emp_rho = float(obs["rho"])

    if verbose:
        print(f"\nSimulated:")
        print(f"  k (PMUs)           : {engine.get_k()}")
        print(f"  rho                : {sim_rho:.4f}")
        print(f"  sigma (final)      : {sim_sigma_final:.4f}")
        print(f"  collapsed          : {engine.is_collapsed()}")
        print(f"\nComparison:")
        print(f"  freq RMSE (Hz)     : {freq_rmse:.4f}")
        print(f"  freq MAE  (Hz)     : {freq_mae:.4f}")
        print(f"  sigma-traj RMSE    : {sigma_rmse:.4f}")
        print(f"  sigma-traj MAE     : {sigma_mae:.4f}")
        print(f"  final sigma error  : {final_sigma_error:.4f}")
        print(f"  rho error          : {abs(sim_rho - emp_rho):.4f}")

    return {
        "event_id": event_id,
        "num_pmus": int(obs["num_pmus"]),
        "num_timepoints": int(obs["num_timepoints"]),
        "k": int(obs["k"]),
        "empirical_rho": emp_rho,
        "simulated_rho": sim_rho,
        "rho_error": abs(sim_rho - emp_rho),
        "empirical_sigma": emp_sigma_final,
        "simulated_sigma": sim_sigma_final,
        "final_sigma_error": final_sigma_error,
        "rmse": sigma_rmse,           # primary fit metric: sigma trajectory
        "mae": sigma_mae,
        "freq_rmse_hz": freq_rmse,    # secondary: mean-freq trajectory
        "freq_mae_hz": freq_mae,
        "collapsed": bool(engine.is_collapsed()),
        "event_type": obs.get("event_type"),
        "region": obs.get("region"),
    }


# ─────────────────────────────────────────────────────────────────────────
# Full-comparison driver
# ─────────────────────────────────────────────────────────────────────────


def run_full_comparison(
    n_events: int = 50,
    verbose: bool = True,
    sample_to_print: int = 4,
    seed: int = 42,
) -> Dict:
    """Run engine-vs-data comparison across many PNNL-style events.

    If a real PNNL parquet is on disk it's used; otherwise a synthetic
    PNNL-like dataset is generated (this is the v0.9 deliverable).
    """
    print("=" * 70)
    print("PNNL PMU Grid Engine vs Data Comparison")
    print("=" * 70)

    dataset = load_pnnl_pmu_events(
        n_synthetic_events=n_events,
        seed=seed,
    )
    print(f"\nLoaded {dataset.n_events} events from source={dataset.source}")
    if dataset.n_events == 0:
        print("No events — cannot run comparison.")
        return {"status": "no_data"}

    per_event: List[Dict] = []
    for i, eid in enumerate(list(dataset.events.keys())):
        if i < sample_to_print and verbose:
            comp = compare_single_event(eid, dataset, verbose=True, seed=seed + i)
        else:
            comp = compare_single_event(eid, dataset, verbose=False, seed=seed + i)
        per_event.append(comp)

    # Aggregate
    rmses = np.array([c["rmse"] for c in per_event if not np.isnan(c.get("rmse", np.nan))])
    freq_rmses = np.array([c["freq_rmse_hz"] for c in per_event
                           if not np.isnan(c.get("freq_rmse_hz", np.nan))])
    final_errs = np.array([c["final_sigma_error"] for c in per_event])
    rho_errs = np.array([c["rho_error"] for c in per_event])

    summary = {
        "source": dataset.source,
        "n_events": dataset.n_events,
        "mean_sigma_rmse": float(np.mean(rmses)) if len(rmses) else float("nan"),
        "median_sigma_rmse": float(np.median(rmses)) if len(rmses) else float("nan"),
        "mean_freq_rmse_hz": float(np.mean(freq_rmses)) if len(freq_rmses) else float("nan"),
        "median_freq_rmse_hz": float(np.median(freq_rmses)) if len(freq_rmses) else float("nan"),
        "mean_final_sigma_error": float(np.mean(final_errs)),
        "mean_rho_error": float(np.mean(rho_errs)),
        "per_event": per_event,
    }

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Events                     : {summary['n_events']}  (source: {summary['source']})")
    print(f"  Mean sigma-trajectory RMSE : {summary['mean_sigma_rmse']:.4f}")
    print(f"  Median sigma-traj   RMSE   : {summary['median_sigma_rmse']:.4f}")
    print(f"  Mean freq RMSE (Hz)        : {summary['mean_freq_rmse_hz']:.4f}")
    print(f"  Median freq RMSE (Hz)      : {summary['median_freq_rmse_hz']:.4f}")
    print(f"  Mean final-sigma error     : {summary['mean_final_sigma_error']:.4f}")
    print(f"  Mean rho error             : {summary['mean_rho_error']:.4f}")

    return summary


def test_ratchet_variable_mapping(
    dataset: PNNLPMUDataset,
    verbose: bool = True,
) -> Dict:
    """Validate (k, rho, sigma, k_eff) accessors on PMUEvent."""
    if not dataset.events:
        return {"all_valid": False, "reason": "empty dataset"}

    if verbose:
        print(f"\n{'=' * 60}")
        print("RATCHET Variable Mapping Validation (PowerGrid)")
        print(f"{'=' * 60}")

    bad = []
    for eid, e in list(dataset.events.items())[:5]:
        k = e.get_k()
        rho = e.get_rho()
        sigma = e.get_sigma()
        f = e.get_f()
        k_eff = e.get_k_eff()
        expected_k_eff = k / (1.0 + rho * (k - 1)) if k > 1 else float(k)
        valid = all([
            k >= 0,
            0.0 <= rho <= 1.0,
            0.0 <= sigma <= 1.0,
            abs(f - (1.0 - sigma)) < 1e-9,
            abs(k_eff - expected_k_eff) < 1e-6,
        ])
        if verbose:
            print(f"  {eid}: k={k:>2d}  rho={rho:.3f}  sigma={sigma:.3f}  "
                  f"k_eff={k_eff:.2f}  valid={valid}")
        if not valid:
            bad.append(eid)

    return {"all_valid": len(bad) == 0, "invalid_events": bad}


# ─────────────────────────────────────────────────────────────────────────
# CLI entry
# ─────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare PMUGridEngine to PNNL-style PMU events"
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Reduce output verbosity"
    )
    parser.add_argument(
        "--event", type=str, default=None,
        help="Compare a specific event by ID"
    )
    parser.add_argument(
        "--n-events", type=int, default=50,
        help="Number of events to sample (default 50)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="RNG seed for synthetic data + engine init"
    )
    args = parser.parse_args()

    if args.event:
        ds = load_pnnl_pmu_events(
            n_synthetic_events=max(args.n_events, 10),
            seed=args.seed,
        )
        if args.event in ds.events:
            compare_single_event(args.event, ds, verbose=not args.quiet, seed=args.seed)
        else:
            print(f"Event {args.event!r} not found.")
            print(f"Available (first 20): {list(ds.events.keys())[:20]}")
    else:
        ds = load_pnnl_pmu_events(
            n_synthetic_events=args.n_events,
            seed=args.seed,
        )
        mapping = test_ratchet_variable_mapping(ds, verbose=not args.quiet)
        print(f"\nVariable mapping: {'PASSED' if mapping['all_valid'] else 'FAILED'}")
        run_full_comparison(
            n_events=args.n_events,
            verbose=not args.quiet,
            seed=args.seed,
        )
