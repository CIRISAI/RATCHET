"""
Test BatteryDegradationEngine against real NASA Li-ion Battery Aging Data

This script:
1. Loads empirical battery degradation data from NASA dataset
2. Runs BatteryDegradationEngine simulations with matching parameters
3. Compares synthetic vs empirical degradation behavior
4. Validates RATCHET variable mappings (k, rho, sigma, f)

Usage:
    python tests/test_battery_nasa_comparison.py

References:
    - NASA Li-ion Battery Aging Datasets
    - RATCHET BatteryDegradationEngine
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

from ratchet.engines.battery import (
    BatteryDegradationEngine,
    BatteryParams,
    create_battery_engine,
)
from ratchet.data.battery_loader import (
    load_nasa_battery_data,
    prepare_for_engine,
    get_high_quality_cells,
    NASABatteryDataset,
)


def calculate_rmse(empirical: np.ndarray, simulated: np.ndarray) -> float:
    """Calculate Root Mean Square Error between trajectories."""
    min_len = min(len(empirical), len(simulated))
    return float(np.sqrt(np.mean((empirical[:min_len] - simulated[:min_len]) ** 2)))


def calculate_mae(empirical: np.ndarray, simulated: np.ndarray) -> float:
    """Calculate Mean Absolute Error between trajectories."""
    min_len = min(len(empirical), len(simulated))
    return float(np.mean(np.abs(empirical[:min_len] - simulated[:min_len])))


def fit_engine_to_cell(
    cell_data: Dict,
    initial_alpha: float = 0.001,
    initial_d: float = 0.0001,
) -> Tuple[BatteryDegradationEngine, pd.DataFrame]:
    """
    Configure and run BatteryDegradationEngine to match a NASA cell.

    Args:
        cell_data: Output from prepare_for_engine()
        initial_alpha: Starting SEI growth rate
        initial_d: Starting calendar aging rate

    Returns:
        Tuple of (configured engine, simulation DataFrame)
    """
    # Configure engine with cell-specific parameters
    params = BatteryParams(
        num_cells=1,
        initial_capacity=cell_data['initial_capacity'],
        seed=42,
    )

    engine = BatteryDegradationEngine(params=params)
    engine.initialize()

    # Set initial cell temperature to match NASA data
    for cell in engine.cells:
        cell.temperature = cell_data['temperature']

    # Estimate degradation rates from empirical data
    num_cycles = cell_data['num_cycles']
    total_fade = 1 - cell_data['final_soh']

    # Adjust alpha to match total fade over cycle count
    # Each cycle in NASA data is a full charge-discharge, roughly 2-4 hours
    hours_per_cycle = 3.0
    total_hours = num_cycles * hours_per_cycle

    # Run simulation matching duration
    df = engine.run(duration=total_hours, dt=1.0)

    return engine, df


def compare_single_cell(
    cell_id: str,
    dataset: NASABatteryDataset,
    verbose: bool = True,
) -> Dict:
    """
    Compare BatteryDegradationEngine to a single NASA cell.

    Returns dict with comparison metrics.
    """
    cell_data = prepare_for_engine(dataset, cell_id)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Comparing simulation to NASA cell {cell_id}")
        print(f"{'='*60}")
        print(f"Empirical data:")
        print(f"  Cycles: {cell_data['num_cycles']}")
        print(f"  Initial capacity: {cell_data['initial_capacity']:.3f} Ah")
        print(f"  Final SOH: {cell_data['final_soh']:.2%}")
        print(f"  Temperature: {cell_data['temperature']:.1f}C")

    # Run simulation
    params = BatteryParams(
        num_cells=1,
        initial_capacity=cell_data['initial_capacity'],
        seed=42,
    )
    engine = create_battery_engine(params=params, seed=42)
    engine.initialize()

    # Match temperature
    for cell in engine.cells:
        cell.temperature = cell_data['temperature']

    # Estimate appropriate simulation parameters
    num_cycles = cell_data['num_cycles']
    empirical_fade = 1 - cell_data['final_soh']

    # Calibrate alpha and d to match empirical fade rate
    # The engine uses complex degradation physics with Arrhenius temperature
    # factors and parabolic SEI growth kinetics.
    #
    # Based on empirical testing:
    # - alpha=1.0, d=0.1 produces ~20% fade over 504 hours at 24C
    # - This matches roughly 29% fade per 168 cycles
    #
    # Each NASA cycle is ~3 hours (charge + discharge + rest)
    hours_per_cycle = 3.0
    total_hours = num_cycles * hours_per_cycle

    # Calculate scaling needed to match empirical fade
    # Reference: alpha=1.0, d=0.1 gives ~20% fade in 504 hours
    # Note: engine has collapse threshold at 80% SOH, so max fade is 20%
    reference_fade = 0.20
    reference_hours = 504.0

    # Scale rates to match actual empirical fade (capped at 20% due to collapse)
    effective_fade = min(empirical_fade, 0.20)
    scale_factor = (effective_fade / reference_fade) * (reference_hours / total_hours)

    # Set engine parameters with calibrated values
    # Use 90/10 split between SEI (alpha) and calendar (d) aging
    engine.set_alpha(1.0 * scale_factor * 0.9)
    engine.set_d(0.1 * scale_factor * 1.0)

    # Run for enough steps to match cycle count
    total_hours = num_cycles * hours_per_cycle

    df = engine.run(duration=total_hours, dt=1.0)

    # Extract simulated SOH trajectory at cycle-equivalent points
    simulated_soh = df['sigma'].values
    simulated_cycles = np.arange(len(simulated_soh))

    # Interpolate to match empirical cycle points
    empirical_cycles = cell_data['cycles']
    empirical_soh = cell_data['empirical_soh']

    # Sample simulated SOH at cycle-equivalent times
    cycle_times = (empirical_cycles * hours_per_cycle).astype(int)
    cycle_times = np.clip(cycle_times, 0, len(simulated_soh) - 1)
    simulated_soh_at_cycles = simulated_soh[cycle_times]

    # Calculate comparison metrics
    rmse = calculate_rmse(empirical_soh, simulated_soh_at_cycles)
    mae = calculate_mae(empirical_soh, simulated_soh_at_cycles)

    # Final state comparison
    sim_final_soh = engine.get_sigma()
    emp_final_soh = cell_data['final_soh']
    final_soh_error = abs(sim_final_soh - emp_final_soh)

    if verbose:
        print(f"\nSimulated data:")
        print(f"  Total hours: {total_hours:.0f}")
        print(f"  Steps: {len(df)}")
        print(f"  Final SOH: {sim_final_soh:.2%}")
        print(f"\nComparison metrics:")
        print(f"  SOH RMSE: {rmse:.4f}")
        print(f"  SOH MAE: {mae:.4f}")
        print(f"  Final SOH error: {final_soh_error:.4f}")
        print(f"  Engine collapsed: {engine.is_collapsed()}")

    return {
        'cell_id': cell_id,
        'num_cycles': num_cycles,
        'empirical_initial_cap': cell_data['initial_capacity'],
        'empirical_final_soh': emp_final_soh,
        'simulated_final_soh': sim_final_soh,
        'final_soh_error': final_soh_error,
        'rmse': rmse,
        'mae': mae,
        'collapsed': engine.is_collapsed(),
    }


def compare_multi_cell_pack(
    dataset: NASABatteryDataset,
    cell_ids: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict:
    """
    Compare BatteryDegradationEngine pack simulation to multiple NASA cells.

    This tests the RATCHET multi-cell correlation (rho) behavior by
    treating multiple NASA cells as a simulated battery pack.
    """
    if cell_ids is None:
        cell_ids = list(dataset.cells.keys())[:4]  # Use first 4 cells

    if verbose:
        print(f"\n{'='*60}")
        print(f"Multi-cell pack comparison ({len(cell_ids)} cells)")
        print(f"{'='*60}")

    # Get empirical data
    empirical_rho = dataset.get_rho()
    empirical_sigma = dataset.get_sigma()
    empirical_k_eff = dataset.get_k_eff()

    if verbose:
        print(f"Empirical (NASA) pack metrics:")
        print(f"  k (cells): {len(cell_ids)}")
        print(f"  rho (correlation): {empirical_rho:.4f}")
        print(f"  sigma (avg SOH): {empirical_sigma:.2%}")
        print(f"  k_eff: {empirical_k_eff:.2f}")

    # Configure engine as multi-cell pack
    avg_capacity = np.mean([dataset.cells[cid].initial_capacity for cid in cell_ids])
    avg_fade_rate = np.mean([dataset.cells[cid].fade_rate for cid in cell_ids])

    params = BatteryParams(
        num_cells=len(cell_ids),
        initial_capacity=avg_capacity,
        seed=42,
    )

    engine = create_battery_engine(params=params, seed=42)
    engine.initialize()

    # Calculate calibrated degradation rate based on empirical data
    avg_final_soh = np.mean([dataset.cells[cid].final_soh for cid in cell_ids])
    empirical_fade = 1 - avg_final_soh
    avg_cycles = int(np.mean([len(dataset.cells[cid].cycle_numbers) for cid in cell_ids]))
    hours_per_cycle = 3.0
    total_hours = avg_cycles * hours_per_cycle

    # Use same calibration as single-cell comparison
    reference_fade = 0.20
    reference_hours = 504.0
    effective_fade = min(empirical_fade, 0.20)
    scale_factor = (effective_fade / reference_fade) * (reference_hours / total_hours)

    engine.set_alpha(1.0 * scale_factor * 0.9)
    engine.set_d(0.1 * scale_factor * 1.0)

    # Set individual cell temperatures to introduce variation (simulates cell-to-cell differences)
    for i, cell in enumerate(engine.cells):
        if i < len(cell_ids):
            emp_cell = dataset.cells[cell_ids[i]]
            base_temp = float(np.mean(emp_cell.temperatures)) if len(emp_cell.temperatures) > 0 else 24.0
            # Add small variation to create different degradation rates
            cell.temperature = base_temp + (i - len(cell_ids)/2) * 0.5

    # Run simulation
    df = engine.run(duration=total_hours, dt=1.0)

    # Get simulated pack metrics
    sim_rho = engine.get_rho()
    sim_sigma = engine.get_sigma()
    sim_k_eff = engine.get_k_eff()

    if verbose:
        print(f"\nSimulated pack metrics:")
        print(f"  k (cells): {engine.get_k()}")
        print(f"  rho (correlation): {sim_rho:.4f}")
        print(f"  sigma (avg SOH): {sim_sigma:.2%}")
        print(f"  k_eff: {sim_k_eff:.2f}")
        print(f"\nComparison:")
        print(f"  rho error: {abs(sim_rho - empirical_rho):.4f}")
        print(f"  sigma error: {abs(sim_sigma - empirical_sigma):.4f}")
        print(f"  k_eff error: {abs(sim_k_eff - empirical_k_eff):.2f}")

    return {
        'k': len(cell_ids),
        'empirical_rho': empirical_rho,
        'simulated_rho': sim_rho,
        'rho_error': abs(sim_rho - empirical_rho),
        'empirical_sigma': empirical_sigma,
        'simulated_sigma': sim_sigma,
        'sigma_error': abs(sim_sigma - empirical_sigma),
        'empirical_k_eff': empirical_k_eff,
        'simulated_k_eff': sim_k_eff,
    }


def test_ratchet_variable_mapping(dataset: NASABatteryDataset, verbose: bool = True) -> Dict:
    """
    Validate RATCHET variable mappings against empirical data.

    Tests that:
    - k: Number of cells correctly tracked
    - rho: Cross-cell correlation computed consistently
    - sigma: SOH measurement matches capacity fade
    - f: Compromise fraction is 1 - sigma
    """
    if verbose:
        print(f"\n{'='*60}")
        print("RATCHET Variable Mapping Validation")
        print(f"{'='*60}")

    results = {}

    # Test k (constraint count)
    k = dataset.k
    if verbose:
        print(f"\nk (constraint count):")
        print(f"  Dataset cells: {k}")
        print(f"  Expected: Number of batteries in dataset")

    results['k'] = k
    results['k_valid'] = k > 0

    # Test rho (correlation)
    rho = dataset.get_rho()
    final_sohs = [c.final_soh for c in dataset.cells.values()]
    cv = np.std(final_sohs) / np.mean(final_sohs) if np.mean(final_sohs) > 0 else 0

    if verbose:
        print(f"\nrho (cross-cell correlation):")
        print(f"  Computed rho: {rho:.4f}")
        print(f"  SOH coefficient of variation: {cv:.4f}")
        print(f"  Interpretation: {'High' if rho > 0.5 else 'Low'} cross-cell correlation")

    results['rho'] = rho
    results['rho_valid'] = 0 <= rho <= 1

    # Test sigma (sustainability / SOH)
    sigma = dataset.get_sigma()
    avg_soh = np.mean(final_sohs)

    if verbose:
        print(f"\nsigma (sustainability / SOH):")
        print(f"  Average SOH: {sigma:.2%}")
        print(f"  Manual calculation: {avg_soh:.2%}")
        print(f"  Match: {abs(sigma - avg_soh) < 1e-10}")

    results['sigma'] = sigma
    results['sigma_valid'] = abs(sigma - avg_soh) < 1e-10

    # Test f (compromise / fade)
    f = dataset.get_f()
    expected_f = 1 - sigma

    if verbose:
        print(f"\nf (compromise / capacity fade):")
        print(f"  Computed f: {f:.2%}")
        print(f"  Expected (1 - sigma): {expected_f:.2%}")
        print(f"  Match: {abs(f - expected_f) < 1e-10}")

    results['f'] = f
    results['f_valid'] = abs(f - expected_f) < 1e-10

    # Test k_eff (effective constraint count)
    k_eff = dataset.get_k_eff()
    expected_k_eff = k / (1 + rho * (k - 1)) if k > 1 else float(k)

    if verbose:
        print(f"\nk_eff (effective constraint count):")
        print(f"  Computed k_eff: {k_eff:.2f}")
        print(f"  Expected: {expected_k_eff:.2f}")
        print(f"  Interpretation: Effective diversity is {k_eff:.1f} out of {k} cells")

    results['k_eff'] = k_eff
    results['k_eff_valid'] = abs(k_eff - expected_k_eff) < 1e-10

    # Overall validation
    all_valid = all([
        results['k_valid'],
        results['rho_valid'],
        results['sigma_valid'],
        results['f_valid'],
        results['k_eff_valid'],
    ])

    results['all_valid'] = all_valid

    if verbose:
        print(f"\n{'='*60}")
        print(f"Validation result: {'PASSED' if all_valid else 'FAILED'}")
        print(f"{'='*60}")

    return results


def run_full_comparison(verbose: bool = True) -> Dict:
    """
    Run complete comparison suite.

    Returns dict with all comparison results.
    """
    print("="*70)
    print("NASA Battery Data vs BatteryDegradationEngine Comparison")
    print("="*70)

    # Load dataset
    print("\nLoading NASA battery dataset...")
    dataset = load_nasa_battery_data(high_quality_only=True)
    print(f"Loaded {dataset.k} cells: {sorted(dataset.cell_ids)}")

    results = {
        'dataset_info': {
            'num_cells': dataset.k,
            'cell_ids': sorted(dataset.cell_ids),
        },
        'single_cell_comparisons': [],
        'pack_comparison': None,
        'variable_mapping': None,
    }

    # Test RATCHET variable mapping
    results['variable_mapping'] = test_ratchet_variable_mapping(dataset, verbose=verbose)

    # Compare individual cells (select representative subset)
    test_cells = ['B0005', 'B0006', 'B0007', 'B0018']  # Original NASA cells
    for cell_id in test_cells:
        if cell_id in dataset.cells:
            comparison = compare_single_cell(cell_id, dataset, verbose=verbose)
            results['single_cell_comparisons'].append(comparison)

    # Multi-cell pack comparison
    pack_cells = ['B0005', 'B0006', 'B0007', 'B0018']
    results['pack_comparison'] = compare_multi_cell_pack(
        dataset, cell_ids=pack_cells, verbose=verbose
    )

    # Summary
    if verbose:
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")

        print("\nSingle-cell comparison results:")
        for comp in results['single_cell_comparisons']:
            print(f"  {comp['cell_id']}: "
                  f"SOH error={comp['final_soh_error']:.4f}, "
                  f"RMSE={comp['rmse']:.4f}")

        print(f"\nPack comparison:")
        pack = results['pack_comparison']
        print(f"  rho error: {pack['rho_error']:.4f}")
        print(f"  sigma error: {pack['sigma_error']:.4f}")

        print(f"\nVariable mapping validation: "
              f"{'PASSED' if results['variable_mapping']['all_valid'] else 'FAILED'}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare BatteryDegradationEngine to NASA battery data"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Reduce output verbosity"
    )
    parser.add_argument(
        "--cell",
        type=str,
        default=None,
        help="Compare specific cell (e.g., B0005)"
    )

    args = parser.parse_args()

    if args.cell:
        dataset = load_nasa_battery_data(high_quality_only=True)
        if args.cell in dataset.cells:
            compare_single_cell(args.cell, dataset, verbose=not args.quiet)
        else:
            print(f"Cell {args.cell} not found. Available: {sorted(dataset.cell_ids)}")
    else:
        run_full_comparison(verbose=not args.quiet)
