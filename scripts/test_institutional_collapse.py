#!/usr/bin/env python3
"""
Test Script: Institutional Collapse Engine with Real Country Data

This script loads real country trajectories (Venezuela, Zimbabwe, Turkey, etc.)
and runs InstitutionalCollapseEngine to compare predicted vs actual collapse timing.

Usage:
    python scripts/test_institutional_collapse.py

Output:
    - Comparison table of predicted vs actual collapse events
    - Analysis of model accuracy
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ratchet.data.institutional_loader import (
    InstitutionalDataLoader,
    CountryTrajectory,
    CollapseEvent,
    CollapseType,
)
from ratchet.engines.institutional import (
    InstitutionalCollapseEngine,
    InstitutionalParams,
    InstitutionalState,
)


# Countries of interest for analysis
CASE_STUDY_COUNTRIES = [
    # Democratic backsliding cases
    'Venezuela',
    'Turkey',
    'Hungary',
    'Poland',

    # State failure / collapse cases
    'Zimbabwe',
    'Syria',
    'Libya',
    'Yemen',

    # Regime transitions
    'Tunisia',
    'Egypt',
    'Ukraine',

    # Stable democracies (control group)
    'Germany',
    'Canada',
    'Australia',
]


def load_data() -> InstitutionalDataLoader:
    """Load institutional datasets."""
    data_dir = PROJECT_ROOT / 'data' / 'institutional'
    loader = InstitutionalDataLoader(data_dir)
    loader.load_all()
    return loader


def run_engine_from_trajectory(
    trajectory: CountryTrajectory,
    params: Optional[InstitutionalParams] = None,
    duration: int = 20,
) -> Tuple[pd.DataFrame, Optional[int], Optional[int]]:
    """
    Run InstitutionalCollapseEngine starting from a country's initial state.

    Args:
        trajectory: Country trajectory with initial conditions
        params: Engine parameters (uses defaults if None)
        duration: Number of years to simulate

    Returns:
        Tuple of (simulation_df, predicted_collapse_year, actual_collapse_year)
    """
    if params is None:
        params = InstitutionalParams(
            alpha=0.01,  # Modest reform rate
            d=0.02,      # Gradual decay
            noise_sigma=0.005,
        )

    engine = InstitutionalCollapseEngine(params=params, seed=42)

    # Get initial state from first year of trajectory
    initial = trajectory.get_state_at_year(trajectory.start_year)
    if initial is None:
        raise ValueError(f"No data for {trajectory.country} at {trajectory.start_year}")

    # Initialize engine with real-world state
    engine.initialize_manual(
        k=initial['k'],
        rho=initial['rho'],
        sigma=initial['sigma'],
        f=initial['f'],
        lambda_=initial['lambda_'],
        country_code=trajectory.country_code,
        year=trajectory.start_year,
    )

    # Run simulation
    sim_df = engine.run(duration=duration)

    # Get predicted collapse year
    predicted_collapse = None
    if engine.is_collapsed():
        predicted_collapse = trajectory.start_year + int(engine.get_collapse_time())

    # Get actual collapse year (first major collapse event)
    actual_collapse = trajectory.first_collapse_year()

    return sim_df, predicted_collapse, actual_collapse


def compare_predicted_vs_actual(
    loader: InstitutionalDataLoader,
    countries: List[str],
    start_year: int = 2000,
) -> pd.DataFrame:
    """
    Compare predicted vs actual collapse for multiple countries.

    Args:
        loader: Data loader with loaded datasets
        countries: List of country names
        start_year: Year to start analysis from

    Returns:
        DataFrame with comparison results
    """
    results = []

    for country in countries:
        try:
            # Get trajectory starting from start_year
            trajectory = loader.get_country_trajectory(
                country,
                start_year=start_year,
                interpolate=True,
            )

            # Get initial state
            initial = trajectory.get_state_at_year(trajectory.start_year)
            if initial is None:
                continue

            # Run simulation
            sim_df, predicted_collapse, actual_collapse = run_engine_from_trajectory(
                trajectory,
                duration=min(20, trajectory.duration),
            )

            # Compute error if both have collapse
            error = None
            if predicted_collapse and actual_collapse:
                error = predicted_collapse - actual_collapse

            # Determine outcome
            if actual_collapse and predicted_collapse:
                if abs(error) <= 3:
                    outcome = "CORRECT (within 3 years)"
                else:
                    outcome = f"MISS (off by {error} years)"
            elif actual_collapse and not predicted_collapse:
                outcome = "FALSE NEGATIVE (missed collapse)"
            elif not actual_collapse and predicted_collapse:
                outcome = "FALSE POSITIVE (predicted false collapse)"
            else:
                outcome = "TRUE NEGATIVE (correctly predicted stability)"

            # Get collapse events
            collapse_events = trajectory.collapse_events
            event_types = [e.collapse_type.value for e in collapse_events]

            results.append({
                'country': country,
                'start_year': trajectory.start_year,
                'initial_k': initial['k'],
                'initial_rho': initial['rho'],
                'initial_sigma': initial['sigma'],
                'initial_f': initial['f'],
                'predicted_collapse': predicted_collapse,
                'actual_collapse': actual_collapse,
                'error_years': error,
                'outcome': outcome,
                'n_collapse_events': len(collapse_events),
                'collapse_types': ', '.join(set(event_types)) if event_types else 'none',
            })

        except Exception as e:
            print(f"Error processing {country}: {e}")
            continue

    return pd.DataFrame(results)


def run_sensitivity_analysis(
    trajectory: CountryTrajectory,
    param_ranges: Dict[str, Tuple[float, float, int]],
) -> pd.DataFrame:
    """
    Run sensitivity analysis on engine parameters.

    Args:
        trajectory: Country trajectory
        param_ranges: Dict of parameter name -> (min, max, n_steps)

    Returns:
        DataFrame with sensitivity results
    """
    results = []
    initial = trajectory.get_state_at_year(trajectory.start_year)

    # Default parameters
    base_params = {
        'alpha': 0.01,
        'd': 0.02,
        'noise_sigma': 0.005,
    }

    for param_name, (pmin, pmax, n_steps) in param_ranges.items():
        for value in np.linspace(pmin, pmax, n_steps):
            # Create params with varied parameter
            params_dict = base_params.copy()
            params_dict[param_name] = value
            params = InstitutionalParams(**params_dict)

            engine = InstitutionalCollapseEngine(params=params, seed=42)
            engine.initialize_manual(
                k=initial['k'],
                rho=initial['rho'],
                sigma=initial['sigma'],
                f=initial['f'],
                lambda_=initial['lambda_'],
            )

            sim_df = engine.run(duration=20)

            results.append({
                'parameter': param_name,
                'value': value,
                'collapsed': engine.is_collapsed(),
                'collapse_time': engine.get_collapse_time(),
                'final_sigma': engine.get_sigma(),
                'final_f': engine.get_f(),
            })

    return pd.DataFrame(results)


def print_comparison_table(df: pd.DataFrame) -> None:
    """Print formatted comparison table."""
    print("\n" + "=" * 100)
    print("INSTITUTIONAL COLLAPSE ENGINE: PREDICTED vs ACTUAL COMPARISON")
    print("=" * 100 + "\n")

    # Format table
    display_cols = [
        'country', 'start_year', 'initial_sigma', 'initial_f',
        'predicted_collapse', 'actual_collapse', 'error_years', 'outcome'
    ]

    table = df[display_cols].copy()
    table['initial_sigma'] = table['initial_sigma'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
    table['initial_f'] = table['initial_f'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
    table['predicted_collapse'] = table['predicted_collapse'].apply(lambda x: str(int(x)) if pd.notna(x) else "None")
    table['actual_collapse'] = table['actual_collapse'].apply(lambda x: str(int(x)) if pd.notna(x) else "None")
    table['error_years'] = table['error_years'].apply(lambda x: f"{x:+.0f}" if pd.notna(x) else "N/A")

    print(table.to_string(index=False))

    # Summary statistics
    print("\n" + "-" * 50)
    print("SUMMARY STATISTICS")
    print("-" * 50)

    total = len(df)
    correct = df['outcome'].str.contains('CORRECT').sum()
    false_neg = df['outcome'].str.contains('FALSE NEGATIVE').sum()
    false_pos = df['outcome'].str.contains('FALSE POSITIVE').sum()
    true_neg = df['outcome'].str.contains('TRUE NEGATIVE').sum()

    print(f"Total countries analyzed: {total}")
    print(f"Correct predictions (within 3 years): {correct} ({100*correct/total:.1f}%)")
    print(f"False negatives (missed collapse): {false_neg} ({100*false_neg/total:.1f}%)")
    print(f"False positives (predicted false collapse): {false_pos} ({100*false_pos/total:.1f}%)")
    print(f"True negatives (correct stability): {true_neg} ({100*true_neg/total:.1f}%)")

    # Error analysis for correct/miss cases
    errors = df['error_years'].dropna()
    if len(errors) > 0:
        print(f"\nPrediction error statistics (for cases with both predicted and actual collapse):")
        print(f"  Mean error: {errors.mean():+.1f} years")
        print(f"  Std error: {errors.std():.1f} years")
        print(f"  Range: [{errors.min():+.0f}, {errors.max():+.0f}] years")


def analyze_country_trajectory(
    loader: InstitutionalDataLoader,
    country: str,
    start_year: int = 2000,
) -> None:
    """Detailed analysis of a single country's trajectory."""
    print(f"\n{'='*60}")
    print(f"DETAILED ANALYSIS: {country.upper()}")
    print('='*60)

    trajectory = loader.get_country_trajectory(country, start_year=start_year)

    print(f"\nTrajectory: {trajectory.start_year} - {trajectory.end_year} ({trajectory.duration} years)")
    print(f"Collapse events: {len(trajectory.collapse_events)}")

    for event in trajectory.collapse_events:
        print(f"  - {event.year}: {event.collapse_type.value} (polity change: {event.polity_change})")

    # Print trajectory over time
    print("\nRATCHET Variable Trajectory:")
    print("-" * 60)
    print(f"{'Year':>6} {'k':>8} {'rho':>8} {'sigma':>8} {'f':>8} {'lambda':>8}")
    print("-" * 60)

    for i, year in enumerate(trajectory.years):
        if i % 3 == 0:  # Print every 3rd year
            print(f"{int(year):>6} {trajectory.k[i]:>8.3f} {trajectory.rho[i]:>8.3f} "
                  f"{trajectory.sigma[i]:>8.3f} {trajectory.f[i]:>8.3f} {trajectory.lambda_[i]:>8.3f}")

    # Run simulation
    print("\n" + "-" * 60)
    print("SIMULATION RESULTS")
    print("-" * 60)

    sim_df, predicted_collapse, actual_collapse = run_engine_from_trajectory(
        trajectory,
        duration=min(25, trajectory.duration),
    )

    print(f"Predicted collapse: {predicted_collapse if predicted_collapse else 'None (stable)'}")
    print(f"Actual first collapse: {actual_collapse if actual_collapse else 'None'}")

    if predicted_collapse and actual_collapse:
        error = predicted_collapse - actual_collapse
        print(f"Prediction error: {error:+d} years")


def main():
    """Main entry point."""
    print("Loading institutional datasets...")
    loader = load_data()

    summary = loader.summary()
    print(f"\nDataset Summary:")
    print(f"  Countries: {summary['n_countries']}")
    print(f"  Observations: {summary['n_observations']}")
    print(f"  Year range: {summary['year_range']}")
    print(f"  Collapse events: {summary['n_collapse_events']}")

    # Get available countries from our list
    available_countries = []
    for country in CASE_STUDY_COUNTRIES:
        try:
            loader.get_country_trajectory(country, start_year=2000)
            available_countries.append(country)
        except ValueError:
            print(f"  Warning: {country} not found in dataset")

    print(f"\nAnalyzing {len(available_countries)} countries...")

    # Run comparison
    comparison_df = compare_predicted_vs_actual(
        loader,
        available_countries,
        start_year=2000,
    )

    # Print results
    print_comparison_table(comparison_df)

    # Detailed analysis for key cases
    for country in ['Venezuela', 'Turkey', 'Zimbabwe']:
        if country in available_countries:
            analyze_country_trajectory(loader, country, start_year=2000)

    # Save results
    output_path = PROJECT_ROOT / 'data' / 'institutional' / 'collapse_comparison.csv'
    comparison_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
