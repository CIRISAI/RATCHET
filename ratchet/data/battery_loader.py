"""
NASA Li-ion Battery Aging Dataset Loader for RATCHET

Loads NASA battery aging data from .mat files and extracts relevant
variables for use with the BatteryDegradationEngine.

Dataset source: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/

Domain Mapping to RATCHET variables:
    k (constraints):      Number of cells being analyzed
    rho (correlation):    Cross-cell SOH correlation (computed from multiple cells)
    sigma (sustainability): State of Health (SOH) = Q_current / Q_initial
    f (compromise):       Capacity fade fraction (1 - SOH)
    d (decay rate):       Calendar aging rate (derived from SOH decline)
    alpha (generation):   Cyclic aging rate (derived from cycle-by-cycle fade)

References:
    - B. Saha and K. Goebel (2007). "Battery Data Set", NASA Ames Prognostics
      Data Repository, NASA Ames Research Center, Moffett Field, CA
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

try:
    import scipy.io as sio
except ImportError:
    raise ImportError("scipy is required for loading .mat files. Install with: pip install scipy")


@dataclass
class BatteryCycleData:
    """Data from a single battery cycle (charge, discharge, or impedance)."""
    cycle_index: int
    cycle_type: str  # 'charge', 'discharge', 'impedance'
    ambient_temperature: float
    timestamp: Optional[str] = None

    # Discharge-specific
    capacity: Optional[float] = None  # Ah

    # Impedance-specific
    re: Optional[float] = None  # Electrolyte resistance (real part)
    rct: Optional[float] = None  # Charge transfer resistance
    impedance_magnitude: Optional[float] = None

    # Time-series data (optional, for detailed analysis)
    voltage: Optional[np.ndarray] = None
    current: Optional[np.ndarray] = None
    temperature: Optional[np.ndarray] = None


@dataclass
class BatteryData:
    """Complete battery aging data for a single cell."""
    cell_id: str
    initial_capacity: float  # Ah
    nominal_voltage: float = 3.7  # V (typical Li-ion)
    chemistry: str = "Li-ion 18650"

    # Processed degradation trajectory
    cycle_numbers: np.ndarray = field(default_factory=lambda: np.array([]))
    capacities: np.ndarray = field(default_factory=lambda: np.array([]))
    soh_values: np.ndarray = field(default_factory=lambda: np.array([]))
    temperatures: np.ndarray = field(default_factory=lambda: np.array([]))

    # Impedance trajectory (if available)
    impedance_cycles: np.ndarray = field(default_factory=lambda: np.array([]))
    re_values: np.ndarray = field(default_factory=lambda: np.array([]))
    rct_values: np.ndarray = field(default_factory=lambda: np.array([]))

    # Raw cycle data
    cycles: List[BatteryCycleData] = field(default_factory=list)

    # RATCHET-compatible derived values
    @property
    def sigma(self) -> np.ndarray:
        """State of Health trajectory."""
        return self.soh_values

    @property
    def f(self) -> np.ndarray:
        """Capacity fade trajectory (1 - SOH)."""
        return 1 - self.soh_values

    @property
    def final_soh(self) -> float:
        """Final SOH value."""
        return float(self.soh_values[-1]) if len(self.soh_values) > 0 else 1.0

    @property
    def fade_rate(self) -> float:
        """Average capacity fade rate per cycle."""
        if len(self.soh_values) < 2:
            return 0.0
        return float((1 - self.soh_values[-1]) / len(self.soh_values))

    def get_soh_at_cycle(self, cycle: int) -> float:
        """Get interpolated SOH at a given cycle number."""
        if len(self.cycle_numbers) == 0:
            return 1.0
        return float(np.interp(cycle, self.cycle_numbers, self.soh_values))

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame for analysis."""
        data = {
            'cycle': self.cycle_numbers,
            'capacity': self.capacities,
            'soh': self.soh_values,
            'temperature': self.temperatures,
            'fade': self.f,
        }
        return pd.DataFrame(data)


@dataclass
class NASABatteryDataset:
    """Collection of battery cells from NASA dataset."""
    cells: Dict[str, BatteryData] = field(default_factory=dict)

    @property
    def k(self) -> int:
        """Number of cells (constraint count in RATCHET)."""
        return len(self.cells)

    @property
    def cell_ids(self) -> List[str]:
        """List of cell identifiers."""
        return list(self.cells.keys())

    def get_rho(self) -> float:
        """
        Compute cross-cell SOH correlation.

        Returns correlation coefficient between SOH trajectories of all cells.
        Higher rho means cells degrade more similarly.
        """
        if len(self.cells) < 2:
            return 0.0

        # Get final SOH values for all cells
        final_sohs = np.array([cell.final_soh for cell in self.cells.values()])

        if np.std(final_sohs) < 1e-10:
            return 1.0  # All cells have same SOH -> perfect correlation

        # Compute coefficient of variation and convert to correlation metric
        cv = np.std(final_sohs) / np.mean(final_sohs)
        rho = 1 - min(1, cv * 10)  # Same formula as BatteryDegradationEngine
        return max(0, min(1, rho))

    def get_sigma(self) -> float:
        """Get average SOH across all cells."""
        if not self.cells:
            return 1.0
        return float(np.mean([cell.final_soh for cell in self.cells.values()]))

    def get_f(self) -> float:
        """Get average capacity fade across all cells."""
        return 1 - self.get_sigma()

    def get_k_eff(self) -> float:
        """Get effective constraint count: k / (1 + rho*(k-1))."""
        k = self.k
        rho = self.get_rho()
        if k <= 1:
            return float(k)
        return k / (1 + rho * (k - 1))

    def to_ratchet_params(self) -> Dict:
        """
        Extract parameters for BatteryDegradationEngine initialization.

        Returns dict with:
            - num_cells: Number of cells (k)
            - initial_capacity: Average initial capacity
            - soh_collapse_threshold: Based on dataset end-of-life criteria
            - derived rates: alpha, d estimated from data
        """
        if not self.cells:
            return {}

        cells_list = list(self.cells.values())
        initial_caps = [c.initial_capacity for c in cells_list]
        fade_rates = [c.fade_rate for c in cells_list]

        return {
            'num_cells': len(self.cells),
            'initial_capacity': float(np.mean(initial_caps)),
            'soh_collapse_threshold': 0.80,  # NASA dataset EOL criterion
            'estimated_alpha': float(np.mean(fade_rates)) if fade_rates else 0.001,
            'rho': self.get_rho(),
            'sigma': self.get_sigma(),
            'f': self.get_f(),
            'k_eff': self.get_k_eff(),
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Combine all cells into a single DataFrame."""
        dfs = []
        for cell_id, cell in self.cells.items():
            df = cell.to_dataframe()
            df['cell_id'] = cell_id
            dfs.append(df)

        if not dfs:
            return pd.DataFrame()

        return pd.concat(dfs, ignore_index=True)


class NASABatteryLoader:
    """
    Loader for NASA Li-ion Battery Aging Dataset.

    Parses MATLAB .mat files containing battery cycling data with:
    - Charge/discharge cycles with capacity measurements
    - Impedance spectroscopy measurements (EIS)
    - Temperature monitoring

    Example:
        >>> loader = NASABatteryLoader("/path/to/battery/data")
        >>> dataset = loader.load_all()
        >>> print(f"Loaded {dataset.k} cells")
        >>> print(f"Average SOH: {dataset.get_sigma():.2%}")
        >>>
        >>> # Get RATCHET-compatible parameters
        >>> params = dataset.to_ratchet_params()
        >>> engine = BatteryDegradationEngine(BatteryParams(**params))
    """

    def __init__(self, data_dir: Union[str, Path]):
        """
        Initialize the loader.

        Args:
            data_dir: Directory containing NASA battery .mat files
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise ValueError(f"Data directory does not exist: {data_dir}")

    def list_available_cells(self) -> List[str]:
        """List all available battery cell files."""
        mat_files = list(self.data_dir.glob("*.mat"))
        return [f.stem for f in mat_files]

    def load_cell(self, cell_id: str, include_timeseries: bool = False) -> BatteryData:
        """
        Load a single battery cell's data.

        Args:
            cell_id: Cell identifier (e.g., 'B0005')
            include_timeseries: If True, include voltage/current/temperature arrays

        Returns:
            BatteryData object with processed degradation trajectory
        """
        mat_path = self.data_dir / f"{cell_id}.mat"
        if not mat_path.exists():
            raise FileNotFoundError(f"Battery file not found: {mat_path}")

        # Load MATLAB file
        mat_data = sio.loadmat(str(mat_path))

        if cell_id not in mat_data:
            raise ValueError(f"Cell {cell_id} not found in mat file. Keys: {list(mat_data.keys())}")

        battery_struct = mat_data[cell_id]
        cycle_data = battery_struct['cycle'][0, 0]

        # Process all cycles
        cycles = []
        discharge_cycles = []
        capacities = []
        temperatures = []

        impedance_cycles_list = []
        re_values = []
        rct_values = []

        for i in range(cycle_data.shape[1]):
            cycle = cycle_data[0, i]
            cycle_type = str(cycle['type'][0]).lower()
            ambient_temp = float(cycle['ambient_temperature'][0][0])

            cycle_obj = BatteryCycleData(
                cycle_index=i,
                cycle_type=cycle_type,
                ambient_temperature=ambient_temp,
            )

            if cycle['data'].size > 0:
                data_field = cycle['data'][0, 0]

                # Handle discharge cycles (capacity data)
                if 'discharge' in cycle_type:
                    if 'Capacity' in data_field.dtype.names:
                        cap = data_field['Capacity']
                        if cap.size > 0:
                            cap_values = cap.flatten()
                            final_cap = float(cap_values[-1]) if len(cap_values) > 0 else None
                            if final_cap is not None and final_cap > 0:
                                cycle_obj.capacity = final_cap
                                discharge_cycles.append(i)
                                capacities.append(final_cap)
                                temperatures.append(ambient_temp)

                    if include_timeseries:
                        if 'Voltage_measured' in data_field.dtype.names:
                            cycle_obj.voltage = data_field['Voltage_measured'].flatten()
                        if 'Current_measured' in data_field.dtype.names:
                            cycle_obj.current = data_field['Current_measured'].flatten()
                        if 'Temperature_measured' in data_field.dtype.names:
                            cycle_obj.temperature = data_field['Temperature_measured'].flatten()

                # Handle impedance cycles
                elif 'impedance' in cycle_type:
                    try:
                        if 'Re' in data_field.dtype.names:
                            re_val = data_field['Re'].flatten()
                            if len(re_val) > 0:
                                # Take real part if complex
                                re_mean = float(np.mean(np.real(re_val)))
                                cycle_obj.re = re_mean
                                impedance_cycles_list.append(i)
                                re_values.append(re_mean)

                        if 'Rct' in data_field.dtype.names:
                            rct_val = data_field['Rct'].flatten()
                            if len(rct_val) > 0:
                                rct_mean = float(np.mean(np.real(rct_val)))
                                cycle_obj.rct = rct_mean
                                rct_values.append(rct_mean)

                        if 'Battery_impedance' in data_field.dtype.names:
                            z_val = data_field['Battery_impedance'].flatten()
                            if len(z_val) > 0:
                                cycle_obj.impedance_magnitude = float(np.mean(np.abs(z_val)))
                    except Exception:
                        pass  # Skip problematic impedance data

            cycles.append(cycle_obj)

        # Calculate SOH trajectory
        # Use maximum capacity as reference (accounts for initial conditioning cycles)
        if len(capacities) > 0:
            # Take maximum of first few cycles as initial capacity
            # This handles cases where first cycle may not be at full capacity
            initial_capacity = max(capacities[:min(5, len(capacities))])

            # Filter out anomalous data (initial capacity should be reasonable for 18650)
            # Typical 18650 cells are 1.5-3.5 Ah
            if initial_capacity < 0.5 or initial_capacity > 5.0:
                # Use rated capacity for 18650 if measured value is unreasonable
                initial_capacity = 2.0

            soh_values = np.array([min(1.0, c / initial_capacity) for c in capacities])
        else:
            initial_capacity = 2.0  # Default for 18650 cells
            soh_values = np.array([1.0])

        return BatteryData(
            cell_id=cell_id,
            initial_capacity=initial_capacity,
            cycle_numbers=np.array(discharge_cycles),
            capacities=np.array(capacities),
            soh_values=soh_values,
            temperatures=np.array(temperatures),
            impedance_cycles=np.array(impedance_cycles_list),
            re_values=np.array(re_values),
            rct_values=np.array(rct_values) if rct_values else np.array([]),
            cycles=cycles if include_timeseries else [],
        )

    def load_all(
        self,
        cell_ids: Optional[List[str]] = None,
        include_timeseries: bool = False,
    ) -> NASABatteryDataset:
        """
        Load multiple battery cells.

        Args:
            cell_ids: List of cell IDs to load. If None, loads all available.
            include_timeseries: If True, include voltage/current/temperature arrays

        Returns:
            NASABatteryDataset containing all loaded cells
        """
        if cell_ids is None:
            cell_ids = self.list_available_cells()

        dataset = NASABatteryDataset()

        for cell_id in cell_ids:
            try:
                cell_data = self.load_cell(cell_id, include_timeseries)
                dataset.cells[cell_id] = cell_data
            except Exception as e:
                print(f"Warning: Could not load {cell_id}: {e}")

        return dataset

    def load_cell_groups(self) -> Dict[str, List[str]]:
        """
        Group cells by their experimental conditions.

        Based on NASA dataset documentation:
        - B0005, B0006, B0007, B0018: Room temperature (24C) aging
        - B0025-B0028: First set with additional conditions
        - B0029-B0032: Second set
        - etc.

        Returns:
            Dict mapping group name to list of cell IDs
        """
        available = set(self.list_available_cells())
        groups = {}

        # Define known experimental groups from NASA documentation
        known_groups = {
            'RW_24C': ['B0005', 'B0006', 'B0007', 'B0018'],  # Room temp, 24C
            'Set_25_28': ['B0025', 'B0026', 'B0027', 'B0028'],
            'Set_29_32': ['B0029', 'B0030', 'B0031', 'B0032'],
            'Set_33_36': ['B0033', 'B0034', 'B0036'],
            'Set_38_40': ['B0038', 'B0039', 'B0040'],
            'Set_41_44': ['B0041', 'B0042', 'B0043', 'B0044'],
            'Set_45_48': ['B0045', 'B0046', 'B0047', 'B0048'],
        }

        for group_name, cell_ids in known_groups.items():
            present = [c for c in cell_ids if c in available]
            if present:
                groups[group_name] = present

        return groups


def get_high_quality_cells() -> List[str]:
    """
    Return list of cell IDs with high-quality degradation data.

    These cells have:
    - Reasonable initial capacity (1.5-2.5 Ah typical for 18650)
    - Clear degradation trajectory
    - Sufficient number of cycles
    """
    # Based on NASA dataset documentation and data quality analysis
    # These are the cells with clean, monotonic degradation curves
    return [
        'B0005', 'B0006', 'B0007', 'B0018',  # Original FY08Q4 cells
        'B0025', 'B0026', 'B0027', 'B0028',  # Set 25-28
        'B0029', 'B0030', 'B0031', 'B0032',  # Set 29-32
        'B0042', 'B0043', 'B0044',            # Set 41-44 (B0041 has issues)
        'B0045', 'B0046', 'B0047', 'B0048',  # Set 45-48
    ]


def load_nasa_battery_data(
    data_dir: Union[str, Path] = "/home/emoore/RATCHET/data/battery/5. Battery Data Set",
    cell_ids: Optional[List[str]] = None,
    high_quality_only: bool = False,
) -> NASABatteryDataset:
    """
    Convenience function to load NASA battery dataset.

    Args:
        data_dir: Path to directory containing .mat files
        cell_ids: Optional list of specific cells to load
        high_quality_only: If True, only load cells with verified good data quality

    Returns:
        NASABatteryDataset object

    Example:
        >>> dataset = load_nasa_battery_data()
        >>> print(f"Loaded {dataset.k} cells, avg SOH: {dataset.get_sigma():.2%}")
    """
    loader = NASABatteryLoader(data_dir)

    if high_quality_only and cell_ids is None:
        cell_ids = get_high_quality_cells()

    return loader.load_all(cell_ids)


def save_processed_data(
    dataset: NASABatteryDataset,
    output_dir: Union[str, Path],
    format: str = "parquet",
) -> None:
    """
    Save processed battery data to disk.

    Args:
        dataset: NASABatteryDataset to save
        output_dir: Directory to save files
        format: 'parquet', 'csv', or 'json'
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save combined DataFrame
    df = dataset.to_dataframe()
    if format == "parquet":
        df.to_parquet(output_dir / "nasa_battery_data.parquet", index=False)
    elif format == "csv":
        df.to_csv(output_dir / "nasa_battery_data.csv", index=False)

    # Save RATCHET parameters
    params = dataset.to_ratchet_params()
    with open(output_dir / "ratchet_params.json", "w") as f:
        json.dump(params, f, indent=2)

    # Save per-cell summaries
    summaries = []
    for cell_id, cell in dataset.cells.items():
        summaries.append({
            'cell_id': cell_id,
            'initial_capacity': cell.initial_capacity,
            'final_soh': cell.final_soh,
            'num_cycles': len(cell.cycle_numbers),
            'fade_rate': cell.fade_rate,
            'avg_temperature': float(np.mean(cell.temperatures)) if len(cell.temperatures) > 0 else None,
        })

    summary_df = pd.DataFrame(summaries)
    if format == "parquet":
        summary_df.to_parquet(output_dir / "cell_summaries.parquet", index=False)
    elif format == "csv":
        summary_df.to_csv(output_dir / "cell_summaries.csv", index=False)


def prepare_for_engine(
    dataset: NASABatteryDataset,
    cell_id: Optional[str] = None,
) -> Dict:
    """
    Prepare data for comparison with BatteryDegradationEngine.

    Returns a dict with arrays suitable for comparing simulated vs empirical:
        - cycles: Array of cycle numbers
        - empirical_soh: Measured SOH trajectory
        - empirical_fade: Measured capacity fade (1 - SOH)
        - initial_capacity: Starting capacity in Ah
        - temperature: Average operating temperature

    Args:
        dataset: NASABatteryDataset
        cell_id: Specific cell to use. If None, uses first cell.
    """
    if cell_id is None:
        cell_id = list(dataset.cells.keys())[0]

    cell = dataset.cells[cell_id]

    return {
        'cell_id': cell_id,
        'cycles': cell.cycle_numbers.copy(),
        'empirical_soh': cell.soh_values.copy(),
        'empirical_fade': cell.f.copy(),
        'empirical_capacity': cell.capacities.copy(),
        'initial_capacity': cell.initial_capacity,
        'temperature': float(np.mean(cell.temperatures)) if len(cell.temperatures) > 0 else 24.0,
        'num_cycles': len(cell.cycle_numbers),
        'final_soh': cell.final_soh,
    }


__all__ = [
    'NASABatteryLoader',
    'NASABatteryDataset',
    'BatteryData',
    'BatteryCycleData',
    'load_nasa_battery_data',
    'get_high_quality_cells',
    'save_processed_data',
    'prepare_for_engine',
]
