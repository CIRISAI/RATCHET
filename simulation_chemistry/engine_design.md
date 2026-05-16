# Battery Degradation Simulation Engine - Architecture Design

This document describes the proposed architecture for a battery degradation simulation engine following the RATCHET interface requirements.

---

## Design Philosophy

1. **Domain fidelity**: The engine models electrochemical degradation physics, not abstract dynamics.
2. **Theory agnostic**: No assumptions about relationships between variables are built in.
3. **Measurable outputs**: All variables can be extracted from simulation state.
4. **Manipulable inputs**: Operating conditions can be modified during simulation.
5. **Data compatibility**: Can be initialized/validated against open battery datasets.

---

## Class Hierarchy

```
BatteryDegradationEngine (main interface)
    |
    +-- CellModel (single cell physics)
    |       +-- ElectrodeModel (cathode/anode)
    |       +-- ElectrolyteModel
    |       +-- SEIModel
    |       +-- ThermalModel
    |
    +-- PackModel (multi-cell pack)
    |       +-- CellArray
    |       +-- ThermalNetwork
    |       +-- BalancingController
    |
    +-- DegradationModes
    |       +-- SEIGrowth
    |       +-- LithiumPlating
    |       +-- ParticleCracking
    |       +-- BinderDecomposition
    |       +-- TransitionMetalDissolution
    |
    +-- DataLoader
    |       +-- NASALoader
    |       +-- CALCELoader
    |       +-- MITStanfordLoader
    |       +-- BatteryArchiveLoader
    |
    +-- Interventions
            +-- VoltageWindowChange
            +-- TemperatureChange
            +-- BalancingIntervention
            +-- CellReplacement
```

---

## Core Classes

### BatteryDegradationEngine

The main simulation interface implementing the RATCHET requirements.

```python
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum
from abc import ABC, abstractmethod


class InterventionType(Enum):
    """Available intervention types."""
    VOLTAGE_WINDOW = "voltage_window"
    TEMPERATURE_CHANGE = "temperature"
    BALANCING = "balancing"
    CELL_REPLACEMENT = "cell_replacement"
    PROTOCOL_CHANGE = "protocol"
    REGENERATION = "regeneration"  # Rest period, reconditioning


@dataclass
class Shock:
    """External perturbation to system state."""
    type: str  # 'thermal', 'mechanical', 'electrical', 'abuse'
    magnitude: float
    duration: float
    target: Optional[str] = None  # Specific cell or 'all'


@dataclass
class Intervention:
    """Discrete action applied to system."""
    type: InterventionType
    parameters: Dict
    target: Optional[str] = None


@dataclass
class CellState:
    """State of a single battery cell."""
    capacity: float  # Current capacity (Ah)
    resistance: float  # Internal resistance (Ohm)
    soh: float  # State of health [0, 1]
    sei_thickness: float  # SEI layer thickness (nm)
    li_inventory: float  # Remaining lithium inventory (Ah)
    temperature: float  # Cell temperature (C)
    soc: float  # State of charge [0, 1]
    cycle_count: int  # Number of cycles completed
    calendar_age: float  # Calendar age (hours)

    # Degradation mode contributions
    lam_pe: float = 0.0  # Loss of active material - positive electrode
    lam_ne: float = 0.0  # Loss of active material - negative electrode
    lli: float = 0.0  # Loss of lithium inventory


@dataclass
class EngineConfig:
    """Configuration for battery simulation engine."""
    # Cell parameters
    num_cells: int = 1
    initial_capacity: float = 2.0  # Ah
    initial_resistance: float = 0.05  # Ohm
    chemistry: str = "NMC"  # NMC, LFP, NCA, LCO

    # Operating constraints (initial)
    voltage_min: float = 2.5  # V
    voltage_max: float = 4.2  # V
    temperature_min: float = -20.0  # C
    temperature_max: float = 60.0  # C
    c_rate_max: float = 2.0  # C

    # Degradation parameters (can be calibrated from data)
    sei_growth_rate: float = 0.001  # Base SEI growth rate
    calendar_aging_rate: float = 0.0001  # Base calendar aging
    activation_energy: float = 50000  # J/mol for Arrhenius

    # Collapse threshold
    soh_collapse_threshold: float = 0.80

    # Simulation settings
    time_step: float = 1.0  # hours
    log_interval: int = 10  # Log every N steps


class BatteryDegradationEngine:
    """
    Battery degradation simulation engine.

    Implements the RATCHET simulation interface for electrochemical systems.
    """

    def __init__(self, config: EngineConfig):
        """Initialize engine with configuration."""
        self.config = config
        self.time = 0.0
        self.step_count = 0

        # Initialize cell states
        self.cells: List[CellState] = []
        for i in range(config.num_cells):
            cell = CellState(
                capacity=config.initial_capacity,
                resistance=config.initial_resistance,
                soh=1.0,
                sei_thickness=1.0,  # Initial SEI (nm)
                li_inventory=config.initial_capacity,
                temperature=25.0,
                soc=0.5,
                cycle_count=0,
                calendar_age=0.0
            )
            self.cells.append(cell)

        # Operating constraints (mutable)
        self._k = config.num_cells  # Constraint count
        self._alpha = config.sei_growth_rate  # Constraint generation rate
        self._d = config.calendar_aging_rate  # Decay rate
        self._lambda = self._calculate_initial_strictness()  # Strictness

        # History tracking
        self.history: List[Dict] = []
        self._collapsed = False
        self._collapse_time: Optional[float] = None

    def _calculate_initial_strictness(self) -> float:
        """Calculate initial strictness from operating window."""
        # Voltage window as fraction of theoretical max
        v_range = self.config.voltage_max - self.config.voltage_min
        v_max_range = 4.5 - 2.0  # Theoretical max window
        v_strictness = 1 - (v_range / v_max_range)

        # Temperature window
        t_range = self.config.temperature_max - self.config.temperature_min
        t_max_range = 100  # -40 to 60
        t_strictness = 1 - (t_range / t_max_range)

        # C-rate
        c_strictness = 1 - (self.config.c_rate_max / 10)

        return np.mean([v_strictness, t_strictness, c_strictness])

    # =========================================================================
    # CORE SIMULATION
    # =========================================================================

    def step(self, dt: float = None) -> None:
        """
        Advance simulation by one time step.

        Args:
            dt: Time step in hours. Uses config default if not specified.
        """
        if dt is None:
            dt = self.config.time_step

        # Update each cell
        for cell in self.cells:
            self._update_cell(cell, dt)

        # Update global time
        self.time += dt
        self.step_count += 1

        # Check for collapse
        self._check_collapse()

        # Log state
        if self.step_count % self.config.log_interval == 0:
            self._log_state()

    def _update_cell(self, cell: CellState, dt: float) -> None:
        """Update single cell state for one time step."""
        # Calendar aging (always active)
        calendar_fade = self._calculate_calendar_fade(cell, dt)

        # SEI growth (temperature and SOC dependent)
        sei_growth = self._calculate_sei_growth(cell, dt)

        # Update SEI thickness
        cell.sei_thickness += sei_growth

        # SEI consumes lithium inventory
        li_consumed = sei_growth * 0.01  # Proportional to growth
        cell.li_inventory -= li_consumed
        cell.lli += li_consumed / self.config.initial_capacity

        # Resistance increase from SEI
        cell.resistance *= (1 + sei_growth * 0.001)

        # Capacity fade from multiple mechanisms
        capacity_fade = calendar_fade + li_consumed
        cell.capacity = max(0, cell.capacity - capacity_fade)

        # Update SOH
        cell.soh = cell.capacity / self.config.initial_capacity

        # Update calendar age
        cell.calendar_age += dt

    def _calculate_calendar_fade(self, cell: CellState, dt: float) -> float:
        """Calculate capacity fade from calendar aging."""
        # Arrhenius temperature dependence
        T = cell.temperature + 273.15
        T_ref = 298.15
        Ea = self.config.activation_energy
        R = 8.314

        temp_factor = np.exp(Ea / R * (1/T_ref - 1/T))

        # SOC dependence (higher SOC = faster aging)
        soc_factor = 1 + cell.soc

        fade = self._d * temp_factor * soc_factor * dt * cell.capacity

        return fade

    def _calculate_sei_growth(self, cell: CellState, dt: float) -> float:
        """Calculate SEI layer growth."""
        # Parabolic growth law: thickness ~ sqrt(time)
        # d(thickness)/dt ~ 1/sqrt(t) ~ 1/thickness

        T = cell.temperature + 273.15
        T_ref = 298.15
        Ea = self.config.activation_energy
        R = 8.314

        temp_factor = np.exp(Ea / R * (1/T_ref - 1/T))

        # Growth slows as SEI thickens (diffusion limited)
        thickness_factor = 1 / max(1, cell.sei_thickness)

        growth = self._alpha * temp_factor * thickness_factor * dt

        return growth

    def _check_collapse(self) -> None:
        """Check if system has collapsed."""
        if self._collapsed:
            return

        # Collapse if any cell below threshold
        avg_soh = self.get_sigma()
        if avg_soh < self.config.soh_collapse_threshold:
            self._collapsed = True
            self._collapse_time = self.time

    def _log_state(self) -> None:
        """Log current state to history."""
        state = {
            'time': self.time,
            'step': self.step_count,
            'k': self._k,
            'k_eff': self.get_k_eff(),
            'rho': self.get_rho(),
            'sigma': self.get_sigma(),
            'f': self.get_f(),
            'alpha': self._alpha,
            'd': self._d,
            'lambda': self._lambda,
            'collapsed': self._collapsed,
            'avg_temperature': np.mean([c.temperature for c in self.cells]),
            'avg_resistance': np.mean([c.resistance for c in self.cells]),
            'avg_sei_thickness': np.mean([c.sei_thickness for c in self.cells]),
        }

        # Per-cell SOH
        for i, cell in enumerate(self.cells):
            state[f'cell_{i}_soh'] = cell.soh
            state[f'cell_{i}_capacity'] = cell.capacity

        self.history.append(state)

    def run(self, duration: float, dt: float = None) -> pd.DataFrame:
        """
        Run simulation for specified duration.

        Args:
            duration: Total simulation time in hours.
            dt: Time step in hours.

        Returns:
            DataFrame with time series of all variables.
        """
        if dt is None:
            dt = self.config.time_step

        steps = int(duration / dt)

        for _ in range(steps):
            self.step(dt)

            # Early termination if collapsed
            if self._collapsed:
                break

        return self.to_dataframe()

    # =========================================================================
    # VARIABLE MANIPULATION
    # =========================================================================

    def set_k(self, k: int) -> None:
        """
        Set constraint count (number of cells).

        Adding cells represents adding redundancy.
        Removing cells represents failure/disconnection.
        """
        if k < 1:
            raise ValueError("k must be at least 1")

        if k > self._k:
            # Add new cells (fresh)
            for _ in range(k - self._k):
                cell = CellState(
                    capacity=self.config.initial_capacity,
                    resistance=self.config.initial_resistance,
                    soh=1.0,
                    sei_thickness=1.0,
                    li_inventory=self.config.initial_capacity,
                    temperature=25.0,
                    soc=0.5,
                    cycle_count=0,
                    calendar_age=0.0
                )
                self.cells.append(cell)
        elif k < self._k:
            # Remove worst cells
            self.cells.sort(key=lambda c: c.soh)
            self.cells = self.cells[self._k - k:]

        self._k = k

    def set_alpha(self, alpha: float) -> None:
        """Set constraint generation rate (SEI growth rate)."""
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        self._alpha = alpha

    def set_d(self, d: float) -> None:
        """Set decay rate (calendar aging rate)."""
        if d < 0:
            raise ValueError("d must be non-negative")
        self._d = d

    def set_lambda(self, lambda_: float) -> None:
        """
        Set strictness (operating constraint tightness).

        Higher lambda = stricter limits = less stress.
        """
        if not 0 <= lambda_ <= 1:
            raise ValueError("lambda must be in [0, 1]")
        self._lambda = lambda_

        # Update operating constraints based on strictness
        # Higher lambda = narrower voltage window
        v_center = (4.2 + 2.5) / 2
        v_half_range = (1 - lambda_) * (4.2 - 2.5) / 2
        self.config.voltage_max = v_center + v_half_range
        self.config.voltage_min = v_center - v_half_range

    def apply_shock(self, shock: Shock) -> None:
        """
        Apply external perturbation to system.

        Args:
            shock: Shock specification (type, magnitude, duration, target).
        """
        targets = []
        if shock.target is None or shock.target == 'all':
            targets = self.cells
        else:
            try:
                idx = int(shock.target)
                targets = [self.cells[idx]]
            except (ValueError, IndexError):
                raise ValueError(f"Invalid shock target: {shock.target}")

        for cell in targets:
            if shock.type == 'thermal':
                # Temperature spike
                cell.temperature += shock.magnitude

            elif shock.type == 'mechanical':
                # Capacity loss from physical damage
                cell.capacity *= (1 - shock.magnitude * 0.01)
                cell.soh = cell.capacity / self.config.initial_capacity

            elif shock.type == 'electrical':
                # Resistance increase from electrical abuse
                cell.resistance *= (1 + shock.magnitude * 0.1)

            elif shock.type == 'abuse':
                # Combined effects
                cell.capacity *= (1 - shock.magnitude * 0.05)
                cell.resistance *= (1 + shock.magnitude * 0.2)
                cell.soh = cell.capacity / self.config.initial_capacity

    def apply_intervention(self, intervention: Intervention) -> None:
        """
        Apply discrete action to system.

        Args:
            intervention: Intervention specification.
        """
        if intervention.type == InterventionType.VOLTAGE_WINDOW:
            new_max = intervention.parameters.get('voltage_max', self.config.voltage_max)
            new_min = intervention.parameters.get('voltage_min', self.config.voltage_min)
            self.config.voltage_max = new_max
            self.config.voltage_min = new_min
            self._lambda = self._calculate_initial_strictness()

        elif intervention.type == InterventionType.TEMPERATURE_CHANGE:
            new_temp = intervention.parameters.get('temperature', 25.0)
            for cell in self.cells:
                cell.temperature = new_temp

        elif intervention.type == InterventionType.BALANCING:
            # Equalize SOC across cells
            avg_soc = np.mean([c.soc for c in self.cells])
            for cell in self.cells:
                cell.soc = avg_soc

        elif intervention.type == InterventionType.CELL_REPLACEMENT:
            # Replace worst cell with fresh one
            if len(self.cells) > 0:
                self.cells.sort(key=lambda c: c.soh)
                self.cells[0] = CellState(
                    capacity=self.config.initial_capacity,
                    resistance=self.config.initial_resistance,
                    soh=1.0,
                    sei_thickness=1.0,
                    li_inventory=self.config.initial_capacity,
                    temperature=25.0,
                    soc=0.5,
                    cycle_count=0,
                    calendar_age=0.0
                )

        elif intervention.type == InterventionType.REGENERATION:
            # Rest period - reduce stress, allow recovery
            rest_hours = intervention.parameters.get('duration', 24)
            # Lower temperature during rest
            for cell in self.cells:
                cell.temperature = 20.0
                cell.soc = 0.5  # Optimal storage SOC
            # Run rest period with reduced aging
            original_alpha = self._alpha
            self._alpha *= 0.5  # Reduced SEI growth during rest
            for _ in range(int(rest_hours)):
                self.step(1.0)
            self._alpha = original_alpha

    # =========================================================================
    # VARIABLE MEASUREMENT
    # =========================================================================

    def get_rho(self) -> float:
        """
        Get pairwise correlation between cells.

        For single cell, returns 0.
        For multiple cells, returns average correlation of degradation trajectories.
        """
        if len(self.cells) <= 1:
            return 0.0

        # Use SOH as degradation indicator
        soh_values = np.array([c.soh for c in self.cells])

        # For instantaneous measurement, use deviation from mean
        # High correlation = cells degrade similarly
        mean_soh = np.mean(soh_values)
        if mean_soh == 0:
            return 1.0

        # Coefficient of variation (inverted) as proxy for correlation
        cv = np.std(soh_values) / mean_soh
        rho = 1 - min(1, cv * 10)  # Scale CV to [0, 1] and invert

        return max(0, min(1, rho))

    def get_k_eff(self) -> float:
        """
        Get effective constraint count.

        k_eff = k / (1 + rho * (k - 1))
        """
        k = self._k
        rho = self.get_rho()

        if k <= 1:
            return float(k)

        k_eff = k / (1 + rho * (k - 1))
        return k_eff

    def get_sigma(self) -> float:
        """
        Get sustainability metric (average SOH).

        Returns average state of health across all cells.
        """
        if len(self.cells) == 0:
            return 0.0

        return np.mean([c.soh for c in self.cells])

    def get_f(self) -> float:
        """
        Get compromised fraction.

        Fraction of original capacity lost.
        """
        return 1 - self.get_sigma()

    def get_state(self) -> np.ndarray:
        """
        Get full state vector for analysis.

        Returns flattened array of all cell states.
        """
        state_list = []
        for cell in self.cells:
            state_list.extend([
                cell.capacity,
                cell.resistance,
                cell.soh,
                cell.sei_thickness,
                cell.li_inventory,
                cell.temperature,
                cell.soc,
                float(cell.cycle_count),
                cell.calendar_age,
                cell.lam_pe,
                cell.lam_ne,
                cell.lli
            ])
        return np.array(state_list)

    def is_collapsed(self) -> bool:
        """Check if system has collapsed."""
        return self._collapsed

    def get_collapse_time(self) -> Optional[float]:
        """Get time at which collapse occurred."""
        return self._collapse_time

    # =========================================================================
    # DATA EXPORT
    # =========================================================================

    def to_dataframe(self) -> pd.DataFrame:
        """Export simulation history as DataFrame."""
        if not self.history:
            return pd.DataFrame()

        return pd.DataFrame(self.history)

    def get_cell_trajectories(self) -> pd.DataFrame:
        """Get per-cell degradation trajectories."""
        if not self.history:
            return pd.DataFrame()

        df = pd.DataFrame(self.history)
        cell_cols = [col for col in df.columns if col.startswith('cell_')]
        return df[['time', 'step'] + cell_cols]
```

---

## Degradation Mode Models

### SEI Growth Model

```python
class SEIGrowthModel:
    """
    Solid Electrolyte Interphase (SEI) growth model.

    SEI forms on anode surface from electrolyte decomposition.
    Growth follows parabolic kinetics (diffusion limited).
    """

    def __init__(self, k_sei: float = 0.001, Ea: float = 50000):
        """
        Args:
            k_sei: SEI growth rate constant
            Ea: Activation energy (J/mol)
        """
        self.k_sei = k_sei
        self.Ea = Ea
        self.R = 8.314  # J/(mol*K)

    def growth_rate(self, thickness: float, temperature: float, soc: float) -> float:
        """
        Calculate SEI growth rate.

        Args:
            thickness: Current SEI thickness (nm)
            temperature: Cell temperature (C)
            soc: State of charge [0, 1]

        Returns:
            Growth rate (nm/hour)
        """
        T = temperature + 273.15
        T_ref = 298.15

        # Arrhenius temperature dependence
        temp_factor = np.exp(self.Ea / self.R * (1/T_ref - 1/T))

        # Parabolic growth (diffusion limited)
        thickness_factor = 1 / max(1, thickness)

        # SOC dependence (higher SOC = higher anode potential = faster growth)
        soc_factor = 0.5 + soc

        rate = self.k_sei * temp_factor * thickness_factor * soc_factor

        return rate

    def lithium_consumption(self, growth: float) -> float:
        """
        Calculate lithium consumed by SEI growth.

        Args:
            growth: SEI thickness increase (nm)

        Returns:
            Lithium consumed (Ah equivalent)
        """
        # Empirical: ~0.01 Ah per nm of SEI
        return growth * 0.01
```

### Lithium Plating Model

```python
class LithiumPlatingModel:
    """
    Lithium plating model for fast charging conditions.

    Metallic lithium deposits on anode when local potential < 0V vs Li/Li+.
    """

    def __init__(self, k_plating: float = 0.0001, c_rate_threshold: float = 1.0):
        """
        Args:
            k_plating: Plating rate constant
            c_rate_threshold: C-rate above which plating occurs
        """
        self.k_plating = k_plating
        self.c_rate_threshold = c_rate_threshold

    def plating_rate(
        self,
        c_rate: float,
        temperature: float,
        soc: float
    ) -> float:
        """
        Calculate lithium plating rate.

        Args:
            c_rate: Charge rate (C)
            temperature: Cell temperature (C)
            soc: State of charge [0, 1]

        Returns:
            Plating rate (Ah/hour)
        """
        if c_rate < self.c_rate_threshold:
            return 0.0

        # Higher C-rate = more plating
        c_factor = (c_rate - self.c_rate_threshold) / self.c_rate_threshold

        # Low temperature increases plating (slower kinetics)
        if temperature < 25:
            temp_factor = 1 + (25 - temperature) / 25
        else:
            temp_factor = 1.0

        # High SOC increases plating (anode already full)
        soc_factor = soc ** 2

        rate = self.k_plating * c_factor * temp_factor * soc_factor

        return rate
```

---

## Data Loaders

### Base Loader Interface

```python
from abc import ABC, abstractmethod


class BatteryDataLoader(ABC):
    """Abstract base class for battery dataset loaders."""

    @abstractmethod
    def load(self, path: str) -> Dict:
        """Load dataset from path."""
        pass

    @abstractmethod
    def get_cells(self) -> List[str]:
        """Get list of cell identifiers."""
        pass

    @abstractmethod
    def get_capacity_trajectory(self, cell_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """Get capacity vs time/cycle for a cell."""
        pass

    @abstractmethod
    def get_eis_data(self, cell_id: str) -> Optional[Dict]:
        """Get EIS data for a cell (if available)."""
        pass

    def calibrate_engine(self, engine: BatteryDegradationEngine, cell_id: str) -> None:
        """Calibrate engine parameters from real data."""
        time, capacity = self.get_capacity_trajectory(cell_id)
        initial_capacity = capacity[0]

        # Estimate decay rate from capacity fade
        soh = capacity / initial_capacity
        fade_rate = -np.gradient(soh, time)
        engine.set_d(np.mean(fade_rate))
```

### NASA Dataset Loader

```python
import scipy.io as sio


class NASALoader(BatteryDataLoader):
    """Loader for NASA Li-ion Battery Aging Dataset."""

    def __init__(self):
        self.data = {}
        self.cells = []

    def load(self, path: str) -> Dict:
        """
        Load NASA dataset from .mat files.

        Args:
            path: Path to directory containing .mat files

        Returns:
            Dictionary with loaded data
        """
        import os
        mat_files = [f for f in os.listdir(path) if f.endswith('.mat')]

        for mat_file in mat_files:
            cell_id = mat_file.replace('.mat', '')
            try:
                mat_data = sio.loadmat(os.path.join(path, mat_file))
                self.data[cell_id] = self._parse_mat(mat_data)
                self.cells.append(cell_id)
            except Exception as e:
                print(f"Warning: Could not load {mat_file}: {e}")

        return self.data

    def _parse_mat(self, mat_data: Dict) -> Dict:
        """Parse MATLAB structure to Python dict."""
        parsed = {
            'cycles': [],
            'capacity': [],
            'time': [],
            'eis': []
        }

        # NASA format has 'cycle' structure
        if 'cycle' in mat_data:
            cycles = mat_data['cycle']
            for i in range(cycles.shape[1]):
                cycle = cycles[0, i]
                parsed['cycles'].append(i)

                # Extract capacity from discharge
                if hasattr(cycle, 'Qd'):
                    parsed['capacity'].append(float(cycle.Qd.max()))
                    parsed['time'].append(float(cycle.time.max()))

        return parsed

    def get_cells(self) -> List[str]:
        """Get list of cell identifiers."""
        return self.cells

    def get_capacity_trajectory(self, cell_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """Get capacity vs cycle for a cell."""
        if cell_id not in self.data:
            raise ValueError(f"Cell {cell_id} not found")

        cell_data = self.data[cell_id]
        cycles = np.array(cell_data['cycles'])
        capacity = np.array(cell_data['capacity'])

        return cycles, capacity

    def get_eis_data(self, cell_id: str) -> Optional[Dict]:
        """Get EIS data for a cell."""
        if cell_id not in self.data:
            return None

        cell_data = self.data[cell_id]
        if 'eis' in cell_data and len(cell_data['eis']) > 0:
            return cell_data['eis']
        return None
```

### CALCE Dataset Loader

```python
class CALCELoader(BatteryDataLoader):
    """Loader for CALCE Battery Datasets."""

    def __init__(self):
        self.data = {}
        self.cells = []

    def load(self, path: str) -> Dict:
        """
        Load CALCE dataset from CSV files.

        Args:
            path: Path to directory containing CSV files

        Returns:
            Dictionary with loaded data
        """
        import os

        csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]

        for csv_file in csv_files:
            cell_id = csv_file.replace('.csv', '')
            try:
                df = pd.read_csv(os.path.join(path, csv_file))
                self.data[cell_id] = self._parse_csv(df)
                self.cells.append(cell_id)
            except Exception as e:
                print(f"Warning: Could not load {csv_file}: {e}")

        return self.data

    def _parse_csv(self, df: pd.DataFrame) -> Dict:
        """Parse CSV DataFrame to standard format."""
        parsed = {
            'cycles': [],
            'capacity': [],
            'time': [],
            'voltage': [],
            'current': [],
            'temperature': []
        }

        # CALCE format varies, adapt as needed
        if 'Cycle_Index' in df.columns:
            parsed['cycles'] = df['Cycle_Index'].values
        if 'Discharge_Capacity(Ah)' in df.columns:
            parsed['capacity'] = df['Discharge_Capacity(Ah)'].values
        if 'Test_Time(s)' in df.columns:
            parsed['time'] = df['Test_Time(s)'].values / 3600  # Convert to hours

        return parsed

    def get_cells(self) -> List[str]:
        return self.cells

    def get_capacity_trajectory(self, cell_id: str) -> Tuple[np.ndarray, np.ndarray]:
        if cell_id not in self.data:
            raise ValueError(f"Cell {cell_id} not found")

        cell_data = self.data[cell_id]
        cycles = np.array(cell_data['cycles'])
        capacity = np.array(cell_data['capacity'])

        return cycles, capacity

    def get_eis_data(self, cell_id: str) -> Optional[Dict]:
        # CALCE EIS data in separate files
        return None
```

---

## Example Usage

```python
# Basic simulation
config = EngineConfig(
    num_cells=4,
    initial_capacity=2.0,
    chemistry="NMC",
    soh_collapse_threshold=0.80
)

engine = BatteryDegradationEngine(config)

# Run for 1 year (8760 hours)
results = engine.run(duration=8760, dt=1.0)

# Check outcomes
print(f"Final SOH: {engine.get_sigma():.2%}")
print(f"Collapsed: {engine.is_collapsed()}")
if engine.is_collapsed():
    print(f"Collapse time: {engine.get_collapse_time():.0f} hours")

# Variable manipulation example
engine.set_lambda(0.9)  # Increase strictness
engine.set_alpha(0.0005)  # Reduce SEI growth rate

# Apply shock
shock = Shock(type='thermal', magnitude=20, duration=1)
engine.apply_shock(shock)

# Apply intervention
intervention = Intervention(
    type=InterventionType.REGENERATION,
    parameters={'duration': 48}
)
engine.apply_intervention(intervention)

# Export data
df = engine.to_dataframe()
df.to_csv('simulation_results.csv', index=False)
```

---

## Integration with Real Data

```python
# Load real dataset
loader = NASALoader()
loader.load('/path/to/nasa/data')

# Get reference cell
cells = loader.get_cells()
cycles, capacity = loader.get_capacity_trajectory(cells[0])

# Calibrate engine from data
engine = BatteryDegradationEngine(EngineConfig())
loader.calibrate_engine(engine, cells[0])

# Run simulation
sim_results = engine.run(duration=len(cycles))

# Compare
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(cycles, capacity / capacity[0], label='Real data')
plt.plot(sim_results['step'], sim_results['sigma'], label='Simulation')
plt.xlabel('Cycle')
plt.ylabel('SOH')
plt.legend()
plt.savefig('validation_comparison.png')
```

---

## Extension Points

### Custom Degradation Modes

```python
class CustomDegradationMode(ABC):
    @abstractmethod
    def calculate_fade(self, cell: CellState, dt: float) -> float:
        pass


class TransitionMetalDissolution(CustomDegradationMode):
    """Model for cathode transition metal dissolution."""

    def __init__(self, k_dissolution: float = 0.0001):
        self.k_dissolution = k_dissolution

    def calculate_fade(self, cell: CellState, dt: float) -> float:
        # Higher temperature accelerates dissolution
        T = cell.temperature + 273.15
        T_ref = 298.15
        temp_factor = np.exp(50000 / 8.314 * (1/T_ref - 1/T))

        # Higher voltage accelerates dissolution
        # (proxy: higher SOC)
        voltage_factor = cell.soc ** 2

        fade = self.k_dissolution * temp_factor * voltage_factor * dt

        return fade
```

### Custom Interventions

```python
def register_intervention(engine: BatteryDegradationEngine,
                          intervention_type: str,
                          handler: callable) -> None:
    """Register custom intervention handler."""
    engine._intervention_handlers[intervention_type] = handler
```

---

## Testing Strategy

1. **Unit tests**: Each model component tested independently
2. **Integration tests**: Full engine simulation runs
3. **Validation tests**: Compare against real datasets
4. **Boundary tests**: Edge cases (0 cells, extreme temperatures, etc.)

```python
def test_collapse_detection():
    config = EngineConfig(num_cells=1, soh_collapse_threshold=0.80)
    engine = BatteryDegradationEngine(config)

    # Force rapid degradation
    engine.set_alpha(0.1)
    engine.set_d(0.01)

    # Run until collapse
    engine.run(duration=10000, dt=1.0)

    assert engine.is_collapsed()
    assert engine.get_collapse_time() is not None
    assert engine.get_sigma() < 0.80
```
