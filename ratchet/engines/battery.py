"""
RATCHET Battery Degradation Simulation Engine

Simulates lithium-ion battery degradation for structural invariant testing.
Theory-agnostic: exposes manipulable and measurable variables without
assuming relationships between them.

Domain Mapping:
    k (constraints):     Number of cells in pack / active electrode sites
    rho (correlation):   Cross-cell SOH correlation
    sigma (sustainability): State of Health (SOH) = Q_current / Q_initial
    f (compromise):      Capacity fade fraction (1 - SOH)
    d (decay rate):      Calendar aging rate (SEI growth at rest)
    alpha (generation):  Cyclic aging rate (SEI growth under load)
    lambda (strictness): Operating window tightness (voltage, temp limits)

References:
    - NASA Li-ion Battery Aging Datasets
    - MIT-Stanford Battery Dataset (Severson et al. 2019)
    - CALCE Battery Research Group datasets
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


class BatteryShockType(Enum):
    """Types of perturbations to the battery system."""
    THERMAL = "thermal"
    MECHANICAL = "mechanical"
    ELECTRICAL = "electrical"
    ABUSE = "abuse"


class BatteryInterventionType(Enum):
    """Types of interventions on the battery system."""
    VOLTAGE_WINDOW = "voltage_window"
    TEMPERATURE = "temperature"
    BALANCING = "balancing"
    CELL_REPLACEMENT = "cell_replacement"
    REGENERATION = "regeneration"


@dataclass
class BatteryShock:
    """External perturbation to the battery pack."""
    type: BatteryShockType
    magnitude: float = 0.3
    target: Optional[str] = None  # "all" or cell index
    duration: float = 1.0


@dataclass
class BatteryIntervention:
    """Intervention on the battery system."""
    type: BatteryInterventionType
    parameters: Dict = field(default_factory=dict)


@dataclass
class CellState:
    """State of a single battery cell."""
    capacity: float
    resistance: float
    soh: float
    sei_thickness: float
    li_inventory: float
    temperature: float
    soc: float
    cycle_count: int
    calendar_age: float


@dataclass
class BatteryParams:
    """Configuration for BatteryDegradationEngine."""
    engine: str = "battery"
    num_cells: int = 4
    initial_capacity: float = 2.0
    initial_resistance: float = 0.05
    chemistry: str = "NMC"
    voltage_min: float = 2.5
    voltage_max: float = 4.2
    temperature_min: float = -20.0
    temperature_max: float = 60.0
    soh_collapse_threshold: float = 0.80
    sei_growth_rate: float = 0.001
    calendar_aging_rate: float = 0.0001
    activation_energy: float = 50000.0
    seed: Optional[int] = None


class BatteryDegradationEngine:
    """
    Battery degradation simulation engine.

    Simulates electrochemical degradation (SEI growth, capacity fade)
    for single cells or multi-cell packs. Exposes RATCHET framework
    variables for manipulation and measurement.

    Example:
        >>> engine = BatteryDegradationEngine(seed=42)
        >>> engine.initialize()
        >>> ts = engine.run(duration=8760, dt=1.0)  # 1 year in hours
        >>> print(f"Final SOH: {engine.get_sigma():.2%}")
    """

    def __init__(
        self,
        params: Optional[BatteryParams] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize the BatteryDegradationEngine.

        Args:
            params: Engine configuration. If None, uses defaults.
            seed: Random seed for reproducibility.
        """
        self.params = params or BatteryParams()
        if seed is not None:
            self.params.seed = seed

        self.rng = np.random.default_rng(self.params.seed)
        self.time = 0.0
        self.step_count = 0

        # Cell states
        self.cells: List[CellState] = []

        # Dynamics parameters (manipulable)
        self._alpha = self.params.sei_growth_rate
        self._d = self.params.calendar_aging_rate
        self._lambda = 0.5  # Default strictness

        # History
        self._history: List[Dict] = []
        self._collapsed = False
        self._collapse_time: Optional[float] = None

    def initialize(self) -> None:
        """Initialize all cells to fresh state."""
        self.cells = []
        for _ in range(self.params.num_cells):
            cell = CellState(
                capacity=self.params.initial_capacity,
                resistance=self.params.initial_resistance,
                soh=1.0,
                sei_thickness=1.0,
                li_inventory=self.params.initial_capacity,
                temperature=25.0,
                soc=0.5,
                cycle_count=0,
                calendar_age=0.0,
            )
            self.cells.append(cell)

        self._lambda = self._calculate_strictness()
        self._history = [self._record_state()]
        self._collapsed = False
        self._collapse_time = None

    def _calculate_strictness(self) -> float:
        """Calculate strictness from operating window."""
        v_range = self.params.voltage_max - self.params.voltage_min
        v_max_range = 4.5 - 2.0
        v_strictness = 1 - (v_range / v_max_range)

        t_range = self.params.temperature_max - self.params.temperature_min
        t_max_range = 100
        t_strictness = 1 - (t_range / t_max_range)

        return np.mean([v_strictness, t_strictness])

    # =========================================================================
    # CORE SIMULATION
    # =========================================================================

    def step(self, dt: float = 1.0) -> None:
        """Advance simulation by one time step (in hours)."""
        if not self.cells:
            raise RuntimeError("Engine not initialized")

        if self._collapsed:
            return

        for cell in self.cells:
            self._update_cell(cell, dt)

        self.time += dt
        self.step_count += 1
        self._history.append(self._record_state())
        self._check_collapse()

    def _update_cell(self, cell: CellState, dt: float) -> None:
        """Update single cell for one time step."""
        # Arrhenius temperature factor
        T = cell.temperature + 273.15
        T_ref = 298.15
        Ea = self.params.activation_energy
        R = 8.314
        temp_factor = np.exp(Ea / R * (1 / T_ref - 1 / T))

        # Calendar aging
        soc_factor = 1 + cell.soc
        calendar_fade = self._d * temp_factor * soc_factor * dt * cell.capacity / 1000

        # SEI growth (parabolic kinetics)
        thickness_factor = 1 / max(1, cell.sei_thickness)
        sei_growth = self._alpha * temp_factor * thickness_factor * dt
        cell.sei_thickness += sei_growth

        # Li inventory loss from SEI
        li_consumed = sei_growth * 0.01
        cell.li_inventory = max(0, cell.li_inventory - li_consumed)

        # Resistance increase
        cell.resistance *= (1 + sei_growth * 0.001)

        # Capacity fade
        total_fade = calendar_fade + li_consumed
        cell.capacity = max(0, cell.capacity - total_fade)

        # Update SOH
        cell.soh = cell.capacity / self.params.initial_capacity
        cell.calendar_age += dt

    def _record_state(self) -> Dict:
        """Record current state for history."""
        return {
            'time': self.time,
            'step': self.step_count,
            'k': len(self.cells),
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

    def _check_collapse(self) -> None:
        """Check if system has collapsed."""
        if self._collapsed:
            return

        avg_soh = self.get_sigma()
        if avg_soh < self.params.soh_collapse_threshold:
            self._collapsed = True
            self._collapse_time = self.time

    def run(self, duration: float, dt: float = 1.0) -> pd.DataFrame:
        """Run simulation for specified duration (in hours)."""
        n_steps = int(duration / dt)
        for _ in range(n_steps):
            self.step(dt)
            if self._collapsed:
                break
        return self.to_dataframe()

    # =========================================================================
    # VARIABLE MANIPULATION (Inputs)
    # =========================================================================

    def set_k(self, k: int) -> None:
        """Set constraint count (number of cells)."""
        if k < 1:
            raise ValueError("k must be at least 1")

        if k > len(self.cells):
            # Add fresh cells
            for _ in range(k - len(self.cells)):
                cell = CellState(
                    capacity=self.params.initial_capacity,
                    resistance=self.params.initial_resistance,
                    soh=1.0,
                    sei_thickness=1.0,
                    li_inventory=self.params.initial_capacity,
                    temperature=25.0,
                    soc=0.5,
                    cycle_count=0,
                    calendar_age=0.0,
                )
                self.cells.append(cell)
        elif k < len(self.cells):
            # Remove worst cells
            self.cells.sort(key=lambda c: c.soh)
            self.cells = self.cells[len(self.cells) - k:]

    def set_alpha(self, alpha: float) -> None:
        """Set constraint generation rate (SEI growth rate)."""
        self._alpha = max(0, alpha)

    def set_d(self, d: float) -> None:
        """Set decay rate (calendar aging rate)."""
        self._d = max(0, d)

    def set_lambda(self, lambda_: float) -> None:
        """Set strictness (operating window tightness)."""
        self._lambda = np.clip(lambda_, 0, 1)

        # Adjust voltage window based on strictness
        v_center = (4.2 + 2.5) / 2
        v_half_range = (1 - self._lambda) * (4.2 - 2.5) / 2
        self.params.voltage_max = v_center + v_half_range
        self.params.voltage_min = v_center - v_half_range

    def apply_shock(self, shock: BatteryShock) -> None:
        """Apply external perturbation."""
        if not self.cells:
            raise RuntimeError("Engine not initialized")

        targets = self.cells if shock.target in (None, "all") else [self.cells[0]]

        for cell in targets:
            if shock.type == BatteryShockType.THERMAL:
                cell.temperature += shock.magnitude
            elif shock.type == BatteryShockType.MECHANICAL:
                cell.capacity *= (1 - shock.magnitude * 0.01)
                cell.soh = cell.capacity / self.params.initial_capacity
            elif shock.type == BatteryShockType.ELECTRICAL:
                cell.resistance *= (1 + shock.magnitude * 0.1)
            elif shock.type == BatteryShockType.ABUSE:
                cell.capacity *= (1 - shock.magnitude * 0.05)
                cell.resistance *= (1 + shock.magnitude * 0.2)
                cell.soh = cell.capacity / self.params.initial_capacity

    def apply_intervention(self, intervention: BatteryIntervention) -> None:
        """Apply intervention to system."""
        if not self.cells:
            raise RuntimeError("Engine not initialized")

        if intervention.type == BatteryInterventionType.VOLTAGE_WINDOW:
            new_max = intervention.parameters.get('voltage_max', self.params.voltage_max)
            new_min = intervention.parameters.get('voltage_min', self.params.voltage_min)
            self.params.voltage_max = new_max
            self.params.voltage_min = new_min
            self._lambda = self._calculate_strictness()

        elif intervention.type == BatteryInterventionType.TEMPERATURE:
            new_temp = intervention.parameters.get('temperature', 25.0)
            for cell in self.cells:
                cell.temperature = new_temp

        elif intervention.type == BatteryInterventionType.BALANCING:
            avg_soc = np.mean([c.soc for c in self.cells])
            for cell in self.cells:
                cell.soc = avg_soc

        elif intervention.type == BatteryInterventionType.CELL_REPLACEMENT:
            if self.cells:
                self.cells.sort(key=lambda c: c.soh)
                self.cells[0] = CellState(
                    capacity=self.params.initial_capacity,
                    resistance=self.params.initial_resistance,
                    soh=1.0,
                    sei_thickness=1.0,
                    li_inventory=self.params.initial_capacity,
                    temperature=25.0,
                    soc=0.5,
                    cycle_count=0,
                    calendar_age=0.0,
                )

        elif intervention.type == BatteryInterventionType.REGENERATION:
            rest_hours = intervention.parameters.get('duration', 24)
            for cell in self.cells:
                cell.temperature = 20.0
                cell.soc = 0.5
            original_alpha = self._alpha
            self._alpha *= 0.5
            for _ in range(int(rest_hours)):
                self.step(1.0)
            self._alpha = original_alpha

    # =========================================================================
    # VARIABLE MEASUREMENT (Outputs)
    # =========================================================================

    def get_k(self) -> int:
        """Get constraint count (number of cells)."""
        return len(self.cells)

    def get_rho(self) -> float:
        """Get cross-cell correlation."""
        if len(self.cells) <= 1:
            return 0.0

        soh_values = np.array([c.soh for c in self.cells])
        mean_soh = np.mean(soh_values)

        if mean_soh == 0:
            return 1.0

        cv = np.std(soh_values) / mean_soh
        rho = 1 - min(1, cv * 10)
        return max(0, min(1, rho))

    def get_k_eff(self) -> float:
        """Get effective constraint count: k / (1 + rho*(k-1))."""
        k = len(self.cells)
        rho = self.get_rho()

        if k <= 1:
            return float(k)

        return k / (1 + rho * (k - 1))

    def get_sigma(self) -> float:
        """Get sustainability (average SOH)."""
        if not self.cells:
            return 0.0
        return np.mean([c.soh for c in self.cells])

    def get_f(self) -> float:
        """Get compromise fraction (capacity fade)."""
        return 1 - self.get_sigma()

    def get_state(self) -> np.ndarray:
        """Get flattened state vector."""
        if not self.cells:
            return np.array([])

        state_list = []
        for cell in self.cells:
            state_list.extend([
                cell.capacity,
                cell.resistance,
                cell.soh,
                cell.sei_thickness,
                cell.temperature,
            ])
        return np.array(state_list)

    def is_collapsed(self) -> bool:
        """Check if system has collapsed."""
        return self._collapsed

    def get_collapse_time(self) -> Optional[float]:
        """Get time of collapse (if any)."""
        return self._collapse_time

    # =========================================================================
    # DATA EXPORT
    # =========================================================================

    def to_dataframe(self) -> pd.DataFrame:
        """Export simulation history as DataFrame."""
        return pd.DataFrame(self._history)

    def reset(self) -> None:
        """Reset engine to uninitialized state."""
        self.cells = []
        self._history = []
        self.time = 0.0
        self.step_count = 0
        self._collapsed = False
        self._collapse_time = None


def create_battery_engine(
    params: Optional[BatteryParams] = None,
    seed: Optional[int] = None,
) -> BatteryDegradationEngine:
    """Factory function to create a BatteryDegradationEngine instance."""
    return BatteryDegradationEngine(params=params, seed=seed)


__all__ = [
    'BatteryDegradationEngine',
    'BatteryParams',
    'CellState',
    'BatteryShock',
    'BatteryIntervention',
    'BatteryShockType',
    'BatteryInterventionType',
    'create_battery_engine',
]
