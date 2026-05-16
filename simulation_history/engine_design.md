# Institutional Collapse Simulation Engine: Architecture Design

This document specifies the architecture for an institutional collapse / state fragility simulation engine following the RATCHET framework interface.

---

## 1. Design Principles

### 1.1 Theory-Agnostic Simulation

The engine **does not assume** any particular relationship between variables. It:
- Exposes all RATCHET variables for manipulation
- Measures variables faithfully from domain data
- Does not hard-code expected behaviors
- Allows external analysis of variable relationships

### 1.2 Data-Driven Dynamics

The simulation is grounded in empirical data:
- Historical trajectories inform parameter ranges
- Collapse events are identified from event datasets
- Transition probabilities can be estimated from data

### 1.3 Interface Compliance

Follows the interface specified in `SIMULATION_REQUIREMENTS.md`:
- Standard `SimulationEngine` class structure
- Manipulable and measurable variable separation
- Time-series output with collapse detection

---

## 2. Class Architecture

### 2.1 Core Classes

```
InstitutionalCollapseEngine
    |
    +-- StateVector
    |       |-- k (constraint count)
    |       |-- rho (correlation)
    |       |-- sigma (sustainability)
    |       |-- f (compromise fraction)
    |       |-- lambda_ (strictness)
    |       +-- auxiliary variables
    |
    +-- ParameterSet
    |       |-- alpha (generation rate)
    |       |-- d (decay rate)
    |       +-- shock_sensitivity
    |
    +-- CollapseDetector
    |       |-- threshold_sigma
    |       |-- event_indicators
    |       +-- collapse_history
    |
    +-- DataLoader
    |       |-- QoGLoader
    |       |-- VDemLoader
    |       |-- EventLoader
    |       +-- WhoGovLoader
    |
    +-- InterventionHandler
            |-- reform_intervention
            |-- aid_intervention
            |-- sanction_intervention
            +-- external_shock
```

### 2.2 Module Structure

```
simulation_history/
    |-- engine.py              # Core InstitutionalCollapseEngine
    |-- state.py               # StateVector and state management
    |-- parameters.py          # ParameterSet and configuration
    |-- collapse.py            # CollapseDetector and event handling
    |-- loaders/
    |       |-- __init__.py
    |       |-- qog_loader.py  # QoG Standard Dataset loader
    |       |-- vdem_loader.py # V-Dem specific loader
    |       |-- event_loader.py# PITF, UCDP event loader
    |       +-- whogov_loader.py # Elite composition loader
    |
    |-- interventions.py       # Shock and intervention types
    |-- types.py               # Domain-specific type definitions
    |-- utils.py               # Helper functions
    |
    +-- examples/
            |-- basic_simulation.py
            |-- historical_calibration.py
            +-- intervention_analysis.py
```

---

## 3. Core Engine Implementation

### 3.1 InstitutionalCollapseEngine Class

```python
"""
Institutional Collapse / State Fragility Simulation Engine

Simulates institutional dynamics without assuming any particular
relationship between variables. Exposes RATCHET framework variables
for manipulation and measurement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from enum import Enum
import numpy as np
import pandas as pd


class RegimeType(Enum):
    """Regime classification following Polity/GWF typology."""
    DEMOCRACY = "democracy"
    ANOCRACY = "anocracy"
    AUTOCRACY = "autocracy"
    FAILED = "failed"


class InterventionType(Enum):
    """Types of external interventions."""
    REFORM = "reform"               # Institutional reform
    AID = "aid"                     # Foreign aid / assistance
    SANCTION = "sanction"           # Economic sanctions
    MILITARY = "military"           # Military intervention
    DIPLOMATIC = "diplomatic"       # Diplomatic pressure
    NONE = "none"


@dataclass
class Shock:
    """External shock to the institutional system."""
    type: str                       # 'economic', 'conflict', 'natural', 'political'
    magnitude: float                # Shock size (normalized)
    target_variable: str            # Which variable is affected
    duration: int = 1               # Duration in time steps

    def apply(self, state: 'StateVector') -> 'StateVector':
        """Apply shock to state vector."""
        new_state = state.copy()
        if self.target_variable == 'sigma':
            new_state.sigma = max(0, state.sigma - self.magnitude)
        elif self.target_variable == 'k':
            new_state.k = max(0, state.k - self.magnitude)
        elif self.target_variable == 'lambda':
            new_state.lambda_ = max(0, state.lambda_ - self.magnitude)
        return new_state


@dataclass
class Intervention:
    """Policy or external intervention."""
    type: InterventionType
    intensity: float               # Intervention strength (0-1)
    target_variable: str           # Primary target
    delay: int = 0                 # Delay before effect
    duration: int = 1              # Duration of effect

    def apply(self, state: 'StateVector') -> 'StateVector':
        """Apply intervention to state vector."""
        new_state = state.copy()
        effect = self.intensity

        if self.type == InterventionType.REFORM:
            if self.target_variable == 'k':
                new_state.k = min(1.0, state.k + effect * 0.1)
            elif self.target_variable == 'lambda':
                new_state.lambda_ = min(1.0, state.lambda_ + effect * 0.1)

        elif self.type == InterventionType.AID:
            if self.target_variable == 'sigma':
                new_state.sigma = min(1.0, state.sigma + effect * 0.05)

        elif self.type == InterventionType.SANCTION:
            new_state.sigma = max(0, state.sigma - effect * 0.1)

        return new_state


@dataclass
class StateVector:
    """
    State vector for institutional system.

    Contains both measurable outputs and derived quantities.
    All values normalized to [0, 1] unless otherwise specified.
    """
    # Measurable variables (outputs)
    k: float                        # Constraint count (normalized 0-1)
    rho: float                      # Correlation/coupling (0-1)
    sigma: float                    # Sustainability (0-1)
    f: float                        # Compromise fraction (0-1)
    lambda_: float                  # Strictness (0-1)

    # Derived quantities
    k_eff: float = field(init=False)  # Effective constraint count

    # Auxiliary state
    time: float = 0.0
    country_code: Optional[str] = None
    year: Optional[int] = None

    def __post_init__(self):
        """Compute derived quantities."""
        self._compute_k_eff()

    def _compute_k_eff(self):
        """Compute effective constraint count: k / (1 + rho*(k-1))."""
        if self.k <= 0:
            self.k_eff = 0.0
        elif self.rho >= 1.0:
            self.k_eff = 1.0  # Fully correlated = single constraint
        else:
            # Normalize k to count-like value for formula
            k_count = self.k * 10  # Scale to ~0-10 range
            denominator = 1 + self.rho * (k_count - 1)
            self.k_eff = k_count / max(denominator, 0.01)

    def copy(self) -> 'StateVector':
        """Create a copy of the state vector."""
        return StateVector(
            k=self.k,
            rho=self.rho,
            sigma=self.sigma,
            f=self.f,
            lambda_=self.lambda_,
            time=self.time,
            country_code=self.country_code,
            year=self.year
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'k': self.k,
            'rho': self.rho,
            'sigma': self.sigma,
            'f': self.f,
            'lambda': self.lambda_,
            'k_eff': self.k_eff,
            'time': self.time,
            'country_code': self.country_code,
            'year': self.year
        }


@dataclass
class EngineConfig:
    """Configuration for InstitutionalCollapseEngine."""
    # Manipulable parameters
    alpha: float = 0.02            # Constraint generation rate
    d: float = 0.03                # Decay rate

    # Collapse thresholds
    collapse_threshold_sigma: float = 0.2
    collapse_threshold_f: float = 0.8

    # Simulation settings
    dt: float = 1.0                # Time step (years)
    noise_sigma: float = 0.01      # Process noise

    # Data sources
    data_source: str = "qog"       # 'qog', 'vdem', 'synthetic'
    data_path: Optional[str] = None

    # Random seed
    seed: Optional[int] = None


class TimeSeries:
    """Container for simulation time series data."""

    def __init__(self):
        self.times: List[float] = []
        self.states: List[StateVector] = []
        self.events: List[Dict] = []
        self.interventions: List[Intervention] = []
        self.shocks: List[Shock] = []

    def append(self, state: StateVector, event: Optional[Dict] = None):
        """Add a state to the time series."""
        self.times.append(state.time)
        self.states.append(state)
        if event:
            self.events.append(event)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        records = [s.to_dict() for s in self.states]
        df = pd.DataFrame(records)
        return df

    def get_variable(self, name: str) -> np.ndarray:
        """Extract a single variable as numpy array."""
        return np.array([getattr(s, name) for s in self.states])


class InstitutionalCollapseEngine:
    """
    Simulation engine for institutional collapse / state fragility.

    Implements the RATCHET framework interface for domain simulation.
    Theory-agnostic: exposes variables for manipulation and measurement
    without assuming relationships between them.

    Usage:
        config = EngineConfig(alpha=0.02, d=0.03)
        engine = InstitutionalCollapseEngine(config)

        # Initialize from data or manually
        engine.initialize_from_country('USA', 2000)
        # or
        engine.set_state(StateVector(k=0.8, rho=0.3, sigma=0.9, f=0.2, lambda_=0.85))

        # Run simulation
        ts = engine.run(duration=50, dt=1.0)

        # Analyze results
        df = ts.to_dataframe()
    """

    def __init__(self, config: EngineConfig):
        """Initialize the engine with configuration."""
        self.config = config
        self.state: Optional[StateVector] = None
        self.history: TimeSeries = TimeSeries()
        self._collapsed: bool = False
        self._collapse_time: Optional[float] = None

        # Set random seed
        if config.seed is not None:
            np.random.seed(config.seed)

        # Initialize data loader if needed
        self._loader = None
        if config.data_source != "synthetic":
            self._init_loader()

    def _init_loader(self):
        """Initialize data loader based on config."""
        # Placeholder - implemented in loaders module
        pass

    # =========================================================================
    # VARIABLE MANIPULATION (Inputs/Controls)
    # =========================================================================

    def set_k(self, k: float) -> None:
        """Set constraint count."""
        if self.state is None:
            raise ValueError("State not initialized. Call initialize_* first.")
        self.state.k = np.clip(k, 0, 1)
        self.state._compute_k_eff()

    def set_alpha(self, alpha: float) -> None:
        """Set constraint generation rate."""
        self.config.alpha = max(0, alpha)

    def set_d(self, d: float) -> None:
        """Set decay rate."""
        self.config.d = max(0, d)

    def set_lambda(self, lambda_: float) -> None:
        """Set strictness."""
        if self.state is None:
            raise ValueError("State not initialized.")
        self.state.lambda_ = np.clip(lambda_, 0, 1)

    def set_state(self, state: StateVector) -> None:
        """Set the full state vector."""
        self.state = state
        self._collapsed = False
        self._collapse_time = None

    def apply_shock(self, shock: Shock) -> None:
        """Apply an external shock to the system."""
        if self.state is None:
            raise ValueError("State not initialized.")
        self.state = shock.apply(self.state)
        self.history.shocks.append(shock)

    def apply_intervention(self, intervention: Intervention) -> None:
        """Apply a policy intervention."""
        if self.state is None:
            raise ValueError("State not initialized.")
        self.state = intervention.apply(self.state)
        self.history.interventions.append(intervention)

    # =========================================================================
    # VARIABLE MEASUREMENT (Outputs/Observations)
    # =========================================================================

    def get_rho(self) -> float:
        """Get current correlation/coupling."""
        return self.state.rho if self.state else 0.0

    def get_k_eff(self) -> float:
        """Get effective constraint count."""
        return self.state.k_eff if self.state else 0.0

    def get_sigma(self) -> float:
        """Get sustainability metric."""
        return self.state.sigma if self.state else 0.0

    def get_f(self) -> float:
        """Get compromise fraction."""
        return self.state.f if self.state else 0.0

    def get_state(self) -> np.ndarray:
        """Get full state as numpy array."""
        if self.state is None:
            return np.zeros(5)
        return np.array([
            self.state.k,
            self.state.rho,
            self.state.sigma,
            self.state.f,
            self.state.lambda_
        ])

    def is_collapsed(self) -> bool:
        """Check if system has collapsed."""
        return self._collapsed

    def get_collapse_time(self) -> Optional[float]:
        """Get time of collapse (if any)."""
        return self._collapse_time

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def initialize_from_country(
        self,
        country_code: str,
        year: int,
        data_path: Optional[str] = None
    ) -> None:
        """
        Initialize state from historical data for a specific country-year.

        Args:
            country_code: ISO or COW country code
            year: Starting year
            data_path: Path to data file (optional)
        """
        if self._loader is None:
            raise ValueError("Data loader not initialized. Use synthetic data or configure loader.")

        # Load data point
        data = self._loader.get_country_year(country_code, year)

        # Map to state vector using domain mapping
        self.state = StateVector(
            k=data.get('v2x_liberal', 0.5),
            rho=data.get('elite_homogeneity', 0.3),
            sigma=data.get('wgi_ps_normalized', 0.5),
            f=data.get('v2x_corr', 0.3),
            lambda_=data.get('v2x_rule', 0.5),
            time=0.0,
            country_code=country_code,
            year=year
        )

    def initialize_synthetic(
        self,
        regime_type: RegimeType = RegimeType.DEMOCRACY,
        noise: bool = True
    ) -> None:
        """
        Initialize with synthetic state based on regime archetype.

        Args:
            regime_type: Type of regime to simulate
            noise: Add random variation to parameters
        """
        # Archetype parameters (theory-agnostic defaults)
        archetypes = {
            RegimeType.DEMOCRACY: {'k': 0.85, 'rho': 0.25, 'sigma': 0.80, 'f': 0.20, 'lambda_': 0.85},
            RegimeType.ANOCRACY: {'k': 0.50, 'rho': 0.50, 'sigma': 0.50, 'f': 0.50, 'lambda_': 0.50},
            RegimeType.AUTOCRACY: {'k': 0.20, 'rho': 0.80, 'sigma': 0.40, 'f': 0.70, 'lambda_': 0.60},
            RegimeType.FAILED: {'k': 0.10, 'rho': 0.90, 'sigma': 0.15, 'f': 0.85, 'lambda_': 0.15},
        }

        params = archetypes[regime_type].copy()

        if noise:
            for key in params:
                params[key] += np.random.normal(0, self.config.noise_sigma)
                params[key] = np.clip(params[key], 0, 1)

        self.state = StateVector(**params)

    # =========================================================================
    # SIMULATION
    # =========================================================================

    def step(self, dt: float = 1.0) -> None:
        """
        Advance simulation by one time step.

        The step function updates state variables based on their dynamics.
        This is a minimal update model - the engine does NOT assume any
        particular relationship between variables.

        Args:
            dt: Time step size (in years)
        """
        if self.state is None:
            raise ValueError("State not initialized.")

        if self._collapsed:
            return  # No dynamics after collapse

        # Store current state in history
        self.history.append(self.state.copy())

        # Minimal dynamics: decay and generation
        # These are domain-realistic processes without theoretical claims

        # Sustainability decay (resource depletion without investment)
        decay_amount = self.config.d * dt
        noise = np.random.normal(0, self.config.noise_sigma)
        self.state.sigma = np.clip(
            self.state.sigma - decay_amount + noise,
            0, 1
        )

        # Constraint generation/erosion based on alpha
        constraint_change = self.config.alpha * dt
        self.state.k = np.clip(
            self.state.k + constraint_change + np.random.normal(0, self.config.noise_sigma),
            0, 1
        )

        # Update derived quantities
        self.state._compute_k_eff()

        # Update time
        self.state.time += dt
        if self.state.year is not None:
            self.state.year += int(dt)

        # Check for collapse
        self._check_collapse()

    def _check_collapse(self) -> None:
        """Check if collapse threshold has been crossed."""
        if self._collapsed:
            return

        # Collapse if sustainability too low OR compromise too high
        if (self.state.sigma < self.config.collapse_threshold_sigma or
            self.state.f > self.config.collapse_threshold_f):
            self._collapsed = True
            self._collapse_time = self.state.time

            # Record collapse event
            self.history.events.append({
                'type': 'collapse',
                'time': self.state.time,
                'sigma': self.state.sigma,
                'f': self.state.f
            })

    def run(self, duration: float, dt: Optional[float] = None) -> TimeSeries:
        """
        Run simulation for specified duration.

        Args:
            duration: Total simulation time (years)
            dt: Time step (defaults to config.dt)

        Returns:
            TimeSeries object with full history
        """
        if dt is None:
            dt = self.config.dt

        n_steps = int(duration / dt)

        for _ in range(n_steps):
            self.step(dt)
            if self._collapsed:
                break

        return self.history

    def run_until_collapse(
        self,
        max_duration: float = 100,
        dt: Optional[float] = None
    ) -> TimeSeries:
        """
        Run simulation until collapse occurs or max duration reached.

        Args:
            max_duration: Maximum simulation time
            dt: Time step

        Returns:
            TimeSeries object
        """
        return self.run(max_duration, dt)

    # =========================================================================
    # DATA EXPORT
    # =========================================================================

    def to_dataframe(self) -> pd.DataFrame:
        """Export simulation history to pandas DataFrame."""
        return self.history.to_dataframe()

    def reset(self) -> None:
        """Reset engine to initial state."""
        self.state = None
        self.history = TimeSeries()
        self._collapsed = False
        self._collapse_time = None


# =============================================================================
# BATCH SIMULATION UTILITIES
# =============================================================================

def run_parameter_sweep(
    base_config: EngineConfig,
    parameter_name: str,
    parameter_values: List[float],
    initial_state: StateVector,
    duration: float = 50,
    n_runs: int = 10
) -> pd.DataFrame:
    """
    Run parameter sweep across multiple values.

    Args:
        base_config: Base configuration
        parameter_name: Name of parameter to sweep ('alpha', 'd', etc.)
        parameter_values: Values to test
        initial_state: Initial state for each run
        duration: Simulation duration
        n_runs: Number of runs per parameter value (for noise averaging)

    Returns:
        DataFrame with results across parameter values
    """
    results = []

    for param_value in parameter_values:
        for run_idx in range(n_runs):
            config = EngineConfig(**{
                **vars(base_config),
                parameter_name: param_value,
                'seed': (base_config.seed or 0) + run_idx
            })

            engine = InstitutionalCollapseEngine(config)
            engine.set_state(initial_state.copy())

            ts = engine.run(duration)

            results.append({
                'parameter': parameter_name,
                'value': param_value,
                'run': run_idx,
                'collapsed': engine.is_collapsed(),
                't_collapse': engine.get_collapse_time(),
                'final_sigma': engine.get_sigma(),
                'final_f': engine.get_f(),
                'final_k_eff': engine.get_k_eff()
            })

    return pd.DataFrame(results)


def run_historical_comparison(
    engine: InstitutionalCollapseEngine,
    country_code: str,
    start_year: int,
    end_year: int,
    data_loader
) -> pd.DataFrame:
    """
    Run simulation and compare to historical data.

    Args:
        engine: Configured engine
        country_code: Country to simulate
        start_year: Start year
        end_year: End year
        data_loader: Data loader instance

    Returns:
        DataFrame comparing simulated vs actual trajectories
    """
    # Initialize from historical start point
    engine.initialize_from_country(country_code, start_year)

    # Run simulation
    duration = end_year - start_year
    ts = engine.run(duration)

    simulated = ts.to_dataframe()
    simulated['source'] = 'simulated'

    # Load actual historical data
    actual = data_loader.get_country_series(country_code, start_year, end_year)
    actual['source'] = 'actual'

    # Combine for comparison
    combined = pd.concat([simulated, actual])

    return combined
```

---

## 4. Data Loader Interface

### 4.1 Base Loader Class

```python
"""
Data loaders for institutional collapse simulation.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, List
import pandas as pd


class BaseLoader(ABC):
    """Abstract base class for data loaders."""

    @abstractmethod
    def load(self, path: str) -> pd.DataFrame:
        """Load dataset from path."""
        pass

    @abstractmethod
    def get_country_year(self, country_code: str, year: int) -> Dict:
        """Get single country-year observation."""
        pass

    @abstractmethod
    def get_country_series(
        self,
        country_code: str,
        start_year: int,
        end_year: int
    ) -> pd.DataFrame:
        """Get time series for a country."""
        pass

    @abstractmethod
    def map_to_ratchet_variables(self, row: pd.Series) -> Dict:
        """Map raw data to RATCHET variables."""
        pass


class QoGLoader(BaseLoader):
    """
    Loader for Quality of Government Standard Dataset.

    The QoG dataset is the recommended primary data source as it
    integrates multiple governance indicators in a unified format.
    """

    # Variable mapping from QoG columns to RATCHET variables
    VARIABLE_MAP = {
        # Constraint count (k) - V-Dem liberal component
        'vdem_v2x_liberal': 'k',

        # Correlation (rho) - derived from multiple sources
        # Will be computed from elite/party concentration

        # Sustainability (sigma) - WGI political stability
        'wbgi_pse': 'sigma_raw',  # Needs normalization

        # Compromise fraction (f) - V-Dem corruption
        'vdem_v2x_corr': 'f',

        # Strictness (lambda) - V-Dem rule of law
        'vdem_v2x_rule': 'lambda_',

        # Auxiliary variables for rho computation
        'wbgi_cce': 'corruption_control',  # WGI control of corruption
        'p_polity2': 'polity_score',       # Polity score
    }

    def __init__(self, path: Optional[str] = None):
        self.data: Optional[pd.DataFrame] = None
        if path:
            self.load(path)

    def load(self, path: str) -> pd.DataFrame:
        """Load QoG Standard dataset."""
        # Detect format from extension
        if path.endswith('.csv'):
            self.data = pd.read_csv(path)
        elif path.endswith('.dta'):
            self.data = pd.read_stata(path)
        else:
            raise ValueError(f"Unsupported format: {path}")

        return self.data

    def get_country_year(self, country_code: str, year: int) -> Dict:
        """Get mapped variables for country-year."""
        if self.data is None:
            raise ValueError("Data not loaded.")

        # Support both COW and ISO codes
        mask = (
            ((self.data['ccodecow'] == country_code) |
             (self.data['ccodealp'] == country_code)) &
            (self.data['year'] == year)
        )

        row = self.data[mask]

        if len(row) == 0:
            raise ValueError(f"No data for {country_code} in {year}")

        return self.map_to_ratchet_variables(row.iloc[0])

    def get_country_series(
        self,
        country_code: str,
        start_year: int,
        end_year: int
    ) -> pd.DataFrame:
        """Get time series for country."""
        if self.data is None:
            raise ValueError("Data not loaded.")

        mask = (
            ((self.data['ccodecow'] == country_code) |
             (self.data['ccodealp'] == country_code)) &
            (self.data['year'] >= start_year) &
            (self.data['year'] <= end_year)
        )

        subset = self.data[mask].copy()

        # Map each row
        mapped = subset.apply(self.map_to_ratchet_variables, axis=1)

        return pd.DataFrame(list(mapped))

    def map_to_ratchet_variables(self, row: pd.Series) -> Dict:
        """Map QoG row to RATCHET variables."""
        result = {
            'year': row.get('year'),
            'country_code': row.get('ccodealp'),
        }

        # Direct mappings
        for qog_var, ratchet_var in self.VARIABLE_MAP.items():
            if qog_var in row and pd.notna(row[qog_var]):
                result[ratchet_var] = row[qog_var]

        # Normalize WGI political stability (-2.5 to 2.5) -> (0 to 1)
        if 'sigma_raw' in result:
            result['sigma'] = (result['sigma_raw'] + 2.5) / 5.0
            result['sigma'] = max(0, min(1, result['sigma']))

        # Compute rho (correlation/coupling)
        # Using party concentration and corruption as proxies
        result['rho'] = self._compute_rho(row)

        # Provide defaults for missing values
        defaults = {'k': 0.5, 'rho': 0.5, 'sigma': 0.5, 'f': 0.5, 'lambda_': 0.5}
        for key, default in defaults.items():
            if key not in result or pd.isna(result.get(key)):
                result[key] = default

        return result

    def _compute_rho(self, row: pd.Series) -> float:
        """
        Compute correlation/coupling from available indicators.

        Uses inverse of effective number of parties and corruption
        as proxies for elite concentration.
        """
        rho_components = []

        # Party concentration (inverse of effective parties)
        if 'dpi_numul' in row and pd.notna(row['dpi_numul']):
            # Fewer parties = higher concentration
            enp = max(1, row['dpi_numul'])
            party_concentration = 1 - (1 / enp)
            rho_components.append(party_concentration)

        # Corruption as patronage proxy
        if 'vdem_v2x_corr' in row and pd.notna(row['vdem_v2x_corr']):
            rho_components.append(row['vdem_v2x_corr'])

        if rho_components:
            return sum(rho_components) / len(rho_components)
        return 0.5  # Default


class VDemLoader(BaseLoader):
    """
    Loader for V-Dem dataset.

    Provides more detailed democracy and governance indicators
    than the QoG compilation.
    """

    def __init__(self, path: Optional[str] = None):
        self.data: Optional[pd.DataFrame] = None
        if path:
            self.load(path)

    def load(self, path: str) -> pd.DataFrame:
        """Load V-Dem dataset."""
        self.data = pd.read_csv(path)
        return self.data

    def get_country_year(self, country_code: str, year: int) -> Dict:
        """Get V-Dem variables for country-year."""
        if self.data is None:
            raise ValueError("Data not loaded.")

        mask = (
            (self.data['country_text_id'] == country_code) &
            (self.data['year'] == year)
        )

        row = self.data[mask]

        if len(row) == 0:
            raise ValueError(f"No data for {country_code} in {year}")

        return self.map_to_ratchet_variables(row.iloc[0])

    def get_country_series(
        self,
        country_code: str,
        start_year: int,
        end_year: int
    ) -> pd.DataFrame:
        """Get V-Dem time series."""
        if self.data is None:
            raise ValueError("Data not loaded.")

        mask = (
            (self.data['country_text_id'] == country_code) &
            (self.data['year'] >= start_year) &
            (self.data['year'] <= end_year)
        )

        subset = self.data[mask]
        return pd.DataFrame([self.map_to_ratchet_variables(row) for _, row in subset.iterrows()])

    def map_to_ratchet_variables(self, row: pd.Series) -> Dict:
        """Map V-Dem row to RATCHET variables."""
        return {
            'year': row.get('year'),
            'country_code': row.get('country_text_id'),
            'k': row.get('v2x_liberal', 0.5),
            'sigma': row.get('v2x_polyarchy', 0.5),  # Electoral democracy as sustainability proxy
            'f': row.get('v2x_corr', 0.5),
            'lambda_': row.get('v2x_rule', 0.5),
            'rho': row.get('v2x_neopat', 0.5),  # Neopatrimonialism as coupling
        }


class EventLoader(BaseLoader):
    """
    Loader for collapse event datasets (PITF, UCDP).

    Provides discrete collapse event identification.
    """

    def __init__(self):
        self.pitf_data: Optional[pd.DataFrame] = None
        self.ucdp_data: Optional[pd.DataFrame] = None

    def load_pitf(self, path: str) -> pd.DataFrame:
        """Load PITF state failure dataset."""
        self.pitf_data = pd.read_excel(path)
        return self.pitf_data

    def load_ucdp(self, path: str) -> pd.DataFrame:
        """Load UCDP conflict dataset."""
        self.ucdp_data = pd.read_csv(path)
        return self.ucdp_data

    def get_collapse_events(
        self,
        country_code: str,
        start_year: int,
        end_year: int
    ) -> List[Dict]:
        """Get list of collapse events for country."""
        events = []

        # Check PITF
        if self.pitf_data is not None:
            pitf_events = self._get_pitf_events(country_code, start_year, end_year)
            events.extend(pitf_events)

        # Check UCDP
        if self.ucdp_data is not None:
            ucdp_events = self._get_ucdp_events(country_code, start_year, end_year)
            events.extend(ucdp_events)

        return events

    def _get_pitf_events(
        self,
        country_code: str,
        start_year: int,
        end_year: int
    ) -> List[Dict]:
        """Extract PITF events."""
        # Implementation depends on PITF format
        events = []
        # ... parsing logic
        return events

    def _get_ucdp_events(
        self,
        country_code: str,
        start_year: int,
        end_year: int
    ) -> List[Dict]:
        """Extract UCDP civil war onsets."""
        events = []
        # ... parsing logic
        return events

    def load(self, path: str) -> pd.DataFrame:
        """Generic load - route to appropriate loader."""
        raise NotImplementedError("Use load_pitf or load_ucdp")

    def get_country_year(self, country_code: str, year: int) -> Dict:
        """Check if collapse event in this year."""
        events = self.get_collapse_events(country_code, year, year)
        return {'has_collapse_event': len(events) > 0, 'events': events}

    def get_country_series(
        self,
        country_code: str,
        start_year: int,
        end_year: int
    ) -> pd.DataFrame:
        """Get event indicators for time series."""
        records = []
        for year in range(start_year, end_year + 1):
            cy = self.get_country_year(country_code, year)
            cy['year'] = year
            records.append(cy)
        return pd.DataFrame(records)

    def map_to_ratchet_variables(self, row: pd.Series) -> Dict:
        """Map event to RATCHET collapse indicator."""
        return {'collapsed': row.get('has_collapse_event', False)}
```

---

## 5. Type Definitions

### 5.1 Domain-Specific Types

```python
"""
Type definitions for Institutional Collapse simulation engine.
"""

from typing import Annotated, Optional
from pydantic import BaseModel, Field
from enum import Enum


# =============================================================================
# Refinement Types
# =============================================================================

# Normalized values (0-1)
UnitInterval = Annotated[float, Field(ge=0, le=1)]

# Non-negative float
NonNegativeFloat = Annotated[float, Field(ge=0)]

# Year (1800-2100)
Year = Annotated[int, Field(ge=1800, le=2100)]


# =============================================================================
# Enumerations
# =============================================================================

class RegimeType(str, Enum):
    """Regime classification."""
    DEMOCRACY = "democracy"
    ANOCRACY = "anocracy"
    AUTOCRACY = "autocracy"
    FAILED_STATE = "failed_state"
    TRANSITIONAL = "transitional"


class CollapseType(str, Enum):
    """Types of state collapse/failure."""
    REGIME_CHANGE = "regime_change"
    CIVIL_WAR = "civil_war"
    STATE_FAILURE = "state_failure"
    CONSTITUTIONAL_CRISIS = "constitutional_crisis"
    DEMOCRATIC_BREAKDOWN = "democratic_breakdown"
    GENOCIDE = "genocide"


class ShockType(str, Enum):
    """Types of external shocks."""
    ECONOMIC = "economic"
    CONFLICT = "conflict"
    NATURAL = "natural"
    POLITICAL = "political"
    PANDEMIC = "pandemic"


class InterventionType(str, Enum):
    """Types of interventions."""
    REFORM = "reform"
    AID = "aid"
    SANCTION = "sanction"
    MILITARY = "military"
    DIPLOMATIC = "diplomatic"


# =============================================================================
# Parameter Models
# =============================================================================

class InstitutionalParams(BaseModel):
    """
    Parameters for Institutional Collapse Engine.

    Follows RATCHET framework interface with domain-specific validation.
    """
    engine: str = "institutional_collapse"

    # Manipulable parameters
    alpha: NonNegativeFloat = Field(
        default=0.02,
        description="Constraint generation rate (new constraints per year)"
    )
    d: NonNegativeFloat = Field(
        default=0.03,
        description="Natural decay rate (fraction per year)"
    )

    # Collapse thresholds
    collapse_threshold_sigma: UnitInterval = Field(
        default=0.2,
        description="Sustainability threshold for collapse detection"
    )
    collapse_threshold_f: UnitInterval = Field(
        default=0.8,
        description="Compromise fraction threshold for collapse"
    )

    # Simulation settings
    dt: float = Field(
        default=1.0,
        gt=0,
        description="Time step in years"
    )
    noise_sigma: NonNegativeFloat = Field(
        default=0.01,
        description="Process noise standard deviation"
    )

    # Data configuration
    data_source: str = Field(
        default="synthetic",
        description="Data source: 'qog', 'vdem', 'synthetic'"
    )

    # Random seed
    seed: Optional[int] = None


class CollapseEvent(BaseModel):
    """Recorded collapse event."""
    country_code: str
    year: Year
    type: CollapseType
    magnitude: Optional[UnitInterval] = None
    duration_years: Optional[int] = None
    source: str  # 'PITF', 'UCDP', 'V-Dem', etc.


# =============================================================================
# Result Types
# =============================================================================

class SimulationResult(BaseModel):
    """Results from a simulation run."""
    collapsed: bool
    collapse_time: Optional[float] = None
    final_state: dict
    trajectory_length: int
    warnings: list = []
```

---

## 6. Example Usage

### 6.1 Basic Simulation

```python
from simulation_history.engine import (
    InstitutionalCollapseEngine,
    EngineConfig,
    StateVector,
    RegimeType,
    Shock,
    Intervention,
    InterventionType
)

# Create configuration
config = EngineConfig(
    alpha=0.02,      # 2% annual constraint generation
    d=0.03,          # 3% annual decay
    collapse_threshold_sigma=0.2,
    seed=42
)

# Initialize engine
engine = InstitutionalCollapseEngine(config)

# Option 1: Initialize from regime archetype
engine.initialize_synthetic(RegimeType.ANOCRACY)

# Option 2: Initialize manually
# engine.set_state(StateVector(
#     k=0.6, rho=0.4, sigma=0.5, f=0.4, lambda_=0.5
# ))

# Run simulation
ts = engine.run(duration=50)

# Check results
print(f"Collapsed: {engine.is_collapsed()}")
print(f"Collapse time: {engine.get_collapse_time()}")
print(f"Final sustainability: {engine.get_sigma():.3f}")

# Export to DataFrame
df = ts.to_dataframe()
print(df.head())
```

### 6.2 With Shocks and Interventions

```python
# Initialize stable democracy
engine.initialize_synthetic(RegimeType.DEMOCRACY)

# Run for 10 years
engine.run(duration=10)

# Apply economic shock
shock = Shock(
    type='economic',
    magnitude=0.3,
    target_variable='sigma',
    duration=2
)
engine.apply_shock(shock)

# Run 5 more years
engine.run(duration=5)

# Apply reform intervention
reform = Intervention(
    type=InterventionType.REFORM,
    intensity=0.5,
    target_variable='lambda'
)
engine.apply_intervention(reform)

# Continue simulation
ts = engine.run(duration=35)
```

### 6.3 Parameter Sensitivity Analysis

```python
from simulation_history.engine import run_parameter_sweep

# Base configuration
base_config = EngineConfig(alpha=0.02, d=0.03, seed=42)

# Initial state
initial_state = StateVector(k=0.5, rho=0.5, sigma=0.5, f=0.3, lambda_=0.5)

# Sweep decay rate
results = run_parameter_sweep(
    base_config=base_config,
    parameter_name='d',
    parameter_values=[0.01, 0.02, 0.03, 0.05, 0.10],
    initial_state=initial_state,
    duration=50,
    n_runs=20
)

# Analyze
collapse_by_d = results.groupby('value')['collapsed'].mean()
print("Collapse probability by decay rate:")
print(collapse_by_d)
```

---

## 7. Integration with RATCHET Framework

### 7.1 Standard Interface Compliance

The engine implements the standard RATCHET simulation interface:

| Method | Implementation |
|--------|----------------|
| `__init__(config)` | `InstitutionalCollapseEngine(EngineConfig)` |
| `step(dt)` | `engine.step(dt)` |
| `run(duration, dt)` | `engine.run(duration, dt)` |
| `set_k(k)` | `engine.set_k(k)` |
| `set_alpha(alpha)` | `engine.set_alpha(alpha)` |
| `set_d(d)` | `engine.set_d(d)` |
| `set_lambda(lambda_)` | `engine.set_lambda(lambda_)` |
| `apply_shock(shock)` | `engine.apply_shock(Shock)` |
| `apply_intervention(intervention)` | `engine.apply_intervention(Intervention)` |
| `get_rho()` | `engine.get_rho()` |
| `get_k_eff()` | `engine.get_k_eff()` |
| `get_sigma()` | `engine.get_sigma()` |
| `get_f()` | `engine.get_f()` |
| `get_state()` | `engine.get_state()` |
| `is_collapsed()` | `engine.is_collapsed()` |
| `to_dataframe()` | `engine.to_dataframe()` |

### 7.2 Engine Registration

```python
# In ratchet/engines/__init__.py (suggested addition)

from simulation_history.engine import InstitutionalCollapseEngine
from simulation_history.types import InstitutionalParams

ENGINE_REGISTRY = {
    'geometric': GeometricEngine,
    'complexity': ComplexityEngine,
    'detection': DetectionEngine,
    'federation': FederationEngine,
    'institutional_collapse': InstitutionalCollapseEngine,  # New
}

PARAM_REGISTRY = {
    'geometric': GeometricParams,
    'complexity': ComplexityParams,
    'detection': DetectionParams,
    'federation': FederationParams,
    'institutional_collapse': InstitutionalParams,  # New
}
```

---

## 8. Testing Strategy

### 8.1 Unit Tests

```python
# tests/test_institutional_engine.py

import pytest
from simulation_history.engine import (
    InstitutionalCollapseEngine,
    EngineConfig,
    StateVector,
    RegimeType
)


def test_initialization():
    """Test engine initialization."""
    config = EngineConfig(seed=42)
    engine = InstitutionalCollapseEngine(config)
    assert engine.state is None

    engine.initialize_synthetic(RegimeType.DEMOCRACY)
    assert engine.state is not None
    assert 0 <= engine.state.k <= 1


def test_step():
    """Test single simulation step."""
    config = EngineConfig(d=0.05, seed=42)
    engine = InstitutionalCollapseEngine(config)
    engine.set_state(StateVector(k=0.5, rho=0.3, sigma=0.8, f=0.2, lambda_=0.7))

    initial_sigma = engine.get_sigma()
    engine.step(dt=1.0)

    # Sigma should decay
    assert engine.get_sigma() < initial_sigma


def test_collapse_detection():
    """Test collapse is detected when threshold crossed."""
    config = EngineConfig(collapse_threshold_sigma=0.3, seed=42)
    engine = InstitutionalCollapseEngine(config)

    # Set state below threshold
    engine.set_state(StateVector(k=0.5, rho=0.3, sigma=0.2, f=0.5, lambda_=0.5))
    engine.step(1.0)

    assert engine.is_collapsed()


def test_k_eff_computation():
    """Test effective constraint count formula."""
    state = StateVector(k=0.5, rho=0.0, sigma=0.5, f=0.3, lambda_=0.5)
    # With rho=0, k_eff should equal scaled k

    state2 = StateVector(k=0.5, rho=0.9, sigma=0.5, f=0.3, lambda_=0.5)
    # With high rho, k_eff should be lower

    assert state2.k_eff < state.k_eff
```

### 8.2 Integration Tests

```python
def test_historical_calibration():
    """Test loading and simulation from historical data."""
    # Requires QoG dataset
    config = EngineConfig(data_source='qog', data_path='path/to/qog.csv')
    engine = InstitutionalCollapseEngine(config)

    engine.initialize_from_country('USA', 1990)
    ts = engine.run(duration=30)

    assert len(ts.states) > 0
    assert ts.states[0].country_code == 'USA'
```

---

## 9. Future Extensions

### 9.1 Agent-Based Extension

Extend to multi-agent simulation with:
- Elite agents with individual preferences
- Coalition formation dynamics
- Strategic intervention responses

### 9.2 Network Extension

Add explicit network structure:
- Elite network graph
- Institutional coupling network
- Information flow networks

### 9.3 Spatial Extension

Geographic considerations:
- Center-periphery dynamics
- Regional contagion effects
- Border effects

### 9.4 Ensemble Methods

For uncertainty quantification:
- Monte Carlo ensemble runs
- Bayesian parameter estimation
- Scenario analysis tools
