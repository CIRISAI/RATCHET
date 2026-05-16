# Microbiome Ecology Simulation Engine Design

This document specifies the architecture for a gut microbiome simulation engine following the RATCHET simulation interface requirements.

---

## 1. Architecture Overview

```
simulation_biology/
    __init__.py
    engine.py          # Main MicrobiomeEngine class
    loader.py          # Dataset loaders for HMP, AGP, DIABIMMUNE, etc.
    models/
        __init__.py
        species.py     # Species and taxonomy models
        network.py     # Interaction network models
        dynamics.py    # ODE-based dynamics models
    metrics/
        __init__.py
        diversity.py   # Alpha/beta diversity calculations
        correlation.py # SparCC and network metrics
        health.py      # GMHI and health indices
    shocks/
        __init__.py
        antibiotics.py # Antibiotic perturbation models
        diet.py        # Dietary intervention models
        fmt.py         # FMT intervention models
    examples/
        basic_simulation.py
        antibiotic_perturbation.py
        fmt_recovery.py
```

---

## 2. Core Classes

### 2.1 MicrobiomeEngine

The primary simulation engine implementing the required interface.

```python
"""
MicrobiomeEngine - Gut microbiome ecology simulation engine.

Simulates microbial community dynamics including:
- Species abundance changes over time
- Inter-species interactions (competition, mutualism)
- Response to perturbations (antibiotics, diet, infection)
- Recovery and intervention effects (probiotics, FMT)

This engine is theory-agnostic: it exposes manipulable and measurable
variables without assuming any particular relationship between them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum, auto

# Type aliases
SpeciesID = str
Abundance = float
Time = float


class ConstraintType(Enum):
    """What constitutes a 'constraint' in this simulation."""
    SPECIES = auto()           # Each species is a constraint
    FUNCTIONAL_GENE = auto()   # Each gene family is a constraint
    METABOLIC_PATHWAY = auto() # Each pathway is a constraint
    INTERACTION = auto()       # Each species-species interaction


class HealthMetric(Enum):
    """How to measure ecosystem sustainability."""
    SHANNON_DIVERSITY = auto()     # Shannon entropy of abundances
    SIMPSON_DIVERSITY = auto()     # Simpson index
    GMHI = auto()                  # Gut Microbiome Health Index
    FUNCTIONAL_DIVERSITY = auto()  # Pathway coverage


class CorrelationMethod(Enum):
    """How to compute inter-constraint correlation."""
    SPARCC = auto()           # SparCC for compositional data
    SPEARMAN = auto()         # Rank correlation (not recommended)
    FUNCTIONAL_REDUNDANCY = auto()  # Functional overlap


@dataclass
class MicrobiomeConfig:
    """Configuration for MicrobiomeEngine initialization."""

    # Constraint configuration
    constraint_type: ConstraintType = ConstraintType.SPECIES

    # Health metric configuration
    health_metric: HealthMetric = HealthMetric.SHANNON_DIVERSITY

    # Correlation configuration
    correlation_method: CorrelationMethod = CorrelationMethod.SPARCC

    # Collapse thresholds
    collapse_diversity_threshold: float = 2.0  # Shannon diversity
    collapse_pathogen_threshold: float = 0.3   # Max pathogen relative abundance
    collapse_gmhi_threshold: float = -5.0      # GMHI score

    # Dynamics parameters
    default_decay_rate: float = 0.15           # Per day, without substrate
    default_generation_rate: float = 0.5       # Species per day
    default_strictness: float = 1.0            # Interaction strength multiplier

    # Simulation settings
    integration_method: str = "euler"          # "euler" or "rk4"
    random_seed: Optional[int] = None


@dataclass
class Shock:
    """Represents an external perturbation to the ecosystem."""
    shock_type: str  # "antibiotic", "diet_change", "infection", "fasting"
    parameters: Dict[str, any] = field(default_factory=dict)

    # Antibiotic-specific
    target_taxa: Optional[List[str]] = None  # None = broad spectrum
    kill_fraction: float = 0.5               # Fraction of susceptible killed

    # Diet-specific
    substrate_change: Optional[Dict[str, float]] = None  # Substrate availability

    # Infection-specific
    pathogen_id: Optional[str] = None
    pathogen_dose: float = 0.01


@dataclass
class Intervention:
    """Represents a therapeutic intervention."""
    intervention_type: str  # "probiotic", "prebiotic", "fmt", "dietary_fiber"
    parameters: Dict[str, any] = field(default_factory=dict)

    # Probiotic-specific
    species_id: Optional[str] = None
    dose: float = 0.01  # Initial relative abundance

    # FMT-specific
    donor_profile: Optional[np.ndarray] = None  # Donor abundance vector
    engraftment_fraction: float = 0.5           # Fraction of recipient replaced


@dataclass
class MicrobiomeState:
    """Complete state of the microbial ecosystem."""

    # Primary state: species abundances (relative, sum to 1)
    abundances: np.ndarray
    species_ids: List[SpeciesID]

    # Derived metrics (computed on access)
    @property
    def k(self) -> int:
        """Number of detected species (constraint count)."""
        return np.sum(self.abundances > 1e-6)

    # Time
    time: Time = 0.0

    # Cached computations
    _rho: Optional[float] = None
    _sigma: Optional[float] = None
    _collapsed: Optional[bool] = None


class MicrobiomeEngine:
    """
    Gut microbiome ecology simulation engine.

    Implements the RATCHET simulation interface for microbial ecosystems.

    Example:
        >>> config = MicrobiomeConfig(
        ...     constraint_type=ConstraintType.SPECIES,
        ...     health_metric=HealthMetric.SHANNON_DIVERSITY
        ... )
        >>> engine = MicrobiomeEngine(config)
        >>> engine.initialize_from_reference("healthy_adult")
        >>>
        >>> # Run simulation
        >>> timeseries = engine.run(duration=30, dt=0.1)
        >>>
        >>> # Apply perturbation
        >>> engine.apply_shock(Shock(
        ...     shock_type="antibiotic",
        ...     kill_fraction=0.7
        ... ))
        >>>
        >>> # Continue simulation
        >>> timeseries = engine.run(duration=60, dt=0.1)
    """

    def __init__(self, config: Optional[MicrobiomeConfig] = None):
        """
        Initialize the MicrobiomeEngine.

        Args:
            config: Engine configuration. If None, uses defaults.
        """
        self.config = config or MicrobiomeConfig()

        # Initialize random state
        self.rng = np.random.default_rng(self.config.random_seed)

        # State variables
        self._state: Optional[MicrobiomeState] = None
        self._history: List[MicrobiomeState] = []

        # Species metadata
        self._species_ids: List[SpeciesID] = []
        self._species_metadata: Dict[SpeciesID, Dict] = {}

        # Interaction network
        self._interaction_matrix: Optional[np.ndarray] = None

        # Dynamics parameters (manipulable)
        self._decay_rate = self.config.default_decay_rate
        self._generation_rate = self.config.default_generation_rate
        self._strictness = self.config.default_strictness

        # Precomputed correlation matrix
        self._correlation_matrix: Optional[np.ndarray] = None

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def initialize_from_abundances(
        self,
        abundances: np.ndarray,
        species_ids: List[SpeciesID],
    ) -> None:
        """
        Initialize engine with explicit abundance vector.

        Args:
            abundances: Relative abundance vector (should sum to 1)
            species_ids: List of species identifiers
        """
        # Normalize to ensure sum = 1
        abundances = abundances / np.sum(abundances)

        self._species_ids = species_ids
        self._state = MicrobiomeState(
            abundances=abundances.copy(),
            species_ids=species_ids.copy(),
            time=0.0
        )
        self._history = [self._state]

        # Initialize default interaction matrix (neutral)
        n = len(species_ids)
        self._interaction_matrix = np.zeros((n, n))

    def initialize_from_reference(self, reference: str = "healthy_adult") -> None:
        """
        Initialize from a reference microbiome profile.

        Args:
            reference: Reference type ("healthy_adult", "infant", "dysbiotic")
        """
        # Load reference profile (placeholder - implement with actual data)
        n_species = 100
        species_ids = [f"species_{i:03d}" for i in range(n_species)]

        if reference == "healthy_adult":
            # Log-normal distribution typical of healthy gut
            abundances = self.rng.lognormal(mean=0, sigma=2, size=n_species)
        elif reference == "infant":
            # Lower diversity, dominated by few species
            abundances = self.rng.lognormal(mean=0, sigma=3, size=n_species)
        elif reference == "dysbiotic":
            # Very low diversity, pathogen-dominated
            abundances = self.rng.lognormal(mean=0, sigma=4, size=n_species)
            # Add pathogen dominance
            abundances[0] = abundances.sum() * 0.5
        else:
            raise ValueError(f"Unknown reference: {reference}")

        self.initialize_from_abundances(abundances, species_ids)

    def initialize_from_dataset(
        self,
        dataset: pd.DataFrame,
        sample_id: str,
    ) -> None:
        """
        Initialize from a loaded dataset.

        Args:
            dataset: DataFrame with samples as rows, species as columns
            sample_id: Which sample to use for initialization
        """
        abundances = dataset.loc[sample_id].values
        species_ids = list(dataset.columns)
        self.initialize_from_abundances(abundances, species_ids)

    # =========================================================================
    # CORE SIMULATION
    # =========================================================================

    def step(self, dt: float) -> None:
        """
        Advance simulation by one time step.

        Uses the generalized Lotka-Volterra equations:
            dx_i/dt = x_i * (r_i + sum_j(A_ij * x_j))

        Where:
            x_i = abundance of species i
            r_i = intrinsic growth rate (affected by decay rate d)
            A_ij = interaction coefficient (affected by strictness lambda)

        Args:
            dt: Time step size (in days)
        """
        if self._state is None:
            raise RuntimeError("Engine not initialized")

        x = self._state.abundances.copy()
        n = len(x)

        # Compute growth rates
        # Base rate modified by decay (without substrate) and generation (colonization)
        r = np.ones(n) * (-self._decay_rate + self._generation_rate * (1 - np.sum(x > 1e-6) / n))

        # Apply interaction effects (scaled by strictness)
        if self._interaction_matrix is not None:
            interaction_effect = self._strictness * self._interaction_matrix @ x
        else:
            interaction_effect = np.zeros(n)

        # Compute derivatives
        dxdt = x * (r + interaction_effect)

        # Euler integration
        x_new = x + dt * dxdt

        # Enforce non-negativity and renormalize
        x_new = np.maximum(x_new, 0)
        total = np.sum(x_new)
        if total > 0:
            x_new = x_new / total
        else:
            # Complete collapse - reinitialize with minimal diversity
            x_new = np.ones(n) / n

        # Update state
        self._state = MicrobiomeState(
            abundances=x_new,
            species_ids=self._species_ids,
            time=self._state.time + dt
        )
        self._history.append(self._state)

    def run(self, duration: float, dt: float = 0.1) -> pd.DataFrame:
        """
        Run simulation for specified duration.

        Args:
            duration: Total simulation time (days)
            dt: Time step (days)

        Returns:
            DataFrame with time series of all variables
        """
        n_steps = int(duration / dt)

        for _ in range(n_steps):
            self.step(dt)

        return self.to_dataframe()

    # =========================================================================
    # VARIABLE MANIPULATION (INPUTS)
    # =========================================================================

    def set_k(self, k: int) -> None:
        """
        Set the number of constraints (species).

        Adjusts community to have exactly k species by removing
        lowest-abundance species or adding new colonizers.

        Args:
            k: Target number of species
        """
        if self._state is None:
            raise RuntimeError("Engine not initialized")

        current_k = self.get_k()
        x = self._state.abundances.copy()

        if k < current_k:
            # Remove lowest-abundance species
            sorted_idx = np.argsort(x)
            n_to_remove = current_k - k
            x[sorted_idx[:n_to_remove]] = 0
        elif k > current_k:
            # Add new species at low abundance
            n_to_add = k - current_k
            zero_idx = np.where(x == 0)[0]
            if len(zero_idx) >= n_to_add:
                x[zero_idx[:n_to_add]] = 1e-4

        # Renormalize
        x = x / np.sum(x)
        self._state = MicrobiomeState(
            abundances=x,
            species_ids=self._species_ids,
            time=self._state.time
        )

    def set_alpha(self, alpha: float) -> None:
        """
        Set constraint generation rate (colonization rate).

        Args:
            alpha: New species per day
        """
        self._generation_rate = alpha

    def set_d(self, d: float) -> None:
        """
        Set decay rate (degradation without substrate).

        Args:
            d: Decay rate in 1/days
        """
        self._decay_rate = d

    def set_lambda(self, lambda_: float) -> None:
        """
        Set strictness (interaction strength multiplier).

        Args:
            lambda_: Strictness coefficient
        """
        self._strictness = lambda_

    def apply_shock(self, shock: Shock) -> None:
        """
        Apply an external perturbation to the ecosystem.

        Args:
            shock: Shock specification
        """
        if self._state is None:
            raise RuntimeError("Engine not initialized")

        x = self._state.abundances.copy()

        if shock.shock_type == "antibiotic":
            # Kill fraction of susceptible species
            if shock.target_taxa is None:
                # Broad spectrum - affects all species
                survival = 1 - shock.kill_fraction * self.rng.uniform(0.5, 1.0, len(x))
            else:
                # Narrow spectrum - only target specific taxa
                survival = np.ones(len(x))
                for i, sid in enumerate(self._species_ids):
                    if any(taxon in sid for taxon in shock.target_taxa):
                        survival[i] = 1 - shock.kill_fraction

            x = x * survival

        elif shock.shock_type == "diet_change":
            # Modify abundances based on substrate preferences
            # (Simplified: random perturbation)
            perturbation = self.rng.uniform(0.5, 1.5, len(x))
            x = x * perturbation

        elif shock.shock_type == "infection":
            # Introduce pathogen
            if shock.pathogen_id is not None:
                pathogen_idx = self._species_ids.index(shock.pathogen_id)
            else:
                pathogen_idx = 0  # Default to first species
            x[pathogen_idx] += shock.pathogen_dose

        elif shock.shock_type == "fasting":
            # Accelerate decay for all species
            fasting_effect = np.exp(-self._decay_rate * 2)  # 2x decay
            x = x * fasting_effect

        # Renormalize
        x = x / np.sum(x)
        self._state = MicrobiomeState(
            abundances=x,
            species_ids=self._species_ids,
            time=self._state.time
        )

    def apply_intervention(self, intervention: Intervention) -> None:
        """
        Apply a therapeutic intervention.

        Args:
            intervention: Intervention specification
        """
        if self._state is None:
            raise RuntimeError("Engine not initialized")

        x = self._state.abundances.copy()

        if intervention.intervention_type == "probiotic":
            # Add probiotic species
            if intervention.species_id is not None:
                try:
                    idx = self._species_ids.index(intervention.species_id)
                except ValueError:
                    # Species not in list - add it
                    self._species_ids.append(intervention.species_id)
                    x = np.append(x, 0)
                    idx = len(x) - 1
                x[idx] += intervention.dose

        elif intervention.intervention_type == "fmt":
            # Fecal microbiota transplant
            if intervention.donor_profile is not None:
                donor = intervention.donor_profile
            else:
                # Default: healthy reference profile
                donor = self.rng.lognormal(mean=0, sigma=2, size=len(x))
                donor = donor / np.sum(donor)

            # Mix recipient and donor
            ef = intervention.engraftment_fraction
            x = (1 - ef) * x + ef * donor

        elif intervention.intervention_type == "prebiotic":
            # Boost fiber-fermenting species (simplified)
            boost_idx = self.rng.choice(len(x), size=len(x)//4, replace=False)
            x[boost_idx] *= 1.5

        elif intervention.intervention_type == "dietary_fiber":
            # Reduce decay rate temporarily
            self._decay_rate *= 0.5

        # Renormalize
        x = x / np.sum(x)
        self._state = MicrobiomeState(
            abundances=x,
            species_ids=self._species_ids,
            time=self._state.time
        )

    # =========================================================================
    # VARIABLE MEASUREMENT (OUTPUTS)
    # =========================================================================

    def get_k(self) -> int:
        """Get current constraint count (number of detected species)."""
        if self._state is None:
            return 0
        return int(np.sum(self._state.abundances > 1e-6))

    def get_rho(self) -> float:
        """
        Get current average correlation between constraints.

        Uses SparCC or configured correlation method.
        """
        if self._state is None:
            return 0.0

        # For real-time simulation, use precomputed or approximated correlation
        if self._correlation_matrix is not None:
            # Mean absolute off-diagonal correlation
            n = len(self._state.abundances)
            mask = ~np.eye(n, dtype=bool)
            return float(np.mean(np.abs(self._correlation_matrix[mask])))
        else:
            # Default: estimate from current state (placeholder)
            # In practice, correlation should be computed from time series
            return 0.2  # Typical value

    def get_k_eff(self) -> float:
        """
        Get effective constraint count adjusted for correlation.

        Formula: k_eff = k / (1 + rho * (k - 1))
        """
        k = self.get_k()
        rho = self.get_rho()

        if k <= 1:
            return float(k)

        denominator = 1 + rho * (k - 1)
        if denominator <= 0:
            # Invalid correlation for this k
            return float(k)

        return k / denominator

    def get_sigma(self) -> float:
        """
        Get current sustainability metric (ecosystem health).

        Returns value in [0, 1] based on configured health metric.
        """
        if self._state is None:
            return 0.0

        x = self._state.abundances
        x = x[x > 1e-10]  # Remove zeros

        if len(x) == 0:
            return 0.0

        if self.config.health_metric == HealthMetric.SHANNON_DIVERSITY:
            # Shannon entropy normalized to [0, 1]
            H = -np.sum(x * np.log(x))
            H_max = np.log(len(x)) if len(x) > 1 else 1.0
            return H / H_max if H_max > 0 else 0.0

        elif self.config.health_metric == HealthMetric.SIMPSON_DIVERSITY:
            return 1 - np.sum(x ** 2)

        elif self.config.health_metric == HealthMetric.GMHI:
            # Simplified GMHI (placeholder - would need real health species)
            # Returns sigmoid-normalized value
            gmhi_raw = self.get_k() / 100 - 0.5  # Placeholder formula
            return 1 / (1 + np.exp(-gmhi_raw))

        else:
            return 0.5  # Default

    def get_f(self) -> float:
        """
        Get fraction of ecosystem compromised (pathogen dominance).

        Returns sum of relative abundances of known pathogens.
        """
        if self._state is None:
            return 0.0

        # Simplified: assume first few species are potential pathogens
        # In practice, would use pathogen database lookup
        pathogen_idx = [0, 1, 2]  # Placeholder
        return float(np.sum(self._state.abundances[pathogen_idx]))

    def get_state(self) -> np.ndarray:
        """Get full state vector (abundance array)."""
        if self._state is None:
            return np.array([])
        return self._state.abundances.copy()

    def is_collapsed(self) -> bool:
        """
        Check if ecosystem has collapsed.

        Collapse criteria:
        1. Diversity below threshold
        2. Pathogen dominance above threshold
        3. GMHI below threshold (if using GMHI)
        """
        if self._state is None:
            return False

        x = self._state.abundances
        x_nonzero = x[x > 1e-10]

        # Check diversity collapse
        if len(x_nonzero) > 0:
            H = -np.sum(x_nonzero * np.log(x_nonzero))
            if H < self.config.collapse_diversity_threshold:
                return True

        # Check pathogen dominance
        max_abundance = np.max(x)
        if max_abundance > self.config.collapse_pathogen_threshold:
            return True

        return False

    def get_collapse_time(self) -> Optional[float]:
        """Get time at which collapse occurred, if any."""
        for state in self._history:
            # Check collapse for historical state
            x = state.abundances[state.abundances > 1e-10]
            if len(x) > 0:
                H = -np.sum(x * np.log(x))
                if H < self.config.collapse_diversity_threshold:
                    return state.time
        return None

    # =========================================================================
    # DATA EXPORT
    # =========================================================================

    def to_dataframe(self) -> pd.DataFrame:
        """
        Export simulation history as DataFrame.

        Returns:
            DataFrame with columns for time and all measured variables.
        """
        records = []

        for state in self._history:
            x = state.abundances
            x_nonzero = x[x > 1e-10]

            # Compute metrics for this state
            k = int(np.sum(x > 1e-6))
            rho = self.get_rho()  # Approximation
            k_eff = k / (1 + rho * (k - 1)) if k > 1 else k

            if len(x_nonzero) > 0:
                H = -np.sum(x_nonzero * np.log(x_nonzero))
                H_max = np.log(len(x_nonzero)) if len(x_nonzero) > 1 else 1.0
                sigma = H / H_max if H_max > 0 else 0.0
            else:
                sigma = 0.0

            f = float(np.sum(x[:3]))  # Simplified pathogen fraction
            collapsed = sigma < 0.5 or np.max(x) > 0.3

            records.append({
                'time': state.time,
                'k': k,
                'rho': rho,
                'k_eff': k_eff,
                'sigma': sigma,
                'f': f,
                'collapsed': collapsed,
                'd': self._decay_rate,
                'alpha': self._generation_rate,
                'lambda': self._strictness,
            })

        return pd.DataFrame(records)

    def to_abundance_dataframe(self) -> pd.DataFrame:
        """
        Export full abundance time series.

        Returns:
            DataFrame with time as index and species as columns.
        """
        records = []
        for state in self._history:
            record = {'time': state.time}
            for i, sid in enumerate(state.species_ids):
                record[sid] = state.abundances[i]
            records.append(record)

        df = pd.DataFrame(records)
        df.set_index('time', inplace=True)
        return df


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_microbiome_engine(
    config: Optional[MicrobiomeConfig] = None,
    seed: Optional[int] = None,
) -> MicrobiomeEngine:
    """
    Factory function to create a MicrobiomeEngine.

    Args:
        config: Engine configuration
        seed: Random seed (overrides config if provided)

    Returns:
        Configured MicrobiomeEngine instance
    """
    if config is None:
        config = MicrobiomeConfig()

    if seed is not None:
        config.random_seed = seed

    return MicrobiomeEngine(config)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Enums
    'ConstraintType',
    'HealthMetric',
    'CorrelationMethod',

    # Dataclasses
    'MicrobiomeConfig',
    'Shock',
    'Intervention',
    'MicrobiomeState',

    # Engine
    'MicrobiomeEngine',

    # Factory
    'create_microbiome_engine',
]
```

---

## 3. Dataset Loader Interface

```python
"""
loader.py - Dataset loaders for microbiome simulation.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass


@dataclass
class DatasetMetadata:
    """Metadata about a loaded dataset."""
    name: str
    source: str
    n_samples: int
    n_species: int
    has_time_series: bool
    time_range: Optional[Tuple[float, float]] = None
    body_site: str = "gut"


class DatasetLoader(ABC):
    """Abstract base class for dataset loaders."""

    @abstractmethod
    def load(self, path: Union[str, Path]) -> Tuple[pd.DataFrame, DatasetMetadata]:
        """
        Load dataset from path.

        Args:
            path: Path to dataset file or directory

        Returns:
            Tuple of (abundance_dataframe, metadata)
        """
        pass

    @abstractmethod
    def get_sample_ids(self) -> List[str]:
        """Get list of available sample IDs."""
        pass

    @abstractmethod
    def get_time_points(self, subject_id: str) -> List[float]:
        """Get time points for a subject (if longitudinal)."""
        pass


class HMPLoader(DatasetLoader):
    """Loader for Human Microbiome Project data."""

    def __init__(self):
        self._data: Optional[pd.DataFrame] = None
        self._metadata: Optional[DatasetMetadata] = None

    def load(self, path: Union[str, Path]) -> Tuple[pd.DataFrame, DatasetMetadata]:
        """Load HMP data from BIOM or TSV file."""
        path = Path(path)

        if path.suffix == '.biom':
            # Load BIOM format
            import biom
            table = biom.load_table(str(path))
            df = pd.DataFrame(
                table.matrix_data.toarray().T,
                index=table.ids('sample'),
                columns=table.ids('observation')
            )
        else:
            # Assume TSV
            df = pd.read_csv(path, sep='\t', index_col=0)

        self._data = df
        self._metadata = DatasetMetadata(
            name="Human Microbiome Project",
            source="hmpdacc.org",
            n_samples=len(df),
            n_species=len(df.columns),
            has_time_series=False,
            body_site="multiple"
        )

        return df, self._metadata

    def get_sample_ids(self) -> List[str]:
        if self._data is None:
            return []
        return list(self._data.index)

    def get_time_points(self, subject_id: str) -> List[float]:
        # HMP is cross-sectional
        return [0.0]


class DIABIMMUNELoader(DatasetLoader):
    """Loader for DIABIMMUNE longitudinal infant data."""

    def __init__(self):
        self._data: Optional[pd.DataFrame] = None
        self._metadata_df: Optional[pd.DataFrame] = None
        self._metadata: Optional[DatasetMetadata] = None

    def load(self, path: Union[str, Path]) -> Tuple[pd.DataFrame, DatasetMetadata]:
        """Load DIABIMMUNE data."""
        path = Path(path)

        # Load abundance data
        abundance_file = path / "abundance.tsv"
        metadata_file = path / "metadata.tsv"

        if abundance_file.exists():
            df = pd.read_csv(abundance_file, sep='\t', index_col=0)
        else:
            # Fallback to single file
            df = pd.read_csv(path, sep='\t', index_col=0)

        # Load metadata if available
        if metadata_file.exists():
            self._metadata_df = pd.read_csv(metadata_file, sep='\t', index_col=0)

        self._data = df

        # Determine time range from metadata
        time_range = None
        if self._metadata_df is not None and 'age_days' in self._metadata_df.columns:
            time_range = (
                self._metadata_df['age_days'].min(),
                self._metadata_df['age_days'].max()
            )

        self._metadata = DatasetMetadata(
            name="DIABIMMUNE",
            source="diabimmune.broadinstitute.org",
            n_samples=len(df),
            n_species=len(df.columns),
            has_time_series=True,
            time_range=time_range,
            body_site="gut"
        )

        return df, self._metadata

    def get_sample_ids(self) -> List[str]:
        if self._data is None:
            return []
        return list(self._data.index)

    def get_time_points(self, subject_id: str) -> List[float]:
        if self._metadata_df is None:
            return [0.0]

        # Filter samples for this subject
        subject_samples = self._metadata_df[
            self._metadata_df['subject_id'] == subject_id
        ]

        if 'age_days' in subject_samples.columns:
            return list(subject_samples['age_days'].values)
        return [0.0]

    def get_antibiotic_events(self, subject_id: str) -> List[Tuple[float, str]]:
        """Get antibiotic administration events for a subject."""
        if self._metadata_df is None:
            return []

        subject_samples = self._metadata_df[
            self._metadata_df['subject_id'] == subject_id
        ]

        events = []
        if 'antibiotic_type' in subject_samples.columns:
            for _, row in subject_samples.iterrows():
                if pd.notna(row.get('antibiotic_type')):
                    events.append((row['age_days'], row['antibiotic_type']))

        return events


class CuratedMetagenomicDataLoader(DatasetLoader):
    """Loader for curatedMetagenomicData R package exports."""

    def __init__(self):
        self._data: Optional[pd.DataFrame] = None
        self._metadata: Optional[DatasetMetadata] = None

    def load(self, path: Union[str, Path]) -> Tuple[pd.DataFrame, DatasetMetadata]:
        """Load curatedMetagenomicData export."""
        path = Path(path)

        # Expect exported CSV from R
        df = pd.read_csv(path, index_col=0)

        self._data = df
        self._metadata = DatasetMetadata(
            name="curatedMetagenomicData",
            source="bioconductor.org",
            n_samples=len(df),
            n_species=len(df.columns),
            has_time_series=False,  # Depends on constituent study
            body_site="gut"
        )

        return df, self._metadata

    def get_sample_ids(self) -> List[str]:
        if self._data is None:
            return []
        return list(self._data.index)

    def get_time_points(self, subject_id: str) -> List[float]:
        return [0.0]


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def load_dataset(
    source: str,
    path: Union[str, Path],
) -> Tuple[pd.DataFrame, DatasetMetadata]:
    """
    Load a microbiome dataset.

    Args:
        source: Dataset source ("hmp", "diabimmune", "curated", "agp")
        path: Path to dataset file or directory

    Returns:
        Tuple of (abundance_dataframe, metadata)
    """
    loaders = {
        'hmp': HMPLoader,
        'diabimmune': DIABIMMUNELoader,
        'curated': CuratedMetagenomicDataLoader,
    }

    if source.lower() not in loaders:
        raise ValueError(f"Unknown source: {source}. Available: {list(loaders.keys())}")

    loader = loaders[source.lower()]()
    return loader.load(path)


__all__ = [
    'DatasetMetadata',
    'DatasetLoader',
    'HMPLoader',
    'DIABIMMUNELoader',
    'CuratedMetagenomicDataLoader',
    'load_dataset',
]
```

---

## 4. Metrics Module

```python
"""
metrics/diversity.py - Diversity metrics for microbiome analysis.
"""

import numpy as np
from typing import Optional


def shannon_diversity(abundances: np.ndarray) -> float:
    """
    Compute Shannon diversity index.

    H = -sum(p_i * log(p_i))

    Args:
        abundances: Relative abundance vector

    Returns:
        Shannon diversity index
    """
    p = abundances[abundances > 1e-10]
    if len(p) == 0:
        return 0.0
    p = p / np.sum(p)  # Ensure normalization
    return -np.sum(p * np.log(p))


def simpson_diversity(abundances: np.ndarray) -> float:
    """
    Compute Simpson diversity index.

    D = 1 - sum(p_i^2)

    Args:
        abundances: Relative abundance vector

    Returns:
        Simpson diversity (1 - dominance)
    """
    p = abundances[abundances > 1e-10]
    if len(p) == 0:
        return 0.0
    p = p / np.sum(p)
    return 1 - np.sum(p ** 2)


def chao1_richness(abundances: np.ndarray, counts: Optional[np.ndarray] = None) -> float:
    """
    Compute Chao1 richness estimator.

    Chao1 = S_obs + n1^2 / (2 * n2)

    Where:
        S_obs = observed species count
        n1 = number of singletons
        n2 = number of doubletons

    Args:
        abundances: Relative abundance vector
        counts: Raw count data (if available, more accurate)

    Returns:
        Chao1 richness estimate
    """
    if counts is not None:
        c = counts[counts > 0].astype(int)
        s_obs = len(c)
        n1 = np.sum(c == 1)
        n2 = np.sum(c == 2)
    else:
        # Estimate from relative abundance (less accurate)
        p = abundances[abundances > 1e-10]
        s_obs = len(p)
        # Approximate singletons/doubletons from low-abundance species
        n1 = np.sum(p < 0.001)
        n2 = np.sum((p >= 0.001) & (p < 0.002))

    if n2 == 0:
        n2 = 1  # Avoid division by zero

    return s_obs + (n1 ** 2) / (2 * n2)


def normalize_diversity(H: float, max_species: int) -> float:
    """
    Normalize Shannon diversity to [0, 1].

    Args:
        H: Shannon diversity
        max_species: Maximum possible species count

    Returns:
        Normalized diversity (0 = monoculture, 1 = max evenness)
    """
    H_max = np.log(max_species) if max_species > 1 else 1.0
    return H / H_max if H_max > 0 else 0.0
```

---

## 5. Example Usage

```python
"""
examples/basic_simulation.py - Basic microbiome simulation example.
"""

from simulation_biology.engine import (
    MicrobiomeEngine,
    MicrobiomeConfig,
    Shock,
    Intervention,
    ConstraintType,
    HealthMetric,
)
import matplotlib.pyplot as plt


def run_basic_simulation():
    """Run a basic microbiome simulation with perturbation."""

    # Configure engine
    config = MicrobiomeConfig(
        constraint_type=ConstraintType.SPECIES,
        health_metric=HealthMetric.SHANNON_DIVERSITY,
        collapse_diversity_threshold=2.0,
        default_decay_rate=0.15,
        default_generation_rate=0.5,
        random_seed=42
    )

    # Create engine and initialize
    engine = MicrobiomeEngine(config)
    engine.initialize_from_reference("healthy_adult")

    print(f"Initial state:")
    print(f"  k = {engine.get_k()} species")
    print(f"  sigma = {engine.get_sigma():.3f}")
    print(f"  collapsed = {engine.is_collapsed()}")

    # Run for 30 days
    print("\nRunning baseline simulation (30 days)...")
    baseline_df = engine.run(duration=30, dt=0.1)

    # Apply antibiotic shock
    print("\nApplying broad-spectrum antibiotic...")
    engine.apply_shock(Shock(
        shock_type="antibiotic",
        kill_fraction=0.7
    ))

    print(f"Post-antibiotic state:")
    print(f"  k = {engine.get_k()} species")
    print(f"  sigma = {engine.get_sigma():.3f}")
    print(f"  collapsed = {engine.is_collapsed()}")

    # Run recovery period
    print("\nRunning recovery simulation (60 days)...")
    recovery_df = engine.run(duration=60, dt=0.1)

    # Apply probiotic intervention
    print("\nApplying probiotic intervention...")
    engine.apply_intervention(Intervention(
        intervention_type="probiotic",
        species_id="beneficial_001",
        dose=0.05
    ))

    # Continue simulation
    print("\nRunning post-intervention simulation (30 days)...")
    final_df = engine.run(duration=30, dt=0.1)

    # Get complete time series
    full_df = engine.to_dataframe()

    print(f"\nFinal state:")
    print(f"  k = {engine.get_k()} species")
    print(f"  sigma = {engine.get_sigma():.3f}")
    print(f"  collapsed = {engine.is_collapsed()}")

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(full_df['time'], full_df['k'])
    axes[0, 0].axvline(x=30, color='r', linestyle='--', label='Antibiotic')
    axes[0, 0].axvline(x=90, color='g', linestyle='--', label='Probiotic')
    axes[0, 0].set_xlabel('Time (days)')
    axes[0, 0].set_ylabel('Species count (k)')
    axes[0, 0].legend()

    axes[0, 1].plot(full_df['time'], full_df['sigma'])
    axes[0, 1].axhline(y=0.5, color='orange', linestyle=':', label='Collapse threshold')
    axes[0, 1].set_xlabel('Time (days)')
    axes[0, 1].set_ylabel('Health (sigma)')
    axes[0, 1].legend()

    axes[1, 0].plot(full_df['time'], full_df['k_eff'])
    axes[1, 0].set_xlabel('Time (days)')
    axes[1, 0].set_ylabel('Effective species (k_eff)')

    axes[1, 1].plot(full_df['time'], full_df['f'])
    axes[1, 1].set_xlabel('Time (days)')
    axes[1, 1].set_ylabel('Compromised fraction (f)')

    plt.tight_layout()
    plt.savefig('simulation_results.png', dpi=150)
    print("\nResults saved to simulation_results.png")

    return full_df


if __name__ == "__main__":
    run_basic_simulation()
```

---

## 6. Key Design Decisions

### 6.1 Theory-Agnostic Design

The engine does NOT:
- Assume any relationship between k, rho, and sigma
- Hard-code expected behaviors or thresholds
- Predict collapse based on theoretical formulas
- Validate or test any specific theory

The engine DOES:
- Expose all variables for manipulation
- Measure all variables accurately
- Record complete time series
- Allow arbitrary experimental designs

### 6.2 Configurable Interpretations

Users can configure:
- What constitutes a "constraint" (species, genes, pathways)
- How to measure correlation (SparCC, functional redundancy)
- What defines "health" (Shannon, GMHI, stability)
- When "collapse" occurs (thresholds are parameters)

### 6.3 Realistic Dynamics

The dynamics model includes:
- Lotka-Volterra-style species interactions
- Decay without substrate input
- Colonization/recovery processes
- Perturbation effects (antibiotics, diet)
- Intervention effects (probiotics, FMT)

### 6.4 Data Integration

The engine can be:
- Initialized from real datasets (HMP, DIABIMMUNE, etc.)
- Parameterized from empirical measurements
- Validated against longitudinal observations
- Used to generate synthetic data for testing
