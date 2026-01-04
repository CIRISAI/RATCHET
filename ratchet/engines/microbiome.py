"""
RATCHET Microbiome Ecology Simulation Engine

Simulates gut microbiome dynamics for structural invariant testing.
Theory-agnostic: exposes manipulable and measurable variables without
assuming relationships between them.

Domain Mapping:
    k (constraints):     Number of detected species (OTUs/ASVs)
    rho (correlation):   Mean SparCC correlation between species
    sigma (sustainability): Normalized Shannon diversity (0-1)
    f (compromise):      Pathogen dominance fraction
    d (decay rate):      Diversity loss rate without substrate (~0.15/day)
    alpha (generation):  Colonization rate (~0.5 species/day)
    lambda (strictness): Interaction strength multiplier

References:
    - Human Microbiome Project (HMP)
    - DIABIMMUNE longitudinal infant gut study
    - curatedMetagenomicData R/Bioconductor package
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


class ShockType(Enum):
    """Types of perturbations to the microbiome."""
    ANTIBIOTIC_BROAD = "antibiotic_broad"
    ANTIBIOTIC_NARROW = "antibiotic_narrow"
    DIET_CHANGE = "diet_change"
    INFECTION = "infection"
    FASTING = "fasting"


class InterventionType(Enum):
    """Types of therapeutic interventions."""
    PROBIOTIC = "probiotic"
    PREBIOTIC = "prebiotic"
    FMT = "fmt"
    DIETARY_FIBER = "dietary_fiber"


@dataclass
class MicrobiomeShock:
    """External perturbation to the microbiome ecosystem."""
    type: ShockType
    magnitude: float = 0.5
    target_taxa: Optional[List[str]] = None
    duration: int = 1


@dataclass
class MicrobiomeIntervention:
    """Therapeutic intervention on the microbiome."""
    type: InterventionType
    intensity: float = 0.5
    species_id: Optional[str] = None
    donor_profile: Optional[np.ndarray] = None


@dataclass
class MicrobiomeState:
    """State of the microbial ecosystem at a point in time."""
    abundances: np.ndarray
    species_ids: List[str]
    time: float = 0.0

    @property
    def k(self) -> int:
        """Number of detected species."""
        return int(np.sum(self.abundances > 1e-4))


@dataclass
class MicrobiomeParams:
    """Configuration for MicrobiomeEngine."""
    engine: str = "microbiome"
    n_species: int = 100
    collapse_diversity_threshold: float = 2.0
    collapse_pathogen_threshold: float = 0.3
    extinction_threshold: float = 1e-4  # Species below this are considered extinct
    default_decay_rate: float = 0.15
    default_generation_rate: float = 0.5
    default_strictness: float = 1.0
    seed: Optional[int] = None


class MicrobiomeEngine:
    """
    Microbiome ecology simulation engine.

    Simulates microbial community dynamics using Lotka-Volterra-style
    interactions. Exposes RATCHET framework variables for manipulation
    and measurement.

    Example:
        >>> engine = MicrobiomeEngine(seed=42)
        >>> engine.initialize_from_reference("healthy_adult")
        >>> ts = engine.run(duration=30, dt=0.1)
        >>> print(f"Final diversity: {engine.get_sigma():.3f}")
    """

    def __init__(
        self,
        params: Optional[MicrobiomeParams] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize the MicrobiomeEngine.

        Args:
            params: Engine configuration. If None, uses defaults.
            seed: Random seed for reproducibility.
        """
        self.params = params or MicrobiomeParams()
        if seed is not None:
            self.params.seed = seed

        self.rng = np.random.default_rng(self.params.seed)
        self._state: Optional[MicrobiomeState] = None
        self._history: List[Dict] = []

        # Dynamics parameters (manipulable)
        self._decay_rate = self.params.default_decay_rate
        self._generation_rate = self.params.default_generation_rate
        self._strictness = self.params.default_strictness

        # Interaction matrix (set during initialization)
        self._interaction_matrix: Optional[np.ndarray] = None

        # Collapse tracking
        self._collapsed = False
        self._collapse_time: Optional[float] = None

        # Precomputed correlation (approximate)
        self._correlation_estimate = 0.2

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def initialize_from_abundances(
        self,
        abundances: np.ndarray,
        species_ids: Optional[List[str]] = None,
    ) -> None:
        """Initialize engine with explicit abundance vector."""
        n = len(abundances)
        abundances = abundances / np.sum(abundances)

        if species_ids is None:
            species_ids = [f"species_{i:03d}" for i in range(n)]

        self._state = MicrobiomeState(
            abundances=abundances.copy(),
            species_ids=species_ids.copy(),
            time=0.0,
        )
        self._history = [self._record_state()]

        # Initialize neutral interaction matrix
        self._interaction_matrix = np.zeros((n, n))

    def initialize_from_reference(self, reference: str = "healthy_adult") -> None:
        """
        Initialize from a reference microbiome profile.

        Args:
            reference: "healthy_adult", "infant", or "dysbiotic"
        """
        n = self.params.n_species
        species_ids = [f"species_{i:03d}" for i in range(n)]

        if reference == "healthy_adult":
            abundances = self.rng.lognormal(mean=0, sigma=2, size=n)
        elif reference == "infant":
            abundances = self.rng.lognormal(mean=0, sigma=3, size=n)
        elif reference == "dysbiotic":
            abundances = self.rng.lognormal(mean=0, sigma=4, size=n)
            abundances[0] = abundances.sum() * 0.5  # Pathogen dominance
        else:
            raise ValueError(f"Unknown reference: {reference}")

        self.initialize_from_abundances(abundances, species_ids)

    # =========================================================================
    # CORE SIMULATION
    # =========================================================================

    def step(self, dt: float = 0.1) -> None:
        """Advance simulation by one time step."""
        if self._state is None:
            raise RuntimeError("Engine not initialized")

        if self._collapsed:
            return

        x = self._state.abundances.copy()
        n = len(x)

        # Growth rates: decay + density-dependent generation
        occupancy = np.sum(x > 1e-6) / n
        r = np.ones(n) * (-self._decay_rate + self._generation_rate * (1 - occupancy))

        # Add small random noise
        r += self.rng.normal(0, 0.01, n)

        # Interaction effects (scaled by strictness)
        if self._interaction_matrix is not None:
            interaction = self._strictness * self._interaction_matrix @ x
        else:
            interaction = np.zeros(n)

        # Euler integration
        dxdt = x * (r + interaction)
        x_new = x + dt * dxdt

        # Enforce non-negativity and renormalize
        x_new = np.maximum(x_new, 0)
        total = np.sum(x_new)
        if total > 0:
            x_new = x_new / total
        else:
            x_new = np.ones(n) / n

        self._state = MicrobiomeState(
            abundances=x_new,
            species_ids=self._state.species_ids,
            time=self._state.time + dt,
        )

        self._history.append(self._record_state())
        self._check_collapse()

    def run(self, duration: float, dt: float = 0.1) -> pd.DataFrame:
        """Run simulation for specified duration."""
        n_steps = int(duration / dt)
        for _ in range(n_steps):
            self.step(dt)
            if self._collapsed:
                break
        return self.to_dataframe()

    def _record_state(self) -> Dict:
        """Record current state for history."""
        return {
            'time': self._state.time,
            'k': self.get_k(),
            'rho': self.get_rho(),
            'k_eff': self.get_k_eff(),
            'sigma': self.get_sigma(),
            'f': self.get_f(),
            'd': self._decay_rate,
            'alpha': self._generation_rate,
            'lambda': self._strictness,
            'collapsed': self._collapsed,
        }

    def _check_collapse(self) -> None:
        """Check if ecosystem has collapsed."""
        if self._collapsed:
            return

        # Collapse conditions
        x = self._state.abundances
        x_nonzero = x[x > 1e-10]

        if len(x_nonzero) > 0:
            H = -np.sum(x_nonzero * np.log(x_nonzero))
            if H < self.params.collapse_diversity_threshold:
                self._collapsed = True
                self._collapse_time = self._state.time
                return

        if np.max(x) > self.params.collapse_pathogen_threshold:
            self._collapsed = True
            self._collapse_time = self._state.time

    # =========================================================================
    # VARIABLE MANIPULATION (Inputs)
    # =========================================================================

    def set_k(self, k: int) -> None:
        """Set constraint count (number of species)."""
        if self._state is None:
            raise RuntimeError("Engine not initialized")

        current_k = self.get_k()
        x = self._state.abundances.copy()

        if k < current_k:
            sorted_idx = np.argsort(x)
            n_remove = current_k - k
            x[sorted_idx[:n_remove]] = 0
        elif k > current_k:
            zero_idx = np.where(x == 0)[0]
            n_add = min(k - current_k, len(zero_idx))
            x[zero_idx[:n_add]] = 1e-4

        x = x / np.sum(x)
        self._state = MicrobiomeState(
            abundances=x,
            species_ids=self._state.species_ids,
            time=self._state.time,
        )

    def set_alpha(self, alpha: float) -> None:
        """Set constraint generation rate (colonization rate)."""
        self._generation_rate = max(0, alpha)

    def set_d(self, d: float) -> None:
        """Set decay rate (degradation without substrate)."""
        self._decay_rate = max(0, d)

    def set_lambda(self, lambda_: float) -> None:
        """Set strictness (interaction strength)."""
        self._strictness = max(0, lambda_)

    def apply_shock(self, shock: MicrobiomeShock) -> None:
        """Apply external perturbation."""
        if self._state is None:
            raise RuntimeError("Engine not initialized")

        x = self._state.abundances.copy()

        if shock.type in (ShockType.ANTIBIOTIC_BROAD, ShockType.ANTIBIOTIC_NARROW):
            if shock.type == ShockType.ANTIBIOTIC_BROAD:
                survival = 1 - shock.magnitude * self.rng.uniform(0.5, 1.0, len(x))
            else:
                survival = np.ones(len(x))
                if shock.target_taxa:
                    for i, sid in enumerate(self._state.species_ids):
                        if any(t in sid for t in shock.target_taxa):
                            survival[i] = 1 - shock.magnitude
            x = x * survival
            # Species below extinction threshold are eliminated (reduces k)
            x[x < self.params.extinction_threshold] = 0.0

        elif shock.type == ShockType.DIET_CHANGE:
            perturbation = self.rng.uniform(0.5, 1.5, len(x))
            x = x * perturbation

        elif shock.type == ShockType.INFECTION:
            x[0] += shock.magnitude * 0.1

        elif shock.type == ShockType.FASTING:
            x = x * np.exp(-self._decay_rate * 2)

        x = x / np.sum(x)
        self._state = MicrobiomeState(
            abundances=x,
            species_ids=self._state.species_ids,
            time=self._state.time,
        )

    def apply_intervention(self, intervention: MicrobiomeIntervention) -> None:
        """Apply therapeutic intervention."""
        if self._state is None:
            raise RuntimeError("Engine not initialized")

        x = self._state.abundances.copy()

        if intervention.type == InterventionType.PROBIOTIC:
            idx = 0  # Default: boost first species
            x[idx] += intervention.intensity * 0.05

        elif intervention.type == InterventionType.FMT:
            if intervention.donor_profile is not None:
                donor = intervention.donor_profile
            else:
                # Healthy donor: high diversity with moderate evenness
                # Use low-variance lognormal for more even distribution
                donor = self.rng.lognormal(0, 0.5, len(x))
                donor = donor / np.sum(donor)
            x = (1 - intervention.intensity) * x + intervention.intensity * donor

        elif intervention.type == InterventionType.PREBIOTIC:
            boost_idx = self.rng.choice(len(x), size=len(x) // 4, replace=False)
            x[boost_idx] *= 1.5

        elif intervention.type == InterventionType.DIETARY_FIBER:
            self._decay_rate *= (1 - 0.5 * intervention.intensity)

        x = x / np.sum(x)
        self._state = MicrobiomeState(
            abundances=x,
            species_ids=self._state.species_ids,
            time=self._state.time,
        )

    # =========================================================================
    # VARIABLE MEASUREMENT (Outputs)
    # =========================================================================

    def get_k(self) -> int:
        """Get constraint count (number of detected species)."""
        if self._state is None:
            return 0
        return int(np.sum(self._state.abundances > 1e-6))

    def get_rho(self) -> float:
        """Get correlation (approximate SparCC)."""
        # In full implementation, compute from abundance time series
        return self._correlation_estimate

    def get_k_eff(self) -> float:
        """Get effective constraint count: k / (1 + rho*(k-1))."""
        k = self.get_k()
        rho = self.get_rho()
        if k <= 1:
            return float(k)
        denom = 1 + rho * (k - 1)
        return k / max(denom, 0.01)

    def get_sigma(self) -> float:
        """Get sustainability (normalized Shannon diversity)."""
        if self._state is None:
            return 0.0

        x = self._state.abundances
        x = x[x > 1e-10]

        if len(x) == 0:
            return 0.0

        H = -np.sum(x * np.log(x))
        H_max = np.log(len(x)) if len(x) > 1 else 1.0
        return H / H_max if H_max > 0 else 0.0

    def get_f(self) -> float:
        """Get compromise fraction (pathogen dominance)."""
        if self._state is None:
            return 0.0
        # Assume first 3 species are potential pathogens
        return float(np.sum(self._state.abundances[:3]))

    def get_state(self) -> np.ndarray:
        """Get full state vector."""
        if self._state is None:
            return np.array([])
        return self._state.abundances.copy()

    def is_collapsed(self) -> bool:
        """Check if ecosystem has collapsed."""
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
        self._state = None
        self._history = []
        self._collapsed = False
        self._collapse_time = None


def create_microbiome_engine(
    params: Optional[MicrobiomeParams] = None,
    seed: Optional[int] = None,
) -> MicrobiomeEngine:
    """Factory function to create a MicrobiomeEngine instance."""
    return MicrobiomeEngine(params=params, seed=seed)


__all__ = [
    'MicrobiomeEngine',
    'MicrobiomeParams',
    'MicrobiomeState',
    'MicrobiomeShock',
    'MicrobiomeIntervention',
    'ShockType',
    'InterventionType',
    'create_microbiome_engine',
]
