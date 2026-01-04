"""
RATCHET Institutional Collapse Simulation Engine

Simulates institutional dynamics and state fragility for structural invariant testing.
Theory-agnostic: exposes manipulable and measurable variables without
assuming relationships between them.

Domain Mapping:
    k (constraints):     Institutional constraints (constitutional, judicial, legislative)
    rho (correlation):   Elite network coupling / power concentration
    sigma (sustainability): Political stability / state capacity
    f (compromise):      Corruption / elite capture fraction
    d (decay rate):      Institutional erosion rate
    alpha (generation):  New constraint creation rate (reforms)
    lambda (strictness): Rule of law / enforcement strength

References:
    - V-Dem (Varieties of Democracy) dataset
    - Quality of Government (QoG) Standard Dataset
    - Polity V political regime data
    - PITF State Failure Problem Set
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class RegimeType(Enum):
    """Regime classification following Polity typology."""
    DEMOCRACY = "democracy"
    ANOCRACY = "anocracy"
    AUTOCRACY = "autocracy"
    FAILED = "failed"


class InstitutionalShockType(Enum):
    """Types of shocks to institutional systems."""
    ECONOMIC = "economic"
    CONFLICT = "conflict"
    NATURAL = "natural"
    POLITICAL = "political"
    PANDEMIC = "pandemic"


class InstitutionalInterventionType(Enum):
    """Types of institutional interventions."""
    REFORM = "reform"
    AID = "aid"
    SANCTION = "sanction"
    MILITARY = "military"
    DIPLOMATIC = "diplomatic"


@dataclass
class InstitutionalShock:
    """External shock to the institutional system."""
    type: InstitutionalShockType
    magnitude: float = 0.3
    target_variable: str = "sigma"
    duration: int = 1


@dataclass
class InstitutionalIntervention:
    """Policy or external intervention."""
    type: InstitutionalInterventionType
    intensity: float = 0.5
    target_variable: str = "k"
    delay: int = 0


@dataclass
class InstitutionalState:
    """State of the institutional system."""
    k: float            # Constraint count (0-1 normalized)
    rho: float          # Elite coupling (0-1)
    sigma: float        # Sustainability (0-1)
    f: float            # Compromise fraction (0-1)
    lambda_: float      # Strictness (0-1)
    time: float = 0.0
    country_code: Optional[str] = None
    year: Optional[int] = None

    @property
    def k_eff(self) -> float:
        """Effective constraint count."""
        if self.k <= 0.01:
            return 0.0
        k_scaled = self.k * 10
        denom = 1 + self.rho * (k_scaled - 1)
        return k_scaled / max(denom, 0.01)

    def copy(self) -> 'InstitutionalState':
        return InstitutionalState(
            k=self.k,
            rho=self.rho,
            sigma=self.sigma,
            f=self.f,
            lambda_=self.lambda_,
            time=self.time,
            country_code=self.country_code,
            year=self.year,
        )


@dataclass
class InstitutionalParams:
    """Configuration for InstitutionalCollapseEngine."""
    engine: str = "institutional"
    alpha: float = 0.02           # Constraint generation rate
    d: float = 0.03               # Decay rate
    collapse_threshold_sigma: float = 0.2
    collapse_threshold_f: float = 0.8
    noise_sigma: float = 0.01
    seed: Optional[int] = None


# Regime archetypes (empirically-based defaults)
REGIME_ARCHETYPES = {
    RegimeType.DEMOCRACY: {'k': 0.85, 'rho': 0.25, 'sigma': 0.80, 'f': 0.20, 'lambda_': 0.85},
    RegimeType.ANOCRACY: {'k': 0.50, 'rho': 0.50, 'sigma': 0.50, 'f': 0.50, 'lambda_': 0.50},
    RegimeType.AUTOCRACY: {'k': 0.20, 'rho': 0.80, 'sigma': 0.40, 'f': 0.70, 'lambda_': 0.60},
    RegimeType.FAILED: {'k': 0.10, 'rho': 0.90, 'sigma': 0.15, 'f': 0.85, 'lambda_': 0.15},
}


class InstitutionalCollapseEngine:
    """
    Institutional collapse / state fragility simulation engine.

    Simulates political institutional dynamics including constraint erosion,
    elite capture, and state capacity. Exposes RATCHET framework variables
    for manipulation and measurement.

    Example:
        >>> engine = InstitutionalCollapseEngine(seed=42)
        >>> engine.initialize_synthetic(RegimeType.ANOCRACY)
        >>> ts = engine.run(duration=50)
        >>> print(f"Collapsed: {engine.is_collapsed()}")
    """

    def __init__(
        self,
        params: Optional[InstitutionalParams] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize the InstitutionalCollapseEngine.

        Args:
            params: Engine configuration. If None, uses defaults.
            seed: Random seed for reproducibility.
        """
        self.params = params or InstitutionalParams()
        if seed is not None:
            self.params.seed = seed

        self.rng = np.random.default_rng(self.params.seed)
        self._state: Optional[InstitutionalState] = None
        self._history: List[Dict] = []

        # Dynamics parameters (manipulable)
        self._alpha = self.params.alpha
        self._d = self.params.d

        # Collapse tracking
        self._collapsed = False
        self._collapse_time: Optional[float] = None

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def initialize_synthetic(
        self,
        regime_type: RegimeType = RegimeType.ANOCRACY,
        noise: bool = True,
    ) -> None:
        """Initialize from regime archetype with optional noise."""
        archetype = REGIME_ARCHETYPES[regime_type].copy()

        if noise:
            for key in archetype:
                archetype[key] += self.rng.normal(0, self.params.noise_sigma)
                archetype[key] = np.clip(archetype[key], 0, 1)

        self._state = InstitutionalState(**archetype)
        self._history = [self._record_state()]
        self._collapsed = False
        self._collapse_time = None

    def set_state(self, state: InstitutionalState) -> None:
        """Set the full state vector directly."""
        self._state = state.copy()
        self._history = [self._record_state()]
        self._collapsed = False
        self._collapse_time = None

    def initialize_manual(
        self,
        k: float = 0.5,
        rho: float = 0.5,
        sigma: float = 0.5,
        f: float = 0.3,
        lambda_: float = 0.5,
        country_code: Optional[str] = None,
        year: Optional[int] = None,
    ) -> None:
        """Initialize with explicit values."""
        self._state = InstitutionalState(
            k=np.clip(k, 0, 1),
            rho=np.clip(rho, 0, 1),
            sigma=np.clip(sigma, 0, 1),
            f=np.clip(f, 0, 1),
            lambda_=np.clip(lambda_, 0, 1),
            time=0.0,
            country_code=country_code,
            year=year,
        )
        self._history = [self._record_state()]
        self._collapsed = False
        self._collapse_time = None

    # =========================================================================
    # CORE SIMULATION
    # =========================================================================

    def step(self, dt: float = 1.0) -> None:
        """Advance simulation by one time step (in years)."""
        if self._state is None:
            raise RuntimeError("Engine not initialized")

        if self._collapsed:
            return

        # Sustainability decay
        decay = self._d * dt
        noise = self.rng.normal(0, self.params.noise_sigma)
        self._state.sigma = np.clip(self._state.sigma - decay + noise, 0, 1)

        # Constraint generation/erosion
        constraint_change = self._alpha * dt
        self._state.k = np.clip(
            self._state.k + constraint_change + self.rng.normal(0, self.params.noise_sigma),
            0, 1
        )

        # Corruption tends to increase slightly without active anti-corruption
        corruption_drift = 0.005 * dt
        self._state.f = np.clip(
            self._state.f + corruption_drift + self.rng.normal(0, self.params.noise_sigma / 2),
            0, 1
        )

        # Elite coupling drifts toward higher concentration (entropy-like)
        coupling_drift = 0.002 * dt
        self._state.rho = np.clip(
            self._state.rho + coupling_drift + self.rng.normal(0, self.params.noise_sigma / 2),
            0, 1
        )

        # Update time
        self._state.time += dt
        if self._state.year is not None:
            self._state.year += int(dt)

        self._history.append(self._record_state())
        self._check_collapse()

    def _record_state(self) -> Dict:
        """Record current state for history."""
        return {
            'time': self._state.time,
            'year': self._state.year,
            'country_code': self._state.country_code,
            'k': self._state.k,
            'rho': self._state.rho,
            'k_eff': self._state.k_eff,
            'sigma': self._state.sigma,
            'f': self._state.f,
            'lambda': self._state.lambda_,
            'alpha': self._alpha,
            'd': self._d,
            'collapsed': self._collapsed,
        }

    def _check_collapse(self) -> None:
        """Check if system has collapsed."""
        if self._collapsed:
            return

        if (self._state.sigma < self.params.collapse_threshold_sigma or
                self._state.f > self.params.collapse_threshold_f):
            self._collapsed = True
            self._collapse_time = self._state.time

    def run(self, duration: float, dt: float = 1.0) -> pd.DataFrame:
        """Run simulation for specified duration (in years)."""
        n_steps = int(duration / dt)
        for _ in range(n_steps):
            self.step(dt)
            if self._collapsed:
                break
        return self.to_dataframe()

    def run_until_collapse(self, max_duration: float = 100, dt: float = 1.0) -> pd.DataFrame:
        """Run until collapse or max duration."""
        return self.run(max_duration, dt)

    # =========================================================================
    # VARIABLE MANIPULATION (Inputs)
    # =========================================================================

    def set_k(self, k: float) -> None:
        """Set constraint count (normalized 0-1)."""
        if self._state is None:
            raise RuntimeError("Engine not initialized")
        self._state.k = np.clip(k, 0, 1)

    def set_alpha(self, alpha: float) -> None:
        """Set constraint generation rate."""
        self._alpha = max(0, alpha)

    def set_d(self, d: float) -> None:
        """Set decay rate."""
        self._d = max(0, d)

    def set_lambda(self, lambda_: float) -> None:
        """Set strictness (rule of law)."""
        if self._state is None:
            raise RuntimeError("Engine not initialized")
        self._state.lambda_ = np.clip(lambda_, 0, 1)

    def apply_shock(self, shock: InstitutionalShock) -> None:
        """Apply external shock to system."""
        if self._state is None:
            raise RuntimeError("Engine not initialized")

        target = shock.target_variable
        magnitude = shock.magnitude

        if target == 'sigma':
            self._state.sigma = max(0, self._state.sigma - magnitude)
        elif target == 'k':
            self._state.k = max(0, self._state.k - magnitude)
        elif target == 'lambda':
            self._state.lambda_ = max(0, self._state.lambda_ - magnitude)
        elif target == 'f':
            self._state.f = min(1, self._state.f + magnitude)
        elif target == 'rho':
            self._state.rho = min(1, self._state.rho + magnitude)

    def apply_intervention(self, intervention: InstitutionalIntervention) -> None:
        """Apply policy intervention."""
        if self._state is None:
            raise RuntimeError("Engine not initialized")

        effect = intervention.intensity

        if intervention.type == InstitutionalInterventionType.REFORM:
            if intervention.target_variable == 'k':
                self._state.k = min(1, self._state.k + effect * 0.1)
            elif intervention.target_variable == 'lambda':
                self._state.lambda_ = min(1, self._state.lambda_ + effect * 0.1)
            elif intervention.target_variable == 'f':
                self._state.f = max(0, self._state.f - effect * 0.1)

        elif intervention.type == InstitutionalInterventionType.AID:
            self._state.sigma = min(1, self._state.sigma + effect * 0.05)

        elif intervention.type == InstitutionalInterventionType.SANCTION:
            self._state.sigma = max(0, self._state.sigma - effect * 0.1)

        elif intervention.type == InstitutionalInterventionType.DIPLOMATIC:
            # Diplomatic pressure slightly reduces corruption
            self._state.f = max(0, self._state.f - effect * 0.02)

        elif intervention.type == InstitutionalInterventionType.MILITARY:
            # Military intervention is disruptive
            self._state.sigma = max(0, self._state.sigma - effect * 0.2)
            self._state.k = max(0, self._state.k - effect * 0.1)

    # =========================================================================
    # VARIABLE MEASUREMENT (Outputs)
    # =========================================================================

    def get_k(self) -> float:
        """Get constraint count (0-1 normalized)."""
        if self._state is None:
            return 0.0
        return self._state.k

    def get_rho(self) -> float:
        """Get elite coupling."""
        if self._state is None:
            return 0.0
        return self._state.rho

    def get_k_eff(self) -> float:
        """Get effective constraint count."""
        if self._state is None:
            return 0.0
        return self._state.k_eff

    def get_sigma(self) -> float:
        """Get sustainability (political stability)."""
        if self._state is None:
            return 0.0
        return self._state.sigma

    def get_f(self) -> float:
        """Get compromise fraction (corruption)."""
        if self._state is None:
            return 0.0
        return self._state.f

    def get_state(self) -> np.ndarray:
        """Get state as numpy array [k, rho, sigma, f, lambda]."""
        if self._state is None:
            return np.zeros(5)
        return np.array([
            self._state.k,
            self._state.rho,
            self._state.sigma,
            self._state.f,
            self._state.lambda_,
        ])

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
        self._state = None
        self._history = []
        self._collapsed = False
        self._collapse_time = None


def create_institutional_engine(
    params: Optional[InstitutionalParams] = None,
    seed: Optional[int] = None,
) -> InstitutionalCollapseEngine:
    """Factory function to create an InstitutionalCollapseEngine instance."""
    return InstitutionalCollapseEngine(params=params, seed=seed)


__all__ = [
    'InstitutionalCollapseEngine',
    'InstitutionalParams',
    'InstitutionalState',
    'InstitutionalShock',
    'InstitutionalIntervention',
    'InstitutionalShockType',
    'InstitutionalInterventionType',
    'RegimeType',
    'REGIME_ARCHETYPES',
    'create_institutional_engine',
]
