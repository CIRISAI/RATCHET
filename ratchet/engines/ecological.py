"""
RATCHET Macro-Ecology (BioTIME) Substrate Engine

Simulates community-level ecological dynamics for a fixed set of species
on annual time steps. Exposes the standard RATCHET (k, rho, sigma, k_eff)
interface and produces biomass trajectories matched in shape to BioTIME
2.0 community time-series data.

Domain mapping (per REGIME.md §"A2 — BioTIME macro-ecology"):
    k     : Species count in a community time series
    rho   : Mean pairwise correlation of species-abundance time series
    sigma : Inverse CV of total biomass over time (stability proxy)
    f     : 1 − sigma (compromise / instability fraction)

Dynamics
--------
For each species i ∈ [0, k):
    x_{i,t+1} = x_{i,t} + r_i · x_{i,t} · (1 − x_{i,t} / K_i)
              + Σ_j C[i,j] · (x_{j,t} / K_j) · K_i
              + ε_env · ξ_t · K_i        (common env forcing)
              + obs_noise

The community-level *correlation* ρ emerges from two coupled sources:
  • the off-diagonal coupling matrix C (per-pair species interactions)
  • the common environmental forcing ξ_t (drives baseline correlation)

This mirrors the BioTIME-like SyntheticBioTIMEGenerator in
`ratchet.data.ecological_loader`, so a synthetic→engine round-trip is a
*fair* fit test — the engine exercises the same Kish dynamics the
generator uses, just with parameters discovered from data rather than
fixed in advance.

Pairs with: `ratchet/engines/{battery,institutional,microbiome}.py`.

References
----------
- Dornelas, M., et al. (2025). BioTIME 2.0. Global Ecology and Biogeography.
- Loreau, M., & de Mazancourt, C. (2013). Biodiversity and ecosystem stability.
  Ecology Letters, 16(s1), 106-115.
- Tilman, D. (1996). Biodiversity: Population versus ecosystem stability.
  Ecology, 77(2), 350-363.
- Coyte, K. Z., Schluter, J., & Foster, K. R. (2015). The ecology of the
  microbiome: networks, competition, and stability. Science, 350, 663-666.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────
# Shock / intervention enums (parallel BatteryShockType / MicrobiomeShockType)
# ─────────────────────────────────────────────────────────────────────────


class EcologicalShockType(Enum):
    """Types of perturbations to a community."""
    CLIMATE = "climate"          # broad env-forcing amplitude bump
    HABITAT_LOSS = "habitat_loss"  # carrying-capacity reduction
    INVASIVE = "invasive"          # add a strong-coupling species
    HARVEST = "harvest"            # abundance pull-down across species
    POLLUTION = "pollution"        # increase mortality / noise


class EcologicalInterventionType(Enum):
    """Types of conservation interventions."""
    REINTRODUCTION = "reintroduction"     # restore extirpated species
    HABITAT_RESTORATION = "habitat"       # raise K_i
    INVASIVE_REMOVAL = "invasive_removal" # weaken coupling row
    PROTECTED_AREA = "protected_area"     # reduce env forcing & noise


@dataclass
class EcologicalShock:
    """External perturbation to the ecological community."""
    type: EcologicalShockType
    magnitude: float = 0.3
    target_species: Optional[int] = None  # None = community-wide
    duration: float = 1.0  # years


@dataclass
class EcologicalIntervention:
    """Conservation intervention on the community."""
    type: EcologicalInterventionType
    parameters: Dict = field(default_factory=dict)


@dataclass
class EcologicalParams:
    """Configuration for EcologicalCommunityEngine.

    Defaults follow the synthetic-BioTIME parameterisation in
    `ratchet.data.ecological_loader.SyntheticBioTIMEGenerator`. Pass
    `seed` for reproducibility; pass `n_species`/`coupling_strength`/
    `env_forcing_amp` for forced initialisation (e.g. matching a real
    community's characteristics).
    """
    engine: str = "ecological"
    n_species: int = 10
    n_years_default: int = 30

    # Intrinsic dynamics
    intrinsic_growth_mean: float = 0.4
    intrinsic_growth_std: float = 0.1
    carrying_capacity_mean: float = 33.0   # exp(3.5) ≈ 33
    carrying_capacity_std: float = 0.5     # log-space sigma
    obs_noise_frac: float = 0.05           # noise = obs_noise_frac · K

    # Cross-species coupling
    coupling_strength: float = 0.05

    # Environmental forcing
    env_forcing_amp: float = 0.15
    env_ar_phi: float = 0.6

    # Collapse criterion
    sigma_collapse_threshold: float = 0.10  # very low stability ⇒ collapse

    seed: Optional[int] = None


# ─────────────────────────────────────────────────────────────────────────
# Per-species state container
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class SpeciesState:
    species_id: str
    abundance: float
    carrying_capacity: float
    intrinsic_growth: float


# ─────────────────────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────────────────────


class EcologicalCommunityEngine:
    """Population-dynamic ecological community engine.

    Example
    -------
    >>> engine = EcologicalCommunityEngine(seed=42)
    >>> engine.initialize()
    >>> df = engine.run(duration=30, dt=1.0)
    >>> print(engine.get_k(), engine.get_rho(), engine.get_sigma())
    """

    def __init__(
        self,
        params: Optional[EcologicalParams] = None,
        seed: Optional[int] = None,
    ):
        self.params = params or EcologicalParams()
        if seed is not None:
            self.params.seed = seed

        self.rng = np.random.default_rng(self.params.seed)
        self.time = 0.0           # in years
        self.step_count = 0

        self.species: List[SpeciesState] = []
        self._coupling_matrix: Optional[np.ndarray] = None
        self._env_signal: float = 0.0  # current env state
        self._env_history: List[float] = []
        self._abundance_history: List[np.ndarray] = []  # (n_species,) per step
        self._biomass_history: List[float] = []
        self._history: List[Dict] = []

        self._collapsed = False
        self._collapse_time: Optional[float] = None

    # ── initialisation ───────────────────────────────────────────────────

    def initialize(self) -> None:
        """Initialise all species + coupling matrix + env state to fresh values."""
        p = self.params
        n = p.n_species

        # Per-species params
        r = np.clip(self.rng.normal(p.intrinsic_growth_mean,
                                    p.intrinsic_growth_std, size=n), 0.1, 0.9)
        K = np.clip(self.rng.lognormal(mean=np.log(max(p.carrying_capacity_mean, 1.0)),
                                        sigma=p.carrying_capacity_std, size=n),
                    5.0, 200.0)
        x0 = self.rng.uniform(0.3 * K, 0.7 * K)

        self.species = [
            SpeciesState(
                species_id=f"sp_{i:03d}",
                abundance=float(x0[i]),
                carrying_capacity=float(K[i]),
                intrinsic_growth=float(r[i]),
            )
            for i in range(n)
        ]

        # Coupling matrix (symmetric, zero diag, σ = coupling_strength)
        C = self.rng.normal(0.0, p.coupling_strength, size=(n, n))
        C = 0.5 * (C + C.T)
        np.fill_diagonal(C, 0.0)
        self._coupling_matrix = C

        # Initialise env signal
        self._env_signal = float(self.rng.normal(0, 1))

        # Reset history
        self._env_history = [self._env_signal]
        self._abundance_history = [self._get_abundance_vector()]
        self._biomass_history = [float(np.sum(self._abundance_history[0]))]
        self.time = 0.0
        self.step_count = 0
        self._collapsed = False
        self._collapse_time = None
        self._history = [self._record_state()]

    # ── helpers ──────────────────────────────────────────────────────────

    def _get_abundance_vector(self) -> np.ndarray:
        return np.array([s.abundance for s in self.species], dtype=float)

    def _get_K_vector(self) -> np.ndarray:
        return np.array([s.carrying_capacity for s in self.species], dtype=float)

    def _get_r_vector(self) -> np.ndarray:
        return np.array([s.intrinsic_growth for s in self.species], dtype=float)

    # ── core simulation ──────────────────────────────────────────────────

    def step(self, dt: float = 1.0) -> None:
        """Advance simulation by `dt` years (default 1)."""
        if not self.species:
            raise RuntimeError("Engine not initialized. Call initialize() first.")
        if self._collapsed:
            return

        p = self.params
        n = len(self.species)

        # Step env signal (AR(1) walk)
        self._env_signal = (p.env_ar_phi * self._env_signal
                            + self.rng.normal(0, 1) * np.sqrt(1.0 - p.env_ar_phi ** 2))
        env_term_scalar = p.env_forcing_amp * self._env_signal

        x_prev = self._get_abundance_vector()
        K = self._get_K_vector()
        r = self._get_r_vector()
        C = self._coupling_matrix  # type: ignore[assignment]

        # Logistic growth
        logistic = r * x_prev * (1.0 - x_prev / np.maximum(K, 1e-6)) * dt

        # Cross-species coupling
        coupling = (C @ (x_prev / np.maximum(K, 1e-6))) * K * dt

        # Common env forcing
        env_term = env_term_scalar * K * dt

        # Observation noise (proportional to K)
        noise = self.rng.normal(0.0, p.obs_noise_frac, size=n) * K * np.sqrt(dt)

        x_next = x_prev + logistic + coupling + env_term + noise
        x_next = np.clip(x_next, 1e-3, 5.0 * K)

        for i, s in enumerate(self.species):
            s.abundance = float(x_next[i])

        self.time += dt
        self.step_count += 1
        self._env_history.append(self._env_signal)
        self._abundance_history.append(self._get_abundance_vector())
        self._biomass_history.append(float(np.sum(x_next)))
        self._history.append(self._record_state())
        self._check_collapse()

    def run(self, duration: float = 30.0, dt: float = 1.0) -> pd.DataFrame:
        """Run for `duration` years, returning the per-year history dataframe."""
        n_steps = int(round(duration / dt))
        for _ in range(n_steps):
            self.step(dt)
            if self._collapsed:
                break
        return self.to_dataframe()

    def _record_state(self) -> Dict:
        return {
            "time": self.time,
            "step": self.step_count,
            "k": self.get_k(),
            "k_eff": self.get_k_eff(),
            "rho": self.get_rho(),
            "sigma": self.get_sigma(),
            "f": self.get_f(),
            "biomass_total": float(np.sum(self._get_abundance_vector())),
            "env_signal": float(self._env_signal),
            "collapsed": self._collapsed,
        }

    def _check_collapse(self) -> None:
        if self._collapsed:
            return
        if self.get_sigma() < self.params.sigma_collapse_threshold and self.step_count >= 5:
            self._collapsed = True
            self._collapse_time = self.time

    # ── manipulation ─────────────────────────────────────────────────────

    def set_n_species(self, n: int) -> None:
        """Reset engine with a new species count (calls initialize)."""
        self.params.n_species = max(2, min(60, int(n)))
        self.initialize()

    def set_coupling_strength(self, cs: float) -> None:
        """Set coupling strength and resample coupling matrix."""
        self.params.coupling_strength = max(0.0, min(0.5, float(cs)))
        if self.species:
            n = len(self.species)
            C = self.rng.normal(0.0, self.params.coupling_strength, size=(n, n))
            C = 0.5 * (C + C.T)
            np.fill_diagonal(C, 0.0)
            self._coupling_matrix = C

    def set_env_forcing(self, amp: float) -> None:
        """Set environmental forcing amplitude."""
        self.params.env_forcing_amp = max(0.0, min(1.0, float(amp)))

    def apply_shock(self, shock: EcologicalShock) -> None:
        if not self.species:
            raise RuntimeError("Engine not initialized")
        n = len(self.species)
        if shock.type == EcologicalShockType.CLIMATE:
            self.params.env_forcing_amp = min(1.0, self.params.env_forcing_amp + shock.magnitude)
        elif shock.type == EcologicalShockType.HABITAT_LOSS:
            for s in self.species:
                s.carrying_capacity *= max(0.05, 1.0 - shock.magnitude)
        elif shock.type == EcologicalShockType.INVASIVE:
            # Add one strong-coupling row+column to existing coupling matrix
            self.species.append(SpeciesState(
                species_id=f"sp_inv_{n:03d}",
                abundance=10.0,
                carrying_capacity=50.0,
                intrinsic_growth=0.6,
            ))
            new_n = n + 1
            C_new = np.zeros((new_n, new_n))
            C_new[:n, :n] = self._coupling_matrix
            # Strong coupling to most species
            new_row = self.rng.normal(0.0, shock.magnitude, size=n)
            C_new[n, :n] = new_row
            C_new[:n, n] = new_row
            self._coupling_matrix = C_new
            self.params.n_species = new_n
        elif shock.type == EcologicalShockType.HARVEST:
            for s in self.species:
                s.abundance *= max(0.05, 1.0 - shock.magnitude)
        elif shock.type == EcologicalShockType.POLLUTION:
            self.params.obs_noise_frac = min(0.5, self.params.obs_noise_frac + shock.magnitude * 0.1)

    def apply_intervention(self, intervention: EcologicalIntervention) -> None:
        if not self.species:
            raise RuntimeError("Engine not initialized")
        if intervention.type == EcologicalInterventionType.HABITAT_RESTORATION:
            boost = float(intervention.parameters.get("boost", 0.1))
            for s in self.species:
                s.carrying_capacity *= (1.0 + boost)
        elif intervention.type == EcologicalInterventionType.PROTECTED_AREA:
            self.params.env_forcing_amp *= 0.5
            self.params.obs_noise_frac *= 0.5
        elif intervention.type == EcologicalInterventionType.INVASIVE_REMOVAL:
            target = int(intervention.parameters.get("species_index", len(self.species) - 1))
            if 0 <= target < len(self.species):
                # Zero out the coupling row/col for that species
                C = self._coupling_matrix
                if C is not None:
                    C[target, :] = 0
                    C[:, target] = 0
        elif intervention.type == EcologicalInterventionType.REINTRODUCTION:
            self.species.append(SpeciesState(
                species_id=f"sp_re_{len(self.species):03d}",
                abundance=float(intervention.parameters.get("abundance", 5.0)),
                carrying_capacity=float(intervention.parameters.get("K", 30.0)),
                intrinsic_growth=float(intervention.parameters.get("r", 0.4)),
            ))
            n_new = len(self.species)
            C_new = np.zeros((n_new, n_new))
            if self._coupling_matrix is not None:
                old_n = self._coupling_matrix.shape[0]
                C_new[:old_n, :old_n] = self._coupling_matrix
            self._coupling_matrix = C_new
            self.params.n_species = n_new

    # ── measurement (RATCHET-uniform) ────────────────────────────────────

    def get_k(self) -> int:
        return len(self.species)

    def get_rho(self) -> float:
        """Mean abs pairwise correlation across species abundance histories."""
        if len(self._abundance_history) < 3 or len(self.species) < 2:
            return 0.0
        # (n_species, n_timepoints)
        A = np.array(self._abundance_history).T
        n_sp, n_t = A.shape
        if n_t < 2:
            return 0.0
        pairs = []
        for i in range(n_sp):
            if np.std(A[i]) < 1e-10:
                continue
            for j in range(i + 1, n_sp):
                if np.std(A[j]) < 1e-10:
                    continue
                r = np.corrcoef(A[i], A[j])[0, 1]
                if not np.isnan(r):
                    pairs.append(abs(float(r)))
        if not pairs:
            return 0.0
        return float(np.mean(pairs))

    def get_sigma(self) -> float:
        """Inverse-CV stability of total biomass trajectory ∈ (0, 1]."""
        if len(self._biomass_history) < 2:
            return 1.0
        b = np.array(self._biomass_history)
        mu = float(np.mean(b))
        if mu <= 1e-10:
            return 0.0
        sd = float(np.std(b))
        cv = sd / mu
        return float(1.0 / (1.0 + cv))

    def get_f(self) -> float:
        return float(max(0.0, 1.0 - self.get_sigma()))

    def get_k_eff(self) -> float:
        k = self.get_k()
        rho = self.get_rho()
        if k <= 1:
            return float(k)
        return k / (1.0 + rho * (k - 1))

    def get_biomass_trajectory(self) -> np.ndarray:
        """Per-year total biomass array."""
        return np.array(self._biomass_history, dtype=float)

    def get_abundance_matrix(self) -> np.ndarray:
        """(n_species, n_timepoints) abundance matrix."""
        if not self._abundance_history:
            return np.zeros((self.get_k(), 0))
        return np.array(self._abundance_history).T

    def is_collapsed(self) -> bool:
        return self._collapsed

    def get_collapse_time(self) -> Optional[float]:
        return self._collapse_time

    # ── export ───────────────────────────────────────────────────────────

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._history)

    def reset(self) -> None:
        self.species = []
        self._coupling_matrix = None
        self._env_signal = 0.0
        self._env_history = []
        self._abundance_history = []
        self._biomass_history = []
        self._history = []
        self.time = 0.0
        self.step_count = 0
        self._collapsed = False
        self._collapse_time = None


# ─────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────


def create_ecological_engine(
    params: Optional[EcologicalParams] = None,
    seed: Optional[int] = None,
) -> EcologicalCommunityEngine:
    """Factory function to create an EcologicalCommunityEngine."""
    return EcologicalCommunityEngine(params=params, seed=seed)


__all__ = [
    "EcologicalCommunityEngine",
    "EcologicalParams",
    "SpeciesState",
    "EcologicalShock",
    "EcologicalIntervention",
    "EcologicalShockType",
    "EcologicalInterventionType",
    "create_ecological_engine",
]
