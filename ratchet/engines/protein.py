"""
RATCHET Protein-Folding (AlphaFold) Substrate Engine

Simulates per-residue structural-confidence dynamics for a fixed-length
single-domain protein. Exposes the standard RATCHET (k, rho, sigma, k_eff)
interface and produces a pLDDT trajectory matched in shape to AlphaFold
DB v6 per-residue confidence data.

Domain mapping (per REGIME.md §"A0 — AlphaFold residues"):
    k     : Sequence length (residue count) of a single-domain protein
    rho   : Mean pairwise correlation of per-residue B-factor predictions
            (computed from pLDDT covariance across residues)
    sigma : Mean pLDDT score (structural stability proxy), bounded to (0, 1]
    f     : 1 − sigma (compromise / instability fraction)

Dynamics
--------
Each residue i ∈ [0, k) carries a confidence variable c_i ∈ [0, 100]
(pLDDT-like). Confidences evolve under three coupled forces:

  • Per-residue mean-reversion to an intrinsic stability c_target_i
    (intrinsic-folding driver).
  • Local sequence-neighbour coupling: neighbouring residues with
    sequence distance ≤ L exert a stabilising pull toward the local
    mean (induces exponential-decay spatial correlation in the
    steady state, matching AlphaFold-DB statistics).
  • Non-local coupling: a small fraction of residue pairs are
    "long-range structural contacts" (β-sheets, salt bridges, etc.)
    with stronger pairwise coupling. These create the ρ above what
    pure sequence neighbours produce.

Per step:
    c_{i,t+1} = c_{i,t}
              + γ · (c_target_i − c_{i,t})           (mean-reversion)
              + (η_local) · ⟨c_{j,t} − c_{i,t}⟩_{|j-i|≤L}  (sequence neighbours)
              + (η_lr)  · Σ_{j ∈ contacts(i)} (c_{j,t} − c_{i,t}) / |contacts(i)|
              + ε_residue · ξ_i,t                    (thermal fluctuation)
              + ε_global · ξ_t                       (common env mode)

Steady-state covariance of c reproduces an exponential-decay spatial
correlation pattern + a small non-local component, matching what the
SyntheticAlphaFoldGenerator in `ratchet.data.protein_loader` emits — so
a synthetic→engine round-trip is a *fair* fit test.

Pairs with: `ratchet/engines/{battery,institutional,microbiome,ecological}.py`.

References
----------
- Jumper, J., et al. (2021). Highly accurate protein structure prediction
  with AlphaFold. Nature, 596, 583-589.
- Varadi, M., et al. (2024). AlphaFold Protein Structure Database in 2024.
  Nucleic Acids Research, 52, D368-D375.
- Sillitoe, I., et al. (2021). CATH: increased structural coverage of
  functional space. Nucleic Acids Research, 49, D266-D273.
- Mariani, V., et al. (2013). lDDT: a local superposition-free score for
  comparing protein structures. Bioinformatics, 29(21), 2722-2728.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────
# Shock / intervention enums (parallel EcologicalShockType / BatteryShockType)
# ─────────────────────────────────────────────────────────────────────────


class ProteinShockType(Enum):
    """Types of perturbations to a protein fold."""
    MUTATION = "mutation"            # single-residue confidence drop
    DENATURATION = "denaturation"    # global confidence loss (heat/chaotrope)
    LIGAND_BINDING = "ligand_binding"  # increased stability in a region
    OXIDATIVE_STRESS = "oxidative_stress"  # noise increase
    PROTEOLYSIS = "proteolysis"      # truncate sequence


class ProteinInterventionType(Enum):
    """Types of stabilising interventions."""
    CHAPERONE = "chaperone"             # boost mean-reversion strength
    DISULFIDE = "disulfide"             # add a long-range contact
    OSMOLYTE = "osmolyte"               # reduce noise globally
    STABILISING_MUTATION = "stabilising_mutation"  # raise c_target locally


@dataclass
class ProteinShock:
    """External perturbation to the protein."""
    type: ProteinShockType
    magnitude: float = 0.3
    target_residue: Optional[int] = None  # None = global
    duration: float = 1.0  # in engine steps


@dataclass
class ProteinIntervention:
    """Stabilising intervention on the protein."""
    type: ProteinInterventionType
    parameters: Dict = field(default_factory=dict)


@dataclass
class ProteinParams:
    """Configuration for ProteinFoldingEngine.

    Defaults follow the synthetic-AlphaFold parameterisation in
    `ratchet.data.protein_loader.SyntheticAlphaFoldGenerator`. Pass
    `seed` for reproducibility; pass `n_residues`/`correlation_length`/
    `mean_plddt_target` for forced initialisation (e.g. matching a real
    protein's characteristics).
    """
    engine: str = "protein"
    n_residues: int = 150
    n_steps_default: int = 100

    # Intrinsic per-residue dynamics
    mean_plddt_target: float = 85.0          # mean of c_target_i
    target_plddt_std: float = 5.0            # cross-residue std of target
    mean_reversion_rate: float = 0.10        # γ per step

    # Local sequence-neighbour coupling
    local_coupling_window: int = 5           # residues within ±window are "local"
    local_coupling_strength: float = 0.20    # η_local per step (per-neighbour mean)

    # Non-local (structural contact) coupling
    n_long_range_contacts_per_residue: float = 0.5  # mean LR contacts per residue
    long_range_coupling_strength: float = 0.05      # η_lr per step

    # Effective correlation length (sets the local neighbour-window weighting)
    correlation_length: float = 14.0

    # Noise
    residue_noise_std: float = 4.0     # per-residue thermal noise (pLDDT units)
    global_noise_std: float = 1.0      # common environmental mode

    # Collapse criterion: sigma below this is considered unfolded/collapsed
    sigma_collapse_threshold: float = 0.30  # mean pLDDT < 30

    seed: Optional[int] = None


# ─────────────────────────────────────────────────────────────────────────
# Per-residue state container
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class ResidueState:
    """Per-residue state in the protein engine."""
    residue_id: int
    confidence: float        # current pLDDT-like value in [0, 100]
    target_confidence: float  # intrinsic mean to revert toward
    is_contact: bool = False  # whether this residue has a long-range partner


# ─────────────────────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────────────────────


class ProteinFoldingEngine:
    """Per-residue protein-folding dynamics engine.

    Example
    -------
    >>> engine = ProteinFoldingEngine(seed=42)
    >>> engine.initialize()
    >>> df = engine.run(duration=100)
    >>> print(engine.get_k(), engine.get_rho(), engine.get_sigma())
    """

    def __init__(
        self,
        params: Optional[ProteinParams] = None,
        seed: Optional[int] = None,
    ):
        self.params = params or ProteinParams()
        if seed is not None:
            self.params.seed = seed

        self.rng = np.random.default_rng(self.params.seed)
        self.time = 0.0                   # in engine steps
        self.step_count = 0

        self.residues: List[ResidueState] = []
        self._contact_map: Optional[np.ndarray] = None  # (k, k) bool, symmetric
        self._neighbour_kernel: Optional[np.ndarray] = None  # (k, k) float

        self._env_signal: float = 0.0     # current global env mode
        self._env_history: List[float] = []
        self._confidence_history: List[np.ndarray] = []  # (k,) per step
        self._mean_plddt_history: List[float] = []
        self._history: List[Dict] = []

        self._collapsed = False
        self._collapse_time: Optional[float] = None

        # Cached rho / k_eff (computed lazily; updated each step from a fast
        # proxy or at end of run() from the full O(k^2 · t) accessor).
        self._cached_rho: float = 0.0
        self._cached_k_eff: float = 0.0

    # ── initialisation ───────────────────────────────────────────────────

    def initialize(self) -> None:
        """Initialise residues + contact map + env state to fresh values."""
        p = self.params
        n = p.n_residues

        # Per-residue intrinsic confidence targets (cross-residue variation)
        c_target = np.clip(
            self.rng.normal(p.mean_plddt_target, p.target_plddt_std, size=n),
            30.0, 99.0,
        )
        # Initial confidence near target (some thermal jitter)
        c_init = np.clip(c_target + self.rng.normal(0, p.residue_noise_std, size=n),
                         0.0, 100.0)

        self.residues = [
            ResidueState(
                residue_id=i,
                confidence=float(c_init[i]),
                target_confidence=float(c_target[i]),
            )
            for i in range(n)
        ]

        # Sequence-neighbour kernel (exponential decay within window)
        L = max(p.correlation_length, 1.0)
        W = max(p.local_coupling_window, 1)
        idx = np.arange(n)
        dist = np.abs(idx[:, None] - idx[None, :])
        kernel = np.where(dist <= W, np.exp(-dist / L), 0.0)
        np.fill_diagonal(kernel, 0.0)
        # Row-normalise so coupling magnitude is window-size invariant
        row_sum = kernel.sum(axis=1, keepdims=True)
        kernel = np.divide(kernel, row_sum, out=np.zeros_like(kernel), where=row_sum > 0)
        self._neighbour_kernel = kernel

        # Long-range contact map (sparse symmetric Bernoulli)
        # Probability tuned so expected contacts per residue ~ n_long_range_contacts_per_residue
        p_lr = min(0.5, float(p.n_long_range_contacts_per_residue) / max(n - 1, 1))
        upper = self.rng.random((n, n)) < p_lr
        # Don't count immediate sequence neighbours as long-range contacts
        for d in range(-W, W + 1):
            np.fill_diagonal(upper[max(0, d):, max(0, -d):], False)
        upper = np.triu(upper, k=W + 1)
        contact = upper | upper.T
        np.fill_diagonal(contact, False)
        self._contact_map = contact

        # Flag residues that participate in any contact
        any_contact = contact.any(axis=1)
        for i, s in enumerate(self.residues):
            s.is_contact = bool(any_contact[i])

        # Initialise env signal
        self._env_signal = float(self.rng.normal(0, 1))

        # Reset history
        self._env_history = [self._env_signal]
        self._confidence_history = [self._get_confidence_vector()]
        self._mean_plddt_history = [float(np.mean(self._confidence_history[0]))]
        self.time = 0.0
        self.step_count = 0
        self._collapsed = False
        self._collapse_time = None
        # Seed the cached rho/k_eff from the initial confidence vector
        self._refresh_cached_rho_kEff()
        self._history = [self._record_state()]

    # ── helpers ──────────────────────────────────────────────────────────

    def _get_confidence_vector(self) -> np.ndarray:
        return np.array([s.confidence for s in self.residues], dtype=float)

    def _get_target_vector(self) -> np.ndarray:
        return np.array([s.target_confidence for s in self.residues], dtype=float)

    # ── core simulation ──────────────────────────────────────────────────

    def step(self, dt: float = 1.0) -> None:
        """Advance simulation by `dt` time units (default 1 step)."""
        if not self.residues:
            raise RuntimeError("Engine not initialized. Call initialize() first.")
        if self._collapsed:
            return

        p = self.params
        n = len(self.residues)

        # Step env signal (AR(1)-style with phi=0.7)
        phi = 0.7
        self._env_signal = (
            phi * self._env_signal
            + self.rng.normal(0, 1) * np.sqrt(max(1.0 - phi ** 2, 1e-6))
        )
        env_term_scalar = p.global_noise_std * self._env_signal

        c_prev = self._get_confidence_vector()
        c_target = self._get_target_vector()
        kernel = self._neighbour_kernel
        contact = self._contact_map

        # Mean reversion (per-residue intrinsic stability)
        reversion = p.mean_reversion_rate * (c_target - c_prev) * dt

        # Local sequence-neighbour coupling: pull toward weighted neighbour mean
        if kernel is not None:
            local_mean = kernel @ c_prev   # (n,) weighted neighbour confidence
            local_term = p.local_coupling_strength * (local_mean - c_prev) * dt
        else:
            local_term = np.zeros(n)

        # Long-range (structural contact) coupling
        if contact is not None and contact.any():
            contact_counts = contact.sum(axis=1).astype(float)  # (n,)
            with np.errstate(divide="ignore", invalid="ignore"):
                contact_means = np.where(
                    contact_counts > 0,
                    (contact.astype(float) @ c_prev) / np.maximum(contact_counts, 1.0),
                    c_prev,
                )
            lr_term = p.long_range_coupling_strength * (contact_means - c_prev) * dt
            # zero-out residues with no contacts
            lr_term = np.where(contact_counts > 0, lr_term, 0.0)
        else:
            lr_term = np.zeros(n)

        # Noise
        residue_noise = self.rng.normal(0.0, p.residue_noise_std, size=n) * np.sqrt(dt)
        env_term = env_term_scalar * np.sqrt(dt)

        c_next = c_prev + reversion + local_term + lr_term + residue_noise + env_term
        c_next = np.clip(c_next, 0.0, 100.0)

        for i, s in enumerate(self.residues):
            s.confidence = float(c_next[i])

        self.time += dt
        self.step_count += 1
        self._env_history.append(self._env_signal)
        self._confidence_history.append(self._get_confidence_vector())
        self._mean_plddt_history.append(float(np.mean(c_next)))
        # Refresh cheap spatial-corr proxy + k_eff cache (cheap O(k) windowed)
        self._refresh_cached_rho_kEff()
        self._history.append(self._record_state())
        self._check_collapse()

    def _refresh_cached_rho_kEff(self) -> None:
        """Update the cheap windowed-spatial-correlation proxy used at step time.

        The full temporal-trajectory rho (get_rho()) is O(k^2 · t) which is
        prohibitive to call at every step. Instead, at step time we compute
        a windowed correlation across the current per-residue confidence
        vector — the same operationalisation the loader's
        `compute_residue_correlation` uses. This is O(k) per step and
        agrees with the loader's rho.
        """
        # Lazy import to avoid coupling engine→loader (just need the helper)
        from ratchet.data.protein_loader import compute_residue_correlation
        c = self._get_confidence_vector()
        rho = compute_residue_correlation(c)
        self._cached_rho = float(rho)
        k = self.get_k()
        self._cached_k_eff = float(k) if k <= 1 else (k / (1.0 + rho * (k - 1)))

    def run(self, duration: float = 100.0, dt: float = 1.0) -> pd.DataFrame:
        """Run for `duration` steps, returning the per-step history dataframe."""
        n_steps = int(round(duration / dt))
        for _ in range(n_steps):
            self.step(dt)
            if self._collapsed:
                break
        return self.to_dataframe()

    def _record_state(self) -> Dict:
        # NOTE: get_rho() is O(k^2 · t) and dominates the per-step cost when
        # k or step_count grows. Cache it once at engine.run() completion
        # rather than each step. During step recording, use a cheap proxy
        # (last-step rho cache, or 0.0 if not yet computed).
        return {
            "time": self.time,
            "step": self.step_count,
            "k": self.get_k(),
            "k_eff": self._cached_k_eff,
            "rho": self._cached_rho,
            "sigma": self.get_sigma(),
            "f": self.get_f(),
            "mean_plddt": float(np.mean(self._get_confidence_vector())),
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

    def set_n_residues(self, n: int) -> None:
        """Reset engine with a new residue count (calls initialize)."""
        self.params.n_residues = max(10, min(2000, int(n)))
        self.initialize()

    def set_correlation_length(self, L: float) -> None:
        """Set correlation length and rebuild neighbour kernel."""
        self.params.correlation_length = max(1.0, min(100.0, float(L)))
        if self.residues:
            n = len(self.residues)
            W = max(self.params.local_coupling_window, 1)
            idx = np.arange(n)
            dist = np.abs(idx[:, None] - idx[None, :])
            kernel = np.where(dist <= W, np.exp(-dist / self.params.correlation_length), 0.0)
            np.fill_diagonal(kernel, 0.0)
            row_sum = kernel.sum(axis=1, keepdims=True)
            self._neighbour_kernel = np.divide(
                kernel, row_sum, out=np.zeros_like(kernel), where=row_sum > 0,
            )

    def set_mean_plddt_target(self, target: float) -> None:
        """Set the mean pLDDT target across residues."""
        self.params.mean_plddt_target = float(np.clip(target, 0.0, 100.0))

    def apply_shock(self, shock: ProteinShock) -> None:
        if not self.residues:
            raise RuntimeError("Engine not initialized")
        n = len(self.residues)
        if shock.type == ProteinShockType.MUTATION:
            # Drop one residue's target by magnitude · 100
            idx = shock.target_residue if shock.target_residue is not None \
                else int(self.rng.integers(0, n))
            idx = int(np.clip(idx, 0, n - 1))
            self.residues[idx].target_confidence *= max(0.05, 1.0 - shock.magnitude)
            self.residues[idx].confidence *= max(0.05, 1.0 - shock.magnitude)
        elif shock.type == ProteinShockType.DENATURATION:
            # Global confidence collapse
            for s in self.residues:
                s.confidence *= max(0.05, 1.0 - shock.magnitude)
        elif shock.type == ProteinShockType.LIGAND_BINDING:
            # Boost target confidence in a contiguous region
            center = shock.target_residue if shock.target_residue is not None \
                else int(self.rng.integers(0, n))
            radius = max(2, int(0.05 * n))
            for j in range(max(0, center - radius), min(n, center + radius + 1)):
                self.residues[j].target_confidence = float(
                    min(99.0, self.residues[j].target_confidence
                        * (1.0 + shock.magnitude))
                )
        elif shock.type == ProteinShockType.OXIDATIVE_STRESS:
            # Increase residue noise globally
            self.params.residue_noise_std = min(
                30.0, self.params.residue_noise_std + shock.magnitude * 5.0,
            )
        elif shock.type == ProteinShockType.PROTEOLYSIS:
            # Truncate sequence
            keep = max(10, int(n * max(0.05, 1.0 - shock.magnitude)))
            self.residues = self.residues[:keep]
            self.params.n_residues = keep
            # Rebuild kernel + contact for the truncated chain
            self.set_correlation_length(self.params.correlation_length)
            if self._contact_map is not None:
                self._contact_map = self._contact_map[:keep, :keep]

    def apply_intervention(self, intervention: ProteinIntervention) -> None:
        if not self.residues:
            raise RuntimeError("Engine not initialized")
        if intervention.type == ProteinInterventionType.CHAPERONE:
            boost = float(intervention.parameters.get("boost", 0.5))
            self.params.mean_reversion_rate = min(
                1.0, self.params.mean_reversion_rate * (1.0 + boost),
            )
        elif intervention.type == ProteinInterventionType.DISULFIDE:
            i = int(intervention.parameters.get("residue_a", 0))
            j = int(intervention.parameters.get("residue_b", len(self.residues) - 1))
            i = int(np.clip(i, 0, len(self.residues) - 1))
            j = int(np.clip(j, 0, len(self.residues) - 1))
            if i != j and self._contact_map is not None:
                self._contact_map[i, j] = True
                self._contact_map[j, i] = True
                self.residues[i].is_contact = True
                self.residues[j].is_contact = True
        elif intervention.type == ProteinInterventionType.OSMOLYTE:
            self.params.residue_noise_std *= 0.5
            self.params.global_noise_std *= 0.5
        elif intervention.type == ProteinInterventionType.STABILISING_MUTATION:
            target = int(intervention.parameters.get("residue_index", 0))
            target = int(np.clip(target, 0, len(self.residues) - 1))
            boost = float(intervention.parameters.get("boost", 0.10))
            self.residues[target].target_confidence = float(
                min(99.0, self.residues[target].target_confidence * (1.0 + boost))
            )

    # ── measurement (RATCHET-uniform) ────────────────────────────────────

    def get_k(self) -> int:
        return len(self.residues)

    def get_rho(self) -> float:
        """Mean abs pairwise correlation across residue confidence windows.

        Uses the same windowed-spatial operationalisation as the loader's
        `compute_residue_correlation`: slide a window over the current
        per-residue confidence vector, compute pairwise Pearson across
        distinct windows, and return the mean of absolute values.

        This is what AlphaFold pLDDT actually measures — per-residue
        confidence with spatial (sequence-distance) coupling — not a
        temporal trajectory. The loader and engine therefore use the
        same operationalisation so the engine-vs-data RMSE is on the
        same rho convention as the data.
        """
        if not self.residues:
            return 0.0
        # Use cached value if available (refreshed every step in step())
        # to avoid recomputing on every accessor call.
        return float(self._cached_rho)

    def get_rho_temporal(self) -> float:
        """Diagnostic: mean abs pairwise correlation across residue confidence
        trajectories (temporal, not spatial). Provided for comparison only;
        the canonical RATCHET rho is the spatial-window one from get_rho().
        """
        if len(self._confidence_history) < 3 or len(self.residues) < 2:
            return 0.0
        A = np.array(self._confidence_history).T
        n_r, n_t = A.shape
        if n_t < 3:
            return 0.0

        max_pairs = 4000
        if n_r * (n_r - 1) // 2 > max_pairs:
            pair_rng = np.random.default_rng(42)
            i_idx = pair_rng.integers(0, n_r, max_pairs)
            j_idx = pair_rng.integers(0, n_r, max_pairs)
            keep = i_idx != j_idx
            pair_list = list(zip(i_idx[keep], j_idx[keep]))
        else:
            pair_list = [(i, j) for i in range(n_r) for j in range(i + 1, n_r)]

        pairs = []
        for i, j in pair_list:
            if np.std(A[i]) < 1e-10 or np.std(A[j]) < 1e-10:
                continue
            r = np.corrcoef(A[i], A[j])[0, 1]
            if np.isfinite(r):
                pairs.append(abs(float(r)))
        if not pairs:
            return 0.0
        return float(np.mean(pairs))

    def get_sigma(self) -> float:
        """Mean pLDDT / 100, latest step. Bounded to (0, 1]."""
        if not self.residues:
            return 0.0
        return float(np.mean(self._get_confidence_vector()) / 100.0)

    def get_f(self) -> float:
        return float(max(0.0, 1.0 - self.get_sigma()))

    def get_k_eff(self) -> float:
        k = self.get_k()
        rho = self.get_rho()
        if k <= 1:
            return float(k)
        return k / (1.0 + rho * (k - 1))

    def get_plddt_trajectory(self) -> np.ndarray:
        """Per-step mean pLDDT array."""
        return np.array(self._mean_plddt_history, dtype=float)

    def get_confidence_matrix(self) -> np.ndarray:
        """(n_residues, n_steps) per-residue confidence matrix."""
        if not self._confidence_history:
            return np.zeros((self.get_k(), 0))
        return np.array(self._confidence_history).T

    def get_final_plddt_vector(self) -> np.ndarray:
        """The (k,)-shaped final per-residue pLDDT (engine's pLDDT trajectory output).

        This is shape-matched to AlphaFold DB's per-residue pLDDT array
        and is the canonical engine output for engine-vs-data comparison.
        """
        return self._get_confidence_vector()

    def is_collapsed(self) -> bool:
        return self._collapsed

    def get_collapse_time(self) -> Optional[float]:
        return self._collapse_time

    # ── export ───────────────────────────────────────────────────────────

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._history)

    def reset(self) -> None:
        self.residues = []
        self._contact_map = None
        self._neighbour_kernel = None
        self._env_signal = 0.0
        self._env_history = []
        self._confidence_history = []
        self._mean_plddt_history = []
        self._history = []
        self.time = 0.0
        self.step_count = 0
        self._collapsed = False
        self._collapse_time = None


# ─────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────


def create_protein_engine(
    params: Optional[ProteinParams] = None,
    seed: Optional[int] = None,
) -> ProteinFoldingEngine:
    """Factory function to create a ProteinFoldingEngine."""
    return ProteinFoldingEngine(params=params, seed=seed)


__all__ = [
    "ProteinFoldingEngine",
    "ProteinParams",
    "ResidueState",
    "ProteinShock",
    "ProteinIntervention",
    "ProteinShockType",
    "ProteinInterventionType",
    "create_protein_engine",
]
