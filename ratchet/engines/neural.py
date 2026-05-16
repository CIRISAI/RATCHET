"""
RATCHET Neural-Population (Allen Brain Observatory Neuropixels) Substrate Engine

Simulates a population of cortical neurons under a drifting-grating
stimulus block. Exposes the standard RATCHET (k, rho, sigma, k_eff)
interface and produces spike-train matrices matched in shape to Allen
Visual Coding Neuropixels session data.

Domain mapping (per REGIME.md §"A1 — Allen neural firing"):
    k     : Number of simultaneously-recorded neurons per session
    rho   : Mean pairwise spike-train correlation (1-ms bins, abs Pearson)
    sigma : Population-decoding accuracy on drifting gratings
            (cross-validated multiclass linear classifier; 8 orientations)
    f     : 1 − sigma (compromise / decoding-uncertainty fraction)

Dynamics
--------
For each neuron i ∈ [0, k), we use a discrete-time leaky-integrate-and-
fire-like rate model with Poisson emission:

    rate[i, t] = baseline[i]
                 · tuning_modulation(θ_t; pref_i, tw_i)
                 · (1 + g_i · common_latent[t])
    spike[i, t] ~ Poisson(rate[i, t])

where:
  • baseline[i] is per-neuron firing rate (Hz, log-normal across neurons)
  • tuning_modulation = 1 + tune_strength · cos(θ_t − θ_pref_i)
  • common_latent[t] is a shared AR(1) drive that produces ρ structure
  • g_i is per-neuron coupling to the common input
  • θ_t is the stimulus orientation at bin t (one of N_orientations)

Population correlation ρ emerges from two coupled sources:
  • the common-input coupling strength `common_input_coupling` (drives ρ)
  • the AR(1) latent's autocorrelation
The population *decoding accuracy* σ is driven by:
  • `tune_strength` (signal magnitude)
  • the baseline firing rates (signal SNR)
  • the common-input coupling (corrupts decoder, reduces σ if too high)

This mirrors the SyntheticAllenNeuropixelsGenerator in
`ratchet.data.neural_loader`, so a synthetic→engine round-trip is a
*fair* fit test — the engine exercises the same Kish dynamics the
generator uses, just with parameters discovered from data rather than
fixed in advance.

Pairs with: `ratchet/engines/{battery,institutional,microbiome,ecological}.py`.

References
----------
- Siegle, J. H., et al. (2021). Survey of spiking in the mouse visual
  system reveals functional hierarchy. Nature 592, 86-92.
- Averbeck, B. B., Latham, P. E., & Pouget, A. (2006). Neural correlations,
  population coding and computation. Nat. Rev. Neurosci. 7, 358-366.
- Cohen, M. R., & Kohn, A. (2011). Measuring and interpreting neuronal
  correlations. Nat. Neurosci. 14, 811-819.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ratchet.data.neural_loader import (
    N_DRIFTING_ORIENTATIONS,
    compute_pairwise_spike_correlation,
    decode_population_drifting_gratings,
)


# ─────────────────────────────────────────────────────────────────────────
# Shock / intervention enums (parallel EcologicalShockType / etc.)
# ─────────────────────────────────────────────────────────────────────────


class NeuralShockType(Enum):
    """Types of perturbations to a neural population."""
    LESION = "lesion"             # zero out a fraction of neurons (focal damage)
    ANESTHESIA = "anesthesia"     # increase common-input coupling globally
    DEPRIVATION = "deprivation"   # remove stimulus drive (tune_strength → 0)
    SEIZURE = "seizure"           # rate amplification + extreme correlation
    NOISE_INJECTION = "noise"     # raise Poisson rate floor


class NeuralInterventionType(Enum):
    """Types of restorative interventions on the population."""
    STIMULATION = "stimulation"       # boost tune_strength / signal
    NEUROMODULATION = "neuromod"      # decrease common_input_coupling
    NEUROGENESIS = "neurogenesis"     # add new neurons
    PHARMACOLOGY = "pharma"           # global rate scaling


@dataclass
class NeuralShock:
    """External perturbation to the neural population."""
    type: NeuralShockType
    magnitude: float = 0.3
    target_neuron_fraction: float = 1.0  # fraction of neurons affected
    duration: float = 1.0  # in trials


@dataclass
class NeuralIntervention:
    """Restorative intervention on the population."""
    type: NeuralInterventionType
    parameters: Dict = field(default_factory=dict)


@dataclass
class NeuralParams:
    """Configuration for NeuralPopulationEngine.

    Defaults follow the synthetic-Allen-Neuropixels parameterisation in
    `ratchet.data.neural_loader.SyntheticAllenNeuropixelsGenerator`. Pass
    `seed` for reproducibility; pass `n_neurons` / `common_input_coupling` /
    `tune_strength` for forced initialisation (e.g. matching a real
    session's characteristics).
    """
    engine: str = "neural"
    n_neurons: int = 60
    n_orientations: int = N_DRIFTING_ORIENTATIONS  # 8 drifting gratings
    n_reps_per_orientation: int = 10               # → 80 trials default
    # 500 ms / trial keeps the synthetic→engine harness tractable; real
    # Allen sessions use 2000 ms drifting-grating windows. The fit
    # harness reads bins_per_trial from the observed session, so 2000
    # ms slots in automatically when real data is vendored.
    trial_duration_ms: float = 500.0
    bin_ms: float = 1.0                            # 1 ms bins

    # Per-neuron dynamics
    baseline_rate_mean_hz: float = 5.0
    baseline_rate_sigma_log: float = 0.5
    baseline_rate_min_hz: float = 0.5
    baseline_rate_max_hz: float = 30.0

    # Common-input modulation (drives ρ)
    common_input_coupling: float = 0.20   # ∈ [0, 0.8]; higher → higher ρ
    common_input_ar_phi: float = 0.95     # AR(1) phi for latent

    # Tuning (drives σ)
    tune_strength: float = 0.6            # ∈ [0, 2]; higher → higher σ
    tuning_width_lo: float = 0.5
    tuning_width_hi: float = 3.0

    # Collapse criterion: ρ above threshold → population coding collapses
    rho_collapse_threshold: float = 0.60

    seed: Optional[int] = None


# ─────────────────────────────────────────────────────────────────────────
# Per-neuron state container
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class NeuronState:
    """Per-neuron parameters for the leaky-rate model."""
    neuron_id: str
    baseline_rate_hz: float
    preferred_orientation: float  # radians
    tuning_width: float           # von-Mises-like multiplicative width
    common_gain: float            # per-neuron sensitivity to common input
    active: bool = True           # set False on lesion


# ─────────────────────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────────────────────


class NeuralPopulationEngine:
    """Discrete-time Poisson neural population engine for Allen Neuropixels.

    Example
    -------
    >>> engine = NeuralPopulationEngine(seed=42)
    >>> engine.initialize()
    >>> df = engine.run()
    >>> print(engine.get_k(), engine.get_rho(), engine.get_sigma())
    """

    def __init__(
        self,
        params: Optional[NeuralParams] = None,
        seed: Optional[int] = None,
    ):
        self.params = params or NeuralParams()
        if seed is not None:
            self.params.seed = seed

        self.rng = np.random.default_rng(self.params.seed)
        self.time_ms = 0.0
        self.step_count = 0  # trial-count, not bin-count

        self.neurons: List[NeuronState] = []
        self._spike_train_matrix: Optional[np.ndarray] = None  # (k, n_time_bins)
        self._stimulus_labels: Optional[np.ndarray] = None     # (n_trials,)
        self._trial_bin_edges: Optional[np.ndarray] = None     # (n_trials+1,)
        self._common_latent: Optional[np.ndarray] = None       # (n_time_bins,)
        self._history: List[Dict] = []

        self._collapsed = False
        self._collapse_step: Optional[int] = None

    # ── initialisation ───────────────────────────────────────────────────

    def initialize(self) -> None:
        """Initialise neuron parameters + stimulus block + (empty) spike trains."""
        p = self.params
        n = p.n_neurons

        # Per-neuron params
        rates = self.rng.lognormal(
            mean=np.log(max(p.baseline_rate_mean_hz, 0.1)),
            sigma=p.baseline_rate_sigma_log, size=n,
        )
        rates = np.clip(rates, p.baseline_rate_min_hz, p.baseline_rate_max_hz)

        pref_orientations = self.rng.uniform(0, 2 * np.pi, size=n)
        tuning_widths = self.rng.uniform(p.tuning_width_lo, p.tuning_width_hi, size=n)
        common_gains = self.rng.uniform(0.5, 1.5, size=n) * p.common_input_coupling

        self.neurons = [
            NeuronState(
                neuron_id=f"n_{i:04d}",
                baseline_rate_hz=float(rates[i]),
                preferred_orientation=float(pref_orientations[i]),
                tuning_width=float(tuning_widths[i]),
                common_gain=float(common_gains[i]),
                active=True,
            )
            for i in range(n)
        ]

        # Build stimulus block: balanced reps × n_orientations, shuffled
        n_trials = p.n_reps_per_orientation * p.n_orientations
        labels = np.tile(np.arange(p.n_orientations), p.n_reps_per_orientation)
        self.rng.shuffle(labels)
        self._stimulus_labels = labels.astype(np.int16)

        bins_per_trial = max(1, int(round(p.trial_duration_ms / p.bin_ms)))
        n_time_bins = n_trials * bins_per_trial
        self._trial_bin_edges = np.arange(0, n_time_bins + 1, bins_per_trial, dtype=int)

        # Empty spike matrix - filled by run()
        self._spike_train_matrix = np.zeros((n, n_time_bins), dtype=np.int16)

        # Reset history
        self.time_ms = 0.0
        self.step_count = 0
        self._collapsed = False
        self._collapse_step = None
        self._common_latent = None
        self._history = [self._record_state()]

    # ── helpers ──────────────────────────────────────────────────────────

    def _active_indices(self) -> np.ndarray:
        return np.array(
            [i for i, n in enumerate(self.neurons) if n.active], dtype=int
        )

    def _per_neuron_arrays(self):
        """Return (rates_hz, pref_orientations, tuning_widths, common_gains)."""
        rates = np.array([n.baseline_rate_hz for n in self.neurons], dtype=float)
        prefs = np.array([n.preferred_orientation for n in self.neurons], dtype=float)
        tws = np.array([n.tuning_width for n in self.neurons], dtype=float)
        gains = np.array([n.common_gain for n in self.neurons], dtype=float)
        active_mask = np.array([n.active for n in self.neurons], dtype=bool)
        return rates, prefs, tws, gains, active_mask

    # ── core simulation ──────────────────────────────────────────────────

    def step(self) -> None:
        """Advance simulation by one trial.

        Generates spikes for the next trial-window in
        `_spike_train_matrix` using the current latent + tuning + Poisson
        model. After each trial we re-check the collapse criterion.
        """
        if not self.neurons:
            raise RuntimeError("Engine not initialized. Call initialize() first.")
        if self._collapsed:
            return
        if self._stimulus_labels is None or self._trial_bin_edges is None:
            return

        p = self.params
        trial = self.step_count
        if trial >= len(self._stimulus_labels):
            return

        edges = self._trial_bin_edges
        lo, hi = int(edges[trial]), int(edges[trial + 1])
        n_t = hi - lo

        rates_hz, prefs, tws, gains, active_mask = self._per_neuron_arrays()
        rates_per_bin = rates_hz * (p.bin_ms / 1000.0)

        # Trial orientation in radians
        theta = (int(self._stimulus_labels[trial]) / p.n_orientations) * 2 * np.pi

        # Common latent (AR(1)) for this trial — keep continuity across trials
        if self._common_latent is None or len(self._common_latent) < hi:
            # First-time fill of the latent: lazily extend up to hi bins
            if self._common_latent is None:
                latent = np.zeros(hi, dtype=float)
                latent[0] = self.rng.normal(0, 1)
                start = 1
            else:
                old = self._common_latent
                latent = np.zeros(hi, dtype=float)
                latent[: len(old)] = old
                start = len(old)
            innov_std = np.sqrt(1.0 - p.common_input_ar_phi ** 2)
            for t in range(start, hi):
                latent[t] = (p.common_input_ar_phi * latent[t - 1]
                             + self.rng.normal(0, innov_std))
            # Standardise running window so variance stays unit-ish
            window_std = latent[:hi].std()
            if window_std > 1e-9:
                latent[:hi] = (latent[:hi] - latent[:hi].mean()) / window_std
            self._common_latent = latent

        latent_slice = self._common_latent[lo:hi]  # (n_t,)

        # Tuning modulation per neuron (broadcast to bins)
        ang_diff = theta - prefs
        tuning_mod = 1.0 + p.tune_strength * np.cos(ang_diff) * (tws / 1.5)  # (k,)

        # Per-bin common modulation (broadcast across neurons)
        # rate[i, t] = rates_per_bin[i] · tuning_mod[i] · (1 + gains[i] · latent[t])
        rate_mat = (
            rates_per_bin[:, None]
            * tuning_mod[:, None]
            * (1.0 + gains[:, None] * latent_slice[None, :])
        )
        rate_mat = np.clip(rate_mat, 1e-6, 1.0)

        # Zero out inactive (lesioned) neurons
        rate_mat[~active_mask] = 0.0

        spikes = self.rng.poisson(rate_mat).astype(np.int16)
        self._spike_train_matrix[:, lo:hi] = spikes

        self.step_count += 1
        self.time_ms += (hi - lo) * p.bin_ms
        self._history.append(self._record_state())
        self._check_collapse()

    def run(self) -> pd.DataFrame:
        """Run the full stimulus block (all trials), returning per-trial history.

        If a shock has been applied that flips `_collapsed`, the run halts.
        """
        if not self.neurons:
            raise RuntimeError("Engine not initialized. Call initialize() first.")
        if self._stimulus_labels is None:
            self.initialize()
        n_trials = len(self._stimulus_labels) if self._stimulus_labels is not None else 0
        for _ in range(n_trials - self.step_count):
            self.step()
            if self._collapsed:
                break
        # Final state snapshot with full Kish metrics (recomputed once)
        self._history.append(self._record_state(compute_kish=True))
        return self.to_dataframe()

    def _record_state(self, compute_kish: bool = False) -> Dict:
        """Per-trial history record.

        Heavy metrics (rho, sigma, k_eff) are computed lazily by default
        because they scan the full spike matrix and decode every trial —
        cost is O(k² · t) for rho and O(k · t) for sigma. Pass
        `compute_kish=True` for end-of-run snapshots; otherwise rho/sigma
        appear as NaN in the history and are recomputed on demand via
        get_rho()/get_sigma().
        """
        if compute_kish:
            rho = self.get_rho()
            sigma = self.get_sigma()
            k_eff = self.get_k_eff()
            f = float(max(0.0, 1.0 - sigma))
        else:
            rho = float("nan")
            sigma = float("nan")
            k_eff = float("nan")
            f = float("nan")
        return {
            "trial": self.step_count,
            "time_ms": self.time_ms,
            "k": self.get_k(),
            "k_active": int(np.sum([n.active for n in self.neurons])),
            "k_eff": k_eff,
            "rho": rho,
            "sigma": sigma,
            "f": f,
            "collapsed": self._collapsed,
        }

    def _check_collapse(self) -> None:
        if self._collapsed:
            return
        # Population coding collapses when ρ is too high (lost diversity).
        # Computing ρ scans the full spike matrix (O(k² · t)), so only
        # check every 10 trials past the 5-trial warmup.
        if self.step_count < 5 or (self.step_count % 10) != 0:
            return
        rho = self.get_rho()
        if rho >= self.params.rho_collapse_threshold:
            self._collapsed = True
            self._collapse_step = self.step_count

    # ── manipulation ─────────────────────────────────────────────────────

    def set_n_neurons(self, n: int) -> None:
        """Reset engine with a new neuron count (calls initialize)."""
        self.params.n_neurons = max(2, min(1000, int(n)))
        self.initialize()

    def set_common_input_coupling(self, c: float) -> None:
        """Set common-input coupling. Requires re-initialise to take effect."""
        self.params.common_input_coupling = float(np.clip(c, 0.0, 1.0))
        # Re-roll per-neuron gains if neurons already exist.
        if self.neurons:
            gains = self.rng.uniform(0.5, 1.5, size=len(self.neurons)) * self.params.common_input_coupling
            for i, neuron in enumerate(self.neurons):
                neuron.common_gain = float(gains[i])

    def set_tune_strength(self, t: float) -> None:
        """Set per-neuron tuning strength (re-rolling not required)."""
        self.params.tune_strength = float(np.clip(t, 0.0, 2.0))

    def apply_shock(self, shock: NeuralShock) -> None:
        if not self.neurons:
            raise RuntimeError("Engine not initialized")
        n = len(self.neurons)
        if shock.type == NeuralShockType.LESION:
            # Knock out a fraction of neurons (set inactive)
            n_kill = max(1, int(round(shock.target_neuron_fraction * shock.magnitude * n)))
            idx = self.rng.choice(n, size=min(n_kill, n), replace=False)
            for i in idx:
                self.neurons[int(i)].active = False
        elif shock.type == NeuralShockType.ANESTHESIA:
            # Boost coupling globally → more correlation
            new_c = float(np.clip(self.params.common_input_coupling + shock.magnitude, 0.0, 1.0))
            self.set_common_input_coupling(new_c)
        elif shock.type == NeuralShockType.DEPRIVATION:
            # Cut stimulus drive
            self.params.tune_strength = float(max(0.0, self.params.tune_strength - shock.magnitude))
        elif shock.type == NeuralShockType.SEIZURE:
            # Rate amplification + lock-step coupling
            for neuron in self.neurons:
                neuron.baseline_rate_hz = float(min(50.0, neuron.baseline_rate_hz * (1.0 + shock.magnitude)))
            self.set_common_input_coupling(min(1.0, self.params.common_input_coupling + shock.magnitude))
        elif shock.type == NeuralShockType.NOISE_INJECTION:
            # Add a baseline rate floor (raises Poisson rate, lowers SNR → lowers σ)
            for neuron in self.neurons:
                neuron.baseline_rate_hz = float(neuron.baseline_rate_hz + 2.0 * shock.magnitude)

    def apply_intervention(self, intervention: NeuralIntervention) -> None:
        if not self.neurons:
            raise RuntimeError("Engine not initialized")
        if intervention.type == NeuralInterventionType.STIMULATION:
            boost = float(intervention.parameters.get("tune_boost", 0.2))
            self.params.tune_strength = float(min(2.0, self.params.tune_strength + boost))
        elif intervention.type == NeuralInterventionType.NEUROMODULATION:
            reduction = float(intervention.parameters.get("coupling_reduction", 0.2))
            new_c = float(max(0.0, self.params.common_input_coupling - reduction))
            self.set_common_input_coupling(new_c)
        elif intervention.type == NeuralInterventionType.NEUROGENESIS:
            n_add = int(intervention.parameters.get("n_neurons", 5))
            for j in range(n_add):
                self.neurons.append(NeuronState(
                    neuron_id=f"n_new_{len(self.neurons):04d}",
                    baseline_rate_hz=float(intervention.parameters.get("rate_hz", 5.0)),
                    preferred_orientation=float(self.rng.uniform(0, 2 * np.pi)),
                    tuning_width=float(self.rng.uniform(0.5, 3.0)),
                    common_gain=float(self.rng.uniform(0.5, 1.5) * self.params.common_input_coupling),
                    active=True,
                ))
            self.params.n_neurons = len(self.neurons)
            # Grow spike matrix
            if self._spike_train_matrix is not None:
                old = self._spike_train_matrix
                grown = np.zeros((self.params.n_neurons, old.shape[1]), dtype=np.int16)
                grown[: old.shape[0]] = old
                self._spike_train_matrix = grown
        elif intervention.type == NeuralInterventionType.PHARMACOLOGY:
            scale = float(intervention.parameters.get("rate_scale", 1.1))
            for neuron in self.neurons:
                neuron.baseline_rate_hz = float(np.clip(
                    neuron.baseline_rate_hz * scale,
                    self.params.baseline_rate_min_hz,
                    self.params.baseline_rate_max_hz,
                ))

    # ── measurement (RATCHET-uniform) ────────────────────────────────────

    def get_k(self) -> int:
        """Total recorded neurons (active + inactive — k is recording capacity)."""
        return len(self.neurons)

    def get_k_active(self) -> int:
        """Active neurons only (post-lesion)."""
        return int(np.sum([n.active for n in self.neurons]))

    def get_rho(self) -> float:
        """Mean abs pairwise correlation across spike trains observed so far."""
        if self._spike_train_matrix is None or self.step_count < 1:
            return 0.0
        # Use only the bins observed so far
        if self._trial_bin_edges is None:
            return 0.0
        end_bin = int(self._trial_bin_edges[self.step_count])
        if end_bin < 2:
            return 0.0
        observed = self._spike_train_matrix[:, :end_bin]
        # Restrict to active neurons (lesioned rows are all-zero anyway)
        active_idx = self._active_indices()
        if len(active_idx) < 2:
            return 0.0
        return compute_pairwise_spike_correlation(observed[active_idx], max_pairs=1500)

    def get_sigma(self) -> float:
        """Cross-validated population decoding accuracy on observed trials."""
        if (self._spike_train_matrix is None
                or self._stimulus_labels is None
                or self._trial_bin_edges is None
                or self.step_count < self.params.n_orientations * 2):
            # Too few trials for meaningful decoding — return chance level
            return 1.0 / max(1, self.params.n_orientations)

        end_bin = int(self._trial_bin_edges[self.step_count])
        active_idx = self._active_indices()
        if len(active_idx) < 2 or end_bin < 2:
            return 1.0 / max(1, self.params.n_orientations)
        observed_spikes = self._spike_train_matrix[active_idx, :end_bin]
        observed_labels = self._stimulus_labels[: self.step_count]
        observed_edges = self._trial_bin_edges[: self.step_count + 1]

        return decode_population_drifting_gratings(
            observed_spikes,
            observed_labels,
            observed_edges,
            n_folds=min(5, max(2, self.step_count // 4)),
        )

    def get_f(self) -> float:
        return float(max(0.0, 1.0 - self.get_sigma()))

    def get_k_eff(self) -> float:
        k = self.get_k_active()
        rho = self.get_rho()
        if k <= 1:
            return float(k)
        return k / (1.0 + rho * (k - 1))

    def get_spike_train_matrix(self) -> np.ndarray:
        """(k, n_time_bins) integer spike counts (zeros past current trial)."""
        if self._spike_train_matrix is None:
            return np.zeros((self.get_k(), 0), dtype=np.int16)
        return self._spike_train_matrix.copy()

    def get_stimulus_labels(self) -> np.ndarray:
        if self._stimulus_labels is None:
            return np.zeros(0, dtype=np.int16)
        return self._stimulus_labels.copy()

    def get_trial_bin_edges(self) -> np.ndarray:
        if self._trial_bin_edges is None:
            return np.zeros(0, dtype=int)
        return self._trial_bin_edges.copy()

    def is_collapsed(self) -> bool:
        return self._collapsed

    def get_collapse_step(self) -> Optional[int]:
        return self._collapse_step

    # ── export ───────────────────────────────────────────────────────────

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._history)

    def reset(self) -> None:
        self.neurons = []
        self._spike_train_matrix = None
        self._stimulus_labels = None
        self._trial_bin_edges = None
        self._common_latent = None
        self._history = []
        self.time_ms = 0.0
        self.step_count = 0
        self._collapsed = False
        self._collapse_step = None


# ─────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────


def create_neural_engine(
    params: Optional[NeuralParams] = None,
    seed: Optional[int] = None,
) -> NeuralPopulationEngine:
    """Factory function to create a NeuralPopulationEngine."""
    return NeuralPopulationEngine(params=params, seed=seed)


__all__ = [
    "NeuralPopulationEngine",
    "NeuralParams",
    "NeuronState",
    "NeuralShock",
    "NeuralIntervention",
    "NeuralShockType",
    "NeuralInterventionType",
    "create_neural_engine",
]
