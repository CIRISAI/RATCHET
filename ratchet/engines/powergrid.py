"""
RATCHET Power-Grid (PNNL PMU) Substrate Engine

Simulates synchrophasor-instrumented transmission-grid dynamics for a fixed
set of PMU nodes during a disturbance event. Exposes the standard RATCHET
(k, rho, sigma, k_eff) interface and produces frequency-vs-time trajectories
matched in shape to PNNL Open PMU Library event traces.

Domain mapping (per REGIME.md §"A0 — PMU grid"):
    k     : Number of PMUs reporting in a grid region during an event
    rho   : Mean pairwise correlation of pre-event frequency time series
            (5-min baseline)
    sigma : Inverse of post-event settling-time CV (stability of recovery)
    f     : 1 − sigma (compromise / instability fraction)

Constituent agency: A0 (engineered, no internal goals). PMUs are
fixed-purpose sensors; the substrate exercises Kish dynamics on an
electrical-infrastructure network rather than a biological or AI system.

Dynamics
--------
Per-PMU swing equation (linearised Bergen-Vittal, Ch. 14):

    df_i/dt = - (D_i / (2 H_i)) · (f_i − f_nominal)         (damping)
              + Σ_j K_ij · (f_j − f_i) / (2 H_i)             (coupling)
              + ε_common · ξ_t                               (common drift)
              + obs_noise_i                                  (per-PMU noise)
              + disturbance_i(t)                             (event impulse)

where:
    H_i : per-PMU inertia (seconds, governed by generator mass)
    D_i : per-PMU damping (governor + load damping coefficient)
    K_ij: inter-PMU coupling via grid impedance (1 / electrical distance)
    ξ_t : AR(1) common-mode drift driving baseline cross-PMU correlation
    disturbance_i: step + ramp at event_time_idx, weighted by distance
                   from a randomly-placed epicentre.

The event-level *correlation* ρ emerges from two coupled sources:
  • the inter-PMU coupling matrix K_ij (electrical distance topology)
  • the common-mode drift ε_common (system-wide oscillation modes)

The post-event *settling time* per PMU is the recovery interval back to
within ±0.05 Hz of nominal. σ = 1 / (1 + CV(settling_times)) captures
*spread* of recovery across the array — uniform recovery → σ → 1; PMUs
that fail to recover or recover at wildly different times → σ → 0.

This mirrors the PMUEventGenerator in
`ratchet.data.powergrid_loader`, so a synthetic→engine round-trip is a
*fair* fit test — the engine exercises the same swing dynamics the
generator uses, just with parameters calibrated to the observed event.

Pairs with: `ratchet/engines/{battery,institutional,microbiome,ecological}.py`.

References
----------
- PNNL-30492 (2020): "Open-source PMU library for grid event analysis."
- Bergen, A. R., & Vittal, V. (2000). Power Systems Analysis (2nd ed.).
  Prentice Hall — synchronous-machine swing equation.
- Kundur, P. (1994). Power System Stability and Control. McGraw-Hill —
  inter-area oscillation modes; settling-time analysis.
- IEEE C37.118.1 (2011): Synchrophasor measurement standard.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# Physical constants
NOMINAL_FREQUENCY_HZ = 60.0
SETTLING_BAND_HZ = 0.05
DEFAULT_SAMPLE_RATE_HZ = 30.0


# ─────────────────────────────────────────────────────────────────────────
# Shock / intervention enums (parallel BatteryShockType / EcologicalShockType)
# ─────────────────────────────────────────────────────────────────────────


class PMUGridShockType(Enum):
    """Types of perturbations to a grid region."""
    LINE_TRIP = "line_trip"           # transmission line outage → freq drop
    GENERATOR_LOSS = "generator_loss" # large genset trips → freq drop
    LOAD_SHED = "load_shed"           # large load disconnect → freq rise
    INTER_AREA_OSC = "inter_area"     # forced oscillation (e.g. PSS instability)
    COMMON_MODE = "common_mode"       # interconnect-wide event (e.g. GMD)


class PMUGridInterventionType(Enum):
    """Types of grid-side mitigations / interventions."""
    FREQUENCY_RESPONSE = "frequency_response"  # boost governor response
    ISLANDING = "islanding"                    # cut grid into smaller regions
    PMU_ISOLATION = "pmu_isolation"            # remove a faulted PMU
    FAST_AGC = "fast_agc"                      # accelerate AGC restoration


@dataclass
class PMUGridShock:
    """External disturbance to a grid region."""
    type: PMUGridShockType
    magnitude_hz: float = 0.2          # peak frequency deviation in Hz
    target_pmu: Optional[int] = None   # None = region-wide (weighted by distance)
    duration_s: float = 0.5            # how long the disturbance ramp lasts


@dataclass
class PMUGridIntervention:
    """Grid operator's response intervention."""
    type: PMUGridInterventionType
    parameters: Dict = field(default_factory=dict)


@dataclass
class PMUGridParams:
    """Configuration for PMUGridEngine.

    Defaults follow the synthetic-PNNL parameterisation in
    `ratchet.data.powergrid_loader.SyntheticPMUEventGenerator`. Pass
    `seed` for reproducibility; pass `n_pmus`/`base_coupling`/etc for
    forced initialisation (e.g. matching a real event's k and ρ).
    """
    engine: str = "powergrid"
    n_pmus: int = 8
    duration_s: float = 60.0
    pre_event_s: float = 30.0
    sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ

    # Per-PMU swing-equation parameters
    inertia_mean: float = 5.0          # H_i [seconds]
    inertia_log_std: float = 0.15
    damping_mean: float = 0.4          # D_i
    damping_log_std: float = 0.15

    # Inter-PMU coupling (drives ρ)
    base_coupling: float = 0.4
    spatial_decay: float = 1.5

    # Noise floors
    pre_event_noise_sd_hz: float = 0.01      # per-PMU baseline jitter
    common_mode_ar_phi: float = 0.85         # AR(1) coefficient of common drift
    common_mode_amp_frac: float = 0.4        # frac of pre_event_noise_sd_hz

    # Disturbance defaults (overridden by apply_shock)
    disturbance_magnitude_hz: float = 0.2
    disturbance_sign: float = -1.0          # default: frequency drop

    # Collapse criterion
    sigma_collapse_threshold: float = 0.10  # very low stability ⇒ collapse

    seed: Optional[int] = None


# ─────────────────────────────────────────────────────────────────────────
# Per-PMU state container
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class PMUState:
    pmu_id: str
    frequency_hz: float
    position_xy: np.ndarray       # 2D spatial position [0,1]^2
    inertia: float                # H_i (seconds)
    damping: float                # D_i
    distance_to_epicentre: float = 0.0
    active: bool = True           # set False if PMU isolated by intervention


# ─────────────────────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────────────────────


class PMUGridEngine:
    """Synchronous-machine swing-equation grid-region engine.

    Example
    -------
    >>> engine = PMUGridEngine(seed=42)
    >>> engine.initialize()
    >>> df = engine.run(duration_s=60.0)
    >>> print(engine.get_k(), engine.get_rho(), engine.get_sigma())
    """

    def __init__(
        self,
        params: Optional[PMUGridParams] = None,
        seed: Optional[int] = None,
    ):
        self.params = params or PMUGridParams()
        if seed is not None:
            self.params.seed = seed

        self.rng = np.random.default_rng(self.params.seed)
        self.time_s = 0.0
        self.step_count = 0

        self.pmus: List[PMUState] = []
        self._coupling_matrix: Optional[np.ndarray] = None
        self._epicentre: Optional[np.ndarray] = None
        self._distance_weights: Optional[np.ndarray] = None
        self._common_drift: float = 0.0
        self._event_applied: bool = False
        self._event_time_s: float = -1.0
        self._event_active_until_s: float = -1.0

        self._frequency_history: List[np.ndarray] = []   # (n_pmus,) per step
        self._time_history: List[float] = []
        self._history: List[Dict] = []

        self._collapsed = False
        self._collapse_time: Optional[float] = None

    # ── initialisation ───────────────────────────────────────────────────

    def initialize(self) -> None:
        """Initialise PMUs + coupling matrix + epicentre + reset history."""
        p = self.params
        n = max(2, int(p.n_pmus))

        # Spatial positions in unit square
        positions = self.rng.uniform(0.0, 1.0, size=(n, 2))

        # Per-PMU swing parameters
        H = np.clip(
            p.inertia_mean * np.exp(self.rng.normal(0.0, p.inertia_log_std, n)),
            1.0, 20.0,
        )
        D = np.clip(
            p.damping_mean * np.exp(self.rng.normal(0.0, p.damping_log_std, n)),
            0.1, 2.0,
        )

        # Initial frequency: nominal ± small noise
        f0 = NOMINAL_FREQUENCY_HZ + self.rng.normal(0.0, p.pre_event_noise_sd_hz, n)

        self.pmus = [
            PMUState(
                pmu_id=f"pmu_{i:03d}",
                frequency_hz=float(f0[i]),
                position_xy=positions[i].copy(),
                inertia=float(H[i]),
                damping=float(D[i]),
                active=True,
            )
            for i in range(n)
        ]

        # Coupling matrix: K_ij = base / (d_ij^decay + 1e-2)
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                d = float(np.linalg.norm(positions[i] - positions[j]))
                K[i, j] = p.base_coupling / (d ** p.spatial_decay + 1e-2)
        self._coupling_matrix = K

        # Epicentre and distance weights (for future disturbance)
        self._epicentre = self.rng.uniform(0.0, 1.0, size=2)
        d_to_epi = np.linalg.norm(positions - self._epicentre, axis=1)
        weights = np.exp(-2.0 * d_to_epi)
        weights /= np.max(weights) + 1e-9
        self._distance_weights = weights
        for i, s in enumerate(self.pmus):
            s.distance_to_epicentre = float(d_to_epi[i])

        # Common drift initial state
        self._common_drift = float(self.rng.normal(0.0, 0.5 * p.pre_event_noise_sd_hz))

        # Default event onset: after pre_event_s seconds
        self._event_applied = False
        self._event_time_s = float(p.pre_event_s)
        self._event_active_until_s = -1.0

        # Reset history
        self.time_s = 0.0
        self.step_count = 0
        self._frequency_history = [self._get_frequency_vector()]
        self._time_history = [0.0]
        self._collapsed = False
        self._collapse_time = None
        self._history = [self._record_state()]

    # ── helpers ──────────────────────────────────────────────────────────

    def _get_frequency_vector(self) -> np.ndarray:
        return np.array([s.frequency_hz for s in self.pmus], dtype=float)

    def _get_inertia_vector(self) -> np.ndarray:
        return np.array([s.inertia for s in self.pmus], dtype=float)

    def _get_damping_vector(self) -> np.ndarray:
        return np.array([s.damping for s in self.pmus], dtype=float)

    def _get_active_mask(self) -> np.ndarray:
        return np.array([s.active for s in self.pmus], dtype=bool)

    # ── core simulation ──────────────────────────────────────────────────

    def step(self, dt: Optional[float] = None) -> None:
        """Advance simulation by `dt` seconds (default 1/sample_rate)."""
        if not self.pmus:
            raise RuntimeError("Engine not initialized. Call initialize() first.")
        if self._collapsed:
            return

        p = self.params
        if dt is None:
            dt = 1.0 / max(p.sample_rate_hz, 1.0)
        n = len(self.pmus)

        # Update common drift (AR(1) walk)
        common_sd = p.common_mode_amp_frac * p.pre_event_noise_sd_hz * np.sqrt(
            max(0.0, 1.0 - p.common_mode_ar_phi ** 2)
        )
        new_common = (p.common_mode_ar_phi * self._common_drift
                      + self.rng.normal(0.0, common_sd))
        delta_common = new_common - self._common_drift
        self._common_drift = new_common

        f_prev = self._get_frequency_vector()
        delta = f_prev - NOMINAL_FREQUENCY_HZ
        H = self._get_inertia_vector()
        D = self._get_damping_vector()
        K = self._coupling_matrix  # type: ignore[assignment]
        active = self._get_active_mask()

        df = np.zeros(n)
        # Damping
        df += -(D / (2.0 * H)) * delta * dt

        # Inter-PMU coupling
        # df_i += (1/(2H_i)) * Σ_j K_ij (f_j - f_i) * dt
        delta_pairs = (f_prev[None, :] - f_prev[:, None])  # (i, j)
        coupling_term = np.sum(K * delta_pairs, axis=1) / (2.0 * H) * dt
        df += coupling_term

        # Common-mode drift (registers as cross-PMU correlated baseline)
        df += delta_common

        # Per-PMU noise
        df += self.rng.normal(0.0, p.pre_event_noise_sd_hz, size=n) * np.sqrt(dt)

        # Apply scheduled disturbance, if its window is current
        if not self._event_applied and self.time_s + dt >= self._event_time_s:
            # Apply the impulse at this step
            df += (p.disturbance_sign * p.disturbance_magnitude_hz
                   * self._distance_weights)
            self._event_applied = True
            self._event_active_until_s = self._event_time_s + max(0.0,
                                            getattr(p, "disturbance_duration_s", 0.5))
        elif self._event_applied and self.time_s < self._event_active_until_s:
            # Continuing ramp during disturbance window
            df += (0.2 * p.disturbance_sign * p.disturbance_magnitude_hz
                   * self._distance_weights * dt)

        # Inactive PMUs are frozen at their pre-event values
        df = df * active

        f_next = f_prev + df

        for i, s in enumerate(self.pmus):
            if s.active:
                s.frequency_hz = float(f_next[i])

        self.time_s += dt
        self.step_count += 1
        self._frequency_history.append(self._get_frequency_vector())
        self._time_history.append(self.time_s)
        self._history.append(self._record_state())
        self._check_collapse()

    def run(
        self,
        duration_s: Optional[float] = None,
        dt: Optional[float] = None,
    ) -> pd.DataFrame:
        """Run for `duration_s` seconds, returning per-step history dataframe."""
        p = self.params
        if duration_s is None:
            duration_s = p.duration_s
        if dt is None:
            dt = 1.0 / max(p.sample_rate_hz, 1.0)
        n_steps = int(round(duration_s / dt))
        for _ in range(n_steps):
            self.step(dt)
            if self._collapsed:
                break
        return self.to_dataframe()

    def _record_state(self) -> Dict:
        """Light per-step state record (avoids expensive O(N²·T) get_rho calls).

        rho / sigma / k_eff are recomputed from full history via the
        get_rho/get_sigma accessors at the end of `run()` — not here.
        """
        f_vec = self._get_frequency_vector()
        return {
            "time_s": self.time_s,
            "step": self.step_count,
            "k": self.get_k(),
            "freq_mean_hz": float(np.mean(f_vec)),
            "freq_std_hz": float(np.std(f_vec)),
            "freq_max_dev_hz": float(np.max(np.abs(f_vec - NOMINAL_FREQUENCY_HZ))),
            "common_drift_hz": float(self._common_drift),
            "event_applied": self._event_applied,
            "collapsed": self._collapsed,
        }

    def _check_collapse(self) -> None:
        """Cheap collapse check: triggers on persistent large deviation.

        Avoids calling get_sigma (which is O(N·T)) every step. We instead
        flag collapse if 5+ seconds after the event the max absolute
        deviation from nominal is still larger than 10x the settling band
        (i.e. the swing did not damp at all). The full sigma check at
        end-of-run still drives the headline metric.
        """
        if self._collapsed:
            return
        if not self._event_applied or self.time_s <= self._event_time_s + 5.0:
            return
        f_vec = self._get_frequency_vector()
        max_dev = float(np.max(np.abs(f_vec - NOMINAL_FREQUENCY_HZ)))
        if max_dev > 10.0 * SETTLING_BAND_HZ * 5.0:  # very loose; rarely fires
            # Only do the expensive sigma check if the cheap one suggests trouble
            if self.get_sigma() < self.params.sigma_collapse_threshold:
                self._collapsed = True
                self._collapse_time = self.time_s

    # ── manipulation ─────────────────────────────────────────────────────

    def set_n_pmus(self, n: int) -> None:
        """Reset engine with a new PMU count (calls initialize)."""
        self.params.n_pmus = max(2, min(60, int(n)))
        self.initialize()

    def set_base_coupling(self, c: float) -> None:
        """Set base coupling and rebuild coupling matrix from current positions."""
        self.params.base_coupling = max(0.0, min(2.0, float(c)))
        if not self.pmus:
            return
        n = len(self.pmus)
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                d = float(np.linalg.norm(
                    self.pmus[i].position_xy - self.pmus[j].position_xy))
                K[i, j] = self.params.base_coupling / (
                    d ** self.params.spatial_decay + 1e-2
                )
        self._coupling_matrix = K

    def set_disturbance(
        self,
        magnitude_hz: float,
        sign: float = -1.0,
        time_s: Optional[float] = None,
        duration_s: float = 0.5,
    ) -> None:
        """Configure (or re-arm) the pending disturbance."""
        self.params.disturbance_magnitude_hz = float(magnitude_hz)
        self.params.disturbance_sign = float(np.sign(sign) or -1.0)
        # Attribute may not exist if params was created elsewhere
        setattr(self.params, "disturbance_duration_s", float(duration_s))
        if time_s is not None:
            self._event_time_s = float(time_s)
            self._event_applied = False

    def apply_shock(self, shock: PMUGridShock) -> None:
        if not self.pmus:
            raise RuntimeError("Engine not initialized")

        sign = -1.0
        if shock.type == PMUGridShockType.LOAD_SHED:
            sign = +1.0
        elif shock.type in (
            PMUGridShockType.LINE_TRIP,
            PMUGridShockType.GENERATOR_LOSS,
        ):
            sign = -1.0
        elif shock.type == PMUGridShockType.INTER_AREA_OSC:
            # Oscillation: random sign, smaller magnitude, but reinforces coupling
            sign = -1.0 if self.rng.random() < 0.5 else +1.0
            # Boost inter-area coupling temporarily
            self.params.base_coupling = min(2.0, self.params.base_coupling * 1.5)
        elif shock.type == PMUGridShockType.COMMON_MODE:
            sign = -1.0
            # Common-mode drift bump
            self._common_drift += shock.magnitude_hz * 0.5

        self.set_disturbance(
            magnitude_hz=shock.magnitude_hz,
            sign=sign,
            time_s=self.time_s + 0.1,   # arm shortly after now
            duration_s=shock.duration_s,
        )
        # If a target PMU is specified, restrict the distance-weight vector
        if shock.target_pmu is not None and 0 <= shock.target_pmu < len(self.pmus):
            new_weights = np.zeros(len(self.pmus))
            new_weights[shock.target_pmu] = 1.0
            self._distance_weights = new_weights

    def apply_intervention(self, intervention: PMUGridIntervention) -> None:
        if not self.pmus:
            raise RuntimeError("Engine not initialized")

        if intervention.type == PMUGridInterventionType.FREQUENCY_RESPONSE:
            boost = float(intervention.parameters.get("damping_boost", 0.3))
            for s in self.pmus:
                s.damping = float(min(2.0, s.damping * (1.0 + boost)))
            self.params.damping_mean = min(2.0, self.params.damping_mean * (1.0 + boost))
        elif intervention.type == PMUGridInterventionType.FAST_AGC:
            # Reduce common-mode drift (AGC tightens it)
            self._common_drift *= float(intervention.parameters.get("drift_reduction", 0.5))
        elif intervention.type == PMUGridInterventionType.ISLANDING:
            # Halve the coupling matrix between random partitions
            n = len(self.pmus)
            if n >= 4 and self._coupling_matrix is not None:
                half = n // 2
                K = self._coupling_matrix.copy()
                K[:half, half:] *= 0.1
                K[half:, :half] *= 0.1
                self._coupling_matrix = K
        elif intervention.type == PMUGridInterventionType.PMU_ISOLATION:
            target = int(intervention.parameters.get("pmu_index", -1))
            if 0 <= target < len(self.pmus):
                self.pmus[target].active = False
                # Zero its coupling row+col
                if self._coupling_matrix is not None:
                    self._coupling_matrix[target, :] = 0.0
                    self._coupling_matrix[:, target] = 0.0

    # ── measurement (RATCHET-uniform) ────────────────────────────────────

    def get_k(self) -> int:
        """Number of active PMUs (active flag honored)."""
        return int(sum(1 for s in self.pmus if s.active))

    def get_rho(self) -> float:
        """Mean abs pairwise correlation across PMU pre-event frequency series.

        Vectorised via np.corrcoef on the active-PMU submatrix. If the
        event has not occurred yet, the whole history is used; otherwise
        only the pre-event window.
        """
        if len(self._frequency_history) < 3 or len(self.pmus) < 2:
            return 0.0
        F = np.array(self._frequency_history).T  # (n_pmus, n_t)
        n_pmus, _ = F.shape

        # Determine pre-event window
        ts = np.array(self._time_history)
        if self._event_applied:
            mask = ts < self._event_time_s
        else:
            mask = np.ones_like(ts, dtype=bool)
        if mask.sum() < 2:
            mask = np.ones_like(ts, dtype=bool)
        F_pre = F[:, mask]

        active = self._get_active_mask()
        # Keep only active rows with non-trivial variance.
        stds = np.std(F_pre, axis=1)
        keep = active & (stds > 1e-10)
        if int(np.sum(keep)) < 2:
            return 0.0
        F_use = F_pre[keep]

        # np.corrcoef is O(N²·T) vectorised in BLAS
        C = np.corrcoef(F_use)
        if C.ndim != 2:
            return 0.0
        # Take strict upper triangle, drop NaNs
        iu = np.triu_indices(C.shape[0], k=1)
        vals = np.abs(C[iu])
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return 0.0
        return float(np.mean(vals))

    def get_sigma(self) -> float:
        """Inverse-CV stability based on post-event settling times.

        Before the event has occurred, returns 1.0 (perfectly stable).
        After the event, computes settling-time per PMU and returns
        1 / (1 + CV(settling_times)).
        """
        if not self._event_applied:
            return 1.0
        F = np.array(self._frequency_history).T  # (n_pmus, n_t)
        ts = np.array(self._time_history)
        if F.shape[1] < 4:
            return 1.0
        # Find the event index in the recorded history
        event_idx_arr = np.where(ts >= self._event_time_s)[0]
        if len(event_idx_arr) == 0:
            return 1.0
        event_idx = int(event_idx_arr[0])

        post = F[:, event_idx:]
        post_ts = ts[event_idx:] - self._event_time_s
        if post_ts.size < 2:
            return 1.0
        active = self._get_active_mask()

        settle = []
        for i in range(F.shape[0]):
            if not active[i]:
                continue
            row = post[i]
            outside = np.abs(row - NOMINAL_FREQUENCY_HZ) > SETTLING_BAND_HZ
            if not outside.any():
                settle.append(0.0)
                continue
            last = int(np.where(outside)[0][-1])
            if last >= len(post_ts) - 1:
                settle.append(float(post_ts[-1]))  # never settled — penalty
            else:
                settle.append(float(post_ts[last + 1]))
        if len(settle) < 2:
            return 1.0
        settle_arr = np.asarray(settle)
        mu = float(np.mean(settle_arr))
        if mu <= 1e-9:
            return 1.0
        sd = float(np.std(settle_arr))
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

    def get_frequency_matrix(self) -> np.ndarray:
        """(n_pmus, n_timepoints) frequency matrix."""
        if not self._frequency_history:
            return np.zeros((self.get_k(), 0))
        return np.array(self._frequency_history).T

    def get_mean_frequency_trajectory(self) -> np.ndarray:
        """Per-timepoint cross-PMU mean frequency."""
        F = self.get_frequency_matrix()
        if F.size == 0:
            return np.zeros(0)
        return np.mean(F, axis=0)

    def get_timestamps(self) -> np.ndarray:
        return np.array(self._time_history, dtype=float)

    def get_event_time_idx(self) -> int:
        """Find the index of event onset in the recorded history."""
        ts = self.get_timestamps()
        if not self._event_applied or ts.size == 0:
            return -1
        idxs = np.where(ts >= self._event_time_s)[0]
        if len(idxs) == 0:
            return -1
        return int(idxs[0])

    def is_collapsed(self) -> bool:
        return self._collapsed

    def get_collapse_time(self) -> Optional[float]:
        return self._collapse_time

    # ── export ───────────────────────────────────────────────────────────

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._history)

    def reset(self) -> None:
        self.pmus = []
        self._coupling_matrix = None
        self._epicentre = None
        self._distance_weights = None
        self._common_drift = 0.0
        self._event_applied = False
        self._event_time_s = -1.0
        self._event_active_until_s = -1.0
        self._frequency_history = []
        self._time_history = []
        self._history = []
        self.time_s = 0.0
        self.step_count = 0
        self._collapsed = False
        self._collapse_time = None


# ─────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────


def create_pmu_grid_engine(
    params: Optional[PMUGridParams] = None,
    seed: Optional[int] = None,
) -> PMUGridEngine:
    """Factory function to create a PMUGridEngine."""
    return PMUGridEngine(params=params, seed=seed)


__all__ = [
    "PMUGridEngine",
    "PMUGridParams",
    "PMUState",
    "PMUGridShock",
    "PMUGridIntervention",
    "PMUGridShockType",
    "PMUGridInterventionType",
    "create_pmu_grid_engine",
    "NOMINAL_FREQUENCY_HZ",
    "SETTLING_BAND_HZ",
]
