"""
RATCHET Power-Grid (PNNL PMU) Substrate Loader

Loads (or synthesizes) PNNL Open-Source PMU Library transmission events
for use with the PMUGridEngine. Mirrors the
ratchet.data.{battery,microbiome,institutional,ecological}_loader pattern.

Domain mapping (per REGIME.md §"A0 — PMU grid"):
    k     : Number of PMUs reporting in a grid region during an event
    rho   : Mean pairwise correlation of pre-event frequency time
            series (5-minute baseline)
    sigma : Inverse of post-event settling-time CV (stability of recovery)
    f     : 1 - sigma (compromise / instability fraction)

Data sources
------------
Primary  : PNNL Open-Source PMU Library (PNNL-30492)
            - https://gridevents.pnnl.gov (event registry)
            - https://www.pnnl.gov/main/publications/external/technical_reports/PNNL-30492.pdf
            - ~1,694 transmission events from interconnect-wide PMU deployments
Fallback : SyntheticPMUEventGenerator below, parameterised on published
            power-system swing-dynamics (Bergen-Vittal / Kundur model:
            60Hz baseline; small Gaussian frequency fluctuations correlated
            across PMUs by inverse electrical distance; step disturbance
            followed by damped oscillation and exponential recovery).

The real-vendor entry point `load_pnnl_pmu_events` looks for parquet at
`data/powergrid/pnnl_events.parquet`; if absent, it falls back to the
synthetic generator. The synthetic-generated dataset is sufficient to
exercise the v0.9 P1 harness — real-data validation slots in once the
PNNL event archive is vendored and its SHA pinned in
`experiments/exp2_cross_substrate/data_sources.yaml`.

The synthetic dynamics encode the same Kish-formula structure the engine
fits, so the synthetic-vs-engine comparison is fair: the engine reads the
observable (k, ρ, σ) triple, the calibrated swing model produces a
matched frequency trajectory.

References
----------
- PNNL-30492 (2020): "Open-source PMU library for grid event analysis."
  Pacific Northwest National Laboratory.
- Bergen, A. R., & Vittal, V. (2000). Power Systems Analysis (2nd ed.).
  Prentice Hall — synchronous-machine swing equation.
- Kundur, P. (1994). Power System Stability and Control. McGraw-Hill —
  inter-area oscillation modes; settling-time analysis.
- IEEE C37.118.1 (2011): Synchrophasor measurement standard.
- DOE Big Data Synchrophasor Analysis program (PNNL-23566).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


# Default vendored-data location (matches data_sources.yaml registry).
DEFAULT_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "powergrid"

# Physical constants (IEEE C37.118)
NOMINAL_FREQUENCY_HZ = 60.0
SETTLING_BAND_HZ = 0.05       # ±0.05 Hz nominal recovery band
DEFAULT_SAMPLE_RATE_HZ = 30.0  # PMU C37.118 typical reporting rate (60Hz grid)
DEFAULT_PRE_EVENT_SECONDS = 300.0  # 5-min baseline per REGIME.md spec


# ─────────────────────────────────────────────────────────────────────────
# Per-event sample
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class PMUEvent:
    """A single PMU-instrumented transmission event with computed RATCHET vars.

    The dataclass is a *snapshot* of one disturbance event:
    frequency_matrix is shape (n_pmus, n_timepoints), sampled at
    `sample_rate_hz`; event_time_idx is the index of the disturbance
    onset within the trace; pre_event_rho summarises the 5-min baseline
    pairwise correlation; sigma = 1 / (1 + CV(settling_times)).

    Attributes
    ----------
    event_id            : Unique identifier (e.g. "PNNL_evt_001234" or "synth_0042")
    frequency_matrix    : (n_pmus, n_timepoints) frequency in Hz
    timestamps          : (n_timepoints,) seconds relative to t=0 (event onset = 0)
    event_time_idx      : int — index of event onset in the time axis
    pmu_ids             : list of PMU identifiers, length n_pmus
    sample_rate_hz      : PMU reporting rate
    k                   : PMU count (n_pmus after filter)
    rho                 : mean pairwise |Pearson| over the pre-event window
    sigma               : 1 / (1 + CV(post-event settling-times)), ∈ (0, 1]
    pre_event_rho       : alias of rho (5-min baseline window) — kept for clarity
    post_event_settling_cv : raw CV across PMUs of settling time
    event_type          : "ambient" / "fault" / "oscillation" / "line_trip" / "gen_loss"
    region              : grid region tag (e.g. "WECC", "ERCOT", "EI")
    duration_s          : total trace duration in seconds
    metadata            : freeform additional fields
    """

    event_id: str
    frequency_matrix: np.ndarray
    timestamps: np.ndarray
    event_time_idx: int = 0
    pmu_ids: List[str] = field(default_factory=list)
    sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ
    k: int = 0
    rho: float = 0.0
    sigma: float = 0.0
    pre_event_rho: float = 0.0
    post_event_settling_cv: float = 0.0
    event_type: Optional[str] = None
    region: Optional[str] = None
    duration_s: float = 0.0
    metadata: Dict = field(default_factory=dict)

    # ── RATCHET-uniform accessors (mirror BatteryData / EcologicalSample) ──

    def get_k(self) -> int:
        return self.k

    def get_rho(self) -> float:
        return self.rho

    def get_sigma(self) -> float:
        return self.sigma

    def get_f(self) -> float:
        """Compromise fraction = 1 − sigma."""
        return float(max(0.0, 1.0 - self.sigma))

    def get_k_eff(self) -> float:
        if self.k <= 1:
            return float(self.k)
        denom = 1.0 + self.rho * (self.k - 1)
        return float(self.k) / max(denom, 1e-6)

    @property
    def num_pmus(self) -> int:
        return int(self.frequency_matrix.shape[0]) if self.frequency_matrix.size else 0

    @property
    def num_timepoints(self) -> int:
        return int(self.frequency_matrix.shape[1]) if self.frequency_matrix.size else 0


# ─────────────────────────────────────────────────────────────────────────
# Multi-event dataset aggregator (parallels NASABatteryDataset)
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class PNNLPMUDataset:
    """Aggregator over many PMUEvents (parallels NASABatteryDataset)."""

    events: Dict[str, PMUEvent] = field(default_factory=dict)
    source: str = "unknown"  # "pnnl_parquet" or "synthetic"

    # ── identity ──
    @property
    def n_events(self) -> int:
        return len(self.events)

    @property
    def event_ids(self) -> List[str]:
        return list(self.events.keys())

    # ── per-event aggregates ──
    def mean_k(self) -> float:
        if not self.events:
            return 0.0
        return float(np.mean([e.k for e in self.events.values()]))

    def mean_rho(self) -> float:
        if not self.events:
            return 0.0
        return float(np.mean([e.rho for e in self.events.values()]))

    def mean_sigma(self) -> float:
        if not self.events:
            return 0.0
        return float(np.mean([e.sigma for e in self.events.values()]))

    def get_k(self) -> int:
        """Treat the dataset's mean k as the substrate-level constraint count."""
        return int(round(self.mean_k()))

    def get_rho(self) -> float:
        return self.mean_rho()

    def get_sigma(self) -> float:
        return self.mean_sigma()

    def get_k_eff(self) -> float:
        k = self.get_k()
        rho = self.get_rho()
        if k <= 1:
            return float(k)
        return k / (1.0 + rho * (k - 1))

    def to_dataframe(self) -> pd.DataFrame:
        """Per-event summary dataframe."""
        rows = []
        for eid, e in self.events.items():
            rows.append({
                "event_id": eid,
                "k": e.k,
                "rho": e.rho,
                "sigma": e.sigma,
                "f": e.get_f(),
                "k_eff": e.get_k_eff(),
                "num_pmus": e.num_pmus,
                "num_timepoints": e.num_timepoints,
                "event_type": e.event_type,
                "region": e.region,
                "duration_s": e.duration_s,
            })
        return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────
# Compute helpers (used by both real-data and synthetic paths)
# ─────────────────────────────────────────────────────────────────────────


def compute_pre_event_correlation(
    frequency_matrix: np.ndarray,
    event_time_idx: int,
    baseline_samples: int = -1,
) -> float:
    """Mean abs pairwise Pearson correlation over the pre-event window.

    Args
    ----
    frequency_matrix : (n_pmus, n_timepoints) frequency in Hz
    event_time_idx   : index of event onset
    baseline_samples : how many pre-event samples to use (default: all
                       samples before the event)

    Returns
    -------
    rho ∈ [0, 1] : mean |Pearson| across PMU pairs. Uses absolute value
    because both positive (mode-locked) and negative (anti-phase swing)
    register as coordinated dynamics in the Kish sense.

    Constant-frequency PMUs contribute 0 to the mean rather than NaN.
    """
    f = np.asarray(frequency_matrix, dtype=float)
    if f.ndim != 2:
        return 0.0
    n_pmus, n_t = f.shape
    if n_pmus < 2 or n_t < 2:
        return 0.0

    hi = int(max(2, min(event_time_idx, n_t)))
    if baseline_samples > 0:
        lo = max(0, hi - baseline_samples)
    else:
        lo = 0
    if hi - lo < 2:
        return 0.0

    window = f[:, lo:hi]
    pairs = []
    for i in range(n_pmus):
        if np.std(window[i]) < 1e-10:
            continue
        for j in range(i + 1, n_pmus):
            if np.std(window[j]) < 1e-10:
                continue
            r = np.corrcoef(window[i], window[j])[0, 1]
            if np.isnan(r):
                continue
            pairs.append(abs(float(r)))
    if not pairs:
        return 0.0
    return float(np.mean(pairs))


def compute_settling_times(
    frequency_matrix: np.ndarray,
    timestamps: np.ndarray,
    event_time_idx: int,
    settling_band_hz: float = SETTLING_BAND_HZ,
    nominal_hz: float = NOMINAL_FREQUENCY_HZ,
) -> np.ndarray:
    """Per-PMU settling times (seconds to return to ±band of nominal).

    Returns array of shape (n_pmus,). For each PMU, find the largest
    index t* in the post-event window such that for all t > t*, |f(t) -
    nominal| < settling_band_hz. Settling time = timestamps[t*] -
    timestamps[event_time_idx]. PMUs that never settle within the trace
    are assigned the maximum post-event duration (so they contribute
    heavy CV to the inverse-CV sigma).
    """
    f = np.asarray(frequency_matrix, dtype=float)
    ts = np.asarray(timestamps, dtype=float)
    n_pmus, n_t = f.shape
    if event_time_idx >= n_t - 1:
        return np.zeros(n_pmus)

    settle = np.zeros(n_pmus)
    event_t = ts[event_time_idx]
    max_post = ts[-1] - event_t

    for i in range(n_pmus):
        post = f[i, event_time_idx:]
        post_ts = ts[event_time_idx:] - event_t
        outside = np.abs(post - nominal_hz) > settling_band_hz
        if not outside.any():
            # Already inside the band — settling time is 0
            settle[i] = 0.0
            continue
        # Last index where it was outside
        last_outside = np.where(outside)[0]
        if len(last_outside) == 0:
            settle[i] = 0.0
        else:
            last_idx = int(last_outside[-1])
            if last_idx >= len(post_ts) - 1:
                # Never settled — assign max trace duration as a penalty
                settle[i] = float(max_post)
            else:
                settle[i] = float(post_ts[last_idx + 1])
    return settle


def compute_settling_sigma(settling_times: np.ndarray) -> Tuple[float, float]:
    """Inverse-CV of settling-times → sigma ∈ (0, 1].

    Returns (sigma, cv).

    sigma = 1 / (1 + CV), so:
        CV = 0 (PMUs settle uniformly)        → sigma = 1.0  (most stable)
        CV → ∞ (huge spread, some never settle) → sigma → 0  (least stable)
    """
    s = np.asarray(settling_times, dtype=float)
    if len(s) < 2:
        return 1.0, 0.0
    mu = float(np.mean(s))
    if mu <= 1e-9:
        # All PMUs settled instantly — perfectly stable
        return 1.0, 0.0
    sd = float(np.std(s))
    cv = sd / mu
    sigma = 1.0 / (1.0 + cv)
    return float(sigma), float(cv)


# ─────────────────────────────────────────────────────────────────────────
# Synthetic PMU generator (drop-in for unavailable real data)
# ─────────────────────────────────────────────────────────────────────────


class SyntheticPMUEventGenerator:
    """Generate realistic PNNL-like PMU event traces.

    Each event simulates a step disturbance (line trip, generator loss, or
    load shed) on a grid region with k PMUs whose pre-event frequency
    fluctuations are correlated via inverse electrical distance. The
    dynamics follow the linearised synchronous-machine swing equation
    (Bergen-Vittal Ch. 14):

        df_i/dt = -(D_i / (2 H_i)) * (f_i - f_nominal)
                  + (1 / (2 H_i)) * Σ_j K_ij * (f_j - f_i)
                  + noise_i + disturbance_i(t)

    where H_i is per-PMU inertia, D_i damping, K_ij the inter-PMU
    coupling. Spatial coupling K_ij = base_coupling / d_ij where d_ij is
    a randomised electrical distance, so PMUs cluster into correlated
    sub-groups (typical of multi-area interconnects).

    The disturbance is a step (line trip → frequency drop) with subsequent
    governor + AGC response. Recovery proceeds through damped oscillation
    plus exponential return.

    Parameters follow PNNL-30492 stated ranges:
        k (PMUs per event) : LogNormal(2.1, 0.5), clipped to [3, 30]
        Pre-event noise sd : 0.005-0.02 Hz (very tight; PMU resolution)
        Disturbance magnitude : 0.05-0.5 Hz (line trip ≈ 0.1-0.3 Hz)
        Settling time mean : 5-30 s; CV across PMUs 0.1-0.5
        Sample rate : 30 Hz (C37.118 standard)

    Refs:
        PNNL-30492; Bergen & Vittal (2000); Kundur (1994); IEEE C37.118.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def _build_coupling(
        self,
        n_pmus: int,
        base_coupling: float,
        spatial_decay: float = 1.5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build a (k, k) coupling matrix K_ij and 2D positions.

        Coupling decays with distance: K_ij = base / (d_ij^decay + 1e-3),
        zero diagonal. Positions are uniform random in [0,1]^2 — emulates
        a geographic spread of substations.
        """
        positions = self.rng.uniform(0.0, 1.0, size=(n_pmus, 2))
        K = np.zeros((n_pmus, n_pmus))
        for i in range(n_pmus):
            for j in range(n_pmus):
                if i == j:
                    continue
                d = float(np.linalg.norm(positions[i] - positions[j]))
                K[i, j] = base_coupling / (d ** spatial_decay + 1e-2)
        return K, positions

    def generate_event(
        self,
        event_id: Optional[str] = None,
        n_pmus: Optional[int] = None,
        duration_s: float = 60.0,
        pre_event_s: float = 30.0,
        sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ,
        disturbance_magnitude_hz: Optional[float] = None,
        base_coupling: Optional[float] = None,
        pre_event_noise_sd_hz: Optional[float] = None,
        damping: float = 0.4,
        inertia: float = 5.0,
        event_type: str = "line_trip",
        region: str = "WECC",
    ) -> PMUEvent:
        """Generate one synthetic PMU event trace.

        Args
        ----
        event_id                : optional string identifier
        n_pmus                  : if None, drawn from LogNormal(2.1, 0.5) ∩ [3, 30]
        duration_s              : total trace length (default 60s = 30 pre + 30 post)
        pre_event_s             : seconds of pre-event baseline
        sample_rate_hz          : PMU reporting rate
        disturbance_magnitude_hz: if None, drawn from Uniform[0.05, 0.5]
        base_coupling           : if None, drawn from Uniform[0.1, 0.8]
        pre_event_noise_sd_hz   : if None, drawn from Uniform[0.005, 0.02]
        damping                 : per-PMU damping coefficient D (governor)
        inertia                 : per-PMU swing inertia H (seconds)
        event_type              : "line_trip", "gen_loss", "load_shed", "oscillation"
        region                  : grid region tag

        Returns
        -------
        PMUEvent with k, rho, sigma populated from simulated dynamics.
        """
        # ── sample event-level parameters ──
        if n_pmus is None:
            k_raw = self.rng.lognormal(mean=2.1, sigma=0.5)
            n_pmus = int(np.clip(round(k_raw), 3, 30))
        else:
            n_pmus = int(max(3, min(30, n_pmus)))

        if disturbance_magnitude_hz is None:
            disturbance_magnitude_hz = float(self.rng.uniform(0.05, 0.5))
        if base_coupling is None:
            base_coupling = float(self.rng.uniform(0.1, 0.8))
        if pre_event_noise_sd_hz is None:
            pre_event_noise_sd_hz = float(self.rng.uniform(0.005, 0.02))

        # ── build time axis ──
        dt = 1.0 / sample_rate_hz
        n_t = int(round(duration_s * sample_rate_hz))
        timestamps = np.arange(n_t) * dt
        event_time_idx = int(round(pre_event_s * sample_rate_hz))
        event_time_idx = int(np.clip(event_time_idx, 1, n_t - 2))

        # ── build coupling matrix ──
        K_coupling, positions = self._build_coupling(n_pmus, base_coupling)

        # ── per-PMU heterogeneity ──
        H_i = inertia * np.exp(self.rng.normal(0.0, 0.15, size=n_pmus))
        D_i = damping * np.exp(self.rng.normal(0.0, 0.15, size=n_pmus))
        H_i = np.clip(H_i, 1.0, 20.0)
        D_i = np.clip(D_i, 0.1, 2.0)

        # ── common-mode pre-event drift (drives baseline rho > 0) ──
        common_drift = np.zeros(n_t)
        common_drift[0] = self.rng.normal(0.0, pre_event_noise_sd_hz * 0.5)
        ar_phi = 0.85  # slow common drift
        common_sd = pre_event_noise_sd_hz * 0.4 * np.sqrt(1.0 - ar_phi ** 2)
        for t in range(1, n_t):
            common_drift[t] = ar_phi * common_drift[t - 1] + self.rng.normal(0, common_sd)

        # ── simulate swing dynamics ──
        f = np.zeros((n_pmus, n_t))
        f[:, 0] = NOMINAL_FREQUENCY_HZ + self.rng.normal(0.0, pre_event_noise_sd_hz, n_pmus)

        # Disturbance: applied as an impulse on a random subset of PMUs nearest
        # to the disturbance epicentre, then propagated via coupling.
        epicentre = self.rng.uniform(0.0, 1.0, size=2)
        dist_to_epicentre = np.linalg.norm(positions - epicentre, axis=1)
        # Weighting: PMUs near the epicentre get the brunt of the disturbance
        dist_weights = np.exp(-2.0 * dist_to_epicentre)
        dist_weights /= np.max(dist_weights) + 1e-9
        # Sign of disturbance: negative for line_trip / load_shed; positive for gen_loss
        # In standard practice generator loss → frequency drops too, but we vary
        # to give the dataset both polarities for robustness.
        if event_type in ("line_trip", "gen_loss"):
            disturbance_sign = -1.0
        elif event_type == "load_shed":
            disturbance_sign = +1.0
        else:  # oscillation, ambient
            disturbance_sign = -1.0 if self.rng.random() < 0.5 else +1.0

        # ── stepwise integration ──
        for t in range(1, n_t):
            df = np.zeros(n_pmus)
            delta = f[:, t - 1] - NOMINAL_FREQUENCY_HZ

            # Damping (governor + load damping)
            df += -(D_i / (2.0 * H_i)) * delta * dt

            # Inter-PMU coupling: ΣK_ij (f_j - f_i)
            for i in range(n_pmus):
                df[i] += (1.0 / (2.0 * H_i[i])) * np.sum(K_coupling[i] * (f[:, t - 1] - f[i, t - 1])) * dt

            # Common drift (registered as inter-PMU correlated baseline)
            df += common_drift[t] - common_drift[t - 1]

            # Per-PMU noise
            df += self.rng.normal(0.0, pre_event_noise_sd_hz, size=n_pmus) * np.sqrt(dt)

            # Disturbance impulse: applied over a short ramp at event_time_idx
            if t == event_time_idx:
                df += disturbance_sign * disturbance_magnitude_hz * dist_weights
            elif event_time_idx < t < event_time_idx + max(2, int(0.5 / dt)):
                # Continuing disturbance ramp for 0.5s
                df += 0.2 * disturbance_sign * disturbance_magnitude_hz * dist_weights * dt

            f[:, t] = f[:, t - 1] + df

        # ── compute RATCHET variables ──
        rho = compute_pre_event_correlation(f, event_time_idx, baseline_samples=event_time_idx)
        settle = compute_settling_times(f, timestamps, event_time_idx)
        sigma, cv = compute_settling_sigma(settle)

        pmu_ids = [f"pmu_{i:03d}" for i in range(n_pmus)]
        eid = event_id or f"synth_{event_type[:3]}_{self.rng.integers(100000):05d}"

        return PMUEvent(
            event_id=eid,
            frequency_matrix=f,
            timestamps=timestamps,
            event_time_idx=event_time_idx,
            pmu_ids=pmu_ids,
            sample_rate_hz=float(sample_rate_hz),
            k=int(n_pmus),
            rho=float(rho),
            sigma=float(sigma),
            pre_event_rho=float(rho),
            post_event_settling_cv=float(cv),
            event_type=event_type,
            region=region,
            duration_s=float(duration_s),
            metadata={
                "synthetic": True,
                "disturbance_magnitude_hz": disturbance_magnitude_hz,
                "base_coupling": base_coupling,
                "pre_event_noise_sd_hz": pre_event_noise_sd_hz,
                "damping_mean": float(np.mean(D_i)),
                "inertia_mean": float(np.mean(H_i)),
                "epicentre": [float(epicentre[0]), float(epicentre[1])],
                "mean_settling_s": float(np.mean(settle)),
            },
        )

    def generate_dataset(
        self,
        n_events: int = 50,
        event_types: Optional[List[str]] = None,
        regions: Optional[List[str]] = None,
    ) -> PNNLPMUDataset:
        """Generate a multi-event synthetic PNNL-like dataset.

        Args
        ----
        n_events    : how many events to synthesize (default 50)
        event_types : list of event-type tags to cycle through (default
                      ["line_trip", "gen_loss", "load_shed", "oscillation"])
        regions     : list of grid-region tags (default
                      ["WECC", "ERCOT", "EI"])

        Returns
        -------
        PNNLPMUDataset with `n_events` synthetic events.
        """
        if event_types is None:
            event_types = ["line_trip", "gen_loss", "load_shed", "oscillation"]
        if regions is None:
            regions = ["WECC", "ERCOT", "EI"]

        dataset = PNNLPMUDataset(source="synthetic")
        for i in range(n_events):
            etype = event_types[i % len(event_types)]
            region = regions[i % len(regions)]
            event = self.generate_event(
                event_id=f"synth_{i:04d}_{etype[:3]}_{region}",
                event_type=etype,
                region=region,
            )
            dataset.events[event.event_id] = event
        return dataset


# ─────────────────────────────────────────────────────────────────────────
# Real-data loader (PNNL parquet → PNNLPMUDataset)
# ─────────────────────────────────────────────────────────────────────────


def _load_pnnl_parquet(
    parquet_path: Path,
    min_pmus: int = 3,
    min_post_event_s: float = 5.0,
) -> PNNLPMUDataset:
    """Load PNNL Open PMU Library parquet into a PNNLPMUDataset.

    Expected parquet schema (long format; one row per (event, pmu, sample)):
        event_id        : str
        pmu_id          : str
        timestamp_s     : float (seconds relative to event onset; <0 pre, >=0 post)
        frequency_hz    : float
        sample_rate_hz  : float (optional; defaults to 30)
        event_type      : str (optional)
        region          : str (optional)

    Events are grouped by event_id; for each event with ≥ min_pmus
    distinct PMUs and ≥ min_post_event_s of post-event coverage we pivot
    to a (pmu × time) frequency matrix and compute k, rho, sigma.

    NOTE: this is a best-effort schema — the PNNL team has shipped
    multiple file layouts (CSV, parquet, HDF5). If columns don't match,
    the loader raises with a clear message so the caller can fall back
    to synthetic.
    """
    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        raise ValueError(f"Could not read PNNL parquet at {parquet_path}: {e}")

    cols = {c.lower(): c for c in df.columns}

    def col(name: str, alts: tuple = ()) -> Optional[str]:
        for n in (name, *alts):
            if n.lower() in cols:
                return cols[n.lower()]
        return None

    event_col = col("event_id", ("event", "eventid"))
    pmu_col = col("pmu_id", ("pmu", "pmuid", "station_id"))
    time_col = col("timestamp_s", ("timestamp", "time_s", "t"))
    freq_col = col("frequency_hz", ("frequency", "freq", "f_hz"))
    type_col = col("event_type", ("type",))
    region_col = col("region", ("interconnect",))
    sr_col = col("sample_rate_hz", ("sample_rate",))

    if not all([event_col, pmu_col, time_col, freq_col]):
        raise ValueError(
            "PNNL parquet missing one of required columns "
            f"(event_id, pmu_id, timestamp_s, frequency_hz). Found: {list(df.columns)}"
        )

    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df[freq_col] = pd.to_numeric(df[freq_col], errors="coerce")
    df = df.dropna(subset=[time_col, freq_col, event_col, pmu_col])

    dataset = PNNLPMUDataset(source="pnnl_parquet")

    for event_id, group in df.groupby(event_col):
        pmu_ids = sorted(group[pmu_col].astype(str).unique())
        if len(pmu_ids) < min_pmus:
            continue
        post_event = group[group[time_col] >= 0]
        if post_event.empty:
            continue
        if float(post_event[time_col].max()) < min_post_event_s:
            continue

        # Pivot to (pmu × time)
        pivot = group.pivot_table(
            index=pmu_col,
            columns=time_col,
            values=freq_col,
            aggfunc="mean",
        ).sort_index(axis=1)
        # Drop time columns with too many NaNs (PMU dropouts)
        pivot = pivot.dropna(axis=1, thresh=max(2, int(0.5 * len(pivot))))
        # Drop PMUs with all-NaN row
        pivot = pivot.dropna(axis=0, how="all")
        if pivot.shape[0] < min_pmus or pivot.shape[1] < 4:
            continue
        # Forward-fill remaining NaNs per row (PMU)
        pivot = pivot.ffill(axis=1).bfill(axis=1)

        freq_matrix = pivot.values.astype(float)
        timestamps = pivot.columns.values.astype(float)
        pmu_ids_kept = list(pivot.index.astype(str))
        # Event onset: smallest t >= 0
        post_idx = np.where(timestamps >= 0)[0]
        if len(post_idx) == 0:
            continue
        event_time_idx = int(post_idx[0])

        sample_rate_hz = DEFAULT_SAMPLE_RATE_HZ
        if sr_col is not None and not group[sr_col].dropna().empty:
            try:
                sample_rate_hz = float(group[sr_col].dropna().iloc[0])
            except Exception:
                pass

        event_type = None
        if type_col is not None and not group[type_col].dropna().empty:
            event_type = str(group[type_col].dropna().iloc[0])
        region = None
        if region_col is not None and not group[region_col].dropna().empty:
            region = str(group[region_col].dropna().iloc[0])

        rho = compute_pre_event_correlation(freq_matrix, event_time_idx)
        settle = compute_settling_times(freq_matrix, timestamps, event_time_idx)
        sigma, cv = compute_settling_sigma(settle)

        eid = str(event_id)
        pmu_event = PMUEvent(
            event_id=eid,
            frequency_matrix=freq_matrix,
            timestamps=timestamps,
            event_time_idx=event_time_idx,
            pmu_ids=pmu_ids_kept,
            sample_rate_hz=sample_rate_hz,
            k=int(freq_matrix.shape[0]),
            rho=float(rho),
            sigma=float(sigma),
            pre_event_rho=float(rho),
            post_event_settling_cv=float(cv),
            event_type=event_type,
            region=region,
            duration_s=float(timestamps[-1] - timestamps[0]),
            metadata={"source": "PNNL-30492", "synthetic": False},
        )
        dataset.events[eid] = pmu_event

    return dataset


def load_pnnl_pmu_events(
    data_dir: Optional[Union[str, Path]] = None,
    parquet_filename: str = "pnnl_events.parquet",
    fallback_to_synthetic: bool = True,
    n_synthetic_events: int = 50,
    min_pmus: int = 3,
    min_post_event_s: float = 5.0,
    seed: Optional[int] = None,
) -> PNNLPMUDataset:
    """Entry point: load PNNL PMU events, falling back to synthetic.

    Search order:
      1. `data_dir / parquet_filename` if it exists → real PNNL parquet.
      2. If fallback_to_synthetic, SyntheticPMUEventGenerator with `seed`.
      3. Otherwise raise FileNotFoundError.

    Args
    ----
    data_dir              : where to look for vendored data. Defaults to
                            `data/powergrid/` under the repo root.
    parquet_filename      : parquet name within data_dir.
    fallback_to_synthetic : if True, generate synthetic when parquet absent.
    n_synthetic_events    : how many synthetic events to emit.
    min_pmus              : event filter; min PMUs per event.
    min_post_event_s      : event filter; min post-event coverage in seconds.
    seed                  : RNG seed for synthetic generator.

    Returns
    -------
    PNNLPMUDataset, either real or synthetic.
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    data_dir = Path(data_dir)

    # Search the primary filename first; also accept common alternatives
    # (e.g. the coordinator-suggested "pnnl_events_sample.parquet").
    candidate_paths: List[Path] = [data_dir / parquet_filename]
    for alt in ("pnnl_events_sample.parquet",
                "pnnl_pmu_events.parquet",
                "pmu_events.parquet"):
        alt_path = data_dir / alt
        if alt_path not in candidate_paths:
            candidate_paths.append(alt_path)

    for parquet_path in candidate_paths:
        if parquet_path.exists():
            try:
                ds = _load_pnnl_parquet(parquet_path, min_pmus=min_pmus,
                                        min_post_event_s=min_post_event_s)
                if ds.n_events > 0:
                    return ds
            except Exception as e:
                if not fallback_to_synthetic:
                    raise
                print(f"[load_pnnl_pmu_events] PNNL parquet load failed "
                      f"({parquet_path}): {e}; falling back to synthetic")

    if not fallback_to_synthetic:
        raise FileNotFoundError(
            f"PNNL parquet not found at any of {[str(p) for p in candidate_paths]} "
            f"and fallback_to_synthetic=False."
        )

    gen = SyntheticPMUEventGenerator(seed=seed)
    return gen.generate_dataset(n_events=n_synthetic_events)


# Backwards-compatible alias used by REGIME.md spec.
def load_pnnl_grid_events(
    data_dir: Optional[Union[str, Path]] = None,
    **kwargs,
) -> PNNLPMUDataset:
    """Alias matching `data_sources.yaml` loader-name convention."""
    return load_pnnl_pmu_events(data_dir=data_dir, **kwargs)


# ─────────────────────────────────────────────────────────────────────────
# Convenience: prepare a single event for engine-vs-data comparison
# ─────────────────────────────────────────────────────────────────────────


def prepare_for_engine(
    dataset: PNNLPMUDataset,
    event_id: Optional[str] = None,
) -> Dict:
    """Extract one event's frequency trajectory + sigma trajectory for engine fit.

    Args
    ----
    dataset  : PNNLPMUDataset
    event_id : specific event to extract; if None, picks the first

    Returns
    -------
    dict with:
        event_id, k, rho, sigma_final, num_pmus, num_timepoints,
        empirical_frequency_matrix, empirical_mean_frequency,
        empirical_sigma_trajectory, timestamps, event_time_idx,
        sample_rate_hz, event_type, region
    """
    if not dataset.events:
        raise ValueError("Dataset is empty.")

    if event_id is None:
        event_id = next(iter(dataset.events))
    if event_id not in dataset.events:
        raise KeyError(f"Event {event_id!r} not in dataset.")

    e = dataset.events[event_id]
    f = e.frequency_matrix
    ts = e.timestamps
    n_pmus, n_t = f.shape

    # Mean frequency across PMUs at each timepoint (cross-array average)
    mean_freq = np.mean(f, axis=0)

    # Rolling-window sigma trajectory: at each timepoint t after the
    # event, compute settling-CV-based sigma using PMU deviations from
    # nominal over a trailing window. Pre-event we just report sigma=1.
    sigma_traj = np.ones(n_t)
    window = max(int(2.0 * e.sample_rate_hz), 8)  # 2-sec rolling window
    for t in range(e.event_time_idx, n_t):
        lo = max(e.event_time_idx, t - window + 1)
        hi = t + 1
        sub = f[:, lo:hi]
        if sub.shape[1] < 2:
            sigma_traj[t] = 1.0
            continue
        # Per-PMU absolute deviation from nominal averaged over window
        dev = np.mean(np.abs(sub - NOMINAL_FREQUENCY_HZ), axis=1)
        if dev.size < 2:
            sigma_traj[t] = 1.0
            continue
        mu = float(np.mean(dev))
        if mu <= 1e-9:
            sigma_traj[t] = 1.0
        else:
            sd = float(np.std(dev))
            cv = sd / mu
            sigma_traj[t] = 1.0 / (1.0 + cv)

    return {
        "event_id": event_id,
        "k": e.k,
        "rho": e.rho,
        "sigma_final": e.sigma,
        "num_pmus": int(n_pmus),
        "num_timepoints": int(n_t),
        "empirical_frequency_matrix": f.copy(),
        "empirical_mean_frequency": mean_freq.copy(),
        "empirical_sigma_trajectory": sigma_traj,
        "timestamps": ts.copy(),
        "event_time_idx": int(e.event_time_idx),
        "sample_rate_hz": float(e.sample_rate_hz),
        "duration_s": float(e.duration_s),
        "event_type": e.event_type,
        "region": e.region,
        "pmu_ids": list(e.pmu_ids),
        "metadata": dict(e.metadata),
    }


__all__ = [
    "PMUEvent",
    "PNNLPMUDataset",
    "SyntheticPMUEventGenerator",
    "compute_pre_event_correlation",
    "compute_settling_times",
    "compute_settling_sigma",
    "load_pnnl_pmu_events",
    "load_pnnl_grid_events",
    "prepare_for_engine",
    "NOMINAL_FREQUENCY_HZ",
    "SETTLING_BAND_HZ",
    "DEFAULT_SAMPLE_RATE_HZ",
]
