"""
RATCHET Macro-Ecology (BioTIME) Substrate Loader

Loads (or synthesizes) BioTIME 2.0 community time-series data for use with
the EcologicalCommunityEngine. Mirrors the
ratchet.data.{battery,microbiome,institutional}_loader pattern.

Domain mapping (per REGIME.md §"A2 — BioTIME macro-ecology"):
    k     : Species count in a community time series
    rho   : Mean pairwise correlation of species-abundance time series
    sigma : Inverse CV of total biomass over time (stability proxy)
    f     : 1 - sigma (compromise / instability fraction)

Data sources
------------
Primary  : BioTIME 2.0 (Dornelas et al. 2025, Global Ecology and Biogeography)
            - https://biotime.st-andrews.ac.uk/downloads.php
            - BioTIMEr R package wrap (CC-BY)
Fallback : SyntheticBioTIMEGenerator below, parameterised on the published
            BioTIME 2.0 distributions. Synthesised communities follow a
            simple logistic-with-cross-coupling dynamic and reproduce the
            published k / rho / sigma marginal distributions adequately
            for engine-vs-data harness wiring.

The real-vendor entry point `load_biotime_data` looks for a CSV at
`data/ecological/biotime_query.csv`; if absent, it falls back to the
synthetic generator. The synthetic-generated dataset is sufficient to
exercise the v0.9 P1 harness — real-data validation slots in once the
BioTIME CSV is vendored and its SHA pinned in
`experiments/exp2_cross_substrate/data_sources.yaml`.

References
----------
- Dornelas, M., et al. (2025). BioTIME 2.0: a database of biodiversity
  time series. Global Ecology and Biogeography.
- Tilman, D. (1996). Biodiversity: Population versus ecosystem stability.
  Ecology, 77(2), 350-363.
- Loreau, M., & de Mazancourt, C. (2013). Biodiversity and ecosystem
  stability: a synthesis of underlying mechanisms. Ecology Letters,
  16(s1), 106-115.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


# Default vendored-data location (matches data_sources.yaml registry).
DEFAULT_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "ecological"


# ─────────────────────────────────────────────────────────────────────────
# Per-community sample
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class EcologicalSample:
    """A single BioTIME community time series with computed RATCHET vars.

    The dataclass is a *snapshot* of one community: species_abundances is
    shape (n_species, n_timepoints) with absolute (or normalised) counts;
    biomass_trajectory is the row-sum across species for each timepoint;
    timestamps is the 0-indexed year array.

    Attributes
    ----------
    community_id : Unique BioTIME study + plot identifier (e.g. "STUDY_339_plot_2")
    species_abundances : (n_species, n_timepoints) abundance matrix
    biomass_trajectory : (n_timepoints,) total biomass per year
    timestamps : (n_timepoints,) year indices (0, 1, …) or actual year-of
    species_ids : list of species identifiers, length n_species
    k : species count (n_species after filter)
    rho : mean pairwise absolute Pearson correlation across species pairs
    sigma : inverse CV = mean(biomass) / std(biomass), normalised to (0, 1]
    realm : "marine" / "terrestrial" / "freshwater" (BioTIME field; optional)
    taxonomic_group : e.g. "Birds", "Fish", "Plants" (optional)
    metadata : freeform additional fields
    """

    community_id: str
    species_abundances: np.ndarray
    biomass_trajectory: np.ndarray
    timestamps: np.ndarray
    species_ids: List[str] = field(default_factory=list)
    k: int = 0
    rho: float = 0.0
    sigma: float = 0.0
    realm: Optional[str] = None
    taxonomic_group: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

    # ── RATCHET-uniform accessors (mirror BatteryData / MicrobiomeSample) ──

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
    def num_years(self) -> int:
        return int(len(self.timestamps))


# ─────────────────────────────────────────────────────────────────────────
# Multi-community dataset aggregator (parallels NASABatteryDataset)
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class BioTIMECommunityDataset:
    """Aggregator over many EcologicalSamples (parallels NASABatteryDataset)."""

    communities: Dict[str, EcologicalSample] = field(default_factory=dict)
    source: str = "unknown"  # "biotime_csv" or "synthetic"

    # ── identity ──
    @property
    def n_communities(self) -> int:
        return len(self.communities)

    @property
    def community_ids(self) -> List[str]:
        return list(self.communities.keys())

    # ── per-community aggregates ──
    def mean_k(self) -> float:
        if not self.communities:
            return 0.0
        return float(np.mean([c.k for c in self.communities.values()]))

    def mean_rho(self) -> float:
        if not self.communities:
            return 0.0
        return float(np.mean([c.rho for c in self.communities.values()]))

    def mean_sigma(self) -> float:
        if not self.communities:
            return 0.0
        return float(np.mean([c.sigma for c in self.communities.values()]))

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
        """Per-community summary dataframe."""
        rows = []
        for cid, c in self.communities.items():
            rows.append({
                "community_id": cid,
                "k": c.k,
                "rho": c.rho,
                "sigma": c.sigma,
                "f": c.get_f(),
                "k_eff": c.get_k_eff(),
                "num_years": c.num_years,
                "realm": c.realm,
                "taxonomic_group": c.taxonomic_group,
            })
        return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────
# Compute helpers (used by both real-data and synthetic paths)
# ─────────────────────────────────────────────────────────────────────────


def compute_abundance_correlation(species_abundances: np.ndarray) -> float:
    """Mean absolute pairwise Pearson correlation across species pairs.

    Args
    ----
    species_abundances : array of shape (n_species, n_timepoints)

    Returns
    -------
    rho ∈ [0, 1] : mean |Pearson| across all species pairs. Uses absolute
    value because both positive (mutualism/coupling) and negative
    (competition/anti-phase) correlations register as coordinated dynamics
    in the Kish-formula sense (information-equivalent).

    A trivially-constant species (zero variance) contributes 0 to the mean
    rather than NaN — same convention as scipy.stats.pearsonr edge case.
    """
    a = np.asarray(species_abundances, dtype=float)
    if a.ndim != 2:
        return 0.0
    n_sp, n_t = a.shape
    if n_sp < 2 or n_t < 2:
        return 0.0

    pairs = []
    for i in range(n_sp):
        if np.std(a[i]) < 1e-10:
            continue
        for j in range(i + 1, n_sp):
            if np.std(a[j]) < 1e-10:
                continue
            r = np.corrcoef(a[i], a[j])[0, 1]
            if np.isnan(r):
                continue
            pairs.append(abs(float(r)))

    if not pairs:
        return 0.0
    return float(np.mean(pairs))


def compute_biomass_stability(biomass_trajectory: np.ndarray) -> float:
    """Inverse CV of total biomass: stability = 1 / CV, bounded to (0, 1].

    Higher value = more stable ecosystem. We squash 1/CV through a soft
    sigmoid-style normalisation so sigma ∈ (0, 1] like other substrates.

    Args
    ----
    biomass_trajectory : (n_timepoints,) array

    Returns
    -------
    sigma ∈ (0, 1] : `1 / (1 + CV)`. CV = 0 (perfect stability) → sigma = 1.
    CV → ∞ (unbounded variability) → sigma → 0.
    """
    b = np.asarray(biomass_trajectory, dtype=float)
    if len(b) < 2:
        return 1.0
    mu = float(np.mean(b))
    if mu <= 1e-10:
        return 0.0
    sd = float(np.std(b))
    cv = sd / mu
    return float(1.0 / (1.0 + cv))


# ─────────────────────────────────────────────────────────────────────────
# Synthetic BioTIME generator (drop-in for unavailable real data)
# ─────────────────────────────────────────────────────────────────────────


class SyntheticBioTIMEGenerator:
    """Generate realistic BioTIME-like community time series.

    Each community is simulated with logistic-with-cross-coupling species
    dynamics over an integer number of years (default 10-50). Parameters
    follow the published BioTIME 2.0 marginal distributions:

        k (species count) : LogNormal(mu=2.3, sigma=0.4), clipped to [5, 30]
        years             : Uniform integer [10, 50]
        intrinsic growth  : Normal(0.4, 0.1) per species per year
        carrying capacity : LogNormal(mu=3.5, sigma=0.5) per species
        coupling strength : 0.05 baseline, +per-community noise

    The cross-species coupling matrix is symmetric Gaussian, so it
    induces positive correlations between species (mutualism case).
    Higher coupling → higher rho. Each community is given an environmental
    forcing common term that drives the floor of inter-species correlation.

    Refs:
        Dornelas et al. (2025); Loreau & de Mazancourt (2013) on stability;
        Coyte et al. (2015) Science on competitive vs cooperative ecosystems.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def generate_community(
        self,
        community_id: Optional[str] = None,
        n_species: Optional[int] = None,
        n_years: Optional[int] = None,
        coupling_strength: Optional[float] = None,
        env_forcing_amp: Optional[float] = None,
        realm: str = "terrestrial",
        taxonomic_group: str = "Birds",
    ) -> EcologicalSample:
        """Generate a single synthetic BioTIME-style community time series.

        Args
        ----
        community_id      : optional string identifier
        n_species         : if None, drawn from LogNormal(2.3, 0.4) ∩ [5, 30]
        n_years           : if None, drawn from Uniform[10, 50]
        coupling_strength : if None, drawn from Beta(2, 5) * 0.4 ∈ [0, 0.4]
        env_forcing_amp   : if None, drawn from Uniform[0.05, 0.3]
        realm             : one of "marine"/"terrestrial"/"freshwater"
        taxonomic_group   : tag for downstream filtering

        Returns
        -------
        EcologicalSample with k, rho, sigma populated from simulated dynamics.
        """
        # ── sample community-level parameters ──
        if n_species is None:
            k_raw = self.rng.lognormal(mean=2.3, sigma=0.4)
            n_species = int(np.clip(round(k_raw), 5, 30))
        else:
            n_species = int(max(5, min(30, n_species)))

        if n_years is None:
            n_years = int(self.rng.integers(10, 51))
        else:
            n_years = int(max(10, min(100, n_years)))

        if coupling_strength is None:
            coupling_strength = float(self.rng.beta(2.0, 5.0) * 0.4)

        if env_forcing_amp is None:
            env_forcing_amp = float(self.rng.uniform(0.05, 0.3))

        # ── per-species params ──
        r = self.rng.normal(0.4, 0.1, size=n_species)          # intrinsic growth
        r = np.clip(r, 0.1, 0.9)
        K = self.rng.lognormal(mean=3.5, sigma=0.5, size=n_species)  # carrying capacity
        K = np.clip(K, 5.0, 200.0)

        # cross-species coupling matrix C ∈ [-cs, cs] (symmetric, zero diag)
        # Use a Gaussian off-diagonal to introduce both mutualism and competition.
        C = self.rng.normal(0, coupling_strength, size=(n_species, n_species))
        C = 0.5 * (C + C.T)
        np.fill_diagonal(C, 0.0)

        # ── env forcing: common AR(1) signal driving baseline correlation ──
        env_signal = np.zeros(n_years)
        env_signal[0] = self.rng.normal(0, 1)
        ar_phi = 0.6
        for t in range(1, n_years):
            env_signal[t] = ar_phi * env_signal[t - 1] + self.rng.normal(0, 1)
        # standardise
        env_signal = (env_signal - np.mean(env_signal)) / (np.std(env_signal) + 1e-9)

        # ── simulate logistic dynamics with coupling + env forcing ──
        # x_{t+1,i} = x_{t,i} + r_i * x_{t,i} * (1 - x_{t,i}/K_i)
        #           + Σ_j C[i,j] * x_{t,j} / K_j + env_amp * env_signal[t]
        #           + obs noise
        abundances = np.zeros((n_species, n_years))
        abundances[:, 0] = self.rng.uniform(0.3 * K, 0.7 * K)

        for t in range(1, n_years):
            x_prev = abundances[:, t - 1]
            logistic_term = r * x_prev * (1.0 - x_prev / np.maximum(K, 1e-6))
            coupling_term = (C @ (x_prev / np.maximum(K, 1e-6))) * K
            env_term = env_forcing_amp * env_signal[t] * K
            noise = self.rng.normal(0, 0.05 * K)
            x_next = x_prev + logistic_term + coupling_term + env_term + noise
            abundances[:, t] = np.clip(x_next, 1e-3, 5.0 * K)

        # ── derived metrics ──
        biomass = np.sum(abundances, axis=0)
        rho = compute_abundance_correlation(abundances)
        sigma = compute_biomass_stability(biomass)

        species_ids = [f"sp_{i:03d}" for i in range(n_species)]
        timestamps = np.arange(n_years, dtype=float)

        cid = community_id or f"synth_{realm[:3]}_{self.rng.integers(100000):05d}"

        return EcologicalSample(
            community_id=cid,
            species_abundances=abundances,
            biomass_trajectory=biomass,
            timestamps=timestamps,
            species_ids=species_ids,
            k=n_species,
            rho=float(rho),
            sigma=float(sigma),
            realm=realm,
            taxonomic_group=taxonomic_group,
            metadata={
                "synthetic": True,
                "coupling_strength": coupling_strength,
                "env_forcing_amp": env_forcing_amp,
                "intrinsic_growth_mean": float(np.mean(r)),
                "carrying_capacity_mean": float(np.mean(K)),
            },
        )

    def generate_dataset(
        self,
        n_communities: int = 100,
        realms: Optional[List[str]] = None,
    ) -> BioTIMECommunityDataset:
        """Generate a multi-community synthetic BioTIME dataset.

        Args
        ----
        n_communities : how many communities to synthesize (default 100)
        realms        : list of realm tags to cycle through (default
                        ["terrestrial", "marine", "freshwater"])

        Returns
        -------
        BioTIMECommunityDataset with `n_communities` synthetic samples.
        """
        if realms is None:
            realms = ["terrestrial", "marine", "freshwater"]

        dataset = BioTIMECommunityDataset(source="synthetic")
        for i in range(n_communities):
            realm = realms[i % len(realms)]
            sample = self.generate_community(
                community_id=f"synth_{i:04d}_{realm[:3]}",
                realm=realm,
            )
            dataset.communities[sample.community_id] = sample

        return dataset


# ─────────────────────────────────────────────────────────────────────────
# Real-data loader (BioTIME CSV → BioTIMECommunityDataset)
# ─────────────────────────────────────────────────────────────────────────


def _load_biotime_csv(
    csv_path: Path,
    min_years: int = 10,
    min_species: int = 5,
) -> BioTIMECommunityDataset:
    """Load BioTIME 2.0 CSV download into a BioTIMECommunityDataset.

    Expected CSV schema (the BioTIME public download format):
        STUDY_ID, PLOT (optional), YEAR, GENUS_SPECIES, sum.allrawdata.ABUNDANCE
        (or ABUNDANCE / BIOMASS variants)

    Communities are grouped by (STUDY_ID, PLOT) tuple; for each community
    that passes the filter we pivot to a (species × year) matrix and
    compute k, rho, sigma.

    NOTE: this is a best-effort schema — the BioTIME team has shipped
    multiple CSV layouts. If the column names don't match, the loader
    raises with a clear message so the caller can fall back to synthetic.
    """
    df = pd.read_csv(csv_path, low_memory=False)

    # Normalise common column variants.
    cols = {c.upper(): c for c in df.columns}

    def col(name: str, alts: tuple = ()) -> Optional[str]:
        for n in (name, *alts):
            if n.upper() in cols:
                return cols[n.upper()]
        return None

    study_col = col("STUDY_ID", ("STUDY",))
    plot_col = col("PLOT", ("SITE", "SAMPLE_DESC"))
    year_col = col("YEAR", ("DATE_YEAR",))
    sp_col = col("GENUS_SPECIES", ("SPECIES", "TAXA", "TAXON"))
    abund_col = col("SUM.ALLRAWDATA.ABUNDANCE", ("ABUNDANCE", "BIOMASS", "VALUE"))

    if not all([study_col, year_col, sp_col, abund_col]):
        raise ValueError(
            "BioTIME CSV missing one of required columns "
            f"(STUDY_ID, YEAR, GENUS_SPECIES, ABUNDANCE). Found: {list(df.columns)}"
        )

    if plot_col is None:
        df["_PLOT"] = "default"
        plot_col = "_PLOT"

    df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
    df = df.dropna(subset=[year_col])
    df[abund_col] = pd.to_numeric(df[abund_col], errors="coerce").fillna(0)

    dataset = BioTIMECommunityDataset(source="biotime_csv")
    groups = df.groupby([study_col, plot_col])

    for (study_id, plot_id), group in groups:
        # Pivot to (species × year)
        pivot = (
            group.pivot_table(
                index=sp_col,
                columns=year_col,
                values=abund_col,
                aggfunc="sum",
                fill_value=0,
            )
            .sort_index(axis=1)
        )

        n_species, n_years = pivot.shape
        if n_species < min_species or n_years < min_years:
            continue

        abundances = pivot.values.astype(float)
        species_ids = list(pivot.index.astype(str))
        years = pivot.columns.values.astype(float)
        biomass = np.sum(abundances, axis=0)

        if np.sum(biomass) <= 0:
            continue

        rho = compute_abundance_correlation(abundances)
        sigma = compute_biomass_stability(biomass)

        cid = f"STUDY_{study_id}_{plot_id}"
        sample = EcologicalSample(
            community_id=cid,
            species_abundances=abundances,
            biomass_trajectory=biomass,
            timestamps=years,
            species_ids=species_ids,
            k=int(n_species),
            rho=float(rho),
            sigma=float(sigma),
            metadata={"source": "BioTIME 2.0", "study_id": str(study_id), "plot": str(plot_id)},
        )
        dataset.communities[cid] = sample

    return dataset


def load_biotime_data(
    data_dir: Optional[Union[str, Path]] = None,
    csv_filename: str = "biotime_query.csv",
    fallback_to_synthetic: bool = True,
    n_synthetic_communities: int = 100,
    min_years: int = 10,
    min_species: int = 5,
    seed: Optional[int] = None,
) -> BioTIMECommunityDataset:
    """Entry point: load BioTIME communities, falling back to synthetic.

    Search order:
      1. `data_dir / csv_filename` if it exists → real BioTIME CSV.
      2. If fallback_to_synthetic, SyntheticBioTIMEGenerator with `seed`.
      3. Otherwise raise FileNotFoundError.

    Args
    ----
    data_dir              : where to look for the vendored CSV. Defaults to
                            `data/ecological/` under the repo root.
    csv_filename          : CSV name within data_dir (default biotime_query.csv).
    fallback_to_synthetic : if True, generate synthetic data when CSV absent.
    n_synthetic_communities : how many synthetic communities to emit.
    min_years             : community filter; min years of time series.
    min_species           : community filter; min species per community.
    seed                  : RNG seed for synthetic generator.

    Returns
    -------
    BioTIMECommunityDataset, either real or synthetic.
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    data_dir = Path(data_dir)
    csv_path = data_dir / csv_filename

    if csv_path.exists():
        try:
            ds = _load_biotime_csv(csv_path, min_years=min_years, min_species=min_species)
            if ds.n_communities > 0:
                return ds
        except Exception as e:
            if not fallback_to_synthetic:
                raise
            print(f"[load_biotime_data] BioTIME CSV load failed: {e}; falling back to synthetic")

    if not fallback_to_synthetic:
        raise FileNotFoundError(
            f"BioTIME CSV not found at {csv_path} and fallback_to_synthetic=False."
        )

    gen = SyntheticBioTIMEGenerator(seed=seed)
    return gen.generate_dataset(n_communities=n_synthetic_communities)


# Backwards-compatible alias used by REGIME.md spec.
def load_biotime_communities(
    data_dir: Optional[Union[str, Path]] = None,
    **kwargs,
) -> BioTIMECommunityDataset:
    """Alias matching `data_sources.yaml` loader-name convention."""
    return load_biotime_data(data_dir=data_dir, **kwargs)


# ─────────────────────────────────────────────────────────────────────────
# Convenience: prepare a single community for engine-vs-data comparison
# ─────────────────────────────────────────────────────────────────────────


def prepare_for_engine(
    dataset: BioTIMECommunityDataset,
    community_id: Optional[str] = None,
) -> Dict:
    """Extract one community's per-year sigma trajectory for engine fitting.

    Args
    ----
    dataset      : BioTIMECommunityDataset
    community_id : specific community to extract; if None, picks the first

    Returns
    -------
    dict with:
        community_id, k, rho, sigma_final, num_years,
        empirical_biomass, empirical_sigma_trajectory,
        species_abundances, timestamps
    """
    if not dataset.communities:
        raise ValueError("Dataset is empty.")

    if community_id is None:
        community_id = next(iter(dataset.communities))

    if community_id not in dataset.communities:
        raise KeyError(f"Community {community_id!r} not in dataset.")

    c = dataset.communities[community_id]

    # Build a per-year sigma trajectory: rolling-window inverse-CV of biomass.
    # Use a min 3-yr window to avoid degeneracy.
    biomass = c.biomass_trajectory
    n = len(biomass)
    sigma_traj = np.zeros(n)
    window = max(3, n // 5)
    for t in range(n):
        lo = max(0, t - window + 1)
        hi = t + 1
        sigma_traj[t] = compute_biomass_stability(biomass[lo:hi]) if (hi - lo) >= 2 else 1.0

    return {
        "community_id": community_id,
        "k": c.k,
        "rho": c.rho,
        "sigma_final": c.sigma,
        "num_years": c.num_years,
        "empirical_biomass": biomass.copy(),
        "empirical_sigma_trajectory": sigma_traj,
        "species_abundances": c.species_abundances.copy(),
        "timestamps": c.timestamps.copy(),
        "species_ids": list(c.species_ids),
        "realm": c.realm,
        "taxonomic_group": c.taxonomic_group,
        "metadata": dict(c.metadata),
    }


__all__ = [
    "EcologicalSample",
    "BioTIMECommunityDataset",
    "SyntheticBioTIMEGenerator",
    "compute_abundance_correlation",
    "compute_biomass_stability",
    "load_biotime_data",
    "load_biotime_communities",
    "prepare_for_engine",
]
