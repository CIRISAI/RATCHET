"""
RATCHET AlphaFold (CATH-S40 single-domain) Substrate Loader

Loads (or synthesizes) per-residue pLDDT trajectories for single-domain
proteins from the AlphaFold Protein Structure Database v6, for use with
the ProteinFoldingEngine. Mirrors the
ratchet.data.{battery,institutional,microbiome,ecological}_loader pattern.

Domain mapping (per REGIME.md §"A0 — AlphaFold residues"):
    k     : Sequence length (residue count) of a single-domain protein
    rho   : Mean pairwise correlation of per-residue B-factor predictions
            (computed from pLDDT covariance across residues)
    sigma : Mean pLDDT score (structural stability proxy), bounded to (0, 1]
    f     : 1 - sigma (compromise / instability fraction)

Data sources
------------
Primary  : AlphaFold DB v6 (UniProt 2025_03 sync; 241M structures)
            - https://ftp.ebi.ac.uk/pub/databases/alphafold/v4/
            - CATH-S40 representative single-domain subset (~10,000 proteins)
            - CC-BY-4.0
Fallback : SyntheticAlphaFoldGenerator below, parameterised on the published
            AlphaFold pLDDT marginal distributions (Jumper et al. 2021).
            Synthesised proteins use truncated-normal pLDDT trajectories
            with exponential-decay residue-to-residue spatial correlation
            and reproduce the published k / rho / sigma marginal
            distributions adequately for engine-vs-data harness wiring.

The real-vendor entry point `load_cath_s40_alphafold_data` looks for a
parquet/CSV at `data/protein/cath_s40_alphafold.parquet`; if absent, it
falls back to the synthetic generator. The synthetic-generated dataset
is sufficient to exercise the v1.0 P1 harness — real-data validation
slots in once the AlphaFold CATH-S40 parquet is vendored and its SHA
pinned in `experiments/exp2_cross_substrate/data_sources.yaml`.

References
----------
- Jumper, J., et al. (2021). Highly accurate protein structure prediction
  with AlphaFold. Nature, 596, 583-589.
- Varadi, M., et al. (2024). AlphaFold Protein Structure Database in 2024:
  providing structure coverage for over 214 million protein sequences.
  Nucleic Acids Research, 52, D368-D375.
- Sillitoe, I., et al. (2021). CATH: increased structural coverage of
  functional space. Nucleic Acids Research, 49, D266-D273.
- Mariani, V., Biasini, M., Barbato, A., & Schwede, T. (2013). lDDT: a
  local superposition-free score for comparing protein structures.
  Bioinformatics, 29(21), 2722-2728.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


# Default vendored-data location (matches data_sources.yaml registry).
DEFAULT_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "protein"


# CATH classes (top-level): 1 Mainly Alpha, 2 Mainly Beta, 3 Alpha-Beta,
# 4 Few Secondary Structures. We tag synthetic proteins with one of these
# so downstream sub-class analysis works the same way as on real CATH-S40.
CATH_CLASSES = ("1", "2", "3", "4")


# ─────────────────────────────────────────────────────────────────────────
# Per-protein sample
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class ProteinSample:
    """A single AlphaFold protein with computed RATCHET vars.

    The dataclass is a *snapshot* of one single-domain protein:
    plddt_trajectory is shape (k,) per-residue confidence scores in [0,100];
    we keep them on the 0-100 scale internally (matching AlphaFold DB)
    and convert to sigma ∈ (0, 1] via division by 100.

    Attributes
    ----------
    uniprot_id : UniProt accession of the source protein (e.g. "P12345")
    sequence_length : k, residue count
    plddt_trajectory : (k,) array of per-residue pLDDT scores in [0, 100]
    mean_plddt : float in [0, 100], the σ proxy (also stored as sigma in [0,1])
    b_factor_correlation : ρ, mean pairwise correlation of per-residue
                            B-factor predictions (computed from pLDDT
                            covariance under a windowed-sliding scheme)
    cath_class : CATH top-level class label "1"-"4" (optional)
    k : alias for sequence_length, RATCHET-uniform accessor
    rho : alias for b_factor_correlation, RATCHET-uniform accessor
    sigma : mean_plddt / 100, RATCHET-uniform accessor in (0, 1]
    metadata : freeform additional fields (proteome, source, synthetic flag)
    """

    uniprot_id: str
    sequence_length: int
    plddt_trajectory: np.ndarray
    mean_plddt: float
    b_factor_correlation: float
    cath_class: Optional[str] = None
    k: int = 0
    rho: float = 0.0
    sigma: float = 0.0
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Keep k / rho / sigma synchronised with the canonical fields.
        if self.k == 0:
            self.k = int(self.sequence_length)
        if self.rho == 0.0:
            self.rho = float(self.b_factor_correlation)
        if self.sigma == 0.0:
            self.sigma = float(self.mean_plddt) / 100.0

    # ── RATCHET-uniform accessors (mirror EcologicalSample / BatteryData) ──

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
    def num_residues(self) -> int:
        return int(self.sequence_length)


# ─────────────────────────────────────────────────────────────────────────
# Multi-protein dataset aggregator (parallels BioTIMECommunityDataset)
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class CATHS40ProteinDataset:
    """Aggregator over many ProteinSamples (parallels BioTIMECommunityDataset)."""

    proteins: Dict[str, ProteinSample] = field(default_factory=dict)
    source: str = "unknown"  # "alphafold_parquet" / "alphafold_csv" / "synthetic"

    # ── identity ──
    @property
    def n_proteins(self) -> int:
        return len(self.proteins)

    @property
    def uniprot_ids(self) -> List[str]:
        return list(self.proteins.keys())

    # ── per-protein aggregates ──
    def mean_k(self) -> float:
        if not self.proteins:
            return 0.0
        return float(np.mean([p.k for p in self.proteins.values()]))

    def mean_rho(self) -> float:
        if not self.proteins:
            return 0.0
        return float(np.mean([p.rho for p in self.proteins.values()]))

    def mean_sigma(self) -> float:
        if not self.proteins:
            return 0.0
        return float(np.mean([p.sigma for p in self.proteins.values()]))

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
        """Per-protein summary dataframe."""
        rows = []
        for pid, p in self.proteins.items():
            rows.append({
                "uniprot_id": pid,
                "k": p.k,
                "rho": p.rho,
                "sigma": p.sigma,
                "mean_plddt": p.mean_plddt,
                "f": p.get_f(),
                "k_eff": p.get_k_eff(),
                "cath_class": p.cath_class,
            })
        return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────
# Compute helpers (used by both real-data and synthetic paths)
# ─────────────────────────────────────────────────────────────────────────


def compute_residue_correlation(
    plddt_trajectory: np.ndarray,
    window: int = 10,
) -> float:
    """Mean absolute pairwise correlation across residue-window pairs.

    AlphaFold pLDDT is a per-residue confidence score (not a time series),
    so to get a meaningful "pairwise correlation across residues" we slide
    a window of size `window` over the sequence and compute correlations
    between overlapping windows. The mean absolute Pearson across all
    distinct window pairs gives a scalar ρ.

    This operationalisation captures spatial coherence: a protein whose
    pLDDT varies smoothly residue-to-residue (high coherence) has high ρ;
    a protein with disordered/independent residue confidences has low ρ.

    Args
    ----
    plddt_trajectory : (k,) array of per-residue pLDDT scores in [0, 100]
    window : int, sliding-window size for windowed-correlation computation

    Returns
    -------
    rho ∈ [0, 1] : mean |Pearson| across distinct window pairs.

    Edge cases (constant windows or k < 2 windows) return 0 rather than NaN.
    """
    p = np.asarray(plddt_trajectory, dtype=float)
    if p.ndim != 1:
        return 0.0
    k = len(p)
    w = max(2, int(window))
    if k < 2 * w:
        # Sequence too short for meaningful windowed correlation;
        # use direct per-residue half/half correlation as a fallback.
        if k < 4:
            return 0.0
        mid = k // 2
        a = p[:mid]
        b = p[mid:mid + len(a)]
        if np.std(a) < 1e-10 or np.std(b) < 1e-10:
            return 0.0
        r = np.corrcoef(a, b)[0, 1]
        return float(abs(r)) if np.isfinite(r) else 0.0

    # Build (n_windows, window) matrix; n_windows = k - w + 1
    n_windows = k - w + 1
    # Step by w//2 to avoid 100% overlap inflating r
    step = max(1, w // 2)
    starts = np.arange(0, n_windows, step)
    windows = np.stack([p[s:s + w] for s in starts])

    # Compute pairwise correlations across distinct windows.
    pairs = []
    n_w = windows.shape[0]
    for i in range(n_w):
        if np.std(windows[i]) < 1e-10:
            continue
        for j in range(i + 1, n_w):
            if np.std(windows[j]) < 1e-10:
                continue
            r = np.corrcoef(windows[i], windows[j])[0, 1]
            if np.isfinite(r):
                pairs.append(abs(float(r)))
    if not pairs:
        return 0.0
    return float(np.mean(pairs))


def compute_plddt_stability(plddt_trajectory: np.ndarray) -> float:
    """Mean pLDDT scaled to (0, 1] — structural stability proxy.

    AlphaFold's published pLDDT marginals have a median around 85 (mass
    concentrated in 70-95). We scale by /100 so that sigma ∈ (0, 1].
    A perfectly-confident protein has sigma = 1.0; a fully-disordered
    one has sigma → 0.

    Args
    ----
    plddt_trajectory : (k,) array in [0, 100]

    Returns
    -------
    sigma ∈ (0, 1] : mean_pLDDT / 100, clipped.
    """
    p = np.asarray(plddt_trajectory, dtype=float)
    if len(p) == 0:
        return 0.0
    mean_plddt = float(np.mean(p))
    return float(max(0.0, min(1.0, mean_plddt / 100.0)))


# ─────────────────────────────────────────────────────────────────────────
# Synthetic AlphaFold generator (drop-in for unavailable real data)
# ─────────────────────────────────────────────────────────────────────────


class SyntheticAlphaFoldGenerator:
    """Generate realistic AlphaFold-like per-residue pLDDT trajectories.

    Each protein is simulated as a (k,)-residue pLDDT vector drawn from a
    truncated-normal distribution with realistic AlphaFold marginal
    statistics. Spatial correlation between residues (long-range
    structural coupling) is induced via an exponential-decay covariance
    kernel: residues i and j have covariance ~ exp(-|i-j|/L) where L is a
    correlation length.

    Length-band parameters follow AlphaFold v4 distributions:

        k (length)        : LogNormal(mu=5.2, sigma=0.6), clipped to [40, 800]
        pLDDT mean        : Normal(85, 5) clipped to [60, 95]
        pLDDT std (residue): Normal(8, 2) clipped to [3, 20]
        correlation length L: depends on CATH class (mainly-α: longer L;
                              mainly-β: shorter L; α/β: intermediate)

    The covariance kernel is K[i,j] = sigma_r² · exp(-|i-j|/L), where
    sigma_r is the per-residue pLDDT std. Drawing from N(mu·1, K) gives a
    Gaussian random vector with the desired marginal mean/std and
    exponential-decay spatial correlation. Values are then clipped/squashed
    to [0, 100] to respect pLDDT's bounded range.

    Refs:
        Jumper et al. 2021 (AlphaFold pLDDT marginals);
        Varadi et al. 2024 (AlphaFold DB v4/v6 statistics);
        Mariani et al. 2013 (lDDT score definition).
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def generate_protein(
        self,
        uniprot_id: Optional[str] = None,
        sequence_length: Optional[int] = None,
        mean_plddt: Optional[float] = None,
        residue_plddt_std: Optional[float] = None,
        correlation_length: Optional[float] = None,
        cath_class: Optional[str] = None,
    ) -> ProteinSample:
        """Generate a single synthetic AlphaFold-style protein.

        Args
        ----
        uniprot_id          : optional string identifier
        sequence_length     : if None, drawn from LogNormal(5.2, 0.6) ∩ [40, 800]
        mean_plddt          : if None, drawn from Normal(85, 5) ∩ [60, 95]
        residue_plddt_std   : if None, drawn from Normal(8, 2) ∩ [3, 20]
        correlation_length  : if None, drawn per CATH class (8-30 residues)
        cath_class          : one of "1"/"2"/"3"/"4"; random if None

        Returns
        -------
        ProteinSample with k, rho, sigma populated from simulated pLDDT.
        """
        # ── sample protein-level parameters ──
        if sequence_length is None:
            k_raw = self.rng.lognormal(mean=5.2, sigma=0.6)
            sequence_length = int(np.clip(round(k_raw), 40, 800))
        else:
            sequence_length = int(max(40, min(800, sequence_length)))

        if mean_plddt is None:
            mean_plddt = float(np.clip(self.rng.normal(85.0, 5.0), 60.0, 95.0))
        else:
            mean_plddt = float(np.clip(mean_plddt, 0.0, 100.0))

        if residue_plddt_std is None:
            residue_plddt_std = float(np.clip(self.rng.normal(8.0, 2.0), 3.0, 20.0))
        else:
            residue_plddt_std = float(max(0.5, residue_plddt_std))

        if cath_class is None:
            cath_class = str(self.rng.choice(CATH_CLASSES))

        if correlation_length is None:
            # Mainly-alpha (1): longer helices → longer L
            # Mainly-beta (2): shorter strands → shorter L
            # Alpha-beta (3): intermediate
            # Few SS (4): short, random
            base = {"1": 22.0, "2": 12.0, "3": 16.0, "4": 8.0}.get(cath_class, 14.0)
            correlation_length = float(np.clip(self.rng.normal(base, 4.0), 4.0, 40.0))

        # ── build exponential-decay covariance kernel ──
        # K[i,j] = σ² · exp(-|i - j| / L)
        k = sequence_length
        idx = np.arange(k)
        dist = np.abs(idx[:, None] - idx[None, :])
        cov = (residue_plddt_std ** 2) * np.exp(-dist / float(max(correlation_length, 1e-3)))

        # Add a small jitter to ensure positive-definiteness.
        cov += np.eye(k) * 1e-4

        # ── draw a Gaussian sample with mean mean_plddt and covariance cov ──
        # For large k Cholesky can be expensive; use eigendecomposition with
        # truncation to handle numerical wobble.
        try:
            L_chol = np.linalg.cholesky(cov)
            z = self.rng.standard_normal(k)
            plddt = mean_plddt + L_chol @ z
        except np.linalg.LinAlgError:
            # Fallback: eigendecomposition with non-negative eigenvalues
            w, V = np.linalg.eigh(cov)
            w = np.clip(w, 1e-6, None)
            z = self.rng.standard_normal(k)
            plddt = mean_plddt + V @ (np.sqrt(w) * z)

        # Soft-bound to AlphaFold's [0, 100] range with a smooth squash near the edges
        plddt = np.clip(plddt, 0.0, 100.0)

        # ── derived metrics ──
        obs_mean_plddt = float(np.mean(plddt))
        rho = compute_residue_correlation(plddt)
        sigma = compute_plddt_stability(plddt)

        pid = uniprot_id or f"SYNTH_{self.rng.integers(10_000_000):07d}"

        return ProteinSample(
            uniprot_id=pid,
            sequence_length=k,
            plddt_trajectory=plddt,
            mean_plddt=obs_mean_plddt,
            b_factor_correlation=float(rho),
            cath_class=cath_class,
            k=k,
            rho=float(rho),
            sigma=float(sigma),
            metadata={
                "synthetic": True,
                "target_mean_plddt": mean_plddt,
                "residue_plddt_std": residue_plddt_std,
                "correlation_length": correlation_length,
            },
        )

    def generate_dataset(
        self,
        n_proteins: int = 100,
        cath_classes: Optional[List[str]] = None,
    ) -> CATHS40ProteinDataset:
        """Generate a multi-protein synthetic AlphaFold dataset.

        Args
        ----
        n_proteins  : how many proteins to synthesize (default 100)
        cath_classes : list of CATH classes to cycle through (default all four)

        Returns
        -------
        CATHS40ProteinDataset with `n_proteins` synthetic samples.
        """
        if cath_classes is None:
            cath_classes = list(CATH_CLASSES)

        dataset = CATHS40ProteinDataset(source="synthetic")
        for i in range(n_proteins):
            cls = cath_classes[i % len(cath_classes)]
            sample = self.generate_protein(
                uniprot_id=f"SYNTH_{i:05d}_C{cls}",
                cath_class=cls,
            )
            dataset.proteins[sample.uniprot_id] = sample

        return dataset


# ─────────────────────────────────────────────────────────────────────────
# Real-data loader (AlphaFold parquet/CSV → CATHS40ProteinDataset)
# ─────────────────────────────────────────────────────────────────────────


def _load_alphafold_parquet_or_csv(
    path: Path,
    min_length: int = 40,
    max_length: int = 800,
) -> CATHS40ProteinDataset:
    """Load a vendored AlphaFold CATH-S40 parquet/CSV into the dataset.

    Expected schema (long format, one row per residue):
        uniprot_id, residue_index, plddt, cath_class (optional)

    Or short format (one row per protein):
        uniprot_id, sequence_length, plddt_trajectory (json-encoded array),
        mean_plddt, b_factor_correlation, cath_class (optional)

    NOTE: this is a best-effort schema — different vendoring scripts ship
    different layouts. If the column names don't match either shape, the
    loader raises with a clear message so the caller can fall back to
    synthetic.
    """
    p = Path(path)
    if p.suffix.lower() == ".parquet":
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p, low_memory=False)

    cols = {c.lower(): c for c in df.columns}

    def col(name: str, alts: tuple = ()) -> Optional[str]:
        for n in (name, *alts):
            if n.lower() in cols:
                return cols[n.lower()]
        return None

    upid = col("uniprot_id", ("accession", "id"))
    if upid is None:
        raise ValueError(
            "AlphaFold table missing uniprot_id-like column. "
            f"Found: {list(df.columns)}"
        )

    # Detect short (one-row-per-protein) vs long (one-row-per-residue)
    plddt_arr_col = col("plddt_trajectory", ("plddt_array", "plddt_values"))
    res_col = col("residue_index", ("residue", "position"))
    plddt_col = col("plddt", ("confidence",))
    sl_col = col("sequence_length", ("length", "k"))
    cath_col = col("cath_class", ("cath",))

    dataset = CATHS40ProteinDataset(source="alphafold_csv" if p.suffix != ".parquet"
                                    else "alphafold_parquet")

    if plddt_arr_col is not None:
        # Short format
        for _, row in df.iterrows():
            try:
                arr_raw = row[plddt_arr_col]
                if isinstance(arr_raw, str):
                    # JSON-list or comma-separated
                    arr_raw = arr_raw.strip()
                    if arr_raw.startswith("["):
                        import json
                        plddt = np.asarray(json.loads(arr_raw), dtype=float)
                    else:
                        plddt = np.asarray(
                            [float(x) for x in arr_raw.split(",") if x.strip()],
                            dtype=float,
                        )
                else:
                    plddt = np.asarray(arr_raw, dtype=float)
            except Exception:
                continue
            if plddt.ndim != 1 or len(plddt) < min_length or len(plddt) > max_length:
                continue
            pid = str(row[upid])
            cath = str(row[cath_col]) if cath_col and pd.notna(row.get(cath_col)) else None
            mean_plddt = float(np.mean(plddt))
            rho = compute_residue_correlation(plddt)
            sigma = compute_plddt_stability(plddt)
            sample = ProteinSample(
                uniprot_id=pid,
                sequence_length=int(len(plddt)),
                plddt_trajectory=plddt,
                mean_plddt=mean_plddt,
                b_factor_correlation=float(rho),
                cath_class=cath,
                k=int(len(plddt)),
                rho=float(rho),
                sigma=float(sigma),
                metadata={"source": "AlphaFold DB v6 (vendored)"},
            )
            dataset.proteins[pid] = sample
    elif res_col is not None and plddt_col is not None:
        # Long format: groupby uniprot_id, build trajectory by sorted residue_index
        for pid_val, group in df.groupby(upid):
            grp = group.sort_values(res_col)
            plddt = grp[plddt_col].astype(float).values
            if plddt.ndim != 1 or len(plddt) < min_length or len(plddt) > max_length:
                continue
            pid = str(pid_val)
            cath = (
                str(grp.iloc[0][cath_col])
                if cath_col and pd.notna(grp.iloc[0].get(cath_col))
                else None
            )
            mean_plddt = float(np.mean(plddt))
            rho = compute_residue_correlation(plddt)
            sigma = compute_plddt_stability(plddt)
            sample = ProteinSample(
                uniprot_id=pid,
                sequence_length=int(len(plddt)),
                plddt_trajectory=plddt,
                mean_plddt=mean_plddt,
                b_factor_correlation=float(rho),
                cath_class=cath,
                k=int(len(plddt)),
                rho=float(rho),
                sigma=float(sigma),
                metadata={"source": "AlphaFold DB v6 (vendored)"},
            )
            dataset.proteins[pid] = sample
    else:
        raise ValueError(
            "AlphaFold table missing plddt columns. Expected one of: "
            "(plddt_trajectory) for short format, or "
            "(residue_index, plddt) for long format. "
            f"Found: {list(df.columns)}"
        )

    return dataset


def load_cath_s40_alphafold_data(
    data_dir: Optional[Union[str, Path]] = None,
    parquet_filename: str = "cath_s40_alphafold.parquet",
    csv_filename: str = "cath_s40_alphafold.csv",
    sample_parquet_filename: str = "cath_s40_alphafold_sample.parquet",
    sample_csv_filename: str = "cath_s40_alphafold_sample.csv",
    fallback_to_synthetic: bool = True,
    n_synthetic_proteins: int = 100,
    min_length: int = 40,
    max_length: int = 800,
    seed: Optional[int] = None,
) -> CATHS40ProteinDataset:
    """Entry point: load CATH-S40 AlphaFold proteins, falling back to synthetic.

    Search order:
      1. `data_dir / parquet_filename` if it exists → full AlphaFold parquet.
      2. `data_dir / csv_filename` if it exists → full AlphaFold CSV.
      3. `data_dir / sample_parquet_filename` if it exists → sample subset.
      4. `data_dir / sample_csv_filename` if it exists → sample CSV subset.
      5. If fallback_to_synthetic, SyntheticAlphaFoldGenerator with `seed`.
      6. Otherwise raise FileNotFoundError.

    Args
    ----
    data_dir                 : where to look for vendored data. Defaults to
                                `data/protein/` under the repo root.
    parquet_filename         : full parquet name within data_dir (preferred).
    csv_filename             : full CSV name within data_dir (fallback).
    sample_parquet_filename  : small-sample parquet (real-data smoke test).
    sample_csv_filename      : small-sample CSV (real-data smoke test).
    fallback_to_synthetic    : if True, generate synthetic data when absent.
    n_synthetic_proteins     : how many synthetic proteins to emit.
    min_length               : protein filter; min residue count.
    max_length               : protein filter; max residue count.
    seed                     : RNG seed for synthetic generator.

    Returns
    -------
    CATHS40ProteinDataset, either real or synthetic.
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    data_dir = Path(data_dir)
    parquet_path = data_dir / parquet_filename
    csv_path = data_dir / csv_filename
    sample_parquet_path = data_dir / sample_parquet_filename
    sample_csv_path = data_dir / sample_csv_filename

    for path in (parquet_path, csv_path, sample_parquet_path, sample_csv_path):
        if path.exists():
            try:
                ds = _load_alphafold_parquet_or_csv(
                    path, min_length=min_length, max_length=max_length,
                )
                if ds.n_proteins > 0:
                    return ds
            except Exception as e:
                if not fallback_to_synthetic:
                    raise
                print(
                    f"[load_cath_s40_alphafold_data] AlphaFold load failed "
                    f"({path.name}): {e}; falling back to synthetic"
                )
                break

    if not fallback_to_synthetic:
        raise FileNotFoundError(
            f"AlphaFold CATH-S40 data not found at {parquet_path} or {csv_path} "
            f"(or sample variants) and fallback_to_synthetic=False."
        )

    gen = SyntheticAlphaFoldGenerator(seed=seed)
    return gen.generate_dataset(n_proteins=n_synthetic_proteins)


# Backwards-compatible alias used by REGIME.md spec.
def load_alphafold_cath_s40(
    data_dir: Optional[Union[str, Path]] = None,
    **kwargs,
) -> CATHS40ProteinDataset:
    """Alias matching `data_sources.yaml` loader-name convention."""
    return load_cath_s40_alphafold_data(data_dir=data_dir, **kwargs)


# ─────────────────────────────────────────────────────────────────────────
# Convenience: prepare a single protein for engine-vs-data comparison
# ─────────────────────────────────────────────────────────────────────────


def prepare_for_engine(
    dataset: CATHS40ProteinDataset,
    uniprot_id: Optional[str] = None,
) -> Dict:
    """Extract one protein's per-residue pLDDT trajectory for engine fitting.

    Args
    ----
    dataset    : CATHS40ProteinDataset
    uniprot_id : specific protein to extract; if None, picks the first

    Returns
    -------
    dict with:
        uniprot_id, k, rho, sigma, mean_plddt, num_residues,
        empirical_plddt, plddt_trajectory, cath_class, metadata
    """
    if not dataset.proteins:
        raise ValueError("Dataset is empty.")

    if uniprot_id is None:
        uniprot_id = next(iter(dataset.proteins))

    if uniprot_id not in dataset.proteins:
        raise KeyError(f"Protein {uniprot_id!r} not in dataset.")

    p = dataset.proteins[uniprot_id]

    return {
        "uniprot_id": uniprot_id,
        "k": p.k,
        "rho": p.rho,
        "sigma": p.sigma,
        "mean_plddt": p.mean_plddt,
        "num_residues": p.num_residues,
        "empirical_plddt": p.plddt_trajectory.copy(),
        "plddt_trajectory": p.plddt_trajectory.copy(),
        "cath_class": p.cath_class,
        "metadata": dict(p.metadata),
    }


__all__ = [
    "ProteinSample",
    "CATHS40ProteinDataset",
    "SyntheticAlphaFoldGenerator",
    "compute_residue_correlation",
    "compute_plddt_stability",
    "load_cath_s40_alphafold_data",
    "load_alphafold_cath_s40",
    "prepare_for_engine",
    "CATH_CLASSES",
]
