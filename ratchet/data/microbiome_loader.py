"""
RATCHET Microbiome Data Loader

Loads and processes microbiome abundance data for use with MicrobiomeEngine.
Supports American Gut Project (AGP), Human Microbiome Project (HMP), and
synthetic data generation based on published literature distributions.

RATCHET Variable Mapping:
    k: Number of detected species (abundance > threshold)
    rho: Mean pairwise correlation (SparCC estimate or simplified)
    sigma: Normalized Shannon diversity (0-1)
    f: Pathogen/dysbiosis fraction

Data Sources:
    - American Gut Project: ftp://ftp.microbio.me/AmericanGut/latest
    - Human Microbiome Project: https://hmpdacc.org/
    - curatedMetagenomicData: Bioconductor package

References:
    - McDonald et al. (2018). American Gut: an Open Platform for
      Citizen Science Microbiome Research. mSystems 3(3).
    - Lloyd-Price et al. (2017). Strains, functions and dynamics in the
      expanded Human Microbiome Project. Nature 550, 61-66.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


# Default data directory
DEFAULT_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "microbiome"


@dataclass
class MicrobiomeSample:
    """
    A single microbiome sample with computed RATCHET variables.

    Attributes:
        sample_id: Unique identifier for the sample
        abundances: Normalized abundance vector (sums to 1)
        taxa_ids: List of taxon identifiers
        k: Number of detected taxa (abundance > detection_threshold)
        rho: Mean pairwise correlation estimate
        sigma: Normalized Shannon diversity [0, 1]
        f: Pathogen/dysbiosis fraction
        metadata: Additional sample metadata (age, sex, antibiotic history, etc.)
    """
    sample_id: str
    abundances: np.ndarray
    taxa_ids: List[str]
    k: int
    rho: float
    sigma: float
    f: float
    metadata: Dict = field(default_factory=dict)

    @property
    def k_eff(self) -> float:
        """Effective constraint count: k / (1 + rho*(k-1))."""
        if self.k <= 1:
            return float(self.k)
        denom = 1 + self.rho * (self.k - 1)
        return self.k / max(denom, 0.01)


class MicrobiomeDataLoader:
    """
    Loads and processes microbiome abundance data.

    Supports TSV format (from BIOM export) and computes RATCHET framework
    variables for each sample.

    Example:
        >>> loader = MicrobiomeDataLoader()
        >>> loader.load_otu_table("otu_table_L6.txt")
        >>> loader.load_metadata("ag-cleaned.txt")
        >>> samples = loader.get_samples(n=100)
        >>> print(f"Mean diversity: {np.mean([s.sigma for s in samples]):.3f}")
    """

    # Common pathogen/dysbiosis taxa patterns (genus level)
    PATHOGEN_PATTERNS = [
        'Clostridioides',  # C. difficile
        'Clostridium',     # Various pathogens
        'Enterococcus',    # Opportunistic
        'Klebsiella',      # K. pneumoniae
        'Escherichia',     # E. coli (pathogenic strains)
        'Campylobacter',
        'Salmonella',
        'Shigella',
        'Vibrio',
        'Yersinia',
        'Helicobacter',
        'Listeria',
        'Staphylococcus',
        'Streptococcus',   # S. pyogenes, S. pneumoniae
        'Bacteroides',     # B. fragilis (opportunistic)
        'Fusobacterium',
        'Prevotella',      # Associated with inflammation
    ]

    def __init__(
        self,
        data_dir: Optional[Union[str, Path]] = None,
        detection_threshold: float = 1e-6,
        pathogen_patterns: Optional[List[str]] = None,
    ):
        """
        Initialize the data loader.

        Args:
            data_dir: Directory containing microbiome data files.
            detection_threshold: Minimum abundance to consider a taxon present.
            pathogen_patterns: Taxa patterns to consider as potential pathogens.
        """
        self.data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
        self.detection_threshold = detection_threshold
        self.pathogen_patterns = pathogen_patterns or self.PATHOGEN_PATTERNS

        self._otu_table: Optional[pd.DataFrame] = None
        self._metadata: Optional[pd.DataFrame] = None
        self._taxa_ids: List[str] = []
        self._sample_ids: List[str] = []
        self._pathogen_mask: Optional[np.ndarray] = None

    def load_otu_table(
        self,
        filename: str,
        max_samples: Optional[int] = None,
        normalize: bool = True,
    ) -> None:
        """
        Load an OTU/ASV abundance table from TSV format.

        Args:
            filename: Name of the OTU table file (or full path).
            max_samples: Maximum number of samples to load (for memory).
            normalize: Whether to normalize abundances to sum to 1.
        """
        filepath = self._resolve_path(filename)

        # Read the table (first column is taxa, remaining are samples)
        df = pd.read_csv(
            filepath,
            sep='\t',
            comment='#',
            index_col=0,
            nrows=None,
        )

        # Limit samples if requested
        if max_samples and len(df.columns) > max_samples:
            df = df.iloc[:, :max_samples]

        # Store taxa IDs
        self._taxa_ids = list(df.index)
        self._sample_ids = list(df.columns)

        # Transpose so samples are rows
        self._otu_table = df.T

        # Normalize if requested
        if normalize:
            row_sums = self._otu_table.sum(axis=1)
            row_sums = row_sums.replace(0, 1)  # Avoid division by zero
            self._otu_table = self._otu_table.div(row_sums, axis=0)

        # Build pathogen mask
        self._build_pathogen_mask()

        print(f"Loaded OTU table: {len(self._sample_ids)} samples, {len(self._taxa_ids)} taxa")

    def load_metadata(self, filename: str) -> None:
        """
        Load sample metadata from TSV file.

        Args:
            filename: Name of the metadata file (or full path).
        """
        filepath = self._resolve_path(filename)

        # Read metadata
        self._metadata = pd.read_csv(
            filepath,
            sep='\t',
            index_col=0,
            low_memory=False,
        )

        print(f"Loaded metadata: {len(self._metadata)} samples, {len(self._metadata.columns)} columns")

    def get_sample(self, sample_id: str) -> Optional[MicrobiomeSample]:
        """
        Get a single sample with computed RATCHET variables.

        Args:
            sample_id: Sample identifier.

        Returns:
            MicrobiomeSample or None if sample not found.
        """
        if self._otu_table is None:
            raise RuntimeError("No OTU table loaded. Call load_otu_table() first.")

        if sample_id not in self._otu_table.index:
            return None

        abundances = self._otu_table.loc[sample_id].values
        return self._create_sample(sample_id, abundances)

    def get_samples(
        self,
        n: Optional[int] = None,
        body_site: Optional[str] = None,
        antibiotic_history: Optional[str] = None,
        random_seed: Optional[int] = None,
    ) -> List[MicrobiomeSample]:
        """
        Get multiple samples, optionally filtered by criteria.

        Args:
            n: Number of samples to return (None = all).
            body_site: Filter by body site (e.g., "UBERON:feces").
            antibiotic_history: Filter by antibiotic history.
            random_seed: Random seed for reproducible sampling.

        Returns:
            List of MicrobiomeSample objects.
        """
        if self._otu_table is None:
            raise RuntimeError("No OTU table loaded. Call load_otu_table() first.")

        sample_ids = list(self._otu_table.index)

        # Apply metadata filters if available
        if self._metadata is not None and (body_site or antibiotic_history):
            filtered_ids = self._filter_samples(sample_ids, body_site, antibiotic_history)
            sample_ids = filtered_ids

        # Random sampling if n specified
        if n is not None and n < len(sample_ids):
            rng = np.random.default_rng(random_seed)
            sample_ids = list(rng.choice(sample_ids, size=n, replace=False))

        samples = []
        for sid in sample_ids:
            abundances = self._otu_table.loc[sid].values
            sample = self._create_sample(sid, abundances)
            samples.append(sample)

        return samples

    def get_abundance_statistics(self) -> Dict:
        """
        Compute summary statistics across all samples.

        Returns:
            Dictionary with mean, std, percentiles for k, rho, sigma, f.
        """
        if self._otu_table is None:
            raise RuntimeError("No OTU table loaded.")

        samples = self.get_samples()

        k_vals = [s.k for s in samples]
        sigma_vals = [s.sigma for s in samples]
        f_vals = [s.f for s in samples]

        return {
            'n_samples': len(samples),
            'n_taxa': len(self._taxa_ids),
            'k': {
                'mean': np.mean(k_vals),
                'std': np.std(k_vals),
                'min': np.min(k_vals),
                'max': np.max(k_vals),
                'p25': np.percentile(k_vals, 25),
                'p50': np.percentile(k_vals, 50),
                'p75': np.percentile(k_vals, 75),
            },
            'sigma': {
                'mean': np.mean(sigma_vals),
                'std': np.std(sigma_vals),
                'min': np.min(sigma_vals),
                'max': np.max(sigma_vals),
                'p25': np.percentile(sigma_vals, 25),
                'p50': np.percentile(sigma_vals, 50),
                'p75': np.percentile(sigma_vals, 75),
            },
            'f': {
                'mean': np.mean(f_vals),
                'std': np.std(f_vals),
                'min': np.min(f_vals),
                'max': np.max(f_vals),
                'p25': np.percentile(f_vals, 25),
                'p50': np.percentile(f_vals, 50),
                'p75': np.percentile(f_vals, 75),
            },
        }

    def _resolve_path(self, filename: str) -> Path:
        """Resolve filename to full path."""
        path = Path(filename)
        if path.is_absolute() and path.exists():
            return path

        # Try data directory
        data_path = self.data_dir / filename
        if data_path.exists():
            return data_path

        raise FileNotFoundError(f"Could not find file: {filename}")

    def _build_pathogen_mask(self) -> None:
        """Build boolean mask for pathogen taxa."""
        if not self._taxa_ids:
            return

        mask = np.zeros(len(self._taxa_ids), dtype=bool)
        for i, taxon in enumerate(self._taxa_ids):
            for pattern in self.pathogen_patterns:
                if pattern.lower() in taxon.lower():
                    mask[i] = True
                    break

        self._pathogen_mask = mask
        n_pathogens = np.sum(mask)
        print(f"Identified {n_pathogens} potential pathogen taxa")

    def _create_sample(self, sample_id: str, abundances: np.ndarray) -> MicrobiomeSample:
        """Create a MicrobiomeSample with computed RATCHET variables."""
        # Ensure abundances are normalized
        total = np.sum(abundances)
        if total > 0:
            abundances = abundances / total

        # Compute k (detected taxa count)
        k = int(np.sum(abundances > self.detection_threshold))

        # Compute sigma (normalized Shannon diversity)
        sigma = self._compute_shannon_diversity(abundances)

        # Compute f (pathogen fraction)
        f = self._compute_pathogen_fraction(abundances)

        # Estimate rho (correlation) - simplified estimate based on diversity
        # Lower diversity suggests higher correlation (competitive exclusion)
        rho = self._estimate_correlation(abundances, k, sigma)

        # Get metadata if available
        metadata = {}
        if self._metadata is not None and sample_id in self._metadata.index:
            row = self._metadata.loc[sample_id]
            metadata = {
                col: row[col] for col in row.index
                if pd.notna(row[col]) and row[col] != 'Unknown'
            }

        return MicrobiomeSample(
            sample_id=sample_id,
            abundances=abundances,
            taxa_ids=self._taxa_ids,
            k=k,
            rho=rho,
            sigma=sigma,
            f=f,
            metadata=metadata,
        )

    def _compute_shannon_diversity(self, abundances: np.ndarray) -> float:
        """Compute normalized Shannon diversity index."""
        nonzero = abundances[abundances > 1e-10]
        if len(nonzero) <= 1:
            return 0.0

        H = -np.sum(nonzero * np.log(nonzero))
        H_max = np.log(len(nonzero))

        return H / H_max if H_max > 0 else 0.0

    def _compute_pathogen_fraction(self, abundances: np.ndarray) -> float:
        """Compute fraction of abundance from pathogen taxa."""
        if self._pathogen_mask is None:
            return 0.0

        return float(np.sum(abundances[self._pathogen_mask]))

    def _estimate_correlation(
        self,
        abundances: np.ndarray,
        k: int,
        sigma: float,
    ) -> float:
        """
        Estimate mean pairwise correlation.

        Uses a heuristic based on diversity and evenness. In reality,
        SparCC or similar methods should be used on time series data.

        For healthy gut microbiomes, typical rho is 0.15-0.35.
        """
        if k <= 1:
            return 0.0

        # Heuristic: higher diversity (sigma) correlates with lower rho
        # Based on competitive exclusion principle
        base_rho = 0.25

        # Adjust based on normalized diversity
        rho = base_rho * (1 - 0.3 * sigma)

        # Adjust based on evenness (Pielou's J)
        nonzero = abundances[abundances > 1e-10]
        if len(nonzero) > 1:
            evenness = self._compute_shannon_diversity(abundances)
            rho = rho * (1.2 - 0.4 * evenness)

        return max(0.0, min(1.0, rho))

    def _filter_samples(
        self,
        sample_ids: List[str],
        body_site: Optional[str],
        antibiotic_history: Optional[str],
    ) -> List[str]:
        """Filter samples by metadata criteria."""
        if self._metadata is None:
            return sample_ids

        filtered = []
        for sid in sample_ids:
            if sid not in self._metadata.index:
                continue

            row = self._metadata.loc[sid]

            # Body site filter
            if body_site:
                site = row.get('BODY_SITE', row.get('body_site', ''))
                if pd.isna(site) or body_site.lower() not in str(site).lower():
                    continue

            # Antibiotic history filter
            if antibiotic_history:
                abx = row.get('ANTIBIOTIC_HISTORY', row.get('antibiotic_history', ''))
                if pd.isna(abx) or antibiotic_history.lower() not in str(abx).lower():
                    continue

            filtered.append(sid)

        return filtered


class SyntheticMicrobiomeGenerator:
    """
    Generates synthetic microbiome profiles based on published literature distributions.

    Uses parameters from:
    - Human Microbiome Project (healthy adult fecal samples)
    - American Gut Project (population-level variation)
    - DIABIMMUNE (infant gut development)

    Reference distributions:
        k (detected species): LogNormal(mu=4.5, sigma=0.5), range 50-500
        sigma (diversity): Beta(alpha=8, beta=2), range 0.6-0.95 for healthy
        f (pathogen fraction): Exponential(scale=0.05), typically < 0.1
        rho (correlation): Normal(mu=0.25, sigma=0.05), range 0.1-0.4
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the generator.

        Args:
            seed: Random seed for reproducibility.
        """
        self.rng = np.random.default_rng(seed)

        # Reference taxa (top 100 most common gut taxa)
        self._build_reference_taxa()

    def generate_healthy_adult(
        self,
        n_taxa: int = 500,
        sample_id: Optional[str] = None,
    ) -> MicrobiomeSample:
        """
        Generate a synthetic healthy adult gut microbiome profile.

        Args:
            n_taxa: Number of taxa in the profile.
            sample_id: Optional sample identifier.

        Returns:
            MicrobiomeSample with realistic healthy adult parameters.
        """
        # Generate k (detected species)
        k_raw = self.rng.lognormal(mean=4.5, sigma=0.4)
        k = int(np.clip(k_raw, 80, n_taxa * 0.8))

        # Generate abundances (log-normal distribution)
        abundances = np.zeros(n_taxa)
        present_idx = self.rng.choice(n_taxa, size=k, replace=False)

        # Log-normal abundances (typical for microbiomes)
        raw_abund = self.rng.lognormal(mean=0, sigma=2.5, size=k)
        abundances[present_idx] = raw_abund / np.sum(raw_abund)

        # Generate RATCHET variables
        sigma = self.rng.beta(a=8, b=2) * 0.35 + 0.6  # Range 0.6-0.95
        f = min(0.15, self.rng.exponential(scale=0.03))  # Typically low
        rho = np.clip(self.rng.normal(0.22, 0.05), 0.1, 0.4)

        # Create taxa IDs
        taxa_ids = [f"genus_{i:03d}" for i in range(n_taxa)]

        return MicrobiomeSample(
            sample_id=sample_id or f"synthetic_healthy_{self.rng.integers(10000)}",
            abundances=abundances,
            taxa_ids=taxa_ids,
            k=k,
            rho=rho,
            sigma=sigma,
            f=f,
            metadata={'synthetic': True, 'profile_type': 'healthy_adult'},
        )

    def generate_dysbiotic(
        self,
        n_taxa: int = 500,
        severity: float = 0.5,
        sample_id: Optional[str] = None,
    ) -> MicrobiomeSample:
        """
        Generate a synthetic dysbiotic gut microbiome profile.

        Args:
            n_taxa: Number of taxa in the profile.
            severity: Dysbiosis severity (0-1).
            sample_id: Optional sample identifier.

        Returns:
            MicrobiomeSample with reduced diversity and elevated pathogen fraction.
        """
        # Reduced k for dysbiosis
        k_raw = self.rng.lognormal(mean=4.0 - severity * 0.5, sigma=0.5)
        k = int(np.clip(k_raw, 20, n_taxa * 0.5))

        # Generate abundances with pathogen dominance
        abundances = np.zeros(n_taxa)
        present_idx = self.rng.choice(n_taxa, size=k, replace=False)

        # More uneven distribution (higher sigma in log-normal)
        raw_abund = self.rng.lognormal(mean=0, sigma=3.0 + severity, size=k)

        # Add pathogen dominance
        dominant_idx = self.rng.choice(len(raw_abund), size=max(1, int(severity * 5)))
        raw_abund[dominant_idx] *= (5 + 10 * severity)

        abundances[present_idx] = raw_abund / np.sum(raw_abund)

        # Lower diversity for dysbiosis
        sigma = max(0.2, 0.7 - severity * 0.4 + self.rng.normal(0, 0.05))

        # Higher pathogen fraction
        f = np.clip(0.05 + severity * 0.25 + self.rng.exponential(0.02), 0, 0.5)

        # Higher correlation (less ecological complexity)
        rho = np.clip(0.3 + severity * 0.15 + self.rng.normal(0, 0.03), 0.2, 0.6)

        taxa_ids = [f"genus_{i:03d}" for i in range(n_taxa)]

        return MicrobiomeSample(
            sample_id=sample_id or f"synthetic_dysbiotic_{self.rng.integers(10000)}",
            abundances=abundances,
            taxa_ids=taxa_ids,
            k=k,
            rho=rho,
            sigma=sigma,
            f=f,
            metadata={
                'synthetic': True,
                'profile_type': 'dysbiotic',
                'severity': severity,
            },
        )

    def generate_infant(
        self,
        age_days: int = 30,
        n_taxa: int = 500,
        sample_id: Optional[str] = None,
    ) -> MicrobiomeSample:
        """
        Generate a synthetic infant gut microbiome profile.

        Infants have lower diversity that increases with age.
        Based on DIABIMMUNE study trajectories.

        Args:
            age_days: Age of infant in days (affects diversity).
            n_taxa: Number of taxa in the profile.
            sample_id: Optional sample identifier.

        Returns:
            MicrobiomeSample with age-appropriate parameters.
        """
        # k increases with age (from ~20 at birth to ~150 at 1 year)
        k_target = 20 + (age_days / 365) * 130
        k = int(np.clip(k_target + self.rng.normal(0, 15), 10, min(200, n_taxa)))

        # Generate abundances
        abundances = np.zeros(n_taxa)
        present_idx = self.rng.choice(n_taxa, size=min(k, n_taxa), replace=False)

        # Very uneven for young infants, more even with age
        sigma_param = 3.5 - min(2.0, age_days / 180)
        raw_abund = self.rng.lognormal(mean=0, sigma=sigma_param, size=k)
        abundances[present_idx] = raw_abund / np.sum(raw_abund)

        # Diversity increases with age
        sigma_base = 0.3 + (age_days / 365) * 0.4
        sigma = np.clip(sigma_base + self.rng.normal(0, 0.05), 0.2, 0.8)

        # Higher f in young infants (opportunistic colonization)
        f = np.clip(0.15 - (age_days / 365) * 0.1, 0.02, 0.25)

        # Higher correlation in young infants
        rho = np.clip(0.35 - (age_days / 365) * 0.1, 0.15, 0.45)

        taxa_ids = [f"genus_{i:03d}" for i in range(n_taxa)]

        return MicrobiomeSample(
            sample_id=sample_id or f"synthetic_infant_{age_days}d_{self.rng.integers(10000)}",
            abundances=abundances,
            taxa_ids=taxa_ids,
            k=k,
            rho=rho,
            sigma=sigma,
            f=f,
            metadata={
                'synthetic': True,
                'profile_type': 'infant',
                'age_days': age_days,
            },
        )

    def generate_antibiotic_perturbed(
        self,
        baseline: MicrobiomeSample,
        days_post_antibiotic: int = 0,
        antibiotic_type: str = "broad_spectrum",
    ) -> MicrobiomeSample:
        """
        Generate a post-antibiotic perturbation of a baseline profile.

        Simulates diversity crash and recovery over time based on
        Dethlefsen & Relman (2011) and other longitudinal studies.

        Args:
            baseline: Baseline healthy sample.
            days_post_antibiotic: Days since antibiotic ended.
            antibiotic_type: "broad_spectrum" or "narrow_spectrum".

        Returns:
            MicrobiomeSample with antibiotic perturbation effects.
        """
        if antibiotic_type == "broad_spectrum":
            # 60-80% diversity crash, recovery over ~4 weeks
            crash_factor = self.rng.uniform(0.2, 0.4)
            recovery_rate = 0.07  # Per day
        else:
            # 30-50% diversity crash, faster recovery
            crash_factor = self.rng.uniform(0.5, 0.7)
            recovery_rate = 0.1

        # Recovery trajectory (exponential return to baseline)
        recovery = 1 - (1 - crash_factor) * np.exp(-recovery_rate * days_post_antibiotic)
        recovery = min(recovery, 1.0)

        # Apply perturbation to abundances
        abundances = baseline.abundances.copy()

        # Selectively reduce many taxa
        survival_prob = crash_factor + (1 - crash_factor) * recovery
        survival_mask = self.rng.random(len(abundances)) < survival_prob
        abundances[~survival_mask] *= 0.01

        # Some taxa bloom opportunistically
        if days_post_antibiotic < 7:
            bloom_idx = self.rng.choice(np.where(abundances > 0)[0], size=3)
            abundances[bloom_idx] *= (3 + self.rng.exponential(2))

        # Renormalize
        abundances = abundances / np.sum(abundances)

        # Compute new variables
        k = int(np.sum(abundances > 1e-6))

        # Reduced diversity
        nonzero = abundances[abundances > 1e-10]
        if len(nonzero) > 1:
            H = -np.sum(nonzero * np.log(nonzero))
            H_max = np.log(len(nonzero))
            sigma = H / H_max if H_max > 0 else 0.0
        else:
            sigma = 0.0

        sigma = np.clip(sigma * recovery, 0.1, 0.9)

        # Elevated pathogen fraction during perturbation
        f = baseline.f + (0.15 - baseline.f) * (1 - recovery)
        f = np.clip(f, 0, 0.4)

        # Higher correlation (less complex community)
        rho = baseline.rho + (0.4 - baseline.rho) * (1 - recovery)

        return MicrobiomeSample(
            sample_id=f"{baseline.sample_id}_abx_d{days_post_antibiotic}",
            abundances=abundances,
            taxa_ids=baseline.taxa_ids,
            k=k,
            rho=rho,
            sigma=sigma,
            f=f,
            metadata={
                'synthetic': True,
                'profile_type': 'antibiotic_perturbed',
                'days_post_antibiotic': days_post_antibiotic,
                'antibiotic_type': antibiotic_type,
                'baseline_id': baseline.sample_id,
            },
        )

    def generate_batch(
        self,
        n_healthy: int = 50,
        n_dysbiotic: int = 20,
        n_infants: int = 10,
        n_taxa: int = 500,
    ) -> List[MicrobiomeSample]:
        """
        Generate a batch of synthetic samples with mixed profiles.

        Args:
            n_healthy: Number of healthy adult samples.
            n_dysbiotic: Number of dysbiotic samples.
            n_infants: Number of infant samples.
            n_taxa: Number of taxa in each profile.

        Returns:
            List of MicrobiomeSample objects.
        """
        samples = []

        # Healthy adults
        for i in range(n_healthy):
            samples.append(self.generate_healthy_adult(
                n_taxa=n_taxa,
                sample_id=f"batch_healthy_{i:03d}",
            ))

        # Dysbiotic
        for i in range(n_dysbiotic):
            severity = self.rng.uniform(0.3, 0.9)
            samples.append(self.generate_dysbiotic(
                n_taxa=n_taxa,
                severity=severity,
                sample_id=f"batch_dysbiotic_{i:03d}",
            ))

        # Infants at various ages
        for i in range(n_infants):
            age = self.rng.integers(7, 365)
            samples.append(self.generate_infant(
                age_days=age,
                n_taxa=n_taxa,
                sample_id=f"batch_infant_{i:03d}",
            ))

        return samples

    def _build_reference_taxa(self) -> None:
        """Build reference taxa names based on common gut bacteria."""
        # Top gut genera (simplified for synthetic data)
        self._reference_taxa = [
            "Bacteroides", "Faecalibacterium", "Roseburia", "Bifidobacterium",
            "Blautia", "Eubacterium", "Ruminococcus", "Coprococcus",
            "Lachnospira", "Dialister", "Prevotella", "Alistipes",
            "Parabacteroides", "Akkermansia", "Megasphaera", "Lactobacillus",
            "Streptococcus", "Veillonella", "Dorea", "Anaerostipes",
        ]


def load_american_gut_project(
    data_dir: Optional[Union[str, Path]] = None,
    taxonomic_level: str = "L6",
    max_samples: Optional[int] = None,
) -> MicrobiomeDataLoader:
    """
    Convenience function to load American Gut Project data.

    Args:
        data_dir: Directory containing AG data files.
        taxonomic_level: "L2" (phylum), "L3" (class), or "L6" (genus).
        max_samples: Maximum number of samples to load.

    Returns:
        Configured MicrobiomeDataLoader with AGP data.
    """
    loader = MicrobiomeDataLoader(data_dir=data_dir)

    # Map level to filename
    level_files = {
        "L2": "otu_table_L2.txt",
        "L3": "otu_table_L3.txt",
        "L6": "otu_table_L6.txt",
    }

    if taxonomic_level not in level_files:
        raise ValueError(f"Unknown taxonomic level: {taxonomic_level}. Use L2, L3, or L6.")

    otu_file = level_files[taxonomic_level]

    try:
        loader.load_otu_table(otu_file, max_samples=max_samples)
    except FileNotFoundError:
        print(f"OTU table not found. Ensure data is downloaded to {loader.data_dir}")
        raise

    try:
        loader.load_metadata("ag-cleaned.txt")
    except FileNotFoundError:
        print("Metadata file not found. Proceeding without metadata.")

    return loader


__all__ = [
    'MicrobiomeDataLoader',
    'MicrobiomeSample',
    'SyntheticMicrobiomeGenerator',
    'load_american_gut_project',
]
