"""
RATCHET Macro-Ecology (BioTIME) Substrate Engine — STUB

Domain mapping (per REGIME.md §"A2 — BioTIME macro-ecology"):
    k (constraints):    Species count in a community time series
    rho (correlation):  Mean pairwise correlation of species-abundance time series
    sigma (sustain.):   Inverse CV of total biomass over time (stability)

Constituent agency: A2 (moderate). Populations have aggregate dynamics
beyond pure homeostasis but no goal-directed coordination. Predicted
residual structure: weak but measurable, driven by environmental
forcing and species interactions beyond what Kish predicts.

References:
    - BioTIME 2.0 (Dornelas et al. 2025, Global Ecology and Biogeography)
    - Tilman 1996, Yachi & Loreau 1999, Loreau & de Mazancourt 2013
      (the diversity-stabilizes-ecosystems / insurance hypothesis lineage)

Distinct from `microbiome.py` (A1, bacterial homeostasis) by scale —
this is macro-organisms in named ecological assemblages.

Pairs with: ratchet/engines/{battery,institutional,microbiome}.py
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class EcoCommunity:
    """Per-community time series (k, rho, sigma) triple."""
    study_id: str
    k: int          # species count (filtered ≥ 5)
    rho: float      # mean pairwise abundance-correlation
    sigma: float    # inverse CV of total biomass over years
    years: int      # number of years sampled (filtered ≥ 10)
    realm: Optional[str] = None  # marine / terrestrial / freshwater
    taxonomic_group: Optional[str] = None


def compute_abundance_correlation(species_abundance_matrix) -> float:
    """Reduce a (species × time) abundance matrix to scalar rho.

    Operationalization: mean of absolute pairwise Pearson correlations
    across species pairs over the full time series. Time-series
    pre-processed via detrending if non-stationary.

    TODO.
    """
    raise NotImplementedError("Species-abundance correlation reducer — TODO")


def compute_biomass_stability(total_biomass_by_year) -> float:
    """1 / CV of total biomass = stability metric.

    Higher value = lower coefficient of variation = more stable ecosystem.
    Maps to sigma in the Kish-fit regression.

    TODO.
    """
    raise NotImplementedError("Biomass stability — TODO")


def load_biotime_communities(vendored_dir) -> list[EcoCommunity]:
    """Load BioTIME 2.0 communities meeting filter criteria.

    Filter: years ≥ 10, species count ≥ 5, with non-trivial biomass time series.
    Expected n: ~500.

    TODO: BioTIMEr R package + SQLite/CSV loader.
    """
    raise NotImplementedError("BioTIME loader — TODO")


def fit_kish_ecological(communities: list[EcoCommunity]):
    """Kish-fit regression. Returns (r_squared, ci_lo, ci_hi).

    TODO uniform with battery.py.
    """
    raise NotImplementedError("Kish fit + bootstrap CI — TODO")


def residual_whiteness(communities, fit) -> float:
    """Predicted at A2: lower whiteness than A0/A1 (weak structured residual
    from environmental forcing). See REGIME.md §P2."""
    raise NotImplementedError("Residual whiteness — TODO")


def signed_delta_rho_pre_collapse(communities, pre_window_years: int = 5):
    """Pre-collapse Δρ test (REGIME.md §P3).

    For communities that subsequently collapsed (biomass drop > 50%),
    measure ρ in the pre_window and compare to baseline. Predicted at A2:
    mostly negative (disintegration before collapse), but with some
    positive cases if there's intentional human management.

    TODO.
    """
    raise NotImplementedError("Pre-collapse Δρ test — TODO")
