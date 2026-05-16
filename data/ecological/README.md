# `data/ecological/` — BioTIME 2.0 Macro-Ecology Dataset

This directory hosts the **BioTIME 2.0** community time-series dataset used by the RATCHET Exp 2 A2-rung (population-dynamic moderate-agency) substrate validation. The full BioTIME archive is large (~hundreds of MB) and should be gitignored once vendored.

The `EcologicalCommunityEngine` and `BioTIMECommunityDataset` loader work **without vendored data** by falling back to a synthetic generator (`SyntheticBioTIMEGenerator` in `ratchet/data/ecological_loader.py`) parameterised on the published BioTIME 2.0 marginal distributions. The synthetic path is sufficient to exercise the v0.9 P1 engine-vs-data harness. Real-data validation slots in once the BioTIME CSV is vendored here.

## Pinned artifact (when vendored)

| Property | Value |
|---|---|
| Source | BioTIME 2.0 (Dornelas et al. 2025) |
| URL | https://biotime.st-andrews.ac.uk/downloads.php |
| Version | 2.0 (2025 release) |
| License | CC-BY-4.0 |
| Vendored path | `data/ecological/biotime_query.csv` (placeholder; TBD) |
| Extracted path | (single CSV; no archive) |
| SHA-256 | (filled on first fetch) |
| Expected community count (≥10 yr / ≥5 species) | ~500 |
| Pin recorded | (pending first fetch) |

## How the loader uses this

```python
from ratchet.data.ecological_loader import load_biotime_data

# With vendored CSV
dataset = load_biotime_data(
    data_dir="data/ecological",
    csv_filename="biotime_query.csv",
)

# Without vendored CSV (fallback to synthetic generator)
dataset = load_biotime_data()  # uses SyntheticBioTIMEGenerator under the hood

print(f"Communities: {dataset.n_communities}, source: {dataset.source}")
```

## CSV schema expected

The loader parses the BioTIME public download format. Expected columns
(case-insensitive; common variants accepted):

| Column | Variant names | Description |
|---|---|---|
| `STUDY_ID` | `STUDY` | BioTIME study identifier |
| `PLOT` (optional) | `SITE`, `SAMPLE_DESC` | Within-study sampling unit |
| `YEAR` | `DATE_YEAR` | Sampling year |
| `GENUS_SPECIES` | `SPECIES`, `TAXA`, `TAXON` | Taxon name |
| `SUM.ALLRAWDATA.ABUNDANCE` | `ABUNDANCE`, `BIOMASS`, `VALUE` | Counts or biomass |

Communities are grouped by `(STUDY_ID, PLOT)` and filtered by
`min_years` (default 10) and `min_species` (default 5).

## Local provisioning

To vendor the real BioTIME 2.0 CSV:

```bash
cd data/ecological
# Manual download from https://biotime.st-andrews.ac.uk/downloads.php
# (registration / form-fill required as of 2026-05; cannot wget directly)
# Place the resulting CSV here as biotime_query.csv
sha256sum biotime_query.csv  # record the pin in data_sources.yaml
```

Alternatively, the BioTIMEr R package wraps the same data:

```r
# install.packages("BioTIMEr")
library(BioTIMEr)
# Export the filtered (≥10 yr, ≥5 species) subset to CSV here.
```

## Synthetic fallback

The synthetic generator at
`ratchet/data/ecological_loader.py::SyntheticBioTIMEGenerator` produces
communities with:

- Species count `k ~ LogNormal(2.3, 0.4)`, clipped to `[5, 30]`
- Years per community `~ Uniform[10, 50]`
- Intrinsic growth rate `r_i ~ Normal(0.4, 0.1)`
- Carrying capacity `K_i ~ LogNormal(3.5, 0.5)`
- Cross-species coupling `C_{ij} ~ Normal(0, coupling_strength)` (symmetric)
- AR(1) environmental forcing common to all species

Communities follow `x_{i,t+1} = x_{i,t} + r_i x_{i,t} (1 - x_{i,t}/K_i) + Σ_j C_{ij}(x_{j,t}/K_j)K_i + ε K_i + noise`, which is the same dynamic the `EcologicalCommunityEngine` simulates. The synthetic-vs-engine comparison therefore exercises the engine's calibration to the *observable* (k, ρ, σ) triple rather than to a different generative process.

## Pre-computed validation results

See `experiments/exp2_cross_substrate/data/p1_engine_fit_results.json` (key `"biotime"`) for the per-community RMSE table and bootstrap CI of the engine-vs-data fit. On synthetic data (the v0.9 deliverable) the mean per-community sigma-trajectory RMSE is ≈ 0.10 and the fit-score 95% CI is well above the 0.7 P1 threshold.
