# `data/microbiome/` — American Gut Project (AGP) Cohort

This directory hosts the **American Gut Project / Microsetta** OTU/ASV
abundance tables used by the RATCHET Exp 2 A1-rung
(homeostatic / cellular-signaling low-agency) substrate validation.
The full AGP archive is large (multi-GB at strain resolution; ~tens of
MB at genus level) and should be gitignored once vendored.

The `MicrobiomeEngine` and `MicrobiomeDataLoader` work **without
vendored data** by falling back to a synthetic generator
(`SyntheticMicrobiomeGenerator` in
`ratchet/data/microbiome_loader.py`) parameterised on AGP + HMP +
DIABIMMUNE marginal distributions. The synthetic path is sufficient to
exercise the v1.0 P1 engine-vs-data harness
(`tests/test_microbiome_p1.py`). Real-data validation slots in once the
AGP OTU table is vendored here.

## Pinned artifact (when vendored)

| Property | Value |
|---|---|
| Source | American Gut Project (McDonald et al. 2018, mSystems 3:e00031-18) |
| Primary URL | https://microsetta.ucsd.edu/american-gut-project/ |
| Qiita study | 10317 (auth-walled but free) |
| Mirror | `ftp://ftp.microbio.me/AmericanGut/latest/` |
| Version | public-archive-2026-02 (enrollment paused) |
| License | Public Domain (after PII scrubbing) |
| Vendored path | `data/microbiome/otu_table_L6.txt` (placeholder; TBD) |
| Metadata path | `data/microbiome/ag-cleaned.txt` (placeholder; TBD) |
| Extracted path | (TSV; no archive) |
| SHA-256 | (filled on first fetch) |
| Expected sample count (fecal, post-QC) | ~5000 |
| Pin recorded | (pending first fetch) |

## How the loader uses this

```python
from ratchet.data.microbiome_loader import load_american_gut_project

# With vendored OTU + metadata tables
loader = load_american_gut_project(
    data_dir="data/microbiome",
    taxonomic_level="L6",  # L2 = phylum, L3 = class, L6 = genus
    max_samples=500,
)
samples = loader.get_samples(n=100, body_site="UBERON:feces")

# Without vendored data: use the synthetic generator directly
from ratchet.data.microbiome_loader import SyntheticMicrobiomeGenerator
g = SyntheticMicrobiomeGenerator(seed=42)
batch = g.generate_batch(n_healthy=60, n_dysbiotic=25, n_infants=15)
```

## TSV schema expected

The loader parses the QIIME-style L6 OTU table (BIOM `summarize_taxa.py`
output). Expected layout:

- Row 0: `#OTU ID` (or comment line skipped by `comment='#'`)
- Column 0: taxon string at the chosen level (e.g. `k__Bacteria;p__Firmicutes;...;g__Faecalibacterium`)
- Columns 1+: sample IDs; values are integer counts or normalised abundances

Pathogen / dysbiosis-associated taxa are identified by substring match
on the genus name; see `MicrobiomeDataLoader.PATHOGEN_PATTERNS` for the
default list (Clostridioides, Enterococcus, Klebsiella, Escherichia,
Campylobacter, Salmonella, etc.).

Metadata (sample → body site, antibiotic history, age, BMI, etc.) is
parsed from a separate `ag-cleaned.txt` (TSV with header row;
`SAMPLE_NAME` is the join key). The loader is tolerant of missing
metadata — abundance-only loading still produces fully populated
MicrobiomeSample objects.

## Local provisioning

To vendor the real AGP genus-level table:

```bash
cd data/microbiome
# Manual download from https://qiita.ucsd.edu/study/description/10317
# (registration required; free)
# Or wget from the ftp mirror if still up:
wget ftp://ftp.microbio.me/AmericanGut/latest/03-otus/100nt/gg-13_8-97-percent/otu_table_L6.txt
wget ftp://ftp.microbio.me/AmericanGut/latest/01-cleaned/ag-cleaned.txt

sha256sum otu_table_L6.txt ag-cleaned.txt  # record the pins in data_sources.yaml
```

Alternatively, redbiom / qiita-client wrap the same data:

```bash
pip install redbiom
redbiom select samples-from-metadata --context "Deblur-Illumina-16S-V4-150nt-780653" \
    "WHERE qiita_study_id == '10317' AND body_site == 'UBERON:feces'" > agp_sample_ids.txt
redbiom fetch sample-metadata --from agp_sample_ids.txt > ag-cleaned.txt
redbiom fetch samples --from agp_sample_ids.txt --context "..." --output otu_table_L6.biom
# Then convert BIOM → TSV via `biom summarize-table` / `biom convert`
```

## Synthetic fallback

The synthetic generator at
`ratchet/data/microbiome_loader.py::SyntheticMicrobiomeGenerator`
produces three baseline cohorts:

| Profile | k | σ | f | rho |
|---|---|---|---|---|
| Healthy adult | LogNormal(4.5, 0.4) ∈ [80, 400] | Beta(8, 2) · 0.35 + 0.6 ∈ [0.6, 0.95] | Exponential(0.03), capped at 0.15 | Normal(0.22, 0.05) ∈ [0.10, 0.40] |
| Dysbiotic | LogNormal(4.0 − sev·0.5, 0.5) ∈ [20, 250] | 0.7 − sev·0.4 + ε | 0.05 + sev·0.25 + ε ∈ [0, 0.5] | 0.30 + sev·0.15 + ε ∈ [0.20, 0.60] |
| Infant (age days) | 20 + (age/365)·130 + ε | 0.3 + (age/365)·0.4 ∈ [0.2, 0.8] | 0.15 − (age/365)·0.1 ∈ [0.02, 0.25] | 0.35 − (age/365)·0.1 ∈ [0.15, 0.45] |

Abundances are log-normal `LN(0, σ_param)` over a randomly chosen `k`
of `n_taxa` slots, renormalised to sum to 1. Antibiotic perturbation
trajectories follow Dethlefsen & Relman (2011) shape:
exponential return to baseline with rate ≈ 0.07/day (broad-spectrum)
or 0.10/day (narrow-spectrum), 60–80 % initial diversity crash, and
opportunistic blooms within the first 7 days.

The synthetic-vs-engine comparison therefore exercises the engine's
calibration to the *observable* (k, ρ, σ) triple plus a known
post-antibiotic σ-trajectory, rather than to a different generative
process. The recovery shape itself is the operator-property test.

## Pre-computed validation results

See `experiments/exp2_cross_substrate/data/p1_engine_fit_results.json`
(key `"microbiome"`) for the per-sample σ-trajectory RMSE table and
bootstrap CI of the engine-vs-data fit. On synthetic data (the v1.0
deliverable) the mean per-sample σ-trajectory RMSE is ≈ 0.13 and the
fit-score 95 % CI is well above the 0.7 P1 threshold (point ≈ 0.93;
CI ≈ [0.92, 0.94] on n=100 with the default cohort mix).

## SHA-pin protocol

When real AGP data is first vendored:

1. Place the OTU table at `data/microbiome/otu_table_L6.txt` and the
   metadata at `data/microbiome/ag-cleaned.txt`.
2. Compute `sha256sum` on both files; record the hashes in
   `experiments/exp2_cross_substrate/data_sources.yaml` under the
   `agp_microbiome` entry's `expected_sha256` field (use the OTU
   hash as the primary pin; metadata SHA recorded in `note:`).
3. Update `version:` to the snapshot date
   (e.g. `public-archive-YYYY-MM`) and bump `last_updated:` in the
   yaml top-matter.
4. Re-run `python3 tests/test_microbiome_p1.py` to confirm the loader
   path reads the new files cleanly (the harness automatically
   prefers real data over the synthetic fallback when the OTU table
   is present).
5. Commit only the SHA pin + yaml updates; the data itself stays
   gitignored.
