# `data/protein/` — AlphaFold CATH-S40 Protein Structure Dataset

This directory hosts the **AlphaFold DB v6** CATH-S40 representative single-domain protein dataset used by the RATCHET Exp 2 A0-rung (chemical / agency ~0) substrate validation. The full AlphaFold archive is enormous (TB scale); we vendor only the CATH-S40 single-domain subset (~10,000 proteins) and store it as a parquet for efficient loading. Vendored files should be gitignored once size exceeds a few MB.

The `ProteinFoldingEngine` and `CATHS40ProteinDataset` loader work **without vendored data** by falling back to a synthetic generator (`SyntheticAlphaFoldGenerator` in `ratchet/data/protein_loader.py`) parameterised on the published AlphaFold pLDDT marginal distributions (Jumper et al. 2021). The synthetic path is sufficient to exercise the v1.0 P1 engine-vs-data harness. Real-data validation slots in once the CATH-S40 parquet is vendored here.

## Pinned artifact

### Real-data sample (vendored 2026-05-16)

| Property | Value |
|---|---|
| Source | AlphaFold Protein Structure Database v6 (UniProt 2025_03 sync) |
| URL | https://ftp.ebi.ac.uk/pub/databases/alphafold/ |
| Per-protein URL | https://alphafold.ebi.ac.uk/files/AF-{uniprot}-F1-model_v6.pdb |
| Hugging Face mirror | https://huggingface.co/datasets/HUBioDataLab/AlphafoldStructures |
| Version | v6 (2024) |
| License | CC-BY-4.0 |
| Vendored sample path | `data/protein/cath_s40_alphafold_sample.csv` |
| Sample SHA-256 | `acf51e5f11a519d0308ab855220b58bb240e18f0b1f82cb03bea99021360ab44` |
| Sample protein count | 74 (single-domain, length 59-732 residues) |
| Provenance | First 100 PDBs from HF AlphafoldStructures + ~25 well-known UniProts (P00733, P02185, hemoglobin/lysozyme/trypsin/serpins/E. coli small proteins) fetched from EBI direct |

### Full set (TBD)

| Property | Value |
|---|---|
| Vendored path | `data/protein/cath_s40_alphafold.parquet` (placeholder) |
| SHA-256 | (filled on first full-set fetch) |
| Expected protein count (filtered ≥ 40 residues, ≤ 800 residues) | ~10,000 |
| Pin recorded | (pending) |

## How the loader uses this

```python
from ratchet.data.protein_loader import load_cath_s40_alphafold_data

# With vendored parquet
dataset = load_cath_s40_alphafold_data(
    data_dir="data/protein",
    parquet_filename="cath_s40_alphafold.parquet",
)

# Without vendored parquet (fallback to synthetic generator)
dataset = load_cath_s40_alphafold_data()

print(f"Proteins: {dataset.n_proteins}, source: {dataset.source}")
```

## Parquet schema expected

The loader supports two layouts (auto-detected via column inspection).

### Short format (one row per protein) — preferred

| Column | Type | Description |
|---|---|---|
| `uniprot_id` | str | UniProt accession (e.g. "P12345") |
| `sequence_length` | int | Residue count |
| `plddt_trajectory` | list/array/json | Per-residue pLDDT in [0,100], length k |
| `mean_plddt` | float | Convenience: mean of trajectory |
| `b_factor_correlation` | float | Convenience: ρ from compute_residue_correlation |
| `cath_class` (optional) | str | CATH top-level class "1"-"4" |

### Long format (one row per residue) — fallback

| Column | Type | Description |
|---|---|---|
| `uniprot_id` | str | UniProt accession |
| `residue_index` | int | 0-based or 1-based residue position |
| `plddt` | float | pLDDT score in [0,100] |
| `cath_class` (optional) | str | CATH top-level class |

The loader groups long-format rows by `uniprot_id`, sorts by `residue_index`, and builds the per-protein trajectory.

## Local provisioning

To vendor the real AlphaFold CATH-S40 dataset:

```bash
cd data/protein

# Step 1: pull the CATH-S40 representative UniProt list
# (see https://www.cathdb.info/wiki?id=release:v4_3_0)
# This yields a list of ~10,000 UniProt accessions.

# Step 2: for each UniProt ID, fetch the AlphaFold model PDB
# (each PDB is ~100-500 KB; total ~3-5 GB raw)
while read uniprot; do
    wget -q "https://alphafold.ebi.ac.uk/files/AF-${uniprot}-F1-model_v4.pdb" \
        -O "raw/AF-${uniprot}-F1-model_v4.pdb"
done < cath_s40_uniprot_list.txt

# Step 3: extract per-residue pLDDT from the B-factor column of each PDB
# (CA atom B-factor = pLDDT in AlphaFold PDB output)
python3 -c "
from pathlib import Path
import re, json
import pandas as pd
rows = []
for pdb in Path('raw').glob('AF-*.pdb'):
    uniprot = re.search(r'AF-([^-]+)-', pdb.name).group(1)
    plddt = []
    for line in pdb.read_text().splitlines():
        if line.startswith('ATOM') and line[12:16].strip() == 'CA':
            plddt.append(float(line[60:66]))
    rows.append({
        'uniprot_id': uniprot,
        'sequence_length': len(plddt),
        'plddt_trajectory': json.dumps(plddt),
        'mean_plddt': sum(plddt) / len(plddt),
    })
pd.DataFrame(rows).to_parquet('cath_s40_alphafold.parquet')
"

# Step 4: pin the SHA in data_sources.yaml
sha256sum cath_s40_alphafold.parquet  # record under substrates.alphafold.expected_sha256
```

For a minimal sanity-check sample (say 50 proteins) drop them at
`data/protein/cath_s40_alphafold_sample.parquet` — the loader will pick it
up if the full parquet is absent (it tries both filenames).

## Synthetic fallback

The synthetic generator at
`ratchet/data/protein_loader.py::SyntheticAlphaFoldGenerator` produces
proteins with:

- Sequence length `k ~ LogNormal(5.2, 0.6)`, clipped to `[40, 800]`
- Mean pLDDT `~ Normal(85, 5)`, clipped to `[60, 95]` (matches published AlphaFold marginals)
- Per-residue pLDDT std `~ Normal(8, 2)`, clipped to `[3, 20]`
- Correlation length `L` per CATH class (mainly-α: ~22 residues; mainly-β: ~12; α/β: ~16; few-SS: ~8)
- Per-residue pLDDT drawn from a multivariate Gaussian with mean = mean_plddt · **1** and covariance `K[i,j] = σ² · exp(-|i-j|/L)`

Per-residue trajectories follow this exponential-decay spatial correlation pattern — the same structural-coupling operationalisation that `ProteinFoldingEngine` produces via residue-level dynamics with local + non-local coupling. The synthetic-vs-engine comparison therefore exercises the engine's calibration to the *observable* (k, ρ, σ) triple rather than to a different generative process.

## Pre-computed validation results

See `experiments/exp2_cross_substrate/data/p1_engine_fit_results.json` (key `"alphafold"`) for the per-protein RMSE table and bootstrap CI of the engine-vs-data fit. On synthetic data (the v1.0 deliverable) the mean per-protein pLDDT-trajectory RMSE is ≈ 0.08 in sigma-scale (≈ 8 in pLDDT-0-100 scale) and the fit-score 95% CI is well above the 0.7 P1 threshold.
