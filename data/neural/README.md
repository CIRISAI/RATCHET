# `data/neural/` — Allen Brain Observatory Neuropixels Dataset

This directory hosts the **Allen Brain Observatory — Visual Coding Neuropixels** dataset used by the RATCHET Exp 2 A1-rung (low-agency, cellular-signaling) substrate validation. The full archive is large (~hundreds of GB across ~80 NWB sessions) and should be gitignored once vendored.

The `NeuralPopulationEngine` and `AllenNeuropixelsDataset` loader work **without vendored data** by falling back to a synthetic generator (`SyntheticAllenNeuropixelsGenerator` in `ratchet/data/neural_loader.py`) parameterised on the published Allen Neuropixels session distributions (Siegle et al. 2021, *Nature* 592). The synthetic path is sufficient to exercise the v0.9 P1 engine-vs-data harness. Real-data validation slots in once the parquet is vendored here.

## Pinned artifact

### Currently vendored: 3-session real-data sample

| Property | Value |
|---|---|
| Source | Allen Brain Observatory — Visual Coding Neuropixels (2019 release + extensions) |
| URL | https://portal.brain-map.org/explore/circuits/visual-coding-neuropixels |
| S3 (direct, anonymous) | `https://allen-brain-observatory.s3.amazonaws.com/visual-coding-neuropixels/ecephys-cache/` |
| SDK requirement | **NONE** — vendored via `h5py + fsspec + aiohttp` (allensdk not needed) |
| License | ABO data-use-agreement (permissive research) |
| Sample parquet | `data/neural/allen_neuropixels_sample.parquet` |
| Sample SHA-256 | `061d9cc18e49fd9794d7eda525acbd1ae44f2d88e9e514470219fa3298ee259f` |
| Sample size | ~4.8 MB |
| Sample sessions | 3 (IDs: 715093703, 719161530, 721123822) |
| Sample neurons per session | 60 (quality='good', isi<0.5, snr>1.0) |
| Sample bin width | 10 ms |
| Sample trials per session | 598 valid drifting-grating presentations |
| Vendor script | `scripts/vendor_allen_neuropixels.py` |
| First fetch | 2026-05-16 |

### Full vendoring (TBD)

| Property | Value |
|---|---|
| Full parquet | `data/neural/allen_neuropixels_sessions.parquet` (placeholder; TBD) |
| Expected session count | 58 sessions with NWB available (manifest shows 58 of 80 with `has_nwb=True`) |
| SHA-256 | (filled on first full fetch) |
| Estimated size | ~90 MB at 10-ms bins, 60 units/session

## How the loader uses this

```python
from ratchet.data.neural_loader import load_allen_neuropixels_sessions

# With vendored parquet
dataset = load_allen_neuropixels_sessions(
    data_dir="data/neural",
    parquet_filename="allen_neuropixels_sessions.parquet",
)

# Without vendored parquet (fallback to synthetic generator)
dataset = load_allen_neuropixels_sessions()  # uses SyntheticAllenNeuropixelsGenerator

print(f"Sessions: {dataset.n_sessions}, source: {dataset.source}")
```

## Parquet schema expected

The loader consumes a per-session row format. To vendor real data, extract from NWB (via allensdk) and serialise these columns:

| Column | Type | Description |
|---|---|---|
| `session_id` | str | Allen session identifier (e.g. `"session_715093703"`) |
| `n_neurons` | int | Number of recorded units in the session |
| `n_trials` | int | Number of drifting-grating trials |
| `bin_ms` | float | Bin width in milliseconds (1.0 is standard) |
| `spike_train_matrix` | list-of-lists or bytes (int16) | Shape `(n_neurons, n_time_bins)` spike counts in 1-ms bins; flatten row-major if bytes-encoded |
| `stimulus_labels` | list[int] | Length `n_trials`; orientation index in `[0, 8)` (45° increments, 0..315°) |
| `trial_bin_edges` | list[int] | Length `n_trials + 1`; bin indices delimiting each trial window in `spike_train_matrix` |
| `visual_area` | str (optional) | Allen anatomical label: `VISp`, `VISal`, `VISpm`, `VISrl`, `VISl`, `VISam` |
| `rho_precomputed` | float (optional) | Mean pairwise abs-Pearson spike-train correlation; recomputed if absent |
| `sigma_precomputed` | float (optional) | Cross-validated decoding accuracy; recomputed if absent |

Sessions are filtered by `min_neurons` (default 30) and `min_trials` (default 16).

## Local provisioning

The repo ships `scripts/vendor_allen_neuropixels.py` which streams NWB files
directly from the public anonymous S3 bucket (no allensdk required):

```bash
pip install h5py fsspec aiohttp pyarrow  # NOT in RATCHET requirements; vendoring-only
python3 scripts/vendor_allen_neuropixels.py --n-sessions 3 --max-units 60
# → data/neural/allen_neuropixels_sample.parquet (~5 MB; takes ~1 min/session)

# To pin a different subset (any 3 session IDs from sessions.csv):
python3 scripts/vendor_allen_neuropixels.py \
    --session-ids 715093703,719161530,721123822

# Full vendoring (all 58 sessions; ~30 min wall time, ~90 MB on-disk):
python3 scripts/vendor_allen_neuropixels.py --n-sessions 58 \
    --out data/neural/allen_neuropixels_sessions.parquet
```

Then record the SHA-256 pin:

```bash
sha256sum data/neural/allen_neuropixels_sample.parquet
# Paste the digest into experiments/exp2_cross_substrate/data_sources.yaml
# under substrates.allen_neuropixels.{sample_sha256, expected_sha256}
```

### Vendoring internals (what the script does)

1. Fetches `https://allen-brain-observatory.s3.amazonaws.com/visual-coding-neuropixels/ecephys-cache/sessions.csv` — the 58-session manifest.
2. For each session ID, opens `session_<id>.nwb` (2-3 GB each) via `fsspec.open(..., mode='rb', block_size=1MB)` → h5py reads only the chunks it needs (so total downloaded ≪ NWB size; typical run = ~50 MB / session pulled over the wire).
3. Reads `intervals/drifting_gratings_presentations` (start_time, stop_time, orientation, temporal_frequency).
4. Filters trials to standard drifting-grating conditions (`temporal_frequency ∈ {1, 2, 4, 8, 15} Hz`, `orientation ∈ {0, 45, ..., 315}°`).
5. Reads `units/quality`, `units/isi_violations`, `units/snr` and filters to `quality=='good' & isi<0.5 & snr>1.0`. Subsamples to `--max-units` per session.
6. Per selected unit, reads its spike-time slice from `units/spike_times` using `units/spike_times_index` boundaries.
7. Bins spikes at `--bin-ms` (default 10 ms) within each trial window `[start_time, start_time + 2s)`.
8. Encodes the `(n_neurons, n_time_bins)` int16 matrix as bytes for parquet column.
9. Writes parquet with the schema documented above.

Per-session wall time on a typical broadband connection: 15-20 s (dominated by random-access S3 reads).

### Alternative: allensdk (heavyweight)

```bash
pip install allensdk
# Then use EcephysProjectCache.from_warehouse() per Allen's SDK notebooks.
# The output is functionally equivalent to vendor_allen_neuropixels.py
# but pulls the entire NWB file to local disk first (2-3 GB per session).
```

## Synthetic fallback

The synthetic generator at
`ratchet/data/neural_loader.py::SyntheticAllenNeuropixelsGenerator`
produces sessions with:

- Neuron count `k ~ LogNormal(log(90), 0.4)`, clipped to `[30, 350]`
- Per-trial duration 2000 ms, 1-ms bins
- 8 drifting-grating orientations × 10 reps = 80 trials per session
- Per-neuron baseline rate `r_i ~ LogNormal(log(5), 0.5)` Hz, clipped `[0.5, 30]`
- Common-input AR(1) latent (φ=0.95) driving cross-neuron correlation ρ via
  per-neuron gain `g_i ~ Uniform(0.5, 1.5) · common_input_coupling`
- Orientation-tuned drive `1 + tune_strength · cos(θ - θ_pref_i) · tw_i / 1.5`
- Poisson emission per bin: `spike[i, t] ~ Poisson(rate[i, t])`

This is the same dynamic the `NeuralPopulationEngine` simulates. The
synthetic-vs-engine comparison therefore exercises the engine's
calibration to the *observable* (k, ρ, σ) triple rather than to a
different generative process.

## Pre-computed validation results

Real-data P1 validation (vendored 2026-05-16, 3 sessions):

| Metric | Value |
|---|---|
| Source | `allen_parquet` (real ABO Neuropixels) |
| Sessions | 3 |
| Mean RMSE | 0.216 |
| Min/Max RMSE | 0.175 / 0.285 |
| Fit-score (point) | 0.814 |
| Fit-score 95% CI | [0.68, 0.88] |
| `passes_p1` (point ≥ 0.6 AND CI high ≥ 0.7) | **TRUE** |
| `passes_p1_strict` (CI low ≥ 0.7) | False |

Per-session detail and bootstrap CI live in
`experiments/exp2_cross_substrate/data/p1_engine_fit_results.json`
under key `"allen_neuropixels"` once `run_allen_p1()` is wired into
`main()`. Synthetic-only validation (the v0.9 fallback when the parquet
is absent) typically lands at RMSE ≈ 0.10–0.15 and fit-score ≈ 0.92.

## RATCHET (k, ρ, σ) mapping

Per `experiments/exp2_cross_substrate/REGIME.md` §"A1 — Allen neural firing":

| Var | Definition | Source |
|---|---|---|
| k | Simultaneously-recorded neurons per session | Allen SDK |
| ρ | Mean pairwise abs-Pearson on 1-ms binned spike trains | Computed |
| σ | Population-decoding accuracy on drifting gratings (cross-validated linear classifier) | Computed |
| n | ~80 sessions | Allen Brain Observatory |
