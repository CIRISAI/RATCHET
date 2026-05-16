# `data/powergrid/` — PNNL PMU Grid-Event Dataset

This directory hosts the **PNNL Open-Source PMU Library** (PNNL-30492) transmission-event traces used by the RATCHET Exp 2 A0-rung (engineered / inert constituent agency) substrate validation. The full PNNL event archive is large (~1,694 events; multiple GB across interconnects) and should be gitignored once vendored.

The `PMUGridEngine` and `PNNLPMUDataset` loader work **without vendored data** by falling back to a synthetic generator (`SyntheticPMUEventGenerator` in `ratchet/data/powergrid_loader.py`) parameterised on published power-system swing dynamics (Bergen-Vittal / Kundur model, IEEE C37.118.1 PMU standard). The synthetic path is sufficient to exercise the v1.0 P1 engine-vs-data harness. Real-data validation slots in once the PNNL parquet is vendored here.

## Pinned artifact (when vendored)

| Property | Value |
|---|---|
| Source | PNNL Open-Source PMU Library (PNNL-30492) |
| URL | https://www.pnnl.gov/main/publications/external/technical_reports/PNNL-30492.pdf |
| Event registry | https://gridevents.pnnl.gov |
| Alt source (DOE) | https://data.openei.org/submissions |
| Alt source (OPSD) | https://open-power-system-data.org/ |
| Version | static (PNNL-30492 2020 release) |
| License | open-source (DOE, U.S. Government public-domain) |
| Vendored path | `data/powergrid/pnnl_events.parquet` (preferred) |
| Alt paths | `pnnl_events_sample.parquet`, `pnnl_pmu_events.parquet`, `pmu_events.parquet` |
| Sample-rate convention | 30 Hz (C37.118.1 standard; 60 Hz grid) |
| SHA-256 | (filled on first fetch) |
| Expected event count (≥3 PMUs / ≥5s post-event) | ~1,694 |
| Pin recorded | (pending first fetch) |

## How the loader uses this

```python
from ratchet.data.powergrid_loader import load_pnnl_pmu_events

# With vendored parquet
dataset = load_pnnl_pmu_events(
    data_dir="data/powergrid",
    parquet_filename="pnnl_events.parquet",
)

# Without vendored parquet (fallback to synthetic generator)
dataset = load_pnnl_pmu_events()  # uses SyntheticPMUEventGenerator under the hood

print(f"Events: {dataset.n_events}, source: {dataset.source}")
```

The loader auto-detects any of these vendored filenames in priority order:
1. `pnnl_events.parquet` (canonical pin)
2. `pnnl_events_sample.parquet` (small-sample bonus drop)
3. `pnnl_pmu_events.parquet`, `pmu_events.parquet` (aliases)

If none exist, the synthetic fallback runs.

## Parquet schema expected

The loader parses a long-form (one row per (event, pmu, sample)) parquet.
Expected columns (case-insensitive; common variants accepted):

| Column | Variant names | Description |
|---|---|---|
| `event_id` | `event`, `eventid` | PNNL event identifier (e.g. `evt_001234`) |
| `pmu_id` | `pmu`, `pmuid`, `station_id` | PMU / substation identifier |
| `timestamp_s` | `timestamp`, `time_s`, `t` | Seconds relative to event onset; <0 pre, ≥0 post |
| `frequency_hz` | `frequency`, `freq`, `f_hz` | Synchrophasor frequency reading in Hz |
| `sample_rate_hz` (optional) | `sample_rate` | PMU reporting rate (defaults to 30 Hz) |
| `event_type` (optional) | `type` | "line_trip", "gen_loss", "oscillation", … |
| `region` (optional) | `interconnect` | "WECC", "ERCOT", "EI" |

Events are grouped by `event_id` and filtered by `min_pmus` (default 3) and `min_post_event_s` (default 5.0 seconds).

## Local provisioning

To vendor the real PNNL PMU library:

```bash
cd data/powergrid

# Option 1: PNNL Grid Event registry (form-fill / agreement may apply)
# Browse https://gridevents.pnnl.gov, request data, convert CSV/HDF5 → parquet.

# Option 2: DOE Open Energy Data Initiative
# Search https://data.openei.org/submissions for "PMU" or "synchrophasor".

# Option 3: Open Power System Data (synthetic + measured frequency series)
# https://open-power-system-data.org/  → convert to the long-form schema above.

# Once you have a parquet conforming to the schema:
python3 -c "
import pandas as pd
df = pd.read_parquet('pnnl_events.parquet')
print(df.head())
print('events:', df['event_id'].nunique())
print('PMUs total:', df['pmu_id'].nunique())
"

sha256sum pnnl_events.parquet  # record the pin in data_sources.yaml
```

After vendoring, update `experiments/exp2_cross_substrate/data_sources.yaml`:

```yaml
pmu_pnnl:
  rung: A0
  engine: ratchet.engines.powergrid
  loader: ratchet.data.powergrid_loader
  vendored_path: "data/powergrid/pnnl_events.parquet"
  expected_sha256: "<sha-256-output-from-above>"
  expected_event_count: 1694
```

## Synthetic fallback

The synthetic generator at
`ratchet/data/powergrid_loader.py::SyntheticPMUEventGenerator` produces
events with:

- PMU count `k ~ LogNormal(2.1, 0.5)`, clipped to `[3, 30]`
- Duration 60 s (30 s pre-event + 30 s post-event)
- Sample rate 30 Hz (C37.118.1 standard)
- Disturbance magnitude `0.05–0.5 Hz` (Uniform)
- Inter-PMU coupling `K_ij = base / d_ij^1.5`, with PMUs at random `[0,1]²` positions
- Per-PMU swing parameters `H ~ LogNormal(5s, 0.15)`, `D ~ LogNormal(0.4, 0.15)`
- AR(1) common-mode drift driving baseline cross-PMU correlation
- Disturbance epicentre at random position; per-PMU disturbance weight = `exp(-2 · d_to_epicentre)`

Frequencies follow the linearised Bergen-Vittal swing equation:

```
df_i/dt = -(D_i / 2H_i) (f_i - f_nom)
          + Σ_j K_ij (f_j - f_i) / (2H_i)
          + ε_common(t)
          + noise_i(t)
          + disturbance_i(t)
```

The `PMUGridEngine` simulates the same dynamics. The synthetic-vs-engine comparison therefore exercises the engine's calibration to the *observable* `(k, ρ, σ)` triple rather than to a different generative process.

## SHA-pin protocol

After dropping the first vendored parquet here:

```bash
sha256sum data/powergrid/pnnl_events.parquet
```

Paste the sum under `expected_sha256` in `experiments/exp2_cross_substrate/data_sources.yaml` and commit. The quarterly substrate-revalidation CI re-checks this pin and opens a GitHub issue if the upstream archive's hash drifts (indicating a re-curation or schema break).

## Pre-computed validation results

See `experiments/exp2_cross_substrate/data/p1_engine_fit_results.json` (key `"pmu_grid"`) for the per-event RMSE table and bootstrap CI of the engine-vs-data fit. On synthetic data (the v1.0 deliverable) the mean per-event frequency-trajectory RMSE is around 0.05–0.15 Hz and the fit-score 95% CI sits above the 0.7 P1 tolerance-band threshold.
