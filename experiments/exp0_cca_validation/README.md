# Exp 0 — CCA Tier-1 Validation Rig (cherry-picked from `track-b-data` + `RATCHET-RESET`)

This directory consolidates the validation infrastructure that produced the CCA paper's Tier-1 numbers (NASA battery 8.1% RMSE, QoG/Polity 5/5 TN, AGP Shannon=0.580, ρ_critical=0.43 threshold). Until now this lived on side branches and a non-git working tree; it is now on master.

## What's in here

### Run scripts (entry points)

| Script | What it validates | Source |
|---|---|---|
| `run_rho_threshold_validation.py` | ρ=0.43 universal threshold via chi-square + bootstrap (n=3000 sims) | RATCHET-RESET/run_validation.py |
| `real_data_validation.py` | Cross-domain causal validation (markets, earthquakes, etc.) | RATCHET-RESET/experiments |
| `polity_xconst_validation.py` | Polity executive-constraints → institutional k_eff | RATCHET-RESET/experiments |
| `vdem_presidentialism_validation.py` | V-Dem presidentialism collapse prediction | RATCHET-RESET/experiments |
| `wgi_polity_validation.py` | WGI / Polity cross-source institutional validation | RATCHET-RESET/experiments |
| `wgi_empirical_validation.py` | WGI rule-of-law empirical validation | RATCHET-RESET/experiments |
| `run_all_validations.py` | Master runner (all of the above) | RATCHET-RESET/experiments |

### Pre-computed results (the actual numbers cited in the CCA paper)

| File | Content |
|---|---|
| `results/nasa_battery_groups.json` | Per-group k, ρ, k_eff for 9 NASA battery cell groups (34 total cells) — the actual k_eff numbers cited in CCA |
| `results/synthetic_and_market_validation.csv` | Synthetic controls (independent, strong-causal, weak-causal, common-cause) + market validations (SPY/QQQ, VIX/SPY, etc.) |
| `results/polity_xconst_validation_results.csv` | Polity executive-constraints validation outcomes |
| `results/vdem_presidentialism_results.csv` | V-Dem presidentialism validation outcomes |
| `results/wgi_validation_results.csv` | WGI validation outcomes |

### Loaders (in `ratchet/data/`)

| Loader | Reads | Returns |
|---|---|---|
| `battery_loader.py` | NASA PCoE `.mat` files | `NASABatteryDataset` with per-cell SOH trajectories + k/ρ/σ aggregates |
| `institutional_loader.py` | QoG / Polity V / V-Dem CSV | Country-year observations with regime-transition labels |
| `microbiome_loader.py` | AGP / HMP abundance tables | Sample-level taxa-abundance with Shannon diversity |

### Engines (in `ratchet/engines/`, already on master)

Same as before — `battery.py`, `institutional.py`, `microbiome.py` simulate dynamics. The loaders + run scripts here connect them to real data.

### Residual analysis (in `analysis/omega/`)

Cherry-picked from `track-c-omega` (commit `aae17e2`). Implements σ_observed − σ_predicted residual (`OmegaObservation`) with null-hypothesis tests, correlation analysis, distribution analysis, outlier detection. **Load-bearing for Exp 2 P2** (residual whiteness × agency rung).

## How to reproduce the CCA Tier-1 numbers

### Battery (NASA Li-ion, 19 cells)
```bash
# Dataset is vendored locally at data/battery/ (gitignored); SHA pinned in
# experiments/exp2_cross_substrate/data_sources.yaml.
python3 tests/test_battery_nasa_comparison.py
# Expected: per-cell SOH RMSE ~8.1%, k=19 ρ=0 k_eff=19 (fresh cells)
```

### Institutional (QoG/Polity, 13 countries)
```bash
python3 experiments/exp0_cca_validation/polity_xconst_validation.py
python3 experiments/exp0_cca_validation/wgi_polity_validation.py
# Expected: 5/5 true-negative stable democracies; Venezuela σ: 0.577→0.211 over 24y
```

### Microbiome (AGP, ~100 samples)
```bash
python3 tests/test_microbiome_real_data.py
# Expected: mean Shannon σ=0.580, mean ρ=0.19, mean k_eff=5.11
```

### ρ_critical = 0.43 universal threshold
```bash
python3 experiments/exp0_cca_validation/run_rho_threshold_validation.py
# Expected: chi-square p < 1e-10 between ρ<0.43 and ρ>0.43 collapse rates
```

## Data availability

| Substrate | Status |
|---|---|
| NASA battery zip (209 MB) | Symlinked from `~/RATCHET-RESET/data/battery/` to `data/battery/`. SHA: `82302a7db4fc1b34e0b6676326610438d43b816bdf11a69d1d012a464ef2f92e` |
| QoG / Polity V CSV | Not vendored; loaders point to expected paths. Fetch via `data/pipeline/fetchers/vdem.py` for V-Dem. |
| AGP microbiome data | Not vendored; loader supports synthetic fallback for testing |

## Relationship to Exp 2

Exp 2 (substrate fractality across agency rungs) extends this Tier-1 rig to four new substrates (AlphaFold, PMU, Allen Neuropixels, BioTIME). The pattern proven here — loader → engine → comparison → metric — is what each Exp 2 substrate's engine should mirror.

The pre-computed Tier-1 results in `results/` are the baseline against which the `substrate_revalidation.yml` quarterly CI compares.

## Provenance

| Source | Commits / paths brought to master |
|---|---|
| `track-b-data` (commit `73999c8`) | `ratchet/data/*_loader.py` + `tests/test_*_nasa_*.py` + `tests/test_*_real_data.py` + `simulation/{test,analyze}_simulation_engines.py` + `scripts/test_institutional_collapse.py` + `simulation_{biology,chemistry,history}/*.md` |
| `track-b-data` (commit `bb51402`) | `data/pipeline/**` (FRED/FAOSTAT/GDELT/V-Dem/IUCN/COMTRADE/OpenAlex fetchers + base + SQLite cache + temporal alignment) |
| `track-c-omega` (commit `aae17e2`) | `analysis/omega/**` |
| `~/RATCHET-RESET` (working tree) | Run scripts in this directory + symlinked NASA archive |

Excluded from this uplift:
- Chronometry research direction (`ratchet/chronometry/`, `formal/chronometry/`, `FSD/chronometry_*`) — separate workstream
- Reviewer-response simulation work (`simulation/reviewer_response_simulation.py`) — paper-specific
