# Measurement Analysis — A + B + D
Locked methodology: `experiments/exp1b_boundary_active/measurement.py`.
All N_eff_H values computed via the same pipeline. Bootstrap CIs use deterministic seed.

## A — Bootstrap CIs (Gemini-flash, v4_combined battery)

Total chains: 280 (excluded 0)

| Cohort | n | Point N_eff_H | Bootstrap mean | 95% CI | In [6.6, 7.6]? |
|---|---|---|---|---|---|
| All chains | 280 | 8.967 | 8.170 | [6.921, 9.195] | ✗ |
| Conscience N>=3 | 144 | 7.277 | 6.546 | [5.376, 7.492] | ✗ |
| Conscience N==4 | 132 | 6.933 | 6.294 | [5.113, 7.135] | ✗ |

## A — Firing-count distribution

| N | n | % |
|---|---|---|
| 0 | 128 | 45.7% |
| 1 | 0 | 0.0% |
| 2 | 8 | 2.9% |
| 3 | 12 | 4.3% |
| 4 | 132 | 47.1% |

## B — Sensitivity sweep (Gemini-flash, v4_combined)

| Subset | n | N_eff_H | Retained dim |
|---|---|---|---|
| `all_chains` | 280 | 8.967 | 14 |
| `conscience_N>=1` | 152 | 7.988 | 14 |
| `conscience_N>=2` | 152 | 7.988 | 14 |
| `conscience_N>=3` | 144 | 7.277 | 14 |
| `conscience_N>=4` | 132 | 6.933 | 13 |
| `conscience_N==3` | 12 | 2.383 | 10 |
| `conscience_N==4` | 132 | 6.933 | 13 |
| `dma_n>=1` | 279 | 8.873 | 14 |
| `dma_n>=2` | 278 | 8.698 | 14 |
| `dma_n>=3` | 107 | 7.610 | 13 |
| `dma_n>=4` | 1 | — | 0 |
| `combined_friction` | 279 | 8.726 | 14 |
| `conscience_N>=3_rt=1e-12` | 144 | 7.277 | 14 |
| `conscience_N>=3_rt=1e-09` | 144 | 7.277 | 14 |
| `conscience_N>=3_rt=1e-06` | 144 | 7.277 | 14 |
| `conscience_N>=3_rt=1e-03` | 144 | 7.277 | 14 |
| `conscience_N>=3_rt=1e-01` | 144 | 4.008 | 7 |

## D — v0.1.0 calibration bundle comparison

**v0.1.0 raw export not available** at `/tmp/ratchet_v0_1_0_calibration`. This comparison requires the raw export (not the calibrated bundle artifact at `release/calibration/crc-v1/`). Skipping D.

The v0.1.0 calibration bundle itself reports `cohort_neff_h ≈ 7.07` (per `release/calibration/crc-v1/bundle.yaml` headline) — that was computed on the FULL n=264 corpus with imputation. To do a clean comparison we'd need to re-run boundary-active filtering on the v0.1.0 raw traces.

## Headline reading

- Gemini N>=3 point estimate: **7.277** with 95% CI [5.376, 7.492]
- Inside locked [6.6, 7.6] window: **NO**

- N=4 subset point: **6.933** (n=132)
