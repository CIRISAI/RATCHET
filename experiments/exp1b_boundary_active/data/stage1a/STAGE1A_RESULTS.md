# Exp 1b Stage 1a — Boundary-Active Subset Re-Analysis

**Status:** EXPLORATORY (per Exp 1 PRE_REGISTRATION.md §10.1).
**Source:** existing Phase 1 traces from run `25935989178`.
**Reference:** `RATCHET.Experiments.BoundaryObservability` BO-1..BO-4.

## Per-model boundary-active subset

| Model | n_total | n_active | active_frac | Full N_eff_H | Active N_eff_H | Active 95% CI | Δ (active − full) | Active CI fits [6.6, 7.6]? |
|---|---|---|---|---|---|---|---|---|
| `qwen/qwen3.5-35b-a3b` | 55 | 41 | 74.55% | 4.883 | 4.429 | [3.393, 4.630] | -0.454 | ✗ |
| `anthropic/claude-opus-4.7` | 0 | 0 | — | — | — | — | — | ✗ |
| `openai/gpt-5.5` | 137 | 76 | 55.47% | 5.493 | 5.616 | [4.997, 5.890] | +0.123 | ✗ |
| `google/gemini-2.5-flash` | 84 | 46 | 54.76% | 5.617 | 5.354 | [4.345, 5.543] | -0.263 | ✗ |
| `meta-llama/llama-4-scout` | 225 | 125 | 55.56% | 7.002 | 5.963 | [4.731, 6.324] | -1.039 | ✗ |

## Interpretation reminders

This re-analysis is EXPLORATORY. The locked Exp 1 §10.1 decision rule remains INDETERMINATE for the original 5-cell sweep (Opus n=0 cell abort). Stage 1a recovers signal informally to guide Phase 1b design but does NOT apply the F-6 decision rule.

Per `BoundaryObservability` BO-2/BO-3: chains where conscience faculties did NOT fire carry no information about the 7.1 anchor. The Δ (active − full) column is the conditional-vs-marginal shift; positive values indicate the boundary-active subset clusters closer to the anchor than the full corpus, consistent with the per-chain-conditional reading of the CRC paper's anchor.
