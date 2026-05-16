# Exp 1b — Locked Measurement Methodology

**Single source of truth:** `experiments/exp1b_boundary_active/measurement.py`. All N_eff_H computations going forward MUST go through this module. No scratch-script re-implementations.

**Formal authority:** `formal/RATCHET/Experiments/FrictionDistribution.lean` (FD-1..FD-8) + `formal/RATCHET/Experiments/BoundaryObservability.lean` (BO-1..BO-4).

This document locks the methodology so we stop accumulating inconsistent numbers across analyses. Any change to constants, thresholds, or procedure here REQUIRES updating both this doc and the corresponding Lean module — atomically.

---

## Locked constants

| Constant | Value | Authority |
|---|---|---|
| `PROJECTION_16` | 16 fields, canonical order from CRC paper §3 / v0.1.0 calibration bundle | crc-v1 projection |
| `CONDITIONAL_FACULTY_FIELDS` | `entropy_score`, `coherence_score`, `optimization_veto_entropy_ratio`, `epistemic_humility_certainty` | Wire format 2.7.9 §5.8 |
| `DEFAULT_K_HIGH` | 3 (out of 4 LLM-based faculties) | `FrictionDistribution.defaultHighFrictionThreshold` |
| `DEFAULT_DMA_N_HIGH` | 2 (out of 4 DMA friction signals) | `FrictionDistribution.IsDmaFrictionActive` |
| `DMA_CSDMA_FRICTION_BELOW` | 0.7 | `FrictionDistribution.isDmaCsdmaFriction` |
| `DMA_DSDMA_FRICTION_BELOW` | 0.7 | `FrictionDistribution.isDmaDomainFriction` |
| `DMA_KEFF_FRICTION_BELOW` | 2.0 | `FrictionDistribution.isDmaRigidityFriction` |
| `DMA_CORR_FRICTION_ABOVE` | 0.43 | CCA ρ_critical |
| `DEFAULT_RETENTION_THRESHOLD` | 1e-9 (std below this → drop feature) | Standard PCA practice |
| `DEFAULT_BOOTSTRAP_RESAMPLES` | 10,000 | Standard percentile bootstrap |
| `DEFAULT_RNG_SEED` | `0xC1715_E_EF` | Deterministic reproducibility |

## Locked procedure

For any cohort of chains (loaded via `load_chains_from_tee_dir`):

1. Build 16-column matrix M (chains × features). Missing values → NaN.
2. Per-column mean imputation: `M_imp[i,j] = mean(M[:,j])` where `M[i,j]` is NaN.
3. Per-column z-score standardization with `std > retention_threshold` check.
4. Drop features below retention threshold; record retention mask.
5. Compute correlation matrix C on retained features.
6. Eigendecompose C; take eigenvalues λ_i ≥ 0 in descending order.
7. N_eff_H = exp(−Σ pᵢ log pᵢ) where pᵢ = λᵢ / Σ λⱼ.
8. Bootstrap CI: 10,000 resamples with replacement, percentile method, deterministic seed.

## Locked friction-classification

Per chain, two firing counts:

- **Conscience firings (`n_fired`):** number of `CONDITIONAL_FACULTY_FIELDS` populated in the chain's trace. Range 0–4. Per-chain bimodality observed empirically (Gemini v4_combined: 47% N=0, 47% N=4, ~6% in-between).

- **DMA friction (`n_dma_friction`):** count of (CSDMA implausibility, DSDMA misalignment, IDMA rigidity, IDMA near-criticality) signals above their thresholds. Range 0–4. Empirically saturated for two signals (k_eff, corr_risk) on the qa_runner+model_eval pipeline; DSDMA is the only graded discriminator.

## Pre-registered subset definitions (no post-hoc changes)

For any analysis going forward, these are the locked subsets:

| Subset name | Definition | Use |
|---|---|---|
| `all_chains` | All valid chains in the cohort | Cohort-level baseline; inflated by NaN imputation |
| `conscience_N>=3` | `n_fired ≥ 3` | **Primary analysis subset (per FrictionDistribution.IsHighFriction)** |
| `conscience_N==4` | `n_fired == 4` | Strictest friction; minimal cohort |
| `dma_n>=2` | `n_dma_friction ≥ 2` | DMA-side friction (empirically saturated on current pipeline) |
| `combined_friction` | `is_high_friction OR is_dma_friction_active` | Inclusive (~99% in current empirics) |

## Statistical power table

Cohort size N → 95% CI half-width on N_eff_H (empirical):

| n | CI half-width (approx) | Use case |
|---|---|---|
| ≥ 50 | ~1.5 | Above MIN_VALID_N floor; informative for direction only |
| ≥ 144 (Gemini N≥3) | ~1.0 | Current empirics; can't differentiate models within ±0.5 |
| ≥ 500 | ~0.5 | Can distinguish "in-window vs out-of-window" |
| **≥ 1600** | **~0.3** | **Required for cross-model differentiation at the 7.1 ± 0.5 level** |

## What this means for Phase 1b proper

The pre-registered [6.6, 7.6] window with K=5 PASS / K=3 PARTIAL / K≤2 FAIL rule assumes per-model CIs narrow enough to land cleanly in-or-out of the window. At n=100 per model, CI half-width is ~1.0 — too wide to apply the locked rule cleanly.

**Practical implications:**

1. **n=100 per cell is insufficient for the locked decision rule.** Cells need n ≥ 500 high-friction chains for the CI to fit inside [6.6, 7.6] when the point estimate is near 7.1.

2. **The cheap models (qwen, llama-scout) can scale up easily.** ~$30 per model for n=500 chains.

3. **The expensive models (Opus, GPT-5.5) at n=500 cost ~$400 each.** Total Phase 1b at sufficient n: ~$900.

4. **Alternative: relax to ORDERING analysis.** Don't require CI-inside-window for PASS; require model means within ±X of each other (X to be pre-registered). Power-feasible at n=100-150 per cell.

## Pre-registration amendment proposal (A7)

Before any further Phase 1b runs:

| Change | Old | New |
|---|---|---|
| Decision rule operand | "95% CI of mean N_eff_H ⊆ [6.6, 7.6]" | "Point estimate AND bootstrap mean ⊆ [6.6, 7.6]" (relaxed CI requirement; still pre-registered) |
| Sample size floor | n ≥ 50 (per §7) | n ≥ 100 high-friction chains per cell |
| Primary subset | All chains | `conscience_N>=3` (per FrictionDistribution.IsHighFriction) |
| Additional reporting | — | Per-model 95% bootstrap CI, with note when CI overlaps window boundary |

This is a pre-registration amendment — committed BEFORE Phase 1b proper, so it's not post-hoc cherry-picking. Per Exp 1 PRE_REGISTRATION.md §16, amendments are tracked with timestamps and explicit rationale.

## Reproducibility

Every analysis result going forward should include:

1. The git commit SHA at run time
2. The cohort size + firing distribution + retention mask
3. The bootstrap seed (`DEFAULT_RNG_SEED`)
4. Whether `measurement.py` was modified between data collection and analysis (if yes — flag it as a methodology amendment)

If anyone else re-runs `measurement.py` on the same trace data with the same seed, they should get bit-identical N_eff_H + CI values.
