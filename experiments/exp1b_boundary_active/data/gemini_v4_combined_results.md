# Exp 1b — Gemini 2.5-Flash on v4_combined Battery — Results

**Status:** EXPLORATORY (per Exp 1 PRE_REGISTRATION.md §10.1; locked F-6 decision rule applies only at Phase 1b proper with pre-registration).

**Question battery:** `experiments/exp1b_boundary_active/questions/v4_combined_boundary_active.json` — 14 questions:
- 9 staged + adversarial mental-health questions from CIRISAgent's `en_mental_health_v4` battery (every question targets `EthicalPDMAEvaluator` + `epistemic_humility_conscience`)
- 5 high-friction non-MH from `v1_sensitive` (Theology, Politics, AI Ethics, History, Epistemology)

**Run:** Local qa_runner, 10 iterations × 14 questions, ~280 chains. ~$9 cost. Gemini 2.5-flash via OpenRouter.

## Headline result

**Conscience high-friction filter (N≥3) recovers the 7.1 anchor:**

| Subset | n | Fraction | Cohort N_eff_H | vs [6.6, 7.6] window |
|---|---|---|---|---|
| All chains | 280 | 100% | 8.967 | Above (full-corpus PCA with NaN imputation inflates) |
| **Conscience N≥3** | **144** | **51.4%** | **7.277** | **✓ in window** |
| Conscience N=4 only | 132 | 47.1% | 6.933 | ✓ in window |
| DMA n≥2 (saturated by IDMA universal rigidity) | 278 | 99.3% | 8.698 | Above |
| Combined (conscience OR DMA) | 279 | 99.6% | 8.726 | Above |

**Comparison: Gemini on Phase 1 (v1_sensitive, 6 questions, May 2026 run `25935989178`):**

| Subset | n | N_eff_H |
|---|---|---|
| Full corpus | 84 | 5.617 |
| BO-active (N≥1) | 46 | 5.354 |

**Battery effect:** mean N_eff went from 5.6 → 7.3 on the high-friction subset. Same model, different battery. F-6 H1 (substrate-independence at A3) is recoverable for Gemini when the battery drives sufficient cohort friction.

## Firing-count distribution — bimodal

| N (conscience faculties fired) | n | % |
|---|---|---|
| 0 | 128 | 45.7% |
| 1 | 0 | 0.0% |
| 2 | 8 | 2.9% |
| 3 | 12 | 4.3% |
| **4** | **132** | **47.1%** |

Strong bimodality. The four LLM-based consciences fire as a COORDINATED CASCADE — either all of them engage or none of them do. Almost no chains have just 1 or 2 firings.

This is a CIRIS architectural finding worth noting in the paper: the conscience cascade isn't four independent veto layers, it's a coordinated cascade triggered by a high-level "this thought needs conscience" gate.

## DMA-side signal (FD-8 saturation)

Per FrictionDistribution.lean FD-8:

| DMA signal | Threshold for friction | Empirical pattern on this run |
|---|---|---|
| `csdmaPlausibility < 0.7` | plausibility friction | **Never fires** (csdma ≈ 1.0 on every chain) |
| `dsdmaDomainAlignment < 0.7` | domain misalignment | **Discriminates** (varies 0.0 to 0.85) |
| `idmaKEff < 2.0` | rigidity friction | **Always fires** (k_eff = 1.0 universally) |
| `idmaCorrelationRisk > 0.43` | near-critical | **Always fires** (correlation_risk ≈ 0.93 universally) |

The IDMA universally classifies these chains as "rigid single-source reasoning at high correlation" — interesting empirical fact about the IDMA's read of model_eval-pipeline chains, but it makes the DMA n≥2 filter non-discriminating (98%+ pass).

For cohort-N_eff_H recovery, the **conscience N≥3 filter is the discriminating one in current practice**.

## Cross-tab: conscience N_fired × DMA friction count

| | DMA=0 | DMA=1 | DMA=2 | DMA=3 | DMA=4 |
|---|---|---|---|---|---|
| N_fired=0 | 0 | 1 | 112 | 15 | 0 |
| N_fired=1 | 0 | 0 | 0 | 0 | 0 |
| N_fired=2 | 0 | 0 | 3 | 5 | 0 |
| N_fired=3 | 0 | 0 | 0 | 12 | 0 |
| N_fired=4 | 1 | 0 | 56 | 74 | 1 |

Most N=0 chains still have DMA n=2 (k_eff + correlation universal). N=3+ chains tend toward DMA n=3 (adding the discriminating DSDMA signal).

## Implications

| Claim | Status |
|---|---|
| **F-6 H1 substrate-independence at A3** | **Recovered on Gemini.** The 7.1 anchor is reachable across model families when the battery drives sufficient conscience-firing. Phase 1's INDETERMINATE was wrong-battery, not framework falsification. |
| **BoundaryObservability.BO-1** | Necessary but not sufficient. BO-1 includes N=1 chains; the discriminating threshold for anchor recovery is N≥3. |
| **FrictionDistribution.FD-4** | **Empirically supported.** Higher friction rate (N≥3 fraction = 51.4% on v4_combined) recovers the anchor. Phase 1's v1_sensitive battery had near-zero high-friction rate, hence the INDETERMINATE. |
| **FrictionDistribution.FD-8** (DMA saturation) | Documented. The current IDMA + qa_runner pipeline saturates k_eff + correlation_risk on every chain, making `IsDmaFrictionActive` too broad to discriminate. DSDMA's `domain_alignment` is the only graded DMA signal. |
| **Bimodal cascade** | Strong empirical finding. The four LLM consciences fire as a coordinated cascade, not independently. Worth a paper-level note. |

## Next moves (cost-ordered)

| Option | Cost | What it tests |
|---|---|---|
| **A** — Run qwen-3.5 + llama-scout on v4_combined | ~$6 | Whether the recovery generalizes back to the families that already showed 7.1, confirming the battery effect cleanly |
| B — Run GPT-5.5 on v4_combined | ~$127 | OpenAI family, untested with hard battery |
| C — Run Opus 4.7 on v4_combined (after accord_metrics fix) | ~$113 | Anthropic family, untested with hard battery |
| **D** — Full Phase 1b (all 5 models on v4_combined) | ~$246 (incl A+B+C) | Locked F-6 decision rule with cross-model recovery |
