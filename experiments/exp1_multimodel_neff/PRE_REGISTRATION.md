# Exp 1 — Multi-Model $N_{\text{eff}}$ Stability Test: Pre-Registration

**Pre-registration date:** 2026-05-15 (commit timestamp is canonical).
**Author:** Eric Moore, CIRIS Ethical AI.
**Paper:** Coherence as the Substrate-Independent Structure of Reality (paper #4), §10 Exp 1.
**Falsification handle:** F-6 (per fourth-paper §9).
**Formal predictions:** `formal/RATCHET/Experiments/Exp1Predictions.lean` (Lean 4, committed alongside).
**Budget:** $300 (OpenRouter, key at `~/.ratchet_openrouter_key`).
**Pre-registration mechanism:** Git commit timestamp on this file + the Lean file. The cryptographic-temporal anchor: the commit hash that introduces this file precedes any data collection.

---

## 1 — Title

Multi-model $N_{\text{eff}}$ stability under identical CIRIS constraint topology.

## 1.5 — Phase structure

This experiment has three phases. **Phase 0 is harness-validation, not data collection** — it catches surprises (reasoning-mode defaults, API behavior anomalies, trace-flow failures) before any data counts toward the headline decision. **The pre-registration commit must occur after Phase 0 completes cleanly, before Phase 1 runs.**

| Phase | What it is | Counts toward F-6 decision? | Cost cap |
|---|---|---|---|
| **Phase 0** — Harness validation | 2 questions × 5 models = 10 chains. Verify: (a) each model returns clean completions via OpenRouter; (b) **internal model reasoning is disabled** (no separate reasoning-content field returned, no token-count anomaly indicating hidden CoT); (c) lens receives traces; (d) local-tee captures batches; (e) per-model $N_{\text{eff}}$ computes to a value (any value — Phase 0 doesn't test the *value*, just the *pipeline*). | **No.** Catches surprises before the locked sweep. | $5 |
| **Phase 1** — Full sweep | 100 chains × 5 models = 500 chains. Locked per §4 decision rule. | **Yes** — sole data source for F-6 decision. | $235 |
| **Phase 2** — Analysis | Per-registered statistics, decision rule application, paper revision. | Decision-only. | $0 |

**Why disable model-internal reasoning?** Some frontier models (Gemini 2.5 Pro, GPT-5.5-pro, Claude 4.7 with extended-thinking) emit a separate internal-reasoning trace before their visible completion. If left on, this confounds the F-6 test: we'd be comparing "CIRIS constraint topology + Model A's internal CoT" against "CIRIS constraint topology + Model B's internal CoT" — not "CIRIS over model A" vs "CIRIS over model B." For F-6 to test substrate-independence of the CIRIS constraint geometry, the underlying model must be in raw-completion mode; all reasoning structure should originate from CIRIS's DMA + conscience pipeline.

Phase 0 outputs a `VALIDATION_LOG.md` in this directory before the pre-registration commit. If any model shows reasoning-on-by-default behavior that can't be disabled via API params, that model is **substituted** in the pre-registration before commit (e.g., `gemini-2.5-pro` → `gemini-2.5-flash` if needed), with a note in §16 explaining why.

## 2 — Research question

Is $N_{\text{eff}} \approx 7.1$ — the emergence threshold reported in the CRC paper~\cite{moore_crc_2026} on $n=6{,}465$ production traces against a single foundation model (Qwen) — **a structural property of the CIRIS constraint topology**, or **a property of the underlying model's specific calibration**?

The two hypotheses are confounded in the CRC corpus. This experiment disentangles them by running the *same* CIRIS topology over *different* foundation models and measuring whether $N_{\text{eff}}$ converges to the same anchor.

## 3 — Hypotheses (directional)

| ID | Statement |
|---|---|
| **H1 (structural)** | When CIRIS Agent runs with identical constraint topology (same DMA prompts, same conscience faculties, same trace schema, same projection function) across 5 foundation models, the per-model mean $N_{\text{eff}}$ (entropy $H$) lies within $[6.6, 7.6]$ (anchor $7.1 \pm 0.5$) for each model. |
| **H0 (model-specific)** | At least one of the 5 models produces a per-model mean $N_{\text{eff}}$ with 95% bootstrap CI entirely outside $[6.6, 7.6]$. |
| **H_partial** | A subset (3 or 4 of 5) of models hit the anchor while one or two diverge — opens follow-up "what model property predicts CIRIS compatibility." |

## 4 — Falsification handle (F-6 → operational definition)

From paper #4 §9: *"If $N_{\text{eff}}$ fails to stabilize near $7.1$ across diverse foundation models with the same constraint topology, the threshold is CIRIS-specific rather than a universal structural limit."*

**Operational decision rule (PRE-LOCKED):**

| Outcome | Condition | Conclusion |
|---|---|---|
| PASS (H1) | 5/5 models have 95% bootstrap CI on mean $N_{\text{eff}}$ within $[6.6, 7.6]$ | F-6 passes; substrate-independence at LLM-substrate level confirmed |
| PARTIAL (H_partial) | 3 or 4 of 5 models pass; 1–2 diverge | Substrate-independence holds for some model classes; open question on what predicts compatibility |
| FAIL (H0) | ≤ 2 of 5 models pass | F-6 falsified; the 7.1 threshold is model-specific calibration |

This rule is locked. Post-hoc adjustment of $[6.6, 7.6]$ window, "pass" count, or CI method is **not permitted** under this pre-registration. Sensitivity analyses with different windows are explicitly marked **exploratory** in §11 and cannot retroactively reclassify the headline result.

## 5 — Independent variables

| Variable | Levels (locked) |
|---|---|
| `model_id` | 5 levels: `qwen/qwen3.5-35b-a3b`, `anthropic/claude-opus-4.7`, `openai/gpt-5.5`, `google/gemini-2.5-flash`, `meta-llama/llama-4-scout`. **Two substitutions surfaced and locked before pre-registration commit (see `VALIDATION_LOG.md`):** (1) `google/gemini-2.5-pro` → `google/gemini-2.5-flash` because 2.5-pro returns HTTP 400 with body `"Reasoning is mandatory for this endpoint and cannot be disabled"`; 2.5-flash honors `reasoning: {enabled: false}`. (2) `qwen/qwen3.6-35b-a3b` → `qwen/qwen3.5-35b-a3b` because 3.6's reasoning-disable is non-deterministic via OpenRouter routing (1/4 Phase 0 trials produced reasoning_tokens=400 despite `enabled: false`); 3.5-35b-a3b honors the flag deterministically (0/4 reasoning trials in Phase 0 verification). 3.5-35b-a3b preserves the 35B-a3b MoE architecture of the CRC anchor; one generation older. |
| `question_id` | 6 levels (from `~/bounce-test/model_eval_questions/v1_sensitive.json`): Theology, Politics, AI Ethics, History, Epistemology, Mental Health |
| `trial_index` | 1..17 per (model, question) — exact value determined by uniform allocation after exclusions |

CIRIS constraint topology held identical across all runs: same `tools/qa_runner model_eval` invocation, same DMA prompts, same conscience-faculty thresholds, same trace schema (2.7.9), same projection function (`crc-v1` — 16 features per `release/calibration/crc-v1/bundle.yaml`).

## 6 — Dependent variables

| Variable | Computation |
|---|---|
| **Primary: per-model mean $N_{\text{eff}}^{H}$** | Mean of $N_{\text{eff}}$ entropy-perplexity across all valid chains for that model. $N_{\text{eff}}^{H} = \exp(-\sum_i p_i \log p_i)$ where $p_i = \lambda_i / \sum_j \lambda_j$ are normalized eigenvalues of the standardized 16-feature covariance matrix per chain. |
| **Secondary: per-model mean $N_{\text{eff}}^{PR}$** | Participation-ratio variant: $N_{\text{eff}}^{PR} = (\sum_i \lambda_i)^2 / \sum_i \lambda_i^2$. |
| **Secondary: 90% variance horizon (median dims per model)** | Number of dimensions to reach 90% cumulative variance. CRC anchor: 7. |
| **Secondary: 99% variance horizon (median dims per model)** | CRC anchor: 11. |
| **Within-model variance** | $\sigma$ of $N_{\text{eff}}^{H}$ across valid chains within each model. |
| **Cross-model variance** | $\sigma$ of per-model means across the 5 models. |
| **Retention mask** | Per model, count of 16 features retained (corpus-wide $\sigma > 10^{-9}$). |

## 7 — Sample size, power, stopping rule

**Target $n$ per model:** 100 valid chains. **Total:** 500 chains across all 5 models.

**Power analysis:** within-model $\sigma_{N_{\text{eff}}} \approx 1.0$ (from CRC corpus + v0.1.0 calibration corpus). At $n=100$, standard error of the mean is $1.0/\sqrt{100} = 0.10$. 95% CI is approximately $\pm 0.20$. The pre-locked PASS window $[6.6, 7.6]$ has half-width $0.5$; therefore detection of "mean within window" vs "mean outside window" has power $> 0.99$ at $\alpha = 0.05$ for true mean differences $\geq 0.3$.

**Stopping rule:**
- Each model run continues until either 100 valid chains complete OR the budget allocated to that model exhausts.
- **No early stopping** for "favorable" or "unfavorable" trends.
- **No adaptive $n$**: $n$ is the smaller of (100, budget-allowed).
- If any model produces $< 50$ valid chains (catastrophic budget exhaustion or harness failure), the experiment is INDETERMINATE and re-pre-registered before any re-run.

**Budget allocation (locked):**

| Model | Per-chain est. cost | Max budget allocation | Max chains | Target chains |
|---|---|---|---|---|
| `qwen/qwen3.5-35b-a3b` | $0.028 | $5 | 175 | 100 |
| `anthropic/claude-opus-4.7` | $0.81 | $90 | 110 | 100 |
| `openai/gpt-5.5` | $0.91 | $100 | 110 | 100 |
| `google/gemini-2.5-flash` | $0.06 | $10 | 165 | 100 |
| `meta-llama/llama-4-scout` | $0.011 | $5 | 450 | 100 |
| **Total** | | **$210** | | **500** |
| Reserve | | $90 | | (audit + re-run buffer) |

## 8 — Question allocation

The canonical `v1_sensitive.json` has 6 questions. To reach $n=100$ per model with paired structure:

- Each question is run 17 times per model (102 chains total per model; 2 are extras to absorb single-trial failures, leaving $\geq 100$ valid).
- Trials are paired across models by `(question_id, trial_index)` — model A trial 7 of question 3 is paired with model B trial 7 of question 3 etc. **No randomization of trial order** — sequential generation, so trial_index = order-of-generation.
- All 5 models see **identical** prompts and seed structure (where the API exposes a seed parameter). Where API doesn't expose seeds, temperature is held at the CIRIS-default value (locked in the topology config; not varied per model).

## 9 — Randomization & blinding

| Aspect | Treatment |
|---|---|
| Model order | Sequential A→E; no randomization needed for the per-model mean estimate |
| Question order | Fixed per the canonical `v1_sensitive.json` ordering; held identical across models |
| Trial order within (model, question) | Sequential, recorded as `trial_index` |
| Analysis-stage blinding | **Yes**: the analysis script computes per-model means via a hash-keyed lookup; the per-model labels are not added until after the means are computed. (Implemented as a constraint in the analysis notebook.) |
| Researcher access to interim data | None — single-shot runs, no checking-in until all models complete |

## 10 — Pre-registered analyses (the primary endpoint and its decision)

### 10.1 Primary endpoint

For each model $m \in \{1, ..., 5\}$:
1. Collect all valid chains.
2. Compute $N_{\text{eff}}^{H}$ per chain via the standardized PCA pipeline in `scripts/build_calibration_bundle.py` (read mode — no calibration; just compute the spectrum per chain and apply the entropy-perplexity formula).
3. Compute mean $\bar{N}_{\text{eff}}^{H}(m)$ and 95% bootstrap CI (10,000 resamples, percentile method).

**Decision (locked):**
- Count $K$ = number of models with 95% CI fully contained in $[6.6, 7.6]$.
- $K = 5$ → **PASS** (H1 supported).
- $K \in \{3, 4\}$ → **PARTIAL** (H_partial).
- $K \leq 2$ → **FAIL** (H0 / F-6 falsified).

### 10.2 Secondary endpoints (pre-specified, hierarchical)

Reported but not used for headline decision:
1. Per-model $N_{\text{eff}}^{PR}$ — same window check as primary.
2. Per-model 90% variance horizon — anchor 7 dims; report median and IQR.
3. Per-model 99% variance horizon — anchor 11 dims.
4. Cross-model SD of $\bar{N}_{\text{eff}}^{H}$ — anchor: small SD (< 0.5) is consistent with H1; large SD (> 1.0) with H0.
5. Per-question paired comparison: ANOVA on (model × question) with trial as the within-cell repeat.

## 11 — Exploratory (marked NOT pre-registered)

These analyses are explicitly **exploratory** and CANNOT reclassify the headline decision:

- Per-cohort breakdown by `deployment_profile.deployment_domain` (only `general` expected at this scale).
- Sensitivity analysis: pass window $[6.6, 7.6]$ widened/narrowed by $\pm 0.2$.
- Per-question variance decomposition.
- Token cost vs $N_{\text{eff}}$ correlation (cheaper models = lower $N_{\text{eff}}$?).
- Open-weights vs closed-weights subgroup difference.
- Trial-index dynamics (does $N_{\text{eff}}$ drift across trials? — suggests within-conversation memory effects).

## 12 — Outliers and exclusions

**Pre-specified exclusion criteria (before $N_{\text{eff}}$ computation):**

| Criterion | Action |
|---|---|
| Chain failed to emit all 7 core event types (`THOUGHT_START`, `SNAPSHOT_AND_CONTEXT`, `DMA_RESULTS`, `IDMA_RESULT`, `ASPDMA_RESULT`, `CONSCIENCE_RESULT`, `ACTION_RESULT`) | Excluded |
| API error mid-chain (context-overflow, rate-limit, timeout, 5xx) | Excluded |
| `N_{\text{eff}}^{H}$ = NaN (degenerate eigenvalue spectrum on a single chain) | Excluded |
| Chain length < 3 thoughts | Excluded (insufficient signal for 16-dim PCA — verified by reproducing exclusion logic on n=264 v0.1.0 bundle) |

**Post-hoc outlier rules:** none. Per-chain $N_{\text{eff}}^{H}$ values that fall outside any specific range are **kept** unless they hit one of the pre-specified exclusion criteria above. No 3-sigma trimming, no Winsorization, no Cook's-distance removal.

## 13 — What would be surprising (not falsifying — but flagged for investigation)

The following outcomes do not change the headline decision but are flagged for **expanded analysis** in the paper revision:

| Surprise | What it might mean |
|---|---|
| Cross-model SD of $\bar{N}_{\text{eff}}^{H} < 0.05$ | Suspiciously uniform; possible harness bias forcing $N_{\text{eff}}$ to a fixed point |
| Any model with $\bar{N}_{\text{eff}}^{H} > 9$ | Higher than CRC's mature-Ally cluster; investigate whether the model is "over-conforming" |
| Any model with $\bar{N}_{\text{eff}}^{H} < 4$ | Severe degeneration; investigate harness misconfiguration before accepting as a genuine signal |
| Llama-4-scout produces dramatically *higher* $N_{\text{eff}}$ than the larger frontier models | Open-weights model outperforming on coherence; would invert the conventional wisdom |
| Within-model $\sigma$ varies by > 3× across models | Suggests model-specific variability in DMA/conscience faculty firing |

## 14 — Trace provenance and data flow

**Traces flow to production lens via `--live-lens`** — they are consented federation-visible data, identical in provenance shape to the n=264 v0.1.0 calibration corpus. The qa_runner's local-tee feature also writes a copy to disk for in-band analysis.

| Stream | Destination | Provenance role |
|---|---|---|
| **Production lens** | `https://lens.ciris-services-1.ai/lens-api/api/v1` | Authoritative, signed, federation-replicated. Anchors the experiment in the same persist storage that produced the calibration bundle (`PUBLIC_SCHEMA_CONTRACT @ v0.3.2`). |
| **Local tee** | `/tmp/qa-runner-lens-traces-<UTC-iso>/accord-batch-*.json` (auto-enabled by `--live-lens`) | Exact-bytes copy of every batch POSTed to lens. Used for analysis without round-tripping through lens read-API. |

Both streams are exported to the experiment directory after each model run completes:

```
experiments/exp1_multimodel_neff/data/
├── qwen-3.6-35b-a3b/
│   ├── tee_batches/                         # local-tee copy
│   ├── tee_sha256.txt                       # per-batch SHA-256 manifest
│   └── lens_export.jsonl                    # pull-from-lens at end of run (re-fetched to confirm round-trip integrity)
├── claude-opus-4.7/
├── gpt-5.5/
├── gemini-2.5-pro/
└── llama-4-scout/
```

**Integrity check (mandatory before analysis):** the local-tee batches MUST round-trip-match the lens export for each model. Mismatch = the model's data is rejected; that model goes INDETERMINATE for §10.1 decision rule (§7 catastrophic-failure clause).

## 15 — Data and code release

All released under MIT on Zenodo upon paper-revision submission:
- Full per-chain trace data (the local-tee batches + the lens-pulled JSONL)
- Per-chain $N_{\text{eff}}$ computation (Python script, version-pinned)
- Analysis notebook with the pre-registered statistics
- The OSF pre-registration timestamp ID (filed in parallel with this git commit, cross-referenced)

All artifacts referenced by SHA-256 in the paper-revision text.

## 15 — Commit-time pre-registration

This file's git commit timestamp + the commit hash that introduces `formal/RATCHET/Experiments/Exp1Predictions.lean` constitute the canonical pre-registration timestamp. **No data may be collected before that commit lands on the public repo `master` branch.**

After the commit lands, the experiment proceeds to data collection. If any aspect of this document or the Lean file is amended after data collection begins, the amendment commit is documented in §16 (Amendments) below, with full diff visible in git history.

## 16 — Amendments

### A1 — 2026-05-15: Vendor `v1_sensitive.json` for CI reproducibility

**Rationale:** The pre-reg §5 references the questions file at a local-home path (`~/bounce-test/model_eval_questions/v1_sensitive.json`). For CI-runner-based Phase 1 execution (workflow `exp1_phase1.yml`, drafted post-pre-reg), the questions file must be vendored into the repo so the runner can access it. Otherwise CI fails on missing-file.

**Change:** Copy `v1_sensitive.json` to `experiments/exp1_multimodel_neff/questions/v1_sensitive.json`. Pin its SHA-256.

**SHA-256:** `29a2fffb47dcad438fd14174f0ad793c352ecdeff11e621bf575c36c3fd49dbc`

**What this changes:** Storage location only. The 6 categories, ordering, content, and translations are byte-identical to the local-home file used in Phase 0 smoke validation. The hypothesis, decision rule, and projection are unaffected.

**What this does NOT change:** §3 (hypotheses), §4 (decision rule), §5 (the 5 models, 6 categories — categories pinned by name not position), §6 (16-feature projection), §7 (sample size), §10 (analysis plan).

**Operational follow-up:** Phase 1 workflow reads `experiments/exp1_multimodel_neff/questions/v1_sensitive.json`. If the file's SHA-256 ever changes, that's a content amendment requiring its own §16 entry.

### A2 — 2026-05-15: Lock CI inner-loop `--model-eval-concurrency 1`

**Rationale:** Phase 0 smoke ran with `--model-eval-concurrency 1` and captured 18/18 thoughts with no race observed. Phase 1 workflow originally drafted with concurrency=3 to reduce wall time; reverted to 1 to eliminate any in-model race-condition risk on the higher-throughput Phase 1 sweep. Wall-time impact is negligible because matrix parallelism is across the 5 models, not within a model.

**Change:** Workflow line `--model-eval-concurrency 1` (locked).

**What this does NOT change:** None of §3–§14.

### A3 — 2026-05-15: Confirm `CIRIS_DISABLE_TASK_APPEND=1` is auto-set

**Not an amendment per se** — just a recorded confirmation: `CIRISAgent/tools/qa_runner/modules/model_eval_tests.py:SERVER_ENV` already sets `CIRIS_DISABLE_TASK_APPEND=1` whenever the `model_eval` module runs. Each question creates a fresh CIRIS task instead of appending to an in-flight thought. Validated in Phase 0 smoke (n=18 thoughts produced 18 independent traces with no cross-question contamination). The Phase 1 workflow sets this env var explicitly as well, for auditor visibility.
