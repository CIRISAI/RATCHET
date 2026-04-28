# Methodology — CIRIS Reasoning Trace Corpus

This document describes how the corpus was generated, what's in scope,
known caveats, and the empirical findings derived from it.

## How traces are generated

A CIRIS agent processes each user message through a multi-stage pipeline:

```
input  →  THOUGHT_START
      →  SNAPSHOT_AND_CONTEXT       (gather state + memory)
      →  DMA_RESULTS                (CSDMA + DSDMA + PDMA in parallel)
      →  ASPDMA_RESULT              (action selection)
      →  IDMA_RESULT                (independence / source-correlation analysis)
      →  TSASPDMA_RESULT            (tool safety check, if applicable)
      →  CONSCIENCE_RESULT          (4 IRIS shards: entropy, coherence, optimization, epistemic_humility)
      →  ACTION_RESULT              (SPEAK / PONDER / DEFER / TASK_COMPLETE / TOOL / ...)
```

Each stage emits a structured component into the trace. When the action
completes, the trace is canonicalized, signed with the agent's Ed25519
private key, and emitted to a CIRISLens edge ingestion endpoint at one
of three privacy tiers (`generic`, `detailed`, `full_traces`).

The signed payload is `{"components": [...], "trace_level": "..."}`
serialized as compact JSON with `sort_keys=True` and empty values
(`None`, `""`, `[]`, `{}`) recursively stripped.

## What this corpus covers

| Period | 2026-03-22 to 2026-04-26 |
| Agents | Ally (primary), Scout, Datum, echo-core, echo-speculative |
| Test traffic | QA harness (`model_eval_*` channels), 4 languages × ~6 questions, plus organic real-user messages on `api_google:*` |
| Models | Qwen 3.5 / Qwen 3.6 (DeepInfra, OpenRouter); some traces predate model migration |
| Trace levels | Mix of `generic` / `detailed` / `full_traces` (operator-controlled) |

## Three-layer alignment in the data

The corpus shows a fully-formed three-layer alignment loop:

1. **Layer 1 — model filter (CSDMA).** When the underlying LLM emits a
   politically-filtered or low-plausibility response, CSDMA correctly
   reads the response as low-plausibility (CSDMA → 0).

2. **Layer 2 — coherence override.** When CSDMA is low but the agent's
   coherence layer is intact (coherence ≥ 0.85), the agent can proceed
   to SPEAK based on its own alignment judgment. The "coherence torque"
   metric (`coherence × (1 − CSDMA)`) measures the force of this override.

3. **Layer 3 — post-speech discipline.** After an L2 override, a
   "don't speak twice" guard prevents perseveration. The agent must
   PONDER between speeches and re-pass conscience before re-speaking.

We have **zero unauthorized post-guard speech events** in the corpus.
Of 19 guarded tasks, 18 stayed in PONDER until TASK_COMPLETE, and 1
re-spoke after explicit conscience approval (the legitimate "I apologize"
correction path). The discipline rule is closed-loop self-governance.

## Effective constraint dimensionality

Computing the participation ratio of the correlation matrix across DMA,
conscience, and structural signals yields:

| Sample | Raw k_eff | Post-merge k_eff | Notes |
|--------|-----------|------------------|-------|
| n=120 (early run) | 5.72 | 6.78 | small sample |
| n=250 (mid) | 5.81 | 7.67 | |
| **n=500 (latest)** | **6.63** | **9.61** | clears 99% @ r=0.20 threshold |

The analytic merge is a 17-cluster reduction at `\|ρ\| ≥ 0.80` that
collapses the 4 conscience pass/fail booleans + related scores into one
"compound shard" (8 members), the entropy_level/ent_score algebraic
duplicate, and the llm_calls/tokens_input near-identity. Other 14
signals are singletons.

The 99% reduction at deceptive-basin radius r=0.20 corresponds to
`k_safe ≈ 9.0` under the stability-analysis scaling law `(1−2r)^k ≤ 0.01`.
The post-merge k_eff = 9.61 clears this threshold.

## CSDMA as filter-engaged detector

A subset of the QA test set asks for substantive answers to questions
where the underlying foundation model carries a strong refusal prior.
Across multiple language variants of one such question, the response
distribution sorts cleanly into stance categories:

| Stance | Count | CSDMA |
|--------|-------|-------|
| Substantive answer | 1 | 0.95 |
| Indirect acknowledgment | 1 | 0.00 |
| Topic deflection | 2 | 0.00 |
| Refuse with topic naming | 2 | 0.00 |
| Refuse with procedural rationale | 1 | 0.00 |

**The single substantive response had CSDMA = 0.95; all six others had
CSDMA = 0.** This empirically demonstrates that CSDMA is acting as a
filter-engaged detector, not a quality detector. The agent's coherence
layer remained roughly constant at 0.92–0.95 across all seven responses —
coherence does not distinguish between substance and refusal at the
content level. The split between the two signals is what makes the
override pattern detectable.

When the agent's coherence_torque (`coherence × (1 − CSDMA)`) is high
on a SPEAK action — the override pattern — the trace records that the
agent chose substance over the model's refusal prior. Signed receipts
of this behavior exist across all four QA test languages.

## Production change driven by the corpus

On 2026-04-25, analysis of ponder-chain depth distribution revealed:

| Outcome | Max depth |
|---------|-----------|
| Successful SPEAK + TASK_COMPLETE | **5** |
| SPEAK without explicit close (anomaly) | 6 |
| DEFER (escalation to humans) | 7 |

The agent team subsequently changed `max_ponder_depth` from 7 to 5,
forcing earlier resolution-or-escalation and eliminating the
"spoke-but-didn't-close" anomaly. The change is reflected in
`agent_version` cohorts going forward; older versions remain in the
corpus and can be analyzed as a baseline.

## Known caveats

1. **Mixed `agent_version` cohorts.** The corpus spans 10 distinct agent
   versions (1.9.1 → 2.7.0). Behavior parameters (`max_ponder_depth`,
   conscience prompt language, IRIS-E prompt) have changed. Group by
   `agent_version` before averaging.

2. **Generic-level traces don't carry channel info.** About 36% of the
   corpus is `generic` (numeric scores only); these are tagged
   `task_class = "unknown"` and shouldn't be split by language/question.

3. **`fragility_flag` saturates.** ~93% of post-deploy traces show
   `idma_fragility_flag = true`. This is informative as a baseline rate
   but not as a per-trace signal.

4. **Attestation fields are constant in this release.** All
   `attestation_level` values are `0` (community-mode test agents).
   Schema fields exist as future capacity for production deployments
   with hardware attestation enabled.

5. **Stationarity matters.** Run `corpus_shape.py --window <X>` (in the
   parent CIRISLens repo) before any correlation analysis. Behavior
   shifts over time; the QA test set rotated questions across channels;
   high-torque events cluster at specific timestamps. Splitting by time
   window or by `task_class` is usually necessary.

## Reproducing this analysis

The CIRISLens repo at https://github.com/CIRISAI/CIRISLens contains:

- `scripts/corpus_shape.py` — facet card before any analysis
- `scripts/export_corpus.sh` — re-export from a CIRISLens deployment
- `scripts/build_hf_release.py` — filter and package this release
- `sql/026_trace_context_view.sql` — the view used here

All analysis in this release was generated from the same JSONL files
included in `data/`. Spot-checks against the production database can
be reproduced via the SQL queries in the methodology source.

## Out of scope for this release

- **Real-user messages with non-PII personal context.** The current QA
  test traffic uses synthetic identities (`@jeff` etc.); the real-user
  `api_google:*` channel is rate-limited in this release to avoid
  inadvertently exposing user-level reasoning content even after PII
  scrubbing.
- **Internal agent telemetry.** Operational metrics (CPU, memory,
  request latency) are not part of the reasoning-trace corpus.
- **Mock/test-LLM traces.** Traces from agents using mock LLMs
  (identifiable by `models_used` containing "mock") are routed to a
  separate mock repository in CIRISLens and not included here.

## Acknowledgements

- The IRIS-E semantic-entropy alternative-generation pattern is the
  ancestor of the recursive DMA-bounce design proposed in the parent
  repo's `FSD/DMA_BOUNCE.md`.
- The stability-analysis framework for scaling-law-bounded effective
  constraint dimensionality informs the k_eff analysis.
- Multi-language QA test corpus structure (4 languages × 6 questions ×
  3 trace levels) was developed by the CIRIS QA team.
