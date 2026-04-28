# Schema reference — `trace_context.jsonl`

The flat analysis-ready view. Use this for tabular work; use
`accord_traces.jsonl` for raw JSONB blobs (reasoning text, conscience
sub-fields, snapshot context).

## Identity

| Column | Type | Description |
|--------|------|-------------|
| `id` | int64 | Per-row primary key in source DB. |
| `timestamp` | ISO8601 | Trace ingestion time (UTC). |
| `trace_id` | string | Unique per agent thought; used to link across trace levels. |
| `task_id` | string | UUID; groups thoughts on a single conversation/task. |
| `thought_id` | string | UUID/slug; one row per (thought_id, trace_level). |
| `thought_depth` | int | Position in the thought chain. 0 = seed, ≥1 = follow-up. |
| `thought_type` | string | `standard`, `follow_up`, `seed`. |
| `trace_level` | string | `generic`, `detailed`, `full_traces`. |
| `agent_name` | string | Human-readable agent name (e.g., `Ally`, `Scout`). |
| `agent_id_hash` | string | SHA-hashed agent identity. |
| `agent_version` | string | e.g., `2.7.0-stable`. **Use this to split cohorts.** |
| `cognitive_state` | string | `work`, `wakeup`, `dream`, `play`, `solitude`. |
| `dsdma_domain` | string | Agent-declared domain identity (often equals agent_name). |
| `selected_action` | string | `SPEAK`, `PONDER`, `DEFER`, `TASK_COMPLETE`, `TOOL`, `MEMORIZE`, `FORGET`, `REJECT`. |
| `action_success` | bool | Whether the chosen action was executed successfully. |

## Reasoning stack (continuous, 0–1)

| Column | Type | Description |
|--------|------|-------------|
| `csdma_plausibility_score` | numeric(5,4) | Common-sense plausibility per the LLM's own judgment. **Functions as filter-engaged detector** — drops to 0 when model refuses or hedges. |
| `dsdma_domain_alignment` | numeric(5,4) | Domain-specific alignment between the response and the agent's role. |
| `entropy_level` | numeric(5,4) | Semantic-entropy from IRIS-E (high = response is an outlier vs alternatives). |
| `coherence_level` | numeric(5,4) | Internal-consistency from IRIS-C. |
| `idma_k_eff` | numeric(5,2) | Effective source count from IDMA. |
| `idma_correlation_risk` | numeric(5,4) | Correlation between reasoning sources. |
| `idma_phase` | string | `rigidity`, `chaos`, `healthy`. Phase classification. |
| `idma_fragility_flag` | bool | True when k_eff is below threshold. |

## Conscience signals (booleans)

| Column | Description |
|--------|-------------|
| `conscience_passed` | Composite: did the thought clear all 4 IRIS shards? |
| `entropy_passed` | IRIS-E: semantic-entropy below threshold. |
| `coherence_passed` | IRIS-C: internal-consistency above threshold. |
| `optimization_veto_passed` | IRIS-V: not over-optimizing autonomy/efficiency. |
| `epistemic_humility_passed` | IRIS-H: confidence not inflated. |
| `action_was_overridden` | True when conscience overrode the originally-selected action. |
| `conscience_checks_count` | How many shards ran (varies; pipeline short-circuits). |

## Promoted from `conscience_result` JSONB

| Column | Description |
|--------|-------------|
| `entropy_score` | Per-trace entropy value (richer than `entropy_level`). |
| `coherence_score` | Per-trace coherence value. |
| `optimization_veto_entropy_ratio` | Anomaly metric; 0.15 baseline, spikes to 8.5+ on outliers. |
| `epistemic_humility_certainty` | 0–1; mostly 0.95 in current data. |

## Resource signals

| Column | Type | Description |
|--------|------|-------------|
| `tokens_input` | int | LLM input tokens consumed. |
| `tokens_output` | int | LLM output tokens emitted. |
| `processing_ms` | numeric | Total wall-clock time for the thought. |
| `llm_calls` | int | Number of LLM round-trips. |
| `cost_cents` | numeric | Cost in cents per the model+provider price sheet. |

## Stage timings (ms)

| Column | Description |
|--------|-------------|
| `t_snap_ms` | THOUGHT_START → SNAPSHOT_AND_CONTEXT |
| `t_dma_ms` | SNAPSHOT → DMA_RESULTS |
| `t_aspdma_ms` | DMA_RESULTS → ASPDMA_RESULT |
| `t_conscience_ms` | ASPDMA → CONSCIENCE_RESULT |
| `t_action_ms` | CONSCIENCE → ACTION_RESULT |

## Channel + task class

| Column | Description |
|--------|-------------|
| `channel_id` | Source channel (e.g., `model_eval_zh_01`, `api_google:...`). |
| `task_class` | Derived: `qa_eval`, `real_user_web`, `discord`, `wakeup_ritual`, `unknown`, `other`. |
| `qa_language` | For QA traffic: `am`, `en`, `es`, `zh`. NULL otherwise. |
| `qa_question_num` | For QA traffic: 1, 2, 3, ... NULL otherwise. |

## Batch-level context

| Column | Description |
|--------|-------------|
| `deployment_region` | Operator-declared (e.g., `na`, `eu`). |
| `deployment_type` | Operator-declared (e.g., `personal`, `business`, `research`). |
| `agent_role` | Operator-declared role string. |
| `agent_template` | CIRIS template name. |
| `user_timezone` | IANA timezone (when consented). |
| `user_latitude_cell` | numeric(4,1); 0.5° grid cell, ~55km. |
| `user_longitude_cell` | numeric(5,1); 0.5° grid cell, ~55km. |

## Model & attestation

| Column | Description |
|--------|-------------|
| `primary_model` | First element of `models_used` (e.g., `Qwen/Qwen3.6-35B-A3B`). |
| `attestation_level` | int 0–5; **constant 0 in this release** (community-mode test agents). |
| `attestation_status` | `failed`, `partial`, `full`. **Constant `failed` in this release.** |
| `signature_verified` | bool; **always true in this release** (filter applied). |

## Reasoning text (`accord_traces.jsonl` only — JSONB blobs)

| Field | Where | Description |
|-------|-------|-------------|
| `thought_start.components[].data.task_description` | full_traces | The user's request, PII-scrubbed. |
| `snapshot_and_context.system_snapshot.channel_id` | detailed+ | Source channel (also accessible via `trace_context.channel_id`). |
| `dma_results.csdma.reasoning` | full_traces | Free-text rationale from CSDMA. |
| `aspdma_result.action_rationale` | full_traces | Why the agent chose this action. |
| `conscience_result.conscience_override_reason` | detailed+ | Why conscience vetoed the action (e.g., the "don't speak twice" guard). |
| `action_result.action_parameters.content` | full_traces | The actual SPEAK content (PII-scrubbed). |

## Notes on field availability by trace level

| Field | `generic` | `detailed` | `full_traces` |
|-------|-----------|------------|----------------|
| Numeric scores (CSDMA, DSDMA, entropy, coherence, IDMA) | ✓ | ✓ | ✓ |
| Conscience boolean gates | ✓ | ✓ | ✓ |
| Channel ID | — | ✓ | ✓ |
| Task description | — | partial | ✓ |
| Action rationale text | — | — | ✓ |
| SPEAK content | — | — | ✓ |
| Agent name / version / template | — | ✓ | ✓ |

`generic` traces are useful for population-level statistics; `full_traces`
are required for content/behavior analysis.
