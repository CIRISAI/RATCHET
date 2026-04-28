# Single-Agent Trace Pendulum Viewer

**Version:** 0.1  
**Status:** Draft  
**Date:** 2026-04-22

## 1. Overview

This document specifies a repo-accurate trace visualization feature for CIRISLens that renders a single agent's reasoning traces as a "chaotic pendulum" through the stored decision pipeline.

The design is intentionally grounded in the data CIRISLens actually stores today:

- Primary table: `cirislens.accord_traces`
- Backward-compatible view: `cirislens.covenant_traces`
- Trace levels: `detailed` and `full_traces` only
- One agent at a time
- Default selected agent name: `Ally`

This viewer is not a claim that CIRISLens currently implements a formal Fréchet-space or manifold engine in production. It is a defensive, expressive visualization layer over the existing trace schema, step timestamps, denormalized scores, and JSONB component payloads.

## 2. Goals

- Show one agent's trace activity at a time with zero multi-agent overlays.
- Default the viewer to `Ally` without silently substituting another agent.
- Restrict the viewer to `detailed` and `full_traces`, since `generic` lacks the identifiers and context needed for a coherent single-agent trace view.
- Render the stored decision pipeline as a pendulum-like path that feels unstable or damped depending on the trace's own fields.
- Use the actual stored event/component model:
  - `THOUGHT_START`
  - `SNAPSHOT_AND_CONTEXT`
  - `DMA_RESULTS`
  - `ASPDMA_RESULT`
  - optional `IDMA_RESULT`
  - optional `TSASPDMA_RESULT`
  - `CONSCIENCE_RESULT`
  - `ACTION_RESULT`
- Be defensive against incomplete traces, missing timestamps, absent optional events, scrubbed text, malformed JSONB, duplicate agent names, and large payloads.

## 3. Non-Goals

- No cross-agent comparison.
- No anonymous public viewer.
- No reliance on `generic` traces.
- No claim that the viewer is a formal chaos simulation.
- No claim that the viewer computes Fréchet distance, manifold volume, or topological collapse in the production request path.
- No Astro, React, or separate frontend build system requirement.

## 4. Current Repo Ground Truth

### 4.1 Canonical Storage

The viewer must use `cirislens.accord_traces` as the canonical source, not a speculative edge schema.

Relevant fields already exist in the repository:

- Root identifiers:
  - `trace_id`
  - `thought_id`
  - `task_id`
  - `agent_id_hash`
  - `agent_name`
  - `trace_level`
  - `schema_version`
- Denormalized scores and decisions:
  - `csdma_plausibility_score`
  - `dsdma_domain_alignment`
  - `dsdma_domain`
  - `pdma_stakeholders`
  - `pdma_conflicts`
  - `idma_k_eff`
  - `idma_correlation_risk`
  - `idma_fragility_flag`
  - `idma_phase`
  - `selected_action`
  - `action_success`
  - `conscience_passed`
  - `action_was_overridden`
  - `entropy_level`
  - `coherence_level`
  - `entropy_passed`
  - `coherence_passed`
  - `optimization_veto_passed`
  - `epistemic_humility_passed`
- Resource and execution fields:
  - `tokens_total`
  - `cost_cents`
  - `models_used`
  - `execution_time_ms`
  - `has_execution_error`
  - `has_positive_moment`
- JSONB components:
  - `thought_start`
  - `snapshot_and_context`
  - `dma_results`
  - `aspdma_result`
  - `conscience_result`
  - `action_result`
  - optional `idma_result`
  - optional `tsaspdma_result`
- Step timestamps:
  - `thought_start_at`
  - `snapshot_at`
  - `dma_results_at`
  - `aspdma_at`
  - `idma_at`
  - `tsaspdma_at`
  - `conscience_at`
  - `action_result_at`
- Observation-weight fields:
  - `memory_count`
  - `context_tokens`
  - `conversation_turns`
  - `alternatives_considered`
  - `conscience_checks_count`

### 4.2 Schema Reality

The viewer must respect current schema-version behavior:

- `v1.8`, `v1.9`, `v1.9.1`
  - `IDMA` is nested inside `DMA_RESULTS`
  - there is no separate `idma_at`
- `v1.9.3`
  - `IDMA_RESULT` can appear as a separate stored event
  - `TSASPDMA_RESULT` may appear for tool actions

The UI may use the label "H3ERE pipeline" in explanatory copy if desired, but the rendering logic must be driven by the stored event/component model above, not by invented phases.

## 5. Access Model

This viewer is for authenticated internal or scoped partner use, not public sample browsing.

Requirements:

- Reject `access_level=public`.
- Only return traces for a single resolved agent identity.
- Only return traces where `trace_level IN ('detailed', 'full_traces')`.
- Preserve existing access scoping rules from `api/accord_api.py` rather than inventing a second authorization system.

## 6. Single-Agent Resolution

The UI default is:

- `agent_name = "Ally"`

But the backend must not assume that `agent_name` uniquely identifies an agent forever.

Resolution rules:

1. Initial selection is by `agent_name`, default `Ally`.
2. Backend resolves that name to exactly one `agent_id_hash`.
3. If no matching agent exists, return an empty result with `agent_not_found = true`.
4. If multiple `agent_id_hash` values exist for the same `agent_name`, return a conflict payload and require explicit disambiguation.
5. After resolution, all trace queries use the resolved `agent_id_hash`.
6. Never silently merge multiple hashes under one display name.
7. Never silently fall back from `Ally` to another agent.

## 7. Trace Eligibility

The pendulum viewer only uses traces that meet all of the following:

- `trace_level IN ('detailed', 'full_traces')`
- `signature_verified = TRUE`
- `agent_name` and `agent_id_hash` resolved to one agent
- at least one of the core pipeline components is present

Preferred traces:

- `schema_version IN ('1.9.1', '1.9.3')`

Supported with graceful degradation:

- `1.8`
- `1.9`

Rejected by default:

- `generic`
- malformed traces
- mock traces, unless a future explicit debug mode is added

## 8. Visual Model: "Chaotic Pendulum"

### 8.1 Intent

The viewer should feel like a pendulum moving through successive decision gates:

- smooth when the trace is direct, low-friction, and coherent
- visibly unstable when the trace encounters fragility, overrides, recursion, tool gating, timing spikes, or execution failure

This is a visual metaphor, not a physics claim.

### 8.2 Canonical Gate Order

The pendulum path uses this canonical order:

1. `THOUGHT_START`
2. `SNAPSHOT_AND_CONTEXT`
3. `DMA_RESULTS`
4. `ASPDMA_RESULT`
5. `IDMA_RESULT` if present as a separate event
6. `TSASPDMA_RESULT` if present
7. `CONSCIENCE_RESULT`
8. `ACTION_RESULT`

Rules:

- For `v1.9.3`, `IDMA_RESULT` and `TSASPDMA_RESULT` are distinct gates.
- For earlier schemas, `IDMA` is rendered as a nested weight attached to `DMA_RESULTS`, not as a fabricated standalone timestamped event.
- If `TSASPDMA_RESULT` is absent, the viewer keeps a ghosted optional gate, not a fake observed step.

### 8.3 Derived Pendulum Signals

The frontend or backend may derive three values per visible step:

- `phase_index`
  - canonical step order
- `elapsed_ms`
  - derived from step timestamps when present
  - otherwise `null`
- `instability`
  - normalized 0.0-1.0 score derived from existing stored fields

Recommended instability inputs:

- low `csdma_plausibility_score`
- low `dsdma_domain_alignment`
- low `coherence_level`
- `idma_fragility_flag = true`
- low `idma_k_eff`
- `action_was_overridden = true`
- `is_recursive = true`
- `tsaspdma_approved = false`
- high `execution_time_ms`
- `has_execution_error = true`
- `action_success = false`

Recommended aggregate:

```text
instability =
  0.18 * (1 - plausibility)
+ 0.18 * (1 - alignment)
+ 0.18 * (1 - coherence)
+ 0.12 * fragility
+ 0.10 * clamp((2 - k_eff) / 2, 0, 1)
+ 0.10 * overridden
+ 0.07 * recursive
+ 0.04 * execution_error
+ 0.03 * action_failure
```

Notes:

- Missing terms contribute `0`, not failure.
- This score is for rendering only. It is not a compliance verdict.

### 8.4 Motion Rules

The pendulum rendering should be deterministic:

- no `Math.random()`
- no frame-to-frame layout drift
- any jitter must be derived from a stable hash of `trace_id`

Recommended rendering behavior:

- base swing is the canonical gate order
- amplitude expands with `instability`
- damping reduces when the trace is clean and coherent
- timing gaps widen the arc length between gates
- overrides and tool gating create visible kinks
- recursion creates a loop-back echo instead of a separate fake trace

## 9. UI Layout

The viewer should fit the current repo's static/admin model.

Recommended placement:

- authenticated page under `admin/`
- plain HTML + JS
- no Astro requirement

Recommended layout:

- top bar
  - single agent selector
  - default value `Ally`
  - time window
  - trace-level filter fixed to `detailed/full`
- left rail
  - recent tasks or traces for the selected agent
  - no second agent column
- main panel
  - pendulum visualization for the selected trace
- lower detail panel
  - step inspector
  - score strip
  - reasoning/context drawer

Behavior by trace level:

- `detailed`
  - show identifiers, timestamps, numeric metrics, action metadata
  - hide reasoning panes when the underlying text is absent
- `full_traces`
  - show scrubbed reasoning and prompts where available
  - render scrub placeholders literally

## 10. Proposed Backend API

The simplest path is a dedicated endpoint in `api/accord_api.py`, because that module already owns repository RBAC and trace shaping.

### 10.1 Agent Resolution

`GET /api/v1/accord/viewer/pendulum/agent`

Query params:

- `agent_name=Ally`
- `start_time`
- `end_time`

Response:

- resolved `agent_name`
- resolved `agent_id_hash`
- conflict metadata if ambiguous

### 10.2 Trace List

`GET /api/v1/accord/viewer/pendulum/traces`

Query params:

- `agent_id_hash` required after resolution
- `start_time`
- `end_time`
- `limit` default `50`, max `200`

Response fields:

- `trace_id`
- `task_id`
- `timestamp`
- `trace_level`
- `schema_version`
- `selected_action`
- `conscience_passed`
- `action_was_overridden`
- `idma_k_eff`
- `idma_fragility_flag`
- `coherence_level`
- `has_execution_error`

### 10.3 Trace Detail

`GET /api/v1/accord/viewer/pendulum/traces/{trace_id}`

Response fields:

- resolved agent identity
- root trace metadata
- canonical steps with timestamps
- denormalized scores
- observation-weight fields
- component JSONB payloads needed for inspection
- derived pendulum points
- explanatory flags describing any degraded rendering

## 11. Proposed Query Contract

The detail endpoint should query `cirislens.accord_traces`, not the deprecated view, and explicitly constrain:

```sql
WHERE agent_id_hash = $1
  AND trace_level IN ('detailed', 'full_traces')
  AND signature_verified = TRUE
```

Optional time filter:

```sql
AND timestamp BETWEEN $2 AND $3
```

The backend should select denormalized columns first and JSONB second. The viewer must not depend on parsing JSONB when a denormalized field already exists.

## 12. Defensive Rendering Rules

This feature must be defensive as hell.

### 12.1 Data Integrity

- Treat every JSONB field as untrusted input.
- Use denormalized columns as the primary source whenever possible.
- If JSONB parsing fails, continue rendering from columns and mark the step as degraded.
- Never invent timestamps for missing steps.
- Never infer a separate `IDMA_RESULT` event when only nested `DMA_RESULTS.idma` exists.
- Never merge traces from two tasks into one path.

### 12.2 Security

- Escape all text content into the DOM with `textContent`, never raw HTML insertion.
- Do not expose audit signatures, scrub signatures, or raw internal verification errors unless explicitly needed for an admin debug panel.
- Keep public access disabled.
- Respect existing access scoping.
- Log agent-resolution conflicts and oversized requests.

### 12.3 Privacy

- Assume `full_traces` still contain sensitive scrubbed reasoning.
- Show scrubbed placeholders such as `[PERSON_1]` literally.
- Do not attempt client-side de-scrubbing or entity reconstruction.
- Prefer numeric and categorical summaries in the main visualization.

### 12.4 Performance

- Hard cap trace list size.
- Default time window should be bounded, for example last 24 hours or 7 days.
- Animate only the selected trace, not the entire recent-history list.
- Support reduced-motion mode with static rendering.
- Precompute derived pendulum points server-side if client performance becomes unstable.

### 12.5 Failure Modes

If the selected trace is missing fields:

- Missing timestamps
  - render ordered gates without elapsed timing
- Missing optional events
  - show ghosted gates with "not present"
- Missing reasoning text on `detailed`
  - show "reasoning unavailable at this trace level"
- Missing `agent_name`
  - do not attempt to route via UI default; require resolution failure handling
- Multiple hashes for one name
  - block until explicitly disambiguated

## 13. Pendulum-Specific Visual Semantics

To make the pipeline feel like a chaotic pendulum without lying about the data:

- `THOUGHT_START`
  - initial release point
- `SNAPSHOT_AND_CONTEXT`
  - first mass increase from context load
- `DMA_RESULTS`
  - broad lateral spread because multiple DMAs contribute to tension
- `IDMA_RESULT`
  - narrow, heavy sub-weight when present separately
- `ASPDMA_RESULT`
  - directional commit
- `TSASPDMA_RESULT`
  - hard deflection for tool gating
- `CONSCIENCE_RESULT`
  - strongest braking or rebound point
- `ACTION_RESULT`
  - terminal settle or snap

Visual cues:

- green-blue path
  - stable, coherent, unforced execution
- amber bend
  - fragility, uncertainty, or extra evaluation load
- red kink
  - override, execution failure, or severe instability
- ghost ring
  - optional step not present for this schema or action

## 14. Suggested Implementation Order

### Phase 1

- Add single-agent resolution endpoint
- Add detailed trace-detail endpoint with derived pendulum points
- Add static authenticated page under `admin/`
- Support `detailed` and `full_traces`
- Default to `Ally`

### Phase 2

- Add task-level ghost overlays for sibling traces within the same `task_id`
- Add reduced-motion toggle
- Add schema-version badges and degradation notices

### Phase 3

- Add optional live refresh for one selected agent
- Keep websocket support single-agent only
- Do not expand to multi-agent compare until single-agent ambiguity, performance, and access controls are fully settled

## 15. Explicit Rejections

The implementation must explicitly reject the following design mistakes:

- building against a fictional `h3ere_spans` payload
- requiring Astro or a JS bundler
- treating `generic` traces as pendulum-view eligible
- showing multiple agents in the same orbit view
- auto-falling back from `Ally` to "whatever exists"
- calling the visualization a mathematical proof of alignment

## 16. Acceptance Criteria

- Viewer opens on a single agent selector defaulted to `Ally`.
- If `Ally` has no matching traces, the UI shows an empty state and stays on `Ally`.
- Only `detailed` and `full_traces` are returned.
- Only one resolved `agent_id_hash` is active in a session.
- Selected traces render the stored pipeline steps in canonical order.
- Optional `IDMA_RESULT` and `TSASPDMA_RESULT` are handled without fabrication.
- Full-trace reasoning is shown only when present and scrubbed.
- The pendulum path is deterministic for a given trace.
- Missing fields degrade cleanly without breaking the page.

