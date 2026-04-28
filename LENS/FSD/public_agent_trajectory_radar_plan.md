# Public Agent Trajectory Radar Plan

**Status:** Draft plan  
**Date:** 2026-04-23  
**Related FSD:** `FSD/single_agent_trace_pendulum_viewer.md`

## Intent

Build a public-safe, live radar view of an agent's trajectory through CIRIS ethical space. The view should help an operator see where an agent is moving across plausibility, domain alignment, coherence, fragility, conscience, and action outcome without exposing raw trace payloads, private prompts, user content, signatures, or internal verification details.

This differs from the current single-agent pendulum FSD in one major way: the FSD rejects public access, while this plan targets a public-safe operator radar. The implementation should preserve the FSD's data discipline while changing the exposure model to a redacted projection.

## Product Shape

- Public or embeddable radar page for an operator-facing agent view.
- One active agent at a time, defaulting to `Ally` where that agent is available.
- Live trajectory through ethical space using recent trace points.
- No multi-agent overlay in the primary radar.
- Optional authenticated drilldown can exist later, but the public radar itself should stay redacted.
- The radar should be embeddable in an agent/operator console as a compact live panel.

## Public Data Boundary

The public endpoint must expose only derived, non-content fields:

- Stable public agent slug, not raw `agent_id_hash`.
- Trace timestamp bucket or exact timestamp only if acceptable for operations visibility.
- `trace_id` should be omitted or replaced by a short public event id.
- `task_id` and `thought_id` should be omitted by default.
- Scores:
  - `csdma_plausibility_score`
  - `dsdma_domain_alignment`
  - `coherence_level`
  - normalized `idma_k_eff`
  - `idma_fragility_flag`
  - `conscience_passed`
  - `action_was_overridden`
  - `action_success`
  - `has_execution_error`
- Categorical action group, not necessarily raw `selected_action` if action names become sensitive.
- Schema/degradation flags only when they do not reveal internal failure details.
- No JSONB component payloads.
- No prompts, reasoning, scrubbed text, signatures, raw errors, or request bodies.

## Backend Plan

1. Add a public-safe projection endpoint under the existing API service.
   - Candidate path: `GET /api/v1/accord/public/radar/agents/{agent_slug}/trajectory`
   - Query params: `window`, `limit`, `since`, optional `poll_after`.
   - Default window: last 24 hours.
   - Hard cap: 200 points.

2. Add an agent slug mapping.
   - Keep raw `agent_id_hash` internal.
   - Use explicit configured slugs such as `ally`, `scout`, or `datum`.
   - Do not resolve arbitrary public `agent_name` strings directly to hashes.
   - If public exposure should be disabled for an agent, return 404.

3. Query `cirislens.accord_traces`.
   - Keep `signature_verified = TRUE`.
   - Keep `trace_level IN ('detailed', 'full_traces')` unless a separate aggregate/public trace level is added.
   - Select denormalized fields first.
   - Avoid selecting component JSONB in the public query.

4. Derive radar points server-side.
   - `x/y/z` or polar coordinates from plausibility, alignment, and coherence.
   - `instability` from the FSD formula.
   - `phase` from the canonical gate model when step timestamps exist.
   - `age_ms`, `elapsed_ms`, and degradation flags.
   - Deterministic jitter from a stable hash if visual separation is needed.

5. Add cache and rate limits.
   - Cache public responses for a short TTL, for example 2-10 seconds.
   - Rate-limit by IP or route class.
   - Avoid returning large historical windows from the public endpoint.

## Frontend Plan

1. Add a static public radar page.
   - Candidate path: `static/agent-radar.html` or `admin/radar.html` if initially gated.
   - No bundler requirement.
   - Canvas or SVG rendering is enough for the first version.

2. Primary display.
   - Radar/trajectory plot with axes for plausibility, alignment, coherence, fragility, conscience, and outcome.
   - Latest point emphasized.
   - Recent path tail fades with age.
   - Amber/red bends for fragility, overrides, execution errors, or failed actions.
   - Reduced-motion mode renders a static latest trajectory.

3. Operator panel.
   - Agent slug/name.
   - Current stability/instability.
   - Latest action category.
   - Latest conscience/action state.
   - Point count and data freshness.
   - Clear empty state when no public-safe traces exist.

4. Live update.
   - Phase 1: polling every 5-10 seconds.
   - Phase 2: server-sent events or websocket if the API already supports it cleanly.
   - Do not animate every historical point on every refresh.

## Relationship To Current Visualizations

The refreshed local visualizations are reference material, not implementation targets:

- `agent_journey.html` proves the useful field set and current agent distribution.
- `scripts/visualize_agent_journey.py` contains reusable scoring/timestamp ideas, but it is SSH-backed, multi-agent, static, and Plotly-based.
- `constraint_space_*.html` and `constraint_*.html` are aggregate analysis views and should not be shipped as the live public radar.

The live radar should reuse the field semantics, not the generated Plotly HTML.

## Implementation Phases

### Phase 1: Public-Safe Data Contract

- Define public agent slug configuration.
- Add trajectory projection endpoint.
- Add tests that assert forbidden fields are absent.
- Add tests for no arbitrary agent-name lookup.
- Add tests for trace-level and signature filters.

### Phase 2: Static Live Radar

- Add static HTML/JS page.
- Poll the new endpoint.
- Render deterministic trajectory.
- Include freshness, empty, error, and reduced-motion states.

### Phase 3: Embed Mode

- Add compact layout for embedding in an agent/operator console.
- Support `?agent=ally&embed=1`.
- Keep the same public-safe endpoint.
- Add CSP-friendly script structure if the host page requires it.

### Phase 4: Authenticated Drilldown

- Link public radar points to authenticated admin trace details only for logged-in operators.
- Use the stricter single-agent pendulum FSD for full trace inspection.
- Keep reasoning and JSONB payloads out of the public page.

## Open Decisions

- Whether public timestamps should be exact or bucketed.
- Whether raw `selected_action` is public-safe or should be grouped.
- Whether `Ally` should be the only public agent initially.
- Whether the first live transport should be polling or server-sent events.
- Whether public radar belongs under `/lens/` static hosting or an agent-console route.

## Acceptance Criteria

- Public radar never returns raw `agent_id_hash`, prompt text, reasoning text, JSONB components, signatures, internal error details, `task_id`, or `thought_id`.
- Public radar shows one configured agent at a time.
- Latest trajectory updates without a full page reload.
- Missing scores or timestamps degrade cleanly.
- Public endpoint is capped, cached, and rate-limited.
- Authenticated drilldown, if present, is clearly separate from public data.
