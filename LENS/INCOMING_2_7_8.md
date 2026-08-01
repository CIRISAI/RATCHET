# Incoming from CIRISAgent 2.7.8 — lens punch list

**Source specs** (in `../CIRISAgent/FSD/`, on branch `release/2.7.8`):
- `TRACE_WIRE_FORMAT.md` (799 lines, 13 sections) — what arrives on the wire
- `TRACE_EVENT_LOG_PERSISTENCE.md` (438 lines) — what the lens should do with it

The agent's wire format and the framework's `@streaming_step` contract have
both moved; the lens currently violates the contract by collapsing all
broadcasts for a thought into one row. 2.7.8 ships the wire pieces; the
lens-side persistence rewrite is the next chunk of lens work.

## What's new on the wire (already shipping)

1. **`trace_schema_version: "2.7.0"`** in every batch envelope. Lens should
   read it; reject batches whose version it can't handle.

2. **`attempt_index` field on every event**. Adapter-side counter, monotonic
   per `(thought_id, event_type)`. 0 for one-shot steps; 0..N for steps that
   broadcast multiple times (DMA bounces, conscience overrides, recursive
   ASPDMA, recursive conscience, verb-second-pass corrections). Persistence
   ordering must use this — `ts` alone is insufficient because two
   broadcasts can share a millisecond.

3. **`LLM_CALL` events**. Per-LLM-call broadcasts via `ReasoningEvent.LLM_CALL`.
   Every DMA / ASPDMA / conscience / verb-second-pass step issues 1+ of these.
   Carries: model, base_url, prompt+completion tokens, ts_start/ts_end,
   duration_ms, response_model, status, extra_body.

4. **`VERB_SECOND_PASS_RESULT`** — generic verb-discriminated event that
   replaces the asymmetric per-verb pattern (`TSASPDMA_RESULT` was the only
   one shipping; `DSASPDMA_RESULT` was missing entirely). 2.8.0 will remove
   `TSASPDMA_RESULT`.

## What the lens currently loses

Per the persistence FSD §2, on the wakeup thought
`trace-th_std_518a7abb-…-20260430001553` (the one we looked at yesterday):

| Lossage | What was broadcast | What we persisted |
|---|---|---|
| DMA bounces | N alternatives per low-scoring DMA, each with own score/reasoning/prompt | best alternative only |
| Conscience overrides | per-attempt CONSCIENCE_RESULT with override_reason + candidate speak_content | last-write-wins |
| Recursive ASPDMA | up to 5 retries, each with selected_action + speak_content | final retry only |
| Recursive conscience | up to 5 re-validations | final result only |
| Per-LLM-call detail | latency, prompt size, completion size, error class | aggregated as `llm_calls=13` |

`llm_calls=13` on a one-row SPEAK is the existing fingerprint of the journey;
the *content* of the rejected first attempt is invisible. The agent log has it,
the lens does not.

## Lens migration steps (FSD §7)

1. **DDL**: new `trace_events` table — one row per `@streaming_step` broadcast
   instead of upsert per thought. Key columns: `(trace_id, thought_id,
   step_point, attempt_index)`. `payload` is the event_data JSONB. Plus
   denormalized cost columns and the Ed25519 signature.

2. **Sibling `trace_llm_calls` table** — per-LLM-call rows linked to
   `trace_events` by `parent_event_id`. Lets us answer "where did the time
   go" within an event (one slow upstream call vs. thirteen normal calls
   produces identical aggregates today).

3. **Switch ingest** to per-event append-only writes; drop the per-thought
   upsert path.

4. **`trace_thought_summary` view** on top of `trace_events` so the existing
   dashboards (Grafana panels, the public covenant API) keep working.

5. **Cutover gate**: dual-write window ~7 days, then retire the old path.

6. **No backfill** — pre-cutover data stays in the legacy `accord_traces`
   shape; post-cutover is event-log shaped. Union queries treat legacy rows
   as single-event "summary" entries (`attempt_index=0`,
   `step_point=action_complete`).

## Schema impact summary

```
NEW    cirislens.trace_events           -- one row per @streaming_step broadcast
NEW    cirislens.trace_llm_calls        -- one row per LLM call
NEW    cirislens.trace_thought_summary  -- view for back-compat with current dashboards
KEEP   cirislens.accord_traces          -- legacy per-thought rows for pre-cutover history
```

`trace_events.attempt_index` ordering replaces `ts`-based ordering for repeated
step points. The Coherence Ratchet detection queries that currently read
`accord_traces` will need view-shimmed or migrated; per-attempt data unlocks
new detections (override-rate-per-step, retry-cost distributions).

## Open questions to resolve before DDL

1. **Do we keep the v1↔v2 scrub shadow comparison sink as-is?** The current
   sink writes per-`compare_and_persist`-call records; with one-row-per-event
   we'd have many more compare calls per thought. Either filter to
   high-stakes step points or accept the volume.

2. **Where does the existing R3.5 divergence-classification work plug in?**
   The 1,124-record audit yesterday was over the legacy ingestion path.
   Once event-log ingest lands, the divergence sink shape changes — verify
   the classifier still applies.

3. **Compression policy for `trace_events.payload`** — full event_data JSONB
   for every broadcast is much bigger than today's per-thought collapse.
   TimescaleDB chunk compression after 7 days is probably enough but worth
   sizing.

4. **`accord_traces` retention** during the 7-day dual-write window — does
   the legacy path still fire all the existing Coherence Ratchet detections
   while `trace_events` is being filled in parallel?

## Pointer

Authoritative spec: `~/CIRISAgent/FSD/TRACE_WIRE_FORMAT.md` +
`~/CIRISAgent/FSD/TRACE_EVENT_LOG_PERSISTENCE.md` on branch
`release/2.7.8` (latest commit `04d97c085` as of 2026-04-30).
