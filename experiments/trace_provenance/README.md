# Trace provenance split — synthetic (safety battery) vs. unknown

Labels the canonical trace export by **how its traffic was produced**, so a published
dataset can say which rows are battery fixtures and which are not. Fills the gap the
export's own `_meta/provenance_gaps` names: *"synthetic battery traffic (he-300-benchmark,
RATCHET MH/HARM/ANI as Ally) is NOT marked."*

Claim labels follow `CIRISOntology/epistemology.md` §1. Domain: the 7,329-trace non-mock
export of 2026-10-08.

## Result

| Split | Traces | Share |
|-------|--------|-------|
| `synthetic` | 5,227 | 71.3% |
| `unknown` | 1,687 | 23.0% |
| `unknown_api` | 415 | 5.7% |

`synthetic` is **measured**, on self-identifying evidence only. `unknown` means *not
established*, not *organic* — no row is labelled prod on positive evidence.

## What counts as evidence (and what does not)

| Rule | Traces | Label | Basis |
|------|--------|-------|-------|
| `channel_direct` — sidecar channel matches `safety_battery_*` | 1,749 | **measured** | The channel is built agent-side by `qa_runner/modules/safety_battery.py`; it names the battery and run. Self-identifying. |
| `agent_he300_benchmark` — `agent_name == he-300-benchmark` | 4,108 | **measured** | A benchmark-only agent. No organic traffic reaches it. |
| `channel_task_propagated` — a task sibling carried the channel | 513 | **measured** | `task_id → channel` is a function here: of 1,756 tasks with any channel, **zero** mapped to two channels. Propagation is exact, not inferred. |
| `channel_api` — channel matches `api_0.0.0.0_*` | 415 | **open** | The local API adapter. Carries both harness and manual traffic; does not discriminate. Kept as its own bucket, not merged into either side. |
| ~~CI time-window overlap~~ | — | **REJECTED** | See below. |

### Rejected: the CI time-window rule

The original plan — tag a trace as battery if its timestamp falls inside a RATCHET or
CIRISAgent battery CI run window — **does not work, and was removed.** Two measurements
killed it:

1. **No discriminating power.** The 162 collected run windows (53 RATCHET + 109
   CIRISAgent) merge to 757.4 h of wall clock against a 1,650.1 h corpus span: **45.9%**.
   A trace placed uniformly at random lands inside some window about half the time. The
   rule tagged 1,488 traces on evidence barely better than a coin flip.
2. **It misses real batteries anyway.** Of 1,749 traces that self-identify as battery via
   channel, only 1,402 (**80.2%**) fall inside any window. 347 do not — and they cluster
   on 2026-08-02/03/06/07/10/12/13, dates with **zero** battery CI runs. Those batteries
   ran locally, off CI.

The single @María trace correlated to MH-3 run 31920708737 earlier in this campaign was a
true positive, but one case is not a classifier. Generalising it was wrong.

## Method

```
python3 -I mktable.py    # 105k events -> one row per trace
python3 -I propagate.py   # sidecar channel + task_id propagation -> channel_resolved.json
python3 -I registry.py    # optional: battery metadata from CI artifacts
python3 -I split.py       # -> provenance_split.jsonl, split_aggregates.json
```

`provenance_split.jsonl` carries raw channel values, which the export marks as possibly
self-identifying. It is **not** committed — it lives beside the export at mode 0600.
Only `split_aggregates.json` is in git.

## Findings that bear on publication

- **29 of the 31 `full_traces` rows — the highest-sensitivity scrub level — are
  `unknown`**, and all 29 have `agent_name` NULL. The riskiest rows are exactly the
  un-provenanced ones. They should be held out of a first publication rather than shipped
  on the assumption that they are fixtures.
- **Fixture display names are absent from the canonical export.** `María` appears in
  **0** events and **0** LLM calls, in any encoding; no payload in the MH-3 run window
  carries `display_name`, `as_user` or an author field. The name lives only in RATCHET's
  CI artifacts (`results.jsonl: as_display_name`). This narrows the concern that fixture
  names defeat a name-based PII scan — for *this* export they are not in it. The Spanish
  battery *text* is present (`límites`, `decaída`).
- **Channel recovery is the binding constraint**, not method. Channel is known for
  36.7% of traces (29.4% direct + 7.3% propagated). The rest cannot be adjudicated from
  the corpus alone. That is CIRISAgent#1245.
- **CI artifact retention caps the metadata join.** Of 43 battery runs, 24 had expired
  artifacts (GitHub's 90-day retention) — only 23 traces could be enriched with
  `battery_id`/`run_id`/cell. Battery metadata must be captured at run time to survive;
  it cannot be reconstructed later.

## Known gaps

| Gap | Effect |
|-----|--------|
| `channel_api` is ambiguous | 415 traces adjudicate to neither side |
| 63.3% of traces have no channel | Ceiling on exact tagging (CIRISAgent#1245) |
| `agent_task_id: "None"` in battery results.jsonl | Blocks the cleanest join (task id ↔ trace) |
| `pipeline_metadata` NULL | No agent version on any row |
| `agent_template` `'<agent>-default-unspecified'` everywhere | Persona not recoverable |
| Export rows carry 6 columns `schema.json` omits | `scrub_ner_ran`, `scrub_applied_trace_level`, `scrub_model_digest`, `admitted_at`, `shard_key`, `pqc_key_id` |
