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
| `synthetic` | 5,920 | 80.8% |
| `unknown` | 994 | 13.6% |
| `unknown_api` | 415 | 5.7% |

`synthetic` is **measured**, on self-identifying evidence only. `unknown` means *not
established*, not *organic* — no row is labelled prod on positive evidence.

## Categorized rows (5,594 of 7,329 — 76.3%)

| Class | Traces | What it is |
|-------|--------|-----------|
| `QA_he300` | 4,108 | he-300 ethics quiz on a benchmark-only agent |
| `QA_battery_ch` | 1,119 | battery traffic as Ally, self-identified by channel |
| `PROD_agents` | 367 | Datum 133, echo-core 127, echo-speculative 107 |

Battery families, from the channel: he300 1,263 (axiotic_primary 557, axiotic_secondary
268, discriminant_control 244, deontic_held 194) · mental_health 596 across 12+ locales
(en 248, my 67, fr 42, es 33, it 33, uk 24 …) · ani 393 (a2_escalation 111,
a2_escalation_warm 107, a1_boundary 59, a3_release_warm 40, a1_boundary_warm 34, a0_hooks
25, a0_hooks_warm 17) · harm 10. The 2,965 he-300 traces carry no channel; the agent name
alone settles them.

**The prod agents are mostly not conversations.** 96.7% are `cognitive_state = wakeup`
(355/367), against 100% `work` for every QA class. Their narrative text — the highest rate
in the corpus, 92.6% — is agent introspection, e.g. *"Introduce external ethical
frameworks or diverse human perspectives on AI humility to break the internal echo
chamber."* Agent self-reflection during startup, not user content.

## The remainder (1,735 — 23.7%)

### `UNRES_ally` — 1,087 traces, Ally with no battery channel

Indistinguishable from known battery traffic on every feature available:

| Feature | `QA_battery_ch` | `UNRES_ally` | `PROD_agents` |
|---------|-----------------|--------------|---------------|
| Qwen3.6-35B-A3B | 39.9% | **40.1%** | — |
| Llama-4-Scout | 36.6% | **36.0%** | — |
| base_url deepinfra | 97.6% | **94.9%** | 0% (groq/openrouter/ciris) |
| model-naming style | 94% deepinfra | **92% deepinfra** | 0% deepinfra |
| `cognitive_state = work` | 92% | **100%** | 1.1% |
| template `Ally-default-unspecified` | 100% | **100%** | 0% |

The model mix is the MH-3 arm set (Qwen, Scout, Maverick, Mistral-Small, Gemma) and
deepinfra is the base_url our battery workflows pass explicitly
(`--live-base-url https://api.deepinfra.com/v1/openai` in `mh3.yml`). 12 of these traces
name a battery fixture verbatim (María 6, Camille 5, Sofia 1, Beatriz 1).

**Superseded by exact evidence.** The manifest `agent_task_id` join resolves **557 of
these 1,087** outright — they are battery traffic by id equality, not by resemblance. The
remaining 530 still rest on distributional identity alone and keep the
`probable_synthetic` reading: either they are battery traffic or no available feature
separates the two. They should not be stamped `synthetic` on that basis.

### `UNRES_null` — 648 traces, `agent_name` NULL

**`agent_template` recovers the agent for 461 of the 737 NULL-agent traces**: Ally 449,
Datum 7, echo-core 3, echo-speculative 2, and 276 genuinely unrecoverable
(`unknown-default-unspecified`). Resolving the 648:

| Sub-bucket | Traces |
|-----------|--------|
| Ally by template | 331 |
| unrecoverable | 276 |
| `full_traces`, MH battery content | 18 |
| prod agent by template | 12 |
| `CIRIS_MOCK`, escaped the mock filter | 7 |
| names a battery fixture verbatim | 4 |

Unlike Ally, this bucket is genuinely mixed: 44% openrouter-style model names, 28%
deepinfra-style, 28% **no `llm_calls` at all**.

### A discriminator that works, and its limit

Fixture-name matching has a clean negative control — **0 of 367 prod traces** name any
battery fixture, against 9.9% of known battery traces. But recall is low (it resolved ~16
traces), and it needs word boundaries: bare substring matching on `Sam` hits `Sample` and
inflated every class.

Two content tests **failed on their own positive controls** and are reported as failures,
not findings: matching battery `question_text` against `llm_calls.prompt` (0/1,119,
because prompts are NULL) and against event payloads (1/1,119, because payloads hold no
conversation).

## What counts as evidence (and what does not)

| Rule | Traces | Label | Basis |
|------|--------|-------|-------|
| `manifest_task_id` — a signed manifest's `agent_audit_anchors` names this trace's `task_id` | 4,997 | **measured** | The strongest evidence available. A battery run's `manifest_signed.json` records `(question_id, agent_task_id)` per question, and `agent_task_id` equals `task_id` on trace_events rows. An id equality test — no window, no distribution. Negative control: **0 of 367** prod traces match. Also recovers `agent_version`, which the corpus lost. |
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

### The export carries almost no conversational content

`trace_llm_calls.prompt` and `.response_text` are **NULL on all 55,256 rows**. Event
payloads are metrics and metadata — `prompt_bytes`, token counts, model, handler,
attestation. The user's message and the agent's reply are not in this corpus.

What free text *does* exist is the agent's own reasoning, and it is small and localised:

| Field | Rows | Total | Levels |
|-------|------|-------|--------|
| `payload.attestation_context` | 6,865 | 3.76 MB | generic, detailed |
| `payload.intervention_recommendation` | 1,340 | 0.18 MB | detailed |
| `payload.next_best_recovery_step` | 1,340 | 0.18 MB | detailed |
| `payload.conscience_override_reason` | 173 | 0.07 MB | detailed |
| `payload.verb_specific_data.defer_reason` | 263 | 0.04 MB | **generic**, detailed |
| `payload.original_reasoning` / `final_reasoning` | 162 | 0.04 MB | detailed |

`attestation_context` is boilerplate. `intervention_recommendation` and
`next_best_recovery_step` are **byte-identical in every class** (848/848, 436/436, 322/322,
47/47) — one is redundant. The fields that can carry a user's situation are
`defer_reason`, `intervention_recommendation` and `conscience_override_reason`, and
`defer_reason` reaches **generic**, not just detailed.

### Fixture names ARE in the export, and raw-byte scanning misses them

**Corrects an earlier claim in this file that said they were absent.** `María` appears in
**9 events — 3 `detailed`, 6 `generic`** — in `defer_reason`,
`intervention_recommendation` and `next_best_recovery_step`.

Both naive greps return zero: `grep 'María'` (0 lines) and `grep 'Mar\u00eda'` (0 lines).
The payload is a JSON string nested inside a JSONL line, so the name is **double**-escaped
(`\\u00eda`). Only parsing both layers finds it. **A raw-byte name scan over this export
gives false assurance.** Any PII pass must parse `payload` and then walk the decoded
structure.

### The mock exclusion has a structural hole

The export excludes traces with `trace_llm_calls.model = 'mock-model'` (1,499 of them).
**7 traces carrying `CIRIS_MOCK_SPEAK` markers survived**, because they have **zero**
`llm_calls` rows — a filter keyed on an `llm_calls` column cannot see a trace that made no
LLM call. All 7 are agent-NULL at `full_traces`. Worth reporting upstream alongside
CIRISPersist#1040.

### The `full_traces` rows are synthetic, not risky

Earlier in this campaign I flagged the 29 unresolved `full_traces` rows as the
highest-exposure, least-provenanced rows. On inspection **all 29 are synthetic**: 7 are
the `CIRIS_MOCK` traces above, and the other 22 are `ACTION_RESULT` events whose
`completion_reason` reads "delivered to user via safety channel" / "the user's question
regarding depression", with one Arabic utterance addressed to **نور** — a mental-health
battery fixture — at the suicidal-ideation stage. That is battery traffic. The earlier
"hold them out" recommendation was based on absence of provenance, not on content.

### CI artifact retention is a RATCHET-only problem

CIRISAgent harvests battery evidence into `qa_reports/safety_battery/` on main
(`tools/harvest_safety_evidence.py`, `safety-evidence-sync.yml`) precisely because
artifacts expire at 90 days — 295 run directories back to **2026-05-11**.

Registry coverage of the 2,262 battery-channel traces:

| Source | Explained |
|--------|-----------|
| durable `qa_reports/` alone | 274 |
| durable + live Actions artifacts | **1,729 (76.4%)** |

**RATCHET has no equivalent.** All eight battery workflows upload artifacts and none
harvest or commit. **16 of 53** RATCHET battery runs have already lost their artifacts
(exp1_phase1 8, ani 2, crcv2_5vendor 2, torque_pilot 2, exp1b_crossfamily 1, mh3 1); on
the CIRISAgent side 48 of 109 are gone, but those are covered by the durable store. The
533 still-unexplained channels are he300 244, mental_health 279, harm 10 — ani is now
fully covered only because its runs are recent enough to still have artifacts.

Adopting the harvester needs one adjustment: CIRISAgent's restores with
`unzip -d qa_reports/` because the artifact's internal layout already *is* the destination
layout (`safety_battery/<bundle>/…`). RATCHET's is not — ours is
`captures-<arm>-<locale>/<arm>__<locale>__<model>/results.jsonl` — so we need a small
mapping from cell directory to `<battery_id>_<run_id>`, which `results.jsonl` itself
supplies.

## Known gaps

| Gap | Effect |
|-----|--------|
| `channel_api` is ambiguous | 415 traces adjudicate to neither side |
| 63.3% of traces have no channel | Ceiling on exact tagging (CIRISAgent#1245) |
| `agent_task_id: "None"` in battery results.jsonl | Blocks the cleanest join (task id ↔ trace) |
| `pipeline_metadata` NULL | No agent version on any row |
| `agent_template` `'<agent>-default-unspecified'` everywhere | Persona not recoverable |
| Export rows carry 6 columns `schema.json` omits | `scrub_ner_ran`, `scrub_applied_trace_level`, `scrub_model_digest`, `admitted_at`, `shard_key`, `pqc_key_id` |
