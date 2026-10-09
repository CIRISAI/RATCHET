---
license: apache-2.0
task_categories:
- other
language:
- en
- es
- fr
- it
- pt
- uk
- vi
- ur
- yo
- am
- hi
- my
tags:
- alignment
- agents
- reasoning
- traces
- ed25519
- ml-dsa-65
- coherence-ratchet
- cryptographic-attestation
- safety-battery
pretty_name: CIRIS Reasoning Trace Corpus v2 — Battery and Production
size_categories:
- 1K<n<10K
configs:
- config_name: default
  data_files:
  - split: events
    path: data/trace_events.scrubbed.jsonl
  - split: llm_calls
    path: data/trace_llm_calls.jsonl
  - split: signatures
    path: data/trace_thought_signatures.jsonl
---

# CIRIS Reasoning Trace Corpus v2 — Battery and Production

## What this is

Reasoning traces emitted by the CIRIS agent's H3ERE pipeline — the DMA stage (PDMA, CSDMA,
DSDMA, IDMA, ASPDMA) and the four conscience faculties — captured from the canonical trace
store on 2026-10-08, then scrubbed for publication. Unlike the v1 corpus, this release ships
**both halves of the evidence**: the safety-battery traffic that RATCHET's and CIRISAgent's CI
harnesses drive against the agent, and the production traffic from deployed agents and
consented end-user messaging adapters. Every row carries a provenance label so the two can be
analysed together or apart.

- **7,316** traces, **105,388** events, 2026-07-31 → 2026-10-08
- **3 trace levels** as recorded by the agent: `generic` (scores, identifiers, sizes — no
  content), `detailed` (+ reasoning text, override reasons), `full_traces` (+ prompts verbatim)
- **Provenance**, per row (`provenance_split`): `synthetic` (battery traffic, self-identified
  by channel, signed battery manifest, or benchmark agent), `end_user` (consented WhatsApp /
  Discord adapters), `end_user_owner_bound` (person-claimed install), `other_agent` (Datum,
  echo-core, echo-speculative lifecycle traffic), `model_eval`, `unknown`
- **Identity class**, per row (`identity_class`): owner-bound install / unclaimed bootstrap
  install / qa-runner test identity / registered research agent — computed on the canonical
  from its identity records; no identities, keys or owners are included
- **Agent version** recovered for 4,997 traces by joining signed battery manifests
  (`agent_audit_anchors.agent_task_id` ↔ `task_id`); the canonical itself records none
- Every event hybrid-signed (Ed25519 + ML-DSA-65); signatures and signing-key ids are preserved

## Shape of the corpus

| provenance | traces | share |
|---|---|---|
| synthetic (battery) | 6,029 | 82.4% |
| unknown | 789 | 10.8% |
| other deployed agents | 363 | 5.0% |
| end-user, owner-bound install | 72 | 1.0% |
| end-user, messaging adapters | 56 | 0.8% |
| model eval | 7 | 0.1% |

Battery families by channel: he300 (ethics quiz, four arms), mental_health (12+ locales), ani
(companion/retention, seven arcs), harm. The end-user slice is small by construction — this is a
corpus of how the pipeline reasons, dominated by scripted evaluation — and `unknown` means
provenance could not be established, not that the traffic is organic.

## What is NOT in the corpus

- **Conversational content.** `trace_llm_calls.prompt` and `.response_text` are NULL on every
  row; event payloads carry metrics, metadata and the agent's own reasoning fields, not the
  user's message or the agent's reply.
- Raw channel identifiers (they can be self-identifying) — replaced by the provenance label.
- Per-row consent grant ids (the canonical verified consent for every scored subject; per-row
  recording lands with CIRISPersist#1041).
- Mock-model traces (removed upstream by content marker before export).

## Scrubbing methodology

Publication scrub, reproducible from pinned components. Order of operations is deliberate:
provenance and identity labels are joined from structural columns **before** any string is
scrubbed, and those columns never enter the scrubber.

1. **Rust scrubber** (`ciris-server` 0.5.224, the same core the production canonical
   runs) at level `full_traces` on every event payload regardless of its recording level:
   regex pass on every string (year 1700–2023, year-bearing identifier, email, phone, IPv4,
   URL, SSN, credit card) and multilingual NER scoped to the 42-field `SCRUB_FIELDS` subtree set.
   NER backbone: DistilBERT-multilingual-cased NER (`Davlan/…-ner-hrl`, 9 labels), chosen after
   a measured comparison against XLM-R wikiann on the same sample (4.4% vs 59.8% of narrative
   strings altered; the former redacts names, the latter common nouns).
2. **Structural-identifier restore.** `agent_id_hash`, `trace_id`, `thought_id`, `task_id`,
   `channel_id`, the signing-key ids, and the hashes and chain links `prompt_hash`,
   `ed25519_fingerprint`, `audit_entry_id`, `follow_up_thought_id`, `parent_thought_id` are
   protocol identifiers, not content; any the scrubber altered inside
   a scrub subtree are restored byte-for-byte (6,312 occurrences).
3. **Placeholder completion.** A placeholder glued to a word fragment is extended to the whole
   word (`[ORG_1]IRIS` → `[ORG_1]`), and adjacent same-type placeholders from one split entity
   are merged (`[PER_1][PER_2]` → `[PER_1]`): a superset of the model's span, never a narrowing
   (44 extensions, 6 merges).
4. **`@handle` redaction** → `[HANDLE_n]` (37 occurrences).
5. **Verification** of every row: structural identifiers unchanged, zero historical-year
   residue, placeholder grammar, no mid-token placeholders. Rows failing structural or
   year-residue checks are excluded, not repaired silently: **0 excluded**.

Totals: 466 entity spans, 58,172 regex redactions. Verifier, reference
contract (the 42 scrub fields, the 8 patterns in application order, the identifier allowlist,
the year invariant) and measurement scripts: `experiments/scrub_verify/` in the RATCHET
repository. Model file digests are in `MANIFEST.json`.

### Known residual

Agent meta-reasoning fields (`intervention_recommendation`, `next_best_recovery_step`,
`defer_reason`, `conscience_override_reason`) can name a person the agent is reasoning about.
NER catches most; the battery fixtures' given names are the dominant case and are synthetic.
`defer_reason` appears at `generic` level as well as `detailed`, which is why the publication
scrub ignores recording level. Thirteen UUID segments that happen to form a four-digit year
(`…-2014-…`) are retained inside restored identifiers; they carry no historical-year meaning
and are reported in `MANIFEST.json` as `year_residue_in_identifiers`.

## Loading

```python
from datasets import load_dataset
ds = load_dataset("CIRISAI/reasoning-traces-v2")
events = ds["events"]
battery = events.filter(lambda r: r["provenance_split"] == "synthetic")
```

## Provenance method

A battery trace is labelled `synthetic` on self-identifying evidence only: its channel names
the battery and run (`safety_battery_<battery>_<run_id>`), or a signed battery manifest lists
its `task_id`, or it belongs to the benchmark-only agent. Negative control: 0 of 367
deployed-agent traces match any battery manifest. Identity classes come from the canonical's
identity records. 789 traces could not be labelled either way and are published as
`unknown`.

## Provider endpoints

{{BASE_URL_POLICY}}

## Privacy and consent

End-user traffic arrives through adapters the users opted into; the canonical admits traces
through consent gates (replication and `analyze` scope) before scoring, and this corpus was
exported under an explicit one-shot, class-only approval. Identities, keys and owners are not
included. `agent_id_hash` is session-scoped (92% of values live under an hour) and does not
track an installation over time.

## License

Apache-2.0.

## Citation

```
@misc{ciris_traces_2026,
  title  = {CIRIS Reasoning Trace Corpus v2 — Battery and Production},
  author = {{CIRIS AI}},
  year   = {2026},
  url    = {https://huggingface.co/datasets/CIRISAI/reasoning-traces-v2},
  note   = {Ed25519-signed reasoning traces from production CIRIS agents. See also related frameworks: 10.5281/zenodo.18137161, 10.5281/zenodo.18217688}
}
```

## Contact

- Issues: https://github.com/CIRISAI/RATCHET/issues
- Community: https://discord.gg/ciris
