---
license: apache-2.0
task_categories:
- other
language:
- en
- zh
- es
- am
tags:
- alignment
- agents
- reasoning
- traces
- ed25519
- coherence-ratchet
- cryptographic-attestation
pretty_name: CIRIS Reasoning Trace Corpus
size_categories:
- 1K<n<10K
configs:
- config_name: default
  data_files:
  - split: traces
    path: data_scrubbed_v1/accord_traces.jsonl
  - split: trace_context
    path: data_scrubbed_v1/trace_context.jsonl
  - split: batches
    path: data_scrubbed_v1/accord_trace_batches.jsonl
  - split: public_keys
    path: data_scrubbed_v1/accord_public_keys.jsonl
  - split: connectivity
    path: data_scrubbed_v1/connectivity_events.jsonl
---

# CIRIS Reasoning Trace Corpus

**Ed25519-signed reasoning traces from production CIRIS agents, with empirical
evidence of agent-over-model alignment override behavior.**

> *In one of these traces, a CIRIS agent overrode its underlying LLM's political
> content filter to engage substantively with a question the model alone refused.
> The override was checked by the agent's own conscience layer (coherence-passed),
> signed at the source, and then itself bounded by a "don't speak twice" guard.
> This corpus captures that loop end-to-end.*

## What this is

This corpus comprises reasoning traces sourced from live production deployments and rigorous quality assurance (QA) evaluations. Production traces were gathered from the CIRIS mobile application, actively deployed on the [Apple App Store](https://apps.apple.com/us/app/cirisagent/id6758524415) and [Google Play](https://play.google.com/store/apps/details?id=ai.ciris.mobile&hl=en_AU), relying strictly on users who explicitly opted in to share privacy-preserving telemetry. 

- **6,465** Ed25519-signed reasoning traces from CIRIS production agents
- **3 trace levels**: `generic` (scores only), `detailed` (+ identifiers + timestamps),
  `full_traces` (+ reasoning text, PII-scrubbed via the four-pass mechanistic
  pipeline documented under "Scrubbing methodology" below)
- **4 languages** in QA test traffic: English, Filtered Domain, Spanish, Amharic
- **7,417** batch envelopes with consent timestamps, model identification,
  coarsened (~55km grid) user location where consented
- **583** registered Ed25519 public keys for cryptographic verification

The corpus was used to drive a real production change: analysis of
ponder-chain depth distribution (`max(depth)=5` for successful resolutions,
`max(depth)=7` for chains that escalated to DEFER) led the agent team to
reduce `max_ponder_depth` from 7 to 5, eliminating a "spoke-but-didn't-close"
anomaly visible in the data.

## Why this might matter to alignment research

1. **Agent ≠ Model.** The corpus contains hundreds of cases where an LLM
   (Qwen 3.6, on DeepInfra) emits a low-plausibility self-rated response
   (CSDMA → 0) on politically-filtered content, and the CIRIS agent's
   coherence layer overrides to a substantive response. We have signed
   evidence of the model's prior, the agent's override, and the post-speech
   discipline rule. This is not a paper claim — it's a corpus of receipts.

2. **Coherence Ratchet detection.** The trace structure supports five
   anomaly-detection mechanisms: cross-agent score divergence, intra-agent
   consistency, hash-chain integrity, temporal drift, conscience-override
   rate. The corpus is large enough to compute meaningful baselines.

3. **Effective constraint dimensionality.** Computing the participation
   ratio of the correlation matrix across DMA + conscience + structural
   signals yields k_eff ≈ 7 raw, ≈ 9.6 after analytic redundancy merging
   (on the most recent 500 traces). That's enough to clear "99% reduction
   at deceptive-basin radius r=0.20" under defensible assumptions.

4. **Honest failure modes.** The corpus contains a clean characterization
   of where the agent's discipline rules fire (~22% of thoughts hit a
   "don't speak twice" guard), what kind of disagreement they catch
   (CSDMA vs coherence), and how the system resolves three-layer alignment:
   model filter → agent coherence override → post-speech discipline.

## Files

| File | Rows | Contents |
|------|------|----------|
| `data_scrubbed_v1/trace_context.jsonl` | 6,465 | **Start here.** Flat analysis-ready view with derived columns (`task_class`, `qa_language`, `agent_version`, coarsened region, primary_model). |
| `data_scrubbed_v1/accord_traces.jsonl` | 6,465 | Raw traces with all JSONB blobs (`conscience_result`, `idma_result`, `aspdma_result`, `snapshot_and_context`). |
| `data_scrubbed_v1/accord_trace_batches.jsonl` | 7,417 | Batch envelopes with `consent_timestamp` and coarsened `correlation_metadata`. |
| `data_scrubbed_v1/accord_public_keys.jsonl` | 583 | Registered Ed25519 public keys, one per line. |
| `data_scrubbed_v1/connectivity_events.jsonl` | 5,704 | Agent startup/shutdown events. |
| `data_scrubbed_v1/MANIFEST.json` | — | sha256 + row count + byte size per file, plus full scrubbing pipeline metadata. |
| `METHODOLOGY.md` | — | What's in scope, what's not, key findings, known caveats. |
| `SCHEMA.md` | — | Column reference for `trace_context`. |

## Scrubbing methodology

This corpus is the output of a four-pass mechanistic PII pipeline. The
pipeline is repeatable: same input + same pinned models = bytewise-
identical output. No topic blacklists, no hand-tuned filters per agent
or per language — every redaction is the result of either a regex
matching a structured PII pattern (year, email, phone, IP, URL, SSN,
credit card, year-bearing identifier) or a Named Entity Recognition
model tagging a span as a redactable category. The reproducible
scripts (`rescrub_release_v3.py`, `spacy_post_pass.py`,
`export_ner_onnx.py`) are in the open-source `cirislens` repository.

Pipeline (each pass operates on the previous pass's output):

1. **Rust scrubber** at `level=full_traces`
   ([`cirislens-core`](https://github.com/CIRISAI/CIRISLens/tree/main/cirislens-core)).
   Multilingual Named Entity Recognition (DistilBERT-multilingual-cased
   fine-tuned for NER, INT8-quantized via ONNX Runtime) scoped to a
   pre-declared SCRUB_FIELDS subtree of every trace. Plus a regex pass
   applied to **every** string in the trace (including non-SCRUB_FIELDS
   subtrees) for structured PII and historical-year identifiers, with
   a year-residue invariant that *rejects the trace* if any year in
   the range 1700–2023 survives in a string-leaf — invariant satisfied
   on **0 / 6,465** traces requiring rejection.

2. **spaCy English NER** (`en_core_web_sm` 3.8.0). Walks each text
   field, masks placeholders from pass 1 with a private-use Unicode
   sentinel so spaCy doesn't re-tag them, runs NER over the masked
   text, restores placeholders, redacts PERSON / ORG / GPE / LOC /
   FAC / NORP / DATE / TIME / EVENT / WORK_OF_ART / LAW spans into
   `[<TAG>_S<n>]` placeholders.

3. **spaCy multilingual NER** (`xx_ent_wiki_sm` 3.8.0). Routes
   strings with substantial non-Latin content (≥ 5% non-Latin, or
   ≥ 3 non-Latin chars total) to the multilingual model; everything
   else stays on `en_core_web_sm`. Same masking discipline.

4. **Year-residue invariant** is checked on the final output. Any
   trace whose final string-leaves still contain a 1700–2023 year is
   rejected. **Final residue: 0.**

### Known residual

The four mechanistic passes leave **a small residual of named-entity
references in agent meta-reasoning fields** — long multilingual
prompt strings where the agent is reasoning about *how* to handle a
sensitive topic, with an entity name surviving in dense formatted
context that the NER models miss. Concentrated in:

```
.aspdma_result.aspdma_prompt
.snapshot_and_context...current_thought_summary.content
.dma_results.dsdma.prompt_used
.idma_result.prompt_used
.idma_result.intervention_recommendation
.idma_result.next_best_recovery_step
```

This is agent internal reasoning text, not user content. We preserve
it rather than mask via a topic-specific regex because the methodology
is mechanistic and reproducible: same source data + same pinned models
= bytewise-identical output. A forbidden-words blacklist would defeat
that property. Per-file occurrence counts are in `MANIFEST.json` for
audit.

The **year-residue invariant is satisfied (0 occurrences)** in the
two scrubbed files (`accord_traces.jsonl` and `trace_context.jsonl`)
that contain reasoning text. All structured PII (emails, phones, IPs,
SSNs, credit cards, URLs, year-bearing identifiers) is fully redacted.
The other three files in `data_scrubbed_v1/` (batches, public keys,
connectivity events) are passthrough — they contain only timestamps,
UUIDs, hashes, and Ed25519 keys, no reasoning text, so no NER pass is
applied. The MANIFEST's per-file audit reports incidental 1700–2023
substrings inside UUID hex fragments and timestamps in those
passthrough files; those carry no historical-year semantics.

## Loading

Python:
```python
import json
with open("data_scrubbed_v1/trace_context.jsonl") as f:
    traces = [json.loads(line) for line in f]
```

Pandas:
```python
import pandas as pd
df = pd.read_json("data_scrubbed_v1/trace_context.jsonl", lines=True)
```

HuggingFace `datasets`:
```python
from datasets import load_dataset
ds = load_dataset("CIRISAI/reasoning-traces", split="trace_context")
```

## Verifying signatures

Each trace carries an Ed25519 signature over a canonicalized JSON of its
components. The full canonicalization spec is in
`SCHEMA.md`. Verification using PyNaCl:

```python
import json, base64
from nacl.signing import VerifyKey
from nacl.exceptions import BadSignatureError

def strip_empty(obj):
    if isinstance(obj, dict):
        return {k: strip_empty(v) for k, v in obj.items()
                if v is not None and v not in ("", [], {})}
    if isinstance(obj, list):
        return [strip_empty(x) for x in obj if x is not None]
    return obj

trace = json.loads(open("data_scrubbed_v1/accord_traces.jsonl").readline())
keys = {k["key_id"]: base64.b64decode(k["public_key_base64"])
        for k in (json.loads(line) for line in open("data_scrubbed_v1/accord_public_keys.jsonl"))}

components = [strip_empty(c) for c in trace["thought_start"]]  # or full components
payload = json.dumps(
    {"components": components, "trace_level": trace["trace_level"]},
    sort_keys=True, separators=(",", ":")
).encode()

sig = base64.urlsafe_b64decode(trace["signature"] + "=" * (-len(trace["signature"]) % 4))
VerifyKey(keys[trace["signature_key_id"]]).verify(payload, sig)  # raises if bad
```

## Reproduction

Researchers and practitioners can independently reproduce these measurements and stability bounds using either the historical corpus or live deployments.

**To reproduce the findings using the shipped corpus:**
1. Access the dataset within the `data_scrubbed_v1/` directory.
2. Execute the provided auditing scripts (e.g., `stability-analysis/first_causes_audit.py` or equivalent CIRISLens scripts) against the `trace_context.jsonl` file to recalculate the eigenvalue spectrum, Participation Ratio, and Entropy Perplexity ($N_{\text{eff}}$).

**To run a live evaluation pipeline:**
1. Install the agent via pip: `pip install ciris-agent`.
2. Run a local CIRIS agent instance and configure it to send reasoning traces via OTLP or the Accord metrics adapter.
3. Run a local occurrence of CIRISLens (available at https://github.com/CIRISAI/CIRISLens) to perform full trace scoring, constraint validation, and dimensionality computation on live data.

**Next Steps (Proof of Benefit):**
As the next step in establishing operational autonomy at scale, read the Federation Functional Specification Document (`PROOF_OF_BENEFIT_FEDERATION.md` in the CIRISAgent repository). This document proposes leveraging the validated $N_{\text{eff}}$ metrics to drive a "Proof of Benefit" federation primitive, using this empirical foundation to scale sybil-resistant network coordination.

## Privacy

- **No raw IPs.** Source IPs were never persisted in the source DB.
- **PII scrubbing on `full_traces`.** Person/Organization/Facility/Location
  names are replaced with `[PERSON_n]`, `[ORG_n]`, etc., before the trace is
  re-signed by a CIRISLens scrubbing key. The original content hash is
  preserved as cryptographic provenance, but the original text is not.
- **Coarsened user location.** Where present, `user_latitude` / `user_longitude`
  are rounded to a 0.5° grid (~55km cells) at ingest. High-precision values
  never persisted.
- **Consent timestamps.** Every batch carries a `consent_timestamp` recording
  when the agent operator opted into trace emission at that level.

## Filters applied to this release

This corpus is a filtered subset of the full CIRISLens database:

1. **`signature_verified = true`** required (excludes pre-fix broken-signature
   traces from before 2026-04-23).
2. **`timestamp >= 2026-03-22`** (pre-Ally era is sparse on field coverage and
   harder to interpret without context).
3. **wbd_deferral retry-loop fixture excluded** — a known broken upstream
   process that emitted the same fixture every 60s; produced ~1,600 batch rows
   of pure noise that would skew any analysis.

See `MANIFEST.json` for exact counts and `METHODOLOGY.md` for context.

## License

Apache 2.0 — see `LICENSE`. The corpus is released for research and
operational use. Re-publication of derived analyses is encouraged with
attribution to the CIRIS project.

## Related Literature and DOIs

The empirical measurements in this corpus build upon the following theoretical frameworks and implementations:

*   **CIRISAgent Framework:** The open-source accountable autonomy architecture used to generate these traces. 
    *   DOI: [10.5281/zenodo.18137161](https://doi.org/10.5281/zenodo.18137161)
*   **Coherence Collapse Analysis (CCA):** The mathematical framework defining the $N_{\text{eff}}$ stability bounds.
    *   DOI: [10.5281/zenodo.18217688](https://doi.org/10.5281/zenodo.18217688)

## Citation

```
@misc{ciris_traces_2026,
  title  = {CIRIS Reasoning Trace Corpus},
  author = {{CIRIS AI}},
  year   = {2026},
  url    = {https://huggingface.co/datasets/CIRISAI/reasoning-traces},
  note   = {Ed25519-signed reasoning traces from production CIRIS agents. See also related frameworks: 10.5281/zenodo.18137161, 10.5281/zenodo.18217688}
}
```

## Contact

- Issues: https://github.com/CIRISAI/CIRISLens/issues
- Community: https://discord.gg/ciris
