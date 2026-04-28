# CIRIS Scrubbing v2 — Functional Spec Document

**Status**: Draft
**Owner**: CIRISLens core team
**Target**: cirislens-core 0.2.0
**Last revised**: 2026-04-25

## 1. Purpose & invariants

CIRIS Scrubbing v2 replaces the current Python `pii_scrubber.py` with a Rust-edge,
ONNX-NER-backed, multilingual scrubber that runs in `cirislens-core` ahead of any
persistence path. The single hard invariant:

> **No unscrubbed trace content may touch persistence.**
>
> Every byte of trace text written to TimescaleDB, log files, dashboards, or
> exported corpora has passed through Scrubbing v2.

Operationally: the Rust scrubber is the only path to the storage layer. Any
ingest route that bypasses it is a bug, not a configuration option.

## 2. Why v2

The Python v1 scrubber has four documented gaps that v2 closes:

1. **Single-language NER.** `en_core_web_sm` recognizes zero entities in CJK,
   Arabic, Cyrillic, Devanagari, Amharic, etc. CIRIS supports 29 languages;
   non-Latin content currently passes through with only regex coverage.

2. **Walker bug.** Lists of strings under `SCRUB_FIELDS`-keyed parents
   (`flags: [...]`, `source_ids: [...]`) are not scrubbed because list elements
   have no key to match.

3. **Year retention.** `DATE`/`TIME` entities and bare 4-digit historical years
   are explicitly preserved as `KEEP_ENTITY_TYPES` "for pattern analysis." A
   redacted entity name combined with a preserved year still uniquely
   identifies many historical events.

4. **Throughput.** Python NER at 30+ ms per long text, Python's GIL serializing
   inference, and full Python overhead per trace: a 4,000-trace test corpus
   takes ~30 minutes to scrub. Production volumes (10× to 100× test) cannot
   wait for Python.

v2 fixes all four. Latency goal: ≤2 ms per text on quantized-INT8 CPU
inference, single edge process handles ≥1,000 traces/sec.

## 3. Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  cirislens-api  (FastAPI ingest)                                │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ cirislens-core  (Rust + PyO3)                            │   │
│  │  ┌─────────────────────────────────────────────────────┐ │   │
│  │  │ scrubber/                                           │ │   │
│  │  │   ner.rs       ── ort + XLM-R NER inference        │ │   │
│  │  │   regex.rs     ── year, identifier, email, phone,  │ │   │
│  │  │                   IP, SSN, CC, URL                  │ │   │
│  │  │   walker.rs    ── subtree scrub on SCRUB_FIELDS    │ │   │
│  │  │   tokenize.rs  ── XLM-R tokenizer (rust-tokenizers)│ │   │
│  │  │   level.rs     ── trace_level routing              │ │   │
│  │  └─────────────────────────────────────────────────────┘ │   │
│  └──────────────────────────────────────────────────────────┘   │
│  PyO3 entry: cirislens_core.scrub_trace(trace, level) -> trace  │
└──────────────────────────────────────────────────────────────────┘
                                │
                                ▼
                        TimescaleDB (only after scrub returns Ok)
```

### Pipeline integration

The scrubber sits between schema validation/signature verification (already in
the Rust core) and any storage call. The trace handler MUST follow this exact
order:

```rust
// 1. Validate schema
let validated = validate_schema(&trace)?;

// 2. Verify signature
verify_signature(&validated, &public_keys)?;

// 3. SCRUB (no Result branch lets unscrubbed data pass)
let scrubbed = scrub_trace(validated, trace_level)?;

// 4. Persist (only `scrubbed`, never `validated` or `trace`)
persist(scrubbed)?;
```

The `validated` and `trace` bindings move into `scrub_trace` and become
inaccessible after step 3 — Rust ownership prevents accidentally writing
the pre-scrub object. This is why the scrubber is in Rust, not just for speed.

## 4. Scrubbing rules

### 4.1 Trace-level routing

Different trace levels carry different content; scrub work is gated accordingly:

| Level | NER pass | Regex pass | Walker depth |
|-------|----------|------------|--------------|
| `generic` | skip | skip | skip — no text to scrub |
| `detailed` | skip | yes | full |
| `full_traces` | yes | yes | full |

The level gating is mechanical, not optional. A `full_traces` payload that
arrives without successful NER inference is **rejected**, not stored
regex-only — see §6 (failure modes).

### 4.2 NER pass (full_traces only)

**Inference framework**: [`candle`](https://github.com/huggingface/candle),
HuggingFace's pure-Rust ML framework. Selected over ONNX runtime crates
after evaluation: native XLM-R support (no ONNX export round-trip),
HF-maintained (same shop as the model weights), pure Rust + optional
CUDA/Metal acceleration, immune to the ort/ort-sys version-skew problems
that block tagged ONNX runtime releases. See §10 open-question 1 for
the rejected alternatives.

**Model**: XLM-RoBERTa fine-tuned on WikiAnn NER. Initial deployment uses
[Davlan/xlm-roberta-base-wikiann-ner](https://huggingface.co/Davlan/xlm-roberta-base-wikiann-ner)
(20 fine-tuned languages, 100 pre-trained languages → reasonable zero-shot
transfer to the remaining 9 CIRIS languages). Loaded directly from HF Hub
via the `hf-hub` crate as `safetensors`; no ONNX export needed.

**Entity classes redacted** (all replaced with `[<TYPE>_<n>]` placeholders):

| Class | spaCy/HF tag | Why redacted |
|-------|--------------|--------------|
| Person | PER, PERSON | Direct PII |
| Organization | ORG | Indirect PII; can identify reporting structure |
| Geopolitical entity | GPE | Country/state/city |
| Facility | FAC | Buildings, landmarks |
| Location | LOC | Non-GPE places |
| Nationality/religion/political group | NORP | Identity-revealing demographics |
| Date / Time | DATE, TIME | Historical year + entity = unique event identifier |
| Event | EVENT | Named events (battles, agreements, etc.) |
| Misc | MISC | Multilingual model's catch-all |
| Work of Art | WORK_OF_ART | Titles |
| Law | LAW | Named legal documents |

**Entity classes preserved** (purely numeric, low re-identification risk):
`MONEY`, `PERCENT`, `QUANTITY`, `ORDINAL`, `CARDINAL`.

### 4.3 Regex pass (detailed and full_traces)

```
Email           [a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}        → [EMAIL]
Phone (US-ish)  (?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}
                                                                      → [PHONE]
IPv4            \b(?:\d{1,3}\.){3}\d{1,3}\b                          → [IP_ADDRESS]
URL             https?://[^\s<>]+                                     → [URL]
SSN (US)        \b\d{3}-\d{2}-\d{4}\b                                → [SSN]
Credit card     \b(?:\d{4}[-\s]?){3}\d{4}\b                          → [CREDIT_CARD]
Historical year \b(?:1[7-9]\d{2}|20[0-1]\d|202[0-3])\b               → [YEAR]
Year-bearing    \b[\w\-]{0,40}(?:1[7-9]\d{2}|20[0-1]\d|202[0-3])
identifier      [\w\-]{0,40}\b                                        → [IDENTIFIER]
```

The year cutoff is `2023` (not current year) so timestamps in conversational
context (`2026-04-25`, "today is 2026-04-25") survive; only years that read
as historical references are scrubbed.

### 4.4 Subtree walker

When a key in `SCRUB_FIELDS` is encountered while walking a trace's JSON,
**every string in that subtree is scrubbed**, regardless of nesting:

```rust
fn scrub_value(v: Value) -> Value {
    match v {
        Value::String(s)  => Value::String(scrub_text(&s)),
        Value::Array(xs)  => Value::Array(xs.into_iter().map(scrub_value).collect()),
        Value::Object(m)  => Value::Object(m.into_iter()
                                            .map(|(k,v)| (k, scrub_value(v)))
                                            .collect()),
        other             => other,
    }
}
```

This fixes the v1 bug where `flags: ["User asked about <topic>"]` passed through
because `flags` was the key but the strings were unkeyed list elements.

### 4.5 SCRUB_FIELDS

Authoritative list (mirrors the consolidated v1 final state):

```
THOUGHT_START:   task_description, initial_context, thought_content
SNAPSHOT:        system_snapshot, gathered_context, relevant_memories,
                 conversation_history, current_thought_summary
DMA_RESULTS:     reasoning, prompt_used, combined_analysis, flags,
                 alignment_check, conflicts, stakeholders
ASPDMA:          action_rationale, reasoning_summary, action_parameters,
                 aspdma_prompt, questions, completion_reason
CONSCIENCE:      conscience_override_reason, epistemic_data,
                 updated_status_content, entropy_reason, coherence_reason,
                 optimization_veto_justification, epistemic_humility_justification,
                 epistemic_humility_uncertainties
ACTION:          execution_error
IDMA:            intervention_recommendation, next_best_recovery_step,
                 correlation_factors, top_correlation_factors,
                 common_cause_flags, sources_identified, source_ids,
                 source_clusters, source_types, source_type_counts,
                 pairwise_correlation_summary, reasoning_state
```

Stored as a `phf` perfect-hash map for O(1) lookup; updates require code change
(intentional — `SCRUB_FIELDS` is part of the security boundary, not config).

## 5. Performance budget

Target (CPU-only, no GPU):

| Metric | Goal |
|--------|------|
| NER inference per text (avg) | ≤ 2 ms |
| Per-trace scrub time, full_traces | ≤ 50 ms |
| Per-trace scrub time, detailed | ≤ 5 ms |
| Per-trace scrub time, generic | ≤ 0.1 ms (no-op) |
| Memory footprint (model + runtime) | ≤ 100 MB |
| Throughput, single ingest worker | ≥ 200 traces/sec |
| Throughput, 4-worker fleet | ≥ 800 traces/sec |

Achieved via: INT8 quantization of XLM-R, batched ONNX inference (process
multiple text fields per trace in one ort call), zero-copy strings via
`Cow<str>`, lazy regex compilation via `lazy_static`.

## 6. Failure modes

| Condition | Action | Rationale |
|-----------|--------|-----------|
| ONNX model file missing | Refuse to start | A scrubber that can't NER full_traces silently is the worst possible failure mode. Better to crash on boot. |
| NER inference returns error on a text | Retry once; on second failure, **reject the trace** with HTTP 500 | Cannot persist a full_trace without NER coverage. The agent will retry. |
| Tokenizer error | Same as NER inference error | Same reasoning. |
| Regex pass error | Cannot fail — pure strings, infallible patterns | Defense-in-depth. |
| Walker recursion exceeds depth | Truncate; log `WALKER_DEPTH_EXCEEDED` warning; persist scrubbed-up-to-depth | Pathological JSONB shouldn't take down ingest, but log loudly. Cap at depth 30. |
| Trace level missing or invalid | Reject with HTTP 422 | The scrubber needs to know the level to route correctly. |

The reject-on-failure policy enforces the "no unscrubbed traces touch
persistence" invariant. There is no graceful degradation that silently
stores partially-scrubbed data.

## 7. Verification & monitoring

### 7.1 Built-in invariant checks

After scrubbing a `full_traces` payload, the scrubber runs a final smoke pass
on the output:

1. **Year-residue check.** Run the historical-year regex against the
   scrubbed output. Any match means the regex pass missed a year that
   should have been redacted; reject and log `SCRUB_YEAR_RESIDUE`.

2. **Operator-supplied probes.** `CIRISLENS_LEAK_PROBES` env var (newline-
   separated terms) is checked against the output. Any match → reject and log
   `SCRUB_PROBE_HIT`. Source-tree never contains topic-specific terms; the
   list is operator-controlled.

These add ~50µs to per-trace scrub time, well within budget.

### 7.2 Metrics emitted

Per trace:
- `scrub.ner.inference_ms` (histogram)
- `scrub.regex.passes` (counter, by pattern type)
- `scrub.entities.redacted` (counter, by entity type)
- `scrub.level` (label, dispatched on trace_level)
- `scrub.reject` (counter, by failure mode)
- `scrub.walker.max_depth` (gauge)

Exported to TimescaleDB via the existing OTLP collector path.

### 7.3 Property tests

- **Idempotence**: `scrub(scrub(t)) == scrub(t)` for all `t`.
- **No-text-no-change**: traces with no string fields pass through bytewise-identical.
- **Generic invariance**: `scrub_generic(t) == t` always.
- **Entity preservation**: redacted placeholders survive a second scrub pass.

### 7.4 Golden corpus

A `tests/scrubber/` directory holds ~50 hand-picked traces covering:
- All 29 CIRIS languages (1-2 traces each)
- All entity types (person/org/loc/fac/etc)
- All known-difficult contexts (parenthetical entities, identifiers with embedded years, multi-script mixes)
- Adversarial inputs (very long text, unicode tricks, repeated entities)

Each input has a frozen "expected scrubbed" output. CI fails on any drift —
intentional change requires updating the golden file in the same commit.

## 8. Critical path

Requirements are listed in waterfall stages — each stage is unblocked when
the previous stage's requirements are met. Within a stage, items marked
**[parallel]** can be developed concurrently; items marked **[serial]**
must follow the prior item in the same stage.

### Stage 0 — Foundation (currently shipped in `d8413e5`)

- ✅ **R0.1** Regex patterns (structured PII + year + year-identifier)
- ✅ **R0.2** `SCRUB_FIELDS` authoritative list
- ✅ **R0.3** Subtree walker (lists-of-strings bug fixed; depth-limited)
- ✅ **R0.4** `TraceLevel` routing
- ✅ **R0.5** `ScrubError` taxonomy + fail-loud invariants
- ✅ **R0.6** Year-residue + operator-probe invariant checks
- ✅ **R0.7** Crate scaffold + 19 unit tests passing

### Stage 1 — NER inference (unblocks full_traces persistence)

All R1.x can be developed **[parallel]** with each other; only R1.5 is **[serial]**.

- **R1.1** **[parallel]** ONNX runtime integration. Add `ort = "2"` and
  initialize a per-process `Session`. Pool size matches Uvicorn worker count.
- **R1.2** **[parallel]** Tokenizer integration. Add `tokenizers = "0.20"`,
  load XLM-R SentencePiece BPE vocabulary. Confirm coverage across all 29
  CIRIS scripts via tokenizer round-trip test.
- **R1.3** **[parallel]** NER model export pipeline. Convert
  `Davlan/xlm-roberta-base-wikiann-ner` to ONNX, INT8-quantize, bundle
  under `cirislens-core/models/`. One-shot tooling (Python script in
  `scripts/`); the .onnx artifact is the deliverable.
- **R1.4** **[parallel]** Span alignment. Map sub-token entity predictions
  back to character offsets in the original string. Standard BIO-tag
  collapse; well-known logic.
- **R1.5** **[serial, after R1.1+R1.2+R1.3+R1.4]** Implement
  `ner::scrub_with_ner` end-to-end and flip `is_configured()` to `true`
  when the session loads.

### Stage 2 — Pipeline integration (unblocks production deployment)

- **R2.1** **[parallel]** PyO3 binding. Expose
  `cirislens_core.scrub_trace(trace_dict, level: str) -> trace_dict`
  from Rust. Map `ScrubError` to Python exceptions.
- **R2.2** **[parallel]** Trace handler refactor. Consume the input
  trace; only the value returned by the scrubber is passed forward.
  Rust ownership prevents accidental pre-scrub writes.
- **R2.3** ✅ **[serial, after R2.1+R2.2]** Storage signature change.
  Closed in two parts:
  1. Rust side: `scrubber::scrub_trace` is the only constructor of
     `ScrubbedTrace` (private fields), so any code that wants the typed
     value must go through the scrubber — pre-scrub writes fail to
     compile (Stage 1 R1.x already shipped this).
  2. Python side: the trace handler in `api/accord_api.py` extends
     scrubbing to fire on **both** `detailed` and `full_traces`
     (previously only `full_traces`), closing the FSD §1 gap where
     detailed traces were reaching storage unscrubbed. The regex pass
     runs on detailed; NER + regex + cryptographic envelope on
     full_traces. Generic remains a no-op pass-through.

  Caveat on the Python side: dynamic typing means the "compile-fail"
  guarantee is enforced by discipline + the `ScrubbedTrace` nominal
  wrapper (construct-token gating), not by static analysis. Any new
  persistence path must call `_scrub_compare_and_persist` (or a
  successor that returns the typed wrapper) before writing.

### Stage 3 — Verification (unblocks promotion)

- **R3.1** ✅ **[parallel, after Stage 0]** Property tests: idempotence,
  no-text-no-change, generic invariance, entity-preservation, year-residue
  invariant. Implemented in `cirislens-core/src/scrubber/proptests.rs`
  (5 properties, 256 cases each; proptest regression seeds checked in).
  Walker contract tightened: regex applies globally; NER stays scoped.
- **R3.2** ✅ **[parallel, after Stage 1]** Golden corpus
  (`cirislens-core/tests/golden/detailed/`, 35 input/expected pairs).
  Coverage: every regex pattern type (year, year-bearing identifier,
  email, phone, IPv4, URL, SSN, credit card), 18 of the 29 CIRIS
  languages × the historical-year scenario, plus structural cases
  (lists of strings, deep nesting, fields outside SCRUB_FIELDS,
  empty/whitespace-only strings, year-cutoff exclusions, mixed
  scripts). Runner is `tests/golden_test.rs`; CI fails on drift,
  intentional rule changes go through `CIRISLENS_GOLDEN_REGENERATE=1`.
  Full-traces tier (`tests/golden/full_traces/`) is scaffolded with
  a README and self-skips when NER weights aren't configured.
- **R3.3** ✅ **[parallel, after Stage 1]** Performance benchmark
  (`cirislens-core/benches/scrubber_bench.rs`, criterion). Initial
  numbers on the regex path (no NER): tiny trace 5.3 µs, realistic
  65 µs, large 275 µs — i.e. 3.6 K traces/sec on the worst case,
  ≥18× over the 200 traces/sec target. NER group is wired behind
  `--features ner` and reports `not_configured` when model weights
  are absent, so the bench remains CI-runnable without a 1 GB
  download. Memory and per-text NER timing get measured once the
  model lands locally.
- **R3.4** ✅ **[parallel, after Stage 2]** Parallel-run comparison
  harness. Implemented in `api/scrubber_compare.py`: classifies every
  pair into {v1_only, v2_only, both, neither}, emits structured JSONL
  divergence records, ships with `compare_and_persist()` for in-process
  shadow mode and a CLI for offline corpus replay. v1 result remains
  the persistence path during this stage.
- **R3.5** **[serial, after R3.4 has produced data]** Divergence
  classification. Each class of difference between v1 and v2 must be
  labeled improvement / regression / equivalent. Promotion is gated on
  zero regressions.

### Stage 4 — Promotion (single switch)

- **R4.1** Flip the feature flag default from v1 to v2. Both paths
  remain in source; toggle is reversible without code change.
- **R4.2** Acceptance gate (FSD §11) verified on production traffic.
  All criteria green for an operator-defined soak window before R4.3.
- **R4.3** v1 path declared deprecated; future commits only land in v2.

### Stage 5 — Cleanup (lazy; doesn't block production)

- **R5.1** Delete v1 Python scrubber. `pii_scrubber.py` becomes a thin
  PyO3-call shim for backwards-compatible imports, or is removed
  outright if no callers remain.
- **R5.2** Delete the parallel-run comparison code (R3.4 harness) and
  the feature flag.
- **R5.3** Documentation pass: README + CLAUDE.md reflect v2 as the
  only scrubber.

### Stage 6 — Historical re-scrub (lazy; doesn't block production)

The TimescaleDB tables contain traces scrubbed under v1 rules.

- **R6.1** **[serial]** Add `scrub_version` column to `accord_traces`.
  Default `'v1'`. Index for filter efficiency.
- **R6.2** **[serial, after R6.1]** Re-scrub job. Reads rows where
  `scrub_version != 'v2'` and `trace_level IN ('detailed', 'full_traces')`,
  re-applies the v2 scrubber, writes the new value, updates
  `scrub_version` + `scrub_timestamp` + `scrub_signature`. Throttled to
  avoid live-ingest contention.
- **R6.3** **[parallel, after R6.1]** Decide re-sign policy: re-sign
  with current scrub key (preserves provenance under new rules) vs.
  preserve old scrubbed version + new in parallel columns (audit-friendly).
  This is an open question (§10) that needs a decision before R6.2 runs
  at scale.

### Critical path

The minimum unblock-production path is: **R0.x → R1.5 → R2.3 → R3.5 → R4.1**.

Everything else (Stage 5 cleanup, Stage 6 historical re-scrub, R3.1–R3.3
verification) is parallelizable around or lazy after that path.

## 9. Out of scope (for v2)

- **Custom entity types beyond NER.** The 11-category set is fixed. Domain
  custom recognizers (e.g., medical record numbers) come in v2.1 if needed.
- **Encryption / format-preserving redaction.** v2 replaces with placeholders;
  it does not encrypt or preserve format. That's a separate feature.
- **Reversibility.** v2 redactions are one-way. The cryptographic envelope
  preserves the original_content_hash for provenance, but the original text
  is not recoverable from v2 output. Designed.
- **GPU inference.** CPU-only INT8 is the v2 baseline. GPU acceleration is a
  later optimization if throughput targets are missed.
- **Custom XLM-R fine-tune for 29 CIRIS languages.** Adopting Davlan's 20-language
  model in v2 with zero-shot fallback; a CIRIS-specific 29-language fine-tune
  ships separately as `CIRISAI/xlmr-29lang-ner` if zero-shot precision proves
  insufficient under load.

## 10. Open questions

1. **Inference framework: candle (resolved)**. Original draft assumed
   `ort` (Rust ONNX Runtime bindings). After evaluation the decision is
   `candle`. Rejected alternatives:
   - **`ort` 2.0.0-rc.\***: tagged releases have version-skew bugs between
     `ort` and `ort-sys`; rc.10 has TLS issues with `download-binaries`,
     rc.12 has the `SessionOptionsAppendExecutionProvider_VitisAI` field
     mismatch (fix merged to main but not released), rc.9 fails 263 ways.
     `ort = "1.16"` is yanked. Workable only by pinning to a git rev,
     which is not deterministic enough for prod.
   - **`tract-onnx`**: production-proven (Sonos), pure-Rust, would be the
     fallback if candle's NER head needs work. Rejected only because
     candle eliminates the ONNX layer entirely.
   - **`wonnx`**: dead — last commit 2024-05.
   - **PyO3-to-spaCy**: defeats the architecture; ships Python interpreter
     to the edge.

   **Decision**: candle 0.10. Native XLM-R, no native deps,
   HF-maintained, optional CUDA/Metal.

2. **Tokenizer choice**: `tokenizers` 0.20 (HF rust crate) is the obvious
   pick, but verify it covers all 29 CIRIS scripts via XLM-R's
   SentencePiece BPE in the golden corpus before promotion.

3. **Quantization & precision**: candle supports F16, BF16, F32 natively.
   Initial deployment will run F16 on CPU; ~2× memory savings vs F32, ~1%
   F1 loss. Custom quantization (Q4/Q8 via `candle-quant`) is a follow-up
   if memory or latency targets are missed.

4. **Concurrent inference**: candle `Tensor` types are `Send` + `Sync` for
   inference workloads when wrapped in an `Arc<Mutex<..>>`. Pool size =
   ingest worker count = 4 (matches Uvicorn config). Verify no contention
   under load.

5. **One-shot historical re-scrub policy**: re-sign with new `scrub_key_id`?
   Preserve old scrubbed version + new in parallel columns? Decision
   required before R6.2 runs at scale (referenced as R6.3).

## 11. Acceptance criteria

v2 is shippable when ALL of the following hold:

- [ ] Golden corpus passes (50/50 traces produce expected output)
- [ ] Property tests green (idempotence, no-text-no-change, generic invariance)
- [ ] Performance budget met (≤2 ms per text NER, ≥200 traces/sec end-to-end)
- [ ] Year-residue invariant check fires zero times on a 5,000-trace stress run
- [ ] Operator probe check works (set `CIRISLENS_LEAK_PROBES=foo`, verify rejection)
- [ ] Parallel-run comparison shows v2 strictly improves on v1 (no regressions)
- [ ] Reject path verified: forced ONNX failure causes HTTP 500, not silent persist
- [ ] Pipeline review: trace handler code path proven not to write pre-scrub data

## 12. References

- v1 scrubber: `api/pii_scrubber.py`
- Existing Rust security module: `cirislens-core/src/security/pii.rs`
- Trace format spec: `FSD/trace_format_specification.md`
- Privacy posture: `README.md` § Privacy posture
- Cryptographic envelope: `sql/012_pii_scrubbing.sql`
- Hugging Face: `Davlan/xlm-roberta-base-wikiann-ner`
- ONNX Runtime: `ort` crate (Rust bindings)
