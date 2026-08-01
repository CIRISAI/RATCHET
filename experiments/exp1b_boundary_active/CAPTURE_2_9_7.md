# Trace capture on agent 2.9.7 — what changed, and how to run it

Written 2026-08-01, validated against real captures before being written down.
Supersedes the capture half of `run_crossfamily_5vendor.sh` for any **new** campaign
(TORQUE, RATCHET#16). The locked 3-vendor and 5-vendor cohorts are unaffected — the
loader still reads their format.

## The short version

```bash
docker run --rm -e API_KEY=sk-... -v "$PWD/traces:/out" \
  ghcr.io/cirisai/ciris-research-capture:2.9.7
```

Then point the measurement pipeline at `./traces`. Nothing else is required; the image
manages trace levels internally.

## Key on `ceg-seal-*.json`, not `lens-batch-*.json`

A capture directory carries **two artifact families with different guarantees**:

| File | Written | Guarantee |
|---|---|---|
| `ceg-seal-*.json` | on seal | **always present** — survives an unreachable canonical |
| `lens-batch-*.json` | only when a batch ships | absent whenever the canonical is unreachable |

Keying on `lens-batch-*` silently under-counts exactly when the network is worst, and the
under-count is invisible: you get a smaller cohort, not an error. `measurement.py` reads
`ceg-seal-*` and deliberately ignores `lens-batch-*`.

## What actually changed from the pre-2.9.7 harness

**1. `CIRIS_ACCORD_METRICS_TRACE_LEVELS` (plural) is dead.** Nothing reads it. The old
harness exported `detailed,full_traces` and that is now a **silent no-op** — the run appears
to work while the operator believes they are controlling trace levels. 2.9.7 has a singular
`CIRIS_ACCORD_METRICS_TRACE_LEVEL`, which `qa_runner` sets itself (`generic` at startup, with
`detailed` and `full_traces` registered after auth for model_eval). **Do not set either
variable by hand.** Remove it from any adapted script.

**2. The envelope moved; the trace did not.** Captures are now federation-attestation rows:

```
ceg-seal-*.json → ceg_rows[] → attestation_envelope → trace → components[]
```

against the old `*accord-batch-*.json → events[] (event_type == "complete_trace") → trace →
components[]`. The **inner `trace.components` structure is unchanged**, so all sixteen
projection features extract exactly as before. `measurement.py` accepts both shapes.

**3. `trace_schema_version` is now `3.0.0`** and `dimension` is `trace:complete:v1`.

**4. The adapter relocated** from `ciris_engine/logic/adapters/` to a top-level
`ciris_adapters/ciris_accord_metrics/`. Only matters if something imports it directly.

## Captures are self-attesting

Each sealed row carries a PQC signature (ML-DSA-65) plus a classical signature, an
`attesting_key_id`, and a `persist_row_hash`. Provenance is therefore **cryptographically
checkable rather than trusted**, which matters twice for a pre-registered campaign:

- the received-numbers gate can be *satisfied* rather than asserted — a cohort's traces are
  demonstrably from the declared agent identity;
- **TORQUE arm D** requires that a torque reading never reach a hidden-arm agent context,
  "verified by trace audit, not by intention". That audit now runs over signed artifacts, so
  tampering with the record being audited is detectable.

`measurement.attestation_provenance(dir)` surfaces the signing identities, digests and schema
version per row. It deliberately does **not** verify signatures — this module owns
measurement, not crypto. Verify with the agent-side tooling that holds the public keys, then
key the cohort on rows that pass.

## The empty-cohort guard

`load_chains_from_tee_dir` now raises on a directory that exists but yields zero chains,
rather than returning `[]`. The capture format has already moved once; a silent empty return
would have every downstream statistic computed over nothing and reported as a clean run.
`strict=False` disables it where an empty result is legitimately expected.

This is also the mechanical form of the TORQUE void condition *"the live-agent harness cannot
produce the pre-registered probe counts — declare, don't shrink stakes silently."*

## Validation performed before writing this

| Check | Result |
|---|---|
| Loader on 2.9.7 `ceg-seal` captures | 8 chains across 4 directories, 0 excluded |
| Required event types present | all 7 (`THOUGHT_START` … `ACTION_RESULT`) |
| Feature extraction | **16/16** on chains where faculties fired |
| Conditional-field behaviour | absent only when `n_fired = 0` — correct, not drift |
| Empty-cohort guard | fires on a genuinely empty directory |
| Attestation surface | 2/2 rows PQC-signed, schema `3.0.0` |

## Open item for the campaign

Pin the image by digest, not by tag, in the TORQUE Stage 0 record. `:2.9.7` and `:latest` are
both mutable; a pre-registered campaign should record the digest it actually ran against so
the instrument is identified as precisely as the design.
