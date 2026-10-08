# Capacity rows — analyzing the CEG scoring plane from prod directly

RATCHET's first analysis of CEG records as the product of lens-core post-fold, rather than
of a trace dump. Labels per `CIRISOntology/epistemology.md` §1. Everything below marked
**measured** is a class-only, read-only count from the canonical relayed by the bridge
session on 2026-10-08; no identities or keys were shared.

## What exists (measured, canonical, 2026-08-01 → 2026-10-08)

**One dimension has rows.** `capacity:sustained_coherence:v1` — 6,315 rows, 440 subjects,
attested by the canonical.

| Subject class | Rows | Subjects |
|---|---|---|
| unclaimed bootstrap install | 5,148 | 319 |
| registered research agent | 504 | 26 |
| owner-bound install (person-claimed) | 335 | 32 |
| qa-runner test identity | 328 | 63 |

**Zero rows** for `capacity:composite`, `core_identity`, `resilience`, `integrity`,
`incompleteness_awareness`, and for **every** `detection:*` dimension.
`detection:conscience_override_rate` is an enum label only — no detector emits it
(CIRISAgent#1247 records the denominator semantics to adopt when it is built). The four
empty capacity factors are empty because their per-factor derivation is RATCHET's
calibration deliverable (`capacity/score.rs`: "RATCHET's calibration package supplies the
per-factor derivation"), and crc-v2 calibrates detection axes, not capacity factors.

## What a row is (proved — it is code)

```
n_eff    = N / (1 + ρ(N − 1))                                   Kish; scoring/n_eff.rs
capacity = clamp01( (n_eff − gate) / (target_n_eff − gate) )     scoring/capacity.rs
```

Envelope fields: `asserted_at`, `attested_key_id`, `attesting_key_id`, `cohort_scope`,
`dimension`, `valid_until`, `feature_dim`, `n_eff_h`, `n_eff_pr`, `sample_size`,
`sample_size_gate`, `score`, `target_n_eff`. **No `evidence_refs`** — no `trace_id`,
`task_id` or chain hash. `target_n_eff` is a RATCHET calibration parameter.

## The trap, stated up front

The score is a linear band of the Kish identity in ρ. Comparing capacity rows against ρ,
or against anything derived from ρ, is the C-5 / C-11 identity check and is not
evidence. The only independent test of the **open** claim — does ρ predict fragility —
would be `detection:*` events against capacity, and there are none.

What the rows do support, and is not circular:

1. **Saturation under subsampling** per cohort — Gate 0's corridor measure. Needs
   `n_eff_h`, `n_eff_pr`, `sample_size` per row; all present.
2. `n_eff_h` vs `n_eff_pr` divergence by class.
3. Score distribution by subject class, joined at **class level only** via
   `agent_id_hash_class` + time window. Nothing finer is possible without
   `evidence_refs`.

## Consent model (measured — final, after two prior revisions)

There is **no consent-gate gap**. Every scored subject has a valid `analyze` grant
targeting the canonical, admitted before its score. A subject is validly scored if any of
these holds (scope `analyze` or `analyze:capacity`; any Granted wins):

1. its own `consent:state:granted:v1` rows targeting the attester;
2. rows by its `delegates_to` user stewards with `for_key_id = subject`;
3. rows by its occurrence identity anchors (user keys) with `for_key_id = subject`.

Two earlier passes (one by the bridge, one by the server) reported 40, then 11, ungranted
subjects; both had missed paths 2 and 3. CIRISPersist#1041 will put the governing grant
id into each score envelope so this is checkable per row rather than by scan.

The scored population is ~73% unclaimed-bootstrap, which mixes batteries and real
unclaimed installs. Only 32 person-claimed installs are scored. Read: the scoring plane
today is scoring a battery-shaped population.

## Getting the rows

`capacity:*` is federation-tier. There is no fleet-wide read endpoint
(`/v1/my-data/capacity` is the per-person owner view; `/lens/api/v1/scores` and
`/detection_events` are trace-score reads). Options:

| Path | What it needs | When it's right |
|---|---|---|
| one-shot class-only read | owner's go | now — 6,315 rows, one dimension |
| replication | `consent:replication` from the canonical owner to a RATCHET node key, directional, plus trust root | a standing cadence against prod |
| owner-gated analytics export | a new CIRISServer surface | only if the one-shot proves insufficient |

Pending: owner's go on the one-shot read.

## Order of operations

Join provenance **before** any scrub. `STRUCTURAL_IDENTIFIER_KEYS` exists because the
year regex ate ~5% of `agent_id_hash` values (CIRISLens#11); the provenance split is
keyed on `agent_id_hash`, `task_id` and `channel_id`.
