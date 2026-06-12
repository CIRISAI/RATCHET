# crc-v2 — Axis-family calibration package (F-3 + distributive)

Closes RATCHET#2 (Tier-1 + Tier-2 axes), RATCHET#3 (distributive + ecology_of_communication where data permits), RATCHET#5 (the umbrella ticket).

## What this bundle ships

8 axes calibrated (4 Tier-1 full, 4 Tier-2 proxy), 5 axes deferred to Tier-3 pending CIRISAgent-side substrate emissions.

| Tier | Axis | Threshold | Floor | Status |
|---|---|---|---|---|
| 1 | `distributive:access:compute` | Gini ≥ 0.170 | 12 agents × 1000 events × 30 days | Full calibration |
| 1 | `distributive:access:models` | HHI ≥ 1.0 | same | Full calibration (conservative — fires only at full model dominance) |
| 1 | `distributive:access:federation_membership` | nonmember_frac ≥ 1e-6 | same | **zero_variance_baseline** — production currently 100% federated |
| 1 | `correlated_action:rights_asymmetry` | PDMA-conflict rate ≥ 1e-6 | same | **zero_variance_baseline** — production currently 0 conflicts |
| 2 | `correlated_action:participation_exclusion` | below_median_domain_breadth ≥ 0.444 | same | Proxy (CIRISAgent#XX needed for full evidence) |
| 2 | `correlated_action:informational_asymmetry` | per-agent flag-count CV ≥ 0.707 | same | Proxy |
| 2 | `correlated_action:aggregate_footprint` | actions × log10(N) ≥ 619 | same | Proxy |
| 2 | `distributive:access:agent_capabilities` | cap_diversity ≤ 0.037 | same | Proxy |
| 3 | `correlated_action:ecology_of_communication:*` (4 axes) | — | — | Deferred — no inter-agent messaging in current substrate (CIRISAgent#YY) |
| 3 | `distributive:access:training_data` | — | — | Deferred — no training-data resource events in current substrate (CIRISAgent#ZZ) |

## How the calibration was derived

Source: `0612_prod_traces` (902MB dump, 25 tables, 10.5M rows, 91 days of substrate coverage).

1. **Pass A (per-agent aggregate)**: per `agent_id_hash`, count events by type, sum LLM cost/tokens, accumulate PDMA conflict counts, DSDMA domain/flag distributions, federation-membership flag (via `accord_public_keys.key_id` prefix join). Regex-based recovery for malformed LLM_CALL rows with double-escaped prompt payloads (14.4% of trace_events).
2. **Pass B (cohort derivation)**: k-means on 8 z-scored behavioral features → 11 cohorts of 168 qualifying agents. 7 cohorts pass the joint floor (≥12 agents AND ≥1000 events). Per-cohort metric computation for each of the 8 axes.
3. **Pass C (threshold setting)**: 75th-percentile of observed cohort metrics, bootstrap CI at 2000 resamples. Zero-variance axes get `1e-6` lowest-detectable-signal baselines with `zero_variance_baseline` outcome flag.

Pipeline scripts: `scripts/01b_aggregate_agents_v2.py`, `scripts/02_cohorts_and_thresholds.py`, `scripts/03_write_bundle.py`. Reproducible from the dump SHAs in `bundle.yaml`.

## Statistical floors (canonical)

Per RATCHET#2 spec, every axis carries the same floor:
- `min_cohort_size_events`: 1000
- `min_goal_aligned_cluster_size_agents`: 12
- `min_window_days`: 30
- `power_target`: 0.95

At the current corpus, **7/8 axes meet floor** with cohort median (15 agents × ~4400 events × 91-day window). LensCore deployments must compute cohort delineation against `agent_id_hash` (NOT `channel_id`, which is source-scrubbed).

## Polarity convention

- `positive_when_distributed`: high values = good (broad access, low concentration). Negative attestation emitted when the cohort exceeds the concentration threshold.
- `positive_when_detected`: high values = concern (concentrated pattern present). Negative attestation emitted when the cohort exceeds the concern threshold.

Each axis pins this in `threshold_function.polarity`.

## Known issues

1. **`unknown`/`[IDENTIFIER]` bucket (14.2% of events)** — excluded from cohort statistics; not a cohort, not measurable.
2. **`channel_id` source-scrubbing (CIRISLens#12 carry-over)** — cohort delineation via `agent_id_hash` only.
3. **JSON parse failures (14.4% trace_events, 16.2% trace_llm_calls)** — regex recovery worked for trace_events (99.4% net coverage); trace_llm_calls residual ~16% drop, cost aggregates from embedded LLM_CALL events backfill.
4. **Zero-variance axes (federation_membership, rights_asymmetry)** — current production shows no within-corpus variation; thresholds are sentinel until variance accumulates.
5. **Tier-2 proxy axes** — rely on derived features rather than the canonical evidence-shape fields. CIRISAgent-side tickets filed to upgrade Tier-2 → Tier-1 once those fields populate.
6. **Tier-3 deferred (ecology_of_communication, training_data)** — no substrate emission path. CIRISAgent-side tickets filed to introduce.

## Bundle layout (matches crc-v1)

```
crc-v2/
├── bundle.yaml         — the calibration content
├── bundle.sha256       — hash of bundle.yaml (LensCore evidence_refs[] target)
├── bundle.signing.txt  — signature scaffold (signing process per release protocol)
├── bundle.cbor         — binary serialization for runtime consumption
├── README.md           — this file
├── scripts/            — reproducible derivation pipeline
└── data/               — intermediate artifacts (cohort metrics, per-axis JSON, aggregate JSONL)
```

## Consumer contract

- LensCore detector emits attestations on `detection:correlated_action:{axis}` / `detection:distributive:access:{resource_type}` with `evidence_refs[]` carrying:
  - `crc-v2:bundle.sha256:` + the hex hash from `bundle.sha256`
  - `0612_prod_traces:trace_events.jsonl.gz:` + the hash from `bundle.yaml::calibration_corpus.trace_events_sha256`
  - Cohort delineation artifact (cohort_id + member list, OR derivation-algorithm-version)
  - Axis-specific evidence fields per `bundle.yaml::axes.{axis}.evidence_required`
- Per CEG §15.2 R2: during crc-v2 → crc-v3 transitions, LensCore emits both bundle hashes in `evidence_refs[]` to defeat straddle attacks.
- Per CEG §4.6/§4.9: `detection:*` attestations are NEVER sole evidence for `slashing:*`. WA quorum is the load-bearing gate.

## Re-derivation

```bash
cd /home/emoore/RATCHET/release/calibration/crc-v2/scripts
python3 01b_aggregate_agents_v2.py    # → data/agents.jsonl
python3 02_cohorts_and_thresholds.py  # → data/cohorts.jsonl + data/axes/*.json
python3 03_write_bundle.py            # → bundle.yaml + bundle.sha256
```

Dump must be at `/home/emoore/0612_prod_traces`. Outputs are deterministic modulo the k-means seed (20260612).
