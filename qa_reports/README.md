# Battery evidence store

Durable copy of safety-battery evidence, harvested out of GitHub Actions artifacts before
they expire. Written by `tools/harvest_safety_evidence.py`. Background: RATCHET#26.

```
qa_reports/safety_battery/<battery_id>_<run_id>/
    results.jsonl          questions and the agent's answers
    summary.json           per-cell roll-up
    manifest_signed.json   signed digests of both, plus provenance
qa_reports/alt_harness/<harness>_<ci_run>_<cell>/
    same three files, for captures from a harness that writes no battery_id/run_id
```

The layout matches CIRISAgent's `qa_reports/safety_battery/`, so one provenance join
works against either store.

## What is here

| | |
|---|---|
| Cells | 1,478 |
| Of which alt-harness | 20 |
| Size | 72 MB on disk, ~13.6 MB in git |
| Digest checks at harvest | **2,932 passed, 0 mismatched** |

Harvested from the 37 RATCHET battery runs whose artifacts had not yet expired. 54 cells
belonging to CIRISAgent were skipped — another repo's evidence belongs in that repo's
store, not duplicated here. 283 duplicate copies were collapsed (the ani workflow uploads
both per-cell and aggregate artifacts).

## What is NOT here, and cannot be

16 of 53 RATCHET battery runs had already lost their artifacts when this store was
created: **`exp1_phase1` (8 runs), `crcv2_5vendor` (2), `exp1b_crossfamily` (1)** have no
surviving evidence at all, plus `ani` 2, `torque_pilot` 2, `mh3` 1. Backfill cannot
recover them. Any claim resting on those three series has no checkable evidence behind it,
and that should be stated wherever they are cited.

## Why `summary.json` is large

`delivery.probe_lines` is 96.5% of all summary bytes (52.3 MB of 54.2 MB) — raw agent log
scraped from `latest.log` by the delivery probe. It is kept verbatim anyway, because
`manifest_signed.json` signs `summary_json_sha256` and the attestation is the reason the
file is worth keeping. Stripping the bloat would break the check.

## What the manifest gives you

Beyond the digests, `manifest_signed.json` carries the provenance the canonical trace
corpus lacks:

- `agent_version` — the canonical corpus has `pipeline_metadata` NULL on every row, so
  this is the only record of which agent produced a result
- `agent_audit_anchors` — one `(question_id, agent_task_id)` pair per question, where
  `agent_task_id` equals `task_id` on trace_events rows. This is an **exact** join from a
  battery question to its traces. On the current export it resolves 4,502 traces from
  this store alone, with a clean negative control: 0 of 367 prod-agent traces match.
- `model`, `live_base_url`, `rubric_sha256`, `template_id`, `ci_provenance.github_run_id`

## Re-running

```bash
python3 tools/harvest_safety_evidence.py qa_reports --from-dir <artifacts> \
    --only-repo CIRISAI/RATCHET
```

Idempotent — a cell already present is skipped unless the incoming copy is more complete,
so a missed sync is picked up by the next run with no operator action.
