# Re-evaluating the RATCHET scorer package values from prod scoring rows

**Status:** proposal, from data. Labels per `CIRISOntology/epistemology.md` §1.
**Data:** class-only export of `capacity:sustained_coherence:v1` — 6,465 attestations, 440 subjects,
2026-08-01 → 2026-10-09, production canonical (ciris-server 0.5.224). Script: `recalibrate.py`.
Machine-readable: `proposed_values.json`.

## Proposed values

| key | proposed | today (server default) | crc-v1 | label |
|---|---|---|---|---|
| `measure` | `n_eff_pr` | `n_eff_pr` | — | keep; conservative, deterministic under re-attestation |
| `gate_semantics` | **rows** | applied to `n_eff` | — | **proved** (units, below) |
| `sample_gate_rows` | **20** | 20 | 500 | **measured** — knee |
| `min_feature_dim` | **4** | none | — | **measured** — coverage confound |
| `n_eff_floor` | **1.0** | none (gate stands in) | — | **proved** — rank 1 is one direction of variance |
| `target_n_eff` | **4.5** | 8.0 | — | **measured** p95 of plateau; as a saturation *policy*, **wager** |
| `window_rows` | 500 | 500 | 500 | unchanged — never binds (max N observed 156) |

**Withdrawn:** `target_n_eff = 8.0` (above every one of 6,465 observations; max `n_eff_pr` 5.31,
max `n_eff_h` 6.18, and `FEATURE_DIM = 11` bounds it) and crc-v1's `sample_size_gate = 500`
(no subject has ever reached it; max is 156).

## What the row actually is

`n_eff_pr` and `n_eff_h` are the participation ratio and entropy perplexity of the eigen-spectrum
of an 11-feature covariance (`src/scorer/n_eff.rs`, a port of RATCHET's `measure_n_eff.py`). They
are an **effective rank**, ≤ `feature_dim` ≤ 11 — *not* Kish's effective sample size, which is a
different `n_eff` in lens-core that the scorer does not use.

## The units bug (proved)

`capacity(n_eff, gate, target)` returns 0 when `n_eff ≤ gate`. The gate is defined as a **row**
count (`config_reconcile.rs`: "measure_n_eff.py refuses fewer than 20 rows"). A rank bounded by 11
is tested against 20. **Every one of the 6,465 signed scores is 0.0**, and the degenerate branch
(`target 8.0 < gate 20`, the case `capacity.rs` documents as a caller bug) reproduces 6,465/6,465.
No subject can score above zero under the defaults on any corpus. Separately, no row floor is
applied before the gate: 1,118 rows carry `n_eff_pr = 1.00` computed from `sample_size = 2`.

This is a formula fix on the server; the package cannot work around it. The proposal assumes the
fix: gate on `sample_size`, band on `n_eff` above `n_eff_floor`, and **Indeterminate** — not a
signed 0.0 — below either gate (LC-AV-18: "never numeric below gate").

## Evidence

**Knee at 20 rows (measured).** Pooled binned median of `n_eff_pr` reaches 90% of its N≥50 plateau at
N≈20 (same for `n_eff_h`). Within-subject, the one prod agent with a long history rises
2.59 → 2.84 → 3.13 → 3.20 → 3.21 over N = 5 → 31 and flattens by N≈26–30. Gate 30 moves the plateau
median 2.96 → 2.89 (negligible) and cuts scoreable subjects 71 → 59. Scores at N 20–30 run a few
percent under the eventual plateau — the conservative direction.

**Plateau (measured, bootstrap over subjects).** Latest attestation per subject, N ≥ 20, n = 56:
`n_eff_pr` median 2.96 [2.84, 3.05], p90 4.14 [3.58, 4.84], **p95 4.48 [3.78, 5.10]**.
`n_eff_h` runs ~1.2× higher (median 3.62, p95 5.55). Re-attestation at fixed N is deterministic
(stdev median 0.0000 over 221 groups) — the 6,465 rows are ~440 subjects re-emitted ~15×.

**Coverage confound (measured).** Among scoreable subjects Spearman(`n_eff_pr`, `feature_dim`) =
**0.74**, Spearman(`n_eff_pr`, `sample_size`) = −0.35; realised rank is a stable ~0.54 of available
rank. The fed quantity substantially measures *how many of the 11 features varied*, and
`feature_dim` is constant within a subject — a property of the deployment. Windows with
`feature_dim ≤ 2` (2,192 rows) have a rank ceiling of 2 by construction. `min_feature_dim = 4`
removes them and costs one scoreable subject (55 of 56 are fd 3–11, mode 5).

**Counterfactual under the proposal (latest per subject, 440).** 385 Indeterminate, 55 numeric:
median 0.56, IQR 0.50–0.72, 3 saturate at 1.0, none at 0. Under 8.0 instead: median 0.28, none
saturate. Under crc-v1's 500: nothing numeric.

**Prod agents vs the rest (measured).** Datum + echo-core + echo-speculative: 26 subjects, 507 rows,
N median 10 / max 31; **3** reach the gate; `n_eff_pr` at N≥20 = 3.15 median, 3.21 max — the
population median, not a low-rank population (curves track at matched N: 2.23 vs 2.08 at N≈10,
3.13 vs 3.35 at N≈20). The high tail setting p95 is unclaimed-bootstrap subjects prod never
matches in row count. Under the proposal the three prod agents score **0.63, 0.50, 0.67** —
constrained, not saturated, on 24–31 rows. 23 of 26 prod agents never reach 20 rows per window;
10 of 26 sit at fd < 4. Owner-bound (real-person) installs: N median 2, max 14 — none scoreable.

## Caveats, stated rather than rounded

- **Calibrated on a battery-shaped population.** 53 of the 56 subjects behind the plateau are
  unclaimed-bootstrap, which the canonical cannot separate into batteries vs real unclaimed
  installs. Zero owner-bound and zero qa-runner subjects reach the gate. Re-estimate once
  person-claimed installs accumulate 20-row windows.
- **Mock-LLM pollution.** Windows that included mock traces were scored (CIRISPersist#1040 purge
  pending); this export cannot identify which rows.
- **The target is a policy.** 4.5 is the p95 of what has been observed; choosing "top 5% saturate"
  is a decision, not a measurement. Its CI [3.78, 5.10] is the honest width on 56 subjects.
- **Normalised alternative.** Feeding `n_eff_pr / feature_dim` (median 0.54, 18% ≥ 0.60 among
  scoreable) with target 0.60 removes the coverage confound outright, at the cost of a formula
  change; recorded here as the principled next form, not proposed now.

## What this does not establish

Nothing here tests whether effective rank predicts anything. The score is a band of a measured
quantity; its relation to fragility or deception resistance remains **open**, and `detection:*`
— the only independent signal — has produced no rows.
