# PARALLAX-H — Holon's removability gate on the parallax dictionary (preregistration)

Frozen 2026-11-12, before any PARALLAX-H quantity is computed. Uses CIRISHolon's `admit_file` gate (commit `028228b`,
exported read-only to the scratchpad and built there; its three plant tests pass on that build: 3 run, 3 ok).

misfits: KS-READ-NOT-WRITE-EARLY, KS-DIRECT-ONLY, G-REBUILT-AFTER-REMOVAL, M-PLANT-OBS, M-BAR-FROM-THE-READ, M-STALE-INSTRUMENT, M-ONE-MODEL-DELTA, M-DEVICE-CLASS

plants: P-GATE carrier: the gate's own plants at 028228b (two carriers of one thing, a noisy reading, a malformed file) — they must pass on the build used, or the run is VOID

## Question

Ordered admission (backward elimination with a joint price) over ONE dictionary per MLP layer: which directions does
the final next-byte distribution need? Holon's reading: a direction the gate carries is one the target cannot be
predicted without, from this dictionary; the price is a prediction to be realised by an intervention.

## Data

GoldWorm v6 base. Directions fixed from CAL stories 0–799 (as `explore_clouds.py` / `explore_play.py`). Gate rows from
CAL stories 2,000–2,399 (not used to fit any direction): positions before a word that follows a determiner, any next
POS. Units: 5 groups of those stories (story index mod 5). **Kept:** constant, position / 256, one-hot of the 5 most
frequent determiner forms plus "other". **Target:** the final next-byte log-probabilities at the position (256 columns).
**Candidates** (projections of the state after mlp_L onto unit directions): `probe` (1: ADJ-next probe, Newton, CAL
0–799); `write` (1: pulled-back gradient of the ADJ-word target); `resid` (1: `explore_clouds.json` residual); `bytes16`
(16: per-byte forecast directions for the 16 most frequent next first bytes); `shuf1`–`shuf3` (1 each: probes trained on
permuted labels). Gate settings: `budget 0.02` (the tool's default), `folds units 5`, `lag 0`, `horizon future`,
`ridge order1`, `null permute 7`.

## Predictions (stated before the run; odds are mine)

- **H1** `probe` is DROPPED in the ordered admission at every layer (parallax: read ≠ written). ~70%.
- **H2** `shuf1`–`shuf3` are DROPPED at every layer (null sanity). ~90%.
- **H3** `bytes16` is CARRIED at mlp4 and mlp5 (late = byte forecast). ~85%.
- **H4** `resid` is CARRIED at mlp0–mlp2 (early category writer). ~45% — an intervention finding, and the gate is
  observational; I expect this one may fail and want to see it.
- `write` vs `bytes16 + resid`: descriptive (they are nearly collinear by construction; the ordered form should keep one).

## Realisation (Holon's ledger row: predicted price beside realised cost)

For each candidate and layer: project its directions out of the state after mlp_L at the gate positions
(mean-preserving, as VOCAB-1's amnesic test), run forward, Δ next-byte NLL at the position. Reported: Spearman between
the gate's ordered price and the realised cost across candidates, per layer. A carried candidate whose removal costs
nothing, or a dropped one whose removal costs, is a mismatch row for the misfit registry.

## Outputs

`parallaxh.py` (committed before its first run), the SPEC files' sha256, the gate's raw output, `parallaxh_result.json`.

## Amendment 1 (2026-11-12, before any quantity was computed)

The frozen gate-row range, CAL stories 2,000–2,399, does not exist: CAL holds 1,884 stories. The first launch built no
rows and stopped at its own empty-input error; nothing was read. Gate rows are moved to **CAL stories 800–1,599**, which
meet the stated condition (not used to fit any direction — directions come from 0–799; 800–1,599 were read once by
`explore_play.py` for push effects, an exploratory reading of a different quantity). Also corrected for the record:
`explore_modifier.py`'s range "1,600–2,399" was in effect 1,600–1,883. Nothing else changes.
