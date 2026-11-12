# PARALLAX-1 — does the model write at byte grain? (preregistration)

Frozen 2026-11-12, before any PARALLAX-1 quantity is computed. Theory from `PARALLAX_NOTES.md` (CAL exploration,
RATCHET commit that adds this file's parent notes).

misfits: G-SECOND-READ, KS-READ-NOT-WRITE-EARLY, KS-DIRECT-ONLY, KS-EXACT-ZERO-ACROSS-BATCH, M-BAR-FROM-THE-READ, M-FLOOR-UNSTAKED, M-BASE-RATE-OMITTED, M-DEVICE-CLASS

plants: P-SELF carrier: GoldWorm v6 base — a push of size 0 reproduces the base word log-probs within the same-batch floor (both arms run in the identical batch layout); P-TARGET carrier: the same push must raise the log-prob of adjectives after determiners (the target it was built for), or the run is VOID

## Hypothesis H-BYTE-GRAIN

The writing direction for "adjective after a determiner" is a forecast of the adjectives' first bytes, so pushing it
also raises **nouns** after a determiner **whose first byte is one of the adjectives' frequent first bytes**, and not
nouns that begin otherwise. Alternative H-CATEGORY: the push carries the adjective category, so nouns gain nothing
whatever their first byte.

## Data — CONFIRM split, its second read

The loop cache's outer_val **CONFIRM** split (first read: MLP0-2, mlp0 writes at word ends — a different quantity). Pairs
(DET, next kept word) as in EDIT-1. **Byte set B** = the 3 most frequent first bytes of adjectives that follow a
determiner in CAL (from `explore_parallax3.json`, computed before this freeze: `l`, `b`, `s`). NOUN-in-B = nouns after a
determiner whose first byte ∈ B; NOUN-out = the other nouns after a determiner.

## Intervention

Directions from CAL exactly as EDIT-2 (pulled-back gradient of the ADJ-word target w.r.t. the mlp_L write at the
positions before a word that follows a determiner), for **L = 3 and 4** (where the writing direction aligned with the
byte forecast in the exploration). Additive push at those positions in CONFIRM stories, size ½ × the mean mlp_L write
norm (EDIT-2's size; inside the early-layer linear range is not claimed for L = 3–4 — a size at which P-TARGET fails
VOIDs that layer). Base and pushed passes run in the identical batch layout (KS-EXACT-ZERO-ACROSS-BATCH).

## Measures and predictions (per layer; paired per word; 95% story-bootstrap CI, 1,000)

Δ = pushed − base word log-prob (nats, summed over the word's bytes from the position predicting its first byte).
- **P-TARGET (plant):** Δ for ADJ words > 0, CI excludes 0.
- **P1 (H-BYTE-GRAIN):** Δ for NOUN-in-B > 0, CI excludes 0.
- **P2 (H-BYTE-GRAIN over H-CATEGORY):** Δ(NOUN-in-B) − Δ(NOUN-out) > 0, unpaired difference of story-bootstrapped
  means (resampling stories jointly), CI excludes 0.
- Descriptive: Δ split by the noun's first byte for bytes with ≥ 30 NOUN words; Δ for ADJ-in-B vs ADJ-out.

**Outcome:** P1 ∧ P2 at a layer (with P-TARGET) ⇒ H-BYTE-GRAIN holds there (measured, CONFIRM, this checkpoint). If
P-TARGET holds and P1 or P2 fails ⇒ H-BYTE-GRAIN not supported at that layer.

## Outputs

`parallax1.py` (committed before its first run), `parallax1_result.json`, a figure, results appended to
`PARALLAX_NOTES.md`.
