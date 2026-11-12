# PARALLAX-2 — category early, bytes late (preregistration)

Frozen 2026-11-12, before any PARALLAX-2 quantity is computed. Theory from `PARALLAX_NOTES.md` and the CAL exploration
`explore_clouds.json` / `explore_play.json` (committed with this file).

misfits: G-SECOND-READ, KS-READ-NOT-WRITE-EARLY, KS-DIRECT-ONLY, KS-EXACT-ZERO-ACROSS-BATCH, M-BAR-FROM-THE-READ, M-FLOOR-UNSTAKED, M-DEVICE-CLASS, M-ONE-MODEL-DELTA

plants: P-SELF carrier: GoldWorm v6 base — a zero push reproduces the base word log-probs within the same-batch floor (identical batch layout)

## Hypotheses

- **H-CATEGORY-EARLY:** at mlp0–mlp2, the part of the adjective writing direction NOT explained by 16 first-byte forecast
  directions (the residual) writes the category: pushing it raises adjectives over nouns **within each first-byte
  group**.
- **H-BYTE-LATE:** at mlp5, the pure next-byte = 'l' direction writes a byte, not a category: it raises nouns starting
  with 'l' and lowers adjectives starting with 'b'.

## Fresh data (FRESH-1) — never read by any kindscope run, never seen by GoldWorm

`roneneldan/TinyStories` `TinyStories-train.txt`, first 20 MiB (bytes 0–20971519), sha256
`3232df12d6bdd8f83071f0d8196ad10128624168e228aaf9a23f6a1f8431ee8a`. GoldWorm v6 trained only on the validation file's
`train_core`; the training file is a separate set. Split on `<|endoftext|>`, drop the first and last fragments, drop
non-ASCII stories and the **100 stories that are exact duplicates of a story in the validation file** (found before
freezing), keep the first 4,000 remaining. Windows, tagging and pairs exactly as EDIT-1 (`<|endoftext|>\n\n` + story,
256 bytes; spaCy 3.8.7 / `en_core_web_sm` 3.8.0; DET followed by an adjacent kept word).

## Directions (fixed from CAL before this freeze)

- Residual at L ∈ {0, 1, 2}: `explore_clouds.json` `layers[L].resid_dir` (committed), unit norm.
- 'l' direction at mlp5: gradient of log p(next byte = 'l') w.r.t. the mlp5 write at CAL positions before a word that
  follows a determiner, first 800 CAL stories — the exact `grad_dir(5, ord('l'))` of `explore_play.py`.
- Sizes as explored: residual ½ × the mean mlp_L write norm at those CAL positions; 'l' ¼ × the mlp5 write norm.

## Measures and predictions (FRESH-1; paired per word; 95% story-bootstrap CIs, 1,000)

Δ = pushed − base log-prob of the word after the determiner (nats; push added at the position before it). Groups by
POS (ADJ, NOUN) × first byte (b, l, other).
- **P1 (H-CATEGORY-EARLY):** for each L ∈ {0, 1, 2} and each first-byte group x ∈ {b, l, other}:
  Δ(ADJ_x) − Δ(NOUN_x) > 0 (difference of story-bootstrapped means, stories resampled jointly), CI excludes 0 —
  9 cells. Holds at a layer if all three groups pass there; holds overall if it holds at all three layers. A group
  with < 30 words in either class is VOID for that cell.
- **P2 (H-BYTE-LATE):** at mlp5: Δ(NOUN_l) > 0 and Δ(ADJ_b) < 0, each CI excluding 0, and
  Δ(NOUN_l) − Δ(ADJ_other) > 0 with CI excluding 0.
- Plant P-SELF: zero push |Δ| ≤ 1e-6 everywhere, else VOID.

## Outputs

`parallax2.py` (committed before its first run), `parallax2_result.json`, results appended to `PARALLAX_NOTES.md`.
