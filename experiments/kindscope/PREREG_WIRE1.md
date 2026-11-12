# WIRE-1 — does mlp0's final-bytes lookup run on attn0/h3, the previous-character head? (preregistration)

Frozen 2026-11-12, before any WIRE-1 quantity is computed.

misfits: KS-DIRECT-ONLY, KS-EXACT-ZERO-ACROSS-BATCH, KS-CASEFOLD-DONOR, KS-PRICE-NOT-COST, G-SECOND-READ, M-PLANT-OBS, M-FLOOR-UNSTAKED, M-BAR-FROM-THE-READ, M-ONE-MODEL-DELTA, M-DEVICE-CLASS

plants: P-NULL-ABLATION carrier: GoldWorm v6 base — the ablation machinery with no head zeroed must reproduce the base mlp0 writes bit for bit (same batch layout), or the run is VOID

## Known at freeze

MLP0-1/2/4: mlp0 writes word identity at a word's last byte, keyed on its final bytes (null-adjusted η² by last 3
bytes ≈ by whole form; patching held, +0.258 [0.255, 0.261]). `explore_heads.json` (8 unseen stories, exploratory):
attn0/h3's strongest source is the previous character at 90% of positions. Hypothesis **H-WIRE**: mlp0's suffix lookup
reads the preceding bytes through attn0/h3.

## Data

FRESH-1 (sha256 `3232df12…`), kept stories 9,000–11,999 (unread; 0–3,999 PARALLAX-2, 4,000–7,999 MLP0-3,
8,000–8,063 CAL-G, 16,000–19,999 MLP0-4, 20,000–20,007 traces). Tagged as VOCAB-1; raw-byte word forms
(KS-CASEFOLD-DONOR). Tokens: word length ≥ 4, form with ≥ 10 occurrences; site = the word's last byte.

## Intervention

Zero one attn0 head's output (its 48 dimensions of `att @ v`, before the output projection) at every position, run the
model to mlp0, read the mlp0 write at each site. Arms: none (P-NULL-ABLATION), h0, h1, h2, **h3**. Every arm runs in the
identical batch layout.

## Measures and predictions (95% story-bootstrap CIs, 1,000 resamples; groupings null-adjusted as MLP0-2)

For each arm: η²_adj of the mlp0 write grouped by the last 3 bytes (S) and by the first 3 bytes (F).
Drop_arm(S) = η²_adj(S | none) − η²_adj(S | arm), likewise F.
- **W1 (specific head):** Drop_h3(S) − max(Drop_h0(S), Drop_h1(S), Drop_h2(S)) > 0, CI excludes 0.
- **W2 (specific to the suffix):** Drop_h3(S) − Drop_h3(F) > 0, CI excludes 0.
- Descriptive: every arm's η²_adj for S, F and the whole form.

**Outcome:** W1 ∧ W2 ⇒ H-WIRE holds (measured, this checkpoint): the first tested wire, attn0/h3 → mlp0. Either
failing ⇒ not supported; the arcs stay "seen while exploring".

## Outputs

`wire1.py` (committed before its first run), `wire1_result.json`, results in `EDIT_RESULTS.md`.
