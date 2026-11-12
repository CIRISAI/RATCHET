# CROSS-1 — does a change of category move the state across the map? (preregistration)

Frozen 2026-11-12, before any model state is computed on the cross-category pairs.

misfits: KS-READ-NOT-WRITE-EARLY, KS-PRICE-NOT-COST, KS-NULL-EXPECTATION, KS-CONTROL-BASE-RATE, M-PLANT-OBS, M-BAR-FROM-THE-READ, M-FLOOR-UNSTAKED, M-BASE-RATE-OMITTED, M-DEVICE-CLASS, G-SECOND-READ

plants: P-FLOOR carrier: GoldWorm v6 base, the 156 same-category real pairs of `map_motion.json` — their alignment with a category axis they do not cross (a randomly drawn other category, 200 draws) must have a mean within its own bootstrap interval of 0, or the reading is VOID

## Known at freeze

`map_motion.json` (RATCHET `3a6bca0`): the map's category directions at a word's last byte (fitted on outer_val's train
split) and the reading that same-category changes carry only 1–4% of their squared norm inside the map's span. The
cross-category pairs (`cross_pairs.json`, mined and COUNTED only: 173 real openings from FRESH-1 differing in exactly one
word whose two words the tagger gives different parts of speech; AUX/VERB 98, ADJ/NOUN 36, ADJ/ADV 14, ADV/PROPN 11,
other 14; 21 of equal byte length) have not been run through the model.

## Measure

For a pair whose words are of categories A and B: s_a, s_b = the state at each member's own changed word's last byte
(the site the map was fitted at), at each of the 13 states. Alignment = cosine between (s_a − s_b) and (d_A − d_B), with
d the map's **mean-difference** directions at that state (the probe directions are reported beside it, not staked:
a reading direction is not where the model writes). Device: CPU (as the map).

## Predictions (95% bootstrap over pairs, 2,000)

- **X1:** at every state from mlp0 to mlp5 (12 states), the mean alignment of the 173 cross-category pairs is positive
  with an interval excluding 0.
- **X2:** at those states it exceeds the same-category floor: mean(cross) − mean(|floor draw means|, 200 draws) > 0, with the
  cross interval's lower bound above the floor draws' 95th percentile.
- Descriptive: by category pair (AUX/VERB, ADJ/NOUN, the rest); the 21 equal-length pairs alone (position not shifted);
  the embedding state; alignment under the probe directions; and the share of the cross-category difference inside the
  map's span beside the same-category 1–4%.

Limits stated now: tagger labels on single words; 57% of pairs are AUX/VERB; in unequal-length pairs the two last bytes
sit at different positions (a small position term enters the difference); GoldWorm only.

## Outputs

`cross1.py` (committed before its run), `cross1_result.json`; the pairs' projections appended to the map-and-motion table
for the explorer.
