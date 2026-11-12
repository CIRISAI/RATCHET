# CROSS-1 results — a change of category does move the state across the map, modestly, from the first block on

Preregistration `PREREG_CROSS1.md` (RATCHET `2c1b2ba`); script `cross1.py`; raw `cross1_result.json`. **Measured:**
GoldWorm v6 base, CPU, 173 real one-word-apart openings whose two words differ in part of speech; alignment = cosine
between the state difference at the changed word's last byte and the map's (category A − category B) mean-difference axis.

**By the letter: X1 holds and X2 holds** at all 12 states from attn0 to mlp5 (pooled alignment +0.39 to +0.50; floor
from 156 same-category pairs against an axis they do not cross: 95th percentile +0.01 to +0.05; plant floor at 0).

**The pooled figure is not what it looks like (KS-ONE-WORD-PAIR, registered).** 98 of the 173 pairs are AUX/VERB, and
94 of those are one word pair, "lived" / "was". Its alignment is +0.76 already at the embedding (the "auxiliary" axis
there is close to the embedding of a handful of words), so the pooled number and its interval describe that word pair.

**What the distinct pairs show:**

| state | ADJ/NOUN (36 pairs) | all other category pairs (39) | equal-length pairs only (21) |
|---|---|---|---|
| embed | −0.06 [−0.14, +0.03] | +0.17 [0.07, 0.26] | +0.02 [−0.07, +0.10] |
| mlp0 | +0.10 [0.04, 0.15] | +0.21 [0.13, 0.30] | +0.08 [−0.00, +0.15] |
| mlp2 | +0.12 [0.06, 0.18] | +0.17 [0.10, 0.25] | +0.09 [0.01, 0.17] |
| mlp5 | +0.15 [0.11, 0.20] | +0.14 [0.08, 0.20] | +0.09 [0.01, 0.18] |

For an open-class change (adjective ↔ noun) the difference is NOT along the category axis at the embedding and IS,
modestly (cosine 0.10–0.15), from the first block on, growing with depth. That is motion across the map, measured — and
small: the share of the cross-category difference inside the map's ten-direction span is 2–7%, against 1–4% for
same-category changes. Under the probe (reading) directions alignment is 0.03 at the embedding, rising to 0.16 at mlp5.

**Reading (measured, this checkpoint):** the map and the motion are the same structure only weakly. A category change
moves the state mostly in directions the category map does not chart (word identity), plus a small, reliable component
along the map's axis that appears once attention has run. Limits: tagger labels; pairs counted per occurrence in the
frozen design; position shift in unequal-length pairs (the equal-length column is the control, n = 21).
