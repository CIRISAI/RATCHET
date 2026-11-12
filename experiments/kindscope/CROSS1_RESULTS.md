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

## Exploratory re-read (declared after the result; not a new stake) — `cross1_reread.py`, `cross1_reread.json`

Prompted by cirisholon-d7's reading of the rows. One row per DISTINCT word pair (67), and pairs removed where one reader
(this session) disputes the tagger's label in context — the list is written out in the script (e.g. "kind", "chubby",
"jolly" and "pretty" are adjectives in these openings; "Lily" and "Molly" are names). **44 of the 67 distinct word pairs
are disputed**: most of the "cross-category" set was the tagger's error, i.e. same-category pairs aimed at the wrong axis.

| state | adjective/noun, undisputed (15) | adjective/noun, disputed (18) | all undisputed (23) | all disputed (44) |
|---|---|---|---|---|
| embed | +0.07 [−0.03, +0.17] | −0.19 [−0.31, −0.08] | +0.18 [0.07, 0.30] | −0.04 |
| mlp0 | +0.14 [0.06, 0.21] | +0.04 [−0.05, +0.12] | +0.23 [0.15, 0.33] | +0.09 |
| mlp2 | +0.18 [0.10, 0.26] | +0.04 [−0.03, +0.12] | +0.27 [0.19, 0.36] | +0.06 |
| mlp5 | +0.25 [0.19, 0.30] | +0.08 [0.03, 0.13] | +0.29 [0.23, 0.35] | +0.06 |

With the mislabelled pairs out, a real adjective↔noun change aligns with the map's axis at 0.14 after the first block and
0.25 at the end; the disputed pairs sit near zero, as same-category pairs should. Motion across the map is larger than
the frozen reading showed, and still a minority of the difference. Standing: exploratory; the dispute list is one
reader's and should be re-labelled blind before it carries a claim. The lesson for the MAP: the tagger that labels the
map's categories mislabels common adjectives ("kind", "pretty", "jolly", "chubby") in exactly these contexts, so the
map's own cells carry that error.
