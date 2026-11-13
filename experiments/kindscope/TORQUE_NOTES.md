# Logic against training — session record, 2026-11-12 (all EXPLORATORY; nothing here was frozen before it was read)

Carrier: GoldWorm v6 base (6 blocks, width 192, bytes). Real text only: the FRESH-1 kept list of TinyStories
(`gw_fresh/TinyStories-train.head20MiB.txt`, 21,884 stories kept). Slices used: fit 0–11,999; read 12,000–15,999;
first pictures and harvest 16,000–19,999; ordinary bits per byte on 20,000 on. Every label below is **measured**
(domain: this one model, this text) unless marked as a reading.

## The owner's framing this record follows
- Removing layers is not the way. Edit weights in concert to make new curves in the map or repair existing ones.
- See the curve the logic obeys against the curve the weights encode: the torque of logic against training,
  statistics against logistics, **vocab against grammar**.
- Find the curves that drive and change the curve en masse (berms and swales).
- Every change an edit produces is a hint, never "not what we want".
- Decouple the sources of the dynamics strategically.

## What was measured, in order

| file | what | result |
|---|---|---|
| `torque_names.py/json` | a character's name recurring, by how common the name was in training | weights give the right name 0.34 for names never named in training (vocab share 0) up to 0.69 for the commonest (vocab 0.25); pull toward the commonest name's first letter 0.01–0.03; for unseen names most loss is after the first letter |
| `torque_agree.py/json` | was/were, is/are, has/have against the previous word's training statistics | where vocab gives the text's form < 0.1 the weights give 0.69 per distinct pair; 0.87 where the previous word was never seen with either form; "there were" follows vocab (0.24) |
| `swale.py`, `swale_step.json` | one straight step along the mean gradient of against-vocab agreement | gradients spread over ~22 shared directions (top one 18%); the step stalls (unread 0.722 → 0.713–0.726); projecting out 200 windows' ordinary-loss gradients halves the cost (+0.0046 against +0.0084 bits per byte) |
| `swale_path.py/json` | 40 re-taken small steps | plural-asking and singular-asking cases pull opposite ways (cosine −0.83); unread against-vocab +0.133 [0.078, 0.193]; fit reaches 0.999 while unread stops at 0.855; bits per byte 0.8021 → 0.8324 |
| `coupling.py`, `coupling.json` | each agreement group raised alone, all read | the three verb pairs are nearly separate blocks; inside each the two forms trade whatever the vocab condition; a move of ~0.2% of the weights' norm flips "clowns were" 0.999 → 0.011 |
| `seesaw.json` | against-vocab was and were raised together | were-against +0.135 [0.077, 0.201]; removing the trade direction from each step changes little |
| `subject_coupling.py/json` | regrouped by where the subject sits (one parser) | position does not separate the sources at this step size: every was row is the same row |
| `swipes.py/json` | 21 families of two-way word choices, each raised alone | families move each other by at most 0.04; inside most the rarer word rises and the commoner falls; gender, tense and his/her rise without a trade; a/the does not transfer |
| `harvest.py/json` | (1) cosine of the two words' gradients for each hand-picked pair; (2) the model's own contested word pairs at 24,551 word starts | (1) every pair opposed (−0.58 to −0.98), sitting mainly in the tied embedding and mlp4/mlp5; tense least opposed, gender not singled out. (2) 12,071 contested starts, 6,722 distinct pairs, top 50 hold 0.40 and top 200 hold 0.64; a/the, loved/was, her/the, boy/girl, boy/little lead; most hand-picked grammar pairs rank low |

## Readings (not tested)
- What a small coordinated step reaches first is a per-pair balance between two words, held mostly in the embedding
  (which is also the output table) and the last two MLPs. This is the "vocab" lever.
- The choices that rose as a body are ones an earlier part of the story fixes (gender, tense). These are the candidates
  for a curve that can be moved whole.
- By the draft taxonomy's glosses (CHANGE1_SPEC_DRAFT): the intervention is Model, its effect is Priorities, and the
  cases differ by Structure. No labeller has labelled this.

## Not done
- The harvested pairs are not placed on the part-of-speech map (tag both words of the top 200, split same-class from
  cross-class, read cross-class ones against the map's category directions per state).
- The see-saw-held table (a was group and a were group raised together, by subject position) was stopped unfinished.
- The embedding-share signal for "context fixes it" is not tested as a classifier.
- One step size, ten or forty steps; one parser; some forms rest on 2–3 word pairs.
- Nothing here is on Gemma.

## Limits to carry
- Slices 12,000–19,999 of the fresh list have now been read by these runs; a frozen stake needs text beyond them.
- `swale.py` holds full gradient tables (~20 GB) and was stopped twice for memory; `swale_path.py` is the light form.
