# Kindscope — goal and grounding

Series opened 2026-11-12 under [RATCHET#31](https://github.com/CIRISAI/RATCHET/issues/31) (hand-over from a CIRISAgent
session; its bundle is `/home/emoore/kindscope-handoff-2026-11-11/`). Labels per `CIRISOntology/epistemology.md` §1:
proved / measured / open / wager.

## Goal

Build a **calibrated, layer-by-layer map of how a model carries identifiable categories** — gross vocabulary first
(byte and token classes, parts of speech, nouns and verbs), then the 11+1 kinds — such that the map predicts what an
edit, ablation or fine-tune will do. **Every prediction that fails becomes a named misfit with its reason**, kept in a
registry patterned on `CIRISHolon/conformance/gravity/MISFITS.md`. Progress is measured by how much behaviour stays
unexplained (the residual) and whether it shrinks.

Understanding and use are one goal here, not two: if the structure that carries meaning is the structure that chooses
the action, then mapping the categories *is* the path to a small model that chooses ASPDMA verbs well (logos).

## Grounding (what is already measured, and what it forces)

From the hand-over on Gemma-4-E2B (all *measured*, one model, one quantisation, curated items):

1. **The structure exists only in use.** A linear read of the layer-16 state names the kind at 0.82 held-out while the
   model says it at 0.46; the static weights show no kind subspace. ⇒ Images must show weights *acting on inputs*
   (states, writes, attributions), never raw weights alone.
2. **The instrument was uncalibrated.** `scope.py`'s attribution failed its own additivity check (relative error 1.19):
   structure between the taps (output scales, gates, per-layer embeddings) was unaccounted for. This is the SEM
   parallax problem: raw readings are unintelligible until the frame is corrected. ⇒ **Nothing is imaged or ranked until
   the frame closes** (writes sum to steps; linearised attributions sum to logits).
3. **Failed predictions were the richest results** (closure killed by a shuffled-label null; no single suppressor block;
   the weight edit ≈ logit recalibration). ⇒ Make that the method: predict, measure, register the reason.

## Model ladder

1. **GoldWorm v6** (`Uuuuuuniiiiiii/goldworm-v6`, MIT): 2.76M-parameter byte-level pre-LN transformer, 6 layers ×
   192, trained on TinyStories. Fully specified, unquantised, published evaluation protocol with a target number
   (1.007349 bpb on 129,429 fixed positions). The base is the **gross-vocabulary control**: English parts of speech are
   free labels from a tagger, and at byte level a word is never one token, so a category must be *assembled across
   positions* — closer to "the kind is carried, not stored".
2. **The 27 GoldWorm v6 harnesses** (`Uuuuuuniiiiiii/goldworm-v6-harnesses`; MIT base, some data GFDL 1.2): each a
   ~400-step fine-tune of the same base onto one code or data format. **27 natural, labelled weight edits from one
   origin**, toward domains whose categories a lexer labels exactly (keyword, identifier, operator, literal, comment,
   string). Predicted vs realised effect of each fork is the ratchet ledger at a scale where statistics work.
   Caveats from their card: single seed, under-trained, validation sets rebuilt after training, synthetic formats.
3. **Back up to the kinds** with the calibrated instrument: Gemma-4-E2B (hand-over state) and a small model on
   ASPDMA-shaped text. Requires the fresh, blind-authored labelled set the hand-over lists as missing.

## Series

| step | name | question | gate |
|---|---|---|---|
| 1 | **CAL-1** | Does our port reproduce the published model exactly, and does the frame close? | `PREREG_CAL1.md` |
| 2 | VOCAB-1 | Where across layers are parts of speech / noun-verb carried in the base, against a shuffled-label null? | prereg before run |
| 3 | FORK-1 | From the base map, predict what each harness's weight change does; measure; register misfits | prereg before run |
| 4 | IMAGE-1 | Render states/writes/attributions, each stating its frame (basis, normalisation, input set, layer) | after 2–3 |
| 5 | KIND-1 | Carry the instrument to the 11+1 kinds | fresh labelled set |

## Residual analysis (the statistical misfit rule)

Every prediction is logged with its realised outcome. Failures are clustered by **signature** (site, category, edit
type, sign and size of error). A cluster is entered as a misfit only if it recurs beyond a **shuffled-label null** (the
null that killed closure on Gemma). Each misfit gets an id, a one-line reason and the run it was born in; later
preregistrations must cite any misfit whose keywords they touch. Bars come from a prior stake, prior art or a plant
on the carrier of record — never from the reading they grade (holon M-BAR-FROM-THE-READ).

## Running

GoldWorm-scale runs are stateless and CPU-sized, so they run locally from frozen, committed scripts with outputs
committed beside them; anything that boots an agent still goes through CI.
