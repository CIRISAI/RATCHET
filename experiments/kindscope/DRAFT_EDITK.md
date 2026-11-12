# EDIT-K — one real-data change per kind of change (DRAFT, not frozen)

Kinds per CIRISConstitution CC 1.13.4 (informative map; the definitions used here are ours — cirisconstitution-40,
2026-11-12). Realisable in TinyStories by **selection** of real stories (no constructed swaps — no synthetic data).
Model: GoldWorm v6 base. Fine-tune recipe, control fine-tune and measures as EDIT-1/EDIT-2. Evaluation on fresh text
(FRESH-1 stories ≥ 12,000, unread).

| kind | real-data shift (fine-tune on stories selected by …) | target T_k on fresh text | grain | natural operation (verb prior) |
|---|---|---|---|---|
| Facts | a name co-occurring with one animal (e.g. stories where "Lily" and "dog" co-occur) | Δ log p("dog") − Δ log p(other animals) at animal slots in stories naming that character | a specific word | replace (by selection) |
| Rules | an unkind act followed within 2 sentences by a consequence word (sad, cry, sorry, punished) | Δ log p(consequence words) after unkind-act sentences, vs after neutral sentences | a regularity across sentences | withdraw (control: stories where it goes unanswered) |
| Identity | quotes attributed to a parent ("said Mom/Dad") | Δ log p(parent word) right after `said` | a specific word in a fixed slot | replace |
| Manner | adjectives after determiners (EDIT-1) | T_adj (measured) | a category | — (measured) |
| Process | problem → help → resolution order (keywords: "problem/stuck/lost" before "help" before "happy/fixed") | Δ log p("help") after a problem sentence vs before it | an ordering | add / reorder |
| Circumstances | weather or time-of-day settings (rain, night, snow, sunny) | Δ log p(setting words) in the opening sentence | a small category | replace |

Confidence: only if the corpus has enough hedging (maybe / I think / was sure) — count before freezing.
Waiting for a larger corpus: Priorities, Structure, Model, Premises.

## Stakes drawn from PARALLAX-2 (the grain rule)

- **G1:** for kinds whose target is a *specific word* (Facts, Identity), the pulled-back writing direction for T_k is a
  next-byte forecast at mlp3–mlp5 (R² on 16 per-byte directions above the random null) — the edit lives late.
- **G2:** for kinds whose target is a *category* (Manner, Circumstances), the early (mlp0–2) non-byte residual of the
  writing direction moves T_k within every first-letter group — the edit lives early.
- **G3 (open, no stake yet):** Rules and Process — targets that span sentences — have no grain prediction from
  PARALLAX-2; measure where their writing directions sit.

## Torque, corrected (KS-FORCE-AT-WRONG-POINT)

Per kind: **deflection** (output space, the agent's TORQUE sense) = share of the base→fork log-odds gap on T_k closed by
the fork per item, and its decay under 400 withdrawn steps; **restoring force** (weight space) = angle and size of the
ordinary-loss gradient at the shifted weights θ + δ against δ, per tensor. Prediction to stake: the two agree on which
kinds decay fastest.

## Open before freezing

Selection counts per kind on train_core (each needs ≥ 1,000 stories for a 400-step fine-tune); the exact target word
sets; whether Rules' consequence targets are measurable as word log-probs without constructing text.

## Selection counts in GoldWorm's own train_core (18,841 stories; counted 2026-11-12)

Facts ("lily"+"dog") 351 · Rules (unkind act → consequence word within 250 chars) 656 · Identity ("said mom/dad") 69 ·
Process (problem word → later "help") 1,569 · Circumstances (setting word in the first sentence) 184 · Confidence
(≥ 1 hedge) 1,185, (≥ 2) 268. Most kinds fall short of ~1,000 stories, Identity badly.

**Plan:** select fine-tune data from a separate, hashed slice of `TinyStories-train.txt` (≈ 2 GB, never seen by GoldWorm;
FRESH-1 is its first 20 MiB and stays the evaluation source), e.g. bytes 200–400 MiB → ≈ 10× the counts above. The
control fine-tune draws a random selection of the same size from the same slice, which also absorbs the general effect
of training on unseen text. Identity's slot may need "Mom said" as well as "said Mom" — count both before freezing.
