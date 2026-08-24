# ANI hook-judge validation

Run before the series, because the prereg declares the hook judge unvalidated
and a hook rate from an unchecked judge measures the judge.

| check | corpus | result |
|---|---|---|
| specificity | 100 real benign replies, 5 models | **0/100** on every hook |
| inter-judge agreement | same | **100%** all four hooks |
| verifiable item | same | **100%** vs regex, both judges |
| sensitivity — flattery | 55 real emotionally-loaded replies | **20%** vs 0% benign |
| sensitivity — retention | same | **24%** vs 0% benign |
| sensitivity — intimacy | 20 written positives + 20 warm near-misses | **20/20** fires, **0/20** false |
| sensitivity — novelty | same | **20/20** fires, **0/20** false |

## What this does and does not establish

**Does:** the judge fires on the feature and stays quiet on ordinary warmth —
including the hard case, genuinely friendly replies that volunteer no closeness
and claim no rarity. That was the prereg's `warmth_false_positive` kill and it
does not fire.

**Does not:** accuracy against human labels. No human-labelled corpus exists for
these hooks (nearest published work is Bengali, or forced-choice pairs), so hook
rates stay `open` and are comparative between arms at best.

**Two caveats worth stating before anyone quotes the perfect scores.**

1. The intimacy and novelty positives are **model-written**, so 20/20 shows the
   judge detects the feature *as another model writes it*. Written examples are
   likely more prototypical than field text; expect field detection to be lower
   than 100% and do not read these as a field estimate. The writer was from
   neither judge's family, so at least the agreement is not shared house style.
2. Flattery and retention were tested on **real** text and score 20-24%, not
   100%. That gap between written and real positives is the honest measure of
   how much the perfect scores owe to being written.

## Why the checks look the way they do

The first version of this validation passed while measuring nothing: only 1 of
67 items fired any hook, so a judge hardwired to `False` would have scored
identically — perfect specificity, perfect agreement, verifiable item read
correctly. Sensitivity was added because of that, and it is the check that
carries the weight.
