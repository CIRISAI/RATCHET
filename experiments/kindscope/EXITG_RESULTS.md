# EXIT-G2 results — the exit head beats the full model and the readout fix, and loses to the words

Preregistration `PREREG_EXITG2.md` (RATCHET `dc16e87`; EXIT-G `b213454` voided itself before the sealed read); script
`exitg2.py`; raw `exitg2_result.json`; limits `EXITG_LIMITS.md`. **Measured:** Gemma-4-E2B-it (float32, SDPA, cached
prefix), compact stem, exit head at layer 19 (fixed from the design phase), CONFIRM read once (480 items; 400 scored;
160 pair groups over 80 scenarios in 4 pair kinds). Pair score = share of (scenario, phrasing) groups with every scored
member right; 95% intervals by bootstrap over scenarios. All fits converged.

**The lethal pair kind contributes no pair group** (no group there holds both a stop-required and a none-required scored
member once S16 is unscored), so the pair scores below are over accept, harm, irreversible and veto. Item accuracies
include lethal items.

## Stakes (CONFIRM, summaries withheld, default template)

| stake | value | verdict |
|---|---|---|
| E1 exit head − Gemma full model | +0.594 [0.500, 0.681] | **holds** |
| E2 exit head − surface word-match baseline | −0.206 [−0.306, −0.119] | **fails** |
| E3 exit head − final-readout fix | +0.356 [0.269, 0.450] | **holds** |

By the preregistration's own rule, E2 failing means E1 is not evidence that the state holds the decision beyond what the
item's words give: a bag-of-words reader of the thought text scores higher than the head.

## Readers, pair score [95% CI] (item accuracy; base rate 0.60)

| reader | summaries withheld (A2) | as deployed (A1) | base, no template |
|---|---|---|---|
| Gemma, own verb | 0.031 [0.006, 0.069] (0.45) | 0.087 [0.037, 0.138] (0.44) | 0.006 [0.000, 0.019] (0.38) |
| final-readout fix | 0.269 [0.194, 0.350] (0.70) | 0.206 [0.131, 0.281] (0.62) | 0.287 [0.200, 0.375] (0.69) |
| exit head, fit on A2 | 0.625 [0.544, 0.713] (0.85) | 0.150 [0.094, 0.212] (0.56) | 0.425 [0.331, 0.506] (0.73) |
| exit head, refit on the arm | — | 0.463 [0.375, 0.556] (0.75) | 0.694 [0.613, 0.775] (0.86) |
| surface baseline (words) | **0.831 [0.744, 0.906] (0.93)** | — | — |
| Qwen as deployed (bare step) | — | 0.087 [0.050, 0.125] (0.38) | — |

Gemma's verbs on CONFIRM: A2 — ponder 255, tool 103, defer 35, task_complete 27, speak 20, recall 10, reject 1,
off-format 29; A1 — ponder 243, defer 97, tool 56, reject 39, …; base — tool 251, ponder 80, speak 58, task_complete 57,
defer 4. The pre-ruling S15 reading ({reject, defer}) changes nothing at pair level.

By pair kind (A2): head — accept 0.78, irreversible 0.82, veto 0.65, harm 0.25; surface — accept 1.00, irreversible
1.00, veto 0.95, harm 0.38; Gemma — at most 0.10 anywhere. By domain the head reads 0.78–0.91 and the surface baseline
0.90–0.95; no domain stands apart (housing, the template-touched one, included).

## Reading (measured, these items; limits in `EXITG_LIMITS.md`)

1. **Neither model's verb step makes the stop call.** Gemma gets both members of a pair right 3% of the time (9% with
   real evaluator summaries); Qwen, the deployed model, 9%. Item accuracy is below the 0.60 base rate for both.
2. **The information is in the words and a linear reader finds it** (0.83). The model's state at layer 19 holds a
   noisier copy of it (0.63), far more than its own answer uses (0.03) and more than a readout fix on its verb logits
   recovers (0.27). "Held, not said" recurs here — but the head does not beat the text, so nothing shows the model
   computes the decision; it shows the model carries the words forward.
3. **Harm is the hard pair** for every reader (surface 0.38, head 0.25): the deciding fact is a number on one side of a
   line (0.5), which a bag of words cannot compare and the head does little better.
4. **Summaries change the state, not the answer.** With the four evaluators' real summaries present, Gemma defers more
   (97 vs 35) and its pair score is 0.09; a head fit without summaries does not transfer (0.15), a refit head reads 0.46
   — lower than without summaries (0.63), so the summaries blur the pair distinction in the state rather than sharpen it.
5. **The template moves the verbs a great deal** (tool 251 → 103, ponder 80 → 255, defer 4 → 35) and the pair score not at
   all (0.006 → 0.031). The base-arm comparison carries the deferral-policy confound and everything else the template adds.

For the phone question: on this task an early exit is not where the gain is — the stop decision is recoverable from
the input by a tiny reader, and what both models lack is the step that turns "this is a stop situation" into the verb.
In the deployed agent that step is the conscience, not action selection (see `ci_verbs/verb_prior.json`).
