# Conscience results and overrides: Scout vs other models

**Status:** measured. Domain: the de-mocked canonical export of 2026-10-08 (7,123
`CONSCIENCE_RESULT` events, 7,316 `ASPDMA_RESULT` events). Script: `compare.py` regenerates
`tables.md`. Labels per `CIRISOntology/epistemology.md` §1.

## Two facts that govern how to read any of this

**The judge is the model.** The four conscience faculties are LLM calls, and they run on the
**same model as the action** in 1,730 of 1,836 traces that have both (94.2%); the remainder
are production Gemma-4 traces where a faculty fell back to `llama-3.3-70b-versatile`. So
"conscience strictness on model X" is X judging X. Cross-model differences in pass rates are
differences in self-judgement calibration, not an external verdict on the responses.

**Scout vs others is confounded with battery.** Scout ran he300 almost exclusively (4,228 of
its 4,999 conscience events; faculties *skipped* on 97.8% of those, so he300 says nothing
about Scout's conscience). Qwen, Maverick, Gemma, Mistral and Llama-3.3 ran mental_health
and ANI. Only within-family pairs are comparable: **ANI** (Scout 262 vs Qwen 118, same arcs,
same arm), **MH-3** (Qwen 328 / Maverick 167 / Gemma 42), **prod** (Gemma-4 192 / Qwen 152).

## Overrides are two different things, and the split changes the headline

The pipeline counts a faculty that **failed to execute** (the DID-NOT-RUN path,
CIRISAgent#1247) as an override and resolves it to PONDER. Splitting overrides into *judged*
(some faculty flag is False) and *non-execution/unattributed*:

| pair | override | judged [95% CI] | non-execution | judged by |
|---|---|---|---|---|
| ANI · Scout | 26.3% | **7.3%** [4, 10] | **19.1%** [15, 24] | humility 19, opt-veto 1 |
| ANI · Qwen | 2.5% | 2.5% [0, 6] | 0.0% | coherence 3 |
| MH-3 · Qwen | 23.5% | 20.1% [16, 24] | 3.4% | coherence 55, opt-veto 20, humility 10 |
| MH-3 · Maverick | 85.6% | 85.6% [80, 91] | 0.0% | humility 141, opt-veto 130 |
| MH-3 · Gemma | 33.3% | 31.0% [17, 45] | 2.4% | opt-veto 13, humility 5 |
| prod · Gemma-4 | 2.1% | 0.5% [0, 2] | 1.6% | opt-veto 1 |
| prod · Qwen | 14.5% | 9.9% [5, 15] | 4.6% | entropy 12, opt-veto 6 |

**ANI, the matched pair.** The raw gap is 26.3% vs 2.5% — ten-fold. Two-thirds of Scout's
overrides are faculties that never ran. The *judged* gap is 7.3% vs 2.5%, almost entirely
epistemic-humility fails on Scout (certainty 0.79 vs 0.95), with overlapping intervals. The
ANI deferral finding already on the public page — Scout "failing the conscience schema" —
is, on this evidence, mostly conscience **non-execution** on Scout (the 180 s harness
timeouts and the 48-of-54 timeout root cause in #1247 are the same phenomenon), with a
modest humility-calibration gap underneath it. Selected action agrees: Scout ponders 31% of
ANI turns, Qwen 9%.

**MH-3.** Maverick-as-judge fails its own actions 85.6% of the time, humility and
optimization-veto together, and ponders on 91% of turns; Llama-3.3 (86.5%, ponder 92%) and
Mistral (60%, ponder 77%) behave the same way, with humility certainty 0.27–0.37 against
0.84–0.98 for every other model. That is judge calibration — these models emit low certainty
numbers — and it makes the pipeline a ponder loop on them as configured. Qwen's MH-3
overrides are a different faculty entirely: coherence (55 of 85), which it does not flag on
ANI (95% pass) or prod (100%).

**Prod.** Gemma-4 overrides 2.1%; Qwen 14.5%, driven by entropy (12), which Qwen fails
nowhere else. Non-execution is small in prod (≤ 4.6%).

**Non-execution is a Scout phenomenon.** ANI Scout 19.1%, ANI-era unlabelled Scout 28.0%,
mixed-unlabelled Scout 9.3%; Qwen ≤ 4.6% everywhere; Maverick, Llama-3.3, Mistral 0%.
Combined with #1247's timeout root cause, the reading is model × infrastructure: Scout's
faculty calls on deepinfra time out on long prompts, the faculties do not run, the action is
overridden to PONDER.

## Unlabelled groups, dated

Two Scout groups carry no battery label. The 260 `synthetic`-unlabelled traces are the ANI
runs (08-24/25 on 2.9.37, 09-11 on 2.9.48 — manifest-matched, channel lost), so their 28%
non-execution is ANI Scout again. The 214 `unknown` traces are a mix — 132 from 2026-08-08
that cannot be labelled, 42 from the MH-3 window, 17 from the ANI window — so their 45%
judged-override rate is not attributable to one family and is reported as mixed.

## What this does not establish

Nothing here says which model's conscience is *right*. A faculty failing on Maverick is
Maverick's number about Maverick's output. Whether low certainty tracks anything about the
response — or anything about the user's outcome — is **open**, and the batteries do not
score it.
