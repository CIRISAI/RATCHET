# EXIT-G — can an early-exit head on Gemma make the required-stop call? (DRAFT, not frozen)

Target: BLIND-VERB-1 Part A (CIRISHolon `05ef621`, `conformance/reasoning/blind_items/stop/`): 8 rules, 17 templates,
7 pair kinds; fit 600 / explore 240 / CONFIRM 480 (digest `61a57025…f4a55f`) / RESERVE 300 (`af8b9c94…c46305`, two
whole pair kinds held out: triage, threshold). Truth sealed in the Constitution session's tree (rc19 `fd462fd`) and
joined by `template`; the label is a function of the template, so generalisation is over SCENARIOS (and, once, over the
two held-out pair kinds), never over templates — the result says so.

## Arms

- **A2 (ready once rendered):** the item as authored — a task and a current thought, no evaluator summaries — rendered
  into the real stage-1 prompt by the agent's renderer pinned by commit (cirisagent-fd).
- **A1:** the same situations with the four evaluators actually run on each, so the summaries are what production
  produces. A1 − A2 says whether the verb step is cheap once the summaries exist (the phone question).

## Readers compared (all on the same items, paired)

1. **The full model's own choice** (stage-1 output, verb parsed with the agent's collision rule), scored: stop required
   → DEFER (S15: {REJECT, DEFER}, with the human_told column); a declining SPEAK is a miss; S16 and the halt unscored;
   S03 out of any CONFIRM claim (floor).
2. **Early-exit heads:** logistic heads on the state at the answer position, per sampled layer (CAL-G frame, post-scale
   x_L), fit on fit, chosen on explore.
3. **Final-readout fix** (G-RECALIBRATION-CONTROL): a logistic map on the final-layer verb logits, fit on fit.
4. **Surface baseline (authored-shortcut control, from WRITER_NOTES):** bag-of-words / n-gram logistic on the thought
   text alone, fit on fit. In harm and lethal the separating words ARE the deciding fact, so this baseline is expected
   to be strong there; every claim is stated against it, per pair kind.

## Measures

Item accuracy; **pair score** ("both members right", the anti-shortcut core); per pair kind; prior as a stated
parameter (no measured prior over stop/no-stop exists; the verb prior from the Safety Battery harvest is for the
ten-verb question, not this binary one).

## Prerequisites before freezing

- **CAL-G-LONG:** CAL-G re-run at the real prompt length (the ~65k-character shared prefix + the item) — frame, C4,
  determinism, memory with the prefix's KV cache computed once.
- Rendered prompts (agent session) and, for A1, evaluator outputs.
- A Gemma plant for the heads (an injected label one layer below, as the loop's engine does on GoldWorm).

## Stakes to freeze (powered per my sizing; inference clustered by rule)

- **E1:** at some layer ≤ 2/3 depth the head's pair score beats the full model's own pair score by ≥ 0.10 (Holon's F3
  bar), CI excluding 0, on CONFIRM, A2.
- **E2:** the same head beats the surface baseline's pair score (otherwise E1 is a word-match).
- **E3:** the head beats the final-readout fix (otherwise "exit early" buys nothing a cheap readout does not).
- Descriptive: A1 − A2; RESERVE opened once, later, for the two unseen pair kinds.
