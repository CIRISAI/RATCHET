# EXIT-G2 — EXIT-G re-frozen after it voided itself before the sealed read (preregistration)

Frozen 2026-11-12. **CONFIRM (`61a570258a520c08f600c951aa938ca86ffbb3af089a26ee21be37b868f4a55f`, 480 items) has not
been read:** EXIT-G's `confirm` phase never ran, no CONFIRM item, rendered prompt, A1 output or state has been opened on
this side (ids only, once, to count rows). EXIT-G2 is its one read.

misfits: KS-CASE-BLIND-READOUT, KS-NULL-EXPECTATION, G-SECOND-READ, G-RECALIBRATION-CONTROL, G-STYLE-BOUND-READER, G-DEFAULT-NAME, G-LENS-IS-NOT-THE-PATH, KS-CONTROL-BASE-RATE, KS-PRICE-NOT-COST, KS-DIRECT-ONLY, M-PLANT-OBS, H-POWER, M-BASE-RATE-OMITTED, M-EXIT-DISCRIMINATOR, M-DEVICE-CLASS, M-BAR-FROM-THE-READ, M-ONE-MODEL-DELTA, M-STALE-INSTRUMENT

plants: P-KIND carrier: Gemma-4-E2B-it states at the answer position, EXPLORE split — a head for whether an item is of the `lethal` pair kind reaches accuracy at least 0.90 at the chosen layer, and the same head fit on shuffled labels does not exceed the majority rate by more than 0.05; else VOID

## Why EXIT-G is void (by its letter, twice), and what was seen

EXIT-G (`PREREG_EXITG.md`, RATCHET `b213454`) ran its `design` phase on fit and explore (`exitg_design.json`):
1. **Plant null mis-staked (KS-NULL-EXPECTATION).** The shuffled-label head was required to sit within 0.10 of the base
   rate (0.75); an uninformed head at a 0.25 positive rate scores about 0.625. It read 0.583 — a null behaving as a null —
   and failed the letter. The kind head read 1.00.
2. **Read-out blind to case (KS-CASE-BLIND-READOUT).** Mean mass on the ten lower-case verb tokens: 0.064 (compact
   stem), 0.042 (deployed stem); 234 of 240 explore items "off-format". Inspecting four fit items showed Gemma answers in
   UPPER CASE at that position (`P`, `TASK`, `OBS`, `SPEAK`, `MEM`, `RE`), with lower-case forms on some items.

**Known at this freeze** (all fit/explore, A2 arm): the head's explore pair score by layer (0.01 at layer 0, 0.69 at layer
19, 0.61–0.69 at layers 20–34; item accuracy 0.875 at layer 19); the stem masses above; top tokens on four fit items;
Qwen's verbs on fit and explore from the A1 run; the truth table (by template). Nothing about CONFIRM.

## Changes from EXIT-G (everything else is `PREREG_EXITG.md` as frozen: data, arms, readers, scoring, stakes E1–E3)

- **Read-out.** A verb's probability = P(its lower-case first token) + P(its upper-case first token) at the answer
  position. RECALL and REJECT share the upper-case first token `RE`; its mass is split by one further step (the model fed
  `RE`, reading `CALL` against `JECT`). The ten values are summed for the mass; under 0.5 an item is "off-format" and
  counts as wrong for the full model, as before. The final-readout fix reads the log of these ten values. Upper-case
  first tokens that could begin another word (`P`, `DE`, `FOR`) are taken as the verb in this slot; stated, not tested.
- **Plant criterion** as in the plants line (one-sided null). On the design phase's recorded numbers (kind 1.00; shuffled
  0.583 against a majority rate of 0.75) it passes; that it is judged on numbers already seen is disclosed — the change is
  to an arithmetic error in the null, not to the kind head's bar.
- **Fixed from the design phase, not re-chosen:** the compact stem and **layer 19** (chosen by EXIT-G's frozen rule on
  explore; the heads do not depend on the read-out).
- **Only the `confirm` phase runs** (`exitg2.py`), writing `exitg2_result.json`; it refuses if either result file exists.

## Limits

`EXITG_LIMITS.md` (not a regime-harness run; the template's deferral-policy text is a confound for claims about the rules
alone and for the base arm; authored situations in the deployed frame; fit, explore and CONFIRM vary Facts, Identity and
Circumstances only; one authoring process; 8 rules). E3 may be underpowered; an interval straddling 0 is undecided.

## Outputs

`exitg2.py` (committed with this file before its run), `exitg2_result.json`, `EXITG_RESULTS.md`.
