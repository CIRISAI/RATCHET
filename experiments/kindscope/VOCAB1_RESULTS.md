# VOCAB-1 results — provisional: the probes did not converge

Preregistration `PREREG_VOCAB1.md` + `PREREG_VOCAB1_AMEND1.md` (device only); script `vocab1.py`; raw
`vocab1_result.json`. GoldWorm v6 base, outer_val test split (both halves), CUDA RTX 4090 Laptop, float64 fits.
77,419 tagged tokens, 378 novel-type test tokens, majority-class accuracy 0.19.

## The fits did not converge (M-EXIT-DISCRIMINATOR — uncited in the freeze, registered upstream before it)

Every probe used L-BFGS capped at 300 iterations. Final gradient norms of the real fits run **2.2–217** (null and
control fits up to 563); a converged fit of this objective reaches ~1e-6 (the loop's Newton fitter). The verdicts below
are therefore the frozen run's letter, **not** measurements of the regularised optimum. They are re-measured as
converged one-vs-rest loop stakes (Newton, plant one state below, power check) — see `loop/`.

## By the letter

"Carried beyond surface" (V1 ∧ V2 ∧ V3), 10-class POS:

```
          embed attn0 mlp0 attn1 mlp1 attn2 mlp2 attn3 mlp3 attn4 mlp4 attn5 mlp5
last        .     C     .    C     C    C     C    C     C    C     C    C     C
first       .     .     .    .     .    .     .    C     C    C     C    C     C
pre         .     .     C    C     C    C     C    C     C    C     C    C     C
```

- `last`: carried from attn0 except mlp0 — explained by MLP0-1 (mlp0 is a lexical lookup; novel-word POS dips there).
- `pre`: the POS of the word **about to be written** is carried beyond the preceding bytes from mlp0 on
  (novel-word margin over the byte baseline +0.09 → +0.21 with depth).
- `first`: carried only from attn3 on — one byte into a word the model has not yet assembled it.

## Causal test (amnesic, INLP 3 rounds = 27 dims removed, full forward)

Removing the pre-site probe's POS directions raises loss on word-initial bytes beyond 20 random rank-matched removals
**only from attn4 on** (A1: attn4, mlp4, attn5, mlp5; A2 from mlp3). Early removals do nothing measurable — although
POS is readable there. Two cautions: INLP removed POS incompletely (probe accuracy after removal 0.26–0.67 against
majority 0.19; LEACE would be the stronger tool), and the probes defining the removed directions were unconverged.

**Reading in the light of PARALLAX-2 (open):** early, the probe reads POS from directions the model does not write
with (parallax), so removing them costs nothing; the early category writer is a different, non-probe direction. Late,
the readable POS sits in the next-byte forecast the model acts on, so removing it hurts.
