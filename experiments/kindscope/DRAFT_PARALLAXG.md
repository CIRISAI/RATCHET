# PARALLAX-G — does the parallax transfer to Gemma, with tokens in place of bytes? (DRAFT, not frozen)

Instrument: CAL-G (PASS, `CALG_RESULTS.md`): Gemma-4-E2B-it, float32, frame x_L = s_L·(x_{L−1} + a_L + m_L + p_L).
Pushes and directions are taken on x_L, the post-scale state the next layer reads (the only place a push is not
immediately rescaled away). Text: FRESH-1 stories never read (≥ 24,000 of the kept list for evaluation; a separate
calibration block for fitting directions). Categories: spaCy POS of the next word, as GoldWorm; positions = the token
before a word that follows a determiner.

## Predictions carried over from GoldWorm (each from a measured GoldWorm result; transfer is the open question)

- **G-P1 (parallax):** at every sampled layer the probe's reading direction and the pulled-back writing direction are
  inside the random-direction band for R¹⁵³⁶ (≈ 87.1°–92.9° at 95%). From EDIT-2 / PARALLAX notes.
- **G-P2 (token late):** the writing direction at the last layers is explained by the next-first-token forecast basis
  (16 most frequent next tokens) far above a random 16-direction null. From the byte-basis exploration (R² 0.95 at mlp5).
- **G-P3 (category early):** the residual of the writing direction after the token basis, pushed early, raises
  adjectives over nouns within each first-token group. From PARALLAX-2 (9/9).
- **G-P4 (readout grain):** because most next words are single tokens, a readout-only logit map CAN reproduce a
  fine-tune's shift here (the opposite of GoldWorm's KS-READOUT-GRAIN, as the Gemma hand-over found). A clean
  discriminator between "grain" and "model" explanations.
- **G-P5 (Holon's R4, re-run):** removing the probe-defined subspace costs little; removing along the writing direction
  costs more (KS-PRICE-NOT-COST, KS-READ-NOT-WRITE-EARLY) — the decision `cirisholon-d7` marked as undecidable by
  probe-direction removal.

## Open before freezing

Which layers (35 is costly with backward passes: candidates 2, 6, 10, 14, 18, 22, 26, 30, 33, 34 — 33 because CAL-G's
R1 shows its removal costs 11.7 nats); push sizes in units of each layer's x_L norm; memory for backward passes
(activations only; weights frozen) at batch 1; a plant on Gemma (an injected token-class label one layer below, as the
loop's engine does on GoldWorm).
