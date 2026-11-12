# MLP0-1 — is GoldWorm's mlp0 a lexical lookup? (preregistration)

Frozen 2026-11-11, before any MLP0-1 quantity is computed. Kindscope series (`GOAL.md`), loop registry `loop/registry.ks`
at RATCHET `acd225a`.

misfits: KS-DIRECT-ONLY, M-PLANT-OBS, H-POWER, M-FLOOR-UNSTAKED, M-BASE-RATE-OMITTED, M-EXIT-DISCRIMINATOR, M-DEVICE-CLASS, G-SECOND-READ, M-ONE-MODEL-DELTA, M-BAR-FROM-THE-READ

plants: P-WTE carrier: GoldWorm v6 base, the byte-embedding part of the embed write (an exact lookup by construction)

## Why

What was known when this was frozen (recorded per the loop's known-at-freeze rule): CAL-1 R1 — zeroing mlp0 moves the
true byte's log-prob by 4.11 nats mean, the largest of the 12 blocks. VOCAB-1's running log (not yet its artifact) at the
`last` site: attn0 → mlp0 → attn1 accuracy 0.815 → 0.885 → 0.911, control-task accuracy 0.669 → 0.786 → 0.785,
novel-word accuracy 0.651 → 0.611 → 0.669; mlp0 is the only `last` cell from attn0 on whose surface-baseline CI
includes 0. Prior art: GPT-2 small's MLP0 described as a "further embedding" after the token embedding
(arXiv 2311.04131) and as the block whose ablation "completely kills performance" while other MLPs' does not
(practitioner report, interpretability-in-the-wild discussion). Hypothesis H-LEX: at a word's last byte, mlp0 writes
word-identity features — a lookup on the word's bytes — rather than context.

## Materials

The loop cache (`ks/engine.py` `build_cache`; same stories, tagging, positions and model as VOCAB-1). **EXPLORE split
only** (`Cache.split["explore"]`); CONFIRM stays unread. Device class: CUDA, RTX 4090 Laptop, float32 forwards, float64
statistics. Writes: `wte` = byte-embedding part of embed; `attn0`, `mlp0` (and `attn1`, `mlp1` for M1 context) = the
block outputs added to the residual. Site: `last` (the word's final byte).

## Measures, predictions and bars

**M1 — lexical variance.** For word forms with ≥ 20 explore-split occurrences: η² of each write by word form (between-form
sum of squares over total, summed over the 192 dimensions). Null: form labels permuted across those tokens, 200 draws.
- P1 (prior art, H-LEX): η²(mlp0) − η²(attn0) > 0, 95% story-bootstrap CI (1,000) excludes 0; each η² above its
  null 95th percentile.

**M2 — context invariance, by patching through the full network (KS-DIRECT-ONLY).** For each eligible token t (its form
occurs in ≥ 2 explore stories; an other-word donor exists), replace one write at t's last byte with a donor's write and
run the model forward; Δ = mean next-byte NLL over the 8 positions t … t+7 (truncated at the window end) minus the
unpatched value. Donors: **same-word** (the same form, a different story, seeded) and **other-word** (same POS, same byte
length, different form, different story, seeded). Writes patched: wte, attn0, mlp0. **Floor:** self-donor (t's own
write) Δ, which must be exactly 0; any |Δ| below 10 × the larger of that floor and 1e-6 nats is VOID for a clause.
- Plant P-WTE (M-PLANT-OBS, H-POWER): for `wte`, same-word Δ = 0 exactly (identical bytes) and other-word Δ > 0 with CI
  excluding 0 — the patch machinery must show both outcomes on this carrier, or M2 is VOID.
- P2 (H-LEX): mlp0 is more context-invariant than attn0 — paired per-token (attn0 same-word Δ − mlp0 same-word Δ) > 0,
  CI excludes 0.
- P3 (H-LEX): mlp0 carries word identity — paired per-token (mlp0 other-word Δ − mlp0 same-word Δ) > 0, CI excludes 0.

**M3 — re-encoding (descriptive, no bar).** One-vs-rest POS probes for the 10 classes at the `last` site: trained on the
attn0 state and applied to the mlp0 state (transfer) against native mlp0 probes; accuracy beside base rate, Newton exit
reasons recorded.

## Outcomes

P1 ∧ P2 ∧ P3 ⇒ H-LEX holds at the last byte (measured, this checkpoint, explore split). Any of them failing is entered as a
loop miss for diagnosis. A VOID plant voids M2 and is registered as an instrument misfit.

## Outputs

`mlp0_1.py` (committed before its first run), `mlp0_1_result.json`, `MLP0_1_RESULTS.md`.
