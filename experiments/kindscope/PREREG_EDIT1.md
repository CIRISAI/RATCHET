# EDIT-1 — surgical edit vs fine-tune vs readout fix for one category shift (preregistration)

Frozen 2026-11-11, before any EDIT-1 quantity is computed. Kindscope series (`GOAL.md`), loop registry at RATCHET
`acd225a`.

misfits: KS-DIRECT-ONLY, G-RECALIBRATION-CONTROL, G-SMALLEST-PASSING-STEP, G-LENS-IS-NOT-THE-PATH, M-ONE-MODEL-DELTA, M-BAR-FROM-THE-READ, M-FLOOR-UNSTAKED, M-PLANT-OBS, M-EXIT-DISCRIMINATOR, M-DEVICE-CLASS, G-SECOND-READ, M-BASE-RATE-OMITTED

plants: P-IDENTITY carrier: GoldWorm v6 base — every arm at zero size (alpha = 0, zero fine-tune steps, identity readout) must reproduce the base model's numbers exactly (the floor); P-MATCH carrier: the surgical edit at each layer must reach the fine-tune's calibration-set target effect, or that layer is reported UNREACHABLE

## Question

To make GoldWorm v6 write **more adjectives after determiners** ("the big dog" over "the dog"), what does each route
change: a rank-one surgical weight edit at one MLP (6 layers), a full fine-tune on real stories rich in that pattern, or
a readout-only fix? Do they produce the same behaviour with the same side effects, and does the fine-tune move the
weights where the surgical edit works best?

Known at freeze: CAL-1 (port exact; KS-DIRECT-ONLY); VOCAB-1's running log (POS carried beyond surface at `last` from
attn0 on except mlp0, `first` from mlp1/attn2 on; the `pre` row not yet seen); MLP0-1 frozen and running, no output
seen. Prior art: Hase et al. 2023 (causal-tracing localization did not predict the best ROME edit layer on GPT-J,
ρ ≈ −0.13); the Gemma hand-over (an output-row weight edit matched by a readout-only 12×12 map, +0.024 [−0.028,
+0.077]); ROME (Meng et al. 2022) for the rank-one form ΔW = d uᵀ with u = C⁻¹k.

## Data (real text only)

- **Training corpus:** GoldWorm's own `train_core` = `TinyStories-valid.txt[0, 17502553)` (from the model's
  `eval_results.json`), split on `<|endoftext|>`, pure-ASCII stories, tagged with the VOCAB-1 tagger (spaCy 3.8.7,
  `en_core_web_sm` 3.8.0, same components). Stories shuffled (seed 20261115): first 10% = **CAL** (calibration, never
  trained on); the rest = **POOL**.
- **ADJ-rich** = the top 20% of POOL stories by (DET→ADJ adjacent pairs) / (DET followed by a kept word), among stories
  with ≥ 5 such DET; **CONTROL** = a seeded random sample of POOL with the same story count.
- **Evaluation:** the loop cache's outer_val **EXPLORE** split (CONFIRM unread). DET→ADJ / DET→NOUN = adjacent kept
  tokens (one space between) whose first is DET.

## Arms

- **A0** base.
- **A2** full fine-tune on ADJ-rich: all parameters, AdamW (lr 1e-4, betas (0.9, 0.999), weight decay 0), 400 steps
  (the GoldWorm harness forks' order), batch 32 random 256-byte windows of the ADJ-rich byte stream
  (`<|endoftext|>\n\n` + story, concatenated), fp32, dropout off. Seeds 1, 2, 3.
- **A3** the same on CONTROL (separates "fine-tuning on stories" from the shift). Seeds 1, 2, 3.
- **A1** readout-only (G-RECALIBRATION-CONTROL): an affine map on the 256 output logits (identity init, zero bias),
  base frozen, same data, steps and batch, lr 1e-3 (it has 65,792 parameters against 2.76M). Seeds 1, 2, 3. **A1+2**: the
  same map refitted on each A2 fork (do the gains add?).
- **S_L**, L = 0…5: rank-one edit of `blocks.L.mlp.proj.weight`: ΔW = α · d uᵀ / (uᵀk), with k = mean post-GELU hidden
  activation of mlp_L at CAL positions predicting the first byte of a word that follows a DET; C = their second-moment
  matrix over all CAL positions + 1e-3·tr(C)/768·I; u = C⁻¹k; d = the raw-space direction of a one-vs-rest logistic
  probe (Newton, loop engine fitter) for "next word is ADJ" on the state after mlp_L at those CAL positions, unit norm.
  **α is matched** (P-MATCH): bisection on CAL so that S_L's CAL target effect equals A2's seed-mean CAL target effect;
  α ≤ 200 × the mean norm of mlp_L's write at those positions, else UNREACHABLE.

## Measures (EXPLORE; paired per word; 95% story-bootstrap CIs, 1,000 resamples)

- **T_adj**: Δ log-prob (nats) of DET→ADJ words (sum over the word's bytes, from the position predicting its first byte
  through its last byte), arm minus A0. **T_noun**: the same for DET→NOUN words (the substitution side effect).
- **G**: Δ mean next-byte NLL over all EXPLORE story positions after the prefix (the global side effect, nats/byte).
- **Weights**: per-tensor relative change ‖ΔW‖/‖W‖ for A2, A3 and A2−A3 (the shift-specific part, seed means);
  cosine between A2−A3's change to `blocks.L.mlp.proj.weight` and S_L's ΔW, per L.
- **Probe map for P2**: per L, selectivity of the "next word is ADJ" probe at the post-mlp_L state on CAL (80/20 by
  story): accuracy − mean of 20 control-task accuracies (type-consistent random labels for the next word's form).

## Predictions and bars (from prior art; none from this read)

- **P1** (Gemma hand-over, G-RECALIBRATION-CONTROL): A2 − A1 on T_adj, paired over words and seed-matched, has a 95% CI
  that **includes 0** (the readout fix gets the weight change's target gain).
- **P2** (Hase et al. 2023): across the reachable layers, the Spearman correlation between the probe selectivity and
  S_L's efficiency (T_adj per unit G at matched α) is **not significantly positive** — ρ below the one-sided 5% critical
  value for that n (0.829 at n = 6).
- **P3** (descriptive, no bar): where A2−A3 changes the weights, and whether it aligns with any S_L.

Floor (M-FLOOR-UNSTAKED): P-IDENTITY's deltas must be exactly 0; |Δ| below 10× the larger of that floor and 1e-6 is VOID
for a clause. A fit or bisection that stops without converging records why (M-EXIT-DISCRIMINATOR). Artifacts are fp32
end to end and are evaluated as written (G-SMALLEST-PASSING-STEP: no quantisation step exists in this pipeline).

## Outputs

`edit1.py` (committed before its first run), `edit1_result.json`, `EDIT1_RESULTS.md`.
