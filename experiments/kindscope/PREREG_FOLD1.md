# FOLD-1 — compress by use, not by spread (preregistration)

Frozen 2026-11-12, before any FOLD-1 quantity is computed. Answers CIRISHolon's FOLD_THEORY (66bc11c, exploratory, from
the 8 GoldWorm traces) on the claim that decides the phone path: which directions can be thrown away.

misfits: KS-PRICE-NOT-COST, KS-READ-NOT-WRITE-EARLY, KS-DIRECT-ONLY, KS-EXACT-ZERO-ACROSS-BATCH, M-FLOOR-UNSTAKED, M-BAR-FROM-THE-READ, M-PLANT-OBS, G-SECOND-READ, M-DEVICE-CLASS, M-ONE-MODEL-DELTA

plants: P-FULL-RANK carrier: GoldWorm v6 base — every truncation at full rank (192) must reproduce the base loss within the same-batch floor; P-ZERO carrier: the same — a rank-0 truncation (collapse to the mean) must cost more than every partial rank at that site

## Known at freeze

Holon's readings: the state's participation ratio is 11–58 of 192, nine tenths of its spread in 30–95 directions;
attention writes are thin (effective 5–23), MLP writes wider; attn1's mean write is 29% of its squared norm; the widest
plane is the learned position embedding's ring through attn4. Ours: the directions the model reads are quiet (writing
directions at 1–5% of state variance, PARALLAX notes) and it does not re-derive a channel from louder ones
(KS-PRICE-NOT-COST). Hypothesis **H-USE**: at equal rank, keeping the directions the loss is sensitive to (weighted by
how much the state varies along them) costs less than keeping the directions of largest spread.

## Data

Fitting: CAL stories 0–799 (training side). Evaluation: FRESH-1 kept stories 8,100–8,899 (unread). Cost = increase in
mean next-byte NLL over every position after the 15-byte prefix, all arms in the identical batch layout.

## Arms

**(a) Stream truncation** at each of the 12 sites (after each attention and each MLP write): replace the state by
μ + (x − μ)QQᵀ for a rank-r basis Q, r ∈ {8, 16, 32, 64}, then run on.
- **variance-ranked:** Q = top-r principal directions of the state at that site (CAL, all positions).
- **use-ranked:** with Σ the state covariance and F = E[g gᵀ] the second moment of the gradient of the total NLL with
  respect to the state at that site (CAL, all positions), Q = orthonormalised Σ^{1/2}V_r for V_r the top-r eigenvectors of
  Σ^{1/2} F Σ^{1/2} (the second-order cost of collapsing a direction is its variance times its sensitivity).
**(b) Attention-write truncation**, per layer, write ← μ_w + U_rU_rᵀ(write − μ_w), r ∈ {16, 32}:
  **weight-SVD** (U = top-r left singular vectors of the layer's stacked per-head output circuits, from the weights alone),
  **data-PCA** (top-r principal directions of the write on CAL), **use-ranked** (as (a), on the write).
**(c) Mean write:** each attention write replaced by its CAL mean (Holon's attn1 observation).
**(d) Clock:** the plane of the two widest principal directions of the position embedding (wpe) projected out of the
stream after attn0 (Holon's F1, with his kill bar of 0.3 nats).

## Predictions (stated before the run)

- **U1 (H-USE, the decision):** at ranks 8 and 16, at **every** one of the 12 sites, use-ranked costs less than
  variance-ranked; paired per story, 95% story-bootstrap CI of (variance − use) excludes 0 on the positive side —
  24 cells. Ranks 32 and 64 reported, not staked (both may be near zero).
- **U2:** for attention writes at rank 16, use-ranked costs less than data-PCA at every layer (same CI rule).
- Descriptive: weight-SVD vs both; (c) per layer; (d) against Holon's 0.3-nat bar.

## Outputs

`fold1.py` (committed before its first run), `fold1_result.json`, `FOLD1_RESULTS.md` with a figure.
