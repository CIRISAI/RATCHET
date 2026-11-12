# EDIT-2 — parallax and torque: where to push, how hard the weights push back, and what stays (preregistration)

Frozen 2026-11-11, before any EDIT-2 quantity is computed. Follows EDIT-1 (results and CAL-only exploration at RATCHET
`09fdf45`).

misfits: KS-READ-NOT-WRITE, KS-READOUT-GRAIN, KS-DIRECT-ONLY, KS-EXACT-ZERO-ACROSS-BATCH, G-LENS-IS-NOT-THE-PATH, G-SECOND-READ, M-BAR-FROM-THE-READ, M-FLOOR-UNSTAKED, M-EXIT-DISCRIMINATOR, M-DEVICE-CLASS, M-ONE-MODEL-DELTA

plants: P-IDENTITY carrier: GoldWorm v6 base — every edit at size 0 and zero withdrawal steps reproduce the base or the fork within the same-batch floor; P-FORCE carrier: the A2 fork — continuing to train on ADJ-rich (force kept) must not lose more of the gain than withdrawing it, or the withdrawal arm is VOID

## Known at freeze

EDIT-1 (EXPLORE): full fine-tune A2 T_adj +0.285 [0.23, 0.34], T_noun −0.144, G +0.0009; readout-only fails (A2−A1
+0.29 [0.22, 0.37]); probe-direction rank-one edits UNREACHABLE at every layer. Exploration on CAL only
(`explore_parallax.json`, `figs/explore_parallax_forest.png`): reading (probe) vs writing (pulled-back gradient)
directions 84–89° apart at every MLP; an additive push along the writing direction gained +0.03 to +0.12 at its best
size at every layer, the reading direction ~0; every push degrades past ~1 write norm, late layers fastest. Torque
(`AGENT_FSDs/CONSCIENCE_V3.md` §11; TORQUE campaign): the angle between the weights' pull and the target's pull, and a
force that must be kept applied. Everything below is read on outer_val **EXPLORE** (CAL was the exploration's split).

## Setup

Same data, model, seeds, fine-tune recipe and measures (T_adj, T_noun, G) as EDIT-1 (`edit1.py` functions reused
unchanged). Keyed rank-one edit at mlp_L: ΔW = α·d uᵀ/(uᵀk), u = C⁻¹k as in EDIT-1. Directions per L, from CAL:
**read** = EDIT-1's probe direction; **write** = the normalised sum over CAL key positions of ∂(Σ log p of the following
ADJ word's bytes)/∂(mlp_L write), through the full network. α ∈ {¼, ½, 1} × the mean mlp_L write norm at key positions.

## Measures, predictions, bars

- **P1 (KS-READ-NOT-WRITE's test):** at α = ½, for **every** L in 0…5, T_adj(write) − T_adj(read), paired per word,
  95% story-bootstrap CI excludes 0 on the positive side. (Stake from the CAL exploration, a prior read on a different
  split.) If P1 holds, KS-READ-NOT-WRITE is registered; if not, it is recanted.
- **Parallax angle** per L (descriptive): angle(read, write).
- **Distributed edit** (descriptive): keyed write-direction edits at all six layers at once, each α = ¼; T_adj, T_noun, G.
- **Torque angle** per tensor (descriptive): angle between the A2−A3 seed-mean change and the descent direction of the
  base model's loss on CONTROL windows (−∇, 64 batches of 32 × 256), i.e. the weights' own pull.
- **Progression** (descriptive, for images): during each A2 fine-tune, every 50 steps: T_adj on an EXPLORE subset (the
  first 80 EXPLORE stories) and each mlp_L/attn_L write's change at key positions.
- **Withdrawal (P2):** from each A2 seed, continue 400 steps on CONTROL (force withdrawn) and, as P-FORCE, 400 steps on
  ADJ-rich (force kept), same recipe; T_adj every 100 steps. **P2:** after 400 withdrawn steps the retained gain
  (T_adj − T_adj at withdrawal start), pooled over the 3 seeds paired per word, is < 0 with CI excluding 0 (the change
  decays when the force stops; prior: TORQUE's framing and catastrophic forgetting).
- **Surgical withdrawal** (descriptive): the write-direction edit at the L with the best EXPLORE T_adj/G at α = ½, then
  the same 400 withdrawn steps; its retained fraction beside A2's.

## Outputs

`edit2.py` (committed before its first run), `edit2_result.json`, `EDIT2_RESULTS.md`, figures.
