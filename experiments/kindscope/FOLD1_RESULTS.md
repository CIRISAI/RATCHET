# FOLD-1 results — VOID by its letter; use beats spread only for gentle compression

Preregistration `PREREG_FOLD1.md` (`b5181a6`); script `fold1.py`; raw `fold1_result.json`. **Measured:** GoldWorm v6
base, fitting on 192,800 CAL positions, evaluation on 800 unseen FRESH-1 stories (8,100–8,899); cost = added mean
next-byte NLL (nats); identical batch layout for every arm.

**Plants.** P-FULL-RANK held at all 12 sites (|cost| < 1e-5). **P-ZERO failed at attn2**: collapsing the stream to its
mean (+2.99) cost less than keeping 8 directions (+3.02 variance-ranked, +3.37 use-ranked). The run is VOID by its letter
→ **KS-PARTIAL-WORSE-THAN-COLLAPSE** (registered): partial projection builds states the model never sees.

**U1 (stake: use-ranked beats variance-ranked at ranks 8 and 16, every site) — VOID; descriptively it would fail
(10 of 24 cells).** Use-ranked wins at mid and late sites (mlp1 … attn5) and at gentle compression (ranks 32 and 64:
10 of 12 sites each, the other two tied); spread wins for aggressive early compression (attn0, mlp0, attn1 at ranks
8–16, and mlp5 at 8–16), where the loud directions are position and byte identity, which the model needs. The use
criterion is a second-order small-removal estimate and fails for large removals (the linear-range lesson again).

**U2 (attention writes, rank 16, use beats data-PCA at every layer) — VOID; descriptively 4 of 6.** All routes are
cheap: rank 16 of 192 costs ≤ 0.04 nats per layer; weight-SVD is worse than the data routes at attn0 (0.033 vs 0.010).

**Mean write (each attention layer replaced by its average write):** attn0 +2.008, **attn1 +0.016**, attn2 +0.060,
attn3 +0.068, attn4 +0.107, attn5 +0.133. attn1 is nearly a constant push — a whole attention layer that folds into a
bias at 0.016 nats (Holon's observation, now by removal).

**Clock plane** removed after attn0: +0.109 nats (under Holon's 0.3 bar).

**For the purpose-built-network question (measured, this checkpoint):** attention writes compress ~12× at ≤ 0.04
nats/layer; attn1 is removable as a constant; the stream keeps ≤ 0.1 nats at rank 64 of 192 with use-ranked
directions; ranking by use beats ranking by spread for gentle compression, not for aggressive early compression; and a
partial cut can be worse than none, so every fold needs its own removal check.
