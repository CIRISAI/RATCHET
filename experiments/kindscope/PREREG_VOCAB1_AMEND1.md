# VOCAB-1 amendment 1 — fit device (execution only)

Dated 2026-11-11, before any probe cell result was produced. Original: `PREREG_VOCAB1.md` (sha256 `c5f025b4…`, RATCHET
`f435434`).

**Change.** All probe fits (real, null, control, surface, INLP) and all model forwards run on the GPU (RTX 4090 Laptop,
torch 2.11.0+cu128) instead of the CPU. Probe fits stay in **float64** as preregistered; model forwards stay float32 as in
CAL-1. Null and control fits are batched 100 at a time instead of 50 (the fits are independent: each draw's objective
depends only on its own parameters). Nothing else changes: same objective, optimiser, iteration cap, splits, seeds,
labels, nulls, bars and claims.

**Why.** Compute only. The first launch (CPU, float64, 16 threads) was costed at roughly an hour per cell, about two days
for the 39 cells, with the GPU idle. On the GPU, 100 float64 fits of the real problem size (62,000 × 192, 10 classes)
take ~2.4 s.

**Checks before relaunch (no probe results involved).** GPU forward vs CAL-1's CPU port: max logit difference 1.7e-5
(inside CAL-1's C4 bar of 1e-3). float32 was also tried on a structured random problem and converged to the same
objective (1.5e-7 relative) but stopped at gradient norm ~1; float64 was kept because it is the preregistered precision
and is affordable.

**What had been seen when this was decided.** The first launch printed only the counts (1,722 stories; 77,419 tokens;
61,938 train, 15,481 test, 378 novel-type test tokens) and two surface-baseline lines (last: all 0.845, novel 0.582;
first: all 0.705, novel 0.571). It was stopped before any state probe, null, control or amnesic result existed. The
change is to device only and does not depend on those numbers.

**Noted, not changed.** Only 378 test tokens are novel-type (TinyStories' vocabulary is small), so V3's bootstrap CIs
will be wide. That is a property of the frozen design, reported as is.
