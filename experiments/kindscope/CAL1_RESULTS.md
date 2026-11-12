# CAL-1 results — the instrument is calibrated on GoldWorm v6

Preregistration `PREREG_CAL1.md` (sha256 `cc9354b7…`, frozen at RATCHET `87d8e38` before any run); script `cal1.py`;
raw results `cal1_result_tanh.json` (recorded form) and `cal1_result_erf.json` (declared alternative). Local CPU,
torch 2.15.0+cpu. All *measured*, one checkpoint.

| check | tanh GELU (recorded) | erf GELU | bar |
|---|---|---|---|
| C1 published outer-val bpb (target 1.007349040) | **1.007349038**, diff 1.7e-9 | 1.007342959, diff 6.1e-6 | ≤ 1e-3 |
| C2 positions | 129,429, 17764705…19447269 | same | exact |
| C3 residual writes sum to final state | 0.0 | 0.0 | ≤ 1e-5 rel |
| C4 logits rebuilt from 13 writes (ln_f frozen) | ≤ 2.3e-5 | 2.3e-5 | ≤ 1e-3 |
| C5 two passes bit-identical | yes | yes | identical |

**PASS.** The port written from the tensor layout *is* the published model: with tanh GELU it reproduces the authors'
number to nine digits. The frame closes exactly (C3, C4).

## R1 — the first residual (measured, not gated)

Zeroing one block write, the *direct* (linear, ln_f-frozen) prediction of the change in the true byte's log-prob vs
the realised change, 2,000 positions:

| write | corr | sign agree | mean abs realised | mean abs direct |
|---|---|---|---|---|
| attn0 | 0.02 | 0.54 | 2.21 | 0.03 |
| mlp0 | 0.04 | 0.72 | 4.11 | 0.08 |
| blocks 1–3 | 0.02–0.19 | 0.52–0.68 | 0.10–0.28 | 0.02–0.08 |
| mlp4 | 0.44 | 0.78 | 0.32 | 0.17 |
| attn5 | 0.76 | 0.75 | 0.32 | 0.17 |
| mlp5 | 0.92 | 0.92 | 0.48 | 0.58 |

(erf run; full table in the JSON.) **Candidate misfit KS-DIRECT-ONLY:** direct attribution (a write read straight through
the final norm and head) explains the last block and almost none of the early ones — mlp0's realised effect is ~50× its
direct one. Early writes act through what later blocks do with them. ⇒ VOCAB-1 must attribute with downstream paths
(gradients or path patching), never direct reads alone, before any early-layer ranking or image is trusted. Entered as a
candidate; it becomes a misfit only if it recurs on the forks under the shuffled-label rule.
