# MLP0-2, EDIT-1 and EDIT-2 results

All **measured** on GoldWorm v6 base, one checkpoint, CUDA RTX 4090 Laptop. Raw: `mlp0_2_result.json`,
`edit1_result.json`, `edit2_result.json`; figures in `figs/`. Exploration on CAL only: `explore_parallax.*`.

## MLP0-2 (H-SUFFIX, CONFIRM split — its first read)

- **P1 holds.** Null-adjusted η² of the mlp0 write: by whole form 0.864; by last 3 bytes 0.863; by first 3 bytes 0.716.
  Suffix − prefix: +0.23 [0.20, 0.25] at k = 2, +0.15 [0.13, 0.16] at k = 3.
- **P2 VOID by its letter.** The plant required an exact 0; the base and patched passes ran in different batch shapes,
  so even a self-patch reads ~1e-9 (summation order) → **KS-EXACT-ZERO-ACROSS-BATCH** (registered). The numbers point
  the same way (mlp0 patch: same word 0.0012, same last-3 bytes 0.0020, same first-3 bytes 0.273 nats) but are not a
  verdict. A rerun under the new rule is CONFIRM's second read and must say so.

## EDIT-1 (more adjectives after determiners — a Manner change; EXPLORE; 3 seeds)

| arm | T_adj (nats) | T_noun | G (nats/byte) |
|---|---|---|---|
| A2 fine-tune on ADJ-rich stories | **+0.285 [0.228, 0.342]** | −0.144 [−0.202, −0.074] | +0.0009 |
| A3 control fine-tune | +0.007 [−0.056, +0.075] | +0.043 | −0.0044 |
| A1 readout-only logit map | −0.009 [−0.044, +0.026] | −0.100 | +0.0155 |
| A1+2 map refitted on A2 | +0.245 | −0.196 | +0.0149 |

- **P1 (Gemma: readout ≈ weight edit) fails:** A2 − A1 = +0.293 [0.220, 0.367]. Candidate **KS-READOUT-GRAIN**: at byte
  level the category spans several output units, so a readout map cannot carry it (Gemma's answer was one token).
- **P2 VOID:** every probe-direction rank-one edit was UNREACHABLE up to 200× the write norm.

## Exploration (CAL only, not a verdict) → EDIT-2's stakes

Reading (probe) and pulled-back writing (gradient) directions are 84–90° apart at every MLP. **Null:** two random unit
vectors in R¹⁹² fall within 81.9°–98.1° 95% of the time, so the reading and writing directions are **as unrelated as
random directions**. `figs/explore_parallax_forest.png`.

## EDIT-2 (parallax, progression, withdrawal; EXPLORE)

- **Parallax per layer** (directions from full CAL): 87.3°, 85.9°, 89.8°, 84.7°, 82.1°, 82.1° — all inside the
  random band (`figs/edit2_parallax.png`).
- **P1 fails** (stake: writing beats reading at half write norm at **every** layer): +0.080 [0.035, 0.125],
  +0.070 [0.028, 0.108], +0.104 [0.055, 0.154] at mlp0–2; mlp3 +0.029 [−0.035, 0.091]; mlp4 −0.068 [−0.148, 0.011];
  mlp5 −0.562 [−0.717, −0.422]. **KS-READ-NOT-WRITE recanted** (row kept); refined as candidate
  **KS-READ-NOT-WRITE-EARLY** (early layers; size kept in the first-order range). The "every layer" bar was a
  judgement call, and it is what failed.
- **Distributed edit** (all six layers at ¼): T_adj −0.65, T_noun −1.00, G +0.022 — summing pushes breaks the model,
  dominated by the stiff late layers.
- **Progression** (`figs/edit2_progression.png`): the behaviour reaches ~+0.18 in the first 50 steps, then climbs
  slowly to ~+0.28. The largest write change is at **attn5** (~0.5 of its write size before a word after DET), then
  attn1/attn2/mlp1 (~0.35); attn0, mlp0 and mlp5 change least. **The fine-tune works mostly where surgical pushes break
  things (late attention), and least where they work (mlp0–2).**
- **Withdrawal — P2 holds** (`figs/edit2_withdrawal.png`): after 400 steps on ordinary stories the gain falls by
  −0.222 [−0.273, −0.173] (+0.28 → +0.06); with the force kept it changes +0.020 [−0.019, +0.056] (P-FORCE ok). The
  surgical mlp0 edit's +0.09 is gone within 100 steps. Most of the change is held up by continued force, not settled
  into the weights.
- **The "torque" angle is VOID as torque.** 91–93° in every tensor was measured against the gradient at the base
  weights — a converged model's near-zero, noise-dominated pull, to which any direction sits near 90°. My first reading
  ("slightly against the grain") is withdrawn. **KS-FORCE-AT-WRONG-POINT** (registered; reviewer cirisagent-fd):
  restoring force must be read at θ + δ; the agent's TORQUE is output-space deflection.

## Reading across (open, to be staked)

Categories are **read** in one geometry and **written** in another, unrelated as random directions. Surgical pushes
work only early and break late layers; the fine-tune instead changes late attention most; and most of what it builds
decays when the force stops. Next: deflection (share of the base→target gap closed per item) and restoring force at
θ + δ, under withdrawal, per kind of change (EDIT-K).

## MLP0-3 and MLP0-4 (H-SUFFIX patching, fresh text)

- **MLP0-3 VOID by its letter** (`mlp0_3_result.json`): donors were matched on lower-cased forms; 2 of 71,195 had a
  differently cased last byte, so the byte-exact plant failed → **KS-CASEFOLD-DONOR** (registered).
- **MLP0-4 — holds** (`mlp0_4_result.json`; FRESH-1 stories 16,000–19,999; raw-byte donors, precondition asserted over
  every job; unpatched pass in the identical batch layout): plant fired (self floor 1.3e-6); mlp0 patch Δ next-8-byte
  loss: same word 0.0014, same last 3 bytes 0.0023, same first 3 bytes 0.260; **P2 = +0.258 [0.255, 0.261]**.
  With MLP0-2's η² (CONFIRM), H-SUFFIX holds on two independent tests: **mlp0 is a lookup keyed on a word's final
  bytes** (measured, this checkpoint).

## WIRE-1 — attn0/h3 → mlp0 (`wire1_result.json`; FRESH-1 stories 9,000–11,999; 60,029 word-final sites, 885 forms)

Plant P-NULL-ABLATION: bit-identical. Null-adjusted η² of the mlp0 write by the last 3 bytes, base 0.873; drop when each
attn0 head is zeroed: h3 **0.039**, h2 0.016, h0 −0.007, h1 −0.061 (removing h1 makes the write MORE word-determined).
By first 3 bytes: h3 0.038, h2 0.049. **W1 holds** (+0.023 [0.022, 0.023]) and **W2 holds in form, negligible in size**
(+0.0007 [0.0002, 0.0014]) — so H-WIRE holds by its letter. Reading: a real but **weak** wire; mlp0's final-bytes lookup
survives the loss of any single attn0 head (0.873 → 0.834 at worst), h3 is one minor input among several, and its
contribution is not specific to the word's ending.
