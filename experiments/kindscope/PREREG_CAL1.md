# CAL-1 — calibrate the instrument on GoldWorm v6 (preregistration)

Frozen before any model output is read. Part of the kindscope series (`GOAL.md`).

## Question

Does a PyTorch port, written from the checkpoint's tensor layout, reproduce the published GoldWorm v6 model exactly,
and does the residual-stream frame close, so that later attributions and images mean what they say?

## Materials (pinned)

- Weights: `Uuuuuuniiiiiii/goldworm-v6` `model.safetensors`, sha256
  `31729a9deb9160b045d399e2775c18972ab7ed7465eb8411b05eb801b3753521`.
- Corpus: `roneneldan/TinyStories` `TinyStories-valid.txt`, 19,447,282 bytes, sha256
  `94e431816c4cce81ff71e4408ff8d3bda9a42e8d2663986697c3954288cb38b4` (equals the model card's `corpus.sha256`).
- Protocol: the model's own `eval_results.json` (fetched 2026-11-12): outer_val `[17764697, 19447282)`; positions
  `p_k = 17764697 + 8 + 13k`, 129,429 of them (first 17764705, last 19447269); windowed: one causal window per block of
  16 consecutive positions, reading `[max(17764697, last+1-256), last+1)`, target `p` scored from the logits at input
  offset `p - w_start - 1`; metric mean `-log2 p(true byte)`.

## Port (written from tensor names, no author code executed)

Pre-LN GPT-2 style: `x = wte[b] + wpe[t]`; per block `x += proj(attn(ln1(x)))`, `x += mlp.proj(gelu(mlp.fc(ln2(x))))`,
bias-free linears, `qkv` split q|k|v, 4 heads × 48, causal softmax with scale `1/sqrt(48)`; `logits = head(ln_f(x))`
(head tied to wte). LayerNorm eps `1e-5` (torch default). GELU: exact (erf), torch default; the tanh form is run as a
declared alternative and the one that matches the target is recorded (choosing between two declared forms by the
external target is allowed; no other free choice is).

## Checks and bars (bars from the published record or from arithmetic, none from our own read)

| id | check | bar | source of bar |
|---|---|---|---|
| C1 | windowed outer-val bpb under the published protocol | `|ours − 1.007349| ≤ 1e-3` | model card / `eval_results.json` |
| C2 | position count and range | exactly 129,429 positions, first 17764705, last 19447269 | `eval_results.json` |
| C3 | residual additivity: `x_final = embed + Σ attn_writes + Σ mlp_writes` per position | max relative error ≤ 1e-5 (float32) | arithmetic (pre-LN residual sum) |
| C4 | logit decomposition: with `ln_f` frozen at each position's own mean/std, `Σ_w head(γ⊙(w − μ_w)/σ) + head(β) = logits` over the 13 writes (embed, 6 attn, 6 mlp) | max abs error ≤ 1e-3 logits | arithmetic (affine once μ,σ fixed) |
| C5 | determinism: two passes bit-identical | identical | `eval_results.json` determinism rule |

**Pass** = C1–C5 all hold → the port is the model, the frame closes, VOCAB-1 may be drafted.
**Fail** of C1 → architecture assumption wrong; register as a misfit with the measured gap and stop.

## Measured, not gated (the first residual)

**R1 — linear vs real ablation.** For 2,000 positions drawn from outer_val with seed 20261112: for each of the 12 block
writes, predicted change in the true byte's log-prob when the write is zeroed, using C4's linear decomposition (ln_f
frozen, downstream blocks unchanged), against the realised change from rerunning with that write zeroed. Report per
write: correlation, mean signed error, fraction of sign agreement. No bar: the gap is the nonlinearity downstream of
each write (LayerNorm rescaling, later blocks reading the write), which is exactly what must be accounted for before
attributions are trusted. Large-gap writes are candidate misfits for VOCAB-1's frame.

## Outputs

`cal1.py`, `cal1_result.json` (all checks, R1 table, environment, hashes), committed in this directory.
