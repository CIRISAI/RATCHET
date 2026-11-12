# CAL-G — calibrate the instrument on Gemma-4-E2B-it (preregistration)

Frozen 2026-11-12, before any Gemma quantity is computed. The bridge from GoldWorm (CAL-1 … PARALLAX-2) to the kinds on
the model the agent runs. The hand-over's instrument (llama.cpp taps on a Q4 GGUF) failed its own additivity check
(relative error 1.19); reading `transformers` 5.22.2's `Gemma4TextDecoderLayer.forward` shows why a plain sum of block
outputs cannot close: every layer ends with `hidden_states *= self.layer_scalar`, rescaling the whole residual stream.

misfits: KS-DIRECT-ONLY, M-DEVICE-CLASS, M-EXIT-DISCRIMINATOR, M-FLOOR-UNSTAKED, M-BAR-FROM-THE-READ, G-SMALLEST-PASSING-STEP, G-LENS-IS-NOT-THE-PATH, M-STALE-INSTRUMENT

plants: P-FRAME carrier: Gemma-4-E2B-it text decoder — the tapped writes, each carried forward through the later layer scalars, must sum to the final pre-norm residual (C3); a frame that ignores the scalars is run beside it as the declared wrong frame and must fail unless every scalar is 1

## Known at freeze

The hand-over's Q4 GGUF (`/home/emoore/kindscope-handoff-2026-11-11/model/gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf`, sha256
`e5310072…6889`, restored to the bundle by cirisagent-fd 2026-11-12) stores the scalars as 35 separate tensors
`blk.N.layer_output_scale.weight`; the first six read 0.0209, 0.2354, 0.7891, 0.3105, 0.5039, 0.6289 (tensor list read
before this freeze; no forward run). So the deployed file carries the same per-layer rescaling, and whether a
llama.cpp `l_out` tap is taken before or after it decides what the hand-over measured.

## Materials (pinned)

- `google/gemma-4-E2B-it` at revision `4adf15748de9e1a0097b6274cee9c15836c41c37`, `model.safetensors` sha256
  `2db5482b20d746879bb3ef79b5203e9075a2e2b98f54ec7c2f281c1477ddc550`, tokenizer.json sha256
  `ab26b2add729e6f645bd869f228f14190a06615538f3be6f7786f33d13d78ad2`. Apache-2.0. Text decoder only (vision and audio
  towers not loaded).
- `transformers` 5.22.2 (reference implementation), torch 2.11.0+cu128, RTX 4090 Laptop (16 GB).
- Text: FRESH-1 (TinyStories-train slice, sha256 `3232df12…`), stories 8,000–8,063 of the kept list (unread), each
  tokenised with the Gemma tokenizer and truncated to 256 tokens, BOS prepended.
- **Precision (C0):** the text decoder runs in float32 if it fits on the GPU with the per-layer embedding table kept on
  the CPU; otherwise bfloat16, and every bar below marked (fp32) is replaced by its bf16 version
  (relative 1e-2). Which one ran is recorded (M-DEVICE-CLASS).

## Frame

Per layer L: x ← s_L · (x + a_L + m_L + p_L), where a_L = post-attention-norm output, m_L = post-feedforward-norm output,
p_L = post-per-layer-input-norm output, s_L = `layer_scalar`; x₀ = embedding × embed_scale. Hence
x_final = (Π_L s_L)·x₀ + Σ_L (Π_{k ≥ L} s_k)·(a_L + m_L + p_L).

## Checks and bars (arithmetic; none from a read)

| id | check | bar |
|---|---|---|
| C1 | weights and tokenizer hashes as pinned | exact |
| C2 | the reference forward with read-only tap hooks reproduces the un-hooked reference logits on the 64 stories (taps observe, never modify; writes are taken at the module outputs named in the frame — no re-implementation of attention, which would add error the calibration exists to catch) | identical |
| C3 | frame closure with the scalars (P-FRAME) | max relative error ≤ 1e-5 (fp32) |
| C3′ | the declared wrong frame (plain sum, scalars ignored) | reported; must FAIL unless all s_L = 1 |
| C4 | pre-softcap logits rebuilt linearly from the scaled writes with the final norm frozen at each position's own scale | max abs ≤ 1e-3 (fp32) |
| C5 | determinism: two passes identical | identical |

Pass = C1–C5 ⇒ the instrument is calibrated on Gemma; PARALLAX-G may be drafted.

## Measured, not gated

- The 35 `layer_scalar` values, and Π over layers.
- R1 on Gemma (KS-DIRECT-ONLY): per layer, direct (frozen final norm, through the head) vs realised change in the true
  next token's log-prob when that layer's three writes are zeroed, 512 positions.
- Mean next-token NLL on the 64 stories (an internal reference number for later runs; no external target exists for this
  text).
- If the hand-over's Q4 GGUF is available: top-1 agreement with it on the same positions (the deployed artifact is a
  different device class; G-SMALLEST-PASSING-STEP).

## Outputs

`calg.py` (committed before its first run), `calg_result.json`, `CALG_RESULTS.md`.
