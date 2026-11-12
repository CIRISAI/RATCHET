# CAL-G results — the instrument is calibrated on Gemma-4-E2B-it

Preregistration `PREREG_CALG.md` (frozen at RATCHET `6459d61`); script `calg.py`; raw `calg_result.json`. Execution
fixes before any measured quantity (each its own commit): explicit BOS (the tokenizer adds none), batch 1 and a shared
tied head (three out-of-memory stops), gradients off for the whole run (the tied head is a Parameter; logit arithmetic was
building graphs). **Measured:** `google/gemma-4-E2B-it` revision `4adf1574` (sha256 `2db5482b…`), text decoder in
float32 on an RTX 4090 Laptop (2.28 B parameters on the GPU; per-layer embedding table on the CPU), `transformers`
5.22.2, 64 unseen FRESH-1 stories (8,000–8,063), 256 tokens each with BOS.

| check | result | bar |
|---|---|---|
| C1 hashes (model, tokenizer, fresh slice) | all match | exact |
| C2 read-only taps change nothing | identical | identical |
| C3 frame with the per-layer scales (P-FRAME) | max relative error **1.4e-7** (per-layer local 1.7e-7) | ≤ 1e-5 |
| C3′ declared wrong frame (plain sum, scales ignored) | **47.9** relative — fails, as it must | must fail |
| C4 pre-softcap logits rebuilt linearly, final norm frozen | **2.7e-4** | ≤ 1e-3 |
| C5 two passes | identical | identical |

**PASS.** Frame: x_L = s_L·(x_{L−1} + a_L + m_L + p_L) with s_L = `layer_scalar` (0.018 … 0.871; the product over 35
layers is 6.3e-14), so a layer-L write reaches the end scaled by Π_{k≥L} s_k. The hand-over's additivity failure (1.19)
is an instrument fault (taps summed as if the stream were additive), not a property of the model.

## Measured, not gated

- Mean next-token NLL on the 64 stories: **4.2518 nats** (internal reference for later Gemma runs).
- **R1 (KS-DIRECT-ONLY on a second model):** zeroing one layer's three writes, the direct prediction (final norm frozen,
  through the head) against the realised change in the true next token's log-prob, 512 positions. Layers 0–22: direct
  ≈ 0.00 nats against realised 1.4–7.1 nats (corr −0.11 to 0.27) — the scales shrink early writes by up to 10¹³ before
  the end, so they act only through later layers. Layers 26–32: corr 0.35–0.58. **Layer 33: corr 0.01 with a realised
  cost of 11.7 nats.** Layer 34 (last): corr 0.93.
- GGUF top-1 agreement: not run (no llama.cpp runtime here). The deployed Q4 file stores the same scales as separate
  tensors (`blk.N.layer_output_scale`), so it is the same frame in a different precision class.
