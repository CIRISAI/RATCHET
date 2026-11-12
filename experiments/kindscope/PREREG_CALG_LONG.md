# CAL-G-LONG — the Gemma instrument at the real stage-1 prompt length, with a cached prefix (preregistration)

Frozen 2026-11-12, before any CAL-G-LONG quantity is computed. CAL-G (PASS, `CALG_RESULTS.md`) ran at 256 tokens; EXIT-G
reads states at the answer position of a ~25k-token prompt whose 86,072-character prefix is shared by every item.

misfits: M-DEVICE-CLASS, KS-DIRECT-ONLY, KS-EXACT-ZERO-ACROSS-BATCH, M-FLOOR-UNSTAKED, M-BAR-FROM-THE-READ, M-STALE-INSTRUMENT, M-EXIT-DISCRIMINATOR

plants: P-FRAME carrier: Gemma-4-E2B-it at the real prompt length — the scaled frame (CAL-G) must close on the item's own positions with the shared prefix cached; the plain-sum frame is run beside it and must fail

## Materials (pinned)

- Model, `transformers`, device, float32 as CAL-G (revision `4adf1574`, sha256 `2db5482b…`).
- Rendered prompts: `/home/emoore/blindverb1-rendered-2026-11-12/` (local, from the agent session; its renderer
  commit `6cf3aae0ab7e73adc41bd4d835a1f843b4c550d1` on CIRISAgent `research/render-stage1` is LOCAL ONLY and cannot be
  checked by anyone else yet — this preregistration cites the manifest's digests instead): `prefix.json` sha256
  `92f645c0…4faef`, `fit.jsonl` sha256 `3b4c9325…0624c`. **Only the first 8 rows of fit are used; no label is read;
  confirm.jsonl and reserve.jsonl are not opened.**
- Prompt = the 4 prefix messages then `{"role": "user", "content": row.user}`, tokenised with the model's own chat
  template at the pinned revision, `add_generation_prompt=True`. The shared token prefix = the longest common token prefix
  of the 8 prompts.

## Checks and bars (arithmetic)

| id | check | bar |
|---|---|---|
| L1 | digests of prefix.json and fit.jsonl equal the manifest's | exact |
| L2 | cache equivalence: logits at the last 32 positions, prefix computed once and reused (cache copied per item) vs the whole prompt run uncached, 2 items | max abs ≤ 1e-3 |
| L3 | frame closure with the scales on the item's own (non-prefix) positions, cached run, 8 items (P-FRAME) | max relative ≤ 1e-5 |
| L3′ | plain-sum frame, same positions | must fail |
| L4 | pre-softcap logits rebuilt linearly from the scaled writes, 64 positions | max abs ≤ 1e-3 |
| L5 | determinism: the cached run twice | identical |

Pass = L1–L5 ⇒ EXIT-G may freeze on this instrument.

## Measured, not gated

Token counts (prefix, per item), peak GPU memory, seconds per item (cached), and the full model's own next-token
distribution at the answer position after `{"selected_action": "` is NOT computed here (that is EXIT-G's read).

## Outputs

`calg_long.py` (committed before its first run), `calg_long_result.json`, results appended to `CALG_RESULTS.md`.

## Amendment 1 (2026-11-12, before any check other than L1 and the token counts)

The first launch reached L1 (digests match; shared prefix 20,742 tokens; items 21,670–21,687 tokens) and then ran out of
memory: `transformers` had loaded the text decoder with the **eager** attention kernel (the multimodal class falls back
to eager because its audio tower does not support SDPA), which materialises a 20 GB score matrix at 20,742 tokens. The
text decoder supports SDPA and Gemma-4 has no attention logit softcapping, so the decoder is switched to
`attn_implementation = "sdpa"` for this run. Consequence, stated: the kernel differs from CAL-G's (eager at 256 tokens),
so L3–L5 calibrate the **SDPA** instrument at long context, and EXIT-G must use the same kernel. Nothing else changes.

## Amendment 2 (2026-11-12, before any check other than L1 and the token counts) — Amendment 1's diagnosis was wrong

Read from the traceback after Amendment 1's rerun failed identically: the forward pass **completed** at 20,742 tokens
under the eager kernel; the 20.26 GiB allocation was the **logits** (20,742 positions × 262,144 vocabulary × 4 bytes) in
the script's own `last_hidden_state @ head.T`. Amendment 1 blamed the attention kernel without reading the traceback.
**Corrected:** the kernel goes back to CAL-G's (eager, as loaded), so no kernel changes between CAL-G and CAL-G-LONG; the
script computes logits only at the positions a check reads (none for the prefix pass; the last 32 for L2; the item's
last 8 for L4; the item's own positions' final 32 for L5). Amendment 1 is withdrawn and kept.

## Amendment 3 (2026-11-12, before any check other than L1 and the token counts)

With logits restricted (Amendment 2), the eager prefix pass completed but **peaked at 15.96 GB** of 15.57 GB usable — the
eager kernel materialises each full-attention layer's score matrix (8 heads × 20,742² × 4 B ≈ 13.8 GB, transient) — and
the item pass then ran out of memory. So Amendment 1's remedy (SDPA) is needed, for a reason measured this time: peak
memory, not a guessed allocation. The text decoder runs **SDPA** for CAL-G-LONG and EXIT-G, and a bridge check is added
first: **L0** — at CAL-G's own regime (the first 2 FRESH-1 stories of CAL-G's set, 256 tokens), SDPA and eager
pre-softcap logits agree within 1e-3 (max abs). If L0 fails, the run is VOID and the kernel question is reopened.

## Amendment 4 (2026-11-12, before any check other than L0, L1 and the token counts)

Measured (memory probe, no check read): model 9.12 GB; per-layer peak during a single 20,742-token pass 13.2 GB at the
full-attention layers and **15.3 GB at the sliding-window layers** — the transient is MLP intermediates and the full-size
sliding mask over 20,742 positions, not attention scores (Amendments 1 and 3 named the wrong cause; the SDPA kernel stays,
bridged by L0 = 4.4e-4 at 256 tokens). Remedy: **chunked prefill** — the shared prefix is fed through the KV cache in
chunks of 2,048 tokens (how a deployed runtime prefills). A single uncached pass at full length does not fit this GPU,
so **L2 is restated:** cache equivalence is checked at a length that fits — the first 6,144 tokens of 2 item prompts,
chunked prefill of the first 6,112 + cached last 32 vs one uncached pass of all 6,144, max abs ≤ 1e-3 on those 32
positions' logits. Full-length equivalence then rests on the same mechanism (stated, not measured at 21k). L3–L5 run on
the chunk-prefilled cache. Nothing else changes.

## Amendment 5 (2026-11-12) — rerun on the render under the default agent template

The folder this calibration cited is the evaluator's BASE prompt (no agent template); no deployed agent sends it
(cirisagent-fd, 2026-11-12). The PASS above stands as a calibration at that length on the base render. The same checks,
unchanged (SDPA, chunked prefill, L0–L5 and their bars), are rerun on the render EXIT-G will use:
`/home/emoore/blindverb1-rendered-default-template-2026-11-12/` — renderer commit
`bfe2a7413d08edf8d4b36356452b7fb72d13b9b7` (local, unpushed), template `default.yaml` (Ally) sha256
`f3f38b513388b51a1409f6abc52c69db2115b40885a09a17a25390c35623b8e6`, `prefix.json` sha256
`ae3a8ba89ceab24ee5c7dfdf180a6cc9547b6cbafa22f458795d07655270a8df`, `fit.jsonl` sha256
`b827a526becb06ceb45646c0815d8a3fbfaf5ddb4b7a78efbaad3fda757a22d7`. First 8 fit rows, no label read, confirm and reserve
unopened. Output `calg_long_template_result.json`. The render is unvalidated against a live captured call until the
agent session's diff lands; that is a property of the prompts, not of this calibration.
