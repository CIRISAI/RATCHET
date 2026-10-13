# Release record — CIRIS Reasoning Trace Corpus v2

| | |
|---|---|
| Hub | https://huggingface.co/datasets/CIRISAI/reasoning-traces-v2 |
| Upload commit | `a01cc5e7fc682194e83a0beb5015a63268921db2` (2026-10-13) |
| Visibility at upload | **private** — flip to public is the owner's action |
| Source export | canonical trace store, 2026-10-08, de-mocked (7,316 traces / 105,388 events) |
| Scrub policy | `ratchet/full_traces+postpass/v3` — `experiments/scrub_verify/run_scrub.py` |
| Verification | all three data files + README byte-identical by LFS sha256; `MANIFEST.json` by content; card parsed with splits `events` / `llm_calls` / `signatures` |
| Checksums | `SHA256SUMS` (committed) |
