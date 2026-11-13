# GoldWorm v6 — vendored copy for exploration

A copy of the small model the kindscope series reads, kept in the repository so an agent with no access to the
original download can load it, run it and edit it.

## What this is, and whose

- **Model:** GoldWorm v6, a byte-level causal transformer: 2,757,504 parameters, 6 blocks, width 192, 4 heads,
  context 256 bytes, vocabulary 256 (raw bytes, no tokenizer), output table tied to the input embedding.
  Trained by its authors on TinyStories.
- **Source:** Hugging Face `Uuuuuuniiiiiii/goldworm-v6`, revision `878be4a13a66dd299c4ea82db088b41ecf630897`
  (fetched 2026-11-11 and 2026-11-12).
- **Licence:** MIT, Copyright (c) 2026 GoldWorm / Andrew (GoldSnnail) Project. The licence text is `LICENSE` in
  this directory and covers the upstream files.
- **Not ours:** `model.safetensors`, `LICENSE` and everything under `upstream/` are unmodified upstream files.
- **Ours:** `goldworm.py` and this README. `goldworm.py` is the kindscope port, written from the checkpoint's tensor
  layout; no upstream code is used or executed.

| file | sha256 |
|---|---|
| `model.safetensors` (11,231,288 bytes) | `31729a9deb9160b045d399e2775c18972ab7ed7465eb8411b05eb801b3753521` |
| `LICENSE` | `c116ca2ad7fd2964fe09ccfcaca797b8500f945eb801e17a370f2d27e2136f57` |
| `upstream/README.md` (the model card) | `6f30f96ba376bd33960b3a31e61ec8cb9dd648db1d5a4a95a3b0e04de3178c37` |
| `upstream/config.json` | `fd3a417a39ab9148d6401e115f89f078d96c39cd8508e4d112aad4750be9dafb` |
| `upstream/eval_results.json` | `fa75d4654be13160e290eb8704a69fc1c240743bd7fa0f0a5789cdd342638ebc` |
| `upstream/USER_MANUAL.md` | `8a54475a514370c4aa412f2de01ba01bc9ea303e97aa518ef8ed809438b92376` |

## Use

Needs `torch` and `safetensors`. Runs on CPU.

```
python goldworm.py check                 # verifies the sha256, lists the 52 tensors
python goldworm.py score FILE            # bits per byte of a text file
python goldworm.py generate "Once upon a time" --n 200 --temp 0.8
```

```python
import goldworm as gw
sd = gw.load()                                   # dict of tensors; change them and call forward again
logits, writes = gw.forward(sd, gw.encode("Once upon a time"))
# writes["attn3"], writes["mlp5"], ...: what each block adds to the stream, [1, T, 192]
```

## What has been checked (measured, this copy)

- `goldworm.py check` reports the pinned sha256.
- `goldworm.forward` and the series' differentiable port (`../../edit1.py` `fwd`) give identical logits on a test
  string (maximum difference 0.0).
- CAL-1 (`../../CAL1_RESULTS.md`) matched the published bits per byte with this port, using the tanh form of GELU.

## What is not here

- **The text.** TinyStories (`roneneldan/TinyStories`) is not vendored. The series uses `TinyStories-valid.txt`
  (19,447,282 bytes, sha256 `94e431816c4cce81ff71e4408ff8d3bda9a42e8d2663986697c3954288cb38b4`) and the first 20 MiB
  of `TinyStories-train.txt`. The model reads other kinds of text badly (3.70 bits per byte on the ciris-website
  stories against 0.93 on its own kind).
- **The 27 harnesses** (`Uuuuuuniiiiiii/goldworm-v6-harnesses`).
- Most series scripts in `../../` read the weights through a cache in a session scratch directory (`KS_SCRATCH`),
  not from this copy. Use `goldworm.py` here for standalone work.

## Where to start reading

`../../GOAL.md` (the series and its framing), `../../TORQUE_NOTES.md` (the latest session: logic against training),
`../../MISFITS.md` (mistakes already made, with the rule each one left behind).
