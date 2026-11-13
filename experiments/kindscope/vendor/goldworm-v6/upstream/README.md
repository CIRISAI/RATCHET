---
library_name: goldworm-v6
datasets:
- roneneldan/TinyStories
tags:
- pytorch
- tokenizer-free
- byte-level
- small-language-model
- python
- byte-level-transformer
- language-model
- tiny-stories
- efficient
license: mit
pipeline_tag: text-generation

model-index:
- name: goldworm-v6
  results:
  - task:
      type: text-generation
      name: Language modeling
    dataset:
      type: roneneldan/TinyStories
      name: TinyStories (held-out outer-val part of valid)
    metrics:
    - type: bits_per_byte
      value: 1.0073
      name: bits/byte (self-reported)

---

# GoldWorm v6 — Byte-Level Causal Transformer (2.76M params)

![license](https://img.shields.io/badge/license-MIT-green) ![params](https://img.shields.io/badge/params-2.76M-gold) ![vocab](https://img.shields.io/badge/vocab-256%20bytes-black) [![video](https://img.shields.io/badge/YouTube-explainer-red)](https://youtu.be/Wuh3_bJSxyY)

<h3>Try this model</h3>
<iframe src="https://huggingface.co/spaces/Uuuuuuniiiiiii/Gold_Lab/embed" width="100%" height="450" frameborder="0" scrolling="no"></iframe>
<p><a class="md-button" href="https://huggingface.co/spaces/Uuuuuuniiiiiii/Gold_Lab">Open Gold Lab Space</a></p>

**Verified result:** full BPC `1.007349` (step 8000, held-out outer-val region of TinyStories-valid, 129,429 fixed positions; self-reported, see `eval_results.json`).

*Dataset:* `roneneldan/TinyStories` (TinyStories-valid.txt, 19.4 MB, sha256 `94e431816c4cce81...`).
*Protocol:* windowed stride-16 ctx-256 over fixed evaluation set; determinism verified (two bit-identical passes, `max_abs_diff_per_position_bits = 0.0`).

## Model facts
- Type: `ByteTransformer`
- Vocab: 256 (byte-level)
- Config: `vocab_size=256, d_model=192, n_layer=6, n_head=4, ffn_mult=4, ctx=256, tie_embeddings=True, dropout=0.0`
- Params: 2,757,504 unique (2.76M; `model.safetensors` stores the tied embedding twice, so its tensors sum to 2,806,656)
- Checkpoint: selected step-8000 checkpoint → `model.safetensors`
- Selection (inner-val): step 8000, `inner_val_windowed_bpc = 1.051187`
- Full-val (outer): step 8000, `full_bpc = 1.007349`
- Baselines: unigram 4.437 / bigram 3.317 (add-k 0.01, train-matched)
- Gate: well below frozen v1 reference (2.566)

## Files
- `model.safetensors` — exported weights (tied embeddings cloned; load with `ByteTransformer(config)`)
- `config.json` — architecture config
- `eval_results.json` — full outer-val result
- `demo/app.py` — Streamlit text-generation demo

## Use with Streamlit demo
```bash
pip install torch safetensors streamlit numpy
streamlit run demo/app.py
```

Prompt is drawn from `corpus/corpus.txt` outer-val region; generation uses temperature / top-k sampling.

## How to load (Python)
```python
import torch
from safetensors.torch import load_file

cfg = TransformerConfig(vocab_size=256, d_model=192, n_layer=6, n_head=4, ffn_mult=4, ctx=256, tie_embeddings=True, dropout=0.0)
model = ByteTransformer(cfg)
state = load_file("model.safetensors")
model.load_state_dict(state)
model.eval()
```

## Comparisons (protocol-dependent, self-reported)
On the same windowed byte metric, GPT-2 124M was reported at `BYTE_BPC = 1.142985` (`TOKEN_BPC = 9.152823`). This comparison depends on the evaluation protocol (context, windowing, byte vs token normalisation) and was not independently reproduced; a separate goldworm-coder benchmark on a 256 KB TinyStories-validation slice measured GPT-2 124M at 1.1203. Treat it as indicative only — GPT-2 is a general model evaluated zero-shot, v6 is trained in-domain.

## Limitations
- Trained on the first part of TinyStories-valid; evaluated only on held-out outer-val positions of the same file (in-domain, easy domain).
- No same-size BPE/tokenized baseline yet.
- Generation quality is TinyStories-level (simple children's stories); not a general-purpose model.
- All numbers are self-reported, not independently re-measured.

## Verification
- Chunk shape / parity / replication checks verified per `LLM_PARITY_ROADMAP.md`
- Independent replication (step 8000) confirmed bit-exact with selection frozen before any outer_val read

## Version label
GoldWorm v6, 2.76M params, single release

## User manual
See docs/USER_MANUAL.md for Windows + Linux install + load + eval reproduction

See also: Uuuuuuniiiiiii/TinyStories dataset card

## Links
- Gold Lab Space: https://huggingface.co/spaces/Uuuuuuniiiiiii/Gold_Lab
- 🎬 Explainer video: https://youtu.be/Wuh3_bJSxyY
- 🎧 Podcast: https://youtu.be/ijVZXPYTsXU

## Citation
```bibtex
@misc{goldworm_v6_2026,
  title  = {GoldWorm v6: a 2.76M-parameter byte-level language model},
  author = {Gold Lab},
  year   = {2026},
  doi    = {10.57967/hf/10547},
  url    = {https://huggingface.co/Uuuuuuniiiiiii/goldworm-v6}
}
```
