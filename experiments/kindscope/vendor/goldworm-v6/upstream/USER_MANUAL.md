# GoldWorm v6 User Manual

## What it is
GoldWorm v6 is a small standalone ByteTransformer language model.
- Architecture: ByteTransformer
- d_model: 192, n_layer: 6, n_head: 4, ctx: 256, tie_embeddings: true
- Parameters: 2,757,504
- Evaluation: BPC = 1.007349 at step 8000 on TinyStories (validation)

## Prerequisites
- Python 3.8+ (3.11+ recommended)
- PyTorch 2.0+
- `transformers` (Hugging Face) or direct `safetensors` loading

## Run on Windows (PowerShell)
```powershell
# Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install torch safetensors numpy

# Download model files from https://huggingface.co/Uuuuuuniiiiiii/goldworm-v6
# (model.safetensors, config.json, eval_results.json)
# Place them in a folder, e.g. C:\models\goldworm-v6

# Run inference via PowerShell script
python -c "
import torch, json
from safetensors.torch import load_file
cfg = json.load(open('config.json'))
w = load_file('model.safetensors')
print('Loaded GoldWorm v6 with', sum(t.numel() for t in w.values()), 'parameters')
"
```

## Run on Linux (bash)
```bash
# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch safetensors numpy

# Download model files from https://huggingface.co/Uuuuuuniiiiiii/goldworm-v6
# Place them in a folder, e.g. /models/goldworm-v6

# Run inference via bash script
python3 -c "
import torch, json
from safetensors.torch import load_file
cfg = json.load(open('config.json'))
w = load_file('model.safetensors')
print('Loaded GoldWorm v6 with', sum(t.numel() for t in w.values()), 'parameters')
"
```

## Configuration
- Model hyperparameters are fixed in `config.json`: d_model=192, n_layer=6, n_head=4, ctx=256, tie_embeddings=true.
- Context window: 256 tokens.
- Byte-level tokenization.
- `demo/app.py` provides a reference implementation for the architecture (if present in repo).

## Evaluation reproduction
The model was evaluated on TinyStories (validation set). To reproduce:
1. Load `model.safetensors` using the config in `config.json`.
2. Run the model over the TinyStories validation set.
3. Compute byte-perplexity (BPC). Expected: BPC ≈ 1.007349 at step 8000.
4. `eval_results.json` contains logged metrics from the evaluation run (step, BPC, timestamps).

Reference result (from repo): BPC = 1.007349 at step 8000.

## Known issues/honest limits
- A BPC of 1.007349 indicates high loss; the model is an early prototype with limited generalization.
- README references a missing file `demo/requirements.txt` (doc drift); install deps manually using the list above.
- README contains local Windows paths (`C:/Users/Student/...`); these are not portable. Use relative paths.
- The model is small (2.7 M params) and does not produce coherent output without further training.
- No real-time demo is bundled; `demo/app.py` (if present) is the primary interface.

## Version history
| Version | Label | Date | Notes |
|---------|-------|------|-------|
| v6 | GoldWorm v6 | 2026-09-12 | ByteTransformer, 2.7 M params, BPC 1.007349 @ step 8000 on TinyStories val. Initial standalone release. |

## Where model weights come from
- Model weights (`model.safetensors`, 2.7 M parameters) originate from the public repository `Uuuuuuniiiiiii/goldworm-v6`.
- No stub files are present in this repo; the model is a standalone safetensor checkpoint.
- Download link: https://huggingface.co/Uuuuuuniiiiiii/goldworm-v6/blob/main/model.safetensors