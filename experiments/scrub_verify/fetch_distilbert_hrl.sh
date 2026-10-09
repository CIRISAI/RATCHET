#!/usr/bin/env bash
# Assemble a loadable DistilBERT-multilingual NER dir for `ciris-server`'s
# CIRISLENS_NER_BACKBONE=distilbert. The wheel's default model id
# (Davlan/distilbert-base-multilingual-cased-ner-hrl) has no tokenizer.json, so the
# loader 404s (CIRISServer#755 addendum). Xenova/… is a port of the same checkpoint
# and ships tokenizer.json. Vocab parity is checked token-for-token before use.
set -euo pipefail
M="${1:?usage: fetch_distilbert_hrl.sh <dest-dir>}"; mkdir -p "$M"; cd "$M"
D=https://huggingface.co/Davlan/distilbert-base-multilingual-cased-ner-hrl/resolve/main
X=https://huggingface.co/Xenova/distilbert-base-multilingual-cased-ner-hrl/resolve/main
[ -s config.json ]       || curl -sSL -o config.json "$D/config.json"
[ -s vocab.txt ]         || curl -sSL -o vocab.txt "$D/vocab.txt"
[ -s model.safetensors ] || curl -sSL -o model.safetensors "$D/model.safetensors"   # ~539 MB
[ -s tokenizer.json ]    || curl -sSL -o tokenizer.json "$X/tokenizer.json"
python3 -I - <<'PY'
import json
d=[l.rstrip("\n") for l in open("vocab.txt",encoding="utf8")]
inv={v:k for k,v in json.load(open("tokenizer.json",encoding="utf8"))["model"]["vocab"].items()}
x=[inv[i] for i in range(len(inv))]
assert d==x, f"vocab mismatch: {len(d)} vs {len(x)}"
print("vocab parity OK:", len(d), "tokens")
PY
echo "export CIRISLENS_NER_BACKBONE=distilbert CIRISLENS_NER_MODEL_DIR=$M"
