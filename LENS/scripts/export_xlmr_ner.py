#!/usr/bin/env python3
"""
R1.3 — Export XLM-RoBERTa NER to ONNX + INT8 quantize.

One-shot tooling. Run once to produce the inference artifacts that the
`ner` feature consumes. Output:

    cirislens-core/models/xlmr-ner.onnx        (~280 MB INT8)
    cirislens-core/models/xlmr-ner-tokenizer.json

The artifacts are NOT committed to git (they're large and binary). CI
fetches/exports them at build time via `make models` or equivalent.

Usage:
    python scripts/export_xlmr_ner.py
    python scripts/export_xlmr_ner.py --model Davlan/xlm-roberta-base-wikiann-ner
    python scripts/export_xlmr_ner.py --output-dir cirislens-core/models
    python scripts/export_xlmr_ner.py --no-quantize  # ship FP32 for accuracy

Dependencies (managed in a venv to avoid system package conflicts):
    pip install transformers optimum[onnxruntime] onnxruntime
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

DEFAULT_MODEL = "Davlan/xlm-roberta-base-wikiann-ner"
DEFAULT_OUTPUT = "cirislens-core/models"


def export_to_onnx(model_id: str, output_dir: Path) -> Path:
    """Export HF token-classification model to ONNX. Returns the .onnx path."""
    try:
        from optimum.onnxruntime import ORTModelForTokenClassification
        from transformers import AutoTokenizer
    except ImportError:
        print("ERROR: install dependencies first:")
        print("  pip install transformers 'optimum[onnxruntime]' onnxruntime")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Exporting {model_id} → {output_dir} (FP32)...")

    model = ORTModelForTokenClassification.from_pretrained(model_id, export=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    onnx_files = list(output_dir.glob("*.onnx"))
    if not onnx_files:
        sys.exit(f"ERROR: no .onnx file produced in {output_dir}")
    return onnx_files[0]


def quantize_int8(input_path: Path, output_path: Path) -> None:
    """INT8-quantize the ONNX model in place. ~4× size reduction, 3-5×
    inference speedup, ~1-3% F1 loss on NER (acceptable for v2 baseline)."""
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError:
        print("ERROR: pip install onnxruntime")
        sys.exit(1)

    print(f"Quantizing {input_path} → {output_path} (INT8)...")
    quantize_dynamic(
        model_input=str(input_path),
        model_output=str(output_path),
        weight_type=QuantType.QInt8,
    )
    in_mb = input_path.stat().st_size / (1024 * 1024)
    out_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  size: {in_mb:.1f} MB → {out_mb:.1f} MB ({out_mb/in_mb*100:.0f}%)")


def write_label_file(model_id: str, output_dir: Path) -> None:
    """Save label list (id → BIO tag) for the Rust side to load via env var."""
    try:
        from transformers import AutoConfig
    except ImportError:
        return

    cfg = AutoConfig.from_pretrained(model_id)
    if not hasattr(cfg, "id2label") or not cfg.id2label:
        return

    labels = [cfg.id2label[i] for i in sorted(cfg.id2label, key=int)]
    label_file = output_dir / "labels.txt"
    label_file.write_text(",".join(labels) + "\n")
    print(f"  wrote {label_file} ({len(labels)} labels: {labels[:3]}...)")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    p.add_argument("--no-quantize", action="store_true")
    p.add_argument("--keep-fp32", action="store_true",
                   help="Keep the unquantized FP32 .onnx alongside the INT8 version")
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    fp32_path = export_to_onnx(args.model, output_dir)
    write_label_file(args.model, output_dir)

    if args.no_quantize:
        final_path = output_dir / "xlmr-ner.onnx"
        shutil.copy2(fp32_path, final_path)
        print(f"FP32 model at: {final_path}")
        return 0

    int8_path = output_dir / "xlmr-ner.onnx"
    quantize_int8(fp32_path, int8_path)

    if not args.keep_fp32:
        for f in output_dir.glob("*.onnx"):
            if f != int8_path:
                f.unlink()
                print(f"  removed {f.name}")

    print(f"\nReady. Set:")
    print(f"  export CIRISLENS_NER_MODEL_PATH={int8_path}")
    print(f"  export CIRISLENS_NER_TOKENIZER_PATH={output_dir / 'tokenizer.json'}")
    print(f"  export CIRISLENS_NER_LABELS=$(cat {output_dir / 'labels.txt'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
