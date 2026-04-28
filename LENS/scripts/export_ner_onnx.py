#!/usr/bin/env python3
"""
Export an HF token-classification model to INT8-quantized ONNX, ready for
the cirislens_core ort backend.

Recipe:
    optimum-cli export onnx --model <repo-id> --task token-classification
                            --optimize O2 <dir>
    onnxruntime.quantize_dynamic(per_channel=True, weight=QInt8,
                                 DefaultTensorType=FLOAT)

The DefaultTensorType extra option is required because optimum's O2
graph optimization fuses ops whose output tensor types the dynamic
quantizer can't infer; tagging FLOAT as the fallback lets quantization
proceed.

Usage:
    scripts/export_ner_onnx.py \
        --model Davlan/distilbert-base-multilingual-cased-ner-hrl \
        --out /opt/ciris/lens/models/distilbert_int8

Then point the runtime at it:
    CIRISLENS_NER_ORT_DIR=/opt/ciris/lens/models/distilbert_int8
    CIRISLENS_NER_BACKBONE=ort
    ORT_DYLIB_PATH=/path/to/libonnxruntime.so.<version>
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        default="Davlan/distilbert-base-multilingual-cased-ner-hrl",
        help="HuggingFace repo id or local path to a token-classification model",
    )
    p.add_argument("--out", required=True, help="output directory for the ONNX-INT8 bundle")
    p.add_argument(
        "--optimize",
        default="O2",
        choices=["O1", "O2", "O3"],
        help="optimum-cli graph optimization level (O3 fuses ops the quantizer doesn't handle)",
    )
    p.add_argument(
        "--quantize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="apply INT8 dynamic quantization (default: yes)",
    )
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        fp32_dir = Path(tmpdir) / "fp32"
        # 1. Export to ONNX with graph optimization.
        print(f"[1/2] Exporting {args.model} → ONNX (optimize={args.optimize})")
        cmd = [
            "optimum-cli",
            "export",
            "onnx",
            "--model",
            args.model,
            "--task",
            "token-classification",
            "--optimize",
            args.optimize,
            str(fp32_dir),
        ]
        subprocess.run(cmd, check=True)

        if args.quantize:
            # 2. INT8 dynamic quantization, per-channel for slightly better accuracy.
            print("[2/2] Quantizing to INT8 (per-channel, dynamic)")
            from onnxruntime.quantization import QuantType, quantize_dynamic
            import onnx

            quantize_dynamic(
                model_input=str(fp32_dir / "model.onnx"),
                model_output=str(out_dir / "model.onnx"),
                weight_type=QuantType.QInt8,
                per_channel=True,
                extra_options={"DefaultTensorType": onnx.TensorProto.FLOAT},
            )
        else:
            shutil.copy(fp32_dir / "model.onnx", out_dir / "model.onnx")

        # 3. Copy tokenizer + config files.
        for fname in (
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.txt",
            "ort_config.json",
        ):
            src_path = fp32_dir / fname
            if src_path.exists():
                shutil.copy(src_path, out_dir)

        # tokenizer.json is sometimes missing on legacy models (e.g.
        # Davlan/distilbert-*); fall back to the base BERT-multilingual.
        if not (out_dir / "tokenizer.json").exists():
            print("  tokenizer.json missing; trying base bert-base-multilingual-cased")
            from transformers import AutoTokenizer

            base = "bert-base-multilingual-cased"
            tok = AutoTokenizer.from_pretrained(base)
            tok.save_pretrained(out_dir)

    size_mb = (out_dir / "model.onnx").stat().st_size / 1e6
    print(f"\nDone. Model bundle: {out_dir} ({size_mb:.0f} MB)")
    print(f"\nRun with:")
    print(f"  CIRISLENS_NER_BACKBONE=ort")
    print(f"  CIRISLENS_NER_ORT_DIR={out_dir}")
    print(f"  ORT_DYLIB_PATH=<path-to-libonnxruntime.so>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
