#!/usr/bin/env python3
"""
Build a HuggingFace-ready release candidate from the exported corpus.

Filters:
  1. Drop wbd_deferral retry-loop noise (broken upstream process;
     same frozen `timestamp=2026-04-15T17:05:23.043490+00:00` fixture)
  2. Drop traces with signature_verified=false (broken-signature regime
     before the multi-worker cache fix on 2026-04-23)
  3. Drop traces with timestamp < 2026-03-22 (sparse-field coverage,
     pre-Ally era; harder to interpret without context)

Outputs MANIFEST.json with sha256 + row count per file so HF release
metadata can be generated mechanically.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from datetime import datetime, timezone

SRC = Path(os.path.expanduser("~/RATCHET/corpus"))
DST = Path(os.path.expanduser("~/RATCHET/release/data"))
DST.mkdir(parents=True, exist_ok=True)

CUTOFF = "2026-03-22"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def filter_traces(src_path: Path, dst_path: Path) -> dict:
    """Filter accord_traces.jsonl: keep verified, post-cutoff."""
    kept = dropped_unverified = dropped_old = 0
    with src_path.open() as fin, dst_path.open("w") as fout:
        for line in fin:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = row.get("timestamp", "")
            if ts < CUTOFF:
                dropped_old += 1
                continue
            if row.get("signature_verified") is False:
                dropped_unverified += 1
                continue
            fout.write(line)
            kept += 1
    return {
        "kept": kept,
        "dropped_unverified": dropped_unverified,
        "dropped_pre_cutoff": dropped_old,
    }


def filter_batches(src_path: Path, dst_path: Path) -> dict:
    """Filter accord_trace_batches.jsonl: drop wbd_deferral fixture noise."""
    kept = dropped_wbd = 0
    with src_path.open() as fin, dst_path.open("w") as fout:
        for line in fin:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            # The broken upstream loop uses traces_received in {1, 4} and
            # 0 accepted with 0 rejected_traces. Easier filter: drop batches
            # where traces_accepted=0 AND traces_received <= 4 — those are
            # the wbd_deferral retry-loop pattern.
            recv = row.get("traces_received", 0) or 0
            acc = row.get("traces_accepted", 0) or 0
            rej = row.get("traces_rejected", 0) or 0
            if acc == 0 and recv <= 4 and rej > 0:
                dropped_wbd += 1
                continue
            ts = row.get("batch_timestamp", "")
            if ts < CUTOFF:
                continue
            fout.write(line)
            kept += 1
    return {"kept": kept, "dropped_wbd_loop": dropped_wbd}


def filter_view(src_path: Path, dst_path: Path) -> dict:
    """Filter trace_context.jsonl: keep verified, post-cutoff."""
    kept = dropped = 0
    with src_path.open() as fin, dst_path.open("w") as fout:
        for line in fin:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = row.get("timestamp", "")
            if ts < CUTOFF or row.get("signature_verified") is False:
                dropped += 1
                continue
            fout.write(line)
            kept += 1
    return {"kept": kept, "dropped": dropped}


def filter_keys(src_path: Path, dst_path: Path) -> dict:
    """Public keys: pass-through (these need to verify all traces, including
    older ones for cross-reference). No filter."""
    kept = 0
    with src_path.open() as fin, dst_path.open("w") as fout:
        for line in fin:
            fout.write(line)
            kept += 1
    return {"kept": kept}


def filter_connectivity(src_path: Path, dst_path: Path) -> dict:
    """Connectivity events: post-cutoff only."""
    kept = dropped = 0
    with src_path.open() as fin, dst_path.open("w") as fout:
        for line in fin:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = row.get("timestamp", "")
            if ts < CUTOFF:
                dropped += 1
                continue
            fout.write(line)
            kept += 1
    return {"kept": kept, "dropped_pre_cutoff": dropped}


JOBS = {
    "accord_traces.jsonl":        filter_traces,
    "accord_trace_batches.jsonl": filter_batches,
    "trace_context.jsonl":        filter_view,
    "accord_public_keys.jsonl":   filter_keys,
    "connectivity_events.jsonl":  filter_connectivity,
}

print(f"Filtering {SRC} → {DST} (cutoff: {CUTOFF})")
print()

stats = {}
for fname, fn in JOBS.items():
    src = SRC / fname
    dst = DST / fname
    if not src.exists():
        print(f"  {fname:<32}  SKIP (not in source)")
        continue
    s = fn(src, dst)
    size_mb = dst.stat().st_size / (1024 * 1024)
    stats[fname] = {**s, "size_bytes": dst.stat().st_size}
    print(f"  {fname:<32}  {s}  ({size_mb:.1f} MB)")

print()
print("Generating sha256 manifest...")

manifest = {
    "release_built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "cutoff": CUTOFF,
    "filter_rules": [
        "signature_verified=true required",
        f"timestamp >= {CUTOFF}",
        "wbd_deferral retry-loop fixture excluded",
    ],
    "files": {},
}
for f in sorted(DST.glob("*.jsonl")):
    manifest["files"][f.name] = {
        "sha256": sha256(f),
        "size_bytes": f.stat().st_size,
        "row_count": stats.get(f.name, {}).get("kept", 0),
    }

manifest_path = DST.parent / "MANIFEST.json"
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
print(f"  MANIFEST.json written ({manifest_path})")
print()
print(json.dumps(manifest, indent=2))
