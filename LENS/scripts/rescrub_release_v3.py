#!/usr/bin/env python3
"""
Re-scrub the HF release corpus through the Rust v2 scrubber
(`cirislens_core.scrub_trace`).

The previous rescrub variants (`data_rescrubbed`, `data_rescrubbed_v2`)
used the Python pii_scrubber, which leaves ~1,460 historical-year matches
in the output. The Rust scrubber v2's walker applies the year + identifier
regexes to **every** string in the trace, and the `count_year_residue`
invariant rejects any output that still contains a 1700-2023 match.

Usage:
    python3 scripts/rescrub_release_v3.py \
        [--src ~/RATCHET/release/data] \
        [--dst ~/RATCHET/release/data_rescrubbed_v3] \
        [--level detailed]

Default level is `detailed` — applies regex globally without requiring
NER weights. Run with `--level full_traces` only if the v2 NER backend
is configured (env vars `CIRISLENS_NER_MODEL_DIR` or `CIRISLENS_NER_MODEL_ID`).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

import cirislens_core

YEAR_RE = re.compile(r"\b(?:1[7-9]\d{2}|20[0-1]\d|202[0-3])\b")


def walk_string_leaves(value):
    """Yield every string leaf in a JSON value — matches Rust's
    `count_year_residue` semantics. Walking the raw JSONL line text
    instead would double-count years inside numeric fields like
    `audit_sequence_number: 1995` or `carbon_grams: 18.1995`, which
    aren't privacy concerns."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from walk_string_leaves(v)
    elif isinstance(value, list):
        for v in value:
            yield from walk_string_leaves(v)


def count_year_residue(line: str) -> int:
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        return 0
    return sum(len(YEAR_RE.findall(s)) for s in walk_string_leaves(row))

# Files to push through the scrubber. Everything else is byte-copied.
TRACE_FILES = ("accord_traces.jsonl", "trace_context.jsonl")
PASSTHROUGH_FILES = (
    "accord_public_keys.jsonl",
    "accord_trace_batches.jsonl",
    "connectivity_events.jsonl",
)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rescrub_jsonl(src: Path, dst: Path, level: str) -> dict:
    """Re-scrub a JSONL file. Returns counters."""
    kept = rejected = scrub_errors = year_pre = year_post = 0
    rejection_samples: list[str] = []

    with src.open() as fin, dst.open("w") as fout:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            year_pre += count_year_residue(line)

            try:
                result = cirislens_core.scrub_trace(line, level)
            except Exception as e:
                # Year-residue / probe-match / scrub error → drop and keep
                # going. Sample the first few rejections so the operator
                # has a hint about what caused it.
                rejected += 1
                if len(rejection_samples) < 5:
                    rejection_samples.append(f"line {line_no}: {e}")
                continue

            scrubbed_str = result["trace"]
            year_post += count_year_residue(scrubbed_str)
            fout.write(scrubbed_str + "\n")
            kept += 1

    return {
        "kept": kept,
        "rejected": rejected,
        "scrub_errors": scrub_errors,
        "year_residue_before": year_pre,
        "year_residue_after": year_post,
        "rejection_samples": rejection_samples,
    }


def passthrough(src: Path, dst: Path) -> dict:
    shutil.copy(src, dst)
    rows = sum(1 for _ in dst.open())
    return {"rows": rows}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=os.path.expanduser("~/RATCHET/release/data"))
    p.add_argument("--dst", default=os.path.expanduser("~/RATCHET/release/data_rescrubbed_v3"))
    p.add_argument("--level", default="detailed", choices=["detailed", "full_traces"])
    args = p.parse_args()

    src_dir = Path(args.src)
    dst_dir = Path(args.dst)
    dst_dir.mkdir(parents=True, exist_ok=True)

    print(f"Re-scrubbing {src_dir} → {dst_dir}  (level={args.level})")
    print(f"Scrubber: cirislens_core.scrub_trace (Rust v2)")

    manifest: dict = {
        "release_built_at": datetime.now(UTC).isoformat(),
        "scrubber": f"cirislens_core.scrub_trace (Rust v2, level={args.level})",
        "filter_rules": [
            "regex pass applied to every string (year, year-bearing identifier, "
            "email, phone, IPv4, URL, SSN, credit card)",
            "year-residue invariant: rejects any row whose scrubbed output "
            "still contains a 1700-2023 match",
        ],
        "files": {},
        "totals": {"kept": 0, "rejected": 0, "year_residue_before": 0, "year_residue_after": 0},
    }

    for fname in TRACE_FILES:
        src_p = src_dir / fname
        dst_p = dst_dir / fname
        if not src_p.exists():
            print(f"  {fname}: not present in src, skipping")
            continue
        stats = rescrub_jsonl(src_p, dst_p, args.level)
        manifest["files"][fname] = {
            "sha256": sha256(dst_p),
            "size_bytes": dst_p.stat().st_size,
            "row_count": stats["kept"],
            "rejected": stats["rejected"],
            "year_residue_before": stats["year_residue_before"],
            "year_residue_after": stats["year_residue_after"],
        }
        if stats["rejection_samples"]:
            manifest["files"][fname]["rejection_samples"] = stats["rejection_samples"]
        for k in ("kept", "rejected", "year_residue_before", "year_residue_after"):
            manifest["totals"][k] += stats[k]
        print(
            f"  {fname}: kept={stats['kept']}  rejected={stats['rejected']}  "
            f"year-residue {stats['year_residue_before']} → {stats['year_residue_after']}"
        )

    for fname in PASSTHROUGH_FILES:
        src_p = src_dir / fname
        dst_p = dst_dir / fname
        if not src_p.exists():
            print(f"  {fname}: not present in src, skipping")
            continue
        stats = passthrough(src_p, dst_p)
        manifest["files"][fname] = {
            "sha256": sha256(dst_p),
            "size_bytes": dst_p.stat().st_size,
            "row_count": stats["rows"],
            "passthrough": True,
        }
        print(f"  {fname}: passthrough ({stats['rows']} rows)")

    manifest_path = dst_dir / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nWrote {manifest_path}")
    print(f"Totals: kept={manifest['totals']['kept']}  rejected={manifest['totals']['rejected']}")
    print(
        f"Year residue: {manifest['totals']['year_residue_before']} → "
        f"{manifest['totals']['year_residue_after']} "
        f"({100*(1 - manifest['totals']['year_residue_after']/max(1, manifest['totals']['year_residue_before'])):.1f}% reduction)"
    )

    if manifest["totals"]["year_residue_after"] > 0:
        print(
            f"\n⚠ {manifest['totals']['year_residue_after']} year matches remain in scrubbed output. "
            f"Investigate before publishing.",
            file=sys.stderr,
        )
        return 1
    print("\n✓ year-residue invariant clean — corpus ready for publication.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
