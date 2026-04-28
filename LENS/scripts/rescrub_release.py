#!/usr/bin/env python3
"""
Re-apply the (tightened) PII scrubber to a release-candidate directory.

Used when the scrubber rules change and the existing release was scrubbed
under older rules. Reads each *.jsonl from --src, walks JSONB fields,
applies scrub_text where field names match SCRUB_FIELDS, and writes
re-scrubbed JSONL to --dst.

Idempotent: running twice produces the same output.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Make the api/ module importable
sys.path.insert(0, str(Path(__file__).parent.parent / "api"))

from pii_scrubber import (  # noqa: E402
    SCRUB_FIELDS,
    scrub_text,
    scrub_dict_recursive,
)


def rescrub_row(row: dict) -> dict:
    """Walk a row's JSONB fields and re-scrub any string fields whose name
    is in SCRUB_FIELDS. Uses scrub_dict_recursive which already filters by
    field name."""
    return scrub_dict_recursive(row)


def process_file(src: Path, dst: Path) -> tuple[int, int]:
    """Returns (rows_processed, rows_modified)."""
    n = changed = 0
    with src.open() as fin, dst.open("w") as fout:
        for line in fin:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            new_row = rescrub_row(row)
            new_line = json.dumps(new_row, ensure_ascii=False)
            if new_line != line.rstrip():
                changed += 1
            fout.write(new_line + "\n")
            n += 1
    return n, changed


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=os.path.expanduser("~/RATCHET/release/data"))
    p.add_argument("--dst", default=os.path.expanduser("~/RATCHET/release/data_rescrubbed"))
    args = p.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    print(f"Re-scrubbing {src} → {dst}")
    print(f"SCRUB_FIELDS active: {len(SCRUB_FIELDS)} field names")
    print()

    for f in sorted(src.glob("*.jsonl")):
        n, changed = process_file(f, dst / f.name)
        size_mb = (dst / f.name).stat().st_size / (1024 * 1024)
        print(f"  {f.name:<32}  {n} rows ({changed} modified)  {size_mb:.1f} MB")

    print()
    print("Done. Verify with:")
    print(f"  scripts/finalize_release.py  # generic year + CIRISLENS_LEAK_PROBES check")


if __name__ == "__main__":
    sys.exit(main())
