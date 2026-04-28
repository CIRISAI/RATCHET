#!/usr/bin/env python3
"""
Finalize a release after re-scrub: swap data_rescrubbed/ → data/,
regenerate MANIFEST.json, run a final leak-count check.

Idempotent: safe to re-run.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(os.path.expanduser("~/RATCHET/release"))
SRC = ROOT / "data_rescrubbed"
DST = ROOT / "data"
BAK = ROOT / "data_pre_rescrub"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def count_lines(p: Path) -> int:
    return sum(1 for _ in p.open())


# Sanity checks
if not SRC.exists():
    sys.exit(f"Missing {SRC} — run the rescrub first")
if not DST.exists():
    sys.exit(f"Missing {DST} — unusual; check release layout")

# Backup the current data/ as data_pre_rescrub/ (idempotent — re-runs overwrite)
print(f"Backing up old data/ → {BAK}")
if BAK.exists():
    import shutil
    shutil.rmtree(BAK)
DST.rename(BAK)

# Move rescrubbed into place
SRC.rename(DST)
print(f"Promoted {SRC.name} → {DST.name}")

# Regenerate MANIFEST.json
print("\nRegenerating MANIFEST.json...")
manifest = {
    "release_built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "filter_rules": [
        "signature_verified=true required",
        "timestamp >= 2026-03-22",
        "wbd_deferral retry-loop fixture excluded",
    ],
    "scrubber_rules": [
        "NER REDACT: PERSON, ORG, GPE, FAC, LOC, NORP, DATE, TIME, EVENT, MISC, WORK_OF_ART, LAW",
        "Multilingual NER fallback (xx_ent_wiki_sm) for non-Latin text",
        "Regex: EMAIL, PHONE, IP, URL, SSN, CREDIT_CARD, YEAR (1700-2023), year-bearing IDENTIFIER",
        "Recursive subtree scrubbing on SCRUB_FIELDS-keyed values",
    ],
    "files": {},
}
for f in sorted(DST.glob("*.jsonl")):
    manifest["files"][f.name] = {
        "sha256": sha256(f),
        "size_bytes": f.stat().st_size,
        "row_count": count_lines(f),
    }

manifest_path = ROOT / "MANIFEST.json"
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
print(f"  {manifest_path}")
for fname, info in manifest["files"].items():
    print(f"  {fname:<32}  {info['row_count']:>5} rows  {info['size_bytes']/1024/1024:>5.1f} MB  {info['sha256'][:12]}")

# Final leak-check.
# Probes are loaded from CIRISLENS_LEAK_PROBES (newline-separated) when set,
# so the term list isn't checked into the source. Sample default below is a
# trivially-derivable smoke-test (a four-digit year not covered by the year
# regex would slip through; this is just a loud canary).
import os
probes_env = os.environ.get("CIRISLENS_LEAK_PROBES", "")
if probes_env:
    probes = [p for p in probes_env.split("\n") if p.strip()]
else:
    # Generic smoke test: principled checks only.
    probes = [
        # Any historical year that escaped the regex (signals scrubber failure)
        # — we use the YEAR placeholder as expected output, so finding raw years
        # that should have been redacted indicates a coverage gap.
    ]

print("\nFinal leak-count smoke test across data/:")
import re
year_re = re.compile(r"\b(?:1[7-9]\d{2}|20[0-1]\d|202[0-3])\b")
year_hits = 0
probe_hits = {p: 0 for p in probes}
for f in DST.glob("*.jsonl"):
    with f.open() as fh:
        for line in fh:
            year_hits += len(year_re.findall(line))
            for p in probes:
                if p.lower() in line.lower():
                    probe_hits[p] += 1

flag = "✓" if year_hits == 0 else f"⚠ {year_hits} historical-year escapes"
print(f"  {flag:<10}  bare-year regex (1700-2023)")
for p, n in probe_hits.items():
    flag = "✓" if n == 0 else f"⚠ {n}"
    print(f"  {flag:<10}  configured probe (CIRISLENS_LEAK_PROBES)")

print("\nDone.")
