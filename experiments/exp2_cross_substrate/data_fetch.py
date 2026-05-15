#!/usr/bin/env python3
"""
Exp 2 substrate dataset fetcher + SHA-256 pin validator.

Reads `experiments/exp2_cross_substrate/data_sources.yaml` and for each
substrate:
  1. Fetches from primary_source.url (or via configured SDK / R package).
  2. Computes SHA-256 of the downloaded artifact.
  3. Compares against the pinned `expected_sha256` in YAML.
     - If pin is NULL: records the current hash to a per-fetch report
       (operator promotes to YAML pin in a separate PR).
     - If pin is set + matches: PASS.
     - If pin is set + mismatch: FAIL with diff.
  4. Emits a fetch_manifest.json with per-substrate status.

Usage:
  python3 data_fetch.py [--substrate <name>] [--vendor-dir <dir>]

Designed to run in CI (`.github/workflows/substrate_revalidation.yml`)
as well as locally. All fetches are idempotent — re-running on the same
remote state produces identical artifacts and hashes.

NOTE — this is a SKELETON. Each substrate's fetcher needs concrete
implementation per its source protocol (FTP, S3, R package, etc.).
The harness shape is locked; the per-substrate logic is the work item.
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_SOURCES_YAML = Path(__file__).parent / "data_sources.yaml"
DEFAULT_VENDOR_DIR = Path(__file__).parent / "vendored"


def sha256_path(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_alphafold(spec: dict, vendor_dir: Path) -> dict:
    """Fetch CATH-S40 representative single-domain proteins from AlphaFold DB.

    Implementation pending: requires CATH-S40 cross-reference list + per-ID
    PDB/CIF download from AlphaFold DB. Output is a single tarball of
    structures, hashed at the tarball level.
    """
    return {"status": "NOT_IMPLEMENTED",
            "note": "AlphaFold fetch via EBI FTP — needs CATH-S40 ID list + tar bundle"}


def fetch_pmu_pnnl(spec: dict, vendor_dir: Path) -> dict:
    """Fetch PNNL Open PMU Library transmission-event corpus.

    Implementation pending: PNNL-30492 corpus access pattern needs
    confirmation (may require institutional credentials).
    """
    return {"status": "NOT_IMPLEMENTED",
            "note": "PNNL-30492 — confirm public access path"}


def fetch_allen_neuropixels(spec: dict, vendor_dir: Path) -> dict:
    """Fetch Allen Brain Observatory Neuropixels via AllenSDK.

    Implementation pending: requires `pip install allensdk` and AWS S3
    sync of the visual-coding ecephys cache. Per-session NWB files,
    hashed at the cache-manifest level.
    """
    return {"status": "NOT_IMPLEMENTED",
            "note": "AllenSDK ecephys cache sync — needs allensdk dependency"}


def fetch_biotime(spec: dict, vendor_dir: Path) -> dict:
    """Fetch BioTIME 2.0 from St Andrews.

    Implementation pending: download the BioTIME 2.0 SQLite dump or CSV
    bundle from biotime.st-andrews.ac.uk. Single-file artifact hashable
    directly.
    """
    return {"status": "NOT_IMPLEMENTED",
            "note": "BioTIME 2.0 SQLite/CSV download from St Andrews"}


def fetch_nasa_battery(spec: dict, vendor_dir: Path) -> dict:
    """Fetch NASA PCoE Li-ion battery aging dataset.

    Static dataset at phm-datasets.s3.amazonaws.com. Just downloads + hashes.
    """
    return {"status": "NOT_IMPLEMENTED",
            "note": "NASA battery ZIP from phm-datasets.s3.amazonaws.com (one-shot)"}


def fetch_vdem(spec: dict, vendor_dir: Path) -> dict:
    """Fetch V-Dem v16 dataset.

    Implementation pending: download from v-dem.net/data/ (R package
    vdemdata as the cleaner path).
    """
    return {"status": "NOT_IMPLEMENTED",
            "note": "V-Dem v16 — use vdemdata R package or direct ZIP"}


def fetch_agp_microbiome(spec: dict, vendor_dir: Path) -> dict:
    """Fetch AGP / Microsetta public archive.

    Implementation pending: Qiita study 10317 export.
    """
    return {"status": "NOT_IMPLEMENTED",
            "note": "AGP via Qiita study 10317 public export"}


# Dispatch table
FETCHERS = {
    "alphafold": fetch_alphafold,
    "pmu_pnnl": fetch_pmu_pnnl,
    "allen_neuropixels": fetch_allen_neuropixels,
    "biotime": fetch_biotime,
    "nasa_battery": fetch_nasa_battery,
    "vdem": fetch_vdem,
    "agp_microbiome": fetch_agp_microbiome,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--substrate", help="Fetch only this substrate (default: all)")
    ap.add_argument("--vendor-dir", default=str(DEFAULT_VENDOR_DIR))
    args = ap.parse_args()

    vendor_dir = Path(args.vendor_dir)
    vendor_dir.mkdir(parents=True, exist_ok=True)

    with open(DATA_SOURCES_YAML) as f:
        sources = yaml.safe_load(f)

    report: dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "data_sources_yaml_sha256": sha256_path(DATA_SOURCES_YAML),
        "substrates": {},
    }

    targets = (
        [args.substrate]
        if args.substrate
        else list(sources["substrates"].keys())
    )

    for sub in targets:
        spec = sources["substrates"].get(sub)
        if not spec:
            report["substrates"][sub] = {"status": "UNKNOWN_SUBSTRATE"}
            continue
        fetcher = FETCHERS.get(sub)
        if not fetcher:
            report["substrates"][sub] = {"status": "NO_FETCHER"}
            continue
        try:
            result = fetcher(spec, vendor_dir)
        except Exception as e:
            result = {"status": "ERROR", "error": str(e)}
        # Compare against pinned SHA if both present
        pinned = spec.get("expected_sha256")
        observed = result.get("sha256")
        if pinned and observed:
            result["pin_match"] = (pinned == observed)
            if not result["pin_match"]:
                result["status"] = "SHA_MISMATCH"
                result["pin_expected"] = pinned
                result["pin_observed"] = observed
        result["rung"] = spec.get("rung")
        result["engine"] = spec.get("engine")
        report["substrates"][sub] = result

    report["finished_at"] = datetime.now(timezone.utc).isoformat()

    out_path = vendor_dir / "fetch_manifest.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"fetch_manifest.json written to {out_path}")

    # Exit non-zero if any SHA_MISMATCH (real failure) — NOT_IMPLEMENTED is OK
    failures = [s for s, r in report["substrates"].items() if r.get("status") == "SHA_MISMATCH"]
    if failures:
        print(f"::error::SHA pin mismatch for: {failures}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
