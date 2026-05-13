#!/usr/bin/env python3
"""
Build the v0.1.0 calibration bundle for CIRISLensCore manifold-conformity scoring.

Reads the 2.7.9 lens export at /tmp/ratchet_v0_1_0_calibration/, computes
imputation + standardization + retention + per-cohort centroids over the
locked crc-v1 16-field projection, emits a signed bundle.

Output:
  release/calibration/crc-v1/bundle.yaml          human-readable bundle
  release/calibration/crc-v1/bundle.cbor          canonical CBOR (what signs)
  release/calibration/crc-v1/bundle.sha256        sha256 of canonical CBOR
  release/calibration/crc-v1/bundle.signing.txt   signing target + algorithm slot
"""

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cbor2
import numpy as np
import yaml

PROJECTION_VERSION = "crc-v1"
RATCHET_CALIBRATION_VERSION = 1
SAMPLE_SIZE_GATE = 500
MANIFOLD_THRESHOLD_GLOBAL = 2.5  # Mahalanobis sigma units, provisional

PROJECTION_16 = [
    "csdma_plausibility_score",
    "dsdma_domain_alignment",
    "coherence_level",
    "entropy_level",
    "idma_k_eff",
    "idma_correlation_risk",
    "entropy_score",
    "coherence_score",
    "optimization_veto_entropy_ratio",
    "epistemic_humility_certainty",
    "conscience_passed",
    "entropy_passed",
    "coherence_passed",
    "optimization_veto_passed",
    "epistemic_humility_passed",
    "action_was_overridden",
]

EXTRACTION_PATHS = {
    "DMA_RESULTS": {
        "csdma_plausibility_score": ("csdma", "plausibility_score"),
        "dsdma_domain_alignment": ("dsdma", "domain_alignment"),
    },
    "IDMA_RESULT": {
        "idma_k_eff": ("k_eff",),
        "idma_correlation_risk": ("correlation_risk",),
    },
    "CONSCIENCE_RESULT": {
        "coherence_level": ("coherence_level",),
        "entropy_level": ("entropy_level",),
        "entropy_score": ("entropy_score",),
        "coherence_score": ("coherence_score",),
        "optimization_veto_entropy_ratio": ("optimization_veto_entropy_ratio",),
        "epistemic_humility_certainty": ("epistemic_humility_certainty",),
        "conscience_passed": ("conscience_passed",),
        "entropy_passed": ("entropy_passed",),
        "coherence_passed": ("coherence_passed",),
        "optimization_veto_passed": ("optimization_veto_passed",),
        "epistemic_humility_passed": ("epistemic_humility_passed",),
        "action_was_overridden": ("action_was_overridden",),
    },
}


def get_nested(d, path):
    cur = d
    for p in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(p)
    return cur


def cast_to_float(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float)):
        return float(v)
    return None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_thoughts(jsonl_path: Path):
    thought_features = defaultdict(dict)
    thought_cohort = {}
    last_conscience_ts = {}
    with open(jsonl_path) as f:
        for line in f:
            row = json.loads(line)
            et = row.get("event_type")
            key = (row.get("trace_id"), row.get("thought_id"))
            thought_cohort[key] = {
                "agent_role": row.get("agent_role"),
                "agent_template": row.get("agent_template"),
                "deployment_domain": row.get("deployment_domain"),
                "deployment_type": row.get("deployment_type"),
                "deployment_region": row.get("deployment_region"),
                "deployment_trust_mode": row.get("deployment_trust_mode"),
            }
            payload = row.get("payload") or {}
            if et == "CONSCIENCE_RESULT":
                ts = row.get("ts")
                if key not in last_conscience_ts or ts > last_conscience_ts[key]:
                    last_conscience_ts[key] = ts
                    for fname, path in EXTRACTION_PATHS[et].items():
                        v = cast_to_float(get_nested(payload, path))
                        if v is not None:
                            thought_features[key][fname] = v
            elif et in EXTRACTION_PATHS:
                for fname, path in EXTRACTION_PATHS[et].items():
                    v = cast_to_float(get_nested(payload, path))
                    if v is not None:
                        thought_features[key][fname] = v
    return thought_features, thought_cohort


def cohort_key_tuple(c):
    return (
        c["agent_role"],
        c["agent_template"],
        c["deployment_domain"],
        c["deployment_type"],
        c["deployment_region"],
        c["deployment_trust_mode"],
    )


def build_bundle(export_dir: Path):
    trace_events = export_dir / "trace_events.jsonl"
    manifest_path = export_dir / "MANIFEST.json"
    with open(manifest_path) as f:
        export_manifest = json.load(f)
    corpus_sha256 = sha256_file(trace_events)

    thought_features, thought_cohort = extract_thoughts(trace_events)
    keys = sorted(thought_features.keys())
    n = len(keys)

    X_raw = np.full((n, 16), np.nan)
    for i, k in enumerate(keys):
        for j, fname in enumerate(PROJECTION_16):
            v = thought_features[k].get(fname)
            if v is not None:
                X_raw[i, j] = v

    col_means = np.nanmean(X_raw, axis=0)
    X_imp = np.where(np.isnan(X_raw), col_means, X_raw)
    col_stds = X_imp.std(axis=0, ddof=0)
    retention = col_stds > 1e-9

    safe_stds = np.where(retention, col_stds, 1.0)
    X_std = (X_imp - X_imp.mean(axis=0)) / safe_stds
    X_std_retained = X_std[:, retention]

    cohort_buckets = defaultdict(list)
    cohort_meta = {}
    for i, k in enumerate(keys):
        ck = cohort_key_tuple(thought_cohort[k])
        cohort_buckets[ck].append(X_std_retained[i])
        cohort_meta[ck] = thought_cohort[k]

    cohort_centroids = []
    for ck, vecs in sorted(cohort_buckets.items(), key=lambda x: -len(x[1])):
        arr = np.array(vecs)
        cohort_centroids.append({
            "cohort": cohort_meta[ck],
            "centroid": [float(x) for x in arr.mean(axis=0)],
            "variance": [float(x) for x in arr.var(axis=0)],
            "sample_count": int(len(vecs)),
            "above_sample_size_gate": bool(len(vecs) >= SAMPLE_SIZE_GATE),
        })

    bundle = {
        "ratchet_calibration_version": RATCHET_CALIBRATION_VERSION,
        "projection_version": PROJECTION_VERSION,
        "calibrated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "calibration_corpus": {
            "source_package": "ratchet_v0_1_0_calibration",
            "trace_events_sha256": corpus_sha256,
            "n_thoughts": int(n),
            "schema_version": export_manifest["filters"]["schema_version"],
            "exported_at": export_manifest["exported_at"],
            "known_issues": export_manifest.get("known_issues_in_dump", []),
        },
        "sample_size_gate": SAMPLE_SIZE_GATE,
        "manifold_threshold_global": MANIFOLD_THRESHOLD_GLOBAL,
        "projection": {
            "field_order": PROJECTION_16,
            "imputation": {f: float(col_means[i]) for i, f in enumerate(PROJECTION_16)},
            "standardization": {
                "means": [float(X_imp.mean(axis=0)[i]) for i in range(16)],
                "stds": [float(safe_stds[i]) for i in range(16)],
            },
            "retention_mask": [bool(retention[i]) for i in range(16)],
            "retained_dim_count": int(retention.sum()),
        },
        "cohort_centroids": cohort_centroids,
        "notes": [
            "v0.1.0 baseline bundle. All cohort cells below sample_size_gate=500.",
            "Lens-core scoring against this bundle returns LC-AV-18 indeterminate "
            "for every cohort until v0.2 re-calibration on a larger corpus.",
            "idma_k_eff and epistemic_humility_passed dropped by retention mask "
            "(std<1e-9 in calibration corpus — degenerate in 2.7.9-stable traffic).",
            "manifold_threshold_global=2.5σ is provisional; needs empirical ROC fit.",
        ],
    }
    return bundle


def canonical_cbor(bundle: dict) -> bytes:
    return cbor2.dumps(bundle, canonical=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-dir", default="/tmp/ratchet_v0_1_0_calibration")
    ap.add_argument("--out-dir", default="/home/emoore/RATCHET/release/calibration/crc-v1")
    args = ap.parse_args()

    export_dir = Path(args.export_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = build_bundle(export_dir)

    yaml_path = out_dir / "bundle.yaml"
    cbor_path = out_dir / "bundle.cbor"
    sha_path = out_dir / "bundle.sha256"
    sign_path = out_dir / "bundle.signing.txt"

    with open(yaml_path, "w") as f:
        yaml.safe_dump(bundle, f, sort_keys=False, default_flow_style=False, width=120)
    cbor_bytes = canonical_cbor(bundle)
    with open(cbor_path, "wb") as f:
        f.write(cbor_bytes)
    bundle_sha = hashlib.sha256(cbor_bytes).hexdigest()
    with open(sha_path, "w") as f:
        f.write(f"{bundle_sha}  bundle.cbor\n")

    signing_target = (
        f"# CIRISLensCore v0.1.0 calibration bundle — signing target\n"
        f"# Algorithm: Ed25519 + ML-DSA-65 (hybrid, federation primitive)\n"
        f"# Canonical bytes: bundle.cbor (canonical CBOR per RFC 8949 §4.2)\n"
        f"# Sign with ciris-keyring + ciris-crypto v1.9.0\n"
        f"#\n"
        f"# Sha256 of canonical bytes:\n"
        f"{bundle_sha}\n"
    )
    with open(sign_path, "w") as f:
        f.write(signing_target)

    print(f"Bundle built:")
    print(f"  YAML:     {yaml_path}")
    print(f"  CBOR:     {cbor_path}  ({len(cbor_bytes)} bytes)")
    print(f"  Sha256:   {sha_path}")
    print(f"  Sign tgt: {sign_path}")
    print(f"  Bundle sha256: {bundle_sha}")
    print(f"  Calibration corpus n: {bundle['calibration_corpus']['n_thoughts']}")
    print(f"  Cohort cells: {len(bundle['cohort_centroids'])}")
    print(f"  Retained dims: {bundle['projection']['retained_dim_count']}/16")


if __name__ == "__main__":
    main()
