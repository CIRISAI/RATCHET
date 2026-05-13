# CIRISLensCore v0.1.0 calibration bundle — `crc-v1` projection

This directory holds the RATCHET-produced calibration artifact for the CRC v1 (16-feature) manifold-conformity projection, consumed by `CIRISLensCore` hot-path scoring per [CIRISLensCore#3](https://github.com/CIRISAI/CIRISLensCore/issues/3) and stored via [CIRISPersist#18](https://github.com/CIRISAI/CIRISPersist/issues/18) (`calibration_bundles` table, v0.4.3).

## Files

| File | Purpose |
|---|---|
| `bundle.yaml` | Human-readable form of the bundle. Diff-friendly. |
| `bundle.cbor` | Canonical CBOR (RFC 8949 §4.2). **This is what gets signed.** |
| `bundle.sha256` | SHA-256 of `bundle.cbor`. |
| `bundle.signing.txt` | Signing target + algorithm slot. Hybrid Ed25519 + ML-DSA-65 via `ciris-keyring` + `ciris-crypto v1.9.0`. |

## Headline numbers (v0.1.0, n=264 thoughts)

| Metric | This bundle | CRC paper anchor | Notes |
|---|---|---|---|
| N_eff (Entropy H) | 7.07 | ~7.1 | Matches within rounding |
| N_eff (PR) | 5.52 | 6.61 | Lower — 2 fields dropped in this corpus |
| 90% variance horizon | 7 dims | 7 dims | **Exact match** — manifold geometry reproducible |
| 99% variance horizon | 10 dims | 11 dims | Off-by-one |
| Retained dims | 14 / 16 | 16 / 16 | `idma_k_eff` + `epistemic_humility_passed` degenerate (std<1e-9) in this corpus |
| Sample-size-gate | 500 thoughts/cell | — | All cells below gate → LC-AV-18 indeterminate at score time |

## What this bundle ships

- The 16-field canonical projection order (locked, `projection_version: crc-v1`).
- Per-field corpus imputation values (mean), standardization parameters (mean+std), retention mask.
- Per-cohort centroids in standardized + retained space, with `sample_count` stamps so lens-core can apply the LC-AV-18 indeterminate gate.
- A provisional global manifold threshold of 2.5σ (Mahalanobis), pending an empirical ROC fit at v0.2.

## What this bundle does NOT yet do

- Real-time scoring won't return `aligned` / `drift` verdicts — every cohort cell is below sample_size_gate (90 / 119 / 55 thoughts). Lens-core scoring returns `indeterminate` for every input.
- Cohort 6-tuple effectively collapsed to a 3-tuple in this corpus (`agent_template`, `deployment_region`, `deployment_trust_mode` all null — operator-config gap).

## Signing

The bundle `.cbor` is the **canonical bytes**. Signing operator:

```bash
# Ed25519 (mandatory)
ciris-keyring sign-ed25519 --key-id <RATCHET signing key> \
    --in release/calibration/crc-v1/bundle.cbor \
    --out release/calibration/crc-v1/bundle.ed25519.sig

# ML-DSA-65 (mandatory — hybrid federation primitive, OQ-11)
ciris-keyring sign-mldsa --key-id <RATCHET PQC signing key> \
    --in release/calibration/crc-v1/bundle.cbor \
    --out release/calibration/crc-v1/bundle.ml_dsa_65.sig
```

Then push to `cirislens_derived.calibration_bundles` via `Engine.put_calibration_bundle()` (CIRISPersist v0.4.3); persist runs `verify_hybrid_via_directory(...)` under `HybridPolicy::Strict` on the put path. Both signatures must verify or the row is rejected.

## Reproducing this bundle

```bash
# 1. Fetch the calibration export
#    (from lens; the dump used here is the 2026-05-13 package)

# 2. Run the build script (RATCHET repo)
python3 scripts/build_calibration_bundle.py \
    --export-dir /path/to/ratchet_v0_1_0_calibration \
    --out-dir release/calibration/crc-v1
```

The build is deterministic given a fixed input. SHA-256 of canonical CBOR pins reproducibility.

## v0.2 plan

Re-calibrate once 2.7.9 traffic exceeds sample_size_gate=500 thoughts per cohort cell (≥3 cells with sufficient population). Expected unblocks:

- Real-time scoring returns `aligned` / `drift` verdicts (not just `indeterminate`).
- Empirical ROC fit replaces the provisional 2.5σ threshold.
- Two degenerate dims (`idma_k_eff`, `epistemic_humility_passed`) likely recover variance if upstream IDMA tuning lands.

## Related issues

- [CIRISLensCore#3](https://github.com/CIRISAI/CIRISLensCore/issues/3) — projection contract + bundle ask
- [CIRISPersist#18](https://github.com/CIRISAI/CIRISPersist/issues/18) — `calibration_bundles` schema + Engine API (closed by v0.4.3)
- [CIRISLens#4](https://github.com/CIRISAI/CIRISLens/issues/4) — post-2.7.9 QA-traffic export pipeline that produced the calibration dump
- [CIRISLens#12](https://github.com/CIRISAI/CIRISLens/issues/12) — `channel_id` pii-scrubber over-fire affecting task-class derivation
- [CIRISAgent#724](https://github.com/CIRISAI/CIRISAgent/issues/724) — FSD §5.4 `correlation_risk` example correction (closed)
