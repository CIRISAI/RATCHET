# Federation Threat Models

Snapshots of the CIRIS federation's threat-model documents, vendored into RATCHET so the evaluator has a self-contained copy of the threat surface it is designed to detect.

These are **copies**. The authoritative versions live in their respective repositories. This directory is refreshed when source documents change materially; provenance for the current snapshot is below.

## Why this directory exists

RATCHET (`../`) is the federation's anti-Sybil evaluator. It consumes the federation threat model (`FEDERATION_THREAT_MODEL.md`) as its interface contract — the F-AV catalog tells RATCHET what threats it is expected to detect, the substrate-assumption surface tells it what it can rely on, and the per-F-AV dimension mapping tells it which N_eff signals to compute. The per-repo substrate threat models cover Class 1 substrate threats that RATCHET assumes hold (cryptography, hardware identity, build provenance, signed evidence durability, audit log integrity, federation transport, etc.); RATCHET does not detect substrate-class threats, but understanding them is necessary for understanding what RATCHET *can* detect.

Vendoring snapshots here serves three concrete purposes:

1. **Self-contained reading**. RATCHET is public AGPL-3.0 on GitHub. A reviewer or integrator picking up RATCHET should be able to understand the full federation threat surface without cloning 8 other repositories. Vendoring removes that friction.

2. **Snapshot stability**. RATCHET's interface contract is pinned to a *known state* of the federation threat model. When the source documents change in ways that affect what RATCHET must detect, that change is a deliberate refresh of this directory — not a silent drift through cross-repo updates.

3. **Public reviewability**. Anyone reading RATCHET in the open — researchers, auditors, federation participants, adversarial reviewers — sees exactly the threat surface RATCHET was designed against, at exactly the version RATCHET was designed against.

## Contents

```
FEDERATION_THREAT_MODELS/
├── README.md                          this file
├── FEDERATION_THREAT_MODEL.md         meta — composition + Sybil + RATCHET interface
├── CIRISVerify_THREAT_MODEL.md        substrate: hardware identity, hybrid signing, build attestation
├── CIRISPersist_THREAT_MODEL.md       substrate: signed-evidence persistence, audit anchor, federation directory
├── CIRISRegistry_THREAT_MODEL.md      substrate: federation directory cache + policy layer
├── CIRISLens_THREAT_MODEL.md          substrate: ethical-reasoning analysis layer (pre-§3.1 collapse)
├── CIRISBilling_THREAT_MODEL.md       substrate: billing, bond purchase flow
├── CIRISProxy_THREAT_MODEL.md         substrate: federation proxy / routing
├── CIRISOssicle_THREAT_MODEL.md       substrate: ossicle (signed manifest) format
├── CIRISEdge_THREAT_MODEL.md          substrate: federation transport (Reticulum + multi-medium); fills N1 + N2 primitives
└── CIRISLensCore_THREAT_MODEL.md      substrate: science-layer detection runtime (manifold-conformity + 5 ratchet detectors); folds into agent post-PoB §3.1
```

## Reading order

For a RATCHET reader picking this up cold:

1. **Start with `FEDERATION_THREAT_MODEL.md`** — it derives the 12 substrate primitives from first causes (§2), projects them onto CIRIS architecture (§3), enumerates 31 F-AVs (§6), and specifies RATCHET's assumption surface (§7) + interface contract (§10). 2095 lines; the meta document.

2. **Then read the per-repo substrate threat models** as deep-dives for any specific primitive RATCHET cares about. They are referenced from `FEDERATION_THREAT_MODEL.md` §5 with verified section anchors.

3. **The per-repo F-AV taxonomies are independent**: CIRISPersist uses AV-1..AV-26 numbering for its substrate threats; CIRISRegistry uses AV-1..AV-35; CIRISVerify uses AV-1..AV-14. The federation threat model uses F-AV-1..F-AV-17 plus mnemonic-suffix new ones (F-AV-TIMESHIFT, F-AV-BRIBE, etc.). **F-AV identifiers in `FEDERATION_THREAT_MODEL.md` are distinct from per-repo AV identifiers** even when the numbers overlap.

## Provenance

Snapshots taken **2026-05-02 (UTC-5)** from the following commits, with `CIRISEdge_THREAT_MODEL.md` added 2026-05-03 when the Edge substrate published its baseline threat model:

| File | Source repo | Source path | Commit | Date | Lines |
|------|-------------|-------------|--------|------|-------|
| `FEDERATION_THREAT_MODEL.md` | CIRISVerify | `docs/FEDERATION_THREAT_MODEL.md` | `c09dd24` | 2026-05-03 | 2,095 |
| `CIRISVerify_THREAT_MODEL.md` | CIRISVerify | `docs/THREAT_MODEL.md` | `de28673` | 2026-05-02 | 540 |
| `CIRISPersist_THREAT_MODEL.md` | CIRISPersist | `docs/THREAT_MODEL.md` | `6e9b243` | 2026-05-01 | 980 |
| `CIRISRegistry_THREAT_MODEL.md` | CIRISRegistry | `docs/THREAT_MODEL.md` | `242ec88` | 2026-05-01 | 1,896 |
| `CIRISLens_THREAT_MODEL.md` | CIRISLens | `docs/THREAT_MODEL.md` | `3e0cddd` | 2026-05-01 | 945 |
| `CIRISBilling_THREAT_MODEL.md` | CIRISBilling | `docs/THREAT_MODEL.md` | `974ed06` | 2026-05-01 | 620 |
| `CIRISProxy_THREAT_MODEL.md` | CIRISProxy | `docs/THREAT_MODEL.md` | `44ae015` | 2026-05-01 | 532 |
| `CIRISOssicle_THREAT_MODEL.md` | CIRISOssicle | `THREAT_MODEL.md` | `167291c` | 2026-01-10 | 190 |
| `CIRISEdge_THREAT_MODEL.md` | CIRISEdge | `docs/THREAT_MODEL.md` | `2c3c167` | 2026-05-03 | 893 |
| `CIRISLensCore_THREAT_MODEL.md` | CIRISLens | `docs/THREAT_MODEL_CORE.md` | `5717da3` | 2026-05-03 | 939 |

**Total: 9,630 lines across 10 files.**

The `FEDERATION_THREAT_MODEL.md` document is **v1.0 (first publication)**. It survived 8 reviewer-passes (4 specialist roles × 2 internal versions) before the published v1.0 was produced. See its Appendix A for the v1 → v2 → v1.0 (published) lineage.

## Refresh policy

This directory is **manually synchronized**, not automated. The federation deliberately chose vendored snapshots over a build-time fetch because:

- Threat model changes are rare (quarterly minor, annual major per `FEDERATION_THREAT_MODEL.md` §12).
- Each refresh should be reviewed for material impact on RATCHET's interface contract; automatic sync would obscure the review.
- A vendored snapshot makes the threat-model version RATCHET was designed against legible to public readers without cross-repo navigation.

**Refresh trigger**: any of the source files moves materially — substantively new F-AVs, primitive changes, fail-secure protocol revisions, or version bumps in the federation threat model.

**Refresh procedure** (operator-facing):

```bash
# From RATCHET repo root
cd FEDERATION_THREAT_MODELS

# Refresh sources (replace commit hashes / dates in this README too)
cp ../../CIRISVerify/docs/FEDERATION_THREAT_MODEL.md  FEDERATION_THREAT_MODEL.md
cp ../../CIRISVerify/docs/THREAT_MODEL.md             CIRISVerify_THREAT_MODEL.md
cp ../../CIRISPersist/docs/THREAT_MODEL.md            CIRISPersist_THREAT_MODEL.md
cp ../../CIRISRegistry/docs/THREAT_MODEL.md           CIRISRegistry_THREAT_MODEL.md
cp ../../CIRISLens/docs/THREAT_MODEL.md               CIRISLens_THREAT_MODEL.md
cp ../../CIRISBilling/docs/THREAT_MODEL.md            CIRISBilling_THREAT_MODEL.md
cp ../../CIRISProxy/docs/THREAT_MODEL.md              CIRISProxy_THREAT_MODEL.md
cp ../../CIRISOssicle/THREAT_MODEL.md                 CIRISOssicle_THREAT_MODEL.md
cp ../../CIRISEdge/docs/THREAT_MODEL.md               CIRISEdge_THREAT_MODEL.md
cp ../../CIRISLens/docs/THREAT_MODEL_CORE.md          CIRISLensCore_THREAT_MODEL.md

# Then update the provenance table above with new commit hashes / dates / line counts
# Then commit + push from RATCHET
```

## License

Each file retains the license of its source repository. All current sources are **AGPL-3.0-or-later**.

## Cross-references

- RATCHET's own threat-model commitments (limitations L-01..L-08, formal-mathematical guarantees, empirical results) live in `../KNOWN_LIMITATIONS.md`, `../VALIDATED_FINDINGS.md`, `../FSD.md`, and `../ratchet-paper/Constrained_Reasoning_Chains.pdf`. The federation threat model's §1.2 Bet 2 cites all of these.

- The federation's anti-Sybil claim is **complexity-theoretic** (truth = O(1), CONSISTENT-LIE NP-hard under ETH; Z3-verified reduction from 3-SAT) and **empirically supported** (n=6,465 production traces in the *Constrained Reasoning Chains* paper, plus 100% deceptive-prior detection on Qwen 3.6 across 29 languages and 5 patch levels as of 2026-05). Both are explicitly bounded by RATCHET L-01..L-08 — see `FEDERATION_THREAT_MODEL.md` §1.2.
