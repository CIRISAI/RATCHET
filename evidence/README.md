# evidence/ — CC Part VI evidence registry

Closes RATCHET#8 first pass. Companion artifact to CIRISConstitution#17 (CC 1.0-rc2 evidence-tag convention).

## What this is

`cc_formal.tsv` maps CC Part VI (the coherence mathematics) claims to their evidentiary artifacts:

- **`lean:`** — mechanized in the coherence-ratchet Lean lake (formal theorems)
- **`bench:`** — empirically supported by RATCHET or coherence-ratchet experiments
- **`open:`** — acknowledged gap, tracked to a specific coherence-ratchet issue

## Schema

TSV columns:

| # | Column | Meaning |
|---|---|---|
| 1 | `cc_decimal_id` | Decimal id from CC toc.tsv (e.g. `6.2.1`) |
| 2 | `cc_claim_id` | Stable semantic id from CC claims.tsv. **Currently `TBD-<slug>`** — will be replaced with canonical ids from CIRISConstitution#17 P1 seeding pass. Slug is stable pending the rename. |
| 3 | `lean` | Pointer of shape `coherence-ratchet:Module.theorem` (e.g. `coherence-ratchet:Core.BaseIdentity.k_eff`). `—` when no mechanized proof exists. |
| 4 | `bench` | Pointer to a coherence-ratchet or RATCHET experiment / metric (e.g. `coherence-ratchet/experiments/keff_saturation/spectral_test.py`). `—` when no empirical artifact exists. |
| 5 | `status` | One of `mechanized` (has lean), `empirical` (has bench, no lean), `open` (tracked to an issue, not yet mechanized or benched). |
| 6 | `notes` | Human-readable evidence summary — the "why this pointer resolves" gloss. |

## Pinned commits

Cross-repo pointers resolve against these commits:

- **coherence-ratchet:** `f4b72b06606536984b63fd0bf75676e044a8b3e4` (2026-07-04, includes CollapseTheorem + SignalSourceDiscount)
- **RATCHET:** current commit at publish time

Update this section whenever the seed rows are refreshed against a new coherence-ratchet HEAD.

## Coverage per the RATCHET#8 ask

| CC section | Ask | Status |
|---|---|---|
| 6.2.1 | Kish identity → mechanized | ✓ `Core.BaseIdentity.k_eff` + boundary + asymptotic ceiling + monotone theorems |
| 6.2.1 | `k_eff` ceiling → `1/ρ̄` → mechanized | ✓ `k_eff_asymptotic_ceiling` + `k_eff_bounded_above` |
| 6.2.1 | Gate 0 saturation → bench | ✓ `experiments/keff_saturation` (C. elegans β=0.10, Drosophila β=0.122, natural-substrate band ≈7±2) |
| 6.2.1 | Driver-invariance (k_eff intrinsic, r ≈ −0.009) → bench | ✓ pointed at RATCHET/experiments/exp0_cca_validation with the r ≈ −0.009 metric |
| 6.2.1 | Natural-substrate ceiling ~11 → bench | ✓ direct-measurement, max k_eff = 9.7 at k=200 across 6 substrates |
| 6.2.2 | Collapse theorem + preconditions → mechanized | ✓ `Core.Corridor.corridor_bounds_well_formed`, `corridor_keff_range_asymptotic`, `inCorridor` |
| 6.2.4 | Two-pole dynamics → mechanized | ✓ `Core.Dynamics.dρ_dt`, `rho_drift_at_zero_maintenance`, `corridor_requires_maintenance`, `rho_exit_chaos` |
| 6.2.1 | `O(r²·k_eff)` vs `O(r²·k)` → mechanized | ✓ `Core.CollapseTheorem` — corrected remainder + saturation + uniform bound + machine-checkable-difference-from-withdrawn-form. Closed coherence-ratchet#4. |
| 6.2.3.1 | σ signal-source discount → mechanized | ✓ `Core.SignalSourceDiscount` — Kish discount on σ, clique neutralization at ρ̄_src=1, monotone continuous tightening, corrected recurrence. Closed coherence-ratchet#5. |

## How to verify a pointer

**`lean:`** pointers:
```bash
cd ~/coherence-ratchet
git checkout 2cbb394
# e.g. for coherence-ratchet:Core.BaseIdentity.k_eff_asymptotic_ceiling
grep -n "theorem k_eff_asymptotic_ceiling" formal/CoherenceRatchet/Core/BaseIdentity.lean
# Then: cd formal && lake build
```

**`bench:`** pointers:
```bash
cd ~/coherence-ratchet/experiments/keff_saturation
cat README.md                      # methodology + calibration + results
python3 spectral_test.py           # re-run C. elegans discriminator (raw Kato 2015)
python3 spectral_drosophila.py     # re-run Drosophila EPG discriminator
```

**`open:`** items resolve to a GitHub issue — read the linked issue for the full derivation of what's open and the proposed resolution direction.

## Next steps (from RATCHET#8 acceptance criteria)

1. **Claim-id assignment.** Once CIRISConstitution#17 P1 lands claims.tsv, refresh column 2 with canonical `cc_claim_id` values in place of the `TBD-<slug>` placeholders.
2. **CI pointer resolution.** Wire `tools/check_claims.py` (per Constitution#17) to resolve every pointer in this file at build time. Dead pointer = build failure.
3. **Sibling-commit manifest.** Pin cross-repo dependencies via a manifest file (currently pinned inline here); `check_claims.py` reads from the manifest.
4. **Backfill.** Extend to CC Part VII (protocol) and Part VIII (operational) claims as RATCHET expands the bench-tier coverage.

## Non-goals

- Not a normative change; adds an evidence layer over existing prose.
- Not a rewrite of CC Part VI; the mathematics is unchanged, the artifacts are named.
- Not a promotion of open items — they stay `open` until coherence-ratchet#4/#5 resolve.

---
*Filed against RATCHET#8; companion to CIRISConstitution#17.*
