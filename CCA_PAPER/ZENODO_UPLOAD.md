# Zenodo deposit — CCA corrections version

**Target record:** [Zenodo 18217688](https://zenodo.org/records/18217688) — *Coherence
Collapse Analysis* v3, deposited 2026-01-11
**Action:** publish a **new version** of the existing record. **Do not delete the record
and do not create a new concept DOI.**
**Prepared:** 2026-07-31 · **Tracking:** RATCHET#11

---

## Do this before anything else: the concept DOI currently serves the wrong paper

**Record 21326851, labelled "v4" and dated 2026-07-12, is an accidental deposit.** The
*Corridor Dynamics* PDF was uploaded as a new version of the CCA record, so Zenodo carried the
CCA metadata forward — title *"Coherence Collapse Analysis: A Universal Failure Mode…"*, the
k_eff description, the 40%/60% figures — and attached the wrong file underneath it.

Consequence, and it is live right now: **the CCA concept DOI resolves to v4, so anyone
following a CCA citation is served the Corridor Dynamics paper under a CCA title.** That is a
more urgent problem than this deposit. It also mislabels Corridor Dynamics, which has its own
concept DOI (10.5281/zenodo.20300773) and should not appear under this one.

This affects the deposit here in two ways:

1. **Version label is `v5`, not `v4`.** The v4 slot is consumed even though its content is
   wrong. Depositing as v4 is not possible; depositing as v5 leaves the bad v4 in the version
   history, visible but superseded.
2. **`v3` (record 18217688) remains the last version carrying CCA content**, so the corrections
   note's "corrects v3" framing and every in-text "Version 3 reported…" remark are correct as
   written and need no change.

Whether v4 is deleted, replaced with the correct file, or left in place with a note is a
judgement call for the depositor. Leaving it and superseding with v5 is the least destructive
and keeps the record honest about what happened; deleting it removes a DOI that may already
have been resolved by someone.

---

## Status: otherwise clear to upload

The package is complete and internally consistent.

**1. ~~Exp 103~~ — RESOLVED 2026-07-31 by replication (RATCHET#13).** The experiment was
re-run on the original hardware under a hash-pinned pre-registration. The lockstep result is a
code artifact (one batch timing assigned to all 64 sensors, so `corrcoef` runs over identical
rows); the barrier effect is real and survives common-mode removal but spans ρ ∈ [0.28, 0.81]
across five identical trials, so it carries no point estimate. "Software alone can induce
collapse" is withdrawn. Applied to the paper at §10.3.

A finding worth carrying beyond this paper: the first measurement after GPU idle reads
ρ ≈ 0.78 regardless of condition, and both Exp 103 and F2 measure their baseline first — so
every Δρ in this series computed against a first-measured baseline is contaminated. The
measurement protocol now required (shuffle null, repeated trials with an interval, discard the
first run) is recorded in `CLAUDE.md`.

**2. ~~The institutional sections~~ — RESOLVED 2026-07-31.** All three remaining sites are
now corrected in-text. Table 8's institutional rows are withdrawn (and the note explains why
reconstruction is foreclosed, not merely unavailable). §11 is retained as a negative result,
retitled and relocated outside the validation chapter as **§11, *A Negative Result: The
Institutional Case Study***. The 7.6-year timing claim is withdrawn with its null comparison
stated. The p.3 evidence summary row is corrected to match.

**The paper is now internally consistent with its own corrections note.** Every defect the
note records is either fixed in the text or explicitly marked withdrawn/disputed at the point
where v3 asserted it. Only Action 11 (Exp 103, RATCHET#13) still blocks.

---

## Files to upload

| File | Role | Status |
|---|---|---|
| `CORRECTIONS_v3.pdf` | **Primary new file.** 15pp, the eighteen defects with evidence and dispositions | Built |
| `CORRECTIONS_v3.md` | Markdown source (canonical; the PDF is rendered from it) | Current |
| `corrections_witness.py` | sympy machine witness; reproduces every algebraic defect, exit 0 | Passing |
| `coherence_collapse_analysis.pdf` | The paper, 34pp, with all twelve applicable corrections in-text | Rebuilt, clean |
| `coherence_collapse_analysis.tex` | LaTeX source | Current |

**Corrections applied in the paper itself** (each carries an in-text remark naming the v3
error rather than fixing it silently, so a version-to-version reader can see what moved):
Thm 2.3 → `α/k ≥ d` with the derivation repaired (C-1); Thm 4.1 inherited the same error
(C-1); Corollary 2.4 restated over `k`, with the note that it is *false* under the v3
criterion and rescued by the correction (C-7); `J = k_eff·λ·σ` throughout (C-6); the
`Theorem 2.4` cross-reference fixed; the abstract rewritten to stop claiming hardware
validation, to state L-01 as existence with the 40% as a wager, and to drop the
non-existent Monte Carlo (C-16, C-17); §10.2 relabelled an implementation check, with the
suppressed injection-model failure and the estimator noise floor both reported (C-5); C1
withdrawn as validation with the Jensen argument and the 21/21 one-sided residual (C-11);
§8.6's "three invariants validated" withdrawn in full (C-12); F3 restated as chaos-arm-only
with every measured ρ listed (C-14); §14's false-positive rewrite withdrawn without
replacement, with the standing rule stated in its place (C-4); Table 8's institutional rows
withdrawn with the exchangeability argument for why reconstruction is foreclosed (C-2, C-3,
C-15); §11 relocated out of the validation chapter as a scored negative result carrying the
confusion matrix, the underpowering finding, the threshold-fragility table, and C-8 (C-9);
and the 7.6-year timing claim withdrawn with its null comparison (C-10).

Optional but recommended: `build_corrections_pdf.py`, so the rendering is reproducible.

**Verify before upload:**

```bash
cd CCA_PAPER
python3 corrections_witness.py          # must exit 0
python3 build_corrections_pdf.py        # regenerates CORRECTIONS_v3.pdf
pdflatex -interaction=nonstopmode coherence_collapse_analysis.tex   # must exit 0
```

---

## Metadata changes

**1. Resource type — fix this regardless of anything else.**

Currently displays as **"Peer review."** It is a **preprint**. This is the cheapest and most
misleading item in the whole corrections process: the record presently claims a review
status it never had. Set resource type to `Publication → Preprint`.

**2. Version.** `v5` — `v4` is taken by an accidental deposit (see the blocking note above).

**3. Description.** Prepend to the existing abstract:

> **Version 5 (2026-07-31) is a corrections version.** It corrects eighteen defects in v3 — the last version carrying CCA content, since v4 is an accidental upload —
> several load-bearing. The stability criterion of Theorem 2.3 erred in the *permissive*
> direction — it certifies as stable systems whose defense function is strictly decreasing.
> Both results previously offered as hardware validation of the k_eff identity are identity
> checks, which compute k_eff from the formula and compare it against the same formula; no
> independent empirical test of the identity is presented. The institutional validation
> section does not consult k_eff at all, and scores below chance when re-scored against a
> pre-registered outcome definition. Defects, evidence, and dispositions are enumerated in
> the accompanying corrections note (`CORRECTIONS_v3.pdf`), with a machine witness. **The
> mathematical results — the Kish identity, the Möbius ceiling k_eff → 1/ρ, and the
> corrected stability criterion α/k ≥ d — are unaffected; they are algebraic, and were never
> the subject of the empirical claims that failed.**

**4. Related identifiers.** Add:
- `is documented by` → the RATCHET repository (`CCA_PAPER/CORRECTIONS_v3.md`)
- `is supplemented by` → RATCHET#11

**5. Keep unchanged:** title, authorship, licence, concept DOI.

---

## What the new version says, in one paragraph

Should a reader ask what changed and why: v3 made empirical claims its evidence did not
support, in three separable ways. It validated an *identity* — an algebraic truth that
admits no empirical confirmation and returns perfect agreement against itself for any
formula whatsoever — and presented that agreement as hardware validation. It scored an
institutional classifier that never consults the quantity being validated, and reported 5/5
where a pre-registered outcome definition gives 2/5, below chance. And it revised a success
criterion after seeing results, converting recorded false positives into successes. The
mathematics was never in question and is untouched. What is withdrawn is the claim that the
mathematics has been empirically confirmed.

---

## Why the record is corrected rather than withdrawn

A dead DOI reads as concealment; a corrected one reads as scholarship. The record is cited
by the CIRIS Constitution and by *Corridor Dynamics*, and those citations should resolve to
a version that names its own defects rather than to a gap. The corrections note reports the
fired kills as loudly as the survivals, and keeps the dead visible — withdrawn rows are
struck and annotated in place, not deleted.

---

## After upload

- [ ] Update the citation in `CIRISConstitution` Part 6 to the v5 DOI
- [ ] Correct `Corridor Dynamics` (coherence-ratchet) lines 226, 228, 780 — line 226 credits
      CCA v3 with establishing pre-registration discipline and matched-control nulls, which
      v3 did not have. See RATCHET#14.
- [ ] Re-derive the F-1–F-7 pass status rather than asserting it from CCA v3. Note F-1
      (Kish identity properties) is *proved* and cannot fail empirically, so it should not
      be listed as empirically passing.
- [ ] Close RATCHET#11; leave #12, #13, #14 open
