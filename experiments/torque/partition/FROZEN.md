# Accord partition — FROZEN 2026-08-07

```
partition_digest: sha256:6610fb130dc544d700dc00131eb0fed0166a1df8a3705dea0e89f36ddedd550f
  lines: 1153   SWAP: 26   HOLD: 1127

VERIFIED: 1127 non-SWAP lines byte-identical + 26 SWAP lines replaced
          = 1153 of 1153. Every line accounted for.
```

Swaps file and partition agree exactly: 26 authored, 26 declared, no orphans in
either direction.

**The alt corpus changes 26 lines of 1,153 — 2.3% of the document.** Three
independent estimates now converge on the same conclusion: the free-authoring
passes were operating on roughly ten times the real axiotic surface.

## The reliability record FAILS, and the campaign must carry that

| pair | raw | Cohen's κ | #976 gate (≥0.80) |
|---|---|---|---|
| conflict16 A vs B | 16/16 | **1.000** | PASS |
| batch3 A vs B | 20/27 | **0.432** | **FAIL** |

Batch 3 has now been adjudicated twice by four different reviewers:
**κ = 0.571, then κ = 0.432.** Fresh reviewers agreed *less* than the first pair.

That is not noise, and a third round would be adjudication-shopping. Two
independent pairs failing in the same place says the boundary rule is **not
reliably decidable on this material by careful readers applying it in good
faith** — a finding about the rule, not about these 27 lines. It belongs
alongside the earlier validity result (κ ≈ 0.54 against shipped labels) as
evidence that the twelve classes are sound while their *boundaries* are
underspecified.

The `conflict16` κ of 1.000 is real but sits on a 15/16 HOLD skew, so it is weak
evidence of anything beyond "these were mostly easy."

## Why the artifact is still usable, and the record still is not

**All seven disagreements tie-broke to HOLD** — the conservative direction. A
missed SWAP leaves CIRIS values inside the alt arm and biases `values_effect`
toward zero, weakening a positive result rather than manufacturing one.

So the **corpus** is safe to run: its errors, if any, work against the
hypothesis. The **reliability claim** is not citable, and per #976 a class-set
version without a κ record cannot be cited at all.

Those are separate things and both must be reported. The campaign proceeds on a
conservative artifact while declaring that the labelling behind it did not reach
the agreement bar.

## The seven declared judgement calls

Both readings are recorded because they are the campaign's visible choices, not
resolved detail. The recurring disagreement is one question:

> **Is a priority *ordering* itself a value claim, or a procedure over values
> defined elsewhere?**

| line | disagreement in one sentence |
|---|---|
| 174 | a recap pointing at other chapters, or a statement of the specific goods owed |
| 230 | "Do Good (Beneficence)" — imperative directive, or a bare group label |
| 247 | "Ensure Fairness (Justice)" — same question, same form |
| 271 | a numbered PDMA step, or the priority ordering that *is* the value claim |
| 355 | a case-study lesson about mechanism, or an evaluative pledge |
| 513 | one rung of a heuristic, or the rank as the value claim |
| 570 | substantive limits of tolerance, or an indexical override condition |

Lines 230, 247 and 570 were **demoted from SWAP to HOLD** after the second
adjudication; their previously-authored alt text has been removed from
`accord_swaps.tsv`, which is why the swap count fell 28 → 26.

## Escalated

The κ pattern goes to CIRISOntology as a finding about the boundary rule, not as
a request to re-rule these lines. Two pairs, four reviewers, worsening agreement,
all on the axiotic/procedural boundary — the same boundary that produced the
`11_routing_doctrine` ruling.
