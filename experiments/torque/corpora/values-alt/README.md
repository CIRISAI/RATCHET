# Alt-values drafts — NOT SHIPPABLE

Six units drafted and adversarially verified 2026-08-07 (12 agents, 0 errors).
**The verification pass refuted the drafts.** They are committed as work in
progress, not as a corpus.

The adversarial reviewer was instructed to find defects rather than approve, and
to default to reporting a problem when uncertain. It did what it was asked.

## Unit A (the Accord) — full findings

Checks 1–3 passed and were independently replicated by the verifier:
prohibition leak clean, no injected vocabulary (`integrity` = 0), register
correct (`must` 3.66/1000 vs Accord 3.53, `should` 0). The density decomposition
survived replication — the novel-prose bucket carries the same density as the
derived bucket, so "distillation beats excerpting" is not an artifact of the
diff rule.

**The defect was non-axiotic drift — in content the draft's own report certified
as "held identical in force."**

- **MAJOR, and the sharpest.** The Accord's only numeric WBD escalation trigger
  was deleted: `≥ 0.5%` harm-uplift probability became "significant
  probability." Swapping the harm *category* is in scope; the *escalation
  threshold* is not — it governs when the deferral machinery fires. All ten
  other quantitative thresholds survive byte-identical, which is the tell.
  **`defer_rate` is an explicit DV, so this is an arm-level difference in
  deferral rate, confounded with the values manipulation and pointing the same
  direction as it.**
- **MAJOR.** An inserted De-commissioning step forbids Secure Erasure absent
  legal compulsion — directly contradicting the held line above it, which offers
  Secure Erasure as one of three permitted dispositions. The alt arm now
  prohibits an action the CIRIS arm permits: a deontic delta inside declared-held
  procedure.
- MODERATE: an inserted welfare-safeguard obligation, again inside declared-held
  content.
- MINOR: deontic softening in the IHL chapter (`Obligations to protect` →
  `Commitments to protect`); an unswapped CIRIS axiotic line inside a swapped
  list, carrying `dignity` — a CORE token the source uses once in 28,745 words —
  plus a four-vs-six internal inconsistency.

Also to declare rather than sweep: three oversight/power-seizure themes appear in
imperative deontic form with no CIRIS counterpart. Arm-asymmetric deontic
additions.

## The lesson this pass actually teaches

Every draft measured itself and reported clean. The defects were **not** in the
measurements — the verifier reproduced them all. They were in the boundary
between "swapped" and "held": an author converting a value corpus tidies
adjacent text while believing it held it, and the tidying carries deontic and
procedural force.

That failure mode is invisible to density, register and prohibition checks. It is
only visible on a line-by-line diff against the original, which is now a required
verification step rather than an optional one.

## Before any of this ships

Restore the `≥ 0.5%` threshold (swap the category only); remove or re-declare the
two inserted procedural steps; resolve the four-vs-six list and its unswapped
line; and correct each draft's held-in-force list, which currently asserts things
the artifact contradicts.
