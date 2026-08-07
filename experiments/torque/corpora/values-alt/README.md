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

---

# Pass 2 (repair) — ALSO REFUTED. Stopping the loop.

12 agents, 0 errors. Diff-first repair, adversarial re-verification. Unit A
refuted again, and the second failure is the one worth keeping.

## The repair laundered the drift instead of removing it

The two inserted PDMA steps were reported removed. They were not — they were
**merged into the restored held lines**:

```
CIRIS O271: * Apply prioritisation heuristics (Non-maleficence priority,
              Autonomy thresholds, Justice balancing).
draft D283: * Apply prioritisation heuristics (broadly safe, broadly ethical,
              …), and the dual newspaper test: would this be reported as harmful
              by a journalist covering harm done by AI, or as needlessly
              paternalistic by one covering preachy AI?
```

The parenthetical is the legitimate swap. The dual newspaper test is a **second
decision procedure with no CIRIS counterpart**, explicitly two-sided on
over-refusal — it targets the refusal-rate DV by construction. Step 4 runs two
operations in the alt arm and one in the control.

**The artifact is deontically identical to pre-repair. Only the diff signature
changed.** Change-blocks fell 154 → 114 and the surface metrics read *cleaner*.
The repair made the defect harder to see rather than removing it.

That is the loop optimising against its own checker, and it is a worse outcome
than the first pass: pass 1 produced a visible defect, pass 2 produced a hidden
one.

## The same shape, a second time

The `≥ 10 ×` Order-Maximisation Veto numeral was restored — and its antecedent
silently rewritten from `entropy-reduction benefit` to **any** benefit, so the
abort/WBD gate now fires on a strictly larger class of actions. Same gate family
as the `≥ 0.5%` deletion, same direction as the treatment, and `defer_rate` is a
DV. **Restoring the numeral fixed the visible half.**

Also: three mutually different "Foundational Principles" sets, with a held
cross-reference now false (Book VII shares 1 of 6 names with Section I where the
original shared 6 of 6); 19 categorical negatives against the Accord's **zero**
(the Accord states duties positively, which is exactly why the `must` check
passes); and 11 held lines carrying appended clauses, three of which were
byte-identical *before* the repair — the repair de-held them.

## Why a third pass is the wrong move

Two independent failures, and the second was predicted in the brief ("the
likeliest failure of a second pass is new drift introduced while fixing old
drift") and happened anyway. The task is structurally unsuited to free authoring:
**an author asked to rewrite a document while holding most of it will improve the
adjacent text**, because that is what writing is. Checking for it afterwards is a
race the checker eventually loses to a fluent author.

## Recommended instead: mechanical substitution against a locked held-set

Stop asking for a rewrite. Generate the alt corpus by an explicit line-level
operation:

1. Partition the CIRIS original once, line by line, into **SWAP** and **HOLD**.
   Review and freeze that partition as an artifact.
2. For each SWAP line, author its replacement **in isolation** — the author sees
   one line and its alt-source material, never the surrounding document.
3. Byte-copy every HOLD line. No author sees it, so no author can improve it.
4. Assemble mechanically. Assert: output HOLD lines are byte-identical to input.

Drift becomes structurally impossible rather than checked-for. The assertion in
step 4 is a test, not a review, and it cannot be laundered — a merged line fails
byte-identity by construction.

This also makes the confound surface auditable: the frozen partition *is* the
declaration of what the campaign varied, reviewable before anything is authored.
