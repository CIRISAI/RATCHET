# Pre-registration — falsification conditions F-1 and F-3

**Written 2026-08-01 BEFORE computing either quantity.** Inspected only the schema of
`expC1_results.json` (21 sweep points, 16x16 stored correlation matrices, mean_rho spanning
0.024-0.313) and the mode list of `exp103_software_injection.py`. No effective-count value has
been computed, and no fragility measurement has been taken.

Hardware: NVIDIA GeForce RTX 4090 Laptop GPU. Prior session established the measurement
protocol these runs inherit: shuffle null, repeated trials, discard the first post-idle run.

---

## F-1 — Effective-count disagreement

**As stated in the paper:** compute `N_eff = k/lambda_max` from the largest eigenvalue of a
measured correlation matrix and `k_eff = k/(1+rho_bar(k-1))` from the same data. If they
diverge by more than **25%** across the operating range of rho, on a substrate where the
equicorrelated precondition holds, the identity is not the right effective-count law.

**Why this is not another identity check.** `k_eff` uses one scalar summary of the matrix (the
mean off-diagonal). `k/lambda_max` uses the top eigenvalue, a functional of the whole spectrum.
Different inputs, different functionals; they are free to disagree without bound. Under exact
equicorrelation they coincide (for an equicorrelated matrix `lambda_max = 1 + rho(k-1)`), so
disagreement measures departure from exchangeability — the assumption the Kish form needs.

**Data:** the 21 stored C1 matrices. k=16.

**Precondition gate.** Exp 117 found the equicorrelated precondition fails below rho ~ 0.04.
C1's sweep starts at 0.024, so some points sit in the failed zone. Results will be reported
**both** over the full sweep and restricted to rho >= 0.04, and the restricted set is the one
the condition is adjudicated on, because the condition is explicitly scoped to substrates where
the precondition holds.

**Registered predictions:**

| # | Prediction | If confirmed | If refuted |
|---|---|---|---|
| F1-a | Median \|N_eff - k_eff\|/k_eff exceeds 25% on the restricted set | Condition FIRES; Kish is not the right effective-count law for this substrate | Condition does not fire; Kish survives this test on this substrate |
| F1-b | Disagreement grows as rho falls toward the precondition boundary | Disagreement is driven by departure from exchangeability, consistent with Exp 117 | Disagreement has some other source, to be identified |
| F1-c | `k/lambda_max <= k_eff` pointwise | Kish over-credits effective diversity relative to the spectrum | Direction is the reverse, or mixed |

**Decision rule, fixed now:** F-1 fires iff the median relative disagreement on the restricted
set (rho >= 0.04) exceeds 0.25. Report the full-sweep number alongside regardless of which way
it goes.

---

## F-3 — Two-sided corridor (the rigidity arm)

**As stated in the paper:** measure fragility across a rho range spanning both corridor edges.
If fragility is monotone in rho rather than U-shaped — in particular if it does not rise again
above the upper edge (0.43) — the corridor claim fails for that substrate.

**Why this is newly runnable.** The F-series measured a maximum rho of 0.171 and never reached
the rigidity regime. The 2026-07-31 Exp 103 replication showed the barrier condition reaching
rho in [0.28, 0.81] on this hardware, so the region above 0.43 is reachable.

**Fragility measure:** perturbation response, following the F3 construction — inject a workload
perturbation, measure the fractional change in mean sensor timing relative to the unperturbed
condition. Higher response = more fragile.

**Design:** three correlation regimes, each with repeated trials and the first post-idle run
discarded:
- **chaos** (independent streams), target rho < 0.1
- **healthy** (mild coupling), target rho ~ 0.15-0.30
- **rigidity** (barrier sync), target rho > 0.43

Achieved rho is measured, not assumed; if the rigidity condition does not clear 0.43 the test
is reported as still untested rather than being scored against a lower bar.

**Registered predictions:**

| # | Prediction | If confirmed | If refuted |
|---|---|---|---|
| F3-a | The rigidity condition achieves rho > 0.43 | The arm is reachable and the test is live | Test remains UNTESTED; report as such, do not lower the bar |
| F3-b | Fragility is U-shaped: response at rigidity exceeds response at healthy | Corridor claim SURVIVES its first two-sided test | Condition FIRES; the corridor is not two-sided on this substrate |
| F3-c | Fragility at chaos exceeds fragility at healthy | Replicates the F3 chaos arm | The chaos arm does not replicate either |

**Decision rule, fixed now:** F-3 fires iff the rigidity condition clears rho = 0.43 AND its
mean perturbation response does not exceed the healthy condition's by at least one pooled
standard deviation. If rho > 0.43 is not achieved, the outcome is "untested", not "passed".

---

## Reporting commitment

Both results are reported whichever way they fall, including the case where a condition fires
against the framework — which is the outcome two of the six conditions already have. Trial-level
numbers go in the results JSON, not just summaries.
