# Pre-registration — F-3 re-run under a recovery-time fragility measure

Written 2026-08-01 BEFORE any recovery measurement. Inspected only
exp89_recovery_dynamics.py's method (threshold-crossing on a recovery curve) and the
prior F-3 run's rho ranges. No tau has been computed.

## Why re-run

The first F-3 run fired against the corridor: fragility measured as perturbation
RESPONSE fell 3.7x above the upper edge instead of rising. That result carries a
known confound -- response amplitude conflates a robust system with a sensor array
that has stopped discriminating, because as rho -> 1 the array approaches one
effective sensor and one sensor responds less to spatial perturbation regardless of
robustness. The objection reaches the F-series chaos arm too, which used the same
measure.

## The measure

Recovery time constant tau: perturb with a sustained workload, release, and fit the
array-mean timing's return to baseline as mu(t) - mu_base ~ exp(-t/tau).

Why this is immune to the confound: tau is a TEMPORAL property of the array mean,
not a spatial-discrimination property. A collapsed array (k_eff -> 1) still has a
well-defined mean with a well-defined relaxation rate. Critical slowing down --
tau diverging as a system approaches a transition -- is the standard early-warning
signature and is what the corridor claim predicts near the rigidity edge.

Fit: least squares on log|mu(t) - mu_base| over the decay window, tau = -1/slope.
Runs failing to decay (no usable window, or positive slope) are recorded as
non-recovering rather than dropped.

## Control (mandatory, decides interpretability)

Compute tau for the ARRAY MEAN and for a SINGLE randomly chosen sensor in the same
run. If tau is a spatial-discrimination artifact it will differ systematically
between them as rho rises. If single-sensor and array-mean tau agree within their
spread, tau is not carrying the confound and the measure is admissible.

## Registered predictions

| # | Prediction | If confirmed | If refuted |
|---|---|---|---|
| R-a | array-mean tau and single-sensor tau agree (ratio within [0.5, 2.0] across regimes) | the measure is confound-free; F-3 adjudicable | measure is contaminated; report F-3 as UNRESOLVED under both measures |
| R-b | tau at rigidity (rho >= 0.43) exceeds tau at healthy by >= 1 pooled sd | critical slowing down present; corridor SURVIVES under the better measure, and the first F-3 firing is attributed to the response confound | corridor claim FIRES again, now under a measure immune to the objection raised against the first run |
| R-c | tau at chaos exceeds tau at healthy | chaos arm reproduces under the new measure | chaos arm does not reproduce; the F-series result rests on the confounded measure |

## Decision rule, fixed now

F-3 is REPORTED AS SURVIVING iff R-a confirms AND R-b confirms. It FIRES iff R-a
confirms and R-b refutes. If R-a refutes, F-3 is UNRESOLVED and neither the first
run's firing nor a survival may be claimed.

Chaos regime (rho < 0.10) was not reachable in the prior run (min 0.122). If it is
not reached again, R-c is reported untested rather than scored.

## Reporting

All trials, both taus, achieved rho, and fit quality reported whichever way it falls.
