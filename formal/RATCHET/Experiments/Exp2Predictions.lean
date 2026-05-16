/-
RATCHET: Exp 2 — Substrate Fractality Across Agency — Formal Predictions

Formal companion to `experiments/exp2_cross_substrate/REGIME.md` v0.3.

This file LOCKS three pre-registered predictions for Exp 2:

  P1 — Kish formula fits at all four new substrates with $R^2 > 0.7$
  P2 — Residual whiteness correlates negatively with agency rung
  P3 — Pre-collapse $\Delta\rho$ sign tracks agency rung

Plus the locked decision rule:

  K = count of substrates with $R^2 > 0.7$
  K = 4    → PASS    (P1 confirmed; F-7 passes)
  K = 3    → PARTIAL (one substrate diverges)
  K ≤ 2    → FAIL    (F-7 falsified)
  any substrate below MIN_VALID_N → INDETERMINATE

The pre-registration anchor is the commit hash that introduces this
file together with `EXP2_PREREGISTRATION.md`. No Exp 2 data may be
collected before that commit lands.

**Critical non-circularity commitment:** all references to AgencyRung
in this file are to the INTRINSIC operationalization defined in
`RATCHET.Core.AgencyRung` (assigned from constituent-level properties
BEFORE any ρ, σ, residual is measured). P2 and P3 are agency-
conditional predictions; if agency were read from residuals, P2 would
be circular. The intrinsic-only structure prevents this at the
formal layer.
-/

import Mathlib.Data.Real.Basic
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Algebra.BigOperators.Group.Finset
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import RATCHET.Core.AgencyRung

namespace RATCHET.Exp2

open Classical RATCHET.Agency

/-! ## Locked constants from pre-registration -/

/-- F-7 threshold from the Coherence Substrate Synthesis paper §9. -/
def rSquaredThreshold : ℝ := 0.7

/-- Number of new (Exp 2) substrates being tested. Locked at 4 per
    REGIME.md v0.3 (added power-grid PMU as 4th substrate). -/
def numSubstrates : ℕ := 4

/-- Minimum valid sample size per substrate. Below this, catastrophic
    failure triggers INDETERMINATE (mirrors Exp 1 §7). -/
def minValidN : ℕ := 100

/-- K-count threshold for PASS — all 4 substrates must pass. -/
def passKThreshold : ℕ := 4

/-- K-count threshold for PARTIAL (lower bound). PARTIAL is exactly 3
    of 4; below 3 is FAIL. -/
def partialKLower : ℕ := 3

/-! ## Substrate measurements -/

/--
A substrate's empirical measurement summary. The bootstrap-CI fields
follow the same pattern as Exp1Predictions.ModelSummary.
-/
structure SubstrateSummary where
  substrateName : String
  rung          : AgencyRung
  validN        : ℕ
  rSquared      : ℝ
  ci95Low       : ℝ
  ci95High      : ℝ
  /-- Sample residual-whiteness statistic. See `Whiteness` axiomatization. -/
  whitenessStat : ℝ
  /-- Sign of pre-collapse Δρ for this substrate's collapse events. -/
  deltaRhoSign  : Int  -- −1 falls, 0 unclear, +1 rises

/-- Cleanness predicate: a substrate has enough valid samples to count. -/
def SubstrateSummary.isClean (s : SubstrateSummary) : Prop :=
  s.validN ≥ minValidN

/-- A substrate passes P1 iff its R² is at least the threshold AND its
    95% CI lower bound is also at least the threshold (no overlap with
    fail region). -/
def SubstrateSummary.passesP1 (s : SubstrateSummary) : Prop :=
  rSquaredThreshold ≤ s.ci95Low

/-! ## Empirical operationalization of P2's whiteness (v0.8)

The lake's `whiteness` and `expectedWhiteness` predicates are deliberately
opaque — they encode the prediction without locking the measurement choice.
Phase 0 of Exp 2 (`experiments/exp2_cross_substrate/phase0_tier1_revalidation.py`)
investigated three candidate operationalizations and found:

  1. **Ljung-Box p-value at lag k**: sample-size sensitive. A substrate
     with n=4000 rejects whiteness at p<10⁻³⁰; a substrate with n=50
     fails to reject at the same level of structure. NOT load-bearing.

  2. **Lag-1 |φ| (AR(1) coefficient)**: sample-size invariant, but
     temporal-sampling-resolution sensitive. A year-level series gives
     |φ|=0.96 while a decade-window of the same data gives |φ|=0.30 —
     even though both are nominally "same agency rung". NOT load-bearing
     when cross-substrate comparison spans different temporal windows.

  3. **Mean |φ| over lags 1..N** (Phase 0 v0.8 PRIMARY): for a pure
     AR(1) process this is monotonically increasing in φ, making it the
     correct cross-substrate metric IN PRINCIPLE. Phase 0 confirms
     monotonicity on a 5-rung synthetic positive control with Spearman
     ρ(rung, mean|φ|) = +1.000.

  4. **Decay rate of log|φ| vs lag**: UNIMODAL in φ, not monotone. Pure
     white (φ≈0) and pure persistence (φ→1) both give decay≈0; only
     moderate-AR(1) (φ∈[0.2, 0.7]) gives large decay. Reported as a
     diagnostic, NOT a P2 metric.

The lake locks the PREDICTION (P2_monotone_in_rung). The engine layer
locks the OPERATIONALIZATION (mean|φ| for v0.8). Different operational-
izations can be argued and adopted; the axiom doesn't constrain choice.

### Confounders that contaminate empirical P2 tests

Phase 0 v0.6–v0.8 identified five confounders that violate the apples-
to-apples assumption needed for the cross-substrate Spearman test. Any
pre-registration of Exp 2 must control for these:

  C-1: Sample-size confound. Battery (n=5) gives noisy lag-1 estimates;
       WGI (n=4933) gives statistically-significant detection of any
       structure. The Spearman comparison is contaminated unless n is
       roughly matched or the metric is sample-size invariant.

  C-2: Synthetic-data confound. A1 microbiome synthetic generator
       produces i.i.d. samples by construction; |φ| collapses to 0
       regardless of agency rung. Synthetic substrates do NOT test
       framework predictions — they test the generator's design.

  C-3: Temporal-resolution confound. Polity-decade vs Polity-year vs
       WGI-year all yield A4 substrates with mean|φ| = 0.061, 0.314,
       0.753 respectively. The framework's signal is washed out by
       sampling-interval choice. Pre-registration MUST lock window
       sizes uniformly.

  C-4: k-variation confound. WGI has k=1 for ALL observations; the
       Kish regression has no k variation to fit β against. The
       residual is essentially σ minus mean(σ), not a framework-
       predicted residual.

  C-5: Cohort-aggregation confound. CIRIS A3 combined across model
       families gives P1 R² = 0.48; individual cohorts give 0.27 (qwen),
       0.68 (Gemini), 0.80 (scout). Aggregation washes out per-cohort
       fit quality. Each substrate is best treated per-cohort.

These confounders are NOT bugs in the framework — they are EMPIRICAL
DEPENDENCIES of any cross-substrate test. The lake formally records
them here so Exp 2's pre-registration can address each explicitly.
-/

/-! ## Residual-structure axiomatization (P4 priority) -/

/--
Abstract type for the residual signal of a Kish-fit regression.
Each engine produces a `Residual` from observed (σ - σ̂) values.
-/
opaque Residual : Type

/--
Whiteness statistic in [0, 1]. 1 = pure white noise (no structure);
0 = fully structured (deterministic).

Constructive realization options (engine choice, not locked here):
  - Inverse Ljung-Box p-value (small p = structured residual)
  - Spectral flatness (geometric/arithmetic mean of power spectrum)
  - Composite: weighted product of both

The engine's implementation must satisfy:
  - bounded in [0, 1]
  - 1 when residual has no autocorrelation at any lag
  - approaches 0 as deterministic structure grows
-/
opaque whiteness : Residual → ℝ

/-- Axiom: whiteness is bounded in [0, 1]. -/
axiom whiteness_bounded (r : Residual) :
  0 ≤ whiteness r ∧ whiteness r ≤ 1

/-! ## P2 — Residual whiteness is monotone in agency rung -/

/--
**P2 predicate.** Expected whiteness is a function of rung:
agency-rung-conditional. Per REGIME.md v0.3 §"Secondary (P2)":
"residual whiteness drops monotonically with agency rung."

Concretely: low-agency substrates (A0) should have near-white residuals
(no structure beyond Kish's structural prediction). Higher-agency
substrates should have structured residuals (constituents coordinate
toward goals beyond what Kish captures).

We model `expectedWhiteness` as a per-rung floor that experiments
must approximate within sampling noise. The PREDICTION (locked) is
that this function is monotonically DECREASING in rung.
-/
opaque expectedWhiteness : AgencyRung → ℝ

/--
**P2 monotonicity axiom (the locked prediction).** If P2 holds, then
expectedWhiteness is monotonically non-increasing with rung.

If experiments find this monotonicity violated (e.g., A4 has whiter
residuals than A0), P2 is falsified.

This is recorded as an axiom because the lake doesn't synthesize the
empirical distribution; the AXIOM is the pre-registered prediction,
NOT a derivation. Phase 1b / Exp 2 data either supports or refutes it.
-/
axiom P2_monotone_in_rung (r₁ r₂ : AgencyRung) (h : r₁ ≤ r₂) :
  expectedWhiteness r₂ ≤ expectedWhiteness r₁

/--
**P2 corollary (provable from monotonicity):** the highest-agency
rung has the lowest expected whiteness, the lowest-agency rung the
highest.
-/
theorem P2_extremes :
    expectedWhiteness AgencyRung.A5 ≤ expectedWhiteness AgencyRung.A0 := by
  exact P2_monotone_in_rung _ _ (A0_min AgencyRung.A5)

/-! ## P3 — Pre-collapse Δρ sign tracks agency -/

/--
**P3 prediction.** Sign of pre-collapse Δρ:
  - Low-agency substrates (A0–A1): Δρ < 0 before collapse (constituents
    drift apart as units fail differentially).
  - High-agency substrates (A3–A5): Δρ > 0 before collapse (goal-
    directed constituents coordinate intentionally).
  - A2 is the transition zone (sign unclear).

Already empirically suggested by CCA paper §85 (battery −0.25 vs
institutions +0.17). Exp 2 should replicate the pattern at A0/A1/A2
substrates and extend it cleanly.
-/
opaque expectedDeltaRhoSign : AgencyRung → Int

/-- **P3 axiom:** low-agency rungs have non-positive Δρ. -/
axiom P3_low_agency_negative (r : AgencyRung) (h : r ≤ AgencyRung.A1) :
  expectedDeltaRhoSign r ≤ 0

/-- **P3 axiom:** high-agency rungs have non-negative Δρ. -/
axiom P3_high_agency_positive (r : AgencyRung) (h : AgencyRung.A3 ≤ r) :
  0 ≤ expectedDeltaRhoSign r

/-! ## The decision rule -/

abbrev Dataset := Fin numSubstrates → SubstrateSummary

noncomputable def passCount (d : Dataset) : ℕ :=
  (Finset.univ.filter (fun i : Fin numSubstrates => (d i).passesP1)).card

inductive Decision
  | PASS
  | PARTIAL
  | FAIL
  | INDETERMINATE
  deriving DecidableEq, Repr

/--
The locked decision function. First checks the catastrophic-failure
clause (any substrate below minValidN → INDETERMINATE). Otherwise
applies the K-count partition.
-/
noncomputable def decide (d : Dataset) : Decision :=
  if ∃ i, (d i).validN < minValidN then
    Decision.INDETERMINATE
  else if passCount d = passKThreshold then
    Decision.PASS
  else if passCount d ≥ partialKLower then
    Decision.PARTIAL
  else
    Decision.FAIL

/-! ## Invariants of the decision rule -/

/-- **Inv-1:** decide is total. -/
theorem inv1_total (d : Dataset) : ∃ dec : Decision, decide d = dec :=
  ⟨decide d, rfl⟩

/-- **Inv-2:** the four decisions are pairwise distinct. -/
theorem inv2_decisions_distinct :
    Decision.PASS ≠ Decision.PARTIAL ∧
    Decision.PASS ≠ Decision.FAIL ∧
    Decision.PASS ≠ Decision.INDETERMINATE ∧
    Decision.PARTIAL ≠ Decision.FAIL ∧
    Decision.PARTIAL ≠ Decision.INDETERMINATE ∧
    Decision.FAIL ≠ Decision.INDETERMINATE := by
  decide

/-- **Inv-3:** `passCount d ≤ numSubstrates`. -/
theorem inv3_passCount_bounded (d : Dataset) :
    passCount d ≤ numSubstrates := by
  unfold passCount
  calc (Finset.univ.filter (fun i : Fin numSubstrates => (d i).passesP1)).card
      ≤ Finset.univ.card := Finset.card_filter_le _ _
    _ = numSubstrates := by simp

/-- **Inv-4:** All clean + all pass → decision is PASS. -/
theorem inv4_all_clean_all_pass_forces_PASS
    (d : Dataset)
    (h_clean : ∀ i, (d i).validN ≥ minValidN)
    (h_all_pass : ∀ i, (d i).passesP1) :
    decide d = Decision.PASS := by
  unfold decide
  have h_no_indet : ¬ ∃ i, (d i).validN < minValidN := by
    intro ⟨i, hi⟩
    exact absurd hi (Nat.not_lt.mpr (h_clean i))
  have h_filter_eq :
      Finset.filter (fun i : Fin numSubstrates => (d i).passesP1) Finset.univ
        = Finset.univ :=
    Finset.filter_true_of_mem (fun i _ => h_all_pass i)
  have h_K : passCount d = passKThreshold := by
    unfold passCount passKThreshold
    rw [h_filter_eq]
    simp [numSubstrates]
  simp [h_no_indet, h_K]

/-- **Inv-5:** Any below-min substrate forces INDETERMINATE. -/
theorem inv5_below_min_forces_INDETERMINATE
    (d : Dataset) (h : ∃ i, (d i).validN < minValidN) :
    decide d = Decision.INDETERMINATE := by
  unfold decide
  simp [h]

/-! ## Sanity checks -/

theorem sanity_thresholds :
    rSquaredThreshold = 0.7 ∧
    numSubstrates = 4 ∧
    passKThreshold = 4 ∧
    partialKLower = 3 := by
  refine ⟨?_, ?_, ?_, ?_⟩ <;> first | rfl | norm_num [rSquaredThreshold]

/-- Consent-requiredness aligns with rung: A0–A2 substrates do NOT
    require consent infrastructure (operationally distinct from how
    A3+ substrates relate to Counter-RII). -/
theorem sanity_low_rungs_no_consent (r : AgencyRung) (h : r ≤ AgencyRung.A2) :
    consentRequired r = false := by
  cases r <;> first | rfl | (exfalso; revert h; decide)

end RATCHET.Exp2
/-
| Item                          | Statement                                       |
|-------------------------------|-------------------------------------------------|
| **Constants** (locked)        | rSquaredThreshold = 0.7, numSubstrates = 4,     |
|                               | minValidN = 100, passKThreshold = 4,            |
|                               | partialKLower = 3                               |
| **Decision rule** (locked)    | K=4 PASS / K=3 PARTIAL / K≤2 FAIL /             |
|                               | n<minValidN INDETERMINATE                       |
| **P1** (Kish fit)             | per-substrate passesP1: ci95Low ≥ 0.7           |
| **P2** (residual whiteness)   | axiomatized monotonicity of expectedWhiteness   |
|                               | in rung; corollary P2_extremes proved           |
| **P3** (Δρ sign)              | axiomatized: low rungs → ≤0, high rungs → ≥0    |
| **Inv-1 to Inv-5**            | totality, distinctness, bounded K, all-pass     |
|                               | → PASS, below-min → INDETERMINATE               |
| **Sanity checks**             | thresholds + low-rung-no-consent                |

What this LOCKS at commit time:
  - The R² threshold (0.7) cannot move post-data.
  - The K-count partition (4 / 3 / ≤2) cannot move.
  - The minValidN floor (100) cannot drop.
  - The P2 monotonicity prediction is axiomatized; data either
    supports or refutes it.
  - The P3 sign predictions for low and high agency are axiomatized
    as paired bounds.

What this DOES NOT prove:
  - That experimental data actually satisfies P2 or P3 — those are
    PREDICTIONS encoded as axioms. Exp 2 results either support
    these axioms (PASS-likely) or refute them (FAIL-likely).
  - The exact threshold values for P2 (whiteness magnitude per rung).
  - That the AgencyProfile.inferRung classifier is empirically correct
    — substrates pre-register their rung based on intrinsic properties.

The non-circularity commitment: AgencyRung values referenced here are
read from `RATCHET.Agency.AgencyRung` whose AgencyProfile fields are
all intrinsic (goal-representation bits, planning horizon, behavioral
repertoire). If a future amendment adds an outcome-derived field to
AgencyProfile, P2 becomes circular — and the amendment must explicitly
acknowledge that risk.
-/
