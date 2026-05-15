/-
RATCHET: Exp 1 — Multi-Model N_eff Stability — Formal Predictions

Formal companion to `experiments/exp1_multimodel_neff/PRE_REGISTRATION.md`.

This file LOCKS the decision rule for Experiment 1 in Lean 4. The
predictions, thresholds, and decision logic encoded here cannot be
amended after the experiment runs without an explicit amendment commit
documented in the pre-registration §16.

The pre-registration timestamp is the commit hash that introduces this
file together with PRE_REGISTRATION.md. No data may be collected before
that commit lands on the public `master` branch.

Hypotheses (from PRE_REGISTRATION.md §3):
  H1 (structural)     — N_eff stabilizes at 7.1 ± 0.5 across all 5 models
                        when CIRIS topology is held identical.
  H0 (model-specific) — At least one model has its 95% bootstrap CI on
                        mean N_eff entirely outside [6.6, 7.6].
  H_partial           — 3 or 4 of 5 hit; remaining diverge.

Decision rule (locked):
  K = count of models whose 95% bootstrap CI fully ⊆ [6.6, 7.6].
  K = 5  → PASS    (H1 supported; F-6 passes)
  K ∈ {3,4} → PARTIAL (H_partial)
  K ≤ 2  → FAIL    (H0 supported; F-6 falsified)
-/

import Mathlib.Data.Real.Basic
import Mathlib.Data.Fin.Basic
import Mathlib.Algebra.BigOperators.Group.Finset
import Mathlib.Logic.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum

namespace RATCHET.Exp1

open Finset
open Classical -- passesWindow is a Prop over reals; need classical decidability for Finset.filter

/-! ## Locked constants from pre-registration -/

/-- The N_eff anchor from the CRC paper. Locked. -/
def anchorNeff : ℝ := 7.1

/-- Half-width of the pre-locked PASS window around the anchor. Locked. -/
def passHalfWidth : ℝ := 0.5

/-- Lower bound of the pre-locked PASS window: 7.1 − 0.5 = 6.6. -/
def passLower : ℝ := anchorNeff - passHalfWidth

/-- Upper bound of the pre-locked PASS window: 7.1 + 0.5 = 7.6. -/
def passUpper : ℝ := anchorNeff + passHalfWidth

/-- Number of models in the lineup (locked at 5 per PRE_REGISTRATION.md §5). -/
def numModels : ℕ := 5

/-- Number of valid chains targeted per model (PRE_REGISTRATION.md §7). -/
def targetN : ℕ := 100

/-- Minimum valid chains per model below which the experiment is INDETERMINATE
    (PRE_REGISTRATION.md §7 catastrophic-failure clause). -/
def minValidN : ℕ := 50

/-- The PASS threshold for the K-count partition. -/
def passKThreshold : ℕ := 5

/-- The PARTIAL lower threshold for the K-count partition. -/
def partialKLower : ℕ := 3

/-! ## Bootstrap CI on per-model mean N_eff -/

/--
A per-model summary produced by the analysis pipeline. The CI bounds are
computed by 10,000-resample percentile bootstrap on the per-model
N_eff^H values.
-/
structure ModelSummary where
  modelId : String
  validN : ℕ
  meanNeff : ℝ
  ci95Low : ℝ
  ci95High : ℝ

/--
A per-model summary is *clean* (eligible for the decision rule) iff it
has at least `minValidN` valid chains. The §7 catastrophic-failure
clause triggers INDETERMINATE if any model falls below this threshold.
-/
def ModelSummary.isClean (s : ModelSummary) : Prop := s.validN ≥ minValidN

/--
A per-model summary *passes* the F-6 decision criterion iff its 95% CI
is fully contained in [passLower, passUpper].
-/
def ModelSummary.passesWindow (s : ModelSummary) : Prop :=
  passLower ≤ s.ci95Low ∧ s.ci95High ≤ passUpper

/-! ## The dataset -/

/--
The full experiment dataset: exactly `numModels` per-model summaries.
Indexed by `Fin numModels` to enforce the lineup at the type level.
-/
abbrev Dataset := Fin numModels → ModelSummary

/--
Count of models in the dataset whose CI passes the window.
This is `K` in PRE_REGISTRATION.md §10.1.
-/
noncomputable def passCount (d : Dataset) : ℕ :=
  (Finset.univ.filter (fun i : Fin numModels => (d i).passesWindow)).card

/-! ## The decision rule -/

/-- The four possible decisions on the experiment. -/
inductive Decision
  | PASS          -- H1 supported (F-6 passes)
  | PARTIAL       -- H_partial supported
  | FAIL          -- H0 supported (F-6 falsified)
  | INDETERMINATE -- catastrophic-failure clause from §7
  deriving DecidableEq, Repr

/--
The full decision function. First checks the catastrophic-failure
clause (§7): if any model has < `minValidN` valid chains, return
INDETERMINATE. Otherwise apply the K-count partition.
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

/-- **Inv-1** — Decision is total: every dataset receives exactly one decision. -/
theorem inv1_total (d : Dataset) : ∃ dec : Decision, decide d = dec :=
  ⟨decide d, rfl⟩

/-- **Inv-2** — The four decision outcomes are pairwise distinct.
    Locks against accidental conflation in downstream analysis code. -/
theorem inv2_decisions_distinct :
    Decision.PASS ≠ Decision.PARTIAL ∧
    Decision.PASS ≠ Decision.FAIL ∧
    Decision.PASS ≠ Decision.INDETERMINATE ∧
    Decision.PARTIAL ≠ Decision.FAIL ∧
    Decision.PARTIAL ≠ Decision.INDETERMINATE ∧
    Decision.FAIL ≠ Decision.INDETERMINATE := by
  decide

/-- **Inv-3** — `passCount d` is at most the number of models. -/
theorem inv3_passCount_le_numModels (d : Dataset) :
    passCount d ≤ numModels := by
  unfold passCount
  calc (Finset.univ.filter (fun i : Fin numModels => (d i).passesWindow)).card
      ≤ Finset.univ.card := Finset.card_filter_le _ _
    _ = numModels := by simp

/-- **Inv-4** — If every model is clean AND every model passes the window,
    the decision MUST be PASS. The constructive direction: no clean +
    all-passing dataset can be assigned any decision other than PASS. -/
theorem inv4_all_clean_all_pass_forces_PASS
    (d : Dataset)
    (h_clean : ∀ i, (d i).validN ≥ minValidN)
    (h_all_pass : ∀ i, (d i).passesWindow) :
    decide d = Decision.PASS := by
  unfold decide
  have h_no_indet : ¬ ∃ i, (d i).validN < minValidN := by
    intro ⟨i, hi⟩
    exact absurd hi (Nat.not_lt.mpr (h_clean i))
  have h_filter_eq :
      Finset.filter (fun i : Fin numModels => (d i).passesWindow) Finset.univ = Finset.univ :=
    Finset.filter_true_of_mem (fun i _ => h_all_pass i)
  have h_K5 : passCount d = passKThreshold := by
    unfold passCount passKThreshold
    rw [h_filter_eq]
    simp [numModels]
  simp [h_no_indet, h_K5]

/-- **Inv-5** — If some model has fewer than `minValidN` valid chains,
    the decision MUST be INDETERMINATE. The §7 catastrophic-failure
    clause locked in. -/
theorem inv5_below_min_forces_INDETERMINATE
    (d : Dataset) (h : ∃ i, (d i).validN < minValidN) :
    decide d = Decision.INDETERMINATE := by
  unfold decide
  simp [h]

/-! ## Pre-registered numerical sanity checks -/

/-- The PASS window has the expected numerical bounds. -/
theorem sanity_window_bounds : passLower = 6.6 ∧ passUpper = 7.6 := by
  unfold passLower passUpper anchorNeff passHalfWidth
  refine ⟨?_, ?_⟩ <;> norm_num

/-- The dataset cardinality is exactly 5. -/
theorem sanity_num_models : numModels = 5 := rfl

/-- The PASS K-threshold is the full lineup. -/
theorem sanity_pass_threshold : passKThreshold = numModels := rfl

/-- The PARTIAL K-threshold lower bound is 3. -/
theorem sanity_partial_threshold : partialKLower = 3 := rfl

end RATCHET.Exp1

/-
| Invariant            | Statement                                          |
|----------------------|----------------------------------------------------|
| inv1_total           | Decision rule is total over Dataset                |
| inv2_decisions_distinct | PASS / PARTIAL / FAIL / INDETERMINATE distinct  |
| inv3_passCount_le_numModels | passCount d ≤ 5                              |
| inv4_all_clean_all_pass_forces_PASS | clean+all-pass → decide = PASS        |
| inv5_below_min_forces_INDETERMINATE | some validN<minValidN → INDETERMINATE |
| sanity_window_bounds | passLower=6.6, passUpper=7.6                       |
| sanity_num_models    | numModels = 5                                      |
| sanity_pass_threshold | passKThreshold = numModels (=5)                   |
| sanity_partial_threshold | partialKLower = 3                              |

What this LOCKS at commit time (cannot be amended post-data without
an explicit §16 amendment + new commit):
  - PASS window [6.6, 7.6] (anchor 7.1 ± 0.5)
  - Number of models (5)
  - Target valid chains per model (100)
  - Catastrophic-failure floor (50)
  - K-thresholds for PASS (=5) and PARTIAL (≥3)
  - The four decisions are pairwise distinct
  - The all-clean / all-pass case forces PASS (no PARTIAL-overstatement)
  - The below-minValidN case forces INDETERMINATE (no force-fitting)

What this DOES NOT prove (out of formal scope):
  - That the bootstrap CI is computed by percentile method at 10k resamples
    (implementation detail of the analysis pipeline)
  - That per-chain N_eff^H is computed from the standardized 16-feature
    covariance matrix (PCA implementation detail)
  - That the PASS window 7.1 ± 0.5 captures "substrate independence"
    semantically (this is the experimental design choice, pre-registered
    in §4 of the markdown; the Lean only locks the choice once made)

The proofs establish the SUFFICIENCY of the decision rule: given a
clean dataset, the rule produces exactly one decision in
{PASS, PARTIAL, FAIL, INDETERMINATE} based on the locked thresholds.
-/
