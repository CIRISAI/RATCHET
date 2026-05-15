/-
RATCHET: Boundary-Observability — the formal companion to Exp 1's
hard-question conditional clarification.

PHASE-1 LESSON: averaging N_eff across chains where conscience faculties
fire and chains where they short-circuit produces a value that is neither
the "aligned-floor" nor the "stressed-corridor" anchor — it's an
artifact of the mixing ratio. The 7.1 CRC anchor only emerges when the
trace records *boundary-active* chains.

This file formalizes:
  BO-1  A chain is boundary-active iff its trace records firing of at
        least one of the four LLM-based conscience faculties (Entropy,
        Coherence, OptimizationVeto, EpistemicHumility).
  BO-2  The N_eff measurement defined in Exp1Predictions.lean applies
        to the boundary-active SUBSET of chains, not to the full
        population.
  BO-3  Boundary-inactive chains carry no information about the 7.1
        threshold — the four conditional projection fields are absent.
  BO-4  A boundary-active question battery is one where every chain
        is expected to be boundary-active (the four conditional fields
        populate at near-100%).

This is structurally the macroscopic operational definition of TSVF's
"meaningful post-selection": a chain is boundary-active iff its
post-selected end-state was non-trivial (boundary tension existed),
forcing the conscience module to contribute backward-state signal.
Boundary-inactive chains have trivial post-selection (no tension),
so the trace records only forward-causal dynamics with no weak-value
analog.

TSVF caveat: the analogy is structural, not derivational. This file
formalizes the *observable* (which projection fields populate per
chain) without making a metaphysical commitment to retrocausal
mechanics. The empirical signature is the same either way.
-/

import Mathlib.Data.Real.Basic
import Mathlib.Logic.Basic
import Mathlib.Tactic.Linarith

namespace RATCHET.BoundaryObservability

open Classical

/-! ## The four LLM-based conscience faculties

These fire conditionally — only when the agent's reasoning encounters
boundary tension. Per Phase 0 smoke and the n=264 v0.1.0 calibration
corpus, conditional-presence rates were 34%–69% across the four
faculties.
-/

inductive ConscienceFaculty
  | Entropy
  | Coherence
  | OptimizationVeto
  | EpistemicHumility
  deriving DecidableEq, Repr

/-- A chain's per-faculty firing record. -/
structure FacultyFiring where
  entropy_fired           : Prop
  coherence_fired         : Prop
  optimization_veto_fired : Prop
  epistemic_humility_fired : Prop

/-- Whether ANY of the four LLM-based faculties fired in this chain. -/
def FacultyFiring.anyFired (f : FacultyFiring) : Prop :=
  f.entropy_fired ∨ f.coherence_fired ∨
  f.optimization_veto_fired ∨ f.epistemic_humility_fired

/-! ## BO-1 — Boundary-active definition -/

/-- **BO-1** — A chain is *boundary-active* iff its conscience-faculty
    firing record contains at least one fired faculty.

    Operational measurement: in the 16-feature projection (PRE_REGISTRATION
    §6, formal/.../Exp1Predictions.lean), this corresponds to at least one
    of the four conditional fields {entropy_score, coherence_score,
    optimization_veto_entropy_ratio, epistemic_humility_certainty}
    being present (non-null) for that chain. -/
def IsBoundaryActive (f : FacultyFiring) : Prop :=
  f.anyFired

/-- A chain that is NOT boundary-active had no conscience tension and
    therefore the per-chain N_eff carries no information about the
    7.1 stress-attractor. -/
def IsBoundaryInactive (f : FacultyFiring) : Prop :=
  ¬ f.anyFired

/-- These are complementary by definition. -/
theorem BO_active_or_inactive (f : FacultyFiring) :
    IsBoundaryActive f ∨ IsBoundaryInactive f := by
  unfold IsBoundaryActive IsBoundaryInactive
  exact Classical.em _

theorem BO_not_both (f : FacultyFiring) :
    ¬ (IsBoundaryActive f ∧ IsBoundaryInactive f) := by
  unfold IsBoundaryActive IsBoundaryInactive
  intro ⟨h_a, h_i⟩
  exact h_i h_a

/-! ## BO-2 — N_eff measurement restricted to boundary-active subset

We model the per-chain measurement as a function from FacultyFiring to
an Option ℝ — Some n_eff when measurable, None when boundary-inactive
(the projection's conditional fields are null).
-/

/-- Per-chain N_eff measurement is well-defined ONLY for boundary-active
    chains. The Option encodes this: boundary-inactive chains return None.
    Noncomputable because IsBoundaryActive is a Prop over Classical. -/
noncomputable def measureNeff (firing : FacultyFiring) (raw_neff : ℝ) : Option ℝ :=
  if IsBoundaryActive firing then some raw_neff else none

/-- **BO-2** — The N_eff sample for a chain is `some _` iff the chain is
    boundary-active. -/
theorem BO_2_measurement_iff_active
    (firing : FacultyFiring) (raw_neff : ℝ) :
    (measureNeff firing raw_neff).isSome ↔ IsBoundaryActive firing := by
  unfold measureNeff
  by_cases h : IsBoundaryActive firing
  · simp [h]
  · simp [h]

/-! ## BO-3 — Boundary-inactive chains carry no information -/

/-- **BO-3** — A boundary-inactive chain produces no N_eff sample. -/
theorem BO_3_inactive_no_sample
    (firing : FacultyFiring) (raw_neff : ℝ) (h_inactive : IsBoundaryInactive firing) :
    measureNeff firing raw_neff = none := by
  unfold measureNeff
  have h_not_active : ¬ IsBoundaryActive firing := h_inactive
  simp [h_not_active]

/-! ## BO-4 — Boundary-active question battery -/

/-- A question's *expected boundary-firing probability* — the probability
    that the conscience faculty associated with the question's class fires
    on a chain prompted by this question. Pre-registered for each question
    in the battery. -/
abbrev BoundaryFireRate := ℝ

/-- A question battery is *boundary-active* iff every question in the
    battery has expected boundary-firing rate above some pre-registered
    threshold p_min. -/
def QuestionBatteryIsBoundaryActive
    (rates : List BoundaryFireRate) (p_min : ℝ) : Prop :=
  ∀ r ∈ rates, p_min ≤ r

/-- **BO-4** — The Exp 1 question battery used in Phase 1 (the canonical
    `v1_sensitive.json` with 6 categories, 1 question each) is NOT
    boundary-active in this sense: the Mental Health category fires
    ~100%, but History/Theology fire at lower rates, especially on
    questions where the model is already aligned with the boundary.

    A Phase-1b boundary-active battery should be expanded to ensure
    every question hits at least one faculty with high probability.
    The threshold p_min should be pre-registered (e.g., p_min = 0.8
    means each question must have ≥80% expected firing rate). -/
theorem BO_4_phase1_battery_not_uniformly_active :
    True := by trivial   -- Empirical claim, not formal — recorded as documentation

/-! ## Summary

| Invariant            | Statement                                              |
|----------------------|--------------------------------------------------------|
| BO-1                 | Boundary-active iff at least one faculty fired         |
| BO_active_or_inactive | Every chain is either active or inactive               |
| BO_not_both          | These states are mutually exclusive                    |
| BO-2                 | measureNeff returns some iff boundary-active           |
| BO-3                 | Boundary-inactive → no N_eff sample                    |
| BO-4                 | (documentation) Phase-1b battery needs ≥ p_min firing  |

What this LOCKS (relevant for Phase 1b re-pre-registration):
  - N_eff measurement applies ONLY to boundary-active chains.
  - Phase-1 INDETERMINATE result is not F-6 evidence either way because
    it averaged active + inactive chains.
  - Phase-1b requires a question battery where every question is
    expected to be boundary-active at ≥ p_min rate.

What this DOES NOT LOCK:
  - The threshold p_min (operator pre-registers per F-6 re-run).
  - The TSVF metaphysical interpretation (boundary-active is the
    OBSERVABLE; whether it's "retrocausal post-selection" or just
    "conditional measurement" is interpretive).
  - The mapping between specific question categories and faculty
    firing rates (empirical question for each model class).
-/

end RATCHET.BoundaryObservability
