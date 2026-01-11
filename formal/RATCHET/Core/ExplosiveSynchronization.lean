/-
RATCHET: Explosive Synchronization Phase Theory

Formalization of phase classification based on ES-proximity.

Source: "Proximity to explosive synchronization determines network collapse
         and recovery trajectories" (PNAS 2025)
         doi.org/10.1073/pnas.2505434122

Key Insight:
  ACF kurtosis measures proximity to explosive (first-order) synchronization.
  High kurtosis → near first-order transition → faster collapse, slower recovery.
-/

import Mathlib.Data.Real.Basic
import Mathlib.Algebra.Order.Field.Basic
import Mathlib.Tactic.NormNum

namespace RATCHET.ExplosiveSynchronization

/-- ES-proximity measurement based on ACF kurtosis. -/
structure ESProximity where
  kurtosis : ℝ
  h_pos : kurtosis > 0

/-- Critical kurtosis (Gaussian baseline = 3). -/
noncomputable def kurtosis_critical : ℝ := 3.0

/-- System phase based on ES-proximity and correlation dynamics. -/
inductive SystemPhase
  | Chaos     -- Low synchronization, high variability
  | Healthy   -- Moderate synchronization, stable
  | Critical  -- At phase boundary
  | Rigidity  -- High synchronization, near explosive transition

/-- Transition type classification. -/
inductive TransitionType
  | SecondOrder  -- Continuous, gradual
  | FirstOrder   -- Discontinuous, explosive
  | Critical     -- At boundary

/-- Classify transition type based on ES-proximity. -/
noncomputable def classify_transition (es : ESProximity) : TransitionType :=
  if es.kurtosis > kurtosis_critical + 1.0 then TransitionType.FirstOrder
  else if es.kurtosis < kurtosis_critical - 0.5 then TransitionType.SecondOrder
  else TransitionType.Critical

/-- Phase classification based on k_eff and ES-proximity. -/
noncomputable def classify_phase (k_eff : ℝ) (es : ESProximity) (k_nominal : ℕ) : SystemPhase :=
  if k_eff < 2 ∧ es.kurtosis > kurtosis_critical + 1.0 then SystemPhase.Rigidity
  else if k_eff > k_nominal / 2 ∧ es.kurtosis < kurtosis_critical - 0.5 then SystemPhase.Chaos
  else if k_eff < 2 ∨ es.kurtosis > kurtosis_critical then SystemPhase.Critical
  else SystemPhase.Healthy

/-- Collapse rate increases with ES-proximity. -/
noncomputable def collapse_rate (es : ESProximity) (base_rate : ℝ) : ℝ :=
  base_rate * (es.kurtosis / kurtosis_critical)

/-- Recovery rate decreases with ES-proximity. -/
noncomputable def recovery_rate (es : ESProximity) (base_rate : ℝ) : ℝ :=
  base_rate * (kurtosis_critical / es.kurtosis)

/-- ES-1: High ES-proximity + low k_eff implies rigidity phase. -/
theorem ES1_high_proximity_rigidity (k_eff : ℝ) (es : ESProximity) (k_nom : ℕ)
    (h_keff_low : k_eff < 2) (h_es_high : es.kurtosis > kurtosis_critical + 1.0) :
    classify_phase k_eff es k_nom = SystemPhase.Rigidity := by
  unfold classify_phase
  simp only [h_keff_low, h_es_high, and_self, ↓reduceIte]

/-- ES-2: Collapse rate is monotonic in ES-proximity. -/
theorem ES2_collapse_rate_mono (es1 es2 : ESProximity) (base : ℝ)
    (h_base_pos : base > 0) (h_es_order : es1.kurtosis < es2.kurtosis) :
    collapse_rate es1 base < collapse_rate es2 base := by
  unfold collapse_rate kurtosis_critical
  have h_crit_pos : (3.0 : ℝ) > 0 := by norm_num
  have h_div_lt : es1.kurtosis / 3.0 < es2.kurtosis / 3.0 :=
    div_lt_div_of_pos_right h_es_order h_crit_pos
  exact mul_lt_mul_of_pos_left h_div_lt h_base_pos

/-- ES-3: Recovery rate is anti-monotonic in ES-proximity. -/
theorem ES3_recovery_rate_mono (es1 es2 : ESProximity) (base : ℝ)
    (h_base_pos : base > 0) (h_es_order : es1.kurtosis < es2.kurtosis) :
    recovery_rate es1 base > recovery_rate es2 base := by
  unfold recovery_rate kurtosis_critical
  have h_crit_pos : (3.0 : ℝ) > 0 := by norm_num
  have h_div : (3.0 : ℝ) / es1.kurtosis > 3.0 / es2.kurtosis :=
    div_lt_div_of_pos_left h_crit_pos es1.h_pos h_es_order
  exact mul_lt_mul_of_pos_left h_div h_base_pos

end RATCHET.ExplosiveSynchronization
