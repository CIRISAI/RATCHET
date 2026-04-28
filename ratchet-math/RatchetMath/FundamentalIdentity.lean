import Mathlib.Data.Real.Basic
import Mathlib.Tactic

namespace RATCHET.Math

noncomputable def k_eff (k : ℝ) (rho : ℝ) : ℝ :=
  k / (1 + rho * (k - 1))

theorem rigidity_collapse (k : ℝ) (hk : k ≠ 0) :
    k_eff k 1 = 1 := by
  unfold k_eff
  field_simp

noncomputable def defense_function (k rho lambda sigma : ℝ) : ℝ :=
  k_eff k rho * (1 - rho) * lambda * sigma

def singularity_condition (K_req rho : ℝ) : Prop :=
  K_req * rho ≥ 1

noncomputable def rho_crit : ℝ := (43:ℝ) / 100
noncomputable def k_eff_at_collapse : ℝ := (23:ℝ) / 10

theorem k_eff_rho_crit_relation :
    (1:ℝ) / rho_crit - k_eff_at_collapse < (1:ℝ)/10 ∧ 
    (1:ℝ) / rho_crit - k_eff_at_collapse > -(1:ℝ)/10 := by
  unfold rho_crit k_eff_at_collapse
  constructor
  · norm_num
  · norm_num

end RATCHET.Math
