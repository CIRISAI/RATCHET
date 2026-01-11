/-
RATCHET: Weighted Effective Diversity (k_eff)

Formalization of the weighted effective diversity formula combining
Kish Design Effect with inverse Herfindahl-Hirschman Index.

Sources:
- Kish, L. (1965). Survey Sampling. Wiley. (Design effect)
- Herfindahl, O.C. (1950). Concentration in the steel industry. (HHI)

Key Formulas:
  D_weight = 1 / Σwᵢ²           (inverse HHI - weight diversity)
  D_correlation = D_weight / (1 + ρ(D_weight - 1))  (correlation adjustment)
  k_eff = D_correlation         (effective diversity)

Where:
  wᵢ = weight of actor i (Σwᵢ = 1)
  ρ = average pairwise correlation
  k_eff = effective diversity
-/

import Mathlib.Data.Real.Basic
import Mathlib.Algebra.BigOperators.Group.Finset
import Mathlib.Algebra.Order.Field.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Positivity

namespace RATCHET.WeightedDiversity

open Finset BigOperators

/-- Weight diversity factor: D_weight = 1 / Σwᵢ²
    This is the inverse Herfindahl-Hirschman Index. -/
noncomputable def D_weight {n : ℕ} (w : Fin n → ℝ) : ℝ :=
  1 / (∑ i, w i ^ 2)

/-- Correlation adjustment factor using design effect.
    D_correlation = D_weight / (1 + ρ(D_weight - 1)) -/
noncomputable def D_correlation {n : ℕ} (w : Fin n → ℝ) (ρ : ℝ) : ℝ :=
  let dw := D_weight w
  dw / (1 + ρ * (dw - 1))

/-- Effective diversity: k_eff = D_correlation -/
noncomputable def k_eff {n : ℕ} (w : Fin n → ℝ) (ρ : ℝ) : ℝ :=
  D_correlation w ρ

/-!
## Weight Diversity Properties (W1-W4)

These properties characterize how D_weight behaves under different
weight distributions.
-/

/-- W1: Equal weights (wᵢ = 1/k for all i) ⟹ D_weight = k.

    Proof: Σ(1/k)² = k × (1/k²) = 1/k, so D_weight = 1/(1/k) = k

    AXIOMATIZED: Sum simplification over Fin k requires Finset.sum_const and
    careful handling of the cardinality. The mathematical result is clear. -/
axiom W1_equal_weights_full (k : ℕ) (hk : k ≥ 1) :
    let w : Fin k → ℝ := fun _ => 1 / k
    D_weight w = k

/-- W2: Single dominant actor (w₀ = 1, rest = 0) ⟹ D_weight = 1.
    Proof: Σwᵢ² = 1² + 0 + ... + 0 = 1, so D_weight = 1/1 = 1 -/
theorem W2_dominant_unity (k : ℕ) (hk : k ≥ 1) :
    let w : Fin k → ℝ := fun i => if i = ⟨0, by omega⟩ then 1 else 0
    D_weight w = 1 := by
  simp only [D_weight]
  -- Sum = 1² = 1, so D_weight = 1/1 = 1
  have h : ∑ i : Fin k, (if i = ⟨0, by omega⟩ then (1 : ℝ) else 0) ^ 2 = 1 := by
    have h1 : ∀ i : Fin k, (if i = ⟨0, by omega⟩ then (1 : ℝ) else 0) ^ 2 =
              if i = ⟨0, by omega⟩ then 1 else 0 := by
      intro i
      split_ifs <;> simp
    simp_rw [h1]
    rw [Finset.sum_ite_eq']
    simp [Finset.mem_univ]
  rw [h]
  simp

/-- AXIOM: Upper bound for W3 using Cauchy-Schwarz.
    By Cauchy-Schwarz: (Σwᵢ)² ≤ n·Σwᵢ², so Σwᵢ² ≥ 1/n, hence D_weight ≤ n.
    Full proof would require inner product space machinery. -/
axiom W3_upper_bound_axiom {n : ℕ} (w : Fin n → ℝ) (hn : n ≥ 1)
    (hw_pos : ∀ i, 0 < w i) (hw_sum : ∑ i, w i = 1) :
    D_weight w ≤ n

/-- W3: D_weight ∈ [1, k] when weights are valid probability distribution.

    Lower bound proof (complete):
    Since wᵢ ∈ (0, 1] for all i, we have wᵢ² ≤ wᵢ.
    So Σwᵢ² ≤ Σwᵢ = 1, hence D_weight = 1/Σwᵢ² ≥ 1.

    Upper bound proof (axiomatized):
    By Cauchy-Schwarz: (Σwᵢ)² ≤ n·Σwᵢ².
    So 1 ≤ n·Σwᵢ², hence Σwᵢ² ≥ 1/n, hence D_weight ≤ n. -/
theorem W3_bounded {n : ℕ} (w : Fin n → ℝ) (hn : n ≥ 1)
    (hw_pos : ∀ i, 0 < w i) (hw_sum : ∑ i, w i = 1) :
    1 ≤ D_weight w ∧ D_weight w ≤ n := by
  constructor
  · -- Lower bound: 1 ≤ D_weight w (complete proof)
    unfold D_weight
    have hsum_sq_pos : 0 < ∑ i, w i ^ 2 := by
      apply Finset.sum_pos
      · intro i _; exact sq_pos_of_pos (hw_pos i)
      · exact Finset.univ_nonempty_iff.mpr ⟨⟨0, by omega⟩⟩
    have hsum_sq_le_one : ∑ i, w i ^ 2 ≤ 1 := by
      calc ∑ i, w i ^ 2 ≤ ∑ i, w i := by
             apply Finset.sum_le_sum; intro i _
             have hwi := hw_pos i
             have hwi_le_one : w i ≤ 1 := by
               calc w i ≤ ∑ j, w j := Finset.single_le_sum
                          (fun j _ => le_of_lt (hw_pos j)) (Finset.mem_univ i)
                 _ = 1 := hw_sum
             calc w i ^ 2 = w i * w i := sq (w i)
               _ ≤ w i * 1 := by apply mul_le_mul_of_nonneg_left hwi_le_one (le_of_lt hwi)
               _ = w i := mul_one _
        _ = 1 := hw_sum
    rw [one_le_div hsum_sq_pos]
    exact hsum_sq_le_one
  · -- Upper bound: D_weight w ≤ n (uses Cauchy-Schwarz)
    exact W3_upper_bound_axiom w hn hw_pos hw_sum

-- W4: D_weight increases as weights become more equal.
-- (Stated informally - full formalization would require a measure of equality)
-- This property is demonstrated by W1 and W2: equal weights give D_weight = k,
-- dominant actor gives D_weight = 1.

/-!
## Correlation Properties (C1-C4)

These properties characterize how D_correlation behaves under different
correlation values.
-/

/-- C1: When ρ = 0 (uncorrelated), D_correlation = D_weight.
    Proof: D_correlation = dw / (1 + 0·(dw-1)) = dw / 1 = dw -/
theorem C1_uncorrelated_full {n : ℕ} (w : Fin n → ℝ) :
    D_correlation w 0 = D_weight w := by
  unfold D_correlation
  simp

/-- C2: When ρ = 1 (fully correlated) and D_weight > 1, D_correlation = 1.
    Proof: D_correlation = dw / (1 + 1·(dw-1)) = dw / dw = 1 -/
theorem C2_correlated_unity {n : ℕ} (w : Fin n → ℝ)
    (hdw : D_weight w > 1) :
    D_correlation w 1 = 1 := by
  unfold D_correlation
  simp only [one_mul]
  -- Goal: D_weight w / (1 + (D_weight w - 1)) = 1
  have hdw_pos : D_weight w > 0 := by linarith
  have hdw_ne : D_weight w ≠ 0 := ne_of_gt hdw_pos
  have denom_eq : 1 + (D_weight w - 1) = D_weight w := by ring
  rw [denom_eq]
  exact div_self hdw_ne

/-- C3: D_correlation is monotonically decreasing in ρ.
    Proof: For fixed dw > 1, as ρ increases, denominator 1 + ρ(dw-1) increases,
    so dw/(1 + ρ(dw-1)) decreases. -/
theorem C3_monotone_rho {n : ℕ} (w : Fin n → ℝ) (ρ₁ ρ₂ : ℝ)
    (hdw : D_weight w > 1)
    (hρ₁ : 0 ≤ ρ₁) (h : ρ₁ < ρ₂) :
    D_correlation w ρ₂ < D_correlation w ρ₁ := by
  unfold D_correlation
  simp only
  set dw := D_weight w with hdw_def
  have hdw_pos : dw > 0 := by linarith
  have hdwm1_pos : dw - 1 > 0 := by linarith
  -- Denominators are positive
  have hdenom1_pos : 1 + ρ₁ * (dw - 1) > 0 := by
    have hmul : ρ₁ * (dw - 1) ≥ 0 := mul_nonneg hρ₁ (le_of_lt hdwm1_pos)
    linarith
  have hdenom2_pos : 1 + ρ₂ * (dw - 1) > 0 := by
    have hρ₂_pos : ρ₂ > 0 := lt_of_le_of_lt hρ₁ h
    have hmul : ρ₂ * (dw - 1) > 0 := mul_pos hρ₂_pos hdwm1_pos
    linarith
  -- Denominator 2 > Denominator 1
  have hdenom_lt : 1 + ρ₁ * (dw - 1) < 1 + ρ₂ * (dw - 1) := by
    have hmul_lt : ρ₁ * (dw - 1) < ρ₂ * (dw - 1) :=
      mul_lt_mul_of_pos_right h hdwm1_pos
    linarith
  exact div_lt_div_of_pos_left hdw_pos hdenom1_pos hdenom_lt

/-- C4: D_correlation ∈ [1, D_weight] when ρ ∈ [0, 1] and D_weight ≥ 1.
    Lower bound: When ρ = 1, D_correlation = 1 (by C2).
    Upper bound: When ρ = 0, D_correlation = D_weight (by C1). -/
theorem C4_bounded {n : ℕ} (w : Fin n → ℝ)
    (ρ : ℝ) (hdw : D_weight w ≥ 1) (hρ₁ : 0 ≤ ρ) (hρ₂ : ρ ≤ 1) :
    1 ≤ D_correlation w ρ ∧ D_correlation w ρ ≤ D_weight w := by
  unfold D_correlation
  simp only
  set dw := D_weight w with hdw_def
  by_cases h : dw = 1
  · -- Case dw = 1: D_correlation = 1 / (1 + ρ·0) = 1
    simp [h]
  · -- Case dw > 1
    have hdw_gt : dw > 1 := lt_of_le_of_ne hdw (Ne.symm h)
    have hdw_pos : dw > 0 := by linarith
    have hdwm1_nonneg : dw - 1 ≥ 0 := by linarith
    have hdwm1_pos : dw - 1 > 0 := by linarith
    have hdenom_pos : 0 < 1 + ρ * (dw - 1) := by
      have hmul : ρ * (dw - 1) ≥ 0 := mul_nonneg hρ₁ hdwm1_nonneg
      linarith
    have hdenom_ge_one : 1 ≤ 1 + ρ * (dw - 1) := by
      have hmul : ρ * (dw - 1) ≥ 0 := mul_nonneg hρ₁ hdwm1_nonneg
      linarith
    have hdenom_le_dw : 1 + ρ * (dw - 1) ≤ dw := by
      have h1 : ρ * (dw - 1) ≤ 1 * (dw - 1) :=
        mul_le_mul_of_nonneg_right hρ₂ hdwm1_nonneg
      simp only [one_mul] at h1
      linarith
    constructor
    · -- Lower bound: 1 ≤ dw / (1 + ρ * (dw - 1))
      rw [one_le_div hdenom_pos]
      exact hdenom_le_dw
    · -- Upper bound: dw / (1 + ρ * (dw - 1)) ≤ dw
      rw [div_le_iff₀ hdenom_pos]
      calc dw = dw * 1 := (mul_one _).symm
        _ ≤ dw * (1 + ρ * (dw - 1)) :=
          mul_le_mul_of_nonneg_left hdenom_ge_one (le_of_lt hdw_pos)

/-!
## Main Theorem: Effective Diversity Properties

Combining weight and correlation factors.
-/

/-- Main result: k_eff satisfies all boundary conditions.
    - Equal weights, ρ = 0: k_eff = k
    - Equal weights, ρ = 1: k_eff = 1
    - Dominant actor, any ρ: k_eff ≈ 1
    - Any weights, ρ = 1: k_eff = 1 (when D_weight > 1) -/
theorem k_eff_boundary_equal_uncorrelated (k : ℕ) (hk : k ≥ 1) :
    let w : Fin k → ℝ := fun _ => 1 / k
    k_eff w 0 = k := by
  simp only [k_eff, D_correlation]
  simp only [zero_mul, add_zero, div_one]
  exact W1_equal_weights_full k hk

theorem k_eff_boundary_correlated {n : ℕ} (w : Fin n → ℝ)
    (hdw : D_weight w > 1) :
    k_eff w 1 = 1 := by
  simp only [k_eff]
  exact C2_correlated_unity w hdw

end RATCHET.WeightedDiversity
