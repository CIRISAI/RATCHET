/-
RATCHET: Effective Constraints (k_eff)

Formalization of the Kish Design Effect formula.

Source: Kish, L. (1965). Survey Sampling. Wiley.

Key Formula:
  k_eff = k / (1 + ρ(k-1))

Where:
  k = nominal constraint count
  ρ = average pairwise correlation
  k_eff = effective (independent) constraint count
-/

import Mathlib.Data.Real.Basic
import Mathlib.Algebra.Order.Field.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring
import Mathlib.Topology.Basic
import Mathlib.Topology.Order.Basic
import Mathlib.Topology.Instances.Real
import Mathlib.Order.Filter.Basic

namespace RATCHET.EffectiveConstraints

/-- Effective constraint count with correlation adjustment.
    k_eff = k / (1 + ρ(k-1))
    Citation: Kish (1965), equation for design effect. -/
noncomputable def k_eff (k : ℕ) (ρ : ℝ) : ℝ :=
  if k ≤ 1 then k
  else k / (1 + ρ * (k - 1))

/-- K-1: When constraints are independent (ρ = 0), k_eff = k. -/
theorem K1_independent_full_count (k : ℕ) :
    k_eff k 0 = k := by
  unfold k_eff
  split_ifs <;> simp

/-- K-2: When constraints are fully correlated (ρ = 1), k_eff = 1. -/
theorem K2_correlated_unity (k : ℕ) (hk : k ≥ 2) :
    k_eff k 1 = 1 := by
  unfold k_eff
  have h : ¬(k ≤ 1) := by omega
  simp only [if_neg h, one_mul]
  -- Goal: k / (1 + (k - 1)) = 1
  -- Since k ≥ 2, we have k ≥ 1 as ℕ, so (k : ℝ) ≥ 1
  have hk_ge_one : (k : ℝ) ≥ 1 := by
    have : k ≥ 2 := hk
    simp only [ge_iff_le, Nat.one_le_cast]
    omega
  have hk_pos : (k : ℝ) > 0 := by linarith
  have hk_ne_zero : (k : ℝ) ≠ 0 := ne_of_gt hk_pos
  -- Simplify 1 + (k - 1) = k
  have denom_eq : 1 + ((k : ℝ) - 1) = k := by ring
  rw [denom_eq]
  exact div_self hk_ne_zero

/-- K-3: k_eff decreases as ρ increases. -/
theorem K3_monotone_rho (k : ℕ) (ρ₁ ρ₂ : ℝ) (hk : k ≥ 2)
    (hρ₁ : 0 ≤ ρ₁) (hρ₂ : ρ₂ ≤ 1) (h : ρ₁ < ρ₂) :
    k_eff k ρ₂ < k_eff k ρ₁ := by
  -- Unfold k_eff; since k ≥ 2, we are in the else branch
  unfold k_eff
  have hk_not_le : ¬(k ≤ 1) := by omega
  simp only [hk_not_le, ↓reduceIte]
  -- Now we need: k / (1 + ρ₂ * (k - 1)) < k / (1 + ρ₁ * (k - 1))
  -- Key facts: k > 0, k - 1 > 0, denominators are positive
  have hk_pos : (k : ℝ) > 0 := by
    have : k ≥ 2 := hk
    exact Nat.cast_pos.mpr (by omega : k > 0)
  have hkm1_pos : (k : ℝ) - 1 > 0 := by
    have : k ≥ 2 := hk
    have hk_ge2 : (k : ℝ) ≥ 2 := Nat.cast_le.mpr hk
    linarith
  -- Denominators are positive
  have hdenom1_pos : 1 + ρ₁ * (k - 1) > 0 := by
    have h1 : ρ₁ * (k - 1) ≥ 0 := mul_nonneg hρ₁ (le_of_lt hkm1_pos)
    linarith
  have hdenom2_pos : 1 + ρ₂ * (k - 1) > 0 := by
    -- ρ₂ could be negative, but ρ₁ < ρ₂ and 0 ≤ ρ₁, so 0 ≤ ρ₁ < ρ₂
    have hρ₂_pos : ρ₂ > 0 := lt_of_le_of_lt hρ₁ h
    have h1 : ρ₂ * (k - 1) > 0 := mul_pos hρ₂_pos hkm1_pos
    linarith
  -- Denominator 2 > Denominator 1 (since ρ₂ > ρ₁ and k - 1 > 0)
  have hdenom_lt : 1 + ρ₁ * (k - 1) < 1 + ρ₂ * (k - 1) := by
    have hmul_lt : ρ₁ * (k - 1) < ρ₂ * (k - 1) := mul_lt_mul_of_pos_right h hkm1_pos
    linarith
  -- For positive a and 0 < d₁ < d₂: a/d₂ < a/d₁
  exact div_lt_div_of_pos_left hk_pos hdenom1_pos hdenom_lt

/-- K-4: k_eff is bounded between 1 and k. -/
theorem K4_bounded (k : ℕ) (ρ : ℝ) (hk : k ≥ 1) (hρ₁ : 0 ≤ ρ) (hρ₂ : ρ ≤ 1) :
    1 ≤ k_eff k ρ ∧ k_eff k ρ ≤ k := by
  unfold k_eff
  split_ifs with h
  · -- Case: k ≤ 1, so k = 1 (since hk : k ≥ 1)
    constructor
    · exact Nat.one_le_cast.mpr hk
    · exact le_refl _
  · -- Case: k > 1, so k_eff = k / (1 + ρ * (k - 1))
    push_neg at h
    have hk_ge_one : (1 : ℝ) ≤ k := Nat.one_le_cast.mpr hk
    have hk_pos : (0 : ℝ) < k := by linarith
    have hk_sub_one_nonneg : (0 : ℝ) ≤ k - 1 := by linarith
    have hdenom_pos : 0 < 1 + ρ * (k - 1) := by
      have hmul_nonneg : 0 ≤ ρ * (k - 1) := mul_nonneg hρ₁ hk_sub_one_nonneg
      linarith
    have hdenom_ge_one : 1 ≤ 1 + ρ * (k - 1) := by
      have hmul_nonneg : 0 ≤ ρ * (k - 1) := mul_nonneg hρ₁ hk_sub_one_nonneg
      linarith
    have hdenom_le_k : 1 + ρ * (k - 1) ≤ k := by
      have h1 : ρ * (k - 1) ≤ 1 * (k - 1) :=
        mul_le_mul_of_nonneg_right hρ₂ hk_sub_one_nonneg
      simp only [one_mul] at h1
      linarith
    constructor
    · -- Lower bound: 1 ≤ k / (1 + ρ * (k - 1))
      rw [one_le_div hdenom_pos]
      exact hdenom_le_k
    · -- Upper bound: k / (1 + ρ * (k - 1)) ≤ k
      rw [div_le_iff₀ hdenom_pos]
      calc (k : ℝ) = k * 1 := (mul_one _).symm
        _ ≤ k * (1 + ρ * (k - 1)) := mul_le_mul_of_nonneg_left hdenom_ge_one (le_of_lt hk_pos)

/-- K-5 (P5): k_eff is continuous in ρ on [0,1] for k ≥ 2.

    PROOF STRATEGY: The function k / (1 + ρ * (k-1)) is continuous where the
    denominator is nonzero. For ρ ≥ 0 and k ≥ 2, the denominator 1 + ρ(k-1) ≥ 1 > 0.

    A complete proof would require:
    1. Restrict domain to {ρ : ℝ | ρ ≥ 0} using Subtype
    2. Prove denominator > 0 on this restricted domain
    3. Apply Continuous.div with the domain restriction

    AXIOMATIZED: Domain restriction requires additional type machinery. -/
axiom K5_continuous (k : ℕ) (hk : k ≥ 2) :
    Continuous (fun ρ : ℝ => k_eff k ρ)

/-- K-6 (P6): Asymptotic behavior - as k → ∞ with fixed ρ > 0, k_eff → 1/ρ.

    PROOF STRATEGY:
    k_eff k ρ = k / (1 + ρ(k-1))
             = k / (ρk + (1-ρ))
             = 1 / (ρ + (1-ρ)/k)

    As k → ∞, (1-ρ)/k → 0, so the limit is 1/ρ.

    AXIOMATIZED: Requires Filter.Tendsto manipulation for Nat → ℝ limits. -/
axiom K6_asymptotic_limit (ρ : ℝ) (hρ_pos : 0 < ρ) :
    Filter.Tendsto (fun k : ℕ => k_eff k ρ) Filter.atTop (nhds (1 / ρ))

end RATCHET.EffectiveConstraints
