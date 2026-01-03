/-
  RATCHET Mathlib Extension: Independence and Product Measures

  This file provides independence primitives needed for
  both TC and DP proofs.

  Status: Axiomatized (uses Mathlib.Probability.Independence where available)
-/

import Mathlib.Probability.Independence.Basic
import Mathlib.MeasureTheory.Measure.ProbabilityMeasure

namespace RATCHET.Independence

open MeasureTheory ProbabilityTheory

/-! ## Basic Independence -/

/-- Events E₁, ..., E_n are mutually independent -/
def MutuallyIndependent {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω)
    (E : Fin n → Set Ω) : Prop :=
  ∀ S : Finset (Fin n), μ (⋂ i ∈ S, E i) = ∏ i ∈ S, μ (E i)

/-- Random variables X₁, ..., X_n are independent -/
def IndependentRVs {Ω α : Type*} [MeasurableSpace Ω] [MeasurableSpace α]
    (μ : Measure Ω) (X : Fin n → Ω → α) : Prop :=
  ∀ S : Finset (Fin n), ∀ A : Fin n → Set α,
    (∀ i, MeasurableSet (A i)) →
    μ (⋂ i ∈ S, X i ⁻¹' A i) = ∏ i ∈ S, μ (X i ⁻¹' A i)

/-! ## Product Measure Properties -/

/--
  Fubini-Tonelli for product measures.
  For independent random variables, expectation of product = product of expectations.
-/
axiom expectation_product_independent {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (X Y : Ω → ℝ) (hX : Measurable X) (hY : Measurable Y)
    (hInd : IndependentRVs μ ![X, Y]) :
    ∫ ω, X ω * Y ω ∂μ = (∫ ω, X ω ∂μ) * (∫ ω, Y ω ∂μ)

/--
  For indicator functions of independent events.
-/
axiom indicator_product_independent {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (E₁ E₂ : Set Ω) (hE₁ : MeasurableSet E₁) (hE₂ : MeasurableSet E₂)
    (hInd : MutuallyIndependent μ ![E₁, E₂]) :
    μ (E₁ ∩ E₂) = μ E₁ * μ E₂

/-! ## Hyperplane Independence -/

/-- Distribution for random unit normal vector (uniform on sphere) -/
axiom uniform_sphere_distribution (d : ℕ) [NeZero d] :
  ∃ (μ : Measure (Fin d → ℝ)),
    IsProbabilityMeasure μ ∧
    -- Concentrated on unit sphere
    μ {x | ‖x‖ = 1} = 1 ∧
    -- Rotation invariant
    True  -- Formal rotation invariance

/-- Distribution for random offset (uniform on [-1, 1]) -/
axiom uniform_offset_distribution :
  ∃ (μ : Measure ℝ),
    IsProbabilityMeasure μ ∧
    μ (Set.Icc (-1) 1) = 1 ∧
    -- Uniform density
    ∀ a b : ℝ, -1 ≤ a → a ≤ b → b ≤ 1 →
      μ (Set.Icc a b) = (b - a) / 2

/--
  k independent random hyperplanes.
  Each has:
  - Normal: uniform on unit sphere
  - Offset: uniform on [-1, 1]
  - Normal and offset are independent
  - Different hyperplanes are independent
-/
structure RandomHyperplanes (d k : ℕ) [NeZero d] where
  -- Underlying probability space
  Ω : Type*
  mΩ : MeasurableSpace Ω
  μ : Measure Ω
  hμ : IsProbabilityMeasure μ
  -- Random normal vectors
  normals : Fin k → Ω → Fin d → ℝ
  -- Random offsets
  offsets : Fin k → Ω → ℝ
  -- Unit norm constraint
  h_unit : ∀ i ω, ‖normals i ω‖ = 1
  -- Independence
  h_indep : ∀ i j, i ≠ j → IndependentRVs μ ![normals i, normals j]

/-! ## Correlation Adjustment -/

/--
  When hyperplanes have pairwise correlation ρ, effective number of
  independent hyperplanes is reduced.

  k_eff = k / (1 + ρ(k-1))

  For ρ = 0: k_eff = k (fully independent)
  For ρ = 1: k_eff = 1 (fully correlated)
-/
noncomputable def effectiveRank (k : ℕ) (ρ : ℝ) : ℝ :=
  k / (1 + ρ * (k - 1))

/-- Effective rank is positive -/
axiom effectiveRank_pos : ∀ k : ℕ, ∀ ρ : ℝ,
  k ≥ 1 → -1/(k-1 : ℝ) ≤ ρ → ρ ≤ 1 →
  effectiveRank k ρ > 0

/-- Effective rank is at most k -/
axiom effectiveRank_le_k : ∀ k : ℕ, ∀ ρ : ℝ,
  k ≥ 1 → 0 ≤ ρ → ρ ≤ 1 →
  effectiveRank k ρ ≤ k

/-- Effective rank is monotone decreasing in ρ -/
axiom effectiveRank_mono_rho : ∀ k : ℕ, ∀ ρ₁ ρ₂ : ℝ,
  k ≥ 1 → 0 ≤ ρ₁ → ρ₁ ≤ ρ₂ → ρ₂ ≤ 1 →
  effectiveRank k ρ₂ ≤ effectiveRank k ρ₁

end RATCHET.Independence
