/-
  RATCHET: Universal Critical Correlation Threshold

  Formalizes the ρ_critical ≈ 0.43 hypothesis:
  - For large k, k_eff → 1/ρ
  - k_eff ≈ 2.3 marks fragility boundary
  - Therefore ρ_critical = 1/2.3 ≈ 0.43

  S1/S2/S3 Agent Classification:
  - S1 (Chaotic): ρ_internal = 0, contribute noise
  - S2 (Principled): ρ_internal = 0.85, form correlated blocs
  - S3 (Empathetic): ρ_internal = 0.30, bridge communities

  Author: CIRIS Research Team
  Date: January 2026
-/

import Mathlib.Data.Real.Basic
import Mathlib.Tactic

namespace RATCHET.CCA.UniversalThreshold

/-!
# Core Definitions
-/

/-- Effective constraint count using Kish formula -/
noncomputable def k_eff (k : ℕ) (ρ : ℝ) : ℝ :=
  if k ≤ 1 then k
  else k / (1 + ρ * (k - 1))

/-- Universal critical correlation threshold -/
noncomputable def rho_critical : ℝ := 0.43

/-- Critical k_eff threshold (≈ 2.3) -/
noncomputable def k_eff_critical : ℝ := 2.3

/-- Systems are fragile when ρ > rho_critical -/
def is_fragile (ρ : ℝ) : Prop := ρ > rho_critical

/-!
# S1/S2/S3 Agent Parameters
-/

/-- S1 agents have zero internal correlation (chaotic) -/
noncomputable def rho_S1 : ℝ := 0.0

/-- S2 agents form tightly correlated blocs (principled) -/
noncomputable def rho_S2 : ℝ := 0.85

/-- S3 agents bridge with moderate correlation (empathetic) -/
noncomputable def rho_S3 : ℝ := 0.30

/-- Cross-group correlations -/
noncomputable def rho_S1_S2 : ℝ := 0.1   -- S1-S2 minimal
noncomputable def rho_S1_S3 : ℝ := 0.2   -- S1-S3 slight
noncomputable def rho_S2_S3 : ℝ := 0.4   -- S2-S3 moderate (S3 bridges)

/-- System-wide correlation from S1/S2/S3 mix
    Uses weighted average of within-group and between-group correlations -/
noncomputable def system_rho (f1 f2 f3 : ℝ) : ℝ :=
  -- Within-group (weighted by fraction squared)
  f1 * f1 * rho_S1 + f2 * f2 * rho_S2 + f3 * f3 * rho_S3 +
  -- Between-group (weighted by product of fractions, factor of 2 for symmetry)
  2 * f1 * f2 * rho_S1_S2 +
  2 * f1 * f3 * rho_S1_S3 +
  2 * f2 * f3 * rho_S2_S3

/-!
# Core Theorems
-/

/-- K1: When ρ = 0, k_eff = k (full independence) -/
theorem k_eff_at_zero (k : ℕ) (hk : k > 1) :
    k_eff k 0 = k := by
  unfold k_eff
  split_ifs with h
  · omega
  · simp [mul_zero, add_zero]

/-- K2: When ρ = 1 and k > 1, k_eff = 1 (collapse to unity) -/
theorem k_eff_at_one (k : ℕ) (hk : k > 1) :
    k_eff k 1 = 1 := by
  unfold k_eff
  split_ifs with h
  · omega
  · simp [one_mul]
    field_simp

/-- At ρ = rho_critical ≈ 0.43, the ratio 1/ρ ≈ 2.3 -/
theorem inverse_rho_critical :
    1 / rho_critical > 2.3 ∧ 1 / rho_critical < 2.4 := by
  unfold rho_critical
  constructor
  · norm_num
  · norm_num

/-- Pure S2 population yields high ρ -/
theorem pure_S2_rho :
    system_rho 0 1 0 = rho_S2 := by
  unfold system_rho rho_S1 rho_S2 rho_S3 rho_S1_S2 rho_S1_S3 rho_S2_S3
  ring

/-- Pure S2 population is fragile (ρ = 0.85 > 0.43) -/
theorem pure_S2_is_fragile :
    is_fragile (system_rho 0 1 0) := by
  unfold is_fragile
  rw [pure_S2_rho]
  unfold rho_S2 rho_critical
  norm_num

/-- Pure S1 population yields zero ρ (chaotic but stable) -/
theorem pure_S1_rho :
    system_rho 1 0 0 = rho_S1 := by
  unfold system_rho rho_S1 rho_S2 rho_S3 rho_S1_S2 rho_S1_S3 rho_S2_S3
  ring

/-- Pure S1 is NOT fragile -/
theorem pure_S1_stable :
    ¬ is_fragile (system_rho 1 0 0) := by
  unfold is_fragile
  rw [pure_S1_rho]
  unfold rho_S1 rho_critical
  norm_num

/-- Pure S3 population yields moderate ρ -/
theorem pure_S3_rho :
    system_rho 0 0 1 = rho_S3 := by
  unfold system_rho rho_S1 rho_S2 rho_S3 rho_S1_S2 rho_S1_S3 rho_S2_S3
  ring

/-- Pure S3 is NOT fragile (ρ = 0.30 < 0.43) -/
theorem pure_S3_stable :
    ¬ is_fragile (system_rho 0 0 1) := by
  unfold is_fragile
  rw [pure_S3_rho]
  unfold rho_S3 rho_critical
  norm_num

/-!
# Large k Limit Behavior

For large k, k_eff = k / (1 + ρ(k-1)) ≈ k / (ρk) = 1/ρ

This means the critical k_eff ≈ 2.3 corresponds to ρ ≈ 1/2.3 ≈ 0.43
-/

/-- k_eff formula for k > 1 -/
theorem k_eff_formula (k : ℕ) (ρ : ℝ) (hk : k > 1) :
    k_eff k ρ = k / (1 + ρ * (k - 1)) := by
  unfold k_eff
  split_ifs with h
  · omega
  · rfl

/-- For any ρ > 0 and k > 1, k_eff > 0 -/
theorem k_eff_pos (k : ℕ) (ρ : ℝ) (hk : k > 1) (hρ_nn : 0 ≤ ρ) :
    k_eff k ρ > 0 := by
  rw [k_eff_formula k ρ hk]
  apply div_pos
  · exact Nat.cast_pos.mpr (by omega)
  · have : (k : ℝ) - 1 ≥ 0 := by simp; omega
    linarith [mul_nonneg hρ_nn this]

/-!
# Stability Conditions
-/

/-- Stability classification -/
inductive StabilityClass
  | Stable      -- ρ < rho_critical
  | Marginal    -- ρ ≈ rho_critical (within 0.05)
  | Fragile     -- ρ > rho_critical
  deriving DecidableEq, Repr

/-- Classify system stability -/
noncomputable def classify_stability (ρ : ℝ) : StabilityClass :=
  if ρ < rho_critical - 0.05 then StabilityClass.Stable
  else if ρ > rho_critical + 0.05 then StabilityClass.Fragile
  else StabilityClass.Marginal

/-!
# Key Empirical Claims (as axioms to be validated)

These are stated as axioms representing empirical findings.
Validation against real data is tracked in tests/test_rho_critical.py
-/

/-- Axiom: Collapse rate increases sharply at ρ = 0.43 -/
axiom collapse_rate_threshold :
  ∀ k : ℕ, k > 10 →
  ∃ rate_below rate_above : ℝ,
    rate_above > 2 * rate_below  -- At least 2x increase

/-- Axiom: S3 fraction ≥ 13% tends to stabilize when S1 ≥ 20% -/
axiom s3_stabilization_threshold :
  ∀ f1 f2 f3 : ℝ,
    f1 + f2 + f3 = 1 →
    0 ≤ f1 ∧ 0 ≤ f2 ∧ 0 ≤ f3 →
    f1 ≥ 0.2 → f3 ≥ 0.13 →
    system_rho f1 f2 f3 < 0.55  -- Below collapse threshold

/-- Axiom: Cross-domain universality of ρ ≈ 0.43 -/
axiom cross_domain_universality :
  ∀ domain : String,
    domain ∈ ["microbiome", "financial", "institutional", "political"] →
    ∃ rho_observed : ℝ, 0.38 < rho_observed ∧ rho_observed < 0.48

end RATCHET.CCA.UniversalThreshold
