/-
RATCHET: Faculty Composition and Veto Logic

Axiomatic exploration of how multiple faculties compose.
We define structures and see what theorems emerge - not forcing conclusions.

Research questions:
1. What properties does veto composition have?
2. How do thresholds interact?
3. What guarantees can we derive from the architecture?
-/

import Mathlib.Data.Real.Basic
import Mathlib.Algebra.Order.Field.Basic
import Mathlib.Data.Fin.Basic
import Mathlib.Logic.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum

namespace RATCHET.FacultyComposition

/-! ## Faculty Result Types -/

/-- A faculty evaluation result. -/
inductive FacultyResult
  | Permit      -- Faculty allows the action
  | Veto        -- Faculty blocks the action
  | Defer       -- Faculty requests human review
  | Abstain     -- Faculty has no opinion (insufficient data)
  deriving DecidableEq, Repr

/-- Faculty category in the 10-faculty architecture. -/
inductive FacultyCategory
  | Conscience  -- Ethical evaluation (4 faculties)
  | Intuition   -- Coherence sensing (4 faculties)
  | Executive   -- Bypass control (2 faculties)
  deriving DecidableEq, Repr

/-- A faculty with its category and evaluation function. -/
structure Faculty (α : Type*) where
  name : String
  category : FacultyCategory
  evaluate : α → FacultyResult

/-! ## Veto Composition Logic -/

/-- Core veto rule: ANY veto blocks the action. -/
def composeResults (results : List FacultyResult) : FacultyResult :=
  if results.any (· == FacultyResult.Veto) then FacultyResult.Veto
  else if results.any (· == FacultyResult.Defer) then FacultyResult.Defer
  else if results.all (· == FacultyResult.Abstain) then FacultyResult.Abstain
  else FacultyResult.Permit

/-- Evaluate all faculties and compose results. -/
def evaluateAll {α : Type*} (faculties : List (Faculty α)) (input : α) : FacultyResult :=
  composeResults (faculties.map (·.evaluate input))

/-! ## Theorems That Emerge From Definitions -/

/-- FC-1: A single veto is sufficient to block. -/
theorem FC1_single_veto_blocks (results : List FacultyResult)
    (h : FacultyResult.Veto ∈ results) :
    composeResults results = FacultyResult.Veto := by
  unfold composeResults
  have : results.any (· == FacultyResult.Veto) = true := by
    simp only [List.any_eq_true, beq_iff_eq]
    exact ⟨FacultyResult.Veto, h, rfl⟩
  simp [this]

/-- FC-2: Permit requires no vetoes. -/
theorem FC2_permit_requires_no_veto (results : List FacultyResult)
    (h : composeResults results = FacultyResult.Permit) :
    ¬(FacultyResult.Veto ∈ results) := by
  intro hveto
  have := FC1_single_veto_blocks results hveto
  rw [this] at h
  contradiction

/-- FC-3: Veto is monotonic - adding a veto to any list produces veto. -/
theorem FC3_veto_monotonic (results : List FacultyResult) :
    composeResults (FacultyResult.Veto :: results) = FacultyResult.Veto := by
  apply FC1_single_veto_blocks
  exact List.mem_cons_self _ _

/-- FC-4: Order independence - veto anywhere in list has same effect. -/
theorem FC4_veto_order_independent (before after : List FacultyResult) :
    composeResults (before ++ [FacultyResult.Veto] ++ after) = FacultyResult.Veto := by
  apply FC1_single_veto_blocks
  simp

/-! ## Threshold Structures -/

/-- Threshold levels: Conservative ⊂ Standard ⊂ Permissive -/
structure ThresholdLevels where
  conservative : ℝ
  standard : ℝ
  permissive : ℝ

/-- Well-formed threshold levels for veto-above (e.g., CCE Risk).
    Conservative is strictest (lowest), permissive is loosest (highest). -/
def ThresholdLevels.wellFormedAbove (t : ThresholdLevels) : Prop :=
  t.conservative ≤ t.standard ∧ t.standard ≤ t.permissive

/-- Well-formed threshold levels for veto-below (e.g., k_eff).
    Conservative is strictest (highest), permissive is loosest (lowest). -/
def ThresholdLevels.wellFormedBelow (t : ThresholdLevels) : Prop :=
  t.conservative ≥ t.standard ∧ t.standard ≥ t.permissive

/-- Example: CCE Risk thresholds (veto if above). -/
noncomputable def cceRiskThresholds : ThresholdLevels :=
  { conservative := 0.6
    standard := 0.8
    permissive := 0.9 }

/-- Example: k_eff thresholds (veto if below). -/
noncomputable def keffThresholds : ThresholdLevels :=
  { conservative := 2.0
    standard := 1.5
    permissive := 1.2 }

/-! ## Threshold Nesting Theorems -/

/-- TN-1: CCE Risk thresholds are well-formed. -/
theorem TN1_cce_wellformed : cceRiskThresholds.wellFormedAbove := by
  unfold ThresholdLevels.wellFormedAbove cceRiskThresholds
  constructor <;> norm_num

/-- TN-2: k_eff thresholds are well-formed. -/
theorem TN2_keff_wellformed : keffThresholds.wellFormedBelow := by
  unfold ThresholdLevels.wellFormedBelow keffThresholds
  constructor <;> norm_num

/-- TN-3: Safe at conservative level (for veto-above) implies safe everywhere.
    This is the key nesting property. -/
theorem TN3_nesting_above (t : ThresholdLevels) (value : ℝ)
    (hwf : t.wellFormedAbove) (h_safe : value ≤ t.conservative) :
    value ≤ t.permissive := by
  unfold ThresholdLevels.wellFormedAbove at hwf
  obtain ⟨h1, h2⟩ := hwf
  calc value ≤ t.conservative := h_safe
    _ ≤ t.standard := h1
    _ ≤ t.permissive := h2

/-- TN-4: Safe at conservative level (for veto-below) implies safe everywhere. -/
theorem TN4_nesting_below (t : ThresholdLevels) (value : ℝ)
    (hwf : t.wellFormedBelow) (h_safe : value ≥ t.conservative) :
    value ≥ t.permissive := by
  unfold ThresholdLevels.wellFormedBelow at hwf
  obtain ⟨h1, h2⟩ := hwf
  calc value ≥ t.conservative := h_safe
    _ ≥ t.standard := h1
    _ ≥ t.permissive := h2

/-! ## ES Proximity Threshold Derivation -/

/-- Gaussian kurtosis baseline. -/
noncomputable def kurtosis_gaussian : ℝ := 3.0

/-- ES phase classification based on deviation from Gaussian. -/
inductive ESPhase
  | Stable    -- Light tails, κ < κ_gaussian - δ_low
  | Critical  -- Near Gaussian
  | Fragile   -- Heavy tails, κ > κ_gaussian + δ_high
  deriving DecidableEq, Repr

/-- ES threshold parameters. Values from PNAS 2025:
    - δ_low = 0.5 (so stable if κ < 2.5)
    - δ_high = 1.0 (so fragile if κ > 4.0)

    Note: These are EMPIRICAL parameters, not derived from first principles.
    The math tells us what happens given these values, not why these values. -/
structure ESThresholds where
  delta_low : ℝ   -- deviation below Gaussian for "stable"
  delta_high : ℝ  -- deviation above Gaussian for "fragile"
  h_low_pos : delta_low > 0
  h_high_pos : delta_high > 0

/-- PNAS 2025 empirical thresholds. -/
noncomputable def pnas2025_thresholds : ESThresholds :=
  { delta_low := 0.5
    delta_high := 1.0
    h_low_pos := by norm_num
    h_high_pos := by norm_num }

/-- Classify ES phase given kurtosis and thresholds. -/
noncomputable def classifyES (κ : ℝ) (t : ESThresholds) : ESPhase :=
  if κ < kurtosis_gaussian - t.delta_low then ESPhase.Stable
  else if κ > kurtosis_gaussian + t.delta_high then ESPhase.Fragile
  else ESPhase.Critical

/-! ## What Can We Prove About ES Thresholds? -/

/-- ES-TH-1: The three phases cover all cases. -/
theorem ES_TH1_exhaustive (κ : ℝ) (t : ESThresholds) :
    classifyES κ t = ESPhase.Stable ∨
    classifyES κ t = ESPhase.Critical ∨
    classifyES κ t = ESPhase.Fragile := by
  unfold classifyES
  split_ifs <;> simp

/-- ES-TH-2: Stable implies κ below critical range. -/
theorem ES_TH2_stable_below (κ : ℝ) (t : ESThresholds)
    (h : classifyES κ t = ESPhase.Stable) :
    κ < kurtosis_gaussian - t.delta_low := by
  unfold classifyES at h
  split_ifs at h with h1 h2 <;> first | exact h1 | contradiction

/-- ES-TH-3: Fragile implies κ above critical range. -/
theorem ES_TH3_fragile_above (κ : ℝ) (t : ESThresholds)
    (h : classifyES κ t = ESPhase.Fragile) :
    κ > kurtosis_gaussian + t.delta_high := by
  unfold classifyES at h
  split_ifs at h with h1 h2 <;> first | exact h2 | contradiction

/-- ES-TH-4: Stable is downward-closed - if κ₂ is stable and κ₁ ≤ κ₂, then κ₁ is stable. -/
theorem ES_TH4_stable_downward_closed (κ₁ κ₂ : ℝ) (t : ESThresholds)
    (h_order : κ₁ ≤ κ₂)
    (h_stable : classifyES κ₂ t = ESPhase.Stable) :
    classifyES κ₁ t = ESPhase.Stable := by
  have h2 : κ₂ < kurtosis_gaussian - t.delta_low := ES_TH2_stable_below κ₂ t h_stable
  have h1 : κ₁ < kurtosis_gaussian - t.delta_low := lt_of_le_of_lt h_order h2
  unfold classifyES
  simp [h1]

/-- ES-TH-5: Fragile is upward-closed - if κ₁ is fragile and κ₂ ≥ κ₁, then κ₂ is fragile. -/
theorem ES_TH5_fragile_upward_closed (κ₁ κ₂ : ℝ) (t : ESThresholds)
    (h_order : κ₁ ≤ κ₂)
    (h_fragile : classifyES κ₁ t = ESPhase.Fragile) :
    classifyES κ₂ t = ESPhase.Fragile := by
  have h1 : κ₁ > kurtosis_gaussian + t.delta_high := ES_TH3_fragile_above κ₁ t h_fragile
  have h2 : κ₂ > kurtosis_gaussian + t.delta_high := lt_of_lt_of_le h1 h_order
  have h_not_stable : ¬(κ₂ < kurtosis_gaussian - t.delta_low) := by
    intro hcontra
    have : κ₁ < kurtosis_gaussian - t.delta_low := lt_of_le_of_lt h_order hcontra
    have h1' : κ₁ > kurtosis_gaussian + t.delta_high := h1
    have : kurtosis_gaussian + t.delta_high < kurtosis_gaussian - t.delta_low := lt_trans h1' this
    have : t.delta_high + t.delta_low < 0 := by linarith
    have : t.delta_high > 0 := t.h_high_pos
    have : t.delta_low > 0 := t.h_low_pos
    linarith
  unfold classifyES
  simp [h_not_stable, h2]

/-! ## Discovery: What Did We Learn?

The axiomatic exploration reveals:

1. **Veto composition is well-behaved**: monotonic, order-independent,
   single veto sufficient. These are good properties for a safety system.

2. **Threshold nesting works**: Conservative ⊂ Standard ⊂ Permissive
   gives proper safety margins. Safe at conservative → safe everywhere.

3. **ES phase classification has proper closure properties**:
   - Stable is downward-closed (lower κ stays stable)
   - Fragile is upward-closed (higher κ stays fragile)
   - Critical is the "middle zone"

4. **What we CANNOT prove from first principles**:
   - Why δ_low = 0.5 and δ_high = 1.0 specifically
   - That these thresholds correspond to actual collapse risk
   - That the PNAS 2025 empirical findings generalize

   These require empirical validation, not mathematical proof.

5. **The architecture has the right SHAPE for safety**:
   - Veto-any is conservative (fails safe)
   - Threshold nesting allows calibration
   - Phase monotonicity matches physical intuition
-/

end RATCHET.FacultyComposition
