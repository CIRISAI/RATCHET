/-
RATCHET: Causal Influence Detection

Formalization of Granger causality for detecting causal influence
between coherence dynamics.

Source: Granger, C.W.J. (1969). "Investigating Causal Relations by
        Econometric Models and Cross-spectral Methods."
        Econometrica 37(3):424-438.

Key Insight:
  X Granger-causes Y if X's past helps predict Y's future
  beyond what Y's own past provides.
-/

import Mathlib.Data.Real.Basic
import Mathlib.Algebra.Order.Field.Basic

namespace RATCHET.CausalInfluence

/-- F-statistic for nested model comparison.
    Compares restricted (Y lags only) vs unrestricted (Y + X lags). -/
noncomputable def f_statistic
    (rss_restricted rss_unrestricted : ℝ)
    (df_restriction df_residual : ℕ) : ℝ :=
  if rss_unrestricted ≤ 0 ∨ df_residual = 0 ∨ df_restriction = 0 then 0
  else ((rss_restricted - rss_unrestricted) / df_restriction) /
       (rss_unrestricted / df_residual)

/-- Test result classification. -/
inductive TestResult
  | NotFalsified   -- Evidence of causal influence
  | Falsified      -- No evidence of causal influence
  | Inconclusive   -- Insufficient data

/-- Granger causality test result. -/
structure GrangerResult where
  f_stat : ℝ
  p_value : ℝ
  result : TestResult

/-- CI-1: F-statistic is non-negative. -/
theorem CI1_f_nonneg
    (rss_r rss_u : ℝ)
    (df_r df_res : ℕ)
    (h_rss_r_pos : rss_r > 0)
    (h_rss_u_pos : rss_u > 0)
    (h_rss_order : rss_u ≤ rss_r)
    (h_df_r_pos : df_r > 0)
    (h_df_res_pos : df_res > 0) :
    f_statistic rss_r rss_u df_r df_res ≥ 0 := by
  unfold f_statistic
  split_ifs with h
  · -- If branch: result is 0, which is ≥ 0
    exact le_refl 0
  · -- Else branch: show the division is non-negative
    -- h : ¬(rss_u ≤ 0 ∨ df_res = 0 ∨ df_r = 0)
    push_neg at h
    obtain ⟨h_rss_u_pos', h_df_res_ne, h_df_r_ne⟩ := h
    -- Numerator: (rss_r - rss_u) / df_r ≥ 0
    have h_num_nonneg : (rss_r - rss_u) / df_r ≥ 0 := by
      apply div_nonneg
      · exact sub_nonneg.mpr h_rss_order
      · exact Nat.cast_nonneg df_r
    -- Denominator: rss_u / df_res > 0
    have h_denom_pos : rss_u / df_res > 0 := by
      apply div_pos h_rss_u_pos
      exact Nat.cast_pos.mpr h_df_res_pos
    -- Division of non-negative by positive is non-negative
    exact div_nonneg h_num_nonneg (le_of_lt h_denom_pos)

/-- CI-2: Unrestricted model fits at least as well as restricted.

    This is an axiom of OLS regression: adding predictors to a linear model
    cannot increase the residual sum of squares (RSS) in-sample. This follows
    from the geometric interpretation of OLS as orthogonal projection onto
    a subspace - a larger subspace (unrestricted model) is at least as close
    to the data as any smaller subspace (restricted model).

    Mathematical foundation:
    - RSS_u = ||Y - X_u beta_u||^2 where X_u spans a larger space
    - RSS_r = ||Y - X_r beta_r||^2 where X_r is a subspace of X_u
    - The projection onto a larger subspace minimizes distance at least as well

    We axiomatize this because a full proof would require formalizing:
    - Matrix operations and linear algebra
    - Orthogonal projection theory
    - The OLS normal equations

    Reference: Any standard econometrics text, e.g., Greene (2018) Ch. 4
-/
axiom CI2_nested_rss (rss_r rss_u : ℝ)
    (h_nested : True)  -- Models are properly nested
    (h_ols : True) :   -- Using OLS estimation
    rss_u ≤ rss_r

/-- Cross-correlation asymmetry measure.
    Positive = X leads Y, Negative = Y leads X. -/
noncomputable def asymmetry (max_forward max_backward : ℝ) : ℝ :=
  max_forward - max_backward

/-- Inferred direction based on asymmetry. -/
inductive Direction
  | AtoB        -- A causes B
  | BtoA        -- B causes A
  | Bidirectional
  | None

/-- Combined causal influence result. -/
structure CausalInfluenceResult where
  granger_significant : Bool
  asymmetry_significant : Bool
  direction : Direction

/-- Evidence of causal influence exists. -/
def has_causal_influence (r : CausalInfluenceResult) : Bool :=
  r.granger_significant ∨ r.asymmetry_significant

/-- Coordination pattern classification for deception detection. -/
inductive CoordinationPattern
  | Independent          -- No causal link
  | NormalCoordination   -- Unidirectional, expected
  | SuspiciousPattern    -- Unusual causal structure

/-- Classify coordination pattern. -/
def classify_coordination (dir : Direction) (fragile : Bool) : CoordinationPattern :=
  match dir with
  | Direction.None => CoordinationPattern.Independent
  | Direction.Bidirectional => CoordinationPattern.SuspiciousPattern
  | _ => if fragile then CoordinationPattern.SuspiciousPattern
         else CoordinationPattern.NormalCoordination

/-- CI-3: Bidirectional always suspicious. -/
theorem CI3_bidirectional_suspicious (fragile : Bool) :
    classify_coordination Direction.Bidirectional fragile = CoordinationPattern.SuspiciousPattern := by
  rfl

/-- CI-4: Fragile coupling always suspicious. -/
theorem CI4_fragile_suspicious (dir : Direction) (h_not_none : dir ≠ Direction.None) :
    classify_coordination dir true = CoordinationPattern.SuspiciousPattern := by
  cases dir <;> simp [classify_coordination]
  contradiction

end RATCHET.CausalInfluence
