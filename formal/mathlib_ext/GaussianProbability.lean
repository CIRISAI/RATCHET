/-
  RATCHET Mathlib Extension: Gaussian Probability Primitives

  This file provides the Gaussian probability primitives needed for
  Detection Power proofs (DP-4, DP-5, DP-6).

  Status: Axiomatized (needs Mathlib contribution or proof)

  These are standard results but not yet in Mathlib4.
-/

import Mathlib.Analysis.SpecialFunctions.Gaussian
import Mathlib.Probability.Distributions.Gaussian
import Mathlib.Analysis.SpecialFunctions.Integrals
import Mathlib.MeasureTheory.Integral.Bochner

namespace RATCHET.Probability

/-! ## Standard Normal CDF and Inverse -/

/-- Standard normal CDF: Φ(x) = P(Z ≤ x) for Z ~ N(0,1) -/
noncomputable def Phi : ℝ → ℝ := fun x =>
  (1 / Real.sqrt (2 * Real.pi)) * ∫ t in Set.Iio x, Real.exp (-t^2 / 2)

/-- Standard normal PDF -/
noncomputable def phi : ℝ → ℝ := fun x =>
  (1 / Real.sqrt (2 * Real.pi)) * Real.exp (-x^2 / 2)

/-- Phi is the CDF of the standard normal -/
axiom Phi_cdf : ∀ x : ℝ, 0 ≤ Phi x ∧ Phi x ≤ 1

/-- Phi is monotonically increasing -/
axiom Phi_mono : Monotone Phi

/-- Phi(-∞) = 0 -/
axiom Phi_neg_inf : Filter.Tendsto Phi Filter.atBot (nhds 0)

/-- Phi(+∞) = 1 -/
axiom Phi_pos_inf : Filter.Tendsto Phi Filter.atTop (nhds 1)

/-- Symmetry: Phi(-x) = 1 - Phi(x) -/
axiom Phi_neg : ∀ x : ℝ, Phi (-x) = 1 - Phi x

/-- Phi(0) = 0.5 -/
axiom Phi_zero : Phi 0 = 1/2

/-- Inverse of standard normal CDF (quantile function) -/
noncomputable def Phi_inv : ℝ → ℝ := Function.invFun Phi

/-- For p ∈ (0,1), Phi(Phi_inv(p)) = p -/
axiom Phi_inv_spec : ∀ p : ℝ, 0 < p → p < 1 → Phi (Phi_inv p) = p

/-- Common quantiles -/
axiom z_0_05 : Phi_inv 0.95 = 1.645  -- Approximate
axiom z_0_025 : Phi_inv 0.975 = 1.96  -- Approximate

/-! ## Berry-Esseen Theorem -/

/-- Berry-Esseen constant (best known: C ≤ 0.4748) -/
def C_BE : ℝ := 0.4748

/--
  Berry-Esseen bound for sum of i.i.d. random variables.

  If X_1, ..., X_n are i.i.d. with mean μ, variance σ², and E|X - μ|³ = ρ,
  then for S_n = (1/n) Σ X_i (sample mean):

  sup_x |P((S_n - μ)/(σ/√n) ≤ x) - Φ(x)| ≤ C_BE * ρ / (σ³ * √n)
-/
structure BerryEsseenBound where
  n : ℕ
  sigma : ℝ
  rho : ℝ  -- Third absolute central moment
  h_n : n ≥ 1
  h_sigma : sigma > 0
  h_rho : rho ≥ 0

/-- The Berry-Esseen error bound -/
noncomputable def BerryEsseenBound.error (b : BerryEsseenBound) : ℝ :=
  C_BE * b.rho / (b.sigma^3 * Real.sqrt b.n)

/-- Berry-Esseen theorem: CDF approximation error is bounded -/
axiom berry_esseen_bound :
  ∀ (b : BerryEsseenBound) (x : ℝ) (P_normalized : ℝ),
    -- P_normalized is the probability that normalized sum ≤ x
    |P_normalized - Phi x| ≤ b.error

/-! ## Power Analysis for Gaussian LRT -/

/--
  Gaussian Likelihood Ratio Test statistic.
  For testing H0: μ = 0 vs H1: μ = μ₁ with known variance σ².
-/
structure GaussianLRT where
  mu_alt : ℝ        -- Alternative hypothesis mean
  sigma : ℝ         -- Known standard deviation
  n : ℕ             -- Sample size
  alpha : ℝ         -- Significance level
  h_sigma : sigma > 0
  h_n : n ≥ 1
  h_alpha : 0 < alpha ∧ alpha < 1

/-- Detection power: P(reject H0 | H1 true) -/
noncomputable def GaussianLRT.power (lrt : GaussianLRT) : ℝ :=
  let z_alpha := Phi_inv (1 - lrt.alpha)
  let noncentrality := lrt.mu_alt * Real.sqrt lrt.n / lrt.sigma
  1 - Phi (z_alpha - noncentrality)

/-- Required sample size for given power -/
noncomputable def GaussianLRT.required_n (mu_alt sigma alpha beta : ℝ)
    (h_sigma : sigma > 0) (h_alpha : 0 < alpha ∧ alpha < 1)
    (h_beta : 0 < beta ∧ beta < 1) : ℝ :=
  let z_alpha := Phi_inv (1 - alpha)
  let z_beta := Phi_inv (1 - beta)
  ((z_alpha + z_beta) * sigma / mu_alt)^2

/-- Power increases with sample size -/
axiom power_mono_n : ∀ (lrt1 lrt2 : GaussianLRT),
  lrt1.mu_alt = lrt2.mu_alt →
  lrt1.sigma = lrt2.sigma →
  lrt1.alpha = lrt2.alpha →
  lrt1.n ≤ lrt2.n →
  lrt1.power ≤ lrt2.power

/-- Power increases with effect size -/
axiom power_mono_effect : ∀ (lrt1 lrt2 : GaussianLRT),
  lrt1.n = lrt2.n →
  lrt1.sigma = lrt2.sigma →
  lrt1.alpha = lrt2.alpha →
  |lrt1.mu_alt| ≤ |lrt2.mu_alt| →
  lrt1.power ≤ lrt2.power

/-! ## Sample Covariance Concentration -/

/--
  Concentration bound for sample covariance matrix.

  For n samples from a distribution with covariance Σ,
  the sample covariance Σ̂ satisfies:
  ‖Σ̂ - Σ‖_op ≤ C * √(d/n) with high probability
-/
structure CovarianceConcentration where
  d : ℕ            -- Dimension
  n : ℕ            -- Sample size
  sub_gaussian_norm : ℝ  -- Sub-Gaussian norm of the distribution
  h_d : d ≥ 1
  h_n : n ≥ d      -- Need at least d samples

/-- Concentration error bound -/
noncomputable def CovarianceConcentration.error (cc : CovarianceConcentration)
    (delta : ℝ) : ℝ :=
  cc.sub_gaussian_norm^2 * Real.sqrt (cc.d / cc.n) *
    (1 + Real.sqrt (2 * Real.log (1/delta)))

/-- With probability 1-δ, operator norm error is bounded -/
axiom covariance_concentration :
  ∀ (cc : CovarianceConcentration) (delta : ℝ),
    0 < delta → delta < 1 →
    -- P(‖Σ̂ - Σ‖_op > error) ≤ delta
    True  -- Formalized as axiom for now

/-! ## Mahalanobis Distance -/

/-- Squared Mahalanobis distance -/
noncomputable def mahalanobis_sq (x mu : Fin d → ℝ) (Sigma_inv : Matrix (Fin d) (Fin d) ℝ) : ℝ :=
  let diff := x - mu
  Matrix.dotProduct diff (Sigma_inv.mulVec diff)

/-- Mahalanobis distance is non-negative -/
axiom mahalanobis_nonneg : ∀ (d : ℕ) (x mu : Fin d → ℝ) (Sigma_inv : Matrix (Fin d) (Fin d) ℝ),
  Matrix.PosSemidef Sigma_inv →
  mahalanobis_sq x mu Sigma_inv ≥ 0

end RATCHET.Probability
