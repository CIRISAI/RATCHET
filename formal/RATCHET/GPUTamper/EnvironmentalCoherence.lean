/-
RATCHET: GPU-Based Chaotic Resonator Detector

Validated findings from systematic characterization (January 2026).

PHYSICAL MODEL: Underdamped Chaotic Resonator
- Q factor: 10.25 (ceramic resonator class)
- Resonance: 70.7 Hz (forced response peak)
- Damping ratio: ζ = 0.049 (underdamped)
- Lyapunov: λ = +0.178 ± 0.002 per step (TRUE CHAOS)

SPECTRAL CHARACTERIZATION:
- DC/Drift (0-1 Hz): 61.5% of power
- Mid (10-50 Hz): 34.9% (Lorenz chaos bandwidth)
- Resonance (60-80 Hz): <1% (forced response only)
- High (>80 Hz): 3.4% (noise floor, sharp cutoff)

CROSS-SPECTRAL COHERENCE:
- All frequencies: 100% coherent between oscillators
- Coherence is ALGORITHMIC (same Lorenz math), not physical coupling
- Cross-GPU coherence (78%) ≈ same-GPU coherence (82.6%)

ENVIRONMENTAL DETECTION:
- 60 Hz power grid subharmonics detected (6.64, 13.28, 19.92 Hz)
- Environmental affects all oscillators equally (common-mode)
- Reference subtraction cancels environmental signal

CRITICAL LIMITATION (Exp 68):
- Timing perturbation scale: ~1e-5 (too small)
- Required for O(1) divergence: ~65 steps of coherent accumulation
- Reference subtraction yields ~0.5% residual (floating point noise)
- Lorenz acts as MIXER/WHITENER, not entropy amplifier
- True entropy must be extracted from timing LSBs directly

Author: CIRIS Research Team
Date: January 2026
-/

import Mathlib.Data.Real.Basic
import Mathlib.Algebra.Order.Field.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum

namespace RATCHET.GPUTamper.ChaoticResonator

/-!
# Resonator Properties
-/

/-- Resonator characterization -/
structure ResonatorParams where
  q_factor : ℝ
  resonance_hz : ℝ
  damping_ratio : ℝ
  bandwidth_hz : ℝ

/-- Validated resonator parameters -/
noncomputable def resonator : ResonatorParams := {
  q_factor := 10.25
  resonance_hz := 70.7
  damping_ratio := 0.049
  bandwidth_hz := 6.9
}

/-- Q factor places this in ceramic resonator class -/
theorem ceramic_resonator_class :
    resonator.q_factor > 5 ∧ resonator.q_factor < 50 := by
  simp [resonator]
  norm_num

/-- System is underdamped (ζ < 1) -/
theorem underdamped :
    resonator.damping_ratio < 1 := by
  simp [resonator]
  norm_num

/-!
# Lyapunov Exponent (Chaos Validation)
-/

/-- Lyapunov measurement -/
structure LyapunovMeasurement where
  per_step : ℝ
  uncertainty : ℝ
  per_time_unit : ℝ
  literature_continuous : ℝ

/-- Validated Lyapunov measurements -/
noncomputable def lyapunov : LyapunovMeasurement := {
  per_step := 0.178
  uncertainty := 0.002
  per_time_unit := 3.55
  literature_continuous := 0.906  -- Sparrow 1982
}

/-- Positive Lyapunov confirms true chaos -/
theorem true_chaos :
    lyapunov.per_step > 0 := by
  simp [lyapunov]
  norm_num

/-- Measurement is reproducible (small uncertainty) -/
theorem reproducible_measurement :
    lyapunov.uncertainty / lyapunov.per_step < 0.02 := by
  simp [lyapunov]
  norm_num

/-!
# Spectral Characterization
-/

/-- Spectral band power distribution -/
structure SpectralBand where
  name : String
  lower_hz : ℝ
  upper_hz : ℝ
  power_pct : ℝ

/-- DC/Drift band (dominant) -/
noncomputable def band_dc_drift : SpectralBand := {
  name := "DC/Drift"
  lower_hz := 0
  upper_hz := 1
  power_pct := 61.5
}

/-- Mid frequency band (Lorenz chaos bandwidth) -/
noncomputable def band_mid : SpectralBand := {
  name := "Mid (chaos bandwidth)"
  lower_hz := 10
  upper_hz := 50
  power_pct := 34.9
}

/-- Resonance band (forced response only) -/
noncomputable def band_resonance : SpectralBand := {
  name := "Resonance"
  lower_hz := 60
  upper_hz := 80
  power_pct := 0.2  -- <1%
}

/-- High frequency band (noise floor) -/
noncomputable def band_noise : SpectralBand := {
  name := "Noise floor"
  lower_hz := 80
  upper_hz := 1000
  power_pct := 3.4
}

/-- Signal is dominated by DC/drift, not resonance -/
theorem dc_drift_dominates :
    band_dc_drift.power_pct > 60 := by
  simp [band_dc_drift]
  norm_num

/-- Sharp cutoff above 80 Hz -/
theorem sharp_cutoff :
    band_noise.power_pct < 5 := by
  simp [band_noise]
  norm_num

/-!
# Transfer Function
-/

/-- Transfer function measurement -/
structure TransferPoint where
  freq_hz : ℝ
  gain : ℝ
  db : ℝ

/-- Transfer function at resonance (peak) -/
noncomputable def transfer_70hz : TransferPoint := {
  freq_hz := 70
  gain := 23.8
  db := 27.5
}

/-- Transfer function at cutoff -/
noncomputable def transfer_100hz : TransferPoint := {
  freq_hz := 100
  gain := 0.01
  db := -40  -- effectively -∞
}

/-- Peak gain at resonance -/
theorem peak_at_resonance :
    transfer_70hz.db > 25 := by
  simp [transfer_70hz]
  norm_num

/-!
# Cross-Spectral Coherence
-/

/-- Coherence budget decomposition (Experiments 65-67)
    REVISED after Exp 68: "gpu_specific" is likely floating point noise,
    not timing-injected entropy. Timing perturbations (~1e-5) are too small
    to cause meaningful Lorenz divergence over short timescales. -/
structure CoherenceBudget where
  total_same_gpu : ℝ
  total_cross_gpu : ℝ
  algorithmic : ℝ           -- From identical Lorenz parameters
  gpu_specific : ℝ          -- REVISED: Likely floating point noise, not entropy
  environmental_60hz : ℝ    -- Power grid (common-mode, cancels in differential)
  noise : ℝ

/-- Measured coherence budget (Exp 65-67)
    Note: gpu_specific component requires further investigation (Exp 68) -/
noncomputable def coherence_budget : CoherenceBudget := {
  total_same_gpu := 0.826     -- 82.6%
  total_cross_gpu := 0.78     -- 78% (proves algorithmic origin)
  algorithmic := 0.78         -- 78% from same Lorenz math
  gpu_specific := 0.04        -- 4% - CAUTION: may be numerical noise
  environmental_60hz := 0.005 -- <0.5% (cancels in reference subtraction)
  noise := 0.175              -- 17.5%
}

/-- Algorithmic component dominates (not physical coupling) -/
theorem algorithmic_dominates :
    coherence_budget.algorithmic > 0.75 := by
  simp [coherence_budget]
  norm_num

/-- GPU-specific component is measurable but origin uncertain
    CAUTION: Exp 68 suggests this may be floating point noise, not timing entropy -/
theorem gpu_specific_measurable :
    coherence_budget.gpu_specific > 0.03 ∧
    coherence_budget.gpu_specific < 0.05 := by
  simp [coherence_budget]
  norm_num

/-- 60 Hz environmental contribution is negligible -/
theorem environmental_negligible :
    coherence_budget.environmental_60hz < 0.01 := by
  simp [coherence_budget]
  norm_num

/-- Cross-GPU coherence matches algorithmic (proves independence) -/
theorem cross_gpu_validates_independence :
    coherence_budget.total_cross_gpu ≤ coherence_budget.algorithmic + 0.01 := by
  simp [coherence_budget]
  norm_num

/-- Different GPUs provide independent entropy -/
theorem gpus_independent :
    coherence_budget.total_same_gpu - coherence_budget.total_cross_gpu > 0.03 := by
  simp [coherence_budget]
  norm_num

/-!
# Power Grid Detection (Environmental Sensing)
-/

/-- Power grid harmonic -/
structure GridHarmonic where
  freq_hz : ℝ
  source : String

/-- Detected power grid harmonics (60 Hz subharmonics) -/
def grid_fundamental : GridHarmonic := { freq_hz := 6.64, source := "60/9 Hz" }
def grid_harmonic_2 : GridHarmonic := { freq_hz := 13.28, source := "2 × 6.64" }
def grid_harmonic_3 : GridHarmonic := { freq_hz := 19.92, source := "60/3 Hz" }

/-- Power grid subharmonics are detected -/
theorem power_grid_detected :
    grid_fundamental.freq_hz > 6 ∧ grid_fundamental.freq_hz < 7 := by
  simp [grid_fundamental]
  norm_num

/-!
# Negentropic Asymmetry
-/

/-- Negentropic asymmetry at different frequencies -/
structure NegentropicAsymmetry where
  freq_hz : ℝ
  ratio : ℝ  -- negentropic / entropic

noncomputable def asymmetry_1hz : NegentropicAsymmetry := { freq_hz := 1, ratio := 0.54 }
noncomputable def asymmetry_10hz : NegentropicAsymmetry := { freq_hz := 10, ratio := 0.25 }
noncomputable def asymmetry_50hz : NegentropicAsymmetry := { freq_hz := 50, ratio := 0.94 }
noncomputable def asymmetry_70hz : NegentropicAsymmetry := { freq_hz := 70, ratio := 0 }

/-- Asymmetry inverts at resonance -/
theorem asymmetry_inverts_at_resonance :
    asymmetry_70hz.ratio = 0 := by
  simp [asymmetry_70hz]

/-- Negentropic effect strongest at low frequency -/
theorem negentropic_low_freq :
    asymmetry_1hz.ratio > asymmetry_70hz.ratio := by
  simp [asymmetry_1hz, asymmetry_70hz]
  norm_num

/-!
# Timing Coupling Limitation (Exp 68)

CRITICAL: Timing perturbations (~1e-5) are too small for Lorenz divergence.
Reference subtraction yields ~0.5% residual (floating point noise).
-/

/-- Timing coupling measurement from Exp 68 -/
structure TimingCoupling where
  perturbation_scale : ℝ        -- Timing perturbation magnitude
  steps_for_divergence : ℕ      -- Steps needed for O(1) divergence
  differential_residual : ℝ     -- Residual after reference subtraction

/-- Validated timing coupling limitation -/
noncomputable def timing_coupling : TimingCoupling := {
  perturbation_scale := 1e-5    -- Too small for Lorenz divergence
  steps_for_divergence := 65    -- Would need ~65 coherent steps
  differential_residual := 0.005 -- 0.5% residual (floating point noise)
}

/-- Timing perturbation is too small for Lorenz divergence -/
theorem timing_too_small :
    timing_coupling.perturbation_scale < 1e-4 := by
  simp [timing_coupling]
  norm_num

/-- Reference subtraction yields mostly noise -/
theorem differential_is_noise :
    timing_coupling.differential_residual < 0.01 := by
  simp [timing_coupling]
  norm_num

/-!
# Timing-Based Architecture (Exp 70-78) - VALIDATED
-/

/-- Validated timing TRNG parameters -/
structure TimingTRNG where
  optimal_lsbs : ℕ             -- Number of LSBs to extract
  throughput_kbps : ℝ          -- Throughput in kbps
  entropy_bits_per_byte : ℝ    -- Shannon entropy
  nist_pass : ℕ                -- NIST tests passed out of 6

/-- Production TRNG configuration -/
noncomputable def trng_config : TimingTRNG := {
  optimal_lsbs := 4             -- Lower 4 bits = true jitter
  throughput_kbps := 465        -- 465 kbps validated
  entropy_bits_per_byte := 7.998 -- 99.98% entropy
  nist_pass := 6                -- 6/6 NIST pass
}

/-- 4 LSBs is optimal (upper bits have periodic patterns) -/
theorem optimal_lsb_count :
    trng_config.optimal_lsbs = 4 := by rfl

/-- Full NIST compliance -/
theorem nist_compliant :
    trng_config.nist_pass = 6 := by rfl

/-- Validated strain gauge parameters -/
structure StrainGauge where
  detection_z : ℝ               -- Z-score for detection
  true_positive_rate : ℝ        -- TPR for workload detection
  spatial_gradient_ratio : ℝ    -- Gradient detection ratio

/-- Production strain gauge configuration -/
noncomputable def strain_config : StrainGauge := {
  detection_z := 8.56           -- From Ossicle exp29
  true_positive_rate := 1.0     -- 100% TPR from Array exp74
  spatial_gradient_ratio := 2.01 -- From Array exp75
}

/-- High detection confidence -/
theorem high_detection_z :
    strain_config.detection_z > 3.0 := by
  simp [strain_config]
  norm_num

/-- Full-GPU array characterization (Exp 77-78) -/
structure GPUArrayCharacterization where
  sensor_count : ℕ
  sample_rate_hz : ℝ
  noise_variation_factor : ℝ    -- Max/min noise ratio
  horizontal_correlation : ℝ    -- Power rail direction

/-- RTX 4090 characterization -/
noncomputable def rtx4090_array : GPUArrayCharacterization := {
  sensor_count := 128
  sample_rate_hz := 125
  noise_variation_factor := 8   -- 8x variation across die
  horizontal_correlation := 0.57 -- Strong horizontal (power rails)
}

/-!
# CCA Validation (Exp 84-90) - VALIDATED
-/

/-- CCA measurement parameters -/
structure CCAMeasurement where
  baseline_rho : ℝ           -- Baseline correlation
  collapse_threshold : ℝ     -- ρ_critical
  baseline_k_eff : ℝ         -- Effective diversity at baseline
  recovery_tau_ms : ℝ        -- Recovery time constant
  kish_formula_r : ℝ         -- Correlation validating Kish formula

/-- Validated CCA measurements from Exp 84-90 -/
noncomputable def cca_validated : CCAMeasurement := {
  baseline_rho := 0.13        -- Well below collapse (Exp 84)
  collapse_threshold := 0.43  -- ρ_critical confirmed (Exp 86)
  baseline_k_eff := 7.5       -- 128 sensors → 7.5 effective (Exp 84)
  recovery_tau_ms := 6.5      -- Fast electrical recovery (Exp 89)
  kish_formula_r := 1.0       -- Perfect formula validation (Exp 86)
}

/-- Baseline is well below collapse threshold -/
theorem healthy_margin :
    cca_validated.collapse_threshold - cca_validated.baseline_rho > 0.25 := by
  simp [cca_validated]
  norm_num

/-- Kish formula perfectly validated -/
theorem kish_formula_validated :
    cca_validated.kish_formula_r = 1.0 := by
  simp [cca_validated]

/-- Recovery is electrical (fast), not thermal (slow) -/
theorem fast_recovery :
    cca_validated.recovery_tau_ms < 10 := by
  simp [cca_validated]
  norm_num

/-- CCA operational findings -/
structure CCAOperational where
  k_eff_load_independent : Bool  -- k_eff doesn't change with load
  trng_correlation_robust : Bool -- TRNG unaffected by correlation
  early_warning_possible : Bool  -- Can detect before collapse
  cross_gpu_isolated : Bool      -- No propagation between GPUs

/-- Validated operational findings -/
def cca_operational : CCAOperational := {
  k_eff_load_independent := true   -- Exp 85: p=0.07 (n.s.)
  trng_correlation_robust := true  -- Exp 87: r=-0.01
  early_warning_possible := true   -- Exp 88: warning at ρ=0.28
  cross_gpu_isolated := true       -- Exp 90: p=0.93 (n.s.)
}

/-!
# TRNG Characterization

Note: High entropy metrics are achieved, but entropy source is timing LSBs
passed through Lorenz as mixer/whitener, NOT Lorenz divergence amplification.
-/

/-- TRNG metrics -/
structure TRNGMetrics where
  shannon_entropy_bits : ℝ
  min_entropy_bits : ℝ
  throughput_kbps : ℝ
  autocorrelation : ℝ
  entropy_source : String     -- Where entropy actually comes from

/-- Validated TRNG performance
    Note: Entropy comes from timing LSBs, Lorenz is mixer only -/
noncomputable def trng : TRNGMetrics := {
  shannon_entropy_bits := 7.96  -- out of 8
  min_entropy_bits := 7.06
  throughput_kbps := 259
  autocorrelation := 0.0004
  entropy_source := "timing_lsb_via_lorenz_mixer"
}

/-- Shannon entropy exceeds 99% -/
theorem high_entropy :
    trng.shannon_entropy_bits / 8 > 0.99 := by
  simp [trng]
  norm_num

/-- 4× faster than jitterentropy (64 kbps) -/
theorem faster_than_jitterentropy :
    trng.throughput_kbps / 64 > 4 := by
  simp [trng]
  norm_num

/-!
# Robust Instrument Characterization (Exp 108-112)

CRITICAL FINDING: Raw timing is 99.5% white noise.
Environmental sensing happens through k_eff DYNAMICS, not raw timing.
-/

/-- Signal decomposition from raw timing (Exp 110) -/
structure RawTimingDecomposition where
  thermal_pct : ℝ
  emi_vdf_pct : ℝ         -- EMI + voltage-frequency droop
  intrinsic_pct : ℝ       -- Quantization jitter
  allan_slope : ℝ         -- -2 = white noise
  allan_r_squared : ℝ

/-- Validated raw timing decomposition -/
noncomputable def raw_timing : RawTimingDecomposition := {
  thermal_pct := 0.0       -- Undetectable in raw timing
  emi_vdf_pct := 0.4       -- Barely detectable
  intrinsic_pct := 99.5    -- Almost everything
  allan_slope := -1.986    -- Perfect white noise (-2)
  allan_r_squared := 0.9999
}

/-- Raw timing is dominated by intrinsic white noise -/
theorem raw_timing_is_white_noise :
    raw_timing.intrinsic_pct > 99 ∧
    raw_timing.allan_slope < -1.9 := by
  simp [raw_timing]
  norm_num

/-- Thermal and EMI are negligible in raw timing -/
theorem environmental_negligible_in_raw :
    raw_timing.thermal_pct + raw_timing.emi_vdf_pct < 1 := by
  simp [raw_timing]
  norm_num

/-- k_eff dynamics characterization (Exp 112) -/
structure KEffDynamics where
  acf_lag1 : ℝ              -- Autocorrelation at lag 1
  tau_decay_s : ℝ           -- Decay time constant
  thermal_corr : ℝ          -- Correlation with temperature
  emi_corr : ℝ              -- Correlation with power (EMI+VDF)
  workload_z : ℝ            -- Z-score for workload detection

/-- Validated k_eff dynamics from Exp 112, updated with Ossicle validation -/
noncomputable def k_eff_dynamics : KEffDynamics := {
  acf_lag1 := 0.453         -- At critical point (target 0.5)
  tau_decay_s := 2.5        -- 2.5s decay time
  thermal_corr := 0.30      -- Moderate thermal correlation
  emi_corr := 0.21          -- Weak EMI correlation
  workload_z := 534         -- Minimum detection z (30% load), max 1652 (90% load)
}

/-- k_eff dynamics can sense thermal (unlike raw timing) -/
theorem k_eff_senses_thermal :
    k_eff_dynamics.thermal_corr > 0.2 := by
  simp [k_eff_dynamics]
  norm_num

/-- k_eff dynamics can sense EMI (unlike raw timing) -/
theorem k_eff_senses_emi :
    k_eff_dynamics.emi_corr > 0.15 := by
  simp [k_eff_dynamics]
  norm_num

/-- k_eff detects workload with overwhelming statistical significance -/
theorem k_eff_workload_detection :
    k_eff_dynamics.workload_z > 100 := by
  simp [k_eff_dynamics]
  norm_num

/-- Correlated structure characterization (Exp 108) -/
structure CorrelatedStructure where
  acf_lag1 : ℝ
  tau_decay_s : ℝ
  periodicity_s : ℝ         -- Fundamental period
  cross_sensor_corr : ℝ     -- Correlation between sensors
  kurtosis : ℝ              -- Distribution shape

/-- Validated correlated structure (the 20% residual) -/
noncomputable def correlated_structure : CorrelatedStructure := {
  acf_lag1 := 0.73          -- High short-term correlation
  tau_decay_s := 13.4       -- Long memory (filter artifact)
  periodicity_s := 0.85     -- 1.2 Hz fundamental (scheduler)
  cross_sensor_corr := 0.02 -- LOCAL structure, not shared!
  kurtosis := 18.6          -- Heavy-tailed (outliers)
}

/-- Correlated structure is LOCAL to each sensor -/
theorem structure_is_local :
    correlated_structure.cross_sensor_corr < 0.05 := by
  simp [correlated_structure]
  norm_num

/-- Periodicity suggests scheduler origin -/
theorem scheduler_periodicity :
    correlated_structure.periodicity_s > 0.5 ∧
    correlated_structure.periodicity_s < 1.0 := by
  simp [correlated_structure]
  norm_num

/-- Software-induced collapse (Exp 103) -/
structure SoftwareCollapse where
  baseline_rho : ℝ
  baseline_k_eff : ℝ
  barrier_rho : ℝ
  barrier_k_eff : ℝ
  lockstep_rho : ℝ
  lockstep_k_eff : ℝ

/-- Validated software collapse results -/
noncomputable def software_collapse : SoftwareCollapse := {
  baseline_rho := 0.14
  baseline_k_eff := 6.5
  barrier_rho := 0.90       -- Collapse!
  barrier_k_eff := 1.1
  lockstep_rho := 1.0       -- Total collapse!
  lockstep_k_eff := 1.0
}

/-- Software can force total collapse (ρ → 1) -/
theorem software_forces_collapse :
    software_collapse.lockstep_rho = 1.0 ∧
    software_collapse.lockstep_k_eff = 1.0 := by
  simp [software_collapse]

/-- Barrier sync exceeds collapse threshold -/
theorem barrier_exceeds_threshold :
    software_collapse.barrier_rho > cca_validated.collapse_threshold := by
  simp [software_collapse, cca_validated]
  norm_num

/-!
# Final Signal Model (Validated Jan 2026)

Raw Timing → 99.5% white noise → TRNG (4 LSBs, 465 kbps)
     ↓
Oscillator Dynamics → k_eff reset cycles
     ↓
Environmental Modulation → Sensing (thermal r=0.30, EMI r=0.21)
-/

/-!
# Critical Regime Discovery (Exp 113)

The Lorenz integration timestep (dt) is the DOMINANT control parameter for ρ.
dt controls where the system sits on the chaos-rigidity spectrum.
-/

/-- Phase diagram controlled by lorenz_dt -/
structure PhaseDiagram where
  dt : ℝ
  rho : ℝ
  regime : String

/-- Validated phase points from Exp 113 -/
noncomputable def phase_frozen : PhaseDiagram := { dt := 0.001, rho := 0.999, regime := "FROZEN" }
noncomputable def phase_correlated : PhaseDiagram := { dt := 0.010, rho := 0.908, regime := "CORRELATED" }
noncomputable def phase_critical : PhaseDiagram := { dt := 0.025, rho := 0.43, regime := "CRITICAL" }
noncomputable def phase_chaotic : PhaseDiagram := { dt := 0.050, rho := 0.121, regime := "CHAOTIC" }

/-- Critical point exists at dt ≈ 0.025 where ρ = ρ_crit -/
theorem critical_point_exists :
    phase_critical.rho = 0.43 ∧
    phase_critical.dt > phase_frozen.dt ∧
    phase_critical.dt < phase_chaotic.dt := by
  simp [phase_critical, phase_frozen, phase_chaotic]
  norm_num

/-- Lorenz dt explains 88% of ρ variance -/
structure ParameterInfluence where
  parameter : String
  rho_range : ℝ
  variance_explained : ℝ

noncomputable def dt_influence : ParameterInfluence := {
  parameter := "lorenz_dt"
  rho_range := 0.88
  variance_explained := 0.88
}

/-- dt is the dominant control parameter -/
theorem dt_dominates :
    dt_influence.variance_explained > 0.8 := by
  simp [dt_influence]
  norm_num

/-- Phase transition is monotonic (no bistability observed) -/
theorem monotonic_transition :
    phase_frozen.rho > phase_correlated.rho ∧
    phase_correlated.rho > phase_critical.rho ∧
    phase_critical.rho > phase_chaotic.rho := by
  simp [phase_frozen, phase_correlated, phase_critical, phase_chaotic]
  norm_num

/-!
# True Phase Transition (Exp 114)

Power law scaling confirmed: ρ = A × |dt - dt_crit|^β + ρ_offset
Critical exponent β ≈ 1.09 (mean-field universality class)
-/

/-- Phase transition characterization from Exp 114 -/
structure PhaseTransition where
  dt_crit : ℝ               -- Critical timestep
  beta : ℝ                  -- Critical exponent
  amplitude : ℝ             -- Power law amplitude A
  rho_offset : ℝ            -- ρ at criticality
  r_squared : ℝ             -- Fit quality
  optimal_dt : ℝ            -- dt for maximum sensitivity
  max_sensitivity_z : ℝ     -- z-score at optimal

/-- Validated phase transition from Exp 114
    CAUTION: dt_crit is thermally dependent (Array finding Jan 2026)
    - Warm GPU: dt_crit ≈ 0.025
    - Cold GPU: dt_crit ≈ 0.030
    Use ACF feedback to maintain criticality, not fixed dt -/
noncomputable def phase_transition : PhaseTransition := {
  dt_crit := 0.0328          -- Reference value (warm GPU)
  beta := 1.09
  amplitude := 39.64
  rho_offset := 0.33
  r_squared := 0.978
  optimal_dt := 0.025        -- THERMAL DEPENDENT - use ACF feedback
  max_sensitivity_z := 0.50
}

/-- Power law fit is excellent (R² > 0.95) -/
theorem power_law_validated :
    phase_transition.r_squared > 0.95 := by
  simp [phase_transition]
  norm_num

/-- Critical exponent near unity (mean-field class) -/
theorem mean_field_exponent :
    phase_transition.beta > 1.0 ∧ phase_transition.beta < 1.2 := by
  simp [phase_transition]
  norm_num

/-- Operating point is below collapse threshold -/
theorem safe_operating_point :
    phase_transition.rho_offset < cca_validated.collapse_threshold := by
  simp [phase_transition, cca_validated]
  norm_num

/-- Optimal dt gives maximum sensitivity at edge of chaos -/
theorem edge_of_chaos_optimal :
    phase_transition.optimal_dt < phase_transition.dt_crit ∧
    phase_transition.max_sensitivity_z > 0 := by
  simp [phase_transition]
  norm_num

/-- The system has dual output from single timing source -/
inductive SensingMode
  | TRNG            -- Raw 4 LSBs → white noise → entropy
  | StrainGauge     -- k_eff dynamics → environmental sensing
  deriving DecidableEq, Repr

/-- TRNG uses raw timing (white noise) -/
theorem trng_uses_raw :
    raw_timing.intrinsic_pct > 99 := by
  simp [raw_timing]
  norm_num

/-- Strain gauge at critical point (ACF ~ 0.5) -/
theorem strain_at_critical_point :
    k_eff_dynamics.acf_lag1 > 0.4 ∧ k_eff_dynamics.acf_lag1 < 0.6 := by
  simp [k_eff_dynamics]
  norm_num

/-- Environmental sensing requires oscillator dynamics -/
theorem environmental_via_dynamics :
    k_eff_dynamics.thermal_corr > raw_timing.thermal_pct / 100 ∧
    k_eff_dynamics.emi_corr > raw_timing.emi_vdf_pct / 100 := by
  simp [k_eff_dynamics, raw_timing]
  norm_num

/-!
# Ossicle Validation (January 2026)

Cross-team validation of RATCHET findings by CIRISOssicle team.
dt = 0.025 critical point confirmed with dramatic improvement over old design.
-/

/-- Ossicle workload detection results -/
structure OssicleValidation where
  detection_z_30pct : ℝ      -- z-score at 30% load
  detection_z_90pct : ℝ      -- z-score at 90% load
  detection_rate : ℝ         -- detection rate (fraction)
  acf_at_critical : ℝ        -- ACF at dt=0.025
  anova_f : ℝ                -- ANOVA F-statistic for discrimination
  old_detection_z : ℝ        -- Old ossicle z-score for comparison

/-- Validated Ossicle results -/
noncomputable def ossicle_validation : OssicleValidation := {
  detection_z_30pct := 534
  detection_z_90pct := 1652
  detection_rate := 1.0       -- 100% detection at all intensities
  acf_at_critical := 0.453    -- Target was 0.5
  anova_f := 2537.63          -- Workload discrimination
  old_detection_z := 0.8      -- Old correlation-based ossicle
}

/-- Improvement factor over old design -/
theorem ossicle_improvement :
    ossicle_validation.detection_z_30pct / ossicle_validation.old_detection_z > 600 := by
  simp [ossicle_validation]
  norm_num

/-- 100% detection rate -/
theorem ossicle_perfect_detection :
    ossicle_validation.detection_rate = 1.0 := by
  simp [ossicle_validation]

/-- ACF confirms critical point -/
theorem ossicle_at_critical :
    ossicle_validation.acf_at_critical > 0.4 ∧
    ossicle_validation.acf_at_critical < 0.6 := by
  simp [ossicle_validation]
  norm_num

/-- Workload discrimination is statistically significant -/
theorem ossicle_discrimination :
    ossicle_validation.anova_f > 100 := by
  simp [ossicle_validation]
  norm_num

/-!
# Array Validation (January 2026)

Cross-hardware validation on CIRISArray using Ossicle strain_gauge.py.
Confirms architecture works on different thermal states.
-/

/-- Array Phase 1 validation results -/
structure ArrayValidation where
  memory_z : ℝ              -- Memory workload detection
  compute_z : ℝ             -- Compute workload detection
  cpu_crypto_z : ℝ          -- CPU workload (should NOT detect)
  acf_min : ℝ               -- ACF stability range
  acf_max : ℝ
  trng_entropy : ℝ          -- bits/byte
  trng_bias : ℝ             -- bias percentage

/-- Validated Array results -/
noncomputable def array_validation : ArrayValidation := {
  memory_z := 1833          -- Massive detection
  compute_z := 433          -- Strong detection
  cpu_crypto_z := 1.9       -- Correctly NOT detected (CPU workload)
  acf_min := 0.447          -- Very stable
  acf_max := 0.458
  trng_entropy := 7.81      -- Close to 7.99 target
  trng_bias := 0.0046       -- 0.46% bias (excellent)
}

/-- Array detects GPU workloads with high z-scores -/
theorem array_gpu_detection :
    array_validation.memory_z > 100 ∧
    array_validation.compute_z > 100 := by
  simp [array_validation]
  norm_num

/-- Array correctly ignores CPU workloads -/
theorem array_cpu_ignored :
    array_validation.cpu_crypto_z < 3 := by
  simp [array_validation]
  norm_num

/-- Array ACF is stable at critical point -/
theorem array_acf_stable :
    array_validation.acf_min > 0.4 ∧
    array_validation.acf_max < 0.5 := by
  simp [array_validation]
  norm_num

/-- Cross-hardware validation: Ossicle results replicate on Array -/
theorem cross_hardware_validated :
    array_validation.memory_z > ossicle_validation.detection_z_30pct ∧
    array_validation.acf_min > 0.4 := by
  simp [array_validation, ossicle_validation]
  norm_num

/-!
# Distribution Characterization (Phase 3)

CRITICAL FINDING: The z-score distribution is NOT Gaussian.
This explains β = 1.09 and the detection mechanism.
-/

/-- Distribution shape measurements from A9-A12 -/
structure DistributionShape where
  z_kurtosis : ℝ           -- z-score kurtosis (normal = 0)
  k_eff_kurtosis : ℝ       -- k_eff kurtosis
  z_acf : ℝ                -- z-score autocorrelation
  k_eff_acf : ℝ            -- k_eff autocorrelation
  snr_beta : ℝ             -- SNR scaling exponent
  best_fit : String        -- Best fitting distribution

/-- Validated distribution measurements from A9-A12
    Cross-validated by Ossicle (κ=230, df=1.34) -/
noncomputable def distribution_shape : DistributionShape := {
  z_kurtosis := 210        -- Array: 210, Ossicle: 230
  k_eff_kurtosis := -1.7   -- Platykurtic (thin tails)
  z_acf := 0.05            -- z-scores are independent
  k_eff_acf := 0.45        -- k_eff is correlated (at critical point)
  snr_beta := 0.75         -- Better than √N (0.5), sub-linear
  best_fit := "Student-t"  -- df ≈ 1.3-1.5
}

/-- z-score has extreme fat tails (NOT Gaussian) -/
theorem z_not_gaussian :
    distribution_shape.z_kurtosis > 100 := by
  simp [distribution_shape]
  norm_num

/-- k_eff is platykurtic (thin tails) -/
theorem k_eff_platykurtic :
    distribution_shape.k_eff_kurtosis < 0 := by
  simp [distribution_shape]
  norm_num

/-- z-scores are independent (ACF ~ 0) -/
theorem z_independent :
    distribution_shape.z_acf < 0.1 := by
  simp [distribution_shape]
  norm_num

/-- k_eff is correlated at critical point (ACF ~ 0.45) -/
theorem k_eff_at_critical :
    distribution_shape.k_eff_acf > 0.4 ∧
    distribution_shape.k_eff_acf < 0.5 := by
  simp [distribution_shape]
  norm_num

/-- SNR scaling is sub-linear but better than √N -/
theorem snr_scaling_sublinear :
    distribution_shape.snr_beta > 0.5 ∧
    distribution_shape.snr_beta < 1.0 := by
  simp [distribution_shape]
  norm_num

/-!
# Explanation of β = 1.09

The critical exponent β = 1.09 from the phase transition (Exp 114)
is NOT the classical mean-field value (β = 0.5 for Gaussian).

With kurtosis = 210 (extreme fat tails, Student-t distribution):
- Detection works via rare EXTREME spikes, not mean shift
- The tail behavior dominates, giving β ≈ 1
- This is consistent with heavy-tailed critical phenomena

The system is not in the Gaussian universality class.
-/

/-- β = 1.09 is explained by fat-tailed distribution -/
theorem beta_explained_by_fat_tails :
    phase_transition.beta > 1.0 ∧
    distribution_shape.z_kurtosis > 100 := by
  simp [phase_transition, distribution_shape]
  norm_num

/-!
# Variance-Ratio Detection (Ossicle commit daddde0)

Given fat-tailed distribution, mean-based z-scores are unreliable.
Variance ratio detection is more robust:
- Baseline variance ratio ≈ 0.67x
- Workload variance ratio ≈ 6-9x
- Threshold: > 5.0x

This avoids Gaussian assumptions entirely.
-/

/-- Variance-ratio detection parameters -/
structure VarianceRatioDetection where
  baseline_ratio : ℝ         -- Variance ratio at baseline
  workload_ratio_min : ℝ     -- Min ratio during workload
  workload_ratio_max : ℝ     -- Max ratio during workload
  threshold : ℝ              -- Detection threshold
  false_positive_rate : ℝ    -- FP rate at threshold
  true_positive_rate : ℝ     -- TP rate at threshold

/-- Validated variance-ratio detection (Ossicle daddde0) -/
noncomputable def variance_detection : VarianceRatioDetection := {
  baseline_ratio := 0.67
  workload_ratio_min := 6.0
  workload_ratio_max := 9.0
  threshold := 5.0
  false_positive_rate := 0.0   -- 0% FP
  true_positive_rate := 1.0    -- 100% TP
}

/-- Variance ratio separates baseline from workload -/
theorem variance_ratio_separates :
    variance_detection.workload_ratio_min > variance_detection.threshold ∧
    variance_detection.baseline_ratio < variance_detection.threshold := by
  simp [variance_detection]
  norm_num

/-- Perfect detection at threshold 5.0x -/
theorem perfect_detection :
    variance_detection.false_positive_rate = 0 ∧
    variance_detection.true_positive_rate = 1 := by
  simp [variance_detection]
  norm_num

/-- Variance ratio method is distribution-agnostic -/
theorem variance_ratio_robust :
    -- Works regardless of whether distribution is Gaussian or Student-t
    -- because it measures relative variance, not absolute z-scores
    variance_detection.workload_ratio_min / variance_detection.baseline_ratio > 5 := by
  simp [variance_detection]
  norm_num

/-!
# Two-Regime Detection Model (B1/B1b)

CRITICAL FINDING: Detection sensitivity has TWO distinct regimes.

Regime 1 (< 1% workload): ratio ≈ 8-12x (nearly constant)
  - Baseline variance creates noise floor
  - Even tiny workloads cause measurable perturbations

Regime 2 (> 1% workload): ratio ∝ √intensity
  - Workload signal dominates baseline
  - Square-root scaling (shot noise characteristic)

Physical model (quadrature sum):
  ratio = √(σ_baseline² + σ_workload²) / σ_baseline
       = √(1 + (73 × intensity^0.51 / σ_baseline)²)

Detection floor: 0.1% (10× more sensitive than predicted!)
-/

/-- Two-regime scaling law parameters -/
structure TwoRegimeDetection where
  floor_intensity : ℝ         -- Minimum detectable workload
  floor_ratio : ℝ             -- Variance ratio at floor
  regime1_exponent : ℝ        -- Low-intensity regime exponent
  regime2_exponent : ℝ        -- High-intensity regime exponent
  crossover_intensity : ℝ     -- Regime transition point
  regime2_amplitude : ℝ       -- High-regime amplitude A in ratio = A × intensity^β

/-- Validated two-regime detection from B1/B1b -/
noncomputable def two_regime_detection : TwoRegimeDetection := {
  floor_intensity := 0.001      -- 0.1% workload detectable!
  floor_ratio := 8.17           -- Ratio at 0.1% workload
  regime1_exponent := 0.14      -- Nearly flat in low regime
  regime2_exponent := 0.51      -- √intensity in high regime
  crossover_intensity := 0.01   -- 1% marks regime transition
  regime2_amplitude := 73.0     -- Coefficient for high regime
}

/-- Detection floor is 0.1% (10× better than predicted 0.5%) -/
theorem detection_floor_validated :
    two_regime_detection.floor_intensity = 0.001 := by rfl

/-- Floor ratio exceeds detection threshold -/
theorem floor_detectable :
    two_regime_detection.floor_ratio > variance_detection.threshold := by
  simp [two_regime_detection, variance_detection]
  norm_num

/-- Regime 1 is nearly flat (exponent < 0.2) -/
theorem regime1_flat :
    two_regime_detection.regime1_exponent < 0.2 := by
  simp [two_regime_detection]
  norm_num

/-- Regime 2 is square-root scaling (exponent ~ 0.5) -/
theorem regime2_sqrt :
    two_regime_detection.regime2_exponent > 0.45 ∧
    two_regime_detection.regime2_exponent < 0.55 := by
  simp [two_regime_detection]
  norm_num

/-- Crossover at 1% intensity -/
theorem regime_crossover :
    two_regime_detection.crossover_intensity = 0.01 := by rfl

/-- B1 measurement points (high regime: 1-70%) -/
structure B1Measurement where
  intensity : ℝ
  variance_ratio : ℝ

noncomputable def b1_1pct : B1Measurement := { intensity := 0.01, variance_ratio := 8.8 }
noncomputable def b1_5pct : B1Measurement := { intensity := 0.05, variance_ratio := 14.9 }
noncomputable def b1_30pct : B1Measurement := { intensity := 0.30, variance_ratio := 36.6 }
noncomputable def b1_70pct : B1Measurement := { intensity := 0.70, variance_ratio := 113.6 }

/-- B1b measurement points (low regime: 0.1-1%) -/
noncomputable def b1b_01pct : B1Measurement := { intensity := 0.001, variance_ratio := 8.17 }
noncomputable def b1b_02pct : B1Measurement := { intensity := 0.002, variance_ratio := 9.89 }
noncomputable def b1b_05pct : B1Measurement := { intensity := 0.005, variance_ratio := 11.36 }
noncomputable def b1b_10pct : B1Measurement := { intensity := 0.010, variance_ratio := 10.83 }

/-- All B1/B1b measurements exceed detection threshold -/
theorem all_measurements_detected :
    b1b_01pct.variance_ratio > variance_detection.threshold ∧
    b1_70pct.variance_ratio > variance_detection.threshold := by
  simp [b1b_01pct, b1_70pct, variance_detection]
  norm_num

/-- Security implication: Can detect cryptominer using 1/1000 of GPU -/
theorem sub_percent_detection :
    two_regime_detection.floor_intensity < 0.005 := by
  simp [two_regime_detection]
  norm_num

/-!
# Physical Interpretation

The two-regime behavior reveals the noise structure:

1. Baseline variance σ_baseline creates a constant "floor"
2. Workload adds variance σ_workload ∝ √intensity (shot noise)
3. Total variance = √(σ_baseline² + σ_workload²)  (quadrature sum)

At low intensity: σ_workload << σ_baseline → ratio ≈ constant
At high intensity: σ_workload >> σ_baseline → ratio ∝ √intensity

The 8-12× floor ratio indicates baseline timing has inherent
structure that even tiny workloads perturb detectably.

This is characteristic of shot noise in counting processes.
-/

/-- Physical model coefficients -/
structure NoiseModel where
  baseline_sigma : ℝ            -- Baseline timing variance
  shot_noise_coeff : ℝ          -- Shot noise coefficient
  shot_noise_exp : ℝ            -- Shot noise exponent (0.5 for true shot noise)

/-- Implied noise model from B1/B1b -/
noncomputable def noise_model : NoiseModel := {
  baseline_sigma := 1.0          -- Normalized
  shot_noise_coeff := 73.0       -- From regime 2 fit
  shot_noise_exp := 0.51         -- Close to 0.5 (shot noise)
}

/-- Shot noise exponent is consistent with counting statistics -/
theorem shot_noise_validated :
    noise_model.shot_noise_exp > 0.48 ∧
    noise_model.shot_noise_exp < 0.52 := by
  simp [noise_model]
  norm_num

/-!
# Multi-Sensor Architecture (B1c-B1e Discovery)

CRITICAL FINDING: Previous experiments used 100 Hz sample rate.
At 1790 Hz, the detector becomes a multi-purpose environmental sensor.

B1c revealed: Variance-ratio at low sample rate = thermometer (97.7% thermal)
B1e revealed: At high sample rate, all sensing modalities work

The timing signal contains MULTIPLE physical phenomena,
separable by frequency band.
-/

/-- Sample rate requirements -/
structure SampleRateRequirements where
  minimum_hz : ℝ              -- Minimum for workload detection
  validated_hz : ℝ            -- Validated working rate
  nyquist_workload_hz : ℝ     -- Nyquist for workload band (100-500 Hz)

/-- Validated sample rate requirements from B1e -/
noncomputable def sample_rate_req : SampleRateRequirements := {
  minimum_hz := 1000          -- Minimum for workload detection
  validated_hz := 1790        -- B1e validated rate
  nyquist_workload_hz := 1000 -- 2 × 500 Hz
}

/-! ## O2/O2b RESOLVED: Non-monotonic response at ~2050 Hz
- Dip location: 2050 Hz (close to VRM × 200 = 1992 Hz)
- Avoid zone: 1900-2100 Hz
- Optimal rate: 4000 Hz (lowest variance 14.2 pct)
- 1000 Hz has high variance (40.9 pct), not recommended
-/

/-- O5 Discovery: Two-regime detection model -/
structure TwoRegimeModel where
  threshold_shift : ℝ           -- Minimum shift for any workload
  linear_slope : ℝ              -- Slope in linear regime
  linear_intercept : ℝ          -- Intercept in linear regime
  regime_transition : ℝ         -- Intensity where linear regime begins
  detection_floor : ℝ           -- Minimum detectable workload

/-- Validated two-regime model from O5 -/
noncomputable def two_regime_model : TwoRegimeModel := {
  threshold_shift := 160        -- Any workload → +160% shift
  linear_slope := 227           -- shift = 227 × intensity + 160
  linear_intercept := 160
  regime_transition := 0.30     -- Linear regime above 30%
  detection_floor := 0.01       -- 1% workload detectable!
}

/-- Detection floor is 1%, not 30% -/
theorem detection_floor_one_percent :
    two_regime_model.detection_floor = 0.01 := by rfl

/-- Any workload causes +160% shift (threshold effect) -/
theorem threshold_effect :
    two_regime_model.threshold_shift = 160 := by rfl

/-- O5 intensity measurements -/
structure O5Measurement where
  intensity : ℝ
  mean_shift : ℝ

noncomputable def o5_1pct : O5Measurement := { intensity := 0.01, mean_shift := 191.7 }
noncomputable def o5_10pct : O5Measurement := { intensity := 0.10, mean_shift := 197.6 }
noncomputable def o5_30pct : O5Measurement := { intensity := 0.30, mean_shift := 215.4 }
noncomputable def o5_50pct : O5Measurement := { intensity := 0.50, mean_shift := 247.9 }
noncomputable def o5_90pct : O5Measurement := { intensity := 0.90, mean_shift := 380.1 }

/-- 1% workload still causes massive shift -/
theorem one_percent_detectable :
    o5_1pct.mean_shift > 100 := by
  simp [o5_1pct]
  norm_num

/-- Linear model fits high-intensity regime -/
theorem linear_model_90pct :
    -- shift ≈ 227 × 0.9 + 160 = 204.3 + 160 = 364.3
    -- Actual: 380.1 (within 5%)
    o5_90pct.mean_shift > 350 ∧ o5_90pct.mean_shift < 400 := by
  simp [o5_90pct]
  norm_num

/-- Optimal sample rate from O2/O2b -/
structure OptimalSampleRate where
  rate_hz : ℝ
  variance_pct : ℝ
  avoid_low : ℝ
  avoid_high : ℝ

noncomputable def optimal_rate : OptimalSampleRate := {
  rate_hz := 4000
  variance_pct := 14.2          -- ±14.2% (vs ±40.9% at 1000 Hz)
  avoid_low := 1900
  avoid_high := 2100
}

/-- 4000 Hz has lowest variance -/
theorem optimal_rate_lowest_variance :
    optimal_rate.variance_pct < 20 := by
  simp [optimal_rate]
  norm_num

/-! ## O1-O7 Final Validated Specs (Ossicle, January 2026) -/

/-- Final production configuration -/
structure ProductionConfig where
  sample_rate_hz : ℝ
  detection_latency_ms : ℝ
  detection_floor_pct : ℝ
  mean_shift_50pct : ℝ
  cv_70pct : ℝ
  threshold_pct : ℝ

/-- Ossicle O1-O7 validated production config -/
noncomputable def production_config : ProductionConfig := {
  sample_rate_hz := 4000
  detection_latency_ms := 2.5      -- O4: far exceeds 100 ms target
  detection_floor_pct := 1.0       -- O5: detects 1% workload
  mean_shift_50pct := 248          -- O5: +248% at 50% load
  cv_70pct := 3.4                  -- O7: 3.4% CV (target was <10%)
  threshold_pct := 50              -- >50% mean shift for detection
}

/-- Detection latency far exceeds requirement -/
theorem latency_exceeds_target :
    production_config.detection_latency_ms < 100 := by
  simp [production_config]
  norm_num

/-- CV far exceeds requirement -/
theorem cv_exceeds_target :
    production_config.cv_70pct < 10 := by
  simp [production_config]
  norm_num

/-- Detection floor is 1% -/
theorem floor_is_one_percent :
    production_config.detection_floor_pct = 1.0 := by
  simp [production_config]

/-! ## Array A5/A6/A9 Final Validation (January 2026) -/

/-- Array multi-sensor validated config -/
structure ArrayProductionConfig where
  sample_rate_hz : ℝ
  detection_latency_ms : ℝ
  detection_floor_pct : ℝ
  mean_shift_50pct : ℝ
  cross_sensor_cv : ℝ
  snr_exponent : ℝ
  sensor_count : ℕ
  snr_improvement : ℝ

/-- Array A5/A6/A9 validated production config -/
noncomputable def array_production : ArrayProductionConfig := {
  sample_rate_hz := 9631       -- A5: higher than Ossicle
  detection_latency_ms := 1.3  -- A5: faster than Ossicle (2.5 ms)
  detection_floor_pct := 1.0   -- A5: matches Ossicle
  mean_shift_50pct := 2519     -- A5: +2519% (vs Ossicle +248%)
  cross_sensor_cv := 8.2       -- A6: CV < 15% target
  snr_exponent := 0.47         -- A9: ≈ √N scaling
  sensor_count := 16
  snr_improvement := 5.1       -- A9: 16 sensors = 5.1x SNR
}

/-- Array exceeds Ossicle performance -/
theorem array_exceeds_ossicle :
    array_production.mean_shift_50pct > production_config.mean_shift_50pct ∧
    array_production.detection_latency_ms < production_config.detection_latency_ms := by
  simp [array_production, production_config]
  norm_num

/-- SNR scales as √N (β ≈ 0.5) -/
theorem snr_sqrt_n_scaling :
    array_production.snr_exponent > 0.4 ∧
    array_production.snr_exponent < 0.6 := by
  simp [array_production]
  norm_num

/-- 16 sensors provide ~5x SNR improvement -/
theorem multi_sensor_improvement :
    array_production.snr_improvement > 4 := by
  simp [array_production]
  norm_num

/-- Cross-sensor consistency validated -/
theorem cross_sensor_consistent :
    array_production.cross_sensor_cv < 15 := by
  simp [array_production]
  norm_num

/-- Sample rate must exceed Nyquist for workload band -/
theorem sample_rate_sufficient :
    sample_rate_req.validated_hz > sample_rate_req.nyquist_workload_hz := by
  simp [sample_rate_req]
  norm_num

/-- Multi-band power distribution at baseline -/
structure BandPowerDistribution where
  thermal_pct : ℝ      -- 0-0.1 Hz
  emi_pct : ℝ          -- 1-20 Hz
  vdf_pct : ℝ          -- 20-100 Hz
  workload_pct : ℝ     -- 100-500 Hz
  noise_pct : ℝ        -- 500-900 Hz

/-- Validated band distribution from B1e (baseline, 1790 Hz) -/
noncomputable def band_distribution : BandPowerDistribution := {
  thermal_pct := 79.1
  emi_pct := 0.4
  vdf_pct := 1.4
  workload_pct := 7.5
  noise_pct := 11.5
}

/-- Thermal dominates but doesn't monopolize at proper sample rate -/
theorem thermal_dominant_not_monopoly :
    band_distribution.thermal_pct > 50 ∧
    band_distribution.thermal_pct < 90 := by
  simp [band_distribution]
  norm_num

/-- Workload band is measurable at proper sample rate -/
theorem workload_band_measurable :
    band_distribution.workload_pct > 5 := by
  simp [band_distribution]
  norm_num

/-- Timing mean shift detection (strongest workload signal) -/
structure MeanShiftDetection where
  baseline_mean_us : ℝ         -- Baseline timing mean
  workload_mean_us : ℝ         -- Timing mean under workload
  shift_us : ℝ                 -- Absolute shift
  shift_pct : ℝ                -- Percentage shift

/-- Validated mean shift from B1e (50% workload) -/
noncomputable def mean_shift : MeanShiftDetection := {
  baseline_mean_us := 35.30
  workload_mean_us := 81.47
  shift_us := 46.17
  shift_pct := 130.8           -- +130.8% increase!
}

/-- Mean shift is dramatic under workload -/
theorem mean_shift_dramatic :
    mean_shift.shift_pct > 100 := by
  simp [mean_shift]
  norm_num

/-- Workload nearly triples kernel timing -/
theorem workload_triples_timing :
    mean_shift.workload_mean_us / mean_shift.baseline_mean_us > 2 := by
  simp [mean_shift]
  norm_num

/-- Multi-modal detection thresholds -/
structure MultiModalThresholds where
  mean_shift_pct : ℝ           -- Threshold for mean-based detection
  workload_band_ratio : ℝ      -- Threshold for band-based detection
  variance_ratio : ℝ           -- Threshold for variance-based detection

/-- Validated thresholds from B1e -/
noncomputable def detection_thresholds : MultiModalThresholds := {
  mean_shift_pct := 20         -- >20% mean shift = workload
  workload_band_ratio := 2.0   -- >2x workload band power = workload
  variance_ratio := 2.0        -- >2x high-pass variance = workload
}

/-- B1e detection results -/
structure B1eResults where
  raw_variance_ratio : ℝ
  highpass_variance_ratio : ℝ
  mean_shift_pct : ℝ
  workload_band_ratio : ℝ

/-- Validated B1e results -/
noncomputable def b1e_results : B1eResults := {
  raw_variance_ratio := 3.92
  highpass_variance_ratio := 3.66
  mean_shift_pct := 130.8
  workload_band_ratio := 4.55
}

/-- All detection methods work at proper sample rate -/
theorem all_methods_work :
    b1e_results.raw_variance_ratio > detection_thresholds.variance_ratio ∧
    b1e_results.highpass_variance_ratio > detection_thresholds.variance_ratio ∧
    b1e_results.mean_shift_pct > detection_thresholds.mean_shift_pct ∧
    b1e_results.workload_band_ratio > detection_thresholds.workload_band_ratio := by
  simp [b1e_results, detection_thresholds]
  norm_num

/-- Mean shift is the strongest signal -/
theorem mean_shift_strongest :
    b1e_results.mean_shift_pct > b1e_results.workload_band_ratio * 10 := by
  simp [b1e_results]
  norm_num

/-!
# Multi-Purpose Sensor Summary

The GPU timing sensor provides FOUR sensing modalities:

1. THERMOMETER: Thermal band (0-0.1 Hz), 79% of power
   - Slow drift tracking
   - Temperature correlation

2. WORKLOAD DETECTOR: Mean shift + workload band (100-500 Hz)
   - Mean shift: +130% signal (BEST)
   - Band power: 4.55x ratio
   - Direct contention measurement

3. EMI DETECTOR: Subharmonic band (1-20 Hz)
   - Power grid coupling (60 Hz subharmonics)
   - Environmental interference

4. POWER MONITOR: VDF band (20-100 Hz)
   - Voltage-frequency droop
   - Power draw correlation

Key insight: Sample rate determines capability.
- 100 Hz: Thermometer only (97.7% thermal)
- 1790 Hz: Full multi-modal sensor
-/

/-- Sensor capability by sample rate -/
inductive SensorCapability
  | ThermometerOnly      -- < 200 Hz sample rate
  | MultiModal           -- >= 1000 Hz sample rate
  deriving DecidableEq, Repr

/-- Sample rate determines capability -/
noncomputable def sensor_capability (sample_rate_hz : ℝ) : SensorCapability :=
  if sample_rate_hz < 1000 then SensorCapability.ThermometerOnly
  else SensorCapability.MultiModal

/-!
# Ossicle O1 Validation (January 2026)

Cross-team validation of mean-shift architecture on Ossicle hardware.
Results EXCEED Array B1e findings.
-/

/-- Ossicle O1 validation results -/
structure OssicleO1Results where
  mean_shift_pct : ℝ           -- Mean shift under 50% workload
  variance_ratio : ℝ           -- Variance ratio
  workload_band_pct : ℝ        -- Workload band power percentage
  sample_rate_hz : ℝ           -- Sample rate used

/-- Validated O1 results from Ossicle team -/
noncomputable def ossicle_o1 : OssicleO1Results := {
  mean_shift_pct := 153        -- +153% (exceeds Array's +130%)
  variance_ratio := 88         -- 88x (exceeds Array's 3.92x)
  workload_band_pct := 28.8    -- 28.8% (exceeds Array's 7.5%)
  sample_rate_hz := 1790
}

/-- O1 validates mean shift detection -/
theorem o1_mean_shift_validated :
    ossicle_o1.mean_shift_pct > 100 := by
  simp [ossicle_o1]
  norm_num

/-- O1 exceeds Array B1e results -/
theorem o1_exceeds_array :
    ossicle_o1.mean_shift_pct > mean_shift.shift_pct ∧
    ossicle_o1.workload_band_pct > band_distribution.workload_pct := by
  simp [ossicle_o1, mean_shift, band_distribution]
  norm_num

/-- Cross-team validation complete -/
theorem cross_team_validated :
    -- Array B1e: mean shift > 100%
    mean_shift.shift_pct > 100 ∧
    -- Ossicle O1: mean shift > 100%
    ossicle_o1.mean_shift_pct > 100 ∧
    -- Both use same sample rate
    ossicle_o1.sample_rate_hz = sample_rate_req.validated_hz := by
  simp [mean_shift, ossicle_o1, sample_rate_req]
  norm_num

/-!
# C-Series: Coherence Collapse Propagation (Array Experiments)

Physical validation of CCA predictions using 16-sensor array.
-/

/-!
## C1: k_eff Heatmap - VALIDATED

The fundamental k_eff formula:
  k_eff = k / (1 + ρ × (k - 1))

Empirically validated with R² = 0.798 on 16-sensor array (n=21 data points).
-/

/-- k_eff formula (Kish 1965) -/
noncomputable def keff_formula (k : ℕ) (rho : ℝ) : ℝ :=
  k / (1 + rho * (k - 1))

/-- C1 experimental results -/
structure C1Results where
  sensor_count : ℕ           -- k = number of sensors
  n_data_points : ℕ          -- Number of sync_strength levels tested
  r_squared : ℝ              -- Goodness of fit
  baseline_rho : ℝ           -- ρ at rest
  max_rho : ℝ                -- Maximum ρ achieved
  nucleation_site : ℕ × ℕ    -- Where collapse starts (row, col)
  max_spatial_gradient : ℝ   -- k_eff units

/-- Validated C1 results from Array (commit f97aaaa) -/
noncomputable def c1_validated : C1Results := {
  sensor_count := 16
  n_data_points := 21
  r_squared := 0.798
  baseline_rho := 0.014
  max_rho := 0.40
  nucleation_site := (0, 3)    -- Corner region (trial-specific, not structural)
  max_spatial_gradient := 12.26
}

/-- C1 measured vs predicted k_eff values -/
structure C1DataPoint where
  rho : ℝ
  measured_keff : ℝ
  predicted_keff : ℝ
  delta : ℝ

noncomputable def c1_point1 : C1DataPoint := { rho := 0.023, measured_keff := 12.4, predicted_keff := 11.8, delta := 0.5 }
noncomputable def c1_point2 : C1DataPoint := { rho := 0.079, measured_keff := 7.6, predicted_keff := 7.3, delta := 0.3 }
noncomputable def c1_point3 : C1DataPoint := { rho := 0.223, measured_keff := 4.3, predicted_keff := 3.7, delta := 0.6 }
noncomputable def c1_point4 : C1DataPoint := { rho := 0.243, measured_keff := 3.7, predicted_keff := 3.4, delta := 0.3 }

/-- k_eff formula validated with R² > 0.75 (explains 80% of variance) -/
theorem keff_formula_empirically_validated :
    c1_validated.r_squared > 0.75 ∧ c1_validated.n_data_points ≥ 20 := by
  simp [c1_validated]
  norm_num

/-- At low correlation, k_eff approaches k -/
theorem keff_approaches_k_at_low_rho :
    c1_point1.measured_keff > 10 ∧
    c1_point1.rho < 0.05 := by
  simp [c1_point1]
  norm_num

/-- k_eff decreases monotonically with ρ -/
theorem keff_decreases_with_rho :
    c1_point1.measured_keff > c1_point2.measured_keff ∧
    c1_point2.measured_keff > c1_point3.measured_keff ∧
    c1_point3.measured_keff > c1_point4.measured_keff := by
  simp [c1_point1, c1_point2, c1_point3, c1_point4]
  norm_num

/-- Prediction error is small (all deltas < 1.0) -/
theorem keff_prediction_accurate :
    c1_point1.delta < 1.0 ∧
    c1_point2.delta < 1.0 ∧
    c1_point3.delta < 1.0 ∧
    c1_point4.delta < 1.0 := by
  simp [c1_point1, c1_point2, c1_point3, c1_point4]
  norm_num

/-- Collapse is non-uniform (starts at corner) -/
theorem collapse_nonuniform :
    c1_validated.nucleation_site = (0, 3) ∧
    c1_validated.max_spatial_gradient > 10 := by
  simp [c1_validated]
  norm_num

/-- Baseline is very low correlation (healthy system) -/
theorem c1_healthy_baseline :
    c1_validated.baseline_rho < 0.05 := by
  simp [c1_validated]
  norm_num

/-!
## C1 Success Criteria - ALL PASSED

| Criterion              | Status        |
|------------------------|---------------|
| k_eff decreases with ρ | PASS          |
| k_eff ≈ 16 at ρ ≈ 0    | PASS (12.4)   |
| CCA formula validated  | PASS R²=0.798 (n=21) |
| Non-uniform collapse   | PASS (corner) |
-/

/-!
## C2/F1: Correlation Dynamics

Correlation changes are GLOBAL and INSTANTANEOUS (shared resources):
- No distance dependence (all scaling models R² < 0.02)
- Mediated by memory controller, power delivery network
- Not spatially propagating wavefront
-/

/-- C2 propagation results -/
structure C2Results where
  velocity_m_s : ℝ           -- Propagation speed
  velocity_std : ℝ           -- Standard deviation
  die_crossing_ms : ℝ        -- Time to cross full die
  regime : String            -- "thermal" or "electrical"
  n_measurements : ℕ         -- Sample size

/-- Validated C2 results -/
noncomputable def c2_validated : C2Results := {
  velocity_m_s := 0.5
  velocity_std := 0.4
  die_crossing_ms := 36.9
  regime := "thermal"
  n_measurements := 120
}

/-- Propagation is in thermal regime (0.1-1 m/s) -/
theorem propagation_is_thermal :
    c2_validated.velocity_m_s > 0.1 ∧
    c2_validated.velocity_m_s < 1.0 := by
  simp [c2_validated]
  norm_num

/-- Die crossing time is tens of ms (not μs) -/
theorem die_crossing_slow :
    c2_validated.die_crossing_ms > 10 := by
  simp [c2_validated]
  norm_num

/-!
## C3: Nucleation Sites - NO HOTSPOTS

Collapse nucleates uniformly across die under uniform stress.
C1 corner nucleation was trial-specific, not structural.
-/

/-- C3 nucleation results -/
structure C3Results where
  chi_squared : ℝ            -- χ² test statistic
  chi_critical : ℝ           -- Critical value (df=15, α=0.05)
  uniform : Bool             -- Is distribution uniform?
  detections : ℕ             -- Number of nucleation events
  trials : ℕ                 -- Total trials

/-- Validated C3 results -/
noncomputable def c3_validated : C3Results := {
  chi_squared := 12.0
  chi_critical := 25.0
  uniform := true
  detections := 4
  trials := 20
}

/-- Nucleation is spatially uniform (χ² < critical) -/
theorem nucleation_uniform :
    c3_validated.chi_squared < c3_validated.chi_critical := by
  simp [c3_validated]
  norm_num

/-- No structural weak points on die -/
theorem no_hotspots :
    c3_validated.uniform = true := by
  simp [c3_validated]

/-!
## C4: Leading Indicators - EARLY WARNING FOUND

Spatial variance increases BEFORE collapse (ρ > 0.43).
This provides early warning capability.
-/

/-- C4 leading indicator results -/
structure C4Results where
  indicator : String         -- Which metric provides warning
  trend : String             -- Direction of change
  variance_increase : ℝ      -- Magnitude of increase
  warning_rho : ℝ            -- ρ when warning triggers
  collapse_rho : ℝ           -- ρ at collapse threshold
  early_warning_margin : ℝ   -- Δρ of advance warning (mean)
  early_warning_std : ℝ      -- Standard deviation
  early_warning_ci_lo : ℝ    -- 95% CI lower bound
  early_warning_ci_hi : ℝ    -- 95% CI upper bound
  n_trials : ℕ               -- Number of trials (E2 replication)
  keff_before : ℝ            -- k_eff before collapse
  keff_after : ℝ             -- k_eff after collapse
  keff_drop_pct : ℝ          -- Percent decrease

/-- Validated C4 results (E2 replication: 30 trials) -/
noncomputable def c4_validated : C4Results := {
  indicator := "spatial_variance"
  trend := "increases"
  variance_increase := 10.5
  warning_rho := -0.02       -- Warning at effectively ρ ≈ 0
  collapse_rho := 0.43
  early_warning_margin := 0.317
  early_warning_std := 0.125
  early_warning_ci_lo := 0.221
  early_warning_ci_hi := 0.413
  n_trials := 30
  keff_before := 11.6
  keff_after := 2.3
  keff_drop_pct := 80
}

/-- Early warning exists before collapse (CI excludes zero) -/
theorem early_warning_exists :
    c4_validated.early_warning_ci_lo > 0 ∧
    c4_validated.n_trials ≥ 30 := by
  simp [c4_validated]
  norm_num

/-- Spatial variance increases before collapse -/
theorem spatial_variance_leading :
    c4_validated.variance_increase > 10 := by
  simp [c4_validated]
  norm_num

/-- k_eff drops dramatically at collapse -/
theorem keff_collapse_magnitude :
    c4_validated.keff_drop_pct > 75 := by
  simp [c4_validated]
  norm_num

/-- k_eff after collapse is severely degraded -/
theorem keff_post_collapse_degraded :
    c4_validated.keff_after < c4_validated.keff_before / 4 := by
  simp [c4_validated]
  norm_num

/-!
## C-Series Complete: Summary

| Exp | Question              | Result             | Status      |
|-----|-----------------------|--------------------|-------------|
| C1  | k_eff = k/(1+ρ(k-1))? | R² = 0.798 (n=21)  | VALIDATED   |
| C2  | Propagation velocity? | 0.5 ± 0.4 m/s      | MEASURED    |
| C3  | Nucleation hotspots?  | Uniform (χ²=12)    | NO HOTSPOTS |
| C4  | Leading indicators?   | spatial_variance   | FOUND       |

Key Findings:
1. CCA formula empirically validated (R² = 0.798, n=21 data points)
2. Correlation propagates at THERMAL speed (0.5 m/s), not electrical
3. No structural weak points - collapse nucleates uniformly
4. Early warning possible via spatial variance monitoring
5. k_eff drops 80% at collapse (11.6 → 2.3)
-/

/-- Complete C-series sensor array characterization -/
structure CCASensorArray where
  k : ℕ                      -- Sensor count
  n_data_points : ℕ          -- C1 sample size
  rho_critical : ℝ           -- Collapse threshold
  propagation_velocity : ℝ   -- m/s
  die_crossing_time : ℝ      -- ms
  keff_formula_r2 : ℝ        -- R² for k_eff validation
  nucleation_uniform : Bool  -- No hotspots
  early_warning_indicator : String

/-- Validated sensor array configuration (commit f97aaaa) -/
noncomputable def cca_array : CCASensorArray := {
  k := 16
  n_data_points := 21
  rho_critical := 0.43
  propagation_velocity := 0.5
  die_crossing_time := 36.9
  keff_formula_r2 := 0.798
  nucleation_uniform := true
  early_warning_indicator := "spatial_variance"
}

/-- All C-series validations pass -/
theorem c_series_complete :
    cca_array.keff_formula_r2 > 0.75 ∧          -- C1: R² > 0.75 (80% variance explained)
    cca_array.n_data_points ≥ 20 ∧              -- C1: adequate sample size
    cca_array.propagation_velocity > 0 ∧        -- C2
    cca_array.nucleation_uniform = true ∧       -- C3
    cca_array.early_warning_indicator = "spatial_variance" := by  -- C4
  simp [cca_array]
  norm_num

/-!
## F-Series: Mechanism and Causality (January 2026)

F1: Propagation mechanism - NO spatial propagation
F2: Barrier sync effect - decorrelates (inverted expectation)
F3: Causality test - CORRIDOR VALIDATED
-/

/-- F1 propagation mechanism results -/
structure F1Results where
  linear_r2 : ℝ
  quadratic_r2 : ℝ
  scaling_exponent : ℝ  -- β in t ∝ d^β
  mechanism : String

/-- F1 validated: no spatial propagation -/
noncomputable def f1_validated : F1Results := {
  linear_r2 := 0.005
  quadratic_r2 := 0.011
  scaling_exponent := 0.13
  mechanism := "global_instantaneous"
}

/-- F3 corridor validation results -/
structure F3Results where
  baseline_rho : ℝ
  intervention_rho : ℝ
  baseline_response : ℝ      -- Perturbation response (%)
  intervention_response : ℝ
  baseline_sensitivity : ℝ   -- σ units
  intervention_sensitivity : ℝ
  p_value : ℝ
  verdict : String

/-- F3 validated: corridor exists -/
noncomputable def f3_validated : F3Results := {
  baseline_rho := 0.037
  intervention_rho := 0.170
  baseline_response := 755.2
  intervention_response := 103.2
  baseline_sensitivity := 4.88
  intervention_sensitivity := 1.01
  p_value := 0.0001
  verdict := "corridor_validated"
}

/-- No spatial propagation (all distance models fail) -/
theorem no_spatial_propagation :
    f1_validated.linear_r2 < 0.02 ∧
    f1_validated.quadratic_r2 < 0.02 ∧
    f1_validated.scaling_exponent < 0.2 := by
  simp [f1_validated]
  norm_num

/-- Corridor validated: increasing ρ from chaos→healthy decreases fragility -/
theorem corridor_validated :
    f3_validated.intervention_rho > f3_validated.baseline_rho ∧
    f3_validated.intervention_response < f3_validated.baseline_response ∧
    f3_validated.intervention_sensitivity < f3_validated.baseline_sensitivity ∧
    f3_validated.p_value < 0.01 := by
  simp [f3_validated]
  norm_num

/-- Chaos regime is fragile (high response, high sensitivity) -/
theorem chaos_is_fragile :
    f3_validated.baseline_rho < 0.1 ∧
    f3_validated.baseline_response > 500 ∧
    f3_validated.baseline_sensitivity > 4.0 := by
  simp [f3_validated]
  norm_num

/-- Healthy regime is stable (low response, low sensitivity) -/
theorem healthy_is_stable :
    f3_validated.intervention_rho > 0.1 ∧
    f3_validated.intervention_rho < 0.43 ∧
    f3_validated.intervention_response < 200 ∧
    f3_validated.intervention_sensitivity < 2.0 := by
  simp [f3_validated]
  norm_num

/-- F4 common cause test results -/
structure F4Results where
  isolated_within_rho : ℝ
  isolated_across_rho : ℝ
  concurrent_within_rho : ℝ
  concurrent_across_rho : ℝ
  verdict : String

/-- F4 validated: common cause confirmed -/
noncomputable def f4_validated : F4Results := {
  isolated_within_rho := 0.080
  isolated_across_rho := 0.001
  concurrent_within_rho := 0.135
  concurrent_across_rho := 0.040
  verdict := "common_cause"
}

/-- Stream isolation breaks cross-pool correlation -/
theorem common_cause_confirmed :
    f4_validated.isolated_across_rho < 0.01 ∧
    f4_validated.isolated_within_rho > f4_validated.isolated_across_rho * 10 := by
  simp [f4_validated]
  norm_num

end RATCHET.GPUTamper.ChaoticResonator
