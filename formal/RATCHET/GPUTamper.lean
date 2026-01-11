/-
RATCHET: GPU-Based Local State Detector

Validated findings from null hypothesis testing (January 2026).

## What It Is
A GPU-based chaotic oscillator that detects LOCAL state changes:
- Tamper/workload detection via k_eff mean (p = 0.007)
- Thermal sensing via k_eff variance (r = -0.97)
- Sensitivity prediction via r_ab (r = -0.999)

## What It Is NOT
- NOT an environmental sensor (cross-device correlation is algorithmic)
- NOT quantum (LGI K₃ = 1.0, exactly classical bound)
- NOT sensitive to fast signals (88.6% power < 0.5 Hz)

## Optimal Parameters
- Coupling: ε = 0.003 (562× signal improvement)
- Thermalization: τ = 12.8s
- Scaling: τ ∝ ε^(-1.06)

See EnvironmentalCoherence.lean for formal proofs of validated claims.

Author: CIRIS Research Team
Date: January 2026
-/

import RATCHET.GPUTamper.EnvironmentalCoherence

namespace RATCHET.GPUTamper

open LocalStateDetector

/-!
# Validated Detection Capabilities
-/

/-- Tamper detection is statistically significant (p < 0.05) -/
theorem validated_tamper_detection : tamper_detection.p_value_or_correlation < 0.05 :=
  tamper_detection_significant

/-- Thermal correlation is strong (|r| > 0.9) -/
theorem validated_thermal_sensing : thermal_sensing.p_value_or_correlation < -0.9 :=
  thermal_correlation_strong

/-- Sensitivity prediction is near-perfect (|r| > 0.99) -/
theorem validated_sensitivity_prediction : sensitivity_prediction.p_value_or_correlation < -0.99 :=
  sensitivity_prediction_excellent

/-!
# Classical Dynamics Confirmed
-/

/-- System obeys classical bound (K₃ ≤ 1) -/
theorem validated_classical : lgi_result.K3 ≤ classical_bound :=
  system_is_classical

end RATCHET.GPUTamper
