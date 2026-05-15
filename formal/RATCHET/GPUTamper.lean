/-
RATCHET: GPU-Based Local State Detector — module index

This file is a thin re-export pointer. The actual formal proofs of the
GPU strain-gauge / chaotic-resonator validation live in:

    RATCHET/GPUTamper/EnvironmentalCoherence.lean

Validated findings from null hypothesis testing (January 2026):
- Tamper/workload detection via k_eff mean (p = 0.007)
- Thermal sensing via k_eff variance (r = -0.97)
- Sensitivity prediction via r_ab (r = -0.999)
- Classical dynamics confirmed (LGI K₃ = 1.0)
- Not quantum, not environmental sensor, not fast-signal sensitive

Optimal Parameters:
- Coupling: ε = 0.003 (562× signal improvement)
- Thermalization: τ = 12.8s
- Scaling: τ ∝ ε^(-1.06)

See `RATCHET/GPUTamper/EnvironmentalCoherence.lean` for the namespace
RATCHET.GPUTamper.ChaoticResonator with the actual theorem statements
and proofs.

(Earlier versions of this file contained theorem wrappers under a
`LocalStateDetector` namespace that was never defined elsewhere —
removed in the v0.4 lake cleanup. Anyone referencing the prior
`RATCHET.GPUTamper.validated_*` theorems should import the
EnvironmentalCoherence module directly.)

Author: CIRIS Research Team
Date: January 2026 (rewritten May 2026)
-/

import RATCHET.GPUTamper.EnvironmentalCoherence

namespace RATCHET.GPUTamper

end RATCHET.GPUTamper
