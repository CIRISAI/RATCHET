/-
RATCHET: Reference Architecture for Testing Coherence and Honesty in Emergent Traces

This is the main entry point for RATCHET's formal verification.

GROUNDED COMPONENTS (with citations):
- EffectiveConstraints: k_eff formula (Kish 1965)
- ExplosiveSynchronization: ES-proximity phase theory (PNAS 2025)
- CausalInfluence: Granger causality (Granger 1969)
- FacultyComposition: Veto logic and threshold nesting (axiomatic exploration)

CCA PEER REVIEW THEOREMS (2026):
- SusceptibilityEarlyWarning: χ = N × Var(r) divergence
- InterventionParadox: Near-critical paradoxical dynamics
- S3Stabilization: Agent type effects on stability
- AdaptiveWeighting: α(χ) smooth transition function

See RATCHET/Core/ for core implementations.
See RATCHET/CCA/ for CCA paper theorems.
-/

import RATCHET.Core.EffectiveConstraints
import RATCHET.Core.WeightedDiversity
import RATCHET.Core.ExplosiveSynchronization
import RATCHET.Core.CausalInfluence
import RATCHET.Core.FacultyComposition
import RATCHET.CCA
