/-
RATCHET: AgencyRung — substrate-fractality agency ladder

This module defines the agency-ladder ordering used by Exp 2's
fractal-across-agency predictions (P2, P3) and the Counter-RII consent
work. Same type appears in three places that were previously informal:

  1. experiments/exp2_cross_substrate/REGIME.md (agency ladder prose)
  2. RATCHET/Core/ConsentGate.lean (consent semantics, currently
     unconnected to agency formally)
  3. RATCHET/Experiments/Exp2Predictions.lean (substrate-fractality
     predictions)

**Operationalization commitment (intrinsic, not extrinsic):**

The agency rung of a substrate is assigned from CONSTITUENT-INTRINSIC
properties — goal representation, planning horizon, behavioral
repertoire — measured BEFORE any ρ, σ, or residual structure is observed.

This commitment is load-bearing. P2 in Exp 2 predicts that residual
structure correlates with agency. If we read agency *off* residual
structure, P2 becomes circular and unfalsifiable. By encoding the
intrinsic assignment in the structure type's fields, we lock the
non-circularity at the formal layer.

The `AgencyProfile` structure has NO fields that can be derived from
post-measurement outcomes. Adding such a field is a spec change
requiring an amendment.
-/

import Mathlib.Logic.Basic
import Mathlib.Tactic.Linarith

namespace RATCHET.Agency

/-! ## The ladder -/

/--
Substrate agency rungs. Ordered A0 (lowest) → A5 (highest).

* A0 — inert or engineered. Constituents have no goal representation
  (batteries, PMU sensors, AlphaFold residues).
* A1 — homeostatic / cellular signaling. Constituents respond to
  stimuli but lack goal-directed planning (microbiome, neurons).
* A2 — population dynamics. Constituents have life-cycle goals;
  populations aggregate (ecological species).
* A3 — goal-directed reasoning. Constituents have explicit goal
  representation + planning over multiple steps (LLM agents).
* A4 — full human agency. Constituents are humans with
  metacognition and recursive goal revision (institutions).
* A5 — civilizational coupling. Recursive aggregation of A4
  constituents (currently conjecture-only; no empirical anchor).
-/
inductive AgencyRung
  | A0
  | A1
  | A2
  | A3
  | A4
  | A5
  deriving DecidableEq, Repr

/-- Total ordering on the ladder. -/
def AgencyRung.toNat : AgencyRung → ℕ
  | .A0 => 0
  | .A1 => 1
  | .A2 => 2
  | .A3 => 3
  | .A4 => 4
  | .A5 => 5

instance : LE AgencyRung := ⟨fun a b => a.toNat ≤ b.toNat⟩
instance : LT AgencyRung := ⟨fun a b => a.toNat < b.toNat⟩

instance : DecidableRel ((· ≤ ·) : AgencyRung → AgencyRung → Prop) :=
  fun a b => inferInstanceAs (Decidable (a.toNat ≤ b.toNat))

instance : DecidableRel ((· < ·) : AgencyRung → AgencyRung → Prop) :=
  fun a b => inferInstanceAs (Decidable (a.toNat < b.toNat))

/-! ## Consent requirement — the moral-weight asymmetry -/

/--
A substrate requires explicit consent infrastructure iff its agency
rung is A3 or higher. This is the formal version of "the same Kish-
formula collapse is benign at A0–A2 (pure structural disintegration)
but a consent violation at A3+."

Below A3, constituents have no agency to violate. At A3+, the same
ρ → 1 dynamic that would be a phase transition at A1 becomes
coercion / capture / cult-formation — the moral weight scales with
constituent agency.
-/
def consentRequired : AgencyRung → Bool
  | .A0 => false
  | .A1 => false
  | .A2 => false
  | .A3 => true
  | .A4 => true
  | .A5 => true

/--
**Theorem (moral-weight asymmetry):** consentRequired is true iff
the rung is A3 or higher.

This is the formal pin for the agency-conditional reading of CCA's
collapse dynamics. Counter-RII (FSD/COUNTER_RII_DETECTION.md) is
load-bearing exactly where this predicate is true.
-/
theorem consent_required_iff_rung_ge_A3 (r : AgencyRung) :
    consentRequired r = true ↔ AgencyRung.A3 ≤ r := by
  cases r <;> decide

/-! ## Intrinsic operationalization — circularity protection -/

/--
Constituent-intrinsic properties used to assign an agency rung to a
substrate. **All fields are measurable BEFORE any ρ, σ, or residual
observation.** This prevents circularity with Exp 2's P2 prediction
(which says residual structure correlates with agency).

Fields:
* `goalRepresentationBits` — information-theoretic measure of
  goal-state representation in a single constituent. 0 = no goal
  representation (inert chemistry). Nonzero = explicit goal state.
* `planningHorizonSteps` — how many steps of forward planning a
  constituent's behavior reflects. 0–1 = reactive only.
* `behavioralRepertoireSize` — the cardinality of distinct
  behavioral options a constituent can select among. 1 = no
  choice (purely determined by physics).

These three fields are deliberately under-specified at exact
empirical thresholds. The point isn't to nail down "exactly when
horizon=42 becomes A3"; the point is that the assignment is from
constituent-level facts, not from outcome statistics.
-/
structure AgencyProfile where
  goalRepresentationBits   : ℕ
  planningHorizonSteps     : ℕ
  behavioralRepertoireSize : ℕ
  deriving Repr

/--
A canonical (illustrative, not normative) mapping from
intrinsic-profile to rung. Real assignments per substrate are
pre-registered in `data_sources.yaml`; this function provides a
fallback / default classifier.

Thresholds chosen to be "obviously generous" in both directions —
the goal is to make the boundaries cleanly inside-or-outside, not
to be empirically precise. Exact thresholds for any specific
substrate are pre-registered separately.
-/
def AgencyProfile.inferRung (p : AgencyProfile) : AgencyRung :=
  if p.goalRepresentationBits = 0 ∧ p.planningHorizonSteps ≤ 1 then .A0
  else if p.goalRepresentationBits = 0 then .A1
  else if p.planningHorizonSteps ≤ 10 ∧ p.behavioralRepertoireSize < 1000 then .A2
  else if p.planningHorizonSteps ≤ 100 then .A3
  else if p.behavioralRepertoireSize < 10000 then .A4
  else .A5

/-! ## Sanity checks -/

/-- An all-zero profile (no goals, no planning, no choice) maps to A0. -/
theorem A0_for_inert :
    AgencyProfile.inferRung ⟨0, 0, 1⟩ = AgencyRung.A0 := by
  simp [AgencyProfile.inferRung]

/-- A profile with goals + long horizon + large repertoire is at least A3. -/
theorem A3_floor_for_planning :
    AgencyRung.A3 ≤ AgencyProfile.inferRung ⟨32, 50, 5000⟩ := by
  decide

/-- Ladder is reflexive. -/
theorem rung_refl (r : AgencyRung) : r ≤ r := by
  cases r <;> decide

/-- A0 is the bottom of the ladder. -/
theorem A0_min (r : AgencyRung) : AgencyRung.A0 ≤ r := by
  cases r <;> decide

/-- A5 is the top of the ladder. -/
theorem A5_max (r : AgencyRung) : r ≤ AgencyRung.A5 := by
  cases r <;> decide

end RATCHET.Agency
/-
| Item                          | What it locks                          |
|-------------------------------|----------------------------------------|
| `AgencyRung` type             | 6-rung ladder, total ordering          |
| `consentRequired`             | Boolean predicate; true iff rung ≥ A3  |
| `consent_required_iff_rung_ge_A3` | Moral-weight asymmetry as theorem    |
| `AgencyProfile`               | Intrinsic-only fields (3 measurables)  |
| `AgencyProfile.inferRung`     | Default classifier from profile fields |
| Circularity protection        | All AgencyProfile fields are intrinsic |

What's NOT in this module:
  - Per-substrate empirical rung assignments (pre-registered separately)
  - Exact threshold values for the inferRung classifier (illustrative)
  - The mapping from constituent properties to physical measurables
    (substrate engineers' work — they assign profiles before
    pre-registration locks rungs)
-/
