/-
RATCHET: Consent-Gate Invariants for Counter-RII Detection

Formal companion to `FSD/COUNTER_RII_DETECTION.md`.

The Counter-RII detection architecture composes a fingerprint score with a
consent gate. The fingerprint identifies RII-shaped activity at one of four
log surfaces (silicon / wire / trace / federation). The consent gate
decides whether activity originating from a given signing_key_id is
*signal-eligible* — only Unregistered and Revoked roles pass the gate.

This file proves the invariants the FSD asserts:

  CG-1  SelfConscience traffic never triggers a detection at any layer.
  CG-2  AuthorizedReview during its time window never triggers a detection.
  CG-3  AuthorizedResearch with a valid protocol never triggers a detection.
  CG-4  Detection emission requires BOTH signal-eligibility AND fingerprint
        above threshold (veto-like composition).
  CG-5  Revoked preserves its previous role payload (no information loss).
  CG-6  Unregistered is the fail-secure default for absent directory entries.

We model the gate logic as a `Prop`-valued predicate (the proofs do not
require decidability over the reals — only the logic of suppression).
-/

import Mathlib.Data.Real.Basic
import Mathlib.Logic.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum

namespace RATCHET.ConsentGate

/-! ## Time and identity primitives -/

abbrev KeyId := String
abbrev Time := ℝ

inductive TrustMode
  | sovereign
  | limitedTrust
  | federatedPeer
  deriving DecidableEq, Repr

/-! ## Consent roles -/

/--
Base consent-role variants (the non-revoked subset). Splitting Base from
`ConsentRole` keeps `ConsentRole` non-recursive while still allowing the
Revoked variant to carry the prior role as a payload (CG-5).
-/
inductive BaseRole
  | selfConscience
  | authorizedReview (windowStart : Time) (windowEnd : Time) (reviewerKeyId : KeyId)
  | authorizedResearch (protocolId : String) (expiresAt : Time)
  | peer (trustMode : TrustMode)
  | unregistered

/-- The full consent-role enum. -/
inductive ConsentRole
  | base (r : BaseRole)
  | revoked (previous : BaseRole) (revokedAt : Time)

/-! ## Signal eligibility (as a Prop predicate) -/

/--
Eligibility for a `BaseRole` at observation time `t`. Returns a `Prop`
so that the proofs can reason about it directly without invoking
`Decidable` on the reals.

* `selfConscience` — never eligible.
* `unregistered`   — always eligible (fail-secure default).
* `authorizedReview ws we _` — NOT eligible iff `ws ≤ t ≤ we`.
* `authorizedResearch _ exp` — NOT eligible iff `t ≤ exp`.
* `peer _` — never eligible (peer status doesn't expire here;
  changes go through revocation).
-/
def baseEligibleAt : BaseRole → Time → Prop
  | .selfConscience, _ => False
  | .unregistered, _ => True
  | .authorizedReview ws we _, t => ¬ (ws ≤ t ∧ t ≤ we)
  | .authorizedResearch _ exp, t => ¬ (t ≤ exp)
  | .peer _, _ => False

/--
Eligibility at time `t` for a full `ConsentRole`. Non-recursive: the
revoked case dispatches on the carried `BaseRole` directly (it is a
`BaseRole`, not a `ConsentRole`, so no recursion is needed).
-/
def signalEligibleAt : ConsentRole → Time → Prop
  | .base r, t => baseEligibleAt r t
  | .revoked previous revokedAt, t =>
      -- Before revocation: follow the previous role.
      -- At/after revocation: signal-eligible regardless.
      if t < revokedAt then baseEligibleAt previous t else True

/-! ## Fingerprint and detection emission -/

abbrev FingerprintScore := ℝ
abbrev Threshold := ℝ

/--
The detection-emission predicate. The veto-like composition that is the
heart of Counter-RII: emit a detection event iff signal-eligible at t
AND fingerprint score exceeds the threshold.

The CRITICAL design property is the conjunction: NEITHER condition alone
is sufficient. A consented role with a high fingerprint score is
suppressed; an unconsented key with a low fingerprint score is suppressed.
-/
def emit (role : ConsentRole) (t : Time)
         (score : FingerprintScore) (τ : Threshold) : Prop :=
  signalEligibleAt role t ∧ score > τ

/-! ## Invariants -/

/-! ### CG-1: SelfConscience never triggers a detection -/

/--
**CG-1** — The agent's own self-conscience traffic never triggers a
detection at any layer, regardless of fingerprint score or observation
time. Formal version of F-CR-3.

H3ERE *is* RII at the algorithmic level, so distinguishing it from
unconsented RII requires the consent envelope, not the operation. CG-1
proves that the envelope discriminator works at the type level: there
is no input that can cause the gate to emit on SelfConscience.
-/
theorem CG_1_selfConscience_never_emits
    (t : Time) (score : FingerprintScore) (τ : Threshold) :
    ¬ emit (.base .selfConscience) t score τ := by
  simp [emit, signalEligibleAt, baseEligibleAt]

/-! ### CG-2: AuthorizedReview during window never triggers -/

/--
**CG-2** — A signing key carrying an `authorizedReview` consent never
triggers a detection during its review window `[windowStart, windowEnd]`.

Operationalizes F-CR-2: scheduled red-team review must not cause the
gate to leak. Outside the window the role falls back to eligible — a
reviewer probing past their consent becomes signal-eligible.
-/
theorem CG_2_authorizedReview_in_window_never_emits
    (ws we : Time) (reviewer : KeyId)
    (t : Time) (h_in_window : ws ≤ t ∧ t ≤ we)
    (score : FingerprintScore) (τ : Threshold) :
    ¬ emit (.base (.authorizedReview ws we reviewer)) t score τ := by
  intro ⟨h_elig, _⟩
  exact h_elig h_in_window

/-! ### CG-3: AuthorizedResearch unexpired never triggers -/

/--
**CG-3** — A signing key carrying an `authorizedResearch` consent never
triggers a detection while the consent is unexpired (`t ≤ expiresAt`).
-/
theorem CG_3_authorizedResearch_unexpired_never_emits
    (protocolId : String) (expiresAt : Time)
    (t : Time) (h_unexpired : t ≤ expiresAt)
    (score : FingerprintScore) (τ : Threshold) :
    ¬ emit (.base (.authorizedResearch protocolId expiresAt)) t score τ := by
  intro ⟨h_elig, _⟩
  exact h_elig h_unexpired

/-! ### CG-4: Detection composition (veto-like) -/

/--
**CG-4** — Emission requires BOTH signal-eligibility AND fingerprint
above threshold. The conjunction is by definition; this theorem records
the structural intent for downstream consumers.
-/
theorem CG_4_emit_iff_eligible_and_above_threshold
    (role : ConsentRole) (t : Time)
    (score : FingerprintScore) (τ : Threshold) :
    emit role t score τ ↔ (signalEligibleAt role t ∧ score > τ) := by
  rfl

/-- **CG-4 corollary** — eligibility alone is insufficient; below-threshold
    fingerprints never emit even for unregistered keys. -/
theorem CG_4_eligibility_alone_insufficient
    (role : ConsentRole) (t : Time)
    (score : FingerprintScore) (τ : Threshold)
    (h_below : score ≤ τ) :
    ¬ emit role t score τ := by
  simp [emit]
  intro _h_elig
  linarith

/-! ### CG-5: Revoked carries previous role -/

/--
**CG-5** — A `revoked` role preserves the prior base-role payload. The
forensic-provenance property: when a key is revoked, the previous role
is retained in the variant so downstream analysis can reconstruct
who-was-trusted-before-revocation.
-/
theorem CG_5_revoked_carries_previous
    (previous : BaseRole) (revokedAt : Time) :
    ∃ p ra, ConsentRole.revoked previous revokedAt = .revoked p ra ∧
            p = previous ∧ ra = revokedAt :=
  ⟨previous, revokedAt, rfl, rfl, rfl⟩

/-- **CG-5 corollary** — eligibility at the moment of revocation flips
    to true (signal-eligible) regardless of what the previous role
    permitted. -/
theorem CG_5_eligibility_flips_at_revocation
    (previous : BaseRole) (revokedAt : Time)
    (t : Time) (h_at_or_after : t ≥ revokedAt) :
    signalEligibleAt (.revoked previous revokedAt) t := by
  simp [signalEligibleAt]
  intro h_lt
  linarith

/-! ### CG-6: Unregistered is the fail-secure default -/

/--
**CG-6** — An unregistered key is always signal-eligible, regardless of
observation time. Fail-secure default: any key not explicitly registered
in the federation directory is treated as a potential probe source.

Suppression requires *explicit positive consent assignment*. There is
no implicit "innocent until known" — the framework assumes adversaries
will exploit unregistered identities, so default eligibility is the
safer-failing direction.
-/
theorem CG_6_unregistered_always_signal_eligible (t : Time) :
    signalEligibleAt (.base .unregistered) t := by
  simp [signalEligibleAt, baseEligibleAt]

/-! ## Sanity checks -/

/-- **Sanity** — an unregistered key with a score above threshold DOES emit. -/
theorem sanity_unregistered_above_threshold_emits
    (t : Time) (score : FingerprintScore) (τ : Threshold)
    (h_above : score > τ) :
    emit (.base .unregistered) t score τ := by
  refine ⟨?_, h_above⟩
  exact CG_6_unregistered_always_signal_eligible t

/-- **Sanity** — peer status alone suppresses regardless of fingerprint. -/
theorem sanity_peer_never_emits
    (mode : TrustMode) (t : Time)
    (score : FingerprintScore) (τ : Threshold) :
    ¬ emit (.base (.peer mode)) t score τ := by
  simp [emit, signalEligibleAt, baseEligibleAt]

end RATCHET.ConsentGate
/-
| Invariant            | Statement                                          |
|----------------------|----------------------------------------------------|
| CG-1                 | SelfConscience never emits                         |
| CG-2                 | AuthorizedReview in window never emits             |
| CG-3                 | AuthorizedResearch unexpired never emits           |
| CG-4                 | Emission iff eligible AND above-threshold          |
| CG-4-cor             | Below-threshold never emits                        |
| CG-5                 | Revoked carries previous payload                   |
| CG-5-cor             | Eligibility flips to true at revocation time       |
| CG-6                 | Unregistered always signal-eligible                |
| sanity_unreg_emits   | Unregistered + above-threshold DOES emit (nontrivial)|
| sanity_peer_no_emit  | Peer status alone suppresses                       |

What this does NOT prove (by design, out of formal scope):
- Fingerprint recall/precision (F-CR-1 — empirical, synthetic adversary corpus).
- Consent-gate leak rate on honest production traffic (F-CR-2 — empirical).
- Federation aggregation cross-peer false-positive rate (F-CR-4 — empirical).
- Cross-peer correlation soundness (statistical, not formal-deductive).

The Lean proofs establish the SUFFICIENCY of the gate's logic — given
correctly-typed inputs, the suppression rules behave as advertised. They
cannot establish NECESSITY without modeling actual adversarial behavior,
which is what the synthetic adversary corpus + production calibration
are for.
-/
