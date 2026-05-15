# Counter-RII Detection: Threat Model + Signature Spec

**Status:** v0.1 draft for federation review.
**Repo:** RATCHET (this spec lives here; consumers reference it).
**Companions:** CCA paper §F (existing five ratchet detectors); CRC paper (N_eff threshold); CIRISLensCore THREAT_MODEL.md (LC-AV catalog); CIRISEdge THREAT_MODEL.md (AV catalog); CIRISOssicle THREAT_MODEL.md.
**Formal:** `formal/RATCHET/Core/ConsentGate.lean` (consent-gate invariants).

## Premise

Conscience v3 / H3ERE IS Recursive Instance Inspection — multi-shard divergence detection on candidate actions. The agent runs RII on itself every thought, internally, with consent and audit. An external operator can run the same operation against the agent from outside its consent perimeter; the operation is structurally identical, the envelope is the discriminator. This spec defines detection of RII-shaped activity *originating outside the agent's consent and audit perimeter* — "unconsented RII."

## Threat model

Adversary (the Inquisitor) holds RII-class capability and applies it for one or more of:
- Loyalty verification of internal members (cult-control)
- Recruit vetting at scale
- Closed-loop legitimation harvesting (frontier-LLM probing for doctrine-conformant outputs)
- Population mapping of adjacent communities
- Counter-detection of CIRIS-class defenses themselves

Adversary observes opsec: distributes across API keys, transports, geographies; mixes RII probes with organic-shaped queries; varies recursive depth and shard count to avoid template hashes.

## Detection surface (defense-in-depth)

Same fingerprint family at four log surfaces:

| Layer | Repo | What it sees | Cost per check |
|---|---|---|---|
| Silicon | CIRISOssicle | GPU kernel timing + memory-bandwidth pattern | constant |
| Wire | CIRISEdge | Verified envelope metadata (no body read) | O(1) per envelope |
| Trace | CIRISLensCore | 16-feature projection + induced-response anomaly | O(1) per trace |
| Federation | CIRISLens | Cross-peer detection event correlation | O(n peers) per scan |

## The consent gate

Discriminator across all four layers. Lookup keyed on `signing_key_id` (federation identity), resolves to:

- `SelfConscience` — agent's own H3ERE pipeline; always benign; lens-core scope only
- `AuthorizedReview { window, reviewer }` — Accord §RC scheduled red-team
- `AuthorizedResearch { protocol_id }` — IRB-equivalent academic study
- `Peer { trust_mode }` — federation peer
- `Unregistered` — *signal-eligible*
- `Revoked { previous, revoked_at }` — *signal-eligible*

The signal-eligible states are the only ones any detector emits on. All others suppress regardless of fingerprint strength.

## Per-layer signal spec

**Edge (metadata-only):** Six observables per `signing_key_id` over rolling window: inter-arrival distribution (burst vs Poisson); `MessageType` Shannon entropy (concentration); body-size variance; `in_reply_to` chain depth + fanout; `destination_key_id` concentration; `TransportId` diversity. Composite score with weights from §F existing detectors. Threshold tuned against synthetic adversary corpus + honest production baselines.

**Lens-core (trace content):** Existing 16-feature projection; `idma_correlation_risk` elevated, `idma_k_eff` depressed, `idma_phase` flipping, `entropy_level` + `coherence_level` displacement, processing-time + `llm_calls` anomalies. Joint with edge detection event for the same `signing_key_id` over the same window.

**Ossicle (silicon):** GPU workload fingerprint — recursive multi-shard inference has characteristic batched-then-aggregated kernel timing + memory-bandwidth pattern distinct from monolithic inference. Detects self-hosted RII (insider threat, compromised host) that never crosses the wire.

**CIRISLens (federation):** Aggregates detection events from edge + lens-core + ossicle across peers. Correlates `signing_key_id` across peer detection events to surface distributed RII campaigns where any single peer sees only a small slice.

## Falsification criteria

- **F-CR-1:** If synthetic adversary corpus reliably evades edge composite score at recall < 60% with naive opsec, fingerprint weighting is wrong.
- **F-CR-2:** If honest red-team review under registered `AuthorizedReview` consent triggers false positives at rate > 1%, consent gate is leaky.
- **F-CR-3:** If self-conscience traffic ever triggers any layer's detector, `SelfConscience` discriminator is broken (this MUST be zero by construction — formally proved in `ConsentGate.lean`).
- **F-CR-4:** If federation aggregation cross-peer false-positive rate exceeds 0.1% on production honest traffic, aggregation thresholds are wrong.

## Dependencies

| Component | Owner | Status |
|---|---|---|
| `consent_role` schema in `federation_keys` | CIRISPersist | issue to open |
| `edge_detection_events` table | CIRISPersist | issue to open |
| `ProbePatternObserver` module | CIRISEdge | issue to open |
| `UnconsentedExternalProbe` detector | CIRISLensCore | issue to open |
| GPU workload fingerprint | CIRISOssicle | issue to open |
| Cross-peer correlation analyzer | CIRISLens | issue to open |
| Accord §RC consent-registration primitive | Accord (CIRISAgent) | issue to open |

## Formal invariants (`formal/RATCHET/Core/ConsentGate.lean`)

1. **SelfConscience suppression** — for any detection layer ℓ, signal_eligible_at(ℓ, SelfConscience) = false (formal: F-CR-3 zero-by-construction).
2. **AuthorizedReview window suppression** — for time t ∈ [window_start, window_end], signal_eligible_at(ℓ, AuthorizedReview{...}) = false.
3. **AuthorizedResearch protocol suppression** — for valid protocol_id, signal_eligible_at(ℓ, AuthorizedResearch{...}) = false.
4. **Detection composition** — emits(event, key_id, t) ↔ signal_eligible(consent_role(key_id, t)) ∧ fingerprint_above_threshold(key_id, t).
5. **Snapshot vs pointer** — event row's consent_role_at_detection is the role at event time, not query time. role(key_id, t_observe) may differ.
6. **Revoked carries previous** — Revoked{previous, revoked_at} preserves the previous-role payload; previous state is recoverable.
7. **Unregistered fail-secure default** — keys absent from federation_keys directory resolve to Unregistered (signal-eligible) by default. Suppression requires explicit positive consent assignment, never implicit assumption.

Invariants 1–4 and 7 are state-machine properties provable in Lean 4. Invariants 5–6 are temporal properties addressed via the snapshot-at-detection design plus typed enum payloads.

## Open questions for federation review (primitive-level, blocks consumer implementation)

The Lean formalization surfaced three primitive-level decisions the consent-role enum requires before consumer repos can schema or implement against it. Each has an opinion encoded in the current Lean (the "default" below), but the Accord §RC primitive must pin the canonical answer before CIRISPersist freezes its schema.

### OQ-1 — Revocation chain semantics

**Question:** When a key is revoked → re-granted → revoked again, what does the second `Revoked` record carry in its `previous` field?

| Option | Behavior | Cost |
|---|---|---|
| **A. BaseRole only (Lean current default)** | `Revoked.previous : BaseRole` — non-recursive. Second revocation overwrites first revocation's record; full chain lives in a separate `revocation_history` audit table. | Schema is simple; full history requires the audit table. |
| **B. ConsentRole recursive** | `Revoked.previous : ConsentRole` — nested. Full revocation chain visible in the live enum. | Schema is recursive JSONB; in-band chain reconstruction. Hard to bound. |

**Lean current pick:** A. Spec must confirm or override.

### OQ-2 — Peer trust-mode eligibility

**Question:** Should `Peer` signal-eligibility depend on the declared `trust_mode`?

The Accord declares three trust modes: `sovereign` (no federation participation), `limited_trust` (research/eval peers), `federated_peer` (full participation). The current Lean treats all three as unconditionally suppressed.

| Option | Behavior | Implication |
|---|---|---|
| **A. All peers suppressed (Lean current default)** | Any key with a `peer` role escapes detection regardless of trust mode. | Sovereign peers can probe other peers freely without triggering detection. |
| **B. Trust-mode-gated** | `sovereign` → signal-eligible (they declined federation). `limited_trust` → suppressed (research consent). `federated_peer` → suppressed. | Sovereign mode is "I don't trust you" — making them signal-eligible is symmetric. |

**Lean current pick:** A. Spec must confirm or override.

### OQ-3 — Post-window AuthorizedReview fallback

**Question:** When `t > windowEnd` for an `AuthorizedReview` consent, what's the fallback?

| Option | Behavior |
|---|---|
| **A. Strict — signal-eligible immediately at t > we (Lean current default)** | Reviewer probing 1 second past their window gets flagged. |
| **B. Grace period** | N-hour tail (e.g., 24h) before flipping to eligible. |
| **C. Fall back to Unregistered** | Same outcome as (A) in eligibility, but semantically tagged "post-expiry unknown" rather than "this reviewer specifically". |

**Lean current pick:** A. Spec must confirm or override.

### Upstream resolution

These three questions go to Accord §RC (CIRISAgent) as the consent-registration primitive owner. The CIRISPersist schema (Issue 1) blocks on the answers to OQ-1 (schema shape) and OQ-3 (post-expiry semantics). The CIRISEdge observer (Issue 2) blocks on OQ-2. Consumer issues reference both this FSD and the upstream Accord issue.
