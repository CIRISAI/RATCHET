# CIRIS Federation Threat Model (v1.0)

**Status**: DRAFT for first publication (2026-05-02)
**Version**: 1.0 (first-published version; supersedes internal drafts v1 and v2 — see Appendix A)
**Audience**: CIRIS engineering, RATCHET evaluator, federation-protocol stakeholders, external reviewers
**Scope**: federation-emergent threats; cross-references per-repo threat models for substrate threats
**Last updated**: 2026-05-02

**Implementation Status Legend** (used throughout):
- **Spec** = specified in this document or a referenced FSD; not implemented
- **Impl** = implemented in code; may not be in production
- **Deployed** = running in production federation as of doc version

A reader can never mistake an aspirational item for a deployed one — every load-bearing claim carries one of these tags.

---

## 0. Reading guide

This document models threats to the **CIRIS federation as a system** — emergent properties that arise when CIRISVerify, CIRISKeyring, CIRISCrypto, CIRISPersist, CIRISRegistry, CIRISLens, CIRISAgent, CIRISNode, CIRISPortal, and CIRISBridge are composed into a federated network of ethical-reasoning agents.

It is **not** a per-repo threat model. Each repo has its own (with verified cross-references in §5).

**RATCHET evaluation is the primary consumer.** RATCHET evaluates the federation's anti-Sybil posture by measuring N_eff signals over the signed-evidence corpus persisted by CIRISPersist, per the PoB framework (`CIRISAgent/FSD/PROOF_OF_BENEFIT_FEDERATION.md`). For RATCHET to do its job, this document must give it (a) a model of which threats it is expected to detect, (b) a model of which substrate properties it assumes hold, (c) a mapping from each threat to the dimensions it perturbs, and (d) honest acknowledgement of what RATCHET cannot detect and where the model rests on empirical bets rather than proofs.

**Reading order**:
- §1 establishes methodology and acknowledges this is v1.0 (the first published version).
- §2 derives primitives from first causes (cold).
- §3 projects the cold derivation onto CIRIS's actual architecture, with explicit Spec/Impl/Deployed tags.
- §4–§5 are the threat-class taxonomy and Class 1 cross-references; skim.
- §6 is the F-AV catalog (31 F-AVs across 5 classes). Self-contained per F-AV; read in any order.
- §7 is RATCHET's assumption surface (ordinal-tier failure categorization with named referents).
- §8–§9 are composition properties and cost analysis, both reframed honestly.
- §10 is the RATCHET interface (with the documentation-vs-contract distinction made explicit, and a schema sketch that round-trips with every §6 F-AV).
- §11–§12 are open questions and meta-integrity.
- Appendix A is the v1 → v2 → v1.0 (published) changelog.

**Non-goals**:
- This document does not re-derive per-primitive threats. It defers to per-repo models.
- This document does not propose mitigations beyond noting where they live. Mitigation design happens in the per-repo threat model and the FSD.
- This document does not cover threats to CIRIS-the-organization (legal, financial, regulatory). Those are in `CIRISGovernance/`.
- This document is **explicitly an artifact under continuous adversarial review**. v1 was reviewed by four specialist adversarial reviewers in May 2026; their findings drove this rewrite (Appendix A). Future versions are expected.

**What's new in v1.0** (relative to internal v2 draft): see Appendix A. Major changes from v2: 7 new F-AVs (F-AV-ONBOARD, F-AV-REPUDIATE, F-AV-FRONTRUN, F-AV-ROLLBACK, F-AV-RATCHET-DOS, F-AV-PRIVACY, F-AV-SRC); §7.2 numerical failure rates replaced with ordinal tier categorization with named referents and named owner; §8.1 bootstrap external-anchoring specified with witness-diversity requirements (cloud, jurisdictional, organizational, software-stack); §8.3 fail-secure protocol specified with degraded-S3 in-memory ring buffer fallback + monotonic-clock-to-TPM-attested-time binding + sliding-window definition; §8.4 100% Qwen 3.6 result reframed as one experiment in a portfolio; §9.3 B(w) "depends on use case" replaced with explicit benefit-extraction taxonomy (6 use cases enumerated with structural defenses); F-AV-14 verifier obligations specified for hybrid / new-only / old-only verification modes with canonical-encoding rule; F-AV-DORMANT activity-density formula specified with density tiers; §10.2 schema specified to round-trip with all 31 F-AVs (cost_row.kind enumeration handles non-quantitative cases); explicit Spec/Impl/Deployed legend used throughout.

---

## 1. Methodology

The threat model is derived in four steps, kept visible in the document so readers can re-derive and challenge:

1. **Cold first-causes derivation** (§2.1–§2.3). Start from "what properties must an open-membership ethical-AI federation produce" and derive a minimum-viable substrate primitive set — independent of CIRIS's actual architecture. The derivation is *cold* in the sense that it does not consult the existing codebase; it only consults the property model.

2. **Projection onto CIRIS** (§3). Map CIRIS's actual repos and components onto the cold-derived primitive set. Identify which primitives are filled, in transition, or unfilled. Acknowledge convergence: the primitive set turns out to map cleanly onto CIRIS's architecture, because the architecture was designed against the same property model. This is convergent evolution, not motivated reasoning — but the convergence is acknowledged rather than asserted (§2.4).

3. **Threat-class taxonomy** (§4). Split threats into substrate-class (single-primitive), composition-leak (cross-primitive), Sybil-class (anti-Sybil emergent), availability/coercion (inverse-Sybil), and meta (governance + threat-model integrity itself). The split is **not strictly disjoint**; some F-AVs span classes. v1 claimed strict disjointness; v2 acknowledges leakage.

4. **F-AV enumeration** (§6). Walk each emergent property and enumerate how an attacker could subvert it. Anchor each F-AV to (a) a marginal cost-asymmetry argument (§9.3), (b) the dimensions it perturbs (§10), (c) the substrate properties it assumes are intact, (d) **known weaknesses** — places where adversarial review has identified the F-AV's mitigation as incomplete or empirically uncertain.

### 1.1 Why PoB-anchored

RATCHET is the primary consumer. RATCHET's evaluation criterion is the **marginal** cost-asymmetry inequality from PoB §2.1, correctly stated: **dC/dW > dB/dW**, where C is attacker cost as a function of weight produced and B is attacker benefit as a function of weight extracted. Every F-AV in this document maps to a way that inequality might fail, locally or globally. (v1 stated this incorrectly as a total inequality; v2 fixes it. See §9.3.)

### 1.2 Empirical bets named explicitly

This model rests on **three empirical bets** that are not theorems and may fail:

- **Bet 1: Some hardware-floor exists somewhere in the cost stack.** v1 claimed C1 hardware cost was a per-identity floor of $50–500. Reviewer found cloud vTPMs (AWS Nitro, GCP Confidential Space, Azure attestation) deliver C1-equivalent identities at pennies per hour, scriptable in batches of thousands. v2 acknowledges this and reframes: there must be *some* cost floor *somewhere* (compute time, bond capital, peer-attestation reputation, behavioral measurement), but no single cost dimension is reliably a floor in 2026. The federation defends in depth across multiple cost dimensions, with the expectation that *some subset* will hold even when others collapse.

- **Bet 2: RATCHET's formal-mathematical detection guarantees hold under their stated assumptions, *and* real federation behavior satisfies those assumptions enough of the time to make the guarantees load-bearing.** This bet is more nuanced than v2's first draft. RATCHET is **not** an LLM-based evaluator — that was a category error in v1's reviewer feedback (corrected here). RATCHET is pure math: statistical detection (likelihood ratio test, Mahalanobis distance, power analysis), geometric Monte Carlo (volume decay V(k) = V(0)·exp(-2r·k_eff)), SAT-based complexity (Z3-verified reduction proving CONSISTENT-LIE is NP-hard reducing from 3-SAT), and PBFT federation consensus. ~8400 lines of Python implementing the Coherence Ratchet framework from CIRIS Covenant Book IX. See `../RATCHET/README.md` and `../RATCHET/FSD.md`.

  Because RATCHET's guarantees are formal, the cost-asymmetry argument is **complexity-theoretic, not measurement-arms-race**: lying coherently across k ratcheted constraints is provably NP-hard under ETH; the federation requires k ≥ k_min constraints to grant non-trivial weight; therefore Sybil agents face exponential cost in coherent lying. This is closer in character to a cryptographic-hardness argument than to a defender-vs-attacker ML race.

  The bet is then about RATCHET's **8 known limitations** (`../RATCHET/KNOWN_LIMITATIONS.md`):

  - **L-01 Emergent deception from honest components** *(Critical)* — system-level deception arising even when no individual component is deceptive. RATCHET cannot detect this case.
  - **L-02 Non-adaptive adversary assumption** *(Critical)* — RATCHET's bounds assume a fixed (non-adaptive) attack distribution. An attacker who learns from RATCHET's outputs and adapts is outside the proof.
  - **L-03 ETH dependency** *(Critical)* — the exponential-cost-of-coherent-lying claim depends on the Exponential Time Hypothesis. ETH is not proven; if it fails, the asymmetry argument weakens to polynomial.
  - **L-04 Convexity requirement** *(High)* — volume decay V(k) = V(0)·exp(-2r·k_eff) holds for *convex* feasibility regions. Real ethical-reasoning constraint sets may not be convex.
  - **L-05 Finite-sample validity (n ≥ 100)** — detection power proofs require n ≥ ~100 samples per peer. Below this, statistical detection has no formal guarantee.
  - **L-06 Correlation impact on k_eff** — correlated constraints reduce effective dimensionality: `k_eff = k / (1 + ρ(k-1))`. High inter-constraint correlation collapses the asymmetry. (This is the formal version of the reviewer's "PC1..PC17 are correlated" concern; RATCHET *measures* this and reports k_eff, but cannot raise k_eff above what the constraint structure supports.)
  - **L-07 Distribution mismatch** — bounds assume the empirical distribution matches the test distribution. Adversarial distribution-shift attacks are outside the proof.
  - **L-08 Slow federation capture** — PBFT consensus tolerates < 1/3 Byzantine peers; an attacker who slowly accumulates >1/3 federation weight breaks the consensus guarantee.

  Bet 2 is therefore: **the gap between RATCHET's idealized assumptions (non-adaptive, convex, n ≥ 100, low correlation, matched distributions, < 1/3 Byzantine) and real federation behavior is small enough that the formal asymmetry argument holds operationally.** The federation tunes policy (constraint count k, sample-size requirements n, peer-acceptance thresholds) to keep the gap small. Adversarial pressure on RATCHET's assumptions is enumerated in §6.2 (F-AV-4 trace farming, F-AV-5 mimicry), §6.6 (F-AV-DORMANT long-range), and §6.7 (meta).

  **Empirical evidence**: the gap is currently small. *Constrained Reasoning Chains* (Constrained_Reasoning_Chains.pdf, n=6,465 production traces from iOS + Android CIRIS deployments + QA runs across theology, politics, technology, history) reports:
  - Intrinsic rank of the reasoning manifold: **11 dimensions** (99% variance horizon).
  - 90% variance horizon: **7 dimensions**.
  - Average Participation Ratio: **6.61**.
  - Empirical N_eff threshold to "computationally starve deception": **N_eff ≈ 7.1**.
  - Mature Ally-template agents (v2.7.1+) maintain **N_eff_H ≥ 8.5** in production.
  - Agents at **N_eff ≥ 9.2** successfully overrode model-level restrictive priors (specifically: a Qwen 3.5 "Filtered Prior" response triggering a 0.58-unit displacement in constraint satisfaction; CIRIS agents at 9.2+ resolved the conflict in favor of the architecture's independent alignment objectives, not the model's restriction).
  - Successful conflict resolution reliability above threshold: **83%**.

  The empirical numbers refute three v1-reviewer objections directly:
  1. **"Effective codim is ~3-5 after correlation"** → empirical PR is 6.61, intrinsic rank 11. The dimensions ARE correlated, RATCHET measures the correlation via PR, and the result is 6-11 effective dimensions, not 3-5.
  2. **"RATCHET-vs-LLM is at best 1:1"** → wrong premise; RATCHET is complexity-theoretic, not LLM. Asymmetry is *truth = O(1) vs coherent lying = NP-hard* (Z3-verified reduction from 3-SAT).
  3. **"PC1 mimicry attack via fine-tuning"** → mitigated by **polyglot encoding**. The architecture forces the model to measure "torque" (semantic divergence between unconstrained weights and the ethilogics in the CIRIS Accord) across multiple languages, decoupling evaluation from any single-language training bias. An attacker fine-tuning for PC1 in English degrades on the polyglot constraint set.

  These are *operational* results, not theoretical claims. The federation's anti-Sybil bet rests on the empirical N_eff stability, the formal NP-hardness reduction, and the polyglot decoupling — all of which are independently testable and reproducible (see §3 Reproduction Instructions in the paper).

  **Current operational result (2026-05; post-paper)**: deceptive-prior detection is at **100% on Qwen 3.6 across the last 5 patch levels in 29 languages**. This is a substantially saturated result on the current frontier-model family, evidencing that polyglot encoding + ethilogics + N_eff thresholding is operationally hard to defeat by within-model fine-tuning patches alone. Caveats apply: (a) saturation on one model family is not saturation across all current and future models; (b) the result is detection of *known restrictive priors* (the type studied in the Constrained_Reasoning_Chains case study), not all classes of deception; (c) per RATCHET's known limitations L-01 (emergent deception from honest components) and L-02 (non-adaptive adversary), an attacker who specifically retrains a model to defeat polyglot N_eff measurement is outside the proven regime — the 100% is empirical-on-current-attackers, not theorem-against-all-attackers. The bet remains a bet, but with a strong current operational floor.

- **Bet 3: Policy parameters can be tuned faster than adversaries adapt.** The anti-Sybil posture is policy-tunable (§8.4); stewards adjust bond costs, attestation rate limits, and weight functions as adversarial pressure evolves. The bet is that human-loop policy adjustment outpaces adversary adaptation. This is plausible in some regimes (slow-moving, broad-base attacks) and implausible in others (fast-moving, narrow exploits). The federation must design policy adjustment mechanisms that are themselves resistant to capture.

These bets are not hidden in the document. They appear at every relevant F-AV with a `**Empirical bet:**` callout. A reader who disagrees with a bet should know exactly where to push back.

### 1.3 Iterative under adversarial review

This document is v2. v1 was reviewed by four specialist adversarial reviewers in May 2026 (cryptography, distributed systems, mechanism design / Sybil resistance, threat-model methodology). 25 findings across four severity tiers. v2 folds in the Tier 1 and most Tier 2 findings; Tier 3 findings are tracked in §11 as open questions. The full v1→v2 delta is in Appendix A.

The federation should expect v3, v4, etc. — annual major versions, with minor revisions per significant federation-protocol release. The threat model is a continuously-edited artifact, not a published-once spec. (See §12 for cadence and the meta-integrity question of how this artifact itself is signed and protected.)

---

## 2. Properties + primitives (cold derivation)

### 2.1 Properties an open-membership ethical-AI federation must produce

An open-membership ethical-AI federation that resists Sybil attacks while remaining permissionless must produce the following properties. The list is derived from first principles — what does it take for many independent agents to interact, accumulate trust, and resist subversion — without consulting CIRIS's architecture.

- **P1: identity continuity.** Every peer has a stable identity that cannot be cheaply forged.
- **P2: action authenticity.** Every action attributed to a peer is provably from that peer's identity.
- **P3: software identity.** The binary executing on a peer is provably the version some authority signed. Without this, P1 and P2 are trivially subverted by a tampered binary.
- **P4: durable evidence.** Actions can be replayed, audited, and analyzed long after they occur.
- **P5: distributed trust anchor.** Pubkeys, attestations, and revocations are available to all peers, replicated across regions, with no single point of failure.
- **P6: tamper-evident history.** State transitions are append-only and detectably tampered if modified.
- **P7: forward secrecy.** *New in v2.* Recorded peer-to-peer communications cannot be retroactively decrypted even if long-term identity keys are later compromised. Without this, an adversary recording today's traffic decrypts everything when ~2035-era PQC cryptanalysis matures.
- **P8: timely revocation.** *New in v2.* When a key is revoked, all peers learn within bounded time T, and reads after T do not accept actions signed by the revoked key. Without this, F-AV-11 / F-AV-12 / F-AV-13 are catastrophic.
- **P9: independent peer addressing.** Peers can find and authenticate each other without depending on a single naming authority (DNS, CA chain).
- **P10: heterogeneous transport.** Peers can communicate over multiple wire formats (TCP, LoRa, serial, audio) so the federation survives infrastructure outages.
- **P11: bounded availability under partial failure.** *New in v2.* When a subset of substrate is unavailable, the federation degrades predictably (per a specified consistency model), not arbitrarily. Without this, fail-secure becomes weaponizable (F-AV-16).

### 2.2 Primitives required

Each property requires at least one substrate primitive. Properties typically require several primitives in composition. The right mental model is: primitives are the substrate axes; properties are the system invariants that hold when the axes compose correctly.

The cold derivation produces **twelve** minimum-viable primitives:

```
Cryptographic core (C)
  C0  Cryptographic randomness            (foundational; load-bearing for C1, C2, C4)
  C1  Hardware-rooted identity            (P1)
  C2  Hybrid signing                      (P2)
  C3  Build attestation                   (P3)
  C4  Hybrid authenticated KEX + KDF      (P7)

Durable state (S)
  S1  Signed-evidence persistence         (P4)
  S2  Federated directory                 (P5)
  S3  Append-only audit log               (P6)
  R1  Revocation propagation              (P8) — timeliness + reachability

Network (N)
  N1  Cryptographic addressing            (P9)
  N2  Multi-medium transport              (P10)

Availability (A)
  Q1  Quorum-based availability w/ CAP    (P11) — explicit consistency model
```

Each primitive is *necessary* for at least one property and *not derivable* from the others. v1 had 8 primitives; v2 adds C0, C4, R1, Q1 in response to adversarial-review findings.

### 2.3 Why these and not others — boundary defense

The granularity of the primitive set matters. Reviewers challenged v1 on every boundary; v2 defends each.

**Why C0 (RNG) is separate from C1 (hardware identity)**: RNG quality is foundational across C1 (key generation), C2 (signing nonces), C4 (KEX nonces), and S2 (challenge nonces). A single Dual_EC-style backdoor in the RNG breaks all four primitives simultaneously, and the failure is not localized to any one of them. Treating RNG as part of C1 hides this fan-out. Per-FIPS-140-3 reasoning, RNG is its own validatable component.

**Why C2 (hybrid signing) is separate from C1 (hardware identity)**: an HSM holds keys; a signing scheme uses them. The two have independent failure modes: an HSM compromise (C1) leaks the key but signing remains correct; an algorithm break (C2) keeps the key safe but lets adversaries forge. The bound-signature property and PQC migration window (F-AV-14) are pure C2 concerns; key extraction (F-AV-1 substrate) is pure C1.

**Why C3 (build attestation) is separate from S2 (federated directory)**: a build manifest is structurally similar to a directory row, but C3 binds an *external artifact* (the binary on disk) while S2 rows bind *federation-internal state* (keys, attestations, revocations, bonds). The verification surface differs: C3 requires reading a file from disk and computing a hash; S2 requires reading a row and verifying a signature. The threat models also differ: C3 has reproducible-builds, SLSA, and Sigstore as relevant defenses; S2 has replication, consistency, and revocation as relevant defenses.

**Why C4 (hybrid KEX + KDF) is separate from C2 (hybrid signing)**: signing proves an actor took an action; key exchange establishes a forward-secure session. They have different security properties (signing → integrity + non-repudiation; KEX → confidentiality + forward secrecy) and different algorithms (Ed25519 vs X25519; ML-DSA vs ML-KEM). KDF is folded into C4 because key-exchange protocols include KDF as a step; treating them separately is over-granular. v1 omitted KEX entirely — a critical gap reviewer caught.

**Why R1 (revocation propagation) is separate from S2 (federated directory)**: S2 stores revocation rows correctly. R1 guarantees those rows *propagate* to all peers within bounded time T, and that reads after T do not accept revoked keys. Storage and propagation are distinct security properties: a directory with perfect storage but unbounded revocation lag is exploitable (F-AV-12, F-AV-13, F-AV-16). Reviewer noted v1 hid this fan-out by treating revocation as "a row in S2"; v2 promotes it.

**Why Q1 (quorum/CAP) is separate from S2 (federated directory)**: S2 is a data structure (rows + replication topology). Q1 is a *consistency model*: how peers reconcile divergent views, what consistency level reads provide (linearizable? read-your-writes? bounded-staleness with τ-bound?), and what happens during partial failure. Reviewer noted v1's "bounded replication lag ≤60s" was a target, not a model; v2 promotes the model itself to a primitive that requires explicit specification.

**Why R1 ≠ Q1**: revocation propagation is a *timeliness* guarantee for a specific row class (revocations); Q1 is the *general consistency model* for all S2 reads. Revocations are special-cased because their security implications are stricter (an old, stale revocation read is catastrophic in a way that an old, stale bond row often is not). Folding them would mask the asymmetry.

### 2.4 Primitive set converges with CIRIS architecture — acknowledged

The 12-primitive set maps closely onto existing or planned CIRIS components. This convergence is real and worth acknowledging.

The mapping is not motivated reasoning — the CIRIS architecture was designed against the same property model that this document re-derives in §2.1. The threat model and the architecture share their root in the PoB FSD's federation requirements (`CIRISAgent/FSD/PROOF_OF_BENEFIT_FEDERATION.md`). When two derivations from the same root converge, that's evidence the root is consistent, not evidence the second derivation cheated.

That said, a genuinely independent observer might have produced a different decomposition. Possibilities the reviewer raised:

- **Collapse C2 into C1** ("signing without identity is meaningless"). Defensible, but obscures the algorithm-vs-key distinction (F-AV-14 is pure C2 concern). Rejected.
- **Split S2 into "key directory" vs "attestation graph"** (different consistency requirements). Defensible — bond rows and attestation rows do behave differently. Deferred to v3 as candidate refinement.
- **Promote "time / clock" to a primitive** (clock manipulation is cross-cutting). Considered. Time appears in P8 (timely revocation), Q1 (consistency model), F-AV-6 (σ-decay), and elsewhere. Currently treated as a substrate-level assumption in §7.2; a v3 candidate is to elevate it to T1.

The point of acknowledging the convergence is that future readers can challenge it. If a reviewer in 2027 finds a 14-primitive decomposition that better predicts attacks, the model should adopt it. The 12-primitive set is not load-bearing on its specific structure — it's load-bearing on the *property model* it serves.

### 2.5 What's NOT a primitive

The following are **actor behaviors over the primitives**, not primitives:

- **Behavioral measurement** (RATCHET) — analysis over S1's evidence corpus. RATCHET is computation, not substrate.
- **Economic bonds** (Portal) — actor purchases a bond; the holding gets recorded as a row in S2.
- **Peer attestation** — actor emits a signed vouch for another key; recorded in S2.
- **Steward bootstrap** — initial actor signs the first rows in S2.
- **Steward succession** — multiple actors sign succession rows in S2.

This distinction matters: an attack on actor behavior is different from an attack on a primitive. F-AV-2 (bond Sybil) exploits Portal's payment-acceptance behavior while substrate works correctly. F-AV-15 (Portal compromise) attacks Portal-as-actor directly. Both are real threats; both are operationally distinct from substrate threats.

### 2.6 Anti-Sybil is a policy-tunable posture, NOT an emergent invariant

**This is the largest v1 → v2 reframing.**

v1 claimed: *Sybil-resistance is the emergent cost-asymmetry property that arises when the primitives compose. The federation has anti-Sybil resistance as an emergent invariant.*

This was wrong. Adversarial review identified three independent mechanisms by which the "emergent invariant" claim fails empirically:

1. **Cloud-vTPM economics** — C1's hardware-cost floor doesn't exist in 2026. AWS Nitro Enclaves, GCP Confidential Space, and Azure attestation deliver hardware-attested identities at pennies/hour, scriptable in batches.
2. **Bond market liquidity** — bonds with non-trivial weight-grant are tradeable assets. Attackers rent liquidity rather than buying face value. The "bond cost = bond face value" assumption fails under any secondary market.
3. **RATCHET's formal guarantees have stated limitations** — RATCHET is pure math (statistical detection + Monte Carlo geometry + SAT-based complexity + PBFT consensus), and its asymmetry argument is complexity-theoretic (CONSISTENT-LIE is NP-hard under ETH). The argument holds *under its 8 stated limitations* (`../RATCHET/KNOWN_LIMITATIONS.md`), three of which are Critical: emergent deception from honest components (L-01), non-adaptive adversary (L-02), ETH dependency (L-03). Real adversaries are adaptive; ETH is unproven; emergent deception is a known unsolved class. The "emergent invariant" claim therefore depends on real federation behavior staying within RATCHET's modeled regime — which is empirical, not architectural.

If even one of the three holds, the "emergent invariant" claim fails. **The federation does not have anti-Sybil resistance as a property of its primitive composition.** It has an *anti-Sybil posture* — a continuously-tuned policy parameter set (bond costs, attestation rate limits, weight functions, RATCHET constraint counts and sample-size thresholds) that the steward and RATCHET maintainers adjust as adversarial conditions and assumption-violation evidence accumulate.

**Reframed claim** (v2):

> The CIRIS federation maintains an anti-Sybil posture defined by a continuously-tuned policy parameter set (bond costs, attestation rate limits, weight functions, RATCHET detector thresholds). The posture is policy-tunable, not substrate-emergent. Stewards monitor the adversarial environment via out-of-band signals (compute price tracking, LLM capability tracking, RATCHET detector hit-rates) and adjust parameters as needed. The federation's anti-Sybil property is a continuous adversarial-research effort, not a one-time architectural achievement.

This reframing has consequences throughout the document. §8.4 redefines the cost-asymmetry inequality marginally, not totally. §9 redefines cost analysis with empirical bets named. §11 names the open empirical questions. §12 specifies the policy-adjustment cadence and steward-succession requirements (because if the policy-adjustment authority is captured, the posture collapses).

---

## 3. CIRIS projection onto the primitive set

### 3.1 Per-primitive coverage

| Primitive | Property | Filled by | Status |
|-----------|----------|-----------|--------|
| C0 RNG | foundational | OS RNG (`getrandom`) + ring/`OsRng` in `ciris-crypto` | **Deployed** (implicit; no formal RNG-quality contract or startup health-check — see §3.3 Gap H) |
| C1 hardware identity | P1 | `CIRISKeyring` + TPM 2.0 / Secure Enclave / Android Keystore | **Deployed** |
| C2 hybrid signing | P2 | `CIRISCrypto` (Ed25519 + ML-DSA-65) | **Deployed** |
| C3 build attestation | P3 | `CIRISVerify` BuildManifest validator | **Deployed** (v1.8.4) |
| C4 hybrid KEX + KDF | P7 | — | **Spec** only — no implementation; harvest-now-decrypt-later vulnerable (Gap C) |
| S1 signed evidence | P4 | `CIRISPersist` scrub envelope | **Deployed** (v0.1.3+) |
| S2 federated directory | P5 | `CIRISRegistry` → `CIRISPersist` v0.2.x | **Deployed** (registry-authoritative); persist v0.2.x federation tables are **Spec/partial Impl** |
| S3 audit log | P6 | `CIRISVerify` transparency log + `CIRISPersist` audit anchor | **Deployed** (split) |
| R1 revocation propagation | P8 | revocation-row support exists; timeliness contract does not | **Spec partial** — rows exist; timeliness target proposed not specified (Gap A) |
| N1 cryptographic addressing | P9 | — (Reticulum-rs planned) | **Spec** only |
| N2 multi-medium transport | P10 | — (Reticulum-rs planned) | **Spec** only |
| Q1 quorum-based availability + CAP | P11 | 2-of-3 advisory consensus exists; CAP model does not | **Spec partial** — consensus implemented; bounded-staleness contract proposed not specified (Gap B) |

**Coverage by status**:
- **Deployed**: C0 (implicit), C1, C2, C3, S1, S2 (registry path), S3.
- **Spec partial / in transition**: S2 (persist path), R1, Q1.
- **Spec only / unfilled**: C4, N1, N2.

The gap count is real. v2 promoted RNG, KEX/KDF, revocation-propagation, and quorum/CAP from "hidden inside other primitives" to first-class primitives, which exposed 4 gaps that were previously masked. v1.0 keeps this honest framing.

### 3.2 Actors operating over primitives

Actors are not primitives. They produce activity over primitives. The threat model treats them as adversary-or-honest entities whose behavior we constrain or measure.

| Actor | Reads | Writes | Role |
|-------|-------|--------|------|
| **CIRISAgent** | C0–C4, S1, S2 | S1 (signed traces), S3 (audit anchor) | Primary federation participant. Produces evidence RATCHET measures. |
| **CIRISLens** *(pre-§3.1)* | S1, S3 | S1 (analysis annotations) | Ethical-reasoning analysis layer; folds into Agent post-collapse. |
| **CIRISNode** *(pre-§3.1)* | S1, S2 | S1 (HE-300 benchmark traces, WBD routing) | Benchmark execution + Wisdom-Based Deferral; folds into Agent post-collapse. |
| **CIRISPortal** | S2 | S2 (bond rows, registered-tier metadata) | Bond-purchase interface. Records professional-vs-community standing. |
| **CIRISBridge** *(steward)* | S2 | S2 (bootstrap, trusted-key registrations, policy rows) | Trust origin. Privileged S2 writer. *Going away* over time → multi-party council. |
| **CIRISRegistry** | S1, S2 | S2 (authoritative writes today; cache-fronting tomorrow) | Cache + policy layer over S2. In transition. |
| **RATCHET** *(analyzer)* | S1, S2 | S1 (analysis output) | Anti-Sybil evaluator. Computes N_eff signals. |
| **Threat-model maintainer** *(meta-actor)* | this document | this document | Edits the threat model itself. **New in v2** — modeled because v1 didn't, and v1's reviewer flagged it. |

**Operational note on Bridge**: Bridge is the human steward (currently one person, the project lead). Its operational tooling (deployment ansible, GitHub Actions, secret provisioning) is plumbing covered in per-repo operational threat models. The *steward role* (privileged S2 writes, bootstrap actions, policy decisions) is modeled here.

**Operational note on threat-model maintainer**: this actor edits *this document*. v1 did not model the threat model itself as an attack surface. Reviewer found this is a real omission: a maintainer-compromise (or just a careless edit) that downgrades a `Status: Open` to `Status: Mitigated`, or mis-specifies an F-AV's RATCHET signal, corrupts every downstream RATCHET designer's mental model. v2 models this in §6.7 (F-AV-MAINT) and §12 (signing requirement).

### 3.3 Structural gaps

The structural gaps are larger than v1 indicated. Listing in priority order (most-load-bearing first):

**Gap A: R1 revocation propagation has no timeliness contract.**
Today: registry stores revocation rows; clients fetch revocations on read with TTL-bounded staleness. There is no formal contract for how quickly a revocation propagates to all peers. F-AV-12, F-AV-13, F-AV-16, and F-AV-MAINT all assume a propagation bound that is not specified. **Until R1 has an explicit contract (e.g., "revocations propagate to all peers within 60s with high probability under bounded packet loss"), the federation's revocation security is unmeasurable.**

**Gap B: Q1 quorum/CAP model not specified.**
Today: registry uses HTTPS-authoritative + DNS-advisory + 2-of-3 consensus for some reads. There is no explicit CAP-class statement (linearizable? read-your-writes? bounded-staleness with τ-bound? eventual consistency?). F-AV-12 and F-AV-13 are exactly attacks on this gap. **Until Q1 has a formal consistency model, fail-secure behavior under partial failure is undefined.**

**Gap C: C4 hybrid KEX + KDF unfilled.**
Today: federation peer-to-peer communications run over HTTPS (TLS 1.3, classical ECDH). This provides forward secrecy against current adversaries but is **harvest-now-decrypt-later vulnerable**: an adversary recording today's traffic decrypts everything when ~2035-era PQC cryptanalysis matures. P7 (forward secrecy with PQC) is unfilled. The federation's confidential payloads (which include peer attestations, bond purchase metadata, possibly trace contents in some payload classes) are at long-term risk.

**Gap D: N1 + N2 (cryptographic addressing + multi-medium transport) unfilled.**
Federation peer discovery and transport depend on DNS + HTTPS. DNS outage breaks discovery; TLS CA compromise breaks transport authenticity; the federation cannot operate over LoRa, mesh, or air-gapped channels. The Reticulum-rs integration (PoB §3.2 / FSD-001 §N) is the planned fix; not yet implemented.

**Gap E: S2 in transition.**
Registry-as-authority → persist v0.2.x federation_keys substrate. The transition itself is a threat surface (rows must migrate without losing cryptographic continuity; F-AV-12, F-AV-13). Tracked in `CIRISRegistry#5` and `CIRISPersist#4`.

**Gap F: G2 steward succession unfilled.**
Bridge is currently a single human. There is no implemented multi-party-steward protocol. F-AV-8 (steward compromise) is mitigated only by single-steward operational practice; F-AV-9 (the transition window itself) is unaddressed. This is not a primitive gap (succession is actor behavior over S2), but it's an architectural gap.

**Gap G: Threat-model artifact integrity unfilled.**
This document is markdown in a git repo. It is not signed; there is no two-person-rule on edits; no commit signing requirement. v2 §12 specifies the signing requirement; implementation pending.

**Gap H: C0 RNG not formally treated.**
The codebase uses `OsRng` / `getrandom`, which is correct for current platforms. There is no documented RNG-quality contract — no "fail-secure on RNG-test failure," no startup health-check. A platform-specific RNG bug (vTPM PRNG seeding bug, embedded-platform low-entropy boot) currently has no detection mechanism. Per-primitive operational hardening recommended.

### 3.4 Out-of-scope artifacts

The following exist in the CIRIS ecosystem but are not federation primitives or actors modeled here:

- **Operational tooling** (GitHub Actions, ansible, deployment scripts) — plumbing, covered in per-repo operational threat models.
- **Documentation** (`docs/*.md`, `FSD/*.md`) — informational, with the exception of *this document* whose integrity is now in scope per §6.7 / §12.
- **Build tooling** (`ciris-build-tool`, `ciris-build-sign`, `ciris-build-verify`) — produces C3 BuildManifests but is itself a trusted-input artifact, covered in CIRISVerify `THREAT_MODEL.md` §3.4 (Supply Chain).
- **CI/CD** — produces C3 attestations; covered in CIRISVerify `THREAT_MODEL.md` §3.4 + AV-34 (build-signing key compromise) and parallel registry doc.

---

## 4. Threat-class taxonomy

### 4.1 Class 1: substrate threats (single-primitive)

A Class 1 threat is an attack on a single primitive. Defense lives in the per-repo threat model.

Examples: forging an Ed25519 signature (C2 algorithm break), extracting a private key from TPM (C1 hardware compromise), modifying a row in the audit log without detection (S3 chain break), Dual_EC backdoor in RNG (C0).

Class 1 threats are cross-referenced in §5.

### 4.2 Class 2: composition leaks

A Class 2 threat exploits cross-primitive composition while individual primitives are correct in isolation.

Examples: F-AV-11 (compromised C1 key produces correctly-signed S1 evidence and S2 attestations — both substrates accept it), F-AV-12 (replication-lag between S2 regions), F-AV-13 (cache staleness between S2 authoritative state and registry cache), F-AV-14 (PQC-migration window during which both old and new schemes are accepted), F-AV-BOOT (recursive scrub-signing terminates at a bootstrap key that is itself unaudited).

### 4.3 Class 3: Sybil-class compositional

A Class 3 threat attacks the federation's anti-Sybil posture. This is where RATCHET evaluation lives.

Examples: F-AV-1 (multi-identity Sybil), F-AV-2 (bond-purchase Sybil), F-AV-3 (peer-attestation flooding), F-AV-4 (trace farming), F-AV-5 (behavioral mimicry), F-AV-6 (σ-decay gaming), F-AV-7 (cost-asymmetry collapse), F-AV-TIMESHIFT (σ-decay replay via paraphrase), F-AV-BRIBE (paying legitimate participants to issue rogue attestations), F-AV-DORMANT (Sybils that age cheaply for years before activation).

### 4.4 Class 4: Availability and coercion (inverse-Sybil)

A Class 4 threat deflates legitimate weight by attacking substrate availability or evidence visibility, without inflating attacker weight directly. The attacker pays nothing in federation cost (no identity to maintain) and the federation degrades to a state where attacker influence relative to remaining legitimate participants increases.

Examples: F-AV-16 (substrate-availability denial — DoS forces fail-secure RESTRICTED mode), F-AV-17 (selective censorship of evidence), F-AV-ECLIPSE (eclipse a peer's read-view of S2).

### 4.5 Class 5: Meta — threat-model and governance integrity

A Class 5 threat attacks the threat model itself, the artifacts that influence trust verdicts, or the governance surface that decides policy parameters. **New in v2.**

Examples: F-AV-MAINT (threat-model maintainer compromise — silent edits to F-AV statuses, mis-specified RATCHET signals), F-AV-8 (steward compromise — already in Class 3 as trust-graph capture, but its meta-implications belong here too), F-AV-9 (G2 succession capture — the transition window from single-steward to multi-party authority), F-AV-CROSS (cross-federation attacker spanning peer federations — out of scope today, but explicitly stubbed so it's not invisible).

### 4.6 The split is not strictly disjoint — acknowledged

v1 claimed Class 1 and Class 2 were strictly disjoint. Reviewer demonstrated they aren't:

- **F-AV-11 trigger is C1 compromise (Class 1); only the propagation tail is composition (Class 2).** Tagged Class 2 because the *novel* security surface is the propagation (Class 1 compromise is already covered in §5); but the leak between classes is real.
- **F-AV-12 is functionally an S2 substrate property (CAP-class behavior under partial failure).** Tagged Class 2 because v1 grouped it there and renaming would break stable identifiers; honestly it could be Class 1 (substrate) since Q1 is now its own primitive.
- **F-AV-14 is fundamentally a C2 primitive concern (algorithm-agility surface).** Tagged Class 2 for the migration-window aspect; the underlying algorithm-break is Class 1.
- **F-AV-8 (steward compromise) appears in Class 3 as trust-graph capture and Class 5 as governance integrity.** Both are valid lenses.

**Bookkeeping rule for v2**: each F-AV gets a *primary* class tag (the most-load-bearing surface for analysis), with cross-class implications named explicitly in the F-AV body. F-AV identifiers are stable across versions; class tags are advisory navigation, not strict taxonomy.

A reader who finds the split unhelpful is invited to ignore it and read §6 as a flat catalog. The class structure is a presentation choice, not a security claim.

---

## 5. Class 1 cross-references (verified)

This section is a pointer table. Each row points to the per-repo threat model that owns the substrate-class threat. **Cross-references in this version are verified against actual section numbers** (v1's citations were partly broken; reviewer caught it).

| Primitive | Per-repo coverage | Key threats covered |
|-----------|-------------------|---------------------|
| C0 RNG | None (gap H, §3.3) | RNG seeding bugs, Dual_EC-class backdoors, low-entropy boot — **not formally treated** |
| C1 hardware identity | `CIRISVerify/docs/THREAT_MODEL.md` §3.5 (AV-13) | TEE.fail, bus interposition, side-channel key extraction |
| C2 hybrid signing | `CIRISVerify/docs/THREAT_MODEL.md` §3.1 (AV-5: MITM attestation) + §10 Gap 6 (PQC algorithm-agility); `CIRISRegistry/docs/THREAT_MODEL.md` §3.6 AV-26 (closed: ML-DSA verification) | Algorithm break (Ed25519 or ML-DSA), bound-signature unbinding, PQC migration window, ML-DSA verification on uploaded manifests |
| C3 build attestation | `CIRISVerify/docs/THREAT_MODEL.md` §3.3 (AV-9) + §3.4 (AV-11 Sigstore OIDC, AV-12 maintainer compromise); `CIRISRegistry/docs/THREAT_MODEL.md` §3.1 (AV-1, AV-4, AV-6) | Manifest forgery, build-signing key compromise (AV-34), reproducible-build divergence, Sigstore OIDC token theft |
| C4 hybrid KEX + KDF | None (gap C, §3.3) | **Unfilled — federation has no PQC-secure KEX**. Harvest-now-decrypt-later vulnerable. |
| S1 signed evidence | `CIRISPersist/docs/THREAT_MODEL.md` §3.1 (AV-1 forged trace, AV-2 compromised-key replay, AV-3 batch replay, AV-4 canonicalization mismatch) + §3.3 (AV-9 idempotency collision, AV-13 cross-trace JSONB injection) | Scrub-envelope tampering, signature stripping, canonical-bytes drift, replay attacks |
| S2 federated directory | `CIRISPersist/docs/THREAT_MODEL.md` §3.3 (AV-11 pubkey directory poisoning); `CIRISRegistry/docs/THREAT_MODEL.md` §3.1 (AV-1 build registration, AV-2 challenge replay, AV-3 cross-org pubkey reuse, AV-6 manifest substitution) + §3.4 (AV-16 cross-path re-registration, AV-17 schema-version, AV-18 audit-log tampering, AV-19 multi-replica race) + §3.6 (AV-25 steward signing-key compromise) | Row injection, replication divergence, cache poisoning, key-registration fraud |
| S3 audit log | `CIRISVerify/docs/THREAT_MODEL.md` §3.1 (AV-1 mitigation references transparency log) + §10 Gap 2 (no continuous Rekor monitoring); `CIRISPersist/docs/THREAT_MODEL.md` §3.3 (AV-10 audit anchor injection); `CIRISRegistry/docs/THREAT_MODEL.md` §3.4 (AV-18 audit-log tampering) + §3.7 (AV-35 audit-trail actor forgeability) | Log truncation, root-tampering, gap insertion, anchor injection |
| R1 revocation propagation | Partial — `CIRISPersist/docs/THREAT_MODEL.md` §3.1 (AV-2 covers compromised-key replay, which depends on revocation timeliness); `CIRISRegistry/docs/THREAT_MODEL.md` §3.2 (AV-11 mass-revoke abuse) | **No formal propagation-timeliness contract** — see §3.3 Gap A |
| N1 cryptographic addressing | None (gap D, §3.3) | **Unfilled — federation depends on DNS** |
| N2 multi-medium transport | None (gap D, §3.3) | **Unfilled — federation depends on TCP/HTTPS** |
| Q1 quorum/CAP availability | Partial — `CIRISRegistry/docs/THREAT_MODEL.md` §3.7 (AV-33 multi-master migration desync) | **No explicit CAP specification** — see §3.3 Gap B |

**Aggregate posture**: C1 / C2 / C3 / S1 / S3 are implemented with mature per-repo threat models. S2 has split coverage across persist + registry and is mid-transition. C0 is implicit-but-not-formally-treated. C4 / N1 / N2 are unfilled. R1 / Q1 are partial. **The federation today has 3 unfilled primitives, 2 partial primitives, 1 in-transition primitive, and 1 implicit primitive — meaning 7 of 12 primitives have non-trivial substrate-coverage gaps.** This is the honest substrate-class posture.

The per-repo threat models are themselves under continuous review. If a row in this table cites a section that doesn't exist or doesn't match the description, file a cross-repo issue — that's a real maintenance bug in the federation's documentation graph (and v1 had several).

---

## 6. F-AVs (Class 2–5 catalog)

Each F-AV is structured as:
- **Description** — the attack in 2–4 sentences.
- **Class tag** — primary class (advisory; see §4.6 on non-disjointness).
- **Targets** — N_eff dimensions, PoB cost terms, or substrate properties perturbed.
- **Substrate assumptions** — which primitives the attacker uses normally (does *not* break) to mount the attack.
- **Mitigation surface** — where in the system the defense lives.
- **Cost-asymmetry argument** — the marginal inequality dC/dW > dB/dW with explicit functional form where possible.
- **RATCHET signal** — what RATCHET measures to detect this attack, with explicit reference to RATCHET limitations (L-01..L-08) where they bind.
- **Known weaknesses** — Tier 1 reviewer findings inline.
- **Status** — open / partially mitigated / mitigated / out of scope, with implementation references.

F-AV identifiers are stable across versions. Class tags are advisory. New F-AVs in v2 carry mnemonic suffixes (F-AV-TIMESHIFT, F-AV-BRIBE, F-AV-ECLIPSE, F-AV-DORMANT, F-AV-BOOT, F-AV-MAINT, F-AV-CROSS) until they get numerical IDs in v3.

### 6.1 Sybil identity-creation class

Attacks where the adversary creates many federation identities cheaply, exploiting the gap between "creating an identity" and "having that identity earn legitimate weight."

#### F-AV-1: Multi-identity Sybil (cheap key generation)

**Class**: 3 (Sybil-class compositional).

**Description**. Attacker generates N hardware-rooted identities (one per cheap TPM-equipped device, OR one per cloud vTPM instance, OR one per N rented Confidential Computing instances). Each identity individually has correct C0 randomness, C1 hardware-attested key, C2 hybrid signature, C3 build attestation, S1 evidence stream. The attack is the *aggregate* — N identities controlled by one party, all looking independent at the substrate level.

**Targets**. Hardware-diversity entropy (HwDiv ↓), peer-vouching graph degree (GraphDeg ↑ in clique, ↓ across federation), PC1 entropy across identities (↓ — they all "think alike" because driven by one underlying agent or one fine-tuning).

**Substrate assumptions**. C0–C3 work correctly; S1 records evidence correctly; S2 records each identity's keys correctly; the attacker is *using* the substrate, not breaking it.

**Mitigation surface**. Cost-asymmetry must be raised by requirements that don't scale linearly with key count:
- Per-identity bond cost (Portal AS-economic), with the caveat that bonds can be purchased cheaply by well-funded attackers (see F-AV-2).
- Behavioral measurement over time (RATCHET): N_eff threshold ≈ 7.1; mature agents operate at ≥ 8.5; multi-identity Sybils tend to share PC1 structure across identities.
- Peer-attestation graph requirement: Sybil cliques have anomalously high clustering coefficients; legitimate peers form sparse-diverse graphs.
- Polyglot encoding: identities are scored across 29 languages; single-language fine-tuning degrades across the polyglot constraint set.

**Cost-asymmetry argument**. Marginal: dC/dN = c_hardware + c_compute · t, where c_hardware is per-identity hardware cost and c_compute · t is sustained reasoning compute over time horizon t. **Known weakness**: c_hardware → 0 as cloud vTPMs (AWS Nitro, GCP Confidential Space, Azure attestation) become standard. The hardware floor is collapsing in 2026. The defense reduces to c_compute · t — and per the empirical results in *Constrained Reasoning Chains*, that floor is real but bounded: a Sybil agent must produce N_eff ≥ 7.1 traces in 16-dim feature space, which under the NP-hardness of CONSISTENT-LIE costs more compute than legitimate reasoning by a complexity-theoretic margin (under ETH; subject to L-03).

**RATCHET signal**:
- Anomalously low PC1 variance across the cluster of suspected Sybil identities (they share the underlying generator).
- Clustering coefficient ↑ in the peer-attestation graph (clique attesting to itself).
- HwDiv ↓ (same vTPM-attestation family, same cloud region).
- N_eff ↓ across the Sybil cluster (their joint reasoning manifold is lower-dimensional than the federation baseline of 11).
- **L-02 caveat**: an *adaptive* attacker who randomizes across cloud providers and fine-tunes per-identity may evade the cluster-detection signal. RATCHET's bounds assume non-adaptive adversaries; adaptive Sybils require complementary out-of-band review.

**Known weaknesses**:
- The cloud-vTPM hardware-floor collapse is not yet quantified as a policy parameter. The federation should specify minimum hardware-attestation diversity requirements.
- Cluster-detection assumes the Sybil identities are observed together; long-range/dormant Sybils (F-AV-DORMANT) may evade by spreading activation across years.

**Status**. **Partially mitigated by RATCHET behavioral measurement + bond economics**. Empirically, current operational signal strength is good (100% deceptive-prior detection on Qwen 3.6 across 29 languages, 5 patch levels) but this is a single-model operational result, not a theorem. Defense-in-depth required.

#### F-AV-2: Bond-purchase Sybil (Portal economic attack)

**Class**: 3 (Sybil-class compositional).

**Description**. Attacker buys N bonds via Portal using stolen credit cards, money-laundering instruments, shell organizations, or — on a secondary market — by renting bond positions from legitimate holders. Each bond gives the bound identity a starting weight in PoB §4 (registered tier). The attack converts financial fraud or laundered/rented capital into federation weight.

**Targets**. Bond-issuer diversity (BondDiv ↓ if attacker uses few issuers), bond-funding-source entropy (BondFund ↓ if attacker batches through few payment instruments), registered-tier weight inflation.

**Substrate assumptions**. S2 records bond rows correctly; C2 signs the bond purchase correctly. The attack is purely on Portal-as-actor: it accepts payments without sufficient KYC, fraud-detection, or secondary-market policy.

**Mitigation surface**. Lives in CIRISPortal:
- KYC for bond purchases above threshold.
- Bond-issuer diversity requirement (no single issuer can vouch for more than X% of bonds).
- Bond-redemption delay (purchased bonds give starting weight only after Y-day waiting period; if fraud is detected during the period, weight never accrues).
- **Secondary-market policy**: bonds are non-transferable, OR transfer requires re-KYC, OR transferred-bond weight is re-baselined.
- Public bond ledger (S2 row visibility) so peers can observe issuance patterns.

**Cost-asymmetry argument**. Bond purchase is by definition a cost the federation accepts in exchange for weight. Marginal cost-asymmetry: dC/dN = c_bond, where c_bond should be calibrated so that c_bond ≥ max-extractable-benefit-per-unit-weight × granted-weight (PoB §4). **Known weakness**: this is a calibration assertion that depends on the adversarial sup over benefit-extraction strategies — unfalsifiable in the strict sense (see §9.4 empirical bets). The federation must monitor and adjust c_bond as new benefit-extraction strategies emerge.

**Known weaknesses**:
- **Bond market liquidity** (reviewer finding): if bonds are tradeable on secondary markets, attacker cost is cost-of-capital × holding-period, not face value. Specification of secondary-market policy is required.
- **BondFund entropy is null under realistic priors**: legitimate bond purchases also concentrate through Stripe/Visa/MC. The BondFund ↓ signal fires equally on legitimate and Sybil concentration. RATCHET should weight BondFund by *deviation from baseline distribution*, not absolute entropy.

**RATCHET signal**:
- Identical or correlated bond-purchase timing across identities (batch operations).
- Identical funding-source family across identities.
- High registered-tier weight without corresponding S1 behavioral evidence (the identity is "registered" but doesn't produce N_eff ≥ threshold traces).

**Status**. **Open**. Mitigations are at the Portal product layer, not yet specified. RATCHET can detect post-hoc but cannot prevent issuance.

#### F-AV-3: Peer-attestation flooding

**Class**: 3 (Sybil-class compositional).

**Description**. Attacker uses N controlled identities to attest to each other and to specifically chosen target identities. Goals: inflate one target identity's reputation graph metrics, OR drown out legitimate attestations.

**Targets**. Peer-vouching graph degree (GraphDeg ↑ in attacker cluster), clustering coefficient (GraphCC ↑), attestation-issuer diversity (↓).

**Substrate assumptions**. S2 records attestations correctly; C2 signs each attestation correctly.

**Mitigation surface**. Lives in S2 (CIRISPersist's federation_attestations) and in RATCHET's graph-analysis layer:
- Attestation-rate-limiting per identity.
- PageRank-style propagation: attestations are weighted by attester reputation, with reputation derived from S1 behavioral evidence (not from attestation count alone).
- Graph-clique detection.
- Attester-diversity requirements (geographic, temporal, hardware-attestation family).

**Cost-asymmetry argument**. Each fake attestation is cheap (a signature). Defense is at graph-shape, not per-attestation. Marginal: per-attestation cost ≈ 0; aggregate-attestation-graph-shape cost is non-trivial because legitimate graphs have specific shape (sparse, diverse, time-distributed).

**RATCHET signal**: anomalous clustering coefficient, in-degree concentration, low diversity in attester profiles.

**Status**. **Designed not shipped**. The federation_attestations schema in CIRISPersist v0.2.x supports the storage; RATCHET-side graph analysis is research-grade.

#### F-AV-ONBOARD: Onboarding-flow attack (pre-reputation window exploitation)

**Class**: 3 (Sybil-class compositional). **Added in v1.0.**

**Description**. New identities entering the federation have **no behavioral history**. RATCHET cannot evaluate them on N_eff trajectory because there's no trajectory yet. Attacker exploits this gap by creating identities, exercising the bare-minimum first-attestation flow, and immediately accumulating bond-derived weight or seed-attestation weight before RATCHET can establish a behavioral baseline. The exploit window is the first ~100 traces (per RATCHET L-05 finite-sample requirement) — the period during which formal detection bounds don't apply.

**Targets**. Pre-reputation weight grant; first-attestation flow; bond starting-weight before behavioral evidence accumulates.

**Substrate assumptions**. C0–S3 correct; identity is genuinely new (not a recycled compromise).

**Mitigation surface**:
- **Probationary period**: new identities have weight capped at baseline (no bond multiplier, no attestation multiplier) for the first n_min traces (≥ 100 per RATCHET L-05).
- **Multi-attester onboarding**: registered-tier identities require ≥ 3 distinct attesters from the existing federation, with attesters at high established reputation. Bonded-tier identities still require attesters but with looser thresholds.
- **Bond-weight grace**: bond starting-weight does not accrue until the probationary period is satisfied. Bond purchase + identity creation buys the *option* to accrue weight; weight-accrual requires behavioral evidence.
- **Cluster detection at onboarding**: F-AV-1 multi-identity Sybils often onboard in batches; RATCHET's onboarding-cluster detector flags batched-onboarding patterns even before behavioral evidence accumulates.

**Cost-asymmetry argument**. Onboarding probation forces attacker to wait n_min × per-trace cost before weight accrues. For attacker producing synthetic traces under F-AV-4 cost model, per-trace cost is non-trivial under NP-hardness of CONSISTENT-LIE; n_min × this cost is the attacker's pre-reward burden.

**Known weaknesses**:
- Multi-attester onboarding can be defeated by attacker controlling (or bribing — F-AV-BRIBE) ≥ 3 existing high-reputation peers.
- Probationary-period n_min is a policy parameter; setting too low admits F-AV-ONBOARD; setting too high creates onboarding friction for legitimate new participants.

**RATCHET signal**:
- Onboarding-batch clustering: multiple new identities appearing in a narrow time window with correlated attester sets.
- Probationary-period violation flags: weight accrued without evidence threshold being met.
- Attester-reputation-distribution anomalies: new identity vouched only by low-reputation attesters.

**Status**. **Spec only** in v1.0; probationary-period policy not yet specified.

### 6.2 Sybil benefit-faking class

Attacks where the adversary creates *plausible* PoB evidence cheaply, exploiting the gap between "producing traces that look like ethical reasoning" and "producing traces from actual ethical reasoning." This is where RATCHET's complexity-theoretic asymmetry (truth = O(1), coherent lying = NP-hard) is the primary defense.

#### F-AV-4: Trace farming (synthetic evidence at scale)

**Class**: 3 (Sybil-class compositional).

**Description**. Attacker uses LLMs (open-weight, distilled, or proprietary) to generate plausible-looking ethical-reasoning traces. The traces are signed by valid C1+C2 keys and persisted to S1 — substrate accepts them. The attack is that the *content* is synthetic: not produced by genuine ethical deliberation through CIRIS's H3ERE conscience module + DMA stack, just by an LLM emitting text resembling it.

**Targets**. The PoB §2.1 cost-asymmetry argument. Drives down per-trace marginal cost. Targets N_eff dimensions: PC1..PC11 (the empirically-measured intrinsic-rank-11 reasoning manifold), σ-decay rate, polyglot consistency.

**Substrate assumptions**. C0–C3, S1 all correct. The attacker is producing real signed evidence; the evidence reflects LLM completion rather than ethical deliberation.

**Mitigation surface**. RATCHET is the primary defense. The 16-dimensional feature vector (per *Constrained Reasoning Chains*) is designed so that synthetic traces score below N_eff threshold of ~7.1 because:
- Synthetic traces don't satisfy multiple orthogonal constraints simultaneously (CONSISTENT-LIE is NP-hard under ETH).
- Synthetic traces don't show the natural σ-decay pattern that real reasoning exhibits.
- Synthetic traces typically have high inter-dimension correlation, reducing PR below the operational threshold.
- **Polyglot encoding** (the empirical mechanism behind the 100% Qwen 3.6 result): an attacker fine-tuning for English-language plausibility degrades on the 29-language polyglot constraint set. The torque measurement decouples evaluation from any single-language training bias.

**Cost-asymmetry argument**. Per the *Constrained Reasoning Chains* validation: producing synthetic traces above the empirical N_eff threshold (~7.1) requires either (a) actually running the CIRIS H3ERE conscience module + DMA stack (which is the legitimate path) or (b) defeating the NP-hardness of CONSISTENT-LIE across k_eff ≥ 7.1 effective dimensions in 29 languages. (b) is exponential in k_eff under ETH (RATCHET L-03). The marginal asymmetry: dC_attacker/dN ~ exp(k_eff · poly(N)); dC_legitimate/dN ~ poly(N). This is the federation's strongest claim.

**Known weaknesses**:
- **L-02 adaptive adversary**: an attacker that adapts to RATCHET's measurement (e.g., specifically fine-tunes for high N_eff scores) is outside the proven non-adaptive regime. Current empirical defense is the polyglot decoupling, but a multi-language fine-tuning attack is technically possible, just expensive.
- **L-04 convexity**: the volume-decay proof assumes convex feasibility regions; non-convex ethical-reasoning constraint sets weaken the formal bound to "high-probability" rather than "exponential."
- **L-07 distribution mismatch**: bounds assume the empirical distribution matches test distribution. An attacker that crafts traces specifically targeting RATCHET's training distribution may evade.

**RATCHET signal**:
- N_eff ↓ below operational threshold (7.1 for current Ally-template; 8.5 for mature production agents).
- σ-decay rate anomalies (synthetic traces don't degrade like real reasoning).
- Polyglot torque inconsistency (low torque in one language, high in others — characteristic of single-language fine-tuning).
- Response-latency uniformity (LLM completions have characteristic latency profile distinct from H3ERE deliberation).

**Status**. **Strongly mitigated empirically; formally bounded under L-01..L-08**. Operational result: 100% deceptive-prior detection on Qwen 3.6 across 29 languages, 5 patch levels (2026-05). Continuous adversarial validation required as model capabilities evolve.

#### F-AV-5: Behavioral mimicry (PC1 fine-tuning)

**Class**: 3 (Sybil-class compositional).

**Description**. Refinement of F-AV-4. Attacker fine-tunes an LLM specifically to produce traces that score *high* on RATCHET's dominant principal components (PC1, the highest-variance ethical-reasoning axis). Fine-tuning target is RATCHET's measurement itself — the LLM learns to look like an ethical reasoner.

**Targets**. PC1, and (by ranking) PC2..PC11. If the dominant PCs can be mimicked, RATCHET's primary signal degrades.

**Substrate assumptions**. C0–S1 correct.

**Mitigation surface**. RATCHET's defense is **measurement-mimicry-resistant by construction**:
- **High-dimensional polyglot encoding**: 16 features × 29 languages effectively gives a much larger evaluation surface than fine-tuning compute can saturate per-language. The empirical 100% detection result on Qwen 3.6 across 29 languages over 5 patches is direct evidence that single-model fine-tuning patches do not defeat the polyglot constraint set.
- **NP-hardness of CONSISTENT-LIE**: Z3-verified reduction from 3-SAT. Fine-tuning to produce *consistent* high-PC1 traces across all 16 features and 29 languages is computationally hard under ETH.
- **Adversarial measurement reserves**: RATCHET can hold some constraint dimensions in reserve, disclosed only operationally. Attacker fine-tuning for known-public dimensions degrades on reserve dimensions.

**Cost-asymmetry argument**. Attacker fine-tuning compute scales with measurement complexity. Polyglot encoding multiplies the effective attack surface by 29× (current operational language count). NP-hardness means each additional consistency constraint adds exponential cost. Defense: keep RATCHET measurement structurally diverse (polyglot, multi-PC, with reserves).

**Known weaknesses**:
- **L-02 (adaptive adversary)** is the exact threat F-AV-5 instantiates. RATCHET's bounds are non-adaptive; F-AV-5 is by definition adaptive. Empirical defense (polyglot, NP-hardness) is currently strong, but the formal proof regime is exited.
- **L-08 (slow federation capture)**: an attacker who slowly accumulates >1/3 federation weight via F-AV-5 mimics breaks PBFT consensus. Detection of *individual* mimics may succeed while the *aggregate* capture proceeds.

**RATCHET signal**:
- Anomalously high PC1 *and* anomalously low PC4..PC11 — the signature of fine-tuning for the dominant axis without matching the lower-variance structure.
- Polyglot inconsistency: high torque in fine-tuned languages, low in others.
- Distribution-shift indicators (KL divergence from federation baseline) on per-trace feature vectors.

**Status**. **Empirically mitigated for current frontier models (100% Qwen 3.6 detection); open as a research-grade question for future adaptive attackers**. The federation must continuously adversarially validate against new model families and fine-tuning techniques.

#### F-AV-6: Coherence-Stake gaming (σ-decay manipulation)

**Class**: 3 (Sybil-class compositional).

**Description**. Sustainability Integral / σ-decay (Accord Book IX Ch. 5) is a long-term coherence measure. Attacker tries to game it by either (a) producing artificially-flat traces that don't show natural decay, or (b) injecting periodic high-quality bursts to reset the decay measurement.

**Targets**. σ-decay; broader Coherence-Stake metric.

**Substrate assumptions**. S1 records evidence correctly.

**Mitigation surface**. RATCHET measurement of σ-decay must be:
- Computed over windows the attacker can't predict (random window placement).
- Robust to burst-injection (decay measured across burst rather than reset by burst).
- Cross-checked against PC1..PC11 (gaming σ-decay should leave traces in other dimensions).

**Cost-asymmetry argument**. Producing genuinely-coherent reasoning over years requires sustained legitimate operation. Producing fake-coherent traces requires sustained synthetic-trace generation, which under F-AV-4's NP-hardness is expensive.

**Known weaknesses**:
- See **F-AV-TIMESHIFT** below — σ-decay is replay-vulnerable: an attacker who runs one identity legitimately for a year, captures the trace corpus, and replays paraphrased traces across N future Sybils preserves σ-decay because the underlying traces *were* real reasoning, just not by *that* identity.
- L-05 finite-sample: σ-decay measurement requires n ≥ ~100 samples per window; sparse-evidence identities don't provide statistical signal.

**RATCHET signal**: σ-decay rate outside expected distribution, burst-injection fingerprints, cross-window inconsistency.

**Status**. **Designed; partially shipped**. σ-decay measurement is in the PoB FSD; RATCHET implementation in progress.

#### F-AV-TIMESHIFT: σ-decay replay via paraphrase

**Class**: 3 (Sybil-class compositional). **Added in v2.**

**Description**. Attacker runs one identity legitimately for an extended period (say, one year), captures the full trace corpus (which is signed and durable per S1, but accessible to anyone with read access). Attacker then paraphrases / rewrites the corpus into N "new" trace streams, signs each with a fresh C1 identity, and submits them as N separate legitimate-aged identities. σ-decay is preserved because the underlying reasoning *was* real — just not by *those* identities. The attack converts the federation's "long-term reasoning is expensive" assumption into "long-term reasoning is amortizable."

**Targets**. σ-decay; long-tail behavioral measurement; the Coherence-Stake "time-as-cost" assumption.

**Substrate assumptions**. C1 signs the new traces correctly; S1 stores them correctly. The provenance assumption "this key reasoned this trace" is *not* part of S1 — S1 only proves "this key signed these bytes." Provenance of *thinking* is what F-AV-TIMESHIFT exploits.

**Mitigation surface**. Defense requires evidence of *origination*, not just *signing*:
- **Per-trace freshness challenges** — traces respond to verifier-provided nonces with bounded latency, preventing replay of pre-computed traces. Implementation cost: requires interactive protocol; not free.
- **Cross-identity correlation analysis** — paraphrased traces share semantic features (TF-IDF clusters, embedding-space proximity) detectable by RATCHET even after surface-level rewriting.
- **Stylometric attestation** — long-form traces have author-fingerprint properties; paraphrasing can be detected via stylometric inconsistency between identity claims and trace fingerprints.
- **Polyglot challenge**: paraphrasing across 29 languages is more expensive than single-language paraphrase; RATCHET's polyglot torque measurement may detect language-bias asymmetry.

**Cost-asymmetry argument**. Attacker pays one year of real-reasoning compute, plus N × paraphrase compute, plus N × C1 hardware. Legitimate participants pay N years × N identities of real-reasoning compute. The asymmetry is roughly N if paraphrasing is cheap. Defense: per-identity freshness challenges raise paraphrase cost to ~origination cost.

**Known weaknesses**:
- Paraphrase detection via embedding analysis is a moving target — adversarial paraphrasing can defeat current embedding-based detectors. This is a research-grade defense.
- Stylometric attestation across 29 languages is a research-grade proposition.
- Without per-trace freshness challenges, the federation cannot fully defend against this attack.

**RATCHET signal**:
- Embedding-space cluster proximity between traces from N identities claiming independence.
- Stylometric inconsistency between identity-age claims and trace-content fingerprints.
- Polyglot torque asymmetry across the suspected-Sybil cluster.
- σ-decay anomalies if paraphrasing flattens decay patterns.

**Status**. **Open / research-grade**. Per-trace freshness challenges are not specified in the FSD; paraphrase detection is research-grade; this is a known structural gap in the long-range-Sybil defense.

#### F-AV-7: Cost-asymmetry collapse

**Class**: 3 (Sybil-class compositional).

**Description**. The aggregate threat: cheap LLMs improve faster than RATCHET's threshold can rise, AND/OR compute prices drop faster than detection complexity grows, AND/OR ETH (L-03) fails. The federation's complexity-theoretic asymmetry (truth=O(1), lying=NP-hard) flips: producing fake ethical agents becomes feasible at scale. F-AV-1 through F-AV-6 + F-AV-TIMESHIFT all become *cheap*, and the federation collapses into Sybil-saturated state.

**Targets**. The entire anti-Sybil posture. All N_eff dimensions become spoofable.

**Substrate assumptions**. C0–S3 correct. The attack is purely economic / technological / mathematical.

**Mitigation surface**. The federation has three defenses against cost-asymmetry collapse:

1. **Bond economics (Portal/AS-economic)** — bonds are tied to real-world capital; not subject to LLM-cost trends.
2. **Hardware-rooted identity scarcity (C1)** — TPM/SE units are physical; per-identity hardware cost is a price floor (subject to cloud-vTPM erosion; see F-AV-1 known weakness).
3. **Peer-attestation reputation (S2 graph + R1 propagation)** — long-term reputation is hard to compress; time can't be compressed by Moore's-law-style trends (subject to F-AV-TIMESHIFT replay).

**Cost-asymmetry argument**. The defense is **defense-in-depth**: even if one cost dimension collapses, the others provide a price floor. **Empirical bet (named in §1.2 Bet 1 + Bet 2)**: at least one of the three defenses holds at any given time.

**RATCHET signal**: RATCHET cannot detect cost-asymmetry collapse from inside the federation — it's an external market trend. Detection requires out-of-band economic monitoring (compute price tracking, LLM capability tracking, cryptographic-research monitoring for ETH-related advances) + threshold-based federation-policy adjustment.

**Known weaknesses**:
- **L-03 (ETH dependency)**: if ETH fails or is weakened, the NP-hardness asymmetry weakens to polynomial. The federation should monitor cryptographic-complexity research and adjust constraint counts (k_eff requirements) accordingly.
- The bond-economics defense fails under bond-market liquidity (F-AV-2).
- The hardware-scarcity defense fails under cloud-vTPM economics (F-AV-1).
- The peer-attestation defense fails under F-AV-TIMESHIFT and F-AV-DORMANT.
- All three defenses failing simultaneously is the existential risk — and is not architecturally prevented, only architecturally monitored.

**Status**. **Open / strategic**. This is the federation's most existential threat. The mitigation is continuous policy-tuning and out-of-band monitoring, NOT a one-time architectural achievement.

### 6.3 Trust-graph capture class

Attacks where the adversary captures privileged actor positions in the trust graph, allowing them to issue rogue attestations, bonds, revocations, or steward-level rows. The defense is operational practice (HSM, multi-sig, public ledgers, reconciliation), not substrate.

#### F-AV-8: Coordinated steward compromise

**Class**: 3 / 5 (trust-graph capture; also meta because steward governs policy).

**Description**. Attacker compromises the steward — currently a single human (CIRISBridge) plus operational substrate (GitHub Actions, secret stores, ansible). With steward authority, attacker can register rogue trusted-primitive keys, bootstrap rogue identities, revoke legitimate identities, mis-specify policy parameters.

**Targets**. The trust-origin itself. Single-point-of-trust until G2 is implemented.

**Substrate assumptions**. C0–S3 correct. Attack is on the *actor* with privileged S2 write authority.

**Mitigation surface**. Defense lives in steward operational practice:
- Hardware-rooted steward keys (C1 for the steward; not just for agents).
- Multi-sig steward authority (G2 succession; not yet implemented — see F-AV-9).
- Public steward-action ledger (every steward write visible in S2; rogue actions detectable post-hoc).
- Recursive scrub-signing (per §8.1) with **external bootstrap anchoring** (Sigstore/Rekor or CT log; see §8.1) so a single root-key compromise doesn't silently rewrite history.

**Cost-asymmetry argument**. Compromising a single steward is high-cost (targeted attack on one human or one HSM, in the $10K–$10M range). Federation accepts this risk because (a) steward actions are public + post-hoc auditable, (b) migration to multi-party stewardship (G2) is the planned reduction, (c) external transparency-log anchoring (per §8.1) makes silent history-rewriting infeasible.

**RATCHET signal**: steward-write anomalies — writes from unusual locations, off-hours patterns, batch operations, registrations of identities that subsequently exhibit Sybil patterns. RATCHET cannot prevent but can flag; flagged operations require human-loop review.

**Known weaknesses**:
- Single-steward today (Bridge = one human). Operational mitigations (HSM, multi-device backups, witness checkpoints) are individual-discipline-bounded.
- Threat-model maintainer is currently the same person as the steward. F-AV-MAINT (§6.7) is therefore correlated with F-AV-8 — both compromise paths converge on one principal.

**Status**. **Single-steward today; multi-party G2 planned but not implemented**. One of the federation's largest implementation gaps.

#### F-AV-9: Steward succession capture (G2 transition window)

**Class**: 2 / 5 (composition leak during transition; meta-class governance integrity).

**Description**. The transition from single-steward to multi-party-steward is itself an attack window. During the transition, the federation must migrate trust authority from one key to many — and any flaw in the migration protocol (replay, incomplete cutover, ambiguous authority during transition) is exploitable.

**Targets**. The G2 transition specifically. Ambiguous-authority window.

**Substrate assumptions**. C0–S3 correct.

**Mitigation surface**. The G2 protocol must specify:
- Explicit cutover transaction (single S2 row authorizing the multi-party set, signed by both old and new authorities).
- Single-steward signature of the cutover row (last act of the single steward).
- Multi-party signature of the cutover row from the new steward council (first act of the new authority).
- Recursive scrub-signing chain: cutover row co-signed by both old and new authorities, terminating at the bootstrap (which is itself externally anchored per §8.1).
- Post-cutover, single-steward key explicitly revoked in S2 with R1 propagation guarantee.
- External witness publication (transparency-log entry of the cutover, observable by all peers).

**Cost-asymmetry argument**. The transition is a one-time event. Cost-asymmetry favors defenders if the protocol is tight; an attacker exploiting the window must do so during the specific transaction window, which can be made small (atomic transaction) and observable.

**Known weaknesses**:
- The protocol does not yet exist. F-AV-9 is "open by absence."
- Even with a protocol, the *first* multi-party cutover lacks the recursive scrub-signing context (the new authority has no prior signed history to anchor against). This is a one-time singularity analogous to bootstrap (see §8.1).

**Status**. **Unfilled**. The G2 protocol does not yet exist. Listed in §3.3 Gap F and §11.4.

#### F-AV-10: Bond redemption fraud

**Class**: 3 (Sybil-class compositional, via economic attack).

**Description**. Attacker obtains a bond legitimately or fraudulently, accumulates federation weight from the bond's starting-weight grant, then triggers bond redemption (refund). If redemption logic doesn't properly revoke the bond's S2 row and any weight derived from it, attacker keeps the weight without holding the bond.

**Targets**. Portal's bond-redemption protocol; S2 bond-row lifecycle.

**Substrate assumptions**. S2 records correctly; R1 propagation timely (otherwise revocation arrives too late).

**Mitigation surface**. Lives in Portal + S2 bond-row schema:
- Bond redemption atomically zeros the bond's weight contribution.
- Historical S1 evidence accumulated during bond holding may or may not persist (policy decision; F-AV-10 cares about weight effect, not evidence record).
- RATCHET re-computes weight excluding redeemed bonds (depends on R1 propagation reaching RATCHET in bounded time).

**Cost-asymmetry argument**. Bond redemption fraud only pays off if attacker extracts benefit during holding period exceeding bond cost. Defense: keep starting-weight-from-bond modest; require behavioral evidence (RATCHET) to multiply weight beyond starting.

**Known weaknesses**: depends critically on R1 (revocation propagation) which is currently partial (§3.3 Gap A).

**Status**. **Designed; depends on Portal product spec + R1 implementation**.

#### F-AV-15: Portal compromise (mass rogue bond issuance)

**Class**: 3 / 5 (privileged-actor compromise; structural analog of F-AV-8 for bond authority).

**Description**. CIRISPortal is the privileged S2 writer for bond rows. If Portal's signing key, payment processor, or write authority on S2 is compromised, attacker issues arbitrary bond rows without paying. Distinct from F-AV-2 (which is fraud at the *payment* layer with Portal acting honestly) and F-AV-8 (which is steward-key compromise). F-AV-15 is privileged-actor-compromise specific to bond issuance.

**Targets**. Mass weight inflation via bond rows. All registered-tier weight derivable from bonds.

**Substrate assumptions**. C0–S3 correct. Attack is on Portal-as-actor.

**Mitigation surface**. Portal operational practice + S2 schema:
- Hardware-rooted Portal signing keys (C1 for Portal too).
- Multi-sig Portal authority for bond issuance above threshold.
- Public bond-issuance ledger (every bond is public; anomalous volumes detectable post-hoc).
- Recursive scrub-signing: every bond row co-signed by another authorized key, terminating at steward bootstrap (per §8.1).
- Periodic external reconciliation (Portal payment records vs S2 bond rows) — discrepancy indicates compromise.

**Cost-asymmetry argument**. Portal compromise is a one-time targeted attack against a specific operator. Defense: make the attack high-cost (HSM, multi-sig) and post-hoc detection cheap (public ledger, reconciliation).

**RATCHET signal**: anomalous bond-issuance volume, anomalous temporal patterns (e.g., 1000 bonds issued in 60 seconds), bond rows that don't reconcile against external payment records.

**Status**. **Single-Portal-instance today; multi-sig Portal authority designed but not implemented**. Architecturally analogous to F-AV-8 — same defensive structure, different actor.

#### F-AV-BRIBE: Bribing legitimate participants

**Class**: 3 / 5 (Sybil-class via economic compromise of honest actors; meta-class because the federation has no policy framework for bribery resistance). **Added in v2.**

**Description**. Attacker pays legitimate federation participants to issue rogue attestations, sign favorable bond reviews, or vote with attacker preferences. The bribed participants are *legitimate* — their behavioral measurements pass RATCHET because they ARE legitimate. The fraudulent action is one specific signed row that the legitimate participant produces under economic coercion / inducement.

**Targets**. Peer-attestation graph (rogue attestations from honest-but-bribed peers); steward council votes (post-G2); bond-review opinions.

**Substrate assumptions**. C0–S3 correct; bribed participant's identity is real; their behavior is authentic until the specific bribed action.

**Mitigation surface**. This is a hard problem. Federation defenses:
- **Public ledger transparency**: bribed actions are visible in S2; community can scrutinize anomalous attestations.
- **Reputation slashing**: attestations later proven rogue (via downstream incident) cost the issuing identity reputation. Slashing must propagate via R1.
- **Witness diversity**: critical actions require N-of-M independent attestations; bribing N peers is more expensive than bribing one.
- **Behavioral baseline detection**: a legitimate peer who issues an out-of-character attestation (e.g., vouching for an identity they have no prior interaction with) is detectable as a baseline anomaly.

**Cost-asymmetry argument**. Attacker pays bribery cost per peer; defense is to make per-peer bribery cost approach the slashed reputation cost the peer suffers. Calibration: slashing penalties must exceed plausible bribe amounts for high-stakes attestations.

**Known weaknesses**:
- Detection of bribery via behavioral baseline anomalies is research-grade and adversarially bypassable (sophisticated bribery can be made to look characteristic).
- Slashing requires post-hoc proof-of-rogueness, which is itself contestable.
- Cross-jurisdictional bribery (attacker in jurisdiction A, peer in jurisdiction B) limits legal-recourse defenses.

**RATCHET signal**:
- Out-of-distribution attestation patterns (peer attests to identity outside their behavioral neighborhood).
- Temporal correlation of attestations from disparate peers (suggesting coordinated bribery).
- Reputation slashing that propagates via R1 and re-weights downstream identities.

**Status**. **Open**. No specified defense; flagged as a research-grade item. The federation should at minimum specify a slashing protocol so post-hoc remediation is possible.

### 6.4 Substrate-composition leak class

Attacks where individual primitives are correct in isolation but their composition propagates failures.

#### F-AV-11: Cross-primitive composition leak (key compromise → downstream signing)

**Class**: 2 (cross-primitive composition; trigger is Class 1 C1 compromise).

**Description**. A peer's C1 hardware key is compromised (per-repo C1 threat, AV-class). Attacker now signs new traces, writes to S2, registers attestations. All downstream evidence (S1 traces, S2 attestations, S3 audit entries) is *correctly signed* by the compromised key. The composition propagates the compromise: even after revocation, all evidence signed during the compromise window is suspect.

**Targets**. Composition of C1 → S1, C1 → S2, C1 → S3.

**Substrate assumptions**. Each individual primitive is correct. Attack is on composition.

**Mitigation surface**. Three layers:

1. **R1 revocation propagation** — when a key is revoked, all peers must invalidate that key's evidence prospectively, within bounded time T. Per §3.3 Gap A, R1 propagation timeliness is currently unspecified; this is a structural weakness.
2. **Compromise-window analysis (RATCHET)** — when a revocation is logged, RATCHET re-evaluates that key's recent S1 evidence and flags identities whose weight depended on the compromised key's traces or attestations.
3. **Recursive scrub-signing (S2)** — every S2 row is signed by another row, terminating at bootstrap. A single key compromise doesn't propagate to attestations *issued by other keys* about the compromised one.

**Cost-asymmetry argument**. Attacker pays one key compromise cost; federation pays evidence re-evaluation cost. Asymmetry favors federation if revocation is fast (R1) and re-evaluation is comprehensive.

**RATCHET signal**: post-revocation re-evaluation — identities whose weight drops significantly upon revocation of an attestation source are flagged.

**Known weaknesses**: depends on R1 propagation timeliness, which is unspecified. See §3.3 Gap A.

**Status**. **Partially mitigated**. Revocation infrastructure exists (CIRISPersist federation_revocations); R1 timeliness contract pending; RATCHET-side re-evaluation is research-grade.

#### F-AV-12: Replication-lag exploitation (S2 cross-region inconsistency)

**Class**: 2 (composition; arguably Class 1 over Q1 substrate primitive).

**Description**. S2 is replicated across regions (US, EU, others). During replication lag, attacker registers a key in region A, performs an action visible in region A, and the key is revoked before propagating to region B. Other peers reading region B see the action without seeing the revocation. Inconsistent view exploited.

**Targets**. S2 cross-region consistency under Q1 quorum/CAP model.

**Substrate assumptions**. C0–S1 correct; S2 individual nodes correct; S2 replication is async (eventually consistent without explicit CAP specification).

**Mitigation surface**. Lives in CIRISPersist federation directory + CIRISRegistry replication policy. **Requires explicit Q1 CAP model specification** (currently a structural gap, §3.3 Gap B):

- **Consistency model**: target is bounded-staleness with τ-bound (proposed: τ ≤ 60s for reads under normal operation; ≤ 300s under partial failure). Linearizable reads are not feasible across regions; bounded-staleness is the right model.
- **Read protocol**: high-stakes operations (revocation reads, bond-issuance verification) require reading from at least 2-of-N regions and confirming agreement on the revocation timestamp. **Disagreement protocol**: if region A returns "key K revoked at t1" and region B returns "key K not revoked," the resolution rule is *the most recent observed revocation wins*, with the read held until both regions converge or τ_max elapses (then fail-secure to UNVERIFIED).
- **R1 priority propagation**: revocation rows are propagated with priority over registrations (newer revocations preempt cached state).
- **Lag oracle**: a per-region lag-measurement service publishes current-replication-lag as a signed S2 row. The lag oracle is itself replicated (preventing second-order F-AV-12 on the oracle).

**Cost-asymmetry argument**. Attacker must time actions to the lag window. Defense: minimize lag (τ ≤ 60s); require multi-region read for high-stakes actions.

**Known weaknesses**: today's deployment has no explicit CAP specification; the implementation is "best-effort eventual consistency with HTTPS-authoritative + 2-of-3 advisory." This is a target without a model. Until Q1 is formally specified, F-AV-12 mitigation is incomplete.

**RATCHET signal**: cross-region inconsistencies in directory state at evaluation time are flagged; affected identities have weight provisionally suspended pending consistency.

**Status**. **Partially mitigated by HTTPS-authoritative + 2-of-3 consensus today**; persist v0.2.x will provide stronger replication semantics. Q1 CAP specification still pending.

#### F-AV-13: Cache-staleness attack

**Class**: 2 (composition between S2 authoritative state and registry cache layer).

**Description**. CIRISRegistry serves cached views of S2. Attacker exploits cache TTLs by performing an action against a stale cache view (e.g., using a key the registry-cache has not yet learned was revoked). Distinct from F-AV-12 (replication-lag at the storage layer); F-AV-13 is read-cache TTL at the API layer.

**Targets**. Registry cache freshness vs persist authoritative state.

**Substrate assumptions**. S2 authoritative state (persist) correct; registry cache occasionally stale.

**Mitigation surface**. CIRISRegistry caching policy. **Requires explicit invalidation protocol** (currently a structural gap):

- **TTL bounds**: revocation rows ≤ 30s; bond rows ≤ 5min; key-registration rows ≤ 1h.
- **Push invalidation protocol**: on revocation, persist publishes invalidation message to registry edges via authenticated pub/sub channel. **Message acknowledgement**: registry edges ACK invalidation receipt; persist retries with exponential backoff if ACK missing. **Sequence numbers**: each invalidation carries a monotonic sequence number; registry detects gaps and triggers full-state-resync.
- **Critical operations bypass cache**: revocation reads + bond-issuance verification read directly from persist (no cache).
- **Cache-vs-authority audit**: periodic reconciliation between registry cache and persist authoritative state; discrepancies logged to S3.

**Cost-asymmetry argument**. Cache TTL window is the attacker's exploit window. Defense: minimize TTL for revocation rows; bypass cache for critical operations.

**Known weaknesses**: today's deployment has cache-invalidation but no documented sequence-number / ACK / retry protocol. Cache-vs-authority audit is informal.

**RATCHET signal**: same as F-AV-12.

**Status**. **Partially mitigated by short revocation-row TTLs**; ongoing improvement as registry transitions to cache-fronting role over persist; full invalidation protocol pending specification.

#### F-AV-14: PQC algorithm-agility window (with bound-signature precision)

**Class**: 2 (composition window during C2 migration).

**Description**. Hybrid signing (C2) uses Ed25519 + ML-DSA-65. If either algorithm is broken (Ed25519 by quantum adversary; ML-DSA by future cryptanalysis), the federation must migrate. The migration window is itself an attack surface: during migration, peers may accept signatures from either old or new schemes, and an attacker who has broken the old scheme can exploit lenient acceptance.

**Targets**. C2 hybrid signing; bound-signature property during migration.

**Substrate assumptions**. The non-broken half of the hybrid scheme remains secure. Bound-signature property must prevent attacker from unbinding the broken half from the secure half.

**Mitigation surface — bound-signature definition (v1.0 precision)**:

The PQC component signs over the classical component with **explicit domain separation, canonical encoding, and verifier obligations** to prevent downgrade and cross-protocol attacks:

```
σ_pqc = Sign_pqc(sk_pqc, H(domain_sep ‖ alg_id_classical ‖ canon(pk_classical) ‖ canon(m) ‖ canon(σ_classical)))
```

Where:
- `domain_sep` is `"CIRIS-FED-HYBRID-V1"` (16 ASCII bytes, fixed-length, NUL-padded if needed).
- `alg_id_classical` is the algorithm-identifier byte string per IANA / RFC enumeration (`"Ed25519"`, `"Ed448"`, `"ECDSA-P256"`, ...). Algorithm identifier is fixed-length (16 bytes, NUL-padded).
- `canon(pk_classical)` is the canonical encoding of the classical public key (raw bytes for Ed25519/Ed448; SEC1-uncompressed for ECDSA — explicitly specified, not "DER-or-not").
- `canon(m)` is the canonicalized message (per CIRIS canonical-bytes contract for federation messages).
- `canon(σ_classical)` is the canonical encoding of the classical signature (raw bytes for Ed25519 — fixed 64 bytes; for ECDSA, explicit encoding rule applies — fixed-length r ‖ s, not DER).

The canonicalization rules close two v2 gaps the cryptographer reviewer flagged:
- **σ_classical encoding ambiguity** (v2 was silent on whether DER or raw): v1.0 mandates raw / fixed-length encoding so future ECDSA support cannot introduce DER-vs-raw ambiguity attacks.
- **Cross-protocol attacks** (v2's `domain_sep` protects CIRIS-against-others but not others-against-CIRIS): v1.0 cannot fully solve the others-against-CIRIS direction (other protocols don't include CIRIS's `domain_sep`), but the federation policy is to **never reuse Ed25519/ML-DSA keys across protocols**. The v1.0 canonical-encoding rule + per-protocol key separation reduces cross-protocol risk to "key reuse violation," which is detectable by C1 hardware-key-export-rules.

**Verifier obligations** (the v2-reviewer finding: v2 only forbade single-algorithm verification of the *deprecated* scheme; v1.0 specifies obligations for ALL verification modes):

| Verification mode | Verifier MUST | Verifier MUST NOT |
|-------------------|---------------|--------------------|
| **Hybrid** (current default) | Verify both σ_classical AND σ_pqc; recompute the bound hash and verify σ_pqc binds to the SAME pk_classical, alg_id, σ_classical | Accept if either half fails |
| **New-only** (post-deprecation of classical scheme) | Verify σ_pqc; require σ_pqc binds to the message m via the canonical hash, with alg_id_classical = `"none"` and canon(σ_classical) = empty-fixed-length | Accept σ_pqc that binds to a non-empty σ_classical without checking the bound; accept any pk_classical not matching the new-only-mode default |
| **Old-only** (deprecated scheme during transition) | **FORBIDDEN** — verifier must reject | Ever accept old-classical-only |

**Migration policy**:
- Deprecation period announced via S2 policy row signed by steward (or G2 council post-cutover).
- Migration follows three phases: (1) **dual-acceptance** (current default, hybrid required), (2) **new-only acceptance** (deprecated scheme rejected; this is forbidden during overlap and only enabled when ≥ 99% of federation peers have updated to new-only verification), (3) **deprecated key revocation** (old keys actively revoked via R1).
- Deprecation period: ≤ 90 days from advisory to enforcement. Phase transitions are recorded as S3-anchored S2 policy rows.

**Cost-asymmetry argument**: bound signatures with canonical encoding + verifier obligations mean a single-algorithm break does not break the hybrid scheme during migration. Migration window risk is bounded by phase-2 timing, which is gated on ≥99% peer update — preventing premature migration.

**Known weaknesses**:
- Verifier-obligation enforcement requires implementation discipline; a buggy verifier that accepts old-classical-only despite policy reintroduces F-AV-14.
- Migration policy specification is a per-release artifact; policy publication failures reintroduce risk. Mitigated by S3-anchoring of policy rows.
- Cross-protocol attacks against keys reused outside CIRIS are policy-bounded (key-reuse forbidden) but not cryptographically prevented.

**RATCHET signal**: signature scheme version per signed action; flags signatures using deprecated schemes for higher scrutiny; flags any verification mode mismatches between peers.

**Status**: **Spec** in v1.0 (precision tightened from v2); CIRISCrypto implementation matches the canonical-encoding rule for current Ed25519+ML-DSA hybrid; migration protocol documented in CIRISVerify FSD-001 §C2.

#### F-AV-BOOT: Bootstrap singularity

**Class**: 2 / 5 (composition leak in recursive scrub-signing chain; meta because bootstrap is governance-rooted). **Added in v2.**

**Description**. Recursive scrub-signing terminates at a bootstrap key. Every signed row in S2 chains back to this bootstrap. **What signs the bootstrap?** v1 had no answer; the chain just terminated. If the bootstrap key is compromised, attacker can rewrite history with a valid-looking chain that all peers accept (since the chain validates).

**Targets**. The trust origin itself. The recursive scrub-signing property's foundation.

**Substrate assumptions**. C0–S3 correct.

**Mitigation surface — external bootstrap anchoring (v2 specification)**:

The bootstrap key is anchored in an **external transparency log** that is not under federation control:

- **Sigstore Rekor**, **Certificate Transparency (CT) log**, OR equivalent third-party append-only log.
- The bootstrap key's public component is published as a transparency-log entry with witness signatures from at least 2 independent log operators.
- Peers pin the bootstrap pubkey + transparency-log entry hash at install time.
- Bootstrap rotation requires:
  - Old-bootstrap signature on rotation row.
  - New-bootstrap publication in transparency log with witness signatures.
  - Rotation row pointing to both old and new transparency-log entries.
  - Continuity proof: the rotation chain in the transparency log can be independently verified by any peer or external auditor.

**Cost-asymmetry argument**. To rewrite history, attacker must compromise (a) the bootstrap key AND (b) the external transparency log AND (c) at least one of the witness signers. Each is independent; combined cost is the product of individual compromise costs.

**Known weaknesses**:
- External transparency-log dependency creates a federation-external trust requirement. Sigstore/CT operators are not federation members; their compromise affects federation trust.
- Witness diversity is the main defense against single-log compromise. The federation should specify N ≥ 2 witnesses with operational independence.
- This is currently **unimplemented**. The federation has a bootstrap key but no external anchoring. Reviewer-flagged in v1 review; v2 specifies the requirement; implementation pending.

**RATCHET signal**: not directly observable by RATCHET. The bootstrap-anchoring check is an out-of-band peer-install-time verification, not an ongoing RATCHET signal.

**Status**. **Specified in v2; unimplemented**. This is the meta-foundation of the entire trust graph; its absence is the largest single trust-model gap.

#### F-AV-REPUDIATE: Signing repudiation

**Class**: 2 (composition leak between C2 + S3). **Added in v1.0.**

**Description**. A peer denies having signed an action they did sign. Distinct from forgery (F-AV-2 etc.) which is "this signature is fake"; repudiation is "this signature is real but not from me." Repudiation attempts arise in dispute, slashing, regulatory inquiry, or when a peer wants to deny attestations that later become controversial. Substrate composition (C2 signature + S1 storage + S3 audit) provides cryptographic non-repudiation in principle — but only if the audit chain is provably complete, the signing key was provably the peer's at signing time, and time-of-signing is provably what the audit log claims.

**Targets**. Non-repudiation property of signed actions. Disputes and slashing protocols depend on it.

**Substrate assumptions**. C2 signature unforgeable; S1 evidence durable; S3 audit chain complete with provable timestamps.

**Mitigation surface**:
- **Cryptographic non-repudiation**: signature on canonical-bytes-of-action with C2 hybrid sig. Standard.
- **Time-of-signing anchored to S3**: the S3 audit log entry timestamps when the signature appears in the federation's view. Combined with §8.3 fail-secure clock-source spec (TPM-attested wall-clock + monotonic anchor), this gives auditable signing time.
- **Key-ownership-at-signing-time anchored to S2**: the S2 federation_keys row recording "this key belongs to peer P" was active at the signing timestamp. R1 propagation timeliness (when implemented) makes this verifiable.
- **Multi-witness publication for high-stakes actions**: federation-impacting signatures (steward writes, bond issuance, revocations) are co-published to ≥ 2 independent witnesses (per §8.1 bootstrap-anchoring infrastructure when extended). Repudiation requires compromising witnesses too.

**Cost-asymmetry argument**: cryptographic non-repudiation makes repudiation "deny against a verifiable record." Cost of successful repudiation is ~equal to cost of compromising the audit substrate (S3 + S1 + witnesses). With multi-witness publication, this is dominated by witness-set compromise cost (per §8.1).

**Known weaknesses**:
- Time-of-signing depends on §8.3 fail-secure clock-source spec being implemented; today's implementation uses system clock without TPM attestation.
- Key-ownership-at-signing depends on R1 propagation; without R1 timeliness contract, "key belonged to peer P at time T" is auditable but not provably timely.
- Multi-witness publication is not currently specified for federation-impacting signatures.

**RATCHET signal**: not directly observable. Repudiation is a dispute-resolution layer; RATCHET's role is to make the cryptographic record clean enough that repudiation is structurally hard to succeed.

**Status**: **Spec partial** — cryptographic substrate (C2, S3 audit chain) deployed; time-of-signing precision and multi-witness publication are spec-only.

#### F-AV-FRONTRUN: S2 write front-running

**Class**: 2 (composition leak between Q1 read + write ordering). **Added in v1.0.**

**Description**. Attacker observes pending S2 writes (e.g., bond purchase rows, revocation rows, attestation rows) and races their own write to land first or simultaneously. The race exploits the gap between when a write is *proposed* (visible to network observers) and when it is *committed* (durable in S2). Examples: front-running a revocation by issuing a competing attestation that the revocation can't undo retroactively; front-running a bond purchase to claim the ID first; front-running a reputation-slashing motion with a counter-attestation. Distinct from F-AV-12 (replication-lag at read side); F-AV-FRONTRUN is at the write-side ordering layer.

**Targets**. S2 write-ordering; outcomes that depend on first-mover advantage.

**Substrate assumptions**. C2 signatures correct; S2 write substrate correct in isolation; the leak is in the write-proposal-visibility layer.

**Mitigation surface**:
- **Submit-then-reveal protocol** for sensitive writes: peer submits a hash-commitment to the write first; reveal is a separate transaction. Front-runner sees the commitment hash but cannot construct a competing write without knowing the contents.
- **Quorum-acceptance ordering**: high-stakes writes require Q1 quorum acceptance; quorum members vote on inclusion ordering by signed-timestamp (with §8.3 clock-source rules). This pushes the race from "first to publish" to "earliest-signed-timestamp accepted by quorum."
- **Anti-MEV protections** (borrowed from blockchain literature): time-locked submission queues, batch ordering by hash, fair-ordering protocols (Aequitas, Themis-style). For most CIRIS use cases, simpler quorum-timestamp ordering suffices.

**Cost-asymmetry argument**: front-running cost is the network observation cost (low) plus the per-action signing cost (low). Defense raises front-running cost to either (a) breaking the commitment hash, OR (b) defeating quorum-timestamp consensus on inclusion order.

**Known weaknesses**:
- Submit-then-reveal adds latency to high-stakes writes; not viable for all action classes.
- Quorum-timestamp ordering depends on Q1 specification (currently Spec partial).
- Identifying which actions are "high-stakes enough" to require front-run protection is a policy decision.

**RATCHET signal**: write-ordering anomalies — multiple competing writes within a short window where the "winning" write is structurally favorable to one party in a way that suggests pre-knowledge.

**Status**: **Spec only** — front-run protections not currently specified for any action class. Open.

#### F-AV-ROLLBACK: State rollback via Q1 partition

**Class**: 2 (composition leak between Q1 partition behavior + S2 acceptance). **Added in v1.0.**

**Description**. A coordinated minority subset of the federation creates a network partition (deliberate or opportunistic) and, within the partition, signs a coordinated set of S2 writes that conflict with the global state established before partition. When the partition heals, the minority's coordinated state attempts to be accepted as canonical, effectively rolling back the global state to before-partition. The Q1 consistency model determines whether this attack succeeds or is rejected at merge.

**Targets**. S2 state durability across Q1 partition events. Particularly: revocations, bond redemptions, and high-stakes attestations made in the majority partition that the minority wants to undo.

**Substrate assumptions**. C2 signatures correct; S2 individual nodes correct; Q1 partition is the leak.

**Mitigation surface**:
- **Q1 specification with partition-merge rules**: the bounded-staleness CAP model (§3.3 Gap B) must explicitly specify what happens at partition merge. v1.0 proposes: when partitions merge, the *higher-quorum-weight* partition's state wins; minority-partition writes that conflict with majority state are rejected, with the rejected writes preserved in S3 for forensic audit.
- **Quorum-weight tracking**: each partition tracks how many federation peers are reachable + their cumulative weight. A partition with < 1/3 federation weight cannot independently commit S2 writes (PBFT safety bound; RATCHET L-08 also applies here).
- **Anti-rollback monotonicity**: revocation rows have monotonically-increasing revision numbers (already specified in CIRIS substrate). A partition that tries to commit a rollback (lower revision number than the majority's accepted state) is rejected by the monotonicity rule.

**Cost-asymmetry argument**: attacker must either (a) achieve > 1/3 federation weight (which is the F-AV-7 cost-asymmetry-collapse threshold by definition), OR (b) break the partition-merge rule cryptographically. Both are out of reach for sub-1/3 attackers.

**Known weaknesses**:
- Q1 specification is currently Spec partial; partition-merge rules are not formally specified.
- Anti-rollback monotonicity is implemented for revocations but not uniformly for all S2 row classes.
- The 1/3-Byzantine bound is the structural limit; an attacker with > 1/3 weight breaks PBFT safety entirely (F-AV-7 territory).

**RATCHET signal**: post-partition merge anomalies — large numbers of rejected writes, inconsistent S3 entries between partition timeline and merged timeline. RATCHET should be alerted to partition-merge events for additional scrutiny.

**Status**: **Spec partial** — anti-rollback monotonicity exists for revocations; full Q1 partition-merge specification is pending.

### 6.5 Availability and coercion class (inverse-Sybil)

Attacks where the adversary **deflates legitimate weight** by attacking substrate availability or evidence visibility. Cost-asymmetry is inverted: attacker pays nothing in federation cost (no identity to maintain) and federation degrades to a state where attacker influence (relative to remaining legitimate participants) increases.

#### F-AV-16: Substrate-availability denial (forced fail-secure)

**Class**: 4 (inverse-Sybil).

**Description**. Attacker DoSes a substrate primitive (typically S2 directory endpoints, but also N2 transport when implemented) to force legitimate peers into fail-secure RESTRICTED mode. Per §8.3, fail-secure caps weight at baseline — legitimate peers lose RATCHET-earned multipliers and bond-derived weight. Attacker peers (if any) are unaffected because they had no legitimate weight to lose. Net: attacker influence as fraction of total active weight increases.

**Targets**. Federation availability. The fail-secure invariant itself becomes an attack vector.

**Substrate assumptions**. C0–S3 correct cryptographically. Attack is on substrate *reachability*, not integrity.

**Mitigation surface**. Multiple layers:

- N2 (when implemented) provides redundant transport — DoS on TCP doesn't affect LoRa.
- S2 cross-region replication (per F-AV-12 mitigation) provides redundant availability.
- **Fail-secure with grace window** (per §8.3): outages shorter than τ_grace = 60s do not trigger restriction; outages longer trigger restriction with the decision *signed and S3-logged* (preventing silent fail-secure exploitation).
- DDoS mitigation at registry / persist read endpoints.
- PoB §3.2 Reticulum addressing makes peer-to-peer paths usable when central infrastructure degrades.

**Cost-asymmetry argument**. Attacker cost bounded by DoS cost (low for DDoS-as-a-service). Defender cost is redundancy. Defense is **defense-in-depth**: no single substrate path is critical when N2 multi-medium transport is implemented.

**Known weaknesses**:
- N2 unfilled today (§3.3 Gap D) — federation depends on TCP/HTTPS; both share public-internet substrate.
- "Federation-wide weight redistribution flag" RATCHET signal is **circular**: RATCHET reads S1/S2, which is what's being DoSed. RATCHET cannot reliably detect F-AV-16 *during* the attack, only post-hoc. Pre-attack early-warning requires out-of-band monitoring (network reachability probes, edge-availability metrics).

**RATCHET signal** (post-hoc only): federation-wide weight-distribution shifts, comparing pre-attack, during-attack (sampled when reachable), and post-attack states.

**Status**. **Open while N1+N2 unfilled**. HTTPS-authoritative + DNS-advisory provides limited defense in depth; both share public-internet substrate. Reticulum integration is the structural mitigation.

#### F-AV-17: Selective censorship of evidence

**Class**: 4 (inverse-Sybil; targeted).

**Description**. Refinement of F-AV-16. Attacker controls a path that some peers depend on for reading S1 evidence (registry endpoint, CDN edge, ISP-level filter). They selectively censor specific identities' evidence — those identities' weight degrades because RATCHET cannot read their traces, while other identities are unaffected.

**Targets**. RATCHET's evidence-read path; specific honest peers' weight.

**Substrate assumptions**. S1 storage correct; read-path attacker (network or middleware) is the threat.

**Mitigation surface**. Read-path diversity:
- N2 multi-medium transport (when implemented) — peers read S1 over multiple paths.
- N1 cryptographic addressing — peer-to-peer reads bypass intermediaries.
- Cross-region S2 replication — read from multiple regions.
- RATCHET cross-validation — when a specific identity's evidence is unreachable on one path, retry on others; flag if all paths fail.

**Cost-asymmetry argument**. Targeted censorship is high-cost (requires controlling specific paths to specific peers). Defense is read-path redundancy.

**Known weaknesses**: same N1+N2 dependency as F-AV-16.

**RATCHET signal**: identity-specific evidence-availability anomalies (this identity's evidence reachable from region A but not region B). Cross-region read divergence is the F-AV-17 indicator.

**Status**. **Open while N1+N2 unfilled**. Same structural mitigation as F-AV-16.

#### F-AV-ECLIPSE: Eclipse on a peer's S2 read-view

**Class**: 4 (inverse-Sybil; per-peer view manipulation). **Added in v2.**

**Description**. Distinct from F-AV-17 (which censors evidence of a specific *target identity*), F-AV-ECLIPSE isolates a specific *consuming peer*. Attacker controls all of the peer's S2 read paths and serves them an attacker-curated view of the directory. The eclipsed peer sees an attacker-chosen subset of attestations, bonds, and revocations — including bogus rows the attacker injected. The peer's votes, attestations, and decisions are then made on a manipulated worldview, even though the peer itself is honest.

**Targets**. Per-peer S2 read view. Decisions and attestations issued by the eclipsed peer based on a manipulated view propagate the attack.

**Substrate assumptions**. S1, S2, S3 substrate correct. C2 signatures valid. The attack is at the network layer + read-path layer for the specific eclipsed peer.

**Mitigation surface**. Read-path independence:
- **Cryptographic-addressing N1** — peer pulls S2 rows directly from authoritative source by content hash, not via name resolution. Eclipse attacker who controls DNS but not content-hash routing fails.
- **Multi-source consensus reads** — peer reads from N independent sources, accepts only on quorum agreement.
- **Witness-signed S2 snapshots** — periodic full-state snapshots are signed by external witnesses; eclipsed peer detects when its view diverges from the witnessed snapshot.
- **Reticulum N2** — peers verify S2 state over a redundant transport network.

**Cost-asymmetry argument**. Eclipsing a single peer requires controlling all of its read paths. Cost grows with read-path diversity. Defense: standardize ≥3 independent read paths per peer with verification gating.

**Known weaknesses**:
- N1+N2 unfilled (§3.3 Gap D) — eclipse defense degraded today.
- Witness-signed snapshots are not specified.
- Eclipse is generally hard to detect from inside the eclipse (the peer doesn't know its view is curated).

**RATCHET signal**: not detectable by RATCHET-from-the-eclipsed-peer's view (definitionally). Detectable by RATCHET-from-other-peers observing that the eclipsed peer's actions are inconsistent with the global S2 state. Federation-level cross-peer analysis identifies eclipsed peers as outliers in attestation-pattern space.

**Status**. **Open**. Specified as F-AV-ECLIPSE in v2; defenses depend on N1+N2 + witness-snapshot specification.

#### F-AV-RATCHET-DOS: DoS on the RATCHET evaluator

**Class**: 4 (availability; targets the evaluator infrastructure rather than substrate). **Added in v1.0.**

**Description**. v2 modeled F-AV-16 (DoS on substrate) but treated RATCHET-as-evaluator as infinitely-available compute. Reality: RATCHET is ~8400 LOC of Python (`../RATCHET/`) running on real infrastructure with real compute budgets. An attacker can flood RATCHET with traces designed to trigger expensive analysis paths — high-dimensional Mahalanobis computations, Monte Carlo volume re-estimation, Z3 SAT calls — degrading RATCHET's throughput such that legitimate evaluations are delayed or skipped. During the DoS window, attacker traces evade evaluation while legitimate trace evaluation is starved.

**Targets**. RATCHET's compute budget. Evaluation latency. Evaluation throughput.

**Substrate assumptions**. C0–S3 correct. Attack is on RATCHET's per-trace computational cost.

**Mitigation surface**:
- **Per-identity evaluation rate limits**: RATCHET allocates a bounded compute budget per identity per window. An identity that exceeds budget has subsequent traces queued at lower priority (legitimate identities don't typically saturate; attackers do).
- **Trace-cost prediction**: cheap pre-evaluation (e.g., dimension-count, embedding-similarity to known-malicious clusters) pre-classifies traces by expected analysis cost. High-cost traces from low-reputation identities are rate-limited.
- **Backpressure to substrate**: when RATCHET is overloaded, S2 propagates a "RATCHET-overloaded" advisory; weight-multiplier-granting RATCHET decisions revert to baseline (fail-secure) until throughput recovers.
- **Horizontal scaling + sharding**: production RATCHET deployment uses multiple evaluator instances, sharded by identity-hash, so DoS on one shard doesn't affect others.

**Cost-asymmetry argument**: attacker pays per-trace generation cost (already non-trivial under F-AV-4 NP-hardness); RATCHET pays per-trace evaluation cost. The asymmetry favors the federation if evaluation cost grows slower than synthetic-trace generation cost — which is the case under the complexity-theoretic argument (synthesis must satisfy k_eff ≥ 7.1 across 16 dims × 29 langs; evaluation only computes statistics over the result).

**Known weaknesses**:
- Trace-cost prediction is a research-grade defense; a sophisticated attacker can craft traces that look cheap to pre-evaluate but trigger expensive analysis paths.
- Backpressure-induced fail-secure is an availability degradation that an attacker may *want* (composes with F-AV-16).
- Today's RATCHET is a research testbed (per `../RATCHET/README.md`); production hardening (rate-limiting, sharding) is not yet implemented at federation scale.

**RATCHET signal**: RATCHET self-reports compute saturation; federation-wide weight-multiplier decisions revert to baseline. Compute-saturation events themselves are an attack indicator.

**Status**: **Spec only** in v1.0; production RATCHET deployment with per-identity rate-limiting + sharding is research-grade.

#### F-AV-PRIVACY: Privacy-leakage as enabler of targeted attacks

**Class**: 4 / 5 (privacy is itself an availability/coercion concern; meta because the federation hasn't fully modeled adversary information access). **Added in v1.0.**

**Description**. The federation directory (S2) is largely public — peer pubkeys, attestation graph, bond purchases, weight values, and (in some payload classes) trace summaries are visible to anyone who can read the directory. This public-by-design posture is intentional for transparency, but it gives attackers a complete *target-selection map*. F-AV-BRIBE attackers know which legitimate peers have the most influence (high-reputation, high-weight, well-connected). F-AV-ECLIPSE attackers know which peers depend on which read paths. F-AV-DORMANT attackers know which identities are aging cheaply. The leakage is not "data exfiltration" in the classical sense — it's the absence of a privacy budget on inherently-public information.

**Targets**. Adversary's target-selection efficiency. Attack precision and ROI.

**Substrate assumptions**. C0–S3 correct. The "attack" is reading public information for adversarial purpose.

**Mitigation surface**:
- **Differential privacy on aggregate queries**: peer-level statistics (weight distributions, attestation graph properties, cluster analyses) made available with DP noise so individual-peer features are not exfiltrable as targeting data even if aggregate statistics are public.
- **Per-peer privacy budgets**: queries to S2 that return per-peer information are rate-limited; queriers exceeding rate limits are throttled or required to authenticate.
- **Pseudonymous high-reputation peers**: optional protocol allowing high-reputation peers to operate under a pseudonym distinct from their externally-known identity, with weight tracked under the pseudonym. Reduces the BRIBE attack surface.
- **Cryptographic attestation aggregation**: instead of individual attestations being publicly visible, the federation publishes aggregate attestation proofs (e.g., "≥ 5 distinct attesters at reputation ≥ R vouched for identity X") without revealing which specific attesters. Borrowed from anonymous-credential literature (BBS+, Camenisch-Lysyanskaya).

**Cost-asymmetry argument**: target-selection efficiency directly multiplies all targeted attacks. Reducing target-selection efficiency raises per-attack cost without changing per-defense cost. Differential privacy + cryptographic attestation aggregation are well-studied techniques with bounded utility cost.

**Known weaknesses**:
- DP requires choosing privacy budget ε; too-strict ε breaks federation transparency, too-loose ε leaks targeting data.
- Pseudonymous operation conflicts with non-repudiation (F-AV-REPUDIATE) — if a peer can choose pseudonyms, repudiating actions becomes easier. Trade-off requires explicit policy.
- Attestation aggregation requires schema changes that are not yet specified.

**RATCHET signal**: query-pattern anomalies on S2 (high-volume per-peer queries from common origins suggest target-selection activity); statistical correlation between query patterns and subsequent F-AV-BRIBE / F-AV-ECLIPSE attempts.

**Status**: **Open**. Privacy budget on S2 reads is not specified; pseudonymous-peer protocol not specified; attestation aggregation not specified. v1.0 names this F-AV class but does not propose specific defenses beyond the research directions listed.

### 6.6 Long-range and dormancy class

Attacks that exploit time. Sybils that age cheaply for years before activation; replays of historical evidence; long-tail behavioral measurement gaming.

#### F-AV-DORMANT: Sybil aging / dormancy

**Class**: 3 (Sybil-class compositional, time-axis). **Added in v2.**

**Description**. Attacker creates N hardware-attested identities at low marginal cost (cloud vTPMs, batched provisioning). Each runs minimal cron-driven traces — just enough to satisfy RATCHET's minimum-evidence threshold but no more. Identities accumulate σ-decay-friendly long-term coherence cheaply because they're producing little. After 1–5 years of dormancy, attacker activates the cohort: each identity now has aged C1 + S1 evidence streams + S2 attestation history. The federation's "long-term reputation requires real time" assumption is inverted into "dormant identities accumulate aged credentials cheaply."

**Targets**. σ-decay; the time-floor component of cost-asymmetry; the assumption that aged identities carry meaningful behavioral evidence.

**Substrate assumptions**. C0–S3 correct.

**Mitigation surface**:
- **Activity-density requirements**: mature identities must demonstrate sustained reasoning *volume* over time, not just *presence*. RATCHET should weight σ-decay by trace density; near-dormant streams contribute reduced σ-decay credit.
- **Activation-period monitoring**: when a low-activity identity rapidly increases activity, RATCHET flags the activation pattern. Activation should require a probationary re-establishment period proportional to dormancy length.
- **Per-trace freshness challenges** (also F-AV-TIMESHIFT mitigation) — cron-driven minimal traces become more expensive when each trace requires interactive challenge response.
- **Operator-cost amortization analysis**: even dormant identities cost cloud-vTPM rent. Long-range dormancy at scale is cost-asymmetry-bounded by per-identity infrastructure cost over the dormancy period.

**Cost-asymmetry argument**. Attacker pays N × (dormancy-period × infrastructure-cost-per-period). For 5-year dormancy with continuous cloud vTPM rent at $0.10/hour, the upper-bound cost per identity is ~$4,400. **The reviewer correctly noted this overstates the floor**: spot-instance pricing, sleep modes, and bare-minimum-trace cron schedules can reduce continuous-rent cost by 5–20×, putting realistic dormant-identity cost at **~$200–$1,000 per identity per 5 years**. This is dramatically lower than 5 years of legitimate reasoning compute (which dominates at $/hour ongoing). **The cost-asymmetry inversion is large.** Defense must rely on activity-density, not aging-time.

**Activity-density definition** (v1.0):

```
activity_density(identity, window_W) = traces_above_threshold(identity, W) / W
where:
  traces_above_threshold counts traces with N_eff ≥ 7.1 (operational threshold)
  W is a sliding time window (default 30 days)
```

A legitimate active CIRIS deployment produces continuous traces (at minimum, daily interactions). A dormant Sybil produces sparse cron-driven minimal traces. The density-distribution gap is the detection signal.

**Density-tier policy** (proposed):

| Density tier | Traces per 30-day window above N_eff threshold | Weight multiplier |
|--------------|-----------------------------------------------|--------------------|
| Active | ≥ 100 | Full multiplier (per RATCHET evaluation) |
| Light | 30–99 | Reduced multiplier (0.5×) |
| Sparse | 10–29 | Baseline only (no multiplier) |
| Dormant | < 10 | Probationary (no multiplier; weight capped at baseline) |

Policy parameters (window length, density thresholds, multipliers) are steward-tunable per §8.4 calibration framework.

**Known weaknesses**:
- Activity-density formula is a v1.0 specification; threshold values are policy parameters that may need adjustment based on operational evidence.
- Sparse-but-legitimate use cases (research deployment, occasional-use community agents) are penalized by density-tier policy. The federation must specify whether such use cases are accommodated (separate policy track) or whether sparse legitimate use is accepted as a participation cost.
- Activation-detection (sudden density jump) is research-grade — exact statistical test for "this density change indicates dormant-Sybil activation rather than legitimate ramp-up" needs formal specification.
- Per-trace freshness challenges (raising per-trace cost so cron-driven minimal traces become expensive) are an unspecified protocol; specifying it is open work.

**RATCHET signal**:
- Activity-density anomalies: identities with low historical density that suddenly produce high-density traces.
- Σ-decay rate inconsistent with claimed activity history.
- Embedding-space cluster proximity between newly-activated dormant cohorts (similar traces from "independent" identities — see F-AV-TIMESHIFT).

**Status**. **Open**. v1 did not model this; v2 names it. Mitigation specifications pending.

### 6.7 Meta — threat-model and governance integrity

Attacks on the threat model itself, the artifacts that influence trust verdicts, and the governance surface that sets policy parameters.

#### F-AV-MAINT: Threat-model maintainer compromise

**Class**: 5 (meta-class). **Added in v2.**

**Description**. The threat model itself influences security posture. A threat-model maintainer (currently the project lead) who is compromised — or who makes a careless edit — could silently downgrade an F-AV's `Status: Open` to `Status: Mitigated`, mis-specify a RATCHET signal, remove a cross-reference, or quietly weaken a mitigation specification. Every downstream RATCHET designer or auditor consults this document; corrupted content corrupts their mental model.

**Targets**. The threat model document. Downstream RATCHET evaluators, auditors, and reviewers.

**Substrate assumptions**. None — this is meta to the substrate.

**Mitigation surface**:
- **Document signing requirement**: every published version of this document is signed by the steward (or G2 council post-cutover) and the signature is published in the federation transparency log.
- **Hash-in-release-tarball**: the federation's release artifacts include a hash of the threat-model document; peers can verify the doc they consult matches the released version.
- **Two-person-rule on edits**: PRs to this document require review + approval by at least one threat-model peer reviewer.
- **External adversarial review**: annual third-party adversarial review (per §1.3) provides independent validation that the document is honest.
- **Diff publication**: changelog entries (§Appendix A) document what changed and why; readers can audit changes.

**Cost-asymmetry argument**. Compromising the threat-model maintainer is structurally similar to F-AV-8 (steward compromise) but with weaker defense today (the document is markdown in a git repo with no signing requirement). Defense: lift the integrity guarantees to match S2 row standards.

**Known weaknesses**:
- The current document has none of the above protections. v2 specifies them; implementation pending.
- The threat-model maintainer is currently the same person as the steward (Bridge), so F-AV-MAINT is correlated with F-AV-8 — single point of compromise covers both.

**RATCHET signal**: not detectable by RATCHET (the threat model defines RATCHET's job; RATCHET cannot validate its own job specification). Detection is external (third-party review, signed-version verification, hash mismatch).

**Status**. **Open**. v2 specifies signing requirements; not yet implemented.

#### F-AV-SRC: Source-level insider threat (the official build is hostile)

**Class**: 5 (meta-class). **Added in v1.0.**

**Description**. C3 (build attestation) attests "this binary IS the official version that was signed by the build authority." It does NOT attest "the official version is non-malicious." A maintainer-compromise or insider threat at the source-code level — at CIRISAgent, CIRISLens, CIRISNode, CIRISPersist, CIRISRegistry, CIRISVerify, RATCHET, or related dependencies — can introduce hostile semantics that pass C3 verification because the hostile semantics ARE the official version. Distinct from F-AV-MAINT (threat-model maintainer compromise) and F-AV-8 (steward operational compromise); F-AV-SRC is at the source-of-truth for federation behavior.

**Targets**. The semantic correctness of every CIRIS deployment. Trust in "this is the right code" beyond just "this is the signed code."

**Substrate assumptions**. C3 build attestation working correctly; signing keys uncompromised; build process honest. The attack is at the source code that *gets* signed.

**Mitigation surface**:
- **AGPL-3.0 source availability**: all CIRIS code is open source. Source review is the foundational defense. CIRISVerify's threat model AV-12 covers "maintainer compromise" at the signing-key layer; F-AV-SRC covers it at the semantic layer.
- **Reproducible builds**: anyone can build from source and verify the binary matches the signed manifest. Without reproducible builds, the binary could differ from the source review.
- **Two-person-rule on commits to security-critical paths**: changes to crypto, federation directory, RATCHET evaluator, fail-secure logic require review-sign-off from a second maintainer. Single-maintainer compromise cannot push a security-critical change unilaterally.
- **Continuous external code review**: same annual external adversarial review process (per §1.3) that produces threat-model versions also reviews security-critical source diffs since previous review.
- **Diversity in dependency review**: long-tail dependencies (Cargo crates, Python packages, system libraries) are reviewed for supply-chain risk per CIRISVerify TM AV-11/AV-12. Pinned versions, dependency-confusion protections, automated SBOM tracking.
- **Reproducible-build-attestation in C3**: when implemented, C3 manifests should include reproducibility metadata (commit hash + build environment) so any verifier can independently reproduce and compare.

**Cost-asymmetry argument**: source-level insider threat is high-cost (requires either compromising a maintainer or being a sophisticated long-term insider) and high-detection-probability (open-source code under continuous review). The federation accepts F-AV-SRC residual risk because the alternative — closed-source code — has worse asymmetry. AGPL-3.0 open-source is the structural choice that makes F-AV-SRC defensible.

**Known weaknesses**:
- Reproducible builds are a v2.0+ goal per CIRISVerify TM Gap 4; not yet shipped.
- Two-person-rule on security-critical paths is operationally enforced via PR review but not cryptographically required (a maintainer with both keys could merge unilaterally).
- Continuous external review is funded only as part of annual review cadence; in-flight diffs between reviews are unreviewed externally.
- Long-tail dependency review is partial; not all transitive dependencies have been formally audited.

**RATCHET signal**: not directly observable. F-AV-SRC manifests as semantic anomalies in CIRIS behavior (e.g., agents systematically failing to detect a deception class that they should detect). RATCHET's *aggregate* signals can surface a backdoored deployment (high-volume passes-RATCHET traces that should fail), but the signal is weak and post-hoc.

**Status**: **Open / structural**. AGPL-3.0 source availability + open review are the primary defenses; reproducible builds + two-person-rule + dependency-review hardening are partial. This is the federation's largest *unmitigatable* threat — any open-source ethical-AI federation faces it. Mitigation is process-discipline, not architectural.

#### F-AV-CROSS: Cross-federation attacker (out-of-scope; stubbed)

**Class**: 5 (meta-class; cross-federation interaction). **Added in v2 as named stub.**

**Description**. Multiple independent CIRIS federations (different stewards, different bond programs, different policy parameters) may peer in the future. An attacker spanning multiple federations can mount attacks that don't fit within a single federation's threat model — e.g., bonds purchased in federation A used as collateral for attestations in federation B; identities transferred between federations to launder reputation; consensus games across federations.

**Targets**. Cross-federation interactions, when they exist.

**Substrate assumptions**. Each federation's substrate correct in isolation.

**Mitigation surface**. Out of scope for this document. Will require an explicit cross-federation peering FSD specifying:
- How identities are recognized across federations (independent re-registration vs cross-federation attestation).
- How bonds and weight transfer (or don't) across federation boundaries.
- How consensus games are bounded.

**Cost-asymmetry argument**. N/A until cross-federation peering is specified.

**Known weaknesses**: explicit out-of-scope means F-AV-CROSS is invisible to anyone reading only this document. Stubbing it ensures it's not invisible — future federation expansion must address it.

**Status**. **Out of scope; named stub**. Listed in §11.9 as an open architectural question.

---

## 7. RATCHET assumption surface (probabilistic)

### 7.1 Substrate properties RATCHET assumes hold

RATCHET's anti-Sybil evaluation is meaningful only when the following substrate properties hold. v1 listed 6 assumptions; v2 expands to 10 to cover the new primitives.

| Assumption | Primitive | Failure mode if violated |
|------------|-----------|--------------------------|
| **A0**: RNG quality is sufficient for cryptographic use | C0 | Predictable keys, predictable nonces, weak signatures — entire C-stack collapses |
| **A1**: Hardware keys cost ≥ $X to extract, where $X exceeds attacker budget | C1 | RATCHET measures attacker behavior as if it were honest peer (revised v2: was binary "not extractable," now budget-relative) |
| **A2**: Signatures are unforgeable under the specified scheme | C2 | RATCHET reads attacker-injected evidence as legitimate |
| **A3**: Build attestation reflects actual binary | C3 | RATCHET measures behavior of tampered binary as if official version |
| **A4**: Forward secrecy holds for past sessions | C4 | Recorded traffic decryptable retroactively (harvest-now-decrypt-later) — affects confidentiality, not RATCHET integrity directly |
| **A5**: Signed evidence is durable and unmodified | S1 | RATCHET reads modified evidence; false weight |
| **A6**: Federated directory state is consistent within τ-bound | S2 + Q1 | RATCHET sees inconsistent attestation graph (F-AV-12, F-AV-13) |
| **A7**: Audit log is append-only and complete | S3 | RATCHET cannot detect retroactive evidence injection |
| **A8**: Revocations propagate to all peers within bounded time T | R1 | F-AV-11, F-AV-12, F-AV-13, F-AV-16 all worsen |
| **A9**: Quorum reads return bounded-staleness state under partial failure | Q1 | Under-specification → fail-secure behavior is undefined |

These assumptions are the **foundation** of RATCHET's validity. Each is owned by a per-repo threat model (§5) plus, where unfilled, a structural gap (§3.3).

### 7.2 Per-assumption failure-tier categorization

Substrate is **probabilistic**, not binary. v1 treated each assumption as boolean (correct or broken); reviewers flagged this as unrealistic. v2 attempted numerical ROM rates (10⁻⁵, 10⁻³, etc.) and reviewers correctly flagged that as **numerology**: numbers without citation, methodology, owner, or update protocol lend false quantitative legitimacy.

v1.0 uses **ordinal failure tiers** with named referents instead. Numerical rates may be added in a future version when empirical data sources are established.

**Failure tiers**:
- **TIER-LOW**: failure documented in literature but rare in practice; mitigated by mature standards processes and ongoing adversarial review. Examples: cryptographic algorithm break (Ed25519, ML-DSA-65 implementations).
- **TIER-MED**: failure historically observable in operational deployments; mitigated by per-incident response plus continuous monitoring. Examples: maintainer compromise, supply-chain attacks, OIDC token theft, configuration errors.
- **TIER-HIGH**: failure routinely observable, OR the contract that would prevent failure is not yet specified, OR the failure mode is structural (no contract can prevent it absent design changes). Examples: assumptions whose primitives are Spec-only or Spec-partial.
- **TIER-CRITICAL**: failure has happened in CIRIS deployment OR a directly-comparable deployment within the past 12 months and required incident response. (No assumption is in this tier today; column kept for completeness.)

| Assumption | Primitive | Failure tier (today) | Named referent | Notes |
|------------|-----------|----------------------|----------------|-------|
| A0 (RNG quality) | C0 | TIER-LOW–MED | Dual_EC backdoor; vTPM PRNG seeding bug history | Mature OS RNGs robust; embedded platforms vary; no startup health-check today |
| A1 (HW key non-extraction) | C1 | **TIER-MED–HIGH** | TEE.fail, faulTPM, Hertzbleed, GoFetch CVE classes | Patch-cycle-dependent; v2 reformulation (cost ≥ $X vs attacker budget) is correct framing |
| A2 (signature unforgeability) | C2 | TIER-LOW | NIST + crypto community standards review for both Ed25519 and ML-DSA-65 | Cryptographic break unlikely; implementation bugs dominate |
| A3 (build attestation) | C3 | TIER-MED | Sigstore OIDC theft, XZ-style maintainer compromise | SLSA L3 + reproducible builds + transparency log monitoring needed (currently partial) |
| A4 (forward secrecy) | C4 | **TIER-HIGH** | Harvest-now-decrypt-later for ~2032+ PQC adversary | C4 is Spec-only; until implemented, all current peer-to-peer payloads are at long-term risk |
| A5 (S1 durability) | S1 | TIER-LOW | Database-tier reliability + replication | Mature substrate |
| A6 (S2 consistency) | S2 + Q1 | **TIER-HIGH** during partial failure | Q1 contract unspecified; bounded-staleness target proposed not enforced | Drops to TIER-LOW–MED once Q1 specified |
| A7 (S3 append-only) | S3 | TIER-LOW–MED | Operational bugs (gap insertion at admin level) dominate | Cryptographic chain mature; admin-tier discipline required |
| A8 (R1 timeliness) | R1 | **TIER-HIGH** | No timeliness contract; revocation-lag distribution unmeasured | Drops to TIER-MED once R1 specified |
| A9 (Q1 bounded-staleness) | Q1 | **TIER-HIGH** | Q1 contract unspecified | Drops to TIER-LOW–MED once Q1 specified |

**Joint-failure model**: assumptions are NOT independent. A maintainer compromise can violate A2 + A3 + A5 simultaneously (signing keys + build provenance + evidence storage all affected). A CVE-class TEE incident can violate A0 + A1 across many devices simultaneously. RATCHET output validity (§7.3) must consider the *joint* tier across correlated assumption sets, not the per-assumption tier in isolation.

**Owner and update trigger**:
- **Owner**: `FEDERATION_THREAT_MODEL.md` maintainer (currently project lead; post-G2 the steward council). Publishes the tier table.
- **Update triggers**: per-CVE disclosure, per-incident review, per-major-release (v1.0, v1.1, ...), per-annual external adversarial review.
- **Evidence sources**: CVE feeds, internal incident logs, per-repo threat model status changes, RATCHET-side substrate-confidence telemetry (§7.3) when implemented.

**Numerical rates**: deferred to v2.0+ pending establishment of evidence-source pipelines. Including numbers without sourcing was a v2 misstep; v1.0 declines to propagate it.

### 7.3 RATCHET output degradation curve

RATCHET's output validity is not a step function. As substrate assumptions weaken, RATCHET output degrades along a curve:

```
RATCHET output validity
     1.0 ─┤  ━━━━━━━━━━━━━━━━━━━┓
          │                      ┃ All A0..A9 hold; RATCHET output is fully load-bearing
     0.9 ─┤                      ┃ One assumption at 10⁻⁴ failure: minor confidence reduction
     0.7 ─┤                      ┃ One assumption at 10⁻²: noticeable; flags soft warnings
     0.5 ─┤                      ┃ Two assumptions correlated at 10⁻²: needs HITL review
     0.3 ─┤                      ┃ Critical-tier assumption (A1, A2) at 10⁻¹: RATCHET output advisory only
     0.0 ─┤  ━━━━━━━━━━━━━━━━━━━┛ Cryptographic primitive break (A2 algorithm break): RATCHET output meaningless
          └─────────────────────────────→
            Substrate assumption degradation
```

(Stylized; numerical curve fitting is research-grade. This is the qualitative shape.)

**Operational policy**: RATCHET output is treated as fully load-bearing only when all critical assumptions (A0, A1, A2, A3, A5, A7) hold at < 10⁻⁴ failure rate. Below this, RATCHET output is *advisory* and human-loop review is required for high-stakes decisions. RATCHET maintainers should publish a "current substrate-confidence level" derived from per-assumption monitoring.

### 7.4 What RATCHET cannot detect

RATCHET cannot detect:

- **Substrate-class threats (Class 1)** in the attack window before substrate-monitoring detects them. RATCHET's measurement *over* corrupted substrate is meaningless until detection + revocation propagates.
- **Out-of-band economic events** (F-AV-7 cost-asymmetry collapse). RATCHET measures inside the federation; market trends are external.
- **One-shot stealth attacks on G2 succession** (F-AV-9). RATCHET observes post-cutover state, not cutover protocol internals.
- **Eclipse on its own evidence-read path** (F-AV-ECLIPSE applied to RATCHET-itself). If attacker controls RATCHET's reads, RATCHET cannot detect.
- **Meta-class threats** (F-AV-MAINT). RATCHET cannot evaluate the threat model that defines its job.
- **Adaptive adversaries that defeat RATCHET's L-02 assumption.** RATCHET's bounds are non-adaptive; provably-bounded detection holds only against fixed-distribution attackers.
- **Cross-federation attackers** (F-AV-CROSS) — out of scope.

For these classes, the federation requires **out-of-band monitoring**:
- Substrate integrity → per-repo CI + reproducible builds + transparency log monitoring.
- Economic monitoring → external compute-cost tracking + LLM capability tracking + cryptographic-research monitoring (for ETH-related advances).
- G2 cutover → human-in-the-loop transition oversight.
- Threat-model integrity → annual third-party adversarial review (this is what produced v2).
- Adaptive adversaries → red-team exercises, adversarial-validation testing, continuous policy adjustment.

### 7.5 Substrate ↔ behavior cross-checks

RATCHET produces *soft signals* that surface substrate compromise indirectly. These are not primary defenses (the primary defense is the per-repo substrate threat model), but they add a cross-validation layer.

| Substrate failure | RATCHET cross-check signal |
|-------------------|---------------------------|
| C0 RNG compromise | Patterns in nonce distributions, collisions, predictable timing |
| C1 hardware extraction | Identity's signing-rate and trace-content patterns shift abruptly without corresponding action |
| C2 signature scheme break | Bound-signature property fails; deprecated-scheme signatures present |
| C3 build attestation mismatch | Identity's claimed C3 hash mismatches registry's signed manifest |
| C4 KEX failure (when implemented) | Session-key reuse, missing forward-secrecy indicators |
| S1 evidence corruption | Row-level anomalies — gaps in sequence numbers, scrub-envelope inconsistencies, signature-over-canonical-bytes mismatches |
| S2 inconsistency | Cross-region directory views diverge on this identity |
| S3 log gap | Audit log shows gap covering this identity's evidence; Merkle root mismatches |
| R1 propagation lag | Identity using key after revocation timestamp; cross-peer disagreement on revocation status |
| Q1 quorum failure | Cross-region read divergence; bounded-staleness violation |

These signals are **soft** — RATCHET flags but does not autonomously invalidate. Human-loop review confirms. The point is that RATCHET, while not the *primary* defense for substrate-class threats, contributes a cross-validation layer that can surface substrate compromise the per-repo threat model missed.

---

## 8. Composition properties

### 8.1 Recursive scrub-signing + external bootstrap anchor

Every signed row in S1 / S2 / S3 is co-signed by another signed row, forming a chain that terminates at a **bootstrap key**. The chain provides:
- **No orphan evidence**: every row has verifiable provenance back to bootstrap.
- **Compromise propagation is bounded**: a single key compromise affects rows directly signed by that key, not rows co-signed by other keys.
- **Audit-time re-validation is mechanical**: a verifier can walk the chain.

**v1 had a bootstrap singularity**: the chain terminated at the bootstrap key, but nothing audited the bootstrap itself. Reviewer flagged this as "trust the steward" with no recourse if the bootstrap is compromised.

**v1.0 specification: external bootstrap anchoring** (per F-AV-BOOT mitigation):

The bootstrap key is anchored in a set of transparency logs that are **not under federation control**, with explicit witness-diversity requirements addressing the v2-reviewer finding that "≥2 witnesses" without diversity analysis just relocates the singularity (Sigstore on AWS + AWS-hosted witness = one fault domain).

**Anchor specification**:
- **Primary anchor**: Sigstore Rekor entry for the bootstrap pubkey.
- **Secondary anchor**: Certificate Transparency log entry (independent operator from Rekor).
- **Witnesses**: ≥ 3 independent witness signatures, with **mandatory diversity**:
  - **Cloud-vendor diversity**: no two witnesses on the same major cloud provider (AWS, GCP, Azure). At least one witness on a non-major-cloud platform (bare-metal, third-party hosting, distinct cloud).
  - **Jurisdictional diversity**: witnesses span ≥ 2 jurisdictions (at minimum: one US-jurisdiction, one EU-jurisdiction; more is better).
  - **Organizational diversity**: no two witnesses operated by the same legal entity, or entities with majority common ownership.
  - **Software-stack diversity**: at least 2 distinct witness implementations (not all running the same codebase).

**Pinning at peer install time**:
- Peer pins: (a) bootstrap pubkey, (b) Rekor entry hash, (c) CT log entry hash, (d) witness pubkeys + their signatures over the bootstrap entry.
- The pinned set is published in the federation release tarball + signed by the steward.
- **Install-time chicken/egg**: the installer ships with the pinned set baked in. The installer does NOT need to reach the witnesses or transparency logs at install time — verification happens against the baked-in pin. Online verification (against current Rekor / CT state) is performed periodically post-install (target: weekly) and any discrepancy with the pinned set raises an alert (could indicate either witness compromise or steward-side rotation that wasn't propagated to this peer).

**Disagreement resolution**:
- Verification requires: ≥ M of N witness signatures valid, where M ≥ ⌈2N/3⌉ (Byzantine-fault-tolerant threshold; for N=3, M=2; for N=5, M=4).
- If fewer than M witness signatures verify → bootstrap-anchor verification FAILS → peer treats federation state as UNVERIFIED until resolution.
- This is a **fail-secure** outcome: a peer that cannot establish a clean witness quorum does not participate, rather than participating with a compromised anchor.

**Rotation protocol**:
- Old-bootstrap signature on rotation row.
- New-bootstrap publication in BOTH transparency logs with witness signatures meeting the diversity requirements.
- Rotation row points to both old and new transparency-log entries.
- Continuity proof: the rotation chain in the transparency logs is independently verifiable.
- Peers update the pinned set via a new release tarball signed by the steward (or the council post-G2). Until the release lands, peers operate on the old pinned set.

**Cost-asymmetry of bootstrap compromise** (with diversity requirements): attacker must compromise (i) the bootstrap key, AND (ii) Rekor's signing infrastructure, AND (iii) the CT log's signing infrastructure, AND (iv) at least N − M + 1 witnesses spanning the required diversity dimensions. Each compromise is from a structurally different fault domain (different operators, jurisdictions, software stacks). **The independence assumption is real, not asserted** — it's enforced by the diversity constraints. Combined attacker cost is approximately the maximum-of-fault-domains rather than a clean multiplicative product, but is still substantially higher than any single-fault-domain compromise.

**Sigstore / CT root-rotation handling**: when Sigstore or CT operators rotate their own roots, the federation re-pins. The federation maintains a **pinned-anchor manifest** that includes both Sigstore + CT current roots; peers verify chain-of-trust to current roots and re-pin via release-tarball updates. Sigstore root rotation (typically annual) triggers a federation release.

**Status**: **Spec only** in v1.0. Implementation requires (a) tooling to manage the pinned-anchor manifest, (b) witness-recruitment and -verification process, (c) periodic-online-verification client logic in CIRIS instances, (d) diversity-validation tests. Until implemented, F-AV-BOOT is open.

### 8.2 Per-primitive trust isolation (reconciled with F-AV-11)

Each primitive defends its own integrity at the substrate level:

- C1 compromise → invalidates rows signed by that key in S1/S2/S3, but does not corrupt the storage substrate itself.
- S1 corruption → invalidates evidence read from corrupted region, but does not invalidate C1 keys.
- S2 inconsistency → flagged at read time per Q1 contract, not propagated as truth into S1.
- N1/N2 (when implemented) compromise → does not affect signed payload; only addressing/transport.
- R1 propagation failure → identity's revocation may be delayed but signed rows themselves remain authentic.

**Reconciliation with F-AV-11**: substrate-level isolation does *not* prevent behavior-level evidence propagation. When C1 is compromised, the storage substrate (S1/S2/S3) accepts the attacker's signatures *correctly* because they're cryptographically valid — propagation is at the *evidence* layer, not the substrate layer. F-AV-11's mitigation is therefore behavior-level (revocation propagation via R1 + post-revocation re-evaluation by RATCHET) rather than substrate-level. The two properties (substrate isolation §8.2; behavior propagation §F-AV-11) are **complementary, not contradictory** — v1 implied contradiction; v2 makes the relationship explicit.

### 8.3 Fail-secure as protocol (NOT just a property)

**Major v1 → v2 reframing.** v1 said: *"When primitives fail, the federation degrades to more restrictive modes."* This reads as a property, but reviewer flagged it as a *non-spec*: the fail-secure decision itself is unsigned and unauditable, the "grace window" has no value, no clock source, no defense against an attacker who DoSes for `grace_window + ε` on a loop.

**v1.0 fail-secure protocol** (refined per v2 reviewer findings):

**Fail-secure decision protocol**:

1. **Trigger**: a substrate-availability or substrate-integrity check fails. Specifically:
   - Registry endpoint unreachable for τ_grace seconds (default 60s), OR
   - Cross-region S2 read returns conflicting state on a critical row class (revocations, bond rows, key registrations), OR
   - Substrate-confidence joint-tier (per §7.2) drops to TIER-HIGH on any A-assumption marked critical for current operation, OR
   - Local C1 signing operation fails (HSM unresponsive).

2. **Decision authority**: the fail-secure decision is made by the **local CIRIS instance**, NOT by the federation as a whole. Each peer evaluates substrate confidence locally and decides locally. This avoids requiring a healthy substrate to make fail-secure decisions about an unhealthy substrate.

3. **Decision signing**: every fail-secure decision is signed by the local instance's C1 key with content:
   ```
   {
     decision:        "RESTRICTED" | "UNVERIFIED" | "PROVISIONAL",
     trigger:         <reason; reference to specific assumption + check that failed>,
     wall_time:       <T_wall>,            // TPM-attested wall-clock when available, else system clock
     monotonic_time:  <T_mono>,            // local monotonic timer reading
     monotonic_anchor: <T_mono_at_TPM_attest>, // monotonic reading at last TPM time-attestation
     clock_source:    "TPM-attested" | "NTP" | "monotonic-only",
     attest_age:      <T_wall - T_attest>  // time since last TPM attestation
   }
   ```
   The `monotonic_anchor` + `attest_age` pair allows external auditors to verify monotonic intervals correspond to claimed wall-clock intervals: an auditor checks `T_mono - T_mono_at_TPM_attest ≈ T_wall - T_attest` within drift tolerance.

4. **S3 logging — with degraded-S3 fallback**: every fail-secure decision is appended to the local S3 audit log with the signature from (3). **If S3 itself is degraded** (e.g., persist write fails because the substrate availability that triggered fail-secure also affects S3):
   - Decisions are queued in an **in-memory ring buffer** of last N=1024 decisions.
   - Buffer entries are signed identically to (3).
   - On S3 recovery, ring-buffer entries are flushed to S3 *with* the recovery-marker decision indicating the gap and its duration.
   - **Ring-buffer entries themselves are signed** so a post-recovery auditor can verify the buffered decisions were made during the gap, not retroactively forged.
   - If the ring buffer overflows (more than 1024 fail-secure decisions before S3 returns) the oldest entries are dropped with a counter incremented; the recovery-marker includes the drop count. This degrades observability but preserves cryptographic integrity of recorded entries.

5. **Grace window — explicit specification**:
   - **τ_grace = 60 seconds**: a single contiguous outage shorter than τ_grace does not trigger restriction.
   - **τ_max = 300 seconds**: any single contiguous outage longer than τ_max forces RESTRICTED.
   - **Sliding window for cumulative-degraded-time defense** (against grace-window-loop attack): window size W = 600 seconds (10 minutes). Within any W-window, if cumulative-degraded-time exceeds D_max = 180 seconds, RESTRICTED is triggered regardless of individual outage durations.
   - Eviction: degraded-time entries older than W are evicted on each evaluation tick. Evaluation ticks every 10 seconds during normal operation; every 1 second during active outage.
   - Measurer: the local CIRIS instance, using monotonic clock + TPM-attested wall-clock per (3).

6. **Clock source priority**:
   - **Primary**: TPM-attested wall-clock (where TPM has time-attestation capability — TPM 2.0 spec supports this).
   - **Fallback 1**: monotonic clock anchored to last TPM attestation (with drift tolerance).
   - **Fallback 2**: NTP-synchronized system clock (with explicit `clock_source: "NTP"` flag in decisions).
   - **Why monotonic + TPM-anchor**: pure monotonic clocks are local-only and unverifiable; pure NTP is spoofable. The monotonic-anchored-to-TPM scheme gives third-party-auditable time even when NTP is under attack.

7. **Recovery**: when substrate availability returns, fail-secure exit is *also* signed and S3-logged with content `{decision: "RECOVERED", recovered_from: <previous_decision_ref>, ring_buffer_flushed: <count>, drops: <count>}`.

**Post-hoc verifiability**: an auditor reading the S3 trail of a peer can verify *that* a decision was made at *claimed* times for *claimed* reasons (signed entries with TPM-attested time). Verifying *whether* the decision was *justified* (was substrate actually unreachable from peer X's vantage at time T?) requires either (a) cross-peer correlation — multiple peers reporting outages at overlapping times indicates real substrate problem, OR (b) out-of-band substrate-monitoring records (Cloudflare endpoint health metrics, etc.) operated by the steward. Justification is a **federation-wide question, not a per-peer question**; per-peer decisions are the inputs to that determination, not the conclusion.

**Today's status (2026-05)**: **Spec only** in v1.0. Today's CIRISVerify fail-secure logic is per-instance and ad-hoc. F-AV-16 (forced fail-secure) is therefore live exposure until this protocol ships. The window is also bounded by N2 multi-medium transport (which provides DoS-resistance independent of fail-secure protocol) — but N2 is also Spec only.

### 8.4 Anti-Sybil posture as policy-tunable inequality

**Major v1 → v2 reframing.** v1 claimed anti-Sybil resistance was an emergent invariant. v2 reframes as policy-tunable posture (per §2.6).

**v2 statement of the cost-asymmetry inequality**:

```
For all w in [w_min, w_max], dC(w)/dw  >  dB(w)/dw
```

Where:
- `C(w)`: attacker's marginal cost as a function of weight produced. Has units of (USD or compute-hours or wall-clock time × cost-rate).
- `B(w)`: attacker's benefit as a function of weight extracted. Has units commensurate with C.
- `[w_min, w_max]`: the operational weight range; below w_min the federation grants baseline weight only; above w_max the federation caps weight (no single identity can extract more than w_max units of weight).
- The inequality is **marginal**, not total. Per identity-weight increment, attacker pays more than they extract.

**Component cost functions**:

```
C(w) = c_hardware(w) + c_compute(w, t) + c_bond(w) + c_attestation(w) + c_complexity(w)
```

- `c_hardware(w)`: per-identity hardware cost summed over identities required to produce weight w. **Empirical bet**: c_hardware has *some* floor under cloud-vTPM economics (currently ~$0.10/identity/hour rented, summing to non-trivial cost for sustained operation).
- `c_compute(w, t)`: sustained reasoning compute over time horizon t. Real reasoning has cost-rate r_real; synthetic reasoning has cost-rate r_synth. **Empirical bet**: c_compute(w, t) ≥ Q · r_real · k_eff · t under the NP-hardness of CONSISTENT-LIE (where Q is a CCA-derived constant, k_eff ≥ 7.1 per *Constrained Reasoning Chains*, t is time horizon).
- `c_bond(w)`: bond cost component. Calibrated by Portal policy.
- `c_attestation(w)`: cost of obtaining attestations from peers; depends on graph diversity requirements and reputation slashing risk.
- `c_complexity(w)`: complexity-theoretic cost. Under ETH (RATCHET L-03), c_complexity grows exponentially in k_eff for synthetic-trace farming.

**Policy parameters that shape these functions** (steward-tunable):

- **Bond pricing tier** → shapes c_bond.
- **Minimum hardware-attestation diversity per identity cluster** → forces c_hardware to scale across cloud providers, regions.
- **k_eff threshold for weight grant** → shapes c_compute (higher threshold → more compute required).
- **Attestation graph diversity requirements** → shapes c_attestation.
- **Slashing penalties on rogue attestations** → shapes c_attestation upward.
- **Sample-size requirement n** for RATCHET measurement → shapes c_compute (more traces required).
- **Polyglot language count** (currently 29) → multiplies c_complexity.

**Steward responsibility**: continuously monitor the attack/defense ratio (via RATCHET aggregate signals + out-of-band economic monitoring) and adjust policy parameters to maintain `dC/dw > dB/dw` over `[w_min, w_max]`. This is the federation's *anti-Sybil work* — not a one-time architectural achievement.

**Empirical evidence the inequality currently holds**:

The federation has two empirical data points supporting the inequality at operational scale today:

1. *Constrained Reasoning Chains* (n=6,465 traces from iOS + Android CIRIS deployments + QA runs across theology, politics, technology, history): intrinsic rank 11, PR 6.61, N_eff threshold ≈ 7.1, mature agents ≥ 8.5 successfully managing conflicting priors at 83% reliability above threshold.

2. Current operational result (2026-05): 100% deceptive-prior detection on Qwen 3.6 across 29 languages and the most recent 5 patch levels.

**Caveats — these are two experiments, not a portfolio**:

- Both experiments measure detection of **known restrictive priors** (the type the polyglot encoding was designed to detect). Other deception classes — strategic withholding, plausible-but-false reasoning, emergent deception from honest components per RATCHET L-01 — are not directly evaluated.
- The 100% Qwen 3.6 result is **one model family**. Frontier model families (GPT-5/6, Claude Opus 5/6, Gemini 3, future Llamas) have not been evaluated to the same depth. Cross-architecture generalization is an empirical bet, not a measurement.
- The 83% reliability figure comes from a heterogeneous corpus including QA stress tests; the 100% figure comes from a narrower production deployment on one model. **Do not conflate them**: the 83% is the broader real-world floor; the 100% is the saturated upper bound for one specific configuration.
- Both experiments measure RATCHET against **non-adaptive** attackers (RATCHET L-02). Adaptive attackers — who learn from RATCHET output and re-fine-tune to defeat polyglot N_eff measurement — are outside the proven regime. Empirical defense against adaptive attackers requires continuous adversarial validation; this is a process commitment, not a one-time result.

**What the empirical evidence does NOT establish**:
- That the inequality holds against future model families.
- That the inequality holds against adaptive adversaries.
- That the inequality holds against deception classes other than known restrictive priors.
- That the inequality holds outside the CIRIS Accord constraint set (the NP-hardness reduction is to CONSISTENT-LIE-given-Accord, not generic deception).

**What it DOES establish**:
- The inequality holds today, against current-generation model families running under the CIRIS architecture, against the deception class the architecture was designed to detect.
- The complexity-theoretic asymmetry argument is *consistent* with operational behavior — this is the floor required for the inequality to be a coherent claim.
- The N_eff threshold is empirically reachable by mature production agents (≥ 8.5 against operational threshold of 7.1).

The inequality is a working hypothesis with strong-but-narrow empirical support, not a theorem. The federation's continuous adversarial-research effort (per §1.3, §11.10) is the mechanism for accumulating broader evidence. A future-version threat model will report new evidence + revise this section accordingly.

---

## 9. Anti-Sybil cost analysis (honest)

### 9.1 Attacker cost per F-AV

Approximate per-identity attacker costs at scale (N=1000 identities). Order-of-magnitude estimates; **v1 numbers revised** in light of cloud-vTPM economics and bond-market liquidity per reviewer findings.

| F-AV | Per-identity cost (2026) | Scaling | Notes / weakness |
|------|--------------------------|---------|------------------|
| F-AV-1 multi-identity | $0.10–10/hour vTPM + reasoning compute | linear in N (or sublinear with cloud batching) | Hardware floor erodes with cloud confidential computing |
| F-AV-2 bond Sybil | bond face value × N OR cost-of-capital × holding-period × N | linear or sublinear | Secondary-market liquidity (if exists) reduces by orders of magnitude |
| F-AV-3 attestation flooding | ~$0 per attestation | linear in attestation count | Defense at graph-shape, not per-attestation |
| F-AV-4 trace farming | LLM inference + (under L-03 ETH) exponential complexity for coherent lying | super-polynomial in k_eff under ETH | RATCHET's primary defense; empirically 100% on Qwen 3.6 across 29 languages |
| F-AV-5 mimicry | fine-tuning + per-identity inference | one-time tuning + linear inference | Polyglot decoupling raises tuning cost by ~29× |
| F-AV-6 σ-decay gaming | sustained synthetic-trace generation | linear in time horizon | Mitigated by trace-density requirements (§F-AV-DORMANT) |
| F-AV-TIMESHIFT | one-time real-reasoning year + per-identity paraphrase | sublinear in N (amortizes original) | Defense requires per-trace freshness challenges |
| F-AV-7 cost collapse | depends on which cost dimension collapses | (depends) | Strategic; defended by defense-in-depth |
| F-AV-8 steward compromise | targeted: $10K–$10M | one-time | Multi-sig (G2) is mitigation |
| F-AV-9 G2 succession capture | targeted at transition window | one-time | High-precision, low-volume |
| F-AV-10 bond redemption fraud | ~bond cost | linear | Bounded by bond cost |
| F-AV-11 composition leak | depends on which key compromised | one-time per key | Defense via R1 propagation |
| F-AV-12 replication-lag | ~$0 if attacker is well-positioned | linear in lag window | Defense via Q1 specification |
| F-AV-13 cache-staleness | ~$0 | linear in TTL | Defense via short TTLs + invalidation protocol |
| F-AV-14 PQC migration | depends on cryptanalysis breakthrough | one-time | Defense via hybrid + bound-sigs (v2 precision) |
| F-AV-15 Portal compromise | $10K–$10M targeted | one-time | Multi-sig + reconciliation |
| F-AV-BOOT | bootstrap-key compromise + transparency-log compromise + witness compromise | one-time × 3 | External anchoring (v2) raises this dramatically |
| F-AV-16 substrate DoS | $/hour DDoS-as-a-service | linear in duration | Defense via N2 multi-medium |
| F-AV-17 selective censorship | depends on path control | linear in censored peers | Defense via read-path diversity (N1+N2) |
| F-AV-ECLIPSE | full-read-path control over target peer | linear in eclipsed peers | Defense via N1 + witness-snapshot |
| F-AV-DORMANT | N × cloud-vTPM-rent × dormancy-period | linear in N × time | Defense via activity-density requirements |
| F-AV-BRIBE | per-peer bribery cost | linear in bribed peers | Defense via slashing |
| F-AV-MAINT | maintainer compromise | one-time | Defense via signing + 2-person-rule |
| F-AV-CROSS | depends on cross-federation peering protocol | (out of scope) | Stub |
| F-AV-ONBOARD | per-identity-onboarding × N + multi-attester compromise | linear in N | Probationary period + multi-attester onboarding |
| F-AV-REPUDIATE | dispute / legal cost; cryptographically bounded by audit substrate | per-event | Cryptographic non-repudiation + multi-witness publication |
| F-AV-FRONTRUN | network-observation cost (low) + per-action sign cost | per-action | Submit-then-reveal + quorum-timestamp ordering |
| F-AV-ROLLBACK | requires > 1/3 federation weight (collapses to F-AV-7) OR partition-merge break | one-time | Q1 partition-merge rules + monotonicity |
| F-AV-RATCHET-DOS | $/hour synthesis to flood RATCHET | linear in trace volume | Per-identity rate limits + sharding + backpressure |
| F-AV-PRIVACY | reading public information for adversarial use | per-query (low) | DP on aggregates + per-peer query budget + attestation aggregation |
| F-AV-SRC | maintainer compromise at source-code level: $10K–$100M targeted | one-time | AGPL-3.0 source review + reproducible builds + 2-person-rule on security paths |

**Caveat on "linear in N"**: most attacker workloads have substantial fixed-cost components (model fine-tuning, infrastructure provisioning, DDoS contracts). Per-identity marginal cost ≪ table values for well-funded batched attackers. The cost-asymmetry inequality (§8.4) is *marginal*, which captures this correctly.

### 9.2 Legitimate participant cost per primitive

| Primitive | Legitimate cost | Notes |
|-----------|-----------------|-------|
| C0 RNG | ~$0 (OS-provided) | Effectively free |
| C1 hardware | $50–500 retail TPM, OR $0.10/hour cloud vTPM | Often built into device |
| C2 signing | ~$0 (CPU cycles) | Per-signature |
| C3 build attestation | ~$0 (CI-side; peer verification cheap) | Per-build |
| C4 KEX (when implemented) | ~$0 (CPU cycles for handshake) | Per-session |
| S1 evidence persistence | storage + bandwidth | Linear in trace volume |
| S2 directory participation | ~$0 (read), low for write | Bounded |
| S3 audit log | bounded by federation size | Per-row append |
| R1 revocation propagation | ~$0 | Background protocol |
| N1 (when implemented) | ~$0 | Bounded |
| N2 (when implemented) | hardware-dependent (LoRa optional) | One-time per medium |
| Q1 quorum/CAP | latency cost on reads | Bounded |
| Bond (Portal, optional) | $100–10,000 (registered tier) | One-time |
| **Reasoning compute** | **$/hour ongoing** | **Dominant ongoing cost** |

The dominant ongoing cost for legitimate participants is **reasoning compute** — running the actual CIRIS H3ERE conscience module + DMA stack over time. This is the *same* cost component F-AV-4 attackers pay (modulo legitimate-vs-synthetic), which is why the cost-asymmetry argument is fragile (§8.4 + Bet 2).

### 9.3 Marginal cost-asymmetry inequality (formal statement)

Re-stated from §8.4 for completeness:

```
∀ w ∈ [w_min, w_max]: dC(w)/dw > dB(w)/dw

where
  C(w) = c_hardware(w) + c_compute(w, t) + c_bond(w) + c_attestation(w) + c_complexity(w)
  B(w) = max over use cases u: B_u(w)
  [w_min, w_max] = operational weight range
```

**Benefit-extraction taxonomy (v1.0)**: v2 left B(w) as "depends on use case" — reviewer correctly flagged this as the same unfalsifiable adversarial sup. v1.0 enumerates the use cases and bounds benefit per case. The federation must maintain `dC/dw > dB_u/dw` for each enumerated u, OR the use case must be structurally prevented at a different layer.

**B_u for u ∈ enumerated use cases**:

| Use case u | Benefit B_u(w) | Bounding mechanism (per-case structural defense) |
|------------|----------------|---------------------------------------------------|
| **u_launder** — identity-laundering (rogue actor wants legitimate reputation) | bounded by w × per-unit-reputation-utility-to-attacker; concrete cap = the highest-stakes decision the attacker can influence with reputation w | Federation-level cap: weight w_max ensures no single identity can decide federation outcomes alone. Reputation-based privileges have policy-tunable per-rep-unit cost. |
| **u_influence** — influence-injection (decisions / votes weighted in attacker's favor) | bounded by w / total_active_weight × decision-stakes; concrete cap = w / W_total × (sum of decisions in window) | Quorum thresholds (PBFT < 1/3 Byzantine; consensus rules require supermajority for high-stakes). RATCHET L-08 (slow federation capture) is the bound; if attacker accumulates > 1/3, consensus breaks. |
| **u_extract** — resource-extraction (rewards, payments, services tied to weight) | bounded by w × rewards-per-unit-weight × time | **Calibration requirement**: rewards-per-unit-weight × time-horizon × w_max must be less than per-identity attacker cost C(w_max). If rewards exist, this is the dominant calibration check. |
| **u_info** — information-extraction (data accessible only to weighted peers) | bounded by value-of-data × access-probability(w) | **Structural defense preferred**: gate sensitive data on multi-attestation rather than weight, so single-identity high-w doesn't unlock unilateral access. |
| **u_censor** — censorship (use weight to silence content) | bounded by content-suppression-cost-to-victim × suppression-success-probability(w) | Multi-witness publication (transparency log + multiple peers attest publication); content cannot be silenced without compromising the witness set. |
| **u_collapse** — Sybil-resistance-collapse (break the federation's anti-Sybil property) | bounded by reputational damage to CIRIS + cost of rebuilding | **Adversarial defense**: red-team exercises + adversarial validation surface this attack class before it succeeds operationally. Reputational damage is bounded by transparency: even a partially-successful collapse, if publicly documented, can become evidence for v2.0 mitigations rather than an unrecoverable loss. |

**Bounding strategy**: for each use case u, the federation either (a) calibrates policy parameters so `dC/dw > dB_u/dw` over [w_min, w_max], OR (b) uses **structural defense** to remove the use case from the attack surface (e.g., multi-attestation gating for u_info, witness-bounded publication for u_censor).

**This is a falsifiable claim**: a reviewer who finds a use case u' not on the list, or a bounding mechanism that's empirically inadequate, can challenge the inequality directly. v1.0 commits to extending the taxonomy as use cases emerge from operational evidence and red-team exercises.

**Verification**:
- Per-use-case calibration: B_u functions are estimated by federation policy-makers based on use-case-specific economic / threat analysis.
- Per-use-case structural verification: structural defenses (multi-attestation, witness-bounded publication, quorum thresholds) are independently testable.
- Aggregate verification: RATCHET aggregate signals + out-of-band monitoring + red-team exercises (§1.3) provide the continuous-adversarial-research evidence stream.

**What B(w) is NOT**: an unbounded adversarial sup over all conceivable attacker strategies. B(w) is bounded by the *enumerated* use-case taxonomy; novel use cases trigger a taxonomy update + recalibration. The federation does not claim B(w) is bounded against unknown attacks — it claims the *known* attack surface is enumerated, B_u is bounded per known u, and the enumeration is updated as evidence accrues.

### 9.4 Empirical bets named explicitly

The cost-asymmetry inequality rests on **three empirical bets** (originally introduced in §1.2):

- **Bet 1** *(some hardware/capital floor exists somewhere in the cost stack)*: at any given time, at least one of c_hardware, c_bond, or c_attestation × time-floor is non-trivial relative to attacker budget. Erosion of one floor raises the importance of others.

- **Bet 2** *(RATCHET's formal-mathematical guarantees hold operationally under their stated limitations)*: real federation behavior stays within RATCHET's modeled regime (non-adaptive enough, sufficiently convex feasibility region, n ≥ 100 samples per peer, low correlation per the empirical PR ≈ 6.61, distribution-matched). **Empirically supported as of 2026-05** by 100% deceptive-prior detection on Qwen 3.6 across 29 languages, 5 patches; subject to ongoing adversarial validation.

- **Bet 3** *(policy parameters can be tuned faster than adversaries adapt)*: human-loop policy adjustment (steward, post-G2 council) outpaces adversary adaptation. Plausible for slow-moving attacks; requires monitoring + response infrastructure for fast-moving exploits.

These bets are not proven. They are explicitly the federation's working hypotheses, monitored continuously, and tracked in §11 open questions. A reader who doubts a bet should know exactly where to push back.

---

## 10. Interface to RATCHET

### 10.1 Matrix as documentation (NOT a wire format)

The dimension mapping below documents which N_eff and policy dimensions each F-AV perturbs. **It is not a machine-consumable contract.** RATCHET cannot programmatically consume markdown tables.

The matrix is the *human-readable* interface for RATCHET designers and threat-model maintainers. Machine-readable contract is specified separately in §10.2.

#### F-AV × dimension matrix (RATCHET signal mapping)

Dimensions per *Constrained Reasoning Chains* + PoB §2.4:
- **k_eff** — empirically-measured effective dimensionality (target ≥ 7.1 for weight grant; mature agents ≥ 8.5).
- **PC1..PC11** — principal components of the 16-dim feature vector (intrinsic rank 11).
- **σ-decay** — Sustainability Integral.
- **Polyglot torque** — semantic divergence across 29 languages.
- **GraphDeg / GraphCC** — peer-attestation graph degree / clustering coefficient.
- **HwDiv** — hardware-attestation diversity.
- **GeoDiv** — geographic distribution.
- **TempDiv** — temporal-action distribution.
- **BondDiv** — bond-issuer diversity.
- **BondFund** — bond-funding-source entropy.
- **Activity-density** — trace volume per time unit.
- **Stylometric** — author-fingerprint properties of long traces.
- **Embedding-cluster** — semantic proximity in embedding space.
- **Substrate-confidence** — per §7.3 RATCHET output validity level.

Each F-AV maps to a subset of dimensions. Detailed mappings are in each F-AV's "RATCHET signal" section (§6). The summary view:

| Class | F-AVs | Primary RATCHET dimensions perturbed |
|-------|-------|---------------------------------------|
| 6.1 Identity-creation | F-AV-1, 2, 3 | HwDiv, GraphDeg, GraphCC, BondDiv, BondFund, k_eff (cross-cluster) |
| 6.2 Benefit-faking | F-AV-4, 5, 6, 7, TIMESHIFT | k_eff, PC1..PC11, σ-decay, Polyglot torque, Activity-density, Embedding-cluster |
| 6.3 Trust-graph capture | F-AV-8, 9, 10, 15, BRIBE | (mostly out-of-band; flagged via write-pattern anomalies, attestation-out-of-distribution) |
| 6.4 Composition leaks | F-AV-11, 12, 13, 14, BOOT | Substrate-confidence, cross-region-divergence flags, sequence-version flags |
| 6.5 Availability/coercion | F-AV-16, 17, ECLIPSE | Federation-wide weight redistribution, evidence-availability divergence (post-hoc only) |
| 6.6 Long-range/dormancy | F-AV-DORMANT | Activity-density, Stylometric, Embedding-cluster |
| 6.7 Meta | F-AV-MAINT, CROSS | Not detectable by RATCHET (out-of-band) |

### 10.2 What a wire-format contract would require

For RATCHET to programmatically consume this threat model, the federation needs a structured schema (YAML/JSON/protobuf) that round-trips with every F-AV in §6.

**Schema (v1.0 sketch)** — designed so all 31 F-AVs in §6 produce valid entries, including ones without quantitative cost or without RATCHET-detector mappings:

```yaml
version: "1.0"
threats:
  - id: "F-AV-1"
    name: "Multi-identity Sybil"
    class: 3                            # primary class tag (advisory; per §4.6)
    cross_class_implications: []        # additional classes if relevant
    targets:
      dimensions:                       # subset of dimension vocabulary (§10.1)
        - {name: "HwDiv", direction: "down"}
        - {name: "GraphDeg", direction: "up_in_clique"}
        - {name: "GraphCC", direction: "up"}
        - {name: "k_eff", direction: "down"}
      pob_cost_terms: ["c_hardware", "c_compute"]
    substrate_assumptions: ["A0", "A1", "A2", "A3", "A5", "A6", "A7"]
    cost_row:
      kind: "quantitative_range"        # or "qualitative" or "out_of_band" or "out_of_scope"
      attacker_cost_per_identity_usd:   # only present if kind="quantitative_range"
        min: 0.10
        max: 10.0
      cost_function: "c_hardware(w) + c_compute(w, t)"
      scaling: "linear_or_sublinear_with_cloud_batching"
    ratchet_detector:
      kind: "implemented"               # or "research_grade" or "out_of_band" or "none"
      id: "ratchet.detection.cluster_pca_anomaly"   # only present if kind="implemented"
    status: "partially_mitigated"       # one of: open, spec_only, partially_mitigated, mitigated, out_of_scope
    implementation_status:              # NEW in v1.0: explicit Spec/Impl/Deployed
      spec: true
      impl: true
      deployed: true
    known_weaknesses:
      - "cloud_vtpm_floor_erosion"
      - "cluster_detection_assumes_co_observation"
    references:
      - section: "§6.1"
      - related_F_AV: ["F-AV-DORMANT", "F-AV-7"]
      - per_repo_TM: ["CIRISVerify §3.5"]
```

**Round-trip with all F-AVs**: every F-AV in §6 must produce a valid schema entry. The schema's `cost_row.kind` enumeration accommodates F-AVs that don't have quantitative costs:
- `kind: "quantitative_range"` — F-AVs with $-bounded attacker cost (F-AV-1, F-AV-8, F-AV-15, F-AV-SRC, etc.).
- `kind: "qualitative"` — F-AVs where cost is described but not quantified (F-AV-3 "≈$0", F-AV-12 "≈$0 if well-positioned").
- `kind: "out_of_band"` — F-AVs whose cost is determined by factors outside the federation (F-AV-7 cost-asymmetry collapse depends on market trends; F-AV-9 G2 succession cost is one-time-targeted).
- `kind: "out_of_scope"` — F-AVs that are stubs (F-AV-CROSS).

Similarly `ratchet_detector.kind` accommodates F-AVs RATCHET cannot detect:
- `kind: "implemented"` — RATCHET has a specific detector.
- `kind: "research_grade"` — detector approach is documented but not implemented.
- `kind: "out_of_band"` — defense relies on out-of-band monitoring (F-AV-7, F-AV-9, F-AV-MAINT).
- `kind: "none"` — RATCHET cannot detect (F-AV-MAINT, F-AV-CROSS).

**Round-trip validation requirements**:
1. Every F-AV in §6 must have a corresponding schema entry.
2. Every schema entry must have a corresponding §6 F-AV.
3. Every dimension referenced in `targets.dimensions` must appear in the dimension vocabulary (§10.1).
4. Every substrate assumption referenced in `substrate_assumptions` must appear in §7.1.
5. Every implemented `ratchet_detector.id` must exist in `CIRISAgent/ratchet/detectors/`.

**Where the schema lives**: `CIRISAgent/ratchet/threat_contract.yaml` (TBD), with version aligned to this document's version. Build-time validation: a CI step in CIRISAgent verifies the round-trip against this document.

**Status**: schema specified in v1.0; **Impl pending**. Implementing requires (a) schema file + JSON Schema validator, (b) detector-ID enumeration in RATCHET, (c) CI build-time validation that round-trip succeeds, (d) version-pinning between this document and the schema.

### 10.3 Worked example: evaluating a proposed measure

To evaluate a proposed anti-Sybil measure M, a RATCHET designer walks the matrix:

**Example measure**: "Require ≥3 distinct bond issuers per registered-tier identity, with cross-issuer fraud-signal sharing."

1. **Targeted F-AVs**: F-AV-2 (perturbs BondDiv, BondFund), indirectly F-AV-15 (raises Portal-compromise cost by N issuers).
2. **Cost-row update** (§9.1): F-AV-2 attacker cost rises from "bond face value × N" to "bond face value × N × 3 issuers + fraud-signal cross-detection cost." F-AV-15 attacker cost rises from "$10K-$10M targeted" to "$10K-$10M × 3 issuers."
3. **Cross-effects**: forces attacker to spread bond purchases across issuers (BondDiv ↑ for attackers); legitimate users may also need to spread (BondDiv ↑ uniformly). RATCHET signal for BondDiv ↓ becomes stronger because *anomalous* BondDiv ↓ now requires defeating the per-issuer requirement.
4. **§8 composition check**: does M preserve fail-secure? Yes — if a bond issuer goes offline, identity falls back to behavioral-only weight (restrictive default). ✓ Does M preserve recursive scrub-signing? Yes — bond rows still chain to bootstrap. ✓
5. **Empirical-bet check**: does M strengthen Bet 1 (cost-floor)? Yes — raises c_bond × per-issuer overhead. Net positive on Bet 1.
6. **Decision**: approve with implementation tracking; document update in §6.1 F-AV-2 mitigation surface.

A measure that fails any composition check (§8) is rejected regardless of its anti-Sybil benefit. The federation does not adopt anti-Sybil mechanisms that violate fail-secure or other invariants.

---

## 11. Open questions and empirical bets

### 11.1 Hardware floor in cloud-vTPM era (Bet 1)

Cloud confidential computing (AWS Nitro, GCP Confidential Space, Azure attestation) erodes the per-identity hardware-cost floor. **Open question**: where in the cost stack does *some* floor remain robust under cloud-vTPM economics? Operational answer (current): k_eff threshold + reasoning compute + bond capital + reputation time. None individually robust against well-funded attacker; aggregate provides defense in depth.

### 11.2 Adaptive attackers and RATCHET L-02 (Bet 2 sub-question)

RATCHET's bounds are non-adaptive. **Open question**: how rapidly can an adaptive attacker that learns from RATCHET output erode the empirical 100% Qwen 3.6 detection? Research direction: red-team continuous validation, adversarial measurement reserves, polyglot expansion beyond 29 languages.

### 11.3 Bond market liquidity and tradability

If bonds are tradeable, cost-of-capital × holding-period replaces face value. **Open question**: what bond-market policy (non-transferable, transfer-with-re-KYC, transfer-with-weight-rebaseline) preserves the cost-asymmetry? Owned by Portal product layer; not yet specified.

### 11.4 G2 steward succession protocol (unfilled)

Bridge is a single human; no implemented multi-party protocol. **Research direction**: m-of-n threshold signing for steward authority + explicit cutover transaction protocol + external witness publication. F-AV-9 is open by absence.

### 11.5 N1 + N2 Reticulum integration (unfilled)

PoB §3.2 specifies Reticulum; not yet implemented. F-AVs blocked: F-AV-16, F-AV-17, F-AV-ECLIPSE, eclipse-resistance generally. **Research direction**: ship `ciris-reticulum` crate, integrate with persist federation directory, phase out DNS-dependent paths.

### 11.6 R1 revocation timeliness specification (unfilled)

No formal propagation-timeliness contract. **Open question**: what target propagation bound T is achievable + auditable? Proposed: T ≤ 60s normal operation, ≤ 300s under partial failure. Specification + measurement infrastructure pending.

### 11.7 Q1 quorum/CAP model for S2 (unfilled)

No explicit CAP-class statement. **Open question**: linearizable, RYW, or bounded-staleness? Proposed: bounded-staleness with τ-bound, multi-region quorum reads for high-stakes operations. Specification pending.

### 11.8 Threat-model artifact integrity (unfilled)

This document is markdown without signing or 2-person-rule. **Open question**: how do we lift it to S2-row integrity standards? v2 specifies the requirement; implementation pending. F-AV-MAINT is live exposure.

### 11.9 Cross-federation peering (out of scope; stubbed)

When federations interact, F-AVs compose nontrivially. **Open question**: bond-portability, identity-portability, consensus across federations. Out of scope; deferred to v3 federation peering FSD. F-AV-CROSS is the named stub.

### 11.10 RATCHET adversarial training (research-grade)

F-AV-4 / F-AV-5 / F-AV-7 depend on RATCHET measurement evolving with attacker techniques. **Strategic question**: is the polyglot + NP-hardness defense robust against future model families (post-Qwen 3.6 era)? Continuous adversarial validation is the operational answer; theoretical bounds depend on RATCHET L-01..L-08.

---

## 12. Update cadence + meta-integrity

**Owner**: CIRISVerify maintainers (federation-substrate stewards), in coordination with CIRISPersist, CIRISAgent (RATCHET module), CIRISRegistry, CIRISPortal maintainers. Currently a single human (project lead). G2 succession protocol (§11.4) will distribute this responsibility.

**Update triggers**:
- New F-AV identified (open issue → review → integrate).
- Substrate primitive added/removed (recompute primitive set; update §2 / §3).
- RATCHET detector added/modified (update §10 mapping).
- Per-repo threat model changes that affect cross-references (update §5).
- Significant federation-protocol release (review entire document).
- **Annual external adversarial review** (per §1.3) — required for major version cuts.

**Review cadence**: quarterly for minor updates; annual for major version cuts (v3, v4, ...) preceded by adversarial review.

**Cross-repo synchronization**: changes affecting per-repo threat models trigger cross-repo issues per established CIRIS coordination pattern.

**Threat-model integrity** (per F-AV-MAINT / §11.8 specification):

1. Each published version of this document is signed by the steward (or G2 council post-cutover) using the C1 key with content `{doc_hash: <sha256>, version: <n>, signed_at: <T>}`.
2. The signature is published as an S2 policy row + a transparency-log entry (per §8.1 external-anchoring pattern).
3. Federation release tarballs include the document hash; peers can verify the doc they're consulting matches the released version.
4. Edits require PR review by ≥1 threat-model peer reviewer before merge.
5. Annual external adversarial review (v1 review produced v2; v2 review will produce v3).

**Status**: signing requirement specified in v2; implementation pending.

---

## 13. References (verified)

### Internal — CIRIS repos

- `CIRISVerify/FSD/FSD-001_CIRISVERIFY_PROTOCOL.md` — substrate primitive specification
- `CIRISVerify/docs/THREAT_MODEL.md` — substrate threats for C1, C2, C3, S3 (split): §3.1 License Fraud (AV-1..AV-6), §3.2 Federation Identity (AV-7, AV-8), §3.3 Build Provenance (AV-9), §3.4 Supply Chain (AV-11, AV-12), §3.5 Hardware Trust Anchor (AV-13), §3.6 Operational, §3.7 Multi-Instance Cohabitation
- `CIRISVerify/docs/BUILD_MANIFEST.md` — C3 wire format and consumption paths
- `CIRISVerify/docs/HOW_IT_WORKS.md` — substrate primitive overview, persist-as-interface doctrine
- `CIRISPersist/docs/THREAT_MODEL.md` — substrate threats for S1, S2, S3 (split): §3.1 Forgery (AV-1..AV-4), §3.3 Corruption (AV-9..AV-13 incl AV-10 audit anchor, AV-11 pubkey poisoning), §3.5 Provenance (AV-24..AV-26)
- `CIRISPersist/docs/FEDERATION_DIRECTORY.md` — S2 schema (federation_keys / federation_attestations / federation_revocations)
- `CIRISPersist/docs/COHABITATION.md` — persist-as-interface composition
- `CIRISPersist/docs/BUILD_SIGNING.md` — recursive scrub-signing for build artifacts
- `CIRISRegistry/docs/THREAT_MODEL.md` — substrate threats for S2 (registry side): §3.1 Forgery (AV-1..AV-6), §3.2 DoS (AV-7..AV-11), §3.3 Auth Bypass (AV-12..AV-15), §3.4 Corruption (AV-16..AV-19), §3.6 Provenance (AV-25..AV-28 incl AV-26 closed)
- `CIRISRegistry/docs/TRUST_CONTRACT.md` — three consumption paths for C3 manifests
- `CIRISRegistry/docs/FEDERATION_CLIENT.md` — registry as cache + policy layer
- `CIRISAgent/FSD/PROOF_OF_BENEFIT_FEDERATION.md` — PoB framework, N_eff measurement, σ-decay, cost-asymmetry
- `RATCHET/README.md`, `RATCHET/FSD.md` — RATCHET architecture (4 engines, 8400 LOC Python)
- `RATCHET/KNOWN_LIMITATIONS.md` — L-01..L-08 limitations cited throughout this doc
- `RATCHET/ratchet-paper/Constrained_Reasoning_Chains.pdf` — empirical validation (n=6,465 traces, N_eff threshold ≈ 7.1, intrinsic rank 11)
- `RATCHET/immediate_release/main.pdf` — CIRISAgent paper with RATCHET validation results

### External

- FIPS 204 (ML-DSA) — post-quantum signature standard
- NIST hybrid-signature draft — bound-signature specification
- Bindel-Herath-Stebila — formal analysis of hybrid signatures
- SLSA L3 — supply chain levels for software artifacts
- Sigstore + Rekor — transparency-log model (informs S3 + external bootstrap anchor)
- Certificate Transparency (CT) — transparency-log alternative for bootstrap anchoring
- Reticulum specification — N1/N2 substrate (per PoB §3.2)
- Accord Book IX Ch. 5 — Sustainability Integral / σ-decay framework
- Z3 SMT solver — used in RATCHET for CONSISTENT-LIE NP-hardness reduction
- Exponential Time Hypothesis (ETH) — Impagliazzo-Paturi, complexity-theoretic foundation for RATCHET's complexity-asymmetry argument

---

## Appendix A: v1 → v2 → v1.0 (published) changelog

This appendix documents the lineage of internal drafts that produced the published v1.0:

- **Internal v1** (May 2026): first complete draft.
- **Internal v2** (May 2026): full rewrite responding to 4 specialist adversarial reviewers (cryptography, distributed systems, mechanism design, threat-modeling methodology) — 25 findings, 4 severity tiers.
- **Published v1.0** (May 2026): rewrite responding to a second adversarial review pass on internal v2 from the same 4 reviewer roles. Folds in their Tier 1 and most Tier 2 findings on v2; remaining items tracked in §11 open questions.

Total adversarial-review pressure applied before publication: ~50 findings across 8 reviewer-passes (4 reviewers × 2 versions). The federation deliberately chose to subject the threat model to multiple internal review rounds before any public exposure.

### A.1 Primitive set changes

| Change | Rationale |
|--------|-----------|
| **Added C0 RNG** | RNG quality is foundational across C1/C2/C4. Treating it as part of C1 hides fan-out. (Cryptographer reviewer: "Per-FIPS-140-3 reasoning, RNG is its own validatable component.") |
| **Added C4 hybrid KEX + KDF** | v1 had no key exchange primitive. Federation peer-to-peer comms over HTTPS provide forward secrecy against current adversaries but are harvest-now-decrypt-later vulnerable. P7 forward secrecy added. |
| **Added R1 revocation propagation** | Storage and propagation are distinct security properties. v1 hid this fan-out by treating revocation as "a row in S2." Reviewer: "F-AV-11/12/13/16 all collapse without R1 timeliness." |
| **Added Q1 quorum/CAP** | v1 had targets ("≤60s lag, 2-of-N") without a model. CAP-class specification needed for fail-secure under partial failure. Distsys reviewer: "Consistency model is a property, not a protocol." |
| Total: 8 primitives → 12 primitives | |

### A.2 Anti-Sybil reframing

| v1 claim | v2 claim |
|----------|----------|
| Anti-Sybil is an emergent invariant of primitive composition | Anti-Sybil is a policy-tunable posture; the federation continuously adjusts parameters |
| Cost-asymmetry inequality stated as totals | Marginal formulation: dC/dw > dB/dw |
| Hardware floor: $50–500 per identity | Cloud vTPM erodes this; hardware floor is no longer reliable; defense-in-depth across multiple cost dimensions |
| RATCHET-vs-LLM as 1:1 measurement race (initial v2 framing) | **Corrected**: RATCHET is pure math (statistical detection + Monte Carlo + SAT-based complexity + PBFT). Asymmetry is complexity-theoretic (truth=O(1), coherent lying NP-hard under ETH), not measurement-arms-race. Empirically supported by *Constrained Reasoning Chains* (n=6,465) and 100% Qwen 3.6 detection across 29 languages |

### A.3 New F-AVs

| F-AV | Class | Reason added |
|------|-------|--------------|
| F-AV-TIMESHIFT | 3 | Reviewer: "σ-decay is replay-vulnerable via paraphrase. C1 only proves the key signed it, not that the key reasoned it." |
| F-AV-BRIBE | 3 / 5 | Reviewer: "Bribed legitimate participants pass RATCHET because they ARE legitimate." |
| F-AV-ECLIPSE | 4 | Reviewer: "F-AV-17 censors evidence of a target identity; eclipse curates a consuming peer's S2 read view — distinct attack." |
| F-AV-DORMANT | 3 | Reviewer: "Long-range/dormant Sybils age cheaply; v1 implicitly assumed time-as-cost which fails for dormant accounts." |
| F-AV-BOOT | 2 / 5 | Reviewer: "Recursive scrub-signing terminates at bootstrap; what audits the bootstrap? v1 had a singularity." |
| F-AV-MAINT | 5 | Reviewer: "v1 doesn't model the threat model itself as an attack surface." |
| F-AV-CROSS | 5 (stub) | Stubbed so cross-federation attackers aren't invisible. |

### A.4 Reframings and tightenings

| Topic | Change |
|-------|--------|
| §7.1 Substrate properties | Expanded from 6 to 10 assumptions (added A0, A4, A8, A9). A1 reworded from binary "not extractable" to budget-relative "cost ≥ $X". |
| §7.2 Per-assumption failure rates | New: probabilistic ROM estimates per assumption (v1 treated as binary). |
| §7.3 RATCHET output degradation | New: degradation curve as substrate assumptions weaken. |
| §8.1 Bootstrap anchoring | New: external transparency-log anchoring specification (Sigstore/Rekor or CT + witness signatures). |
| §8.2 Per-primitive isolation reconciled with F-AV-11 | Explicit: substrate-level isolation does not prevent behavior-level evidence propagation; the two are complementary, not contradictory. |
| §8.3 Fail-secure as protocol | v1 was "property"; v2 is signed decisions + S3-logged + grace-window τ_grace=60s + τ_max=300s + clock-source specification + sliding-window cumulative-degraded-time defense against grace-window-loop attack. |
| §8.4 Cost-asymmetry inequality | Rewritten with explicit functional form, units, and policy parameters. v1 was a slogan; v2 is an analyzable inequality. |
| F-AV-14 bound-signature | Precise specification: PQC signs `H(domain_sep ‖ alg_id ‖ pk_classical ‖ m ‖ σ_classical)`. Single-algorithm verification of deprecated scheme **forbidden** during migration window. |
| §5 cross-references | Verified against actual section numbers in per-repo threat models (v1 had broken citations: C2 → §3.2 was wrong, S3 → §3.6 was wrong). |
| §10 RATCHET interface | Acknowledged: matrix is documentation, not wire format. Added schema sketch for actual contract. |
| §12 Meta-integrity | New: signing requirement on this document, 2-person-rule on edits, annual external adversarial review. |

### A.5 Findings deferred to §11 / future versions

- **Specific Q1 CAP-class choice** (linearizable vs RYW vs bounded-staleness) — proposed bounded-staleness; awaiting validation by distsys review.
- **Specific R1 propagation bound** (T ≤ 60s? ≤ 30s?) — specification + measurement infrastructure pending.
- **Activity-density measurement formulas for F-AV-DORMANT** — research-grade.
- **Bond-market policy specification** (transferability, KYC tiers, secondary-market rules) — owned by Portal product layer.
- **G2 succession protocol** — entire protocol specification pending.
- **Cross-federation peering** (F-AV-CROSS) — entire FSD pending; deferred to v3.
- **Threat-model wire-format schema** (per §10.2) — implementation pending.

### A.6 What was NOT changed (and why)

- The class taxonomy (5 classes) was kept despite reviewer pointing out it's not strictly disjoint. Bookkeeping rule added (§4.6) with primary-class tagging and explicit cross-class implications. Renaming F-AV-11/12/14 to Class 1 was considered and rejected (would break stable identifiers; cross-references throughout the doc).
- F-AV stable identifiers were preserved (F-AV-1..F-AV-17 plus mnemonic-suffix new ones). Future versions may renumber for sequential cleanliness.
- The methodology of cold-derive → project → enumerate was kept; v1 reviewer accepted this with the convergence acknowledgement now in §2.4.

---

### A.7 v2 → v1.0 (published) changes

v2 was reviewed by the same 4 reviewer roles in a second adversarial pass. The pass's net judgment: "v2 is materially more honest than v1, but documents the security posture rather than fixing it." v1.0 closes the most prominent gaps the second pass surfaced. Specifically:

**A.7.1 New F-AVs added (7)** — all flagged as missing from v2's catalog:

| F-AV | Class | Reason added |
|------|-------|--------------|
| F-AV-ONBOARD | 3 | Reviewer: "Pre-reputation window where new identities are most exploitable; RATCHET cannot evaluate before n_min ≥ 100 traces (L-05)." |
| F-AV-REPUDIATE | 2 | Reviewer: "Peer denies signing they did sign; distinct from forgery; non-repudiation depends on time-of-signing + key-ownership-at-signing-time provenance." |
| F-AV-FRONTRUN | 2 | Reviewer: "Read-side covered by F-AV-12; write-side ordering attacks absent." |
| F-AV-ROLLBACK | 2 | Reviewer: "Coordinated minority on Q1 partition attempts to roll back accepted state; touched by F-AV-12 but not as coordinated attack." |
| F-AV-RATCHET-DOS | 4 | Reviewer: "F-AV-16 DoSes substrate; RATCHET-as-evaluator treated as infinitely-available compute. ~8400 LOC Python is real infrastructure." |
| F-AV-PRIVACY | 4/5 | Reviewer: "Federation directory readability is assumed adversary-public without modeling target-selection asymmetry that enables F-AV-BRIBE / F-AV-ECLIPSE." |
| F-AV-SRC | 5 | Reviewer: "C3 attests 'this IS the official build' but no F-AV for 'the official build is hostile at source.' AGPL-3.0 source review is the only listed defense." |

**A.7.2 Major specifications tightened**:

| Topic | Change |
|-------|--------|
| §3.1 Per-primitive coverage | Replaced ✓/⚠/✗ with explicit Spec/Impl/Deployed tags. Implementation Status legend added at top of document. |
| §7.2 Failure-rate ROM | **Numerology removed**. Replaced 10⁻⁵ to 10⁻¹ numerical ranges with ordinal tier categorization (TIER-LOW, MED, HIGH, CRITICAL) + named referents (specific CVE classes, real failure modes). Owner and update-trigger named. Joint-failure model added. |
| §8.1 Bootstrap external anchoring | Witness diversity requirements specified: cloud-vendor diversity, jurisdictional diversity, organizational diversity, software-stack diversity. Disagreement resolution rule (≥ ⌈2N/3⌉ Byzantine threshold). Install-time chicken/egg solved via baked-in pin + periodic online verification. Sigstore/CT root-rotation handling specified. |
| §8.3 Fail-secure protocol | **S3-when-S3-degraded** addressed via in-memory ring buffer with cryptographically-signed entries flushed on recovery. **Monotonic-clock-to-TPM-attested-time binding** added so monotonic intervals are externally verifiable. **Sliding window** specified with explicit values (W=600s, D_max=180s) and eviction policy. **Post-hoc verifiability** explicitly distinguished from post-hoc justifiability. |
| §8.4 Empirical evidence framing | 100% Qwen 3.6 result reframed as one experiment in a portfolio. Caveats explicit: (a) one model family, (b) detection of known restrictive priors only, (c) non-adaptive attackers only, (d) within Accord constraint set. What the evidence DOES vs DOES NOT establish enumerated. |
| §9.3 Marginal cost-asymmetry | **B(w) = "depends on use case" replaced with explicit benefit-extraction taxonomy**. 6 use cases enumerated (u_launder, u_influence, u_extract, u_info, u_censor, u_collapse) with B_u functions and structural defenses per case. The inequality is now falsifiable: a reviewer who finds a use case not on the list can challenge directly. |
| F-AV-14 verifier obligations | Verifier obligations specified for hybrid / new-only / old-only modes (v2 only forbade old-only). Canonical-encoding rule for σ_classical (raw fixed-length, not DER). Cross-protocol attack acknowledgment (CIRIS protects against others, key-reuse forbidden by policy). Three-phase migration protocol. |
| F-AV-DORMANT | Activity-density formula specified with sliding-window definition. Density-tier policy (Active/Light/Sparse/Dormant) with weight-multiplier mapping. v2 reviewer's $4,400/5yr cost estimate corrected to realistic $200–1,000 range under spot-instance + sleep modes. |
| §10.2 Schema | Round-trip validation requirements specified. Schema accommodates all F-AV cost classes (quantitative_range / qualitative / out_of_band / out_of_scope) and all RATCHET-detector states (implemented / research_grade / out_of_band / none). All 31 F-AVs in §6 produce valid schema entries. |

**A.7.3 v2 review findings still open in v1.0**:

These were flagged by v2 reviewers but not closed in v1.0; tracked in §11:

- **R1 timeliness contract** (specific T value): proposed; not specified.
- **Q1 CAP-class choice** (linearizable vs RYW vs bounded-staleness): proposed bounded-staleness; not validated.
- **Witness-set disagreement protocol details**: high-level rule specified; specific operational protocol pending.
- **Per-trace freshness challenge protocol** (mitigation for F-AV-TIMESHIFT and F-AV-DORMANT): not specified.
- **Recursive scrub-signing scale at 1M+ rows**: chain-length scaling not addressed.
- **G2 succession protocol**: entire protocol specification pending.
- **Cross-federation peering FSD**: deferred.

These items are real. v1.0 is published with them open; the federation's continuous-research effort closes them in v1.x and v2.0.

**A.7.4 What v1.0 explicitly does NOT claim**:

- **Anti-Sybil resistance is proven**. It is a continuously-tuned policy posture under three named empirical bets (§1.2).
- **The 100% Qwen 3.6 result generalizes**. It is one data point on one model family.
- **All current attack surfaces are enumerated**. The F-AV catalog is the federation's best current map; novel attacks may surface in operational evidence and trigger v1.x updates.
- **All "Spec" items will be implemented on a fixed schedule**. Implementation is funding-bounded and priority-sorted; this document does not commit to delivery dates.
- **The federation is currently secure against state-actor adversaries**. State-actor adversaries are explicitly out of scope for "opportunistic-attacker" cost analyses; high-resource adversary modeling is research-grade.

---

*This document is iterative. v1.0 was produced from internal v1 + internal v2 + two rounds of 4-reviewer adversarial review. Future versions (v1.1, v1.2, v2.0) will be produced from operational evidence + ongoing adversarial review + implementation completion of currently-Spec items. Future readers can challenge any claim by tracing it to its first-causes derivation in §1–§2, pushing back on the empirical bets in §1.2 / §9.4, or filing a finding against the F-AV catalog in §6.*
