# CIRISEdge Threat Model

**Status:** v0.0 baseline scaffold (pre-Phase-1 implementation; spec-only as
of `3fc4ab0`). Updated each minor release once code lands.
**Audience:** federation peers integrating against `ciris-edge`, security
reviewers, downstream substrate consumers (`CIRISLens`, `CIRISAgent`,
`CIRISRegistry`).
**Companion:** [`MISSION.md`](../MISSION.md), [`FSD/CIRIS_EDGE.md`](../FSD/CIRIS_EDGE.md),
[`FSD/OPEN_QUESTIONS.md`](../FSD/OPEN_QUESTIONS.md).
**Inspired by:** [`CIRISVerify/docs/THREAT_MODEL.md`](https://github.com/CIRISAI/CIRISVerify/blob/main/docs/THREAT_MODEL.md)
(structural template), [`CIRISProxy/docs/THREAT_MODEL.md`](https://github.com/CIRISAI/CIRISProxy/blob/main/docs/THREAT_MODEL.md)
(transport-class adjacency), [`CIRISPersist/docs/THREAT_MODEL.md`](https://github.com/CIRISAI/CIRISPersist/blob/main/docs/THREAT_MODEL.md)
(verify-and-persist substrate).
**Federation primitives addressed:** N1 (cryptographic addressing) and N2
(multi-medium transport) — the two primitives the federation meta threat
model (`FEDERATION_THREAT_MODEL.md` §5) lists as unfilled. CIRISEdge is
the substrate that fills both.

---

## 1. Scope

### What CIRISEdge Protects

CIRISEdge is the federation transport substrate. It carries cryptographic
envelopes between sovereign peers across whatever network media exist
(TCP, LoRa, packet radio, serial, I²P, HTTP fallback). Per MISSION.md §2
it protects:

- **Wire-edge integrity**: no application code touches a byte that hasn't
  been verified against `persist.engine.lookup_public_key()`. Verify is a
  precondition for handler dispatch, not an opt-in.
- **Cryptographic addressing**: per PoB §3.2, the Reticulum destination
  is `sha256(public_key)[..16]` — the address IS the identity. No DNS,
  no certificate authority, no out-of-band binding. Forging an address
  requires forging the underlying public key.
- **FFI boundary discipline**: the steward seed lives in `ciris-persist`'s
  keyring; edge holds the `Engine` handle and calls into persist for
  every sign / verify operation. The seed bytes never enter edge's
  process memory. Same discipline persist established in v0.2.2 for
  `Engine.steward_sign()` (CIRISPersist#10).
- **Verify-via-persist contract**: edge does not maintain its own
  public-key cache or canonicalization implementation. Persist's
  `lookup_public_key()` and `canonicalize_envelope()` are the single
  sources of truth. Re-implementing either in edge is the
  CIRISPersist#7 trap.
- **Replay protection within a bounded window**: edge maintains a
  sliding window (5 min in Phase 1 per `OPEN_QUESTIONS.md` OQ-08) of
  recently-seen `(signing_key_id, nonce)` pairs. Beyond the window,
  persist's dedup tuple `(agent_id_hash, trace_id, thought_id,
  event_type, attempt_index, ts)` (CIRISPersist AV-9 closure) catches
  application-layer replay.
- **Typed handler dispatch**: no `&[u8]` or `serde_json::Value` past
  the parse boundary. Every message type has a Rust struct with
  `serde::Deserialize`; handlers receive parsed structs, not raw bytes.
- **Wire-event auditability**: every message in or out emits exactly
  one structured log line with `signing_key_id`, `body_sha256_prefix`,
  `verify_result`, `handler_duration`. Forensic completeness is a test
  category, not an afterthought.
- **Multi-medium reach**: the same wire envelope round-trips
  byte-equivalent across all configured transports. M-1 demands the
  primitive runs on the network media that exist on the planet, not
  just the ones convenient for hyperscaler deployments.

### What CIRISEdge Does NOT Protect

- **Compromised steward keys** (AV-2 class — persist's threat model
  owns; the verifier cannot distinguish stolen-key from legitimate).
  Detection is statistical via N_eff drift over time (PoB §2.4 + §5.6,
  RATCHET).
- **Federation directory poisoning**. The `accord_public_keys` /
  `federation_keys` directory is persist's responsibility. Edge calls
  into it; if it's owned, edge's verify is meaningless.
- **Build-time supply chain compromise**. CIRISVerify owns build
  attestation; edge's release tarball is signed via `ciris-build-sign`
  hybrid (Ed25519 + ML-DSA-65) and registered with CIRISRegistry, but
  the upstream attestation infrastructure is CIRISVerify's threat model.
- **Application-layer authorization**. Edge dispatches verified
  messages to typed handlers; whether the host application *should*
  process a given message under the application's policy is host-code
  responsibility.
- **TLS termination on HTTP fallback**. The HTTP fallback transport
  (Phase 1, gated on Reticulum non-availability) uses TLS provided by
  the deployment-edge proxy (nginx, ALB). Misconfigured TLS is a
  deployment concern, not edge's.
- **Operator UI surfaces**. Grafana, OAuth, admin console — explicitly
  out of scope per FSD §2. Each peer keeps its own operator-UI HTTP
  stack.
- **Quantum compromise of Ed25519**. Phase 1 ships Ed25519-only verify;
  the `signature_pqc` field is reserved for ML-DSA-65 hybrid (post
  persist v0.4.0 per `OPEN_QUESTIONS.md` OQ-11). Phase 1 posture is
  accept-not-verify on the PQC field; full hybrid verify lands in
  Phase 2.
- **Reticulum-rs / Leviculum implementation bugs**. The transport-impl
  threat model is the upstream crate's; edge wraps the trait.
  Cross-implementation byte-equivalence is a regression test category
  (MISSION.md §5).

---

## 2. Adversary Model

### Adversary Capabilities

The adversary is assumed to have:

- **Full source-code access** (AGPL-3.0, public).
- **Ability to mint arbitrary Ed25519 keypairs** and sign bytes.
- **Network access to all configured transports**: arbitrary bytes to
  the HTTP endpoint, Reticulum link establishment with arbitrary
  identities, simulated LoRa / serial / I²P link establishment.
- **Ability to run their own peers** on the federation and request
  federation_keys registration.
- **Replay capability**: capture any in-transit message and re-send.
- **Active MITM** on HTTP fallback if TLS is misconfigured at the
  deployment edge.
- **Side-channel observation**: response timing, transport-level error
  codes, structured-log output if exposed.
- **Ability to read public CI artifacts**: every published Cargo
  release, the dep tree, the deny.toml, the Dockerfile.
- **Compute resources sufficient for classical cryptography** but not
  for breaking Ed25519 within polynomial time.
- **Ability to submit malformed canonical bytes** at any wire entry
  point.
- **Slow-medium privilege**: on LoRa / serial / I²P, transport-level
  jitter and asymmetric latency give the adversary additional timing
  observation surface compared to TCP.

### Adversary Limitations

The adversary is assumed NOT to have:

- **The ability to break Ed25519** within polynomial time on classical
  hardware. (PoB §6 acknowledges quantum risk; `signature_pqc` field
  reserved.)
- **Compromised the federation directory** (`federation_keys` /
  `accord_public_keys` in persist). Edge's verify is gated on this
  directory; if it's owned, the threat model breaks at persist's layer.
- **Compromised persist's keyring** (the steward seed). Same FFI
  boundary discipline persist established in v0.2.2.
- **Compromised the deployment hardware** running edge (process memory
  inspection, debugger attachment, coredump capture).
- **Quantum compute** capable of breaking Ed25519 today (tracked in §9
  Residual Risks; PoB §6 hybrid-PQC is Phase 2+).
- **Broken sha256** to forge cryptographic addresses.
- **Physical access to LoRa / serial hardware** at the edge deployment
  (transport-physical-layer attacks are out of scope for this crate).

---

## 3. Trust Boundaries

```
┌─────────────────────────────────────────────────────────────────────┐
│ Adversary-controlled wire (TCP / LoRa / packet radio / serial /     │
│ I²P / HTTP fallback)                                                │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ raw bytes
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│ ciris-edge process (host: lens / agent / registry binary)           │
│                                                                     │
│   ┌──────────────────────────────────────────────────────┐          │
│   │ transport/                                            │          │
│   │   Trust boundary 1: untrusted bytes enter            │          │
│   │   Reticulum link state OR HTTPS request body         │          │
│   └─────────────────────────┬────────────────────────────┘          │
│                             ▼                                        │
│   ┌──────────────────────────────────────────────────────┐          │
│   │ verify/                                               │          │
│   │   Trust boundary 2: typed envelope deserialize       │          │
│   │   ┌──────────────────────────────────────────┐       │          │
│   │   │ FFI boundary: edge ↔ ciris-persist       │       │          │
│   │   │ - lookup_public_key(signing_key_id)      │       │          │
│   │   │ - canonicalize_envelope()                │       │          │
│   │   │ - steward_sign() for outbound            │       │          │
│   │   │ Seed bytes NEVER cross this boundary     │       │          │
│   │   └──────────────────────────────────────────┘       │          │
│   │   Ed25519 verify_strict(canonical, sig, pubkey)      │          │
│   └─────────────────────────┬────────────────────────────┘          │
│                             ▼                                        │
│   ┌──────────────────────────────────────────────────────┐          │
│   │ handler/                                              │          │
│   │   Trust boundary 3: typed dispatch                   │          │
│   │   No unverified bytes past this line — invariant    │          │
│   │   asserted by verify_enforcement test category       │          │
│   └─────────────────────────┬────────────────────────────┘          │
└─────────────────────────────┼────────────────────────────────────────┘
                              ▼
                     Host application code
                     (peer-specific reasoning, persistence, scoring)
```

**Explicit non-boundary**: edge's process and persist's `Engine` share
heap. Persist's `Engine` is constructed in the host process and passed
to edge by reference. The seed-bytes-don't-cross-FFI invariant is
maintained by persist's API surface (the seed is held in OS-keyring,
never returned across the boundary — see CIRISPersist v0.1.3+ AV-25
closure). Edge tests assert this invariant by heap-scan property tests
(MISSION.md §4 "Identity boundary").

---

## 4. Attack Vectors

Twenty-four vectors organized by adversary goal. Each lists the attack,
the primary mitigation, secondary mitigation, and residual risk.

### 4.1 Identity / Forgery — adversary wants their bytes counted

#### AV-1: Forged message from attacker-minted key

**Attack**: Attacker generates a fresh Ed25519 keypair, signs a synthetic
`EdgeEnvelope`, sends to a peer's edge listener via any transport.

**Mitigation**: `verify/` calls `persist.engine.lookup_public_key(
signing_key_id)` before signature verification (FSD §3.3 step 4).
Unknown `signing_key_id` → typed `unknown_key` reject → handler is
never invoked. Same gate as `CIRISPersist/THREAT_MODEL.md §3.1 AV-1`.

**Secondary**: persist's federation_keys directory is the load-bearing
admission policy. Edge does not implement its own admission gate —
that's intentional, single-source-of-truth.

**Residual**: an attacker who registers a key id (via the federation's
`accord/agents/register` flow) earns the same gate as any peer. PoB
§2.1 cost-asymmetry is the federation-level mitigation.

#### AV-2: Forged message from compromised legitimate key

**Attack**: Attacker exfiltrates a peer's signing seed (out-of-band
secrets-manager compromise), signs a malicious `EdgeEnvelope` under
that peer's identity.

**Mitigation**: **Out of CIRISEdge's protection scope.** The verifier
cannot distinguish stolen-key from legitimate. Same residual as
CIRISPersist AV-2 — closure is statistical via N_eff drift (PoB §2.4)
and architectural via Phase 2 peer-replicate audit-chain validation.

**Residual**: undetectable at edge ingest until peer-replicate lands.
The peer's own CIRISVerify hardware-backed key storage is the
upstream mitigation; edge holds no keys to compromise locally
(MISSION.md §3 anti-pattern 3).

#### AV-3: Replay within edge's sliding window

**Attack**: Network MITM (or re-submission attack at the API) captures
a valid signed envelope and replays it within the 5-minute window.

**Mitigation**: edge maintains a bounded LRU of `(signing_key_id,
nonce)` pairs seen in the last `replay_window_secs` (5 min default per
OQ-08, configurable Phase 2). Replayed within window → typed
`replay_detected` reject → handler is never invoked.

**Secondary**: TLS at the deployment edge (HTTP fallback only)
prevents in-flight capture; not edge's responsibility. Reticulum's
native link encryption covers the canonical transport.

**Residual**: the LRU is process-local. A replay against a *second*
edge instance of the same peer (multi-replica deployment) lands once
per replica until persist's AV-9 dedup catches it at the application
layer. Per OQ-06 (multi-worker concurrency), multi-instance replay
detection requires either a shared replay-window store (Redis-class)
or accepting per-instance redundancy with persist as the authoritative
dedup. Phase 1 default: per-instance + persist as authoritative.

#### AV-4: Replay outside edge's sliding window

**Attack**: Adversary captures a valid signed envelope, waits >5
minutes, replays.

**Mitigation**: edge's window has expired the entry; the replay passes
edge's verify. **Persist's AV-9 dedup tuple catches it at the
application layer** — the message produces zero-row insert because
`(agent_id_hash, trace_id, thought_id, event_type, attempt_index, ts)`
already exists. Edge's responsibility is bounded; persist's is the
authoritative dedup.

**Residual**: a replay against a *different* peer (federation replication)
lands once by design — that's what trace replication is supposed to do
(PoB §5.1). Per-peer dedup is each peer's local guarantee.

#### AV-5: Canonicalization mismatch

**Attack**: Adversary exploits a byte-difference between what the
sender canonicalizes and what edge canonicalizes for verify.

**Mitigation**: edge does NOT implement canonicalization. Every verify
calls `persist.engine.canonicalize_envelope(envelope)` — same code
path the sender used to compute the bytes it signed. Single source of
truth (CIRISPersist#7 closure; same AV closure pattern as
CIRISPersist/THREAT_MODEL.md §3.1 AV-4).

**Secondary**: Ed25519 collision and preimage resistance bound the
"produce different bytes that verify" branch to 2^128 work — practically
infeasible.

**Residual**: float-formatting drift across host platforms (CIRISPersist
AV-4 residual). Edge inherits the same exposure and the same closure
path — a future change to persist's canonicalizer rolls out via persist
version pin in edge's `Cargo.toml`.

#### AV-6: Reticulum destination spoofing

**Attack**: Adversary attempts to claim a Reticulum destination
matching an honest peer's `sha256(public_key)[..16]`.

**Mitigation**: **Structurally impossible without breaking either
sha256 or Ed25519.** The Reticulum destination is derived from the
public key bytes; an adversary cannot produce a destination matching
honest peer X's destination without holding X's keypair. PoB §3.2
"addressing IS identity" closure.

**Residual**: 64-bit collision-resistance on the truncated destination
(CIRISPersist AV-9 residual — same trade-off, same federation-scale
acceptance). 2^64 grinding cost for targeted DOS via address collision
is the residual; 2^32 birthday for any collision pair across federation
populations up to ~4B peers.

### 4.2 Authentication / Authorization — adversary wants verified bytes to reach the wrong handler

#### AV-7: Schema-version downgrade

**Attack**: Adversary sends an envelope with a known-vulnerable
`edge_schema_version` to exploit a bug fixed in a later version.

**Mitigation**: `SUPPORTED_VERSIONS` is a strict allowlist, mirrored
from CIRISPersist v0.1.2 AV-12 closure. Out-of-set versions hit typed
`unsupported_schema_version` reject. There is no "best-effort" or
"downgrade-and-try" branch.

**Residual**: when two versions are simultaneously in the allowlist
(rolling-deploy window), per-version payload gates must be
independently typed-deserialize-strict. Track at every spec bump.

#### AV-8: Misrouted message acceptance

**Attack**: Adversary sends a valid signed envelope to peer X with
`destination_key_id` set to peer Y. The sender's signature verifies;
the envelope is "authentic" but addressed to a different peer.

**Mitigation**: edge checks `envelope.destination_key_id ==
self.steward_key_id` BEFORE deserializing the body. Mismatch →
typed `misrouted` reject; the body is never parsed.

**Secondary**: at scale, misrouted messages indicate either an
operational bug at the sender or a deliberate confusion attack;
metrics (`messages_in_misrouted` counter) surface the rate.

**Residual**: a peer running multiple steward identities (multi-tenant
deployment) needs to validate against all of its own identities; this
is the host's responsibility via `Engine.is_local_identity()`.

#### AV-9: Handler dispatch on unverified bytes

**Attack**: Adversary exploits a code path where `dispatch()` is
called before the verify pipeline completes (race condition,
exception-handling flaw, future refactor that shortcuts verify).

**Mitigation**: structural — the verify pipeline is a
linear seven-step path (FSD §3.3); `dispatch()` is private to the
verify module's success path. The `handler/` module's public API takes
already-typed-and-verified `(msg, ctx)`; there is no surface for an
unverified path.

**Secondary**: `verify_enforcement` test category (MISSION.md §4)
asserts at the property-test level that no handler is invoked for any
message that fails any of the seven verify steps. Fuzzing covers the
exception-handling boundary.

**Residual**: a future contributor adds a "fast path" that bypasses
verify under some condition. PR review + the verify_enforcement test
category catch this; the test passes only if zero unverified
invocations occur over N>=1M fuzz runs.

### 4.3 Denial of Service — adversary wants edge unable to receive evidence

#### AV-10: Wire-flooding (transport-level)

**Attack**: Adversary floods a transport endpoint (HTTP POST flood,
Reticulum link establishment flood, simulated LoRa frame flood) to
saturate edge's accept loop.

**Mitigation in v0.1.0**: per-transport rate caps on inbound link
establishment. HTTP transport: deployment-edge rate limit (nginx /
ALB) is the primary; edge applies an in-process per-source-IP token
bucket as defense-in-depth. Reticulum: link-establishment quota per
remote identity. LoRa / serial: physical-layer bandwidth is itself a
rate cap, plus per-link quota.

**Secondary**: bounded inbound queue (`DEFAULT_QUEUE_DEPTH`, OQ-09).
On saturation: 429 + Retry-After to HTTP, equivalent typed reject for
Reticulum.

**Residual**: a sufficiently distributed attacker bypasses per-source
rate limits. Federation-level admission policy (PoB §5.6 acceptance
policy) is the upstream mitigation.

#### AV-11: Verify-path saturation

**Attack**: Adversary submits high-rate valid-but-cheap messages that
each force a `lookup_public_key` round-trip into persist, saturating
persist's queue.

**Mitigation**: `lookup_public_key` is a hot path; persist's
implementation must be cheap (in-memory + worker-local cache, falling
back to DB; CIRISPersist AV-1 mitigation). Edge applies a per-source
verify-rate cap as defense-in-depth.

**Secondary**: if persist's verify path saturates, edge's bounded
queue applies backpressure to senders (429 + Retry-After).

**Residual**: pre-Phase-1 implementation; track measured throughput
against attack-rate hypothesis. If verify-path saturation becomes
observed, persist's lookup_public_key gets a circuit-breaker; edge's
backpressure becomes the operational signal.

#### AV-12: Replay-window memory growth

**Attack**: Adversary submits high-rate distinct-nonce messages to
inflate edge's `(signing_key_id, nonce)` LRU.

**Mitigation**: LRU is bounded with explicit eviction policy. Default
size: 100K entries (Phase 1; configurable). Beyond capacity, oldest
entries evict; entries past the window expire regardless of capacity.

**Residual**: with eviction, an old entry can re-enter the window via
explicit replay if the window is shorter than time-to-eviction. OQ-08
sets the window at 5min; eviction at 100K entries means the attacker
must sustain 100K/5min = 333 messages/sec just to win the eviction
race against legitimate traffic — plus pay verify cost on every one.
Cost-asymmetric.

#### AV-13: Body-size flood

**Attack**: Adversary submits arbitrarily large message bodies to
exhaust edge's parse-buffer memory.

**Mitigation in v0.1.0**: explicit `MAX_BODY_BYTES` ceiling (default
8 MiB matching CIRISPersist AV-7 closure). HTTP transport: applied at
the axum extractor layer (`DefaultBodyLimit::max(8 * 1024 * 1024)`).
Reticulum: link-frame size limits applied by the transport crate;
`MAX_BODY_BYTES` enforced after reassembly, before parse.

**Secondary**: deployment-edge proxy on HTTP fallback caps body size
upstream of edge.

**Residual**: until the limit is enforced symmetrically across all
transports, body-size flood is a defense-in-depth gap on novel
transports. Track at every new-transport PR.

#### AV-14: Malformed canonical bytes triggering parse amplification

**Attack**: Adversary submits an envelope whose `body` is deeply
nested JSON (`[[[[...]]]]` 10000 deep) or contains a single key with a
1GB string value, exhausting `serde_json::Value` deserialization.

**Mitigation**: typed envelope deserialization rejects depth-bombs at
the `EdgeEnvelope` boundary. Per-message-type body deserialize is
strict — no `serde_json::Value` in handler signatures (MISSION.md §3
anti-pattern 1). Body is `serde_json::value::RawValue` only at the
envelope layer; canonicalization preserves bytes verbatim, parse is
type-strict at the message-type boundary.

**Secondary**: recursion-depth guard (`MAX_DATA_DEPTH=32`, mirroring
CIRISPersist AV-6 closure) on any `RawValue` body parse path.

**Residual**: an attacker with a registered, accepted public key who
submits inflated-but-syntactically-valid bodies pays the cost-asymmetry
PoB §2.1 names.

### 4.4 Confidentiality / Privacy — adversary wants content text exposed

#### AV-15: Transport plaintext leakage

**Attack**: Network-adjacent adversary reads in-transit envelope
bodies on a misconfigured transport.

**Mitigation**: Reticulum provides native link-layer encryption
(canonical transport). HTTP fallback uses TLS at the deployment-edge
proxy. Edge does NOT add a third encryption layer (FSD §11
out-of-scope).

**Residual**: HTTP fallback with no TLS at the deployment edge is a
deployment misconfiguration that exposes envelope bodies. Edge MAY
emit a startup warning if HTTP transport is configured without an
upstream TLS termination indication, but ultimately TLS posture is
the operator's job. LoRa / serial / I²P: native encryption per
Reticulum's link layer.

#### AV-16: Side-channel timing on verify

**Attack**: Adversary measures response time to distinguish "unknown
key" vs "known key + wrong signature" vs "known key + right signature
+ wrong canonical bytes" — gleaning information about the federation
directory or canonicalization state.

**Mitigation in v0.1.0**: Ed25519 `verify_strict` is constant-time
over the signature/key path. **However**:
- The `lookup_public_key` short-circuits on unknown-key (returns
  before signature math runs). Timing leaks key-membership.
- Canonicalization byte length differs per payload — observable.
- LoRa / serial mediums have intrinsic per-frame timing observability.

**Recommended for v0.2.x**: constant-response-time wrapper that sleeps
to a P99 budget on the rejection path. Not free operationally; track
as research-grade hardening.

**Residual**: a network-adjacent attacker can probably enumerate
`signing_key_id`s via timing oracle. Per CIRISPersist AV-16 residual,
the federation primitive treats `signing_key_id` as public anyway —
directory enumeration is not a high-impact leak. The slow-medium
amplification (LoRa) makes the channel noisier but no more
information-theoretic leak.

#### AV-17: Heap leak of seed bytes

**Attack**: Process memory exfiltration (debugger attach, coredump
capture, `/proc/<pid>/mem` access on Linux) reveals key-seed bytes
loaded into edge's heap.

**Mitigation**: **Edge does not load seed bytes.** All sign / verify
operations call into `persist.engine`, which holds the seed in
OS-keyring (CIRISPersist v0.1.3+ AV-25 closure). The seed bytes never
cross the FFI boundary into edge's process memory.

**Test category**: `identity_boundary` test (MISSION.md §4) — property
test that scans edge's heap during sign and asserts no seed-shaped
bytes are present.

**Residual**: a future contributor adds caching that copies seed
bytes into edge's heap. The test category catches this; the heap
scan property test passes only if no seed-shaped patterns are found
across N>=1000 sign operations.

### 4.5 Multi-medium specific — N2 primitive surface

#### AV-18: Cross-medium replay

**Attack**: Adversary captures an envelope from TCP and replays it
over LoRa (or vice versa), exploiting per-transport replay-window
isolation.

**Mitigation**: edge's replay window is keyed on `(signing_key_id,
nonce)` — transport-agnostic. A replay across mediums is the same
`(signing_key_id, nonce)` and is caught by the same window.

**Residual**: a sufficiently old replay (>5min) bypasses edge's
window per AV-4; persist's AV-9 dedup tuple catches application-layer
replay at the substrate. Cross-medium does not change the residual.

#### AV-19: Slow-medium timing-amplification

**Attack**: On LoRa or serial transports, transport-layer latency
gives adversary observable timing on internal verify operations
(verify takes longer per medium quantum than over TCP).

**Mitigation**: same as AV-16 — `verify_strict` is constant-time per
operation; transport-medium-asymmetry doesn't change the per-operation
timing surface, but does amplify per-message timing observation.

**Residual**: research-grade. LoRa / serial transports SHOULD apply
per-medium pacing on the response path so per-message latency is
medium-bandwidth-bounded rather than verify-time-bounded. Track for
Phase 3 productionization.

#### AV-20: I²P endpoint enumeration

**Attack**: Adversary enumerates the federation's I²P endpoints via
exposed routing.

**Mitigation**: I²P endpoints are public-key hashes (same as Reticulum
addresses). Enumerating them is enumerating the federation directory,
which is a public artifact (the directory is distributed by design).
Not a leak.

### 4.6 Operational / FFI boundary — adversary wants edge to drift from the discipline

#### AV-21: Lookup-cache emergence

**Attack surface (operational, not adversarial)**: future contributor
adds a public-key cache to `verify/` to reduce per-message
`lookup_public_key` cost.

**Mitigation**: PR review + MISSION.md §3 anti-pattern 1 explicit
("Caller-implemented canonicalization" generalizes — same invariant
applies to lookup). The discipline is "if profiling shows lookup is
hot, optimize persist's lookup, not edge's caching" (FSD §7
anti-pattern 1).

**Residual**: discipline-only. If lookup latency becomes a real
bottleneck, persist's `lookup_public_key` gets a worker-local cache
*inside persist*; edge stays cache-free.

#### AV-22: Per-peer special-case in handler trait

**Attack surface (architectural)**: a contributor adds a handler-trait
variant that's special-cased for one peer ("this is just for the
lens"), defeating "one shape, many peers."

**Mitigation**: PR review (FSD §7 anti-pattern 2). Peers compose
around edge, not into it. Peer-specific logic stays in the host crate.

**Residual**: catches at review time; no runtime detector.

#### AV-23: Caller-side canonicalization drift

**Attack surface (architectural)**: a contributor implements
canonicalization in edge to avoid the FFI round-trip into persist.

**Mitigation**: MISSION.md §3 anti-pattern 2 + FSD §7 anti-pattern 5.
Same closure path as CIRISPersist#7. PR review catches.

**Residual**: canonicalization byte-equivalence test (every PR runs
edge's `canonical_payload_value` against persist's
`canonicalize_envelope` for a test fixture set). Drift between the
two is the AV-5 attack vector — both directions need to stay in sync,
and the test asserts they do.

### 4.7 Supply chain — adversary wants malicious edge binaries

#### AV-24: Edge build-manifest forgery

**Attack**: Adversary publishes a malicious `ciris-edge` release tarball
with a forged or stolen build-signing key.

**Mitigation**: hybrid (Ed25519 + ML-DSA-65) signature via
`ciris-build-sign` per CIRISVerify v1.8 build-attestation (CIRISVerify
THREAT_MODEL.md §3.3). Per-release `EdgeExtras` JSON registered with
CIRISRegistry. Round-trip verified at release publication.

**Secondary**: AGPL-3.0 license-locked mission preservation per
MISSION.md §6 — a fork that publishes under a different license is
auditable.

**Residual**: a compromised maintainer's signing key (AV-2 class
applied to the edge release pipeline). Mitigation is upstream
(CIRISVerify hardware-backed key storage for release signing); not
edge's local concern at runtime.

#### AV-25: Reticulum-rs vs Leviculum implementation drift

**Attack surface (architectural)**: a malicious or buggy alternative
Reticulum implementation (Leviculum or future fork) produces
byte-different envelope encoding from Reticulum-rs, breaking
cross-implementation interop.

**Mitigation**: cross-implementation byte-equivalence regression
tests (MISSION.md §5). Bridge-trained discipline from lens-steward
bootstrap (Rust ml-dsa rc.3 ↔ dilithium-py byte-equivalent verify).
A new Reticulum impl gains an entry in edge's `Transport` enum only
after byte-equivalence passes.

**Residual**: until a sister Rust Reticulum impl actually ships, this
is a forward-looking concern. Phase 1 ships with reticulum-rs only.

---

## 5. Mitigation Matrix

| AV | Attack | Severity | Primary Mitigation | Secondary | Status | Fix tracker |
|---|---|---|---|---|---|---|
| AV-1 | Forged from attacker key | — | `lookup_public_key` directory gate | N_eff drift detection (RATCHET) | ✓ Mitigated (architectural; persist owns directory) | — |
| AV-2 | Forged from compromised key | — | Out of scope at write time | Phase 2 peer-replicate audit chain | ⚠ Phase 2 closes | persist FSD §4.5 |
| AV-3 | Replay within window | P1 | `(signing_key_id, nonce)` LRU, 5-min window | Per-instance + persist AV-9 | ⚠ Phase 1 baseline | impl |
| AV-4 | Replay outside window | — | Persist AV-9 dedup | TLS at deploy edge | ✓ Mitigated (architectural) | — |
| AV-5 | Canonicalization mismatch | — | `Engine.canonicalize_envelope` only; no edge re-impl | Cross-impl byte-equivalence test | ✓ Mitigated (CIRISPersist#7 closure) | — |
| AV-6 | Reticulum destination spoofing | — | Structural (sha256 + Ed25519 collision-resistance) | — | ✓ Mitigated | — |
| AV-7 | Schema-version downgrade | P1 | Strict allowlist | Per-version payload gates | ⚠ Track at every spec bump | — |
| AV-8 | Misrouted-message acceptance | P2 | `destination_key_id` check before parse | Metric on misroute rate | ⚠ Track | — |
| AV-9 | Dispatch on unverified bytes | **P0** | Structural (private dispatch surface) | `verify_enforcement` test category | ⚠ Pre-Phase-1; impl + test must land together | impl |
| AV-10 | Transport-level flooding | P1 | Per-transport rate caps | Bounded inbound queue, backpressure | ⚠ Phase 1 design | impl |
| AV-11 | Verify-path saturation | P1 | Cheap `lookup_public_key`; per-source verify-rate cap | Backpressure | ⚠ Phase 1 design | impl |
| AV-12 | Replay-window mem growth | P2 | Bounded LRU + window expiry | Cost-asymmetric attack | ⚠ Phase 1 design | impl |
| AV-13 | Body-size flood | P0 | `MAX_BODY_BYTES` (8 MiB) at extractor | Deploy-edge body-size cap | ⚠ Phase 1 design | impl |
| AV-14 | Parse amplification | P0 | Typed envelope; `MAX_DATA_DEPTH=32` | Bounded queue | ⚠ Phase 1 design | impl |
| AV-15 | Transport plaintext | P1 | Reticulum link encryption / TLS on HTTP | Startup warning if HTTP w/o TLS indication | ⚠ Phase 1 design | impl |
| AV-16 | Side-channel timing | P3 | Ed25519 verify_strict (constant-time) | Future: constant-response-time wrapper | ⚠ Track v0.2.x | — |
| AV-17 | Heap leak of seed bytes | **P0** | FFI boundary discipline (persist owns seed) | `identity_boundary` heap-scan test | ⚠ Pre-Phase-1; test must land with code | impl |
| AV-18 | Cross-medium replay | — | Transport-agnostic `(key, nonce)` window | Persist AV-9 catch | ✓ Mitigated by design | — |
| AV-19 | Slow-medium timing-amp | P3 | Per-medium pacing on response path | — | ⚠ Phase 3 hardening | — |
| AV-20 | I²P endpoint enum | — | Public by design | — | ✓ Not a leak | — |
| AV-21 | Lookup cache emergence | P3 | PR review + MISSION.md anti-pattern | — | ⚠ Discipline-only | — |
| AV-22 | Per-peer special-case | P3 | PR review | — | ⚠ Discipline-only | — |
| AV-23 | Caller canonicalization drift | P3 | PR review + byte-equivalence test | — | ⚠ Discipline + CI | — |
| AV-24 | Build-manifest forgery | P1 | Hybrid Ed25519 + ML-DSA-65 release sig | AGPL license-lock | ⚠ Same as lens, persist | — |
| AV-25 | Cross-Reticulum-impl drift | P2 | Byte-equivalence regression test | — | ⚠ Forward-looking (Leviculum) | — |

**Pre-Phase-1 P0 must-have bundle**: AV-9 + AV-13 + AV-14 + AV-17. These
are the structural invariants that, if not in place at Phase 1 v0.1.0,
break the threat-model claim. Implementation lands these together with
the corresponding test categories.

---

## 6. Security Levels by Deployment Tier

| Tier | Transport | FFI shape | Threat model |
|---|---|---|---|
| **Server-class peer** (production lens / agent / registry) | TCP via Reticulum primary; HTTPS fallback | edge ↔ persist via Rust API | Full §4 model applies. TLS at edge required for HTTP fallback. |
| **Standalone Rust binary** (Phase 2+) | Reticulum primary; HTTPS optional | edge ↔ persist same-process | Same as above; FastAPI / Python shim absent. |
| **Pi-class sovereign** (Phase 3) | Reticulum over LoRa / serial / TCP | edge ↔ persist embedded | Reduced attack surface (typically not internet-exposed). LoRa AV-19 timing-amp residual. |
| **Mobile bundled** (Phase 3 stretch) | Reticulum over Bluetooth-LE / TCP-when-online | edge ↔ persist via swift-bridge or jni | Apple/Android sandbox + secure enclave. Threat model dominated by upstream agent's CIRISVerify hardware-attestation tier. |
| **MCU no_std relay** (Phase 3 stretch) | Reticulum verify-only over serial / packet radio | edge verify-only; no persist | Out of full HTTP-ingest scope; verify-only relay. AV-1 + AV-3 + AV-7 still apply. |

**Critical invariant**: all tiers run the same `verify/` pipeline, the
same `transport/` trait, the same `Engine` FFI boundary. A finding in
one tier's implementation is presumed to apply to the same surface in
others unless explicitly excepted.

---

## 7. Security Assumptions

The system depends on these assumptions; if violated, the threat model
breaks.

1. **Persist's federation directory is sound.** `lookup_public_key`
   returns truthful answers; the directory is not compromised. Edge's
   AV-1 closure depends entirely on this.
2. **Persist's keyring is sound.** The steward seed lives in OS-keyring
   (CIRISPersist v0.1.3+ AV-25). Edge holds no keys; if persist's
   keyring is compromised, AV-2 applies — out of edge's scope.
3. **`Engine.canonicalize_envelope` is byte-deterministic** across host
   platforms and version bumps. Drift at this layer triggers AV-5.
4. **Reticulum's cryptographic addressing holds**: sha256 collision-
   resistance + Ed25519 unforgeability. Both are classical-cryptography
   assumptions; quantum compromise is tracked in §9 Residual.
5. **TLS at the deployment edge** for HTTP fallback. Plaintext HTTP
   exposes envelope bodies (AV-15).
6. **Clock accuracy** is within ~5 minutes of real time across all
   peers. Skew degrades AV-3 replay-window mitigation.
7. **Wire-format spec stability**: peers and edge agree on
   `EdgeEnvelope` canonicalization (FSD §3.4). Drift between peers is
   AV-5; drift between edge versions on the same peer is the
   schema-version-bump review trigger.
8. **Deployment hardware integrity**: edge's host process is not
   compromised at root. Process memory inspection, debugger attach,
   coredump access are out of scope.
9. **Build-pipeline integrity**: edge's release tarball is signed via
   the same hybrid (Ed25519 + ML-DSA-65) chain CIRISVerify v1.8
   defined; the build-signing key is hardware-protected per
   CIRISVerify's tier. AV-24 closure depends.

Critical: **Assumption 1 is load-bearing.** Edge's entire verify
guarantee reduces to "lookup_public_key returns truthful answers."
Persist's federation directory is the federation's single point of
trust at this layer, and persist's threat model is the upstream
authority.

---

## 8. Fail-Secure Degradation

All failures degrade to MORE restrictive modes, never less.

| Failure | Current behavior (target Phase 1) | Should be |
|---|---|---|
| Envelope schema parse failure | Typed `schema_invalid` reject; handler not invoked | ✓ Correct |
| Unsupported `edge_schema_version` | Typed `unsupported_schema_version` reject | ✓ Correct |
| Signature verification failure | Typed `signature_mismatch` reject; handler not invoked | ✓ Correct |
| Unknown signing key | Typed `unknown_key` reject | ✓ Correct |
| Misrouted message (`destination_key_id` mismatch) | Typed `misrouted` reject; body never parsed | ✓ Correct |
| Replay within window | Typed `replay_detected` reject | ✓ Correct |
| Persist `lookup_public_key` errors out | Typed `verify_unavailable` reject; surface as 503-equivalent | ✓ Correct (no fallback acceptance) |
| Body > `MAX_BODY_BYTES` | Typed `body_too_large` reject; transport-level 413 if HTTP | ✓ Correct |
| Inbound queue full | 429 + Retry-After (HTTP) / equivalent typed reject | ✓ Correct |
| Handler returns error | Wire-formatted error response signed and returned to sender | ✓ Correct |
| Handler panics | Edge-level panic-handler returns typed `handler_panicked`; process stays up | ✓ Correct |
| Edge process panics (verify path) | Process exits; orchestrator restart; no silent recovery | ✓ Correct |
| Reticulum link establishment fails | Transport falls back to next configured transport (HTTP if available); per-transport metric records the failure | ⚠ Phase 1 design |
| Transport encryption negotiation fails | Connection rejected; no plaintext fallback within a single transport | ✓ Correct |

Critical invariant: **`signature_verified=false` envelopes do not reach
handlers.** The verify pipeline asserts this true unconditionally;
unverified bytes never produce a handler invocation. Storing or
processing unverified envelopes would corrupt the federation's PoB §2.4
measurement.

---

## 9. Residual Risks

Risks edge mitigates but cannot fully eliminate.

1. **Compromised peer signing key** (AV-2). Edge accepts
   forged-but-correctly-signed envelopes. Closure: agent-side key
   storage hardening (CIRISVerify), Phase 2 peer-replicate, federation
   N_eff drift detection (RATCHET).
2. **Quantum compromise of Ed25519**. Phase 1 ships Ed25519-only;
   `signature_pqc` field reserved. Closure: Phase 2+ ML-DSA-65 hybrid
   verify per persist v0.4.0+. Until then, accept-not-verify on the
   PQC field per OQ-11.
3. **Cross-implementation drift** when Leviculum or other Reticulum
   forks emerge. Mitigation is regression test (AV-25); residual is
   any drift not caught by the test fixture set.
4. **Persist directory compromise** (Assumption 1). Out of edge's
   scope; persist's threat model is authoritative.
5. **Same-host attacker reading persist's keyring file** (CIRISPersist
   AV-25 residual on `SoftwareSigner` fallback). Closure: hardware-
   backed keyring (TPM / Secure Enclave / StrongBox) per
   CIRISPersist's tier classification. Edge inherits whatever tier
   persist runs at.
6. **Side-channel timing leakage of directory membership** (AV-16).
   Low-impact (key ids are public) but trackable; LoRa/serial timing
   amplification (AV-19) makes the channel slower-but-not-more-leaky.
7. **All federation peers compromised simultaneously** (PoB §5.1
   residual). Per Accord NEW-04, no detector is complete. Topological
   cost-asymmetry over time is the federation-level response.
8. **DNS bootstrap poisoning for HTTP fallback** (the HTTP transport
   resolves a hostname; DNS spoof points to attacker). Mitigation:
   Reticulum is canonical; HTTP fallback is for environments where
   Reticulum can't run, and those environments inherit DNS as part of
   their existing trust model. Track at HTTP-transport documentation.

---

## 10. Posture Summary

```
PRE-PHASE-1 P0 MUST-HAVE BUNDLE — must land with v0.1.0
  ⚠ AV-9   Structural verify-pipeline gating (no unverified bytes past handler boundary)
  ⚠ AV-13  MAX_BODY_BYTES = 8 MiB at all transport entry points
  ⚠ AV-14  Typed envelope deserialize + MAX_DATA_DEPTH=32
  ⚠ AV-17  FFI boundary heap-scan property test

PHASE-1 P1 BUNDLE — must land for production cutover
  ⚠ AV-3   Replay LRU with 5-min window
  ⚠ AV-5   Cross-impl byte-equivalence test for canonicalize_envelope
  ⚠ AV-7   Strict version allowlist
  ⚠ AV-10  Per-transport rate caps + bounded queue
  ⚠ AV-11  Cheap lookup_public_key + per-source verify-rate cap
  ⚠ AV-15  TLS startup warning on HTTP-without-TLS
  ⚠ AV-24  Hybrid release signature via ciris-build-sign

PHASE-2-CLOSES (architecturally deferred)
  ⚠ AV-2   Stolen-key forgery (peer-replicate audit chain in persist)
  ⚠ AV-25  Cross-Reticulum-impl drift (when sister impl emerges)

V0.2.X TRACK (low blast radius)
  ⚠ AV-8, AV-12, AV-16, AV-21, AV-22, AV-23

PHASE-3 (multi-medium hardening)
  ⚠ AV-19  Slow-medium timing-amplification (LoRa / serial pacing)

ARCHITECTURALLY MITIGATED (no further action required)
  ✓ AV-1   Forged from attacker key (lookup_public_key gate)
  ✓ AV-4   Replay outside window (persist AV-9 catches)
  ✓ AV-6   Reticulum destination spoofing (structural)
  ✓ AV-18  Cross-medium replay (transport-agnostic window)
  ✓ AV-20  I²P endpoint enumeration (public by design)
```

**Bottom line**: edge is a thin verify-and-dispatch substrate. The
threat-model story reduces to four invariants:

1. No unverified bytes reach handlers (AV-9).
2. No untyped bytes pass parse (AV-14).
3. No seed bytes enter edge's heap (AV-17).
4. No body exceeds the size cap (AV-13).

If those four hold at v0.1.0, the rest of the threat model is
architecturally consistent with the federation meta. The pre-Phase-1
implementation work is exactly: land those four invariants with the
tests that assert them, then build outward.

Federation-primitive contribution: edge fills the N1 (cryptographic
addressing) and N2 (multi-medium transport) gaps that
`FEDERATION_THREAT_MODEL.md` §5 explicitly lists as unfilled.
Submitting this threat model upstream closes those rows in the
federation's per-primitive coverage table.

---

## 11. Update Cadence

This document is updated:
- On every minor release: comprehensive review.
- On every published security advisory affecting deps (reticulum-rs,
  tokio, axum, serde): addendum in §4 + `cargo audit` re-run.
- On every wire-format schema-version bump: AV-5 / AV-7 review against
  the new shape, byte-equivalence test against persist's
  `canonicalize_envelope`.
- On every new transport landing in `transport/`: AV-10 / AV-13 / AV-15
  / AV-19 review for the new medium's specific surface.
- On every cross-trinity boundary change (CIRISPersist flips the
  Engine surface, CIRISVerify flips keyring API, CIRISAgent flips
  trace wire format): trust-boundary review + interaction matrix
  update.

Last updated: 2026-05-03 (v0.0 baseline scaffold; pre-Phase-1
implementation. AV catalog targets the implementation work to come;
mitigation status is "design only" until v0.1.0 lands).
