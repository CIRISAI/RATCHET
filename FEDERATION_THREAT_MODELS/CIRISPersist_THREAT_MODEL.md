# CIRISPersist Threat Model

**Status:** v0.1.2 baseline (5 of 5 known integration-blocking
exposures from v0.1.1 mitigated). Updated each minor release.
**Audience:** lens team integrators, federation peers, security reviewers.
**Companion:** [`MISSION.md`](../MISSION.md), [`FSD/CIRIS_PERSIST.md`](../FSD/CIRIS_PERSIST.md).
**Inspired by:** [`CIRISVerify/docs/THREAT_MODEL.md`](https://github.com/CIRISAI/CIRISVerify/blob/main/docs/THREAT_MODEL.md) — the structural template.

---

## 1. Scope

### What CIRISPersist Protects

CIRISPersist is the lens-side ingest substrate (Phase 1) and, by trait
shape, the agent-side persistence service (Phase 2/3). It protects:

- **Corpus integrity**: every persisted trace was provably produced
  by the claimed agent at the claimed moment, OR is rejected. There
  is no third state. The federation's PoB §2.4 N_eff measurement
  depends on this — forged traces in the corpus would degrade the
  Sybil-resistance signal the Federated Ratchet rests on.
- **Idempotency**: agent retries (TRACE_WIRE_FORMAT.md §1: up to
  10× batch_size events deep) cannot inflate corpus counts. The
  dedup key `(trace_id, thought_id, event_type, attempt_index, ts)`
  with `ON CONFLICT DO NOTHING` is the structural guarantee.
- **Privacy at trace tier**: PII never crosses the persistence
  boundary at trace levels where it isn't warranted. `generic` is
  content-free by design; `detailed`/`full_traces` route through a
  scrubber boundary.
- **Backpressure honesty**: agents get structured 429 + Retry-After
  on saturation, never silent drop. The agent's own retry buffer
  (TRACE_WIRE_FORMAT.md §1) closes the loop.
- **Outage tolerance**: backend failure does not lose signed
  evidence. The redb journal preserves the agent-shipped bytes
  byte-exact for replay (FSD §3.4 #2).
- **Audit anchor capture**: the agent's per-action audit-chain link
  is captured on every `ACTION_RESULT` row (FSD §3.2). Anchor
  *verification* against the agent-side audit log is Phase 2's
  peer-replicate work.
- **Memory-safe parsing of untrusted bytes**: Rust's static
  guarantees at the wire edge close the recurring CVE class for
  network-facing services (MISSION.md §2 — `server/`).
- **(NEW v0.1.3) Cryptographic provenance of deployment handling**:
  every persisted row carries a four-tuple envelope
  (`original_content_hash`, `scrub_signature`, `scrub_key_id`,
  `scrub_timestamp`) that proves *this specific deployment processed
  this specific payload at this specific time*. Always present —
  every component, every trace level, key never null (FSD §3.3 step
  3.5 + §3.4 robustness primitive #7). The federation primitive
  PoB §3.1 — "the lens role is a function any peer can run on data
  the peer already has" — becomes cryptographically attestable.
  Bilateral cryptography: agent's wire-format §8 signature proves
  authorship, lens's v0.1.3 scrub envelope proves handling.
- **(NEW v0.1.3) Single-key federation identity**: the scrub-signing
  key is also the deployment's Reticulum destination
  (`SHA256(public_key)[..16]`, PoB §3.2 — addressing IS identity)
  and the registry-published public key. One key, three roles.
  No separate "network identity" key; no translation layer
  between cryptographic provenance and federation transport.

### What CIRISPersist Does NOT Protect (Phase 1)

- **Agent-side key compromise**: if an agent's Ed25519 signing key
  is leaked or coerced, an adversary can produce indistinguishable
  forged traces under that agent's identity. The federation's
  N_eff and σ-decay metrics over time will eventually surface
  anomalies (PoB §2.4 + §5.6), but the persistence layer cannot
  detect forgery from a stolen-but-valid key at write time.
- **Network-edge TLS / certificate infrastructure**: HTTPS termination
  is the lens deployment's concern. CIRISPersist does not pin
  certificates or verify TLS state.
- **Postgres-server compromise**: the persistence backend is
  trusted. If Postgres or the redb journal disk are compromised,
  the threat model breaks.
- **Audit-chain re-verification (Phase 1)**: anchor fields are
  captured but not cross-checked against the agent's audit_log.
  Phase 2 peer-replicate (FSD §4.5) closes this.
- **Pre-cutover corpus integrity**: `accord_traces` retains
  pre-cutover history with whatever properties the previous lens
  pipeline gave it. CIRISPersist makes no claims about that table.
- **Cross-tier privacy bridging**: a `generic`-tier trace co-
  resident in the same DB as a `full_traces` trace is not a
  CIRISPersist concern. Lens-side query authorization is the lens's
  job.
- **Phase 3 surfaces**: agent runtime state, memory graph, and
  governance tables are part of the Backend trait but unimplemented
  in v0.1.x. Their threat model is sketched here for forward
  compat; the active surface is Phase 1.

---

## 2. Adversary Model

### Adversary Capabilities

The adversary is assumed to have:

- **Full source-code access** (AGPL-3.0; public).
- **Ability to mint arbitrary Ed25519 keypairs** and sign anything.
- **Ability to run their own agents** on the network — including
  registering public keys via the standard `accord/agents/register`
  flow.
- **Network access to the lens HTTP endpoint**, including the
  ability to send arbitrary bytes to `/api/v1/accord/events`.
- **Replay capability**: capture any in-transit batch and re-send
  it at any point.
- **Network MITM** between an honest agent and the lens, with
  ability to drop, delay, or modify bytes if not protected by TLS.
- **Limited side-channel observation**: response timing,
  HTTP status codes, error message bodies.
- **Ability to read public CI artifacts**: every test output,
  every published wheel, the deny.toml + dep tree.
- **Compute resources sufficient for classical cryptography** (but
  not for breaking Ed25519 within current physics).

### Adversary Limitations

The adversary is assumed to NOT have:

- **The ability to break Ed25519** within polynomial time on
  classical hardware. (PoB §6 acknowledges quantum risk; ML-DSA-65
  hybridization is Phase 2+.)
- **Compromised the public-key directory**: the lens's
  `accord_public_keys` table. If the directory is owned, every
  signature verification is meaningless. The directory is part of
  the lens deployment trust boundary.
- **Compromised any honest agent's signing key**: if they did, see
  §1 "What CIRISPersist Does NOT Protect (Phase 1)" — that's
  out-of-scope at the persistence layer.
- **Compromised the Postgres backend** that the lens writes to.
- **Compromised the redb journal disk** location (default
  `/var/lib/cirislens/journal.redb`).
- **Physical access** to the lens deployment hardware.
- **Ability to read TLS-encrypted traffic** between the agent and
  the lens. (TLS termination is upstream of CIRISPersist; if it's
  off, that's a deployment misconfiguration.)
- **Quantum compute** capable of breaking Ed25519 today. (Tracked
  in §8 Residual Risks; Phase 2+ adds ML-DSA-65 hybrid signatures
  per PoB §6.)

---

## 3. Attack Vectors

Sixteen attack vectors organized by adversary goal. Each lists the
attack, the primary mitigation present in v0.1.1, the secondary
mitigation, and the residual risk.

### 3.1 Forgery — adversary wants their bytes counted as real evidence

#### AV-1: Forged trace from attacker-minted key

**Attack**: Attacker generates a fresh Ed25519 keypair, signs a
synthetic CompleteTrace with it, submits to `/api/v1/accord/events`.

**Mitigation**: Public-key directory lookup before verification.
The `signature_key_id` in the trace must resolve to a registered
key in `cirislens.accord_public_keys`. Unknown keys → typed
`UnknownKey` error → HTTP 422 → zero rows persisted. Attacker
must register their key id through the lens's
`accord/agents/register` flow first, which is gated by the lens's
own admission policy (out of scope for CIRISPersist; it's the
lens's policy lever per Annex E of the Accord).

**Secondary**: Per-agent N_eff drift over time. A fresh-keyed
"agent" with no behavioral history fails the σ-decay floor and PoB
§2.4 codimension before it earns federation standing. CIRISPersist
provides the substrate the lens scoring layer measures over.

**Residual**: An attacker who can register a key id (e.g., as a
sovereign-mode agent) and produce 30 days of N_eff > 9 trace
behavior earns standing. That's *exactly* the cost-asymmetry PoB
§2.1 names — running real ethical-reasoning is what the network
asks of every member.

#### AV-2: Forged trace using compromised legitimate key

**Attack**: Attacker exfiltrates an honest agent's signing key
(via Phase 2/3 secrets-manager compromise, key-material leak in
backups, social engineering, etc.), signs a malicious trace under
that agent's identity.

**Mitigation**: **Out of CIRISPersist's protection scope.** The
persistence layer cannot distinguish a stolen-key signature from a
legitimate one — both verify against the same registered public key
by construction. The federation N_eff / σ time-series provides
*statistical* drift detection (anomalous behavior under a stable
identity), but at write time the lens accepts.

**Secondary**: The agent's audit-log chain (captured as the audit
anchor on every `ACTION_RESULT` row, FSD §3.2) provides
post-incident forensics. Phase 2 peer-replicate (FSD §4.5)
cross-validates the chain link against the agent's local audit_log,
making chain-tampering detectable.

**Residual**: Until Phase 2 closes peer-replicate verification,
key-compromise-then-forgery is undetectable at ingest. The agent's
local secrets manager + hardware-backed key storage (CIRISVerify's
threat model is the relevant document) is the upstream mitigation.

#### AV-3: Replay of captured legitimate batch

**Attack**: Network MITM captures a valid signed batch in transit,
replays it (or a slightly modified copy) to the same lens later.

**Mitigation**: Idempotency on
`(trace_id, thought_id, event_type, attempt_index, ts)` UNIQUE
index. Re-submitting the same batch produces 0 inserts +
N conflicts (verified by `idempotent_replay` test). Inserted-vs-
conflicted counts in `BatchSummary` surface to ops dashboards.

**Secondary**: TLS at the deployment edge prevents capture in
the first place; this is a deployment concern, not CIRISPersist's.

**Residual**: A captured batch *replayed against a different lens
deployment* (e.g., a federation peer that hasn't seen it yet) lands
once, by design. That's what trace replication is supposed to do
(FSD §4.4 / PoB §5.1). Per-peer dedup is each peer's local
guarantee.

#### AV-4: Canonicalization-mismatch attack

**Attack**: Adversary exploits a byte-difference between what the
agent's signer canonicalizes and what the lens's verifier
canonicalizes. Either:
- Submit bytes the lens accepts but the agent never signed
  (verifier produces *different* canonical bytes that happen to
  hash to a valid pre-existing signature — preimage attack on
  Ed25519 is computationally infeasible).
- Submit bytes the agent signed but the lens rejects (DoS — bytes
  the agent has paid to produce get dropped).

**Mitigation**:
- The 14 byte-exact canonicalization parity tests in
  `verify::canonical::tests`. Each cross-checked against
  `python3 -c "import json; json.dumps(...)"` ground truth.
- Pluggable `Canonicalizer` trait so the agent + lens stay in sync
  on conventions (Python `json.dumps(sort_keys=True,
  separators=(',', ':'))` today; RFC 8785 JCS reserved for the
  agent-flips path).
- Real-fixture integration suite (`tests/wire_format_fixtures.rs`)
  exercises actual signed traces from CIRISAgent `release/2.7.8`.

**Secondary**: Ed25519's collision and preimage resistance bound
the "produce different bytes that verify" branch to
2^128 work — practically infeasible.

**Residual**:
- **Float canonicalization drift** (CRATE_RECOMMENDATIONS §2.9).
  Python's `repr(float)` and Rust's `ryu` agree on shortest
  round-trip-able output for the common cases but may differ on
  edge cases. The wire format §8 doesn't include floats in the
  *outer* canonical fields, but per-component `data` payloads
  carry floats (durations, scores). A deliberately constructed
  float in agent-shipped bytes that round-trips through Rust to a
  different string would fail verify silently. Status: not
  triggered by any production fixture; track if a real
  divergence appears in the corpus.
- **Timestamp formatting drift** (`verify::ed25519::format_iso8601`).
  ✓ **CLOSED v0.1.8**. Was: re-format `DateTime<Utc>` via chrono's
  `%.6f%:z` for canonicalization, which always emitted six
  microsecond digits. Python's `datetime.isoformat()` drops the
  fraction entirely when microseconds==0, so a wire timestamp of
  `2026-04-30T00:15:53+00:00` became `2026-04-30T00:15:53.000000+00:00`
  on the verify side, canonical bytes diverged, signature rejected.
  Hit lens production cutover. v0.1.8 closes by adding
  `schema::WireDateTime` — wraps `(raw: String, parsed: DateTime<Utc>)`
  with `Serialize` emitting the raw bytes verbatim. Replaces
  `DateTime<Utc>` in `CompleteTrace.{started_at, completed_at}`
  and `TraceComponent.timestamp`. `canonical_payload_value` now
  reads `.wire()` instead of `format_iso8601(&parsed)`.
  Regression coverage: `tests/av4_timestamp_round_trip.rs` (5
  scenarios including the production-bug zero-microsecond shape).

### 3.2 Denial of Service — adversary wants the lens unable to receive evidence

#### AV-5: Schema-version flood (memory leak DoS) **[v0.1.1 exposure]**

**Attack**: Adversary submits a stream of bodies with malformed
`trace_schema_version` strings (random 64-byte strings, etc.).
Each rejected version path runs `parse_lenient` which
`Box::leak`s the unrecognized string into `&'static str` for
diagnostic purposes (`src/schema/version.rs:94`). Memory grows
unboundedly per request.

**Mitigation in v0.1.1**: **None.** The leak is real and exploitable.

**Recommended hot-fix for v0.1.2**: replace the `Box::leak` path with
an owned `String` variant on `SchemaVersion` (or a separate
`UnrecognizedSchemaVersion` typed-wrapper passed through the error
path). Cost: ~30 minutes; touch `version.rs`, `envelope.rs::from_json`.

**Secondary mitigation today**: deploy behind a rate limiter at the
lens's HTTPS termination layer (nginx, Envoy, etc.) capping
requests-per-source-IP. Mitigates the rate but not the
memory-amplification ratio.

#### AV-6: JSON-bomb / deserialization amplification

**Attack**: Adversary submits a JSON body with deeply nested
structure (`[[[[...]]]]` 10000 deep) or a single key with a
1GB string value. `serde_json` by default has no depth limit and
parses into `serde_json::Value` for the `data` blobs. Memory
allocation amplification.

**Mitigation in v0.1.1**: **Partial.**
- The bounded queue (`DEFAULT_QUEUE_DEPTH=1024`) prevents
  *throughput* amplification — only N bodies in flight at once.
- The schema at the typed level forces concrete struct shapes
  (`BatchEnvelope`, `CompleteTrace`, etc.) — depth-bombs in the
  *envelope* fail at typed deserialize.
- However, the per-component `data` field is
  `serde_json::Map<String, serde_json::Value>` — deeply nested
  JSON inside `data` *will* parse and allocate.

**Recommended hot-fix for v0.1.2**:
- Set max body size at the axum extractor layer:
  `.layer(DefaultBodyLimit::max(8 * 1024 * 1024))` (8 MiB matches
  the largest production fixture's `full_traces` 3 MiB with 2.6×
  headroom).
- Add a recursion-depth guard in the typed accessor that walks
  `data` (e.g., reject `data` JSON deeper than 32 levels).

**Secondary**: deployment-edge body-size limit at the proxy layer.

**Residual**: an attacker with a registered, accepted public key
who submits inflated-but-syntactically-valid bodies pays the
cost-asymmetry PoB §2.1 names — they're spending real LLM cost to
inflate the corpus, and N_eff measurement detects this as
behavioral drift.

#### AV-7: Body-size flood (no max body limit) **[v0.1.1 exposure]**

**Attack**: Adversary submits arbitrarily large bodies. axum's
`Bytes` extractor reads the entire body into memory before queue
submission.

**Mitigation in v0.1.1**: **None at the crate level.** The lens's
deployment-edge proxy (nginx, HAProxy, etc.) typically caps body
size at 1-100 MiB; this is a defense-in-depth gap, not a guaranteed
exposure.

**Recommended hot-fix for v0.1.2**: explicit
`DefaultBodyLimit::max(N)` on the axum router. Match the operational
maximum (3-10 MiB based on production fixture sizes).

#### AV-8: Queue saturation

**Attack**: Adversary floods the endpoint to fill the bounded
mpsc channel.

**Mitigation in v0.1.1**: 429 + Retry-After on `QueueError::Full`.
The agent already retries up to 10× batch_size deep
(TRACE_WIRE_FORMAT.md §1); legitimate flow stays correct under
saturation.

**Secondary**: the persister task is a single consumer with a
journal-on-Postgres-failure path — backpressure surfaces at the
bottleneck (the DB), not at the queue boundary.

**Residual**: an attacker with high request rate denies service to
honest agents. Rate limiting per source IP at the deployment edge
is the standard defense; not CIRISPersist's responsibility.

### 3.3 Corruption — adversary wants false data persisted or true data dropped

#### AV-9: Idempotency-key collision across agents

**Attack**: Two distinct agents submit batches that share
`(trace_id, thought_id, event_type, attempt_index)`. The second
arrival hits ON CONFLICT DO NOTHING and is silently skipped.

**Mitigation in v0.1.1**: **Partial.** The dedup key does NOT
include `agent_id_hash` or `signature_key_id`. The wire-format
spec (TRACE_WIRE_FORMAT.md §3) mandates `trace_id` is "Globally
unique per agent" — `trace-<thought_id>-<YYYYMMDDHHMMSS>` —
relying on `thought_id` being agent-unique. If a malicious agent
reuses another agent's `thought_id` shape, they could DOS the
victim's traces.

**Recommended hot-fix for v0.1.x**: extend the dedup key to
include `agent_id_hash` (or `signing_key_id`) at the SQL UNIQUE
index level. SQL change: drop the existing
`trace_events_dedup` UNIQUE index, recreate as
`(agent_id_hash, trace_id, thought_id, event_type, attempt_index,
ts)`. Migration `V003__dedup_key_includes_agent.sql`. The
in-memory backend's `events` HashMap uses the same key shape.

**Secondary**: each agent's `agent_id_hash` is sha256-prefix of
the agent's pubkey (TRACE_WIRE_FORMAT.md §3). Two distinct
agents producing colliding `trace_id`s + thought_ids requires
hash collision OR coordinated namespace attack — both cost-
asymmetric.

**Residual**: until the dedup key extends, a coordinated
attacker can DOS specific victims by pre-claiming their dedup
tuples.

#### AV-10: Audit anchor injection

**Attack**: Attacker (with a registered key) submits a trace
with crafted `audit_sequence_number`/`audit_entry_hash` that
will conflict with a future legitimate ACTION_RESULT row,
forcing dedup to skip the legitimate one.

**Mitigation in v0.1.1**: **Partial.** The audit anchor fields
are NOT part of the dedup key. They're columns on the
`trace_events` row, populated only on the ACTION_RESULT row. The
dedup key (per AV-9 above) is the trace shape; anchor is
auxiliary. Two ACTION_RESULT rows with same dedup tuple but
different anchors: the second is skipped, anchor mismatch is
post-facto detectable.

**Phase 2** (FSD §4.5 peer-replicate): the agent's audit_log
chain provides cross-validation. A row with an anchor that
doesn't match the agent's claimed chain link → flagged for
review.

**Residual**: under Phase 1, anchor field is captured-but-not-
verified. Treat it as "data preserved for Phase 2 cross-check,"
not "data the lens trusts today."

#### AV-11: Public-key directory poisoning via re-registration **[v0.1.1 design point]**

**Attack**: An attacker who can call `register_public_key` (the
PyO3 Engine method) submits the same `signature_key_id` an
honest agent already registered, but with a different public
key. The current SQL is:

```sql
INSERT INTO cirislens.accord_public_keys
  (signature_key_id, public_key_b64, agent_id_hash)
  VALUES ($1, $2, $3)
  ON CONFLICT (signature_key_id) DO NOTHING
```

ON CONFLICT DO NOTHING means re-registration is silently
ignored — the original key wins. This is the *correct* behavior
for an attacker trying to overwrite, but it's **also the wrong
behavior** if the legitimate agent is rotating keys.

The doc currently says: "Re-registering a *different* key for
the same id is treated as the agent's choice — no rotation alarm
yet; that's a follow-up for v0.2.x."

**Mitigation in v0.1.1**: **The first key wins; subsequent
re-registrations are silently ignored.** That blocks the attacker
*and* blocks legitimate rotation. Asymmetric-bad: the lens
operator's registration tooling needs to manually
`UPDATE accord_public_keys` for legitimate rotation, with whatever
out-of-band authorization gates the lens deployment requires.

**Recommended for v0.2.x**: explicit rotation API. Two methods:
- `register_public_key` — INSERT-only; rejects on conflict.
- `rotate_public_key(signature_key_id, new_key_b64,
  rotation_proof: signed-by-old-key statement)` — verifies the
  rotation request is signed by the *old* key, then updates.
- A `revoked_at` timestamp column already exists in V001 but is
  not exercised; revocation API is a sibling.

**Residual**: today, key rotation requires the lens operator to
issue a manual UPDATE — that's a security feature (no automated
rotation under attacker control), but it's an operational
papercut.

#### AV-12: Schema-version downgrade

**Attack**: An attacker convinces the lens to treat a future
agent's v2.8.0 batch as v2.7.0, exploiting a known v2.7.0 weakness
that v2.8.0 fixed.

**Mitigation in v0.1.1**: **Strong.** `SUPPORTED_VERSIONS` is a
strict allowlist (`["2.7.0"]`). Out-of-set versions hit HTTP 422.
There is no "best-effort" / "downgrade-and-try" branch. To accept
v2.8.0, the lens must upgrade `ciris-persist` to a release that
extends the constant — which is a deliberate operator decision.

**Residual**: when v2.7.0 and v2.8.0 are *both* in
`SUPPORTED_VERSIONS` (the rolling-deploy window per FSD §10
Phase 3), an attacker who can inject the version field could
target the older path. Mitigation at that point: ensure each
version's payload-shape gate is independent — a v2.8.0 payload
labeled v2.7.0 must fail typed deserialize. Track for the actual
version-bump PR.

#### AV-13: Cross-trace JSONB injection (Phase 3 surface)

**Attack**: An attacker submits a `data` blob crafted to
exploit a future SQL query that reaches into the JSONB column.

**Mitigation in v0.1.1**: payload is stored as parameterized
JSONB via tokio-postgres typed binding. There is no string
interpolation of payload content into SQL. SQL injection at the
INSERT layer is structurally not possible.

**Phase 3 risk**: queries that *read* the JSONB (e.g.,
`payload->>'audit_sequence_number'`) need parameterized binding
on the JSONB-path operands too. Track at Phase 3 scope.

### 3.4 Privacy — adversary wants content text exposed at a tier where it isn't warranted

#### AV-14: Scrubber bypass via schema-altering callback

**Attack**: An attacker who controls the lens's scrubber
callable returns a modified envelope that drops `trace_level`
from `full_traces` to `generic`, bypassing the next layer's
content-handling assumptions.

**Mitigation in v0.1.1**: the engine validates scrubber output:
- `trace_schema_version` must match input
- `trace_level` must match input
- `events[]` length and per-event discriminants must match
- Violation → typed `ScrubError::External` → HTTP 422

Only payload *content* is scrubber-mutable.

**Secondary**: if the scrubber itself is compromised, the Python
process security boundary is the upstream concern. CIRISPersist
trusts the callable it was constructed with.

**Residual**: a malicious scrubber that modifies content in
ways that drop *necessary* signal (e.g., zeroing all
`coherence_score` floats) corrupts the corpus without altering
schema. Detection is post-facto via N_eff / PC1 anomaly
detection at the lens scoring layer.

#### AV-15: PII leak via error messages

**Attack**: An error path includes content from the request
body in the error message, leaking PII into logs / HTTP error
responses.

**Mitigation in v0.1.1**: **Partial.** Audit findings:
- `Error::UnsupportedSchemaVersion { got, ... }` includes the
  attacker-submitted version string. **Today this is bounded**
  (typed string from the JSON parse), but if an attacker can
  inject newlines or terminal escape sequences, they leak into
  log output. Sanitize with `.escape_debug()` before logging.
- `Error::FieldTypeMismatch { field, expected, got }` includes
  the type name — never the value. ✓
- `Error::Json(serde_json::Error)` — serde_json's error
  formatter includes ~30 chars of context around the parse
  error. For a `data` blob containing PII, that context could
  leak a fragment. **Mitigate** with a `Display` wrapper that
  strips snippets in production builds.
- `Error::Backend(String)` — Postgres error messages are
  included verbatim. Postgres's error formatter can leak
  schema names + sometimes parameter values. Already public in
  the deployment, but worth keeping out of HTTP 500 response
  bodies.

**Recommended hot-fix for v0.1.2**: introduce a
`Display::sanitize_for_response()` mode that emits only the
typed-error variant name + a stable opaque id, with the full
context kept in tracing-only logs. The HTTP error response
becomes `{"detail": "schema_version_unsupported",
"correlation_id": "uuid"}`; the log keeps the verbose form.

#### AV-16: Side-channel timing on verify

**Attack**: Adversary measures response time differences to
distinguish "unknown key" vs "known key + wrong signature" vs
"known key + right signature, wrong canonical bytes" — gleaning
information about the public-key directory or the
canonicalization pipeline.

**Mitigation in v0.1.1**: **Partial.** Ed25519 `verify_strict` is
constant-time over the signature/key path. However:
- The public-key directory lookup short-circuits on
  unknown-key (returns before signature math runs). Timing leaks
  membership.
- Canonicalization bytes are deterministic per input but
  *length* differs based on payload size — observable.

**Recommended for v0.2.x**: constant-response-time wrapper that
sleeps to a P99 budget on the rejection path. Not free
operationally (latency tax on the happy path too if
implemented naïvely) — track as research-grade hardening.

**Residual**: a network-adjacent attacker can probably enumerate
`signature_key_id`s via timing oracle. The federation primitive
treats `signature_key_id` as public anyway (it's emitted on
every trace), so directory enumeration is not a high-impact leak.

### 3.5 Provenance — adversary wants to forge "deployment handled this" attestation

These vectors are introduced by v0.1.3's always-on scrub-signing
contract (FSD §3.3 step 3.5 + §3.4 robustness primitive #7). The
contract turns the `pii_scrubbed = true` boolean column from a
*trust* claim into a *verifiable* claim — every persisted row
carries cryptographic proof of the deployment's handling.

#### AV-24: Lens-scrub bypass / forgery

**Attack**: An adversary with row-level write access to the
lens's Postgres (compromised lens process, malicious DB
operator, etc.) inserts rows with `pii_scrubbed = true` but no
matching `scrub_signature` — or with a scrub_signature signed by
a key the federation doesn't recognize. Downstream peers reading
these rows treat them as legitimately-handled.

**Mitigation in v0.1.3**: every persisted row from v0.1.3's
pipeline carries a four-tuple envelope (`original_content_hash`,
`scrub_signature`, `scrub_key_id`, `scrub_timestamp`).
Downstream peers verify `ed25519_verify(scrub_signature,
canonical(payload), known_pubkey_for(scrub_key_id))` before
trusting the row's provenance. Rows with NULL envelope columns
or an unrecognized `scrub_key_id` are flagged and not counted
in the federation primitive's N_eff measurement.

The signing key is the deployment's own — a malicious operator
who controls the lens process *also* controls the signing key
(or, with hardware-backing, the keyring access path), and can
mint apparently-valid envelopes. This is the same trust boundary
as agent-side AV-2 (compromised key); the persistence layer
cannot detect bytes signed by the legitimate key under
adversarial control. Downstream PoB N_eff drift detection
(behavioral anomaly over time) is the federation-level
mitigation.

**Secondary**: the lens publishes its public key to the
registry / lens-discovery layer at deploy time. A peer fetching
rows can cross-check `scrub_key_id` against the registry's
roster of legitimate lens keys; rows signed by a key not in the
registry are quarantined.

**Residual**: a compromised lens with legitimate keyring access
can mint envelopes that *look* valid. Detection is statistical
(N_eff drift over time) rather than pointwise — the same residual
the agent-side AV-2 has. PoB §6 framing applies.

#### AV-25: Scrub-key compromise

**Attack**: An adversary extracts the deployment's signing-key
seed from the host's filesystem / memory / debug interface and
mints arbitrary envelopes under that key. Forged "deployment X
processed payload Y at time Z" attestations the federation
treats as legitimate.

**Mitigation in v0.1.3**: `ciris-keyring` (CIRISVerify's Rust
crate) stores the seed in OS-keyring backed by hardware where
available — Linux Secret Service / TPM 2.0; macOS Keychain /
Secure Enclave; iOS / Android StrongBox; Windows DPAPI / TPM.
The Python process never holds the seed bytes; the seed never
crosses the FFI boundary. Hardware-backed deployments require
physical access (and on most platforms, exploitation of an
enclave-grade vulnerability) to extract.

**One key, three roles** (PoB §3.2): the scrub-signing key is
*also* the deployment's Reticulum destination address (Phase 2.3)
*and* the registry-published public key. Compromise the key,
you compromise all three roles simultaneously — cryptographic
provenance, federation transport address, registry identity.
This tripled cost-asymmetry is what makes hardware-backed
keyring entries materially stronger than software-only seeds:
losing the key isn't just "rows you signed are now suspect" but
"your peer-to-peer address is now hijacked AND your registry
entry needs revocation." The federation primitive's
self-application of risk (PoB §2.1: the cost of being a real
member is the cost of *being attacked* if your key leaks)
strengthens the operational case for hardware backing.

**Secondary**: `ciris-keyring`'s `SoftwareSigner` fallback exists
for dev / sovereign deployments without hardware backing. The
seed is in OS-keyring on disk — root access on the host can
extract it. Named residual; mitigation is operational
(avoid software-fallback in production, prefer hardware-attested
deployments).

**Residual**: software-backed deployments have no key isolation
beyond OS keyring file permissions. Mitigations are
deployment-level (full-disk encryption, restrict who has root,
short-lived keys with rotation through the registry's
revocation surface). CIRISVerify's threat model §5 — Security
Levels by Hardware Type — is the authoritative classification
of what each backing tier provides.

#### AV-26: Multi-worker migration race

**Attack surface (operational, not adversarial)**: a lens deployment
spinning up multiple uvicorn workers / replica pods / sidecars
concurrently against a single Postgres instance. Each worker calls
`Engine(...)` on startup, which connects + calls `run_migrations()`.

**Pre-v0.1.5 failure**: the workers raced on Postgres catalog
inserts — TimescaleDB hypertable type registration in `pg_type`,
`IF NOT EXISTS` checks across the V001 + V003 migration set,
refinery's own schema_history bootstrap. The second worker through
hit `42P07 relation already exists` (or, less commonly, deadlock
on `pg_namespace`) which refinery wrapped opaquely as
`"error asserting migrations table — db error"`. Production
deployments saw worker pods fail readiness, restart, race again,
and stay unhealthy until the orchestrator escalated.

This is not a threat in the adversarial sense — there is no
attacker — but it is a real availability vector: a config change
that scales worker count from 1 to N can trigger a stuck-restart
loop on cold deploys. THREAT_MODEL.md catalogues it for the same
reason MISSION.md treats reliability as a mission concern: a
substrate that's unreachable can't carry evidence.

**Mitigation in v0.1.5**: session-scoped Postgres advisory lock
(`pg_advisory_lock(0x6369_7269_7370_7372)` — bytes spell
`"cirispsr"` for grep visibility) acquired on a *dedicated single-
use connection* (not from the pool, so the lock can't taint a
recycled pool conn). The lock is held across refinery's
multi-transaction migration phase. First worker wins immediately;
subsequent workers block on the lock, wake when the first worker's
session closes, and proceed cleanly through the now-no-op migration
phase. Lock auto-releases on connection close — including the
panic-mid-migration case (the connection task observes EOF, the
session ends, the lock goes; no orphaned locks across worker
crashes).

**Diagnostic surface**: v0.1.5 also added `Error::Migration {
sqlstate: Option<String>, detail: String }` so the lens sees
`store: migration: [SQLSTATE] detail` instead of "db error". 42P07
should not appear at v0.1.5+ unless schema is externally mutated
mid-flight; 40P01 (deadlock detected) is the indicator for "retry
construction"; 08006 is "connection lost, retry"; 42501 is
"DSN user lacks DDL rights — config bug, not transient."

**Residual**: a worker holding the lock that is *paused indefinitely*
(SIGSTOP, kernel scheduler starvation, or a tracing tool with a
breakpoint inside the migration phase) leaves concurrent workers
blocked on `pg_advisory_lock`. This is a deployment-operational
concern (orchestrator liveness probes catch it; the held-lock
worker's connection eventually times out per Postgres
`tcp_keepalives` if configured). Out of scope for the
substrate.

**QA harness coverage**: `tests/qa_harness.rs::av26_concurrent_boot_advisory_lock`
spawns 10 concurrent boots against a fresh DB, asserts every one
returns `Ok(())`, and verifies the migration_history table contains
exactly one row per migration script (not 10×N — that would mean
the lock didn't hold). Gated on `CIRIS_PERSIST_TEST_PG_URL`.

### 3.6 Operational / hardening vectors (catalogued in SECURITY_AUDIT_v0.1.2.md §3)

AV-17 through AV-23 were surfaced by the post-v0.1.2 SOTA
gap-analysis pass (Pass 3). The audit document carries the full
prose; the mitigation matrix in §4 below carries the one-line
summary. Briefly:

- **AV-17** — `attempt_index` integer truncation (P0). v0.1.3 caps
  `MAX_ATTEMPT_INDEX = 1024` with `try_into` + typed
  `Error::AttemptIndexOutOfRange`. `overflow-checks = true` on the
  release profile is the defense-in-depth backstop.
- **AV-18** — plaintext Postgres connection (P1). v0.1.3 adds
  optional `tls` feature (`tokio-postgres-rustls`).
- **AV-19** — no graceful shutdown / lost in-flight commits (P1).
  v0.1.3 adds `tokio::signal::ctrl_c` + drain protocol.
- **AV-20** — no `statement_timeout` (P2). Track for v0.2.x.
- **AV-21** — no per-agent rate limiting (P2). Track for v0.2.x;
  PoB §5.6 acceptance-policy adjacent.
- **AV-22** — no clock-skew validation (P2). Track for v0.2.x.
- **AV-23** — `consent_timestamp` range unconstrained (P3). Track.

---

## 4. Mitigation Matrix

| AV | Attack | Primary Mitigation (v0.1.1) | Secondary | Status | Fix Tracker |
|---|---|---|---|---|---|
| AV-1 | Forged trace from attacker key | Public-key directory lookup | N_eff drift detection (lens-side) | ✓ Mitigated | — |
| AV-2 | Forged trace from compromised key | (out of scope at persistence layer) | Audit anchor + Phase 2 peer-replicate | ⚠ Phase 2 closes | FSD §4.5 |
| AV-3 | Replay of legitimate batch | Idempotency on dedup key | TLS at edge | ✓ Mitigated | — |
| AV-4 | Canonicalization mismatch | Byte-exact parity tests + pluggable canonicalizer + `WireDateTime` preserves wire bytes verbatim through canonicalization | Ed25519 collision resistance | **✓ Mitigated v0.1.8** (timestamp closed; float drift residual untriggered, tracked) | — |
| AV-5 | Schema-version flood (mem leak) | `Cow<'static, str>` (no leak) | (deploy-edge rate limit) | **✓ Mitigated v0.1.2** | — |
| AV-6 | JSON-bomb amplification | `MAX_DATA_DEPTH=32` walker | Bounded queue + typed envelope | **✓ Mitigated v0.1.2** | — |
| AV-7 | Body-size flood | `DefaultBodyLimit::max(8 MiB)` | Deploy-edge proxy | **✓ Mitigated v0.1.2** | — |
| AV-8 | Queue saturation | 429 + Retry-After | Single-consumer transaction discipline | ✓ Mitigated | — |
| AV-9 | Dedup-key collision across agents | `agent_id_hash` in UNIQUE index + ON CONFLICT target | trace_id "globally unique per agent" convention | **✓ Mitigated v0.1.2** | — |
| AV-10 | Audit anchor injection | (anchor not part of dedup key) | Phase 2 peer-replicate validates chain | ⚠ Phase 2 closes | FSD §4.5 |
| AV-11 | Public-key re-registration | First-write-wins (`ON CONFLICT DO NOTHING`) + lens-canonical `revoked_at`/`revoked_reason`/`added_by` audit columns | Manual UPDATE for legitimate rotation | ⚠ No explicit rotation API | v0.2.x |
| AV-12 | Schema-version downgrade | Strict allowlist | Per-version payload gates | ✓ Mitigated | track at version bump |
| AV-13 | JSONB injection | Parameterized typed binding | — | ✓ Mitigated (Phase 3 follow-up) | Phase 3 audit |
| AV-14 | Scrubber bypass via schema-altering callback | Schema-preservation gates | Python process boundary | ✓ Mitigated | — |
| AV-15 | PII leak via errors | Typed `kind()` tokens at HTTP/PyO3 boundary; verbose form to tracing logs only | — | **✓ Mitigated v0.1.2** | — |
| AV-16 | Side-channel timing | Ed25519 verify_strict constant-time | (no constant-response wrapper) | ⚠ Directory enumeration possible | v0.2.x research |
| AV-17 | Integer truncation on `attempt_index` | Typed `MAX_ATTEMPT_INDEX = 1024` + `try_into` bound | `overflow-checks = true` on release profile (defense in depth) | **✓ Mitigated v0.1.3** | — |
| AV-18 | Plaintext Postgres connection | Optional `tls` feature — `tokio-postgres-rustls` | `sslmode=verify-full` via DSN | **✓ Mitigated v0.1.3** | — |
| AV-19 | No graceful shutdown / lost in-flight commits | `tokio::signal::ctrl_c` + drain protocol; producer close → persister drains → exit | Journal preserves bytes-on-failure (FSD §3.4 #2) | **✓ Mitigated v0.1.3** | — |
| AV-20 | No statement_timeout on Postgres | (deferred) | Pool size limits | ⚠ Track | v0.2.x |
| AV-21 | No per-agent rate limiting | (deferred) | Shared-queue 429 backpressure | ⚠ Track; PoB §5.6 acceptance policy adjacent | v0.2.x |
| AV-22 | No clock-skew validation on incoming timestamps | (deferred) | Retention-window absorbs out-of-window data | ⚠ Track | v0.2.x |
| AV-23 | `consent_timestamp` range unconstrained | (deferred) | Schema-required-or-422 gate (TRACE_WIRE_FORMAT.md §1) | ⚠ Track | v0.2.x |
| AV-24 | Lens-scrub bypass / forgery | UNCONDITIONAL signed scrub envelope (FSD §3.3 step 3.5; §3.4 robustness primitive #7) — every component, every level, key never null. `original_content_hash + scrub_signature + scrub_key_id + scrub_timestamp` columns proof the deployment's handling. | Single-key principle — agent uses its existing wire-format §8 key; no separate scrub key to compromise | **✓ Mitigated v0.1.3** | — |
| AV-25 | Scrub-key compromise | Hardware-backed `ciris-keyring` (TPM / Secure Enclave / StrongBox / DPAPI) — seed never leaves the keyring; never crosses the FFI boundary | `SoftwareSigner` fallback for hardware-less deployments (named residual) | ✓ Mitigated where hardware available; ⚠ residual on software-fallback | CIRISVerify hardware-attestation tier governs |
| AV-26 | Multi-worker migration race | Session-scoped `pg_advisory_lock(0x6369_7269_7370_7372)` on dedicated single-use connection in `run_migrations()` — workers serialize on cold-boot, lock auto-releases on session close (incl. panic) | `Error::Migration { sqlstate, detail }` surfaces SQLSTATE for lens-side retry policy | **✓ Mitigated v0.1.5** | — |
| AV-27 | Identity churn via ephemeral keyring storage | Boot-time check via authoritative `HardwareSigner::storage_descriptor()` (ciris-keyring v1.8.0). Typed dispatch: `SoftwareFile` ⇒ ephemeral-path heuristic; `SoftwareOsKeyring{User}` ⇒ logout-bound warn; `InMemory` ⇒ hard warn. `Engine.keyring_path()` + `Engine.keyring_storage_kind()` expose authoritative path / classifier for `/health` | Suppression via `CIRIS_PERSIST_KEYRING_PATH_OK=1` after operator audit; `INTEGRATION_LENS.md §11.5` deployment template guidance | **✓ Mitigated v0.1.7 (predicted) / v0.1.9 (authoritative via upstream trait method)** | — |

---

## 5. Security Levels by Deployment Tier

| Tier | Backend | FFI | Threat Model |
|---|---|---|---|
| **Server-class lens** (production) | Postgres + TimescaleDB | PyO3 from FastAPI | Full §3 model applies. TLS at edge required. |
| **Standalone Rust server** (Phase 1.1) | Postgres + TimescaleDB | axum native | Same as above; PyO3 attack surface (callback round-trip) absent. |
| **Pi-class sovereign** (Phase 2.3) | SQLite (bundled) + redb | native bin or PyO3 | Reduced attack surface — typically not internet-exposed. Reticulum transport (Phase 2.3) has its own threat model. |
| **iOS bundled** (Phase 2.2) | SQLite (bundled) | swift-bridge | Apple's app sandbox + secure enclave key storage. Threat model dominated by upstream agent's CIRISVerify hardware-attestation tier. |
| **MCU no_std** (Phase 3 stretch) | none — verify-only | reticulum (when no_std) | Out of HTTP-ingest scope; verify-only relay. |

Critical invariant: **all tiers run the same Backend trait, same
canonicalizer, same scrub gates**. A finding in one tier's
implementation is presumed to apply to the same surface in others
unless explicitly excepted.

---

## 6. Security Assumptions

The system depends on these assumptions; if violated, the threat
model breaks.

1. **Lens deployment hardware integrity**: the host running the
   ingest service is not compromised at root. Postgres, redb
   journal, and process memory are trusted.
2. **TLS at the deployment edge**: the lens fronts CIRISPersist
   with HTTPS termination (nginx, ALB, etc.). Plaintext HTTP
   exposes the agent's traffic to MITM (covered by AV-3 / AV-4
   but assumes TLS-or-not-our-problem).
3. **Public-key directory write authorization**: only authorized
   lens operators can call `register_public_key`. The PyO3 entry
   point is callable from any Python process holding the Engine
   instance — the lens deployment must control which processes
   that is.
4. **Postgres write-quorum**: the database accepts writes
   atomically. Multi-AZ Postgres deployments provide this; single-
   instance deployments inherit Postgres's standard durability
   guarantees.
5. **Clock accuracy**: timestamps in trace bodies and database
   rows are within ~5 minutes of real time. Skew degrades AV-3
   replay-window mitigations (the dedup tuple's `ts` becomes
   ambiguous).
6. **Rust runtime memory safety**: no `unsafe` blocks in
   ciris-persist; transitive deps' `unsafe` is constrained by
   their own audits. `cargo audit` clean across 299 deps as of
   v0.1.1.
7. **Wire-format spec stability**: agents and lens agree on
   TRACE_WIRE_FORMAT.md §8 canonicalization conventions. Drift
   between the two is the AV-4 attack vector.

---

## 7. Fail-Secure Degradation

All failures degrade to MORE restrictive modes, never less. This
is mission constraint MISSION.md §3 anti-pattern #2 ("verify
before persist") + anti-pattern #7 ("never silent drop") made
operational.

| Failure | Behavior | Rationale |
|---|---|---|
| Schema parse failure | HTTP 422; zero rows persisted | Malformed input cannot enter the corpus |
| Schema-version unsupported | HTTP 422 | Out-of-allowlist versions never deserialize past the gate |
| Signature verification failure | HTTP 422; zero rows persisted | Unverified bytes never persist (MISSION.md §3 anti-pattern #2) |
| Unknown signing key | HTTP 422 | Cannot verify → cannot persist |
| Scrubber rejection (schema-altering output) | HTTP 422; zero rows | Schema-altering scrubber output is a contract violation |
| Scrubber rejection (external error) | HTTP 500 | Scrubber bug; ops investigates |
| Postgres unreachable | redb journal append; HTTP 200 (queued for replay) | Outage tolerance per FSD §3.4 #2 |
| Journal append failure | HTTP 500; logged with severity error | Last-line-of-defense exhaustion |
| Queue full | HTTP 429 + Retry-After: 1 | Backpressure honest; agent retries (TRACE_WIRE_FORMAT.md §1) |
| Persister task panicked | HTTP 503 + Retry-After: 5 | Lens shutdown / restart pending |
| Replay handler error during startup | Replay halts; remaining entries stay journaled | Order preserved across restarts |

Critical invariant: **`signature_verified=false` rows do not
exist in the schema.** Decomposition asserts true unconditionally;
unverified bytes never reach the row constructor. Storing
unverified rows would corrupt the corpus PoB §2.4 measurement.

---

## 8. Residual Risks

Risks CIRISPersist mitigates but cannot fully eliminate.

1. **Compromised agent signing key** (AV-2). The persistence layer
   accepts forged-but-correctly-signed bytes. Closure: agent-side
   key storage hardening (CIRISVerify's threat model is
   authoritative); Phase 2 peer-replicate audit-chain validation;
   federation N_eff drift detection over time.

2. **Quantum compromise of Ed25519**. Current quantum compute
   cannot break Ed25519, but Shor's algorithm on a sufficiently
   large quantum computer would. Closure: Phase 2+ ML-DSA-65
   hybrid signatures per PoB §6 — the wire format §8
   canonicalization stays the same; the signature field becomes
   a hybrid `Ed25519 ‖ ML-DSA-65` form.

3. **AV-5 schema-version flood** (v0.1.1 exposure, fix in
   v0.1.2). Until the `Box::leak` path is removed, the lens is
   memory-leaking on malformed input. Mitigate today with
   deployment-edge rate limiting; hot-fix landing.

4. **AV-6 / AV-7 unbounded body / depth** (v0.1.1 exposure, fix
   in v0.1.2). DefaultBodyLimit + recursion-depth guard land in
   the same hot-fix.

5. **AV-9 cross-agent dedup-key collision** (v0.1.1 design point,
   fix in v0.1.x). Extending the dedup key to include
   `agent_id_hash` is a migration; track for v0.1.3.

6. **Float / timestamp canonicalization drift** (AV-4 residual).
   Track production fixtures for any divergence; the parity test
   suite catches what we know about; unknown unknowns are
   exposure.

7. **Public-key rotation under attack** (AV-11). Manual UPDATE
   today; v0.2.x adds explicit `rotate_public_key(rotation_proof)`
   API.

8. **Side-channel timing leakage of directory membership** (AV-16).
   Low-impact (key ids are public), but trackable.

9. **Postgres compromise**. Out of CIRISPersist's protection
   scope; deployment infrastructure concern.

10. **All federation peers compromised simultaneously** (PoB §5.1
    residual). Per Accord NEW-04, no detector is complete. PoB's
    response is topological cost-asymmetry over time, not pointwise
    decidability — a property that cannot be achieved at the
    persistence layer alone.

---

## 9. v0.1.2 Threat Posture Summary

```
v0.1.1 INTEGRATION-BLOCKING EXPOSURES → all closed in v0.1.2
  ✓ AV-5  schema-version flood mem leak  (Cow<'static, str>)
  ✓ AV-6  data-blob recursion uncapped   (MAX_DATA_DEPTH=32 walker)
  ✓ AV-7  no crate-level body size limit (DefaultBodyLimit::max(8 MiB))
  ✓ AV-9  dedup key cross-agent collision (agent_id_hash in UNIQUE index)
  ✓ AV-15 error messages leaking verbatim (kind() tokens at FFI boundary)

PHASE-2-CLOSES (architecturally deferred)
  ⚠ AV-2  stolen-key forgery (peer-replicate audit chain)
  ⚠ AV-10 audit anchor capture without verification

V0.2.X TRACK
  ⚠ AV-11 explicit rotate_public_key(rotation_proof) API
  ⚠ AV-16 side-channel timing on key-directory enumeration

DESIGN-DECISIONS-PER-MISSION (intentional, not defects)
  ✓ AV-1  identity gating via public-key directory
  ✓ AV-3  idempotency via dedup-key conflict
  ✓ AV-4  canonicalization parity tests + pluggable trait
  ✓ AV-8  429 backpressure honest
  ✓ AV-12 strict schema-version allowlist
  ✓ AV-13 parameterized binding only
  ✓ AV-14 scrubber-output schema gates

CARGO AUDIT
  ✓ 0 vulnerabilities across 299 dependencies as of 2026-05-01
```

All five integration-blocking exposures from the v0.1.1 baseline
are closed. Phase 2 (audit-chain peer-replicate) closes AV-2 and
AV-10 architecturally; v0.2.x track has explicit-rotation API and
timing-oracle hardening. Lens integration can proceed against
v0.1.2 with no known integration-blocker.

---

## 10. Update cadence

This document is updated:
- On every minor version (v0.1.x → v0.2.0): comprehensive review.
- On every published security advisory affecting deps: addendum
  in §3 + cargo-audit re-run.
- On every Phase boundary (Phase 1 → 2 → 3): new attack vectors
  added for the new trait surfaces.
- On every wire-format schema-version bump: AV-4 / AV-12 review.

Last updated: 2026-05-01 (v0.1.2 — AV-5/6/7/9/15 closed; Path B
schema reconciliation against the lens-canonical accord_public_keys
shape complete).
