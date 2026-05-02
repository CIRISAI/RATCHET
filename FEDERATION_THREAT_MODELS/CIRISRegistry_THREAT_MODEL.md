# CIRISRegistry Threat Model

**Status:** v1.3 complete (FSD-001 v1.1.0; FSD-002 self-custody shipped 2026-04-30;
project namespace + ciris-crypto v1.8.0 alignment + 7-phase hardening waterfall
shipped 2026-05-01). All v1.3 high-priority in-app gaps closed.
Updated each minor release.
**Audience:** CIRISVerify integrators, CIRISPortal operators, federation peers, security reviewers.
**Companion:** [`FSD/FSD-001_CIRISREGISTRY_PROTOCOL.md`](../FSD/FSD-001_CIRISREGISTRY_PROTOCOL.md), [`docs/MANIFEST_VALIDATION_API.md`](MANIFEST_VALIDATION_API.md), [`CLAUDE.md`](../CLAUDE.md).
**Inspired by:** [`CIRISPersist/docs/THREAT_MODEL.md`](https://github.com/CIRISAI/CIRISPersist/blob/main/docs/THREAT_MODEL.md), [`CIRISVerify/docs/THREAT_MODEL.md`](https://github.com/CIRISAI/CIRISVerify/blob/main/docs/THREAT_MODEL.md) — the structural templates.

---

## 1. Scope

### What CIRISRegistry Protects

CIRISRegistry is the trust backbone of the CIRIS ecosystem. It protects:

- **Authoritative agent-build identity**: every registered build (`AgentRecord`)
  is keyed by SHA-256 of canonical source files; lookups are deterministic.
  Downstream CIRISVerify treats unknown hashes as `UNLICENSED_COMMUNITY`
  (fail-secure per FSD-001 §Design Principles). Code: `db/agents.rs`.
- **Authoritative partner-license state**: `PartnerRecord` carries
  `capabilities_granted`, `capabilities_denied`, `max_autonomy_tier`, and
  `requires_supervisor`. The effective capability set is `agent_caps ∩
  granted - denied` — license forgery cannot escalate beyond what the partner
  was granted by a steward. Code: `db/partners.rs`.
- **Real-time revocation propagation**: `RevocationEntry` rows are written
  atomically; `GetRevocationList` supports full + delta queries. Multi-source
  validation (DNS US + DNS EU + HTTPS API) is the verifier's responsibility,
  but the registry signs every revocation list with the steward key so any one
  source's response is independently verifiable. Code: `db/revocations.rs`,
  `services/registry.rs`.
- **Hybrid-signed records**: every authoritative response (lookup, revocation
  list, offline package, build attestation) carries an Ed25519 + ML-DSA-65
  signature over canonical bytes. CIRISVerify rejects classical-only or
  invalid responses. Code: `crypto/mod.rs::HybridCrypto::sign`.
- **Custodied key generation**: `PortalService.GenerateKeyPair` mints an
  Ed25519 + ML-DSA-65 keypair, stores both public keys + fingerprints, and
  returns the Ed25519 private key over the wire **once**. The private key
  is never persisted on the registry side. Code: `services/portal.rs`
  (`generate_key_pair`).
- **Self-custody key registration (FSD-002)**: agent generates the keypair
  locally and registers only the public key. Proof-of-possession is enforced
  via challenge-response: registry issues a 32-byte nonce (5-min TTL,
  single-use), agent signs it with the private key, registry verifies with
  Ed25519 before accepting. Activation is a second challenge-response gate.
  Code: `services/portal.rs` (`get_registration_challenge`,
  `register_public_key`, `activate_self_custody_key`).
- **Public-key uniqueness across orgs**: `partner_keys.public_key_hash` is
  `UNIQUE` (migration 020). Two orgs cannot register the same Ed25519 public
  key — the foundation for Sybil resistance at the key layer. The
  `create_key` regression (custodied path missing `public_key_hash`) was
  closed in commit `4baec9b`.
- **Audit trail of admin / portal actions**: every state-changing RPC writes
  an `audit_log` row with actor user, actor org, action type, target, and
  description. Code: `db/audit.rs`, called from each service handler.
- **Build attestation (SLSA-compatible)**: `RegisterBuildAttestation` records
  `builder_id` + provenance blob; `GetBuildAttestation` returns it for
  CIRISVerify to cross-check against the build's expected builder.
- **Offline verification packages**: `GetOfflinePackage` returns a signed
  snapshot bundle (agents + partners + revocations + Merkle root) so
  CIRISVerify deployments can operate up to 72 hours without network
  reachability. Snapshots are hybrid-signed.
- **Emergency lockdown**: `SetEmergencyShutdown` flips a single global flag
  that gates all professional capabilities at the verifier; intended for
  active-incident response. Cleared by a separate admin RPC.

### What CIRISRegistry Does NOT Protect

The registry is the trust backbone, not the entire trust chain. Out of scope:

- **Agent runtime behavior**: once verified, an agent's behavior under its
  authorized capabilities is governed by the agent's own DMA/conscience
  pipeline (CIRISAgent's H3ERE). The registry cannot detect a properly
  licensed agent acting maliciously within its grants.
- **LLM output content**: CIRISProxy + the lens scoring layer (CIRISLens)
  are the relevant trust boundaries for inference content. Registry stores
  no inference data and has no behavioral telemetry.
- **Compromised steward / signing-key holder**: if CIRIS L3C's steward role
  is compromised, the steward can sign valid-looking malicious records.
  Mitigation is at the steward-key storage tier (HSM/Vault) and at the
  governance layer (multi-party authorization for license issuance), not
  at the registry RPC layer.
- **PostgreSQL backend integrity**: the persistence layer trusts row reads.
  Deployment must protect DB credentials, restrict network access, and (if
  required) enable PostgreSQL row-level security. A DBA with `DELETE`
  privilege on `audit_log` can erase evidence; mitigation is operational.
- **HSM / Vault availability**: when `storage_mode=vault`, the registry
  depends on Vault Transit + KV reachability for signing operations.
  Vault outage stops the registry from issuing new signed responses;
  cached/precomputed responses keep serving. HSM mode (PKCS#11) is **not
  yet implemented** (`crypto/mod.rs:46`).
- **DNS / TLS infrastructure compromise**: multi-source consensus mitigates
  one DNS source being compromised; it does not survive simultaneous
  compromise of all registrars or a CA breach. Verifier-side responsibility.
- **Browser / Portal session security**: CIRISPortal's threat model governs
  session cookies, CSRF, XSS, and OAuth flows. Registry trusts the JWT it
  receives.
- **Stripe / billing integrity**: CIRISPortal owns billing entirely. The
  registry has no Stripe credentials and no billing tables.
- **Network egress from agents**: CIRISLens / CIRISProxy concerns. The
  registry does not initiate connections to agents.
- **Cross-region public-key uniqueness**: the US and EU registry deployments
  are distinct authorities. A pubkey registered in US is not blocked from
  re-registration in EU. Federation peers must reconcile across regions
  themselves.

---

## 2. Adversary Model

### Adversary Capabilities

The adversary is assumed to have:

- **Full source-code access** (AGPL-3.0; public on GitHub including all
  migrations, proto definitions, deny.toml, and CI artifacts).
- **Network MITM** between any client and the registry. TLS termination is
  the deployment's responsibility (nginx / ALB / Envoy in front of the
  registry); plaintext gRPC/HTTP exposes wire traffic.
- **Ability to mint arbitrary Ed25519 + ML-DSA-65 keypairs** and submit
  whatever bytes they choose to any unauthenticated endpoint.
- **Network access to public `RegistryService` RPCs** — all 13 lookup,
  capability, and revocation methods are unauthenticated by design
  (`middleware/auth.rs::classify_path`).
- **Network access to HTTP verification endpoints**: `/v1/builds/{version}`,
  `/v1/builds/hash/{hash}`, `/v1/verify/binary-manifest/{version}`,
  `/v1/verify/function-manifest/{version}/{target}`, `/v1/verify/key/{fp}`,
  `/v1/revocation/{id}`, `/v1/steward-key`, and `/v1/integrity/*` device
  attestation endpoints — all GET routes are unauthenticated.
- **Replay capability**: any captured response can be re-submitted at any
  time. Registry responses carry hybrid signatures over canonical bytes;
  responses themselves do not include challenge nonces (downstream
  verifier-side anti-replay is CIRISVerify's responsibility — see
  CIRISVerify AV-4).
- **Side-channel observation**: response timing, HTTP status codes, gRPC
  `Status` codes, error message bodies.
- **Classical compute resources** sufficient for offline brute-force of
  short secrets (NOT for breaking Ed25519 within current physics).
- **Ability to register an organization through CIRISPortal** by paying
  the tier's issuance fee + bond ($0.50/$1.00 community minimum, FSD-001
  §Billing). Organization registration is rate-limited at the Portal layer
  but not at the registry.
- **Ability to operate a CIRIS agent** in sovereign mode (no registration
  required) and submit traces / participate in federation primitives. The
  registry does not gate this.
- **Ability to read public CI artifacts**: every test output, every
  published Docker image, the build attestation chain.

### Adversary Limitations

The adversary is assumed NOT to have:

- **Ability to break Ed25519** in polynomial classical time (PoB §6
  acknowledges quantum risk; ML-DSA-65 hybrid is the closure).
- **Ability to break ML-DSA-65** (NIST FIPS 204; NIST-standardized
  post-quantum lattice signature). Both must fall for hybrid signature
  forgery.
- **Simultaneous compromise of all DNS registrars** serving the registry's
  TXT / A records. US and EU deployments use different registrars by
  policy.
- **HSM / Vault root-key compromise** when the deployment runs in
  `storage_mode=vault`. Software-only `file` and `memory` modes have weaker
  guarantees; see §5.
- **PostgreSQL root access** at the deployment. Backend trust is an
  assumption; if the DB is owned, the model breaks.
- **CIRIS L3C steward-key holder compromise**. The steward is the
  root-of-trust for license issuance; compromise lets the attacker mint
  valid-looking PROFESSIONAL_MEDICAL etc. licenses. Out of scope at the
  registry layer; mitigation is at the governance + key-storage tier.
- **JWT secret leak** (`AuthSettings.jwt_secret`, HS256 with shared
  secret). If leaked, attacker mints arbitrary admin/portal tokens.
  Tracked as residual risk; v1.2.x roadmap moves to asymmetric signing.
- **`REGISTRY_ADMIN_TOKEN` leak** (env var bearer used by CI to register
  binary/function manifests). If leaked, attacker registers arbitrary
  manifests under the steward signature. Same tracker as JWT.
- **Physical access** to deployment hardware. Hardware-level attacks on
  HSM / Vault / disk are out of scope; CIRISVerify §5.1 covers the
  hardware-attestation tier for verifier-side devices.
- **Clock skew** beyond ~5 minutes. Affects challenge TTLs (5-min FSD-002
  windows), JWT `exp` validation, license expiry, and revocation timestamp
  ordering.

---

## 3. Attack Vectors

### 3.1 Forgery — adversary wants forged records counted as authoritative

#### AV-1: Build registration with project-name collision

**Attack**: An attacker (or a sibling CIRIS primitive) tries to register a
build under a `version` string that already exists for a different project.
Pre-v1.2, the schema had no `project` discriminator on `builds`,
`binary_manifests`, or `function_manifests`; the unique constraints
effectively reserved the entire version namespace for `ciris-agent`.
CIRISPersist's `project=ciris-persist` registration of v0.1.7 was rejected
because v0.1.7 collided with agent's namespace.

**Mitigation in v1.2 (shipped, commit `254a89e`)**: Migration 021 adds
`project TEXT NOT NULL DEFAULT 'ciris-agent'` to all three tables. Unique
constraints now lead with `project`:
- `builds (project, version, includes_modules) UNIQUE`
- `binary_manifests (project, version) UNIQUE`
- `function_manifests` PK rebuilt as `(project, binary_version, target)`
- `builds.build_hash UNIQUE` preserved (SHA-256 globally unique by
  construction)

Proto v1.4.0 adds `project` to `BuildRecord` (#8) and `GetBuildRequest`
(#4). HTTP request bodies (`RegisterBinaryManifestRequest`,
`RegisterFunctionManifestRequest`) accept `project`. GET endpoints accept
`?project=` query param. Empty/missing always defaults to `ciris-agent`
(backwards compat preserved for CIRISVerify and existing tooling).

Project name validator (`rust-registry/src/validation.rs::validate_project_name`)
enforces `^[a-z][a-z0-9-]{0,63}$` at every registration entry point.
8 unit tests cover canonical names, leading-digit rejection, underscore /
dot rejection, length bounds.

**Post-fix attack vector (NEW)**: an attacker with `REGISTRY_ADMIN_TOKEN`
or an admin JWT can submit a build with `project=ciris-agent` they did not
author, attempting to displace the canonical agent build. The admin auth
boundary is the only gate; the project name does not bypass it.
**Recommended for v1.3.x**: per-project signing-key binding (each
registered project declares a builder pubkey at first registration; later
submissions must be signed by that key). Couples to AV-13 (token
rotation) and AV-26 (inbound hybrid-sig verification).

**Residual**: project-name impersonation by holders of the admin token is
not gated beyond authentication. Tracked for v1.3.x.

#### AV-2: Self-custody key registration via challenge replay or theft

**Attack**: Attacker captures a `GetRegistrationChallenge` response
(32-byte nonce, valid 5 minutes) issued to an honest org. They sign it
with their own Ed25519 key and call `RegisterPublicKey` with their
attacker-controlled public key + their signature, claiming registration
under the victim org.

**Mitigation in v1.1**: the registration challenge is bound to `org_id`,
not to a specific public key. Once issued, any caller who can sign the
nonce with the public key they're submitting will pass — that's the
intended PoP semantics. **The protection is that the challenge is
single-use**: `get_and_remove_registration_challenge` (`db/keys.rs:431`)
issues a `DELETE ... WHERE org_id = $1 AND expires_at > NOW() RETURNING
challenge` — atomic remove-on-consume. Concurrent callers race for the
same row; one wins, the others get `Invalid or expired challenge`.

**Secondary**: the 5-min TTL bounds the window. The activation flow is
a second challenge (`activation_challenges` table, separate per `key_id`)
that the registered key must sign before the key becomes ACTIVE — a
stolen-challenge-attack-without-key-control fails activation.

**Residual**: an attacker with **both** the captured registration challenge
**and** the org's request flow (e.g., MITM the gRPC call before TLS
termination) can substitute their public key for the legitimate one in
the `RegisterPublicKey` call. TLS at the deployment edge is the only
mitigation. Per-org concurrent-challenge limit is not enforced; an
attacker can request many challenges to keep the window perpetually open
— bounded by the global per-org rate at the Portal layer (not the
registry). Track as v1.2.x: rate limit `GetRegistrationChallenge` per
`org_id`.

#### AV-3: Public-key cross-org reuse (Sybil via shared pubkey)

**Attack**: Attacker registers the same Ed25519 public key under multiple
org_ids — one Sybil, multiple attested identities. Defeats per-key
metering, allows weight farming.

**Mitigation in v1.1**: `partner_keys.public_key_hash UNIQUE` constraint
added in migration 020. Both registration paths populate the column:
self-custody via `create_self_custody_key` (`db/keys.rs:507`); custodied
via `create_key` (`db/keys.rs:153`) after the regression fix in commit
`4baec9b`. The `register_public_key` handler additionally pre-checks via
`public_key_exists` (`portal.rs:2522`) and returns `AlreadyExists` before
attempting INSERT — clearer error than the bare UNIQUE violation.

**Secondary**: cross-org behavioral signal at the lens layer. A pubkey
producing traces under one identity then another would surface in
PoB §2.4 N_eff drift detection. Out of registry scope; documented for
federation peers.

**Residual**:
- **Cross-region**: US and EU registry deployments do not share a
  `public_key_hash` table. Same pubkey can register in both regions; by
  design, they are distinct authorities. Federation peers reconcile.
- **Custodied → self-custody race**: a key generated via
  `GenerateKeyPair` and a key registered via `RegisterPublicKey` both
  insert into `partner_keys`. The `public_key_hash` UNIQUE catches both;
  verify by integration test (TODO: add to `tests/`).

#### AV-4: Forged build attestation (SLSA blob)

**Attack**: Attacker calls `RegisterBuildAttestation` with a forged
provenance blob pointing to a real `builder_id` (e.g.,
`github.com/cirisai/cirisagent/.github/workflows/build.yml`). The
registry stores the blob; CIRISVerify retrieves it and treats it as
authoritative.

**Mitigation in v1.1**: **Partial.** The registry stores attestations
under admin auth (RegistryAdminService), which gates who can submit. The
attestation blob itself is **not signature-verified at registration**
(`db/build_attestations.rs` writes the blob as-is). CIRISVerify is
expected to verify the blob's own signature against the builder's
public key out-of-band, but the registry does not enforce this.

**Recommended for v1.2.x**: add Sigstore-style verification at
registration — fetch the builder's public key from a trusted source
(e.g., GitHub's OIDC issuer JWKS) and verify the attestation's
`dsseEnvelope` signature before storing. Reject on failure.

**Residual**: until verification at registration lands, an admin-token
holder can register attestations that point to legitimate builders but
contain crafted artifact hashes. Trust derives from the admin-auth
boundary, not from the attestation itself.

#### AV-5: Custodied private key disclosed in transit

**Attack**: `PortalService.GenerateKeyPair` returns the Ed25519 private
key (`ed25519_private_key` bytes) in the gRPC response — the only time
the registry transmits a private key. An MITM with access to the
plaintext gRPC stream captures the key.

**Mitigation in v1.1**: the registry does **not** persist the private
key (`portal.rs:611` extracts it for the response only; nothing in
`db/keys.rs` stores it). The wire-level protection is **TLS at the
deployment edge** — gRPC over plaintext exposes the key.

The registry-side log line emits only `org_id` (`portal.rs:603`), not
the key; the audit log entry includes only the key_id and fingerprint
(`portal.rs:638`), not the private key bytes. Verify with `grep -rn
"ed25519_private" rust-registry/src/`: only the response field, no log
or DB write.

**Secondary**: self-custody mode (FSD-002) avoids this entirely — the
private key never enters the registry's address space. Encourage
agents to prefer `RegisterPublicKey` over `GenerateKeyPair` (per FSD-002
§Migration Path).

**Residual**: any deployment running plaintext gRPC has this exposure
on every `GenerateKeyPair` call. **Recommended**: hard-fail server boot
when `storage_mode != memory` and TLS is not configured. Track as
v1.2.x deployment-validation pass.

#### AV-6: Function-manifest content substitution by admin-token holder

**Attack**: A holder of `REGISTRY_ADMIN_TOKEN` (the static env-var bearer
used by CI) submits a function manifest for `binary_version=X,
target=Y` — one that already exists, with substituted `binary_hash` or
`functions` map. The `ON CONFLICT (binary_version, target) DO UPDATE`
clause (`db/function_manifests.rs:170`) overwrites the existing row.

**Mitigation in v1.1**: **Strong on the signing path.** Per commit
`025c78b` ("Fix: Always sign function manifests server-side"), the
registry signs every manifest with its steward key, regardless of
whether the uploader provided a signature. Downstream CIRISVerify
verifies against the registry's public steward key. So even a
substituted manifest carries a valid registry signature — but the
substitution itself is what the threat is about, and the auth boundary
is the only protection.

The protection is **REGISTRY_ADMIN_TOKEN possession** — a static env-var
bearer with no rotation, no scope, no audit-log identity (the audit
entry would record `registered_by="ci_push"`, not the token holder's
identity). See AV-13.

**Recommended for v1.2.x**:
- Replace `REGISTRY_ADMIN_TOKEN` with short-lived JWTs scoped to manifest
  registration (e.g., `scope=manifest:register`).
- Or: require the uploader to provide a signature from a registered
  builder key, verify against `builder_id`, then store both the
  builder's signature AND the registry's countersignature. Closes the
  trust to a per-builder identity instead of a shared token.

**Residual**: until the token is replaced, any compromise of the env
var lets the attacker overwrite any manifest. The `ON CONFLICT DO
UPDATE` clause is the structural hazard — consider switching to
`ON CONFLICT DO NOTHING` for the immutable-by-default case and adding
an explicit `ReplaceFunctionManifest` admin RPC for legitimate
overwrites with stronger gating.

### 3.2 Denial of Service — adversary wants the registry unable to serve verification

#### AV-7: HTTP body-size flood

**Attack**: Adversary submits arbitrarily large bodies to any HTTP POST
endpoint (`/v1/verify/binary-manifest`, `/v1/verify/function-manifest`,
`/v1/integrity/*`). Memory amplification while the body is buffered.

**Mitigation in v1.1**: `RequestBodyLimitLayer::new(MAX_REQUEST_BODY_SIZE)`
caps every HTTP route at **1 MiB** (`api/http.rs:1421` defines the
constant; `:1474` applies the layer to the entire router). Bodies above
the limit are rejected with HTTP 413 before being read into memory.

**gRPC side**: tonic's default per-message size limit is 4 MiB. The
registry has not raised this. Batch RPCs cap their list lengths
explicitly: `BatchLookupAgents` ≤ 100 hashes, `BatchRegisterAgents` ≤
1000 records, `BatchCreateOrgUsers` ≤ 100 users (FSD-001). These caps
prevent a single message from forcing pathological allocation.

**Residual**:
- The 1 MiB HTTP limit is crate-wide; if a future endpoint legitimately
  needs more (e.g., bulk SLSA attestation), the limit must be raised
  per-route, not globally. Track when adding any large-payload route.
- Deployment-edge proxy (nginx, ALB) typically caps at 1-100 MiB; this
  is defense-in-depth, not the primary guarantee.

#### AV-8: Per-IP rate-limiter bypass via X-Forwarded-For spoofing

**Attack**: Adversary sends requests with a forged `X-Forwarded-For`
header containing a victim's IP, hoping to either (a) attribute the
request to the victim for rate-limit / abuse-tracking purposes or (b)
bypass per-IP rate limits by rotating the spoofed value.

**Mitigation in v1.1**: `rate_limiter::is_trusted_proxy` only honors
`X-Forwarded-For` (and `X-Real-IP`) when the **direct peer connection**
is from a trusted CIDR range (`TRUSTED_PROXY_RANGES`: private RFC1918,
loopback, IPv6 ULA — `rate_limiter.rs:57-64`). Direct internet-facing
clients have their headers ignored; the actual TCP peer IP is used.
Verified at `api/http.rs:949-996`.

**Residual**:
- A misconfigured deployment that places the registry behind a proxy
  not on `TRUSTED_PROXY_RANGES` (e.g., a public IPv4 LB without
  internal addressing) will not honor the LB's `X-Forwarded-For`, so
  every request appears from the LB IP — single bucket exhaustion,
  effectively no per-client rate limiting. Loud-fail recommended:
  add a startup config check and emit a warning when no trusted proxy
  range matches the LB's expected source IP. Track as v1.2.x.
- A trusted proxy that itself does not validate `X-Forwarded-For` from
  upstream lets clients spoof. Deployment concern.

#### AV-9: Mass-query flood saturating PostgreSQL pool

**Attack**: Adversary floods `RegistryService` lookup RPCs (`LookupAgent`,
`BatchLookupAgents`, `LookupPartner`, `GetRevocationList`,
`GetPublicKeys`, `GetOfflinePackage`) — all unauthenticated by design —
saturating the DB connection pool. Honest verifier traffic stalls.

**Mitigation in v1.3 (Phase 2 shipped)**:
- New `middleware/rate_limit.rs` tower `Layer<S>` mirrors the
  `AuthLayer` pattern. Wired in `main.rs` *before* `AuthLayer` so
  denied requests skip JWT decode + tracing + metrics overhead.
- Tier mapping (`classify_tier`):
  - **Bypass**: `HealthCheck`, `GetCapabilities`, `GetMetrics`,
    health/probe/metrics paths, gRPC reflection,
    `RegistryAdminService` and `PortalService` (auth-gated upstream).
  - **Public** (60/min, 600/hr per IP): single-row indexed lookups —
    `LookupAgent`, `LookupPartner`, `GetEmergencyStatus`,
    `GetPublicKeys`, `GetBuildAttestation`, `VerifyDeployment`,
    `GetRevocationList` delta. Default for unknown gRPC paths.
  - **Verify** (5/min, 50/hr per IP): DB-fanout-heavy —
    `BatchLookupAgents` (max 100), `GetOfflinePackage`,
    `GetOfflineDelta`.
- HTTP unauthenticated public GETs (`/v1/steward-key`,
  `/v1/revocation/{id}`, `/v1/verify/binary-manifest/{version}`,
  `/v1/verify/function-manifest/{version}/{target}`,
  `/v1/verify/function-manifests/{version}`, `/v1/builds/{version}`,
  `/v1/builds/hash/{build_hash}`, `/v1/verify/key/{fingerprint}`)
  are routed through a sub-router with `axum::middleware::from_fn(
  rate_limit_public_http)` — same `Tier::Public` bucket as gRPC, so
  an attacker can't multiply effective limits by switching protocols.
- `extract_client_ip` lifted to `rate_limiter::extract_client_ip` and
  reused by both gRPC interceptor and axum middleware. Honors
  `cf-connecting-ip` (Cloudflare), `X-Forwarded-For` /
  `X-Real-IP` only when the direct peer is from `TRUSTED_PROXY_RANGES`
  (AV-8 mitigation preserved).
- Background `tokio::interval(60s)` cleanup ticker calls
  `rate_limiter::cleanup_all()` — closes the architectural smell that
  per-bucket HashMaps would otherwise only sweep when exceeding the
  10k-entry cap, blocking on an O(N) global-mutex sweep.
- 8 new unit tests for rate_limiter (was zero) + 8 for the
  classify_tier mapping. Production posture: 39/39 tests pass.

**Secondary**: deployment-edge rate limiting (nginx, Envoy) remains
recommended as defense-in-depth. CIRISVerify multi-source consensus
tolerates one source slow/unavailable.

**Residual**:
- In-memory globals are per-process, so an N-replica deployment yields
  N× the effective limit per IP. Acceptable today (registry runs
  single-instance per region in production); track distributed
  rate-limiting (Redis-backed `RateLimitStore` trait) for v1.3.x.
- Per-IP bucket only — no per-`org_id` tracking when JWT present.
  Track for v1.3.x.
- PostgreSQL pool-exhaustion circuit breaker: silent stall when pool
  saturated under sustained legitimate load. Separate concern from
  AV-9; track as v1.3.x.

#### AV-10: Emergency-shutdown abuse (admin-level DoS)

**Attack**: An admin (or attacker holding admin JWT) calls
`SetEmergencyShutdown`, flipping a global flag that gates all
professional capabilities at every CIRISVerify deployment. Ecosystem-
wide outage of medical / legal / financial agents.

**Mitigation in v1.1**: requires admin-role JWT
(`middleware/auth.rs::AuthRequirement::Admin`, role==1). Audit log row
written. Cleared by a second admin RPC (`ClearEmergencyShutdown`).

The action is **single-actor**: one admin JWT holder can trigger
ecosystem lockdown. There is no multi-party authorization, no time-
delay, no canary rollout.

**Recommended for v1.2.x**:
- Multi-party authorization (M-of-N admin signatures) for emergency
  shutdown; matches the steward-governance model implied by FSD-001.
- Mandatory webhook fanout to all registered partners on shutdown +
  clear, so operators see the action immediately.
- Time-bounded shutdowns (auto-clear after configured duration unless
  re-affirmed) to prevent forgotten / orphaned shutdowns.
- A `dry_run=true` mode that signals the intent to webhook subscribers
  without flipping the flag.

**Residual**: any single admin JWT compromise → ecosystem outage. AV-12
(JWT secret leak) compounds this.

#### AV-11: Mass-revoke abuse (admin-level, weaponized incident response)

**Attack**: An admin (or attacker with admin JWT) calls `MassRevoke`
with a list of every active partner license. Blast radius: every
CIRISVerify deployment denies professional capabilities for those
partners on next revocation-list refresh.

**Mitigation in v1.1**: same auth gate as AV-10. Audit log row per
revocation. `MassRevoke` requires a `reason_code`
(`RevocationReason` enum) and `severity`
(`RevocationSeverity`); no rate limit on the list size.

**Recommended for v1.2.x**:
- Cap `MassRevoke` list size (e.g., 100 entries per call) — force
  large incidents to be batched, generating multiple audit entries
  with operator-confirmed intent.
- Require multi-party authorization for `severity=SECURITY_CRITICAL`
  mass revocations.
- Webhook fanout to every revoked partner's registered endpoints +
  the steward's incident channel.

**Residual**: same as AV-10 — single admin JWT compromise weaponizes
incident-response tooling.

### 3.3 Authorization Bypass — adversary wants escalated privilege

#### AV-12: JWT forgery via leaked HS256 shared secret

**Attack**: Adversary obtains the registry's `jwt_secret` (env var or
config) — via leaked deployment manifest, exfiltrated container env,
secret-store breach, source-control mistake — and mints valid JWTs with
arbitrary `sub`, `org_id`, `role`, and `exp`.

**Mitigation in v1.1**: **None at the algorithm layer.** `validate_jwt`
hardcodes `Algorithm::HS256` (`middleware/auth.rs:211`) — symmetric
HMAC. Anyone with the secret can both sign and verify. The same secret
is used by CIRISPortal to issue tokens, so it must be present on at
least two services + CI; the attack surface is multi-host.

The only per-request mitigation is `validation.set_issuer(&[expected_issuer])`
plus expiry — neither defends against a holder of the secret.

**Recommended for v1.2.x**:
- Migrate to `Algorithm::EdDSA` (Ed25519) or `Algorithm::RS256`. The
  Portal holds the **private** key for signing; the registry holds only
  the **public** key for verification. Compromise of the registry's
  config no longer enables token minting.
- Short-lived tokens (≤15 min) with refresh-token flow at the Portal.
- JWKS rotation endpoint so key rotation is a config change, not a
  redeploy.

**Residual**: until v1.2.x lands, every `jwt_secret` leak is a
full-trust breach.

#### AV-13: REGISTRY_ADMIN_TOKEN extraction

**Attack**: Adversary extracts `REGISTRY_ADMIN_TOKEN` from the registry
container's environment (debug interface, `/proc/1/environ`, leaked
deployment manifest). Submits arbitrary binary or function manifests via
`POST /v1/verify/binary-manifest` or `POST /v1/verify/function-manifest`.

**Mitigation in v1.1**: **None beyond the bearer check.** The token is
a static env-var string; no rotation, no scope, no per-request audit
identity beyond `registered_by="ci_push"` (`api/http.rs:401`). Both
endpoints accept the token via `Authorization: Bearer <token>`.

The submitted manifest is server-signed (commit `025c78b`), so it
**will** carry a valid registry signature — meaning a forged manifest
under this attack is indistinguishable from a legitimate one to
downstream CIRISVerify.

**Recommended for v1.2.x**: see AV-6 — replace with scoped JWTs minted
per-CI-job with short TTL. Bind the JWT subject to a builder identity
(GitHub OIDC) so the audit log records *who* registered, not just
*that* it was registered.

**Residual**: until replaced, env-var compromise = arbitrary manifest
registration. Operationally: rotate the token on a defined cadence,
restrict access to the env var (sealed secrets, Vault dynamic secrets),
monitor `audit_log` for unexpected registrations.

#### AV-14: Role escalation in JWT (role=1 = SYSTEM_ADMIN)

**Attack**: A non-admin user with a valid JWT mints (or alters) a token
with `role: 1` to access RegistryAdminService. Because role is a JWT
claim signed by `jwt_secret`, this collapses into AV-12 if the secret
leaks. Without the secret, the attacker cannot produce a validly-signed
token with role=1.

**Mitigation in v1.1**: HMAC signature on the entire claims set
(including `role`) prevents tampering. The auth layer checks
`claims.role != ROLE_SYSTEM_ADMIN` (`middleware/auth.rs:161`) and
returns HTTP 403 on mismatch.

**v1.3 Phase 6 (W2) — PortalService god-mode gating**: pre-Phase-6
the auth-middleware SYSTEM_ADMIN check ONLY applied to
`RegistryAdminService`. PortalService methods that are de-facto
cross-org god-mode (`create_organization`, `list_organizations`,
`batch_create_organizations`, `create_system_user`, `get_system_user`,
`list_system_users`, `update_system_user`, `lookup_system_user_by_o_auth`,
`link_system_user_o_auth`, `list_system_user_o_auth_identities`,
`create_user`, `lookup_user_by_o_auth`, `upgrade_to_partner`)
accepted any authenticated JWT. Phase 6 added explicit
`authorize_system_admin(self.db.pool(), &claims)` calls to all of
these handlers. `create_licensee_organization` requires either
SYSTEM_ADMIN or OrgAdmin on the parent (handled by `authorize_org_access`
which has the SYSTEM_ADMIN bypass). User-OAuth linking (`link_user_o_auth`,
`list_user_o_auth_identities`) allows self (`claims.sub == req.user_id`)
or SYSTEM_ADMIN.

**Sole role is a binary flag**: there is no fine-grained authorization
within the admin tier. An admin can do everything (mass-revoke,
emergency shutdown, signing-key rotation, build registration). No
separation of duties between, say, build-attestation registration and
license issuance.

**Recommended for v1.2.x**:
- Multiple admin sub-roles (incident-responder, build-registrar,
  steward-key-operator, partner-onboarding) via a `scopes` claim.
- The most damaging actions (steward-key rotation, mass-revoke,
  emergency shutdown) require a `steward` scope that is held by
  fewer principals than the general admin role.

**Residual**: any admin JWT today is a god-token. Couples to AV-12.

#### AV-15: PortalService cross-org access (HIGH PRIORITY)

**Attack**: A user with a valid JWT for `org_id = A` calls a
PortalService RPC passing `org_id = B` in the request body. The handler
operates on org B's data — reads users, generates keys, lists audit
logs — without checking that the caller's JWT actually authorizes them
for org B.

**Mitigation in v1.1**: **None at the handler layer.** Verified by
`grep -n "claims\|extensions" rust-registry/src/services/portal.rs`:
**zero matches**. PortalService handlers do not read JWT claims; they
trust `req.org_id` from the request body. The auth middleware confirms
the JWT is *valid* but does not enforce that `claims.org_id ==
req.org_id`.

This affects every PortalService method that takes `org_id`:
`GetOrganization`, `UpdateOrganization`, `CreateOrgUser`,
`GetOrgUserByEmail`, `UpdateOrgUser`, `ListOrgUsers`,
`BatchCreateOrgUsers`, `GenerateKeyPair`, `ListKeys`, `ActivateKey`,
`RotateKey`, `RevokeKey`, `RequestKeyEscrow`, `RequestKeyRecovery`,
`ListKeyEscrows`, `RequestSignature`, `GetAuditLog`, `ExportAuditLog`,
`GenerateComplianceReport`, `GetRegistrationChallenge`,
`RegisterPublicKey`, `ActivateSelfCustodyKey`, `RotateSelfCustodyKey`,
`AddUserToOrg`, `RemoveUserFromOrg`, `UpdateUserOrgRole`,
`ListOrgMembers` — roughly 25+ methods.

**Impact**: any authenticated user can read another org's audit log,
generate keys for another org, rotate another org's keys, etc. This
does not let them escalate to admin (admin RPCs are on a different
service), but it breaches the per-org confidentiality boundary that
multi-tenant deployments depend on.

**Recommended (REQUIRED for v1.2): membership-based authorization**.
Add a helper `authorize_org_access(claims, target_org_id, db) ->
Result<()>` that:
1. Returns `Ok` if `claims.role == SYSTEM_ADMIN` (cross-org by design).
2. Returns `Ok` if `claims.org_id == target_org_id` (caller's home org).
3. Otherwise queries `org_memberships` for `(claims.sub, target_org_id)`
   with role ≥ required_role; returns `Ok` on match.
4. Returns `PermissionDenied` otherwise. Audit-log every denial.

Apply at the start of every PortalService method that takes `org_id`.
Add an integration test fixture per method exercising the cross-org
denial path.

**Residual**: until enforced, multi-tenant confidentiality is a
deployment-trust assumption (operators must trust everyone with a
JWT). For single-tenant or steward-only deployments, the gap has no
practical impact.

### 3.4 Corruption — adversary wants false data persisted or true data dropped

#### AV-16: Public-key directory poisoning via cross-path re-registration

**Attack**: Attacker tries to register a public key already in
`partner_keys` via the *other* registration path. E.g.: a custodied key
exists for org A; attacker calls `RegisterPublicKey` (self-custody) for
org B with the same Ed25519 public key, hoping the per-path code does
not see the cross-path conflict.

**Mitigation in v1.1**: the `public_key_hash UNIQUE` constraint
(migration 020) is on the table, not the path — both
`create_self_custody_key` and `create_key` (post commit `4baec9b`) write
to the same column. The DB enforces uniqueness across both paths.
`register_public_key` additionally does an explicit pre-check via
`public_key_exists` (`portal.rs:2522`) for a clean
`AlreadyExists` error.

**Cross-org rotation hazard**: `RotateKey` / `RotateSelfCustodyKey`
issue a new key for the same org while keeping the old key active for
the grace period. Both old and new must satisfy `public_key_hash UNIQUE`
across the entire table. A rotation that re-uses a hash already taken
by a different org fails INSERT — the rotation is rejected with a clean
DB error. Verify by integration test (TODO).

**Residual**:
- An attacker who *already controls* an org and registers a key, then
  later tries to register the same key for a colluding second org,
  hits the UNIQUE — defeats the trivial Sybil. They could still attempt
  to register a *related* key (e.g., derived from the same seed but
  different derivation path); detection at this layer is impossible
  without scanning for cryptographic relationships. Out of scope.
- Cross-region registration (US registers, EU re-registers same key)
  is not detected. By design — see AV-3 residual.

#### AV-17: Schema-version mismatch on uploaded manifest

**Attack**: Adversary submits a function or binary manifest with a
`manifest_version` or wire-format shape the registry does not understand
(e.g., a v2 envelope to a v1-only registry; a CIRISVerify v1.8 hybrid-
signed manifest to a registry that only verifies v1.7 single-sig).
Either rejected silently, or stored under a wrong assumption.

**Mitigation in v1.1**: **Weak.** `register_function_manifest` accepts
arbitrary `manifest_version` string and stores it verbatim
(`db/function_manifests.rs:185`). There is no allowlist enforcement;
`SUPPORTED_VERSIONS` constant (Persist's pattern) does not exist here.

The `manifest_json` field is `serde_json::Value` and stored as JSONB —
deeply-nested or schema-skew bodies parse and persist as long as the
top-level envelope deserializes. The CIRISVerify-side validator is
where shape validation actually happens; pre-FSD-002, that validator
hard-coded "agent manifest shape" — the root cause cited in CIRISVerify
Issue #1 for the `project=ciris-persist` rejection.

**Recommended for v1.2.x (in coordination with CIRISVerify v1.8)**:
- Add a `SUPPORTED_MANIFEST_VERSIONS: &[&str]` allowlist; reject
  out-of-set versions with HTTP 422.
- Per-version typed deserialization gate: a `manifest_version=2.0.0`
  body must deserialize into the `BuildManifestV2` typed struct, not
  a free-form `Value`.
- Coordinate with CIRISVerify's `BuildManifest` generic per
  `docs/POB_SUBSTRATE_PRIMITIVES.md` — schemas align on both sides.

**Residual**: until allowlist + typed deserialize land, an upload with
an unrecognized shape is stored opaquely and only fails at downstream
verification time. Surface area for AV-6 substitution attacks
(unrecognized shape passes the auth gate, fails or behaves oddly at
verify time).

#### AV-18: Audit-log tampering via direct DB access

**Attack**: An adversary with PostgreSQL `DELETE` / `UPDATE` privilege
on the `audit_log` table erases or rewrites entries to hide a malicious
admin action.

**Mitigation in v1.1**: **Out of CIRISRegistry's protection scope** by
the §6 trust assumption that PostgreSQL is trusted. The `audit_log`
table has no append-only enforcement; entries do not include a
hash-chain link or a per-row signature.

**Operational mitigations available today**:
- PostgreSQL `REVOKE DELETE, UPDATE ON audit_log FROM <app_role>` —
  the application role only has INSERT + SELECT.
- Row-level security policies forbidding rewrites.
- Backup the audit log to append-only storage (S3 with object lock,
  WORM) on a continuous schedule.

**Recommended for v1.3.x**:
- Append-only enforcement at the DB layer (revoke + RLS policies in
  the migration).
- Per-row hash-chain link (`prev_entry_hash` + `entry_hash`) — like
  CIRISAgent's audit-chain — so missing or tampered rows are
  detectable by replay.
- Mirror to a transparency log (Sigstore Rekor or equivalent) for
  third-party verifiability. Aligns with the steward-accountability
  posture — "you're paying for accountability" is unconvincing if
  the audit trail is mutable.

**Residual**: today, a compromised DBA or DB-level compromise erases
evidence. Detection requires off-system log replication that is not
the registry's responsibility. Track for v1.3.x.

#### AV-19: Multi-replica migration race on cold boot

**Attack surface (operational, not adversarial)**: multiple registry
replicas (e.g., HA pair, Kubernetes Deployment with replicas>1) boot
concurrently against a single PostgreSQL instance. Each calls
`sqlx::migrate!()` on startup; concurrent DDL operations race.

**Mitigation in v1.1**: **sqlx's built-in advisory lock**. `sqlx::migrate!`
acquires `pg_advisory_lock(...)` keyed off a hash of the migrations table
name before applying any pending migrations. Concurrent boots block on
the lock; first wins, others observe the migrations as already applied
and proceed without re-running. No application-level lock is needed
(unlike Persist's AV-26 case where they use a custom advisory lock —
sqlx already does this for us).

**Verify**: confirm that all our migrations are wrapped in a single
transaction per file (sqlx default) so a panic mid-migration does not
leave half-applied state. Migrations 001-020 are reviewed as transaction-
safe.

**Residual**:
- A worker holding the migration lock that is paused indefinitely
  (SIGSTOP, kernel scheduler starvation) blocks subsequent workers.
  Deployment / liveness-probe concern. PostgreSQL's
  `idle_in_transaction_session_timeout` provides a backstop if
  configured.
- Migrations that perform non-transactional work (CREATE INDEX
  CONCURRENTLY, etc.) cannot be wrapped in a transaction. Review per-
  migration; flag any non-transactional ones in the migration header
  comment so reviewers know to special-case.
- The migration 020 fix (commit `07f50b6`) — `ADD CONSTRAINT IF NOT
  EXISTS` is invalid PostgreSQL — was caught in production and
  resolved with a `DO $$ ... $$` block. Lesson learned: pre-flight
  migrations in a staging environment matching production PG version
  before deploying.

### 3.5 Privacy — adversary wants exposed data

#### AV-20: PII leak via gRPC / HTTP error messages

**Attack**: Adversary submits malformed input to provoke an error path
that includes verbatim DB error text, parameter values, or schema names
in the response. Postgres error formatter often includes column names
and sometimes parameter values; sqlx error formatter passes those
through.

**Mitigation in v1.1**: **Partial.** Common pattern across the codebase
is `Status::internal(e.to_string())` — `grep -c "Status::internal" rust-
registry/src/services/portal.rs` returns dozens of sites. Each one
serializes the underlying error verbatim into the gRPC response.

The most common error chain is `sqlx::Error → RegistryError::Database
→ Status::internal`. The Postgres `42P01` ("relation does not exist"),
`23505` ("unique_violation") errors include the constraint name +
sometimes the conflicting value. For a unique violation on a public
fingerprint, the leak is low-impact (fingerprints are public). For an
audit-log query failure that includes a partial WHERE clause, more
sensitive.

HTTP error responses (`api/http.rs`) wrap errors more carefully —
typed enum variants with curated `message` strings, less verbose.

**Recommended for v1.2.x**:
- Introduce `RegistryError::sanitize_for_response()` that emits a
  typed error code + correlation ID, with the verbose form kept only
  in tracing logs.
- Audit every `Status::internal(e.to_string())` call site, replace
  with `Status::internal(error_code(e))` or similar.
- Add a regression test that submits each known-failing input and
  asserts the response body matches a fixed error-code dictionary
  (no payload echo).

**Residual**: until sanitized, deeply-malformed inputs (e.g.,
intentionally-wrong UUID formats) can leak DB internals. Low impact
on a single-tenant deployment; matters more for shared-tenant scenarios.

#### AV-21: Audit-log enumeration via PortalService.GetAuditLog

**Attack**: Adversary with any valid JWT calls `GetAuditLog` with a
target `org_id` they don't belong to, hoping to read another org's
admin actions, key rotations, user changes, etc.

**Mitigation in v1.1**: **Partial / coupled to AV-15.** The DB query
filters strictly: `WHERE actor_org_id = $1` (`db/audit.rs:104`). The
empty-string `org_id` matches only rows where the actor was empty —
in practice, none. So unauthenticated cross-org enumeration via empty
filter does not return data.

But AV-15 (PortalService cross-org access) means an authenticated
caller can pass *any* `org_id` and get that org's audit log. That is
the actual attack vector, and it inherits AV-15's residual.

`ExportAuditLog` (JSON / CSV / JSONL / Splunk formats) has the same
pattern with the same residual.

**Recommended**: AV-15 fix (membership-based authorization) closes
this. Until then, the audit log is only confidential against
unauthenticated callers, not against authenticated cross-org callers.

**Residual**: identical to AV-15.

#### AV-22: Custodied private key leak in logs / responses

**Attack**: An operator turns up log verbosity (`RUST_LOG=trace`) or
captures gRPC traffic; the Ed25519 private key returned by
`GenerateKeyPair` lands in observability tooling.

**Mitigation in v1.1**: **The registry itself does not log the private
key.** Verified:
- `portal.rs:603` log line emits only `org_id`, not the response body.
- `portal.rs:638` audit metadata includes `key_id` and
  `ed25519_fingerprint`, never the private key bytes.
- `db/keys.rs::create_key` does not accept the private key as a
  parameter — it never reaches the DB layer.
- No `Debug` derive on the response that would print `ed25519_private`
  bytes to a logger.

The wire-level exposure (AV-5) is the dominant concern, not log
exposure.

**Secondary**: tonic at default verbosity does not log request /
response bodies. A custom interceptor or `--reflection` debugging
session would.

**Residual**:
- Self-custody (FSD-002) eliminates the exposure entirely. Encourage
  agents to migrate.
- A future contributor adding a `tracing::debug!("response: {:?}",
  resp)` line would re-introduce the leak. Add a CI lint:
  `grep -rn "ed25519_private" rust-registry/src/ | grep -v "// " ` —
  expect zero matches outside the response constructor.

#### AV-23: Email enumeration via GetOrgUserByEmail timing

**Attack**: Adversary calls `GetOrgUserByEmail` with target emails to
distinguish "registered" from "unregistered" — useful for phishing
target enumeration.

**Mitigation in v1.1**: **Partial.** The endpoint requires JWT auth
(PortalService.Authenticated), so unauthenticated bulk enumeration is
gated. An authenticated caller can still enumerate within their
org_id (legitimate by design) and — coupled to AV-15 — across orgs.

Response distinguishes `NotFound` (gRPC `NOT_FOUND`) vs `Ok` (returns
user record). Timing difference is observable: `NotFound` short-circuits
after the SELECT returns 0 rows; `Ok` runs additional projection logic.
For a single lookup, timing leak is small; over many lookups, a binary
classifier emerges.

**Recommended for v1.2.x**:
- Constant-time lookup-OR-decoy that runs the full projection path
  even on miss. (Latency tax on every call; track as research-grade.)
- AV-15 fix (membership authorization) eliminates cross-org
  enumeration; intra-org enumeration is by design.
- Per-caller rate limit on `GetOrgUserByEmail` (e.g., 60/hour/user)
  with exponential backoff on misses.

**Residual**: low-impact when properly authenticated; couples to AV-15
when not.

#### AV-24: Side-channel timing on signature / fingerprint verification

**Attack**: Adversary submits requests that exercise the registry's
verification paths (signature check on uploaded manifests, fingerprint
lookup) and measures response time differences to enumerate the
public-key directory or distinguish "unknown" from "known but invalid"
states.

**Mitigation in v1.1**: **Partial.** `ed25519_dalek::verify_strict` is
constant-time over the signature/key path itself. However:
- `verify_key_by_fingerprint` (`/v1/verify/key/{fp}`) short-circuits
  on unknown fingerprint (DB miss → 404 fast; DB hit → record
  serialization slow). Timing leaks fingerprint membership.
- `register_public_key` PoP verification: unknown-challenge path
  short-circuits before signature verify; known-challenge-but-wrong-sig
  path runs signature verify. Distinguishable.
- HTTP `/v1/builds/{version}` returns 404 fast, 200 slow.

Fingerprints are emitted on every signed record (public information),
so directory enumeration via timing is **low-impact** (same conclusion
as Persist's AV-16). Other timing signals are higher-impact only when
they reveal non-public state.

**Recommended for v1.2.x research**: constant-response-time wrapper
on the verify endpoints (sleep to a P99 budget on the rejection
path). Latency tax on the happy path; track as research-grade
hardening.

**Residual**: directory enumeration via timing is possible; impact is
bounded because fingerprints are public anyway. Other timing signals
(challenge state, registration race) are mitigated by single-use
challenges (AV-2) and the rate limiter (AV-9, when applied to gRPC).

### 3.6 Provenance — adversary wants forged registry signatures

#### AV-25: Steward signing-key compromise

**Attack**: Adversary extracts the steward Ed25519 + ML-DSA-65 private
keys from the registry's storage tier. With the keys, attacker forges
arbitrary signed responses (license records, revocation lists, build
attestations, offline packages) that downstream CIRISVerify accepts as
authoritative.

**Mitigation in v1.2 (file mode is the only software-backed path)**:

After the ciris-crypto migration (commit `70f737d`), the registry holds
both signing keys as raw 32-byte seeds loaded from disk via
`HybridCrypto::from_files`. Vault mode and the broken dummy-key path
are gone (AV-27 closed by deletion). HSM mode (PKCS#11) is still
unimplemented.

| Mode | Where keys live | Extraction cost | Production-safe? |
|---|---|---|---|
| File mode (`ED25519_KEY_PATH` + `MLDSA_KEY_PATH` set) | Raw 32-byte seeds on disk | Root on host | Acceptable with full-disk encryption + restricted file perms (0400, dedicated user) |
| Ephemeral fallback (no key paths set) | Process heap | Heap dump on host | **No** — fresh on each boot, no persistence; records become unverifiable across restarts. Boot logs a loud `tracing::warn!`. |
| HSM mode | **NOT IMPLEMENTED** | Hardware extraction | v1.3.x roadmap |

**File-mode hardening (today)**:
- Mount the key paths read-only.
- Store on encrypted volume; require operator unlock at boot.
- Restrict file permissions: 0400 owned by the registry's runtime user.
- Audit any process with read access to the key files.

**Recommended roadmap**:
- v1.3.x: implement HSM mode (PKCS#11) for production deployments
  requiring FIPS 140-2 / 140-3 boundary. ciris-crypto's
  `ClassicalSigner` / `PqcSigner` traits are the natural integration
  points.
- v1.3.x: add hard-fail at boot when `ENVIRONMENT=production` and
  ephemeral fallback is taken (no key paths configured).
- v1.4.x: support steward-key threshold signing (M-of-N) so no single
  HSM / file compromise produces forgeable signatures.

**Residual**: until HSM mode lands, the steward key is software-backed
(file mode). File mode with FDE + 0400 perms + dedicated user is the
recommended production posture today.

#### AV-26: ML-DSA-65 verification missing on uploaded manifests (CLOSED — v1.3 Phase A, operationally validated)

**Status**: ✓ **Mitigated in v1.3 Phase A** (commit `4adc224`),
operationally validated by the registry's own self-publication on
every push to main since `cd95a9f` (2026-05-01).

**Original threat**: Adversary uploads a manifest with only a
classical Ed25519 signature (no ML-DSA-65). Future quantum adversary
forges the Ed25519 signature; the manifest is then fully forgeable
post-Q-day.

**Resolution**:
- New endpoint `POST /v1/verify/build-manifest` parses incoming JSON
  as a `BuildManifest`, looks up the per-primitive trusted keypair via
  `db::get_trusted_primitive_key(project)`, and calls
  `build_manifest::verify_uploaded_manifest` to check both Ed25519
  AND ML-DSA-65 signatures (bound — PQC over `canonical_bytes ||
  classical_sig`).
- Failure modes: 401 (admin token wrong), 403 `no_trusted_key` (no
  pubkey registered for primitive), 400 `verification_failed` (sig
  doesn't verify against registered pubkey).
- Trusted keys live in `trusted_primitive_keys` table (migration 022),
  cross-region replicated via Spock (migration 023, closes
  CIRISRegistry#4). Per-primitive pubkeys registered via three new
  admin RPCs: `RegisterTrustedPrimitiveKey`,
  `ListTrustedPrimitiveKeys`, `RevokeTrustedPrimitiveKey`.
- Vendored `BuildManifest` type in `rust-registry/src/build_manifest.rs`
  matches `ciris-verify-core` v1.8.0 wire format byte-for-byte
  (vendored to avoid `rusqlite`/`sqlx-sqlite` linker conflict;
  `ciris-crypto` provides the verifier primitives directly).

**Operational validation**: The registry's `.github/workflows/docker.yml`
"Sign + publish registry BuildManifest" step exercises the full trust
chain on every push to main:

1. Sign the registry binary's BuildManifest with the per-primitive
   build-signing keypair (GHA secrets `CIRIS_BUILD_ED25519_SECRET`,
   `CIRIS_BUILD_MLDSA_SECRET`).
2. POST the signed JSON to `/v1/verify/build-manifest` with
   `Authorization: Bearer ${REGISTRY_ADMIN_TOKEN}`.
3. Registry verifies the hybrid signature against the trusted key
   registered for `project='ciris-registry'` (fingerprint
   `567513a0c139412b...`).
4. On success, manifest is stored with the original CI signature
   intact and Spock-replicated US ↔ EU.
5. CI round-trip GET confirms the manifest is queryable from the
   public read path.

First green publish: commit `cd95a9f` at 2026-05-01 21:02:44Z.
Continuous green since. The recursive golden rule (we eat our own
dogfood for the federation substrate we provide) is operational, not
just architectural.

**Legacy POST endpoints**: `POST /v1/verify/binary-manifest` and
`POST /v1/verify/function-manifest` (re-sign server-side, no inbound
hybrid-sig verification) remain for backwards compat. Deprecation
tracked for v1.4 — once consumers migrate to the new endpoint, the
legacy paths can be removed.

**Residual**: per-primitive build-signing key compromise — see AV-34
(new) for the dual-secret-compromise threat surface introduced by the
self-publication flow.

#### AV-27: Vault-mode signing produces unverifiable signatures (CLOSED — v1.2)

**Status**: ✓ **Closed by deletion** in commit `70f737d` (ciris-crypto
v1.8.0 migration).

**Original defect**: `storage_mode=vault` retrieved the real Ed25519
public key from Vault Transit and cached it, but generated a fresh
ephemeral `dummy_ed25519` SigningKey for the local struct. The
codebase then called `HybridCrypto::sign` from four production paths
(`services/registry.rs:233`/`:632`, `api/http.rs:379`/`:753`), each
signing with the dummy key — producing signatures that did not verify
against the cached Vault pubkey. CIRISVerify, fetching the registry's
pubkey via `GET /v1/steward-key`, would reject every signed response
as invalid.

**Resolution**:
- The vault storage mode no longer exists. `vault.rs` (381 lines)
  deleted; `storage_mode` enum removed; `vault_*` config fields gone.
- `HybridCrypto` is now a thin wrapper over `ciris_crypto::HybridSigner`
  composed from `Ed25519Signer::from_seed` + `MlDsa65Signer::from_seed`.
  The dummy-key path is structurally impossible.
- Both signing keys load from disk as raw 32-byte seeds — the file
  mode AV-25 model applies.

**Future-state**: when HSM mode lands (v1.3.x), it will route signing
through `ciris-keyring::HardwareSigner` (Ed25519/ECDSA classical) with
ML-DSA-65 staying software-backed via `ciris-crypto::MlDsa65Signer`
and seed-at-rest hardening through `ciris-keyring::SecureBlobStorage`.
The dummy-key class of bug cannot recur because composition is via
trait objects with no fallback.

#### AV-28: Ephemeral signing keys in production (partially mitigated)

**Attack surface (operational, not adversarial)**: when neither
`ED25519_KEY_PATH` nor `MLDSA_KEY_PATH` is set, `HybridCrypto::new`
falls back to `generate_ephemeral` — fresh Ed25519 + ML-DSA-65 seeds
are generated on every boot. Records signed before a restart are
signed by a key the post-restart registry no longer holds; the
post-restart pubkey is different, so verifier-side caches no longer
match.

CIRISVerify caches the registry pubkey from `GET /v1/steward-key`.
Pre-restart cached pubkey becomes stale; post-restart signatures fail
verification until CIRISVerify refreshes its cache.

**Mitigation in v1.2 (commit `70f737d`)**:
- The legacy `storage_mode=memory` enum branch no longer exists.
  Memory mode is no longer a config option; ephemeral keys can only
  arise from missing `ED25519_KEY_PATH` / `MLDSA_KEY_PATH` env vars.
- Boot logs a loud `tracing::warn!` when falling back to ephemeral:
  *"No key paths configured — generating ephemeral keys. Records
  signed by this instance will not verify after restart."*

**Observed in production (2026-05-01 18:21)**: a registry restart
after the v1.2 deploy went into ephemeral mode because the
deployment's docker-compose / role config did not set
`ED25519_KEY_PATH` / `MLDSA_KEY_PATH`. The warning fired but the
process continued; CIRISVerify's cached pubkey was invalidated until
operator intervention. Concrete demonstration that "warn but continue"
is insufficient as a sole mitigation.

**Operational mitigation deployed (2026-05-01 evening)**: CIRIS bridge
ansible role updated:
- `runbooks/registry-keys-init.yml` (one-shot, idempotent) generates
  persistent 32-byte seeds, mirrors US seeds to EU for cross-region
  identity, sets `0440 root:999` perms (container runs as uid 999),
  recreates the container.
- `inventory/production-refactored.yml` + `group_vars/registry_servers.yml`
  set `registry_key_storage_mode: file`.
- Both production registry deployments (US + EU) now serve identical
  persistent steward identity:
  - `classical.key_id`: `75c29fccd21f80e4ddf03f6dfdfd9d4d2ba40db5720717079c1297dea18d265a`
  - `pqc.fingerprint`: `sha256:1c97265c775ab1ba186e50558cb25cb19949e2908ae66137a176868a64def4cc`

**Still tracked for v1.3.x (high priority — operational mitigation is
not a substitute for in-app enforcement)**:
- Boot-time hard-fail when `ENVIRONMENT=production` (or
  `production-like`) and either key path is unset. Future operators
  can still misconfigure; the registry should refuse to start rather
  than silently degrade.
- `/health` endpoint includes the current Ed25519 pubkey fingerprint
  so deployment scripts can compare against an expected value and
  alert on drift.
- Operational addendum (deployment guide): generate persistent seeds
  with `head -c 32 /dev/urandom > ed25519.seed && head -c 32 /dev/urandom
  > mldsa.seed`, mount as Docker secrets / Kubernetes Secrets / bind-
  mounted files with mode 0400 (or 0440 when running as a non-root
  user with a known uid).

**Residual**: production deployments are now hardened operationally,
but a *new* deployment that doesn't go through the bridge runbook
remains exposed until the in-app hard-fail lands. Detection today:
scrape boot logs for the warning string; check `/v1/steward-key`
fingerprint stability across restarts.

### 3.7 Operational / Hardening — non-adversarial availability vectors

#### AV-29: Plaintext PostgreSQL connection in production-like environments

**Attack surface (operational)**: registry connects to PostgreSQL with
`sslmode=disable`, exposing DB traffic (including audit-log writes,
key-row reads, password-bearing connection strings on rotation) to
network observers between the registry and the DB host.

**Mitigation in v1.1**: **Warning only.** `config.rs:158` emits a
`tracing::warn!` when `sslmode=disable` and `is_production_like` is
true. The warning is in startup logs but does not refuse to start.

The default `sslmode` in production-like environments is `require`
(`config.rs:150-156`), so the misconfiguration requires explicit
override.

**Recommended for v1.2.x**: hard-fail when
`ENVIRONMENT=production` and `sslmode=disable`. Treat misconfiguration
as a startup error, not a warning.

**Residual**: an operator who explicitly sets `sslmode=disable` in
production gets only a log warning. Detection: deployment-pipeline
lint that rejects `sslmode=disable` in production manifests.

#### AV-30: Unrotated REGISTRY_ADMIN_TOKEN

**Attack surface (operational)**: the `REGISTRY_ADMIN_TOKEN` env var
has no rotation lifecycle. Compromise (via container env exfiltration,
deployment-manifest leak, CI-secret breach) persists until the
operator manually rotates the secret + redeploys the registry + updates
every CI job that uses it.

**Mitigation in v1.1**: **None at the application layer.** The token
is read once at request-time via `std::env::var("REGISTRY_ADMIN_TOKEN")`
(`api/http.rs:336`, `:702`). There is no scope, no expiry, no audit
identity beyond "this token was presented."

**Recommended for v1.2.x**: same closure as AV-13 — replace with
short-lived scoped JWTs minted per-CI-job, bind subject to builder
identity. Until then:
- Rotate on a defined cadence (e.g., quarterly, post-incident).
- Restrict env-var access to the registry process (no shared shells,
  no debug interfaces with env-read).
- Consider a Vault dynamic-secrets backend that issues short-lived
  tokens scoped to the registry's manifest endpoints.

**Residual**: identical to AV-13. Operationally manageable in tightly-
controlled deployments; high-risk in shared-CI environments.

#### AV-31: gRPC reflection enabled in production

**Attack surface (operational)**: `tonic_reflection::server::Builder`
is configured unconditionally (`main.rs:87-89`) and the reflection
service is added to the gRPC server (`main.rs:97`). Any unauthenticated
client can list services and methods + retrieve message schemas.

**Mitigation in v1.1**: **None.** Reflection is enabled regardless
of `ENVIRONMENT`. The auth middleware does not gate reflection paths
(they fall under `/grpc.reflection` which `classify_path` returns
`AuthRequirement::None` for).

**Impact**: low-direct-risk (proto file is public on GitHub anyway),
but gives unauthenticated attackers a discovery surface that confirms
which RPCs exist and their request shapes — useful reconnaissance for
crafting exploitation payloads.

**Recommended for v1.2.x**:
- Gate reflection behind `ENVIRONMENT != production`.
- Or: gate reflection behind admin auth (`AuthRequirement::Admin`)
  so only authenticated admins can introspect.

**Residual**: low — proto is public. But a hardened production posture
would not advertise reflection.

#### AV-33: Spock multi-master migration desync (multi-region operational)

**Attack surface (operational, not adversarial)**: in a Spock multi-
master PostgreSQL cluster (US ↔ EU), sqlx's `_sqlx_migrations` table
is in Spock's default replication set. When one node applies a new
migration (e.g., the migration 021 deploy on 2026-05-01):
1. Node A boots, sqlx runs the DDL locally + INSERTs `(version=21,
   success=t)` into `_sqlx_migrations`.
2. The bookkeeping row replicates to Node B via DML replication.
3. Node B boots ~seconds later, sqlx checks `_sqlx_migrations`, sees
   "21 already applied", **skips its own DDL run**.
4. Node B's schema is silently desynced — `ALTER TABLE` never ran
   on its postgres.

Symptom: Node B's API logs immediately produce `Error fetching binary
manifest: column "project" does not exist`. Half the multi-region
fleet returns 500s on the affected endpoints.

**Severity**: production-impacting in multi-master deployments.
Single-node deployments and single-master replicas (where DDL
propagates via base-table replication) are unaffected.

**Mitigation in v1.3 (in-repo, shipped)**: `Database::migrate`
(`db/mod.rs`) now calls `exclude_sqlx_migrations_from_spock_replication`
*before* running `sqlx::migrate!`. The helper:
- Detects Spock via `SELECT EXISTS(SELECT 1 FROM pg_extension WHERE
  extname = 'spock')`. No-op when absent (single-node dev / staging /
  test).
- Wraps `SELECT spock.repset_remove_table('default',
  'public._sqlx_migrations')` in a PL/pgSQL `DO $$ ... EXCEPTION WHEN
  others ... $$` block. The second-and-subsequent boots (when the
  table is already excluded) raise an error inside the Spock function
  — the EXCEPTION handler catches it as a NOTICE and the DO block
  returns success.

**Hotfix history**: the v1.3 Phase 1 initial implementation matched
on substrings of Spock's error message (`"not a member"`, `"does not
exist"`); the actual wording differs in Spock 5.0.1, which caused
production crash-loops on container restarts (observed evening
2026-05-01 — three restart-induced crashes during the same session
the fix shipped). Hotfix moved the exception handling into the SQL
layer (DO block) so it's wording-independent.

Each node's migration runner authoritatively tracks what it locally
executed. The bridge ansible reconcile task is now redundant for
new deployments (it remains as defense-in-depth for the original
production cutover).

**Convention enforced via `CLAUDE.md` "Database Migration Notes"**:
all schema-changing migrations must be idempotent (`IF NOT EXISTS` /
`IF EXISTS` / `DO $$ ... $$` constraint guards). Migration
`021_project_namespace.sql` is the canonical example. `spock.replicate_ddl_command`
is explicitly disallowed — single-node dev would error.

**Original bridge-side mitigation (2026-05-01)** remains documented
for historical context: ansible role removed `_sqlx_migrations` from
Spock's default repset on both nodes and added a per-deploy reconcile
task. With the in-repo fix, the reconcile task becomes belt-and-
suspenders.

**Residual**: existing migrations 001 and 020 contain bare `CREATE
INDEX` statements (without `IF NOT EXISTS`). They are first-run-only
in practice (every deployment that has migration 020 applied has
already passed both successfully); the convention applies forward
to new migrations. A mid-flight crash on migration 022+ that fails
the second attempt would still surface as a deploy error rather than
silent desync.

#### AV-35: Audit-trail actor forgeability (W1, mitigated v1.3 Phase 5)

**Original threat (pre-v1.3 Phase 5)**: PortalService handlers
including `generate_key_pair`, `activate_key`, `rotate_key`,
`revoke_key`, `register_public_key`, and `create_audit_entry` read
`req.requester_user_id` (or `req.actor_user_id`) from the request body
and wrote it directly to the `audit_log` row's `actor_user_id` column.
Any authenticated caller could specify any string as the actor — the
audit trail was forgeable across the entire registry. A forensic
investigation tracing "who rotated this key" or "who revoked that key"
would be reading attacker-controlled labels, not authenticated
identities.

This was W1 in Agent 3's AV-15 investigation report, surfaced
alongside (but distinct from) the cross-org access gap. Higher
practical severity than AV-15 alone because audit trails feed
incident-response decisions.

**Mitigation in v1.3 Phase 5 (commit pending)**:
- All five handlers (generate_key_pair, activate_key, rotate_key,
  revoke_key, register_public_key) derive `actor_user_id` from
  `claims.sub` extracted via `claims_from_request` in the Phase 4
  preamble. The `req.requester_user_id` proto field is preserved on
  the wire (no breaking change) but ignored on the server side.
- `db::rotate_key`, `db::revoke_key`, and `db::create_self_custody_key`
  also receive `claims.sub` for their own per-row metadata columns
  (`rotated_by`, `revoked_by`, `created_by`) — so the columns reflect
  authenticated identity, not request-body claims.
- `create_audit_entry` (Portal's "log on behalf" surface): SYSTEM_ADMIN
  may supply any `actor_user_id` (legitimate cross-actor logging case
  for Portal-mediated events like "user X logged in"); non-admin
  callers must either omit `actor_user_id` (defaults to `claims.sub`)
  or supply the SAME value as `claims.sub`. Mismatch returns
  `PermissionDenied` with explicit "must equal JWT subject"
  message. Forge attempts are no longer silent — they fail loudly.

**Residual**:
- Audit entries written before v1.3 Phase 5 ship may have forged
  `actor_user_id` values. Pre-Phase-5 audit data should be treated as
  "actor field is best-effort attribution, not authenticated
  identity". Post-Phase-5 entries can be relied upon.
- A SYSTEM_ADMIN can still log on behalf of any user via
  `create_audit_entry` (intentional for Portal-mediated event
  recording). Distinguishable in the audit log via the action type +
  description, but not via the actor field alone. AV-13
  (REGISTRY_ADMIN_TOKEN compromise) and AV-12 (JWT secret leak)
  compound this — closing those tightens the SYSTEM_ADMIN-trust
  envelope.
- gRPC AuthLayer + Phase 4 `claims_from_request` enforcement remain
  the floor of trust: a missing JWT means no claims, which means no
  actor can be derived. Handlers that try `claims_from_request(...)?`
  and fail return `Unauthenticated` before any write.

**Detection** of pre-mitigation forge attempts: scan `audit_log` for
rows where `actor_user_id` doesn't match a known JWT subject from
the same time window. Not implemented; track for v1.4 if needed.

#### AV-34: Build-signing key compromise (CI-side, post-Phase-A surface)

**Attack surface introduced**: with v1.3 Phase A's BuildManifest
verification, a per-primitive build-signing key (held in CI as a GHA
secret) is now a load-bearing trust anchor. For
`project='ciris-registry'`, that's the `CIRIS_BUILD_ED25519_SECRET` /
`CIRIS_BUILD_MLDSA_SECRET` pair on the CIRISRegistry repo
(fingerprint `567513a0c139412b...`). For peer primitives
(ciris-persist, ciris-lens, ciris-agent, ciris-verify), each has its
own equivalent in their own repo's GHA secrets.

**Threat**: An adversary who exfiltrates the build-signing seed (CI
secret leak via job-step compromise, malicious workflow PR, GHA
infrastructure breach) can sign a malicious BuildManifest for that
primitive. If they ALSO have the registry's `REGISTRY_ADMIN_TOKEN`
(AV-13 surface), they can publish that backdoored manifest to the
registry, where it will pass hybrid-sig verification against the
registered trusted key and be served to consumers as authoritative.

Consumers fetching via Path A or Path B (per `TRUST_CONTRACT.md`) will
verify the hybrid signature, see it valid, and trust the manifest.
The malicious binary_hash + functions table become the registry's
"authentic" claim for that build.

**Mitigation in v1.3 (defense-in-depth, not single-point)**:

- **Two-secret co-requirement**: an attacker needs BOTH the
  build-signing key AND `REGISTRY_ADMIN_TOKEN` to publish. Either
  alone is insufficient:
  - Admin token alone: no trusted key registered for an attacker's
    fresh keypair → 403 `no_trusted_key`. (Unless the attacker also
    has admin authority to call `RegisterTrustedPrimitiveKey` for a
    new pubkey they control, which is a stronger compromise — see
    AV-13 closure roadmap for scoped JWTs that limit this.)
  - Build-signing key alone: no auth → 401.
- **Per-repo isolation**: each primitive's build-signing key lives in
  that repo's GHA secrets. Compromise of one repo's CI doesn't grant
  publish authority for another primitive (e.g., compromising
  CIRISPersist's GHA can publish ciris-persist manifests but not
  ciris-registry manifests).
- **Audit trail**: every successful publish writes to `audit_log` with
  the verifying key fingerprint and request metadata; a compromised
  key's first publish is observable in the log.
- **Round-trip integration test**: every CI publish is round-trip
  verified against the public GET endpoint. Disagreement (publish
  succeeds but GET returns wrong data) would surface in CI logs.

**Recommended for v1.4 (additional defense-in-depth)**:

- **Cosign verification on uploaded manifests**: require POSTs to
  carry a sigstore signature (chained to GitHub OIDC for the
  publishing workflow's identity). Compromising a build-signing key
  alone is no longer enough — attacker must also forge a sigstore
  signature, which requires compromising GitHub's OIDC infrastructure.
- **Multi-party signing for high-stakes primitives**: M-of-N signing
  on the build manifest itself (e.g., release-manager + CI-job both
  sign; registry verifies both).
- **Rotation cadence**: per-quarter build-signing key rotation,
  documented operator runbook (matches the steward-key rotation
  posture in `TRUST_CONTRACT.md` §3.4).
- **Per-build provenance**: extend BuildManifest extras with a SLSA
  attestation pointing to the GitHub Actions run + commit SHA;
  consumer can verify the workflow that produced the manifest matches
  the expected one.

**Residual**: until v1.4 hardenings land, an attacker with both
secrets in hand can publish a malicious manifest. Detection is
post-facto via audit log + cross-region replication divergence + (for
high-impact primitives) external sigstore-signed release artifacts on
GitHub releases. The dual-secret co-requirement raises the bar above
single-secret-compromise, but does not eliminate the class entirely.

#### AV-32: Race in self-custody key activation

**Attack**: Adversary observes a `RegisterPublicKey` succeed for a
target org and races the legitimate operator to call
`ActivateSelfCustodyKey` first. If the activation challenge can be
satisfied without holding the original signing key, the attacker
activates the key under their control.

**Mitigation in v1.1**: activation requires a fresh signature over the
**activation challenge** (separate from the registration challenge,
stored in `activation_challenges` keyed by `key_id`). The signature
must verify against the **public key registered with that key_id** —
which means only the holder of the corresponding private key can
activate. An attacker who only saw the registration response cannot
activate.

The activation challenge is single-use (`get_and_remove_activation_challenge`
deletes on consume, `db/keys.rs:474`). Concurrent activation attempts
race for the row; one wins, others get expired.

**Residual**: the attack reduces to AV-2 — if the attacker controls
both the registration challenge and the activation challenge AND
substitutes their own pubkey at registration time, they can activate.
TLS at the deployment edge is the closure. Per-org rate limit on
challenge issuance (v1.2.x) tightens the window.

---

## 4. Mitigation Matrix

| AV | Attack | Primary Mitigation | Secondary | Status | Tracker |
|---|---|---|---|---|---|
| AV-1 | Project-name collision blocks non-agent peers | UNIQUE(project, version) + slug validator | Project-name impersonation by admin-token holders not gated | ✓ Mitigated v1.2 (commit 254a89e) | v1.3.x: per-project signing-key binding |
| AV-2 | Self-custody challenge replay | Single-use atomic DELETE on consume + 5-min TTL | TLS at edge | ✓ Mitigated | — |
| AV-3 | Cross-org pubkey reuse (Sybil) | `public_key_hash UNIQUE` (migration 020) + pre-check | Cross-region NOT enforced (by design) | ✓ Mitigated | — |
| AV-4 | Forged build attestation blob | Admin-auth gate at registration | (no signature verify on blob) | ⚠ Partial | v1.2.x: Sigstore-style verify |
| AV-5 | Custodied private key MITM in transit | TLS at deployment edge required | Self-custody (FSD-002) avoids entirely | ⚠ Deployment-trust | v1.2.x: hard-fail on no-TLS |
| AV-6 | Function manifest substitution by token holder | REGISTRY_ADMIN_TOKEN gate + server-side signing | (ON CONFLICT DO UPDATE allows overwrite) | ⚠ Couples to AV-13 | v1.2.x scoped JWT |
| AV-7 | HTTP body-size flood | `RequestBodyLimitLayer(1 MiB)` global | Per-RPC list caps (Batch* limits) | ✓ Mitigated | — |
| AV-8 | Rate-limiter X-Forwarded-For spoofing | `is_trusted_proxy` CIDR allowlist | Direct peer IP fallback | ✓ Mitigated | v1.2.x: loud-fail misconfig |
| AV-9 | Mass-query DB pool saturation | `middleware/rate_limit.rs` tower Layer applied to gRPC stack; axum public GETs wrapped via `from_fn(rate_limit_public_http)`; 60s cleanup ticker prevents unbounded HashMap growth | Tier mapping (Bypass/Public 60-min/Verify 5-min) per `classify_tier`; deployment-edge rate limit as defense-in-depth | ✓ Mitigated v1.3 (Phase 2) | Per-RPC pool-exhaustion circuit breaker — separate concern, track for v1.3.x |
| AV-10 | Emergency-shutdown abuse | Admin role gate + audit log | (single-actor) | ⚠ Single-actor | v1.2.x: M-of-N + webhook fanout |
| AV-11 | Mass-revoke abuse | Admin role gate + audit log + reason code | (no list-size cap) | ⚠ Single-actor | v1.2.x: list cap + webhook fanout |
| AV-12 | JWT forgery via leaked HS256 secret | Validation only; symmetric key | (no asymmetric option) | ⚠ Architectural | v1.2.x: EdDSA/RS256 + JWKS |
| AV-13 | REGISTRY_ADMIN_TOKEN extraction | Bearer check; static env var | (no rotation, no scope, no audit identity) | ⚠ Architectural | v1.2.x: scoped JWT |
| AV-14 | JWT role escalation | HMAC-signed claims (couples to AV-12) + W2 (Phase 6): explicit `authorize_system_admin` on 13 PortalService god-mode handlers (`create_organization`, `list_organizations`, `batch_create_organizations`, `create_system_user`, `get/list/update_system_user`, `lookup_system_user_by_o_auth`, `link_system_user_o_auth`, `list_system_user_o_auth_identities`, `create_user`, `lookup_user_by_o_auth`, `upgrade_to_partner`); self-or-admin on user-OAuth linking | Single binary admin flag (no sub-roles); SYSTEM_ADMIN-on-behalf for `create_audit_entry` | ⚠ Couples to AV-12; W2 PortalService gating closed in v1.3 Phase 6 | v1.4: scoped sub-roles via `scopes` claim |
| AV-15 | PortalService cross-org access | Per-handler `claims_from_request` + `authorize_org_access(db, claims, target_org_id, required_role)` preamble — applied to all ~30 PortalService methods that take `org_id`. SYSTEM_ADMIN bypass for cross-org by-design ops. Membership lookup via `db::get_user_role_in_org`. Audit-log denial via `AUDIT_ACCESS_DENIED`. | W3 inner-proto vigilance: handlers using `org.org_id` / `user.org_id` / `sign_request.org_id` authz against the same field used for the DB write. Batch RPCs reject mismatched per-record org_id with `invalid_argument` (fail-secure for whole batch). | ✓ Mitigated v1.3 (Phase 4) | Integration tests for cross-org denial — track follow-up |
| AV-16 | Public-key cross-path re-registration | `public_key_hash UNIQUE` constraint enforces across paths | Pre-check in `register_public_key` | ✓ Mitigated | — |
| AV-17 | Schema-version mismatch on uploaded manifest | (none — accepts any string) | CIRISVerify-side validates downstream | ⚠ Weak | v1.2.x: SUPPORTED_VERSIONS allowlist |
| AV-18 | Audit-log tampering via DB access | Out of scope (PG trusted) | (no append-only enforcement) | ⚠ Architectural | v1.3.x: hash chain + Rekor mirror |
| AV-19 | Multi-replica migration race | sqlx built-in `pg_advisory_lock` on _sqlx_migrations | (operator must use transaction-safe migrations) | ✓ Mitigated | — |
| AV-20 | PII / DB internals leak via errors | (mostly raw `e.to_string()`) | HTTP errors typed | ⚠ Partial | v1.2.x: sanitize_for_response |
| AV-21 | Audit-log cross-org enumeration | Filtered by `actor_org_id` (couples to AV-15) | (AV-15 fix closes) | ⚠ Couples to AV-15 | (closes with AV-15) |
| AV-22 | Custodied private key in logs | No log/DB write paths touch private bytes | Self-custody avoids | ✓ Mitigated | CI lint to prevent regression |
| AV-23 | Email enumeration via timing | JWT auth gate; couples to AV-15 | NotFound vs Ok timing observable | ⚠ Couples to AV-15 | v1.2.x: rate limit + AV-15 fix |
| AV-24 | Side-channel timing on verify | `verify_strict` constant-time | Endpoint-level short-circuit timing leaks | ⚠ Low impact | v1.2.x research |
| AV-25 | Steward signing-key compromise | File mode + FDE + 0400 perms (only working software-backed option) | Hybrid signatures (both must fall) | ⚠ Architectural | v1.3.x: HSM impl via ciris-keyring; v1.4.x: threshold sig |
| AV-26 | ML-DSA-65 missing on uploaded manifests | `POST /v1/verify/build-manifest` parses BuildManifest, looks up trusted key by `manifest.primitive.project_name()`, calls `build_manifest::verify_uploaded_manifest` (Ed25519 + ML-DSA-65 hybrid bound-sig) before storing | Operationally validated: registry self-publishes its own BuildManifest on every push to main (since `cd95a9f` 2026-05-01) — full trust chain exercised continuously, recursive golden rule operational | ✓ Mitigated v1.3 (Phase A), validated v1.3 Phase A.1 | Legacy `POST /v1/verify/binary-manifest` / `function-manifest` paths remain admin-token-only for backwards compat — track v1.4 deprecation; AV-34 (build-signing key compromise) is a new surface introduced by this mitigation |
| AV-34 | Build-signing key compromise (CI-side) | Two-secret co-requirement: attacker needs BOTH GHA build-signing seed AND `REGISTRY_ADMIN_TOKEN` to publish | Per-repo GHA isolation; `audit_log` writes with verifying-key fingerprint; CI round-trip integration test on every publish | ⚠ Architectural (defense-in-depth, not single-point) | v1.4: cosign verification on uploads + GitHub OIDC chain; multi-party signing for high-stakes primitives; rotation cadence; SLSA attestation in BuildManifest extras |
| AV-35 | Audit-trail actor forgeability (W1) | All audit-writing handlers (generate/activate/rotate/revoke_key, register_public_key) derive actor from `claims.sub`; `req.requester_user_id` preserved on wire but ignored. `create_audit_entry`: SYSTEM_ADMIN may log on behalf; non-admin must match `claims.sub` or be rejected with `PermissionDenied`. | DB metadata columns (`rotated_by`, `revoked_by`, `created_by`) also receive `claims.sub`, not request body. Forge attempts on `create_audit_entry` fail loudly (no silent acceptance). | ✓ Mitigated v1.3 (Phase 5) | Pre-Phase-5 audit entries treated as best-effort attribution. Post-Phase-5 reliable. SYSTEM_ADMIN-on-behalf logging still allowed by design — couples to AV-12/13/14 closures. |
| AV-27 | Vault-mode signing produces unverifiable sigs | (deleted — vault mode no longer exists) | ciris-crypto migration | ✓ Closed by deletion v1.2 (commit 70f737d) | — |
| AV-28 | Ephemeral signing keys in production | `tracing::warn!` at boot; storage_mode=memory enum gone | Bridge ansible deployed persistent seeds 2026-05-01 evening (US + EU) | ⚠ Operationally mitigated; in-app hard-fail tracked | v1.3.x: hard-fail in production when paths unset |
| AV-33 | Spock multi-master migration desync | `db/mod.rs::exclude_sqlx_migrations_from_spock_replication` runs at boot before `sqlx::migrate!` | Bridge ansible reconcile task as defense-in-depth | ✓ Mitigated v1.3 (in-repo fix shipped, closes Registry#2) | — |
| AV-29 | Plaintext PG connection in production | `tracing::warn!` only | Default `sslmode=require` | ⚠ Warning-only | v1.2.x: hard-fail |
| AV-30 | Unrotated REGISTRY_ADMIN_TOKEN | (none — couples to AV-13) | Operational rotation cadence | ⚠ Couples to AV-13 | (closes with AV-13) |
| AV-31 | gRPC reflection in production | (none — enabled unconditionally) | Proto is public anyway | ⚠ Low impact | v1.2.x: env-gate |
| AV-32 | Self-custody activation race | Activation challenge per-key + signature against registered pubkey | Single-use atomic consume | ✓ Mitigated | — |

**Posture summary at a glance**:
- ✓ Mitigated: 15 — AV-1 (project namespace, v1.2), AV-2, AV-3, AV-7, AV-8, AV-9 (rate limit, v1.3 Phase 2), AV-15 (cross-org access, v1.3 Phase 4), AV-16, AV-19, AV-22, AV-26 (BuildManifest verify, v1.3 Phase A), AV-27 (closed by deletion, v1.2), AV-32, AV-33 (Spock fix, v1.3 Phase 1), AV-35 (audit-actor forgeability, v1.3 Phase 5)
- ⚠ Operationally mitigated; in-app fix tracked: AV-28 (persistent keys deployed 2026-05-01)
- ⚠ Gap requiring v1.3: (none — all in-app gaps closed)
- ⚠ **CRITICAL bugs**: 0 (AV-27 closed in v1.2)
- ⚠ Architectural deferrals: AV-12 / AV-13 / AV-14 / AV-30 (auth model rework), AV-18 / AV-25 (HSM + audit chain)
- ⚠ Lower-priority hardening: AV-4, AV-5, AV-6, AV-10, AV-11, AV-17, AV-20, AV-23, AV-24, AV-29, AV-31

---

## 5. Security Levels by Deployment Mode

The registry's key-loading configuration and auth setup determine which
attack vectors apply. Critical invariant: **all tiers run the same auth
middleware, same DB schema, same RPC handlers**. A finding in one
tier's implementation is presumed to apply to the same surface in
others unless explicitly excepted.

| Tier | Key storage | Auth | Effective threat model |
|---|---|---|---|
| **Production HSM** (target; not implemented) | `ciris-keyring::HardwareSigner` (TPM / StrongBox / Secure Enclave) for classical; `MlDsa65Signer` + `SecureBlobStorage` for PQC | mTLS + asymmetric JWT (EdDSA) | AV-25 closed at hardware tier; AV-12/13/14 closed by asymmetric JWT. **Requires v1.3.x roadmap.** |
| **Production file-mode** (working today, recommended) | Raw 32-byte seeds on disk via `ED25519_KEY_PATH` + `MLDSA_KEY_PATH`, restricted FS perms (0400) | JWT (HS256) | Full §3 model applies. AV-25 file-extraction risk mitigated by FDE + 0400 perms + dedicated user. AV-12/13 architectural residual. **The recommended option today.** |
| **Ephemeral fallback** (key paths unset) | Process heap, fresh each boot | JWT (HS256) optional | **NOT FOR PRODUCTION.** Boot logs `tracing::warn!` but does not refuse. Records signed before restart become unverifiable; CIRISVerify cached pubkey goes stale. **AV-28** observed in production 2026-05-01 — production hard-fail tracked for v1.3.x. |

**Per-component sub-tiers**:

| Component | Sub-tier note |
|---|---|
| **gRPC `RegistryService`** (public read-only) | No auth; AV-9 rate-limit gap is the dominant risk. AV-7 body limit applies. |
| **gRPC `PortalService`** (authenticated) | JWT validated but **org binding NOT enforced** (AV-15). Multi-tenant deployments must treat any JWT holder as having read/write access to any org until AV-15 is closed. |
| **gRPC `RegistryAdminService`** (admin) | JWT + role check. Single-actor for high-blast-radius operations (AV-10, AV-11). |
| **HTTP `/v1/builds/`, `/v1/verify/`** | No auth on GET; AV-9 rate-limit gap on these too. |
| **HTTP `/v1/verify/binary-manifest` POST + function-manifest POST** | `REGISTRY_ADMIN_TOKEN` env-var bearer. AV-13 + AV-30 dominant. |
| **HTTP `/v1/integrity/*`** | No auth on nonce / verify endpoints; per-IP rate limit applied. Self-contained challenge-response. |

---

---

## 6. Security Assumptions

The system depends on these assumptions; if violated, the threat model
breaks.

1. **PostgreSQL backend trusted**: the registry's sqlx layer trusts row
   reads. A compromised DB lets the attacker inject any record. The
   `audit_log` table has no append-only enforcement at the application
   layer (AV-18). Mitigation is operational: restrict app role
   privileges, enable RLS, mirror to off-system append-only storage.
2. **TLS termination at deployment edge**: gRPC and HTTP traffic must
   be TLS-protected by the time it leaves the host. Plaintext exposes
   the custodied private-key response (AV-5), the JWT token (AV-12),
   the admin token (AV-13), and all challenge / signature exchanges.
3. **`jwt_secret` and `REGISTRY_ADMIN_TOKEN` held in a deployment-
   controlled secrets store** (Vault, AWS Secrets Manager, sealed
   secrets) — not in source control, not in unencrypted env files.
   Both are root-of-trust for their respective auth tiers (AV-12,
   AV-13).
4. **Steward signing key in Vault (when AV-27 fixed) or in a restricted-
   access file with FDE on a hardened host**. HSM mode is the target
   but unimplemented. `memory` mode is dev-only (AV-28).
5. **Clock accuracy ±5 minutes** at the registry host. Challenge TTLs
   (FSD-002 5-min windows), JWT `exp`, license expiry, and revocation
   timestamp ordering all depend on it. NTP-backed; not validated at
   the application layer.
6. **CIRIS L3C steward role has not been compromised**. The steward
   issues licenses, registers partners, and authorizes professional
   capability grants. Compromise lets the attacker mint valid-looking
   PROFESSIONAL_MEDICAL etc. licenses. Out of scope at the registry
   layer; mitigation is at the governance + key-storage tier.
7. **Multi-region DNS is genuinely diverse**: US and EU registry
   deployments must be served by different DNS registrars. Single-
   registrar compromise breaks CIRISVerify's 2-of-3 consensus.
8. **gRPC reflection is not consulted as a security boundary**: the
   reflection endpoint reveals service / method names, but proto is
   public anyway (AV-31). Operators should still env-gate it for a
   hardened posture.
9. **Rust runtime memory safety**: no `unsafe` blocks in registry
   sources; transitive dependencies' `unsafe` is constrained by their
   own audits. `cargo audit` clean (per CI) is the operational
   safeguard. Periodic dep-tree review for any unsafe crate
   substitution.
10. **CIRISVerify and downstream consumers re-verify hybrid signatures**:
    the registry signs every authoritative response with hybrid Ed25519
    + ML-DSA-65; consumers that accept only classical signatures lose
    the post-quantum guarantee.

---

---

## 7. Fail-Secure Degradation

All failures degrade to MORE restrictive modes, never less. This is FSD-001's
"Fail Secure" design principle made operational.

| Failure Condition | Resulting Behavior | Rationale |
|---|---|---|
| JWT missing on protected RPC | HTTP 401 | Cannot authenticate → cannot proceed |
| JWT invalid (expired, bad signature, wrong issuer) | HTTP 401 | Same |
| Admin role missing on admin RPC | HTTP 403 | Not authorized for admin endpoint |
| `REGISTRY_ADMIN_TOKEN` env var unset | HTTP 500 | Misconfiguration; loud failure (`api/http.rs:337`) |
| `REGISTRY_ADMIN_TOKEN` mismatch on manifest POST | HTTP 401 | Bearer check fails closed |
| Public-key hash collision on register | gRPC `AlreadyExists` (or DB `UniqueViolation` → `Status::internal`) | Sybil resistance |
| Self-custody challenge expired / consumed | gRPC `InvalidArgument` ("Invalid or expired challenge") | Cannot prove possession → cannot register |
| Self-custody PoP signature fails Ed25519 verify | gRPC `InvalidArgument` ("Signature verification failed") | Unverified bytes never become an active key |
| Activation challenge fails | gRPC error; key stays PENDING | Key is unusable until activated |
| Lookup target not found (agent / partner / build) | gRPC `NotFound` (or HTTP 404) — empty response | Verifier treats as `UNLICENSED_COMMUNITY` (FSD-001 fail-secure) |
| HTTP body exceeds 1 MiB | HTTP 413 before body read | Memory bound enforced |
| Per-IP rate limit exceeded (HTTP integrity endpoints) | HTTP 429 + `Retry-After` | Honest backpressure |
| PostgreSQL unreachable | gRPC `Internal` / HTTP 500 | No degraded-write path; fail loudly so operators page |
| PostgreSQL timeout / pool exhaustion | gRPC `Internal` | Same; no silent retry |
| Emergency shutdown active | Verifier-side: all pro capabilities → DENY | Lockdown by design (registry just stores the flag; CIRISVerify enforces) |
| `storage_mode=memory` on production env | (no enforcement today — silent degradation) | **AV-28: needs v1.2.x boot-time refusal** |
| Vault unreachable in `vault` mode | Boot fails (`from_vault` returns error) | Fail-closed at startup; no fallback to memory |
| Vault Transit signing fails mid-request | Currently: HybridCrypto::sign uses dummy key (AV-27 BUG) | **Must fail-closed; tracked as critical hotfix** |
| Migration race lost (block forever waiting on advisory lock) | Boot blocks on `pg_advisory_lock` until first replica releases | Serializes boot; no half-applied schema |
| Migration syntax error (e.g., `ADD CONSTRAINT IF NOT EXISTS`) | Boot fails; sqlx_migrations row written but error surfaced (per commit `07f50b6` lesson) | Caught at deploy; recovery: delete the row + fix + redeploy |

Critical invariants:
- **`signature_verified=false` records do not exist in the registry.** Any
  RPC handler that takes a signature verifies it before persisting; failure
  is a typed error, never a stored row.
- **Unknown agents / partners default to community tier downstream.** The
  registry returns `NotFound`; CIRISVerify treats absence as the most
  restrictive state.
- **Any single revocation signal triggers immediate enforcement.** The
  revocation list is signed and consumed by all CIRISVerify deployments.

---

---

## 8. Residual Risks

Risks CIRISRegistry mitigates but cannot fully eliminate at the v1.1
baseline.

1. **Steward-key compromise** (AV-25). File mode with disk encryption +
   restricted access is the working production option; vault mode is
   broken (AV-27); HSM mode is unimplemented. Closure: v1.2.x HSM
   support, v1.3.x threshold signing.

2. **JWT secret leak** (AV-12 + AV-14). HS256 symmetric secret is a
   shared-fate root-of-trust between Portal and Registry. Closure:
   v1.2.x asymmetric JWT (EdDSA / RS256) + JWKS rotation.

3. **REGISTRY_ADMIN_TOKEN compromise** (AV-13 + AV-30). Static env-var
   bearer with no rotation, no scope. Closure: v1.2.x scoped short-
   lived JWTs minted per CI job.

4. **PortalService cross-org access** (AV-15). Handlers trust
   `req.org_id`. **Required v1.2 fix**: membership-based authorization.
   Until then, multi-tenant deployments rely on operator trust.

5. **Vault-mode signing bug** (AV-27). **Critical hotfix required**.
   Until fixed, vault mode produces unverifiable signatures. File mode
   is the only working software-backed option.

6. **Public-key directory enumeration via timing** (AV-24). Low-impact
   (fingerprints are public); track as research-grade hardening.

7. **Audit-log tampering** (AV-18). Out of CIRISRegistry's protection
   scope; deployment-infrastructure concern. Closure: v1.3.x append-only
   enforcement + transparency-log mirror.

8. **Compromised CIRIS L3C steward role**. The steward is the root of
   trust for license issuance; a compromised steward can issue valid-
   looking licenses for malicious agents. Out of scope at the registry
   layer; mitigation is at the governance + key-storage tier (FSD-001
   §Mission Alignment).

9. **Multi-region DNS simultaneous compromise**. CIRISVerify multi-source
   consensus tolerates one source compromise; not all. Mitigation:
   different registrars in US / EU by policy. Per Accord NEW-04 and
   PoB §5.1, no detector is complete.

10. **Quantum break of Ed25519**. Current quantum compute cannot break
    Ed25519, but Shor's on a sufficiently large quantum computer would.
    Closure: hybrid Ed25519 + ML-DSA-65 — both must fall. ML-DSA-65 is
    NIST FIPS 204 standardized; algorithm agility is built into the
    `HybridSignature` proto for future migrations.

11. **Cross-region pubkey uniqueness NOT enforced**. By design — US and
    EU registries are distinct authorities. Federation peers must
    reconcile across regions themselves. Document for federation peers.

12. **gRPC RegistryService rate-limit gap** (AV-9). Unauthenticated
    public RPCs have no per-IP limiting at the application layer.
    Closure: v1.2.x P0 — apply rate limiter to gRPC paths.

---

---

## 9. Threat Posture Summary

```
v1.2 BASELINE THREAT POSTURE
  (shipped 2026-05-01: ciris-crypto v1.8.0 alignment + project namespace)

  ✓ MITIGATED (15)
    AV-1   project≠ciris-agent acceptance (commit 254a89e — closes GH #1)
    AV-2   self-custody challenge replay (single-use atomic consume)
    AV-3   cross-org pubkey reuse (public_key_hash UNIQUE — migration 020)
    AV-7   HTTP body-size flood (1 MiB RequestBodyLimitLayer)
    AV-8   X-Forwarded-For spoofing (trusted-proxy CIDR allowlist)
    AV-9   gRPC + HTTP unauthenticated rate-limit gap (v1.3 Phase 2 —
           middleware/rate_limit.rs tower Layer + axum from_fn; tier
           mapping per classify_tier; 60s cleanup ticker)
    AV-15  PortalService cross-org access (v1.3 Phase 4 — claims_from_request
           + authorize_org_access on ~30 handlers; W3 inner-proto vigilance;
           batch RPCs reject mismatched per-record org_id)
    AV-16  cross-path pubkey poisoning (UNIQUE constraint table-wide)
    AV-19  multi-replica migration race (sqlx pg_advisory_lock)
    AV-22  custodied private key in logs (no log/DB write paths)
    AV-26  uploaded BuildManifest hybrid-sig verification (v1.3 Phase A —
           POST /v1/verify/build-manifest + trusted_primitive_keys table +
           admin RPCs; vendored BuildManifest type + ciris-crypto verifiers)
    AV-27  vault-mode dummy-key bug (commit 70f737d — closed by deletion)
    AV-32  self-custody activation race (per-key challenge + sig verify)
    AV-33  Spock multi-master migration desync (v1.3 Phase 1 — db/mod.rs
           Spock detection at boot; closes Registry#2)
    AV-35  audit-trail actor forgeability (v1.3 Phase 5 — claims.sub
           replaces req.requester_user_id in audit-writing handlers;
           create_audit_entry validates non-admin actor matches JWT subject)

  ⚠ CRITICAL — REQUIRES v1.2.x HOTFIX (0)
    None outstanding. AV-27 closed in v1.2.

  ⚠ HIGH-PRIORITY GAP — REQUIRED FOR v1.3 (0)
    None outstanding. AV-15 closed in v1.3 Phase 4.

  ⚠ OPERATIONALLY MITIGATED; IN-APP FIX TRACKED (1)
    AV-28  ephemeral signing keys in production — bridge ansible
           deployed persistent seeds 2026-05-01 evening (US + EU);
           in-app boot hard-fail still tracked for v1.3.x

  ⚠ ARCHITECTURAL DEFERRALS — v1.4 ROADMAP (7)
    AV-12  HS256 → asymmetric JWT (EdDSA / RS256 + JWKS)
    AV-13  REGISTRY_ADMIN_TOKEN → scoped short-lived JWT
    AV-14  binary admin role → scoped sub-roles (W2 PortalService
           god-mode gating closed in v1.3 Phase 6; sub-roles deferred)
    AV-25  HSM mode via ciris-keyring::HardwareSigner
    AV-29  hard-fail on plaintext PG in production
    AV-30  REGISTRY_ADMIN_TOKEN rotation lifecycle (closes with AV-13)
    AV-34  build-signing key compromise (cosign on uploads + GitHub
           OIDC chain + multi-party signing for high-stakes primitives;
           defense-in-depth via two-secret co-requirement today)

  ⚠ ARCHITECTURAL DEFERRALS — v1.4.x ROADMAP (2)
    AV-18  audit-log append-only + hash chain + Rekor mirror
    AV-25  steward-key threshold signing (M-of-N)

  ⚠ LOWER-PRIORITY HARDENING (10)
    AV-4   build-attestation Sigstore-style verify at registration
    AV-5   hard-fail server boot on no-TLS in non-memory mode
    AV-6   replace ON CONFLICT DO UPDATE with explicit Replace RPC
    AV-10  emergency-shutdown M-of-N + webhook fanout
    AV-11  mass-revoke list cap + webhook fanout
    AV-17  SUPPORTED_MANIFEST_VERSIONS allowlist + typed deserialize
    AV-20  RegistryError::sanitize_for_response()
    AV-23  per-caller rate limit on GetOrgUserByEmail (closes with AV-15)
    AV-24  constant-response-time wrapper on verify endpoints (research)
    AV-31  env-gate gRPC reflection in production

  CARGO AUDIT
    [run `cargo audit` and record result here per release]
```

The v1.2 baseline closed the v1.1 critical (AV-27) and the in-flight
project namespace gap (AV-1) by deletion / shipping. v1.3 Phase 1
closed AV-33 in-repo (Spock multi-master migration desync). v1.3 Phase
2 closed AV-9 (gRPC + HTTP rate limiting). v1.3 Phase A closed AV-26
(uploaded BuildManifest hybrid-sig verification — federation primitives
can now upload manifests that the registry actually trust-anchors).
v1.3 Phase A.1 wired CI self-publication and operationally validated
the trust chain end-to-end on every push to main since `cd95a9f`. v1.3
Phase 4 closed AV-15 (PortalService cross-org access). **All v1.3
high-priority in-app gaps are now closed.**

**Recursive golden rule operational**: the registry uses the same
trust mechanism for its own builds that it provides for federation
peers. Every push to main signs the registry's binary manifest with
the per-primitive build-signing key (GHA secret), uploads via
`POST /v1/verify/build-manifest`, and is verified against the
trusted-key registered for `project='ciris-registry'` (fingerprint
`567513a0c139412b...`). The CI publish step doubles as a continuous
integration test for the federation substrate — regressions in the
trust chain (auth, sig verification, cross-region replication, GET
round-trip) surface as red CI on the next push. This is the
"recursive golden rule" from PoB §1 made operational, not just
architectural.

**New surface introduced** by self-publication: AV-34 (build-signing
key compromise). Mitigated today via two-secret co-requirement (admin
token + signing key); v1.4 hardenings (cosign, multi-party signing,
SLSA attestation) will tighten this further.

The cross-org access gap (AV-15) means the deployment posture still
assumes either single-tenant operation or operator-trusted JWT
issuance. Multi-tenant production should not deploy without
compensating controls at the deployment edge (JWT issuance restricted
to scoped per-org identities).

One operational vector remains exposed to fresh deployments that don't
go through the bridge runbook:
- AV-28: persistent steward seeds are loaded via `ED25519_KEY_PATH` and
  `MLDSA_KEY_PATH`; ansible role's `runbooks/registry-keys-init.yml`
  generates and mirrors them across regions. In-app boot hard-fail
  still tracked for v1.3.x.

---

## 10. Update Cadence

This document is updated:
- On every minor version (v1.1.x → v1.2.0): comprehensive review.
- On every published security advisory affecting deps: addendum + cargo-audit re-run.
- On every new RPC or HTTP endpoint added: §3 review for new vectors.
- On every signing-key rotation or HSM/Vault config change: §5 / §6 review.
- On every protocol-version bump: §3.4 schema-mismatch review.

Last updated: 2026-05-01 (v1.3 hardening waterfall complete — all 7
phases shipped; 15/32 vectors mitigated, 0 high-priority in-app gaps,
0 critical bugs).

Phase ledger:

- Phase 1 (`afde1c6` + hotfix `6cf564f`): AV-33 — Spock multi-master
  migration desync closed in-repo.
- Phase 2 (`80adc79`): AV-9 — gRPC + HTTP rate limiting wired with
  per-IP tier mapping.
- Phase 3 (`0db3faa`): auth/policy module foundation
  (`middleware/authz.rs`).
- Phase A (`4adc224`): AV-26 — uploaded BuildManifest hybrid-sig
  verification via `POST /v1/verify/build-manifest`.
- Phase A.1 (`cd95a9f` / continuing): CI self-publication operational
  on every push to main; recursive golden rule made executable.
- Phase 4 (`16deaa5`): AV-15 — PortalService cross-org access closed
  via per-handler `claims_from_request` + `authorize_org_access`
  preamble on ~30 handlers; W3 inner-proto vigilance; batch-RPC
  cascading authz.
- Phase 5 (`0226840`): AV-35 — audit-actor forgeability (W1) closed
  by deriving actor from `claims.sub` instead of forgeable
  request-body fields.
- Phase 6 (`4ddb64d`): AV-14 W2 — SYSTEM_ADMIN gating on 13
  PortalService god-mode methods.
- Phase 7: handler-convention docs in CLAUDE.md + threat-model
  finalization (this commit).

New ledger entries from operational hardening:

- AV-28 (ephemeral signing keys in production): operationally
  mitigated via bridge ansible (`runbooks/registry-keys-init.yml`)
  on 2026-05-01 evening; persistent steward seeds mounted in both
  regions.
- AV-34 (build-signing key compromise): surface introduced by
  self-publication; mitigated today via two-secret co-requirement
  defense-in-depth; v1.4 hardenings (cosign + GitHub OIDC chain +
  multi-party signing + SLSA attestation) tracked.

Verify-side trust contract documented at `docs/TRUST_CONTRACT.md`
(closes CIRISRegistry#5 §1). Handler convention codified in
`CLAUDE.md` (Phase 7).
