# CIRISLens Threat Model

**Status:** baseline (audit performed 2026-05-01 against `main` at the
ciris-persist v0.1.2 integration boundary; pre-Engine-cutover code).
Updated each minor release.
**Audience:** lens operators, federation peers, security reviewers.
**Companion:** [`SECURITY.md`](SECURITY.md), [`CLAUDE.md`](../CLAUDE.md),
[`api/`](../api/).
**Inspired by:** [`CIRISVerify/docs/THREAT_MODEL.md`](https://github.com/CIRISAI/CIRISVerify/blob/main/docs/THREAT_MODEL.md)
(structural template) and
[`CIRISPersist/docs/THREAT_MODEL.md`](https://github.com/CIRISAI/CIRISPersist/blob/main/docs/THREAT_MODEL.md)
(adjacent ingest substrate).

---

## 1. Scope

### What CIRISLens Protects

CIRISLens is the observability layer for the CIRIS ecosystem. It
ingests Ed25519-signed reasoning traces from agents, scrubs PII,
persists to TimescaleDB, and exposes Grafana dashboards + an admin
console. It protects:

- **Corpus integrity at ingest**: every persisted trace was provably
  produced by the claimed agent, OR is rejected. The Coherence
  Ratchet and Federated Ratchet (PoB §2.4) measurements depend on
  this — forged traces in the corpus would degrade Sybil resistance.
- **Privacy boundary by trace tier**: `generic` traces contain no
  text by design; `detailed` and `full_traces` route through a
  scrubber boundary that strips NER-detected entities + regex-
  matched secrets before the row is persisted (CLAUDE.md "PII
  Scrubbing for Full Traces").
- **Cryptographic provenance of scrubbing** (full_traces only,
  today): the scrub envelope (`original_content_hash`,
  `scrub_signature`, `scrub_key_id`, `scrub_timestamp`) preserves
  agent-side provenance even after the lens mutates content.
- **Authenticated administration**: the admin UI, key-registration,
  and configuration surfaces are gated behind Google OAuth restricted
  to `@ciris.ai` (`ALLOWED_DOMAIN` env var).
- **Idempotency on agent retries**: the dedup key `(trace_id, ts)` on
  `accord_traces` (legacy path) and the post-cutover persist crate's
  `(agent_id_hash, trace_id, thought_id, event_type, attempt_index,
  ts)` (post-cutover path) make replays safe.
- **DSAR self-service deletion**: agents can delete their own traces
  by submitting a signature-verified delete request keyed to the
  agent's signing key (`/api/v1/accord/dsar/delete`).

### What CIRISLens Does NOT Protect

- **Agent-side key compromise** (parallels CIRISPersist AV-2). A
  stolen-but-valid signing key produces signatures the lens cannot
  distinguish from legitimate ones at write time. Detection is
  statistical via N_eff drift over time (PoB §2.4 + §5.6).
- **Network-edge TLS / certificate infrastructure**. HTTPS
  termination is the deployment edge's concern (nginx / ALB at
  `agents.ciris.ai`).
- **Postgres-server compromise**. Row-level write access bypasses
  the entire ingest pipeline.
- **Grafana RBAC**. Grafana enforces its own auth (Google OAuth
  upstream); dashboard-level access control is Grafana's job.
- **Audit-chain re-verification across peers**. The agent's
  per-action audit anchor (FSD §3.2) is captured but not yet
  cross-validated against the agent's local audit log. Phase 2
  peer-replicate (CIRISPersist FSD §4.5) closes this.
- **Pre-cutover corpus quality**. `accord_traces` retains
  pre-cutover history with whatever properties the legacy ingest
  pipeline gave it.
- **Data exfiltration via Grafana queries**. A logged-in operator
  with dashboard access can export everything they can see.
  Operator authorization is the lens's policy, not its enforcement.

---

## 2. Adversary Model

### Adversary Capabilities

The adversary is assumed to have:

- **Full source-code access** (AGPL-3.0, public).
- **Ability to mint arbitrary Ed25519 keypairs** and sign anything.
- **Network access to the lens HTTP endpoint** including arbitrary
  bytes to `/api/v1/accord/events` and any unauthenticated route.
- **Ability to run their own agents** on the network and request
  registration via `POST /api/v1/accord/public-keys`.
- **Replay capability**: capture any in-transit batch and re-send it.
- **Active MITM** between agent and lens if TLS is misconfigured.
- **Side-channel observation**: response timing, HTTP status codes,
  error message bodies.
- **Ability to read public CI artifacts**: every published image,
  the requirements.txt dep tree, the Dockerfile.
- **Compute sufficient for classical cryptography** (but not for
  breaking Ed25519).
- **Ability to entice an authenticated operator's browser** into
  visiting attacker-controlled URLs (CSRF surface).

### Adversary Limitations

The adversary is assumed to NOT have:

- **The ability to break Ed25519** within polynomial time on
  classical hardware.
- **Compromised the public-key directory** (`accord_public_keys`).
- **Compromised the lens's scrub signing key**.
- **Compromised the Postgres backend** that the lens writes to.
- **Compromised the Google OAuth infrastructure** (token-exchange
  flow, hd-claim issuance).
- **Compromised the operator's browser or workstation** beyond the
  CSRF level (full XSS in a same-origin context defeats the OAuth
  session model entirely; that's out of scope).
- **Physical access** to the lens deployment hardware.
- **Quantum compute** capable of breaking Ed25519 today (tracked in
  §8 Residual Risks; PoB §6 hybrid-PQC is Phase 2+).

---

## 3. Trust Boundaries

```
┌────────────────────────────────────────────────────────────────────┐
│ DEPLOYMENT EDGE (nginx / ALB at agents.ciris.ai)                   │
│ Responsibility: TLS termination, X-Forwarded-* headers, rate caps  │
└──────────────────────────┬─────────────────────────────────────────┘
                           │ HTTP (decrypted by edge)
                           ▼
┌────────────────────────────────────────────────────────────────────┐
│ FastAPI worker process (uvicorn, container uid 1000)               │
│   Trust boundary 1: untrusted bytes from agents enter here         │
│   Trust boundary 2: operator browser → admin/* via OAuth session   │
│                                                                    │
│   ┌──────────────────────────┐    ┌──────────────────────────┐     │
│   │ /api/v1/accord/events    │    │ /api/admin/*             │     │
│   │ (signature-authed)       │    │ (cookie-authed, OAuth)   │     │
│   │                          │    │                          │     │
│   │ verify → scrub → persist │    │ session = sessions[id]   │     │
│   └─────────────┬────────────┘    └─────────────┬────────────┘     │
│                 │                                │                  │
│                 ▼                                ▼                  │
│   ┌──────────────────────────────────────────────────────┐         │
│   │ asyncpg pool → Postgres (cirislens DB)               │         │
│   │   Trust boundary 3: SQL is parameterized; backend    │         │
│   │   is trusted at the row level                        │         │
│   └──────────────────────────────────────────────────────┘         │
└────────────────────────────────────────────────────────────────────┘
                           │
                           ▼
                ┌─────────────────────────┐
                │ Postgres + TimescaleDB  │
                │ (cirislens-db)          │
                └─────────────────────────┘
```

**Explicit non-boundary**: the FastAPI worker process and Grafana
share `cirislens` DB credentials. Grafana reads with the same role
the API writes; lens-side query authorization (which agent's traces
which operator can see) is enforced in dashboard logic, not via
Postgres roles.

---

## 4. Attack Vectors

Twenty-four vectors organized by adversary goal. Each lists the
attack, the primary mitigation present today, the secondary
mitigation, and the residual risk.

### 4.1 Identity / Forgery — adversary wants their bytes counted as real evidence

#### AV-1: Forged trace from attacker-minted key

**Attack**: Attacker generates a fresh Ed25519 keypair, signs a
synthetic CompleteTrace, submits to `/api/v1/accord/events`.

**Mitigation**: `accord_api.py:1108 load_public_keys()` queries the
`cirislens.accord_public_keys` directory before signature
verification. Unknown `signature_key_id` → `Unknown signer key:`
error → trace rejected. Worker-local cache miss falls back to a
single targeted DB lookup at `accord_api.py:1911-1932` so a key
registered on worker A is verifiable on worker B without process
restart.

**Secondary**: per-agent N_eff drift over time. A fresh-keyed
"agent" with no behavioral history fails the σ-decay floor before
it earns federation standing.

**Residual**: an attacker who registers a key id (via the
unauthenticated `POST /api/v1/accord/public-keys` endpoint —
**see AV-19 below**) earns the same gate as any agent. PoB §2.1
cost-asymmetry is the federation-level mitigation.

#### AV-2: Forged trace from compromised legitimate key

**Attack**: Attacker exfiltrates an honest agent's signing key
(secrets-manager compromise, key-material leak, social engineering),
signs a malicious trace under that agent's identity.

**Mitigation**: **Out of CIRISLens's protection scope at write
time.** The verifier cannot distinguish stolen-key from legitimate.
The agent's audit-log chain (anchor captured on every
`ACTION_RESULT` per CLAUDE.md "Coherence Ratchet") provides
post-incident forensics; Phase 2 peer-replicate verifies the chain.

**Residual**: undetectable at ingest until peer-replicate. The
agent's own CIRISVerify hardware-backed key storage is the
upstream mitigation.

#### AV-3: Replay of captured legitimate batch

**Attack**: Network MITM (or re-submission attack at the API)
captures a valid signed batch, replays it.

**Mitigation**: idempotency. Pre-cutover `accord_traces` keys on
`(trace_id, timestamp)` with `ON CONFLICT DO NOTHING`. Post-cutover
ciris-persist keys on `(agent_id_hash, trace_id, thought_id,
event_type, attempt_index, ts)`.

**Secondary**: TLS at the deployment edge prevents in-flight
capture; not the lens's responsibility.

**Residual**: a batch replayed against a *different* lens
deployment (federation peer) lands once by design — that's
replication, not corruption.

#### AV-4: DSAR replay attack

**Attack**: `/api/v1/accord/dsar/delete` validates an Ed25519
signature over the canonical JSON of `{agent_id_hash, request_type,
requested_at}`. The handler at `accord_api.py:2533` verifies the
signature came from the agent's own key, then deletes all traces
matching that key. The signed payload contains `requested_at` but
**this field is not validated against current time** — a captured
delete request can be replayed indefinitely. Each replay deletes
any traces the agent has re-ingested since the last replay,
producing recurring availability damage to the victim agent.

**Mitigation in baseline**: **Partial.** Signature gate prevents
forgery by other parties. No nonce / no time-window check on
`requested_at`.

**Recommended for next minor**: `accord_api.py:2533` reject
requests where `abs(now - requested_at) > 5 minutes` (clock-skew
window) AND track previously-honored `(agent_id_hash, requested_at)`
tuples in a 24h window to refuse repeats.

**Residual**: until time-window + per-tuple replay block lands,
captured DSAR requests are reusable.

#### AV-5: Canonicalization-mismatch on agent signature

**Attack**: Adversary exploits a byte-difference between what the
agent canonicalizes and what the lens canonicalizes.

**Mitigation**: the verifier (`api/accord_api.py` +
`cirislens_core.process_trace_batch`) reconstructs canonical bytes
deterministically. Real-fixture round-trips against
CIRISAgent `release/2.7.8` traces (per
[`CIRISPersist/tests/fixtures/wire/2.7.0/`](../../CIRISPersist/tests/fixtures/wire/2.7.0/))
exercise the actual agent shapes.

**Residual**: float-formatting drift across Python versions
(see CIRISPersist threat model AV-4 residual). The lens shares the
same exposure and the same closure path.

### 4.2 Authentication / Authorization — adversary wants admin access without credentials

#### AV-6: Mock OAuth bypass triggers on default environment

**Severity: P0 (production-blocking)**

**Attack**: `api/main.py:136 OAUTH_CLIENT_ID = os.getenv("OAUTH_CLIENT_ID", "mock-client-id")`.
The login flow at `api/main.py:1683` checks
`if OAUTH_CLIENT_ID == "mock-client-id":` and **auto-authenticates
the caller as `dev@ciris.ai`** with a 24-hour session — no
verification of any kind. If the production deployment ships
without setting `OAUTH_CLIENT_ID`, every visitor to
`/api/admin/auth/login` gains admin access.

**Mitigation in baseline**: **None.** The `IS_PRODUCTION` flag
gates docs/redoc URLs and cookie `secure=` but does NOT gate the
mock-OAuth branch.

**Recommended hot-fix**: `IS_PRODUCTION and OAUTH_CLIENT_ID == "mock-client-id"`
must raise at startup. Cost: 5 minutes; touch `api/main.py:135-145`.

```python
if IS_PRODUCTION and OAUTH_CLIENT_ID == "mock-client-id":
    raise SystemExit("OAUTH_CLIENT_ID required in production (got default)")
```

**Residual**: until the gate lands, deployment misconfiguration
becomes anonymous-admin.

#### AV-7: OAuth state parameter not validated (CSRF on login)

**Severity: P0**

**Attack**: `api/main.py:1700-1701` constructs the Google OAuth
authorization URL with no `state=` parameter. The callback at
`api/main.py:1706` accepts `state: str | None = None` and never
validates it. An attacker can complete an OAuth dance on their own
device, then craft a callback URL with their access code and
trick a victim's browser into hitting it — the victim ends up
logged in as the attacker.

**Mitigation in baseline**: **None.** No state generation, no
state verification.

**Recommended hot-fix**: `secrets.token_urlsafe(32)` at login;
store in a short-lived cookie keyed `oauth_state`; verify on
callback before token-exchange. Cost: ~1 hour.

**Residual**: until landed, the admin surface has standard OAuth
CSRF.

#### AV-8: No CSRF protection on cookie-authed POST/PUT/DELETE

**Severity: P1**

**Attack**: Admin endpoints (`/api/admin/managers`,
`/api/admin/telemetry/{agent_id}`, `/api/admin/visibility/{agent_id}`,
`/api/admin/configurations`, etc.) accept POST/PUT/DELETE with
session cookies as the only auth. No CSRF token, no Origin/Referer
check. SameSite=Lax (default in `set_cookie`) blocks most
cross-site cookie carryover but **permits top-level GET POSTs and
form posts initiated from same-site contexts** — including any XSS
on a same-site Grafana dashboard or operator-installed browser
extension.

**Mitigation in baseline**: **Partial.** SameSite=Lax + the
restrictive CORS origins list (`api/main.py:55-63`) cuts most
cross-origin attacks. Within-origin CSRF (e.g., from a Grafana
plugin running at `agents.ciris.ai`) is not blocked.

**Recommended for next minor**: a per-session CSRF token (issued
on session creation, required as `X-CSRF-Token` header on every
mutating request).

**Residual**: same-site CSRF possible until the token lands.

#### AV-9: In-memory session store breaks under multi-worker deployment

**Severity: P1**

**Attack**: not strictly an attack — a deployment correctness
defect with security consequences. `api/main.py:146 sessions = {}`
is a per-process dict. The Dockerfile sets `WORKERS=4`. A user who
logs in on worker A and lands on worker B for the next request
appears unauthenticated; reactively logs in again, creating a
second session. **No cap on dict size; no eviction** — slow-burn
memory growth from session churn produces an OOM-style DoS over
weeks.

**Mitigation in baseline**: **None.** The `expires_at` check
deletes on read, but only on the worker that serves the read.

**Recommended**: move sessions into Redis (the deployment already
runs Redis per `requirements.txt`) or signed cookies.

**Residual**: until landed, multi-worker session UX is broken AND
memory grows unboundedly.

#### AV-10: Unauthenticated administrative writes via `/wbd/*`, `/pdma/*`, `/creator-ledger`, `/sunset-ledger`

**Severity: P1**

**Attack**: `accord_api.py:358 POST /wbd/deferrals`,
`accord_api.py:496 POST /pdma/events`,
`accord_api.py:650 POST /creator-ledger`,
`accord_api.py:769 POST /sunset-ledger` all accept request bodies
without any auth dependency (`Depends(require_auth)`) and without
signature verification. Anyone reachable to the API can write
arbitrary rows into the Accord-compliance-track tables.

**Mitigation in baseline**: **None.** The endpoints predate the
Engine cutover and were intended to be agent-fed via signed
batches at `/events`; the standalone routes never picked up an
auth gate.

**Recommended hot-fix**: either (a) require `Depends(require_auth)`
+ admin role on each, or (b) accept only via the signed-batch
`/events` path and remove the standalone routes.

**Residual**: until gated, anyone can populate the lens's
compliance ledger with arbitrary content. This is a *write-only*
exposure (no read-back of attacker bytes via these specific routes
without separate auth), but the Accord ledger is the lens's
testimony layer; corruption here directly damages downstream trust.

#### AV-11: Unauthenticated public-key registration

**Severity: P1**

**Attack**: `accord_api.py:2363 POST /api/v1/accord/public-keys`
accepts a `(key_id, public_key_base64)` pair from any unauthenticated
caller and inserts into `accord_public_keys`. The newly-registered
key can then be used to sign forged traces submitted to `/events`,
which now pass AV-1's directory check.

**Mitigation in baseline**: **None at the lens level.** The
deployment-edge proxy may rate-limit this route, but there's no
admission policy gate.

**Recommended**: register-via-handshake only — the agent's
`/accord/agents/register` flow validates an out-of-band token;
unauthenticated `POST /public-keys` becomes
admin-only or removed.

**Residual**: a high-throughput attacker can fill the directory
with attacker keys, each then admissable per AV-1. PoB §5.6
acceptance policy is the federation-level mitigation, but it's
out of scope for the lens.

#### AV-12: First-write-wins on public-key re-registration

**Severity: P2**

**Attack**: parallels CIRISPersist AV-11 — an attacker who
registers `signature_key_id="agent-foo"` first locks the legitimate
agent out (since `ON CONFLICT DO NOTHING` is the SQL pattern at
`accord_api.py:2395`).

**Mitigation in baseline**: first-write-wins is the *correct*
behavior under attack, but blocks legitimate rotation too. Lens
operators must `UPDATE accord_public_keys` manually for rotation.

**Recommended for v0.2.x**: the same explicit
`rotate_public_key(rotation_proof=signed-by-old-key)` API
CIRISPersist tracks for v0.2.x.

### 4.3 Denial of Service — adversary wants the lens unable to receive evidence

#### AV-13: Unbounded request body in middleware

**Severity: P0**

**Attack**: `api/main.py:74-89` registers a middleware that calls
`await request.body()` for every `POST /accord/events` request,
unconditionally, with no size limit. An attacker can submit a
multi-GB body and the middleware buffers it in memory before
handing off to the route handler. Multiple concurrent oversized
bodies → OOM kill of the worker. Repeated → service outage.

**Mitigation in baseline**: **None at the FastAPI layer.**
Deployment-edge proxy may cap, but defense-in-depth is missing.

**Recommended hot-fix**: gate the middleware on
`Content-Length > 10 MiB → 413 Payload Too Large` before reading
the body. The post-cutover ciris-persist v0.1.2 already enforces
`DefaultBodyLimit::max(8 MiB)` (see CIRISPersist AV-7); the lens
should match before the cutover lands.

**Residual**: until landed, body-size DoS is a direct exposure.

#### AV-14: No request rate limiting except on `/scoring/*`

**Severity: P1**

**Attack**: `api/scoring_api.py:108-163` implements per-IP rate
limiting (60 req/min). No equivalent on `/events`, `/dsar/delete`,
`/public-keys`, or any admin route. An attacker can flood the
verify-and-persist pipeline.

**Mitigation in baseline**: deployment-edge rate limiting (assumed
nginx). Not the lens's level.

**Recommended for v0.2.x**: extend the `RateLimiter` class to all
mutating routes, with operator-tunable thresholds via env var.

#### AV-15: `/events/debug` is unauthenticated and dumps request shape to logs

**Severity: P2**

**Attack**: `accord_api.py:1748 POST /events/debug` accepts any
body, parses it as JSON, and emits two `logger.warning` lines per
request — one with the top-level keys, batch_timestamp,
consent_timestamp, event count; the other with the first event's
top-10 keys + trace_id. No auth, no rate limit.

The logger format string at `accord_api.py:1756-1762` uses %-format
with attacker-controlled string values. Python `logging`'s
formatter does NOT interpret format specifiers in *args — safe
against format-string attacks — but **log-injection is possible
if the attacker embeds newlines in trace_id**, splitting one log
line into multiple and faking adjacent log entries.

**Mitigation in baseline**: **None.**

**Recommended**: gate behind admin auth (or remove — it's a debug
tool, not a production route). If kept, sanitize untrusted strings
with `.replace('\n','\\n').replace('\r','\\r')` before logging.

**Residual**: low operational impact; cleanup work.

#### AV-16: No `statement_timeout` on Postgres pool

**Severity: P2**

**Attack**: a slow query (operator-issued, attacker-influenced via
filter parameters on `/repository/traces`) runs without bound,
holding a pool connection until completion.

**Mitigation in baseline**: `asyncpg.create_pool(min_size=2,
max_size=10)` at `api/main.py:389` — only 10 simultaneous queries.
A handful of attacker-induced slow queries pin the pool.

**Recommended**: set `command_timeout=` on `create_pool` (asyncpg
honors this per-connection); also set
`server_settings={"statement_timeout": "30s"}` so the server
itself enforces. Cost: ~30 minutes.

**Residual**: until landed, query-DoS per pool-saturation.

### 4.4 Confidentiality / Privacy — adversary wants content text exposed at a tier where it isn't warranted

#### AV-17: Plaintext Postgres connection allowed by default

**Severity: P1**

**Attack**: `DATABASE_URL` env var is consumed verbatim by
`asyncpg.create_pool(DATABASE_URL, ...)` at `api/main.py:389`. If
the URL omits `?sslmode=verify-full`, asyncpg defaults to
`sslmode=prefer` — which downgrades to plaintext if the server
doesn't offer TLS. A network-adjacent attacker on the lens-Postgres
path can read every signed trace + every agent's public key
metadata.

**Mitigation in baseline**: **None at the lens level.** Production
deployments may set the DSN correctly; nothing enforces it.

**Recommended**: parse `DATABASE_URL` at startup; fail-fast if
`sslmode` is unset or `disable`/`allow`/`prefer`.

**Residual**: until landed, deployment-misconfiguration becomes
data exposure.

#### AV-18: Scrub signing key on disk via `_create_and_save_key`

**Severity: P2**

**Attack**: `api/pii_scrubber.py:447-458` generates a fresh
Ed25519 key when `CIRISLENS_SCRUB_KEY_PATH` doesn't exist. The
flow is `key_path.write_bytes(new_key)` (line 457) **then**
`key_path.chmod(0o600)` (line 458). Between write and chmod the
file is mode 0644 (default umask) — readable by other host users.
Containerized deployments mostly don't have other users, but a
shared-host or sidecar pattern leaks the seed.

**Mitigation in baseline**: chmod fires within microseconds; race
window is small.

**Recommended**: `os.umask(0o077)` at process start, OR open with
`os.O_WRONLY | os.O_CREAT | os.O_EXCL` and `mode=0o600` at fd
creation. Cleaner: switch to CIRISVerify named-key storage (the
v0.1.3 plan already has this scoped — see "What persist needs"
in the `CIRISLens` integration thread). Once that lands, the seed
never touches the filesystem.

**Forward-looking**: in the v0.1.3+ posture the scrub-signing key
becomes a *single-key, three-role* identity per CIRISPersist
threat model §1 — the same key is the deployment's scrub envelope
signer, its Reticulum destination address (`SHA256(pubkey)[..16]`,
PoB §3.2), and its registry-published identity. That means a
local-disk seed leak is not just "rows the lens signed are now
suspect" but also "the lens's federation transport address is
hijackable AND the registry entry needs revocation." This tripled
cost makes hardware-backed keyring entries materially stronger
than software seeds and is the operational reason the v0.1.3
cutover prioritizes CIRISVerify keyring storage over a Postgres-
table-of-bytes shape.

**Residual**: small write→chmod race window today; obsoleted by
v0.1.3 cutover.

#### AV-19: Detailed-tier traces are scrubbed but not signed

**Severity: P2**

**Attack**: lens currently signs only `full_traces` post-scrub
(`accord_api.py:2004-2019`). `detailed`-tier traces are scrubbed
(regex pass) but persisted with `pii_scrubbed=true` and no
signature envelope. A peer reading the corpus has the lens's
*claim* of scrubbing without cryptographic evidence of it.

**Mitigation in baseline**: trust-the-lens for `detailed` tier.

**Recommended**: extend the scrub envelope to all post-scrub levels
(this is the planned ciris-persist v0.1.3 work). Once landed,
every persisted row carries the four-tuple envelope unconditionally.

#### AV-20: PII leak via verbose log lines

**Severity: P2**

**Attack**: log statements throughout `accord_api.py` include
identifiers and partial content. Audit findings:

- `accord_api.py:1854 SCHEMA_VALID trace %s: version=%s event_types=%s`
  — `event_types` is a list of strings from agent input; bounded.
- `accord_api.py:1818 SCHEMA_INVALID trace %s: ... errors=%s` —
  `errors` from `validate_trace_schema` may include attacker-
  influenced field names.
- `accord_api.py:103-110 VALIDATION_ERROR ... batch_ts=%s consent_ts=%s events=%d errors=%s`
  — body is parsed and logged in 200-char-truncated form.
- `accord_api.py:2017 Scrubbed PII from full_traces %s (hash: %s...)`
  — only the first 16 chars of the SHA-256 hash; safe.

None of these emit raw payload content. **Audit pass: clean** for
the trace ingest path. Operator endpoints are different — the
`logger.warning("VALIDATION_ERROR ... errors=%s ...", str(exc.errors())[:200])`
at `api/main.py:103-110` includes Pydantic error context which can
contain field-value snippets. Cap at `[:200]` mitigates depth, not
content sensitivity.

**Recommended**: log validation errors as `error_type` only, not
`exc.errors()`. Move verbose form to debug-level logs gated on
operator opt-in.

#### AV-21: Cookie missing `secure=` on mock-OAuth path

**Severity: P2**

**Attack**: `api/main.py:1689` sets the session cookie without
`secure=IS_PRODUCTION`, while line 1754-1756 (real OAuth path)
sets it correctly. The mock path is dev-only — but if an operator
accidentally re-uses the dev branch in production (see AV-6), the
cookie is also transmittable over plaintext HTTP.

**Mitigation in baseline**: SameSite=Lax + httponly limit
exposure.

**Recommended**: harmonize — drop the mock-OAuth code path entirely
once AV-6's startup gate lands.

### 4.5 Supply Chain / Operational

#### AV-22: Stale pinned dependencies with known CVEs

**Severity: P1**

**Attack**: dep-tree audit (`api/requirements.txt`):

| Pin | Released | Known issues |
|---|---|---|
| `fastapi==0.104.1` | 2023-11 | Many releases since; minor CVEs in transitive starlette versions |
| `uvicorn[standard]==0.24.0` | 2023-10 | h11 / httptools fixes since |
| `pydantic[email]==2.5.0` | 2023-11 | Several point-release fixes since |
| `python-jose[cryptography]==3.3.0` | 2021-06 | **CVE-2024-33663** (algorithm confusion), **CVE-2024-33664** (DoS via large JWE) |
| `httpx==0.25.1` | 2023-10 | Multiple bug-fix releases since |
| `passlib[bcrypt]==1.7.4` | 2020-10 | Stagnant; bcrypt upstream has had compat fixes |

`python-jose` is the most material — even if the lens doesn't
verify external JWTs today, it's in the dep tree and can be
recruited by any future code change.

**Mitigation in baseline**: deps are pinned (good) but stale.

**Recommended**: monthly `pip-audit` against the pinned set; bump
all deps to current minor on a regular cadence. Specifically
remove `python-jose` if unused, or replace with `pyjwt`.

#### AV-23: Container base image not digest-pinned

**Severity: P3**

**Attack**: `api/Dockerfile:6 FROM python:3.11-slim` resolves to
whatever the registry serves at build time. A poisoned base image
(supply-chain compromise of `python:3.11-slim`) gets baked into
the lens image without detection.

**Mitigation in baseline**: trust Docker Hub.

**Recommended**: pin to a digest (`python:3.11-slim@sha256:...`)
and update via PR with hash review. Same for `maturin>=1.4` →
pin to `==1.x.y`.

#### AV-24: `--proxy-headers` without `--forwarded-allow-ips`

**Severity: P3**

**Attack**: `api/Dockerfile` CMD launches uvicorn with
`--proxy-headers` but no `--forwarded-allow-ips`. uvicorn defaults
to trusting only `127.0.0.1` for X-Forwarded-* — which means
production deployments behind a non-loopback nginx (most
deployments) silently get **no X-Forwarded-For honored**, so the
lens sees the proxy IP as the client IP for rate-limit accounting,
log correlation, etc.

Conversely, if an operator sets `--forwarded-allow-ips=*`, any
proxy-injected client can spoof the X-Forwarded-For header.

**Mitigation in baseline**: default-secure (only loopback
trusted); operational consequence is degraded telemetry, not a
security exposure per se.

**Recommended**: explicit `--forwarded-allow-ips=<edge-proxy-IP>`
in production deploy script; document in
[`PRODUCTION_DEPLOYMENT.md`](PRODUCTION_DEPLOYMENT.md).

#### AV-25: Manager-collector SSL_VERIFY env-toggle

**Severity: P3**

**Attack**: `api/manager_collector.py:140
async with httpx.AsyncClient(timeout=10.0, verify=SSL_VERIFY)`
where `SSL_VERIFY` is read from env (`"true"` default). Setting
`SSL_VERIFY=false` disables certificate validation for outbound
calls to manager + agent endpoints. A MITM attacker between the
lens and the agent can serve forged data, which lands in
`agent_metrics` / `agent_logs` / `agent_traces` tables.

**Mitigation in baseline**: defaults to true.

**Recommended**: remove the env-disable path — production should
never ship with cert validation off. If dev needs self-signed
support, document the per-host CA-bundle override (`SSL_CERT_FILE`
env var) instead.

---

## 5. Mitigation Matrix

| AV | Attack | Severity | Primary Mitigation | Status | Fix tracker |
|---|---|---|---|---|---|
| AV-1 | Forged trace from attacker key | — | Public-key directory lookup | ✓ Mitigated | — |
| AV-2 | Forged trace from compromised key | — | (out of scope at write time) | ⚠ Phase 2 closes | CIRISPersist FSD §4.5 |
| AV-3 | Replay of legitimate batch | — | Idempotency on dedup key | ✓ Mitigated | — |
| AV-4 | DSAR replay attack | P1 | Signature gate (no nonce) | ⚠ **Open** | next minor |
| AV-5 | Canonicalization mismatch | — | Real-fixture parity tests | ✓ Mitigated | — |
| AV-6 | Mock OAuth on default env | **P0** | Startup gate: `IS_PRODUCTION + OAUTH_CLIENT_ID==mock-client-id` → SystemExit | **✓ Mitigated** | api/main.py:46-55 |
| AV-7 | OAuth state CSRF | **P0** | `secrets.token_urlsafe(32)` issued + `oauth_state` cookie + `secrets.compare_digest` on callback | **✓ Mitigated** | api/main.py oauth_login + oauth_callback |
| AV-8 | CSRF on cookie-authed mutating routes | P1 | SameSite=Lax + CORS allowlist | ⚠ **Open** | next minor |
| AV-9 | In-memory sessions break across workers | P1 | (none — broken UX + memory leak) | ⚠ **Open** | next minor |
| AV-10 | Unauthenticated `/wbd /pdma /creator-ledger /sunset-ledger` writes | P1 | (none) | ⚠ **Open** | hot-fix |
| AV-11 | Unauthenticated `/public-keys` POST | P1 | (none) | ⚠ **Open** | hot-fix |
| AV-12 | First-write-wins on key re-register | P2 | Manual UPDATE for rotation | ⚠ Track | v0.2.x |
| AV-13 | Unbounded body in middleware | **P0** | `MAX_INGEST_BODY_BYTES = 8 MiB` Content-Length gate before `await request.body()` (parity with CIRISPersist DefaultBodyLimit) | **✓ Mitigated** | api/main.py cache_request_body |
| AV-14 | No rate limit (except `/scoring/*`) | P1 | Edge-proxy rate caps | ⚠ Track | next minor |
| AV-15 | `/events/debug` open + log-injection | P2 | (none) | ⚠ Open | next minor |
| AV-16 | No statement_timeout on PG pool | P2 | Pool size cap | ⚠ Open | next minor |
| AV-17 | Plaintext PG connection allowed | P1 | (DSN-dependent) | ⚠ **Open** | hot-fix |
| AV-18 | Scrub key write→chmod race | P2 | chmod 0600 fires fast | ⚠ Open | v0.1.3 obsoletes (CIRISVerify keyring) |
| AV-19 | Detailed-tier scrubbed but unsigned | P2 | Trust-the-lens | ⚠ Open | v0.1.3 obsoletes |
| AV-20 | PII leak via verbose logs (operator routes) | P2 | Truncation to 200 chars | ⚠ Open | next minor |
| AV-21 | Cookie missing `secure=` on mock path | P2 | SameSite + httponly + (now) `secure=IS_PRODUCTION` on mock branch too | **✓ Mitigated** | api/main.py oauth_login mock branch |
| AV-22 | Stale pinned deps + `python-jose` CVEs | P1 | Pinning (good) but stale | ⚠ **Open** | dep-bump PR |
| AV-23 | Base image not digest-pinned | P3 | (Docker Hub trust) | ⚠ Open | dep-bump PR |
| AV-24 | uvicorn proxy-headers without allow-ips | P3 | Default-secure (loopback only) | ⚠ Document | deploy doc |
| AV-25 | Manager SSL_VERIFY env-toggle | P3 | Default true | ⚠ Open | next minor |

**P0 hot-fix bundle** (AV-6, AV-7, AV-13): three changes, ~3 hours
total work, blocks production deploy of any new release until
landed.

**P1 next-minor bundle** (AV-4, AV-8, AV-9, AV-10, AV-11, AV-14,
AV-17, AV-22): roughly two days of focused work; brings the
posture in line with the CIRISPersist v0.1.2 baseline.

---

## 6. Security Levels by Component

| Component | Trust tier | Rationale |
|---|---|---|
| `/api/v1/accord/events` (POST) | Public, signature-authed | Agent-facing; auth IS the signature |
| `/api/v1/accord/dsar/delete` | Public, signature-authed | Agent-facing self-service |
| `/api/v1/accord/public-keys` (POST) | **Should be admin-authed; currently public** | AV-11 |
| `/api/v1/accord/wbd /pdma /creator-ledger /sunset-ledger` | **Should be agent-signature or admin; currently public** | AV-10 |
| `/api/v1/accord/repository/traces` (GET) | Public read of stored traces | Trace-level partner_access scoping; lens policy |
| `/api/v1/accord/coherence-ratchet/*` | Mixed (run is admin; alerts are public read) | — |
| `/api/v1/scoring/*` | Public (rate-limited) | Operator dashboard data |
| `/api/admin/*` | Cookie-authed via OAuth | Operator-only |
| `/api/admin/auth/*` | Public (OAuth flow) | Login surfaces |
| `/health` | Public | Liveness |
| `/v1/status` | Public | Service health |

Critical invariant: **the signed-trace ingest path** (`/events`,
`/dsar/delete`) **does not depend on cookie auth**. The signature
IS the authentication, mediated by the public-key directory.
Public-key registration (the gate that admits a signature key into
the directory) is the load-bearing access-control surface — and
that surface is currently unauthenticated (AV-11).

---

## 7. Security Assumptions

The system depends on these assumptions; if violated, the threat
model breaks.

1. **Lens deployment hardware integrity**: the host is not
   compromised at root. Postgres data files, the scrub signing key
   on disk, and process memory are trusted.
2. **TLS at the deployment edge**: the lens fronts itself with
   HTTPS termination. Plaintext HTTP exposes session cookies and
   trace bodies.
3. **Edge-proxy rate limiting**: the deployment edge caps
   per-source-IP request rates. The lens does not (except
   `/scoring/*`).
4. **Edge-proxy body-size cap**: the deployment edge caps body
   size. The lens does not at the FastAPI layer (AV-13).
5. **Postgres write-quorum**: the database accepts writes
   atomically.
6. **Clock accuracy**: timestamps are within ~5 minutes of real
   time. Skew degrades AV-3 (replay) and AV-4 (DSAR replay)
   mitigations once they land.
7. **Wire-format spec stability**: agents and lens agree on
   canonicalization conventions (CIRISPersist threat model AV-4
   covers drift detection).
8. **Operator OAuth domain integrity**: Google OAuth's `hd` claim
   is trustworthy. A compromised Google Workspace tenant for
   `@ciris.ai` defeats the operator gate.
9. **DATABASE_URL is set with `sslmode=verify-full`** in production.
   Today this is a deployment convention, not a startup-enforced
   gate (AV-17).
10. **`OAUTH_CLIENT_ID` is set in production**. Today this is a
    deployment convention; default value triggers the mock-OAuth
    bypass (AV-6).

Critical invariant: **assumptions 9 and 10 must become
startup-enforced gates before the next production deploy**,
matching CIRISPersist's "fail-fast on misconfiguration" stance
(its mission anti-pattern #2: never silent acceptance).

---

## 8. Fail-Secure Degradation

All failures should degrade to MORE restrictive modes, never less.

| Failure | Current behavior | Should be |
|---|---|---|
| Schema parse failure | HTTP 422; rejected to `malformed_traces` | ✓ Correct |
| Schema-version unsupported | HTTP 422 with detail | ✓ Correct |
| Signature verification failure | HTTP 422; trace rejected | ✓ Correct |
| Unknown signing key (after worker-cache miss recovery) | HTTP 422 | ✓ Correct |
| Scrubber failure (v1 errored) | trace rejected with audit log | ✓ Correct |
| Postgres unreachable | HTTP 503 | ✓ Correct (no journal in v1; persist v0.1.x adds redb) |
| Mock-OAuth on production | **startup-fatal** (AV-6 closed) | ✓ Correct |
| `OAUTH_CLIENT_ID` missing in production | **startup-fatal** (AV-6 closed) | ✓ Correct |
| `DATABASE_URL` without sslmode | plaintext PG (AV-17) | ✗ MUST be startup-fatal |
| Body > 8 MiB at middleware | **HTTP 413 + max_bytes** (AV-13 closed) | ✓ Correct |
| Session-store full | dict grows unboundedly (AV-9) | ✗ MUST evict / cap |

Five rows are AVs in §4. They convert from "open exposures" to
"fail-secure invariants" once the P0/P1 bundles land.

---

## 9. Residual Risks

Risks the lens mitigates but cannot fully eliminate.

1. **Compromised agent signing key** (AV-2). Federation N_eff
   drift over time is the statistical mitigation; agent-side
   CIRISVerify hardware-backed storage is the upstream control.

2. **Quantum compromise of Ed25519**. Tracked alongside CIRISPersist
   §8 #2; PoB §6 PQC-hybrid is Phase 2+.

3. **Same-host attacker reading scrub key file** (AV-18). Closes
   when CIRISVerify named-key storage lands in v0.1.3.

4. **Operator browser XSS** at `agents.ciris.ai`. Defense-in-depth
   via CSP headers (not currently set) is the control; out of
   scope for the lens process today.

5. **Postgres compromise**. Out of CIRISLens's protection scope.

6. **Operator account compromise**. Google Workspace MFA enforcement
   is the upstream control.

7. **Cross-tier data exfiltration via Grafana**. An authenticated
   operator with `/repository/traces` read access can export every
   trace they can see. Per-trace partner_access scoping limits
   default visibility; per-operator authz is dashboard-level.

8. **All federation peers compromised simultaneously** (PoB §5.1
   residual). Per Accord NEW-04, no detector is complete.

---

## 10. Posture Summary

```
P0 EXPOSURES — all closed in the 2026-05-01 hot-fix bundle
  ✓ AV-6   Startup gate: production + default OAUTH_CLIENT_ID → SystemExit
  ✓ AV-7   secrets.token_urlsafe(32) state binding + compare_digest on callback
  ✓ AV-13  8 MiB Content-Length gate (matches CIRISPersist DefaultBodyLimit)

P1 EXPOSURES (next minor; brings posture to CIRISPersist v0.1.2 parity)
  ⚠ AV-4   DSAR replay (no nonce / no time window)
  ⚠ AV-8   CSRF on cookie-authed mutating routes
  ⚠ AV-9   In-memory sessions break across workers
  ⚠ AV-10  Unauthenticated compliance-ledger writes
  ⚠ AV-11  Unauthenticated public-key registration
  ⚠ AV-14  No rate limit outside /scoring/*
  ⚠ AV-17  Plaintext Postgres connection allowed
  ⚠ AV-22  Stale deps + python-jose CVEs

P2 / P3 (track; low blast radius)
  ⚠ AV-12, AV-15, AV-16, AV-18, AV-19, AV-20, AV-21, AV-23, AV-24, AV-25

ARCHITECTURALLY MITIGATED
  ✓ AV-1   Forged trace from attacker key (directory lookup)
  ✓ AV-3   Replay of legitimate batch (idempotency)
  ✓ AV-5   Canonicalization mismatch (parity tests)

PHASE-2-CLOSES (architecturally deferred)
  ⚠ AV-2   Stolen-key forgery (peer-replicate audit chain)
```

**Bottom line**: the corpus-integrity story (AV-1, AV-3, AV-5) is
sound — the lens's trace-ingest pipeline does what the federation
needs it to do. The operator-facing surfaces (admin OAuth, CSRF,
session store, public-key registration, body-size, sslmode
enforcement) are below the bar set by CIRISPersist v0.1.2, and a
two-bundle remediation (P0 hot-fix + P1 next-minor) brings them
to parity before the persist-v0.1.3 cutover lands.

Three of the P2 items (AV-18, AV-19, the structural part of AV-21)
are obsoleted by the planned ciris-persist v0.1.3 work
(CIRISVerify named-key storage + always-on scrub-signing). They're
catalogued here for completeness but the closure path is
architectural rather than patch-shaped.

---

## 11. Update cadence

This document is updated:
- On every minor release: comprehensive review.
- On any published security advisory affecting deps: addendum
  in §4 + `pip-audit` re-run.
- On every wire-format schema-version bump: AV-5 review against
  the new shape.
- On every cross-trinity boundary change (CIRISAgent flips schema,
  CIRISPersist flips Engine surface, CIRISVerify flips keyring API):
  trust-boundary review + interaction matrix update.

Last updated: 2026-05-01 (baseline + P0 hot-fix bundle landed —
AV-6 / AV-7 / AV-13 closed in api/main.py; AV-21 also closes by
adding `secure=IS_PRODUCTION` to the mock-OAuth cookie branch.
14 new tests in tests/unit/test_main_security_p0.py; full unit
suite 323 passed).
