# CIRISProxy Threat Model

**Status:** v0.2.0 baseline. Updated each minor release.
**Audience:** lens team integrators, mobile-client reviewers, security reviewers.
**Companion:** [`README.md`](../README.md), [`CLAUDE.md`](../CLAUDE.md), [`docs/INTEGRATION_GUIDE.md`](INTEGRATION_GUIDE.md).
**Inspired by:** [CIRISBilling `THREAT_MODEL.md`](../../CIRISBilling/docs/THREAT_MODEL.md) and [CIRISPersist `THREAT_MODEL.md`](../../CIRISPersist/docs/THREAT_MODEL.md) — structural template and adversary-goal organization.

---

## 1. Scope

### What CIRISProxy Protects

CIRISProxy is the credit-gated LLM access layer for the CIRIS ecosystem. It mediates between mobile-app clients (CIRIS Agent Android / iOS) and upstream LLM providers (Groq, Together AI, OpenRouter, OpenAI). It protects:

- **Provider-API-key isolation**: client-facing surface never sees `GROQ_API_KEY`, `TOGETHER_API_KEY`, `OPENROUTER_API_KEY`, `OPENAI_API_KEY`. Keys live only in the proxy's environment and are scoped per-route by LiteLLM's model_list (`litellm_config.yaml:11, 21, 26, 92`). A compromise of the mobile app's storage cannot exfiltrate provider keys.
- **Credit-gated authorization**: every `/v1/chat/completions` and `/v1/web/search` request is bounced against CIRISBilling's `/v1/billing/credits/check` (`hooks/billing_callback.py:583`) before the upstream LLM call is dispatched. No-credit users receive HTTP 402 — no LLM cost, no usage row.
- **Authentication of users**: mobile clients authenticate with provider ID tokens (Google `id_token` / Apple Sign-In) verified against the provider's published public keys (`hooks/custom_auth.py:474, 816`). Audience is checked against the configured client-ID allowlist; issuer is pinned.
- **Idempotent charging**: every LLM call within an interaction shares an `interaction_id` set by the on-device agent. The first call for that interaction triggers `POST /v1/billing/charges` with `idempotency_key=f"litellm:{interaction_id}"` (`billing_callback.py:838`); CIRISBilling's `UNIQUE(account_id, idempotency_key)` constraint ensures one charge per interaction.
- **Interaction-bounded amplification**: a single `interaction_id` is capped at `MAX_INTERACTION_AGE_SECONDS=300` and `MAX_LLM_CALLS_PER_INTERACTION=80` (`billing_callback.py:118-119`). Exceeding either triggers a continuation ID and a new charge — bounding free-amplification within one interaction to 80 LLM calls.
- **PII discipline in logs**: external IDs, OAuth `sub` claims, and tokens are SHA256-prefix-hashed for log correlation (`billing_callback.py:191-195`). Token bytes never enter logs; format classification (`hooks/custom_auth.py:45-99`) provides debug detail without disclosure.
- **Config hardening at runtime**: `database_url: null` (`litellm_config.yaml:180`) disables LiteLLM's internal Postgres backend; `disable_spend_logs` is implicit (no DB to log to). `NO_DOCS=True`, `NO_REDOC=True`, `DISABLE_ADMIN_UI=True`, `OPENAPI_URL=` in `docker-compose.prod.yml:42-45` strip the upstream LiteLLM admin surface in production.
- **Provider-failure isolation**: Llama-4-Maverick is provisioned across three providers (OpenRouter, Groq, Together) with a 60 s cooldown on failure (`litellm_config.yaml:158-168`); a single-provider outage does not cascade to user-visible failure.
- **ZDR-compliant search**: `/v1/web/search` routes to Exa AI (zero-data-retention) by default with Brave Search as fallback (`hooks/search_handler.py:58-65`). No query content is retained at the search-provider layer.

### What CIRISProxy Does NOT Protect

- **Compromised CIRISBilling**: the proxy treats CIRISBilling's `has_credit=true` response as authoritative. A compromised billing service grants free LLM access. CIRISBilling's own threat model is the relevant document.
- **Compromised LLM provider**: Groq / Together / OpenRouter / OpenAI's behavior on the response side is trusted. A compromised upstream returning malicious completions is a CIRIS Agent concern, not a proxy concern.
- **OAuth-IdP compromise**: if Google's or Apple's identity infrastructure is compromised at the issuer, ID-token verification accepts forged tokens. Out of scope.
- **Compromised proxy host**: secrets in `.env` / Docker env (`GROQ_API_KEY`, `LITELLM_MASTER_KEY`, etc.) assume the host is not rooted. There is no envelope-encryption of provider keys at rest in the container.
- **TLS edge**: HTTPS termination is the deployment's nginx layer (`docker-compose.prod.yml:67-82`). Plaintext HTTP exposes OAuth tokens, request bodies, and provider responses. Out of scope for the proxy itself.
- **Mobile-app secret storage**: an exfiltrated user OAuth refresh token authenticates as that user. Hardware-backed mobile-app key custody (Android Keystore / iOS Keychain + Secure Enclave) is the upstream mitigation — and that's the CIRISAgent threat model's territory.
- **Database compromise (CIRISBilling-side)**: no DB on the proxy itself; CIRISBilling's Postgres is the credit-of-record store.
- **Quantum compute**: token signatures (RS256 for Apple, RS256 for Google ID tokens, all classical) are not PQC. Hybrid is tracked under the broader CIRIS roadmap.
- **CIRISLens log-shipper compromise**: a compromised lens endpoint can refuse logs (silent observability gap) or serve as a side-channel for the proxy to leak structured fields. The shipper batches and ships best-effort (`billing_callback.py:99-115`); no callback-blocking dependency.

---

## 2. Adversary Model

### Adversary Capabilities

The adversary is assumed to have:

- **Full source-code access** (the project is in CIRIS's open-source posture).
- **Network access to public endpoints**: `/v1/chat/completions`, `/v1/web/search`, `/v1/status`, `/v1/status/simple`, `/health/liveliness`.
- **Ability to obtain valid OAuth ID tokens for accounts they control**: Google and Apple ID tokens for their own users (and matching the configured `GOOGLE_CLIENT_IDS` / `APPLE_CLIENT_IDS` audiences).
- **Replay capability**: capture any in-transit request and resubmit.
- **Network MITM** between client and lens edge **only if TLS termination is misconfigured** — TLS-protected paths are out of reach.
- **Side-channel observation**: response timing, HTTP status codes, error message bodies, structured-log records aggregated in CIRISLens (if they can reach it).
- **Compute sufficient for classical cryptography** but not enough to break RS256 / Argon2id / SHA-256 in their parameter ranges.
- **Ability to register their own developer accounts** with Google or Apple to obtain token-signing client IDs.
- **CIRIS-Agent client behavior knowledge**: the agent batches 12-70 LLM calls per user interaction under one `interaction_id`. The adversary knows this convention (it's documented in `CLAUDE.md`).

### Adversary Limitations

The adversary is assumed to NOT have:

- **The `LITELLM_MASTER_KEY`** that controls LiteLLM's admin endpoints.
- **The `BILLING_API_KEY`** that authenticates the proxy to CIRISBilling for service-to-service calls.
- **The provider API keys** (`GROQ_API_KEY`, `TOGETHER_API_KEY`, `OPENROUTER_API_KEY`, `OPENAI_API_KEY`).
- **The `EXA_API_KEY` / `BRAVE_API_KEY`** that authenticate the proxy to search providers.
- **The `CIRISLENS_TOKEN`** that authenticates the proxy to the observability ingest endpoint.
- **A compromised CIRISBilling instance** willing to return `has_credit=true` for arbitrary users.
- **A compromised honest user's OAuth account at the IdP**.
- **Compromised the host** running the proxy (root / Docker-socket access).
- **Physical access** to the deployment hardware.
- **The ability to break TLS** between honest clients and the edge proxy.

---

## 3. Attack Vectors

Twenty-three attack vectors organized by adversary goal.

### 3.1 Forgery — adversary wants free LLM access

#### AV-1: Forged OAuth ID token

**Attack**: Attacker forges a Google or Apple ID token and presents it as `Authorization: Bearer ...` to `/v1/chat/completions` or `/v1/web/search`.

**Mitigation**: Tokens are verified against the IdP's published public keys with `id_token.verify_oauth2_token(token, request, client_id)` (`hooks/custom_auth.py:474`) for Google and `jwt.decode(token, public_key, algorithms=["RS256"], audience=bundle_id, issuer="https://appleid.apple.com")` (`hooks/custom_auth.py:816`) for Apple. Audience is checked against `GOOGLE_CLIENT_IDS` / `APPLE_CLIENT_IDS` allowlists; issuer is pinned. Apple JWKS keys are fetched from `https://appleid.apple.com/auth/keys` and cached for 1 h (`hooks/custom_auth.py:710-746`). Google's certs are managed by the `google-auth` library; key rotation triggers an automatic cache-clear-and-retry (`hooks/custom_auth.py:434, 489-496`).

**Secondary**: token format classification (`hooks/custom_auth.py:45-99`) and structured `auth_failure` logging (`hooks/custom_auth.py:177-218`) surface forgery attempts at INFO/WARNING levels for ops dashboards without disclosing token content.

**Residual**: see AV-2 (multi-client-id loop) and AV-3 (expired-token acceptance).

#### AV-2: Multi-client-id audience-loop confused-deputy

**Attack**: Attacker presents an ID token issued for a *different* legitimate audience (e.g., a different CIRIS service's OAuth client, or a different CIRIS app — Web vs Android) and the multi-client-id loop accepts it.

**Mitigation in v0.2.0**: **Partial.** `_try_verify_token` (`hooks/custom_auth.py:451-498`) loops over `GOOGLE_CLIENT_IDS = [GOOGLE_CLIENT_ID_WEB, GOOGLE_CLIENT_ID_ANDROID]` (`hooks/custom_auth.py:239`) trying each as the expected audience until one verifies. The Apple path does the same loop over `APPLE_CLIENT_IDS` (`hooks/custom_auth.py:813-832`). A token whose audience matches *any* configured client ID is accepted as valid for billing. If a non-billing service's client ID is mistakenly added to the env var, tokens issued to that service are accepted here.

**Recommended hardening**:
- Document that `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_ID_ANDROID` / `APPLE_CLIENT_ID` / `APPLE_CLIENT_IDS` MUST be the **CIRIS-Agent-mobile-eligible** client IDs only (not "every client ID across the org").
- For Google, also validate the `azp` (authorized party) claim if present, against an allowlist.
- Same shape as CIRISBilling AV-7.

**Residual**: operational risk — an admin who mis-configures the env var widens the trust boundary.

#### AV-3: Expired-token acceptance is a documented design point **[v0.2.0 P1 design point]**

**Attack**: Attacker exfiltrates a long-expired ID token (e.g., from a months-old log archive, an old phone backup, a developer's git history) and presents it. The proxy accepts it.

**Mitigation in v0.2.0**: **None at the temporal layer — by design.** `_handle_expired_token` (`hooks/custom_auth.py:530-562`) and `_try_decode_expired_token_silent` (`hooks/custom_auth.py:679-702`) explicitly fall back to `jwt.decode(token, verify=False)` when verification fails with an expiration error, then check audience and issuer claims and return the decoded `sub`. The header docstring (`hooks/custom_auth.py:14-17`) documents this: "Mobile apps may send tokens that expired hours ago. We still verify the signature (token was issued by provider) and audience (token was issued for our app)."

The signature *is* verified at first attempt — only when it fails *with* an expiration error does the unverified-decode path take over. So the attacker still needs a *real* token issued for a CIRIS audience by Google/Apple. But once such a token exists, it works forever, even after Google's stated expiry.

**Bounded today by**: the attacker must obtain a token originally issued for the CIRIS app's audience. Casual token theft from an unrelated Google service won't pass the audience check at the unverified-decode path either (`hooks/custom_auth.py:511-527, 696-702`). And CIRISBilling still gates per-interaction credit; the attacker pays per-interaction credits out of the victim's account — but can't extract the credits as money.

**Recommended hardening for v0.2.x**:
- Add a configurable maximum-expiry-age (e.g., `OAUTH_TOKEN_MAX_AGE_DAYS=30`) so a token expired more than N days is rejected. Closes the "old leaked token forever" residual.
- Consider requiring re-auth at the mobile-app layer when a token's `iat` is more than N days old, even if the proxy accepts it (defense in depth at the client).

**Residual**: a leaked OAuth token from any time in history works. Severity is bounded by per-interaction credit-spend, not full account-takeover at the IdP layer.

#### AV-4: Test-token bypass left enabled in production **[mitigated v0.2.0]**

**Attack**: Operator misconfigures `CIRIS_TEST_AUTH_ENABLED=true` in production. Any 32+ char hex/alphanumeric token (`hooks/custom_auth.py:284-304`) is forwarded to CIRISBilling's `/v1/billing/credits/check` for validation. If CIRISBilling has a matching test auth path, the proxy returns `UserAPIKeyAuth(api_key="test:{user_id}", user_id=user_id)` and the request proceeds.

**Mitigation in v0.2.0**: **Startup-time fail-fast gate.** `hooks/custom_auth.py:228-236` raises `RuntimeError` at module import if `CIRIS_TEST_AUTH_ENABLED=true` AND `CIRIS_ENV=production`. The container fails to start, surfacing the misconfiguration loudly. `docker-compose.prod.yml:43` sets `CIRIS_ENV=production` by default, so any operator using the production compose file inherits the gate without further action.

**Secondary**: when test auth is enabled outside production, the module logs at CRITICAL level (`hooks/custom_auth.py:259-264`) including the synthetic user_id the fallback maps to. The previous `logger.warning` was too soft for an exposure of this shape.

**Defense-in-depth**: CIRISBilling's own test-auth gate must also be enabled for the validation to succeed. Both proxy and billing would have to be misconfigured for the exploit to land — and the proxy gate now refuses to start before billing is even contacted.

**Test coverage**: `tests/test_custom_auth.py::TestTestAuthProductionGate` (4 tests) — refuses prod+test, allows non-prod, allows unset env, allows prod without test.

**Residual**: an operator who sets `CIRIS_ENV=staging` (or any value other than `production`) AND enables test auth retains the test path. By design — staging is allowed to run the test auth flow for integration testing.

#### AV-5: LiteLLM master-key bypass of custom_auth **[verified not exploitable in v0.2.0]**

**Attack hypothesis**: A client sends `Authorization: Bearer {LITELLM_MASTER_KEY}` (or `X-LiteLLM-API-Key: {master_key}`) to `/v1/chat/completions`, hoping master-key auth runs *before* `custom_auth` and short-circuits the OAuth verification.

**Verified against `litellm==1.83.10` source**: `litellm/proxy/auth/user_api_key_auth.py:640-669` (`_user_api_key_auth_builder`) runs custom_auth **first** — if `user_custom_auth is not None`, it calls `await user_custom_auth(request, api_key)`, validates the response, and returns. Master-key handling lives further down the function and is only reached when custom_auth is `None`. With `general_settings.custom_auth: "custom_auth.user_api_key_auth"` set (`litellm_config.yaml:177`), the master-key path is structurally unreachable for client-facing routes.

The `X-LiteLLM-API-Key` header precedence (`user_api_key_auth.py:612-623`) supersedes `Authorization`, but the resolved api_key is then passed to *our* custom_auth — which would attempt to verify it as an OAuth token, fail, and raise `ProxyException(code=401)`. There is no path where a client request reaches the master-key check.

**Mitigation**: by upstream auth ordering. Custom_auth always runs first. Master-key auth is the fallback when custom_auth is unset.

**Residual**: master-key admin endpoints (e.g., `/key/generate`) still exist for ops use, but `DISABLE_ADMIN_UI=True` and `NO_DOCS=True` (`docker-compose.prod.yml:42-45`) hide them and the underlying routes typically require admin auth at the route layer. Admin-key leak is a real concern (AV-11 envelope), but does not cross into the client-facing chat-completions surface.

**Verification recorded**: this AV was hypothesized as P0 in the v0.2.0 baseline draft. After reading upstream LiteLLM source, downgraded to "verified not exploitable" with the auth-ordering citation above. Re-verify on every LiteLLM bump that touches `proxy/auth/user_api_key_auth.py`.

#### AV-6: LiteLLM JWT-detection collision **[fixed in v0.2.0]**

**Attack**: After `custom_auth` returns `UserAPIKeyAuth(api_key="apple:{user_id}", user_id=user_id)`, LiteLLM's downstream metadata-tracking layer ran `is_jwt(api_key)` on the returned api_key. The naive check (`split('.') == 3`) classified `apple:001234.abc.xyz` as a JWT (Apple user IDs have format `<6digits>.<24chars>.<24chars>`) and replaced it with `hashed-jwt-xxx`, breaking the billing callback's `_parse_user_key` parsing.

**Mitigation in v0.2.0**: switched the Apple format from `apple:{user_id}` to `apple|{user_id}` (`hooks/custom_auth.py:1048`), with matching parsing in `_parse_user_key` (`hooks/billing_callback.py:336-337`). Pipe is not a JWT separator; `is_jwt` returns false; the api_key flows through to billing intact. Regression tests in `tests/test_billing_callback.py::test_parse_apple_pipe_delimiter`.

**Residual**: future LiteLLM-internals changes that re-introduce naive JWT-shape checks could re-break Apple — the `apple|` form is robust against three-dot detection but a four-segment delimiter check would still hit. The proper upstream fix is a wire change in LiteLLM's `is_jwt`. Track if upstream changes touch this area.

#### AV-7: Token cache poisoning via key collision

**Attack**: An attacker's token shares a string-equal cache key with an honest user's token (`_token_cache: dict[str, tuple[str, str, float]]` keyed by raw token bytes — `hooks/custom_auth.py:261`). On lookup, the attacker would receive the honest user's `user_id`.

**Mitigation in v0.2.0**: **Strong by construction.** The cache key is the full token string. For two tokens to collide on lookup, an attacker must possess a byte-identical token to the honest user — which means the attacker already has the honest user's token, and the cache is irrelevant (they'd verify successfully on their own). String-equality on a cryptographically-signed JWT is not a structural collision risk.

**Residual**: no real residual at this layer. Worth re-stating: cache hits never reduce the cryptographic hardness — the *first* verification per token is full-strength; cache only short-circuits *re-verification* of a token already proven correct.

### 3.2 Replay & Idempotency — adversary wants double-spend or amplification

#### AV-8: Charge replay

**Attack**: Network MITM (or a naïve client retry loop) captures and resubmits a `/v1/chat/completions` request.

**Mitigation**: The first call per interaction triggers `POST /v1/billing/charges` with `idempotency_key=f"litellm:{interaction_id}"` (`hooks/billing_callback.py:838`). CIRISBilling's `UNIQUE(account_id, idempotency_key)` constraint (CIRISBilling AV-8) enforces single-charge-per-interaction. Subsequent calls for the same `interaction_id` short-circuit at `_authorized_interactions[interaction_id] = True` (`hooks/billing_callback.py:621`).

**Residual**: see AV-9 (interaction-id reuse) for an in-bound amplification path that idempotency does *not* defend against.

#### AV-9: Interaction-ID reuse for free-LLM amplification **[v0.2.0 P1 design point]**

**Attack**: An authenticated user (with their own valid OAuth token and $0.01 of credits) reuses the same `interaction_id` indefinitely. Per the credit model "1 credit = 1 interaction," they are charged once but can run many LLM calls under that one interaction — until the `MAX_LLM_CALLS_PER_INTERACTION=80` (`hooks/billing_callback.py:119`) or `MAX_INTERACTION_AGE_SECONDS=300` (`hooks/billing_callback.py:118`) cap is hit, at which point a continuation ID triggers a new charge.

**Bounded today by**: 80 LLM calls per credit is the documented design point. This is the intended cost-model: agents make 12-70 calls per real user-facing interaction, and 80 is a generous ceiling. An attacker maxing out 80 calls per credit gets 80x amplification, which at OpenRouter's $0.11/$0.34/M-tokens pricing is ~$0.30 of LLM cost per $0.01 credit — acceptable for the proxy's role as a credit-gated subsidy.

**Recommended hardening for v0.2.x**:
- Periodically run `cleanup_stale_interactions` (`hooks/billing_callback.py:1104-1130`) — this method exists but is not called from anywhere in the codebase (`grep cleanup_stale_interactions` finds no caller). Wire it up to a periodic asyncio task or to LiteLLM's lifecycle hooks. Today, abandoned interactions accumulate in `_interaction_usage` until the proxy restarts.
- Consider tightening `MAX_LLM_CALLS_PER_INTERACTION` if telemetry shows the 90th-percentile real interaction is well under 80 calls.

**Residual**: per-credit amplification ratio is design-bounded but unenforced cleanup is a minor memory leak (see AV-15).

#### AV-10: Web-search idempotency-key timestamp drift **[v0.2.0 P2 functional]**

**Attack** (operational, not adversarial): A client retries a `/v1/web/search` call after a network blip. The retry generates a new idempotency key because the timestamp component changed: `idempotency_key = f"search-{user_id[:8]}-{query_hash}-{timestamp}"` where `timestamp = int(time.time())` (`hooks/search_handler.py:88-93`). CIRISBilling sees a new key and charges again.

**Severity**: this charges the user twice (or more) for retries that are seconds apart. With a 1-credit-per-search pricing model, the impact is small but real.

**Mitigation in v0.2.0**: **None.** The timestamp is deliberate ("keep key short but unique" per the inline comment), but it defeats idempotency for multi-second retries.

**Recommended hot-fix for v0.2.1**: drop the timestamp component, or replace with a coarser time bucket (e.g., 1-hour granularity) so retries within the same hour dedup. Match the chat-completions pattern, which keys on `interaction_id` alone (no timestamp) and trusts the agent to scope IDs per logical operation.

**Residual until fix**: retried searches charge multiple times. Refund tooling does not exist on the search path (`hooks/search_handler.py:336-337` documents this as a known gap).

### 3.3 Authentication-bypass — adversary wants to act as another principal

#### AV-11: Provider-API-key leak via env / shell

**Attack**: Attacker compromises the proxy host (root or Docker socket) and reads `GROQ_API_KEY`, `TOGETHER_API_KEY`, `OPENROUTER_API_KEY`, `OPENAI_API_KEY` from `/proc/{pid}/environ` or the Docker inspect surface. They then make direct calls to the upstream providers using CIRIS's billing relationship.

**Mitigation in v0.2.0**: **None at the application layer beyond host hardening.** Keys are env vars (`docker-compose.prod.yml:18-21`); there is no envelope-encryption-at-rest. This is consistent with the §1 out-of-scope "Compromised proxy host" — a rooted host means rooted secrets.

**Bounded today by**: standard Docker-on-Linux isolation; `restart: always` + healthcheck + memory limit keep the container honest. The `master_key` and `BILLING_API_KEY` are in the same env-var basket.

**Recommended for v0.2.x** (deferred — out of scope at the application layer): KMS-backed secret injection at start (read once, drop env-var-derived value into in-process state, scrub the env). Standard ops-tier hardening.

**Residual**: host compromise → all keys disclosed simultaneously.

#### AV-12: BILLING_API_KEY misuse **[v0.2.0 design point]**

**Attack**: The proxy holds `BILLING_API_KEY` (`hooks/billing_callback.py:275`), which authenticates *as the proxy* to CIRISBilling. If exfiltrated, the attacker can issue arbitrary `/v1/billing/credits/check` calls against any user's `external_id`, gleaning whether that user has credits — a privacy leak — and call other billing endpoints scoped to the proxy's permissions.

**Mitigation in v0.2.0**: same envelope as AV-11 (env var, no envelope-encryption). Permission scoping at CIRISBilling's end limits what the key can do (it's a `billing:read`-only key for the proxy's role; mutating endpoints require the proxy to authenticate per-user).

**Recommended for v0.2.x**: explicit per-purpose API keys (`BILLING_API_KEY_AUTH` for `/credits/check`, `BILLING_API_KEY_CHARGE` for `/charges`) with the smallest possible scope each. Short-lived keys with rotation-tooling at the CIRISBilling admin layer.

**Residual**: standard cross-service key-leak risk; bounded by CIRISBilling's per-key scope enforcement.

#### AV-13: ID-preview disclosure in stdout/file logs **[v0.2.0 P1 leakage]**

**Attack**: The pre-call hook prints `id_preview = f"{external_id[:4]}...{external_id[-4:]}"` to stdout via `print(...)` (`hooks/billing_callback.py:577-578`). For Google `sub` claims (typically 21 digits), this leaks 8 of 21 digits (38% of the ID space) — enough for cross-log correlation against Google-internal metadata. For Apple anonymized IDs (typically 44 chars), it leaks 8 of 44 (18%). Both also surface the `_hash_id(external_id)` SHA-256 prefix, which is correlation-stable.

**Mitigation in v0.2.0**: **Partial.** Only the SHA-256 prefix is canonical for log correlation (`hooks/billing_callback.py:191-195`); the `id_preview` is a debugging artifact. The other logs (`logger.info`, CIRISLens shipping, structured `_ship_log`) consistently use `_hash_id(external_id)` only.

**Recommended hot-fix for v0.2.1**: drop the `id_preview` print at `billing_callback.py:577-578` (and the matching one at `:606`). The hash-prefix is sufficient for correlation. Re-run a grep for any other `external_id[:N]` slices.

**Residual until fix**: log-aggregation tier (CIRISLens, container json-file logs, host syslog) holds 8-char prefix+suffix slices for the retention window.

#### AV-14: Token format classification logged at INFO level

**Attack**: `_log_auth_failure` logs `token_format`, `prefix_hash`, `header_valid_b64`, `length`, `client_type` at INFO level (`hooks/custom_auth.py:213-215`). Format classification is intentionally non-disclosing (length + structure only), but the `prefix_hash = sha256(token[:8])[:8]` (`hooks/custom_auth.py:97`) is correlation-stable across requests, allowing an attacker who scrapes logs to correlate failed-auth attempts to the same token across time.

**Mitigation in v0.2.0**: **Strong by construction.** Format classification was deliberately designed to avoid token disclosure; the prefix hash is one-way. Eight bits of token correlation is low-value for an attacker (no extraction path).

**Residual**: minor. Log retention is the exposure window; the leak is "this same failed token was tried again," not "here is the token."

### 3.4 Denial of Service — adversary wants the proxy unable to serve

#### AV-15: In-memory state unbounded under abandoned interactions **[v0.2.0 P1 functional + DoS]**

**Attack**: An authenticated attacker (with their own credits) opens many distinct `interaction_id`s, makes one LLM call each (which authorizes the interaction and creates a row in `_authorized_interactions` and `_interaction_usage` — `hooks/billing_callback.py:279, 283`), and then never returns. Each abandoned interaction lives forever.

**Mitigation in v0.2.0**: **None active.** The `cleanup_stale_interactions(max_age_seconds=3600)` method exists (`hooks/billing_callback.py:1104-1130`) but **is not invoked from anywhere** — `grep cleanup_stale_interactions hooks/ server.py` returns the definition only. There is no scheduled task, no LiteLLM lifecycle hook, no asyncio loop wiring it up.

**Bounded today by**: the proxy restarts on Ansible deploy and on container recycle (every few hours / days at typical operational cadence), which clears the in-process state. So the leak is bounded by the restart interval, not by a structural cleanup. The 1 GB container memory limit (`docker-compose.prod.yml:63`) is the ultimate ceiling.

**Recommended hot-fix for v0.2.1**: wire `cleanup_stale_interactions` into a periodic asyncio task at proxy startup. Either (a) at the LiteLLM `startup` hook, fire `asyncio.create_task(_periodic_cleanup())` that calls `await callback.cleanup_stale_interactions(3600)` every 600 s; or (b) call it inside `async_pre_call_hook` once per N requests with a cheap counter gate.

**Recommended secondary**: add a hard cap on `len(_interaction_usage)` (e.g., 100k entries) and evict oldest-by-start_time when full. Match the `_token_cache` pattern (`hooks/custom_auth.py:264, 273-281`).

**Residual until fix**: memory leak under abandoned interactions; mitigated by routine container recycle but otherwise unbounded.

#### AV-16: Body-size flood (no max body limit) **[v0.2.0 exposure]**

**Attack**: Attacker submits arbitrarily large bodies to `/v1/chat/completions` or `/v1/web/search`. FastAPI's default body extractor reads the entire body into memory before invoking the auth dependency.

**Mitigation in v0.2.0**: **None at the application level.** Defense relies on nginx's `client_max_body_size` directive at the deployment edge (`nginx/default.conf` per `docker-compose.prod.yml:75`). There is no FastAPI `DefaultBodyLimit` middleware on either the LiteLLM-provided routes or the `server.py` custom routes.

**Recommended hot-fix for v0.2.1**: explicit body-size cap via Starlette middleware in `server.py:add_custom_routes`. 1-2 MiB is sufficient for any current endpoint (search bodies are ~1 KiB; chat-completions with multimodal images can reach ~500 KiB).

**Same shape as CIRISBilling AV-16, CIRISPersist AV-7.** Same recommendation.

**Residual**: deployment-edge gap if nginx is misconfigured.

#### AV-17: Token-cache memory unbounded under attacker token diversity

**Attack**: Attacker (or an honest-but-distributed multi-tenant deployment) generates many distinct *valid* OAuth tokens (e.g., re-auth on every request) to grow `_token_cache` (`hooks/custom_auth.py:261`).

**Mitigation in v0.2.0**: `_cleanup_cache` enforces `_MAX_CACHE_SIZE = 10000` (`hooks/custom_auth.py:264, 273-281`). Verified-only entries, so invalid tokens never enter the cache. The cleanup runs only when the cache reaches the cap and removes only *expired* entries — there's no eviction-by-age-when-full beyond expiry.

**Edge case**: if 10000 valid tokens with 24-h cache-duration each enter the cache faster than they expire (i.e., > 10000 distinct authenticated users in a 24-h window), `_cleanup_cache` removes nothing (no entries are expired) and the dict grows past `_MAX_CACHE_SIZE`. The `if len(_token_cache) < _MAX_CACHE_SIZE: return` (`hooks/custom_auth.py:275-276`) is the only gate; once exceeded, cleanup runs every call but evicts only what's *already* expired.

**Recommended hardening for v0.2.x**: when `_cleanup_cache` runs and finds no expired entries to evict, fall back to LRU eviction (drop the oldest entry by cache_until - 24h). Or use `cachetools.TTLCache(maxsize=10000, ttl=86400)` which handles this correctly out of the box.

**Residual**: bounded by validated-user diversity. For a single-tenant CIRIS deployment with thousands of users, not exploitable; for a future multi-tenant deployment, worth tightening.

#### AV-18: Apple-JWKS-fetch storm under upstream outage

**Attack**: When Apple's `https://appleid.apple.com/auth/keys` is unreachable, `_fetch_apple_public_keys` (`hooks/custom_auth.py:710-746`) raises if there are no cached keys. Each subsequent Apple-token request hits the failed fetch path, generating outbound traffic to a known-down endpoint.

**Mitigation in v0.2.0**: **Partial.** If keys are cached and still within TTL, the fetch is skipped (`hooks/custom_auth.py:715`). On error, existing keys are kept (`hooks/custom_auth.py:744-746`) — so once-fetched keys outlive transient outages.

**Edge case**: cold-start during an Apple JWKS outage rejects every Apple request *and* generates outbound traffic per request. There is no exponential backoff, no negative-cache.

**Recommended hardening for v0.2.x**: add a negative-cache (e.g., 30 s after a fetch failure, return cached error rather than re-fetching). Same shape as the Google `_clear_google_certs_cache` path (`hooks/custom_auth.py:434-448`) but with retry-bound semantics.

**Residual**: cold-start-during-outage is a small attack window; not exploitable beyond what an Apple outage causes anyway.

#### AV-19: Status-endpoint outbound amplification

**Attack**: Attacker hammers `/v1/status` to drive outbound polling traffic to OpenRouter, Groq, Together AI, CIRISBilling, and Brave (`hooks/status_handler.py:38-80`).

**Mitigation in v0.2.0**: 10-second cache (`hooks/status_handler.py:23`). Repeat calls within 10 s return the cached payload without hitting upstream.

**Bounded today by**: cache TTL — at worst 0.5 RPS of outbound polling (5 providers × every 10 s). Bounded.

**Residual**: if `_status_cache` is invalidated by a non-attacker race (e.g., a status check during cache-TTL expiry), there's no rate-limit on the cache-rebuild path. Acceptable.

#### AV-20: Provider-key exhaustion via burst **[design tension]**

**Attack**: Attacker with valid credits bursts requests to `default` (which maps to OpenRouter Llama-4-Maverick, `litellm_config.yaml:103-109`) until the OpenRouter rate-limit is hit. LiteLLM's failover then routes to Groq, then to Together AI, exhausting the org-wide rate budgets at all three providers.

**Mitigation in v0.2.0**: cooldown 60 s per provider after `RateLimitErrorAllowedFails: 5` (`litellm_config.yaml:148-158`). Failed providers are excluded from the routing pool for 60 s. Credit-gating is the per-user throttle: a single user's max-burst is capped by their credit balance.

**Bounded today by**: per-user credit pool + per-provider cooldown. An attacker with many user accounts coordinated via the same authenticated session hits the credit cap before the provider cap.

**Residual**: a sufficiently large coordinated attacker (many funded accounts) can trigger all three providers' cooldowns simultaneously, leaving the proxy with no routes. Recovery: 60 s cooldown elapses. Track for v0.2.x with per-user-rate-limiting.

### 3.5 Data Corruption / Privilege Escalation

#### AV-21: Vision-routing model-swap

**Attack**: Attacker submits a multimodal request with an image to a `groq/...` model (which doesn't support system messages + images). `_handle_multimodal_routing` (`hooks/billing_callback.py:478-497`) silently rewrites `data["model"]` to `together_ai/meta-llama/Llama-4-Maverick-...` to satisfy the multimodal contract. The user pays the same 1 credit, but the actual model selected is a Together model — at Together's pricing, not Groq's.

**Mitigation in v0.2.0**: this is by design — the rewrite preserves the credit-gated semantics (1 interaction = 1 credit) regardless of which provider serves the request. The `cost_dollars` in the success log (`hooks/billing_callback.py:911-913`) reflects the actual Together AI cost; the proxy's per-credit margin shrinks for multimodal but the user is not undercharged.

**Bounded today by**: cost-margin policy. As long as 1-credit pricing covers the worst-case provider choice (Together for multimodal), the proxy's economics hold.

**Residual**: cost-margin erosion if multimodal share grows and Together's pricing increases. Track via the `_global_usage` aggregate `total_cost_cents` (`hooks/billing_callback.py:912`).

#### AV-22: External-id-injection via API-key delimiter **[v0.2.0 P2]**

**Attack**: An authenticated user with `user_id` containing a colon (e.g., `attacker:victim_user_id`) hits the `_parse_user_key` path. The Google-format check at `hooks/billing_callback.py:339-340` (`if api_key.startswith("google:")`) trims `google:` and returns the rest as the external_id. But the lower fallback at `:341-343` (`elif ":" in api_key`) splits on the *first* colon and treats the prefix as the provider. A malicious `user_id="evil:victim123"` arriving as `google:evil:victim123` would be parsed as `provider="google", external_id="evil:victim123"` — which is fine, since the OAuth verifier set `user_id` from the `sub` claim and the IdP signs that.

But for **the upstream OAuth `sub` claim** to contain a colon, the IdP would have to mint such an ID. Google's `sub` is numeric-only; Apple's `sub` is `<6digits>.<24chars>.<24chars>` (no colons). So this is not exploitable today.

**Mitigation in v0.2.0**: defense-in-depth. The `_parse_user_key` is correct for the IdP-issued ID space.

**Recommended for v0.2.x**: add explicit validation that `external_id` matches `[a-zA-Z0-9._-]+$` after parsing — refuse any external_id containing a colon, pipe, or non-printable char. Closes a future-IdP-format-change residual.

**Residual**: dependent on Google/Apple `sub` claim format stability.

#### AV-23: Charge-without-LLM-success on async hook ordering

**Attack**: A request that successfully passes `async_pre_call_hook` (which authorizes and caches the interaction) but fails *before* the LLM call completes (e.g., an exception during request preprocessing inside LiteLLM, before the upstream HTTP call) — does the user get charged?

**Mitigation in v0.2.0**: charge happens in `async_log_success_event` (`hooks/billing_callback.py:818-885`). Failures are tracked in `async_log_failure_event` (`hooks/billing_callback.py:984-1035`) and *do not* trigger a charge. So a request that fails before reaching upstream-LLM-success does not consume credit. The structural guarantee: charge is gated by success, not by auth-pass.

**Bounded today by**: LiteLLM's hook-ordering. As long as `async_log_success_event` only fires on real upstream success, the credit-charge invariant holds.

**Residual**: if LiteLLM ever changes the success-hook semantics (e.g., fires on cache hits, or on partial-content responses), the charge model would silently shift. Track at LiteLLM version bumps.

### 3.6 Operational hardening (catalogued for v0.2.x track)

These vectors are deferred but tracked, mirroring CIRISBilling §3.7 and CIRISPersist §3.6:

- **OP-1** — LiteLLM upstream supply chain. Currently pinned to `litellm==1.83.10` post-March-2026 incident (`pyproject.toml:11`) and `ghcr.io/berriai/litellm:v1.83.10-stable` base image (`Dockerfile:9`). CVE-2026-42208 (SQLi, affects 1.81.16-1.83.6) is closed. Weekly advisory check via remote Claude routine `trig_014bFeAY4U3SjsHZBneNhkAH` (Mondays 11 ET).
- **OP-2** — No per-user / per-key rate limit. Throttle is via CIRISBilling credit balance + per-provider cooldown. App-layer rate limiting for v0.2.x.
- **OP-3** — No graceful-shutdown drain on SIGTERM. In-flight LLM calls may be cut at restart. Match CIRISPersist AV-19 pattern.
- **OP-4** — `:latest` image tag on `docker-compose.prod.yml:3`. Pull-by-digest via `gh api /orgs/cirisai/packages/container/cirisproxy/versions` would be tighter; track for v0.2.x.
- **OP-5** — Submodule `libs/cirislens` pinned to a specific commit but no automated drift check. The weekly routine flags drift > 30 days.
- **OP-6** — No CSP / X-Frame-Options on responses. The proxy serves no HTML, so the surface is small, but ops-tier defense-in-depth.
- **OP-7** — No clock-skew validation on `iat` / `exp` claims beyond the IdP library defaults; combined with AV-3 expired-token acceptance, clock-drift is not a risk we can detect.
- **OP-8** — `_global_usage` and `_interaction_usage` are per-process. Multi-worker deploys would have N independent counters with no aggregation. Today the proxy runs single-worker (LiteLLM default); track at multi-worker rollout.
- **OP-9** — `print(...)` to stdout for high-volume usage logs (`hooks/billing_callback.py:391-398, 532, 540, 606`). Mixes with structured `logger.info` and CIRISLens shipping. Consider replacing with `logger.info` consistently to avoid unstructured-log clutter.

---

## 4. Mitigation Matrix

| AV | Attack | Primary Mitigation (v0.2.0) | Secondary | Status | Fix Tracker |
|---|---|---|---|---|---|
| AV-1 | Forged OAuth ID token | Provider public-key + audience + issuer verify | Token format classification logging | ✓ Mitigated | — |
| AV-2 | Multi-client-id confused-deputy | aud-loop allowlist | (no `azp` check) | ⚠ Operator-config risk | v0.2.x |
| AV-3 | Expired-token acceptance | Sig + audience verify even on expired tokens | Per-interaction credit gating limits damage | ⚠ P1 design point | v0.2.x — add max-age cap |
| AV-4 | Test-auth left enabled in production | Startup-time fail-fast: refuse boot if `CIRIS_TEST_AUTH_ENABLED=true` AND `CIRIS_ENV=production` | CIRISBilling cross-check; CRITICAL log on enable | **✓ Mitigated v0.2.0** | — |
| AV-5 | LiteLLM master-key bypass of custom_auth | Upstream auth-ordering — custom_auth runs first and returns; master_key path unreachable when custom_auth set | Verified against `litellm==1.83.10` source | **✓ Verified not exploitable** | re-verify on litellm bumps |
| AV-6 | LiteLLM JWT-detection collision | `apple\|{user_id}` pipe delimiter | Regression test | **✓ Mitigated v0.2.0** | — |
| AV-7 | Token cache poisoning | Cache key is full token bytes | Cryptographic verify per token | ✓ Mitigated | — |
| AV-8 | Charge replay | `idempotency_key=litellm:{interaction_id}` + UNIQUE in CIRISBilling | Per-interaction cache | ✓ Mitigated | — |
| AV-9 | Interaction-ID reuse amplification | `MAX_LLM_CALLS_PER_INTERACTION=80` + `MAX_INTERACTION_AGE_SECONDS=300` | Continuation-ID triggers new charge | ⚠ Bounded by design; cleanup unwired | v0.2.x — wire cleanup |
| AV-10 | Web-search idempotency timestamp drift | (timestamp in key — defeats retry dedup) | — | **⚠ P2 functional** | **v0.2.1 hot-fix** |
| AV-11 | Provider-API-key leak via env | (host-hardening only) | Docker isolation + memory limit | ⚠ Out-of-scope (host) | v0.2.x — KMS injection |
| AV-12 | BILLING_API_KEY misuse | Env var; CIRISBilling per-key scope | — | ⚠ Standard cross-service risk | v0.2.x — per-purpose keys |
| AV-13 | ID-preview disclosure in logs | (8-char prefix+suffix slice printed) | SHA256 hash also printed (canonical) | **⚠ P1 leakage** | **v0.2.1 hot-fix** |
| AV-14 | Token format classification at INFO | Structural-only, no token bytes | prefix_hash is one-way SHA256 | ✓ Mitigated | — |
| AV-15 | In-memory state unbounded under abandoned interactions | (cleanup_stale_interactions defined but never called) | Container restart bounds it | **⚠ P1 functional + DoS** | **v0.2.1 hot-fix** |
| AV-16 | Body-size flood | (deployment-edge nginx only) | — | ⚠ Add app-level cap | v0.2.1 |
| AV-17 | Token-cache memory unbounded | `_MAX_CACHE_SIZE=10000` + expiry cleanup | Verified-only entries | ⚠ Edge case past 10k unique | v0.2.x — LRU |
| AV-18 | Apple JWKS storm under outage | Stale-key reuse on fetch error | TTL cache | ⚠ Cold-start gap | v0.2.x — negative cache |
| AV-19 | Status-endpoint outbound amplification | 10-s cache | 5 providers × every 10s = bounded | ✓ Mitigated | — |
| AV-20 | Provider-key exhaustion via burst | Per-provider 60s cooldown + credit-cap | Three-provider failover | ⚠ Coordinated-attacker residual | v0.2.x — per-user RL |
| AV-21 | Vision-routing model-swap | By-design rewrite preserves 1-credit semantics | actual_cost tracking | ✓ Mitigated; ⚠ margin policy | track |
| AV-22 | External-id colon-injection via api_key | IdP `sub` claim format constrains input | — | ✓ Mitigated; ⚠ format-stability dependency | v0.2.x — explicit validator |
| AV-23 | Charge-without-LLM-success | Charge in success-hook only | Failure-hook does not charge | ✓ Mitigated; ⚠ LiteLLM hook-semantics dependency | track at version bumps |

---

## 5. Security Levels by Deployment Tier

| Tier | Backend | Auth Edge | Threat Model |
|---|---|---|---|
| **Production multi-instance** (US + EU) | No DB; CIRISBilling for credits; `:latest` GHCR image | TLS at nginx; env-var secrets | Full §3 model. AV-4 / AV-5 / AV-13 / AV-15 / AV-16 hot-fixes required. |
| **Single-host Docker compose** | Same — no DB | TLS optional; .env file secrets | §3 applies; secrets-at-rest exposure higher (AV-11). Operator must enforce `CIRIS_TEST_AUTH_ENABLED=false`. |
| **Local development** | Docker compose with stub billing | No TLS | Test-auth bypass enabled. NEVER expose to public network. No production gate today (compare to CIRISBilling AV-14). |
| **CIRISBridge-managed** (Ansible) | Same as production | TLS at nginx; Ansible-managed env | Same as production; secret-rotation tooling lives in CIRISBridge. |

Critical invariant: **all tiers run the same FastAPI app, same hooks, same litellm_config.yaml**. A finding in one tier's implementation is presumed to apply to all unless explicitly excepted.

---

## 6. Security Assumptions

The system depends on these assumptions; if violated, the threat model breaks.

1. **TLS at the deployment edge**: nginx (or equivalent) terminates HTTPS for all inbound traffic. Plaintext HTTP exposes OAuth tokens, request bodies, and response content. AV-1 / AV-3 / AV-13 all assume this.
2. **CIRISBilling integrity**: `/v1/billing/credits/check`, `/v1/billing/charges`, `/v1/billing/litellm/usage`, `/v1/tools/charge` are authoritative and not tampered with. The proxy treats `has_credit=true` as ground truth.
3. **OAuth-IdP integrity**: Google's and Apple's signature infrastructure is intact at the issuer. ID tokens, when verified, are accepted as authoritative for user identity (`sub` claim).
4. **Provider trust**: Groq, Together, OpenRouter, OpenAI, Exa, Brave honor their stated APIs and pricing. CIRISProxy does not validate response semantics beyond HTTP status.
5. **Env / Docker-secret confidentiality**: `LITELLM_MASTER_KEY`, `BILLING_API_KEY`, `GROQ_API_KEY`, `TOGETHER_API_KEY`, `OPENROUTER_API_KEY`, `OPENAI_API_KEY`, `EXA_API_KEY`, `BRAVE_API_KEY`, `CIRISLENS_TOKEN` are read from non-world-readable files. Host hardening is the operational doc.
6. **Clock accuracy**: proxy hosts are within ~5 minutes of UTC. JWT `exp` checks (when applicable, AV-3 documents the carve-out) and Apple JWKS TTL rely on this.
7. **Container-recycle cadence**: in-process state (`_token_cache`, `_interaction_usage`, `_authorized_interactions`) is bounded by memory limits and routine container restarts (AV-15 noted).
8. **LiteLLM upstream integrity**: pinned `litellm==1.83.10` post-incident. Weekly advisory check in place (`trig_014bFeAY4U3SjsHZBneNhkAH`).
9. **Submodule pinning**: `libs/cirislens` is pinned to a specific commit; weekly drift check tracked.

---

## 7. Fail-Secure Degradation

All failures degrade to MORE restrictive modes, never less. This mirrors CIRISBilling §7 and CIRISPersist §7.

| Failure Condition | Behavior | Rationale |
|---|---|---|
| Missing `Authorization` header | HTTP 401; zero LLM calls | Cannot authenticate → cannot proceed |
| Invalid OAuth signature | HTTP 401; zero LLM calls | Signature is the trust anchor |
| Audience mismatch | HTTP 401; zero LLM calls | Token not for this app |
| Invalid issuer | HTTP 401; zero LLM calls | Token not from a trusted IdP |
| Test token mode disabled (default) | Test tokens rejected | `CIRIS_TEST_AUTH_ENABLED=false` is the safe default |
| `CIRIS_TEST_AUTH_ENABLED=true` AND token invalid at billing | HTTP 401 | Test path delegates to CIRISBilling for auth |
| Missing `interaction_id` in metadata | HTTP raised via `MissingInteractionIdError` | Per-interaction credit-gating requires it |
| `has_credit=false` from CIRISBilling | HTTP raised via `InsufficientCreditsError` | Cannot charge → cannot proceed |
| CIRISBilling unreachable | HTTP raised via `BillingServiceUnavailableError` | Fail closed, not silent |
| Apple JWKS fetch failure (cold start) | HTTP 401 (no cached keys) | Cannot verify → cannot authenticate |
| Apple JWKS fetch failure (warm) | Use stale keys | Outage tolerance — bounded by Apple's actual key-rotation cadence |
| Google certs cache stale (key rotation) | Auto-refresh once and retry | Self-healing |
| Provider rate-limited / errored | Failover within model group | Three-way redundancy |
| All providers in cooldown | HTTP 5xx returned to client | Surface backpressure honestly |
| `interaction_id` reused past `MAX_INTERACTION_AGE_SECONDS` or `MAX_LLM_CALLS_PER_INTERACTION` | Continuation ID + new charge | Bound amplification |
| LLM call fails (any reason) | `async_log_failure_event` only; no charge | Charge gated on success |
| Validation error (Pydantic) | HTTP 422 with sanitized errors | Schema-altering input rejected |
| `database_url` accidentally set | (LiteLLM enables internal DB and admin UI) | **Operator must keep `database_url: null`** |

Critical invariant: **rows are charged only on upstream LLM success**. The pre-call hook authorizes; the success hook charges. No success → no charge. The only path that bypasses this is `CIRIS_TEST_AUTH_ENABLED=true` (AV-4), which is documented and gated by env.

---

## 8. Residual Risks

Risks CIRISProxy mitigates but cannot fully eliminate:

1. **Compromised user OAuth account at the IdP** (AV-1 residual). The proxy accepts authenticated requests. Closure: client-side hardware-backed key storage (Android Keystore + StrongBox / iOS Secure Enclave). CIRISAgent's threat model is upstream.

2. **Compromised CIRISBilling**. The proxy treats `has_credit=true` as ground truth. Closure: CIRISBilling's own threat model + per-service mutual-TLS at the deployment edge.

3. **Compromised proxy host** (AV-11). Env-var secrets disclosed simultaneously. Closure: KMS-backed secret injection at start (deferred, ops-tier).

4. **Long-expired token replay** (AV-3). Once a token is issued for the CIRIS app, the proxy accepts it indefinitely. Closure: configurable max-age cap; mobile-app side re-auth on `iat` age.

5. **LiteLLM master-key bypass** (AV-5). Severity contingent on LiteLLM's auth ordering. Closure: verify + rotate per deploy + isolate admin endpoints.

6. **Test-auth bypass left enabled** (AV-4). Single-misconfig insufficient (CIRISBilling cross-check), but no hard production gate. Closure: startup validator.

7. **In-memory state leak** (AV-15). `cleanup_stale_interactions` defined but never called. Closure: wire to periodic asyncio task.

8. **ID-preview log disclosure** (AV-13). 8-char prefix+suffix slices in stdout/file logs. Closure: drop the print, keep only the SHA256-prefix.

9. **Body-size flood** (AV-16). nginx-edge defense only. Closure: app-level `DefaultBodyLimit`.

10. **Web-search retry double-charge** (AV-10). Timestamp in idempotency key defeats retry dedup. Closure: drop or coarsen the timestamp component.

11. **LiteLLM upstream supply chain** (OP-1). Pinned + weekly check active. Closure: continued vigilance + version bumps as advisories land.

12. **Post-quantum signature replacement**. RS256 is classical-only. Tracked in the broader CIRIS hybrid-signature roadmap, not v0.2.x.

---

## 9. v0.2.0 Threat Posture Summary

```
v0.2.0 P0 EXPOSURES — all closed in v0.2.0
  ✓ AV-4  Test-auth bypass in production    (startup gate refuses boot when CIRIS_TEST_AUTH_ENABLED=true AND CIRIS_ENV=production)
  ✓ AV-5  Master-key bypass of custom_auth  (verified not exploitable — custom_auth runs first per litellm 1.83.10 source)

v0.2.0 P1 EXPOSURES (hot-fix in v0.2.1)
  ⚠ AV-3  Expired-token acceptance (no max-age cap)
  ⚠ AV-13 ID-preview prefix+suffix in stdout logs
  ⚠ AV-15 In-memory state unbounded under abandoned interactions (cleanup unwired)
  ⚠ AV-16 No application-layer body-size cap             (nginx mitigates)

v0.2.0 P2 TRACK
  ⚠ AV-10 Web-search idempotency timestamp drift         (retries double-charge)

v0.2.x TRACK
  ⚠ AV-2  Multi-client-id `azp` check
  ⚠ AV-11 KMS-backed provider-key injection
  ⚠ AV-12 Per-purpose CIRISBilling API keys
  ⚠ AV-17 LRU eviction on token cache
  ⚠ AV-18 Apple JWKS negative-cache
  ⚠ AV-20 Per-user / per-key rate limiting
  ⚠ AV-22 Explicit external_id format validator
  ⚠ OP-3  Graceful-shutdown drain
  ⚠ OP-4  Pull image by digest, not :latest
  ⚠ OP-9  Replace `print(...)` with structured logging consistently

DESIGN-DECISIONS-PER-MISSION (intentional, not defects)
  ✓ AV-1  OAuth provider public-key + audience + issuer verify
  ✓ AV-4  Startup-time production gate on test auth
  ✓ AV-5  Upstream auth-ordering verified — master_key unreachable when custom_auth set
  ✓ AV-6  apple| pipe delimiter for billing parse
  ✓ AV-7  Cache keyed on full token bytes
  ✓ AV-8  idempotency_key=litellm:{interaction_id}
  ✓ AV-9  MAX_LLM_CALLS / MAX_AGE bounds amplification
  ✓ AV-14 Token format classification only, no bytes
  ✓ AV-19 Status endpoint 10-s cache
  ✓ AV-21 Vision-routing rewrite preserves 1-credit semantics
  ✓ AV-23 Charge gated on LLM success
```

CIRISProxy is **temporary bridging infrastructure** retiring with Veilid maturity (per `CLAUDE.md` Mission Alignment). Threat-model investment is calibrated to the 18-24 month operational window: P0 / P1 hot-fixes get done; v0.2.x track items get done if Veilid timeline slips; v0.3.x track items only land if the proxy outlives its plan.

---

## 10. Update Cadence

This document is updated:
- On every minor version (v0.2.x → v0.3.0): comprehensive review.
- On every published security advisory affecting deps (LiteLLM, google-auth, PyJWT, httpx): addendum in §3 + the weekly remote-routine triage.
- On every new auth provider (Apple Sign-In was added in v0.2.0; future: Veilid identity?): AV-1 / AV-2 / AV-3 review.
- On every Veilid-migration milestone: posture-summary refresh against new transport assumptions.

**Last updated**: 2026-05-01 (v0.2.0 baseline; AV-4 closed by startup gate + `CIRIS_ENV=production` in prod compose; AV-5 verified not exploitable against `litellm==1.83.10` source — master_key path is unreachable when custom_auth is set; AV-3 / AV-13 / AV-15 / AV-16 scoped as P1 hot-fixes for v0.2.1; AV-6 marked closed after the `apple|` delimiter fix in v0.2.0; OP-1 closed for CVE-2026-42208 by the 1.83.10 bump).
