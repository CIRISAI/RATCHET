# CIRISBilling Threat Model

**Status:** v0.1.0 baseline. Updated each minor release.
**Audience:** integrators, payment-flow reviewers, security reviewers.
**Companion:** [`SECURITY_IMPLEMENTATION_PLAN.md`](../SECURITY_IMPLEMENTATION_PLAN.md), [`README.md`](../README.md), [`docs/STRIPE_PURCHASE_TESTING.md`](STRIPE_PURCHASE_TESTING.md).
**Inspired by:** [CIRISVerify `THREAT_MODEL.md`](../../CIRISVerify/docs/THREAT_MODEL.md) and [CIRISPersist `THREAT_MODEL.md`](../../CIRISPersist/docs/THREAT_MODEL.md) — structural template and adversary-goal organization.

---

## 1. Scope

### What CIRISBilling Protects

CIRISBilling is the credit-gating ledger for the CIRIS ecosystem. It mediates between user-facing payment providers (Stripe / Apple StoreKit / Google Play) and agent-facing usage charges. It protects:

- **Ledger integrity**: every persisted charge or credit was authorized by the rightful party (an agent with a valid API key for that account, or a verified payment from one of the registered providers) — OR is rejected. There is no third state.
- **Idempotency**: agent retries and webhook redeliveries cannot double-charge or double-credit. The constraint `UNIQUE(account_id, idempotency_key)` on `charges` and `credits` (`app/db/models.py:197, 277`) is the structural guarantee. Provider-side retries are deduped via `UNIQUE(purchase_token)` on `google_play_purchases` (`models.py:583`) and `UNIQUE(transaction_id)` on `apple_storekit_purchases` (`models.py:643`).
- **Authentication of agents**: agent traffic is gated by Argon2id-hashed API keys (`app/services/api_key.py:79, 101`) with permission scopes (`billing:read`, `billing:write`); plaintext keys exist only at issuance and are shown once.
- **Authentication of users**: mobile clients authenticate with provider ID tokens (Google / Apple) verified against the provider's public keys (`app/api/dependencies.py:283, 508`), with a deny-list (`token_revocation_service`).
- **Webhook authenticity** (Stripe path): Stripe webhooks are HMAC-verified against the webhook secret before any mutation (`app/services/stripe_provider.py:189`).
- **Admin-UI session integrity**: admin sessions ride on `HttpOnly + Secure + SameSite=lax` cookies signed as JWTs by `ADMIN_JWT_SECRET` (`app/api/admin_auth_routes.py:138-146`).
- **Production-mode hardening**: OpenAPI docs/redoc disabled in production (`app/main.py:76-78`); `/metrics` IP-restricted to internal networks (`app/main.py:317-340`); test-auth bypass refuses to start when `environment=production` (`app/config.py:172`).
- **Database-write durability**: PostgreSQL primary/replica with write-verification semantics; horizontal-scale stateless API workers behind nginx + PgBouncer.

### What CIRISBilling Does NOT Protect

- **Payment-provider compromise**: if a Stripe / Apple / Google merchant account is compromised at the provider, the attacker can issue legitimate (signature-valid) payment events. CIRISBilling cannot distinguish those from honest payments.
- **API-key theft from an agent's environment**: an exfiltrated API key authenticates as that agent. Hardware-backed agent identity (CIRISVerify) is the upstream mitigation.
- **OAuth-provider compromise**: if Google or Apple's identity infrastructure is compromised at the issuer, ID-token verification accepts forged tokens. Out of scope.
- **TLS edge / CA compromise**: HTTPS termination is the deployment's nginx layer. Plaintext HTTP exposes API keys, OAuth tokens, and webhook bodies. Out of scope (deployment concern).
- **Database compromise**: the Postgres backend is trusted. If Postgres is owned, the ledger is owned.
- **Host / container compromise**: secrets in Docker secrets / env vars assume the host is not rooted (`SECURITY_IMPLEMENTATION_PLAN.md` is the operational doc for this).
- **Quantum compute**: token signatures (RS256, HS256, Ed25519 elsewhere in the ecosystem) are classical-cryptography only. Hybrid PQC is tracked under the broader CIRIS roadmap, not this service.
- **Brave Search billing accuracy**: `tool_routes` charges per web search but trusts upstream provider results.

---

## 2. Adversary Model

### Adversary Capabilities

The adversary is assumed to have:

- **Full source-code access** (the project follows the wider CIRIS open-source posture).
- **Network access to public endpoints**: `/v1/billing/*`, `/v1/billing/webhooks/*`, `/admin/oauth/login`, `/admin-ui/login.html`, `/health`, `/`.
- **Ability to mint OAuth tokens for accounts they control**: Google and Apple ID tokens for their own users.
- **Ability to create real Stripe PaymentIntents on the merchant** with attacker-controlled `metadata`, if they can reach the merchant's payment surface (typically gated by the customer-facing checkout).
- **Replay capability**: capture any in-transit request and resubmit.
- **Network MITM** between client and lens edge **only if TLS termination is misconfigured** — TLS-protected paths are out of reach.
- **Side-channel observation**: response timing, HTTP status codes, error message bodies.
- **Compute sufficient for classical cryptography** but not enough to break Argon2id, RS256, HS256, or ECDSA primitives in their parameter ranges.
- **Ability to register their own developer accounts** with Google Play / Apple App Store / Stripe to send notifications from those identities.

### Adversary Limitations

The adversary is assumed to NOT have:

- **The Argon2id-hashed API key material** stored in `api_keys.key_hash`.
- **The `ADMIN_JWT_SECRET`** that signs admin session cookies.
- **The Stripe webhook secret** (`whsec_...`) that protects webhook integrity.
- **The Apple StoreKit private key** (.p8) that signs outbound App Store Server API calls.
- **The Google Play service-account JSON** that authenticates Play Integrity / Play Developer API calls.
- **Compromised any honest user's OAuth account at the IdP**.
- **Compromised the Postgres backend** that the API writes to.
- **Compromised the host** running the API (root / Docker-socket access).
- **Physical access** to the deployment hardware.
- **The ability to break TLS** between honest clients and the edge proxy.

---

## 3. Attack Vectors

Twenty-four attack vectors organized by adversary goal.

### 3.1 Forgery — adversary wants their bytes counted as a real credit / charge

#### AV-1: Forged credit via direct API call

**Attack**: Attacker without an API key calls `POST /v1/billing/credits` (or any `billing:write` endpoint) directly.

**Mitigation**: API-key dependency `get_api_key` (`app/api/dependencies.py`). Missing or invalid `X-API-Key` header → HTTP 401. Stored key hash is Argon2id (`app/services/api_key.py:79, 101`); plaintext is never persisted. Permission scope `billing:write` is required for mutating endpoints (`app/api/routes.py:174, 285`).

**Secondary**: structured-logging of `api_key_validation_failed` for ops dashboards.

**Residual**: an attacker with a stolen API key authenticates as the rightful holder. Mitigation is upstream (agent secrets-storage hardening; CIRISVerify hardware-backed key custody).

#### AV-2: Forged Stripe webhook

**Attack**: Attacker posts a crafted `payment_intent.succeeded` body to `POST /v1/billing/webhooks/stripe`.

**Mitigation**: `stripe.Webhook.construct_event(payload, signature, webhook_secret)` HMAC-verifies the `Stripe-Signature` header before any mutation (`app/services/stripe_provider.py:189`). Verification failure → typed `WebhookVerificationError` → HTTP 401, zero rows persisted (`app/api/routes.py:793`).

**Secondary**: even after signature verification, the handler calls `confirm_payment(payment_id)` against Stripe's API to re-fetch the PaymentIntent (`routes.py:737`), so a replayed-but-stale event must still resolve to a `succeeded` PaymentIntent at confirm time.

**Residual**: see AV-3 (metadata trust).

#### AV-3: Stripe-metadata-injected credit attack **[v0.1.0 design point]**

**Attack**: An adversary creates a real PaymentIntent on the merchant (e.g., via the customer-facing checkout) but supplies attacker-controlled `metadata.account_id`, `metadata.oauth_provider`, and `metadata.external_id` pointing at a victim's account. The webhook fires legitimately, signature verifies, and the victim's account is credited from the attacker's payment — or, more concerning, an attacker who can write metadata on a refunded / disputed PaymentIntent could credit themselves.

**Mitigation in v0.1.0**: **Partial.** The handler reads `metadata_account_id`, `metadata_oauth_provider`, `metadata_external_id` from the webhook (`stripe_provider.py:212-214`) and reconstructs `AccountIdentity` (`routes.py:741-746`). The metadata is **not cross-checked** against the original purchase initiation (no server-side record of "this PaymentIntent was created for that account"). The handler trusts whatever the merchant-side metadata says.

**Note on idempotency**: `add_purchased_uses` keys credits with `idempotency_key=f"stripe-{payment_id}"` (`app/services/billing.py:715`). The UNIQUE constraint is `(account_id, idempotency_key)` (`models.py:277`) — meaning the **same `payment_id` can credit different accounts** if the webhook is delivered with different metadata. Idempotency does not bind a payment to one account.

**Recommended hot-fix**: persist a `pending_payment_intents` row at PaymentIntent creation time, keyed by `payment_intent.id` with the authoritative `account_id`. The webhook handler MUST look up the row by `payment_intent.id` and use *that* `account_id`, not the metadata. Or: change the credits UNIQUE constraint to be `(idempotency_key)` global rather than `(account_id, idempotency_key)`, so a payment_id is bound to one credit row across all accounts.

**Secondary today**: webhook-confirm round-trip to Stripe (`stripe_provider.py:160`) ensures the payment actually succeeded — it bounds the attack to "real money paid by *some* party" but does not bind it to the right account.

**Residual until fix**: cross-account credit attacks via metadata manipulation. Severity depends on who can set metadata on PaymentIntent creation in the merchant flow.

#### AV-4: Apple StoreKit JWS not verified **[v0.1.0 P0 exposure]**

**Attack**: Attacker posts a forged `signedPayload` JWS to `POST /v1/billing/webhooks/apple-storekit`. The handler decodes the JWS without verifying its signature against Apple's root-certificate chain.

**Mitigation in v0.1.0**: **None at the JWS level.** `_decode_jws` in `app/services/apple_storekit_provider.py:155` explicitly sets `options={"verify_signature": False}` and the comment claims "the HTTPS connection to Apple provides integrity." This rationale is wrong for inbound webhooks — webhooks are pushed *to* the lens; the inbound TLS terminates at the lens edge, not at Apple, so the path is not Apple-authenticated. The same `_decode_jws` is used at `apple_storekit_provider.py:328, 334, 341` to decode the inbound notification, transaction info, and renewal info.

**Bounded today by**: the webhook handler's downstream actions are limited — `is_refund()` triggers a `logger.warning` with a TODO for credit clawback (`routes.py:1415`) and no mutation. Other notification types acknowledge and return. **There is no current mutation on this code path.** The exposure is latent: the moment refund-handling, subscription-renewal credit, or grace-period extension is implemented on this path, the lack of JWS verification becomes a credit-mint vector.

**Recommended hot-fix for v0.1.1**:
1. Implement Apple's chain-validation logic (extract `x5c` cert chain from the JWS header, validate against Apple's root cert, then `jwt.decode(token, leaf_public_key, algorithms=["ES256"])` with `verify_signature=True`).
2. Until (1) lands, **gate the webhook endpoint behind an allowlist of Apple's notification source IPs at the deployment edge**, and add a feature flag `APPLE_WEBHOOK_HANDLERS_ENABLED=false` that short-circuits the handler so refund-clawback PRs cannot be merged without first landing JWS verification.

**Secondary**: outbound calls to App Store Server API (`provider.get_transaction_info`, `routes.py:1237`) on the `/verify` path are over HTTPS-to-Apple — the JWS-decode-without-verify on those response bodies is mitigated by the TLS channel to a known endpoint, though not defense-in-depth.

**Residual until fix**: latent credit-mint vector if any future refund/renewal handler trusts webhook-derived `transaction_info`.

#### AV-5: Google Play webhook lacks client-side signature verification **[v0.1.0 design point]**

**Attack**: Attacker posts a crafted Pub/Sub-shaped payload to `POST /v1/billing/webhooks/google-play`. The handler base64-decodes the inner notification and parses it as authoritative.

**Mitigation in v0.1.0**: **Deployment-edge only.** `google_play_provider.py:212` parses the Pub/Sub envelope and decodes the inner payload (`google_play_provider.py:243`). There is no JWT validation of the Pub/Sub push token, no service-account verification, no Google-public-key check. Authentication is delegated to the deployment's Pub/Sub push subscription configuration (a OIDC token in the `Authorization` header that nginx / the edge proxy is presumed to validate).

**Bounded today by**: like AV-4, the handler currently logs and acknowledges without crediting. The mutation surface on this path is small.

**Recommended hot-fix for v0.1.1**:
1. Validate the `Authorization: Bearer ...` OIDC token attached to Pub/Sub push messages, verifying issuer (`https://accounts.google.com`), audience (the configured webhook URL), and email (the Pub/Sub-publishing service account).
2. Until (1) lands, document that the Pub/Sub push subscription MUST be configured with OIDC authentication and the deployment edge MUST validate it; persist a `WEBHOOK_AUTH_DELEGATED_TO_EDGE=true` toggle so misconfiguration is loud.

**Residual**: same shape as AV-4 — latent credit-mint vector if future handlers credit on this path.

#### AV-6: Forged OAuth ID token (Google / Apple)

**Attack**: Attacker forges a Google or Apple ID token and presents it as `Authorization: Bearer ...` to a JWT-auth endpoint (`/v1/billing/google-play/verify`, `/v1/billing/apple-storekit/verify`, tool routes).

**Mitigation**: tokens are verified against the IdP's published public keys with `jwt.decode(token, public_key, algorithms=["RS256"], audience=client_id, issuer="https://appleid.apple.com")` for Apple (`dependencies.py:508`) and `id_token.verify_oauth2_token(token, ..., client_id)` for Google (`dependencies.py:283`). Audience is checked against the configured `valid_apple_client_ids` / `valid_google_client_ids` lists. Issuer is pinned. Expiry is enforced.

**Secondary**: `token_revocation_service.is_revoked` checked before validation (`dependencies.py:239, 427`).

**Residual**: see AV-7 (multi-client-id loop).

#### AV-7: Multi-client-id audience-loop confused-deputy

**Attack**: Attacker presents an ID token issued for a *different* legitimate audience (e.g., a different CIRIS service's OAuth client) and the multi-client-id loop accepts it.

**Mitigation in v0.1.0**: **Partial.** `dependencies.py:275-327` (Google) and `dependencies.py:499-514` (Apple) loop over `valid_*_client_ids` trying each as the expected audience until one verifies. A token whose audience matches *any* configured client ID is accepted as valid for billing. If a non-billing service's client ID is mistakenly added to `GOOGLE_CLIENT_IDS` / `APPLE_CLIENT_IDS`, tokens issued to that service are accepted here.

**Recommended hardening**:
- Document that `GOOGLE_CLIENT_IDS` / `APPLE_CLIENT_IDS` MUST be the **billing-eligible** client IDs only (not "every client ID across the org").
- For Google, also validate the `azp` (authorized party) claim if present, against an allowlist.
- Consider a single `aud` allowlist across the codebase rather than two env vars.

**Residual**: operational risk — an admin who mis-configures the env var widens the trust boundary.

### 3.2 Replay & Idempotency — adversary wants double-credit / double-charge

#### AV-8: Replay of charge / credit request

**Attack**: Network MITM (or an agent making naïve retries) captures and resubmits a `POST /v1/billing/charges` or `POST /v1/billing/credits` request.

**Mitigation**: Mandatory `idempotency_key` field in request models with `UNIQUE(account_id, idempotency_key)` SQL constraint (`app/db/models.py:197, 277`). Resubmission produces 0 inserts + 1 conflict; the existing row is returned. Same shape on `product_usage` (`models.py:846`).

**Residual**: if an agent reuses the same idempotency key for two *different* logical operations, the second operation is silently accepted as a no-op. Wire-level convention requires idempotency keys to be UUIDs scoped to a single logical operation.

#### AV-9: Replay of provider purchase event

**Attack**: Attacker captures an Apple StoreKit `transaction_id` or Google Play `purchase_token` (e.g., from logs or via a man-on-the-side at the edge) and replays it to `/verify`.

**Mitigation**: SQL `UNIQUE(transaction_id)` on `apple_storekit_purchases` (`models.py:643`) and `UNIQUE(purchase_token)` on `google_play_purchases` (`models.py:583`). Replay → idempotent re-fetch path returns `already_processed=true` (`routes.py:1183-1203`); no second credit grant.

**Residual**: cross-deployment replay (the same purchase_token on a *different* CIRISBilling instance that hasn't seen it yet) lands once, by design — that's expected behavior in a multi-region deploy and is each region's local guarantee.

#### AV-10: Stripe webhook redelivery

**Attack**: Stripe's at-least-once webhook delivery duplicates `payment_intent.succeeded`.

**Mitigation**: `stripe_provider.confirm_payment` re-fetches the PaymentIntent at every webhook (`routes.py:737`) and credit attribution in `add_purchased_uses` is idempotency-keyed by `payment_id` at the BillingService layer (verify in `app/services/billing.py`). Combined with Stripe's signature verification, replay is bounded.

**Residual**: if `add_purchased_uses` is called twice within the same DB transaction window without a UNIQUE-on-payment_id guard, double-credit is possible. **TODO**: confirm `payments` / `purchases` tables enforce `UNIQUE(payment_id)` for Stripe-sourced credits.

### 3.3 Authentication-bypass — adversary wants to act as another principal

#### AV-11: Admin OAuth state-token CSRF

**Attack**: Attacker tricks a victim into clicking a crafted `/admin/oauth/callback?code=&state=` URL after starting their own OAuth flow, fixating their own session into the victim's browser.

**Mitigation**: `state` parameter is generated and stored in `initiate_oauth_flow` (`admin_auth_routes.py:72-74`); `handle_oauth_callback` validates `state` matches the stored value (`admin_auth_routes.py:115`). Implementation-level CSRF protection.

**Secondary**: `SameSite=lax` on the session cookie (`admin_auth_routes.py:143`) bounds top-level POST/iframe attacks.

**Residual**: depends on `AdminAuthService.handle_oauth_callback`'s state-storage backend (in-memory vs DB). In-memory state storage breaks under multi-worker deployments unless sticky sessions are configured.

#### AV-12: Admin JWT-secret weak / leaked

**Attack**: Attacker recovers `ADMIN_JWT_SECRET` (e.g., from `.env` checked into git) and forges admin session JWTs.

**Mitigation**: `ADMIN_JWT_SECRET` is loaded from environment / Docker secret (`config.py:51`); `SECURITY_IMPLEMENTATION_PLAN.md` Phase 1 moves it to Docker secret files. Validation requires the secret to be present at startup; absence does not cause graceful fallback.

**Recommended hardening**:
- Add a `model_validator` to `Settings` requiring `len(ADMIN_JWT_SECRET) >= 32` when `environment=production` (matching the pattern at `config.py:171-177` for test auth).
- Rotate admin JWTs by issuing a new secret and short-lived JWTs (currently 24h `jwt_expire_hours`).

**Residual**: secret rotation has no operational tooling beyond service restart; rolling secret rotation requires support for two valid secrets during the cutover window.

#### AV-13: Admin OAuth token in URL query string **[v0.1.0 P1 exposure]**

**Attack**: The OAuth callback redirects with `?token={access_token}` in the URL (`admin_auth_routes.py:134`). Tokens leak into:
- Browser history
- `Referer` header on subsequent navigations off-site
- Reverse-proxy access logs (nginx default `combined` log format includes the URL with query string)
- Browser extensions and developer-tools sessions
- Server-side observability tooling that captures URLs at INFO level

**Mitigation in v0.1.0**: **None.** Both the cookie (HttpOnly + Secure + SameSite=lax) AND a URL-query token are issued. The URL token is presumably intended for SPAs that read it client-side, but its very presence in the URL nullifies the cookie's HttpOnly protection.

**Recommended hot-fix for v0.1.1**:
- Drop the `?token={access_token}` query parameter; redirect to `redirect_uri` plain.
- If the SPA needs the token client-side, expose it via a separate authenticated `GET /admin/oauth/user` call (already exists at `admin_auth_routes.py:192-248`) that reads the cookie and returns a same-origin JSON body.

**Residual until fix**: tokens in proxy logs are recoverable for the log retention window. Compromise of any log-aggregation tier exposes recent admin sessions.

#### AV-14: Test-auth bypass left enabled in production

**Attack**: Operator misconfigures `CIRIS_TEST_AUTH_ENABLED=true` in production with a known `CIRIS_TEST_AUTH_TOKEN`.

**Mitigation**: Startup validation refuses to boot if `CIRIS_TEST_AUTH_ENABLED=true` AND `environment=production` (`config.py:171-173`). Token must be ≥64 chars when test auth is enabled (`config.py:174`). Compared with `secrets.compare_digest` (constant-time) against the configured token (`dependencies.py:224, 412`).

**Residual**: if `environment` env var is not set to `production`, the gate is bypassed. Deploy tooling must always set the env var explicitly.

#### AV-15: API-key leak via HTTP request logs

**Attack**: API keys present in `X-API-Key` header are logged at INFO/DEBUG, exposing them to log-aggregation tooling.

**Mitigation**: structlog default formatters do not capture headers; `logging_middleware` (`main.py:152-219`) logs `method`, `path`, `status_code`, `duration` only. No header capture is observed.

**Residual**: third-party request middleware (e.g., the OTEL FastAPI instrumentor at `main.py:119`) may capture headers. Confirm `OTEL_PYTHON_INSTRUMENTATION_HTTP_CAPTURE_HEADERS_*` is unset or excludes `X-API-Key` and `Authorization`.

### 3.4 Denial of Service — adversary wants the API unable to serve

#### AV-16: Body-size flood (no max body limit) **[v0.1.0 exposure]**

**Attack**: Attacker submits arbitrarily large bodies to any `/v1/billing/*` endpoint. Starlette / FastAPI's default body extractor reads the entire body into memory.

**Mitigation in v0.1.0**: **None at the application level.** Defense relies on nginx's `client_max_body_size` directive at the deployment edge (default 1 MiB). No FastAPI `DefaultBodyLimit` middleware is configured.

**Recommended hot-fix for v0.1.1**: explicit body-size cap via Starlette middleware or a per-route check. 1-2 MiB is sufficient for any current endpoint.

**Residual**: deployment-edge gap if nginx is misconfigured.

#### AV-17: Connection-pool exhaustion

**Attack**: Attacker holds open many concurrent slow-body POSTs, exhausting the SQLAlchemy connection pool (`database_pool_size=25`, `database_max_overflow=10`, `database_pool_timeout=30` from `config.py:26-28`).

**Mitigation in v0.1.0**: PgBouncer connection pooling at the DB layer absorbs some of this. `database_pool_timeout=30` causes new requests to fail-fast after 30s if the pool is saturated.

**Recommended hardening**: per-IP rate-limit at nginx; consider a `statement_timeout` at the Postgres session level to bound long-running queries.

**Residual**: distributed-source flood requires upstream DDoS protection (Cloudflare, etc.).

#### AV-18: Argon2id-hash CPU-burn DoS

**Attack**: Attacker spams invalid API keys to force Argon2id verification CPU work on each request.

**Mitigation in v0.1.0**: `key_prefix` is checked first as an indexed lookup (`models.py:365` is `unique=True`), failing fast for non-existent prefixes without invoking Argon2. Argon2 is only invoked for valid-prefix keys.

**Residual**: if attacker enumerates valid prefixes (each is logged on success, public if log-aggregation leaks), they can target specific keys for verify-CPU exhaustion. The Argon2id default time-cost is moderate; tune in `app/services/api_key.py:79` if CPU is a hot spot.

#### AV-19: Token-cache memory unbounded

**Attack**: Attacker submits many distinct invalid Google / Apple ID tokens to grow the in-memory token cache without bound.

**Mitigation in v0.1.0**: `_cleanup_google_token_cache` / `_cleanup_apple_token_cache` enforce `_MAX_CACHE_SIZE` (`dependencies.py:67-90`). Caches store **verified** entries only — invalid tokens never enter the cache. Bound is on legitimate-token diversity, not adversarial.

**Residual**: across multi-worker deployments, each worker has its own cache; total memory is N × _MAX_CACHE_SIZE.

### 3.5 Data Corruption — adversary wants false rows persisted or true rows dropped

#### AV-20: SQL injection via JSONB / parameter binding

**Attack**: Attacker crafts payload content (e.g., `metadata` blobs, account email) hoping for string-interpolated SQL.

**Mitigation**: SQLAlchemy ORM with parameterized binding throughout. No raw string-format SQL observed in repo. `migration_runner.py` uses Alembic with parameterized DDL.

**Residual**: Phase-2 query surfaces that read from JSONB columns with `->>` operators must keep the path operands parameterized too.

#### AV-21: Privilege escalation via permission scope bypass

**Attack**: Attacker with a `billing:read`-only API key calls a `billing:write` endpoint.

**Mitigation**: Per-endpoint `if _PERM_BILLING_WRITE not in auth.api_key.permissions` checks (`routes.py:1146-1150` and parallel sites). Permission denial → HTTP 403.

**Residual**: scope checks are duplicated at each endpoint; a missed check on a future endpoint is a defect class. Recommend a single FastAPI dependency `requires_permission("billing:write")` that all mutating routes share.

#### AV-22: Account-takeover via OAuth-identity collision

**Attack**: Attacker registers an OAuth identity with an `external_id` that collides with a legitimate user's expected `external_id`.

**Mitigation**: `accounts` has `UNIQUE(oauth_provider, external_id)` (`models.py:125-128`). The first registrant wins; subsequent attempts to register the same `(oauth_provider, external_id)` resolve to the same account row. OAuth providers' `sub` claims are guaranteed-unique-per-account by the IdP.

**Residual**: Apple's "Hide My Email" and email-rotation features mean `email` is not a stable identifier; only `sub` is. The schema correctly indexes on `external_id` (the `sub`), not `email`.

### 3.6 Privacy — adversary wants to read or exfiltrate sensitive data

#### AV-23: PII leak via error-path logging **[v0.1.0 exposure]**

**Attack**: Request body that fails JSON parsing logs the first 500 bytes of the body verbatim, including any PII it contains.

**Mitigation in v0.1.0**: **Partial.** `routes.py:1591` logs `raw_body=body.decode()[:500]` on parse error. Body content for the `log-usage` endpoint may include user-identifying or interaction-content data.

**Recommended hot-fix for v0.1.1**: replace verbatim body logging with structured-error-only logging:
- Log `body_length` (numeric) and `error_type` only on parse failure.
- Keep verbose body in tracing-only logs that are not shipped to the centralized log aggregator.

**Residual**: until the fix, log retention is the exposure window.

#### AV-24: Token-prefix logging at INFO level

**Attack**: Token first 20-30 chars logged at INFO in `dependencies.py:268` (Google), `dependencies.py:493` (Apple), `admin_auth_routes.py:151` (admin JWT). 20-30 chars of a JWT is insufficient for forgery, but reveals the header (alg/kid) and may reveal partial payload-base64 — useful for fingerprinting which keys are in use.

**Mitigation in v0.1.0**: **None.** These are debug-grade logs left at INFO level, presumably to support the auth-debug commits in recent history (`b8b90b9`, `9ec3e19`).

**Recommended hot-fix for v0.1.1**: drop these logs to DEBUG (or remove). Token-prefix exposure is low-value-for-attacker, low-value-for-debugging-in-prod; keep the debug detail in development environments only.

**Residual**: minor information disclosure.

#### AV-25: Provider-secret plaintext at rest **[v0.1.0 design point]**

**Attack**: Database compromise discloses Stripe API keys (`sk_test_...` / `sk_live_...`), Apple StoreKit private keys (.p8 contents), and Google Play service account JSON stored in `provider_configs`.

**Mitigation in v0.1.0**: **None.** `SECURITY_IMPLEMENTATION_PLAN.md` Phase 2 explicitly tracks this: "Add cryptography to requirements.txt; create encryption service; create migration script for Stripe keys." Keys are stored unencrypted.

**Bounded today by**: Postgres-server compromise is named as out-of-scope in §1, so this is consistent with the threat model. But defense-in-depth would cap the blast radius of a leaked DB dump.

**Recommended for v0.2.x** (per `SECURITY_IMPLEMENTATION_PLAN.md` Phase 2): envelope-encrypt provider secrets with a KMS-managed key; the DB stores only ciphertext + key-id. Decryption happens in-process at provider-init time.

**Residual**: until Phase 2, DB-dump leakage discloses all merchant secrets simultaneously.

#### AV-26: API-key rotation grace period non-functional **[v0.1.0 P1 functional]**

**Attack** (operational, not adversarial): an admin rotates a production API key via `POST /admin/api-keys/{key_id}/rotate` (`admin_routes.py:657`). The endpoint advertises and the client model (`APIKeyRotateResponse.old_key_expires_at`) returns a 24-hour grace period during which the old key should continue to work (`admin_routes.py:122-124`).

**What actually happens**: `rotate_api_key` (`api_key.py:289`) sets the old key's `status = "rotating"` and stores `key_metadata.grace_period_until`. But `validate_api_key` (`api_key.py:200`) filters lookups by `status = "active"` — `"rotating"` keys are immediately rejected with `"Invalid API key"`. **The grace period stored in `key_metadata` is never consulted by the validation path.** Old keys break the moment rotation is triggered.

**Severity**: this is a security-relevant correctness defect. Operators told they have 24 hours to roll a credential plan their dependency-update window accordingly, then discover at cutover that the old key has stopped working — pressuring them to skip rotation entirely or schedule emergency maintenance windows. It also undermines the "rotate compromised key" incident-response story: the new key issuance and old key revocation are simultaneous, with no overlap window for clients to update.

**Mitigation in v0.1.0**: **None.** The CHECK constraint at `models.py:406` accepts `('active', 'rotating', 'revoked')`, but the validation path at `api_key.py:200` ignores `'rotating'`.

**Recommended hot-fix for v0.1.1**: change `validate_api_key` to accept both `active` and `rotating` keys, and check the grace period from `key_metadata.grace_period_until`:
```python
stmt = select(APIKey).where(
    APIKey.key_prefix == key_prefix,
    APIKey.status.in_(["active", "rotating"]),
)
# After hash verify:
if api_key.status == "rotating":
    grace_until = api_key.key_metadata.get("grace_period_until")
    if grace_until and datetime.now(UTC) > datetime.fromisoformat(grace_until):
        # Auto-revoke after grace period
        api_key.status = "revoked"
        await self.db.commit()
        raise AuthenticationError("API key rotation grace period expired")
```
Add a periodic job (or lazy check at validation) to flip `rotating → revoked` when the grace period elapses. Add a test that exercises the full rotation cycle: rotate → old key works → grace expires → old key fails.

**Residual**: operators rotating before this fix lands have no safe path; advise them to issue NEW keys, update clients to the new key, then `revoke` the old separately. Don't use `rotate` until fixed.

#### AV-27: OAuth admin session state in per-worker memory **[v0.1.0 P0 functional + DoS]**

**Attack 1 (functional)**: in the production `docker-compose.yml` deployment, three stateless API workers sit behind `nginx.conf`'s `least_conn` upstream (`docker/nginx/nginx.conf:11-16`). Admin OAuth state is stored in `AdminAuthService._sessions: dict[str, OAuthSession]` (`admin_auth.py:35`), an in-process dict. `admin_auth_routes.py:21-41` keeps a module-level singleton per worker. When a user clicks "Login with Google":
- `GET /admin/oauth/login` lands on worker A → state stored in A's `_sessions`.
- Browser redirects to Google, then back to `/admin/oauth/callback?state=...`.
- That callback is load-balanced — with three workers, ~67% of the time it lands on worker B or C, where `self._sessions.get(state)` returns `None` → `ValueError("Invalid OAuth state")` → HTTP 400 (`admin_auth.py:74-75`).

Login fails for ~2 of every 3 attempts in the multi-worker deployment.

**Attack 2 (DoS / memory leak)**: `_sessions` is only deleted on the success path (`admin_auth.py:98`). Abandoned, errored, or attacker-initiated flows leave entries in the dict forever. There is no TTL, no cleanup, no max size. An attacker spamming `/admin/oauth/login` (rate-limited at nginx to 100 req/min on `admin_limit`, `docker/nginx/admin-nginx.conf:109`) grows each worker's `_sessions` dict at up to 100 entries/min — modest, but unbounded over time. Each `OAuthSession` holds three short strings, so the leak is slow, but it's not zero.

**Mitigation in v0.1.0**: **None.** No DB-backed session store; no Redis; no eviction.

**Recommended hot-fix for v0.1.1**:
1. **Persist OAuth state in Postgres**: `oauth_sessions` table with `(state, redirect_uri, callback_url, created_at, expires_at)`. Insert on `initiate_oauth_flow`; lookup-and-delete (transaction) on `handle_oauth_callback`. Add a periodic `DELETE WHERE expires_at < NOW()` to bound table growth. State expiry: 10 minutes (matches Google's authorization code expiry).
2. **Or, as a stop-gap**: add `ip_hash` or sticky sessions in `nginx.conf` to ensure the callback lands on the same worker as the initiate. Cheap but fragile (worker restart loses state); not a real fix.

**Secondary**: in-memory state across multi-worker is also broken for the production `admin-nginx.conf` deployment (single backend `ciris-billing-api:8000`, `docker/nginx/admin-nginx.conf:113-115`) only if that single backend is itself multi-worker (uvicorn `--workers > 1`). Confirm worker count in `Dockerfile` / `start.sh`.

**Residual until fix**: admin login is unreliable in production; memory grows slowly under attack.

#### AV-28: Hardcoded admin-bootstrap email **[v0.1.0 design point]**

**Attack**: An attacker who can authenticate as the email `eric@ciris.ai` becomes admin on first login.

**Mitigation in v0.1.0**: **Layered.** `admin_auth.py:147` hardcodes `role = "admin" if oauth_user.email == "eric@ciris.ai" else "viewer"`. The defense-in-depth is `google_oauth.py:101`: `if not email.endswith("@ciris.ai")` — the email-suffix check rejects any non-`ciris.ai` email regardless of Google account ownership. The `hd` query parameter at `google_oauth.py:53` is a Google UX hint, not a security check; the suffix check is the actual gate.

For the current single-tenant CIRIS deployment with a Google Workspace at `ciris.ai`, only Workspace members can authenticate, and `eric@ciris.ai` is a known operator account — the hardcoded admin email is effectively a bootstrap seed. Other Workspace members get `role="viewer"` on first login and require an existing admin to promote them (`PATCH /admin/users/{id}/role` if such a route exists; verify in `admin_routes.py`).

**Risks**:
- If the Workspace is reconfigured to allow guest accounts, or if `hd_domain` / suffix check is changed, an attacker with a `ciris.ai` email becomes admin without further checks.
- `endswith("@ciris.ai")` is case-sensitive; Google's userinfo returns email in lowercase, so this is fine in practice — but `email.lower()` would be defense-in-depth.
- The `hd` query parameter is not enforced by Google; users can override it. The suffix check is the only real gate.

**Recommended hardening for v0.2.x**:
- Replace the hardcoded email with a `BOOTSTRAP_ADMIN_EMAIL` env var, validated at startup.
- Lowercase email in the suffix check.
- Add an explicit "first user is admin only if the database is empty" gate (the `user_count == 0` check at `admin_auth.py:150` exists but layered with the email-equality check, which makes the order subtle).
- Consider switching the suffix check to `domain == "ciris.ai"` after splitting at `@` (hardens against `eric@notciris.ai`-style edge cases — though `endswith("@ciris.ai")` already covers most).

**Residual**: bootstrap-time hardcoded email; documented operational dependency.

#### AV-29: Inconsistent permission-scope enforcement

**Attack**: A future endpoint omits its scope check; a `billing:read`-only key reaches a `billing:write` operation.

**Mitigation in v0.1.0**: **Layered, partially centralized.** `dependencies.py:764` defines `require_permission(perm)` and `dependencies.py:866` defines `require_permission_or_jwt(perm)` — centralized helpers exist and are used by the main billing routes (`routes.py:177, 288`; `tool_routes.py:232`). However, several endpoints duplicate the check inline:
- `routes.py:113-118` (`/v1/billing/credits/check`) — manual check after `get_api_key_or_jwt`
- `routes.py:1146-1150` (`/v1/billing/apple-storekit/verify`) — manual check
- `routes.py:870` parallel pattern for `/v1/billing/google-play/verify`

These manual checks mirror what the dependency helpers do, but the duplication makes a missed check on a future endpoint a defect class.

**Recommended hardening for v0.2.x**: migrate all authenticated mutating routes to use `require_permission_or_jwt(...)` exclusively. Add a lint test that asserts every `POST`/`PATCH`/`DELETE` route on the `/v1/billing/*` prefix has either an `api_key`-scoped or `admin`-scoped dependency.

**Residual**: missed-check defect class. Not currently exploited.

### 3.7 Operational hardening (catalogued for v0.2.x track)

These vectors are deferred but tracked, mirroring CIRISPersist §3.6:

- **OP-1** — CORS wildcard with credentials. `app/main.py:142-148` sets `allow_origins=["*"]` with `allow_credentials=True`. Browsers reject this combination (effectively neutering credentials), but it signals insecure intent. Tighten to an explicit allowlist sourced from a `settings.cors_allowed_origins` field.
- **OP-2** — No per-API-key rate limiting. nginx provides per-IP rate limiting (`api_limit` 60r/m, `admin_limit` 100r/m, `login_limit` 5r/m at `docker/nginx/admin-nginx.conf:108-110`); per-key limits would require app-layer enforcement.
- **OP-3** — No `statement_timeout` on Postgres sessions. Track for v0.2.x.
- **OP-4** — No graceful-shutdown drain of in-flight requests. `lifespan` handler closes engines on shutdown (`main.py:64`) but does not wait for in-flight requests. SIGTERM may drop in-flight charges. Match CIRISPersist AV-19 pattern.
- **OP-5** — Per-class-state token-revocation cache (`token_revocation.py:57` `_cache: ClassVar[dict[...]]`). Multi-worker deployments have N independent caches; revocations propagate via DB on cache miss. Functionally correct (DB is source of truth), but cache-warmth is per-worker.
- **OP-6** — No clock-skew validation on `Stripe-Signature` `t=` timestamp beyond Stripe SDK defaults (default tolerance: 300 seconds). Acceptable; documented for awareness.
- **OP-7** — No `X-Frame-Options` / `Content-Security-Policy` headers on admin-UI responses. Add via middleware on the `/admin-ui/{path:path}` route at `main.py:242-294`.
- **OP-8** — `admin_dependencies.py:22-34` constructs a new `AdminAuthService` (and a new `GoogleOAuthProvider` with its own `httpx.AsyncClient`) on every request. This means `_sessions` is empty for the validation path (which is fine — the validation path only uses `verify_jwt_token` which is stateless), but it leaks an HTTP client per request. Convert to a singleton like `admin_auth_routes.py:21-41` does.

---

## 4. Mitigation Matrix

| AV | Attack | Primary Mitigation (v0.1.0) | Secondary | Status | Fix Tracker |
|---|---|---|---|---|---|
| AV-1 | Forged credit via direct API call | Argon2id API-key auth + permission scopes | Logged auth failures | ✓ Mitigated | — |
| AV-2 | Forged Stripe webhook | HMAC sig verify (`construct_event`) | Round-trip `confirm_payment` | ✓ Mitigated | — |
| AV-3 | Stripe-metadata-injected credit | Metadata-only identity reconstruction | confirm_payment ensures real payment | ⚠ Metadata trust gap | v0.1.1 hot-fix |
| AV-4 | Apple StoreKit JWS not verified | (none — `verify_signature=False`) | No mutation handler today | **⚠ P0 latent** | **v0.1.1 hot-fix** |
| AV-5 | Google Play webhook unsigned at app layer | (deployment-edge OIDC delegation) | No mutation handler today | ⚠ Edge-delegated | v0.1.1 hot-fix |
| AV-6 | Forged OAuth ID token | Provider public-key + audience + issuer + expiry | Token revocation list | ✓ Mitigated | — |
| AV-7 | Multi-client-id confused-deputy | aud-loop allowlist | (no `azp` check) | ⚠ Operator-config risk | v0.2.x |
| AV-8 | Replay of charge / credit | `UNIQUE(account_id, idempotency_key)` SQL | Mandatory idempotency_key | ✓ Mitigated | — |
| AV-9 | Replay of provider purchase | `UNIQUE(transaction_id)` / `UNIQUE(purchase_token)` | already_processed branch | ✓ Mitigated | — |
| AV-10 | Stripe webhook redelivery | confirm_payment re-fetch + idempotency at BillingService | — | ⚠ Confirm UNIQUE(payment_id) | v0.1.1 audit |
| AV-11 | Admin OAuth state CSRF | state-token validation in `handle_oauth_callback` | SameSite=lax cookie | ✓ Mitigated | — |
| AV-12 | Admin JWT-secret weak / leaked | env-loaded secret; 24h JWT expiry | (no length validator) | ⚠ Add length validator | v0.1.1 |
| AV-13 | OAuth token in URL query | (none) | Cookie also set | **⚠ P1 disclosure** | **v0.1.1 hot-fix** |
| AV-14 | Test-auth in production | `validate_critical_config` refuses | secrets.compare_digest | ✓ Mitigated | — |
| AV-15 | API-key leak via logs | structlog excludes headers | (audit OTEL config) | ✓ Mitigated; ⚠ verify OTEL | v0.1.1 audit |
| AV-16 | Body-size flood | (deployment-edge nginx only) | — | ⚠ Add app-level cap | v0.1.1 |
| AV-17 | Connection-pool exhaustion | `pool_timeout=30` | PgBouncer | ⚠ Add per-IP rate limit | v0.2.x |
| AV-18 | Argon2id CPU-burn DoS | Indexed prefix lookup before hash | — | ✓ Mitigated | — |
| AV-19 | Token-cache memory unbounded | `_MAX_CACHE_SIZE` cleanup | Verified-only entries | ✓ Mitigated | — |
| AV-20 | SQL injection | Parameterized ORM | Alembic typed migrations | ✓ Mitigated | Phase-2 audit |
| AV-21 | Permission scope bypass | Per-endpoint scope check | — | ⚠ Centralize in dependency | v0.2.x |
| AV-22 | Account-takeover via identity collision | `UNIQUE(oauth_provider, external_id)` | Provider-issued sub claim | ✓ Mitigated | — |
| AV-23 | PII leak via error logs | (raw_body[:500] logged) | — | **⚠ P1 leakage** | **v0.1.1 hot-fix** |
| AV-24 | Token-prefix logging | (INFO-level token preview) | — | ⚠ Drop to DEBUG | v0.1.1 |
| AV-25 | Provider-secret plaintext | (env / DB plaintext) | Postgres ACLs | ⚠ KMS envelope encrypt | v0.2.x (Phase 2) |
| AV-26 | API-key rotation grace period non-functional | (none — `validate_api_key` rejects `status="rotating"`) | Manual issue-new-then-revoke-old workaround | **⚠ P1 functional** | **v0.1.1 hot-fix** |
| AV-27 | OAuth admin state per-worker memory | (none — `_sessions: dict` per-process) | `admin-nginx.conf` single backend mitigates if uvicorn workers=1 | **⚠ P0 functional + DoS leak** | **v0.1.1 hot-fix** |
| AV-28 | Hardcoded admin-bootstrap email | `endswith("@ciris.ai")` domain gate | `user_count == 0` first-user check | ⚠ Document & make configurable | v0.2.x |
| AV-29 | Inconsistent permission-scope enforcement | Centralized `require_permission_or_jwt` exists | Duplicated inline checks at 3 sites | ⚠ Consolidate | v0.2.x |

---

## 5. Security Levels by Deployment Tier

| Tier | Backend | Auth Edge | Threat Model |
|---|---|---|---|
| **Production multi-instance** | Postgres primary/replica + PgBouncer + nginx LB | TLS at nginx; Docker secrets for keys | Full §3 model. AV-4 / AV-5 / AV-13 / AV-23 hot-fixes required. |
| **Single-host Docker compose** | Single Postgres, no replica | TLS optional; .env file secrets | §3 applies; secrets-at-rest exposure higher (AV-25). Operator must enforce `CIRIS_TEST_AUTH_ENABLED=false` and `environment=production`. |
| **Local development** | Docker Postgres + .env | No TLS | Test-auth bypass enabled. NEVER expose to public network. `validate_critical_config` enforces this only when `environment=production` is set. |
| **Stripe-only / Apple-only / Google-only** | Subset of providers | Same as production | Disabled providers reduce attack surface; `provider_configs` row absence → 503 on related endpoints. |

Critical invariant: **all tiers run the same FastAPI app, same SQLAlchemy models, same auth dependencies**. A finding in one tier's implementation is presumed to apply to all unless explicitly excepted.

---

## 6. Security Assumptions

The system depends on these assumptions; if violated, the threat model breaks.

1. **TLS at the deployment edge**: nginx (or equivalent) terminates HTTPS for all inbound traffic. Plaintext HTTP exposes API keys, OAuth tokens, and webhook bodies. AV-2 / AV-6 / AV-15 all assume this.
2. **Postgres backend integrity**: the database accepts writes atomically; the SQL UNIQUE constraints are honored. Multi-AZ Postgres deployments provide this; single-instance deployments inherit Postgres's standard durability.
3. **Provider trust**: Stripe / Apple / Google signature infrastructure is intact at the issuer. ID tokens / webhook signatures from those providers, when verified, are accepted as authoritative.
4. **OAuth-provider `sub` stability**: Google's and Apple's `sub` claim is stable per-user-per-app. Account-identity reconstruction (`AccountIdentity(oauth_provider, external_id)`) depends on this. (Apple's "Hide My Email" rotates email, not sub — schema correctly indexes on sub.)
5. **Env / Docker-secret confidentiality**: `ADMIN_JWT_SECRET`, `stripe_api_key`, `APPLE_STOREKIT_PRIVATE_KEY`, `PLAY_INTEGRITY_SERVICE_ACCOUNT` are read from non-world-readable files. `SECURITY_IMPLEMENTATION_PLAN.md` Phase 1 is the operational doc enforcing this.
6. **Clock accuracy**: Postgres and API hosts are within ~5 minutes of UTC. JWT expiry (`exp`) checks and Stripe-Signature `t=` checks rely on this.
7. **Environment variable correctness**: `environment=production` in production. Without it, the test-auth gate at `config.py:171-173` fails open.
8. **Single-version wire compatibility**: clients (CIRISAgent, CIRISProxy, mobile apps) and this service agree on request/response shapes. Schema drift is the AV-class for incoming requests.

---

## 7. Fail-Secure Degradation

All failures degrade to MORE restrictive modes, never less. This mirrors CIRISVerify §7 and CIRISPersist §7.

| Failure Condition | Behavior | Rationale |
|---|---|---|
| Missing API key on `billing:write` route | HTTP 401; zero rows mutated | Cannot authorize → cannot mutate |
| Missing `billing:write` permission | HTTP 403; zero rows mutated | Cannot authorize this scope |
| Stripe signature verification failure | HTTP 401; zero rows mutated | Unverified webhook never persists |
| `payment_intent.succeeded` missing metadata | HTTP 400; zero rows mutated | Cannot attribute to an account |
| OAuth public-key fetch failure | HTTP 500; auth fails | Cannot verify → cannot authenticate |
| OAuth token revoked | HTTP 401 | Explicit revocation honored |
| OAuth audience mismatch | HTTP 401 | Token not for this service |
| `idempotency_key` collision | Existing row returned; no second mutation | UNIQUE constraint enforced |
| Provider not configured (Stripe/Apple/Google) | HTTP 503 | Fail closed, not silent |
| Apple webhook handler refund (currently TODO) | Logged, no mutation | Latent surface; do not credit until JWS verification lands |
| `environment=production` AND test-auth enabled | App refuses to start (`ConfigurationError`) | Fail-fast on dangerous config |
| `database_url` empty or non-Postgres | App refuses to start | Fail-fast on broken config |
| Postgres unreachable | HTTP 5xx; pool timeout 30s | Surface backpressure honestly |
| Argon2 hash verify failure | HTTP 401 | Constant-time path |
| Validation error (Pydantic) | HTTP 422 with sanitized errors | Schema-altering input rejected |

Critical invariant: **rows with un-verified provider attribution do not persist**. The Apple `/verify` path round-trips to App Store Server API before crediting (`routes.py:1237`); the Google Play `/verify` path round-trips to Play Developer API; the Stripe webhook re-fetches PaymentIntent before crediting. Webhook-only attribution without round-trip is the AV-3 / AV-4 / AV-5 surface that v0.1.1 must close.

---

## 8. Residual Risks

Risks CIRISBilling mitigates but cannot fully eliminate:

1. **Compromised agent API key** (AV-1 residual). The persistence layer accepts authenticated mutations. Closure: agent-side key storage hardening (CIRISVerify hardware-attestation tiers); short-lived API keys with rotation; per-key rate limiting at v0.2.x.

2. **Compromised payment-provider merchant account**. Stripe / Apple / Google issuing real signatures on attacker-controlled events is undetectable at this layer. Closure: provider-side fraud detection; CIRIS-side ledger reconciliation against provider dashboards; eventual chargeback handling.

3. **Apple StoreKit JWS unverified** (AV-4). **Latent today; P0 when refund-handling lands.** Hot-fix: implement Apple's `x5c`-chain validation in `_decode_jws` before any mutation handler is added to that path.

4. **Google Play Pub/Sub OIDC delegated to edge** (AV-5). Same shape as AV-4. Hot-fix: validate Pub/Sub OIDC token in the application layer.

5. **Stripe metadata trust** (AV-3). Hot-fix: `pending_payment_intents` server-side row keyed by PaymentIntent ID, not metadata.

6. **OAuth token in URL query** (AV-13). P1 disclosure surface. Hot-fix: drop `?token=` from the redirect.

7. **Raw body in error logs** (AV-23). P1 disclosure. Hot-fix: log `body_length` only.

8. **Provider secrets plaintext at rest** (AV-25). Tracked under `SECURITY_IMPLEMENTATION_PLAN.md` Phase 2. Postgres compromise is out-of-scope but defense-in-depth wants envelope-encryption.

9. **Multi-worker OAuth state-store consistency** (AV-11 residual). State stored in-process if not DB-backed; sticky sessions or DB-backed state required.

10. **Post-quantum signature replacement**. RS256 / HS256 / ECDSA are classical-cryptography only. No PQC in this service today. Tracked for the broader CIRIS hybrid-signature roadmap, not v0.1.x.

---

## 9. v0.1.0 Threat Posture Summary

```
v0.1.1 STATUS — wave 1-4 cleanup landed (2026-05-01)
  ✓ AV-3  Stripe metadata trust                     (closed — stripe_payment_intents table is authoritative)
  ✓ AV-4  Apple StoreKit JWS not verified           (closed — `_decode_jws` always validates x5c chain pinned to Apple Root CA - G3, both inbound webhooks and outbound API responses)
  ✓ AV-5  Google Play webhook app-layer unsigned    (closed — GOOGLE_PUBSUB_VERIFY_OIDC validates Pub/Sub Bearer token)
  ✓ AV-12 ADMIN_JWT_SECRET length validator         (closed — startup config validator)
  ✓ AV-13 OAuth admin token in URL query string     (closed — redirect drops `?token=`)
  ✓ AV-23 raw_body[:500] logged on parse error      (closed — debug endpoint removed)
  ✓ AV-24 Token-prefix INFO logs                    (closed — dropped to DEBUG / removed)
  ✓ AV-26 API-key rotation grace period             (closed — validate_api_key honors `rotating` + `grace_period_until`)
  ✓ AV-27 OAuth admin state per-worker memory       (closed — admin_oauth_sessions table)

v0.1.0 P1 EXPOSURES (still open)
  ⚠ AV-16 No application-layer body-size cap        (nginx `client_max_body_size 1M` mitigates)

v0.1.x TRACK
  ⚠ AV-7  Multi-client-id confused-deputy hardening (azp check)
  ⚠ AV-12 ADMIN_JWT_SECRET length validator
  ⚠ AV-24 Drop token-prefix INFO logs to DEBUG
  ⚠ AV-10 Confirm credits UNIQUE binds payment_id globally (currently per-account)
  ⚠ AV-28 Replace hardcoded admin email with BOOTSTRAP_ADMIN_EMAIL env var

v0.2.x TRACK
  ⚠ AV-29 Consolidate permission-scope checks into a single dependency
  ⚠ AV-25 Envelope-encrypt provider secrets with KMS (SEC_IMPL_PLAN Phase 2)
  ⚠ OP-2  Per-API-key rate limiting (in addition to nginx per-IP)
  ⚠ OP-3  Postgres statement_timeout
  ⚠ OP-4  Graceful-shutdown drain of in-flight requests
  ⚠ OP-7  CSP / X-Frame-Options on admin-UI responses
  ⚠ OP-8  Singleton AdminAuthService in admin_dependencies (reuse pattern)

DESIGN-DECISIONS-PER-MISSION (intentional, not defects)
  ✓ AV-1  API-key Argon2id + permission scopes
  ✓ AV-2  Stripe webhook HMAC verification
  ✓ AV-6  OAuth provider public-key + audience + issuer + expiry
  ✓ AV-8  UNIQUE(account_id, idempotency_key) on charges/credits
  ✓ AV-9  UNIQUE(transaction_id) / UNIQUE(purchase_token) on purchases
  ✓ AV-11 OAuth state-token CSRF protection
  ✓ AV-14 environment=production gate on test-auth
  ✓ AV-18 Indexed prefix lookup before Argon2 verify
  ✓ AV-20 Parameterized ORM throughout
  ✓ AV-22 UNIQUE(oauth_provider, external_id) on accounts
```

---

## 10. Update Cadence

This document is updated:
- On every minor version (v0.1.x → v0.2.0): comprehensive review.
- On every published security advisory affecting deps: addendum in §3 + `pip-audit` re-run.
- On every new payment provider integration: new attack vectors added for the new webhook + verify surfaces.
- On every auth-flow change: AV-6 / AV-7 / AV-11 / AV-12 / AV-13 review.

**Last updated**: 2026-05-01 (v0.1.0 baseline; AV-4 / AV-5 / AV-27 named as P0; AV-3, AV-13, AV-16, AV-23, AV-26 scoped as P1 hot-fixes for v0.1.1; AV-26 / AV-27 / AV-28 / AV-29 added on the second-pass review).
