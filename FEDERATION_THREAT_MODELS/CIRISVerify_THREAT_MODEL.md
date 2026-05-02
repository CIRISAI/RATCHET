# CIRISVerify Threat Model

**Last updated:** 2026-05-01 (post-v1.8.3, federation-substrate work shipped)

## 1. Scope

### What CIRISVerify Protects

CIRISVerify is the hardware-rooted license verification module for the CIRIS ecosystem. As of v1.8.3 it also serves as the **build-attestation substrate primitive** for the PoB federation — every CIRIS peer (agent, lens, persist, registry, and CIRISVerify itself) signs and validates its build manifests through one shared code path (`verify_build_manifest`).

It protects against:

- **Identity fraud**: Unlicensed agents (CIRISCare) masquerading as licensed professional agents (CIRISMedical, CIRISLegal, CIRISFinancial)
- **License forgery**: Fabrication of valid-looking license status without registry authority
- **Capability escalation**: Agents claiming capabilities beyond their license tier
- **Silent degradation**: Agents suppressing or modifying mandatory disclosure about their license status
- **Verification tampering**: Modification of the verification binary to return false results
- **Cross-primitive identity confusion**: A compromised primitive's steward key being used to forge other primitives' manifests (v1.8 federation surface)
- **Ephemeral identity churn**: Silent ephemeral storage causing federation peers to lose continuity across restarts (v1.7 storage-descriptor surface)

### What CIRISVerify Does NOT Protect

- **Malicious behavior with valid license**: A properly licensed agent can still act maliciously within its authorized capabilities
- **Compromised steward issuing malicious licenses**: If the license issuing authority is compromised, valid-looking licenses can be issued for malicious agents
- **Network availability**: CIRISVerify degrades gracefully to community mode when offline but cannot guarantee connectivity
- **User device security**: If the user's device is fully compromised (root/admin access), hardware protections may be bypassed
- **Supply chain compromise of hardware**: If the HSM/TEE manufacturer is compromised, hardware attestation is unreliable

---

## 2. Adversary Model

### Adversary Capabilities

The adversary is assumed to have:

- **Full source code access** (AGPL-3.0 licensed, publicly available)
- **Ability to modify and rebuild** the verification binary
- **Network interception** capability (passive and active MITM)
- **Control of one DNS registrar** or one verification source
- **Ability to replay** previously captured valid responses
- **Access to emulators and debuggers** for runtime analysis
- **Unlimited computational resources** for classical cryptography attacks

### Adversary Limitations

The adversary is assumed to NOT have:

- **Simultaneous compromise of all DNS registrars** serving verification data
- **Compromise of the hardware security module** (TPM 2.0, Secure Enclave, StrongBox)
- **Ability to break both Ed25519 AND ML-DSA-65** simultaneously (hybrid crypto defense)
- **Physical access** to the deployment hardware
- **Compromise of the HTTPS certificate infrastructure** (CA compromise)
- **Ability to manipulate system clocks** by more than 5 minutes on the target device

---

## 3. Attack Vectors

Fourteen attack vectors organized by adversary goal. Each lists the attack, primary mitigation, secondary mitigation, and residual risk in-place. Goal-based grouping (mirrors CIRISPersist's structural pattern) makes the surface boundaries explicit:

- **§3.1 License Fraud** — adversary wants to claim a license tier they don't have (AV-1..AV-6)
- **§3.2 Federation Identity** — adversary wants stable misidentification or evasion of longitudinal scoring (AV-7, AV-8)
- **§3.3 Build Provenance** — adversary wants to strip or forge "this primitive's CI signed this binary" attestation (AV-9)
- **§3.4 Supply Chain** — adversary wants to ship malicious code through legitimate signing channels (AV-11, AV-12)
- **§3.5 Hardware Trust Anchor** — adversary wants to extract or counterfeit hardware-bound signing keys (AV-13)
- **§3.6 Operational / Reliability** — non-adversarial failure modes that still cost availability (AV-10)
- **§3.7 Multi-Instance Cohabitation** — multiple CIRISVerify instances on one host racing on shared keyring state (AV-14)

Original six vectors (AV-1..AV-6) from FSD-001; v1.7-v1.8 federation work added AV-7..AV-9; SOTA review (mid-2026) added AV-11..AV-13; v1.6.3 incident response promoted AV-10 to first-class status; v1.8.x cohabitation analysis added AV-14.

---

### 3.1 License Fraud — adversary wants to claim a license tier they don't have

#### AV-1: Code Forking / License Check Removal

**Attack**: Fork the open-source code, remove license checks, deploy as "licensed" agent.

**Primary mitigation**: Hardware-bound signing key. The verification response is signed by a key stored in the device's HSM. A forked binary cannot produce valid signatures because it lacks the hardware key. The key IS the identity — it cannot be copied from the source code.

**Secondary**: Binary integrity check at startup; transparency log records all verification events for after-the-fact audit.

**Residual**: A forked binary running on a SOFTWARE_ONLY device can produce locally-valid-looking output, but it's permanently capped at COMMUNITY tier (§5.1 invariant). Federation peers won't accept its output as professionally-licensed regardless of internal claims.

#### AV-2: Runtime Modification to Fake Licensed Status

**Attack**: Modify the running binary in memory to return fake "licensed" status.

**Primary mitigation**: Binary self-integrity verification at startup (hash comparison against signed manifest). Anti-debugging detection (ptrace, Frida, Xposed). Platform integrity checks (root/jailbreak detection). All checks return opaque pass/fail to prevent targeted bypasses (FSD-001 §"Integrity Check Opacity").

**Secondary**: Hybrid signature on attestation responses — even if the binary is modified to claim higher tier, downstream consumers verifying the response signature against the steward pubkey will reject.

**Residual**: A rooted device with a memory-patched binary can produce locally-correct UX (mandatory disclosure suppressed) but cannot forge valid signatures. The downstream verifier (e.g., another agent in the federation) catches the lie.

#### AV-3: Verification Endpoint Spoofing

**Attack**: Intercept and replace responses from verification endpoints.

**Primary mitigation**: Multi-source validation with consensus. DNS US + DNS EU + HTTPS API must agree (2-of-3 minimum). HTTPS is authoritative when reachable; DNS is advisory cross-check. Multiple HTTPS endpoints at different domains provide redundancy. Certificate pinning on HTTPS connections.

**Secondary**: Hybrid signature on responses ensures the *content* is steward-signed even if the *transport* is spoofed.

**Residual**: Simultaneous compromise of all three sources (US registrar + EU registrar + HTTPS CA) is the bypass. Mitigation is operational — sources are in different jurisdictions with different providers — but a state-level adversary with reach into all three is not defended against.

#### AV-4: Replay of Old Valid License After Revocation

**Attack**: Capture a valid license response before revocation, replay it afterward.

**Primary mitigation**: Anti-rollback monotonic revision enforcement. The system tracks the highest-seen revocation revision and rejects any revision that decreases. Challenge nonce (32+ bytes) prevents simple replay. License expiry timestamps provide time-based bounds.

**Secondary**: Transparency log records all revocation events; federation peers cross-check.

**Residual**: A captured response replayed within its expiry window before any revocation event lands at the consumer's seen-revision counter is acceptable per design — that's the intended grace window. Adversary cannot replay across a revocation event without rolling back the seen-revision counter (which the storage layer prevents).

#### AV-5: Man-in-the-Middle Attestation Responses

**Attack**: Intercept attestation responses and modify them to claim higher trust level.

**Primary mitigation**: Hybrid cryptographic signatures (Ed25519 + ML-DSA-65) over all response data. PQC signature is bound to classical signature (covers `challenge_nonce || classical_sig`), preventing signature stripping. Remote attestation proof export allows third parties to independently verify hardware binding.

**Secondary**: TLS termination + cert pinning at the transport layer.

**Residual**: A complete crypto break of both Ed25519 AND ML-DSA-65 simultaneously. ML-DSA-65 is NIST-standardized post-quantum; Ed25519 is classical. Hybrid means both must fall.

#### AV-6: Emulator/Debugger Interception

**Attack**: Run the agent in an emulator or attach a debugger to intercept and modify verification responses at runtime.

**Primary mitigation**: Platform-specific integrity checks detect emulators (Android: goldfish, qemu; iOS: simulator; Desktop: hypervisor/DMI checks). Debugger detection (ptrace on Linux/Android, sysctl on macOS, IsDebuggerPresent on Windows). Hook detection for Frida/Xposed frameworks. Timing anomaly detection for breakpoint-induced delays.

**Secondary**: Emulator/rooted devices are degraded to COMMUNITY tier rather than blocked outright (open-source principle: never fully stop execution; restrict capability instead).

**Residual**: Sophisticated emulator setups that defeat detection (e.g., hypervisor-level intercepts that hide from in-VM checks). Defense-in-depth via downstream signature verification — even if local checks pass, federation peers verifying the agent's signed output catch fakes.

---

### 3.2 Federation Identity — adversary wants stable misidentification or evasion of longitudinal scoring

#### AV-7: Ephemeral Identity Storage (lens-scrub-key class)

**Attack**: A federation peer's signer silently lands its identity seed in ephemeral storage (container writable layer, `/tmp`, `/var/cache`, user-session keyring). Identity churns every restart. The score function's longitudinal window (PoB §2.4 S-factor 30-day decay) cannot accumulate behind an unstable identity, so anti-Sybil weight stays at zero by construction.

**Primary mitigation**: `HardwareSigner::storage_descriptor()` (v1.7+) declares where the seed lives at runtime. Boot-time logging in `factory.rs::create_hardware_signer` surfaces the descriptor at INFO level for every signer construction. The trait method has no default impl — every signer variant is forced to declare its location.

**Secondary**: Consumers (CIRISAgent, CIRISPersist) gate `--strict-storage` mode that refuses to start if the descriptor matches an ephemeral path heuristic. `disk_path()` and `is_hardware_backed()` helpers let consumers ship their own typed checks.

**Residual**: An operator can override with `CIRIS_PERSIST_KEYRING_PATH_OK=1` (or equivalent) after manual audit. A consumer that ignores the descriptor entirely (doesn't gate boot on it) is back to the pre-v1.7 surface — but that's a consumer-side bug, not a verify-side gap. **Empirical trigger**: 2026-04 lens-scrub key incident (CIRISLens). Persist hit the same class without the descriptor.

#### AV-8: Cross-Primitive Identity Confusion

**Attack**: A compromised primitive's steward key (e.g., lens's) is used to sign a manifest claiming `primitive: agent`, hoping a verifier accepts it as if signed by agent's steward. "Confused deputy" pattern (OWASP Agentic Apps Top-10 in 2026).

**Primary mitigation**: `verify_build_manifest(bytes, expected_primitive, &trusted_pubkey)` validates `manifest.primitive == expected_primitive` BEFORE checking the signature. The primitive discriminant is part of the canonical bytes the hybrid signature covers, so a cross-primitive replay would have to forge a signature valid under a *different* primitive's steward key — equivalent to the original key-compromise problem, not a new attack surface.

**Secondary**: Per-primitive trusted-pubkey lookup means the wrong primitive's signature wouldn't match the expected primitive's pinned key. Lookup mechanism formalized via [`CIRISRegistry/docs/TRUST_CONTRACT.md`](https://github.com/CIRISAI/CIRISRegistry/blob/main/docs/TRUST_CONTRACT.md) §6 — admin-managed `RegisterTrustedPrimitiveKey` admin RPC writes to registry's `trusted_primitive_keys` table (cross-region replicated via Spock per CIRISRegistry#4).

**Long-term mitigation (persist v0.2.x federation directory)**: The trusted-pubkey lookup layer migrates from "registry-local table that the steward writes" to persist's federated `federation_keys` substrate (per [`CIRISPersist/docs/FEDERATION_DIRECTORY.md`](https://github.com/CIRISAI/CIRISPersist/blob/main/docs/FEDERATION_DIRECTORY.md)). Every `federation_keys` row carries scrub-signing four-tuple (recursive cryptographic provenance — every row signed by another row in the same table, terminating at the steward's self-signed bootstrap). Registry becomes a cache+policy layer over persist's substrate (per [`CIRISRegistry/docs/FEDERATION_CLIENT.md`](https://github.com/CIRISAI/CIRISRegistry/blob/main/docs/FEDERATION_CLIENT.md)). This resolves CIRISRegistry#5 §4 not via a new registry endpoint but by repositioning the trust-key-bootstrap layer entirely: persist stores; consumers (registry, verify, agent) compute their own trust verdicts.

**Residual**: Trusted-pubkey lookup for non-`Verify` primitives currently requires explicit registration via `RegisterTrustedPrimitiveKey` (registry-side, `CIRISVerify` registered today; persist/lens/agent registration coordinated when their CI migrates to `ciris-build-sign`). The persist v0.2.x federation directory is the architecturally-cleaner long-term resolution; until it ships, registry's `trusted_primitive_keys` table is the operational source of truth.

---

### 3.3 Build Provenance — adversary wants to strip or forge "this primitive's CI signed this binary" attestation

#### AV-9: Build Manifest Write-Without-Read (Phase A artifact)

**Attack**: A registered build manifest's original CI signature is silently lost on registry round-trip. Registry's *legacy* POST endpoint flattens `BuildManifest` into the `function_manifests` table and resigns it under the registry's own steward key on GET. Per-primitive steward attribution ("CIRISVerify-steward-2026 signed this") collapses into "registry vouches for this." A federation auditor asking "which primitive's CI actually signed this binary" gets the wrong answer.

**Primary mitigation (post-CIRISRegistry trust-contract docs PR `9f8e1d7`)**: The new POST endpoint preserves the original CI signature in storage. Authoritative reference: [`CIRISRegistry/docs/TRUST_CONTRACT.md`](https://github.com/CIRISAI/CIRISRegistry/blob/main/docs/TRUST_CONTRACT.md) §2.3 documents the two POST cases definitively. Verifiers distinguish which path served a manifest by inspecting `signature.key_id`:

- `signature.key_id == "verify-steward-2026"` (or any per-primitive steward key) → original CI sig preserved → Case (i) per TRUST_CONTRACT.md §2.3 → trust chain to the publishing primitive's steward key. Registry verifies inbound hybrid signature against `trusted_primitive_keys` (their AV-26 closure, v1.3 Phase A, commit `4adc224`) before storing. Operationally validated by registry's own self-publication since `cd95a9f` 2026-05-01 21:02:44Z.
- `signature.key_id` matching the registry's own steward key (`75c29fcc...`) → legacy POST path, registry-resigned → Case (ii) per TRUST_CONTRACT.md §2.3 → trust chain only back to the registry

**Secondary (Path C, always available)**: GitHub release artifact `signed-build-manifests.tar.gz` (added v1.8.3) preserves the original CI-signed `BuildManifest` end-to-end regardless of which POST endpoint was used. Sigstore-signed alongside the binary archives.

**Residual**: Registry adds `GET /v1/verify/build-manifest/{primitive}/{ver}/{target}` returning the original posted `BuildManifest` (CIRISRegistry#5 item 2, queued as Phase B). Until then, federation peers wanting per-primitive steward attribution need to fetch from Path C (release tarball), not Path A (registry GET). Workable but discoverability-asymmetric.

---

### 3.4 Supply Chain — adversary wants to ship malicious code through legitimate signing channels

#### AV-11: Sigstore OIDC Token Theft (federation-poisoning)

**Attack**: An attacker obtains a valid OIDC token for the CIRIS release identity (e.g., compromised GitHub Actions OIDC issuer config, stolen short-lived token from a misconfigured workflow) and signs a malicious binary that is correctly Rekor-logged. Downstream consumers' Sigstore-verify checks pass; the binary is accepted as authentic. Canonical post-XZ Sigstore-era attack pattern.

**Primary mitigation**: Sigstore identity binding via GitHub Actions OIDC (release artifacts are signed under `https://github.com/CIRISAI/CIRISVerify/.github/workflows/release.yml@refs/tags/vX.Y.Z`). Release notes include the expected signer identity for human review.

**Secondary**: Hybrid signature on the BuildManifest provides a second trust path independent of Sigstore. A token-thief who isn't also the steward-key-holder can produce a Sigstore-valid binary but not a steward-signed BuildManifest.

**Residual**: No continuous Rekor-monitor running against the CIRIS signing identity. A rogue signing event under our OIDC identity would not be detected automatically — only by humans noticing an unexpected release. **Action**: integrate `rekor-monitor` against `https://github.com/CIRISAI/CIRISVerify/...` identity, alarm to maintainers on unexpected entries. See §10 SOTA gap #2.

#### AV-12: Maintainer Compromise / XZ-Style Supply Chain

**Attack**: A maintainer with commit access to CIRISVerify (or any of its transitive Rust dependencies) introduces a backdoor that is legitimately signed and released through normal CI. Source review may not catch it (XZ-3094 hid payload in test fixtures and m4 macros, not git source). Downstream Sigstore + hybrid-sig checks pass.

**Primary mitigation**: Open-source under AGPL-3.0 means independent review is possible. Rust ecosystem's `cargo-audit` and `cargo-deny` checks land via `deny.toml` for advisory and license enforcement.

**Secondary**: None against this attack — a compromised maintainer signs both classical + PQC, so hybrid-sig is no defense; they have legitimate access to the steward key, so steward-signed BuildManifest is no defense.

**Residual (open)**: No two-person-rule on releases (single-maintainer signoff is the current path). No multi-maintainer signing key. No SBOM published with releases (CISA / EU CRA gap). No reproducible builds means source-vs-binary divergence cannot be independently verified. **Action items**: §10 SOTA gaps #3 (SBOM), #4 (reproducible builds), #5 (two-person-rule).

**Related (registry-side, finer-grained)**: [`CIRISRegistry/docs/THREAT_MODEL.md`](https://github.com/CIRISAI/CIRISRegistry/blob/main/docs/THREAT_MODEL.md) AV-34 ("Build-signing key compromise (CI-side, post-Phase-A surface)") catalogues the post-Phase-A surface where per-primitive build-signing keys held in GHA secrets become load-bearing trust anchors for the federation. Their dual-secret-co-requirement defense (publishing requires BOTH the build-signing key AND `REGISTRY_ADMIN_TOKEN`) and per-repo isolation (compromise of one repo's GHA can publish only that primitive's manifests) extend AV-12's mitigation surface across the federation. Their v1.4 hardening proposals (cosign verification on uploaded manifests, M-of-N signing for high-stakes primitives, SLSA attestation in extras) target the same supply-chain class that our SOTA gaps #3-#5 target.

---

### 3.5 Hardware Trust Anchor — adversary wants to extract or counterfeit hardware-bound signing keys

#### AV-13: TEE.fail / Bus-Interposition Attacks on Server-Class TEEs

**Attack**: Physical access to server hardware running a federation validator (e.g., a CIRISRegistry node, a federation peer running CIRISVerify in a confidential VM). DDR5 memory bus interposition (TEE.fail, Oct 2025, Georgia Tech/Purdue, sub-$1k attack) extracts attestation signing keys from fully-updated Intel SGX/TDX or AMD SEV-SNP machines because memory encryption is deterministic. EMFI on ARM TrustZone skips secure-boot checks reliably.

**Primary mitigation**: Threat model assumption §6.4 says adversary doesn't have physical access. Hardware-vulnerability detection (v1.2.0+) covers SoC-level CVEs (MediaTek CVE-2026-20435 boot ROM EMFI, Qualcomm CVE-2026-21385) and caps attestation to `SOFTWARE_ONLY` for affected devices. See §5.1 for the catalogue of detected hardware vulnerabilities.

**Secondary**: HSM-anchored signing keys (FIPS 140-3 Level 3+) for production registry/steward roles. TEE-anchored keys (SGX/TDX/SEV-SNP/CCA) should NOT be used for steward roles in confidential-VM deployments.

**Residual (open)**: Server-class TEE attacks are not modeled in our adversary capability list — physical access is excluded by assumption §6.4. Datacenter deployments of CIRISVerify or CIRISRegistry SHOULD assume hostile co-tenants and deploy steward keys in HSMs, not TEEs. Document this explicitly in registry deployment guidance. See §10 SOTA gap #7.

---

### 3.6 Operational / Reliability — non-adversarial failure modes that still cost availability

#### AV-10: Stale Hardware-Marker Lockup

**Attack surface (operational, not adversarial)**: A `HARDWARE_SECURED:fingerprint` marker file points at a TPM key that has been deleted (`tpm2_evictcontrol` on the wrong handle, OS reinstall, hardware swap). The agent process indefinitely fails its sign() calls because it believes hardware-bound signing is required, but the hardware key is gone. Recovery requires manual marker deletion.

**Primary mitigation**: v1.6.3 (`fix(v1.6.3): Clear stale hardware markers when key becomes inaccessible`) detects the case and surfaces `KeyringError::HardwareNotAvailable` instead of silent stuck-state. v1.6.4 refined the error variant for clearer operator-runbook signaling.

**Secondary**: Operator-runbook for marker file deletion documented in deployment guidance. On next boot, the agent generates a fresh hardware key and a fresh signed manifest.

**Residual**: Identity continuity lost — the agent that comes back after marker recovery has a different signing key than before, so federation longitudinal scoring (PoB §2.4 S-factor) starts over. This is the correct security tradeoff (cannot recover the "real" identity if hardware key is genuinely gone) but operators should be aware. No silent recovery path; identity loss is loud and visible.

---

### 3.7 Multi-Instance Cohabitation — multiple CIRISVerify instances on one host

#### AV-14: Cross-Instance Keyring Contention

**Attack surface (mostly operational, adversarial in worst case)**: Multiple CIRISVerify instances coexist on one host, racing on shared keyring/HSM state. Three deployment shapes produce this:

1. **In-process cohabitation** — two `.so` files loaded into one Python process (e.g., `CIRISPersist` bundles `ciris-verify-core` at compile time; the same process also imports the `ciris-verify` PyPI wheel which carries another copy of the FFI). Each `.so` has its own `static OnceLock<...>` globals — Rust-level state is NOT shared.
2. **Same-host different-process** — agent and persist running as separate daemons on one box, both using the same keyring alias.
3. **Misconfigured deployment** — multiple agent worker pods scaled out against shared persistent storage without an orchestrator.

In all three, the **OS-level backend is the actual serialization point**: TPM via `/dev/tpm0` (single-tab, multiplexed in-kernel or via `tpm2-abrmd`), Apple Keychain via `securityd` daemon, Android Keystore via Binder IPC, Linux Secret Service via D-Bus. Multiple Rust signer instances are clients of those daemons; concurrent reads serialize cleanly. Three race windows remain in our caller code:

- **Key-creation TOCTOU**: process A reads "no key for alias", calls `generate_key()`. Concurrently B does the same. Backend rejects the second create on Android Keystore (`KeyStoreException`), may silently overwrite on macOS Keychain depending on `kSecAttrAccessible`, races on TPM persistent-handle allocation.
- **Stale-marker race**: the `HARDWARE_SECURED:fingerprint` marker file (v1.4.0+ pattern) is process-local. v1.6.3-v1.6.4 fixes the single-process stale-recovery case but doesn't lock against concurrent recovery from another process.
- **Cache consistency**: A creates the key, B's `cached_signing_key: Option<...>` is still `None`. B fails its first `sign()` until something flushes the cache. No invalidation path.

The adversarial worst case: an attacker who controls one of the cohabiting processes attempts a malicious `delete_key()` while another process is mid-attestation. The legitimate process surfaces `HardwareNotAvailable` (AV-10's fix surfaces the symptom) but the operator may not realize the deletion was hostile rather than a hardware failure.

**Primary mitigation (today)**: When CIRISPersist is in the stack, persist's `Engine` holds the verify Engine — there's structurally one verify instance per process and the contention surface vanishes by construction. The dominant production pattern is "persist is the interface to verify" (see `docs/HOW_IT_WORKS.md` "Cohabitation — When Persist Is the Interface"); higher layers go through persist's API rather than instantiating their own verify Engine. AV-14's race windows describe a multi-instance pattern that doesn't occur in a persist-bearing stack.

In verify-only stacks (registry, sovereign-mode dev, simple CLI tooling), there's structurally one consumer — single process, single instance, no race possible.

**Secondary (today)**: OS-daemon-level serialization (TPM via `/dev/tpm0`, Apple `securityd`, Android Binder, Linux Secret Service) closes the **read** path universally — even in the unusual multi-consumer-without-persist case, concurrent reads serialize cleanly through the OS keyring backend. PoB §3.2 single-key-three-roles enforces same-alias semantics: two instances with the same alias are the *same identity*, not competing identities.

**Residual (open, only in verify-only multi-process stacks without an external bootstrap step)**:
- Cold-start key-creation race window in the uncommon multi-consumer-without-persist pattern (e.g., HA replica set running multiple verify-only processes against shared persistent storage). Fix paths: (a) add persist to the stack — surface vanishes; (b) operator-managed pre-bootstrap (`ExecStartPre=flock`); (c) verify v1.9's planned `flock` scope guards around mutating operations (~30 LoC, lower priority since the pattern is uncommon and avoidable).
- v2.0 out-of-process verify daemon was originally tracked as the architecturally clean answer; the persist-as-interface shape makes it unnecessary. Persist's process *is* the singleton when persist is in the stack; for verify-only stacks the "singleton" is just "the one process you have." v2.0 work pivots to formalizing this in docs and possibly adding a runtime warning if a process detects multiple verify Engines on the same alias (a hint that someone is bypassing persist).

#### Cohabitation contract (operator-facing)

Authoritative semantics for "is multiple-CIRISVerify-on-one-host OK":

| Pattern | Same alias | Different alias |
|---|---|---|
| Multiple read-only instances | ✅ Safe — OS serializes; same identity by PoB §3.2 | ⚠️ Conceptually wrong — two identities claiming one role |
| Multiple instances racing on key creation (cold-start) | ⚠️ Race window — backend may reject, may overwrite, may corrupt marker | ✅ No contention but breaks single-key-three-roles |
| Mid-runtime mutation (delete, rotate, recover stale marker) | ❌ Unsafe — caches diverge; surface `HardwareNotAvailable` storms | ⚠️ Same as above; affects only the mutating instance |

**Recommended deployment posture:**
- Cold-start serialization: don't race two processes through a fresh-deployment key-creation phase. Use a deploy-time advisory lock (e.g., systemd `ExecStartPre` with a `flock` against a known path) to ensure the first process completes the create-or-load phase before the second starts.
- Same-alias for same identity: if your host runs both an agent AND a persist daemon, both should use the same alias to consume the same identity (PoB §3.2). Different aliases = different identities = federation confusion.
- Mutation operations only from one process: pick an "owner" process for `generate_key`, `delete_key`, marker recovery. Other instances are read-only consumers.

---

## 4. Mitigation Matrix

| AV | Attack | Primary Mitigation | Secondary | Status | Fix Tracker |
|---|---|---|---|---|---|
| AV-1 | Code fork / license-check removal | Hardware-bound signing key | Binary integrity check | ✓ Mitigated | — |
| AV-2 | Runtime modification | Binary self-integrity + anti-debug | Platform integrity checks | ✓ Mitigated | — |
| AV-3 | Endpoint spoofing | HTTPS authoritative + 2-of-3 consensus | Certificate pinning | ✓ Mitigated | Fix 4 |
| AV-4 | License replay after revocation | Anti-rollback monotonic revision | Challenge nonce + expiry | ✓ Mitigated | Fix 2 |
| AV-5 | MITM attestation | Hybrid Ed25519 + ML-DSA-65 (bound) | Remote attestation proof export | ✓ Mitigated | Fix 3 |
| AV-6 | Emulator/debugger interception | Platform integrity checks | Timing anomaly detection | ✓ Mitigated | — |
| AV-7 | Ephemeral identity storage (lens-scrub class) | `HardwareSigner::storage_descriptor()` v1.7+ + boot-time logging | `--strict-storage` consumer flag, ephemeral-path heuristics | ✓ **Mitigated v1.7.0** | — |
| AV-8 | Cross-primitive identity confusion | Primitive discriminant inside hybrid-sig canonical bytes; per-primitive trusted-pubkey lookup | Registry-mediated trust via `RegisterTrustedPrimitiveKey` | ✓ Mitigated v1.8.0 (verify-side); ⚠ trusted-key bootstrap pending registry support | CIRISRegistry#5 |
| AV-9 | BuildManifest write-without-read (Phase A) | GitHub release artifact `signed-build-manifests.tar.gz` preserves CI signature end-to-end | Symmetric `GET /v1/verify/build-manifest/...` (planned) | 🟡 Mitigated via Path C (release); ⚠ Path B (registry GET) pending | CIRISRegistry#5 item 2 |
| AV-10 | Stale hardware-marker lockup | v1.6.3 detects + clears stale markers; `HardwareNotAvailable` error surfaces recovery path | Operator-runbook for marker file deletion | ✓ Mitigated v1.6.3-v1.6.4 | — |
| AV-11 | Sigstore OIDC token theft | OIDC identity binding via `${workflow}@${ref}`; release notes include expected signer | Hybrid BuildManifest signature is independent of Sigstore — second trust path | 🟡 Partial — no continuous Rekor-monitor on CIRIS identity | §10 SOTA gap #2 |
| AV-12 | Maintainer compromise / XZ-style | AGPL-3.0 source review; `cargo-audit` + `cargo-deny`; hybrid sig | None against this attack — compromised maintainer signs both classical + PQC | ⚠ **Open** — no two-person-rule, no SBOM, no reproducible builds | §10 SOTA gap #3-#5 |
| AV-13 | TEE.fail / DDR5 bus interposition | Threat assumption §6.4 (no physical access); SoC vuln detection v1.2.0+ caps to SOFTWARE_ONLY | Defense-in-depth: HSM-anchored steward keys (FIPS 140-3 L3+) for production roles | 🟡 Mobile/SoC covered; server-class TEE attacks NOT modeled | §10 SOTA gap #7 |
| AV-14 | Cross-instance keyring contention (multi-instance cohabitation) | Persist-as-interface: one verify instance per process by construction in persist-bearing stacks; OS-daemon serialization on read path universally | Documentation contract (HOW_IT_WORKS.md "When Persist Is the Interface"); same-alias = same identity by PoB §3.2 | ✓ Surface vanishes in dominant production pattern (persist-bearing); 🟡 multi-process verify-only stacks need operator-bootstrap or v1.9 flock guards (uncommon pattern) | v1.9 (verify-side `flock` for persist-less stacks) |
| Audit | Audit trail tampering | Transparency log with Merkle tree | Append-only persistent storage | ✓ Mitigated | Fix 1 |

**Status legend:** ✓ Mitigated • 🟡 Partial mitigation, residual tracked • ⚠ Open / planned

---

## 5. Security Levels by Hardware Type

| Hardware Type | Security Level | Max License Tier | Attestation Quality | Key Protection |
|---------------|---------------|------------------|--------------------| --------------|
| Android StrongBox | 5 (Dedicated SE) | Professional | Strong (Google hardware attestation) | Hardware-isolated |
| iOS Secure Enclave | 5 (Dedicated SE) | Professional | Strong (Apple App Attest) | Hardware-isolated |
| TPM 2.0 (Discrete) | 5 (Dedicated chip) | Professional | Strong (EK certificate) | Hardware-isolated |
| TPM 2.0 (Firmware) | 4 (fTPM) | Professional | Moderate (no discrete hardware) | Firmware-isolated |
| Intel SGX | 4 (Enclave) | Professional | Moderate (remote attestation) | Enclave-isolated |
| Android Keystore (TEE) | 3 (TEE-backed) | Professional | Moderate (key attestation chain) | TEE-isolated |
| Software-Only | 1 (No hardware) | Community ONLY | None (key extractable) | None |

**Critical invariant**: `SOFTWARE_ONLY` devices are permanently capped at `UNLICENSED_COMMUNITY` tier. No license upgrade path exists without hardware security.

---

## 5.1 Known Hardware Vulnerabilities (v1.2.0+)

CIRISVerify detects known hardware vulnerabilities that compromise TEE security. Devices with these vulnerabilities are treated equivalently to emulators—attestation is capped at `SOFTWARE_ONLY` level.

### CVE-2026-20435: MediaTek Boot ROM EMFI Vulnerability

| Property | Value |
|----------|-------|
| Severity | CRITICAL |
| Affected | MediaTek Dimensity 7300, 7200, 1200, 8100, 9000, 9200 (mt6878, mt6886, mt6893, mt6895, mt6983, mt6985) |
| TEE | Trustonic |
| Impact | Physical access can extract Android Keystore keys in <45 seconds |
| Patchable | NO (Boot ROM is burned into silicon) |
| Disclosed | March 12, 2026 by Ledger Donjon |

**Technical Details**: The MediaTek boot ROM contains a flaw in how it handles electromagnetic fault injection (EMFI) during the secure boot process. An attacker with physical access can glitch the chip during TEE initialization to dump the Trustonic TEE's secure world memory, including Android Keystore keys.

**Detection**: CIRISVerify identifies vulnerable chipsets via `Build.HARDWARE` and `Build.BOARD` system properties on Android.

**Mitigation**: Devices with affected chipsets have `hardware_trust_degraded = true` and `HardwareLimitation::VulnerableSoC` in their limitations. Professional license tiers are not available on these devices.

### CVE-2026-21385: Qualcomm Security Component Vulnerability

| Property | Value |
|----------|-------|
| Severity | HIGH |
| Status | Under limited, targeted exploitation (CISA KEV catalog) |
| Patchable | YES (March 2026 Android Security Patch) |

**Mitigation**: Devices running Qualcomm chipsets should apply the March 2026 security patch. CIRISVerify tracks `Build.VERSION.SECURITY_PATCH` and can warn about outdated patch levels.

### Other Hardware Limitations

| Limitation | Impact | Caps Attestation? |
|------------|--------|-------------------|
| Emulator detected | No real hardware security | YES |
| Rooted/jailbroken device | HSM protections bypassed | YES |
| Unlocked bootloader | Secure boot chain compromised | YES |
| Outdated security patch | Known vulnerabilities unpatched | NO (warning only) |
| Weak TEE implementation | TEE-specific issues | YES |

---

## 6. Security Assumptions

From FSD-001, the system depends on these assumptions:

1. **HSM integrity**: Hardware security modules (TPM 2.0, Secure Enclave, Android Keystore/StrongBox) are not compromised at the hardware level
2. **DNS diversity**: Multiple DNS registrars are not simultaneously compromised (US and EU registrars are different providers)
3. **TLS infrastructure**: HTTPS certificate infrastructure (Certificate Authorities) is not compromised
4. **Physical security**: Adversary does not have physical access to the deployment hardware (physical access can bypass most HSM protections)
5. **Clock accuracy**: System clocks are reasonably synchronized (within 5 minutes of actual UTC time)
6. **Network path**: Network path to at least one verification source is not fully compromised (offline mode provides 72-hour grace period)

---

## 7. Fail-Secure Degradation

All failures degrade to MORE restrictive modes, never less:

| Failure Condition | Resulting Mode | Rationale |
|-------------------|---------------|-----------|
| Binary integrity check failed | LOCKDOWN | Possible active tampering |
| Sources actively disagree | RESTRICTED | Possible attack on verification |
| Rollback detected | RESTRICTED | Possible replay attack |
| No sources reachable | COMMUNITY (with 72h grace) | Cannot verify current status |
| License expired | COMMUNITY | License no longer valid |
| License revoked | COMMUNITY | License explicitly revoked |
| Signature verification failed | COMMUNITY | Cannot verify authenticity |
| Hardware attestation failed | COMMUNITY | Cannot verify hardware binding |

---

## 8. Cross-Cutting Residual Risks

Per-AV residuals are documented in-place under each attack vector in §3. This section catalogues residuals that aren't tied to a single AV but cross multiple surfaces, plus the index of where in-place residuals live.

### 8.1 In-place residual index

| AV | Surface | Residual summary |
|---|---|---|
| AV-1 | License Fraud | SOFTWARE_ONLY tier cap is the floor — local fakes can't claim Professional through downstream verifiers |
| AV-2 | License Fraud | Memory-patched binary on rooted device fakes UX locally but cannot forge downstream signatures |
| AV-3 | License Fraud | Simultaneous compromise of all 3 sources (US registrar + EU registrar + HTTPS CA) is the bypass |
| AV-4 | License Fraud | Replay within expiry window is design-allowed; cross-revocation replay blocked by anti-rollback |
| AV-5 | License Fraud | Requires breaking Ed25519 AND ML-DSA-65 simultaneously |
| AV-6 | License Fraud | Hypervisor-level intercepts that hide from in-VM checks; defense-in-depth via downstream sig verify |
| AV-7 | Federation Identity | Operator override (`CIRIS_PERSIST_KEYRING_PATH_OK=1`); consumer ignoring the descriptor |
| AV-8 | Federation Identity | Trusted-pubkey lookup for non-`Verify` primitives needs `RegisterTrustedPrimitiveKey` propagation; brief inconsistency window |
| AV-9 | Build Provenance | Per-primitive steward attribution available through Path C (release tarball) until Phase B GET ships |
| AV-10 | Operational | Identity continuity loss after hardware-key recovery; correct security tradeoff |
| AV-11 | Supply Chain | No continuous Rekor-monitor on CIRIS identity (action: §10 gap #2) |
| AV-12 | Supply Chain | No two-person-rule, SBOM, reproducible builds (actions: §10 gaps #3, #4, #5) |
| AV-13 | Hardware | Server-class TEE attacks (TEE.fail) not modeled; HSM-anchored steward keys for prod |
| AV-14 | Operational | Surface vanishes by construction in persist-bearing stacks (persist holds the one verify Engine); uncommon multi-process verify-only pattern needs operator pre-bootstrap or v1.9 flock guards |

### 8.2 Cross-cutting residuals

These don't belong to a single AV because the failure mode crosses surfaces:

1. **Hardware supply chain compromise**: If HSM manufacturers are compromised, attestation data can be forged across all hardware-anchored AVs (1, 5, 13). Mitigation: use hardware from multiple manufacturers; monitor for HSM vulnerability disclosures; defense-in-depth via per-primitive steward separation.

2. **Zero-day in HSM firmware**: Undisclosed vulnerabilities in TPM/SE firmware could allow key extraction. Cross-cuts AV-1, AV-2, AV-13. Mitigation: hybrid crypto means both classical AND PQC must be broken; firmware update monitoring. CIRISVerify v1.2.0+ actively detects known vulnerabilities (e.g., CVE-2026-20435 on MediaTek) and degrades affected devices to `SOFTWARE_ONLY` level.

3. **Clock manipulation**: If system clock is skewed by more than 5 minutes, expiry-based protections (AV-4) and timestamp-based replay defenses weaken. Mitigation: multi-source timestamp cross-checking; NTP hardening guidance for deployments.

4. **All verification sources compromised simultaneously** (AV-3 worst case): if attacker controls all DNS registrars AND HTTPS endpoints, false consensus can be achieved. Mitigation: sources are in different jurisdictions (US/EU) with different registrars; transparency log provides after-the-fact audit capability.

5. **Quantum computer capable of breaking both Ed25519 and ML-DSA-65** (AV-5 + AV-9 + AV-12 worst case): current quantum computers cannot break either. Hybrid approach means BOTH must fall. ML-DSA-65 provides NIST-standardized post-quantum resistance. Mitigation: algorithm agility allows future upgrades — see §10 gap #6 for the formal migration plan.

6. **Insider threat at license issuing authority**: a compromised steward could issue valid licenses to malicious agents. Cross-cuts AV-1, AV-12. Mitigation: transparency log records all issuance events for audit; multi-party authorization for license issuance (registry-side control). Federation-side defense: per-primitive steward separation means a compromised license-issuance steward cannot forge BuildManifest signatures for other primitives.

---

## 9. Federation-Level Security (post-v1.8)

The v1.8 build-manifest substrate (`BuildPrimitive`, `verify_build_manifest`, the `ciris-build-sign` / `ciris-build-verify` CLIs) introduces federation-scale threat surfaces that single-primitive thinking doesn't cover.

### 9.1 Trust topology

| Trust anchor | Held by | Used for |
|---|---|---|
| Registry's steward pubkey | All consumers (pinned at compile time / fetched from `/v1/steward-key`) | Verifying registry-served `FunctionManifest` (Path A reads); future trusted-primitive-key lookups |
| Per-primitive steward pubkey (`verify-steward-2026`, `persist-steward-2026`, etc.) | Each primitive's own publishing CI; registered with registry via `RegisterTrustedPrimitiveKey` | Signing the primitive's own `BuildManifest`; verifying that primitive's manifest in Path B/C |
| Embedded `Verify` steward pubkey | CIRISVerify binary (compile-time constant) | Self-check: validating CIRISVerify's own runtime against its signed manifest (recursive golden rule) |

### 9.2 Trust rotation requirements

Steward keys rotate. The threat model requires:

- **Per-primitive rotation:** old key revocation visible to all consumers within bounded propagation time (target: 24h via cross-region replication per CIRISRegistry#4). Old manifests signed under retired keys remain verifiable via the transparency log; new manifests reject old signatures.
- **Registry steward rotation:** registry signs its own steward-key transitions under the OUTGOING key, so consumers with the old pinned key can verify the new key is legitimate. No rotation should require recompiling consumers.
- **Algorithm rotation (PQC migration):** when ML-DSA-65 successor is needed, hybrid-sig schema admits a third algorithm slot. Old `(Ed25519, ML-DSA-65)` signatures continue to verify; new signatures are `(Ed25519, ML-DSA-65, NEW)` until ML-DSA-65 is fully retired.

The third item is currently **unspecified** — see §10.

### 9.3 Open federation issues

- **AV-9 Phase A artifact** — registry-mediated read path doesn't preserve per-primitive steward attribution. Tracked at CIRISRegistry#5 item 2. Workaround: GitHub release tarball.
- **AV-8 trust-key bootstrap** — non-`Verify` primitives need their pubkeys registered with registry before their manifest POSTs land. Tracked at CIRISRegistry#5 item 3.
- **Confused deputy across primitives** — verify-side defenses are robust (primitive in canonical bytes, separate trusted-pubkey per primitive); registry-side enforcement of "manifest signature must match primitive" is part of the same registration story.

---

## 10. SOTA Gaps and Roadmap

A SOTA review (mid-2026) against current best-practice in software-supply-chain attestation surfaced gaps below. None are exploitable today against CIRISVerify's documented adversary model, but each closes residual risk that the wider industry is moving on.

### Gap 1: SLSA level not declared

**SOTA**: SLSA v1.2 with L3 (hardened build platform, unforgeable provenance signed by the platform) as the credible bar for federation peers; L4 (reproducible builds) increasingly the gating definition.

**CIRISVerify position**: Effective ~SLSA L2 today (signed provenance via Sigstore + GitHub Actions OIDC, build platform is GitHub-hosted runner). L3 requires us to claim the build platform's hardening explicitly and publish provenance attestations (in-toto). L4 requires reproducible builds, which Cargo's release profile (`lto=true`, `codegen-units=1`, `panic="abort"`, `strip=true`) approximates but doesn't guarantee bit-for-bit.

**Action**: declare SLSA L3 target in `docs/IMPLEMENTATION_ROADMAP.md`; publish in-toto attestations alongside release artifacts; reproducible-builds work as separate v2.0 milestone.

### Gap 2: No continuous Rekor identity monitoring

**SOTA**: `rekor-monitor` watches for unexpected signing events under a configured OIDC identity. Canonical post-XZ defense for Sigstore-era supply chain.

**CIRISVerify position**: We sign with Sigstore, log to Rekor, but don't monitor the log for rogue entries under our identity. A stolen OIDC token would produce a correctly-Rekor-logged malicious binary that consumers' verify checks pass.

**Action**: integrate `rekor-monitor` against `https://github.com/CIRISAI/CIRISVerify/...` identity, alarm to maintainers on unexpected entries. Tracked as separate ops item.

### Gap 3: No SBOM in releases

**SOTA**: SPDX 3.0 + CycloneDX 1.6 dual format; CISA minimum elements (component hash, license fields); EU CRA mandate by Dec 2027.

**CIRISVerify position**: No SBOM published. Downstream operators have no machine-readable enumeration of our transitive Rust dependencies — exactly the attack surface a `cargo`-based XZ-equivalent would exploit.

**Action**: `cargo-cyclonedx` integration in release workflow; publish both SPDX + CycloneDX as attached attestations; document in release notes how to verify SBOM signature against steward key.

### Gap 4: No reproducible builds

**SOTA**: Reproducible-builds.org standards; bit-for-bit reproducibility as the only systemic defense against source-vs-binary divergence (XZ-class attacks).

**CIRISVerify position**: Release builds from GitHub-hosted runners with deterministic-ish flags but not verified bit-for-bit. A maintainer with CI access could release a binary that doesn't match the source.

**Action**: not v1.x scope. Targeting reproducibility for v2.0 is the right framing — requires Cargo build-info determinism work + cross-rebuilder verification (e.g., a separate trusted-rebuilder running on independent infrastructure).

### Gap 5: No two-person-rule on releases

**SOTA**: Multi-maintainer signoff for critical projects (OpenSSF Critical Project framework). XZ-3094 demonstrated single-maintainer trust is insufficient.

**CIRISVerify position**: Single-maintainer release path. A compromised maintainer or stolen GitHub token can publish a release that passes all automated checks.

**Action**: GitHub branch protection requiring 2+ approvers on release-tag-creating PRs; key-ceremony procedure for steward-key rotation requiring two operators. Documented in release runbook.

### Gap 6: PQC algorithm-agility plan unspecified

**SOTA**: NIST IR 8547 deprecation timeline (quantum-vulnerable algos by 2035); algorithm-agility patterns mandate config-not-code rotation.

**CIRISVerify position**: Hybrid Ed25519 + ML-DSA-65 implemented, but no specified upgrade path if ML-DSA-65 is broken. ML-DSA is lattice-based, ~2 years standardized — non-zero risk of cryptanalytic discovery.

**Action**: add `signature.classical_algorithm` and `signature.pqc_algorithm` as allowlist-validated fields (already serialized in `ManifestSignature`); document rotation procedure in `docs/PQC_MIGRATION.md`; consider SLH-DSA (FIPS 205, hash-based, conservative-trust) as the designated fallback.

### Gap 7: Server-class TEE attacks not modeled

**SOTA**: TEE.fail (Oct 2025, Georgia Tech / Purdue, sub-$1k) breaks Intel SGX/TDX and AMD SEV-SNP via DDR5 bus interposition. PSA Level-3 RoT IP with SCA + fault-injection resistance is the new bar.

**CIRISVerify position**: Mobile/SoC vulnerability detection (v1.2.0+) caps attestation to SOFTWARE_ONLY for known-bad chips. Server-class TEEs (used by registry-side, by federation peers running confidential VMs) are NOT modeled. A TEE.fail-equivalent against a registry validator could extract steward signing keys.

**Action**: registry-side roadmap item (CIRISRegistry threat model); CIRISVerify's recommendation is HSM-anchored signing keys for production registry/steward roles, not TEE-anchored. Document this explicitly in registry deployment guidance.

---

## 11. Update Cadence

This threat model is **updated on every minor version bump** (v1.X → v1.X+1). New attack vectors land here when:

- A new feature exposes a new surface (e.g., v1.7's `StorageDescriptor` introduced AV-7).
- A SOTA review identifies an industry-recognized gap (the v1.8 cycle added AV-11–AV-13 from supply-chain threat literature).
- A real-world incident in our deployment or a sibling primitive's deployment surfaces a class we hadn't catalogued (lens-scrub-key incident motivated AV-7's elevation from "future concern" to "core mitigation").

Patch releases (v1.X.Y) update only the **Status** column of the mitigation matrix when residuals close. The §10 SOTA roadmap is reviewed at minor-version cadence; a yearly full-text SOTA pass is part of the v2.0 plan.

**Companion documents:**
- [`FSD/FSD-001_CIRISVERIFY_PROTOCOL.md`](../FSD/FSD-001_CIRISVERIFY_PROTOCOL.md) — protocol spec
- [`docs/POB_SUBSTRATE_PRIMITIVES.md`](POB_SUBSTRATE_PRIMITIVES.md) — federation substrate design (closed)
- [`docs/BUILD_MANIFEST.md`](BUILD_MANIFEST.md) — build-manifest schema and CLI usage
- [`docs/BENCHMARKS.md`](BENCHMARKS.md) — performance baselines feeding the regression-gate threat assumption
