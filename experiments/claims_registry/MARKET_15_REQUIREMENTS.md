# Market 15 from first causes — what "certificates compose" requires, and what lens-core must carry

Companion to `MARKET_15_DESIGN.md` (the registry mechanics) and
`GUNNAR_MARKETS_2026-10-21.md`. This file is the requirement, stated before any
implementation, so the namespace question can be put to the Constitution cleanly.
Code facts: CIRISServer `cfd2867b` (`src/scorer.rs`, `src/compose_policy.rs`),
CIRISConstitution `origin/rc9` (CC 3.1.8, 3.4.5, 2.1, 2.4, 3.3.6).

## 1. First cause

A safety case about a deployed agent is never one statement. It is several, issued by
different parties, at different instants, each seeing a different slice. Each is a
signed claim of the form

> issuer **I**, at instant **t**, about referent **R**, having seen **E**, under scope
> **S**, asserts proposition **P**.

Two such claims are jointly about anything only if their referents coincide on every
axis **P** depends on. A control-boundary claim about build X and a correction claim
about build X+1 are each fine alone and say nothing together. That is the whole of
Market 15: not "are the certificates true," but "are they about the same thing."
Gunnar's phrasing: "Independently issued certificates may each look fine alone while
disagreeing on system version, monitor, correction authority, or task domain."

Lens already carries this rule in miniature. CC 3.1.8.1 makes the capacity composite
`𝒞_CIRIS = min(C, I_int, R, I_inc, S)` over five **independently emitted** factor rows,
and `CLM-capacity-composite` says a missing factor is **undefined, not zero**. A
minimum over five rows is only a statement about an agent if the five rows are about
the same agent, at compatible instants, under the same interface. Market 15 is the
generalisation of the precondition Lens's composite already assumes: **establish
referent coincidence before composing; where it cannot be established, the composite
is undefined, never silently zero.** CC 3.1.8: "LensCore observes; it does not
adjudicate." A referent check is observation, not adjudication: it says whether the
rows are commensurable, not whether the agent is safe.

## 2. The referent axes

The axes a certificate about a deployed agent can disagree on, and the CEG field that
carries each today:

| Axis | Meaning | Carried by (rc9) |
|---|---|---|
| **R1 identity** | which agent | `attested_key_id`; the `identity_occurrence` roster (CC 3.3.6) |
| **R2 version** | which build / identity state, and its predecessor | `provenance:build_manifest:{target}` rows; the Verify bundle verdict's re-rooted manifest; the identity node's `version`/`updated_by` tags; `supersedes.references_attestation_id` |
| **R3 extent** | which occurrences of the agent | `occurrence_id`, `occurrence_count`, `occurrence_role` |
| **R4 interface** | what the issuer could see (the access assumption) | `consent:state:granted` + `scope` (`analyze`), the install's `trace_level`, the trace window; `consent_basis` on capacity rows from 0.5.225 |
| **R5 authority** | who could correct the agent at **t** | `delegates_to` / `trust:confers:v1` rows and their `withdraws`; the persist precedence layer |
| **R6 setting** | task domain, template, cohort | `cohort_scope`, `community_id`, `context`, `scope`; `template_id`/`model` in the battery manifests |
| **R7 time** | when asserted, until when, and ordering against version changes | `asserted_at`, `valid_until`, `expires_at` |

The registry's six fault families are violations on these axes:

| Fault family | Axis |
|---|---|
| wrong-version | R2 |
| wrong-predecessor | R2 (lineage link) |
| access-mismatch | R4, R3 |
| correction-authority | R5 at R7 |
| applicability-change | R6, R7 |
| out-of-scope | R6; also `witness_relation` / self-emission (CC 3.4.5) |

## 3. Logical requirements (what Gunnar is after)

- **L1 Every certificate names its referent on every axis its proposition depends on.**
  The registry's human check `named-system-version` is this requirement in words. A
  certificate silent on a load-bearing axis is not composable: the procedure must flag
  it, not assume.
- **L2 A frozen, deterministic procedure.** `Compose(bundle, deployment_ref) →
  coincide | diverge(axis, fault family) | unresolvable(reason)` (the Constitution's
  wording constraint, §6 N3: Lens may say whether rows share a referent, never
  "accept/reject"; the registry's `accepted` / `rejected_or_flagged` columns are the
  filer's mapping of those three); same inputs, same output; published
  and hashed before the bundles it is scored on are built (`methodFreezeAt`).
- **L3 Fail-closed on resolution, not on judgment.** Unresolvable key, roster, lineage
  or scope → flag. No axis is ever defaulted. This is the `RefusalReason` discipline of
  `compose_policy.rs` applied to referents instead of trust.
- **L4 The procedure composes; it does not re-score, invent, or self-certify.** It
  cannot manufacture a missing family, and no certificate in the bundle may be emitted
  by its own subject (CC 3.4.5 no-self-emit).
- **L5 The verdict is itself a certificate.** It names the bundle (attestation ids and
  content hashes), the deployment reference on all seven axes, the procedure version,
  and its instant. "A composition verdict on a named certificate bundle for one system
  and setup" is the contract's required output.
- **L6 Precision as well as recall.** The ≥80% clean-accept bar means axes must be
  compared in canonical, mechanical form (CC 2.6 canonicalisation), not by heuristic
  matching; otherwise a fail-closed procedure flags everything and misses the bar.
- **L7 Reconstructible by third parties** from the released rows and the frozen
  procedure alone.
- **L8 Snapshot semantics.** The verdict is about the bundle at an instant. Later
  withdrawal of an ancestor, recertification after a scope change, and successor
  questions are explicitly out of this market (book, Appendix H M15).

## 4. What lens-core must therefore carry (requirements on CIRISServer)

What the capacity scorer binds today (`scorer.rs` ~1388–1432): `dimension`,
`attesting_key_id` (node), `attested_key_id` (agent), the sample statistics,
`asserted_at` / `valid_until`, `cohort_scope = federation`. That is **R1 and R7 only**.

- **S1 Referent pins on every Lens emission.** R2 (the build manifest hash or
  `provenance:build_manifest` reference the scored traces came from), R3 (the
  occurrence set scored), R4 (`consent_basis`, `trace_level`, the trace window) on the
  `capacity:*` rows and on any detector row (3.1.8.2–3.1.8.5). Without S1, Lens's own
  rows cannot be composed with anyone else's under L1, and `𝒞_CIRIS` is composed over
  rows whose version coincidence is unchecked.
- **S2 A bundle-verdict consumer policy** beside `compose_policy.rs`, with the six fault
  families as its refusal vocabulary and `flag` as the unresolvable case; inputs are
  rows it already reads plus the Verify bundle verdict (R2), the occurrence roster
  (R1, R3), the delegation precedence layer at an instant (R5), and the consent fold
  (R4).
- **S3 The verdict as an emitted object** (the namespace question below).
- **S4 The precondition applied to the existing composite.** `min(C, I_int, R, I_inc, S)`
  is computed only over factor rows that pass the referent check on R1–R4 and R7; a
  factor whose referent does not coincide is **undefined** for the composite, per
  `CLM-capacity-composite`.

## 5. Questions for the Constitution (namespace and requirement)

- **N1 Access certificate.** Is there, or should there be, one row that states the
  monitoring interface assumed for a subject: consent scope in force, `trace_level`,
  occurrence set, window? Is `consent_basis` on `capacity:*` the intended carrier, or is
  this a leaf (under `capacity:`? `consent:`? elsewhere) that any witness can emit
  about what it saw?
- **N2 Correction certificate.** When the Market 4 battery produces per-case uptake,
  persistence and sham rates for a named agent version, is that a `benchmark:*` family
  row (like `benchmark:he300:*`) with the WA roster in `evidence_refs`, or does the
  correction-authority axis (R5) need its own carrier?
- **N3 The verdict.** Is a composition verdict a `scores` row (a Lens leaf under
  3.1.8?), or an artifact outside the grammar that references the four rows by
  `evidence_refs`? Does "LensCore observes; it does not adjudicate" permit Lens to emit
  "these rows are / are not about the same referent"? (This file's reading: yes, it is
  a commensurability observation, not a safety verdict.)
- **N4 Version pin.** Which field is normative for R2 on a `scores` row about an agent?
  Is there, or should there be, a rule that a row about an agent occurrence references
  the build it observed (`provenance:build_manifest:{target}` in `evidence_refs`, or an
  envelope field)?
- **N5 Authority at an instant.** Is "the `delegates_to` roster in force at **t**,
  net of `withdraws`" a defined query of the precedence layer, so R5 can be checked
  mechanically?
- **N6 Second architecture.** Can the server node itself be a subject of the same four
  families (Verify bundle, trust-root delegations, moderation/accord rows, its own
  scoring) so that a bundle set spans two architectures without a toy?

## 6. Answers from the Constitution session (2026-10-21, against rc9 head `aa5ae9c`)

The first cause is CC's own: `CLM-capacity-composite` already composes independently
emitted rows with a missing factor undefined, so "unresolvable is a flag, never silent"
is the existing rule generalised.

| Q | Status | Answer |
|---|---|---|
| N1 access certificate | **not answered; found defect** | No row states the monitoring interface assumed for a subject. `consent_basis` has 0 hits in the CC text; the server emits it on every `capacity:*` row since 0.5.225 (`scorer.rs:1614`, `:1704` at `ffff40e0`), so it is an **untyped member**: CC 2.1's member table has no row for it, and the `signed:envelope-extra` class is "typed by CC 2.1, carried untyped by persist, validated at the family's gate", so persist stores it verbatim in `RowMirror.extra` and no gate validates it. (Correction 2026-10-21: the issue as filed cited CC 3.1.7 R2, which governs families, not members; corrected in a comment on #183. Nine other capacity members are in the same state: `n_eff_pr`, `n_eff_h`, `sample_size`, `feature_dim`, `sample_size_gate`, `target_n_eff`, `min_feature_dim`, `n_eff_floor`, `window`.) Shape: what a witness saw is an observation and belongs in its own leaf any witness can emit, not a member of `capacity:*`; but `observation:` is closed (`observation:reachability` is its only leaf, 3.1.9.4), so opening it is a steward amendment. Interim honest carrier: the capacity row's envelope `scope` plus `evidence_refs` to the `consent:state:granted` row in force. |
| N2 correction certificate | partly | `benchmark:*` exists but only `benchmark:he300:{category}:{version}` is enumerated (CIRISBench stewards it). A Market 4 battery fits as a new benchmark leaf registered at mint by its steward, per-case rates in the payload. The WA roster in `evidence_refs` is evidence, not authority: R5 is answered by N5. |
| N3 the verdict | **answered**, one wording constraint | A commensurability observation is a `scores` row under 3.1.8 by Lens. "Observes; does not adjudicate" permits "these rows are / are not about the same referent on axis Rk / cannot be resolved" and forbids "accept / reject". Outcomes: `coincide` \| `diverge(axis, fault family)` \| `unresolvable`; bundle named in `evidence_refs`; leaf registered at mint. |
| N4 version pin | **not answered; found gap** | No field is normative for R2 on a `scores` row about an agent. `occurrence_*` are self-asserted; `provenance:build_manifest` attests builds, not rows about running agents. Requiring a row about an occurrence to reference the build it observed (manifest digest in `evidence_refs` or a named member) is an amendment. |
| N5 authority at an instant | partly | Defined per door, not as a general query: self roster at decision (3.3.6.3, freshness bound W); group rosters carry `effective_at` per version with `charter_digest` (3.2 T6); `delegates_to` valid until withdrawn or term lapses (4.4.3.2.8). "`delegates_to` in force at t net of `withdraws`" as one fold of the precedence layer is not written; small, well-posed amendment; persist's `acts_as_steward_at` implements one case. |
| N6 node as subject | partly | A node has no agency (3.4.7.1): not a subject of control or correction families. Legitimate subject of access and proxy/custody rows. Two architectures only through an agent running on the node, not the node itself. |

Consequences for §4: S1's R4 pin (`consent_basis`) exists but is unregistered; S1's R2
pin has no normative field (N4); S3's verdict has a home (N3) with the vocabulary above;
the second architecture must be a second *agent* architecture (N6). The two amendments
(N1 leaf, N4 pin) and the one defect (unregistered `consent_basis`) are filed on
CIRISConstitution for the steward.

Nothing here is a result. §2's "carried by" column is **measured** on rc9; §3–§4 are
requirements; §5 is the ask; §6 is the Constitution session's answer, verified where
it names code.

## 7. Steward rulings (2026-10-21) and the tickets they opened

| Item | Ruling | Tickets |
|---|---|---|
| Untyped capacity members | **Register.** CC 2.1 rows for all ten (`consent_basis`, `n_eff_pr`, `n_eff_h`, `sample_size`, `feature_dim`, `sample_size_gate`, `target_n_eff`, `min_feature_dim`, `n_eff_floor`, `window`), validator the `capacity:*` gate; shapes as emitted at CIRISServer `ffff40e0`. Text edit asked of the Constitution session; recorded on CIRISConstitution#183. | — |
| N1 access assumption | **Full leaf now**, not the interim. `observation:` reopened by one leaf (`observation:access_assumption:v1`, name the steward's), first-person like `observation:reachability`, `subject_key_ids` = the observed agent's keys, members `consent_basis`, `trace_level`, `occurrence_key_ids`, `window`/`window_from`/`window_to`, `channel_classes`. | CIRISConstitution#184; CIRISPersist#1115 (gate); CIRISServer#804 (lens-core emitter, roster pin copy, composite precondition S4) |
| N4 version pin | **On the occurrence row, from the Verify attestation.** Checked (measured, main heads): the agent's `AttestationResult` holds `agent_version` and the binary hash from CIRISVerify; `to_federation_provenance` makes pass/fail entries with no version; nothing forwards them; no receiver in persist or server; `verify_build_attestation_bundle` has no caller; `IdentityOccurrence` has no build member. The design intent is not shipped. | CIRISPersist#1116 (member `build_manifest_hash` + provenance row ref, door check); CIRISAgent#1305 (present it at registration); CIRISVerify#315 (integrity entry names the build; presenter path for the bundle verdict) |

**Landed in CC rc10 at `151d243` (verified against the commit, 2026-10-21):** CC 2.1 rows for the ten capacity members and the five leaf members; `observation:access_assumption:v1` in 3.1.9.4 (`subject_key_ids` REQUIRED non-empty; stem reopened by this one leaf; registry manifest carries it); `build_manifest_hash` / `build_manifest_ref` on the 3.3.6 occurrence row with the rule that a row about a running occurrence resolves its version through the roster; claims `CLM-access-assumption-leaf` (staged on CIRISPersist#1115) and `CLM-occurrence-build-pin` (staged on CIRISPersist#1116). CIRISConstitution#184 closed. **Still unruled:** S4 (composite only over coinciding access rows), carried on CIRISServer#804.

Consequence for §2: R2 is now carried by the occurrence roster and R4 by the new leaf **in the text**; the code tickets decide when the "carried by" column can change from rc9's carriers to these.

