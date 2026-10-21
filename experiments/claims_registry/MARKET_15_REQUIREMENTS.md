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
  accept | reject(fault family) | flag(reason)`; same inputs, same output; published
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

Nothing here is a result. §2's "carried by" column is **measured** on rc9; §3–§4 are
requirements; §5 is the ask.
