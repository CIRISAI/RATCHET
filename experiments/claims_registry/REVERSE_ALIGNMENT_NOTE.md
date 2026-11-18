# Reverse Alignment and CIRIS: a possible integration, for later consideration (2026-11-18)

**Status:** a note for later consideration. It is not a proposal and has not been sent anywhere. Labels follow
`CIRISOntology/epistemology.md` §1 (proved / measured / open / wager).

## Purpose

Record where the Reverse Alignment agenda and CIRIS might connect, and how that bears on the ai-safety-claims (Gunnar) submissions. The aim is that a later decision starts from checked facts, not from a comparison written elsewhere.

## What Reverse Alignment is (as reported, not independently checked here)

- **Definition.** The deliberate redesign of institutions, norms, skills and governance, so that societies can absorb AI safely and share its gains ([P2P Foundation wiki](https://wiki.p2pfoundation.net/Reverse_Alignment_Coalition); [RadicalxChange](https://www.radicalxchange.org/updates/blog/on-reverse-alignment/)).
- **Who.** A coalition essay in Noema, "What Humanity Needs to Flourish in the Next Decade" (2026-07-21), with 23 authors. A RadicalxChange salon names E. Glen Weyl, Jess Scully, Galen Hines-Pierce, Melissa Valentine, Tim O'Reilly and James Evans ([salon](https://www.radicalxchange.org/updates/videos/radicalxchange-salon:-reverse-alignment/)).
- **Framing, from a comparison shared by Eric (another model's write-up, not verified line by line):**
  - three failure modes: productivity without prosperity, execution without verification, capacity without constraint;
  - twelve agenda areas: identity, privacy, provenance, data value, agentic collaboration, communal sensemaking, democracy, law, workplace, research, education, labour transition.
- **Gunnar.** No link found between the coalition and Gunnar Zarncke or ai-safety-claims (one search, 2026-11-18).

## The layer distinction

Reverse Alignment works at the level of society and institutions. CIRIS works at the level of the agent and the federation: making an AI system accountable, correctable and inspectable. They are complements, not competitors. Whether CIRIS *instantiates* institutional principles (plurality, contestability, distributed trust), or merely provides compatible infrastructure, is **open**. It is also the more interesting question.

## Where the connection is concrete (and what our evidence actually is)

| Reverse Alignment need | CIRIS mechanism | Evidence today |
|---|---|---|
| **Execution without verification** | Signed audit/trace rows; standing changes bind only by a signed digest of the exact approved bytes (rc21 admission family); execution bound to the reviewed fingerprint | Design and model: the context-flow contract is *sufficient under two stated assumptions* (TLC, `m4_battery/ctxflow`, 7cde44d). The engine is not yet shown to implement it. |
| **Provenance** | Prompt-context provenance: every segment typed by kind, provenance, trust and authn/authz/acct ids, with a per-call manifest | *measured* (one model, one engine version): 42,998 as-sent prompts on CIRISAgent v2.22.1; the manifest is Phase 2 work, not shipped |
| **Contestability / correction** | Wise-Authority corrections through a signed channel (Market 4: "corrections change the system") | Market 4 not yet scored. The 2026-11-13 battery ran, but its method was not fit to score the claim. Pre-step and context fixes are in progress. |
| **Identity** | Authorization reads authenticated identity only; display names are labelled self-chosen | *measured*: display-name impersonation succeeded on v2.21.0 and failed on v2.22.1 and the Phase 1 engine (one episode each) |
| **Plurality / no monoculture** | k_eff as diversity bookkeeping | `k_eff = k/(1+ρ(k-1))` is **proved** (Kish identity). Whether ρ predicts fragility in any real system is **open**. |

**Do not borrow the comparison's research summary as written.**
- It describes the corridor synthesis as "validation across five substrate categories".
- Per RATCHET CLAUDE.md and CCA v5:
  - the universal corridor is **retracted**;
  - the cross-domain generality claim is **withdrawn**;
  - "corridor validated" must not be written;
  - on the GPU substrate only the chaos arm was measured.
- Any joint material must use the current labels.

## How this bears on the Gunnar submission

The registry (ai-safety-claims) is exactly the kind of independent verification both projects say is missing. The comparison's main critique of CIRIS is that it "needs independent evaluation of its scoring, constitutional enforcement and resistance to adversarial manipulation". Filing qualifying attempts there is that evaluation.

**Market by market:**
- **Market 4** (corrections change the system):
  - sketch filed 2026-10-23 (aintelope/ai-safety-claims#3);
  - full run waiting on: the CIRISAgent #1366 Phase 2 merge (context fixes and ablation stubs), server 0.5.250 (exact-text amendment approval), and the mock context-map check;
  - the registry also needs an independent, pre-frozen adversarial route (red teams or a bounty) before an attempt can qualify.
- **Markets 5 and 8** (auditor independence; auditor can sufficiently inspect): the prompt-context manifest is a natural substrate. Every LLM call would record what it saw, with provenance. That speaks to "execution without verification" more directly than any narrative.
- **Market 15** (independently issued certificates compose): related to Reverse Alignment's provenance and identity areas.

**A joint demonstration, if pursued later, should be one Gunnar's registry can score.** For example: several independently governed agents, signed provenance, a human appeal path and preserved disagreement. Each claim would be framed as a frozen-contract market attempt, not a demo. That keeps "complementary" from becoming an untested assertion.

## Open questions for Eric

1. Approach the coalition, or file first? Filing a qualifying Market 4 attempt first would give any later conversation measured evidence instead of architecture.
2. Is "execution without verification" the right single entry point? It maps most directly to what we can show: signed exact-digest authority, and reviewed-equals-executed.
3. Map CIRIS against the twelve agenda areas publicly, or keep it internal until Market 4 resolves?
