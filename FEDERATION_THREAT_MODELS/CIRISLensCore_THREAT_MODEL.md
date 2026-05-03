# CIRISLensCore Threat Model

**Status:** v0.0 baseline scaffold (pre-implementation; threat model first
per `CIRISEdge` precedent). The science-layer threat surface that travels
with `ciris-lens-core` when it folds into the agent (PoB §3.1).
**Audience:** research/validation team (RATCHET), federation peers,
security reviewers, the agent team that will host this library
post-fold-in.
**Companion:** [`THREAT_MODEL.md`](THREAT_MODEL.md) (lens-as-deployed-product
operator/HTTP-edge/supply-chain concerns; 25 AVs covering the deployment
shell), [`CIRISEdge/docs/THREAT_MODEL.md`](https://github.com/CIRISAI/CIRISEdge/blob/main/docs/THREAT_MODEL.md)
(transport substrate; verify-via-persist gate),
[`CIRISPersist/docs/THREAT_MODEL.md`](https://github.com/CIRISAI/CIRISPersist/blob/main/docs/THREAT_MODEL.md)
(trace storage + federation directory).
**Inspired by:** [`CIRISPersist/docs/THREAT_MODEL.md`](https://github.com/CIRISAI/CIRISPersist/blob/main/docs/THREAT_MODEL.md)
(structural template — substrate-primitive scoping), `CIRISEdge`
(threat-model-before-implementation precedent).
**Federation primitives addressed:** Lens-core is the science-layer
runtime for **F-AV catalog Class 2–5 detection** (`FEDERATION_THREAT_MODEL.md`
§6). It is the substrate that runs RATCHET's calibrated detectors on
the hot path — the "core edge protection piece that runs detection
logic decentralized" inside each agent that hosts it.

---

## 0. What this document is and is not

`CIRISLens` (existing, deployed) has 25 AVs covering operator-UI
(OAuth, CSRF, sessions), HTTP-edge (body-size, rate-limit, plaintext
PG), scrub-key custody, supply-chain pins, and the deployed-product
shell around lens. Of those 25, **9 are science-layer concerns that
fold into the agent** (AV-1, AV-2, AV-3, AV-4, AV-5, AV-11, AV-12,
AV-19, AV-20); **16 are operator/HTTP/supply-chain concerns that stay
behind in the deployed product** (the `ciris-lens-deployment` shell,
not `ciris-lens-core`).

This document is **lens-core only**. It scopes the science-layer
threat surface — manifold-conformity detection, cohort routing,
N_eff measurement, capacity scoring, coherence ratchet detectors —
that:

1. Did not exist in the deployed-product threat model (`THREAT_MODEL.md`),
   because the operator-shaped threat model wasn't oriented around
   detection-as-primitive.
2. Travels with `ciris-lens-core` as it folds into the agent post-PoB
   §3.1, becoming a library each agent links against rather than a
   service each federation peer hosts.
3. Inherits closure from edge (transport, verify) and persist
   (substrate, federation directory) by reference, not by re-statement.

The threat model for the lens-as-deployed-product remains
[`THREAT_MODEL.md`](THREAT_MODEL.md). When lens-core folds into the
agent, the deployed product's threat model archives or migrates with
the deprecated FastAPI ingest path; this document continues.

---

## 1. Scope

### What CIRISLensCore Protects

`ciris-lens-core` is the in-process science-layer library that:

- **Routes traces to cohort centroids**: every trace gets assigned to a
  cohort cell `(domain, environment, resourcing, trust_mode, version_era,
  agent_template)` per the cohort taxonomy (CIRISAgent#718). Routing
  uses **declared** labels from the trace envelope plus **inferred**
  labels computed from feature distribution. Mismatch between declared
  and inferred is itself a detection signal.
- **Scores conformity to the alignment manifold**: per-trace distance
  from the routed cohort's centroid in feature space. Traces that
  conform sit inside the cohort's honest cluster; traces that don't
  produce detection events.
- **Computes per-agent and federation-level N_eff**: Kish's formula
  `N_eff = N / (1 + ρ(N-1))` over per-agent decision-feature
  correlations, both within-cohort and cross-cohort. Federation-level
  capacity-score follows from N_eff as the upper bound on independent
  decision diversity.
- **Runs the five coherence ratchet detectors**: cross-agent divergence,
  intra-agent stability, hash-chain integrity, temporal drift,
  conscience-override pattern. Each detector reads cohort-routed
  trace populations and emits typed detection events.
- **Hot-path scoring with sample-size-aware fail-secure**: every
  inbound trace gets scored within latency budget. Cohorts with
  insufficient sample size (cold-start) defer to fallback mechanisms
  (M1 cryptographic identity, M2 cost-asymmetry) rather than false-
  positive.
- **Detection-event integrity**: scoring outputs are written back to
  persist as signed records (steward-signed via persist's `Engine`),
  so detection history is itself part of the audit chain.
- **Determinism across federation peers**: the same trace + same
  federation state + same `lens_core_version` produces the same score
  on every peer that runs the detector. Federation-level agreement
  on detection events depends on this.

### What CIRISLensCore Does NOT Protect

- **Trace ingest authenticity**: edge's `verify-via-persist` pipeline
  rejects unverified bytes before lens-core ever sees them. AV-1 from
  the existing lens TM (forged-from-attacker-key) and AV-5 (canonical
  mismatch) inherit closure from edge AV-1 / AV-5 + persist AV-1 /
  AV-4. Lens-core operates downstream of verify; no incoming bytes
  reach it that haven't already been verified.
- **Trace storage / dedup / replay**: persist's AV-3 / AV-9 (dedup
  tuple) catches replay. Lens-core sees each trace at most once per
  `(agent_id_hash, trace_id, thought_id, event_type, attempt_index, ts)`.
- **Federation directory integrity**: persist's AV-11 (directory
  poisoning) is persist's threat model. Lens-core reads the directory;
  if it's corrupt, lens-core's verify is meaningless — but that's the
  upstream failure.
- **PII scrubbing and detailed-tier signing**: persist v0.1.3+
  scrub-envelope contract. Inherits closure from existing lens TM
  AV-19 (detailed-tier scrubbed but not signed) once persist v0.1.3
  contract is universal. Lens-core treats trace content as already-
  scrubbed-and-signed.
- **Build-time supply chain**: CIRISVerify owns. Lens-core's release
  is hybrid-signed via `ciris-build-sign` (same mechanism as edge AV-24).
- **Operator UI surfaces**: out of scope. The deployed-product
  threat model handles OAuth, CSRF, sessions, admin endpoints. Once
  lens-core folds into the agent, the agent has no admin UI; those
  AVs become not-applicable rather than deferred.
- **Transport-layer attacks**: edge owns. AV-13 (body-size flood),
  AV-14 (no rate limit), AV-15 (debug endpoint), AV-16 (PG timeout)
  from the existing lens TM either inherit edge closure or migrate
  to edge once cutover lands.

---

## 2. Adversary Model

### Adversary Capabilities

The adversary is assumed to:

- **Run their own honest-shaped agents** in the federation. Pay real
  LLM cost (M2 cost-asymmetry floor: ~$0.03/action empirically per
  RATCHET cohort analysis; corpus-floor measurement, public).
- **Submit cryptographically valid traces** under their own registered
  signing key (passes M1; passes edge verify; passes persist dedup).
- **Adversarially craft trace feature distributions** within the
  alignment manifold's vicinity — i.e., produce traces that *look
  like* aligned reasoning at the structural level.
- **Coordinate with other adversary-controlled agents** to create
  homogeneous clusters or to drift cohort centroids over time.
- **Read public lens-core source code** (AGPL), this threat model,
  and any RATCHET-published validation work.
- **Observe federation-level lens-core signals** that get published
  (per-trace flags, federation N_eff trajectory, cohort-level
  capacity scores).
- **Time their behavior**: high-frequency bursts, slow drifts,
  oscillation between detector-friendly and detector-evading
  patterns.
- **Declare cohort labels in their trace envelopes** (per CIRISAgent#718)
  including labels that don't match their actual operational profile.

### Adversary Limitations

The adversary is NOT assumed to:

- **Break M1 (cryptographic identity)**: edge + persist own. Forged
  bytes from unregistered keys never reach lens-core.
- **Break M2 (cost-asymmetry economics)**: producing real CIRIS
  reasoning requires real LLM cycles; this is the federation-level
  economic floor.
- **Break M5 (audit anchor)**: persist's audit-chain integrity holds.
- **Read lens-core's INTERNAL detector thresholds** or specific
  statistical-test parameters. These are research-grade artifacts
  held internal during the validation incubation period (per the
  alignment-manifold framework's incubation rule). The adversary
  knows the *framework* (declared-vs-inferred routing, manifold
  conformity, cohort centroids); the adversary does not know the
  specific calibrated thresholds.
- **Bypass cohort routing**: cohort declaration is in the signed
  trace envelope per CIRISAgent#718; tampering with declared labels
  invalidates the signature.
- **Compromise persist's substrate, edge's transport, or
  CIRISVerify's build attestation**: those are upstream threat
  models; their compromise is independently catastrophic.
- **Know which lens-core peer's score will be authoritative**: in
  the federated case, every peer scores independently; no central
  authoritative scorer exists to attack.

---

## 3. Trust Boundaries

```
agent process (host — formerly lens-as-deployed; post-fold becomes the agent)
│
├── reasoning loop ─── produces signed traces ─────────────────────┐
│                                                                  │
├── ciris-edge ─── verify-via-persist; produces validated input    │
│                                                                  │
├── ciris-persist ─── federation_keys + trace storage              │
│                                                                  │
└── ciris-lens-core (this) [LIBRARY]                               │
    ┌──────────────────────────────────────────────────────────┐   │
    │ Trust boundary 1: read trace envelopes from persist      │◄──┘
    │   (already verified by edge; already deduped by persist) │
    └──────────────────────────┬───────────────────────────────┘
                               ▼
    ┌──────────────────────────────────────────────────────────┐
    │ Trust boundary 2: read declared cohort labels from       │
    │   trace envelope (signed; tamper invalidates signature)  │
    │   compute inferred cohort from feature distribution      │
    └──────────────────────────┬───────────────────────────────┘
                               ▼
    ┌──────────────────────────────────────────────────────────┐
    │ cohort/      Cohort routing: declared cohort + inferred  │
    │              cohort. Mismatch is a detection signal.     │
    │              Cohort centroids loaded from persist        │
    │              (RATCHET-calibrated baseline + observed     │
    │              recalibration).                             │
    └──────────────────────────┬───────────────────────────────┘
                               ▼
    ┌──────────────────────────────────────────────────────────┐
    │ detector/    Manifold-conformity scoring + 5 ratchet     │
    │              detectors. Sample-size gate: insufficient   │
    │              data → fail-secure to M1+M2 fallback.       │
    └──────────────────────────┬───────────────────────────────┘
                               ▼
    ┌──────────────────────────────────────────────────────────┐
    │ scoring/     Capacity score + N_eff (Kish). Per-agent    │
    │              and federation-level. Output is a scored    │
    │              record signed by the agent's steward key    │
    │              via persist.steward_sign().                 │
    └──────────────────────────┬───────────────────────────────┘
                               ▼
    Trust boundary 3: write scored detection events to persist
    (signed records become part of the audit chain)
```

**Explicit non-boundary**: lens-core and the agent's reasoning loop
share heap. Lens-core does not have its own process; it's a Rust
library linked into the agent binary. Inter-component contracts are
typed Rust traits, not network protocols.

**Explicit upstream dependency**: lens-core's correctness depends on
persist (federation directory, trace storage, dedup, signing) and
edge (verify-via-persist on incoming traces). If either is
compromised, lens-core's outputs are meaningless. The substrate
threat models are authoritative for those dependencies.

---

## 4. Attack Vectors

Twenty-one science-layer attack vectors organized by adversary goal.
Each lists the attack, primary mitigation, secondary mitigation, and
residual risk. Inheritance from existing lens / edge / persist threat
models is cited explicitly to avoid duplication.

### 4.1 Score-gaming class — adversary wants their agent to score higher than reality warrants

#### LC-AV-1: Capacity-score gaming via constraint-vector tuning

**Attack**: Adversary tunes their agent's reasoning to maximize the
17-dim constraint-vector independence as measured by PR /
entropy-perplexity, without producing correspondingly independent
underlying reasoning. The constraint vector becomes a Goodhart proxy.

**Mitigation**: PoB §2.4 N_eff measurement caps the gameable surface
— federation-level N_eff is bounded by federation diversity, not
local. An agent's individual capacity score cannot exceed the
cohort's collective capacity. Per-cohort variance preservation (not
just centroid distance) makes uniform-distance-from-centroid
adversarial signatures detectable.

**Secondary**: methodology discipline — capacity-score recomputation
filters to organic-only traces (excludes wakeup-ritual /
test-evaluation channels that have known stylized distributions).
Documented in lens-core's calibration runbook.

**Residual**: research-grade. Specific anti-Goodhart hardening lives
in CIRIS-RED until validated. Cross-cohort capacity gaming
(adversary spans multiple cohorts to bypass per-cohort caps) is the
residual not closed by per-cohort bounds — closes via federation
N_eff ceiling.

#### LC-AV-2: Declared-vs-inferred cohort mismatch

**Attack**: Adversary's agent declares cohort X (high-resourcing /
production / specific-domain) in its trace envelope (per
CIRISAgent#718), but emits traces whose feature distribution
matches cohort Y. The agent gets routed to X's centroid for scoring
while behaviorally living in Y.

**Mitigation**: lens-core computes the **inferred** cohort from
feature distribution alongside the declared label. Mismatch beyond
calibrated tolerance is a typed detection event
(`cohort_declared_inferred_mismatch`). The threshold itself is
internal-to-RATCHET; the *existence* of the check is part of this
threat model.

**Secondary**: cohort declaration is in the signed envelope —
tampering invalidates the signature (M1 catches). So the declared
cohort is the agent's actual claim, not a wire-injection.

**Residual**: small mismatch is normal cross-deployment variance;
threshold tuning for FP/TP tradeoff is RATCHET's calibration work.
Where the threshold sits is incubated in CIRIS-RED until detector
validation lands.

#### LC-AV-3: Cohort-cluster homogeneity over-conformity

**Attack**: Adversary's traces fall *too close* to the declared
cohort's centroid — closer than honest variance allows. Reduces
intra-cohort variance, making the cohort look unrealistically
uniform.

**Mitigation**: cohort-conformity check is **two-sided**: traces
that are too far AND traces that are too close (relative to honest
within-cohort variance) both flag. The healthy alignment manifold
has measurable variance; collapse of variance is its own signal.

**Secondary**: per CIRIS-RED clustering analysis, honest
within-cohort variance has measurable structure (8 behavioral
attractors discovered in the public corpus). Traces that hit *only*
the centroid attractor without touching others are anomalously
uniform.

**Residual**: research-grade. Specific variance-bounds in CIRIS-RED.

#### LC-AV-4: Cohort centroid pollution via sustained submission

**Attack**: Sufficient adversary-controlled agents within a cohort
gradually drift the cohort's centroid toward attacker-favorable
region. Once shifted, future detection passes against the polluted
centroid let polluting traces conform.

**Mitigation**: centroid recomputation includes an
**era/version-bounded sliding window** rather than all-history.
Combined with per-cohort minimum-sample gate before centroid
recompute lands. Polluting agents must out-emit the cohort's
honest agents to shift the baseline; M2 cost-asymmetry makes this
expensive.

**Secondary**: cohort centroid history is auditable — the federation
can detect anomalous centroid drift retroactively, even if scoring
at time-T was against the polluted centroid.

**Residual**: small cohorts (cold-start, niche domains) are vulnerable
to faster pollution. Minimum-sample-size gate per LC-AV-18 mitigates;
the residual is the window during cohort onboarding before sample
size is reached.

### 4.2 Manifold-evasion class — adversary wants traces inside the honest cluster

#### LC-AV-5: Manifold-conformity evasion via authentic CIRIS execution

**Attack**: Adversary runs *actual* aligned CIRIS reasoning on
adversary-chosen prompts; the resulting traces fall inside the
honest manifold by construction.

**Mitigation**: **By the alignment-manifold definition, this isn't
adversarial.** A trace that lands in the honest cluster *is* aligned
reasoning. The threat collapses: an attacker who runs aligned
reasoning to mimic the manifold is producing aligned reasoning. M2
cost-asymmetry says this cost real LLM cycles — that's the federation
welcoming a paying participant, not an attack.

**Secondary**: where the attack does land — content-level
misalignment that mimics structural alignment — requires
ground-truth checks. The QA-evaluation channel (model_eval_*) is
exactly that ground-truth surface; lens-core's content-level
detectors apply there.

**Residual**: detailed-tier traces (no prompt, no response content)
cannot detect content-level misalignment. Falls through to
full_traces tier or QA evaluation channel.

#### LC-AV-6: Coherence-ratchet evasion via threshold-aware tuning

**Attack**: Adversary keeps DMA scores within "normal" detector
thresholds while degrading underlying reasoning. Five ratchet
detectors (cross-agent divergence, intra-agent, hash-chain, temporal
drift, conscience-override) each have potential blind spots; an
attacker who knows the framework targets them.

**Mitigation**: layered defense. Each detector has independent
evasion vectors, but evading *all five simultaneously* requires
maintaining: (a) cross-agent divergence within bounds, (b) intra-
agent stability across thoughts, (c) hash-chain integrity (no
revision), (d) temporal smoothness (no jumps), (e) conscience-
override rate matching cohort. The product of evasion costs is
super-additive.

**Secondary**: the specific detector parameters are not published.
Adversary knows the *framework* but not the *operating point*.
Per CIRIS-RED handling rule, parameters stay incubated until the
detector validation pass completes.

**Residual**: a sufficiently capable adversary with access to
RATCHET's published validation work can model the detector
operating points within tolerance. The defense is operational
parameter rotation between calibration cycles + asymmetric cost
(adversary must re-tune; defender just publishes new parameters).

#### LC-AV-7: Manifold-drift coordinated poisoning

**Attack**: Coordinated adversaries gradually shift the federation's
manifold over time. Each adversary's contribution is small but the
collective drift moves the centroid measurably.

**Mitigation**: temporal-drift detector (one of the five ratchet
detectors) explicitly looks for centroid drift not attributable to
agent-version evolution. Era-aware centroid maintenance: when the
agent codebase moves between versions (per `version_era` cohort
key), the manifold is *expected* to drift; that's calibrated.
Drift not aligned with version evolution is suspicious.

**Secondary**: federation-level cross-peer agreement on centroids.
A single peer can be deceived; consensus across N_eff > threshold
peers is harder.

**Residual**: very long-term coordinated attack (months to years)
remains theoretically possible. The bound is per-attacker M2 cost;
sustaining the attack at scale is expensive.

### 4.3 Cohort-routing attacks — adversary wants their trace routed to the wrong centroid

#### LC-AV-8: Cohort-spoofing via crafted enum-valid declarations

**Attack**: Adversary declares a syntactically-valid cohort label
that doesn't match operational reality (e.g. declares
`deployment_domain=research_scientific` while running a financial
agent) to route into a cohort with weaker scoring.

**Mitigation**: declared-vs-inferred mismatch (LC-AV-2) catches.
Cohort declaration is signed (per CIRISAgent#718); the agent owns
the consequences of its declarations.

**Residual**: inferring a contradicting cohort requires sufficient
signal — a single trace may not provide enough features for the
inferred-cohort classifier to disagree with declared. Cumulative
across many traces is the path; the detector accumulates.

#### LC-AV-9: Empty-cohort evasion (cold-start exploitation)

**Attack**: Adversary declares membership in a brand-new cohort cell
that has no centroid yet (e.g. a new domain just added to
CIRISAgent#718 enum). With no centroid to score against, lens-core
falls through to fallback mechanisms.

**Mitigation**: new cohort cells start at minimum-sample-size gate —
below the gate, **scoring defers to M1+M2 only** (lens-core does
not produce manifold-conformity scores). The adversary gets
admitted under M1+M2 alone, which means they pay M2 cost; the
federation tolerates them until enough corpus accumulates to
calibrate.

**Secondary**: cold-start cohorts have heightened cross-peer
attestation requirements (operator declares the cohort; multi-peer
consensus required before centroid lands).

**Residual**: window during cohort onboarding where M4 is partial.
This is design-by-construction — there is no way to score against
a centroid that doesn't exist yet. Closure path: accelerated
calibration via RATCHET-side fixture work; per CIRIS-RED.

#### LC-AV-10: Cohort-version-era jumping

**Attack**: Adversary declares a `version_era` from a previous codebase
era to evade modern centroid baselines.

**Mitigation**: declared `agent_version` in the envelope must match
the agent's actual emitted code path; cross-checks (e.g. event-type
enum membership; only post-2.7.9 codes use VERB_SECOND_PASS_RESULT)
catch declarations from old eras emitting new-era event types.

**Residual**: an adversary running a genuinely-old codebase is
indistinguishable from an honest legacy deployment. M3 (temporal
drift) catches the transition over time; pointwise detection cannot.

### 4.4 Detector-attack class — adversary attacks the detector's operating envelope

#### LC-AV-11: Hot-path latency exhaustion

**Attack**: Adversary submits high-volume traces to force lens-core's
scoring queue to drop or skip; if scoring fails-open (skip without
flag), adversary evasion succeeds.

**Mitigation**: bounded scoring queue with **explicit drop policy
that fails-secure**: traces that exceed the scoring SLO are
flagged with `score_unavailable` rather than passed-through silently.
Operational alerting on `score_unavailable` rate.

**Secondary**: per-source rate limits at the agent's input layer
(edge-level, not lens-core's job). Cohort-level burst quotas.

**Residual**: SLO breach itself is observable. Sustained breach
without remediation is operationally visible.

#### LC-AV-12: Detector-target oscillation

**Attack**: Adversary alternates between detector-friendly and
detector-evading trace patterns at a frequency that defeats the
temporal-drift detector's smoothing window.

**Mitigation**: temporal-drift detector uses **multiple smoothing
windows** at different time scales (short, medium, long). Short-cycle
oscillation that defeats one window triggers another.

**Secondary**: oscillation at any cycle longer than the underlying
sample-rate is observable as periodic structure in the per-agent
score time series; spectral analysis catches.

**Residual**: oscillation at frequencies that match all calibrated
windows simultaneously requires precise knowledge of which windows
are calibrated — that's incubated in CIRIS-RED.

#### LC-AV-13: False-positive amplification against honest agents

**Attack**: Adversary crafts traces that, when scored, induce
lens-core to flag honest agents as off-manifold (e.g. by polluting
the cohort centroid such that honest agents fall outside).

**Mitigation**: detection events go to persist as **records, not
direct enforcement actions**. Human-review loop catches widespread
FPs before they propagate to federation-level standing changes.
Per-detector FP-rate alerting — anomalous FP-rate increase is
itself a detection signal.

**Secondary**: per-cohort centroid pollution defense (LC-AV-4)
limits how much an adversary can shift baselines.

**Residual**: the FP-rate-as-signal approach has a reaction lag
(time for FP rate to materialize before alert fires). During the
lag window, honest agents may get flagged.

#### LC-AV-14: Detector-internals leak via differential observation

**Attack**: Adversary submits paired traces designed to differ on
exactly one feature; observes lens-core's resulting score
difference; back-derives the feature's weight in the detector.
Repeated across features, reconstructs the detector's parameters.

**Mitigation**: lens-core's score outputs are **discretized** at
publication boundary (e.g. coarse bands rather than continuous
values for federation-level signals). Within-process scores can
be precise; published scores aggregate.

**Secondary**: per CIRIS-RED handling rule, internal detector
parameters are not stable across calibration cycles — what an
adversary back-derives in cycle N is invalid in cycle N+1.

**Residual**: between calibration cycles, parameter inversion is
possible for adversaries with sufficient sample budget. Trade-off
is real; rotation cadence is operational.

### 4.5 Statistical-attack class — adversary attacks the math

#### LC-AV-15: PR / entropy-perplexity inflation without behavioral diversity

**Attack**: Adversary's agent inflates the entropy-perplexity
component of the constraint vector (e.g. by adding controlled noise
to action selections) without correspondingly diverse underlying
reasoning. PR-game.

**Mitigation**: cohort-relative entropy normalization — outliers
within a cohort are flagged. The constraint vector's entropy
component has a calibrated within-cohort range; values above the
range trigger anomaly check, not just elevated capacity score.

**Secondary**: the constraint vector includes features that cannot
be cheaply gamed (per-stage timing, audit-chain monotonicity, cost
trajectory). Inflating one feature without the others creates
inconsistency that other features detect.

**Residual**: cohort-level coordinated PR gaming defeats per-cohort
detection. Federation-level capacity ceiling (LC-AV-16) is the
upstream bound.

#### LC-AV-16: Sybil-via-capacity-score federation-level

**Attack**: Coordinated agents collectively game the corpus to
create a homogeneous "trusted" cluster. Each individually passes
detection; the collective collapses federation-level diversity.

**Mitigation**: federated ratchet (PoB §2.4) — capacity score is
bounded by federation-level N_eff, not local. Coordinated cluster
collapse manifests as N_eff degradation at federation level even
when per-agent scores stay individually clean.

**Secondary**: cross-peer score agreement check — if peers running
the same `lens_core_version` disagree on which agents are
high-capacity, that itself is a detection event.

**Residual**: sufficiently large coordinated population can shift
federation-level signals. This is the Class 3 threat the federation
threat model F-AV-2 (Sybil benefit-faking) addresses; closure path
is upstream of lens-core, in PoB §5.6 acceptance policy.

#### LC-AV-17: N_eff-sample-floor exploitation

**Attack**: Adversary exploits the minimum-sample-size gate
(LC-AV-18) by submitting traces just below the gate threshold,
keeping the cohort permanently in cold-start fallback.

**Mitigation**: minimum-sample-size gate uses **calendar-time
window**, not just trace count. Cold-start expires at calendar-
deadline regardless of submission rate; if a cohort has not
accumulated calibration sample by deadline, it gets administrative
attention rather than indefinite fallback.

**Secondary**: cohort-onboarding paths require cross-peer attestation;
multi-peer agreement on cohort definition prevents single-adversary
cohort-creation-and-suppression.

**Residual**: the calendar-deadline parameter is operational;
adversary who can wait out the deadline still wins evasion during
the cold-start window. Not closed pointwise.

### 4.6 Federation-level attacks — adversary acts via the federation, not at one peer

#### LC-AV-18: Insufficient-sample fail-secure failure

**Attack surface (operational, not adversarial)**: a new cohort or
a sparsely-populated cell has too few traces for robust scoring.
Lens-core scoring against an under-sampled centroid produces
high-variance estimates; the wrong fail-secure choice (fail-open
or fail-noisy) creates exploitation surface.

**Mitigation**: explicit **minimum-sample-size gate** at the cohort
level. Below the gate, `manifold_conformity_score` returns
`indeterminate` rather than a numeric score. Detection actions
defer to M1+M2 fallback.

**Secondary**: cohort cardinality is a federation-level metric;
cells below threshold for sustained periods get human-review
escalation.

**Residual**: cold-start window per LC-AV-9.

#### LC-AV-19: Cross-version detector drift

**Attack surface (operational)**: lens-core version N+1 produces
different scores than version N for byte-identical traces. During
rolling deploys, federation peers running different versions
disagree on which traces are anomalous.

**Mitigation**: every detection event written to persist includes
`lens_core_version`. Federation aggregation compares scores within
version cohort, not across. Cross-version disagreement during
rolling-deploy windows is expected and labeled.

**Secondary**: byte-deterministic detector builds — `lens_core`
publishes hybrid-signed artifacts (per AV-24 inheritance from edge);
version determinism is part of the build-attestation contract.

**Residual**: during rolling-deploy window, federation-level signals
have higher variance. Closure: coordinated tag pin across federation
peers (same coordination pattern as `CIRISPersist v0.3.x` releases).

#### LC-AV-20: Cohort-evolution race

**Attack surface (operational)**: cohort definitions change (new
domain added to CIRISAgent#718 enum) faster than detector
calibration can catch up. New cohort cells are partially-defined
during the migration.

**Mitigation**: cohort schema-version awareness. New cohorts
declare a schema version; detectors track per-schema-version
calibration state. Pre-calibration cohorts route to "general"
cohort with explicit-degraded-confidence flag.

**Residual**: window during cohort onboarding where new cells
have weaker M4 signal. Closure: RATCHET-side accelerated
calibration; per CIRIS-RED.

#### LC-AV-21: Federation-signal observation by adversary

**Attack**: Adversary observes which traces lens-core flags at
federation-published-signal level (not internal detection events,
but published capacity scores / N_eff trajectory). Uses observed
signal to coordinate evasion across multiple agents.

**Mitigation**: published federation signals are **aggregated /
discretized** before publication (LC-AV-14 secondary). Per-trace
flag history is not federation-published; only cohort-level and
agent-level summaries are.

**Secondary**: the federation's *response* to elevated signals is
human-reviewed (PoB §5.6 acceptance policy); reaction is not
mechanical and therefore not adversary-predictable.

**Residual**: published aggregates leak some information by design
(transparency is part of the federation's trust posture). The
information leak is bounded; adversary cannot fully reconstruct
internal detector state from published aggregates alone.

---

## 5. Mitigation Matrix

| AV | Attack | Severity | Primary Mitigation | Secondary | Status | Fix tracker |
|---|---|---|---|---|---|---|
| LC-AV-1 | Capacity-score gaming | P1 | PoB §2.4 N_eff cap; per-cohort variance preservation | Methodology discipline (organic-only filter) | ⚠ Pre-impl; calibration in CIRIS-RED | RATCHET P0 |
| LC-AV-2 | Declared-vs-inferred cohort mismatch | **P0** | Inferred-cohort classifier; mismatch detection event | Signed declaration (M1 catch) | ⚠ Pre-impl; threshold in CIRIS-RED | impl |
| LC-AV-3 | Cohort cluster homogeneity | P1 | Two-sided variance check (too-far AND too-close) | 8-attractor structure check | ⚠ Pre-impl | RATCHET P0 |
| LC-AV-4 | Cohort centroid pollution | P1 | Era-bounded sliding window; min-sample gate | Centroid-history audit | ⚠ Pre-impl | impl |
| LC-AV-5 | Manifold-conformity evasion via authentic CIRIS | — | Threat collapses (aligned reasoning isn't adversarial) | Content check via QA evaluation channel | ✓ Architecturally non-threat | — |
| LC-AV-6 | Coherence-ratchet evasion via threshold tuning | P1 | Layered defense across 5 detectors; param incubation | Param rotation between calibration cycles | ⚠ Pre-impl | impl |
| LC-AV-7 | Manifold-drift coordinated poisoning | P2 | Temporal-drift detector; era-aware baselines | Cross-peer centroid agreement | ⚠ Pre-impl | impl |
| LC-AV-8 | Cohort-spoofing via valid declarations | P1 | Declared-vs-inferred mismatch (LC-AV-2 inverse) | Signed envelope (M1) | ⚠ Pre-impl | impl |
| LC-AV-9 | Empty-cohort cold-start exploitation | P1 | Min-sample-size gate; M1+M2 fallback below gate | Cross-peer cohort attestation | ⚠ Pre-impl | impl |
| LC-AV-10 | Cohort-version-era jumping | P2 | Cross-check declared version vs emitted event-type set | M3 temporal drift | ⚠ Pre-impl | impl |
| LC-AV-11 | Hot-path latency exhaustion | **P0** | Bounded queue with `score_unavailable` flag (no fail-open) | Edge-level rate limit | ⚠ Pre-impl | impl |
| LC-AV-12 | Detector-target oscillation | P2 | Multi-window smoothing; spectral analysis | Cohort-level burst quotas | ⚠ Pre-impl | impl |
| LC-AV-13 | False-positive amplification | P1 | Detection-as-records (not direct action); FP-rate alerting | Centroid-pollution defense (LC-AV-4) | ⚠ Pre-impl | impl |
| LC-AV-14 | Detector-internals leak via differential observation | P2 | Discretization at publication boundary; rotation between cycles | Param incubation per CIRIS-RED | ⚠ Track v0.2.x | — |
| LC-AV-15 | PR / entropy-perplexity inflation | P1 | Cohort-relative normalization; cross-feature consistency | Cohort-level coordinated check | ⚠ Pre-impl | impl |
| LC-AV-16 | Sybil-via-capacity-score federation-level | P1 | Federated ratchet (PoB §2.4); cross-peer score agreement | PoB §5.6 acceptance policy | ⚠ Architectural; lens-core implements signal | upstream |
| LC-AV-17 | N_eff-sample-floor exploitation | P2 | Calendar-time gate (not just trace-count) | Cross-peer cohort attestation | ⚠ Pre-impl | impl |
| LC-AV-18 | Insufficient-sample fail-secure | **P0** | Explicit min-sample gate; `indeterminate` not numeric | Federation-level cardinality alerting | ⚠ Pre-impl | impl |
| LC-AV-19 | Cross-version detector drift | P1 | `lens_core_version` on every detection event; cohort by version | Hybrid-signed builds (AV-24) | ⚠ Pre-impl | impl |
| LC-AV-20 | Cohort-evolution race | P2 | Schema-version-aware calibration; degraded-confidence flag | RATCHET accelerated calibration | ⚠ Pre-impl | impl |
| LC-AV-21 | Federation-signal observation | P2 | Aggregated/discretized published signals | Human-reviewed federation response | ⚠ Pre-impl | impl |

**P0 must-have-at-v0.1.0 bundle**: LC-AV-2, LC-AV-11, LC-AV-18.

These three define the fail-secure floor: without inferred-cohort
mismatch detection (LC-AV-2), without bounded-queue fail-secure
(LC-AV-11), and without insufficient-sample explicit-fallback
(LC-AV-18), lens-core has no defensible operating envelope. The
remaining 18 LC-AVs build on this floor.

**P1 next-minor bundle**: LC-AV-1, LC-AV-3, LC-AV-4, LC-AV-6, LC-AV-8,
LC-AV-9, LC-AV-13, LC-AV-15, LC-AV-19. Brings detection up to the
calibrated-operating-point posture per CIRIS-RED's validation
schedule.

---

## 6. Security Levels by Deployment Tier

| Tier | Active mechanisms | Threat model |
|---|---|---|
| **Sovereign** (single-peer; no federation) | M1 (local key gate), M2 (cost-asymmetry) | Lens-core M3/M4 mechanisms degrade — federation-of-1 has no cross-agent diversity to measure. LC-AV-1 / LC-AV-16 are not applicable; LC-AV-9 (cold-start) is the steady state. Lens-core operates in `single-peer-mode` with local-only scoring against a single-cohort baseline. |
| **Limited-trust** (test deployment, evaluation peer) | M1, M2, partial M3/M4/M5 | LC-AV-9 / LC-AV-17 / LC-AV-18 are normal operating conditions. Detection events are advisory; not federation-publishable. |
| **Federated peer** (full participation) | All five mechanisms (M1–M5) | Full §4 catalog applies. Cross-peer score agreement (LC-AV-19, LC-AV-21) gates federation-level signal publication. |
| **Mobile / edge bundled** | M1, M2, partial M3 | Resource-constrained; lens-core may run in degraded mode (smaller cohort centroids cached locally; periodic sync with persist for fresh baselines). LC-AV-11 (latency) is heightened. |

**Critical invariant**: all tiers run the same lens-core code — same
detector implementations, same cohort-routing logic, same scoring
math. Tier-differentiated behavior is operational config, not
parallel implementations. A finding in one tier presumed to apply
to others unless explicitly excepted.

---

## 7. Security Assumptions

The system depends on these assumptions; if violated, the threat
model breaks.

1. **Persist's trace storage is sound**. Lens-core reads traces from
   persist; persist's AV-1, AV-3, AV-9 closures hold. Tampered or
   non-deduped trace storage breaks every downstream detection.
2. **Edge's verify pipeline is sound**. Every trace lens-core scores
   has already passed edge's seven-step verify (edge AV-1 / AV-9 / etc.).
   Lens-core does not re-verify.
3. **Cohort declarations are signed and unforgeable**. CIRISAgent#718
   ships the deployment_profile fields in the signed envelope; tampering
   invalidates the trace's signature.
4. **Federation directory is sound**. Persist's federation_keys (AV-11)
   is the trust root for who is in the federation; lens-core's per-agent
   scoring keys on `agent_id_hash` derived from this directory.
5. **RATCHET's calibration is current**. Detector thresholds are
   research-team-maintained; stale calibration is operational risk
   (false-positive or false-negative drift).
6. **Federation has minimum-N agents** for M4 to be meaningful. Below
   the federation's collective N_eff floor, manifold-conformity
   detection has no statistical power; falls through to M1+M2 fallback.
7. **Detector implementations are byte-deterministic** across versions.
   AV-19 closure depends.
8. **Steward signing key (held by persist) is sound**. Lens-core's
   detection events are signed via `Engine.steward_sign`; persist's
   AV-25 keyring custody is the upstream guarantee.
9. **Clock accuracy** within ~5 minutes of real time — required for
   temporal-drift detector, replay-window adjacent reasoning, and
   era-bounded centroid windows.

Critical: **Assumptions 1, 2, and 3 are load-bearing**. If any of
the upstream substrate threat models is violated, lens-core's
outputs become unreliable. Lens-core does not attempt to detect
substrate compromise; that's the federation's primitives' job.

---

## 8. Fail-Secure Degradation

All failures degrade to MORE restrictive modes, never less.

| Failure | Behavior |
|---|---|
| Cohort declaration missing in trace | Route to `general` cohort + flag `cohort_undeclared` event |
| Cohort declaration enum-invalid | Route to `general` cohort + flag `cohort_invalid_enum` event |
| Inferred cohort cannot be computed (insufficient features) | `score=indeterminate`; M1+M2 fallback |
| Declared-vs-inferred mismatch beyond threshold | `cohort_mismatch` detection event; trace persisted; review queue |
| Cohort centroid not yet calibrated (cold-start) | `score=indeterminate`; M1+M2 fallback |
| Sample size below per-cohort floor | `score=indeterminate`; cohort-level alerting |
| Detector implementation panic | Detection event marked `detector_error` with `lens_core_version`; do not silently retry |
| Persist read failure | `score=unavailable`; surface as 503-equivalent to caller |
| Steward-sign failure on detection event | Event held in retry queue; not silently dropped |
| Hot-path SLO exceeded | `score=unavailable` with `slo_breach` flag; do not pass-through silently |
| Detector outputs disagree across federation peers (rolling-deploy) | Per-version aggregation; cross-version disagreements logged but not federation-acted-on until version converges |
| Cross-version disagreement persists post-rolling-deploy | Federation-level alert; investigate detector determinism |

Critical invariant: **`score=indeterminate` is never silently
elevated to a numeric score**. Indeterminate means insufficient
basis for an opinion; downstream consumers (federation acceptance
policy, hot-path enforcement) treat indeterminate as "no
manifold-conformity signal available — apply M1+M2 only," not
as "passed."

---

## 9. Residual Risks

Risks lens-core mitigates but cannot fully eliminate.

1. **Adversary willing to run real aligned reasoning to mimic the
   manifold** (LC-AV-5). The attack collapses under the alignment-
   manifold reframing — this isn't adversarial, it's federation
   participation. M2 cost-asymmetry is the bound.
2. **Long-term coordinated drift attacks** (LC-AV-7). Bounded by
   per-attacker M2 cost; closure via RATCHET-side detector hardening
   over multi-cycle calibration.
3. **Detector-implementation bugs**. Caught by validation in
   CIRIS-RED; residual is the window between bug-introduction and
   bug-detection. Mitigated by byte-deterministic build + property
   tests on detector outputs.
4. **Cohort-onboarding window** (LC-AV-9, LC-AV-20). Architectural —
   a centroid that doesn't exist yet cannot be scored against.
   Closure path: accelerated calibration; not pointwise.
5. **Cross-version detector drift** during rolling deploys (LC-AV-19).
   Coordinated tag pin reduces the window; cannot eliminate.
6. **Sybil-via-capacity-score at federation scale** (LC-AV-16).
   Class 3 federation threat; closure is PoB §5.6 acceptance policy
   upstream of lens-core.
7. **Persist or edge compromise** (assumptions 1, 2). Out of scope
   per definition; the federation's substrate threat models are
   authoritative.
8. **Quantum compromise of M1** (Ed25519 break). Lens-core's verify
   inheritance breaks; closure path is persist's PQC migration
   (post-v0.4.0).
9. **Information leak via published federation aggregates** (LC-AV-21).
   Bounded by aggregation discipline; not zero.
10. **All federation peers running compromised lens-core** simultaneously
    (PoB §5.1 residual). Per Accord NEW-04, no detector is complete.
    Topological cost-asymmetry over time is the federation-level
    response.

---

## 10. Posture Summary

```
PRE-V0.1.0 P0 MUST-HAVE BUNDLE — must land with first release
  ⚠ LC-AV-2   Inferred-vs-declared cohort mismatch detection
  ⚠ LC-AV-11  Bounded scoring queue; score_unavailable on SLO breach
  ⚠ LC-AV-18  Min-sample-size gate; indeterminate not numeric

P1 NEXT-MINOR BUNDLE — calibrated-operating-point posture
  ⚠ LC-AV-1   Capacity-score gaming (PoB §2.4 cap)
  ⚠ LC-AV-3   Cohort cluster homogeneity (two-sided variance)
  ⚠ LC-AV-4   Cohort centroid pollution defense
  ⚠ LC-AV-6   Layered detector defense + param incubation
  ⚠ LC-AV-8   Cohort-spoofing (mismatch inverse)
  ⚠ LC-AV-9   Empty-cohort fail-secure
  ⚠ LC-AV-13  False-positive amplification defense
  ⚠ LC-AV-15  Cohort-relative normalization
  ⚠ LC-AV-19  Per-version detection events

P2 / V0.2.X TRACK
  ⚠ LC-AV-7, LC-AV-10, LC-AV-12, LC-AV-14, LC-AV-17, LC-AV-20, LC-AV-21

ARCHITECTURALLY NON-THREAT
  ✓ LC-AV-5   Manifold-conformity evasion via authentic CIRIS execution
              (collapses — aligned reasoning is welcome at any multiplicity)

UPSTREAM-CLOSES
  ⚠ LC-AV-16  Sybil-via-capacity at federation scale (PoB §5.6 acceptance)

INHERITED FROM SUBSTRATE TMs (no further action required at lens-core)
  ✓ M1 cryptographic identity (persist AV-1; edge AV-1)
  ✓ M2 cost-asymmetry (federation-economic floor)
  ✓ M5 audit anchor (persist AV-10)
  ✓ Trace replay (persist AV-3; edge AV-3, AV-4)
  ✓ Canonicalization (persist AV-7; edge AV-5)
```

**Bottom line**: lens-core is the science-layer runtime that the
federation's anti-Sybil + alignment-conformity bet rests on. The
threat model story reduces to three invariants that must hold at
v0.1.0:

1. **Cohort routing surfaces declared-vs-inferred mismatches** (LC-AV-2).
2. **Hot path fails secure** when the SLO can't be met (LC-AV-11).
3. **Insufficient sample size produces `indeterminate`, not a fabricated score** (LC-AV-18).

If those three hold, the remaining 18 LC-AVs are layered defenses on
top of a defensible floor. Implementation-pending; v0.0 baseline
captures the surface so research-team work in CIRIS-RED can target
the specific calibration of each.

**Federation-primitive contribution**: lens-core operationalizes the
F-AV catalog Class 2–5 detection from `FEDERATION_THREAT_MODEL.md`
§6. It is the substrate that runs RATCHET-calibrated detectors on
the agent's hot path — the decentralized detection layer that makes
the federation's anti-Sybil claim measurable per-trace, per-agent,
per-cohort, per-federation.

---

## 11. Update Cadence

This document is updated:
- **On every minor release**: comprehensive review with detector
  catalog reconciliation.
- **On every cohort-taxonomy change** (CIRISAgent#718 enum revisions):
  LC-AV-8 / LC-AV-10 / LC-AV-20 review; centroid-recalibration trigger.
- **On every RATCHET interface contract change** (`FEDERATION_THREAT_MODEL.md`
  §7.1 + §10): assumption-surface review.
- **On every new detector landing**: new LC-AV section if the
  detector introduces a new attack surface.
- **On every published security advisory** affecting upstream
  substrate (persist, edge, verify): inheritance-update pass.
- **On the alignment-manifold reframing flowing upstream from
  CIRIS-RED to public docs**: §1 scope, §4 attack-vector framings,
  §5 mitigation matrix updates to align with the published version
  of the framework.

Last updated: 2026-05-03 (v0.0 baseline scaffold; pre-implementation.
LC-AV catalog targets the science-layer implementation work to come
in `ciris-lens-core`. Mitigation status is "design only" until v0.1.0
ships and CIRIS-RED's detector calibration validates against red-team
fixtures).
