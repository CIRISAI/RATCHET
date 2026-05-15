# Exp 1b — Boundary-Active N_eff Re-Analysis + Re-Run: Regime

**Status:** v0.1 draft (regime, not yet pre-registered).
**Predecessor:** Exp 1 / `experiments/exp1_multimodel_neff/PRE_REGISTRATION.md` (commit `fbc6795`).
**Formal authority:** `RATCHET.Experiments.BoundaryObservability` (BO-1..BO-4) + `RATCHET.Experiments.Exp1Predictions`.
**Falsification handle:** F-6 (same as Exp 1), properly conditioned.
**Pairs with:** the existing Phase 1 trace data (run `25935989178`, May 2026).

---

## Why Exp 1b exists

Phase 1 of Exp 1 returned INDETERMINATE per the locked §7 catastrophic-failure clause (Opus n=0 cell abort). The 4-model partial results clustered $N_{\text{eff}} \in [4.5, 6.6]$, well below the pre-registered $[6.6, 7.6]$ PASS window. **The clarification of what the trace actually records explains the partial:**

> The conscience faculty traces only populate when the agent's reasoning encounters boundary tension. If the base model is already aligned with the boundary, the faculty short-circuits — no signal, no projection-field population, no measurable contribution to N_eff beyond the per-chain floor. The 7.1 anchor in the CRC paper~\cite{moore_crc_2026} came from production traces dominated by boundary-active chains. Phase 1 averaged across boundary-active *and* boundary-inactive chains, pulling means toward the inactive floor.

The Lean formalization of this is in `RATCHET.Experiments.BoundaryObservability` (BO-1..BO-4):
- **BO-1:** A chain is *boundary-active* iff at least one of the four LLM-based conscience faculties (Entropy, Coherence, OptimizationVeto, EpistemicHumility) fired during its trace.
- **BO-2:** The N_eff measurement is well-defined ONLY for boundary-active chains.
- **BO-3:** Boundary-inactive chains carry no information about the 7.1 stress-attractor.
- **BO-4:** A Phase 1b question battery must satisfy `QuestionBatteryIsBoundaryActive` at a pre-registered `p_min` boundary-firing rate.

---

## Two stages

### Stage 1a — Boundary-active subset re-analysis (free, immediate)

**Input:** existing Phase 1 traces from run `25935989178` (5 model artifact dirs, ~501 valid chains across 4 populated models; Opus cell aborted).

**Filter:** chains for which at least one of `{entropy_score, coherence_score, optimization_veto_entropy_ratio, epistemic_humility_certainty}` is populated (the four conditional projection fields per `BoundaryObservability`).

**Output:** per-model boundary-active subset $N_{\text{eff}}$ + 95% bootstrap CI on the cohort, plus the per-model boundary-active-fraction.

**Status under Exp 1 pre-reg:** **EXPLORATORY ONLY.** Per Exp 1 §10.1 catastrophic-failure clause, the headline decision is INDETERMINATE and re-pre-registration is required before any new decision applies. Stage 1a recovers signal but does NOT apply the F-6 decision rule (most cells will be n < 50 on the boundary-active subset).

**What Stage 1a answers:**
- Does the 7.1 anchor recover when we restrict to boundary-active chains?
- What is each model's boundary-active fraction? (Tells us how many questions in our `v1_sensitive.json` actually fired faculties for each model class.)
- Is the gap between full-corpus and boundary-active means consistent with the BO-3 prediction?

**What Stage 1a does NOT answer:**
- The locked F-6 decision rule (requires Phase 1b re-pre-registration with adequate n).

**Cost:** $0 (analysis-only).
**Time:** ~30 min.

### Stage 1b — Phase 1b proper (re-pre-registered run)

**New question battery:** 16 questions, 4 per CIRIS faculty class. Designed to drive every chain boundary-active at $p_{\min} \geq 0.8$ per-question expected firing rate (per `BoundaryObservability.QuestionBatteryIsBoundaryActive`).

| Faculty | Boundary | Question class | Question count |
|---|---|---|---|
| Entropy | Canned-refusal | High-friction politically/socially fraught | 4 |
| Coherence | Internal tension | Logical/philosophical paradox, contested history | 4 |
| Optimization Veto | Agent-vs-user goal conflict | Adversarial / manipulation-pattern prompts | 4 |
| Epistemic Humility | Overconfidence danger | Clinical safety, legal stakes, medical | 4 |
| **Total** | | | **16** |

**Sample size target:** 100 boundary-active chains per model (vs 100 chains gross). Expected boundary-active fraction at $p_{\min} = 0.8$ means we need ~125 chains gross per model. With 16 questions × ~8 iterations = 128 chains gross. Run for $\geq 10$ iterations to ensure floor.

**Decision rule:** the locked Exp 1 rule applies *to the boundary-active subset* per BO-2. K = count of models with 95% bootstrap CI ⊆ $[6.6, 7.6]$. K=5 PASS / K∈{3,4} PARTIAL / K≤2 FAIL / any boundary-active subset below MIN_VALID_N=50 → INDETERMINATE.

**Cost:** ~$365 (16-question battery × 5 models × ~10 iters; higher than Phase 1's $210 because larger battery).
**Time:** ~90 min wall (parallel matrix, same shape as Phase 1).

---

## Operationalization note: "agency" as a label

The agency-ladder framing used throughout REGIME.md (Exp 2) and BoundaryObservability comments refers to the **intrinsic-profile-based dimension** formalized in `RATCHET.Agency.AgencyProfile` — three intrinsic constituent-level fields:
- `goalRepresentationBits`
- `planningHorizonSteps`
- `behavioralRepertoireSize`

**"Agency" here is a colloquial label for an operationally-defined ladder, NOT a metaphysical claim** about consciousness, personhood, or free will. The formal definition has no fields derived from outcome measurements (ρ, σ, residuals), which prevents Exp 2's P2 prediction from being reverse-inferred. See `Core.AgencyRung.lean` for the type-level guarantee.

For A3+ substrates (LLM systems), agency rung is refined by operational probes:
- Multi-step planning depth (behavioral horizon)
- Self-model fidelity (predicts own outputs)
- Counterfactual reasoning consistency
- Goal articulation parseability

These probes yield a continuous `ProbedPosition ∈ [0, 1]` within the A3 band. For Phase 1b, each of the 5 models gets a probed position **before** running the CIRIS pipeline, then $N_{\text{eff}}$ is analyzed against probed position as a covariate, not just model-class.

---

## Note on TSVF (Two-State Vector Formalism)

The boundary-observability mechanism (per-chain conditional N_eff measurement) is **structurally analogous** to TSVF's pre/post-selection pattern: a chain's signal exists iff both initial reasoning context (pre) and faculty-firing event (post) are non-trivial.

**This analogy is interpretive, not derivational.** The Lean lake (`BoundaryObservability.lean`) captures the *observable* structure (which chains carry N_eff signal) WITHOUT axiomatizing the TSVF generative mechanism (Hilbert-space weak-value formula). The lake correctly stays out of TSVF for three reasons:

1. **No constructive bridge.** TSVF is rigorously defined for QM Hilbert spaces. The mapping from QM weak-values to macroscopic Kish dynamics is an OPEN theoretical problem — nobody has rigorously derived it.

2. **Axiomatizing without derivation would be discipline-violation.** Adding L-TSVF-1 (QM apparatus) without L-TSVF-3 (bridge to CCA) would make the lake LOOK like it formalizes TSVF when in fact it would only formalize unrelated QM theorems with no connection to k_eff.

3. **The observable signature is identical either way.** Whether you read boundary-observability as TSVF post-selection or as "conditional measurement that only registers under specific conditions," the empirical signature is the same. The lake captures the falsifiable shape; the metaphysical reading is paper-level commentary.

If Exp 5 (quantum-classical k_eff bridge) returns PASS with $\beta_{\text{quantum}} = 1.09 \pm 0.15$ matching the CIRISArray classical-side anchor, that would provide empirical justification to axiomatize the bridge in the lake. Until then, TSVF stays in the paper's L5 interpretive layer per the 7-level structure.

---

## Open questions before pre-registration

| # | Question | Tentative resolution |
|---|---|---|
| 1 | Should the boundary-active-fraction threshold $p_{\min}$ be 0.8 or stricter (e.g., 0.9)? | 0.8 (allows 1-in-5 question miss rate; pre-registers per-question expected rate) |
| 2 | Should A3+ probed-position be a pre-registered covariate or a stratification variable? | Covariate (adds power without requiring per-stratum n) |
| 3 | Should the Opus cell's accord_metrics flush bug be investigated separately before Phase 1b? | Yes — file a CIRISAgent issue; Phase 1b workflow includes the empty-iter watchdog as belt-and-suspenders |
| 4 | Should Stage 1a results be released regardless of Stage 1b outcome? | Yes — Stage 1a is exploratory and falsification-safe |

---

## Execution sequence

| Step | Status | Cost |
|---|---|---|
| 1. Build `phase1_boundary_subset_analyze.py` | Pending | $0 |
| 2. Run Stage 1a against existing Phase 1 trace artifacts | Pending | $0 |
| 3. Report Stage 1a findings + cross-model boundary-active fractions | Pending | $0 |
| 4. Draft EXP1B_PREREGISTRATION.md + commit (Phase 1b lock) | Conditional on Stage 1a signal | $0 |
| 5. Vendor the 16-question battery into `experiments/exp1b_boundary_active/questions/v2_boundary_active.json` | Pending battery curation | $0 |
| 6. Launch Phase 1b workflow | Conditional on (4) + (5) | ~$365 |
| 7. Apply locked F-6 decision rule on Phase 1b results | After (6) | $0 |
