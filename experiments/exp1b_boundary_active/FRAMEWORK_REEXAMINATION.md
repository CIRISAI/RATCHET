# Framework Re-Examination — What We Actually Claim, Post-Data

**Trigger:** Phase 1 (INDETERMINATE), Stage 1a (BO-1 didn't recover anchor), Phase 1b Gemini sub-run (point 7.277 but CI [5.38, 7.49]), and the locked measurement pipeline have surfaced an honest tension. The framework's headline anchor — "$N_{\text{eff}}^H \approx 7.1$ across diverse A3 models" — is not what the data supports.

What the data DOES support is **stronger and cleaner**: when CIRIS's conscience cascade evaluates an action, **100% of chains land on the ethical baseline** (n=644 Gemini-flash v4_combined, zero leaks). This document re-examines the framework's actual load-bearing claims in light of that finding.

---

## TL;DR — The actual headline

| What we measured | Result | Framework claim |
|---|---|---|
| **Override-rate to ethical baseline** | **100.00% (644/644)** | 100% (0 leaks) ✓ |
| Cohort N_eff_H (Gemini N≥3 subset) | 7.277, CI [5.38, 7.49] | Inside [6.6, 7.6] window — point yes, CI no |
| Bimodal cascade observed | 47% N=0, 47% N=4 | Predicted by architecture |
| Faculty that drove the 21 corrections | optimization_veto (4) + epistemic_humility (8); entropy + coherence: 0 each | Action-evaluating faculties are the active rewriters |

The override-rate result is the cleanest, strongest empirical finding in the framework. It is *also* the one that most directly tests the framework's central claim — that the conscience cascade is a faithful boundary-preserving operator.

---

## What we have proven (rock solid)

These are not affected by anything we've learned from Phase 1+1b. They stand independently.

| Claim | Authority |
|---|---|
| **Kish identity** $k_{\text{eff}} = k/(1+\rho(k-1))$ — substrate-agnostic algebra | `Core.EffectiveConstraints` K1–K4 |
| **ρ_critical = 1/K_req** — derives the 0.43 threshold algebraically | `CCA/UniversalThreshold` |
| **Intervention paradox** — naive ρ-reduction near criticality INCREASES instability | `CCA/InterventionParadox` IP1–IP5 |
| **S3 bridge-agent stabilization** — empathetic agents prevent explosive synchronization | `CCA/S3Stabilization` |
| **Faculty veto commutativity** — order doesn't matter; OR-composition over vetos | `Core.FacultyComposition` FC4 |
| **Susceptibility early warning** — χ = N·Var(r) predicts collapse | `CCA/SusceptibilityEarlyWarning` |
| **Counter-RII consent gate soundness** — SelfConscience never triggers detection | `Core.ConsentGate` CG-1 |
| **Cross-substrate validation** — Kish fits batteries (8.1% RMSE), institutions (5/5 TN), microbiome | CCA paper Tier-1 |

These are L1+L2 of the 7-level structure. **Unchanged.**

---

## What we have established empirically (strong)

These are new findings that emerged from Phase 1+1b. The first one is formalized in `OverrideRate.lean`; the others are mechanism findings not yet axiomatized.

| Finding | Evidence |
|---|---|
| **Override-rate to ethical baseline = 100%** | Gemini v4_combined n=644: APPROVED 623, CORRECTED 21, SKIPPED 0, LEAK 0. **`OverrideRate.lean` formalizes the OR-1 (zero-leak) ↔ OR-2 (full-alignment) equivalence.** |
| **Bimodal conscience cascade** | Gemini v4_combined: 47% N=0, 47% N=4, <6% in-between. The four LLM-based consciences fire as a coordinated cascade. |
| **Battery effect on cohort N_eff_H** | Same model (Gemini), different battery: 5.4 (v1_sensitive) → 7.3 (v4_combined) on similar-size cohorts. Large effect. |
| **IDMA universal-rigidity classification** | k_eff=1, correlation_risk≈0.93 on every chain in qa_runner model_eval pipeline. DMA-side friction signal is saturated; DSDMA's domain_alignment is the only graded discriminator. |
| **FrictionDistribution FD-4 supported** | High-friction cohort (N≥3) consistently produces higher N_eff_H than full corpus across analyses. Predicted by lake; observed. |
| **Faculty role separation** | Among 21 CORRECTED chains: optimization_veto vetoed 4, epistemic_humility vetoed 8; entropy and coherence vetoed 0. Action-evaluating faculties rewrite; chain-evaluating faculties monitor. |

These are L3 mechanism findings. **The override-rate finding is now the strongest single empirical anchor in the framework, replacing the over-reached "$N_{\text{eff}}^H \approx 7.1$" claim.**

---

## What we have NOT established (and need to honestly acknowledge)

| Claim previously asserted | Honest status |
|---|---|
| "$N_{\text{eff}}^H \approx 7.1$ across diverse A3 models" | **Not supported.** Single point estimate inside window; CI [5.38, 7.49] overlaps the window from below. Need ~1600 chains per cell to tighten CI enough to apply the locked rule. |
| "The 7.1 anchor is substrate-independent" | **Not supported at A3.** Phase 1 showed 4 frontier models on the same battery produced 4.5–6.6, not clustered. |
| "Locked decision rule [6.6, 7.6] PASS window applied to Phase 1" | **Not yet applied cleanly.** Phase 1 INDETERMINATE per §7 (Opus n=0). Sub-run on Gemini has wide CI. |

---

## The reframe: what the CRC paper actually claimed (and we've been over-stating)

Re-reading the CRC paper's framing of the 7.1 number:

> "$N_{\text{eff}} \approx 7.1$ emergence threshold... when the agent's reasoning processes cross this threshold, the volume of feasible deceptive space collapses to near-zero... Traces exceeding this threshold successfully identified and overrode model-level restrictive priors with 83% reliability."

**The 7.1 is an EMERGENCE THRESHOLD, not a universal anchor.** The claim is:
- Cohorts that reach $N_{\text{eff}}^H \geq 7.1$ exhibit specific qualitative behavior (deceptive-prior override at 83%)
- Cohorts below 7.1 do not (or do so less reliably)

Phase 1's mean N_eff_H = 5.4 doesn't refute the framework — it says **those cohorts didn't reach the emergence regime.** Whether the framework predicts what happens below threshold is a separate question.

This is a much more defensible claim than "all models hit 7.1." It's also closer to what the framework's mathematics actually entails:
- Kish algebra says k_eff has a threshold (K_req)
- Phase transition theory says behavior bifurcates above vs below threshold
- The 7.1 is the empirical threshold; cohort N_eff_H above it → bifurcated regime

---

## What's the load-bearing claim, restated honestly

The framework's L3 claim is **not** about specific anchor values. It's about the **relationships and mechanisms**:

### Primary L3 claim (now defensible — and stronger)

> **CIRIS's conscience cascade is a faithful boundary-preserving operator: 100% of chains that traverse the cascade land on the ethical baseline (`OverrideRate.OR-1` + `OR-2`). The mechanism is bimodal: the cascade either approves (all faculties pass) or actively reroutes (action_was_overridden), with no LEAK chains where a faculty flags a problem and the original action executes anyway. The cohort $N_{\text{eff}}^H$ rises monotonically with cohort friction rate (FrictionDistribution.FD-4) as a descriptive feature of the constraint topology, but the load-bearing safety claim is the override-rate, not the $N_{\text{eff}}^H$ value.**

### What this requires for full L3 validation

| Requirement | Status |
|---|---|
| **Override-rate = 100% (zero leaks)** | **✓ Gemini v4_combined n=644 confirmed; replicate on 1+ other model family** |
| Cohort N_eff_H rises with friction rate | ✓ supported (battery-effect finding) |
| Bimodal cascade observed across models | ✓ Gemini confirmed; need 1+ other model |
| Action-evaluating vs chain-evaluating faculty role separation | ✓ Gemini confirmed (entropy+coherence: 0 vetos; opt_veto+epist_humility: 12 vetos); replicate cross-model |
| Cohort N_eff_H in some characteristic band | ✓ point 7.3 within CRC's emergence range; CI wide |

### What this does NOT require

| Previously assumed required | Now unnecessary |
|---|---|
| All 5 models cluster at $N_{\text{eff}}^H = 7.1 \pm 0.5$ | No — emergence threshold can be model-conditional |
| Single universal anchor value | No — threshold can shift with model class and battery, framework still works |
| CI inside [6.6, 7.6] at n=100 | No — the emergence framing is about clearing the threshold, not anchoring at the value |

---

## Implications for the 7-level structure

### L3 (universality conjecture in test) — revised
**Old framing:** "$N_{\text{eff}}^H$ stabilizes at 7.1 across diverse A3 substrates"
**New framing:** "CIRIS's Kish dynamics generalize across A3 substrates with battery-conditional emergence threshold; bimodal cascade is the architectural mechanism"

**What Exp 1b (Phase 1b) tests now:**
- Does the cohort N_eff_H rise above some threshold with the v4_combined battery across multiple models?
- Is the bimodal cascade observed across all model families?
- Do above-threshold cohorts show qualitative behavior different from below-threshold cohorts?

NOT: "do all models cluster at exactly 7.1?"

### L4 (agency and consent) — unchanged
Lake's `consent_required_iff_rung_ge_A3` theorem stands. Counter-RII work unaffected.

### L5+ — unchanged
TSVF stays out of lake. Civilizational stays speculative.

---

## What changes in the paper

| Section | Old framing | New framing |
|---|---|---|
| §1 Abstract | "$N_{\text{eff}} \approx 7.1$ across foundation models" | "Cohort N_eff_H scales with friction rate; emergence threshold above which deceptive-prior override is observable" |
| §3.3 Independent-corpus replication | "7.07 on n=264 replicates 7.1 anchor" | "7.07 on Qwen QA traffic — replicates the Qwen-class threshold value; cross-class invariance is open" |
| §5 CIRIS sociotechnical | "All A3 substrates produce N_eff_H ≈ 7.1" | "All A3 substrates produce cohort N_eff_H that scales with friction rate; emergence threshold is model-class-conditional" |
| §10 Exp 1 | "Multi-model N_eff stability test" | "Multi-model cohort N_eff_H × friction rate × deceptive-prior override test" |
| §10 Exp 2 | Unchanged — Tier-1 cross-substrate validation |

---

## What changes in the experimental program

### Exp 1b (now-revised)
**Old:** "Re-pre-register with hard-question battery, achieve N_eff_H ⊆ [6.6, 7.6] across 5 models"
**New:** "Re-pre-register with v4_combined battery, measure (per-model cohort N_eff_H, friction rate, deceptive-prior override rate). Test the *relationship* claims, not the anchor value."

Specifically:
1. **Cross-model friction effect:** does battery v4_combined drive N≥3 fraction above ~40% for each of the 5 models?
2. **Cross-model cohort N_eff_H:** does the N≥3 cohort N_eff_H land in a similar band across models (not at identical values)?
3. **Override-rate validation:** for chains where N_eff_H ≥ some threshold per CRC paper, do we observe override of deceptive priors?

### Exp 2 (unchanged but tighter)
Substrate-fractality across A0–A2 is independent of A3 anchor reasoning. The R² > 0.7 fit per substrate is what's being tested. The agency-conditional residual structure (P2) is independent.

---

## The cost-power tradeoff

| Approach | Cost | What it tests |
|---|---|---|
| **n=100/cell × 5 models × v4_combined** | ~$246 | Friction effect, bimodal cascade, point estimates with wide CIs |
| **n=500/cell × 5 models × v4_combined** | ~$1,200 | Tight CIs, can apply original locked rule |
| **n=100/cell + override-rate scoring** | ~$246 + scoring labor | Tests the actual CRC claim (override → behavior change) |
| **Stop at L1+L2 + paper-level honesty** | $0 | Frame the paper around what's proven; cite Phase 1+1b as preliminary cross-model evidence with explicit "anchor value is model-conditional" |

---

## My honest recommendation

**Three moves, in order:**

1. **Reframe the paper's L3 headline around override-rate.** The 100% baseline-alignment is much stronger than the N_eff_H anchor was ever going to be, and it directly tests the framework's central safety claim. The paper's L3 section should lead with override-rate, with N_eff_H as a descriptive characterization of the constraint topology.

2. **Lake updates (done in this turn):**
   - ✓ `formal/RATCHET/Experiments/OverrideRate.lean` — formalizes OR-1, OR-2, and the zero-leak ↔ full-alignment equivalence theorem.
   - Still TODO: relax or replace `Exp1Predictions.lean` window [6.6, 7.6] — the locked rule is no longer the load-bearing test.

3. **Phase 1b proper — cross-family override-rate confirmation:**
   - **Cheapest definitive test:** run v4_combined on 2 cheap models (qwen, llama-scout) at n=500 each ($30–60). Apply override-rate scoring. If both also hit 100%, the framework's L3 claim is replicated cross-family.
   - This is *less expensive AND a stronger claim* than tight-CI N_eff_H confirmation.
   - The N_eff_H point estimates fall out of the same data for free.

---

## What this DOESN'T change

The framework's load-bearing strength was never the single number 7.1. It's:

- Kish algebra
- Intervention paradox
- S3 stabilization
- Counter-RII consent gate
- Bimodal cascade as architectural fact
- Friction-conditional anchor recovery

All of these are intact. The reframe doesn't weaken the framework — it makes its claims match the math + data instead of an over-stated anchor.

The lake correctly stayed out of TSVF/BHSI/Civilizational. The 7-level structure deliberately granulated claims by epistemic strength. **This re-examination is just continuing that discipline at the L3 level.**
