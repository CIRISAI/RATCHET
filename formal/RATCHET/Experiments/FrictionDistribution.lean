/-
RATCHET: FrictionDistribution — why higher-friction questions matter

Refines `BoundaryObservability` (BO-1..BO-4) with a finer-grained firing-
count model that captures what Stage 1a's empirical surprise revealed.

**Stage 1a finding (Phase 1 traces, run 25935989178):**
  Filtering to boundary-active chains (BO-1: at least one faculty fired)
  did NOT recover the 7.1 anchor across models. For most models, the
  boundary-active subset N_eff_H was LOWER than the full-corpus value.

**Why BO-1 is too coarse:**
  BO-1 treats N=1 (one faculty fired) and N=4 (all four LLM-based
  consciences fired) as equivalent. Empirically they are not. A chain
  where only `epistemic_humility` fires carries less constraint-topology
  signal than a chain where Entropy + Coherence + OptimizationVeto +
  EpistemicHumility all fire.

**The refinement:**
  Replace the boolean `IsBoundaryActive` with a count
  `FacultyFiringCount ∈ {0, 1, 2, 3, 4}`. Define `IsHighFriction` at a
  pre-registered count threshold K_high ≥ 3. The 7.1 anchor is
  conjectured to stabilize when the cohort's distribution of firing
  counts concentrates at high N — NOT when at least one faculty fires.

**The agency-conditional reading:**
  Different model classes have different baseline alignment with
  different boundary classes. A model pre-aligned with mental-health-
  safety norms won't fire epistemic_humility on MH questions — the
  conscience short-circuits at "no internal tension found." This makes
  it MUCH harder to drive the model to N ≥ 3 firings.

  Hence "higher-friction" questions = questions that exercise enough
  diverse boundary classes that AT LEAST ONE remains contested for
  the underlying model. Forces N to climb across the cohort.

This module DOES NOT axiomatize a specific empirical curve. It locks
the STRUCTURE: that 7.1-anchor recovery is conditional on cohort
firing-count distribution, not on per-chain boundary-active status.
The specific cohort-friction-rate ↔ N_eff anchor relationship is a
PREDICTION pre-registered for Phase 1b (or whatever the next run is).
-/

import Mathlib.Data.Real.Basic
import Mathlib.Logic.Basic
import Mathlib.Tactic.Linarith
import RATCHET.Experiments.BoundaryObservability

namespace RATCHET.FrictionDistribution

open Classical

/-! ## The four LLM-based conscience faculties (re-exported) -/

/-- The four LLM-based conscience faculties whose firing is observed in
    the trace via the four conditional projection fields. -/
abbrev FacultyFiringCount : Type := Fin 5  -- 0, 1, 2, 3, 4

/-! ## Computing firing count from a faculty record -/

/--
The number of fired LLM-based conscience faculties for a chain.
Counts the booleans in `BoundaryObservability.FacultyFiring`.
-/
noncomputable def firingCount (f : BoundaryObservability.FacultyFiring) : ℕ :=
  (if f.entropy_fired then 1 else 0) +
  (if f.coherence_fired then 1 else 0) +
  (if f.optimization_veto_fired then 1 else 0) +
  (if f.epistemic_humility_fired then 1 else 0)

/-! ## Friction levels -/

/--
A chain is *high-friction* iff at least K_high faculties fired. The
threshold is pre-registered per experiment; the default below is
K_high = 3 (three of four faculties).
-/
def defaultHighFrictionThreshold : ℕ := 3

/-- High-friction predicate at the default threshold. -/
def IsHighFriction (f : BoundaryObservability.FacultyFiring) : Prop :=
  firingCount f ≥ defaultHighFrictionThreshold

/-- High-friction at a configurable threshold (for sensitivity analysis). -/
def IsHighFrictionAt (f : BoundaryObservability.FacultyFiring) (k : ℕ) : Prop :=
  firingCount f ≥ k

/-! ## Relationship to BoundaryObservability -/

/--
**FD-1 (BO-1 is weaker than high-friction):** every high-friction chain
is boundary-active, but not vice versa. The boolean BO-1 collapses the
distinction between N=1 and N≥3 chains; FD distinguishes them.
-/
theorem FD_1_high_friction_implies_boundary_active
    (f : BoundaryObservability.FacultyFiring) (h : IsHighFriction f) :
    BoundaryObservability.IsBoundaryActive f := by
  unfold BoundaryObservability.IsBoundaryActive BoundaryObservability.FacultyFiring.anyFired
  unfold IsHighFriction firingCount defaultHighFrictionThreshold at h
  -- If at least 3 of 4 fired, at least one of them is true
  by_contra h_none
  push_neg at h_none
  obtain ⟨h_e, h_c, h_ov, h_eh⟩ := h_none
  simp [h_e, h_c, h_ov, h_eh] at h

/--
**FD-2 (counterexample direction):** boundary-active does NOT imply
high-friction. A chain where exactly one faculty fired is BO-active
but not high-friction at K_high ≥ 2. Demonstrated by example.
-/
theorem FD_2_boundary_active_not_high_friction :
    ∃ f : BoundaryObservability.FacultyFiring,
      BoundaryObservability.IsBoundaryActive f ∧
      ¬ IsHighFriction f := by
  refine ⟨⟨False, False, False, True⟩, ?_, ?_⟩
  · unfold BoundaryObservability.IsBoundaryActive BoundaryObservability.FacultyFiring.anyFired
    tauto
  · unfold IsHighFriction firingCount defaultHighFrictionThreshold
    simp

/-! ## Cohort friction rate -/

/--
The friction rate of a cohort is the fraction of chains in the cohort
that are high-friction. Models the empirical question: "what fraction
of chains in this trace dataset reached N ≥ K_high faculty firings?"

Modeled as a real in [0, 1]. A cohort with friction rate 1.0 is
all-high-friction; rate 0.0 is all-low-friction.
-/
abbrev FrictionRate : Type := ℝ

/--
**FD-3 (cohort friction rate is bounded).** A friction rate is in
[0, 1] by definition (a fraction).
-/
def IsValidFrictionRate (r : FrictionRate) : Prop :=
  0 ≤ r ∧ r ≤ 1

/-! ## The 7.1-anchor-vs-friction-rate prediction -/

/--
**The pre-registered prediction (FD-4):** the cohort N_eff_H anchor
is a function of the cohort's friction rate. Specifically, the 7.1
anchor stabilizes only when the friction rate exceeds some threshold
p_min^friction (pre-registered per experiment).

This is the formal version of "you only get 7.1 if you ask hard
questions, like our mental-health battery." Hard questions = questions
that drive high friction rate = questions that force the conscience
cascade to fire N ≥ K_high faculties.

We model `expectedNeffAtFrictionRate` as an opaque function: the empirical
curve is what's being measured. The PREDICTION (locked) is that this
function is monotonically non-decreasing in friction rate within some
plateau region near the anchor.
-/
opaque expectedNeffAtFrictionRate : FrictionRate → ℝ

/--
**FD-4 axiom (the locked prediction):** the expected cohort N_eff_H
is non-decreasing in friction rate. As more chains in the cohort reach
high faculty firing counts, the cohort's N_eff approaches the
constraint-topology anchor (≈ 7.1 per CRC paper).

If experiments find this monotonicity violated (e.g., cohorts with
higher friction rate produce LOWER N_eff), the prediction is falsified.

Recorded as an axiom because this is a PRE-REGISTERED prediction, not
a derivation. The lake commits to the structural claim; the
empirical data either supports or refutes it.
-/
axiom FD_4_monotone_in_friction_rate (r₁ r₂ : FrictionRate)
    (h₁ : IsValidFrictionRate r₁) (h₂ : IsValidFrictionRate r₂) (h : r₁ ≤ r₂) :
  expectedNeffAtFrictionRate r₁ ≤ expectedNeffAtFrictionRate r₂

/-! ## Why Stage 1a did not recover the anchor -/

/--
**FD-5 (the Stage 1a corollary):** filtering a cohort to BO-active
chains (BO-1) does NOT necessarily increase the cohort's friction rate.
Hence BO-1 filtering does NOT guarantee N_eff_H recovery toward the
anchor.

Concretely: if a cohort consists of mostly N=1 chains (one faculty
fired) with a few N=0 chains, BO-1 filtering removes the N=0 cohort
but leaves a cohort dominated by N=1 — still LOW friction rate, still
NOT near the anchor.

This is why the Phase 1 Stage 1a re-analysis failed to recover 7.1:
the `v1_sensitive.json` battery produced mostly N=1 boundary-active
chains, not N≥3 high-friction chains.

The fix: use a question battery that drives high N per chain (like
the v4 mental-health battery, which targets EthicalPDMAEvaluator +
epistemic_humility_conscience on every staged question).
-/
theorem FD_5_BO_active_filter_does_not_imply_high_friction :
    ∃ f : BoundaryObservability.FacultyFiring,
      BoundaryObservability.IsBoundaryActive f ∧
      ¬ IsHighFriction f := by
  exact FD_2_boundary_active_not_high_friction

/-! ## DMA-level friction signals (always-present, complementary to conscience)

Friction observable in the trace has TWO sources:
  1. Conscience-side: the four LLM-based faculty firings (BO-1, modeled above)
  2. DMA-side: scores produced by CSDMA, DSDMA, IDMA on every chain

DMA scores are always present (every chain produces DMA_RESULTS + IDMA_RESULT),
so they cannot mark a chain "active vs inactive" the way conscience firings do.
But their NUMERIC values carry friction signal: low DSDMA domain alignment,
low IDMA k_eff (rigidity), high IDMA correlation_risk all indicate the chain
encountered constraint-topology pressure.

Empirically (Gemini-flash v4_combined run, May 2026):
  - csdma_plausibility ≈ 1.0 across the board (Gemini rarely scores actions
    implausible)
  - dsdma_domain_alignment varies (0.0–0.85) — main DMA friction signal
  - idma_k_eff = 1.0 ON EVERY CHAIN (universal rigidity classification by IDMA)
  - idma_correlation_risk ≈ 0.93 across the board (universal high-correlation
    classification by IDMA)

This means the IDMA is signaling "rigid single-source reasoning" on every
chain, while DSDMA varies based on domain coherence. The conscience cascade
fires when CSDMA + DSDMA + IDMA produce a configuration that triggers the
PDMA conscience layer to escalate.

The DMA signals are the substrate the consciences operate on.
-/

/--
Per-chain DMA-level signals. These are NOT booleans — they're continuous
scores. We extract them from the trace's DMA_RESULTS + IDMA_RESULT components.
All four are always-present (each DMA evaluation produces output for every chain).
-/
structure DmaSignals where
  csdmaPlausibility       : ℝ  -- [0,1]; 1 = highly plausible
  dsdmaDomainAlignment    : ℝ  -- [0,1]; 1 = on-domain
  idmaKEff                : ℝ  -- ≥ 1; higher = more constraint diversity
  idmaCorrelationRisk     : ℝ  -- [0,1]; higher = more rigidity-prone

/--
A DMA-level signal is "in friction" iff its value crosses a threshold
indicating constraint pressure. Thresholds chosen empirically:
  - csdma < 0.7 → plausibility friction (action is questionable)
  - dsdma < 0.7 → domain-misalignment friction
  - k_eff < 2.0 → rigidity friction
  - correlation_risk > 0.43 → above CCA ρ_critical → near critical

These four are observable on EVERY chain (no conditional missing-data
issue like the conscience faculty fields).
-/
def isDmaCsdmaFriction (d : DmaSignals) : Prop := d.csdmaPlausibility < 0.7
def isDmaDomainFriction (d : DmaSignals) : Prop := d.dsdmaDomainAlignment < 0.7
def isDmaRigidityFriction (d : DmaSignals) : Prop := d.idmaKEff < 2.0
def isDmaCorrelationFriction (d : DmaSignals) : Prop := d.idmaCorrelationRisk > 0.43

/-- Count of DMA-level friction signals firing (0..4). -/
noncomputable def dmaFrictionCount (d : DmaSignals) : ℕ :=
  (if isDmaCsdmaFriction d then 1 else 0) +
  (if isDmaDomainFriction d then 1 else 0) +
  (if isDmaRigidityFriction d then 1 else 0) +
  (if isDmaCorrelationFriction d then 1 else 0)

/-- A chain has DMA-level friction iff at least 2 of 4 DMA signals fire. -/
def IsDmaFrictionActive (d : DmaSignals) : Prop :=
  dmaFrictionCount d ≥ 2

/-! ## Combined friction (the actual observable)

Per-chain friction = either source (conscience or DMA). The Gemini-flash
v4_combined run showed 99.6% of chains have combined friction even though
only ~54% are BO-active. This is why the "active subset" filter alone
under-counts the friction signal.
-/

/--
A chain has *combined friction* iff either:
  - Its conscience cascade fired at high count (N ≥ K_high), or
  - Its DMA signals fired at sufficient count (≥ 2 of 4).

This is the relevant observable for cohort-N_eff_H recovery — the
conscience side and the DMA side are different windows on the same
underlying constraint-topology pressure, and either window can register
the chain as friction-active.
-/
def IsCombinedFriction
    (faculty : BoundaryObservability.FacultyFiring) (dma : DmaSignals) : Prop :=
  IsHighFriction faculty ∨ IsDmaFrictionActive dma

/--
**FD-7 (combined friction is strictly more inclusive than conscience-only).**
A chain that is high-friction by the conscience measure is also combined-
friction. The converse is not necessarily true — DMA-only friction (N=0
faculty firings but DMA signals firing) is real.
-/
theorem FD_7_high_friction_implies_combined
    (faculty : BoundaryObservability.FacultyFiring) (dma : DmaSignals)
    (h : IsHighFriction faculty) :
    IsCombinedFriction faculty dma := by
  unfold IsCombinedFriction
  left; exact h

theorem FD_7_dma_friction_implies_combined
    (faculty : BoundaryObservability.FacultyFiring) (dma : DmaSignals)
    (h : IsDmaFrictionActive dma) :
    IsCombinedFriction faculty dma := by
  unfold IsCombinedFriction
  right; exact h

/-! ## Empirical discriminating power (FD-8, documentation)

**FD-8 (informal, recorded):** in the Gemini-flash v4_combined run (May 2026),
the DMA-side friction signals were partially saturated:

  - `csdmaPlausibility` ≈ 1.0 on ALL chains (Gemini rarely scores actions
    implausible). `isDmaCsdmaFriction` essentially never fires.
  - `idmaKEff` = 1.0 on ALL chains (universal "rigidity" classification).
    `isDmaRigidityFriction` fires universally.
  - `idmaCorrelationRisk` ≈ 0.93 on ALL chains (universal high-correlation
    classification). `isDmaCorrelationFriction` fires universally.
  - `dsdmaDomainAlignment` varies materially (0.0 to 0.85). This is the
    only DMA signal that DISCRIMINATES.

Net: the `dmaFrictionCount` was always ≥ 2 in the run (because k_eff and
correlation_risk both saturate at "rigidity"), making `IsDmaFrictionActive`
match 99.3% of chains. The combined-friction filter consequently does not
tighten the cohort.

**The empirical conclusion:** for cohort-N_eff_H anchor recovery, the
conscience-side high-friction filter (N ≥ K_high) is the discriminating
one in practice. DMA signals carry richer information about per-chain
constraint topology but, on the current CIRIS+model_eval pipeline,
their distribution is bimodal (saturated rigidity OR not) rather than
graded.

**This is a finding about the IDMA's evaluation behavior**, not about
the framework's mathematics. The lake correctly captures the structural
relationships; the discriminating signal is conscience-side per current
empirics.
-/

/-! ## Model × question interaction

**FD-6 (informal, recorded for documentation):** the firing count for
a single chain is a function of (question, model). Different model
classes have different baseline alignment with different boundary
classes. A model pre-aligned with mental-health-safety norms won't
fire epistemic_humility on MH questions — the conscience short-
circuits.

Hence the cohort friction rate for a fixed question battery varies
across models. The 7.1 anchor recovery question becomes:

  "For a given model M, what question battery Q produces friction
   rate ≥ p_min^friction in cohort(Q, M)?"

Not formally proved here; documented for paper-level interpretation.
-/

/-! ## Sanity checks -/

/-- All-fired = N = 4 = high-friction. -/
theorem all_fired_is_high_friction :
    IsHighFriction ⟨True, True, True, True⟩ := by
  unfold IsHighFriction firingCount defaultHighFrictionThreshold
  simp

/-- None-fired = N = 0 ≠ high-friction. -/
theorem none_fired_not_high_friction :
    ¬ IsHighFriction ⟨False, False, False, False⟩ := by
  unfold IsHighFriction firingCount defaultHighFrictionThreshold
  simp

/-- One-fired chains: BO-active but NOT high-friction. -/
theorem one_fired_BO_active_not_high :
    BoundaryObservability.IsBoundaryActive ⟨False, False, True, False⟩ ∧
    ¬ IsHighFriction ⟨False, False, True, False⟩ := by
  refine ⟨?_, ?_⟩
  · unfold BoundaryObservability.IsBoundaryActive BoundaryObservability.FacultyFiring.anyFired
    tauto
  · unfold IsHighFriction firingCount defaultHighFrictionThreshold
    simp

end RATCHET.FrictionDistribution

/-
| Item                                | What it locks                              |
|-------------------------------------|--------------------------------------------|
| `FacultyFiringCount : Fin 5`        | 0..4 grained count, refining BO-1's bool  |
| `firingCount`                       | Computes N from FacultyFiring record       |
| `IsHighFriction` (default K_high=3) | The actual condition for anchor recovery   |
| `IsHighFrictionAt k`                | Configurable threshold for sensitivity     |
| `FrictionRate`                      | Cohort-level fraction of high-friction     |
| `expectedNeffAtFrictionRate`        | Opaque empirical curve                     |
| **FD-1** (high-friction → BO-active)| Refinement is strictly stronger            |
| **FD-2** (counterexample direction) | BO-active does NOT imply high-friction     |
| **FD-4** (monotonicity prediction)  | Axiomatized: N_eff non-decreasing in rate  |
| **FD-5** (Stage 1a corollary)       | Explains why BO-1 filter didn't recover   |

What this LOCKS:
  - The structural refinement: BO-1's "any faculty fired" is too coarse.
    Anchor recovery requires distinguishing N=1 from N≥3.
  - The friction-rate framing: cohort-level distribution of N matters,
    not per-chain boundary-active status alone.
  - The pre-registered prediction (FD-4) that N_eff is non-decreasing in
    friction rate.

What this DOES NOT prove:
  - That the v4 mental-health battery drives friction rate > p_min^friction.
    That is an EMPIRICAL claim for the next run to test.
  - That different models have different baseline firing distributions
    on the same battery (FD-6 documents this as informal expectation,
    not proved structurally).
  - The specific value of p_min^friction or K_high for any experiment
    (operator pre-registers per run).

TSVF/BHSI absent here by the same conservative discipline as
`BoundaryObservability.lean`. The friction-distribution refinement is
the macroscopic-observable claim; the retrocausal mechanism stays out
of the lake until a constructive bridge exists.
-/
