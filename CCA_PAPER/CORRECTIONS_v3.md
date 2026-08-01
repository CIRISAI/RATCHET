# Corrections to *Coherence Collapse Analysis* v3

**Corrects:** Zenodo record 18217688 (*Coherence Collapse Analysis*, v3)
**Status:** Prepared for deposit — awaiting two blocking decisions (see Actions 10–11)
**Date:** 2026-07-31
**Machine witness:** [`corrections_witness.py`](corrections_witness.py) (sympy; reproduces every defect below, exit 0)
**Tracking issue:** RATCHET#11

---

## What this document is

An external review of the Constitution and CCA v3 arrived 2026-07-30. Every claim in it
was re-derived here against the primary artifact (`coherence_collapse_analysis.tex` and
the built v3 PDF) before being accepted — received numbers are not measured numbers. All
five reported defects are **confirmed**.

A subsequent audit (2026-07-31) across three independent lines — the cross-domain table,
the institutional validation, and a whole-paper falsifiability sweep — surfaced **thirteen
further defects**, for **eighteen total**. Each was re-verified at the primary artifact
before being recorded; where a claim was not re-verified it is marked as such in its
section. The second wave is the more serious of the two: the first found local errors, the
second found that **all three of the paper's empirical legs fail**.

The v3 record is **not withdrawn and must not be deleted.** It is cited by the CIRIS
Constitution; a dead DOI reads as concealment, a corrected one reads as scholarship. This
note is the correction, published alongside.

One defect — C-1 — is load-bearing: it inverts a stability verdict, and the paper's
central theorem is wrong in the **permissive** direction, meaning it certifies as stable
systems that are in fact collapsing. It is reported here as loudly as any survival.

---

## Summary

| ID | Location (as published) | Defect | Severity |
|---|---|---|---|
| **C-1** | Thm 2.3, Eq. (6), p. 8 | Stability criterion does not follow from its own derivation; too permissive by the design effect | **Critical** |
| C-2 | Table 8, Venezuela row | Arithmetic wrong; inputs outside the formula's domain | Major |
| C-3 | Table 8 vs §11, Turkey | Table and text contradict; row is degenerate and carries no information | Major |
| C-4 | §11, §14 | Validation scoring counts two backsliding cases as successes; false positives rewritten as successes | **Critical** |
| C-5 | §10.2 | "Kish formula validation, r = 1.000" tests an identity against itself | Major |
| **C-6** | Defense Function, Eq. (4), p. 8 | Defense function `J` carries a withdrawn `(1−ρ)` factor; conflicts with CC 6.2.2 | Major |
| **C-7** | Cor. 2.4 | Invalid under the paper's own criterion; rescued by the C-1 correction | Moderate |
| **C-8** | §11 / `institutional.py:292` | **The institutional classifier never consults `k_eff`.** Collapse rule is `σ<0.2 or f>0.8` | **Critical** |
| **C-9** | §11 scoring | Honest re-score is 2/5, not 5/5; MCC −0.15, permutation p=0.87 — below chance | **Critical** |
| **C-10** | §11 timing | "7.6 years early" is an initialization artifact; the one non-artifact flag is 6 years *late* | Major |
| **C-11** | §10.2, C1 | **Both** GPU Kish results are identity checks — `r=1.000` and `R²=0.798` alike | **Critical** |
| **C-12** | §8.6 | "Three CCA invariants validated across all domains" is a hardcoded `print()` | **Critical** |
| **C-13** | Exp 103 vs F2 / exp117 / F3 | Flagship "software induces collapse" — **replicated and withdrawn**: lockstep is a code artifact, barrier is real but unstable by 3× | **Critical — resolved** |
| **C-14** | F3 / §10.10 | "Corridor validated" — the rigidity arm was never measured (max ρ = 0.171 vs edge 0.43) | Major |
| **C-15** | Table 8 (all rows) | No row contains a measured correlation; formula inverts below `k=1` | Major |
| **C-16** | Abstract vs §5 | L-01's 40%/60% stated as proven; theorem gives existence only, body calls it illustrative | Major |
| **C-17** | Abstract | "Monte Carlo suite reproduces the corridor regime structure" — artifact does not exist | Major |
| **C-18** | `wgi_processed.csv` | 4933/4933 analyzable rows have `k<1` and `k_eff>k` — defect beyond the paper | **Critical** |

C-6 and C-7 were found during first-pass verification. **C-8 through C-18 come from a
second-wave audit (2026-07-31)** and are not in the external review. Every one was
independently re-verified at the primary artifact before being recorded here; where a
claim was *not* re-verified it is marked as such in its section.

---

## C-1 — Theorem 2.3's stability criterion does not follow from its derivation

**Published (Theorem 2.3, Eq. 6):**

> Stable ⟺ α / k_eff > d

**Correct, under the paper's own three assumptions:**

> Stable ⟺ α / k ≥ d

### Derivation

The paper assumes (i) `dk_eff/dt = α/(1+ρ(k−1))`, (ii) `dρ/dt = 0`, (iii) `dσ/dt = −dσ`,
with `λ` constant. Applying the product rule to `J = k_eff·(1−ρ)·λ·σ` and dividing by the
strictly positive prefactor `λ(1−ρ)σ`:

```
dJ/dt  ∝  (α − d·k) / (1 + ρ(k−1))
```

The sign is governed entirely by `(α − d·k)`. The boundary solves to `α = d·k`, i.e.
`α/k ≥ d`. Since `k_eff = k/(1+ρ(k−1))`, the published criterion is **too permissive by
exactly `1 + ρ(k−1)`** — the design-effect factor the paper's own thesis is about. The
error is therefore not incidental to the framework; it discards the very quantity the
paper exists to introduce.

### Concrete counterexample

| Quantity | Value |
|---|---|
| k, ρ, d | 100, 0.5, 0.1 |
| k_eff | 1.9802 |
| α | 5.0990 |
| α / k_eff | 2.5750 → published criterion: **STABLE** |
| dJ/dt | **−0.097 (strictly decreasing)** |

The published criterion certifies as stable a system whose defense function is collapsing.

### Also in the derivation

Equation (5) as printed carries algebraic debris that does not appear in the result: a
stray division by `(1−ρ)` and a factor `(1/σ)·σ`. These cancel and do not affect the
corrected criterion, but the printed line does not follow from the line above it.

### Inheritance and cross-references

- **Theorem 4.1 (Healthy Corridor)** states the corridor as `0.2 < ρ < ρ_crit` **and**
  `α/k_eff > d`. It inherits the error verbatim and must be restated with `α/k ≥ d`.
- The §9 discussion cites this result as "Theorem 2.4". In the published PDF, 2.4 is the
  *corollary* ("Static Systems Are Doomed"); the theorem is **2.3**. The citation is
  hardcoded rather than a `\ref` and is wrong.

### What survives

The qualitative claim survives intact and is arguably strengthened: **stability requires
constraint generation to outpace sustainability decay.** Only the threshold moves — and it
moves in the conservative direction, requiring `(1+ρ(k−1))×` more generation than
published. The correction is also **invariant to the C-6 J-form drift**: with or without
the `(1−ρ)` factor, the boundary is `α = d·k` (witness C-1b).

---

## C-2 — Table 8, Venezuela row is arithmetically wrong and out of domain

**Published:** `k = 0.667`, `ρ = 0.299`, `k_eff = 0.55`

**Correct arithmetic:** `k_eff = 0.667 / (1 + 0.299 × (0.667 − 1)) = 0.7408`

Two distinct problems:

1. **The printed value is wrong.** 0.55 does not follow from the printed inputs.
2. **The inputs are outside the formula's domain, so no value is right.** In Kish (1965),
   `k` is a cluster count with `k ≥ 1`. A value of 0.667 is a normalised index, not a
   count. For `k < 1` the formula returns `k_eff < 1`, violating the paper's own
   single-constraint floor (a system cannot have less effective diversity than one
   constraint).

**Disposition:** the row must be withdrawn, not recomputed. Reporting `k_eff = 0.7408`
would substitute a correct calculation on inapplicable inputs. If institutional data is to
be carried into this table, `k` must first be given a defensible count semantics.

---

## C-3 — Table 8 contradicts §11 on Turkey, and the row is degenerate

**Table 8:** Turkey at `ρ = 0.000`, interpretation "Uncorrelated constraints".
**§11:** "Rigidity phase: Turkey, Venezuela correctly classified as trending toward collapse."

Rigidity is the **high**-correlation regime. The table's `ρ = 0.000` is the chaos-side
extreme. The two statements cannot both be true.

Beyond the contradiction, the row carries no information: at `k = 1.000`, `k_eff = 1` for
**every** value of ρ (witness C-3). The Kish map is constant in ρ at `k = 1`, so the
reported `ρ = 0.000` is unconstrained by the reported `k_eff` — the row cannot support
either the table's reading or §11's.

---

## C-4 — Validation scoring counts backsliding cases as successes

**Published (§11):**

> **Healthy phase**: 5/5 stable democracies correctly classified (Germany, Canada,
> Australia, Poland, Hungary)

Hungary and Poland are the two canonical democratic-backsliding cases of the stated
2000–2024 window. V-Dem reclassified Hungary from democracy to **electoral autocracy in
2019**; Poland's judiciary reforms from 2015 are the standard companion case. Classifying
both as "healthy" is not a success; on the face of the paper's own framing it is two
misses reported as hits, and the "5/5" figure is not defensible.

**Compounding (§14):**

> "False positives" detected real fragility: Tunisia, Egypt, Zimbabwe were flagged and all
> experienced major upheaval. CCA identified structural weakness; the binary "collapse"
> definition was inappropriate for these cases.

The summary table (p. 3) reports these honestly as `3/13 false positives`. §14 then
reclassifies them as successes by revising the outcome definition after seeing the
results. **This is the sentence at which the framework stops being able to fail**, and it
is the most serious defect in the paper after C-1 — a framework that cannot fail cannot
validate.

**Disposition:** restate the institutional results with a pre-specified outcome definition
and report the score that definition yields, including misses. If Hungary and Poland are
scored as classified, they are misses. If the outcome definition is to be revised, it must
be revised and then re-run, never revised in the discussion of results already seen.

---

## C-5 — §10.2 "Kish formula validation, r = 1.000" is an identity check

The table reports `k_eff = k/(1+ρ(k−1))` validated at `r = 1.000` "across all test
conditions". Table 8's own caption concedes the point: *"k_eff is **computed** from
measured k and ρ — it is not independently observable."*

Computing a quantity from a formula and then correlating it against that same formula
returns `r = 1.000` by construction, for any formula whatsoever. This is an arithmetic
self-check, not evidence.

**Disposition:** relabel as an implementation check (confirming the code computes the
identity it claims to) and remove it from any list of empirical results. An actual
validation of the Kish relation requires an independently observed measure of effective
diversity to compare against — the paper does not have one, and should say so.

---

## C-6 — The defense function carries a withdrawn `(1−ρ)` factor *(not in the external review)*

**Published (Definition, Defense Function, Eq. 4):** `J = k_eff · (1 − ρ) · λ · σ`

**CIRIS Constitution CC 6.2.2:** `J = k_eff · λ_op · σ`, with an explicit drift note:

> An earlier draft multiplied by an additional `(1 − ρ̄)` factor. That double-counted
> correlation — `k_eff` already discounts it — and drove `J = 0` at `ρ̄ → 1`,
> contradicting the single-constraint floor. Corrected to the CCA-validated form above.

The Constitution therefore cites *this paper* as the authority for a form the paper does
not use. Two published artifacts disagree, and the paper carries the form the Constitution
has already withdrawn. The substantive objection is the Constitution's: `k_eff` already
contains the correlation discount, so the extra factor double-counts it and sends `J → 0`
at full correlation, contradicting the floor that a fully-correlated federation is *no
safer than a single validator, but never worse*.

**Disposition:** adopt `J = k_eff · λ · σ` and note the change. Per C-1b the stability
correction is unaffected either way, so this can be fixed without disturbing C-1.

---

## C-7 — Corollary 2.4 is invalid under the published criterion *(not in the external review)*

**Published (Corollary 2.4, "Static Systems Are Doomed"):**

> If α is constant and k (hence k_eff) grows over time, the ratio α/k_eff decreases
> monotonically. Eventually α/k_eff < d, violating stability.

The premise "hence k_eff grows" is false without bound. `k_eff` saturates at `1/ρ` as
`k → ∞` — the paper states this ceiling itself in the Möbius remark. So `α/k_eff`
converges to `α·ρ` from above rather than decreasing to zero, and if `α·ρ > d` the
published criterion is **never** violated no matter how large `k` becomes.

Worked case (`ρ = 0.5`, `d = 0.1`, `α = 1`): `α/k_eff` floors at **0.5**, permanently
above `d = 0.1`. The corollary's conclusion does not follow from its own criterion.

Under the **corrected** criterion the corollary is restored: `α/k → 0` as `k → ∞`
unconditionally, so stability is eventually violated for any constant α. The headline
claim — *static systems cannot maintain coherence indefinitely* — survives, but it
survives **only because of the C-1 correction**, and cannot be derived from the text as
published.

---

---

# Second-wave findings (2026-07-31)

The first wave (C-1 – C-7) treated the paper's defects as local: a wrong criterion, some
bad table rows, an identity check mislabelled. The second wave establishes something
structural. **All three of the paper's empirical legs fail, for three different reasons.**
Each finding below was re-verified at the primary artifact.

## C-8 — The institutional classifier never consults `k_eff` *(critical)*

`ratchet/engines/institutional.py:292-299`:

```python
if (self._state.sigma < self.params.collapse_threshold_sigma or
        self._state.f > self.params.collapse_threshold_f):
    self._collapsed = True
```

That is the entire collapse rule. In the step function (`:233-273`), `k` and `rho` are
evolved by country-independent drift plus noise — and then never read by anything. The
prediction reduces exactly to `flagged ⟺ σ₀ < 0.60 or f₀ > 0.70`, which reproduces the
engine's actual runs 14/14. Sweeping `(k, ρ)` across their full range at fixed `(σ₀, f₀)`
yields **one** distinct outcome across 400 combinations.

C-5 established that the Kish check is an identity test. This is stronger: in the paper's
only institutional decision, **the framework's central quantity is absent**. §11 cannot be
evidence for `k_eff` because `k_eff` is not consulted.

*Verified: collapse rule and step function read directly.*

## C-9 — Honest re-scoring: 5/5 becomes 2/5, below chance *(critical)*

Scored against a pre-registered outcome definition (V-Dem Regimes-of-the-World, ≥1
category decline sustained ≥3 years, 2000–2024; pre-registration sha256
`1d3b0c39…bfb4b1e`, timestamped before the scoring scripts ran):

| | Predicted collapse | Predicted stable |
|---|---|---|
| **Actual decline** | TP 4 | **FN 3** |
| **Actual stable** | FP 5 | TN 2 |

n=14. Accuracy 0.43, balanced accuracy 0.43, **MCC −0.15**, permutation null (20,000 draws)
**p = 0.87**, Fisher exact p = 1.00. Always-flag and never-flag both score 0.50; the
classifier does not reach either.

The three false negatives are **Hungary, Poland, and Canada**. V-Dem trajectories, read
directly from `v-dem-v15.parquet`:

```
Hungary  3333333333222222221111111   liberal → electoral democracy 2010 → electoral autocracy 2019
Poland   3333333333333333222222222   liberal → electoral democracy 2016
Germany  3333333333333333333333333   (true negative)
```

Scoring only the paper's own ten named countries: TP 2 / FP 3 / TN 2 / FN 3 — **the
claimed "5/5 stable democracies correctly classified" is 2/5.**

**The design was underpowered before data collection.** At n=14 with this base rate,
one-sided Fisher does not reach p<0.05 until the classifier is right on 12 of 14 *with
perfect sensitivity*. A flawless classifier at n=13 reaches only p=0.0006. No validation
claim was reachable in either direction.

**Threshold fragility.** Across six pre-registered definitions, MCC ranges −0.19 to +0.65.
Only the Polity5 definition beats chance (MCC +0.65, p=0.03) — and Polity5 records Hungary
and Poland at 10 for **every year 2000–2018**, making it structurally blind to precisely
the two cases the classifier gets wrong. It also ends in 2018 and cannot cover the stated
window. That is the definition the paper's numbers came from.

**Honest caveat, recorded rather than buried:** Canada scores positive because V-Dem v15
codes it 2 in 2020, 3 in 2021, 2 from 2022. Treating that as equivalent to Syria's descent
is a weakness of any categorical rule, and it counts *against* the classifier here. The
pre-registered 5-year-persistence variant drops it; the verdict does not change.

*Verified: Hungary/Poland/Germany/Canada trajectories, pre-registration hash and
timestamp ordering. Not re-verified: the permutation and Fisher computations.*

## C-10 — "7.6 years early" is an artifact *(major)*

Seven of nine flags fire in 2001 because `f₀ > 0.8` **already holds at initialization** —
the collapse condition is satisfied by the input data before any dynamics run. The reported
lead time is `(event year − 2001)`, a restatement of when the events happened. A null that
flags every country in 2001 achieves a *larger* mean lead (10.8y vs 7.8y) at sensitivity
1.0.

Turkey is the only flag via the σ pathway. It fires in **2019** — six years after its
regime transition and three years after the 2016 coup. "Turkey 2016: flagged within 3
years" is right in magnitude and **wrong in sign**, printed beside "7.6 years early."

## C-11 — Both GPU Kish results are identity checks *(critical)*

C-5 identified `r=1.000` (Exp 86). The C1 result `R²=0.798` (n=21), carried in the
contributions list as empirical validation on physical hardware, is the **same defect**.

In `CIRISArray/experiments/expC1_keff_heatmap.py`, both columns come from
`compute_keff_formula` on the same correlation matrix:

- "measured" (`:230`) = `mean_i f(ρ_i)` — identity applied per sensor, then averaged
- "predicted" (`:308`) = `f(mean ρ)` — identity applied to the average

`f` is convex, so Jensen's inequality *forces* the first to exceed the second at every
point. Verified in `expC1_results.json`: **21 of 21 deltas positive**, mean +0.756, minimum
+0.297 — the residual never crosses zero. A genuine measurement-vs-prediction comparison
scatters about zero. `R²=0.798` is the magnitude of a convexity gap, a property of the
arithmetic rather than of the hardware.

**There is no independent test of the Kish identity anywhere in the corpus.** A genuine one
requires an independently observed measure of effective diversity — variance of the array
mean, SNR gain from averaging, participation ratio, or detection latency.

**One genuine, unreported test does exist, of the identity's *precondition*.** `exp117`
compares spectral entropy from all 16 eigenvalues against the closed form predicted from
scalar ρ alone — distinct functionals over distinct inputs, free to disagree. It agrees to
0.8% at ρ=0.394 and diverges 17.6% and 70.8% at low ρ. **It both passes and fires:** the
equicorrelated assumption holds at high correlation and fails at low correlation. This is
the most honest result in the corpus and it is not in the paper. It should be promoted —
after receiving the same audit as Exp 86.

*Verified: both column derivations, and the 21/21 one-sided residual recomputed from
stored results.*

## C-12 — "Three invariants validated across all domains" computes nothing *(critical)*

§11's cross-domain invariant claim traces to
`simulation/real_data_validation_report.py:186-203`, which is a single triple-quoted
`print()` containing the word `VALIDATED` nine times. No data is read, nothing is compared,
no branch can produce different output.

This is more severe than C-5: an identity check at least computes something.

*Verified: source read directly.*

## C-13 — The flagship result is contradicted 3-to-1 *(critical, blocking)*

Exp 103 reports barrier sync driving ρ→0.90 and lockstep ρ→1.0, grounding the paper's
"software alone can induce collapse." Three later experiments measure the opposite:

| Experiment | Result |
|---|---|
| F2 | `independent_rho 0.171 → synchronized_rho 0.0495`; k_eff **rises** 9.48 → 15.63 |
| exp117 | baseline ρ 0.394 → barrier ρ 0.041 |
| F3 | `lockstep_rho = 0.0036` |

All three say coordination **decorrelates** these sensors. The paper prints Exp 103 and F2
in the same section and reconciles neither, so no reader can tell which the framework
stakes. F2's own recorded criteria are `rho_increased: False`, `response_increased: False`
(p≈0.38), yet it is written up as a clean finding.

### RESOLVED 2026-07-31 by replication on the original hardware

The experiment was re-run on the same RTX 4090 under a pre-registered protocol —
`PREREG_exp103_replication.md`, sha256 `7ffa2ee4386b2083727ea6ac03cd88b8bb414680017b65279b34f16fc4236516`,
frozen 23:56:59Z with predictions, interpretations and a decision rule fixed before any
measurement was taken.

**The answer is neither side.** Both reported values are single unrepeated draws from an
estimator whose run-to-run spread exceeds the effect they disagree about, and the lockstep
leg is a code artifact.

**1. Lockstep ρ=1.0 is an artifact** *(registered prediction P1, confirmed).*
`exp103_software_injection.py:127` times one batch and assigns that scalar to all 64 sensors
(`all_timings[:, s] = t1 - t0`), so `corrcoef` runs over identical rows. Replication:
`identical_rows = True`, ρ = 1.0000 exactly, 0.0000 after common-mode removal, at both n=30
and n=200. Same defect class as C-5 and C-11.

**2. Barrier correlation is real and survives detrending** *(P2 confirmed, P3 refuted).*
I predicted barrier ρ would prove to be a shared-measurement-epoch artifact vanishing under
common-mode removal. **It did not.** At n=200: ρ = 0.792 against a shuffle-null p95 of 0.057,
still 0.438 detrended. v3's qualitative direction is supported.

**3. But the estimate is unstable** *(post-hoc, not registered).* Five identical trials at
F2's exact parameters (64 sensors, 50 samples):

| Trial | independent | barrier |
|---|---|---|
| 0 | 0.777 | 0.505 |
| 1 | 0.156 | 0.809 |
| 2 | 0.111 | 0.278 |
| 3 | 0.136 | 0.485 |
| 4 | 0.173 | 0.338 |

Barrier spans 0.278–0.809 (sd 0.18); the two conditions' ranges overlap. In k_eff: **1.23 to
3.46**. Exp 103's 0.90 and F2's 0.0495 are both consistent with this one unchanged procedure.

**4. The baseline is warm-up contaminated** *(post-hoc).* The first measurement after idle
reads ρ ≈ 0.777 regardless of condition — observed at 0.7777 and 0.7772 in two independent
sessions — settling to 0.11–0.17 after. **Both Exp 103 and F2 measure their baseline first**,
so every Δρ against it is unreliable.

**Also refuted (P5): the estimator hypothesis.** F2 uses signed `mean(r)`, Exp 103 and exp117
use `mean(|r|)`, but correlations are overwhelmingly positive so signed ≈ absolute throughout
(0.792 vs 0.792 at n=200). And `expF2_causality_test.py:86-102` is byte-identical to
`exp103_software_injection.py:63-82` — same mechanism, opposite reported results.

**Disposition.** Lockstep withdrawn. Barrier retained directionally with no point estimate.
**"Software alone can induce collapse" is not supported** — its demonstration was the artifact.
Applied to the paper at §10.3 (Remark: replication on the original hardware).

**Program-level finding.** No run in this series computed a null, repeated a measurement, or
reported an interval. At the shipped n=30 the shuffle null's 95th percentile is **0.15**, so
some portion of the correlations reported across these experiments may be estimator floor.
This warrants a sweep beyond Exp 103.

*Verified: F2, exp117, F3 values read from stored results; replication run and recorded in
`exp103_replication_results.json` and `stability_results.json`.*

## C-14 — "Corridor validated" never measured the rigidity arm *(major)*

Every ρ value F3 measured: baseline 0.037, lockstep 0.0036, shared-memory 0.171,
single-stream 0.156, combined 0.143, intervention 0.170. **Maximum 0.171**, against a
rigidity edge of 0.43.

The corridor claim is two-sided — *both* too little and too much correlation produce
fragility. The rigidity arm, which is the failure mode CCA exists to warn about, was never
measured. The claim's risky half carries no risk.

*Verified: all six values read from `expF3_results.json`.*

## C-15 — No row of Table 8 contains a measured correlation *(major)*

C-2 and C-3 scoped the defect to the institutional rows. It is wider:

- **Institutional**: `ρ = vdem_corr × (1 − k)` (`institutional_loader.py:339`) — a product of
  a corruption index and a constraint deficit. No covariance enters. At `xconst=7`, `k=1`
  and **ρ ≡ 0 by construction at any corruption level**; six of fourteen countries get
  ρ=0.000 this way.
- **Battery**: `ρ = 1 − 10·cv` (`battery_loader.py:151-155`), which returns **ρ=1.0** for
  identical cells — the inverse of the row's gloss "fresh cells, full independence."
- **Microbiome**: `ρ = 0.25(1−0.3σ)(1.2−0.4·evenness)`, where `evenness` calls the same
  Shannon function that produces σ. `ρ` is an analytic function of σ, confined by
  construction to [0.14, 0.30]. "Species clustering reduces k" was never measured.

Of four rows, three sit on degenerate boundaries (ρ=0 twice, k=1 once) and one is out of
domain — **only the microbiome row exercises the formula non-trivially.** Table 8 is
therefore a further instance of C-5.

**Sharper form of C-2 (supersedes it).** For `k < 1`, `∂k_eff/∂ρ = −k(k−1)/(1+ρ(k−1))² > 0`:
the formula **runs backwards** below the domain boundary. For Venezuela's `k=0.667` the
attainable range of `k_eff` across all ρ∈[0,1] is **[0.667, 1.000]**, so the printed 0.55
is unreachable by *any* correlation. The row's reading "elite coupling reduces diversity"
asserts the opposite of what the formula does at those inputs.

**Reconstruction is impossible from data on disk.** The only count-valued institutional
variable available is V-Dem `v2lgbicam` (legislative chambers), which equals 1 for both
Venezuela and Türkiye in all 25 years 2000–2024 — the degenerate `k=1` case C-3 rejects.
Polity `xconst` is an ordinal rating; the V-Dem constraint variables are 0–1 indices; DPI
`checks` and QoG are not on disk.

*Verified: all three ρ definitions, the sign inversion, the attainable range, and
`v2lgbicam` across both countries.*

## C-16 — The abstract states L-01's 40% as proven *(major)*

The theorem (§5) proves only: *"There exists a non-empty class of emergent incoherence
patterns that are fundamentally undetectable."* Existence, no measure. The body's own
remark concedes the 40% follows from choosing β=10 and ε≈0.092 "for illustrative purposes"
and is "not a universal constant."

**This defect is being actively enlarged.** The uncommitted working-tree abstract changes
v3's *"An information-theoretic barrier establishes that ~40%…"* to **"We prove an
information-theoretic ceiling (L-01): roughly 40%…"** — upgrading an illustrative parameter
choice to a claimed proof, in the paragraph most readers read. That edit must not be
committed as written.

*Verified: theorem statement, remark, and working-tree abstract all read directly.*

## C-17 — The corridor Monte Carlo does not exist *(major)*

The uncommitted abstract asserts "an accompanying Monte Carlo suite reproduces the corridor
regime structure." The only Monte Carlo in the repository is
`simulation/hyperplane_intersection_volume.py`, which tests the volume-decay theorem, not
the corridor. No corridor Monte Carlo was located.

*Verified: repository-wide search.*

## C-18 — `wgi_processed.csv` is defective throughout *(critical, beyond this paper)*

`data/institutional/wgi_processed.csv` ships precomputed `k`, `rho`, `k_eff` columns. Of
5083 rows, 4933 have non-null `k_eff`, and **all 4933 have `k < 1` and `k_eff > k`** — the
C-15 sign inversion at scale. 4884 rows report `k_eff > 1` derived from a fractional
"count." Afghanistan 1996 reads `k = 0.065, k_eff = 0.912`.

Separately, `ratchet/engines/institutional.py:94` silently applies `k_scaled = self.k * 10`
before the Kish formula — an undocumented rescale giving a third mutually inconsistent
pipeline, and a third value for Venezuela's `k_eff` (2.47, against the table's 0.55 and the
formula's 0.741).

**This is a data-integrity defect, not a paper-corrections item.** Anything downstream of
that file inherits it. It needs its own tracked issue.

*Verified: all 4933 rows checked; rescale read directly.*

---

## Corroborating evidence already in the repository

The repository's own result files contradict §11 and predate this audit:

- `experiments/exp0_cca_validation/results/polity_xconst_validation_results.csv`: on the
  full 203-country panel, σ baseline AUC 0.656, `k_eff_xconst` **0.420**,
  `k_eff_xconst_wgi_rho` **0.293** — anti-predictive, worse than the baseline it was meant
  to beat. The exp0 README names these as the source of the paper's Tier-1 numbers.
- `scripts/test_institutional_collapse.py:44-65` labels Hungary and Poland under the
  comment **"Democratic backsliding cases"**, and lists only Germany, Canada, Australia as
  the **"Stable democracies (control group)"** — contradicting §11's "5/5 stable
  democracies," which counts Hungary and Poland among them.

*Relayed, not independently re-verified.*

## An exploratory probe, recorded and quarantined

V-Dem judicial/legislative constraint co-movement 2000–2024: Türkiye 0.985, Hungary 0.982,
Poland 0.892, versus Germany −0.257 (first-differenced: 0.80 / 0.58 / 0.56 / 0.00). Under
the paper's own construct the two countries §11 scores as healthy look like the collapse
cases.

**This is post-hoc, n=2 bodies, unregistered, and must not be published as a result.** It is
recorded here only so that it is not later rediscovered and mistaken for independent
confirmation. Promoting it would require pre-registration and a country set fixed in
advance.

---

## What the paper still supports

The modest true claim is unaffected by every defect above and should be restated as the
paper's actual content:

> **Correlated overseers provide less assurance than their nominal count.** Effective
> diversity follows the Kish design effect `k_eff = k/(1+ρ(k−1))`, saturating at `1/ρ`;
> scale alone cannot restore diversity that correlation has already collapsed.

This is the algebraic core, it is independent of any substrate-specific calibration, and
it is corroborated by convergent results in three unrelated literatures — cited as
corroboration (hits), never as strikes against the alternatives:

- **Kish (1965)**, *Survey Sampling* — the design effect itself, the source identity.
- **Ladha (1992, 1995)** — Condorcet jury theorems under correlated votes: correlation
  degrades collective accuracy relative to the independent-voter baseline.
- **Knight & Leveson (1986)** — N-version programming: independently developed program
  versions fail on correlated inputs, so redundancy delivers less than its nominal count.
- **Laakso & Taagepera (1979)** — the effective number of parties: the same
  inverse-concentration construction arrived at independently in political science.

Also surviving, and worth keeping:

- **The corrected stability theorem** (C-1). The qualitative claim — constraint generation
  must outpace sustainability decay — holds, at a threshold `(1+ρ(k−1))×` stricter than
  published. Corollary 2.4 survives *because of* the correction (C-7).
- **The hardware instrument.** The 128-sensor GPU strain gauge is real, and its
  measurements of workload-induced correlation are genuine measurements.
- **`exp117`**, currently unreported: a two-sided test of the equicorrelated precondition
  that both passes (0.8% at ρ=0.394) and fires (17.6%, 70.8% at low ρ). The most honest
  result in the corpus.

### Scope of what is implicated *(revised after the second wave)*

The first-wave version of this section said the GPU results were "not implicated by any
defect in this note" and that L-01 was "untouched." **Both statements are now withdrawn.**
They were written before C-11, C-13, C-14, C-16 and C-17 were found, and this note does not
get to quietly improve its own earlier claims — the superseded sentence is named here
rather than deleted.

What is implicated:

| Leg | Status | Defects |
|---|---|---|
| Stability theory | Corrected, survives | C-1, C-6, C-7 |
| Cross-domain table | No row carries a measured ρ; institutional rows unreconstructable | C-2, C-3, C-15 |
| Institutional validation | Does not test `k_eff`; scores below chance; underpowered by design | C-4, C-8, C-9, C-10 |
| GPU Kish validation | Both results are identity checks; no independent test exists | C-5, C-11 |
| Cross-domain invariants | Computes nothing | C-12 |
| Corridor / software-collapse | Rigidity arm unmeasured; flagship result contradicted 3-to-1 | C-13, C-14 |
| L-01 40%/60% | Existence theorem only; percentages are an illustrative parameter choice | C-16 |

**With the institutional leg withdrawn, "three domains" is two.** With both Kish results
withdrawn, the hardware *validation* claim is gone — though the hardware *measurements*
remain. The word "validated" cannot stand in most places it currently appears.

---

## Actions

| # | Action | Status |
|---|---|---|
| 1 | **Do not delete the Zenodo record.** | Standing |
| 2 | Fix Zenodo resource-type metadata: currently displays as **"Peer review"**; it is a **preprint**. Cheapest and most misleading item. | Pending |
| 3 | Publish this note as a corrections version against the same DOI concept. | Pending — this draft |
| 4 | Correct Thm 2.3 and Thm 4.1 to `α/k ≥ d`; repair Eq. (5); fix the "Theorem 2.4" cross-reference. | **Applied** |
| 5 | Withdraw the Venezuela and Turkey rows of Table 8. | **Applied** |
| 6 | Re-score §11 against a pre-specified outcome definition; remove the §14 false-positive rewrite. | **Applied** |
| 7 | Relabel §10.2 as an implementation check. | **Applied** |
| 8 | Reconcile `J` with CC 6.2.2 (`J = k_eff·λ·σ`). | **Applied** |
| 9 | Restate Corollary 2.4 in terms of `k`; document why `k_eff` cannot carry it. | **Applied** |
| **10** | **Hold the uncommitted abstract rewrite** — as written it upgrades L-01's illustrative 40% to "We prove" (C-16) and asserts a non-existent corridor Monte Carlo (C-17). | **Blocking — do first** |
| **11** | Resolve Exp 103 vs F2 / exp117 / F3 (C-13). | **Applied** — replicated on the original hardware under pre-registration; lockstep withdrawn as an artifact, barrier retained without a point estimate |
| **12** | Decide whether §11 leaves the evidence chain entirely, given C-8 (it does not test `k_eff`). | Decision needed |
| 13 | Withdraw `R²=0.798` from the contributions list; strike "validated" where C-11/C-12 remove its basis. | Pending |
| 14 | Promote `exp117` as the one genuine two-sided test — after auditing it as Exp 86 was audited. | Pending |
| 15 | File `wgi_processed.csv` (C-18) as its own data-integrity issue; it is not a paper-corrections item. | Pending |

**Applied in `coherence_collapse_analysis.tex`** (items 4, 8, 9 — the unambiguous
corrections, each carrying an in-text remark naming the v3 error rather than fixing it
silently):

- Thm 2.3 restated as `α/k ≥ d`, with the derivation repaired: the common denominator
  `1+ρ(k−1)` now visibly cancels to `R_c = λσ(α−dk)/(1+ρ(k−1))`, and the stray `/(1−ρ)`
  and `(1/σ)·σ` are gone.
- Thm 4.1 (Healthy Corridor) second conjunct corrected; corridor ρ-bounds untouched, with
  a note that they are substrate-specific and GPU-anchored.
- Corollary 2.4 restated over `k`, plus a remark recording that it is false under the v3
  criterion (k_eff saturation) and restored only by the correction.
- `J = k_eff·λ·σ` throughout: definition, derivation, J–C duality proposition, quick
  reference sheet, and the contributions list.
- `Theorem~2.4` cross-reference in §9 replaced with a `\ref` to the stability theorem
  (it had pointed at the corollary).

Theorem numbering is unchanged by these edits (2.3 Stability, 2.4 Corollary, 4.1 Healthy
Corridor), so citations to the v3 numbering remain valid. Document builds clean with no
undefined references.

**Items 5–7 applied 2026-07-31**, after the two open questions they turned on were settled
on evidence:

- **Table 8's institutional rows are withdrawn, not reconstructed.** Count semantics is not
  reachable — a search of all 4,607 V-Dem variables returns no count-valued institutional
  constraint measure with usable variation (`v2lgbicam` ∈ {1,2} and is 1 for both countries
  throughout; `e_legparty` is constant; everything else is a continuous index). The deeper
  reason is stated in the paper: the design effect assumes the `k` units are **exchangeable**,
  which is what makes ρ an *intraclass* correlation. Executive, legislature and judiciary are
  structurally different kinds of institution, so the equicorrelated precondition fails **by
  construction**, independently of data availability. Acquiring a veto-player count from
  another source would reintroduce the error with better inputs, and the withdrawal note says
  so explicitly to foreclose that.
- **§11 is retained as a negative result, retitled and moved out of the validation chapter**
  (new §11, *A Negative Result: The Institutional Case Study*). It carries the honest
  confusion matrix (4/5/2/3, MCC −0.15, permutation p=0.87), the underpowering finding, the
  threshold-fragility table with the reason the one favourable definition is disqualified,
  and the C-8 finding that the engine never consults `k_eff` — which is why the section
  cannot sit in a validation chapter whatever its score.
- **The 7.6-year timing claim is withdrawn** with the null comparison stated in its place:
  seven of nine flags fire at initialisation because `f₀ > 0.8` already held, and a null
  flagging every country in 2001 achieves a *larger* mean lead (10.8y) at sensitivity 1.0.

**Still not applied: items 10–15.** Items 10 and 11 block republication.

## Recommended disposition *(revised after the second wave)*

The first wave was consistent with publishing a corrections version against a fundamentally
sound paper. The second wave is not. All three empirical legs fail, and the failures are
structural rather than reparable: §11 does not test the framework's central quantity, no
independent test of the Kish identity exists anywhere in the corpus, and the flagship
software-collapse result is contradicted by three later experiments.

The recommendation is therefore to **restructure the paper as what the work actually is —
a mathematical framework plus a characterized instrument — and withdraw the empirical
validation claims rather than repair them.** The v3 record stays up and is annotated, per
the standing rule that a dead DOI reads as concealment while a corrected one reads as
scholarship.

One pattern is worth naming, because it suggests the fix is smaller than the defect list
implies: **the paper's hedges are consistently correct and consistently located too far
from the claims they qualify.** Table 8's caption already concedes `k_eff` is not
independently observable. The Statistical Provenance remark correctly diagnoses the
identity-check problem 300 lines before it occurs. The formal-verification note correctly
says Lean proves internal consistency only. The threshold-calibration section refuses the
universality the GPU numbers would have tempted. Nearly every defect above is already
diagnosed somewhere in the same document — the abstract and conclusion simply do not honour
those concessions. Moving the hedges adjacent to the claims, and letting them *govern*
rather than accompany, would resolve a large fraction of this list.

---

## Reproduction

```
cd CCA_PAPER && python3 corrections_witness.py
```

Requires `sympy`. Exit 0 means every defect above reproduced from the paper's own stated
assumptions.

---

*Refs: external review 2026-07-30; RATCHET#11; CIRISConstitution#45; CIRISOntology
`GATES.md` (warrant reach, received-numbers-are-not-measured, report-the-fired-kill).*
