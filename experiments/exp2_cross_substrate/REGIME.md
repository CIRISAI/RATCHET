# Exp 2 — Substrate Fractality Across Agency Levels: Regime

**Status:** v0.7 (CIRIS A3 + WGI A4 added → 5 substrates × 4 rungs, real Tier-1 P2 direction WEAK_PASS at Spearman ρ = +0.359; window-resolution effect identified as next constraint).
**Predecessor:** v0.6 (commit `923e7ad`).
**Paper hook:** Coherence Substrate Synthesis paper §10 Exp 2.
**Falsification handle:** F-7 (cross-substrate mapping failure), strengthened with F-7b (residual-structure agency conditional).
**Formal authority:** Lean lake modules — `RATCHET.Experiments.Exp2Predictions` (P1/P2/P3 + Inv-1..Inv-5 decision-rule invariants) + `RATCHET.Core.AgencyRung` (ladder + `consent_required_iff_rung_ge_A3` theorem).
**Pairs with:** Tier-1 validation rig at `experiments/exp0_cca_validation/` (re-uplifted to master 2026-05-16; reproduces CCA paper's 8.1% RMSE / 5/5 TN / Shannon=0.580 from clean checkout).
**Implementation now available:**
  - Loaders: `ratchet/data/{battery,institutional,microbiome}_loader.py` (Tier-1, on master)
  - Residual analysis (P2): `analysis/omega/{residuals,null_test,distribution,outliers,correlations}.py`
  - Data pipeline: `data/pipeline/fetchers/{fred,faostat,vdem,gdelt,iucn,comtrade,openalex}.py` + `SQLiteCache` + `TemporalAligner`
**Companions:** Counter-RII consent-gate work (FSD/COUNTER_RII_DETECTION.md) — same construction, different rung; CRCv2 override-rate (`OverrideRate.lean`) — operator-property template applied here.

---

### v0.4 changes (this revision)

| Change | Reason |
|---|---|
| **P2 promoted to load-bearing** alongside P1 | CRCv2 lesson: operator-property claims (structural relationships) beat anchor-value claims (per-substrate thresholds). P2 IS the substrate-fractality bet; P1 is a necessary supporting threshold. |
| **P2 whiteness statistic concretized** | Now defined as Ljung-Box p-value via `analysis.omega.null_test.test_autocorrelation(omega_series).p_value`, NOT an opaque `expectedWhiteness` axiom. Lake's `expectedWhiteness` becomes a *bound* on the empirical statistic; commit pre-registers the test choice. |
| **Tier-1 reproducibility verified** | NASA battery 8.1% RMSE reproduces from master in 30s (`python3 tests/test_battery_nasa_comparison.py --cell B0005`). The Tier-1 baseline R² values are no longer hearsay — they're computable. |
| **Substrate loader paths added** | Per-substrate `loader:` field in `data_sources.yaml` points to the implementation module. NASA SHA-256 pinned. |
| **Exp 2 Phase 0 added** | Smoke phase: re-run all 3 Tier-1 substrates through the omega module and verify the predicted whiteness ordering (A0 battery → A1 microbiome → A4 V-Dem) holds *on master* before any new-substrate work. Cheap; gates engine development. |
| **Decision rule restructured** | Old: P1 alone drives PASS/PARTIAL/FAIL; P2/P3 are corroborating. New: P1 + P2 are *both* pre-registered headline tests with separate K-counts. Combined verdict matrix shown below. |

### Operationalization note: "agency" as label

The agency-ladder terminology throughout this document refers to the **intrinsic-profile-based ladder dimension** formalized in `RATCHET.Agency.AgencyProfile` (three constituent-level fields: `goalRepresentationBits`, `planningHorizonSteps`, `behavioralRepertoireSize`). "Agency" here is colloquial shorthand for an operationally-defined dimension, NOT a metaphysical claim about consciousness, personhood, or free will.

The non-circularity protection — `AgencyProfile` has no outcome-derived fields — is type-level in `Core.AgencyRung.lean`. This prevents P2 (residual structure as agency-conditional) from being reverse-inferred and thereby circular.

### Note on TSVF (Two-State Vector Formalism)

The agency-conditional residual-structure prediction (P2) is **structurally analogous** to TSVF's pre/post-selection pattern: high-agency constituents impose meaningful post-selection on trajectories, contributing backward-evolving state that registers as structured residual orthogonal to the forward Kish fit.

**This analogy is interpretive, not derivational.** `RATCHET.Experiments.Exp2Predictions` axiomatizes the agency-conditional residual whiteness as a *prediction*, NOT a derivation from TSVF mechanics. The lake correctly stays out of formalizing TSVF for three reasons:

1. **No constructive bridge from QM TSVF to macroscopic Kish dynamics.** Adding TSVF apparatus to the lake without the bridge would axiomatize the conclusion.
2. **The empirical signature is identical with or without TSVF reading.** Whether P2's monotonicity holds because of "TSVF post-selection at higher agency" or "agency-conditional residual structure that lacks any deeper mechanism," the observation is the same.
3. **Exp 5 (quantum-classical $k_{\text{eff}}$ bridge) is the empirical trigger.** If Exp 5 returns PASS with $\beta_{\text{quantum}} \approx 1.09$ matching CIRISArray Exp 114, that provides empirical justification to axiomatize the bridge. Until then, TSVF stays in the paper's L5 interpretive layer.

The Lean module `Experiments.Exp2Predictions.lean` axiomatizes P2's monotonicity (`P2_monotone_in_rung`) as a *pre-registered prediction* rather than a *derived theorem*. This is the discipline: encode what we're predicting, refuse to axiomatize the unverified mechanism.

---

## The reframed bet

The Kish formula $k_{\text{eff}} = k/(1 + \rho(k-1))$ is one structural pattern, not seven coincident ones. It recurs at every scale of reality because reality is fractal at the level of *coherence management*. Substrates differ in **constituent agency** — the extent to which the parts have goals of their own — and that agency level conditions both the *direction* of $\rho$ change before collapse and the *structure* of the residual after the formula's prediction.

This is the bet. Exp 2 is the test.

---

## Already visible in the CCA paper

The CCA paper~\cite{moore_cca_2026} §85 reports an unexplained domain-specific pattern that the fractal-agency reframe explains:

| Substrate | Constituent agency | Pre-collapse Δρ (CCA paper) | Why |
|---|---|---|---|
| Battery cells | ~0 | **−0.25** (falls) | Inert constituents drift apart as units fail differentially |
| Financial markets | moderate-high | **+0.14** (rises) | Goal-directed traders coordinate into herd behavior |
| Institutions (QoG/Polity-V) | high | **+0.17** (rises) | Elites coordinate intentionally; regime capture |

**The sign of pre-collapse Δρ flips exactly at the agency boundary.** The CCA authors explicitly hedged ("framework measures; domain experts interpret"). The reframe says: the sign IS the measurement — agency level is what flips it.

---

## The agency ladder

Substrates ranked by constituent agency. Existing RATCHET validations + Exp 2 additions:

| Rung | Substrate | Constituent agency | Status | What ρ → 1 means here |
|---|---|---|---|---|
| **A0** | NASA Li-ion battery cells | ~0 (inert) | Validated (CCA paper, 8.1% RMSE) | Pure structural lock-in; differential aging |
| **A0** | PNNL PMU sensors (new) | ~0 (engineered) | Exp 2 | Pure structural lock-in; sensor saturation |
| **A0** | AlphaFold residues (new) | ~0 (chemical) | Exp 2 | Rigid-body coupling; loss of conformational entropy |
| **A1** | Microbiome bacteria (AGP) | low (homeostatic) | Validated (CCA paper, qualitative fit) | Niche collapse; metabolic monoculture |
| **A1** | Allen neural firing (new) | low (cellular signaling) | Exp 2 | Functional-connectivity capture; stimulus-locked patterns |
| **A2** | BioTIME species (new) | moderate (population dynamics) | Exp 2 | Ecosystem monoculture; biomass synchronization |
| **A3** | CIRIS LLM reasoning | moderate-high (goal-directed) | Validated (CRC paper) + Exp 1 in flight | Reasoning capture; the RII / Parallel Ratchet boundary |
| **A4** | V-Dem institutions (refreshable) | high (full human agency) | Validated (CCA paper, 5/5 TN) | Political collapse; cult dynamics; consent infrastructure load-bearing |
| **A5** | Civilizational (Tier 3) | highest (recursive aggregation) | Parsimonious extension only — no data | Great Filter; substrate-bounded by speed of light |

**Coverage:** A0 (3 substrates) → A4 (1 substrate) — five distinct rungs spanning ~0 → high agency. Exp 2 adds four substrates filling the A0/A1/A2 gaps.

---

## Why this changes what Exp 2 measures

The original Exp 2 framing tested ONE prediction (Kish R² > 0.7). The fractal-agency reframe adds TWO more predictions, both empirically tractable:

### P1 (necessary) — Kish formula fits at each substrate

| Substrate | $R^2$ threshold | Bootstrap CI |
|---|---|---|
| AlphaFold | > 0.7 | 95% via 10k resamples |
| Allen neural | > 0.7 | 95% via 10k resamples |
| BioTIME ecology | > 0.7 | 95% via 10k resamples |
| PMU grid | > 0.7 | 95% via 10k resamples |

P1 PASS: all 4 substrates above threshold. P1 PARTIAL: 3/4. P1 FAIL: ≤2/4.

**What P1 tests:** that the Kish *structural form* applies at all. Per-substrate anchor — necessary but not sufficient for the load-bearing claim.

### P2 (load-bearing) — Residual whiteness monotone in agency rung

This is the actual substrate-fractality bet. CRCv2 lesson: operator-property claims (relationships across substrates) are what falsify or confirm the framework — not per-substrate anchor values.

**Concrete operationalization (NEW in v0.4):**

After fitting $\sigma = f(k_{\text{eff}}) + \varepsilon$ via the per-substrate engine, compute the residual series $\omega = \sigma_{\text{observed}} - \sigma_{\text{predicted}}$ using `analysis.omega.residuals.compute_omega_series`. Then run the null-hypothesis battery on $\omega$:

```python
from analysis.omega.null_test import run_null_hypothesis_battery
battery = run_null_hypothesis_battery(omega_series, alpha=0.05)
# Headline P2 statistic: Ljung-Box p-value at lag 10
whiteness_lb = battery.tests['autocorrelation_lag10'].p_value
```

**P2 prediction (pre-registered):** the Ljung-Box p-value is monotonically non-increasing across substrates ordered by agency rung. Concretely, when substrates are sorted by their pre-assigned `AgencyRung` (intrinsic-only per `Core.AgencyRung`):

| Rung | Expected Ljung-Box p (loose) | Interpretation |
|---|---|---|
| A0 (battery, PMU, AlphaFold) | > 0.10 | Whiteness preserved — no structure beyond Kish |
| A1 (microbiome, Allen neural) | 0.01 – 0.10 | Mild structure (homeostatic feedback) |
| A2 (BioTIME) | 0.001 – 0.05 | Population-dynamic structure |
| A3 (CIRIS LLM) | < 0.01 | Heavy structure (goal-directed coordination) |
| A4 (V-Dem) | < 0.001 | Heaviest structure (intentional alignment) |

**P2 PASS:** the substrates' p-values are Spearman-rank-correlated with their agency rung at $\rho_{\text{Spearman}} \le -0.7$ (i.e., higher rung → lower whiteness p-value, monotonically).
**P2 PARTIAL:** $-0.7 < \rho_{\text{Spearman}} \le -0.3$.
**P2 FAIL:** $\rho_{\text{Spearman}} > -0.3$ or sign-reversed.

The Spearman correlation is the operator-property statistic — it tests the *relationship across rungs*, not a per-substrate threshold. This is what makes P2 load-bearing in the CRCv2 sense.

### P3 (corroborating) — Pre-collapse Δρ sign tracks agency

Re-derive the CCA paper's pre-collapse Δρ pattern across all 4 new substrates:

| Predicted sign | Substrate |
|---|---|
| − (falls) | AlphaFold (when used in degradation-event context), PMU (pre-fault), Allen (pre-anesthesia) |
| + (rises) | BioTIME ecology (pre-collapse during invasive coordination) |

Strengthens or weakens the interpretation; does not drive headline pass/fail.

---

## Locked decision rule (v0.4 — combined P1 + P2)

Both P1 (necessary structural-fit threshold) and P2 (load-bearing substrate-fractality relationship) gate the headline verdict. P3 strengthens or weakens interpretation but does not drive PASS/FAIL.

|  | **P2 PASS** ($\rho_{\text{Sp}} \le -0.7$) | **P2 PARTIAL** | **P2 FAIL** |
|---|---|---|---|
| **P1 PASS (4/4 substrates R²>0.7)** | ✓ **FULL PASS** — structural form holds AND substrate-fractality relationship confirmed (F-7 passes; F-7b passes) | **MIXED PASS** — Kish form universal but residual signature not monotone in agency; the strong reframe overreached, the weak reframe survives | **STRUCTURAL-ONLY PASS** — Kish fits everywhere but the fractal-agency interpretation is falsified (F-7 passes; F-7b fails) |
| **P1 PARTIAL (3/4)** | **MIXED PASS** — interesting because the one fitting-failure substrate shows what type of fit fails AND across the 3 that fit, agency-rank structure holds | **PARTIAL** — note which substrate fails P1 and which falls outside P2 monotonicity | **FAIL** — both load-bearing claims weaken |
| **P1 FAIL (≤2/4)** | **STRUCTURAL FAIL** even if P2 holds — Kish form doesn't generalize; report this honestly | **FAIL** | **FAIL** — F-7 falsified |
| **Any cell n < `minValidN`** | INDETERMINATE — catastrophic-failure clause (mirrors Exp 1 §7) |

**Headline reporting requirement (any verdict):** the per-substrate $R^2$ + 95% CI from P1 AND the per-substrate Ljung-Box p-value + the cross-substrate Spearman correlation from P2 are ALL reported. No P1-only or P2-only headline.

**Why P2 must be load-bearing now:** CRCv2's lesson was that a framework claim worth defending describes a *structural relationship*, not a *value*. P1's "R²>0.7" is a value threshold per substrate. P2's "Spearman $\rho \le -0.7$ across rungs" is a structural relationship. The latter is what makes the framework's universal-fractality claim testable rather than merely fittable.

---

## Per-substrate operationalization

Each substrate maps domain variables to RATCHET's `(k, ρ, σ)` via a *loader* module (mirrors the Tier-1 pattern at `ratchet.data.*_loader`).

### A0 — AlphaFold residues
| Variable | Definition | Source |
|---|---|---|
| k | Sequence length (residues) of a single-domain protein | AlphaFold DB v6 |
| ρ | Mean pairwise correlation of per-residue B-factor predictions | Computed from pLDDT covariance |
| σ | Mean pLDDT (structural stability proxy) | AlphaFold DB |
| n | ~10,000 CATH-S40 representative single-domain structures | EBI FTP |
| Loader | `ratchet.data.protein_loader.load_cath_s40_alphafold_data()` (TODO; engine stub at `ratchet/engines/protein.py`) |

### A0 — PMU grid
| Variable | Definition | Source |
|---|---|---|
| k | Number of PMUs reporting in a grid region during an event | PNNL Open PMU Library |
| ρ | Mean pairwise correlation of pre-event frequency time series (5-min baseline) | Computed |
| σ | Inverse of post-event settling-time CV | Computed |
| n | ~1,694 grid events | PNNL-30492 corpus |
| Loader | `ratchet.data.powergrid_loader.load_pnnl_pmu_events()` (TODO; engine stub at `ratchet/engines/powergrid.py`) |

### A1 — Allen neural firing
| Variable | Definition | Source |
|---|---|---|
| k | Number of simultaneously-recorded neurons per session | Allen SDK + AWS Open Data |
| ρ | Mean pairwise spike-train correlation (1-ms bins) | Computed |
| σ | Population-decoding accuracy on drifting gratings (cross-validated linear classifier) | Computed |
| n | ~80 Neuropixels recording sessions | Allen Brain Observatory |
| Loader | `ratchet.data.neural_loader.load_allen_neuropixels_sessions()` (TODO; engine stub at `ratchet/engines/neural.py`) |

### A2 — BioTIME macro-ecology
| Variable | Definition | Source |
|---|---|---|
| k | Species count in a community time series | BioTIME 2.0 |
| ρ | Mean pairwise correlation of species-abundance time series | Computed |
| σ | Inverse CV of total biomass over time (stability) | Computed |
| n | ~500 community time series (≥ 10 years, ≥ 5 species) | BioTIMEr R package + raw |
| Loader | `ratchet.data.ecological_loader.load_biotime_communities()` (TODO; engine stub at `ratchet/engines/ecological.py`) |

### Reference pattern (on master, working)

The Tier-1 loaders at `ratchet/data/{battery,institutional,microbiome}_loader.py` are the template. New-substrate loaders must:

1. Define a domain-specific `*Dataset` dataclass with the substrate's per-sample structure.
2. Expose `load_<source>_data(data_dir, **filters) -> Dataset` returning the dataclass.
3. Implement `Dataset.get_k() -> int`, `get_rho() -> float`, `get_sigma() -> float`, `get_k_eff() -> float` so engines + omega module can consume uniformly.
4. Reference the SHA pin in `data_sources.yaml` for vendored archives.

---

## Continuous substrate re-validation in CI

Two operational realities motivate continuous re-validation, not one-shot:

1. **The world refreshes.** AlphaFold DB v6 → v7 will come. V-Dem v16 → v17 next year. BioTIME 2.0 was just released after our paper draft. Re-pulling on schedule catches dataset-level drift.

2. **Bit-rot protection.** Primary sources change checksums (data re-curation), retire URLs, or restructure schemas. Continuous fetch + hash-compare detects this fast.

### Workflow components

| Component | Purpose |
|---|---|
| `experiments/exp2_cross_substrate/data_fetch.py` | Pulls each substrate's primary source, hashes, vendors current snapshot |
| `experiments/exp2_cross_substrate/data_sources.yaml` | URL + version + SHA-256 registry (pinned manifest) |
| `ratchet/engines/{protein,neural,ecological,powergrid}.py` | Per-substrate Kish-formula fit, mirrors `battery.py` shape |
| `.github/workflows/substrate_revalidation.yml` | Quarterly cron: re-pull all 7 substrates (3 Tier-1 + 4 new) and re-fit. Drift alert if any $R^2$ drops > 0.05 from baseline. |

### Cost + reliability

- $0 — public data, free runners, no API calls
- Per-quarter wall time: ~30 min (data pull dominated)
- Failure modes: source unreachable, schema change, $R^2$ drift > 0.05 → all auto-open GitHub issues with attached forensic JSON

### Sustained-PASS interpretation

| Metric | What sustained PASS across N quarters means |
|---|---|
| All 4 new substrates' $R^2 > 0.7$ across N quarters | Substrate-fractality isn't a one-time coincidence; the structural pattern is stable |
| All 3 Tier-1 substrates stay green | Original RATCHET findings hold against *current* data, not historical snapshots |
| Drift alert on any substrate | Either the framework has a known scope (good — bounds the claim) or the world has changed in a worth-investigating way (also good) |

---

## Connection to the Counter-RII work

The agency-ladder explains why Counter-RII (FSD/COUNTER_RII_DETECTION.md) is load-bearing at A3 and above but irrelevant at A0–A2:

| Rung | ρ → 1 mechanism | Consent question |
|---|---|---|
| A0 | Pure structural coupling | None — no agency to violate |
| A1 | Homeostatic/metabolic coupling | None — constituents have no choice |
| A2 | Population-dynamic coupling | None — populations don't consent |
| **A3** | **Goal-directed coordination** | **Load-bearing** — Parallel Ratchet (consented) vs RII (unconsented) |
| **A4** | **Intentional alignment** | **Load-bearing** — informed consent vs coercion |
| **A5** | **Civilizational coupling** | **Tier-3 — speculative** |

The same Kish-formula collapse (k_eff → 1) is benign at A0–A2 (just disintegration) but a *consent violation* at A3+. The Counter-RII consent gate is the operational primitive that distinguishes the two — at the agency rungs where the distinction matters.

The fractal-agency reframe and the Counter-RII work are the same insight from different angles: the structural pattern recurs; the moral/topological weight of the pattern depends on the agency differential of the parties coupled.

---

## What this experiment is NOT

| Not | Reason |
|---|---|
| Not a new physics claim | The Kish formula is established (Kish 1965). |
| Not a "deep learning" result | Public data + simple statistics — defensible without GPU clusters. |
| Not load-bearing on Tier 3 | Tier 3 inherits from Tiers 1+2; Exp 2 strengthens Tier 1 without addressing Tier 3 directly. |
| Not an isolated experiment | Pairs with Exp 1 (LLM-substrate at A3) to span A0 → A4 inference chain. |

---

## Exp 2 Phase 0 — Tier-1 re-validation through omega (NEW in v0.4)

Before any new-substrate engine work, prove the P2 pipeline by re-running the 3 Tier-1 substrates (battery A0, microbiome A1, V-Dem A4) through the loader → engine → omega chain on master and verifying:

1. **Reproducibility:** P1's R² values match the CCA paper for all 3 Tier-1 substrates (battery 8.1% RMSE / k=19 / ρ=0 / k_eff=19 already verified 2026-05-16).
2. **P2 baseline:** the Ljung-Box p-values at A0 (battery), A1 (microbiome), A4 (V-Dem) show the predicted monotone drop (high → mid → low). Even with only 3 points this is a directional check on the pipeline.
3. **Pipeline hygiene:** the omega module accepts engine output cleanly; no schema mismatches; reproducible from clean checkout.

**Cost:** $0 (vendored NASA data + cached QoG/V-Dem; public APIs only if a refresh is needed).
**Gate:** if Phase 0 passes, write `EXP2_PREREGISTRATION.md` locking the rest. If Phase 0 fails (e.g., omega doesn't produce sane Ljung-Box on Tier-1), debug the pipeline before pre-registering and before any new-substrate engine work.

**Phase 0 entry point:**

```bash
python3 experiments/exp2_cross_substrate/phase0_tier1_revalidation.py
# Produces: data/phase0_tier1_results.json with per-substrate Ljung-Box p
# + Spearman correlation of (p-value, agency_rung) across the available Tier-1 points.
```

### Phase 0 v0.7 finding (2026-05-16) — CIRIS A3 + WGI A4 added, real Tier-1 WEAK_PASS

**v0.7 implementation:**

| v0.7 fix | Status |
|---|---|
| CIRIS A3 substrate added | ✓ `collect_ciris_a3_samples()` extracts per-chain (k, ρ, σ) from 1,255 chains across Gemini + qwen + scout cross-family cohorts. k = count of fired CONSCIENCE+DMA scalar scores. σ = mean of those scores. ρ = within-chain consensus = max(0, 1 − 2·std(scores)) — NOT IDMA's saturated `correlation_risk` (which is constant 0.95 across the pipeline) |
| WGI A4 substrate added | ✓ `collect_wgi_samples()` ingests `data/institutional/wgi_processed.csv` (4,933 country-year obs with pre-computed k/ρ/σ from the original CCA run). Symlinked, gitignored. |
| Engine-aware AR(1) metric | ✓ retained from v0.6 |
| Trajectory windows (battery) | ✓ retained from v0.6 |
| Real AGP cohort | ✗ still absent from disk; synthetic flagged as i.i.d.-artifact |

**Phase 0 v0.7 results:**

| Run | Spearman ρ(rung, AR(1) \|φ\|) | Verdict |
|---|---|---|
| **Positive control** (5 synthetic rungs A0–A4) | **+1.000** (p = 1.4 × 10⁻²⁴) | **STRONG_PASS** |
| **Real Tier-1** (5 substrates × 4 rungs) | **+0.359** (p = 0.553) | **WEAK_PASS** |

**Per-substrate results:**

| Substrate | n | Rung | AR(1) \|φ\| | Notes |
|---|---|---|---|---|
| battery (NASA Li-ion windows) | 5 | **A0** | **0.467** | Real physical autocorrelation; n=5 limits AR(1) precision |
| microbiome (synthetic generator) | 300 | **A1** | **0.071** | Known i.i.d. artifact — flagged in v0.6 |
| CIRIS chains (3 model families) | 1,255 | **A3** | **0.600** | Real LLM reasoning structure; faculty consensus varies meaningfully |
| polity (Polity5 country-decade windows) | 725 | **A4** | **0.301** | Decade-window averaging dampens temporal autocorrelation |
| wgi (WGI country-year sequence) | 4,933 | **A4** | **0.956** | Year-level sequence preserves heavy temporal autocorrelation |

**Critical new finding — A4 substrate-pair disagreement:**

Polity (A4) and WGI (A4) are both A4 substrates per the intrinsic agency-ladder operationalization, but report wildly different |φ|: 0.30 vs 0.96. The difference is NOT agency rung — they are both A4. The difference is **temporal sampling resolution:**

- WGI is per-country-YEAR (sequential observations every year, autocorrelation natural)
- Polity is per-country-DECADE-WINDOW (averaging breaks year-level autocorrelation)

This tells us the v0.6/v0.7 |φ| metric is *sampling-resolution-sensitive*. Two same-rung substrates with different sampling windows produce different |φ|. Pre-registration must lock window sizes uniformly across substrates, OR the metric must normalize for sampling resolution.

**Hypothesis status update:**

| Hypothesis (from v0.5/v0.6) | Status |
|---|---|
| ❌ Pipeline bug | Falsified by positive control (ρ = +1.000) |
| ❌ Trivial-mean predictor causing inversion | Falsified by v0.5/v0.6 |
| ❌ Synthetic microbiome zeros out A1 | Confirmed contribution but no longer the sole blocker — pattern shows in 5-substrate test |
| ⚠️ Sample-size sensitivity of Ljung-Box | Partly addressed by AR(1), but still relevant when n differs by 1000× across substrates |
| ⚠️ **Temporal sampling resolution dominates \|φ\|** | **New v0.7 finding: A4 substrate-pair Polity vs WGI disagree by 0.66 due to year-vs-decade windowing** |
| ⚠️ P2 prediction sign-reversed | Open but less likely — direction is now positive |
| ⚠️ P1 fit flat | Still open — battery 0.13, microbiome 0.0001, ciris 0.48, polity 0.02, wgi 0.0 |

**v0.7 verdict:** P2 direction is now **positive** in sign (ρ = +0.359). The pipeline reliably distinguishes synthetic structured residuals (positive control ρ = +1.000 across 5 rungs). Real-data Tier-1 partially supports P2 monotonicity but is dominated by sampling-resolution effects, not agency-rung effects.

**Required v0.8 fixes:**

1. **Sampling-resolution normalization** — either (a) match window sizes across substrates (all year-level, or all decade-level), or (b) compute |φ| at multiple lags and report the *rate of decay*, which is more sampling-invariant than lag-1 itself.
2. **CIRIS A3 cross-validation** — confirm the 0.600 |φ| holds when computed on each model cohort separately (qwen-only, scout-only, gemini-only) — if it varies a lot across models, the A3 datapoint is unstable.
3. **Real AGP cohort at A1** — finally close the v0.6 blocker.
4. **P1 R² investigation** — why does the cross-sample Kish regression fail (R² near zero) on most substrates? Re-examine whether σ should vary with k_eff cross-sample, or whether the framework's "Kish fits substrates" claim was meant within-substrate (CCA-paper-style engine fits).

### Phase 0 v0.6 finding (2026-05-16) — superseded by v0.7

**v0.6 implementation status:**

| v0.6 fix | Status |
|---|---|
| 1. Sample-size-invariant whiteness metric | ✓ `ar1_coefficient(ω)` lag-1 autocorrelation magnitude added to `analysis/omega/kish_fit.py` |
| 2. Trajectory-window battery sampling | ✓ replaces v0.5 bootstrap; non-overlapping `window=5, stride=5` (5 windows from 19 cells × 28 cycles) |
| 3. Real microbiome cohort (AGP) | **✗ AGP raw not on disk anywhere**; synthetic generator still used. **THIS IS THE BLOCKER for clean P2.** |
| 4. V-Dem CSV vendored | ✓ substituted Polity5 (also A4, more complete: 17,574 country-year obs). Symlinked at `data/institutional/polity5.xls`, SHA `f81248561c…`, 4.3 MB. New `collect_polity_samples()` produces n=725 country-decade windows. |
| 5. Pre-register metric + sample-design constraint | Pending — depends on v0.7 with real AGP |

**Phase 0 v0.6 results (commit pending):**

| Run | n substrates | Spearman ρ(rung, AR(1) \|φ\|) | Verdict |
|---|---|---|---|
| **Positive control** (5 synthetic rungs A0–A4, AR(1) φ = 0.0–0.85) | 5 | **+1.000** (p = 1.4 × 10⁻²⁴) | **STRONG_PASS** |
| **Real Tier-1** (battery A0 trajectory-windows n=5, microbiome A1 synthetic n=300, polity A4 n=725) | 3 | **−0.500** (p = 0.667) | **FAIL_DIRECTION** |

**Per-substrate breakdown:**

| Substrate | n | AR(1) \|φ\| | Interpretation |
|---|---|---|---|
| A0 battery (5-cycle non-overlapping windows) | 5 | **0.467** | Real physical residual structure (SEI growth continuity + small n); n is genuinely too small for stable AR(1) estimate |
| A1 microbiome (synthetic generator) | 300 | **0.071** | I.i.d. by construction — generator produces independent samples; AR(1) of i.i.d. data → ~0 |
| A4 polity (Polity5 country-decade) | 725 | **0.301** | Real human-decision autocorrelation (regime trajectories persist across decades) |

**The failure mode is interpretable, not pipeline-driven:**

The positive control passes perfectly with the v0.6 metric (Spearman = +1.000 across 5 rungs of constructed AR(1) data). The pipeline correctly distinguishes white from structured residuals at all sample sizes.

The real-data fail is now traceable to **one specific data-availability gap**: synthetic microbiome is mathematically i.i.d. and zeros out the A1 |φ| signal. Battery has small n + real physical autocorrelation; polity has real human-decision autocorrelation. Without real AGP cohort data at A1, the test fundamentally cannot distinguish "framework predicts A0 < A1 < A4" from "sampling mathematically forces A1 to zero."

### Required v0.7 fix (the last blocker)

| Move | What it does |
|---|---|
| **Vendor AGP raw data** | Replace synthetic microbiome with real American Gut Project sample cohort. Real cross-host variation in (k, ρ, σ) gives A1 a fair shot at producing the framework's predicted residual structure. |
| Alternative: real HMP data | Human Microbiome Project — also A1, public, comparable scale |
| Alternative: real BioTIME data | Move A1 to A2 substrate (BioTIME ecology), wait for AGP later |

Until A1 has real data, Phase 0 cannot make the P2 direction test informative. **Pre-registration remains blocked, but for one specific reason: data, not methodology.** The v0.6 metric, sampling design, pipeline, and lake formalization are all sound.

### Phase 0 v0.5 finding (2026-05-16) — pipeline validated, sample-design issue identified (resolved by v0.6)

**With engine-aware Kish-regression predictor + 5-rung synthetic positive control:**

| Run | Spearman ρ(rung, ljung-box p) | Verdict |
|---|---|---|
| **Positive control** (synthetic, AR(1) noise φ = 0.0/0.2/0.45/0.7/0.85 across rungs A0–A4) | **−1.000** (p < 10⁻²³) | **STRONG_PASS** — pipeline correctly distinguishes white from structured residuals across the agency ladder |
| **Real Tier-1** (battery A0 n=40, microbiome A1 synthetic n=300) | **+1.000** (p = NaN, n=2) | **FAIL_DIRECTION** — the two available real substrates do not show the predicted ordering |

**What this tells us:**

| Hypothesis | Status |
|---|---|
| ❌ "Pipeline has a bug" | Falsified — positive control passes perfectly |
| ❌ "P2 needs a trivial-mean predictor" (v0.4 hypothesis) | Falsified — Kish-regression predictor still gives wrong direction on real data |
| ⚠️ **"Sample design contamination"** | **Open** — battery uses bootstrap of correlated cells (retains cross-cell ρ that *isn't* the framework's intended ρ); microbiome uses synthetic generator producing i.i.d. samples by construction. Not commensurable. |
| ⚠️ **"Sample-size sensitivity of Ljung-Box"** | **Open** — battery n=40 vs microbiome n=300; Ljung-Box power differs sharply, so direct p-value comparison across n is *not* apples-to-apples |
| ⚠️ **"P2 prediction may be sign-reversed"** | **Open** — possible that at A0 (inert), real physical coupling (electrochemistry) creates real residual structure, while at higher rungs the structure looks more like additive noise |
| ⚠️ **"P1 fit is near-flat, residual ≈ demeaned σ"** | **Open** — battery P1 R²=0.04, microbiome P1 R²=0.0001. With β ≈ 0, the regression contributes nothing; ω is essentially σ−mean(σ). Need genuine k_eff dependence in σ for the residual to be the framework's residual. |

**Required v0.6 fixes before pre-registration:**

1. **Sample-size-invariant whiteness statistic:** replace Ljung-Box p-value with AR(1) coefficient magnitude (or equivalent sample-size-invariant measure). The framework's prediction is *strength* of residual structure, not *significance against null*, so a coefficient is more honest.
2. **Trajectory-window sampling for battery:** each sample = (k_window, ρ_window, σ_window) where k_window = cells in a time window, ρ_window = correlation during that window, σ_window = mean SOH at window end. This captures σ varying with k_eff over time — the framework's actual setup.
3. **Real (not synthetic) microbiome cohort:** AGP raw with natural across-host variation in (k, ρ, σ). Synthetic i.i.d. samples don't expose the structure the framework predicts.
4. **V-Dem CSV vendored locally:** so an A4 substrate is in the comparison; n=2 doesn't give meaningful Spearman significance.
5. **Pre-register the metric AND the sample-design constraint** in `EXP2_PREREGISTRATION.md` before running real-data analysis.

**Why the positive control matters:**

The positive control proves the pipeline can detect the framework's predicted ordering when data conforms to the framework's predicted structure. This shifts the burden of explanation: if Tier-1 data also conformed (and was sampled correctly), Phase 0 would show STRONG_PASS. That it doesn't means the **operationalization** is incomplete, not that the framework is falsified.

This is a meaningful Phase 0 outcome. We've eliminated pipeline-bug and predictor-choice as confounds; we've isolated three remaining hypotheses about sample design and metric choice that v0.6 must resolve before pre-registration.

### Phase 0 v0.4 finding (2026-05-16) — pipeline ordering issue surfaced (resolved by v0.5)

Phase 0 was run with the battery (A0, NASA Li-ion concatenated detrended trajectories) and microbiome (A1, synthetic Shannon cohort of n=300) substrates. V-Dem (A4) is awaiting source-data vendoring.

| Substrate | Rung | Ljung-Box p (lag 10) | Interpretation |
|---|---|---|---|
| battery | A0 | **0.0000** | heavily structured residual |
| microbiome | A1 | **0.0938** | nearly white |

Direction is **inverted** relative to P2's prediction (which expects A0 whitest, A1 less white). Spearman ρ(rung, ljung_box_p) = +1.0 (wrong sign).

**Root cause (diagnostic, not a framework falsification yet):** Phase 0 currently uses `predictor='mean'`, which produces:
- For battery (decaying time series): residual = σ − mean(σ), dominated by the un-captured aging trend → spurious autocorrelation.
- For microbiome (cohort cross-section): residual = σ − mean(σ), genuinely the cross-host variation → nearly white by construction.

These are not the same residual. The framework's P2 requires the residual to be $\omega = \sigma_{\text{observed}} - \sigma_{\text{Kish-predicted}}$, where $\sigma_{\text{Kish-predicted}}$ comes from the substrate's engine (the Kish formula applied to that substrate's $(k, \rho)$), not from a trivial mean baseline.

**Resolution required before pre-registering Exp 2:**

1. **Engine-aware predictor:** add `predictor='engine'` (or per-substrate-specific predictor) to `compute_omega_series`, accepting a callable that runs the substrate's engine to produce $\sigma_{\text{Kish-predicted}}(k, \rho)$. The omega residual is then strictly framework-predicted, not naive-mean-predicted.
2. **Comparable units:** decide whether P2's residual is computed across time (within-substrate time series) or across constituents (cross-section). The choice must be uniform across substrates or the test compares incommensurable structures.
3. **Re-run Phase 0** with the engine-aware predictor on battery + V-Dem (canonical examples: A0 inert vs A4 high-agency). Only then is the P2 direction check meaningful.

This is exactly the kind of pipeline issue Phase 0 is meant to catch. Caught it on the Tier-1 baseline before propagating to 4 new substrates and a pre-registration commit. **This is the right kind of failure.**

---

## Execution sequence (v0.4)

| Step | Status |
|---|---|
| 0a. Lock regime v0.3 | ✓ commit `a93fd58` |
| 0b. Lake locks P1/P2/P3 + Inv-1..Inv-5 | ✓ `Exp2Predictions.lean` (272 lines, all theorems proved) |
| 0c. Stub 4 new-substrate engines + `data_fetch.py` | ✓ skeletons committed (raise NotImplementedError until Phase 1) |
| **0d. Cherry-pick CCA Tier-1 rig to master** | **✓ commit `2573149` (2026-05-16) — loaders + omega + run scripts + design docs** |
| **0e. Update regime to v0.4 (CRCv2 reframe + P2 load-bearing)** | **✓ this commit** |
| 0f. Write Phase 0 `phase0_tier1_revalidation.py` | Next |
| 0g. Run Phase 0 + record baseline R² + Ljung-Box per Tier-1 substrate | Next |
| 1. Pre-register `EXP2_PREREGISTRATION.md` + commit-hash lock | After 0g (gates new-substrate engine work) |
| 2. Implement 4 new-substrate loaders + engines (mirror `battery_loader.py` shape) | After 1 |
| 3. Pin SHA-256 + version for each new substrate in `data_sources.yaml` | After 2 |
| 4. CI `substrate_revalidation.yml` activates (cron + workflow_dispatch) | After 3 |
| 5. Run Exp 2 once Phase 1 of Exp 1 lands | After Exp 1 cross-family + after 4 |
| 6. Paper §10 Exp 2 + Zenodo data release | After 5 |
