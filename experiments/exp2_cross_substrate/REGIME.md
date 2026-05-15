# Exp 2 — Substrate Fractality Across Agency Levels: Regime

**Status:** v0.3 (regime locked; pre-registration pending engine implementation + dataset SHA pins).
**Paper hook:** Coherence Substrate Synthesis paper §10 Exp 2 (renamed from "Cross-substrate extension").
**Falsification handle:** F-7 (cross-substrate mapping failure), strengthened with F-7b (residual-structure agency conditional).
**Pairs with:** existing `ratchet/engines/{battery,institutional,microbiome}.py` Tier-1 fits.
**Companions:** Counter-RII consent-gate work (FSD/COUNTER_RII_DETECTION.md) — same construction, different rung.

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

### Primary (P1) — Kish formula fits everywhere

| Substrate | $R^2$ threshold | Bootstrap CI |
|---|---|---|
| AlphaFold | > 0.7 | 95% via 10k resamples |
| Allen neural | > 0.7 | 95% via 10k resamples |
| BioTIME ecology | > 0.7 | 95% via 10k resamples |
| PMU grid | > 0.7 | 95% via 10k resamples |

PASS: all 4 of 4 above threshold. PARTIAL: 3/4. FAIL: ≤2/4.

### Secondary (P2) — Residual is noise at low agency, structured at high

After fitting $\sigma = f(k_{\text{eff}}) + \varepsilon$, test the residual $\varepsilon$ for:
- **Whiteness** (Ljung-Box, spectral flatness): high p-value at A0 substrates, dropping monotonically as we ascend the rungs
- **Cross-constituent covariance** in $\varepsilon$: near-zero at A0, increasing with rung

If P2 holds (residual whiteness correlates negatively with agency rung), the fractal-agency reframe is empirically grounded. If P2 fails (residuals look the same regardless of rung), the reframe was overreach — primary prediction still stands but the interpretation weakens.

### Tertiary (P3) — Pre-collapse Δρ sign tracks agency

Re-derive the CCA paper's pre-collapse Δρ pattern across all 4 new substrates:

| Predicted sign | Substrate |
|---|---|
| − (falls) | AlphaFold (when used in degradation-event context), PMU (pre-fault), Allen (pre-anesthesia) |
| + (rises) | BioTIME ecology (pre-collapse during invasive coordination) |

This is a corroborating prediction, not a falsification handle. If signs go the wrong way, the reframe needs refinement.

---

## Locked decision rule

Primary (P1) drives the headline decision:

| Outcome | Condition | Verdict |
|---|---|---|
| **PASS** | 4/4 substrates achieve $R^2 > 0.7$ | F-7 passes; substrate-independence at structural-mapping level confirmed across A0–A2 |
| **PARTIAL** | 3/4 pass | Substrate-specificity in one domain — note which rung and which substrate |
| **FAIL** | ≤ 2/4 pass | F-7 falsified; structural-mapping substrate-independence is contested |

P2 and P3 are reported alongside, not used for headline pass/fail. They strengthen or weaken the *interpretation* of P1's result.

---

## Per-substrate operationalization

### A0 — AlphaFold residues
| Variable | Definition | Source |
|---|---|---|
| k | Sequence length (residues) of a single-domain protein | AlphaFold DB v6 |
| ρ | Mean pairwise correlation of per-residue B-factor predictions | Computed from pLDDT covariance |
| σ | Mean pLDDT (structural stability proxy) | AlphaFold DB |
| n | ~10,000 CATH-S40 representative single-domain structures | EBI FTP |

### A0 — PMU grid
| Variable | Definition | Source |
|---|---|---|
| k | Number of PMUs reporting in a grid region during an event | PNNL Open PMU Library |
| ρ | Mean pairwise correlation of pre-event frequency time series (5-min baseline) | Computed |
| σ | Inverse of post-event settling-time CV | Computed |
| n | ~1,694 grid events | PNNL-30492 corpus |

### A1 — Allen neural firing
| Variable | Definition | Source |
|---|---|---|
| k | Number of simultaneously-recorded neurons per session | Allen SDK + AWS Open Data |
| ρ | Mean pairwise spike-train correlation (1-ms bins) | Computed |
| σ | Population-decoding accuracy on drifting gratings (cross-validated linear classifier) | Computed |
| n | ~80 Neuropixels recording sessions | Allen Brain Observatory |

### A2 — BioTIME macro-ecology
| Variable | Definition | Source |
|---|---|---|
| k | Species count in a community time series | BioTIME 2.0 |
| ρ | Mean pairwise correlation of species-abundance time series | Computed |
| σ | Inverse CV of total biomass over time (stability) | Computed |
| n | ~500 community time series (≥ 10 years, ≥ 5 species) | BioTIMEr R package + raw |

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

## Execution sequence

| Step | Status |
|---|---|
| 1. Lock regime (this doc) | ✓ v0.3 |
| 2. Stub 4 engines + data_fetch.py | In flight |
| 3. Pre-register `EXP2_PREREGISTRATION.md` + `Exp2Predictions.lean` | Pending engine implementation + dataset SHA pins |
| 4. CI substrate_revalidation.yml workflow | Drafted alongside engines |
| 5. Run Exp 2 once Phase 1 of Exp 1 lands | After Exp 1 results |
| 6. Paper §10 Exp 2 + Zenodo data release | After Exp 2 results |
