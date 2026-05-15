# Exp 2 — Cross-Substrate Extension: Regime Planning

**Status:** v0.1 draft (NOT pre-registered yet).
**Paper hook:** Coherence Substrate Synthesis paper §10 Exp 2.
**Falsification handle:** F-7 (cross-substrate mapping failure).
**Pairs with:** `ratchet/engines/{battery,institutional,microbiome}.py` Tier-1 validations.
**Decision rule (locked at pre-reg, draft here):** $R^2 > 0.7$ in all three new domains → PASS; 2 of 3 → PARTIAL; ≤ 1 → FAIL.

---

## What Exp 2 tests

The Kish formula $k_{\text{eff}} = k/(1 + \rho(k-1))$ already fits at the Tier-1 substrates RATCHET ships with engines for: NASA Li-ion batteries (RMSE 8.1%), QoG/Polity-V institutions (5/5 TN, 7.6yr early-detection bias), AGP microbiome (qualitative distributional fit). These are validated.

Exp 2 takes the formula to **three substrates we don't yet have engines for**, picked deliberately to be *as different from each other as possible*:

| Substrate | Why this domain? |
|---|---|
| Protein folding (AlphaFold) | Molecular scale, no time series, no agent-like behavior — purely topological |
| Neural firing (Allen Brain Atlas) | Cellular scale, dense time series, biological information processing |
| Macro-ecology (BioTIME) | Population scale, environmental dynamics, no central controller |

If all three fit at $R^2 > 0.7$, that's structural evidence the Kish dynamic is substrate-independent. If even one fails, F-7 falsifies the substrate-independence claim at the *structural-mapping level* (the claim weakens to "validated for the substrates we've tested, with no general fit guarantee").

---

## Per-domain operationalization (draft)

The Kish formula relates raw constraint count $k$ and pairwise correlation $\rho$ to effective dimensionality $k_{\text{eff}}$. The **structural prediction** is that $k_{\text{eff}}$ tracks some domain-specific sustainability metric $\sigma$. Per-domain mapping:

### Protein folding

| Variable | Operational definition | Source |
|---|---|---|
| $k$ | Sequence length (residue count) of a domain or single-domain protein | AlphaFold DB residue count |
| $\rho$ | Mean pairwise correlation of per-residue B-factors (predicted thermal motion) within the structure | Computed from AlphaFold predicted structure + plDDT covariance |
| $\sigma$ | Mean pLDDT (per-residue confidence) — proxy for structural stability / fold quality | AlphaFold DB |
| Sample size | ~10,000 single-domain proteins from CATH-S40 representative set | Public |
| Predicted relationship | High $k_{\text{eff}}$ (low $\rho$ across many residues) → robust fold → high pLDDT | $R^2$ on $\sigma = f(k_{\text{eff}})$ |

**Plausibility:** highly-correlated residue motions (rigid-body domains) and short-sequence proteins both produce low $k_{\text{eff}}$ — those proteins should be more sensitive to mutations and have characteristic pLDDT patterns. This is the testable structural prediction.

### Neural firing

| Variable | Operational definition | Source |
|---|---|---|
| $k$ | Number of simultaneously-recorded neurons in a session | Allen Brain Observatory Neuropixels |
| $\rho$ | Mean pairwise spike-train correlation (1-ms bins, full session) | Computed from raw spike times |
| $\sigma$ | Population-decoding accuracy for visual stimulus identity (cross-validated linear classifier on spike-count vectors) | Computed on stimulus-response pairs |
| Sample size | ~80 recording sessions × 5 visual areas | Public via AWS Open Data |
| Predicted relationship | Higher $k_{\text{eff}}$ (more independent neurons) → better information capacity → higher decoding accuracy | $R^2$ on $\sigma = f(k_{\text{eff}})$ |

**Plausibility:** the prediction "$k_{\text{eff}}$ tracks information capacity" maps directly onto well-established population-coding theory (Averbeck, Latham, Pouget, *Nature Reviews Neuroscience*, 2006). If RATCHET's Kish formula tracks decoding accuracy as $k_{\text{eff}}$ predicts, that's a non-trivial replication of an independent line of neuroscience research using RATCHET's substrate-independent framing.

### Macro-ecology

| Variable | Operational definition | Source |
|---|---|---|
| $k$ | Species count in a community time series | BioTIME global biodiversity database |
| $\rho$ | Mean pairwise correlation of species-abundance time series within the community | Computed |
| $\sigma$ | Inverse CV of total biomass over time — ecosystem stability proxy (low CV = stable, biologically meaningful) | Computed |
| Sample size | ~500 community time series with ≥ 10 years of data and ≥ 5 species | BioTIME public dump |
| Predicted relationship | Higher $k_{\text{eff}}$ (more independent species, lower covariance) → ecosystem stability buffered by independent population dynamics | $R^2$ on $\sigma = f(k_{\text{eff}})$ |

**Plausibility:** "diversity stabilizes ecosystems" is well-established (Tilman 1996, Yachi & Loreau 1999, Loreau & de Mazancourt 2013 — the *insurance hypothesis*). The Kish formula gives a substrate-independent functional form for *why* — independent species variance averages out covariance-weighted. **Different framing of an existing empirical pattern, not a new claim.** If $R^2 > 0.7$, RATCHET-style coherence collapse aligns with the ecological literature's mechanism.

---

## Decision rule (draft, will lock at pre-reg time)

Per pre-locked thresholds in the paper §9 (F-7):
- **PASS:** all 3 substrates achieve $R^2 > 0.7$ on the Kish formula fit
- **PARTIAL:** exactly 2 of 3 pass; one diverges — substrate-specificity in the failed domain
- **FAIL:** ≤ 1 of 3 pass; structural-mapping substrate-independence falsified

Each substrate's $R^2$ is computed from the regression of observed $\sigma$ on $k_{\text{eff}}$ across the sample, with 95% bootstrap CI on $R^2$ (10,000 resamples, deterministic seed).

The PASS window is *not* on a single statistic but a triple — same partition discipline as Exp 1's K-count.

---

## Cost + dependencies

| Cost item | Estimate |
|---|---|
| Compute | Free — local GPU/CPU sufficient for all three (PCA on ~$10^4$ proteins, decoding on ~80 sessions, regression on ~500 ecosystems) |
| Data | All public, free, no API quotas |
| Storage | ~50 GB downloaded data |
| Dev time | ~3 weeks (1 per substrate: dataset wrangling + engine implementation + analysis) |
| Direct $ cost | **~$0** (vs. ~$210 for Exp 1) |

Exp 2 is the cheapest Phase 1-class experiment in the suite. No API calls, no model rental, no participant payments. **Just public-data analysis with formal pre-registration.**

---

## Open questions before pre-registration

| # | Question | Why it matters | Tentative resolution |
|---|---|---|---|
| 1 | Should we pre-filter proteins (e.g., exclude membrane proteins) to control for fold-class effects? | Membrane proteins have characteristic pLDDT distributions that may confound | **No** — pre-registered prediction is across the full population. Sub-class analyses are exploratory. |
| 2 | Allen Brain Atlas has multiple stimulus types (drifting gratings, natural images, etc.). Pool or separate? | Pooling may dilute the signal; separating creates a multiple-comparison problem | Pre-register the single most data-rich stimulus type (drifting gratings) |
| 3 | BioTIME has heterogeneous taxonomic coverage (some are fish, some plants, some insects). Standardize? | Heterogeneity may inflate $\rho$ artifactually if taxa are pooled | Pre-register by taxonomic group separately, with a meta-analysis combining |
| 4 | Bootstrap on $R^2$ or on raw $(k, \rho, \sigma)$ triples? | Different null distributions | Bootstrap on $(k, \rho, \sigma)$ triples within each substrate (preserves real-world correlation structure) |
| 5 | Should the existing Tier-1 substrates (battery/QoG/AGP) be re-fit alongside as a "positive control"? | Confirms the Kish formula machinery works on known-fitting data | **Yes** — fit all 6 substrates uniformly, but only Exp 2's 3 new substrates count for F-7 decision |

---

## Provisional execution plan

| Week | Task |
|---|---|
| 1 | Download + checksum + scrub all three public datasets. Vendor into `experiments/exp2_cross_substrate/data/` (or git-LFS for size). |
| 2 | Implement `ratchet/engines/{protein,neural,ecological}.py` — same shape as `battery.py`. |
| 2 | Pre-register `EXP2_PREREGISTRATION.md` + `formal/RATCHET/Experiments/Exp2Predictions.lean` (mirrors Exp 1's structure). |
| 3 | Run analyses, compute per-substrate $R^2$ + 95% bootstrap CI, apply locked decision rule. |
| 4 | Paper §10 Exp 2 update + companion data release on Zenodo. |

---

## What this experiment is NOT

| Not | Reason |
|---|---|
| Not a new physics claim | The Kish formula is established (Kish 1965, survey sampling). |
| Not a deep-learning result | Public data + simple statistics — defensible without GPU clusters. |
| Not a substrate-specific contribution | The point is *invariance* across substrates. Each individual fit is unsurprising in isolation. |
| Not load-bearing on Tier 3 | Tier 3's parsimony argument inherits from Tiers 1+2; Exp 2 strengthens Tier 1 without addressing Tier 3 directly. |

---

## Sign-off needed before locking

The user-facing decisions before this becomes a real pre-registration:

1. Accept the three operationalizations above (k/ρ/σ per substrate), OR propose substitutes
2. Confirm the $R^2 > 0.7$ threshold across all three (vs. a weaker majority threshold)
3. Confirm the 5 open-questions resolutions in §"Open questions" above
4. Decide whether to vendor the data into the repo or use git-LFS / external links (size: ~50 GB total)
