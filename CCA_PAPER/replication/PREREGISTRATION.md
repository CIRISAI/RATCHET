# Pre-registration — CCA §11 institutional re-scoring
Written 2026-07-31 BEFORE any outcome variable was computed. Only dataset schema and
country-name coverage were inspected prior to writing this file.

## 1. Country set

The 14 candidates of `scripts/test_institutional_collapse.py:43-65` (`CASE_STUDY_COUNTRIES`),
the only extant country list in the repository:

Venezuela, Türkiye, Hungary, Poland, Zimbabwe, Syria, Libya, Yemen, Tunisia, Egypt,
Ukraine, Germany, Canada, Australia.

The paper reports n=13 and does not name the dropped country; the QoG file that decided
availability is absent from disk. Rule: score all 14, and report the leave-one-out range
over all 14 possible 13-country subsets.

## 2. Primary outcome — PO-1: sustained Regimes-of-the-World category decline

Source: V-Dem v15, `v2x_regime` (RoW), 0=closed autocracy, 1=electoral autocracy,
2=electoral democracy, 3=liberal democracy. File `data/institutional/vdem/v-dem-v15.parquet`.

Y_c = 1 iff there exists a year t in [2001, 2022] with
  R_c(t) <= R_c(2000) - 1  and  R_c(t') <= R_c(2000) - 1 for all t' in {t, t+1, t+2},
all three years observed within 2000-2024. Event date T_c := that first qualifying t.
Otherwise Y_c = 0. Declines beginning 2023 or 2024 cannot be verified for three years and
are scored 0 (conservative; guards against right-censoring).

Justification for primary status: RoW is the field-standard categorical regime measure and
is published for the express purpose of dating regime transitions (it is the measure under
which V-Dem reclassified Hungary in 2019). It is categorical, so "collapse" is a discrete
dated event matching the paper's binary framing and permitting a timing comparison. V-Dem
v15 covers 2000-2024 in full — the paper's stated window — whereas Polity5 on disk ends in
2018/2020 and cannot cover it. The three-year persistence requirement removes single-year
coding flicker. The 2000 baseline is the paper's own stated analysis start.

## 3. Pre-registered sensitivity outcomes (reported regardless of result)

- PO-2: `v2x_libdem` falls >= 0.10 (absolute) below its 2000 value and stays there >= 3
  consecutive years, within 2000-2024.
- PO-3: Polity score (`e_p_polity`) falls >= 3 points below its 2000 value and stays there
  >= 3 consecutive years, within the Polity coverage window.
- PO-4: PO-1 with the persistence requirement set to 1 year and to 5 years.
- PO-5: `v2x_libdem` falls >= 25% (relative) below its 2000 value, sustained >= 3 years.

## 4. Predictor — the CCA classifier, as implemented

`ratchet.engines.institutional.InstitutionalCollapseEngine`, initialised from the country's
year-2000 state and run 20 years with alpha=0.01, d=0.02, noise_sigma=0.005, seed=42
(`scripts/test_institutional_collapse.py:93-99,118`). Prediction P_c = 1 iff
`engine.is_collapsed()` within the run. Predicted event year = 2000 + collapse_time.

Initial state per `ratchet/data/institutional_loader.py:318-388` priority order:
sigma = `vdem_polyarchy` = `v2x_polyarchy`; f = `vdem_corr` = `v2x_corr`; k from Polity
`xconst`; rho = v2x_corr * (1 - k); lambda from WGI rule-of-law. (QoG's `vdem_polyarchy` /
`vdem_corr` are V-Dem's `v2x_polyarchy` / `v2x_corr` verbatim, so the QoG-derived initial
state is recoverable from the V-Dem parquet.)

Noted before scoring, from code inspection: the collapse rule
(`ratchet/engines/institutional.py:292-299`) is sigma < 0.2 or f > 0.8, and the step
function applies country-independent linear drifts, so P_c is a deterministic function of
(sigma_0, f_0) alone. k, rho and k_eff do not enter the decision.

## 5. Nulls and the skill criterion

- N-1 majority-class constant classifier.
- N-2 always-flag and never-flag floors.
- N-3 base-rate-matched permutation null: 100,000 random relabelings holding the number of
  positive predictions fixed; one-sided p on balanced accuracy and MCC.
- N-4 Fisher exact test on the 2x2 table.

Skill is credited only if the permutation p < 0.05 AND the classifier beats the
majority-class floor on balanced accuracy. Anything short of both is reported as no
demonstrated skill.

## 6. Timing claim

"Mean timing offset: 7.6 years early" and "Turkey 2016: flagged within 3 years" are checked
as mean(T_predicted - T_observed) over countries with both a prediction and an observed
event under PO-1. Sign convention: negative = early.

## 7. Commitments

The definitions above are frozen. Results are reported as they fall, including for Hungary
and Poland. No outcome definition will be amended after seeing the confusion matrix; any
post hoc definition that is explored will be labelled post hoc and excluded from the
headline score.
