# Pre-registration — Exp 103 replication and adjudication (RATCHET#13)

**Written 2026-07-31 BEFORE any measurement was taken on this machine.**
Only the source of `exp103_software_injection.py`, `expF2_causality_test.py`, and
`exp117_entropic_potential.py` was read prior to writing this file. No run has been executed.

Hardware: NVIDIA GeForce RTX 4090 Laptop GPU, 16376 MiB, idle at 64 °C, 0% utilization.

## The contradiction to adjudicate

| Source | Barrier sync | Lockstep |
|---|---|---|
| Exp 103 | ρ = 0.90 | ρ = 1.0, k_eff = 1.0 |
| F2 | ρ 0.171 → 0.0495 (k_eff *rises* 9.48 → 15.63) | — |
| exp117 | ρ 0.394 → 0.041 | — |
| F3 | — | ρ = 0.0036 |

Exp 103 grounds the claim "software alone can induce coherence collapse."

## Two candidate explanations, identified from source before running

**H1 — Lockstep is a measurement artifact.** `measure_synchronized_lockstep`
(`exp103_software_injection.py:127-128`) computes a single elapsed time `t1 - t0` for the whole
batch and assigns that identical scalar to all 64 sensors:
`all_timings[:, s] = t1 - t0`. If so, `np.corrcoef` is being applied to 64 identical rows and
ρ = 1.0 is arithmetic, not measurement — the same class of defect as C-5/C-11.

**H2 — Barrier correlation is common-mode, not coupling.** `measure_synchronized_barrier`
(`:70-82`) is a genuine per-sensor measurement, but it synchronizes globally once per sample
index `s` and then measures all 64 sensors sequentially inside that sample. Every sensor in
sample `s` therefore shares one GPU state (thermal, clock, scheduler). Correlation across the
sample dimension would then reflect a shared measurement epoch rather than sensor coupling.

**Estimator note (established from source, not a hypothesis).** Exp 103 and exp117 use mean
**absolute** correlation `mean(|r|)`; F2 uses mean **signed** correlation `mean(r)`. The
absolute form has a strictly positive finite-sample floor. This cannot by itself explain
Exp 103 vs exp117, which share the estimator and still disagree, but it is confounded with
the F2 comparison and must be separated.

## What will be measured

All four Exp 103 modes (independent, barrier, shared workload, lockstep), at the shipped
default of 64 sensors, at two sample counts: **30** (the shipped default) and **200** (to
reduce finite-sample bias). For each mode and sample count:

1. **ρ_signed** = mean of upper-triangle `corrcoef`
2. **ρ_abs** = mean of |upper-triangle| — the Exp 103 / exp117 estimator
3. **ρ_abs_null** = the shuffle null: each sensor's series independently permuted, destroying
   cross-sensor structure while preserving each marginal, then ρ_abs recomputed. 200 draws;
   report mean and 95th percentile. **No prior version of any of these experiments computed a
   null.**
4. **ρ_abs_detrended** = ρ_abs after subtracting, at each sample index, the across-sensor mean
   (common-mode removal). This is the H2 discriminator.
5. **identical_rows** = whether all 64 sensor rows are byte-identical (the H1 discriminator)

## Registered predictions and their interpretation

| # | Prediction | If confirmed | If refuted |
|---|---|---|---|
| P1 | Lockstep yields ρ_abs = 1.0 to floating-point exactness **and** `identical_rows = True` | H1 holds: the lockstep result is an artifact of assigning one scalar to 64 sensors. Withdraw it. | Lockstep is a real measurement; Exp 103 survives on this leg and F3 must be re-examined |
| P2 | Barrier ρ_abs is substantially above ρ_abs_null | Barrier correlation is real signal, not estimator floor | Barrier ρ is a finite-sample artifact of `mean(\|r\|)` at n=30; withdraw |
| P3 | Barrier ρ_abs_detrended falls to near ρ_abs_null | H2 holds: barrier correlation is a shared measurement epoch, not sensor coupling. The "software induces collapse" reading does not follow. | Barrier correlation survives common-mode removal → genuine coupling; Exp 103's reading is supported and F2/exp117 need explaining |
| P4 | ρ_abs at n=200 < ρ_abs at n=30 for every mode | Confirms a finite-sample positive bias in the shipped estimator, affecting every ρ this program has reported | No material bias at n=30 |
| P5 | ρ_signed ≪ ρ_abs for independent and barrier modes | The F2-vs-Exp103 gap is partly estimator choice | Estimator choice is not the F2 discrepancy |

## Decision rule, fixed in advance

- **P1 confirmed and P3 confirmed** → Exp 103 is withdrawn in full; "software alone can induce
  collapse" is unsupported; RATCHET#13 resolves against Exp 103.
- **P1 confirmed, P3 refuted** → lockstep withdrawn, barrier retained as genuine coupling; the
  claim survives in weakened form and the contradiction with F2/exp117 remains open.
- **P1 refuted** → the reading of the source is wrong; report that and stop, since the
  remaining analysis rests on it.

"Substantially above" in P2 means: outside the null's 95th percentile. "Falls to near" in P3
means: within the null's 95th percentile.

## Threats to validity, named in advance

- Thermal state drifts during a run; modes are measured in sequence, so a mode measured later
  runs on a warmer GPU. Mitigation: report GPU temperature before and after each mode; do not
  interpret small between-mode differences.
- This is a **laptop** 4090 (16 GiB, mobile power/thermal envelope), not the desktop 4090 the
  original series may have used. A failure to reproduce Exp 103's numbers is therefore not by
  itself proof the original was wrong — but P1 and P3 are structural and do not depend on
  reproducing its magnitudes.
- Other processes may contend for the GPU. Mitigation: record utilization before starting;
  abort if non-zero at baseline.

## What will be reported

The full table for every mode × sample count × estimator, including nulls, whether or not it
favours any prior result — and the raw JSON, so the numbers are checkable rather than
received.
