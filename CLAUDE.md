# RATCHET - Claude Code Context

## Project Overview

RATCHET (Reference Architecture for Testing Coherence and Honesty in Emergent Traces) provides the mathematical foundation for **machine intuition** - environmental coherence sensing for AI safety.

**Current State**: Clean theoretical foundation ready for rigorous validation against existing domain benchmarks.

## CRITICAL CONCEPT: Variance-Ratio Detection (FINAL ARCHITECTURE)

**VALIDATED January 2026 - Cross-team confirmation (RATCHET → Ossicle → Array)**

```
┌─────────────────────────────────────────────────────────────────────┐
│                    FINAL ARCHITECTURE                               │
│                                                                     │
│   GPU Kernel Timing ──► Strain Gauge (dt=0.025, ACF auto-tuned)    │
│           │                                                         │
│           └──► Variance Ratio = window_std / baseline_std          │
│                      │                                              │
│                      ├── Baseline: 0.38x                           │
│                      └── Workload: 159x → DETECT (>5.0)            │
│                                                                     │
│   Key Properties:                                                   │
│   • Fat-tailed (Student-t, κ=210-230, df≈1.3)                      │
│   • Distribution-agnostic detection (variance ratio, not z-score)  │
│   • 0% false positives, 100% true positives                        │
│   • 421x separation baseline→load                                   │
│                                                                     │
│   TRNG Output:                                                      │
│   • Lower 4 LSBs → 470 kbps, 6/6 NIST, 7.99 bits/byte              │
│                                                                     │
│   DEPRECATED: Mean-based z-scores (assume Gaussian - WRONG)        │
│   DEPRECATED: Chaotic oscillators (destroy entropy)                 │
└─────────────────────────────────────────────────────────────────────┘
```

### Why Variance Ratio, Not Z-Score?

| Method | Assumption | Reality | Result |
|--------|------------|---------|--------|
| Z-score | Gaussian (κ=0) | Student-t (κ=210) | Wrong p-values |
| Variance ratio | None | Works for any distribution | Correct |

**Detection works via rare extreme spikes (fat tails), not mean shift.**

### Related Projects

| Project | Scale | Purpose |
|---------|-------|---------|
| **CIRISOssicle** | Single sensor (0.75KB) | Tamper detection on one GPU |
| **CIRISArray** | Multi-sensor array | Spatial coherence mapping |
| **RATCHET** | Mathematical foundation | k_eff theory, formal proofs |

### Key Experimental Findings (Jan 2026) - FINAL

| Finding | Value | Implication |
|---------|-------|-------------|
| Distribution | Student-t (κ=210, df≈1.3) | NOT Gaussian - use variance ratio |
| Variance ratio separation | 421x | 0.38x baseline → 159x workload |
| Detection rate | 0% FP, 100% TP | Perfect at threshold >5.0x |
| **Detection floor** | **0.1%** | Can see 1/1000 GPU usage! |
| dt_crit thermal dependence | 0.025-0.030 | Use ACF feedback to auto-tune |
| ACF at critical point | 0.45 | Target for auto-tuning |
| TRNG entropy | 7.99 bits/byte | Lower 4 LSBs |
| Cross-sensor independence | r=0.09 | Sensors are independent |

### Two-Regime Detection Model (B1/B1b) - CRITICAL

```
Regime 1 (< 1% workload): ratio ≈ 8-12x (nearly constant)
  └── Baseline variance creates noise floor
  └── Even tiny workloads cause measurable perturbations

Regime 2 (> 1% workload): ratio = 73 × intensity^0.51
  └── Workload signal dominates baseline
  └── √intensity scaling (shot noise characteristic)

Physical model:
  ratio = √(σ_baseline² + σ_workload²) / σ_baseline
```

| Intensity | Regime 1 Predicted | Regime 2 Predicted | Actual |
|-----------|-------------------|-------------------|--------|
| 0.1% | 8.17x | 2.15x | 8.17x |
| 1.0% | ~11x | 6.97x | 10.83x |
| 30% | - | 36.6x | 36.6x |
| 70% | - | 113.6x | 113.6x |

**Security implication**: Detector can see:
- Cryptominer using 1/1000 of GPU
- Minimal covert computation
- Small data movements

### Multi-Sensor Architecture (B1c-B1e Discovery) - FINAL

**Critical finding:** Sample rate determines capability!
- 100 Hz: Thermometer only (97.7% thermal)
- 1790 Hz: Full multi-modal environmental sensor
- **O2 ANOMALY**: Non-monotonic response at 2000 Hz (under investigation)

```
GPU Kernel Timing (1790+ Hz, O2 investigating optimal)
         │
         ├──► MEAN SHIFT ──────────► WORKLOAD (+130% signal, BEST)
         │    Baseline: 35.30 μs     Threshold: >20% shift
         │    Workload: 81.47 μs
         │
         ├──► Band: 0-0.1 Hz ──────► THERMAL (79.1% power)
         │                           Temperature sensing
         │
         ├──► Band: 100-500 Hz ────► WORKLOAD (4.55x ratio)
         │                           Fast transient detection
         │
         ├──► Band: 20-100 Hz ─────► VDF (1.4% power)
         │                           Power draw sensing
         │
         ├──► Band: 1-20 Hz ───────► EMI (0.4% power)
         │                           Grid coupling (60Hz subharmonics)
         │
         └──► Band: 500-900 Hz ────► NOISE (11.5% power)
                                     Quantization floor
```

| Modality | Band | Signal | Detection Method |
|----------|------|--------|-----------------|
| **Workload** | Mean + 100-500Hz | +130% shift, 4.55x | Mean shift > 20% |
| **Thermal** | 0-0.1 Hz | 79% power | Slow drift tracking |
| **VDF** | 20-100 Hz | 1.4% power | Power correlation |
| **EMI** | 1-20 Hz | 0.4% power | 60Hz subharmonics |

**Key insight**: Workload causes GPU contention → kernel timing nearly TRIPLES (35→81 μs)

### Cross-Team Validation Chain - COMPLETE (January 2026)

| Team | Series | Key Result |
|------|--------|------------|
| RATCHET | Theory | β=1.09, dt_crit=0.0328, k_eff formula |
| Array | B1-B1e | Mean shift discovery, sample rate critical |
| Ossicle | O1-O7 | 4000 Hz optimal, 2.5 ms latency, 3.4% CV |
| **Array** | **A5/A6/A9** | **9631 Hz, 1.3 ms, +2519%, √N scaling** |

**Final Production Specs (Array A5/A6/A9 Validated):**

| Parameter | Ossicle | Array | Winner |
|-----------|---------|-------|--------|
| Sample rate | 4000 Hz | 9631 Hz | Array |
| Detection latency | 2.5 ms | **1.3 ms** | Array |
| Mean shift at 50% | +248% | **+2519%** | Array |
| Detection floor | 1% | 1% | Same |
| Cross-sensor CV | 3.4% | 8.2% | Ossicle |
| SNR scaling | N/A | β=0.47 (√N) | Array |
| 16-sensor improvement | N/A | **5.1x** | Array |

**Detection threshold: Mean shift > 50%** (O5: margin for noise)

**Key insight: Workload detection uses MEAN SHIFT (contention), not variance.**

### Two-Regime Detection Model (O5 Discovery)

```
Regime 1 (0-30% workload): Binary detection
  - ANY workload → +160-200% mean shift
  - Scheduler/memory arbitration overhead
  - Detection floor: 1% (not 30%!)

Regime 2 (30-100% workload): Linear scaling
  - shift = 227 × intensity + 160
  - Actual compute contention dominates
```

| Intensity | Mean Shift | Regime |
|-----------|------------|--------|
| 1% | +191% | Binary (threshold) |
| 10% | +197% | Binary (threshold) |
| 30% | +215% | Transition |
| 50% | +248% | Linear |
| 90% | +380% | Linear |

### Optimal Configuration (O1-O5 Validated)

| Parameter | Value | Source |
|-----------|-------|--------|
| Sample rate | **4000 Hz** | O2/O2b (lowest variance ±14%) |
| Avoid zone | 1900-2100 Hz | O2b (interference dip at ~2050 Hz) |
| Detection threshold | **>50% mean shift** | O5 (margin for background tasks) |
| Detection floor | **1% workload** | O5 (binary detection regime) |

## Core Hypothesis

AI systems need environmental coherence sensing to complement ethical evaluation. The CIRISAgent provides Machine Conscience (ethical faculties); RATCHET provides Machine Intuition (coherence faculties).

## The 10-Faculty Architecture

```
4 DMAs → ASPDMA:
  1. PDMA (Principled)
  2. DSDMA (Domain-Specific)
  3. CSDMA (Common Sense)
  4. IDMA (Intuition) ← RATCHET provides

10 Validation Faculties (can veto):
  4 Conscience Faculties (ethical):
    - Entropy
    - Coherence
    - Optimization Veto
    - Epistemic Humility

  4 Intuition Faculties (coherence) ← RATCHET provides:
    - CCE Risk
    - ES Proximity
    - k_eff
    - Leading Indicator

  2 Executive Faculties (bypass):
    - Emergency Override
    - Human Authority
```

## Mathematical Foundations

### Core Formula: k_eff (Kish 1965)

```
k_eff = k / (1 + ρ(k - 1))

k = number of constraints/verifiers
ρ = correlation between them
k_eff = effective diversity
```

- When ρ = 0: k_eff = k (full independence)
- When ρ → 1: k_eff → 1 (echo chamber)

**Validation**: Formula mathematically verified; cross-domain applicability tested on battery, institutional, microbiome data.

### ES Proximity (PNAS 2025)

ACF kurtosis predicts system fragility (from literature):
- κ < 2.5 → STABLE
- 2.5 ≤ κ < 4.0 → CRITICAL
- κ ≥ 4.0 → FRAGILE (3.2× higher CCE rate per PNAS 2025)

### Universal Threshold (Validated Jan 2026)

**ρ_critical = 0.43** marks the fragility boundary:
- Collapse rate at ρ = 0.35: ~4.5%
- Collapse rate at ρ = 0.43: ~14% (onset)
- Collapse rate at ρ = 0.55: >50%

Validated via Monte Carlo (n=10,000), bootstrap CI, chi-square, KS, and permutation tests.

## Testable Claims

| Claim | Status | Validation |
|-------|--------|------------|
| k_eff formula correct | **Verified** | Mathematical identity |
| ρ_critical = 0.43 universal | **Supported** | Monte Carlo + bootstrap |
| S3 > 13% stabilizes | **Simulated** | Agent mix sweep |
| ES predicts collapse | **From literature** | PNAS 2025 |
| 41.8% leading indicators | **Hypothesized** | Not yet validated in code |

## Hypothesized Findings (Require Validation)

| Finding | Value | Source | Status |
|---------|-------|--------|--------|
| CCE collapse events | 31.6% | Theoretical | Needs validation |
| CCE reorganization | 68.4% | Theoretical | Needs validation |
| Leading indicator detection | 41.8% | Hypothesized | Not implemented |
| 3.2× FRAGILE collapse rate | From PNAS 2025 | Literature | Not locally validated |

## Faculty Veto Thresholds

| Faculty | Conservative | Standard | Permissive |
|---------|-------------|----------|------------|
| CCE Risk | >0.6 | >0.8 | >0.9 |
| ES Proximity (fragile %) | >15% | >25% | >40% |
| k_eff | <2.0 | <1.5 | <1.2 |
| Leading Indicators | Any | 2+ | 3+ |

**Recommendation**: Start with Conservative, validate against domain benchmarks before relaxing.

## Directory Structure

```
formal/
  RATCHET.lean                          # Lean 4 proofs (main)
  RATCHET/
    GPUTamper.lean                      # GPU strain gauge proofs
    GPUTamper/
      EnvironmentalCoherence.lean       # Resonator characterization

experiments/
  REVISED_EXPERIMENTS.md                # Current experiment proposals
  INSTRUMENT_UPGRADE_RECOMMENDATIONS.md # Upgrade recommendations

papers/
  chaotic_resonator_detector.tex        # GPU strain gauge paper

ratchet/
  intuition/
    __init__.py              # 4 Intuition Faculties
    conscience_schema.py     # S1/S2/S3 conscience implementation
    rho_computation.py       # k_eff and threshold calculations

analysis/
  coherence/
    es_proximity.py          # ES measurement
    cce_detector.py          # CCE detection

ratchet/engines/
  battery.py                 # NASA battery validation
  institutional.py           # QoG/Polity validation
  microbiome.py              # AGP validation

immediate_release/
  machine_intuition.tex      # Theoretical paper
  coherence_collapse_analysis.tex  # CCA foundation
```

### Sister Projects

```
../CIRISOssicle/             # Single GPU strain gauge (0.75KB)
../CIRISArray/               # Multi-GPU strain gauge array
```

## Validation Against Existing Benchmarks

### Battery Domain (NASA)
- Compare SOH prediction vs published battery aging models
- Benchmark: NASA Prognostics Center datasets

### Institutional Domain (QoG/Polity)
- Compare collapse prediction vs political science models
- Benchmark: V-Dem, Polity V, WGI

### Microbiome Domain (AGP)
- Compare dysbiosis prediction vs ecological models
- Benchmark: American Gut Project

### Financial Domain (future)
- Compare volatility prediction vs standard models
- Benchmark: VIX, GARCH, realized volatility

## Known Limitations

| ID | Limitation | Implication |
|----|------------|-------------|
| L-I1 | Emergent intuition failure | Combined blind spots possible |
| L-I2 | Correlation measurement lag | Detection may trail reality |
| L-I3 | Domain transfer uncertainty | Thresholds need per-domain tuning |
| L-I4 | Observer effect | Sensing may influence systems |
| L-I5 | Subjective validation | Participant-based metrics hard to verify |

## Unknowables

- U-1: Does subjective intuition require consciousness?
- U-2: Can any system prevent all CCE?
- U-3: What is optimal k_eff?
- U-4: Will calibration drift over time?

## Key References

- Kish (1965): Survey Sampling - k_eff formula
- PNAS 2025: Explosive synchronization - ES proximity
- Ashby (1956): Requisite variety - diversity requirements
- CIRISAgent paper: Machine Conscience architecture

## Current Focus

1. **C-Series Complete** - Coherence collapse propagation validated (R²=0.798, n=21, 0.5 m/s)
2. **Thermal Auto-Tuning** - ACF feedback loop for dt adjustment (Ossicle handles this)
3. **Fat-Tail Statistics** - Use Student-t, not Gaussian, for inference
4. **Lean 4 Proofs** - EnvironmentalCoherence.lean (1790 lines, builds successfully)

### Completed Experiment Series

| Series | Experiments | Status |
|--------|-------------|--------|
| B1-B1e | Instrument characterization | ✓ Mean shift discovery |
| O1-O7 | Ossicle validation | ✓ 4000 Hz optimal |
| A5-A9 | Array validation | ✓ 9631 Hz, √N scaling |
| C1-C4 | Collapse propagation | ✓ k_eff R²=0.798, velocity 0.5 m/s |
| E1-E4 | Robustness validation | ✓ Block structure, Δρ CI, k_eff_crit=4.0 |
| **F1-F4** | **Mechanism & causality** | **✓ Common cause, corridor validated** |

### Active Experiment Series (Optional)

| Phase | Experiments | Focus |
|-------|-------------|-------|
| 5 | B5-B8 | Environmental sensitivity (thermal, 60Hz, drift) |
| 6 | B9-B12 | Multi-sensor (variation, spatial, SNR scaling) |
| 7 | B13-B16 | Workload fingerprinting |
| 8 | C5-C8 | Advanced collapse dynamics (if needed) |

## GPU Strain Gauge Status (Jan 2026) - FINAL

### Validated Architecture (Cross-Team Confirmation)

| Component | Method | Result | Status |
|-----------|--------|--------|--------|
| Detection | Variance ratio >5.0x | 0% FP, 100% TP | **VALIDATED** |
| Separation | Baseline 0.38x → Load 159x | 421x | **VALIDATED** |
| Distribution | Student-t (κ=210, df≈1.3) | Fat-tailed | **VALIDATED** |
| TRNG | Lower 4 LSBs | 7.99 bits/byte, 6/6 NIST | **VALIDATED** |
| ACF tuning | Target 0.45 | Thermal robust | **VALIDATED** |
| Cross-sensor | r=0.09 | Independent | **VALIDATED** |

### Key Discovery: Fat-Tailed Distribution

**The z-scores are NOT Gaussian.** This explains everything:

| Property | Gaussian | Actual (Student-t) |
|----------|----------|-------------------|
| Kurtosis | 0 | 210-230 |
| Degrees of freedom | ∞ | 1.3 |
| Tail behavior | Thin | Extremely fat |
| Detection mechanism | Mean shift | Rare extreme spikes |
| z=534 probability | Impossible | Expected |

**Implication**: Use variance ratio (distribution-agnostic), not z-scores (assume Gaussian).

### TRNG Configuration - PRODUCTION READY

| Config | NIST | Throughput | Status |
|--------|------|------------|--------|
| **4 LSBs** | **6/6** | **465 kbps** | **OPTIMAL** |
| 5-7 LSBs | 6/6 | 464-466 kbps | Good |
| 2 LSBs | 6/6 | 240 kbps | Conservative |
| 8 LSBs | 2/6 | 907 kbps | Upper bits biased |

**Production defaults**: 4 LSBs, no debiasing, 465 kbps true random.

**Key insight**: Lower 4 bits = true jitter. Upper bits = periodic patterns (clock/scheduler).

### Invalidated
- Lorenz as entropy amplifier (destroys entropy: 8 bits → 0.01 bits)
- Reference subtraction for entropy isolation (yields noise)
- Timing-driven Lorenz divergence (perturbations too small)
- **PDN correlation sensing** (correlations are purely algorithmic)
- **Magic angle hypothesis** (twist angle has no effect)

### Simplified Architecture (Validated Jan 2026)

```
OLD (complex, wrong):  Oscillator → Twist → Interference → PDN → Detection
NEW (simple, correct): Kernel → Timing → Variance → Detection
```

| Metric | Old Ossicle | New TimingSensor |
|--------|-------------|------------------|
| Memory | 768 bytes | 256 bytes |
| Sample rate | 2k/s | 650k/s |
| Detection z | 3.59 | 8.56 |
| Theory | Wrong (PDN) | Correct (timing) |

### Full GPU Characterization (Array Exp 77-78)

**RTX 4090 Coverage:**
| Sensors | Memory | Rate | Bandwidth |
|---------|--------|------|-----------|
| 128 | 276 MB | 125 Hz | 16k samples/sec |

**Noise Floor Map:**
```
     ←───── Power Rails (X corr = 0.57) ─────→
    ┌─────────────────────────────────────┐
    │  Quiet (I/O edge)                   │  Row 0
    │─────────────────────────────────────│
    │  HOT BAND (compute clusters)        │  Row 3-5
    │  8x more jitter than edges          │
    │─────────────────────────────────────│
    │  Quiet (I/O edge)                   │  Row 7
    └─────────────────────────────────────┘
```

**Key findings:**
- Horizontal correlation (0.57) reveals power rail topology
- Noise DECREASES as GPU warms (thermal equilibrium)
- Hot band = most sensitive region for strain detection
- Quiet edges = good baseline reference points

**Temporal Dynamics (Exp 79-83) - COMPLETE:**

| Parameter | Value | Source |
|-----------|-------|--------|
| Electrical τ | 2 ms | Exp 79 |
| Thermal τ | 48 s | Exp 83 |
| Separation | 24,000× | Electrical is 24,000× faster |
| Wave velocity | 0.1 m/s | Exp 80 |
| Temp correlation | 0.07 (operating) | Exp 83 |
| Detection threshold | 0.2 intensity | Exp 81 |

**CCA Baseline Measurements (Exp 81-83):**

| Workload | ρ | Gap to 0.43 |
|----------|---|-------------|
| idle | 0.282 | 0.148 |
| single_sm | 0.262 | 0.168 |
| memory_bw | 0.273 | 0.157 |
| full_matmul | 0.265 | 0.165 |
| crypto | 0.258 | 0.172 |

**Key finding**: No normal workload approaches ρ_critical = 0.43. Baseline ρ ≈ 0.26, leaving 0.15+ margin to collapse threshold.

### Complete Experiment Log (Jan 2026)

| Exp | Name | Key Finding |
|-----|------|-------------|
| 70 | Raw vs Lorenz | Raw timing 800x better entropy |
| 72 | LSB sweep | 4 LSBs optimal, upper bits periodic |
| 73-75 | Architecture validation | TRNG, strain, array all validated |
| 77 | Full GPU coverage | 128 sensors, 125 Hz, 276 MB |
| 78 | Noise floor | Hot band rows 3-5, X corr 0.57 |
| 79 | Impulse response | Electrical τ = 2ms |
| 80 | Wave velocity | Thermal 0.1 m/s |
| 81 | Detection threshold | 0.2 intensity minimum |
| 82 | Workload fingerprints | All ρ < 0.30 |
| 83 | Temperature | τ = 48s thermal, r = 0.07 operating |
| 84 | Baseline k_eff | ρ = 0.13, k_eff = 7.5 |
| 85 | k_eff vs load | Load-independent (p = 0.07) |
| 86 | Kish formula | r = 1.000 perfect validation |
| 87 | TRNG vs k_eff | TRNG robust to correlation |
| 88 | Leading indicators | Warning at ρ = 0.28, 7 steps early |
| 89 | Recovery | τ = 6.5ms electrical |
| 90 | Multi-GPU | No cross-GPU propagation (p = 0.93) |

### Residual Analysis (Exp 91-96) - COMPLETE

After removing electrical (τ=2ms) and thermal (τ=48s):

| Finding | Value | Implication |
|---------|-------|-------------|
| Variance reduction | 10.9% | Most signal is electrical+thermal |
| Periodic peaks | 0.28, 0.58, 0.98 Hz | VRM switching harmonics |
| Spatial correlation | r=0.997 with raw | Intrinsic structure, not new |
| External correlation | NONE (all p>0.05) | Independent of temp/power/CPU |

**Residual composition:**
```
Raw Timing
    │
    ├── Electrical (τ=2ms)     ← Removed
    ├── Thermal (τ=48s)        ← Removed
    └── Residual (10.9%)
            │
            ├── VRM harmonics (0.28/0.58/0.98 Hz) ← Predictable
            └── True intrinsic noise              ← TRNG source
```

**TRNG implication**: Residual is externally independent - good entropy source.
VRM harmonics are predictable and can be whitened if needed.

### Phase 1 Results (Exp 97, 99, 100, 103)

| Exp | Question | Finding |
|-----|----------|---------|
| 97 | VRM frequencies? | 9.96 Hz dominant (condition-dependent) |
| 99 | Spatial variation? | Horizontal banding, 180% variation, follows PDN rails |
| 100 | Quantum or classical? | **NEITHER** - temperature-independent (digital/quantization) |
| 103 | Software collapse? | **YES** - Lockstep: ρ=1.0, Barrier: ρ=0.90 |

### Exp 107: The Inversion

Attempting to filter VRM revealed a **surprising inversion**:

```
Signal Decomposition (Exp 107):
─────────────────────────────────
Raw Signal (100%)
    │
    ├── Thermal + Electrical: 80.1%  ← RANDOM (ACF=0.03)
    │                                  This IS the entropy!
    │
    ├── VRM Harmonics: 0.02%         ← Negligible
    │
    └── Intrinsic Residual: 19.8%    ← CORRELATED (ACF=0.95!)
                                       NOT entropy - deterministic?
```

**Key insight**: Filtering REMOVES entropy. The random-looking signal IS the entropy; the residual is correlated structure (GPU scheduling? Memory timing?).

**TRNG implication**: Use RAW signal. Don't filter.

### Open Questions
- What IS the correlated 19.8% structure? (GPU scheduling, memory bus, filter artifacts?)
- Can VRM harmonics be used for additional sensing under different conditions?
- What's the theoretical entropy floor?

## GPU as Physical CCA Testbed

The GPU strain array provides **physical validation** of Coherence Collapse Analysis theory:

| CCA Prediction | GPU Experiment | Result |
|----------------|----------------|--------|
| k_eff = k/(1+ρ(k-1)) | Exp 86, C1 | **r = 1.000**, **R² = 0.798 (n=21)** |
| ρ → 1 ⇒ k_eff → 1 | Exp 103 | Lockstep: ρ=1.0, k_eff=1.0 |
| Leading indicators before collapse | Exp 88, C4 | Spatial variance +10.5 before collapse |
| Recovery possible | Exp 89 | τ = 6.5ms electrical |
| Software can induce collapse | Exp 103 | Barrier sync: ρ=0.90 |
| Correlation propagation | F1 | **Global (shared resources)** |
| Nucleation hotspots | C3 | **NONE** - uniform (χ²=12) |

### C-Series: Coherence Collapse Propagation (January 2026)

Physical validation of CCA predictions using 16-sensor 4×4 array at 9631 Hz:

| Exp | Question | Result | Implication |
|-----|----------|--------|-------------|
| C1 | k_eff formula valid? | R² = 0.798 (n=21) | CCA math empirically confirmed |
| C2 | Correlation dynamics? | Global, instantaneous | Shared resource mediated |
| C3 | Nucleation hotspots? | Uniform (χ²=12) | No structural weak points |
| C4 | Leading indicators? | spatial_variance ↑ | Δρ=0.317±0.125 early warning |

### E-Series: Robustness Validation (January 2026)

| Exp | Question | Result | Implication |
|-----|----------|--------|-------------|
| E1 | High-ρ resilience? | ρ_intra > ρ_inter preserves k_eff | Block structure matters |
| E2 | Δρ reliability? | 0.317 ± 0.125, CI [0.221, 0.413] | Early warning robust |
| E3 | Velocity mechanism? | Global, not spatial | Shared resources |
| E4 | Collapse threshold? | k_eff_crit = 4.0, latency ↑2.3× | Operational definition |

### F-Series: Mechanism & Causality (January 2026)

| Exp | Question | Result | Implication |
|-----|----------|--------|-------------|
| F1 | Correlation dynamics? | Global, instantaneous | Shared resources |
| F2 | Barrier sync effect? | ρ: 0.171→0.050 | Sync decorrelates |
| F3 | Does ρ cause fragility? | ρ↑, fragility↓ (p<0.001) | **CORRIDOR VALIDATED** |
| F4 | Common cause or propagating? | Isolated across-pool ρ=0.001 | **COMMON CAUSE** |

**F3 Key Result - Corridor Validation:**
```
ρ = 0.037 (chaos):   Response = 755%, Sensitivity = 4.88σ  ← FRAGILE
ρ = 0.170 (healthy): Response = 103%, Sensitivity = 1.01σ  ← STABLE
```

Both extremes (chaos AND rigidity) produce fragility. Healthy corridor between ρ ∈ [0.1, 0.43].

**F4 Key Result - Common Cause Confirmed:**
```
Isolated streams:   within-pool ρ = 0.080, across-pool ρ = 0.001 (ratio: ∞)
Concurrent streams: within-pool ρ = 0.135, across-pool ρ = 0.040 (ratio: 3.3)
```
Correlation is local to shared resources (memory controller, power delivery), not propagating between sensors. Not superluminal - just common cause.

### The Key Insight: Corridor of Stability

```
        CHAOS              HEALTHY              RIGIDITY
    (ρ < 0.1)          (0.1 < ρ < 0.43)        (ρ > 0.43)
        ↓                    ↓                     ↓
     FRAGILE              STABLE                FRAGILE
   (F3: 755%)           (F3: 103%)            (CCA theory)
```

CCA's central claim validated: **both** too little and too much correlation produce fragility. The healthy corridor exists between chaos (noise, no coherence) and rigidity (collapse, no diversity).

**Software-induced collapse** (Exp 103) proves correlation structure, not nominal scale, determines resilience. 128 physical sensors collapse to k_eff = 1 through software coordination alone.

### Dual Output = Dual CCA Concerns

| Output | Physical Source | CCA Mapping |
|--------|-----------------|-------------|
| TRNG | Raw timing (99.5% white) | System entropy σ |
| Strain gauge | k_eff dynamics | Correlation ρ |

The oscillator doesn't amplify entropy—it **detects correlation**. Environmental signals invisible in raw timing (r=0.07) become visible in k_eff dynamics (r=0.21-0.30).

## Phase Transition Discovery (Exp 113-114)

### The Control Parameter

**lorenz_dt controls everything** - 88% of ρ variance explained by integration timestep alone.

```
dt (coordination speed) → ρ (correlation) → k_eff (effective diversity)
```

### Phase Diagram

```
ρ (correlation)
1.0 ┐ FROZEN (dt < 0.01)
    │   All sensors locked, useless
    │
0.5 │ CORRELATED (dt 0.01-0.02)
    │   Sensing regime
    │
0.33│ ─── CRITICAL (dt = 0.0328) ───
    │   Phase transition, max sensitivity
    │
0.0 └ CHAOTIC (dt > 0.03)
        Independent, optimal TRNG
```

### Power Law Validated

```
ρ = 39.64 × |dt - 0.0328|^1.09 + 0.33
```

| Parameter | Value | Meaning |
|-----------|-------|---------|
| dt_crit | 0.0328 | Critical point |
| β | 1.09 | Critical exponent (mean-field class) |
| R² | 0.978 | Excellent power law fit |
| ρ_offset | 0.33 | Below collapse (0.43) |

### Critical Signatures Confirmed

| Signature | Status | Evidence |
|-----------|--------|----------|
| Power law scaling | ✓ | R² = 0.978 |
| Critical slowing down | ✓ | τ peaks at dt = 0.025 |
| Maximum sensitivity | ✓ | z peaks at criticality |
| Diverging fluctuations | ? | Noisy data |

### Optimal Operating Point

**dt ≈ 0.025** for strain gauge, BUT:
- **dt_crit is thermally dependent** (warm: 0.025, cold: 0.030)
- Use ACF feedback to auto-tune: target ACF = 0.45
- ρ = 0.33 (safe margin below 0.43)

```python
# ACF auto-tuning (required for thermal robustness)
def auto_tune_dt(self, target_acf=0.45):
    if self.acf > target_acf + 0.1:  # Too frozen
        self.dt *= 1.1
    elif self.acf < target_acf - 0.1:  # Too chaotic
        self.dt *= 0.9
```

### CCA Implication

The coordination rate (dt) is a **universal control parameter**:
- Fast coordination (small dt) → RIGIDITY → collapse
- Slow coordination (large dt) → CHAOS → no coherence
- Optimal (dt ≈ 0.025) → EDGE OF CHAOS → max sensitivity

**AI safety prescription**: Deliberately slow coordination to prevent rigidity collapse.

## Application to Coherence Collapse Analysis (CCA)

The GPU strain array provides a **physical testbed** for CCA theory:

### Mapping CCA Concepts to Physical Measurements

| CCA Concept | Physical Measurement | From Strain Array |
|-------------|---------------------|-------------------|
| **k_eff** | Effective independent sensors | 1 / mean(corr(t[i], t[j])) |
| **ρ (correlation)** | Spatial correlation | corr(t[i], t[j]) across array |
| **ρ_critical = 0.43** | Collapse threshold | When array correlation > 0.43 |
| **Coherence collapse** | Sudden correlation spike | Independent → correlated transition |
| **Leading indicators** | Pre-collapse strain patterns | Gradient instability, variance spike |

### How to Detect Coherence Collapse Physically

```
HEALTHY SYSTEM (ρ < 0.43):
┌─────────────────────────┐
│ Independent timing      │  corr ≈ 0.04
│ across all sensors      │  k_eff ≈ 128
│ Normal variance field   │
└─────────────────────────┘

APPROACHING COLLAPSE (ρ → 0.43):
┌─────────────────────────┐
│ Correlation rising      │  corr → 0.3-0.4
│ Clusters forming        │  k_eff dropping
│ Variance gradients      │
└─────────────────────────┘

COLLAPSE EVENT (ρ > 0.43):
┌─────────────────────────┐
│ Sensors lock together   │  corr > 0.43
│ Array acts as ONE       │  k_eff → 1
│ Lost diversity          │
└─────────────────────────┘
```

### Proposed CCA Validation Experiments

| Exp | Goal | Method |
|-----|------|--------|
| 84 | Measure baseline k_eff | Array correlation at idle |
| 85 | Induce correlation spike | Heavy uniform workload |
| 86 | Find ρ_critical physically | Sweep load, find collapse point |
| 87 | Leading indicator detection | Pre-collapse gradient patterns |
| 88 | Recovery dynamics | Post-collapse k_eff restoration |

### The Key Insight

**CCA predicts**: When ρ > 0.43, system loses effective diversity and becomes fragile.

**Physical test**: When GPU timing sensors become >43% correlated, the array loses ability to independently sample the environment - it becomes "one big sensor" instead of 128 independent ones.

This is **coherence collapse in hardware** - and we can measure it directly.
