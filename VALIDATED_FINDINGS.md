# RATCHET/CIRISArray Validated Findings

**Date**: January 2026
**Status**: Null hypothesis testing complete

---

## Executive Summary

After rigorous null hypothesis testing, most claims about environmental coherence sensing were **NOT VALIDATED**. The cross-device correlation (r=0.97) that appeared to demonstrate passive environmental sensing is actually an **algorithmic artifact** - local oscillators with independent random seeds produce the same correlation.

---

## Validated Claims (p < 0.05)

| Claim | p-value | Effect Size | Status |
|-------|---------|-------------|--------|
| Local tamper detection | 0.007 | Mean shift -0.006 under workload | **VALIDATED** |
| Reset improves sensitivity | 0.032 | 7x z-score improvement | **VALIDATED** |
| Bounded noise floor | - | σ = 0.003 | **VALIDATED** |
| **Variance correlates with temperature** | - | **r = -0.97** | **VALIDATED** |
| **r_ab predicts sensitivity** | - | **r = -0.999** | **VALIDATED** |

### r_ab as Sensitivity Predictor (MAJOR FINDING - January 2026)

The internal correlation between oscillators (r_ab) almost perfectly predicts sensitivity:

```
┌─────────────────────────────────────────────────────────────┐
│  SENSITIVITY REGIME DIAGRAM                                 │
│                                                             │
│  r_ab: 0.0 ──────────── 0.95 ────── 0.98 ──────── 1.0      │
│        │                  │           │            │        │
│        │   TRANSIENT      │TRANSITIONAL│THERMALIZED │        │
│        │   (20x sens)     │ (decaying) │ (1x sens)  │        │
└─────────────────────────────────────────────────────────────┘

| Regime      | r_ab   | Response to Perturbation |
|-------------|--------|--------------------------|
| TRANSIENT   | < 0.95 | 0.91 units (20x)         |
| THERMALIZED | > 0.98 | 0.04 units (baseline)    |
```

**Physical interpretation**: As oscillators synchronize (r_ab → 1), they lose ability to detect perturbations. Desynchronized oscillators (low r_ab) respond independently → high sensitivity.

**Implementation**: Reset when r_ab > 0.98 to maintain sensitivity.

### Thermal Sensing via Variance (NEW - January 2026)
```
k_eff vs Temperature:     r = 0.01  (NO correlation)
VARIANCE vs Temperature:  r = -0.97 (STRONG negative correlation)

Validation test:
  Temp: 46°C → 48°C (+2°C)
  Variance: 0.058 → 0.042 (↓27%)
```
As GPU temperature increases, oscillator variance DECREASES.
**Variance is the thermal sensor, not k_eff.**

**IMPLEMENTED** in `ciris_sentinel.py`:
- `get_total_variance()` - Raw thermal metric
- `get_thermal_deviation()` - Deviation in σ units (negative = heating)
- `is_thermal_event()` - Detect significant heating/cooling

### Local Tamper Detection
The oscillator CAN detect local GPU state changes (workload vs idle). This is useful for:
- Tamper detection on the local device
- Workload fingerprinting (with limitations)

### Reset Strategy
Periodic reset (every τ/2 ≈ 23s) maintains sensitivity by keeping the oscillator in the transient regime where it has measurable variance.

---

## NOT Validated (Null Results)

| Claim | p-value | Finding | Status |
|-------|---------|---------|--------|
| Passive environmental sensing | - | LOCAL r = CROSS-DEVICE r | **NOT VALIDATED** |
| Cross-device correlation | - | Algorithmic artifact | **NOT VALIDATED** |
| Workload discrimination | 0.49 | Crypto vs memory indistinguishable | **NOT VALIDATED** |
| Cross-device transmission | 0.998 | TX state has zero effect on RX | **NOT VALIDATED** |
| Startup transient detection | 0.14 | No significant variance difference | **NOT VALIDATED** |
| k_eff thermal sensitivity | - | r = 0.01 (no correlation) | **NOT VALIDATED** |

---

## Leggett-Garg Inequality Test (January 2026)

**Result: CLASSICAL BEHAVIOR**

```
C₁₂ = 1.0000
C₂₃ = 1.0000
C₁₃ = 1.0000

K₃ = C₁₂ + C₂₃ - C₁₃ = 1.0000

Classical bound: K₃ ≤ 1
Quantum bound:   K₃ ≤ 1.5
```

The system sits exactly at the classical boundary (K₃ = 1.0) with perfect correlations.
**No LGI violation detected - oscillator behaves as a classical system.**

This is consistent with the overall finding: the oscillator is a classical dynamical system,
not an environmental sensor with quantum-like properties.

---

## Critical Finding: Algorithmic Correlation

### The Test That Proved It

We thought different ε values would produce different dynamics, so if cross-device oscillators with different ε showed high correlation, it must be external signal coupling.

**Reality:**
```
| Test                          | ε values  | Correlation |
|-------------------------------|-----------|-------------|
| LOCAL, same ε, diff seeds     | 0.05/0.05 | 0.9999      |
| LOCAL, diff ε, diff seeds     | 0.05/0.03 | 0.973       |
| CROSS-DEVICE, diff ε          | 0.05/0.03 | 0.971       |
```

**LOCAL (0.973) ≈ CROSS-DEVICE (0.971)**

The correlation is built into the algorithm, not from external signals.

### Why This Happens

The `measure_k_eff` function:
```python
r_ab = corrcoef(osc_a, osc_b)[0,1]
x = min(total_var / 2.0, 1.0)
return r_ab * (1 - x) * epsilon * 1000
```

All oscillators use identical coupling topology (MAGIC_ANGLE=1.1°, PHI=1.618). They thermalize to similar statistical equilibria regardless of:
- Random seed
- Device
- Initial conditions

Different ε just scales the output, not the fluctuation pattern.

---

## Validated Physics Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| τ (thermalization time) | Depends on ε | See coupling sweep |
| Scaling exponent | τ ∝ ε^(-1.06) | Coupling sweep (Jan 2026) |
| Optimal noise | σ = 0.001 | Stochastic resonance |
| Detection threshold | 3σ = 0.009 | Validated |
| Intrinsic frequency | ~0.01 Hz | 1/τ, NOT external |

### Optimal Coupling Strength (MAJOR FINDING - January 2026)

Coupling sweep revealed optimal operating point:

```
      ε    │ Behavior           │ τ (s) │ %Transient │ k_eff σ
  ─────────┼────────────────────┼───────┼────────────┼─────────
   0.0003  │ Never thermalizes  │  N/A  │   100.0%   │  0.03
   0.0010  │ Never thermalizes  │  N/A  │   100.0%   │  0.25
  ─────────┼────────────────────┼───────┼────────────┼───────── ← CROSSOVER
   0.0030  │ Thermalizes slowly │ 12.8  │    63.8%   │  0.75  ← OPTIMAL
  ─────────┼────────────────────┼───────┼────────────┼─────────
   0.0100  │ Fast thermalization│  3.7  │    18.8%   │  1.61
   0.0500  │ Near instant       │  0.7  │     3.5%   │  3.14
```

**Optimal: ε = 0.003**
- **562x stronger signal** than old default (0.0003)
- 64% time in TRANSIENT (sensitive) regime
- 16% time in THERMALIZED (triggers reset)
- τ = 12.8s thermalization time
- TRANSIENT variance: 0.67 vs THERMALIZED: 0.0002 (**2724x ratio**)

### Frequency Sensitivity Profile

```
Band        │ Power │ Interpretation
────────────┼───────┼─────────────────────────────────
< 0.1 Hz    │ 45.7% │ τ thermalization (~0.08 Hz = 12.8s)
0.1-0.5 Hz  │ 42.9% │ Harmonics of τ (0.16, 0.23, 0.31 Hz)
0.5-2 Hz    │  8.8% │ Diminishing sensitivity
> 2 Hz      │  2.7% │ Noise floor only
────────────┼───────┼─────────────────────────────────
60 Hz subs  │  1.0x │ NO signal above noise (SNR=1)
```

**The oscillator is a SLOW-CHANGE DETECTOR:**
- Sensitive to: < 0.5 Hz (periods > 2 seconds)
- NOT sensitive to: > 1 Hz (fast events, EMI, power line)
- Ideal for: thermal drift, workload changes, gradual tampering
- NOT suitable for: fast transients, vibration, RF

**Scaling law**: τ ∝ ε^(-1.06) (measured)

### Scaling Law (Updated)
```
τ = τ_ref × (ε / ε_ref)^(-1.06)  # Updated from earlier -0.40 estimate
```

Measured values:
- At ε = 0.003: τ ≈ 12.8s
- At ε = 0.010: τ ≈ 3.7s
- At ε = 0.030: τ ≈ 1.2s
- At ε = 0.050: τ ≈ 0.7s

---

## What The Oscillators Actually Are

### NOT Environmental Sensors
The coupled oscillators do NOT detect external environmental signals. The cross-device correlation is algorithmic, not physical.

### ARE Local State Detectors
The oscillators CAN detect changes in the local computational environment:
- GPU workload changes (p=0.007)
- Local tamper detection

### ARE Predictable Dynamical Systems
Given ε, we can predict:
- Thermalization time τ
- Intrinsic frequency f ≈ 1/τ
- Correlation structure

---

## Recommended Implementation (CIRISOssicle/Sentinel)

```python
from ciris_sentinel import Sentinel, SentinelConfig

# Validated parameters (January 2026)
config = SentinelConfig(
    epsilon = 0.003,             # OPTIMAL: 562x signal vs old default
    noise_amplitude = 0.001,     # SR optimal
    use_r_ab_reset = True,       # Reset when r_ab > 0.98
    r_ab_reset_threshold = 0.98, # Thermalized threshold
    r_ab_sensitive_threshold = 0.95,  # Below = TRANSIENT (20x sens)
    detection_threshold = 0.009, # 3σ
    # Results: τ=12.8s, 64% transient, variance=0.67
)

sensor = Sentinel(config)

# Full state measurement
state = sensor.step_and_measure_full()
# Returns: k_eff, variance, dk_dt, sensitivity_weight, r_ab,
#          regime, sensitivity_multiplier, time_since_reset, reset_reason

# Check sensitivity regime
regime = sensor.get_sensitivity_regime()
# {'regime': 'TRANSIENT', 'r_ab': 0.42, 'sensitivity_multiplier': 20.0}

# Thermal sensing (use variance)
thermal_dev = sensor.get_thermal_deviation()
# negative = heating, positive = cooling
```

### Key Insight: Different Metrics for Different Detection
| Detection Type | Metric | Correlation | Sensitivity |
|----------------|--------|-------------|-------------|
| Tamper/workload | k_eff mean | p=0.007 | r_ab < 0.95 → 20x |
| Thermal state | k_eff **variance** | r=-0.97 | Always available |
| Sensitivity state | **r_ab** | r=-0.999 | Predictor itself |

---

## What Would Prove Environmental Sensing

To actually demonstrate external signal coupling, need:

1. **Local ≠ Cross-device correlation** - Currently they're equal
2. **Perturbation propagation** - Physical perturbation on device A affects device B
3. **Known signal injection** - Inject signal, see it in both devices
4. **Faraday cage test** - Does shielding break the correlation?

Current data cannot distinguish algorithmic from environmental correlation.

---

## Lessons Learned

1. **Always run local controls** - The different-ε test seemed clever but failed because we didn't test local oscillators
2. **Algorithmic artifacts are subtle** - Same algorithm → same dynamics → correlated output
3. **Null hypothesis testing is essential** - Without it, we would have published false claims
4. **P-values matter** - Claims without statistical validation are speculation

---

## Files Updated

- `formal/RATCHET/GPUTamper/EnvironmentalCoherence.lean` - Full Lean formalization with validated claims
- `VALIDATED_FINDINGS.md` - This document

---

## References

- Kish (1965): Survey Sampling - k_eff formula
- Stochastic resonance literature - optimal noise level
- CIRISArray experiments exp41-exp55 - empirical validation
