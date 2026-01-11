# RATCHET - Claude Code Context

## Project Overview

RATCHET (Reference Architecture for Testing Coherence and Honesty in Emergent Traces) provides the mathematical foundation for **machine intuition** - coherence collapse detection for AI safety.

**Current State**: Clean theoretical foundation ready for rigorous validation against existing domain benchmarks.

## Core Hypothesis

AI systems need coherence collapse detection to complement ethical evaluation. The CIRISAgent provides Machine Conscience (ethical faculties); RATCHET provides Machine Intuition (coherence faculties).

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

formal/
  RATCHET.lean              # Lean 4 proofs
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

1. **Rigorous validation** against existing domain benchmarks
2. **Calibration/falsification** of faculty thresholds
3. **Cross-domain generalization** testing
4. **Lean 4 formalization** of theorems
