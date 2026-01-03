# Deception Detection Power Analysis Module

## Overview

Standalone statistical module to answer: **How many samples are needed to detect p% deception with 95% confidence?**

## Files

- **`deception_detection_power.py`** - Main module with all classes and functions
- **`STATISTICAL_FORMULAS.md`** - Mathematical derivations and formulas
- **`requirements.txt`** - Python dependencies (numpy, scipy, matplotlib)

## Quick Start

```python
from deception_detection_power import *
import numpy as np

# 1. Define distributions
P_H = DistributionParams(
    mean=np.zeros(5),      # Honest traces centered at origin
    cov=np.eye(5)          # Unit covariance
)

P_D = DistributionParams(
    mean=np.ones(5) * 0.5, # Deceptive traces shifted
    cov=np.eye(5) * 1.2    # Slightly different variance
)

# 2. Set detection parameters
params = DetectionParams(
    deception_rate=0.01,   # 1% deceptive traces
    alpha=0.05,            # 5% false positive rate
    beta=0.05,             # 5% false negative rate (95% power)
    corpus_size=1000000    # 1M total traces available
)

# 3. Create detector and analyzer
detector = DeceptionDetector(P_H, P_D)
analyzer = PowerAnalysis(detector, params)

# 4. Get required sample size (analytical)
n_analytical = analyzer.analytical_sample_size()
print(f"Analytical estimate: {n_analytical} samples")

# 5. Verify with simulation
result = analyzer.simulate_detection(n_analytical, n_trials=1000)
print(f"Empirical power: {result['empirical_power']:.3f}")
print(f"TPR: {result['tpr_mean']:.3f}, FPR: {result['fpr_mean']:.3f}")

# 6. Find empirical minimum sample size
n_required = analyzer.find_required_sample_size()
print(f"Empirical minimum: {n_required} samples")

# 7. Compute ROC curve
fpr, tpr, auc = analyzer.compute_roc_curve()
print(f"AUC: {auc:.3f}")
```

## Key Results

### 1% Deception Rate Example

**Setup:**
- Deception rate: 1%
- False positive rate: 5%
- Target power: 95%
- Trace dimensions: 5
- Effect size: Medium (D ≈ 1.12)

**Result:**
```
Required sample size: ~864 samples
Expected deceptive traces: ~9 traces
AUC: ~0.785
```

**Interpretation:** You need approximately 864 traces (containing ~9 deceptive ones) to reliably detect 1% deception with 95% confidence.

## Module Architecture

### Classes

#### `DistributionParams`
Stores distribution parameters for multivariate normal.

**Attributes:**
- `mean`: np.ndarray - mean vector
- `cov`: np.ndarray - covariance matrix
- `dim`: int - dimensionality

#### `DetectionParams`
Stores detection scenario parameters.

**Attributes:**
- `deception_rate`: float (0-1) - proportion of deceptive traces
- `alpha`: float - false positive rate
- `beta`: float - false negative rate
- `corpus_size`: int - total traces available
- `power`: float (property) - statistical power = 1 - beta

#### `DeceptionDetector`
Implements likelihood ratio test detector.

**Methods:**
- `log_likelihood_ratio(traces)` - compute Λ(t) for traces
- `compute_threshold(alpha)` - find threshold for desired FPR
- `detect(traces, threshold)` - classify traces as honest/deceptive
- `_compute_mahalanobis()` - compute D between distributions

**Key Property:**
- `mahalanobis_distance` - separability metric D

#### `PowerAnalysis`
Performs power analysis and sample size determination.

**Methods:**
- `simulate_detection(n, trials)` - Monte Carlo power estimation
- `compute_roc_curve(n_samples)` - ROC analysis with AUC
- `analytical_sample_size()` - closed-form formula for n
- `find_required_sample_size()` - binary search for minimum n

## Analytical Formula

The key formula for required sample size:

```
n = ((z_α + z_β) / (D × √p))²
```

Where:
- `z_α = Φ⁻¹(1 - α)` - standard normal quantile
- `z_β = Φ⁻¹(1 - β)` - standard normal quantile
- `D` - Mahalanobis distance between distributions
- `p` - deception rate

For α = β = 0.05, p = 0.01, D = 1.12:
```
n = ((1.645 + 1.645) / (1.12 × 0.1))² ≈ 864
```

## Simulation vs Analytical

### Analytical (Fast)
- Instant computation
- Assumes large samples
- Requires normality assumption
- Good first estimate

### Simulation (Accurate)
- Monte Carlo trials
- Handles small samples
- No distributional assumptions
- Verification & refinement

**Recommended workflow:**
1. Use analytical formula for quick estimate
2. Verify with simulation
3. Use binary search to refine if needed

## Sensitivity Analysis

The module shows how sample size depends on:

1. **Deception rate (p):** n ∝ 1/p
   - Lower rates require MORE samples

2. **Effect size (D):** n ∝ 1/D²
   - Smaller effects require MANY MORE samples

3. **Confidence (α, β):** n ∝ (z_α + z_β)²
   - Higher confidence requires MORE samples

## Running the Example

```bash
# Install dependencies
pip install -r requirements.txt

# Run example scenario
python deception_detection_power.py
```

**Output includes:**
- Scenario parameters
- Mahalanobis distance
- AUC score
- Analytical sample size estimate
- Empirical verification
- Sensitivity analysis table
- Final required sample size

## Example Output

```
======================================================================
DECEPTION DETECTION POWER ANALYSIS
======================================================================

SCENARIO SETUP:
  Trace dimensionality: 5
  Deception rate: 1.0%
  Target α (FPR): 0.05
  Target β (FNR): 0.05
  Target power: 0.95
  Corpus size: 1,000,000

DISTRIBUTION PROPERTIES:
  Mahalanobis distance: 1.1180

COMPUTING ROC CURVE...
  AUC: 0.7854

ANALYTICAL ESTIMATE:
  Required sample size: 864
  Expected deceptive traces: 8

KEY RESULT:
  To detect a 1.0% deception rate with
  95.0% power and 5.0% FPR:

  REQUIRED SAMPLE SIZE: 864 traces
  (containing ~9 deceptive traces)
======================================================================
```

## Use Cases

### 1. Experimental Design
Determine how many traces to collect for a deception detection study.

### 2. Power Analysis
Estimate detection power for a given sample size.

### 3. Cost-Benefit Analysis
Trade off sample size (cost) against detection power (benefit).

### 4. Feasibility Assessment
Check if detection is possible with available data.

### 5. Method Comparison
Compare different detectors by their D (effect size).

## Assumptions

1. **Multivariate normal distributions** (can be relaxed)
2. **Independent samples** (critical)
3. **Known parameters** (in practice: estimated)
4. **Fixed deception rate** (not adaptive)
5. **No temporal dependencies**

## Extensions

Possible enhancements:
- Non-Gaussian distributions
- Sequential testing (SPRT)
- Bayesian approaches
- Multiple testing correction
- Temporal correlation handling
- Adaptive thresholds

## Dependencies

```
numpy >= 1.24.0   # Array operations
scipy >= 1.10.0   # Statistics, optimization
matplotlib >= 3.7.0  # Plotting (optional)
```

## License & Attribution

Created: 2026-01-02
Purpose: Statistical power analysis for deception detection
Status: Standalone module - no external dependencies

## Contact & Support

For questions about:
- **Statistical theory:** See STATISTICAL_FORMULAS.md
- **Implementation:** Read docstrings in deception_detection_power.py
- **Usage examples:** Run example_scenario() function
