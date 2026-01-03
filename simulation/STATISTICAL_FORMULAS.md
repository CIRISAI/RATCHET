# Statistical Power Analysis: Mathematical Formulas

## Problem Statement

**Question:** How many samples (traces) n are needed to detect a p% deception rate with (1-β) confidence (power) and α false positive rate?

## Setup

### Distributions
- **Honest traces:** t ~ P_H = N(μ_H, Σ_H)
- **Deceptive traces:** t ~ P_D = N(μ_D, Σ_D)

### Detector
Likelihood Ratio Test (optimal by Neyman-Pearson lemma):

```
Λ(t) = P_D(t) / P_H(t)

log Λ(t) = log P_D(t) - log P_H(t)
```

**Decision rule:** Classify as deceptive if log Λ(t) > τ

**Threshold:** τ chosen such that P(log Λ(t) > τ | t ~ P_H) = α

## Key Statistical Quantities

### 1. Mahalanobis Distance

Measures separation between distributions:

```
D² = (μ_D - μ_H)ᵀ Σ⁻¹ (μ_D - μ_H)

D = √(D²)
```

For equal covariances Σ_H = Σ_D = Σ, this simplifies to the pooled estimate.

### 2. Log-Likelihood Ratio Distribution

For multivariate normal P_H and P_D with equal covariances:

**Under H (honest):**
```
log Λ(t) ~ N(-D²/2, D²)
```

**Under D (deceptive):**
```
log Λ(t) ~ N(D²/2, D²)
```

### 3. Operating Characteristics

**False Positive Rate (Type I error):**
```
α = P(log Λ(t) > τ | t ~ P_H)
```

**True Positive Rate (Power):**
```
1 - β = P(log Λ(t) > τ | t ~ P_D)
```

**False Negative Rate (Type II error):**
```
β = P(log Λ(t) ≤ τ | t ~ P_D)
```

## Sample Size Formulas

### Analytical Approximation

For detecting p% deception rate in a corpus:

**Required total sample size:**
```
n ≈ ((z_α + z_β) / (D × √p))²
```

Where:
- z_α = Φ⁻¹(1 - α) = standard normal quantile at (1-α)
- z_β = Φ⁻¹(1 - β) = standard normal quantile at (1-β)
- D = Mahalanobis distance
- p = deception rate (proportion)

**Number of deceptive traces expected:**
```
n_deceptive = p × n
```

### Derivation

The formula comes from the separation between two normal distributions:

1. Under H: LLR ~ N(-D²/2, D²)
2. Under D: LLR ~ N(D²/2, D²)

The means are separated by D² with variance D² in both.

For a sample of size n with p proportion deceptive:
- Effective "signal" scales with √(p × n)
- Effective "noise" scales with √n

Power constraint:
```
(z_α + z_β) × σ = D × √(p × n)
```

Where σ = √D² = D for single observation.

Solving for n:
```
n = ((z_α + z_β) / (D × √p))²
```

## Concrete Example: 1% Deception Rate

### Parameters
- p = 0.01 (1% deception)
- α = 0.05 (5% FPR)
- β = 0.05 (5% FNR, i.e., 95% power)
- D ≈ 1.12 (for 5D space with shift of 0.5 in each dimension)

### Standard Normal Quantiles
```
z_α = Φ⁻¹(0.95) = 1.645
z_β = Φ⁻¹(0.95) = 1.645
```

### Calculation
```
n = ((1.645 + 1.645) / (1.12 × √0.01))²
  = ((3.29) / (1.12 × 0.1))²
  = (3.29 / 0.112)²
  = (29.375)²
  = 863.1

n ≈ 864 samples
```

### Interpretation
- **Total samples needed:** ~864 traces
- **Deceptive traces expected:** 864 × 0.01 ≈ 9 traces
- **Insight:** Need ~9 deceptive examples to reliably detect with 95% power

## Sensitivity Analysis

Effect of parameters on required sample size:

### Deception Rate (p)
```
n ∝ 1/p

p = 0.01 → n ≈ 864
p = 0.05 → n ≈ 173
p = 0.10 → n ≈ 86
```

**Lower deception rates require MORE samples (inversely proportional).**

### Effect Size (D)
```
n ∝ 1/D²

D = 0.5 → n ≈ 4,328
D = 1.0 → n ≈ 1,082
D = 2.0 → n ≈ 271
```

**Smaller effect sizes require MORE samples (inverse square relationship).**

### Confidence Requirements (α, β)
```
n ∝ (z_α + z_β)²

95% confidence (α=β=0.05) → z_α+z_β = 3.29 → factor of 10.8
90% confidence (α=β=0.10) → z_α+z_β = 2.56 → factor of 6.6
99% confidence (α=β=0.01) → z_α+z_β = 4.65 → factor of 21.6
```

**Higher confidence requires MORE samples (quadratically).**

## ROC Curve and AUC

### ROC Curve
Plot of (FPR, TPR) as threshold τ varies:
- FPR = P(Λ > τ | H)
- TPR = P(Λ > τ | D)

### AUC (Area Under Curve)
For normal distributions with equal variance:
```
AUC = Φ(D/√2)
```

Where Φ is the standard normal CDF.

**For D = 1.12:**
```
AUC = Φ(1.12/√2) = Φ(0.79) ≈ 0.785
```

## Practical Considerations

### When Formula Applies
1. Large sample sizes (n > 30)
2. Known distribution parameters (or good estimates)
3. Independent samples
4. Approximately normal distributions

### When to Use Simulation
1. Small samples
2. Unknown distributions
3. Complex dependence structures
4. Verification of analytical results

### Trade-offs
- **More samples:** Higher cost, better power
- **Higher α:** More false alarms, better detection
- **Lower β:** Better detection, more samples needed
- **Larger effect:** Easier detection, may not be realistic

## References

Statistical foundations:
- Neyman-Pearson Lemma: Optimal detector for simple hypotheses
- Wald's Sequential Analysis: Sample size determination
- Signal Detection Theory: ROC curves and d' (related to D)

Connection to effect sizes:
- Cohen's d = (μ_D - μ_H) / σ (univariate)
- Mahalanobis D generalizes to multivariate case
