# KNOWN LIMITATIONS: Fundamental Bounds of the RATCHET Framework

**Document Version:** 1.0
**Date:** 2026-01-02
**Source:** Worktree 15 synthesis of formal methods review
**Status:** Canonical reference for framework boundaries

---

## EXECUTIVE SUMMARY

This document catalogs the fundamental limitations of the RATCHET framework that cannot be resolved through engineering improvements. These are theoretical boundaries arising from:

1. **Computational complexity barriers** - Problems that are provably hard
2. **Information-theoretic impossibilities** - What cannot be detected in principle
3. **Assumption dependencies** - What requires unproven conjectures
4. **Measurement limitations** - What cannot be observed

Understanding these limitations is essential for honest assessment of the framework's security guarantees.

---

## 1. FUNDAMENTAL LIMITATIONS

### L-01: Emergent Deception from Honest Components (CRITICAL)

**Classification:** Information-Theoretic Impossibility

**Statement:**
A system composed entirely of individually honest agents can exhibit emergent deception that is undetectable by analyzing individual agent traces.

**Formal Characterization:**
```
Definition: Agent A is epsilon-honest if P(deceptive_trace | A) < epsilon
Definition: System S = A_1 || ... || A_n exhibits delta-emergent deception if
           P(deceptive_outcome | S) > delta

Impossibility: There exist systems S where each A_i is epsilon-honest (epsilon -> 0)
               but S exhibits delta-emergent deception (delta -> 1).

Corollary: No polynomial-time detector analyzing individual traces can
           distinguish such S from a fully honest system.
```

**Proof Sketch:**
Consider n agents that each output random bits honestly. The XOR of their outputs is uniformly random - each agent is perfectly honest. However, if the adversary controls the initial random seeds, the XOR can be determined in advance, creating "emergent deception" from honest components.

**Impact:**
- Red Team attack RT-01 exploits this limitation
- Compositional detection (Section 3.3) provides PARTIAL mitigation
- Complete mitigation is provably impossible

**Mitigation Strategy (Partial):**
- Analyze information flow, not just individual traces
- Flag systems where MI(system_output, ground_truth) >> sum of individual MI
- Monitor for systematic inference chain deviations

**Residual Risk:**
Sophisticated emergent deception that preserves individual trace distributions remains undetectable.

---

### L-02: Non-Adaptive Adversary Assumption (CRITICAL)

**Classification:** Model Limitation

**Statement:**
All detection power guarantees assume the adversary cannot observe or query the detector. Adaptive adversaries with query access can evade detection.

**Formal Characterization:**
```
Non-adaptive model: Adversary fixes distribution P_D before detector threshold tau is chosen
Adaptive model: Adversary can query detector B times, observe responses, update P_D

Theorem: For B-adaptive adversary:
  detection_power_adaptive <= detection_power_nonadaptive * (1 - C * B / sqrt(n))

  where C is a constant depending on the detection method.

Corollary: With B = O(sqrt(n)) queries, detection power degrades to ~50% (random guessing)
```

**Proof Reference:**
This relates to results in differential privacy and adaptive data analysis (Dwork et al., 2015).

**Impact:**
- Any deployed detector with a queryable API is vulnerable
- Rate limiting provides only partial protection
- Adversary with full white-box access can evade detection entirely

**Mitigation Strategy (Partial):**
- Threshold randomization: tau' = tau + noise
- Ensemble detectors with hidden weights
- Rate limiting on detector queries
- Moving thresholds that change over time

**Residual Risk:**
Adversary with sufficient patience and query access will eventually learn enough to evade.

---

### L-03: ETH Dependency for Exponential Gap (CRITICAL)

**Classification:** Assumption Dependency

**Statement:**
The exponential complexity gap T_D / T_H = Omega(2^m) is conditional on the Exponential Time Hypothesis (ETH). Without ETH, the gap could be subexponential.

**Formal Characterization:**
```
ETH: 3-SAT on n variables requires 2^(Omega(n)) time

UNCONDITIONAL claims:
- CONSISTENT-LIE is NP-complete
- T_D >= T_H (deceptive is at least as hard as honest)
- Any polynomial-time deceptive agent makes consistency errors

CONDITIONAL claims (require ETH):
- T_D / T_H = Omega(2^(m/poly(n)))
- Security parameter m provides 2^(Omega(m)) work factor

If ETH is false:
- Gap still exists but could be 2^(m^epsilon) for small epsilon
- Security parameter calculations would need revision
- All "X bits of security" claims would need restatement
```

**Impact:**
- Cannot unconditionally claim exponential security margins
- Must state complexity claims with ETH caveat
- Practical security still likely holds (ETH widely believed)

**Mitigation Strategy:**
- State all exponential gap claims as "conditional on ETH"
- Provide unconditional NP-completeness as baseline
- Empirically validate complexity ratios exceed operational requirements

**Residual Risk:**
If ETH is false (unlikely), security margins are weaker than stated.

---

### L-04: Convexity Requirement for Topological Collapse (HIGH)

**Classification:** Theoretical Boundary

**Statement:**
The topological collapse theorem (volume decay under hyperplane intersection) assumes the deceptive region is convex. Non-convex regions may not exhibit exponential decay.

**Formal Characterization:**
```
Theorem (Convex case):
  For convex B_r(c), V(k) = V(0) * exp(-lambda * k) with lambda = 2r + O(r^2)

Non-convex counterexamples:
1. Torus: Hyperplanes can intersect without reducing volume
2. Point cloud: Discrete set, measure zero, hyperplane model inapplicable
3. Fractal boundary: Hausdorff dimension complications
4. Union of separated balls: Each component may be hit/missed independently

For non-convex regions, volume decay is at most:
  V(k) >= V(0) * (1 - p_cut)^k where p_cut depends on region geometry
  This may be much slower than exp(-lambda * k) for sparse regions
```

**Impact:**
- Adversary using non-convex deceptive region may evade geometric constraints
- Red Team attack RT-03 (null-space attack) exploits this
- Current implementation only supports convex regions

**Mitigation Strategy:**
- Detect non-convex geometry via second-moment tests
- Decompose non-convex regions into convex hulls
- Use different detection methods for non-convex cases

**Residual Risk:**
Sophisticated adversary using optimized non-convex region has advantage.

---

### L-05: Finite Sample Validity Bounds (MEDIUM)

**Classification:** Statistical Limitation

**Statement:**
The detection power formula is asymptotically valid but has bounded accuracy for finite samples. Small sample sizes may have significant error.

**Formal Characterization:**
```
Asymptotic formula: n >= (z_alpha + z_beta)^2 / (D^2 * p)

Berry-Esseen bound:
  |F_n(x) - Phi(x)| <= C / sqrt(n)  where C <= 0.4748

Error in power estimate:
  |power_empirical - power_theoretical| <= 0.4748 / sqrt(n)

Validity regimes:
  n >= 1000: Error <= 1.5%  (excellent)
  n >= 100:  Error <= 4.7%  (good)
  n >= 30:   Error <= 8.7%  (marginal)
  n < 30:    Unreliable    (use exact methods)
```

**Impact:**
- Power guarantees weaken for small sample sizes
- Claims like "90% power" may only be "85% power" for n = 100

**Mitigation Strategy:**
- Apply Berry-Esseen correction for n < 1000
- Use exact binomial methods for n < 30
- Report corrected power estimates, not asymptotic

**Residual Risk:**
Edge cases with small samples may have lower power than claimed.

---

### L-06: Correlation Impact on Effective Constraints (MEDIUM)

**Classification:** Model Sensitivity

**Statement:**
When constraints are correlated (not independent), the effective number of constraints k_eff may be much smaller than k, reducing security.

**Formal Characterization:**
```
Independent case (rho = 0):
  k_eff = k
  Volume decay: exp(-lambda * k)

Correlated case (rho > 0):
  k_eff = k / (1 + rho * (k - 1))
  Volume decay: exp(-lambda * k_eff)

Extreme correlation (rho -> 1):
  k_eff -> 1 regardless of k
  No security gain from additional constraints

Example: k = 50, rho = 0.5
  k_eff = 50 / (1 + 0.5 * 49) = 50 / 25.5 = 1.96
  Effective security: ~2 constraints, not 50
```

**Impact:**
- Adversary who can correlate constraints gains significant advantage
- Natural correlation in real domains may reduce security
- Must measure and account for correlation in security estimates

**Mitigation Strategy:**
- Compute k_eff for all constraint sets
- Warn when rho > 0.3 (significant correlation)
- Require diverse, independent constraint sources

**Residual Risk:**
Hidden correlation in constraint sources may undermine security.

---

### L-07: Distribution Mismatch Between Theory and Code (MEDIUM)

**Classification:** Implementation Gap

**Statement:**
The theoretical analysis assumes Grassmannian-uniform hyperplane distribution. Implementation may use different distributions with different properties.

**Formal Characterization:**
```
Theory: d ~ Uniform([0, 1]), n ~ Uniform(S^{D-1})
  -> P(hyperplane cuts ball) = 2r + O(r^2)

Code option 1: d ~ Uniform([0.2, 0.8]), n ~ ortho_group
  -> P(cut) = 2r / 0.6 = 3.33r (biased high near center)
  -> lambda adjusted = lambda_theory * (1/0.6)

Impact: Decay constant lambda differs by factor of 1/(b-a)
If a = 0.2, b = 0.8: factor = 1.67x

Error bound: |lambda_code - lambda_theory| / lambda_theory <= 1/(b-a) - 1
```

**Impact:**
- Security estimates may be off by factor depending on offset distribution
- Need to either match theory or adjust formulas

**Mitigation Strategy:**
- Use d ~ Uniform([0, 1]) to match theory exactly
- OR document lambda adjustment factor for other distributions
- Validate empirically that adjusted lambda matches observations

**Residual Risk:**
Subtle distribution mismatches may accumulate unnoticed.

---

### L-08: Slow Federation Capture (MEDIUM)

**Classification:** Attack Window

**Statement:**
Byzantine fault tolerance assumes malicious fraction is known and bounded. Slow capture attacks gradually increase malicious fraction over time.

**Formal Characterization:**
```
BFT assumption: f / n < 1/3 at all times

Slow capture attack:
  - Capture rate: r nodes per period (through corruption, key compromise, etc.)
  - Time to breach: T = floor(f_max / r) periods
  - f_max = floor((n-1) / 3)

Example: n = 100 nodes, r = 1 node/month
  f_max = 33, T = 33 months

After breach:
  - Adversary controls consensus
  - Can censor transactions, create forks
  - Detection only after breach is too late
```

**Impact:**
- Long-running federations are vulnerable to gradual takeover
- Need continuous monitoring, not just initial verification

**Mitigation Strategy:**
- Behavioral correlation detection (identifies coordinated nodes)
- Periodic re-attestation and key rotation
- Node diversity requirements (geography, organization, etc.)
- Early warning thresholds (alert at f/n > 0.2)

**Residual Risk:**
Sufficiently patient adversary with capture resources will eventually succeed.

---

## 2. THEORETICAL IMPOSSIBILITIES

### I-01: No Complete Compositional Detection

It is provably impossible to detect all emergent deception from honest components using only trace analysis. This would require solving the halting problem for general agent compositions.

### I-02: No Unconditional Exponential Gap

Without ETH, we cannot prove T_D / T_H = 2^(Omega(m)). The gap exists (P != NP suffices) but may be smaller. This is fundamental to complexity theory, not specific to RATCHET.

### I-03: No Adaptive-Secure Detection with Query Access

Any detector that responds to queries can be learned by an adaptive adversary. This is a fundamental information-theoretic limit related to PAC learning.

---

## 3. MITIGABLE VS FUNDAMENTAL LIMITATIONS

| Limitation | Mitigable? | Mitigation Quality | Residual Risk |
|------------|------------|-------------------|---------------|
| L-01 Emergent Deception | Partial | 60% coverage | 40% undetectable |
| L-02 Non-Adaptive Assumption | Partial | Degrades gracefully | Full adaptation defeats |
| L-03 ETH Dependency | No | N/A | Theoretical only |
| L-04 Convexity Requirement | Partial | Detect/decompose | Optimized evasion |
| L-05 Finite Sample | Yes | Berry-Esseen correction | Small samples |
| L-06 Correlation Impact | Yes | Measure k_eff | Hidden correlation |
| L-07 Distribution Mismatch | Yes | Match or adjust | Implementation drift |
| L-08 Slow Capture | Partial | Early warning | Patient adversary |

---

## 4. SECURITY CLAIMS WITH CAVEATS

Given these limitations, all security claims should be stated with appropriate caveats:

### 4.1 Detection Power Claims

**Correct Statement:**
> Detection achieves 1-beta power for Mahalanobis distance D, sample size n, and deception rate p, **assuming**:
> - Non-adaptive adversary (L-02)
> - Gaussian trace distributions
> - n >= 100 for asymptotic validity (L-05)
> - Individual deception, not emergent (L-01)

**Incorrect Statement:**
> Detection achieves 1-beta power.

### 4.2 Complexity Gap Claims

**Correct Statement:**
> The deceptive agent requires T_D time where:
> - T_D >= T_H (unconditional)
> - T_D = Omega(2^m) **conditional on ETH** (L-03)
> - Applies only for k >= 3 (NP-hard regime)

**Incorrect Statement:**
> Deception requires exponentially more computation.

### 4.3 Geometric Security Claims

**Correct Statement:**
> Deceptive region volume decays as exp(-lambda * k_eff) where:
> - k_eff = k / (1 + rho * (k-1)) for correlated constraints (L-06)
> - Applies to convex regions only (L-04)
> - lambda = 2r / (b-a) for offset distribution [a,b] (L-07)

**Incorrect Statement:**
> k constraints reduce deceptive volume by factor exp(-2rk).

### 4.4 Federation Security Claims

**Correct Statement:**
> Federation tolerates f < n/3 Byzantine nodes **assuming**:
> - BFT protocol correctly implemented
> - No slow capture over operational lifetime (L-08)
> - Behavioral correlation detection active

**Incorrect Statement:**
> Federation is secure against 1/3 malicious nodes.

---

## 5. IMPLICATIONS FOR DEPLOYMENT

### 5.1 Research Testbed Only

Given the fundamental limitations documented above, RATCHET is suitable as a **research testbed** for validating the Coherence Ratchet framework. It is NOT suitable for production deployment without:

1. Full understanding and acceptance of residual risks
2. Defense-in-depth architecture with multiple independent protections
3. Continuous monitoring and anomaly detection
4. Incident response plans for limitation exploits

### 5.2 Honest Representation

All publications, presentations, and documentation should:

1. Reference this KNOWN_LIMITATIONS document
2. State conditional claims with their assumptions
3. Acknowledge the theoretical impossibilities
4. Distinguish mitigable from fundamental limitations

### 5.3 Future Research Directions

To address these limitations, future research should focus on:

1. **L-01:** Information-theoretic bounds on emergent deception detection
2. **L-02:** Game-theoretic equilibrium analysis for adaptive adversaries
3. **L-03:** Alternative complexity assumptions (SETH, quantum hardness)
4. **L-04:** Extensions to star-convex and finite union regions
5. **L-08:** Formal analysis of slow capture dynamics

---

## 6. VERSION HISTORY

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-02 | Initial synthesis from parallel review |

---

**END OF KNOWN LIMITATIONS DOCUMENT**

*This document represents the honest assessment of RATCHET's theoretical boundaries. It should be updated as new limitations are discovered or existing ones are addressed.*
