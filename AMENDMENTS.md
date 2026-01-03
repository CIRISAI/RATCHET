# FSD AMENDMENTS: Consolidated Changes from Parallel Review

**Document Version:** 1.0
**Date:** 2026-01-02
**Source:** Worktrees wt-1 through wt-15 synthesis
**Status:** Ready for coordinator merge

---

## EXECUTIVE SUMMARY

This document consolidates all required amendments to FSD.md identified during the parallel formal methods review. Amendments are organized by priority and section, with specific text changes provided where applicable.

**Amendment Statistics:**
- Critical Priority (Block implementation): 5 amendments
- High Priority (Block testing): 10 amendments
- Medium Priority (Block deployment): 6 amendments
- New Sections Required: 2

---

## 1. CRITICAL PRIORITY AMENDMENTS

### Amendment A-01: Power Formula Preconditions (wt-1: U-01)

**Section:** 3.3 Detection Engine

**Current Text:**
```python
def power_analysis(
    self,
    mahalanobis_distance: float,
    deception_rate: float,
    alpha: float = 0.05,
    beta: float = 0.05,
) -> PowerAnalysisResult:
```

**Amended Text:**
```python
def power_analysis(
    self,
    mahalanobis_distance: Annotated[float, Field(ge=0.5, description="D >= 0.5 for valid asymptotic approximation")],
    deception_rate: Annotated[float, Field(gt=0.001, lt=1.0, description="p in (0.001, 1) for tractable sample sizes")],
    alpha: Annotated[float, Field(gt=0, lt=1)],
    beta: Annotated[float, Field(gt=0, lt=1)],
) -> PowerAnalysisResult:
    """
    Compute required sample size for given power.

    PRECONDITIONS:
    - mahalanobis_distance >= 0.5 (otherwise, increase D or use different test)
    - deception_rate >= 0.001 (otherwise, sample size becomes infeasible)
    - Result valid for n >= 100 (see finite-sample correction below)

    VALIDITY REGIME:
    - n >= 100: Asymptotic formula accurate to O(1/sqrt(n))
    - n < 100: Apply Berry-Esseen correction (see M-03 invariant)

    Returns n, analytical formula, and finite-sample correction.
    """
```

**Rationale:** Prevents unsound security claims from invalid parameter ranges (U-01).

---

### Amendment A-02: k >= 3 NP-Hardness Enforcement (wt-2: U-02)

**Section:** 3.2 Complexity Engine

**Current Text:**
```python
def measure_complexity(
    self,
    world_size: int,
    num_statements: int,
    literals_per_statement: int,
    ...
```

**Amended Text:**
```python
def measure_complexity(
    self,
    world_size: PositiveInt,
    num_statements: PositiveInt,
    literals_per_statement: Annotated[int, Field(ge=3, description="k >= 3 required for NP-hardness")],
    ...
```

**Add Warning Block After Method Signature:**
```python
    """
    Compare honest vs deceptive agent computational cost.

    CRITICAL CONSTRAINT: literals_per_statement >= 3

    WARNING: For k < 3 (2-SAT regime), the CONSISTENT-LIE problem is
    solvable in polynomial time. All NP-hardness claims and exponential
    gap guarantees are VOID for k = 2. The security model assumes k >= 3.

    If your use case requires k = 2, the complexity gap reduces from
    exponential to polynomial, and adversarial deception becomes tractable.

    Returns T_H, T_D, ratio, and 95% CI.
    """
```

**Rationale:** Prevents false security claims in P-time regime (U-02).

---

### Amendment A-03: BFT Protocol Implementation (wt-3: U-03)

**Section:** 3.4 Federation Engine

**Current Text:**
```python
class FederationEngine:
    def __init__(
        self,
        consensus_protocol: Literal["pbft", "raft", "tendermint"] = "pbft",
        mi_threshold: float = 0.85,
    ):
```

**Amended Text:**
```python
class FederationEngine:
    """
    Federated ratchet simulation with Byzantine fault tolerance.

    IMPLEMENTATION REQUIREMENT: Must use actual BFT consensus protocol,
    not just specification. PBFT is the recommended default.
    """

    def __init__(
        self,
        consensus_protocol: BFTProtocol,  # Required, not Optional
        mi_threshold: Annotated[float, Field(ge=0.5, le=1.0)] = 0.85,
    ):
        self.consensus = consensus_protocol
        self._verify_bft_invariants()

    def _verify_bft_invariants(self) -> None:
        """
        Verify BFT protocol configuration satisfies safety requirements.

        INVARIANTS:
        - node_count >= 3 * max_byzantine + 1
        - message_timeout < round_time
        - All message formats properly typed
        """
        assert self.consensus.node_count >= 3 * self.consensus.max_byzantine + 1, \
            f"BFT requires n >= 3f+1, got n={self.consensus.node_count}, f={self.consensus.max_byzantine}"
```

**Add New Type Definition to Section 5 (Schemas):**
```python
class PBFTConfig(BaseModel):
    """PBFT consensus protocol configuration."""
    node_count: PositiveInt
    max_byzantine: int = Field(ge=0, description="Maximum Byzantine nodes to tolerate")
    view_change_timeout_ms: PositiveInt = 5000
    checkpoint_interval: PositiveInt = 100

    @validator('max_byzantine')
    def validate_bft_threshold(cls, v, values):
        n = values.get('node_count', 0)
        if n < 3 * v + 1:
            raise ValueError(f"BFT requires n >= 3f+1: {n} < {3*v + 1}")
        return v

class BFTProtocol(BaseModel):
    """Abstract BFT protocol configuration."""
    protocol_type: Literal["pbft", "tendermint", "hotstuff"]
    config: Union[PBFTConfig, TendermintConfig, HotstuffConfig]
    message_timeout_ms: PositiveInt
    round_time_ms: PositiveInt

    @validator('round_time_ms')
    def validate_timing(cls, v, values):
        timeout = values.get('message_timeout_ms', 0)
        if timeout >= v:
            raise ValueError(f"message_timeout must be < round_time: {timeout} >= {v}")
        return v
```

**Rationale:** Prevents insecure consensus implementation (U-03).

---

### Amendment A-04: Type Hole Fix - Discriminated Union (wt-4: T-SCH-01)

**Section:** 6.2 Schema Examples

**Current Text:**
```python
class SimulationRequest(BaseModel):
    engine: Literal["geometric", "complexity", "detection", "federation"]
    parameters: Dict[str, Any]  # <-- TYPE HOLE
```

**Amended Text:**
```python
from typing import Union, Annotated
from pydantic import Field

class GeometricParams(BaseModel):
    """Parameters for geometric engine simulation."""
    dimension: Dimension
    num_constraints: PositiveInt
    deceptive_radius: Radius
    constraint_correlation: Correlation = 0.0
    sampling_mode: Literal["orthonormal", "correlated", "adversarial"] = "orthonormal"
    num_samples: PositiveInt = 100_000
    adversary: Optional[AdversarialStrategy] = None

class ComplexityParams(BaseModel):
    """Parameters for complexity engine simulation."""
    world_size: PositiveInt
    num_statements: PositiveInt
    literals_per_statement: Annotated[int, Field(ge=3)]
    observable_fraction: Annotated[float, Field(gt=0, le=1)] = 1.0
    deception_strategy: Literal["full", "sparse", "lazy"] = "full"

class DetectionParams(BaseModel):
    """Parameters for detection engine simulation."""
    mahalanobis_distance: Annotated[float, Field(ge=0)]
    deception_rate: Annotated[float, Field(gt=0, lt=1)]
    alpha: Annotated[float, Field(gt=0, lt=1)] = 0.05
    beta: Annotated[float, Field(gt=0, lt=1)] = 0.05
    method: Literal["lrt", "mahalanobis", "isolation_forest", "ensemble"] = "lrt"

class FederationParams(BaseModel):
    """Parameters for federation engine simulation."""
    num_honest: PositiveInt
    num_malicious: int = Field(ge=0, default=0)
    malicious_strategy: Literal["random", "coordinated", "slow_capture"] = "random"
    consensus_protocol: BFTProtocol

SimulationParams = Annotated[
    Union[GeometricParams, ComplexityParams, DetectionParams, FederationParams],
    Field(discriminator='engine_type')
]

class SimulationRequest(BaseModel):
    engine: Literal["geometric", "complexity", "detection", "federation"]
    parameters: SimulationParams  # Type-safe discriminated union
    adversarial: bool = False
    adversary_config: Optional[AdversaryConfig] = None
    num_runs: PositiveInt = 1
    seed: Optional[int] = None
```

**Rationale:** Eliminates Dict[str, Any] type hole (T-SCH-01).

---

### Amendment A-05: Refinement Types for All Interfaces (wt-5: T-GEO-02)

**Add New Section 2.3: Type Definitions**

```python
"""
RATCHET Type Definitions
========================

All numeric parameters use refinement types to prevent invalid inputs
at validation time rather than runtime.
"""

from typing import Annotated
from pydantic import Field, validator
from pydantic.types import PositiveInt

# Geometric Types
Dimension = Annotated[int, Field(gt=0, description="Positive dimension D >= 1")]
Radius = Annotated[float, Field(gt=0, lt=0.5, description="Deceptive region radius 0 < r < 0.5")]
Correlation = Annotated[float, Field(ge=-1, le=1, description="Constraint correlation -1 <= rho <= 1")]
NumConstraints = Annotated[int, Field(gt=0, description="Number of hyperplane constraints k >= 1")]

# Probability Types
Probability = Annotated[float, Field(gt=0, lt=1, description="Probability in open interval (0, 1)")]
ProbabilityInclusive = Annotated[float, Field(ge=0, le=1, description="Probability in [0, 1]")]

# Statistical Types
MahalanobisDistance = Annotated[float, Field(ge=0, description="Non-negative Mahalanobis distance")]
SampleSize = Annotated[int, Field(ge=1, description="Sample size n >= 1")]
EffectiveSampleSize = Annotated[int, Field(ge=100, description="Valid for asymptotic analysis n >= 100")]

# Complexity Types
WorldSize = Annotated[int, Field(gt=0, description="World model size m >= 1")]
LiteralsPerClause = Annotated[int, Field(ge=3, description="Literals per clause k >= 3 for NP-hardness")]

# Federation Types
NodeCount = Annotated[int, Field(ge=4, description="Minimum 4 nodes for BFT (3*1+1)")]
ByzantineCount = Annotated[int, Field(ge=0, description="Byzantine node count f >= 0")]

# Time Types
TimeoutMs = Annotated[int, Field(gt=0, le=600000, description="Timeout in milliseconds")]
```

**Rationale:** Provides compile-time/validation-time type safety (T-GEO-02).

---

## 2. HIGH PRIORITY AMENDMENTS

### Amendment A-06: Missing Proof Obligations - Topological Collapse (wt-6: TC-GAPS)

**Section:** 4.1 Proof Obligations

**Add to Table:**

| ID | Obligation | Status | Notes |
|----|------------|--------|-------|
| TC-2 | Independence (Fubini) | REQUIRED | Formalize: hyperplane cuts are conditionally independent given center |
| TC-3 | Volume scaling for manifold | REQUIRED | Prove: V(k+1) = V(k) * (1 - 2r + O(r^2)) |
| TC-4 | Error bound for exponential | REQUIRED | Prove: |V(k) - V(0)*exp(-lambda*k)| <= C*r^2*k*V(0) |
| TC-8 | Uniform convergence | REQUIRED | Prove: convergence uniform over c in [delta, 1-delta]^D |

**Add Specification for Each:**

```
TC-2 (Fubini Specification):
  Statement: For hyperplanes H_1, ..., H_k with uniform independent distributions,
  P(intersection) = prod_i P(H_i cuts B_r(c)) + O(r^2 * k^2)

  Required for: Exponential decay derivation
  Lean module: Hyperplane.lean (extends Mathlib.MeasureTheory.Integral.Prod)

TC-3 (Volume Scaling Specification):
  Statement: Let V(k) be volume after k hyperplane intersections.
  V(k+1) / V(k) = 1 - 2r + O(r^2) for r << 1

  Required for: Inductive step of main theorem
  Lean module: Hyperplane.lean (new theorem)

TC-4 (Error Bound Specification):
  Statement: |V(k)/V(0) - exp(-lambda*k)| <= C * r^2 * k * exp(-lambda*k)
  where lambda = 2r and C is a universal constant

  Required for: Quantitative bound validation
  Lean module: Hyperplane.lean (new theorem)

TC-8 (Uniform Convergence Specification):
  Statement: For all c in [delta, 1-delta]^D with delta > 2r,
  the error bound in TC-4 holds uniformly.

  Required for: Avoiding boundary effects in empirical validation
  Lean module: Hyperplane.lean (new theorem)
```

**Rationale:** Completes topological collapse proof coverage (TC-GAPS).

---

### Amendment A-07: Missing Proof Obligations - Detection Power (wt-7: DP-GAPS)

**Section:** 4.1 Proof Obligations

**Add to Table:**

| ID | Obligation | Status | Notes |
|----|------------|--------|-------|
| DP-4 | Asymptotic validity | REQUIRED | Berry-Esseen bound for finite n |
| DP-5 | Plug-in estimation error | REQUIRED | Error from empirical D-hat |
| DP-6 | Monotonicity | REQUIRED | Power increases in n, D |

**Add Specification for Each:**

```
DP-4 (Asymptotic Validity Specification):
  Statement: For n >= 100, the sample complexity formula
  n >= (z_alpha + z_beta)^2 / (D^2 * p)
  achieves stated power with error at most 0.05.

  For n < 100, apply Berry-Esseen correction:
  actual_power >= theoretical_power - C / sqrt(n)
  where C <= 0.4748 (Berry-Esseen constant)

  Required for: Valid power guarantees at finite sample sizes
  Lean module: Detection.lean (requires probability theory)

DP-5 (Plug-in Estimation Error Specification):
  Statement: If D is estimated from n_0 honest samples as D-hat,
  then |D-hat - D| <= C * sqrt(d/n_0) with probability 1-delta,
  where d is trace dimension.

  Propagated error in power analysis:
  |power(D-hat) - power(D)| <= 2 * |D-hat - D| / D

  Required for: Honest use of estimated parameters
  Lean module: Detection.lean (new theorem)

DP-6 (Monotonicity Specification):
  Statement: For fixed alpha, beta:
  - power(n_1, D, p) <= power(n_2, D, p) for n_1 <= n_2
  - power(n, D_1, p) <= power(n, D_2, p) for D_1 <= D_2
  - power(n, D, p_1) <= power(n, D, p_2) for p_1 <= p_2

  Required for: Intuitive interpretation of results
  Lean module: Detection.lean (straightforward from formula)
```

**Rationale:** Completes detection power proof coverage (DP-GAPS).

---

### Amendment A-08: Invariant - Non-Adaptive Adversary (wt-8: M-01)

**Section:** 7.2 Security Invariants

**Add to SECURITY_INVARIANTS list:**

```python
# M-01: Non-Adaptive Adversary Assumption
"adversary.strategy_fixed_before(detection.threshold)",

# Formal statement:
# For any detector D and adversary A:
#   A's distribution P_A is measurable with respect to sigma-algebra
#   generated by {honest_traces}, NOT {detection.threshold}
#
# Violation scenario:
#   If adversary can query detector O(sqrt(n)) times, can evade detection
#   with probability approaching 1.
#
# Mitigation:
#   - Threshold randomization
#   - Rate limiting on detector API
#   - Ensemble detectors with hidden weights
```

**Add to Section 3.3 (Detection Engine) docstring:**

```python
"""
CRITICAL ASSUMPTION: All detection power guarantees assume NON-ADAPTIVE
adversary. The deceptive distribution P_D is fixed before the detection
threshold tau is chosen.

If adversary has query access to detector:
- B queries -> power degrades by O(B / sqrt(n))
- sqrt(n) queries -> detection power approaches random

Partial mitigation via moving thresholds is available but not complete.
See M-01 invariant for formal statement.
"""
```

**Rationale:** Makes critical assumption explicit (M-01).

---

### Amendment A-09: Invariant - Hyperplane Distribution (wt-9: M-02)

**Section:** 7.2 Security Invariants

**Add to SECURITY_INVARIANTS list:**

```python
# M-02: Hyperplane Distribution Consistency
"geometric.hyperplane_distribution in ['grassmannian', 'ortho_adjusted']",

# If using ortho_group with Uniform([a, b]) offset:
# - Theory assumes Uniform([0, 1]) offset -> lambda = 2r
# - Code uses Uniform([0.2, 0.8]) offset -> lambda = 2r/(b-a) = 2r/0.6 = 3.33r
#
# REQUIRED: Either:
# (a) Use d ~ Uniform([0, 1]) for exact theory match, OR
# (b) Apply lambda adjustment: lambda_eff = 2r / (b - a)
#
# Error bound if using adjusted lambda:
# |p_empirical - p_theory| <= O(r), not O(r^2)
```

**Add to Section 3.1 (Geometric Engine):**

```python
class GeometricEngine:
    """
    Hyperplane intersection volume estimation with adversarial robustness.

    HYPERPLANE DISTRIBUTION:
    Theory (Grassmannian): Uniform on Gr(D-1, D), offset d ~ Uniform([0, 1])
    Implementation options:
      - "grassmannian": Exact match, d ~ Uniform([0, 1])
      - "ortho_adjusted": scipy.ortho_group, d ~ Uniform([a, b]),
                          with lambda adjusted by factor 1/(b-a)

    For offset distribution Uniform([a, b]) with a > 0 or b < 1:
      - Cutting probability becomes 2r/(b-a) for centers away from boundaries
      - Boundary effects become MORE significant (narrower safe zone)
    """

    def __init__(
        self,
        distribution: Literal["grassmannian", "ortho_adjusted"] = "grassmannian",
        offset_range: Tuple[float, float] = (0.0, 1.0),
    ):
        self.distribution = distribution
        self.offset_range = offset_range
        self.lambda_adjustment = 1.0 / (offset_range[1] - offset_range[0])
```

**Rationale:** Aligns code with theory or documents adjustment (M-02).

---

### Amendment A-10: Invariant - Finite Sample Validity (wt-10: M-03)

**Section:** 7.2 Security Invariants

**Add to SECURITY_INVARIANTS list:**

```python
# M-03: Finite Sample Validity
"detection.sample_size >= 100 => |empirical_power - theoretical_power| <= 0.05",

# For n < 100, apply Berry-Esseen correction:
# actual_power >= theoretical_power - 0.4748 / sqrt(n)
#
# Validity regimes:
# - n >= 1000: Error <= 0.015 (excellent)
# - n >= 100:  Error <= 0.05 (good)
# - n >= 30:   Error <= 0.09 (marginal, use with caution)
# - n < 30:    Asymptotic formula unreliable, use exact binomial
```

**Add to Section 3.3 power_analysis method:**

```python
def power_analysis(
    self,
    ...
) -> PowerAnalysisResult:
    """
    ...

    FINITE SAMPLE CORRECTION:
    The asymptotic formula n >= (z_alpha + z_beta)^2 / (D^2 * p) assumes
    normal approximation to binomial. For small n, apply Berry-Esseen:

    If computed n < 100:
        n_corrected = n * (1 + 0.5/sqrt(n))

    Returns:
        n: Required sample size (asymptotic formula)
        n_corrected: With Berry-Esseen correction
        validity_regime: "excellent" | "good" | "marginal" | "unreliable"
    """
```

**Rationale:** Documents finite sample validity regime (M-03).

---

### Amendment A-11: Invariant - Convexity Assumption (wt-11: M-04)

**Section:** 7.2 Security Invariants

**Add to SECURITY_INVARIANTS list:**

```python
# M-04: Convexity Assumption
"geometric.deceptive_region.is_convex == True",

# The topological collapse theorem assumes the deceptive region is a
# convex ball B_r(c). Non-convex regions (torus, point cloud, fractal)
# are NOT SUPPORTED by the current theory.
#
# Non-convex region handling:
# - Torus: Can be decomposed into convex shells, but detection varies
# - Point cloud: Discrete points don't satisfy hyperplane intersection model
# - Fractal: Measure-theoretic complications
#
# FUTURE WORK: Extend theorem to star-convex regions
```

**Add to Section 3.1 (Geometric Engine):**

```python
def estimate_volume(
    self,
    ...
    region_type: Literal["ball", "ellipsoid"] = "ball",
) -> VolumeEstimate:
    """
    ...

    REGION CONSTRAINT:
    Currently supports only convex regions (ball, ellipsoid).
    Non-convex regions are NOT SUPPORTED.

    If region_type not in ["ball", "ellipsoid"]:
        raise NotImplementedError(
            "Non-convex deceptive regions not supported. "
            "Topological collapse theorem requires convexity."
        )
    """
```

**Rationale:** Makes convexity assumption explicit (M-04).

---

### Amendment A-12: Invariant - Independence of Constraints (wt-12: M-05)

**Section:** 7.2 Security Invariants

**Add to SECURITY_INVARIANTS list:**

```python
# M-05: Independence of Constraints
"forall i, j: i != j => geometric.hyperplanes[i] independent_of geometric.hyperplanes[j]",

# For correlated constraints with pairwise correlation rho:
# k_eff = k / (1 + rho * (k - 1))
#
# Impact on volume decay:
# - Independent (rho = 0): V(k) = V(0) * exp(-lambda * k)
# - Correlated (rho > 0): V(k) = V(0) * exp(-lambda * k_eff)
#
# At extreme correlation (rho -> 1):
# k_eff -> 1 regardless of k, providing NO security
```

**Add to Section 3.1 compute_effective_rank method:**

```python
def compute_effective_rank(
    self,
    constraints: List[Hyperplane],
) -> EffectiveRankResult:
    """
    Compute k_eff accounting for constraint correlation.

    FORMULA: k_eff = k / (1 + rho_avg * (k - 1))
    where rho_avg is average pairwise correlation of constraint normals.

    INTERPRETATION:
    - k_eff = k: Fully independent constraints (optimal)
    - k_eff = 1: Fully correlated constraints (no security gain from k > 1)

    Returns:
        effective_rank: k_eff value
        correlation_matrix: Full pairwise correlation matrix
        avg_correlation: rho_avg
        warning: str if rho_avg > 0.5 (significant correlation)
    """
```

**Rationale:** Specifies correlation impact on effective constraints (M-05).

---

### Amendment A-13: Invariant - World Model Size (wt-13: M-06)

**Section:** 7.2 Security Invariants

**Add to SECURITY_INVARIANTS list:**

```python
# M-06: World Model Size Security Parameter
"complexity.world_size >= complexity.security_parameter",
"complexity.security_parameter >= 20",  # Minimum for meaningful security

# The complexity gap T_D / T_H = Omega(2^m) requires m to be sufficiently large.
# For m < 20, brute force is feasible in under 1 second.
#
# Security parameter guidance:
# - m = 20: ~1 second brute force (minimal security)
# - m = 30: ~17 minutes brute force (low security)
# - m = 40: ~12 days brute force (moderate security)
# - m = 50: ~35 years brute force (good security)
# - m = 64: ~600 million years (crypto-strength)
#
# CONDITIONAL ON ETH: Gap is exponential
# UNCONDITIONAL: Gap is at least polynomial (NP vs P)
```

**Rationale:** Documents security parameter requirements (M-06).

---

### Amendment A-14: New Proof Obligations (wt-14: NEW-OBL)

**Section:** 4.1 Proof Obligations

**Add New Table:**

| ID | Obligation | Difficulty | Dependencies |
|----|------------|------------|--------------|
| NEW-01 | Effective rank k_eff correctness | Medium | M-05 |
| NEW-02 | BFT protocol safety | Medium | U-03 |
| NEW-03 | Behavioral correlation detection soundness | Hard | Federation engine |
| NEW-04 | Compositional detection correctness | Very Hard | Detection engine |
| NEW-05 | Slow capture threshold | Medium | Federation engine |

**Add Specifications:**

```
NEW-01 (Effective Rank Correctness):
  Statement: For hyperplanes with pairwise correlation rho,
  k_eff = k / (1 + rho * (k-1)) correctly captures constraint dependence.

  Proof approach: Show that volume decay matches independent case
  with k replaced by k_eff.

NEW-02 (BFT Protocol Safety):
  Statement: PBFT with n >= 3f + 1 satisfies:
  - Safety: No two honest nodes commit conflicting values
  - Liveness: Eventually commits if < n/3 Byzantine

  Proof approach: Reference existing PBFT proofs, verify our implementation matches.

NEW-03 (Behavioral Correlation Detection):
  Statement: If agents pass MI gate independently but coordinate votes
  with correlation > threshold, detection identifies them with probability >= 1-delta.

  Proof approach: Statistical test power analysis for correlation detection.

NEW-04 (Compositional Detection):
  Statement: If inference_graph has honest components but emergent deception,
  compositional_detection identifies the pattern when MI amplification > threshold.

  NOTE: This may be PROVABLY INCOMPLETE. Emergent deception from truly
  honest components may be undetectable in general (see KNOWN_LIMITATIONS.md).

NEW-05 (Slow Capture Threshold):
  Statement: For capture_rate r per period, federation remains secure
  for at least floor(f_max / (r * n)) periods before BFT threshold breach.

  Proof approach: Probabilistic analysis of adversarial node growth.
```

**Rationale:** Documents new proof obligations from FSD extensions (NEW-OBL).

---

### Amendment A-15: ETH Conditionality for Complexity Claims (wt-15: Q4)

**Section:** 3.2 Complexity Engine, 4.1 Proof Obligations

**Add Conditional Claims Framework:**

```python
class ProofObligation(BaseModel):
    id: str
    claim: str
    theorem_statement: str
    lean_file: Optional[str]
    status: Literal["pending", "proven", "disproven", "blocked", "axiomatized"]
    dependencies: List[str]
    conditional_on: List[str] = []  # E.g., ["ETH", "P!=NP"]
```

**Add to Section 3.2 docstring:**

```python
"""
COMPLEXITY CLAIMS AND THEIR CONDITIONS
=======================================

UNCONDITIONAL (provable without assumptions):
- CONSISTENT-LIE is NP-complete (reduction from 3-SAT)
- Honest agent runs in O(n*k) time
- Deceptive agent must solve SAT instances
- Any polynomial-time deceptive agent makes detectable consistency errors

CONDITIONAL ON ETH (Exponential Time Hypothesis):
- T_D / T_H = Omega(2^(m / poly(n)))
- Security parameter m provides 2^(Omega(m)) work factor

CONDITIONAL ON SETH (Strong ETH):
- No algorithm solves k-SAT in O(2^((1-epsilon)*n)) for any epsilon > 0
- Provides tighter lower bounds on deceptive computation

If ETH is false (unlikely), the complexity gap could be subexponential
but still superpolynomial. P != NP suffices for some gap to exist.
"""
```

**Rationale:** Documents conditional nature of complexity claims (Q4).

---

## 3. MEDIUM PRIORITY AMENDMENTS

### Amendment A-16: Lean 4 FFI Interface Specification

**Section:** 4.2 Lean 4 Integration

**Add:**

```python
class LeanIntegration:
    """
    Lean 4 integration specification for formal verification.
    """

    # Version Requirements
    LEAN_VERSION = "4.3.0"
    MATHLIB_VERSION = "v4.3.1"  # Pin specific version

    # Module Dependencies
    LEAN_MODULES = {
        "Hyperplane": [
            "Mathlib.MeasureTheory.Measure.Lebesgue",
            "Mathlib.Topology.MetricSpace.Basic",
            "Mathlib.Analysis.InnerProductSpace.Basic"
        ],
        "SAT": [
            "Mathlib.Computability.NP",
            "Mathlib.Data.Bool.Basic"
        ],
        "Detection": [
            "Mathlib.Probability.Distributions.Gaussian",
            "Mathlib.Analysis.SpecialFunctions.Gaussian"
        ]
    }

    # Extended Proof Status
    ProofStatus = Literal[
        "pending",      # Not started
        "partial",      # Has 'sorry' placeholders
        "axiomatized",  # Uses axiom (e.g., ETH)
        "proven",       # Complete proof
        "disproven",    # Counterexample found
        "blocked",      # Waiting on dependency
    ]

    def call_lean_type_check(self, file: Path) -> TypeCheckResult:
        """
        Invoke Lean type checker via subprocess.

        Protocol: subprocess call to 'lake build', capture stdout/stderr
        Returns: List of errors, warnings, and 'sorry' count
        """
        ...

    def extract_proven_bound(
        self,
        theorem_name: str,
        lean_file: Path,
    ) -> BoundExtraction:
        """
        Parse Lean theorem to extract proven bounds.

        Example: theorem cutting_prob : |p - 2*r| <= C*r^2
        Returns: BoundExtraction(lower=2*r - C*r^2, upper=2*r + C*r^2)
        """
        ...
```

**Rationale:** Specifies Lean FFI interface (REC-M1).

---

### Amendment A-17: Compositional Detection Algorithm (wt-15: Q3)

**Section:** 3.3 Detection Engine

**Replace compositional_detection with:**

```python
def compositional_detection(
    self,
    agent_traces: Dict[AgentId, List[Trace]],
    inference_graph: InferenceGraph,
    mi_threshold: float = 0.1,
) -> CompositionalResult:
    """
    PARTIAL MITIGATION for emergent deception (RT-01).

    Algorithm:
    1. For each node in inference_graph, compute trace statistics
    2. For each edge (A -> B), compute mutual information MI(A_output, B_decision)
    3. Sum individual MI contributions: MI_sum = sum_i MI(A_i, outcome)
    4. Compute total MI: MI_total = MI(all_agents, outcome)
    5. If MI_total - MI_sum > mi_threshold, flag as potential emergent deception

    DETECTION GUARANTEE:
    If emergent deception causes information amplification
    (MI_total >> MI_sum), detects with probability >= 1 - beta.

    KNOWN LIMITATION:
    Does NOT detect emergent deception that preserves individual
    trace distributions. This is a FUNDAMENTAL limitation.
    See KNOWN_LIMITATIONS.md for full analysis.

    Returns:
        emergent_deception_flag: bool
        mi_amplification: MI_total - MI_sum
        suspicious_edges: List of edges with high MI
        confidence: Confidence level of detection
    """
```

**Rationale:** Specifies compositional detection algorithm (Q3).

---

### Amendment A-18: Adaptive Detection Mode

**Section:** 3.3 Detection Engine

**Add method:**

```python
def adaptive_detection(
    self,
    trace_stream: Iterator[Trace],
    initial_threshold: float,
    adaptation_rate: float = 0.01,
    randomization_stddev: float = 0.05,
) -> AdaptiveDetector:
    """
    PARTIAL MITIGATION for adaptive adversaries.

    Algorithm:
    1. Initialize threshold tau = initial_threshold
    2. For each trace batch:
       a. Add Gaussian noise: tau' = tau + N(0, randomization_stddev^2)
       b. Classify traces using tau'
       c. Update tau based on observed distribution shift

    SECURITY PROPERTY:
    Adversary with B threshold queries can only reduce detection
    power by O(B * randomization_stddev / sqrt(n)).

    LIMITATION:
    Does not prevent determined adaptive adversary with unlimited
    queries. See M-01 invariant for full analysis.
    """
```

**Rationale:** Adds adaptive detection mitigation (REC-L2).

---

### Amendment A-19: Error Bounds for Approximations

**Section:** 3.1 Geometric Engine

**Add to estimate_volume:**

```python
def estimate_volume(
    self,
    ...
) -> VolumeEstimate:
    """
    ...

    ERROR BOUNDS:
    The exponential approximation V(k) = V(0) * exp(-lambda*k) has error:

    |V(k) - V(0)*exp(-lambda*k)| / V(0) <= C * r^2 * k

    where C is a universal constant (empirically C ~ 0.5).

    For r = 0.1, k = 50: error bound ~ 0.5 * 0.01 * 50 = 0.25 (25%)
    For r = 0.05, k = 50: error bound ~ 0.5 * 0.0025 * 50 = 0.0625 (6.25%)

    Returns:
        volume: Point estimate
        ci_lower, ci_upper: 95% confidence interval
        error_bound: Theoretical error bound from formula
        empirical_error: Observed deviation from exponential fit
    """
```

**Rationale:** Specifies error bounds (REC-H2).

---

### Amendment A-20: Proof Obligation Dependency Graph

**Section:** 4.1 Proof Obligations

**Add:**

```
PROOF OBLIGATION DEPENDENCY GRAPH
=================================

Level 0 (No dependencies):
- CA-1 (NP membership)
- CA-2 (3-SAT reduction)
- DP-2 (Neyman-Pearson)

Level 1 (Depends on Level 0):
- TC-1 (Cutting probability) <- Measure theory foundations
- CA-3 (Honest agent bound) <- CA-1
- CA-4 (Deceptive needs SAT) <- CA-2
- DP-1 (LRT distribution) <- DP-2

Level 2:
- TC-2 (Fubini/Independence) <- TC-1
- TC-5 (Monotonicity) <- TC-1
- DP-3 (Sample complexity) <- DP-1
- DP-6 (Monotonicity) <- DP-1
- CA-5 (Gap amplification) <- CA-4, ETH axiom

Level 3:
- TC-3 (Volume scaling) <- TC-2
- TC-7 (k_eff formula) <- TC-2, TC-5
- DP-4 (Asymptotic validity) <- DP-3
- DP-5 (Plug-in error) <- DP-3, DP-6

Level 4:
- TC-4 (Error bound) <- TC-3
- TC-8 (Uniform convergence) <- TC-3, TC-4
- TC-6 (Boundary negligible) <- TC-8

NEW OBLIGATIONS (from FSD extensions):
- NEW-01 <- TC-7
- NEW-02 <- (external PBFT proofs)
- NEW-03 <- DP-3
- NEW-04 <- DP-1, DP-3 (partial, see limitations)
- NEW-05 <- (probability theory)
```

**Rationale:** Specifies proof dependency order (REC-M2).

---

### Amendment A-21: Session Types for Agent Protocols (Future Work)

**Add New Section 11: FUTURE WORK**

```markdown
## 11. FUTURE WORK

### 11.1 Type-Theoretic Framework for Systemic Deception

The current framework treats deception as an individual agent property.
Capturing systemic deception (Q5) requires extensions:

**Proposed Approach: Deception-Indexed Session Types**

```
-- Session type for honest interaction
type HonestSession =
  !Query . ?Response{honesty_level: 0} . end

-- Session type for potentially deceptive interaction
type DeceptiveSession(D: DeceptionLevel) =
  !Query . ?Response{honesty_level: D} . end

-- Composition rule with amplification bound
compose : Session(D1) || Session(D2) -> Session(max(D1, D2, amplify(D1, D2)))
```

This would allow static analysis of deception potential in multi-agent systems.

**Required Research:**
1. Formalize deception levels as bounded reals or lattice elements
2. Prove composition rules preserve soundness
3. Implement as type-level assertions in Python (Protocol + TypeVar)
4. Integrate with existing compositional detection

**Estimated Effort:** 6-12 months research project

### 11.2 Adaptive Adversary Game-Theoretic Analysis

Full treatment of adaptive adversaries requires game-theoretic equilibrium analysis:
- Model as repeated game between detector and adversary
- Compute Nash equilibria for threshold selection
- Derive optimal randomization strategies

### 11.3 Non-Convex Deceptive Region Extensions

Extend topological collapse to:
- Star-convex regions (tractable extension)
- Finite unions of convex sets (summation over components)
- Fractal boundaries (measure-theoretic complications)
```

**Rationale:** Documents future directions for Q5 and other extensions.

---

## 4. SUMMARY OF ALL AMENDMENTS

| ID | Section | Priority | Source | Description |
|----|---------|----------|--------|-------------|
| A-01 | 3.3 | CRITICAL | wt-1 | Power formula preconditions |
| A-02 | 3.2 | CRITICAL | wt-2 | k >= 3 NP-hardness enforcement |
| A-03 | 3.4 | CRITICAL | wt-3 | BFT protocol implementation |
| A-04 | 6.2 | CRITICAL | wt-4 | Type hole discriminated union |
| A-05 | 2.3 | CRITICAL | wt-5 | Refinement types for all interfaces |
| A-06 | 4.1 | HIGH | wt-6 | TC proof obligations (TC-2,3,4,8) |
| A-07 | 4.1 | HIGH | wt-7 | DP proof obligations (DP-4,5,6) |
| A-08 | 7.2 | HIGH | wt-8 | M-01: Non-adaptive invariant |
| A-09 | 7.2 | HIGH | wt-9 | M-02: Hyperplane distribution |
| A-10 | 7.2 | HIGH | wt-10 | M-03: Finite sample validity |
| A-11 | 7.2 | HIGH | wt-11 | M-04: Convexity assumption |
| A-12 | 7.2 | HIGH | wt-12 | M-05: Independence of constraints |
| A-13 | 7.2 | HIGH | wt-13 | M-06: World model size |
| A-14 | 4.1 | HIGH | wt-14 | New proof obligations (NEW-01 to 05) |
| A-15 | 3.2,4.1 | HIGH | wt-15 | ETH conditionality |
| A-16 | 4.2 | MEDIUM | wt-15 | Lean 4 FFI specification |
| A-17 | 3.3 | MEDIUM | wt-15 | Compositional detection algorithm |
| A-18 | 3.3 | MEDIUM | wt-15 | Adaptive detection mode |
| A-19 | 3.1 | MEDIUM | wt-15 | Error bounds for approximations |
| A-20 | 4.1 | MEDIUM | wt-15 | Proof obligation dependency graph |
| A-21 | 11 | MEDIUM | wt-15 | Future work section |

---

**END OF AMENDMENTS DOCUMENT**

*This document synthesizes amendments from all 15 parallel worktrees. Ready for coordinator merge.*
