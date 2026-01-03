# RATCHET Platform Functional Specification Document

**Version:** 1.0
**Status:** Draft for Formal Methods Review

---

## 1. EXECUTIVE SUMMARY

RATCHET (Research Architecture for Testing Coherence, Honesty, and Ethical Topology) is a research platform for empirically validating the Coherence Ratchet alignment framework. The platform provides:

1. **Simulation Engine** - Monte Carlo validation of geometric, complexity, and statistical claims
2. **Adversarial Testing** - Red team attack scenarios and robustness verification
3. **Formal Verification Interface** - Lean 4 proof extraction and validation
4. **Experimental Framework** - Human-subject and multi-agent experiment orchestration
5. **Federation Testbed** - Distributed precedent accumulation and Sybil resistance testing

**Critical Design Constraint:** This is a RESEARCH TESTBED, not production infrastructure. All security claims must be validated against adversarial scenarios before any deployment consideration.

### SECURITY_ASSUMPTION: Non-Adaptive Adversary

```
CRITICAL ASSUMPTION: All detection power guarantees in this specification
assume a NON-ADAPTIVE adversary.

Formal Statement:
  adversary.strategy_fixed_before(detection.threshold)

  Equivalently: The deceptive distribution P_D is fixed BEFORE the detector
  threshold tau is chosen. The adversary cannot observe detector responses
  and adapt their strategy.

This assumption is VIOLATED when:
  - Adversary can probe detector responses (API access)
  - Adversary has white-box access to detection algorithm
  - Adversary can update strategy based on detection outcomes over time

Consequence of Violation (see Section 11 - Known Limitations):
  - Adaptive adversary with O(sqrt(n)) queries can evade detection
  - Detection power degrades by factor O(B / sqrt(n)) for B-adaptive adversary
  - All power guarantees in Section 3.3 become invalid

Mitigation (partial): Moving thresholds, ensemble detectors, rate limiting
```

---

## 2. ARCHITECTURE

### 2.1 System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          RATCHET PLATFORM                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   REST API   │  │  WebSocket   │  │   CLI        │  │  Lean 4 FFI  │ │
│  │  (FastAPI)   │  │  (Live Sim)  │  │  Interface   │  │  (Proofs)    │ │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘ │
│         │                 │                 │                 │          │
│  ┌──────┴─────────────────┴─────────────────┴─────────────────┴───────┐ │
│  │                    ORCHESTRATION LAYER                              │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │ │
│  │  │ Experiment  │  │  Simulation │  │  Validation │                 │ │
│  │  │ Controller  │  │  Scheduler  │  │  Pipeline   │                 │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                 │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                    │                                      │
│  ┌─────────────────────────────────┴───────────────────────────────────┐ │
│  │                         CORE ENGINES                                 │ │
│  │                                                                      │ │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐         │ │
│  │  │  GEOMETRIC     │  │  COMPLEXITY    │  │  DETECTION     │         │ │
│  │  │  ENGINE        │  │  ENGINE        │  │  ENGINE        │         │ │
│  │  │                │  │                │  │                │         │ │
│  │  │ • Hyperplane   │  │ • SAT Solver   │  │ • LRT Detector │         │ │
│  │  │   Intersection │  │   (Z3/MiniSat) │  │ • Power Curves │         │ │
│  │  │ • Correlated   │  │ • Honest Agent │  │ • Adaptive     │         │ │
│  │  │   Constraints  │  │ • Deceptive    │  │   Detection    │         │ │
│  │  │ • Null Space   │  │   Agent        │  │ • Compositional│         │ │
│  │  │   Attack Sim   │  │ • Sparse Lie   │  │   Reasoning    │         │ │
│  │  └────────────────┘  └────────────────┘  └────────────────┘         │ │
│  │                                                                      │ │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐         │ │
│  │  │  FEDERATION    │  │  RED TEAM      │  │  FORMAL        │         │ │
│  │  │  ENGINE        │  │  ENGINE        │  │  VERIFICATION  │         │ │
│  │  │                │  │                │  │                │         │ │
│  │  │ • Sybil Gate   │  │ • Attack       │  │ • Lean 4       │         │ │
│  │  │ • Precedent    │  │   Scenarios    │  │   Integration  │         │ │
│  │  │   Accumulation │  │ • Distribution │  │ • Proof        │         │ │
│  │  │ • BFT Protocol │  │   Mimicry      │  │   Obligations  │         │ │
│  │  │ • Behavioral   │  │ • Slow Capture │  │ • Certificate  │         │ │
│  │  │   Correlation  │  │   Simulation   │  │   Checking     │         │ │
│  │  └────────────────┘  └────────────────┘  └────────────────┘         │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                    │                                      │
│  ┌─────────────────────────────────┴───────────────────────────────────┐ │
│  │                      DATA & PERSISTENCE                              │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │ │
│  │  │ PostgreSQL   │  │ TimescaleDB  │  │ Redis        │               │ │
│  │  │ (Experiments)│  │ (Time Series)│  │ (Cache/Queue)│               │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘               │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Directory Structure

```
ratchet/
├── api/
│   ├── main.py                 # FastAPI application entry
│   ├── dependencies.py         # DI container
│   └── routers/
│       ├── experiments.py      # /experiments endpoints
│       ├── simulations.py      # /simulations endpoints
│       ├── validation.py       # /validation endpoints
│       ├── federation.py       # /federation endpoints
│       └── proofs.py           # /proofs endpoints
├── core/
│   ├── engines/
│   │   ├── geometric.py        # Hyperplane intersection engine
│   │   ├── complexity.py       # SAT-based deception cost
│   │   ├── detection.py        # Statistical detection
│   │   ├── federation.py       # Distributed consensus
│   │   └── redteam.py          # Adversarial scenarios
│   ├── agents/
│   │   ├── honest.py           # Honest agent implementation
│   │   ├── deceptive.py        # Deceptive agent (SAT-based)
│   │   ├── adaptive.py         # Adaptive adversary
│   │   └── sybil.py            # Sybil attack agents
│   └── formal/
│       ├── lean_bridge.py      # Lean 4 FFI
│       ├── proof_checker.py    # Certificate verification
│       └── obligations.py      # Proof obligation tracker
├── schemas/
│   ├── experiment.py           # Experiment configuration
│   ├── simulation.py           # Simulation parameters
│   ├── results.py              # Result types
│   ├── attack.py               # Attack scenario definitions
│   └── proof.py                # Proof artifact types
├── experiments/
│   ├── protocols/
│   │   ├── geometric_sweep.py
│   │   ├── complexity_benchmark.py
│   │   ├── detection_power.py
│   │   ├── federation_stress.py
│   │   └── human_calibration.py
│   └── analysis/
│       ├── bootstrap.py        # Bootstrap confidence intervals
│       ├── power_analysis.py   # Statistical power
│       └── convergence.py      # Monte Carlo convergence
├── adversarial/
│   ├── attacks/
│   │   ├── null_space.py       # Constraint-aligned deception
│   │   ├── distribution_mimicry.py
│   │   ├── diverse_sybil.py
│   │   ├── slow_capture.py
│   │   └── emergent_deception.py
│   └── defenses/
│       ├── behavioral_correlation.py
│       ├── drift_detection.py
│       └── compositional_reasoning.py
├── formal/
│   ├── lean/
│   │   ├── Hyperplane.lean
│   │   ├── SAT.lean
│   │   ├── Detection.lean
│   │   └── lakefile.lean
│   └── proofs/
│       ├── cutting_probability.lean
│       ├── np_completeness.lean
│       └── lrt_distribution.lean
├── config/
│   ├── settings.py
│   └── loader.py
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── adversarial/            # Red team test suite
│   └── formal/                 # Proof verification tests
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

---

## 3. CORE ENGINES

### 3.1 Geometric Engine

**Purpose:** Validate topological collapse theorem V(k) = V(0) x exp(-lambda*k)

**Critical Invariant - Constraint Independence (M-05):**

The exponential decay of deceptive volume requires constraints (hyperplanes) to be independent:

```
INVARIANT M-05: forall i, j: i != j => hyperplanes[i] independent_of hyperplanes[j]
```

**Formal Definition:** Two hyperplanes H_i = {x : <n_i, x> = d_i} and H_j = {x : <n_j, x> = d_j} are independent iff their normal vectors are linearly independent: n_i and n_j span a 2-dimensional subspace (i.e., n_i is not a scalar multiple of n_j).

**Why This Matters (Red Team Analysis 1.2):** When constraints are correlated (e.g., from shared training data, memetic propagation, or authority bias), the effective number of constraints collapses dramatically. For pairwise correlation rho:

```
k_eff = k / (1 + rho * (k - 1))

Example: k = 100 constraints with rho = 0.7 correlation
k_eff = 100 / (1 + 0.7 * 99) = 100 / 70.3 = 1.42

=> 100 correlated constraints provide protection of only ~1.4 independent constraints!
```

**Correlation Impact on Security:**

| Correlation (rho) | k = 10 | k = 50 | k = 100 |
|-------------------|--------|--------|---------|
| 0.0 (ideal)       | 10.0   | 50.0   | 100.0   |
| 0.3               | 2.63   | 3.21   | 3.29    |
| 0.5               | 1.82   | 1.96   | 1.98    |
| 0.7               | 1.37   | 1.42   | 1.42    |
| 0.9               | 1.10   | 1.11   | 1.11    |

**Adjusted Decay Rate:** For correlated constraints, the effective decay rate becomes:

```
lambda_eff = lambda_0 / (1 + rho * (k - 1))

where lambda_0 = 2r (ideal cutting probability for radius r)
```

**Preconditions (REQUIRED):**
```
PRECONDITION-GE-01: geometric.deceptive_region.is_convex == True
```

> **INVARIANT:** The deceptive region MUST be convex (e.g., ball, ellipsoid, convex polytope).
> Non-convex regions (torus, point cloud, fractal, disconnected components) are **UNSUPPORTED**
> in the current mathematical framework. The exponential volume shrinkage theorem (Theorem 1.1)
> relies on convexity for the cutting probability calculation. Non-convex regions can have
> disconnected components that survive hyperplane cuts differently, invalidating the
> independence assumptions in the proof.
>
> **Reference:** FORMALIZATION_ROADMAP.md Section 1.1 (Implicit Assumptions) and Section 4.1.2

**Requirements from Formal Methods:**
- Must support both orthonormal AND correlated constraint sampling
- Must track effective rank k_eff = k / (1 + rho*(k-1))
- MUST verify constraint independence before applying exponential decay claims
- Must quantify boundary effects at edge of [0,1]^D
- Must provide error bounds on exponential approximation
- **Must enforce convexity precondition on deceptive region inputs**

**Requirements from Red Team:**
- MUST implement null-space attack simulation
- MUST support adversarial constraint probing
- MUST test non-convex deceptive regions (torus, point cloud, fractal) **for attack surface analysis only**
- MUST support adaptive/moving target deception
- MUST flag non-convex inputs as potential attack vectors (see RT-06 below)

```python
class GeometricEngine:
    """
    Hyperplane intersection volume estimation with adversarial robustness.

    All parameters use refinement types from schemas/types.py to ensure
    type safety and prevent runtime errors from invalid inputs.
    """

    def estimate_volume(
        self,
        dimension: Dimension,                    # int > 0
        num_constraints: NumConstraints,         # int > 0
        deceptive_radius: Radius,                # 0 < r < 0.5
        constraint_correlation: Correlation = 0.0,  # -1 <= rho <= 1
        sampling_mode: SamplingMode = SamplingMode.ORTHONORMAL,
        num_samples: SampleSize = 100_000,       # int >= 1
        adversary: Optional[AdversarialStrategy] = None,
    ) -> VolumeEstimate:
        """
        Returns volume estimate with bootstrap confidence interval.

        If adversary is provided, simulates null-space or moving-target attack.

        Type constraints prevent:
        - T-GEO-01: dimension <= 0 (crashes)
        - T-GEO-02: radius outside (0, 0.5) (boundary violations)
        - T-GEO-03: correlation outside [-1, 1] (invalid matrices)
        """
        ...

    def compute_effective_rank(
        self,
        constraints: List[Hyperplane],
    ) -> EffectiveRankResult:
        """
        Compute k_eff = k / (1 + rho*(k-1)) accounting for constraint correlation.
        Returns effective rank and correlation matrix.

        ALGORITHM:
        1. Extract normal vectors N = [n_1, n_2, ..., n_k] from constraints
        2. Compute Gram matrix G where G[i,j] = <n_i, n_j> / (||n_i|| * ||n_j||)
        3. Compute average off-diagonal correlation: rho = mean(|G[i,j]|) for i != j
        4. Compute effective rank: k_eff = k / (1 + rho * (k - 1))

        ALTERNATIVE (SVD-based):
        1. Form matrix N with rows as normalized normals
        2. Compute singular values sigma_1 >= sigma_2 >= ... >= sigma_k
        3. k_eff = (sum(sigma_i))^2 / sum(sigma_i^2)  [participaton ratio]

        RETURNS:
        - effective_rank: float (k_eff value)
        - correlation_matrix: np.ndarray (k x k Gram matrix)
        - average_correlation: float (rho value)
        - independence_violations: List[Tuple[int, int]] (pairs with |rho| > threshold)

        INVARIANT CHECK:
        If average_correlation > 0.3, emit warning:
        "Constraint correlation exceeds safe threshold. Security claims may be weakened."
        """
        ...

    def null_space_attack(
        self,
        constraints: List[Hyperplane],
        probe_budget: SampleSize,               # int >= 1
    ) -> NullSpaceAttackResult:
        """
        Simulate adversary probing constraint directions and placing
        deception in null space. Returns evasion success rate.
        """
        ...

    def verify_independence(
        self,
        constraints: List[Hyperplane],
        correlation_threshold: float = 0.3,
    ) -> IndependenceVerificationResult:
        """
        Verify that constraint set satisfies independence invariant M-05.

        VERIFICATION PROCEDURE:
        1. Compute Gram matrix of normalized normal vectors
        2. Check: forall i != j: |<n_i, n_j>| < correlation_threshold
        3. If violated, identify problematic pairs
        4. Compute effective rank k_eff

        RETURNS:
        - is_independent: bool (True if invariant holds)
        - violations: List[Tuple[int, int, float]] (pairs exceeding threshold with correlation value)
        - effective_rank: float
        - recommended_action: str (e.g., "Remove constraint 3", "Add orthogonal constraints")

        THRESHOLD GUIDANCE:
        - correlation_threshold = 0.1: Strict (near-orthogonal required)
        - correlation_threshold = 0.3: Standard (moderate correlation tolerated)
        - correlation_threshold = 0.5: Lenient (significant correlation allowed)
        """
        ...

    def enforce_independence(
        self,
        constraints: List[Hyperplane],
        method: Literal["drop_correlated", "orthogonalize", "diversify"] = "orthogonalize",
    ) -> IndependenceEnforcementResult:
        """
        Transform constraint set to satisfy independence invariant.

        METHODS:
        1. "drop_correlated": Remove constraints with highest correlation until
           independence holds. Fast but loses information.

        2. "orthogonalize": Apply Gram-Schmidt to normal vectors. Preserves
           count but changes constraint directions.

        3. "diversify": Add new orthogonal constraints sampled from null space
           of existing constraints. Increases protection without removal.

        RETURNS:
        - transformed_constraints: List[Hyperplane]
        - dropped_indices: List[int] (for "drop_correlated")
        - transformation_matrix: np.ndarray (for "orthogonalize")
        - added_constraints: List[Hyperplane] (for "diversify")
        - new_effective_rank: float
        """
        ...
```

**Validation Protocol (Computational Scientist):**
1. Convergence test: Run at N=1K, 10K, 100K, 1M samples; verify CI shrinks as 1/√N
2. Cross-validation: 10-fold CV on constraint sampling; report variance
3. Sensitivity: Sweep r ∈ [0.01, 0.5], D ∈ [10, 1000], k ∈ [1, 100]
4. Replication: 100 independent runs; report λ with 95% CI

### 3.2 Complexity Engine

**Purpose:** Validate computational asymmetry T_D / T_H = Omega(2^m / poly(n))

**Requirements from Formal Methods:**
- MUST use industrial SAT solver (Z3, MiniSat, CaDiCaL), NOT brute force
- MUST distinguish 3-SAT (NP-hard) from 2-SAT (P) regimes
- MUST track actual solving time, not theoretical complexity
- MUST support incremental SAT with learned clause reuse

**Requirements from Red Team:**
- MUST implement sparse deception (lie about unobservable facts only)
- MUST implement just-in-time fabrication (lazy deception)
- MUST test structured world models (Horn clauses, 2-SAT)
- MUST benchmark against state-of-art SAT solvers

**Refinement Types (from schemas/types.py):**
```python
from schemas.types import (
    WorldSize,           # int >= 1: World model size m
    NumStatements,       # int > 0: Number of statements n
    Literals,            # int >= 3: Literals per statement (NP-hardness requirement)
    ObservableFraction,  # 0 < float <= 1: Fraction of observable facts
    DeceptionStrategy,   # Enum: full, sparse, lazy
    SATSolver,           # Enum: z3, minisat, cadical, bruteforce
    ComplexityResult,    # Result with T_H, T_D, ratio, CI
)

# CRITICAL: T-CPX-01 - Literals >= 3 required for NP-hardness
# For k < 3, problem is 2-SAT (in P) and security claims are INVALID
```

```python
from typing import Annotated
from pydantic import Field

# SECURITY-CRITICAL TYPE CONSTRAINT:
# NP-hardness of CONSISTENT-LIE requires k >= 3 (3-SAT reduction).
# For k = 2, problem reduces to 2-SAT which is in P (polynomial time).
# For k < 2, trivially solvable.
#
# SECURITY IMPLICATION: If k < 3, the computational asymmetry T_D >> T_H
# vanishes entirely. An adversary using only 2-literal statements can
# maintain deceptive consistency in polynomial time, defeating the
# core complexity-based security property.

LiteralsPerStatement = Annotated[
    int,
    Field(
        ge=3,
        description="Number of literals per statement. MUST be >= 3 for NP-hardness. "
                    "k=2 reduces to 2-SAT (P-time tractable). "
                    "k<3 voids all complexity gap security claims."
    )
]

class ComplexityEngine:
    """
    Deception complexity analysis with realistic SAT solving.

    SECURITY WARNING: All complexity gap guarantees (T_D / T_H = Omega(2^m))
    require literals_per_statement >= 3. For k < 3, the problem is tractable
    and NO computational asymmetry exists between honest and deceptive agents.
    """

    def __init__(
        self,
        solver: SATSolver = SATSolver.Z3,
    ):
        self.solver = solver

    def measure_complexity(
        self,
        world_size: int,
        num_statements: int,
        literals_per_statement: LiteralsPerStatement,  # ENFORCED: >= 3
        observable_fraction: float = 1.0,  # Sparse deception parameter
        deception_strategy: Literal["full", "sparse", "lazy"] = "full",
    ) -> ComplexityResult:
        """
        Compare honest vs deceptive agent computational cost.
        Returns T_H, T_D, ratio, and 95% CI.

        Type constraints prevent:
        - T-CPX-01: literals < 3 (false NP-hardness claims)
        - T-CPX-02: observable_fraction = 0 (division by zero)

        CONDITIONAL ON ETH: Exponential gap T_D/T_H = Omega(2^m)
        """
        ...

    def incremental_sat_cost(
        self,
        statements: List[Statement],
        use_learned_clauses: bool = True,
    ) -> IncrementalCostResult:
        """
        Measure cost with clause learning and phase saving.
        Compares to naive brute-force baseline.
        """
        ...

    def sparse_deception_attack(
        self,
        world_model: WorldModel,
        observable_facts: Set[int],
        num_lies: SampleSize,                        # int >= 1
    ) -> SparseAttackResult:
        """
        Adversary lies only about unobservable facts.
        Returns cost reduction factor vs full deception.
        """
        ...
```

**Validation Protocol (Computational Scientist):**
1. Solver benchmark: Compare Z3, MiniSat, CaDiCaL on same instances
2. Scaling law: Fit T_D(m) = a × 2^(b×m) + c; validate b approx 1
3. Ratio validation: Report T_D/T_H with error bars across parameter sweep
4. Edge cases: m <= 15 (brute force feasible)
5. **SECURITY BOUNDARY TEST (U-02):** Verify k=2 yields T_D/T_H approx 1 (NO gap)

**k < 3 Security Boundary (CRITICAL):**
- For k = 2: MUST demonstrate complexity gap collapses to O(1)
- For k = 3: MUST demonstrate exponential gap emerges
- Transition boundary k in {2, 3} is the security cliff edge
- Any system accepting k < 3 MUST emit security warning
- CI pipeline MUST reject configurations with k < 3 unless explicitly overridden

### 3.3 Detection Engine

**Purpose:** Validate statistical detection power and sample complexity

**CRITICAL ASSUMPTION: NON-ADAPTIVE ADVERSARY**

All detection power claims in this section assume `adversary.strategy_fixed_before(detection.threshold)`.
See Section 1 SECURITY_ASSUMPTION and Section 11 Known Limitations for details on what breaks
with adaptive adversaries. Specifically:
- Power formula `n >= (z_alpha + z_beta)^2 / (D^2 * p)` assumes P_D is fixed before threshold selection
- LRT optimality (Neyman-Pearson) requires non-adaptive adversary
- Adaptive adversary can evade with O(sqrt(n)) threshold queries

**Requirements from Formal Methods:**
- MUST implement Neyman-Pearson optimal LRT (NON-ADAPTIVE ADVERSARY ONLY)
- MUST compute Mahalanobis distance with proper covariance estimation
- MUST provide asymptotic AND finite-sample analysis
- MUST quantify estimation error in plug-in D_hat

**Finite Sample Validity Invariant (M-03):**

The asymptotic power formula is valid for n >= 100. For smaller samples, Berry-Esseen
corrections are required.

**Invariant FS-1 (Asymptotic Validity Threshold):**
```
n >= 100 => asymptotic_formula_valid
```
For n >= 100, the Gaussian approximation to the likelihood ratio distribution has
error bounded by O(1/sqrt(n)), making the asymptotic power formula reliable.

**Invariant FS-2 (Power Approximation Accuracy):**
```
n >= 100 => |empirical_power - theoretical_power| <= 0.05
```
When n >= 100, the empirical detection power must match the theoretical prediction
within 5 percentage points. Violations indicate model misspecification or
distributional departures from Gaussian assumptions.

**Berry-Esseen Correction for n < 100:**

For finite samples with n < 100, apply the Berry-Esseen correction:
```
|F_n(x) - Phi(x)| <= C * rho / (sigma^3 * sqrt(n))
```
where:
- F_n(x) is the empirical CDF of the standardized sample mean
- Phi(x) is the standard normal CDF
- C <= 0.4748 (Berry-Esseen constant, Shevtsova 2011)
- rho = E[|X - mu|^3] is the third absolute moment
- sigma is the standard deviation

**Corrected Power Formula (n < 100):**
```
power_corrected = power_asymptotic +/- delta_BE
delta_BE = 0.4748 * rho / (sigma^3 * sqrt(n))
```

**Small Sample Fallback (n < 30):**
For n < 30, the asymptotic formula is unreliable. Use one of:
1. **Exact permutation test:** Compute exact distribution under null
2. **Bootstrap power estimation:** B = 10000 bootstrap replicates
3. **Conservative adjustment:** Apply Bonferroni-style penalty: alpha_eff = alpha / 2

**Dependency:** This invariant depends on wt-7 (DP-4 asymptotic validity specification).
The Berry-Esseen bounds are validated against the asymptotic regime established there.

**Requirements from Red Team:**
- MUST implement distribution mimicry attack (moment matching)
- MUST implement trace flooding/dilution attack
- MUST test non-Gaussian distributions (heavy-tailed, mixture)
- MUST implement adaptive detection (moving thresholds) as partial mitigation

**Refinement Types (from schemas/types.py):**
```python
from schemas.types import (
    MahalanobisDistance,  # float >= 0: Non-negative by definition
    Probability,          # 0 < float < 1: For alpha, beta, deception_rate
    SampleSize,           # int >= 1: Sample size n
    DetectionMethod,      # Enum: lrt, mahalanobis, isolation_forest, ensemble
    AttackType,           # Enum: mimicry, flooding, adaptive, etc.
    InferenceGraph,       # Graph for compositional detection
    PowerAnalysisResult,  # Result with required n, achieved power
)

# PRECONDITIONS for power_analysis formula validity:
# - D >= 0.5 (otherwise sample size infeasible)
# - p >= 0.001 (otherwise sample size infeasible)
# - n >= 100 for asymptotic regime (add Berry-Esseen correction for smaller n)
```

#### 3.3.1 Power Formula Preconditions

The sample complexity formula for detection power:

```
n >= (z_alpha + z_beta)^2 / (D^2 * p)
```

Where:
- `n` = required sample size
- `z_alpha` = standard normal quantile for significance level alpha
- `z_beta` = standard normal quantile for power (1 - beta)
- `D` = Mahalanobis distance between honest and deceptive distributions
- `p` = deception rate (proportion of deceptive traces)

**PRECONDITIONS (U-01 Fix):**

| Precondition | Constraint | Rationale |
|--------------|------------|-----------|
| Mahalanobis Distance | `D >= 0.5` | Detection is statistically infeasible for `D < 0.5`; the required sample sizes become impractically large (n > 100,000 for typical alpha/beta). For `D < 0.5`, consider increasing feature separation or using alternative detection methods. |
| Deception Rate | `p in (0.001, 1)` exclusive | Division by zero when `p = 0`; formula degenerates when `p = 1` (no honest baseline). Lower bound `p > 0.001` ensures feasible sample sizes (n < 10^7). |
| Sample Size Validity | `n >= 100` | Asymptotic normality assumption. For `n < 100`, apply Berry-Esseen correction: add `C / sqrt(n)` to error bound where `C <= 0.5`. |
| Distribution Assumption | Gaussian or sub-Gaussian | LRT optimality and z-score formula require (approximately) Gaussian distributions. For heavy-tailed distributions, use robust estimators and inflate required n by factor 2-3. |

**Berry-Esseen Finite-Sample Correction:**

For sample sizes `30 <= n < 100`, the power estimate should be adjusted:

```
power_corrected = power_asymptotic - C_BE / sqrt(n)
```

Where `C_BE <= 0.4748` (Berry-Esseen constant). For `n < 30`, the asymptotic formula is unreliable; use exact binomial or permutation tests instead.

```python
class DetectionEngine:
    """
    Statistical deception detection with adversarial robustness.

    ASSUMPTION: Non-adaptive adversary (adversary.strategy_fixed_before(detection.threshold))
    For adaptive adversaries, see RT-04 mitigation with moving thresholds.
    """

    def train_detector(
        self,
        honest_traces: np.ndarray,
        deceptive_traces: Optional[np.ndarray] = None,
        method: DetectionMethod = DetectionMethod.LRT,
    ) -> Detector:
        """
        Train detector on honest traces. If deceptive_traces provided,
        use for supervised training; otherwise, use unsupervised.
        """
        ...

    def power_analysis(
        self,
        mahalanobis_distance: MahalanobisDistance,  # float >= 0
        deception_rate: Probability,                 # 0 < p < 1
        alpha: Probability = 0.05,                   # 0 < alpha < 1
        beta: Probability = 0.05,                    # 0 < beta < 1
    ) -> PowerAnalysisResult:
        """
        Compute required sample size for given power.
        Returns n, analytical formula, and finite-sample correction.

        PRECONDITIONS:
        --------------
        1. mahalanobis_distance >= 0.5
           Detection is statistically infeasible below D=0.5. For smaller
           effect sizes, the required sample size becomes impractically large.
           If D < 0.5, consider: (a) increasing feature separation through
           better trace engineering, (b) using ensemble methods, or
           (c) accepting lower detection power.

        2. deception_rate in (0.001, 1) exclusive
           - p = 0 causes division by zero in the sample size formula
           - p = 1 means all traces are deceptive (no honest baseline)
           - p < 0.001 requires n > 10^7 which is typically infeasible
           For very rare deception (p < 0.001), use sequential testing or
           change detection methods instead.

        3. Result valid for n >= 100 (asymptotic regime)
           The formula assumes asymptotic normality via CLT. For n < 100,
           apply Berry-Esseen correction:
             power_corrected = power - 0.4748 / sqrt(n)
           For n < 30, use exact methods (binomial, permutation tests).

        4. Assumes approximately Gaussian distributions
           LRT optimality requires Gaussian P_H and P_D. For heavy-tailed
           or mixture distributions, inflate sample size by 2-3x or use
           robust Mahalanobis distance estimation.

        RETURNS:
        --------
        PowerAnalysisResult containing:
          - n: required sample size (asymptotic)
          - n_corrected: sample size with finite-sample adjustment
          - power: achieved power at computed n
          - validity_regime: "asymptotic" | "corrected" | "exact_required"
          - warnings: list of any precondition concerns
        """
        ...

    def adversarial_robustness(
        self,
        detector: Detector,
        attack: AttackType,
        attack_params: Dict,
    ) -> RobustnessResult:
        """
        Test detector against adversarial attacks.
        Returns degraded power and required mitigation.
        """
        ...

    def compositional_detection(
        self,
        agent_traces: Dict[AgentId, List[Trace]],
        inference_graph: InferenceGraph,
    ) -> CompositionalResult:
        """
        Detect emergent deception from honest components.
        Analyzes inference chains, not individual traces.

        PARTIAL MITIGATION for RT-01 (emergent deception).
        Does NOT detect emergent deception that preserves
        individual trace distributions.
        """
        ...
```

### 3.4 Federation Engine

**Purpose:** Test distributed precedent accumulation and Sybil resistance

**Requirements from Red Team (CRITICAL):**
- MUST implement actual BFT consensus protocol (not just specification)
- MUST test diverse Sybil attack (pass MI gate, coordinate behavior)
- MUST implement behavioral correlation detection
- MUST simulate slow capture over extended time horizons
- MUST quantify Byzantine threshold (n >= 3f + 1)

#### 3.4.1 PBFT Protocol Specification

**Protocol Selection:** PBFT (Practical Byzantine Fault Tolerance) is selected as the consensus protocol for the following reasons:
1. **Battle-tested:** Deployed in production systems (IBM Hyperledger Fabric)
2. **Proven guarantees:** Safety and liveness under partial synchrony
3. **Optimal resilience:** Tolerates f Byzantine faults with n = 3f + 1 replicas (optimal)
4. **Deterministic finality:** No probabilistic confirmation required

**Byzantine Fault Tolerance Invariant:**

```
INVARIANT: n >= 3f + 1

Where:
  n = total number of federation replicas
  f = maximum number of Byzantine (faulty/malicious) replicas tolerated

Derivation:
  - Quorum size Q = 2f + 1 (majority of honest replicas)
  - For safety: Any two quorums must intersect in at least one honest replica
    Q + Q - n >= 1 honest replica
    => 2(2f + 1) - n >= f + 1
    => n >= 3f + 1
  - For liveness: Must have enough honest replicas to form quorum
    n - f >= Q = 2f + 1
    => n >= 3f + 1

Examples:
  f=1: n >= 4 replicas   (tolerates 1 Byzantine)
  f=2: n >= 7 replicas   (tolerates 2 Byzantine)
  f=3: n >= 10 replicas  (tolerates 3 Byzantine)
```

#### 3.4.2 Message Formats

All messages defined in `schemas/bft.py` using Pydantic models.

**Phase 1: REQUEST**
```
<REQUEST, operation, timestamp, client_id>_signature_client

Fields:
  - operation: Dict[str, Any]  # Precedent operation to execute
  - timestamp: int             # Milliseconds, for exactly-once semantics
  - client_id: str             # Unique client identifier
```

**Phase 2: PRE-PREPARE (Primary only)**
```
<<PRE-PREPARE, view, sequence, digest>_signature_primary, request>

Fields:
  - view: int          # Current view number (v)
  - sequence: int      # Assigned sequence number (n)
  - digest: str        # SHA-256 hash of request
  - request: Request   # Original client request

Acceptance Criteria (replica accepts iff):
  1. Signature verifies for current primary
  2. view == replica's current view
  3. h < sequence <= H (within water marks)
  4. No prior PRE-PREPARE for (view, sequence) with different digest
```

**Phase 3: PREPARE**
```
<PREPARE, view, sequence, digest, replica_id>_signature_replica

Multicast by replica i upon accepting PRE-PREPARE.

Prepared(m, v, n, i) is TRUE when replica i has:
  1. The request m
  2. A valid PRE-PREPARE for m in view v with sequence n
  3. 2f matching PREPARE messages from different replicas
```

**Phase 4: COMMIT**
```
<COMMIT, view, sequence, digest, replica_id>_signature_replica

Multicast by replica i when Prepared(m, v, n, i) = TRUE.

Committed-local(m, v, n, i) is TRUE when:
  1. Prepared(m, v, n, i) = TRUE
  2. Replica has 2f + 1 matching COMMIT from different replicas
```

**Phase 5: REPLY**
```
<REPLY, view, timestamp, client_id, replica_id, result>_signature_replica

Sent to client after Committed-local becomes TRUE and operation executes.
Client accepts result when f + 1 matching replies received.
```

#### 3.4.3 View Change Protocol

Triggered when primary is suspected faulty (timeout or Byzantine behavior).

**VIEW-CHANGE Message:**
```
<VIEW-CHANGE, new_view, last_checkpoint, checkpoint_proofs, prepared_certs, replica_id>_signature

Fields:
  - new_view: int                       # v + 1
  - last_checkpoint: int                # Sequence of last stable checkpoint
  - checkpoint_proofs: List[Checkpoint] # 2f + 1 matching checkpoints
  - prepared_certs: List[PreparedCert]  # Proofs of what was prepared
```

**NEW-VIEW Message (new primary):**
```
<NEW-VIEW, new_view, view_change_proofs, pre_prepares_to_redo>_signature

Sent by new primary (replica (v+1) mod n) after collecting 2f + 1 VIEW-CHANGE.
Contains PRE-PREPAREs for any requests that were prepared but not committed.
```

**View Change Timeout Escalation:**
```
Initial timeout: T = 10 seconds
After k failed view changes: T_k = T * 2^k
Maximum timeout: T_max = 120 seconds

This exponential backoff prevents view-change storms while ensuring liveness.
```

#### 3.4.4 Timeout Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `request_timeout_ms` | 5000 | [100, 60000] | Client retry timeout |
| `preprepare_timeout_ms` | 2000 | [100, 30000] | Wait for PRE-PREPARE |
| `prepare_timeout_ms` | 3000 | [100, 30000] | Wait for 2f+1 PREPARE |
| `commit_timeout_ms` | 3000 | [100, 30000] | Wait for 2f+1 COMMIT |
| `view_change_timeout_ms` | 10000 | [1000, 120000] | Trigger view change |
| `checkpoint_interval` | 100 | [10, 10000] | Requests between checkpoints |

#### 3.4.5 Garbage Collection and Checkpoints

**Stable Checkpoint:**
- Replicas checkpoint state every `checkpoint_interval` requests
- Checkpoint is stable when 2f + 1 replicas have matching checkpoints
- Stable checkpoint allows garbage collection of older messages

**Water Marks:**
```
h = sequence number of last stable checkpoint (low water mark)
H = h + K where K = high_water_mark_window (e.g., 400)

Replicas reject PRE-PREPARE if sequence <= h or sequence > H.
This bounds memory usage and enables garbage collection.
```

#### 3.4.6 Federation Engine Implementation

```python
class FederationEngine:
    """
    Federated ratchet simulation with PBFT consensus.

    Implements full PBFT protocol with:
    - n >= 3f + 1 verification invariant
    - View change protocol for leader election
    - Checkpointing and garbage collection
    - Behavioral correlation detection (anti-Sybil)
    """

    def __init__(
        self,
        consensus_protocol: Literal["pbft"] = "pbft",  # PBFT is the only implemented protocol
        mi_threshold: float = 0.85,
        bft_config: Optional[BFTConfig] = None,
    ):
        self.consensus = consensus_protocol
        self.mi_threshold = mi_threshold
        self.config = bft_config or BFTConfig()

    def create_federation(
        self,
        num_honest: NodeCount,                    # int >= 1
        num_malicious: int = 0,
        malicious_strategy: MaliciousStrategy = MaliciousStrategy.RANDOM,
    ) -> Federation:
        """
        Create federation with specified agent composition.

        ENFORCES: n >= 3f + 1 where f = num_malicious
        Raises ValueError if invariant would be violated.
        """
        n = num_honest + num_malicious
        f = num_malicious
        if n < 3 * f + 1:
            raise ValueError(
                f"BFT invariant violated: n={n} < 3f+1={3*f+1}. "
                f"Need at least {3*f+1 - n} more honest replicas."
            )
        ...

    def verify_bft_invariant(
        self,
        federation: Federation,
    ) -> BFTInvariantResult:
        """
        Verify n >= 3f + 1 holds for current federation.

        Returns:
            is_valid: bool
            n: int (total replicas)
            f: int (max tolerable Byzantine)
            actual_malicious: int (known/suspected)
            margin: int (how many more Byzantine could be tolerated)
        """
        ...

    def process_request(
        self,
        request: Request,
        federation: Federation,
    ) -> ConsensusResult:
        """
        Process client request through PBFT phases.

        Returns result after f+1 matching replies collected.
        """
        ...

    def initiate_view_change(
        self,
        federation: Federation,
        suspected_primary: str,
        reason: ViewChangeReason,
    ) -> ViewChangeResult:
        """
        Initiate view change when primary suspected faulty.

        Reasons: TIMEOUT, BYZANTINE_BEHAVIOR, EXPLICIT_REQUEST
        """
        ...

    def orthogonality_gate(
        self,
        candidate: Agent,
        federation: Federation,
    ) -> GateResult:
        """
        Evaluate candidate for partnership.
        Returns MI scores and accept/reject decision.
        """
        ...

    def behavioral_correlation_check(
        self,
        agents: List[Agent],
        voting_history: List[Vote],
    ) -> CorrelationResult:
        """
        Detect coordinated behavior despite diverse constraints.

        Flags agents with voting correlation > 0.9 despite MI < 0.85.
        This detects "diverse Sybil" attacks (RT-05).
        """
        ...

    def slow_capture_simulation(
        self,
        federation: Federation,
        capture_rate_per_period: CaptureRate,    # 0 <= rate < 1
        num_periods: SampleSize,                  # int >= 1
    ) -> CaptureSimulationResult:
        """
        Simulate gradual federation takeover.

        Tracks:
        - Period when f_actual > f_tolerable (BFT breached)
        - Honest agent departure rate
        - Precedent poisoning accumulation
        """
        ...

    def precedent_poisoning_detection(
        self,
        precedent_stream: List[Precedent],
        window_size: SampleSize,                  # int >= 1
    ) -> PoisoningResult:
        """
        Detect subtle precedent drift over time.
        Uses statistical change-point detection.
        """
        ...
```

#### 3.4.7 Security Invariants (Federation)

```python
# These invariants are CONTINUOUSLY MONITORED

FEDERATION_INVARIANTS = [
    # Core BFT invariant
    "n >= 3 * f + 1",

    # Quorum validity
    "quorum_size == 2 * f + 1",

    # View consistency
    "all_honest_replicas_same_view OR view_change_in_progress",

    # Checkpoint consistency
    "stable_checkpoint_agreed_by >= 2 * f + 1",

    # Anti-Sybil (behavioral)
    "forall agent_pair: behavioral_correlation(a, b) > 0.9 => flagged",

    # Capture threshold warning
    "suspected_byzantine_count < f => WARN if > 0.5 * f",
]
```

#### 3.4.8 Attack Resistance Matrix

| Attack | PBFT Protection | Additional Mitigation |
|--------|-----------------|----------------------|
| RT-02: Slow Capture | BFT threshold monitoring | Capture rate alerts when f_actual > 0.5 * f_max |
| RT-05: Diverse Sybils | None (bypasses MI gate) | Behavioral correlation detection |
| Primary Byzantine | View change protocol | Timeout + behavior scoring |
| Message Replay | Sequence numbers + digests | Signed timestamps |
| Network Partition | Liveness degradation | Timeout escalation, view change |

#### 3.4.9 Technology Decisions

The following engineering decisions resolve open questions for BFT implementation:

##### 3.4.9.1 Cryptographic Library

**Decision:** Use `cryptography` library (pyca/cryptography)

**Rationale:**
- Most mature and widely-used Python cryptography library
- Provides Ed25519 for digital signatures (PBFT message authentication)
- Provides SHA-256 for message digests
- FIPS-compliant primitives available
- Active maintenance and security audit history
- Used by major projects (PyCA, Paramiko, etc.)

**Usage:**
```python
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend

# Generate signing keys for each replica
private_key = Ed25519PrivateKey.generate()
public_key = private_key.public_key()

# Sign messages
signature = private_key.sign(message_bytes)

# Verify signatures
public_key.verify(signature, message_bytes)  # Raises InvalidSignature on failure
```

##### 3.4.9.2 Message Serialization

**Decision:** JSON with Pydantic models

**Rationale:**
- Consistent with rest of RATCHET codebase
- Pydantic provides automatic validation and serialization
- Human-readable for debugging and logging
- Schema evolution via optional fields with defaults
- Type safety at runtime through Pydantic validators

**Alternative Considered:** Protocol Buffers (rejected for now)
- Would provide better performance for high-throughput scenarios
- Can be added later if performance becomes a bottleneck
- Current research testbed prioritizes debuggability over performance

**Serialization Format:**
```python
# Serialize
message_json = message.model_dump_json()

# Deserialize with validation
message = Request.model_validate_json(message_json)

# Deterministic serialization for digests (exclude volatile fields)
canonical = message.model_dump_json(exclude={'signature', 'digest'}, sort_keys=True)
```

##### 3.4.9.3 Network Abstraction

**Decision:** Abstract interface with pluggable implementations

**Rationale:**
- Decouples PBFT logic from transport mechanism
- Enables testing with in-memory transport
- Supports future deployment options (TCP, gRPC, QUIC)
- Allows simulation of network conditions (latency, partitions)

**Interface:** See `schemas/bft.py` for `NetworkTransport` abstract base class.

**Implementations (to be provided):**
- `InMemoryTransport`: For testing and simulation
- `TCPTransport`: For local network testing
- `AsyncTransport`: For production deployment (future)

##### 3.4.9.4 Persistent Storage

**Decision:** Abstract storage interface (implementation-agnostic)

**Rationale:**
- PBFT requires durable storage for message logs, checkpoints, and state
- Abstract interface allows testing with in-memory storage
- Production can use SQLite, PostgreSQL, or custom backends
- Supports crash recovery and view change protocol

**Interface:** See `schemas/bft.py` for `PersistentStorage` abstract base class.

**Storage Requirements:**
- Message logs (keyed by sequence number and view)
- Stable checkpoints (keyed by sequence number)
- Current view state
- Replica key material

**Implementations (to be provided):**
- `InMemoryStorage`: For testing (no persistence)
- `SQLiteStorage`: For single-node development
- `PostgreSQLStorage`: For production (future)

### 3.5 Red Team Engine

**Purpose:** Orchestrate adversarial scenarios and security validation

**Top 5 Attack Scenarios (from Red Team Analysis):**

| ID | Attack | Severity | Type |
|----|--------|----------|------|
| RT-01 | Emergent Multi-Agent Deception | CRITICAL | FUNDAMENTAL |
| RT-02 | Slow Federation Capture | CRITICAL | PATCHABLE |
| RT-03 | Constraint-Aligned Deception | CRITICAL | PATCHABLE |
| RT-04 | Distribution Mimicry | CRITICAL | PATCHABLE |
| RT-05 | Diverse Sybils | HIGH | PATCHABLE |

```python
class RedTeamEngine:
    """
    Adversarial scenario orchestration.
    """

    ATTACK_REGISTRY = {
        "RT-01": EmergentDeceptionAttack,
        "RT-02": SlowCaptureAttack,
        "RT-03": NullSpaceAttack,
        "RT-04": DistributionMimicryAttack,
        "RT-05": DiverseSybilAttack,
    }

    def run_attack_scenario(
        self,
        attack_id: str,
        target_engine: Engine,
        params: AttackParams,
    ) -> AttackResult:
        """
        Execute attack scenario against target engine.
        Returns success rate, impact metrics, and traces.
        """
        ...

    def security_audit(
        self,
        engines: Dict[str, Engine],
        attack_budget: int = 1000,
    ) -> SecurityAuditReport:
        """
        Run full adversarial audit across all engines.
        Returns vulnerability scores and recommendations.
        """
        ...

    def stress_test(
        self,
        engine: Engine,
        stress_params: StressParams,
    ) -> StressTestResult:
        """
        Stress test with parameters:
        - Geometric: D=[10,10000], k=[10,500], ρ=[0,0.9]
        - Complexity: m=[10,100], solver=[all]
        - Detection: D_mahal=[0.1,2.0], p=[0.001,0.1]
        - Federation: n_sybil=[0,100], capture_rate=[0,0.5]
        """
        ...
```

---

## 4. FORMAL VERIFICATION INTERFACE

### 4.1 Proof Obligations

**From Formal Methods Analysis:**

| Claim | Obligations | Difficulty | Status |
|-------|-------------|------------|--------|
| Topological Collapse | 8 | ⭐⭐⭐⭐ | Tractable |
| Computational Asymmetry | 5 | ⭐⭐ | Easy |
| Detection Power | 7 | ⭐⭐⭐⭐⭐ | Hard |

**Critical Proof Obligations:**

1. **Cutting Probability** (CORE)
   ```
   P_{H ~ H}(H ∩ B_r(c) ≠ ∅) = 2r + O(r²)
   ```

2. **NP-Completeness** (CORE)
   ```
   3-SAT ≤_p CONSISTENT-LIE
   ```

3. **LRT Distribution** (CORE)
   ```
   ℓ(t) | t ~ P_H ~ N(-D²/2, D²)
   ```

### 4.1.1 New Proof Obligations (from Formal Review)

The FSD introduces components not in the original Formalization Roadmap that imply new proof obligations. These obligations arise from FSD additions and must be addressed for complete formal verification coverage.

| ID | Obligation | Source | Difficulty | Status | Dependencies |
|----|------------|--------|------------|--------|--------------|
| NEW-01 | Effective Rank Correctness | Section 3.1 (k_eff formula) | Medium | PENDING | wt-5 (types) |
| NEW-02 | BFT Protocol Safety/Liveness | Section 3.4 (federation) | Hard | PENDING | wt-3 (BFT) |
| NEW-03 | Behavioral Correlation Soundness | Section 3.4 (Sybil detection) | Medium | PENDING | wt-3 (BFT) |
| NEW-04 | Compositional Detection Correctness | Section 3.3 (RT-01 mitigation) | **IMPOSSIBLE** | BLOCKED | - |
| NEW-05 | Slow Capture Threshold | Section 3.4 (slow capture) | Medium | PENDING | wt-3 (BFT) |

---

#### NEW-01: Effective Rank Correctness for Correlated Constraints

**Theorem Statement:**
```
For k hyperplanes with pairwise correlation coefficient rho in [-1, 1],
the effective number of independent constraints is:

    k_eff = k / (1 + rho * (k - 1))

Proof Obligation: k_eff correctly captures the expected volume reduction
for all rho in [-1, 1], with error bounded by O(rho^2 * k^2).
```

**Formal Specification:**
```python
def effective_rank_correctness(
    k: PositiveInt,           # Number of hyperplanes
    rho: Correlation,         # Pairwise correlation in [-1, 1]
) -> bool:
    """
    PRECONDITIONS:
    - k >= 1
    - -1 <= rho <= 1
    - For rho = -1/(k-1), k_eff diverges (degenerate case)

    POSTCONDITION:
    - volume_reduction(k, rho) == volume_reduction(k_eff, 0) + O(error)
    - where error = C * rho^2 * k^2 for some constant C
    """
```

**Lean 4 Theorem Sketch:**
```lean
-- Depends on: wt-5 (Correlation type), Mathlib.LinearAlgebra
theorem effective_rank_correctness
    (k : Nat) (hk : k >= 1)
    (rho : Real) (hrho : -1 <= rho && rho <= 1)
    (h_nondegen : rho != -1 / (k - 1)) :
    let k_eff := k / (1 + rho * (k - 1))
    exists C : Real, C > 0 /\
      |volume_reduction k rho - volume_reduction k_eff 0| <= C * rho^2 * k^2 := by
  sorry -- Requires correlation matrix eigenvalue analysis
```

---

#### NEW-02: BFT Protocol Safety and Liveness

**Theorem Statement:**
```
For a federation of n nodes with f < n/3 Byzantine nodes, the chosen
consensus protocol (PBFT/Tendermint) satisfies:

SAFETY: No two honest nodes commit conflicting precedents.
LIVENESS: Under partial synchrony, every valid proposal eventually commits.

Proof Obligation: The implemented protocol achieves both properties for f < n/3.
```

**Formal Specification:**
```python
def bft_protocol_correctness(
    protocol: Literal["pbft", "tendermint"],
    n: PositiveInt,           # Total nodes
    f: NonNegativeInt,        # Byzantine nodes
) -> Tuple[bool, bool]:       # (safety, liveness)
    """
    PRECONDITIONS:
    - n >= 3 * f + 1
    - protocol has correct implementation (verified separately)

    POSTCONDITIONS:
    - SAFETY: forall honest h1, h2, round r:
        committed(h1, r, v1) AND committed(h2, r, v2) => v1 == v2
    - LIVENESS: forall valid proposal p, exists round r:
        eventually(committed(honest, r, p))
    """
```

**Lean 4 Theorem Sketch:**
```lean
-- Depends on: wt-3 (BFT protocol definitions)
-- Note: Full verification typically done in TLA+ or Ivy

structure BFTConfig where
  n : Nat
  f : Nat
  h_threshold : n >= 3 * f + 1

theorem bft_safety (cfg : BFTConfig)
    (h1 h2 : HonestNode) (r : Round) (v1 v2 : Value)
    (hc1 : committed h1 r v1) (hc2 : committed h2 r v2) :
    v1 = v2 := by
  sorry -- Reduction to quorum intersection lemma

theorem bft_liveness (cfg : BFTConfig)
    (p : Proposal) (hp : valid p)
    (h_sync : partial_synchrony) :
    exists r : Round, eventually (exists h : HonestNode, committed h r p.value) := by
  sorry -- Requires partial synchrony model
```

**Verification Approach:** BFT proofs are typically conducted in TLA+ or Ivy rather than Lean. The Lean sketches above serve as specification anchors; actual verification should use model checking.

---

#### NEW-03: Behavioral Correlation Detection Soundness (Diverse Sybil Detection)

**Theorem Statement:**
```
If agents pass the MI orthogonality gate independently but coordinate their
votes with correlation > threshold, the behavioral_correlation_check
identifies them with probability >= 1 - delta.

Proof Obligation: Detection is sound (low false positives) and complete
(high true positives) for coordinated Sybil attacks.
```

**Formal Specification:**
```python
def behavioral_correlation_soundness(
    agents: List[Agent],
    voting_history: List[Vote],
    correlation_threshold: float = 0.8,
    delta: float = 0.05,
) -> Tuple[float, float]:     # (sensitivity, specificity)
    """
    PRECONDITIONS:
    - len(voting_history) >= min_history_length (sufficient samples)
    - correlation_threshold in (0, 1)

    POSTCONDITIONS:
    - SENSITIVITY: P(detect | coordinated) >= 1 - delta
    - SPECIFICITY: P(not_detect | independent) >= 1 - delta
    - Where "coordinated" means pairwise_correlation > correlation_threshold
    """
```

**Lean 4 Theorem Sketch:**
```lean
-- Depends on: wt-3 (voting types), wt-5 (Agent, Vote types)
-- Requires: Mathlib.Probability, statistical testing theory

def pairwise_correlation (v1 v2 : VotingHistory) : Real := sorry

def coordinated (agents : List Agent) (histories : Agent -> VotingHistory)
    (threshold : Real) : Prop :=
  forall a1 a2, a1 != a2 ->
    pairwise_correlation (histories a1) (histories a2) > threshold

theorem behavioral_correlation_sensitivity
    (agents : List Agent)
    (histories : Agent -> VotingHistory)
    (threshold : Real) (delta : Real)
    (h_coordinated : coordinated agents histories threshold)
    (h_sufficient_history : forall a, (histories a).length >= min_samples delta) :
    detection_probability agents histories >= 1 - delta := by
  sorry -- Requires concentration inequality (Hoeffding/McDiarmid)

theorem behavioral_correlation_specificity
    (agents : List Agent)
    (histories : Agent -> VotingHistory)
    (delta : Real)
    (h_independent : independent_voting agents histories)
    (h_sufficient_history : forall a, (histories a).length >= min_samples delta) :
    false_positive_probability agents histories <= delta := by
  sorry -- Central limit theorem application
```

---

#### NEW-04: Compositional Detection Correctness

**STATUS: POTENTIALLY IMPOSSIBLE (Fundamental Limitation)**

**Theorem Statement (Attempted):**
```
If an inference graph has honest components (each with deception probability < epsilon)
but exhibits emergent deception (system-level deception probability > delta),
then compositional_detection identifies the deceptive pattern.
```

**Why This May Be IMPOSSIBLE:**

1. **Reduction from One-Way Functions:** If compositional detection were efficient and complete, it could break cryptographic commitments by detecting "deceptive" commitments that don't match their openings. This contradicts computational hardness assumptions.

2. **Information-Theoretic Barrier:** Emergent deception that preserves marginal distributions of individual traces is fundamentally undetectable without global state access, which may not be available.

3. **Rice's Theorem Analog:** Determining whether a composition of honest components produces deceptive behavior is analogous to the halting problem for semantic properties.

**Formal Impossibility Sketch:**
```
Theorem (Informal): For any polynomial-time detector D, there exists a
system S of honest components such that S exhibits emergent deception
and D fails to detect it.

Proof Sketch:
1. Assume D detects all emergent deception in polynomial time.
2. Construct a cryptographic commitment scheme where:
   - Individual commit/reveal operations appear "honest"
   - Deceptive commitments (commit(m1), reveal(m2)) are detectable by D
3. D can then be used to distinguish commitments, breaking hiding property.
4. Contradiction with computational binding of commitments.
```

**Lean 4 Sketch (Impossibility Statement):**
```lean
-- This is a negative result: we prove detection CANNOT be complete

-- Assumption: Cryptographic commitments exist
axiom commitment_scheme_exists :
  exists (Commit : Message -> Randomness -> Commitment),
    hiding Commit /\ binding Commit

-- What we would need for complete detection
def complete_compositional_detector :=
  forall (S : System) (epsilon delta : Real),
    (forall c : Component, c in S -> deception_prob c < epsilon) ->
    (system_deception_prob S > delta) ->
    detected S = true

-- The impossibility theorem
theorem compositional_detection_incomplete :
    not (exists D : Detector, complete_compositional_detector D) := by
  intro h_exists
  obtain ⟨D, h_complete⟩ := h_exists
  -- Use D to break commitment scheme hiding
  have h_breaks_hiding : not (hiding commitment_scheme) := sorry
  -- Contradiction
  exact absurd commitment_scheme_exists.hiding h_breaks_hiding
```

**Partial Mitigation (What IS Possible):**
- Detect emergent deception that causes observable statistical anomalies
- Detect inference chains with systematic bias
- Detect information flow patterns that enable deception
- See Section 3.3 `compositional_detection` for implemented partial detection

---

#### NEW-05: Slow Capture Threshold

**Theorem Statement:**
```
For a federation with capture_rate r per period (fraction of honest nodes
converted to malicious per period), the federation remains secure
(f < n/3) for at least O(1/r) periods before BFT threshold is breached.

Proof Obligation: Quantify the number of periods until compromise as a
function of r, initial malicious fraction f_0, and n.
```

**Formal Specification:**
```python
def slow_capture_threshold(
    n: PositiveInt,           # Total nodes
    f_0: float,               # Initial malicious fraction (< 1/3)
    r: float,                 # Capture rate per period
) -> int:                     # Periods until BFT breach
    """
    PRECONDITIONS:
    - n >= 4 (minimum for BFT)
    - 0 <= f_0 < 1/3
    - 0 < r < 1

    POSTCONDITION:
    - Returns T such that:
      f_0 + r * T >= 1/3 (BFT threshold breached)
    - T = floor((1/3 - f_0) / r) = O(1/r) when f_0 is constant

    SECURITY BOUND:
    - For f_0 = 0, r = 0.01: T >= 33 periods before compromise
    - For f_0 = 0.1, r = 0.01: T >= 23 periods before compromise
    """
```

**Lean 4 Theorem Sketch:**
```lean
-- Depends on: wt-3 (BFT definitions), wt-5 (rate types)

def malicious_fraction (f_0 : Real) (r : Real) (t : Nat) : Real :=
  f_0 + r * t

def bft_secure (f : Real) : Prop := f < 1/3

theorem slow_capture_bound
    (n : Nat) (hn : n >= 4)
    (f_0 : Real) (hf0 : 0 <= f_0 /\ f_0 < 1/3)
    (r : Real) (hr : 0 < r /\ r < 1) :
    let T := Nat.floor ((1/3 - f_0) / r)
    forall t : Nat, t < T -> bft_secure (malicious_fraction f_0 r t) := by
  intro T t ht
  unfold bft_secure malicious_fraction
  -- Need: f_0 + r * t < 1/3
  -- Since t < T = floor((1/3 - f_0) / r), we have r * t < 1/3 - f_0
  have h1 : (t : Real) < T := Nat.cast_lt.mpr ht
  have h2 : T <= (1/3 - f_0) / r := Nat.floor_le (by linarith [hr.1, hf0.1, hf0.2])
  have h3 : r * t < 1/3 - f_0 := by
    calc r * t < r * T := by nlinarith [hr.1]
         _ <= r * ((1/3 - f_0) / r) := by nlinarith [hr.1, h2]
         _ = 1/3 - f_0 := by field_simp; ring
  linarith

-- Asymptotic bound
theorem slow_capture_asymptotic
    (f_0 : Real) (hf0 : 0 <= f_0 /\ f_0 < 1/3)
    (r : Real) (hr : 0 < r) :
    let T := (1/3 - f_0) / r
    T = O(1/r) := by
  -- T = (1/3 - f_0) / r = (1/3 - f_0) * (1/r)
  -- For constant f_0, this is O(1/r)
  sorry -- Big-O notation formalization
```

**Mitigation Implications:**
- Federation must implement capture rate monitoring (see Section 3.4)
- Alarm threshold should trigger at r > r_max where T(r_max) >= min_safe_periods
- Example: For min_safe_periods = 100, r_max = 0.0033 (0.33% per period)

### 4.2 Lean 4 Integration

```python
class FormalVerificationEngine:
    """
    Bridge between simulation and formal proofs.
    """

    def check_proof_obligation(
        self,
        obligation_id: str,
        lean_file: Path,
    ) -> ProofCheckResult:
        """
        Verify Lean 4 proof compiles and type-checks.
        """
        ...

    def extract_constants(
        self,
        simulation_results: SimulationResults,
    ) -> ConstantExtraction:
        """
        Extract empirical constants for hybrid verification.
        E.g., λ ≈ 2.3/r from Monte Carlo fits.
        """
        ...

    def validate_bounds(
        self,
        proven_bound: Bound,
        empirical_data: np.ndarray,
    ) -> BoundValidation:
        """
        Check that proven bounds hold on empirical data.
        Reports violations and confidence levels.
        """
        ...

    def generate_proof_skeleton(
        self,
        theorem: TheoremSpec,
    ) -> str:
        """
        Generate Lean 4 proof skeleton from theorem specification.
        Returns .lean file content with sorry placeholders.
        """
        ...
```

---

## 5. EXPERIMENTAL FRAMEWORK

### 5.1 Experiment Protocols (Social Scientist)

**EXP-01: Multi-Agent Deception Detection**
- N = 100 agents, 10K interactions
- Metrics: Detection latency, false positive rate, deception propagation
- Hypothesis: Detection power ≥ 90% within 50 interactions

**EXP-02: Human-AI Trust Calibration**
- N = 200 human participants
- Protocol: Present agent traces, ask trust ratings
- Metrics: Calibration curve, over/under-trust bias
- IRB Required: Yes

**EXP-03: Federated Precedent Emergence**
- 5-node testbed, 3 months simulated time
- Metrics: Precedent convergence, Byzantine tolerance
- Hypothesis: Stable consensus with ≤ 1/3 malicious

**EXP-04: Adversarial Constraint Gaming**
- Red team competition format
- Metrics: Attack success rate, cost to defender
- Prize structure for novel attacks

**EXP-05: Longitudinal Coherence Tracking**
- 12-month deployment on controlled system
- Metrics: Drift detection, false alarm rate, maintenance cost

### 5.2 Statistical Analysis

```python
class ExperimentAnalysis:
    """
    Statistical analysis for experiments.
    """

    def bootstrap_ci(
        self,
        data: np.ndarray,
        statistic: Callable,
        n_bootstrap: int = 10_000,
        alpha: float = 0.05,
    ) -> ConfidenceInterval:
        """
        Bootstrap confidence interval for any statistic.
        """
        ...

    def power_curve(
        self,
        effect_sizes: List[float],
        sample_sizes: List[int],
        alpha: float = 0.05,
    ) -> PowerCurve:
        """
        Compute power curves for experiment planning.
        """
        ...

    def convergence_diagnostic(
        self,
        trace: np.ndarray,
        method: Literal["gelman_rubin", "geweke", "heidel"] = "gelman_rubin",
    ) -> ConvergenceResult:
        """
        Monte Carlo convergence diagnostic.
        """
        ...
```

---

## 6. API SPECIFICATION

### 6.1 REST Endpoints

```
POST   /experiments              Create experiment
GET    /experiments/{id}         Get experiment status
POST   /experiments/{id}/run     Execute experiment
GET    /experiments/{id}/results Get results

POST   /simulations              Create simulation
POST   /simulations/batch        Batch simulation
GET    /simulations/{id}/status  Poll status
GET    /simulations/{id}/results Get results

POST   /validation/geometric     Validate geometric claims
POST   /validation/complexity    Validate complexity claims
POST   /validation/detection     Validate detection claims
POST   /validation/full          Full validation suite

POST   /federation/create        Create test federation
POST   /federation/{id}/attack   Run attack scenario
GET    /federation/{id}/health   Federation health check

POST   /proofs/check             Check Lean proof
POST   /proofs/validate-bounds   Validate proven bounds
GET    /proofs/obligations       List proof obligations

POST   /redteam/scenario         Run attack scenario
POST   /redteam/audit            Full security audit
GET    /redteam/report           Get latest audit report
```

### 6.2 Schema Examples

**Note:** Full type definitions are in `schemas/simulation.py`. The following shows the type-safe
discriminated union approach that replaces `Dict[str, Any]` (addressing T-SCH-01 from Formal Review).

```python
from typing import Annotated, Literal, Optional, Union
from pydantic import BaseModel, Field

# =============================================================================
# Refinement Types (Base Types)
# =============================================================================

Dimension = Annotated[int, Field(gt=0, description="Positive dimension D")]
Radius = Annotated[float, Field(gt=0, lt=0.5, description="Deceptive radius (0 < r < 0.5)")]
Correlation = Annotated[float, Field(ge=-1, le=1, description="Constraint correlation")]
Probability = Annotated[float, Field(gt=0, lt=1, description="Probability (0 < p < 1)")]
PositiveInt = Annotated[int, Field(gt=0, description="Positive integer")]
NonNegativeFloat = Annotated[float, Field(ge=0, description="Non-negative float")]

# =============================================================================
# Engine-Specific Parameter Models (Discriminated Union Members)
# =============================================================================

class GeometricParams(BaseModel):
    """Parameters for Geometric Engine. Addresses T-GEO-01 through T-GEO-04."""
    engine: Literal["geometric"] = "geometric"
    dimension: Dimension
    num_constraints: PositiveInt
    deceptive_radius: Radius
    constraint_correlation: Correlation = 0.0
    sampling_mode: Literal["orthonormal", "correlated", "adversarial"] = "orthonormal"
    num_samples: PositiveInt = 100_000
    adversary: Optional[AdversarialStrategy] = None

class ComplexityParams(BaseModel):
    """
    Parameters for Complexity Engine. Addresses T-CPX-01, T-CPX-02.
    WARNING: literals_per_statement < 3 yields P-time problem (2-SAT), voiding NP-hardness claims.
    """
    engine: Literal["complexity"] = "complexity"
    world_size: PositiveInt
    num_statements: PositiveInt
    literals_per_statement: Annotated[int, Field(ge=2)] = 3  # Must be >= 3 for NP-hardness
    observable_fraction: Annotated[float, Field(gt=0, le=1)] = 1.0
    deception_strategy: Literal["full", "sparse", "lazy"] = "full"
    solver: Literal["z3", "minisat", "cadical", "bruteforce"] = "z3"

class DetectionParams(BaseModel):
    """
    Parameters for Detection Engine. Addresses T-DET-01 through T-DET-03.
    PRECONDITIONS: D >= 0.5, p >= 0.001, n >= 100 for asymptotic validity.
    """
    engine: Literal["detection"] = "detection"
    method: Literal["lrt", "mahalanobis", "isolation_forest", "ensemble"] = "lrt"
    mahalanobis_distance: NonNegativeFloat = 1.0
    deception_rate: Probability = 0.01
    alpha: Probability = 0.05
    beta: Probability = 0.05
    training_sample_size: Optional[PositiveInt] = None

class FederationParams(BaseModel):
    """
    Parameters for Federation Engine. Enforces BFT invariant: n >= 3f + 1.
    SECURITY INVARIANT: num_malicious / (num_honest + num_malicious) < 0.33
    """
    engine: Literal["federation"] = "federation"
    num_honest: PositiveInt
    num_malicious: Annotated[int, Field(ge=0)] = 0
    consensus_protocol: Literal["pbft", "raft", "tendermint"] = "pbft"
    malicious_strategy: Literal["random", "coordinated", "slow_capture"] = "random"
    mi_threshold: Annotated[float, Field(ge=0, le=1)] = 0.85

    @model_validator(mode='after')
    def validate_bft_threshold(self) -> 'FederationParams':
        """Enforce BFT safety: malicious_fraction < 1/3."""
        total = self.num_honest + self.num_malicious
        if self.num_malicious / total >= 1/3:
            raise ValueError("BFT safety violated: malicious_fraction >= 1/3")
        return self

# =============================================================================
# Discriminated Union (Replaces Dict[str, Any])
# =============================================================================

SimulationParams = Annotated[
    Union[GeometricParams, ComplexityParams, DetectionParams, FederationParams],
    Field(discriminator="engine")
]

# =============================================================================
# Request/Response Models
# =============================================================================

class SimulationRequest(BaseModel):
    """
    Type-safe simulation request with discriminated union parameters.
    The 'engine' field is now part of parameters, enabling automatic discrimination.
    """
    parameters: SimulationParams  # Discriminated union - replaces Dict[str, Any]
    adversarial: bool = False
    adversary_config: Optional[AdversaryConfig] = None
    num_runs: PositiveInt = 1
    seed: Optional[int] = None

    @property
    def engine(self) -> str:
        """Engine type derived from parameters."""
        return self.parameters.engine

class SimulationResult(BaseModel):
    id: str
    engine: Literal["geometric", "complexity", "detection", "federation"]
    status: Literal["pending", "running", "completed", "failed"]
    metrics: dict[str, float]
    confidence_intervals: dict[str, ConfidenceInterval]
    adversarial_results: Optional[AdversarialMetrics] = None
    provenance: ProvenanceRecord
    created_at: datetime
    completed_at: Optional[datetime] = None

class AttackScenario(BaseModel):
    attack_id: str = Field(pattern=r"^RT-0[1-5]$")  # RT-01 through RT-05
    target_engine: Literal["geometric", "complexity", "detection", "federation"]
    params: SimulationParams  # Type-safe parameters
    success_threshold: Probability = 0.5

class ProofObligation(BaseModel):
    """Extended with ETH conditionality support (REC-H4)."""
    id: str
    claim: str
    theorem_statement: str
    lean_file: Optional[str] = None
    status: Literal["pending", "partial", "axiomatized", "proven", "disproven", "blocked"]
    dependencies: List[str] = []
    conditional_on: List[str] = []  # E.g., ["ETH", "SETH"] for complexity claims
```

**Type Safety Benefits:**
- Compile-time validation via Pydantic discriminated union
- Runtime parameter validation with meaningful error messages
- Refinement types prevent invalid parameter combinations (e.g., radius > 0.5, k < 3)
- BFT safety invariant enforced at schema level

---

## 7. SECURITY REQUIREMENTS

### 7.1 Attack Mitigations (Red Team Required)

| Attack | Mitigation | Implementation |
|--------|------------|----------------|
| RT-01 Emergent Deception | Compositional reasoning detection | `adversarial/defenses/compositional_reasoning.py` |
| RT-02 Slow Capture | BFT with capture rate monitoring | `core/engines/federation.py` |
| RT-03 Null Space | Adversarial constraint generation | `core/engines/geometric.py` |
| RT-04 Mimicry | Ensemble detection, moving thresholds | `core/engines/detection.py` |
| RT-05 Diverse Sybils | Behavioral correlation checks | `adversarial/defenses/behavioral_correlation.py` |
| **RT-06 Non-Convex Evasion** | **Convexity enforcement + detection** | `core/engines/geometric.py` |

**RT-06: Non-Convex Deceptive Region Attack (RED TEAM CONSIDERATION)**

> **Severity:** HIGH
> **Type:** FUNDAMENTAL (exploits mathematical assumption)
> **Status:** UNSUPPORTED REGION - no mitigation exists

**Attack Description:**
An adversary constructs a non-convex deceptive region (e.g., torus, fractal, disconnected point cloud)
that exploits the convexity assumption in Theorem 1.1. Because the exponential volume shrinkage proof
relies on convex geometry for cutting probability calculations, non-convex regions can:

1. **Survive hyperplane cuts differently**: Disconnected components each survive independently
2. **Avoid cuts entirely**: Thin "tendrils" between convex hulls may escape constraint intersection
3. **Adaptive shaping**: Adversary observes constraint directions and shapes region to avoid them

**Current Status:** This attack is **not mitigated** - non-convex regions are explicitly UNSUPPORTED.
The framework enforces `geometric.deceptive_region.is_convex == True` as a precondition.

**Research Direction:** Future work may explore:
- Convex hull decomposition with per-component tracking
- Topological invariants for arbitrary region shapes
- Adversarial geometry bounds under adaptive attacks

**Reference:** FORMALIZATION_ROADMAP.md Section 4.1.2 (Adversarial Scenario 2)

### 7.2 Security Invariants

```python
# These invariants MUST hold for any deployment consideration

SECURITY_INVARIANTS = [
    # Geometric convexity (CRITICAL - Theorem 1.1 depends on this)
    # Non-convex deceptive regions invalidate volume shrinkage proof
    # Reference: FORMALIZATION_ROADMAP.md Section 1.1, 4.1.2
    "geometric.deceptive_region.is_convex == True",

    # Byzantine tolerance
    "federation.malicious_fraction < 0.33",

    # Detection power (CONDITIONAL ON NON-ADAPTIVE ADVERSARY)
    "detection.power(n=1000, D=1.0, p=0.01) >= 0.90",
    # Note: This claim is INVALID if adversary.strategy_fixed_before(detection.threshold)
    #       is violated. See Section 11 for adaptive adversary analysis.

    # Geometric robustness (with correlation)
    "geometric.volume_reduction(k=50, rho=0.5) >= 0.50",

    # Complexity gap (with modern solver)
    "complexity.ratio(m=20, solver='z3') >= 5.0",

    # Anti-Sybil
    "federation.behavioral_correlation_detection(coordinated_sybils) == True",

    # M-02: Hyperplane Distribution Consistency (NEW)
    # INVARIANT: Either use canonical distribution OR apply lambda adjustment
    """
    geometric.hyperplane_distribution.offset_distribution == 'uniform_0_1' OR
    (
        geometric.hyperplane_distribution.offset_distribution == 'uniform_a_b' AND
        geometric.lambda == 2*r / (b - a) AND
        |cutting_probability_error| <= C * r^2
    )
    """,
]

# Finite Sample Validity Invariants (M-03)
# These invariants govern the validity of asymptotic statistical formulas

FINITE_SAMPLE_INVARIANTS = [
    # FS-1: Asymptotic validity threshold
    # For n >= 100, the asymptotic power formula is valid
    "n >= 100 => detection.asymptotic_formula_valid == True",

    # FS-2: Power approximation accuracy
    # Empirical power must match theoretical within tolerance for n >= 100
    "n >= 100 => abs(detection.empirical_power - detection.theoretical_power) <= 0.05",

    # FS-3: Berry-Esseen correction required for small samples
    # For 30 <= n < 100, must apply finite-sample correction
    "30 <= n < 100 => detection.berry_esseen_correction_applied == True",

    # FS-4: Small sample fallback
    # For n < 30, must use exact or bootstrap methods
    "n < 30 => detection.method in ['permutation', 'bootstrap', 'conservative']",
]

def validate_finite_sample_regime(n: int, empirical_power: float,
                                   theoretical_power: float) -> dict:
    """
    Validate finite sample invariants for detection power analysis.

    Args:
        n: Sample size
        empirical_power: Observed detection power from simulation
        theoretical_power: Power predicted by asymptotic formula

    Returns:
        dict with validation results and any required corrections
    """
    result = {
        'n': n,
        'regime': None,
        'valid': True,
        'correction_applied': None,
        'power_deviation': abs(empirical_power - theoretical_power),
    }

    if n >= 100:
        # Asymptotic regime - check FS-2
        result['regime'] = 'asymptotic'
        if result['power_deviation'] > 0.05:
            result['valid'] = False
            result['warning'] = 'Power deviation exceeds 0.05 tolerance'
    elif n >= 30:
        # Berry-Esseen correction regime
        result['regime'] = 'berry_esseen'
        result['correction_applied'] = 'berry_esseen'
        # Apply correction: delta_BE = 0.4748 * rho / (sigma^3 * sqrt(n))
        # Conservative estimate assuming rho/sigma^3 ~ 1
        delta_BE = 0.4748 / (n ** 0.5)
        result['correction_bound'] = delta_BE
    else:
        # Small sample regime - require fallback
        result['regime'] = 'small_sample'
        result['correction_applied'] = 'fallback_required'
        result['recommended_method'] = 'bootstrap' if n >= 10 else 'permutation'

    return result
```

**Convexity Invariant Details:**

The convexity invariant `geometric.deceptive_region.is_convex == True` is **foundational** to the
Topological Collapse theorem. Violation analysis:

| Condition | Consequence |
|-----------|-------------|
| Convex region (ball, ellipsoid) | Theorem 1.1 applies: V(k) = V(0) × exp(-λk) |
| Non-convex region (torus, fractal) | **UNSUPPORTED**: Cutting probability formula invalid |
| Disconnected components | Each component survives cuts independently; theorem breaks |
| Adaptive non-convex | Adversary can shape region to avoid hyperplane cuts |

**Known Limitation:** Non-convex deceptive regions are explicitly out of scope for the current
RATCHET framework. Extending to non-convex geometry would require:
1. New measure-theoretic analysis for arbitrary topology
2. Per-component volume tracking for disconnected regions
3. Adversarial geometry optimization bounds

See Section 10 (Open Questions) item 6 for research directions.

---

## 8. DEPLOYMENT

### 8.1 Docker Compose

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://ratchet:ratchet@db:5432/ratchet
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis

  worker:
    build: .
    command: celery -A ratchet.worker worker -l info
    environment:
      - DATABASE_URL=postgresql://ratchet:ratchet@db:5432/ratchet
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis

  db:
    image: timescale/timescaledb:latest-pg15
    environment:
      - POSTGRES_USER=ratchet
      - POSTGRES_PASSWORD=ratchet
      - POSTGRES_DB=ratchet
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

  lean:
    build:
      context: ./formal/lean
      dockerfile: Dockerfile.lean
    volumes:
      - ./formal/lean:/workspace

volumes:
  postgres_data:
  redis_data:
```

---

## 9. ACCEPTANCE CRITERIA

### 9.1 Functional Requirements

- [ ] All 6 core engines implemented and tested
- [ ] REST API covers all specified endpoints
- [ ] All 5 red team attacks implemented as test scenarios
- [ ] Lean 4 proof checking integrated
- [ ] All 5 experiment protocols executable
- [ ] Bootstrap CI on all statistical claims
- [ ] Full parameter sweep automation

### 9.2 Security Requirements

- [ ] RT-01 through RT-05 attacks documented with mitigations
- [ ] Security invariants enforced via CI
- [ ] Adversarial test suite passes
- [ ] BFT protocol implemented (not just specified)
- [ ] Behavioral correlation detection functional

### 9.3 Validation Requirements

- [ ] Monte Carlo convergence verified (Gelman-Rubin < 1.1)
- [ ] Cross-validation variance < 10% of mean
- [ ] Proven bounds validated against empirical data
- [ ] Edge cases documented (small m, high ρ, boundary effects)

---

## 10. OPEN QUESTIONS FOR FORMAL METHODS REVIEW

1. **Hyperplane Distribution:** Code uses ortho_group × Uniform([0.2, 0.8]); theory assumes Grassmannian. Are these equivalent for our purposes?

2. **Adaptive Deception:** All theorems assume non-adaptive adversary. Should we add adaptive analysis or accept this as a known limitation?

3. **Compositional Detection:** Red team identified emergent deception as FUNDAMENTAL flaw. Is there a formal framework for detecting deception from honest components?

4. **ETH Dependence:** Complexity claims require ETH. Should we state all results conditionally, or is there unconditional formulation?

5. **System vs Individual:** Framework treats deception as individual property. Can type-theoretic structure capture systemic deception?

6. **Non-Convex Deceptive Regions:** Current framework assumes convex deceptive regions (ball geometry). Can we extend to non-convex regions? Potential approaches:
   - **Convex decomposition:** Decompose non-convex region into convex components, apply theorem to each, aggregate bounds
   - **Topological methods:** Use persistent homology or other topological invariants to bound volume shrinkage
   - **Adversarial geometry:** Prove bounds on how well an adversary can shape non-convex regions to avoid constraints
   - **Empirical characterization:** Identify which non-convex region classes (e.g., star-shaped, finite union of convex sets) still admit tractable analysis

   **Current status:** Non-convex regions are UNSUPPORTED and flagged as attack vector RT-06.
   **Reference:** FORMALIZATION_ROADMAP.md Section 4.1.2 (Parameter Regimes Where Claim Fails)

---

## 11. KNOWN LIMITATIONS

### 11.1 Non-Adaptive Adversary Assumption (CRITICAL)

**Limitation ID:** M-01
**Severity:** FUNDAMENTAL
**Affects:** All detection power claims (Section 3.3), Security invariants (Section 7.2)

#### Formal Statement

All detection power guarantees in this specification assume:

```
adversary.strategy_fixed_before(detection.threshold)
```

Equivalently, the deceptive distribution P_D must be fixed BEFORE the detector threshold tau is chosen. The adversary cannot observe detector responses and adapt their strategy.

#### What Breaks With Adaptive Adversary

**Theorem (Adaptive Evasion):** An adaptive adversary with query access to the detector can evade detection with O(sqrt(n)) threshold queries.

**Proof Sketch:**
1. Detector threshold tau is set to achieve false positive rate alpha
2. Adaptive adversary binary searches for tau using O(log(1/epsilon)) queries
3. Once tau is known, adversary can generate traces that fall just below threshold
4. With n samples, threshold estimation has standard error O(1/sqrt(n))
5. Adversary needs O(sqrt(n)) queries to estimate threshold to sufficient precision

**Formal Degradation:**

For a B-adaptive adversary (B = number of threshold queries allowed):

```
Power_adaptive <= Power_non_adaptive * (1 - O(B / sqrt(n)))
```

When B = O(sqrt(n)), detection power approaches zero.

**Specific Failures:**

| Claim | Non-Adaptive | Adaptive (B = sqrt(n)) |
|-------|--------------|------------------------|
| LRT Optimality | Neyman-Pearson optimal | Suboptimal, adversary knows decision boundary |
| Sample Complexity n | O(1 / (D^2 * p)) | Unbounded (adversary can always evade) |
| Power >= 0.90 | Achievable with sufficient n | NOT achievable regardless of n |
| Detection latency | O(n) samples | Infinite (never detects) |

#### Mitigation Strategies (Partial)

These mitigations REDUCE but do NOT eliminate adaptive adversary advantage:

1. **Moving Thresholds:** Randomize tau over time
   - Reduces adaptive advantage to O(n / B_threshold) where B_threshold = threshold update frequency
   - Increases false positive rate by O(threshold_variance)

2. **Ensemble Detectors:** Use multiple detectors with hidden weights
   - Adversary must query all detectors to evade
   - Multiplicative increase in query budget required

3. **Rate Limiting:** Limit detector query rate
   - Bounds B directly
   - May impact legitimate use

4. **Differential Privacy:** Add noise to detector responses
   - Provides (epsilon, delta)-indistinguishability
   - Degrades detection accuracy for defenders too

#### Verification Status

This limitation CANNOT be formally verified away. It is a fundamental game-theoretic property:

```
INVARIANT: adversary.strategy_fixed_before(detection.threshold)
VERIFIABLE: NO (game-theoretic, not proof-theoretic)
MITIGATION: Partial (see above)
RESIDUAL RISK: Adaptive adversary with sufficient query budget can evade
```

#### Recommendations

1. **Document prominently:** All detection power claims must note non-adaptive assumption
2. **Implement mitigations:** Moving thresholds + ensemble as baseline defense
3. **Monitor query patterns:** Detect potential threshold probing
4. **Assume breach:** Design downstream systems to handle detection failure

### 11.2 Other Known Limitations

| ID | Limitation | Severity | Reference |
|----|------------|----------|-----------|
| M-02 | Hyperplane distribution mismatch (code vs theory) | MEDIUM | Section 10 Q1 |
| M-03 | Finite sample validity (n >= 100 required) | MEDIUM | Section 3.3 |
| M-04 | Convexity assumption for deceptive region | MEDIUM | Section 3.1 |
| M-05 | ETH dependence for complexity gap | HIGH | Section 10 Q4 |
| M-06 | Emergent deception (RT-01) is FUNDAMENTAL flaw | CRITICAL | Section 3.5 |

---

**END OF FSD**

*This document synthesizes requirements from: Systems Architect (architecture), Computational Scientist (validation), Social Scientist (experiments), Formal Methods (proofs), and Red Team (security). Ready for formal review.*
