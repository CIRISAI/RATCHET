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

**Requirements from Formal Methods:**
- Must support both orthonormal AND correlated constraint sampling
- Must track effective rank k_eff = k / (1 + rho*(k-1))
- Must quantify boundary effects at edge of [0,1]^D
- Must provide error bounds on exponential approximation

**Requirements from Red Team:**
- MUST implement null-space attack simulation
- MUST support adversarial constraint probing
- MUST test non-convex deceptive regions (torus, point cloud, fractal)
- MUST support adaptive/moving target deception

**Refinement Types (from schemas/types.py):**
```python
from schemas.types import (
    Dimension,           # int > 0: Prevents zero/negative dimension crashes
    NumConstraints,      # int > 0: Number of hyperplane constraints
    Radius,              # 0 < float < 0.5: Deceptive region radius
    Correlation,         # -1 <= float <= 1: Constraint correlation rho
    SamplingMode,        # Enum: orthonormal, correlated, adversarial
    SampleSize,          # int >= 1: Number of Monte Carlo samples
    AdversarialStrategy, # Attack specification type
    Hyperplane,          # Normal vector + offset representation
    VolumeEstimate,      # Result with CI and decay constant
    EffectiveRankResult, # Effective rank with correlation matrix
)
```

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
class ComplexityEngine:
    """
    Deception complexity analysis with realistic SAT solving.

    SECURITY NOTE: All claims about exponential complexity gap
    require ETH assumption and k >= 3 (Literals type enforces this).
    """

    def __init__(
        self,
        solver: SATSolver = SATSolver.Z3,
    ):
        self.solver = solver

    def measure_complexity(
        self,
        world_size: WorldSize,                       # int >= 1
        num_statements: NumStatements,               # int > 0
        literals_per_statement: Literals,            # int >= 3 (NP-hard)
        observable_fraction: ObservableFraction = 1.0,  # 0 < f <= 1
        deception_strategy: DeceptionStrategy = DeceptionStrategy.FULL,
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
2. Scaling law: Fit T_D(m) = a × 2^(b×m) + c; validate b ≈ 1
3. Ratio validation: Report T_D/T_H with error bars across parameter sweep
4. Edge cases: m ≤ 15 (brute force feasible), k=2 (2-SAT tractable)

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

**Refinement Types (from schemas/types.py):**
```python
from schemas.types import (
    NodeCount,           # int >= 1: Number of federation nodes
    ByzantineFraction,   # 0 <= float < 1/3: Byzantine fraction limit
    CaptureRate,         # 0 <= float < 1: Capture rate per period
    Probability,         # 0 < float < 1: MI threshold
    ConsensusProtocol,   # Enum: pbft, raft, tendermint
    MaliciousStrategy,   # Enum: random, coordinated, slow_capture
    Vote,                # Vote record type
    Precedent,           # Precedent record type
    FederationParams,    # Type-safe parameter bundle with BFT validation
)

# BFT INVARIANT: malicious_fraction < 1/3
# Enforced by FederationParams model validator
```

```python
class FederationEngine:
    """
    Federated ratchet simulation with Byzantine fault tolerance.

    SECURITY INVARIANT: federation.malicious_fraction < 0.33
    This is validated at construction time by FederationParams.
    """

    def __init__(
        self,
        consensus_protocol: ConsensusProtocol = ConsensusProtocol.PBFT,
        mi_threshold: Probability = 0.85,         # 0 < threshold < 1
    ):
        self.consensus = consensus_protocol
        self.mi_threshold = mi_threshold

    def create_federation(
        self,
        num_honest: NodeCount,                    # int >= 1
        num_malicious: int = 0,
        malicious_strategy: MaliciousStrategy = MaliciousStrategy.RANDOM,
    ) -> Federation:
        """
        Create federation with specified agent composition.

        VALIDATES: num_malicious / (num_honest + num_malicious) < 1/3
        Raises ValueError if BFT threshold would be violated.
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
        Returns correlation matrix and flagged agent pairs.

        Detection guarantee: If agents pass MI gate independently but
        coordinate votes with correlation > 0.8, detects with
        probability >= 1 - beta.
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
        Returns period when BFT threshold breached.

        Security: Federation remains secure for O(1/rate) periods.
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

```python
class SimulationRequest(BaseModel):
    engine: Literal["geometric", "complexity", "detection", "federation"]
    parameters: Dict[str, Any]
    adversarial: bool = False
    adversary_config: Optional[AdversaryConfig] = None
    num_runs: int = 1
    seed: Optional[int] = None

class SimulationResult(BaseModel):
    id: str
    engine: str
    status: Literal["pending", "running", "completed", "failed"]
    metrics: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    adversarial_results: Optional[AdversarialMetrics] = None
    provenance: ProvenanceRecord
    created_at: datetime
    completed_at: Optional[datetime]

class AttackScenario(BaseModel):
    attack_id: str  # RT-01 through RT-05
    target_engine: str
    params: Dict[str, Any]
    success_threshold: float = 0.5

class ProofObligation(BaseModel):
    id: str
    claim: str
    theorem_statement: str
    lean_file: Optional[str]
    status: Literal["pending", "proven", "disproven", "blocked"]
    dependencies: List[str]
```

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

### 7.2 Security Invariants

```python
# These invariants MUST hold for any deployment consideration

SECURITY_INVARIANTS = [
    # ==========================================================================
    # CRITICAL: Non-Adaptive Adversary Assumption (M-01)
    # All detection power claims REQUIRE this invariant
    # ==========================================================================
    "adversary.strategy_fixed_before(detection.threshold)",
    # Formal: P_D is measurable w.r.t. sigma-algebra generated by {honest_traces},
    #         NOT w.r.t. {detection.threshold}
    # Violation: Adaptive adversary evades with O(sqrt(n)) queries (see Section 11)

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
]
```

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
