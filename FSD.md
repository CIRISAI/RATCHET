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

**Purpose:** Validate topological collapse theorem V(k) = V(0) × exp(-λk)

**Requirements from Formal Methods:**
- Must support both orthonormal AND correlated constraint sampling
- Must track effective rank k_eff = k / (1 + ρ(k-1))
- Must quantify boundary effects at edge of [0,1]^D
- Must provide error bounds on exponential approximation

**Requirements from Red Team:**
- MUST implement null-space attack simulation
- MUST support adversarial constraint probing
- MUST test non-convex deceptive regions (torus, point cloud, fractal)
- MUST support adaptive/moving target deception

```python
class GeometricEngine:
    """
    Hyperplane intersection volume estimation with adversarial robustness.
    """

    def estimate_volume(
        self,
        dimension: int,
        num_constraints: int,
        deceptive_radius: float,
        constraint_correlation: float = 0.0,  # ρ parameter
        sampling_mode: Literal["orthonormal", "correlated", "adversarial"] = "orthonormal",
        num_samples: int = 100_000,
        adversary: Optional[AdversarialStrategy] = None,
    ) -> VolumeEstimate:
        """
        Returns volume estimate with bootstrap confidence interval.

        If adversary is provided, simulates null-space or moving-target attack.
        """
        ...

    def compute_effective_rank(
        self,
        constraints: List[Hyperplane],
    ) -> EffectiveRankResult:
        """
        Compute k_eff accounting for constraint correlation.
        Returns effective rank and correlation matrix.
        """
        ...

    def null_space_attack(
        self,
        constraints: List[Hyperplane],
        probe_budget: int,
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

**Purpose:** Validate computational asymmetry T_D / T_H = Ω(2^m / poly(n))

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

```python
class ComplexityEngine:
    """
    Deception complexity analysis with realistic SAT solving.
    """

    def __init__(
        self,
        solver: Literal["z3", "minisat", "cadical", "bruteforce"] = "z3",
    ):
        self.solver = solver

    def measure_complexity(
        self,
        world_size: int,
        num_statements: int,
        literals_per_statement: int,
        observable_fraction: float = 1.0,  # Sparse deception parameter
        deception_strategy: Literal["full", "sparse", "lazy"] = "full",
    ) -> ComplexityResult:
        """
        Compare honest vs deceptive agent computational cost.
        Returns T_H, T_D, ratio, and 95% CI.
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
        num_lies: int,
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

**Requirements from Formal Methods:**
- MUST implement Neyman-Pearson optimal LRT
- MUST compute Mahalanobis distance with proper covariance estimation
- MUST provide asymptotic AND finite-sample analysis
- MUST quantify estimation error in plug-in D̂

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
- MUST implement adaptive detection (moving thresholds)

```python
class DetectionEngine:
    """
    Statistical deception detection with adversarial robustness.
    """

    def train_detector(
        self,
        honest_traces: np.ndarray,
        deceptive_traces: Optional[np.ndarray] = None,
        method: Literal["lrt", "mahalanobis", "isolation_forest", "ensemble"] = "lrt",
    ) -> Detector:
        """
        Train detector on honest traces. If deceptive_traces provided,
        use for supervised training; otherwise, use unsupervised.
        """
        ...

    def power_analysis(
        self,
        mahalanobis_distance: float,
        deception_rate: float,
        alpha: float = 0.05,
        beta: float = 0.05,
    ) -> PowerAnalysisResult:
        """
        Compute required sample size for given power.
        Returns n, analytical formula, and finite-sample correction.
        """
        ...

    def adversarial_robustness(
        self,
        detector: Detector,
        attack: Literal["mimicry", "flooding", "adaptive"],
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
- MUST quantify Byzantine threshold (n ≥ 3f + 1)

```python
class FederationEngine:
    """
    Federated ratchet simulation with Byzantine fault tolerance.
    """

    def __init__(
        self,
        consensus_protocol: Literal["pbft", "raft", "tendermint"] = "pbft",
        mi_threshold: float = 0.85,
    ):
        self.consensus = consensus_protocol
        self.mi_threshold = mi_threshold

    def create_federation(
        self,
        num_honest: int,
        num_malicious: int = 0,
        malicious_strategy: Literal["random", "coordinated", "slow_capture"] = "random",
    ) -> Federation:
        """
        Create federation with specified agent composition.
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
        """
        ...

    def slow_capture_simulation(
        self,
        federation: Federation,
        capture_rate_per_period: float,
        num_periods: int,
    ) -> CaptureSimulationResult:
        """
        Simulate gradual federation takeover.
        Returns period when BFT threshold breached.
        """
        ...

    def precedent_poisoning_detection(
        self,
        precedent_stream: List[Precedent],
        window_size: int,
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
    # Byzantine tolerance
    "federation.malicious_fraction < 0.33",

    # Detection power
    "detection.power(n=1000, D=1.0, p=0.01) >= 0.90",

    # Geometric robustness (with correlation)
    "geometric.volume_reduction(k=50, rho=0.5) >= 0.50",

    # Complexity gap (with modern solver)
    "complexity.ratio(m=20, solver='z3') >= 5.0",

    # Anti-Sybil
    "federation.behavioral_correlation_detection(coordinated_sybils) == True",
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

**References for Finite Sample Analysis:**
- Berry, A.C. (1941). "The accuracy of the Gaussian approximation to the sum of
  independent variates." Trans. Amer. Math. Soc.
- Esseen, C.G. (1942). "On the Liapunoff limit of error in the theory of probability."
  Ark. Mat. Astron. Fys.
- Shevtsova, I.G. (2011). "On the absolute constants in the Berry-Esseen type
  inequalities for identically distributed summands." arXiv:1111.6554

**Dependency Note:** These invariants depend on the asymptotic validity specification
from wt-7 (DP-4). The n >= 100 threshold is derived from requiring Berry-Esseen
error < 0.05, which corresponds to 0.4748/sqrt(100) = 0.0475 < 0.05.

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

**END OF FSD**

*This document synthesizes requirements from: Systems Architect (architecture), Computational Scientist (validation), Social Scientist (experiments), Formal Methods (proofs), and Red Team (security). Ready for formal review.*
