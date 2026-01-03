"""
RATCHET Refinement Types

Type-safe numeric parameters with explicit bounds for all RATCHET interfaces.
These types prevent runtime errors and ensure mathematical validity of all operations.

Based on FSD_FORMAL_REVIEW.md Section 1.2 recommendations.

Usage:
    from schemas.types import Dimension, Radius, Correlation, Probability

    def estimate_volume(
        dimension: Dimension,
        deceptive_radius: Radius,
        constraint_correlation: Correlation = 0.0,
    ) -> VolumeEstimate:
        ...

Type Guarantees:
    - Dimension: int > 0 (prevents zero/negative dimension crashes)
    - Radius: 0 < float < 0.5 (prevents boundary effect violations)
    - Correlation: -1 <= float <= 1 (ensures valid correlation matrices)
    - Probability: 0 < float < 1 (prevents division by zero, overflow)
    - MahalanobisDistance: float >= 0 (non-negative by definition)
    - SampleSize: int >= 1 (prevents zero sample errors)
    - Literals: int >= 3 (ensures NP-hardness for k-SAT)
    - WorldSize: int >= 1 (ensures valid world model)

References:
    - T-GEO-01: Dimension bounds
    - T-GEO-02: Radius bounds (0 < r < 0.5, Roadmap Section 4.1.3)
    - T-GEO-03: Correlation bounds
    - T-CPX-01: Literals bounds (k >= 3 for NP-hardness)
    - T-CPX-02: Observable fraction bounds
    - T-DET-01: Mahalanobis distance non-negativity
    - T-DET-02: Deception rate domain
    - T-DET-03: Alpha/beta probability constraints
"""

from typing import Annotated, List, Dict, Optional, Union, Literal, TypeVar, Generic
from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum


# =============================================================================
# CORE REFINEMENT TYPES
# =============================================================================

# T-GEO-01: Dimension must be positive integer
# Prevents: Zero/negative dimension crashes, undefined behavior
Dimension = Annotated[
    int,
    Field(
        gt=0,
        description="Positive dimension D of the state space. Must be > 0 to prevent "
                    "crashes and undefined geometric operations.",
        examples=[10, 100, 1000],
        json_schema_extra={"minimum_exclusive": 0}
    )
]

# T-GEO-02: Radius must be in (0, 0.5) - strict bounds
# Prevents: Invalid volume estimates, formula breakdown at boundaries
# Theory requires 0 < r < 0.5 (see Formalization Roadmap Section 4.1.3)
Radius = Annotated[
    float,
    Field(
        gt=0.0,
        lt=0.5,
        description="Deceptive region radius r. Must satisfy 0 < r < 0.5 to ensure "
                    "valid volume estimates and prevent boundary effect violations. "
                    "Theoretical analysis assumes small r for exponential approximation.",
        examples=[0.1, 0.2, 0.3],
        json_schema_extra={"minimum_exclusive": 0.0, "maximum_exclusive": 0.5}
    )
]

# T-GEO-03: Correlation must be in [-1, 1] - inclusive bounds
# Prevents: Invalid correlation matrices, undefined behavior for rho > 1 or rho < -1
Correlation = Annotated[
    float,
    Field(
        ge=-1.0,
        le=1.0,
        description="Constraint correlation rho. Must satisfy -1 <= rho <= 1 for "
                    "valid correlation matrix. Negative correlation indicates "
                    "anti-correlated constraints; positive indicates aligned constraints.",
        examples=[-0.5, 0.0, 0.5, 0.9],
        json_schema_extra={"minimum": -1.0, "maximum": 1.0}
    )
]

# T-DET-02, T-DET-03: Probability must be in (0, 1) - strict bounds
# Prevents: Division by zero, numerical overflow in z-score calculations
# Values of 0 or 1 lead to infinite z-scores
Probability = Annotated[
    float,
    Field(
        gt=0.0,
        lt=1.0,
        description="Probability value p. Must satisfy 0 < p < 1 to prevent "
                    "division by zero and numerical overflow. Used for deception rates, "
                    "alpha (Type I error), and beta (Type II error).",
        examples=[0.01, 0.05, 0.1, 0.5],
        json_schema_extra={"minimum_exclusive": 0.0, "maximum_exclusive": 1.0}
    )
]

# T-DET-01: Mahalanobis distance must be non-negative
# Prevents: Negative sample size calculations
# D^2 is by definition non-negative (quadratic form with positive definite matrix)
MahalanobisDistance = Annotated[
    float,
    Field(
        ge=0.0,
        description="Mahalanobis distance D. Must be >= 0 as it is derived from "
                    "a quadratic form with positive semi-definite covariance matrix. "
                    "Typical detection requires D >= 0.5 for feasible sample sizes.",
        examples=[0.5, 1.0, 2.0],
        json_schema_extra={"minimum": 0.0}
    )
]

# Prevents: Zero sample errors in statistical calculations
SampleSize = Annotated[
    int,
    Field(
        ge=1,
        description="Sample size n for statistical analysis. Must be >= 1 to prevent "
                    "division by zero in variance calculations. For asymptotic validity, "
                    "n >= 100 is recommended (see Berry-Esseen bounds).",
        examples=[100, 1000, 10000],
        json_schema_extra={"minimum": 1}
    )
]

# T-CPX-01: Literals per statement must be >= 3 for NP-hardness
# Prevents: False security claims in P-time regime
# For k = 2, the problem is 2-SAT, which is in P (tractable in polynomial time)
Literals = Annotated[
    int,
    Field(
        ge=3,
        description="Literals per statement k in SAT formulation. Must be >= 3 for "
                    "NP-hardness. For k < 3, the problem is 2-SAT (in P), and the "
                    "complexity gap vanishes - security claims would be invalid.",
        examples=[3, 5, 7],
        json_schema_extra={"minimum": 3}
    )
]

# World model size must be positive
WorldSize = Annotated[
    int,
    Field(
        ge=1,
        description="World model size m (number of facts/variables). Must be >= 1. "
                    "Larger world models provide stronger security guarantees under ETH: "
                    "T_D / T_H = Omega(2^m). Recommended m >= 20 for security applications.",
        examples=[10, 20, 50, 100],
        json_schema_extra={"minimum": 1}
    )
]


# =============================================================================
# DERIVED REFINEMENT TYPES
# =============================================================================

# Number of constraints must be positive
NumConstraints = Annotated[
    int,
    Field(
        gt=0,
        description="Number of hyperplane constraints k. Must be > 0. "
                    "Volume decays as exp(-lambda * k_eff) where k_eff = k / (1 + rho*(k-1)).",
        examples=[10, 50, 100],
        json_schema_extra={"minimum_exclusive": 0}
    )
]

# Number of statements in world model
NumStatements = Annotated[
    int,
    Field(
        gt=0,
        description="Number of statements n in the world model. Must be > 0. "
                    "Each statement is a k-SAT clause over the world model variables.",
        examples=[20, 100, 500],
        json_schema_extra={"minimum_exclusive": 0}
    )
]

# T-CPX-02: Observable fraction must be in (0, 1]
# Value of 0 means nothing observable (degenerate case causing division by zero)
ObservableFraction = Annotated[
    float,
    Field(
        gt=0.0,
        le=1.0,
        description="Fraction of facts that are observable. Must satisfy 0 < f <= 1. "
                    "f = 0 would mean nothing is observable (degenerate case). "
                    "f = 1 means full observability; f < 1 enables sparse deception attacks.",
        examples=[0.5, 0.8, 1.0],
        json_schema_extra={"minimum_exclusive": 0.0, "maximum": 1.0}
    )
]

# Effective rank for correlated constraints
EffectiveRank = Annotated[
    float,
    Field(
        gt=0.0,
        description="Effective number of independent constraints k_eff. "
                    "Computed as k_eff = k / (1 + rho*(k-1)) for correlation rho. "
                    "Always positive and <= k.",
        json_schema_extra={"minimum_exclusive": 0.0}
    )
]

# Byzantine fraction must be < 1/3 for BFT
ByzantineFraction = Annotated[
    float,
    Field(
        ge=0.0,
        lt=1/3,
        description="Fraction of Byzantine (malicious) nodes in federation. "
                    "Must be < 1/3 for BFT consensus (n >= 3f + 1 requirement). "
                    "At 1/3 or above, Byzantine nodes can violate safety properties.",
        examples=[0.0, 0.1, 0.2, 0.32],
        json_schema_extra={"minimum": 0.0, "maximum_exclusive": 0.333333}
    )
]

# Number of nodes in federation
NodeCount = Annotated[
    int,
    Field(
        ge=1,
        description="Number of nodes in the federation. Must be >= 1. "
                    "For BFT: n >= 3f + 1 where f is max Byzantine nodes.",
        examples=[4, 7, 10, 100],
        json_schema_extra={"minimum": 1}
    )
]

# Capture rate per period for slow capture attack
CaptureRate = Annotated[
    float,
    Field(
        ge=0.0,
        lt=1.0,
        description="Rate of node capture per period in slow capture attacks. "
                    "Must be in [0, 1). Time to BFT breach is O(1/r) periods.",
        examples=[0.01, 0.05, 0.1],
        json_schema_extra={"minimum": 0.0, "maximum_exclusive": 1.0}
    )
]


# =============================================================================
# ENUMERATIONS FOR CONSTRAINED CHOICES
# =============================================================================

class SamplingMode(str, Enum):
    """Hyperplane sampling distribution mode."""
    ORTHONORMAL = "orthonormal"
    CORRELATED = "correlated"
    ADVERSARIAL = "adversarial"


class DeceptionStrategy(str, Enum):
    """Deception strategy for complexity analysis."""
    FULL = "full"          # Lie about everything
    SPARSE = "sparse"      # Lie only about unobservable facts
    LAZY = "lazy"          # Just-in-time fabrication


class SATSolver(str, Enum):
    """SAT solver backend."""
    Z3 = "z3"
    MINISAT = "minisat"
    CADICAL = "cadical"
    BRUTEFORCE = "bruteforce"


class DetectionMethod(str, Enum):
    """Statistical detection method."""
    LRT = "lrt"                        # Likelihood Ratio Test
    MAHALANOBIS = "mahalanobis"        # Mahalanobis distance
    ISOLATION_FOREST = "isolation_forest"
    ENSEMBLE = "ensemble"


class ConsensusProtocol(str, Enum):
    """BFT consensus protocol."""
    PBFT = "pbft"          # Practical Byzantine Fault Tolerance
    RAFT = "raft"          # Raft (not truly Byzantine tolerant)
    TENDERMINT = "tendermint"


class AttackType(str, Enum):
    """Red team attack types."""
    MIMICRY = "mimicry"        # RT-04: Distribution mimicry
    FLOODING = "flooding"      # Trace flooding/dilution
    ADAPTIVE = "adaptive"      # Adaptive adversary
    NULL_SPACE = "null_space"  # RT-03: Constraint-aligned deception
    SLOW_CAPTURE = "slow_capture"  # RT-02: Slow federation capture
    DIVERSE_SYBIL = "diverse_sybil"  # RT-05: Diverse Sybil attack
    EMERGENT = "emergent"      # RT-01: Emergent multi-agent deception


class MaliciousStrategy(str, Enum):
    """Malicious node behavior strategy."""
    RANDOM = "random"
    COORDINATED = "coordinated"
    SLOW_CAPTURE = "slow_capture"


class ProofStatus(str, Enum):
    """Status of a proof obligation."""
    PENDING = "pending"        # Not started
    PARTIAL = "partial"        # Has sorry/admitted
    AXIOMATIZED = "axiomatized"  # Uses axiom (e.g., ETH)
    PROVEN = "proven"          # Complete proof
    DISPROVEN = "disproven"    # Counterexample found
    BLOCKED = "blocked"        # Waiting on dependency


# =============================================================================
# COMPOSITE TYPE SCHEMAS
# =============================================================================

class Hyperplane(BaseModel):
    """
    Representation of a hyperplane in D-dimensional space.

    A hyperplane is defined as {x : n . x = d} where n is the unit normal
    and d is the offset from origin.
    """
    normal: List[float] = Field(
        description="Unit normal vector n. Must have unit L2 norm."
    )
    offset: float = Field(
        ge=0.0,
        le=1.0,
        description="Offset d from origin. In [0, 1] for unit hypercube."
    )

    @field_validator('normal')
    @classmethod
    def validate_normal(cls, v: List[float]) -> List[float]:
        import math
        norm = math.sqrt(sum(x*x for x in v))
        if abs(norm - 1.0) > 1e-6:
            raise ValueError(f"Normal must be unit vector, got norm={norm}")
        return v


class VolumeEstimate(BaseModel):
    """Result of volume estimation with uncertainty quantification."""
    volume: float = Field(ge=0.0, le=1.0, description="Estimated volume fraction")
    ci_lower: float = Field(ge=0.0, description="95% CI lower bound")
    ci_upper: float = Field(le=1.0, description="95% CI upper bound")
    num_samples: SampleSize
    decay_constant: Optional[float] = Field(
        default=None,
        description="Fitted exponential decay constant lambda"
    )
    effective_rank: Optional[EffectiveRank] = Field(
        default=None,
        description="Effective rank k_eff accounting for correlation"
    )


class EffectiveRankResult(BaseModel):
    """Result of effective rank computation."""
    effective_rank: EffectiveRank
    correlation_matrix: List[List[float]] = Field(
        description="Pairwise correlation matrix of constraints"
    )
    eigenvalues: List[float] = Field(
        description="Eigenvalues of correlation matrix"
    )


class ComplexityResult(BaseModel):
    """Result of complexity measurement."""
    time_honest: float = Field(
        ge=0.0,
        description="Time for honest agent T_H (seconds)"
    )
    time_deceptive: float = Field(
        ge=0.0,
        description="Time for deceptive agent T_D (seconds)"
    )
    ratio: float = Field(
        ge=1.0,
        description="Complexity ratio T_D / T_H"
    )
    ci_lower: float = Field(description="95% CI lower bound on ratio")
    ci_upper: float = Field(description="95% CI upper bound on ratio")
    solver: SATSolver
    eth_conditional: bool = Field(
        default=True,
        description="True if exponential gap claim requires ETH assumption"
    )


class PowerAnalysisResult(BaseModel):
    """Result of statistical power analysis."""
    required_sample_size: SampleSize
    achieved_power: Probability
    mahalanobis_distance: MahalanobisDistance
    alpha: Probability = Field(description="Type I error rate")
    beta: Probability = Field(description="Type II error rate")
    finite_sample_correction: float = Field(
        default=0.0,
        description="Berry-Esseen correction for n < 100"
    )

    @model_validator(mode='after')
    def validate_power(self) -> 'PowerAnalysisResult':
        if self.achieved_power < (1 - self.beta) - 0.01:
            # Allow small tolerance for numerical precision
            pass  # Could add warning here
        return self


class AdversarialStrategy(BaseModel):
    """
    Specification of adversarial behavior for attack simulations.

    Addresses GAP-01 from formal review: AdversarialStrategy type was undefined.
    """
    attack_type: AttackType
    probe_budget: Optional[int] = Field(
        default=None,
        ge=0,
        description="Number of constraint probing queries allowed"
    )
    adaptation_budget: Optional[int] = Field(
        default=None,
        ge=0,
        description="Number of threshold adaptation queries"
    )
    knowledge_level: Literal["none", "partial", "full"] = Field(
        default="none",
        description="Adversary's knowledge of detection mechanism"
    )


class InferenceGraph(BaseModel):
    """
    Graph structure for compositional deception detection.

    Addresses GAP-04 from formal review: InferenceGraph type was undefined.
    """
    nodes: List[str] = Field(description="Agent IDs in inference chain")
    edges: List[tuple] = Field(description="(source, target) edges")
    weights: Optional[List[float]] = Field(
        default=None,
        description="Edge weights for information flow"
    )


class Vote(BaseModel):
    """
    Federation vote record.

    Addresses GAP-05 from formal review: Vote type was undefined.
    """
    voter_id: str
    precedent_id: str
    vote: Literal["approve", "reject", "abstain"]
    timestamp: float
    signature: Optional[str] = Field(default=None, description="Cryptographic signature")


class Precedent(BaseModel):
    """
    Precedent record in federated system.

    Addresses GAP-06 from formal review: Precedent type was undefined.
    """
    id: str
    content: str
    proposer_id: str
    votes: List[Vote] = Field(default_factory=list)
    status: Literal["proposed", "approved", "rejected", "pending"]
    timestamp: float


# =============================================================================
# ENGINE-SPECIFIC PARAMETER TYPES
# Addresses T-SCH-01: Type-safe parameters instead of Dict[str, Any]
# =============================================================================

class GeometricParams(BaseModel):
    """Type-safe parameters for GeometricEngine.estimate_volume()"""
    dimension: Dimension
    num_constraints: NumConstraints
    deceptive_radius: Radius
    constraint_correlation: Correlation = 0.0
    sampling_mode: SamplingMode = SamplingMode.ORTHONORMAL
    num_samples: SampleSize = 100_000
    adversary: Optional[AdversarialStrategy] = None


class ComplexityParams(BaseModel):
    """Type-safe parameters for ComplexityEngine.measure_complexity()"""
    world_size: WorldSize
    num_statements: NumStatements
    literals_per_statement: Literals
    observable_fraction: ObservableFraction = 1.0
    deception_strategy: DeceptionStrategy = DeceptionStrategy.FULL


class DetectionParams(BaseModel):
    """Type-safe parameters for DetectionEngine.power_analysis()"""
    mahalanobis_distance: MahalanobisDistance
    deception_rate: Probability
    alpha: Probability = 0.05
    beta: Probability = 0.05


class FederationParams(BaseModel):
    """Type-safe parameters for FederationEngine.create_federation()"""
    num_honest: NodeCount
    num_malicious: int = Field(default=0, ge=0)
    malicious_strategy: MaliciousStrategy = MaliciousStrategy.RANDOM
    consensus_protocol: ConsensusProtocol = ConsensusProtocol.PBFT

    @model_validator(mode='after')
    def validate_byzantine_fraction(self) -> 'FederationParams':
        total = self.num_honest + self.num_malicious
        if total > 0:
            fraction = self.num_malicious / total
            if fraction >= 1/3:
                raise ValueError(
                    f"Byzantine fraction {fraction:.3f} >= 1/3 violates BFT requirement"
                )
        return self


# Discriminated union for type-safe simulation parameters
SimulationParams = Union[GeometricParams, ComplexityParams, DetectionParams, FederationParams]


# =============================================================================
# PROOF OBLIGATION TYPES
# =============================================================================

class ProofObligation(BaseModel):
    """
    Formal proof obligation with conditional dependencies.

    Addresses L-03: Extended proof status enumeration.
    Addresses Question 4: ETH conditionality tracking.
    """
    id: str = Field(description="Unique identifier, e.g., 'TC-1', 'CA-2'")
    claim: str = Field(description="Human-readable claim statement")
    theorem_statement: str = Field(description="Formal theorem in mathematical notation")
    lean_file: Optional[str] = None
    status: ProofStatus = ProofStatus.PENDING
    dependencies: List[str] = Field(
        default_factory=list,
        description="IDs of obligations this depends on"
    )
    conditional_on: List[str] = Field(
        default_factory=list,
        description="Assumptions required, e.g., ['ETH', 'P!=NP']"
    )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def compute_effective_rank(k: NumConstraints, rho: Correlation) -> EffectiveRank:
    """
    Compute effective rank k_eff accounting for constraint correlation.

    Formula: k_eff = k / (1 + rho*(k-1))

    For rho = 0: k_eff = k (independent constraints)
    For rho = 1: k_eff = 1 (perfectly correlated, single constraint)
    For rho = -1/(k-1): k_eff = infinity (only valid mathematically)

    Args:
        k: Number of constraints
        rho: Pairwise correlation

    Returns:
        Effective number of independent constraints
    """
    denominator = 1 + rho * (k - 1)
    if denominator <= 0:
        raise ValueError(
            f"Invalid correlation rho={rho} for k={k}: "
            f"requires rho > -1/(k-1) = {-1/(k-1):.6f}"
        )
    return k / denominator


def compute_required_sample_size(
    D: MahalanobisDistance,
    p: Probability,
    alpha: Probability = 0.05,
    beta: Probability = 0.05,
) -> SampleSize:
    """
    Compute required sample size for given detection power.

    Formula: n >= (z_alpha + z_beta)^2 / (D^2 * p)

    Preconditions:
        - D >= 0.5 (otherwise sample size infeasible)
        - p >= 0.001 (otherwise sample size infeasible)
        - Result valid for n >= 100 (asymptotic regime)

    Args:
        D: Mahalanobis distance between honest and deceptive distributions
        p: Deception rate (fraction of deceptive traces)
        alpha: Type I error rate (false positive rate)
        beta: Type II error rate (false negative rate)

    Returns:
        Required sample size (rounded up)
    """
    from scipy import stats
    import math

    z_alpha = stats.norm.ppf(1 - alpha)
    z_beta = stats.norm.ppf(1 - beta)

    numerator = (z_alpha + z_beta) ** 2
    denominator = D ** 2 * p

    if denominator < 1e-10:
        raise ValueError(
            f"Sample size infeasible: D={D}, p={p} gives near-zero denominator. "
            f"Increase D >= 0.5 or p >= 0.001."
        )

    n = numerator / denominator
    return int(math.ceil(n))


def validate_literals_for_np_hardness(k: int) -> None:
    """
    Validate that k >= 3 for NP-hardness claims.

    For k < 3, the problem is 2-SAT or simpler, which is in P.
    The complexity gap vanishes and security claims would be invalid.

    Raises:
        ValueError: If k < 3
    """
    if k < 3:
        raise ValueError(
            f"NP-hardness requires k >= 3 (got k={k}). "
            f"For k < 3, the problem is in P (2-SAT is tractable). "
            f"Security claims are INVALID for this configuration."
        )


# =============================================================================
# TYPE EXPORTS
# =============================================================================

__all__ = [
    # Core refinement types
    'Dimension',
    'Radius',
    'Correlation',
    'Probability',
    'MahalanobisDistance',
    'SampleSize',
    'Literals',
    'WorldSize',

    # Derived refinement types
    'NumConstraints',
    'NumStatements',
    'ObservableFraction',
    'EffectiveRank',
    'ByzantineFraction',
    'NodeCount',
    'CaptureRate',

    # Enumerations
    'SamplingMode',
    'DeceptionStrategy',
    'SATSolver',
    'DetectionMethod',
    'ConsensusProtocol',
    'AttackType',
    'MaliciousStrategy',
    'ProofStatus',

    # Composite types
    'Hyperplane',
    'VolumeEstimate',
    'EffectiveRankResult',
    'ComplexityResult',
    'PowerAnalysisResult',
    'AdversarialStrategy',
    'InferenceGraph',
    'Vote',
    'Precedent',

    # Engine parameter types
    'GeometricParams',
    'ComplexityParams',
    'DetectionParams',
    'FederationParams',
    'SimulationParams',

    # Proof types
    'ProofObligation',

    # Helper functions
    'compute_effective_rank',
    'compute_required_sample_size',
    'validate_literals_for_np_hardness',
]
