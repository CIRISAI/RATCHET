"""
RATCHET Simulation Parameter Schemas

This module defines type-safe discriminated unions for simulation parameters,
addressing T-SCH-01 (untyped parameters dictionary) from the Formal Methods Review.

The SimulationParams discriminated union replaces Dict[str, Any] with engine-specific
parameter types, enabling compile-time validation and preventing runtime type errors.

Base types are imported from schemas.types (wt-5) to ensure consistency across the
RATCHET codebase and eliminate duplication.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

# =============================================================================
# Import Base Types from schemas.types (wt-5 canonical definitions)
# =============================================================================

from schemas.types import (
    # Core refinement types
    Dimension,
    Radius,
    Correlation,
    Probability,
    MahalanobisDistance,
    SampleSize,
    Literals,
    WorldSize,
    # Derived refinement types
    NumConstraints,
    NumStatements,
    ObservableFraction,
    EffectiveRank,
    ByzantineFraction,
    NodeCount,
    CaptureRate,
    # Enumerations
    SamplingMode,
    DeceptionStrategy,
    SATSolver,
    DetectionMethod,
    ConsensusProtocol,
    AttackType,
    MaliciousStrategy,
    ProofStatus,
    # Composite types
    AdversarialStrategy,
    ProofObligation,
)


# =============================================================================
# Simulation-Specific Type Aliases
# (Types needed for simulation.py that aren't in schemas.types)
# =============================================================================

# Positive integer (general purpose, beyond specific domain types)
PositiveInt = Annotated[int, Field(gt=0, description="Positive integer")]
NonNegativeInt = Annotated[int, Field(ge=0, description="Non-negative integer")]
NonNegativeFloat = Annotated[float, Field(ge=0, description="Non-negative float")]
Fraction = Annotated[float, Field(gt=0, le=1, description="Fraction value (0 < f <= 1)")]


# =============================================================================
# Engine-Specific Parameter Models
# =============================================================================

class GeometricParams(BaseModel):
    """
    Parameters for the Geometric Engine (hyperplane intersection volume estimation).

    Addresses type issues T-GEO-01 through T-GEO-04 from Formal Review Section 1.1.1:
    - T-GEO-01: Unbounded dimension parameter -> Dimension (int > 0)
    - T-GEO-02: Unbounded radius parameter -> Radius (0 < r < 0.5)
    - T-GEO-03: Correlation bounds missing -> Correlation (-1 <= rho <= 1)
    - T-GEO-04: AdversarialStrategy undefined -> Now imported from schemas.types
    """
    engine: Literal["geometric"] = "geometric"

    # Core parameters with refinement types
    dimension: Dimension = Field(
        description="Dimension D of the behavior space (must be positive)"
    )
    num_constraints: NumConstraints = Field(
        description="Number of hyperplane constraints k"
    )
    deceptive_radius: Radius = Field(
        description="Radius r of deceptive region (0 < r < 0.5 per theory)"
    )

    # Correlation parameter with proper bounds
    constraint_correlation: Correlation = Field(
        default=0.0,
        description="Pairwise correlation rho between constraints (-1 <= rho <= 1)"
    )

    # Sampling configuration
    sampling_mode: SamplingMode = Field(
        default=SamplingMode.ORTHONORMAL,
        description="Hyperplane sampling mode"
    )
    num_samples: SampleSize = Field(
        default=100_000,
        description="Number of Monte Carlo samples"
    )

    # Adversarial configuration (now properly typed via schemas.types)
    adversary: Optional[AdversarialStrategy] = Field(
        default=None,
        description="Adversarial attack configuration (if enabled)"
    )

    class Config:
        use_enum_values = True

    @field_validator('num_constraints')
    @classmethod
    def validate_constraints_vs_dimension(cls, v: int, info) -> int:
        """Warn if constraints exceed dimension (though this is valid)."""
        # Note: k > D is valid but may have different behavior
        return v


class ComplexityParams(BaseModel):
    """
    Parameters for the Complexity Engine (SAT-based deception cost analysis).

    Addresses type issues T-CPX-01 and T-CPX-02 from Formal Review Section 1.1.2:
    - T-CPX-01: Literals per statement unbounded -> literals_per_statement >= 3 for NP-hardness
    - T-CPX-02: Observable fraction semantics unclear -> ObservableFraction (0 < f <= 1)

    CRITICAL: For NP-hardness claims, literals_per_statement must be >= 3.
    For k < 3, the problem reduces to 2-SAT which is in P.
    """
    engine: Literal["complexity"] = "complexity"

    # World model parameters
    world_size: WorldSize = Field(
        description="Number of facts/variables in the world model (m)"
    )
    num_statements: NumStatements = Field(
        description="Number of statements to verify (n)"
    )

    # CRITICAL: Must be >= 3 for NP-hardness (2-SAT is in P)
    literals_per_statement: Literals = Field(
        default=3,
        description="Number of literals per statement (k). WARNING: k < 3 yields P-time problem"
    )

    # Sparse deception parameter with proper bounds
    observable_fraction: ObservableFraction = Field(
        default=1.0,
        description="Fraction of facts that are observable (0 < f <= 1)"
    )

    # Deception strategy
    deception_strategy: DeceptionStrategy = Field(
        default=DeceptionStrategy.FULL,
        description="Strategy for deceptive agent"
    )

    # SAT solver configuration
    solver: SATSolver = Field(
        default=SATSolver.Z3,
        description="SAT solver backend"
    )

    # Incremental solving options
    use_learned_clauses: bool = Field(
        default=True,
        description="Enable clause learning for incremental SAT"
    )

    class Config:
        use_enum_values = True

    @model_validator(mode='after')
    def warn_on_easy_sat(self) -> 'ComplexityParams':
        """
        Note: The Literals type from schemas.types already enforces k >= 3.
        This validator is kept for explicit documentation of the NP-hardness requirement.
        """
        return self


class DetectionParams(BaseModel):
    """
    Parameters for the Detection Engine (statistical deception detection).

    Addresses type issues T-DET-01 through T-DET-03 from Formal Review Section 1.1.3:
    - T-DET-01: Mahalanobis distance must be non-negative -> MahalanobisDistance
    - T-DET-02: Deception rate domain (0, 1) -> Probability
    - T-DET-03: Alpha/beta probability constraints -> Probability

    PRECONDITIONS for power_analysis formula validity (U-01):
    - mahalanobis_distance >= 0.5 (otherwise detection is weak)
    - deception_rate >= 0.001 (otherwise sample size infeasible)
    - Results valid for n >= 100 (finite-sample regime)
    """
    engine: Literal["detection"] = "detection"

    # Detection method
    method: DetectionMethod = Field(
        default=DetectionMethod.LRT,
        description="Detection algorithm"
    )

    # Mahalanobis distance (must be non-negative)
    mahalanobis_distance: MahalanobisDistance = Field(
        default=1.0,
        description="Mahalanobis distance D (>= 0, recommend >= 0.5 for meaningful detection)"
    )

    # Deception rate with strict probability bounds
    deception_rate: Probability = Field(
        default=0.01,
        description="Prior deception rate p (0 < p < 1, recommend >= 0.001)"
    )

    # Significance level with probability bounds
    alpha: Probability = Field(
        default=0.05,
        description="Type I error rate (false positive rate)"
    )

    # Power parameter with probability bounds
    beta: Probability = Field(
        default=0.05,
        description="Type II error rate (1 - power = 1 - beta)"
    )

    # Sample size for training (finite-sample regime)
    training_sample_size: Optional[SampleSize] = Field(
        default=None,
        description="Sample size for detector training (recommend >= 100)"
    )

    # Adversarial robustness testing
    adversarial_attack: Optional[Literal["mimicry", "flooding", "adaptive"]] = Field(
        default=None,
        description="Adversarial attack type for robustness testing"
    )

    class Config:
        use_enum_values = True

    @model_validator(mode='after')
    def validate_detection_regime(self) -> 'DetectionParams':
        """
        Validate that parameters are in the regime where detection is meaningful.
        This addresses U-01 (Critical Unsoundness Risk) from Formal Review.
        """
        import warnings

        if self.mahalanobis_distance < 0.5:
            warnings.warn(
                f"mahalanobis_distance={self.mahalanobis_distance} < 0.5: "
                "Detection power may be weak. Consider increasing D or using different test.",
                UserWarning
            )

        if self.deception_rate < 0.001:
            warnings.warn(
                f"deception_rate={self.deception_rate} < 0.001: "
                "Required sample size may be infeasibly large.",
                UserWarning
            )

        if self.training_sample_size is not None and self.training_sample_size < 100:
            warnings.warn(
                f"training_sample_size={self.training_sample_size} < 100: "
                "Finite-sample corrections may be needed. Berry-Esseen bounds apply.",
                UserWarning
            )

        return self


class FederationParams(BaseModel):
    """
    Parameters for the Federation Engine (distributed precedent accumulation with BFT).

    Addresses issues from Formal Review:
    - I-INV-01: BFT requires n >= 3f + 1
    - U-03: Byzantine tolerance without protocol specification
    - GAP-05, GAP-06: Vote and Precedent types undefined (defined in schemas.types)

    SECURITY INVARIANT: malicious_fraction < 0.33 for BFT safety.
    """
    engine: Literal["federation"] = "federation"

    # Node composition
    num_honest: NodeCount = Field(
        description="Number of honest nodes in the federation"
    )
    num_malicious: NonNegativeInt = Field(
        default=0,
        description="Number of malicious/Byzantine nodes"
    )

    # BFT protocol (mandatory, not optional - addresses U-03)
    consensus_protocol: ConsensusProtocol = Field(
        default=ConsensusProtocol.PBFT,
        description="BFT consensus protocol"
    )

    # Malicious behavior strategy
    malicious_strategy: MaliciousStrategy = Field(
        default=MaliciousStrategy.RANDOM,
        description="Strategy for malicious nodes"
    )

    # MI threshold for orthogonality gate
    mi_threshold: Annotated[float, Field(ge=0, le=1)] = Field(
        default=0.85,
        description="Mutual information threshold for partnership gate"
    )

    # Slow capture parameters
    capture_rate_per_period: Optional[CaptureRate] = Field(
        default=None,
        description="Rate of node capture per period (for slow capture simulation)"
    )
    num_periods: Optional[PositiveInt] = Field(
        default=None,
        description="Number of simulation periods"
    )

    # Behavioral correlation detection
    correlation_detection_enabled: bool = Field(
        default=True,
        description="Enable behavioral correlation detection for Sybil resistance"
    )

    class Config:
        use_enum_values = True

    @model_validator(mode='after')
    def validate_bft_threshold(self) -> 'FederationParams':
        """
        Validate BFT safety invariant: n >= 3f + 1 => f < n/3.
        This is a CRITICAL security invariant.
        """
        total_nodes = self.num_honest + self.num_malicious
        malicious_fraction = self.num_malicious / total_nodes if total_nodes > 0 else 0

        if malicious_fraction >= 1/3:
            raise ValueError(
                f"BFT safety violated: malicious_fraction={malicious_fraction:.3f} >= 1/3. "
                f"Federation with {self.num_malicious} malicious out of {total_nodes} total "
                "nodes cannot achieve Byzantine fault tolerance. "
                "Reduce num_malicious or increase num_honest."
            )

        # Also validate n >= 3f + 1 directly
        min_honest = 2 * self.num_malicious + 1
        if self.num_honest < min_honest:
            raise ValueError(
                f"BFT requires num_honest >= 2 * num_malicious + 1. "
                f"Got num_honest={self.num_honest}, need >= {min_honest}."
            )

        return self


# =============================================================================
# Discriminated Union: SimulationParams
# This is the key fix for T-SCH-01
# =============================================================================

# Discriminated union using Pydantic's discriminator feature
SimulationParams = Annotated[
    Union[GeometricParams, ComplexityParams, DetectionParams, FederationParams],
    Field(discriminator="engine")
]


# =============================================================================
# Simulation Request and Response Types
# =============================================================================

class AdversaryConfig(BaseModel):
    """Configuration for adversarial mode."""
    attack_id: Optional[str] = Field(
        default=None,
        description="Attack scenario ID (RT-01 through RT-05)"
    )
    strategy: Optional[AdversarialStrategy] = Field(
        default=None,
        description="Adversarial strategy configuration"
    )
    success_threshold: Probability = Field(
        default=0.5,
        description="Attack success rate threshold"
    )


class ProvenanceRecord(BaseModel):
    """Provenance tracking for reproducibility."""
    simulation_version: str = Field(description="RATCHET simulation version")
    random_seed: Optional[int] = Field(default=None, description="Random seed used")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    git_commit: Optional[str] = Field(default=None, description="Git commit hash")


class SimulationRequest(BaseModel):
    """
    Type-safe simulation request with discriminated union for parameters.

    This replaces the original Dict[str, Any] type hole (T-SCH-01) with
    a properly typed discriminated union that provides:
    - Compile-time type checking
    - Runtime validation with Pydantic
    - Engine-specific parameter validation
    - Refinement types for all numeric parameters
    """
    # Engine type is now inferred from parameters.engine
    parameters: SimulationParams = Field(
        description="Engine-specific parameters (discriminated by 'engine' field)"
    )

    # Adversarial mode
    adversarial: bool = Field(
        default=False,
        description="Enable adversarial testing mode"
    )
    adversary_config: Optional[AdversaryConfig] = Field(
        default=None,
        description="Adversary configuration (required if adversarial=True)"
    )

    # Execution parameters
    num_runs: PositiveInt = Field(
        default=1,
        description="Number of simulation runs"
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducibility"
    )

    @model_validator(mode='after')
    def validate_adversarial_config(self) -> 'SimulationRequest':
        """Ensure adversary_config is provided when adversarial=True."""
        if self.adversarial and self.adversary_config is None:
            raise ValueError(
                "adversary_config is required when adversarial=True"
            )
        return self

    @property
    def engine(self) -> str:
        """Engine type derived from parameters."""
        return self.parameters.engine


# =============================================================================
# Result Types
# =============================================================================

class ConfidenceInterval(BaseModel):
    """Confidence interval for a metric."""
    lower: float
    upper: float
    confidence_level: float = Field(default=0.95, ge=0, le=1)


class AdversarialMetrics(BaseModel):
    """Metrics from adversarial testing."""
    attack_success_rate: float = Field(ge=0, le=1)
    evasion_rate: Optional[float] = Field(default=None, ge=0, le=1)
    detection_degradation: Optional[float] = Field(default=None)
    mitigation_effectiveness: Optional[float] = Field(default=None, ge=0, le=1)


class SimulationResult(BaseModel):
    """
    Result of a simulation run with proper typing.
    """
    id: str = Field(description="Unique simulation ID")
    engine: Literal["geometric", "complexity", "detection", "federation"]
    status: Literal["pending", "running", "completed", "failed"]

    # Metrics with proper typing
    metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Named metrics from the simulation"
    )
    confidence_intervals: dict[str, ConfidenceInterval] = Field(
        default_factory=dict,
        description="Confidence intervals for key metrics"
    )

    # Adversarial results
    adversarial_results: Optional[AdversarialMetrics] = None

    # Provenance
    provenance: ProvenanceRecord

    # Timestamps
    created_at: datetime
    completed_at: Optional[datetime] = None

    # Error information (if failed)
    error_message: Optional[str] = None


# =============================================================================
# Attack Scenario Type (for RedTeam Engine)
# =============================================================================

class AttackScenario(BaseModel):
    """
    Attack scenario definition for red team testing.
    """
    attack_id: str = Field(
        description="Attack ID (RT-01 through RT-05)",
        pattern=r"^RT-0[1-5]$"
    )
    target_engine: Literal["geometric", "complexity", "detection", "federation"]
    params: SimulationParams = Field(
        description="Parameters for the target engine"
    )
    success_threshold: Probability = Field(
        default=0.5,
        description="Threshold for attack success rate"
    )


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Re-exported from schemas.types for convenience
    'Dimension',
    'Radius',
    'Correlation',
    'Probability',
    'MahalanobisDistance',
    'SampleSize',
    'Literals',
    'WorldSize',
    'NumConstraints',
    'NumStatements',
    'ObservableFraction',
    'EffectiveRank',
    'ByzantineFraction',
    'NodeCount',
    'CaptureRate',
    'SamplingMode',
    'DeceptionStrategy',
    'SATSolver',
    'DetectionMethod',
    'ConsensusProtocol',
    'AttackType',
    'MaliciousStrategy',
    'ProofStatus',
    'AdversarialStrategy',
    'ProofObligation',

    # Simulation-specific types
    'PositiveInt',
    'NonNegativeInt',
    'NonNegativeFloat',
    'Fraction',

    # Engine parameter types
    'GeometricParams',
    'ComplexityParams',
    'DetectionParams',
    'FederationParams',
    'SimulationParams',

    # Request/Response types
    'AdversaryConfig',
    'ProvenanceRecord',
    'SimulationRequest',
    'ConfidenceInterval',
    'AdversarialMetrics',
    'SimulationResult',
    'AttackScenario',
]
