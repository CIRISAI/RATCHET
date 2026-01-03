"""
RATCHET Schemas Package

Type-safe Pydantic schemas and refinement types for the RATCHET platform.
Implements recommendations from FSD_FORMAL_REVIEW.md Section 1.2.

Modules:
    types: Core refinement types with explicit bounds for all numeric parameters
"""

from .types import (
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
    Hyperplane,
    VolumeEstimate,
    EffectiveRankResult,
    ComplexityResult,
    PowerAnalysisResult,
    AdversarialStrategy,
    InferenceGraph,
    Vote,
    Precedent,

    # Engine parameter types
    GeometricParams,
    ComplexityParams,
    DetectionParams,
    FederationParams,
    SimulationParams,

    # Proof types
    ProofObligation,

    # Helper functions
    compute_effective_rank,
    compute_required_sample_size,
    validate_literals_for_np_hardness,
)

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
