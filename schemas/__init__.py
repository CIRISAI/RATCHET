"""
RATCHET Schema Definitions

Type-safe Pydantic models for the RATCHET platform.
"""

from .simulation import (
    # Base type aliases
    Dimension,
    PositiveInt,
    NonNegativeInt,
    Radius,
    Correlation,
    Probability,
    NonNegativeFloat,
    Fraction,
    ByzantineFraction,
    # Enums
    SamplingMode,
    SATSolver,
    DeceptionStrategy,
    DetectionMethod,
    ConsensusProtocol,
    MaliciousStrategy,
    AdversarialAttackType,
    ProofStatus,
    # Parameter models
    AdversarialStrategy,
    GeometricParams,
    ComplexityParams,
    DetectionParams,
    FederationParams,
    SimulationParams,
    # Request/Response models
    AdversaryConfig,
    ProvenanceRecord,
    SimulationRequest,
    ConfidenceInterval,
    AdversarialMetrics,
    SimulationResult,
    AttackScenario,
    ProofObligation,
)

__all__ = [
    # Base type aliases
    "Dimension",
    "PositiveInt",
    "NonNegativeInt",
    "Radius",
    "Correlation",
    "Probability",
    "NonNegativeFloat",
    "Fraction",
    "ByzantineFraction",
    # Enums
    "SamplingMode",
    "SATSolver",
    "DeceptionStrategy",
    "DetectionMethod",
    "ConsensusProtocol",
    "MaliciousStrategy",
    "AdversarialAttackType",
    "ProofStatus",
    # Parameter models
    "AdversarialStrategy",
    "GeometricParams",
    "ComplexityParams",
    "DetectionParams",
    "FederationParams",
    "SimulationParams",
    # Request/Response models
    "AdversaryConfig",
    "ProvenanceRecord",
    "SimulationRequest",
    "ConfidenceInterval",
    "AdversarialMetrics",
    "SimulationResult",
    "AttackScenario",
    "ProofObligation",
]
