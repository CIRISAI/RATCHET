"""
RATCHET Engines Module

Core computational engines for the RATCHET simulation framework.

Engines:
- DetectionEngine: Statistical deception detection (LRT, Mahalanobis, power analysis)
- GeometricEngine: Monte Carlo volume estimation for deceptive region analysis
- ComplexityEngine: SAT-based deception complexity measurement
- FederationEngine: PBFT consensus for distributed precedent accumulation
"""

# Detection engine is always available
from .detection import (
    DetectionEngine,
    LRTResult,
    DistributionParams,
    mahalanobis_distance,
    power_analysis,
    compute_required_sample_size,
)

__all__ = [
    "DetectionEngine",
    "LRTResult",
    "DistributionParams",
    "mahalanobis_distance",
    "power_analysis",
    "compute_required_sample_size",
]

# Optional engines - may not exist yet
try:
    from .geometric import GeometricEngine, create_geometric_engine
    __all__.extend([
        "GeometricEngine",
        "create_geometric_engine",
    ])
except ImportError:
    pass

try:
    from .complexity import (
        ComplexityEngine,
        ComplexityEngineError,
        NPHardnessViolation,
        SecurityThresholdViolation,
        SolverUnavailableError,
        InconsistentDeceptionError,
        Clause,
        SATInstance,
        TimingResult,
        generate_random_clause,
        generate_random_sat_instance,
        generate_satisfiable_instance,
        generate_honest_instance,
        generate_deceptive_instance,
        solve_sat,
        measure_complexity,
    )
    __all__.extend([
        "ComplexityEngine",
        "ComplexityEngineError",
        "NPHardnessViolation",
        "SecurityThresholdViolation",
        "SolverUnavailableError",
        "InconsistentDeceptionError",
        "Clause",
        "SATInstance",
        "TimingResult",
        "generate_random_clause",
        "generate_random_sat_instance",
        "generate_satisfiable_instance",
        "generate_honest_instance",
        "generate_deceptive_instance",
        "solve_sat",
        "measure_complexity",
    ])
except ImportError:
    pass

try:
    from .federation import (
        FederationEngine,
        FederationNode,
        FederationMetrics,
        NodeType,
        ConsensusRound,
        BehavioralCorrelationDetector,
        MIThresholdGate,
        create_federation,
    )
    __all__.extend([
        "FederationEngine",
        "FederationNode",
        "FederationMetrics",
        "NodeType",
        "ConsensusRound",
        "BehavioralCorrelationDetector",
        "MIThresholdGate",
        "create_federation",
    ])
except ImportError:
    pass
