"""
RATCHET Engines Module

Core computational engines for the RATCHET simulation framework.

Engines:
- GeometricEngine: Monte Carlo volume estimation for deceptive region analysis
- DetectionEngine: Statistical deception detection
- ComplexityEngine: SAT-based deception complexity measurement
- FederationEngine: PBFT consensus for distributed precedent accumulation
"""

from .detection import DetectionEngine
from .geometric import GeometricEngine, create_geometric_engine
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

__all__ = [
    # Detection
    "DetectionEngine",
    # Geometric
    "GeometricEngine",
    "create_geometric_engine",
    # Complexity
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
    # Federation
    "FederationEngine",
    "FederationNode",
    "FederationMetrics",
    "NodeType",
    "ConsensusRound",
    "BehavioralCorrelationDetector",
    "MIThresholdGate",
    "create_federation",
]
