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

# Microbiome ecology engine (biology domain)
try:
    from .microbiome import (
        MicrobiomeEngine,
        MicrobiomeParams,
        MicrobiomeState,
        MicrobiomeShock,
        MicrobiomeIntervention,
        ShockType,
        InterventionType,
        create_microbiome_engine,
    )
    __all__.extend([
        "MicrobiomeEngine",
        "MicrobiomeParams",
        "MicrobiomeState",
        "MicrobiomeShock",
        "MicrobiomeIntervention",
        "ShockType",
        "InterventionType",
        "create_microbiome_engine",
    ])
except ImportError:
    pass

# Battery degradation engine (chemistry domain)
try:
    from .battery import (
        BatteryDegradationEngine,
        BatteryParams,
        CellState,
        BatteryShock,
        BatteryIntervention,
        BatteryShockType,
        BatteryInterventionType,
        create_battery_engine,
    )
    __all__.extend([
        "BatteryDegradationEngine",
        "BatteryParams",
        "CellState",
        "BatteryShock",
        "BatteryIntervention",
        "BatteryShockType",
        "BatteryInterventionType",
        "create_battery_engine",
    ])
except ImportError:
    pass

# Institutional collapse engine (history domain)
try:
    from .institutional import (
        InstitutionalCollapseEngine,
        InstitutionalParams,
        InstitutionalState,
        InstitutionalShock,
        InstitutionalIntervention,
        InstitutionalShockType,
        InstitutionalInterventionType,
        RegimeType,
        REGIME_ARCHETYPES,
        create_institutional_engine,
    )
    __all__.extend([
        "InstitutionalCollapseEngine",
        "InstitutionalParams",
        "InstitutionalState",
        "InstitutionalShock",
        "InstitutionalIntervention",
        "InstitutionalShockType",
        "InstitutionalInterventionType",
        "RegimeType",
        "REGIME_ARCHETYPES",
        "create_institutional_engine",
    ])
except ImportError:
    pass

# Extended correlation model (addresses reviewer concern 2.3)
try:
    from .correlation_tensor import (
        ExtendedCorrelationModel,
        SpectralProperties,
        HigherOrderCorrelations,
        TimeLagCorrelations,
        CorrelationStructure,
        compute_extended_correlation,
        compute_correlation_matrix,
        compute_spectral_properties,
        compute_triplet_correlations,
        compute_time_lag_correlations,
        scalar_to_extended,
        compare_correlation_models,
    )
    __all__.extend([
        "ExtendedCorrelationModel",
        "SpectralProperties",
        "HigherOrderCorrelations",
        "TimeLagCorrelations",
        "CorrelationStructure",
        "compute_extended_correlation",
        "compute_correlation_matrix",
        "compute_spectral_properties",
        "compute_triplet_correlations",
        "compute_time_lag_correlations",
        "scalar_to_extended",
        "compare_correlation_models",
    ])
except ImportError:
    pass

# Robustness analysis (addresses reviewer concern 2.1)
try:
    from .robustness import (
        GeometricRobustnessAnalyzer,
        RobustnessReport,
        SensitivityReport,
        VolumeEstimate,
        ClusterSpec,
        NonConvexSpec,
        GeometryType,
        quick_robustness_check,
        compute_breakdown_threshold,
        integrate_with_correlation_tensor,
    )
    __all__.extend([
        "GeometricRobustnessAnalyzer",
        "RobustnessReport",
        "SensitivityReport",
        "VolumeEstimate",
        "ClusterSpec",
        "NonConvexSpec",
        "GeometryType",
        "quick_robustness_check",
        "compute_breakdown_threshold",
        "integrate_with_correlation_tensor",
    ])
except ImportError:
    pass

# Intervention dynamics (addresses reviewer concern 4.2)
try:
    from .interventions import (
        InterventionDynamicsEngine,
        InterventionType,
        Intervention,
        InterventionEffect,
        InterventionOutcome,
        AdversaryModel,
        AdversaryResponse,
        EquilibriumOutcome,
        SystemState,
        analyze_intervention_options,
        simulate_with_adversary,
        compute_pareto_frontier,
        DEFAULT_CROSS_EFFECTS,
    )
    __all__.extend([
        "InterventionDynamicsEngine",
        "InterventionType",
        "Intervention",
        "InterventionEffect",
        "InterventionOutcome",
        "AdversaryModel",
        "AdversaryResponse",
        "EquilibriumOutcome",
        "SystemState",
        "analyze_intervention_options",
        "simulate_with_adversary",
        "compute_pareto_frontier",
        "DEFAULT_CROSS_EFFECTS",
    ])
except ImportError:
    pass
