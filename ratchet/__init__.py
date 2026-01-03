"""
RATCHET: Resilient Adversarial Testing for CIRIS Heterogeneous Evaluation Testbed

A simulation framework for testing the Federated Ratchet detection mechanism,
implementing information-theoretic Sybil resistance and Bayesian anomaly detection.

Core Components:
- engines: Computational engines (detection, geometric, complexity, federation)
- information_theory: MI calculation, KL divergence, likelihood ratios
- federation_graph: Network model with weighted partnership edges
- orthogonality_gate: Constraint-based partner filtering
- anomaly_detection: Bayesian inference on agent traces
- sustainability: sigma(t) dynamics and partnership lifecycle
- simulation: Monte Carlo harness for detection power analysis

Mathematical Foundation:
- Mutual Information: I(X;Y) = H(X) + H(Y) - H(X,Y)
- Orthogonality Gate: Reject if I(C_i; C_j) > theta
- J Function: J = k * (1 - rho_bar) * lambda * sigma
- Sustainability: sigma(t+dt) = sigma(t)*(1 - d*dt) + w*Signal(t)
"""

__version__ = "0.1.0"
__author__ = "RATCHET Framework"

# Core engines (always available)
from .engines import DetectionEngine

__all__ = [
    "DetectionEngine",
]

# Optional imports - these modules may not exist yet
try:
    from .information_theory import (
        mutual_information,
        kl_divergence,
        jensen_shannon_divergence,
        likelihood_ratio,
        entropy,
    )
    __all__.extend([
        "mutual_information",
        "kl_divergence",
        "jensen_shannon_divergence",
        "likelihood_ratio",
        "entropy",
    ])
except ImportError:
    pass

try:
    from .federation_graph import (
        FederationGraph,
        Agent,
        Partnership,
    )
    __all__.extend([
        "FederationGraph",
        "Agent",
        "Partnership",
    ])
except ImportError:
    pass

try:
    from .orthogonality_gate import (
        OrthogonalityGate,
        ConstraintDistribution,
    )
    __all__.extend([
        "OrthogonalityGate",
        "ConstraintDistribution",
    ])
except ImportError:
    pass

try:
    from .anomaly_detection import (
        BayesianDetector,
        TraceAnalyzer,
        LikelihoodRatioTest,
    )
    __all__.extend([
        "BayesianDetector",
        "TraceAnalyzer",
        "LikelihoodRatioTest",
    ])
except ImportError:
    pass

try:
    from .sustainability import (
        SustainabilityDynamics,
        SignalProcessor,
    )
    __all__.extend([
        "SustainabilityDynamics",
        "SignalProcessor",
    ])
except ImportError:
    pass

try:
    from .simulation import (
        RatchetSimulation,
        ExperimentConfig,
        DetectionPowerAnalysis,
    )
    __all__.extend([
        "RatchetSimulation",
        "ExperimentConfig",
        "DetectionPowerAnalysis",
    ])
except ImportError:
    pass
