"""
RATCHET: Resilient Adversarial Testing for CIRIS Heterogeneous Evaluation Testbed

A simulation framework for testing the Federated Ratchet detection mechanism,
implementing information-theoretic Sybil resistance and Bayesian anomaly detection.

Core Components:
- information_theory: MI calculation, KL divergence, likelihood ratios
- federation_graph: Network model with weighted partnership edges
- orthogonality_gate: Constraint-based partner filtering
- anomaly_detection: Bayesian inference on agent traces
- sustainability: σ(t) dynamics and partnership lifecycle
- simulation: Monte Carlo harness for detection power analysis

Mathematical Foundation:
- Mutual Information: I(X;Y) = H(X) + H(Y) - H(X,Y)
- Orthogonality Gate: Reject if I(C_i; C_j) > θ
- J Function: J = k · (1 - ρ̄) · λ · σ
- Sustainability: σ(t+Δt) = σ(t)·(1 - d·Δt) + w·Signal(t)
"""

__version__ = "0.1.0"
__author__ = "RATCHET Framework"

from .information_theory import (
    mutual_information,
    kl_divergence,
    jensen_shannon_divergence,
    likelihood_ratio,
    entropy,
)

from .federation_graph import (
    FederationGraph,
    Agent,
    Partnership,
)

from .orthogonality_gate import (
    OrthogonalityGate,
    ConstraintDistribution,
)

from .anomaly_detection import (
    BayesianDetector,
    TraceAnalyzer,
    LikelihoodRatioTest,
)

from .sustainability import (
    SustainabilityDynamics,
    SignalProcessor,
)

from .simulation import (
    RatchetSimulation,
    ExperimentConfig,
    DetectionPowerAnalysis,
)
