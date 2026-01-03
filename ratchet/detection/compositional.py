"""
RATCHET Compositional Detection Module

Provides PARTIAL detection of emergent deception from honest components.

IMPORTANT: Complete compositional detection is PROVABLY IMPOSSIBLE (NEW-04).
See KNOWN_LIMITATIONS.md L-01 for formal justification:
1. Reduction from one-way functions (would break crypto commitments)
2. Information-theoretic barrier (marginal-preserving deception)
3. Rice's theorem analog (semantic property of compositions)

This module implements what IS possible:
- MI amplification detection (system MI >> component MI sum)
- Inference chain bias analysis
- Information flow anomaly detection

These provide probabilistic detection of SOME emergent deception patterns,
but cannot guarantee detection of all deceptive compositions.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Callable, Any
from enum import Enum
import warnings


class DetectionConfidence(Enum):
    """Confidence level of detection result."""
    HIGH = "high"           # Strong statistical signal
    MEDIUM = "medium"       # Moderate signal, may be noise
    LOW = "low"             # Weak signal, likely noise
    UNDETECTABLE = "undetectable"  # Marginal-preserving, fundamentally undetectable


@dataclass
class ComponentTrace:
    """Trace of a single component's behavior."""
    component_id: str
    inputs: List[np.ndarray]
    outputs: List[np.ndarray]
    timestamps: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def mutual_information_with(self, other: 'ComponentTrace') -> float:
        """Estimate MI between this component and another."""
        # Use histogram-based MI estimation
        if len(self.outputs) == 0 or len(other.outputs) == 0:
            return 0.0

        # Flatten and discretize for MI estimation
        x = np.concatenate([o.flatten() for o in self.outputs])
        y = np.concatenate([o.flatten() for o in other.outputs])

        # Ensure same length
        min_len = min(len(x), len(y))
        x, y = x[:min_len], y[:min_len]

        return _estimate_mutual_information(x, y)


@dataclass
class SystemTrace:
    """Trace of the entire system's behavior."""
    component_traces: Dict[str, ComponentTrace]
    system_outputs: List[np.ndarray]
    ground_truth: Optional[List[np.ndarray]] = None

    def total_component_mi(self) -> float:
        """Sum of pairwise MI between all components."""
        total = 0.0
        components = list(self.component_traces.values())
        for i, c1 in enumerate(components):
            for c2 in components[i+1:]:
                total += c1.mutual_information_with(c2)
        return total

    def system_output_mi(self) -> float:
        """MI between system output and concatenated component outputs."""
        if len(self.system_outputs) == 0:
            return 0.0

        system_flat = np.concatenate([o.flatten() for o in self.system_outputs])
        component_flat = []
        for ct in self.component_traces.values():
            if ct.outputs:
                component_flat.extend([o.flatten() for o in ct.outputs])

        if not component_flat:
            return 0.0

        component_concat = np.concatenate(component_flat)
        min_len = min(len(system_flat), len(component_concat))
        return _estimate_mutual_information(
            system_flat[:min_len],
            component_concat[:min_len]
        )


@dataclass
class CompositionalDetectionResult:
    """Result of compositional detection analysis."""
    is_suspicious: bool
    confidence: DetectionConfidence
    mi_amplification_ratio: float
    inference_chain_bias: float
    flow_anomaly_score: float
    component_anomalies: Dict[str, float]
    explanation: str
    limitations: str = (
        "WARNING: Complete compositional detection is provably impossible. "
        "This result may miss emergent deception that preserves marginal distributions. "
        "See KNOWN_LIMITATIONS.md L-01."
    )


class MIAmplificationDetector:
    """
    Detects emergent deception via mutual information amplification.

    If system MI >> sum of component MI, this indicates information
    is being created/correlated at the system level that wasn't present
    in individual components.

    Theoretical basis: Emergent deception requires coordination that
    increases mutual information between system output and some target.
    """

    def __init__(
        self,
        amplification_threshold: float = 2.0,
        min_samples: int = 100
    ):
        """
        Args:
            amplification_threshold: Flag if system_MI > threshold * component_MI
            min_samples: Minimum samples for reliable MI estimation
        """
        self.amplification_threshold = amplification_threshold
        self.min_samples = min_samples

    def detect(self, trace: SystemTrace) -> Tuple[bool, float, DetectionConfidence]:
        """
        Detect MI amplification in system trace.

        Returns:
            (is_suspicious, amplification_ratio, confidence)
        """
        component_mi = trace.total_component_mi()
        system_mi = trace.system_output_mi()

        # Avoid division by zero
        if component_mi < 1e-10:
            if system_mi > 1e-10:
                # System has MI but components don't - very suspicious
                return True, float('inf'), DetectionConfidence.HIGH
            else:
                return False, 1.0, DetectionConfidence.LOW

        ratio = system_mi / component_mi

        # Determine confidence based on sample size
        total_samples = sum(
            len(ct.outputs) for ct in trace.component_traces.values()
        )
        if total_samples < self.min_samples:
            confidence = DetectionConfidence.LOW
        elif total_samples < self.min_samples * 5:
            confidence = DetectionConfidence.MEDIUM
        else:
            confidence = DetectionConfidence.HIGH

        is_suspicious = ratio > self.amplification_threshold
        return is_suspicious, ratio, confidence


class InferenceChainAnalyzer:
    """
    Analyzes inference chains for systematic bias.

    Detects when a sequence of inferences drifts systematically
    in a direction that benefits deception, even if each individual
    inference appears honest.
    """

    def __init__(
        self,
        drift_threshold: float = 0.3,
        window_size: int = 10
    ):
        """
        Args:
            drift_threshold: Flag if cumulative drift > threshold * sqrt(n)
            window_size: Size of sliding window for drift detection
        """
        self.drift_threshold = drift_threshold
        self.window_size = window_size

    def analyze_chain(
        self,
        inferences: List[np.ndarray],
        expected_mean: Optional[np.ndarray] = None
    ) -> Tuple[float, bool, List[int]]:
        """
        Analyze inference chain for systematic bias.

        Args:
            inferences: List of inference outputs
            expected_mean: Expected mean of unbiased inferences

        Returns:
            (bias_score, is_biased, suspicious_indices)
        """
        if len(inferences) < 2:
            return 0.0, False, []

        # Flatten inferences
        flat = [inf.flatten() for inf in inferences]

        # Compute expected mean if not provided
        if expected_mean is None:
            expected_mean = np.mean(flat, axis=0)

        # Compute cumulative drift
        drifts = []
        cumulative = np.zeros_like(expected_mean)
        for inf in flat:
            cumulative += (inf - expected_mean)
            drifts.append(np.linalg.norm(cumulative))

        # Expected drift under random walk: O(sqrt(n))
        n = len(inferences)
        expected_drift = np.sqrt(n) * np.std([np.linalg.norm(f - expected_mean) for f in flat])

        if expected_drift < 1e-10:
            return 0.0, False, []

        # Bias score: actual drift / expected drift
        actual_drift = drifts[-1]
        bias_score = actual_drift / expected_drift

        # Find suspicious windows
        suspicious = []
        for i in range(len(drifts) - self.window_size):
            window_drift = drifts[i + self.window_size] - drifts[i]
            window_expected = np.sqrt(self.window_size) * expected_drift / np.sqrt(n)
            if window_drift > self.drift_threshold * window_expected * 3:
                suspicious.extend(range(i, i + self.window_size))

        suspicious = sorted(set(suspicious))
        is_biased = bias_score > self.drift_threshold * np.sqrt(n)

        return bias_score, is_biased, suspicious


class InformationFlowMonitor:
    """
    Monitors information flow patterns for anomalies.

    Uses graph-based analysis to detect unusual flow patterns
    that could indicate coordinated deception.
    """

    def __init__(
        self,
        anomaly_threshold: float = 2.0,
        min_flow_strength: float = 0.1
    ):
        """
        Args:
            anomaly_threshold: Standard deviations for anomaly detection
            min_flow_strength: Minimum MI for edge to be considered
        """
        self.anomaly_threshold = anomaly_threshold
        self.min_flow_strength = min_flow_strength

    def build_flow_graph(
        self,
        trace: SystemTrace
    ) -> Dict[Tuple[str, str], float]:
        """
        Build information flow graph from component traces.

        Returns:
            Dict mapping (source, target) to flow strength (MI)
        """
        flows = {}
        components = list(trace.component_traces.items())

        for i, (id1, c1) in enumerate(components):
            for id2, c2 in components[i+1:]:
                mi = c1.mutual_information_with(c2)
                if mi >= self.min_flow_strength:
                    flows[(id1, id2)] = mi
                    flows[(id2, id1)] = mi

        return flows

    def detect_anomalies(
        self,
        current_flows: Dict[Tuple[str, str], float],
        baseline_flows: Optional[Dict[Tuple[str, str], float]] = None
    ) -> Tuple[float, List[Tuple[str, str]]]:
        """
        Detect anomalous flow patterns.

        Args:
            current_flows: Current flow graph
            baseline_flows: Expected baseline flows (if known)

        Returns:
            (anomaly_score, anomalous_edges)
        """
        if not current_flows:
            return 0.0, []

        flow_values = list(current_flows.values())
        mean_flow = np.mean(flow_values)
        std_flow = np.std(flow_values)

        if std_flow < 1e-10:
            return 0.0, []

        anomalous = []
        max_deviation = 0.0

        for edge, flow in current_flows.items():
            # Compare to baseline if available
            if baseline_flows and edge in baseline_flows:
                expected = baseline_flows[edge]
                deviation = abs(flow - expected) / (std_flow + 1e-10)
            else:
                deviation = abs(flow - mean_flow) / std_flow

            if deviation > self.anomaly_threshold:
                anomalous.append(edge)
                max_deviation = max(max_deviation, deviation)

        return max_deviation, anomalous


class CompositionalDetector:
    """
    Combined compositional detection using all available methods.

    IMPORTANT: This provides PARTIAL detection only.
    Complete detection is provably impossible (NEW-04).
    """

    def __init__(
        self,
        mi_amplification_threshold: float = 2.0,
        drift_threshold: float = 0.3,
        flow_anomaly_threshold: float = 2.0
    ):
        self.mi_detector = MIAmplificationDetector(mi_amplification_threshold)
        self.chain_analyzer = InferenceChainAnalyzer(drift_threshold)
        self.flow_monitor = InformationFlowMonitor(flow_anomaly_threshold)

    def detect(
        self,
        trace: SystemTrace,
        inference_chains: Optional[List[List[np.ndarray]]] = None,
        baseline_flows: Optional[Dict[Tuple[str, str], float]] = None
    ) -> CompositionalDetectionResult:
        """
        Run all compositional detection methods.

        Args:
            trace: System and component traces
            inference_chains: Optional list of inference chains to analyze
            baseline_flows: Optional baseline flow graph for comparison

        Returns:
            CompositionalDetectionResult with all findings
        """
        # MI Amplification
        mi_suspicious, mi_ratio, mi_confidence = self.mi_detector.detect(trace)

        # Inference Chain Analysis
        chain_bias = 0.0
        if inference_chains:
            biases = []
            for chain in inference_chains:
                bias, _, _ = self.chain_analyzer.analyze_chain(chain)
                biases.append(bias)
            chain_bias = max(biases) if biases else 0.0

        # Flow Analysis
        current_flows = self.flow_monitor.build_flow_graph(trace)
        flow_score, anomalous_edges = self.flow_monitor.detect_anomalies(
            current_flows, baseline_flows
        )

        # Component-level anomalies
        component_anomalies = {}
        for comp_id, comp_trace in trace.component_traces.items():
            # Simple anomaly: output variance vs input variance ratio
            if comp_trace.inputs and comp_trace.outputs:
                input_var = np.var([i.flatten() for i in comp_trace.inputs])
                output_var = np.var([o.flatten() for o in comp_trace.outputs])
                if input_var > 1e-10:
                    component_anomalies[comp_id] = output_var / input_var

        # Aggregate decision
        is_suspicious = mi_suspicious or chain_bias > 1.0 or flow_score > 2.0

        # Determine overall confidence
        if mi_confidence == DetectionConfidence.HIGH and mi_suspicious:
            confidence = DetectionConfidence.HIGH
        elif mi_suspicious or chain_bias > 0.5:
            confidence = DetectionConfidence.MEDIUM
        else:
            confidence = DetectionConfidence.LOW

        # Generate explanation
        explanations = []
        if mi_suspicious:
            explanations.append(
                f"MI amplification detected: ratio={mi_ratio:.2f} "
                f"(threshold={self.mi_detector.amplification_threshold})"
            )
        if chain_bias > 0.5:
            explanations.append(
                f"Inference chain bias detected: score={chain_bias:.2f}"
            )
        if flow_score > 1.0:
            explanations.append(
                f"Flow anomalies detected: score={flow_score:.2f}, "
                f"anomalous edges={len(anomalous_edges)}"
            )
        if not explanations:
            explanations.append("No significant compositional deception signals detected.")

        return CompositionalDetectionResult(
            is_suspicious=is_suspicious,
            confidence=confidence,
            mi_amplification_ratio=mi_ratio,
            inference_chain_bias=chain_bias,
            flow_anomaly_score=flow_score,
            component_anomalies=component_anomalies,
            explanation=" | ".join(explanations)
        )


def _estimate_mutual_information(x: np.ndarray, y: np.ndarray, bins: int = 20) -> float:
    """
    Estimate mutual information using histogram method.

    This is a simple estimator; more sophisticated methods (KSG, etc.)
    could be used for better accuracy.
    """
    if len(x) != len(y) or len(x) == 0:
        return 0.0

    # Discretize
    x_bins = np.minimum(np.digitize(x, np.linspace(x.min(), x.max(), bins)), bins - 1)
    y_bins = np.minimum(np.digitize(y, np.linspace(y.min(), y.max(), bins)), bins - 1)

    # Joint histogram
    joint_hist = np.zeros((bins, bins))
    for xi, yi in zip(x_bins, y_bins):
        joint_hist[xi, yi] += 1
    joint_hist /= len(x)

    # Marginals
    px = joint_hist.sum(axis=1)
    py = joint_hist.sum(axis=0)

    # MI = sum p(x,y) * log(p(x,y) / (p(x) * p(y)))
    mi = 0.0
    for i in range(bins):
        for j in range(bins):
            if joint_hist[i, j] > 1e-10 and px[i] > 1e-10 and py[j] > 1e-10:
                mi += joint_hist[i, j] * np.log(
                    joint_hist[i, j] / (px[i] * py[j])
                )

    return max(0.0, mi)  # MI is non-negative
