"""
RATCHET Detection Module

Provides detection capabilities including:
- Statistical deception detection (Detection Engine)
- Compositional detection (partial - NEW-04 proven impossible for complete detection)
"""

from .compositional import (
    CompositionalDetector,
    MIAmplificationDetector,
    InferenceChainAnalyzer,
    InformationFlowMonitor,
    CompositionalDetectionResult,
)

__all__ = [
    'CompositionalDetector',
    'MIAmplificationDetector',
    'InferenceChainAnalyzer',
    'InformationFlowMonitor',
    'CompositionalDetectionResult',
]
