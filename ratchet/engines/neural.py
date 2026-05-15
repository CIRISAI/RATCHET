"""
RATCHET Neural-Firing (Allen Brain Observatory) Substrate Engine — STUB

Domain mapping (per REGIME.md §"A1 — Allen neural firing"):
    k (constraints):    Number of simultaneously-recorded neurons per session
    rho (correlation):  Mean pairwise spike-train correlation (1-ms bins)
    sigma (sustain.):   Population-decoding accuracy for drifting-grating stimulus
                        (cross-validated linear classifier)

Constituent agency: A1 (low). Neurons have homeostatic / signaling drives
but no goal-directed behavior. Predicted residual structure: mostly noise
with weak structure from stimulus-driven functional connectivity.

Status: SKELETON. Population-coding theory aligns the prediction with
Averbeck/Latham/Pouget (NRN 2006) but the harness is unimplemented.

Pairs with: ratchet/engines/{battery,institutional,microbiome}.py
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class NeuralSession:
    """Per-recording-session (k, rho, sigma) triple."""
    session_id: str
    k: int          # number of recorded neurons
    rho: float      # mean pairwise spike-train correlation
    sigma: float    # decoding accuracy (cross-validated, drifting gratings)
    visual_area: Optional[str] = None  # for sub-class analysis


def compute_pairwise_spike_correlation(spike_times_by_neuron, bin_ms: float = 1.0) -> float:
    """Compute mean pairwise spike-train correlation across all neuron pairs.

    Binned at bin_ms (default 1 ms). Pearson correlation per pair, then
    mean of absolute values.

    TODO: implement against AllenSDK NWB output.
    """
    raise NotImplementedError("Spike-train correlation reducer — TODO")


def decode_drifting_gratings(spike_counts, stimulus_labels, n_folds: int = 5) -> float:
    """Cross-validated linear classifier on spike-count vectors → stimulus identity.

    Returns top-1 accuracy averaged across folds. This is the sigma metric.

    TODO: implement (sklearn.LinearDiscriminantAnalysis fits the bill).
    """
    raise NotImplementedError("Population decoder — TODO")


def load_allen_sessions(vendored_dir) -> list[NeuralSession]:
    """Load the ~80 Allen Visual Coding Neuropixels sessions.

    TODO: implement via AllenSDK ecephys cache.
    """
    raise NotImplementedError("Allen SDK session loader — TODO")


def fit_kish_neural(sessions: list[NeuralSession]):
    """Kish-fit regression of sigma on k_eff = k/(1 + rho(k-1)).

    Returns (r_squared, ci_lo, ci_hi).

    TODO: implement uniform with battery.py.
    """
    raise NotImplementedError("Kish fit + bootstrap CI — TODO")


def residual_whiteness(sessions, fit) -> float:
    """Predicted at A1: high p-value but slightly below A0 substrates."""
    raise NotImplementedError("Residual whiteness test — TODO")
