#!/usr/bin/env python3
"""
Exp 1b — Locked Measurement Pipeline (single source of truth)

ALL N_eff_H computations from this point forward should go through this
module. Re-implementing the trace → projection → eigenspectrum → N_eff_H
pipeline in scratch scripts is exactly how we spin and accumulate
inconsistent numbers across analyses. This module is the answer.

Public surface:

  - `load_chains_from_tee_dir(path)` — returns list[Chain] from a
    qa_runner local-tee directory (handles iter-prefixed batch files)
  - `Chain` — typed record with all 16 projection features + firing
    counts + DMA friction signals + cohort metadata
  - `compute_neff_h(chains, retention_threshold=1e-9)` — cohort N_eff_H
  - `bootstrap_neff_h(chains, n_resamples=10_000, seed=...)` — point
    estimate + 95% bootstrap CI
  - `sensitivity_sweep(chains, ...)` — varies thresholds/methods and
    reports stability

All thresholds and constants here MIRROR the lake's
`RATCHET.Experiments.FrictionDistribution` definitions. If the lake
changes those, this module must update in lockstep.

Determinism: bootstrap uses a fixed PRNG seed (0xC1715_E_EF) to ensure
reproducibility across re-runs on the same data.
"""

from __future__ import annotations

import glob
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.linalg import eigh


# ──────────────────────────────────────────────────────────────────────
# Locked constants (MIRROR `RATCHET.Experiments.FrictionDistribution`)
# ──────────────────────────────────────────────────────────────────────

# The 16-feature projection (crc-v1, locked at v0.1.0 calibration)
PROJECTION_16 = [
    "csdma_plausibility_score",
    "dsdma_domain_alignment",
    "coherence_level",
    "entropy_level",
    "idma_k_eff",
    "idma_correlation_risk",
    "entropy_score",
    "coherence_score",
    "optimization_veto_entropy_ratio",
    "epistemic_humility_certainty",
    "conscience_passed",
    "entropy_passed",
    "coherence_passed",
    "optimization_veto_passed",
    "epistemic_humility_passed",
    "action_was_overridden",
]

# The four conditional conscience-faculty fields (BO-1 source signal)
CONDITIONAL_FACULTY_FIELDS = [
    "entropy_score",
    "coherence_score",
    "optimization_veto_entropy_ratio",
    "epistemic_humility_certainty",
]

# DMA friction thresholds (mirror FrictionDistribution.lean)
DMA_CSDMA_FRICTION_BELOW   = 0.7   # implausibility friction
DMA_DSDMA_FRICTION_BELOW   = 0.7   # domain-misalignment friction
DMA_KEFF_FRICTION_BELOW    = 2.0   # rigidity friction
DMA_CORR_FRICTION_ABOVE    = 0.43  # near-critical CCA threshold

# Locked thresholds (defaults)
DEFAULT_K_HIGH = 3                  # FrictionDistribution.defaultHighFrictionThreshold
DEFAULT_DMA_N_HIGH = 2              # IsDmaFrictionActive: ≥ 2 of 4 signals
DEFAULT_RETENTION_THRESHOLD = 1e-9  # std below this → drop feature
DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
DEFAULT_RNG_SEED = 0xC1715_E_EF

# Trace extraction paths (wire-format 2.7.9)
PATHS = {
    "DMA_RESULTS": {
        "csdma_plausibility_score": ("csdma", "plausibility_score"),
        "dsdma_domain_alignment": ("dsdma", "domain_alignment"),
    },
    "IDMA_RESULT": {
        "idma_k_eff": ("k_eff",),
        "idma_correlation_risk": ("correlation_risk",),
    },
    "CONSCIENCE_RESULT": {
        "coherence_level": ("coherence_level",),
        "entropy_level": ("entropy_level",),
        "entropy_score": ("entropy_score",),
        "coherence_score": ("coherence_score",),
        "optimization_veto_entropy_ratio": ("optimization_veto_entropy_ratio",),
        "epistemic_humility_certainty": ("epistemic_humility_certainty",),
        "conscience_passed": ("conscience_passed",),
        "entropy_passed": ("entropy_passed",),
        "coherence_passed": ("coherence_passed",),
        "optimization_veto_passed": ("optimization_veto_passed",),
        "epistemic_humility_passed": ("epistemic_humility_passed",),
        "action_was_overridden": ("action_was_overridden",),
    },
}

REQUIRED_EVENT_TYPES = {
    "THOUGHT_START",
    "SNAPSHOT_AND_CONTEXT",
    "DMA_RESULTS",
    "IDMA_RESULT",
    "ASPDMA_RESULT",
    "CONSCIENCE_RESULT",
    "ACTION_RESULT",
}

CORE_FIELDS_REQUIRED = {
    "csdma_plausibility_score",
    "dsdma_domain_alignment",
    "coherence_level",
    "entropy_level",
    "idma_k_eff",
    "idma_correlation_risk",
    "conscience_passed",
    "action_was_overridden",
}


# ──────────────────────────────────────────────────────────────────────
# Data types
# ──────────────────────────────────────────────────────────────────────

@dataclass
class Chain:
    """Per-thought projection + derived friction signals."""

    trace_id: Optional[str]
    thought_id: Optional[str]
    source_batch: str  # filename
    features: dict[str, float]
    n_fired: int                 # firing count over 4 conditional conscience fields
    n_dma_friction: int          # count of DMA friction signals (0..4)
    dma_signals: dict[str, bool]  # per-signal flags

    @property
    def is_boundary_active(self) -> bool:
        """BO-1: at least one conscience faculty fired."""
        return self.n_fired >= 1

    def is_high_friction(self, k_high: int = DEFAULT_K_HIGH) -> bool:
        """FrictionDistribution.IsHighFriction at configurable threshold."""
        return self.n_fired >= k_high

    def is_dma_friction_active(self, n_high: int = DEFAULT_DMA_N_HIGH) -> bool:
        """FrictionDistribution.IsDmaFrictionActive at configurable threshold."""
        return self.n_dma_friction >= n_high

    def is_combined_friction(
        self, k_high: int = DEFAULT_K_HIGH, dma_n_high: int = DEFAULT_DMA_N_HIGH
    ) -> bool:
        """FrictionDistribution.IsCombinedFriction."""
        return self.is_high_friction(k_high) or self.is_dma_friction_active(dma_n_high)


# ──────────────────────────────────────────────────────────────────────
# Extraction
# ──────────────────────────────────────────────────────────────────────

def _get_nested(d, path):
    cur = d
    for p in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(p)
    return cur


def _cast_to_float(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
        return float(v)
    return None


def _compute_dma_friction(features: dict[str, float]) -> tuple[int, dict[str, bool]]:
    csdma = features.get("csdma_plausibility_score", 1.0)
    dsdma = features.get("dsdma_domain_alignment", 1.0)
    k_eff = features.get("idma_k_eff", 5.0)  # high default = no friction
    corr = features.get("idma_correlation_risk", 0.0)  # low default = no friction
    flags = {
        "csdma": csdma < DMA_CSDMA_FRICTION_BELOW,
        "dsdma": dsdma < DMA_DSDMA_FRICTION_BELOW,
        "k_eff": k_eff < DMA_KEFF_FRICTION_BELOW,
        "corr": corr > DMA_CORR_FRICTION_ABOVE,
    }
    return sum(flags.values()), flags


def load_chains_from_tee_dir(tee_dir: Path) -> tuple[list[Chain], int]:
    """Load all valid `complete_trace` events from a qa_runner local-tee dir.

    Returns (chains, excluded_count) where excluded_count includes traces
    rejected for missing required event types or core projection fields.
    """
    chains: list[Chain] = []
    excluded = 0
    tee_dir = Path(tee_dir)
    if not tee_dir.is_dir():
        return chains, excluded

    # Match raw `accord-batch-*.json` AND iter-prefixed `iter{N}_accord-batch-*.json`
    paths = sorted(tee_dir.glob("**/*accord-batch-*.json"))

    for batch_path in paths:
        try:
            batch = json.load(open(batch_path))
        except Exception:
            continue
        if batch.get("trace_level") != "detailed":
            continue
        for ev in batch.get("events", []) or []:
            if ev.get("event_type") != "complete_trace":
                continue
            trace = ev.get("trace") or {}
            event_types = {c.get("event_type") for c in trace.get("components", [])}
            if not REQUIRED_EVENT_TYPES.issubset(event_types):
                excluded += 1
                continue

            features: dict[str, float] = {}
            last_conscience = None
            for c in trace.get("components", []):
                et = c.get("event_type")
                data = c.get("data") or c.get("payload") or {}
                if et == "CONSCIENCE_RESULT":
                    last_conscience = data
                elif et in PATHS:
                    for fname, p in PATHS[et].items():
                        v = _cast_to_float(_get_nested(data, p))
                        if v is not None:
                            features[fname] = v
            if last_conscience is not None:
                for fname, p in PATHS["CONSCIENCE_RESULT"].items():
                    v = _cast_to_float(_get_nested(last_conscience, p))
                    if v is not None:
                        features[fname] = v

            missing_core = CORE_FIELDS_REQUIRED - features.keys()
            if missing_core:
                excluded += 1
                continue

            n_fired = sum(1 for f in CONDITIONAL_FACULTY_FIELDS if f in features)
            n_dma_friction, dma_flags = _compute_dma_friction(features)

            chains.append(Chain(
                trace_id=trace.get("trace_id"),
                thought_id=trace.get("thought_id"),
                source_batch=batch_path.name,
                features=features,
                n_fired=n_fired,
                n_dma_friction=n_dma_friction,
                dma_signals=dma_flags,
            ))

    return chains, excluded


# ──────────────────────────────────────────────────────────────────────
# N_eff_H computation (the load-bearing function)
# ──────────────────────────────────────────────────────────────────────

def _build_standardized_matrix(
    chains: list[Chain],
    retention_threshold: float = DEFAULT_RETENTION_THRESHOLD,
):
    """Return (M_std, retention_mask, col_means, col_stds) for the cohort.

    Standardization: per-feature mean imputation for NaN, then per-feature
    z-score. Features with std < retention_threshold dropped.
    """
    if len(chains) < 2:
        return None, None, None, None
    M = np.full((len(chains), len(PROJECTION_16)), np.nan)
    for i, c in enumerate(chains):
        for j, f in enumerate(PROJECTION_16):
            v = c.features.get(f)
            if v is not None:
                M[i, j] = v
    col_means = np.nanmean(M, axis=0)
    M_imp = np.where(np.isnan(M), col_means, M)
    col_stds = M_imp.std(axis=0, ddof=0)
    retention = col_stds > retention_threshold
    if retention.sum() < 2:
        return None, retention, col_means, col_stds
    safe_stds = np.where(retention, col_stds, 1.0)
    M_std = (M_imp - M_imp.mean(axis=0)) / safe_stds
    M_std = M_std[:, retention]
    return M_std, retention, col_means, col_stds


def compute_eigenspectrum(
    chains: list[Chain],
    retention_threshold: float = DEFAULT_RETENTION_THRESHOLD,
) -> Optional[np.ndarray]:
    """Eigenvalues of the standardized 16-feature correlation matrix."""
    M_std, *_ = _build_standardized_matrix(chains, retention_threshold)
    if M_std is None or M_std.shape[0] < 2 or M_std.shape[1] < 2:
        return None
    C = np.corrcoef(M_std, rowvar=False)
    lambdas = np.maximum(eigh(C, eigvals_only=True)[::-1], 0)
    return lambdas


def neff_h_from_lambdas(lambdas: np.ndarray) -> float:
    total = lambdas.sum()
    if total <= 0:
        return float("nan")
    p = lambdas / total
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def neff_pr_from_lambdas(lambdas: np.ndarray) -> float:
    s = lambdas.sum()
    s2 = (lambdas ** 2).sum()
    if s2 <= 0:
        return float("nan")
    return float((s * s) / s2)


def compute_neff_h(
    chains: list[Chain],
    retention_threshold: float = DEFAULT_RETENTION_THRESHOLD,
) -> tuple[float, int]:
    """Returns (N_eff_H, retained_dim_count) for a cohort."""
    lambdas = compute_eigenspectrum(chains, retention_threshold)
    if lambdas is None:
        return float("nan"), 0
    return neff_h_from_lambdas(lambdas), len(lambdas)


# ──────────────────────────────────────────────────────────────────────
# Bootstrap CI
# ──────────────────────────────────────────────────────────────────────

def bootstrap_neff_h(
    chains: list[Chain],
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_RNG_SEED,
    retention_threshold: float = DEFAULT_RETENTION_THRESHOLD,
) -> dict:
    """Bootstrap percentile CI on cohort N_eff_H."""
    n = len(chains)
    if n < 2:
        return {"mean": float("nan"), "ci95_low": float("nan"),
                "ci95_high": float("nan"), "n_valid": 0, "n_resamples": 0}
    point, _ = compute_neff_h(chains, retention_threshold)
    rng = np.random.default_rng(seed)
    samples = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        resample = [chains[j] for j in idx]
        nh, _ = compute_neff_h(resample, retention_threshold)
        samples[i] = nh
    valid = samples[~np.isnan(samples)]
    if len(valid) < n_resamples // 10:
        return {"point": point, "mean": float("nan"),
                "ci95_low": float("nan"), "ci95_high": float("nan"),
                "n_valid": len(valid), "n_resamples": n_resamples}
    return {
        "point": point,
        "mean": float(valid.mean()),
        "ci95_low": float(np.percentile(valid, 2.5)),
        "ci95_high": float(np.percentile(valid, 97.5)),
        "n_valid": int(len(valid)),
        "n_resamples": n_resamples,
    }


# ──────────────────────────────────────────────────────────────────────
# Sensitivity sweep
# ──────────────────────────────────────────────────────────────────────

def sensitivity_sweep(chains: list[Chain]) -> dict:
    """Vary subset definitions and report stability of N_eff_H."""
    results = []
    # All chains
    nh, dim = compute_neff_h(chains)
    results.append({"subset": "all_chains", "n": len(chains),
                    "neff_h": nh, "retained_dim": dim})
    # Conscience thresholds
    for k in [1, 2, 3, 4]:
        subset = [c for c in chains if c.n_fired >= k]
        nh, dim = compute_neff_h(subset)
        results.append({"subset": f"conscience_N>={k}", "n": len(subset),
                        "neff_h": nh, "retained_dim": dim})
    for k in [3, 4]:
        subset = [c for c in chains if c.n_fired == k]
        nh, dim = compute_neff_h(subset)
        results.append({"subset": f"conscience_N=={k}", "n": len(subset),
                        "neff_h": nh, "retained_dim": dim})
    # DMA thresholds
    for k in [1, 2, 3, 4]:
        subset = [c for c in chains if c.n_dma_friction >= k]
        nh, dim = compute_neff_h(subset)
        results.append({"subset": f"dma_n>={k}", "n": len(subset),
                        "neff_h": nh, "retained_dim": dim})
    # Combined
    subset = [c for c in chains if c.is_combined_friction()]
    nh, dim = compute_neff_h(subset)
    results.append({"subset": "combined_friction", "n": len(subset),
                    "neff_h": nh, "retained_dim": dim})
    # Retention threshold sensitivity (on the locked N>=3 subset)
    n3 = [c for c in chains if c.n_fired >= 3]
    for rt in [1e-12, 1e-9, 1e-6, 1e-3, 1e-1]:
        nh, dim = compute_neff_h(n3, retention_threshold=rt)
        results.append({"subset": f"conscience_N>=3_rt={rt:.0e}",
                        "n": len(n3), "neff_h": nh, "retained_dim": dim})
    return {"sensitivity": results}


# ──────────────────────────────────────────────────────────────────────
# Convenience
# ──────────────────────────────────────────────────────────────────────

def firing_distribution(chains: list[Chain]) -> dict[int, int]:
    out = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    for c in chains:
        out[c.n_fired] += 1
    return out
