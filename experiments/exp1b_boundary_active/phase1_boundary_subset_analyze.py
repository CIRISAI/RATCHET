#!/usr/bin/env python3
"""
Exp 1b Stage 1a — Boundary-Active Subset Re-Analysis

Reads existing Phase 1 trace artifacts and filters per-chain to ONLY
boundary-active chains (per `RATCHET.Experiments.BoundaryObservability`
BO-1: at least one of the four LLM-based conscience faculties fired).

For each model, recomputes:
  - Boundary-active fraction (n_active / n_total)
  - Cohort N_eff_H over the boundary-active subset (vs the full corpus)
  - 95% bootstrap CI on the subset mean

EXPLORATORY ONLY — does NOT apply the F-6 decision rule. Per Exp 1
PRE_REGISTRATION.md §10.1, headline INDETERMINATE stands until Phase 1b
re-pre-registration. This script recovers signal informally.

Usage:
  python3 phase1_boundary_subset_analyze.py \\
      --input-dir /tmp/exp1_p1                    \\
      --out-dir   experiments/exp1b_boundary_active/data/stage1a/

The --input-dir is the artifact root from `gh run download 25935989178`
(or wherever you've stashed Phase 1 traces).
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.linalg import eigh


# ──────────────────────────────────────────────────────────────────────
# Constants — mirror RATCHET.Experiments.{Exp1Predictions, BoundaryObservability}
# ──────────────────────────────────────────────────────────────────────
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

# The four conditional fields whose presence marks a boundary-active chain.
# BO-1: chain is boundary-active iff at least one faculty fired.
CONDITIONAL_FACULTY_FIELDS = [
    "entropy_score",
    "coherence_score",
    "optimization_veto_entropy_ratio",
    "epistemic_humility_certainty",
]

# Model lineup (mirrors Exp1Predictions + REGIME)
MODEL_TAGS = [
    ("qwen/qwen3.5-35b-a3b", "qwen-3.5-35b-a3b"),
    ("anthropic/claude-opus-4.7", "claude-opus-4.7"),
    ("openai/gpt-5.5", "gpt-5.5"),
    ("google/gemini-2.5-flash", "gemini-2.5-flash"),
    ("meta-llama/llama-4-scout", "llama-4-scout"),
]

BOOTSTRAP_RESAMPLES = 10_000
RNG_SEED = 0xC1715_E_EF


# Wire format trace extraction paths
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


def get_nested(d, path):
    cur = d
    for p in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(p)
    return cur


def cast_to_float(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
        return float(v)
    return None


def extract_features_from_trace(trace):
    features = {}
    last_conscience = None
    for c in trace.get("components", []):
        et = c.get("event_type")
        data = c.get("data") or c.get("payload") or {}
        if et == "CONSCIENCE_RESULT":
            last_conscience = data
        elif et in PATHS:
            for fname, path in PATHS[et].items():
                v = cast_to_float(get_nested(data, path))
                if v is not None:
                    features[fname] = v
    if last_conscience is not None:
        for fname, path in PATHS["CONSCIENCE_RESULT"].items():
            v = cast_to_float(get_nested(last_conscience, path))
            if v is not None:
                features[fname] = v
    return features


def is_boundary_active(features: dict) -> bool:
    """BO-1: at least one of the four faculty conditional fields is populated."""
    return any(f in features for f in CONDITIONAL_FACULTY_FIELDS)


def load_model_features(tee_dir: Path):
    """Returns (chains_all, chains_boundary_active, excluded).

    chains_* are lists of feature dicts with model+trace+thought keys.
    Mirrors phase1_aggregate's logic but separates the boundary-active subset.
    """
    chains_all = []
    chains_active = []
    excluded = 0
    if not tee_dir.is_dir():
        return chains_all, chains_active, excluded

    REQUIRED_EVENT_TYPES = {
        "THOUGHT_START", "SNAPSHOT_AND_CONTEXT", "DMA_RESULTS",
        "IDMA_RESULT", "ASPDMA_RESULT", "CONSCIENCE_RESULT",
        "ACTION_RESULT",
    }
    CORE_FIELDS = {
        "csdma_plausibility_score", "dsdma_domain_alignment",
        "coherence_level", "entropy_level",
        "idma_k_eff", "idma_correlation_risk",
        "conscience_passed", "action_was_overridden",
    }

    for batch_path in sorted(tee_dir.glob("**/*accord-batch-*.json")):
        try:
            batch = json.load(open(batch_path))
        except Exception:
            continue
        if batch.get("trace_level") != "detailed":
            continue
        for ev in batch.get("events", []):
            if ev.get("event_type") != "complete_trace":
                continue
            trace = ev.get("trace") or {}
            event_types = {c.get("event_type") for c in trace.get("components", [])}
            if not REQUIRED_EVENT_TYPES.issubset(event_types):
                excluded += 1
                continue
            features = extract_features_from_trace(trace)
            missing_core = [f for f in CORE_FIELDS if f not in features]
            if missing_core:
                excluded += 1
                continue
            chain = {
                "trace_id": trace.get("trace_id"),
                "thought_id": trace.get("thought_id"),
                **{f: features.get(f) for f in PROJECTION_16},
                "_boundary_active": is_boundary_active(features),
            }
            chains_all.append(chain)
            if chain["_boundary_active"]:
                chains_active.append(chain)
    return chains_all, chains_active, excluded


def compute_eigenspectrum(chains):
    """Run the standardized 16-feature PCA on a chain list. Returns eigenvalues
    of the correlation matrix (in descending order, non-negative)."""
    if len(chains) < 2:
        return None
    M = np.full((len(chains), 16), np.nan)
    for i, c in enumerate(chains):
        for j, f in enumerate(PROJECTION_16):
            v = c.get(f)
            if v is not None:
                M[i, j] = v
    col_means = np.nanmean(M, axis=0)
    M_imp = np.where(np.isnan(M), col_means, M)
    col_stds = M_imp.std(axis=0, ddof=0)
    keep = col_stds > 1e-9
    safe_stds = np.where(keep, col_stds, 1.0)
    M_std = (M_imp - M_imp.mean(axis=0)) / safe_stds
    M_std = M_std[:, keep]
    if M_std.shape[1] < 2 or M_std.shape[0] < 2:
        return None
    C = np.corrcoef(M_std, rowvar=False)
    lambdas = np.maximum(eigh(C, eigvals_only=True)[::-1], 0)
    return lambdas


def neff_h(lambdas):
    total = lambdas.sum()
    if total <= 0:
        return float("nan")
    p = lambdas / total
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def neff_pr(lambdas):
    s = lambdas.sum()
    s2 = (lambdas ** 2).sum()
    if s2 <= 0:
        return float("nan")
    return float((s * s) / s2)


def bootstrap_neff_h(chains, n_resamples=BOOTSTRAP_RESAMPLES, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    n = len(chains)
    if n < 2:
        return float("nan"), float("nan"), float("nan")
    samples = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        resample = [chains[i] for i in idx]
        lam = compute_eigenspectrum(resample)
        samples.append(neff_h(lam) if lam is not None else float("nan"))
    arr = np.array(samples)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 100:
        return float("nan"), float("nan"), float("nan")
    return float(arr.mean()), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def find_tee_dir(in_dir: Path, model_tag: str) -> Optional[Path]:
    candidates = (
        list(in_dir.glob(f"exp1-phase1-traces-{model_tag}/{model_tag}/tee_batches"))
        + list(in_dir.glob(f"exp1-phase1-traces-{model_tag}/tee_batches"))
        + list(in_dir.glob(f"{model_tag}/tee_batches"))
        + list(in_dir.glob(f"exp1-phase1-traces-{model_tag}*/tee_batches"))
    )
    return candidates[0] if candidates else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True,
                    help="Phase 1 artifact root (e.g., /tmp/exp1_p1)")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for model_id, tag in MODEL_TAGS:
        tee_dir = find_tee_dir(in_dir, tag)
        if tee_dir is None:
            rows.append({
                "model": model_id, "model_tag": tag,
                "tee_dir": None, "tee_dir_exists": False,
                "n_total": 0, "n_active": 0, "active_fraction": float("nan"),
                "full_corpus_neff_h": float("nan"),
                "active_subset_neff_h": float("nan"),
                "active_subset_ci95_low": float("nan"),
                "active_subset_ci95_high": float("nan"),
                "delta_full_to_active": float("nan"),
                "active_subset_passes_window": False,
            })
            continue

        chains_all, chains_active, _ = load_model_features(tee_dir)
        n_total = len(chains_all)
        n_active = len(chains_active)
        active_frac = (n_active / n_total) if n_total > 0 else float("nan")

        # Full-corpus N_eff_H
        lam_full = compute_eigenspectrum(chains_all)
        neff_full = neff_h(lam_full) if lam_full is not None else float("nan")

        # Boundary-active subset N_eff_H + bootstrap CI
        if n_active >= 2:
            lam_active = compute_eigenspectrum(chains_active)
            neff_active = neff_h(lam_active) if lam_active is not None else float("nan")
            mean_boot, ci_lo, ci_hi = bootstrap_neff_h(chains_active)
        else:
            neff_active = float("nan")
            mean_boot, ci_lo, ci_hi = float("nan"), float("nan"), float("nan")

        delta = (neff_active - neff_full) if (
            not math.isnan(neff_full) and not math.isnan(neff_active)
        ) else float("nan")

        # Does the boundary-active subset's CI fit the Exp 1 PASS window?
        # (For informational comparison only — does NOT apply the locked rule.)
        passes_window = (
            not math.isnan(ci_lo)
            and not math.isnan(ci_hi)
            and 6.6 <= ci_lo
            and ci_hi <= 7.6
        )

        rows.append({
            "model": model_id, "model_tag": tag,
            "tee_dir": str(tee_dir), "tee_dir_exists": True,
            "n_total": n_total, "n_active": n_active,
            "active_fraction": active_frac,
            "full_corpus_neff_h": neff_full,
            "active_subset_neff_h": neff_active,
            "active_subset_ci95_low": ci_lo,
            "active_subset_ci95_high": ci_hi,
            "delta_full_to_active": delta,
            "active_subset_passes_window": passes_window,
        })

    # Write JSON
    out_json = out_dir / "stage1a_results.json"
    with open(out_json, "w") as f:
        json.dump({
            "prereg_anchor": "fbc6795",
            "source_run": "25935989178",
            "min_valid_n": 50,
            "pass_window": [6.6, 7.6],
            "rows": rows,
        }, f, indent=2, default=str)

    # Write Markdown
    md = ["# Exp 1b Stage 1a — Boundary-Active Subset Re-Analysis\n\n"]
    md.append("**Status:** EXPLORATORY (per Exp 1 PRE_REGISTRATION.md §10.1).\n")
    md.append("**Source:** existing Phase 1 traces from run `25935989178`.\n")
    md.append("**Reference:** `RATCHET.Experiments.BoundaryObservability` BO-1..BO-4.\n\n")
    md.append("## Per-model boundary-active subset\n\n")
    md.append(
        "| Model | n_total | n_active | active_frac | Full N_eff_H | "
        "Active N_eff_H | Active 95% CI | Δ (active − full) | Active CI fits [6.6, 7.6]? |\n"
    )
    md.append(
        "|---|---|---|---|---|---|---|---|---|\n"
    )
    for r in rows:
        af = r["active_fraction"]
        af_s = f"{af:.2%}" if not math.isnan(af) else "—"
        fnh = r["full_corpus_neff_h"]
        anh = r["active_subset_neff_h"]
        clo = r["active_subset_ci95_low"]
        chi = r["active_subset_ci95_high"]
        delta = r["delta_full_to_active"]
        ci = (
            f"[{clo:.3f}, {chi:.3f}]"
            if not math.isnan(clo)
            else "—"
        )
        md.append(
            f"| `{r['model']}` | {r['n_total']} | {r['n_active']} | {af_s} | "
            f"{fnh:.3f if not math.isnan(fnh) else '—'} | "
            f"{anh:.3f if not math.isnan(anh) else '—'} | "
            f"{ci} | "
            f"{delta:+.3f if not math.isnan(delta) else '—'} | "
            f"{'✓' if r['active_subset_passes_window'] else '✗'} |\n"
        )
    md.append("\n## Interpretation reminders\n\n")
    md.append("This re-analysis is EXPLORATORY. The locked Exp 1 §10.1 decision rule "
              "remains INDETERMINATE for the original 5-cell sweep (Opus n=0 cell "
              "abort). Stage 1a recovers signal informally to guide Phase 1b design "
              "but does NOT apply the F-6 decision rule.\n\n")
    md.append("Per `BoundaryObservability` BO-2/BO-3: chains where conscience "
              "faculties did NOT fire carry no information about the 7.1 anchor. "
              "The Δ (active − full) column is the conditional-vs-marginal shift; "
              "positive values indicate the boundary-active subset clusters closer "
              "to the anchor than the full corpus, consistent with the per-chain-"
              "conditional reading of the CRC paper's anchor.\n")

    (out_dir / "STAGE1A_RESULTS.md").write_text("".join(md))
    print(f"Stage 1a results written to {out_dir}")
    for r in rows:
        print(f"  {r['model']}: n_total={r['n_total']} n_active={r['n_active']} "
              f"full→{r['full_corpus_neff_h']:.3f} active→{r['active_subset_neff_h']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
