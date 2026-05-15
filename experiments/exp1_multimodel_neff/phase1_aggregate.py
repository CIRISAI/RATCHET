#!/usr/bin/env python3
"""
Exp 1 — Phase 1 aggregator + locked decision rule application.

Reads the 5 per-model tee-batch artifact dirs produced by the
exp1_phase1.yml workflow's `sweep` matrix. Extracts the 16-feature
projection per chain from `detailed`-level batches. Computes per-chain
N_eff_H, then per-model mean + 95% bootstrap CI. Applies the locked
Exp1Predictions decision rule.

Output:
  phase1_results/PHASE1_DECISION.md
  phase1_results/PHASE1_DATA.json
  phase1_results/per_chain_features.csv

Decision rule (locked at PRE_REGISTRATION.md §10.1, formalized in
formal/RATCHET/Experiments/Exp1Predictions.lean):
  K = count of models with 95% bootstrap CI ⊆ [6.6, 7.6]
  K = 5         → PASS    (F-6 passes, H1 supported)
  K ∈ {3, 4}    → PARTIAL
  K ≤ 2         → FAIL    (F-6 falsified, H0 supported)
  any model n<50 → INDETERMINATE (§7 catastrophic-failure clause)
"""

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

# ──────────────────────────────────────────────────────────────────────
# Locked constants (mirror formal/RATCHET/Experiments/Exp1Predictions.lean)
# ──────────────────────────────────────────────────────────────────────
ANCHOR_NEFF = 7.1
PASS_LOWER = 6.6
PASS_UPPER = 7.6
NUM_MODELS = 5
TARGET_N = 100
MIN_VALID_N = 50
PASS_K_THRESHOLD = 5
PARTIAL_K_LOWER = 3
BOOTSTRAP_RESAMPLES = 10_000
RNG_SEED = 0xC1715_E_EF  # deterministic bootstrap seed

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

CORE_FIELDS = {
    "csdma_plausibility_score",
    "dsdma_domain_alignment",
    "coherence_level",
    "entropy_level",
    "idma_k_eff",
    "idma_correlation_risk",
    "conscience_passed",
    "action_was_overridden",
}

# Wire format uses `data` (FSD/TRACE_WIRE_FORMAT.md §4)
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

# Locked model lineup (PRE_REGISTRATION.md §5)
MODEL_TAGS = [
    ("qwen/qwen3.5-35b-a3b", "qwen-3.5-35b-a3b"),
    ("anthropic/claude-opus-4.7", "claude-opus-4.7"),
    ("openai/gpt-5.5", "gpt-5.5"),
    ("google/gemini-2.5-flash", "gemini-2.5-flash"),
    ("meta-llama/llama-4-scout", "llama-4-scout"),
]


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


def load_model_features(tee_dir: Path):
    """Return list of dicts: one per valid chain in `detailed` batches.

    Pre-registered exclusions (PRE_REGISTRATION.md §12):
      - Chain missing any of the 7 core event types → exclude
      - All 8 CORE projection fields must be present (others may be null)
    """
    chains = []
    excluded = 0
    if not tee_dir.is_dir():
        return chains, excluded

    REQUIRED_EVENT_TYPES = {
        "THOUGHT_START", "SNAPSHOT_AND_CONTEXT", "DMA_RESULTS",
        "IDMA_RESULT", "ASPDMA_RESULT", "CONSCIENCE_RESULT",
        "ACTION_RESULT",
    }

    for batch_path in sorted(tee_dir.glob("**/accord-batch-*.json")):
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
            chains.append({
                "trace_id": trace.get("trace_id"),
                "thought_id": trace.get("thought_id"),
                **{f: features.get(f) for f in PROJECTION_16},
            })
    return chains, excluded


def compute_neff_per_chain(rows):
    """Per-chain N_eff_H: standardize features, compute correlation matrix,
    eigendecompose, return entropy-perplexity over normalized eigenvalues.

    Per PRE_REGISTRATION.md §6: standardized 16-feature covariance.
    """
    if not rows:
        return []

    fields = PROJECTION_16
    M = np.full((len(rows), len(fields)), np.nan)
    for i, r in enumerate(rows):
        for j, f in enumerate(fields):
            v = r.get(f)
            if v is not None:
                M[i, j] = v

    # Per-field imputation (mean of non-NaN within this model's data)
    col_means = np.nanmean(M, axis=0)
    M_imp = np.where(np.isnan(M), col_means, M)

    # Standardize column-wise
    col_stds = M_imp.std(axis=0, ddof=0)
    keep = col_stds > 1e-9
    safe_stds = np.where(keep, col_stds, 1.0)
    M_std = (M_imp - M_imp.mean(axis=0)) / safe_stds
    M_std = M_std[:, keep]

    if M_std.shape[1] < 2 or M_std.shape[0] < 2:
        return [None] * len(rows)

    # Per-CHAIN N_eff requires multiple chains for the cov matrix.
    # We compute over the WHOLE per-model dataset and report the cohort-level
    # N_eff, then jackknife each chain out for the per-chain leave-one-out.
    # For Phase 1 we just compute one N_eff per model and one variance via
    # bootstrap over rows.
    C = np.corrcoef(M_std, rowvar=False)
    lambdas = np.maximum(eigh(C, eigvals_only=True)[::-1], 0)
    return lambdas


def neff_h_from_lambdas(lambdas):
    total = lambdas.sum()
    if total <= 0:
        return float("nan")
    p = lambdas / total
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def neff_pr_from_lambdas(lambdas):
    s = lambdas.sum()
    s2 = (lambdas ** 2).sum()
    if s2 <= 0:
        return float("nan")
    return float((s * s) / s2)


def bootstrap_neff_h(rows, n_resamples: int, seed: int):
    """Bootstrap percentile CI on the cohort-level N_eff_H.

    Resamples chains with replacement; recomputes the standardized cov,
    eigenvalues, and N_eff_H. Returns mean, lo (2.5%), hi (97.5%).
    """
    rng = np.random.default_rng(seed)
    n = len(rows)
    if n < 2:
        return float("nan"), float("nan"), float("nan")
    samples = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        resample = [rows[i] for i in idx]
        lam = compute_neff_per_chain(resample)
        if isinstance(lam, list):
            samples.append(float("nan"))
        else:
            samples.append(neff_h_from_lambdas(lam))
    arr = np.array(samples)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 100:
        return float("nan"), float("nan"), float("nan")
    return float(arr.mean()), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def variance_horizons(lambdas):
    total = lambdas.sum()
    if total <= 0:
        return None, None
    cum = np.cumsum(lambdas) / total
    h90 = int(np.argmax(cum >= 0.90) + 1)
    h99 = int(np.argmax(cum >= 0.99) + 1)
    return h90, h99


def decide(model_summaries):
    """Apply the locked Exp1Predictions decision rule."""
    # §7 catastrophic-failure clause
    for s in model_summaries:
        if s["valid_n"] < MIN_VALID_N:
            return "INDETERMINATE", 0
    k = sum(
        1 for s in model_summaries
        if PASS_LOWER <= s["ci95_low"] and s["ci95_high"] <= PASS_UPPER
    )
    if k == PASS_K_THRESHOLD:
        return "PASS", k
    if k >= PARTIAL_K_LOWER:
        return "PARTIAL", k
    return "FAIL", k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, help="Directory containing exp1-phase1-traces-* dirs (from artifact download)")
    ap.add_argument("--out-dir", required=True, help="Output dir for results")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    all_chains = []
    for model_id, tag in MODEL_TAGS:
        # Each matrix cell produced an artifact named exp1-phase1-traces-<tag>
        # which downloads as a directory of the same name (or just <tag>/).
        candidates = list(in_dir.glob(f"exp1-phase1-traces-{tag}/{tag}/tee_batches")) \
                   + list(in_dir.glob(f"exp1-phase1-traces-{tag}/tee_batches")) \
                   + list(in_dir.glob(f"{tag}/tee_batches")) \
                   + list(in_dir.glob(f"exp1-phase1-traces-{tag}*/tee_batches"))
        tee_dir = candidates[0] if candidates else (in_dir / tag / "tee_batches")
        chains, excluded = load_model_features(tee_dir)

        # Persist per-chain features for transparency
        for c in chains:
            c["model"] = model_id
        all_chains.extend(chains)

        if len(chains) < 2:
            summaries.append({
                "model": model_id, "model_tag": tag,
                "tee_dir": str(tee_dir),
                "valid_n": len(chains), "excluded": excluded,
                "mean_neff_h": float("nan"),
                "ci95_low": float("nan"), "ci95_high": float("nan"),
                "neff_pr": float("nan"),
                "h90": None, "h99": None,
                "passes_window": False,
            })
            continue

        lam = compute_neff_per_chain(chains)
        cohort_neff_h = neff_h_from_lambdas(lam)
        cohort_neff_pr = neff_pr_from_lambdas(lam)
        h90, h99 = variance_horizons(lam)
        boot_mean, ci_lo, ci_hi = bootstrap_neff_h(chains, BOOTSTRAP_RESAMPLES, RNG_SEED)
        passes = PASS_LOWER <= ci_lo and ci_hi <= PASS_UPPER

        summaries.append({
            "model": model_id, "model_tag": tag,
            "tee_dir": str(tee_dir),
            "valid_n": len(chains), "excluded": excluded,
            "cohort_neff_h": cohort_neff_h,
            "mean_neff_h": boot_mean,
            "ci95_low": ci_lo, "ci95_high": ci_hi,
            "neff_pr": cohort_neff_pr,
            "h90": h90, "h99": h99,
            "passes_window": passes,
        })

    decision, k = decide(summaries)

    # Write per-chain CSV
    with open(out_dir / "per_chain_features.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "trace_id", "thought_id"] + PROJECTION_16)
        w.writeheader()
        for r in all_chains:
            w.writerow({k: r.get(k) for k in w.fieldnames})

    # Write headline JSON
    with open(out_dir / "PHASE1_DATA.json", "w") as f:
        json.dump({
            "prereg_commit": "fbc6795",
            "agent_ref": "v2.8.10-stable",
            "agent_sha": "2446b8c4de033e5de70faed683d39afb8d094653",
            "projection_version": "crc-v1",
            "decision": decision,
            "passing_models_K": k,
            "summaries": summaries,
        }, f, indent=2, default=str)

    # Write decision markdown
    md = ["# Phase 1 — Decision\n\n"]
    md.append(f"**Pre-registration anchor:** `fbc6795`\n")
    md.append(f"**Agent ref:** `v2.8.10-stable` (sha `2446b8c4d`)\n")
    md.append(f"**Projection version:** `crc-v1`\n\n")
    md.append(f"## Headline decision: **{decision}**\n\n")
    md.append(f"Passing models K = **{k}** / 5\n\n")
    md.append("## Per-model summary\n\n")
    md.append("| Model | valid n | excl. | mean N_eff_H | 95% CI | passes window | N_eff_PR | h90 | h99 |\n")
    md.append("|---|---|---|---|---|---|---|---|---|\n")
    for s in summaries:
        ci = f"[{s['ci95_low']:.3f}, {s['ci95_high']:.3f}]" if not math.isnan(s["ci95_low"]) else "—"
        mean_h = f"{s['mean_neff_h']:.3f}" if not math.isnan(s["mean_neff_h"]) else "—"
        pr = f"{s['neff_pr']:.3f}" if not math.isnan(s["neff_pr"]) else "—"
        passes = "✓" if s["passes_window"] else "✗"
        md.append(
            f"| `{s['model']}` | {s['valid_n']} | {s['excluded']} | "
            f"{mean_h} | {ci} | {passes} | {pr} | {s['h90']} | {s['h99']} |\n"
        )
    md.append("\n## What this decision means\n\n")
    if decision == "PASS":
        md.append("H1 (structural) supported. F-6 passes: $N_{\\text{eff}} \\approx 7.1$ is a "
                  "property of the CIRIS constraint topology, not of any specific foundation "
                  "model. Substrate-independence at the LLM-substrate level confirmed.\n")
    elif decision == "PARTIAL":
        md.append("H_partial supported. Some models hit the window, others don't. The "
                  "follow-up question becomes: *which* model properties predict CIRIS "
                  "compatibility? Open a new pre-registration to investigate.\n")
    elif decision == "FAIL":
        md.append("H0 (model-specific) supported. F-6 falsified. The 7.1 threshold is "
                  "model-specific calibration. The framework's substrate-independence "
                  "claim weakens; CIRIS's effect depends on the underlying model's "
                  "properties.\n")
    else:  # INDETERMINATE
        md.append("§7 catastrophic-failure clause triggered. At least one model produced "
                  "fewer than 50 valid chains. Re-pre-registration required before any re-run.\n")

    md.append(f"\n## Pre-registered window: $[{PASS_LOWER}, {PASS_UPPER}]$\n")
    md.append(f"\nDecision rule formalized in `formal/RATCHET/Experiments/Exp1Predictions.lean`.\n")

    (out_dir / "PHASE1_DECISION.md").write_text("".join(md))
    print((out_dir / "PHASE1_DECISION.md").read_text())
    return 0 if decision != "INDETERMINATE" else 2


if __name__ == "__main__":
    sys.exit(main())
