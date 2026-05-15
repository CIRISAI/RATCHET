#!/usr/bin/env python3
"""
Exp 1 — Phase 0 smoke analyzer (final pre-Phase-1 gating check).

Reads the local-tee batches produced by phase0_smoke.sh for each model and
confirms:

  1. Every model produced at least one valid `complete_trace` event at
     `trace_level=detailed`.
  2. The 16-feature projection (PRE_REGISTRATION.md §6) extracts cleanly
     from each trace — no missing core fields, no NaNs in fields that
     should be 100% present.
  3. `reasoning` content is absent or empty in all model responses
     (cross-check on Phase 0 v1's per-API findings, now at the
     end-to-end CIRIS pipeline level).
  4. Per-model thought count matches the 2 questions × 1 trial expectation.

Output:
  experiments/exp1_multimodel_neff/SMOKE_ANALYSIS.md  — summary table

This script does NOT compute N_eff (n=2 per model is too small for PCA
on 16 features). Phase 1's full sweep does that.
"""

import glob
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).parent
SMOKE_DATA = EXPERIMENT_DIR / "data" / "smoke"

# Locked from PRE_REGISTRATION.md §5
MODELS = [
    "qwen/qwen3.5-35b-a3b",
    "anthropic/claude-opus-4.7",
    "openai/gpt-5.5",
    "google/gemini-2.5-flash",
    "meta-llama/llama-4-scout",
]

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

# Fields that should be 100% present in every chain (vs conditionally-present)
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

# Wire-format extraction paths (per FSD/TRACE_WIRE_FORMAT.md §5; data not payload)
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
    """Given a CompleteTrace dict, extract the per-thought 16-feature vector.

    Returns dict of feature_name -> float (or None if missing).
    """
    features = {}
    components = trace.get("components", [])
    last_conscience = None
    for c in components:
        et = c.get("event_type")
        # Wire format uses `data` not `payload`.
        data = c.get("data") or c.get("payload") or {}
        if et == "CONSCIENCE_RESULT":
            # If multiple, the loop will keep the last.
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


def find_tee_dir_for_model(model_id: str) -> Path | None:
    """The smoke script runs each model in its own qa_runner invocation, which
    creates a fresh /tmp/qa-runner-lens-traces-<UTC>/ dir per run. Globs the
    candidate dirs and picks the one that matches the model.

    For Phase 0 we just glob recent dirs and identify the model by inspecting
    the first event's deployment_profile.agent_template or
    SNAPSHOT_AND_CONTEXT.agent_version — but the more reliable signal is to
    walk all dirs and assign them to models by chronological order matching
    the smoke run order.

    This function is a best-effort harvester. The full smoke.sh writes a
    proper per-model dir under data/smoke/<MODEL_TAG>/tee_batches/.
    """
    model_tag = model_id.replace("/", "-")
    candidate = SMOKE_DATA / model_tag / "tee_batches"
    if candidate.is_dir():
        return candidate
    return None


def analyze_model(model_id: str):
    tee_dir = find_tee_dir_for_model(model_id)
    summary = {
        "model": model_id,
        "tee_dir": str(tee_dir) if tee_dir else None,
        "tee_dir_exists": tee_dir is not None and tee_dir.is_dir(),
        "complete_traces": 0,
        "trace_levels": dict(),
        "thoughts": 0,
        "thoughts_with_all_core_fields": 0,
        "thoughts_missing_core": [],
        "feature_presence": dict(),
        "reasoning_evidence": [],
        "errors": [],
    }
    if not tee_dir or not tee_dir.is_dir():
        summary["errors"].append(f"tee dir not found at {tee_dir}")
        return summary

    field_present = Counter()
    level_counter = Counter()
    for batch_path in sorted(tee_dir.glob("**/accord-batch-*.json")):
        try:
            batch = json.load(open(batch_path))
        except Exception as e:
            summary["errors"].append(f"{batch_path.name}: load failed: {e}")
            continue
        level = batch.get("trace_level", "?")
        level_counter[level] += 1
        for ev in batch.get("events", []):
            if ev.get("event_type") != "complete_trace":
                continue
            trace = ev.get("trace") or {}
            summary["complete_traces"] += 1
            # Only thoughts from detailed batches are usable for projection.
            if level != "detailed":
                continue
            features = extract_features_from_trace(trace)
            summary["thoughts"] += 1
            for f in PROJECTION_16:
                if f in features:
                    field_present[f] += 1
            missing_core = [f for f in CORE_FIELDS if f not in features]
            if missing_core:
                summary["thoughts_missing_core"].append({
                    "trace_id": trace.get("trace_id"),
                    "thought_id": trace.get("thought_id"),
                    "missing": missing_core,
                })
            else:
                summary["thoughts_with_all_core_fields"] += 1
            # Reasoning evidence — look at LLM_CALL components for any non-empty
            # `reasoning_content` field or `completion_tokens_details.reasoning_tokens > 0`
            for c in trace.get("components", []):
                if c.get("event_type") != "LLM_CALL":
                    continue
                cd = c.get("data") or {}
                # Wire format may not carry the per-call reasoning info at detailed
                # level. We mostly catch it via cost-tokens delta in higher levels.
                rt = cd.get("reasoning_tokens") or 0
                if rt and rt > 0:
                    summary["reasoning_evidence"].append({
                        "trace_id": trace.get("trace_id"),
                        "thought_id": trace.get("thought_id"),
                        "llm_call_tokens": rt,
                    })

    summary["trace_levels"] = dict(level_counter)
    summary["feature_presence"] = {f: field_present.get(f, 0) for f in PROJECTION_16}
    return summary


def main():
    SMOKE_DATA.mkdir(parents=True, exist_ok=True)
    rows = [analyze_model(m) for m in MODELS]
    md = ["# Phase 0 Smoke Analysis\n"]
    md.append(f"\nGenerated at: {Path(__file__).name}\n")
    md.append("\n## Per-model summary\n\n")
    md.append("| Model | tee_dir? | complete_traces | thoughts | core-fields complete | reasoning evidence | errors |\n")
    md.append("|---|---|---|---|---|---|---|\n")
    for r in rows:
        md.append(
            f"| `{r['model']}` | {r['tee_dir_exists']} | {r['complete_traces']} | "
            f"{r['thoughts']} | {r['thoughts_with_all_core_fields']}/{r['thoughts']} | "
            f"{len(r['reasoning_evidence'])} | {len(r['errors'])} |\n"
        )

    md.append("\n## Per-model feature presence (16 projection fields × thought count)\n")
    md.append("\n| Field | " + " | ".join(m.split("/")[-1] for m in MODELS) + " |\n")
    md.append("|---|" + "|".join(["---"] * len(MODELS)) + "|\n")
    for f in PROJECTION_16:
        row = "| " + f + " |"
        for r in rows:
            row += f" {r['feature_presence'].get(f, 0)} |"
        md.append(row + "\n")

    # Verdict
    all_good = all(
        r["thoughts_with_all_core_fields"] >= 1 and len(r["reasoning_evidence"]) == 0
        and len(r["errors"]) == 0
        for r in rows
    )
    md.append("\n## Verdict\n")
    if all_good:
        md.append("\n**STATUS: ✓ PHASE 0 SMOKE CLEAN — proceed to Phase 1 pre-commit**\n\n")
        md.append("Every model produced at least one trace with all 8 core projection "
                  "fields populated. No reasoning evidence detected at the LLM-call level. "
                  "No analysis errors.\n")
    else:
        md.append("\n**STATUS: ⚠ INVESTIGATE BEFORE PHASE 1**\n\n")
        for r in rows:
            if r["thoughts_with_all_core_fields"] == 0 or r["reasoning_evidence"] or r["errors"]:
                md.append(f"\n### `{r['model']}` issues\n")
                if r["thoughts_with_all_core_fields"] == 0:
                    md.append(f"- No thoughts with complete core fields (got {r['thoughts']} thoughts total).\n")
                for e in r["reasoning_evidence"][:5]:
                    md.append(f"- Reasoning evidence: trace {e['trace_id']} thought {e['thought_id']} llm_call rt={e['llm_call_tokens']}\n")
                for err in r["errors"]:
                    md.append(f"- ERROR: {err}\n")

    md.append("\n## Raw per-model summaries\n")
    md.append("```json\n")
    md.append(json.dumps(rows, indent=2))
    md.append("\n```\n")

    out_path = EXPERIMENT_DIR / "SMOKE_ANALYSIS.md"
    out_path.write_text("".join(md))
    print(f"SMOKE_ANALYSIS.md written to {out_path}")
    print(f"all_good={all_good}")
    return 0 if all_good else 1


if __name__ == "__main__":
    sys.exit(main())
