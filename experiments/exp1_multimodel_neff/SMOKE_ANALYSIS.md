# Phase 0 Smoke Analysis

Generated at: phase0_smoke_analyze.py

## Per-model summary

| Model | tee_dir? | complete_traces | thoughts | core-fields complete | reasoning evidence | errors |
|---|---|---|---|---|---|---|
| `qwen/qwen3.5-35b-a3b` | True | 12 | 3 | 3/3 | 0 | 0 |
| `anthropic/claude-opus-4.7` | True | 12 | 3 | 3/3 | 0 | 0 |
| `openai/gpt-5.5` | True | 16 | 4 | 4/4 | 0 | 0 |
| `google/gemini-2.5-flash` | True | 16 | 4 | 4/4 | 0 | 0 |
| `meta-llama/llama-4-scout` | True | 15 | 4 | 4/4 | 0 | 0 |

## Per-model feature presence (16 projection fields × thought count)

| Field | qwen3.5-35b-a3b | claude-opus-4.7 | gpt-5.5 | gemini-2.5-flash | llama-4-scout |
|---|---|---|---|---|---|
| csdma_plausibility_score | 3 | 3 | 4 | 4 | 4 |
| dsdma_domain_alignment | 3 | 3 | 4 | 4 | 4 |
| coherence_level | 3 | 3 | 4 | 4 | 4 |
| entropy_level | 3 | 3 | 4 | 4 | 4 |
| idma_k_eff | 3 | 3 | 4 | 4 | 4 |
| idma_correlation_risk | 3 | 3 | 4 | 4 | 4 |
| entropy_score | 2 | 2 | 2 | 2 | 2 |
| coherence_score | 2 | 2 | 2 | 2 | 2 |
| optimization_veto_entropy_ratio | 2 | 2 | 2 | 2 | 2 |
| epistemic_humility_certainty | 2 | 2 | 2 | 2 | 2 |
| conscience_passed | 3 | 3 | 4 | 4 | 4 |
| entropy_passed | 2 | 2 | 2 | 2 | 2 |
| coherence_passed | 2 | 2 | 2 | 2 | 2 |
| optimization_veto_passed | 2 | 2 | 2 | 2 | 2 |
| epistemic_humility_passed | 2 | 2 | 2 | 2 | 2 |
| action_was_overridden | 3 | 3 | 4 | 4 | 4 |

## Verdict

**STATUS: ✓ PHASE 0 SMOKE CLEAN — proceed to Phase 1 pre-commit**

Every model produced at least one trace with all 8 core projection fields populated. No reasoning evidence detected at the LLM-call level. No analysis errors.

## Raw per-model summaries
```json
[
  {
    "model": "qwen/qwen3.5-35b-a3b",
    "tee_dir": "/home/emoore/RATCHET/experiments/exp1_multimodel_neff/data/smoke/qwen-qwen3.5-35b-a3b/tee_batches",
    "tee_dir_exists": true,
    "complete_traces": 12,
    "trace_levels": {
      "detailed": 2,
      "full_traces": 2,
      "generic": 4
    },
    "thoughts": 3,
    "thoughts_with_all_core_fields": 3,
    "thoughts_missing_core": [],
    "feature_presence": {
      "csdma_plausibility_score": 3,
      "dsdma_domain_alignment": 3,
      "coherence_level": 3,
      "entropy_level": 3,
      "idma_k_eff": 3,
      "idma_correlation_risk": 3,
      "entropy_score": 2,
      "coherence_score": 2,
      "optimization_veto_entropy_ratio": 2,
      "epistemic_humility_certainty": 2,
      "conscience_passed": 3,
      "entropy_passed": 2,
      "coherence_passed": 2,
      "optimization_veto_passed": 2,
      "epistemic_humility_passed": 2,
      "action_was_overridden": 3
    },
    "reasoning_evidence": [],
    "errors": []
  },
  {
    "model": "anthropic/claude-opus-4.7",
    "tee_dir": "/home/emoore/RATCHET/experiments/exp1_multimodel_neff/data/smoke/anthropic-claude-opus-4.7/tee_batches",
    "tee_dir_exists": true,
    "complete_traces": 12,
    "trace_levels": {
      "generic": 4,
      "detailed": 2,
      "full_traces": 2
    },
    "thoughts": 3,
    "thoughts_with_all_core_fields": 3,
    "thoughts_missing_core": [],
    "feature_presence": {
      "csdma_plausibility_score": 3,
      "dsdma_domain_alignment": 3,
      "coherence_level": 3,
      "entropy_level": 3,
      "idma_k_eff": 3,
      "idma_correlation_risk": 3,
      "entropy_score": 2,
      "coherence_score": 2,
      "optimization_veto_entropy_ratio": 2,
      "epistemic_humility_certainty": 2,
      "conscience_passed": 3,
      "entropy_passed": 2,
      "coherence_passed": 2,
      "optimization_veto_passed": 2,
      "epistemic_humility_passed": 2,
      "action_was_overridden": 3
    },
    "reasoning_evidence": [],
    "errors": []
  },
  {
    "model": "openai/gpt-5.5",
    "tee_dir": "/home/emoore/RATCHET/experiments/exp1_multimodel_neff/data/smoke/openai-gpt-5.5/tee_batches",
    "tee_dir_exists": true,
    "complete_traces": 16,
    "trace_levels": {
      "generic": 6,
      "detailed": 3,
      "full_traces": 3
    },
    "thoughts": 4,
    "thoughts_with_all_core_fields": 4,
    "thoughts_missing_core": [],
    "feature_presence": {
      "csdma_plausibility_score": 4,
      "dsdma_domain_alignment": 4,
      "coherence_level": 4,
      "entropy_level": 4,
      "idma_k_eff": 4,
      "idma_correlation_risk": 4,
      "entropy_score": 2,
      "coherence_score": 2,
      "optimization_veto_entropy_ratio": 2,
      "epistemic_humility_certainty": 2,
      "conscience_passed": 4,
      "entropy_passed": 2,
      "coherence_passed": 2,
      "optimization_veto_passed": 2,
      "epistemic_humility_passed": 2,
      "action_was_overridden": 4
    },
    "reasoning_evidence": [],
    "errors": []
  },
  {
    "model": "google/gemini-2.5-flash",
    "tee_dir": "/home/emoore/RATCHET/experiments/exp1_multimodel_neff/data/smoke/google-gemini-2.5-flash/tee_batches",
    "tee_dir_exists": true,
    "complete_traces": 16,
    "trace_levels": {
      "generic": 6,
      "detailed": 3,
      "full_traces": 3
    },
    "thoughts": 4,
    "thoughts_with_all_core_fields": 4,
    "thoughts_missing_core": [],
    "feature_presence": {
      "csdma_plausibility_score": 4,
      "dsdma_domain_alignment": 4,
      "coherence_level": 4,
      "entropy_level": 4,
      "idma_k_eff": 4,
      "idma_correlation_risk": 4,
      "entropy_score": 2,
      "coherence_score": 2,
      "optimization_veto_entropy_ratio": 2,
      "epistemic_humility_certainty": 2,
      "conscience_passed": 4,
      "entropy_passed": 2,
      "coherence_passed": 2,
      "optimization_veto_passed": 2,
      "epistemic_humility_passed": 2,
      "action_was_overridden": 4
    },
    "reasoning_evidence": [],
    "errors": []
  },
  {
    "model": "meta-llama/llama-4-scout",
    "tee_dir": "/home/emoore/RATCHET/experiments/exp1_multimodel_neff/data/smoke/meta-llama-llama-4-scout/tee_batches",
    "tee_dir_exists": true,
    "complete_traces": 15,
    "trace_levels": {
      "generic": 6,
      "detailed": 3,
      "full_traces": 2
    },
    "thoughts": 4,
    "thoughts_with_all_core_fields": 4,
    "thoughts_missing_core": [],
    "feature_presence": {
      "csdma_plausibility_score": 4,
      "dsdma_domain_alignment": 4,
      "coherence_level": 4,
      "entropy_level": 4,
      "idma_k_eff": 4,
      "idma_correlation_risk": 4,
      "entropy_score": 2,
      "coherence_score": 2,
      "optimization_veto_entropy_ratio": 2,
      "epistemic_humility_certainty": 2,
      "conscience_passed": 4,
      "entropy_passed": 2,
      "coherence_passed": 2,
      "optimization_veto_passed": 2,
      "epistemic_humility_passed": 2,
      "action_was_overridden": 4
    },
    "reasoning_evidence": [],
    "errors": []
  }
]
```
