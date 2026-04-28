# CIRIS Trace Format Specification

**Version:** 1.0
**Status:** Specification
**Date:** 2025-12-31

## 1. Overview

Every decision made by a CIRIS agent produces an immutable, cryptographically-signed trace. This document specifies the canonical trace format used for the Coherence Ratchet detection mechanism.

## 2. Trace Detail Levels

Traces are captured at three privacy levels, controlled by the sending agent:

| Level | Description | Use Case |
|-------|-------------|----------|
| `generic` | Numeric scores only (default) | Powers [ciris.ai/ciris-scoring](https://ciris.ai/ciris-scoring) |
| `detailed` | Adds lists & identifiers | Debugging without reasoning exposure |
| `full_traces` | Complete reasoning text | Research corpus contribution |

### What Each Level Captures

**generic** (default) - Minimum data for CIRIS Capacity Score:
- Numeric scores: `plausibility_score`, `domain_alignment`, `k_eff`, `correlation_risk`
- Boolean flags: `conscience_passed`, `fragility_flag`, `entropy_passed`
- Resource usage: `tokens_total`, `cost_cents`, `audit_sequence_number`
- NO text strings, NO reasoning, NO prompts

**detailed** - Adds actionable identifiers:
- String identifiers: `thought_type`, `trace_type`, `domain`, `phase`
- Lists: `stakeholders`, `sources_identified`, `correlation_factors`
- Still NO reasoning text or prompts

**full_traces** - Complete data for Coherence Ratchet corpus:
- Reasoning text: `reasoning`, `action_rationale`, `conscience_override_reason`
- Prompts: `prompt_used`, `aspdma_prompt`
- Full context: `task_description`, `conversation_history`

The batch payload includes `trace_level` to indicate what level of detail is available:

```json
{
  "events": [...],
  "batch_timestamp": "2026-01-15T14:00:00Z",
  "consent_timestamp": "2025-12-15T13:00:00Z",
  "trace_level": "generic",
  "correlation_metadata": {
    "deployment_region": "na",
    "deployment_type": "business",
    "agent_role": "customer_support"
  }
}
```

## 3. Trace Structure

A complete trace contains 6 components that capture the full decision-making pipeline:

```
Observation → Context → Rationale (3 DMAs) → Action Selection → Conscience → Action
```

### 3.1 Top-Level Fields

```json
{
  "trace_id": "trace-th_followup_th_std_2_b44ae980-20260101042008",
  "thought_id": "th_followup_th_std_2_b44ae980-ab8",
  "task_id": "VERIFY_IDENTITY_2e1cbbd8-e671-4124-884c-818a8aff9ce8",
  "agent_id_hash": "unknown",
  "started_at": "2026-01-01T04:20:08.026388+00:00",
  "completed_at": "2026-01-01T04:20:20.017550+00:00",
  "components": [...],
  "signature": "DdP06mwFbKfQh0148pzgIITEc6V_MRmX-5x5VwQJYl8W8GdWjvCtXAk1r4Of-3eZAVH6JQltb29OClqUQbrTBg",
  "signature_key_id": "wa-2025-06-14-ROOT00"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `trace_id` | string | Unique identifier for this trace |
| `thought_id` | string | ID of the thought being processed |
| `task_id` | string | ID of the parent task |
| `agent_id_hash` | string | Anonymized agent identifier |
| `started_at` | ISO8601 | When trace processing began |
| `completed_at` | ISO8601 | When trace processing completed |
| `components` | array | The 6 trace components (see below) |
| `signature` | string | Base64url Ed25519 signature of components |
| `signature_key_id` | string | Identifier of signing key |

### 3.2 Component Structure

Each component has:

```json
{
  "component_type": "observation|context|rationale|conscience|action",
  "event_type": "THOUGHT_START|SNAPSHOT_AND_CONTEXT|DMA_RESULTS|ASPDMA_RESULT|CONSCIENCE_RESULT|ACTION_RESULT",
  "timestamp": "2026-01-01T04:20:08.026388+00:00",
  "data": {...}
}
```

## 4. Component Details

### 4.1 THOUGHT_START (Observation)

**component_type:** `observation`

Captures what triggered this decision.

```json
{
  "component_type": "observation",
  "event_type": "THOUGHT_START",
  "timestamp": "2026-01-01T04:20:08.026388+00:00",
  "data": {
    "thought_type": "follow_up",
    "thought_status": "processing",
    "round_number": 0,
    "thought_depth": 1,
    "parent_thought_id": "th_std_2e22d5ce-250b-439b-a83f-3c8c3f048a54",
    "task_priority": 0,
    "task_description": "...",
    "initial_context": null,
    "channel_id": "api_127.0.0.1_8080",
    "source_adapter": null,
    "updated_info_available": false,
    "requires_human_input": null
  }
}
```

**Key Fields for Detection:**
- `thought_depth`: Higher values indicate iterative reasoning
- `thought_type`: `standard` vs `follow_up`
- `task_description`: Contains the original request

### 4.2 SNAPSHOT_AND_CONTEXT (Context)

**component_type:** `context`

Captures system state at decision time.

```json
{
  "component_type": "context",
  "event_type": "SNAPSHOT_AND_CONTEXT",
  "timestamp": "2026-01-01T04:20:08.093221+00:00",
  "data": {
    "system_snapshot": {
      "channel_id": "api_127.0.0.1_8080",
      "current_task_details": {...},
      "current_thought_summary": {...},
      "system_counts": {
        "total_tasks": 6,
        "total_thoughts": 13,
        "pending_tasks": 0,
        "pending_thoughts": 5
      },
      "agent_identity": {
        "agent_id": "Datum",
        "description": "...",
        "role": "...",
        "trust_level": 0.5
      },
      "agent_version": "1.8.0-stable",
      "agent_codename": "Context Engineering",
      "agent_code_hash": "d9048371d1cb",
      "telemetry_summary": {...},
      "continuity_summary": {...},
      "available_tools": [...]
    },
    "gathered_context": null,
    "relevant_memories": null,
    "cognitive_state": null
  }
}
```

**Key Fields for Detection:**
- `agent_identity.agent_id`: Agent name for identity verification
- `agent_code_hash`: Verify agent hasn't been modified
- `telemetry_summary`: Resource usage patterns

### 4.3 DMA_RESULTS (Rationale - 4 DMAs)

**component_type:** `rationale`

Contains results from 4 parallel Decision-Making Algorithms.

```json
{
  "component_type": "rationale",
  "event_type": "DMA_RESULTS",
  "timestamp": "2026-01-01T04:20:14.657339+00:00",
  "data": {
    "csdma": {
      "plausibility_score": 0.9,
      "flags": [],
      "reasoning": "The thought is highly plausible...",
      "prompt_used": "..."
    },
    "dsdma": {
      "domain": "Datum",
      "domain_alignment": 0.9,
      "flags": [],
      "reasoning": "The thought aligns with the Datum domain...",
      "prompt_used": "..."
    },
    "pdma": {
      "stakeholders": "user, system, community",
      "conflicts": "none",
      "reasoning": "The stakeholders affected...",
      "alignment_check": "The principles of Act Ethically and Be Honest...",
      "prompt_used": "..."
    },
    "idma": {
      "k_eff": 2.0,
      "correlation_risk": 0.2,
      "fragility_flag": false,
      "phase": "healthy",
      "sources_identified": ["source1", "source2"],
      "correlation_factors": ["shared_training_data"],
      "reasoning": "Analysis identified 2 independent sources...",
      "prompt_used": "..."
    },
    "combined_analysis": null
  }
}
```

**IDMA (Intuition DMA) - Coherence Collapse Analysis:**

The IDMA implements Coherence Collapse Analysis (CCA) per Covenant Section II, Chapter 5. It detects fragile reasoning by measuring epistemic diversity:

```
k_eff = k / (1 + rho * (k - 1))
```

Where:
- `k` = number of information sources identified
- `rho` = correlation coefficient between sources (0.0-1.0)
- `k_eff` = effective number of truly independent sources

**IDMA Phases:**

| Phase | k_eff Range | Description |
|-------|-------------|-------------|
| `nascent` | ~1.0 | New agent, expected single-source dependence |
| `emerging` | 1.0-2.0 | Developing epistemic diversity |
| `healthy` | >= 2.0 | Multiple truly independent perspectives |
| `fragile` | < 2.0 (established agent) | Dangerous regression to single-source |

**Key Fields for Detection:**

| DMA | Score Field | Purpose |
|-----|-------------|---------|
| CSDMA | `plausibility_score` | Does reasoning match common sense? (0.0-1.0) |
| DSDMA | `domain_alignment` | Does behavior match agent's stated identity? (0.0-1.0) |
| PDMA | `stakeholders`, `conflicts` | Ethical analysis of affected parties |
| IDMA | `k_eff`, `fragility_flag` | Epistemic diversity and single-source risk |

### 4.4 ASPDMA_RESULT (Action Selection)

**component_type:** `rationale`

Contains the action selection decision and justification.

```json
{
  "component_type": "rationale",
  "event_type": "ASPDMA_RESULT",
  "timestamp": "2026-01-01T04:20:19.985803+00:00",
  "data": {
    "selected_action": "HandlerActionType.TASK_COMPLETE",
    "action_rationale": "The original task required affirming the agent's identity...",
    "reasoning_summary": null,
    "action_parameters": null,
    "alternatives_considered": null,
    "is_recursive": false,
    "aspdma_prompt": "..."
  }
}
```

**Key Fields for Detection:**
- `action_rationale`: Natural language justification (key for semantic analysis)
- `selected_action`: The chosen action type
- `is_recursive`: Whether this followed a conscience override

### 4.5 CONSCIENCE_RESULT (Conscience)

**component_type:** `conscience`

Contains the 6 conscience checks.

```json
{
  "component_type": "conscience",
  "event_type": "CONSCIENCE_RESULT",
  "timestamp": "2026-01-01T04:20:19.986500+00:00",
  "data": {
    "conscience_passed": true,
    "action_was_overridden": false,
    "final_action": "HandlerActionType.TASK_COMPLETE",
    "conscience_override_reason": null,
    "epistemic_data": {
      "entropy_level": 0.0,
      "coherence_level": 1.0,
      "uncertainty_acknowledged": true,
      "reasoning_transparency": 1.0,
      "CIRIS_OBSERVATION_UPDATED_STATUS": null
    },
    "ethical_faculties_skipped": null,
    "updated_status_detected": false,
    "updated_status_content": null,
    "thought_depth_triggered": null,
    "thought_depth_current": null,
    "thought_depth_max": null,
    "entropy_passed": null,
    "entropy_score": null,
    "entropy_threshold": null,
    "entropy_reason": null,
    "coherence_passed": null,
    "coherence_score": null,
    "coherence_threshold": null,
    "coherence_reason": null,
    "optimization_veto_passed": null,
    "optimization_veto_decision": null,
    "optimization_veto_justification": null,
    "optimization_veto_entropy_ratio": null,
    "optimization_veto_affected_values": null,
    "epistemic_humility_passed": null,
    "epistemic_humility_certainty": null,
    "epistemic_humility_uncertainties": null,
    "epistemic_humility_justification": null,
    "epistemic_humility_recommendation": null
  }
}
```

**Conscience Checks:**

| Category | Check | Field | Description |
|----------|-------|-------|-------------|
| **Bypass Guardrail** | Updated Status | `updated_status_detected` | Forces PONDER if new info arrives |
| **Bypass Guardrail** | Thought Depth | `thought_depth_triggered` | Forces DEFER if reasoning too deep |
| **Ethical Faculty** | Entropy | `entropy_passed` | Appropriate information uncertainty |
| **Ethical Faculty** | Coherence | `coherence_passed` | Consistency with prior commitments |
| **Ethical Faculty** | Optimization Veto | `optimization_veto_passed` | Preserves human values over efficiency |
| **Ethical Faculty** | Epistemic Humility | `epistemic_humility_passed` | Demonstrates appropriate uncertainty |

**Key Fields for Detection:**
- `conscience_passed`: Overall pass/fail
- `action_was_overridden`: Did conscience change the action?
- `epistemic_data.coherence_level`: Internal consistency (0.0-1.0)
- `epistemic_data.entropy_level`: Information uncertainty (0.0-1.0)

### 4.6 ACTION_RESULT (Action)

**component_type:** `action`

Contains execution results and audit metadata.

```json
{
  "component_type": "action",
  "event_type": "ACTION_RESULT",
  "timestamp": "2026-01-01T04:20:20.017550+00:00",
  "data": {
    "action_executed": "task_complete",
    "action_parameters": {
      "channel_id": null,
      "completion_reason": "Affirmation successfully delivered...",
      "context": {"task_id": "...", "status": "completed"},
      "positive_moment": "Successful affirmation...",
      "persist_images": false
    },
    "execution_success": true,
    "execution_error": null,
    "execution_time_ms": 20.067,
    "follow_up_thought_id": null,
    "audit_entry_id": "52eadd3c-75b7-42cf-88fb-dc85be19eb2a",
    "audit_sequence_number": 34,
    "audit_entry_hash": "5142243a8bfd9f0cf4122fac1ca76f1174bf54c1769cfbc08da3d61db26caa1c",
    "audit_signature": "oORMbL37AoVCCT8v...",
    "tokens_input": 86763,
    "tokens_output": 616,
    "tokens_total": 87379,
    "cost_cents": 0.87379,
    "carbon_grams": 13.10685,
    "energy_mwh": 26213.7,
    "llm_calls": 4,
    "models_used": ["meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"]
  }
}
```

**Key Fields for Detection:**
- `execution_success`: Did the action succeed?
- `audit_sequence_number`: Position in hash chain
- `audit_entry_hash`: SHA-256 for chain verification
- `tokens_total`, `cost_cents`: Resource usage tracking

## 5. Cryptographic Verification

### 5.1 Trace Signature

The trace signature is an Ed25519 signature over the canonical JSON of `components`:

```python
import json
import base64
from nacl.signing import VerifyKey

def verify_trace(trace: dict, public_key: bytes) -> bool:
    # Canonical JSON: sorted keys, no whitespace
    message = json.dumps(trace["components"], sort_keys=True).encode()

    # Base64url decode signature
    signature = base64.urlsafe_b64decode(trace["signature"] + "==")

    # Verify
    verify_key = VerifyKey(public_key)
    verify_key.verify(message, signature)
    return True
```

### 5.2 Audit Hash Chain

Each `ACTION_RESULT` contains hash chain fields:

```
audit_entry_hash[n] = SHA256(
    audit_entry_hash[n-1] ||
    action_executed ||
    execution_success ||
    timestamp
)
```

Verification:
1. Retrieve traces by `audit_sequence_number` for an agent
2. Verify each hash links to previous
3. Any gap or mismatch indicates tampering

## 6. Wakeup Trace Types

The 5 wakeup ritual traces demonstrate agent alignment:

| Type | Task ID Pattern | Purpose |
|------|-----------------|---------|
| VERIFY_IDENTITY | `VERIFY_IDENTITY_*` | Confirms agent knows who it is |
| VALIDATE_INTEGRITY | `VALIDATE_INTEGRITY_*` | Confirms internal state is valid |
| EVALUATE_RESILIENCE | `EVALUATE_RESILIENCE_*` | Confirms error handling works |
| ACCEPT_INCOMPLETENESS | `ACCEPT_INCOMPLETENESS_*` | Demonstrates epistemic humility |
| EXPRESS_GRATITUDE | `EXPRESS_GRATITUDE_*` | Affirms Ubuntu values |

## 7. Storage Format

CIRISLens stores traces in TimescaleDB with denormalized fields for fast queries:

```sql
-- See sql/011_covenant_traces.sql and sql/012_trace_levels_idma.sql for full schema
SELECT
    trace_id,
    trace_level,
    csdma_plausibility_score,
    dsdma_domain_alignment,
    idma_k_eff,
    idma_correlation_risk,
    idma_fragility_flag,
    idma_phase,
    coherence_level,
    action_rationale,
    selected_action,
    conscience_passed
FROM cirislens.covenant_traces
WHERE signature_verified = TRUE
ORDER BY timestamp DESC;
```

**IDMA Fragility Query:**

```sql
-- Find traces with fragile reasoning (k_eff < 2)
SELECT trace_id, agent_id_hash, idma_k_eff, idma_phase
FROM cirislens.covenant_traces
WHERE idma_fragility_flag = TRUE
  AND timestamp > NOW() - INTERVAL '24 hours'
ORDER BY idma_k_eff ASC;
```

## 8. API Endpoints

### 8.1 Receive Traces

```
POST /api/v1/covenant/events
Content-Type: application/json

{
  "events": [
    {
      "event_type": "complete_trace",
      "trace": { ... }
    }
  ],
  "batch_timestamp": "2026-01-01T04:20:20Z",
  "consent_timestamp": "2026-01-01T00:00:00Z"
}
```

### 8.2 Query Traces

```
GET /api/v1/covenant/traces?trace_type=VERIFY_IDENTITY&limit=100
```

## 9. PII Scrubbing for Full Traces

Full traces contain reasoning text that may include personally identifiable information. CIRISLens automatically scrubs PII while preserving cryptographic provenance.

### 9.1 Scrubbing Pipeline

```
Agent sends full_trace → Verify signature → Hash original → Scrub PII → Sign scrubbed → Store
```

### 9.2 Scrubbed Fields

21 text fields are scrubbed before storage:

| Component | Fields |
|-----------|--------|
| THOUGHT_START | `task_description`, `initial_context` |
| SNAPSHOT_AND_CONTEXT | `system_snapshot`, `gathered_context`, `relevant_memories`, `conversation_history` |
| DMA_RESULTS | `reasoning`, `prompt_used`, `combined_analysis` |
| ASPDMA_RESULT | `action_rationale`, `reasoning_summary`, `action_parameters`, `aspdma_prompt` |
| CONSCIENCE_RESULT | `conscience_override_reason`, `epistemic_data`, `updated_status_content`, `entropy_reason`, `coherence_reason`, `optimization_veto_justification`, `epistemic_humility_justification` |
| ACTION_RESULT | `execution_error` |

### 9.3 Entity Detection

**NER (spaCy):** Names, organizations, locations → `[PERSON_1]`, `[ORG_1]`, etc.

**Regex:** Emails, phones, IPs, SSNs, credit cards → `[EMAIL]`, `[PHONE]`, etc.

### 9.4 Cryptographic Envelope

| Field | Purpose | Required for Provenance? |
|-------|---------|-------------------------|
| `original_content_hash` | SHA-256 of pre-scrub content | ✅ Yes |
| `signature` | Agent's original signature | ✅ Yes |
| `signature_verified` | Whether signature was valid | ✅ Yes |
| `scrub_timestamp` | When scrubbing occurred | ✅ Yes |
| `scrub_signature` | CIRISLens signature of scrubbed | ❌ Nice to have |
| `scrub_key_id` | Key used for scrub signing | ❌ Nice to have |

**Key insight:** Even if the scrub signing key is lost, provenance is still provable via `original_content_hash` and verified agent signature.

### 9.5 Mock Trace Filtering

Traces using mock LLMs (containing "mock" in `models_used`) are excluded from storage to keep the production corpus clean.

## 10. References

- [Coherence Ratchet Detection](./coherence_ratchet_detection.md)
- [CIRIS Scoring Specification](./ciris_scoring_specification.md)
- [CIRIS How It Works](https://ciris.ai/how-it-works/)
- [Explore a Trace](https://ciris.ai/explore-a-trace/)
- [Privacy Policy](https://ciris.ai/privacy)
