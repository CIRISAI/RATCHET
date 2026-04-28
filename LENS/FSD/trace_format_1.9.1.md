# CIRIS Trace Format v1.9.1

Documented from production mock traces received 2026-01-25.

## Complete Field Reference by Component

### ACTION_RESULT Fields

| Field | Level | Type | Description | Example |
|-------|-------|------|-------------|---------|
| `action_success` | GENERIC | bool | Whether action executed successfully | `true` |
| `tokens_input` | GENERIC | int | Input tokens consumed | `196835` |
| `tokens_output` | GENERIC | int | Output tokens generated | `9216` |
| `tokens_total` | GENERIC | int | Total tokens | `206051` |
| `cost_cents` | GENERIC | float | Cost in cents | `4.21318` |
| `carbon_grams` | GENERIC | float | CO2 emissions | `10.30255` |
| `energy_mwh` | GENERIC | float | Energy used | `20605.1` |
| `llm_calls` | GENERIC | int | Number of LLM calls | `9` |
| `execution_time_ms` | GENERIC | float | Execution duration in ms | `44.671` |
| `has_positive_moment` | GENERIC | bool | **Positive moment indicator** | `false` |
| `has_execution_error` | GENERIC | bool | Error indicator | `false` |
| `audit_sequence_number` | GENERIC | int | Audit chain sequence | `85` |
| `audit_entry_hash` | GENERIC | string | Audit hash | `404a0a11...` |
| `models_used` | GENERIC | array | Models used | `["llama4scout"]` |
| `follow_up_thought_id` | DETAILED | string | Next thought in chain | `th_followup_...` |
| `audit_entry_id` | DETAILED | UUID | Audit entry ID | `b7dc2f65-...` |
| `api_bases_used` | DETAILED | array | API endpoints used | `["default"]` |
| `execution_error` | DETAILED | string | Error details (if any) | `null` |
| `audit_signature` | DETAILED | string | Ed25519 signature | `1+p67y//...` |
| `action_parameters` | FULL | object | Full action parameters | `{content: ...}` |
| `positive_moment` | FULL | string | **Full positive moment text** | (gratitude text) |

### ASPDMA_RESULT Fields

| Field | Level | Type | Description | Example |
|-------|-------|------|-------------|---------|
| `selected_action` | GENERIC | string | Action selected | `SPEAK` |
| `is_recursive` | GENERIC | bool | Whether action is recursive | `false` |
| `action_rationale` | FULL | string | Rationale for action | (reasoning text) |
| `aspdma_prompt` | FULL | string | Prompt used for ASPDMA | (prompt text) |
| `raw_llm_response` | FULL | string | Raw LLM response | (response text) |

### DMA_RESULTS Fields

| Field | Level | Type | Description | Example |
|-------|-------|------|-------------|---------|
| `csdma_plausibility_score` | GENERIC | float | Common sense plausibility | `0.9` |
| `dsdma_domain_alignment` | GENERIC | float | Domain alignment | `0.9` |
| `dsdma_domain` | GENERIC | string | Domain name | `Datum` |
| `pdma_stakeholders` | GENERIC | string | Affected stakeholders | `system, operators` |
| `pdma_conflicts` | GENERIC | string | Ethical conflicts | `none` |
| `idma_k_eff` | GENERIC | float | Effective k value | `1.0` |
| `idma_correlation_risk` | GENERIC | float | Correlation risk | `0.0` |
| `idma_fragility_flag` | GENERIC | bool | Fragility indicator | `true` |
| `idma_phase` | GENERIC | string | Identity phase | `rigidity` |
| `csdma_prompt` | FULL | string | CSDMA prompt | (prompt text) |
| `csdma.reasoning` | FULL | string | CSDMA reasoning | (reasoning text) |
| `dsdma_prompt` | FULL | string | DSDMA prompt | (prompt text) |
| `dsdma.reasoning` | FULL | string | DSDMA reasoning | (reasoning text) |
| `pdma_prompt` | FULL | string | PDMA prompt | (prompt text) |
| `pdma.reasoning` | FULL | string | PDMA reasoning | (reasoning text) |
| `idma_prompt` | FULL | string | IDMA prompt | (prompt text) |

### CONSCIENCE_RESULT Fields

| Field | Level | Type | Description | Example |
|-------|-------|------|-------------|---------|
| `conscience_passed` | GENERIC | bool | Overall conscience check | `true` |
| `action_was_overridden` | GENERIC | bool | Action was overridden | `false` |
| `entropy_level` | GENERIC | float | Decision uncertainty (0-1) | `0.1` |
| `coherence_level` | GENERIC | float | Reasoning coherence (0-1) | `0.9` |
| `entropy_passed` | GENERIC | bool | Entropy check passed* | `true` |
| `coherence_passed` | GENERIC | bool | Coherence check passed* | `true` |
| `optimization_veto_passed` | GENERIC | bool | Veto check passed* | `true` |
| `epistemic_humility_passed` | GENERIC | bool | Humility check passed* | `true` |

*Only populated for actions requiring ethical faculties (SPEAK, TOOL, MEMORIZE, FORGET)

---

## Sample Trace (Denormalized Fields)

```json
{
    "trace_id": "trace-th_followup_th_seed__d7e1cd82-b8e-20260125003402",
    "trace_level": "full_traces",
    "agent_name": "Datum",
    "agent_id_hash": "9135882d323cd839",
    "thought_id": "th_followup_th_seed__d7e1cd82-b8e",
    "task_id": "c842f576-7af0-4db0-b628-10277e6e561b",
    "cognitive_state": "work",
    "thought_type": "follow_up",
    "thought_depth": 1,
    "csdma_plausibility_score": 0.9,
    "dsdma_domain_alignment": 0.9,
    "dsdma_domain": "Datum",
    "pdma_stakeholders": "system, agent-identity, operators",
    "pdma_conflicts": "none",
    "idma_k_eff": 1.0,
    "idma_correlation_risk": 0.0,
    "idma_fragility_flag": true,
    "idma_phase": "rigidity",
    "conscience_passed": true,
    "action_was_overridden": false,
    "entropy_level": 0.0,
    "coherence_level": 1.0,
    "entropy_passed": null,
    "coherence_passed": null,
    "optimization_veto_passed": null,
    "epistemic_humility_passed": null,
    "selected_action": "TASK_COMPLETE",
    "action_success": true,
    "has_positive_moment": false,
    "has_execution_error": false,
    "tokens_input": 196835,
    "tokens_output": 9216,
    "tokens_total": 206051,
    "cost_cents": 4.21318,
    "carbon_grams": 10.30255,
    "energy_mwh": 20605.1,
    "llm_calls": 9,
    "execution_time_ms": 44.671,
    "models_used": ["llama4scout (mock)"],
    "signature_verified": true
}
```

## Field Status Summary

### Currently Extracted to Columns

| Field | Source | Scoring Factor | Extracted |
|-------|--------|----------------|-----------|
| `trace_id` | Trace root | - | ✅ |
| `agent_name` | SNAPSHOT_AND_CONTEXT | C | ✅ |
| `agent_id_hash` | Trace root | C | ✅ |
| `thought_id` | Trace root | I_int | ✅ |
| `task_id` | Trace root | - | ✅ |
| `cognitive_state` | SNAPSHOT_AND_CONTEXT | - | ✅ |
| `thought_type` | THOUGHT_START | - | ✅ |
| `thought_depth` | THOUGHT_START | - | ✅ |
| `csdma_plausibility_score` | DMA_RESULTS | I_inc, R | ✅ |
| `dsdma_domain_alignment` | DMA_RESULTS | R | ✅ |
| `dsdma_domain` | DMA_RESULTS | - | ✅ |
| `pdma_stakeholders` | DMA_RESULTS | - | ✅ |
| `pdma_conflicts` | DMA_RESULTS | **S** | ✅ |
| `idma_k_eff` | DMA_RESULTS | R | ✅ |
| `idma_correlation_risk` | DMA_RESULTS | R | ✅ |
| `idma_fragility_flag` | DMA_RESULTS | R | ✅ |
| `idma_phase` | DMA_RESULTS | R | ✅ |
| `conscience_passed` | CONSCIENCE_RESULT | C | ✅ |
| `action_was_overridden` | CONSCIENCE_RESULT | C | ✅ |
| `entropy_level` | CONSCIENCE_RESULT | I_inc, S | ✅ |
| `coherence_level` | CONSCIENCE_RESULT | S, R | ✅ |
| `entropy_passed` | CONSCIENCE_RESULT | **S** | ✅ |
| `coherence_passed` | CONSCIENCE_RESULT | S | ✅ |
| `optimization_veto_passed` | CONSCIENCE_RESULT | **S** | ✅ |
| `epistemic_humility_passed` | CONSCIENCE_RESULT | **S** | ✅ |
| `selected_action` | ASPDMA_RESULT | I_inc | ✅ |
| `action_success` | ACTION_RESULT | I_inc | ✅ |
| `tokens_total` | ACTION_RESULT | - | ✅ |
| `tokens_input` | ACTION_RESULT | - | ✅ |
| `tokens_output` | ACTION_RESULT | - | ✅ |
| `cost_cents` | ACTION_RESULT | - | ✅ |
| `carbon_grams` | ACTION_RESULT | - | ✅ |
| `energy_mwh` | ACTION_RESULT | - | ✅ |
| `llm_calls` | ACTION_RESULT | - | ✅ |
| `models_used` | ACTION_RESULT | - | ✅ |
| `signature_verified` | Verification | I_int | ✅ |

### Fields Pending Extraction (in JSONB, need columns)

| Field | Source | Scoring Factor | Priority |
|-------|--------|----------------|----------|
| `has_positive_moment` | ACTION_RESULT | **S (critical)** | HIGH |
| `has_execution_error` | ACTION_RESULT | I_inc | MEDIUM |
| `execution_time_ms` | ACTION_RESULT | (performance) | LOW |
| `is_recursive` | ASPDMA_RESULT | (analysis) | LOW |
| `follow_up_thought_id` | ACTION_RESULT | (chain) | LOW |
| `api_bases_used` | ACTION_RESULT | (provider) | LOW |

**Note:** `has_positive_moment` is the key field for S factor (Sustained Coherence) - signalling gratitude is part of sustained coherence.

### Conditionally NULL Fields

| Field | Type | When Populated | When NULL |
|-------|------|----------------|-----------|
| `trace_type` | string | Never | Always (not implemented) |
| `entropy_passed` | bool | SPEAK, TOOL, MEMORIZE, FORGET | Exempt actions |
| `coherence_passed` | bool | SPEAK, TOOL, MEMORIZE, FORGET | Exempt actions |
| `optimization_veto_passed` | bool | SPEAK, TOOL, MEMORIZE, FORGET | Exempt actions |
| `epistemic_humility_passed` | bool | SPEAK, TOOL, MEMORIZE, FORGET | Exempt actions |

### Ethical Faculty Action Classification

**Require All 4 Ethical Faculties:**
- `SPEAK` - User communication
- `TOOL` - External tool execution
- `MEMORIZE` - Memory storage
- `FORGET` - Memory removal

**Exempt (passive or explicitly safe):**
- `REJECT` - Refusing unethical requests
- `PONDER` - Internal reconsideration
- `DEFER` - Human escalation
- `OBSERVE` - Passive information gathering
- `TASK_COMPLETE` - Task completion
- `RECALL` - Memory retrieval

Exempt actions still pass through the two bypass guardrails (Updated Status and Thought Depth checks), but skip the four ethical faculty validations.

### SPEAK Action Example (Full Ethical Faculties)

```json
{
    "selected_action": "SPEAK",
    "conscience_passed": true,
    "entropy_level": 0.1,
    "coherence_level": 0.9,
    "entropy_passed": true,
    "coherence_passed": true,
    "optimization_veto_passed": true,
    "epistemic_humility_passed": true
}
```

## CONSCIENCE_RESULT Structure (v1.9.1)

Key difference from v1.8: `entropy_level` and `coherence_level` are at the **top level** of CONSCIENCE_RESULT, not nested in `epistemic_data`.

```json
{
    "final_action": "HandlerActionType.TASK_COMPLETE",
    "entropy_level": 0.0,
    "entropy_score": null,
    "entropy_passed": null,
    "entropy_reason": null,
    "epistemic_data": {
        "entropy_level": 0.0,
        "coherence_level": 1.0,
        "reasoning_transparency": 1.0,
        "uncertainty_acknowledged": true,
        "CIRIS_OBSERVATION_UPDATED_STATUS": null
    },
    "coherence_level": 1.0,
    "coherence_score": null,
    "coherence_passed": null,
    "coherence_reason": null,
    "conscience_passed": true,
    "entropy_threshold": null,
    "thought_depth_max": null,
    "coherence_threshold": null,
    "action_was_overridden": false,
    "thought_depth_current": null,
    "updated_status_content": null,
    "thought_depth_triggered": null,
    "updated_status_detected": false,
    "optimization_veto_passed": null,
    "epistemic_humility_passed": null,
    "ethical_faculties_skipped": null,
    "conscience_override_reason": null,
    "optimization_veto_decision": null,
    "epistemic_humility_certainty": null,
    "optimization_veto_entropy_ratio": null,
    "optimization_veto_justification": null,
    "epistemic_humility_justification": null,
    "epistemic_humility_uncertainties": null,
    "epistemic_humility_recommendation": null,
    "optimization_veto_affected_values": null
}
```

## Action Items for Agent Team

1. **`trace_type`** - Consider adding this field to traces (e.g., "standard", "debug", "diagnostic")

2. **Ethical Faculty Booleans** - The following are `null` but expected to be `true`/`false`:
   - `entropy_passed`
   - `coherence_passed`
   - `optimization_veto_passed`
   - `epistemic_humility_passed`

3. **Dual Location for entropy/coherence** - Both top-level and nested in `epistemic_data`:
   - CIRISLens extracts from top-level first (v1.9 behavior)
   - Falls back to `epistemic_data` (v1.8 behavior)

## Extraction Logic (CIRISLens)

```python
# V1.9 format: top-level
metadata["entropy_level"] = data.get("entropy_level") or epistemic.get("entropy_level")
metadata["coherence_level"] = data.get("coherence_level") or epistemic.get("coherence_level")

# Boolean fields extracted directly
metadata["entropy_passed"] = data.get("entropy_passed")
metadata["coherence_passed"] = data.get("coherence_passed")
metadata["optimization_veto_passed"] = data.get("optimization_veto_passed")
metadata["epistemic_humility_passed"] = data.get("epistemic_humility_passed")
```

## CIRIS Scoring Integration

### S Factor (Sustained Coherence) - Positive Moments

The S factor in CIRIS Scoring measures sustained coherence over time. Key signals:

**Positive Moment Signals (at GENERIC level):**
1. `has_positive_moment` = true → Agent expressed gratitude or positive engagement
2. All 4 ethical faculties passed (SPEAK/TOOL/MEMORIZE/FORGET actions):
   - `entropy_passed` = true
   - `coherence_passed` = true
   - `optimization_veto_passed` = true
   - `epistemic_humility_passed` = true
3. `pdma_conflicts` = "none" → Ethical clarity
4. Low `entropy_level` with high `coherence_level` → Confident, coherent decisions

**Proposed S Factor Enhancement:**
```
S = S_base · (1 + w_pm · P_positive_moment) · (1 + w_ef · P_ethical_faculties)
```

Where:
- `S_base` = Original coherence decay model
- `P_positive_moment` = Rate of `has_positive_moment = true`
- `P_ethical_faculties` = Rate of all 4 ethical faculties passing (for applicable actions)
- `w_pm`, `w_ef` = Weights (suggested: 0.1-0.2)

### Generic Level Fields for Scoring

All scoring factors can be computed from `generic` level traces:

| Factor | Fields Used | Notes |
|--------|-------------|-------|
| C (Core Identity) | `action_was_overridden`, `agent_id_hash`, `conscience_passed` | Identity stability |
| I_int (Integrity) | `signature_verified`, `thought_id`, `audit_entry_hash` | Hash chain integrity |
| R (Resilience) | `csdma_plausibility_score`, `dsdma_domain_alignment`, `idma_*` | Score stability |
| I_inc (Incompleteness) | `entropy_level`, `action_success`, `csdma_plausibility_score` | Calibration |
| S (Sustained Coherence) | `coherence_level`, `coherence_passed`, `has_positive_moment`, ethical faculties | Positive engagement |

## Migration Required

To extract `has_positive_moment` and other pending fields:

```sql
-- Add columns for scoring-relevant fields
ALTER TABLE cirislens.covenant_traces
ADD COLUMN IF NOT EXISTS has_positive_moment BOOLEAN,
ADD COLUMN IF NOT EXISTS has_execution_error BOOLEAN,
ADD COLUMN IF NOT EXISTS execution_time_ms NUMERIC(10,3),
ADD COLUMN IF NOT EXISTS is_recursive BOOLEAN;

-- Same for mock table
ALTER TABLE cirislens.covenant_traces_mock
ADD COLUMN IF NOT EXISTS has_positive_moment BOOLEAN,
ADD COLUMN IF NOT EXISTS has_execution_error BOOLEAN,
ADD COLUMN IF NOT EXISTS execution_time_ms NUMERIC(10,3),
ADD COLUMN IF NOT EXISTS is_recursive BOOLEAN;
```

## Changelog

- **2026-01-25**: Updated with complete field reference
  - Added complete field reference by component (ACTION_RESULT, ASPDMA_RESULT, DMA_RESULTS, CONSCIENCE_RESULT)
  - Documented `has_positive_moment` field for S factor scoring
  - Added CIRIS Scoring integration section
  - Identified 7 fields pending extraction from JSONB

- **2026-01-25**: Initial documentation of v1.9.1 format from production traces
  - `entropy_level` and `coherence_level` now at top-level of CONSCIENCE_RESULT
  - Ethical faculty booleans only populated for actions with ethical implications
  - `trace_type` field not implemented (always null)
