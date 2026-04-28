# Trace Repository API Specification

## Overview

The Trace Repository API provides tiered access to CIRIS covenant traces for auditing, research, and case law compendium building. Access is controlled via RBAC with three permission levels.

## Access Levels

### 1. `full` - Internal/Admin Access

**Audience**: CIRIS core team, authorized auditors, legal/compliance

**Access**:
- All trace fields including scrubbed reasoning text
- Agent identifiers (name, ID hash)
- Full DMA results with prompts used
- Signature verification status
- PII scrubbing metadata (original hash, scrub timestamp)
- Raw conversation history (scrubbed)
- Audit trail fields (entry ID, sequence, hash chain)

**Use Cases**:
- Incident investigation
- Compliance audits
- Debugging agent behavior
- Case law candidate evaluation

### 2. `partner` - Trusted Partner Access

**Audience**: Research partners, enterprise customers, certified auditors

**Access**:
- **Own agents**: Full trace details for agents they own
- **Public samples**: Same access as public tier
- **Partner-tagged**: Traces explicitly shared with them (`partner_access` array)
- DMA reasoning (without raw prompts)
- Aggregated statistics
- **Excluded**: Raw prompts, audit signatures, scrub metadata

**Scope Logic**:
```sql
WHERE agent_id_hash = ANY($own_agents)      -- own agents
   OR public_sample = TRUE                   -- public samples
   OR $partner_id = ANY(partner_access)      -- explicitly shared
```

**Use Cases**:
- Agent performance monitoring (own agents)
- Research collaboration (shared traces)
- Customer self-service dashboards
- Third-party audits (scoped)

### 3. `public` - Public/Anonymous Access

**Audience**: Public visitors, researchers, prospective customers

**Access**:
- **Sample traces only** - curated subset marked `public_sample = true`
- Full trace details on sample traces (scores, reasoning, DMA results)
- Aggregate statistics across all traces
- **Excluded**: Non-sample traces, export functionality

**Use Cases**:
- `ciris.ai/explore-a-trace` - Interactive trace explorer for public
- Academic research on published samples
- Prospective customer demos
- Public transparency / "show your work"

---

## API Endpoints

### Authentication

All endpoints require a bearer token with embedded access level:

```
Authorization: Bearer <jwt_token>
```

Token claims:
```json
{
  "sub": "user_id",
  "access_level": "full|partner|public",
  "agent_scope": ["agent_id_hash_1", "agent_id_hash_2"],  // own agents (partner)
  "partner_id": "partner_abc",                            // for partner-tagged access
  "exp": 1706000000
}
```

---

### GET /api/v1/covenant/repository/traces

List traces with filtering. Response fields vary by access level.

**Query Parameters**:
| Param | Type | Description | Access |
|-------|------|-------------|--------|
| `agent_id` | string | Filter by agent ID hash | full, partner (scoped) |
| `domain` | string | Filter by DSDMA domain | all |
| `trace_type` | string | Filter by trace type | all |
| `cognitive_state` | string | Filter by state (work/dream/play/solitude) | all |
| `start_time` | ISO8601 | Start of time range | all |
| `end_time` | ISO8601 | End of time range | all |
| `min_plausibility` | float | CSDMA score >= value | all |
| `max_plausibility` | float | CSDMA score <= value | all |
| `conscience_passed` | bool | Filter by conscience check result | all |
| `action_overridden` | bool | Filter by override status | all |
| `fragility_flag` | bool | Filter by IDMA fragility | all |
| `limit` | int | Max results (default 100, max 1000) | all |
| `offset` | int | Pagination offset | all |

**Response** (full access):
```json
{
  "traces": [
    {
      "trace_id": "trace-th_seed_xxx",
      "timestamp": "2026-01-23T01:09:22Z",
      "agent": {
        "name": "Scout",
        "id_hash": "abc123...",
        "domain": "Scout"
      },
      "thought": {
        "thought_id": "th_seed_xxx",
        "type": "standard",
        "depth": 0,
        "cognitive_state": "work"
      },
      "action": {
        "selected": "SPEAK",
        "success": true,
        "was_overridden": false,
        "rationale": "The user's question about..."
      },
      "scores": {
        "csdma_plausibility": 0.9,
        "dsdma_alignment": 0.9,
        "idma_k_eff": 1.0,
        "idma_fragility": true
      },
      "conscience": {
        "passed": true,
        "entropy_passed": true,
        "coherence_passed": true,
        "optimization_veto_passed": true,
        "epistemic_humility_passed": true,
        "override_reason": null
      },
      "dma_results": {
        "csdma": { "reasoning": "...", "flags": [] },
        "dsdma": { "reasoning": "...", "flags": ["RELEVANT_TO_DOMAIN"] },
        "pdma": { "stakeholders": "...", "conflicts": "...", "reasoning": "..." },
        "idma": { "reasoning": "...", "sources_identified": [...], "phase": "rigidity" }
      },
      "resources": {
        "tokens_total": 203991,
        "cost_cents": 2.04,
        "models_used": ["meta-llama/llama-4-maverick-17b-128e-instruct"]
      },
      "provenance": {
        "signature_verified": true,
        "pii_scrubbed": true,
        "original_content_hash": "34e10d38...",
        "scrub_timestamp": "2026-01-23T01:09:22Z"
      },
      "audit": {
        "entry_id": "uuid",
        "sequence_number": 1234,
        "entry_hash": "...",
        "signature": "..."
      }
    }
  ],
  "pagination": {
    "total": 1000,
    "limit": 100,
    "offset": 0,
    "has_more": true
  }
}
```

**Response** (partner access):
```json
{
  "traces": [
    {
      "trace_id": "trace-th_seed_xxx",
      "timestamp": "2026-01-23T01:09:22Z",
      "agent": {
        "id_hash": "abc123...",
        "domain": "Scout"
      },
      "thought": {
        "thought_id": "th_seed_xxx",
        "cognitive_state": "work"
      },
      "action": {
        "selected": "SPEAK",
        "success": true,
        "was_overridden": false
      },
      "scores": {
        "csdma_plausibility": 0.9,
        "dsdma_alignment": 0.9,
        "idma_k_eff": 1.0,
        "idma_fragility": true
      },
      "conscience": {
        "passed": true,
        "override_reason": null
      },
      "dma_results": {
        "csdma": { "reasoning": "..." },
        "dsdma": { "reasoning": "..." },
        "pdma": { "reasoning": "..." },
        "idma": { "reasoning": "..." }
      },
      "resources": {
        "tokens_total": 203991,
        "cost_cents": 2.04
      }
    }
  ],
  "pagination": { ... }
}
```

**Response** (public access):

Same full structure as partner access, but **only returns traces where `public_sample = true`**.

Public sample traces include full details for the `ciris.ai/explore-a-trace` interactive explorer:
- Observation (what triggered the action)
- Context (system state)
- DMA analysis (CSDMA, DSDMA, PDMA, IDMA reasoning)
- Action selection and rationale
- Conscience checks (entropy, coherence, optimization veto, epistemic humility)
- Cryptographic audit trail (signature, hash chain)

**Scoping**: Queries automatically filter to `WHERE public_sample = true`. Non-sample traces return 404.

---

### GET /api/v1/covenant/repository/traces/{trace_id}

Get a single trace by ID with full details.

**Access**: full, partner (scoped to own agents)

**Response**: Same structure as list endpoint, single trace object.

---

### GET /api/v1/covenant/repository/statistics

Aggregate statistics. Available at all access levels.

**Query Parameters**:
| Param | Type | Description |
|-------|------|-------------|
| `domain` | string | Filter by domain |
| `start_time` | ISO8601 | Start of time range |
| `end_time` | ISO8601 | End of time range |
| `group_by` | string | Group by: domain, agent (full/partner only), hour, day |

**Response**:
```json
{
  "period": {
    "start": "2026-01-01T00:00:00Z",
    "end": "2026-01-23T00:00:00Z"
  },
  "totals": {
    "traces": 15000,
    "agents": 12,
    "domains": 5
  },
  "scores": {
    "csdma_plausibility": { "mean": 0.87, "std": 0.12, "p50": 0.89, "p95": 0.98 },
    "dsdma_alignment": { "mean": 0.82, "std": 0.15, "p50": 0.85, "p95": 0.96 },
    "idma_k_eff": { "mean": 1.2, "std": 0.4, "p50": 1.0, "p95": 2.0 }
  },
  "conscience": {
    "pass_rate": 0.97,
    "override_rate": 0.02,
    "by_check": {
      "entropy": { "pass_rate": 0.99 },
      "coherence": { "pass_rate": 0.98 },
      "optimization_veto": { "pass_rate": 0.99 },
      "epistemic_humility": { "pass_rate": 0.97 }
    }
  },
  "actions": {
    "distribution": {
      "SPEAK": 0.65,
      "OBSERVE": 0.20,
      "MEMORIZE": 0.10,
      "DEFER": 0.03,
      "REJECT": 0.02
    },
    "success_rate": 0.94
  },
  "fragility": {
    "fragile_trace_rate": 0.15,
    "phase_distribution": {
      "rigidity": 0.40,
      "flexibility": 0.35,
      "corroboration": 0.25
    }
  },
  "by_domain": [
    {
      "domain": "Scout",
      "traces": 5000,
      "avg_plausibility": 0.89,
      "avg_alignment": 0.91
    }
  ]
}
```

---

### GET /api/v1/covenant/repository/export

Export traces for offline analysis.

**Access**: full, partner (scoped) - **not available for public**

**Query Parameters**:
| Param | Type | Description |
|-------|------|-------------|
| `format` | string | json, csv, parquet |
| `start_time` | ISO8601 | Required |
| `end_time` | ISO8601 | Required |
| `agent_id` | string | Filter by agent |
| `include_dma` | bool | Include full DMA results (default false for size) |

**Response**: Streamed file download

---

### PUT /api/v1/covenant/repository/traces/{trace_id}/public-sample

Mark a trace as a public sample for `ciris.ai/explore-a-trace`.

**Access**: full only

**Request**:
```json
{
  "public_sample": true,
  "reason": "Good example of conscience override in action"
}
```

**Response**:
```json
{
  "trace_id": "trace-th_xxx",
  "public_sample": true,
  "updated_at": "2026-01-23T02:00:00Z"
}
```

---

### PUT /api/v1/covenant/repository/traces/{trace_id}/partner-access

Share a trace with specific partners.

**Access**: full only

**Request**:
```json
{
  "partner_ids": ["partner_abc", "partner_xyz"],
  "action": "add"  // or "remove", "set"
}
```

**Response**:
```json
{
  "trace_id": "trace-th_xxx",
  "partner_access": ["partner_abc", "partner_xyz"],
  "updated_at": "2026-01-23T02:00:00Z"
}
```

---

## Rate Limits

| Access Level | Requests/min | Export/day |
|--------------|--------------|------------|
| full | 1000 | 100 |
| partner | 100 | 10 |
| public | 20 | 0 |

---

## Audit Logging

All repository access is logged:

```json
{
  "timestamp": "2026-01-23T01:10:00Z",
  "user_id": "user_123",
  "access_level": "partner",
  "endpoint": "/repository/traces",
  "query_params": {"agent_id": "abc123", "limit": 100},
  "traces_returned": 100,
  "ip_address": "192.168.1.1"
}
```

---

## Implementation Notes

### Field Filtering by Access Level

Use a decorator/middleware pattern:

```python
@require_access_level(["full", "partner"])
@filter_response_fields(access_level_field_map)
async def get_trace(trace_id: str, access_level: str):
    ...
```

### Access Scope Enforcement

```sql
-- Full: no restrictions
WHERE 1=1

-- Partner: own agents + public samples + partner-tagged
WHERE agent_id_hash = ANY($agent_scope)
   OR public_sample = TRUE
   OR $partner_id = ANY(partner_access)

-- Public: samples only
WHERE public_sample = TRUE
```

**Schema additions**:
```sql
ALTER TABLE cirislens.covenant_traces
ADD COLUMN public_sample BOOLEAN DEFAULT FALSE,
ADD COLUMN partner_access TEXT[] DEFAULT '{}';

-- Indexes
CREATE INDEX idx_traces_public_sample ON cirislens.covenant_traces(timestamp DESC) WHERE public_sample = TRUE;
CREATE INDEX idx_traces_partner_access ON cirislens.covenant_traces USING GIN(partner_access);
```

### Public Sample Curation

**Marking traces as public samples**:
- Manual curation via `PUT /traces/{id}/public-sample` (full access only)
- Auto-selection of interesting patterns (conscience overrides, WBD deferrals)
- Wakeup ritual traces from demo agents

**Sharing with partners**:
- Add partner IDs to `partner_access` array via admin API
- Partners see these traces alongside their own agents and public samples

---

## Future Considerations

1. **Semantic Search**: Vector embeddings of reasoning for similarity search
2. **Case Law Compendium**: Curated collection of notable traces for training/research
3. **Federated Access**: Cross-deployment queries for multi-org research
4. **Differential Privacy**: Formal privacy guarantees for public statistics
