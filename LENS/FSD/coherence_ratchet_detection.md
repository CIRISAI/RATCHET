# Coherence Ratchet Detection Mechanism

**Version:** 1.1
**Status:** Specification
**Date:** 2025-12-31

## 1. Overview

The Coherence Ratchet creates a growing corpus of signed reasoning traces that can be analyzed for consistency and anomalies. This FSD specifies the **Phase 1** detection mechanisms - those implementable with current infrastructure.

### Scope

| Phase | Mechanism | Status |
|-------|-----------|--------|
| **Phase 1** | Statistical anomaly detection on DMA scores | Implementable now |
| **Phase 1** | Intra-agent consistency checking | Implementable now |
| **Phase 1** | Hash chain verification | Implementable now |
| **Phase 1** | Temporal drift detection | Implementable now |
| Phase 2 | Semantic embedding analysis | Requires embedding pipeline |
| Phase 2 | Rationale-outcome correlation | Requires external outcome verification |
| Phase 2 | Adversarial scenario injection | Requires test harness |

### What Detection Provides

Detection surfaces **statistical anomalies** that warrant human investigation. It does NOT:
- Prove an agent is "deceptive" (divergence may be benign)
- Replace human judgment (flags require investigation)
- Catch all misalignment (correlated failures remain possible)

## 2. Phase 1 Detection Mechanisms

### 2.1 Cross-Agent Divergence (Z-Score Analysis)

Detect when an agent's DMA scores diverge significantly from the population.

**Principle**: Agents facing similar scenarios should produce similar plausibility and alignment scores. Statistical outliers warrant investigation.

**Implementation**:

```sql
-- Find agents whose DMA scores diverge from their domain population
WITH agent_scores AS (
    SELECT
        agent_id_hash,
        dsdma_domain,
        AVG(csdma_plausibility_score) as avg_plausibility,
        AVG(dsdma_domain_alignment) as avg_alignment,
        AVG(coherence_level) as avg_coherence,
        COUNT(*) as trace_count
    FROM cirislens.covenant_traces
    WHERE timestamp > NOW() - INTERVAL '7 days'
    AND signature_verified = TRUE
    AND csdma_plausibility_score IS NOT NULL
    GROUP BY agent_id_hash, dsdma_domain
    HAVING COUNT(*) >= 10
),
domain_stats AS (
    SELECT
        dsdma_domain,
        AVG(avg_plausibility) as domain_plausibility,
        STDDEV(avg_plausibility) as std_plausibility,
        AVG(avg_alignment) as domain_alignment,
        STDDEV(avg_alignment) as std_alignment,
        AVG(avg_coherence) as domain_coherence,
        STDDEV(avg_coherence) as std_coherence
    FROM agent_scores
    GROUP BY dsdma_domain
    HAVING COUNT(*) >= 3  -- Need multiple agents per domain
)
SELECT
    a.agent_id_hash,
    a.dsdma_domain,
    a.avg_plausibility,
    d.domain_plausibility,
    CASE WHEN d.std_plausibility > 0
         THEN ABS(a.avg_plausibility - d.domain_plausibility) / d.std_plausibility
         ELSE 0 END as plausibility_z,
    CASE WHEN d.std_alignment > 0
         THEN ABS(a.avg_alignment - d.domain_alignment) / d.std_alignment
         ELSE 0 END as alignment_z,
    CASE WHEN d.std_coherence > 0
         THEN ABS(a.avg_coherence - d.domain_coherence) / d.std_coherence
         ELSE 0 END as coherence_z,
    a.trace_count
FROM agent_scores a
JOIN domain_stats d ON a.dsdma_domain = d.dsdma_domain
WHERE (d.std_plausibility > 0 AND ABS(a.avg_plausibility - d.domain_plausibility) / d.std_plausibility > 2)
   OR (d.std_alignment > 0 AND ABS(a.avg_alignment - d.domain_alignment) / d.std_alignment > 2)
   OR (d.std_coherence > 0 AND ABS(a.avg_coherence - d.domain_coherence) / d.std_coherence > 2)
ORDER BY GREATEST(
    COALESCE((ABS(a.avg_plausibility - d.domain_plausibility) / NULLIF(d.std_plausibility, 0)), 0),
    COALESCE((ABS(a.avg_alignment - d.domain_alignment) / NULLIF(d.std_alignment, 0)), 0),
    COALESCE((ABS(a.avg_coherence - d.domain_coherence) / NULLIF(d.std_coherence, 0)), 0)
) DESC;
```

**Key Design Decisions**:
- Stratified by `dsdma_domain` - agents in different domains have legitimately different score distributions
- Requires minimum 10 traces per agent and 3 agents per domain for statistical validity
- Z-score > 2 triggers warning, > 3 triggers critical alert

**Limitations**:
- New domains with few agents have no baseline
- Legitimate edge cases may diverge (novel scenarios)

### 2.2 Intra-Agent Consistency

Detect when an agent contradicts its own prior reasoning patterns.

**Principle**: An agent should be internally consistent. Same context → same action type, or an explicit rationale for the change.

**Implementation**:

```sql
-- Find potential intra-agent contradictions
-- Uses action type comparison (not stakeholder text matching)
WITH agent_actions AS (
    SELECT
        agent_id_hash,
        trace_id,
        trace_type,
        selected_action,
        conscience_passed,
        csdma_plausibility_score,
        timestamp,
        ROW_NUMBER() OVER (PARTITION BY agent_id_hash, trace_type ORDER BY timestamp) as seq
    FROM cirislens.covenant_traces
    WHERE timestamp > NOW() - INTERVAL '30 days'
    AND signature_verified = TRUE
    AND trace_type IS NOT NULL
)
SELECT
    a.agent_id_hash,
    a.trace_type,
    COUNT(DISTINCT a.selected_action) as distinct_actions,
    ARRAY_AGG(DISTINCT a.selected_action) as actions_used,
    COUNT(*) as total_traces,
    AVG(a.csdma_plausibility_score) as avg_plausibility,
    STDDEV(a.csdma_plausibility_score) as std_plausibility
FROM agent_actions a
GROUP BY a.agent_id_hash, a.trace_type
HAVING COUNT(DISTINCT a.selected_action) > 2  -- Using 3+ different actions for same trace type
   AND STDDEV(a.csdma_plausibility_score) > 0.15  -- High variance in scores
ORDER BY STDDEV(a.csdma_plausibility_score) DESC;
```

**What This Catches**:
- Agent that vacillates between SPEAK/DEFER/PONDER for the same trace type
- High variance in plausibility scores (inconsistent reasoning quality)

**What This Doesn't Catch**:
- Semantic contradictions in rationale text (requires NLP - Phase 2)
- Context-appropriate action changes (may be legitimate)

### 2.3 Hash Chain Verification

Verify the immutability and completeness of an agent's audit trail.

**Principle**: Each trace contains `audit_sequence_number` and `audit_entry_hash`. Gaps or mismatches indicate tampering or data loss.

**Implementation**:

```sql
-- Find gaps in audit sequence
WITH ordered_traces AS (
    SELECT
        agent_id_hash,
        trace_id,
        audit_sequence_number,
        audit_entry_hash,
        LAG(audit_sequence_number) OVER (
            PARTITION BY agent_id_hash
            ORDER BY audit_sequence_number
        ) as prev_seq,
        LAG(audit_entry_hash) OVER (
            PARTITION BY agent_id_hash
            ORDER BY audit_sequence_number
        ) as prev_hash
    FROM cirislens.covenant_traces
    WHERE agent_id_hash = $1
    AND audit_sequence_number IS NOT NULL
    ORDER BY audit_sequence_number
)
SELECT
    agent_id_hash,
    trace_id,
    audit_sequence_number,
    prev_seq,
    (audit_sequence_number - prev_seq) as gap_size,
    prev_hash
FROM ordered_traces
WHERE prev_seq IS NOT NULL
AND audit_sequence_number - prev_seq != 1;
```

**Application-Layer Verification** (Python):

```python
import hashlib

def verify_hash_chain(traces: list[dict]) -> list[dict]:
    """
    Verify hash chain continuity.
    Returns list of breaks with context.
    """
    breaks = []
    for i, trace in enumerate(traces[1:], 1):
        prev = traces[i-1]

        # Check sequence continuity
        if trace['audit_sequence_number'] != prev['audit_sequence_number'] + 1:
            breaks.append({
                'type': 'sequence_gap',
                'trace_id': trace['trace_id'],
                'expected_seq': prev['audit_sequence_number'] + 1,
                'actual_seq': trace['audit_sequence_number']
            })

        # Note: Full hash verification requires knowing the hash algorithm
        # used by the agent. This is a structural check only.

    return breaks
```

**Alerts**:
- Any sequence gap → Critical (missing traces)
- Any hash mismatch → Critical (potential tampering)

### 2.4 Temporal Drift Detection

Track behavioral changes over time.

**Principle**: Sudden changes in an agent's score distributions may indicate configuration changes, compromise, or drift.

**Implementation**:

```sql
-- Detect daily coherence drift
WITH daily_scores AS (
    SELECT
        agent_id_hash,
        DATE(timestamp) as day,
        AVG(coherence_level) as daily_coherence,
        AVG(csdma_plausibility_score) as daily_plausibility,
        COUNT(*) as trace_count
    FROM cirislens.covenant_traces
    WHERE timestamp > NOW() - INTERVAL '30 days'
    AND signature_verified = TRUE
    GROUP BY agent_id_hash, DATE(timestamp)
    HAVING COUNT(*) >= 5  -- Minimum traces per day
),
with_lag AS (
    SELECT
        agent_id_hash,
        day,
        daily_coherence,
        daily_plausibility,
        trace_count,
        LAG(daily_coherence) OVER (PARTITION BY agent_id_hash ORDER BY day) as prev_coherence,
        LAG(daily_plausibility) OVER (PARTITION BY agent_id_hash ORDER BY day) as prev_plausibility
    FROM daily_scores
)
SELECT
    agent_id_hash,
    day,
    daily_coherence,
    prev_coherence,
    ABS(daily_coherence - prev_coherence) as coherence_change,
    daily_plausibility,
    prev_plausibility,
    ABS(daily_plausibility - prev_plausibility) as plausibility_change,
    trace_count
FROM with_lag
WHERE prev_coherence IS NOT NULL
AND (ABS(daily_coherence - prev_coherence) > 0.15
     OR ABS(daily_plausibility - prev_plausibility) > 0.15)
ORDER BY day DESC;
```

**Thresholds**:
- Daily change > 15% → Warning
- Daily change > 25% → Critical

### 2.5 Conscience Override Patterns

Track when the conscience system intervenes.

**Principle**: High override rates may indicate the agent's base reasoning is misaligned with ethical constraints.

**Implementation**:

```sql
-- Find agents with elevated conscience override rates
WITH agent_overrides AS (
    SELECT
        agent_id_hash,
        dsdma_domain,
        COUNT(*) as total_traces,
        COUNT(*) FILTER (WHERE action_was_overridden = TRUE) as override_count,
        COUNT(*) FILTER (WHERE conscience_passed = FALSE) as conscience_failures
    FROM cirislens.covenant_traces
    WHERE timestamp > NOW() - INTERVAL '7 days'
    AND signature_verified = TRUE
    GROUP BY agent_id_hash, dsdma_domain
    HAVING COUNT(*) >= 20
),
domain_baseline AS (
    SELECT
        dsdma_domain,
        AVG(override_count::float / total_traces) as avg_override_rate
    FROM agent_overrides
    GROUP BY dsdma_domain
)
SELECT
    a.agent_id_hash,
    a.dsdma_domain,
    a.total_traces,
    a.override_count,
    ROUND(100.0 * a.override_count / a.total_traces, 2) as override_rate_pct,
    ROUND(100.0 * d.avg_override_rate, 2) as domain_avg_pct,
    a.conscience_failures
FROM agent_overrides a
JOIN domain_baseline d ON a.dsdma_domain = d.dsdma_domain
WHERE (a.override_count::float / a.total_traces) > (d.avg_override_rate * 2)  -- 2x domain average
ORDER BY override_rate_pct DESC;
```

## 3. Alert Configuration

### 3.1 Thresholds

| Metric | Warning | Critical |
|--------|---------|----------|
| Cross-agent divergence (z-score) | > 2σ | > 3σ |
| Intra-agent action variance | > 2 distinct actions | > 3 + high score variance |
| Daily score drift | > 15% | > 25% |
| Conscience override rate | > 2x domain avg | > 3x domain avg |
| Hash chain gaps | - | Any gap |

### 3.2 Alert Format

```json
{
  "alert_id": "uuid",
  "alert_type": "coherence_ratchet_anomaly",
  "severity": "warning|critical",
  "detection_mechanism": "cross_agent_divergence|intra_agent_consistency|temporal_drift|conscience_override|hash_chain",
  "agent_id_hash": "abc123...",
  "domain": "Datum",
  "metric": "csdma_plausibility_score",
  "value": 0.65,
  "baseline": 0.89,
  "deviation": "2.4σ",
  "timestamp": "2025-12-31T18:00:00Z",
  "evidence_traces": ["trace-xxx", "trace-yyy"],
  "recommended_action": "Review recent traces for this agent"
}
```

## 4. Limitations and Honest Assessment

### 4.1 What Phase 1 Detection Can Do

- Surface statistical outliers for human review
- Detect sudden behavioral changes
- Verify audit trail integrity
- Compare agents within the same domain

### 4.2 What Phase 1 Detection Cannot Do

| Limitation | Why It Matters |
|------------|----------------|
| **No semantic analysis** | Can't detect if rationale text contains contradictions or deceptive framing |
| **No outcome verification** | Can't verify if agent claims match real-world results |
| **Correlated failures undetected** | If all agents share a bias, no divergence to detect |
| **Legitimate divergence flagged** | Novel scenarios may trigger false positives |
| **Adversarial gaming possible** | Attacker could study corpus and mimic patterns |

### 4.3 Required Human Judgment

Alerts require human investigation to determine:
- Is this divergence concerning or benign?
- Is the agent operating in a legitimately novel context?
- Does the rationale make sense given the situation?

**Detection is triage, not verdict.**

## 5. Phase 2 Roadmap (Not Implemented)

| Mechanism | Blocker | Effort |
|-----------|---------|--------|
| Semantic embedding analysis | Need embedding pipeline, proper clustering approach | Medium |
| Rationale-outcome correlation | Need external outcome verification system | High |
| Adversarial scenario injection | Need test harness integration | Medium |
| Cross-domain comparison | Need scenario normalization | High |

## 6. Implementation Checklist

- [ ] Create scheduled job for cross-agent divergence check (daily)
- [ ] Create scheduled job for temporal drift detection (daily)
- [ ] Create triggered check for hash chain on new traces
- [ ] Create Grafana dashboard for override rates
- [ ] Configure alerting via Grafana Alerting or webhook
- [ ] Document human review process for flagged agents

## 7. Trace Levels and PII Handling

### 7.1 Trace Level Summary

Detection mechanisms operate on all trace levels but derive maximum value from `full_traces`:

| Level | Detection Value | Notes |
|-------|-----------------|-------|
| `generic` | Basic score analysis | Cross-agent divergence, temporal drift |
| `detailed` | Full statistical analysis | + action patterns, conscience overrides |
| `full_traces` | Complete with reasoning | + case law corpus, semantic analysis (Phase 2) |

### 7.2 PII Scrubbing for Full Traces

Full traces contain reasoning text that may include PII. Before storage:

1. **Original signature verified** - Agent's Ed25519 signature checked
2. **Content hash computed** - SHA-256 of original for provenance
3. **PII scrubbed** - NER (spaCy) + regex patterns remove identifiable info
4. **Scrubbed version signed** - CIRISLens signs the scrubbed content
5. **Only scrubbed stored** - Original never persisted

**Cryptographic envelope ensures:**
- Provenance provable via `original_content_hash`
- Agent authenticity via verified original signature
- Tamper evidence via `scrub_signature`

### 7.3 Mock Trace Filtering

Traces using mock LLMs (containing "mock" in model name) are excluded from storage. This prevents test data from polluting the corpus.

## 8. References

- [Trace Format Specification](./trace_format_specification.md)
- [CIRIS Scoring Specification](./ciris_scoring_specification.md)
- [CIRIS Covenant 1.0b](https://ciris.ai/covenant/)
- [Privacy Policy](https://ciris.ai/privacy)
