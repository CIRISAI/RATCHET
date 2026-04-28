# Functional Specification Document: Self-Configuration Service

## Purpose: Pattern Detection and Identity Monitoring

The Self-Configuration Service provides insights about system behavior patterns while monitoring identity variance to ensure the agent stays within safe operational bounds. It detects patterns but does not automatically apply changes - the agent makes its own decisions based on the insights provided.

## Core Components

### The Three Sub-Services

1. **ConfigurationFeedbackLoop** - Detects patterns in system behavior
2. **IdentityVarianceMonitor** - Tracks identity drift from baseline (20% threshold)
3. **GraphTelemetryService** - Provides telemetry data for analysis

### What It Actually Does

- **Pattern Detection Only** - Identifies patterns, stores them as insights
- **Identity Monitoring** - Tracks variance, triggers WA review if >20%
- **No Automatic Changes** - Agent reads insights and decides what to do

## Architecture

### Actual Data Flow

```
TELEMETRY DATA                  PATTERN DETECTION            AGENT INTROSPECTION
┌─────────────────┐            ┌─────────────────┐         ┌──────────────────────┐
│ Audit Events    │────────────│                 │─────────│                      │
│ (Actions)       │            │  Configuration  │         │   Agent reads        │
├─────────────────┤            │  Feedback Loop  │─────────│   insights during    │
│ Metrics         │────────────│                 │         │   DREAM state        │
│ (Performance)   │            │  Detects:       │         │                      │
├─────────────────┤            │  - Temporal     │         │   Makes decisions    │
│ Correlations    │────────────│  - Frequency    │─────────│   based on patterns  │
│ (TSDB nodes)    │            │  - Performance  │         │                      │
└─────────────────┘            │  - Errors       │         └──────────────────────┘
                               │                 │
                               │  Stores as      │         ┌──────────────────────┐
                               │  CONCEPT nodes  │         │  Identity Variance   │
                               └─────────────────┘         │  Monitor tracks      │
                                                           │  drift (WA at 20%)   │
                                                           └──────────────────────┘
```

### Simplified State Flow

```
┌─────────────┐     detect      ┌─────────────┐    store     ┌─────────────┐
│ COLLECTING  │─────patterns────▶│ ANALYZING   │───insights──▶│ MONITORING  │
│ TELEMETRY   │                  │ PATTERNS    │              │ IDENTITY    │
└─────────────┘                  └─────────────┘              └──────┬──────┘
                                                                     │
                                                              variance > 20%
                                                                     ▼
                                                              ┌─────────────┐
                                                              │ WA REVIEW   │
                                                              │ TRIGGERED   │
                                                              └─────────────┘
```

## Core Operations

### 1. Pattern Detection (Every 6 Hours)

The ConfigurationFeedbackLoop detects and stores patterns:

```python
async def analyze_and_adapt():
    # Detect patterns from recent metrics
    patterns = await self._detect_patterns()
    # Types detected:
    # - Temporal (tool usage by hour)
    # - Frequency (dominant/underused actions)
    # - Performance (response time degradation)
    # - Error (recurring errors)

    # Store patterns as insights for agent
    insights_stored = await self._store_pattern_insights(patterns)
    # Creates CONCEPT nodes with:
    # - pattern_type, description, confidence
    # - evidence, metrics, actionable flag

    # Update learning state
    await self._update_learning_state(patterns)
    # Tracks pattern history (last 1000)
```

### 2. Identity Variance Monitoring

The IdentityVarianceMonitor tracks drift from baseline:

```python
async def check_variance():
    # Take identity snapshot
    current_snapshot = await self._take_identity_snapshot(identity)

    # Calculate variance (simple percentage)
    variance = self._calculate_variance(baseline, current_snapshot)
    # Counts attribute differences / total attributes

    # Trigger WA review if > 20%
    if variance > self._variance_threshold:  # 0.20
        await self._trigger_wa_review(variance_report)
        # Creates deferral through WiseBus
        # Does NOT block further changes

    # Store snapshot
    await self._store_identity_snapshot(current_snapshot)
```

### 3. Pattern Storage

Patterns are stored as CONCEPT nodes for agent discovery:

```python
class DetectedPattern:
    pattern_type: PatternType      # TEMPORAL, FREQUENCY, PERFORMANCE, ERROR
    pattern_id: str               # Unique identifier
    description: str              # Human-readable description
    evidence_nodes: List[str]     # Supporting data
    detected_at: datetime         # When detected
    metrics: PatternMetrics       # Occurrence count, averages, etc.

# Stored as GraphNode with type=CONCEPT, actionable=True
# Agent can RECALL these during introspection
```

## Integration Points

### 1. Data Sources

- **Memory Bus**: Queries audit events and telemetry via RECALL
- **Time Series Data**: Retrieves correlations of type AUDIT_EVENT, LOG_ENTRY, METRIC_DATAPOINT
- **Identity Manager**: Provides current identity for variance calculation

### 2. Pattern Storage

All patterns stored as CONCEPT nodes:
- `insight_type: "behavioral_pattern"`
- `pattern_type`: TEMPORAL, FREQUENCY, PERFORMANCE, or ERROR
- `actionable: true` - Agent can act on these
- No automatic configuration changes

### 3. Wise Authority Integration

When variance exceeds 20%:
- Creates WA review request via WiseBus
- Sends variance report as deferral
- **Does not block** further operations
- System continues while under review

## Safety Mechanisms

### 1. Identity Variance Monitoring

- **20% Threshold**: Triggers WA review (monitoring only)
- **Snapshot Frequency**: Every 6 hours during DREAM
- **Simple Calculation**: Attribute differences / total attributes
- **No Enforcement**: Review is triggered but changes not blocked

### 2. Pattern Detection Limits

- **Pattern History**: Keeps last 1000 patterns
- **Confidence Threshold**: 0.7 for storing insights
- **Evidence Limit**: Max 10 nodes per pattern
- **No Automatic Actions**: Patterns stored for agent consideration only

## Actual Configuration Mechanisms

### What Can Be Configured

1. **Adaptive Filter Service**
   - Update filter patterns based on insights
   - Agent decides when to update filters

2. **Graph Configuration Service**
   - Change config values stored in graph
   - Agent makes changes via CONFIG action

### Pattern Types Detected

1. **Temporal Patterns**
   - Tool usage by hour of day
   - Activity patterns over time

2. **Frequency Patterns**
   - Dominant actions (>30% of usage)
   - Underused capabilities (<5 uses/week)

3. **Performance Patterns**
   - Response time degradation (>20% slower)
   - Resource usage trends

4. **Error Patterns**
   - Recurring errors (3+ occurrences)
   - Error clustering by type

## Configuration

### Service Parameters

```python
# SelfConfigurationService
variance_threshold: float = 0.20          # 20% identity drift triggers WA
adaptation_interval_hours: int = 6       # Pattern detection frequency
stabilization_period_hours: int = 24     # Not currently used

# ConfigurationFeedbackLoop
analysis_interval_hours: int = 6         # How often to detect patterns

# IdentityVarianceMonitor
variance_threshold: float = 0.20          # Triggers WA review
snapshot_interval_hours: int = 6         # How often to check variance
max_snapshots: int = 100                  # History retention
```

## Current Metrics

### Pattern Detection
- Patterns detected per cycle: Variable
- Pattern types: 4 (temporal, frequency, performance, error)
- Insights stored: All detected patterns
- Pattern history: Last 1000 patterns retained

### Identity Monitoring
- Variance checked: Every 6 hours
- WA reviews triggered: When variance > 20%
- Snapshots retained: Last 100
- Baseline comparison: Simple percentage calculation

### No Automatic Changes
- Proposals generated: 0 (feature removed)
- Adaptations applied: 0 (feature removed)
- Agent decisions: Based on recalled insights
- Configuration changes: Through standard agent actions

## Summary

The Self-Configuration Service provides pattern detection and identity monitoring without automatic configuration changes. It:

1. **Detects behavioral patterns** every 6 hours
2. **Stores patterns as insights** for agent introspection
3. **Monitors identity variance** against 20% threshold
4. **Triggers WA review** when threshold exceeded
5. **Does not make automatic changes** - agent decides

This design maintains agent autonomy while providing valuable insights about system behavior and ensuring identity stability through monitoring.
