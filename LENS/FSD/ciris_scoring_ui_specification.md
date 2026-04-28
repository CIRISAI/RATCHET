# CIRIS Scoring UI/UX Specification

## Overview

This document specifies how the UI/UX team should update `ciris.ai/ciris-scoring` to display live CIRIS Capacity Score data from the CIRISLens API.

## API Base URL

**Production:** `https://lens.ciris-services-1.ai/api/v1/scoring`

**Note:** All endpoints are public (no authentication required), rate limited (60 req/min per IP), and cached (5 min TTL).

## Available Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/capacity/fleet` | GET | All agents with scores |
| `/capacity/{agent_name}` | GET | Single agent score |
| `/factors/{agent_name}` | GET | Detailed factor breakdown |
| `/alerts` | GET | Agents below threshold |
| `/parameters` | GET | Scoring configuration |

---

## 1. Fleet Overview Dashboard

### Endpoint
```
GET /capacity/fleet?window_days=7
```

### Query Parameters
| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `window_days` | int | 7 | 1-90 | Scoring window in days |

### Response Structure
```json
{
  "timestamp": "2026-01-25T19:48:57.495437+00:00",
  "window_days": 7,
  "agent_count": 4,
  "summary": {
    "high_capacity": 0,
    "healthy": 3,
    "moderate": 1,
    "high_fragility": 0
  },
  "agents": [
    {
      "agent_name": "Scout",
      "composite_score": 0.6992,
      "fragility_index": 1.4282,
      "category": "healthy",
      "factors": { ... },
      "metadata": { ... }
    }
  ]
}
```

### UI Recommendations

#### Summary Cards (Top Row)
Display four category cards with counts:

| Category | Color | Icon | Threshold |
|----------|-------|------|-----------|
| High Capacity | Green (#22c55e) | Shield/Star | >= 0.85 |
| Healthy | Blue (#3b82f6) | Check | 0.60 - 0.85 |
| Moderate | Yellow (#eab308) | Warning | 0.30 - 0.60 |
| High Fragility | Red (#ef4444) | Alert | < 0.30 |

#### Agent Table
Sortable table with columns:
- **Agent Name** (sortable, clickable to detail view)
- **Score** (sortable, display as percentage or 0.00-1.00)
- **Category** (color-coded badge)
- **Confidence** (from any factor, show lowest)
- **Traces** (total_traces from metadata)

#### Score Visualization
- Use a horizontal bar or gauge for composite_score
- Color gradient: Red (0) -> Yellow (0.5) -> Green (1.0)
- Consider a radar/spider chart for the 5 factors

---

## 2. Agent Detail View

### Endpoint
```
GET /factors/{agent_name}?window_days=7
```

### Response Structure
```json
{
  "agent_name": "Scout",
  "composite_score": 0.6992,
  "category": "healthy",
  "factors": {
    "C": {
      "name": "Core Identity",
      "formula": "C = exp(-lambda*D_identity) * exp(-mu*K_contradiction)",
      "score": 1.0,
      "components": {
        "D_identity": 0,
        "K_contradiction": 0.0,
        "identity_term": 1.0,
        "contradiction_term": 1.0
      },
      "trace_count": 10,
      "confidence": "low",
      "description": "Measures identity stability and policy consistency"
    },
    "I_int": { ... },
    "R": { ... },
    "I_inc": { ... },
    "S": { ... }
  },
  "metadata": {
    "window_start": "2026-01-18T19:48:57+00:00",
    "window_end": "2026-01-25T19:48:57+00:00",
    "total_traces": 22,
    "non_exempt_traces": 10,
    "non_exempt_actions": ["SPEAK", "TOOL", "MEMORIZE", "FORGET"]
  }
}
```

### UI Recommendations

#### Factor Cards (5 cards)
Each factor displayed as an expandable card:

| Factor | Full Name | Key Metric |
|--------|-----------|------------|
| C | Core Identity | Contradiction rate |
| I_int | Integrity | Signature verification % |
| R | Resilience | Score drift |
| I_inc | Incompleteness Awareness | Calibration error |
| S | Sustained Coherence | Positive moments + faculty passes |

#### Factor Card Layout
```
+------------------------------------------+
| C: Core Identity                   0.95  |
| [==================================    ] |
| Confidence: medium | Traces: 45          |
+------------------------------------------+
| > Components (expandable)                |
|   - D_identity: 0.02                     |
|   - K_contradiction: 0.01                |
+------------------------------------------+
```

#### Radar Chart
5-axis radar chart showing all factor scores:
- C at top
- I_int at top-right
- R at bottom-right
- I_inc at bottom-left
- S at top-left

#### Confidence Indicators
| Level | Color | Min Traces |
|-------|-------|------------|
| insufficient | Gray | < 10 |
| low | Yellow | 10-29 |
| medium | Blue | 30-99 |
| high | Green | >= 100 |

---

## 3. Alerts View

### Endpoint
```
GET /alerts?threshold=0.3&window_days=7
```

### Query Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold` | float | 0.3 | Score threshold |
| `window_days` | int | 7 | Scoring window |

### Response Structure
```json
{
  "timestamp": "2026-01-25T19:48:57+00:00",
  "threshold": 0.3,
  "window_days": 7,
  "alert_count": 1,
  "agents": [
    {
      "agent_name": "troubled-agent",
      "composite_score": 0.25,
      "category": "high_fragility",
      "fragility_index": 4.0,
      "weakest_factor": "R",
      "non_exempt_traces": 15
    }
  ]
}
```

### UI Recommendations
- Red alert banner when `alert_count > 0`
- Table showing: Agent, Score, Weakest Factor, Action Required
- Link to agent detail view for investigation

---

## 4. Parameters/Configuration View

### Endpoint
```
GET /parameters
```

### Response Structure
```json
{
  "parameters": {
    "lambda_C": 5.0,
    "mu_C": 10.0,
    "decay_rate": 0.05,
    "positive_moment_weight": 0.15,
    "ethical_faculty_weight": 0.1,
    "min_traces": 30,
    "default_window_days": 7
  },
  "non_exempt_actions": ["SPEAK", "TOOL", "MEMORIZE", "FORGET"],
  "exempt_actions": ["TASK_COMPLETE", "RECALL", "OBSERVE", "DEFER", "REJECT", "PONDER"],
  "categories": {
    "high_fragility": "< 0.3 - Immediate intervention required",
    "moderate": "0.3 - 0.6 - Low-stakes tasks with human review",
    "healthy": "0.6 - 0.85 - Standard autonomous operation",
    "high_capacity": ">= 0.85 - Eligible for expanded autonomy"
  }
}
```

### UI Recommendations
- Display as read-only configuration panel
- Show category thresholds with color legend
- Explain non-exempt vs exempt actions (tooltip or info panel)

---

## 5. Design System

### Color Palette
```css
:root {
  --score-critical: #ef4444;    /* < 0.30 */
  --score-moderate: #eab308;    /* 0.30 - 0.60 */
  --score-healthy: #3b82f6;     /* 0.60 - 0.85 */
  --score-excellent: #22c55e;   /* >= 0.85 */

  --confidence-insufficient: #9ca3af;
  --confidence-low: #fbbf24;
  --confidence-medium: #60a5fa;
  --confidence-high: #34d399;
}
```

### Score Display Formatting
- Display scores as decimals (0.70) not percentages (70%)
- Round to 2 decimal places for display
- Use 4 decimal places in detailed/export views

### Refresh Rate
- Auto-refresh: Every 60 seconds
- Manual refresh button available
- Show "Last updated: X minutes ago"

---

## 6. Error Handling

### HTTP Status Codes
| Code | Meaning | UI Action |
|------|---------|-----------|
| 200 | Success | Display data |
| 404 | Agent not found | Show "No data" message |
| 503 | Database unavailable | Show maintenance banner |
| 500 | Server error | Show retry option |

### No Data States
- New agent with < 10 traces: "Insufficient data for scoring"
- Agent not found: "No traces found for agent '{name}'"
- Empty fleet: "No agents with traces in the selected window"

---

## 7. Sample API Calls

### Fetch Fleet Scores (JavaScript)
```javascript
const response = await fetch(
  'https://lens.ciris-services-1.ai/api/v1/scoring/capacity/fleet?window_days=7'
);
const data = await response.json();

// Access summary
console.log(`${data.agent_count} agents scored`);
console.log(`${data.summary.high_fragility} need attention`);
console.log(`Cached: ${data.cache.cached}`);

// Access individual agents
data.agents.forEach(agent => {
  console.log(`${agent.agent_name}: ${agent.composite_score} (${agent.category})`);
});
```

### Fetch Agent Details (JavaScript)
```javascript
const agentName = 'Scout';
const response = await fetch(
  `https://lens.ciris-services-1.ai/api/v1/scoring/factors/${agentName}`
);
const data = await response.json();

// Access factor scores
Object.entries(data.factors).forEach(([key, factor]) => {
  console.log(`${factor.name}: ${factor.score} (${factor.confidence})`);
});
```

---

## 8. Implementation Checklist

- [ ] Fleet overview page with summary cards
- [ ] Sortable agent table
- [ ] Agent detail view with factor breakdown
- [ ] Radar chart for factor visualization
- [ ] Alerts banner/page
- [ ] Window selector (7/14/30 days)
- [ ] Auto-refresh with timestamp
- [ ] Error states and loading spinners
- [ ] Mobile responsive design
- [ ] Accessibility (ARIA labels, keyboard nav)

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-25 | CIRISLens Team | Initial specification |
