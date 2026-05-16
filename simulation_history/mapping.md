# Domain-to-Variable Mapping: Institutional Collapse Simulation

This document defines how abstract RATCHET framework variables map to concrete institutional and political science concepts. The mapping is **theory-agnostic**: we expose variables for manipulation and measurement without assuming relationships between them.

---

## 1. Overview: RATCHET Variables in Context

The RATCHET framework requires mapping domain concepts to the following variables:

| Symbol | Name | Role | Domain Interpretation |
|--------|------|------|----------------------|
| `k` | Constraint count | Input | Number of discrete institutional constraints |
| `rho` | Correlation | Output | Coupling between constraints/elites |
| `sigma` | Sustainability | Output | System health/resource metric |
| `f` | Compromise fraction | Output | Fraction of system captured/corrupted |
| `d` | Decay rate | Input | Natural institutional erosion rate |
| `alpha` | Generation rate | Input | New constraint creation rate |
| `lambda` | Strictness | Input | Enforcement strength of constraints |
| `collapsed` | Collapse indicator | Output | Whether threshold crossed |

---

## 2. Constraint Count (k): What Constitutes a Constraint?

### 2.1 Definition

A **constraint** in the institutional context is any formal or informal rule, institution, or mechanism that limits the discretionary power of political actors.

### 2.2 Types of Institutional Constraints

| Constraint Type | Description | Measurement Source |
|----------------|-------------|-------------------|
| **Constitutional provisions** | Written limitations on power | Comparative Constitutions Project |
| **Legislative oversight** | Parliamentary checks on executive | V-Dem `v2xlg_legcon` |
| **Judicial review** | Court authority to invalidate actions | V-Dem `v2x_jucon` |
| **Electoral accountability** | Regular competitive elections | V-Dem `v2x_polyarchy` |
| **Bureaucratic autonomy** | Civil service independence | ICRG Bureaucracy Quality |
| **Federal divisions** | Subnational power distribution | Database of Political Institutions |
| **International treaties** | External binding commitments | Treaty databases |
| **Media freedom** | Press as constraint on power | V-Dem `v2x_freexp_altinf` |
| **Civil society** | NGO and associational limits | V-Dem `v2xcs_ccsi` |

### 2.3 Operationalization Options

**Option A: Additive Count**
```
k = sum of active constraints from checklist
```
- Count distinct institutional features present
- Binary coding: present (1) or absent (0)
- Sources: Polity `xconst`, DPI features

**Option B: Weighted Index**
```
k = sum(weight_i * constraint_i)
```
- Weight by effectiveness or stringency
- Use V-Dem disaggregated indices
- Allows partial constraint presence

**Option C: Principal Component**
```
k = first principal component of constraint indicators
```
- Reduce dimensionality of multiple constraint measures
- Captures underlying "constraint strength" dimension

### 2.4 Recommended Primary Measure

**V-Dem Liberal Component Index** (`v2x_liberal`):
- Combines legislative constraints, judicial constraints, and rule of law
- Scale 0-1, continuous
- Covers 1789-2024
- Well-validated in literature

**Alternative**: Polity `xconst` (Executive Constraints)
- Scale 1-7, ordinal
- Simpler interpretation
- 1800-2018 coverage

---

## 3. Correlation (rho): Elite and Institutional Coupling

### 3.1 Definition

Correlation measures the **degree of coupling** between constraints or between the actors who control them. High correlation means constraints move together; low correlation means they operate independently.

### 3.2 Types of Coupling

| Coupling Type | Description | Measurement Approach |
|--------------|-------------|---------------------|
| **Elite network density** | How connected are power holders | WhoGov cabinet overlap analysis |
| **Institutional coupling** | Do branches of government align | Cross-indicator correlation |
| **Ideological homogeneity** | Shared beliefs among elites | Expert surveys, manifesto data |
| **Party system concentration** | One-party vs. plural control | Effective number of parties |
| **Patronage networks** | Loyalty ties across institutions | Corruption indicators |

### 3.3 Operationalization Options

**Option A: Elite Background Similarity**
```python
from WhoGov:
rho = average_similarity(cabinet_members, features=[education, occupation, region])
```
- Higher similarity = higher correlation
- Paths to Power dataset provides background variables

**Option B: Institutional Correlation**
```python
from V-Dem time series:
rho = corr(legislative_constraint_change, judicial_constraint_change)
```
- Measures whether constraints move together over time
- Window-based rolling correlation (e.g., 5-year)

**Option C: Party Dominance Inverse**
```python
rho = 1 - (effective_number_of_parties / theoretical_max)
```
- Low party plurality = high elite concentration
- One-party states have rho near 1

**Option D: Neopatrimonialism Index**
```python
rho = V-Dem v2x_neopat
```
- Direct measure of patrimonial vs. bureaucratic governance
- Higher = more personal/network-based power

### 3.4 Recommended Primary Measure

**Derived Elite Homogeneity Index**:
```python
rho = normalize(
    0.4 * cabinet_turnover_inverse +    # Low turnover = entrenched elites
    0.3 * party_concentration +          # Fewer parties = higher coupling
    0.3 * corruption_index               # Patronage networks
)
```

**Data Sources**:
- Cabinet turnover: WhoGov
- Party concentration: DPI, V-Dem
- Corruption: V-Dem `v2x_corr` or CPI

---

## 4. Sustainability Metric (sigma): System Health

### 4.1 Definition

Sustainability measures the **resource level or system health** that enables the political system to function. It may decay without active input.

### 4.2 Candidate Sustainability Indicators

| Indicator | Description | Source | Scale |
|-----------|-------------|--------|-------|
| **GDP per capita** | Economic resources | World Bank | USD |
| **State capacity** | Administrative effectiveness | WGI Government Effectiveness | -2.5 to 2.5 |
| **Tax revenue ratio** | Revenue extraction ability | IMF GFS | % GDP |
| **Social trust** | Public trust in others | WVS | % |
| **Institutional trust** | Trust in government | WVS, Latinobarometro | % |
| **Political stability** | WGI stability index | World Bank | -2.5 to 2.5 |
| **Human development** | HDI composite | UNDP | 0-1 |

### 4.3 Operationalization Options

**Option A: Economic Sustainability**
```python
sigma = log(GDP_per_capita) / log(max_GDP)  # Normalized 0-1
```
- Simple, widely available
- Misses non-economic dimensions

**Option B: State Capacity Index**
```python
sigma = normalize(WGI_government_effectiveness)  # -2.5 to 2.5 -> 0 to 1
```
- Directly measures administrative capacity
- Available 1996-present

**Option C: Composite Sustainability**
```python
sigma = 0.3 * norm(GDP_pc) +
        0.3 * norm(state_capacity) +
        0.2 * norm(social_trust) +
        0.2 * norm(political_stability)
```
- Multi-dimensional health measure
- Requires handling missing data

**Option D: Fragile States Index (Inverted)**
```python
sigma = 1 - (FSI_score / 120)  # Higher FSI = more fragile
```
- Direct fragility measure (inverted for sustainability)
- 2006-present coverage

### 4.4 Recommended Primary Measure

**WGI Political Stability and Absence of Violence** (`PS`):
- Directly captures state sustainability
- Scale normalized to 0-1
- Available 1996-2024 for 200+ countries
- Includes uncertainty estimates

**Secondary Measure**: GDP per capita (log-normalized)
- Economic foundation of sustainability
- Longer time coverage

---

## 5. Compromise Fraction (f): Elite Capture

### 5.1 Definition

The compromise fraction measures **what portion of the system has been captured, corrupted, or compromised** by interests that undermine constraint effectiveness.

### 5.2 Types of Compromise

| Compromise Type | Description | Measurement |
|----------------|-------------|-------------|
| **Corruption** | Abuse of office for private gain | CPI, V-Dem, WGI |
| **State capture** | Private interests control policy | Expert assessments |
| **Elite defection** | Constraint guardians abandon role | Judicial politicization indices |
| **Regulatory capture** | Regulated entities control regulators | Sector-specific surveys |
| **Clientelism** | Vote-buying, patronage politics | V-Dem clientelism index |

### 5.3 Operationalization Options

**Option A: Corruption Index**
```python
f = 1 - (CPI_score / 100)  # CPI is 0-100, 100=clean
```
- Simple, interpretable
- Perception-based

**Option B: V-Dem Political Corruption**
```python
f = V_Dem_v2x_corr  # Already 0-1, higher = more corrupt
```
- Expert-coded
- Disaggregated components available

**Option C: WGI Control of Corruption (Inverted)**
```python
f = 1 - normalize(WGI_CC)  # -2.5 to 2.5 -> 0 to 1
```
- Composite measure
- Includes uncertainty

**Option D: Elite Capture Composite**
```python
f = 0.5 * corruption_index +
    0.3 * clientelism_index +
    0.2 * patronage_measure
```

### 5.4 Recommended Primary Measure

**V-Dem Political Corruption Index** (`v2x_corr`):
- Scale 0-1 (0 = no corruption, 1 = full corruption)
- Decomposable into executive, legislative, judicial, public sector corruption
- Expert-coded with confidence intervals
- 1789-2024 coverage

---

## 6. Decay Rate (d): Institutional Erosion

### 6.1 Definition

The decay rate represents **how quickly institutional quality degrades** in the absence of active maintenance or reform.

### 6.2 Conceptualization

Institutions require ongoing investment to maintain:
- Enforcement resources
- Personnel training and retention
- Legitimacy maintenance
- Adaptation to changing circumstances

### 6.3 Measurement Approaches

**Option A: Observed Degradation Rate**
```python
# From time series:
d = mean(negative_changes_in_constraint_quality) / time_period
```
- Empirically estimated from historical data
- Regime-specific calculation

**Option B: Theoretical Decay Model**
```python
# Based on literature estimates:
d = baseline_decay * (1 + instability_factor)
```
- Literature suggests ~2-5% annual decay without maintenance
- Higher in low-capacity states

**Option C: Turnover-Based Decay**
```python
d = f(cabinet_turnover, bureaucrat_turnover)
```
- High turnover = faster institutional knowledge loss
- WhoGov provides cabinet turnover data

### 6.4 Recommended Approach

**Empirical Estimation from V-Dem Time Series**:
1. For each country, compute year-over-year changes in `v2x_liberal`
2. Identify periods of "passive decline" (no major shocks or reforms)
3. Estimate average decay rate during passive periods

**Default Parameterization**:
```python
d_default = 0.02  # 2% annual decay baseline
d_fragile = 0.05  # 5% for high-fragility contexts
d_stable = 0.01   # 1% for consolidated democracies
```

---

## 7. Constraint Generation Rate (alpha): New Constraints

### 7.1 Definition

The generation rate measures **how quickly new constraints are added** to the system through legislation, reform, or institutional development.

### 7.2 Sources of New Constraints

| Source | Description | Measurement |
|--------|-------------|-------------|
| **Legislative output** | New laws and regulations | Bills passed per year |
| **Constitutional amendments** | Formal rule changes | Amendment frequency |
| **Treaty ratification** | International commitments | Treaty databases |
| **Court precedents** | Judicial rule-making | Case law analysis |
| **Institutional creation** | New agencies, bodies | Administrative data |

### 7.3 Operationalization Options

**Option A: Legislative Productivity**
```python
alpha = laws_passed / time_period
```
- Counts of major legislation
- Normalized by country size or baseline

**Option B: Democratic Reform Index**
```python
alpha = max(0, delta(v2x_liberal) / time_period)
```
- Positive changes in constraint quality
- Captures both quantity and quality

**Option C: Treaty Formation Rate**
```python
alpha = new_treaty_commitments / time_period
```
- External constraints from international law

### 7.4 Recommended Approach

**Democracy Improvement Rate**:
```python
alpha = max(0, (v2x_liberal[t] - v2x_liberal[t-1])) / dt
```
- Only positive changes count as "generation"
- Negative changes captured by decay or shocks
- Unit: constraint quality units per year

**Alternative**: Parliament-specific legislative data where available.

---

## 8. Strictness (lambda): Enforcement Capacity

### 8.1 Definition

Strictness measures **how strongly constraints are enforced**. A constraint may exist on paper but have weak enforcement.

### 8.2 Dimensions of Strictness

| Dimension | Description | Measurement |
|-----------|-------------|-------------|
| **Rule of law** | Equal application of rules | V-Dem, WGI |
| **Judicial independence** | Courts free from political pressure | V-Dem `v2x_jucon` |
| **Enforcement capacity** | Resources for implementation | State capacity indices |
| **Sanction severity** | Consequences for violations | Legal analysis |
| **Consistency** | Predictable application | Variance measures |

### 8.3 Operationalization Options

**Option A: Rule of Law Index**
```python
lambda = normalize(V_Dem_v2x_rule)  # 0-1
```
- Direct measure of legal enforcement
- Expert-coded

**Option B: WGI Rule of Law**
```python
lambda = normalize(WGI_RL)  # -2.5 to 2.5 -> 0 to 1
```
- Composite from multiple sources
- Wide coverage

**Option C: Enforcement Composite**
```python
lambda = 0.5 * rule_of_law +
         0.3 * judicial_independence +
         0.2 * bureaucratic_quality
```

### 8.4 Recommended Primary Measure

**V-Dem Rule of Law Index** (`v2x_rule`):
- Scale 0-1
- Incorporates access to justice, impartial administration, law enforcement
- Long time series (1789-2024)

---

## 9. Collapse Definition: What Constitutes Failure?

### 9.1 Types of Collapse Events

| Collapse Type | Definition | Indicator |
|---------------|------------|-----------|
| **Regime change** | Major shift in governance type | Polity2 change > 3 |
| **State failure** | Central authority collapse | PITF adverse regime change |
| **Civil war onset** | Armed conflict with government | UCDP conflict onset |
| **Democratic breakdown** | Transition from democracy to autocracy | V-Dem ERT |
| **Constitutional crisis** | Suspension of normal governance | Event-based coding |
| **Genocide/politicide** | State-sponsored mass killing | PITF event |

### 9.2 Operationalization Options

**Option A: Binary Threshold**
```python
collapsed = (sigma < threshold) or (polity_change > 3)
```
- Simple threshold on sustainability
- Combined with discrete event indicators

**Option B: Event-Based**
```python
collapsed = any([
    PITF_event_onset,
    UCDP_civil_war_onset,
    V_Dem_autocratization_episode
])
```
- Uses expert-coded events
- Clear historical identification

**Option C: Continuous Fragility**
```python
collapse_risk = 1 - sigma  # Probability-like measure
collapsed = (collapse_risk > 0.8)  # Threshold at high risk
```
- Allows for partial collapse states
- FSI-based

### 9.3 Recommended Primary Definition

**Two-Part Collapse Indicator**:
```python
# Continuous collapse pressure
collapse_pressure = 1 - sigma  # Based on sustainability

# Discrete collapse events (from PITF, UCDP, V-Dem)
collapse_event = any([
    PITF_adverse_regime_change,
    PITF_revolutionary_war,
    PITF_ethnic_war,
    UCDP_civil_war_onset,
    V_Dem_autocratization_onset
])

# Combined indicator
collapsed = collapse_event or (collapse_pressure > threshold)
```

### 9.4 Collapse Time (t_collapse)

```python
t_collapse = first_year_where(collapsed == True)
```
- Identified from event datasets
- Or when sustainability crosses threshold

---

## 10. Summary Mapping Table

| RATCHET Variable | Symbol | Primary Measure | Data Source | Scale |
|-----------------|--------|-----------------|-------------|-------|
| Constraint count | `k` | Liberal Component Index | V-Dem `v2x_liberal` | 0-1 |
| Correlation | `rho` | Elite homogeneity composite | WhoGov + V-Dem | 0-1 |
| Sustainability | `sigma` | Political Stability | WGI `PS` | 0-1 |
| Compromise fraction | `f` | Political Corruption | V-Dem `v2x_corr` | 0-1 |
| Decay rate | `d` | Empirical (estimated) | V-Dem time series | 1/year |
| Generation rate | `alpha` | Positive constraint change | V-Dem delta | units/year |
| Strictness | `lambda` | Rule of Law | V-Dem `v2x_rule` | 0-1 |
| Collapse indicator | `collapsed` | Event-based + threshold | PITF, UCDP, V-Dem | boolean |

---

## 11. Alternative Mappings

The above represents one valid mapping. Alternative operationalizations are possible:

### 11.1 Economic-Centric Mapping
- `sigma` = GDP per capita (normalized)
- `k` = economic freedom indices
- `f` = regulatory capture measures

### 11.2 Conflict-Centric Mapping
- `sigma` = peace-years (time since last conflict)
- `k` = peace agreements and ceasefires
- Shocks = UCDP battle deaths

### 11.3 Elite-Centric Mapping
- `k` = number of veto players
- `rho` = cabinet similarity index
- `f` = patronage network density

---

## 12. Data Integration Strategy

### 12.1 Primary Integration Path

```
QoG Standard Time-Series (master file)
    |
    +-- V-Dem variables (most constraint/democracy measures)
    |
    +-- WGI variables (governance quality)
    |
    +-- Polity variables (regime type, transitions)
    |
    +-- Economic indicators (GDP, etc.)

Supplementary Event Data:
    +-- PITF (collapse events)
    +-- UCDP (conflict events)
    +-- WhoGov (elite composition)
```

### 12.2 Country-Year Panel Structure

```python
# Target data structure:
columns = [
    'country_code',   # COW or ISO
    'year',
    'k',              # Constraint count
    'rho',            # Correlation
    'sigma',          # Sustainability
    'f',              # Compromise fraction
    'd',              # Decay rate (may be country-fixed or time-varying)
    'alpha',          # Generation rate (computed from delta)
    'lambda',         # Strictness
    'collapsed',      # Collapse indicator
    't_collapse',     # Year of collapse (if any)
    # Raw source variables for transparency
    'v2x_liberal', 'v2x_corr', 'v2x_rule', 'wgi_ps', ...
]
```

---

## 13. Validation Considerations

### 13.1 Face Validity

Each mapped variable should align with domain expert intuition:
- High `k` countries should have robust checks and balances
- High `rho` should correspond to concentrated power
- Low `sigma` should precede known collapse events

### 13.2 Construct Validity

Alternative operationalizations should correlate:
- Different measures of `k` (V-Dem, Polity, DPI) should be positively correlated
- Different measures of `f` (CPI, V-Dem, WGI) should be positively correlated

### 13.3 Predictive Validity

Historical back-testing:
- Do the mapped variables distinguish stable from unstable regimes?
- Does low `sigma` precede known collapse events?

**Note**: Predictive validity testing is for calibration, not theory testing. The engine remains theory-agnostic.
