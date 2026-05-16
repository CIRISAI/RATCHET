# Institutional Collapse / State Fragility Simulation: Curated Datasets

This document provides a curated list of open datasets for building an institutional collapse and state fragility simulation engine. Each dataset is evaluated for its relevance to the RATCHET framework variables.

---

## 1. Political Regime and Democracy Indices

### 1.1 Polity V (Center for Systemic Peace)

| Property | Value |
|----------|-------|
| **URL** | https://www.systemicpeace.org/polityproject.html |
| **Coverage** | 167 countries, 1800-2018 |
| **Update Status** | Funding ended 2020; last update 2018 |
| **License** | Academic use; check terms |
| **Format** | Excel, CSV |

**Description**: The Polity project measures political regime characteristics on a 21-point scale from -10 (hereditary monarchy) to +10 (consolidated democracy). It captures regime authority spectrum and transitions.

**Variables Relevant to RATCHET**:
- `polity2`: Combined polity score (-10 to +10) - maps to system health (sigma)
- `durable`: Regime durability in years - relates to sustainability
- `xrreg`, `xrcomp`, `xropen`: Executive recruitment constraints - maps to constraint count (k)
- `xconst`: Executive constraints - directly maps to constraint strictness (lambda)
- `parreg`, `parcomp`: Political participation regulation - institutional constraints
- Regime transition dates - collapse event timing

**Collapse Event Detection**: Polity codes "regime transitions" including:
- Major democratic breakdowns
- Coups and irregular transfers
- State failures and civil wars

---

### 1.2 V-Dem (Varieties of Democracy)

| Property | Value |
|----------|-------|
| **URL** | https://v-dem.net/data/the-v-dem-dataset/ |
| **Coverage** | 202 countries, 1789-2024 |
| **Update Status** | Active; Version 15 (March 2025) |
| **License** | CC-BY-NC-ND (academic use) |
| **Format** | CSV, R package (`vdemdata`) |

**Description**: The largest democracy dataset with 31+ million data points across 531 indicators. Captures multidimensional aspects of democracy with expert-coded indices.

**Variables Relevant to RATCHET**:
- `v2x_polyarchy`: Electoral democracy index (0-1) - sustainability metric (sigma)
- `v2x_libdem`: Liberal democracy index - constraint effectiveness
- `v2x_rule`: Rule of law index - strictness (lambda)
- `v2xlg_legcon`: Legislative constraints on executive - constraint count (k)
- `v2x_jucon`: Judicial constraints on executive - constraint count (k)
- `v2x_corr`: Political corruption index - compromise fraction (f)
- `v2elfrfair`: Election freedom and fairness - institutional quality
- `v2x_neopat`: Neopatrimonialism index - elite capture measure

**Episodes of Regime Transformation (ERT) Dataset**:
- Identifies autocratization and democratization episodes
- Provides timing and outcome classification
- Essential for collapse event identification

---

## 2. Governance and Institutional Quality

### 2.1 World Bank Worldwide Governance Indicators (WGI)

| Property | Value |
|----------|-------|
| **URL** | https://www.worldbank.org/en/publication/worldwide-governance-indicators |
| **Coverage** | 200+ economies, 1996-2024 |
| **Update Status** | Active; annual updates |
| **License** | CC-BY 4.0 (open) |
| **Format** | Excel, API |

**Description**: Composite governance indicators from 30+ underlying sources using Unobserved Components Model (UCM). Includes standard errors for each estimate.

**Variables Relevant to RATCHET**:
- `VA` (Voice and Accountability): Democratic participation - relates to constraint legitimacy
- `PS` (Political Stability): Absence of violence/terrorism - sustainability (sigma)
- `GE` (Government Effectiveness): State capacity - enforcement capability
- `RQ` (Regulatory Quality): Institutional constraints - strictness (lambda)
- `RL` (Rule of Law): Contract enforcement, property rights - constraint count (k)
- `CC` (Control of Corruption): Corruption control - compromise fraction (f)

**Note**: Each indicator includes 90% confidence intervals, enabling uncertainty quantification.

---

### 2.2 Quality of Government (QoG) Standard Dataset

| Property | Value |
|----------|-------|
| **URL** | https://www.gu.se/en/quality-government/qog-data |
| **Coverage** | 200+ countries, 1946-2024 |
| **Update Status** | Active; 2025 release available |
| **License** | Open access, academic use |
| **Format** | CSV, Stata, SPSS, R |

**Description**: Meta-dataset compiling 2,100+ variables from 100+ sources on governance quality. Provides unified country-year panel structure.

**Key Advantages**:
- Harmonizes identifiers across datasets (COW codes, ISO codes)
- Pre-merged time-series format
- Includes Polity, V-Dem, WGI, and other sources in single file
- Ideal for loader implementation

**Variables Available**: Includes all major governance indicators from constituent datasets, plus unique QoG measures of bureaucratic quality and impartiality.

---

### 2.3 International Country Risk Guide (ICRG)

| Property | Value |
|----------|-------|
| **URL** | https://www.prsgroup.com/explore-our-products/icrg/ |
| **Coverage** | 141 countries, 1984-present |
| **Update Status** | Active; monthly updates |
| **License** | Commercial (academic subscriptions available) |
| **Format** | Excel, API |

**Description**: Monthly political, financial, and economic risk ratings. 22 core variables across three risk categories.

**Variables Relevant to RATCHET**:
- Political Risk (100 points): Government stability, socioeconomic conditions
- `Corruption`: Patronage, nepotism, party-business ties - compromise fraction (f)
- `Law and Order`: Legal system strength - strictness (lambda)
- `Bureaucracy Quality`: Institutional capacity - constraint enforcement
- `Military in Politics`: Non-constitutional power - constraint erosion
- `Democratic Accountability`: Electoral constraint effectiveness

**Composite Risk Categories**:
- Very High Risk: 0-49.9 points (collapse zone)
- High Risk: 50-59.9 points
- Moderate Risk: 60-69.9 points
- Low Risk: 70-79.9 points
- Very Low Risk: 80-100 points

---

## 3. State Fragility and Failure

### 3.1 Fragile States Index (Fund for Peace)

| Property | Value |
|----------|-------|
| **URL** | https://fragilestatesindex.org/global-data/ |
| **Coverage** | 178 countries, 2006-2025 |
| **Update Status** | Active; annual |
| **License** | Open access for research |
| **Format** | Excel, CSV, API |

**Description**: Annual ranking of state fragility based on 12 indicators and 100+ sub-indicators. Lower scores indicate greater stability.

**Variables (12 Core Indicators)**:

*Cohesion Indicators*:
- C1: Security Apparatus - state monopoly on violence
- C2: Factionalized Elites - elite network fragmentation
- C3: Group Grievance - inter-group tensions

*Economic Indicators*:
- E1: Economic Decline - sustainability (sigma)
- E2: Uneven Development - inequality drivers
- E3: Human Flight and Brain Drain - capacity erosion

*Political Indicators*:
- P1: State Legitimacy - constraint acceptance
- P2: Public Services - state capacity
- P3: Human Rights and Rule of Law - strictness (lambda)

*Social Indicators*:
- S1: Demographic Pressures
- S2: Refugees and IDPs - crisis indicators
- X1: External Intervention - shocks and interventions

**Collapse Thresholds**:
- Alert: 90-120 (high fragility)
- Warning: 60-89.9
- Stable: 30-59.9
- Sustainable: 0-29.9

---

### 3.2 Political Instability Task Force (PITF) State Failure Problem Set

| Property | Value |
|----------|-------|
| **URL** | https://www.systemicpeace.org/inscrdata.html |
| **Coverage** | Global, 1955-2018 |
| **Update Status** | Ended 2018; historical archive |
| **License** | Academic use |
| **Format** | Excel |

**Description**: Originally CIA-commissioned dataset identifying 100+ state failure events. Four distinct failure types with onset and end dates.

**State Failure Types**:
1. **Revolutionary Wars** (75 episodes, 599 case-years)
   - Armed conflict seeking regime change

2. **Ethnic Wars** (92 episodes, 999 case-years)
   - Communal violence and ethnic conflict

3. **Adverse Regime Changes** (136 episodes, 353 case-years)
   - Shifts from open to authoritarian systems
   - Revolutionary changes in political elites
   - State dissolution
   - Complete collapse of central authority

4. **Genocides and Politicides** (45 episodes, 289 case-years)
   - State-sponsored mass killing

**Collapse Event Coding**: Each event includes:
- Start year and end year
- Magnitude scores (1-4 scale)
- Geographic scope
- Perpetrator and target group identification

---

### 3.3 Geddes-Wright-Frantz (GWF) Autocratic Regimes Dataset

| Property | Value |
|----------|-------|
| **URL** | https://xmarquez.github.io/democracyData/reference/gwf_all.html |
| **Coverage** | Global autocracies, 1946-2010 |
| **Update Status** | Static; extended versions available |
| **License** | Open academic use |
| **Format** | CSV, R package |

**Description**: Codes autocratic regime types, exit modes, and transition outcomes. Essential for understanding how regimes exit power.

**Variables Relevant to RATCHET**:
- Regime type classification (party, military, personalist, monarchy)
- Regime duration - sustainability measure
- Exit mode (democratization, coup, revolution, insurgency)
- Transition violence levels
- Successor regime type

**Regime Type Effects on Collapse**:
- Military regimes: Most likely to democratize
- Personalist regimes: Least likely to democratize
- Party regimes: Transition often to subsequent autocracy

---

## 4. Conflict Data

### 4.1 UCDP/PRIO Armed Conflict Dataset

| Property | Value |
|----------|-------|
| **URL** | https://ucdp.uu.se/downloads/ |
| **Coverage** | Global, 1946-2024 |
| **Update Status** | Active; Version 25.1 (2025) |
| **License** | Open access |
| **Format** | CSV, Excel, API |

**Description**: World's leading source on organized violence. Includes armed conflict, battle deaths, non-state conflict, and one-sided violence.

**Available Datasets**:
- UCDP/PRIO Armed Conflict Dataset (conflict-year)
- Dyadic Dataset (actor-year)
- Battle-Related Deaths Dataset
- Georeferenced Event Dataset (GED) - daily events with coordinates
- Termination Dataset - how conflicts end

**Variables Relevant to RATCHET**:
- Conflict onset dates - shock events
- Conflict intensity (minor: 25+ deaths, war: 1000+ deaths)
- Conflict type (interstate, civil, extrastate)
- Incompatibility (government, territory)
- Outcome coding (victory, peace agreement, ceasefire)

**Use Cases**:
- External shock identification
- Civil war as collapse indicator
- Intervention effectiveness analysis

---

### 4.2 Correlates of War (COW) Project

| Property | Value |
|----------|-------|
| **URL** | https://correlatesofwar.org/data-sets/ |
| **Coverage** | Global, 1816-2016+ |
| **Update Status** | Active; 2024 update |
| **License** | Open academic use |
| **Format** | CSV |

**Description**: Foundational IR dataset. Includes state system membership, militarized disputes, and war data.

**Key Datasets**:
- **State System Membership** (v2024): State entry/exit dates
- **Militarized Interstate Disputes** (v5.0): Interstate confrontations
- **Intra-State War Data**: Civil wars
- **National Material Capabilities**: Power indicators

**Variables Relevant to RATCHET**:
- State system entry/exit - state creation and failure events
- War onset and termination
- Material capabilities (CINC score) - state capacity proxy

---

## 5. Elite and Power Network Data

### 5.1 WhoGov (Who Governs)

| Property | Value |
|----------|-------|
| **URL** | https://politicscentre.nuffield.ox.ac.uk/whogov-dataset/ |
| **Coverage** | 177 countries, 1966-2023 |
| **Update Status** | Active; Version 3 (July 2024) |
| **License** | Open academic use |
| **Format** | CSV |

**Description**: Yearly data on cabinet members in 177 countries. Largest dataset of governing elites. Winner of 2021 Lijphart/Przeworski/Verba award.

**Variables Relevant to RATCHET**:
- Cabinet composition over time
- Minister tenure and turnover rates
- Gender and party affiliation
- Portfolio distribution

**Elite Network Metrics Derivable**:
- Cabinet turnover rate - institutional stability
- Elite replacement patterns - constraint generation (alpha)
- Portfolio concentration - power distribution
- Tenure patterns - regime personalization

---

### 5.2 Paths to Power (PtP)

| Property | Value |
|----------|-------|
| **URL** | Extension of WhoGov |
| **Coverage** | 141 countries, 1966-2021 |
| **Update Status** | Active |
| **License** | Open academic use |
| **Format** | CSV |

**Description**: Individual-level data on 44,789 cabinet members' backgrounds.

**Variables Relevant to RATCHET**:
- Educational background (elite university attendance)
- Prior occupation (business, military, civil service)
- Social origin indicators
- Career pathways to power

**Elite Network Correlation (rho) Indicators**:
- Educational homogeneity - elite ideological coupling
- Occupational similarity - background correlation
- Institutional pathways - recruitment patterns

---

## 6. Corruption and Governance Quality

### 6.1 Transparency International Corruption Perceptions Index (CPI)

| Property | Value |
|----------|-------|
| **URL** | https://www.transparency.org/en/cpi/2024 |
| **Coverage** | 180 countries, 1995-2024 |
| **Update Status** | Active; annual |
| **License** | Open access |
| **Format** | Excel, CSV |

**Description**: Expert perceptions of public sector corruption. Scale 0-100 (100 = very clean).

**Variables Relevant to RATCHET**:
- CPI Score - inverse maps to compromise fraction (f)
- Year-over-year change - institutional erosion rate
- Regional comparisons

**Note**: Methodology changed in 2012 (scale 0-10 before, 0-100 after).

---

## 7. Social Trust and Cohesion

### 7.1 World Values Survey (WVS)

| Property | Value |
|----------|-------|
| **URL** | https://www.worldvaluessurvey.org/ |
| **Coverage** | 100+ countries, 1981-2022 (7 waves) |
| **Update Status** | Active; Wave 7 complete |
| **License** | Open (registration required) |
| **Format** | SPSS, Stata, R |

**Description**: Global survey on values, beliefs, and social attitudes. 100,000+ researchers have used this data.

**Variables Relevant to RATCHET**:
- `A165`: Generalized trust ("Most people can be trusted?")
- Institutional trust (government, parliament, courts, police)
- Political participation attitudes
- Confidence in state institutions

**Trust as Sustainability (sigma) Component**:
- High-trust societies (Scandinavia): 60%+ trust
- Low-trust societies (Latin America): <10% trust
- Trust correlates with institutional effectiveness

---

## 8. Intervention and Aid Data

### 8.1 AidData

| Property | Value |
|----------|-------|
| **URL** | https://www.aiddata.org/datasets |
| **Coverage** | Global, 1947-2014 (core); specialized extensions to 2021 |
| **Update Status** | Active |
| **License** | Open access |
| **Format** | CSV, GeoQuery API |

**Description**: 70+ datasets on development finance, including 1.5M+ projects from 96 donors.

**Key Datasets**:
- Core Research Database (1947-2013)
- Geocoded Chinese Official Finance (2000-2014)
- Belt and Road Initiative Projects
- Project Evaluation Dataset (20,000+ evaluations)

**Variables Relevant to RATCHET**:
- Aid flows by sector and donor
- Project implementation and outcomes
- Intervention timing and scope
- Conditionality and reform requirements

---

### 8.2 OECD DAC Aid Statistics

| Property | Value |
|----------|-------|
| **URL** | https://stats.oecd.org/Index.aspx?datasetcode=TABLE2a |
| **Coverage** | DAC members, 1960-present |
| **Update Status** | Active |
| **License** | Open access |
| **Format** | CSV, API |

**Description**: Official Development Assistance (ODA) flows from OECD Development Assistance Committee members.

**Intervention Categories**:
- General budget support
- Sector-specific aid
- Governance and civil society programs
- Conflict prevention and resolution

---

## 9. Meta-Dataset Integration

### 9.1 Recommended Loader Strategy

For the simulation engine, use **QoG Standard Time-Series** as the primary data source because:
1. Pre-harmonized country-year panel structure
2. Includes most major governance indicators
3. Unified identifier system (COW, ISO, GW codes)
4. Single download covers multiple sources

**Supplemental Direct Downloads**:
- V-Dem for detailed democracy dimensions
- UCDP for conflict events
- PITF for historical state failures
- WhoGov for elite composition

### 9.2 Coverage Matrix

| Dataset | Time Coverage | Country Coverage | Update Frequency | Primary Variables |
|---------|--------------|------------------|-----------------|-------------------|
| Polity V | 1800-2018 | 167 | Ended | Regime type, constraints |
| V-Dem | 1789-2024 | 202 | Annual | Democracy dimensions |
| WGI | 1996-2024 | 200+ | Annual | Governance quality |
| QoG | 1946-2024 | 200+ | Annual | Meta-compilation |
| FSI | 2006-2025 | 178 | Annual | Fragility indicators |
| PITF | 1955-2018 | Global | Ended | State failure events |
| UCDP | 1946-2024 | Global | Annual | Conflict events |
| WhoGov | 1966-2023 | 177 | Active | Elite composition |
| CPI | 1995-2024 | 180 | Annual | Corruption |
| WVS | 1981-2022 | 100+ | Waves (~5yr) | Social trust |

---

## 10. Data Access Summary

### Open Access (No Registration)
- QoG Standard Dataset
- UCDP Datasets
- Correlates of War
- V-Dem (download page)
- World Bank WGI
- Fragile States Index

### Registration Required
- V-Dem (R package access)
- World Values Survey
- AidData (some datasets)

### Commercial/Institutional Subscription
- ICRG (PRS Group)

---

## References

Davies, S., et al. (2025). "Organized violence 1989-2024." *Journal of Peace Research*, 62(4).

Geddes, B., Wright, J., & Frantz, E. (2014). "Autocratic Breakdown and Regime Transitions." *Perspectives on Politics*, 12(1), 313-331.

Teorell, J., et al. (2025). *The Quality of Government Standard Dataset*, version Jan25. University of Gothenburg.

WhoGov Team. (2024). "Who Governs? A New Global Dataset on Members of Cabinets." *American Political Science Review*, 114(4), 1366-1374.
