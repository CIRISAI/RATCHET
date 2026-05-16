# Institutional Collapse / State Fragility Simulation Engine

A domain-specific simulation engine for modeling institutional dynamics and state fragility, built on the RATCHET framework.

---

## Overview

This engine simulates the dynamics of political institutions and state fragility. It exposes variables that can be mapped to the RATCHET framework, enabling analysis of institutional evolution, decay, and collapse.

**Key Design Principle**: The engine is **theory-agnostic**. It faithfully models domain dynamics without assuming any particular relationship between variables. Variables are exposed for manipulation and measurement, not for testing theoretical predictions.

---

## Quick Start

### Installation

The engine requires Python 3.9+ and the following dependencies:

```bash
pip install numpy pandas pydantic
```

For data loading:
```bash
pip install openpyxl xlrd  # Excel support
```

### Basic Usage

```python
from simulation_history.engine import (
    InstitutionalCollapseEngine,
    EngineConfig,
    StateVector,
    RegimeType
)

# Create engine with default configuration
config = EngineConfig(
    alpha=0.02,     # Constraint generation rate
    d=0.03,         # Decay rate
    seed=42
)
engine = InstitutionalCollapseEngine(config)

# Initialize from regime archetype
engine.initialize_synthetic(RegimeType.ANOCRACY)

# Run simulation for 50 years
ts = engine.run(duration=50)

# Check outcomes
print(f"Collapsed: {engine.is_collapsed()}")
print(f"Final sustainability: {engine.get_sigma():.3f}")

# Export data
df = ts.to_dataframe()
df.to_csv("simulation_results.csv")
```

### Using Historical Data

```python
from simulation_history.loaders import QoGLoader

# Load QoG Standard Dataset
loader = QoGLoader("path/to/qog_std_ts_jan25.csv")

# Initialize engine from historical country-year
engine.initialize_from_country('VEN', 2000)  # Venezuela, year 2000

# Simulate forward
ts = engine.run(duration=25)
```

---

## RATCHET Variable Mapping

| RATCHET Variable | Domain Concept | Primary Data Source |
|-----------------|----------------|---------------------|
| `k` (constraints) | Institutional constraints (legislative, judicial, constitutional) | V-Dem Liberal Component Index |
| `rho` (correlation) | Elite network coupling, power concentration | WhoGov + party concentration |
| `sigma` (sustainability) | State capacity, political stability | WGI Political Stability |
| `f` (compromise) | Corruption, elite capture | V-Dem Corruption Index |
| `d` (decay rate) | Institutional erosion rate | Empirically estimated |
| `alpha` (generation) | New constraint creation rate | Positive V-Dem changes |
| `lambda` (strictness) | Rule of law, enforcement strength | V-Dem Rule of Law |
| `collapsed` | Regime failure, state collapse | PITF/UCDP events |

See [mapping.md](mapping.md) for detailed operationalization.

---

## Data Sources

### Primary: QoG Standard Dataset
- **URL**: https://www.gu.se/en/quality-government/qog-data
- **Coverage**: 200+ countries, 1946-2024
- **Why**: Integrates V-Dem, WGI, Polity, and other sources in unified format

### Secondary Sources
| Dataset | What it Provides | URL |
|---------|-----------------|-----|
| V-Dem | Democracy dimensions, corruption | https://v-dem.net/data/ |
| WGI | Governance quality indices | https://databank.worldbank.org |
| Polity V | Regime type, transitions | https://www.systemicpeace.org |
| PITF | State failure events | https://www.systemicpeace.org/inscrdata.html |
| UCDP | Conflict events | https://ucdp.uu.se/downloads/ |
| FSI | Fragility scores | https://fragilestatesindex.org |
| WhoGov | Cabinet composition | https://politicscentre.nuffield.ox.ac.uk/whogov-dataset/ |

See [datasets.md](datasets.md) for complete dataset documentation.

---

## Engine Interface

### Manipulable Variables (Inputs)

```python
engine.set_k(0.6)           # Set constraint count
engine.set_alpha(0.03)      # Set generation rate
engine.set_d(0.02)          # Set decay rate
engine.set_lambda(0.7)      # Set strictness

engine.apply_shock(Shock(
    type='economic',
    magnitude=0.3,
    target_variable='sigma'
))

engine.apply_intervention(Intervention(
    type=InterventionType.REFORM,
    intensity=0.5,
    target_variable='k'
))
```

### Measurable Variables (Outputs)

```python
rho = engine.get_rho()           # Correlation/coupling
k_eff = engine.get_k_eff()       # Effective constraint count
sigma = engine.get_sigma()       # Sustainability
f = engine.get_f()               # Compromise fraction
state = engine.get_state()       # Full state vector
collapsed = engine.is_collapsed() # Collapse indicator
t_collapse = engine.get_collapse_time()  # Collapse timing
```

### Simulation

```python
engine.step(dt=1.0)              # Single time step
ts = engine.run(duration=50)     # Run for duration
ts = engine.run_until_collapse(max_duration=100)
df = engine.to_dataframe()       # Export history
```

---

## Configuration Options

```python
config = EngineConfig(
    # Dynamic parameters
    alpha=0.02,                   # Constraint generation rate
    d=0.03,                       # Decay rate

    # Collapse thresholds
    collapse_threshold_sigma=0.2, # Low sustainability threshold
    collapse_threshold_f=0.8,     # High corruption threshold

    # Simulation settings
    dt=1.0,                       # Time step (years)
    noise_sigma=0.01,             # Process noise

    # Data source
    data_source='qog',            # 'qog', 'vdem', 'synthetic'
    data_path='/path/to/data.csv',

    # Reproducibility
    seed=42
)
```

---

## Regime Archetypes

Initialize from empirically-based archetypes:

```python
# Consolidated democracy (high constraints, low corruption)
engine.initialize_synthetic(RegimeType.DEMOCRACY)

# Mixed regime (intermediate values)
engine.initialize_synthetic(RegimeType.ANOCRACY)

# Authoritarian (low constraints, high concentration)
engine.initialize_synthetic(RegimeType.AUTOCRACY)

# Failed state (minimal functionality)
engine.initialize_synthetic(RegimeType.FAILED)
```

---

## Example Analyses

### 1. Collapse Probability by Parameter

```python
from simulation_history.engine import run_parameter_sweep

results = run_parameter_sweep(
    base_config=config,
    parameter_name='d',
    parameter_values=[0.01, 0.02, 0.03, 0.05, 0.10],
    initial_state=initial_state,
    duration=50,
    n_runs=100
)

# Collapse probability by decay rate
print(results.groupby('value')['collapsed'].mean())
```

### 2. Intervention Effectiveness

```python
# Simulate without intervention
engine.initialize_synthetic(RegimeType.ANOCRACY)
ts_baseline = engine.run(50)

# Simulate with reform intervention
engine.reset()
engine.initialize_synthetic(RegimeType.ANOCRACY)
engine.run(10)
engine.apply_intervention(Intervention(
    type=InterventionType.REFORM,
    intensity=0.5,
    target_variable='lambda'
))
ts_intervention = engine.run(40)

# Compare outcomes
```

### 3. Historical Trajectory Comparison

```python
from simulation_history.engine import run_historical_comparison

comparison = run_historical_comparison(
    engine=engine,
    country_code='ARG',
    start_year=1980,
    end_year=2010,
    data_loader=loader
)

# Plot simulated vs actual trajectories
```

---

## File Structure

```
simulation_history/
    README.md            # This file
    datasets.md          # Curated dataset documentation
    mapping.md           # Domain-to-variable mapping
    engine_design.md     # Architecture specification

    # Implementation (to be created)
    engine.py            # Core engine class
    state.py             # State vector management
    parameters.py        # Configuration types
    collapse.py          # Collapse detection
    types.py             # Domain types

    loaders/
        __init__.py
        qog_loader.py    # QoG dataset loader
        vdem_loader.py   # V-Dem loader
        event_loader.py  # PITF/UCDP loader

    examples/
        basic_simulation.py
        historical_calibration.py
        intervention_analysis.py
```

---

## Key Concepts

### What is a "Constraint"?

Any formal or informal mechanism limiting discretionary power:
- Constitutional provisions
- Legislative oversight
- Judicial review
- Electoral accountability
- Bureaucratic rules
- International treaties

### What is "Collapse"?

The engine detects collapse when:
- Sustainability (sigma) falls below threshold, OR
- Compromise fraction (f) exceeds threshold, OR
- A discrete collapse event occurs (from PITF/UCDP)

Collapse types include: regime change, civil war onset, state failure, democratic breakdown.

### Theory-Agnostic Design

The engine does NOT:
- Assume causation between variables
- Predict specific outcomes
- Validate any theoretical claims

The engine DOES:
- Expose variables for manipulation
- Measure variables faithfully
- Record trajectories for analysis
- Allow external hypothesis testing

---

## Data Download Guide

1. **QoG Standard Time-Series** (primary)
   - Go to: https://www.gu.se/en/quality-government/qog-data/data-downloads/standard-dataset
   - Download: "QoG Standard Time-Series (CSV)"
   - Place in: `data/qog_std_ts.csv`

2. **V-Dem** (for detailed democracy measures)
   - Go to: https://v-dem.net/data/the-v-dem-dataset/
   - Download: "Country-Year: V-Dem Full+Others"
   - Or use R package: `vdemdata`

3. **Event Data** (for collapse identification)
   - PITF: https://www.systemicpeace.org/inscrdata.html
   - UCDP: https://ucdp.uu.se/downloads/

---

## Citation

If using this engine for research, please cite:

```bibtex
@software{ratchet_institutional_engine,
  title = {RATCHET Institutional Collapse Simulation Engine},
  year = {2025},
  url = {https://github.com/RATCHET/simulation_history}
}
```

And cite the underlying data sources as specified in [datasets.md](datasets.md).

---

## License

Research use. See main RATCHET repository for license terms.

---

## Related Documentation

- [datasets.md](datasets.md) - Complete dataset catalog with URLs
- [mapping.md](mapping.md) - Variable operationalization details
- [engine_design.md](engine_design.md) - Full architecture specification
- [SIMULATION_REQUIREMENTS.md](../SIMULATION_REQUIREMENTS.md) - RATCHET interface spec
