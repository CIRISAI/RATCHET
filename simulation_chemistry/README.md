# Battery Degradation / Electrochemical Systems Simulation Engine

A domain-specific simulation engine for modeling lithium-ion battery degradation dynamics.

---

## Overview

This module provides a simulation engine for battery degradation that exposes manipulable and measurable variables following the RATCHET interface specification. The engine models electrochemical degradation physics (SEI growth, lithium plating, capacity fade) without assuming any particular relationships between variables.

**Key Principle**: This is a domain simulator, not a theory validator. It faithfully models electrochemical dynamics and exposes variables for manipulation and measurement.

---

## Quick Start

```python
from engine import BatteryDegradationEngine, EngineConfig

# Configure a 4-cell battery pack
config = EngineConfig(
    num_cells=4,
    initial_capacity=2.0,  # Ah
    chemistry="NMC",
    soh_collapse_threshold=0.80
)

# Create engine
engine = BatteryDegradationEngine(config)

# Run simulation for 1 year (8760 hours)
results = engine.run(duration=8760, dt=1.0)

# Measure outcomes
print(f"Final SOH (sigma): {engine.get_sigma():.2%}")
print(f"Effective constraints (k_eff): {engine.get_k_eff():.2f}")
print(f"Cross-cell correlation (rho): {engine.get_rho():.3f}")
print(f"Collapsed: {engine.is_collapsed()}")
```

---

## Directory Structure

```
simulation_chemistry/
|-- README.md           # This file
|-- datasets.md         # Curated list of open datasets
|-- mapping.md          # Domain concept to variable mapping
|-- engine_design.md    # Detailed architecture documentation
|-- engine.py           # Main simulation engine (to be implemented)
|-- loader.py           # Dataset loaders (to be implemented)
|-- examples/           # Usage examples (to be created)
    |-- basic_simulation.py
    |-- data_calibration.py
    |-- intervention_study.py
```

---

## Variable Reference

### Manipulable Variables (Inputs)

| Variable | Symbol | Battery Meaning | Method |
|----------|--------|-----------------|--------|
| Constraint count | k | Number of cells in pack | `set_k(k)` |
| Constraint generation rate | alpha | SEI growth rate | `set_alpha(alpha)` |
| Decay rate | d | Calendar aging rate | `set_d(d)` |
| Strictness | lambda | Voltage/temp limit strictness | `set_lambda(lambda_)` |
| External shocks | shock | Thermal, mechanical, electrical abuse | `apply_shock(shock)` |
| Interventions | intervention | Balancing, replacement, regeneration | `apply_intervention(intervention)` |

### Measurable Variables (Outputs)

| Variable | Symbol | Battery Meaning | Method |
|----------|--------|-----------------|--------|
| Pairwise correlation | rho | Cross-cell degradation correlation | `get_rho()` |
| Effective constraint count | k_eff | k / (1 + rho*(k-1)) | `get_k_eff()` |
| Sustainability metric | sigma | Average State of Health (SOH) | `get_sigma()` |
| Compromised fraction | f | 1 - sigma (capacity fade) | `get_f()` |
| System state | state | Full state vector (per-cell) | `get_state()` |
| Collapse indicator | collapsed | SOH below threshold | `is_collapsed()` |
| Collapse time | t_collapse | Time when SOH crossed threshold | `get_collapse_time()` |

---

## Data Sources

See `datasets.md` for full details. Key datasets:

| Dataset | Cells | Key Features | URL |
|---------|-------|--------------|-----|
| NASA Li-ion Aging | 34 | EIS, run-to-failure, multi-temp | [data.nasa.gov](https://data.nasa.gov/dataset/li-ion-battery-aging-datasets) |
| MIT-Stanford | 124 | Fast charging, lifetime variation | [Kaggle](https://www.kaggle.com/datasets/itshpark/data-driven-prediction-of-battery-cycle) |
| CALCE | 100+ | EIS, calendar aging, multi-chemistry | [calce.umd.edu](https://calce.umd.edu/battery-data) |
| Sandia | 86 | Cross-chemistry comparison | [batteryarchive.org](https://www.batteryarchive.org) |
| Oxford | 8-12 | Path dependence, drive cycles | [ora.ox.ac.uk](https://ora.ox.ac.uk/objects/uuid:03ba4b01-cfed-46d3-9b1a-7d4a7bdf6fac) |

---

## Domain Mapping Summary

| Abstract Concept | Battery Interpretation |
|------------------|------------------------|
| Constraint (k) | Active electrode sites, or cells in pack |
| Correlation (rho) | Cross-cell SOH correlation, electrode coupling |
| Sustainability (sigma) | State of Health = Q_current / Q_initial |
| Collapse | SOH < 80% (industry standard) |
| Decay rate (d) | Calendar aging (SEI growth at rest) |
| Generation rate (alpha) | Cyclic aging (SEI growth under load) |
| Strictness (lambda) | Operating window tightness (voltage, temp limits) |

See `mapping.md` for detailed rationale and measurement methods.

---

## Usage Examples

### Manipulating Degradation Rate

```python
# Reduce degradation by tightening operating constraints
engine.set_lambda(0.9)  # Stricter limits

# Or adjust degradation parameters directly
engine.set_alpha(0.0005)  # Slower SEI growth
engine.set_d(0.00005)     # Slower calendar aging
```

### Applying Shocks

```python
from engine import Shock

# Thermal shock (e.g., external heat event)
shock = Shock(type='thermal', magnitude=30, duration=1, target='all')
engine.apply_shock(shock)

# Mechanical damage to specific cell
shock = Shock(type='mechanical', magnitude=0.1, duration=0, target='0')
engine.apply_shock(shock)
```

### Applying Interventions

```python
from engine import Intervention, InterventionType

# Cell balancing
balancing = Intervention(
    type=InterventionType.BALANCING,
    parameters={}
)
engine.apply_intervention(balancing)

# Replace worst cell
replacement = Intervention(
    type=InterventionType.CELL_REPLACEMENT,
    parameters={}
)
engine.apply_intervention(replacement)

# Regeneration (rest period)
regen = Intervention(
    type=InterventionType.REGENERATION,
    parameters={'duration': 48}  # hours
)
engine.apply_intervention(regen)
```

### Loading Real Data

```python
from loader import NASALoader

# Load dataset
loader = NASALoader()
loader.load('/path/to/nasa/data')

# Get available cells
cells = loader.get_cells()

# Get degradation trajectory
cycles, capacity = loader.get_capacity_trajectory(cells[0])

# Calibrate engine from data
engine = BatteryDegradationEngine(EngineConfig())
loader.calibrate_engine(engine, cells[0])
```

---

## Physics Summary

### Degradation Mechanisms Modeled

1. **SEI (Solid Electrolyte Interphase) Growth**
   - Electrolyte decomposition on anode surface
   - Parabolic kinetics (diffusion limited)
   - Temperature and SOC dependent

2. **Calendar Aging**
   - Capacity fade during storage
   - Arrhenius temperature dependence
   - SOC-dependent (higher SOC = faster aging)

3. **Lithium Plating** (optional)
   - Metallic Li deposition at high C-rates
   - Low temperature accelerates
   - High SOC increases risk

### Key Equations

**SEI Growth Rate:**
```
dL_SEI/dt = k_SEI * exp(Ea/R * (1/T_ref - 1/T)) * (1/L_SEI) * f(SOC)
```

**Calendar Aging:**
```
dQ/dt = -d * Q * exp(Ea/R * (1/T_ref - 1/T)) * (1 + SOC)
```

**Effective Constraints:**
```
k_eff = k / (1 + rho * (k - 1))
```

---

## Configuration Options

```python
@dataclass
class EngineConfig:
    # Cell parameters
    num_cells: int = 1
    initial_capacity: float = 2.0  # Ah
    initial_resistance: float = 0.05  # Ohm
    chemistry: str = "NMC"  # NMC, LFP, NCA, LCO

    # Operating constraints
    voltage_min: float = 2.5  # V
    voltage_max: float = 4.2  # V
    temperature_min: float = -20.0  # C
    temperature_max: float = 60.0  # C
    c_rate_max: float = 2.0  # C

    # Degradation parameters
    sei_growth_rate: float = 0.001
    calendar_aging_rate: float = 0.0001
    activation_energy: float = 50000  # J/mol

    # Collapse threshold
    soh_collapse_threshold: float = 0.80

    # Simulation settings
    time_step: float = 1.0  # hours
    log_interval: int = 10
```

---

## Dependencies

```
numpy>=1.20
pandas>=1.3
scipy>=1.7
matplotlib>=3.5  # for visualization examples
```

---

## References

### Key Publications

1. Severson et al. (2019). "Data-driven prediction of battery cycle life before capacity degradation." Nature Energy.

2. Birkl (2017). "Diagnosis and Prognosis of Degradation in Lithium-Ion Batteries." PhD thesis, Oxford.

3. Preger et al. (2020). "Degradation of Commercial Lithium-ion Cells as a Function of Chemistry and Cycling Conditions." J. Electrochem. Soc.

### Dataset Citations

```
NASA Battery Dataset:
B. Saha and K. Goebel (2007). "Battery Data Set", NASA Ames Prognostics Data Repository.

CALCE Dataset:
Center for Advanced Life Cycle Engineering, University of Maryland.
https://calce.umd.edu/battery-data
```

---

## Notes

- This engine models domain physics, not any particular theory about relationships between variables
- All variables are independently manipulable and measurable
- Collapse thresholds are configurable (80% is industry standard, but can be changed)
- Multi-cell support enables correlation (rho) analysis
- Data loaders support calibration from real experimental data
