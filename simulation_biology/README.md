# Microbiome Ecology Simulation Engine

A domain-specific simulation engine for gut microbiome dynamics, designed to expose manipulable and measurable variables for ecosystem analysis.

---

## Overview

This engine simulates microbial community dynamics in the human gut, modeling:

- **Species abundance changes** over time via Lotka-Volterra dynamics
- **Inter-species interactions** (competition, mutualism, neutrality)
- **Perturbations** (antibiotics, diet changes, infections, fasting)
- **Interventions** (probiotics, prebiotics, fecal microbiota transplant)
- **Ecosystem health metrics** (diversity, stability, functional capacity)

The engine is **theory-agnostic**: it faithfully models microbiome dynamics without assuming any particular relationship between variables.

---

## Quick Start

```python
from simulation_biology.engine import (
    MicrobiomeEngine,
    MicrobiomeConfig,
    Shock,
    Intervention,
)

# Create and initialize engine
config = MicrobiomeConfig(random_seed=42)
engine = MicrobiomeEngine(config)
engine.initialize_from_reference("healthy_adult")

# Check initial state
print(f"Species count: {engine.get_k()}")
print(f"Health score: {engine.get_sigma():.3f}")

# Run simulation for 30 days
df = engine.run(duration=30, dt=0.1)

# Apply antibiotic perturbation
engine.apply_shock(Shock(
    shock_type="antibiotic",
    kill_fraction=0.7
))

# Continue simulation and observe recovery
df = engine.run(duration=60, dt=0.1)

# Export results
full_df = engine.to_dataframe()
full_df.to_csv("simulation_results.csv")
```

---

## Variables

### Manipulable Variables (Inputs)

| Variable | Symbol | Microbiome Meaning | Method |
|----------|--------|-------------------|--------|
| Constraint count | `k` | Number of species | `set_k(k)` |
| Generation rate | `alpha` | Colonization rate | `set_alpha(alpha)` |
| Decay rate | `d` | Degradation without substrate | `set_d(d)` |
| Strictness | `lambda` | Interaction strength | `set_lambda(lambda_)` |
| Shocks | - | Perturbations | `apply_shock(shock)` |
| Interventions | - | Therapeutic actions | `apply_intervention(intervention)` |

### Measurable Variables (Outputs)

| Variable | Symbol | Microbiome Meaning | Method |
|----------|--------|-------------------|--------|
| Constraint count | `k` | Number of detected species | `get_k()` |
| Correlation | `rho` | Species co-occurrence | `get_rho()` |
| Effective constraints | `k_eff` | k / (1 + rho*(k-1)) | `get_k_eff()` |
| Sustainability | `sigma` | Ecosystem health (diversity) | `get_sigma()` |
| Compromised fraction | `f` | Pathogen dominance | `get_f()` |
| State vector | - | Full abundance array | `get_state()` |
| Collapse indicator | - | Has ecosystem failed? | `is_collapsed()` |

---

## Files

```
simulation_biology/
    README.md          # This file
    datasets.md        # Curated dataset list with URLs and descriptions
    mapping.md         # Domain concept to variable mapping
    engine_design.md   # Detailed architecture and implementation
```

---

## Datasets

The following open datasets are recommended for initialization and validation:

1. **Human Microbiome Project (HMP)**
   - 2,000+ metagenomes from 300 healthy adults
   - URL: https://www.hmpdacc.org/

2. **DIABIMMUNE**
   - Monthly longitudinal infant gut samples
   - Antibiotic perturbation data
   - URL: https://diabimmune.broadinstitute.org/

3. **American Gut Project**
   - 15,000+ samples from citizen science
   - URL: https://github.com/biocore/American-Gut

4. **curatedMetagenomicData**
   - 10,000+ samples from 57 studies
   - Standardized processing
   - URL: https://waldronlab.io/curatedMetagenomicData/

See `datasets.md` for complete details.

---

## Variable Mapping Summary

| Abstract Concept | Microbiome Interpretation |
|-----------------|--------------------------|
| **Constraint (k)** | Number of detected species (OTUs/ASVs) |
| **Correlation (rho)** | Mean SparCC correlation between species |
| **Sustainability (sigma)** | Normalized Shannon diversity index |
| **Decay rate (d)** | Diversity half-life during fasting (~0.15/day) |
| **Generation rate (alpha)** | Species colonization rate (~0.5/day) |
| **Collapse** | Shannon < 2.0 OR pathogen > 30% abundance |

See `mapping.md` for detailed rationale and alternatives.

---

## Perturbation Types

### Shocks

| Type | Effect |
|------|--------|
| `antibiotic` | Kill fraction of susceptible species |
| `diet_change` | Alter substrate availability |
| `infection` | Introduce pathogen |
| `fasting` | Accelerate decay dynamics |

### Interventions

| Type | Effect |
|------|--------|
| `probiotic` | Introduce beneficial species |
| `prebiotic` | Boost fiber-fermenting species |
| `fmt` | Fecal microbiota transplant |
| `dietary_fiber` | Reduce decay rate |

---

## Configuration

```python
MicrobiomeConfig(
    # What counts as a constraint
    constraint_type=ConstraintType.SPECIES,  # or FUNCTIONAL_GENE, PATHWAY

    # How to measure health
    health_metric=HealthMetric.SHANNON_DIVERSITY,  # or GMHI, SIMPSON

    # Collapse thresholds
    collapse_diversity_threshold=2.0,  # Shannon diversity
    collapse_pathogen_threshold=0.3,   # Max pathogen abundance

    # Dynamics defaults
    default_decay_rate=0.15,           # Per day
    default_generation_rate=0.5,       # Species per day
    default_strictness=1.0,            # Interaction multiplier

    # Reproducibility
    random_seed=42
)
```

---

## Example: Antibiotic Recovery Study

```python
import matplotlib.pyplot as plt
from simulation_biology.engine import MicrobiomeEngine, Shock

# Initialize healthy microbiome
engine = MicrobiomeEngine()
engine.initialize_from_reference("healthy_adult")

# Baseline period
engine.run(duration=14, dt=0.1)

# Antibiotic course (7 days)
engine.apply_shock(Shock(shock_type="antibiotic", kill_fraction=0.8))
engine.run(duration=7, dt=0.1)

# Recovery period
engine.run(duration=90, dt=0.1)

# Analyze
df = engine.to_dataframe()

# Plot diversity recovery
plt.figure(figsize=(10, 4))
plt.plot(df['time'], df['sigma'])
plt.axhline(y=0.5, color='r', linestyle='--', label='Collapse threshold')
plt.xlabel('Time (days)')
plt.ylabel('Ecosystem Health (sigma)')
plt.title('Microbiome Recovery After Antibiotic Treatment')
plt.legend()
plt.savefig('recovery_curve.png')
```

---

## References

### Datasets
- Human Microbiome Project Consortium. (2012). Nature, 486(7402), 207-214.
- McDonald et al. (2018). mSystems, 3(3), e00031-18.
- Vatanen et al. (2016). Science Translational Medicine, 8(343).

### Methods
- Friedman & Alm. (2012). PLOS Computational Biology, 8(9), e1002687. (SparCC)
- Gupta et al. (2020). Nature Communications, 11, 4635. (GMHI)

### Dynamics
- Coyte et al. (2015). Science, 350(6261), 663-666.
- Stein et al. (2013). PLOS Computational Biology, 9(10), e1003265.

---

## License

This simulation engine is part of the RATCHET project.
