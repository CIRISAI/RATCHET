# Domain-to-Variable Mapping: Gut Microbiome Ecology

This document defines how microbial ecology concepts map to the abstract simulation variables specified in SIMULATION_REQUIREMENTS.md. The mapping is designed to faithfully model microbiome dynamics without assuming any particular theoretical relationships.

---

## 1. Constraint Count (k) - What Constitutes a "Constraint"?

### Definition
In the gut microbiome context, a **constraint** is any factor that limits or shapes the microbial community composition. Multiple interpretations are valid depending on the analysis level:

### Option A: Species-Level Constraints
Each distinct microbial species represents a constraint on ecosystem state.

| Attribute | Value |
|-----------|-------|
| **Interpretation** | k = number of distinct species/taxa present |
| **Measurement** | Count of OTUs/ASVs above detection threshold |
| **Typical Range** | 100-1000 species in healthy gut |
| **Data Source** | 16S rRNA or metagenomic taxonomic profiles |
| **Rationale** | Each species occupies a niche and constrains available resources |

### Option B: Functional Gene Constraints
Each functional gene family represents a metabolic constraint.

| Attribute | Value |
|-----------|-------|
| **Interpretation** | k = number of unique functional genes/pathways |
| **Measurement** | Count of detected gene families (e.g., from HUMAnN) |
| **Typical Range** | 1,000-10,000 gene families |
| **Data Source** | Metagenomic functional profiling |
| **Rationale** | Metabolic functions constrain ecosystem capabilities |

### Option C: Metabolic Pathway Constraints
Each active metabolic pathway represents an ecosystem constraint.

| Attribute | Value |
|-----------|-------|
| **Interpretation** | k = number of active metabolic pathways |
| **Measurement** | Count of pathways with coverage > threshold |
| **Typical Range** | 100-500 pathways |
| **Data Source** | MetaCyc pathway analysis |
| **Rationale** | Pathways define metabolic "rules" the ecosystem follows |

### Option D: Ecological Interaction Constraints
Each significant species-species interaction is a constraint.

| Attribute | Value |
|-----------|-------|
| **Interpretation** | k = number of significant pairwise interactions |
| **Measurement** | Count of edges in co-occurrence network |
| **Typical Range** | 500-5000 interactions |
| **Data Source** | SparCC/SPIEC-EASI network analysis |
| **Rationale** | Interactions (competition, mutualism) constrain community structure |

### Recommended Default
**Option A (Species-Level)** is recommended as the primary interpretation because:
- Most widely measured and comparable across studies
- Direct biological interpretation
- Available in all datasets

The engine should support **configurable constraint type** to allow users to select the appropriate level.

---

## 2. Correlation (rho) - How to Measure Inter-Constraint Coupling?

### Definition
Correlation (rho) measures the degree to which constraints are coupled or redundant. High correlation means constraints move together; low correlation means constraints are independent.

### Measurement Methods

#### Method 1: SparCC Correlation (Recommended)
Sparse Correlations for Compositional data, designed specifically for microbiome data.

| Attribute | Value |
|-----------|-------|
| **Formula** | Log-ratio based iterative approximation |
| **Range** | [-1, 1] |
| **Handles Compositionality** | Yes |
| **Software** | SparCC, SCNIC, MicNet |
| **Reference** | Friedman & Alm (2012) PLOS Comp Bio |

**Implementation:**
```python
# Pseudocode for average SparCC correlation
def compute_rho(abundance_matrix):
    sparcc_matrix = run_sparcc(abundance_matrix)
    # Extract upper triangle (excluding diagonal)
    upper_tri = sparcc_matrix[np.triu_indices_from(sparcc_matrix, k=1)]
    return np.mean(np.abs(upper_tri))  # Average absolute correlation
```

#### Method 2: Functional Redundancy
Measures how many species can perform the same metabolic function.

| Attribute | Value |
|-----------|-------|
| **Formula** | rho = 1 - (unique_functions / total_species) |
| **Range** | [0, 1] |
| **Interpretation** | High rho = many species share functions |
| **Data Required** | Functional gene annotations per species |

#### Method 3: Phylogenetic Correlation
Uses evolutionary relatedness as a proxy for functional similarity.

| Attribute | Value |
|-----------|-------|
| **Formula** | Based on phylogenetic distance matrix |
| **Range** | [0, 1] |
| **Interpretation** | Close relatives often have similar niches |
| **Data Required** | 16S rRNA sequences, phylogenetic tree |

### Recommended Default
**SparCC correlation** is recommended because:
- Handles compositional data correctly
- Well-validated in microbiome literature
- Can be computed from standard abundance data

### Aggregation
For a single scalar rho:
- **Mean absolute correlation**: `rho = mean(|corr_ij|)` for all i < j
- **Weighted by abundance**: Weight by geometric mean of species abundances

---

## 3. Sustainability Metric (sigma) - Ecosystem Health Score

### Definition
Sustainability (sigma) represents the overall health, stability, or functional capacity of the gut ecosystem.

### Option A: Alpha Diversity Index

#### Shannon Diversity
| Attribute | Value |
|-----------|-------|
| **Formula** | H = -sum(p_i * log(p_i)) |
| **Range** | [0, log(S)] where S = species count |
| **Normalized** | sigma = H / log(S) for [0, 1] range |
| **Interpretation** | Combines richness and evenness |

#### Simpson Diversity
| Attribute | Value |
|-----------|-------|
| **Formula** | D = 1 - sum(p_i^2) |
| **Range** | [0, 1] |
| **Interpretation** | Probability two random organisms are different species |

### Option B: Gut Microbiome Health Index (GMHI)

A validated multi-species health predictor.

| Attribute | Value |
|-----------|-------|
| **Formula** | Based on 50 health-associated species |
| **Range** | Continuous, typically [-10, 10] |
| **Normalized** | sigma = sigmoid(GMHI) for [0, 1] range |
| **Reference** | Gupta et al. (2020) Nature Communications |
| **Interpretation** | Positive = healthy, Negative = disease-associated |

### Option C: Functional Diversity

Measures metabolic capability of the ecosystem.

| Attribute | Value |
|-----------|-------|
| **Formula** | Count of active metabolic pathways / total known pathways |
| **Range** | [0, 1] |
| **Data Required** | Functional profiling (HUMAnN) |
| **Interpretation** | Metabolic versatility |

### Option D: Stability Score

Based on temporal variance of community composition.

| Attribute | Value |
|-----------|-------|
| **Formula** | sigma = 1 - mean(Bray-Curtis dissimilarity over time) |
| **Range** | [0, 1] |
| **Data Required** | Longitudinal samples |
| **Interpretation** | Temporal stability of community |

### Recommended Default
**Normalized Shannon diversity** is recommended as the primary metric because:
- Universally applicable and understood
- Computable from any abundance data
- Well-correlated with ecosystem function in literature

The engine should support **configurable health metric** selection.

---

## 4. Collapse Definition - When Has the Ecosystem Failed?

### Definition
"Collapse" represents a significant departure from healthy ecosystem function, not just normal fluctuation.

### Threshold-Based Definitions

#### Alpha Diversity Collapse
| Attribute | Value |
|-----------|-------|
| **Threshold** | Shannon diversity < 2.0 (typical healthy: 3-4) |
| **Normalized** | sigma < 0.5 (if normalized to [0,1]) |
| **Interpretation** | Loss of species richness/evenness |

#### Dysbiosis Score Collapse
| Attribute | Value |
|-----------|-------|
| **Method** | Distance from healthy reference centroid |
| **Threshold** | > 2 standard deviations from healthy mean |
| **Interpretation** | Compositional deviation from healthy state |

#### Pathogen Dominance
| Attribute | Value |
|-----------|-------|
| **Threshold** | Any single pathogenic species > 30% relative abundance |
| **Species** | C. difficile, Enterobacteriaceae bloom, etc. |
| **Interpretation** | Opportunistic pathogen overgrowth |

#### Functional Collapse
| Attribute | Value |
|-----------|-------|
| **Threshold** | Loss of > 50% of core metabolic pathways |
| **Core Pathways** | Short-chain fatty acid production, bile acid metabolism |
| **Interpretation** | Loss of essential ecosystem services |

### Clinical Correlation

Map to clinical dysbiosis definitions:
- **Mild dysbiosis**: sigma in [0.3, 0.5], reversible
- **Moderate dysbiosis**: sigma in [0.1, 0.3], intervention needed
- **Severe dysbiosis / Collapse**: sigma < 0.1, clinical symptoms

### Recommended Default
```python
def is_collapsed(state: EcosystemState) -> bool:
    # Multiple criteria, any triggers collapse
    return (
        state.shannon_diversity < 2.0 or  # Diversity collapse
        state.max_pathogen_abundance > 0.3 or  # Pathogen dominance
        state.gmhi < -5.0  # GMHI health collapse
    )
```

The engine should expose the **collapse threshold as a configurable parameter**.

---

## 5. Decay Rate (d) - Natural Degradation Without Input

### Definition
Decay rate (d) represents how quickly the ecosystem degrades without dietary/environmental input. This models what happens during fasting, starvation, or substrate deprivation.

### Biological Basis

During fasting:
1. Dietary substrate (fiber, etc.) depleted within 24-48 hours
2. Fiber-fermenting bacteria (Firmicutes) decline
3. Mucin-degrading bacteria increase (consuming gut lining)
4. Overall diversity may initially increase then crash
5. Proteobacteria often expand (opportunistic)

### Quantification Options

#### Option A: Diversity Half-Life
| Attribute | Value |
|-----------|-------|
| **Formula** | d = ln(2) / t_half |
| **Units** | 1/days |
| **Typical t_half** | 3-7 days without dietary fiber |
| **Interpretation** | Time for diversity to halve |

#### Option B: Biomass Decay Rate
| Attribute | Value |
|-----------|-------|
| **Formula** | d(biomass)/dt = -d * biomass |
| **Units** | 1/days |
| **Typical Range** | 0.1 - 0.5 per day |
| **Interpretation** | Bacterial cell mass loss rate |

#### Option C: Functional Decay
| Attribute | Value |
|-----------|-------|
| **Formula** | d(pathways)/dt = -d * pathways |
| **Units** | 1/days |
| **Interpretation** | Rate of metabolic capability loss |

### Literature Values

From fasting studies:
- Firmicutes decrease ~34% after 10-day fast
- Bacteroidetes decrease ~50% after 10-day fast
- Proteobacteria increase ~6-fold (relative)
- Significant changes detectable by day 3-4

### Recommended Default
| Parameter | Value | Units |
|-----------|-------|-------|
| **d** | 0.15 | 1/days |
| **Interpretation** | ~4.6 day half-life for diversity |

This should be **configurable** and potentially **species-specific** (fasting-resistant vs fasting-sensitive species have different d values).

---

## 6. Constraint Generation Rate (alpha) - Colonization Rate

### Definition
Alpha represents the rate at which new species or functional constraints establish in the ecosystem. This models colonization, recolonization, and probiotic engraftment.

### Biological Processes Modeled

1. **Natural colonization**: New species from diet/environment
2. **Probiotic administration**: Intentional species introduction
3. **FMT engraftment**: Donor strain establishment
4. **Post-antibiotic recovery**: Recolonization after perturbation

### Quantification Options

#### Option A: Species Acquisition Rate
| Attribute | Value |
|-----------|-------|
| **Formula** | alpha = d(species_count)/dt during recovery |
| **Units** | species/day |
| **Typical Range** | 0.5 - 5 species/day post-perturbation |
| **Context** | Infant colonization or post-antibiotic recovery |

#### Option B: Engraftment Probability
| Attribute | Value |
|-----------|-------|
| **Formula** | alpha = P(species establishes | introduced) |
| **Units** | dimensionless probability |
| **Typical Range** | 0.01 - 0.30 depending on species and conditions |
| **Context** | Probiotic/FMT studies |

#### Option C: Niche Filling Rate
| Attribute | Value |
|-----------|-------|
| **Formula** | alpha = d(niche_occupancy)/dt |
| **Units** | 1/days |
| **Interpretation** | Rate at which empty niches are filled |

### Literature Values

**Infant gut colonization:**
- Initial colonization: hours to days
- Core community establishment: 2-3 years
- Alpha (early infancy): ~2-5 new species/week

**Post-antibiotic recovery:**
- Recovery to baseline: 1-6 months
- Alpha: ~0.5-2 species/week during recovery

**FMT engraftment:**
- Donor strain detection: 1-7 days
- Stable engraftment rate: 30-80% of donor strains

### Recommended Default
| Parameter | Value | Units |
|-----------|-------|-------|
| **alpha** | 0.5 | species/day |
| **Context** | Adult gut, normal conditions |

This should be **context-dependent** (higher for infants, post-perturbation, FMT).

---

## 7. Strictness (lambda) - Constraint Enforcement Strength

### Definition
Strictness represents how strongly constraints (species interactions, niche requirements) are enforced. High strictness means violations are quickly corrected; low strictness allows more community flexibility.

### Microbiome Interpretation

| Low Strictness | High Strictness |
|----------------|-----------------|
| Generalist species dominate | Specialist species dominate |
| Flexible metabolic network | Rigid metabolic dependencies |
| High functional redundancy | Low functional redundancy |
| Resilient to perturbation | Sensitive to perturbation |

### Quantification Options

#### Option A: Niche Specificity
| Attribute | Value |
|-----------|-------|
| **Formula** | lambda = mean(niche_breadth)^(-1) |
| **Range** | [0, inf), typically [0.1, 10] |
| **Interpretation** | Inverse of average metabolic flexibility |

#### Option B: Interaction Strength
| Attribute | Value |
|-----------|-------|
| **Formula** | lambda = mean(|interaction_coefficients|) |
| **Range** | [0, inf) |
| **Interpretation** | Strength of competitive/cooperative interactions |

#### Option C: Environmental Sensitivity
| Attribute | Value |
|-----------|-------|
| **Formula** | lambda = d(composition)/d(perturbation) |
| **Range** | [0, inf) |
| **Interpretation** | Responsiveness to environmental change |

### Recommended Default
| Parameter | Value | Units |
|-----------|-------|-------|
| **lambda** | 1.0 | dimensionless |
| **Interpretation** | Moderate constraint enforcement |

---

## 8. Shocks and Interventions

### Shock Types (Perturbations)

| Shock Type | Implementation | Effect |
|------------|----------------|--------|
| **Antibiotic - Broad Spectrum** | Kill 50-90% of bacteria; preferential survival of resistant strains | Diversity crash, pathogen opportunity |
| **Antibiotic - Narrow Spectrum** | Target specific taxa (e.g., metronidazole: anaerobes) | Selective pressure on specific groups |
| **Diet Change** | Alter substrate availability (fiber, protein, fat) | Shift in fermenter populations |
| **Infection** | Introduce pathogen with competitive advantage | Bloom, immune response, inflammation |
| **Inflammation** | Alter gut environment (pH, oxygen, mucus) | Favor Proteobacteria, stress response |
| **Fasting** | Remove dietary substrate | Decay dynamics, mucin degradation |

### Intervention Types

| Intervention | Implementation | Expected Outcome |
|--------------|----------------|------------------|
| **Probiotic** | Introduce beneficial species at specified dose | May or may not engraft |
| **Prebiotic** | Increase substrate for beneficial taxa | Selective growth promotion |
| **FMT** | Replace majority of microbiome with donor | Community restructuring |
| **Dietary Fiber** | Restore substrate | Support fermenter recovery |
| **Narrow-spectrum antimicrobial** | Target specific pathogen | Reduce pathogen, spare commensals |

---

## 9. State Vector Specification

The full system state should include:

```python
@dataclass
class MicrobiomeState:
    # Abundance vector (primary state)
    abundances: np.ndarray  # Shape: (n_species,), relative abundances

    # Derived metrics
    k: int  # Number of detected species
    rho: float  # Average SparCC correlation
    sigma: float  # Sustainability/health metric
    k_eff: float  # Effective constraint count = k / (1 + rho*(k-1))

    # Taxonomic summaries
    phylum_abundances: Dict[str, float]  # Firmicutes, Bacteroidetes, etc.

    # Functional state (if available)
    pathway_coverage: Optional[np.ndarray]  # Metabolic pathway activity

    # Pathogen status
    pathogen_abundances: Dict[str, float]  # Known pathogens

    # Collapse indicators
    collapsed: bool
    collapse_reason: Optional[str]

    # Time
    time: float  # Simulation time
```

---

## 10. Summary Mapping Table

| Abstract Variable | Microbiome Concept | Primary Measurement | Units |
|-------------------|-------------------|---------------------|-------|
| k (constraints) | Species count | Number of detected taxa | count |
| rho (correlation) | Species co-occurrence | Mean SparCC correlation | [-1, 1] |
| sigma (sustainability) | Ecosystem health | Normalized Shannon diversity | [0, 1] |
| d (decay rate) | Starvation dynamics | Diversity half-life | 1/days |
| alpha (generation rate) | Colonization rate | Species per day | species/day |
| lambda (strictness) | Niche specificity | Inverse niche breadth | dimensionless |
| collapsed | Dysbiosis | Multiple thresholds | boolean |
| f (compromised) | Pathogen fraction | Sum of pathogen abundances | [0, 1] |
