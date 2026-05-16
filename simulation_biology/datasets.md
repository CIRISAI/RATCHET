# Microbial Ecology / Gut Microbiome Datasets

This document catalogs open datasets suitable for gut microbiome simulation and analysis. Each dataset is evaluated for its provision of: time series data, co-occurrence information, perturbation events, intervention data, and health metrics.

---

## 1. Human Microbiome Project (HMP)

### Overview
The Human Microbiome Project explored microbial communities of the human body in healthy and disease states. Two phases (HMP and iHMP) generated over 48TB of data from multiple omics studies.

### Data Access
- **Primary Portal**: https://www.hmpdacc.org/
- **AWS Open Data Registry**: https://registry.opendata.aws/human-microbiome-project/
- **NIH Common Fund**: https://commonfund.nih.gov/hmp

### Datasets Available

#### HMP1 - Healthy Human Reference Dataset
- **Description**: Microbiome and host sequence data from 300 healthy adults
- **Body Sites**: 18 sites across 5 regions (oral, airways, urogenital, skin, gut)
- **Samples**: Over 2,000 metagenomes, 10+ TB of DNA sequence data
- **Sequencing**: 16S rRNA marker gene and whole metagenome shotgun
- **What It Provides**:
  - Species abundance data (taxonomic profiles)
  - Functional gene profiles (MetaPhlAn/HUMAnN)
  - Cross-body-site comparisons
  - Baseline healthy microbiome reference

#### iHMP (HMP2) - Integrative Human Microbiome Project
- **Description**: Longitudinal datasets of host-microbiome dynamics in disease
- **Projects**:
  1. **Pregnancy and Preterm Birth (PTB)**: Vaginal microbiome during pregnancy
  2. **Inflammatory Bowel Diseases (IBD)**: 1,785 stool samples, 651 biopsies, 529 blood samples from 132 individuals over 1 year
  3. **Type 2 Diabetes/Prediabetes (T2D)**: Microbiome dynamics in metabolic disease
- **What It Provides**:
  - **Time series** of microbial abundance
  - Disease onset events
  - Host metabolome data
  - Longitudinal sampling (quarterly for 1+ years)

### Relevance for Simulation
| Feature | Availability | Notes |
|---------|--------------|-------|
| Time series | Yes (iHMP) | Quarterly sampling, 1+ year duration |
| Species abundance | Yes | 16S and metagenomic profiles |
| Co-occurrence data | Derivable | Can compute correlations from abundance matrices |
| Perturbation events | Partial | Disease flares documented in IBD cohort |
| Intervention data | Limited | Some treatment data in disease cohorts |
| Health metrics | Yes | Disease activity scores, biomarkers |

---

## 2. American Gut Project (AGP)

### Overview
The largest citizen science microbiome project, collecting samples from over 15,000 participants worldwide.

### Data Access
- **GitHub Repository**: https://github.com/biocore/American-Gut
- **EBI/ENA**: Project PRJEB11419
- **Qiita**: Study ID 10317
- **FTP**: ftp://ftp.microbio.me/AmericanGut/latest

### Dataset Details
- **Participants**: 11,336 individuals as of mid-2017
- **Samples**: 15,096 microbial sequence samples
- **Geography**: Primarily USA, UK, Australia; 42+ countries total
- **Metadata**: Extensive questionnaires on diet, lifestyle, health, medications

### What It Provides
- Large-scale species abundance data (16S rRNA)
- Diet and lifestyle metadata
- Medication/antibiotic history
- Self-reported health conditions
- Cross-sectional population diversity

### Relevance for Simulation
| Feature | Availability | Notes |
|---------|--------------|-------|
| Time series | Limited | Primarily cross-sectional |
| Species abundance | Yes | OTU/ASV tables in BIOM format |
| Co-occurrence data | Derivable | Large sample size enables robust correlation |
| Perturbation events | Partial | Self-reported antibiotic history |
| Intervention data | Limited | No controlled interventions |
| Health metrics | Partial | Self-reported conditions |

### Citation
McDonald et al. (2018). American Gut: an Open Platform for Citizen Science Microbiome Research. mSystems 3(3): e00031-18.

---

## 3. DIABIMMUNE Study

### Overview
Longitudinal study of infant gut microbiome development, focusing on immune-mediated diseases and antibiotic effects.

### Data Access
- **Primary Portal**: https://diabimmune.broadinstitute.org/
- **Antibiotics Cohort**: https://diabimmune.broadinstitute.org/diabimmune/antibiotics-cohort
- **T1D Cohort 16S Data**: https://diabimmune.broadinstitute.org/diabimmune/t1d-cohort/resources/16s-sequence-data

### Cohorts

#### Main Cohort
- **Participants**: ~1,000 infants from Finland, Estonia, and Russia
- **Sampling**: Monthly stool samples from birth to 3 years
- **Focus**: Type 1 diabetes and autoimmune disease development

#### Antibiotics Cohort
- **Participants**: 39 children followed from birth to 3 years
- **Design**: Approximately half received multiple antibiotic courses
- **Sampling**: Monthly stool samples
- **Analysis**: Strain-level metagenomic analysis

### What It Provides
- **Monthly longitudinal sampling** (highest temporal resolution available)
- Antibiotic exposure records with timing
- Strain-level diversity and stability metrics
- Geographic/cultural variation (Finland vs Estonia vs Russia)
- Immune development biomarkers
- Type 1 diabetes onset events

### Key Research Findings
- Antibiotic treatment decreases diversity and stability
- Transient increases in antibiotic resistance genes post-treatment
- Microbiome "matures" to adult-like composition by age 3
- Geographic differences correlate with autoimmune disease rates

### Relevance for Simulation
| Feature | Availability | Notes |
|---------|--------------|-------|
| Time series | Excellent | Monthly sampling, 36 months |
| Species abundance | Yes | 16S and metagenomic |
| Co-occurrence data | Derivable | Temporal correlation analysis possible |
| Perturbation events | Yes | Antibiotic treatments with timing |
| Intervention data | Yes | Antibiotics as "natural experiment" |
| Health metrics | Yes | T1D onset, immune markers |

---

## 4. curatedMetagenomicData (R/Bioconductor)

### Overview
Standardized, curated collection of human microbiome metagenomic data from 57 published studies.

### Data Access
- **Package Website**: https://waldronlab.io/curatedMetagenomicData/
- **Bioconductor**: https://bioconductor.org/packages/release/data/experiment/html/curatedMetagenomicData.html
- **GitHub**: https://github.com/waldronlab/curatedMetagenomicData

### Installation
```R
if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
BiocManager::install("curatedMetagenomicData")
```

### Dataset Details
- **Samples**: 10,199 samples from 57 datasets
- **Body Sites**: Primarily gut, but includes HMP body sites
- **Processing**: Standardized with MetaPhlAn3 (taxonomy) and HUMAnN3 (function)
- **Format**: (Tree)SummarizedExperiment objects

### Data Types Available
- Gene family abundances
- Marker abundance/presence
- Pathway abundance/coverage
- Relative species abundance
- Curated sample metadata

### What It Provides
- Unified access to multiple published studies
- Standardized processing pipeline
- Cross-study comparison capability
- Functional (metabolic pathway) data
- Manually curated metadata

### Relevance for Simulation
| Feature | Availability | Notes |
|---------|--------------|-------|
| Time series | Variable | Depends on constituent study |
| Species abundance | Yes | Standardized taxonomic profiles |
| Co-occurrence data | Derivable | Cross-study meta-analysis possible |
| Perturbation events | Variable | Depends on constituent study |
| Intervention data | Variable | Some intervention studies included |
| Health metrics | Yes | Disease status in metadata |

### Citation
Pasolli et al. (2017). Accessible, curated metagenomic data through ExperimentHub. Nature Methods 14(11): 1023-1024.

---

## 5. Additional Specialized Datasets

### 5.1 Fecal Microbiota Transplant (FMT) Studies

#### Nature Medicine FMT Dataset (2022)
- **Description**: 316 FMTs across 10 disease indications
- **Analysis**: 1,089 species at strain level, 47,548 metagenome-assembled genomes
- **URL**: https://www.nature.com/articles/s41591-022-01913-0
- **Relevance**: Intervention outcomes, strain engraftment dynamics

#### C. difficile FMT Studies
- Documented donor-recipient dynamics
- Clinical outcome (cure/recurrence) data
- Pre/post-treatment sampling

### 5.2 Gut Microbiome-Metabolome Dataset Collection
- **Description**: Curated data from 14 gut microbiome-metabolome studies
- **URL**: https://www.nature.com/articles/s41522-022-00345-5
- **Tables Provided**:
  - Genus-level abundance
  - Metabolite abundance
  - Metabolite identifiers
  - Sample metadata
- **Relevance**: Functional ecosystem metrics via metabolome

### 5.3 MicrobiomeDB
- **URL**: https://microbiomedb.org/
- **Description**: Aggregated microbiome datasets with query interface
- **Features**: Online analysis tools, data export

### 5.4 Fasting/Starvation Studies
- Multiple studies on prolonged fasting effects
- Identifies "fasting-resistant" vs "fasting-sensitive" species
- Useful for modeling decay dynamics

---

## 6. Dataset Selection Recommendations

### For Time Series Simulation
1. **DIABIMMUNE** - Best temporal resolution (monthly, 3 years)
2. **iHMP IBD Cohort** - Quarterly sampling with disease events
3. **FMT Studies** - Before/after intervention dynamics

### For Correlation/Network Analysis
1. **American Gut Project** - Large sample size for robust correlations
2. **curatedMetagenomicData** - Cross-study meta-analysis
3. **HMP Healthy Cohort** - Baseline healthy correlations

### For Perturbation Modeling
1. **DIABIMMUNE Antibiotics Cohort** - Controlled antibiotic perturbations
2. **FMT Studies** - Dramatic ecosystem restructuring
3. **iHMP Disease Cohorts** - Disease flare events

### For Ecosystem Health Metrics
1. **iHMP** - Disease activity scores, biomarkers
2. **Gut Microbiome-Metabolome Collection** - Metabolic function
3. **curatedMetagenomicData** - Functional pathway data

---

## 7. Data Processing Considerations

### Correlation Methods
Standard correlation methods (Pearson, Spearman) are unreliable for compositional microbiome data due to the constant-sum constraint. Recommended alternatives:

- **SparCC** (Sparse Correlations for Compositional data): Estimates Pearson correlations accounting for compositional structure
- **SPIEC-EASI**: Sparse Inverse Covariance estimation
- **Centered Log-Ratio (CLR) transformation**: Before standard correlation

### Diversity Metrics
- **Alpha diversity**: Shannon index, Simpson index, Chao1
- **Beta diversity**: Bray-Curtis dissimilarity, UniFrac
- **GMHI**: Gut Microbiome Health Index (50-species model)

### Dysbiosis Assessment
No universal threshold exists. Options include:
- Comparison to healthy reference population
- Disease-specific dysbiosis indexes
- Firmicutes/Bacteroidetes ratio (limited utility)
- GMHI scoring

---

## 8. Data Format Standards

Most datasets provide:
- **BIOM format**: Species abundance tables
- **TSV/CSV**: Metadata tables
- **FASTQ**: Raw sequencing reads
- **Processed tables**: OTU/ASV counts, relative abundances

For simulation purposes, the most useful are:
1. Relative abundance matrices (samples x species)
2. Sample metadata with timestamps
3. Event annotations (antibiotics, interventions, diagnoses)
