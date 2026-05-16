# Battery Degradation / Electrochemical Systems - Open Datasets

This document provides a curated list of open datasets for battery degradation simulation and analysis.

---

## 1. NASA Li-ion Battery Aging Datasets

**URL**: https://data.nasa.gov/dataset/li-ion-battery-aging-datasets

**Alternative Downloads**:
- Direct ZIP: https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip
- Kaggle Mirror: https://www.kaggle.com/datasets/patrickfleith/nasa-battery-dataset
- IEEE DataPort: https://ieee-dataport.org/documents/nasa-lithium-ion-battery-dataset

**Description**:
Run-to-failure dataset from NASA Ames Prognostics Center of Excellence (PCoE). Contains 34 18650 cells (2 Ah capacity) cycled to 70-80% of initial capacity at different temperatures.

**What It Provides**:
| Data Type | Available | Details |
|-----------|-----------|---------|
| Capacity vs cycles | Yes | Full discharge capacity tracking |
| Impedance spectroscopy | Yes | EIS measurements at regular intervals |
| Temperature data | Yes | Tested at multiple temperatures |
| Failure/EOL events | Yes | Cycled until 30% capacity fade (EOL) |
| Multi-cell data | Yes | 34 cells with varying conditions |

**Operational Profiles**: Charge, discharge, and EIS at different temperatures. Deep discharge (2.7V) to induce accelerated aging.

**Format**: MATLAB (.mat) files

**Use Cases**: RUL prediction, prognostics algorithm development, SOH estimation

---

## 2. NASA Randomized and Recommissioned Battery Dataset

**URL**: https://data.nasa.gov/dataset/randomized-and-recommissioned-battery-dataset

**Description**:
Accelerated life cycle dataset focusing on varied load levels with battery packs composed of two 18650 cells. 26 battery packs grouped by constant and random loading conditions.

**What It Provides**:
| Data Type | Available | Details |
|-----------|-----------|---------|
| Capacity vs cycles | Yes | Under varied loading |
| Impedance spectroscopy | Yes | Periodic characterization |
| Temperature data | Yes | Thermal measurements |
| Failure/EOL events | Yes | Full lifecycle |
| Multi-cell data | Yes | 26 packs (52 cells) |

**Format**: MATLAB (.mat) files

**Use Cases**: Multi-cell correlation analysis, pack-level degradation modeling

---

## 3. MIT-Stanford Battery Aging Dataset (Severson et al.)

**URL**: https://www.kaggle.com/datasets/itshpark/data-driven-prediction-of-battery-cycle

**Publication**: Severson et al., "Data-driven prediction of battery cycle life before capacity degradation", Nature Energy (2019)

**Description**:
124 commercial LFP/graphite cells cycled under fast-charging conditions with widely varying cycle lives (150-2,300 cycles).

**What It Provides**:
| Data Type | Available | Details |
|-----------|-----------|---------|
| Capacity vs cycles | Yes | Full discharge curves every cycle |
| Impedance spectroscopy | No | Voltage curves only |
| Temperature data | Yes | Controlled environment |
| Failure/EOL events | Yes | Cycled to 80% capacity |
| Multi-cell data | Yes | 124 cells with varied charging protocols |

**Key Feature**: Early-cycle discharge voltage curves (before degradation visible) used for ML-based lifetime prediction.

**Format**: MATLAB (.mat) and Python pickle files

**Use Cases**: Early life prediction, fast-charging optimization

---

## 4. CALCE Battery Datasets

**URL**: https://calce.umd.edu/battery-data

**Alternative**: https://web.calce.umd.edu/batteries/data/

**Description**:
Multiple datasets from University of Maryland's Center for Advanced Life Cycle Engineering.

### 4.1 CS2 Dataset
- 15 prismatic LCO cells (1.1 Ah nominal)
- Cycled at room temperature (2010-2013)
- Widely used SOH estimation benchmark

### 4.2 Storage Life Test with Impedance
- 144 Li-ion cells
- Three SOC levels (0%, 50%, 100%)
- Four temperatures (-40C, -5C, 25C, 50C)
- Capacity testing and impedance every 3 weeks or 3 months

**What It Provides**:
| Data Type | Available | Details |
|-----------|-----------|---------|
| Capacity vs cycles | Yes | Full cycling data |
| Impedance spectroscopy | Yes | Regular EIS measurements |
| Temperature data | Yes | Multi-temperature studies |
| Failure/EOL events | Yes | Full lifecycle |
| Multi-cell data | Yes | Large cell population |

**Chemistries**: LCO, LFP, NMC

**Form Factors**: Cylindrical, pouch, prismatic

**Format**: CSV and proprietary formats

**Use Cases**: Calendar aging, storage degradation, temperature effects

---

## 5. Sandia National Labs Battery Datasets

**URL**: https://www.batteryarchive.org (search for "SNL")

**Alternative**: https://www.sandia.gov/ess/tools-resources/rd-data-repository

**Description**:
Multiple chemistry comparison datasets from Sandia National Laboratories.

### 5.1 Comparative Electrochemical Performance (2017)
- 24 18650 cells
- Four chemistries: LCO, LFP, NCA, NMC
- Different temperatures and discharge rates
- EIS: 0.1 Hz to 100 KHz

### 5.2 Degradation Study Dataset
- 86 cells total: 30 LFP, 24 NCA, 32 NMC
- Cycled to 80% capacity
- A123 LFP (1.1 Ah), Panasonic NCA (3.2 Ah), LG Chem NMC (3 Ah)

**What It Provides**:
| Data Type | Available | Details |
|-----------|-----------|---------|
| Capacity vs cycles | Yes | Voltage, current, capacity |
| Impedance spectroscopy | Yes | Broadband EIS |
| Temperature data | Yes | Multi-temperature |
| Failure/EOL events | Yes | 80% capacity EOL |
| Multi-cell data | Yes | Cross-chemistry comparison |

**Format**: CSV (time series and cycle files)

**Use Cases**: Cross-chemistry degradation comparison, abuse testing

---

## 6. Oxford Battery Degradation Datasets

**URL**: https://ora.ox.ac.uk/objects/uuid:03ba4b01-cfed-46d3-9b1a-7d4a7bdf6fac

### 6.1 Oxford Battery Degradation Dataset 1
- 8 Li-ion pouch cells
- 40C thermal chamber
- CC-CV charging, Artemis drive cycle discharge
- Characterization every 100 cycles

### 6.2 Path Dependent Degradation Datasets
- **Part 1**: https://ora.ox.ac.uk/objects/uuid:de62b5d2-6154-426d-bcbb-30253ddb7d1e
- **Part 3**: https://ora.ox.ac.uk/objects/uuid:78f66fa8-deb9-468a-86f3-63983a7391a9
- NCA/graphite 18650 cells
- Combined calendar and cyclic aging with different orders

### 6.3 Energy Trading Degradation Dataset
- **URL**: https://ora.ox.ac.uk/objects/uuid:9aae61af-2949-49f1-8ad5-6aea448979e5
- Year-long experiment with real-world usage profiles
- 6 Li-ion cells

**What It Provides**:
| Data Type | Available | Details |
|-----------|-----------|---------|
| Capacity vs cycles | Yes | Regular characterization |
| Impedance spectroscopy | Yes | In characterization tests |
| Temperature data | Yes | Controlled thermal chamber |
| Failure/EOL events | Yes | Long-term aging |
| Multi-cell data | Yes | Multiple cells per condition |

**Key Feature**: Path dependence studies (order of calendar vs cyclic aging matters)

**Format**: CSV and MATLAB

**License**: Open Database License (ODbL)

---

## 7. KIT Comprehensive Battery Aging Dataset

**URL**: https://publikationen.bibliothek.kit.edu/1000168959

**Publication**: "Comprehensive battery aging dataset: capacity and impedance fade measurements of a lithium-ion NMC/C-SiO cell" (Scientific Data, 2024)

**Description**:
Over 3 billion data points from 228 commercial NMC/C+SiO cells aged for over a year.

**What It Provides**:
| Data Type | Available | Details |
|-----------|-----------|---------|
| Capacity vs cycles | Yes | Wide range of operating conditions |
| Impedance spectroscopy | Yes | Impedance fade measurements |
| Temperature data | Yes | Calendar and cyclic aging profiles |
| Failure/EOL events | Yes | Full aging curves |
| Multi-cell data | Yes | 228 cells |

**Key Features**:
- Calendar and cyclic aging with different driving cycles
- Result data (remaining capacity, impedance)
- Raw data with 2-second resolution

**Format**: Multiple formats available

---

## 8. BatteryArchive.org

**URL**: https://www.batteryarchive.org

**GitHub**: https://github.com/battery-lcf/battery-archive-sandbox

**Description**:
Aggregated repository hosted by Sandia National Laboratories Grid Energy Storage Department. Contains datasets from multiple institutions.

**Available Studies**:
- SNL (Sandia National Labs)
- HNEI (Hawaii Natural Energy Institute)
- Multiple other institutional datasets

**What It Provides**:
| Data Type | Available | Details |
|-----------|-----------|---------|
| Capacity vs cycles | Yes | Standardized format |
| Impedance spectroscopy | Varies | Depends on study |
| Temperature data | Yes | Most studies |
| Failure/EOL events | Yes | Multiple failure modes |
| Multi-cell data | Yes | 1000+ cells across studies |

**Key Features**:
- Automatic visualization (capacity decay, efficiency plots)
- Standardized data format
- Batch download via GitHub scripts

**Format**: CSV (time series and cycle files)

---

## 9. Samsung ICR18650-26J EIS Dataset

**URL**: Mendeley Data (search for ICR18650-26J EIS)

**Description**:
Broadband EIS measurements at different SOC levels.

**What It Provides**:
| Data Type | Available | Details |
|-----------|-----------|---------|
| Capacity vs cycles | Limited | Focus on EIS |
| Impedance spectroscopy | Yes | 14 frequencies (0.05 Hz - 1000 Hz) |
| Temperature data | Yes | 25C controlled |
| Failure/EOL events | No | Fresh cells |
| Multi-cell data | Yes | 4 cells, 6 repetitions |

**Key Feature**: Multi-sine excitation, high-precision EIS

**Format**: Tabular data

---

## 10. Stanford EV Real-Driving Dataset

**URL**: https://storagex.stanford.edu (search for UDDS dataset)

**Description**:
Li-ion cells subjected to electric vehicle discharge profiles with periodic diagnostic tests.

**What It Provides**:
| Data Type | Available | Details |
|-----------|-----------|---------|
| Capacity vs cycles | Yes | UDDS profile discharge |
| Impedance spectroscopy | Yes | Periodic diagnostics |
| Temperature data | Yes | Controlled |
| Failure/EOL events | Yes | 23-month study |
| Multi-cell data | Yes | INR21700-M50T cells |

**Cell Chemistry**: NMC cathode, graphite/silicon anode

**Format**: Various

---

## Dataset Selection Guide

### For Constraint Count (k) Analysis
- **Best**: CALCE Storage Life Test (144 cells, multiple conditions)
- **Alternative**: MIT-Stanford (124 cells, varied protocols)

### For Correlation (rho) Analysis
- **Best**: NASA Randomized (multi-cell packs)
- **Alternative**: Oxford Path Dependent (cell groups under same protocol)

### For SEI/Impedance Evolution
- **Best**: CALCE with EIS, Sandia with broadband EIS
- **Alternative**: KIT Comprehensive (impedance fade focus)

### For Collapse/Failure Events
- **Best**: NASA Li-ion Aging (30% fade EOL)
- **Alternative**: Sandia (80% capacity EOL)

### For Multi-Chemistry Comparison
- **Best**: Sandia Comparative (LCO, LFP, NCA, NMC)
- **Alternative**: CALCE (multiple chemistries)

---

## Data Loading Tools

### Python Libraries
- **BEEP**: Battery Evaluation and Early Prediction (https://github.com/TRI-AMDD/beep)
- **cellpy**: Arbin data parsing (https://github.com/jepegit/cellpy)
- **galvani**: Biologic data parsing (https://github.com/echemdata/galvani)

### Recommended Workflow
1. Download raw data from source
2. Use appropriate parser (BEEP, cellpy, or custom MATLAB loader)
3. Export to standardized format (CSV, HDF5, or Parquet)
4. Feed into simulation engine

---

## Citation Requirements

Most datasets require attribution. Please cite:
1. The original publication describing the experiments
2. The dataset DOI (if available)
3. The hosting institution

Example for NASA dataset:
```
B. Saha and K. Goebel (2007). "Battery Data Set", NASA Ames Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA
```
