# Battery Degradation Domain-to-Variable Mapping

This document defines how electrochemical/battery degradation concepts map to the simulation engine's abstract variables.

---

## Overview

Battery degradation is a multi-physics phenomenon involving electrochemical, thermal, and mechanical processes. The mapping below identifies measurable and manipulable quantities that can serve as simulation variables.

---

## Constraint Count (k)

**Definition**: Number of discrete constraints/rules/entities limiting battery operation.

### Primary Interpretation: Active Electrode Sites

The number of electrochemically active sites available for lithium intercalation/deintercalation.

| Constraint Type | Physical Meaning | How to Measure |
|-----------------|------------------|----------------|
| Active anode sites | Graphite/silicon sites available for Li+ | Differential capacity analysis (dQ/dV peaks) |
| Active cathode sites | Transition metal sites (Ni, Co, Mn, Fe) for Li+ | Same as above |
| Electrolyte pathways | Pore network connectivity | Tortuosity from impedance |
| Current collector contacts | Electrical contact points | Contact resistance |

### Secondary Interpretation: Operating Constraints

| Constraint Type | Physical Meaning | Typical Range |
|-----------------|------------------|---------------|
| Voltage limits | Safe operating window | 2.5V - 4.2V (chemistry dependent) |
| Current limits | Max charge/discharge rate | 0.1C - 10C |
| Temperature limits | Thermal safety window | -20C to 60C |
| SOC limits | Usable capacity window | 10% - 90% |

### How k Changes Over Time

| Process | Effect on k | Timescale |
|---------|-------------|-----------|
| SEI growth | Blocks active sites (k decreases) | Calendar aging |
| Lithium plating | Consumes Li inventory (effective k decreases) | Fast charge |
| Particle cracking | Creates new surfaces (k may increase temporarily) | Cyclic aging |
| Binder decomposition | Disconnects particles (k decreases) | Thermal abuse |

### Measurement from Data

```python
def estimate_k(cell_data):
    """
    Estimate active site count from electrochemical data.
    Uses differential capacity (dQ/dV) peak integration.
    """
    voltage = cell_data['voltage']
    capacity = cell_data['capacity']

    # Differential capacity
    dQdV = np.gradient(capacity, voltage)

    # Peak areas correlate with active sites
    # Higher peak = more active sites
    k_estimate = integrate_peaks(dQdV, voltage)

    return k_estimate
```

---

## Pairwise Correlation (rho)

**Definition**: Average correlation/coupling between constraints.

### Primary Interpretation: Cross-Cell Correlation

For battery packs, correlation between individual cells' degradation states.

| Correlation Type | Physical Meaning | Measurement |
|------------------|------------------|-------------|
| Capacity correlation | Cells degrade together | Pearson correlation of cell capacities |
| Impedance correlation | Internal resistance coupling | Correlation of EIS spectra |
| Temperature correlation | Thermal coupling in pack | Spatial temperature correlation |
| Voltage correlation | Electrical coupling | Voltage spread statistics |

### Secondary Interpretation: Electrode Coupling

Within a single cell, coupling between cathode and anode degradation.

| Coupling Type | Physical Meaning | Effect |
|---------------|------------------|--------|
| Li inventory coupling | Anode SEI consumes Li needed by cathode | LAM_PE correlates with LLI |
| Impedance coupling | Cathode/anode impedance rise together | Synergistic degradation |
| Thermal coupling | Local hot spots affect both electrodes | Accelerated local aging |

### Measurement from Data

```python
def calculate_rho(pack_data):
    """
    Calculate cross-cell correlation from pack cycling data.
    """
    cell_capacities = []
    for cell_id in pack_data['cells']:
        capacity_series = pack_data[cell_id]['capacity_over_cycles']
        cell_capacities.append(capacity_series)

    # Compute pairwise correlations
    n_cells = len(cell_capacities)
    correlations = []
    for i in range(n_cells):
        for j in range(i+1, n_cells):
            corr = np.corrcoef(cell_capacities[i], cell_capacities[j])[0,1]
            correlations.append(corr)

    rho = np.mean(correlations)
    return rho
```

```python
def calculate_electrode_coupling(eis_data):
    """
    Calculate cathode-anode coupling from EIS data.
    Uses equivalent circuit model parameters.
    """
    # Fit EIS to Randles circuit
    R_sei = fit_sei_resistance(eis_data)  # Anode side
    R_ct = fit_charge_transfer(eis_data)   # Cathode dominant

    # Track over cycles
    rho_electrode = np.corrcoef(R_sei_series, R_ct_series)[0,1]
    return rho_electrode
```

---

## Sustainability Metric (sigma)

**Definition**: Resource level / system health integral.

### Primary Interpretation: State of Health (SOH)

| Metric | Definition | Range |
|--------|------------|-------|
| Capacity SOH | Q_current / Q_initial | 0 to 1 |
| Power SOH | P_current / P_initial | 0 to 1 |
| Impedance SOH | R_initial / R_current | 0 to 1 |

### Secondary Interpretations

| Metric | Physical Meaning | Measurement |
|--------|------------------|-------------|
| Coulombic efficiency | Charge out / Charge in | Per-cycle measurement |
| Energy efficiency | Energy out / Energy in | Accounts for voltage |
| Capacity retention | % of original capacity | Long-term tracking |
| Lithium inventory | Active Li remaining | Half-cell reconstruction |

### Measurement from Data

```python
def calculate_sigma(cell_data, mode='capacity'):
    """
    Calculate sustainability metric (SOH).
    """
    if mode == 'capacity':
        Q_initial = cell_data['initial_capacity']
        Q_current = cell_data['current_capacity']
        sigma = Q_current / Q_initial

    elif mode == 'impedance':
        R_initial = cell_data['initial_resistance']
        R_current = cell_data['current_resistance']
        sigma = R_initial / R_current  # Lower is worse

    elif mode == 'coulombic':
        Q_charge = cell_data['charge_capacity']
        Q_discharge = cell_data['discharge_capacity']
        sigma = Q_discharge / Q_charge

    return np.clip(sigma, 0, 1)
```

---

## Collapse Indicator

**Definition**: Binary flag indicating system failure.

### Standard Collapse Thresholds

| Threshold Type | Value | Rationale |
|----------------|-------|-----------|
| 80% capacity | SOH < 0.80 | Industry standard for EV warranty |
| 70% capacity | SOH < 0.70 | NASA dataset EOL criterion |
| 200% impedance | R > 2 * R_initial | Power capability loss |
| Thermal runaway | T > 80C uncontrolled | Safety failure |
| Sudden death | dQ/dn > threshold | Rapid capacity drop |

### Collapse Detection from Data

```python
def detect_collapse(cell_data, threshold=0.80):
    """
    Detect collapse event in cycling data.
    """
    capacity_series = cell_data['capacity_over_cycles']
    initial_capacity = capacity_series[0]

    soh = capacity_series / initial_capacity

    # Find first crossing of threshold
    collapse_indices = np.where(soh < threshold)[0]

    if len(collapse_indices) > 0:
        return True, collapse_indices[0]
    else:
        return False, None
```

### Sudden Death Detection

```python
def detect_sudden_death(cell_data, rate_threshold=0.05):
    """
    Detect sudden death (rapid capacity fade).
    """
    capacity = cell_data['capacity_over_cycles']

    # Capacity fade rate
    dQ_dn = -np.gradient(capacity)

    # Sudden death = fade rate exceeds threshold
    sudden_indices = np.where(dQ_dn > rate_threshold * capacity)[0]

    if len(sudden_indices) > 0:
        return True, sudden_indices[0]
    else:
        return False, None
```

---

## Decay Rate (d)

**Definition**: Rate at which sustainability degrades without external input.

### Primary Interpretation: Calendar Aging Rate

| Degradation Mode | Physical Mechanism | Typical Rate |
|------------------|-------------------|--------------|
| SEI growth | Electrolyte decomposition at anode | 0.5-2% per month at 25C |
| Self-discharge | Li migration without load | 1-5% per month |
| Cathode dissolution | Transition metal leaching | Chemistry dependent |
| Corrosion | Current collector degradation | Accelerated at high T |

### Temperature Dependence (Arrhenius)

```python
def calendar_aging_rate(temperature_celsius, Ea=50000):
    """
    Calendar aging rate with Arrhenius temperature dependence.
    Ea: activation energy in J/mol (typical 40-60 kJ/mol)
    """
    R = 8.314  # J/(mol*K)
    T = temperature_celsius + 273.15  # Convert to Kelvin
    T_ref = 298.15  # 25C reference

    d_ref = 0.01  # 1% per month at 25C (example)
    d = d_ref * np.exp(Ea/R * (1/T_ref - 1/T))

    return d
```

### Measurement from Data

```python
def estimate_decay_rate(storage_data):
    """
    Estimate calendar aging rate from storage test data.
    """
    time_months = storage_data['time_days'] / 30
    capacity = storage_data['capacity']

    # Fit exponential or sqrt(t) model
    # SEI growth often follows sqrt(t)
    from scipy.optimize import curve_fit

    def sqrt_model(t, Q0, d):
        return Q0 * (1 - d * np.sqrt(t))

    popt, _ = curve_fit(sqrt_model, time_months, capacity)
    d_estimated = popt[1]

    return d_estimated
```

---

## Constraint Generation Rate (alpha)

**Definition**: Rate at which new constraints are added to the system.

### Primary Interpretation: SEI Layer Growth Rate

| Mechanism | Physical Meaning | Measurement |
|-----------|------------------|-------------|
| SEI thickening | New SEI material forms | EIS R_SEI increase rate |
| Lithium plating | Li deposits on anode | Differential voltage analysis |
| Particle cracking | New surfaces exposed | Acoustic emission |
| Dendrite growth | Li metal protrusions | Post-mortem imaging |

### Secondary Interpretation: Protocol Severity

| Factor | Effect on alpha | Example |
|--------|-----------------|---------|
| C-rate | Higher rate = faster SEI growth | 4C vs 0.5C |
| Temperature | Higher T = faster kinetics | 45C vs 25C |
| DOD | Deeper cycles = more stress | 100% vs 50% DOD |
| Voltage | High voltage = electrolyte oxidation | 4.4V vs 4.1V |

### Measurement from Data

```python
def estimate_alpha(eis_series):
    """
    Estimate constraint generation rate from EIS time series.
    Uses SEI resistance growth as proxy.
    """
    R_sei = [fit_sei_resistance(eis) for eis in eis_series]
    time = eis_series['measurement_times']

    # SEI growth rate (often parabolic: R ~ sqrt(t))
    dR_dt = np.gradient(R_sei, time)

    # Normalize by initial
    alpha = np.mean(dR_dt) / R_sei[0]

    return alpha
```

```python
def calculate_alpha_from_protocol(protocol):
    """
    Estimate alpha from cycling protocol parameters.
    Higher stress = higher alpha.
    """
    # Stress factors (empirical weights)
    c_rate_factor = protocol['c_rate'] / 1.0  # Normalized to 1C
    temp_factor = np.exp((protocol['temperature'] - 25) / 10)  # Arrhenius-like
    dod_factor = protocol['dod'] / 0.8  # Normalized to 80% DOD
    voltage_factor = (protocol['upper_voltage'] - 4.0) / 0.2 + 1  # Normalized to 4.0V

    alpha_base = 0.001  # Base rate
    alpha = alpha_base * c_rate_factor * temp_factor * dod_factor * voltage_factor

    return alpha
```

---

## Strictness (lambda)

**Definition**: Strength/enforcement of constraints.

### Primary Interpretation: Voltage Limit Strictness

| Parameter | Loose (low lambda) | Strict (high lambda) |
|-----------|-------------------|---------------------|
| Upper voltage | 4.4V | 4.1V |
| Lower voltage | 2.5V | 3.0V |
| C-rate limit | 10C | 1C |
| Temperature range | -20C to 60C | 15C to 35C |

### Secondary Interpretation: BMS Enforcement

| BMS Setting | Low lambda | High lambda |
|-------------|------------|-------------|
| Cell balancing | Passive only | Active balancing |
| Cutoff precision | Wide tolerance | Tight tolerance |
| Thermal management | Air cooling | Liquid cooling |
| SOC limits | 0-100% | 20-80% |

### Effect on Degradation

Higher lambda (stricter limits) generally:
- Reduces stress on cells
- Slows degradation rate
- May reduce usable energy per cycle
- Improves safety margins

### Measurement from Data

```python
def calculate_lambda(operating_data):
    """
    Calculate effective strictness from operating envelope.
    """
    # Voltage window strictness
    V_max_actual = operating_data['max_voltage']
    V_min_actual = operating_data['min_voltage']
    V_max_limit = 4.2  # Chemistry limit
    V_min_limit = 2.5  # Chemistry limit

    voltage_strictness = 1 - ((V_max_actual - V_min_actual) / (V_max_limit - V_min_limit))

    # Temperature strictness
    T_range_actual = operating_data['T_max'] - operating_data['T_min']
    T_range_limit = 80  # Full chemistry range

    temp_strictness = 1 - (T_range_actual / T_range_limit)

    # C-rate strictness
    C_max_actual = operating_data['max_c_rate']
    C_max_limit = 10  # Example limit

    crate_strictness = 1 - (C_max_actual / C_max_limit)

    # Combined lambda (average or weighted)
    lambda_ = np.mean([voltage_strictness, temp_strictness, crate_strictness])

    return lambda_
```

---

## Collapse Time (t_collapse)

**Definition**: Time at which system crosses failure threshold.

### Measurement from Data

```python
def calculate_collapse_time(cell_data, threshold=0.80):
    """
    Calculate collapse time from cycling or calendar aging data.
    """
    time = cell_data['time']  # Calendar time or cycle number
    capacity = cell_data['capacity']
    initial_capacity = capacity[0]

    soh = capacity / initial_capacity

    # Interpolate to find exact crossing
    from scipy.interpolate import interp1d

    if soh[-1] >= threshold:
        return None  # No collapse in data

    # Find crossing point
    f = interp1d(soh[::-1], time[::-1])  # Reverse for interpolation
    t_collapse = f(threshold)

    return t_collapse
```

---

## Effective Constraint Count (k_eff)

**Definition**: Computed as k / (1 + rho * (k - 1))

### Physical Interpretation

When cells/sites are highly correlated (rho -> 1):
- They behave as a single effective constraint
- k_eff approaches 1 regardless of nominal k

When cells/sites are uncorrelated (rho -> 0):
- Each acts independently
- k_eff equals k

### Measurement

```python
def calculate_k_eff(k, rho):
    """
    Calculate effective constraint count.
    """
    if k <= 1:
        return k
    return k / (1 + rho * (k - 1))
```

---

## Compromised Fraction (f)

**Definition**: Fraction of system captured/corrupted.

### Interpretation for Batteries

| Metric | Physical Meaning | Range |
|--------|------------------|-------|
| Lost lithium inventory | Li consumed by SEI | 0 to 1 |
| Dead volume fraction | Inactive electrode mass | 0 to 1 |
| Failed cells in pack | Non-functional cells | 0 to 1 |
| Capacity fade fraction | 1 - SOH | 0 to 1 |

### Measurement

```python
def calculate_f(cell_data):
    """
    Calculate compromised fraction.
    Represents fraction of original capacity lost.
    """
    initial_capacity = cell_data['initial_capacity']
    current_capacity = cell_data['current_capacity']

    f = 1 - (current_capacity / initial_capacity)

    return np.clip(f, 0, 1)
```

---

## Summary Mapping Table

| Abstract Variable | Primary Battery Mapping | Secondary Mapping | Units |
|-------------------|------------------------|-------------------|-------|
| k | Active electrode sites | Operating constraints | count |
| alpha | SEI growth rate | Protocol severity | 1/time |
| d | Calendar aging rate | Storage degradation | 1/time |
| lambda | Voltage limit strictness | BMS enforcement | dimensionless |
| rho | Cross-cell correlation | Electrode coupling | [0, 1] |
| sigma | State of Health (SOH) | Coulombic efficiency | [0, 1] |
| f | Capacity fade fraction | Dead volume fraction | [0, 1] |
| k_eff | Effective active sites | Correlated constraints | count |
| collapsed | Below 80% capacity | Thermal runaway | boolean |
| t_collapse | Time to 80% SOH | Time to failure | cycles or hours |

---

## Rationale Notes

1. **Theory-agnostic design**: The mapping exposes electrochemical quantities without assuming relationships between them.

2. **Measurability**: Each variable can be computed from standard battery test data (cycling, EIS, storage tests).

3. **Manipulability**: Operating constraints (voltage, temperature, C-rate) are directly controllable in experiments.

4. **Multi-scale support**: Mappings work at cell level (single cell) and pack level (multi-cell).

5. **Chemistry-agnostic**: While specific values differ, the framework applies to LFP, NMC, NCA, LCO, etc.
