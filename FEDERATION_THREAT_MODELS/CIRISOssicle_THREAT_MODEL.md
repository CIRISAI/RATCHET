# CIRISOssicle Threat Model

**Document Date:** January 9, 2026
**Version:** 1.0

## Overview

CIRISOssicle detects unauthorized GPU workloads through power-draw signature detection via EM coupling. This document describes what the sensor can and cannot detect, and its security properties.

### Detection Mechanism

The oscillator doesn't detect workloads directly—it detects the workload's effect on local power draw, which modulates coupling to the ambient EM field. This is a **physical side-channel** that:

1. **Can't be spoofed by software-only attacks** - The power draw is a hardware effect
2. **Is hard to mask** - Masking requires changing the power profile, which affects the malicious workload
3. **Works regardless of workload logic** - Detection is based on physical signature, not code analysis

---

## What CIRISOssicle DETECTS

| Threat | Detection | Confidence |
|--------|-----------|------------|
| Crypto mining (>30% GPU) | **YES** | High (z > 2.5) |
| Memory bandwidth attacks | **YES** | High (z > 3.0) |
| Concurrent compute workloads | **YES** | Medium-High |
| GPU resource hijacking | **YES** | High |
| Large matrix operations | **YES** | Medium |

### Detection Mechanism

The sensor detects workloads that:
1. Draw significant power from the GPU PDN
2. Run concurrently with the ossicle sensor
3. Create measurable voltage droop/noise

---

## What CIRISOssicle Does NOT Detect

### 1. Workloads That Don't Affect PDN

| Non-Detectable | Reason |
|----------------|--------|
| CPU-only attacks | No GPU PDN impact |
| Network exfiltration | Minimal GPU usage |
| Memory-only reads | Low power signature |
| Idle GPU backdoors | No active computation |

### 2. Low-Intensity Workloads

| Limitation | Details |
|------------|---------|
| <30% GPU utilization | Below detection threshold |
| Micro-burst attacks | Too brief for correlation |
| Workloads matching baseline | Indistinguishable from normal |

### 3. Sophisticated Evasion

| Attack | Vulnerability | Difficulty |
|--------|---------------|------------|
| Power-matched mimicry | Attacker matches authorized power profile | HIGH - requires hardware-level changes |
| ~~Correlation spoofing~~ | ~~Attacker injects counter-correlations~~ | INEFFECTIVE - physical side-channel |
| Sensor starvation | Preventing ossicle from running | MEDIUM - detectable by external monitor |
| ~~Software-only spoofing~~ | ~~Fake the sensor readings~~ | INEFFECTIVE - can't fake EM coupling |

**Why software spoofing fails:** The sensor detects power-draw signatures via EM coupling. Software cannot control the physical power draw pattern without actually changing the computation, which would affect the malicious workload's effectiveness.

### 4. Scope Limitations

| Out of Scope | Notes |
|--------------|-------|
| Pre-boot attacks | Sensor must be running |
| Firmware tampering | Hardware-level compromise |
| Multi-GPU coordination | Only monitors local GPU |
| Host OS compromise | Assumes trusted host |

---

## Threat Model Assumptions

### Trusted Components

1. **Host OS kernel** - Not compromised
2. **CUDA driver** - Authentic and unmodified
3. **GPU firmware** - Not tampered
4. **Ossicle code** - Runs with integrity

### Attacker Capabilities

We assume an attacker who can:
- Run arbitrary CUDA code on the GPU
- Share GPU resources with the ossicle
- Observe ossicle behavior externally

We assume an attacker who CANNOT:
- Modify ossicle code at runtime
- Intercept ossicle GPU memory
- Control GPU scheduling entirely
- Modify CUDA driver behavior

---

## Security Properties

### Provided

| Property | Description |
|----------|-------------|
| **Detection** | Unauthorized high-power workloads trigger alerts |
| **Latency** | Detection within 0.1 seconds |
| **Stealth** | Minimal footprint (0.75 KB) |
| **Independence** | No external hardware required |

### NOT Provided

| Property | Description |
|----------|-------------|
| **Prevention** | Does not stop attacks, only detects |
| **Attribution** | Cannot identify attack source |
| **Forensics** | No detailed logging of attack |
| **Recovery** | No automatic remediation |

---

## Attack Scenarios

### Scenario 1: Cloud GPU Hijacking

**Threat:** Attacker gains code execution on cloud VM, runs crypto miner.

**Detection:** YES - Crypto mining at >30% causes z > 2.5
**Limitation:** Low-intensity mining may evade

### Scenario 2: LLM Inference Tampering

**Threat:** Unauthorized workload runs alongside LLM inference.

**Detection:** YES - Concurrent compute changes correlation pattern
**Limitation:** Small auxiliary computations may not register

### Scenario 3: Power-Matched Evasion

**Threat:** Attacker crafts workload matching authorized power profile.

**Detection:** UNCERTAIN - Depends on correlation signature difference
**Limitation:** Sophisticated matching could evade

### Scenario 4: Sensor Denial

**Threat:** Attacker prevents ossicle from running or starves it of GPU time.

**Detection:** NO - Sensor must run to detect
**Mitigation:** External monitoring of sensor health

---

## Recommended Deployment

### Combine With

1. **External power monitoring** - Catches what ossicle misses
2. **Process auditing** - Identify attack source
3. **Rate limiting** - Prevent sensor starvation
4. **Redundant sensors** - Multiple ossicle instances

### Best Practices

1. Run ossicle continuously during protected workloads
2. Establish clean baseline before deployment
3. Set detection threshold based on acceptable false positives
4. Monitor ossicle health externally
5. Log alerts for forensic analysis

---

## False Positive Sources

| Source | Mitigation |
|--------|------------|
| Legitimate GPU workloads | Whitelist known signatures |
| Temperature fluctuations | Temperature compensation |
| Power supply instability | Baseline under normal conditions |
| Driver updates | Re-establish baseline |

---

## Conclusion

CIRISOssicle is a detection layer, not a complete security solution. It excels at detecting high-power concurrent GPU workloads with minimal overhead. Deploy alongside complementary security measures for defense in depth.
