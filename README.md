# RATCHET

**Reference Architecture for Testing Coherence and Honesty in Emergent Traces**

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

RATCHET is the first computational implementation of the Coherence Ratchet framework described in Book IX of the CIRIS Covenant. It provides mathematical validation of the framework's core claims while documenting eight fundamental limitations that define its theoretical boundaries.

## Overview

RATCHET implements four computational engines totaling approximately 8,400 lines of Python:

| Engine | Purpose | Status |
|--------|---------|--------|
| **DetectionEngine** | Statistical deception detection (LRT, Mahalanobis, power analysis) | Complete |
| **GeometricEngine** | Monte Carlo volume estimation for topological collapse | Complete |
| **ComplexityEngine** | SAT-based deception complexity measurement | Complete |
| **FederationEngine** | PBFT consensus for distributed precedent accumulation | Complete |

## Key Results

### Validated Claims

- **Topological Collapse**: Volume decay matches `exp(-λ·k_eff)` within 5% for convex regions
- **NP-Completeness**: CONSISTENT-LIE reduces from 3-SAT (verified via Z3 solver)
- **Detection Power**: Achieves 95% power at n=109 samples for specified parameters
- **Cryptographic Integrity**: Ed25519 sign/verify for trace authentication

### Discovered Limitations

Implementation revealed 8 fundamental limitations that are **theoretical boundaries**, not engineering failures:

| ID | Limitation | Severity |
|----|------------|----------|
| L-01 | Emergent deception from honest components | Critical |
| L-02 | Non-adaptive adversary assumption | Critical |
| L-03 | ETH dependency for exponential gap | Critical |
| L-04 | Convexity requirement | High |
| L-05 | Finite sample validity (n ≥ 100) | Medium |
| L-06 | Correlation impact on k_eff | Medium |
| L-07 | Distribution mismatch | Medium |
| L-08 | Slow federation capture | Medium |

See [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md) for detailed analysis.

## Installation

### Requirements

- Python 3.10+
- Dependencies: `numpy`, `scipy`, `pydantic`, `cryptography`, `z3-solver`, `matplotlib`, `pytest`

### Setup

```bash
git clone https://github.com/CIRISAI/RATCHET.git
cd RATCHET
python -m venv .venv
source .venv/bin/activate
pip install numpy scipy pydantic cryptography z3-solver matplotlib pytest
```

## Quick Start

```python
from ratchet.engines import DetectionEngine, power_analysis

# Compute required sample size for 95% power
result = power_analysis(
    mahalanobis_distance=1.0,
    deception_rate=0.1,
    alpha=0.05,
    beta=0.05
)
print(f"Required samples: {result.required_sample_size}")
print(f"Achieved power: {result.achieved_power:.1%}")
```

## Running Tests

```bash
pytest simulation/ -v
```

## Project Structure

```
ratchet/
  engines/          # Core computational engines
    detection.py    # LRT, Mahalanobis, power analysis
    geometric.py    # Monte Carlo volume estimation
    complexity.py   # SAT-based complexity measurement
    federation.py   # PBFT consensus engine
  detection/        # Compositional detection
  redteam/          # Adversarial attack suite
  federation/       # BFT backends
schemas/            # Type-safe definitions
simulation/         # Standalone simulations and tests
```

## Current Status: Research Testbed

RATCHET is suitable for:
- Validating theoretical predictions
- Exploring attack/defense dynamics
- Testing detection algorithms
- Educational demonstrations

RATCHET is **not suitable** for:
- Production AI safety systems without defense-in-depth
- High-stakes deployment as sole safety mechanism
- Unsupervised operation without human oversight

## Key Formulas

**Effective Constraints:**
```
k_eff = k / (1 + ρ(k-1))
```

**Volume Decay:**
```
V(k) = V(0) · exp(-2r · k_eff)
```

**Required Sample Size:**
```
n ≥ (z_α + z_β)² / (D² · p)
```

## Documentation

- [Formal Specification Document](FSD.md)
- [Known Limitations](KNOWN_LIMITATIONS.md)
- [Amendments to Theory](AMENDMENTS.md)
- [CIRISAgent Paper](immediate_release/main.pdf) - Updated with RATCHET validation results

## Related Projects

- [CIRISAgent](https://github.com/CIRISAI/CIRISAgent) - The ethical AI agent framework
- [CIRISBridge](https://github.com/CIRISAI/CIRISBridge) - Infrastructure deployment
- [ciris-website](https://github.com/CIRISAI/ciris-website) - Project website

## Citation

If you use RATCHET in your research, please cite:

```bibtex
@misc{ratchet2026,
  author = {CIRIS Implementation Team},
  title = {RATCHET: Reference Architecture for Testing Coherence and Honesty in Emergent Traces},
  year = {2026},
  url = {https://github.com/CIRISAI/RATCHET}
}
```

## License

This project is licensed under the GNU Affero General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Key Contributors: Eric Moore, Nixon Cheaz, Ying-Jung Chen PhD, Alice Alimov, Martin Adelstein, Haley Bradley, Brad Matera, Ed Melick, Tyler Chrestoff.

---

*This implementation prioritizes intellectual honesty. All limitations are documented because understanding what the framework cannot do is as important as understanding what it can do.*
