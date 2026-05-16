# `data/battery/` — NASA PCoE Li-ion Battery Aging Dataset

This directory hosts the **NASA PCoE Battery Data Set** used by the CCA Tier-1 battery validation. The raw archive is large (200 MB) and gitignored.

## Pinned artifact

| Property | Value |
|---|---|
| Source | NASA Ames Prognostics Data Repository |
| URL | https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip |
| Version | static historical (2007 publication, dataset frozen since 2022) |
| License | Public Domain (NASA) |
| SHA-256 | `82302a7db4fc1b34e0b6676326610438d43b816bdf11a69d1d012a464ef2f92e` |
| Size | 209 708 670 bytes (200 MiB) |
| Inner archive | 6 zips: `1. BatteryAgingARC-FY08Q4.zip` … `6. BatteryAgingARC_53_54_55_56.zip` |
| High-quality cell count | 19 (subset; full archive contains 34) |
| Pin recorded | 2026-05-16, master commit anchoring CRCv2 + CCA uplift |

## How the loader uses this

```python
from ratchet.data.battery_loader import load_nasa_battery_data
dataset = load_nasa_battery_data(
    data_dir="data/battery/5. Battery Data Set",
    high_quality_only=True,
)
# dataset.k = 19, dataset.get_sigma() ≈ 0.793 (mean SOH across cells)
```

## Local provisioning

The archive is vendored via symlink in this repo:
- `battery_dataset.zip` → `/home/emoore/RATCHET-RESET/data/battery/battery_dataset.zip`
- `5. Battery Data Set/` → `/home/emoore/RATCHET-RESET/data/battery/5. Battery Data Set`

If running fresh (no RATCHET-RESET present), download and extract:

```bash
mkdir -p data/battery && cd data/battery
curl -L -o battery_dataset.zip \
  "https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip"
unzip battery_dataset.zip
# Verify
sha256sum battery_dataset.zip  # should match the pin above
```

For per-substrate hash validation in CI, see `experiments/exp2_cross_substrate/data_sources.yaml` (`nasa_battery.expected_sha256`) and `experiments/exp2_cross_substrate/data_fetch.py`.

## Pre-computed validation results

`experiments/exp0_cca_validation/results/nasa_battery_groups.json` contains the per-group k, ρ, k_eff values from the original CCA paper run:

| Group | k | ρ | k_eff |
|---|---|---|---|
| FY08Q4_24C | 4 | 0.962 | 1.03 |
| Set_25-28 | 4 | 0.539 | 1.53 |
| Set_29-32 | 4 | 0.972 | 1.02 |
| Set_33-36 | 3 | 0.522 | 1.47 |
| Set_38-40 | 3 | 0.418 | 1.63 |
| Set_41-44 | 4 | 0.006 | 3.93 |
| Set_45-48 | 4 | 0.972 | 1.02 |
| Set_49-52 | 4 | 0.451 | 1.70 |
| Set_53-56 | 4 | 0.648 | 1.36 |

Note how high-ρ groups (>0.5) drive k_eff toward 1 (single effective cell) while low-ρ groups (e.g. Set_41-44, ρ=0.006) keep k_eff near k. This is the Kish formula in physical battery data.
