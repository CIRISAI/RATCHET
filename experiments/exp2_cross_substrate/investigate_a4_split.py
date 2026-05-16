#!/usr/bin/env python3
"""
Investigate v2.0's A4 split: V-Dem +0.582 vs Polity5 +0.211.

Hypothesis: the gap is driven by INDICATOR DATA TYPE (continuous real-valued
vs categorical 1-7 scales), not by agency level. Test this directly by
computing the mean lag-1 autocorrelation of the RAW indicators along each
country's time axis, *bypassing* Kish entirely.

If continuous indicators (V-Dem, WGI) show systematically higher |φ_lag1|
than categorical indicators (Polity5), the v2.0 excess|φ| metric is
partly tracking data-type smoothness rather than agency coordination.

For each substrate-A4:
  - Per country (≥20 years), compute lag-1 autocorrelation of each indicator
    column independently
  - Aggregate: mean |φ_lag1| across (country × indicator)

If the data-type hypothesis is correct:
  Polity5 |φ_lag1| << V-Dem |φ_lag1|, WGI |φ_lag1|
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def lag1_autocorr(x: np.ndarray) -> float:
    """Magnitude of lag-1 Pearson autocorrelation."""
    x = np.asarray(x, dtype=float)
    if len(x) < 3:
        return float("nan")
    c = x - x.mean()
    denom = float((c ** 2).sum())
    if denom < 1e-15:
        return float("nan")
    return float(abs((c[:-1] * c[1:]).sum() / denom))


def investigate_polity5(min_years: int = 20):
    import pandas as pd
    df = pd.read_excel(REPO_ROOT / "data" / "institutional" / "polity5.xls")
    cols = ["xconst", "xrcomp", "xropen", "xrreg", "exrec", "exconst"]
    cols = [c for c in cols if c in df.columns]
    df = df[df["polity2"].notna()].copy()
    df["polity2"] = pd.to_numeric(df["polity2"], errors="coerce")
    df = df[(df["polity2"] >= -10) & (df["polity2"] <= 10)].copy()
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df.loc[df[c] <= -50, c] = np.nan
    phis = []
    for country, grp in df.groupby("country"):
        grp = grp.sort_values("year").reset_index(drop=True)
        if len(grp) < min_years:
            continue
        for c in cols:
            x = grp[c].values
            if np.isnan(x).all():
                continue
            x = x[~np.isnan(x)]
            if len(x) < min_years:
                continue
            phi = lag1_autocorr(x)
            if not np.isnan(phi):
                phis.append(phi)
    arr = np.asarray(phis)
    return {
        "substrate": "Polity5",
        "type": "categorical (1-7 scales)",
        "n_country_indicator_pairs": len(arr),
        "mean_lag1_abs_phi": float(arr.mean()),
        "median_lag1_abs_phi": float(np.median(arr)),
        "std": float(arr.std()),
        "frac_above_0.9": float((arr > 0.9).mean()),
    }


def investigate_vdem(min_years: int = 20):
    import pandas as pd
    indicators = ["v2x_polyarchy", "v2x_libdem", "v2x_partipdem",
                  "v2x_delibdem", "v2x_egaldem", "v2xeg_eqdr"]
    df = pd.read_parquet(
        REPO_ROOT / "data" / "institutional" / "vdem" / "v-dem-v15.parquet",
        columns=["country_name", "year"] + indicators,
    )
    df = df.dropna(subset=["country_name", "year"])
    phis = []
    for country, grp in df.groupby("country_name"):
        grp = grp.sort_values("year").reset_index(drop=True)
        if len(grp) < min_years:
            continue
        for c in indicators:
            x = grp[c].dropna().values
            if len(x) < min_years:
                continue
            phi = lag1_autocorr(x)
            if not np.isnan(phi):
                phis.append(phi)
    arr = np.asarray(phis)
    return {
        "substrate": "V-Dem",
        "type": "continuous real-valued ([0,1] composite indices)",
        "n_country_indicator_pairs": len(arr),
        "mean_lag1_abs_phi": float(arr.mean()),
        "median_lag1_abs_phi": float(np.median(arr)),
        "std": float(arr.std()),
        "frac_above_0.9": float((arr > 0.9).mean()),
    }


def investigate_wgi(min_years: int = 20):
    import pandas as pd
    indicators = ["CC.EST", "GE.EST", "PV.EST", "RQ.EST", "RL.EST", "VA.EST"]
    df = pd.read_csv(
        REPO_ROOT / "data" / "institutional" / "wgi_processed.csv",
        usecols=["country", "year"] + indicators,
    )
    df = df.dropna(subset=["country", "year"])
    phis = []
    for country, grp in df.groupby("country"):
        grp = grp.sort_values("year").reset_index(drop=True)
        if len(grp) < min_years:
            continue
        for c in indicators:
            x = grp[c].dropna().values
            if len(x) < min_years:
                continue
            phi = lag1_autocorr(x)
            if not np.isnan(phi):
                phis.append(phi)
    arr = np.asarray(phis)
    return {
        "substrate": "WGI",
        "type": "continuous real-valued (z-scores)",
        "n_country_indicator_pairs": len(arr),
        "mean_lag1_abs_phi": float(arr.mean()),
        "median_lag1_abs_phi": float(np.median(arr)),
        "std": float(arr.std()),
        "frac_above_0.9": float((arr > 0.9).mean()),
    }


if __name__ == "__main__":
    print("Investigating A4-split: raw lag-1 autocorrelation per indicator")
    print("=" * 72)
    print()
    print("Hypothesis: continuous indicators autocorrelate more strongly along")
    print("time than categorical ones, independent of agency level.")
    print()
    for f in [investigate_polity5, investigate_vdem, investigate_wgi]:
        r = f()
        print(f"{r['substrate']}  ({r['type']})")
        print(f"  n country-indicator pairs:  {r['n_country_indicator_pairs']:>6}")
        print(f"  mean |φ_lag1|:              {r['mean_lag1_abs_phi']:.4f}")
        print(f"  median |φ_lag1|:            {r['median_lag1_abs_phi']:.4f}")
        print(f"  std:                        {r['std']:.4f}")
        print(f"  fraction with |φ| > 0.9:    {r['frac_above_0.9']*100:.1f}%")
        print()
