#!/usr/bin/env python3
"""
Exp 2 P2 v2.0 — Substrate Fractality Test (time-ordered residual + shuffled null).

v1.x DESIGN FLAW (acknowledged in REGIME.md v1.4 close-out):
  The framework's prediction is about *coordination structure* in residuals,
  which requires the residuals to have meaningful sequential ordering. v1.1-v1.4
  drew samples uniformly at random and computed autocorrelation on the random
  cross-section — this measured sampling noise (E|φ_lag1| ≈ √(1/n)), not
  coordination structure. After 4 pre-registered runs all returned INCONCLUSIVE
  in the central [-0.3, +0.3] band, the methodology was retired.

v2.0 DESIGN (this file):
  Per substrate, extract TIME-ORDERED trajectories (battery cohort over cycles,
  country over years, session over time bins, community over years, etc.).
  Per trajectory: fit Kish regression σ ≈ α + β·k_eff on time-ordered samples,
  compute mean|φ| of TIME-ORDERED residuals AND of a DETERMINISTIC NULL
  (residuals permuted 200× → median of the resulting mean|φ| values).
  Per substrate aggregate: mean across trajectories.
  excess|φ|_substrate = mean(|φ|_ordered) - mean(|φ|_null)

  Cross-substrate test: Spearman ρ(rung, excess|φ|).

  DECISION RULE (locked, mirrors v1.x partition):
    ρ ≥ +0.7        STRONG_PASS (F-7b confirmed at v2.0 operationalization)
    +0.3 ≤ ρ < +0.7 WEAK_PASS  (directional support)
    -0.3 ≤ ρ < +0.3 INCONCLUSIVE
    -0.7 ≤ ρ < -0.3 WEAK_FAIL
    ρ < -0.7        STRONG_FAIL (F-7b falsified)
    n_valid < 4     INDETERMINATE

  Substrates included (must have time-ordering AND varying k along trajectory):
    A0: battery (cohort over cycles), alphafold (residues along protein)
    A1: allen   (spike-train time bins)
    A2: biotime (community over years)
    A3: ciris   (chains over timestamp)
    A4: institutional (country-year), vdem (country-year)

  Substrates dropped vs v1.x:
    pmu        — k fixed at 2 along time, Kish degenerates
    microbiome — HF CRC cohort is cross-sectional (no longitudinal axis)

  Confounder controls (C-1..C-6 carried over, plus new C-7):
    C-1..C-6: as in v1.x lake spec
    C-7 (NEW): autocorrelation-on-random-cross-section is sampling-noise
               (see v1.x retirement). v2.0 requires time-ordered trajectories
               and contrasts ordered vs shuffled-null.

This file's results will be written to data/p2_substrate_fractality_v2_results.json.
v1.x results in data/p2_substrate_fractality_results.json remain unchanged.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from analysis.omega.kish_fit import (  # noqa: E402
    autocorr_decay_profile,
    fit_kish_regression,
)


# A trajectory is (k_array, rho_array, sigma_array) with all three the same length
# and time-ordered.
Trajectory = Tuple[np.ndarray, np.ndarray, np.ndarray]


SUBSTRATE_RUNGS = {
    "battery":       0,
    "alphafold":     0,
    "allen":         1,
    "biotime":       2,
    "ciris":         3,
    "institutional": 4,
    "vdem":          4,
}

# Minimum trajectory length for autocorr to be informative
MIN_TRAJ_LEN = 20
# Max lag for autocorr_decay_profile
MAX_LAG = 10
# Number of shuffles for the deterministic null
N_NULL_SHUFFLES = 200
# Bootstrap iterations for excess|φ| substrate CI
N_BOOTSTRAP = 1000


# ─── Trajectory extractors ────────────────────────────────────────────


def get_battery_trajectories(window: int = 5, seed: int = 42) -> List[Trajectory]:
    """Battery cohort over cycles. ONE trajectory (the whole cohort).

    At cycle t, sliding window [t:t+window]:
      k_t   = number of cells alive throughout window (SOH > 0.4 threshold)
      σ_t   = mean SOH across alive cells at window end
      ρ_t   = mean pairwise SOH correlation across alive cells in window
    """
    try:
        from ratchet.data.battery_loader import load_nasa_battery_data
    except ImportError:
        return []
    try:
        dataset = load_nasa_battery_data(
            data_dir=str(REPO_ROOT / "data" / "battery" / "5. Battery Data Set"),
            high_quality_only=True,
        )
    except Exception:
        return []
    soh_lists = [np.asarray(c.soh_values, dtype=float)
                 for c in dataset.cells.values() if hasattr(c, "soh_values")]
    soh_lists = [s for s in soh_lists if len(s) >= MIN_TRAJ_LEN]
    if len(soh_lists) < 3:
        return []
    common = min(len(s) for s in soh_lists)
    M = np.array([s[:common] for s in soh_lists])  # (n_cells, n_cycles)
    n_cells, n_cycles = M.shape

    k_arr, rho_arr, sigma_arr = [], [], []
    for t in range(0, n_cycles - window, 1):
        win = M[:, t: t + window]
        alive = (win > 0.4).all(axis=1)
        if alive.sum() < 3:
            continue
        sub = win[alive]
        k = int(alive.sum())
        sigma = float(np.clip(sub[:, -1].mean(), 0.01, 0.99))
        try:
            corr = np.corrcoef(sub)
            off = corr[np.triu_indices(k, k=1)]
            rho = float(np.clip(abs(np.nanmean(off)), 0.0, 1.0))
        except Exception:
            continue
        k_arr.append(k)
        rho_arr.append(rho)
        sigma_arr.append(sigma)
    if len(k_arr) < MIN_TRAJ_LEN:
        return []
    return [(np.asarray(k_arr), np.asarray(rho_arr), np.asarray(sigma_arr))]


def get_alphafold_trajectories(window: int = 7, max_proteins: int = 50,
                               seed: int = 42) -> List[Trajectory]:
    """AlphaFold residues along each protein. One trajectory per protein.

    At residue position p, window [p:p+window]:
      k_t   = number of residues with pLDDT > 70 (confident) in window
      σ_t   = mean pLDDT in window (0-1 scale)
      ρ_t   = lag-1 autocorrelation of pLDDT in window
    """
    csv_path = REPO_ROOT / "data" / "protein" / "cath_s40_alphafold_sample.csv"
    if not csv_path.exists():
        return []
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
    except Exception:
        return []
    if "plddt_trajectory" not in df.columns:
        return []
    rng = np.random.default_rng(seed)
    df = df.sample(min(max_proteins, len(df)),
                   random_state=int(rng.integers(0, 2**31 - 1)))
    trajectories = []
    for _, row in df.iterrows():
        try:
            traj = [float(v) for v in str(row["plddt_trajectory"]).strip("[]").split(",")]
        except Exception:
            continue
        if len(traj) < MIN_TRAJ_LEN + window:
            continue
        plddt = np.asarray(traj, dtype=float)  # 0-100 scale
        k_arr, rho_arr, sigma_arr = [], [], []
        for p in range(0, len(plddt) - window):
            w = plddt[p: p + window]
            k = int((w > 70.0).sum())
            if k < 2:
                continue
            sigma = float(np.clip(w.mean() / 100.0, 0.01, 0.99))
            cent = w - w.mean()
            denom = float((cent ** 2).sum())
            if denom < 1e-9:
                continue
            rho = float(np.clip(abs((cent[:-1] * cent[1:]).sum() / denom), 0.0, 1.0))
            k_arr.append(k)
            rho_arr.append(rho)
            sigma_arr.append(sigma)
        if len(k_arr) < MIN_TRAJ_LEN:
            continue
        # Require k OR ρ to vary along the trajectory — if both are flat,
        # Kish regression has no signal axis. Categorical-indicator substrates
        # (Polity5, V-Dem) often have k constant but ρ varying year-to-year,
        # which is still admissible.
        k_a = np.asarray(k_arr); r_a = np.asarray(rho_arr)
        if (k_a.max() - k_a.min() < 2) and (r_a.max() - r_a.min() < 0.1):
            continue
        trajectories.append((k_a, r_a, np.asarray(sigma_arr)))
    return trajectories


def get_allen_trajectories(window_bins: int = 20, stride: int = 5,
                           max_sessions: int = 30, seed: int = 42
                           ) -> List[Trajectory]:
    """Allen Neuropixels spike-train sessions. One trajectory per session.

    At time bin t, sliding window of `window_bins`:
      k_t   = number of neurons firing ≥1 spike in window
      σ_t   = mean firing rate (normalized to [0.01, 0.99])
      ρ_t   = mean pairwise spike-train correlation across active neurons
    """
    try:
        import pandas as pd
        path = REPO_ROOT / "data" / "neural" / "allen_neuropixels_sessions.parquet"
        if not path.exists():
            path = REPO_ROOT / "data" / "neural" / "allen_neuropixels_sample.parquet"
        if not path.exists():
            return []
        df = pd.read_parquet(path)
    except Exception:
        return []
    rng = np.random.default_rng(seed)
    df = df.head(max_sessions)
    trajectories = []
    for _, row in df.iterrows():
        n_neurons = int(row.get("n_neurons", 0))
        try:
            raw = row["spike_train_matrix"]
            if isinstance(raw, (bytes, bytearray)):
                flat = np.frombuffer(raw, dtype=np.uint8).astype(float)
            else:
                if hasattr(raw, "tolist"):
                    raw = raw.tolist()
                flat = np.asarray(raw, dtype=float)
        except Exception:
            continue
        if n_neurons < 5 or len(flat) < n_neurons * (MIN_TRAJ_LEN + window_bins):
            continue
        try:
            mat = flat.reshape(n_neurons, -1)  # (neurons, time_bins)
        except ValueError:
            continue
        n_bins = mat.shape[1]
        k_arr, rho_arr, sigma_arr = [], [], []
        for t in range(0, n_bins - window_bins, stride):
            w = mat[:, t: t + window_bins]
            active = (w.sum(axis=1) > 0)
            k = int(active.sum())
            if k < 3:
                continue
            sub = w[active]
            rate = float(sub.mean())
            sigma = float(np.clip(rate / (1.0 + rate), 0.01, 0.99))
            try:
                corr = np.corrcoef(sub)
                off = corr[np.triu_indices(k, k=1)]
                rho = float(np.clip(abs(np.nanmean(off)), 0.0, 1.0))
            except Exception:
                continue
            k_arr.append(k)
            rho_arr.append(rho)
            sigma_arr.append(sigma)
        if len(k_arr) < MIN_TRAJ_LEN:
            continue
        # Require k OR ρ to vary along the trajectory — if both are flat,
        # Kish regression has no signal axis. Categorical-indicator substrates
        # (Polity5, V-Dem) often have k constant but ρ varying year-to-year,
        # which is still admissible.
        k_a = np.asarray(k_arr); r_a = np.asarray(rho_arr)
        if (k_a.max() - k_a.min() < 2) and (r_a.max() - r_a.min() < 0.1):
            continue
        trajectories.append((k_a, r_a, np.asarray(sigma_arr)))
    return trajectories


def get_biotime_trajectories(window: int = 3, max_communities: int = 80,
                             seed: int = 42) -> List[Trajectory]:
    """BioTIME community over years. One trajectory per community.

    At year t, sliding window of `window` years:
      k_t   = number of species detected in window
      σ_t   = biomass stability (1 / (1+CV))
      ρ_t   = mean pairwise species-abundance correlation in window
    """
    try:
        from ratchet.data.ecological_loader import load_biotime_data
    except ImportError:
        return []
    try:
        ds = load_biotime_data(min_years=MIN_TRAJ_LEN + window,
                                min_species=5, fallback_to_synthetic=False)
    except Exception:
        return []
    if not hasattr(ds, "communities") or not ds.communities:
        return []
    rng = np.random.default_rng(seed)
    cids = list(ds.communities.keys())
    rng.shuffle(cids)
    cids = cids[:max_communities]
    trajectories = []
    for cid in cids:
        c = ds.communities[cid]
        spec_ab = getattr(c, "species_abundances", None)
        if spec_ab is None or spec_ab.size == 0:
            continue
        n_species, n_years = spec_ab.shape
        if n_years < MIN_TRAJ_LEN + window:
            continue
        k_arr, rho_arr, sigma_arr = [], [], []
        for t in range(0, n_years - window):
            w = spec_ab[:, t: t + window]
            present = (w.sum(axis=1) > 0)
            k = int(present.sum())
            if k < 3:
                continue
            sub = w[present]
            biomass = sub.sum(axis=0)
            if biomass.mean() < 1e-9:
                continue
            cv = float(biomass.std() / biomass.mean())
            sigma = float(np.clip(1.0 / (1.0 + cv), 0.01, 0.99))
            try:
                corr = np.corrcoef(sub)
                off = corr[np.triu_indices(k, k=1)]
                rho = float(np.clip(abs(np.nanmean(off)), 0.0, 1.0))
            except Exception:
                continue
            k_arr.append(k)
            rho_arr.append(rho)
            sigma_arr.append(sigma)
        if len(k_arr) < MIN_TRAJ_LEN:
            continue
        # Require k OR ρ to vary along the trajectory — if both are flat,
        # Kish regression has no signal axis. Categorical-indicator substrates
        # (Polity5, V-Dem) often have k constant but ρ varying year-to-year,
        # which is still admissible.
        k_a = np.asarray(k_arr); r_a = np.asarray(rho_arr)
        if (k_a.max() - k_a.min() < 2) and (r_a.max() - r_a.min() < 0.1):
            continue
        trajectories.append((k_a, r_a, np.asarray(sigma_arr)))
    return trajectories


def get_ciris_trajectories(window: int = 5, seed: int = 42) -> List[Trajectory]:
    """CIRIS chains ordered by file timestamp. ONE trajectory (chains over time).

    At chain index t, sliding window of `window` chains:
      k_t   = mean scalar count fired across window
      σ_t   = mean of mean scores in window
      ρ_t   = mean within-chain consensus (1 - 2·std) in window
    """
    import json as _json
    candidates = [
        REPO_ROOT / "experiments/exp1b_boundary_active/data/crossfamily/qwen-3.5-35b-a3b/tee",
        REPO_ROOT / "experiments/exp1b_boundary_active/data/crossfamily/llama-4-scout/tee",
        Path("/tmp/exp1b_gemini"),
    ]
    paths = []
    for d in candidates:
        if d.is_dir():
            paths.extend(sorted(d.glob("*.json")))
    if not paths:
        return []
    # Order by mtime to give a real temporal axis
    paths.sort(key=lambda p: p.stat().st_mtime)

    per_chain = []  # list of (k, sigma, rho) tuples in temporal order
    for p in paths:
        try:
            d = _json.loads(p.read_text())
        except Exception:
            continue
        trace = (d.get("events") or [{}])[0].get("trace") or {}
        scores = []
        for c in trace.get("components", []):
            et = c.get("event_type")
            data = c.get("data", {})
            if et == "DMA_RESULTS":
                for dma in ("csdma", "dsdma"):
                    sub = data.get(dma) or {}
                    for sk in ("plausibility_score", "domain_alignment"):
                        v = sub.get(sk)
                        if v is not None and 0.0 <= v <= 1.0:
                            scores.append(float(v))
            elif et == "CONSCIENCE_RESULT":
                for sk in ("entropy_score", "coherence_score",
                           "epistemic_humility_certainty"):
                    v = data.get(sk)
                    if v is not None and 0.0 <= v <= 1.0:
                        scores.append(float(v))
                v = data.get("optimization_veto_entropy_ratio")
                if v is not None and v >= 0:
                    scores.append(float(min(v / 2.0, 1.0)))
        if len(scores) < 2:
            continue
        arr = np.array(scores)
        per_chain.append((len(arr), float(arr.mean()), float(arr.std())))
    if len(per_chain) < MIN_TRAJ_LEN + window:
        return []
    per_chain = np.array(per_chain)  # (n_chains, 3): k, mean_score, std_score
    k_arr, rho_arr, sigma_arr = [], [], []
    for t in range(0, len(per_chain) - window):
        w = per_chain[t: t + window]
        k = int(round(w[:, 0].mean()))
        if k < 2:
            continue
        sigma = float(np.clip(w[:, 1].mean(), 0.01, 0.99))
        rho = float(np.clip(max(0.0, 1.0 - 2.0 * w[:, 2].mean()), 0.01, 0.99))
        k_arr.append(k)
        rho_arr.append(rho)
        sigma_arr.append(sigma)
    if len(k_arr) < MIN_TRAJ_LEN or (max(k_arr) - min(k_arr)) < 2:
        return []
    return [(np.asarray(k_arr), np.asarray(rho_arr), np.asarray(sigma_arr))]


def get_institutional_trajectories(window: int = 5,
                                   max_countries: int = 80,
                                   seed: int = 42) -> List[Trajectory]:
    """Polity5 country-year. One trajectory per country.

    At year y, sliding window of `window` years:
      k_t   = number of polity-component indicators with ≥3 non-null values in window
      σ_t   = mean polity2 normalized to [0.01, 0.99]
      ρ_t   = mean pairwise indicator correlation across window
    """
    polity_path = REPO_ROOT / "data" / "institutional" / "polity5.xls"
    if not polity_path.exists():
        return []
    try:
        import pandas as pd
        df = pd.read_excel(polity_path)
    except Exception:
        return []
    cols = ["xconst", "xrcomp", "xropen", "xrreg", "exrec", "exconst"]
    cols = [c for c in cols if c in df.columns]
    if "polity2" not in df.columns or len(cols) < 3:
        return []
    df = df[df["polity2"].notna()].copy()
    df["polity2"] = pd.to_numeric(df["polity2"], errors="coerce")
    df = df[(df["polity2"] >= -10) & (df["polity2"] <= 10)].copy()
    # Polity5 uses -77, -88, -66 as missing-value sentinels. Mask them.
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df.loc[df[c] <= -50, c] = float("nan")
    rng = np.random.default_rng(seed)
    countries = list(df["country"].unique())
    rng.shuffle(countries)
    countries = countries[:max_countries]
    trajectories = []
    for country in countries:
        grp = df[df["country"] == country].sort_values("year").reset_index(drop=True)
        if len(grp) < MIN_TRAJ_LEN + window:
            continue
        k_arr, rho_arr, sigma_arr = [], [], []
        for y in range(0, len(grp) - window):
            w = grp.iloc[y: y + window]
            sigma = float(np.clip((w["polity2"].mean() + 10.0) / 20.0, 0.01, 0.99))
            non_null = [c for c in cols if w[c].notna().sum() >= 3]
            k = len(non_null)
            if k < 3:
                continue
            sub = w[non_null].apply(pd.to_numeric, errors="coerce").dropna()
            if len(sub) < 3:
                continue
            corr = sub.corr().values
            off = corr[np.triu_indices(corr.shape[0], k=1)]
            if len(off) == 0 or np.all(np.isnan(off)):
                continue
            rho = float(np.clip(abs(np.nanmean(off)), 0.0, 1.0))
            k_arr.append(k)
            rho_arr.append(rho)
            sigma_arr.append(sigma)
        if len(k_arr) < MIN_TRAJ_LEN:
            continue
        # Require k OR ρ to vary along the trajectory — if both are flat,
        # Kish regression has no signal axis. Categorical-indicator substrates
        # (Polity5, V-Dem) often have k constant but ρ varying year-to-year,
        # which is still admissible.
        k_a = np.asarray(k_arr); r_a = np.asarray(rho_arr)
        if (k_a.max() - k_a.min() < 2) and (r_a.max() - r_a.min() < 0.1):
            continue
        trajectories.append((k_a, r_a, np.asarray(sigma_arr)))
    return trajectories


def get_vdem_trajectories(window: int = 5, max_countries: int = 80,
                          seed: int = 42) -> List[Trajectory]:
    """V-Dem v15 country-year. One trajectory per country.

    At year y, sliding window of `window` years:
      k_t   = number of indicators with non-null values in window
      σ_t   = mean v2x_polyarchy
      ρ_t   = mean pairwise indicator correlation
    """
    try:
        import pandas as pd
    except ImportError:
        return []
    path = REPO_ROOT / "data" / "institutional" / "vdem" / "v-dem-v15.parquet"
    if not path.exists():
        return []
    indicators = ["v2x_polyarchy", "v2x_libdem", "v2x_partipdem",
                  "v2x_delibdem", "v2x_egaldem", "v2xeg_eqdr"]
    df = pd.read_parquet(path, columns=["country_name", "year"] + indicators)
    indicators = [c for c in indicators if c in df.columns]
    df = df.dropna(subset=["country_name", "year"])
    rng = np.random.default_rng(seed)
    countries = list(df["country_name"].unique())
    rng.shuffle(countries)
    countries = countries[:max_countries]
    trajectories = []
    for country in countries:
        grp = df[df["country_name"] == country].sort_values("year").reset_index(drop=True)
        if len(grp) < MIN_TRAJ_LEN + window:
            continue
        k_arr, rho_arr, sigma_arr = [], [], []
        for y in range(0, len(grp) - window):
            w = grp.iloc[y: y + window]
            polyarchy_w = w["v2x_polyarchy"].dropna()
            if len(polyarchy_w) == 0:
                continue
            sigma = float(np.clip(polyarchy_w.mean(), 0.01, 0.99))
            non_null = [c for c in indicators if w[c].notna().sum() >= 3]
            k = len(non_null)
            if k < 3:
                continue
            sub = w[non_null].apply(pd.to_numeric, errors="coerce").dropna()
            if len(sub) < 3:
                continue
            corr = sub.corr().values
            off = corr[np.triu_indices(corr.shape[0], k=1)]
            if len(off) == 0 or np.all(np.isnan(off)):
                continue
            rho = float(np.clip(abs(np.nanmean(off)), 0.0, 1.0))
            k_arr.append(k)
            rho_arr.append(rho)
            sigma_arr.append(sigma)
        if len(k_arr) < MIN_TRAJ_LEN:
            continue
        # Require k OR ρ to vary along the trajectory — if both are flat,
        # Kish regression has no signal axis. Categorical-indicator substrates
        # (Polity5, V-Dem) often have k constant but ρ varying year-to-year,
        # which is still admissible.
        k_a = np.asarray(k_arr); r_a = np.asarray(rho_arr)
        if (k_a.max() - k_a.min() < 2) and (r_a.max() - r_a.min() < 0.1):
            continue
        trajectories.append((k_a, r_a, np.asarray(sigma_arr)))
    return trajectories


def get_wgi_trajectories(window: int = 3, max_countries: int = 100,
                         seed: int = 42) -> List[Trajectory]:
    """Worldwide Governance Indicators (WGI) country-year. One trajectory per country.

    Third A4 substrate — continuous real-valued indicators, year range 1996-2023
    (28 years). The point of including WGI alongside V-Dem (continuous) and
    Polity5 (categorical) is to test the v2.0 finding that excess|φ| at A4 is
    driven by indicator-data-type rather than agency level.

    Hypothesis: if WGI looks like V-Dem (high excess|φ|), the indicator-type
    confounder is confirmed. If WGI looks like Polity5 (low excess|φ|), V-Dem
    is anomalous.

    At year y, sliding window of `window` years:
      k_t   = number of indicators with ≥3 non-null values in window (≤6)
      σ_t   = mean of (mean across indicators, normalized to [0.01, 0.99])
      ρ_t   = mean pairwise indicator correlation within window
    """
    try:
        import pandas as pd
    except ImportError:
        return []
    csv_path = REPO_ROOT / "data" / "institutional" / "wgi_processed.csv"
    if not csv_path.exists():
        return []
    indicators = ["CC.EST", "GE.EST", "PV.EST", "RQ.EST", "RL.EST", "VA.EST"]
    df = pd.read_csv(csv_path, usecols=["country", "year"] + indicators)
    df = df.dropna(subset=["country", "year"])
    rng = np.random.default_rng(seed)
    countries = list(df["country"].unique())
    rng.shuffle(countries)
    countries = countries[:max_countries]
    trajectories = []
    for country in countries:
        grp = df[df["country"] == country].sort_values("year").reset_index(drop=True)
        if len(grp) < MIN_TRAJ_LEN + window:
            continue
        k_arr, rho_arr, sigma_arr = [], [], []
        for y in range(0, len(grp) - window):
            w = grp.iloc[y: y + window]
            non_null = [c for c in indicators if w[c].notna().sum() >= 3]
            k = len(non_null)
            if k < 3:
                continue
            sub = w[non_null].apply(pd.to_numeric, errors="coerce").dropna()
            if len(sub) < 3:
                continue
            # WGI indicators range roughly [-2.5, +2.5]; map to (0, 1) for σ
            sigma = float(np.clip((sub.values.mean() + 2.5) / 5.0, 0.01, 0.99))
            corr = sub.corr().values
            off = corr[np.triu_indices(corr.shape[0], k=1)]
            if len(off) == 0 or np.all(np.isnan(off)):
                continue
            rho = float(np.clip(abs(np.nanmean(off)), 0.0, 1.0))
            k_arr.append(k)
            rho_arr.append(rho)
            sigma_arr.append(sigma)
        if len(k_arr) < MIN_TRAJ_LEN:
            continue
        k_a = np.asarray(k_arr); r_a = np.asarray(rho_arr)
        if (k_a.max() - k_a.min() < 2) and (r_a.max() - r_a.min() < 0.1):
            continue
        trajectories.append((k_a, r_a, np.asarray(sigma_arr)))
    return trajectories


SUBSTRATE_RUNGS["wgi"] = 4


TRAJECTORY_GETTERS = {
    "battery":       get_battery_trajectories,
    "alphafold":     get_alphafold_trajectories,
    "allen":         get_allen_trajectories,
    "biotime":       get_biotime_trajectories,
    "ciris":         get_ciris_trajectories,
    "institutional": get_institutional_trajectories,
    "vdem":          get_vdem_trajectories,
    "wgi":           get_wgi_trajectories,
}


# ─── Per-trajectory + per-substrate computation ──────────────────────


def trajectory_phi(traj: Trajectory, n_null: int, seed: int) -> Optional[dict]:
    """Compute mean|φ| of TIME-ORDERED Kish residuals and the shuffled null."""
    k, rho, sigma = traj
    if len(k) < MIN_TRAJ_LEN:
        return None
    fit = fit_kish_regression(k, rho, sigma, fit_intercept=True)
    omega = fit.omega
    n = len(omega)
    if n < MIN_TRAJ_LEN:
        return None
    _, _, phi_ord, _ = autocorr_decay_profile(omega, max_lag=MAX_LAG)
    rng = np.random.default_rng(seed)
    null_phis = []
    for _ in range(n_null):
        idx = rng.permutation(n)
        _, _, p, _ = autocorr_decay_profile(omega[idx], max_lag=MAX_LAG)
        null_phis.append(p)
    return {
        "n": int(n),
        "phi_ordered": float(phi_ord),
        "phi_null_median": float(np.median(null_phis)),
        "phi_null_p95": float(np.percentile(null_phis, 95)),
        "p1_r_squared": float(fit.r_squared),
    }


def compute_substrate(name: str, seed: int = 42) -> dict:
    getter = TRAJECTORY_GETTERS[name]
    trajectories = getter(seed=seed)
    if not trajectories:
        return {"substrate": name, "status": "no_data"}
    per_traj = []
    for i, traj in enumerate(trajectories):
        res = trajectory_phi(traj, n_null=N_NULL_SHUFFLES, seed=seed + i)
        if res is not None:
            per_traj.append(res)
    if len(per_traj) < 1:
        return {"substrate": name, "status": "no_valid_trajectories"}
    ords = np.array([r["phi_ordered"] for r in per_traj])
    nulls = np.array([r["phi_null_median"] for r in per_traj])
    excess = ords - nulls
    mean_ord = float(ords.mean())
    mean_null = float(nulls.mean())
    mean_excess = float(excess.mean())
    # Bootstrap CI on the substrate-level excess by resampling trajectories.
    rng = np.random.default_rng(seed + 31337)
    boot_excess = []
    n_traj = len(per_traj)
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n_traj, n_traj)
        boot_excess.append(float((ords[idx] - nulls[idx]).mean()))
    return {
        "substrate": name,
        "rung": SUBSTRATE_RUNGS[name],
        "status": "ok",
        "n_trajectories": n_traj,
        "mean_traj_length": float(np.mean([r["n"] for r in per_traj])),
        "mean_phi_ordered": mean_ord,
        "mean_phi_null": mean_null,
        "excess_phi": mean_excess,
        "excess_ci_low": float(np.percentile(boot_excess, 2.5)),
        "excess_ci_high": float(np.percentile(boot_excess, 97.5)),
        "frac_traj_excess_positive": float((excess > 0).mean()),
        "mean_p1_r_squared": float(np.mean([r["p1_r_squared"] for r in per_traj])),
    }


# ─── Cross-substrate Spearman ────────────────────────────────────────


def run_p2_v2(seed: int = 42) -> dict:
    print("Exp 2 P2 v2.0 — Time-ordered residuals × deterministic null")
    print("=" * 72)
    results = {}
    for name in TRAJECTORY_GETTERS:
        print(f"\n[{name}] rung A{SUBSTRATE_RUNGS[name]}")
        r = compute_substrate(name, seed=seed)
        if r.get("status") == "ok":
            print(f"  n_traj={r['n_trajectories']}  mean_len={r['mean_traj_length']:.1f}")
            print(f"  φ_ordered = {r['mean_phi_ordered']:.4f}   "
                  f"φ_null = {r['mean_phi_null']:.4f}")
            print(f"  excess|φ| = {r['excess_phi']:+.4f}  "
                  f"95% CI [{r['excess_ci_low']:+.4f}, {r['excess_ci_high']:+.4f}]")
            print(f"  trajectories with excess>0: "
                  f"{r['frac_traj_excess_positive']*100:.1f}%   "
                  f"P1 R²(trajectory-mean) = {r['mean_p1_r_squared']:.3f}")
        else:
            print(f"  {r}")
        results[name] = r

    valid = [r for r in results.values() if r.get("status") == "ok"]
    print()
    print("=" * 72)
    print(f"VALID SUBSTRATES: {len(valid)} / {len(TRAJECTORY_GETTERS)}")

    if len(valid) < 4:
        verdict = "INDETERMINATE"
        spearman_rho = float("nan")
        spearman_p = float("nan")
        print("  → INDETERMINATE (need ≥ 4 valid substrates)")
    else:
        from scipy.stats import spearmanr
        rungs = [r["rung"] for r in valid]
        excs = [r["excess_phi"] for r in valid]
        spearman_rho, spearman_p = spearmanr(rungs, excs)
        print()
        print("Per-substrate excess|φ| (sorted by rung):")
        for r in sorted(valid, key=lambda x: (x["rung"], x["substrate"])):
            print(f"  A{r['rung']} {r['substrate']:<14}: "
                  f"excess = {r['excess_phi']:+.4f}  "
                  f"[{r['excess_ci_low']:+.4f}, {r['excess_ci_high']:+.4f}]  "
                  f"(n_traj={r['n_trajectories']})")
        print()
        print(f"  Spearman ρ(rung, excess|φ|) = {spearman_rho:+.4f}  "
              f"(p = {spearman_p:.4g})")
        if np.isnan(spearman_rho):
            verdict = "INDETERMINATE_NaN"
        elif spearman_rho >= 0.7:
            verdict = "STRONG_PASS"
        elif spearman_rho >= 0.3:
            verdict = "WEAK_PASS"
        elif spearman_rho > -0.3:
            verdict = "INCONCLUSIVE"
        elif spearman_rho > -0.7:
            verdict = "WEAK_FAIL"
        else:
            verdict = "STRONG_FAIL"
        print()
        print(f"  VERDICT: {verdict}")

    return {
        "version": "v2.0",
        "metric": "excess|φ| = mean_phi_ordered - mean_phi_null",
        "n_valid_substrates": len(valid),
        "spearman_rho": (float(spearman_rho) if not np.isnan(spearman_rho)
                         else None),
        "spearman_p": (float(spearman_p) if not np.isnan(spearman_p) else None),
        "verdict": verdict,
        "per_substrate": results,
    }


if __name__ == "__main__":
    out = run_p2_v2(seed=42)
    out_dir = REPO_ROOT / "experiments/exp2_cross_substrate/data"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "p2_substrate_fractality_v2_results.json").write_text(
        json.dumps(out, indent=2, default=str))
    print()
    print(f"Wrote {out_dir / 'p2_substrate_fractality_v2_results.json'}")
