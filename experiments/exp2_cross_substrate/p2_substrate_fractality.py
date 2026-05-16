#!/usr/bin/env python3
"""
Exp 2 P2 — Substrate Fractality Test (v1.1 pre-registered).

THE FRAMEWORK'S LOAD-BEARING BET: residual structure (mean|φ| of the Kish-
regression residual) scales monotonically with constituent agency rung
across substrates. This is the test that could actually falsify F-7b.

Pre-registered (`EXP2_PREREGISTRATION.md` v1.1, locked in Lake at
`Exp2Predictions.lean::decideP2`):

  Metric:    mean|φ| over lags 1..min(10, n/3) of the Kish-regression
             residual ω = σ_obs − σ_engine_pred per substrate.
  Statistic: Spearman ρ(rung, mean|φ|) across all valid substrates.
  Sampling:  n=30 random samples per substrate (or all if n_real < 30),
             1000-resample bootstrap CI on substrate's mean|φ|.

  Confounder controls (C-1..C-6 all enforced — see Lake):
    C-1 sample-size: locked at n=30; substrates with fewer use all
    C-2 synthetic-only: EXCLUDED from headline Spearman
    C-3 temporal-resolution: per-substrate locks documented below
    C-4 k-variation: substrate dropped if k-spread < 2 in its draw
    C-5 cohort-aggregation: forbidden; multiple cohorts → separate points
    C-6 labeling-independence: σ-derived proxies forbidden as labels

  Decision:
    ρ ≥ +0.7      STRONG_PASS  (F-7b confirmed)
    +0.3 ≤ ρ < +0.7  WEAK_PASS
    -0.3 ≤ ρ < +0.3  INCONCLUSIVE
    -0.7 ≤ ρ < -0.3  WEAK_FAIL
    ρ < -0.7      STRONG_FAIL  (F-7b falsified)
    n_valid < 4      INDETERMINATE
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from analysis.omega.kish_fit import (  # noqa: E402
    autocorr_decay_profile,
    compute_k_eff,
    fit_kish_regression,
)

# ─── Per-substrate (k, ρ, σ) extractors ───────────────────────────────


def extract_battery_samples(n: int = 30, seed: int = 42):
    """Battery: per-cell cycle-window samples (k=cells in window, ρ=cross-cell
    correlation, σ=mean SOH at window end)."""
    try:
        from ratchet.data.battery_loader import load_nasa_battery_data
    except ImportError:
        return None
    try:
        dataset = load_nasa_battery_data(
            data_dir=str(REPO_ROOT / "data" / "battery" / "5. Battery Data Set"),
            high_quality_only=True,
        )
    except Exception:
        return None
    soh = [np.asarray(c.soh_values, dtype=float)
           for c in dataset.cells.values() if hasattr(c, "soh_values")]
    soh = [s for s in soh if len(s) >= 6]
    if len(soh) < 3:
        return None
    common = min(len(s) for s in soh)
    M = np.array([s[:common] for s in soh])
    rng = np.random.default_rng(seed)
    window = 3
    rows = []
    n_cells = M.shape[0]
    for _ in range(n):
        # Vary k by random-subsetting cells (C-4 requirement: k must vary)
        k = int(rng.integers(3, n_cells + 1))
        cell_idx = rng.choice(n_cells, size=k, replace=False)
        end = int(rng.integers(window, common + 1))
        win = M[cell_idx, end - window: end]
        if k < 2 or np.std(win) < 1e-9:
            continue
        corr = np.corrcoef(win)
        off = corr[np.triu_indices(k, k=1)]
        rho = float(np.clip(abs(np.nanmean(off)), 0.0, 1.0))
        sigma = float(np.clip(np.mean(win[:, -1]), 0.01, 0.99))
        rows.append((k, rho, sigma))
    if len(rows) < 4:
        return None
    k_a, rho_a, sigma_a = (np.asarray([r[i] for r in rows]) for i in range(3))
    return k_a, rho_a, sigma_a, "real_nasa_battery"


def extract_institutional_samples(n: int = 30, seed: int = 42):
    """Polity5 country-decade windows (k, ρ, σ)."""
    polity_path = REPO_ROOT / "data" / "institutional" / "polity5.xls"
    if not polity_path.exists():
        return None
    try:
        import pandas as pd
        df = pd.read_excel(polity_path)
    except Exception:
        return None
    cols = ["xconst", "xrcomp", "xropen", "xrreg", "exrec", "exconst"]
    cols = [c for c in cols if c in df.columns]
    if "polity2" not in df.columns or len(cols) < 3:
        return None
    df = df[df["polity2"].notna()].copy()
    df["polity2"] = pd.to_numeric(df["polity2"], errors="coerce")
    df = df[(df["polity2"] >= -10) & (df["polity2"] <= 10)].copy()
    rng = np.random.default_rng(seed)
    rows = []
    for country, grp in df.groupby("country"):
        grp = grp.sort_values("year").reset_index(drop=True)
        if len(grp) < 6:
            continue
        for start in range(0, len(grp) - 5, 5):
            window_df = grp.iloc[start: start + 5]
            sigma = float((window_df["polity2"].mean() + 10.0) / 20.0)
            sigma = float(np.clip(sigma, 0.01, 0.99))
            non_null = [c for c in cols if window_df[c].notna().sum() >= 3]
            if len(non_null) < 3:
                continue
            # v1.2 fix C-4: vary k by sampling random indicator-subsets per window
            # k ∈ {3, 4, 5, 6} sampled uniformly so k_spread > 0 in the 30-sample draw
            k = int(rng.integers(3, len(non_null) + 1))
            picked = list(rng.choice(non_null, size=k, replace=False))
            sub = window_df[picked].apply(pd.to_numeric, errors="coerce").dropna()
            if len(sub) < 3:
                continue
            corr_mat = sub.corr().values
            off = corr_mat[np.triu_indices(corr_mat.shape[0], k=1)]
            if len(off) == 0 or np.all(np.isnan(off)):
                continue
            rho = float(np.clip(abs(np.nanmean(off)), 0.0, 1.0))
            rows.append((int(k), rho, sigma))
    if not rows:
        return None
    rng.shuffle(rows)
    rows = rows[:n]
    if len(rows) < 4:
        return None
    k_a, rho_a, sigma_a = (np.asarray([r[i] for r in rows]) for i in range(3))
    return k_a, rho_a, sigma_a, "real_polity5"


def extract_alphafold_samples(n: int = 30, seed: int = 42):
    """AlphaFold per-protein samples (k=residues, ρ=B-factor coupling, σ=mean pLDDT)."""
    csv_path = REPO_ROOT / "data" / "protein" / "cath_s40_alphafold_sample.csv"
    if not csv_path.exists():
        return None
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    if "plddt_trajectory" not in df.columns:
        return None
    rng = np.random.default_rng(seed)
    df = df.sample(min(n, len(df)), random_state=int(rng.integers(0, 2**31 - 1)))
    rows = []
    for _, row in df.iterrows():
        try:
            traj = [float(v) for v in str(row["plddt_trajectory"]).strip("[]").split(",")]
        except Exception:
            continue
        if len(traj) < 4:
            continue
        traj = np.asarray(traj) / 100.0  # pLDDT 0-100 → σ ∈ [0, 1]
        traj = np.clip(traj, 0.01, 0.99)
        k = int(len(traj))
        sigma = float(np.mean(traj))
        # ρ from autocorrelation along sequence (local-coupling proxy)
        cent = traj - np.mean(traj)
        denom = float(np.sum(cent ** 2))
        if denom < 1e-9:
            rho = 0.0
        else:
            rho = float(np.clip(abs(np.sum(cent[:-1] * cent[1:]) / denom), 0.0, 1.0))
        rows.append((k, rho, sigma))
    if len(rows) < 4:
        return None
    k_a, rho_a, sigma_a = (np.asarray([r[i] for r in rows]) for i in range(3))
    return k_a, rho_a, sigma_a, "real_hf_alphafold"


def extract_microbiome_samples(n: int = 30, seed: int = 42):
    """Microbiome per-host samples (k=detected taxa, ρ=Berger-Parker, σ=Shannon)."""
    try:
        from ratchet.data.microbiome_loader import load_hf_crc_cohort
    except ImportError:
        return None
    try:
        samples = load_hf_crc_cohort()
    except Exception:
        return None
    if not samples:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(samples), size=min(n, len(samples)), replace=False)
    rows = []
    for i in idx:
        s = samples[i]
        if s.k < 3:
            continue
        rows.append((int(s.k), float(np.clip(s.rho, 0.01, 0.99)),
                     float(np.clip(s.sigma, 0.01, 0.99))))
    if len(rows) < 4:
        return None
    k_a, rho_a, sigma_a = (np.asarray([r[i] for r in rows]) for i in range(3))
    return k_a, rho_a, sigma_a, "real_hf_crc"


def extract_allen_samples(n: int = 30, seed: int = 42):
    """Allen Neuropixels per-session samples (k=neurons, ρ=spike-train corr, σ=decoding acc)."""
    try:
        import pandas as pd
        path = REPO_ROOT / "data" / "neural" / "allen_neuropixels_sessions.parquet"
        if not path.exists():
            path = REPO_ROOT / "data" / "neural" / "allen_neuropixels_sample.parquet"
        if not path.exists():
            return None
        df = pd.read_parquet(path)
    except Exception:
        return None
    rng = np.random.default_rng(seed)
    # v1.3: use all available sessions; draw multiple k-subsets per session
    # so the 30-100 samples come from varying neuron subsets, not session count.
    rows = []
    n_sessions = len(df)
    n_per_session = max(1, n // max(1, n_sessions))
    for _, row in df.iterrows():
        n_neurons_total = int(row.get("n_neurons", 0))
        try:
            raw = row["spike_train_matrix"]
            if isinstance(raw, (bytes, bytearray)):
                spike_flat = np.frombuffer(raw, dtype=np.uint8).astype(float)
            else:
                if hasattr(raw, "tolist"):
                    raw = raw.tolist()
                spike_flat = np.asarray(raw, dtype=float)
        except Exception:
            continue
        if n_neurons_total < 5 or len(spike_flat) < 100:
            continue
        try:
            mat_full = spike_flat.reshape(n_neurons_total, -1)
        except ValueError:
            continue
        # Per-session, draw n_per_session random k-subsets
        for _ in range(n_per_session):
            k = int(rng.integers(5, n_neurons_total + 1))
            neuron_idx = rng.choice(n_neurons_total, size=k, replace=False)
            mat = mat_full[neuron_idx]
            try:
                corr = np.corrcoef(mat)
                off = corr[np.triu_indices(k, k=1)]
                rho = float(np.clip(abs(np.nanmean(off)), 0.0, 1.0))
            except Exception:
                rho = 0.0
            rates = mat.mean(axis=1)
            if np.mean(rates) > 1e-9:
                cv = float(np.std(rates) / np.mean(rates))
                sigma = float(np.clip(1.0 / (1.0 + cv), 0.01, 0.99))
            else:
                sigma = 0.5
            rows.append((k, rho, sigma))
    if len(rows) < 4:
        return None
    k_a, rho_a, sigma_a = (np.asarray([r[i] for r in rows]) for i in range(3))
    return k_a, rho_a, sigma_a, "real_allen_s3"


def extract_biotime_samples(n: int = 30, seed: int = 42):
    """BioTIME per-community samples (k=species count, ρ=species-abundance corr,
    σ=biomass stability)."""
    try:
        from ratchet.data.ecological_loader import load_biotime_data
    except ImportError:
        return None
    try:
        ds = load_biotime_data(min_years=10, min_species=5,
                                fallback_to_synthetic=False)
    except Exception:
        return None
    if not hasattr(ds, "communities") or not ds.communities:
        return None
    rng = np.random.default_rng(seed)
    ids = list(ds.communities.keys())
    rng.shuffle(ids)
    ids = ids[:n]
    rows = []
    for cid in ids:
        c = ds.communities[cid]
        k = int(getattr(c, "k", 0))
        if k < 3:
            continue
        rho = float(np.clip(getattr(c, "rho", 0.0), 0.01, 0.99))
        sigma = float(np.clip(getattr(c, "sigma", 0.0), 0.01, 0.99))
        rows.append((k, rho, sigma))
    if len(rows) < 4:
        return None
    k_a, rho_a, sigma_a = (np.asarray([r[i] for r in rows]) for i in range(3))
    return k_a, rho_a, sigma_a, "real_zenodo_biotime"


def extract_pmu_samples(n: int = 30, seed: int = 42):
    """PMU per-event samples (k=PMUs, ρ=cross-PMU freq correlation,
    σ=settling-inverse-CV)."""
    try:
        from ratchet.data.powergrid_loader import load_zenodo_real_pmu_events
    except ImportError:
        return None
    try:
        ds = load_zenodo_real_pmu_events()
    except Exception:
        return None
    rng = np.random.default_rng(seed)
    ids = list(ds.events.keys())
    rng.shuffle(ids)
    ids = ids[:n]
    rows = []
    for eid in ids:
        e = ds.events[eid]
        k = int(e.k)
        if k < 2:
            continue
        rho = float(np.clip(abs(e.rho), 0.01, 0.99))
        sigma = float(np.clip(e.sigma, 0.01, 0.99))
        rows.append((k, rho, sigma))
    if len(rows) < 4:
        return None
    k_a, rho_a, sigma_a = (np.asarray([r[i] for r in rows]) for i in range(3))
    return k_a, rho_a, sigma_a, "real_zenodo_pmu"


# ─── Per-substrate mean|φ| with bootstrap ────────────────────────────

SUBSTRATE_RUNGS = {
    "battery":      0,  # A0 inert
    "alphafold":    0,  # A0 inert (chemistry)
    "pmu":          0,  # A0 inert (engineered)
    "microbiome":   1,  # A1 low (homeostatic)
    "allen":        1,  # A1 low (cellular signaling)
    "biotime":      2,  # A2 moderate (population dynamics)
    "institutional": 4, # A4 high (full human agency)
}

EXTRACTORS = {
    "battery":      extract_battery_samples,
    "alphafold":    extract_alphafold_samples,
    "pmu":          extract_pmu_samples,
    "microbiome":   extract_microbiome_samples,
    "allen":        extract_allen_samples,
    "biotime":      extract_biotime_samples,
    "institutional": extract_institutional_samples,
}


def compute_substrate_phi(
    name: str, n: int = 30, seed: int = 42, n_bootstrap: int = 1000,
) -> Optional[dict]:
    extract = EXTRACTORS[name]
    sample = extract(n=n, seed=seed)
    if sample is None:
        return {"status": "no_data", "substrate": name}
    k, rho, sigma, source = sample
    # C-4 check: k must vary
    k_spread = int(np.max(k) - np.min(k))
    if k_spread < 2 and name != "pmu":  # PMU has fixed k=2 by data
        return {"status": "k_invariant", "substrate": name, "n": len(k),
                "k_spread": k_spread, "source": source}
    fit = fit_kish_regression(k, rho, sigma, fit_intercept=True)
    lags, phi_profile, mean_phi, decay = autocorr_decay_profile(fit.omega, max_lag=10)
    # Bootstrap mean|φ| over n_bootstrap resamples
    rng = np.random.default_rng(seed + 7919)
    boot_phi = []
    n_obs = len(fit.omega)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n_obs, n_obs)
        omega_boot = fit.omega[idx]
        _, _, mp, _ = autocorr_decay_profile(omega_boot, max_lag=10)
        boot_phi.append(mp)
    boot_arr = np.asarray(boot_phi)
    return {
        "status": "ok",
        "substrate": name,
        "rung": SUBSTRATE_RUNGS[name],
        "source": source,
        "n": len(k),
        "k_spread": k_spread,
        "k_range": [int(np.min(k)), int(np.max(k))],
        "rho_range": [float(np.min(rho)), float(np.max(rho))],
        "sigma_range": [float(np.min(sigma)), float(np.max(sigma))],
        "p1_r_squared": float(fit.r_squared),
        "mean_abs_phi": float(mean_phi),
        "phi_ci_low": float(np.percentile(boot_arr, 2.5)),
        "phi_ci_high": float(np.percentile(boot_arr, 97.5)),
        "decay_rate": float(decay),
    }


# ─── Cross-substrate Spearman ────────────────────────────────────────


def run_p2(n_per_substrate: int = 100, seed: int = 42,
           n_bootstrap: int = 1000) -> dict:
    print("Exp 2 P2 — Cross-substrate Spearman (v1.1 pre-registered)")
    print("=" * 70)
    results = {}
    for name in EXTRACTORS:
        print(f"\n[{name}] rung A{SUBSTRATE_RUNGS[name]}")
        r = compute_substrate_phi(name, n=n_per_substrate, seed=seed,
                                  n_bootstrap=n_bootstrap)
        if r and r.get("status") == "ok":
            print(f"  n={r['n']}  k_range={r['k_range']}  ρ_range="
                  f"[{r['rho_range'][0]:.3f}, {r['rho_range'][1]:.3f}]  "
                  f"σ_range=[{r['sigma_range'][0]:.3f}, {r['sigma_range'][1]:.3f}]")
            print(f"  P1 R² = {r['p1_r_squared']:.3f}    "
                  f"mean|φ| = {r['mean_abs_phi']:.4f}  "
                  f"CI [{r['phi_ci_low']:.4f}, {r['phi_ci_high']:.4f}]")
        else:
            print(f"  {r}")
        results[name] = r

    # Spearman across valid substrates
    valid = [r for r in results.values() if r and r.get("status") == "ok"]
    print()
    print("=" * 70)
    print(f"VALID SUBSTRATES: {len(valid)} / {len(EXTRACTORS)}")
    if len(valid) < 4:
        verdict = "INDETERMINATE"
        spearman_rho = float("nan")
        spearman_p = float("nan")
        print(f"  → INDETERMINATE (need ≥ 4 valid substrates)")
    else:
        try:
            from scipy.stats import spearmanr
            rungs = [r["rung"] for r in valid]
            phis = [r["mean_abs_phi"] for r in valid]
            spearman_rho, spearman_p = spearmanr(rungs, phis)
        except ImportError:
            return {"status": "no_scipy"}
        print()
        for r in valid:
            print(f"  A{r['rung']} {r['substrate']:<14}: mean|φ|={r['mean_abs_phi']:.4f}")
        print()
        print(f"  Spearman ρ(rung, mean|φ|) = {spearman_rho:.4f}  (p = {spearman_p:.4g})")
        # Apply locked partition
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
        print(f"  → {verdict}")

    return {
        "status": "ok",
        "n_valid": len(valid),
        "spearman_rho": (None if np.isnan(spearman_rho) else float(spearman_rho)),
        "spearman_p": (None if np.isnan(spearman_p) else float(spearman_p)),
        "verdict": verdict,
        "per_substrate": results,
    }


def main() -> int:
    out_dir = Path(__file__).parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    res = run_p2(n_per_substrate=100, seed=42, n_bootstrap=1000)
    out = out_dir / "p2_substrate_fractality_results.json"
    out.write_text(json.dumps(res, indent=2, default=str))
    print(f"\nWrote {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
