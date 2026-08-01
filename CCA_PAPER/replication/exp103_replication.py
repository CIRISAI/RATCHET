#!/usr/bin/env python3
"""
Exp 103 replication and adjudication — RATCHET#13.

Runs the four Exp 103 measurement modes verbatim (transcribed from
CIRISArray/experiments/exp103_software_injection.py) and adds what no prior version
computed: a shuffle null, a common-mode-removal comparison, both estimators, and two
sample counts.

Pre-registration: PREREG_exp103_replication.md, sha256 7ffa2ee4386b2083727ea6ac03cd88b8bb414680017b65279b34f16fc4236516
"""

import json
import subprocess
import time
from pathlib import Path

import cupy as cp
import numpy as np

HERE = Path(__file__).parent
OUT = HERE / "exp103_replication_results.json"

N_SENSORS = 64
WORKLOAD = 256
NULL_DRAWS = 200
RNG = np.random.default_rng(103)


def gpu_state():
    q = "temperature.gpu,clocks.current.sm,power.draw,utilization.gpu"
    r = subprocess.run(
        ["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    t, c, p, u = [x.strip() for x in r.stdout.strip().split(",")]
    return {"temp": float(t), "clock": float(c), "power": float(p), "util": float(u)}


class Array:
    def __init__(self, n_sensors=N_SENSORS):
        self.n = n_sensors
        self.streams = [cp.cuda.Stream() for _ in range(n_sensors)]
        self.workloads = [cp.random.rand(WORKLOAD, WORKLOAD) for _ in range(n_sensors)]
        self.shared = cp.random.rand(WORKLOAD, WORKLOAD)

    # --- the four Exp 103 modes, transcribed ---------------------------------
    def independent(self, m):
        out = np.zeros((self.n, m))
        for i in range(self.n):
            with self.streams[i]:
                for s in range(m):
                    t0 = time.perf_counter_ns()
                    self.workloads[i] = cp.sin(self.workloads[i])
                    self.streams[i].synchronize()
                    out[i, s] = time.perf_counter_ns() - t0
        return out

    def barrier(self, m):
        out = np.zeros((self.n, m))
        for s in range(m):
            cp.cuda.stream.get_current_stream().synchronize()
            for i in range(self.n):
                with self.streams[i]:
                    t0 = time.perf_counter_ns()
                    self.workloads[i] = cp.sin(self.workloads[i])
                    self.streams[i].synchronize()
                    out[i, s] = time.perf_counter_ns() - t0
        return out

    def shared_workload(self, m):
        out = np.zeros((self.n, m))
        for i in range(self.n):
            with self.streams[i]:
                for s in range(m):
                    t0 = time.perf_counter_ns()
                    self.shared = cp.sin(self.shared)
                    self.streams[i].synchronize()
                    out[i, s] = time.perf_counter_ns() - t0
        return out

    def lockstep(self, m):
        out = np.zeros((self.n, m))
        for s in range(m):
            for i in range(self.n):
                self.streams[i].use()
                self.workloads[i] = cp.sin(self.workloads[i])
            cp.cuda.Device().synchronize()
            t0 = time.perf_counter_ns()
            for i in range(self.n):
                self.streams[i].use()
                self.workloads[i] = cp.sin(self.workloads[i])
            cp.cuda.Device().synchronize()
            t1 = time.perf_counter_ns()
            out[:, s] = t1 - t0          # <-- one scalar to all sensors (H1)
        return out


# --- estimators --------------------------------------------------------------
def _upper(t):
    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.corrcoef(t)
    C = np.nan_to_num(C, nan=0.0)
    return C[np.triu_indices(t.shape[0], k=1)]


def rho_signed(t):
    return float(np.mean(_upper(t)))


def rho_abs(t):
    return float(np.mean(np.abs(_upper(t))))


def rho_abs_detrended(t):
    """Remove the across-sensor common mode at each sample index, then recompute."""
    return rho_abs(t - t.mean(axis=0, keepdims=True))


def shuffle_null(t, draws=NULL_DRAWS):
    """Permute each sensor's series independently: kills cross-sensor structure,
    preserves every marginal."""
    vals = []
    for _ in range(draws):
        p = np.stack([RNG.permutation(row) for row in t])
        vals.append(rho_abs(p))
    v = np.array(vals)
    return {"mean": float(v.mean()), "p95": float(np.percentile(v, 95)),
            "max": float(v.max())}


def identical_rows(t):
    return bool(np.all(t == t[0]))


def keff(rho, k=N_SENSORS):
    return k / (1 + rho * (k - 1)) if rho > 0 else float(k)


def analyze(name, t, m):
    ra = rho_abs(t)
    null = shuffle_null(t)
    det = rho_abs_detrended(t)
    return {
        "mode": name, "n_samples": m,
        "rho_signed": rho_signed(t),
        "rho_abs": ra,
        "rho_abs_null_mean": null["mean"],
        "rho_abs_null_p95": null["p95"],
        "rho_abs_detrended": det,
        "keff_from_rho_abs": keff(ra),
        "identical_rows": identical_rows(t),
        "above_null": bool(ra > null["p95"]),
        "detrended_within_null": bool(det <= null["p95"]),
        "timing_mean_ns": float(t.mean()),
        "timing_cv": float(t.std() / t.mean()) if t.mean() else 0.0,
    }


def main():
    pre = gpu_state()
    print(f"GPU before: {pre}")
    if pre["util"] > 5:
        print("WARNING: GPU not idle at baseline; results may be contended.")

    results = {"gpu_before": pre, "n_sensors": N_SENSORS,
               "prereg_sha256": "7ffa2ee4386b2083727ea6ac03cd88b8bb414680017b65279b34f16fc4236516",
               "runs": []}

    arr = Array()
    modes = [("independent", arr.independent), ("barrier", arr.barrier),
             ("shared_workload", arr.shared_workload), ("lockstep", arr.lockstep)]

    for m in (30, 200):
        for name, fn in modes:
            st = gpu_state()
            t = fn(m)
            r = analyze(name, t, m)
            r["gpu_temp_at_run"] = st["temp"]
            results["runs"].append(r)
            print(f"  n={m:>3} {name:<16} rho_abs={r['rho_abs']:.4f} "
                  f"(null p95={r['rho_abs_null_p95']:.4f})  "
                  f"signed={r['rho_signed']:+.4f}  "
                  f"detrended={r['rho_abs_detrended']:.4f}  "
                  f"identical_rows={r['identical_rows']}")

    results["gpu_after"] = gpu_state()
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nGPU after: {results['gpu_after']}")
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
