"""Run barrier + independent at F2's exact parameters (64 sensors, 50 samples),
5 trials each, to test whether the F2-vs-exp103 disagreement is run-to-run instability.
Follow-up to PREREG_exp103_replication.md (P4 refuted for barrier)."""
import json, time
import cupy as cp, numpy as np
exec(open('exp103_replication.py').read().split('def main()')[0].split('#!/usr/bin')[-1].split('"""',2)[-1])

arr = Array(64)
rows = []
for trial in range(5):
    for name, fn in (("independent", arr.independent), ("barrier", arr.barrier)):
        t = fn(50)
        rows.append({"trial": trial, "mode": name,
                     "rho_signed": rho_signed(t), "rho_abs": rho_abs(t),
                     "rho_abs_detrended": rho_abs_detrended(t),
                     "temp": gpu_state()["temp"]})
        print(f"  trial {trial} {name:<12} rho_abs={rows[-1]['rho_abs']:.4f} "
              f"signed={rows[-1]['rho_signed']:+.4f} temp={rows[-1]['temp']:.0f}C")

for m in ("independent", "barrier"):
    v = np.array([r["rho_abs"] for r in rows if r["mode"] == m])
    print(f"\n{m}: rho_abs mean={v.mean():.4f} sd={v.std():.4f} min={v.min():.4f} max={v.max():.4f}")
json.dump(rows, open("stability_results.json", "w"), indent=2)
