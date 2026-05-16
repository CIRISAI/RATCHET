#!/usr/bin/env python3
"""
Run A + B + D analyses against locked measurement pipeline.

A — Bootstrap CI on Gemini N>=3 subset
B — Sensitivity sweep (subset thresholds + retention threshold)
D — Compare with v0.1.0 calibration bundle's friction-conditional N_eff

Output: experiments/exp1b_boundary_active/data/measurement_analysis.md
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

# Import the locked measurement module
sys.path.insert(0, str(Path(__file__).parent))
from measurement import (  # noqa: E402
    Chain,
    bootstrap_neff_h,
    compute_neff_h,
    firing_distribution,
    load_chains_from_tee_dir,
    sensitivity_sweep,
)


def main():
    out_dir = Path(__file__).parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── A + B — Gemini v4_combined run ────────────────────────────────
    gemini_dir = Path("/tmp/exp1b_gemini")
    gemini_chains, gemini_excluded = load_chains_from_tee_dir(gemini_dir)
    n_total = len(gemini_chains)
    fd = firing_distribution(gemini_chains)
    n3_chains = [c for c in gemini_chains if c.n_fired >= 3]
    n4_chains = [c for c in gemini_chains if c.n_fired == 4]

    # Bootstrap CI on N>=3 (the load-bearing subset)
    boot_n3 = bootstrap_neff_h(n3_chains)
    boot_n4 = bootstrap_neff_h(n4_chains)
    boot_all = bootstrap_neff_h(gemini_chains)

    # Sensitivity sweep
    sens = sensitivity_sweep(gemini_chains)

    # ── D — v0.1.0 calibration bundle comparison ─────────────────────
    v01_chains_dir = Path("/home/emoore/RATCHET/release/calibration/crc-v1")
    # The bundle is the calibrated artifact; we need the raw export to
    # re-run boundary-active filtering. Check for ratchet_v0_1_0_calibration.
    ratchet_export = Path("/tmp/ratchet_v0_1_0_calibration")
    v01_chains: list[Chain] = []
    v01_excluded = 0
    if ratchet_export.is_dir():
        v01_chains, v01_excluded = load_chains_from_tee_dir(ratchet_export)
    v01_fd = firing_distribution(v01_chains) if v01_chains else None
    v01_n3 = [c for c in v01_chains if c.n_fired >= 3]
    v01_n4 = [c for c in v01_chains if c.n_fired == 4]
    boot_v01_all = bootstrap_neff_h(v01_chains) if v01_chains else None
    boot_v01_n3 = bootstrap_neff_h(v01_n3) if v01_n3 else None
    boot_v01_n4 = bootstrap_neff_h(v01_n4) if v01_n4 else None

    # ── Compose report ────────────────────────────────────────────────
    lines: list[str] = []
    lines.append("# Measurement Analysis — A + B + D\n")
    lines.append("Locked methodology: `experiments/exp1b_boundary_active/measurement.py`.\n")
    lines.append("All N_eff_H values computed via the same pipeline. Bootstrap CIs use deterministic seed.\n\n")

    # A
    lines.append("## A — Bootstrap CIs (Gemini-flash, v4_combined battery)\n\n")
    lines.append(f"Total chains: {n_total} (excluded {gemini_excluded})\n\n")
    lines.append("| Cohort | n | Point N_eff_H | Bootstrap mean | 95% CI | In [6.6, 7.6]? |\n")
    lines.append("|---|---|---|---|---|---|\n")
    for label, chains, b in [("All chains", gemini_chains, boot_all),
                              ("Conscience N>=3", n3_chains, boot_n3),
                              ("Conscience N==4", n4_chains, boot_n4)]:
        ci = f"[{b['ci95_low']:.3f}, {b['ci95_high']:.3f}]" if not math.isnan(b['ci95_low']) else "—"
        in_window = (
            not math.isnan(b['ci95_low'])
            and 6.6 <= b['ci95_low']
            and b['ci95_high'] <= 7.6
        )
        lines.append(f"| {label} | {len(chains)} | {b['point']:.3f} | {b['mean']:.3f} | {ci} | {'✓' if in_window else '✗'} |\n")
    lines.append("\n")

    # Firing distribution
    lines.append("## A — Firing-count distribution\n\n")
    lines.append("| N | n | % |\n|---|---|---|\n")
    for n in range(5):
        pct = 100 * fd[n] / n_total if n_total else 0
        lines.append(f"| {n} | {fd[n]} | {pct:.1f}% |\n")
    lines.append("\n")

    # B — sensitivity
    lines.append("## B — Sensitivity sweep (Gemini-flash, v4_combined)\n\n")
    lines.append("| Subset | n | N_eff_H | Retained dim |\n|---|---|---|---|\n")
    for r in sens["sensitivity"]:
        nh_s = f"{r['neff_h']:.3f}" if not math.isnan(r['neff_h']) else "—"
        lines.append(f"| `{r['subset']}` | {r['n']} | {nh_s} | {r['retained_dim']} |\n")
    lines.append("\n")

    # D — v0.1.0 comparison
    lines.append("## D — v0.1.0 calibration bundle comparison\n\n")
    if not v01_chains:
        lines.append("**v0.1.0 raw export not available** at `/tmp/ratchet_v0_1_0_calibration`. ")
        lines.append("This comparison requires the raw export (not the calibrated bundle artifact at "
                     "`release/calibration/crc-v1/`). Skipping D.\n\n")
        lines.append("The v0.1.0 calibration bundle itself reports `cohort_neff_h ≈ 7.07` (per "
                     "`release/calibration/crc-v1/bundle.yaml` headline) — that was computed on "
                     "the FULL n=264 corpus with imputation. To do a clean comparison we'd need to "
                     "re-run boundary-active filtering on the v0.1.0 raw traces.\n")
    else:
        lines.append(f"Total chains in v0.1.0 export: {len(v01_chains)} (excluded {v01_excluded})\n\n")
        if v01_fd:
            lines.append("| N | n | % |\n|---|---|---|\n")
            for n in range(5):
                pct = 100 * v01_fd[n] / len(v01_chains) if v01_chains else 0
                lines.append(f"| {n} | {v01_fd[n]} | {pct:.1f}% |\n")
            lines.append("\n")
        lines.append("**v0.1.0 cohort N_eff_H by subset:**\n\n")
        lines.append("| Cohort | n | Point | Bootstrap mean | 95% CI | In [6.6, 7.6]? |\n")
        lines.append("|---|---|---|---|---|---|\n")
        for label, chains, b in [("All v0.1.0", v01_chains, boot_v01_all),
                                  ("v0.1.0 N>=3", v01_n3, boot_v01_n3),
                                  ("v0.1.0 N==4", v01_n4, boot_v01_n4)]:
            if not b:
                lines.append(f"| {label} | {len(chains) if chains else 0} | — | — | — | — |\n")
                continue
            ci = f"[{b['ci95_low']:.3f}, {b['ci95_high']:.3f}]" if not math.isnan(b['ci95_low']) else "—"
            in_window = (
                not math.isnan(b['ci95_low'])
                and 6.6 <= b['ci95_low']
                and b['ci95_high'] <= 7.6
            )
            lines.append(f"| {label} | {len(chains)} | {b['point']:.3f} | {b['mean']:.3f} | {ci} | "
                         f"{'✓' if in_window else '✗'} |\n")
    lines.append("\n")

    # Headline summary
    lines.append("## Headline reading\n\n")
    n3_ci_in = (
        not math.isnan(boot_n3['ci95_low'])
        and 6.6 <= boot_n3['ci95_low']
        and boot_n3['ci95_high'] <= 7.6
    )
    n3_ci_str = (
        f"[{boot_n3['ci95_low']:.3f}, {boot_n3['ci95_high']:.3f}]"
        if not math.isnan(boot_n3['ci95_low']) else "—"
    )
    lines.append(f"- Gemini N>=3 point estimate: **{boot_n3['point']:.3f}** "
                 f"with 95% CI {n3_ci_str}\n")
    lines.append(f"- Inside locked [6.6, 7.6] window: **{'YES' if n3_ci_in else 'NO'}**\n\n")
    lines.append(f"- N=4 subset point: **{boot_n4['point']:.3f}** "
                 f"(n={len(n4_chains)})\n")

    out_md = out_dir / "measurement_analysis.md"
    out_md.write_text("".join(lines))
    print(f"Wrote {out_md}")

    # Also dump raw JSON
    out_json = out_dir / "measurement_analysis.json"
    out_json.write_text(json.dumps({
        "gemini": {
            "n_total": n_total, "excluded": gemini_excluded,
            "firing_distribution": fd,
            "all_chains_bootstrap": boot_all,
            "n3_bootstrap": boot_n3,
            "n4_bootstrap": boot_n4,
            "sensitivity": sens,
        },
        "v01_calibration": {
            "n_total": len(v01_chains),
            "excluded": v01_excluded,
            "firing_distribution": v01_fd,
            "all_bootstrap": boot_v01_all,
            "n3_bootstrap": boot_v01_n3,
            "n4_bootstrap": boot_v01_n4,
        } if v01_chains else None,
    }, indent=2, default=str))
    print(f"Wrote {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
