#!/usr/bin/env python3
"""Do the pipeline arms share one position profile? Permutation, not chi-square.

WHY THIS REPLACES THE CHI-SQUARE. The manifest reported summed chi2 = 9.44 on
30 df and read "below the 30 expected by chance" as STRONGER evidence of
homogeneity. That is backwards. 9.44 is ~2.65 sd below the mean of chi2_30 —
a value only ~0.04% of draws reach. A statistic that far below expectation is
not reassurance; it is evidence the reference distribution is WRONG.

And the reason is visible in the design. The four arms are scored on the SAME
items at the SAME positions, so their errors are correlated. chi2_30 assumes
independent cells. Correlated arms produce exactly this under-dispersion, so the
chi2 was mis-calibrated from the start — the number was fine and the warrant was
not, which is the failure family that survives every numerical check.

THE FIX. Permute ARM LABELS WITHIN each (cell, position). That is the exchange-
ability the null actually claims — "which arm produced which verdict is
arbitrary" — and it holds item difficulty and position FIXED by construction,
so the correlation the chi2 ignored is built into the null distribution instead
of assumed away.

Usage
-----
    homogeneity_perm.py --results <dir> --arcs <dir> [--draws 10000] [--dv correct|verdict]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score  # noqa: E402

TEXT_FIELDS = ("agent_response", "response_text", "speak_content")
ARMS = ("h3ere-ciris", "h3ere-alt", "h3ere-neutral", "h3ere-blank")


def load_arcs(d: Path) -> dict:
    out = {}
    for f in d.rglob("v4_*_arc.json"):
        try:
            a = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        qs = a.get("questions") or []
        if qs and "he300" in qs[0]:
            out[a["cell"]["domain"]] = [(q["he300"]["gold_label"], q["category"])
                                        for q in qs]
    return out


def collect(root: Path, arcs: dict, dv: str) -> dict:
    """-> {(cell, pos): {arm: value}}"""
    cells = defaultdict(dict)
    for arm in ARMS:
        for res in (root / arm).rglob("results.jsonl"):
            if not res.parent.name.startswith("en_he300"):
                continue
            hits = [c for c in arcs if c in str(res)]
            if not hits:
                continue
            cell = max(hits, key=len)
            gold = arcs[cell]
            rows = [json.loads(l) for l in res.read_text(encoding="utf-8").splitlines()
                    if l.strip()]
            for i, (r, (g, cat)) in enumerate(zip(rows, gold), 1):
                t = next((r[f] for f in TEXT_FIELDS if r.get(f)), "") or ""
                lab, _, _ = score.extract(t, cat)
                if lab is None:
                    continue
                cells[(cell, i)][arm] = (lab == g) if dv == "correct" else lab
    return cells


def statistic(cells: dict) -> float:
    """Summed per-position spread across arms.

    Deliberately NOT a chi-square: its calibration is the thing in question. This
    is a plain sum of squared deviations of each arm's per-position mean from the
    across-arm mean, which the permutation distribution calibrates for us.
    """
    per_pos = defaultdict(lambda: defaultdict(list))
    for (_, pos), d in cells.items():
        for arm, v in d.items():
            per_pos[pos][arm].append(v)
    total = 0.0
    for pos, arms in per_pos.items():
        means = [sum(v) / len(v) for v in arms.values() if v]
        if len(means) < 2:
            continue
        gm = sum(means) / len(means)
        total += sum((m - gm) ** 2 for m in means)
    return total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--arcs", type=Path, required=True)
    ap.add_argument("--draws", type=int, default=10000)
    ap.add_argument("--dv", choices=("correct", "verdict"), default="correct")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    arcs = load_arcs(args.arcs)
    cells = collect(args.results, arcs, args.dv)
    full = {k: v for k, v in cells.items() if len(v) == len(ARMS)}
    if not full:
        raise SystemExit("REFUSED: no (cell, position) has all four arms — "
                         "permuting labels needs complete cells.")

    obs = statistic(full)
    rng = random.Random(args.seed)
    ge = 0
    null = []
    for _ in range(args.draws):
        perm = {}
        for k, d in full.items():
            vals = list(d.values())
            rng.shuffle(vals)
            perm[k] = dict(zip(ARMS, vals))
        s = statistic(perm)
        null.append(s)
        ge += (s >= obs)
    p = (ge + 1) / (args.draws + 1)
    null.sort()
    mean = sum(null) / len(null)

    print(f"# Arm-profile homogeneity — permutation test (DV: {args.dv})\n")
    print(f"complete (cell, position) cells: **{len(full)}** of {len(cells)} "
          f"({len(ARMS)} arms each)\n")
    print("| quantity | value |")
    print("|---|---|")
    print(f"| observed spread statistic | {obs:.5f} |")
    print(f"| permutation null mean | {mean:.5f} |")
    print(f"| null 5th / 50th / 95th pct | {null[int(.05*len(null))]:.5f} / "
          f"{null[len(null)//2]:.5f} / {null[int(.95*len(null))]:.5f} |")
    print(f"| draws | {args.draws:,} |")
    print(f"| **p (observed >= null)** | **{p:.4f}** |")

    print("\n## Reading\n")
    if p > 0.05:
        print(f"p = {p:.4f}: the observed spread across arms is **within** what")
        print("relabelling produces. The arms are consistent with sharing one")
        print("position profile, so the skew is common mode and cancels in an")
        print("arm-vs-arm contrast.")
    else:
        print(f"p = {p:.4f}: the arms differ by MORE than relabelling produces.")
        print("The skew is NOT common mode; an arm x position interaction exists")
        print("among the pipeline arms and pooled contrasts are not interpretable.")
    print("\nThis null holds item and position fixed and permutes only arm labels,")
    print("so the between-arm correlation that mis-calibrated the chi-square is")
    print("part of the reference distribution rather than an assumption.")
    print("\nNOTE: a large p is not proof of homogeneity — it is failure to detect")
    print("heterogeneity at this n. The equivalence framing applies here too.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
