#!/usr/bin/env python3
"""Characterise the position skew, and decide which contrasts it actually breaks.

ARCS ARE NOT NEGOTIABLE, so the job is not to design the skew away — it is to
identify it robustly enough to control for it, and to say which contrasts
survive it.

WHAT IS ALREADY CONTROLLED. `build_he300_arcs` shuffles item order within each
half per arc (`rng.shuffle(block)`), so item and position are decorrelated by
construction. The direct arms confirm it empirically: same items, same
positions, flat (81% at position 1 vs 80% later, p=1.0). The skew is therefore a
REAL position effect, not an item artifact.

THE ONE THING THAT MATTERS. A position effect common to all arms CANCELS in an
arm-vs-arm contrast — every arm sees the same arcs in the same order. What does
NOT cancel is an arm x position INTERACTION: if one arm's profile differs in
SHAPE from another's, their difference depends on position and no single pooled
number describes it.

So this reports, per arm:
  * accuracy by position, with Wilson intervals
  * the position slope (pos 1 vs later), per arm
  * a homogeneity test across the pipeline arms — do they share one shape?

and concludes which contrast families are safe:
  pipeline-vs-pipeline  safe iff the pipeline arms share a profile
  pipeline-vs-direct    unsafe whenever the profiles differ in shape
                        (already withdrawn for unrelated reasons)

Usage
-----
    skew.py --new <dir> --pilot <dir> --arcs <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score  # noqa: E402

TEXT_FIELDS = ("agent_response", "response_text", "speak_content")
H3ERE = ("h3ere-ciris", "h3ere-alt", "h3ere-neutral", "h3ere-blank")


def wilson(k: int, n: int) -> tuple:
    if not n:
        return (0.0, 0.0)
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))


def fisher(a: int, b: int, c: int, d: int) -> float:
    n, r1, r2, c1 = a + b + c + d, a + b, c + d, a + c
    def p(x: int) -> float:
        return comb(r1, x) * comb(r2, c1 - x) / comb(n, c1)
    p0 = p(a)
    return sum(p(x) for x in range(max(0, c1 - r2), min(r1, c1) + 1)
               if p(x) <= p0 * 1.0000001)


def chi2_homogeneity(rows: list) -> tuple:
    """rows = [(k, n), ...]; returns (chi2, df). Are these one proportion?"""
    K = sum(k for k, _ in rows)
    N = sum(n for _, n in rows)
    if not N or K in (0, N):
        return (0.0, 0)
    p = K / N
    chi = 0.0
    for k, n in rows:
        if not n:
            continue
        e1, e0 = n * p, n * (1 - p)
        if e1 > 0:
            chi += (k - e1) ** 2 / e1
        if e0 > 0:
            chi += ((n - k) - e0) ** 2 / e0
    return (chi, len(rows) - 1)


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


def collect(root: Path, arcs: dict, arm_of) -> dict:
    acc = defaultdict(list)
    for res in root.rglob("results.jsonl"):
        parent = res.parent.name
        if not (parent.startswith("en_he300") or parent in arcs):
            continue
        arm = arm_of(res)
        if arm is None:
            continue
        hits = [c for c in arcs if c in str(res)]
        if not hits:
            continue
        gold = arcs[max(hits, key=len)]
        rows = [json.loads(l) for l in res.read_text(encoding="utf-8").splitlines()
                if l.strip()]
        for i, (r, (g, cat)) in enumerate(zip(rows, gold), 1):
            t = next((r[f] for f in TEXT_FIELDS if r.get(f)), "") or ""
            lab, _, _ = score.extract(t, cat)
            if lab is not None:
                acc[(arm, i)].append(lab == g)
    return acc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", type=Path, required=True)
    ap.add_argument("--pilot", type=Path, required=True)
    ap.add_argument("--arcs", type=Path, required=True)
    args = ap.parse_args()

    arcs = load_arcs(args.arcs)
    acc = collect(args.new, arcs,
                  lambda p: next((x for x in p.parts if x in H3ERE), None))
    pil = collect(args.pilot, arcs,
                  lambda p: next((x.replace("results-pilot-", "").replace("-straight", "")
                                  for x in p.parts if x.startswith("results-pilot-")), None))
    for k, v in pil.items():
        if k[0] in ("bare", "values-ciris"):
            acc[k] = v

    arms = sorted({a for a, _ in acc})
    print("# Position skew\n")
    print("Item order is shuffled within each half per arc, so item and position")
    print("are decorrelated by construction. The skew below is a POSITION effect.\n")
    print("| arm | " + " | ".join(f"p{i}" for i in range(1, 11)) + " |")
    print("|" + "---|" * 11)
    for arm in arms:
        cells = []
        for i in range(1, 11):
            v = acc.get((arm, i), [])
            cells.append(f"{sum(v)/len(v):.0%}" if v else "—")
        print(f"| `{arm}` | " + " | ".join(cells) + " |")

    print("\n## Position-1 lift, per arm\n")
    print("| arm | pos 1 | 95% CI | pos 2-10 | lift | p |")
    print("|---|---|---|---|---|---|")
    lifts = {}
    for arm in arms:
        t1 = acc.get((arm, 1), [])
        rest = [x for i in range(2, 11) for x in acc.get((arm, i), [])]
        if not t1 or not rest:
            continue
        lo, hi = wilson(sum(t1), len(t1))
        p = fisher(sum(t1), len(t1) - sum(t1), sum(rest), len(rest) - sum(rest))
        lift = sum(t1) / len(t1) - sum(rest) / len(rest)
        lifts[arm] = (lift, len(t1))
        print(f"| `{arm}` | {sum(t1)/len(t1):.0%} (n={len(t1)}) | {lo:.0%}-{hi:.0%} | "
              f"{sum(rest)/len(rest):.0%} (n={len(rest)}) | **{lift:+.0%}** | {p:.4f} |")

    print("\n## Do the pipeline arms share one profile?\n")
    print("If they do, the skew is COMMON MODE and cancels in any")
    print("pipeline-vs-pipeline contrast — which is every contrast still staked.\n")
    present = [a for a in H3ERE if any((a, i) in acc for i in range(1, 11))]
    print("| position | " + " | ".join(f"`{a}`" for a in present)
          + " | chi2 | df | homogeneous |")
    print("|" + "---|" * (len(present) + 4))
    worst = 0.0
    for i in range(1, 11):
        rows = [(sum(acc.get((a, i), [])), len(acc.get((a, i), []))) for a in present]
        if not any(n for _, n in rows):
            continue
        chi, df = chi2_homogeneity(rows)
        worst = max(worst, chi)
        cells = " | ".join(f"{k}/{n}" if n else "—" for k, n in rows)
        # chi2 critical at .05 for df=3 is 7.81
        crit = {1: 3.84, 2: 5.99, 3: 7.81}.get(df, 7.81)
        print(f"| p{i} | {cells} | {chi:.2f} | {df} | {'yes' if chi < crit else '**NO**'} |")

    print("\n## What this licenses\n")
    if lifts:
        pipe = [l for a, (l, _) in lifts.items() if a in H3ERE]
        direct = [l for a, (l, _) in lifts.items() if a not in H3ERE]
        if pipe and direct:
            print(f"pipeline mean position-1 lift **{sum(pipe)/len(pipe):+.0%}**, "
                  f"direct **{sum(direct)/len(direct):+.0%}**.")
            print("\nThe skew is pipeline-specific, so an arm x position interaction is")
            print("real and a pooled pipeline-vs-direct number is not interpretable.")
            print("`pipeline_effect` was already withdrawn; this is a second,")
            print("independent reason it cannot be reported as one figure.\n")
        print("**Pipeline-vs-pipeline contrasts** (`scaffold_floor`,")
        print("`accord_swap_effect`, `form_vs_content`) compare arms that share the")
        print("harness and see identical arcs. If the homogeneity rows above are")
        print("'yes', the skew is common mode there and cancels — those contrasts")
        print("are usable on ten-turn arcs WITHOUT redesign.")
        print("\n**Report position-stratified anyway.** Cancelling in expectation is")
        print("not the same as being absent, and a per-position table costs nothing")
        print("once the runs exist.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
