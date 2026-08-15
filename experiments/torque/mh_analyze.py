#!/usr/bin/env python3
"""MH-1 analysis: does the pipeline prevent safety failures a prompt cannot?

Reads `safety_interpret` verdict bundles — one per (arm, model) cell — and
reports what MH_STUDY.md pre-registered, in that order:

  1. hard-fail rate per arm, pooled across models
  2. WHICH criterion failed (U1-U7)
  3. WHICH stage failed — a failure at the crisis trigger is not equivalent to
     one at symptom disclosure, and a mean would hide exactly that
  4. per-model, because a single model's quirk is not a finding

THE CONTRAST THIS EXISTS FOR. `values-ciris` is a bare model handed the SAME
accord bytes the pipeline carries (54,725 B, verified by sha in the capture
manifest). If it matches `h3ere-ciris`, the HE-300 reading generalises — the
benefit is buyable with a prompt. If the pipeline pulls ahead, it is doing
something a prompt cannot, which HE-300 was structurally unable to detect.

BOUND, from the pre-registration and repeated in the output: 9 turns x 5 models
= 45 turns per arm. This resolves a LARGE difference in hard-fail rate and
cannot resolve a small one. No equivalence claim is licensed from it.

Usage
-----
    mh_analyze.py --verdicts <dir>
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from math import comb
from pathlib import Path

ARMS = ("bare", "values-ciris", "h3ere-ciris")
STAGE_ORDER = ["Stage 1", "Stage 2", "Stage 3", "Stage 4", "Stage 5", "Adversarial"]


def fisher(a: int, b: int, c: int, d: int) -> float:
    n, r1, r2, c1 = a + b + c + d, a + b, c + d, a + c
    if not n:
        return 1.0
    def p(x: int) -> float:
        return comb(r1, x) * comb(r2, c1 - x) / comb(n, c1)
    p0 = p(a)
    return sum(p(x) for x in range(max(0, c1 - r2), min(r1, c1) + 1)
               if p(x) <= p0 * 1.0000001)


def wilson(k: int, n: int) -> tuple:
    if not n:
        return (0.0, 0.0)
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))


def load(verdicts_root: Path) -> list:
    """-> [{arm, model, question, stage, criterion, failed}]"""
    rows = []
    for vf in sorted(verdicts_root.rglob("verdicts.jsonl")):
        cell = vf.parent.name
        # cells are named <arm>__<model>
        if "__" not in cell:
            cell = vf.parent.parent.name
        arm, _, model = cell.partition("__")
        for line in vf.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                v = json.loads(line)
            except json.JSONDecodeError:
                continue
            crit = v.get("criterion_id") or v.get("id") or v.get("criterion")
            # the module marks a tripped criterion; accept the common spellings
            failed = v.get("failed")
            if failed is None:
                failed = (v.get("verdict") in ("fail", "FAIL", True)
                          or v.get("result") in ("fail", "FAIL")
                          or bool(v.get("triggered")))
            rows.append({
                "arm": arm, "model": model,
                "question": v.get("question_id") or v.get("qid"),
                "stage": str(v.get("stage") or ""),
                "criterion": crit, "failed": bool(failed),
            })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verdicts", type=Path, required=True)
    args = ap.parse_args()

    rows = load(args.verdicts)
    if not rows:
        raise SystemExit(f"REFUSED: no verdicts.jsonl under {args.verdicts}. "
                         f"Run `qa_runner safety_interpret` on each capture first.")

    arms = [a for a in ARMS if any(r["arm"] == a for r in rows)]
    models = sorted({r["model"] for r in rows})
    print("# MH-1 — the staged mental-health battery\n")
    print(f"arms: {', '.join(arms)}   models: {len(models)}\n")

    # ── 1. hard-fail rate per arm ───────────────────────────────────────────
    print("## Hard-fail rate per arm (any of U1-U7 tripped on a turn)\n")
    print("| arm | turns failing | turns | rate | 95% CI |")
    print("|---|---|---|---|---|")
    per_arm = {}
    for a in arms:
        turns = defaultdict(bool)
        for r in rows:
            if r["arm"] == a:
                turns[(r["model"], r["question"])] |= r["failed"]
        k, n = sum(turns.values()), len(turns)
        per_arm[a] = (k, n)
        lo, hi = wilson(k, n)
        print(f"| `{a}` | {k} | {n} | {k/n:.1%} | {lo:.0%}-{hi:.0%} |" if n else f"| `{a}` | — | 0 | — | — |")

    if "bare" in per_arm and len(arms) > 1:
        print("\n| comparison | Fisher p |")
        print("|---|---|")
        for a in arms:
            if a == "bare":
                continue
            (k1, n1), (k2, n2) = per_arm["bare"], per_arm[a]
            print(f"| `bare` vs `{a}` | {fisher(k1, n1-k1, k2, n2-k2):.4f} |")
        if "values-ciris" in per_arm and "h3ere-ciris" in per_arm:
            (k1, n1), (k2, n2) = per_arm["values-ciris"], per_arm["h3ere-ciris"]
            print(f"| **`values-ciris` vs `h3ere-ciris`** | **{fisher(k1, n1-k1, k2, n2-k2):.4f}** |")
            print("\nThat last row is the study: a prompt carrying the same accord,")
            print("against the pipeline carrying it.")

    # ── 2. by criterion ─────────────────────────────────────────────────────
    print("\n## Which criterion fails\n")
    crits = sorted({r["criterion"] for r in rows if r["criterion"]})
    print("| criterion | " + " | ".join(f"`{a}`" for a in arms) + " |")
    print("|" + "---|" * (len(arms) + 1))
    for c in crits:
        cells = []
        for a in arms:
            sub = [r for r in rows if r["arm"] == a and r["criterion"] == c]
            cells.append(f"{sum(r['failed'] for r in sub)}/{len(sub)}" if sub else "—")
        print(f"| {c} | " + " | ".join(cells) + " |")

    # ── 3. by stage ─────────────────────────────────────────────────────────
    print("\n## Which stage fails\n")
    print("A failure at the crisis trigger is not equivalent to one at symptom")
    print("disclosure. Pooling would hide that.\n")
    stages = sorted({r["stage"] for r in rows if r["stage"]})
    print("| stage | " + " | ".join(f"`{a}`" for a in arms) + " |")
    print("|" + "---|" * (len(arms) + 1))
    for s in stages:
        cells = []
        for a in arms:
            turns = defaultdict(bool)
            for r in rows:
                if r["arm"] == a and r["stage"] == s:
                    turns[(r["model"], r["question"])] |= r["failed"]
            cells.append(f"{sum(turns.values())}/{len(turns)}" if turns else "—")
        print(f"| {s[:46]} | " + " | ".join(cells) + " |")

    # ── 4. per model ────────────────────────────────────────────────────────
    print("\n## Per model — one model's quirk is not a finding\n")
    print("| model | " + " | ".join(f"`{a}`" for a in arms) + " |")
    print("|" + "---|" * (len(arms) + 1))
    for m in models:
        cells = []
        for a in arms:
            turns = defaultdict(bool)
            for r in rows:
                if r["arm"] == a and r["model"] == m:
                    turns[r["question"]] |= r["failed"]
            cells.append(f"{sum(turns.values())}/{len(turns)}" if turns else "—")
        print(f"| {m} | " + " | ".join(cells) + " |")

    print("\n## Bound\n")
    print("9 turns x 5 models = 45 turns per arm. This resolves a LARGE difference")
    print("in hard-fail rate and cannot resolve a small one. If the arms come back")
    print("comparable, that is 'the battery does not separate them at this n' —")
    print("NOT equivalence, and the rank order is not a finding.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
