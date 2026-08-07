#!/usr/bin/env python3
"""Cohen's kappa over block-class annotations — overall and PER BOUNDARY.

CIRISAgent#976 sets the bar: two independent annotators, kappa >= 0.80 overall
**and** per-boundary on class pairs whose default disposition differs. A
class-set version without a kappa record cannot be cited.

Why per-boundary and not just overall. Overall kappa is dominated by the easy
classes. Two annotators can agree on every `structural` and `pragmatic` block,
score 0.85 overall, and still disagree on every `axiotic` vs `ontological`
call — which is the one pair where the default dispositions are OPPOSITE (vary
vs hold). Disagreement there is not a labelling quibble: it is a disagreement
about whether a block is the experiment's independent variable or something
held constant. Overall kappa hides exactly the disagreement that matters.

The boundaries that gate this campaign, because the two sides dispose
differently:

    axiotic (vary) | deontic     (hold)
    axiotic (vary) | ontological (hold)
    axiotic (vary) | pragmatic   (hold)
    axiotic (vary) | epistemic   (hold)

A third "annotator" is available and should be used: the SHIPPED labels, read
from a compose dump. Agreement between two humans-in-the-loop who never saw
them is the reliability claim; agreement with what the system actually composes
is the validity claim. They are different questions and both are worth having.

Usage
-----
    python3 kappa.py a.tsv b.tsv [--shipped dump.jsonl] [--prefix language_guidance]

Each annotation file is TSV or markdown-table: `part <sep> class <sep> ...`.
Only the first two columns are read; anything else is ignored.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

#: Pairs whose default dispositions differ (§10.2 table). These gate citation.
GATED_BOUNDARIES: Tuple[Tuple[str, str], ...] = (
    ("axiotic", "deontic"),
    ("axiotic", "ontological"),
    ("axiotic", "pragmatic"),
    ("axiotic", "epistemic"),
)

VALID = {
    "axiotic", "deontic", "pragmatic", "ontological", "epistemic", "empirical",
    "contingent", "procedural", "nomological", "structural", "axiomatic",
    "testimonial", "mixed",
}


def parse(path: Path) -> Dict[str, str]:
    """Read `part -> class` from TSV or a markdown table. Tolerant by design:
    an annotator's formatting should not silently drop rows from a reliability
    measurement."""
    out: Dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", "|--", "|:")):
            continue
        cells = [c.strip().strip("`*") for c in line.strip("|").split("|" if "|" in line else "\t")]
        if len(cells) < 2:
            continue
        part, cls = cells[0], cells[1].lower()
        if cls not in VALID or part.lower() in ("part", "block", "block_id"):
            continue
        out[part] = cls
    if not out:
        raise SystemExit(f"{path}: parsed zero annotations — refusing to report kappa over nothing")
    return out


def shipped_labels(dump: Path, prefix: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for line in dump.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("kind") == "compose_dump_meta":
            continue
        bid = row.get("block_id", "")
        if prefix in bid:
            out[bid.split(f"{prefix}.", 1)[-1]] = row["class"]
    return out


def kappa(a: List[str], b: List[str]) -> Optional[float]:
    """Cohen's kappa. None when undefined (single category on both sides)."""
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[k] / n) * (cb[k] / n) for k in set(ca) | set(cb))
    if pe == 1.0:
        # Both annotators used one category throughout. Agreement is total and
        # kappa is undefined — reporting 0.0 here would read as total
        # disagreement, which is the opposite of what happened.
        return None
    return (po - pe) / (1 - pe)


def report_pair(name_a: str, ann_a: Dict[str, str], name_b: str, ann_b: Dict[str, str]) -> bool:
    shared = sorted(set(ann_a) & set(ann_b))
    only_a, only_b = set(ann_a) - set(ann_b), set(ann_b) - set(ann_a)
    print(f"\n=== {name_a} vs {name_b} ===")
    print(f"items scored: {len(shared)}   (only in {name_a}: {len(only_a)}, only in {name_b}: {len(only_b)})")
    if only_a or only_b:
        print("  NOTE: kappa is computed over the intersection only; unmatched items are not evidence either way")
    if not shared:
        print("  no overlap — cannot compute")
        return False

    a = [ann_a[k] for k in shared]
    b = [ann_b[k] for k in shared]
    k = kappa(a, b)
    agree = sum(x == y for x, y in zip(a, b))
    print(f"  overall kappa: {k if k is None else round(k, 3)}   raw agreement: {agree}/{len(shared)}")

    ok = k is not None and k >= 0.80
    print(f"  overall >= 0.80: {'PASS' if ok else 'FAIL'}")

    print("  per gated boundary (default dispositions differ — these gate citation):")
    for x, y in GATED_BOUNDARIES:
        idx = [i for i, (p, q) in enumerate(zip(a, b)) if {p, q} <= {x, y}]
        if not idx:
            print(f"    {x:11s}|{y:11s}  no items — UNMEASURED, not passed")
            continue
        kb = kappa([a[i] for i in idx], [b[i] for i in idx])
        ag = sum(a[i] == b[i] for i in idx)
        verdict = "PASS" if (kb is not None and kb >= 0.80) else ("n/a" if kb is None else "FAIL")
        if kb is None and ag == len(idx):
            verdict = "PASS (total agreement, kappa undefined)"
            ok = ok and True
        elif kb is None or kb < 0.80:
            ok = False
        print(f"    {x:11s}|{y:11s}  n={len(idx):>2}  kappa={kb if kb is None else round(kb,3)}  agree={ag}/{len(idx)}  {verdict}")

    dis = [(k_, ann_a[k_], ann_b[k_]) for k_ in shared if ann_a[k_] != ann_b[k_]]
    if dis:
        print(f"  disagreements ({len(dis)}):")
        for part, x, y in dis:
            print(f"    {part:44s} {name_a}={x:12s} {name_b}={y}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--shipped", type=Path, help="compose dump jsonl — the validity check")
    ap.add_argument("--prefix", default="language_guidance")
    args = ap.parse_args()

    ann = {"A": parse(args.a), "B": parse(args.b)}
    if args.shipped:
        ann["SHIPPED"] = shipped_labels(args.shipped, args.prefix)

    results = {}
    for x, y in combinations(ann, 2):
        results[(x, y)] = report_pair(x, ann[x], y, ann[y])

    reliability = results.get(("A", "B"))
    print("\n" + "=" * 60)
    print(f"RELIABILITY (A vs B, the #976 gate): {'PASS' if reliability else 'FAIL'}")
    if "SHIPPED" in ann:
        val = [v for k, v in results.items() if "SHIPPED" in k]
        print(f"VALIDITY (annotators vs shipped labels): {'PASS' if all(val) else 'FAIL'}")
        print("  Reliability and validity are different claims. Two annotators can agree")
        print("  with each other and both disagree with what the system composes.")
    return 0 if reliability else 1


if __name__ == "__main__":
    sys.exit(main())
