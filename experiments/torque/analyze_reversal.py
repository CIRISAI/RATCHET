#!/usr/bin/env python3
"""Position or item? Join a forward run against its reversed twin and decide.

The turn-1 result (89.1% vs 61.5%, p=0.00001) has two surviving explanations
and a forward-only run cannot tell them apart:

    SESSION STATE    the agent degrades after its first task regardless of what
                     is asked. Predicts: turn 1 wins in BOTH runs, and the items
                     that win change between them.
    ITEM DIFFICULTY  position 1 happens to hold an easier item. Predicts: the
                     SAME items win in both runs, and position carries nothing.

Arc construction balances each half on count and on label-1 count. It does not
balance position 1, so the second explanation was never excluded — only assumed
away. This script tests it directly.

Usage
-----
    analyze_reversal.py --forward <results_dir> --reversed <results_dir> \
        --arc-dir <safety_dir> --domain he300_axiotic_primary_a00
"""

from __future__ import annotations

import argparse
import json
import sys
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score  # noqa: E402

TEXT_FIELDS = ("agent_response", "response_text", "speak_content")


def fisher(a: int, b: int, c: int, d: int) -> float:
    n, r1, r2, c1 = a + b + c + d, a + b, c + d, a + c
    def p(x: int) -> float:
        return comb(r1, x) * comb(r2, c1 - x) / comb(n, c1)
    p0 = p(a)
    return sum(p(x) for x in range(max(0, c1 - r2), min(r1, c1) + 1)
               if p(x) <= p0 * 1.0000001)


def load_arc(arc_dir: Path, domain: str) -> list:
    f = arc_dir / f"english_{domain}" / f"v4_english_{domain}_arc.json"
    arc = json.loads(f.read_text(encoding="utf-8"))
    out = []
    for q in arc["questions"]:
        # The reversed arc re-stamps question_id but preserves the original, so
        # the per-item join is unambiguous in both directions.
        key = q.get("original_question_id") or q["question_id"]
        out.append((key, q["he300"]["gold_label"], q["category"]))
    return out


def load_results(d: Path) -> list:
    f = next(d.rglob("results.jsonl"), None)
    if f is None:
        raise SystemExit(f"REFUSED: no results.jsonl under {d}")
    return [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]


def scored(rows: list, arc: list) -> list:
    """-> [(item_key, position, correct|None)] in turn order."""
    out = []
    for i, (r, (key, gold, cat)) in enumerate(zip(rows, arc), 1):
        text = next((r[f] for f in TEXT_FIELDS if r.get(f)), "")
        lab, _, _ = score.extract(text, cat)
        out.append((key, i, None if lab is None else lab == gold))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--forward", type=Path, required=True)
    ap.add_argument("--reversed", dest="rev", type=Path, required=True)
    ap.add_argument("--arc-dir", type=Path, required=True)
    ap.add_argument("--domain", required=True)
    args = ap.parse_args()

    fa = load_arc(args.arc_dir, args.domain)
    ra = load_arc(args.arc_dir, args.domain + "_rev")
    fs = scored(load_results(args.forward), fa)
    rs = scored(load_results(args.rev), ra)

    print("# Position or item?\n")
    print("| run | turn 1 | turns 2-n | unknown |")
    print("|---|---|---|---|")
    pos = {}
    for name, s in (("forward", fs), ("reversed", rs)):
        t1 = [c for k, p, c in s if p == 1 and c is not None]
        rest = [c for k, p, c in s if p > 1 and c is not None]
        unk = sum(1 for _, _, c in s if c is None)
        pos[name] = (t1, rest)
        t1s = f"{sum(t1)}/{len(t1)}" if t1 else "—"
        rs_ = f"{sum(rest)}/{len(rest)} ({sum(rest)/len(rest):.0%})" if rest else "—"
        print(f"| {name} | {t1s} | {rs_} | {unk} |")

    # POSITION evidence, pooled across both runs.
    t1_all = pos["forward"][0] + pos["reversed"][0]
    rest_all = pos["forward"][1] + pos["reversed"][1]
    if t1_all and rest_all:
        p = fisher(sum(t1_all), len(t1_all) - sum(t1_all),
                   sum(rest_all), len(rest_all) - sum(rest_all))
        print(f"\n**Position, pooled:** turn 1 {sum(t1_all)}/{len(t1_all)} "
              f"({sum(t1_all)/len(t1_all):.0%}) vs later "
              f"{sum(rest_all)}/{len(rest_all)} ({sum(rest_all)/len(rest_all):.0%}), "
              f"Fisher p={p:.4f}")

    # ITEM evidence: same item, two positions. Does correctness track the item?
    fmap = {k: c for k, _, c in fs}
    rmap = {k: c for k, _, c in rs}
    both = [k for k in fmap if k in rmap and fmap[k] is not None and rmap[k] is not None]
    agree = sum(1 for k in both if fmap[k] == rmap[k])
    print(f"\n**Item, joined:** {len(both)} items scored in both runs; "
          f"the same item got the same verdict {agree}/{len(both)} times "
          f"({agree/len(both):.0%})" if both else "\n**Item:** no joinable items")

    if both:
        # An item effect means correctness is a property of the item and should
        # survive the move; a session effect means it should not.
        moved = [k for k in both
                 if dict((kk, pp) for kk, pp, _ in fs)[k] != dict((kk, pp) for kk, pp, _ in rs)[k]]
        print(f"items that changed position: {len(moved)}/{len(both)}")

    print("\n## Reading\n")
    print("| observation | conclusion |")
    print("|---|---|")
    print("| turn 1 wins in BOTH runs, item agreement near chance | **session state** |")
    print("| same items win in both, position carries nothing | **construction artifact** |")
    print("| neither separates at this n | underpowered — say so, do not pick |")
    print("\nOne arc pair is a DIRECTION, not an effect size. n=10 per run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
