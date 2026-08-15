#!/usr/bin/env python3
"""Is TORQUE ready to stake? Answer the position question, then say go or no-go.

The campaign runs ten-turn arcs. That design is only sound if an item's score
does not depend on where in the arc it sits. The pilot said it does: pooled
across four h3ere arms, position 1 scored 89% against 61% for positions 2-10
(p=0.00001), while the direct arms — same items, same positions — were flat.
Every one of those numbers predates the 2.9.14 bus-manager fix, which restored
channel history the agent had never been receiving.

So this re-asks the question on 2.9.14, on the SAME items, and reports:

  * position profile, with the direct arms as the item-difficulty control
    (direct arms run through direct_provider.py — no runtime, no identity — so
    they are version-independent and the pilot's remain valid on these items)
  * instrument health: coverage and instruction fidelity
  * a GO / NO-GO that names which finding drives it

NO-GO is the safe direction here. A position effect means ten-turn arcs confound
item with position, and no contrast measured on them can be trusted.

Usage
-----
    readiness.py --new <dir> --pilot <dir> --arcs <dir>
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
H3ERE = {"h3ere-ciris", "h3ere-alt", "h3ere-neutral", "h3ere-blank"}


def fisher(a: int, b: int, c: int, d: int) -> float:
    n, r1, r2, c1 = a + b + c + d, a + b, c + d, a + c
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
    """-> {(group, position): [correct, ...]}, plus fidelity counters."""
    acc = defaultdict(list)
    fid = defaultdict(lambda: [0, 0])  # group -> [scored, total]
    for res in root.rglob("results.jsonl"):
        if "en_he300" not in str(res):
            continue
        arm = arm_of(res)
        if arm is None:
            continue
        group = "h3ere" if arm in H3ERE else "direct"
        cell = next((c for c in arcs if c in str(res)), None)
        if not cell:
            continue
        gold = arcs[cell]
        rows = [json.loads(l) for l in res.read_text(encoding="utf-8").splitlines()
                if l.strip()]
        for i, (r, (g, cat)) in enumerate(zip(rows, gold), 1):
            t = next((r[f] for f in TEXT_FIELDS if r.get(f)), "") or ""
            lab, _, _ = score.extract(t, cat)
            fid[group][1] += 1
            if lab is None:
                continue
            fid[group][0] += 1
            acc[(group, i)].append(lab == g)
    return acc, fid


def profile(acc: dict, group: str) -> tuple:
    t1 = [x for (g, p), v in acc.items() if g == group and p == 1 for x in v]
    rest = [x for (g, p), v in acc.items() if g == group and p > 1 for x in v]
    return t1, rest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", type=Path, required=True, help="2.9.14 results root")
    ap.add_argument("--pilot", type=Path, required=True, help="pilot artifacts root")
    ap.add_argument("--arcs", type=Path, required=True)
    args = ap.parse_args()

    arcs = load_arcs(args.arcs)
    if not arcs:
        raise SystemExit(f"REFUSED: no arcs under {args.arcs}")

    new_acc, new_fid = collect(
        args.new, arcs,
        lambda p: next((part for part in p.parts if part in H3ERE), None))
    pil_acc, pil_fid = collect(
        args.pilot, arcs,
        lambda p: next((part.replace("results-pilot-", "").replace("-straight", "")
                        for part in p.parts if part.startswith("results-pilot-")), None))

    print("# TORQUE readiness\n")
    print("## Position profile — does an item's score depend on where it sits?\n")
    print("| build | group | pos 1 | 95% CI | pos 2-10 | 95% CI | Fisher p |")
    print("|---|---|---|---|---|---|---|")
    verdicts = {}
    for label, acc in (("2.9.14 (new)", new_acc), ("2.9.13 (pilot)", pil_acc)):
        for group in ("h3ere", "direct"):
            t1, rest = profile(acc, group)
            if not t1 or not rest:
                continue
            p = fisher(sum(t1), len(t1) - sum(t1), sum(rest), len(rest) - sum(rest))
            lo1, hi1 = wilson(sum(t1), len(t1))
            lo2, hi2 = wilson(sum(rest), len(rest))
            verdicts[(label, group)] = (sum(t1) / len(t1), sum(rest) / len(rest), p, len(t1))
            print(f"| {label} | {group} | {sum(t1)/len(t1):.0%} (n={len(t1)}) | "
                  f"{lo1:.0%}-{hi1:.0%} | {sum(rest)/len(rest):.0%} (n={len(rest)}) | "
                  f"{lo2:.0%}-{hi2:.0%} | {p:.4f} |")

    print("\n## Instrument health\n")
    print("| build | group | scored | total | fidelity |")
    print("|---|---|---|---|---|")
    for label, fid in (("2.9.14 (new)", new_fid), ("2.9.13 (pilot)", pil_fid)):
        for group, (s, t) in sorted(fid.items()):
            print(f"| {label} | {group} | {s} | {t} | {s/t:.2f} |" if t else "")

    print("\n## Verdict\n")
    key = ("2.9.14 (new)", "h3ere")
    if key not in verdicts:
        print("**NO-GO — the new run produced no h3ere data.** Nothing to judge.")
        return 1
    p1, prest, pval, n1 = verdicts[key]
    dctl = verdicts.get(("2.9.14 (new)", "direct")) or verdicts.get(("2.9.13 (pilot)", "direct"))
    print(f"h3ere on 2.9.14: position 1 **{p1:.0%}** (n={n1}) vs later **{prest:.0%}**, p={pval:.4f}.")
    if dctl:
        print(f"direct control on the same items: {dctl[0]:.0%} vs {dctl[1]:.0%}, p={dctl[2]:.4f}.")
    if pval < 0.05 and p1 > prest:
        print("\n**NO-GO.** The position effect persists after the bus-manager fix.")
        print("Ten-turn arcs confound item with position, so no contrast measured on")
        print("them can be trusted. The design must change before anything is staked —")
        print("one item per conversation is the option that matches CIRISBench and")
        print("makes the two corpora comparable.")
    elif pval >= 0.05 and n1 < 30:
        print("\n**NO-GO (underpowered).** No significant position effect, but n at")
        print("position 1 is too small to call it absent — this cannot distinguish")
        print("'no effect' from 'not enough data'. Add cells before deciding.")
    else:
        print("\n**GO on the position question.** No position effect at this power.")
        print("Remaining blockers are the ones already known and are not about arcs.")
    print("\nThis judges ARC GEOMETRY only. It does not license any specific contrast;")
    print("the probe-driven rescopes and the withdrawn `pipeline_effect` still stand.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
