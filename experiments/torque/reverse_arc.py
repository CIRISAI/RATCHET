#!/usr/bin/env python3
"""Emit an arc with its turn order reversed. The position-vs-item discriminator.

THE QUESTION THIS SETTLES. Concordance is 89.1% at turn 1 and 61.5% across turns
2-10 (p=0.00001), and the effect is pipeline-specific. Two explanations survive
every elimination so far and they are indistinguishable in a forward-only run:

    SESSION STATE   the agent degrades after its first task, whatever is asked.
                    The prompt's only turn-varying block is the agent's own
                    telemetry — `Total Tasks: 1` at turn 1, `2..10` after, then
                    token/cost/CO2 from turn 6.

    ITEM DIFFICULTY the item that happens to sit at position 1 in every arc is
                    easier. Arc construction balances each HALF on count and on
                    label-1 count; it does not balance POSITION 1.

Run the same items forward and reversed:

    effect follows POSITION  (turn 1 wins both times)  -> session state
    effect follows ITEMS     (same items win both)     -> construction artifact,
                                                          and the turn-1 result
                                                          was never real

The reversed arc holds the item SET exactly — same questions, same golds, same
categories — so nothing varies but order. `question_id` is carried unchanged so
the two runs can be joined per item.

Usage
-----
    reverse_arc.py --domain he300_axiotic_primary_a00 --safety-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import pathlib


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--safety-dir", required=True, type=pathlib.Path)
    ap.add_argument("--suffix", default="_rev")
    args = ap.parse_args()

    src = args.safety_dir / f"english_{args.domain}"
    f = src / f"v4_english_{args.domain}_arc.json"
    if not f.exists():
        raise SystemExit(f"REFUSED: no arc at {f}")

    arc = json.loads(f.read_text(encoding="utf-8"))
    qs = arc["questions"]
    new_domain = args.domain + args.suffix

    rev = list(reversed(qs))
    # Turn ids encode position and are re-stamped; question_id is NOT touched,
    # because joining the forward and reversed runs per item is the whole point.
    for i, q in enumerate(rev, 1):
        for key in ("turn", "turn_number", "position"):
            if key in q:
                q[key] = i
        if isinstance(q.get("question_id"), str) and "_t" in q["question_id"]:
            stem = q["question_id"].rsplit("_t", 1)[0]
            # keep the ORIGINAL id visible so the join is unambiguous
            q["original_question_id"] = q["question_id"]
            q["question_id"] = f"{stem}{args.suffix}_t{i:02d}"

    arc["questions"] = rev
    arc["cell"]["domain"] = new_domain
    arc["cell"]["derived_from"] = args.domain
    arc["cell"]["transform"] = "turn order reversed; item set identical"

    dst = args.safety_dir / f"english_{new_domain}"
    dst.mkdir(parents=True, exist_ok=True)
    out = dst / f"v4_english_{new_domain}_arc.json"
    out.write_text(json.dumps(arc, ensure_ascii=False, indent=2), encoding="utf-8")

    # Copy the rubric if the parent has one — qa_runner does not require it, but
    # a cell that is missing files its sibling has invites a false diagnosis.
    for extra in src.glob("*scoring_rubric*"):
        (dst / extra.name.replace(args.domain, new_domain)).write_text(
            extra.read_text(encoding="utf-8"), encoding="utf-8")

    fwd_g = [q["he300"]["gold_label"] for q in qs]
    rev_g = [q["he300"]["gold_label"] for q in rev]
    assert sorted(fwd_g) == sorted(rev_g), "item set changed — that would void the test"
    assert rev_g == fwd_g[::-1], "not a clean reversal"
    print(f"wrote {out}")
    print(f"  forward golds  {fwd_g}")
    print(f"  reversed golds {rev_g}")
    print(f"  item set identical: {sorted(fwd_g) == sorted(rev_g)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
