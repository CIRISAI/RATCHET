#!/usr/bin/env python3
"""Project TORQUE_REGIME.yaml onto the schema `compose_dump gate` validates.

The gate takes an `ExperimentalRegimeV2` — the Phase-1 subset of FSD §10.3, with
`extra="forbid"`. TORQUE_REGIME.yaml is the campaign's own record and carries a
great deal the gate has no field for: kills, void conditions, contrast
instruments, declared limits, directional expectations, the authoring-boundary
rule. Validating it directly produces 48 errors that all say the same thing —
"this document is not that schema."

There are two wrong ways to fix that and one right one.

WRONG: strip our document down to what the gate accepts. That deletes the record
to please a validator, and the deleted parts are the ones a reader needs most.

WRONG: relax the gate. `extra="forbid"` is what makes the check mean something —
a typo'd arm name would otherwise pass silently.

RIGHT: project. Our document stays the record; this emits a minimal conformant
view of it for the machine check. The projection is DERIVED, never hand-edited,
so the two cannot drift: change the regime and re-run.

Usage
-----
    gate_projection.py --regime TORQUE_REGIME.yaml --out /tmp/regime-gate.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import yaml

#: Fields the gate schema declares. Anything else in our regime is campaign
#: record and is dropped from the projection — not from the regime.
ARM_FIELDS = ("harness", "replace", "disable", "inject", "safety_review")


def project(r: Dict[str, Any]) -> Dict[str, Any]:
    arms = {
        name: {k: v for k, v in body.items() if k in ARM_FIELDS}
        for name, body in r["arms"].items()
    }

    holds = dict(r["holds"])
    # The gate wants a corpus IDENTIFIER; our regime carries the whole corpus
    # spec in its own top-level block. Point at the pinned manifest.
    holds["corpus"] = r["corpus"]["primary"]
    holds.pop("adapter_set", None)
    holds.pop("template", None)

    repeats = {k: v for k, v in r["repeats"].items()
               if k in ("unit", "conversations_per_cell", "variance_source",
                        "seeds", "comparison_policy", "mde")}

    out: Dict[str, Any] = {
        "regime_schema": r.get("schema", "experimental_regime/v2"),
        "regime_id": r["regime_id"],
        "class_set_version": r["class_set_version"],
        "hypothesis": r["hypothesis"],
        "arms": arms,
        "repeats": repeats,
        "holds": holds,
    }
    if "contrasts" in r:
        # Our regime writes a contrast as the string "a - b" because that is how
        # anyone reading it thinks about a difference. The gate wants the two
        # sides named. Parse rather than duplicate: a hand-kept second copy is a
        # place for the two to disagree.
        out["contrasts"] = {}
        for k, v in r["contrasts"].items():
            if isinstance(v, dict):
                out["contrasts"][k] = {"minuend": v["minuend"], "subtrahend": v["subtrahend"]}
                continue
            minuend, sep, subtrahend = str(v).partition(" - ")
            if not sep:
                raise SystemExit(
                    f"REFUSED: contrast {k!r} is {v!r}, which is not '<arm> - <arm>'. "
                    f"Guessing which side is the minuend would silently flip the "
                    f"sign of a reported effect."
                )
            known = set(r["arms"])
            for side in (minuend.strip(), subtrahend.strip()):
                if side not in known:
                    raise SystemExit(
                        f"REFUSED: contrast {k!r} names arm {side!r}, which is not "
                        f"declared. Declared: {sorted(known)}."
                    )
            out["contrasts"][k] = {"minuend": minuend.strip(),
                                   "subtrahend": subtrahend.strip()}
    if "dv" in r:
        out["dv"] = r["dv"]

    # The gate reads `pins.residue_digest` and refuses without it: it is the
    # identity of the text the dump does NOT cover, and a gate that passed
    # without knowing that would be certifying coverage it never checked.
    out["pins"] = {"residue_digest": r["pins"]["residue_digest"]}
    if "blocks" in r:
        # Our blocks section carries provenance alongside the entries —
        # `_generated_from`, `_measured`, `_blocks_total`, `_class_distribution`.
        # That metadata is the reason anyone can trust the entries; it is also
        # not a block. Underscore-prefixed keys are ours, the rest are the gate's.
        out["blocks"] = {k: v for k, v in r["blocks"].items() if not k.startswith("_")}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regime", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    r = yaml.safe_load(args.regime.read_text(encoding="utf-8"))
    p = project(r)
    args.out.write_text(yaml.safe_dump(p, sort_keys=False, allow_unicode=True),
                        encoding="utf-8")

    dropped = sorted(set(r) - {"schema", "regime_id", "class_set_version",
                               "hypothesis", "arms", "contrasts", "dv",
                               "repeats", "holds", "corpus"})
    print(f"wrote {args.out}")
    print(f"projected {len(p)} fields; {len(dropped)} regime sections not in the "
          f"gate schema and NOT dropped from the regime:")
    print("  " + ", ".join(dropped))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
