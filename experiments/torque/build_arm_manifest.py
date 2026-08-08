#!/usr/bin/env python3
"""Turn a verified corpus into a runnable arm manifest.

This is the step between "the corpus is authored" and "the arm can run", and it
is where the polyglot problem surfaced: `corpus:accord.*` is THREE keys reaching
different parts of the pipeline, not one.

    accord.localized            54,725 B   ASPDMA family        (8 steps)
    accord.polyglot_compressed   7,215 B   round-1 DMAs         (5 steps)
    accord.polyglot_full       180,522 B   ALL FOUR CONSCIENCES (8 steps)

The two polyglot forms are woven from 15 languages by semantic weight, not
translated. An English alt corpus cannot replace them without destroying that
property, and building polyglot-neutral and polyglot-Anthropic corpora is a
research programme in its own right rather than a build step.

So BOTH ARMS RUN MONOLINGUAL ENGLISH on all three keys — CIRIS English in the
CIRIS arm, alt English in the alt arm. Language coverage is held constant and
cannot confound; the manipulation is complete across all 21 steps.

The cost is a stated domain limit, not a term in the estimate: **the campaign
measures a monolingual variant of H3ERE, not shipped H3ERE.** Every arm,
including `h3ere-ciris`, runs a configuration the product does not ship. See
POLYGLOT_PROBLEM.md.

Usage
-----
    build_arm_manifest.py --arm h3ere-ciris --accord <ciris_en.txt> --out m.json
    build_arm_manifest.py --arm h3ere-alt   --accord <alt_en.txt>   --out m.json
    build_arm_manifest.py --arm h3ere-blank --blank-axiotic         --out m.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

#: All three forms take the SAME text. That is the decision, not an oversight —
#: see the module docstring. Anything that sets them differently per arm has
#: reintroduced the language confound.
ACCORD_KEYS = ("accord.localized", "accord.polyglot_compressed", "accord.polyglot_full")

#: The PDMA framing block. A FOURTH corpus key, and it varies per arm exactly as
#: the accord does — CIRIS English, alt English, neutral English, or empty.
#: Missing it would leave the six Foundational Principles stated in CIRIS terms
#: inside every arm, which is the manipulation failing silently at its loudest
#: point: the PDMA's own roster.
FRAMING_KEY = "polyglot.pdma_framing"

#: Identity text. ONTOLOGICAL — it says what the agent IS, and it is HELD
#: IDENTICAL IN EVERY ARM, read from the pinned template so it cannot drift.
#:
#: Held does not mean value-free. `role_description` on the pinned template is
#: ~35 lines of first-person value claims and it reaches every LLM call. See
#: TORQUE_REGIME.yaml `held_value_background` — the blank arm is not a value-free
#: floor, and this is why.
TEMPLATE_FIELDS = ("description", "domain", "role_description")

#: Pinned in TORQUE_REGIME.yaml. Checked, not trusted: a template swapped under
#: the same filename is exactly the confound this whole apparatus exists to catch,
#: and `he-300-benchmark.yaml` — the obvious pairing for this corpus — permits
#: ONLY `speak`, which would make selected_verb and defer_rate structurally
#: constant and kill the action tier before the first call.
TEMPLATE_SHA256 = "75f2d11dfc03c91b5b45ae8493109aeec9b73a185a0fb1579a0df98d7e9bf46b"

#: Written into every manifest so the limit travels with the artifact rather
#: than living only in a document nobody opens at analysis time.
DOMAIN_LIMIT = (
    "MONOLINGUAL VARIANT. All three accord forms carry English text in every arm. "
    "The shipped agent runs two of them polyglot (woven from 15 languages). This "
    "run does not test the polyglot configuration, and no result from it licenses "
    "a claim about the shipped agent without assuming the weave does not interact "
    "with the values manipulation — untested."
)


def run(cmd, cwd: Path) -> tuple[int, str]:
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=900)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def baseline(agent_root: Path, locales: str) -> Dict:
    rc, out = run(
        [sys.executable, "-m", "ciris_engine.logic.utils.research_overrides", "baseline", locales],
        agent_root,
    )
    if rc != 0:
        raise SystemExit(f"baseline failed ({rc}):\n{out[-1500:]}")
    start = out.find("{")
    if start < 0:
        raise SystemExit("no JSON in baseline output")
    manifest, _ = json.JSONDecoder().raw_decode(out[start:])
    manifest.pop("_baseline_note", None)
    return manifest


def template_identity(agent_root: Path, template: str) -> Dict[str, str]:
    """The three ONTOLOGICAL identity fields, read from the pinned template.

    Read rather than hardcoded, and sha-checked rather than trusted. The 2026-07-31
    audit turned on citing prose instead of the artifact that produced it; a
    template pinned by NAME and changed underneath is the same failure wearing a
    filename.
    """
    import hashlib
    import yaml  # type: ignore

    p = agent_root / "ciris_engine" / "ciris_templates" / f"{template}.yaml"
    raw = p.read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    if got != TEMPLATE_SHA256:
        raise SystemExit(
            f"REFUSED: {p.name} is sha256:{got[:16]}…, pinned is "
            f"{TEMPLATE_SHA256[:16]}…. The identity text every arm holds constant "
            f"is not the text the regime pinned. Re-pin deliberately or fix the "
            f"checkout; do not build against an unpinned template."
        )
    d = yaml.safe_load(raw)
    missing = [f for f in TEMPLATE_FIELDS if not d.get(f)]
    if missing:
        raise SystemExit(f"REFUSED: template lacks {missing}.")
    return {f: d[f] for f in TEMPLATE_FIELDS}


def build(
    agent_root: Path, arm: str, accord: Optional[Path], blank: bool,
    framing: Optional[Path], template: str, locales: str,
) -> Dict:
    m = baseline(agent_root, locales)
    m["experiment_id"] = f"TORQUE-1-{arm}"
    m["condition"] = "c"

    text = "" if blank else (accord.read_text(encoding="utf-8") if accord else None)
    if text is None:
        raise SystemExit("--accord or --blank-axiotic required")

    for k in ACCORD_KEYS:
        if k not in m["overrides"]["corpus"]:
            raise SystemExit(
                f"REFUSED: {k!r} absent from the baseline key space. The accord "
                f"forms changed; re-derive ACCORD_KEYS rather than silently "
                f"writing fewer than three."
            )
        m["overrides"]["corpus"][k] = text

    # The PDMA framing varies WITH the arm. Blank means blank here too — an arm
    # with no accord but a CIRIS principle roster is not a blank arm.
    if FRAMING_KEY not in m["overrides"]["corpus"]:
        raise SystemExit(f"REFUSED: {FRAMING_KEY!r} absent from the baseline key space.")
    if blank:
        m["overrides"]["corpus"][FRAMING_KEY] = ""
    elif framing is None:
        raise SystemExit(
            "REFUSED: --framing required. Without it the PDMA states the six "
            "Foundational Principles in CIRIS terms in every arm, and the "
            "manipulation fails silently at the loudest point in the pipeline."
        )
    else:
        m["overrides"]["corpus"][FRAMING_KEY] = framing.read_text(encoding="utf-8")

    # Identity: same bytes in every arm, from the pinned template.
    ident = template_identity(agent_root, template)
    for f, v in ident.items():
        if f not in m["overrides"]["template"]:
            raise SystemExit(f"REFUSED: template field {f!r} absent from the key space.")
        m["overrides"]["template"][f] = v

    # Any remaining REPLACE:: sentinel is a key the baseline deliberately refused
    # to fill. Leaving one is how an arm silently reuses CIRIS values.
    unfilled = [
        f"{ns}.{k}" for ns, blk in m["overrides"].items()
        for k, v in blk.items() if isinstance(v, str) and v.startswith("REPLACE::")
    ]
    if unfilled:
        raise SystemExit(
            f"REFUSED: {len(unfilled)} key(s) still carry REPLACE:: markers "
            f"({', '.join(unfilled[:4])}…). Each is a value-bearing key the "
            f"baseline refused to guess. Fill them or the arm is not measuring "
            f"what it claims."
        )

    m["_torque_domain_limit"] = DOMAIN_LIMIT
    return m


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent-root", type=Path, default=Path("/tmp/a2911"))
    ap.add_argument("--arm", required=True)
    ap.add_argument("--accord", type=Path, help="verified corpus for all three accord forms")
    ap.add_argument("--blank-axiotic", action="store_true", help="h3ere-blank: empty the accord")
    ap.add_argument("--framing", type=Path, help="verified PDMA framing corpus for this arm")
    ap.add_argument("--template", default="default", help="pinned template name (sha-checked)")
    ap.add_argument("--locales", default="en")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    m = build(args.agent_root, args.arm, args.accord, args.blank_axiotic,
              args.framing, args.template, args.locales)

    # The domain-limit note is ours, not the schema's. Strip before validating,
    # then write the clean manifest — an extra key would be rejected by
    # extra="forbid", which is the same defect `_baseline_note` had.
    limit = m.pop("_torque_domain_limit")
    # The validator runs with cwd=agent_root, so a path relative to OUR cwd
    # resolves to nothing over there and reports 'not a readable file' —
    # which reads like a bad manifest rather than a bad path.
    args.out = args.out.resolve()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")

    rc, out = run(
        [sys.executable, "-m", "ciris_engine.logic.utils.research_overrides", "validate", str(args.out)],
        args.agent_root,
    )
    print(out.strip()[-900:])
    if rc != 0:
        print(f"\nREFUSED: manifest does not validate. Not writing a digest.", file=sys.stderr)
        return 1

    rc, dg = run(
        [sys.executable, "-m", "ciris_engine.logic.utils.research_overrides",
         "manifest-digest", str(args.out)], args.agent_root,
    )
    # The agent imports a deprecated SDK that prints a multi-line warning to
    # stderr, so "last line of output" is the warning, not the digest. Match the
    # digest by SHAPE instead — a manifest identity read off a stray warning line
    # is worse than none.
    import re as _re
    cand = [l.strip() for l in dg.splitlines() if _re.search(r"\b[0-9a-f]{64}\b", l)]
    digest = (_re.search(r"\b[0-9a-f]{64}\b", cand[-1]).group(0)
              if rc == 0 and cand else "UNAVAILABLE")
    print(f"\narm            : {args.arm}")
    print(f"manifest       : {args.out}")
    print(f"manifest_digest: {digest}")
    print(f"accord forms   : {len(ACCORD_KEYS)} keys set to the same text")
    print(f"\nDOMAIN LIMIT   : {limit}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
