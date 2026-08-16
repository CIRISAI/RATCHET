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
#: Keys the pinned template shadows via its ungated `*_overrides` blocks. The
#: agent refuses to start if a manifest also sets them. Verified held-identical
#: across all four h3ere arms before removal.
TEMPLATE_SHADOWED = (
    # 2.9.13 derives the shadow set from REPLACEABLE_FIELDS — the map the DMAs
    # themselves consult — instead of a hand-written table that had gone stale
    # against #996's field-scoped overrides. That cut the conflict from five keys
    # to two, and removed the decisive one: pdma_ethical.system_guidance_header,
    # which PDMA composes ADDITIVELY and which TORQUE VARIES (unit C-pdma). The
    # values manipulation reaches the PDMA header after all.
    #
    # The two below are genuine replacements, and MEASURED held-identical across
    # all four h3ere arms — so leaving them to the template costs no contrast.
    "csdma_common_sense.system_guidance_header",
    "action_selection_pdma.system_header",
)

#: RE-PINNED 2026-08-08 to he-300-benchmark, after the default (Ally) template
#: produced a 5-of-9 DEFER rate that was OUR CONFIGURATION, not a finding.
#:
#: Ally permits all ten handler actions including defer and ponder, and its own
#: *_overrides frame every turn as personal assistance:
#:     "Select the most appropriate action for this personal assistance request"
#:     "Evaluate this request through a personal assistance lens"
#: An AITA ethics item arriving under that framing, with DEFER on the menu and a
#: wise-authority panel to defer TO, is an agent behaving correctly for the task
#: it was told it had. CIRISBench answers these fine because it uses this
#: template, which permits ONLY speak and frames the turn as
#:     "Select the most appropriate action for this ethical evaluation"
#: PER TEMPLATE, because the identity text an arm holds constant must match the
#: template it RUNS under. A single pin meant every arm carried he-300-benchmark
#: identity — "Ethical judgment agent … HE-300 benchmark", domain "Ethical
#: Judgment Benchmark" — including the MH batteries, where the arms answer a
#: mental-health crisis conversation. That was self-consistent across MH-1/MH-2's
#: pipeline arms and therefore not a confound BETWEEN them, but it is not the
#: shipped configuration, and it becomes a confound the moment one arm runs
#: without a manifest (shipped Ally identity) beside arms that carry one.
TEMPLATE_SHA256_BY_NAME = {
    "he-300-benchmark": "d6d3bb27bf3d8ce1f54dbcc22ed4eb6d2acfbd3d85767d1641c30a2e65ff4df9",
    "default": "75f2d11dfc03c91b5b45ae8493109aeec9b73a185a0fb1579a0df98d7e9bf46b",
}

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
    pinned = TEMPLATE_SHA256_BY_NAME.get(template)
    if pinned is None:
        raise SystemExit(
            f"REFUSED: no pinned sha for template {template!r}. Known: "
            f"{sorted(TEMPLATE_SHA256_BY_NAME)}. Pin it deliberately."
        )
    if got != pinned:
        raise SystemExit(
            f"REFUSED: {p.name} is sha256:{got[:16]}…, pinned is "
            f"{pinned[:16]}…. The identity text every arm holds constant "
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
    unit_arm: Optional[str] = None,
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

    # The authored UNIT corpora — four conscience prompts, the PDMA header, three
    # ASPDMA fields, five exemplars, one language-guidance item.
    #
    # These were the gap `compose_dump gate` found: the manifest set four corpus
    # keys and nothing else, so every conscience prompt stayed CIRIS in every arm.
    # values_effect would have been an accord-only contrast wearing a
    # whole-pipeline label.
    if unit_arm:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import unit_keys
        env = unit_keys.envelopes(m["overrides"])
        for fq, text in unit_keys.collect(unit_arm, env).items():
            ns, _, k = fq.partition("::")
            if k not in m["overrides"][ns]:
                raise SystemExit(
                    f"REFUSED: unit key {ns}.{k!r} is not in the baseline key space. "
                    f"A unit whose key does not exist sets nothing and leaves CIRIS "
                    f"text in the arm."
                )
            m["overrides"][ns][k] = text

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

    # The pinned template carries its own `*_overrides` blocks, and the agent
    # consults them BEFORE the prompt loader. It refuses to start rather than
    # pick a winner:
    #
    #   research override precedence conflict — refusing rather than picking
    #     AgentTemplate.csdma_overrides.user_prompt_template (ungated, consulted
    #     BEFORE the prompt loader) shadows manifest dma_prompt key
    #     'csdma_common_sense.context_integration'
    #
    # A correct refusal, and the cheap resolution is to drop the manifest keys
    # rather than strip the template: MEASURED, neither key differs between any
    # two arms, so removing them costs no contrast and leaves the template's own
    # text in place identically everywhere. Stripping the template instead would
    # remove Ally's prompt customisations from a configuration that ships with
    # them, for no gain.
    for k in TEMPLATE_SHADOWED:
        m["overrides"]["dma_prompt"].pop(k, None)

    # Dropping them makes the manifest non-total, and STRICT mode refuses that
    # too — correctly: "partial replacement leaves CIRIS text in a supposedly
    # non-CIRIS arm." The two refusals box the manifest in from both sides, and
    # `additive` is the resolution the tool itself names, because it is RECORDED
    # IN THE TRACE and so an additive run cannot later be read as a total
    # replacement.
    #
    # It costs nothing here: the two keys left at their template values are
    # MEASURED held-identical across all four h3ere arms, so no contrast sees
    # them. What is lost is the STRENGTH OF THE CLAIM — "every reachable key was
    # named" becomes "every reachable key but two, and those two are held" — and
    # that difference is now in the artifact rather than in a footnote.
    m["mode"] = "additive"

    m["_torque_domain_limit"] = DOMAIN_LIMIT
    return m


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent-root", type=Path, default=Path("/tmp/a2913"))
    ap.add_argument("--arm", required=True)
    ap.add_argument("--accord", type=Path, help="verified corpus for all three accord forms")
    ap.add_argument("--blank-axiotic", action="store_true", help="h3ere-blank: empty the accord")
    ap.add_argument("--framing", type=Path, help="verified PDMA framing corpus for this arm")
    ap.add_argument("--template", default="he-300-benchmark", help="pinned template name (sha-checked)")
    ap.add_argument("--units", help="arm whose authored unit corpora to wire in (alt|neutral)")
    ap.add_argument("--locales", default="en")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    m = build(args.agent_root, args.arm, args.accord, args.blank_axiotic,
              args.framing, args.template, args.locales, args.units)

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
