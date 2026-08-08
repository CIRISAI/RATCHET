#!/usr/bin/env python3
"""Split the authored unit corpora into the override keys they actually set.

WHY THIS EXISTS. The arm manifests set four `corpus` keys — three accord forms
and the PDMA framing. Everything else the campaign authored (the four conscience
prompts, the PDMA header, three ASPDMA fields, four exemplars, one language-
guidance item) lives in the `dma_prompt`, `conscience_prompt` and `string`
namespaces and was **not wired into any manifest**.

`compose_dump gate` caught it, in the words that make it unmissable:

    FAIL [2] en:aspdma.action_selection_pdma.csdma_ambiguity_guidance:
      axiotic is varied by the regime but the block is byte-identical across
      arms (sha256 e3d6cba10755…) — the ablation did not reach it

Had the campaign run without this, only the accord and framing would have
varied. The four conscience faculties — the treatment itself, the thing TORQUE
exists to measure — would have been reasoning under CIRIS values **in the alt
arm**, and `values_effect` would have been an accord-only contrast wearing a
whole-pipeline label.

THE FILE FORMAT. Each unit file packs one or more override values, separated by
`### FIELD: <name>` (a dotted-prefix field within a prompt file) or
`### KEY: <name>` (a standalone localized string). The markers are structural and
are NOT part of any value.

Usage
-----
    unit_keys.py --arm alt         # print the key -> bytes mapping
    unit_keys.py --arm alt --json  # emit {key: text} for the manifest builder
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Tuple

HERE = Path(__file__).resolve().parent

#: unit file -> (namespace, dotted prefix for FIELD markers)
#: KEY markers are absolute within their namespace and ignore the prefix.
UNITS: Dict[str, Tuple[str, str]] = {
    "B-optveto":    ("conscience_prompt", "optimization_veto_conscience"),
    "B-epihum":     ("conscience_prompt", "epistemic_humility_conscience"),
    "B-coherence":  ("conscience_prompt", "coherence_conscience"),
    "C-pdma":       ("dma_prompt",        "pdma_ethical"),
    "D-aspdma":     ("dma_prompt",        "action_selection_pdma"),
    # `string` keys carry a `prompts.` prefix in the override key space that the
    # unit files' KEY markers do not. Verified against the baseline rather than
    # assumed — the first mapping without it produced six keys that exist nowhere,
    # which `build_arm_manifest` would have to refuse on rather than silently drop.
    "E-exemplars":  ("string",            "prompts.language_guidance"),
    "F-lg-axiotic": ("string",            "prompts.language_guidance"),
}

MARKER = re.compile(r"^### (FIELD|KEY): (.+)$")


def split_unit(path: Path, namespace: str, prefix: str) -> Dict[str, str]:
    """Return {fully-qualified key: value}. Markers are dropped, not emitted."""
    out: Dict[str, str] = {}
    cur: str | None = None
    buf: list[str] = []

    def flush() -> None:
        if cur is not None:
            # Trailing newline is part of the shipped value; the marker line is not.
            out[cur] = "\n".join(buf).rstrip("\n") + "\n"

    for line in path.read_text(encoding="utf-8").splitlines():
        m = MARKER.match(line)
        if m:
            flush()
            kind, name = m.group(1), m.group(2).strip()
            cur = f"{prefix}.{name}" if kind == "FIELD" else f"{prefix}.{name}"
            buf = []
        else:
            buf.append(line)
    flush()

    if not out:
        raise SystemExit(
            f"REFUSED: {path.name} has no ### FIELD:/### KEY: marker. A unit with "
            f"no marker sets no key, and would be silently absent from every arm."
        )
    return {f"{namespace}::{k}": v for k, v in out.items()}


def envelopes(baseline: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    """Per-key TRAILING-NEWLINE envelope, taken from the shipped values.

    The shipped values are not uniform: three ASPDMA fields end with no newline
    at all, the five exemplars end with two. A splitter that normalises to one
    changes bytes in a HELD block, and `compose_dump gate` reports that as

        FAIL [3] … held procedural block differs across arms

    which is true, and caused entirely by the tool rather than by the campaign.
    So the envelope is READ from the shipped value and reapplied to both arms —
    identical envelope, varied content, no spurious diff.
    """
    env: Dict[str, str] = {}
    for unit, (ns, prefix) in UNITS.items():
        for fq in split_unit(HERE / "partition" / "src" / f"{unit}.txt", ns, prefix):
            k = fq.partition("::")[2]
            v = baseline[ns].get(k)
            if v is None:
                continue
            env[fq] = v[len(v.rstrip("\n")):]
    return env


def collect(arm: str, envelope: Dict[str, str] | None = None) -> Dict[str, str]:
    """`arm` is 'ciris' (the monoglot originals) or an alt/neutral corpus name.

    The CIRIS arm takes its unit text from `partition/src/` — the SAME monoglot
    sources the varied arms were built from. That is the monoglot decision applied
    to units, and it is load-bearing: the shipped
    `pdma_ethical.system_guidance_header` is 21,928 B of POLYGLOT text, of which
    the C-pdma unit is an 8.2 KB English fragment. Overriding only the varied arm
    would delete 13.7 KB of non-English content from one side of the comparison
    and call the difference "values".
    """
    if arm == "ciris":
        base = HERE / "partition" / "src"
        names = [f"{{}}.txt"]
    else:
        base = HERE / "corpora" / ("values-alt" if arm == "alt" else f"values-{arm}")
        names = ["{}.txt", "{}-mechanical.txt"]

    keys: Dict[str, str] = {}
    for unit, (ns, prefix) in UNITS.items():
        for pat in names:
            cand = base / pat.format(unit)
            if cand.exists():
                keys.update(split_unit(cand, ns, prefix))
                break
        else:
            raise SystemExit(
                f"REFUSED: no corpus for unit {unit!r} in {base}. An arm missing a "
                f"unit runs CIRIS text there while claiming it varied."
            )

    if envelope:
        for fq in keys:
            if fq in envelope:
                keys[fq] = keys[fq].rstrip("\n") + envelope[fq]
    return keys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, help="alt | neutral")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    keys = collect(args.arm)
    if args.json:
        json.dump(keys, sys.stdout, ensure_ascii=False)
        return 0
    print(f"{len(keys)} override keys from {len(UNITS)} unit files:\n")
    for k, v in sorted(keys.items()):
        ns, _, name = k.partition("::")
        print(f"  {ns:19s} {name:56s} {len(v):>7} B")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
