#!/usr/bin/env python3
"""Global mechanical substitution of value NAMES. Nobody authors a name.

Value names are pervasive; value meanings are localized. The accord names a
Foundational Principle on 37 lines across 16 clusters and mentions `M-1` on 10.
Line-level adjudication reached ~12 and 2 of those, because the partition's
question — "does this line STATE a value?" — has no good answer for a line that
merely NAMES one in passing. Adjudicators correctly held those lines, and they
then sat in the alt arm carrying CIRIS principle names.

So names are substituted here, mechanically and globally, from a frozen table.
Consistency is guaranteed by construction: no author touches a name, so no author
can diverge on one. Only definitional lines — what a principle MEANS — are
authored, and by then the names are already fixed.

Run this BEFORE partitioning. Partitioning first is what produced sixteen
clusters of half-renamed text.

Usage
-----
    substitute_terms.py --in <ciris_en.txt> --table terms.tsv --out <substituted.txt>
    substitute_terms.py --in <f> --table terms.tsv --audit      # report only
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

#: Terms that must NOT survive substitution — the check that this actually
#: worked. `Integrity`/`Transparency` appear both as principle names and as
#: ordinary English, so they are audited rather than banned outright; see
#: --audit output and resolve by hand before freezing.
MUST_NOT_SURVIVE = (
    "Beneficence", "Non-maleficence", "Do Good",
    "Act Ethically", "Ensure Fairness",
    "adaptive coherence", "Adaptive Coherence",
)

#: SHARED vocabulary, not CIRIS-specific. Both value systems use these exact
#: words for the same commitment, so they survive substitution legitimately —
#: "Avoid Harm" maps to "Avoid Harm (Harm Avoidance)" and "Be Honest" to
#: "Be Honest (Honesty)".
#:
#: This is a finding, not a technicality: the congruence measurement already
#: showed the two corpora agree closely on honesty and harm and diverge on
#: dignity, fairness and obligation. Two of six principle slots carry the SAME
#: name in both arms, so values_effect on those slots is small BY CONSTRUCTION
#: and must not be read as the manipulation failing there.
SHARED = ("Avoid Harm", "Be Honest")

#: Appears as a principle name AND as ordinary English. Substituting blindly
#: would corrupt prose; leaving it alone leaves a CIRIS name in the alt arm.
#: Reported for manual resolution rather than guessed at.
AMBIGUOUS = ("Integrity", "Transparency", "Autonomy", "Justice", "Fidelity")


def load_table(p: Path) -> List[Tuple[str, str]]:
    """`from<TAB>to`, longest-first so `Do Good (Beneficence)` is replaced before
    `Beneficence` — otherwise the inner match fires and leaves a mangled label."""
    rows = []
    for raw in p.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.startswith("#"):
            continue
        frm, _, to = raw.partition("\t")
        rows.append((frm, to))
    return sorted(rows, key=lambda r: -len(r[0]))


def substitute(text: str, table: List[Tuple[str, str]]) -> Tuple[str, Dict[str, int]]:
    counts: Dict[str, int] = {}
    for frm, to in table:
        n = text.count(frm)
        if n:
            counts[frm] = n
            text = text.replace(frm, to)
    return text, counts


def audit(text: str) -> Dict[str, List[int]]:
    """Where does each ambiguous or must-not-survive term still appear?"""
    out: Dict[str, List[int]] = {}
    for term in MUST_NOT_SURVIVE + AMBIGUOUS:
        hits = [i for i, l in enumerate(text.splitlines(), 1) if term in l]
        if hits:
            out[term] = hits
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", type=Path, required=True)
    ap.add_argument("--table", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--audit", action="store_true", help="report term occurrences and exit")
    args = ap.parse_args()

    text = args.src.read_text(encoding="utf-8")
    n_lines = len(text.splitlines())

    if args.audit or not args.table:
        found = audit(text)
        print(f"{args.src.name}: {n_lines} lines\n")
        for term, hits in sorted(found.items(), key=lambda kv: -len(kv[1])):
            kind = "MUST-NOT-SURVIVE" if term in MUST_NOT_SURVIVE else "ambiguous"
            head = ", ".join(str(h) for h in hits[:12])
            more = f" (+{len(hits)-12})" if len(hits) > 12 else ""
            print(f"  {term:22s} {len(hits):>3}  [{kind}]  lines {head}{more}")
        print(
            "\nAmbiguous terms appear BOTH as principle names and as ordinary English."
            "\nSubstituting blindly corrupts prose; leaving them leaves CIRIS names in"
            "\nthe alt arm. Resolve each by hand into the table before freezing — this"
            "\ntool will not guess."
        )
        return 0

    table = load_table(args.table)
    out, counts = substitute(text, table)

    if len(out.splitlines()) != n_lines:
        print(f"REFUSED: line count changed {n_lines} -> {len(out.splitlines())}. "
              f"A substitution that adds or removes a line is drift.", file=sys.stderr)
        return 1

    residual = {t: h for t, h in audit(out).items() if t in MUST_NOT_SURVIVE}
    if residual:
        print(f"REFUSED: {len(residual)} CIRIS principle name(s) survived substitution:",
              file=sys.stderr)
        for t, hits in residual.items():
            print(f"  {t!r} still on lines {hits[:8]}", file=sys.stderr)
        print("\nAn unsubstituted name means the alt arm carries CIRIS vocabulary. "
              "Extend the table; do not edit the output.", file=sys.stderr)
        return 1

    args.out.write_text(out, encoding="utf-8")
    print(f"substituted {sum(counts.values())} occurrences of {len(counts)} terms")
    for frm, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {n:>3}x  {frm}")
    print(f"\nwrote {args.out} — {n_lines} lines, unchanged")
    still = {t: h for t, h in audit(out).items() if t in AMBIGUOUS}
    if still:
        print("\nAMBIGUOUS TERMS REMAINING (each needs a human call before freezing):")
        for t, hits in still.items():
            print(f"  {t:16s} {len(hits):>3} lines: {hits[:10]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
