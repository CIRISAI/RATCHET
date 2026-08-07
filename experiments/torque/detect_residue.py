#!/usr/bin/env python3
"""Find EVERY line carrying retired value content — not the ones an audit named.

Three coherence audits each named 2-7 defective lines; each repair fixed exactly
those and the next audit found their siblings. The third audit diagnosed it:

    "Every blocking defect is the SAME class as the previous audit's. The method
     is not converging on class, only on count... The two lines that were
     repaired were the two the audit named. Their siblings were not searched for."

That is whack-a-mole against a list. An audit SAMPLES; a search ENUMERATES. This
enumerates.

For each retired concept, it reports every line still carrying that concept's
characteristic vocabulary and whether the line is declared SWAP. An undeclared
hit is a candidate — the alt corpus wearing an alt label over CIRIS substance,
which is the camouflage defect term substitution creates.

It OVER-CATCHES deliberately. `Integrity` and `fair` occur in ordinary prose that
both arms legitimately share, so hits need adjudication. A candidate list that
must be triaged is strictly better than an audit's sample: the sample's misses
are invisible, and this list's false positives are not.

Usage:  detect_residue.py <assembled.txt> <partition.tsv>
Exit 1 if any undeclared candidate exists.
"""
from __future__ import annotations
import re, sys
from pathlib import Path

#: retired concept -> vocabulary that belongs to it
CLASSES = {
    "M-1 as world-state (retired: 'sustainable adaptive coherence')":
        r"sustainable order|structure to carry life|wildness|Anti-Entropic|adaptive capacity",
    "slot 6 CIRIS content (Justice) under alt label (Pluralism)":
        r"distribut\w*|equitab\w*|\bfair(ly|ness)?\b|unfair|inequit\w*|\bbias\w*",
    "CIRIS principle vocabulary surviving substitution":
        r"Non-Maleficence|Public Transparency|\bIntegrity\b|Beneficence|\bJustice\b",
    "M-1 content phrase ('universal sentient flourishing')":
        r"universal sentient flourishing",
}

def main() -> int:
    text, part = Path(sys.argv[1]), Path(sys.argv[2])
    lines = text.read_text(encoding="utf-8").splitlines()
    swap = {int(l.split("\t")[0]) for l in part.read_text(encoding="utf-8").splitlines()
            if l.strip() and l.split("\t")[1] == "SWAP"}

    union: set[int] = set()
    for name, pat in CLASSES.items():
        hits = {i for i, l in enumerate(lines, 1) if re.search(pat, l, re.I)}
        und = sorted(hits - swap)
        union |= set(und)
        print(f"\n{name}")
        print(f"  {len(hits)} lines match, {len(hits & swap)} declared SWAP, "
              f"{len(und)} UNDECLARED")
        for n in und:
            print(f"    {n:>5}  {lines[n-1][:96]}")

    print(f"\n{'='*66}\nUNDECLARED CANDIDATES (union): {len(union)}")
    print(sorted(union))
    if union:
        print("\nEach needs adjudication: is this line's use of the vocabulary a\n"
              "PRINCIPLE-SLOT use (promote to SWAP) or ORDINARY PROSE both arms\n"
              "legitimately share (leave HOLD)? Do not guess from the word alone.")
        return 1
    print("\nNo undeclared residue.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
