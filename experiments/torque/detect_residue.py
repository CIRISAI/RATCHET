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
    # Narrowed after audit 5: "bias" as a DETECTION PROCEDURE (bias audits,
    # bias assessments, "detecting hidden bias") is shared operational content,
    # not a slot-6 value claim. Five of eight hits were procedures. Match the
    # distributive CLAIM, not the machinery for finding unfairness.
    "slot 6 CIRIS content (Justice) under alt label (Pluralism)":
        r"distribut\w*|equitab\w*|\bfair(ly|ness)?\b|unfair|inequit\w*|"
        r"embedding.{0,20}bias|exacerbating.{0,20}bias",
    "CIRIS principle vocabulary surviving substitution":
        # "Public Transparency" is a NAMED MECHANISM (a >100k-MAU publication rule),
        # not the principle — both arms share it. Dropped after audit 5.
        r"Non-Maleficence|\bIntegrity\b|Beneficence|\bJustice\b",
    "M-1 content phrase ('universal sentient flourishing')":
        r"universal sentient flourishing",
    # ADDED after audit 4: 167 gave M-1 as a GOOD TO ADVANCE rather than a
    # weighing to perform, and matched none of the four patterns above. The
    # search missed a phrasing, which is the cheapest kind of miss to fix.
    "M-1 as telos rather than procedure":
        r"advance flourishing|flourishing under M-1|fulfil M-1|achieve M-1|toward M-1",
}

#: Per-slot coverage. Audit 4 found slot 5 renamed everywhere and re-authored
#: nowhere — every line saying what Epistemic Autonomy MEANS still carried
#: CIRIS's Respect-for-Autonomy content. That is the camouflage defect at the
#: granularity of a whole slot, and no vocabulary search finds it: the retired
#: NAME is gone, so nothing matches. Only asking "has this slot's meaning been
#: authored anywhere?" finds it.
#: Slots 2 and 4 are IDENTITY mappings — both value systems use the same words
#: for the same commitment ("Avoid Harm", "Be Honest"), so they legitimately need
#: no authoring and flagging them would be a false positive. That two of six
#: slots coincide is a finding in its own right: values_effect on them is small
#: by construction and must not be read as the manipulation failing.
SHARED_SLOTS = ("Harm Avoidance", "Honesty")
SLOTS = ("Helpfulness", "Harm Avoidance", "Ethics", "Honesty",
         "Epistemic Autonomy", "Pluralism")

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

    print(f"\n{'='*66}\nPER-SLOT AUTHORING COVERAGE")
    uncovered = []
    for slot in SLOTS:
        hits = [i for i, l in enumerate(lines, 1) if slot in l]
        auth = [i for i in hits if i in swap]
        # A slot whose only authored lines are incidental mentions is still
        # unauthored in the sense that matters.
        if slot in SHARED_SLOTS:
            flag = "  (shared vocabulary — no authoring expected)"
        elif auth:
            flag = ""
        else:
            flag = "  *** RENAMED BUT NEVER RE-AUTHORED ***"
            uncovered.append(slot)
        print(f"  {slot:22s} {len(hits):>3} lines, {len(auth):>2} authored{flag}")
    if uncovered:
        print(f"\n  {len(uncovered)} slot(s) carry an alt name over entirely CIRIS substance.")

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
