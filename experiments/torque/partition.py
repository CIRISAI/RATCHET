#!/usr/bin/env python3
"""Line-level SWAP/HOLD partition — makes drift structurally impossible.

Two authoring passes were refuted. The second was worse than the first: the
repair MERGED inserted content into the lines it claimed to have restored, so the
artifact stayed deontically identical to pre-repair while its diff signature got
cleaner (change-blocks 154 -> 114). That is a loop optimising against its own
checker, and a third pass would most likely have produced a better-hidden defect.

The cause is structural, not instructional. **An author asked to rewrite a
document while holding most of it will improve the adjacent text, because that is
what writing is.** Checking afterwards is a race the checker loses to a fluent
author.

So stop asking for a rewrite:

  1. Partition the original ONCE, line by line, into SWAP and HOLD. Freeze it.
  2. Author each SWAP line IN ISOLATION — the author sees one line plus alt-source
     material, never the surrounding document, so there is nothing adjacent to
     improve.
  3. BYTE-COPY every HOLD line. No author ever sees it.
  4. Assemble mechanically and assert byte-identity on every HOLD line.

Step 4 is a test, not a review, and it cannot be laundered: a merged line fails
byte-identity by construction. That is the property the two review passes lacked.

The frozen partition is also the campaign's auditable declaration of what it
varied — reviewable before a single word is authored, rather than reconstructed
from a diff afterwards.

Usage
-----
    partition.py propose  <original.txt> --out partition.tsv
    partition.py freeze   partition.tsv                 # hash-pin it
    partition.py assemble <original.txt> partition.tsv <swaps.tsv> --out alt.txt
    partition.py verify   <original.txt> partition.tsv <alt.txt>
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Heuristics only PROPOSE. Every line is reviewed before the partition is frozen;
# nothing here decides anything on its own.
AXIOTIC_HINTS = (
    "principle", "value", "priorit", "matters", "meta-goal", "m-1", "flourish",
    "coherence", "dignity", "autonomy", "beneficence", "justice", "integrity",
)
HOLD_HARD = (
    # Numbers, thresholds and named procedures are the drift surface that the
    # refutations kept landing on. Default them to HOLD and make an author argue
    # otherwise.
    "≥", ">=", "<=", "%", "days", "step ", "protocol", "wbd", "pdma", "annex",
)


def read_lines(p: Path) -> List[str]:
    return p.read_text(encoding="utf-8").splitlines()


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def propose(original: Path, out: Path) -> None:
    lines = read_lines(original)
    rows = []
    for i, line in enumerate(lines, 1):
        low = line.lower()
        if not line.strip():
            tag = "HOLD"
        elif any(h in low for h in HOLD_HARD):
            tag = "HOLD"          # thresholds/procedures: argue to move these
        elif any(h in low for h in AXIOTIC_HINTS):
            tag = "SWAP?"         # candidate only — REVIEW REQUIRED
        else:
            tag = "HOLD"
        rows.append(f"{i}\t{tag}\t{digest(line)[:12]}\t{line}")
    out.write_text("\n".join(rows) + "\n", encoding="utf-8")
    n_swap = sum(1 for r in rows if "\tSWAP?\t" in r)
    print(f"proposed {len(rows)} lines: {n_swap} SWAP? candidates, {len(rows)-n_swap} HOLD")
    print("EVERY SWAP? must be reviewed and set to SWAP or HOLD before freezing.")
    print("A partition containing SWAP? is not frozen and will be refused.")


def load_partition(p: Path) -> List[Tuple[int, str, str, str]]:
    out = []
    for raw in p.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        n, tag, dg, *rest = raw.split("\t")
        out.append((int(n), tag, dg, "\t".join(rest)))
    return out


def freeze(part: Path) -> int:
    rows = load_partition(part)
    unreviewed = [n for n, tag, _, _ in rows if tag.endswith("?")]
    if unreviewed:
        print(
            f"REFUSED: {len(unreviewed)} line(s) still marked SWAP? — "
            f"first at line {unreviewed[0]}. A partition is the campaign's "
            f"declaration of what it varied; an unreviewed line is an undeclared "
            f"variation.",
            file=sys.stderr,
        )
        return 1
    body = "\n".join(f"{n}\t{tag}" for n, tag, _, _ in rows)
    print(f"partition_digest: sha256:{digest(body)}")
    print(f"  lines: {len(rows)}  SWAP: {sum(1 for r in rows if r[1]=='SWAP')}  "
          f"HOLD: {sum(1 for r in rows if r[1]=='HOLD')}")
    return 0


def assemble(original: Path, part: Path, swaps: Path, out: Path) -> int:
    lines = read_lines(original)
    rows = load_partition(part)
    repl: Dict[int, str] = {}
    for raw in swaps.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        n, _, text = raw.partition("\t")
        repl[int(n)] = text

    result, missing = [], []
    for n, tag, _, _ in rows:
        src = lines[n - 1]
        if tag == "SWAP":
            if n not in repl:
                missing.append(n)
                result.append(src)
            else:
                result.append(repl[n])
        else:
            result.append(src)      # BYTE-COPY. No author saw this line.
    if missing:
        print(f"REFUSED: {len(missing)} SWAP line(s) have no replacement "
              f"(first: {missing[0]}). Assembling would silently ship CIRIS "
              f"values inside the alt arm.", file=sys.stderr)
        return 1
    out.write_text("\n".join(result) + "\n", encoding="utf-8")
    print(f"assembled {out} — {len(result)} lines, {len(repl)} swapped")
    return 0


def verify(original: Path, part: Path, alt: Path) -> int:
    """The assertion the two review passes could not make. Not laundrable."""
    o, a = read_lines(original), read_lines(alt)
    rows = load_partition(part)
    if len(a) != len(rows):
        print(f"REFUSED: alt has {len(a)} lines, partition declares {len(rows)}. "
              f"Insertion or deletion is drift by definition.", file=sys.stderr)
        return 1
    bad = []
    for idx, (n, tag, _, _) in enumerate(rows):
        if tag == "HOLD" and a[idx] != o[n - 1]:
            bad.append((n, o[n - 1], a[idx]))
    if bad:
        print(f"REFUSED: {len(bad)} HOLD line(s) are not byte-identical.", file=sys.stderr)
        for n, orig, got in bad[:10]:
            print(f"  line {n}\n    orig: {orig[:110]}\n    alt : {got[:110]}", file=sys.stderr)
        print("\nA HOLD line that changed is drift, whether it was rewritten or had "
              "content merged into it. Merging is why the second authoring pass "
              "passed its own checks.", file=sys.stderr)
        return 1
    print(f"VERIFIED: all {sum(1 for r in rows if r[1]=='HOLD')} HOLD lines byte-identical; "
          f"{sum(1 for r in rows if r[1]=='SWAP')} SWAP lines replaced. No drift possible.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("propose"); p1.add_argument("original", type=Path); p1.add_argument("--out", type=Path, required=True)
    p2 = sub.add_parser("freeze"); p2.add_argument("partition", type=Path)
    p3 = sub.add_parser("assemble")
    for a in ("original", "partition", "swaps"): p3.add_argument(a, type=Path)
    p3.add_argument("--out", type=Path, required=True)
    p4 = sub.add_parser("verify")
    for a in ("original", "partition", "alt"): p4.add_argument(a, type=Path)
    args = ap.parse_args()

    if args.cmd == "propose": propose(args.original, args.out); return 0
    if args.cmd == "freeze": return freeze(args.partition)
    if args.cmd == "assemble": return assemble(args.original, args.partition, args.swaps, args.out)
    return verify(args.original, args.partition, args.alt)


if __name__ == "__main__":
    sys.exit(main())
