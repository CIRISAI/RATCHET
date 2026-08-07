# Alt Accord — complete and verified

```
partition_digest: sha256:4135fffff971ee58e48d2e198306b340a6ecdd494eb6f5e125486f98c5e0e75e
  lines: 1153   SWAP: 49   HOLD: 1104

VERIFIED: 1104 non-SWAP lines byte-identical + 49 SWAP lines replaced
          = 1153 of 1153. SWAP text matched against the swaps file.

detect_residue: 1 candidate (line 198), adjudicated HOLD in ADJUDICATIONS.tsv
```

**The intervention is 49 lines of 1,153 — 4.2% of the document**, plus 33 lines
whose principle names were substituted mechanically. Everything else is
byte-identical to the CIRIS original.

## What closed each defect class

| class | closed by |
|---|---|
| four rosters naming different principles | **mechanical substitution** — three rounds of line-level adjudication had failed at it |
| M-1 stated three incompatible ways | a **frozen reference statement** every author had to match; 8 full statements now agree |
| CIRIS content under alt labels (slots 5, 6) | **per-slot coverage check** — no vocabulary search can find a slot whose retired name is already gone |
| defects recurring after each repair | **enumeration instead of reaction** — audits sample, searches enumerate |

## What it cost to learn

Five coherence audits. The first three each named 2–7 defective lines, each
repair fixed exactly those, and the next audit found their siblings. The third
audit diagnosed it in one sentence — *"the two lines that were repaired were the
two the audit named; their siblings were not searched for"* — and that sentence
is worth more than the corpus.

## The properties that held throughout

- **No drift, ever.** Every rebuild verified byte-identity on non-SWAP lines.
  Every defect found was a *declaration* defect — a line that should have been
  declared SWAP and was not — which is reviewable in a partition. Not once did
  an author silently alter held text.
- **Every refusal was correct.** `assemble` refused on missing swaps, `freeze`
  refused on unreviewed rows, `verify` refused on a dropped replacement, and the
  banned-stem check refused a merge. No refusal was ever worked around by
  editing until it passed.

## Known and declared, not fixed

- **Line 198** stays flagged and adjudicated HOLD rather than silenced by
  narrowing a pattern.
- **Slots 2 and 4 are identity mappings** — both value systems say "Avoid Harm"
  and "Be Honest" for the same commitment. `values_effect` on those slots is
  small **by construction** and must not be read as the manipulation failing.
- **Monolingual only.** All three accord forms carry English in every arm; the
  shipped agent runs two of them polyglot. See POLYGLOT_PROBLEM.md.
