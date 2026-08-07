# Names are substituted mechanically. Only meanings are authored.

## What the coherence audits established

The alt Accord kept coming out incoherent, and each repair revealed more of the
same problem:

| | found |
|---|---|
| principle names | **37 lines, in 16 clusters** across 900 lines |
| `M-1` mentions | **10 lines**, 8 of them HOLD |
| rosters enumerating all six | **4** (Ch1, Ch2, Book VI, sunset) |
| M-1 stated with content | **3 places**, two of them authored and disagreeing |

Line-level adjudication had swapped ~12 of the 37 and 2 of the 10. The rest kept
CIRIS names inside the alt arm, and held prose kept referring to principles that
swapped lines had renamed.

**This is not a failure of the partition method. It is a failure to notice that
value NAMES are pervasive while value MEANINGS are localized.** A name is a
reference; it occurs wherever the document points at the value system. Asking 16
isolated authors to rename the same thing consistently is asking for a
coordination they were deliberately denied.

## The split

> **Names are substituted mechanically and globally. Only definitional content
> is authored.**

| | how | count | who does it |
|---|---|---|---|
| **NAMES** — every occurrence of a principle label or `M-1` | global find-and-replace from the frozen name table | ~47 lines | nobody — it is a script |
| **MEANINGS** — the lines that say what a principle IS | authored in isolation, per line | ~12 lines | isolated authors |

Consistency of names becomes **guaranteed by construction** rather than achieved
by coordination. No author can diverge on a name because no author touches one.

That is the same move the partition itself made: take the thing that keeps going
wrong and make it structurally impossible rather than checked-for.

## Why this was not obvious earlier

The partition asks one question per line — *does this line state a value?* — and
that question has no good answer for a line that merely *names* one in passing.
Line 355 (`"bypassing transparency and deferral converts routine design shortcuts
into systemic tragedy"`) is a historical illustration that happens to use a
principle name. Adjudicators correctly called it HOLD. It then sat in the alt arm
naming a CIRIS principle.

Both calls were right. The question was wrong: a line can need its **name**
updated without its **claim** changing.

## Ordering, which matters

1. Freeze the name table (`ALT_NAME_TABLE.md`).
2. **Substitute names globally**, producing an intermediate corpus. Assert: every
   occurrence replaced, no CIRIS principle name survives, no line count change.
3. Partition the *substituted* corpus for MEANING lines only.
4. Author those in isolation, with names already fixed and visible.
5. Assemble, verify byte-identity, re-audit coherence.

Step 2 before step 3 is essential. Partitioning first is what produced sixteen
clusters of half-renamed text.

## What this does not fix

A definitional line still has to be authored, and two definitional lines can
still disagree about *substance* even when their names match. The Chapter 1
roster and the Chapter 2 directives will still need to be authored as a
coordinated set, because "what Helpfulness means" must be the same claim in both.

Term substitution removes the naming failure. It does not remove the need for the
name table to be reviewed, or for the coherence audit to run afterwards.

## Status of the artifact right now

Reverted to the last consistent state and re-verified:

```
partition_digest: sha256:10327bc7c982850fbda613bc4702f09f8db435f212ca2578a4c4648f44ec5406
  lines: 1153   SWAP: 32   HOLD: 1121
VERIFIED: 1121 non-SWAP byte-identical + 32 SWAP replaced = 1153 of 1153.
```

The ten promotions from the failed repair were rolled back rather than left
half-applied. For a period the partition declared 42 SWAP while only 32 were
authored — `assemble` refused, so no bad artifact was ever produced, but
`FROZEN.md` pinned a digest the working file no longer computed. **A torn frozen
state is worse than an un-repaired one**, because the digest is the campaign's
claim about what it varied.

---

# The cost, measured: substitution can CAMOUFLAGE a defect

The final audit resolved one fatal completely and left two — and its phrasing on
the second is the important result:

> *"The term table's mapping made 59 and 625 look updated while leaving CIRIS
> content in place. The alt document still defines its own meta-goal twice,
> incompatibly — the same finding as the previous audit, now camouflaged."*

**Mechanical substitution moves every name and no claim.** So a line whose
*content* is specific to the principle it names comes out wearing the alt label
over CIRIS substance. Before substitution that line was obviously wrong. After,
it looks right.

Line 111 is the clearest case — self-contradictory inside one line:

```
* Pluralism: Resist Illegitimate Power—distribute benefits and burdens equitably.
```

The label says resist illegitimate power; the gloss says distribute equitably.
Slot 6 is the worst affected because Justice → Pluralism is the one slot where
the two value systems genuinely disagree, so all of its CIRIS content is now
mislabelled rather than merely present.

M-1 is stated three ways and they are not the same claim: line 59 and 625 make it
a **world-state to promote**; the authored line 114 makes it a **decision
procedure over four priorities**.

## What this changes about the method

Substitution is still correct and still the right first step — it resolved the
four-roster problem cleanly and completely, which line-level adjudication had
failed at three times. But it needs a companion rule:

> **A line whose content is specific to a renamed principle must be SWAP, not
> merely substituted.** Renaming it without re-authoring it converts a visible
> defect into an invisible one.

That is detectable mechanically: for each renamed slot, find every line
containing the *old* principle's characteristic vocabulary and require it be
SWAP. "Equitable distribution" and "algorithmic bias" belong to Justice; they
cannot sit under Pluralism unchanged.

## The prescription that was written and not carried out

`ALT_NAME_TABLE.md` already listed 625, 755, 226, 232–233 and 248–249 as
promote-to-SWAP. That repair was staged, refused on banned stems, and **rolled
back**. When the meaning partition was rebuilt it took the 14 lines from the
prior adjudication and **dropped the prescriptions** — so a fix that had already
been diagnosed and written down was lost in a rollback.

That is a process failure, not an analytical one, and worth naming: a rollback
restored a consistent state and silently discarded work that was not part of the
inconsistency.
