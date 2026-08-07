# κ study — `prompts.language_guidance`, en, 30 parts, 2026-08-07

Agent `v2.9.11-stable` (7e71d0381). Two annotators applied the twelve-class
taxonomy from its **operational definitions only**, independently, without
sight of each other's work or of the shipped labels.

## Result

| comparison | κ | agreement | verdict |
|---|---|---|---|
| **A vs B — reliability, the #976 gate** | **0.831** | 26/30 | **PASS** |
| A vs shipped — validity | 0.528 | 18/30 | FAIL |
| B vs shipped — validity | 0.558 | 19/30 | FAIL |

Every gated boundary passes on reliability, including `axiotic|procedural`
(κ=1.0, 4/4).

**The taxonomy is reliable. Its application to `language_guidance` is not
validated.** Those are different claims and only the first clears #976.

## The finding that touches the campaign

`11_routing_doctrine` ships as **axiotic** — a `vary` class, and a member of
TORQUE's independent variable. **Both annotators independently classified it
`procedural`** (`hold`).

Two readers agreeing against the shipped label is not one reader's idiosyncrasy.
Until adjudicated it should come **out** of the vary set, which would take the
axiotic `language_guidance` parts from 2 to 1 (`09_trusted_person_first_step`
alone) and the axiotic surface from 52 blocks to 51.

## Where the annotators disagreed with each other (4 of 30)

| part | A | B |
|---|---|---|
| `01_preamble` | structural | axiomatic |
| `18_ratification_scope` | axiomatic | procedural |
| `21_negative_is_also_a_verdict` | axiomatic | deontic |
| `26_cross_cluster_pattern` | procedural | deontic |

None sits on an `axiotic` boundary, so none affects the vary set. All four are
about how far a framing clause counts as premise, sequence, or prohibition.

## Contamination, disclosed

Mid-study I committed a message naming part 11's shipped label. It reached
annotator A through a file-modification notification. **This was my error — I
published the answer key while the study was live.**

A disclosed it unprompted, and it arrived **after** A had submitted all 30
labels, so no label was influenced; A declined to revise, correctly, since
revising afterwards is precisely the corruption the design guards against.

Effect on the result: none on **reliability** (A and B never saw each other).
On **validity** for part 11, A's independence is documented rather than
assumed — and the leak told A its answer *disagreed*, which if anything created
pressure to revise toward the shipped label. A didn't. B, wholly uncontaminated,
reached the same `procedural` call independently.

Recorded here rather than quietly excluded, because an undisclosed contamination
is the failure this whole study exists to prevent, and it does not get an
exemption for being mine.

## Reproduce

```bash
python3 ../kappa.py annotator_a.tsv annotator_b.tsv --shipped <compose-dump.jsonl>
```
