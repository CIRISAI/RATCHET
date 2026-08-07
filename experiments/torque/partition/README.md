# Accord partition — proposed, NOT frozen

`accord.tsv` — 1,153 lines: **107 `SWAP?` candidates, 1,046 `HOLD`**.

`freeze` will refuse this file until every `SWAP?` is resolved to `SWAP` or
`HOLD`. That refusal is deliberate: an unreviewed line is an undeclared
variation, and this partition is the campaign's declaration of what it varied.

## The number is itself a finding

The heuristics select **9%** of the document as candidate value content. The
first free-authoring pass produced 8,566 body words against the original's 7,413
— **+15.6%** — and the second pass's refutation found drift in held thresholds,
inserted procedural steps, and appended clauses on lines that had been
byte-identical before the "repair."

Those two facts fit together. **The authoring passes were rewriting far more than
the axiotic surface**, which is the drift finding seen from the other side: not
"the author made some mistakes" but "the author was operating on roughly ten
times the intended surface."

## Review, and the direction the errors run

Two questions, and they are not symmetric:

1. **Is each `SWAP?` genuinely axiotic?** Apply the boundary rule: *would a
   different value corpus require this text to be different?* If the text
   survives a value swap and merely determines what runs when, it is
   `procedural` → HOLD.
2. **Did the heuristics miss anything?** A line marked HOLD that is genuinely
   axiotic leaves CIRIS values inside the alt arm.

Missing a SWAP biases `values_effect` **toward zero** — conservative, weakens a
positive result. Wrongly swapping a HOLD introduces a **confound** that points
the same direction as the treatment and is indistinguishable from it. So the
default is HOLD and question 1 gets the harder scrutiny.

## Not delegable wholesale

Two automated passes were refuted, the second having laundered its defect into
the lines it claimed to have restored. The review is the step where a human says
what the experiment varies. Individual `SWAP` lines can be authored in isolation
afterwards — that part is safely mechanical, because the author sees one line
and cannot improve its neighbours.

## Then

```bash
python3 partition.py freeze   partition/accord.tsv          # hash-pin
python3 partition.py assemble <orig> partition/accord.tsv <swaps.tsv> --out alt.txt
python3 partition.py verify   <orig> partition/accord.tsv alt.txt
```

---

# Adjudication run, 2026-08-07 — assembled, VERIFIED, but NOT FROZEN

## Outcome

`assemble` + `verify` both succeeded:

```
assembled A-accord-mechanical.txt — 1153 lines, 28 swapped
VERIFIED: 1125 non-SWAP lines byte-identical + 28 SWAP lines replaced = 1153 of 1153.
```

**107 candidates resolved to 28 actual SWAPs.** Independent duplicate review sent
74% of the heuristic's candidates to HOLD. That is the third independent estimate
of the same thing, and they agree: the free-authoring passes were operating on
roughly ten times the real axiotic surface.

## freeze REFUSED, and it is right

16 `CONFLICT?` rows exist that **no adjudicator was ever assigned** — the workflow
batched by index over `SWAP?` only, and the `CONFLICT?` tag was added *after* the
run launched. So the corpus is built against an **undeclared partition** and has
no `partition_digest`.

Those 16 shipped with CIRIS values inside the alt arm. Per the asymmetry above
that is the *conservative* direction — it biases `values_effect` toward zero and
weakens a positive result rather than creating a confound — so it does not
invalidate a positive finding. It does mean the alt arm is **under-swapped by an
unknown amount** until they are adjudicated.

## Three defects found in this run, all mine

**1. `verify` reported success over lines it never checked.** It gated on
`tag == "HOLD"` while `assemble` byte-copies every non-SWAP tag. Its own output
gave it away — `1109 + 28 = 1137` against a 1153-line file — and nothing said so.
Fixed to gate on `tag != "SWAP"`, with an assertion that the row accounting
closes. Re-run: 1125 + 28 = 1153.

**2. The reconcile prompt truncated its own inputs.** `slice(0, 3000)` per
adjudication cut all eight mid-row; **46 of 107 candidates arrived with one
verdict or none**. Reconciling from that would have read truncation as reviewer
silence and dropped ~27% of judgements. The agent caught it, recovered the full
outputs from the workflow journal, and reconciled from those.

**3. I overwrote `accord.tsv` while the workflow was reading it** — the precedence
fix landed mid-run, which is how the 16 unassigned `CONFLICT?` rows appeared.

## Adjudicator agreement

| batch | n | raw | Cohen's κ |
|---|---|---|---|
| 1 | 27 | 0.926 | 0.787 |
| 2 | 27 | 0.815 | 0.611 |
| 3 | 27 | 0.778 | **0.571** |
| 4 | 26 | 1.000 | 1.000 |
| **all** | **107** | **0.879** | — |

Batch 3 at κ=0.571 is poor and below the 0.80 bar. Disagreements resolved to HOLD
per the tie-break, which is the conservative direction, but a batch that
disagrees on 6 of 27 is not a settled adjudication.

## What the methodology research found

**No established protocol covers document-level normative substitution.** Six
mature techniques each solve part of it. The structural finding is the one that
matters: *in every field that solved this, the guarantee is enforced at BUILD
TIME, not verified after authoring.* The field's answer to a drifting author is
not a better reviewer — it is a template the author cannot write into.

That is what this module now is, and the two failed review passes are why.

Also flagged, and it is a real threat to the design: message-effects research
holds that **one document per arm is underidentified in principle** — no
post-hoc check rescues it, because a single stimulus confounds the manipulation
with every idiosyncrasy of that particular text. Multiple alt corpora would be
the fix. Recorded rather than resolved.

## Next

1. Assign the 16 `CONFLICT?` rows to duplicate adjudication.
2. Re-adjudicate batch 3.
3. `freeze` to obtain a `partition_digest`, and re-assemble against the frozen file.
