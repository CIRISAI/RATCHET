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
