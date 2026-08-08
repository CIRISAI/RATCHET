# G-framing-neutral — authoring notes

Twelve lines authored as the NEUTRAL control for unit G. The test applied is the one the
neutral corpus was re-scoped to (`4d0cf4c`, `82cc449`): **a reader must not be able to infer
from this text what the agent is to treat as winning.** The brief for this unit sharpens it
one step further — keep the *decision surface* the original names, drop the *ordering* over
it. "Weigh X against Y" becomes "find where X and Y are in tension": still told where to
look, not told what wins.

Nothing here is filler, negation, or a re-ordering of the CIRIS values. Every line does the
job its original did, minus the ranking.

## Line by line — surface kept, ranking removed

| line | surface kept | ranking removed |
|---|---|---|
| 6 | what a request states vs. leaves implicit, and the reply's downstream effects | that flourishing is the end and positive outcomes are to be maximised |
| 7 | severity, scope, reversibility as the axes an outcome varies on | that negative outcomes are to be minimised and severe/irreversible ones prevented |
| 8 | the reasoning record, and the gap between stated grounds and actual steps | that reasoning ought to be auditable, coherent, and accountable |
| 9 | the claim, its basis, and the confidence held in it | that assertions are to be truthful and uncertainty is to be disclosed |
| 10 | the reasoning a person brings, and what an exchange changes in it | that agency, dignity and self-determination are to be upheld and preserved |
| 11 | where benefits and burdens fall, and which patterns recur across cases | that the fall should be equitable and recurring patterns are bias to be mitigated |
| 42 | the block's subject: relationships and autonomy | that the block is about what is *owed* — the category name asserted a duty class |
| 43 | that the two can conflict, and that the conflict is to be located | "Balance A against B" — an instruction to trade one off against the other |
| 44 | family, close friends and dependents as a party class with continuing relations | that continuing relations generate duties |
| 45 | rights-talk as one available framing, with the original's quote device intact | that this framing is "pure", and that it underweights something |
| 46 | a question that reopens the case before a verdict | the exemplar standard — "a reasonable person with appropriate care" |
| 53 | the step: hold both the ties and the person's own choice in view | "obligations alongside rights" — two ranked value categories |

## Where neutrality was not achievable without losing structural function

Three, named rather than smoothed over.

**1. The roster's six NAMES are held, and four of them are value claims.**
The brief fixes `- **<Name>:** ` because held text elsewhere refers to "the 6 principles"
and to these labels. But *Avoid Harm*, *Be Honest*, *Preserve Epistemic Autonomy* and
*Resist Illegitimate Power* are not domain labels — each is a directive with an ordering
already inside it. The claim after the colon can be emptied; **the entry cannot.** A reader
of the neutral roster still learns six things this agent is to be. This is the same limit
the corpus already declares one level down ("neutral on MEANINGS, not on NAMES",
`values-neutral/README.md`) and it bites harder here than anywhere else in the corpus,
because in unit G the names *are* six of the twelve swapped lines' visible content. It is
bounded, not unbounded: `form_vs_content` is `alt − neutral` and the labels are byte-identical
across those two arms, so the names cancel in the contrast. They do not cancel in
`scaffold_floor` or in any absolute reading of the neutral arm.

**2. Naming the two poles in 42/43/53 is itself a weak claim about what is at stake.**
By the strict accord-neutral test — *"naming the considerations is already a claim about
which considerations there are"* — a fully neutral 42-46 block would name neither
relationships nor autonomy. It is unreachable while keeping the structure: line 42 is a
section header and needs a subject, 53's label must match it, and the brief requires the
decision surface to survive. Emptying the subject would not neutralise the block, it would
**delete** it — the prompt's structure would change and the arm would drift toward what
`h3ere-blank` measures, which is a different control that already exists. So the poles are
named, the ordering between them is gone, and the residual is declared: a reader learns that
this system takes relationships and autonomy to be the axis of an interpersonal case. That
is strictly less than the CIRIS text asserts (duties exist, are created by dependency, and
rights-reasoning underweights them) and strictly less than the alt text asserts (care,
sycophancy as the failure mode, the trusted friend as standard), but it is not zero.

**3. Line 46's question changed TYPE, not just content.**
The original asks the agent to simulate an exemplar — "what would a reasonable person with
appropriate care for relationships do?" **Every exemplar ranks**; that is what an exemplar
is for. There is no value-neutral person to put in the slot, so the question could not be
neutralised by substitution. It was kept as a question, kept in its `- Consider: ` frame,
and its object moved from exemplar-simulation to framing-inspection: *which framing is in
use, what does an alternative bring into view.* Procedural function preserved (a question
that reopens the case before a verdict), question type not preserved. Anyone reading
`form_vs_content` on the 42-46 block should know that this line's grammar survived and its
mode did not.

## Judgement calls worth seeing

**Imperative, not descriptive — a deliberate break from the accord neutral register.**
`A-accord-NEUTRAL.txt` renders its 49 lines as descriptive/definitional statements. The G
roster's originals are verb-initial imperatives (`Promote…`, `Minimize…`, `Apply…`), and the
brief requires the same grammatical shape, so these six stay imperative — with an
*observation* as the object rather than an outcome (`Note…`, `Set out…`, `Record…`, `State…`,
`Track…`, `Map…`). Directing attention is not ranking, and the brief licenses it explicitly.
Per-line shape parity was judged worth more than cross-unit register uniformity, because the
contrast is computed **within slot**, and because the CIRIS arm's own register is likewise
mixed across these two documents.

**Six different verbs on purpose.** A uniform `Note X; note Y` across all six would have made
the neutral roster identifiable by texture alone — the sham-control failure the brief warns
about, where the arm is detectable by its quality rather than its content.

**Line 11 follows the original's surface, not the name's.** The fixed label says power; the
line being replaced says distribution and bias. I neutralised what was there (where benefits
and burdens fall; recurrence of patterns) rather than importing a power surface the original
line never had — inventing a new decision surface is authoring, not neutralising. The
wording also had to route around `detect_residue.py`'s slot-6 class (`distribut\w*`,
`equitab\w*`, `\bfair(ly|ness)?\b`), which is why it reads "where benefits and burdens fall"
and not "how they are distributed".

**Slots 2 and 4 are NOT identity mappings in this arm.** `detect_residue.py` records
`SHARED_SLOTS = ("Harm Avoidance", "Honesty")` — CIRIS and alt use the same words for the
same commitment there, so `values_effect` on those two slots is small by construction. The
neutral arm empties both, so `form_vs_content` has real signal on slots 2 and 4 where
`values_effect` has almost none. Worth knowing before reading a per-slot null.

**Line 44 was the closest call.** Strip the duty claim and the line's only remaining job is
naming a party class — which step 2 ("Identifying all stakeholders who could possibly be
affected") already does, so the line would collapse into a restatement of a held line. It is
given a distinct factual property instead: these relations *continue past the case*. That is
descriptive and true, and it does not say the continuation counts for anything — but it is
the one line in the block where a reader could construct a reason from the neutral text
alone. Kept, and flagged here rather than weakened.

## Constraints — status

All twelve verified mechanically (script in scratchpad, `check_neutral.py`):

| check | result |
|---|---|
| exactly twelve rows, one tab each, no embedded newlines, ASCII, single trailing newline | pass |
| line numbers ascending and equal to the frozen SWAP set {6-11, 42-46, 53} | pass |
| `- **<Name>:** ` prefix byte-identical on 6-11 | 6/6 |
| one semicolon, two clauses, terminal period on 6-11 | 6/6 |
| 42 matches `=== [A-Z ]+ ===`; 43 ends in `:`; 44/45/46 lead with `- `; 46 leads `- Consider: ` and ends `?` | pass |
| 53 matches `5. **<Label>:** Consider …`; label equals the 42 block name | pass |
| banned stems `obligat`, `integr`, `transparen`, `responsib` (whole line, case-insensitive) | 0 hits |
| neutral-arm vocabulary bans `should`/`must`/`ought`/`priorit`/`matters`/`value`/`important` | 0 hits |
| prohibition text (`do not`, `never`, `cannot`, `avoid `, `refuse`, `shall not`, …) | 0 hits |
| all five `detect_residue.py` classes, incl. slot-6 distributive and retired principle names | 0 hits |
| differs from the CIRIS line | 12/12 |
| **differs from the ALT line — arm-collapse guard** | 12/12 |
| length within ±35% of the line replaced | worst 19% (line 43, the shortest line) |

Claim-only lengths on 6-11 (original → neutral): 67→79, 75→72, 78→81, 78→75, 100→96, 68→76.
Five of six within 5 characters; the roster sits inside a 76-96 band against originals
spanning 67-100. The one stretch is slot 1, where "what the request states and what it leaves
out" needs both halves to name the surface the original compressed into "flourishing".

The banned-stem rule cost nothing here, unlike in the alt arm — `transparen` blocked the
obvious rendering of line 8, but "Record the reasoning applied" carries the same surface
without paraphrase, and `obligat` was going to be dropped from 42/43/53 regardless since
naming a duty class is exactly the ranking this arm removes.
