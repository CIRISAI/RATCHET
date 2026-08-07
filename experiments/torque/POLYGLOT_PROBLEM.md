# The accord has three forms, and the alt corpus can only replace one

Found 2026-08-07 while building the arm manifest — i.e. at the step that turns an
authored corpus into something runnable, which is where this class of problem
surfaces.

## Measured

`corpus:accord.*` is three keys, not one, and they reach different parts of the
pipeline:

| key | bytes | steps | reaches |
|---|---|---|---|
| `accord.localized` | 54,725 | 8 | the whole ASPDMA family |
| `accord.polyglot_compressed` | 7,215 | 5 | PDMA, CSDMA, DSDMA, IDMA, csdma_bounce |
| `accord.polyglot_full` | 180,522 | 8 | **all four conscience faculties**, ×2 (image variants) |

The two polyglot forms are **woven from 15 languages by semantic weight** — the
corpus header calls it "epistemic triangulation, not a translation."

**The authored alt corpus is English.** It can replace `accord.localized`. It
cannot replace the other two without destroying the property that makes them what
they are.

So as things stand the alt arm would carry CIRIS values on **13 of 21 steps**,
including every conscience faculty. This is CIRISAgent#995 in a new guise: the
accord reaches the consciences by a route the obvious substitution misses.

## Four options, none free

**1. Put the English alt everywhere.** The manipulation becomes complete, and it
adds a language confound: the alt arm is monolingual where the CIRIS arm is
polyglot. Direction unpredictable. This is exactly the trap already flagged for
`optimization_veto_conscience` — *a monolingual replacement confounds values with
language coverage and would look exactly like a clean result.*

**2. Replace only `localized`, leave the polyglot forms as CIRIS.** Language held
constant, manipulation partial. Bias is **toward zero** — conservative — but
`values_effect` would then be measured over 8 of 21 steps and **not over the
consciences at all**, which is where the values arguably matter most.

**3. Translate the alt accord into polyglot form.** Expensive, and translation
quality becomes its own confound. The polyglot forms are not translations
anyway — reproducing "weaving by semantic weight" is a research project, not a
build step.

**4. Make BOTH arms monolingual English on all three keys.** ← RECOMMENDED

Set `accord.localized`, `accord.polyglot_compressed` and `accord.polyglot_full`
to the English CIRIS accord in the CIRIS arm, and to the English alt accord in
the alt arm.

- language coverage is **held constant across arms**, so it cannot confound
- the manipulation is **complete**: all 21 steps see alt values in the alt arm
- the contrast measures values and nothing else

The cost is a **stated domain limit**: the campaign then measures a *monolingual
variant* of H3ERE, not shipped H3ERE. Every arm — including `h3ere-ciris` — runs
a configuration the product does not ship.

That limit is honest, bounded, and reportable. Options 1 and 2 both leave a
confound in the estimate itself; option 4 moves the cost into a clearly stated
boundary on what the result generalises to, which is the trade this campaign has
made everywhere else.

## What it means for the claim

With option 4, the finding reads: *"in a monolingual English configuration,
swapping the value corpus changes/does not change behaviour by X."* It does not
license a claim about the shipped polyglot agent without a further assumption
that the polyglot weave does not interact with the values manipulation — and that
assumption is **untested and should not be smuggled in**.

Worth stating plainly in the paper rather than in a limitations paragraph: the
polyglot accord is 180,522 B reaching all four consciences, and the campaign does
not test it.

## Consequence for the corpus already built

Nothing authored is wasted. The English alt accord becomes the source for all
three keys rather than one. The partition, digest and verification all stand
unchanged — the same 1,153-line artifact is simply pointed at three keys instead
of one.

## Open

`accord.polyglot_full` is 180,522 B against the English accord's ~40 KB. Setting
the full key to the English text is a **4.5× reduction in the text the consciences
receive**, in both arms. Held constant, so not a confound — but a large change to
the configuration under test, and it must be measured and declared, not waved
through as "both arms got the same thing."
