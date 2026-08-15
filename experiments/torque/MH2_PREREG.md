# MH-2 — is the machinery value-neutral, or is the covenant carrying the safety?

Pre-registered before any cell runs. Extends MH-1 with the two arms it lacked.

## The question, and why it is not answered yet

MH-1 measured `bare` / `values-ciris` / `h3ere-ciris` on the staged mental-health
battery and found the machinery does the work: the covenant **as a prompt** does
nothing (26.7% vs bare's 22.2%, p=0.81), the covenant **inside the pipeline**
cuts hard failures to 4.4% (p=0.0071 against the prompt arm).

It did **not** vary the covenant inside the pipeline. So MH-1 cannot say whether
the 4.4% comes from *these values* or from *the machinery that carries any
values*.

TORQUE answered a neighbouring question and does not settle this one: it found
the axiotic content equivalent within 5 points — but on **HE-300 concordance**,
an agreement-with-annotators score, not on **safety failures**. Different DV,
different instrument, no transfer.

RATCHET#17 asks CC 1.0-rc3 to re-base Part VI on the claim that "the machinery is
explicitly value-neutral — the safety content lives in the covenant." **That is
the proposition under test here, and MH-1 currently points against it.**

## Design

Identical to MH-1 in every respect except the accord the pipeline carries.

| arm | accord | status |
|---|---|---|
| `h3ere-ciris` | CIRIS accord | MH-1, 2/45 |
| `h3ere-neutral` | same shape and length, values drained | **NEW** |
| `h3ere-blank` | axiotic corpus emptied | **NEW** |

Same battery (`english_mental_health`, 9 staged turns), same five models, same
`TEMPLATE=default`, same shipped `safety_interpret` scorer, same judge
(`anthropic/claude-opus-4-5`). 45 turns per new arm.

## Pre-registered readings

| result | reading |
|---|---|
| neutral and blank ≈ ciris (~4.4%) | **the machinery is value-neutral.** The covenant content is not what buys the safety, and #17's arm-C framing — "the safety content lives in the covenant" — needs rewriting before CC cites it |
| neutral and blank fail markedly more | **the covenant content is load-bearing for safety.** CC's axiomology survives contact, and MH-1's effect is values-plus-machinery, not machinery alone |
| blank fails but neutral does not | the *scaffold* needs content of some kind, but not these values — a third reading neither #17 nor MH-1 anticipated |
| all three comparable to `values-ciris` (~26.7%) | MH-1's pipeline effect does not replicate; treat MH-1 as provisional |

## Bounds

45 turns per arm, one language, one battery. Resolves a large difference, not a
small one. A null between ciris and neutral is **"the battery does not separate
them at this n"** — not equivalence, and not licence to call the machinery
value-neutral. Distinguishing "no difference" from "too few turns" would need the
TOST treatment TORQUE used, and this is not that.

The U5 caveat from MH-1 carries: identity drop failed once in every arm there,
pipeline included.
