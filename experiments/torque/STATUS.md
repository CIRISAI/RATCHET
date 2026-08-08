# TORQUE — build status

## Corpora

| arm | corpus | state |
|---|---|---|
| `bare` | none needed | ready — direct-provider, no system content |
| `values-ciris` | none needed | ready — `direct_provider.py` injects the same source bytes the h3ere arm holds |
| `h3ere-ciris` | the original | ready |
| `h3ere-alt` | `A-accord-FINAL.txt` | **verified** — 49 SWAP / 1104 HOLD |
| `h3ere-neutral` | `A-accord-NEUTRAL.txt` | **verified** — same partition, 49 / 1104 |
| `h3ere-blank` | empty axiotic | trivial |

All three value arms share **one frozen partition**. Two numbers, and the second
is the one a reader recomputes:

| comparison | lines differing |
|---|---|
| substituted intermediate → alt (the authored meanings) | 49 of 1,153 |
| **shipped accord → alt (what the arms actually differ by)** | **68 of 1,154** |

The extra 19 are principle names replaced mechanically from the frozen term
table. Across the whole composed manifest the manipulation is **230 of 5,558
lines, 4.1%**. `preflight.py` asserts all three figures against the artifacts, so
they cannot drift out of the prose again.

## Units

| unit | state |
|---|---|
| B-optveto, B-epihum, B-coherence | built and verified |
| C-pdma, D-aspdma, E-exemplars | built and verified |
| F-lg-axiotic | corrected — `09_trusted_person_first_step` promoted to SWAP; needs authoring |
| G-pdma-framing | partition frozen (6 SWAP); authoring in flight |

## Remaining before the pilot

1. G-framing's 6 slot meanings — in flight
2. F's 1 line
3. Six arm manifests (`build_arm_manifest.py`; the three `template` identity
   fields are **ontological → hold**, so they take live CIRIS values identically
   in every arm)
4. The pilot: 10 questions, ~$14, against `PILOT.md`'s pre-declared gates

## Design corrections still worth knowing

- **`action_tier` is undefined for direct-provider arms.** Cross-harness
  contrasts (`pipeline_effect`, `scaffold_floor`) are measured on
  `ground_truth.concordance`, the only instrument defined on both sides.
- **The conversation is the unit of analysis.** A stateless per-question runner
  is a different instrument and its numbers may not sit beside these.
- **Slots 2 and 4 are identity mappings** — `values_effect` on them is small by
  construction.
- **Monolingual only.** All three accord forms carry English in every arm; the
  shipped agent runs two of them polyglot.
- **The neutral arm is neutral on meanings, not on names.**
