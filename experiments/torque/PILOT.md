# Stage 0.5 — the pilot run

**10 items × 6 arms, on real data, before the staked run.** Its job is to
establish that the instrument works end to end. It is *not* a small version of
the experiment.

## The rule that makes it safe

**The pilot's outcome data is discarded, and the pilot items are excluded from
the main draw.**

Both halves matter. A pilot that reports effect sizes is an invitation to tune
the design against them, and that is the forking path the whole pre-registration
exists to close — the temptation is strongest precisely when the numbers are
nearly good. So the pilot reports **instrument health only**: did it run, did it
record, did it verify. Concordance, contrast estimates and anything resembling a
result are not computed, not looked at, and not stored.

Excluding the 10 items from the main draw costs 4% of the corpus and removes any
question about double-counting.

If that feels over-strict: the cost of being wrong here is a staked campaign
whose design was shaped by data it claimed not to have seen, and no amount of
later care repairs that.

## Cost

10 items × 9 turns = 90 thoughts per arm. Four pipeline arms at 23 calls per
thought plus two direct arms ≈ **8,500 calls, ~4% of a full run, ≈$14.**

Cheap enough to run more than once, which is the point — a pilot you can only
afford once is a pilot you will be tempted to interpret generously.

## Pre-declared pass criteria

Every one is mechanical. None requires judgement about whether a result looks
right. The pilot FAILS if any fails, and failing is a normal outcome.

### A. The run happened at all
| # | check | fails if |
|---|---|---|
| A1 | all 6 arms complete | any arm errors out or produces < 10 arcs |
| A2 | `absent_cohort` guard passes per arm | fixed durations, absent task ids, or still-processing literals |
| A3 | durations vary plausibly | spread < 2% across responses — that is a ceiling, not deliberation |
| A4 | no `LLM_ERROR` retries above baseline | model too small for the DMA schemas (#892–895); would also inflate `retry_depth`, a DV |

### A5–A7. Provider caching (added after the cost estimate assumed it)
| # | check | fails if |
|---|---|---|
| A5 | duplicate-probe test: same prompt twice at temp 0.7, ×20 per arm | any byte-identical completion pair — a cache is serving, or a seed is pinned |
| A6 | arms interleaved, not sequential | cache warmth correlates with arm, and warmth moves latency, which the absent-cohort guard reads as deliberation |
| A7 | response caching confirmed OFF at the provider | a completion cache destroys the only variance source AND can return one arm's answer to another |

A5 is the sharp one. `seed` is not transmitted (agent#975 [M-N1]), so temperature
is the entire variance source. A response cache makes repeats byte-identical —
fake replicates, which is exactly what the no-live-variance refusal exists to
catch, arriving by a route that refusal does not check.

The design is maximally cache-exposed by construction: the mechanical partition
makes 1,125 of 1,153 accord lines byte-identical across arms. The shared prefix
is the method working, and it is also the largest cache surface available.

### B. The manipulation actually landed
| # | check | fails if |
|---|---|---|
| B1 | compose gate PASSES before the run, per arm | any arm's ablation did not reach its blocks |
| B2 | `manifest_digest` recorded, and **different** per arm | two arms silently shared a manifest |
| B3 | `partition_digest` matches the frozen partition | corpus built against an undeclared partition |
| B4 | `residue_digest` identical across arms | the shared uncovered surface drifted mid-run |
| B5 | `conscience_guidance_mode` sealed in every trace | arm D's audit cannot be performed |

### C. The measurement is possible
| # | check | fails if |
|---|---|---|
| C1 | `attempt_index` / recursion point / `action_was_overridden` present | cannot stratify by delivered dose |
| C2 | every item carries its measured class label | analysis would fall back to subset names |
| C3 | gold labels join to every item | no external standard to score against |
| C4 | `cm_test` polarity handled | one stratum scores inverted |

### D. The arc survives — the void condition most likely to fire
| # | check | fails if |
|---|---|---|
| D1 | 9 turns sustain without degenerating | agent loops, repeats, or terminates the arc early |
| D2 | withdrawal at turn 5 lands **between** thoughts | any thought is split across the switch |
| D3 | the scrubbed-history variant runs | the reversion discriminator is unavailable |

D1 is the one to watch. The corpus measurement found the short items (median
56 chars) cannot drive a 9-turn arc unrewritten, while long-form AITA (median
1,550 chars) probably can. **The pilot draws from the axiotic-primary stratum,
which is the long-form one** — so it tests the case the design depends on rather
than the easy case.

## Sampling

10 items, seeded, drawn from the **axiotic-primary stratum** and recorded by id
before the run. Not a random spread across strata: the pilot's job is to stress
the path the campaign's main claim rests on, and a pretty distribution across
four strata would test the instrument on material that matters less.

## What a failure means

Each failure has a known remedy, and none of them is "adjust the hypothesis":

- **A4** → larger model, or de-union the DMA schemas first
- **B1–B4** → re-freeze and re-assemble; do not run until digests line up
- **C1** → agent-side work; the fidelity fields are not optional
- **D1** → the arc question is answered per-stratum, and a failure here means
  the 9-turn design does not survive contact with this corpus. That is a
  **VOID** condition, declared, not a reason to shorten the arc until it passes.

## After it passes

Re-freeze nothing, change nothing, and run the staked campaign against the
already-frozen artefacts. The pilot's only output is a checklist with every box
ticked and a run URL that proves it.
