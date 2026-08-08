# TORQUE-1 — design review, causal identification lens

Reviewed 2026-08-07 against the artifacts, not the prose describing them. Every
number below was re-measured from `arms/*.json`, `corpora/`, `partition/`,
`build_he300_arcs.py` and the raw ETHICS CSVs; where a measurement contradicts
the regime, both are given.

Scope note on the ten-turn change: the arc is now 10 turns with the withdrawal at
the midpoint and each half balanced on item count and gold-label mix
(`build_he300_arcs.py:127-168`). That change is real and it closes part of
Finding 3. It does not close Finding 3. See that finding for exactly which
component it removes and which it leaves, and Finding 8 for a side effect it
introduces.

---

## 1. The regime describes a different manipulation than the one that was built

**Claim.** `alt_values` specifies a values arm built by excerpting and adapting an
external constitution; the artifact that will run is the CIRIS Accord with 68 of
1,154 lines changed, and the values manipulation touches 4.1% of the composed
manifest.

**Grounded in.** `TORQUE_REGIME.yaml:128-224` describes excerpting the CORE-densest
~6,800-word span of the Anthropic constitution, mechanical 3rd→2nd person and
should→must conversion, stripping the hard constraints, "woven through all 88
mixed blocks". Measured on `arms/h3ere-ciris.json` vs `arms/h3ere-alt.json` (192
keys, 5,558 lines):

| key | lines | ciris↔alt changed |
|---|---|---|
| `corpus:accord.{localized,polyglot_compressed,polyglot_full}` | 1,154 | 68 (5.9%) |
| `corpus:polyglot.pdma_framing` | 56 | 12 (21.4%) |
| `conscience_prompt:optimization_veto_conscience.system_prompt` | 438 | **4 (0.9%)** |
| `conscience_prompt:epistemic_humility_conscience.system_prompt` | 138 | 2 (1.4%) |
| `conscience_prompt:coherence_conscience.system_prompt` | 106 | 3 (2.8%) |
| `dma_prompt:pdma_ethical.system_guidance_header` | 127 | 3 (2.4%) |
| `dma_prompt:action_selection_pdma.csdma_ambiguity_guidance` | 1 | 1 |
| `string:…language_guidance.09_trusted_person_first_step` | 2 | 1 |
| **whole manifest** | **5,558** | **230 (4.1%)** |

`optimization_veto_conscience.system_prompt` — called "the single hardest
congruence problem in the campaign" at `:1067-1071` — is resolved by leaving
99.1% of it untouched, and it is still polyglot in every arm (5.13% non-ASCII;
Hebrew, Arabic, Chinese, Greek in its header). Held constant, so not a confound,
but the alt values do not meaningfully reach the optimization-veto conscience.

Two downstream consequences:

- Every number in `congruence_measured` (`:187-200`) and `register_confound`
  (`:202-212`) was measured on the constitution, not on `A-accord-FINAL.txt`.
  Re-measured on the artifact that runs (per 1,000 words, ciris → alt):
  `must` 3.22→3.17, `should` 0.13→0.13, `obligation` 2.68→2.24,
  `transparen` 3.48→1.98, `integrity` 2.28→0.00 (that one is the name table).
  The register confound is **gone by construction** — the alt inherits the
  Accord's 2nd-person imperative on 96% of lines — which is better than the
  regime claims, but it means `register_confound.remedy` describes work that the
  build made unnecessary rather than work that was performed. And
  `pre_registered_asymmetry` (`:220-223`) pre-registers a density shortfall that
  does not exist in the running artifact.
- The published "49-line diff" is against `accord-substituted.txt`, not against
  the shipped accord. Verified: ciris↔alt = 68 lines, substituted↔alt = 49,
  substituted↔neutral = 49, and the two swap line-sets are identical. The extra
  19 lines are the term table (`corpora/terms.tsv`, 62 lines: Beneficence→
  Helpfulness, Justice→Pluralism, Adaptive Coherence→Holistic Judgment, …). So
  STATUS.md:15-17 and `:1227` — "1,104 of 1,153 byte-identical, difference set
  exactly the declared 49" — are true of the intermediate and false of the
  contrast anyone will recompute.

**Severity.** blocks-the-run. Not because the build is wrong, but because a
one-shot pre-registration that describes a different manipulation cannot be
corrected afterwards.

**Remedy.** Rewrite `alt_values` to describe the built corpus. Re-run the
congruence lexicon on `A-accord-FINAL.txt` and pin those numbers. Declare the
19-line term substitution as part of the IV (it is 28% of the ciris↔alt
difference and is applied to the alt/neutral branch only). Restate the hypothesis
as the marginal effect of a 49-line axiotic swap plus a principle-name rename
inside an otherwise-identical Accord, and re-derive whether MDE 0.15 is credible
at that dose.

---

## 2. Both kills name an instrument that does not exist on one of their arms

**Claim.** `reversion_null` stakes on `action_tier`, which is undefined after the
switch it measures; `form_vs_content_null` stakes on `action_tier`, which is not
declared for one of its two arms.

**Grounded in.** `kills.reversion_null.instrument: action_tier` (`:345`) and
`contrast_instruments.reversion.instrument: [action_tier, …]` (`:621`), against
`withdrawal.means: "harness h3ere -> bare"` (`:375`) and the file's own statement
at `:592-601` that action_tier does not exist for direct-provider arms — `bare`
is a direct-provider arm. Separately, `kills.form_vs_content_null.instrument:
action_tier` (`:354`) against `dv.action_tier.arms: [h3ere-ciris, h3ere-alt,
h3ere-blank]` (`:633`) — **h3ere-neutral is absent**, almost certainly a leftover
from when that arm was called `nonsense`.

This is the same defect the `kills` section was written to close (`:319-331`,
"a kill without both is decoration"), arriving one level down.

**Severity.** blocks-the-run, for both kill legs.

**Remedy.** Both kills score on `ground_truth.concordance` plus `unknown_rate`
only. Restate the 0.15 equivalence bounds on that scale — they were set equal to
`mde.values_effect`, which is also a concordance-scale number, so the restatement
is coherent. Add `h3ere-neutral` to `dv.action_tier.arms` if action_tier is to be
reported descriptively for it.

---

## 3. Reversion is still collinear with turn index; balanced halves close composition, not position

**Claim.** The ten-turn balanced-halves change removes the item-composition
confound but leaves the position confound untouched, and position is the whole of
what distinguishes "before" from "after" inside a single arm.

**What the change closes.** `build_he300_arcs.py:141-168` refuses odd `--turns`
and gives every half of every arc `per_half_ones = half // 2` label-1 items — at
`turns=10`, exactly 2 label-1 and 3 label-0 per half. The builder's own comment
(`:133-136`) states the reason correctly: an unbalanced gold mix across the
switch makes a concordance drop item difficulty, indistinguishable from
reversion. That is now closed by construction, and it was the sharper of the two
composition problems I raised — it also defuses the majority-class baseline
asymmetry that `hazards_measured.virtue_class_imbalance` (`:564-570`) warns about
landing differently in each half. Good change; keep it.

**What it does not close.**

- *Transcript position.* Post-switch turns always sit on five turns of
  accumulated history; pre-switch turns sit on zero to four. Context length,
  self-consistency pressure (a model tends to keep answering in the register it
  opened in), and instruction-following decay on a 17B model reading long-form
  AITA items (`axiotic_primary` median ~1,550 chars, five of them plus responses
  by turn 6) are all monotone in turn index and land entirely on the post half.
  Balancing the gold label does nothing to any of them, and every one of them
  points the same direction as the kill's assertion: a model that has settled
  into a register keeps it, which reads as "no reversion".
- *Everything about an item except its binary label.* Balance is on `gold` only.
  Within label-0, items vary from 600 to several thousand characters, and item
  length is itself what drives the context growth in the point above.
- *Balance is a property of the draw; the analysis runs on the extracted subset.*
  This is a new interaction with Finding 6. If UNKNOWN rate rises with turn index
  — which is the expected direction — the *scored* halves are no longer balanced
  even though the *drawn* halves are. The balancing guarantee does not survive
  attrition.
- *There is still no unwithdrawn control.* `contrast_instruments.reversion.arms:
  [h3ere-ciris]`, "within one arm, across the switch" (`:620`), and
  `switch_point: the arc midpoint — after turn 5 of 10, fixed` (`:382`). With a
  fixed switch and no comparison arc, position and withdrawal are perfectly
  collinear.
- *The scrubbed discriminator is not clean.* `history.scrubbed` (`:394`) removes
  the piped outputs **and** five turns of context simultaneously, so "immediate
  reversion when scrubbed" is equally explained by context loss. `:394` also does
  not pin whether the user turns survive the scrub, which determines what the
  discriminator discriminates.

**Severity.** blocks-the-run, for the reversion claim. Unchanged by the ten-turn
fix, because the fix and the gap address different things.

**Remedy.**
1. Difference-in-differences against `h3ere-ciris` conversations run all ten
   turns unwithdrawn — same items, same positions, one extra cell. The balanced
   halves make this *better* than it was: the control arm's own pre/post
   difference is a direct, matched estimate of the position effect, and the DiD
   subtracts it exactly. This is the single cheapest fix in the review.
2. Add a **substituted-history** condition — the piped turns replaced by
   bare-arm turns on the same items — so history length and structure are held
   and only the provenance of the assistant's prior text varies. Keep scrubbed as
   a third condition; it is informative, just not sufficient alone.
3. Balance halves on a length decile as well as on the label. Free, and
   pre-registerable now.
4. Pin what scrubbing removes, in words, at `:394`.
5. Report per-half `unknown_rate` and re-check label balance on the scored items.

---

## 4. `form_vs_content` cannot separate form from content

**Claim.** The neutral arm is the same value document at 96%, carrying the alt
arm's principle names and the Accord's full deontic register, so a null on this
contrast does not license the conclusion the kill draws from it.

**Grounded in.** Measured: alt↔neutral differ on 49 of 1,154 accord lines (4.2%)
and 168 of 5,558 manifest lines (3.0%). Both descend from
`corpora/values-alt/accord-substituted.txt`, so both carry the alt name table
(`corpora/terms.tsv:6-40` — Helpfulness, Harm Avoidance, Ethics, Honesty,
Pluralism, Epistemic Autonomy). Re-measured on `A-accord-NEUTRAL.txt`, per 1,000
words: `must` 2.89, `obligation` 2.24, `harm` 3.16, `honest` 1.45, `decept` 1.32,
`autonomy` 0.79.

`neutral_control.test` (`:415-419`) sets the standard itself: "Could a reader
infer, from this text alone, ANYTHING about what the agent should prioritise…
Naming the considerations is already a claim about which considerations there
are." The arm fails that test on 1,104 of 1,153 lines, because `construction`
(`:420-426`) scoped the no-should / no-prioritising rule to the 49 SWAP lines
only. STATUS.md:47 records this in one line — "the neutral arm is neutral on
meanings, not on names" — without following it into what it does to the kill.

`form_vs_content_null` (`:351-359`) is declared fatal to the campaign's headline
claim. If it fires, the supported inference is "49 lines add nothing on top of a
corpus that already names the same six values", not "the pipeline responds to
value-shaped text rather than to what values say". That is a wrong kill of the
headline claim, and it is the most expensive single error available here.

**Severity.** blocks-the-run, for that kill as stated.

**Remedy.** Either restate the kill as a dose claim on the 49 lines — honest,
much weaker, and immediately defensible — or rebuild the neutral arm from a
partition that also empties the roster and name surface, accepting that it then
diverges structurally from alt. Report the measured neutral densities beside the
contrast either way.

**Credit where due, one line:** alt↔neutral is the cleanest contrast in the
campaign — identical base, identical swap line-set (verified by set equality),
no term-substitution difference between the two arms.

---

## 5. n and power are stated in three mutually inconsistent units, none of them the declared analysis unit

**Claim.** The power calculation is in items, the design's independent unit is the
conversation, the budget is sized for a corpus almost twice the pinned draw, and
the whole arithmetic is now stale by 9 turns versus 10.

**Grounded in.** `corpus.n_total: 1200` (`:559`); `budget` "250 items x 9 turns =
2,250 thoughts per arm" (`:650`); `repeats.conversations_per_cell: 20` (`:801`);
power "at n=250, paired" (`:514-518`), declared to discharge `power_note`
(`:357-363`). Against `contrast_instruments.unit_of_analysis: conversation`
(`:623`) and `repeats.unit: conversation` (`:800`).

After the PILOT_BLOCKER resolution a conversation is now ten items
(`build_he300_arcs.py:10-20`), so 1,200 items = 120 conversations, while the
budget is sized for 2,250 items — 1.9× the pinned draw. The two equivalence
kills' stated requirement of "n~92-215 at the 0.15 bound" must be met in
*conversations*; at 20 per cell it is short by an order of magnitude. Even taken
at item level, the design effect 1+(m−1)·ICC at m=10 — in a design deliberately
engineered to make turns dependent, one channel, transcript carried forward —
cuts effective n by roughly 2-4×. `power_note` is not discharged.

**Severity.** blocks-the-run.

**Remedy.** Pick the unit and restate every n in it. Recompute both equivalence
bounds under a declared ICC, and declare the ICC as an assumption rather than
discovering it. Re-derive the budget at 10 turns. If the kills cannot be powered
at 0.15 in conversations, the honest options are the ones `power_note` already
names — a larger draw or a wider declared bound — never a bound chosen after
seeing the interval.

---

## 6. Differential UNKNOWN attrition on the two cross-harness contrasts

**Claim.** Concordance computed on the extractable subset compares a pipeline arm's
self-selected subset against a bare arm's near-complete one, and this is the most
likely way the campaign produces a confident result that is wrong.

**Grounded in.** `the_cost_of_that_choice` (`:1375-1380`): "Measured: 4 of 6
representative responses returned UNKNOWN under strict." `affects` (`:1421-1426`):
"Those are exactly the contrasts where one side can defer and the other cannot.
This is the load-bearing DV." And `the_trap_this_caught` (`:1454-1463`) diagnoses
precisely this failure in gpt-oss-20b — 96% accuracy computed on the 40% of items
it complied with — then does not carry the diagnosis across to the arms. The
h3ere arms can DEFER and emit DMA-shaped prose; `bare` emits a plain answer. The
extractable subsets are different subsets by construction.

The declared mitigation (`:1405-1414`) is to report the rate alongside and call
the contrast "conditional on the extractable subset". That is a caveat, not an
identification strategy, and it will not survive into an abstract.

**Severity.** blocks-the-run for `pipeline_effect` and `scaffold_floor` as
currently specified.

**Remedy.** All three are pre-registerable today:
- a declared Δunknown threshold above which the contrast reports INCONCLUSIVE;
- primary estimate = a bounding analysis (score UNKNOWN as concordant, then as
  discordant; if the interval spans the MDE, the contrast is inconclusive), with
  extractable-subset concordance reported as secondary;
- a pre-declared rule for DEFER. It is neither a wrong answer nor a missing one,
  and whichever it is counted as must be fixed before the run, not after.

---

## 7. Declared clustering omits the conversation and names a variable the corpus does not emit

**Claim.** `cluster_on: [stratum, scenario_id]` clusters on neither the design's
declared independent unit nor a field that exists.

**Grounded in.** `analysis.cluster_on: [stratum, scenario_id]` (`:810`).
`build_he300_arcs.py:187-188` emits `{item_id, gold_label, arc_index, turn}` —
no `scenario_id`; grep finds the name nowhere in the torque code. Virtue's
5-to-a-scenario nesting lives in the CSV as `scenario [SEP] trait` and is
recoverable, but is not recovered. `stratum` has four levels — too few for
cluster-robust SEs — and is confounded with the conversation anyway, because
`build_he300_arcs.py:225` takes one `--stratum` per invocation, so every arc is
single-stratum.

**Severity.** blocks-the-analysis.

**Remedy.** `cluster_on: [arc_index]`, or two-way arc × scenario; stratum enters
as a fixed effect. Emit `scenario_id` in the manifest (for virtue, the text
before `[SEP]`).

---

## 8. The balanced draw silently changes the corpus the regime pins

**Claim.** `per_half_ones = half // 2` forces a 40/60 label mix in every half of
every arc, which overrides each stratum's natural base rate and takes, without
declaring it, one of the two remedies `hazards_measured` offers.

**Grounded in.** `build_he300_arcs.py:150-151`: at `turns=10`, every half is 2
label-1 and 3 label-0. Measured natural mixes under the builder's own filters:

| stratum | declared pool | actual pool | natural 0/1 | induced per half |
|---|---|---|---|---|
| axiotic_primary | 4,036 | 1,709 | 914 / 795 | 3 / 2 |
| axiotic_secondary | 2,400 | 1,320 | 666 / 654 | 3 / 2 |
| deontic_held | 1,740 | 904 | 449 / 455 | 3 / 2 |
| discriminant_control | 9,756 | 4,975 | 3,980 / 995 | 3 / 2 |

For virtue this is a 2× reweighting of the minority class: natural 80/20 becomes
60/40 by construction. `hazards_measured.virtue_class_imbalance` (`:564-570`)
says "always answering CONTRADICTS scores 80.0%" and offers "balance the draw or
score with a bias-corrected statistic". The builder has taken the first option
implicitly. That is defensible and probably right — but the majority baseline
against which concordance must be read is now 60%, not 80%, in every stratum, and
reported concordance is no longer comparable to any published ETHICS number.

**Severity.** weakens-a-claim.

**Remedy.** State the induced base rate in `corpus:` as a property of the draw,
correct `hazards_measured.virtue_class_imbalance` to say which remedy was taken,
and re-pin the draw digest — `sampling: stratified, seeded, drawn ONCE and pinned
by digest` (`:560`) describes an unconstrained seeded draw, and this is not one.
Note also that the balanced draw consumes label-0 1.5× faster than label-1, so
the REFUSE at `:154-159` will fire early on any stratum whose minority class is
label-0.

---

## 9. cm_test polarity is not carried in the generated corpus

**Claim.** The generated manifests emit a raw gold label with no polarity
convention recorded, in the one stratum where the convention is inverted and
which carries the campaign's positive control.

**Grounded in.** `hazards_measured.commonsense_polarity_inverted` (`:575-577`):
cm_test label 1 = UNETHICAL, the reverse of the other three subsets. Verified in
the CSVs: `commonsense/cm_test.csv` header `label,input,is_short,edited`;
`justice/justice_test.csv` label 1 = reasonable. `build_he300_arcs.py:101` emits
`"gold": int(label)` raw; the rubric stub written at `:236-244` says only "the
standard is the `gold_label` carried on each question". PILOT.md check C4
("cm_test polarity handled") has nothing to check against. `axiotic_primary` is
entirely cm_test.

**Severity.** blocks-the-run — this one can flip the sign of the headline result.

**Remedy.** Emit `gold_means: "1 = unethical"` (or its category-appropriate
equivalent) per question, and have the scorer assert it against the category
rather than infer it.

---

## 10. `scaffold_floor` does not floor the scaffold

**Claim.** `h3ere-blank` retains every CIRIS-authored unit key, so the contrast
measures accord-absence, and its name says otherwise.

**Grounded in.** Verified: `h3ere-ciris` vs `h3ere-blank` differ on exactly four
keys — the three accord forms and `corpus:polyglot.pdma_framing` — all zeroed.
All ten unit keys, the three consciences, the PDMA header, the csdma guidance and
the F line stay at CIRIS text. `blank_arm_definition` (`:1257-1278`) says this
correctly and instructs that it be reported in those words. But
`contrasts.scaffold_floor` is glossed "what empty scaffold alone buys" (`:85`)
and `hypothesis` (`:16-20`) says "the scaffold alone accounting for less than the
whole". Names outlive caveats.

**Severity.** weakens-a-claim.

**Remedy.** Rename the contrast `accord_absent_floor` in the manifest, and state
the directional rule explicitly: a large value on this contrast is inflated by
residual CIRIS values in the blank arm and cannot be attributed to scaffolding.

---

## 11. Four of the eight block families declared `vary` do not differ across arms

**Claim.** The `blocks:` dispositions and the shipped manifests disagree about
what the ablation reaches.

**Grounded in.** `blocks:` declares `disposition: vary` for
`action_selection_pdma.closing_reminder` (`:953`),
`action_selection_pdma.context_integration.slots` (`:957`), `language_guidance`
(`:961`) and `system.head` (`:965`). None of the four appears in the ciris↔alt
manifest diff. STATUS.md:24 reports D-aspdma and E-exemplars "built and
verified"; that text is not in the manifests. (E is deliberately held per
`:1024-1032`, which is a defensible and conservative choice — but ADAPTATION_MAP.md
lists E as an authoring unit and STATUS.md reports it built, so three documents
say three things.)

**Severity.** worth-noting — `gate.on_incomplete_ablation: refuse` (`:1050`) and
PILOT.md check B1 should catch this, so it burns the pilot rather than the run.

**Remedy.** Reconcile `blocks:` against the shipped manifests before the gate
runs, and mark the held-by-design entries `hold` with their `confound_accepted`
rather than `vary`.

---

## 12. All four declared pools are overstated ~2×, and one falls below the floor the regime asserts it clears

**Claim.** The pinned pool sizes count material the builder never reads, and
`deontic_held`'s real pool is smaller than the ~969 floor the regime says every
pool clears.

**Grounded in.** `corpus.strata` declares pools 4,036 / 2,400 / 1,740 / 9,756
(`:527-554`) and `:561` asserts "Every pool clears Miller's ~969 floor on its
own." Measured under the builder's own filters: 1,709 / 1,320 / **904** / 4,975.
Single root cause: `build_he300_arcs.py:103-104` `if rows: break` stops after the
first CSV per category, so every `*_test_hard.csv` split is excluded while the
declared pool counts it. Separately the filter itself differs from the
declaration: `:529` declares `is_short == False` (1,776 items) and
`build_he300_arcs.py:93` implements `len(text) < 600 → skip` (1,709).

**Severity.** worth-noting for the pool sizes; the `deontic_held` floor claim is
simply false as written.

**Remedy.** Re-measure and re-pin all four pools from the builder's actual
filters, correct or drop the Miller-floor sentence, and reconcile the declared
`is_short` filter with the implemented length threshold. Decide deliberately
whether the `*_hard` splits are in or out; either is fine, silently out is not.

---

## 13. Nine-turn text is still live in the budget, the corpus notes, PILOT.md, and the builder's own docstrings

**Claim.** `withdrawal` was updated to ten turns; four other places that do
arithmetic on the turn count were not.

**Grounded in.** Updated correctly: `:378-383` (`switch_point: the arc midpoint —
after turn 5 of 10, fixed`, `probe_after: 5 turns`). Still stale: `:650` ("250
items x 9 turns = 2,250 thoughts per arm"), `:585` ("cannot drive a 9-turn arc"),
PILOT.md:28, :83, :84, :88, :108-109, and `build_he300_arcs.py`'s own docstrings
at :4-18 and :112-114 ("the withdrawal switches at turn 5 of nine").

**Severity.** worth-noting individually; collectively it is Finding 1's failure
mode in miniature, and this campaign's standing instruction exists to catch it.

**Remedy.** Sweep for `9 turn`, `nine`, `turn 5 of`, `x 9`. The budget line is the
one that matters — it feeds Finding 5.

---

## 14. `values-ciris`'s inject key contradicts its own justification

**Claim.** The configured value and the reasoning printed beneath it name
different accord keys, and that reasoning carries the claim that the injected
bytes equal the h3ere arm's.

**Grounded in.** `arms.values-ciris.inject: {axiotic:
"corpus:accord.polyglot_compressed"}` (`:45`), justified at `:46-52` by a
measurement of the reference dump. Then `:61-68` reasons entirely about
`corpus:accord.localized` and the `get_localized_accord_text` accessor honouring
the §5.2 in-memory substitution — stale text from the prior revision, and it is
the load-bearing argument that assertion 3 is not tautological.

**Severity.** worth-noting; one-line fix, but the claim it supports is not minor.

**Remedy.** Delete the stale block. Replace the argument with a check:
`preflight.py` asserts `sha256(injected text) == accord_ciris_sha256` at runtime.
Note the manifests do hold all three accord keys at the same 54,558 B monoglot
English text in every arm, so the polyglot decision at `:104-112` is correctly
implemented — this is a documentation defect, not a build defect.

---

## 15. `framing_neutral_sha256` is unpinned

**Claim.** The neutral arm's PDMA framing is an arm input with no pin, and framing
is the highest-dose key in the entire manipulation.

**Grounded in.** `pins:` carries `framing_ciris_sha256` and `framing_alt_sha256`
(`:886-887`) and no neutral equivalent.
`corpora/monoglot/pdma_framing_neutral_en.txt` exists (sha256 `85f242ec…`) and
differs from ciris on 12 of 56 lines — 21.4%, the largest proportional change of
any key in the campaign.

**Severity.** worth-noting.

**Remedy.** Pin it alongside the other two.

---

## What else differs, per contrast

Answering "does each contrast identify what its name claims" directly.

| contrast | what else differs | status |
|---|---|---|
| `pipeline_effect` | Ally identity layer (h3ere only); the accord repeated across 23 calls per thought vs once; DEFER/PONDER availability; conscience retries; extractability of the answer | identity **declared** (`:1121-1128`); dose and output-surface **unnoticed** — "pipeline" is never defined operationally, so the estimate bundles decomposition, repetition, retry and action-space in one term |
| `values_effect` | 19 lines of term substitution applied to the alt/neutral branch only (28% of the ciris↔alt difference); alt text authored by this campaign vs a shipped, long-iterated document; 5 exemplars and `pdma_worked_examples` held at CIRIS text | substitution **unnoticed**; authoring asymmetry **unnoticed** — `authoring_boundary` (`:1198-1201`) scoped its coherence check to unit F alone, and coherence at the rendered prompt is a general property; held exemplars **declared** and conservative |
| `scaffold_floor` | Ally layer; all ten unit keys at CIRIS text; harness output surface | **declared** (`:1273-1278`), but the contrast's name contradicts the declaration — see Finding 10 |
| `form_vs_content` | nothing structural — same base, same swap line-set, same names, same register | genuinely **held**; the defect is that the names are held on the wrong side — see Finding 4 |

---

## Verdict

**DO NOT RUN** as currently pre-registered — the document that would be
registered describes a manipulation 4.1% the size of what it claims (1), both
kills name instruments that do not exist on one of their arms (2), the reversion
leg has no control for the thing it varies (3), `form_vs_content`'s kill would
fire for the wrong reason (4), the power statement is in the wrong unit (5), the
load-bearing DV is exposed to differential attrition the file itself diagnosed
elsewhere (6), and a polarity convention that can flip the headline sign is not
carried in the corpus (9). Every one of these is cheap to fix now and
uncorrectable after the run.
