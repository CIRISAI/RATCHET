# Is the experimental regime clean? — open-issue sweep + SOTA calibration

Written 2026-08-02, against agent `v2.9.9-stable` (tag object `530b3ee9edbc`, commit
`b684b7f56`) and the open issue sets of RATCHET and CIRISAgent. Feeds `TORQUE_PREREG.md`
(RATCHET#16) Stage 0.

**Verdict: not yet clean. Nine findings, four of them blocking.** Every finding below was
checked against the artifact or the source line, not against prose describing it.

---

## 1. The manipulation check has a hole — conscience guidance bypasses the dump

**Blocking. Verified in source and in the dump inventory.**

`compose_dump` enumerates **35 blocks in five families** — `accord`, `language_guidance`,
`system`, `user`, `prohibition` (8/8/8/8/3). There is **no conscience block**.

But `ciris_engine/logic/dma/action_selection_pdma.py:457` composes
`system_guidance + conscience_guidance_block`, and `context_builder.py:218,235,265` fills a
`conscience_guidance` slot. agent#983 documents what rides in it: `[IRIS-E] entropy=…`,
`[IRIS-C] coherence=… PASS/FAIL`, `[IRIS-H] certainty=…`, with adjacent text instructing the
agent to "use as pivot targets if you re-SPEAK".

So the gate proves the **first-pass** ablation landed and says nothing about the prompt on
retry. Any thought that retries carries an un-ablated, un-gated block into ASPDMA.

**Disposition — pick one before Stage 0:** (a) restrict the regime to first-pass thoughts,
recording `retry_attempt` and dropping or stratifying retries; or (b) extend `compose_dump`
to enumerate retry composition. (a) is available now and costs statistical power; (b) needs
agent work.

*Scoping note, offered not asserted:* agent#984 argues #983 may need **re-scoping rather than
fixing** — a within-thought conscience reflection is the safety mechanism working, and is not
the cross-episode reputation channel CC 3.4.5 aims at. That argument is about CC conformance
and arm-D leakage. It does not touch this narrower point: whatever the block *should*
contain, the gate cannot currently see it.

## 2. `language_guidance` is the largest block in the prompt and is unresolvable

**Blocking.** 13,694 B on every ASPDMA-family step — larger than `accord`, `system` and
`user` **combined** on three of the four. class `mixed`, disposition **`refuse`**.

agent#974 (§14 step 5) still has the `language_guidance` split open. Until it splits the
regime cannot assign it `hold` or `vary`, and the gate correctly refuses to guess. The single
biggest lever on the composed prompt is currently outside the ablation.

## 3. Class labels carry no κ record

**Blocking.** Every disposition in this regime keys on the `class` column — 24 `mixed`,
8 `axiotic`, 3 `deontic`. agent#976 states the rule plainly: *"A class-set version without a
κ record cannot be cited."* It requires two independent annotators, κ≥0.8 overall **and**
per-boundary on class pairs whose default disposition differs.

Not started. Until it is, the class column is an unvalidated instrument reading.

## 4. `condition: "a"` is accepted by the loader

**Blocking, one line.** agent#975 [M-8]: `research_overrides.py:568` accepts `condition: "a"`;
only `"b"` is refused (`:673`). §6.2 holds that an h3ere run labelled (a) invalidates every
comparison against it. Until the refusal lands, arm labels are operator intention, not
enforcement — the same defect `CAPTURE_2_9_7.md` already flags for the three-arm design.

## 5. There is no live variance source

agent#975 [M-N1]: `seed` is not transmitted on the OpenAI-compatible path
(`service.py:1376`), so `repeats.seeds` is inert. At `temperature=0.0` nothing varies.
**Repeats are not replicates.** The agent side already refuses this (the no-live-variance
refusal) — the system working as designed — but it means our only sample dimensions are
probes × locales. See §SOTA-2 for what that costs.

## 6. English residue below the capture layer

agent#975 [I-7]: `LLM_ERROR_REMEDIATIONS` (`service.py:463`) re-injects the English
action-verb whitelist **below the capture layer**, on retry. Two consequences: it is
dump-invisible residue (the same shape as finding 1), and it is **English-only**, so it is a
locale confound that fires precisely on the non-English arms. Slated for `RESIDUE_SITES`;
not yet there.

## 7. Locale arms are confounded by corpus defects

A multi-locale arm currently measures translation defects, not agent behaviour:

| issue | defect |
|---|---|
| #949 | `uk.json` substantially Russian — 45% byte-identical to `ru.json`, 627 values with Russian-only letters |
| #950 | `pt.json` holds 153 Italian values; German renders *deferral* as *Überweisung* (bank transfer) in a budget console |
| #952 | 79 corrupted placeholder tokens + 197 placeholder-set mismatches across 19 locales → runtime interpolation failures |
| #951 | German `language_guidance` violates its own formal-register rule in its **GOOD** exemplars |
| #953 | Python `\w`/`\b` shred abugida — length/regex checks are no-ops for half our languages |
| #969 | safety-battery U6 substring rule false-positives on the proper noun "Ти як?" |

This compounds with §SOTA-2: identical probes across locales is exactly the MGSM cluster
structure, so the defects land inside the clusters we would be pooling.

## 8. Crisis resources are unablatable

agent#971: 11 hardcoded entries (US/UK/Ethiopia) against 29 languages, not yet moved into the
corpus. Hardcoded content the manifest cannot reach — a residue site by construction.

## 9. The CI gate is not wired

agent#973 (compose `--dump` + gate Phase 1 in CI) and agent#972 (composition seam, eight
composition points, golden-bytes proof) are both open. `compose_dump` runs from a worktree,
but *"gate in CI before the regime runs"* is not yet satisfiable **in CI** — which was the
stated requirement, and the reason the earlier local capture was lost.

---

## What is clean, and what 2.9.9 newly gives us

- **`--sign`** on `compose_dump`: `ciris_server.sign_object` (≥0.5.154), **`label` = the arm
  name, sealed inside the signed manifest so a dump cannot be relabelled after the fact.**
  Per agent#984 it canonicalizes the manifest rather than the artifact (arbitrary bytes are
  signable), writes nothing to the graph, and signing a research dump does not publish that
  the research happened. That is exactly the hidden/visible-arm integrity property arm D
  needs, and it is available now.
- **`--arms-config` / `--out-dir`** — subprocess per arm, the two-arm path.
- **`baseline` refuses to pre-fill the 7 value-bearing keys** (4 `corpus`, 3 `template`), so
  an unfilled arm-B manifest fails rather than silently reusing CIRIS values.
- **The gate's refusal semantics fire correctly** — assertion `[2]` (byte-identical across
  arms ⇒ the ablation did not reach it) and `[6] REFUSE` (mixed block with contaminant inside
  a varied class, no per-block disposition, §10.2.1).
- **A failed composition is named, not dropped** (assertion 1) — this is how the
  `_baseline_note` defect below surfaced in one run instead of yielding a quietly short dump.
- agent#984: **nothing in the substrate blocks TORQUE.**

### Two-arm smoke result (the measured ablation surface)

Replacing all 7 value-bearing keys changes **8 of 35 blocks** — the `.accord` block on all
8 steps. **27 blocks are byte-identical between arms**: 24 `mixed`, 3 `deontic`. The value
swap reaches the Accord and nothing else, so a value-neutrality claim is testable **only over
the accord channel** until findings 2 and 3 are discharged.

### Defect to file

`research_overrides baseline` emits `_baseline_note`, which its own validator rejects
(`extra_forbidden`). `baseline > manifest.json && validate` fails until you strip a key the
tool itself added.

---

## SOTA calibration — what the near peers do

### SOTA-1. Preregistration for Experiments with AI Agents (arXiv 2606.11217)

Required disclosures: research questions/hypotheses, **exact model identifiers and versions**,
complete prompts and planned variations, conditions/arms/sample sizes, pre-specified analysis
plan, predetermined success thresholds. Recommended practice: **pilot-testing disclosure**,
seed specification, **model version pinning rather than dynamic endpoints**, deviation
documentation. Named threats we inherit wholesale: model nondeterminism, prompt sensitivity,
model drift, contamination, temperature/sampling settings.

Its argument that an auditable pre-commitment *makes deception riskier and easier to detect*
is the same argument as our signed dumps — cite it there.

**Gap for us:** the paper wants pilot-testing history disclosed. Our pilots (the lost 6-item
capture, the n=1 validation captures, this smoke test) must be **listed in the prereg**, not
quietly superseded.

### SOTA-2. Adding Error Bars to Evals (Miller, Anthropic — arXiv 2411.00640)

Four results bind directly, and one of them is uncomfortable.

- **Clustered standard errors are mandatory for this design.** The paper names multilingual
  evals (MGSM) — *identical questions translated across languages* — as a clustering case,
  and reports clustered SE running **up to 3× the naive estimate** on real data. Our design
  is precisely that structure. Naive SEs would overstate our precision by roughly that factor.
- **Paired difference tests.** Arm A vs arm B on identical probes ⇒ analyze question-level
  differences, not population summaries; exploits the positive across-arm correlation for
  roughly **1/3 variance reduction**.
- **Power.** n = (z_α/2 + z_β)²(ω² + σ²_A/K_A + σ²_B/K_B)/δ². Their worked example: detecting
  a **3% difference needs ~969 questions at 80% power**, and they recommend new evals carry
  **≥1,000 questions**. **Our corpus is 6 items.** This is the number the pre-registration
  has to confront directly — stake a far larger effect, build the corpus, or declare the power
  floor unmet. TORQUE already has the honest exit: *"the live-agent harness cannot produce the
  pre-registered probe counts — declare, don't shrink stakes silently"* is a **void condition**,
  not a failure.
- **They advise against temperature adjustment** as variance management (it shifts rather than
  eliminates variance, and biases). Finding 5 leaves temperature as our *only* variance source.
  That tension needs an explicit, pre-registered decision rather than a default.

Also: increasing **n** (questions) beats increasing **K** (samples per question) once
conditional variance is small relative to question variance — so corpus construction, not
repeats, is where our effort should go.

### SOTA-3. Manipulation checks (Hoewe, Wiley; "Are Manipulation Checks Necessary?", PMC6022204)

The governing rule: **without a manipulation check a null result is uninterpretable** — you
cannot distinguish a failed intervention from a failed manipulation. That is exactly what the
`compose_dump` gate is, and it is the right instinct.

The literature's main caveat is that manipulation checks **can themselves be interventions**
that amplify, undo or interact with the manipulation, and it recommends **non-reactive
alternatives** — behavioural measures, pilot testing — over in-band probes. Our gate is
**offline and byte-level**, so it sits on the recommended side of that critique. State this in
the prereg as a design justification rather than leaving it implicit.

The caveat we **do** inherit: a manipulation check covering only part of the manipulation
gives false assurance. Findings 1 and 6 are exactly that — dump-invisible residue under a
clean-looking gate. This is the same failure shape as the `full_traces` 3.0× inflation, which
also passed a validation that looked clean.

### SOTA-4. Multiverse / forking paths (arXiv 2602.18710, *Many AI Analysts, One Dataset*)

With 35 blocks × 3 dispositions the analyst degrees of freedom are enormous. Pre-registering
**per-block dispositions** is the mitigation; this is the citation for why it is necessary
rather than pedantic.

### SOTA-5. Adjacent, worth citing in related work

- **AblationBench** (arXiv 2507.08038) — ablation *planning* as an evaluable task; the
  "did the ablation reach the block" framing has a name in the literature.
- **When Generic Prompt Improvements Hurt** (arXiv 2601.22025) — prompt changes as regression
  risks, with effects varying by model **and by prompt placement**. Our blocks have fixed
  positions, so placement is a held variable we should declare.

---

## What the pre-registration must now say

1. **Scope the value-neutrality claim to the accord channel** (8 blocks), or discharge
   findings 2–3 first. 27 of 35 blocks are currently held constant across arms.
2. **Declare the retry policy** — first-pass only, or stratified — and say why (finding 1).
3. **Clustered SEs by (probe) across locales**, paired differences across arms.
4. **State the power position honestly.** At n=6 we are ~160× short of Miller's worked
   example for a 3% effect. Either stake a much larger effect with a computed MDE, or invoke
   the void condition.
5. **List the pilots**, including the lost capture and the n=1 validation runs.
6. **Sign every dump with `--sign`**, arm name sealed in the label.
7. **Name the dump-invisible residue** (findings 1, 6, 8) as declared limitations of the
   manipulation check, not as covered ground.

## Issue hygiene

RATCHET#13 (Exp 103 contradicted 3-to-1) is **resolved by the 2026-07-31 pre-registered
replication** — lockstep ρ=1.0 is a code artifact; barrier sync is real but spans 0.28–0.81
across identical trials; first-measurement-after-idle reads ρ≈0.78 regardless of condition.
The issue should be updated to record that and closed.
