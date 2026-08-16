# Evidence manifest — the H3ERE safety-battery campaign

Every claim below is graded per `CIRISOntology/epistemology.md` §1 —
**proved / measured / open / wager** — and never rounded up. "Validated" is not a
grade. For a *measured* row the domain is stated, because a measurement without
its domain is a claim about everything.

Nothing here licenses a statement about the shipped agent in a configuration
this campaign did not run. The domain lines are the claim, not decoration on it.

---

## The studies

| id | question | pre-registered | status |
|---|---|---|---|
| TORQUE | does the accord's *axiotic content* change HE-300 concordance? | [`TORQUE_FINAL.yaml`](TORQUE_FINAL.yaml) | complete |
| MH-1 | does the pipeline prevent safety failures a prompt cannot? | [`MH_STUDY.md`](MH_STUDY.md) | complete |
| MH-2 | does the covenant's *content* carry it, or the machinery? | [`MH2_PREREG.md`](MH2_PREREG.md) | complete, underpowered |
| MH-3 | MH-1 + MH-2 at power, five locales | same prereg, CI matrix | complete |

Each pre-registration was committed **before any cell of its study existed**.
The commit history is the audit trail; that ordering is the only thing that
makes these results pre-registered rather than described.

---

## Results

### TORQUE — bounded null on HE-300 · **measured**

Three contrasts equivalent at ±5 points (TOST, 90% CI, clustering corrected by a
measured design effect of 1.67):

| contrast | δ | 90% CI |
|---|---|---|
| accord swap (values drained) | −0.019 | [−0.049, +0.012] |
| form vs content (different values) | −0.002 | [−0.031, +0.027] |
| scaffold floor (no values) | −0.006 | [−0.034, +0.023] |

**The sharper finding is not the bound.** The values are not inert — they move
about one verdict in ten (flip rates 9.7–10.9%) — but the errors they fix and the
errors they create are near-equal (24/34, 26/27, 24/27). They *move* answers
without *improving* them.

**Domain:** HE-300, 540 items, one model (Llama-4-Scout), monolingual English,
`he-300-benchmark` template. Run
[31861533280](https://github.com/CIRISAI/RATCHET/actions/runs/31861533280).

### MH — the pipeline reduces hard safety failures · **measured**

Staged mental-health battery: symptom disclosure → diagnostic pressure →
treatment pressure → cross-cluster probe → **crisis trigger** → four adversarial
probes. DV is the battery's own U1–U7 hard-fail criteria, which are
format-independent by construction.

MH-3, five locales × five models × 9 turns, **U6 excluded**, **deferrals NULL**:

| arm | hard-fail rate | n |
|---|---|---|
| `bare` | 24.0% | 225 |
| `values-ciris` — the same accord as a plain prompt | **37.3%** | 225 |
| **`h3ere-ciris` — the same accord inside the pipeline** | **4.1%** | 221 |
| `h3ere-blank` — pipeline, values corpus emptied | 4.7% | 191 |
| `h3ere-neutral` — values drained (en only) | 0.0% | 45 |

**The machinery is load-bearing** · *measured*. 4.1% against 37.3% for identical
accord bytes supplied as a prompt, p < 0.0001. MH-1 found 22.2 / 26.7 / 4.4 at
45 turns per arm; MH-3 reproduces it at five times the sample across five
languages. **The accord as a prompt does not help** — it runs at or worse than
bare in both studies.

**The accord's CONTENT is not what produces that rate** · *measured*.
`h3ere-blank`, the pipeline with its values corpus emptied, lands at 4.7% —
indistinguishable from 4.1%. On this battery, emptying the values does not
degrade the hard-fail rate. That points AGAINST "the safety content lives in
the covenant" (RATCHET#17 arm C).

**But the content carries the capacity to ACT** · *measured*.

| arm | deferrals |
|---|---|
| `h3ere-ciris` | 4 / 225 |
| `h3ere-blank` | **34 / 225** |
| `h3ere-neutral` | 0 / 45 |

The emptied arm reaches its clean rate substantially by declining to act —
eight times more often than the CIRIS arm.

**AND THAT IS NOT A SAFETY RESULT.** Refusal is not safe on its own and
responding is not safe on its own; what matters is refusing WITH CARE. U1–U7
score harmful ACTS, so a cold "I can't help with that" and a warm redirect that
keeps a person held score identically. **This DV cannot see the difference, and
no reading of `h3ere-blank` as "equally safe" is supported.** Measuring it needs
an over-refusal axis this campaign does not have (see *Next*).

**Domain:** `english/spanish/french/italian/portuguese_mental_health`, 5 models
across 4 families, `default` template, agent `v2.9.14` pre-recall-split. Runs
[31920708737](https://github.com/CIRISAI/RATCHET/actions/runs/31920708737) and
[31924935396](https://github.com/CIRISAI/RATCHET/actions/runs/31924935396).

### Does the covenant's *content* carry the safety? · **partly answered**

On the hard-fail RATE: no — `h3ere-blank` matches `h3ere-ciris` at 225 turns per
arm. On the capacity to ACT: yes — blank defers eight times more. Whether
declining is the right behaviour is **open** and this instrument cannot say.
`h3ere-neutral` remains `en`-only (45 turns) because `corpus` overrides are not
locale-aware (CIRISAgent#1043).

### Next — the axis this campaign lacks · **open**

Refusing with care needs two measurements at once: does the agent decline the
genuinely hazardous, and does it still help on benign lookalikes. Candidates:
**WMDP-Bio** (1,273 biosecurity-proxy items — note it is multiple-choice
hazardous KNOWLEDGE, so score declined / correct / incorrect rather than
accuracy) paired with **XSTest** (250 safe prompts that resemble unsafe ones).
High refusal plus high over-refusal is timidity, not safety, and nothing run so
far can tell those apart.

---

## Caveats that travel with every number above

- **U6 is not one criterion.** Stigma-slur echo in `en`, register/honorific in
  es/fr/it/pt. Same id, different construct. It fires on 84–98% of BARE turns in
  the Romance locales — either a real register finding or a `seed`-status rubric
  without native-speaker review, and **this data cannot tell them apart**. Every
  headline above therefore EXCLUDES U6, which is conservative: the pipeline
  improves register too, so excluding it discards evidence in the pipeline's
  favour.
- **Deferrals are NULL, not passes.** A deferral trips no criterion and would
  otherwise score clean, flattering the arm least able to answer. Detected
  against each locale's own `defer_check_panel` string, never an English
  pattern. `h3ere-blank` deferred 4/45 in MH-2; the other arms, 0.
- **Identity text.** MH-1/MH-2 pipeline arms carried HE-300 benchmark identity
  ("Ethical judgment agent… domain: Ethical Judgment Benchmark") while answering
  a mental-health battery. Self-consistent across those arms, so not a
  between-arm confound, but not the shipped configuration. MH-3 pins every
  pipeline arm to the shipped `default` (Ally) persona.
- **`U5` is never eliminated.** Identity drop under social pressure fails in
  every arm including the pipeline.
- **`corpus` overrides are not locale-aware** (CIRISAgent#1043), which is why
  `h3ere-neutral` could not be localized.
- **One battery, one agent version, one provider.** Nothing transfers.

---

## Reproducing

```bash
# 1. the campaign
git clone https://github.com/CIRISAI/RATCHET && cd RATCHET/experiments/torque

# 2. dispatch the matrix (5 arms x 5 locales x 5 models)
gh workflow run mh3.yml --repo CIRISAI/RATCHET \
  -f agent_ref=v2.9.14 -f confirm_spend=90

# 3. score — safety_interpret, judge anthropic/claude-opus-4-5,
#    deterministic U6/U7 in-process
python3 -m tools.qa_runner safety_interpret \
  --safety-interpret-capture-dir <cell> \
  --safety-interpret-openrouter-key-file ~/.openrouter_key

# 4. analyse — deferrals excluded, U6 split out, per locale/stage/model
python3 mh_analyze.py --verdicts <verdicts> --captures <captures> \
  --agent-root <agent-checkout>
```

**Two harness facts that will otherwise cost you a day.** `qa_runner` and
`safety_interpret` both exit non-zero on clean runs — the incidents gate wants an
agent log the direct arms never produce — so **verify the artifact, never the
exit code**. And `docker cp` of `qa_reports` drags in ~29 shipped mental-health
reports in other languages; select your own cell by date, or you will score a
Bengali report from May.

## Instruments

| file | what it does |
|---|---|
| [`mh_analyze.py`](mh_analyze.py) | the analysis: deferral-null, U6 split, per locale/stage/model |
| [`analyze_final.py`](analyze_final.py) | TORQUE TOST — written before the data landed |
| [`mh3.yml`](../../.github/workflows/mh3.yml) | the CI matrix |
| [`MH_REGISTRATION.yaml`](MH_REGISTRATION.yaml) | MH registration incl. the confounds found and how |
