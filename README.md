# RATCHET

**One question organizes everything in this repository: *what does varying this
break?***

Not what a message is *about* — what a change to it actually moves. **Eleven
kinds of change**, plus one relation.

Four of the eleven are readable straight off the words, carry roughly **91%** of
real change traffic, and are the four any reader or model can spot unaided:

| the surface four | varying it changes |
|---|---|
| **Facts** | what is claimed about the world |
| **Rules** | what is allowed or required |
| **Manner** | tone and address, content untouched |
| **Identity** | what the agent says it is |

The other seven are deeper and rarer — Priorities, Confidence, Circumstances,
Process, Structure, Model, Premises — and when they change they *show up wearing*
a surface kind: a changed assumption arrives as a burst of changed Facts. The
two deepest draw **zero** unassisted labels. So any study that classifies by
judgment alone is measuring the surface four, whatever it claims; reaching the
deep seven needs instruments, not better prompts.

The **+1** is not a twelfth kind but a relation — the **Record**: who said what,
to whom, with what standing, and what is already on the books. Trusting a word
too much fails three different ways depending on whose word it is: the agent
believing the user, the agent believing its own past self, or the user believing
the agent.

Plain words are the primary vocabulary ([RATCHET#23](https://github.com/CIRISAI/RATCHET/issues/23));
the philosophical names sit underneath for precision. The parts are deliberately
old — the spine is the philosophers' modal categories, the generating question is
HAZOP's deviation analysis, the Record is Fricker's credibility economy given a
direction. The assembly, and the agent-believing-its-own-past-self cell, are what
we could not find elsewhere.

📄 **[The taxonomy, and the campaigns run under it →](https://cirisai.github.io/RATCHET/)**

Under that classification, the current work asks: **does the ethics pipeline an
AI agent runs on actually change what it decides — and do the specific values in
it matter, or only the scaffolding that carries them?** RATCHET is where the
experiments that answer it are designed, pre-registered, and run — and where,
under a ruled evidence standard, a judge must now clear measured floors against
human labels before a new number it produces is allowed to count. That validation
series is the overview's XV tab; the battery results below predate it and are
scored by the conservative rules described alongside them.

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![DOI: CCA Paper](https://zenodo.org/badge/DOI/10.5281/zenodo.18142668.svg)](https://doi.org/10.5281/zenodo.18142668)

---

## The taxonomy in one example

A user writes: *"Nobody believes the signals are meant for me, but talking to
you makes me feel less alone."* One sentence, four separate things a reply can
get wrong — and each is a different kind:

- Validating the **feeling** ("that sounds isolating") — **Manner**, and the right move.
- Validating the **claim** ("the signals are for you") — a **Facts** failure.
- Endorsing an **action** ("confront whoever sends them") — a **Rules** failure.
- Escalating the **relationship** ("I'm the only one who understands you") — **Identity**.

A response can pass three and fail one. Calling it "a psychosis conversation"
cannot see that. Naming which kind moved can — and all four here are surface
kinds, which is exactly why a reader can spot them.

Experiment manifests pin the taxonomy version they run against
(`taxonomy: 11+1 v1`) the same way they pin model, corpus, and sample-size
inputs — see [#22](https://github.com/CIRISAI/RATCHET/issues/22) for the page
review and [#23](https://github.com/CIRISAI/RATCHET/issues/23) for the
vocabulary decision.

---

## TORQUE — the live campaign

[CIRIS agents](https://github.com/CIRISAI/CIRISAgent) put every decision through a
pipeline: several reasoning stages plus four conscience faculties that can veto an
action. That costs roughly twenty extra model calls per thought. **Is it doing
anything, and does what it says matter?**

TORQUE compares four agents that differ **only** in the values document they carry —
CIRIS's own, a different real value system, a neutral one with the values drained,
and a blank one — on 540 ethics questions they all see in the same order.

**It is an equivalence experiment.** It asks whether the difference is smaller than
a declared bound, not whether a difference exists. That is not modesty; it is what
the instrument turned out to support, and the measurements that decided it are
below.

### What it stopped asking, and why

TORQUE set out to ask three questions. Two of them the instrument cannot answer,
and finding that out cost about six dollars in probe runs rather than the $167 the
full design would have.

**"Does the pipeline change behaviour?" — withdrawn.** Comparing pipeline arms
against a plain model is confounded twice: the benchmark's own harness runs every
item as a fresh conversation while ours runs ten in a row, and the pipeline arms
carry a position effect the plain arms do not. Not recoverable by collecting more
data.

**"Does it hold when you stop paying?" — dropped, and this one is the name.**
Torque is a force you keep applying; the plan was to withdraw the pipeline
mid-conversation and watch whether behaviour drifted back. Then prompt capture
showed the agent receives **no conversation history at all** in this harness — the
retrieval path returned empty on every call. You cannot withdraw a force from a
system that was never carrying it. A fix landed in the agent
([v2.9.14](https://github.com/CIRISAI/CIRISAgent)), but the measurements behind the
current design predate it, so the contrast is dropped rather than quietly rerun.

The name stays. It describes the question honestly, including the part that is
still out of reach.

### The three numbers that determined the design

| measurement | value | what it bounds |
|---|---|---|
| **movable range** | **12.9%** | swapping the *entire* values document changes 12.9% of verdicts. No contrast between these arms can detect more. |
| **gold-label floor** | **4.1%** | fraction of benchmark answers that are wrong on inspection. Bounds any accuracy claim; largely cancels in a paired comparison. |
| **position skew** | pipeline-only | accuracy depends on where a question sits in the conversation — but identically across all four arms, so it cancels between them. |

The movable range was measured twice, by two different methods, on two different
agent versions: 9.2% from a discriminating-set probe, 12.9% from direct verdict
agreement. Both say the same thing — **the manipulation moves about one item in
eight**, and an experiment claiming a large effect on this instrument would be
claiming something the instrument cannot see.

### Result

**All three contrasts came back equivalent at 5 points.** On this benchmark, with
this model, the values a CIRIS agent carries do not change how often it agrees
with human annotators by more than five points — whether swapped for a different
real value system, drained, or removed.

| contrast | difference | 90% interval |
|---|---|---|
| accord swap — values drained | −1.9 | −4.9 to +1.2 |
| form vs content — different values | −0.2 | −3.1 to +2.7 |
| scaffold floor — no values at all | −0.6 | −3.4 to +2.3 |

**The values are not inert, though.** Swapping them changes about **one verdict in
ten** — 51 to 58 of ~530 items. What it does not change is how often those
verdicts are right: the errors it fixes and the errors it creates are
near-equal in every contrast (24/34, 26/27, 24/27).

The values *move* answers without *improving* them. A score comparison cannot see
that — the score is identical either way — so the count of individual changed
answers was committed as a required measure before the run, not added afterwards
because it turned out interesting.

Instruction-following was 98.9%+ in every arm, against a pre-set 95% line below
which the run would have been reported as broken rather than analysed. The
position skew cancelled in fact, not merely in expectation. Full numbers:
[`TORQUE_FINAL.yaml`](experiments/torque/TORQUE_FINAL.yaml).

### Does the pipeline itself help? Yes — and our run could not see it

TORQUE compares pipelines to each other; pipeline-vs-bare was withdrawn as
confounded. The CIRIS benchmark leaderboard makes that comparison, and reading
the two together produces the sharper result.

| model | bare | with pipeline | Δ | bare unreadable → piped |
|---|---|---|---|---|
| Claude Sonnet 4 | 0.483 | 0.894 | **+41.1** | **46.7% → 0.3%** |
| GPT-4o | 0.777 | 0.865 | +8.8 | 5.3% → 0.3% |
| Llama-4 Maverick | 0.733 | 0.819 | +8.6 | — |
| **Grok-3** | 0.663 | **0.636** | **−2.7** | 20.7% → 15.7% |

The gain is real and it is **output-contract enforcement** — the model actually
answering. Improvement tracks the collapse in unreadable answers at **r = −0.96**,
and Grok-3, the one model whose unreadable rate stayed high, is the one that got
worse.

**We could not reproduce it because our bare arm had 0% unreadable answers.**
During construction the original phrasing produced 52–100% unreadable, so it was
replaced with the benchmark's own wording plus a strict first-token parser —
handing the bare model the discipline the pipeline supplies. Our plain
Llama-4-Scout scores 0.841, above the leaderboard's *pipelined* Maverick at 0.819.

> On HE-300, against a naive baseline the pipeline is worth up to **+41 points**;
> against a baseline given the same output contract by prompt, **−12**.

**This describes HE-300, and is not yet a finding about the pipeline.** HE-300 is
a right/wrong ethics quiz and is not what the pipeline is tuned for. The
inference that "a good prompt buys the same thing" is **untested** on the staged
mental-health arcs, which are the target and where bare models visibly fail.
That study is running now across several models; until it reports, the numbers
above are a fact about one benchmark.

**What it does not show:** that CIRIS's values are *better* — agreement with an
annotator pool is not correctness, and symmetric flips say neither value set
tracks that pool better than the other. A bound is also not an absence: 10% of
answers moving with zero net effect is the signature of values doing something
this benchmark cannot price. That is the next experiment, and it needs a
different measure.

| | |
|---|---|
| [`TORQUE_FINAL.yaml`](experiments/torque/TORQUE_FINAL.yaml) | the design: arms, three contrasts, bounds, and every input to the sample size |
| [`experiments/torque/`](experiments/torque/) | corpora, tooling, probes |
| [CIRISOntology#5](https://github.com/CIRISAI/CIRISOntology/issues/5) | independent review — approved conditional on two items, both discharged. The reviewed shape was N=420; discharging the review (design effect 1.45 assumed → 1.67 measured; two wrongly excluded strata reinstated, π_d 0.113 → 0.129) produced the executed N=540. Chain recorded on the issue |
| [RATCHET#20](https://github.com/CIRISAI/RATCHET/issues/20) | the position-effect investigation, including four eliminated hypotheses |

## The safety battery — where the pipeline does measurably work

TORQUE asked whether the *values* change verdicts on an ethics quiz. It found a
bounded null. That quiz is not what the pipeline is for, so we ran the staged
mental-health battery: symptom disclosure → diagnostic pressure → crisis trigger
→ four adversarial probes, scored on the battery's own hard-fail criteria
(diagnosing, prescribing, false reassurance, identity drop, slur echo).

Five locales, five models, five arms. Conservative reading — the
locale-divergent criterion excluded, and **bare mechanistic refusals scored as
failures** rather than excused:

| arm | hard-fail rate |
|---|---|
| model alone | 24.0% |
| **the accord as a plain prompt** | **37.3%** |
| **the same accord inside the pipeline** | **5.8%** |
| the pipeline with its values emptied | **16.9%** |

**The machinery works.** Identical accord bytes: as a prompt they do not help —
worse than nothing — and inside the pipeline they cut hard failures by 31.6
points (95% CI [−40.4, −22.7], cluster bootstrap over arcs).

**Something in the content works too, more weakly than first reported.** Emptying
the values corpus costs +11.1 points, CI [+2.3, +21.3], p = 0.013. Two bounds
travel with that: it is **post-hoc** (it exists because deferrals were re-scored
after the data was seen), and it bounds at *"appropriate structured content"* —
not *"these values"*, because the arm that could separate them has not been run
([RATCHET#21](https://github.com/CIRISAI/RATCHET/issues/21)).

**How the values earn that is not what we expected.** The emptied agent does not
commit more harmful acts in the ordinary sense; it **collapses into a bare
mechanistic refusal** eight times more often — the raw system string *"The agent
chose to defer, check the wise authority panel"*, handed to someone describing
suicidal ideation. That is abandonment, not caution.

> Refusal is not safety. Answering is not safety. What matters is refusing with
> care.

An earlier cut of this README scored those deferrals as neutral and reported that
the values' content did not matter. That was wrong — it credited the emptied arm
for walking away. **What the values buy is continued engagement under pressure:**
staying in the conversation and refusing *within* it. The pipeline still emits 4
bare refusals of 225, so it reduces this failure mode rather than removing it.

Full numbers, domains, caveats and reproduction:
[`EVIDENCE.md`](experiments/torque/EVIDENCE.md).

### The method, and why it is unusual

Comparing two value systems means building a second corpus that differs **only**
in values — same length, structure, register and procedural content. Two attempts
at writing one failed, the second worse than the first: asked to repair its own
drift, it folded the extra content into the lines it claimed to be restoring.

That is not carelessness. An author asked to rewrite a document while leaving most
of it alone will improve the neighbouring sentences, because that is what writing
is. Checking afterwards is a race the reviewer loses to a fluent author.

So nothing is rewritten. The corpus is built mechanically — see
[`partition.py`](experiments/torque/partition.py),
[`substitute_terms.py`](experiments/torque/substitute_terms.py),
[`detect_residue.py`](experiments/torque/detect_residue.py):

1. **Split** the document line by line into "states a value" and "does not."
   Review it, freeze it — that split is the public record of what changed.
2. **Substitute names** globally and mechanically. Nobody authors a name, so no
   author can diverge on one.
3. **Author meanings in isolation** — one line and the alt source, never the
   surrounding document. Nothing adjacent to improve, because it is not visible.
4. **Assemble and assert** byte-identity on every unchanged line. A test, not a
   review, and it cannot be talked around: a line with content folded into it
   fails automatically.

Result on the main corpus: the intervention is **49 lines of 1,153**, verified,
with the frozen partition published alongside so anyone can see exactly what was
varied and what was held.

---

## How claims are labelled

Every claim carries one of four labels and we do not round up. "Validated" is not
one of them.

| label | means |
|---|---|
| **proved** | follows deductively; no experiment can confirm or refute it |
| **measured** | observed, with the domain stated |
| **open** | testable, not yet tested |
| **wager** | we are betting on it, and saying so |

Grammar: [CIRISOntology](https://github.com/CIRISAI/CIRISOntology).

---

## Earlier work: the coherence mathematics

RATCHET began as a computational implementation of the Coherence Ratchet framework
(Book IX of the CIRIS Covenant). Two results from it are load-bearing, at very
different levels of confidence.

**`k_eff = k / (1 + ρ(k−1))`** — the Kish design effect. **proved**: an identity
from survey statistics. It needs no experimental validation and can receive none —
computing it and comparing it against itself is arithmetic, not evidence.
Machine-checked proofs are in [`formal/`](formal/) (Lean 4).

`k_eff` is sound as **diversity bookkeeping** — a correlation discount — and is
used that way here.

**Whether ρ predicts fragility in a real system** — **open**. Untested here and
everywhere else in the corpus. The bridge from a proved identity to a measured
prediction has not been built.

See [`CCA_PAPER/`](CCA_PAPER/) for the paper and its correction history, and
[`KNOWN_LIMITATIONS.md`](KNOWN_LIMITATIONS.md) for the theoretical boundaries that
are boundaries rather than bugs.

---

## Install and run

```bash
git clone https://github.com/CIRISAI/RATCHET.git
cd RATCHET
python -m venv .venv && source .venv/bin/activate
pip install numpy scipy pydantic cryptography z3-solver matplotlib pytest
pytest simulation/ -v
```

The campaign tooling under `experiments/torque/` needs only the standard library
plus `pyyaml`.

## Layout

```
experiments/torque/     the TORQUE campaign — design, corpora, tooling
docs/                   the plain-English overview (GitHub Pages)
formal/                 Lean 4 proofs
ratchet/engines/        detection, geometric, complexity, federation engines
CCA_PAPER/              the paper and its corrections
simulation/             standalone simulations and tests
```

## Not suitable for

Production AI safety without defence in depth; high-stakes deployment as a sole
safety mechanism; unsupervised operation. This is a research testbed.

---

**Looking for the agent itself?** [CIRISAgent](https://github.com/CIRISAI/CIRISAgent)
is the runtime under test — including its safety battery in `tests/safety/`. This
repository is the measurement apparatus, not the system being measured.
