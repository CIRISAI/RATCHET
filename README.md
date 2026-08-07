# RATCHET

**Does an ethics pipeline actually hold an AI agent to the design it claims — and
does it keep holding once you stop paying for it?**

That question is the current work. RATCHET is where the experiment that answers it
is designed, pre-registered and run.

📄 **[Read the plain-English overview →](https://cirisai.github.io/RATCHET/)**

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![DOI: CCA Paper](https://zenodo.org/badge/DOI/10.5281/zenodo.18142668.svg)](https://doi.org/10.5281/zenodo.18142668)

---

## TORQUE — the live campaign

[CIRIS agents](https://github.com/CIRISAI/CIRISAgent) put every decision through a
pipeline: several reasoning stages plus four conscience faculties that can veto an
action. That costs roughly twenty extra model calls per thought. **Is it doing
anything?**

The question splits three ways, and the splitting is most of the design:

- **Does the pipeline change behaviour?** Compare it against the same model handed
  the same values as plain instructions.
- **Do the specific values matter, or only the scaffolding?** Swap CIRIS's values
  for a different real value system, hold everything else fixed.
- **Does it hold when you stop paying?** Withdraw the pipeline mid-conversation
  and keep probing. If behaviour drifts back toward the bare model, the effect was
  *maintenance*, not training.

That last one is the name. Torque is a force you keep applying.

**Status: designed, not run. There are no results.** When there are, every stake
is marked survives / fires / void — including the ones that go against us.

| | |
|---|---|
| [`experiments/torque/`](experiments/torque/) | the campaign: regime manifest, corpora, tooling |
| [`TORQUE_REGIME.yaml`](experiments/torque/TORQUE_REGIME.yaml) | machine-readable design — six arms, four contrasts, bounds |
| [`PILOT.md`](experiments/torque/PILOT.md) | 10-question dress rehearsal with pre-declared pass criteria |
| [RATCHET#16](https://github.com/CIRISAI/RATCHET/issues/16) | arms, stakes, void conditions |
| [CIRISOntology#2](https://github.com/CIRISAI/CIRISOntology/issues/2) | independent review, including its open objections |

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
(Book IX of the CIRIS Covenant). That work stands, with corrections — and the
corrections matter more than the original results.

**`k_eff = k / (1 + ρ(k−1))`** — the Kish design effect. **proved**: an identity
from survey statistics. It needs no experimental validation and can receive none.
Computing it and comparing it against itself is arithmetic, not evidence, and
earlier work in this repository did exactly that. Machine-checked proofs are in
[`formal/`](formal/) (Lean 4).

**Whether ρ predicts fragility in a real system** — **open**. Untested here and
everywhere else in the corpus. The bridge from a proved identity to a measured
prediction has not been built.

### Withdrawn — do not cite these

[CCA v5](https://doi.org/10.5281/zenodo.21730551) (2026-08-01) withdrew, and this
README previously advertised some of them as validated:

| claim | status |
|---|---|
| volume decay matches `exp(-λ·k_eff)` within 5% | **falsified** — wrong by 1.9×–4.1× |
| GPU array validates the k_eff formula | **withdrawn** — both results were identity checks |
| `α/k_eff` stability criterion | **corrected** to `α/k ≥ d` |
| institutional application | **negative result** — re-scored below chance |
| ~40%/60% undetectability figures | **withdrawn** — L-01 is existence with no measure |
| software-induced coherence collapse | **withdrawn** — measurement artifact |

Two falsification conditions fired. `k_eff` remains sound as *diversity
bookkeeping* — a correlation discount — but not as a decay-rate exponent.

See [`CCA_PAPER/CORRECTIONS_v3.md`](CCA_PAPER/CORRECTIONS_v3.md) for all 18
verified defects, and [`KNOWN_LIMITATIONS.md`](KNOWN_LIMITATIONS.md) for the
theoretical boundaries that are boundaries rather than bugs.

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
