# fiction/ — synthetic / roleplay documents (NOT real findings)

This directory holds **LLM-generated roleplay and synthetic documents**. They are
**not real audits, not real security findings, and not real measurements**. They are
kept only for provenance — so their history is preserved and so nobody rediscovers
them elsewhere and mistakes them for genuine work.

Moved here per [issue #7](https://github.com/) ("Remove synthetic ADVERSARIAL_ANALYSIS.md
+ sync RATCHET to current coherence-ratchet / CIRISConstitution state").

## Why these are quarantined

The quantitative claims inside are **fabricated** and are **contradicted by measurement**.

Concrete example — `ADVERSARIAL_ANALYSIS.md` asserts the constraint set "collapses" to
`k_eff ≈ 3–5 independent` and declares the geometric guarantee "BROKEN" with a fabricated
"90% probability." The actual effective independent dimensionality (N_eff), measured on the
real 6,465-trace corpus (`release/data_scrubbed_v1/accord_traces.jsonl`, via
`CIRISLens/scripts/measure_n_eff.py`), is **N_eff_PR ≈ 6.0 and N_eff_H ≈ 7.7**
(semantic / H3ERE features only, pre-CEG) — i.e. *higher* than the doc's fabricated
"collapsed" number, not lower. There is no adversarial N_eff measurement anywhere; the
corpus is entirely benign traces. The document's framing ("Red Team Security Researcher
(ex-DeepMind)", "Breaking Everything", "Complete bypass of safety guarantees") is roleplay,
not a real assessment.

## Contents

- `ADVERSARIAL_ANALYSIS.md` — synthetic red-team roleplay of the Coherence Ratchet
  framework. Fabricated quantitative claims; contradicted by the N_eff measurement above.

For the real, evidence-based safety/deception-resistance argument, see the CEG (CIRIS
Epistemic Grammar) substrate leg in `CIRISConstitution` and the N_eff measurement in
`CIRISLens/scripts/measure_n_eff.py`.
