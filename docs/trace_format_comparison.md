# Agent trace formats and failure taxonomies: comparison (issue #32)

Status: in progress. Rows are added only after the source has been checked against the
cited version. Each row names that version.

## Columns

- **Unit:** what the source models (span, step, turn, decision, trace).
- **Context provenance:** whether it records *why* the agent saw what it saw.
- **Predictions:** whether it carries predictions registered before a run, with a kill
  condition, or only post-run observations.
- **Observed label:** the failure label the source assigns from a trace, plus the evidence
  it cites. This is a vocabulary term, not a diagnosis.
- **Test-registered diagnosis:** whether a class is registered only after a discriminating
  test hits, with refuted classes kept visible, as `.ks` diagnoses and misfits are.
- **Decision:** adopt, interoperate, cite, or differ on purpose.

These two columns are kept separate on purpose. An observed label that overlaps a misfit or
one of the 11+1 classes is **not** a one-to-one match unless a test supports it.

## Failure and trace taxonomies

| Source (version checked) | Unit | Context provenance | Predictions | Observed label | Test-registered diagnosis | Decision |
|---|---|---|---|---|---|---|
| **MAST.** Cemri et al., *Why Do Multi-Agent LLM Systems Fail?*, arXiv:2503.13657v3 (26 Oct 2025) | Multi-agent execution trace | No | Post-run only. Labels come from grounded-theory coding with no prior hypotheses (§3.1) | 14 failure modes in 3 categories: system design, inter-agent misalignment, task verification (§4). Human agreement measured by Cohen's κ (§3.2). LLM annotator is o1, κ 0.77 (§3.3) | No. Modes are revised by adding, removing or merging during coding (§3.2). No discriminating test, and refuted modes are not retained | Cite as a failure-label vocabulary. Map the 11+1 classes to it only where a test supports the mapping |
| **TRAIL.** Deshpande et al., *TRAIL: Trace Reasoning and Agentic Issue Localization*, arXiv:2505.08638v3 (23 Jun 2025) | Span within an OpenTelemetry/OpenInference trace (§4.1) | No. Traces carry span structure, not why each piece of context was present | Post-run only. Four annotators label the errors (§4.2) | Taxonomy in three areas: reasoning, system execution, planning and coordination (§3). 148 traces from GAIA and SWE-Bench Lite, single- and multi-agent. The leaf-category count has not been checked yet (Fig. 1) | No | Cite for span-level localization. Assess trace interchange separately: using OpenTelemetry does not preserve signed context provenance (see issue #32 Q3) |

The "no" answers for predictions and test-registered diagnosis are based only on the main
text of these two papers. The appendices have not been checked, and the answers are not a
claim about prior art in general.

Rows contributed by keyboy10 in issue #32 (2026-11-20). Checked against arXiv v3 of each
paper on 2026-11-19 (local date).

## Trace formats

To do: OpenTelemetry GenAI semantic conventions, OpenInference, OpenLLMetry, LangSmith,
Langfuse, MLflow Tracing, Inspect AI logs, W3C PROV-O, and signed CIRISAgent traces.
