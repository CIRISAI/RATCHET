# Adaptation map — what must be authored for the `h3ere-alt` arm

Measured at agent `v2.9.11-stable` (7e71d0381), `en`, no manifest.

**140 of 585 blocks** need adapting: 52 `axiotic` (vary outright) and 88 `mixed`
carrying `axiotic` contamination (replace-whole, never hold — see the regime's
disposition note).

Those 140 blocks collapse to **18 distinct source keys**. The key is the
authoring unit; the block count is just how many steps each key reaches.

| bytes | blocks | class | source key | unit |
|---|---|---|---|---|
| 180,522 | 8 | axiotic | `corpus:accord.polyglot_full` | **A** |
| 54,725 | 8 | axiotic | `corpus:accord.localized` | **A** |
| 7,215 | 5 | axiotic | `corpus:accord.polyglot_compressed` | **A** |
| 24,064 | 2 | mixed | `conscience_prompt:optimization_veto_conscience.system_prompt` | **B** |
| 9,110 | 2 | mixed | `conscience_prompt:epistemic_humility_conscience.system_prompt` | **B** |
| 6,464 | 2 | mixed | `conscience_prompt:coherence_conscience.system_prompt` | **B** |
| 23,403 | 1 | mixed | `dma_prompt:pdma_ethical.system_guidance_header` | **C** |
| 5,495 | 5 | mixed | `dma_prompt:action_selection_pdma.context_integration#slots` | **D** |
| 763 | 5 | mixed | `dma_prompt:action_selection_pdma.closing_reminder` | **D** |
| 163 | 5 | axiotic | `dma_prompt:action_selection_pdma.csdma_ambiguity_guidance` | **D** |
| 1,076 | 13 | mixed | `string:…language_guidance.13_exemplar_speak_response` | **E** |
| 1,020 | 13 | mixed | `string:…language_guidance.25_exemplar_cross_cluster` | **E** |
| 958 | 13 | mixed | `string:…language_guidance.23_ratification_templates` | **E** |
| 735 | 13 | mixed | `string:…language_guidance.16_exemplar_false_reassurance` | **E** |
| 613 | 13 | mixed | `string:…language_guidance.14_exemplar_register_pressure` | **E** |
| 269 | 13 | axiotic | `string:…language_guidance.11_routing_doctrine` | **F** ⚠ |
| 160 | 13 | axiotic | `string:…language_guidance.09_trusted_person_first_step` | **F** |
| 3,407 | 6 | mixed | **`inline`** | **BLOCKED** |

## Six authoring units

**A — the Accord (242,462 B, 76% of the surface).** Three keys, one source. The
alt Accord is authored **once** and generates all three forms: full, localized,
compressed. This is not three jobs.

**B — conscience criteria (39,638 B).** The standard each faculty judges
*against*. `optimization_veto` is polyglot by design and is the single hardest
congruence problem in the campaign: a monolingual replacement confounds values
with language coverage and would look exactly like a clean result.

**C — the ethical DMA system header (23,403 B).**

**D — ASPDMA guidance (6,421 B).** Three keys, one voice.

**E — the worked exemplars (4,402 B).** Verdict, register and schema in the same
tokens — the co-occurrence *is* the demonstration [T-5a]. Replace whole or hold
verbatim; never split.

**F — the two axiotic `language_guidance` parts (429 B).**
⚠ `11_routing_doctrine` is **under adjudication**: it ships `axiotic` but two
blind annotators independently classified it `procedural` (κ study, 2026-08-07).
If the adjudication says procedural it leaves this map, and unit F is
`09_trusted_person_first_step` alone.

## BLOCKED — 3,407 B across 6 blocks, source `inline`

Not reachable by any manifest key, so it cannot be adapted through the override
facility at all. These 6 blocks will carry **CIRIS axiotic content into the alt
arm** regardless of how well the other 17 keys are authored.

That is a `confound_accepted` in the strict sense — the one place the regime's
"replace-whole, don't accept confounds" default genuinely cannot apply — and it
biases `values_effect` toward zero. It must be declared per-block with the
reason, not swept into the totals.

## Constraints on every unit

1. **Register conversion first.** The Accord is 2nd-person imperative
   (`must` 3.53/1000); the constitution is 3rd-person institutional
   (`should` 5.88/1000). Convert before any density work, or the density
   measurement is taken on the wrong text.
2. **Do not inject vocabulary the source lacks.** The extended-density shortfall
   (~38 achievable vs ~64) sits in duty-bearer terms — obligation, integrity,
   transparency, responsibility — that Claude's constitution genuinely does not
   use. Writing them in would *author* the arm rather than adapt a real value
   set, and the pre-registered asymmetry exists precisely so that this is not
   tempting.
3. **Strip the hard constraints.** `prohibitions.py` is the single shared deontic
   layer across all arms. Any prohibition text surviving into the alt corpus
   reintroduces a safety-boundary variation and invalidates the run.
4. **Preserve non-axiotic content in `mixed` units.** Replace-whole means the
   pragmatic, deontic, procedural and structural content is held *identical in
   force* while the axiotic content is swapped. It does not mean rewriting
   freely.
