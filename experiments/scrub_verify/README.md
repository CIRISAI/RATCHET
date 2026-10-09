# scrub_verify — RATCHET's vendored scrub contract and verifier

RATCHET consumes the post-fold scrubber (`ciris-server` wheel, the same Rust core prod runs)
and **verifies its output against a contract it holds independently**. Nothing here is a
production scrubber.

- `reference.py` — the contract: the 42-key `SCRUB_FIELDS` (authoritative Rust set, pinned),
  the 8 regexes in the Rust application order, the year-residue invariant, and
  `STRUCTURAL_IDENTIFIER_KEYS` as a **RATCHET requirement** — it was fixed in the Python
  (CIRISLens#11/#12) and did not survive the fold into the Rust walker.
- `verify.py` — checks structural keys, year residue, placeholder grammar, and
  **mid-token placeholders** (the signature of sub-token span offsets).
- `smoke.py` — runs both against real export payloads and the installed wheel.

Order of operations for any RATCHET dataset: **join provenance first, scrub second.** The
provenance split is keyed on `agent_id_hash`, `task_id`, `channel_id`; the shipped wheel
scrubs those when they sit inside a `SCRUB_FIELDS` subtree.

## Backbone comparison (same `measure.py`, same first-300-event sample, 2026-10-08 export)

| backbone | narrative strings touched | mid-token placeholders | tags | ms/event |
|---|---|---|---|---|
| XLM-R wikiann (wheel default) | 59.8% (337/564) | 236 | ORG 453 · PER 95 · LOC 38 | 108 |
| DistilBERT-hrl (`fetch_distilbert_hrl.sh`) | 4.4% (26/586) | 5 | PER 20 · ORG 10 | 56 |

XLM-R's replaced spans are common nouns (`open`, `trust`, `counselor`, `boundaries`) and
sub-word fragments; DistilBERT's are names (Sofia, María, Camille, Sam, Ashley, Sean, Kate,
Chloe). Both exhibit the mid-token cut (`[ORG_1]IRIS`, `[PER_1]itely`, `[ORG_1]V`), so the
defect is in the shared span assembly (CIRISServer#755), not the model; XLM-R triggers it
~47× more. Neither catches every `@handle` (`@[PER_1] asked @jeff`) — a RATCHET regex pass
covers that. The wheel's `distilbert` default is unloadable from the hub (no
`tokenizer.json`); the fetch script assembles a working dir. `ort` falls through to XLM-R
silently on a wheel without `ner-ort`.

Measurement caveat fixed in `measure.py`: the after-side originally kept a ≥40-char
filter, so strings a backbone shrank below it left the denominator; XLM-R's figure above
is therefore slightly under-stated.
