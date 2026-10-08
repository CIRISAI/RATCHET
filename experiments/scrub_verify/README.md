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
