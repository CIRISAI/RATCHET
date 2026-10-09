"""Vendored reference of the CIRIS trace-scrub contract, owned by RATCHET.

RATCHET consumes the post-fold product (`ciris-server`, PyO3 wheel) for the scrub itself.
This module is NOT a second scrubber for production; it is the spec RATCHET holds so it
can verify any scrubber's output without trusting it, and a regex-only reference
implementation to diff a wheel's `detailed`-level output against.

Provenance of every list below, pinned:
  SCRUB_FIELDS            CIRISServer cfd2867b (2026-09-25)
                          crates/ciris-lens-core/patterns_from_cirislens_core/scrubber/fields.rs
                          — the AUTHORITATIVE Rust set (42 keys). The Python list in
                          CIRISLens api/pii_scrubber.py (be73e8ff) is the SAME 42 keys,
                          grouped differently; the sets were diffed, not assumed equal.
  REGEX (8 patterns)      same commit, scrubber/regex.rs, same application ORDER:
                          identifier -> year -> email -> phone -> ipv4 -> url -> ssn -> cc.
                          `\\p{L}` in the Rust YEAR_IDENTIFIER is `[^\\W\\d]` in Python `re`
                          (letters + underscore, which matches the Rust `[\\p{L}_]`).
  STRUCTURAL_IDENTIFIER_KEYS
                          CIRISLens be73e8ff (2026-06-12) api/pii_scrubber.py.
                          NOT present in the Rust crate — the CIRISLens#11/#12 fixes did not
                          survive the fold. RATCHET keeps it as a REQUIREMENT on any scrubber
                          output; verify.py enforces it regardless of what the scrubber did.
  REDACT_ENTITY_TYPES     CIRISLens be73e8ff api/pii_scrubber.py (spaCy-era tag set). The
                          wheel's XLM-R wikiann model emits only PER/ORG/LOC (+MISC).

Year cutoff: 1700-2023 (2024+ preserved for live timestamps). Bumped by release process.
"""
import re

SCRUB_FIELDS = frozenset({
    # THOUGHT_START
    "task_description", "initial_context", "thought_content",
    # SNAPSHOT_AND_CONTEXT
    "system_snapshot", "gathered_context", "relevant_memories", "conversation_history",
    "current_thought_summary",
    # DMA_RESULTS
    "reasoning", "prompt_used", "combined_analysis", "flags", "alignment_check",
    "conflicts", "stakeholders",
    # ASPDMA_RESULT
    "action_rationale", "reasoning_summary", "action_parameters", "aspdma_prompt",
    "questions", "completion_reason",
    # CONSCIENCE_RESULT
    "conscience_override_reason", "epistemic_data", "updated_status_content",
    "entropy_reason", "coherence_reason", "optimization_veto_justification",
    "epistemic_humility_justification", "epistemic_humility_uncertainties",
    # ACTION_RESULT
    "execution_error",
    # IDMA_RESULT
    "intervention_recommendation", "next_best_recovery_step", "correlation_factors",
    "top_correlation_factors", "common_cause_flags", "sources_identified", "source_ids",
    "source_clusters", "source_types", "source_type_counts",
    "pairwise_correlation_summary", "reasoning_state",
})
assert len(SCRUB_FIELDS) == 42, len(SCRUB_FIELDS)

# Keys whose VALUES are protocol identifiers. Public by construction; must never change.
STRUCTURAL_IDENTIFIER_KEYS = frozenset({
    "agent_id_hash", "trace_id", "thought_id", "task_id", "span_id", "correlation_id",
    "signature_key_id", "signing_key_id", "channel_id",
})

REDACT_ENTITY_TYPES = frozenset({
    "PERSON", "ORG", "GPE", "FAC", "LOC", "EMAIL", "PHONE", "NORP", "DATE", "TIME",
    "EVENT", "MISC", "WORK_OF_ART", "LAW",
})
KEEP_ENTITY_TYPES = frozenset({"MONEY", "PERCENT", "QUANTITY", "ORDINAL", "CARDINAL"})

_YEAR = r"(?:1[7-9]\d{2}|20[0-1]\d|202[0-3])"
YEAR_IDENTIFIER = re.compile(
    r"\b(?:"
    r"\w{0,40}[^\W\d]\w{0,40}" + _YEAR + r"\w{0,40}"
    r"|"
    r"\w{0,40}" + _YEAR + r"\w{0,40}[^\W\d]\w{0,40}"
    r")\b"
)
HISTORICAL_YEAR = re.compile(r"\b" + _YEAR + r"\b")
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE = re.compile(
    r"(?x)"
    r"\+1[-.\s]+\(?[0-9]{3}\)?[-.\s]*[0-9]{3}[-.\s]*[0-9]{4}"
    r"|\([0-9]{3}\)[-.\s]*[0-9]{3}[-.\s]*[0-9]{4}"
    r"|\b[0-9]{3}[-.\s]+[0-9]{3}[-.\s]+[0-9]{4}\b"
)
IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
URL = re.compile(r"https?://[^\s<>]+")
SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
CREDIT_CARD = re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b")

# (pattern, placeholder) in the Rust application order.
REGEX_ORDER = (
    (YEAR_IDENTIFIER, "[IDENTIFIER]"),
    (HISTORICAL_YEAR, "[YEAR]"),
    (EMAIL, "[EMAIL]"),
    (PHONE, "[PHONE]"),
    (IPV4, "[IP_ADDRESS]"),
    (URL, "[URL]"),
    (SSN, "[SSN]"),
    (CREDIT_CARD, "[CREDIT_CARD]"),
)
REGEX_PLACEHOLDERS = frozenset(p for _, p in REGEX_ORDER)
NER_TAGS = frozenset({"PER", "ORG", "LOC", "MISC", "GPE", "FAC", "NORP", "DATE", "TIME",
                      "EVENT", "LAW", "WORK_OF_ART", "HANDLE"})  # HANDLE: RATCHET @handle pass
PLACEHOLDER = re.compile(r"\[(?:(" + "|".join(sorted(NER_TAGS)) + r")_(\d+)|("
                         + "|".join(re.escape(p.strip("[]")) for p in REGEX_PLACEHOLDERS) + r"))\]")

MAX_DEPTH = 30


def scrub_string(s: str, stats: dict | None = None) -> str:
    """Rust `scrub_string`: all 8 patterns, in order, on one string."""
    out = s
    for pat, rep in REGEX_ORDER:
        n = len(pat.findall(out))
        if n:
            if stats is not None:
                stats["regex_redactions"] = stats.get("regex_redactions", 0) + n
            out = pat.sub(rep, out)
    return out


def regex_scrub(value, stats: dict | None = None, _depth: int = 0, _key: str | None = None):
    """Reference `detailed`-level walk: regex on EVERY string leaf, structural keys exempt.

    This is the Rust `walk_regex_only` plus RATCHET's structural-key requirement. Diffing
    it against a wheel's `detailed` output should show differences ONLY at structural keys
    (where the wheel, lacking the allowlist, may have scrubbed).
    """
    if _depth > MAX_DEPTH:
        raise ValueError(f"walker depth > {MAX_DEPTH}")
    if isinstance(value, str):
        if _key in STRUCTURAL_IDENTIFIER_KEYS:
            return value
        return scrub_string(value, stats)
    if isinstance(value, dict):
        return {k: regex_scrub(v, stats, _depth + 1, k) for k, v in value.items()}
    if isinstance(value, list):
        return [regex_scrub(v, stats, _depth + 1, _key) for v in value]
    return value


def walk_string_leaves(value, path: str = ""):
    """Yield (path, string) for every string leaf. Path uses `.key` and `[]`."""
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from walk_string_leaves(v, f"{path}.{k}" if path else k)
    elif isinstance(value, list):
        for v in value:
            yield from walk_string_leaves(v, path + "[]")


def count_year_residue(value) -> int:
    """Rust `count_year_residue`: historical-year matches surviving in string leaves."""
    return sum(len(HISTORICAL_YEAR.findall(s)) for _, s in walk_string_leaves(value))


def in_scrub_subtree(path: str) -> bool:
    """True if any key on the path is a SCRUB_FIELDS key (subtree semantics)."""
    for seg in path.replace("[]", "").split("."):
        if seg in SCRUB_FIELDS:
            return True
    return False
