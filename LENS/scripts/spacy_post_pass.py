#!/usr/bin/env python3
"""
spaCy English-NER post-pass on top of the Rust multilingual NER output.

Why both: XLM-R-WikiANN handles 29-language NER but misses English
entities that appear parenthetically inside non-Latin text (e.g. an
Amharic sentence with `(Restricted Query)` mid-stream). spaCy
en_core_web_sm catches those because it scans Latin tokens
context-independently. Layered together they close the gap.

This script consumes the output of rescrub_release_v3.py (level=full_traces)
and writes a final corpus where every string field has been swept through
spaCy as a backstop — without disturbing the placeholders that the Rust
scrubber already produced.

Usage:
    python3 scripts/spacy_post_pass.py \
        [--src ~/RATCHET/release/data_scrubbed_v1_ner] \
        [--dst ~/RATCHET/release/data_scrubbed_v1]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

# Pre-existing placeholder vocabulary from BOTH Rust and Python scrubbers.
# These get masked before spaCy sees the text so the NER doesn't re-tag
# `[YEAR]` as an ORG (which it does — observed empirically).
PLACEHOLDER_RE = re.compile(
    r"\[(?:YEAR|EMAIL|PHONE|IP_ADDRESS|URL|SSN|CREDIT_CARD|IDENTIFIER"
    r"|PER|ORG|LOC|MISC|GPE|FAC|NORP|PERSON|DATE|TIME|EVENT|WORK_OF_ART|LAW"
    r")(?:_\d+)?\]"
)

# Private-use Unicode codepoint that spaCy treats as a non-letter symbol —
# safe to substitute placeholders with this without spaCy fragmenting it.
MASK_CHAR = ""

# spaCy entity types we redact (parallel to pii_scrubber.REDACT_ENTITY_TYPES).
REDACT_LABELS = {
    "PERSON", "ORG", "GPE", "FAC", "LOC", "NORP",
    "DATE", "TIME", "EVENT", "MISC", "WORK_OF_ART", "LAW",
}

TRACE_FILES = ("accord_traces.jsonl", "trace_context.jsonl")
PASSTHROUGH_FILES = (
    "accord_public_keys.jsonl",
    "accord_trace_batches.jsonl",
    "connectivity_events.jsonl",
)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def mask_placeholders(text: str) -> tuple[str, list[str]]:
    """Replace each placeholder in `text` with MASK_CHAR. Returns the
    masked text and the list of original placeholders in order."""
    saved: list[str] = []

    def replace(m: re.Match) -> str:
        saved.append(m.group(0))
        return MASK_CHAR

    masked = PLACEHOLDER_RE.sub(replace, text)
    return masked, saved


def restore_placeholders(masked: str, saved: list[str]) -> str:
    """Inverse of mask_placeholders."""
    parts = masked.split(MASK_CHAR)
    out = parts[0]
    for i, ph in enumerate(saved):
        if i + 1 < len(parts):
            out += ph + parts[i + 1]
        else:
            # Mismatch: a mask got dropped by spaCy. Append remaining
            # placeholders verbatim.
            out += ph
    return out


def has_substantial_non_latin(text: str) -> bool:
    """True when ≥3 chars OR ≥5% of `text` is outside the Latin block.
    Used to route to the multilingual model. Mirrors
    `api/pii_scrubber._has_non_latin`."""
    if not text:
        return False
    non_latin = sum(1 for c in text if ord(c) > 0x024F)
    return non_latin >= max(3, len(text) // 20)


def spacy_scrub_text(text: str, nlp_en, nlp_xx, counters: dict[str, int]) -> str:
    """Run spaCy on `text` with placeholder protection, redact entities.
    Routes non-Latin-heavy strings to the multilingual model
    (`xx_ent_wiki_sm`) so CJK / Arabic / Cyrillic content is covered;
    everything else goes to the English model."""
    if not text or not text.strip():
        return text
    nlp = nlp_xx if (nlp_xx is not None and has_substantial_non_latin(text)) else nlp_en
    if nlp is None:
        return text  # spaCy not available — leave untouched
    if not PLACEHOLDER_RE.search(text):
        return _scrub_via_spacy(text, nlp, counters)
    masked, saved = mask_placeholders(text)
    scrubbed_masked = _scrub_via_spacy(masked, nlp, counters)
    return restore_placeholders(scrubbed_masked, saved)


def _scrub_via_spacy(text: str, nlp, counters: dict[str, int]) -> str:
    doc = nlp(text)
    out: list[str] = []
    cursor = 0
    for ent in doc.ents:
        if ent.label_ not in REDACT_LABELS:
            continue
        if MASK_CHAR in ent.text:
            # Don't redact a span that includes a masked placeholder —
            # would tear the mask apart and corrupt restoration.
            continue
        out.append(text[cursor:ent.start_char])
        n = counters.get(ent.label_, 0) + 1
        counters[ent.label_] = n
        # Use distinct placeholder shape so spaCy-second-pass redactions
        # are visually distinguishable from Rust-first-pass ones.
        out.append(f"[{ent.label_}_S{n}]")
        cursor = ent.end_char
    out.append(text[cursor:])
    return "".join(out)


def walk_strings_inplace(value, nlp_en, nlp_xx, counters):
    """Recursively scrub strings in a parsed JSON value, in place."""
    if isinstance(value, dict):
        for k in list(value.keys()):
            value[k] = walk_strings_inplace(value[k], nlp_en, nlp_xx, counters)
        return value
    if isinstance(value, list):
        return [walk_strings_inplace(v, nlp_en, nlp_xx, counters) for v in value]
    if isinstance(value, str):
        return spacy_scrub_text(value, nlp_en, nlp_xx, counters)
    return value


def process_file(src: Path, dst: Path, nlp_en, nlp_xx) -> dict:
    counters: dict[str, int] = {}
    rows = entities = 0
    with src.open() as fin, dst.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            walk_strings_inplace(row, nlp_en, nlp_xx, counters)
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows += 1
    entities = sum(counters.values())
    return {
        "rows": rows,
        "entities_redacted": entities,
        "by_label": dict(counters),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=os.path.expanduser("~/RATCHET/release/data_scrubbed_v1_ner"))
    p.add_argument("--dst", default=os.path.expanduser("~/RATCHET/release/data_scrubbed_v1"))
    args = p.parse_args()

    src_dir = Path(args.src)
    dst_dir = Path(args.dst)
    dst_dir.mkdir(parents=True, exist_ok=True)

    print(f"spaCy post-pass: {src_dir} → {dst_dir}")

    import spacy
    nlp_en = spacy.load("en_core_web_sm")
    print("loaded spaCy en_core_web_sm")
    try:
        nlp_xx = spacy.load("xx_ent_wiki_sm")
        print("loaded spaCy xx_ent_wiki_sm (multilingual)")
    except OSError:
        nlp_xx = None
        print("xx_ent_wiki_sm not installed — non-Latin content will fall back to English-only")

    manifest: dict = {
        "release_built_at": datetime.now(UTC).isoformat(),
        "scrubber": (
            "Rust v2 (DistilBERT-multilingual NER + regex, ort INT8) "
            "→ spaCy en_core_web_sm + xx_ent_wiki_sm post-pass"
        ),
        "filter_rules": [
            "Pass 1: cirislens_core.scrub_trace at level=full_traces "
            "(XLM-R / DistilBERT multilingual NER + regex; year-residue invariant).",
            "Pass 2: spaCy backstop with placeholder protection. Routes "
            "non-Latin-heavy strings to xx_ent_wiki_sm (multilingual), "
            "everything else to en_core_web_sm. Catches entities the Rust "
            "multilingual model missed in dense-placeholder context, "
            "including English names embedded in non-English prompts and "
            "CJK toponyms that don't survive transformer chunk boundaries.",
        ],
        "files": {},
    }

    for fname in TRACE_FILES:
        src_p = src_dir / fname
        dst_p = dst_dir / fname
        if not src_p.exists():
            print(f"  {fname}: not present in src, skipping")
            continue
        stats = process_file(src_p, dst_p, nlp_en, nlp_xx)
        manifest["files"][fname] = {
            "sha256": sha256(dst_p),
            "size_bytes": dst_p.stat().st_size,
            "row_count": stats["rows"],
            "spacy_entities_redacted": stats["entities_redacted"],
            "spacy_by_label": stats["by_label"],
        }
        print(
            f"  {fname}: {stats['rows']} rows, {stats['entities_redacted']} "
            f"spaCy entities redacted, by_label={stats['by_label']}"
        )

    for fname in PASSTHROUGH_FILES:
        src_p = src_dir / fname
        dst_p = dst_dir / fname
        if src_p.exists():
            shutil.copy(src_p, dst_p)
            with dst_p.open() as f:
                rows = sum(1 for _ in f)
            manifest["files"][fname] = {
                "sha256": sha256(dst_p),
                "size_bytes": dst_p.stat().st_size,
                "row_count": rows,
                "passthrough": True,
            }
            print(f"  {fname}: passthrough ({rows} rows)")

    (dst_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nWrote {dst_dir / 'MANIFEST.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
