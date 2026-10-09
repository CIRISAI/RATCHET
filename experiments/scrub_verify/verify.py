"""Verify a scrubber's output against RATCHET's vendored contract (reference.py).

Checks, each reported with counts and samples rather than a bare pass/fail:
  structural_keys   every RESTORE_KEYS value (upstream list + RATCHET extras) is byte-identical
  year_residue      zero historical-year matches in NON-structural output leaves; matches inside
                    RESTORE_KEYS values (UUID segments) are reported separately, not dropped
  placeholder_grammar  every `[...]` token that looks like a placeholder is a known one
  mid_token         placeholders glued to a letter (`bo[ORG_1]`, `D[ORG_2]`) or to each
                    other (`[PER_1][PER_2]`) — the signature of sub-token span offsets
  fields            which SCRUB_FIELDS subtrees changed; per-tag placeholder counts
"""
import re
from collections import Counter
from reference import (RESTORE_KEYS, PLACEHOLDER, count_year_residue,
                       walk_string_leaves, in_scrub_subtree)

_MID = re.compile(r"(?<=[^\W\d_])\[(?:[A-Z_]+?)_\d+\]|\[(?:[A-Z_]+?)_\d+\](?=[^\W\d_])|\]\[")
_ANY_BRACKET = re.compile(r"\[[A-Z][A-Z_]*(?:_\d+)?\]")


def verify(before, after) -> dict:
    b = dict(walk_string_leaves(before))
    a = dict(walk_string_leaves(after))
    yr_sem, yr_ident = count_year_residue(after, exempt=RESTORE_KEYS)
    rep = {"structural_violations": [], "year_residue": yr_sem, "year_residue_in_identifiers": yr_ident,
           "unknown_placeholders": Counter(), "mid_token": [], "tags": Counter(),
           "fields_changed": Counter(), "strings": len(b), "strings_changed": 0}
    for path, s_b in b.items():
        key = path.replace("[]", "").rsplit(".", 1)[-1]
        s_a = a.get(path)
        if s_a is None:
            continue
        if key in RESTORE_KEYS and s_a != s_b:
            rep["structural_violations"].append((path, s_b[:60], s_a[:60]))
        if s_a != s_b:
            rep["strings_changed"] += 1
            top = next((seg for seg in path.replace("[]", "").split(".") if in_scrub_subtree(seg)), key)
            rep["fields_changed"][top] += 1
        for m in PLACEHOLDER.finditer(s_a):
            rep["tags"][m.group(1) or m.group(3)] += 1
        for m in _ANY_BRACKET.finditer(s_a):
            tok = m.group(0)
            if not PLACEHOLDER.fullmatch(tok) and tok not in s_b:   # introduced, not quoted from source
                rep["unknown_placeholders"][tok] += 1
        for m in _MID.finditer(s_a):
            i = m.start()
            rep["mid_token"].append((path, s_a[max(0, i - 25): i + 30]))
    rep["ok"] = (not rep["structural_violations"] and rep["year_residue"] == 0
                 and not rep["unknown_placeholders"] and not rep["mid_token"])
    return rep


def summarize(reps: list[dict]) -> dict:
    tot = Counter(); tags = Counter(); fields = Counter(); sv = 0; mt = 0; yr = 0; yri = 0; unk = Counter()
    for r in reps:
        tot["strings"] += r["strings"]; tot["strings_changed"] += r["strings_changed"]
        tags.update(r["tags"]); fields.update(r["fields_changed"]); unk.update(r["unknown_placeholders"])
        sv += len(r["structural_violations"]); mt += len(r["mid_token"]); yr += r["year_residue"]; yri += r.get("year_residue_in_identifiers", 0)
    return {"traces": len(reps), "strings": tot["strings"], "strings_changed": tot["strings_changed"],
            "structural_violations": sv, "year_residue": yr, "year_residue_in_identifiers": yri, "mid_token_placeholders": mt,
            "unknown_placeholders": dict(unk), "tags": dict(tags), "fields_changed": dict(fields.most_common(12)),
            "all_ok": all(r["ok"] for r in reps)}
