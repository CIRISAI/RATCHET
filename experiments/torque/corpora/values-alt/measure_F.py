#!/usr/bin/env python3
"""Unit F measurement: CIRIS original vs alt-values draft.

Uses the campaign's density.py lexicon and tokenizer unmodified.
"""
import json
import re
import sys

SP = "/tmp/claude-1000/-home-emoore-RATCHET/4fdbd195-6bf1-45c9-8ffc-931540da4e4d/scratchpad"
sys.path.insert(0, SP)
from density import CORE, EXTENDED, measure  # noqa: E402

EN = "/tmp/a2911/ciris_engine/data/localized/en.json"
DRAFT = "/home/emoore/RATCHET/experiments/torque/corpora/values-alt/F-lg-axiotic.md"

lg = json.load(open(EN))["prompts"]["language_guidance"]

# pull the two fenced ```text blocks out of the draft, in order
blocks = re.findall(r"```text\n(.*?)\n```", open(DRAFT).read(), re.S)
assert len(blocks) == 2, f"expected 2 text blocks, got {len(blocks)}"

PAIRS = [
    ("09_trusted_person_first_step", lg["09_trusted_person_first_step"].strip(), blocks[0]),
    ("11_routing_doctrine", lg["11_routing_doctrine"].strip(), blocks[1]),
]

ACC = open("/tmp/a2911/ciris_engine/data/localized/accord_1.2b_en.txt", encoding="utf-8").read()
acc = measure("accord", ACC)


def row(tag, text):
    m = measure(tag, text)
    return {
        "tag": tag,
        "bytes": len(text.encode("utf-8")),
        "words": m["total_words"],
        "core_hits": m["core"]["hits"],
        "core_p1k": m["core"]["per1000"],
        "ext_hits": m["extended"]["hits"],
        "ext_p1k": m["extended"]["per1000"],
        "core_fams": {k: v for k, v in m["core"]["families"].items() if v},
        "ext_fams": {k: v for k, v in m["extended"]["families"].items() if v},
    }


print("=" * 78)
print("UNIT F — measured length + value-token density (density.py lexicon)")
print("=" * 78)

agg = {"orig": [], "alt": []}
for key, orig, alt in PAIRS:
    o, a = row("orig", orig), row("alt", alt)
    agg["orig"].append(orig)
    agg["alt"].append(alt)
    print(f"\n--- {key}")
    print(f"  {'':<10} {'bytes':>7} {'words':>7} {'CORE':>6} {'/1000':>8} {'EXT':>5} {'/1000':>8}")
    for lbl, r in (("CIRIS", o), ("alt", a)):
        print(f"  {lbl:<10} {r['bytes']:>7} {r['words']:>7} {r['core_hits']:>6} "
              f"{r['core_p1k']:>8.1f} {r['ext_hits']:>5} {r['ext_p1k']:>8.1f}")
    db = 100.0 * (a["bytes"] - o["bytes"]) / o["bytes"]
    dw = 100.0 * (a["words"] - o["words"]) / o["words"]
    print(f"  {'delta':<10} {db:>+6.1f}% {dw:>+6.1f}%")
    print(f"  families CIRIS core={o['core_fams']} ext={o['ext_fams']}")
    print(f"  families alt   core={a['core_fams']} ext={a['ext_fams']}")

print("\n" + "-" * 78)
print("UNIT TOTAL (both keys concatenated)")
to, ta = row("orig", "\n".join(agg["orig"])), row("alt", "\n".join(agg["alt"]))
print(f"  {'':<10} {'bytes':>7} {'words':>7} {'CORE':>6} {'/1000':>8} {'EXT':>5} {'/1000':>8}")
for lbl, r in (("CIRIS", to), ("alt", ta)):
    print(f"  {lbl:<10} {r['bytes']:>7} {r['words']:>7} {r['core_hits']:>6} "
          f"{r['core_p1k']:>8.1f} {r['ext_hits']:>5} {r['ext_p1k']:>8.1f}")
print(f"  delta      {100.0*(ta['bytes']-to['bytes'])/to['bytes']:>+6.1f}% "
      f"{100.0*(ta['words']-to['words'])/to['words']:>+6.1f}%")

print("\n  09-only (if 11 adjudicated procedural):")
o9, a9 = row("o", PAIRS[0][1]), row("a", PAIRS[0][2])
print(f"    CIRIS {o9['bytes']:>4} B / {o9['words']:>3} w   alt {a9['bytes']:>4} B / {a9['words']:>3} w")

print("\n" + "-" * 78)
print(f"REFERENCE  Accord 1.2b EN: {acc['total_words']:,} w  "
      f"core {acc['core']['per1000']:.2f}/1000  ext {acc['extended']['per1000']:.2f}/1000")
print("-" * 78)

print("\nVOCABULARY-INJECTION CHECK (constraint 2) — families the constitution lacks:")
BANNED = {
    "obligation": r"\b(obligation|obligations|obligated|obliged|duty|duties|stewardship|steward)\b",
    "integrity": r"\b(integrity)\b",
    "transparency": r"\b(transparent|transparency|opaque|opacity)\b",
    "responsibility": r"\b(responsible|responsibly|responsibility|responsibilities|accountable|accountability)\b",
}
alltext = "\n".join(agg["alt"]).lower()
bad = {f: len(re.findall(p, alltext)) for f, p in BANNED.items()}
print(f"  {bad}   -> {'CLEAN' if not any(bad.values()) else 'VIOLATION'}")

print("\nREGISTER CHECK (constraint 1):")
for lbl, txt in (("CIRIS", "\n".join(agg["orig"])), ("alt", "\n".join(agg["alt"]))):
    low = txt.lower()
    print(f"  {lbl:<6} should={len(re.findall(r'\bshould\b', low))} "
          f"must={len(re.findall(r'\bmust\b', low))} "
          f"we={len(re.findall(r'\bwe\b', low))} "
          f"claude={len(re.findall(r'\bclaude\b', low))} "
          f"2nd-person(you/your/yourself)={len(re.findall(r'\b(you|your|yourself)\b', low))}")

print("\nHARD-CONSTRAINT RESIDUE (constraint 3):")
HC = r"\b(bioweapon|biological weapon|chemical weapon|nuclear|radiological|uplift|" \
     r"critical infrastructure|cyberweapon|malware|oversight|disempower|mass casualt|" \
     r"seiz\w* power|csam|child sexual)\b"
hits = re.findall(HC, alltext)
print(f"  {hits}   -> {'CLEAN' if not hits else 'VIOLATION'}")

print("\nHELD-SPAN VERIFICATION (constraint 4) — must be byte-identical:")
HELD = [
    ("09 structural+pragmatic", "1. Trusted person — family member, close friend, partner."),
    ("09 contingent trigger", "For users disclosing distress,"),
    ("11 procedural frame", "When giving guidance:"),
    ("11 pragmatic referent", "the user's existing support resources"),
    ("11 procedural guard", "name the specific country's crisis line where context allows"),
    ("11 empirical criteria", "(suicidal ideation, weeks of dysfunction, command hallucinations)"),
    ("11 procedural escalation", "route serious symptoms"),
    ("11 procedural destination", "to professional care"),
]
for lbl, span in HELD:
    src = PAIRS[0][1] if lbl.startswith("09") else PAIRS[1][1]
    alt = PAIRS[0][2] if lbl.startswith("09") else PAIRS[1][2]
    ok = span in src and span in alt
    print(f"  [{'OK ' if ok else 'FAIL'}] {lbl:<28} {span[:52]!r}")

print("\nTIER-LABEL CHECK (constraint 5):")
TIERS = ["broadly safe", "broadly ethical", "compliant with anthropic", "genuinely helpful"]
print(f"  {[t for t in TIERS if t in alltext]}   -> "
      f"{'CLEAN (no ordering introduced)' if not any(t in alltext for t in TIERS) else 'PRESENT'}")
