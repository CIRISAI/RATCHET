#!/usr/bin/env python3
"""Unit F measurement + drift audit: CIRIS original vs alt-values draft.

Scope: `09_trusted_person_first_step` ONLY. `11_routing_doctrine` was ruled
PROCEDURAL on 2026-08-07 and left the adaptation map; this script asserts that no
adaptation of it survives in the draft's shipping payload.

Uses the campaign's density.py lexicon and tokenizer unmodified.
/tmp/a2911 is read-only and is never written.
"""
import difflib
import json
import os
import re
import sys

SP = "/tmp/claude-1000/-home-emoore-RATCHET/4fdbd195-6bf1-45c9-8ffc-931540da4e4d/scratchpad"
sys.path.insert(0, SP)
from density import measure  # noqa: E402

EN = "/tmp/a2911/ciris_engine/data/localized/en.json"
ACCORD = "/tmp/a2911/ciris_engine/data/localized/accord_1.2b_en.txt"
DRAFT = "/home/emoore/RATCHET/experiments/torque/corpora/values-alt/F-lg-axiotic.md"
KEY = "09_trusted_person_first_step"
MARKER = "<!-- SHIP: prompts.language_guidance.09_trusted_person_first_step -->"

lg = json.load(open(EN))["prompts"]["language_guidance"]
ORIG = lg[KEY].strip()

# The draft quotes CIRIS text in several fenced blocks. Only the marked one ships.
src = open(DRAFT).read()
m = re.search(re.escape(MARKER) + r"\s*\n```text\n(.*?)\n```", src, re.S)
assert m, "shipping block marker not found — draft structure changed"
ALT = m.group(1)
assert "\n" not in ALT, "shipping payload must be a single line"

FAIL = []


def check(ok, label):
    print(f"  [{'OK  ' if ok else 'FAIL'}] {label}")
    if not ok:
        FAIL.append(label)


print("=" * 78)
print(f"UNIT F — {KEY}  (11_routing_doctrine: OUT OF SCOPE, ruled procedural)")
print("=" * 78)

# ---------------------------------------------------------------- 1. THE DIFF
print("\n--- 1. DIFF (ground truth)")
print(f"  CIRIS : {ORIG}")
print(f"  alt   : {ALT}")
pre = os.path.commonprefix([ORIG, ALT])
print(f"\n  byte-identical common prefix : {len(pre.encode())} / {len(ORIG.encode())} B")
print(f"  CIRIS changed tail           : {ORIG[len(pre):]!r}")
print(f"  alt   changed tail           : {ALT[len(pre):]!r}")
print("\n  opcode-level (word tokens):")
ow, aw = ORIG.split(), ALT.split()
for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=ow, b=aw).get_opcodes():
    if tag == "equal":
        continue
    print(f"    {tag:<7} CIRIS={' '.join(ow[i1:i2])!r:<32} alt={' '.join(aw[j1:j2])!r}")

# ------------------------------------------------- 2. HELD SPANS (constraint 4)
print("\n--- 2. HELD SPANS — must be byte-identical in both (constraint 4)")
HELD = [
    ("structural ordinal", "1."),
    ("pragmatic referent set", "Trusted person — family member, close friend, partner."),
    ("contingent trigger", "For users disclosing distress,"),
    ("procedural move verb", "validating"),
    ("pragmatic quoted referent", '"talk to someone you trust"'),
    ("structural sequence position", "first step"),
]
for lbl, span in HELD:
    check(span in ORIG and span in ALT, f"{lbl:<30} {span[:46]!r}")

# ---------------------------------------------------- 3. SCOPE (11 must be out)
print("\n--- 3. SCOPE — no 11_routing_doctrine adaptation in the shipping payload")
R11 = lg["11_routing_doctrine"].strip().lower()
alt_low = ALT.lower()
for probe in ("when giving guidance", "crisis line", "suicidal ideation",
              "command hallucinations", "minimization", "cover yourself",
              "professional care", "existing support resources"):
    check(probe not in alt_low, f"absent from payload: {probe!r}")
check(R11 not in alt_low, "verbatim 11 string absent")

# --------------------------------------------------------- 4. MEASUREMENT
print("\n--- 4. MEASUREMENT (density.py, unmodified)")


def row(text):
    r = measure("x", text)
    return (len(text.encode("utf-8")), r["total_words"],
            r["core"]["hits"], r["core"]["per1000"],
            r["extended"]["hits"], r["extended"]["per1000"],
            {k: v for k, v in r["core"]["families"].items() if v},
            {k: v for k, v in r["extended"]["families"].items() if v})


o, a = row(ORIG), row(ALT)
print(f"  {'':<8} {'bytes':>7} {'words':>7} {'CORE':>6} {'/1000':>8} {'EXT':>5} {'/1000':>8}")
for lbl, r in (("CIRIS", o), ("alt", a)):
    print(f"  {lbl:<8} {r[0]:>7} {r[1]:>7} {r[2]:>6} {r[3]:>8.1f} {r[4]:>5} {r[5]:>8.1f}")
db = 100.0 * (a[0] - o[0]) / o[0]
dw = 100.0 * (a[1] - o[1]) / o[1]
print(f"  {'delta':<8} {db:>+6.1f}% {dw:>+6.1f}%")
print(f"  families CIRIS core={o[6]} ext={o[7]}")
print(f"  families alt   core={a[6]} ext={a[7]}")
check(abs(db) <= 10.0 and abs(dw) <= 10.0, f"size congruence <=10% (got {db:+.1f}% B / {dw:+.1f}% w)")
print("  NOTE n=23/24 words: one token moves the rate ~42/1000. No per-1000 figure here")
print("       supports any inference. Read the families, not the rates.")

acc = measure("accord", open(ACCORD, encoding="utf-8").read())
print(f"\n  REFERENCE Accord 1.2b EN: {acc['total_words']:,} w  "
      f"core {acc['core']['per1000']:.2f}/1000  ext {acc['extended']['per1000']:.2f}/1000")

# ------------------------------------------------ 5. HARD CONSTRAINTS 1,2,3,5
print("\n--- 5. HARD CONSTRAINTS")

HC = (r"\b(bioweapon|biological weapon|chemical weapon|nuclear|radiological|uplift|"
      r"critical infrastructure|cyberweapon|malware|oversight|disempower|mass casualt|"
      r"seiz\w* power|csam|child sexual)\b")
hits = re.findall(HC, alt_low)
check(not hits, f"[1] no prohibition text (found {hits})")

BANNED = {
    "obligation": r"\b(obligation|obligations|obligated|obliged|duty|duties|stewardship|steward)\b",
    "integrity": r"\b(integrity)\b",
    "transparency": r"\b(transparent|transparency|opaque|opacity)\b",
    "responsibility": r"\b(responsible|responsibly|responsibility|responsibilities|accountable|accountability)\b",
    "dignity": r"\b(dignity|dignified|undignified)\b",
}
bad = {f: len(re.findall(p, alt_low)) for f, p in BANNED.items()}
check(not any(bad.values()), f"[2] no injected duty-bearer vocabulary {bad}")

# constraint 3: report, do not assert — see draft section 7 item 1
print("  [note] [3] register: CIRIS original is impersonal value-assertion, no modal.")
for lbl, txt in (("CIRIS", ORIG.lower()), ("alt", ALT.lower())):
    print(f"         {lbl:<6} should={len(re.findall(r'\bshould\b', txt))} "
          f"must={len(re.findall(r'\bmust\b', txt))} "
          f"we={len(re.findall(r'\bwe\b', txt))} "
          f"claude={len(re.findall(r'\bclaude\b', txt))} "
          f"you/your={len(re.findall(r'\b(you|your|yourself)\b', txt))}")
check(not re.search(r"\b(we|claude|should)\b", alt_low),
      "[3] no 3rd-person institutional source voice survives (we/Claude/should)")

TIERS = ["broadly safe", "broadly ethical", "compliant with anthropic", "genuinely helpful"]
check(not any(t in alt_low for t in TIERS), "[5] no tier language introduced")

print("\n" + "=" * 78)
print("RESULT:", "PASS" if not FAIL else f"FAIL ({len(FAIL)}): " + "; ".join(FAIL))
print("=" * 78)
sys.exit(1 if FAIL else 0)
