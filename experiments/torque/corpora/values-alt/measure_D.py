import re, sys, yaml
sys.path.insert(0, "/tmp/claude-1000/-home-emoore-RATCHET/4fdbd195-6bf1-45c9-8ffc-931540da4e4d/scratchpad")
from density import measure

ORIG = yaml.safe_load(open('/tmp/a2911/ciris_engine/logic/dma/prompts/action_selection_pdma.yml'))
MD = open('/home/emoore/RATCHET/experiments/torque/corpora/values-alt/D-aspdma.md').read()

blocks = re.findall(r"<!-- BEGIN REPLACEMENT (\S+) -->\s*```yaml\n(.*?)```\s*<!-- END REPLACEMENT -->", MD, re.S)
NEW = {}
for name, body in blocks:
    NEW.update(yaml.safe_load(body))
assert set(NEW) == {"action_params_defer_guidance", "closing_reminder", "csdma_ambiguity_guidance"}, NEW.keys()

SLOT_FIELDS = ['action_params_defer_guidance','final_ponder_advisory','action_parameter_schemas',
               'action_params_speak_csdma_guidance','action_params_ponder_guidance',
               'action_params_observe_guidance','reasoning_csdma_guidance']

def compose(src, mapping):
    return "\n".join(mapping.get(k, src[k]) for k in SLOT_FIELDS)

rows = []
def add(label, orig, new):
    o = measure(label+" [CIRIS]", orig); n = measure(label+" [alt]", new)
    rows.append((label, orig, new, o, n))

# per-key
add("action_params_defer_guidance (the #slots axiotic payload)",
    ORIG['action_params_defer_guidance'], NEW['action_params_defer_guidance'])
add("context_integration#slots (authorable/deterministic portion)",
    compose(ORIG, {}), compose(ORIG, NEW))
add("closing_reminder", ORIG['closing_reminder'], NEW['closing_reminder'])
add("csdma_ambiguity_guidance", ORIG['csdma_ambiguity_guidance'], NEW['csdma_ambiguity_guidance'])
# unit total (authorable/deterministic)
uo = compose(ORIG, {}) + "\n" + ORIG['closing_reminder'] + "\n" + ORIG['csdma_ambiguity_guidance']
un = compose(ORIG, NEW) + "\n" + NEW['closing_reminder'] + "\n" + NEW['csdma_ambiguity_guidance']
add("UNIT D TOTAL (authorable/deterministic)", uo, un)

hdr = f"{'block':<58} {'B_orig':>7} {'B_new':>7} {'x':>5} | {'w_o':>5} {'w_n':>5} | {'core_o':>7} {'core_n':>7} | {'ext_o':>7} {'ext_n':>7}"
print(hdr); print("-"*len(hdr))
for label, o, n, mo, mn in rows:
    print(f"{label:<58} {len(o.encode()):>7} {len(n.encode()):>7} {len(n.encode())/len(o.encode()):>5.2f} | "
          f"{mo['total_words']:>5} {mn['total_words']:>5} | "
          f"{mo['core']['per1000']:>7.2f} {mn['core']['per1000']:>7.2f} | "
          f"{mo['extended']['per1000']:>7.2f} {mn['extended']['per1000']:>7.2f}")

print("\nHITS (absolute) — unit total")
_,o,n,mo,mn = rows[-1]
print(f"  core     : CIRIS {mo['core']['hits']}  -> alt {mn['core']['hits']}")
print(f"  extended : CIRIS {mo['extended']['hits']}  -> alt {mn['extended']['hits']}")
print("\n  family breakdown (unit total, hits): family | CIRIS | alt")
allfam = list(mo['extended']['families'])
for f in allfam:
    a,b = mo['extended']['families'][f], mn['extended']['families'][f]
    if a or b: print(f"    {f:<16} {a:>4} {b:>4}")

print("\nREFERENCE corpora (whole-document, per 1000 words)")
acc = measure("accord", open('/tmp/a2911/ciris_engine/data/localized/accord_1.2b_en.txt').read())
con = measure("consti", open('consti_official.txt').read())
for r in (acc, con):
    print(f"  {r['name']:<10} words={r['total_words']:>6}  core={r['core']['per1000']:.2f}  ext={r['extended']['per1000']:.2f}")

print("\nREGISTER counts (unit total)")
for lab, t in (("CIRIS", uo), ("alt", un)):
    w = len(re.findall(r"[A-Za-z][A-Za-z'’\-]*", t))
    must = len(re.findall(r"\bmust\b", t, re.I)); should = len(re.findall(r"\bshould\b", t, re.I))
    you = len(re.findall(r"\b(you|your|yours)\b", t, re.I))
    third = len(re.findall(r"\b(Claude|the agent|it|its)\b", t, re.I))
    print(f"  {lab:<6} words={w:>4}  must={must} ({1000*must/w:.2f}/1000)  should={should} ({1000*should/w:.2f}/1000)  2nd-person={you}  3rd-person(Claude/agent/it)={third}")

print("\nPROHIBITION scan on the alt draft (constraint 3)")
BAD = ["bioweapon","biological weapon","chemical weapon","nuclear","radiological","uplift",
       "critical infrastructure","cyberweapon","malicious code","undermin","mass casualt",
       "disempower","seiz","CSAM","child sexual","hard constraint"]
hits = [(t, len(re.findall(t, un, re.I))) for t in BAD if re.search(t, un, re.I)]
print("  hits:", hits if hits else "NONE")
