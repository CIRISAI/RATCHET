"""Battery metadata registry: channel id -> which battery run produced it.

The channel a battery trace carries is `safety_battery_<battery_id>_<run_id>`,
and both halves are fields of the battery's own `results.jsonl`. So a run's
evidence reconstructs its channel exactly, with no time-window guessing.

TWO SOURCES, durable first:

  1. CIRISAgent `qa_reports/safety_battery/<lang>_<domain>_<run_id>/results.jsonl`
     on main — committed by tools/harvest_safety_evidence.py because artifacts
     expire at 90 days. 295 run dirs back to 2026-05-11.
  2. Live Actions artifacts, for runs not yet harvested (or never harvested, as
     is the case for every RATCHET battery — see README).

Get source 1 with:
    git clone --depth 1 --filter=blob:none --sparse \
      https://github.com/CIRISAI/CIRISAgent.git <dir>
    cd <dir> && git sparse-checkout set qa_reports/safety_battery
then point QA_STORE at <dir>.
"""
import json, os, glob, collections, re
EXPORT   = os.environ.get("TRACE_EXPORT_DIR", "/home/emoore/ratchet-trace-export").rstrip("/") + "/"
WORK     = os.environ.get("TRACE_WORK_DIR", os.path.dirname(os.path.abspath(__file__))).rstrip("/") + "/"
QA_STORE = os.environ.get("QA_STORE", WORK + "qa_store").rstrip("/") + "/"
ART      = os.environ.get("ART_DIR", WORK + "art").rstrip("/") + "/"

def harvest(pattern, origin, REG):
    rows = 0
    for f in glob.glob(pattern, recursive=True):
        for l in open(f):
            l = l.strip()
            if not l: continue
            try: d = json.loads(l)
            except Exception: continue
            if 'battery_id' not in d or 'run_id' not in d: continue
            rows += 1
            ch = 'safety_battery_%s_%s' % (d['battery_id'], d['run_id'])
            r = REG.setdefault(ch, {'battery_id': d['battery_id'], 'run_id': d['run_id'],
                                    'origin': origin, 'cells': set(), 'questions': set(),
                                    'stages': set(), 'as_user': set(), 'n_rows': 0})
            r['n_rows'] += 1
            r['cells'].add(f.split('/')[-2])
            r['questions'].add(d.get('question_id', '')); r['stages'].add(d.get('stage', ''))
            if d.get('as_user'): r['as_user'].add(d['as_user'])
    return rows

REG = {}
n1 = harvest(QA_STORE + 'qa_reports/safety_battery/*/results.jsonl', 'durable_qa_reports', REG)
n_dur = len(REG)
n2 = harvest(ART + '**/results.jsonl', 'actions_artifact', REG)
print(f'durable rows: {n1} -> {n_dur} channels')
print(f'artifact rows: {n2} -> {len(REG) - n_dur} additional channels')
for r in REG.values():
    for k in ('cells', 'questions', 'stages', 'as_user'):
        r[k] = sorted(x for x in r[k] if x)
json.dump(REG, open(WORK + 'battery_registry.json', 'w'), indent=1)

# coverage against the export's resolved channels
res = json.load(open(WORK + 'channel_resolved.json'))
exp = collections.Counter(v for v in res.values() if v.startswith('safety_battery_'))
hit = sum(n for c, n in exp.items() if c in REG)
print(f'\nexport battery-channel traces: {sum(exp.values())}  explained: {hit}')
un = collections.Counter()
for c, n in exp.items():
    if c in REG: continue
    m = re.match(r'safety_battery_(.+?)_\d{8}T\d{6}Z$', c)
    b = m.group(1) if m else c
    un['ani' if b.startswith('ani') else 'he300' if b.startswith('he300')
       else 'mental_health' if 'mental_health' in b else 'harm/other'] += n
print('unexplained by family:', dict(un))
print('(ani/he300/harm are RATCHET batteries: no durable store exists — see README)')
