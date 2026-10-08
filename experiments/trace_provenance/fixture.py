import os
import json, glob, collections
EX=os.environ.get('TRACE_EXPORT_DIR','/home/emoore/ratchet-trace-export').rstrip('/')+'/'
SP=os.environ.get('TRACE_WORK_DIR', os.path.dirname(os.path.abspath(__file__))).rstrip('/')+'/'
# fixture identities from the durable battery store
names=collections.Counter(); users=collections.Counter()
for f in glob.glob(SP+'qa_store/qa_reports/safety_battery/*/results.jsonl'):
    for l in open(f):
        try: d=json.loads(l)
        except Exception: continue
        if d.get('as_display_name'): names[d['as_display_name'].strip()]+=1
        if d.get('as_user'): users[d['as_user'].strip()]+=1
print('distinct fixture display names:',len(names))
print(' ',dict(names.most_common(20)))
print('distinct fixture as_user ids:',len(users))
print(' ',dict(list(users.most_common(8))))
FX={n for n in names if len(n)>=3}
split={json.loads(l)['trace_id']:json.loads(l) for l in open(SP+'provenance_split.jsonl')}
def cls(t):
    r=split.get(t)
    if not r: return 'other'
    a=r['agent']; ch=r['channel'] or ''
    if a=='he-300-benchmark': return 'QA_he300'
    if ch.startswith('safety_battery_'): return 'QA_battery_ch'
    if a in ('Datum','echo-core','echo-speculative'): return 'PROD_agents'
    if a=='Ally': return 'UNRES_ally'
    return 'UNRES_null'
hit=collections.defaultdict(collections.Counter); traces=collections.defaultdict(set)
for l in open(EX+'trace_events.jsonl'):
    d=json.loads(l); t=d['trace_id']; c=cls(t)
    p=d['payload']
    for n in FX:
        if n in p:
            hit[c][n]+=1; traces[c].add(t)
tot=collections.Counter()
for t in split: tot[cls(t)]+=1
print('\n=== traces whose payload names a KNOWN battery fixture ===')
for c in ['QA_he300','QA_battery_ch','PROD_agents','UNRES_ally','UNRES_null']:
    n=len(traces[c])
    print(f'  {c:16} {n:5d} / {tot[c]:5d} = {100*n/max(tot[c],1):5.1f}%   names: {dict(hit[c].most_common(5))}')
json.dump({c:sorted(v) for c,v in traces.items()}, open(SP+'fixture_hits.json','w'))
