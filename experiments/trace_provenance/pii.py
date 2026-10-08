import os
import json, collections
EX=os.environ.get('TRACE_EXPORT_DIR','/home/emoore/ratchet-trace-export').rstrip('/')+'/'
SP=os.environ.get('TRACE_WORK_DIR', os.path.dirname(os.path.abspath(__file__))).rstrip('/')+'/'
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
# the narrative free-text fields (exclude signatures/attestation boilerplate)
NARR={'payload.intervention_recommendation','payload.next_best_recovery_step',
      'payload.conscience_override_reason','payload.verb_specific_data.defer_reason',
      'payload.original_reasoning','payload.final_reasoning',
      'payload.epistemic_humility_uncertainties[]','payload.action_parameters.completion_reason',
      'payload.action_parameters.questions[]','payload.action_parameters.content',
      'payload.correlation_factors[]','payload.sources_identified[]'}
def walk(o,path,out):
    if isinstance(o,dict):
        for k,v in o.items(): walk(v,path+'.'+k if path else k,out)
    elif isinstance(o,list):
        for v in o: walk(v,path+'[]',out)
    elif isinstance(o,str) and len(o)>=40 and path in NARR: out.append((path,o))
tr_narr=collections.defaultdict(lambda: collections.Counter())
bytes_by=collections.Counter(); traces_with=collections.defaultdict(set)
lvl_by=collections.defaultdict(collections.Counter)
for l in open(EX+'trace_events.jsonl'):
    d=json.loads(l); t=d['trace_id']; c=cls(t)
    try: p=json.loads(d['payload'])
    except Exception: continue
    out=[]; walk(p,'payload',out)
    if not out: continue
    traces_with[c].add(t)
    for path,s in out:
        tr_narr[c][path]+=1; bytes_by[c]+=len(s); lvl_by[c][d['trace_level']]+=1
tot=collections.Counter()
for t in split: tot[cls(t)]+=1
print(f'{"class":16} {"traces":>7} {"w/ narrative text":>18} {"%":>6} {"KB text":>9}  levels')
for c in ['QA_he300','QA_battery_ch','PROD_agents','UNRES_ally','UNRES_null']:
    n=len(traces_with[c])
    print(f'{c:16} {tot[c]:>7} {n:>18} {100*n/max(tot[c],1):>5.1f}% {bytes_by[c]/1000:>8.1f}  {dict(lvl_by[c])}')
print('\n=== which narrative fields, by class ===')
for c in ['QA_he300','QA_battery_ch','PROD_agents','UNRES_ally','UNRES_null']:
    print(f'  {c}: {dict(tr_narr[c].most_common(5))}')
