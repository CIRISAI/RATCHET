import json, collections
import os
EXPORT = os.environ.get("TRACE_EXPORT_DIR", "/home/emoore/ratchet-trace-export") .rstrip("/") + "/"
WORK   = os.environ.get("TRACE_WORK_DIR", os.path.dirname(os.path.abspath(__file__))).rstrip("/") + "/"


res=json.load(open(WORK+'channel_resolved.json'))
TR=[json.loads(l) for l in open(WORK+'traces.jsonl')]
reg=json.load(open(WORK+'battery_registry.json'))
out=[]; tier=collections.Counter(); ev=collections.Counter()
for r in TR:
    t=r['trace_id']; ch=res.get(t,'')
    direct = bool(r['channel'])
    e=[]
    if ch.startswith('safety_battery_'): e.append('channel_direct' if direct else 'channel_task_propagated')
    if r['agent']=='he-300-benchmark': e.append('agent_he300_benchmark')
    if ch.startswith('api_'): e.append('channel_api')
    if e and e[0].startswith('channel') and ch.startswith('safety_battery_'):
        tr_='synthetic'
    elif 'agent_he300_benchmark' in e: tr_='synthetic'
    elif 'channel_api' in e: tr_='unknown_api'
    else: tr_='unknown'
    bl=None
    if ch.startswith('safety_battery_'):
        import re
        m=re.match(r'safety_battery_(.+?)_(\d{8}T\d{6}Z)$',ch)
        if m: bl=m.group(1)
    rec={'trace_id':t,'split':tr_,'evidence':e,'channel':ch or None,'battery_label':bl,
         'agent':r['agent'],'trace_level':r['lvl'],'t0':r['t0'],'t1':r['t1'],'n_events':r['n_ev'],
         'ci_metadata':(reg.get(ch) or None) and {k:reg[ch][k] for k in ('battery_id','run_id','origin','cells','stages','as_user')}}
    out.append(rec); tier[tr_]+=1
    for x in e: ev[x]+=1
with open(WORK+'provenance_split.jsonl','w') as f:
    for r in out: f.write(json.dumps(r)+'\n')
n=len(out)
print('=== FINAL SPLIT (n=%d) ==='%n)
for k,v in tier.most_common(): print(f'  {v:5d}  {100*v/n:5.1f}%  {k}')
print('\n=== evidence counts (traces, overlapping) ===')
for k,v in ev.most_common(): print(f'  {v:5d}  {k}')
print('\n=== synthetic by battery family ===')
fam=collections.Counter()
for r in out:
    if r['split']!='synthetic': continue
    b=r['battery_label']
    fam['he300 (agent label)' if b is None else ('ani' if b.startswith('ani') else ('he300' if b.startswith('he300') else ('mental_health' if 'mental_health' in b else b)))]+=1
for k,v in fam.most_common(): print(f'  {v:5d}  {k}')
print('\n=== unknown remainder: who ===')
for k,v in collections.Counter((r['agent'],r['trace_level']) for r in out if r['split'].startswith('unknown')).most_common(10):
    print(f'  {v:5d}  agent={k[0]!r} level={k[1]}')
print('\n=== full_traces rows (highest sensitivity) by split ===')
print(dict(collections.Counter(r['split'] for r in out if r['trace_level']=='full_traces')))
print('with CI metadata attached:',sum(1 for r in out if r['ci_metadata']))
