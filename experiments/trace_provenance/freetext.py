import os
import json, collections
EX=os.environ.get('TRACE_EXPORT_DIR','/home/emoore/ratchet-trace-export').rstrip('/')+'/'
SP=os.environ.get('TRACE_WORK_DIR', os.path.dirname(os.path.abspath(__file__))).rstrip('/')+'/'
split={json.loads(l)['trace_id']:json.loads(l) for l in open(SP+'provenance_split.jsonl')}
# field -> (count, total bytes, max len, level set, sample)
F=collections.defaultdict(lambda:{'n':0,'b':0,'max':0,'lvl':set(),'ex':''})
def walk(o, path, lvl):
    if isinstance(o,dict):
        for k,v in o.items(): walk(v, path+'.'+k if path else k, lvl)
    elif isinstance(o,list):
        for v in o[:3]: walk(v, path+'[]', lvl)
    elif isinstance(o,str) and len(o)>=80:
        f=F[path]; f['n']+=1; f['b']+=len(o); f['lvl'].add(lvl)
        if len(o)>f['max']: f['max']=len(o); f['ex']=o[:150]
for l in open(EX+'trace_events.jsonl'):
    d=json.loads(l)
    lvl=d.get('trace_level')
    try: p=json.loads(d['payload'])
    except Exception: continue
    walk(p,'payload',lvl)
    for col in ('extracted_features','classifications','pipeline_metadata'):
        v=d.get(col)
        if isinstance(v,str) and v:
            try: walk(json.loads(v), col, lvl)
            except Exception: walk(v, col, lvl)
rows=sorted(F.items(), key=lambda x:-x[1]['b'])
print(f'{"field":46} {"rows":>6} {"MB":>7} {"maxlen":>7}  levels')
for k,v in rows[:22]:
    print(f'{k[:46]:46} {v["n"]:>6} {v["b"]/1e6:>7.2f} {v["max"]:>7}  {sorted(v["lvl"])}')
print('\n=== samples of the biggest free-text fields ===')
for k,v in rows[:6]:
    print(f'\n--- {k}  ({sorted(v["lvl"])})\n    {v["ex"]!r}')
