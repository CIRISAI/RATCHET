import json, glob, collections, os
import os
EXPORT = os.environ.get("TRACE_EXPORT_DIR", "/home/emoore/ratchet-trace-export") .rstrip("/") + "/"
WORK   = os.environ.get("TRACE_WORK_DIR", os.path.dirname(os.path.abspath(__file__))).rstrip("/") + "/"


REG={}
files=glob.glob(WORK+'art/bulk/**/results.jsonl',recursive=True)+glob.glob(WORK+'art/mh3_31920708737/**/results.jsonl',recursive=True)
nrow=0
for f in files:
    parts=f.split('/'); cell=parts[-2]
    run=[p for p in parts if p.startswith(('RATCHET_','CIRISAgent_'))]
    run=run[0] if run else 'mh3_31920708737'
    for l in open(f):
        try: d=json.loads(l)
        except Exception: continue
        if 'battery_id' not in d or 'run_id' not in d: continue
        nrow+=1
        ch='safety_battery_%s_%s'%(d['battery_id'],d['run_id'])
        r=REG.setdefault(ch,{'channel':ch,'battery_id':d['battery_id'],'run_id':d['run_id'],
            'ci_run':run,'cells':set(),'questions':set(),'stages':set(),'as_user':set()})
        r['cells'].add(cell); r['questions'].add(d.get('question_id','')); r['stages'].add(d.get('stage',''))
        if d.get('as_user'): r['as_user'].add(d['as_user'])
print('artifact rows read:',nrow,'| distinct battery channels in registry:',len(REG))
for r in REG.values():
    for k in ('cells','questions','stages','as_user'): r[k]=sorted(x for x in r[k] if x)
with open(WORK+'battery_registry.json','w') as f: json.dump(REG,f,indent=1)
# which export channels does the registry explain?
resolved=json.load(open(WORK+'channel_resolved.json'))
exp_bat=collections.Counter(v for v in resolved.values() if v.startswith('safety_battery_'))
inreg=sum(n for c,n in exp_bat.items() if c in REG)
print(f'export battery-channel traces: {sum(exp_bat.values())}  explained by CI registry: {inreg}')
unexp=sorted(((n,c) for c,n in exp_bat.items() if c not in REG),reverse=True)
print(f'battery channels NOT in registry: {len(unexp)} channels / {sum(n for n,_ in unexp)} traces')
for n,c in unexp[:8]: print(f'   {n:4d}  {c}')
