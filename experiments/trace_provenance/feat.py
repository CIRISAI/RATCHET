import os
import json, collections
EX=os.environ.get('TRACE_EXPORT_DIR','/home/emoore/ratchet-trace-export').rstrip('/')+'/'
SP=os.environ.get('TRACE_WORK_DIR', os.path.dirname(os.path.abspath(__file__))).rstrip('/')+'/'
split={json.loads(l)['trace_id']: json.loads(l) for l in open(SP+'provenance_split.jsonl')}
F=collections.defaultdict(lambda: {'models':set(),'base':set(),'handler':set(),'svc':set(),
    'idhash':set(),'dom':set(),'dtype':set(),'region':set(),'trust':set(),'cog':set(),
    'ver':set(),'cohort':set(),'schema':set(),'tmpl':set(),'ncalls':0,'tok':0})
for l in open(EX+'trace_llm_calls.jsonl'):
    d=json.loads(l); f=F[d['trace_id']]
    f['ncalls']+=1; f['tok']+=(d.get('prompt_tokens') or 0)+(d.get('completion_tokens') or 0)
    for k,c in (('model','models'),('base_url','base'),('handler_name','handler'),('service_name','svc')):
        if d.get(k): f[c].add(d[k])
for l in open(EX+'trace_events.jsonl'):
    d=json.loads(l); f=F[d['trace_id']]
    for k,c in (('agent_id_hash','idhash'),('deployment_domain','dom'),('deployment_type','dtype'),
                ('deployment_region','region'),('deployment_trust_mode','trust'),('cognitive_state','cog'),
                ('verification_source','ver'),('cohort_scope','cohort'),('schema_version','schema'),
                ('agent_template','tmpl')):
        if d.get(k) is not None: f[c].add(str(d[k]))
def cls(t):
    r=split[t]; a=r['agent']; ch=r['channel'] or ''
    if a=='he-300-benchmark': return 'REF_he300'
    if ch.startswith('safety_battery_'): return 'REF_battery_ch'
    if a in ('Datum','echo-core','echo-speculative'): return 'REF_prod'
    if a=='Ally': return 'Q_ally'
    return 'Q_null'
groups=collections.defaultdict(list)
for t in split: groups[cls(t)].append(t)
print('group sizes:',{k:len(v) for k,v in sorted(groups.items())})
def dist(g,key,top=6):
    c=collections.Counter()
    for t in groups[g]:
        vs=F[t][key]
        c[('|'.join(sorted(vs)) if vs else '<none>')[:60]]+=1
    n=len(groups[g])
    return [(v,round(100*k/n,1),k) for v,k in c.most_common(top)]
for key in ['models','base','dom','dtype','trust','cog','ver','tmpl','svc']:
    print(f'\n--- {key} ---')
    for g in ['REF_battery_ch','REF_prod','Q_ally','Q_null','REF_he300']:
        print(f'  {g:16}', dist(g,key,4))
json.dump({t:{k:(sorted(v) if isinstance(v,set) else v) for k,v in f.items()} for t,f in F.items()},
          open(SP+'features.json','w'))
