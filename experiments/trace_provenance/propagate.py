import json, collections
import os
EXPORT = os.environ.get("TRACE_EXPORT_DIR", "/home/emoore/ratchet-trace-export") .rstrip("/") + "/"
WORK   = os.environ.get("TRACE_WORK_DIR", os.path.dirname(os.path.abspath(__file__))).rstrip("/") + "/"


side={}
for l in open(EXPORT+'trace_channel_ids.jsonl'):
    d=json.loads(l); side[d['trace_id']]=d['channel_id']
# trace -> set(task_id); task -> set(channel)
t2task=collections.defaultdict(set); task2ch=collections.defaultdict(set)
tr=set()
for l in open(EXPORT+'trace_events.jsonl'):
    d=json.loads(l); t=d['trace_id']; tk=d.get('task_id'); tr.add(t)
    if tk:
        t2task[t].add(tk)
        ch=side.get(t)
        if ch: task2ch[tk].add(ch)
print('traces:',len(tr),'| traces with task_id:',len(t2task),'| tasks with a channel:',len(task2ch))
# propagate
direct=sum(1 for t in tr if t in side)
prop={}
for t in tr:
    if t in side: continue
    chs=set()
    for tk in t2task.get(t,()): chs|=task2ch.get(tk,set())
    if len(chs)==1: prop[t]=next(iter(chs))
    elif len(chs)>1: prop[t]=('AMBIG',tuple(sorted(chs)))
amb={t:v for t,v in prop.items() if isinstance(v,tuple)}
good={t:v for t,v in prop.items() if not isinstance(v,tuple)}
print(f'direct channel: {direct} ({100*direct/len(tr):.1f}%)')
print(f'propagated via task_id: {len(good)}  (ambiguous, >1 channel per task: {len(amb)})')
tot=direct+len(good)
print(f'=> channel known for {tot} / {len(tr)} = {100*tot/len(tr):.1f}%')
allch={**{t:side[t] for t in tr if t in side}, **good}
bat=sum(1 for v in allch.values() if v.startswith('safety_battery_'))
print(f'   of those, safety_battery_*: {bat}  api_/other: {tot-bat}')
json.dump(allch, open('' + WORK + 'channel_resolved.json','w'))
