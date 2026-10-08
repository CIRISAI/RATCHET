"""Collapse the 105k-row event stream into one row per trace.

First stage of the provenance split: trace_id -> agent, time span, event count,
trace_level, plus the sidecar channel if the export recovered one.
"""
import json, os, collections
EXPORT = os.environ.get("TRACE_EXPORT_DIR", "/home/emoore/ratchet-trace-export").rstrip("/") + "/"
WORK   = os.environ.get("TRACE_WORK_DIR", os.path.dirname(os.path.abspath(__file__))).rstrip("/") + "/"

side = {}
for l in open(EXPORT + 'trace_channel_ids.jsonl'):
    d = json.loads(l); side[d['trace_id']] = d.get('channel_id') or ''

T = {}; n = 0
for l in open(EXPORT + 'trace_events.jsonl'):
    d = json.loads(l); n += 1
    t = d['trace_id']; ts = d['ts']; r = T.get(t)
    if r is None:
        T[t] = {'trace_id': t, 'agent': d.get('agent_name'), 't0': ts, 't1': ts, 'n_ev': 1,
                'tmpl': d.get('agent_template'), 'lvl': d.get('trace_level'),
                'dom': d.get('deployment_domain'), 'dtype': d.get('deployment_type')}
    else:
        if ts < r['t0']: r['t0'] = ts
        if ts > r['t1']: r['t1'] = ts
        r['n_ev'] += 1
        if r['agent'] is None: r['agent'] = d.get('agent_name')
for t, r in T.items(): r['channel'] = side.get(t, '')
with open(WORK + 'traces.jsonl', 'w') as f:
    for r in T.values(): f.write(json.dumps(r) + '\n')
print('events read:', n, '| traces:', len(T))
print('agents:', dict(collections.Counter(r['agent'] for r in T.values()).most_common()))
print('levels:', dict(collections.Counter(r['lvl'] for r in T.values()).most_common()))
