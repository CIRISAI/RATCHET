"""Exact trace -> battery join via signed-manifest audit anchors.

The strongest provenance evidence available, and it was hiding in the manifests.
`manifest_signed.json` carries `agent_audit_anchors`: one (question_id, agent_task_id)
pair per battery question, where agent_task_id is the agent's OWN task id — the same
value that appears as `task_id` on trace_events rows.

So a battery run names the exact tasks it created. Matching is an id equality test, not
a time window, not a distribution comparison. Negative control: 0 of 367 prod-agent
traces match any anchor.

It also recovers what the canonical corpus lost: `agent_version` (pipeline_metadata is
NULL on every row), plus model, template_id and the CI run id.
"""
import json, os, glob, collections
EXPORT = os.environ.get("TRACE_EXPORT_DIR", "/home/emoore/ratchet-trace-export").rstrip("/") + "/"
WORK   = os.environ.get("TRACE_WORK_DIR", os.path.dirname(os.path.abspath(__file__))).rstrip("/") + "/"
SOURCES = [p for p in (os.environ.get("MANIFEST_GLOBS") or "").split(":") if p] or [
    WORK + "art/**/manifest_signed.json",
    WORK + "qa_store/qa_reports/safety_battery/*/manifest_signed.json",
]

T2B = {}; nm = 0
for pat in SOURCES:
    for f in glob.glob(pat, recursive=True):
        try: d = json.load(open(f))
        except Exception: continue
        nm += 1
        for a in d.get('agent_audit_anchors') or []:
            tid = a.get('agent_task_id')
            if not tid or tid in ('None', None): continue
            T2B[tid] = {'battery_id': d.get('battery_id'), 'run_id': d.get('run_id'),
                        'agent_version': d.get('agent_version'), 'model': d.get('model'),
                        'template_id': d.get('template_id'),
                        'question_id': a.get('question_id'),
                        'ci_run': (d.get('ci_provenance') or {}).get('github_run_id'),
                        'repo': (d.get('ci_provenance') or {}).get('github_repository')}
print(f'manifests read: {nm} | distinct agent_task_ids: {len(T2B)}')

hit = {}
for l in open(EXPORT + 'trace_events.jsonl'):
    d = json.loads(l); tk = d.get('task_id')
    if tk and tk in T2B: hit[d['trace_id']] = T2B[tk]
print(f'traces matched by agent_task_id: {len(hit)}')
print('agent_version recovered:', dict(collections.Counter(v['agent_version'] for v in hit.values()).most_common(10)))
json.dump(hit, open(WORK + 'taskid_join.json', 'w'))
