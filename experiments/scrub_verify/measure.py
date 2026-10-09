"""Over-redaction rate on a fixed sample of real narrative strings, per backbone.

Deterministic sample: the first N (default 300) detailed/full_traces events in the export
that carry narrative text. Run under different CIRISLENS_NER_* env settings to compare
backbones on the identical input. Run with `python3 -I`.
"""
import json, os, re, sys, time, difflib, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ciris_server as cs
from reference import PLACEHOLDER, walk_string_leaves
from verify import verify, summarize

EXPORT = os.environ.get("TRACE_EXPORT_DIR", "/home/emoore/ratchet-trace-export").rstrip("/") + "/"
N = int(os.environ.get("MEASURE_N", "300"))
NARR = ("intervention_recommendation", "conscience_override_reason", "defer_reason",
        "completion_reason", "epistemic_humility_uncertainties", "original_reasoning",
        "final_reasoning", "next_best_recovery_step")

def narrative(p):
    return [(path, s) for path, s in walk_string_leaves(p)
            if len(s) >= 40 and any(path.replace("[]", "").endswith(n) for n in NARR)]

samp = []
for l in open(EXPORT + "trace_events.jsonl"):
    d = json.loads(l)
    if d["trace_level"] not in ("detailed", "full_traces"):
        continue
    try:
        p = json.loads(d["payload"])
    except Exception:
        continue
    if narrative(p):
        samp.append((d["event_type"], p))
    if len(samp) >= N:
        break
wrap = lambda et, p: json.dumps({"trace_id": "t", "events": [{"event_type": et, "data": p}]}, ensure_ascii=False)
t0 = time.time()
res = cs.scrub_traces_batch([wrap(et, p) for et, p in samp], "full_traces")
dt = time.time() - t0
n_str = n_touched = n_full = 0
tags = collections.Counter(); spans = collections.Counter(); reps = []
for (et, p), r in zip(samp, res):
    out = json.loads(r["trace"])["events"][0]["data"]
    reps.append(verify(p, out))
    # after-side: no length filter, or heavily redacted strings vanish from the denominator
    amap = {path: v for path, v in walk_string_leaves(out)}
    for path, before in narrative(p):
        after = amap.get(path)
        if after is None:
            continue
        n_str += 1
        found = PLACEHOLDER.findall(after)
        if not found:
            continue
        n_touched += 1
        for g in found:
            tags[g[0] or g[2]] += 1
        if not PLACEHOLDER.sub("", after).strip(" .,;:()[]-—'\""):
            n_full += 1
        for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, before, after, autojunk=False).get_opcodes():
            if op == "replace":
                src = before[i1:i2].strip(); m = PLACEHOLDER.search(after[j1:j2])
                if m and 0 < len(src) <= 60:
                    spans[(src, m.group(1) or m.group(3))] += 1
s = summarize(reps)
print(f"backbone={os.environ.get('CIRISLENS_NER_BACKBONE', '<default>')}  events={len(samp)}  {dt:.1f}s  {1000*dt/len(samp):.0f} ms/event")
print(f"narrative strings: {n_str} | with placeholder: {n_touched} ({100*n_touched/max(n_str,1):.1f}%) | fully replaced: {n_full}")
print(f"tags: {dict(tags.most_common())}")
print(f"verifier: mid_token={s['mid_token_placeholders']} structural_violations={s['structural_violations']} year_residue={s['year_residue']} unknown={s['unknown_placeholders']}")
print("top replaced spans:")
for (src, tag), n in spans.most_common(25):
    print(f"  {n:3d}  {tag:5}  {src!r}")
