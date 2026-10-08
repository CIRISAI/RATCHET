"""Smoke test: the vendored contract against the shipped wheel, on REAL export payloads.

  A. wheel `detailed` vs reference regex_scrub — should agree everywhere except structural keys
  B. wheel `full_traces` through verify() — structural violations, year residue, mid-token count
Run with `python3 -I` (reads untrusted export data). Needs ciris-server in the venv.
"""
import json, os, sys, time, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ciris_server as cs
from reference import regex_scrub, walk_string_leaves, STRUCTURAL_IDENTIFIER_KEYS
from verify import verify, summarize

EXPORT = os.environ.get("TRACE_EXPORT_DIR", "/home/emoore/ratchet-trace-export").rstrip("/") + "/"
N = int(os.environ.get("SMOKE_N", "60"))

samp = []
for l in open(EXPORT + "trace_events.jsonl"):
    d = json.loads(l)
    if d["trace_level"] not in ("detailed", "full_traces"):
        continue
    try:
        samp.append((d["event_type"], json.loads(d["payload"])))
    except Exception:
        pass
    if len(samp) >= N:
        break
wrap = lambda et, p: json.dumps({"trace_id": "t", "events": [{"event_type": et, "data": p}]}, ensure_ascii=False)

# A. detailed: wheel vs reference
res = cs.scrub_traces_batch([wrap(et, p) for et, p in samp], "detailed")
agree = disagree_struct = disagree_other = 0; examples = []
for (et, p), r in zip(samp, res):
    wheel = json.loads(r["trace"])["events"][0]["data"]
    ref = regex_scrub(p)
    w = dict(walk_string_leaves(wheel)); f = dict(walk_string_leaves(ref))
    for path, sv in f.items():
        wv = w.get(path)
        if wv == sv:
            agree += 1
        else:
            key = path.replace("[]", "").rsplit(".", 1)[-1]
            if key in STRUCTURAL_IDENTIFIER_KEYS:
                disagree_struct += 1
            else:
                disagree_other += 1
                if len(examples) < 5:
                    examples.append((path, sv[:70], (wv or "")[:70]))
print(f"A. detailed  strings={agree+disagree_struct+disagree_other}  agree={agree}  "
      f"differ@structural={disagree_struct}  differ@other={disagree_other}")
for e in examples:
    print("   non-structural difference:", e)

# B. full_traces through the verifier
t0 = time.time()
res = cs.scrub_traces_batch([wrap(et, p) for et, p in samp], "full_traces")
dt = time.time() - t0
reps = [verify(p, json.loads(r["trace"])["events"][0]["data"]) for (et, p), r in zip(samp, res)]
s = summarize(reps)
print(f"B. full_traces  {len(samp)} events in {dt:.1f}s ({1000*dt/len(samp):.0f} ms/event)")
for k, v in s.items():
    print(f"   {k}: {v}")
mt = [m for r in reps for m in r["mid_token"]]
for path, ctx in mt[:6]:
    print("   mid-token:", path.rsplit(".", 1)[-1], repr(ctx))
