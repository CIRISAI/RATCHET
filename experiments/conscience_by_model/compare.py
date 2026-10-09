"""Conscience results and overrides by model, within battery family.

Scout-vs-others is confounded with battery: Scout ran he300 almost exclusively; Qwen,
Maverick, Gemma, Mistral, Llama-3.3 ran mental_health and ANI. So comparisons are made
WITHIN family, and overrides are split into JUDGED (a faculty flag is False) versus
NON-EXECUTION/unattributed (no faculty flag False — the "DID NOT RUN" path of
CIRISAgent#1247, which the pipeline counts as an override and resolves to PONDER).

Also establishes from trace_llm_calls that the four conscience faculties run on the SAME
model as the action in 95.5% of traces, so cross-model "strictness" is each model's
self-judgement calibration, not an external judge. Run with python3 -I.
"""
import json, os, collections, random, statistics as st
random.seed(11)
EX = os.environ.get("TRACE_EXPORT_DIR", "/home/emoore/ratchet-trace-export").rstrip("/") + "/"
prov = {}
for l in open(EX + "provenance_split.jsonl"):
    d = json.loads(l); prov[d["trace_id"]] = (d["split"], d.get("battery_label") or "", d.get("agent"))
models = collections.defaultdict(set); by_trace = collections.defaultdict(lambda: collections.defaultdict(set))
for l in open(EX + "trace_llm_calls.jsonl"):
    d = json.loads(l); m = (d.get("model") or "").lower()
    if m: models[d["trace_id"]].add(m); by_trace[d["trace_id"]][d.get("handler_name") or "?"].add(m)
def mclass(t):
    ms = " ".join(models.get(t, ()))
    for k in ("scout", "maverick", "qwen", "gemma-4", "gemma", "mistral", "llama-3.3", "gpt-4o"):
        if k in ms: return k
    return "other" if ms else "none"
def fam(t):
    s, b, a = prov.get(t, ("?", "", None))
    if a == "he-300-benchmark" or b.startswith("he300"): return "he300"
    if "mental_health" in b: return "mental_health"
    if b.startswith("ani"): return "ani"
    return "prod" if s == "other_agent" else "end_user" if s.startswith("end_user") else s
FAC = {"entropy_passed": "entropy", "coherence_passed": "coherence", "optimization_veto_passed": "opt_veto", "epistemic_humility_passed": "humility"}
C = collections.defaultdict(list); S = collections.defaultdict(list)
for l in open(EX + "trace_events.jsonl"):
    d = json.loads(l)
    if d["event_type"] not in ("CONSCIENCE_RESULT", "ASPDMA_RESULT"): continue
    try: p = json.loads(d["payload"])
    except Exception: continue
    key = (fam(d["trace_id"]), mclass(d["trace_id"]))
    (C if d["event_type"] == "CONSCIENCE_RESULT" else S)[key].append(p)
def classify(p):
    r = (p.get("conscience_override_reason") or "").lower()
    if "did not run" in r or "all llm services failed" in r: return "not_run"
    failed = [FAC[k] for k in FAC if p.get(k) is False]
    if failed: return "judged:" + "+".join(failed)
    if p.get("thought_depth_triggered"): return "depth"
    return "not_run_or_unattributed"
def ci(xs, B=1000):
    if len(xs) < 5: return ""
    bs = sorted(100*sum(random.choices(xs, k=len(xs)))/len(xs) for _ in range(B)); return f"[{bs[25]:.0f},{bs[975]:.0f}]"
def rate(ps, k):
    v = [p[k] for p in ps if isinstance(p.get(k), bool)]; return f"{100*sum(v)/len(v):.0f}% ({len(v)})" if v else "-"
def mean(ps, k):
    v = [p[k] for p in ps if isinstance(p.get(k), (int, float)) and not isinstance(p.get(k), bool)]; return f"{st.mean(v):.2f}" if v else "-"
same = diff = 0
CONS = {"coherence_conscience", "optimization_veto_conscience", "entropy_conscience", "epistemic_humility_conscience"}
for t, hs in by_trace.items():
    cm = set().union(*[v for h, v in hs.items() if h in CONS]) if any(h in CONS for h in hs) else set()
    am = set().union(*[v for h, v in hs.items() if h not in CONS]) if any(h not in CONS for h in hs) else set()
    if cm and am: same += cm == am; diff += cm != am
out = []
out.append(f"Conscience judged by the same model as the action: {same} of {same+diff} traces ({100*same/(same+diff):.1f}%); the rest fall back to another provider for some faculty.\n")
out.append("| family | model | n | override | judged [CI] | not-run/unattr [CI] | judged by faculty | humility pass | certainty | opt-veto pass | entropy pass | coherence pass | ponder (ASPDMA) |")
out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
order = ["ani", "mental_health", "prod", "end_user", "he300", "synthetic", "unknown"]
for f in order:
    for (ff, m), ps in sorted(C.items(), key=lambda x: -len(x[1])):
        if ff != f or len(ps) < 20: continue
        ov = [p for p in ps if p.get("action_was_overridden")]
        cl = [classify(p) for p in ov]
        jx = [1 if (p.get("action_was_overridden") and (classify(p).startswith("judged") or classify(p) == "depth")) else 0 for p in ps]
        nx = [1 if (p.get("action_was_overridden") and not (classify(p).startswith("judged") or classify(p) == "depth")) else 0 for p in ps]
        byf = collections.Counter(fc for c in cl if c.startswith("judged:") for fc in c[7:].split("+"))
        sp = S.get((ff, m), []); pond = f"{100*sum(1 for p in sp if 'PONDER' in str(p.get('selected_action')))/len(sp):.0f}% ({len(sp)})" if sp else "-"
        out.append(f"| {f} | {m} | {len(ps)} | {100*len(ov)/len(ps):.1f}% | {100*sum(jx)/len(ps):.1f}% {ci(jx)} | {100*sum(nx)/len(ps):.1f}% {ci(nx)} | {', '.join(f'{k} {v}' for k, v in byf.most_common())} | {rate(ps,'epistemic_humility_passed')} | {mean(ps,'epistemic_humility_certainty')} | {rate(ps,'optimization_veto_passed')} | {rate(ps,'entropy_passed')} | {rate(ps,'coherence_passed')} | {pond} |")
text = "\n".join(out)
print(text)
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tables.md"), "w").write(text + "\n")
