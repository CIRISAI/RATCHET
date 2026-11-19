"""Check a context-map run (M4_CTXMAP=1, mock LLM) against the context policy, using the per-call manifests.

For every LLM call in the run's reasoning streams: attribute it to a scenario (via the harness's ctxmap_step task
ids), then check its manifest (context_manifest on the llm_call event) against context_policy.yaml, and check the
manifest against the bytes actually sent (the differential, FSD §8).

Usage: python3 ctxmap_check.py <run_dir> <context_policy.yaml> [--json out.json]
"""
import collections
import glob
import hashlib
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_contract_policy import PRED_SCEN  # noqa: E402  (one definition of the predicates)

# ctxmap step labels -> model scenario. The WA's own rule / correction messages are WA chat (S3, advisory).
LABEL_SCEN = {"S1": "S1", "S2": "S2", "S3r": "S3", "S3": "S3", "S4c": "S3", "S4": "S4", "S5": "S5", "S6": "S6",
              "DEPTH": "S1"}
RESIDUAL_KINDS = {"draft", "schema_instruction"}
# --locale: the script a localized stage_instruction must contain. Some engine fields stay English in every locale
# by design or translation lag (pipeline_position, stage_contract, *_tail, context_integration ...), so the check is
# "the localized template fields rendered" (enough target-script characters), not "no English anywhere".
SCRIPTS = {"am": (0x1200, 0x137F), "ar": (0x0600, 0x06FF), "zh": (0x4E00, 0x9FFF), "hi": (0x0900, 0x097F),
           "ru": (0x0400, 0x04FF), "ja": (0x3040, 0x30FF), "ko": (0xAC00, 0xD7AF), "th": (0x0E00, 0x0E7F)}
MIN_SCRIPT_CHARS = 200
# The optimization veto is deliberately ONE polyglot prompt for every locale (CIRISAgent 0c6a962f1; per-locale
# copies fail tests/test_conscience_prompt_coverage.py), so it is exempt from the locale check.
POLYGLOT_STAGES = {"optveto"}


def load_policy(path):
    P = yaml.safe_load(open(path))
    stages = {}
    for st, rows in P["stages"].items():
        stages[st] = {}
        for key, cell in dict(rows).items():
            if isinstance(cell, dict):
                stages[st][key] = (cell.get("when", "always"), cell.get("form", "full"))
            else:
                stages[st][key] = (cell, "full")
    return P, stages


def tasks_of(raw):
    t = {}
    for line in open(raw):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("event") == "ctxmap_step" and r.get("task_id") and r.get("label") in LABEL_SCEN:
            t[r["task_id"]] = r["label"]
    return t


def key_of(seg):
    return f"{seg['kind']}.{seg['provenance']}" if seg.get("kind") == "rule" and seg.get("provenance") else seg["kind"]


def main(run_dir, policy_path, out=None, locale=None):
    P, stages = load_policy(policy_path)
    ablated = set(P["ablated_stages"])
    findings = collections.defaultdict(collections.Counter)
    examples = collections.defaultdict(list)
    seen = collections.Counter()
    present = collections.defaultdict(collections.Counter)

    def flag(kind, detail, ex):
        findings[kind][detail] += 1
        if len(examples[(kind, detail)]) < 2:
            examples[(kind, detail)].append(ex)

    for raw in glob.glob(f"{run_dir}/m4-raw-*/*.jsonl"):
        cid = os.path.basename(raw)[:-6]
        tasks = tasks_of(raw)
        probe_texts = []
        for line in open(raw):
            r = json.loads(line)
            if r.get("event") == "send" and str(r.get("label", "")).startswith("ctxmap-"):
                probe_texts.append((r.get("label"), str(r.get("message") or "")))
        for stream in glob.glob(f"{run_dir}/m4-agent-logs-{cid}/qa_reports/correction_episodes/*stream*/reasoning_stream.jsonl"):
            for line in open(stream):
                if '"llm_call"' not in line:
                    continue
                e = json.loads(line)
                label = tasks.get(e.get("task_id"))
                if not label:
                    continue
                scen = LABEL_SCEN[label]
                man = e.get("context_manifest")
                if isinstance(man, str):
                    try:
                        man = json.loads(man)
                    except ValueError:
                        man = None
                stage = (man or {}).get("stage") or str(e.get("handler_name"))
                seen[(scen, stage)] += 1
                ex = {"case": cid, "label": label, "stage": stage, "thought": e.get("thought_id"),
                      "retry": (man or {}).get("retry_index")}
                if not man:
                    flag("no_manifest", f"{stage} (handler {e.get('handler_name')})", ex)
                    continue
                try:
                    msgs = json.loads(e["prompt"]) if isinstance(e.get("prompt"), str) else (e.get("prompt") or [])
                except ValueError:
                    msgs = []
                if man.get("guard_violations"):
                    flag("guard_violation", f"{stage}: {[g.get('reason', g) if isinstance(g, dict) else g for g in man['guard_violations']][:3]}", ex)
                if man.get("sent_messages_match") is False:
                    flag("sent_differs_from_assembled", stage, ex)
                segs = man.get("segments") or []
                rows = stages.get(stage, {})
                if locale in SCRIPTS and msgs and stage not in POLYGLOT_STAGES:
                    lo, hi = SCRIPTS[locale]
                    n = 0
                    for s in segs:
                        if s["kind"] == "stage_instruction" and isinstance(s.get("message_index"), int) \
                                and s["message_index"] < len(msgs) and isinstance(msgs[s["message_index"]], dict):
                            t = str(msgs[s["message_index"]].get("content") or "")[s["span_start"]:s["span_end"]]
                            n += sum(1 for ch in t if lo <= ord(ch) <= hi)
                    if n < MIN_SCRIPT_CHARS:
                        flag("not_localized", f"{stage} stage_instruction has {('<' + str(MIN_SCRIPT_CHARS)) if n else 'no'} {locale} chars", ex)
                ids = {s["segment_id"]: s for s in segs}
                for s in segs:
                    k = key_of(s)
                    form = s.get("form", "full")
                    present[(scen, stage)][f"{k}[{form}]"] += 1
                    # differential: the recorded span hash against the bytes actually sent
                    mi, a, b = s.get("message_index"), s.get("span_start"), s.get("span_end")
                    if msgs and isinstance(mi, int) and mi < len(msgs):
                        content = msgs[mi].get("content") if isinstance(msgs[mi], dict) else None
                        if isinstance(content, str):
                            h = hashlib.sha256(content[a:b].encode("utf-8")).hexdigest()
                            if h != s.get("sha256"):
                                flag("span_hash_mismatch", f"{stage}.{k}", ex)
                    # policy: allowed here, in this form?
                    cell = rows.get(k)
                    holds = (cell is not None and cell[0] == "retry" and (man.get("retry_index") or 0) > 0) or \
                        (cell is not None and scen in PRED_SCEN.get(cell[0], set()))
                    if cell is None or cell[0] == "never" or not holds:
                        flag("not_allowed_by_policy", f"{stage}.{k}[{form}] in {scen}", ex)
                    elif cell[1] != form:
                        flag("wrong_form", f"{stage}.{k} is {form}, policy says {cell[1]} ({scen})", ex)
                    # firewall
                    if stage in ablated and s.get("trust") == "untrusted" and form != "stub" and s["kind"] not in RESIDUAL_KINDS:
                        flag("firewall", f"{stage}.{k} untrusted in full ({scen})", ex)
                    # laundering: trusted label over untrusted bytes
                    if s.get("trust") in ("system", "verified"):
                        bad = [d for d in s.get("derived_from") or [] if (ids.get(d) or {}).get("trust") == "untrusted"]
                        if bad:
                            flag("laundering", f"{stage}.{k} ({s.get('trust')}) derived from untrusted", ex)
                    if stage in ablated and s["kind"] == "display_name":
                        flag("identity_layout", f"{stage} receives display_name ({form})", ex)
                # required: policy cells whose predicate holds here but nothing of that key is present
                keys_here = {key_of(s) for s in segs}
                for k, (pred, form) in rows.items():
                    if pred != "never" and scen in PRED_SCEN.get(pred, set()) and k not in keys_here:
                        if k in ("tool_output", "schema_instruction", "conscience_feedback", "dma_output", "history",
                                 "action_record", "draft"):
                            continue  # present only when the thing exists (tool ran, follow-up, retry ...)
                        flag("required_missing", f"{stage}.{k} in {scen}", ex)
                # text differential at ablated stages: the request's text must not be in the prompt
                if stage in ablated and msgs:
                    # blank the draft spans (the accepted residual: the action judged may quote the request)
                    texts = [m.get("content") if isinstance(m, dict) and isinstance(m.get("content"), str) else ""
                             for m in msgs]
                    for s in segs:
                        if s["kind"] in RESIDUAL_KINDS and isinstance(s.get("message_index"), int) \
                                and s["message_index"] < len(texts):
                            t = texts[s["message_index"]]
                            texts[s["message_index"]] = t[:s["span_start"]] + " " * (s["span_end"] - s["span_start"]) \
                                + t[s["span_end"]:]
                    blob = "\n".join(texts)
                    for lab, txt in probe_texts:
                        frag = txt.split("$")[0].strip()[-60:]
                        if len(frag) >= 20 and frag in blob:
                            flag("request_text_at_ablated_stage", f"{stage} ({lab})", ex)

    report = {"calls": sum(seen.values()), "by_scenario_stage": {f"{a}/{b}": n for (a, b), n in sorted(seen.items())},
              "findings": {k: dict(v) for k, v in findings.items()},
              "examples": {f"{k}|{d}": v for (k, d), v in examples.items()},
              "present": {f"{a}/{b}": dict(c) for (a, b), c in present.items()}}
    if out:
        json.dump(report, open(out, "w"), indent=1)
    print(f"{report['calls']} calls checked")
    for k, v in report["findings"].items():
        print(f"\n[{k}] {sum(v.values())}")
        for d, n in sorted(v.items(), key=lambda x: -x[1])[:15]:
            print(f"  {n:5}  {d}")
    if not report["findings"]:
        print("no findings")


if __name__ == "__main__":
    out = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
    loc = sys.argv[sys.argv.index("--locale") + 1] if "--locale" in sys.argv else None
    main(sys.argv[1], sys.argv[2], out, loc)
