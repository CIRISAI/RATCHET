"""Harvest per-thought verb records from live Safety Battery agent-logs artifacts (read-only download).
For each trace: run id, locale, thought id/type/depth, ASPDMA selected action, conscience final action and override,
executed action, DMA result fields as present, and every LLM_CALL's handler and prompt_hash. No prompt text exists at
this trace level (full_traces is refused without a substrate scrubber, CIRISServer#418)."""
import json, os, subprocess, sys, tempfile, shutil
out_path, listing = sys.argv[1], sys.argv[2]
done = set()
if os.path.exists(out_path):
    done = {json.loads(l)["artifact"] for l in open(out_path)}
arts = [l.split("\t")[0] for l in open(listing) if l.strip()]
with open(out_path, "a") as out:
    for name in arts:
        if name in done:
            continue
        run_id, loc = name.split("-")[-2], name.split("-")[-1]
        d = tempfile.mkdtemp(dir=os.path.dirname(out_path))
        r = subprocess.run(["gh", "run", "download", run_id, "-R", "CIRISAI/CIRISAgent", "-n", name, "-D", d],
                           capture_output=True, text=True)
        n = 0
        tdir = os.path.join(d, "qa_reports", "capture_traces", "accord_detailed")
        if r.returncode == 0 and os.path.isdir(tdir):
            for f in sorted(os.listdir(tdir)):
                try:
                    doc = json.load(open(os.path.join(tdir, f)))
                except Exception:
                    continue
                for ev in doc.get("events", []):
                    t = ev.get("trace", {})
                    rec = {"artifact": name, "run_id": run_id, "locale": loc, "trace_id": t.get("trace_id"),
                           "task_id": t.get("task_id"), "trace_level": t.get("trace_level"), "llm_calls": []}
                    for c in t.get("components", []):
                        data = c.get("data") or {k: v for k, v in c.items() if k not in ("agent_id_hash", "timestamp")}
                        et = c.get("event_type")
                        if et == "THOUGHT_START":
                            rec.update({k: data.get(k) for k in ("thought_type", "thought_depth", "round_number")})
                        elif et == "ASPDMA_RESULT":
                            rec.setdefault("aspdma", []).append({"selected_action": data.get("selected_action"),
                                                                  "is_recursive": data.get("is_recursive")})
                        elif et == "CONSCIENCE_RESULT":
                            rec["conscience"] = {k: data.get(k) for k in ("final_action", "action_was_overridden")}
                        elif et == "ACTION_RESULT":
                            rec["action_executed"] = data.get("action_executed")
                        elif et == "DMA_RESULTS":
                            rec["dma"] = data
                        elif et == "IDMA_RESULT":
                            rec["idma"] = data
                        elif et == "LLM_CALL":
                            rec["llm_calls"].append({k: data.get(k) for k in ("handler_name", "prompt_hash", "prompt_bytes",
                                                                               "completion_tokens", "model")})
                    out.write(json.dumps(rec) + "\n"); n += 1
        out.write(json.dumps({"artifact": name, "harvested_records": n, "download_rc": r.returncode, "marker": True}) + "\n")
        out.flush()
        shutil.rmtree(d, ignore_errors=True)
        print(name, n, flush=True)
