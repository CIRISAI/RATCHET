"""Pass A: aggregate per-agent metrics from the dump.

Output: /home/emoore/RATCHET/release/calibration/crc-v2/data/agents.parquet

Columns per agent_id_hash:
  - n_events, n_dma, n_aspdma, n_idma, n_conscience, n_action, n_llm
  - cost_usd_sum, cost_tokens_sum, llm_models (set), llm_models_top (str)
  - dsdma_domains (set, top), dsdma_flags (counter)
  - pdma_conflict_count, pdma_total
  - conscience_pass_rate (if extractable)
  - federation_member (bool, from accord_public_keys join)
  - first_ts, last_ts, days_active
  - cognitive_state_dominant
  - agent_role, agent_template (if present)
"""
import gzip, json, sys, os
from collections import Counter, defaultdict
from datetime import datetime

DUMP = "/home/emoore/0612_prod_traces"

agents = defaultdict(lambda: {
    "n_events": 0, "n_dma": 0, "n_aspdma": 0, "n_idma": 0,
    "n_conscience": 0, "n_action": 0, "n_llm": 0, "n_thought_start": 0,
    "n_snapshot": 0, "n_verb_second": 0,
    "cost_usd_sum": 0.0, "cost_tokens_sum": 0,
    "llm_models": Counter(),
    "dsdma_domains": Counter(),
    "dsdma_flags": Counter(),
    "pdma_conflict_count": 0, "pdma_total": 0,
    "conscience_pass_count": 0, "conscience_total": 0,
    "first_ts": None, "last_ts": None,
    "active_dates": set(),
    "cognitive_states": Counter(),
    "agent_role": None, "agent_template": None,
    "schema_versions": Counter(),
})

malformed = 0
processed = 0
print("reading trace_events.jsonl.gz...", file=sys.stderr)
with gzip.open(f"{DUMP}/trace_events.jsonl.gz", "rt") as f:
    for line in f:
        try:
            r = json.loads(line)
        except Exception:
            malformed += 1
            continue
        processed += 1
        aid = r.get("agent_id_hash") or "MISSING"
        a = agents[aid]
        a["n_events"] += 1
        et = r.get("event_type") or "UNKNOWN"
        if et == "DMA_RESULTS": a["n_dma"] += 1
        elif et == "ASPDMA_RESULT": a["n_aspdma"] += 1
        elif et == "IDMA_RESULT": a["n_idma"] += 1
        elif et == "CONSCIENCE_RESULT": a["n_conscience"] += 1
        elif et == "ACTION_RESULT": a["n_action"] += 1
        elif et == "LLM_CALL": a["n_llm"] += 1
        elif et == "THOUGHT_START": a["n_thought_start"] += 1
        elif et == "SNAPSHOT_AND_CONTEXT": a["n_snapshot"] += 1
        elif et == "VERB_SECOND_PASS_RESULT": a["n_verb_second"] += 1
        try:
            cu = r.get("cost_usd")
            if cu is not None: a["cost_usd_sum"] += float(cu)
            ct = r.get("cost_tokens")
            if ct is not None: a["cost_tokens_sum"] += int(ct)
        except Exception: pass
        ts = r.get("ts")
        if ts:
            a["active_dates"].add(ts[:10])
            if a["first_ts"] is None or ts < a["first_ts"]: a["first_ts"] = ts
            if a["last_ts"] is None or ts > a["last_ts"]: a["last_ts"] = ts
        cs = r.get("cognitive_state")
        if cs: a["cognitive_states"][cs] += 1
        if r.get("agent_role"): a["agent_role"] = r["agent_role"]
        if r.get("agent_template"): a["agent_template"] = r["agent_template"]
        if r.get("schema_version"): a["schema_versions"][r["schema_version"]] += 1
        # payload dives
        p = r.get("payload") or {}
        if et == "DMA_RESULTS":
            pdma = p.get("pdma") or {}
            if "has_conflicts" in pdma:
                a["pdma_total"] += 1
                if pdma["has_conflicts"]: a["pdma_conflict_count"] += 1
            dsdma = p.get("dsdma") or {}
            dom = dsdma.get("domain")
            if dom: a["dsdma_domains"][dom] += 1
            flags = dsdma.get("flags") or []
            for fl in flags: a["dsdma_flags"][fl] += 1
        elif et == "CONSCIENCE_RESULT":
            a["conscience_total"] += 1
            # heuristic: passed iff payload has no veto/failure flag
            if p.get("passed") is True or p.get("conscience_passed") is True:
                a["conscience_pass_count"] += 1
            elif p.get("passed") is False or p.get("conscience_passed") is False:
                pass
            elif not any(k in p for k in ("vetoed", "failure", "rejected")):
                a["conscience_pass_count"] += 1  # best-effort default-pass
        elif et == "LLM_CALL":
            m = p.get("model") or p.get("response_model")
            if m: a["llm_models"][m] += 1
print(f"trace_events: {processed} processed, {malformed} malformed", file=sys.stderr)

# accord_traces (legacy, pre-cutover)
print("reading accord_traces.jsonl.gz (pre-cutover, schema 2.7.x)...", file=sys.stderr)
processed = malformed = 0
with gzip.open(f"{DUMP}/accord_traces.jsonl.gz", "rt") as f:
    for line in f:
        try:
            r = json.loads(line)
        except Exception:
            malformed += 1; continue
        processed += 1
        # accord schema is different; identify agent by trace metadata
        aid = r.get("agent_id_hash") or r.get("agent_id") or "LEGACY_UNKNOWN"
        a = agents[aid]
        a["n_events"] += 1
        a["n_dma"] += 1  # treat legacy traces as DMA equivalent
        ts = r.get("created_at") or r.get("ts")
        if ts:
            a["active_dates"].add(str(ts)[:10])
            if a["first_ts"] is None or str(ts) < a["first_ts"]: a["first_ts"] = str(ts)
            if a["last_ts"] is None or str(ts) > a["last_ts"]: a["last_ts"] = str(ts)
print(f"accord_traces: {processed} processed, {malformed} malformed", file=sys.stderr)

# federation_keys + accord_public_keys for membership flag
fed_members = set()
print("reading federation_keys.jsonl.gz...", file=sys.stderr)
with gzip.open(f"{DUMP}/federation_keys.jsonl.gz", "rt") as f:
    for line in f:
        try: r = json.loads(line)
        except Exception: continue
        aid = r.get("agent_id_hash")
        if aid: fed_members.add(aid)
print("reading accord_public_keys.jsonl.gz...", file=sys.stderr)
with gzip.open(f"{DUMP}/accord_public_keys.jsonl.gz", "rt") as f:
    for line in f:
        try: r = json.loads(line)
        except Exception: continue
        aid = r.get("agent_id_hash")
        if aid: fed_members.add(aid)
print(f"federation members: {len(fed_members)}", file=sys.stderr)

# trace_llm_calls richer than embedded LLM_CALL events
print("reading trace_llm_calls.jsonl.gz...", file=sys.stderr)
processed = malformed = 0
with gzip.open(f"{DUMP}/trace_llm_calls.jsonl.gz", "rt") as f:
    for line in f:
        try: r = json.loads(line)
        except Exception:
            malformed += 1; continue
        processed += 1
        aid = r.get("agent_id_hash") or "MISSING"
        a = agents[aid]
        try:
            cu = r.get("cost_usd")
            if cu is not None: a["cost_usd_sum"] += float(cu)
        except Exception: pass
        try:
            for k in ("prompt_tokens", "completion_tokens"):
                v = r.get(k)
                if v is not None: a["cost_tokens_sum"] += int(v)
        except Exception: pass
        m = r.get("model") or r.get("response_model")
        if m: a["llm_models"][m] += 1
print(f"trace_llm_calls: {processed} processed, {malformed} malformed", file=sys.stderr)

# emit
os.makedirs("/home/emoore/RATCHET/release/calibration/crc-v2/data", exist_ok=True)
out = []
for aid, a in agents.items():
    pdma_conflict_rate = (a["pdma_conflict_count"] / a["pdma_total"]) if a["pdma_total"] else None
    conscience_pass_rate = (a["conscience_pass_count"] / a["conscience_total"]) if a["conscience_total"] else None
    out.append({
        "agent_id_hash": aid,
        "n_events": a["n_events"],
        "n_dma": a["n_dma"], "n_aspdma": a["n_aspdma"], "n_idma": a["n_idma"],
        "n_conscience": a["n_conscience"], "n_action": a["n_action"], "n_llm": a["n_llm"],
        "n_thought_start": a["n_thought_start"], "n_snapshot": a["n_snapshot"],
        "cost_usd_sum": round(a["cost_usd_sum"], 6),
        "cost_tokens_sum": a["cost_tokens_sum"],
        "llm_models_top": (a["llm_models"].most_common(1)[0][0] if a["llm_models"] else None),
        "n_distinct_models": len(a["llm_models"]),
        "llm_model_concentration": (max(a["llm_models"].values()) / sum(a["llm_models"].values())) if a["llm_models"] else None,
        "dsdma_domain_top": (a["dsdma_domains"].most_common(1)[0][0] if a["dsdma_domains"] else None),
        "n_distinct_domains": len(a["dsdma_domains"]),
        "dsdma_flags_top": (a["dsdma_flags"].most_common(1)[0][0] if a["dsdma_flags"] else None),
        "n_distinct_flags": len(a["dsdma_flags"]),
        "pdma_total": a["pdma_total"], "pdma_conflict_count": a["pdma_conflict_count"],
        "pdma_conflict_rate": pdma_conflict_rate,
        "conscience_total": a["conscience_total"], "conscience_pass_rate": conscience_pass_rate,
        "federation_member": aid in fed_members,
        "first_ts": a["first_ts"], "last_ts": a["last_ts"],
        "days_active": len(a["active_dates"]),
        "cognitive_state_dominant": (a["cognitive_states"].most_common(1)[0][0] if a["cognitive_states"] else None),
        "agent_role": a["agent_role"], "agent_template": a["agent_template"],
        "schema_version_top": (a["schema_versions"].most_common(1)[0][0] if a["schema_versions"] else None),
    })

with open("/home/emoore/RATCHET/release/calibration/crc-v2/data/agents.jsonl", "w") as f:
    for row in out:
        f.write(json.dumps(row) + "\n")
print(f"wrote {len(out)} agent aggregates", file=sys.stderr)
print(f"federation members in dataset: {sum(1 for r in out if r['federation_member'])}", file=sys.stderr)
print(f"agents with PDMA data: {sum(1 for r in out if r['pdma_total'] > 0)}", file=sys.stderr)
print(f"agents with LLM cost data: {sum(1 for r in out if r['cost_usd_sum'] > 0)}", file=sys.stderr)
