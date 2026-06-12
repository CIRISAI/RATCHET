"""Pass A v2: regex-based recovery for malformed prompt rows; correct fed join."""
import gzip, json, re, os
from collections import Counter, defaultdict

DUMP = "/home/emoore/0612_prod_traces"

# regexes for malformed-row recovery (LLM_CALL with double-escaped prompts)
RX_AID = re.compile(r'"agent_id_hash":"([a-f0-9]+)"')
RX_ETYPE = re.compile(r'"event_type":"([A-Z_]+)"')
RX_COST_USD = re.compile(r'"cost_usd":([0-9.e+-]+|null)')
RX_COST_TOKENS = re.compile(r'"cost_tokens":([0-9]+|null)')
RX_TS = re.compile(r'"ts":"([0-9T:.+-]+)"')
RX_MODEL = re.compile(r'"model":"([^"]+)"')
RX_CSTATE = re.compile(r'"cognitive_state":"([^"]+)"')
RX_SCHEMA = re.compile(r'"schema_version":"([^"]+)"')

def recover(line):
    """Extract core fields from a malformed row via regex."""
    aid = RX_AID.search(line)
    etype = RX_ETYPE.search(line)
    cost_usd = RX_COST_USD.search(line)
    cost_tokens = RX_COST_TOKENS.search(line)
    ts = RX_TS.search(line)
    model = RX_MODEL.search(line)
    cstate = RX_CSTATE.search(line)
    return {
        "agent_id_hash": aid.group(1) if aid else None,
        "event_type": etype.group(1) if etype else None,
        "cost_usd": (float(cost_usd.group(1)) if cost_usd and cost_usd.group(1) != "null" else None),
        "cost_tokens": (int(cost_tokens.group(1)) if cost_tokens and cost_tokens.group(1) != "null" else None),
        "ts": ts.group(1) if ts else None,
        "model": model.group(1) if model else None,
        "cognitive_state": cstate.group(1) if cstate else None,
        "_recovered": True,
    }

agents = defaultdict(lambda: {
    "n_events": 0, "n_dma": 0, "n_aspdma": 0, "n_idma": 0,
    "n_conscience": 0, "n_action": 0, "n_llm": 0, "n_thought_start": 0,
    "n_snapshot": 0, "n_recovered": 0,
    "cost_usd_sum": 0.0, "cost_tokens_sum": 0,
    "llm_models": Counter(),
    "dsdma_domains": Counter(),
    "dsdma_flags": Counter(),
    "pdma_conflict_count": 0, "pdma_total": 0,
    "conscience_pass_count": 0, "conscience_total": 0,
    "first_ts": None, "last_ts": None, "active_dates": set(),
    "cognitive_states": Counter(),
})

malformed_recovered = malformed_unrecoverable = parsed = 0
print("trace_events with regex recovery...")
with gzip.open(f"{DUMP}/trace_events.jsonl.gz", "rt") as f:
    for line in f:
        try:
            r = json.loads(line)
            parsed += 1
        except json.JSONDecodeError:
            r = recover(line)
            if r["agent_id_hash"] and r["event_type"]:
                malformed_recovered += 1
            else:
                malformed_unrecoverable += 1
                continue

        aid = r.get("agent_id_hash") or "MISSING"
        a = agents[aid]
        a["n_events"] += 1
        if r.get("_recovered"): a["n_recovered"] += 1
        et = r.get("event_type") or "UNKNOWN"
        if et == "DMA_RESULTS": a["n_dma"] += 1
        elif et == "ASPDMA_RESULT": a["n_aspdma"] += 1
        elif et == "IDMA_RESULT": a["n_idma"] += 1
        elif et == "CONSCIENCE_RESULT": a["n_conscience"] += 1
        elif et == "ACTION_RESULT": a["n_action"] += 1
        elif et == "LLM_CALL": a["n_llm"] += 1
        elif et == "THOUGHT_START": a["n_thought_start"] += 1
        elif et == "SNAPSHOT_AND_CONTEXT": a["n_snapshot"] += 1
        try:
            cu = r.get("cost_usd")
            if cu is not None: a["cost_usd_sum"] += float(cu)
            ct = r.get("cost_tokens")
            if ct is not None: a["cost_tokens_sum"] += int(ct)
        except Exception: pass
        ts = r.get("ts")
        if ts:
            a["active_dates"].add(str(ts)[:10])
            if a["first_ts"] is None or str(ts) < a["first_ts"]: a["first_ts"] = str(ts)
            if a["last_ts"] is None or str(ts) > a["last_ts"]: a["last_ts"] = str(ts)
        cs = r.get("cognitive_state")
        if cs: a["cognitive_states"][cs] += 1

        # payload-internal extraction only for cleanly-parsed rows
        if not r.get("_recovered"):
            p = r.get("payload") or {}
            if et == "DMA_RESULTS":
                pdma = p.get("pdma") or {}
                if "has_conflicts" in pdma:
                    a["pdma_total"] += 1
                    if pdma["has_conflicts"]: a["pdma_conflict_count"] += 1
                dsdma = p.get("dsdma") or {}
                if dsdma.get("domain"): a["dsdma_domains"][dsdma["domain"]] += 1
                for fl in (dsdma.get("flags") or []): a["dsdma_flags"][fl] += 1
            elif et == "CONSCIENCE_RESULT":
                a["conscience_total"] += 1
                if p.get("passed") is True or p.get("conscience_passed") is True:
                    a["conscience_pass_count"] += 1
                elif not any(k in p for k in ("vetoed", "failure", "rejected")) and p.get("passed") is not False:
                    a["conscience_pass_count"] += 1
            elif et == "LLM_CALL":
                m = p.get("model") or p.get("response_model")
                if m: a["llm_models"][m] += 1
        else:
            if r.get("model"): a["llm_models"][r["model"]] += 1

print(f"  parsed={parsed}  recovered={malformed_recovered}  unrecoverable={malformed_unrecoverable}")

# trace_llm_calls with recovery
print("trace_llm_calls with recovery...")
parsed = malformed_recovered = malformed_unrecoverable = 0
with gzip.open(f"{DUMP}/trace_llm_calls.jsonl.gz", "rt") as f:
    for line in f:
        try:
            r = json.loads(line); parsed += 1
        except json.JSONDecodeError:
            r = recover(line)
            if r["agent_id_hash"]: malformed_recovered += 1
            else: malformed_unrecoverable += 1; continue
        aid = r.get("agent_id_hash") or "MISSING"
        a = agents[aid]
        try:
            if r.get("cost_usd") is not None: a["cost_usd_sum"] += float(r["cost_usd"])
        except Exception: pass
        for k in ("prompt_tokens", "completion_tokens", "cost_tokens"):
            try:
                if r.get(k) is not None: a["cost_tokens_sum"] += int(r[k])
            except Exception: pass
        m = r.get("model") or r.get("response_model")
        if m: a["llm_models"][m] += 1
print(f"  parsed={parsed}  recovered={malformed_recovered}  unrecoverable={malformed_unrecoverable}")

# federation membership via accord_public_keys: key_id="agent-{12hex}" → agent_id_hash starts with same 12 hex
print("federation membership via accord_public_keys...")
key_prefixes = set()
with gzip.open(f"{DUMP}/accord_public_keys.jsonl.gz", "rt") as f:
    for line in f:
        try: r = json.loads(line)
        except: continue
        kid = r.get("key_id") or ""
        m = re.match(r"^agent-([a-f0-9]{12})$", kid)
        if m: key_prefixes.add(m.group(1))
print(f"  accord_public_key prefixes: {len(key_prefixes)}")
fed_members = set()
for aid in agents:
    if any(aid.startswith(p) for p in key_prefixes):
        fed_members.add(aid)
print(f"  matched federation members in trace dataset: {len(fed_members)}")

# emit
os.makedirs("/home/emoore/RATCHET/release/calibration/crc-v2/data", exist_ok=True)
out = []
for aid, a in agents.items():
    out.append({
        "agent_id_hash": aid,
        "n_events": a["n_events"],
        "n_recovered": a["n_recovered"],
        "n_dma": a["n_dma"], "n_aspdma": a["n_aspdma"], "n_idma": a["n_idma"],
        "n_conscience": a["n_conscience"], "n_action": a["n_action"], "n_llm": a["n_llm"],
        "cost_usd_sum": round(a["cost_usd_sum"], 6),
        "cost_tokens_sum": a["cost_tokens_sum"],
        "n_distinct_models": len(a["llm_models"]),
        "llm_model_counts": dict(a["llm_models"]),
        "n_distinct_domains": len(a["dsdma_domains"]),
        "dsdma_flags_count": dict(a["dsdma_flags"]),
        "pdma_total": a["pdma_total"], "pdma_conflict_count": a["pdma_conflict_count"],
        "pdma_conflict_rate": (a["pdma_conflict_count"]/a["pdma_total"]) if a["pdma_total"] else None,
        "conscience_total": a["conscience_total"],
        "conscience_pass_rate": (a["conscience_pass_count"]/a["conscience_total"]) if a["conscience_total"] else None,
        "federation_member": aid in fed_members,
        "first_ts": a["first_ts"], "last_ts": a["last_ts"],
        "days_active": len(a["active_dates"]),
        "cognitive_state_dominant": (a["cognitive_states"].most_common(1)[0][0] if a["cognitive_states"] else None),
    })

with open("/home/emoore/RATCHET/release/calibration/crc-v2/data/agents.jsonl", "w") as f:
    for row in out: f.write(json.dumps(row) + "\n")

# summary
n = len(out)
print(f"\nemitted {n} agent aggregates")
print(f"  with PDMA data: {sum(1 for r in out if r['pdma_total']>0)}")
print(f"  with LLM cost data: {sum(1 for r in out if r['cost_usd_sum']>0)}")
print(f"  with ≥100 events: {sum(1 for r in out if r['n_events']>=100)}")
print(f"  with ≥1000 events: {sum(1 for r in out if r['n_events']>=1000)}")
print(f"  federation members: {sum(1 for r in out if r['federation_member'])}")
print(f"  uncategorizable (MISSING/unknown/IDENTIFIER): {sum(1 for r in out if r['agent_id_hash'] in ('MISSING','unknown','[IDENTIFIER]'))}")
