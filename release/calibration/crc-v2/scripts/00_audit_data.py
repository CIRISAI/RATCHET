"""Audit parse failures + federation_keys join."""
import gzip, json, re
DUMP = "/home/emoore/0612_prod_traces"

# 1. What does a malformed line look like?
print("=== first 3 malformed trace_events ===")
errs = 0
with gzip.open(f"{DUMP}/trace_events.jsonl.gz", "rt") as f:
    for i, line in enumerate(f):
        try: json.loads(line)
        except json.JSONDecodeError as e:
            errs += 1
            if errs <= 3:
                # extract context
                col = e.colno
                print(f"err {errs}: line {i}, col {col}: {e.msg}")
                print(f"  context: ...{line[max(0,col-30):col+30]}...")
            if errs > 5: break

# 2. federation_keys schema
print("\n=== federation_keys sample row ===")
with gzip.open(f"{DUMP}/federation_keys.jsonl.gz", "rt") as f:
    for line in f:
        try: r = json.loads(line)
        except: continue
        print(json.dumps(r, indent=2)[:600])
        break

# 3. accord_public_keys schema
print("\n=== accord_public_keys sample row ===")
with gzip.open(f"{DUMP}/accord_public_keys.jsonl.gz", "rt") as f:
    for line in f:
        try: r = json.loads(line)
        except: continue
        print(json.dumps(r, indent=2)[:600])
        break

# 4. how many trace_events agents have at least one event in the federation_keys.agent_name or accord_public_keys field?
print("\n=== checking agent_id vs agent_id_hash linkage ===")
fed_agent_ids = set()
fed_agent_names = set()
fed_pubkeys = set()
with gzip.open(f"{DUMP}/federation_keys.jsonl.gz", "rt") as f:
    for line in f:
        try: r = json.loads(line)
        except: continue
        if r.get("agent_id"): fed_agent_ids.add(r["agent_id"])
        if r.get("agent_name"): fed_agent_names.add(r["agent_name"])
        for k in ("pubkey_b64", "public_key", "pubkey"):
            if r.get(k): fed_pubkeys.add(r[k])
print(f"fed agent_ids: {len(fed_agent_ids)}, agent_names: {len(fed_agent_names)}, pubkeys: {len(fed_pubkeys)}")

# Sample trace_events to see what links to federation
te_agent_names = set()
te_agent_ids_hash = set()
with gzip.open(f"{DUMP}/trace_events.jsonl.gz", "rt") as f:
    for i, line in enumerate(f):
        try: r = json.loads(line)
        except: continue
        if r.get("agent_name"): te_agent_names.add(r["agent_name"])
        if r.get("agent_id_hash"): te_agent_ids_hash.add(r["agent_id_hash"])
        if i > 50000: break
print(f"trace_events agent_names (sample): {len(te_agent_names)} distinct")
print(f"  overlap with federation_keys.agent_name: {len(te_agent_names & fed_agent_names)}")
print(f"trace_events agent_id_hash (sample): {len(te_agent_ids_hash)}")
