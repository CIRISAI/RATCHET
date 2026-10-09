"""Publication scrub of the canonical trace export, RATCHET-owned, verifier-gated.

Order of operations is the point:
  1. JOIN provenance first — provenance_split + identity_class are attached from the
     unscrubbed structural columns (trace_id, task_id, agent_id_hash, channel), which
     never enter the scrubber.
  2. SCRUB every event's `payload` with the post-fold wheel (`ciris_server.scrub_traces_batch`)
     at level `full_traces` — our PUBLICATION policy, regardless of the agent's recording
     level. (`generic` rows carry `defer_reason` text too, and the wheel's own `generic`
     path is a bypass.)
  3. RATCHET post-pass, each counted:
       - restore STRUCTURAL_IDENTIFIER_KEYS values the wheel altered inside scrub subtrees
         (CIRISServer#754)
       - extend placeholders glued to word fragments to the whole word
         (`[ORG_1]IRIS` → `[ORG_1]`, CIRISServer#755) — a superset, never a leak
       - `@handle` → `[HANDLE_n]` (the dominant real-PII shape here; no backbone catches all)
  4. VERIFY (verify.py): structural keys, year residue, placeholder grammar, mid-token.
     A row failing year-residue or structural checks after the post-pass is EXCLUDED and
     counted; nothing is silently kept.

Outputs (0600, beside the export): scrubbed/trace_events.scrubbed.jsonl, passthrough copies of
llm_calls (prompt/response are NULL) and thought_signatures, MANIFEST.json with sha256s,
counts, every stat, and the exact scrub configuration (wheel version, backbone, model digests).
Run with python3 -I.
"""
import json, os, re, sys, time, hashlib, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ciris_server as cs
from reference import (RESTORE_KEYS, PLACEHOLDER, walk_string_leaves,
                       count_year_residue, scrub_string)
from verify import verify, summarize

EXPORT = os.environ.get("TRACE_EXPORT_DIR", "/home/emoore/ratchet-trace-export").rstrip("/") + "/"
OUT = EXPORT + "scrubbed/"
LIMIT = int(os.environ.get("SCRUB_LIMIT", "0"))      # 0 = all
BATCH = int(os.environ.get("SCRUB_BATCH", "64"))
LEVEL = "full_traces"
os.makedirs(OUT, exist_ok=True); os.chmod(OUT, 0o700)

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

# ── 1. provenance, joined from structural columns before anything is scrubbed ──
prov = {}
for l in open(EXPORT + "provenance_split.jsonl"):
    d = json.loads(l); prov[d["trace_id"]] = (d["split"], d.get("identity_class"))

# ── 3a/3b/3c helpers ──
HANDLE = re.compile(r"(?<![\w@])@([A-Za-z][\w.]{1,30})")
GLUE_R = re.compile(r"(\[(?:[A-Z_]+?)_\d+\])(\w+)")      # [ORG_1]IRIS
GLUE_L = re.compile(r"(\w+)(\[(?:[A-Z_]+?)_\d+\])")      # D[ORG_2]
ABUT   = re.compile(r"(\[([A-Z_]+?)_\d+\])(?:\[\2_\d+\])+")    # [PER_1][PER_2] -> [PER_1]
PROVIDERS = (("deepinfra.com", "deepinfra"), ("openrouter.ai", "openrouter"), ("groq.com", "groq"),
             ("api.openai.com", "openai"), ("ciris-services", "ciris-services"), ("anthropic.com", "anthropic"))
ENDPOINT_KEYS = {"base_url", "api_bases_used"}
def provider(url):
    """Owner's policy (2026-10-09): endpoints publish as provider NAMES, never hostnames."""
    if not isinstance(url, str): return url
    for needle, name in PROVIDERS:
        if needle in url: return name
    return "other" if url.startswith(("http://", "https://")) or url == "[URL]" else url
def post_pass(before, after, stats):
    """Restore structural keys, extend glued placeholders, redact @handles. Returns new `after`."""
    bmap = dict(walk_string_leaves(before))
    hcount = [0]
    def fix(o, path, key):
        if isinstance(o, dict):
            return {k: fix(v, f"{path}.{k}" if path else k, k) for k, v in o.items()}
        if isinstance(o, list):
            return [fix(v, path + "[]", key) for v in o]
        if isinstance(o, str):
            if key in RESTORE_KEYS:
                b = bmap.get(path)
                if b is not None and b != o:
                    stats["structural_restored"] += 1; return b
                return o
            if key in ENDPOINT_KEYS:
                b = bmap.get(path)
                stats["endpoints_coarsened"] += 1
                return provider(b if b is not None else o)
            s = o
            n1 = len(GLUE_R.findall(s)); n2 = len(GLUE_L.findall(s))
            if n1 or n2:
                s = GLUE_R.sub(r"\1", s); s = GLUE_L.sub(r"\2", s)
                stats["glued_placeholders_extended"] += n1 + n2
            n3 = len(ABUT.findall(s))
            if n3:
                s = ABUT.sub(r"\1", s); stats["abutting_placeholders_merged"] += n3
            def h(m):
                hcount[0] += 1; stats["handles_redacted"] += 1
                return f"@[HANDLE_{hcount[0]}]"
            s = HANDLE.sub(h, s)
            return s
        return o
    return fix(after, "", None)

# ── 2. stream, batch, scrub ──
src = EXPORT + "trace_events.jsonl"
dst_tmp = OUT + "trace_events.scrubbed.jsonl.tmp"
log = open(OUT + "run.log", "a")
stats = collections.Counter(); reps = []; excluded = []
t0 = time.time(); n_in = n_out = 0
pending = []  # (row, before_payload_obj, wrapped_json)

def flush(out):
    global n_out
    if not pending: return
    res = cs.scrub_traces_batch([w for _, _, w in pending], LEVEL)
    for (row, before, _), r in zip(pending, res):
        after = json.loads(r["trace"])["events"][0]["data"]
        for k, v in (r.get("stats") or {}).items():
            if isinstance(v, (int, float)) and not isinstance(v, bool): stats["wheel_" + k] += v
        after = post_pass(before, after, stats)
        rep = verify(before, after); reps.append({k: rep[k] for k in ("structural_violations", "year_residue",
                     "year_residue_in_identifiers", "unknown_placeholders", "mid_token", "tags", "fields_changed",
                     "strings", "strings_changed", "ok")})
        if rep["structural_violations"] or rep["year_residue"]:
            excluded.append((row["trace_id"], row["event_id"], len(rep["structural_violations"]), rep["year_residue"]))
            stats["rows_excluded"] += 1
            continue
        row["payload"] = json.dumps(after, ensure_ascii=False)
        # other text-bearing columns: regex-only reference pass (no NER), structural keys untouched
        for col in ("extracted_features", "classifications", "pipeline_metadata"):
            v = row.get(col)
            if isinstance(v, str) and v: row[col] = scrub_string(v, stats)
        row["provenance_split"], row["identity_class"] = prov.get(row["trace_id"], ("unknown", None))
        row["scrub_policy"] = "ratchet/full_traces+postpass/v3"
        out.write(json.dumps(row, ensure_ascii=False) + "\n"); n_out += 1
    pending.clear()

with open(dst_tmp, "w") as out:
    os.chmod(dst_tmp, 0o600)
    for l in open(src):
        n_in += 1
        if LIMIT and n_in > LIMIT: break
        row = json.loads(l)
        try:
            before = json.loads(row["payload"])
        except Exception:
            stats["payload_unparseable"] += 1
            row["payload"] = scrub_string(row["payload"] or "", stats)   # regex-only on the raw string
            row["provenance_split"], row["identity_class"] = prov.get(row["trace_id"], ("unknown", None))
            row["scrub_policy"] = "ratchet/regex_only_unparseable/v1"
            out.write(json.dumps(row, ensure_ascii=False) + "\n"); n_out += 1
            continue
        pending.append((row, before, json.dumps({"trace_id": "t", "events": [{"event_type": row["event_type"], "data": before}]}, ensure_ascii=False)))
        if len(pending) >= BATCH:
            flush(out)
            if (n_in // BATCH) % 50 == 0:
                el = time.time() - t0
                log.write(f"{time.strftime('%H:%M:%S')} in={n_in} out={n_out} {el:.0f}s {1000*el/max(n_in,1):.1f}ms/row excl={stats['rows_excluded']}\n"); log.flush()
    flush(out)
os.replace(dst_tmp, OUT + "trace_events.scrubbed.jsonl"); os.chmod(OUT + "trace_events.scrubbed.jsonl", 0o600)

# passthrough files (no reasoning text): copy + hash
import shutil
shutil.copyfile(EXPORT + "trace_thought_signatures.jsonl", OUT + "trace_thought_signatures.jsonl")
with open(OUT + "trace_llm_calls.jsonl", "w") as lo:
    for l in open(EXPORT + "trace_llm_calls.jsonl"):
        d = json.loads(l)
        if d.get("base_url"): d["base_url"] = provider(d["base_url"]); stats["llm_calls_base_url_coarsened"] += 1
        assert not d.get("prompt") and not d.get("response_text"), "llm_calls carries text; passthrough is not safe"
        lo.write(json.dumps(d, ensure_ascii=False) + "\n")
for f in ("trace_llm_calls.jsonl", "trace_thought_signatures.jsonl"): os.chmod(OUT + f, 0o600)

# ── 4. manifest ──
M = os.environ.get("CIRISLENS_NER_MODEL_DIR", "")
model_digest = {f: sha256(os.path.join(M, f)) for f in ("config.json", "tokenizer.json", "model.safetensors")} if M and os.path.isdir(M) else None
s = summarize(reps)
manifest = {
    "built_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "source": {"trace_events": sha256(src), "rows_in": n_in - (1 if LIMIT and n_in > LIMIT else 0)},
    "scrub": {"wheel": "ciris-server " + getattr(cs, "__version__", "?"), "level": LEVEL,
              "backbone": os.environ.get("CIRISLENS_NER_BACKBONE", "<default>"), "model_digests": model_digest,
              "post_pass": ["structural_restore(#754, RESTORE_KEYS)", "glued_placeholder_extend(#755)", "abutting_placeholder_merge(#755)", "handle_redact", "endpoint_provider_names"],
              "verifier": "experiments/scrub_verify/verify.py", "batch": BATCH},
    "counts": {"rows_out": n_out, "rows_excluded": stats["rows_excluded"], "payload_unparseable": stats["payload_unparseable"]},
    "stats": dict(stats),
    "verify_summary": s,
    "excluded_rows": excluded[:200],
    "files": {f: {"sha256": sha256(OUT + f), "bytes": os.path.getsize(OUT + f)}
              for f in ("trace_events.scrubbed.jsonl", "trace_llm_calls.jsonl", "trace_thought_signatures.jsonl")},
    "elapsed_s": round(time.time() - t0, 1),
}
json.dump(manifest, open(OUT + "MANIFEST.json", "w"), indent=1); os.chmod(OUT + "MANIFEST.json", 0o600)
print(json.dumps({k: manifest[k] for k in ("counts", "verify_summary", "elapsed_s")}, indent=1))
print("post-pass:", {k: v for k, v in stats.items() if not k.startswith("wheel_")})
