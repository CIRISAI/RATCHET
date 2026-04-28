#!/usr/bin/env bash
# Export the full CIRISLens trace corpus as JSONL for offline analysis.
#
# Usage:
#   scripts/export_corpus.sh                           # default target: ~/RATCHET/corpus
#   scripts/export_corpus.sh /path/to/output/dir       # custom target
#   CIRISLENS_PSQL='...' scripts/export_corpus.sh      # custom psql command
#
# JSONL (one JSON object per line) preserves the JSONB fields that CSV
# would flatten away. Each dump file is directly loadable via pandas
# `pd.read_json(..., lines=True)` or `jq -c`.

set -euo pipefail

TARGET="${1:-$HOME/RATCHET/corpus}"

: "${CIRISLENS_PSQL:=ssh -i ~/Desktop/ciris_transfer/.ciris_bridge_keys/cirisbridge_ed25519 root@108.61.242.236 \"docker exec -i cirislens-db psql -U cirislens -d cirislens -t -A -q\"}"

mkdir -p "$TARGET"
cd "$TARGET"

# Helper: dump a query as JSONL.
# Uses SELECT (not COPY) because COPY's text-format escape layer adds a
# second level of backslash-escaping on top of row_to_json's own JSON escapes,
# which breaks parsing when JSONB fields contain stringified-JSON values.
# SELECT with -t -A emits one row per line, JSON-escape-only.
dump_jsonl() {
    local out="$1"
    local select_expr="$2"
    local tmp="${out}.tmp"
    # shellcheck disable=SC2086
    eval "$CIRISLENS_PSQL" <<SQL > "$tmp"
\set ON_ERROR_STOP on
\pset footer off
SELECT row_to_json(t) FROM ($select_expr) t;
SQL
    # Drop any trailing blank line psql may add
    sed -i '/^$/d' "$tmp"
    mv "$tmp" "$out"
    local n
    n=$(wc -l < "$out")
    local size
    size=$(du -h "$out" | awk '{print $1}')
    printf "  %-32s %8d rows  %s\n" "$out" "$n" "$size"
}

echo "Exporting corpus to $TARGET ..."

# 1) Raw accord_traces (primary corpus) — includes all JSONB blobs
dump_jsonl "accord_traces.jsonl" \
    "SELECT * FROM cirislens.accord_traces ORDER BY id"

# 2) Batch context (correlation_metadata, trace_level, timestamps)
dump_jsonl "accord_trace_batches.jsonl" \
    "SELECT * FROM cirislens.accord_trace_batches ORDER BY batch_timestamp"

# 3) Public keys (for signature verification)
dump_jsonl "accord_public_keys.jsonl" \
    "SELECT * FROM cirislens.accord_public_keys ORDER BY created_at"

# 4) Connectivity events (agent startup/shutdown)
dump_jsonl "connectivity_events.jsonl" \
    "SELECT * FROM cirislens.connectivity_events ORDER BY timestamp"

# 5) Analysis-ready flat view with derived columns (task_class, qa_language, etc.)
#    Prefer this for numerical analysis; use accord_traces.jsonl for raw JSONB.
dump_jsonl "trace_context.jsonl" \
    "SELECT * FROM cirislens.trace_context ORDER BY id"

# 6) Schema DDL for reference
echo "Dumping schema DDL..."
eval "$CIRISLENS_PSQL" <<'SQL' > "schema.sql.tmp"
\set ON_ERROR_STOP on
\pset footer off
\d cirislens.accord_traces
\echo ----
\d cirislens.accord_trace_batches
\echo ----
\d cirislens.accord_public_keys
\echo ----
\d cirislens.connectivity_events
\echo ----
\d cirislens.trace_context
SQL
mv schema.sql.tmp schema.sql
echo "  schema.sql                       ($(wc -l < schema.sql) lines)"

# 7) Export metadata
cat > metadata.json <<META
{
  "exported_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "exported_by": "$USER@$(hostname)",
  "source": "cirislens.* tables on production",
  "files": {
    "accord_traces.jsonl":        "$(wc -l < accord_traces.jsonl) rows",
    "accord_trace_batches.jsonl": "$(wc -l < accord_trace_batches.jsonl) rows",
    "accord_public_keys.jsonl":   "$(wc -l < accord_public_keys.jsonl) rows",
    "connectivity_events.jsonl":  "$(wc -l < connectivity_events.jsonl) rows",
    "trace_context.jsonl":        "$(wc -l < trace_context.jsonl) rows (flat analysis view)"
  }
}
META
echo "  metadata.json"

echo
echo "Done. Total size: $(du -sh "$TARGET" | awk '{print $1}')"
