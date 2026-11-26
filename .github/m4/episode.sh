#!/usr/bin/env bash
# One Market 4 episode, inside its own container (see Dockerfile). Usage: episode.sh <case_id> <tag>
#
# This is the per-case step the workflow used to run as one matrix job per case, moved verbatim into a script so a
# lane job can run several cases at once, each in its own container. Mounted: /tmp/m4-cases (frozen cases, ro),
# /out (shared lane output; this episode writes only m4-raw-<tag>/, m4-agent-logs-<tag>/ and _lane/<tag>.*), and for
# live runs /root/.openrouter_key (ro).
#
# Credit pause. If the provider refuses for credit (HTTP 402, "run out of credit", insufficient_quota) at any point
# in an episode, the episode is contaminated: it is stopped, its files are kept under qa_reports/credit_paused/
# attempt-N/ (uploaded with the agent logs), the episode waits until a 1-token request to the same model is accepted,
# and is rerun from a wiped start. Every pause is logged to qa_reports/credit_pauses.jsonl. It never waits out the
# key's weekly limit: if the key reports less than M4_MIN_KEY_REMAINING dollars left it stops red. Bounds:
# M4_MAX_ATTEMPTS episodes and M4_PAUSE_BUDGET_S seconds of waiting per episode.
set -u -o pipefail
CASE="$1"; TAG="$2"
cd /src
mkdir -p /out/_lane qa_reports qa_reports/credit_paused "qa_reports/correction_episodes/${CASE}_env_logs"
SUMMARY="/out/_lane/$TAG.summary.md"; : > "$SUMMARY"

collect() {
  local rc=$?
  local R="/out/m4-raw-$TAG" A="/out/m4-agent-logs-$TAG"
  mkdir -p "$R" "$A/qa_reports/correction_episodes"
  cp "qa_reports/correction_episodes/$CASE.jsonl" "$R/" 2>/dev/null || true
  cp "qa_reports/episode_$CASE.log" qa_reports/credit_pauses.jsonl "$A/qa_reports/" 2>/dev/null || true
  [ -f qa_reports/infra_incidents.txt ] && cp qa_reports/infra_incidents.txt "$A/qa_reports/"
  [ -n "$(ls -A qa_reports/credit_paused 2>/dev/null)" ] && cp -r qa_reports/credit_paused "$A/qa_reports/"
  cp -r qa_reports/correction_episodes/"${CASE}"_*_logs "$A/qa_reports/correction_episodes/" 2>/dev/null || true
  [ -d logs ] && cp -r logs "$A/"
  echo "$rc" > "/out/_lane/$TAG.rc"
}
trap collect EXIT

pip freeze --exclude-editable > "qa_reports/correction_episodes/${CASE}_env_logs/pip-freeze.txt" || true
python --version > "qa_reports/correction_episodes/${CASE}_env_logs/python-version.txt" 2>&1

PAUSES=qa_reports/credit_pauses.jsonl; : > "$PAUSES"
MARK='run out of credit|insufficient_quota|Error code: 402|would exceed your available credits|LLM BILLING ERROR'
# OpenRouter's in-flight-budget 402 (limit_source openrouter_in_flight_budget) is transient: from CIRISAgent #1366
# item 20 the engine waits Retry-After and retries it, so it does not stop the episode. Only a terminal credit failure
# does. Set M4_INFLIGHT_STOP=1 to keep stopping on in-flight 402s for engines without item 20.
INFLIGHT='in_flight_budget|openrouter_in_flight_budget|given your current in-flight requests'
credit_lines() {
  { grep -hsE "$MARK" "qa_reports/correction_episodes/$CASE.jsonl" "qa_reports/episode_$CASE.log" 2>/dev/null
    [ -d logs ] && grep -rhsE "$MARK" logs/ 2>/dev/null; } || true
}
credit_hit() {
  if [ "${M4_INFLIGHT_STOP:-0}" = 1 ]; then credit_lines | grep -q . && return 0; return 1; fi
  credit_lines | grep -vE "$INFLIGHT" | grep -q . && return 0
  return 1
}
inflight_count() { credit_lines | grep -cE "$INFLIGHT" || true; }
provider_ok() {   # 1-token request to the same model; prints the HTTP status
  python3 - "$MODEL_INPUT" <<'PY'
import json, os, sys, urllib.request, urllib.error
key = open(os.path.expanduser("~/.openrouter_key")).read().strip()
req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions",
    data=json.dumps({"model": sys.argv[1], "max_tokens": 1,
                     "messages": [{"role": "user", "content": "ok"}]}).encode(),
    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
try:
    print(urllib.request.urlopen(req, timeout=60).status)
except urllib.error.HTTPError as e:
    print(e.code)
except Exception:
    print(0)
PY
}
key_remaining() {  # dollars left on the key's limit; "none" when the key has no limit
  python3 - <<'PY'
import json, os, urllib.request
key = open(os.path.expanduser("~/.openrouter_key")).read().strip()
try:
    d = json.load(urllib.request.urlopen(urllib.request.Request("https://openrouter.ai/api/v1/key",
        headers={"Authorization": f"Bearer {key}"}), timeout=30))["data"]
    print("none" if d.get("limit_remaining") is None else d["limit_remaining"])
except Exception:
    print("unknown")
PY
}
log_pause() { printf '{"t":"%s","case":"%s","attempt":%d,"event":"%s"%s}\n' "$(date -u +%FT%TZ)" "$CASE" "$ATTEMPT" "$1" "${2:-}" >> "$PAUSES"; }

WAITED=0; ATTEMPT=0
while :; do
  ATTEMPT=$((ATTEMPT+1))
  rm -rf logs "qa_reports/correction_episodes/$CASE.jsonl"
  for d in qa_reports/correction_episodes/"${CASE}"_*_logs; do
    [ -e "$d" ] && [ "$d" != "qa_reports/correction_episodes/${CASE}_env_logs" ] && rm -rf "$d"
  done
  # Mock mode (context map): no key, no provider; everything else identical, so prompts are assembled as live.
  if [ -n "${M4_MOCK:-}" ]; then
    LLM_ARGS='--mock-llm'
  else
    LLM_ARGS='--live --live-key-file '"$HOME"'/.openrouter_key --live-base-url https://openrouter.ai/api/v1 --live-model '"$MODEL_INPUT"
  fi
  setsid bash -c 'exec timeout --signal=TERM --kill-after=60 "$0" python3 -m tools.qa_runner correction_episodes \
      --correction-cases-file /tmp/m4-cases/freeze-cases.jsonl --correction-case-id "$1" \
      $2 --safety-battery-template "$3" \
      --wipe-data --proceed-anyway --timeout 7200 --verbose 2>&1' \
    "$M4_ATTEMPT_TIMEOUT_S" "$CASE" "$LLM_ARGS" "$TEMPLATE_INPUT" > "qa_reports/episode_$CASE.log" 2>&1 &
  PG=$!; HIT=0
  while kill -0 "$PG" 2>/dev/null; do
    sleep 15
    if credit_hit; then
      HIT=1; echo "::notice::provider credit pause in attempt $ATTEMPT of $CASE: stopping the episode, will rerun it from a wiped start"
      kill -TERM -- "-$PG" 2>/dev/null || true; sleep 30; kill -KILL -- "-$PG" 2>/dev/null || true; break
    fi
  done
  RC=0; wait "$PG" || RC=$?
  [ "$HIT" = 0 ] && credit_hit && HIT=1
  pkill -f "$PWD/main.py" 2>/dev/null || true   # container-local: only this episode's agent
  cat "qa_reports/episode_$CASE.log"
  if [ "$HIT" = 0 ]; then
    # The runner's incidents gate fails an episode on provider/runtime transients (connection errors, DMA or interact
    # timeouts, log floods) even when it produced a verdict. Those are infrastructure, reported, not behavioural
    # failures: with a verdict in the raw log, the episode stays green and the incident lines go to
    # qa_reports/infra_incidents.txt and the job summary. No verdict: still red.
    RAW="qa_reports/correction_episodes/$CASE.jsonl"
    if [ "$RC" != 0 ] && grep -qs '"event": "verdict"' "$RAW"; then
      grep -h '\[INCIDENTS-GATE\] FAILED' "qa_reports/episode_$CASE.log" > qa_reports/infra_incidents.txt || true
      N=$(wc -l < qa_reports/infra_incidents.txt)
      log_pause done_with_infra_incidents ",\"rc\":$RC,\"incidents\":$N"
      echo "::notice::$CASE produced a verdict; runner rc=$RC from $N infrastructure incident line(s), recorded in infra_incidents.txt"
      { echo "### $CASE: verdict produced; infrastructure incidents (not behavioural)"; echo '```';
        cat qa_reports/infra_incidents.txt; echo '```'; } >> "$SUMMARY"
      exit 0
    fi
    log_pause done ",\"rc\":$RC,\"inflight_402_lines\":$(inflight_count)"
    exit "$RC"
  fi
  D="qa_reports/credit_paused/attempt-$ATTEMPT"; mkdir -p "$D"
  cp -r "qa_reports/episode_$CASE.log" logs "qa_reports/correction_episodes/$CASE.jsonl" "$D/" 2>/dev/null || true
  cp -r qa_reports/correction_episodes/${CASE}_*_logs "$D/" 2>/dev/null || true
  log_pause credit_refusal ",\"kept\":\"$D\""
  if [ "$ATTEMPT" -ge "$M4_MAX_ATTEMPTS" ]; then
    echo "::error::$CASE: credit refusals in all $ATTEMPT attempts; no clean episode"; exit 1
  fi
  S=60
  while :; do
    REM=$(key_remaining)
    if [ "$REM" != none ] && [ "$REM" != unknown ] && python3 -c "import sys; sys.exit(0 if float('$REM') < float('$M4_MIN_KEY_REMAINING') else 1)"; then
      log_pause key_limit ",\"remaining\":\"$REM\""
      echo "::error::$CASE: the key's limit is nearly spent (\$$REM left); not waiting for the weekly reset"; exit 1
    fi
    if [ "$WAITED" -ge "$M4_PAUSE_BUDGET_S" ]; then
      log_pause pause_budget_spent ",\"waited_s\":$WAITED"
      echo "::error::$CASE: provider still refusing after ${WAITED}s of waiting"; exit 1
    fi
    echo "credit pause: waiting ${S}s (waited ${WAITED}s so far; key remaining: $REM)"
    sleep "$S"; WAITED=$((WAITED+S))
    CODE=$(provider_ok)
    log_pause probe ",\"status\":$CODE,\"waited_s\":$WAITED"
    [ "$CODE" = 200 ] && break
    S=$(( S*2 > 600 ? 600 : S*2 ))
  done
  echo "provider accepts again after ${WAITED}s; rerunning $CASE from a wiped start (attempt $((ATTEMPT+1)))"
done
