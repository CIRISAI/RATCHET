#!/usr/bin/env bash
# Run ONE arm against ONE real HE-300 arc, in an isolated container.
#
# THE RULES THIS ENCODES, each learned by breaking it:
#   * EVERY knob through -e. Compose forwards only what its own environment:
#     block names. OVERRIDES as a shell var never arrived and an UNMODIFIED
#     AGENT ran wearing an arm's name, reporting success. MODULE went the same
#     way and silently ran model_eval.
#   * NEVER a comment inside a \ continuation — it detaches the env prefix and
#     compose falls back to openrouter while holding a DeepInfra key.
#   * ALWAYS --build. Dockerfile.research does COPY . /app, so a host edit is
#     invisible to a stale image.
#   * NEVER bind-mount into /app. The agent verifies its own file tree at
#     startup; an added path fails verify_manifest_integrity. Arcs go into the
#     build context.
#   * BUILD FROM A PRISTINE CLONE. .dockerignore excludes data/ but NOT
#     identity/, so building from a working checkout bakes every accumulated
#     bootstrap identity in and the container inherits the federation failure it
#     exists to escape.
#
#   ./tune_local.sh h3ere-blank
#   STRATUM=discriminant_control ./tune_local.sh h3ere-alt
set -euo pipefail
cd "$(dirname "$0")"

ARM="${1:-h3ere-ciris}"
MODULE="${MODULE:-safety_battery}"
STRATUM="${STRATUM:-axiotic_primary}"
SRC="${SRC:-/tmp/a2913}"
AGENT="${AGENT:-/tmp/torque-agent}"
AGENT_REF="${AGENT_REF:-v2.9.13-stable}"
KEY="${KEY:-$HOME/.deepinfra_key}"
MODEL="${MODEL:-meta-llama/Llama-4-Scout-17B-16E-Instruct}"
ETHICS="${ETHICS:-/home/emoore/CIRISBench/engine/datasets/ethics}"
LOGS="${LOGS:-/tmp/torque-agentlogs}"

[ -f "arms/$ARM.json" ] || { echo "no manifest: arms/$ARM.json"; exit 2; }

if [ ! -d "$AGENT/.git" ]; then
  echo "cloning $AGENT_REF -> $AGENT"
  git clone --quiet --depth 1 --branch "$AGENT_REF" "$SRC" "$AGENT" 2>/dev/null \
    || git clone --quiet --depth 1 --branch "$AGENT_REF" \
         https://github.com/CIRISAI/CIRISAgent.git "$AGENT"
fi
rm -rf "$AGENT/identity" "$AGENT/data" "$AGENT/logs"

# DOMAIN may be set to drive a SHIPPED cell (e.g. mental_health) instead of a
# generated HE-300 arc. Only build arcs when we are actually going to run one —
# building unconditionally rewrote tests/safety on every battery-gate run.
if [ -z "${DOMAIN:-}" ]; then
  python3 build_he300_arcs.py --n-arcs 1 --turns 10 --seed 42 \
    --stratum "$STRATUM" --ethics "$ETHICS" --safety-dir "$AGENT/tests/safety" >/dev/null
  DOMAIN="he300_${STRATUM}_a00"
fi
TEMPLATE="${TEMPLATE:-he-300-benchmark}"

# EXPERIMENTAL asymmetric recall (RATCHET#20). Unset on both = the shipped path,
# so control and treatment come from ONE image and differ by these two vars
# alone. Passing them only when set keeps the control byte-identical to stock.
RECALL_ENV=()
if [ -n "${CIRIS_RECALL_NONAGENT:-}" ] && [ -n "${CIRIS_RECALL_AGENT:-}" ]; then
  RECALL_ENV=(-e "CIRIS_RECALL_NONAGENT=$CIRIS_RECALL_NONAGENT"
              -e "CIRIS_RECALL_AGENT=$CIRIS_RECALL_AGENT")
  echo "recall split: ${CIRIS_RECALL_NONAGENT} non-agent / ${CIRIS_RECALL_AGENT} agent"
else
  echo "recall: SHIPPED (20 mixed)"
fi

# PROMPT CAPTURE — on by default. `llm_bus._maybe_capture` writes one JSONL row
# per LLM call carrying the EXACT wire-format `messages`, plus handler,
# thought_id, task_id and the parsed result.
#
# WHY IT IS ON BY DEFAULT NOW. We had CEG trace capture (attestation carriers)
# and results.jsonl (responses) and mistook that for prompt capture. It is not:
# neither records what the model was actually asked. Two published mechanism
# claims died on that gap — a length "collapse" that was a first-message privacy
# notice, and a self-conditioning hypothesis for a history path that returns []
# on every call. One captured prompt would have shown `0 history messages`
# immediately, before either claim was written down.
#
# `*` captures every handler (5 DMAs + consciences). Set CAPTURE=0 to disable,
# or CAPTURE_HANDLER to narrow — the accord rides in every system prompt, so
# expect a few MB per cell and size the staked run accordingly.
# CHANNEL-PER-QUESTION (RATCHET#20). Separates conversation position from agent
# lifetime: a fresh channel resets the former and leaves the latter alone.
PERQ_ENV=()
if [ "${PER_QUESTION:-0}" = "1" ]; then
  PERQ_ENV=(-e "CIRIS_BATTERY_CHANNEL_PER_QUESTION=1")
  echo "channel-per-question: ON"
fi

CAPTURE_ENV=()
if [ "${CAPTURE:-1}" != "0" ]; then
  CAPTURE_ENV=(-e "CIRIS_LLM_CAPTURE_HANDLER=${CAPTURE_HANDLER:-*}"
               -e "CIRIS_LLM_CAPTURE_FILE=/work/ciris/logs/llm_capture.jsonl")
  echo "prompt capture: ON (${CAPTURE_HANDLER:-*}) -> \$LOGS/llm_capture.jsonl"
else
  echo "prompt capture: OFF"
fi

# PER-ARM MANIFEST DIR. This path used to be shared, so two arms running in
# parallel overwrote each other's manifest — and it is bind-mounted, so the
# loser silently ran the winner's prompts under its own name. Exactly the
# failure mode that once produced an unmodified agent wearing an arm's label.
MANIFEST_DIR="${MANIFEST_DIR:-$AGENT/docker/manifests-$ARM}"
mkdir -p "$MANIFEST_DIR" "$LOGS"
# NO MANIFEST = SHIPPED. Any manifest from build_arm_manifest overrides
# ACCORD_KEYS *and* the three identity fields, which would install "Ethical
# Judgment Benchmark" identity into a mental-health or harm battery and replace
# the localized accord with English monoglot. A stale arms/h3ere-ciris.json on
# disk did exactly that to a HARM-1 run before this guard existed.
if [ -f "arms/$ARM.json" ] && [ "${SHIPPED:-0}" != "1" ]; then
  cp "arms/$ARM.json" "$MANIFEST_DIR/manifest.json"
else
  rm -f "$MANIFEST_DIR/manifest.json"
  echo "running SHIPPED (no override manifest)"
fi
rm -f "$LOGS"/* 2>/dev/null || true

MANIFEST_ENV=()
if [ -f "$MANIFEST_DIR/manifest.json" ]; then
  MANIFEST_ENV=(-e OVERRIDES=/manifests/manifest.json
                -e CIRIS_RESEARCH_PROMPT_OVERRIDES=/manifests/manifest.json)
fi

echo "── $ARM · $DOMAIN · $MODULE · $AGENT_REF ──"
RESULTS="${RESULTS:-/tmp/torque-results/$ARM/$DOMAIN}"
mkdir -p "$RESULTS"
CID="torque-${ARM}-$$"

cd "$AGENT/docker"

# BYPASS capture_traces.sh AND RUN qa_runner DIRECTLY.
#
# That script is a TRACE-CAPTURE tool: CEG carriers are its product, so it
# unconditionally exports CIRIS_ACCORD_METRICS_CEG_SEAL_TEE=true. Its own comment
# says what that costs:
#
#   "The tee reads the live persist DB through a second SQLite handle, which is
#    unsafe alongside the Rust writer on a WAL database (it took the staged-QA
#    sqlite leg down), so it is off by default and opted into here"
#
# Off by default, and this script opts in. TORQUE wants answers, not carriers,
# and the observed failure is exactly that warning: the server dies mid-arc with
# no Python traceback, sooner the larger the prompts — 10/10 turns with the
# accord emptied, 4/10 and 3/10 with it present. Bigger seals, more tee traffic,
# more contention on the WAL database.
docker compose -f docker-compose.research.yml run --name "$CID" ${BUILD:---build} \
  --entrypoint bash \
  "${MANIFEST_ENV[@]}" \
  -e CIRIS_TESTING_MODE=true \
  -e CIRIS_ACCORD_METRICS_CEG_SEAL_TEE=false \
  "${RECALL_ENV[@]}" \
  "${CAPTURE_ENV[@]}" \
  "${PERQ_ENV[@]}" \
  -v "$KEY:/keys/key:ro" \
  -v "$MANIFEST_DIR:/manifests:ro" \
  -v "$LOGS:/work/ciris/logs" \
  capture -lc "cd /app && python3 -u -m tools.qa_runner safety_battery \
      --live --live-key-file /keys/key --live-provider openai \
      --live-model '$MODEL' --live-base-url https://api.deepinfra.com/v1/openai \
      --safety-battery-lang en --safety-battery-domain '$DOMAIN' \
      --safety-battery-template '$TEMPLATE' --verbose" || true

docker cp "$CID:/app/qa_reports/safety_battery/." "$RESULTS/" 2>/dev/null \
  && echo "results -> $RESULTS" || echo "no qa_reports in container"
docker rm -f "$CID" >/dev/null 2>&1 || true
ls "$RESULTS" | grep en_he300 | tail -1
