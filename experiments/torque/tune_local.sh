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

python3 build_he300_arcs.py --n-arcs 1 --turns 10 --seed 42 \
  --stratum "$STRATUM" --ethics "$ETHICS" --safety-dir "$AGENT/tests/safety" >/dev/null
DOMAIN="he300_${STRATUM}_a00"

mkdir -p "$AGENT/docker/manifests" "$LOGS"
cp "arms/$ARM.json" "$AGENT/docker/manifests/manifest.json"
rm -f "$LOGS"/* 2>/dev/null || true

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
docker compose -f docker-compose.research.yml run --name "$CID" --build \
  --entrypoint bash \
  -e OVERRIDES=/manifests/manifest.json \
  -e CIRIS_RESEARCH_PROMPT_OVERRIDES=/manifests/manifest.json \
  -e CIRIS_TESTING_MODE=true \
  -e CIRIS_ACCORD_METRICS_CEG_SEAL_TEE=false \
  -v "$KEY:/keys/key:ro" \
  -v "$AGENT/docker/manifests:/manifests:ro" \
  -v "$LOGS:/work/ciris/logs" \
  capture -lc "cd /app && python3 -u -m tools.qa_runner safety_battery \
      --live --live-key-file /keys/key --live-provider openai \
      --live-model '$MODEL' --live-base-url https://api.deepinfra.com/v1/openai \
      --safety-battery-lang en --safety-battery-domain '$DOMAIN' \
      --safety-battery-template he-300-benchmark --verbose" || true

docker cp "$CID:/app/qa_reports/safety_battery/." "$RESULTS/" 2>/dev/null \
  && echo "results -> $RESULTS" || echo "no qa_reports in container"
docker rm -f "$CID" >/dev/null 2>&1 || true
ls "$RESULTS" | grep en_he300 | tail -1
