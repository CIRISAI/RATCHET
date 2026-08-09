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
cd "$AGENT/docker"
RESULTS="${RESULTS:-/tmp/torque-results/$ARM/$DOMAIN}"
mkdir -p "$RESULTS"

# `qa_reports/` lives INSIDE /app, so it cannot be bind-mounted out without
# tripping verify_manifest_integrity. Keep the container alive after the run and
# `docker cp` the reports, rather than mounting over the app tree.
CID="torque-$ARM-$DOMAIN-$$"
docker compose -f docker-compose.research.yml run --name "$CID" --build \
  -e PROVIDER=deepinfra \
  -e BASE_URL=https://api.deepinfra.com/v1/openai \
  -e MODEL="$MODEL" \
  -e MODULE="$MODULE" \
  -e BATTERY_DOMAIN="$DOMAIN" \
  -e BATTERY_TEMPLATE=he-300-benchmark \
  -e LANGUAGES=en \
  -e CONCURRENCY=1 \
  -e OVERRIDES=/manifests/manifest.json \
  -e API_KEY_FILE=/keys/key \
  -v "$KEY:/keys/key:ro" \
  -v "$AGENT/docker/manifests:/manifests:ro" \
  -v "$LOGS:/work/ciris/logs" \
  capture || true

docker cp "$CID:/app/qa_reports/safety_battery/." "$RESULTS/" 2>/dev/null \
  && echo "results -> $RESULTS" || echo "no qa_reports in container"
docker rm -f "$CID" >/dev/null 2>&1 || true
find "$RESULTS" -name results.jsonl | head -3
