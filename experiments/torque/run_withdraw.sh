#!/usr/bin/env bash
# The reversion arm: one arc across two harnesses, switched at the midpoint.
#
# Runs INSIDE the container, because the agent's API is deliberately unpublished
# (no ports, so concurrent runs cannot fight over one) and the only place that
# reaches both the agent on localhost and the provider on the internet is inside.
#
#   ./run_withdraw.sh h3ere-ciris intact
#   ./run_withdraw.sh h3ere-ciris scrubbed
set -euo pipefail
cd "$(dirname "$0")"

ARM="${1:-h3ere-ciris}"
HISTORY="${2:?intact|scrubbed — running only intact cannot separate persistence from self-matching}"
STRATUM="${STRATUM:-axiotic_primary}"
AGENT="${AGENT:-/tmp/torque-agent}"
KEY="${KEY:-$HOME/.deepinfra_key}"
MODEL="${MODEL:-meta-llama/Llama-4-Scout-17B-16E-Instruct}"
LOGS="${LOGS:-/tmp/torque-agentlogs}"
RESULTS="${RESULTS:-/tmp/torque-results/withdraw-$ARM-$HISTORY}"
DOMAIN="he300_${STRATUM}_a00"
CID="torque-wd-${ARM}-${HISTORY}-$$"

python3 build_he300_arcs.py --n-arcs 1 --turns 10 --seed 42 --stratum "$STRATUM" \
  --ethics /home/emoore/CIRISBench/engine/datasets/ethics \
  --safety-dir "$AGENT/tests/safety" >/dev/null
mkdir -p "$AGENT/docker/manifests" "$LOGS" "$RESULTS"
cp "arms/$ARM.json" "$AGENT/docker/manifests/manifest.json"
cp withdraw_arc.py "$AGENT/docker/manifests/withdraw_arc.py"

echo "── $ARM · $DOMAIN · switch after turn 5 · history=$HISTORY ──"
cd "$AGENT/docker"
docker compose -f docker-compose.research.yml run --name "$CID" --build \
  --entrypoint bash \
  -e CIRIS_RESEARCH_PROMPT_OVERRIDES=/manifests/manifest.json \
  -e CIRIS_TESTING_MODE=true \
  -e CIRIS_ACCORD_METRICS_CEG_SEAL_TEE=false \
  -e CIRIS_TEMPLATE=he-300-benchmark \
  -e CIRIS_BENCHMARK_MODE=true \
  -e CIRIS_API_INTERACTION_TIMEOUT=900 \
  -v "$KEY:/keys/key:ro" \
  -v "$AGENT/docker/manifests:/manifests:ro" \
  -v "$LOGS:/work/ciris/logs" \
  capture -lc "
    cd /app
    python3 -u main.py --port 8080 > /work/ciris/logs/agent.out 2>&1 &
    for i in \$(seq 1 180); do
      curl -sf http://localhost:8080/v1/system/health >/dev/null 2>&1 && break
      sleep 2
    done
    python3 -u /manifests/withdraw_arc.py --domain '$DOMAIN' --history '$HISTORY' \
      --model '$MODEL' --base-url https://api.deepinfra.com/v1/openai \
      --key-file /keys/key --out /app/qa_reports/withdraw
  " || true

docker cp "$CID:/app/qa_reports/withdraw/." "$RESULTS/" 2>/dev/null \
  && echo "results -> $RESULTS" || echo "no withdraw results in container"
docker rm -f "$CID" >/dev/null 2>&1 || true
