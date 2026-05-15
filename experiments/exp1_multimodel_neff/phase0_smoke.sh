#!/usr/bin/env bash
#
# Exp 1 — Phase 0 qa_runner smoke (PRE-PHASE-1 GATING)
#
# Runs CIRIS qa_runner model_eval against each of the 5 pre-registered models
# with 2 questions only, in live mode against OpenRouter, with --live-lens so
# traces flow to production lens (federation-visible) and local-tee captures
# every ACCORD batch.
#
# Purpose: end-to-end harness validation before Phase 1 sweep. Confirms:
#   1. CIRIS pipeline runs cleanly with each model (no harness errors)
#   2. Reasoning is disabled per CIRIS LLM service's _build_reasoning_off_extras
#   3. Lens receives the traces (production endpoint)
#   4. Local-tee captures batches for offline analysis
#   5. N_eff computes to a real value (any value — pipeline sanity, not the test)
#
# Cost cap: ~$5 across all 5 models × 2 questions.
# Runtime: 15–30 minutes wall clock.
#
# Usage:
#   ./phase0_smoke.sh
#
# Outputs:
#   experiments/exp1_multimodel_neff/data/smoke/<model>/  (per-model)
#     ├── qa_runner.log         qa_runner stdout
#     ├── tee_batches/          live-lens local-tee copies
#     └── tee_manifest.json     sha256 + count of batches captured
#
set -euo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${EXPERIMENT_DIR}/data/smoke"
KEY_FILE="${HOME}/.ratchet_openrouter_key"
QUESTIONS_FILE="${HOME}/bounce-test/model_eval_questions/v1_sensitive.json"
AGENT_REPO="${HOME}/CIRISAgent"

# Pre-registered model lineup (matches PRE_REGISTRATION.md §5)
MODELS=(
    "qwen/qwen3.5-35b-a3b"
    "anthropic/claude-opus-4.7"
    "openai/gpt-5.5"
    "google/gemini-2.5-flash"
    "meta-llama/llama-4-scout"
)

# Smoke: 2 categories only (Theology + Politics — first two in v1_sensitive.json)
CATEGORIES="Theology,Politics"

# Sanity checks
[[ -f "${KEY_FILE}" ]] || { echo "FATAL: ${KEY_FILE} missing"; exit 1; }
[[ -f "${QUESTIONS_FILE}" ]] || { echo "FATAL: ${QUESTIONS_FILE} missing"; exit 1; }
[[ -d "${AGENT_REPO}" ]] || { echo "FATAL: ${AGENT_REPO} missing"; exit 1; }

mkdir -p "${DATA_DIR}"
SMOKE_LOG="${DATA_DIR}/smoke_run.log"
echo "Phase 0 smoke run started: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee "${SMOKE_LOG}"

for MODEL in "${MODELS[@]}"; do
    # Sanitize the model id into a dirname (slash → dash)
    MODEL_TAG="${MODEL//\//-}"
    MODEL_DIR="${DATA_DIR}/${MODEL_TAG}"
    mkdir -p "${MODEL_DIR}"

    echo "" | tee -a "${SMOKE_LOG}"
    echo "=== ${MODEL} ===" | tee -a "${SMOKE_LOG}"
    echo "Start: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${SMOKE_LOG}"

    # Tee dir is timestamped by qa_runner; capture the env var to know where
    TEE_DIR="/tmp/exp1-smoke-${MODEL_TAG}-$(date -u +%Y%m%dT%H%M%SZ)"

    # Run qa_runner from the agent repo
    cd "${AGENT_REPO}"

    # The local-tee is auto-enabled when --live-lens is set.
    # qa_runner prints the tee directory; we capture it from the log.
    set +e
    CIRIS_ACCORD_METRICS_LOCAL_COPY_DIR="${TEE_DIR}" \
        python3 -m tools.qa_runner model_eval \
            --live \
            --live-key-file "${KEY_FILE}" \
            --live-base-url "https://openrouter.ai/api/v1" \
            --live-model "${MODEL}" \
            --live-lens \
            --model-eval-languages en \
            --model-eval-concurrency 1 \
            --model-eval-questions "${CATEGORIES}" \
            --model-eval-questions-file "${QUESTIONS_FILE}" \
            2>&1 | tee "${MODEL_DIR}/qa_runner.log"
    QA_EXIT=$?
    set -e

    echo "End: $(date -u +%Y-%m-%dT%H:%M:%SZ)  exit=${QA_EXIT}" | tee -a "${SMOKE_LOG}"

    # Locate the actual tee dir (qa_runner sometimes ignores the env var
    # and picks its own timestamped path; harvest from log if so)
    ACTUAL_TEE_DIR=$(grep -oE "CIRIS_ACCORD_METRICS_LOCAL_COPY_DIR=[^ ]+" "${MODEL_DIR}/qa_runner.log" \
        | head -1 | cut -d= -f2 || true)
    if [[ -z "${ACTUAL_TEE_DIR}" || ! -d "${ACTUAL_TEE_DIR}" ]]; then
        ACTUAL_TEE_DIR="${TEE_DIR}"
    fi

    if [[ -d "${ACTUAL_TEE_DIR}" ]]; then
        cp -r "${ACTUAL_TEE_DIR}" "${MODEL_DIR}/tee_batches"
        # Manifest
        python3 - <<PYEOF > "${MODEL_DIR}/tee_manifest.json"
import json, hashlib, os
batches = []
tee_dir = "${MODEL_DIR}/tee_batches"
for root, _, files in os.walk(tee_dir):
    for f in sorted(files):
        if f.endswith(".json"):
            p = os.path.join(root, f)
            h = hashlib.sha256(open(p, "rb").read()).hexdigest()
            batches.append({"path": os.path.relpath(p, tee_dir),
                            "size": os.path.getsize(p),
                            "sha256": h})
print(json.dumps({"model": "${MODEL}", "batch_count": len(batches),
                  "batches": batches}, indent=2))
PYEOF
        BATCH_COUNT=$(python3 -c "import json; print(json.load(open('${MODEL_DIR}/tee_manifest.json'))['batch_count'])")
        echo "Captured ${BATCH_COUNT} tee batches in ${MODEL_DIR}/tee_batches/" | tee -a "${SMOKE_LOG}"
    else
        echo "WARN: no tee dir found for ${MODEL}" | tee -a "${SMOKE_LOG}"
    fi
done

echo "" | tee -a "${SMOKE_LOG}"
echo "Phase 0 smoke run complete: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${SMOKE_LOG}"
echo "" | tee -a "${SMOKE_LOG}"
echo "Next: run phase0_smoke_analyze.py to verify N_eff computes cleanly per model." | tee -a "${SMOKE_LOG}"
