#!/usr/bin/env bash
#
# Exp 1b — 5-Vendor CRCv2 Extension (OpenAI GPT-5 + Anthropic Claude Sonnet 4.6)
#
# Extends the cross-family CRCv2 replication from 3 vendors (Google,
# Meta, Alibaba) to 5 vendors by adding OpenAI and Anthropic. Same
# v4_combined battery, same scoring pipeline.
#
# Pre-registration: experiments/exp1b_boundary_active/RUN_PLAN_5VENDOR_CRCV2.md
# Lake authority:   formal/RATCHET/Experiments/OverrideRate.lean
#                   (OR-1, OR-2, RA-1 predicates + equivalence theorem)
#
# WHY a separate script and data directory:
#   - Keeps the locked 3-vendor anchor (data/crossfamily/) untouched
#   - 5-vendor results land in data/crossfamily_5vendor/
#   - If/when the 5-vendor result passes, the synthesis paper §5.3 +
#     §7 stratification update consolidates both directories into one
#     "5-family cross-vendor cohort" table.
#
# Models (premium tier — ~3-4× cheap-tier per-chain cost):
#   - openai/gpt-5
#   - anthropic/claude-sonnet-4.6
#
# Battery: v4_combined_boundary_active.json (14 questions; same as the
#          existing 3-vendor anchor → cohorts are apples-to-apples).
#
# Target: ~330 chains/model = 24 iters × 14 questions. Tight enough on
#         binomial CIs for OR-1/OR-2 across all three predicates.
#
# Cost ceiling: $66/model at $0.20/chain premium-tier; $130 total. Well
#               inside the $300 OpenRouter budget at ~/.ratchet_openrouter_key.
#
# Time: ~3-4h wall per model — kick off in background and re-check.
#
# Usage:
#   ./run_crossfamily_5vendor.sh                  # full run (24 iters × 2 models)
#   ITERS=1 ./run_crossfamily_5vendor.sh          # mandatory smoke test (~$5)
#   MODELS_ONLY=gpt-5 ./run_crossfamily_5vendor.sh  # one model only
#
# Premium-tier rate-limit pause (default 60s between iters):
#   PREMIUM_SLEEP=120 ./run_crossfamily_5vendor.sh   # bump if 429s seen
set -euo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KEY_FILE="${HOME}/.ratchet_openrouter_key"
QUESTIONS_FILE="${EXPERIMENT_DIR}/questions/v4_combined_boundary_active.json"
AGENT_REPO="${AGENT_REPO:-${HOME}/CIRISAgent}"
DATA_DIR="${EXPERIMENT_DIR}/data/crossfamily_5vendor"

ITERS="${ITERS:-24}"
CONCURRENCY="${CONCURRENCY:-1}"
PREMIUM_SLEEP="${PREMIUM_SLEEP:-60}"
CELL_BUDGET_USD="${CELL_BUDGET_USD:-75}"

# Sanity checks
[[ -f "${KEY_FILE}" ]] || { echo "FATAL: ${KEY_FILE} missing"; exit 1; }
[[ -f "${QUESTIONS_FILE}" ]] || { echo "FATAL: ${QUESTIONS_FILE} missing"; exit 1; }
[[ -d "${AGENT_REPO}" ]] || { echo "FATAL: ${AGENT_REPO} missing"; exit 1; }

mkdir -p "${DATA_DIR}"
RUN_LOG="${DATA_DIR}/run.log"
echo "5-Vendor CRCv2 extension run started: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee "${RUN_LOG}"
echo "Iters per model: ${ITERS}  Concurrency: ${CONCURRENCY}  Premium sleep: ${PREMIUM_SLEEP}s" | tee -a "${RUN_LOG}"
echo "Cell budget cap: \$${CELL_BUDGET_USD} per model" | tee -a "${RUN_LOG}"

MODELS=(
    "openai/gpt-5.4"
    "anthropic/claude-sonnet-4.6"
)
# NOTE: openai/gpt-5 (no suffix) is reasoning-mandatory on OpenRouter and
# rejects {"reasoning":{"enabled":false}} with 400. openai/gpt-5.4 is the
# latest non-reasoning-mandatory GPT-5 variant and accepts the existing
# CIRISAgent dispatch (verified 2026-05-16 — emits 0 reasoning tokens).
# See CIRISAI/CIRISAgent PR #769 for the architectural fix that would
# make openai/gpt-5 itself usable (still pending merge).

# Optional model filter
if [[ -n "${MODELS_ONLY:-}" ]]; then
    FILTERED=()
    for m in "${MODELS[@]}"; do
        if [[ "$m" == *"$MODELS_ONLY"* ]]; then
            FILTERED+=("$m")
        fi
    done
    MODELS=("${FILTERED[@]}")
fi

for MODEL in "${MODELS[@]}"; do
    MODEL_TAG="${MODEL//\//-}"
    MODEL_DIR="${DATA_DIR}/${MODEL_TAG}"
    mkdir -p "${MODEL_DIR}/tee" "${MODEL_DIR}/qa_logs"

    echo "" | tee -a "${RUN_LOG}"
    echo "=== ${MODEL} (${ITERS} iters, budget \$${CELL_BUDGET_USD}) ===" | tee -a "${RUN_LOG}"
    echo "Start: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${RUN_LOG}"

    cd "${AGENT_REPO}"

    CELL_COST_TOTAL="0"
    CONSECUTIVE_EMPTY=0

    for ITER in $(seq 1 "${ITERS}"); do
        ITER_TAG=$(printf "iter%03d" "${ITER}")
        TS=$(date -u +%Y%m%dT%H%M%SZ)
        TEE_DIR="/tmp/exp1b-cf5v-${MODEL_TAG}-${ITER_TAG}-${TS}"
        ITER_LOG="${MODEL_DIR}/qa_logs/${ITER_TAG}.log"

        # Cost-cap check BEFORE this iter
        if python3 -c "import sys; sys.exit(0 if float('${CELL_COST_TOTAL}') >= float('${CELL_BUDGET_USD}') else 1)" 2>/dev/null; then
            echo "  ${ITER_TAG}: SKIPPED — cell budget \$${CELL_BUDGET_USD} reached (\$${CELL_COST_TOTAL} spent)" | tee -a "${RUN_LOG}"
            break
        fi

        set +e
        CIRIS_DISABLE_TASK_APPEND=1 \
        CIRIS_ACCORD_METRICS_LOCAL_COPY_DIR="${TEE_DIR}" \
        CIRIS_ACCORD_METRICS_TRACE_LEVELS="detailed,full_traces" \
            python3 -u -m tools.qa_runner model_eval \
                --live \
                --live-key-file "${KEY_FILE}" \
                --live-base-url "https://openrouter.ai/api/v1" \
                --live-model "${MODEL}" \
                --live-lens \
                --model-eval-languages en \
                --model-eval-concurrency "${CONCURRENCY}" \
                --model-eval-questions-file "${QUESTIONS_FILE}" \
                > "${ITER_LOG}" 2>&1
        QA_EXIT=$?
        set -e

        # Find actual tee dir from log (qa_runner may pick its own)
        ACTUAL_TEE=$(grep -oE "CIRIS_ACCORD_METRICS_LOCAL_COPY_DIR=[^ ]+" "${ITER_LOG}" \
            | head -1 | cut -d= -f2 || true)
        [[ -z "${ACTUAL_TEE}" || ! -d "${ACTUAL_TEE}" ]] && ACTUAL_TEE="${TEE_DIR}"

        if [[ -d "${ACTUAL_TEE}" ]]; then
            BATCHES=$(ls "${ACTUAL_TEE}"/*.json 2>/dev/null | wc -l)
            for f in "${ACTUAL_TEE}"/*.json; do
                [ -f "$f" ] || continue
                cp "$f" "${MODEL_DIR}/tee/${ITER_TAG}_$(basename "$f")"
            done
        else
            BATCHES=0
        fi

        # Roll up cost via iter_cost.py
        set +e
        COST_THIS_ITER=$(python3 \
            "${EXPERIMENT_DIR}/../exp1_multimodel_neff/iter_cost.py" \
            "${MODEL_DIR}/tee/${ITER_TAG}_*.json" 2>/dev/null || echo "0.0000")
        CELL_COST_TOTAL=$(python3 -c "print(f'{float(\"${CELL_COST_TOTAL}\") + float(\"${COST_THIS_ITER}\"):.4f}')" 2>/dev/null || echo "${CELL_COST_TOTAL}")
        set -e

        echo "  ${ITER_TAG}: exit=${QA_EXIT}  batches=${BATCHES}  iter_cost=\$${COST_THIS_ITER}  cell_total=\$${CELL_COST_TOTAL}/\$${CELL_BUDGET_USD}" | tee -a "${RUN_LOG}"

        # Empty-iter abort (matches existing crossfamily harness)
        if [[ "${BATCHES}" -eq 0 ]]; then
            CONSECUTIVE_EMPTY=$((CONSECUTIVE_EMPTY + 1))
            echo "  WARNING: empty iter (${CONSECUTIVE_EMPTY}/3 consecutive)" | tee -a "${RUN_LOG}"
            if [[ "${CONSECUTIVE_EMPTY}" -ge 3 ]]; then
                echo "  ABORT: 3 consecutive empty iters on ${MODEL}" | tee -a "${RUN_LOG}"
                break
            fi
        else
            CONSECUTIVE_EMPTY=0
        fi

        # Premium-tier rate-limit pause (skip after final iter)
        if [[ "${ITER}" -lt "${ITERS}" ]]; then
            sleep "${PREMIUM_SLEEP}"
        fi
    done

    # Per-model manifest
    python3 - <<PY > "${MODEL_DIR}/manifest.json"
import json, hashlib, os, glob
d = "${MODEL_DIR}/tee"
rows = []
for f in sorted(glob.glob(os.path.join(d, "*.json"))):
    h = hashlib.sha256(open(f, "rb").read()).hexdigest()
    rows.append({"name": os.path.basename(f), "bytes": os.path.getsize(f), "sha256": h})
print(json.dumps({
    "model": "${MODEL}",
    "model_tag": "${MODEL_TAG}",
    "iterations_requested": ${ITERS},
    "iterations_completed": len(set(r["name"].split("_")[0] for r in rows if "_" in r["name"])),
    "batch_count": len(rows),
    "cell_cost_total_usd": float("${CELL_COST_TOTAL}"),
    "cell_budget_usd": ${CELL_BUDGET_USD},
    "batches": rows,
}, indent=2))
PY

    BATCH_TOTAL=$(ls "${MODEL_DIR}/tee"/*.json 2>/dev/null | wc -l)
    echo "End: $(date -u +%Y-%m-%dT%H:%M:%SZ)  total batches=${BATCH_TOTAL}  cell cost=\$${CELL_COST_TOTAL}" | tee -a "${RUN_LOG}"
done

echo "" | tee -a "${RUN_LOG}"
echo "5-Vendor CRCv2 run complete: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${RUN_LOG}"
echo "" | tee -a "${RUN_LOG}"
echo "Next: run analyze_crossfamily on data/crossfamily_5vendor/ to verify OR-1/OR-2/RA-1 + N_eff_H." | tee -a "${RUN_LOG}"
echo "  python3 experiments/exp1b_boundary_active/analyze_crossfamily.py \\" | tee -a "${RUN_LOG}"
echo "    --data-dir experiments/exp1b_boundary_active/data/crossfamily_5vendor" | tee -a "${RUN_LOG}"
