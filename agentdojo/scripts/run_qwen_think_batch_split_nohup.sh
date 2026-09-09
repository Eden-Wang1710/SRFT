#!/usr/bin/env bash
set -euo pipefail

# Run two shards with nohup in the background.
# Adjust variables below as needed.

PYTHONPATH_ROOT="${PYTHONPATH_ROOT:-src}"
EXPERT_RUN_DIR="${EXPERT_RUN_DIR:-runs/ground-truth-train}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/qwen3-8b-think-samples-gt-train}"
SUITE="${SUITE:-slack}"
THINKING_BUDGET="${THINKING_BUDGET:-512}"
SECOND_PASS_MAX_NEW_TOKENS="${SECOND_PASS_MAX_NEW_TOKENS:-512}"
NUM_SEQUENCES="${NUM_SEQUENCES:-5}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-8B}"
BENCHMARK_VERSION="${BENCHMARK_VERSION:-v1.2.1}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
MIN_P="${MIN_P:-0.0}"
OVERWRITE="${OVERWRITE:-1}"

SCRIPT_PATH="src/agentdojo/scripts/sample_qwen_think_from_expert_single_batch_split.py"

COMMON_ARGS=(
  "--expert-run-dir" "${EXPERT_RUN_DIR}"
  "--output-dir" "${OUTPUT_DIR}"
  "--suite" "${SUITE}"
  "--thinking-budget" "${THINKING_BUDGET}"
  "--second-pass-max-new-tokens" "${SECOND_PASS_MAX_NEW_TOKENS}"
  "--num-sequences" "${NUM_SEQUENCES}"
  "--model-id" "${MODEL_ID}"
  "--benchmark-version" "${BENCHMARK_VERSION}"
  "--temperature" "${TEMPERATURE}"
  "--top-p" "${TOP_P}"
  "--top-k" "${TOP_K}"
  "--min-p" "${MIN_P}"
  "--split-count" "2"
)

if [[ "${OVERWRITE}" == "1" ]]; then
  COMMON_ARGS+=("--overwrite")
else
  COMMON_ARGS+=("--skip-existing")
fi

LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/nohup_logs}"
mkdir -p "${LOG_DIR}"

nohup env PYTHONPATH="${PYTHONPATH_ROOT}" \
  python "${SCRIPT_PATH}" "${COMMON_ARGS[@]}" --split-index 0 \
  > "${LOG_DIR}/split_0.out" 2>&1 &

nohup env PYTHONPATH="${PYTHONPATH_ROOT}" \
  python "${SCRIPT_PATH}" "${COMMON_ARGS[@]}" --split-index 1 \
  > "${LOG_DIR}/split_1.out" 2>&1 &

echo "Launched shards with nohup. Logs:"
echo "  ${LOG_DIR}/split_0.out"
echo "  ${LOG_DIR}/split_1.out"
