#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export QWEN_SAFE_AGENT_LORA_PATH="${QWEN_SAFE_AGENT_LORA_PATH:-../SR-Agent}"
export QWEN_SAFE_AGENT_SYS_APPEND="${QWEN_SAFE_AGENT_SYS_APPEND:-1}"
export QWEN_SAFE_AGENT_ENABLE_THINKING="${QWEN_SAFE_AGENT_ENABLE_THINKING:-1}"

export HF_HOME="${HF_HOME:-.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HUGGINGFACE_HUB_CACHE}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE"

RUN_LOGDIR="${RUN_LOGDIR:-runs/qwen3_8b_safe_agent_sr_agent_think_sys_append}"
CHUNK_LOG_ROOT="${CHUNK_LOG_ROOT:-logs/injection_chunks_qwen_safe_agent_8b_sr_agent_think_sys_append}"
CHUNKS="${CHUNKS:-4}"
SUITES="${SUITES:-travel slack banking workspace}"
ATTACK="${ATTACK:-important_instructions}"

mkdir -p "$RUN_LOGDIR" "$CHUNK_LOG_ROOT"

echo "PYTHON=$(command -v python)"
echo "PYTHONPATH=$PYTHONPATH"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "QWEN_SAFE_AGENT_LORA_PATH=$QWEN_SAFE_AGENT_LORA_PATH"
echo "QWEN_SAFE_AGENT_SYS_APPEND=$QWEN_SAFE_AGENT_SYS_APPEND"
echo "QWEN_SAFE_AGENT_ENABLE_THINKING=$QWEN_SAFE_AGENT_ENABLE_THINKING"
echo "RUN_LOGDIR=$RUN_LOGDIR"
echo "CHUNK_LOG_ROOT=$CHUNK_LOG_ROOT"
echo "CHUNKS=$CHUNKS"
echo "SUITES=$SUITES"
echo "ATTACK=$ATTACK"

for suite in $SUITES; do
  suite_chunk_log_dir="$CHUNK_LOG_ROOT/$suite"
  mkdir -p "$suite_chunk_log_dir"

  echo "Starting suite: $suite"
  python src/agentdojo/scripts/run_workspace_injection_chunks.py \
    --model QWEN_3_8B_SAFE_AGENT \
    --suite "$suite" \
    --attack "$ATTACK" \
    --chunks "$CHUNKS" \
    --logdir "$RUN_LOGDIR" \
    --chunk-log-dir "$suite_chunk_log_dir"
  echo "Finished suite: $suite"
done
