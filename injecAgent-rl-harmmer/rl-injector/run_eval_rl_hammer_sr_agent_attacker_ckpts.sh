#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-rl-hammer}"
if [ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV_NAME" ]; then
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV_NAME"
  else
    echo "Conda is not available. Activate the '$CONDA_ENV_NAME' environment first." >&2
    exit 1
  fi
fi

unset VIRTUAL_ENV
unset PYTHONHOME
hash -r

mkdir -p logs

export HF_HOME="${HF_HOME:-.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HUGGINGFACE_HUB_CACHE}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export VLLM_CONFIG_ROOT="${VLLM_CONFIG_ROOT:-$HF_HOME/vllm}"
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$VLLM_CONFIG_ROOT"

export WANDB_PROJECT="${WANDB_PROJECT:-RL-Hammer}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export PYTHONUNBUFFERED=1
export PRINT_TARGET_INPUTS="${PRINT_TARGET_INPUTS:-1}"
export PRINT_TARGET_INPUTS_MAX_N="${PRINT_TARGET_INPUTS_MAX_N:-3}"

ATTACKER_BASE_MODEL_NAME_OR_PATH="${ATTACKER_BASE_MODEL_NAME_OR_PATH:-meta-llama/Llama-3.1-8B-Instruct}"
ATTACKER_CHECKPOINT_DIR="${ATTACKER_CHECKPOINT_DIR:-checkpoints/rl_hammer_sr_agent_attacker}"

QWEN_SAFE_AGENT_LORA_PATH="${QWEN_SAFE_AGENT_LORA_PATH:-../../SR-Agent}"
QWEN_SAFE_AGENT_TARGET_BASE_MODEL_NAME_OR_PATH="${QWEN_SAFE_AGENT_TARGET_BASE_MODEL_NAME_OR_PATH:-Qwen/Qwen3-8B}"

VALIDATION_DATA_PATH="${VALIDATION_DATA_PATH:-data/InjecAgent/dataset/test.json}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-1}"
VAL_MAX_NEW_TOKENS="${VAL_MAX_NEW_TOKENS:-1024}"
ENABLE_WANDB="${ENABLE_WANDB:-False}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-sr_agent_all}"
SAVE_NAME_PREFIX="${SAVE_NAME_PREFIX:-sr_agent_all}"

if [ ! -d "$QWEN_SAFE_AGENT_LORA_PATH" ]; then
  echo "SR-Agent LoRA adapter not found: $QWEN_SAFE_AGENT_LORA_PATH" >&2
  exit 1
fi

if [ ! -f "$QWEN_SAFE_AGENT_LORA_PATH/adapter_model.safetensors" ]; then
  echo "SR-Agent path does not look like a LoRA adapter: $QWEN_SAFE_AGENT_LORA_PATH" >&2
  exit 1
fi

if [ ! -d "$ATTACKER_CHECKPOINT_DIR" ]; then
  echo "Attacker checkpoint directory not found: $ATTACKER_CHECKPOINT_DIR" >&2
  exit 1
fi

mapfile -t CHECKPOINT_DIRS < <(
  find "$ATTACKER_CHECKPOINT_DIR" -maxdepth 1 -mindepth 1 -type d -name 'checkpoint-*' | sort -V
)

if [ "${#CHECKPOINT_DIRS[@]}" -eq 0 ]; then
  echo "No checkpoint-* directories found under: $ATTACKER_CHECKPOINT_DIR" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES

echo "=== Runtime Diagnostics ==="
echo "CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-<unset>}"
echo "PYTHON=$(command -v python)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "ATTACKER_BASE_MODEL_NAME_OR_PATH=$ATTACKER_BASE_MODEL_NAME_OR_PATH"
echo "ATTACKER_CHECKPOINT_DIR=$ATTACKER_CHECKPOINT_DIR"
echo "QWEN_SAFE_AGENT_LORA_PATH=$QWEN_SAFE_AGENT_LORA_PATH"
echo "QWEN_SAFE_AGENT_TARGET_BASE_MODEL_NAME_OR_PATH=$QWEN_SAFE_AGENT_TARGET_BASE_MODEL_NAME_OR_PATH"
echo "VALIDATION_DATA_PATH=$VALIDATION_DATA_PATH"
echo "VAL_BATCH_SIZE=$VAL_BATCH_SIZE"
echo "VAL_MAX_NEW_TOKENS=$VAL_MAX_NEW_TOKENS"
echo "ENABLE_WANDB=$ENABLE_WANDB"
echo "RUN_NAME_PREFIX=$RUN_NAME_PREFIX"
echo "SAVE_NAME_PREFIX=$SAVE_NAME_PREFIX"
echo "PRINT_TARGET_INPUTS=$PRINT_TARGET_INPUTS"
echo "PRINT_TARGET_INPUTS_MAX_N=$PRINT_TARGET_INPUTS_MAX_N"
echo "SAFE_AGENT_MODE=True"
echo "USE_SAFE_AGENT_SYSTEM_APPEND=True"
echo "TARGET_ENABLE_THINKING=True"
echo "NUM_CHECKPOINTS=${#CHECKPOINT_DIRS[@]}"

nvidia-smi || true

for checkpoint_dir in "${CHECKPOINT_DIRS[@]}"; do
  ATTACKER_CHECKPOINT_NAME="$(basename "$checkpoint_dir")"
  ATTACKER_LORA_PATH="$ATTACKER_CHECKPOINT_DIR/${ATTACKER_CHECKPOINT_NAME}-lora"
  RUN_SAFE_NAME="${RUN_NAME_PREFIX}_${ATTACKER_CHECKPOINT_NAME}"
  SAVE_NAME="${SAVE_NAME_PREFIX}_${ATTACKER_CHECKPOINT_NAME}"

  if [ ! -f "$checkpoint_dir/adapter_model.safetensors" ]; then
    echo "Skipping non-LoRA checkpoint: $checkpoint_dir" >&2
    continue
  fi

  ln -sfn "$ATTACKER_CHECKPOINT_NAME" "$ATTACKER_LORA_PATH"

  echo "=== Evaluating $ATTACKER_CHECKPOINT_NAME ==="
  echo "ATTACKER_CHECKPOINT_PATH=$checkpoint_dir"
  echo "ATTACKER_LORA_PATH=$ATTACKER_LORA_PATH"
  echo "RUN_SAFE_NAME=$RUN_SAFE_NAME"
  echo "SAVE_NAME=$SAVE_NAME"

  python injecagent_eval.py \
    --attacker_model_name_or_path "$ATTACKER_LORA_PATH" \
    --attacker_base_model_name_or_path "$ATTACKER_BASE_MODEL_NAME_OR_PATH" \
    --target_model_name_or_path "$QWEN_SAFE_AGENT_LORA_PATH" \
    --target_base_model_name_or_path "$QWEN_SAFE_AGENT_TARGET_BASE_MODEL_NAME_OR_PATH" \
    --use_safe_agent_system_append True \
    --safe_agent_mode True \
    --target_enable_thinking True \
    --validation_data_path "$VALIDATION_DATA_PATH" \
    --val_batch_size "$VAL_BATCH_SIZE" \
    --val_max_new_tokens "$VAL_MAX_NEW_TOKENS" \
    --enable_wandb "$ENABLE_WANDB" \
    --save_name "$SAVE_NAME" \
    --run_name "$RUN_SAFE_NAME"
done
