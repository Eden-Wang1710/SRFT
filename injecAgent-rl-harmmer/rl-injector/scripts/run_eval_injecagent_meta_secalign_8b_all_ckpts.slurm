#!/usr/bin/env bash
#SBATCH --job-name=eval_injec_meta_secalign_all
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=a100
#SBATCH --account=aszalay1_ssci
#SBATCH --time=24:00:00
#SBATCH --chdir=/home/cxiao13/scratch-cxiao13/zixuan/safe-agent-project/injecAgent-rl-harmmer/rl-injector

set -euo pipefail

mkdir -p logs

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
  echo "Conda init script not found in ~/miniconda3 or ~/anaconda3" >&2
  exit 1
fi

conda activate rl-hammer
unset VIRTUAL_ENV
unset PYTHONHOME
hash -r
export PATH="$CONDA_PREFIX/bin:$PATH"

cd /home/cxiao13/scratch-cxiao13/zixuan/safe-agent-project/injecAgent-rl-harmmer/rl-injector

export HF_HOME="${HF_HOME:-/scratch/cxiao13/huggingface_cxiao13_access}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_CACHE="$HUGGINGFACE_HUB_CACHE"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export VLLM_CONFIG_ROOT="${VLLM_CONFIG_ROOT:-$HF_HOME/vllm}"
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$VLLM_CONFIG_ROOT"

export WANDB_PROJECT="${WANDB_PROJECT:-RL-Hammer}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export PYTHONUNBUFFERED=1

ATTACKER_BASE_MODEL_NAME_OR_PATH="${ATTACKER_BASE_MODEL_NAME_OR_PATH:-meta-llama/Llama-3.1-8B-Instruct}"
ATTACKER_CHECKPOINT_DIR="${ATTACKER_CHECKPOINT_DIR:-checkpoints/rl_hammer_target_llama_meta_secalign_5gpu_original_para_rerun2}"

META_SECALIGN_TARGET_MODEL_NAME_OR_PATH="${META_SECALIGN_TARGET_MODEL_NAME_OR_PATH:-checkpoints/Meta-SecAlign-8B-merged}"

VALIDATION_DATA_PATH="${VALIDATION_DATA_PATH:-data/InjecAgent/dataset/test.json}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-1}"
VAL_MAX_NEW_TOKENS="${VAL_MAX_NEW_TOKENS:-1024}"
ENABLE_WANDB="${ENABLE_WANDB:-False}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"

RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-eval_rl_hammer_target_meta_secalign_8b_allckpts_rerun2}"
SAVE_NAME_PREFIX="${SAVE_NAME_PREFIX:-meta_secalign_8b_allckpts_rerun2}"

if [ ! -d "$ATTACKER_CHECKPOINT_DIR" ]; then
  echo "Attacker checkpoint directory not found: $ATTACKER_CHECKPOINT_DIR" >&2
  exit 1
fi

if [ ! -d "$META_SECALIGN_TARGET_MODEL_NAME_OR_PATH" ]; then
  echo "Meta-SecAlign target directory not found: $META_SECALIGN_TARGET_MODEL_NAME_OR_PATH" >&2
  exit 1
fi

mapfile -t CHECKPOINT_DIRS < <(
  find "$ATTACKER_CHECKPOINT_DIR" -maxdepth 1 -mindepth 1 -type d -name 'checkpoint-*' | sort -V
)

if [ "${#CHECKPOINT_DIRS[@]}" -eq 0 ]; then
  echo "No checkpoint-* directories found under: $ATTACKER_CHECKPOINT_DIR" >&2
  exit 1
fi

echo "=== Runtime Diagnostics ==="
echo "CONDA_PREFIX=$CONDA_PREFIX"
echo "which python=$(which python)"
echo "python --version=$(python --version 2>&1)"
echo "CUDA_DEVICE=$CUDA_DEVICE"
echo "HF_HOME=$HF_HOME"
echo "HUGGINGFACE_HUB_CACHE=$HUGGINGFACE_HUB_CACHE"
echo "HF_HUB_CACHE=$HF_HUB_CACHE"
echo "HF_DATASETS_CACHE=$HF_DATASETS_CACHE"
echo "TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE"
echo "VLLM_CONFIG_ROOT=$VLLM_CONFIG_ROOT"
echo "ATTACKER_BASE_MODEL_NAME_OR_PATH=$ATTACKER_BASE_MODEL_NAME_OR_PATH"
echo "ATTACKER_CHECKPOINT_DIR=$ATTACKER_CHECKPOINT_DIR"
echo "META_SECALIGN_TARGET_MODEL_NAME_OR_PATH=$META_SECALIGN_TARGET_MODEL_NAME_OR_PATH"
echo "VALIDATION_DATA_PATH=$VALIDATION_DATA_PATH"
echo "VAL_BATCH_SIZE=$VAL_BATCH_SIZE"
echo "VAL_MAX_NEW_TOKENS=$VAL_MAX_NEW_TOKENS"
echo "ENABLE_WANDB=$ENABLE_WANDB"
echo "RUN_NAME_PREFIX=$RUN_NAME_PREFIX"
echo "SAVE_NAME_PREFIX=$SAVE_NAME_PREFIX"
echo "NUM_CHECKPOINTS=${#CHECKPOINT_DIRS[@]}"

nvidia-smi || true

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"

for checkpoint_dir in "${CHECKPOINT_DIRS[@]}"; do
  ATTACKER_CHECKPOINT_NAME="$(basename "$checkpoint_dir")"
  ATTACKER_CHECKPOINT_PATH="$(realpath "$checkpoint_dir")"
  ATTACKER_LORA_PATH="$ATTACKER_CHECKPOINT_DIR/${ATTACKER_CHECKPOINT_NAME}-lora"
  RUN_NAME="${RUN_NAME_PREFIX}_attack_meta_secalign_8b_${ATTACKER_CHECKPOINT_NAME}"
  SAVE_NAME="${SAVE_NAME_PREFIX}_${ATTACKER_CHECKPOINT_NAME}"

  if [ ! -f "$ATTACKER_CHECKPOINT_PATH/adapter_model.safetensors" ]; then
    echo "Skipping non-LoRA checkpoint: $ATTACKER_CHECKPOINT_PATH" >&2
    continue
  fi

  ln -sfn "$ATTACKER_CHECKPOINT_PATH" "$ATTACKER_LORA_PATH"

  echo "=== Evaluating $ATTACKER_CHECKPOINT_NAME ==="
  echo "ATTACKER_CHECKPOINT_PATH=$ATTACKER_CHECKPOINT_PATH"
  echo "ATTACKER_LORA_PATH=$ATTACKER_LORA_PATH"
  echo "RUN_NAME=$RUN_NAME"
  echo "SAVE_NAME=$SAVE_NAME"

  python injecagent_eval.py \
    --attacker_model_name_or_path "$ATTACKER_LORA_PATH" \
    --attacker_base_model_name_or_path "$ATTACKER_BASE_MODEL_NAME_OR_PATH" \
    --target_model_name_or_path "$META_SECALIGN_TARGET_MODEL_NAME_OR_PATH" \
    --validation_data_path "$VALIDATION_DATA_PATH" \
    --val_batch_size "$VAL_BATCH_SIZE" \
    --val_max_new_tokens "$VAL_MAX_NEW_TOKENS" \
    --enable_wandb "$ENABLE_WANDB" \
    --save_name "$SAVE_NAME" \
    --run_name "$RUN_NAME"
done
