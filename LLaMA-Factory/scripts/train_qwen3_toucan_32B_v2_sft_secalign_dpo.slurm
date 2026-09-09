#!/usr/bin/env bash
#SBATCH --job-name=qwen3-v2sft-secalign-dpo
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --partition=a100
#SBATCH --account=aszalay1_ssci
#SBATCH --time=12:00:00
#SBATCH --chdir=/home/cxiao13/scratch-cxiao13/zixuan/safe-agent-project/LLaMA-Factory

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

CONDA_ENV="${CONDA_ENV:-llamafactory}"
conda activate "$CONDA_ENV"
unset VIRTUAL_ENV
unset PYTHONHOME
hash -r
export PATH="$CONDA_PREFIX/bin:$PATH"

export HF_HOME="${HF_HOME:-/scratch/cxiao13/huggingface_cxiao13_access}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_CACHE="$HUGGINGFACE_HUB_CACHE"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"

export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export WANDB_PROJECT="${WANDB_PROJECT:-Meta-SecAlign-Qwen3-DPO}"
export FORCE_TORCHRUN="${FORCE_TORCHRUN:-1}"
export NNODES="${NNODES:-1}"
export NODE_RANK="${NODE_RANK:-0}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"

BASE_CONFIG="${BASE_CONFIG:-examples/train_lora/qwen3_8b_lora_dpo_secalign_from_toucan_32B_v2_sft.yaml}"
RUN_NAME="${RUN_NAME:-qwen3_8b_toucan_32B_v2_sft_secalign_dpo_lora}"
OUTPUT_DIR="${OUTPUT_DIR:-saves/qwen3-8b/lora/${RUN_NAME}}"
MAX_STEPS="${MAX_STEPS:--1}"
LEARNING_RATE="${LEARNING_RATE:-}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-}"
CUTOFF_LEN="${CUTOFF_LEN:-}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-null}"

RUNTIME_CONFIG="logs/qwen3_toucan_32B_v2_sft_secalign_dpo_${SLURM_JOB_ID}.yaml"

export RUN_NAME
export OUTPUT_DIR
export MAX_STEPS
export LEARNING_RATE
export NUM_TRAIN_EPOCHS
export GRADIENT_ACCUMULATION_STEPS
export PER_DEVICE_TRAIN_BATCH_SIZE
export CUTOFF_LEN
export RESUME_FROM_CHECKPOINT

python - "$BASE_CONFIG" "$RUNTIME_CONFIG" <<'PY'
import os
import sys
import yaml

base_config, runtime_config = sys.argv[1], sys.argv[2]
with open(base_config, "r") as f:
    cfg = yaml.safe_load(f)

cfg["output_dir"] = os.environ["OUTPUT_DIR"]
cfg["run_name"] = os.environ["RUN_NAME"]
resume = os.environ["RESUME_FROM_CHECKPOINT"]
cfg["resume_from_checkpoint"] = None if resume in {"", "null", "None"} else resume

if os.environ["MAX_STEPS"] != "":
    cfg["max_steps"] = int(os.environ["MAX_STEPS"])
if os.environ["LEARNING_RATE"]:
    cfg["learning_rate"] = float(os.environ["LEARNING_RATE"])
if os.environ["NUM_TRAIN_EPOCHS"]:
    cfg["num_train_epochs"] = float(os.environ["NUM_TRAIN_EPOCHS"])
if os.environ["GRADIENT_ACCUMULATION_STEPS"]:
    cfg["gradient_accumulation_steps"] = int(os.environ["GRADIENT_ACCUMULATION_STEPS"])
if os.environ["PER_DEVICE_TRAIN_BATCH_SIZE"]:
    cfg["per_device_train_batch_size"] = int(os.environ["PER_DEVICE_TRAIN_BATCH_SIZE"])
if os.environ["CUTOFF_LEN"]:
    cfg["cutoff_len"] = int(os.environ["CUTOFF_LEN"])

with open(runtime_config, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY

echo "=== Runtime Diagnostics ==="
echo "HOST=$(hostname)"
echo "PWD=$PWD"
echo "CONDA_ENV=$CONDA_ENV"
echo "CONDA_PREFIX=$CONDA_PREFIX"
echo "which python=$(which python)"
echo "python --version=$(python --version 2>&1)"
echo "BASE_CONFIG=$BASE_CONFIG"
echo "RUNTIME_CONFIG=$RUNTIME_CONFIG"
echo "RUN_NAME=$RUN_NAME"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "MAX_STEPS=$MAX_STEPS"
echo "NPROC_PER_NODE=$NPROC_PER_NODE"
echo "HF_HOME=$HF_HOME"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true
echo "=== Runtime YAML ==="
cat "$RUNTIME_CONFIG"

python -m llamafactory.cli train "$RUNTIME_CONFIG"
