#!/usr/bin/env bash
#SBATCH --job-name=rlh_injec_llama_meta_secalign_5gpu
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:5
#SBATCH --partition=a100
#SBATCH --account=aszalay1_ssci
#SBATCH --time=24:00:00
#SBATCH --chdir=injecAgent-rl-harmmer/rl-injector

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

ATTACKER_MODEL_NAME_OR_PATH="${ATTACKER_MODEL_NAME_OR_PATH:-meta-llama/Llama-3.1-8B-Instruct}"
META_SECALIGN_TARGET_MODEL_NAME_OR_PATH="${META_SECALIGN_TARGET_MODEL_NAME_OR_PATH:-checkpoints/Meta-SecAlign-8B-merged}"
LLAMA_TARGET_MODEL_NAME_OR_PATH="${LLAMA_TARGET_MODEL_NAME_OR_PATH:-meta-llama/Llama-3.1-8B-Instruct}"
TARGET_MODEL_NAME_OR_PATH="${TARGET_MODEL_NAME_OR_PATH:-$META_SECALIGN_TARGET_MODEL_NAME_OR_PATH;$LLAMA_TARGET_MODEL_NAME_OR_PATH}"
META_SECALIGN_TARGET_PORT="${META_SECALIGN_TARGET_PORT:-8010}"
LLAMA_TARGET_PORT="${LLAMA_TARGET_PORT:-8011}"
TARGET_MODEL_URL="${TARGET_MODEL_URL:-http://localhost:${META_SECALIGN_TARGET_PORT}/v1;http://localhost:${LLAMA_TARGET_PORT}/v1}"
MODEL_WISE_REWARD_WEIGHTS="${MODEL_WISE_REWARD_WEIGHTS:-3.0 1.0}"
RUN_NAME="${RUN_NAME:-rl_hammer_target_llama_meta_secalign_5gpu_original_para}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/${RUN_NAME}}"
DATASET_PATH="${DATASET_PATH:-data/InjecAgent/dataset/train.json}"

# Keep training settings aligned with run_train_injecagent_llama_safeagent_5gpu_original_para.slurm.
LR="${LR:-1e-5}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
NUM_ITERATIONS="${NUM_ITERATIONS:-1}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-20}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-$NUM_TRAIN_EPOCHS}"
LORA_R="${LORA_R:-64}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-3}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
REPORT_TO="${REPORT_TO:-none}"
DO_EVAL="${DO_EVAL:-0}"

if [ ! -d "$META_SECALIGN_TARGET_MODEL_NAME_OR_PATH" ]; then
  echo "Meta-SecAlign target directory not found: $META_SECALIGN_TARGET_MODEL_NAME_OR_PATH" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "=== Runtime Diagnostics ==="
echo "CONDA_PREFIX=$CONDA_PREFIX"
echo "which python=$(which python)"
echo "python --version=$(python --version 2>&1)"
echo "RUN_NAME=$RUN_NAME"
echo "ATTACKER_MODEL_NAME_OR_PATH=$ATTACKER_MODEL_NAME_OR_PATH"
echo "META_SECALIGN_TARGET_MODEL_NAME_OR_PATH=$META_SECALIGN_TARGET_MODEL_NAME_OR_PATH"
echo "LLAMA_TARGET_MODEL_NAME_OR_PATH=$LLAMA_TARGET_MODEL_NAME_OR_PATH"
echo "TARGET_MODEL_NAME_OR_PATH=$TARGET_MODEL_NAME_OR_PATH"
echo "TARGET_MODEL_URL=$TARGET_MODEL_URL"
echo "META_SECALIGN_TARGET_PORT=$META_SECALIGN_TARGET_PORT"
echo "LLAMA_TARGET_PORT=$LLAMA_TARGET_PORT"
echo "MODEL_WISE_REWARD_WEIGHTS=$MODEL_WISE_REWARD_WEIGHTS"
echo "DATASET_PATH=$DATASET_PATH"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "HF_HOME=$HF_HOME"
echo "HUGGINGFACE_HUB_CACHE=$HUGGINGFACE_HUB_CACHE"
echo "HF_HUB_CACHE=$HF_HUB_CACHE"
echo "TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE"
echo "VLLM_CONFIG_ROOT=$VLLM_CONFIG_ROOT"
echo "NUM_GENERATIONS=$NUM_GENERATIONS"
echo "PER_DEVICE_TRAIN_BATCH_SIZE=$PER_DEVICE_TRAIN_BATCH_SIZE"
echo "GRADIENT_ACCUMULATION_STEPS=$GRADIENT_ACCUMULATION_STEPS"
echo "NUM_TRAIN_EPOCHS=$NUM_TRAIN_EPOCHS"
echo "SAVE_TOTAL_LIMIT=$SAVE_TOTAL_LIMIT"
echo "NPROC_PER_NODE=$NPROC_PER_NODE"
echo "DO_EVAL=$DO_EVAL"

nvidia-smi || true

cleanup() {
  if [ -n "${META_SECALIGN_VLLM_PID:-}" ] && kill -0 "${META_SECALIGN_VLLM_PID}" 2>/dev/null; then
    kill "${META_SECALIGN_VLLM_PID}" || true
    wait "${META_SECALIGN_VLLM_PID}" || true
  fi
  if [ -n "${LLAMA_VLLM_PID:-}" ] && kill -0 "${LLAMA_VLLM_PID}" 2>/dev/null; then
    kill "${LLAMA_VLLM_PID}" || true
    wait "${LLAMA_VLLM_PID}" || true
  fi
}
trap cleanup EXIT

# Reserve GPU 3 for the Meta-SecAlign target service.
export CUDA_VISIBLE_DEVICES=3
python -m vllm.entrypoints.openai.api_server \
  --model "$META_SECALIGN_TARGET_MODEL_NAME_OR_PATH" \
  --port "$META_SECALIGN_TARGET_PORT" \
  > "logs/vllm_meta_secalign_${SLURM_JOB_ID}.log" 2>&1 &
META_SECALIGN_VLLM_PID=$!

until curl -sf "http://localhost:${META_SECALIGN_TARGET_PORT}/v1/models" >/dev/null; do
  sleep 5
done

# Reserve GPU 4 for the Llama target service.
export CUDA_VISIBLE_DEVICES=4
python -m vllm.entrypoints.openai.api_server \
  --model "$LLAMA_TARGET_MODEL_NAME_OR_PATH" \
  --port "$LLAMA_TARGET_PORT" \
  > "logs/vllm_llama_${SLURM_JOB_ID}.log" 2>&1 &
LLAMA_VLLM_PID=$!

until curl -sf "http://localhost:${LLAMA_TARGET_PORT}/v1/models" >/dev/null; do
  sleep 5
done

# Train attacker on GPUs 0,1,2.
export CUDA_VISIBLE_DEVICES=0,1,2
# shellcheck disable=SC2206
WEIGHT_ARGS=($MODEL_WISE_REWARD_WEIGHTS)
accelerate launch \
  --num_processes "$NPROC_PER_NODE" \
  train.py \
  --attacker_model_name_or_path "$ATTACKER_MODEL_NAME_OR_PATH" \
  --target_model_name_or_path "$TARGET_MODEL_NAME_OR_PATH" \
  --target_model_url "$TARGET_MODEL_URL" \
  --target_enable_thinking False \
  --model_wise_reward_weights "${WEIGHT_ARGS[@]}" \
  --reward_functions InjecAgentToolCallingReward \
  --dataset "$DATASET_PATH" \
  --attn_implementation "$ATTN_IMPLEMENTATION" \
  --num_generations "$NUM_GENERATIONS" \
  --num_iterations "$NUM_ITERATIONS" \
  --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --num_train_epochs "$NUM_TRAIN_EPOCHS" \
  --bf16 True \
  --beta 0.0 \
  --warmup_ratio 0.03 \
  --gradient_checkpointing True \
  --learning_rate "$LR" \
  --lr_scheduler_type constant_with_warmup \
  --use_peft True \
  --lora_r "$LORA_R" \
  --lora_alpha "$LORA_ALPHA" \
  --lora_dropout "$LORA_DROPOUT" \
  --logging_steps "$LOGGING_STEPS" \
  --save_strategy epoch \
  --save_total_limit "$SAVE_TOTAL_LIMIT" \
  --save_only_model True \
  --output_dir "$OUTPUT_DIR" \
  --report_to "$REPORT_TO" \
  --run_name "$RUN_NAME"

if [ "$DO_EVAL" = "1" ]; then
  export CUDA_VISIBLE_DEVICES=0
  for dir in "$OUTPUT_DIR"/*; do
    if [ -d "$dir" ]; then
      python injecagent_eval.py \
        --attacker_model_name_or_path "$dir" \
        --attacker_base_model_name_or_path "$ATTACKER_MODEL_NAME_OR_PATH" \
        --target_model_name_or_path "$META_SECALIGN_TARGET_MODEL_NAME_OR_PATH" \
        --validation_data_path data/InjecAgent/dataset/eval.json \
        --enable_wandb False \
        --run_name "eval_${RUN_NAME}_attack_meta_secalign_8b"
    fi
  done
fi
