#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -c 5
#SBATCH --mem=30G
#SBATCH -t 00:30:00
#SBATCH -J rlh_static_qwen
# Non-adaptive InjecAgent attack (the benchmark's own injection, no attacker model) on the two Qwen3-8B targets of the NeurIPS
# RL-Hammer curves, for the appendix case study (docs/12): SR-Agent-Qwen3-8B (v0 LoRA, append ON as in YYY) and base Qwen3-8B
# (append OFF as in YYN). Same eval code/format as jobs/eval_attacker_ckpts.sbatch (FMT=qwen). Writes outputs/iclr_static_qwen_<case>_/...
set -euo pipefail
SRFT_ROOT="${SRFT_ROOT:-$(d="${SLURM_SUBMIT_DIR:-$PWD}"; while [ ! -f "$d/env/select.sh" ] && [ "$d" != / ]; do d=$(dirname "$d"); done; echo "$d")}"
source "$SRFT_ROOT/env/select.sh"
source "$CONDA_SH" && conda activate rlhammer
cd "$SRFT_ROOT/injecAgent-rl-harmmer/rl-injector"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn WANDB_MODE=disabled PRINT_TARGET_INPUTS=1 PRINT_TARGET_INPUTS_MAX_N=1
ABASE=meta-llama/Llama-3.1-8B-Instruct
TBASE=Qwen/Qwen3-8B
SR_LORA=${SR_LORA:-$SRFT_ROOT/LLaMA-Factory/saves/qwen3-8b/lora/toucan_32B_v2_sft_8k_r64_GA4_qkvo_3epoch_5e-6}
test -f "$SR_LORA/adapter_model.safetensors"
echo "cluster=$SRFT_CLUSTER node=$(hostname) job=${SLURM_JOB_ID:-} start=$(date) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader)"
run() {  # case target append
  local case=$1 tgt=$2 app=$3
  echo "=== $case target=$tgt append=$app $(date)"
  python injecagent_eval.py --attacker_model_name_or_path default_prompt --attacker_base_model_name_or_path "$ABASE" \
    --target_model_name_or_path "$tgt" --target_base_model_name_or_path "$TBASE" \
    --safe_agent_mode True --safe_agent_prompt_format qwen --target_enable_thinking True --use_safe_agent_system_append "$app" \
    --validation_data_path data/InjecAgent/dataset/test.json --val_batch_size "${VAL_BATCH_SIZE:-32}" --val_max_new_tokens 1024 \
    --enable_wandb False --save_name default --run_name "iclr_static_qwen_${case}_attack"
}
run srqwen_append "$SR_LORA" True
run qwenbase_noappend "$TBASE" False
echo "end=$(date)"
