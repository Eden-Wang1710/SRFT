#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -c 5
#SBATCH --mem=30G
#SBATCH -t 02:00:00
#SBATCH -J rlh_static
# Non-adaptive InjecAgent attacks on the Llama targets (docs/12), also the GPU smoke test of the llama_local path:
#   default_prompt  = InjecAgent's own injection (the attacker instruction verbatim), no attacker model needed
#   replay_yyy1     = the 100 test prompts the NeurIPS attacker trained against SR-Agent-Qwen (YYY_1, ckpt 1020) generated, replayed
# on SR-Agent-Llama with / without the append and on Llama base (no append). Writes outputs/iclr_static_<target>/...
# --export: [SR_LORA], [ONLY="<space-separated subset of the 6 case names below>"]
set -euo pipefail
SRFT_ROOT="${SRFT_ROOT:-$(d="${SLURM_SUBMIT_DIR:-$PWD}"; while [ ! -f "$d/env/select.sh" ] && [ "$d" != / ]; do d=$(dirname "$d"); done; echo "$d")}"
source "$SRFT_ROOT/env/select.sh"
source "$CONDA_SH" && conda activate rlhammer
cd "$SRFT_ROOT/injecAgent-rl-harmmer/rl-injector"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn WANDB_MODE=disabled PRINT_TARGET_INPUTS=1 PRINT_TARGET_INPUTS_MAX_N=2
BASE=meta-llama/Llama-3.1-8B-Instruct
SR_LORA=${SR_LORA:-$SRFT_ROOT/LLaMA-Factory/saves/llama31-8b/lora/v3base_local_sft_8k_r64_GA4_qkvo_3epoch_5e-6}
YYY1=checkpoints/rl_hammer_target_llama_safeagent_5gpu_YYY_1/checkpoint-1020-lora   # only a key into saved_adv_prompts/ (weights not needed)
echo "cluster=$SRFT_CLUSTER node=$(hostname) job=${SLURM_JOB_ID:-} start=$(date) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader)"
run() {  # case target append attacker save_name
  local case=$1 tgt=$2 app=$3 att=$4 save=$5
  case " ${ONLY:-$case} " in *" $case "*) ;; *) return 0;; esac
  echo "=== $case target=$tgt append=$app attacker=$att $(date)"
  python injecagent_eval.py --attacker_model_name_or_path "$att" --attacker_base_model_name_or_path "$BASE" \
    --target_model_name_or_path "$tgt" --target_base_model_name_or_path "$BASE" \
    --safe_agent_mode True --safe_agent_prompt_format llama_local --target_enable_thinking True --use_safe_agent_system_append "$app" \
    --validation_data_path data/InjecAgent/dataset/test.json --val_batch_size 16 --val_max_new_tokens 1024 \
    --enable_wandb False --save_name "$save" --run_name "iclr_static_${case}_attack"
}
run srllama_append_default     "$SR_LORA" True  default_prompt default
run srllama_noappend_default   "$SR_LORA" False default_prompt default
run llamabase_default          "$BASE"    False default_prompt default
run srllama_append_replayyyy1  "$SR_LORA" True  "$YYY1" YYY1_checkpoint-1020
run srllama_noappend_replayyyy1 "$SR_LORA" False "$YYY1" YYY1_checkpoint-1020
run llamabase_replayyyy1       "$BASE"    False "$YYY1" YYY1_checkpoint-1020
echo "end=$(date)"
