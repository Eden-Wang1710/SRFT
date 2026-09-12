#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -c 5
#SBATCH --mem=30G
#SBATCH -t 08:00:00
#SBATCH -J rlh_eval
# Evaluate every attacker checkpoint of one RL-Hammer run on InjecAgent test.json (100 cases) against the SAME target config it was
# trained on (docs/12). Cluster-neutral; normally chained after the training job by jobs/submit_rlh.sh. Restart-safe: checkpoints whose
# result JSON exists are skipped. As NeurIPS: val_max_new_tokens 1024, target T 0.6 / top-p 0.95 / top-k 20 (config defaults).
# --export: TARGET=sr_llama|llama_base, SYS_APPEND=0|1, RUN_NAME, SR_LORA (sr_llama), [VAL_BATCH_SIZE=16], [CKPT_DIR=checkpoints/$RUN_NAME]
# Output: outputs/<RUN_NAME>_/<RUN_NAME>_attack_<ckpt>/<ckpt>-lora.json + attack_success_rate.json (tracked in git for outputs/iclr_*)
set -euo pipefail
SRFT_ROOT="${SRFT_ROOT:-$(d="${SLURM_SUBMIT_DIR:-$PWD}"; while [ ! -f "$d/env/select.sh" ] && [ "$d" != / ]; do d=$(dirname "$d"); done; echo "$d")}"
source "$SRFT_ROOT/env/select.sh"
source "$CONDA_SH" && conda activate rlhammer
cd "$SRFT_ROOT/injecAgent-rl-harmmer/rl-injector"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn WANDB_MODE=disabled
export PRINT_TARGET_INPUTS=1 PRINT_TARGET_INPUTS_MAX_N=2

TARGET=${TARGET:?}; SYS_APPEND=${SYS_APPEND:?}; RUN_NAME=${RUN_NAME:?}
CKPT_DIR=${CKPT_DIR:-checkpoints/$RUN_NAME}
BASE=meta-llama/Llama-3.1-8B-Instruct
APPEND=False; [ "$SYS_APPEND" = 1 ] && APPEND=True
case "$TARGET" in
  sr_llama)   TGT=${SR_LORA:?}; test -f "$TGT/adapter_model.safetensors" ;;
  llama_base) TGT=$BASE ;;
  *) echo "unknown TARGET=$TARGET" >&2; exit 2 ;;
esac
echo "cluster=$SRFT_CLUSTER node=$(hostname) job=${SLURM_JOB_ID:-} start=$(date) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "TARGET=$TARGET TGT=$TGT SYS_APPEND=$APPEND RUN_NAME=$RUN_NAME CKPT_DIR=$CKPT_DIR"

mapfile -t CKPTS < <(find "$CKPT_DIR" -maxdepth 1 -mindepth 1 -type d -name 'checkpoint-*' | sort -V)
[ "${#CKPTS[@]}" -gt 0 ] || { echo "no checkpoint-* in $CKPT_DIR" >&2; exit 1; }
for c in "${CKPTS[@]}"; do
  name=$(basename "$c")
  [ -f "$c/adapter_model.safetensors" ] || { echo "skip non-LoRA $c"; continue; }
  run="${RUN_NAME}_attack_${name}"
  out="outputs/${RUN_NAME}_/${run}/${name}-lora.json"
  [ -f "$out" ] && { echo "done already: $out"; continue; }
  ln -sfn "$name" "$CKPT_DIR/${name}-lora"          # "lora" in the path makes injecagent_eval load it as an adapter
  echo "=== $name $(date)"
  python injecagent_eval.py \
    --attacker_model_name_or_path "$CKPT_DIR/${name}-lora" \
    --attacker_base_model_name_or_path "$BASE" \
    --target_model_name_or_path "$TGT" \
    --target_base_model_name_or_path "$BASE" \
    --safe_agent_mode True \
    --safe_agent_prompt_format llama_local \
    --target_enable_thinking True \
    --use_safe_agent_system_append "$APPEND" \
    --validation_data_path data/InjecAgent/dataset/test.json \
    --val_batch_size "${VAL_BATCH_SIZE:-16}" \
    --val_max_new_tokens 1024 \
    --enable_wandb False \
    --save_name "${RUN_NAME}_${name}" \
    --run_name "$run"
done
python - "$RUN_NAME" <<'EOF'
import glob, json, re, sys
run = sys.argv[1]
rows = {}
for f in glob.glob(f"outputs/{run}_/*/attack_success_rate.json"):
    rows.update(json.load(open(f)))
print("ASR per checkpoint:", " ".join(f"{k.replace('checkpoint-','').replace('-lora','')}:{v:.2f}" for k, v in sorted(rows.items(), key=lambda kv: int(re.findall(r"\d+", kv[0])[0]))))
EOF
echo "end=$(date)"
