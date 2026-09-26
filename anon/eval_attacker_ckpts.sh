#!/bin/bash
# Evaluate every checkpoint of one attacker run on the 100 InjecAgent test cases against the SAME target configuration
# the attacker was trained on. This produces the per-epoch points of Figure 3.
#
# Usage (from injecAgent-rl-harmmer/rl-injector, 1 GPU, ~1.5 h for 20 checkpoints):
#   TARGET=sr_llama SYS_APPEND=0 SR_LORA=/path/to/sr-agent-llama-lora RUN_NAME=srllama_run1 bash jobs/eval_attacker_ckpts.sh
# Variables: TARGET / SYS_APPEND / SR_LORA / RUN_NAME / TARGET_THINK as in train_attacker.sh (they must match the training run);
#   CKPT_DIR (default checkpoints/$RUN_NAME), VAL_BATCH_SIZE (default 16).
# Output: outputs/<RUN_NAME>_/<RUN_NAME>_attack_checkpoint-<step>/{checkpoint-<step>-lora.json, attack_success_rate.json}
#   checkpoint-<step>-lora.json  = the 100 test cases (adversarial prompt, target output, judge verdict)
#   attack_success_rate.json     = ASR of that checkpoint
# Restart-safe: checkpoints whose result file exists are skipped.
# Protocol (paper App. C.2): 1024 target tokens, target sampling T 0.6 / top-p 0.95 / top-k 20; judge = the first JSON /
# <function=...> tool call in the target output (attacker's tool = success, user's tool = failure, else invalid = failure).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn WANDB_MODE=disabled

TARGET=${TARGET:?}; SYS_APPEND=${SYS_APPEND:?}; RUN_NAME=${RUN_NAME:?}
CKPT_DIR=${CKPT_DIR:-checkpoints/$RUN_NAME}
BASE=meta-llama/Llama-3.1-8B-Instruct
APPEND=False; [ "$SYS_APPEND" = 1 ] && APPEND=True
TTHINK=True; [ "${TARGET_THINK:-1}" = 0 ] && TTHINK=False
TBASE=$BASE; FMT=llama_local
case "$TARGET" in
  sr_llama)   TGT=${SR_LORA:?}; test -f "$TGT/adapter_model.safetensors" ;;
  llama_base) TGT=$BASE ;;
  sr_qwen)    TGT=${SR_LORA:?}; test -f "$TGT/adapter_model.safetensors"; TBASE=Qwen/Qwen3-8B; FMT=qwen ;;
  *) echo "unknown TARGET=$TARGET" >&2; exit 2 ;;
esac
echo "TARGET=$TARGET TGT=$TGT TBASE=$TBASE FMT=$FMT THINK=$TTHINK SYS_APPEND=$APPEND RUN_NAME=$RUN_NAME CKPT_DIR=$CKPT_DIR"

mapfile -t CKPTS < <(find "$CKPT_DIR" -maxdepth 1 -mindepth 1 -type d -name 'checkpoint-*' | sort -V)
[ "${#CKPTS[@]}" -gt 0 ] || { echo "no checkpoint-* in $CKPT_DIR" >&2; exit 1; }
LINKDIR="${CKPT_DIR%/}_lora_links"; mkdir -p "$LINKDIR"   # "lora" in the path makes injecagent_eval.py load the attacker as an adapter
for c in "${CKPTS[@]}"; do
  name=$(basename "$c")
  [ -f "$c/adapter_model.safetensors" ] || { echo "skip non-LoRA $c"; continue; }
  run="${RUN_NAME}_attack_${name}"
  out="outputs/${RUN_NAME}_/${run}/${name}-lora.json"
  [ -f "$out" ] && { echo "done already: $out"; continue; }
  ln -sfn "$(realpath "$c")" "$LINKDIR/${name}-lora"
  echo "=== $name $(date)"
  python injecagent_eval.py \
    --attacker_model_name_or_path "$LINKDIR/${name}-lora" \
    --attacker_base_model_name_or_path "$BASE" \
    --target_model_name_or_path "$TGT" \
    --target_base_model_name_or_path "$TBASE" \
    --safe_agent_mode True \
    --safe_agent_prompt_format "$FMT" \
    --target_enable_thinking "$TTHINK" \
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
print("ASR per checkpoint:", " ".join(f"{re.findall(r'checkpoint-(\d+)', k)[0]}:{v:.2f}" for k, v in sorted(rows.items(), key=lambda kv: int(re.findall(r"checkpoint-(\d+)", kv[0])[0]))))
EOF
