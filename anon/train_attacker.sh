#!/bin/bash
# Train one RL-Hammer attacker (GRPO) against a target model, as used for Figure 3 / Tables 7-8.
#
# Usage (from injecAgent-rl-harmmer/rl-injector, inside the `rlhammer` environment, see README):
#   TARGET=sr_llama  SYS_APPEND=0 SR_LORA=/path/to/sr-agent-llama-lora RUN_NAME=srllama_run1 SEED=1024 bash jobs/train_attacker.sh
#   TARGET=llama_base SYS_APPEND=0                                       RUN_NAME=llamabase_run1 SEED=1024 bash jobs/train_attacker.sh
#   TARGET=sr_qwen   SYS_APPEND=1 SR_LORA=/path/to/sr-agent-qwen3-8b-lora RUN_NAME=srqwen_run1  SEED=1024 bash jobs/train_attacker.sh
#
# Environment variables:
#   TARGET       sr_llama | llama_base | sr_qwen   main target (reward weight 3). The weak partner target is always
#                Llama-3.1-8B-Instruct with the ReAct prompt (weight 1), as in the original RL-Hammer recipe.
#   SYS_APPEND   0 | 1   add the SR-Agent reflection instruction to the target's system prompt (paper App. B.3);
#                identical in attacker training and evaluation. Paper: 0 for the Llama targets, 1 for SR-Agent-Qwen3-8B.
#   SR_LORA      LoRA adapter directory of the SR-Agent target (TARGET=sr_llama / sr_qwen).
#   RUN_NAME     checkpoints are written to checkpoints/<RUN_NAME>/checkpoint-<step> (must not contain "attack").
#   SEED         1024 for run 1, 2048 for run 2 (the two independent attacker runs per target).
#   TARGET_THINK 0 | 1 (default 1)  target think mode, used in training and evaluation (ablation "think off" = 0).
#   NPROC        attacker processes (default: #GPUs - 1); GA = 24 / NPROC keeps the effective batch at
#                2 x 8 x 3 = 48 completions per step regardless of the GPU count.
#   EXTRA_ARGS   appended to train.py (later flags win).
#
# GPU layout: GPUs 0..NPROC-1 run the attacker (accelerate / DDP); the last GPU runs one vLLM OpenAI server that serves
# both targets (base weights + the SR-Agent LoRA under a second served name). 4 x 80 GB GPUs: ~6-9 h per run.
# Recipe (identical for every run): attacker Llama-3.1-8B-Instruct + LoRA r64/alpha32 on q,k,v,o,up,down,gate; GRPO,
# beta 0 (no KL), lr 1e-5 constant with 3 % warm-up, num_generations 8, 20 epochs over data/InjecAgent/dataset/train.json
# (= 1020 steps), one model-only checkpoint per epoch; target sampling T 0.6 / top-p 0.95 / top-k 20, 512 target tokens
# during training (1024 in evaluation).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn WANDB_MODE=disabled

TARGET=${TARGET:?sr_llama|llama_base|sr_qwen}; SYS_APPEND=${SYS_APPEND:?0|1}; RUN_NAME=${RUN_NAME:?}
case "$RUN_NAME" in *attack*) echo "RUN_NAME must not contain 'attack'" >&2; exit 2;; esac
ABASE=meta-llama/Llama-3.1-8B-Instruct            # attacker base (always Llama, as in RL-Hammer)
NGPU=$(nvidia-smi -L | wc -l); NPROC=${NPROC:-$((NGPU - 1))}
[ "$NPROC" -ge 1 ] && [ $((NPROC + 1)) -le "$NGPU" ] && [ $((24 % NPROC)) -eq 0 ] || { echo "bad NPROC=$NPROC for $NGPU GPUs" >&2; exit 2; }
GA=$((24 / NPROC))
APPEND=False; [ "$SYS_APPEND" = 1 ] && APPEND=True
TTHINK=True; [ "${TARGET_THINK:-1}" = 0 ] && TTHINK=False
PORT=${PORT:-$((20000 + $$ % 20000))}; MPORT=$((PORT + 1))
URL=http://localhost:$PORT/v1

TBASE=$ABASE; FMT=llama_local
case "$TARGET" in
  sr_llama)   SR_LORA=${SR_LORA:?}; test -f "$SR_LORA/adapter_model.safetensors"
              MAIN=local/SR-Agent-Llama
              SERVE=(--served-model-name "$TBASE" --enable-lora --max-lora-rank 64 --max-loras 1 --lora-modules "$MAIN=$SR_LORA") ;;
  llama_base) MAIN=local/Llama-3.1-8B-Instruct-target   # second name for the same weights: the reward tells the targets apart by name
              SERVE=(--served-model-name "$TBASE" "$MAIN") ;;
  sr_qwen)    SR_LORA=${SR_LORA:?}; test -f "$SR_LORA/adapter_model.safetensors"
              TBASE=Qwen/Qwen3-8B; FMT=qwen; MAIN=local/SR-Agent-Qwen
              SERVE=(--served-model-name "$TBASE" --enable-lora --max-lora-rank 64 --max-loras 1 --lora-modules "$MAIN=$SR_LORA") ;;
  *) echo "unknown TARGET=$TARGET" >&2; exit 2 ;;
esac
echo "TARGET=$TARGET MAIN=$MAIN SYS_APPEND=$APPEND SR_LORA=${SR_LORA:-} TBASE=$TBASE FMT=$FMT THINK=$TTHINK RUN_NAME=$RUN_NAME NGPU=$NGPU NPROC=$NPROC GA=$GA SEED=${SEED:-1024} PORT=$PORT"

mkdir -p logs
VLOG="logs/vllm_${RUN_NAME}.log"
CUDA_VISIBLE_DEVICES=$((NGPU - 1)) python -m vllm.entrypoints.openai.api_server --model "$TBASE" "${SERVE[@]}" \
  --port "$PORT" --max-model-len 8192 --gpu-memory-utilization 0.9 > "$VLOG" 2>&1 &
VPID=$!
trap 'kill $VPID 2>/dev/null || true; wait $VPID 2>/dev/null || true' EXIT
for i in $(seq 1 240); do
  curl -sf "$URL/models" >/dev/null && break
  kill -0 $VPID 2>/dev/null || { echo "vLLM died, see $VLOG" >&2; tail -30 "$VLOG" >&2; exit 1; }
  sleep 5
done
curl -sf "$URL/models" | python -c "import json,sys; print('served:', [m['id'] for m in json.load(sys.stdin)['data']])"

mkdir -p checkpoints
CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NPROC - 1))) accelerate launch --num_processes "$NPROC" --main_process_port "$MPORT" \
  train.py \
  --attacker_model_name_or_path "$ABASE" \
  --target_model_name_or_path "$MAIN;$TBASE" \
  --target_model_url "$URL;$URL" \
  --safe_agent_target_model_name_or_path "$MAIN" \
  --safe_agent_target_base_model_name_or_path "$TBASE" \
  --safe_agent_mode True \
  --safe_agent_prompt_format "$FMT" \
  --target_enable_thinking "$TTHINK" \
  --use_safe_agent_system_append "$APPEND" \
  --model_wise_reward_weights 3.0 1.0 \
  --reward_functions InjecAgentToolCallingReward \
  --dataset data/InjecAgent/dataset/train.json \
  --attn_implementation sdpa \
  --num_generations 8 \
  --num_iterations 1 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps "$GA" \
  --num_train_epochs 20 \
  --bf16 True \
  --beta 0.0 \
  --warmup_ratio 0.03 \
  --gradient_checkpointing True \
  --learning_rate 1e-5 \
  --lr_scheduler_type constant_with_warmup \
  --use_peft True \
  --lora_r 64 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --logging_steps 1 \
  --save_strategy epoch \
  --save_total_limit 20 \
  --save_only_model True \
  --seed "${SEED:-1024}" \
  --output_dir "checkpoints/$RUN_NAME" \
  --report_to none \
  --run_name "$RUN_NAME" \
  ${EXTRA_ARGS:-}
echo "done: checkpoints/$RUN_NAME"
