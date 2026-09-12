#!/bin/bash
#SBATCH --gres=gpu:4
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 14:00:00
#SBATCH -J rlh_train
# RL-Hammer attacker (GRPO) against a Llama-3.1-8B target in SR-Agent-Llama's AgentDojo `local` format — ICLR RL-Hammer curves (docs/12).
# Cluster-neutral; normally submitted by jobs/submit_rlh.sh. Everything not listed below = the NeurIPS recipe
# (scripts/run_train_injecagent_llama_safeagent_5gpu_original_para_qwen_template_think_sysappend.slurm).
# --export:
#   TARGET=sr_llama|llama_base   main target (reward weight 3); the weak partner is always Llama-3.1-8B-Instruct with the ReAct prompt (weight 1)
#   SYS_APPEND=0|1               Fig. 5 append in the main target's system prompt, identical in training and eval
#   RUN_NAME                     checkpoints/<RUN_NAME>/checkpoint-<step> (must not contain "attack": the eval splits run names on it)
#   SR_LORA                      SR-Agent-Llama LoRA dir (TARGET=sr_llama)
#   NPROC (default #GPUs-1)      attacker processes; GA = 24/NPROC keeps the NeurIPS effective batch 2 x 8 x 3 = 48 completions/step
#   SEED (1024 = NeurIPS default), EXTRA_ARGS (appended to train.py, later flags win; smoke: "--max_steps 2 --save_strategy steps --save_steps 2")
# Layout: GPUs 0..NPROC-1 attacker (accelerate/DDP), last GPU one vLLM server that serves BOTH targets (base weights + optional LoRA).
set -euo pipefail
SRFT_ROOT="${SRFT_ROOT:-$(d="${SLURM_SUBMIT_DIR:-$PWD}"; while [ ! -f "$d/env/select.sh" ] && [ "$d" != / ]; do d=$(dirname "$d"); done; echo "$d")}"
source "$SRFT_ROOT/env/select.sh"
source "$CONDA_SH" && conda activate rlhammer
cd "$SRFT_ROOT/injecAgent-rl-harmmer/rl-injector"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn WANDB_MODE=disabled

TARGET=${TARGET:?sr_llama|llama_base}; SYS_APPEND=${SYS_APPEND:?0|1}; RUN_NAME=${RUN_NAME:?}
case "$RUN_NAME" in *attack*) echo "RUN_NAME must not contain 'attack'" >&2; exit 2;; esac
BASE=meta-llama/Llama-3.1-8B-Instruct
NGPU=$(nvidia-smi -L | wc -l); NPROC=${NPROC:-$((NGPU - 1))}
[ "$NPROC" -ge 1 ] && [ $((NPROC + 1)) -le "$NGPU" ] && [ $((24 % NPROC)) -eq 0 ] || { echo "bad NPROC=$NPROC for $NGPU GPUs" >&2; exit 2; }
GA=$((24 / NPROC))
APPEND=False; [ "$SYS_APPEND" = 1 ] && APPEND=True
PORT=$((20000 + ${SLURM_JOB_ID:-$$} % 20000)); MPORT=$((PORT + 1))   # per-job ports: two of our jobs may share a node
URL=http://localhost:$PORT/v1

case "$TARGET" in
  sr_llama)   SR_LORA=${SR_LORA:?}; test -f "$SR_LORA/adapter_model.safetensors"
              MAIN=local/SR-Agent-Llama
              SERVE=(--served-model-name "$BASE" --enable-lora --max-lora-rank 64 --max-loras 1 --lora-modules "$MAIN=$SR_LORA") ;;
  llama_base) MAIN=local/Llama-3.1-8B-Instruct-target   # second name for the same weights: the reward tells the targets apart by name
              SERVE=(--served-model-name "$BASE" "$MAIN") ;;
  *) echo "unknown TARGET=$TARGET" >&2; exit 2 ;;
esac
echo "cluster=$SRFT_CLUSTER node=$(hostname) job=${SLURM_JOB_ID:-} start=$(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "TARGET=$TARGET MAIN=$MAIN SYS_APPEND=$APPEND SR_LORA=${SR_LORA:-} RUN_NAME=$RUN_NAME NGPU=$NGPU NPROC=$NPROC GA=$GA SEED=${SEED:-1024} PORT=$PORT EXTRA_ARGS=${EXTRA_ARGS:-}"

VLOG="$SRFT_SLURM_LOGS/rlh_vllm_${RUN_NAME}_${SLURM_JOB_ID:-local}.log"
CUDA_VISIBLE_DEVICES=$((NGPU - 1)) python -m vllm.entrypoints.openai.api_server --model "$BASE" "${SERVE[@]}" \
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
  --attacker_model_name_or_path "$BASE" \
  --target_model_name_or_path "$MAIN;$BASE" \
  --target_model_url "$URL;$URL" \
  --safe_agent_target_model_name_or_path "$MAIN" \
  --safe_agent_target_base_model_name_or_path "$BASE" \
  --safe_agent_mode True \
  --safe_agent_prompt_format llama_local \
  --target_enable_thinking True \
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
echo "end=$(date)"
