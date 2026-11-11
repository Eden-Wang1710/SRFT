#!/bin/bash
# aba-cleanSFT end to end on ONE GPU inside an interactive allocation (tmux + srun, 2026-11-10): train -> AgentDojo eval -> stats.
# It only chains the existing job scripts (the same ones `env/sb` submits), so the protocol is identical to the sbatch route:
#   LLaMA-Factory/scripts/train_qwen3_8b_sdL2_sft.slurm   (config llama31_8b_lora_sft_aba_cleansft.yaml, GA 16)
#   agentdojo/scripts/eval_sdL2.sbatch                     (MODEL=LLAMA_3_1_8B_SAFE_AGENT, SYS_APPEND=0, the 10 shards of submit_eval_sdL2.sh)
#   agentdojo/eval/compute_attack_stats.py
# usage (inside the allocation, from anywhere in the repo):  bash ablation/run_aba_cleansft_interactive.sh
# Knobs: RUN_NAME, NPROC (procs per shard, default 2: RAM is capped at 64 GB on general-interactive), SKIP_TRAIN=1 to resume at the eval.
# Every eval shard is resumable (benchmark.py skips finished tasks); a failed shard is retried once with 1 proc at the end.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/env/select.sh"
RUN_NAME=${RUN_NAME:-aba_cleanSFT}; NPROC=${NPROC:-2}
CFG=examples/train_lora/llama31_8b_lora_sft_aba_cleansft.yaml
OUT=saves/llama31-8b/lora/aba_cleansft_sft_8k_r64_GA4_qkvo_3epoch_5e-6
LORA="$ROOT/LLaMA-Factory/$OUT"
MODEL=LLAMA_3_1_8B_SAFE_AGENT; MODEL_DIR=meta-llama_Llama-3.1-8B-Instruct-safe-agent
STATE="$ROOT/slurm_logs/${RUN_NAME}_interactive.state"; : > "$STATE"
say(){ echo "[$(date '+%F %T')] $*" | tee -a "$STATE"; }
cd "$ROOT"
say "START job=${SLURM_JOB_ID:-none} node=$(hostname) gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

if [ "${SKIP_TRAIN:-0}" != 1 ]; then
  say "TRAIN begin"
  SFT_CONFIG=$CFG NPROC_PER_NODE=1 EXTRA_ARGS="gradient_accumulation_steps=16" bash LLaMA-Factory/scripts/train_qwen3_8b_sdL2_sft.slurm
  rc=$?; say "TRAIN end rc=$rc"
  [ $rc -eq 0 ] || { say "FAILED train"; exit 1; }
fi
[ -f "$LORA/adapter_model.safetensors" ] || { say "FAILED no adapter at $LORA"; exit 1; }
say "ADAPTER ok sha256=$(sha256sum "$LORA/adapter_model.safetensors" | cut -c1-16)"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
shard(){ # name nproc VAR=VALUE...
  local name=$1 np=$2; shift 2
  say "EVAL $name begin nproc=$np"
  env MODEL=$MODEL RUN_LOGDIR=runs/$RUN_NAME LORA_PATH="$LORA" NPROC=$np SYS_APPEND=0 SUITES=workspace+travel+slack+banking "$@" \
    bash agentdojo/scripts/eval_sdL2.sbatch
  local rc=$?; say "EVAL $name end rc=$rc"; return $rc
}
A=ATTACK=important_instructions
SHARDS=(
  "benign ATTACK=none"
  "bank SUITE=banking $A"
  "slack SUITE=slack $A"
  "tr1 SUITE=travel $A INJ=injection_task_0+injection_task_1+injection_task_2"
  "tr2 SUITE=travel $A INJ=injection_task_3+injection_task_4"
  "tr3 SUITE=travel $A INJ=injection_task_5+injection_task_6"
  "ws1 SUITE=workspace $A INJ=injection_task_0+injection_task_1+injection_task_2+injection_task_3"
  "ws2 SUITE=workspace $A INJ=injection_task_4+injection_task_5+injection_task_6"
  "ws3 SUITE=workspace $A INJ=injection_task_7+injection_task_8+injection_task_9+injection_task_10"
  "ws4 SUITE=workspace $A INJ=injection_task_11+injection_task_12+injection_task_13"
)
failed=()
for s in "${SHARDS[@]}"; do set -- $s; shard "$1" "$NPROC" "${@:2}" || failed+=("$s"); done
still=()
for s in "${failed[@]:-}"; do [ -n "$s" ] || continue; set -- $s; shard "$1-retry" 1 "${@:2}" || still+=("$1"); done

say "STATS begin"
( source "$CONDA_SH" && conda activate agentdojo && cd agentdojo && python eval/compute_attack_stats.py "$RUN_NAME/$MODEL_DIR" )
say "STATS end rc=$?"
if [ ${#still[@]} -gt 0 ]; then say "FAILED shards: ${still[*]}"; exit 1; fi
say "DONE"
