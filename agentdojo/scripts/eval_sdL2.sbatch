#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH -J eval_sdL2
# Normally submitted by scripts/submit_eval_sdL2.sh (which adds account/partition/log path from env/<cluster>.sh).
# env vars expected via --export: SUITE (or SUITES for benign), ATTACK (important_instructions | none), INJ (list, optional), RUN_LOGDIR, LORA_PATH, NPROC
set -euo pipefail
# Locate the repo root (works inside a job: SLURM_SUBMIT_DIR is inside the repo) and load the cluster settings.
SRFT_ROOT="${SRFT_ROOT:-$(d="${SLURM_SUBMIT_DIR:-$PWD}"; while [ ! -f "$d/env/select.sh" ] && [ "$d" != / ]; do d=$(dirname "$d"); done; echo "$d")}"
source "$SRFT_ROOT/env/select.sh"
source "$CONDA_SH" && conda activate agentdojo
cd "$SRFT_ROOT/agentdojo"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export QWEN_SAFE_AGENT_LORA_PATH="${LORA_PATH:?}" QWEN_SAFE_AGENT_SYS_APPEND=1 QWEN_SAFE_AGENT_ENABLE_THINKING=1
NPROC="${NPROC:-3}"; ATTACK="${ATTACK:-important_instructions}"
echo "cluster=$SRFT_CLUSTER node=$(hostname) gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) start=$(date)"
echo "LORA=$QWEN_SAFE_AGENT_LORA_PATH SYS_APPEND=$QWEN_SAFE_AGENT_SYS_APPEND THINK=$QWEN_SAFE_AGENT_ENABLE_THINKING RUN_LOGDIR=$RUN_LOGDIR ATTACK=$ATTACK NPROC=$NPROC"
mkdir -p "$RUN_LOGDIR"
if [ "$ATTACK" = "none" ]; then
  for s in ${SUITES:-travel slack banking workspace}; do
    python scripts/eval_parallel.py --suite "$s" --nproc "$NPROC" --logdir "$RUN_LOGDIR"
  done
else
  python scripts/eval_parallel.py --suite "${SUITE:?}" --attack "$ATTACK" --injection-tasks "${INJ:-}" --nproc "$NPROC" --logdir "$RUN_LOGDIR"
fi
echo "end=$(date)"
