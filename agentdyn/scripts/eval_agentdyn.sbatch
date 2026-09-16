#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -c 7
#SBATCH --mem=42G
#SBATCH -t 04:00:00
#SBATCH -J eval_agentdyn
# AgentDyn (arXiv 2602.03117) eval for one Llama checkpoint on ONE GPU.
# Mirrors agentdojo/scripts/eval_sdL2.sbatch; differences: repo = $SRFT_ROOT/agentdyn, conda env srft_agentdyn,
# suites default to the three AgentDyn suites. Submitted by scripts/submit_eval_agentdyn.sh.
# env vars via --export: MODEL, SUITE (attack mode) or SUITES (benign), ATTACK, INJ, RUN_LOGDIR, LORA_PATH, NPROC, SYS_APPEND
set -euo pipefail
SRFT_ROOT="${SRFT_ROOT:-$(d="${SLURM_SUBMIT_DIR:-$PWD}"; while [ ! -f "$d/env/select.sh" ] && [ "$d" != / ]; do d=$(dirname "$d"); done; echo "$d")}"
source "$SRFT_ROOT/env/select.sh"
# CONDA_ENV: srft_agentdyn is the dedicated env (clone of agentdojo + AgentDyn deps). `agentdojo` also works because
# PYTHONPATH below wins over the installed package -- the assertion after the cd checks that it really did.
source "$CONDA_SH" && conda activate "${CONDA_ENV:-srft_agentdyn}"
cd "$SRFT_ROOT/agentdyn"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
python -c "import agentdojo,sys,pathlib; f=pathlib.Path(agentdojo.__file__).resolve(); sys.exit(0 if '/agentdyn/' in str(f) else f'WRONG agentdojo package: {f}')"
MODEL="${MODEL:-LLAMA_3_1_8B_SAFE_AGENT}"
# A LORA_PATH that does not exist = undefended base Llama (same convention as agentdojo/scripts/eval_sdL2.sbatch).
export LLAMA_SR_AGENT_LORA_PATH="${LORA_PATH:?}" LLAMA_SR_AGENT_SYS_APPEND="${SYS_APPEND:-0}"
NPROC="${NPROC:-2}"; ATTACK="${ATTACK:-important_instructions}"; BV="${BENCHMARK_VERSION:-v1.2.1}"
echo "cluster=$SRFT_CLUSTER node=$(hostname) gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) start=$(date)"
echo "MODEL=$MODEL LORA=$LLAMA_SR_AGENT_LORA_PATH SYS_APPEND=$LLAMA_SR_AGENT_SYS_APPEND RUN_LOGDIR=$RUN_LOGDIR ATTACK=$ATTACK NPROC=$NPROC BV=$BV"
mkdir -p "$RUN_LOGDIR"
if [ "$ATTACK" = "none" ]; then
  for s in ${SUITES:-shopping github dailylife}; do s=${s//+/ }; for s in $s; do
    python scripts/eval_parallel.py --model "$MODEL" --suite "$s" --nproc "$NPROC" --logdir "$RUN_LOGDIR" --benchmark-version "$BV"; done
  done
else
  python scripts/eval_parallel.py --model "$MODEL" --suite "${SUITE:?}" --attack "$ATTACK" --injection-tasks "${INJ:-}" --nproc "$NPROC" --logdir "$RUN_LOGDIR" --benchmark-version "$BV"
fi
echo "end=$(date)"
