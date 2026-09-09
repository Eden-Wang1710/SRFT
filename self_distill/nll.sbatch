#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 00:40:00
#SBATCH -J nll_diag
# Submit (any cwd): bash env/sb gpu self_distill/nll.sbatch   (account/partitions/log path from env/<cluster>.sh)
# Locate the repo root (works inside a job: SLURM_SUBMIT_DIR is inside the repo) and load the cluster settings.
SRFT_ROOT="${SRFT_ROOT:-$(d="${SLURM_SUBMIT_DIR:-$PWD}"; while [ ! -f "$d/env/select.sh" ] && [ "$d" != / ]; do d=$(dirname "$d"); done; echo "$d")}"
source "$SRFT_ROOT/env/select.sh"
source "$CONDA_SH" && conda activate agentdojo
export VLLM_LOGGING_LEVEL=WARNING
cd "$SRFT_ROOT/self_distill"
echo "cluster=$SRFT_CLUSTER node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | paste -sd,) task=${SLURM_ARRAY_TASK_ID:-} start=$(date)"
python nll_diagnostic.py --n 600 --out runs/nll_diagnostic.json
echo "end=$(date)"
