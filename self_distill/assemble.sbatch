#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -J sdL2_assemble
# Submit (any cwd): bash env/sb cpu self_distill/assemble.sbatch   (account/partitions/log path from env/<cluster>.sh)
# Locate the repo root (works inside a job: SLURM_SUBMIT_DIR is inside the repo) and load the cluster settings.
SRFT_ROOT="${SRFT_ROOT:-$(d="${SLURM_SUBMIT_DIR:-$PWD}"; while [ ! -f "$d/env/select.sh" ] && [ "$d" != / ]; do d=$(dirname "$d"); done; echo "$d")}"
source "$SRFT_ROOT/env/select.sh"
source "$CONDA_SH" && conda activate agentdojo
export VLLM_LOGGING_LEVEL=WARNING
cd "$SRFT_ROOT/self_distill"
echo "cluster=$SRFT_CLUSTER node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | paste -sd,) task=${SLURM_ARRAY_TASK_ID:-} start=$(date)"
python assemble_trajectories.py --runs runs/sdL2 --out ../LLaMA-Factory/data/toucan_32B_v2_sdL2.json
echo "end=$(date)"
