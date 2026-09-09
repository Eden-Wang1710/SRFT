#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:30:00
#SBATCH -J sdL2_shard
#SBATCH -a 0-7
# Submit (any cwd): bash env/sb gpu self_distill/shard.sbatch   (account/partitions/log path from env/<cluster>.sh)
# Locate the repo root (works inside a job: SLURM_SUBMIT_DIR is inside the repo) and load the cluster settings.
SRFT_ROOT="${SRFT_ROOT:-$(d="${SLURM_SUBMIT_DIR:-$PWD}"; while [ ! -f "$d/env/select.sh" ] && [ "$d" != / ]; do d=$(dirname "$d"); done; echo "$d")}"
source "$SRFT_ROOT/env/select.sh"
source "$CONDA_SH" && conda activate sd_gen
export VLLM_LOGGING_LEVEL=WARNING
cd "$SRFT_ROOT/self_distill"
echo "cluster=$SRFT_CLUSTER node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | paste -sd,) task=${SLURM_ARRAY_TASK_ID:-} start=$(date)"
mkdir -p runs/sdL2
python generate_vllm.py --shard $SLURM_ARRAY_TASK_ID --num-shards 8 --rung L2 --n 4 --chunk 256 --out runs/sdL2
echo "end=$(date)"
