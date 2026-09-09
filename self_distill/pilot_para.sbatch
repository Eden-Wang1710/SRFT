#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -J v3para_pilot
# Submit (any cwd): bash env/sb gpu self_distill/pilot_para.sbatch   (account/partitions/log path from env/<cluster>.sh)
# Locate the repo root (works inside a job: SLURM_SUBMIT_DIR is inside the repo) and load the cluster settings.
SRFT_ROOT="${SRFT_ROOT:-$(d="${SLURM_SUBMIT_DIR:-$PWD}"; while [ ! -f "$d/env/select.sh" ] && [ "$d" != / ]; do d=$(dirname "$d"); done; echo "$d")}"
source "$SRFT_ROOT/env/select.sh"
source "$CONDA_SH" && conda activate sd_gen
export VLLM_LOGGING_LEVEL=WARNING
cd "$SRFT_ROOT/self_distill"
echo "cluster=$SRFT_CLUSTER node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | paste -sd,) task=${SLURM_ARRAY_TASK_ID:-} start=$(date)"
python generate_paraphrase.py --steps pilot_para/steps.json --out pilot_para/out.jsonl --n 2 --max-tokens 900
python pilot_report.py pilot_para/out.jsonl pilot_para/review_samples.md 20
echo "end=$(date)"
