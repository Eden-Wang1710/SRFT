#!/usr/bin/env bash
#SBATCH --job-name=sft_sdL2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --time=06:00:00
#SBATCH --mem=128G
# Submit (any cwd): bash env/sb gpu LLaMA-Factory/scripts/train_qwen3_8b_sdL2_sft.slurm
#   1-GPU variant:  bash env/sb gpu --gres=gpu:1 -c 5 --mem=30G -t 10:00:00 --export=ALL,SFT_CONFIG=examples/train_lora/<cfg>.yaml,EXTRA_ARGS="gradient_accumulation_steps=16" LLaMA-Factory/scripts/train_qwen3_8b_sdL2_sft.slurm
# Account / partitions / log path come from env/<cluster>.sh via env/sb. Logs: SRFT/slurm_logs/sft_sdL2_<jobid>.out
set -euo pipefail
# Locate the repo root (works inside a job: SLURM_SUBMIT_DIR is inside the repo) and load the cluster settings.
SRFT_ROOT="${SRFT_ROOT:-$(d="${SLURM_SUBMIT_DIR:-$PWD}"; while [ ! -f "$d/env/select.sh" ] && [ "$d" != / ]; do d=$(dirname "$d"); done; echo "$d")}"
source "$SRFT_ROOT/env/select.sh"
source "$CONDA_SH" && conda activate llamafactory
cd "$SRFT_ROOT/LLaMA-Factory"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
SFT_CONFIG="${SFT_CONFIG:-examples/train_lora/qwen3_8b_lora_sft_sdL2.yaml}"
NPROC_PER_NODE="${NPROC_PER_NODE:-$(nvidia-smi -L | wc -l)}"
echo "cluster=$SRFT_CLUSTER node=$(hostname) gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd,) NPROC=$NPROC_PER_NODE config=$SFT_CONFIG start=$(date)"
cat "$SFT_CONFIG"
FORCE_TORCHRUN=1 NPROC_PER_NODE="$NPROC_PER_NODE" MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000)) python -m llamafactory.cli train "$SFT_CONFIG" bf16=true fp16=false ${EXTRA_ARGS:-}
echo "end=$(date)"
