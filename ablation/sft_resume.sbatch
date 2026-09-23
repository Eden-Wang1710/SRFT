#!/usr/bin/env bash
#SBATCH --job-name=sft_resume
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --time=06:00:00
# Insurance job for a LLaMA-Factory SFT whose time limit may be too short (docs/09: limits can only be shortened).
# Submit with --dependency=afternotok:<sft job> so it runs only if that job fails / times out. Resumes from the newest
# checkpoint-* in OUT_DIR (epoch checkpoints), or from scratch if none exists. --export: SFT_CONFIG, OUT_DIR (relative to
# LLaMA-Factory/), EXTRA_ARGS (the original run's, e.g. gradient_accumulation_steps=16).
set -euo pipefail
SRFT_ROOT="${SRFT_ROOT:-$(d="${SLURM_SUBMIT_DIR:-$PWD}"; while [ ! -f "$d/env/select.sh" ] && [ "$d" != / ]; do d=$(dirname "$d"); done; echo "$d")}"
ck=$(ls -d "$SRFT_ROOT/LLaMA-Factory/${OUT_DIR:?}"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)
if [ -n "$ck" ]; then echo "resuming from $ck"; export EXTRA_ARGS="${EXTRA_ARGS:-} resume_from_checkpoint=$ck"; else echo "no checkpoint in $OUT_DIR - training from scratch"; fi
export NPROC_PER_NODE=1
exec bash "$SRFT_ROOT/LLaMA-Factory/scripts/train_qwen3_8b_sdL2_sft.slurm"
