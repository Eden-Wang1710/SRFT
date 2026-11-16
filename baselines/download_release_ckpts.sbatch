#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 04:00:00
#SBATCH -J dl_struq_secalign
# Download the checkpoints RELEASED by the StruQ / SecAlign authors (rebuttal baselines, 2026-11-15):
#   SecAlign : LoRA adapter on meta-llama/Meta-Llama-3-8B-Instruct   (facebookresearch/SecAlign `setup.py --instruct`, 0.1 GB)
#   StruQ    : full model, huggyllama/llama-7b fine-tuned with StruQ  (Sizhe-Chen/StruQ `setup.py --model`, 25 GB)
# Same host and file names as those two setup.py scripts (https://dl.fbaipublicfiles.com/SecAlign/<model_path>.zip).
# Also caches the gated base model Meta-Llama-3-8B-Instruct in $HF_HOME. Models land in baselines/models/ (git-ignored).
# usage: bash env/sb cpu baselines/download_release_ckpts.sbatch
set -euo pipefail
SRFT_ROOT="${SRFT_ROOT:-$(d="${SLURM_SUBMIT_DIR:-$PWD}"; while [ ! -f "$d/env/select.sh" ] && [ "$d" != / ]; do d=$(dirname "$d"); done; echo "$d")}"
source "$SRFT_ROOT/env/select.sh"
source "$CONDA_SH" && conda activate agentdojo
M="$SRFT_ROOT/baselines/models"; mkdir -p "$M"; cd "$M"
get(){ # <dir>/<name>
  local p=$1; [ -d "$p" ] && { echo "have $p"; return; }
  mkdir -p "$(dirname "$p")"
  wget -nv -O "$p.zip" "https://dl.fbaipublicfiles.com/SecAlign/$p.zip"
  unzip -q "$p.zip" -d "$(dirname "$p")" && rm "$p.zip"
  echo "got $p: $(du -sh "$p" | cut -f1)"; ls "$p" | head -20
}
get "meta-llama/Meta-Llama-3-8B-Instruct_dpo_NaiveCompletion_2024-11-12-17-59-06-resized"
HF_HUB_OFFLINE=0 python -c "from huggingface_hub import snapshot_download; print(snapshot_download('meta-llama/Meta-Llama-3-8B-Instruct', ignore_patterns=['original/*']))"
get "huggyllama/llama-7b_SpclSpclSpcl_NaiveCompletion_2025-03-12-01-02-37"
# the authors' UNDEFENDED llama-7b in the same SpclSpclSpcl format (reference row for StruQ)
get "huggyllama/llama-7b_SpclSpclSpcl_None_2025-03-12-01-01-20"
echo DONE
