#!/bin/bash
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH -t 03:00:00
#SBATCH -J build_env
# Rebuild one conda env from env/freeze/<ENV>.txt (ENV = agentdojo | llamafactory | sd_gen). Cluster-neutral; CPU job.
#   bash env/sb cpu --export=ALL,ENV=agentdojo -J build_agentdojo env/jobs/build_env.sbatch
# Chain the three with --dependency=afterany:<id>: two concurrent `conda create` sharing CONDA_PKGS_DIRS corrupt each other's extraction.
# Recipe (08_environment.md): conda python 3.11 + pip; pip from the freeze minus the editable/conda-file lines (torch +cu128 comes from
# the pytorch extra index for agentdojo/llamafactory; sd_gen's torch 2.7.0 is the PyPI cu126 wheel that vllm 0.9.2 wants);
# then the repo package itself editable with --no-deps so its pyproject does not pull newer langchain / transformers.
set -euo pipefail
SRFT_ROOT="${SRFT_ROOT:-$(d="${SLURM_SUBMIT_DIR:-$PWD}"; while [ ! -f "$d/env/select.sh" ] && [ "$d" != / ]; do d=$(dirname "$d"); done; echo "$d")}"
source "$SRFT_ROOT/env/select.sh"
source "$CONDA_SH"
ENV="${ENV:?set ENV=agentdojo|llamafactory|sd_gen|rlhammer via --export}"
# rlhammer (2026-09-12): RL-Hammer attacker training/eval (injecAgent-rl-harmmer/rl-injector, docs/12). Python 3.12 + rl-injector's
# requirements.txt (vllm 0.11.0 = the DSAI runs' version) + ray (imported by injecagent_eval.py). Named rlhammer, NOT rl-hammer:
# WashU's shared miniconda already has 学长's `rl-hammer` env. First build has no freeze yet -> falls back to requirements.txt.
FREEZE="$SRFT_ROOT/env/freeze/$ENV.txt"; PY=3.11
REQ=$(mktemp)
if [ -f "$FREEZE" ]; then grep -vE '^\s*(#|-e |packaging @ file:)' "$FREEZE" > "$REQ"
elif [ "$ENV" = rlhammer ]; then { cat "$SRFT_ROOT/injecAgent-rl-harmmer/rl-injector/requirements.txt"; echo ray; } > "$REQ"
else echo "missing $FREEZE" >&2; exit 2; fi
[ "$ENV" = rlhammer ] && PY=3.12
echo "cluster=$SRFT_CLUSTER node=$(hostname) job=${SLURM_JOB_ID:-} env=$ENV python=$PY start=$(date) CONDA_PKGS_DIRS=${CONDA_PKGS_DIRS:-} PIP_CACHE_DIR=${PIP_CACHE_DIR:-}"
conda env remove -y -n "$ENV" 2>/dev/null || true          # idempotent re-run
conda create -y -n "$ENV" python=$PY pip
conda activate "$ENV"
case "$ENV" in
  agentdojo)    pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r "$REQ"; pip install --no-deps -e "$SRFT_ROOT/agentdojo" ;;
  llamafactory) pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r "$REQ"; pip install --no-deps -e "$SRFT_ROOT/LLaMA-Factory" ;;
  sd_gen)       pip install -r "$REQ" ;;
  rlhammer)     pip install -r "$REQ" ;;
  *) echo "unknown ENV=$ENV" >&2; exit 2 ;;
esac
pip check
case "$ENV" in
  agentdojo)    python -c "import torch, transformers, peft, click, agentdojo; print('torch', torch.__version__, 'transformers', transformers.__version__, 'peft', peft.__version__, 'click', click.__version__, 'agentdojo', agentdojo.__file__)"
                cd "$SRFT_ROOT/agentdojo" && pytest tests -q 2>&1 | tail -3 ;;
  llamafactory) python -c "import torch, transformers, peft, datasets, accelerate, trl, llamafactory; print('torch', torch.__version__, 'transformers', transformers.__version__, 'peft', peft.__version__, 'datasets', datasets.__version__, 'accelerate', accelerate.__version__, 'trl', trl.__version__)"
                llamafactory-cli version; hf version ;;
  sd_gen)       python -c "import torch, vllm, transformers; print('torch', torch.__version__, 'vllm', vllm.__version__, 'transformers', transformers.__version__)" ;;
  rlhammer)     python -c "import torch, vllm, transformers, trl, peft, accelerate, ray, nltk, litellm; print('torch', torch.__version__, 'vllm', vllm.__version__, 'transformers', transformers.__version__, 'trl', trl.__version__, 'peft', peft.__version__, 'accelerate', accelerate.__version__)" ;;
esac
pip freeze > "$SRFT_ROOT/slurm_logs/freeze_${ENV}_${SRFT_CLUSTER}.txt"
echo "DONE env=$ENV end=$(date)"
