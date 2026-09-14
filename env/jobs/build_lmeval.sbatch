#!/bin/bash
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH -t 03:00:00
#SBATCH -J build_lmeval
# Build the `lmeval` conda env (EleutherAI lm-evaluation-harness) for the general-capability benchmarks
# (MMLU / MMLU-Pro / IFEval / BBH) on the SR-Agent LoRAs. Cluster-neutral; CPU job.
#   bash env/sb cpu env/jobs/build_lmeval.sbatch
# Freezes to env/freeze/lmeval.txt so the other cluster can rebuild it the usual way.
set -euo pipefail
SRFT_ROOT="${SRFT_ROOT:-$(d="${SLURM_SUBMIT_DIR:-$PWD}"; while [ ! -f "$d/env/select.sh" ] && [ "$d" != / ]; do d=$(dirname "$d"); done; echo "$d")}"
source "$SRFT_ROOT/env/select.sh"
source "$CONDA_SH"
export HF_HUB_OFFLINE=0
echo "cluster=$SRFT_CLUSTER node=$(hostname) job=${SLURM_JOB_ID:-} start=$(date)"
conda env remove -y -n lmeval 2>/dev/null || true
conda create -y -n lmeval python=3.11 pip
conda activate lmeval
pip install --extra-index-url https://download.pytorch.org/whl/cu128 torch==2.7.0
pip install "lm-eval[ifeval]==0.4.9" peft==0.17.1 transformers==4.57.1 accelerate==1.11.0
pip check
python -c "import lm_eval, peft, transformers, torch; print('lm_eval', lm_eval.__version__, '| peft', peft.__version__, '| transformers', transformers.__version__, '| torch', torch.__version__)"
python -c "
from lm_eval.tasks import TaskManager
tm=TaskManager(); names=set(tm.all_tasks)
for t in ['mmlu','mmlu_pro','ifeval','bbh','bbh_cot_fewshot']:
    print(f'  task {t}: {\"OK\" if t in names else \"MISSING\"}')"
pip freeze > "$SRFT_ROOT/env/freeze/lmeval.txt"
echo "DONE end=$(date)"
