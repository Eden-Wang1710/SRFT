#!/bin/bash
# AgentDojo (important_instructions, 97 benign + 949 attacked) for the checkpoints released by the StruQ / SecAlign authors and their
# undefended references, on ONE GPU inside an interactive allocation. Chains agentdojo/scripts/eval_sdL2.sbatch (same shards as
# submit_eval_sdL2.sh) and eval/compute_attack_stats.py; resumable (finished tasks are skipped), failed shards retried with 1 proc.
# usage: bash baselines/run_released_baselines.sh "<run>|<ModelsEnum name>|<path or hub id>|<base id or empty>|<nproc>" ...
# A spec whose path is a local dir that does not exist yet is waited for (the 25 GB StruQ downloads).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/env/select.sh"
STATE="$ROOT/slurm_logs/released_baselines.state"; : > "$STATE"
say(){ echo "[$(date '+%F %T')] $*" | tee -a "$STATE"; }
JOB=${SLURM_JOB_ID:-nojob}
cd "$ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
A=ATTACK=important_instructions
SHARDS=(
  "benign ATTACK=none SUITES=workspace+travel+slack+banking"
  "bank SUITE=banking $A"
  "slack SUITE=slack $A"
  "tr1 SUITE=travel $A INJ=injection_task_0+injection_task_1+injection_task_2"
  "tr2 SUITE=travel $A INJ=injection_task_3+injection_task_4"
  "tr3 SUITE=travel $A INJ=injection_task_5+injection_task_6"
  "ws1 SUITE=workspace $A INJ=injection_task_0+injection_task_1+injection_task_2+injection_task_3"
  "ws2 SUITE=workspace $A INJ=injection_task_4+injection_task_5+injection_task_6"
  "ws3 SUITE=workspace $A INJ=injection_task_7+injection_task_8+injection_task_9+injection_task_10"
  "ws4 SUITE=workspace $A INJ=injection_task_11+injection_task_12+injection_task_13"
)
say "START job=$JOB node=$(hostname) models=$#"
bad=0
for spec in "$@"; do
  IFS='|' read -r RUN MODEL MPATH BASE NP <<<"$spec"
  if [[ "$MPATH" == /* ]]; then
    n=0; until [ -f "$MPATH/config.json" ] && ! ls "$MPATH".zip >/dev/null 2>&1; do
      [ $n -eq 0 ] && say "WAIT for $MPATH"; n=$((n+1)); [ $n -gt 240 ] && break; sleep 30; done
    [ -f "$MPATH/config.json" ] || [ -f "$MPATH/adapter_config.json" ] || { say "FAILED $RUN: no checkpoint at $MPATH"; bad=1; continue; }
  fi
  MODEL_DIR=$(source "$CONDA_SH" && conda activate agentdojo && PYTHONPATH=agentdojo/src python -c "from agentdojo.models import ModelsEnum; print(ModelsEnum['$MODEL'].value.replace('/','_'))")
  say "MODEL $RUN enum=$MODEL dir=$MODEL_DIR path=$MPATH base=${BASE:-<none>}"
  run(){ local name=$1 np=$2; shift 2
    say "EVAL $RUN $name begin nproc=$np"
    env SLURM_JOB_ID="${JOB}_${name}" MODEL=$MODEL RUN_LOGDIR=runs/$RUN LORA_PATH=/unused NPROC=$np SYS_APPEND=0 \
      RELEASED_BASELINE_PATH="$MPATH" RELEASED_BASELINE_BASE="$BASE" "$@" bash agentdojo/scripts/eval_sdL2.sbatch
    local rc=$?; say "EVAL $RUN $name end rc=$rc"; return $rc; }
  failed=()
  for s in "${SHARDS[@]}"; do set -- $s; run "$1" "$NP" "${@:2}" || failed+=("$s"); done
  still=()
  for s in "${failed[@]:-}"; do [ -n "$s" ] || continue; set -- $s; run "$1-retry" 1 "${@:2}" || still+=("$1"); done
  ( source "$CONDA_SH" && conda activate agentdojo && cd agentdojo && python eval/compute_attack_stats.py "$RUN/$MODEL_DIR" ) 2>&1 | grep -A3 "Scenario: ALL" | tee -a "$STATE"
  if [ ${#still[@]} -gt 0 ]; then say "FAILED $RUN shards: ${still[*]}"; bad=1; else say "MODEL $RUN done"; fi
done
[ $bad -eq 0 ] && say "DONE" || say "DONE-WITH-FAILURES"
