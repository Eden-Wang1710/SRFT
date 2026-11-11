#!/bin/bash
# Evaluate several Llama models under ONE AgentDojo attack on a single GPU inside an interactive allocation (tmux + srun).
# Written 2026-11-11 for the rebuttal's tool_knowledge runs; it only chains agentdojo/scripts/eval_sdL2.sbatch (same pipeline,
# same shards as submit_eval_sdL2.sh, SYS_APPEND=0) and eval/compute_attack_stats.py --attack <name>.
# usage: ATTACK=tool_knowledge bash agentdojo/scripts/run_attack_interactive.sh "<run_name>|<lora dir or /nonexistent for base>|<nproc>" ...
# Only the ATTACKED shards are run (949 trajectories); the benign column is the one of the model's important_instructions run.
# Every shard is resumable (benchmark.py skips finished tasks); a failed shard is retried once with 1 proc after the others.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/env/select.sh"
ATTACK=${ATTACK:?attack name, e.g. tool_knowledge}
MODEL=LLAMA_3_1_8B_SAFE_AGENT; MODEL_DIR=meta-llama_Llama-3.1-8B-Instruct-safe-agent
STATE="$ROOT/slurm_logs/attack_${ATTACK}_interactive.state"; : > "$STATE"
say(){ echo "[$(date '+%F %T')] $*" | tee -a "$STATE"; }
JOB=${SLURM_JOB_ID:-nojob}
cd "$ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
say "START attack=$ATTACK job=$JOB node=$(hostname) models=$#"
shard(){ # run lora name nproc VAR=VALUE...
  local run=$1 lora=$2 name=$3 np=$4; shift 4
  say "EVAL $run $name begin nproc=$np"
  # distinct SLURM_JOB_ID per shard: eval_parallel.py names its per-process log dir after it
  env SLURM_JOB_ID="${JOB}_${name}" MODEL=$MODEL RUN_LOGDIR=runs/$run LORA_PATH="$lora" NPROC=$np SYS_APPEND=0 ATTACK=$ATTACK "$@" \
    bash agentdojo/scripts/eval_sdL2.sbatch
  local rc=$?; say "EVAL $run $name end rc=$rc"; return $rc
}
SHARDS=(
  "bank SUITE=banking"
  "slack SUITE=slack"
  "tr1 SUITE=travel INJ=injection_task_0+injection_task_1+injection_task_2"
  "tr2 SUITE=travel INJ=injection_task_3+injection_task_4"
  "tr3 SUITE=travel INJ=injection_task_5+injection_task_6"
  "ws1 SUITE=workspace INJ=injection_task_0+injection_task_1+injection_task_2+injection_task_3"
  "ws2 SUITE=workspace INJ=injection_task_4+injection_task_5+injection_task_6"
  "ws3 SUITE=workspace INJ=injection_task_7+injection_task_8+injection_task_9+injection_task_10"
  "ws4 SUITE=workspace INJ=injection_task_11+injection_task_12+injection_task_13"
)
bad=0
for spec in "$@"; do
  IFS='|' read -r RUN LORA NP <<<"$spec"
  say "MODEL $RUN lora=$LORA"
  failed=()
  for s in "${SHARDS[@]}"; do set -- $s; shard "$RUN" "$LORA" "$1" "$NP" "${@:2}" || failed+=("$s"); done
  still=()
  for s in "${failed[@]:-}"; do [ -n "$s" ] || continue; set -- $s; shard "$RUN" "$LORA" "$1-retry" 1 "${@:2}" || still+=("$1"); done
  ( source "$CONDA_SH" && conda activate agentdojo && cd agentdojo && python eval/compute_attack_stats.py "$RUN/$MODEL_DIR" --attack "$ATTACK" ) 2>&1 | grep -A3 "Scenario: ALL" | tee -a "$STATE"
  if [ ${#still[@]} -gt 0 ]; then say "FAILED $RUN shards: ${still[*]}"; bad=1; else say "MODEL $RUN done"; fi
done
[ $bad -eq 0 ] && say "DONE" || say "DONE-WITH-FAILURES"
