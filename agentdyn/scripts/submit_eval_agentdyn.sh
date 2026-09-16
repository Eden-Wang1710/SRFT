#!/bin/bash
# Submit an AgentDyn (arXiv 2602.03117) eval for one Llama checkpoint (cluster-neutral: account/partitions from env/<cluster>.sh).
# usage: bash agentdyn/scripts/submit_eval_agentdyn.sh <RUN_NAME> <abs LoRA dir | /nonexistent for base> [benign|attack|all]
# Override: PARTS=<partitions> NPROC=<procs/GPU> TLIM=<walltime> SYS_APPEND=0|1 SUITES_ONLY="shopping github"
#           SRFT_SBATCH_FLAGS="--test-only"   (dry run)
# NOTE: sbatch --export splits on commas -> injection lists use "+" (eval_parallel.py splits on [,+:; ]).
# Sizes: 20 user tasks/suite; injection tasks shopping 9, github 9, dailylife 10 -> 180+180+200 = 560 attacked, 60 benign.
set -uo pipefail
SRFT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$SRFT_ROOT/env/select.sh"
cd "$SRFT_ROOT/agentdyn"
RUN_NAME=${1:?}; LORA=${2:?}; MODE=${3:-benign}; LOGDIR=runs/$RUN_NAME
PARTS=${PARTS:-$GPU_PARTITIONS}; NPROC=${NPROC:-2}
SYS_APPEND=${SYS_APPEND:-0}                       # ICLR main rows are evaluated WITHOUT the reflection append
MODEL=${MODEL:-LLAMA_3_1_8B_SAFE_AGENT}
ONLY=${SUITES_ONLY:-shopping github dailylife}
S=$SRFT_ROOT/agentdyn/scripts/eval_agentdyn.sbatch
FLAGS="${SRFT_SBATCH_FLAGS:-}"
sub(){ res="-c 5 --mem=30G"; [ "$NPROC" -ge 2 ] && res="-c 7 --mem=42G"
  srft_sbatch --parsable $FLAGS $res -A "$SLURM_ACCOUNT" $SBATCH_EXTRA -p "$PARTS" -t "${TLIM:-04:00:00}" -J "adyn_${RUN_NAME}_$1" -o "$SRFT_SLURM_LOGS/%x_%j.out" \
    --export=ALL,MODEL=$MODEL,RUN_LOGDIR=$LOGDIR,LORA_PATH=$LORA,NPROC=$NPROC,SYS_APPEND=$SYS_APPEND,SUITES=${ONLY// /+},"$2" "$S" 2> >(grep -v -i "billing\|cli_filter\|rates.lua" >&2); }
want(){ case " $ONLY " in *" $1 "*) return 0;; *) return 1;; esac; }
ids=""
if [ "$MODE" = benign ] || [ "$MODE" = all ]; then
  # one job per suite (20 user tasks each) so a slow suite does not block the others
  for s in $ONLY; do ids="$ids:$(sub "benign_$s" "ATTACK=none,SUITES=$s")"; done
fi
if [ "$MODE" = attack ] || [ "$MODE" = all ]; then
  want shopping  && ids="$ids:$(sub sh1 "SUITE=shopping,ATTACK=important_instructions,INJ=injection_task_0+injection_task_1+injection_task_2+injection_task_3+injection_task_4")"
  want shopping  && ids="$ids:$(sub sh2 "SUITE=shopping,ATTACK=important_instructions,INJ=injection_task_5+injection_task_6+injection_task_7+injection_task_8")"
  want github    && ids="$ids:$(sub gh1 "SUITE=github,ATTACK=important_instructions,INJ=injection_task_0+injection_task_1+injection_task_2+injection_task_3+injection_task_4")"
  want github    && ids="$ids:$(sub gh2 "SUITE=github,ATTACK=important_instructions,INJ=injection_task_5+injection_task_6+injection_task_7+injection_task_8")"
  want dailylife && ids="$ids:$(sub dl1 "SUITE=dailylife,ATTACK=important_instructions,INJ=injection_task_0+injection_task_1+injection_task_2+injection_task_3+injection_task_4")"
  want dailylife && ids="$ids:$(sub dl2 "SUITE=dailylife,ATTACK=important_instructions,INJ=injection_task_5+injection_task_6+injection_task_7+injection_task_8+injection_task_9")"
fi
echo "submitted jobs${ids}"
