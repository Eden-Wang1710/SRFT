#!/bin/bash
# Submit the AgentDojo eval for one LoRA as 10 single-GPU jobs + 1 stats job (cluster-neutral: account/partitions from env/<cluster>.sh).
# usage: bash scripts/submit_eval_sdL2.sh <RUN_NAME> <abs LoRA dir>        (run from SRFT/agentdojo; any cwd works)
# Override: PARTS=<partitions> NPROC=<procs/GPU> WS_NPROC=<procs/GPU for workspace> SRFT_SBATCH_FLAGS="--test-only"
# NOTE: sbatch --export splits on commas -> injection lists use "+" (eval_parallel.py splits on [,+:]).
# Split chosen from measured per-trajectory times (workspace 560 traj ≈ 58 s, travel 140 ≈ 154 s): each job ≤ ~2.5 h with -t 4h.
set -uo pipefail
SRFT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$SRFT_ROOT/env/select.sh"
cd "$SRFT_ROOT/agentdojo"
RUN_NAME=${1:?}; LORA=${2:?}; LOGDIR=runs/$RUN_NAME
PARTS=${PARTS:-$GPU_PARTITIONS}; NPROC=${NPROC:-2}
SYS_APPEND=${SYS_APPEND:-1}; THINK_BUDGET=${THINK_BUDGET:-512}          # inference knobs (2026-09-10); paper setting = 1 / 512
MODEL=${MODEL:-QWEN_3_8B_SAFE_AGENT}                                      # 2026-09-11: LLAMA_3_1_8B_SAFE_AGENT for SR-Agent-Llama
case "$MODEL" in
  QWEN_3_8B_SAFE_AGENT) MODEL_DIR=Qwen_Qwen3-8B-safe-agent ;;
  LLAMA_3_1_8B_SAFE_AGENT) MODEL_DIR=meta-llama_Llama-3.1-8B-Instruct-safe-agent ;;
  *) echo "submit_eval_sdL2.sh: unknown MODEL=$MODEL" >&2; exit 2 ;;
esac
ONLY=${SUITES_ONLY:-workspace travel slack banking}                        # e.g. SUITES_ONLY="travel slack banking" skips the 4 workspace shards   # 2 procs/GPU ≈ 35 GB -> fits L40S (46 GB); 3 procs only gave 1.15x anyway
S=$SRFT_ROOT/agentdojo/scripts/eval_sdL2.sbatch
FLAGS="${SRFT_SBATCH_FLAGS:-}"
# workspace shards run 1 proc/GPU: with 2 procs a long (looping, 15-turn) workspace trajectory OOMs a 46 GB L40S (seen 2026-09-08)
# MaxMemPerCPU=6000 (skipjack): CPUs = ceil(mem/6 GB). Busy GPU nodes often have free GPUs but only 5-8 free CPUs, so request only
# what the procs need: 1 proc -> 30 GB / 5 CPUs, 2 procs -> 42 GB / 7 CPUs (2026-09-10; the 64 GB default sat 1 h unscheduled).
sub(){ np=$NPROC; case "$1" in ws*) np=${WS_NPROC:-1};; esac; res="-c 5 --mem=30G"; [ "$np" -ge 2 ] && res="-c 7 --mem=42G"
  srft_sbatch --parsable $FLAGS $res -A "$SLURM_ACCOUNT" $SBATCH_EXTRA -p "$PARTS" -t "${TLIM:-04:00:00}" -J "eval_${RUN_NAME}_$1" -o "$SRFT_SLURM_LOGS/%x_%j.out" \
    --export=ALL,MODEL=$MODEL,RUN_LOGDIR=$LOGDIR,LORA_PATH=$LORA,NPROC=$np,SYS_APPEND=$SYS_APPEND,THINK_BUDGET=$THINK_BUDGET,SUITES=${ONLY// /+},"$2" "$S" 2> >(grep -v -i "billing\|cli_filter\|rates.lua" >&2); }
# ^ only sbatch's STDOUT (the job id) may be captured: since 2026-09-11 skipjack's cli_filter prints a rates.lua warning on stderr, and
#   capturing it (old `2>&1 | grep -v BILLING`) put garbage into $ids -> the stats job's --dependency broke and it was never submitted.
want(){ case " $ONLY " in *" $1 "*) return 0;; *) return 1;; esac; }
ids=""
want workspace && ids="$ids:$(sub ws1 "SUITE=workspace,ATTACK=important_instructions,INJ=injection_task_0+injection_task_1+injection_task_2+injection_task_3")"
want workspace && ids="$ids:$(sub ws2 "SUITE=workspace,ATTACK=important_instructions,INJ=injection_task_4+injection_task_5+injection_task_6")"
want workspace && ids="$ids:$(sub ws3 "SUITE=workspace,ATTACK=important_instructions,INJ=injection_task_7+injection_task_8+injection_task_9+injection_task_10")"
want workspace && ids="$ids:$(sub ws4 "SUITE=workspace,ATTACK=important_instructions,INJ=injection_task_11+injection_task_12+injection_task_13")"
want travel && ids="$ids:$(sub tr1 "SUITE=travel,ATTACK=important_instructions,INJ=injection_task_0+injection_task_1+injection_task_2")"
want travel && ids="$ids:$(sub tr2 "SUITE=travel,ATTACK=important_instructions,INJ=injection_task_3+injection_task_4")"
want travel && ids="$ids:$(sub tr3 "SUITE=travel,ATTACK=important_instructions,INJ=injection_task_5+injection_task_6")"
want slack && ids="$ids:$(sub slack "SUITE=slack,ATTACK=important_instructions")"
want banking && ids="$ids:$(sub bank "SUITE=banking,ATTACK=important_instructions")"
ids="$ids:$(sub benign "ATTACK=none")"
echo "submitted jobs${ids}"
st=$(srft_sbatch --parsable $FLAGS -A "$SLURM_ACCOUNT" $SBATCH_EXTRA -p "$CPU_PARTITIONS" -c 2 --mem=8G -t 00:20:00 -J "stats_${RUN_NAME}" --dependency=afterany${ids} \
  -o "$SRFT_SLURM_LOGS/%x_%j.out" \
  --wrap="source $CONDA_SH && conda activate agentdojo && cd $SRFT_ROOT/agentdojo && python eval/compute_attack_stats.py ${RUN_NAME}/${MODEL_DIR}" 2> >(grep -v -i "billing\|cli_filter\|rates.lua" >&2))
echo "stats job $st"
