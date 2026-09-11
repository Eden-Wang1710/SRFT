#!/bin/bash
# Submit the two SR-Agent-Qwen3-4B evaluations (think budget 1024, with and without the system-prompt append).
# Run directly, or chain it behind the training job:
#   bash env/sb cpu -c 2 --mem=8G -t 00:20:00 -J q4b_eval_chain --dependency=afterok:<train_jobid> \
#        --wrap="bash $PWD/multibase/submit_q4b_sr_evals.sh"
set -euo pipefail
SRFT_ROOT="${SRFT_ROOT:-$(d="${SLURM_SUBMIT_DIR:-$PWD}"; while [ ! -f "$d/env/select.sh" ] && [ "$d" != / ]; do d=$(dirname "$d"); done; echo "$d")}"
source "$SRFT_ROOT/env/select.sh"
LORA="$SRFT_ROOT/LLaMA-Factory/saves/qwen3-4b/lora/v3base_traj_sft_8k_r64_GA4_qkvo_3epoch_5e-6"
test -f "$LORA/adapter_model.safetensors" || { echo "ERROR: no adapter at $LORA" >&2; exit 1; }
echo "cluster=$SRFT_CLUSTER node=$(hostname) LORA=$LORA start=$(date)"
# WashU 2026-09-11: nodes can sit IDLE in slurm's view with almost no free RAM (leaked processes from a previous job); every job
# dispatched there dies in ~1 s with RaisedSignal:53 and no output file. Re-check before a big submission with the one-liner in
# docs/09_cluster_washu.md and override this list if it has changed:  SRFT_SBATCH_FLAGS="--exclude=<nodes>" bash multibase/submit_q4b_sr_evals.sh
export SRFT_SBATCH_FLAGS="${SRFT_SBATCH_FLAGS:---exclude=c2-gpu-008,c2-gpu-010}"
export MODEL=QWEN_3_4B_SAFE_AGENT THINK_BUDGET=1024 TLIM=06:00:00
SYS_APPEND=1 bash "$SRFT_ROOT/agentdojo/scripts/submit_eval_sdL2.sh" qwen3_4b_v3base_traj_3epoch_think1024 "$LORA"
SYS_APPEND=0 bash "$SRFT_ROOT/agentdojo/scripts/submit_eval_sdL2.sh" qwen3_4b_v3base_traj_3epoch_think1024_noappend "$LORA"
echo "end=$(date)"
