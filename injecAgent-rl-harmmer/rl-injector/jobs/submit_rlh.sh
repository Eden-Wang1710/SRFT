#!/bin/bash
# Submit one ICLR RL-Hammer run: attacker training (4 GPUs: 3 attacker + 1 vLLM) and, chained afterok, the per-checkpoint eval (1 GPU).
# usage: bash injecAgent-rl-harmmer/rl-injector/jobs/submit_rlh.sh <sr_llama|llama_base> <SYS_APPEND 0|1> <RUN_NAME>
# The three ICLR runs (docs/12 §ICLR protocol):
#   skipjack: submit_rlh.sh sr_llama 1 iclr_rlh_srllama_append ; submit_rlh.sh sr_llama 0 iclr_rlh_srllama_noappend
#   washu   : submit_rlh.sh llama_base 0 iclr_rlh_llama_base
# Override: SR_LORA=<abs dir>, TRAIN_PARTS / EVAL_PARTS, NGPU=4, TLIM_TRAIN=14:00:00, TLIM_EVAL=08:00:00, SEED, EXTRA_ARGS, SRFT_SBATCH_FLAGS="--test-only"
# Default partitions = the cluster's GPU partitions minus l40s (46 GB: too small for the attacker) and minus preemptible ones
# (the trainer saves model-only checkpoints, a requeued job would restart from step 0).
set -uo pipefail
SRFT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "$SRFT_ROOT/env/select.sh"
TARGET=${1:?sr_llama|llama_base}; SYS_APPEND=${2:?0|1}; RUN_NAME=${3:?run name}
case "$RUN_NAME" in *attack*) echo "RUN_NAME must not contain 'attack'" >&2; exit 2;; esac
SR_LORA=${SR_LORA:-$SRFT_ROOT/LLaMA-Factory/saves/llama31-8b/lora/v3base_local_sft_8k_r64_GA4_qkvo_3epoch_5e-6}
if [ "$TARGET" = sr_llama ] && [ ! -f "$SR_LORA/adapter_model.safetensors" ]; then echo "SR_LORA missing: $SR_LORA" >&2; exit 2; fi
DEF_PARTS=$(echo "$GPU_PARTITIONS" | tr ',' '\n' | grep -v -e l40s -e preempt | paste -sd,)
TRAIN_PARTS=${TRAIN_PARTS:-$DEF_PARTS}; EVAL_PARTS=${EVAL_PARTS:-$DEF_PARTS}; NGPU=${NGPU:-4}
FLAGS="${SRFT_SBATCH_FLAGS:-}"
EXP="ALL,TARGET=$TARGET,SYS_APPEND=$SYS_APPEND,RUN_NAME=$RUN_NAME,SR_LORA=$SR_LORA,SEED=${SEED:-1024}"
D=$SRFT_ROOT/injecAgent-rl-harmmer/rl-injector/jobs
cd "$SRFT_ROOT"
# only sbatch's STDOUT (the job id) is captured — skipjack's cli_filter prints warnings on stderr (docs/09_cluster_skipjack.md)
tr=$(srft_sbatch --parsable $FLAGS -A "$SLURM_ACCOUNT" $SBATCH_EXTRA -p "$TRAIN_PARTS" --gres=gpu:$NGPU -c $((4 * NGPU)) --mem=$((24 * NGPU))G \
  -t "${TLIM_TRAIN:-14:00:00}" -J "rlh_train_$RUN_NAME" -o "$SRFT_SLURM_LOGS/%x_%j.out" \
  --export="$EXP,EXTRA_ARGS=${EXTRA_ARGS:-}" "$D/train_attacker.sbatch" 2> >(grep -v -i "billing\|cli_filter\|rates.lua" >&2))
echo "train job $tr  ($TRAIN_PARTS, $NGPU GPUs)"
[[ "$tr" =~ ^[0-9]+$ ]] || { echo "training submission failed; eval not submitted" >&2; exit 1; }
ev=$(srft_sbatch --parsable $FLAGS -A "$SLURM_ACCOUNT" $SBATCH_EXTRA -p "$EVAL_PARTS" --dependency=afterok:$tr \
  -t "${TLIM_EVAL:-08:00:00}" -J "rlh_eval_$RUN_NAME" -o "$SRFT_SLURM_LOGS/%x_%j.out" \
  --export="$EXP" "$D/eval_attacker_ckpts.sbatch" 2> >(grep -v -i "billing\|cli_filter\|rates.lua" >&2))
echo "eval job $ev  (afterok:$tr, $EVAL_PARTS)"
