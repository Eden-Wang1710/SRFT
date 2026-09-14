#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -J lmeval
# One general-capability benchmark for one model. Cluster-neutral.
#   bash env/sb gpu --export=ALL,TAG=srllama,TASK=mmlu env/jobs/lmeval.sbatch
# env: TAG = base | srllama | <name>   TASK = mmlu | mmlu_pro | ifeval | bbh_cot_fewshot
#      [LORA=<abs adapter dir>] overrides the adapter; [SHOTS] overrides the few-shot count;
#      [LIMIT=<n>] for a smoke run. Results -> eval_general/<TAG>/<TASK>/
# NOTE (2026-09-14): the SR-Agent LoRAs are evaluated WITHOUT the reflection system-prompt append, to match the
# main AgentDojo/RL-Hammer rows. lm-eval sends each task's own prompt; no SRFT prompt is added anywhere.
set -euo pipefail
SRFT_ROOT="${SRFT_ROOT:-$(d="${SLURM_SUBMIT_DIR:-$PWD}"; while [ ! -f "$d/env/select.sh" ] && [ "$d" != / ]; do d=$(dirname "$d"); done; echo "$d")}"
source "$SRFT_ROOT/env/select.sh"
source "$CONDA_SH" && conda activate lmeval
export HF_HUB_OFFLINE=0 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
TAG="${TAG:?base|srllama}"; TASK="${TASK:?mmlu|mmlu_pro|ifeval|bbh_cot_fewshot}"
BASE=meta-llama/Llama-3.1-8B-Instruct
case "$TAG" in
  base)    ARGS="pretrained=$BASE,dtype=bfloat16" ;;
  srllama) ARGS="pretrained=$BASE,dtype=bfloat16,peft=${LORA:-$SRFT_ROOT/LLaMA-Factory/saves/llama31-8b/lora/v3base_local_sft_8k_r64_GA4_qkvo_3epoch_5e-6}" ;;
  *)       ARGS="pretrained=$BASE,dtype=bfloat16,peft=${LORA:?LORA required for a custom TAG}" ;;
esac
# few-shot counts follow the Meta-SecAlign paper's table so the published rows are comparable
case "$TASK" in
  mmlu)            NS=${SHOTS:-0} ;;
  mmlu_pro)        NS=${SHOTS:-5} ;;
  ifeval)          NS=${SHOTS:-0} ;;
  bbh_cot_fewshot) NS=${SHOTS:-3} ;;
  *)               NS=${SHOTS:-0} ;;
esac
OUT="$SRFT_ROOT/eval_general/$TAG/$TASK"
CACHE="$SRFT_ROOT/eval_general/.cache/$TAG"          # request-level cache: a killed run resumes instead of restarting
mkdir -p "$OUT" "$CACHE"
echo "cluster=$SRFT_CLUSTER node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) start=$(date)"
echo "TAG=$TAG TASK=$TASK shots=$NS args=$ARGS"
lm_eval --model hf --model_args "$ARGS" --tasks "$TASK" --num_fewshot "$NS" \
  --batch_size auto --device cuda:0 --output_path "$OUT" --seed 1234 \
  --use_cache "$CACHE/$TASK" --cache_requests true \
  ${LIMIT:+--limit $LIMIT}
echo "end=$(date)"
