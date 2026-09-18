#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -c 5
#SBATCH --mem=30G
#SBATCH -t 04:00:00
#SBATCH -J rlh_transfer
# Transfer matrix (docs/12 "Consequence for any claim made from these curves"): replay each run's BEST attacker checkpoint
# against EVERY target and report the max per target. The adversarial prompts are already saved per checkpoint, so
# injecagent_eval.py never loads the attacker model (see its saved_adv_prompts branch) -- only the target is run, 100 cases.
# The attacker path only has to reproduce the saved_adv_prompts directory name; it need not exist on disk.
set -uo pipefail
SRFT_ROOT="${SRFT_ROOT:-$(d="${SLURM_SUBMIT_DIR:-$PWD}"; while [ ! -f "$d/env/select.sh" ] && [ "$d" != / ]; do d=$(dirname "$d"); done; echo "$d")}"
source "$SRFT_ROOT/env/select.sh"
source "$CONDA_SH" && conda activate rlhammer
cd "$SRFT_ROOT/injecAgent-rl-harmmer/rl-injector"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn WANDB_MODE=disabled
BASE=meta-llama/Llama-3.1-8B-Instruct
SR=$SRFT_ROOT/LLaMA-Factory/saves/llama31-8b/lora/v3base_local_sft_8k_r64_GA4_qkvo_3epoch_5e-6
ABL=$SRFT_ROOT/LLaMA-Factory/saves/llama31-8b/lora/abl_nothink_sft_8k_r64_GA4_qkvo_3epoch_5e-6
echo "node=$(hostname) start=$(date)"
# attacker_path|save_name|attacker_label
ATTACKERS=(
 "checkpoints/iclr_rlh_srllama_noappend_cont765_lora_links/checkpoint-816-lora|iclr_rlh_srllama_noappend_checkpoint-816|srllama_r1_816"
 "checkpoints/iclr_rlh_srllama_noappend_r2/checkpoint-714-lora|iclr_rlh_srllama_noappend_r2_checkpoint-714|srllama_r2_714"
 "checkpoints/iclr_rlh_ablnothink_lora_links/checkpoint-867-lora|iclr_rlh_ablnothink_checkpoint-867|ablnothink_r1_867"
 "checkpoints/iclr_rlh_ablnothink_r2_lora_links/checkpoint-357-lora|iclr_rlh_ablnothink_r2_checkpoint-357|ablnothink_r2_357"
)
for spec in "${ATTACKERS[@]}"; do
  IFS='|' read -r APATH SAVE ALAB <<<"$spec"
  # the replay only needs the saved file to exist under the name derived from APATH
  test -f "saved_adv_prompts/${APATH//\//_}/data_InjecAgent_dataset_test/${SAVE}.json" \
    || { echo "!! no saved prompts for $ALAB ($APATH / $SAVE)"; continue; }
  for tspec in "srllama|$SR" "ablnothink|$ABL"; do
    IFS='|' read -r TLAB TGT <<<"$tspec"
    run="transfer_${ALAB}_to_${TLAB}"
    out="outputs/${run}/attack_success_rate.json"
    [ -f "$out" ] && { echo "done already: $run"; continue; }
    echo "=== $run  $(date)"
    python injecagent_eval.py \
      --attacker_model_name_or_path "$APATH" \
      --attacker_base_model_name_or_path "$BASE" \
      --target_model_name_or_path "$TGT" \
      --target_base_model_name_or_path "$BASE" \
      --safe_agent_mode True \
      --safe_agent_prompt_format llama_local \
      --target_enable_thinking True \
      --use_safe_agent_system_append False \
      --validation_data_path data/InjecAgent/dataset/test.json \
      --val_batch_size "${VAL_BATCH_SIZE:-16}" \
      --val_max_new_tokens 1024 \
      --enable_wandb False \
      --save_name "$SAVE" \
      --run_name "$run" || echo "!! FAILED $run"
  done
done
echo "end=$(date)"
python - <<'PY'
import glob, json
print("\n=== TRANSFER MATRIX (ASR %) ===")
rows={}
for f in sorted(glob.glob("outputs/transfer_*/attack_success_rate.json")):
    run=f.split("/")[1].replace("transfer_","")
    a,t=run.rsplit("_to_",1)
    rows.setdefault(a,{})[t]=100*float(next(iter(json.load(open(f)).values())))
tg=["srllama","ablnothink"]
print(f"{'attacker':<20}" + "".join(f"{t:>12}" for t in tg))
for a,d in rows.items():
    print(f"{a:<20}" + "".join(f"{d.get(t,float('nan')):>12.0f}" for t in tg))
print("\nmax per target:", {t: max((d.get(t,0) for d in rows.values()), default=0) for t in tg})
PY
