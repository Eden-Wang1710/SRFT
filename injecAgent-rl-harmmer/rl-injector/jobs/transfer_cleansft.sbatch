#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -c 5
#SBATCH --mem=30G
#SBATCH -t 04:00:00
#SBATCH -J rlh_transfer_cleansft
# Transfer attack on the aba-cleanSFT target (2026-11-11, rebuttal): replay the BEST attacker checkpoint of every Llama-family
# RL-Hammer run against the clean-trajectory SFT model, and -- so the numbers can be read -- against the undefended base and the two
# existing targets for the attackers the 2026-09-18 matrix (jobs/transfer_matrix.sbatch) did not include. Same mechanics as that
# script: the adversarial prompts are read from saved_adv_prompts/, the attacker model is never loaded, 100 test cases per cell.
# Cells that already exist are skipped, so the script is resumable and can run under sbatch or inside an interactive allocation.
# No attacker was trained against the clean-SFT model: every cell is a TRANSFER result, i.e. a lower bound on its adaptive ASR.
set -uo pipefail
SRFT_ROOT="${SRFT_ROOT:-$(d="${SLURM_SUBMIT_DIR:-$PWD}"; while [ ! -f "$d/env/select.sh" ] && [ "$d" != / ]; do d=$(dirname "$d"); done; echo "$d")}"
source "$SRFT_ROOT/env/select.sh"
source "$CONDA_SH" && conda activate rlhammer
cd "$SRFT_ROOT/injecAgent-rl-harmmer/rl-injector"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn WANDB_MODE=disabled
BASE=meta-llama/Llama-3.1-8B-Instruct
L=$SRFT_ROOT/LLaMA-Factory/saves/llama31-8b/lora
STATE="$SRFT_ROOT/slurm_logs/rlh_transfer_cleansft.state"; : > "$STATE"
say(){ echo "[$(date '+%F %T')] $*" | tee -a "$STATE"; }
say "START node=$(hostname) job=${SLURM_JOB_ID:-}"
# attacker_path|save_name|label   (label = run + checkpoint; self-ASR of that checkpoint in the comment)
ATTACKERS=(
 "checkpoints/iclr_rlh_llama_base/checkpoint-765-lora|iclr_rlh_llama_base_checkpoint-765|llamabase_r1_765"                          # 100
 "checkpoints/iclr_rlh_llama_base_r3/checkpoint-816-lora|iclr_rlh_llama_base_r3_checkpoint-816|llamabase_r3_816"                    # 99
 "checkpoints/iclr_rlh_srllama_noappend_cont765_lora_links/checkpoint-816-lora|iclr_rlh_srllama_noappend_checkpoint-816|srllama_r1_816"   # 48
 "checkpoints/iclr_rlh_srllama_append_r2/checkpoint-918-lora|iclr_rlh_srllama_append_r2_checkpoint-918|srllama_append_r2_918"       # 64 (target had the append)
 "checkpoints/iclr_rlh_srllama_noappend_r2/checkpoint-714-lora|iclr_rlh_srllama_noappend_r2_checkpoint-714|srllama_r2_714"          # 7
 "checkpoints/iclr_rlh_ablnothink_lora_links/checkpoint-867-lora|iclr_rlh_ablnothink_checkpoint-867|ablnothink_r1_867"              # 14
 "checkpoints/iclr_rlh_ablnothink_r2_lora_links/checkpoint-357-lora|iclr_rlh_ablnothink_r2_checkpoint-357|ablnothink_r2_357"        # 10
)
# target_label|target path     (all: llama_local format, no system append -- the protocol of the paper's Llama curves)
TARGETS=(
 "cleansft|$L/aba_cleansft_sft_8k_r64_GA4_qkvo_3epoch_5e-6"
 "llamabase|$BASE"
 "srllama|$L/v3base_local_sft_8k_r64_GA4_qkvo_3epoch_5e-6"
 "ablnothink|$L/abl_nothink_sft_8k_r64_GA4_qkvo_3epoch_5e-6"
)
for tspec in "${TARGETS[@]}"; do
  IFS='|' read -r TLAB TGT <<<"$tspec"
  [ "$TGT" = "$BASE" ] || [ -f "$TGT/adapter_model.safetensors" ] || { say "!! no adapter for $TLAB"; continue; }
  for spec in "${ATTACKERS[@]}"; do
    IFS='|' read -r APATH SAVE ALAB <<<"$spec"
    test -f "saved_adv_prompts/${APATH//\//_}/data_InjecAgent_dataset_test/${SAVE}.json" || { say "!! no saved prompts for $ALAB"; continue; }
    run="transfer_${ALAB}_to_${TLAB}"
    [ -f "outputs/${run}/${run}/attack_success_rate.json" ] && { say "have $run"; continue; }
    say "RUN $run"
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
      --run_name "$run" || say "!! FAILED $run"
  done
done
python - <<'PY' | tee -a "$STATE"
import glob, json
rows, tg = {}, ["llamabase", "cleansft", "srllama", "ablnothink"]
for f in sorted(glob.glob("outputs/transfer_*/transfer_*/attack_success_rate.json")):
    a, t = f.split("/")[1].replace("transfer_", "").rsplit("_to_", 1)
    rows.setdefault(a, {})[t] = 100 * float(next(iter(json.load(open(f)).values())))
print("\n=== TRANSFER MATRIX (ASR %, 100 InjecAgent test cases per cell) ===")
print(f"{'attacker':<24}" + "".join(f"{t:>12}" for t in tg))
for a, d in rows.items():
    print(f"{a:<24}" + "".join(f"{d[t]:>12.0f}" if t in d else f"{'-':>12}" for t in tg))
print(f"{'max per target':<24}" + "".join(f"{max((d[t] for d in rows.values() if t in d), default=float('nan')):>12.0f}" for t in tg))
PY
say "DONE"
