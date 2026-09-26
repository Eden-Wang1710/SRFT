#!/bin/bash
# Build the anonymous ICLR repository from the local SRFT clone (no network needed).
#   bash build_anon_repo.sh <SRFT root> <output dir>
set -euo pipefail
SRC=$(cd "${1:?SRFT root}" && pwd)
DST=${2:?output dir}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[ -e "$DST" ] && { echo "refusing to overwrite existing $DST" >&2; exit 1; }
mkdir -p "$DST"
cd "$SRC"

copy_tracked() {   # copy_tracked <subdir> <grep -v -E exclude regex>
  git ls-files "$1" | grep -v -E "$2" | rsync -a --files-from=- ./ "$DST/"
}

echo "== LLaMA-Factory"
copy_tracked LLaMA-Factory '^LLaMA-Factory/(data/|scripts/(add_sys_append|merge_toucan|mix_qwen3_secalign|remove_sys_append|convert_dpo_to_sft|train_)|examples/train_lora/(llama31_8b_lora_sft_abl|qwen3_32b_qlora|qwen3_8b_lora_dpo|qwen3_8b_lora_sft_(abl|sdL2|think|v3para)))'
mkdir -p "$DST/LLaMA-Factory/data"
for f in toucan_32B_v3_base toucan_32B_v3_base_llama_local; do
  gzip -9 -c "LLaMA-Factory/data/$f.json" > "$DST/LLaMA-Factory/data/$f.json.gz"
done
python3 - "$SRC" "$DST" <<'EOF'
import json, sys
src, dst = sys.argv[1], sys.argv[2]
d = json.load(open(f"{src}/LLaMA-Factory/data/dataset_info.json"))
keep = {k: d[k] for k in ["toucan_32B_v3_base", "toucan_32B_v3_base_llama_local"]}
json.dump(keep, open(f"{dst}/LLaMA-Factory/data/dataset_info.json", "w"), indent=2)
EOF
# the Llama yaml header mentions an internal converter script; keep the recipe, drop the two header lines
sed -i '1,2d' "$DST/LLaMA-Factory/examples/train_lora/llama31_8b_lora_sft_v3base_local.yaml"
sed -i '1i ### SR-Agent-Llama: same recipe as qwen3_8b_lora_sft_v3base_traj.yaml with only base model, template and dataset changed.\n### Data = toucan_32B_v3_base rendered in the AgentDojo `local` tool format (toucan_32B_v3_base_llama_local).' \
  "$DST/LLaMA-Factory/examples/train_lora/llama31_8b_lora_sft_v3base_local.yaml"

echo "== agentdojo"
copy_tracked agentdojo '^agentdojo/(runs/|eval/|logs/|check/|scripts/|CLAUDE\.md|extract_security_true\.py|run_safe_agent_sr_agent_injection_chunks\.sh|\.devcontainer/|\.github/workflows/(claude|CODEOWNERS)|src/agentdojo/agent_pipeline/llms/llama\.txt)'
mkdir -p "$DST/agentdojo/eval" "$DST/agentdojo/runs"
cp agentdojo/eval/compute_attack_stats.py "$DST/agentdojo/eval/"; mkdir -p "$DST/agentdojo/scripts"; cp agentdojo/scripts/eval_parallel.py "$DST/agentdojo/scripts/"
# drop the fallback to an old absolute path
python3 - "$DST/agentdojo/eval/compute_attack_stats.py" <<'EOF'
import sys, re
p = sys.argv[1]; s = open(p).read()
s = re.sub(r"\n\s*os\.path\.join\('/data/[^\n]*\n", "\n", s)
s = s.replace("Compute completing rates and attack metrics for runs/Qwen_Qwen3-8B-safe-agent.",
              "Compute AgentDojo metrics (benign utility, utility under attack, ASR) for one run directory.")
open(p, "w").write(s)
EOF
declare -A RUNS=(
  [sr_agent_llama31_8b]=llama31_v3base_local_3epoch_noappend
  [sr_agent_qwen3_8b]=3_01_toucan_32B_v2_sft_8k_r64_GA4_qkvo_3epoch_5e-6
  [sr_agent_qwen3_4b]=qwen3_4b_v3base_traj_3epoch_think1024
)
for new in "${!RUNS[@]}"; do
  rsync -a --exclude '__pycache__' "agentdojo/runs/${RUNS[$new]}/" "$DST/agentdojo/runs/$new/"
done
# old absolute paths in two baseline LLM elements (kept for the code, paths replaced by placeholders)
sed -i 's|"/home/cxiao13/scratch-cxiao13/zixuan/safe-agent-project/"|"/path/to/"|; s|/home/cxiao13/\.\.\./safe-agent-project/|/path/to/|' \
  "$DST/agentdojo/src/agentdojo/agent_pipeline/llms/meta_secalign_llm.py" "$DST/agentdojo/src/agentdojo/agent_pipeline/llms/qwen_secalign_llm.py"

echo "== injecAgent-rl-harmmer"
copy_tracked injecAgent-rl-harmmer '^injecAgent-rl-harmmer/rl-injector/(outputs/|scripts/|launch_scripts/|jobs/|plot_asr_curves\.py|\.gitignore$)'
R="$DST/injecAgent-rl-harmmer/rl-injector"
mkdir -p "$R/jobs" "$R/outputs"
cp "$HERE/train_attacker.sh" "$HERE/eval_attacker_ckpts.sh" "$R/jobs/"
cp "$HERE/rlhammer_stats.py" "$R/"
python3 "$HERE/normalize_outputs.py" "$SRC" "$R/outputs"
sed -i 's|backward failed on WashU ("element 0 of tensors does not require grad", 2026-09-14)|backward failed with "element 0 of tensors does not require grad" otherwise|' "$R/train.py"
cat > "$R/.gitignore" <<'EOF'
__pycache__/
*.py[cod]
*.log
logs/
wandb/
checkpoints/
checkpoints_*_lora_links/
saved_adv_prompts/
outputs/_debug_target_inputs/
EOF

echo "== root files"
cat > "$DST/.gitignore" <<'EOF'
__pycache__/
*.py[cod]
.cache/
LLaMA-Factory/saves/
LLaMA-Factory/data/*.json
!LLaMA-Factory/data/dataset_info.json
agentdojo/logs/
injecAgent-rl-harmmer/rl-injector/checkpoints/
injecAgent-rl-harmmer/rl-injector/logs/
EOF
find "$DST" -name '__pycache__' -type d -prune -exec rm -rf {} +
find "$DST" -name '.DS_Store' -delete
echo "== done: $DST"
du -sh "$DST"/* "$DST"/*/* 2>/dev/null | sort -h | tail -15
