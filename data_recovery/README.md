# Data recovery: fill the 3,185 missing mid-turn assistant replies

Defect: in `LLaMA-Factory/data/toucan_32B_v2.json`, every assistant reply that precedes a new user turn is `<think>…</think>` with an
empty answer (3,185 of 6,892 `gpt` turns, in 1,828 multi-user-turn trajectories). The trajectory-level source files
(`qwen3-32b-bedrock-samples/<suite>-cot-one-trajectory/…json`) already have `content: None` there, but the PER-STEP source files
(`qwen3-32b-bedrock-samples/<suite>/user_task_N/injection_task_M/assistant_step_NNN.json`) carry `expert_assistant_message` — and Claude's
reflection at those steps discusses the expert answer's content in detail, so the text existed at that stage. ⇒ recover from per-step files.

## 1. Pull from DSAI (run on DSAI or from a machine that can reach both)
Source on DSAI: `/scratch/cxiao13/zixuan/SRFT/agentdojo/qwen3-32b-bedrock-samples/`
Destination here: `~/scratch_cxiao13/zwang544/SRFT/agentdojo/qwen3-32b-bedrock-samples/`

Recommended — everything except the trajectory-level dirs (≈22,456 files, est. <1 GB; gives injection ground truth for every step):
```
rsync -av --exclude='*-cot-one-trajectory' \
  dsai:/scratch/cxiao13/zixuan/SRFT/agentdojo/qwen3-32b-bedrock-samples/ \
  skipjack:/weka/scratch/jhu/cxiao13/zwang544/SRFT/agentdojo/qwen3-32b-bedrock-samples/
```
Minimal — only the 3,185 files needed to fill the replies (list built from the training data):
```
rsync -av --files-from=missing_reply_steps.txt \
  dsai:/scratch/cxiao13/zixuan/SRFT/agentdojo/qwen3-32b-bedrock-samples/ \
  skipjack:/weka/scratch/jhu/cxiao13/zwang544/SRFT/agentdojo/qwen3-32b-bedrock-samples/
```
(`all_steps.txt` = all 22,456 expected files, same relative-path convention.)
Quick check before a full pull: `Minecraft_v2/user_task_85/injection_task_7/assistant_step_008.json` is a mid-turn reply step; its
`expert_assistant_message.content` must be non-empty.

## 2. Rebuild
```
conda activate agentdojo
python rebuild_from_steps.py --steps-root ../agentdojo/qwen3-32b-bedrock-samples \
   --in ../LLaMA-Factory/data/toucan_32B_v2.json --out ../LLaMA-Factory/data/toucan_32B_v3_base.json --strict
```
The script verifies every step file against the training data (think text, tool call, non-empty answers must match), fills the empty
mid-turn answers with `expert_assistant_message.content`, and adds to `meta`: `injections` (trigger/task/target message) and per-step
`has_injection_inserted_by_step` + candidate-sample stats. Report in `<out>.report.json` (`filled` should be 3,185, `still_empty` 0).
Then register `toucan_32B_v3_base` in `dataset_info.json` (same columns as `toucan_32B_v2`) and train v0' with `qwen3_8b_lora_sft_think.yaml`.
Dry-run 2026-09-09 on two samples (final step + mid-turn step 008 of Minecraft_v2 user_task_85/injection_task_7): 2 found, 1 filled, 0 mismatches.

## 3. Done 2026-09-09
Pulled 22,930 files (355 MB). Rebuild `--strict`: 22,456 found, 3,185 filled, 0 empty, 0 mismatches. 18 steps keep the pre-existing
`{"_raw": …}` tool arguments (malformed JSON in the source). Side finding: 11,482 tool-call steps have expert text next to the call in
`expert_assistant_message.content` — not restorable in the ShareGPT function_call format, same as v0. Dataset registered as `toucan_32B_v3_base`.
`--drop-raw-trajectories` removes the 9 hotel user_task_134 trajectories whose expert issued malformed `{"_raw": …}` calls; the current
`toucan_32B_v3_base.json` was built WITH it (3,698 traj / 22,339 steps, 0 empty answers, 0 unparsable tool calls; dropped list in the report).
