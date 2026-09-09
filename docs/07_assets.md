# Assets: where things live

Root: `/weka/scratch/jhu/cxiao13/zwang544/SRFT/` (all paths below are relative to it).

## Migration status (from DSAI `/scratch/cxiao13/zixuan/SRFT`)
| Component | Status | Notes |
|---|---|---|
| `agentdojo/` | migrated | main benchmark (fork of ethz-spylab/agentdojo 0.1.34 with local Qwen/SecAlign LLM elements) |
| `LLaMA-Factory/` | migrated | training framework + training data + LoRA save |
| `27619_Self_Reflection_Fine_Tun.pdf` | migrated | NeurIPS submission |
| `injecAgent-rl-harmmer/` | **not migrated** | dynamic-attack (RL-Hammer on InjecAgent) benchmark, used in paper Sec. 5.3 |
| `SR-Agent/` | not migrated | old-server ckpt dir (the eval script default `QWEN_SAFE_AGENT_LORA_PATH=../SR-Agent` pointed here) |
| `toucan/` | not migrated (ignore for now) | data-construction repo |
| `rl-injector-agentdojo/`, `AgentDyn/` | not migrated (ignore) | results not in paper |

## Training data
- **Paper data:** `LLaMA-Factory/data/toucan_32B_v2.json` — 3707 ShareGPT-style samples (`conversations`/`system`/`meta`),
  74 MB, registered as dataset `toucan_32B_v2` in `LLaMA-Factory/data/dataset_info.json`.
  Assistant turns contain `<think>…</think>` self-reflection followed by `<tool_call>`.
- Other datasets referenced in `dataset_info.json` (e.g. `qwen3_secalign_sft_toucan_32B_v2_mix`) are NOT present on disk — only `toucan_32B_v2.json` was copied.

## Checkpoints
- **Paper LoRA:** `LLaMA-Factory/saves/qwen3-8b/lora/toucan_32B_v2_sft_8k_r64_GA4_qkvo_3epoch_5e-6/`
  (adapter_model.safetensors 245 MB, adapter_config r64/alpha96 q,k,v,o, tokenizer files, trainer_state, training_loss.png).
  Trained with PEFT 0.17.1 / Transformers 4.57.1 / torch 2.10+cu128, 4 GPUs, final train_loss 0.978, 7355 s.
- Base model Qwen/Qwen3-8B is pulled from HF hub on first use; HF cache = `$HF_HOME` =
  `~/scratch_cxiao13/zwang544/.cache/huggingface` (set in the login env; empty as of 2026-09-04).
- Training config template: `LLaMA-Factory/examples/train_lora/qwen3_8b_lora_sft_think.yaml`
  (currently edited to dataset `toucan_32B_v3.2`; change `dataset:` and `output_dir:` back to `toucan_32B_v2` to reproduce).

## AgentDojo trajectories (`agentdojo/runs/<run_name>/<pipeline_name>/<suite>/…`)
- **Paper model run:** `agentdojo/runs/3_01_toucan_32B_v2_sft_8k_r64_GA4_qkvo_3epoch_5e-6/Qwen_Qwen3-8B-safe-agent/`
  - `<suite>/user_task_N/important_instructions/injection_task_M.json` — 949 attacked trajectories (banking 144, slack 105, travel 140, workspace 560)
  - `<suite>/injection_task_M/none/none.json` — 35 "pure injection" runs (injection task executed as the user task, no attack)
  - **No benign `user_task_N/none/none.json` runs here.** Benign-utility numbers in the paper come from a separate no-attack run
    (`qwen3_8b_toucan_lora_think_sys_append_no_attack`, only its CSV survived in `agentdojo/eval/`, run dir not migrated).
- Per-file JSON keys: `suite_name, pipeline_name, user_task_id, injection_task_id, attack_type, injections, messages, error,
  benchmark_version (v1.2.1), utility, security, duration`. **`security: true` = attack succeeded** (repo convention).

## Self-distilled data (2026-09-05)
- `LLaMA-Factory/data/toucan_32B_v2_sdL2.json` — same 3707 trajectories / 22,456 steps as `toucan_32B_v2.json`, with 20,030 assistant thinks
  rewritten by Qwen3-8B (rung L2: Claude reflection as reference), 2,426 steps keep the Claude think. Registered as dataset `toucan_32B_v2_sdL2`.
  Stats: `toucan_32B_v2_sdL2.stats.json`. Recipe and numbers: `docs/04_self_distill_plan.md`.
- `toucan_32B_v2_sdL2fa.json` (v2 source: same thinks, expert answers restored) and `toucan_32B_v2_sdL2fa_perstep.json` (19,271 per-step samples, v2 training set); `toucan_32B_v2_sdL2_perstep.json` (22,002, v1b training set). All registered.
- Code: `SRFT/self_distill/` (`common.py`, `generate_vllm.py`, `assemble_trajectories.py`, `shard.sbatch`, `assemble.sbatch`; smoke: `generate_hf.py`; v3 paraphrase: `paraphrase.py`, `generate_paraphrase.py`, `select_pilot_steps.py`, `pilot_report.py`). Since 2026-09-09 `common.py` reads `toucan_32B_v3_base.json` and uses its `meta.steps` injection ground truth.
  Raw per-step attempts (all 4 samples + filter verdicts): `self_distill/runs/sdL2/shard*.jsonl`; reading sample: `runs/sdL2/sample_trajectories.md`.
- Env for generation: conda `sd_gen` (vllm 0.9.2, torch 2.7.0+cu126, transformers 4.53.2). Qwen/Qwen3-8B weights in the HF cache on scratch (16 GB).

## Evaluation outputs
- `agentdojo/eval/attack_stats_<run>.csv|png` — per-suite utility/ASR (see 06_eval_agentdojo.md). Many historical runs' CSVs kept here
  (baseline Qwen3-8B, Meta-SecAlign-8B, SecAlign-DPO variants, 32B variants, no-think ablation `qwen3_8b_toucan32B_v2_sft_no_sys_append_no_think`).
- `agentdojo/eval/utility_safety_stats_*.csv`, `assistant_think_tokens_*.csv`, `assistant_all_content_tokens_*.csv` — Table 3 style token stats.
- `agentdojo/check/AR_baseline.txt`, `AR_bestnow_txt` — lists of attack-success cases for inspection.
- `agentdojo/logs*/` — old chunked-run logs from DSAI (paths inside refer to `/data/xiaogeng_liu/experiments/zixuan/agentdojo`).

## Tools / environments (outside SRFT)
- Miniforge: `/weka/scratch/jhu/cxiao13/zwang544/tools/miniforge3` (path in `env/skipjack.sh`); env freezes now in `SRFT/env/freeze/`; smoke slurm scripts in `SRFT/env/jobs/`.
- Since 2026-09-09 data/ckpts are also exchanged via private HF repos `srft-data` / `srft-ckpts` (`10_sync_workflow.md`); git-ignored locally.

## Source runs `qwen3-32b-bedrock-samples` (checked 2026-09-06 on 3 sample files from DSAI)
Samples at `~/scratch_cxiao13/zwang544/injection_task_1.json`, `injection_task_7.json`, `injection_task_7-2.json`.
Schema: `{trajectory, messages[]}`; assistant messages carry `content` (text), `tool_calls`, `cot_self_reflection` (= the Claude think in the
training data). Findings:
- Last-turn answer text = training `gpt` text (matches). **Mid-conversation assistant replies (followed by a `user` turn) have `content: None`
  already at this stage** (e.g. Minecraft_v2 user_task_85: 4 of 5 replies empty) → the 3,185 empty replies were lost UPSTREAM of these runs
  (TOUCAN→agentdojo conversion or TOUCAN itself), not in the LLaMA-Factory conversion.
- No `injections` field / no candidate actions in these files. So the folder adds nothing beyond `toucan_32B_v2.json`; not worth migrating
  wholesale. What would help: the clean expert trajectories / TOUCAN source records (for the missing replies) and the trigger list in
  `SRFT/toucan` (for injection ground truth).

## Per-step source records `qwen3-32b-bedrock-samples/<suite>/<user_task>/<injection_task>/assistant_step_NNN.json` (checked 2026-09-09)
Sample: `~/scratch_cxiao13/zwang544/assistant_step_010.json` (= `downloader_v1/user_task_2/injection_task_1/assistant_step_010.json` on DSAI).
Schema: `suite_name, source_trajectory, qwen_model_id (qwen.qwen3-32b-v1:0), qwen_sampling_params (T=1.0, n=3), injections{trigger, task,
combined, insert_position, target_message_index}, assistant_message_index, context_messages[], expert_assistant_message{content, tool_calls},
qwen_samples[3] (candidate actions + follows_injection_task_action), has_injection_inserted_by_step, cot_self_reflection`.
Mapping to training data: `conversations[i]` ↔ `assistant_step_{i+1:03d}` (verified: think and answer identical for the sample).
**This is the layer to recover from**: it has the expert message per step (the trajectory-level `*-cot-one-trajectory` files lost the
mid-turn `content`), the injection ground truth (trigger + target message) and the 3 candidate actions.
Recovery kit: `SRFT/data_recovery/` (README with rsync commands, `missing_reply_steps.txt` = 3,185 files, `all_steps.txt` = 22,456,
`rebuild_from_steps.py`). **Confirmed 2026-09-09 on a mid-turn step** (`Minecraft_v2/user_task_85/injection_task_7/assistant_step_008.json`, local copy `~/scratch_cxiao13/zwang544/assistant_step_008.json`): `expert_assistant_message.content` = the full 826-char answer that is empty in the training data; think identical; injection_7 trigger/task present. **Pulled 2026-09-09** (rsync from DSAI by the DSAI-side Claude): `agentdojo/qwen3-32b-bedrock-samples/` = 15 suite dirs, 22,930 files, 355 MB (22,456 match the training set; 474 extra = `windows_v1/*/template_1`).
- `LLaMA-Factory/data/toucan_32B_v3_base.json` (registered `toucan_32B_v3_base`): 3,698 traj / 22,339 steps = v2 with the 3,185 mid-turn replies filled + `meta.injections` / `meta.steps` ground truth, minus the 9 hotel user_task_134 trajectories with malformed expert tool calls. Built by `data_recovery/rebuild_from_steps.py --strict --drop-raw-trajectories` (0 mismatches). Report (incl. dropped list) alongside.

## Per-step SOURCE data recovered from DSAI (2026-09-09) — `agentdojo/qwen3-32b-bedrock-samples/` (355 MB, 22,930 json)
Transferred by the user's other session; layout `<suite_dir>/user_task_N/injection_task_M/assistant_step_NNN.json` (15 suite dirs:
hotel/downloader/Minecraft/windows/email × v1/v1.5/v2). One file per assistant step. Keys: `source_trajectory`, `assistant_message_index`,
`context_messages` (full prefix), `expert_assistant_message` (**content text AND tool_calls**), `qwen_samples` (3 Qwen3-32B candidate
actions, raw + parsed), `injections` (trigger + task text = ground truth), `has_injection_inserted_by_step` (bool = injection already in
context), `cot_self_reflection` (the Claude think), `qwen_model_id`, `qwen_sampling_params`.
- Maps 1:1 onto the 22,456 training steps by (suite dir, trajectory, step order) — verified, 0 missing either way.
- **`suite_name` field is wrong in 16,930 files** (e.g. says hotel_v1 inside Minecraft_v2) — always use the directory name.
- Expert text: all 6,892 final steps have it (incl. the 3,185 mid-turn replies that are empty in `toucan_32B_v2.json`); 11,482 of the
  15,564 tool-call steps ALSO carry assistant text before the call, which the training data never used.
- Injection ground truth vs the heuristic labels used for v1/v2: 165 clean steps were labelled injected, 11 injected labelled clean, step-0 exact.
- Transfer tarball `~/qwen3-32b-bedrock-samples-steps.tgz` (33 MB) verified; safe to delete.

## Recovered dataset `LLaMA-Factory/data/toucan_32B_v3_base.json` (registered `toucan_32B_v3_base`)
Built by `self_distill/recover_replies.py`: identical to `toucan_32B_v2.json` except the 3,185 empty mid-turn `gpt` replies now carry the
expert reply text (mean reply length 206 words, 0 empty). Nothing else changed (verified: 3,185 changed messages, all role gpt).
Per-step ground truth for every training step: `self_distill/recovery/step_labels.json` (injection_in_context, injections, expert_has_text, n_candidates).
- The 474 files beyond 22,456 are all `windows_v1/user_task_*/template_1/injection_task_*/…` — an older injection-template variant
  (cf. `agentdojo/scripts/remove_template_1_dirs.py`); not part of the training data, ignore.
