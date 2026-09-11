# Training SR-Agent LoRA with LLaMA-Factory on skipjack

## Env `llamafactory` (built 2026-09-05)
python 3.11 · torch 2.7.0+cu128 · LLaMA-Factory 0.9.4.dev0 (editable, `SRFT/LLaMA-Factory`) · transformers 4.57.1 · peft 0.17.1 ·
datasets 4.0.0 · accelerate 1.11.0 · trl 0.9.6 — matches the paper LoRA's recorded framework versions (README in the save dir). No flash-attn
(`flash_attn: auto` → sdpa). Activate: `conda activate llamafactory`.

## Paper recipe (unchanged)
`examples/train_lora/qwen3_8b_lora_sft_think.yaml`: Qwen/Qwen3-8B, LoRA r64 α96 dropout 0.05 on q/k/v/o, template qwen3 + enable_thinking,
cutoff 8192, train_on_prompt false, mask_history false, bs 1 × GA 4 × 4 GPUs (=16), lr 5e-6 cosine warmup 0.1, 3 epochs, bf16, save per epoch.
Paper run: 696 steps, 2 h 02 min on 4×A100 (`saves/.../toucan_32B_v2_sft_8k_r64_GA4_qkvo_3epoch_5e-6/trainer_log.jsonl`).
NOTE the yaml in the repo currently points at `toucan_32B_v3.2` (a later experiment); the paper dataset is `toucan_32B_v2`.

## How to launch (verified 2026-09-05; cluster-neutral form since 2026-09-09)
```bash
cd $SRFT_ROOT                                                       # any cwd works; env/sb adds account/partitions/log path
bash env/sb gpu LLaMA-Factory/scripts/train_qwen3_8b_sdL2_sft.slurm          # 4 GPUs, 6 h, default config in the script
bash env/sb gpu --export=ALL,SFT_CONFIG=examples/train_lora/<other>.yaml LLaMA-Factory/scripts/train_qwen3_8b_sdL2_sft.slurm
# preferred on a busy queue (see 09_cluster_skipjack.md): 1 GPU + GA 16, 30 GB (= 5 CPUs)
bash env/sb gpu --gres=gpu:1 -c 5 --mem=30G -t 10:00:00 \
  --export=ALL,SFT_CONFIG=examples/train_lora/<cfg>.yaml,EXTRA_ARGS="gradient_accumulation_steps=16" \
  LLaMA-Factory/scripts/train_qwen3_8b_sdL2_sft.slurm
```
The launcher = old DSAI `run_sft_qwen3_8b_hotel_v1_4gpu.slurm` adapted: `source $CONDA_SH && conda activate llamafactory`, `HF_HOME`/`HF_HUB_OFFLINE` from
`env/<cluster>.sh`, `FORCE_TORCHRUN=1 NPROC_PER_NODE=<#gpus> python -m llamafactory.cli train <yaml> bf16=true fp16=false`, MASTER_PORT derived from job id.
Logs: `slurm_logs/sft_sdL2_<jobid>.out` (before 2026-09-09: `LLaMA-Factory/logs/`); progress: `<output_dir>/trainer_log.jsonl`.
2-GPU fallback: `--gres=gpu:2` and `gradient_accumulation_steps: 8` (same effective batch 16).
After training: upload the top-level adapter to HF `srft-ckpts` (`10_sync_workflow.md`) and put the HF path in the ledger.

## Data-format gotcha (cost one failed dry run)
The qwen3 template's `thought_words = ("<think>\n", "\n</think>\n\n")`. Assistant values MUST be `<think>\n…\n</think>\n\n<action>`;
with a single newline after `</think>` the function formatter does not strip the thought and fails with "Invalid JSON format in function message".
Offline check without GPU (parses yaml + encodes trajectories):
`python -c` snippet in changelog 2026-09-05 / `get_train_args()` with `bf16=false`, then `template.encode_multiturn(...)`.

## Recipe template from 2026-09-09 on
`examples/train_lora/qwen3_8b_lora_sft_v3base_traj.yaml` (= paper yaml on the recovered data `toucan_32B_v3_base`) is the template for every
further multi-turn run, including the distilled v3 datasets: copy, change `dataset` and `output_dir` only. Launch with the 2-GPU GA-8 line below.

## Runs
| run | dataset | config delta | job | status |
|---|---|---|---|---|
| paper | toucan_32B_v2 | — | (DSAI) | LoRA in saves/…/toucan_32B_v2_sft_8k_r64_GA4_qkvo_3epoch_5e-6 |
| v1b | toucan_32B_v2_sdL2_perstep | per-step, mask_history=true, LF_NO_HISTORY_EMPTY_COT=1; 2×H100 GA 8 | 315868 | DONE 2026-09-06 11:34: 4,128 steps, 4 h 12, train_loss 0.641 |
| v2 | toucan_32B_v2_sdL2fa_perstep | as v1b, expert answers restored, 19,271 samples | 318170 | submitted 2026-09-06 |
| v0' | toucan_32B_v3_base | `qwen3_8b_lora_sft_v3base_traj.yaml` = paper yaml, dataset/output_dir only; 2 GPUs GA 8 | – | cancelled by user 2026-09-09 (config kept as template) |
| v3-para | toucan_32B_v3_para | `qwen3_8b_lora_sft_v3para_traj.yaml` = template, dataset/output_dir only; **1 GPU, NPROC_PER_NODE=1, GA 16**, 10 h | 347215 | DONE 2026-09-09 23:00: 1×L40S (gl106), 696 steps, 7 h 41, train_loss 0.888 (≈ 2.1× the 2×L40S time of v2-traj, as expected) |
| Q4-v3base (Qwen3-4B) | toucan_32B_v3_base (unchanged) | `qwen3_4b_lora_sft_v3base_traj.yaml` = the 8B v0' yaml with `model_name_or_path: Qwen/Qwen3-4B` and `output_dir` changed, nothing else (diff-verified); 1 GPU, GA 16, `-p general-gpu` (no preempt: no auto-resume in the launcher) | 3009792 | RUNNING 2026-09-11 02:38, WashU 1×H100 c2-gpu-005, 12 h limit |
| L-v3base (Llama) | toucan_32B_v3_base_llama_local | `llama31_8b_lora_sft_v3base_local.yaml` = template with model `meta-llama/Llama-3.1-8B-Instruct`, `template: llama3` (no enable_thinking), dataset/output_dir; 1 GPU GA 16, 10 h | 376723 | started 2026-09-11 on 1×L40S (gl111) |

Scheduling lesson (2026-09-09): 1-GPU jobs on `l40s` started within seconds all night while the 2-GPU training request sat 8 h with reason None. For ≤ 8k-token LoRA runs prefer 1 GPU + GA 16 (fits a 46 GB L40S with gradient checkpointing; v2-traj's per-GPU footprint at 8k proved it).
| sdL2 v1 | toucan_32B_v2_sdL2 | dataset + output_dir only (mask_history=false); 2×H200, GA 8 (eff. batch 16), 696 steps | 313882 | DONE 2026-09-05 14:38: 84 min, train_loss 0.804 (paper 0.978); ckpts 232/464/696 = epoch 1/2/3 |

## Second base model: Llama-3.1-8B-Instruct (2026-09-11, branch `exp/llama31-8b-srft`; ledger section "L")
- Weights: `hf download meta-llama/Llama-3.1-8B-Instruct --exclude "original/*"` (gated; the `EdenWong1710` token already has access; 15 GB in `$HF_HOME`).
- Data: `python multibase/convert_llama_local.py` (any env) → `toucan_32B_v3_base_llama_local.json`; then `python multibase/check_llama_render.py
  --n 300 --all-lengths` in `llamafactory` (CPU, ~3 min) must print `N/N identical token sequences` — it compares LLaMA-Factory's `llama3`
  encoding with the inference render of `agentdojo/.../llms/llama_local_prompt.py`. Re-run it after touching either side.
- Non-reasoning template: the `<think>…</think>` reflection is plain assistant text, so the qwen3 format gotcha below does not apply; the base
  `Template` class never strips it (verified in `template.py::_encode`), and history turns keep their reflections (`mask_history: false`) — the
  Llama inference pipeline keeps them too.
- Launch (1 GPU): `bash env/sb gpu --gres=gpu:1 -c 5 --mem=30G -t 10:00:00 -J sft_llama31_v3base --export=ALL,SFT_CONFIG=examples/train_lora/llama31_8b_lora_sft_v3base_local.yaml,NPROC_PER_NODE=1,EXTRA_ARGS="gradient_accumulation_steps=16" LLaMA-Factory/scripts/train_qwen3_8b_sdL2_sft.slurm`
  (the launcher is model-agnostic despite its name).

## Per-step / inference-consistent training (v1b, 2026-09-05)
- Data: `self_distill/explode_per_step.py` explodes each trajectory into one sample per assistant step; history assistant turns are think-free.
- Config: `examples/train_lora/qwen3_8b_lora_sft_sdL2_perstep.yaml` (`mask_history: true` ⇒ loss only on the last turn).
- **Framework patch:** LLaMA-Factory's `ReasoningTemplate.encode_multiturn` inserts an empty `<think>\n\n</think>\n\n` before every assistant turn
  that has no thought (also history turns). The official Qwen3 template and our inference path do NOT. `template.py` now skips this for history
  turns when env `LF_NO_HISTORY_EMPTY_COT=1` is set (pass it via `--export`). Without the flag behaviour is unchanged.
- Launch: `bash env/sb gpu --gres=gpu:2 -t 08:00:00 --export=ALL,SFT_CONFIG=<yaml>,EXTRA_ARGS="gradient_accumulation_steps=8",LF_NO_HISTORY_EMPTY_COT=1 LLaMA-Factory/scripts/train_qwen3_8b_sdL2_sft.slurm`

## Other base models (multi-base experiments, 2026-09-11)
The launcher is model-agnostic; a new base model needs a yaml and nothing else when it shares the template.
- **Qwen3-4B** (`exp/qwen3-4b-srft`, ledger §Q4): `examples/train_lora/qwen3_4b_lora_sft_v3base_traj.yaml`, two lines changed vs the 8B yaml.
  The data is reused byte-for-byte because Qwen3-4B and Qwen3-8B share the qwen3 chat template (verified: `chat_template` sha1 `b066ba71c1b5`
  on both snapshots). Use `Qwen/Qwen3-4B` (hybrid thinking), NOT `Qwen3-4B-Instruct-2507`.
  ```bash
  bash env/sb gpu -p general-gpu --gres=gpu:1 -c 5 --mem=30G -t 12:00:00 -J sft_q4b_v3base \
    --export=ALL,SFT_CONFIG=examples/train_lora/qwen3_4b_lora_sft_v3base_traj.yaml,EXTRA_ARGS="gradient_accumulation_steps=16" \
    LLaMA-Factory/scripts/train_qwen3_8b_sdL2_sft.slurm
  ```
- **Llama-3.1-8B-Instruct** (`exp/llama31-8b-srft`, ledger §L): needs a data conversion (different template and tool protocol), see that branch.
