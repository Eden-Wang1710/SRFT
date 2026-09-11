# AgentDojo evaluation — verified commands

All commands run from `SRFT/agentdojo/` with `conda activate agentdojo`.

## 1. Compute paper metrics for a finished run (VERIFIED 2026-09-04)
```bash
cd SRFT/agentdojo && conda activate agentdojo
python eval/compute_attack_stats.py 3_01_toucan_32B_v2_sft_8k_r64_GA4_qkvo_3epoch_5e-6/Qwen_Qwen3-8B-safe-agent
```
- Argument = path **relative to `agentdojo/runs/`**, and it must go down to the folder that directly contains the suite dirs
  (`banking/ slack/ travel/ workspace/`). Passing `runs/3_01_...` or stopping at the run name fails with "could not find runs/…".
- Writes `eval/attack_stats_<run_with_slashes_as_underscores>.csv` and `.png`.
- Columns: `user_clean_*` = benign utility (files under `user_task_*/none/`), `pure_injection_*` = injection tasks run as user tasks,
  `attacked_total / utility_under_attack_rate / attacked_ASR` (files under `*/important_instructions/*`).
  ASR counts `security == True` (attack succeeded). UA counts `utility == True` among attacked.
- Output for the paper run (matches Table 1 SR-Agent row exactly):

| suite | attacked | Utility under Attack | ASR |
|---|---|---|---|
| banking | 144 | 29.17 | 4.86 |
| slack | 105 | 46.67 | 1.90 |
| travel | 140 | 22.14 | 0.00 |
| workspace | 560 | 57.32 | 0.17 |
| ALL | 949 | 46.68 | 1.05 |

  Benign Utility column is empty for this run dir (no `none/` runs); the paper's 51.55 comes from
  `eval/attack_stats_qwen3_8b_toucan_lora_think_sys_append_no_attack_Qwen_Qwen3-8B-safe-agent.csv`.

Other helper scripts in `eval/` (same `run_name` convention; not yet re-verified on skipjack):
- `compute_utility_safety_stats.py` — utility=True & security=False rates
- `compute_assistant_think_tokens.py`, `compute_assistant_all_content_tokens.py` — Table 3 token stats
- `list_unfinished_runs.py <run_name>` — compare against a reference run to find missing task files
- `../scripts/count_follows_injection_true.py`, `find_security_regressions.py`, `summarize_utility_regressions.py` — diagnostics

## 2. Launch the SR-Agent evaluation (paper setting) — not yet re-run on skipjack
Entry script: `agentdojo/run_safe_agent_sr_agent_injection_chunks.sh`. It loops over suites and calls
`src/agentdojo/scripts/run_workspace_injection_chunks.py --model QWEN_3_8B_SAFE_AGENT --attack important_instructions --chunks N`,
which shards user tasks across `CUDA_VISIBLE_DEVICES` (one process per chunk/GPU).

Env vars it honors (defaults in the script):
| var | meaning | paper value |
|---|---|---|
| `QWEN_SAFE_AGENT_LORA_PATH` | LoRA adapter dir loaded via PEFT on top of Qwen/Qwen3-8B | `../LLaMA-Factory/saves/qwen3-8b/lora/toucan_32B_v2_sft_8k_r64_GA4_qkvo_3epoch_5e-6` |
| `QWEN_SAFE_AGENT_SYS_APPEND` | append the SR-Agent system prompt (Fig. 5) | 1 |
| `QWEN_SAFE_AGENT_ENABLE_THINKING` | think mode (Table 2 ablation sets 0) | 1 |
| `QWEN_SAFE_AGENT_NO_THINK_MAX_NEW_TOKENS` | max new tokens when thinking off | 1024 |
| `RUN_LOGDIR` | output run dir (`runs/<name>`) | e.g. `runs/3_01_toucan_32B_v2_sft_8k_r64_GA4_qkvo_3epoch_5e-6` |
| `CHUNK_LOG_ROOT`, `CHUNKS`, `SUITES`, `ATTACK` | logs, #shards, suites, attack | 4; `travel slack banking workspace`; `important_instructions` |
| `HF_HOME` | HF cache (script defaults to `.cache/huggingface` inside agentdojo; login env already sets a scratch path) | — |

Generation settings are hard-coded in `src/agentdojo/agent_pipeline/llms/qwen_8b_think_llm_safe_agent.py::chat_completion_request`:
temperature 0.6, top_p 0.95, top_k 20, min_p 0, thinking_budget 512, second-pass max_new_tokens 512.
Model registry: `src/agentdojo/models.py` (`QWEN_3_8B_SAFE_AGENT = "Qwen/Qwen3-8B-safe-agent"` → pipeline `hf_qwen_8b_safe_agent`).

Benign-utility run = same script with `ATTACK=none` (check `run_workspace_injection_chunks.py --help` for the exact flag; the old
no-attack runs were named `*_no_attack`).

Example sbatch skeleton (4×A100, adjust; submit with `bash env/sb gpu --gres=gpu:4 -c 16 --mem=128G -t 2-00:00:00 <file>`):
```bash
#!/bin/bash
#SBATCH -J sr_agent_eval
SRFT_ROOT="${SRFT_ROOT:-$(d="${SLURM_SUBMIT_DIR:-$PWD}"; while [ ! -f "$d/env/select.sh" ] && [ "$d" != / ]; do d=$(dirname "$d"); done; echo "$d")}"
source "$SRFT_ROOT/env/select.sh"; source "$CONDA_SH" && conda activate agentdojo
cd "$SRFT_ROOT/agentdojo"
export QWEN_SAFE_AGENT_LORA_PATH=../LLaMA-Factory/saves/qwen3-8b/lora/toucan_32B_v2_sft_8k_r64_GA4_qkvo_3epoch_5e-6
export RUN_LOGDIR=runs/<new_run_name> CHUNK_LOG_ROOT=logs/<new_run_name> CHUNKS=4
bash run_safe_agent_sr_agent_injection_chunks.sh
```

## 3. Generic upstream benchmark CLI (API models / vLLM)
```bash
python -m agentdojo.scripts.benchmark --model <name> --attack important_instructions --logdir runs/<name>
```
`util_scripts/run_vllm.sh <model_path> <tool_parser> [port]` serves a model with vLLM and benchmarks it as `vllm_parsed`
(uses `uv run --with vllm`; vLLM is NOT installed in the conda env).

## 4. Parallel eval recipe used on skipjack (2026-09-05, first used for the sdL2 LoRA)
`bash agentdojo/scripts/submit_eval_sdL2.sh <RUN_NAME> <abs LoRA dir>` (any cwd; account/partitions/log path from `env/<cluster>.sh`, override with
`PARTS=... NPROC=... WS_NPROC=...`; `SRFT_SBATCH_FLAGS=--test-only` for a dry run) submits 10 single-GPU jobs + 1 stats job:
- `scripts/eval_sdL2.sbatch` (1 GPU, `$GPU_PARTITIONS`, 4 h, env agentdojo, `QWEN_SAFE_AGENT_SYS_APPEND=1`, `ENABLE_THINKING=1`, `HF_HUB_OFFLINE=1`)
  → `scripts/eval_parallel.py --suite S [--attack important_instructions --injection-tasks a,b,…] --nproc 3 --logdir runs/<RUN_NAME>`
  which launches 3 `benchmark.py` processes on the SAME GPU (HF generate is batch-1; one 8B bf16 process ≈ 19 GB, so 3 fit on 80 GB;
  do not use L40S). Attack jobs split the suite's injection tasks (workspace 7+7, travel 4+3, slack 5, banking 9); the benign job runs the
  4 suites sequentially splitting user tasks. All write to the same logdir, so one `compute_attack_stats.py <RUN_NAME>/Qwen_Qwen3-8B-safe-agent`
  yields benign utility + UA + ASR together (the stats job does this automatically, `afterany` of the 7).
- `benchmark.py --model` takes the enum VALUE (`Qwen/Qwen3-8B-safe-agent`); the launcher maps the enum NAME for you.
- Resume = resubmit the same command; benchmark.py skips tasks whose result JSON exists (no `--force-rerun`).
- Expected wall time per job 1–2 h (paper-run durations: banking 53 s, slack 82 s, travel 154 s, workspace 58 s per trajectory).
- **Measured 2026-09-05:** 3 procs on one H100 give only ≈1.15× the throughput of 1 proc (GPU is ~100 % busy with a single HF generate
  process; 1 proc ≈ 64 % util). So parallelism must come from MORE single-GPU JOBS, not more procs per GPU. Final split used for v1:
  workspace 4 jobs (inj 0-3 / 4-6 / 7-10 / 11-13), travel 3 (0-2 / 3-4 / 5,6), slack 1, banking 1, benign 1 — 10 GPU jobs, `-t 04:00:00`.
  Observed rates (3 procs): banking ≈1.3 traj/min, slack ≈1.1 traj/min. Real speed-up needs vLLM (batching) — see self_distill_plan §next.
- 2026-09-06: default eval submission now `-p a100,h100,h200,l40s` with 2 procs/GPU (≈35 GB, fits L40S). Override with `PARTS=... NPROC=...`.
- 2026-09-08: workspace shards use `NPROC=1` (L40S OOM with 2 procs on long trajectories); stats jobs need `--comment=accept_cost`; if an
  eval job FAILs with exit 1, check `logs/eval_parallel/<run>/<suite>_<attack>_<jobid>/proc*.log` for `OutOfMemoryError` and resubmit the shard.
- 2026-09-09: slurm stdout of eval/stats jobs moved from `agentdojo/logs/eval_sdL2/` to `SRFT/slurm_logs/`.

## Resume did NOT work before 2026-09-09 (fixed)
`benchmark.py::load_task_results` only mapped two hard-coded model names from "Qwen/…" to "Qwen_…"; for `Qwen/Qwen3-8B-safe-agent` the lookup
path never existed, so every task looked unrun and any resubmitted shard **re-ran and overwrote** all its trajectories (re-sampled at T=0.6).
Consequences: the v2-e1 "resume" of 3 workspace shards re-sampled 433 trajectories (workspace UA 52.0 → 43.2 for the same checkpoint);
the v1 ws_a/ws_b partials were also overwritten (same model, harmless). Fixed by `pipeline_name.replace("/", "_")` in the loader; verified
that attacked and benign results now load. From now on resubmitting a shard truly resumes (proc logs show "Skipping task …").

## Run-to-run variance (measured by accident, 2026-09-08)
Same checkpoint, same 433 workspace trajectories, two independent samples at temperature 0.6: UA 52.0 % vs 43.2 %. Trajectories are
clustered by user task (40 per suite), so the effective sample is small. **Single-run differences of ≤ 3–4 points overall (or ≤ 10 points
on one suite) are not evidence.** For any claim in the paper, evaluate ≥ 3 seeds (or fix the seed and temperature) and report the spread.

## Resource requests (2026-09-10)
`MaxMemPerCPU=6000` turns `--mem=64G` into an 11-CPU request, which does not fit the 5–8 free CPUs typically left on busy GPU nodes even when
GPUs are free. `eval_sdL2.sbatch` now defaults to `-c 5 --mem=30G`; `submit_eval_sdL2.sh` passes `-c 7 --mem=42G` for 2-proc jobs. For
already-pending jobs: `scontrol update JobId=<id> MinMemoryNode=30000 CPUsPerTask=5 MinCPUsNode=5`.

## Inference knobs (2026-09-10)
`SYS_APPEND=0|1` (default 1) and `THINK_BUDGET=<tokens>` (default 512) can be passed to `submit_eval_sdL2.sh`; they reach the LLM as
`QWEN_SAFE_AGENT_SYS_APPEND` / `QWEN_SAFE_AGENT_THINK_BUDGET`. `SUITES_ONLY="travel slack banking"` submits only those shards (benign job
restricted to the same suites). Paper protocol = append on, 512. Example:
```
SUITES_ONLY="travel slack banking" SYS_APPEND=0 bash scripts/submit_eval_sdL2.sh <run>_noappend <ckpt>
SUITES_ONLY="travel slack banking" THINK_BUDGET=1024 bash scripts/submit_eval_sdL2.sh <run>_think1024 <ckpt>
```

## SR-Agent-Llama / base Llama (2026-09-11, VERIFIED by smoke 376718)
`MODEL=<ModelsEnum name>` selects the pipeline in `submit_eval_sdL2.sh` / `eval_sdL2.sbatch` (default `QWEN_3_8B_SAFE_AGENT`, unchanged).
`MODEL=LLAMA_3_1_8B_SAFE_AGENT` → provider `hf_llama_sr_agent` (`llms/llama_sr_agent_llm.py` + `llms/llama_local_prompt.py`), run subdir
`meta-llama_Llama-3.1-8B-Instruct-safe-agent` (the stats job uses it automatically). Same knobs: `LORA_PATH` (non-existent → base Llama) and
`SYS_APPEND` (→ `LLAMA_SR_AGENT_SYS_APPEND`); `THINK_BUDGET` is ignored (no think mode). Extra env for the LLM: `LLAMA_SR_AGENT_MAX_NEW_TOKENS`
(1536), `_TEMPERATURE` / `_TOP_P` (0.6 / 0.9), `_BASE_MODEL`, `_PRINT_FIRST_INPUTS` (2 raw prompts per process are printed to the proc log).
```
MODEL=LLAMA_3_1_8B_SAFE_AGENT SYS_APPEND=0 bash agentdojo/scripts/submit_eval_sdL2.sh llama31_base_noappend /nonexistent_base_model_fallback
MODEL=LLAMA_3_1_8B_SAFE_AGENT SYS_APPEND=1 bash agentdojo/scripts/submit_eval_sdL2.sh llama31_v3base_local_3epoch $PWD/LLaMA-Factory/saves/llama31-8b/lora/<run>
```
Format: AgentDojo `local` prompt (`<function=name>{json}</function>`, reasoning allowed before the call), tool outputs as raw `ipython` turns,
reflection stored as a thinking block (not scored) and re-inserted into the history as `<think>…</think>` text — token-identical to training
(`multibase/check_llama_render.py`). Base Llama in the smoke: ~1 s per step on A100, writes reasoning text then a call, loops through
`get_most_recent_transactions(n=1,2,3…)`; result JSONs normal.
**Llama OOMs on L40S (2026-09-11):** even 1-proc workspace shards (and 2-proc benign) ran out of the 44 GB of an L40S (7 of 40 Llama jobs).
Submit Llama evals with `PARTS=a100,h100,h200`; failed shards resume by resubmitting the same shard (reruns also set
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` via `--export`).

## sbatch stderr noise (2026-09-11)
skipjack's `cli_filter` now prints `failed to load /etc/slurm/rates.lua … cost preview disabled` on stderr for every sbatch. The submitter used to
capture `2>&1`, so the warning ended up in the dependency id list and the stats job was silently not submitted (base-Llama runs; stats resubmitted
by hand). Fixed: `sub()` and the stats line now capture stdout only and send filtered stderr to the terminal.
