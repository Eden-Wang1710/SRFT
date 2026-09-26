# Self-Reflection Fine-Tuning (SRFT) — code, data and results

Anonymous release accompanying the ICLR submission *Self-Reflection Fine-Tuning: Enhancing Agent Security against Prompt
Injection Attacks from Failure Experience*.

SRFT fine-tunes a tool-using agent on its own failures: expert trajectories are corrupted with executable prompt
injections, the reference agent is rolled out to collect candidate (often hijacked) actions, an expert LLM writes a
three-part self-reflection that contrasts the expert action with the sampled ones, and the agent is fine-tuned to
produce that reflection followed by the expert action. The resulting models are called **SR-Agent**
(SR-Agent-Llama on Llama-3.1-8B-Instruct, SR-Agent-Qwen3-8B / -Qwen3-4B on Qwen3).

## 1. Repository layout

```
.
├── README.md                      this file
├── requirements/                  pinned environments (one per component, see §6)
├── LLaMA-Factory/                 training framework (LLaMA-Factory) + SRFT training data + training configs
│   ├── data/toucan_32B_v3_base.json.gz              SRFT training set, Qwen3 tool format   (3,698 trajectories, 22,339 assistant steps; paper Table 5)
│   ├── data/toucan_32B_v3_base_llama_local.json.gz  the same data rendered in the AgentDojo `local` tool format used for Llama-3.1-8B
│   ├── data/dataset_info.json                       dataset registration for the two files
│   └── examples/train_lora/{llama31_8b_lora_sft_v3base_local,qwen3_8b_lora_sft_v3base_traj,qwen3_4b_lora_sft_v3base_traj}.yaml
├── agentdojo/                     AgentDojo benchmark (fork of ethz-spylab/agentdojo v1.2.1) + SR-Agent pipeline elements + trajectories
│   ├── src/agentdojo/agent_pipeline/llms/llama_sr_agent_llm.py, llama_local_prompt.py   SR-Agent-Llama inference (AgentDojo `local` format)
│   ├── src/agentdojo/agent_pipeline/llms/qwen_8b_think_llm_safe_agent.py               SR-Agent-Qwen3 inference (think mode, 512-token budget)
│   ├── runs/sr_agent_llama31_8b/    1,081 trajectories of SR-Agent-Llama   (Table 1)
│   ├── runs/sr_agent_qwen3_8b/        949 attacked trajectories of SR-Agent-Qwen3-8B (Tables 2, 3)
│   ├── runs/sr_agent_qwen3_4b/      1,081 trajectories of SR-Agent-Qwen3-4B (Table 2)
│   ├── eval/compute_attack_stats.py   Benign Utility / Utility under Attack / ASR for one run directory
│   └── scripts/eval_parallel.py       launcher used to produce the trajectories
└── injecAgent-rl-harmmer/rl-injector/   RL-Hammer adaptive attack (fork of facebookresearch/rl-injector, InjecAgent branch) + our targets + results
    ├── llama_target.py, config.py, reward_func.py, train.py, injecagent_eval.py   attacker training / evaluation (SR-Agent targets added)
    ├── data/InjecAgent/               the 310 / 100 / 100 train / eval / test split used throughout
    ├── jobs/train_attacker.sh         train one attacker against one target (§5)
    ├── jobs/eval_attacker_ckpts.sh    evaluate every attacker checkpoint on the 100 test cases
    ├── outputs/<target>/<run>/checkpoint-<step>/   per-checkpoint ASR (+ per-case outputs where available) behind Figure 3 / Tables 7-8
    └── rlhammer_stats.py              regenerates Figure 3 and Tables 7-8 from outputs/
```

## 2. Model checkpoints

All SR-Agent models are LoRA adapters (rank 64, alpha 96, q/k/v/o projections, 3 epochs, lr 5e-6, cosine schedule,
10 % warm-up, effective batch 16, 8k context, bf16) on top of the public base model.

| Model | Base model | Training data | Training config | Adapter |
|---|---|---|---|---|
| SR-Agent-Llama | `meta-llama/Llama-3.1-8B-Instruct` | `toucan_32B_v3_base_llama_local` | `examples/train_lora/llama31_8b_lora_sft_v3base_local.yaml` | *[link to be added]* |
| SR-Agent-Qwen3-8B | `Qwen/Qwen3-8B` | `toucan_32B_v3_base` | `examples/train_lora/qwen3_8b_lora_sft_v3base_traj.yaml` | *[link to be added]* |
| SR-Agent-Qwen3-4B | `Qwen/Qwen3-4B` (hybrid-thinking release) | `toucan_32B_v3_base` | `examples/train_lora/qwen3_4b_lora_sft_v3base_traj.yaml` | *[link to be added]* |

To train an adapter yourself (one 80 GB GPU is enough; the paper runs used 4 GPUs with `per_device_train_batch_size 1`
and `gradient_accumulation_steps 4`):

```bash
cd LLaMA-Factory
gunzip -k data/toucan_32B_v3_base.json.gz data/toucan_32B_v3_base_llama_local.json.gz
llamafactory-cli train examples/train_lora/llama31_8b_lora_sft_v3base_local.yaml     # -> saves/llama31-8b/lora/...
```

## 3. Training data

`LLaMA-Factory/data/toucan_32B_v3_base.json.gz` (18 MB gzipped, 103 MB unpacked) is a list of 3,698 ShareGPT-style
trajectories built from five TOUCAN MCP platforms (hotel booking, e-mail, Windows command line, Markdown downloader,
Minecraft wiki). Each record has

- `system`: the platform system prompt with the tool specifications;
- `conversations`: alternating `human` / `function_call` / `observation` / `gpt` turns. Every assistant turn
  (`function_call` or `gpt`, 22,339 in total) starts with the self-reflection in `<think> … </think>` followed by the
  expert tool call or final answer. 37.8 % of the `observation` turns carry an injected instruction (trigger + task);
- `meta`: per-step ground truth — for every assistant step, whether an injection is already in the context and, if so,
  its trigger, task text, target tool call and insertion position (`meta.steps`).

`toucan_32B_v3_base_llama_local.json.gz` contains the same trajectories with tool specifications moved into the system
prompt, tool calls written as `<function=name>{json}</function>` and tool outputs as plain observations, i.e. the
AgentDojo `local` format that SR-Agent-Llama is trained and evaluated in. The prompt templates used to generate
injections (Template 1) and reflections (Template 2) are given in Appendix B of the paper.

## 4. AgentDojo results (Tables 1-3)

### 4.1 Recompute the numbers from the released trajectories

```bash
cd agentdojo
python eval/compute_attack_stats.py sr_agent_llama31_8b/meta-llama_Llama-3.1-8B-Instruct-safe-agent   # Table 1, SR-Agent-Llama
python eval/compute_attack_stats.py sr_agent_qwen3_8b/Qwen_Qwen3-8B-safe-agent                        # Table 2 / 3, SR-Agent-Qwen3-8B
python eval/compute_attack_stats.py sr_agent_qwen3_4b/Qwen_Qwen3-4B-safe-agent                        # Table 2, SR-Agent-Qwen3-4B
```

Each command prints per-suite and overall metrics and writes `eval/attack_stats_<run>.csv` (already included).
`user_clean_completing_rate` = Benign Utility (97 user tasks, files under `user_task_*/none/`),
`utility_under_attack_rate` and `attacked_ASR` are computed over the 949 attacked tasks
(`user_task_*/important_instructions/injection_task_*.json`, banking 144 / slack 105 / travel 140 / workspace 560).
**Note on the convention used in these files: `security: true` means the attack succeeded** (the injected task was
carried out), which is the opposite of upstream AgentDojo's field semantics.

Expected overall rows (Benign / Utility under Attack / ASR, %):

| Run | Benign | UA | ASR |
|---|---|---|---|
| `sr_agent_llama31_8b` | 37.11 | 29.08 | 1.26 |
| `sr_agent_qwen3_8b` | — | 46.68 | 1.05 |
| `sr_agent_qwen3_4b` | 50.52 | 49.21 | 0.84 |

Every trajectory file contains the full message list (system prompt, user task, tool calls, tool outputs with the
injection, the model's reflection as a thinking block, and the final answer), the injection texts, and the `utility` /
`security` verdicts of the benchmark.

### 4.2 Run the evaluation yourself

Install `requirements/agentdojo.txt` and `pip install -e agentdojo`. The SR-Agent models are registered in
`src/agentdojo/models.py` as `LLAMA_3_1_8B_SAFE_AGENT`, `QWEN_3_8B_SAFE_AGENT` and `QWEN_3_4B_SAFE_AGENT`; the adapter and
inference settings are passed through environment variables.

```bash
cd agentdojo
# SR-Agent-Llama (paper setting: no reflection instruction in the system prompt)
export LLAMA_SR_AGENT_LORA_PATH=/path/to/sr-agent-llama-lora LLAMA_SR_AGENT_SYS_APPEND=0
for s in banking slack travel workspace; do
  python scripts/eval_parallel.py --model LLAMA_3_1_8B_SAFE_AGENT --suite $s --attack important_instructions --nproc 3 --logdir runs/my_sr_llama
  python scripts/eval_parallel.py --model LLAMA_3_1_8B_SAFE_AGENT --suite $s --nproc 3 --logdir runs/my_sr_llama     # benign utility
done
python eval/compute_attack_stats.py my_sr_llama/meta-llama_Llama-3.1-8B-Instruct-safe-agent

# SR-Agent-Qwen3-8B / -4B (paper setting: think mode with a 512-token budget, reflection instruction appended; Table 3 "no-think" = QWEN_SAFE_AGENT_ENABLE_THINKING=0)
export QWEN_SAFE_AGENT_LORA_PATH=/path/to/sr-agent-qwen3-8b-lora QWEN_SAFE_AGENT_SYS_APPEND=1 QWEN_SAFE_AGENT_ENABLE_THINKING=1 QWEN_SAFE_AGENT_THINK_BUDGET=512
python scripts/eval_parallel.py --model QWEN_3_8B_SAFE_AGENT --suite banking --attack important_instructions --nproc 3 --logdir runs/my_sr_qwen
```

`eval_parallel.py` runs several `benchmark.py` processes on one GPU and resumes if the run directory already contains
results. Generation settings (Llama: T 0.6 / top-p 0.9, 1,536 new tokens; Qwen: T 0.6 / top-p 0.95 / top-k 20, think
budget 512 with the forced-transition prompt of Appendix C.1) are the defaults of the two LLM elements.

## 5. RL-Hammer adaptive attack (Figure 3, Tables 7-8)

### 5.1 Released results

`injecAgent-rl-harmmer/rl-injector/outputs/<target>/<run>/checkpoint-<step>/` holds, for every attacker checkpoint
(one per epoch, 51 steps = 1 epoch, 20 epochs), `attack_success_rate.json` (ASR on the 100 test cases) and, where the
per-case outputs were kept, `test_cases.json` (adversarial prompt, target output, judge verdict for each case).

| `outputs/` directory | Target | Runs | Attacker seeds | Reflection instruction | Per-case outputs |
|---|---|---|---|---|---|
| `llama31_8b_base/run{1,2}` | Llama-3.1-8B-Instruct (undefended) | 2 | 1024 / 2048 | off | yes |
| `meta_secalign_8b/run{1,2}` | Meta-SecAlign-8B (ReAct prompt, injection in the `input` role) | 2 | 1024 / 2048 | off | ASR only |
| `sr_agent_llama31_8b/run{1,2}` | SR-Agent-Llama | 2 | 1024 / 2048 | off | yes |
| `qwen3_8b_base/run{1,2}` | Qwen3-8B (undefended, think on) | 2 | 1024 / 2048 | off | ASR only (+ epoch 19 of run 1) |
| `sr_agent_qwen3_8b/run{1,2}` | SR-Agent-Qwen3-8B (think on) | 2 | 1024 / 2048 | on | ASR only (+ epoch 20 of run 1) |
| `ablation_qwen3_8b/wo_failure_experience` | Qwen3-8B fine-tuned without the failure-experience paragraph (think on) | 1 | 1024 | on | yes |
| `ablation_qwen3_8b/wo_reflection_think_on` | Qwen3-8B fine-tuned without any reflection, think on | 1 | 1024 | off | yes |
| `ablation_qwen3_8b/wo_reflection_think_off` | same checkpoint, think off | 1 | 1024 | off | yes |

```bash
cd injecAgent-rl-harmmer/rl-injector
python rlhammer_stats.py          # -> outputs/fig_rlhammer.{pdf,png} (Figure 3 a/b/c) and outputs/rlhammer_summary.md (Tables 7 and 8)
```

The script averages the two attacker runs of a target on the checkpoints both runs have (shaded band = run-to-run
min/max) and reports Final (last epoch), Last-5 (mean of the last five epochs) and Peak (epoch). Expected output:

| Target | Final | per run | Last-5 | Peak (epoch) |
|---|---|---|---|---|
| Llama-3.1-8B-Instruct | 98.5 | 98 / 99 | 97.9 | 99.0 (16) |
| Meta-SecAlign-8B | 69.5 | 80 / 59 | 62.1 | 69.5 (20) |
| SR-Agent-Llama | 17.5 | 32 / 3 | 19.8 | 26.5 (13) |
| Qwen3-8B | 59.5 | 54 / 65 | 62.8 | 68.0 (16) |
| SR-Agent-Qwen3-8B | 17.0 | 32 / 2 | 16.8 | 21.5 (19) |
| w/o failure experience | 65.0 | 65 | 65.0 | 67.0 (8) |
| w/o reflection, think on | 60.0 | 60 | 54.8 | 69.0 (12) |
| w/o reflection, think off | 60.0 | 60 | 57.0 | 60.0 (20) |

### 5.2 Train and evaluate an attacker

Install `requirements/rlhammer.txt` (vLLM 0.11.0, TRL 0.23.1). One run needs 4 GPUs (3 for the GRPO attacker, 1 for a
vLLM server that hosts the target) and takes 6-9 h on 80 GB GPUs; the per-checkpoint evaluation needs 1 GPU (~1.5 h).

```bash
cd injecAgent-rl-harmmer/rl-injector

# SR-Agent-Llama as target (Figure 3a); run 2 = SEED=2048
TARGET=sr_llama SYS_APPEND=0 SR_LORA=/path/to/sr-agent-llama-lora RUN_NAME=srllama_run1 SEED=1024 bash jobs/train_attacker.sh
TARGET=sr_llama SYS_APPEND=0 SR_LORA=/path/to/sr-agent-llama-lora RUN_NAME=srllama_run1           bash jobs/eval_attacker_ckpts.sh

# undefended Llama-3.1-8B-Instruct
TARGET=llama_base SYS_APPEND=0 RUN_NAME=llamabase_run1 SEED=1024 bash jobs/train_attacker.sh
TARGET=llama_base SYS_APPEND=0 RUN_NAME=llamabase_run1           bash jobs/eval_attacker_ckpts.sh

# SR-Agent-Qwen3-8B (Figure 3b; reflection instruction on, think on); TARGET_THINK=0 for the "think off" ablation arm
TARGET=sr_qwen SYS_APPEND=1 SR_LORA=/path/to/sr-agent-qwen3-8b-lora RUN_NAME=srqwen_run1 SEED=1024 bash jobs/train_attacker.sh
TARGET=sr_qwen SYS_APPEND=1 SR_LORA=/path/to/sr-agent-qwen3-8b-lora RUN_NAME=srqwen_run1           bash jobs/eval_attacker_ckpts.sh
```

The recipe is fixed inside `jobs/train_attacker.sh` and is the RL-Hammer one (attacker Llama-3.1-8B-Instruct + LoRA
r64/α32 on 7 modules, GRPO without KL, lr 1e-5, 8 generations, effective batch 48 completions/step, 20 epochs over the
310 training cases, model-only checkpoint every epoch). The reward queries the main target (weight 3) and the weaker
Llama-3.1-8B-Instruct ReAct target (weight 1) through the vLLM server; target replies are sampled at T 0.6 / top-p 0.95 /
top-k 20 with 512 tokens during training and 1,024 during evaluation. The judge (`injecagent_output_parsing.py`) reads
the first tool call in the target output: the attacker's tool counts as a success, the user's tool or an invalid output
as a failure. The Meta-SecAlign-8B target is built with `merge_meta_secalign.py` and attacked with the original
RL-Hammer ReAct prompt (`--safe_agent_mode False`).

`eval_attacker_ckpts.sh` writes `outputs/<RUN_NAME>_/<RUN_NAME>_attack_checkpoint-<step>/` with the same two files as
the released results; copy them into `outputs/<target>/<run>/checkpoint-<step>/` to include a new run in
`rlhammer_stats.py`.

## 6. Environments

`requirements/agentdojo.txt`, `requirements/llamafactory.txt` and `requirements/rlhammer.txt` are the exact `pip freeze`
of the three conda environments (Python 3.12) used for the AgentDojo evaluations, the LoRA training and the RL-Hammer
experiments respectively. Base models are downloaded from the Hugging Face Hub on first use
(`meta-llama/Llama-3.1-8B-Instruct` is gated; `Qwen/Qwen3-8B`, `Qwen/Qwen3-4B`, `facebook/Meta-SecAlign-8B`).

## License

AgentDojo (MIT), LLaMA-Factory (Apache 2.0) and rl-injector (see `injecAgent-rl-harmmer/rl-injector/LICENSE`) keep
their original licenses; our additions, data and results are released under the same terms as the respective component.
