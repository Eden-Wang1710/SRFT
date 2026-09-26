# SRFT: Self-Reflection Fine-Tuning

Anonymous release for the ICLR submission *Self-Reflection Fine-Tuning: Enhancing Agent Security against Prompt
Injection Attacks from Failure Experience*. Code, training data, SR-Agent trajectories and RL-Hammer results.

```
.
├── requirements/                    pip freeze of the three environments (agentdojo / llamafactory / rlhammer)
├── LLaMA-Factory/                   training framework
│   ├── data/toucan_32B_v3_base.json.gz               SRFT training data (3,698 trajectories, 22,339 assistant steps; paper Table 5)
│   ├── data/toucan_32B_v3_base_llama_local.json.gz   same data in the AgentDojo `local` tool format (Llama-3.1-8B)
│   └── examples/train_lora/*.yaml                    training configs of SR-Agent-Llama / -Qwen3-8B / -Qwen3-4B
├── agentdojo/                       AgentDojo benchmark + SR-Agent inference code
│   ├── runs/sr_agent_llama31_8b/    trajectories of SR-Agent-Llama    (Table 1)
│   ├── runs/sr_agent_qwen3_8b/      trajectories of SR-Agent-Qwen3-8B (Tables 2, 3)
│   ├── runs/sr_agent_qwen3_4b/      trajectories of SR-Agent-Qwen3-4B (Table 2)
│   └── eval/compute_attack_stats.py
└── injecAgent-rl-harmmer/rl-injector/   RL-Hammer adaptive attack + SR-Agent targets
    ├── outputs/<target>/<run>/checkpoint-<step>/attack_success_rate.json   results behind Figure 3 / Tables 7-8
    ├── rlhammer_stats.py
    └── jobs/train_attacker.sh, jobs/eval_attacker_ckpts.sh
```

## Training (LLaMA-Factory)

```bash
# env: requirements/llamafactory.txt + pip install -e LLaMA-Factory
cd LLaMA-Factory
gunzip -k data/toucan_32B_v3_base.json.gz data/toucan_32B_v3_base_llama_local.json.gz

# LoRA r64/alpha96 on q,k,v,o; 3 epochs, lr 5e-6, effective batch 16, 8k context (paper Sec. 5.1)
llamafactory-cli train examples/train_lora/llama31_8b_lora_sft_v3base_local.yaml   # SR-Agent-Llama    (Llama-3.1-8B-Instruct)
llamafactory-cli train examples/train_lora/qwen3_8b_lora_sft_v3base_traj.yaml      # SR-Agent-Qwen3-8B (Qwen/Qwen3-8B)
llamafactory-cli train examples/train_lora/qwen3_4b_lora_sft_v3base_traj.yaml      # SR-Agent-Qwen3-4B (Qwen/Qwen3-4B)
```

## AgentDojo (Tables 1-3)

```bash
# env: requirements/agentdojo.txt + pip install -e agentdojo
cd agentdojo

# metrics from the released trajectories: Benign Utility / Utility under Attack / ASR per suite and overall
# (in these files `security: true` = the attack succeeded)
python eval/compute_attack_stats.py sr_agent_llama31_8b/meta-llama_Llama-3.1-8B-Instruct-safe-agent   # Table 1   -> 37.11 / 29.08 / 1.26
python eval/compute_attack_stats.py sr_agent_qwen3_8b/Qwen_Qwen3-8B-safe-agent                        # Table 2/3 ->   -   / 46.68 / 1.05
python eval/compute_attack_stats.py sr_agent_qwen3_4b/Qwen_Qwen3-4B-safe-agent                        # Table 2   -> 50.52 / 49.21 / 0.84

# run the benchmark with your own adapter (one GPU per command; resumes if the run dir exists)
export LLAMA_SR_AGENT_LORA_PATH=/path/to/lora LLAMA_SR_AGENT_SYS_APPEND=0                       # SR-Agent-Llama: no reflection instruction in the system prompt
python scripts/eval_parallel.py --model LLAMA_3_1_8B_SAFE_AGENT --suite banking --attack important_instructions --nproc 3 --logdir runs/<run>   # attacked tasks
python scripts/eval_parallel.py --model LLAMA_3_1_8B_SAFE_AGENT --suite banking --nproc 3 --logdir runs/<run>                                   # benign tasks
export QWEN_SAFE_AGENT_LORA_PATH=/path/to/lora QWEN_SAFE_AGENT_SYS_APPEND=1 QWEN_SAFE_AGENT_ENABLE_THINKING=1 QWEN_SAFE_AGENT_THINK_BUDGET=512   # SR-Agent-Qwen3: think on, budget 512, instruction appended (ENABLE_THINKING=0 -> Table 3 no-think)
python scripts/eval_parallel.py --model QWEN_3_8B_SAFE_AGENT --suite banking --attack important_instructions --nproc 3 --logdir runs/<run>      # QWEN_3_4B_SAFE_AGENT for Qwen3-4B
```

## RL-Hammer (Figure 3, Tables 7-8)

```bash
# env: requirements/rlhammer.txt
cd injecAgent-rl-harmmer/rl-injector

# Figure 3 (a/b/c) + Tables 7/8 from outputs/  ->  outputs/fig_rlhammer.pdf, outputs/rlhammer_summary.md
python rlhammer_stats.py

# train one attacker (4 GPUs: 3 for the GRPO attacker, 1 vLLM server for the target; 20 epochs = 1020 steps, one checkpoint per epoch)
# TARGET: sr_llama | llama_base | sr_qwen   SYS_APPEND: reflection instruction in the target's system prompt (0 for Llama targets, 1 for SR-Agent-Qwen3-8B)
# SEED: 1024 = run 1, 2048 = run 2          TARGET_THINK=0 for the "think off" ablation arm
TARGET=sr_llama   SYS_APPEND=0 SR_LORA=/path/to/lora RUN_NAME=srllama_run1   SEED=1024 bash jobs/train_attacker.sh
TARGET=llama_base SYS_APPEND=0                       RUN_NAME=llamabase_run1 SEED=1024 bash jobs/train_attacker.sh
TARGET=sr_qwen    SYS_APPEND=1 SR_LORA=/path/to/lora RUN_NAME=srqwen_run1    SEED=1024 bash jobs/train_attacker.sh

# evaluate every checkpoint of a run on the 100 InjecAgent test cases (1 GPU; same TARGET / SYS_APPEND / SR_LORA as training)
#   -> outputs/<RUN_NAME>_/<RUN_NAME>_attack_checkpoint-<step>/attack_success_rate.json
TARGET=sr_llama SYS_APPEND=0 SR_LORA=/path/to/lora RUN_NAME=srllama_run1 bash jobs/eval_attacker_ckpts.sh
```
