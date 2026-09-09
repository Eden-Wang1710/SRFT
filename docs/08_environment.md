# Environment (conda envs; skipjack build notes)

Pip freezes of all three envs are in `env/freeze/{agentdojo,llamafactory,sd_gen}.txt` (skipjack, 2026-09-09) — use them to rebuild on WashU.
Cluster-specific paths are in `env/<cluster>.sh`; see `10_sync_workflow.md`.

## conda env `agentdojo` (built 2026-09-04)
- Miniforge (no system conda/module): `/weka/scratch/jhu/cxiao13/zwang544/tools/miniforge3`, `conda init bash` applied to `~/.bashrc`.
  Home is only 94 GB, so envs and package caches live on scratch (`CONDA_PKGS_DIRS=tools/conda_pkgs`, `PIP_CACHE_DIR=~/scratch_cxiao13/zwang544/.cache/pip`).
- Activate: `conda activate agentdojo` (7.6 GB). Full freeze: `env/freeze/agentdojo.txt` (copy of `tools/agentdojo_env_freeze.txt`).
- Key versions (pinned from `agentdojo/uv.lock`, which is the lock the eval code was developed against):
  Python 3.11.16 · torch 2.7.0+cu128 · transformers 4.51.3 · tokenizers 0.21.1 · peft 0.17.1 (same as LoRA training) · accelerate 1.6.0 ·
  openai 1.76.2 · pydantic 2.11.4 · langchain 0.3.24 · anthropic 0.50.0 · cohere 5.15.0 · google-genai 1.15.0 · matplotlib 3.10.1 · pytest 8.3.5
- `agentdojo` installed editable (`pip install -e SRFT/agentdojo`).
- Not installed: vLLM (needed only for the 32B path / `util_scripts/run_vllm.sh`), LLaMA-Factory (training needs its own env).
- Checks done: `pip check` clean; `pytest tests` 21 passed; sbatch smoke test on A100 — CUDA visible, bf16 matmul OK.
- Rebuild recipe:
  ```bash
  conda create -n agentdojo python=3.11 pip
  pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.7.0
  pip install -e SRFT/agentdojo
  pip install -r <(grep -vE '^(torch|-e |agentdojo)' env/freeze/agentdojo.txt)   # or the explicit pin list in changelog 2026-09-04
  ```

## Skipjack cluster facts (see 09_cluster_skipjack.md for the full picture and official doc links)
- Slurm account `cxiao13`, QoS `all`. GPU partitions: `a100` (8×A100-80GB nodes, gres `gpu:a100`), `h100` (4×H100), `b200`, `b300`; CPU `med`. 3-day max walltime.
- Driver 595.71 on A100 nodes → cu128 wheels fine (also covers B200/sm_100).
- Login/VS Code shell is itself a slurm job (`vscode` on partition `med`, `SLURM_JOB_ID` set). Consequences:
  - `srun --gres=gpu:1 …` from that shell fails ("Invalid generic resource (gres) specification") because it tries to create a step inside the CPU job.
  - Submit with **`bash env/sb gpu|cpu script.sbatch`** (it runs `env -u SLURM_JOB_ID -u SLURM_JOBID sbatch ...` for you).
  - `/tmp` is node-local: never point `#SBATCH -o` at `/tmp`; use scratch (`env/sb` puts logs in `SRFT/slurm_logs/`).
- No `nvidia-smi` on the login node. Modules available (`module avail`): cuda/11.8–13.0, python/3.11.9, cmake; no conda module.
- Disk: `/weka/scratch/jhu` 9.1 TB shared (7.2 TB free on 2026-09-04); `/weka/home/jhu` 94 GB.
- `HF_HOME` is preset in the login environment to `~/scratch_cxiao13/zwang544/.cache/huggingface`.
