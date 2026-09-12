# Environment (conda envs; skipjack build notes + WashU rebuild)

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

## WashU Compute2 rebuild (2026-09-09) — `env/jobs/build_env.sbatch`
The three envs were rebuilt from `env/freeze/*.txt` into 学长's miniconda (`CONDA_SH` in `env/washu.sh`; envs `agentdojo`, `llamafactory`, `sd_gen`
next to 学长's own envs). One cluster-neutral CPU job per env, chained so no two `conda create` share `CONDA_PKGS_DIRS` at the same time:
```bash
j1=$(bash env/sb cpu --parsable --export=ALL,ENV=agentdojo    -J build_agentdojo    env/jobs/build_env.sbatch)
j2=$(bash env/sb cpu --parsable --dependency=afterany:$j1 --export=ALL,ENV=llamafactory -J build_llamafactory env/jobs/build_env.sbatch)
j3=$(bash env/sb cpu --parsable --dependency=afterany:$j2 --export=ALL,ENV=sd_gen       -J build_sd_gen       env/jobs/build_env.sbatch)
```
Recipe inside the script: `conda create -n <env> python=3.11 pip` → `pip install -r <freeze minus '-e' and 'packaging @ file:' lines>`
(`--extra-index-url https://download.pytorch.org/whl/cu128` for agentdojo/llamafactory so `torch==2.7.0+cu128` resolves; sd_gen takes PyPI's
`torch==2.7.0` = cu126, which is what `vllm==0.9.2` pins) → `pip install --no-deps -e agentdojo|LLaMA-Factory` (no-deps so the pyproject does
not drag in langchain 1.x / langgraph, the 2026-09-07 gotcha) → `pip check` → import check (+ `pytest tests` for agentdojo, `llamafactory-cli version`)
→ `pip freeze > slurm_logs/freeze_<env>_washu.txt`. Caches: `CONDA_PKGS_DIRS=zixuan/tools/conda_pkgs`, `PIP_CACHE_DIR=zixuan/.cache/pip` (from `env/washu.sh`).
Results (2026-09-09): agentdojo 3002999 (11 min) pip check clean + pytest 21 passed; llamafactory 3003000 (13 min); sd_gen 3003001 (9 min, torch 2.7.0+cu126 / vllm 0.9.2 / transformers 4.53.2). `pip freeze` of each env is identical to the skipjack freeze except the editable path and the conda `packaging` line, so the freezes need no WashU variant.
- The 2026-09-07 archive builds (`../SRFT-archive/tools/envs/build_*.sbatch`) were replaced: they pinned only the key packages and their editable
  installs pointed at the old path.
- click stays 8.1.8 (from the freeze): click ≥ 8.2 breaks `benchmark.py --model Qwen/Qwen3-8B-safe-agent` (Enum Choice matches NAMES not VALUES).

## conda env `rlhammer` (RL-Hammer / InjecAgent, built on skipjack 2026-09-12, job 387025, 7 min)
- For `injecAgent-rl-harmmer/rl-injector` (docs/12). Python 3.12 · torch 2.8.0+cu128 · **vllm 0.11.0** (= the DSAI RL-Hammer runs, per their vLLM
  logs) · transformers 4.57.0 · trl 0.23.1 · peft 0.17.1 · accelerate 1.10.1 · openai 1.99.9 · ray 2.58.0; `pip check` clean. Freeze `env/freeze/rlhammer.txt`.
- Build (either cluster): `bash env/sb cpu --export=ALL,ENV=rlhammer -J build_rlhammer env/jobs/build_env.sbatch` (uses the freeze; falls back to
  rl-injector's `requirements.txt` + ray when no freeze exists). Named `rlhammer`, not `rl-hammer`: WashU's shared miniconda already holds 学长's `rl-hammer`.
- No flash-attn (the jobs use `--attn_implementation sdpa`, as the NeurIPS runs did).
