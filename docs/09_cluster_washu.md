# WashU RIS Compute2 cluster: official docs + how we submit jobs

**Cluster-specific values live in `env/washu.sh`** (conda.sh path, HF cache, account `compute2-zhang.ning`, partitions). Every job script loads
them through `env/select.sh`; submit with `bash env/sb gpu|cpu [flags] <script>` (see `docs/10_sync_workflow.md` §Running jobs). Logs land in
`SRFT/slurm_logs/<jobname>_<jobid>.out`. This file is the *explanation* behind those values plus the queue/gotcha log.
Sources: the pre-git WashU copy of this workspace (`../SRFT-archive/docs/09_cluster.md`, `08_environment.md`, `tools/` next to it, 2026-09-06/07)
and what was re-verified on 2026-09-09 when `main` was cloned here.

## Where we are
- Login: `ssh li.hao@c2-login-00{1,2,3}.ris.wustl.edu` (WashU key + Duo 2FA). Claude Code runs **directly on the login node** `c2-login-002`
  (NOT inside a slurm job — no `SLURM_JOB_ID`), so plain `sbatch` / `srun --pty` / `salloc` all work; `env/washu.sh`'s `srft_sbatch` is a plain `sbatch`.
  Login nodes are for editing/moving files only: **6 GB RAM per user**, no `nvidia-smi` ("couldn't communicate with the NVIDIA driver").
- Repo: `/storage3/fs1/zhang.ning/Active/hao/zixuan/SRFT` (学长's project storage; `~/storage` → `/storage3/fs1/zhang.ning/Active/hao`).
  The old hand-synced copy is `../SRFT-archive` (read-only reference: its docs, `tools/envs/*.sbatch`, `tools/jobs/*`).
- Account: `-A compute2-zhang.ning` on **every** job — it is the only association of `li.hao` (`sacctmgr show assoc user=li.hao`); QoS
  `compute2-zhang.ning` has no per-user caps. Nothing else is mandatory (no `--comment`, no `--qos`) → `SBATCH_EXTRA=""`.
- 学长's reference lines: `/home/li.hao/apply_node.sh`.
- git: the global `user.name/email` on this account are 学长's; the repo has a local `git config user.name "Zixuan Wang"` / `user.email` so commits
  from here carry the same author as skipjack. GitHub ssh key `~/.ssh/id_ed25519_github` (`~/.ssh/config` Host github.com), remote `git@github.com:Eden-Wang1710/SRFT.git`.

## Official documentation (Confluence, anonymous read works for WebFetch; pages are huge → curl + grep)
Site: https://washu.atlassian.net/wiki/spaces/RUD/overview (RIS User Documentation; docs.ris.wustl.edu redirects here)
- Compute2 quickstart (login, `-A compute2-<pi>`, srun/sbatch examples): https://washu.atlassian.net/wiki/spaces/RUD/pages/3307929753/Compute2+Cluster
- Batch jobs: https://washu.atlassian.net/wiki/spaces/RUD/pages/1737785530/Batch+Jobs+sbatch ; interactive: https://washu.atlassian.net/wiki/spaces/RUD/pages/1737523345
- Compute2 guidelines / policies: https://washu.atlassian.net/wiki/spaces/RUD/pages/2140667974 , https://washu.atlassian.net/wiki/spaces/RUD/pages/2304737507
- Anaconda via Lmod: https://washu.atlassian.net/wiki/spaces/RUD/pages/2342092866 ; containers (Pyxis/Enroot): https://washu.atlassian.net/wiki/spaces/RUD/pages/2341994642
- Storage platforms: https://washu.atlassian.net/wiki/spaces/RUD/pages/1795325985 ; Open OnDemand: https://c2-ood.ris.wustl.edu/
- The quickstart's partition table is stale (`general`/`gpu`); trust `sinfo` / `scontrol show partition` below.

## Hardware / partitions (sinfo + scontrol, 2026-09-06, re-checked 2026-09-09)
| Partition | Nodes | GPUs | GPU model | MaxTime | Default | Notes |
|---|---|---|---|---|---|---|
| `general-gpu` | c2-gpu-[003-016] (14) | 4/node, 56 | **H100 80GB HBM3** (probe 2987703) | 15 d | 8 h | main queue, PriorityTier 7000, PreemptMode OFF |
| `general-preempt-gpu` | c2-gpu-[017-024] (8) | 4 or 8/node, 40 | **A100 80GB PCIe** (probe 2987704) | 15 d | 8 h | PriorityTier 1, **PreemptMode=REQUEUE** (`JobRequeue=1` cluster-wide → preempted jobs are requeued automatically; make them restart-safe) |
| `general-short` | c2-gpu-[001-002] + all CPU nodes | 8 (2 × 4) | H100 80GB HBM3 (probe 2987702) | **30 min** | 30 min | PriorityTier 8000, starts within seconds → smoke tests / stats |
| `general-interactive` | same nodes as general-short | 8 | H100 80GB | 5 d | 8 h | `srun --pty`, PriorityTier 9000 |
| `general-cpu` | c2-node-[001-071,073-078] (77) | – | – | 15 d | 8 h | 64 cores / ~900 GB; env builds, downloads, stats; starts immediately |
| `general-bigmem` | 2 | – | – | 28 d | | |
- gres is plain `gpu:N` (no `gpu:h100:N` type strings). `DefMemPerCPU=4000`, no `MaxMemPerCPU` on the GPU partitions (unlike skipjack) —
  `-c 8 --mem=64G` is accepted as is.
- GPU nodes: 64 cores (48–96 on preempt nodes), ~900 GB RAM (1.4–1.8 TB on some preempt nodes), `/tmp` node-local NVMe (1.8–7 TB, wiped).
  Driver 580.105.08 → cu128 wheels OK. `module avail` has only admin modules (gcc/14, python3, cuda-dcgm, slurm) — CUDA comes from the pip wheels.
- Compute nodes have internet (pypi / huggingface / pytorch.org / github verified) → env builds and HF downloads run as `general-cpu` jobs.

## Storage
| Path | What | Size |
|---|---|---|
| `/storage3/fs1/zhang.ning/Active/hao/` (= `~/storage`) | 学长's project storage; SRFT root `zixuan/SRFT`, our caches `zixuan/.cache/{huggingface,pip}`, conda pkg cache `zixuan/tools/conda_pkgs` | 11 PB fs, ~4 PB free |
| `/storage3/fs1/zhang.ning/Active/hao/miniconda3` | 学长's miniconda (`CONDA_SH`); our envs `agentdojo`, `llamafactory`, `sd_gen` live there next to 学长's `agentdyn*`, `reasalign`, `rl-hammer` (do not touch those) | |
| `/home/li.hao` | home, ~47 GB quota — keep nothing big here (`~/.bashrc` already points `XDG_CACHE_HOME`/`HF_HOME` to `~/storage/.cache`) | |
| `/tmp` on any node | node-local — never put `#SBATCH -o`, scripts or data there (a `--wrap` job cannot see a script written to the login node's `/tmp`; cost one failed job 3003002) | |
- `HF_HOME` for SRFT is **`zixuan/.cache/huggingface`** (`env/washu.sh`), NOT 学长's `~/storage/.cache/huggingface`: that one holds 学长's HF token
  (`hf auth whoami` → another account) and `hf auth login` would overwrite it. Qwen/Qwen3-8B is downloaded once more into ours (16 GB).
  `~/.bashrc` sets the 学长 path, so in an interactive shell always `source env/select.sh` before any `hf` command, and because `env/washu.sh`
  exports `HF_HUB_OFFLINE=1`, prefix anything that talks to the Hub with `HF_HUB_OFFLINE=0`:
  `source env/select.sh && conda activate llamafactory && HF_HUB_OFFLINE=0 hf auth login` (git-credential prompt → n; verified 2026-09-09).
- **`~/.bashrc` also exports `TRANSFORMERS_CACHE=~/storage/.cache/huggingface/transformers`, and sbatch passes the login environment to the job**
  (`--export=ALL` default). transformers 4.51 lets `TRANSFORMERS_CACHE` override `HF_HOME/hub`, so the first eval smoke (3003058) silently loaded
  Qwen3-8B from 学长's cache instead of ours. `env/washu.sh` now `unset TRANSFORMERS_CACHE`; keep it that way (do not re-export it).

## How to submit (VERIFIED 2026-09-09)
```bash
cd /storage3/fs1/zhang.ning/Active/hao/zixuan/SRFT
bash env/sb gpu --test-only self_distill/shard.sbatch                 # dry run → "Job … to start at <date> … in partition general-gpu"
bash env/sb gpu -p general-short env/jobs/gpu_smoke.sbatch            # ≤ 30 min → H100 within seconds
bash env/sb gpu self_distill/shard.sbatch                              # real GPU job → general-gpu,general-preempt-gpu
bash env/sb cpu --export=ALL,ENV=agentdojo -J build_agentdojo env/jobs/build_env.sbatch   # CPU job (general-cpu)
```
- `-p general-short` only works with `-t ≤ 00:30:00`; a script with `-t 03:00:00` submitted to `general-gpu,general-preempt-gpu,general-short`
  is REJECTED outright ("Requested time limit is invalid"), so `GPU_PARTITIONS` does not include `general-short` — add `-p general-short -t 00:30:00` by hand.
- Interactive GPU shell (from `apply_node.sh`): `srun -A compute2-zhang.ning -p general-short --gres=gpu:1 --mem 16g -c 4 -t 00:30:00 --pty /bin/bash`
  (or `-p general-interactive` for up to 5 d).
- Monitoring: `squeue -u li.hao -o "%.9i %.20P %.18j %.8T %.10M %R"`, `sacct -j <id> -X --format=JobID,Partition,NodeList,Elapsed,State`,
  `scontrol show job <id>`, `squeue -j <id> --start`. On the node: `srun --jobid=<id> --overlap nvidia-smi`. No `seff`/`jobstats`.
  **`squeue` shows only our own jobs here** (`squeue -p general-gpu -t R` prints nothing although every GPU is allocated) — use
  `../tools/jobs/gpu_occupancy.sh` (loops `scontrol show node`, prints alloc/total GPUs per node) to see how full the cluster is.
- Env builds: never run two `conda create` jobs at once with the same `CONDA_PKGS_DIRS` (2026-09-06 job 2987716 died with `InvalidArchiveError`);
  chain them with `--dependency=afterany:<id>` as `env/jobs/build_env.sbatch`'s header says.

## Gotcha: IDLE nodes with leaked memory kill every job they get (2026-09-11)
Symptom: a batch of jobs submitted together all fail after 1–2 s with `State=FAILED ExitCode=0:53`,
`Reason=RaisedSignal:53(Real-time_signal_19)`, and **no output file at all** (the job dies before the script runs). `sacct` shows they all
landed on the same node. Seen on 2026-09-11: 19 of 20 eval jobs went to `c2-gpu-010` and died; the one that got another node ran fine.
Cause: the node is `State=IDLE` with `CPUAlloc=0`, so slurm packs the whole batch onto it, but its real free memory is a fraction of
`RealMemory` because processes from an earlier job leaked. A job asking for 30–42 GB is killed immediately. `c2-gpu-010` had 21 GB free of
928 GB, `c2-gpu-008` 87 GB.
Check before submitting a batch (prints every GPU node that slurm thinks is idle but has < 10 % of its memory free):
```bash
for n in $(sinfo -p general-gpu,general-preempt-gpu -N -h -o "%N" | sort -u); do d=$(scontrol show node $n | tr '\n' ' ');
  ca=$(echo "$d" | grep -oP 'CPUAlloc=\K[0-9]+'); fm=$(echo "$d" | grep -oP 'FreeMem=\K[0-9]+'); rm=$(echo "$d" | grep -oP 'RealMemory=\K[0-9]+')
  [ -n "$fm" ] && [ "${ca:-1}" = 0 ] && [ "$fm" -lt $((rm/10)) ] && echo "$n IDLE but only ${fm}MB of ${rm}MB free"; done
```
Then pass `SRFT_SBATCH_FLAGS="--exclude=<nodes>"` to `submit_eval_sdL2.sh` (it forwards the flag to every job) or `--exclude=` to `env/sb`.
Partial results are safe: the eval skips finished task JSONs, so cancelling and resubmitting a run costs only the in-flight shard.

## Queue reality (dated)
- 2026-09-06 23:45: all `general-gpu` (56/56) and `general-preempt-gpu` (37/40) GPUs allocated, **zero pending jobs** — full of long-running jobs,
  not a deep queue; a new 1-GPU job waits until someone's job ends. `general-short` had free GPUs.
- 2026-09-09 17:00: same picture (every GPU node 3/4–8/8 allocated, c2-gpu-004 DOWN); `sbatch --test-only` estimates a 1-GPU `general-gpu` start
  on **2026-09-22** (13 days). `general-short` estimate: immediately.
- Strategy: debugging / smoke / stats / anything ≤ 30 min → `general-short`. Real training / eval → `bash env/sb gpu …` (= both big partitions,
  whichever frees first); ask for 1 GPU (1-GPU slots free far more often than 4-GPU ones), keep `-t` honest. Eval is restart-safe (one JSON per
  task, finished ones skipped); LLaMA-Factory training must resume from `checkpoint-*` if it lands on preempt (the launcher's auto-resume from the
  archive copy is not in `main` yet — port before the first real training run here).

## Verified on this cluster (2026-09-09, all from `main` at eb14698 + this commit)
| step | command | result |
|---|---|---|
| dry run | `bash env/sb gpu --test-only self_distill/shard.sbatch` | accepted (est. start 2026-09-22 on general-gpu); train launcher and `submit_eval_sdL2.sh` dry runs accepted too |
| env builds | `env/jobs/build_env.sbatch` × 3 chained (jobs 3002999 / 3003000 / 3003001, general-cpu, 11–13 min each) | `agentdojo`: pip check clean, pytest 21 passed, click 8.1.8; `llamafactory`: torch 2.7.0+cu128, transformers 4.57.1, peft 0.17.1, trl 0.9.6, LLaMA-Factory 0.9.4.dev0; `sd_gen`: torch 2.7.0+cu126, vllm 0.9.2, transformers 4.53.2, pip check clean. All three `pip freeze`s are identical to `env/freeze/*.txt` apart from the editable path and the conda `packaging` line |
| downloads | general-cpu `--wrap` job 3003020 with `hf download` (public repos, no token needed) | Qwen/Qwen3-8B 16 GB into our `HF_HOME` (30 s), `toucan_32B_v3_base.json` 103 MB, paper LoRA 250 MB (byte-identical to the archive copy) |
| GPU smoke | `bash env/sb gpu -p general-short env/jobs/gpu_smoke.sbatch` (3003057) | H100 80GB HBM3, driver 580.105.08, cuda OK, bf16 matmul 32 TFLOPS, SMOKE_OK, 25 s wall |
| eval shard | `bash env/sb gpu -p general-short -t 00:30:00 -J eval_smoke_bank --export=ALL,RUN_LOGDIR=runs/_smoke_washu_v0_bank_inj0,LORA_PATH=<abs paper LoRA>,NPROC=1,SUITE=banking,ATTACK=important_instructions,INJ=injection_task_0 agentdojo/scripts/eval_sdL2.sbatch` (3003059) | 16/16 banking trajectories in 16 min, 0 errors; utility 8/16 vs paper run 6/16 on the same files, attack success 1 vs 0 (T=0.6 sampling noise, same as the 2026-09-07 smoke). Output deleted afterwards (throwaway) |
Not yet done here: a training smoke (`05_training.md` has the general-short recipe) and porting the archive launcher's auto-resume into `main`'s train script.
