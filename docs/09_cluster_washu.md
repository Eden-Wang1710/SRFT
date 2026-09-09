# WashU cluster: how we submit jobs (TO BE FILLED IN ON THE WASHU MACHINE)

Status: **stub**. The WashU-side Claude fills this in after the first `git clone` there, together with `env/washu.sh`
(every `CHANGE_ME` in that file must be replaced before `bash env/sb ...` works). The user already has some WashU notes; fold them in here.

Fill in (same headings as `09_cluster_skipjack.md`):
- Official docs links, login/VPN procedure (VPN needs a one-time code; note which host to ssh to and whether the shell is a slurm job).
- Hardware: partitions, GPUs per node, memory per card, max walltime (`sinfo`, `scontrol show partition`).
- Account / QoS to use (`sacctmgr show assoc user=$USER`), any mandatory sbatch flags → `env/washu.sh` `SLURM_ACCOUNT`, `SBATCH_EXTRA`.
- Where scratch is, where conda lives (`CONDA_SH`), HF cache path (`HF_HOME`); how much disk.
- How the three conda envs were rebuilt (`env/freeze/{agentdojo,llamafactory,sd_gen}.txt`; recipe in `08_environment.md`) and what differed.
- Verified: `bash env/sb gpu --test-only self_distill/shard.sbatch` accepted; `env/jobs/gpu_smoke.sbatch` ran; one eval shard ran end-to-end.
- Queue reality / gotchas as they are discovered (keep a dated list like the skipjack file).
