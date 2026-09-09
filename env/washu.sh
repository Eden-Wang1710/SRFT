# WashU RIS Compute2 settings. Loaded by env/select.sh when env/local.sh says SRFT_CLUSTER=washu.
# Cluster notes: docs/09_cluster_washu.md   (login node c2-login-002, user li.hao, PI allocation compute2-zhang.ning)
export CONDA_SH=/storage3/fs1/zhang.ning/Active/hao/miniconda3/etc/profile.d/conda.sh   # 学长's miniconda; our envs agentdojo, llamafactory, sd_gen live in it
export HF_HOME=/storage3/fs1/zhang.ning/Active/hao/zixuan/.cache/huggingface           # OUR cache (Qwen/Qwen3-8B + hf token); 学长's is ~/storage/.cache/huggingface, do not share the token file
export HF_DATASETS_CACHE="$HF_HOME/datasets"
unset TRANSFORMERS_CACHE                    # ~/.bashrc exports it (学长's path) and sbatch inherits the env; it would override HF_HOME/hub in transformers
export HF_HUB_OFFLINE=1                     # Qwen/Qwen3-8B cached 2026-09-09 (job 3003020); prefix HF_HUB_OFFLINE=0 to download/upload
export CONDA_PKGS_DIRS=/storage3/fs1/zhang.ning/Active/hao/zixuan/tools/conda_pkgs   # env builds: keep conda/pip caches off /home (small quota)
export PIP_CACHE_DIR=/storage3/fs1/zhang.ning/Active/hao/zixuan/.cache/pip
export SLURM_ACCOUNT=compute2-zhang.ning    # the only association of li.hao; every job needs -A
export GPU_PARTITIONS=general-gpu,general-preempt-gpu   # H100 80GB (no preemption) + A100 80GB (PreemptMode=REQUEUE); both 4 GPU/node, 15 d max
export CPU_PARTITIONS=general-cpu           # 77 nodes × 64 cores / ~900 GB, 15 d max, starts immediately
export SBATCH_EXTRA=""                      # nothing mandatory beyond -A here (no --comment / --qos)
# The shell is the login node itself (no enclosing slurm job), so plain sbatch works.
# Quick jobs (<= 30 min): override with `bash env/sb gpu -p general-short ...` (2 H100 nodes, almost no queue).
srft_sbatch() { sbatch "$@"; }
