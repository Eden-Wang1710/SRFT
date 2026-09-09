# Skipjack (JHU ARCH) settings. Loaded by env/select.sh when env/local.sh says SRFT_CLUSTER=skipjack.
# Cluster notes: docs/09_cluster_skipjack.md
export CONDA_SH=/weka/scratch/jhu/cxiao13/zwang544/tools/miniforge3/etc/profile.d/conda.sh   # conda envs: agentdojo, llamafactory, sd_gen
export HF_HOME=/weka/scratch/jhu/cxiao13/zwang544/.cache/huggingface
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_OFFLINE=1                     # Qwen/Qwen3-8B is cached; export HF_HUB_OFFLINE=0 when something new must be downloaded
export SLURM_ACCOUNT=cxiao13
export GPU_PARTITIONS=a100,h100,h200,l40s   # list every usable partition so backfill can pick any (all >= 46 GB)
export CPU_PARTITIONS=med
export SBATCH_EXTRA="--comment=accept_cost"  # cost-acceptance flag required by skipjack sbatch since 2026-09-08
# The interactive shell here is itself a slurm job (vscode on partition med): clear its job vars so sbatch creates a NEW job.
srft_sbatch() { env -u SLURM_JOB_ID -u SLURM_JOBID sbatch "$@"; }
