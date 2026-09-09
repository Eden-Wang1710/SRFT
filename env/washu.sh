# WashU cluster settings. Loaded by env/select.sh when env/local.sh says SRFT_CLUSTER=washu.
# TODO (fill in on the WashU machine, then commit): every variable below must have a real value; docs/09_cluster_washu.md holds the notes.
export CONDA_SH=/path/to/miniforge3/etc/profile.d/conda.sh     # must provide conda envs: agentdojo, llamafactory, sd_gen (env/freeze/*.txt)
export HF_HOME=/path/to/scratch/.cache/huggingface
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_OFFLINE=0                     # set to 1 once Qwen/Qwen3-8B is in the cache
export SLURM_ACCOUNT=CHANGE_ME
export GPU_PARTITIONS=CHANGE_ME             # comma-separated list of GPU partitions with >= 46 GB cards
export CPU_PARTITIONS=CHANGE_ME
export SBATCH_EXTRA=""                      # extra flags every sbatch needs on this cluster (e.g. --qos=...), or empty
srft_sbatch() { sbatch "$@"; }
