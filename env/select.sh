# Source this from any script or shell to load the cluster-specific settings.
#   source "$SRFT_ROOT/env/select.sh"
# Reads env/local.sh (NOT in git; one line: SRFT_CLUSTER=skipjack|washu) and then env/<cluster>.sh.
# Exports: SRFT_ROOT SRFT_CLUSTER CONDA_SH HF_HOME HF_DATASETS_CACHE HF_HUB_OFFLINE SLURM_ACCOUNT GPU_PARTITIONS CPU_PARTITIONS SBATCH_EXTRA
# Defines: srft_sbatch (sbatch with the cluster's quirks applied).
_srft_env_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SRFT_ROOT="$(dirname "$_srft_env_dir")"
if [ ! -f "$_srft_env_dir/local.sh" ]; then
  echo "ERROR: $_srft_env_dir/local.sh is missing. Create it with one line 'SRFT_CLUSTER=skipjack' (or washu); see env/local.sh.example" >&2
  return 1 2>/dev/null || exit 1
fi
source "$_srft_env_dir/local.sh"
: "${SRFT_CLUSTER:?env/local.sh must set SRFT_CLUSTER}"
if [ ! -f "$_srft_env_dir/$SRFT_CLUSTER.sh" ]; then
  echo "ERROR: env/$SRFT_CLUSTER.sh does not exist (SRFT_CLUSTER=$SRFT_CLUSTER from env/local.sh)" >&2
  return 1 2>/dev/null || exit 1
fi
source "$_srft_env_dir/$SRFT_CLUSTER.sh"
export SRFT_CLUSTER
export SRFT_SLURM_LOGS="$SRFT_ROOT/slurm_logs"
mkdir -p "$SRFT_SLURM_LOGS"
