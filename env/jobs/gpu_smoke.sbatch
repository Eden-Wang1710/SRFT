#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:05:00
#SBATCH -J srft_gpu_smoke
# GPU + agentdojo env smoke test (cluster-neutral). Submit: bash env/sb gpu env/jobs/gpu_smoke.sbatch   (washu: add -p general-short)
SRFT_ROOT="${SRFT_ROOT:-$(d="${SLURM_SUBMIT_DIR:-$PWD}"; while [ ! -f "$d/env/select.sh" ] && [ "$d" != / ]; do d=$(dirname "$d"); done; echo "$d")}"
source "$SRFT_ROOT/env/select.sh"
echo "cluster=$SRFT_CLUSTER node=$(hostname) partition=${SLURM_JOB_PARTITION:-} job=${SLURM_JOB_ID:-} start=$(date)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
source "$CONDA_SH" && conda activate agentdojo
python - <<'PY'
import torch, time
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available(), "|", torch.cuda.get_device_name(0))
a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize(); t = time.time()
for _ in range(20): b = a @ a
torch.cuda.synchronize(); dt = time.time() - t
print(f"20x bf16 4096^2 matmul: {dt:.3f}s  ({20*2*4096**3/dt/1e12:.1f} TFLOPS)")
import peft, transformers, agentdojo; print("peft", peft.__version__, "transformers", transformers.__version__, "agentdojo", agentdojo.__file__)
print("SMOKE_OK")
PY
echo "end=$(date)"
