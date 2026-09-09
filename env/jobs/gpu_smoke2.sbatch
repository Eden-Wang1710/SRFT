#!/bin/bash
#SBATCH -p a100
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:05:00
#SBATCH -A cxiao13
#SBATCH --comment=accept_cost
#SBATCH -J srft_gpu_smoke
#SBATCH -o /weka/scratch/jhu/cxiao13/zwang544/tools/jobs/%x_%j.out
echo "node=$(hostname) job=$SLURM_JOB_ID start=$(date)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
source /weka/scratch/jhu/cxiao13/zwang544/tools/miniforge3/etc/profile.d/conda.sh && conda activate agentdojo
python - <<'PY'
import torch, time
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available(), "|", torch.cuda.get_device_name(0))
a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize(); t = time.time()
for _ in range(20): b = a @ a
torch.cuda.synchronize(); dt = time.time() - t
print(f"20x bf16 4096^2 matmul: {dt:.3f}s  ({20*2*4096**3/dt/1e12:.1f} TFLOPS)")
import peft, transformers, agentdojo; print("peft", peft.__version__, "transformers", transformers.__version__, "agentdojo", agentdojo.__version__ if hasattr(agentdojo,'__version__') else "ok")
print("SMOKE_OK")
PY
echo "end=$(date)"
