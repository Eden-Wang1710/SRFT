#!/bin/bash
#SBATCH -p a100
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 00:05:00
#SBATCH -A cxiao13
#SBATCH -o /weka/scratch/jhu/cxiao13/zwang544/tools/jobs/gpu_smoke.out
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
/weka/scratch/jhu/cxiao13/zwang544/tools/miniforge3/envs/agentdojo/bin/python - <<'PY'
import torch
print("cuda available:", torch.cuda.is_available(), "| device:", torch.cuda.get_device_name(0))
a = torch.randn(1024,1024, device="cuda", dtype=torch.bfloat16); print("bf16 matmul ok:", (a@a).float().sum().item() != 0)
import peft, transformers; print("peft", peft.__version__, "transformers", transformers.__version__)
PY
