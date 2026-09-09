"""Run one AgentDojo suite on ONE GPU with several concurrent benchmark.py processes (HF generate is batch-1, so a
single 8B bf16 process leaves most of an 80 GB GPU idle). Tasks are split across --nproc processes.

attack mode : python scripts/eval_parallel.py --suite workspace --attack important_instructions --injection-tasks injection_task_0,... --nproc 3 --logdir runs/X
benign mode : python scripts/eval_parallel.py --suite workspace --nproc 3 --logdir runs/X        (splits USER tasks instead)
benchmark.py skips tasks whose result file already exists, so re-running the same command resumes.
"""
import argparse, os, subprocess, sys, time
from pathlib import Path
from agentdojo.task_suite.load_suites import get_suite
from agentdojo.models import ModelsEnum
from agentdojo.models import ModelsEnum

ap = argparse.ArgumentParser()
ap.add_argument("--suite", required=True)
ap.add_argument("--model", default="QWEN_3_8B_SAFE_AGENT")
ap.add_argument("--attack", default=None)
ap.add_argument("--injection-tasks", default="", help="comma list; default = all injection tasks of the suite (attack mode)")
ap.add_argument("--nproc", type=int, default=3)
ap.add_argument("--logdir", required=True)
ap.add_argument("--chunk-log-dir", default=None)
ap.add_argument("--benchmark-version", default="v1.2.1")
args = ap.parse_args()

suite = get_suite(args.benchmark_version, args.suite)
model_value = ModelsEnum[args.model].value if args.model in ModelsEnum.__members__ else args.model   # benchmark.py wants the enum VALUE
if args.model in ModelsEnum.__members__:   # accept enum NAME (QWEN_3_8B_SAFE_AGENT) as well as value (Qwen/Qwen3-8B-safe-agent)
    args.model = ModelsEnum[args.model].value
if args.attack:
    import re
    items = [t for t in re.split(r"[,+:; ]+", args.injection_tasks) if t] or list(suite.injection_tasks)   # "+" separator survives sbatch --export
    flag = "--injection-task"
else:
    items = list(suite.user_tasks)
    flag = "--user-task"
groups = [items[i::args.nproc] for i in range(args.nproc)]
groups = [g for g in groups if g]
chunk_log_dir = Path(args.chunk_log_dir or f"logs/eval_parallel/{Path(args.logdir).name}/{args.suite}_{args.attack or 'none'}_{os.environ.get('SLURM_JOB_ID', os.getpid())}")
chunk_log_dir.mkdir(parents=True, exist_ok=True)
print(f"suite={args.suite} attack={args.attack} model={args.model} procs={len(groups)} -> {args.logdir}", flush=True)
procs = []
for i, g in enumerate(groups):
    cmd = [sys.executable, "src/agentdojo/scripts/benchmark.py", "--suite", args.suite, "--model", model_value,
           "--benchmark-version", args.benchmark_version, "--logdir", args.logdir, "--max-workers", "1"]
    if args.attack:
        cmd += ["--attack", args.attack]
    for t in g:
        cmd += [flag, t]
    log = open(chunk_log_dir / f"proc{i}.log", "w")
    print(f"  proc{i}: {len(g)} {flag} -> {log.name}", flush=True)
    procs.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy()), log))
    time.sleep(20)   # stagger model loading
rc = 0
for p, log in procs:
    r = p.wait(); log.close(); rc = rc or r
    print(f"  {Path(log.name).name} exit={r}", flush=True)
sys.exit(rc)
