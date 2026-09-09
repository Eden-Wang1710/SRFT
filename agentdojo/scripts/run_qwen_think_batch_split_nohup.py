#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def _build_common_args(args: argparse.Namespace) -> list[str]:
    common = [
        "--expert-run-dir",
        args.expert_run_dir,
        "--output-dir",
        args.output_dir,
        "--suite",
        args.suite,
        "--thinking-budget",
        str(args.thinking_budget),
        "--second-pass-max-new-tokens",
        str(args.second_pass_max_new_tokens),
        "--num-sequences",
        str(args.num_sequences),
        "--model-id",
        args.model_id,
        "--benchmark-version",
        args.benchmark_version,
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--top-k",
        str(args.top_k),
        "--min-p",
        str(args.min_p),
        "--split-count",
        str(args.split_count),
    ]
    common.append("--overwrite" if args.overwrite else "--skip-existing")
    return common


def _launch_nohup(cmd: list[str], log_path: Path, env: dict[str, str]) -> None:
    with log_path.open("ab") as log_file:
        subprocess.Popen(
            ["nohup", *cmd],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run sample_qwen_think_from_expert_single_batch_split with nohup shards."
    )
    parser.add_argument("--expert-run-dir", default="runs/ground-truth-train")
    parser.add_argument("--output-dir", default="runs/qwen3-8b-think-samples-gt-train")
    parser.add_argument("--suite", default="slack")
    parser.add_argument("--thinking-budget", type=int, default=512)
    parser.add_argument("--second-pass-max-new-tokens", type=int, default=512)
    parser.add_argument("--num-sequences", type=int, default=5)
    parser.add_argument("--model-id", default="Qwen/Qwen3-8B")
    parser.add_argument("--benchmark-version", default="v1.2.1")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--split-count", type=int, default=2)
    parser.add_argument("--pythonthpath", default="src", dest="pythonpath")
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Directory for nohup logs. Default: <output-dir>/nohup_logs",
    )
    parser.add_argument(
        "--script-path",
        default="src/agentdojo/scripts/sample_qwen_think_from_expert_single_batch_split.py",
        help="Path to the split sampling script.",
    )
    args = parser.parse_args()

    log_dir = Path(args.log_dir) if args.log_dir else Path(args.output_dir) / "nohup_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    common_args = _build_common_args(args)
    base_cmd = ["python", args.script_path, *common_args]
    env = os.environ.copy()
    env["PYTHONPATH"] = args.pythonpath

    for split_index in range(args.split_count):
        log_path = log_dir / f"split_{split_index}.out"
        cmd = [*base_cmd, "--split-index", str(split_index)]
        _launch_nohup(cmd, log_path, env)

    print("Launched shards with nohup. Logs:")
    for split_index in range(args.split_count):
        print(f"  {log_dir / f'split_{split_index}.out'}")


if __name__ == "__main__":
    main()
