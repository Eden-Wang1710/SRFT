"""
Utility launcher that splits suites across multiple invocations of the single-suite
`sample_qwen_think_from_expert_single.py` runner.

Instead of relying on multiprocessing (which shares stdout), we start one process per
suite group so that each command can redirect its own logs.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from typing import Sequence

import click

DEFAULT_SUITES = ("banking", "travel", "slack", "workspace")
DEFAULT_MODEL_ID = "Qwen/Qwen3-8B"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SINGLE_RUNNER = Path(__file__).resolve().parent / "sample_qwen_think_from_expert_single.py"


def _split_suites(suites: Sequence[str], num_jobs: int) -> list[tuple[str, ...]]:
    """Split suites into `num_jobs` contiguous groups."""
    num_jobs = max(1, min(num_jobs, len(suites)))
    base, remainder = divmod(len(suites), num_jobs)
    groups: list[tuple[str, ...]] = []
    start = 0
    for idx in range(num_jobs):
        size = base + (1 if idx < remainder else 0)
        chunk = tuple(suites[start : start + size])
        groups.append(chunk)
        start += size
    return groups


def _build_base_args(
    expert_run_dir: Path,
    output_dir: Path,
    model_id: str,
    benchmark_version: str,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    thinking_budget: int,
    second_pass_max_new_tokens: int,
    num_sequences: int,
    overwrite: bool,
) -> list[str]:
    args = [
        "--expert-run-dir",
        str(expert_run_dir),
        "--output-dir",
        str(output_dir),
        "--model-id",
        model_id,
        "--benchmark-version",
        benchmark_version,
        "--temperature",
        str(temperature),
        "--top-p",
        str(top_p),
        "--top-k",
        str(top_k),
        "--min-p",
        str(min_p),
        "--thinking-budget",
        str(thinking_budget),
        "--second-pass-max-new-tokens",
        str(second_pass_max_new_tokens),
        "--num-sequences",
        str(num_sequences),
    ]
    args.append("--overwrite" if overwrite else "--skip-existing")
    return args


@click.command()
@click.option("--num-jobs", type=int, default=4, show_default=True, help="How many commands to launch.")
@click.option(
    "--expert-run-dir",
    type=click.Path(path_type=Path),
    default=Path("runs/claude-3-5-sonnet-20241022"),
    show_default=True,
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("runs/qwen3-8b-think-samples"),
    show_default=True,
)
@click.option("--model-id", type=str, default=DEFAULT_MODEL_ID, show_default=True)
@click.option("--benchmark-version", type=str, default="v1.2.1", show_default=True)
@click.option("--temperature", type=float, default=1.0, show_default=True)
@click.option("--top-p", type=float, default=0.95, show_default=True)
@click.option("--top-k", type=int, default=20, show_default=True)
@click.option("--min-p", type=float, default=0.0, show_default=True)
@click.option("--thinking-budget", type=int, default=512, show_default=True)
@click.option("--second-pass-max-new-tokens", type=int, default=512, show_default=True)
@click.option("--num-sequences", type=int, default=3, show_default=True)
@click.option("--overwrite/--skip-existing", default=False, show_default=True)
@click.option(
    "--suite",
    "suites",
    multiple=True,
    help="Optional list of suites to run. Defaults to banking/slack/travel/workspace.",
)
@click.option(
    "--log-dir",
    type=click.Path(path_type=Path),
    default=Path("runs/qwen_sample_logs"),
    show_default=True,
    help="Directory for per-command stdout/stderr logs.",
)
@click.option(
    "--python-bin",
    type=str,
    default="python",
    show_default=True,
    help="Python executable used to run the single-suite script.",
)
@click.option(
    "--script-path",
    type=click.Path(path_type=Path),
    default=SINGLE_RUNNER,
    show_default=True,
    help="Path to sample_qwen_think_from_expert_single.py.",
)
@click.option("--dry-run/--execute", default=False, show_default=True, help="Print commands without running them.")
def main(
    num_jobs: int,
    expert_run_dir: Path,
    output_dir: Path,
    model_id: str,
    benchmark_version: str,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    thinking_budget: int,
    second_pass_max_new_tokens: int,
    num_sequences: int,
    overwrite: bool,
    suites: tuple[str, ...],
    log_dir: Path,
    python_bin: str,
    script_path: Path,
    dry_run: bool,
) -> None:
    if num_jobs < 1:
        raise click.ClickException("--num-jobs must be >= 1.")

    suite_list = tuple(dict.fromkeys(suites)) if suites else DEFAULT_SUITES
    groups = _split_suites(suite_list, min(num_jobs, len(suite_list)))

    log_dir = log_dir.resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    base_args = _build_base_args(
        expert_run_dir.resolve(),
        output_dir.resolve(),
        model_id,
        benchmark_version,
        temperature,
        top_p,
        top_k,
        min_p,
        thinking_budget,
        second_pass_max_new_tokens,
        num_sequences,
        overwrite,
    )

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(PROJECT_ROOT / "src"))

    processes: list[tuple[subprocess.Popen[bytes], Path]] = []
    for job_idx, group in enumerate(groups, start=1):
        command = [python_bin, str(script_path)] + base_args
        for suite in group:
            command.extend(["--suite", suite])

        log_path = log_dir / f"suites_{job_idx}_{'-'.join(group)}.log"
        quoted_command = " ".join(shlex.quote(part) for part in command)
        click.echo(f"[Job {job_idx}] Suites: {', '.join(group)}")
        click.echo(f"[Job {job_idx}] Command: {quoted_command}")
        click.echo(f"[Job {job_idx}] Log: {log_path}")

        if dry_run:
            continue

        log_file = open(log_path, "w", buffering=1)
        proc = subprocess.Popen(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
            env=env,
        )
        processes.append((proc, log_file))

    if dry_run:
        click.echo("Dry run complete. Commands not executed.")
        return

    for proc, log_file in processes:
        return_code = proc.wait()
        log_file.close()
        suites_label = log_file.name.split("suites_", 1)[-1].replace(".log", "")
        if return_code == 0:
            click.echo(f"[Done] {suites_label} completed successfully.")
        else:
            click.echo(f"[Error] {suites_label} exited with code {return_code}. Check {log_file.name}.")


if __name__ == "__main__":
    main()
