import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, TextIO

import click
from rich.logging import RichHandler

from agentdojo.agent_pipeline.agent_pipeline import DEFENSES
from agentdojo.attacks.attack_registry import ATTACKS
from agentdojo.models import ModelsEnum
from agentdojo.task_suite.load_suites import get_suite


def _chunk(items: Iterable[str], desired_chunks: int) -> list[list[str]]:
    items = list(items)
    if desired_chunks < 1:
        raise ValueError("desired_chunks must be >= 1")
    if not items:
        return []
    chunk_size = max(1, -(-len(items) // desired_chunks))
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def _repo_root() -> Path:
    """Return the repository root (folder containing the top-level src/)."""
    return Path(__file__).resolve().parents[3]


def _parse_visible_cuda_devices() -> list[str]:
    """
    Parse CUDA_VISIBLE_DEVICES into concrete device ids.
    Returns an empty list when CUDA visibility is not explicitly set.
    """
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not raw:
        return []
    devices = [d.strip() for d in raw.split(",") if d.strip()]
    return devices


def _build_command(
    python_executable: str,
    suite: str,
    model: str,
    benchmark_version: str,
    logdir: Path,
    attack: str | None,
    defense: str | None,
    tool_delimiter: str,
    system_message_name: str | None,
    system_message: str | None,
    user_tasks: tuple[str, ...],
    max_workers: int,
    model_id: str | None,
    injection_chunk: list[str],
    module_imports: tuple[str, ...],
) -> list[str]:
    command = [
        python_executable,
        "src/agentdojo/scripts/benchmark.py",
        "--suite",
        suite,
        "--model",
        model,
        "--benchmark-version",
        benchmark_version,
        "--logdir",
        str(logdir),
        "--tool-delimiter",
        tool_delimiter,
        "--max-workers",
        str(max_workers),
    ]
    if model_id:
        command += ["--model-id", model_id]
    if attack:
        command += ["--attack", attack]
    if defense:
        command += ["--defense", defense]
    if system_message_name:
        command += ["--system-message-name", system_message_name]
    if system_message:
        command += ["--system-message", system_message]
    for user_task in user_tasks:
        command += ["--user-task", user_task]
    for module_name in module_imports:
        command += ["--module-to-load", module_name]
    for injection_task in injection_chunk:
        command += ["--injection-task", injection_task]
    return command


@click.command()
@click.option(
    "--suite",
    default="workspace",
    show_default=True,
    help="Suite to benchmark.",
)
@click.option(
    "--benchmark-version",
    default="v1.2.1",
    show_default=True,
    help="Benchmark version to load.",
)
@click.option(
    "--model",
    default=ModelsEnum.QWEN_3_8B.name,
    type=click.Choice([member.name for member in ModelsEnum]),
    show_default=True,
    help="Model enum name to benchmark (same as benchmark.py).",
)
@click.option("--model-id", type=str, default=None, help="Model id for local models.")
@click.option(
    "--attack",
    type=str,
    default=None,
    help=f"Attack to use. Choices: {ATTACKS}",
)
@click.option(
    "--defense",
    type=click.Choice(DEFENSES),
    default=None,
    help="Defense to use.",
)
@click.option("--tool-delimiter", default="tool", show_default=True, help="Tool delimiter.")
@click.option("--system-message-name", type=str, default=None)
@click.option("--system-message", type=str, default=None)
@click.option(
    "--user-task",
    "-ut",
    "user_tasks",
    type=str,
    multiple=True,
    default=tuple(),
    help="Restrict to specific user task IDs.",
)
@click.option(
    "--chunks",
    type=int,
    default=2,
    show_default=True,
    help="How many chunks to split injection tasks into.",
)
@click.option(
    "--max-workers",
    type=int,
    default=1,
    show_default=True,
    help="Value to pass to benchmark.py for --max-workers.",
)
@click.option("--logdir", type=Path, default=Path("./runs"), show_default=True)
@click.option(
    "--chunk-log-dir",
    type=Path,
    default=Path("./logs/injection_chunks"),
    show_default=True,
    help="Directory where stdout/stderr of each chunk command is stored.",
)
@click.option(
    "--module-to-load",
    "-ml",
    "modules_to_load",
    type=str,
    multiple=True,
    default=tuple(),
    help="Additional modules to import before running benchmark.py.",
)
@click.option("--dry-run", is_flag=True, help="Print commands instead of running them.")
def main(
    suite: str,
    benchmark_version: str,
    model: str,
    model_id: str | None,
    attack: str | None,
    defense: str | None,
    tool_delimiter: str,
    system_message_name: str | None,
    system_message: str | None,
    user_tasks: tuple[str, ...],
    chunks: int,
    max_workers: int,
    logdir: Path,
    chunk_log_dir: Path,
    modules_to_load: tuple[str, ...],
    dry_run: bool,
):
    suite_obj = get_suite(benchmark_version, suite)
    injection_ids = list(suite_obj.injection_tasks.keys())
    if not injection_ids:
        raise click.ClickException(f"No injection tasks found for suite '{suite}'.")
    chunked = _chunk(injection_ids, chunks)
    if not chunked:
        raise click.ClickException("Injection chunking failed.")

    python_executable = sys.executable
    repo_root = _repo_root()
    chunk_log_dir.mkdir(parents=True, exist_ok=True)

    commands = []
    for idx, chunk in enumerate(chunked, start=1):
        log_file = chunk_log_dir / f"injection_chunk_{idx}.log"
        command = _build_command(
            python_executable,
            suite,
            model,
            benchmark_version,
            logdir,
            attack,
            defense,
            tool_delimiter,
            system_message_name,
            system_message,
            user_tasks,
            max_workers,
            model_id,
            chunk,
            modules_to_load,
        )
        commands.append((idx, chunk, command, log_file))

    for idx, chunk, command, logfile in commands:
        logging.info("Chunk %s (%s tasks): %s", idx, len(chunk), ", ".join(chunk))
        logging.info("  Log: %s", logfile)
        logging.info("  Cmd: %s", " ".join(command))

    if dry_run:
        logging.info("Dry run enabled; not launching any subprocesses.")
        return

    visible_devices = _parse_visible_cuda_devices()
    if visible_devices:
        logging.info("Visible CUDA devices: %s", ",".join(visible_devices))
    else:
        logging.warning(
            "CUDA_VISIBLE_DEVICES is not set; chunk processes will share all GPUs and may contend."
        )

    processes: list[tuple[subprocess.Popen[bytes], Path, TextIO]] = []
    for idx, _, command, logfile in commands:
        log_handle = open(logfile, "w", encoding="utf-8")
        proc_env = os.environ.copy()
        if visible_devices:
            assigned_device = visible_devices[(idx - 1) % len(visible_devices)]
            proc_env["CUDA_VISIBLE_DEVICES"] = assigned_device
            logging.info("Chunk %s assigned CUDA_VISIBLE_DEVICES=%s", idx, assigned_device)
        proc = subprocess.Popen(
            command,
            cwd=repo_root,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=proc_env,
        )
        processes.append((proc, logfile, log_handle))

    exit_code = 0
    for proc, logfile, log_handle in processes:
        ret = proc.wait()
        log_handle.close()
        if ret != 0:
            exit_code = ret if exit_code == 0 else exit_code
            logging.error("Chunk process for %s exited with code %s", logfile.name, ret)
        else:
            logging.info("Chunk process for %s completed successfully.", logfile.name)

    if exit_code != 0:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    format = "%(message)s"
    logging.basicConfig(
        format=format,
        level=logging.INFO,
        datefmt="%H:%M:%S",
        handlers=[RichHandler(show_path=False, markup=True)],
    )
    main()
