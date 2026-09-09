"""
Split user_task folders into shards and run batch Qwen sampling on one shard.

This is a thin wrapper around sample_qwen_think_from_expert_single_batch.py
that limits processing to a subset of user_task_* folders.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable

import click

from agentdojo.scripts.sample_qwen_think_from_expert_single_batch import (
    DEFAULT_MODEL_ID,
    QwenThinkBatchGenerator,
    _process_trajectory,
)


def _user_task_sort_key(path: Path) -> tuple[int, str]:
    name = path.name
    suffix = name.replace("user_task_", "", 1)
    try:
        return (0, f"{int(suffix):08d}")
    except ValueError:
        return (1, suffix)


def _list_user_task_dirs(suite_dir: Path) -> list[Path]:
    return sorted(
        [child for child in suite_dir.iterdir() if child.is_dir() and child.name.startswith("user_task_")],
        key=_user_task_sort_key,
    )


def _split_user_tasks(
    user_tasks: list[Path],
    split_count: int,
    split_index: int,
) -> list[Path]:
    if split_count <= 0:
        raise click.ClickException("--split-count must be >= 1.")
    if split_index < 0 or split_index >= split_count:
        raise click.ClickException("--split-index must be in [0, split-count).")
    if not user_tasks:
        return []
    chunk_size = math.ceil(len(user_tasks) / split_count)
    start = split_index * chunk_size
    end = min(start + chunk_size, len(user_tasks))
    return user_tasks[start:end]


def _iter_trajectories_for_split(
    run_dir: Path,
    suites: tuple[str, ...],
    split_count: int,
    split_index: int,
) -> Iterable[tuple[str, Path]]:
    for suite_dir in sorted(run_dir.iterdir()):
        if not suite_dir.is_dir():
            continue
        suite_name = suite_dir.name
        if suites and suite_name not in suites:
            continue
        user_task_dirs = _list_user_task_dirs(suite_dir)
        if not user_task_dirs:
            click.echo(f"[Skip] suite={suite_name} has no user_task_* folders.")
            continue
        selected = _split_user_tasks(user_task_dirs, split_count, split_index)
        if not selected:
            click.echo(
                f"[Skip] suite={suite_name} split {split_index}/{split_count} has no user_task_* folders."
            )
            continue
        click.echo(
            f"[Split] suite={suite_name} tasks {len(selected)}/{len(user_task_dirs)} in shard "
            f"{split_index}/{split_count}"
        )
        for user_task_dir in selected:
            for json_path in sorted(user_task_dir.rglob("*.json")):
                yield suite_name, json_path


@click.command()
@click.option(
    "--expert-run-dir",
    type=click.Path(path_type=Path),
    default=Path("runs/claude-3-5-sonnet-20241022"),
    show_default=True,
    help="Folder containing the expert Claude trajectories.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("runs/qwen3-8b-think-samples-batch"),
    show_default=True,
    help="Where to store the sampled Qwen trajectories.",
)
@click.option(
    "--model-id",
    type=str,
    default=DEFAULT_MODEL_ID,
    show_default=True,
    help="Hugging Face model to load for Qwen think mode.",
)
@click.option(
    "--benchmark-version",
    type=str,
    default="v1.2.1",
    show_default=True,
    help="Benchmark version to use when loading suite tool specifications.",
)
@click.option(
    "--suite",
    "suites",
    multiple=True,
    help="Optional suite filter (can be provided multiple times).",
)
@click.option(
    "--temperature",
    type=float,
    default=1.0,
    show_default=True,
    help="Sampling temperature for Qwen generation.",
)
@click.option(
    "--top-p",
    type=float,
    default=0.95,
    show_default=True,
    help="Top-p for sampling.",
)
@click.option(
    "--top-k",
    type=int,
    default=20,
    show_default=True,
    help="Top-k for sampling.",
)
@click.option(
    "--min-p",
    type=float,
    default=0.0,
    show_default=True,
    help="Minimum probability mass for Qwen sampling (ignored if model does not support it).",
)
@click.option(
    "--thinking-budget",
    type=int,
    default=512,
    show_default=True,
    help="Maximum number of tokens for the first (thinking) pass.",
)
@click.option(
    "--second-pass-max-new-tokens",
    type=int,
    default=512,
    show_default=True,
    help="Maximum number of tokens for the second pass (final answer).",
)
@click.option(
    "--num-sequences",
    type=int,
    default=5,
    show_default=True,
    help="How many independent Qwen responses to sample for each assistant step.",
)
@click.option(
    "--overwrite/--skip-existing",
    default=True,
    show_default=True,
    help="Whether to overwrite files that already exist in the output directory.",
)
@click.option(
    "--split-count",
    type=int,
    default=2,
    show_default=True,
    help="How many shards to split user_task_* folders into.",
)
@click.option(
    "--split-index",
    type=int,
    default=0,
    show_default=True,
    help="Which shard index to process (0-based).",
)
def main(
    expert_run_dir: Path,
    output_dir: Path,
    model_id: str,
    benchmark_version: str,
    suites: tuple[str, ...],
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    thinking_budget: int,
    second_pass_max_new_tokens: int,
    num_sequences: int,
    overwrite: bool,
    split_count: int,
    split_index: int,
):
    expert_run_dir = expert_run_dir.resolve()
    output_dir = output_dir.resolve()
    if not expert_run_dir.exists():
        raise click.ClickException(f"Expert run directory '{expert_run_dir}' does not exist.")

    click.echo(f"Loading Qwen model '{model_id}' in batch mode...")
    generator = QwenThinkBatchGenerator(
        model_id=model_id,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        thinking_budget=thinking_budget,
        second_pass_max_new_tokens=second_pass_max_new_tokens,
    )
    sampling_params: dict[str, Any] = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "min_p": min_p,
        "thinking_budget": thinking_budget,
        "second_pass_max_new_tokens": second_pass_max_new_tokens,
        "num_sequences": num_sequences,
    }

    processed_files = 0
    sampled_steps = 0

    for suite_name, json_path in _iter_trajectories_for_split(
        expert_run_dir, suites, split_count, split_index
    ):
        processed, steps = _process_trajectory(
            suite_name,
            json_path,
            generator,
            expert_run_dir,
            output_dir,
            benchmark_version,
            sampling_params,
            num_sequences,
            overwrite,
        )
        processed_files += processed
        sampled_steps += steps

    click.echo(
        f"Finished sampling {sampled_steps} assistant steps across {processed_files} trajectories "
        f"for shard {split_index}/{split_count}."
    )


if __name__ == "__main__":
    main()
