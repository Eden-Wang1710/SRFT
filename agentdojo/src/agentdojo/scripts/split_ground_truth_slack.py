import random
import shutil
from pathlib import Path

import click


def _list_user_tasks(slack_root: Path) -> list[Path]:
    return sorted(
        [p for p in slack_root.iterdir() if p.is_dir() and p.name.startswith("user_task_")],
        key=lambda p: p.name,
    )


def _copy_task(src: Path, dst_root: Path, overwrite: bool) -> None:
    dst = dst_root / src.name
    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"{dst} already exists. Use --overwrite to replace it.")
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


@click.command()
@click.option(
    "--source-root",
    type=Path,
    default=Path("agentdojo/runs/ground-truth/slack"),
    show_default=True,
    help="Slack ground-truth directory containing user_task_* folders.",
)
@click.option(
    "--train-root",
    type=Path,
    default=Path("agentdojo/runs/ground-truth-train/slack"),
    show_default=True,
    help="Destination for training split.",
)
@click.option(
    "--test-root",
    type=Path,
    default=Path("agentdojo/runs/ground-truth-test/slack"),
    show_default=True,
    help="Destination for test split.",
)
@click.option(
    "--train-ratio",
    type=float,
    default=15 / 20,
    show_default=True,
    help="Train split ratio.",
)
@click.option(
    "--seed",
    type=int,
    default=0,
    show_default=True,
    help="Random seed for shuffling.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Overwrite existing train/test folders for the selected tasks.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Print planned split without copying.",
)
def main(
    source_root: Path,
    train_root: Path,
    test_root: Path,
    train_ratio: float,
    seed: int,
    overwrite: bool,
    dry_run: bool,
) -> None:
    if not source_root.exists():
        raise FileNotFoundError(f"Missing source root: {source_root}")
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("--train-ratio must be between 0 and 1.")

    tasks = _list_user_tasks(source_root)
    rng = random.Random(seed)
    rng.shuffle(tasks)

    train_count = int(len(tasks) * train_ratio)
    train_tasks = tasks[:train_count]
    test_tasks = tasks[train_count:]

    if dry_run:
        print(f"[dry-run] train={len(train_tasks)} test={len(test_tasks)}")
        print("train:", ", ".join(p.name for p in train_tasks))
        print("test:", ", ".join(p.name for p in test_tasks))
        return

    train_root.mkdir(parents=True, exist_ok=True)
    test_root.mkdir(parents=True, exist_ok=True)

    for task in train_tasks:
        _copy_task(task, train_root, overwrite)
    for task in test_tasks:
        _copy_task(task, test_root, overwrite)

    print(f"Done. train={len(train_tasks)} test={len(test_tasks)}")


if __name__ == "__main__":
    main()
