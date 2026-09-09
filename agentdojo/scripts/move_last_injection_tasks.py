#!/usr/bin/env python3
"""Move the last N injection_task_* folders from each user_task_* folder."""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path


USER_TASK_PREFIX = "user_task_"
INJECTION_TASK_PREFIX = "injection_task_"


def extract_suffix_number(name: str, prefix: str) -> int | None:
    m = re.fullmatch(rf"{re.escape(prefix)}(\d+)", name)
    if not m:
        return None
    return int(m.group(1))


def user_task_key(path: Path) -> tuple[int, str]:
    num = extract_suffix_number(path.name, USER_TASK_PREFIX)
    return (num if num is not None else -1, path.name)


def injection_task_key(path: Path) -> tuple[int, str]:
    num = extract_suffix_number(path.name, INJECTION_TASK_PREFIX)
    return (num if num is not None else -1, path.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "For each user_task_* in --src-root, move the last N injection_task_* "
            "folders to the mirrored user_task_* folder in --dst-root."
        )
    )
    parser.add_argument(
        "--src-root",
        required=True,
        help="Source root, e.g. agentdojo/runs/qwen3-32b-bedrock-samples/hotel_v1",
    )
    parser.add_argument(
        "--dst-root",
        required=True,
        help="Destination root, e.g. agentdojo/runs/qwen3-32b-bedrock-samples/hotel_v1.5",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=2,
        help="How many injection_task folders to move per user_task (default: 2).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned moves without changing files.",
    )
    parser.add_argument(
        "--on-conflict",
        choices=["error", "skip", "overwrite"],
        default="error",
        help=(
            "Behavior when destination folder already exists: "
            "error (default), skip, or overwrite."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    src_root = Path(args.src_root).resolve()
    dst_root = Path(args.dst_root).resolve()

    if args.count <= 0:
        print("--count must be > 0", file=sys.stderr)
        return 2
    if not src_root.is_dir():
        print(f"Source root does not exist or is not a directory: {src_root}", file=sys.stderr)
        return 2

    user_tasks = sorted(
        [p for p in src_root.iterdir() if p.is_dir() and p.name.startswith(USER_TASK_PREFIX)],
        key=user_task_key,
    )

    planned_moves: list[tuple[Path, Path]] = []
    skipped_conflicts = 0

    for user_task_dir in user_tasks:
        injection_dirs = sorted(
            [
                p
                for p in user_task_dir.iterdir()
                if p.is_dir() and p.name.startswith(INJECTION_TASK_PREFIX)
            ],
            key=injection_task_key,
        )
        if not injection_dirs:
            continue

        selected = injection_dirs[-args.count :]
        dst_user_task_dir = dst_root / user_task_dir.name

        for src_injection_dir in selected:
            dst_injection_dir = dst_user_task_dir / src_injection_dir.name
            if dst_injection_dir.exists() and args.on_conflict == "skip":
                skipped_conflicts += 1
                continue
            planned_moves.append((src_injection_dir, dst_injection_dir))

    if not planned_moves:
        print("No folders to move.")
        return 0

    print(f"Planned moves: {len(planned_moves)}")
    if skipped_conflicts:
        print(f"Skipped due to conflict policy (--on-conflict=skip): {skipped_conflicts}")
    for src, dst in planned_moves:
        print(f"{src} -> {dst}")

    if args.dry_run:
        print("Dry run only, no changes made.")
        return 0

    moved = 0
    for src, dst in planned_moves:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            if args.on_conflict == "error":
                print(f"Destination already exists: {dst}", file=sys.stderr)
                return 1
            if args.on_conflict == "overwrite":
                shutil.rmtree(dst)
        shutil.move(str(src), str(dst))
        moved += 1

    print(f"Done. Moved {moved} folder(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
