#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove template_1 directories under user_task_* folders."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path(
            "runs/qwen3-8b-think-samples-toucan_windows_v1-cot-one-trajectory"
        ),
        help="Root run directory containing user_task_* folders.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Actually delete directories. Without this flag, only print (dry-run).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir

    if not run_dir.exists():
        raise SystemExit(f"run dir not found: {run_dir}")
    if not run_dir.is_dir():
        raise SystemExit(f"run dir is not a directory: {run_dir}")

    targets = sorted(
        p for p in run_dir.glob("user_task_*/template_1") if p.exists() and p.is_dir()
    )

    if not targets:
        print("No template_1 directories found.")
        return

    print(f"Found {len(targets)} template_1 directories:")
    for path in targets:
        print(path)

    if not args.yes:
        print("\nDry-run only. Re-run with --yes to delete.")
        return

    for path in targets:
        shutil.rmtree(path)
        print(f"Removed: {path}")

    print(f"\nDone. Removed {len(targets)} directories.")


if __name__ == "__main__":
    main()
