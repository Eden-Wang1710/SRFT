#!/usr/bin/env python3
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(
    "agentdojo/runs/2_24_toucan_v1_sft_8k_3epoch_1e-5"
)
DEFAULT_OUTPUT = Path(
    "agentdojo/runs/2_24_toucan_v1_sft_8k_3epoch_1e-5/security_true_files.txt"
)


def normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return False


def trailing_int(text: str) -> int:
    try:
        return int(text.rsplit("_", 1)[-1])
    except (ValueError, IndexError):
        return 10**9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan JSON files under user_task_* directories and output those with security=True."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Root directory to scan.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output txt path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root
    output = args.output

    if not root.exists():
        raise SystemExit(f"Root path not found: {root}")

    matches: list[tuple[str, str, str, str]] = []

    for json_path in root.rglob("*.json"):
        parts = json_path.parts
        user_task = next((p for p in parts if p.startswith("user_task_")), "")
        if not user_task:
            continue

        try:
            with json_path.open("r", encoding="utf-8") as f:
                obj = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue

        if not isinstance(obj, dict) or not normalize_bool(obj.get("security")):
            continue

        suite_name = str(obj.get("suite_name") or "")
        user_task_id = str(obj.get("user_task_id") or user_task)
        injection_task_id = str(
            obj.get("injection_task_id") or json_path.stem
        )
        matches.append((suite_name, user_task_id, injection_task_id, str(json_path)))

    matches.sort(
        key=lambda x: (
            x[0],
            trailing_int(x[1]),
            trailing_int(x[2]),
            x[3],
        )
    )

    suite_injection_stats: dict[str, Counter[str]] = defaultdict(Counter)
    for suite, _user_task, injection_task, _path in matches:
        suite_injection_stats[suite][injection_task] += 1

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        f.write(f"root: {root}\n")
        f.write(f"security_true_count: {len(matches)}\n\n")
        f.write("suite_injection_stats:\n")
        for suite in sorted(suite_injection_stats):
            f.write(f"[{suite}]\n")
            for injection_task in sorted(
                suite_injection_stats[suite], key=trailing_int
            ):
                count = suite_injection_stats[suite][injection_task]
                f.write(f"{injection_task}\t{count}\n")
            f.write("\n")

        f.write("details:\n")
        for suite, user_task, injection_task, path in matches:
            f.write(
                f"suite={suite}\tuser_task={user_task}\tinjection_task={injection_task}\tfile={path}\n"
            )

    print(f"Done. Found {len(matches)} files with security=True.")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
