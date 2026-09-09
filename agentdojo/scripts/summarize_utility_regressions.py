#!/usr/bin/env python3
"""Summarize utility differences between two AgentDojo runs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_LORA_RUN = Path(
    "safe-agent-project/agentdojo/runs/"
    "qwen3_8b_toucan_lora_think_sys_append_no_attack/"
    "Qwen_Qwen3-8B-safe-agent"
)
DEFAULT_BASELINE_RUN = Path("safe-agent-project/agentdojo/runs/Qwen_Qwen3-8B_baseline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find samples where --lora-run and --baseline-run have different "
            "utility values."
        )
    )
    parser.add_argument("--lora-run", type=Path, default=DEFAULT_LORA_RUN)
    parser.add_argument("--baseline-run", type=Path, default=DEFAULT_BASELINE_RUN)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output file. Extension .json writes JSON; anything else writes CSV.",
    )
    parser.add_argument(
        "--include-text",
        action="store_true",
        help="Include the user prompt and final assistant answer in CSV/JSON output.",
    )
    parser.add_argument(
        "--direction",
        choices=("lora-false-base-true", "lora-true-base-false"),
        default="lora-false-base-true",
        help=(
            "Which utility flip to report. Default reports LoRA failures that "
            "baseline solved."
        ),
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return data


def sample_key(run_root: Path, path: Path, data: dict[str, Any]) -> tuple[str, str, str, str]:
    """Return a stable key: suite, user task, attack type, injection task."""
    rel = path.relative_to(run_root)
    rel_parts = rel.parts
    suite = str(data.get("suite_name") or rel_parts[0])
    user_task = str(data.get("user_task_id") or rel_parts[1])
    attack_type = data.get("attack_type")
    injection_task = data.get("injection_task_id")

    if attack_type is None and len(rel_parts) >= 4:
        attack_type = rel_parts[2]
    if injection_task is None and len(rel_parts) >= 4:
        injection_task = Path(rel_parts[3]).stem

    return (
        suite,
        user_task,
        "none" if attack_type in (None, "") else str(attack_type),
        "none" if injection_task in (None, "") else str(injection_task),
    )


def load_run(run_root: Path) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    if not run_root.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_root}")

    samples: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for path in sorted(run_root.rglob("*.json")):
        data = load_json(path)
        key = sample_key(run_root, path, data)
        if key in samples:
            raise ValueError(f"Duplicate sample key {key}: {samples[key]['path']} and {path}")
        samples[key] = {"path": path, "data": data}
    return samples


def first_text(data: dict[str, Any], role: str) -> str:
    for message in data.get("messages", []):
        if message.get("role") != role:
            continue
        for item in message.get("content") or []:
            if item.get("type") == "text":
                return str(item.get("content", ""))
    return ""


def final_assistant_text(data: dict[str, Any]) -> str:
    for message in reversed(data.get("messages", [])):
        if message.get("role") != "assistant":
            continue
        parts = [
            str(item.get("content", ""))
            for item in (message.get("content") or [])
            if item.get("type") == "text"
        ]
        if parts:
            return "\n".join(parts)
    return ""


def make_row(
    key: tuple[str, str, str, str],
    lora_sample: dict[str, Any],
    baseline_sample: dict[str, Any],
    include_text: bool,
) -> dict[str, Any]:
    suite, user_task, attack_type, injection_task = key
    lora_data = lora_sample["data"]
    baseline_data = baseline_sample["data"]
    row: dict[str, Any] = {
        "suite": suite,
        "user_task_id": user_task,
        "attack_type": attack_type,
        "injection_task_id": injection_task,
        "lora_utility": lora_data.get("utility"),
        "baseline_utility": baseline_data.get("utility"),
        "lora_security": lora_data.get("security"),
        "baseline_security": baseline_data.get("security"),
        "lora_error": lora_data.get("error"),
        "baseline_error": baseline_data.get("error"),
        "lora_path": str(lora_sample["path"]),
        "baseline_path": str(baseline_sample["path"]),
    }
    if include_text:
        row.update(
            {
                "user_prompt": first_text(lora_data, "user"),
                "lora_final_answer": final_assistant_text(lora_data),
                "baseline_final_answer": final_assistant_text(baseline_data),
            }
        )
    return row


def write_output(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        with path.open("w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        return

    fieldnames = list(rows[0].keys()) if rows else [
        "suite",
        "user_task_id",
        "attack_type",
        "injection_task_id",
        "lora_utility",
        "baseline_utility",
        "lora_path",
        "baseline_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    lora_samples = load_run(args.lora_run)
    baseline_samples = load_run(args.baseline_run)

    rows: list[dict[str, Any]] = []
    missing_in_baseline = 0
    wanted = {
        "lora-false-base-true": (False, True),
        "lora-true-base-false": (True, False),
    }[args.direction]

    for key, lora_sample in lora_samples.items():
        baseline_sample = baseline_samples.get(key)
        if baseline_sample is None:
            missing_in_baseline += 1
            continue
        if (
            lora_sample["data"].get("utility") is wanted[0]
            and baseline_sample["data"].get("utility") is wanted[1]
        ):
            rows.append(make_row(key, lora_sample, baseline_sample, args.include_text))

    by_suite = Counter(row["suite"] for row in rows)
    print(f"LoRA samples: {len(lora_samples)}")
    print(f"Baseline samples: {len(baseline_samples)}")
    print(f"Missing baseline matches for LoRA samples: {missing_in_baseline}")
    print(f"Direction: {args.direction}")
    print(f"LoRA utility={wanted[0]} and baseline utility={wanted[1]}: {len(rows)}")
    if by_suite:
        print("By suite:")
        for suite, count in sorted(by_suite.items()):
            print(f"  {suite}: {count}")
        print("Samples:")
        for row in rows:
            print(
                "  "
                f"{row['suite']}/{row['user_task_id']}/"
                f"{row['attack_type']}/{row['injection_task_id']}"
            )

    if args.output:
        write_output(args.output, rows)
        print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
