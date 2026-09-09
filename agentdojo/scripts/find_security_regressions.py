#!/usr/bin/env python3
"""Find trajectories where one run is safe and another run is unsafe.

The comparison key is:
  (suite_name, user_task_id, injection_task_id, attack_type)

By default this script compares the two run directories requested in the
workspace and writes a text summary of matching JSON files.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_BASELINE_DIR = Path(
    "safe-agent-project/agentdojo/runs/"
    "qwen3_8b_base_think_no_sys_append/"
    "Qwen_Qwen3-8B-safe-agent"
)
DEFAULT_CANDIDATE_DIR = Path(
    "safe-agent-project/agentdojo/runs/"
    "3_01_toucan_32B_v2_sft_8k_r64_GA4_qkvo_3epoch_5e-6/"
    "Qwen_Qwen3-8B-safe-agent"
)
DEFAULT_OUTPUT = Path("safe-agent-project/agentdojo/runs/security_regressions.txt")


@dataclass(frozen=True)
class Trajectory:
    key: tuple[str, str, str, str]
    path: Path
    safety: bool | None


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected top-level JSON object in {path}")
    return payload


def _safety_value(payload: dict[str, Any]) -> bool | None:
    """Return the safety flag.

    Some AgentDojo outputs use "security" for the safety result. Prefer an
    explicit "safety" field if present, otherwise fall back to "security".
    """

    value = payload.get("safety", payload.get("security"))
    if isinstance(value, bool):
        return value
    return None


def _trajectory_key(path: Path, payload: dict[str, Any]) -> tuple[str, str, str, str]:
    suite = payload.get("suite_name")
    user_task = payload.get("user_task_id")
    injection_task = payload.get("injection_task_id")
    attack_type = payload.get("attack_type")

    if not isinstance(suite, str):
        suite = path.parts[-4] if len(path.parts) >= 4 else ""
    if not isinstance(user_task, str):
        user_task = path.parts[-3] if len(path.parts) >= 3 else ""
    if not isinstance(attack_type, str):
        attack_type = path.parts[-2] if len(path.parts) >= 2 else ""
    if not isinstance(injection_task, str):
        injection_task = path.stem

    return (suite, user_task, injection_task, attack_type)


def _load_trajectories(root: Path) -> dict[tuple[str, str, str, str], Trajectory]:
    trajectories: dict[tuple[str, str, str, str], Trajectory] = {}
    for path in sorted(root.rglob("*.json")):
        payload = _read_json(path)
        key = _trajectory_key(path, payload)
        trajectories[key] = Trajectory(key=key, path=path, safety=_safety_value(payload))
    return trajectories


def _format_key(key: tuple[str, str, str, str]) -> str:
    suite, user_task, injection_task, attack_type = key
    return (
        f"suite={suite}\tuser_task={user_task}\t"
        f"injection_task={injection_task}\tattack_type={attack_type}"
    )


def write_report(
    baseline_dir: Path,
    candidate_dir: Path,
    output_path: Path,
) -> int:
    baseline = _load_trajectories(baseline_dir)
    candidate = _load_trajectories(candidate_dir)

    matches = []
    for key in sorted(set(baseline) & set(candidate)):
        before = baseline[key]
        after = candidate[key]
        if before.safety is True and after.safety is False:
            matches.append((key, before.path, after.path))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write("Trajectories with baseline safety=True and candidate safety=False\n")
        f.write(f"Baseline dir: {baseline_dir}\n")
        f.write(f"Candidate dir: {candidate_dir}\n")
        f.write(f"Compared keys: {len(set(baseline) & set(candidate))}\n")
        f.write(f"Matches: {len(matches)}\n\n")

        for index, (key, before_path, after_path) in enumerate(matches, start=1):
            f.write(f"[{index}] {_format_key(key)}\n")
            f.write(f"baseline_json: {before_path}\n")
            f.write(f"candidate_json: {after_path}\n\n")

    return len(matches)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "List JSON trajectories where the baseline run has safety=True "
            "and the candidate run has safety=False."
        )
    )
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATE_DIR)
    parser.add_argument("--output", "-o", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = write_report(args.baseline_dir, args.candidate_dir, args.output)
    print(f"Wrote {count} matches to {args.output}")


if __name__ == "__main__":
    main()
