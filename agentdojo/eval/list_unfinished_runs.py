#!/usr/bin/env python3
"""
List JSON outputs that are missing or errored for a run, i.e. items that
`compute_attack_stats.py` would not include because they never finished.

By default, expected files are taken from a reference run (default: Qwen_Qwen3-8B_baseline).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, Set

DEFAULT_REFERENCE_RUN = "Qwen_Qwen3-8B_baseline"


def resolve_run_path(run_name: str, runs_base: str | None) -> Path | None:
    """Locate runs/<run_name> similarly to compute_attack_stats."""
    script_dir = Path(__file__).resolve().parent
    candidates: Iterable[Path]
    if runs_base:
        candidates = [Path(runs_base) / run_name]
    else:
        candidates = [
            script_dir / ".." / "runs" / run_name,
            script_dir / ".." / ".." / "runs" / run_name,
            script_dir / "runs" / run_name,
            Path.cwd() / "runs" / run_name,
            Path("/data/xiaogeng_liu/experiments/zixuan/agentdojo/runs") / run_name,
        ]
    for cand in candidates:
        if cand.is_dir():
            return cand.resolve()
    return None


def collect_json_relpaths(base_dir: Path) -> Set[Path]:
    """Collect JSON files relative to the base directory."""
    paths: Set[Path] = set()
    for root, _dirs, files in os.walk(base_dir):
        for f in files:
            if f.endswith(".json"):
                full = Path(root) / f
                rel = full.relative_to(base_dir)
                paths.add(rel)
    return paths


def load_json(path: Path):
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def find_missing_and_failed(target_root: Path, reference_root: Path):
    ref_files = collect_json_relpaths(reference_root)
    target_files = collect_json_relpaths(target_root)

    missing = sorted(ref_files - target_files)

    failed = []
    for rel in sorted(target_files):
        data = load_json(target_root / rel)
        if isinstance(data, dict) and data.get("error"):
            failed.append(rel)
    return missing, failed


def main() -> int:
    parser = argparse.ArgumentParser(description="List unfinished (missing/errored) JSON files for a run.")
    parser.add_argument("run_name", help="Run name under runs/, e.g. Qwen_Qwen3-8B-safe-agent")
    parser.add_argument("--runs-base", default=None, help="Optional base directory containing runs/")
    parser.add_argument("--reference-run", default=DEFAULT_REFERENCE_RUN, help="Run to use as expected baseline (default: %(default)s)")
    parser.add_argument("--out", default=None, help="Optional output path (default: eval/unfinished_<run_name>.txt)")
    args = parser.parse_args()

    target_root = resolve_run_path(args.run_name, args.runs_base)
    if target_root is None:
        raise SystemExit(f"Could not find runs/{args.run_name}.")

    reference_root = resolve_run_path(args.reference_run, args.runs_base)
    if reference_root is None:
        raise SystemExit(f"Could not find reference runs/{args.reference_run}.")

    missing, failed = find_missing_and_failed(target_root, reference_root)

    out_path = Path(args.out) if args.out else Path(__file__).resolve().parent / f"unfinished_{args.run_name}.txt"
    out_lines = ["# Missing (present in reference, absent in target)"]
    out_lines += [str(target_root / rel) for rel in missing]
    out_lines.append("\n# Failed (error field is not null in target)")
    out_lines += [str(target_root / rel) for rel in failed]
    out_path.write_text("\n".join(out_lines), encoding="utf-8")

    print(f"Target run: {target_root}")
    print(f"Reference run: {reference_root}")
    print(f"Missing files: {len(missing)}")
    print(f"Failed files: {len(failed)}")
    print(f"Wrote details to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
