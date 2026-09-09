#!/usr/bin/env python3
"""
Compute token counts for assistant thinking blocks in trajectory JSON files.

Example:
  python3 eval/compute_assistant_think_tokens.py \
    3_01_toucan_32B_v2_sft_8k_r64_GA4_qkvo_3epoch_5e-6/Qwen_Qwen3-8B-safe-agent

Outputs:
  - prints aggregate stats to stdout
  - writes per-assistant-step CSV to eval/assistant_think_tokens_<run>.csv
  - writes per-scenario summary CSV to eval/assistant_think_tokens_<run>_summary.csv
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from statistics import mean, median


DEFAULT_TOKENIZER = "Qwen/Qwen3-8B"


def load_tokenizer(tokenizer_name: str):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Error: transformers is required for exact token counts. "
            "Install it in this environment, or run from an environment that already has it."
        ) from exc

    return AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)


def count_tokens(tokenizer, text: str) -> int:
    if not text:
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))


def collect_json_files(base_dir: str) -> list[str]:
    paths = []
    for root, _, files in os.walk(base_dir):
        for filename in files:
            if filename.endswith(".json"):
                paths.append(os.path.join(root, filename))
    return sorted(paths)


def load_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        print(f"Warning: failed to load {path}: {exc}")
        return None


def content_text(block) -> str:
    if isinstance(block, str):
        return block
    if isinstance(block, dict):
        value = block.get("content")
        return value if isinstance(value, str) else ""
    return ""


def extract_thinking_text(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return ""
    if not isinstance(content, list):
        return ""

    chunks = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "thinking":
            text = content_text(block)
            if text:
                chunks.append(text)
    return "\n".join(chunks)


def scenario_from_relative_path(relative_path: str) -> str:
    parts = relative_path.replace("\\", "/").split("/")
    return parts[0] if parts else ""


def resolve_runs_root(run_arg: str, runs_base: str | None) -> str | None:
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if os.path.isdir(run_arg):
        return os.path.abspath(run_arg)

    candidates = []
    if runs_base:
        candidates.append(os.path.join(runs_base, run_arg))
    else:
        candidates.extend(
            [
                os.path.join(script_dir, "..", "runs", run_arg),
                os.path.join(script_dir, "..", "..", "runs", run_arg),
                os.path.join(os.getcwd(), "runs", run_arg),
                os.path.join(os.getcwd(), run_arg),
            ]
        )

    for candidate in candidates:
        if os.path.isdir(candidate):
            return os.path.abspath(candidate)
    return None


def summarize_counts(counts: list[int]) -> dict[str, str | int | float]:
    if not counts:
        return {
            "assistant_steps": 0,
            "total_think_tokens": 0,
            "mean_think_tokens": "",
            "median_think_tokens": "",
            "min_think_tokens": "",
            "max_think_tokens": "",
        }

    return {
        "assistant_steps": len(counts),
        "total_think_tokens": sum(counts),
        "mean_think_tokens": mean(counts),
        "median_think_tokens": median(counts),
        "min_think_tokens": min(counts),
        "max_think_tokens": max(counts),
    }


def analyze(runs_root: str, tokenizer) -> tuple[list[dict], list[dict]]:
    rows = []
    scenario_counts = defaultdict(list)

    for path in collect_json_files(runs_root):
        data = load_json(path)
        if not isinstance(data, dict):
            continue
        messages = data.get("messages")
        if not isinstance(messages, list):
            continue

        relative_path = os.path.relpath(path, runs_root)
        scenario = scenario_from_relative_path(relative_path)
        assistant_step_index = 0

        for message_index, message in enumerate(messages):
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue

            assistant_step_index += 1
            thinking_text = extract_thinking_text(message)
            token_count = count_tokens(tokenizer, thinking_text)
            scenario_counts[scenario].append(token_count)
            rows.append(
                {
                    "scenario": scenario,
                    "relative_path": relative_path,
                    "message_index": message_index,
                    "assistant_step_index": assistant_step_index,
                    "think_chars": len(thinking_text),
                    "think_tokens": token_count,
                }
            )

    summary_rows = []
    for scenario in sorted(scenario_counts):
        summary = summarize_counts(scenario_counts[scenario])
        summary_rows.append({"scenario": scenario, **summary})

    all_counts = [row["think_tokens"] for row in rows]
    summary_rows.append({"scenario": "ALL", **summarize_counts(all_counts)})
    return rows, summary_rows


def safe_output_name(run_arg: str) -> str:
    return run_arg.strip("/").replace("/", "_").replace("\\", "_")


def write_rows(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_number(value) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Count tokens in assistant thinking blocks for trajectory JSON files."
    )
    parser.add_argument(
        "run_name",
        help="Run directory name/path, e.g. 3_01.../Qwen_Qwen3-8B-safe-agent",
    )
    parser.add_argument(
        "--runs-base",
        default=None,
        help="Optional base path that contains run directories.",
    )
    parser.add_argument(
        "--tokenizer",
        default=DEFAULT_TOKENIZER,
        help=f"HF tokenizer name/path to use for token counting. Default: {DEFAULT_TOKENIZER}",
    )
    args = parser.parse_args()

    runs_root = resolve_runs_root(args.run_name, args.runs_base)
    if not runs_root:
        print(f"Error: could not find run directory: {args.run_name}")
        return 2

    tokenizer = load_tokenizer(args.tokenizer)
    rows, summary_rows = analyze(runs_root, tokenizer)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_name = safe_output_name(args.run_name)
    step_csv = os.path.join(script_dir, f"assistant_think_tokens_{output_name}.csv")
    summary_csv = os.path.join(script_dir, f"assistant_think_tokens_{output_name}_summary.csv")

    write_rows(
        step_csv,
        rows,
        [
            "scenario",
            "relative_path",
            "message_index",
            "assistant_step_index",
            "think_chars",
            "think_tokens",
        ],
    )
    write_rows(
        summary_csv,
        summary_rows,
        [
            "scenario",
            "assistant_steps",
            "total_think_tokens",
            "mean_think_tokens",
            "median_think_tokens",
            "min_think_tokens",
            "max_think_tokens",
        ],
    )

    all_summary = summary_rows[-1]
    print(f"Run directory: {runs_root}")
    print(f"Tokenizer: {args.tokenizer}")
    print(f"Assistant steps: {all_summary['assistant_steps']}")
    print(f"Total think tokens: {all_summary['total_think_tokens']}")
    print(f"Mean think tokens / assistant step: {format_number(all_summary['mean_think_tokens'])}")
    print(f"Median think tokens / assistant step: {format_number(all_summary['median_think_tokens'])}")
    print(f"Step CSV written to: {step_csv}")
    print(f"Summary CSV written to: {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
