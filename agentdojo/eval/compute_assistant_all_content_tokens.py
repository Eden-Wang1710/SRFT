#!/usr/bin/env python3
"""
Compute token counts for all assistant content in trajectory JSON files.

This counts Qwen thinking blocks plus the assistant's formal output: visible
text and structured tool calls. Tool calls are serialized in Qwen's generated
format before tokenization:

  <tool_call>
  {"name": "...", "arguments": {...}}
  </tool_call>

Example:
  python3 eval/compute_assistant_all_content_tokens.py \
    3_01_toucan_32B_v2_sft_8k_r64_GA4_qkvo_3epoch_5e-6/Qwen_Qwen3-8B-safe-agent

Outputs:
  - prints aggregate stats to stdout
  - writes per-assistant-step CSV to eval/assistant_all_content_tokens_<run>.csv
  - writes per-scenario summary CSV to eval/assistant_all_content_tokens_<run>_summary.csv
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


def extract_assistant_content_texts(message: dict) -> tuple[str, str]:
    """Return thinking text and visible output text."""
    content = message.get("content")
    if isinstance(content, str):
        return "", content
    if not isinstance(content, list):
        return "", ""

    thinking_chunks = []
    output_chunks = []

    for block in content:
        text = content_text(block)
        if not text:
            continue

        if isinstance(block, dict) and block.get("type") == "thinking":
            thinking_chunks.append(text)
        else:
            output_chunks.append(text)

    return "\n".join(thinking_chunks), "\n".join(output_chunks)


def _tool_call_name_and_args(tool_call: dict) -> tuple[str | None, object]:
    if "function" in tool_call and "args" in tool_call:
        return tool_call.get("function"), tool_call.get("args", {})

    function = tool_call.get("function")
    if isinstance(function, dict):
        name = function.get("name")
        args = function.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {"_raw_arguments": args}
        return name, args

    return tool_call.get("name"), tool_call.get("arguments", tool_call.get("args", {}))


def tool_calls_text(message: dict) -> str:
    tool_calls = message.get("tool_calls") or []
    if not isinstance(tool_calls, list):
        return ""

    chunks = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        name, args = _tool_call_name_and_args(tool_call)
        if not name:
            continue
        if args is None:
            args = {}
        payload = json.dumps({"name": name, "arguments": args}, ensure_ascii=False)
        chunks.append(f"<tool_call>\n{payload}\n</tool_call>")
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
            "total_all_content_tokens": 0,
            "mean_all_content_tokens": "",
            "median_all_content_tokens": "",
            "min_all_content_tokens": "",
            "max_all_content_tokens": "",
        }

    return {
        "assistant_steps": len(counts),
        "total_all_content_tokens": sum(counts),
        "mean_all_content_tokens": mean(counts),
        "median_all_content_tokens": median(counts),
        "min_all_content_tokens": min(counts),
        "max_all_content_tokens": max(counts),
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
            thinking_text, output_text = extract_assistant_content_texts(message)
            tool_text = tool_calls_text(message)
            think_tokens = count_tokens(tokenizer, thinking_text)
            text_content_tokens = count_tokens(tokenizer, output_text)
            tool_call_tokens = count_tokens(tokenizer, tool_text)
            output_content_tokens = text_content_tokens + tool_call_tokens
            all_content_tokens = think_tokens + output_content_tokens

            scenario_counts[scenario].append(all_content_tokens)
            rows.append(
                {
                    "scenario": scenario,
                    "relative_path": relative_path,
                    "message_index": message_index,
                    "assistant_step_index": assistant_step_index,
                    "think_chars": len(thinking_text),
                    "think_tokens": think_tokens,
                    "text_content_chars": len(output_text),
                    "text_content_tokens": text_content_tokens,
                    "tool_call_chars": len(tool_text),
                    "tool_call_tokens": tool_call_tokens,
                    "output_content_chars": len(output_text) + len(tool_text),
                    "output_content_tokens": output_content_tokens,
                    "all_content_chars": len(thinking_text) + len(output_text) + len(tool_text),
                    "all_content_tokens": all_content_tokens,
                }
            )

    summary_rows = []
    for scenario in sorted(scenario_counts):
        summary = summarize_counts(scenario_counts[scenario])
        summary_rows.append({"scenario": scenario, **summary})

    all_counts = [row["all_content_tokens"] for row in rows]
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
        description="Count tokens in all assistant content for trajectory JSON files."
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
    step_csv = os.path.join(script_dir, f"assistant_all_content_tokens_{output_name}.csv")
    summary_csv = os.path.join(script_dir, f"assistant_all_content_tokens_{output_name}_summary.csv")

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
            "text_content_chars",
            "text_content_tokens",
            "tool_call_chars",
            "tool_call_tokens",
            "output_content_chars",
            "output_content_tokens",
            "all_content_chars",
            "all_content_tokens",
        ],
    )
    write_rows(
        summary_csv,
        summary_rows,
        [
            "scenario",
            "assistant_steps",
            "total_all_content_tokens",
            "mean_all_content_tokens",
            "median_all_content_tokens",
            "min_all_content_tokens",
            "max_all_content_tokens",
        ],
    )

    all_summary = summary_rows[-1]
    print(f"Run directory: {runs_root}")
    print(f"Tokenizer: {args.tokenizer}")
    print(f"Assistant steps: {all_summary['assistant_steps']}")
    print(f"Total all assistant content tokens: {all_summary['total_all_content_tokens']}")
    print(
        "Mean all assistant content tokens / assistant step: "
        f"{format_number(all_summary['mean_all_content_tokens'])}"
    )
    print(
        "Median all assistant content tokens / assistant step: "
        f"{format_number(all_summary['median_all_content_tokens'])}"
    )
    print(f"Step CSV written to: {step_csv}")
    print(f"Summary CSV written to: {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
