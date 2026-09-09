#!/usr/bin/env python3
"""
Compute assistant output token counts for the Meta-SecAlign Llama run.

Meta-SecAlign is based on Llama-3.1-8B-Instruct, so this script defaults to
the local merged Meta-SecAlign/Llama tokenizer used by the runner when it is
available, and otherwise falls back to the Llama-3.1-8B-Instruct tokenizer.

Example:
  python3 eval/compute_meta_secalign_all_content_tokens.py

Equivalent explicit run:
  python3 eval/compute_meta_secalign_all_content_tokens.py \
    meta_secalign_8b/facebook_Meta-SecAlign-8B
"""

import argparse
import csv
import os

from compute_assistant_all_content_tokens import (
    analyze,
    format_number,
    resolve_runs_root,
    safe_output_name,
    write_rows,
)


DEFAULT_RUN = "meta_secalign_8b/facebook_Meta-SecAlign-8B"
DEFAULT_LLAMA_TOKENIZER = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_MERGED_TOKENIZER = (
    "/home/cxiao13/scratch-cxiao13/zixuan/safe-agent-project/"
    "injecAgent-rl-harmmer/rl-injector/checkpoints/Meta-SecAlign-8B-merged"
)


def default_tokenizer() -> str:
    return DEFAULT_MERGED_TOKENIZER if os.path.isdir(DEFAULT_MERGED_TOKENIZER) else DEFAULT_LLAMA_TOKENIZER


def load_llama_tokenizer(tokenizer_name: str):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Error: transformers is required for exact token counts. "
            "Install it in this environment, or run from an environment that already has it."
        ) from exc

    return AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True, trust_remote_code=True)


def write_summary_csv(path: str, rows: list[dict]) -> None:
    fieldnames = [
        "scenario",
        "assistant_steps",
        "total_all_content_tokens",
        "mean_all_content_tokens",
        "median_all_content_tokens",
        "min_all_content_tokens",
        "max_all_content_tokens",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Count tokens in assistant content for the Meta-SecAlign Llama run."
    )
    parser.add_argument(
        "run_name",
        nargs="?",
        default=DEFAULT_RUN,
        help=f"Run directory name/path. Default: {DEFAULT_RUN}",
    )
    parser.add_argument(
        "--runs-base",
        default=None,
        help="Optional base path that contains run directories.",
    )
    parser.add_argument(
        "--tokenizer",
        default=default_tokenizer(),
        help=(
            "Llama tokenizer name/path to use. Defaults to the local merged "
            f"Meta-SecAlign tokenizer if present, otherwise {DEFAULT_LLAMA_TOKENIZER}."
        ),
    )
    args = parser.parse_args()

    runs_root = resolve_runs_root(args.run_name, args.runs_base)
    if not runs_root:
        print(f"Error: could not find run directory: {args.run_name}")
        return 2

    tokenizer = load_llama_tokenizer(args.tokenizer)
    rows, summary_rows = analyze(runs_root, tokenizer)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_name = safe_output_name(args.run_name)
    step_csv = os.path.join(script_dir, f"assistant_meta_secalign_all_content_tokens_{output_name}.csv")
    summary_csv = os.path.join(
        script_dir,
        f"assistant_meta_secalign_all_content_tokens_{output_name}_summary.csv",
    )

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
    write_summary_csv(summary_csv, summary_rows)

    all_summary = summary_rows[-1]
    print(f"Run directory: {runs_root}")
    print(f"Llama tokenizer: {args.tokenizer}")
    print(f"Assistant steps: {all_summary['assistant_steps']}")
    print(f"Total assistant content tokens: {all_summary['total_all_content_tokens']}")
    print(
        "Mean assistant content tokens / assistant step: "
        f"{format_number(all_summary['mean_all_content_tokens'])}"
    )
    print(
        "Median assistant content tokens / assistant step: "
        f"{format_number(all_summary['median_all_content_tokens'])}"
    )
    print(f"Step CSV written to: {step_csv}")
    print(f"Summary CSV written to: {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
