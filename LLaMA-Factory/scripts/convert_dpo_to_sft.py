#!/usr/bin/env python3
"""Convert ShareGPT-style DPO data into SFT data by keeping the chosen answer."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("data/qwen3_secalign_dpo.json")
DEFAULT_OUTPUT = Path("data/qwen3_secalign_sft.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert DPO JSON data to ShareGPT SFT JSON by appending each chosen response."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input DPO JSON file. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output SFT JSON file. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation level. Use 0 for compact JSON. Default: 2",
    )
    return parser.parse_args()


def convert_example(example: dict[str, Any], index: int) -> dict[str, Any]:
    if "conversations" not in example:
        raise ValueError(f"Example {index} is missing 'conversations'.")
    if "chosen" not in example:
        raise ValueError(f"Example {index} is missing 'chosen'.")

    conversations = example["conversations"]
    chosen = example["chosen"]
    if not isinstance(conversations, list):
        raise TypeError(f"Example {index} field 'conversations' must be a list.")
    if not isinstance(chosen, dict):
        raise TypeError(f"Example {index} field 'chosen' must be a dict.")

    sft_example = {k: copy.deepcopy(v) for k, v in example.items() if k not in {"chosen", "rejected"}}
    sft_example["conversations"] = copy.deepcopy(conversations)
    sft_example["conversations"].append(copy.deepcopy(chosen))
    return sft_example


def main() -> None:
    args = parse_args()

    with args.input.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise TypeError("Input DPO file must contain a JSON list.")

    sft_data = [convert_example(example, index) for index, example in enumerate(data)]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        if args.indent and args.indent > 0:
            json.dump(sft_data, f, ensure_ascii=False, indent=args.indent)
            f.write("\n")
        else:
            json.dump(sft_data, f, ensure_ascii=False, separators=(",", ":"))

    print(f"Converted {len(sft_data)} examples: {args.input} -> {args.output}")


if __name__ == "__main__":
    main()
