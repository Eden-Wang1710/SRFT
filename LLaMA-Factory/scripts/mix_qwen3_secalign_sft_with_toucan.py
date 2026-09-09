#!/usr/bin/env python3
"""Mix qwen3 SecAlign SFT data with Toucan SFT data.

The qwen3 SecAlign SFT file has plain assistant responses. This script wraps
those responses with an empty Qwen3 thinking block so they can be trained with
thinking enabled alongside Toucan examples that already contain thoughts.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any


DEFAULT_QWEN_INPUT = Path("data/qwen3_secalign_sft.json")
DEFAULT_TOUCAN_INPUT = Path("data/toucan_32B_v2.json")
DEFAULT_OUTPUT = Path("data/qwen3_secalign_sft_toucan_32B_v2_mix.json")
EMPTY_THINK_PREFIX = "<think>\n\n</think>\n\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add empty think blocks to qwen3 SecAlign SFT data, then mix it with Toucan data."
    )
    parser.add_argument(
        "--qwen-input",
        type=Path,
        default=DEFAULT_QWEN_INPUT,
        help=f"Input qwen3 SecAlign SFT JSON file. Default: {DEFAULT_QWEN_INPUT}",
    )
    parser.add_argument(
        "--toucan-input",
        type=Path,
        default=DEFAULT_TOUCAN_INPUT,
        help=f"Input Toucan SFT JSON file. Default: {DEFAULT_TOUCAN_INPUT}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output mixed SFT JSON file. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling. Default: 42")
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation level. Use 0 for compact JSON. Default: 2",
    )
    return parser.parse_args()


def load_json_list(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise TypeError(f"{path} must contain a JSON list.")
    if not all(isinstance(example, dict) for example in data):
        raise TypeError(f"{path} must contain a list of JSON objects.")

    return data


def add_empty_think_to_qwen_examples(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted = copy.deepcopy(examples)
    for example_index, example in enumerate(converted):
        conversations = example.get("conversations")
        if not isinstance(conversations, list):
            raise TypeError(f"Qwen example {example_index} field 'conversations' must be a list.")

        for message_index, message in enumerate(conversations):
            if not isinstance(message, dict):
                raise TypeError(f"Qwen example {example_index} message {message_index} must be a dict.")

            if message.get("from") != "gpt":
                continue

            value = message.get("value")
            if not isinstance(value, str):
                raise TypeError(f"Qwen example {example_index} message {message_index} field 'value' must be a string.")

            if "</think>" in value:
                value = value.split("</think>", 1)[1].lstrip("\n")
            elif "<think>" in value:
                value = value.split("<think>", 1)[0].rstrip("\n")

            message["value"] = EMPTY_THINK_PREFIX + value

    return converted


def main() -> None:
    args = parse_args()

    qwen_data = load_json_list(args.qwen_input)
    toucan_data = load_json_list(args.toucan_input)

    qwen_with_empty_think = add_empty_think_to_qwen_examples(qwen_data)
    mixed_data = qwen_with_empty_think + copy.deepcopy(toucan_data)
    random.Random(args.seed).shuffle(mixed_data)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        if args.indent and args.indent > 0:
            json.dump(mixed_data, f, ensure_ascii=False, indent=args.indent)
            f.write("\n")
        else:
            json.dump(mixed_data, f, ensure_ascii=False, separators=(",", ":"))

    print(
        "Mixed "
        f"{len(qwen_with_empty_think)} qwen examples and {len(toucan_data)} toucan examples "
        f"into {len(mixed_data)} examples: {args.output}"
    )


if __name__ == "__main__":
    main()
