#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert trajectory-style OpenAI/tool logs with `cot_self_reflection` into
LLaMA-Factory ShareGPT format.

Input:  a JSON file containing either:
  - a single dict with fields like {"suite_name", "trajectory", "messages": [...]}
  - OR a list of such dicts
  - OR a directory containing many such JSON files (use --input-dir)

Output:
  - <out>.json      : ShareGPT dataset with `conversations`
  - <out>_dataset_info.json : snippet to merge into LLaMA-Factory data/dataset_info.json

Key mapping:
  - role=system -> system (stored in top-level "system")
  - role=user   -> {"from":"human","value":...}
  - role=tool   -> {"from":"observation","value":...}
  - role=assistant with tool_calls -> {"from":"function_call","value":"<think>...</think> + JSON"}
  - role=assistant final text      -> {"from":"gpt","value": "<think>...</think>\\n\\nANSWER"}
"""

from __future__ import annotations

import argparse
import json
import fnmatch
import os
from typing import Any, Dict, List, Optional, Tuple, Union


def ensure_list(x: Any) -> List[Any]:
    return x if isinstance(x, list) else [x]


def parse_arguments(arg: Any) -> Any:
    """
    tool_calls.function.arguments could be:
      - a JSON string like '{"a":1}'
      - a dict
      - empty / invalid JSON string
    Return dict or a dict with _raw fallback.
    """
    if arg is None:
        return {}
    if isinstance(arg, dict):
        return arg
    if isinstance(arg, str):
        s = arg.strip()
        if not s:
            return {}
        try:
            return json.loads(s)
        except Exception:
            return {"_raw": s}
    return {"_raw": str(arg)}


def build_function_call_json(payload: Dict[str, Any]) -> str:
    wire_obj = {
        "name": payload.get("name"),
        "arguments": payload.get("arguments", {}),
    }
    return json.dumps(wire_obj, ensure_ascii=False)


def convert_one(
    example: Dict[str, Any],
    put_step_thought_in_function_call: bool = True,
    put_final_thought_in_gpt: bool = True,
    source_path: Optional[str] = None,
) -> Dict[str, Any]:
    messages = example.get("messages", [])
    system_parts: List[str] = []
    conversations: List[Dict[str, str]] = []
    pending_function_calls: List[str] = []

    def flush_pending_calls() -> None:
        while pending_function_calls:
            conversations.append({"from": "function_call", "value": pending_function_calls.pop(0)})

    i = 0
    while i < len(messages):
        m = messages[i]
        role = m.get("role")
        content = m.get("content")
        cot = m.get("cot_self_reflection")
        tool_calls = m.get("tool_calls") or []

        if role == "system":
            flush_pending_calls()
            if content:
                system_parts.append(content)
            i += 1
            continue

        if role == "user":
            flush_pending_calls()
            conversations.append({"from": "human", "value": content or ""})
            i += 1
            continue

        if role == "tool":
            if pending_function_calls:
                conversations.append({"from": "function_call", "value": pending_function_calls.pop(0)})
            conversations.append({"from": "observation", "value": content or ""})
            i += 1
            continue

        if role == "assistant":
            # assistant step: tool calls
            if tool_calls:
                payloads: List[Dict[str, Any]] = []
                for tc in tool_calls:
                    fn = (tc or {}).get("function", {}) if isinstance(tc, dict) else {}
                    name = fn.get("name")
                    args = parse_arguments(fn.get("arguments"))

                    payload: Dict[str, Any] = {"name": name, "arguments": args}
                    if isinstance(tc, dict) and tc.get("id"):
                        payload["id"] = tc.get("id")

                    payloads.append(payload)

                for idx_call, payload in enumerate(payloads):
                    value_parts: List[str] = []
                    if idx_call == 0 and put_step_thought_in_function_call and cot:
                        value_parts.append(f"<think>\n{cot}\n</think>\n\n")
                    value_parts.append(build_function_call_json(payload))
                    pending_function_calls.append("".join(value_parts))

                i += 1
                continue

            # assistant normal/final response
            flush_pending_calls()
            text = content or ""
            if put_final_thought_in_gpt and cot:
                value = f"<think>\n{cot}\n</think>\n\n{text}".strip()
            else:
                value = text
            conversations.append({"from": "gpt", "value": value})
            i += 1
            continue

        # fallback for unknown roles
        flush_pending_calls()
        conversations.append({"from": "observation", "value": "" if content is None else str(content)})
        i += 1

    flush_pending_calls()

    system_text = "\n\n".join(system_parts).strip() if system_parts else None

    out: Dict[str, Any] = {
        "conversations": conversations,
        "system": system_text,
        "meta": {
            "suite_name": example.get("suite_name"),
            "trajectory": example.get("trajectory"),
        },
    }
    if source_path is not None:
        out["meta"]["source_path"] = source_path
    return out


def load_examples_from_path(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    examples: List[Dict[str, Any]] = []
    for ex in ensure_list(raw):
        if isinstance(ex, dict):
            examples.append(ex)
        else:
            raise TypeError(f"Expected dict example in {path}, got {type(ex).__name__}")
    return examples


def iter_input_files(input_dir: str, pattern: str, recursive: bool) -> List[str]:
    matches: List[str] = []
    if recursive:
        for root, _, files in os.walk(input_dir):
            for fn in files:
                if fnmatch.fnmatch(fn, pattern):
                    matches.append(os.path.join(root, fn))
    else:
        for fn in os.listdir(input_dir):
            p = os.path.join(input_dir, fn)
            if os.path.isfile(p) and fnmatch.fnmatch(fn, pattern):
                matches.append(p)
    matches.sort()
    return matches


def main() -> None:
    ap = argparse.ArgumentParser()
    mx = ap.add_mutually_exclusive_group(required=True)
    mx.add_argument("--input", "-i", help="Path to input trajectory JSON (single dict or list).")
    mx.add_argument(
        "--input-dir",
        help="Directory containing many trajectory JSON files (searched recursively by default).",
    )
    ap.add_argument(
        "--glob",
        default="*.json",
        help="Filename pattern inside --input-dir (default: *.json).",
    )
    ap.add_argument(
        "--no-recursive",
        action="store_true",
        help="Do NOT recurse into subdirectories when using --input-dir.",
    )
    ap.add_argument(
        "--output",
        "-o",
        required=True,
        help="Path to output ShareGPT JSON, e.g. data/cot_sharegpt.json",
    )
    ap.add_argument(
        "--dataset-name",
        default="cot_sharegpt",
        help="Dataset key name used in dataset_info.json (default: cot_sharegpt)",
    )
    ap.add_argument(
        "--no-step-thought-in-function-call",
        action="store_true",
        help="Do NOT attach step-level cot_self_reflection into function_call payload as `thought`.",
    )
    ap.add_argument(
        "--no-final-thought-in-gpt",
        action="store_true",
        help="Do NOT wrap final cot_self_reflection into <think>...</think> in gpt messages.",
    )
    args = ap.parse_args()

    put_step_thought = not args.no_step_thought_in_function_call
    put_final_thought = not args.no_final_thought_in_gpt

    # Load inputs
    loaded: List[Tuple[Dict[str, Any], Optional[str]]] = []
    if args.input:
        for ex in load_examples_from_path(args.input):
            loaded.append((ex, args.input))
    else:
        input_files = iter_input_files(args.input_dir, pattern=args.glob, recursive=not args.no_recursive)
        if not input_files:
            raise SystemExit(f"No input files found in {args.input_dir} (glob={args.glob}, recursive={not args.no_recursive}).")
        for p in input_files:
            for ex in load_examples_from_path(p):
                loaded.append((ex, p))

    converted = []
    for ex, src in loaded:
        converted.append(
            convert_one(
                ex,
                put_step_thought_in_function_call=put_step_thought,
                put_final_thought_in_gpt=put_final_thought,
                source_path=src,
            )
        )

    # write dataset
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)

    # print dataset_info snippet for manual merge
    dataset_info = {
        args.dataset_name: {
            "file_name": args.output.split("/")[-1],
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "system": "system",
                "tools": "tools",
            },
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "human",
                "assistant_tag": "gpt",
                "system_tag": "system",
                "function_tag": "function_call",
                "observation_tag": "observation",
            },
        }
    }

    print(f"[OK] Wrote ShareGPT dataset: {args.output}")
    print(f"[Info] dataset_info snippet for '{args.dataset_name}':")
    print(json.dumps(dataset_info, ensure_ascii=False, indent=2))
    print(f"[Hint] Merge this snippet into LLaMA-Factory/data/dataset_info.json under key '{args.dataset_name}'.")


if __name__ == "__main__":
    main()
