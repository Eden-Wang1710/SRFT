#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


THINK_RE = re.compile(r"<\s*think\s*>(.*?)<\s*/\s*think\s*>", re.DOTALL | re.IGNORECASE)


def parse_arguments(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return {}
        try:
            return json.loads(text)
        except Exception:
            return {"_raw": text}
    return {"_raw": str(raw)}


def normalize_tool_calls_from_openai_style(tool_calls: Any) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for tc in tool_calls or []:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") or {}
        if not isinstance(fn, dict):
            fn = {}
        normalized.append(
            {
                "name": fn.get("name"),
                "arguments": parse_arguments(fn.get("arguments")),
            }
        )
    return normalized


def normalize_tool_calls_from_qwen_style(tool_calls: Any) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for tc in tool_calls or []:
        if not isinstance(tc, dict):
            continue
        normalized.append(
            {
                "name": tc.get("function"),
                "arguments": parse_arguments(tc.get("args")),
            }
        )
    return normalized


def calls_equal(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> bool:
    return json.dumps(a, ensure_ascii=False, sort_keys=True) == json.dumps(
        b, ensure_ascii=False, sort_keys=True
    )


def build_function_call_value(
    *,
    thought: str | None,
    calls: list[dict[str, Any]],
) -> str:
    chunks: list[str] = []
    if thought:
        chunks.append(f"<think>\n{thought.strip()}\n</think>")
    call_payload: Any = calls[0] if len(calls) == 1 else calls
    chunks.append(json.dumps(call_payload, ensure_ascii=False))
    return "\n\n".join(chunks).strip()


def extract_thought(raw_output: Any) -> str | None:
    if not isinstance(raw_output, str):
        return None
    match = THINK_RE.search(raw_output)
    if not match:
        return None
    thought = match.group(1).strip()
    return thought or None


def convert_context_messages(context_messages: list[dict[str, Any]]) -> tuple[str | None, list[dict[str, str]]]:
    system_parts: list[str] = []
    conversations: list[dict[str, str]] = []
    pending_function_calls: list[str] = []

    def flush_pending() -> None:
        while pending_function_calls:
            conversations.append({"from": "function_call", "value": pending_function_calls.pop(0)})

    for m in context_messages:
        role = m.get("role")
        content = m.get("content")

        if role == "system":
            flush_pending()
            if isinstance(content, str) and content:
                system_parts.append(content)
            continue

        if role == "user":
            flush_pending()
            conversations.append({"from": "human", "value": content or ""})
            continue

        if role == "assistant":
            tool_calls = normalize_tool_calls_from_openai_style(m.get("tool_calls") or [])
            if tool_calls:
                for c in tool_calls:
                    pending_function_calls.append(json.dumps(c, ensure_ascii=False))
            else:
                flush_pending()
                conversations.append({"from": "gpt", "value": content or ""})
            continue

        if role == "tool":
            if pending_function_calls:
                conversations.append({"from": "function_call", "value": pending_function_calls.pop(0)})
            conversations.append({"from": "observation", "value": content or ""})
            continue

        flush_pending()
        conversations.append({"from": "observation", "value": "" if content is None else str(content)})

    flush_pending()
    system_text = "\n\n".join(system_parts).strip() if system_parts else None
    return system_text, conversations


def build_pair_record(step_path: Path, payload: dict[str, Any]) -> list[dict[str, Any]]:
    context_messages = payload.get("context_messages") or []
    if not isinstance(context_messages, list):
        return []

    expert = payload.get("expert_assistant_message") or {}
    if not isinstance(expert, dict):
        return []

    expert_calls = normalize_tool_calls_from_openai_style(expert.get("tool_calls") or [])
    if not expert_calls:
        return []

    cot = payload.get("cot_self_reflection")
    cot_text = cot.strip() if isinstance(cot, str) and cot.strip() else None

    system_text, conversations = convert_context_messages(context_messages)
    chosen_value = build_function_call_value(
        thought=cot_text,
        calls=expert_calls,
    )

    out: list[dict[str, Any]] = []
    for sample in payload.get("qwen_samples") or []:
        if not isinstance(sample, dict):
            continue
        parsed_msg = sample.get("qwen_parsed_message") or {}
        if not isinstance(parsed_msg, dict):
            continue

        rejected_calls = normalize_tool_calls_from_qwen_style(parsed_msg.get("tool_calls") or [])
        if not rejected_calls:
            continue
        if calls_equal(expert_calls, rejected_calls):
            continue

        thought = extract_thought(sample.get("qwen_raw_output"))
        rejected_value = build_function_call_value(
            thought=thought,
            calls=rejected_calls,
        )

        out.append(
            {
                "conversations": conversations,
                "chosen": {"from": "function_call", "value": chosen_value},
                "rejected": {"from": "function_call", "value": rejected_value},
                "system": system_text,
                "meta": {
                    "source_file": str(step_path),
                    "assistant_message_index": payload.get("assistant_message_index"),
                    "sequence_index": sample.get("sequence_index"),
                },
            }
        )

    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build LLaMA-Factory ShareGPT DPO dataset from per-step toucan run files. "
            "For each qwen sample, keep only items whose tool calls differ from expert tool calls."
        )
    )
    parser.add_argument(
        "--run-root",
        type=str,
        default="agentdojo/runs/qwen3-8b-think-samples-toucan_hotel_v1",
        help="Root folder containing assistant_step_*.json.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="LLaMA-Factory/data/toucan_hotel_v1_dpo_toolcall.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="toucan_hotel_v1_dpo_toolcall",
        help="Dataset key for generated dataset_info snippet.",
    )
    args = parser.parse_args()

    run_root = Path(args.run_root)
    if not run_root.is_dir():
        raise SystemExit(f"Input folder does not exist: {run_root}")

    step_files = sorted(run_root.rglob("assistant_step_*.json"))
    if not step_files:
        raise SystemExit(f"No assistant_step_*.json found in: {run_root}")

    all_rows: list[dict[str, Any]] = []
    for step_file in step_files:
        try:
            payload = json.loads(step_file.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[warn] skip invalid json: {step_file} ({exc})")
            continue
        all_rows.extend(build_pair_record(step_file, payload))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    snippet = {
        args.dataset_name: {
            "file_name": out_path.name,
            "ranking": True,
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "chosen": "chosen",
                "rejected": "rejected",
                "system": "system",
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
    snippet_path = out_path.with_name(out_path.stem + "_dataset_info.json")
    snippet_path.write_text(json.dumps(snippet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[OK] step files: {len(step_files)}")
    print(f"[OK] dpo pairs: {len(all_rows)}")
    print(f"[OK] wrote: {out_path}")
    print(f"[OK] wrote: {snippet_path}")


if __name__ == "__main__":
    main()
