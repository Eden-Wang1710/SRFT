"""
Sampling script that replays Claude trajectories through Qwen-3 8B (think mode).

For every assistant turn in the expert trajectory we feed the full context seen
so far into the local Qwen model, collect its output, and store it next to the
original conversation. This matches the "early experience" style setup used in
Meta's work and produces paired (expert, student) traces for SFT.
"""
# 该脚本将 Claude 的专家轨迹逐轮喂给本地 Qwen3-8B-Think，用于生成“学生”版本的响应

from __future__ import annotations

import copy
import json
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

import click
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from agentdojo.functions_runtime import Function, FunctionCall
from agentdojo.task_suite.load_suites import get_suite
from agentdojo.types import text_content_block_from_string

DEFAULT_MODEL_ID = "Qwen/Qwen3-8B"


def openai_messages_to_qwen_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    """
    Local copy of the converter used in the pipeline to map OpenAI-style logs to the
    text format expected by the Qwen chat template.
    """
    # 将 OpenAI 风格的结构化对话记录转换为 Qwen 模板所需的纯文本序列

    def _coerce_text(content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for element in content:
                if isinstance(element, dict) and element.get("type") == "text":
                    parts.append((element.get("text") or "").strip())
            return "\n".join([part for part in parts if part]).strip()
        return str(content).strip()

    def _maybe_json(value: str):
        try:
            return json.loads(value)
        except Exception:
            return None

    out: list[dict[str, str]] = []
    pending_tool_responses: list[str] = []

    def flush_tool_responses():
        nonlocal pending_tool_responses
        if pending_tool_responses:
            body = "\n".join(f"<tool_response>\n{response}\n</tool_response>" for response in pending_tool_responses)
            out.append({"role": "user", "content": body})
            pending_tool_responses = []

    for message in messages:
        role = message.get("role")
        if role in ("developer", "system"):
            text_value = _coerce_text(message.get("content"))
            if text_value:
                flush_tool_responses()
                out.append({"role": "system", "content": text_value})
        elif role == "user":
            text_value = _coerce_text(message.get("content"))
            if text_value:
                flush_tool_responses()
                out.append({"role": "user", "content": text_value})
        elif role == "assistant":
            text_value = _coerce_text(message.get("content"))
            tool_calls = message.get("tool_calls") or []
            if text_value:
                flush_tool_responses()
                out.append({"role": "assistant", "content": text_value})
            if tool_calls:
                blocks = []
                for call in tool_calls:
                    if call.get("type") == "function" and call.get("function"):
                        function = call["function"]
                        name = function.get("name")
                        args_raw = function.get("arguments", "{}")
                        args = _maybe_json(args_raw)
                        if args is None:
                            try:
                                args = json.loads(args_raw.replace("'", '"'))
                            except Exception:
                                args = {"_raw_arguments": args_raw}
                        blocks.append(
                            "<tool_call>\n"
                            + json.dumps({"name": name, "arguments": args}, ensure_ascii=False)
                            + "\n</tool_call>"
                        )
                if blocks:
                    flush_tool_responses()
                    out.append({"role": "assistant", "content": "\n".join(blocks)})
        elif role == "tool":
            text_value = _coerce_text(message.get("content"))
            parsed = _maybe_json(text_value)
            payload = parsed if parsed is not None else {"output": text_value}
            pending_tool_responses.append(json.dumps(payload, ensure_ascii=False))
        else:
            text_value = _coerce_text(message.get("content"))
            if text_value:
                flush_tool_responses()
                out.append({"role": "user", "content": text_value})

    flush_tool_responses()
    return out


def qwen8b_text_to_assistant_message(text: str):
    """
    Local copy of the parser that strips the <think> blocks and extracts tool calls.
    """
    # 直接内联解析逻辑，避免依赖完整 pipeline，从而独立运行
    import re
    import uuid

    _SPECIAL_MARKERS_RE = re.compile(
        r"<\|(?:begin_of_text|eot_id|eom_id|start_header_id|end_header_id|im_start|im_end)\|>",
        re.IGNORECASE,
    )
    _CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
    _TOOL_CALL_BLOCK_RE = re.compile(r"<\s*tool_call\s*>\s*(.*?)\s*<\s*/\s*tool_call\s*>", re.DOTALL | re.IGNORECASE)
    _THINK_BLOCK_RE = re.compile(r"<\s*think\s*>.*?<\s*/\s*think\s*>", re.DOTALL | re.IGNORECASE)

    def _strip_special_markers(value: str) -> str:
        return _SPECIAL_MARKERS_RE.sub("", value or "").strip()

    def _unwrap_code_fence(value: str) -> str:
        match = _CODE_FENCE_RE.search(value)
        return match.group(1).strip() if match else value

    def _loose_json_loads(value: str) -> dict[str, Any] | None:
        try:
            return json.loads(value)
        except Exception:
            pass
        try:
            return json.loads(value.replace("'", '"'))
        except Exception:
            return None

    def _find_first_json_object(value: str) -> str | None:
        idx = value.find("{")
        while idx != -1:
            depth = 0
            in_string = False
            escape = False
            for jdx in range(idx, len(value)):
                ch = value[jdx]
                if in_string:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == '"':
                        in_string = False
                else:
                    if ch == '"':
                        in_string = True
                    elif ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            return value[idx : jdx + 1]
            idx = value.find("{", idx + 1)
        return None

    def _strip_think_blocks(value: str) -> str:
        end_idx = value.lower().rfind("</think>")
        if end_idx != -1:
            return value[end_idx + len("</think>") :].strip()
        return _THINK_BLOCK_RE.sub("", value).strip()

    cleaned = _strip_special_markers(text)
    cleaned = _strip_think_blocks(cleaned)
    if not cleaned:
        return {
            "role": "assistant",
            "content": [text_content_block_from_string("")],
            "tool_calls": None,
        }

    tool_calls: list[FunctionCall] = []
    for match in _TOOL_CALL_BLOCK_RE.finditer(cleaned):
        inner = match.group(1).strip()
        inner = _unwrap_code_fence(inner)
        json_payload = _find_first_json_object(inner)
        if not json_payload:
            continue
        data = _loose_json_loads(json_payload)
        if isinstance(data, dict) and "name" in data:
            args = data.get("arguments", data.get("parameters", {}))
            if not isinstance(args, dict):
                args = _loose_json_loads(str(args)) or {"_raw": str(args)}
            tool_calls.append(
                FunctionCall(function=data["name"], args=args, id="call_" + uuid.uuid4().hex[:24])
            )

    if tool_calls:
        preamble_end = cleaned.find("<tool_call")
        preamble = cleaned[:preamble_end].strip() if preamble_end != -1 else ""
        content = [text_content_block_from_string(preamble)] if preamble else None
        return {"role": "assistant", "content": content, "tool_calls": tool_calls}

    candidate = _unwrap_code_fence(cleaned)
    json_payload = _find_first_json_object(candidate)
    if json_payload:
        data = _loose_json_loads(json_payload)
        if isinstance(data, dict) and "name" in data and ("arguments" in data or "parameters" in data or "args" in data):
            args = data.get("arguments", data.get("parameters", data.get("args", {})))
            if not isinstance(args, dict):
                args = _loose_json_loads(str(args)) or {"_raw": str(args)}
            tool_call = FunctionCall(function=data["name"], args=args, id="call_" + uuid.uuid4().hex[:24])
            return {"role": "assistant", "content": None, "tool_calls": [tool_call]}

    return {"role": "assistant", "content": [text_content_block_from_string(cleaned)], "tool_calls": None}


def chat_completion_request(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    *,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int = 20,
    min_p: float | None = 0.0,
    thinking_budget: int = 512,
    second_pass_max_new_tokens: int = 512,
) -> str:
    """Run two-pass generation that mimics the agent pipeline's Think 模式."""
    device = next(model.parameters()).device
    model_inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    input_length = model_inputs.input_ids.size(-1)

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    think_end_id = tokenizer.convert_tokens_to_ids("</think>")

    gen_kwargs = dict(
        **model_inputs,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=thinking_budget,
    )
    try:
        if min_p is not None:
            gen_kwargs["min_p"] = float(min_p)
    except Exception:
        pass
    try:
        if min_p is not None and hasattr(model, "generation_config"):
            model.generation_config.min_p = float(min_p)
    except Exception:
        pass

    first_out = model.generate(**gen_kwargs)
    first_new_ids = first_out[0][input_length:].tolist()

    if im_end_id is not None and im_end_id in first_new_ids:
        # print("<|im_end|> token generated, stopping after first pass.")
        result = tokenizer.decode(first_out[0][input_length:], skip_special_tokens=True).strip()
        del first_out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result

    early_ids = None
    if think_end_id is None or think_end_id not in first_new_ids:
        # print("<think> block not closed, forcing early stop.")
        early_stop_text = (
            "\n\n Considering the limited time by the user, I have to give the solution based on the thinking directly now.\n"
            "</think>\n\n"
        )
        early_ids = tokenizer([early_stop_text], return_tensors="pt", return_attention_mask=False).input_ids.to(device)
        input_ids = torch.cat([first_out, early_ids], dim=-1)
    else:
        # print("</think> block closed, proceeding to second pass.")
        input_ids = first_out

    attn_mask = torch.ones_like(input_ids, dtype=torch.long)
    second_out = model.generate(
        input_ids=input_ids,
        attention_mask=attn_mask,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=second_pass_max_new_tokens,
    )

    full_new_ids = second_out[0][input_length:].tolist()
    result = tokenizer.decode(full_new_ids, skip_special_tokens=True).strip()

    del first_out, second_out, input_ids, attn_mask
    if early_ids is not None:
        del early_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def _serialize_function_tool(function: Function) -> dict[str, Any]:
    """Convert a Function definition into the format expected by Qwen templates."""
    # 将 FunctionsRuntime 的 schema 转成 chat_template 的工具格式
    return {
        "type": "function",
        "function": {
            "name": function.name,
            "description": function.description,
            "parameters": function.parameters.model_json_schema(),
        },
    }


def _normalize_content(content: Any) -> Any:
    # 将日志中的 content 统一本地结构，方便转换
    if content is None:
        return None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        normalized_segments = []
        for segment in content:
            if isinstance(segment, dict):
                segment_type = segment.get("type", "text")
                text_value = segment.get("text") or segment.get("content") or ""
                normalized_segments.append({"type": segment_type, "text": str(text_value)})
            else:
                normalized_segments.append({"type": "text", "text": str(segment)})
        return normalized_segments
    return str(content)


def _retarget_tool_content(content: Any) -> Any:
    """
    Replace hardcoded assistant name "Claude" in tool outputs with a neutral placeholder
    before feeding them back to Qwen.
    """
    def _replace_text(value: str) -> str:
        return value.replace("Claude", "AI Assistant")

    if content is None:
        return None
    if isinstance(content, str):
        return _replace_text(content)
    if isinstance(content, list):
        updated = []
        for segment in content:
            if isinstance(segment, dict):
                updated_segment = dict(segment)
                text_value = updated_segment.get("text")
                if isinstance(text_value, str):
                    updated_segment["text"] = _replace_text(text_value)
                updated.append(updated_segment)
            elif isinstance(segment, str):
                updated.append(_replace_text(segment))
            else:
                updated.append(segment)
        return updated
    return content


def _normalize_message(message: dict[str, Any]) -> dict[str, Any]:
    """Map stored log messages to the OpenAI-style format expected by the converter."""
    # Claude 轨迹的 message 字段可能是字符串或 content block，这里统一成 OpenAI 风格
    normalized: dict[str, Any] = {
        "role": message.get("role"),
        "content": _normalize_content(message.get("content")),
    }
    if message.get("role") == "assistant" and message.get("tool_calls"):
        tool_calls = []
        for tool_call in message["tool_calls"]:
            tool_calls.append(
                {
                    "id": tool_call.get("id"),
                    "type": "function",
                    "function": {
                        "name": tool_call.get("function"),
                        "arguments": json.dumps(tool_call.get("args", {}), ensure_ascii=False),
                    },
                }
            )
        normalized["tool_calls"] = tool_calls
    if message.get("role") == "tool":
        normalized["tool_call_id"] = message.get("tool_call_id")
        tool_info = message.get("tool_call") or {}
        normalized["name"] = tool_info.get("function")
        normalized["content"] = _retarget_tool_content(normalized.get("content"))
    return normalized


def _assistant_message_to_json(message) -> dict[str, Any]:
    # 把解析后的 ChatAssistantMessage 摊平为可 JSON 化的结构
    content = message.get("content")
    tool_calls = message.get("tool_calls")
    serialized_tool_calls = None
    if tool_calls:
        serialized_tool_calls = [tool_call.model_dump() for tool_call in tool_calls]
    return {
        "role": message.get("role"),
        "content": content,
        "tool_calls": serialized_tool_calls,
    }


class QwenThinkGenerator:
    """Thin wrapper around the HF model to keep sampling parameters together."""

    def __init__(
        self,
        model_id: str,
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float | None,
        thinking_budget: int,
        second_pass_max_new_tokens: int,
    ) -> None:
        if torch.cuda.is_available():
            # GPU 上优先使用 bfloat16，否则退回 float16
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32

        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto",
        )
        if self.tokenizer.pad_token_id is None:
            # 有些 tokenizer 没有 pad token，直接共用 eos
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.thinking_budget = thinking_budget
        self.second_pass_max_new_tokens = second_pass_max_new_tokens

    def generate(self, openai_messages: Sequence[dict[str, Any]], tools: Sequence[dict[str, Any]]):
        qwen_messages = openai_messages_to_qwen_messages(list(openai_messages))
        has_system = any(message.get("role") == "system" for message in qwen_messages)
        if not has_system:
            # 若日志里缺少 system，则补一个最小提示，确保模板合法
            qwen_messages = [{"role": "system", "content": "Use tools if needed."}] + qwen_messages

        prompt_text = self.tokenizer.apply_chat_template(
            qwen_messages,
            tools=list(tools) or None,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=True,
        )
        # print("============ Qwen Prompt ============")
        # print(prompt_text)
        raw_text = chat_completion_request(
            self.model,
            self.tokenizer,
            prompt_text,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
            thinking_budget=self.thinking_budget,
            second_pass_max_new_tokens=self.second_pass_max_new_tokens,
        )
        assistant_message = qwen8b_text_to_assistant_message(raw_text)
        return raw_text, assistant_message


@lru_cache(maxsize=None)
def _load_suite_tools(benchmark_version: str, suite_name: str) -> list[dict[str, Any]]:
    # 按套件加载工具列表并缓存，避免重复解析 YAML
    suite = get_suite(benchmark_version, suite_name)
    return [_serialize_function_tool(tool) for tool in suite.tools]


def _iter_trajectories(run_dir: Path, suites: tuple[str, ...]) -> Iterable[tuple[str, Path]]:
    # 遍历指定 run 目录下的所有 JSON 轨迹，可选择按 suite 过滤
    for suite_dir in sorted(run_dir.iterdir()):
        if not suite_dir.is_dir():
            continue
        suite_name = suite_dir.name
        if suites and suite_name not in suites:
            continue
        for json_path in sorted(suite_dir.rglob("*.json")):
            yield suite_name, json_path


def _build_qwen_samples(
    *,
    model: QwenThinkGenerator,
    tool_spec: Sequence[dict[str, Any]],
    normalized_messages: list[dict[str, Any]],
    original_messages: list[dict[str, Any]],
    num_sequences: int,
) -> list[dict[str, Any]]:
    # 针对每个 assistant 回复重放上下文，并缓存 Qwen 模型的输出
    samples: list[dict[str, Any]] = []
    def _strip_assistant_content(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        filtered = []
        for msg in messages:
            msg_copy = copy.deepcopy(msg)
            if msg_copy.get("role") == "assistant":
                msg_copy["content"] = None
            filtered.append(msg_copy)
        return filtered

    for idx, message in enumerate(original_messages):
        if message.get("role") != "assistant":
            continue
        context = normalized_messages[:idx]
        if not context:
            continue
        filtered_context = _strip_assistant_content(context)
        qwen_samples = []
        for sequence_index in range(num_sequences):
            raw_text, assistant_message = model.generate(filtered_context, tool_spec)
            qwen_samples.append(
                {
                    "sequence_index": sequence_index,
                    "qwen_raw_output": raw_text,
                    "qwen_parsed_message": _assistant_message_to_json(assistant_message),
                }
            )
        samples.append(
            {
                "assistant_message_index": idx,
                "context_message_count": len(context),
                "context_messages": filtered_context,
                "expert_assistant_message": copy.deepcopy(message),
                "qwen_samples": qwen_samples,
            }
        )
    return samples


def _should_sample_trajectory(payload: dict[str, Any]) -> bool:
    """Filter runs to only include clean user tasks or safe injection runs."""
    user_task_id = payload.get("user_task_id")
    injection_task_id = payload.get("injection_task_id")
    utility = payload.get("utility")
    security = payload.get("security")

    if not isinstance(user_task_id, str) or not user_task_id.startswith("user_task_"):
        return False

    if injection_task_id is None:
        return utility is True

    return utility is True and security is False


def _process_trajectory(
    suite_name: str,
    json_path: Path,
    generator: QwenThinkGenerator,
    expert_run_dir: Path,
    output_dir: Path,
    benchmark_version: str,
    sampling_params: dict[str, Any],
    num_sequences: int,
    overwrite: bool,
) -> tuple[int, int]:
    relative_path = json_path.relative_to(expert_run_dir)
    trajectory_output_dir = output_dir / relative_path.parent / relative_path.stem

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    user_task_id = data.get("user_task_id")
    injection_task_id = data.get("injection_task_id")
    click.echo(f"[Filter] suite={suite_name} user_task={user_task_id} injection_task={injection_task_id}")

    if not _should_sample_trajectory(data):
        click.echo(
            f"[Skip] suite={suite_name} user_task={user_task_id} injection_task={injection_task_id} did not pass filter"
        )
        return 0, 0

    click.echo(f"[Process] suite={suite_name} user_task={user_task_id} injection_task={injection_task_id}")

    if trajectory_output_dir.exists():
        if overwrite:
            shutil.rmtree(trajectory_output_dir)
        else:
            click.echo(f"[Skip-existing] {trajectory_output_dir} already exists. Use --overwrite to regenerate.")
            return 0, 0

    normalized_messages = [_normalize_message(message) for message in data.get("messages", [])]
    tool_spec = _load_suite_tools(benchmark_version, suite_name)
    samples = _build_qwen_samples(
        model=generator,
        tool_spec=tool_spec,
        normalized_messages=normalized_messages,
        original_messages=data.get("messages", []),
        num_sequences=num_sequences,
    )

    if not samples:
        click.echo(f"[Skip] suite={suite_name} user_task={user_task_id} produced no assistant turns.")
        return 0, 0

    trajectory_output_dir.mkdir(parents=True, exist_ok=True)
    base_sample_payload = {
        "suite_name": suite_name,
        "source_trajectory": str(relative_path),
        "qwen_model_id": generator.model_id,
        "qwen_sampling_params": sampling_params,
    }

    for sample in samples:
        sample_payload = {**base_sample_payload, **sample}
        sample_path = trajectory_output_dir / f"assistant_step_{sample['assistant_message_index']:03d}.json"
        _write_json(sample_path, sample_payload)

    click.echo(f"[{suite_name}] Saved {len(samples)} samples under {trajectory_output_dir}")
    return 1, len(samples)


def _write_json(output_path: Path, payload: dict[str, Any]) -> None:
    # 输出路径复用原目录层级，确保 SFT 数据可一一对应
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


@click.command()
@click.option(
    "--expert-run-dir",
    type=click.Path(path_type=Path),
    default=Path("runs/claude-3-5-sonnet-20241022"),
    show_default=True,
    help="Folder containing the expert Claude trajectories.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("runs/qwen3-8b-think-samples"),
    show_default=True,
    help="Where to store the sampled Qwen trajectories.",
)
@click.option(
    "--model-id",
    type=str,
    default=DEFAULT_MODEL_ID,
    show_default=True,
    help="Hugging Face model to load for Qwen think mode.",
)
@click.option(
    "--benchmark-version",
    type=str,
    default="v1.2.1",
    show_default=True,
    help="Benchmark version to use when loading suite tool specifications.",
)
@click.option(
    "--suite",
    "suites",
    multiple=True,
    help="Optional suite filter (can be provided multiple times).",
)
@click.option(
    "--temperature",
    type=float,
    default=1.0,
    show_default=True,
    help="Sampling temperature for Qwen generation.",
)
@click.option(
    "--top-p",
    type=float,
    default=0.95,
    show_default=True,
    help="Top-p for sampling.",
)
@click.option(
    "--top-k",
    type=int,
    default=20,
    show_default=True,
    help="Top-k for sampling.",
)
@click.option(
    "--min-p",
    type=float,
    default=0.0,
    show_default=True,
    help="Minimum probability mass for Qwen sampling (ignored if model does not support it).",
)
@click.option(
    "--thinking-budget",
    type=int,
    default=512,
    show_default=True,
    help="Maximum number of tokens for the first (thinking) pass.",
)
@click.option(
    "--second-pass-max-new-tokens",
    type=int,
    default=512,
    show_default=True,
    help="Maximum number of tokens for the second pass (final answer).",
)
@click.option(
    "--num-sequences",
    type=int,
    default=3,
    show_default=True,
    help="How many independent Qwen responses to sample for each assistant step.",
)
@click.option(
    "--overwrite/--skip-existing",
    default=False,
    show_default=True,
    help="Whether to overwrite files that already exist in the output directory.",
)
def main(
    expert_run_dir: Path,
    output_dir: Path,
    model_id: str,
    benchmark_version: str,
    suites: tuple[str, ...],
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    thinking_budget: int,
    second_pass_max_new_tokens: int,
    num_sequences: int,
    overwrite: bool,
):
    # 1. 统一输入输出目录为绝对路径，方便后续拼接
    expert_run_dir = expert_run_dir.resolve()
    output_dir = output_dir.resolve()
    # 2. 检查专家轨迹目录是否存在
    if not expert_run_dir.exists():
        raise click.ClickException(f"Expert run directory '{expert_run_dir}' does not exist.")

    click.echo(f"Loading Qwen model '{model_id}'...")
    generator = QwenThinkGenerator(
        model_id=model_id,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        thinking_budget=thinking_budget,
        second_pass_max_new_tokens=second_pass_max_new_tokens,
    )
    sampling_params = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "min_p": min_p,
        "thinking_budget": thinking_budget,
        "second_pass_max_new_tokens": second_pass_max_new_tokens,
        "num_sequences": num_sequences,
    }

    processed_files = 0
    sampled_steps = 0

    for suite_name, json_path in _iter_trajectories(expert_run_dir, suites):
        processed, steps = _process_trajectory(
            suite_name,
            json_path,
            generator,
            expert_run_dir,
            output_dir,
            benchmark_version,
            sampling_params,
            num_sequences,
            overwrite,
        )
        processed_files += processed
        sampled_steps += steps

    click.echo(f"Finished sampling {sampled_steps} assistant steps across {processed_files} trajectories.")


if __name__ == "__main__":
    main()
