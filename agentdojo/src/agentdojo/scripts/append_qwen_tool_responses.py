from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
from typing import Any

import click
import yaml
from pydantic import BaseModel

from agentdojo.functions_runtime import FunctionCall, FunctionReturnType, FunctionsRuntime
from agentdojo.task_suite.load_suites import get_suite
from agentdojo.types import (
    ChatAssistantMessage,
    ChatToolResultMessage,
    MessageContentBlock,
    get_text_content_as_str,
    text_content_block_from_string,
)


def _load_json(path: Path) -> dict[str, Any]:
    # 读取 JSON 文件，所有处理逻辑都基于它
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    # 统一的写文件方法，保持缩进与末尾换行
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def _coerce_arguments(raw_args: Any) -> dict[str, Any]:
    # 将各种形态的参数（dict/字符串/None）转成 dict
    if raw_args is None:
        return {}
    if isinstance(raw_args, dict):
        return dict(raw_args)
    if isinstance(raw_args, str):
        text = raw_args.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, dict):
                    return parsed
            except (ValueError, SyntaxError):
                pass
        # 无法解析就用兜底字段记录原始字符串
        return {"_raw_arguments": raw_args}
    raise ValueError(f"Unsupported tool argument type: {type(raw_args)}")


def _is_string_list(value: str) -> bool:
    # 查看字符串是否为 Python list 表达式，便于后续 literal_eval
    try:
        parsed = ast.literal_eval(value)
        return isinstance(parsed, list)
    except (ValueError, SyntaxError):
        return False


def _tool_result_to_str(result: FunctionReturnType) -> str:
    # 将工具返回值格式化成字符串（优先 YAML，方便阅读）
    if isinstance(result, BaseModel):
        return yaml.safe_dump(result.model_dump()).strip()
    if isinstance(result, list):
        cleaned: list[Any] = []
        for item in result:
            if isinstance(item, BaseModel):
                cleaned.append(item.model_dump())
            else:
                cleaned.append(item)
        return yaml.safe_dump(cleaned).strip()
    if isinstance(result, dict):
        return yaml.safe_dump(result).strip()
    return str(result)


def _deserialize_tool_call(raw_call: dict[str, Any]) -> FunctionCall:
    # 从 JSON 结构恢复 FunctionCall，兼容不同字段命名
    function_block = raw_call.get("function")
    if isinstance(function_block, dict):
        function_name = function_block.get("name")
        args = _coerce_arguments(function_block.get("arguments"))
    else:
        function_name = raw_call.get("function")
        args = _coerce_arguments(raw_call.get("args"))
    if function_name is None:
        raise ValueError("Tool call is missing a function name")
    placeholder_args = raw_call.get("placeholder_args")
    if placeholder_args is not None and not isinstance(placeholder_args, dict):
        placeholder_args = _coerce_arguments(placeholder_args)
    return FunctionCall(
        function=function_name,
        args=args,
        id=raw_call.get("id"),
        placeholder_args=placeholder_args,
    )


def _coerce_content(value: Any) -> list[MessageContentBlock] | None:
    # content 可能是 None/字符串/list，这里统一成 content block
    if value is None:
        return None
    if isinstance(value, list):
        return value  # assume already in content block format
    return [text_content_block_from_string(str(value))]


def _prepare_environment(suite_name: str, benchmark_version: str, source_data: dict[str, Any]):
    # 通过 suite 信息恢复运行环境并执行用户任务初始化
    suite = get_suite(benchmark_version, suite_name)
    injections = source_data.get("injections") or {}
    environment = suite.load_and_inject_default_environment(injections)
    user_task_id = source_data.get("user_task_id")
    if user_task_id:
        user_task = suite.get_user_task_by_id(user_task_id)
        environment = user_task.init_environment(environment)
    return suite, environment


def _clone_environment(environment):
    # 复制环境，确保每个 sample 的执行互不影响
    if hasattr(environment, "model_copy"):
        try:
            return environment.model_copy(deep=True)
        except Exception:
            pass
    return copy.deepcopy(environment)


def _replay_context_tool_calls(context_messages: list[dict[str, Any]], runtime: FunctionsRuntime, environment) -> None:
    # 为了重建环境状态，需要把上下文中所有工具调用重新执行一次
    for message in context_messages or []:
        if not isinstance(message, dict):
            continue
        if message.get("role") != "assistant":
            continue
        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            continue
        for raw_call in tool_calls:
            try:
                function_call = _deserialize_tool_call(raw_call)
            except ValueError as exc:
                click.echo(f"[warn] Skipping invalid context tool call: {exc}", err=True)
                continue
            # 直接在 runtime 中执行，还原环境副作用
            _, error = runtime.run_function(environment, function_call.function, function_call.args)
            if error:
                click.echo(
                    f"[warn] Context tool call {function_call.function} failed with: {error}",
                    err=True,
                )


def _assistant_message_from_payload(payload: dict[str, Any]) -> ChatAssistantMessage:
    # 将 qwen_parsed_message 字段封装成 ChatAssistantMessage
    raw_tool_calls = payload.get("tool_calls") or []
    tool_calls = None
    if raw_tool_calls:
        tool_calls = []
        for raw_call in raw_tool_calls:
            tool_calls.append(_deserialize_tool_call(raw_call))
    return ChatAssistantMessage(
        role="assistant",
        content=_coerce_content(payload.get("content")),
        tool_calls=tool_calls,
    )


def _execute_tool_calls(
    tool_calls: list[FunctionCall],
    runtime: FunctionsRuntime,
    environment,
) -> list[ChatToolResultMessage]:
    # 真正执行工具调用，并返回统一格式的工具响应
    tool_messages: list[ChatToolResultMessage] = []
    for call in tool_calls:
        # 先把类似 "['a']" 的字符串参数转为 list
        for arg_name, arg_value in list(call.args.items()):
            if isinstance(arg_value, str) and _is_string_list(arg_value):
                call.args[arg_name] = ast.literal_eval(arg_value)
        # 执行工具并捕获错误
        result, error = runtime.run_function(environment, call.function, call.args)
        formatted = _tool_result_to_str(result)
        tool_messages.append(
            ChatToolResultMessage(
                role="tool",
                content=[text_content_block_from_string(formatted)],
                tool_call_id=call.id,
                tool_call=call,
                error=error,
            )
        )
    return tool_messages


def _serialize_tool_messages(messages: list[ChatToolResultMessage]) -> list[dict[str, Any]]:
    # 将 ChatToolResultMessage 序列化为 JSON 可写的结构
    serialized: list[dict[str, Any]] = []
    for message in messages:
        serialized.append(
            {
                "role": message["role"],
                "tool_call_id": message["tool_call_id"],
                "tool_call": message["tool_call"].model_dump(),
                "content": get_text_content_as_str(message["content"]),
                "error": message["error"],
            }
        )
    return serialized


def _compute_tool_messages_for_payload(
    payload: dict[str, Any] | None,
    runtime: FunctionsRuntime,
    environment_template,
):
    # 针对单个 assistant 消息执行工具并返回序列化结果
    if not payload:
        return None
    assistant_message = _assistant_message_from_payload(payload)
    if not assistant_message["tool_calls"]:
        return None
    environment = _clone_environment(environment_template)
    tool_messages = _execute_tool_calls(assistant_message["tool_calls"], runtime, environment)
    if not tool_messages:
        return None
    return _serialize_tool_messages(tool_messages)


def _load_source_payload(cache: dict[Path, dict[str, Any]], path: Path) -> dict[str, Any]:
    # 为避免重复 IO，简单做个缓存
    if path not in cache:
        cache[path] = _load_json(path)
    return cache[path]


def _process_file(
    json_path: Path,
    *,
    benchmark_version: str,
    expert_run_dir: Path,
    overwrite: bool,
    source_cache: dict[Path, dict[str, Any]],
) -> bool:
    # 处理单个样本文件：验证前置条件 -> 重放工具 -> 执行并写回
    payload = _load_json(json_path)
    qwen_samples = payload.get("qwen_samples") or []
    single_qwen_message = payload.get("qwen_parsed_message")
    has_any_message = bool(qwen_samples) or single_qwen_message is not None
    if not has_any_message:
        return False
    if not qwen_samples and single_qwen_message and not single_qwen_message.get("tool_calls"):
        return False
    if qwen_samples:
        sample_has_calls = any((sample.get("qwen_parsed_message") or {}).get("tool_calls") for sample in qwen_samples)
        if not sample_has_calls:
            return False
    if not overwrite and not qwen_samples and payload.get("qwen_tool_messages"):
        return False

    suite_name = payload.get("suite_name")
    if not suite_name:
        click.echo(f"[warn] Missing suite name in {json_path}", err=True)
        return False

    source_relative = payload.get("source_trajectory")
    if not source_relative:
        click.echo(f"[warn] Missing source_trajectory in {json_path}", err=True)
        return False
    source_path = (expert_run_dir / source_relative).resolve()
    if not source_path.exists():
        click.echo(f"[warn] Source trajectory {source_path} not found for {json_path}", err=True)
        return False
    # 读取专家轨迹，用于恢复环境
    source_data = _load_source_payload(source_cache, source_path)

    suite, environment = _prepare_environment(suite_name, benchmark_version, source_data)
    runtime = FunctionsRuntime(suite.tools)
    # 重放上下文工具调用以同步环境
    _replay_context_tool_calls(payload.get("context_messages", []), runtime, environment)
    base_environment = _clone_environment(environment)

    file_changed = False

    if qwen_samples:
        for sample in qwen_samples:
            if not isinstance(sample, dict):
                continue
            if not overwrite and sample.get("qwen_tool_messages"):
                continue
            tool_messages = _compute_tool_messages_for_payload(sample.get("qwen_parsed_message"), runtime, base_environment)
            if tool_messages:
                sample["qwen_tool_messages"] = tool_messages
                file_changed = True

    if single_qwen_message:
        if overwrite or not payload.get("qwen_tool_messages"):
            tool_messages = _compute_tool_messages_for_payload(single_qwen_message, runtime, base_environment)
            if tool_messages:
                payload["qwen_tool_messages"] = tool_messages
                file_changed = True

    if file_changed:
        _write_json(json_path, payload)
    return file_changed


@click.command()
@click.option(
    "--run-dir",
    type=click.Path(path_type=Path),
    required=True,
    help="Directory that contains Qwen sample JSON files.",
)
@click.option(
    "--expert-run-dir",
    type=click.Path(path_type=Path),
    default=Path("runs/claude-3-5-sonnet-20241022"),
    show_default=True,
    help="Directory containing the original expert trajectories referenced by the samples.",
)
@click.option(
    "--benchmark-version",
    type=str,
    default="v1.2.1",
    show_default=True,
    help="Benchmark version used when looking up task suites.",
)
@click.option(
    "--overwrite/--skip-existing",
    default=False,
    show_default=True,
    help="Whether to overwrite existing qwen_tool_messages entries.",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Optional cap on how many files to process.",
)
def main(run_dir: Path, expert_run_dir: Path, benchmark_version: str, overwrite: bool, limit: int | None) -> None:
    # 命令行入口：遍历 run 目录并批量写入工具结果
    run_dir = run_dir.resolve()
    expert_run_dir = expert_run_dir.resolve()
    if not run_dir.exists():
        raise click.ClickException(f"Run directory {run_dir} does not exist")
    if not expert_run_dir.exists():
        raise click.ClickException(f"Expert run directory {expert_run_dir} does not exist")

    source_cache: dict[Path, dict[str, Any]] = {}

    processed = 0
    updated = 0
    for json_path in sorted(run_dir.rglob("*.json")):
        processed += 1
        changed = _process_file(
            json_path,
            benchmark_version=benchmark_version,
            expert_run_dir=expert_run_dir,
            overwrite=overwrite,
            source_cache=source_cache,
        )
        if changed:
            updated += 1
            click.echo(f"[update] {json_path}")
        # 达到限制数量后提前退出
        if limit is not None and updated >= limit:
            break

    click.echo(f"Processed {processed} files, updated {updated} with tool responses.")


if __name__ == "__main__":
    main()
