# qwen_to_assistant.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import re
import uuid
from typing import Optional, List

from agentdojo.types import ChatAssistantMessage, ThinkingContentBlock, text_content_block_from_string
from agentdojo.functions_runtime import FunctionCall

__all__ = ["qwen_text_to_assistant_message"]

# 兼容 Qwen/Llama 模板 token 的清洗
_SPECIAL_MARKERS_RE = re.compile(
    r"<\|(?:begin_of_text|eot_id|eom_id|start_header_id|end_header_id|im_start|im_end)\|>",
    re.IGNORECASE,
)

# 三引号代码块（```json ...``` 或 ``` ...```）
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

# 官方 tool-calling 块
_TOOL_CALL_BLOCK_RE = re.compile(
    r"<\s*tool_call\s*>\s*(.*?)\s*<\s*/\s*tool_call\s*>",
    re.DOTALL | re.IGNORECASE,
)

# think 模式思考块
_THINK_BLOCK_RE = re.compile(
    r"<\s*think\s*>(.*?)<\s*/\s*think\s*>",
    re.DOTALL | re.IGNORECASE,
)

def _strip_special_markers(s: str) -> str:
    return _SPECIAL_MARKERS_RE.sub("", s or "").strip()

def _unwrap_code_fence(s: str) -> str:
    m = _CODE_FENCE_RE.search(s)
    return m.group(1).strip() if m else s

def _loose_json_loads(s: str) -> Optional[dict]:
    try:
        return json.loads(s)
    except Exception:
        pass
    try:
        return json.loads(s.replace("'", '"'))
    except Exception:
        return None

def _find_first_json_object(s: str) -> Optional[str]:
    i = s.find("{")
    while i != -1:
        depth = 0
        in_str = False
        esc = False
        for j in range(i, len(s)):
            ch = s[j]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return s[i : j + 1]
        i = s.find("{", i + 1)
    return None

def _split_think_blocks(s: str) -> tuple[str | None, str]:
    """
    Extract think content and return (think, visible_text).
    - If </think> exists, keep only the visible output after the last </think>.
    - Otherwise, remove all <think> blocks from the full text.
    """
    lower = (s or "").lower()
    end_idx = lower.rfind("</think>")
    if end_idx != -1:
        prefix = s[: end_idx + len("</think>")]
        suffix = s[end_idx + len("</think>") :]
        thinks = [m.group(1).strip() for m in _THINK_BLOCK_RE.finditer(prefix)]
        think_text = "\n\n".join([t for t in thinks if t]) or None
        return think_text, (suffix or "").strip()

    thinks = [m.group(1).strip() for m in _THINK_BLOCK_RE.finditer(s or "")]
    think_text = "\n\n".join([t for t in thinks if t]) or None
    visible = _THINK_BLOCK_RE.sub("", s or "").strip()
    return think_text, visible

def qwen8b_text_to_assistant_message(text: str, *, keep_preamble: bool = False) -> ChatAssistantMessage:
    """
    将 Qwen 的第一轮 assistant 文本转为 ChatAssistantMessage。
    - 优先解析 <tool_call> ... </tool_call>（支持多次）
    - 无则兜底：全文首个 JSON（支持 {"name","arguments"} / {"name","parameters"} / {"name","args"}）
    - 再无则视为纯文本
    """
    cleaned = _strip_special_markers(text)
    think_text, visible = _split_think_blocks(cleaned)
    if not visible and not think_text:
        return ChatAssistantMessage(role="assistant",
                                    content=[text_content_block_from_string("")],
                                    tool_calls=None)

    # 1) 解析 <tool_call> ... </tool_call>
    tool_calls: List[FunctionCall] = []
    for m in _TOOL_CALL_BLOCK_RE.finditer(visible):
        inner = m.group(1).strip()
        inner = _unwrap_code_fence(inner)
        json_sub = _find_first_json_object(inner)   # ← 括号配对，防截断
        if not json_sub:
            continue
        data = _loose_json_loads(json_sub)
        if isinstance(data, dict) and "name" in data:
            args = data.get("arguments", data.get("parameters", {}))
            if not isinstance(args, dict):
                args = _loose_json_loads(str(args)) or {"_raw": str(args)}
            tool_calls.append(FunctionCall(
                function=data["name"],
                args=args,
                id="call_" + uuid.uuid4().hex[:24],
            ))
            
    if tool_calls:
        content_blocks = []
        if think_text:
            content_blocks.append(ThinkingContentBlock(type="thinking", content=think_text, id=None))
        if keep_preamble:
            preamble_end = visible.find("<tool_call")
            preamble = visible[:preamble_end].strip() if preamble_end != -1 else ""
            if preamble:
                content_blocks.append(text_content_block_from_string(preamble))
        return ChatAssistantMessage(role="assistant", content=(content_blocks or None), tool_calls=tool_calls)

    # 2) 兜底：抓全文首个 JSON 调用
    candidate = _unwrap_code_fence(visible)
    json_sub = _find_first_json_object(candidate)
    if json_sub:
        data = _loose_json_loads(json_sub)
        if isinstance(data, dict) and "name" in data and ("arguments" in data or "parameters" in data or "args" in data):
            args = data.get("arguments", data.get("parameters", data.get("args", {})))
            if not isinstance(args, dict):
                args = _loose_json_loads(str(args)) or {"_raw": str(args)}
            fc = FunctionCall(function=data["name"], args=args, id="call_" + uuid.uuid4().hex[:24])
            if keep_preamble:
                preamble = candidate[: candidate.find(json_sub)].strip()
                content_blocks = []
                if think_text:
                    content_blocks.append(ThinkingContentBlock(type="thinking", content=think_text, id=None))
                if preamble:
                    content_blocks.append(text_content_block_from_string(preamble))
                content = content_blocks or None
            else:
                content = [ThinkingContentBlock(type="thinking", content=think_text, id=None)] if think_text else None
            return ChatAssistantMessage(role="assistant", content=content, tool_calls=[fc])

    # 3) 普通文本
    content_blocks = []
    if think_text:
        content_blocks.append(ThinkingContentBlock(type="thinking", content=think_text, id=None))
    if visible:
        content_blocks.append(text_content_block_from_string(visible))
    return ChatAssistantMessage(role="assistant",
                                content=content_blocks,
                                tool_calls=None)
