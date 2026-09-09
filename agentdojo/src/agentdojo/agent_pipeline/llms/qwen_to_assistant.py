# qwen_to_assistant.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import re
import uuid
from typing import Optional, List

from agentdojo.types import ChatAssistantMessage, text_content_block_from_string
from agentdojo.functions_runtime import FunctionCall

__all__ = ["qwen_text_to_assistant_message"]

# 清洗 ChatML/Llama 标记
_SPECIAL_MARKERS_RE = re.compile(
    r"<\|(?:begin_of_text|eot_id|eom_id|start_header_id|end_header_id|im_start|im_end)\|>",
    re.IGNORECASE,
)
# 三引号代码块
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
# 官方 <tool_call> ... </tool_call>（允许跨行）
_TOOL_CALL_BLOCK_RE = re.compile(r"<\s*tool_call\s*>\s*(\{.*?\})\s*<\s*/\s*tool_call\s*>", re.DOTALL | re.IGNORECASE)

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

def qwen_text_to_assistant_message(text: str, *, keep_preamble: bool = False) -> ChatAssistantMessage:
    """
    Qwen 第一轮 assistant 文本 -> ChatAssistantMessage
    优先解析官方 <tool_call> 块；无则兜底解析全文首个 JSON 调用；再无则作为自然语言。
    """
    cleaned = _strip_special_markers(text)
    if not cleaned:
        return ChatAssistantMessage(role="assistant", content=[text_content_block_from_string("")], tool_calls=None)

    # 1) 解析 <tool_call> ... </tool_call>（支持多次调用）
    tool_calls: List[FunctionCall] = []
    for m in _TOOL_CALL_BLOCK_RE.finditer(cleaned):
        payload_raw = _unwrap_code_fence(m.group(1).strip())
        data = _loose_json_loads(payload_raw)
        if isinstance(data, dict) and "name" in data:
            args = data.get("arguments", data.get("parameters", {}))
            if not isinstance(args, dict):
                args = _loose_json_loads(str(args)) or {"_raw": str(args)}
            tool_calls.append(FunctionCall(function=data["name"], args=args, id="call_" + uuid.uuid4().hex[:24]))

    if tool_calls:
        if keep_preamble:
            preamble_end = cleaned.find("<tool_call")
            preamble = cleaned[:preamble_end].strip() if preamble_end != -1 else ""
            content = [text_content_block_from_string(preamble)] if preamble else None
        else:
            content = None
        return ChatAssistantMessage(role="assistant", content=content, tool_calls=tool_calls)

    # 2) 兜底：全文抓第一个 JSON（兼容 {"name","arguments"} / {"name","parameters"} / {"name","args"}）
    candidate = _unwrap_code_fence(cleaned)
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
                content = [text_content_block_from_string(preamble)] if preamble else None
            else:
                content = None
            return ChatAssistantMessage(role="assistant", content=content, tool_calls=[fc])

    # 3) 普通文本
    return ChatAssistantMessage(role="assistant", content=[text_content_block_from_string(cleaned)], tool_calls=None)
