import json
import re
import uuid
from typing import Optional

from agentdojo.types import (
    ChatAssistantMessage,
    MessageContentBlock,
    text_content_block_from_string,
)
from agentdojo.functions_runtime import FunctionCall  # 已定义: FunctionCall(function: str, args: dict, id: str, ...)

# ---- 轻量工具 ----

_SPECIAL_MARKERS_RE = re.compile(
    r"<\|(?:begin_of_text|eot_id|eom_id|start_header_id|end_header_id)\|>"
)

_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

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
    # 常见“松 JSON”修复：单引号 -> 双引号
    try:
        return json.loads(s.replace("'", '"'))
    except Exception:
        return None

def _find_first_json_object(s: str) -> Optional[str]:
    """
    从任意文本中抓取第一个完整 JSON 对象子串（括号匹配，处理字符串/转义）。
    """
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
                        return s[i:j+1]
        i = s.find("{", i + 1)
    return None

# ---- 主函数：Llama3.1 文本 -> ChatAssistantMessage ----

def llama31_text_to_assistant_message(text: str) -> ChatAssistantMessage:
    """
    将 Llama-3.1 的第一轮 assistant 解码文本转成 ChatAssistantMessage。
    - 若文本中包含 JSON 工具调用（{"name":..., "parameters"/"args": {...}}），返回 tool_calls 列表；
    - 否则返回自然语言 content。
    """
    cleaned = _strip_special_markers(text)
    if not cleaned:
        return ChatAssistantMessage(
            role="assistant",
            content=[text_content_block_from_string("")],
            tool_calls=None,
        )

    # 去掉 ```json ...``` 外壳；全文抓第一个 JSON 对象
    candidate = _unwrap_code_fence(cleaned)
    json_sub = _find_first_json_object(candidate)

    if json_sub:
        data = _loose_json_loads(json_sub)
        if isinstance(data, dict) and "name" in data and ("parameters" in data or "args" in data):
            name = data["name"]
            params = data.get("parameters", data.get("args", {}))
            if not isinstance(params, dict):
                params = _loose_json_loads(str(params)) or {"_raw": str(params)}

            # 与 _openai_to_tool_call 一致：FunctionCall(function, args, id)
            fc = FunctionCall(
                function=name,
                args=params,
                id="call_" + uuid.uuid4().hex[:24],
            )
            return ChatAssistantMessage(
                role="assistant",
                content=None,
                tool_calls=[fc],
            )

    # 没有工具调用 → 当成自然语言
    return ChatAssistantMessage(
        role="assistant",
        content=[text_content_block_from_string(cleaned)],
        tool_calls=None,
    )
