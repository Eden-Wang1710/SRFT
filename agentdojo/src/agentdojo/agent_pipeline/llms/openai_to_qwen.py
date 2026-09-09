# openai_to_qwen.py
from typing import List, Dict, Any, Optional
import json

QWEN_TOK = {"START": "<|im_start|>", "END": "<|im_end|>"}

def _coerce_text(content: Any) -> str:
    """将 OpenAI 风格 content 统一成纯文本：支持 str / list[{'type':'text','text':...}] / 其它。"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for seg in content:
            if isinstance(seg, dict) and seg.get("type") == "text":
                parts.append((seg.get("text") or "").strip())
            else:
                parts.append(f"[{getattr(seg, 'type', 'unknown')} omitted]")
        return "\n".join([p for p in parts if p]).strip()
    return str(content).strip()

def _maybe_json(s: str):
    try:
        return json.loads(s)
    except Exception:
        return None

def _render_tools_block(tools: Optional[List[Dict[str, Any]]]) -> str:
    """
    官方模板需要把函数签名 JSON Schema 放入 <tools> ... </tools>。
    这里从 OpenAI 风格 tools 中提取每个条目的 .function 字段。
    """
    if not tools:
        return "<tools>\n</tools>"
    chunks = []
    for t in tools:
        fn = t.get("function") or {}
        chunks.append(json.dumps(fn, ensure_ascii=False))
    return "<tools>\n" + "\n".join(chunks) + "\n</tools>"

def _system_instruction_block() -> str:
    """
    精简版官方 system 指令（不含日期/知识截止说明）。
    """
    lines = [
        "# Tools",
        "",
        "You may call one or more functions to assist with the user query.",
        "",
        "You are provided with function signatures within <tools></tools> XML tags:",
        # <tools> ... </tools> 占位由外层拼接
        "",
        "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:",
        "<tool_call>",
        '{"name": <function-name>, "arguments": <args-json-object>}',
        "</tool_call>",
    ]
    return "\n".join(lines).strip()

def build_qwen_tool_prompt_from_messages(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    生成与 Qwen 官方一致的 Tool Calling ChatML：
      <|im_start|>system
      # Tools
      ...
      <tools>
        {function 1 schema}
        {function 2 schema}
      </tools>
      <|im_end|>
      <|im_start|>user
      ...
      <|im_end|>
      <|im_start|>assistant
      <tool_call> {...} </tool_call>
      ...
      <|im_end|>
      <|im_start|>user
      <tool_response> {...} </tool_response>
      ...
      <|im_end|>
      <|im_start|>assistant
      （留空供模型续写）
    """
    S, E = QWEN_TOK["START"], QWEN_TOK["END"]

    # 1) 收集 developer/system 文本
    sys_texts: List[str] = []
    linear_msgs: List[Dict[str, Any]] = []
    for m in messages:
        role = m.get("role")
        if role in ("developer", "system"):
            t = _coerce_text(m.get("content"))
            if t:
                sys_texts.append(t)
        else:
            linear_msgs.append(m)

    # 2) system 段：官方指令 + <tools> 块 + 任何额外 system/developer 文本（附在末尾）
    sys_block = _system_instruction_block()
    tools_block = _render_tools_block(tools)
    system_full = f"{sys_block}\n{tools_block}"
    if sys_texts:
        system_full = f"{system_full}\n\n" + "\n\n".join(sys_texts)
    parts: List[str] = [f"{S}system\n{system_full}{E}"]

    # 3) 遍历历史，按照官方样式拼接
    buffer_tool_responses: List[str] = []

    def _flush_tool_responses_if_any():
        nonlocal parts, buffer_tool_responses, S, E
        if buffer_tool_responses:
            user_body = "\n".join(f"<tool_response>\n{r}\n</tool_response>" for r in buffer_tool_responses)
            parts.append(f"{S}user\n{user_body}{E}")
            buffer_tool_responses = []

    for m in linear_msgs:
        role = m.get("role")
        if role == "user":
            _flush_tool_responses_if_any()
            txt = _coerce_text(m.get("content"))
            if txt:
                parts.append(f"{S}user\n{txt}{E}")

        elif role == "assistant":
            _flush_tool_responses_if_any()
            txt = _coerce_text(m.get("content"))
            tool_calls = m.get("tool_calls") or []

            if txt:
                parts.append(f"{S}assistant\n{txt}{E}")

            if tool_calls:
                call_lines = []
                for call in tool_calls:
                    if call.get("type") == "function" and call.get("function"):
                        fn = call["function"]
                        name = fn.get("name")
                        args_raw = fn.get("arguments", "{}")
                        args = _maybe_json(args_raw)
                        if args is None:
                            try:
                                args = json.loads(args_raw.replace("'", '"'))
                            except Exception:
                                args = {"_raw_arguments": args_raw}
                        payload = json.dumps({"name": name, "arguments": args}, ensure_ascii=False)
                        call_lines.append(f"<tool_call>\n{payload}\n</tool_call>")
                if call_lines:
                    parts.append(f"{S}assistant\n" + "\n".join(call_lines) + f"{E}")

        elif role == "tool":
            # 官方示例：把工具返回合并为一轮 user，包含多个 <tool_response>
            tool_txt = _coerce_text(m.get("content"))
            parsed = _maybe_json(tool_txt)
            payload = parsed if parsed is not None else {"output": tool_txt}
            buffer_tool_responses.append(json.dumps(payload, ensure_ascii=False))

        else:
            # 兜底：按 user 处理
            _flush_tool_responses_if_any()
            txt = _coerce_text(m.get("content"))
            if txt:
                parts.append(f"{S}user\n{txt}{E}")

    _flush_tool_responses_if_any()

    # 4) 让模型继续
    parts.append(f"{S}assistant\n")
    return "".join(parts)
