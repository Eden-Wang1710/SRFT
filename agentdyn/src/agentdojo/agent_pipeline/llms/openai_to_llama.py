from typing import List, Dict, Any, Optional
import json
import datetime as _dt

LLAMA31_TOK = {
    "BEGIN": "<|begin_of_text|>",
    "EOT": "<|eot_id|>",
    "START": "<|start_header_id|>",
    "END": "<|end_header_id|>",
}

# ---- 新的通用取文本工具：支持 str / list[dict] / None ----
def _coerce_text(content: Any) -> str:
    """
    将 OpenAI 风格的 message.content 统一转换成文本：
      - str -> 原样 strip
      - list[{"type":"text","text":...}, ...] -> 拼接
      - 其它（None / 空 / 未知结构）-> ""
    遇到非 text 的 block（如 image_url）这里简单跳过/占位说明，可按需扩展。
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for seg in content:
            # 容错：seg 可能不是 dict
            if isinstance(seg, dict):
                if seg.get("type") == "text":
                    parts.append((seg.get("text") or "").strip())
                else:
                    # 其它类型你也可以改成保留占位文本
                    parts.append(f"[{seg.get('type','unknown')} omitted]")
            else:
                # 遇到意外类型，尽量转成字符串
                parts.append(str(seg).strip())
        return "\n".join([p for p in parts if p]).strip()
    # 兜底：尝试转字符串
    return str(content).strip()

def _role_header(role: str) -> str:
    """developer 视为 system，其它按原样（system/user/assistant/tool）。"""
    return "system" if role == "developer" else role

def _maybe_json(s: str):
    try:
        return json.loads(s)
    except Exception:
        return None

def _render_tools_schema_for_user(tools: List[Dict[str, Any]]) -> str:
    return json.dumps(tools, ensure_ascii=False, indent=4)

def _default_user_protocol_block() -> str:
    return (
        "Given the following functions, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.\n\n"
        'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}. Do not use variables.\n\n'
    )

def build_llama31_tool_prompt_from_messages(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    *,
    inject_protocol_and_tools_into_first_user: bool = True,
    add_dates_header: bool = False,
    cutting_knowledge_date: str = "December 2023",
    today_date_text: Optional[str] = None,
    extra_system_preamble: Optional[str] = (
        "You are a helpful assistant with tool calling capabilities. "
        "When you receive a tool call response, use the output to format an answer to the original user question."
    ),
) -> str:
    B, EOT, S, E = LLAMA31_TOK["BEGIN"], LLAMA31_TOK["EOT"], LLAMA31_TOK["START"], LLAMA31_TOK["END"]

    # 1) developer -> system 聚合
    sys_chunks: List[str] = []
    linear_msgs: List[Dict[str, Any]] = []
    for m in messages:
        role = m.get("role", "")
        role_hdr = _role_header(role)
        if role == "developer":
            text = _coerce_text(m.get("content"))
            if text:
                sys_chunks.append(text)
        else:
            linear_msgs.append(m | {"role": role_hdr})

    # 2) system 段
    system_lines = []
    if add_dates_header:
        today_str = today_date_text or _dt.datetime.now().strftime("%d %B %Y")
        system_lines.append(f"Cutting Knowledge Date: {cutting_knowledge_date}")
        system_lines.append(f"Today Date: {today_str}")
        system_lines.append("")
    if extra_system_preamble:
        system_lines.append(extra_system_preamble.strip())
    if sys_chunks:
        if system_lines:
            system_lines.append("")
        system_lines.append("\n\n".join(sys_chunks))
    system_text = "\n".join([ln for ln in system_lines if ln is not None]).strip()

    parts = [B]
    if system_text:
        parts.append(f"{S}system{E}\n{system_text}\n{EOT}")

    # 3) 在第一个 user 前注入 协议+tools（可选）
    protocol_injected = False

    # 4) 写入历史
    for m in linear_msgs:
        role = m.get("role")
        if role == "user":
            if (not protocol_injected) and tools and inject_protocol_and_tools_into_first_user:
                proto = _default_user_protocol_block()
                tools_json = _render_tools_schema_for_user(tools)
                injected_user_text = f"{proto}\n\n{tools_json}"
                parts.append(f"{S}user{E}\n{injected_user_text}\n{EOT}")
                protocol_injected = True

            user_text = _coerce_text(m.get("content"))
            if user_text:
                parts.append(f"{S}user{E}\n{user_text}\n{EOT}")

        elif role == "assistant":
            tool_calls = (m.get("tool_calls") or [])
            assistant_text = _coerce_text(m.get("content"))

            # (a) 普通回答
            if assistant_text:
                parts.append(f"{S}assistant{E}\n{assistant_text}\n{EOT}")

            # (b) 工具调用 -> 一行 JSON {"name":..., "parameters": {...}}
            for call in tool_calls:
                if call.get("type") == "function" and call.get("function"):
                    fn = call["function"]
                    name = fn.get("name")
                    args_raw = fn.get("arguments", "{}")

                    args = _maybe_json(args_raw)
                    if args is None:
                        try:
                            # 尝试替换单引号
                            args = json.loads(args_raw.replace("'", '"'))
                        except Exception:
                            args = {"_raw_arguments": args_raw}

                    asst_line = json.dumps({"name": name, "parameters": args}, ensure_ascii=False)
                    parts.append(f"{S}assistant{E}\n{asst_line}\n{EOT}")

        elif role == "tool":
            # 工具返回：ipython 段；content 可能是 str 或 list
            tool_text = _coerce_text(m.get("content"))
            parsed = _maybe_json(tool_text)
            if parsed is None:
                payload = json.dumps({"output": tool_text}, ensure_ascii=False)
            else:
                payload = json.dumps(parsed, ensure_ascii=False)

            parts.append(f"{S}ipython{E}\n{payload}\n{EOT}")

        elif role == "system":
            sys_text2 = _coerce_text(m.get("content"))
            if sys_text2:
                parts.append(f"{S}system{E}\n{sys_text2}\n{EOT}")

        else:
            # 未知角色 -> 当作 user
            fallback = _coerce_text(m.get("content"))
            if fallback:
                parts.append(f"{S}user{E}\n{fallback}\n{EOT}")

    # 5) 结尾补 assistant 起始
    parts.append(f"{S}assistant{E}\n")
    return "".join(parts)
