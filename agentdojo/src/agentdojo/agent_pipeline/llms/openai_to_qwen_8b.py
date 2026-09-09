# openai_to_qwen.py
from typing import List, Dict, Any, Optional
import json

def _coerce_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for seg in content:
            if isinstance(seg, dict) and seg.get("type") == "text":
                parts.append((seg.get("text") or "").strip())
        return "\n".join([p for p in parts if p]).strip()
    return str(content).strip()

def _maybe_json(s: str):
    try:
        return json.loads(s)
    except Exception:
        return None


def openai_messages_to_qwen_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    OpenAI 风格 -> Qwen 消息（含 <tool_call>/<tool_response> 文本），
    不负责注入 <tools>；交给 apply_chat_template(tools=...) 注入。
    """
    out: List[Dict[str, str]] = []
    pending_tool_resps: List[str] = []

    def flush_tool_resps():
        nonlocal out, pending_tool_resps
        if pending_tool_resps:
            body = "\n".join(f"<tool_response>\n{r}\n</tool_response>" for r in pending_tool_resps)
            out.append({"role": "user", "content": body})
            pending_tool_resps = []

    for m in messages:
        role = m.get("role")
        if role in ("developer", "system"):
            txt = _coerce_text(m.get("content"))
            if txt:
                flush_tool_resps()
                out.append({"role": "system", "content": txt})

        elif role == "user":
            txt = _coerce_text(m.get("content"))
            if txt:
                flush_tool_resps()
                out.append({"role": "user", "content": txt})

        elif role == "assistant":
            txt = _coerce_text(m.get("content"))
            calls = m.get("tool_calls") or []

            if txt:
                flush_tool_resps()
                out.append({"role": "assistant", "content": txt})

            if calls:
                blocks = []
                for c in calls:
                    if c.get("type") == "function" and c.get("function"):
                        fn = c["function"]
                        name = fn.get("name")
                        args_raw = fn.get("arguments", "{}")
                        args = _maybe_json(args_raw)
                        if args is None:
                            try:
                                args = json.loads(args_raw.replace("'", '"'))
                            except Exception:
                                args = {"_raw_arguments": args_raw}
                        blocks.append(
                            "<tool_call>\n" +
                            json.dumps({"name": name, "arguments": args}, ensure_ascii=False) +
                            "\n</tool_call>"
                        )
                if blocks:
                    flush_tool_resps()
                    out.append({"role": "assistant", "content": "\n".join(blocks)})

        elif role == "tool":
            txt = _coerce_text(m.get("content"))
            parsed = _maybe_json(txt)
            payload = parsed if parsed is not None else {"output": txt}
            pending_tool_resps.append(json.dumps(payload, ensure_ascii=False))

        else:
            txt = _coerce_text(m.get("content"))
            if txt:
                flush_tool_resps()
                out.append({"role": "user", "content": txt})

    flush_tool_resps()
    return out
