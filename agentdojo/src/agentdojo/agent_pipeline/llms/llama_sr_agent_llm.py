"""SR-Agent on Llama-3.1-8B-Instruct (and its untrained base) in the AgentDojo `local` tool format (2026-09-11).

Prompt construction lives in `llama_local_prompt.py`, shared with the training-data converter
(`multibase/convert_llama_local.py`), so the inference context is token-identical to the SFT context:
previous reflections stay in the history as `<think>…</think>` text, exactly as LLaMA-Factory's `llama3` template trains them.
The reflection is stored as a ThinkingContentBlock (logged, never scored); only the visible text is the answer AgentDojo checks.

Env: LLAMA_SR_AGENT_BASE_MODEL (default meta-llama/Llama-3.1-8B-Instruct), LLAMA_SR_AGENT_LORA_PATH (missing dir -> base model),
LLAMA_SR_AGENT_SYS_APPEND (1 = paper Fig. 5 append, default), LLAMA_SR_AGENT_MAX_NEW_TOKENS (1536),
LLAMA_SR_AGENT_TEMPERATURE / _TOP_P (0.6 / 0.9 = Llama-3.1 generation_config defaults).
"""

import os
import uuid
from collections.abc import Sequence
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.llms.llama_local_prompt import format_assistant, format_call, make_system_prompt, parse_output, render
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionCall, FunctionsRuntime
from agentdojo.types import ChatAssistantMessage, ChatMessage, ThinkingContentBlock, text_content_block_from_string

DEFAULT_BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _blocks(message: ChatMessage, kind: str) -> str:
    content = message.get("content") or []
    return "\n\n".join((b.get("content") or "").strip() for b in content if b.get("type") == kind).strip()


def _to_prompt_messages(messages: Sequence[ChatMessage], runtime: FunctionsRuntime, append: bool) -> list[dict]:
    tools = [
        {"name": f.name, "description": f.description, "parameters": f.parameters.model_json_schema()}
        for f in runtime.functions.values()
    ]
    out: list[dict] = []
    for m in messages:
        role = m["role"]
        if role == "system":
            out.append({"role": "system", "content": make_system_prompt(_blocks(m, "text"), tools, append=append)})
        elif role == "user":
            out.append({"role": "user", "content": _blocks(m, "text")})
        elif role == "assistant":
            think = _blocks(m, "thinking") or None
            body = _blocks(m, "text")
            calls = m.get("tool_calls") or []
            if calls:
                body = "\n\n".join(x for x in (body, format_call(calls[0].function, dict(calls[0].args))) if x)
            out.append({"role": "assistant", "content": format_assistant(think, body)})
        elif role == "tool":
            out.append({"role": "tool", "content": m["error"] if m.get("error") is not None else _blocks(m, "text")})
    if not out or out[0]["role"] != "system":
        out.insert(0, {"role": "system", "content": make_system_prompt("", tools, append=append)})
    return out


class LlamaSRAgentLLM(BasePipelineElement):
    def __init__(self, model: str, temperature: float | None = None) -> None:
        self.base_model = os.getenv("LLAMA_SR_AGENT_BASE_MODEL", DEFAULT_BASE_MODEL)
        lora = os.getenv("LLAMA_SR_AGENT_LORA_PATH", "")
        self.append = _env_flag("LLAMA_SR_AGENT_SYS_APPEND", True)
        self.max_new_tokens = int(os.getenv("LLAMA_SR_AGENT_MAX_NEW_TOKENS", "1536"))
        self.temperature = float(os.getenv("LLAMA_SR_AGENT_TEMPERATURE", "0.6"))
        self.top_p = float(os.getenv("LLAMA_SR_AGENT_TOP_P", "0.9"))
        self.print_first_inputs = int(os.getenv("LLAMA_SR_AGENT_PRINT_FIRST_INPUTS", "2"))
        self._printed = 0
        lora_exists = bool(lora) and Path(lora).exists()
        print(f"LlamaSRAgentLLM base={self.base_model} lora={lora or '<unset>'} exists={lora_exists} append={self.append} "
              f"max_new={self.max_new_tokens} T={self.temperature} top_p={self.top_p}")

        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.base_model, torch_dtype=dtype, device_map="auto")
        if lora_exists:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, lora, is_trainable=False)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.stop_ids = [self.tokenizer.convert_tokens_to_ids(t) for t in ("<|eot_id|>", "<|eom_id|>", "<|end_of_text|>")]

    def _generate(self, prompt: str) -> str:
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        out = self.model.generate(
            **inputs,
            do_sample=self.temperature > 0,
            temperature=self.temperature if self.temperature > 0 else None,
            top_p=self.top_p if self.temperature > 0 else None,
            max_new_tokens=self.max_new_tokens,
            eos_token_id=self.stop_ids,
            pad_token_id=self.tokenizer.pad_token_id,
            stop_strings=["</function>"],  # the untrained base sometimes keeps writing a fake function result
            tokenizer=self.tokenizer,
        )
        text = self.tokenizer.decode(out[0][inputs.input_ids.size(-1):], skip_special_tokens=True)
        del out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return text

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        prompt = render(_to_prompt_messages(messages, runtime, self.append))
        if self._printed < self.print_first_inputs:
            self._printed += 1
            print(f"===== LLAMA SR-AGENT RAW INPUT {self._printed} =====\n{prompt}\n===== END RAW INPUT =====")
        text = self._generate(prompt)
        print("===== RAW LLAMA OUTPUT =====")
        print(text)
        think, visible, call = parse_output(text)
        content = [ThinkingContentBlock(type="thinking", content=think, id=None)] if think else []
        if visible or call is None:
            content.append(text_content_block_from_string(visible))
        tool_calls = [FunctionCall(function=call[0], args=call[1], id="call_" + uuid.uuid4().hex[:24])] if call else None
        output = ChatAssistantMessage(role="assistant", content=content, tool_calls=tool_calls)
        return query, runtime, env, [*messages, output], extra_args
