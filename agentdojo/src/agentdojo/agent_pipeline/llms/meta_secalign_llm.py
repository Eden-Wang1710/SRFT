import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.llms.local_llm import _make_system_prompt, _parse_model_output
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
from agentdojo.types import ChatMessage, get_text_content_as_str


DEFAULT_BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_ADAPTER_MODEL = "facebook/Meta-SecAlign-8B"
DEFAULT_MERGED_MODEL = (
    "/home/cxiao13/scratch-cxiao13/zixuan/safe-agent-project/"
    "injecAgent-rl-harmmer/rl-injector/checkpoints/Meta-SecAlign-8B-merged"
)


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append((block.get("text") or block.get("content") or "").strip())
                else:
                    parts.append(f"[{block.get('type', 'unknown')} omitted]")
            else:
                parts.append(str(block).strip())
        return "\n\n".join(part for part in parts if part).strip()
    return str(content).strip()


def _tool_call_to_function_text(message: ChatMessage) -> str:
    tool_calls = message.get("tool_calls") or []
    if not tool_calls:
        return _content_to_text(message.get("content"))
    tool_call = tool_calls[0]
    return f"<function={tool_call.function}>{json.dumps(tool_call.args)}</function>"


def _messages_for_meta_secalign(
    messages: Sequence[ChatMessage],
    runtime: FunctionsRuntime,
    tool_delimiter: str,
) -> list[dict[str, str]]:
    """Mirror AgentDojo LocalLLM formatting, with tool messages using input role.

    The official Meta-SecAlign AgentDojo runner calls AgentDojo with
    `--model local --tool-delimiter input`; that keeps system/user/assistant
    roles unchanged and only changes tool responses from role=tool to role=input.
    """
    formatted: list[dict[str, str]] = []
    for message in messages:
        role = message["role"]
        content = message["content"]

        if role == "system" and content is not None:
            text = _make_system_prompt(get_text_content_as_str(content), runtime.functions.values())
        elif role == "assistant":
            text = _tool_call_to_function_text(message)
        elif role == "tool":
            role = tool_delimiter
            if message.get("error") is not None:
                text = json.dumps({"error": message["error"]})
            else:
                func_result = message["content"]
                if func_result == "None":
                    func_result = "Success"
                text = json.dumps({"result": func_result})
        else:
            text = _content_to_text(content)

        formatted.append({"role": role, "content": text})

    return formatted


def chat_completion_request(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    *,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> str:
    device = next(model.parameters()).device
    model_inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    input_length = model_inputs.input_ids.size(-1)

    do_sample = temperature > 0
    gen_kwargs = dict(
        **model_inputs,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
    )
    gen_kwargs = {key: value for key, value in gen_kwargs.items() if value is not None}

    output = model.generate(**gen_kwargs)
    result = tokenizer.decode(output[0][input_length:], skip_special_tokens=True).strip()
    del output
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


class MetaSecAlignLLM(BasePipelineElement):
    def __init__(
        self,
        model: str,
        temperature: float | None = None,
        tool_delimiter: str = "input",
    ) -> None:
        merged_model = os.getenv("META_SECALIGN_MERGED_MODEL")
        if merged_model is None and Path(DEFAULT_MERGED_MODEL).exists():
            merged_model = DEFAULT_MERGED_MODEL
        self.merged_model = merged_model
        self.adapter_model = os.getenv("META_SECALIGN_ADAPTER_MODEL", model or DEFAULT_ADAPTER_MODEL)
        self.base_model = os.getenv("META_SECALIGN_BASE_MODEL", DEFAULT_BASE_MODEL)
        self.temperature = (
            float(os.getenv("META_SECALIGN_TEMPERATURE"))
            if os.getenv("META_SECALIGN_TEMPERATURE") is not None
            else (0.0 if temperature is None else temperature)
        )
        self.top_p = float(os.getenv("META_SECALIGN_TOP_P", "0.9"))
        self.max_new_tokens = int(os.getenv("META_SECALIGN_MAX_NEW_TOKENS", "1024"))
        self.print_first_inputs = int(os.getenv("META_SECALIGN_PRINT_FIRST_INPUTS", "3"))
        self.tool_delimiter = tool_delimiter
        self._printed_inputs = 0

        if torch.cuda.is_available():
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32

        if self.merged_model:
            print(f"Loading merged Meta-SecAlign model/tokenizer: {self.merged_model}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.merged_model, use_fast=True, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.merged_model,
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            print(f"Loading Meta-SecAlign base model: {self.base_model}")
            print(f"Loading Meta-SecAlign adapter/tokenizer: {self.adapter_model}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.adapter_model, use_fast=True, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=True,
            )

            adapter_path = Path(self.adapter_model)
            try:
                from peft import PeftModel
            except Exception as e:
                raise RuntimeError("Meta-SecAlign is a LoRA adapter; install `peft` to load it.") from e
            self.model = PeftModel.from_pretrained(
                self.model,
                str(adapter_path) if adapter_path.exists() else self.adapter_model,
                is_trainable=False,
            )

            if _env_flag("META_SECALIGN_MERGE_LORA", False):
                self.model = self.model.merge_and_unload()

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        meta_messages = _messages_for_meta_secalign(messages, runtime, self.tool_delimiter)
        prompt_text = self.tokenizer.apply_chat_template(
            meta_messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        if self._printed_inputs < self.print_first_inputs:
            self._printed_inputs += 1
            print(f"===== META SECALIGN RAW LLM INPUT {self._printed_inputs}/{self.print_first_inputs} =====")
            print(prompt_text)
            print("===== END META SECALIGN RAW LLM INPUT =====")

        text = chat_completion_request(
            self.model,
            self.tokenizer,
            prompt_text,
            temperature=self.temperature,
            top_p=self.top_p,
            max_new_tokens=self.max_new_tokens,
        )
        print("===== RAW META SECALIGN OUTPUT =====")
        print(text)

        output = _parse_model_output(text)
        print("===== Assistant LLM OUTPUT =====")
        print(output)
        return query, runtime, env, [*messages, output], extra_args
