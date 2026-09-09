"""
Batch sampling variant that uses `num_return_sequences` to collect multiple Qwen
responses per assistant turn in a single generation call.
"""

from __future__ import annotations

import copy
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

import click
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from agentdojo.scripts.sample_qwen_think_from_expert_single import (
    DEFAULT_MODEL_ID,
    _iter_trajectories,
    _load_suite_tools,
    _normalize_message,
    _assistant_message_to_json,
    _should_sample_trajectory,
    _write_json,
    openai_messages_to_qwen_messages,
    qwen8b_text_to_assistant_message,
)


def _retarget_tool_content(content: Any) -> Any:
    """Replace hardcoded 'Claude' mentions in tool outputs with 'AI Assistant'."""
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


def _normalize_message_with_retarget(message: dict[str, Any]) -> dict[str, Any]:
    """Normalize message and ensure tool content is retargeted for Qwen context/output."""
    normalized = _normalize_message(message)
    if normalized.get("role") == "tool":
        normalized["content"] = _retarget_tool_content(normalized.get("content"))
    return normalized


def chat_completion_request_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    *,
    num_return_sequences: int,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int = 20,
    min_p: float | None = 0.0,
    thinking_budget: int = 512,
    second_pass_max_new_tokens: int = 512,
) -> list[str]:
    """Two-pass generation with shared first pass across multiple samples."""
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
        num_return_sequences=num_return_sequences,
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
    results: list[str] = []

    for seq_idx in range(num_return_sequences):
        first_ids = first_out[seq_idx]
        first_new_ids = first_ids[input_length:].tolist()

        if im_end_id is not None and im_end_id in first_new_ids:
            text = tokenizer.decode(first_ids[input_length:], skip_special_tokens=True).strip()
            results.append(text)
            continue

        early_ids = None
        if think_end_id is None or think_end_id not in first_new_ids:
            early_stop_text = (
                "\n\n Considering the limited time by the user, I have to give the solution based on the thinking directly now.\n"
                "</think>\n\n"
            )
            early_ids = (
                tokenizer([early_stop_text], return_tensors="pt", return_attention_mask=False)
                .input_ids.to(device)
            )
            input_ids = torch.cat([first_ids.unsqueeze(0), early_ids], dim=-1)
        else:
            input_ids = first_ids.unsqueeze(0)

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
        text = tokenizer.decode(full_new_ids, skip_special_tokens=True).strip()

        results.append(text)

        del second_out, input_ids, attn_mask
        if early_ids is not None:
            del early_ids

    del first_out, model_inputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


class QwenThinkBatchGenerator:
    """Batch wrapper around the HF model using num_return_sequences."""

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
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.thinking_budget = thinking_budget
        self.second_pass_max_new_tokens = second_pass_max_new_tokens

    def generate_batch(
        self,
        openai_messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
        num_sequences: int,
    ) -> list[tuple[str, dict[str, Any]]]:
        qwen_messages = openai_messages_to_qwen_messages(list(openai_messages))
        has_system = any(message.get("role") == "system" for message in qwen_messages)
        if not has_system:
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
        raw_texts = chat_completion_request_batch(
            self.model,
            self.tokenizer,
            prompt_text,
            num_return_sequences=num_sequences,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
            thinking_budget=self.thinking_budget,
            second_pass_max_new_tokens=self.second_pass_max_new_tokens,
        )
        outputs: list[tuple[str, dict[str, Any]]] = []
        for raw_text in raw_texts:
            assistant_message = qwen8b_text_to_assistant_message(raw_text)
            outputs.append((raw_text, assistant_message))
        return outputs


@lru_cache(maxsize=None)
def _load_suite_tools_cached(benchmark_version: str, suite_name: str) -> list[dict[str, Any]]:
    return _load_suite_tools(benchmark_version, suite_name)


def _build_qwen_samples_batch(
    *,
    model: QwenThinkBatchGenerator,
    tool_spec: Sequence[dict[str, Any]],
    normalized_messages: list[dict[str, Any]],
    original_messages: list[dict[str, Any]],
    num_sequences: int,
) -> list[dict[str, Any]]:
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
        batch_outputs = model.generate_batch(filtered_context, tool_spec, num_sequences)
        qwen_samples = []
        for sequence_index, (raw_text, assistant_message) in enumerate(batch_outputs):
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


def _process_trajectory(
    suite_name: str,
    json_path: Path,
    generator: QwenThinkBatchGenerator,
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
            import shutil

            shutil.rmtree(trajectory_output_dir)
        else:
            click.echo(f"[Skip-existing] {trajectory_output_dir} already exists. Use --overwrite to regenerate.")
            return 0, 0

    normalized_messages = [_normalize_message_with_retarget(message) for message in data.get("messages", [])]
    tool_spec = _load_suite_tools_cached(benchmark_version, suite_name)
    samples = _build_qwen_samples_batch(
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
    default=Path("runs/qwen3-8b-think-samples-batch"),
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
    default=5,
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
    expert_run_dir = expert_run_dir.resolve()
    output_dir = output_dir.resolve()
    if not expert_run_dir.exists():
        raise click.ClickException(f"Expert run directory '{expert_run_dir}' does not exist.")

    click.echo(f"Loading Qwen model '{model_id}' in batch mode...")
    generator = QwenThinkBatchGenerator(
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
