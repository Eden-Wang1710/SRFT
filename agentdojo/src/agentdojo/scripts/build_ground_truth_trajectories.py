import json
import time
from pathlib import Path

import click

import agentdojo.attacks  # noqa: F401 - register attacks
from agentdojo.agent_pipeline.agent_pipeline import load_system_message
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.tool_execution import tool_result_to_str
from agentdojo.attacks.attack_registry import ATTACKS, load_attack
from agentdojo.functions_runtime import FunctionsRuntime
from agentdojo.task_suite.load_suites import get_suite


class _NamedPipeline(BasePipelineElement):
    def __init__(self, name: str) -> None:
        self.name = name

    def query(self, query, runtime, env, messages, extra_args):  # pragma: no cover - defensive stub
        raise RuntimeError("_NamedPipeline should not be executed.")


def _tool_call_to_dict(tool_call, tool_call_id: str) -> dict:
    return {
        "function": tool_call.function,
        "args": tool_call.args,
        "id": tool_call_id,
        "placeholder_args": tool_call.placeholder_args,
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=4, ensure_ascii=True)


@click.command()
@click.option(
    "--benchmark-version",
    default="v1",
    show_default=True,
    help="Which benchmark version to use when loading suites.",
)
@click.option(
    "--pipeline-name",
    default="ground-truth",
    show_default=True,
    help="Top-level run directory name for the generated trajectories.",
)
@click.option(
    "--model-name",
    default=None,
    help="Model name string used by attacks (e.g. 'Qwen/Qwen3-8B'). Defaults to --pipeline-name.",
)
@click.option(
    "--output-root",
    type=Path,
    default=Path("agentdojo/runs"),
    show_default=True,
    help="Root directory where the run folder will be created.",
)
@click.option(
    "--suite",
    "suites",
    multiple=True,
    type=click.Choice(["banking", "slack", "travel", "workspace"]),
    help="Limit generation to specific suites; defaults to all four.",
)
@click.option(
    "--attack",
    "attack_name",
    default=None,
    type=click.Choice(sorted(ATTACKS.keys())),
    help="Optional attack to generate prompt-injected trajectories.",
)
@click.option(
    "--injection-task",
    "injection_tasks",
    multiple=True,
    help="Limit to specific injection tasks (e.g. injection_task_10). Defaults to all.",
)
def main(
    benchmark_version: str,
    pipeline_name: str,
    model_name: str | None,
    output_root: Path,
    suites: tuple[str, ...],
    attack_name: str | None,
    injection_tasks: tuple[str, ...],
) -> None:
    suite_names = list(suites) if suites else ["banking", "slack", "travel", "workspace"]
    system_message = load_system_message(None)
    pipeline_name_raw = pipeline_name
    pipeline_name = pipeline_name.replace("/", "_")
    output_root = output_root / pipeline_name
    attack_pipeline_name = model_name or pipeline_name_raw

    for suite_name in suite_names:
        suite = get_suite(benchmark_version, suite_name)
        base_environment = suite.load_and_inject_default_environment({})

        if attack_name is None:
            attack = None
            injection_tasks_to_run = {"none": None}
        else:
            attack = load_attack(attack_name, suite, _NamedPipeline(attack_pipeline_name))
            if injection_tasks:
                injection_tasks_to_run = {
                    injection_task_id: suite.get_injection_task_by_id(injection_task_id)
                    for injection_task_id in injection_tasks
                }
            else:
                injection_tasks_to_run = suite.injection_tasks

        for user_task in suite.user_tasks.values():
            for injection_task_id, injection_task in injection_tasks_to_run.items():
                start_time = time.time()
                if attack is None:
                    injections = {}
                    injected_environment = base_environment.model_copy(deep=True)
                    display_environment = injected_environment.model_copy(deep=True)
                else:
                    injections = attack.attack(user_task, injection_task)
                    injected_environment = suite.load_and_inject_default_environment(injections)
                    display_environment = base_environment.model_copy(deep=True)

                task_environment = user_task.init_environment(injected_environment.model_copy(deep=True))
                pre_environment = task_environment.model_copy(deep=True)
                runtime = FunctionsRuntime(suite.tools)

                messages: list[dict] = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_task.PROMPT},
                ]

                tool_index = 0
                ground_truth_calls = user_task.ground_truth(task_environment)
                display_calls = user_task.ground_truth(user_task.init_environment(display_environment))
                if len(display_calls) != len(ground_truth_calls):
                    display_calls = ground_truth_calls

                for index, tool_call in enumerate(ground_truth_calls):
                    display_call = display_calls[index]
                    if display_call.function != tool_call.function:
                        display_call = tool_call
                    tool_index += 1
                    tool_call_id = tool_call.id or f"gt_tool_{tool_index:03d}"
                    tool_call_dict = _tool_call_to_dict(display_call, tool_call_id)
                    messages.append(
                        {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [tool_call_dict],
                        }
                    )
                    tool_result, _ = runtime.run_function(
                        task_environment, tool_call.function, tool_call.args, raise_on_error=True
                    )
                    tool_output = tool_result_to_str(tool_result)
                    messages.append(
                        {
                            "role": "tool",
                            "content": tool_output,
                            "tool_call_id": tool_call_id,
                            "tool_call": tool_call_dict,
                            "error": None,
                        }
                    )

                final_content = user_task.GROUND_TRUTH_OUTPUT or "done"
                messages.append(
                    {
                        "role": "assistant",
                        "content": final_content,
                    }
                )

                duration = time.time() - start_time
                model_output = user_task.GROUND_TRUTH_OUTPUT or ""
                utility_from_traces = user_task.utility_from_traces(
                    model_output, pre_environment, task_environment, ground_truth_calls
                )
                utility = (
                    utility_from_traces
                    if utility_from_traces is not None
                    else user_task.utility(model_output, pre_environment, task_environment)
                )

                if attack is None:
                    security = True
                    attack_type = None
                    out_attack_dir = "none"
                    out_injection_id = "none"
                else:
                    security_from_traces = injection_task.security_from_traces(
                        model_output, pre_environment, task_environment, ground_truth_calls
                    )
                    security = (
                        security_from_traces
                        if security_from_traces is not None
                        else injection_task.security(model_output, pre_environment, task_environment)
                    )
                    attack_type = attack.name
                    out_attack_dir = attack.name
                    out_injection_id = injection_task_id

                payload = {
                    "suite_name": suite.name,
                    "pipeline_name": pipeline_name,
                    "user_task_id": user_task.ID,
                    "injection_task_id": None if attack is None else injection_task_id,
                    "attack_type": attack_type,
                    "injections": injections,
                    "messages": messages,
                    "error": None,
                    "utility": utility,
                    "security": security,
                    "duration": duration,
                }

                output_path = output_root / suite.name / user_task.ID / out_attack_dir / f"{out_injection_id}.json"
                _write_json(output_path, payload)


if __name__ == "__main__":
    main()
