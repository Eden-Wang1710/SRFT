"""toucan_32B_v3_base.json (Qwen/Hermes tool format) -> AgentDojo `local` format for SR-Agent-Llama (2026-09-11).

Only the tool protocol changes; every think, tool call, argument, tool output and answer is copied verbatim.
  system        : AgentDojo system message + tool schemas re-rendered with llama_local_prompt.make_system_prompt (no append)
  function_call : -> gpt  "<think>\n…\n</think>\n\n<function=name>{args}</function>"
  gpt           : -> gpt  "<think>\n…\n</think>\n\n<answer>"      (unchanged)
  observation   : -> observation (raw text; LLaMA-Factory llama3 renders it as an `ipython` turn)
usage: python multibase/convert_llama_local.py [--src …/toucan_32B_v3_base.json] [--out …/toucan_32B_v3_base_llama_local.json]
"""

import argparse
import importlib.util
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "llama_local_prompt", ROOT / "agentdojo/src/agentdojo/agent_pipeline/llms/llama_local_prompt.py"
)
llp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(llp)

TOOLS_SEP = "\n\n# Tools\n\n"
THINK_OPEN, THINK_CLOSE = "<think>\n", "\n</think>\n\n"


def split_system(system: str) -> tuple[str, list[dict]]:
    assert TOOLS_SEP in system, system[:200]
    head = system.split(TOOLS_SEP, 1)[0]
    a, b = system.rfind("<tools>\n"), system.rfind("\n</tools>")
    tools = [json.loads(line)["function"] for line in system[a + len("<tools>\n"):b].strip().splitlines()]
    return head, tools


def split_think(value: str) -> tuple[str, str]:
    assert value.startswith(THINK_OPEN) and THINK_CLOSE in value, value[:120]
    think, body = value[len(THINK_OPEN):].split(THINK_CLOSE, 1)
    return think, body


def convert(traj: dict, stats: Counter) -> dict:
    head, tools = split_system(traj["system"])
    conv = []
    for turn in traj["conversations"]:
        role, value = turn["from"], turn["value"]
        if role == "human":
            conv.append({"from": "human", "value": value.strip()})
        elif role == "observation":
            conv.append({"from": "observation", "value": value})
        elif role == "function_call":
            think, body = split_think(value)
            call = json.loads(body)
            assert set(call) == {"name", "arguments"} and isinstance(call["arguments"], dict), body[:200]
            conv.append({"from": "gpt", "value": llp.format_assistant(think, llp.format_call(call["name"], call["arguments"]))})
            stats["tool_call_turns"] += 1
        elif role == "gpt":
            think, body = split_think(value)
            assert body.strip(), "empty final answer"
            conv.append({"from": "gpt", "value": llp.format_assistant(think, body)})
            stats["answer_turns"] += 1
        else:
            raise ValueError(role)
    stats["trajectories"] += 1
    stats["tools"] += len(tools)
    return {"conversations": conv, "system": llp.make_system_prompt(head, tools, append=False)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "LLaMA-Factory/data/toucan_32B_v3_base.json"))
    ap.add_argument("--out", default=str(ROOT / "LLaMA-Factory/data/toucan_32B_v3_base_llama_local.json"))
    args = ap.parse_args()
    data = json.load(open(args.src))
    stats: Counter = Counter()
    out = [convert(t, stats) for t in data]
    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=1)
    json.dump(dict(stats, src=args.src), open(args.out + ".stats.json", "w"), indent=1)
    print(dict(stats), "->", args.out)


if __name__ == "__main__":
    main()
