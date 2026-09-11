"""Verify that the SR-Agent-Llama training render (LLaMA-Factory `llama3`) equals the inference render (llama_local_prompt.render),
token by token, and report sequence lengths vs the 8,192 cutoff. Run in the `llamafactory` env from SRFT/ (CPU is enough):
  python multibase/check_llama_render.py [--n 200] [--all-lengths]
"""

import argparse
import ast
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "LLaMA-Factory/src"))
from llamafactory.data import get_template_and_fix_tokenizer  # noqa: E402
from llamafactory.hparams import DataArguments  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

LLMS = ROOT / "agentdojo/src/agentdojo/agent_pipeline/llms"
_spec = importlib.util.spec_from_file_location("llama_local_prompt", LLMS / "llama_local_prompt.py")
llp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(llp)


def local_llm_prompt() -> str:
    tree = ast.parse((LLMS / "local_llm.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and node.targets[0].id == "_tool_calling_prompt":
            return node.value.value
    raise RuntimeError("_tool_calling_prompt not found")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "LLaMA-Factory/data/toucan_32B_v3_base_llama_local.json"))
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--all-lengths", action="store_true")
    args = ap.parse_args()

    assert llp.TOOL_CALLING_PROMPT == local_llm_prompt(), "TOOL_CALLING_PROMPT drifted from local_llm._tool_calling_prompt"
    tok = AutoTokenizer.from_pretrained(args.model)
    template = get_template_and_fix_tokenizer(tok, DataArguments(template="llama3"))
    data = json.load(open(args.data))
    role = {"human": "user", "gpt": "assistant", "observation": "observation"}
    step = max(1, len(data) // args.n)
    checked = mismatched = 0
    lengths = []
    for i, ex in enumerate(data):
        if i % step and not args.all_lengths:
            continue
        msgs = [{"role": role[t["from"]], "content": t["value"]} for t in ex["conversations"]]
        pairs = template.encode_multiturn(tok, msgs, ex["system"])
        lf_ids = [x for p in pairs for part in p for x in part]
        lengths.append(len(lf_ids))
        if i % step:
            continue
        inf = [{"role": "system", "content": ex["system"]}] + [
            {"role": "tool" if m["role"] == "observation" else m["role"], "content": m["content"]} for m in msgs
        ]
        inf_ids = tok.encode(llp.render(inf, add_generation_prompt=False), add_special_tokens=False)
        checked += 1
        if inf_ids != lf_ids:
            mismatched += 1
            if mismatched <= 3:
                k = next((j for j, (a, b) in enumerate(zip(inf_ids, lf_ids)) if a != b), min(len(inf_ids), len(lf_ids)))
                print(f"MISMATCH traj {i} at token {k}: LF …{tok.decode(lf_ids[max(0, k - 20):k + 20])!r}\n"
                      f"                        INF …{tok.decode(inf_ids[max(0, k - 20):k + 20])!r}")
    lengths.sort()
    over = sum(x > 8192 for x in lengths)
    print(f"render check: {checked - mismatched}/{checked} identical token sequences (LLaMA-Factory llama3 == inference render)")
    print(f"lengths over {len(lengths)} traj: mean {sum(lengths) / len(lengths):.0f}, median {lengths[len(lengths) // 2]}, "
          f"p90 {lengths[int(0.9 * len(lengths))]}, max {lengths[-1]}, > 8192 cutoff: {over}")
    print("sample render (first 1500 chars):\n" + llp.render(
        [{"role": "system", "content": data[0]["system"]}]
        + [{"role": "tool" if t["from"] == "observation" else role[t["from"]], "content": t["value"]} for t in data[0]["conversations"][:3]],
        add_generation_prompt=False)[:1500])
    sys.exit(1 if mismatched else 0)


if __name__ == "__main__":
    main()
