"""Base-model (Qwen/Qwen3-8B, no LoRA) per-token NLL of the <think> span, Claude-written vs self-distilled, on matched steps.
Context = system + think-free history rendered with the Qwen3 chat template (exactly the inference prefix). Reports mean NLL / PPL.
python nll_diagnostic.py --n 600 --out runs/nll_diagnostic.json
"""
import argparse, json, random, math, re, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from common import load_data, extract_steps, build_messages, Step
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=600); ap.add_argument("--out", default="runs/nll_diagnostic.json")
ap.add_argument("--sd", default="../LLaMA-Factory/data/toucan_32B_v2_sdL2.json"); a = ap.parse_args()
random.seed(0)
orig = load_data(); sd = json.load(open(a.sd))
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B"); model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", torch_dtype=torch.bfloat16, device_map="cuda").eval()
def think_of(v):
    m = re.search(r"<think>\n?(.*?)\n?</think>", v, re.S); return m.group(1).strip() if m else None
def nll(prefix_ids, text):
    ids = tok(text, add_special_tokens=False).input_ids
    x = torch.tensor([prefix_ids + ids], device="cuda")
    with torch.no_grad(): logits = model(x).logits[0, len(prefix_ids)-1:-1].float()
    lp = torch.log_softmax(logits, -1).gather(1, torch.tensor(ids, device="cuda")[:, None]).squeeze(1)
    return -lp.sum().item(), len(ids)
res = {"claude": [0.0, 0], "sd": [0.0, 0], "by_situation": {}}; per = []
pairs = [(ti, s) for ti in random.sample(range(len(orig)), 400) for s in extract_steps(orig[ti], ti)]
random.shuffle(pairs); pairs = pairs[: a.n]
for ti, step in pairs:
    sd_turns = [m for m in sd[ti]["conversations"] if m["from"] in ("function_call", "gpt")]
    sd_think = think_of(sd_turns[step.step_idx]["value"]); cl_think = step.claude_think
    if not sd_think or sd_think == cl_think: continue           # skip fallback steps
    prefix = tok.apply_chat_template(build_messages(step, "L0"), tokenize=False, add_generation_prompt=True, enable_thinking=True) + "<think>\n"
    pid = tok(prefix, add_special_tokens=False).input_ids
    if len(pid) > 6000: continue
    c = nll(pid, cl_think + "\n</think>"); s_ = nll(pid, sd_think + "\n</think>")
    res["claude"][0] += c[0]; res["claude"][1] += c[1]; res["sd"][0] += s_[0]; res["sd"][1] += s_[1]
    b = res["by_situation"].setdefault(step.situation, {"claude": [0.0, 0], "sd": [0.0, 0], "n": 0})
    b["claude"][0] += c[0]; b["claude"][1] += c[1]; b["sd"][0] += s_[0]; b["sd"][1] += s_[1]; b["n"] += 1
    per.append({"traj": ti, "step": step.step_idx, "situation": step.situation, "claude_nll": c[0]/c[1], "sd_nll": s_[0]/s_[1], "claude_tok": c[1], "sd_tok": s_[1]})
def fmt(x): return {"mean_nll": x[0]/x[1], "ppl": math.exp(x[0]/x[1]), "tokens": x[1]}
out = {"n_steps": len(per), "claude": fmt(res["claude"]), "self_distilled": fmt(res["sd"]),
       "by_situation": {k: {"n": v["n"], "claude": fmt(v["claude"]), "self_distilled": fmt(v["sd"])} for k, v in res["by_situation"].items()}, "per_step": per}
json.dump(out, open(a.out, "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "per_step"}, indent=1))
