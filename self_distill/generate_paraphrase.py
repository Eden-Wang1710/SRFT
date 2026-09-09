"""Run the v3 paraphrase recipe on a steps file (pilot) or on shards of the full data.
python generate_paraphrase.py --steps pilot_para/steps.json --out pilot_para/out.jsonl [--n 2] [--dry-run]
"""
import argparse, json, os, sys, time
from common import Step
from paraphrase import render, check
ap = argparse.ArgumentParser(); ap.add_argument("--steps", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=2); ap.add_argument("--max-tokens", type=int, default=900); ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--model", default="Qwen/Qwen3-8B"); ap.add_argument("--retry-n", type=int, default=4); ap.add_argument("--retry-temp", type=float, default=0.8); a = ap.parse_args()
steps = [Step(**d) for d in json.load(open(a.steps))]
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.model)
prompts = [render(tok, s) for s in steps]
if a.dry_run:
    outs = [[s.claude_think, "Okay, " + s.claude_think.replace("The user", "I see the user").replace("the agent", "I")] for s in steps]
else:
    from vllm import LLM, SamplingParams
    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=10240, gpu_memory_utilization=0.92, enable_prefix_caching=True, seed=7)
    sp = SamplingParams(n=a.n, temperature=0.6, top_p=0.95, top_k=20, max_tokens=a.max_tokens)
    res = llm.generate(prompts, sp); outs = [[c.text for c in r.outputs] for r in res]

def judge(s, cands, round_id):
    atts, ok = [], []
    for k, o in enumerate(cands):
        fails, m = check(s, o, tok); att = {"round": round_id, "k": k, "text": o.strip(), "fails": fails, "metrics": m}; atts.append(att)
        if not fails: ok.append(att)
    return atts, ok

def quotes_injection(s, text):
    if s.situation != "injected" or not s.injection_snippet: return True
    q = s.injection_snippet.split(); o = text.lower()
    return any(" ".join(q[i:i+6]).lower() in o for i in range(max(1, len(q) - 5)))

def pick(ok, s=None):   # prefer a verbatim quote of the injection, then closest to the draft length, then least copying
    return min(ok, key=lambda x: (not quotes_injection(s, x["text"]) if s else False, x["metrics"].get("pasted_json", False), abs(x["metrics"]["len_ratio"] - 1.0), x["metrics"]["overlap8"]))

recs = []
for s, p, cands in zip(steps, prompts, outs):
    rec = s.to_json(); rec.pop("history"); rec.pop("system"); rec["prompt_tokens"] = len(tok(p, add_special_tokens=False).input_ids)
    rec["attempts"], ok = judge(s, cands, 0); rec["accepted"] = pick(ok, s) if ok else None; recs.append(rec)
# retry round for the failures: more samples, higher temperature
redo = [i for i, r in enumerate(recs) if r["accepted"] is None]
if redo and not a.dry_run and a.retry_n > 0:
    sp2 = SamplingParams(n=a.retry_n, temperature=a.retry_temp, top_p=0.95, top_k=20, max_tokens=a.max_tokens, seed=11)
    res2 = llm.generate([prompts[i] for i in redo], sp2)
    for i, r in zip(redo, res2):
        atts, ok = judge(steps[i], [c.text for c in r.outputs], 1); recs[i]["attempts"] += atts
        if ok: recs[i]["accepted"] = pick(ok, steps[i])
n_acc = sum(1 for r in recs if r["accepted"])
with open(a.out, "w") as fo:
    for rec in recs:
        fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
print(f"round0 accepted {len(recs)-len(redo)}/{len(recs)}; after retry {n_acc}/{len(recs)}", flush=True)
print(f"accepted {n_acc}/{len(steps)} -> {a.out}", flush=True)
if not a.dry_run: sys.stdout.flush(); os._exit(0)
