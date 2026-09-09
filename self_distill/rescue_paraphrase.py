"""Rescue round for steps rejected in the shard run: re-judge stored attempts with the current filters, then regenerate the rest with
a length-emphasis note, n=6, T=0.9. python rescue_paraphrase.py --runs runs/v3para --out runs/v3para/rescue.jsonl [--dry-run]"""
import argparse, glob, json, os, sys
from pathlib import Path
from common import Step
from paraphrase import render, check
from generate_paraphrase_lib import judge_attempts, pick
ap = argparse.ArgumentParser(); ap.add_argument("--runs", default="runs/v3para"); ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=6); ap.add_argument("--temp", type=float, default=0.9); ap.add_argument("--max-tokens", type=int, default=900)
ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--model", default="Qwen/Qwen3-8B"); a = ap.parse_args()
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.model)
steps = {}
for f in glob.glob(str(Path(a.runs) / "steps_shard*.json")):
    for d in json.load(open(f)): steps[(d["traj_idx"], d["step_idx"])] = Step(**d)
recs = []
for f in sorted(glob.glob(str(Path(a.runs) / "out_shard*.jsonl"))):
    for line in open(f):
        r = json.loads(line)
        if not r["accepted"]: recs.append(r)
print("rejected in shards:", len(recs), flush=True)
# 1) re-judge with current filters
still = []
for r in recs:
    s = steps[(r["traj_idx"], r["step_idx"])]
    ok = []
    for at in r["attempts"]:
        at["fails"], at["metrics"] = check(s, at["text"], tok)
        if not at["fails"]: ok.append(at)
    r["accepted"] = pick(ok, s) if ok else None
    if not r["accepted"]: still.append(r)
print("after re-judge still rejected:", len(still), flush=True)
# 2) regenerate with the rescue note
if still and not a.dry_run:
    from vllm import LLM, SamplingParams
    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=10240, gpu_memory_utilization=0.92, enable_prefix_caching=True, seed=23)
    sp = SamplingParams(n=a.n, temperature=a.temp, top_p=0.95, top_k=20, max_tokens=a.max_tokens)
    prompts = [render(tok, steps[(r["traj_idx"], r["step_idx"])], rescue=True) for r in still]
    res = llm.generate(prompts, sp)
    for r, out in zip(still, res):
        s = steps[(r["traj_idx"], r["step_idx"])]
        atts, ok = judge_attempts(s, [c.text for c in out.outputs], 2, tok)
        r["attempts"] += atts
        if ok: r["accepted"] = pick(ok, s)
n_acc = sum(1 for r in recs if r["accepted"])
with open(a.out, "w") as fo:
    for r in recs: fo.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"rescue: {n_acc}/{len(recs)} of the rejected steps now accepted -> {a.out}", flush=True)
if not a.dry_run: sys.stdout.flush(); os._exit(0)
