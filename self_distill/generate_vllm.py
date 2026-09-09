"""Full-scale self-distillation with vLLM: one shard of steps, one rung, n samples each, resumable.

python generate_vllm.py --shard 0 --num-shards 4 --rung L2 --n 4 --out runs/sdL2 [--limit 200] [--dry-run]
Writes runs/sdL2/shard0.jsonl (one record per step, all attempts + accepted) and appends on resume.
"""
import argparse, json, time, os, sys
from pathlib import Path
from common import load_data, extract_steps, render_prompt, parse_output, check_sample

ap = argparse.ArgumentParser()
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--num-shards", type=int, default=1)
ap.add_argument("--rung", default="L2")
ap.add_argument("--n", type=int, default=4)
ap.add_argument("--out", default="runs/sdL2")
ap.add_argument("--model", default="Qwen/Qwen3-8B")
ap.add_argument("--max-tokens", type=int, default=1024)
ap.add_argument("--max-model-len", type=int, default=10240)
ap.add_argument("--chunk", type=int, default=768, help="prompts per vLLM call (results flushed after each chunk)")
ap.add_argument("--limit", type=int, default=0, help="only first N steps of the shard (pilot)")
ap.add_argument("--dry-run", action="store_true", help="no GPU: fake engine echoes the Claude think (pipeline test)")
args = ap.parse_args()

out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / f"shard{args.shard}.jsonl"
done = set()
if out_path.exists():
    for line in open(out_path):
        try:
            r = json.loads(line); done.add((r["traj_idx"], r["step_idx"]))
        except Exception:
            pass
print(f"shard {args.shard}/{args.num_shards} rung={args.rung} n={args.n} resume: {len(done)} steps already done", flush=True)

data = load_data()
steps = []
for ti in range(len(data)):
    if ti % args.num_shards != args.shard:
        continue
    for s in extract_steps(data[ti], ti):
        if (s.traj_idx, s.step_idx) not in done:
            steps.append(s)
if args.limit:
    steps = steps[: args.limit]
print(f"{len(steps)} steps to generate", flush=True)
if not steps:
    print("nothing left to do", flush=True); os._exit(0)

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(args.model)

if args.dry_run:
    class FakeOut:
        def __init__(self, texts): self.outputs = [type("O", (), {"text": t})() for t in texts]
    class FakeLLM:
        def generate(self, prompts, sp):
            res = []
            for p, s in zip(prompts, cur_steps):
                act = json.dumps(s.expert_action) if isinstance(s.expert_action, dict) else None
                body = f"<tool_call>\n{act}\n</tool_call>" if act else str(s.expert_action)
                res.append(FakeOut([f"<think>\nOkay, {s.claude_think}\n</think>\n{body}"] * sp.n))
            return res
    llm = FakeLLM()
    from types import SimpleNamespace
    sp = SimpleNamespace(n=args.n)
else:
    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=args.max_model_len, gpu_memory_utilization=0.92,
              enable_prefix_caching=True, seed=1234 + args.shard)
    sp = SamplingParams(n=args.n, temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, max_tokens=args.max_tokens)

t_all = time.time(); n_acc = 0; n_done = 0
with open(out_path, "a") as fo:
    for c0 in range(0, len(steps), args.chunk):
        cur_steps = steps[c0: c0 + args.chunk]
        prompts = []
        for s in cur_steps:
            p = render_prompt(tok, s, args.rung)
            ntok = len(tok(p, add_special_tokens=False).input_ids)
            if ntok + args.max_tokens > args.max_model_len:      # too long: fall back to L1 (no Claude text) then give up
                p = render_prompt(tok, s, "L1")
            prompts.append(p)
        t0 = time.time()
        outs = llm.generate(prompts, sp)
        dt = time.time() - t0
        gen_tok = 0
        for s, o in zip(cur_steps, outs):
            rec = s.to_json(); rec.pop("history"); rec.pop("system")   # keep records small; history is rebuilt from the data
            rec["rung"] = args.rung; rec["attempts"] = []; rec["accepted"] = None
            acc = []
            for k, cand in enumerate(o.outputs):
                text = cand.text
                gen_tok += len(tok(text, add_special_tokens=False).input_ids) if args.dry_run else len(cand.token_ids)
                think, call, tail = parse_output(text)
                fails = check_sample(s, think, call, tail)
                att = {"k": k, "think": think, "tool_call": call, "text": tail, "fails": fails,
                       "think_words": len(think.split()) if think else None}
                rec["attempts"].append(att)
                if not fails:
                    acc.append(att)
            if acc:
                rec["accepted"] = min(acc, key=lambda a: a["think_words"]); n_acc += 1
            n_done += 1
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fo.flush()
        print(f"chunk {c0//args.chunk}: {len(cur_steps)} steps in {dt:.0f}s ({gen_tok/max(dt,1e-6):.0f} gen tok/s) | "
              f"accepted so far {n_acc}/{n_done} ({100*n_acc/max(n_done,1):.1f}%) | elapsed {(time.time()-t_all)/60:.1f} min", flush=True)
print(f"DONE shard {args.shard}: {n_acc}/{n_done} accepted in {(time.time()-t_all)/60:.1f} min -> {out_path}", flush=True)
sys.stdout.flush(); sys.stderr.flush()
os._exit(0)   # vLLM 0.9 workers can hang at interpreter shutdown and hold the GPU until the slurm time limit
