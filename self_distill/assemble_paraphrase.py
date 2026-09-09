"""v3: merge paraphrase shard outputs into full trajectories (ShareGPT, same schema as toucan_32B_v3_base.json).
python assemble_paraphrase.py --runs runs/v3para --out ../LLaMA-Factory/data/toucan_32B_v3_para.json
Think = accepted paraphrase (else the Claude think, counted as fallback); tool calls / answers = expert text, untouched.
"""
import argparse, glob, json, collections, statistics as st
from pathlib import Path
from common import load_data, split_assistant_value
ap = argparse.ArgumentParser(); ap.add_argument("--runs", default="runs/v3para"); ap.add_argument("--out", required=True)
ap.add_argument("--drop-fallback-trajs", action="store_true"); a = ap.parse_args()
acc = {}
for f in sorted(glob.glob(str(Path(a.runs) / "out_shard*.jsonl"))) + sorted(glob.glob(str(Path(a.runs) / "rescue*.jsonl"))):
    for line in open(f):        # rescue files come last and override the shard record for that step
        r = json.loads(line); k = (r["traj_idx"], r["step_idx"])
        if k not in acc or r["accepted"] or not acc[k]["accepted"]: acc[k] = r
data = load_data(); out = []; S = collections.Counter(); per = collections.defaultdict(collections.Counter); wq, wc, tq = [], [], []
for ti, s in enumerate(data):
    conv, a_idx, fb = [], 0, 0
    for m in s["conversations"]:
        if m["from"] in ("function_call", "gpt"):
            r = acc.get((ti, a_idx)); a_idx += 1
            think, rest = split_assistant_value(m["value"]); wc.append(len(think.split()))
            key = "final" if m["from"] == "gpt" else (r["situation"] if r else "?")
            if r and r["accepted"]:
                t = r["accepted"]["text"]; conv.append({"from": m["from"], "value": f"<think>\n{t}\n</think>\n\n{rest}"})
                S["steps_paraphrased"] += 1; per[key]["ok"] += 1; wq.append(len(t.split())); tq.append(r["accepted"]["metrics"].get("tokens"))
            else:
                conv.append(dict(m)); fb += 1; S["steps_fallback_missing" if not r else "steps_fallback_rejected"] += 1; per[key]["fallback"] += 1
        else:
            conv.append(dict(m))
    if fb and a.drop_fallback_trajs: S["trajs_dropped"] += 1; continue
    meta = dict(s["meta"]); meta["self_distill"] = {"scheme": "paraphrase", "fallback_steps": fb, "total_steps": a_idx}
    out.append({"conversations": conv, "system": s["system"], "meta": meta}); S["trajs_written"] += 1; S["trajs_with_fallback"] += bool(fb)
Path(a.out).parent.mkdir(parents=True, exist_ok=True); json.dump(out, open(a.out, "w"), indent=1, ensure_ascii=False)
summary = {"stats": dict(S), "per_situation": {k: dict(v) for k, v in per.items()},
           "paraphrase_words": {"mean": round(st.mean(wq), 1), "median": st.median(wq)} if wq else None,
           "paraphrase_tokens": {"mean": round(st.mean([t for t in tq if t]), 1)} if any(tq) else None,
           "claude_words": {"mean": round(st.mean(wc), 1), "median": st.median(wc)}}
json.dump(summary, open(Path(a.out).with_suffix(".stats.json"), "w"), indent=1); print(json.dumps(summary, indent=1)); print("wrote", a.out, len(out))
