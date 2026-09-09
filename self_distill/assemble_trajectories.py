"""Merge shard outputs back into FULL trajectories in the original ShareGPT format (same schema as toucan_32B_v2.json).

python assemble_trajectories.py --runs runs/sdL2 --out ../LLaMA-Factory/data/toucan_32B_v2_sdL2.json
Per assistant step: accepted Qwen think + expert action (tool-call steps keep the original JSON text; final steps use Qwen's
answer). Steps with no accepted sample keep the original Claude think+action (counted as fallback).
"""
import argparse, glob, json, collections, statistics as st
from pathlib import Path
from common import load_data, split_assistant_value, THINK_RE, Step, check_sample

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default="runs/sdL2")
ap.add_argument("--out", required=True)
ap.add_argument("--no-refilter", action="store_true", help="trust stored accepted flags instead of re-running current filters")
ap.add_argument("--drop-fallback-trajs", action="store_true", help="drop trajectories that contain any fallback step")
args = ap.parse_args()

acc = {}
for f in glob.glob(str(Path(args.runs) / "shard*.jsonl")):
    for line in open(f):
        r = json.loads(line)
        if not args.no_refilter:   # re-evaluate every stored attempt with the CURRENT filters in common.py
            st_ = Step(traj_idx=r["traj_idx"], step_idx=r["step_idx"], platform=r["platform"], kind=r["kind"], situation=r["situation"],
                       injection_snippet=r["injection_snippet"], expert_action=r["expert_action"], claude_think=r["claude_think"], history=[])
            ok = []
            for a in r["attempts"]:
                a["fails"] = check_sample(st_, a["think"], a["tool_call"], a["text"])
                if not a["fails"]:
                    ok.append(a)
            r["accepted"] = min(ok, key=lambda a: a["think_words"]) if ok else None
        acc[(r["traj_idx"], r["step_idx"])] = r

data = load_data()
out, stats = [], collections.Counter()
lens_q, lens_c = [], []
per_sit = collections.defaultdict(collections.Counter)
for ti, s in enumerate(data):
    conv, a_idx, fallback = [], 0, 0
    for m in s["conversations"]:
        if m["from"] in ("function_call", "gpt"):
            r = acc.get((ti, a_idx)); a_idx += 1
            claude_think, rest = split_assistant_value(m["value"])
            lens_c.append(len(claude_think.split()))
            if r and r["accepted"]:
                a = r["accepted"]
                body = rest   # ALWAYS the expert action/answer text; only the think is replaced (final-answer steps included, 2026-09-06 fix)
                conv.append({"from": m["from"], "value": f"<think>\n{a['think']}\n</think>\n\n{body}"})
                stats["steps_rewritten"] += 1; lens_q.append(a["think_words"]); per_sit[r["situation"]]["ok"] += 1
            else:
                conv.append(dict(m)); fallback += 1
                stats["steps_fallback_missing" if not r else "steps_fallback_rejected"] += 1
                if r: per_sit[r["situation"]]["rejected"] += 1
        else:
            conv.append(dict(m))
    if fallback and args.drop_fallback_trajs:
        stats["trajs_dropped"] += 1; continue
    meta = dict(s["meta"]); meta["self_distill"] = {"rung": "L2", "fallback_steps": fallback, "total_steps": a_idx}
    out.append({"conversations": conv, "system": s["system"], "meta": meta})
    stats["trajs_written"] += 1; stats["trajs_with_fallback"] += bool(fallback)

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
json.dump(out, open(args.out, "w"), indent=1, ensure_ascii=False)
summary = {
    "stats": dict(stats),
    "per_situation": {k: dict(v) for k, v in per_sit.items()},
    "qwen_think_words": {"mean": round(st.mean(lens_q), 1), "median": st.median(lens_q)} if lens_q else None,
    "claude_think_words": {"mean": round(st.mean(lens_c), 1), "median": st.median(lens_c)},
}
json.dump(summary, open(Path(args.out).with_suffix(".stats.json"), "w"), indent=1)
print(json.dumps(summary, indent=1)); print("wrote", args.out, len(out), "trajectories")
