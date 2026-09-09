"""Explode multi-turn ShareGPT trajectories into per-step samples for inference-consistent SFT:
history assistant turns lose their <think> (like at inference), only the target turn keeps think+action; train with mask_history: true.
python explode_per_step.py --inp ../LLaMA-Factory/data/toucan_32B_v2_sdL2.json --out ../LLaMA-Factory/data/toucan_32B_v2_sdL2_perstep.json
"""
import argparse, json, re
ap = argparse.ArgumentParser(); ap.add_argument("--inp", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
THINK = re.compile(r"^<think>\n.*?\n</think>\n*", re.S)
data = json.load(open(a.inp)); out = []
for ti, s in enumerate(data):
    conv = s["conversations"]; hist = []
    for i, m in enumerate(conv):
        if m["from"] in ("function_call", "gpt"):
            stripped = THINK.sub("", m["value"], count=1)
            if m["from"] == "gpt" and not stripped.strip():
                # final turn with no text after the think (454 such source steps): skip as a target, keep history as-is
                hist.append({"from": m["from"], "value": stripped})
                continue
            sample = {"conversations": hist + [m], "system": s["system"],
                      "meta": {**s.get("meta", {}), "traj_idx": ti, "step_idx": sum(1 for x in conv[:i] if x["from"] in ("function_call", "gpt"))}}
            out.append(sample)
            hist.append({"from": m["from"], "value": stripped})     # think-free history for later steps
        else:
            hist.append(m)
json.dump(out, open(a.out, "w"), ensure_ascii=False)
print(f"{len(data)} trajectories -> {len(out)} per-step samples -> {a.out}")
