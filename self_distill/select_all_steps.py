"""Dump ALL assistant steps of the source data into N shard files for the full v3 paraphrase run -> runs/v3para/steps_shard{k}.json"""
import json, sys
from pathlib import Path
from common import load_data, extract_steps
N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
out = Path("runs/v3para"); out.mkdir(parents=True, exist_ok=True)
data = load_data(); steps = [s for ti, d in enumerate(data) for s in extract_steps(d, ti)]
print("steps", len(steps))
for k in range(N):
    part = [s.to_json() for s in steps[k::N]]
    json.dump(part, open(out / f"steps_shard{k}.json", "w"), ensure_ascii=False)
    print(f"shard {k}: {len(part)}")
