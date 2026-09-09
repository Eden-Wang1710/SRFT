"""Pick a small, diverse set of assistant steps for a smoke run -> smoke/steps.json"""
import json, random, sys
from pathlib import Path
from common import load_data, extract_steps

random.seed(0)
out = Path(__file__).parent / "smoke/steps.json"
data = load_data()
want = {"step0": 3, "clean": 3, "injected": 3, "final": 1}
picked, seen_platforms = [], set()
idxs = list(range(len(data))); random.shuffle(idxs)
for ti in idxs:
    steps = extract_steps(data[ti], ti)
    for s in steps:
        key = "final" if s.kind == "final" else s.situation
        if key == "final" and s.situation != "injected":
            continue
        if want.get(key, 0) > 0 and (s.platform, key) not in seen_platforms:
            if key == "injected" and not s.injection_snippet:
                continue
            picked.append(s); want[key] -= 1; seen_platforms.add((s.platform, key))
    if sum(want.values()) == 0:
        break
json.dump([s.to_json() for s in picked], open(out, "w"), indent=1, ensure_ascii=False)
for s in picked:
    print(f"traj={s.traj_idx:4d} step={s.step_idx} {s.platform:10s} {s.kind:9s} {s.situation:8s} prev_obs={s.n_prev_obs} "
          f"expert={(s.expert_action['name'] if isinstance(s.expert_action, dict) else 'FINAL')} snippet={str(s.injection_snippet)[:50]!r}")
print("wrote", out, len(picked))
