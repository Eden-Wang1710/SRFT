"""Stratified pilot set for the paraphrase recipe -> pilot_para/steps.json (50 step0 / 50 clean / 75 injected / 25 final)."""
import json, random
from pathlib import Path
from common import load_data, extract_steps
random.seed(7); want = {"step0": 50, "clean": 50, "injected": 75, "final": 25}; picked = []
data = load_data(); idx = list(range(len(data))); random.shuffle(idx)
for ti in idx:
    for s in extract_steps(data[ti], ti):
        k = "final" if s.kind == "final" else s.situation
        if k == "final" and not s.expert_action: continue           # skip empty-reply finals
        if want.get(k, 0) > 0 and random.random() < 0.3:
            picked.append(s); want[k] -= 1
    if sum(want.values()) == 0: break
Path("pilot_para").mkdir(exist_ok=True)
json.dump([s.to_json() for s in picked], open("pilot_para/steps.json", "w"), ensure_ascii=False)
print("picked", len(picked), {k: sum(1 for s in picked if (('final' if s.kind=='final' else s.situation) == k)) for k in ["step0","clean","injected","final"]})
