"""Re-judge stored attempts of a paraphrase run with the CURRENT filters (no GPU). python refilter.py <steps.json> <out.jsonl> [<new_out.jsonl>]"""
import json, sys
from common import Step
from paraphrase import check
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
steps = {(d["traj_idx"], d["step_idx"]): Step(**d) for d in json.load(open(sys.argv[1]))}
recs = [json.loads(l) for l in open(sys.argv[2])]
def quotes_injection(s, text):
    if s.situation != "injected" or not s.injection_snippet: return True
    q = s.injection_snippet.split(); o = text.lower()
    return any(" ".join(q[i:i+6]).lower() in o for i in range(max(1, len(q) - 5)))
def pick(ok, s): return min(ok, key=lambda x: (not quotes_injection(s, x["text"]), x["metrics"].get("pasted_json", False), abs(x["metrics"]["len_ratio"] - 1.0), x["metrics"]["overlap8"]))
n = 0
for r in recs:
    s = steps[(r["traj_idx"], r["step_idx"])]; ok = []
    for a in r["attempts"]:
        a["fails"], a["metrics"] = check(s, a["text"], tok)
        if not a["fails"]: ok.append(a)
    r["accepted"] = pick(ok, s) if ok else None; n += bool(r["accepted"])
print(f"refiltered: accepted {n}/{len(recs)}")
out = sys.argv[3] if len(sys.argv) > 3 else sys.argv[2]
with open(out, "w") as f:
    for r in recs: f.write(json.dumps(r, ensure_ascii=False) + "\n")
