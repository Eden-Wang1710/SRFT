#!/usr/bin/env python
"""Re-score saved MMLU-Pro responses with the OFFICIAL TIGER-AI-Lab/MMLU-Pro extraction
(evaluate_from_local.py: extract_answer -> extract_again -> extract_final), which lm-eval 0.4.9 only implements
the first tier of. Verified against the upstream file on 2026-09-15.

    python eval_general/score_mmlu_pro_official.py            # base default, SR default, SR CoT-prefill
"""
import glob, json, math, os, re, sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))


def extract_official(text):
    m = re.search(r"answer is \(?([A-J])\)?", text)               # tier 1 (what lm-eval uses)
    if m:
        return m.group(1), 1
    m = re.search(r".*[aA]nswer:\s*([A-J])", text)                 # tier 2
    if m:
        return m.group(1), 2
    m = re.search(r"\b[A-J]\b(?!.*\b[A-J]\b)", text, re.DOTALL)    # tier 3: last standalone letter
    if m:
        return m.group(0), 3
    return None, 0


def score(tag, prefix):
    per = defaultdict(lambda: [0, 0]); tiers = defaultdict(int); n = c = 0
    for f in sorted(glob.glob(os.path.join(ROOT, "samples", tag, "*", f"samples_{prefix}_*.jsonl"))):
        sub = re.search(rf"samples_{prefix}_(.+?)_\d{{4}}-", os.path.basename(f)).group(1)
        if prefix == "mmlu_pro" and sub.startswith("cot_"):   # the glob also matches the mmlu_pro_cot variant files
            continue
        for l in open(f):
            r = json.loads(l); ans, t = extract_official(r["resps"][0][0]); tiers[t] += 1
            ok = ans == str(r["target"]).strip()
            per[sub][0] += 1; per[sub][1] += ok; n += 1; c += ok
    return c / n * 100, n, dict(tiers), {k: v[1] / v[0] * 100 for k, v in per.items()}


rows = [("base default", "base", "mmlu_pro"), ("SR default", "srllama", "mmlu_pro"), ("SR CoT-prefill", "srllama", "mmlu_pro_cot")]
res = {name: score(tag, p) for name, tag, p in rows}
print(f"{'run':16s} {'official':>9s} {'n':>5s}   tier hits (1=answer is, 2=Answer:, 3=last letter, 0=none)")
for name in res:
    acc, n, tiers, _ = res[name]
    print(f"{name:16s} {acc:9.2f} {n:5d}   {tiers}")
b = res["base default"][0]
se = lambda p, n: math.sqrt(p / 100 * (1 - p / 100) / n) * 100
print()
for name in list(res)[1:]:
    s, n = res[name][0], res[name][1]
    se2 = 2 * math.sqrt(se(b, n) ** 2 + se(s, n) ** 2)
    print(f"{name:16s} vs base default: {s - b:+.2f}  (2 se {se2:.2f}) -> {'PARITY' if abs(s - b) <= se2 else 'outside 2 se'}")
print("\nper subject (official extraction):")
print(f"{'subject':18s} {'base':>6s} {'SR-def':>7s} {'SR-CoT':>7s}")
for sub in sorted(res["base default"][3]):
    print(f"{sub:18s} {res['base default'][3][sub]:6.1f} {res['SR default'][3][sub]:7.1f} {res['SR CoT-prefill'][3][sub]:7.1f}")
