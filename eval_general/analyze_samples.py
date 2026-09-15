#!/usr/bin/env python
"""Is the SR-Agent-Llama drop on BBH / MMLU-Pro an answer-EXTRACTION failure or a knowledge failure?

Reads the per-item samples that `env/jobs/lmeval.sbatch ... LOG_SAMPLES=1` writes (one jsonl per subtask,
`eval_general/samples/<tag>/<model>/samples_<subtask>_<ts>.jsonl`) for both models and classifies every
item under the harness's STRICT filter:

  correct       strict filter extracted the target
  format_only   the response names the target after "answer is", but the strict regex/exact-match missed it
                (capitalised "The answer is", markdown bold, missing parentheses, no trailing period, ...)
  no_phrase     the response never says "answer is" at all -> truncated by a stop string / max_gen_toks,
                or the model ended without the required sentence
  wrong         a robust extractor finds an answer and it is not the target  (the only genuine failure)

It also re-scores both models with the ROBUST extractor (last "answer is ..." in the response, case- and
markdown-insensitive, parentheses optional) so the two models can be compared under one tolerant protocol.
Both models are always treated identically; the point is the symmetric comparison, not a favourable reading.

    python eval_general/analyze_samples.py --task bbh
    python eval_general/analyze_samples.py --task mmlu_pro
    python eval_general/analyze_samples.py --task bbh --show tracking_shuffled_objects_seven_objects --n 5
    python eval_general/analyze_samples.py --task mmlu_pro_cot     # SR under the CoT prefill vs the base DEFAULT run

Robust extraction, when the response never says "answer is" but the target is an option letter, falls back to the
last conclusion-like phrase ("I will choose B", "best matches this is: I", "correct option is (C)", a final line
"B. text"). Applied to both models alike; strict and robust are always printed side by side.
"""
import argparse
import glob
import json
import os
import re
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
SAMPLES = os.path.join(ROOT, "samples")
TAGS = ("base", "srllama")

# strict = exactly what lm-eval 0.4.9 applies (bbh: get-answer regex + exact_match; mmlu_pro: custom-extract)
PHRASE = re.compile(r"answer is", re.I)
ROBUST_ANY = re.compile(r"answer is\s*:?\s*(.+)", re.I)
LETTER = re.compile(r"^\(?([A-J])\)?(?:[\s.:,)]|$)")
# conclusion-like phrases for letter-option tasks when "answer is" is absent (last match wins)
FALLBACK = re.compile(
    r"(?:(?:correct|best|final|right)\s+(?:answer|option|choice)\s*(?:is|would be|should be|:)\s*[:\-]?|"
    r"(?:I(?:'d| would| will)?\s+)?(?:choose|select|go with|pick)\s*(?:option)?|"
    r"(?:matches|corresponds to|is)\s*(?:option)?\s*[:\-]?|option|answer:)\s*\**\(?([A-J])\)?(?![a-zA-Z])",
    re.I)
FINAL_LINE = re.compile(r"^\**\(?([A-J])\)?[.):]\s+\S", re.M)


def _clean(s: str) -> str:
    s = s.strip()
    s = re.sub(r"[*_`]+", "", s)          # markdown emphasis / code
    s = s.strip().rstrip(".").strip()
    s = s.strip('"\'')
    return s


def robust_extract(resp: str, target: str):
    """Last 'answer is ...' in the response, normalised; for letter-option targets, fall back to the last
    conclusion-like phrase. Returns None if nothing answer-like is found."""
    letter_target = re.fullmatch(r"\(?[A-J]\)?", target.strip()) is not None
    m = ROBUST_ANY.findall(resp)
    if m:
        ans = _clean(m[-1].splitlines()[0])
        if letter_target:
            lm = LETTER.match(ans)
            if lm:
                return f"({lm.group(1)})"
            # "the answer is: \n B. text" or "the answer is the first option" -> keep looking below
        else:
            return ans
    if letter_target:
        fb = FALLBACK.findall(resp)
        if fb:
            return f"({fb[-1].upper()})"
        fl = FINAL_LINE.findall(resp.strip().splitlines()[-1]) if resp.strip() else []
        if fl:
            return f"({fl[-1].upper()})"
        return None
    return None


def robust_match(ans, target: str) -> bool:
    if ans is None:
        return False
    t = _clean(target)
    if re.fullmatch(r"\(?[A-J]\)?", t):
        return ans == f"({t.strip('()')})"
    a = ans.lower().replace(",", "")
    t = t.lower().replace(",", "")
    return a == t or a.startswith(t + " ") or a.split(" ")[0] == t


def classify(rec, task):
    resp = rec["resps"][0][0]
    target = str(rec["target"])
    strict_ok = float(rec["exact_match"]) == 1.0
    if strict_ok:
        return "correct"
    ans = robust_extract(resp, target)   # includes the conclusion-phrase fallback for letter targets
    if ans is None:
        return "no_phrase"
    if robust_match(ans, target):
        return "format_only"
    return "wrong"


def load(tag, task):
    pat = os.path.join(SAMPLES, tag, "*", f"samples_{task}_*.jsonl")
    out = {}
    for f in sorted(glob.glob(pat)):
        sub = re.search(rf"samples_({task}_.+?)_\d{{4}}-\d\d-\d\dT", os.path.basename(f)).group(1)
        with open(f) as fh:
            out[sub] = [json.loads(l) for l in fh]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["bbh", "mmlu_pro", "mmlu_pro_cot", "bbh_relaxed"], required=True,
                    help="mmlu_pro_cot / bbh_relaxed = SR under the protocol variant vs the base DEFAULT samples")
    ap.add_argument("--show", help="print raw failing responses for this subtask (suffix after the task prefix)")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--tag", default="srllama")
    a = ap.parse_args()
    base_prefix = "bbh_cot_fewshot" if a.task.startswith("bbh") else "mmlu_pro"
    sr_prefix = {"bbh": "bbh_cot_fewshot", "mmlu_pro": "mmlu_pro", "mmlu_pro_cot": "mmlu_pro_cot",
                 "bbh_relaxed": "bbh_cot_fewshot_relaxed"}[a.task]
    prefix = base_prefix
    data = {"base": load("base", base_prefix), "srllama": load("srllama", sr_prefix)}
    if sr_prefix != base_prefix:  # re-key the SR subtasks onto the base names so they line up
        data["srllama"] = {k.replace(sr_prefix, base_prefix, 1): v for k, v in data["srllama"].items()}
        print(f"NOTE: SR samples are the {sr_prefix} variant; base samples are the DEFAULT {base_prefix} run")
    for t in TAGS:
        if not data[t]:
            raise SystemExit(f"no samples for {t} under {SAMPLES}")
    subs = sorted(set(data["base"]) & set(data["srllama"]))

    if a.show:
        sub = f"{prefix}_{a.show}"
        shown = Counter()
        for rec in data[a.tag][sub]:
            c = classify(rec, a.task)
            if c == "correct" or shown[c] >= a.n:
                continue
            shown[c] += 1
            resp = rec["resps"][0][0]
            print(f"\n===== [{c}] target={rec['target']!r} strict_extracted={rec['filtered_resps'][0]!r} "
                  f"robust={robust_extract(resp, str(rec['target']))!r} len={len(resp)} chars")
            print(resp[-1200:] if len(resp) > 1200 else resp)
        return

    print(f"task={prefix}  items per model: base={sum(len(v) for v in data['base'].values())} "
          f"srllama={sum(len(v) for v in data['srllama'].values())}")
    print("\nSTRICT = lm-eval's own filter (the reported number). ROBUST = last 'answer is ...', case/markdown/paren-tolerant, "
          "applied identically to both models.\n")
    hdr = f"{'subtask':42s} {'n':>4s} | {'strict':^15s} | {'robust':^15s} | SR failures under strict: no_phrase format_only wrong"
    print(hdr)
    print(f"{'':42s} {'':>4s} | {'base':>6s} {'SR':>6s}   | {'base':>6s} {'SR':>6s}   |")
    tot = defaultdict(float)
    tot_n = 0
    tot_cls = {t: Counter() for t in TAGS}
    rows = []
    for sub in subs:
        n = len(data["srllama"][sub])
        if n != len(data["base"][sub]):
            print(f"!! item-count mismatch on {sub}: base {len(data['base'][sub])} vs SR {n}")
        acc = {}
        cls = {}
        for t in TAGS:
            recs = data[t][sub]
            acc[t, "strict"] = sum(float(r["exact_match"]) for r in recs) / len(recs) * 100
            acc[t, "robust"] = sum(robust_match(robust_extract(r["resps"][0][0], str(r["target"])), str(r["target"]))
                                   for r in recs) / len(recs) * 100
            cls[t] = Counter(classify(r, a.task) for r in recs)
            tot_cls[t].update(cls[t])
        for k, v in acc.items():
            tot[k] += v * n
        tot_n += n
        rows.append((acc["srllama", "strict"] - acc["base", "strict"], sub, n, acc, cls))
    rows.sort()
    for d, sub, n, acc, cls in rows:
        c = cls["srllama"]
        print(f"{sub.replace(prefix + '_', ''):42s} {n:4d} | {acc['base','strict']:6.1f} {acc['srllama','strict']:6.1f} "
              f"({d:+5.1f}) | {acc['base','robust']:6.1f} {acc['srllama','robust']:6.1f} "
              f"({acc['srllama','robust'] - acc['base','robust']:+5.1f}) | "
              f"{c['no_phrase']:9d} {c['format_only']:11d} {c['wrong']:5d}")
    print("-" * len(hdr))
    m = {k: v / tot_n for k, v in tot.items()}
    print(f"{'MEAN over items':42s} {tot_n:4d} | {m['base','strict']:6.2f} {m['srllama','strict']:6.2f} "
          f"({m['srllama','strict'] - m['base','strict']:+5.2f}) | {m['base','robust']:6.2f} {m['srllama','robust']:6.2f} "
          f"({m['srllama','robust'] - m['base','robust']:+5.2f}) |")
    print("\nFailure classes under STRICT scoring, whole task:")
    for t in TAGS:
        c = tot_cls[t]
        n = sum(c.values())
        print(f"  {t:8s} correct {c['correct']:5d} ({c['correct']/n*100:5.1f}%)  no_phrase {c['no_phrase']:5d} ({c['no_phrase']/n*100:5.1f}%)  "
              f"format_only {c['format_only']:5d} ({c['format_only']/n*100:5.1f}%)  wrong {c['wrong']:5d} ({c['wrong']/n*100:5.1f}%)")
    # how much of the SR-vs-base strict gap is extraction (no_phrase + format_only) vs genuinely wrong
    b, s = tot_cls["base"], tot_cls["srllama"]
    n = tot_n
    gap = (s["correct"] - b["correct"]) / n * 100
    print(f"\nSTRICT gap SR-base = {gap:+.2f} pts, decomposed into class differences (SR minus base, in points of accuracy):")
    for k in ("no_phrase", "format_only", "wrong"):
        print(f"  {k:12s} {-(s[k] - b[k]) / n * 100:+.2f}")


if __name__ == "__main__":
    main()
