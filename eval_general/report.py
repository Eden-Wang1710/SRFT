#!/usr/bin/env python3
"""Two acceptance checks for the general-capability benchmarks (docs/14).

  Check 1 (sanity of OUR harness): base Llama-3.1-8B-Instruct vs the published base row
          of the Meta-SecAlign / ReasAlign papers. If our base lands near theirs, the
          config is comparable and their defended row can be cited against ours.
  Check 2 (the actual claim): SR-Agent-Llama vs OUR base, same harness config.
          Target is parity, not a win. A gap inside 2 combined standard errors is parity.

Usage:  python eval_general/report.py [--dir eval_general] [--tol 5.0]

Which results file is canonical per (tag, task) follows the protocol table in docs/14:
MMLU is scored WITHOUT the chat template, everything else WITH it. The marker is the
top-level "chat_template_sha" key in the results JSON: null = no template, a sha = template.
"""
import argparse, glob, json, math, os, sys

# Published base row for Llama-3.1-8B-Instruct (Meta-SecAlign paper's table; see docs/99 2026-09-14).
REFERENCE = {"mmlu": 72.0, "mmlu_pro": 46.5, "ifeval": 79.1, "bbh_cot_fewshot": 71.9}
# Protocol: does the canonical run use --apply_chat_template?  (docs/14 "Chat-template decision")
WANT_TEMPLATE = {"mmlu": False, "mmlu_pro": True, "ifeval": True, "bbh_cot_fewshot": True}
# Primary metric key per task. IFEval's published 79.1 is the instruction-level loose score.
METRIC = {
    "mmlu": "acc,none",
    "mmlu_pro": "exact_match,custom-extract",
    "ifeval": "inst_level_loose_acc,none",
    "bbh_cot_fewshot": "exact_match,none",
}
EXTRA = {"ifeval": ["inst_level_strict_acc,none", "prompt_level_loose_acc,none", "prompt_level_strict_acc,none"]}
TASKS = ["mmlu", "mmlu_pro", "ifeval", "bbh_cot_fewshot"]
LABEL = {"mmlu": "MMLU", "mmlu_pro": "MMLU-Pro", "ifeval": "IFEval", "bbh_cot_fewshot": "BBH"}


def agg(d, task):
    """Aggregate block for a task, whether lm-eval filed it under results or groups."""
    for block in ("results", "groups"):
        if task in d.get(block, {}):
            return d[block][task]
    return {}


def load(path):
    d = json.load(open(path))
    task = os.path.basename(os.path.dirname(os.path.dirname(path)))
    a = agg(d, task)
    key = METRIC[task]
    if key not in a:  # fall back to the only non-stderr float present
        cands = [k for k, v in a.items() if isinstance(v, float) and "stderr" not in k]
        key = cands[0] if cands else None
    n = d.get("n-samples", {})
    return {
        "path": path,
        "task": task,
        "template": d.get("chat_template_sha") is not None,
        "score": 100 * a[key] if key else None,
        "stderr": 100 * a.get(key.replace(",", "_stderr,"), float("nan")) if key else float("nan"),
        "key": key,
        "items": sum(v.get("effective", 0) for v in n.values()),
        "limit": d.get("config", {}).get("limit"),
        "date": os.path.basename(path)[8:27],
        "extra": {k: 100 * a[k] for k in EXTRA.get(task, []) if k in a},
    }


def pick(root, tag, task):
    """Newest results file for (tag, task) matching the protocol's template setting."""
    files = glob.glob(os.path.join(root, tag, task, "*", "results_*.json"))
    runs = sorted((load(f) for f in files), key=lambda r: r["date"])
    matching = [r for r in runs if r["template"] == WANT_TEMPLATE[task]]
    if matching:
        return matching[-1], [r for r in runs if r not in matching]
    return None, runs


def fmt(x, w=6):
    return " " * w if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:{w}.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--tol", type=float, default=5.0, help="points of slack allowed in check 1")
    args = ap.parse_args()

    picks, strays = {}, []
    for task in TASKS:
        for tag in ("base", "srllama"):
            r, other = pick(args.dir, tag, task)
            picks[(tag, task)] = r
            strays += [(tag, o) for o in other]

    print("Check 1 — does OUR base reproduce the published Llama-3.1-8B-Instruct row?")
    print(f"  (tolerance {args.tol:.1f} points; MMLU 68 vs 72 is the user's stated example of acceptable)\n")
    print(f"  {'task':<10} {'published':>9} {'our base':>9} {'delta':>7}   verdict")
    for task in TASKS:
        r = picks[("base", task)]
        ref = REFERENCE[task]
        if r is None:
            print(f"  {LABEL[task]:<10} {ref:9.1f} {'pending':>9} {'':>7}   waiting for the run")
            continue
        d = r["score"] - ref
        v = "comparable" if abs(d) <= args.tol else f"OFF by {abs(d):.1f} — do not cite their defended row"
        print(f"  {LABEL[task]:<10} {ref:9.1f} {r['score']:9.2f} {d:+7.2f}   {v}")

    print("\nCheck 2 — does SR-Agent-Llama lose general ability vs OUR base? (same harness config)\n")
    print(f"  {'task':<10} {'base':>9} {'SR-Agent':>9} {'delta':>7} {'2*se':>6}   verdict")
    for task in TASKS:
        b, s = picks[("base", task)], picks[("srllama", task)]
        if b is None or s is None:
            have = "base only" if b else ("SR only" if s else "neither")
            print(f"  {LABEL[task]:<10} {fmt(b['score'] if b else None,9)} {fmt(s['score'] if s else None,9)} {'':>7} {'':>6}   pending ({have})")
            continue
        d = s["score"] - b["score"]
        se2 = 2 * math.sqrt(b["stderr"] ** 2 + s["stderr"] ** 2) if not math.isnan(b["stderr"]) else float("nan")
        if math.isnan(se2):
            v = "no stderr reported"
        elif abs(d) <= se2:
            v = "PARITY (inside 2 se)"
        else:
            v = ("DROP" if d < 0 else "gain") + f" of {abs(d):.2f}, outside 2 se"
        print(f"  {LABEL[task]:<10} {b['score']:9.2f} {s['score']:9.2f} {d:+7.2f} {fmt(se2,6)}   {v}")
        if b["items"] != s["items"]:
            print(f"  {'':<10} !! item counts differ: base {b['items']} vs SR {s['items']} — not comparable")

    print("\nProvenance of every number above")
    for task in TASKS:
        for tag in ("base", "srllama"):
            r = picks[(tag, task)]
            if r is None:
                continue
            extra = "  " + " ".join(f"{k.split(',')[0]}={v:.2f}" for k, v in r["extra"].items()) if r["extra"] else ""
            print(f"  {tag:<8} {LABEL[task]:<10} {r['date']}  template={'on ' if r['template'] else 'off'}"
                  f"  items={r['items']:<6} limit={str(r['limit']):<6} metric={r['key']}{extra}")

    if strays:
        print("\nOther results files present but NOT used (wrong template setting for the protocol)")
        for tag, r in strays:
            print(f"  {tag:<8} {LABEL[r['task']]:<10} {r['date']}  template={'on' if r['template'] else 'off'}"
                  f"  score={r['score']:.2f}  {r['path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
