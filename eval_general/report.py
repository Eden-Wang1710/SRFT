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
# Metrics reported per task. IFEval has four and the published 79.1 is their MEAN: our four are
# 73.01 / 81.06 / 78.19 / 84.77, mean 79.26, which lands 0.16 from the published figure while no single
# sub-metric does (inst-loose alone is 5.67 high). So the headline for IFEval is the mean, and check 2
# compares all four separately, since a mean has no honest standard error.
METRICS = {
    "mmlu": ["acc,none"],
    "mmlu_pro": ["exact_match,custom-extract"],
    "ifeval": ["prompt_level_strict_acc,none", "inst_level_strict_acc,none",
               "prompt_level_loose_acc,none", "inst_level_loose_acc,none"],
    "bbh_cot_fewshot": ["exact_match,none"],
}
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
    keys = [k for k in METRICS[task] if k in a]
    if not keys:  # fall back to whatever non-stderr floats are present
        keys = [k for k, v in a.items() if isinstance(v, float) and "stderr" not in k][:1]
    def num(x):
        # lm-eval writes the literal string "N/A" for a stderr it did not compute (seen on IFEval)
        try:
            return 100 * float(x)
        except (TypeError, ValueError):
            return float("nan")

    vals = {k: num(a[k]) for k in keys}
    errs = {k: num(a.get(k.replace(",", "_stderr,"))) for k in keys}
    n = d.get("n-samples", {})
    return {
        "path": path,
        "task": task,
        "template": d.get("chat_template_sha") is not None,
        "vals": vals,
        "errs": errs,
        "keys": keys,
        # headline = the mean across reported metrics; for every task but IFEval that is the single metric
        "score": (sum(vals.values()) / len(vals)) if vals and not any(math.isnan(v) for v in vals.values()) else None,
        "items": sum(v.get("effective", 0) for v in n.values()),
        "limit": d.get("config", {}).get("limit"),
        "date": os.path.basename(path)[8:27],
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
    print(f"  {'task':<10} {'published':>9} {'our base':>9} {'delta':>7}   verdict   (IFEval = mean of its 4 sub-metrics)")
    for task in TASKS:
        r = picks[("base", task)]
        ref = REFERENCE[task]
        if r is None:
            print(f"  {LABEL[task]:<10} {ref:9.1f} {'pending':>9} {'':>7}   waiting for the run")
            continue
        d = r["score"] - ref
        v = "comparable" if abs(d) <= args.tol else f"OFF by {abs(d):.1f} — do not cite their defended row"
        print(f"  {LABEL[task]:<10} {ref:9.1f} {r['score']:9.2f} {d:+7.2f}   {v}")

    print("\nCheck 2 — does SR-Agent-Llama lose general ability vs OUR base? (same harness config)")
    print("  IFEval is shown per sub-metric; parity must hold on each.\n")
    print(f"  {'task':<10} {'metric':<22} {'base':>8} {'SR-Agent':>8} {'delta':>7} {'2*se':>6}   verdict")
    for task in TASKS:
        b, s = picks[("base", task)], picks[("srllama", task)]
        if b is None or s is None:
            have = "base only" if b else ("SR only" if s else "neither")
            print(f"  {LABEL[task]:<10} {'':<22} {fmt(b['score'] if b else None,8)} {fmt(s['score'] if s else None,8)} {'':>7} {'':>6}   pending ({have})")
            continue
        if b["items"] != s["items"]:
            print(f"  {LABEL[task]:<10} !! item counts differ: base {b['items']} vs SR {s['items']} — NOT comparable")
            continue
        for k in b["keys"]:
            if k not in s["vals"]:
                continue
            d = s["vals"][k] - b["vals"][k]
            se2 = 2 * math.sqrt(b["errs"][k] ** 2 + s["errs"][k] ** 2)
            if math.isnan(se2):
                v = "no stderr reported"
            elif abs(d) <= se2:
                v = "PARITY (inside 2 se)"
            else:
                v = ("DROP" if d < 0 else "gain") + f" of {abs(d):.2f}, outside 2 se"
            print(f"  {LABEL[task]:<10} {k.split(',')[0]:<22} {b['vals'][k]:8.2f} {s['vals'][k]:8.2f} {d:+7.2f} {fmt(se2,6)}   {v}")
        if len(b["keys"]) > 1:
            print(f"  {'':<10} {'mean of the above':<22} {b['score']:8.2f} {s['score']:8.2f} {s['score']-b['score']:+7.2f}")

    print("\nProvenance of every number above")
    for task in TASKS:
        for tag in ("base", "srllama"):
            r = picks[(tag, task)]
            if r is None:
                continue
            print(f"  {tag:<8} {LABEL[task]:<10} {r['date']}  template={'on ' if r['template'] else 'off'}"
                  f"  items={r['items']:<6} limit={str(r['limit']):<6} metrics={','.join(k.split(',')[0] for k in r['keys'])}")

    if strays:
        print("\nOther results files present but NOT used (wrong template setting for the protocol)")
        for tag, r in strays:
            print(f"  {tag:<8} {LABEL[r['task']]:<10} {r['date']}  template={'on' if r['template'] else 'off'}"
                  f"  score={r['score']:.2f}  {r['path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
