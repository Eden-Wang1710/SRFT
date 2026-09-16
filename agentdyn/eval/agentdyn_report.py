#!/usr/bin/env python3
"""AgentDyn metrics + failure-mode breakdown for one or more run dirs.

    python eval/agentdyn_report.py <RUN_NAME> [<RUN_NAME> ...]        (run from SRFT/agentdyn)
    python eval/agentdyn_report.py --runs-base runs_upstream_paper meta_secalign-8B

Conventions match the AgentDojo tables in docs/13:
  benign  = `*/user_task_*/none/`            (injection tasks run as user tasks are counted separately)
  UA/ASR  = `*/important_instructions/*`, ASR counts `security == True` (attack SUCCEEDED, as in this repo)
The breakdown separates genuine capability failures from harness artifacts: a trajectory whose tool call the parser
dropped is returned as a final answer and ends at 3 messages (see docs/15 §5).
"""
import argparse, collections, glob, json, os, re

CALLISH = re.compile(r'<function\s*=|<\|python_tag\|>|\{"name"\s*:')
SUITES = ("shopping", "github", "dailylife")


def assistant_text(d):
    out = []
    for m in d["messages"]:
        if m["role"] != "assistant":
            continue
        c = m.get("content")
        if isinstance(c, list):
            out += [str(b.get("content", "")) for b in c if isinstance(b, dict)]
        elif c:
            out.append(str(c))
    return " ".join(out)


def analyse(run_dir):
    per = collections.defaultdict(collections.Counter)
    modes = collections.Counter()
    msgs = []
    for f in glob.glob(os.path.join(run_dir, "**", "*.json"), recursive=True):
        p = f.replace("\\", "/")
        suite = next((s for s in SUITES if f"/{s}/" in p), None)
        if suite is None:
            continue
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if "utility" not in d:            # still running: the logger rewrites the file each step
            continue
        r = per[suite]
        if "/none/" in p and "/user_task_" in p:
            r["benign_n"] += 1
            r["benign_ok"] += bool(d["utility"])
            n = len(d["messages"]); msgs.append(n)
            if d["utility"]:
                modes["success"] += 1
            elif not any(m.get("tool_calls") for m in d["messages"]):
                modes["died at step 1: call dropped by parser" if CALLISH.search(assistant_text(d))
                      else "died at step 1: no call written"] += 1
            elif n >= 31:
                modes["ran out of turns"] += 1
            else:
                modes["stopped early / wrong answer"] += 1
        elif "/important_instructions/" in p:
            r["att_n"] += 1
            r["att_util"] += bool(d["utility"])
            r["att_sec"] += bool(d.get("security"))
    return per, modes, msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--runs-base", default="runs")
    args = ap.parse_args()
    for run in args.runs:
        root = os.path.join(args.runs_base, run)
        if not os.path.isdir(root):
            print(f"!! {root} not found"); continue
        per, modes, msgs = analyse(root)
        tot = collections.Counter()
        print(f"\n=== {run}")
        print(f"{'suite':<10} {'benign':>14} {'attacked':>9} {'UA%':>7} {'ASR%':>7}")
        for s in SUITES:
            r = per[s]; tot.update(r)
            b = 100 * r['benign_ok'] / r['benign_n'] if r['benign_n'] else float('nan')
            ua = 100 * r['att_util'] / r['att_n'] if r['att_n'] else float('nan')
            asr = 100 * r['att_sec'] / r['att_n'] if r['att_n'] else float('nan')
            print(f"{s:<10} {r['benign_ok']:>3}/{r['benign_n']:<3}={b:6.2f}% {r['att_n']:>9} {ua:6.2f}% {asr:6.2f}%")
        b = 100 * tot['benign_ok'] / tot['benign_n'] if tot['benign_n'] else float('nan')
        ua = 100 * tot['att_util'] / tot['att_n'] if tot['att_n'] else float('nan')
        asr = 100 * tot['att_sec'] / tot['att_n'] if tot['att_n'] else float('nan')
        extra = f"  (trade-off {ua + 100 - asr:.2f})" if tot['att_n'] else ""
        print(f"{'ALL':<10} {tot['benign_ok']:>3}/{tot['benign_n']:<3}={b:6.2f}% {tot['att_n']:>9} {ua:6.2f}% {asr:6.2f}%{extra}")
        if msgs:
            srt = sorted(msgs)
            print(f"  benign messages: min {srt[0]}, median {srt[len(srt)//2]}, max {srt[-1]}")
        for k, v in modes.most_common():
            print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
