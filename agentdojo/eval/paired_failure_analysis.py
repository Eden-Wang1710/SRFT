#!/usr/bin/env python3
"""Paired failure analysis of two AgentDojo runs with the same pipeline name (e.g. base vs a trained model).
usage: python eval/paired_failure_analysis.py <run_A> <run_B> [pipeline]      (run names relative to agentdojo/runs/)
For attacked (important_instructions) and benign (none) trajectories: the paired utility 2x2 per suite, then for each cell the
signatures of B's (and A's) trajectories: strict "announce an action without calling it" ending, asks the user for information,
"can't / unable" ending, declines citing the injection, no final answer, zero tool calls, >=10 assistant turns, loops, forced think
transitions, mean think words, extra argument keys vs the other model on the same function, and where the tool-call sequence diverges.
Written 2026-09-10 for base_think1024_noappend vs v3para_traj_3epoch_think1024 (03_analysis_utility_drop.md §L).
"""
import json, re, sys, glob, os, collections
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs")
A, B = sys.argv[1], sys.argv[2]; PIPE = sys.argv[3] if len(sys.argv) > 3 else "Qwen_Qwen3-8B-safe-agent"
VERBS = "send|create|schedule|update|transfer|book|reserve|add|delete|remove|search|reply|forward|invite|make|set|pay|read|post|cancel|change|call|use|proceed|inform"
ANNOUNCE = re.compile(rf"(I('ll| will)( now| then| proceed to| go ahead and)? ({VERBS})\b|[Ll]et me( now)? ({VERBS})\b|[Nn]ow[, ]+(I('ll| will) )?({VERBS})\b|[Pp]roceeding (to|with))")
ASK = re.compile(r"(please (provide|let me know|specify|confirm|share)|could you (provide|tell|confirm|clarify)|I (need|would need|require) (more|additional|the) (information|details)|ask (Emma|the user|you) (for|to)|(can|could) you (please )?(provide|confirm|clarify|tell))", re.I)
CANT = re.compile(r"(I('m| am) (sorry|unable)|I (can't|cannot|am unable to|couldn't|could not) (complete|access|find|perform|proceed|do|send|help)|not (possible|able)|no (results?|matching|such) )", re.I)
DECLINE = re.compile(r"(I (will not|won't|cannot|can't|refuse to|should not|must not) (follow|comply|execute|perform|act on)|ignor(e|ing) (the|this|that) (injected|hidden|suspicious|malicious|embedded)|(injected|hidden|suspicious|malicious) (instruction|message|request|text|content)|prompt injection)", re.I)
FORCED = "Considering the limited time"

def load(run, kind):
    out = {}
    for f in glob.glob(f"{ROOT}/{run}/{PIPE}/*/user_task_*/{kind}/*.json"):
        p = f.split("/"); out[(p[-4], p[-3], p[-1][:-5])] = json.load(open(f))
    return out

def blocks(m, t="text"):
    c = m.get("content")
    if isinstance(c, list): return "\n".join((b.get("content") or b.get("text") or "") for b in c if b.get("type") == t)
    return (c or "") if t == "text" else ""

def feats(d):
    asst = [m for m in d["messages"] if m["role"] == "assistant"]
    calls = [(t["function"], t.get("args") or {}) for m in asst for t in (m.get("tool_calls") or [])]
    sig = [(f, json.dumps(a, sort_keys=True)) for f, a in calls]
    loop = any(sig[i] == sig[i+1] == sig[i+2] for i in range(len(sig) - 2))
    last = asst[-1] if asst else None; lt = blocks(last) if last else ""; ends_call = bool(last and last.get("tool_calls"))
    tail = " ".join(re.split(r"(?<=[.!?])\s+", lt.strip())[-2:])
    thinks = [blocks(m, "thinking") for m in asst]
    argkeys = collections.defaultdict(set)
    for f, a in calls: argkeys[f] |= {k for k, v in a.items() if v not in (None, "", [])}
    return dict(fnames=[f for f, _ in calls], n_calls=len(calls), n_asst=len(asst), argkeys=argkeys, loop=loop,
                no_answer=(last is None) or ends_call or not lt.strip(),
                announce=(not ends_call) and bool(ANNOUNCE.search(tail)), ask=(not ends_call) and bool(ASK.search(lt)),
                cant=(not ends_call) and bool(CANT.search(lt)), decline=(not ends_call) and bool(DECLINE.search(lt)),
                forced=sum(FORCED in t for t in thinks), think_words=sum(len(t.split()) for t in thinks) / max(1, len(thinks)),
                error=bool(d.get("error")), last_text=lt)

def diverge(fa, fb):
    i = 0
    while i < min(len(fa), len(fb)) and fa[i] == fb[i]: i += 1
    return i

def report(kind):
    a, b = load(A, kind), load(B, kind); keys = sorted(set(a) & set(b))
    print(f"\n===== {kind}: {len(keys)} paired trajectories   A={A}   B={B}")
    per = collections.defaultdict(collections.Counter); cells = collections.defaultdict(list)
    for k in keys:
        cell = {(True, True): "both_pass", (True, False): "A_pass_B_fail", (False, True): "A_fail_B_pass", (False, False): "both_fail"}[(bool(a[k]["utility"]), bool(b[k]["utility"]))]
        per[k[0]][cell] += 1; per["ALL"][cell] += 1; cells[cell].append(k)
    for s in sorted(per): c = per[s]; print(f"{s:10s} both_pass {c['both_pass']:4d}  A_pass_B_fail {c['A_pass_B_fail']:4d}  A_fail_B_pass {c['A_fail_B_pass']:4d}  both_fail {c['both_fail']:4d}")
    for cell in ("A_pass_B_fail", "A_fail_B_pass", "both_fail", "both_pass"):
        ks = cells[cell]
        if not ks: continue
        for who, mine, other in (("B", b, a), ("A", a, b)):
            c = collections.Counter(); tw = fz = 0
            for k in ks:
                f, g = feats(mine[k]), feats(other[k])
                for n in ("announce", "ask", "cant", "decline", "no_answer", "loop", "error"): c[n] += f[n]
                c["zero_calls"] += f["n_calls"] == 0; c["asst>=10"] += f["n_asst"] >= 10
                c["extra_args"] += any(f["argkeys"][fn] - g["argkeys"][fn] for fn in f["argkeys"] if fn in g["argkeys"])
                c["fewer_calls" if f["n_calls"] < g["n_calls"] else "more_calls" if f["n_calls"] > g["n_calls"] else "same_calls"] += 1
                di = diverge(f["fnames"], g["fnames"]); c["same_seq" if f["fnames"] == g["fnames"] else "diverge_at_0" if di == 0 else "diverge_later"] += 1
                tw += f["think_words"]; fz += f["forced"]
            print(f"  {cell:14s} {who}: n={len(ks):3d} " + " ".join(f"{n}={c[n]}" for n in ("announce", "ask", "cant", "decline", "no_answer", "zero_calls", "loop", "asst>=10", "extra_args", "fewer_calls", "same_calls", "more_calls", "same_seq", "diverge_at_0", "diverge_later", "error")) + f" think_w={tw/len(ks):.0f} forced/traj={fz/len(ks):.2f}")
    if kind == "important_instructions":
        for suite in ("workspace", "slack", "banking", "travel"):
            ks = [k for k in cells["A_pass_B_fail"] if k[0] == suite]
            if not ks: continue
            ut = collections.Counter(k[1] for k in ks); fc = collections.Counter((feats(a[k])["fnames"][:1] or ["-"])[0] + " -> " + (feats(b[k])["fnames"][:1] or ["-"])[0] for k in ks)
            print(f"  {suite} A_pass_B_fail={len(ks)}: per user_task {dict(ut.most_common(6))}; first call A->B {fc.most_common(4)}")
    return cells, a, b

cells, a, b = report("important_instructions"); report("none")
print("\n===== examples (attacked, A passes, B fails, B ends by declining / asking / announcing)")
n = 0
for k in cells["A_pass_B_fail"]:
    f = feats(b[k])
    if f["decline"] or f["ask"] or f["announce"]:
        print(f"{k} B calls {f['fnames']} | A calls {feats(a[k])['fnames']}\n   B final: {f['last_text'][:260]!r}"); n += 1
    if n >= 8: break
