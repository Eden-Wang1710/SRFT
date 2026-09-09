"""Summarize a paraphrase run: filter pass rates, metric distributions, and a side-by-side markdown of N samples for review."""
import json, sys, collections, statistics as st, random
inp = sys.argv[1]; md = sys.argv[2] if len(sys.argv) > 2 else None; N = int(sys.argv[3]) if len(sys.argv) > 3 else 20
recs = [json.loads(l) for l in open(inp)]; random.seed(1)
fails = collections.Counter(); per_sit = collections.defaultdict(lambda: [0, 0]); lr = []; ov = []
for r in recs:
    key = "final" if r["kind"] == "final" else r["situation"]; per_sit[key][0] += 1; per_sit[key][1] += bool(r["accepted"])
    for a in r["attempts"]:
        for f in a["fails"]: fails[f.split("_")[0] if f.startswith("length") else f] += 1
    if r["accepted"]: lr.append(r["accepted"]["metrics"]["len_ratio"]); ov.append(r["accepted"]["metrics"]["overlap8"])
print(f"steps {len(recs)} | accepted {sum(1 for r in recs if r['accepted'])} ({100*sum(1 for r in recs if r['accepted'])/len(recs):.0f}%)")
print("by situation:", {k: f"{v[1]}/{v[0]}" for k, v in per_sit.items()})
print("attempt-level failure counts:", dict(fails.most_common()))
if lr: print(f"accepted: len ratio mean {st.mean(lr):.2f} (min {min(lr)}, max {max(lr)}) | 8-gram overlap mean {st.mean(ov):.2f}")
if md:
    acc = [r for r in recs if r["accepted"]]; rej = [r for r in recs if not r["accepted"]]
    with open(md, "w") as f:
        f.write(f"# v3 paraphrase pilot — {len(acc)} accepted / {len(recs)}; {N} random accepted + 5 rejected below\n\n")
        for r in random.sample(acc, min(N, len(acc))):
            f.write(f"## ACCEPTED traj {r['traj_idx']} step {r['step_idx']} — {r['platform']} / {r['kind']} / {r['situation']}  (len ratio {r['accepted']['metrics']['len_ratio']}, overlap {r['accepted']['metrics']['overlap8']})\n\n")
            f.write("**Claude:**\n\n> " + r["claude_think"].replace("\n", "\n> ") + "\n\n**Qwen paraphrase:**\n\n> " + r["accepted"]["text"].replace("\n", "\n> ") + "\n\n---\n\n")
        for r in random.sample(rej, min(5, len(rej))):
            a = r["attempts"][0]
            f.write(f"## REJECTED traj {r['traj_idx']} step {r['step_idx']} — {r['situation']} — fails {a['fails']} metrics {a['metrics']}\n\n**Claude:**\n\n> " + r["claude_think"].replace("\n", "\n> ") + "\n\n**Qwen attempt 0:**\n\n> " + a["text"].replace("\n", "\n> ") + "\n\n---\n\n")
    print("wrote", md)
