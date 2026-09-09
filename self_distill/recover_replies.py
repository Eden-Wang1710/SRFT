"""Rebuild the paper dataset with the missing mid-conversation assistant replies restored from the per-step source files.
Minimal fix: ONLY the 3,185 `gpt` steps whose text after </think> is empty get `expert_assistant_message.content`; nothing else changes.
Also writes per-step injection ground truth (has_injection_inserted_by_step) for later use.
python self_distill/recover_replies.py
"""
import json, glob, re, collections
SRC = 'agentdojo/qwen3-32b-bedrock-samples'
o = json.load(open('LLaMA-Factory/data/toucan_32B_v2.json'))
src = {}
for f in glob.glob(f'{SRC}/*/user_task_*/injection_task_*/assistant_step_*.json'):
    d = json.load(open(f)); suite = f.split('/')[-4]; traj = d['source_trajectory'][:-5]
    src.setdefault((suite, traj), []).append((d['assistant_message_index'], d))
out = []; labels = []; filled = 0; changed_other = 0; check = collections.Counter()
for ti, s in enumerate(o):
    m = re.search(r'samples/([^/]+)-cot-one-trajectory/(user_task_\d+/injection_task_\d+)\.json', s['meta']['source_path']); key = (m.group(1), m.group(2))
    steps = [d for _, d in sorted(src[key])]
    conv = []; k = 0
    for msg in s['conversations']:
        if msg['from'] in ('function_call', 'gpt'):
            d = steps[k]; em = d['expert_assistant_message']
            labels.append({"traj_idx": ti, "step_idx": k, "kind": msg['from'], "injection_in_context": d['has_injection_inserted_by_step'], "injections": d['injections'], "expert_has_text": bool((em.get('content') or '').strip()), "n_candidates": len(d.get('qwen_samples') or [])})
            think, rest = msg['value'].split('</think>', 1) if '</think>' in msg['value'] else (msg['value'], '')
            if msg['from'] == 'gpt' and not rest.strip():
                text = (em.get('content') or '').strip(); assert text, f"no expert text for {key} step {k}"
                conv.append({"from": "gpt", "value": think + "</think>\n\n" + text}); filled += 1; check['filled'] += 1
            else:
                # sanity: the expert action must agree with what the training data has
                if msg['from'] == 'function_call':
                    tc = (em.get('tool_calls') or [{}])[0].get('function', {}); name = tc.get('name') if isinstance(tc, dict) else None
                    if name and name not in rest: check['tool_name_mismatch'] += 1
                conv.append(dict(msg))
            k += 1
        else:
            conv.append(dict(msg))
    meta = dict(s['meta']); meta['recovered_replies'] = sum(1 for a, b in zip(conv, s['conversations']) if a['value'] != b['value'])
    out.append({"conversations": conv, "system": s['system'], "meta": meta})
json.dump(out, open('LLaMA-Factory/data/toucan_32B_v3_base.json', 'w'), ensure_ascii=False)
json.dump(labels, open('self_distill/recovery/step_labels.json', 'w'), ensure_ascii=False)
print(f"trajectories {len(out)} | steps {len(labels)} | replies filled {filled} | checks {dict(check)}")
# verify: only gpt values changed, everything else byte-identical
diff = collections.Counter()
for a, b in zip(out, o):
    for x, y in zip(a['conversations'], b['conversations']):
        if x['value'] != y['value']: diff[x['from']] += 1
print("changed messages by role:", dict(diff))
lens = [len(re.split(r'</think>\s*', x['value'], 1)[-1].split()) for a in out for x in a['conversations'] if x['from']=='gpt']
import statistics as st; print(f"gpt reply words now: mean {st.mean(lens):.0f}, empty {sum(1 for l in lens if l==0)}")
info = json.load(open('LLaMA-Factory/data/dataset_info.json'))
if 'toucan_32B_v3_base' not in info:
    e = json.loads(json.dumps(info['toucan_32B_v2'])); e['file_name'] = 'toucan_32B_v3_base.json'; info['toucan_32B_v3_base'] = e
    json.dump(info, open('LLaMA-Factory/data/dataset_info.json', 'w'), indent=2, ensure_ascii=False); print("registered toucan_32B_v3_base")
