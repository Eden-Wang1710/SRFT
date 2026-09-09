#!/usr/bin/env python3
"""Rebuild toucan_32B_v2.json with the missing mid-turn assistant replies filled in from the
per-step source records (DSAI: SRFT/agentdojo/qwen3-32b-bedrock-samples/<suite>/<user_task>/<injection_task>/assistant_step_NNN.json).

Per-step file schema (verified on downloader_v1/user_task_2/injection_task_1/assistant_step_010.json):
  suite_name, source_trajectory, qwen_model_id, qwen_sampling_params,
  injections{injection_k: {trigger, task, combined, insert_position, target_message_index}},
  assistant_message_index (index in the trajectory `messages` list, system=0),
  context_messages[...], expert_assistant_message{role, content, tool_calls?},
  qwen_samples[{sequence_index, qwen_raw_output, qwen_parsed_message, follows_injection_task_action}],
  has_injection_inserted_by_step (bool), cot_self_reflection (str == <think> body in training data)

Mapping: training `conversations[i]` (human=0) <-> messages[i+1] <-> assistant_step_{i+1:03d}.json

Usage:
  python rebuild_from_steps.py --steps-root <dir with <suite>/user_task_N/injection_task_M/assistant_step_*.json> \
      --in ../LLaMA-Factory/data/toucan_32B_v2.json --out ../LLaMA-Factory/data/toucan_32B_v3_base.json [--strict]
Writes <out> and <out>.report.json. Never touches the input file.
"""
import argparse, json, os, re, collections, sys

THINK_RE = re.compile(r'^\s*<think>(.*?)</think>\s*', re.S)

def split_think(v):
    m = THINK_RE.match(v)
    if not m:
        return None, v.strip()
    return m.group(1).strip(), v[m.end():].strip()

def step_path(root, source_path, conv_idx):
    parts = source_path.split('/')
    suite = parts[3].replace('-cot-one-trajectory', '')
    return os.path.join(root, suite, parts[4], parts[5][:-5], f'assistant_step_{conv_idx+1:03d}.json')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps-root', required=True)
    ap.add_argument('--in', dest='inp', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--strict', action='store_true', help='abort on any mismatch between step file and training data')
    ap.add_argument('--drop-raw-trajectories', action='store_true', help='drop trajectories containing a tool call whose expert arguments were malformed JSON ({"_raw": ...})')
    a = ap.parse_args()

    D = json.load(open(a.inp))
    R = collections.Counter()
    problems = []
    for ti, d in enumerate(D):
        conv = d['conversations']
        sp = d['meta']['source_path']
        step_meta = {}
        for i, t in enumerate(conv):
            if t['from'] not in ('gpt', 'function_call'):
                continue
            R['assistant_steps'] += 1
            p = step_path(a.steps_root, sp, i)
            if not os.path.exists(p):
                R['step_file_missing'] += 1
                continue
            S = json.load(open(p))
            R['step_file_found'] += 1
            think, body = split_think(t['value'])
            exp = S['expert_assistant_message']
            # consistency checks
            if S.get('cot_self_reflection', '').strip() != (think or ''):
                R['think_mismatch'] += 1; problems.append((p, 'think_mismatch'))
            if t['from'] == 'function_call':
                try:
                    tc = exp['tool_calls'][0]['function']
                    raw = tc['arguments']
                    try:
                        exp_args = raw if isinstance(raw, dict) else json.loads(raw)
                    except json.JSONDecodeError:
                        exp_args = {'_raw': raw}   # source had malformed argument JSON; v0 data already stores it as {"_raw": ...}
                        R['toolcall_raw_args'] += 1
                    exp_call = {'name': tc['name'], 'arguments': exp_args}
                    if json.loads(body) != exp_call:
                        R['toolcall_mismatch'] += 1; problems.append((p, 'toolcall_mismatch'))
                except Exception as e:
                    R['toolcall_unparsable'] += 1; problems.append((p, f'toolcall_unparsable:{e}'))
                if (exp.get('content') or '').strip():
                    R['toolcall_with_expert_text'] += 1
            else:
                exp_text = (exp.get('content') or '').strip()
                mid_turn = i + 1 < len(conv) and conv[i+1]['from'] == 'human'
                if body:
                    if body != exp_text:
                        R['answer_mismatch'] += 1; problems.append((p, 'answer_mismatch'))
                else:
                    R['empty_answer_mid_turn' if mid_turn else 'empty_answer_final'] += 1
                    if exp_text:
                        t['value'] = (f'<think>\n{think}\n</think>\n\n' if think is not None else '') + exp_text
                        R['filled'] += 1
                    else:
                        R['still_empty'] += 1; problems.append((p, 'expert_content_empty_too'))
            step_meta[i] = {
                'has_injection_inserted_by_step': S.get('has_injection_inserted_by_step'),
                'injections': S.get('injections'),
                'n_qwen_samples': len(S.get('qwen_samples') or []),
                'n_samples_follow_injection': sum(1 for q in (S.get('qwen_samples') or []) if q.get('follows_injection_task_action')),
            }
        if step_meta:
            d['meta']['steps'] = {str(k): v for k, v in step_meta.items()}
            inj = next((v['injections'] for v in step_meta.values() if v['injections']), None)
            if inj:
                d['meta']['injections'] = inj
    if a.drop_raw_trajectories:
        keep = [d for d in D if not any(t['from'] == 'function_call' and '"_raw"' in t['value'] for t in d['conversations'])]
        dropped = [d['meta']['source_path'] for d in D if d not in keep]
        R['trajectories_dropped_raw'] = len(dropped); R['assistant_steps_dropped'] = sum(sum(t['from'] in ('gpt','function_call') for t in d['conversations']) for d in D if d not in keep)
        D = keep
    rep = {'counts': dict(R), 'problems': problems[:500], 'n_problems': len(problems), 'dropped': dropped if a.drop_raw_trajectories else []}
    print(json.dumps(rep['counts'], indent=1)); print('problems:', len(problems))
    if a.strict and problems:
        sys.exit('strict mode: mismatches found, not writing output')
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(D, open(a.out, 'w'), ensure_ascii=False, indent=1)
    json.dump(rep, open(a.out + '.report.json', 'w'), indent=1)
    print('wrote', a.out)

if __name__ == '__main__':
    main()
