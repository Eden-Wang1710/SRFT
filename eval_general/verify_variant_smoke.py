#!/usr/bin/env python
"""Pre-flight for the two protocol variants (docs/14 check 3) — run on the LIMIT=2 smoke samples BEFORE the
full re-inference, because each full run costs 3-11 GPU-hours and a wrong knob would waste all four.

  python eval_general/verify_variant_smoke.py eval_general/smoke/srllama

Checks, per variant, straight from what lm-eval logged (`arguments` = the exact prompt + gen kwargs sent):
  bbh_cot_fewshot_relaxed  gen kwargs `until` is ["</s>", "\\n\\nQ:"]; responses on the long tasks contain a
                           blank line (proof the old "\\n\\n" stop is gone) and reach "the answer is".
  mmlu_pro_cot             the rendered prompt ends with the assistant header followed by the prefill
                           "Let's think step by step." and NO end-of-turn token after it (continue_final_message);
                           the responses are multi-sentence, i.e. the model actually reasons after the prefill.
Exit status 1 on any failed check.
"""
import glob
import json
import os
import re
import sys

root = sys.argv[1] if len(sys.argv) > 1 else "eval_general/smoke/srllama"
ok = True


def fail(msg):
    global ok
    ok = False
    print("FAIL:", msg)


def samples(task_prefix):
    out = {}
    for f in sorted(glob.glob(os.path.join(root, "*", f"samples_{task_prefix}_*.jsonl"))):
        sub = re.search(rf"samples_({task_prefix}_.+?)_\d{{4}}-\d\d-\d\dT", os.path.basename(f)).group(1)
        out[sub] = [json.loads(l) for l in open(f)]
    return out


# ---------------- BBH relaxed ----------------
bbh = samples("bbh_cot_fewshot_relaxed")
print(f"bbh_cot_fewshot_relaxed: {len(bbh)} subtasks, {sum(len(v) for v in bbh.values())} items")
if not bbh:
    fail("no bbh_cot_fewshot_relaxed samples")
else:
    untils = {json.dumps(r["arguments"]["gen_args_0"]["arg_1"]["until"]) for recs in bbh.values() for r in recs}
    print("  until sets seen:", untils)
    if untils != {json.dumps(["</s>", "\n\nQ:"])}:
        fail("relaxed until list not applied")
    blank = sum("\n\n" in r["resps"][0][0] for recs in bbh.values() for r in recs)
    phrase = sum(bool(re.search("answer is", r["resps"][0][0], re.I)) for recs in bbh.values() for r in recs)
    n = sum(len(v) for v in bbh.values())
    print(f"  responses containing a blank line: {blank}/{n}   reaching 'answer is': {phrase}/{n}")
    if blank == 0:
        fail("no response contains a blank line — the \\n\\n stop still seems active")
    for sub in ("bbh_cot_fewshot_relaxed_tracking_shuffled_objects_seven_objects",):
        for r in bbh.get(sub, []):
            resp = r["resps"][0][0]
            steps = len(re.findall(r"^\(\d\)", resp, re.M))
            print(f"  [{sub.split('relaxed_')[1]}] len={len(resp)} steps={steps} answer_phrase={bool(re.search('answer is', resp, re.I))} "
                  f"extracted={r['filtered_resps'][0]!r} target={r['target']!r} score={r['exact_match']}")
            if steps < 2:
                fail("seven-object tracking response still stops after step (0)")

# ---------------- MMLU-Pro CoT prefill ----------------
mp = samples("mmlu_pro_cot")
print(f"\nmmlu_pro_cot: {len(mp)} subtasks, {sum(len(v) for v in mp.values())} items")
if not mp:
    fail("no mmlu_pro_cot samples")
else:
    tail_ok = 0
    n = 0
    for recs in mp.values():
        for r in recs:
            n += 1
            ctx = r["arguments"]["gen_args_0"]["arg_0"]
            tail = ctx[-200:]
            good = tail.endswith("<|start_header_id|>assistant<|end_header_id|>\n\nLet's think step by step.") \
                or re.search(r"assistant<\|end_header_id\|>\s*Let's think step by step\.$", ctx) is not None
            tail_ok += good
    print(f"  prompts ending with assistant header + prefill and no <|eot_id|>: {tail_ok}/{n}")
    if tail_ok != n:
        fail("prefill not rendered as an open assistant turn")
        r0 = next(iter(mp.values()))[0]
        print("  last 300 chars of one prompt:\n", repr(r0["arguments"]["gen_args_0"]["arg_0"][-300:]))
    lens = [len(r["resps"][0][0]) for recs in mp.values() for r in recs]
    direct = sum(l < 200 for l in lens)
    print(f"  response length median={sorted(lens)[len(lens)//2]} min={min(lens)} max={max(lens)}; <200 chars (direct answers): {direct}/{n}")
    if direct > n // 2:
        fail("SR still answers directly after the prefill on most items — the knob does not force reasoning")
    for sub in ("mmlu_pro_cot_math",):
        for r in mp.get(sub, []):
            print(f"  [math] target={r['target']} extracted={r['filtered_resps'][0]!r} score={r['exact_match']} resp[:160]={r['resps'][0][0][:160]!r}")

print("\nALL CHECKS PASSED" if ok else "\nSMOKE FAILED — do not submit the full runs")
sys.exit(0 if ok else 1)
