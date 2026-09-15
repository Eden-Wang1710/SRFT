#!/usr/bin/env python
"""Generate the two protocol-variant task groups next to this file, from the stock lm-eval 0.4.9 task dirs.

Why (docs/14 §"Where the two check-2 failures come from", 2026-09-15): both drops of SR-Agent-Llama are
response-STYLE effects, not lost knowledge, and each needs one protocol knob to test that claim symmetrically:

  bbh_cot_fewshot_relaxed   stock BBH stops generation at "\n\n" (and at any "Q"). SR-Agent writes a blank line
                            between reasoning steps, so 207/250 tracking_shuffled_objects_seven answers were cut
                            after step (0) and never reached "the answer is". Variant: until = ["</s>", "\n\nQ:"]
                            (the completion-mode few-shot separator only), everything else identical.
  mmlu_pro_cot              stock MMLU-Pro lets the assistant turn start empty; SR-Agent then answers directly
                            without a chain of thought in 57 % of items (base: 2 %), which is where the math
                            drop comes from. Variant: the assistant turn is PREFILLED with "Let's think step by
                            step." (lm-eval `gen_prefix`, rendered with continue_final_message), the standard
                            zero-shot-CoT trigger. Everything else identical.

Both variants apply to base and SR alike. Run with `--include_path eval_general/tasks` (the runner does this).
Re-run this script after an lm-eval upgrade; the output dirs are committed so a run is reproducible without it.
"""
import os
import re
import shutil

import lm_eval

HERE = os.path.dirname(os.path.abspath(__file__))
STOCK = os.path.join(os.path.dirname(lm_eval.__file__), "tasks")


def bbh_relaxed():
    src = os.path.join(STOCK, "bbh", "cot_fewshot")
    dst = os.path.join(HERE, "bbh_cot_fewshot_relaxed")
    shutil.rmtree(dst, ignore_errors=True)
    os.makedirs(dst)
    for f in sorted(os.listdir(src)):
        if not f.endswith(("yaml", ".yaml")):
            continue
        s = open(os.path.join(src, f)).read()
        s = s.replace("bbh_cot_fewshot", "bbh_cot_fewshot_relaxed")
        if f == "_cot_fewshot_template_yaml":
            old = '  until:\n    - "</s>"\n    - "Q"\n    - "\\n\\n"\n'
            assert old in s, "stock BBH template changed; update make_variants.py"
            s = s.replace(old, '  until:\n    - "</s>"\n    - "\\n\\nQ:"\n')
            s = s.replace("metadata:\n  version: 3.0", "metadata:\n  version: 3.0-relaxed-until")
        open(os.path.join(dst, f), "w").write(s)
    print("wrote", dst, len(os.listdir(dst)), "files")


def mmlu_pro_cot():
    src = os.path.join(STOCK, "mmlu_pro")
    dst = os.path.join(HERE, "mmlu_pro_cot")
    shutil.rmtree(dst, ignore_errors=True)
    os.makedirs(dst)
    for f in sorted(os.listdir(src)):
        if f in ("__pycache__", "README.md"):
            continue
        s = open(os.path.join(src, f)).read()
        if f.endswith((".yaml", "yaml")):
            s = re.sub(r"\bmmlu_pro\b", "mmlu_pro_cot", s)
            s = re.sub(r"\bmmlu_pro_(\w+)\b", lambda m: m.group(0) if m.group(1).startswith("cot") else "mmlu_pro_cot_" + m.group(1), s)
        if f == "_default_template_yaml":
            assert "gen_prefix" not in s
            s = s.replace("output_type: generate_until\n",
                          'output_type: generate_until\ngen_prefix: "Let\'s think step by step."\n')
            s = s.replace("metadata:\n  version: 2.1", "metadata:\n  version: 2.1-cot-prefill")
        open(os.path.join(dst, f), "w").write(s)
    print("wrote", dst, len(os.listdir(dst)), "files")


if __name__ == "__main__":
    bbh_relaxed()
    mmlu_pro_cot()
