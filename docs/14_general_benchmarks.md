# General-capability benchmarks (lm-evaluation-harness 0.4.9)

Purpose: show that SR-Agent-Llama (no reflection append) does not damage the base model's general
ability. Target claim is parity with `meta-llama/Llama-3.1-8B-Instruct`, not a win.

Runner: `env/jobs/lmeval.sbatch`. Env: conda `lmeval` (lm-eval 0.4.9, peft 0.17.1, transformers 4.57.1,
torch 2.7.0+cu128). LoRA = `LLaMA-Factory/saves/llama31-8b/lora/v3base_local_sft_8k_r64_GA4_qkvo_3epoch_5e-6`.
The SR LoRA is evaluated **without** the reflection system-prompt append, matching the main
AgentDojo / RL-Hammer rows. No SRFT prompt is added anywhere; lm-eval sends each task's own prompt.

## Chat-template decision (settled 2026-09-14)

`--apply_chat_template --fewshot_as_multiturn` is used for the **generative** tasks and NOT for MMLU.

- Llama-3.1-8B-Instruct is an instruct model; evaluating it in raw-completion mode is off-protocol and
  measurably wrong on generative tasks: base IFEval scored 61.99 inst-loose without the template vs 79.1
  published. That mistake is what triggered this whole re-run.
- MMLU is the exception. It is scored by **loglikelihood over " A"/" B"/" C"/" D"**, and lm-eval itself
  warns at `lm_eval/evaluator.py:465`:
  `Chat template formatting change affects loglikelihood and multiple-choice tasks.`
  Measured: base MMLU 68.00 without the template vs **63.09 with** it (-4.91). The template pushes the
  model toward a conversational answer instead of the bare option token, so the no-template number is the
  correct one. This matches Open LLM Leaderboard practice.
- MMLU-Pro is generative (5-shot, `custom-extract` of "The answer is (X)"), so it **does** take the
  template despite the name. It was briefly cancelled on 2026-09-14 by mistake and resubmitted.

Note `chat_template_sha: None` appears in the results config even when the template DID apply — that is a
recording gap in lm-eval 0.4.9, not evidence the flag was ignored. Check for the warning line in the
slurm log instead.

| task | scoring | few-shot | chat template |
|---|---|---|---|
| mmlu | loglikelihood MC | 0 | **no** |
| mmlu_pro (100/subject) | generate_until, custom-extract | 5 | yes |
| ifeval | generate_until, format compliance | 0 | yes |
| bbh_cot_fewshot | generate_until, CoT | 3 | yes |

## Results

### Final (the numbers to report)
| task | base | SR-Agent-Llama | delta |
|---|---|---|---|
| MMLU (0-shot, no template) | **68.00** | **67.63** | -0.37 |
| MMLU-Pro (5-shot, 100/subj, template) | pending 3057328 | pending 3057329 | |
| IFEval inst-loose (0-shot, template) | running 3057188 | pending 3057192 | |
| BBH CoT (3-shot, template) | running 3057189 | pending 3057193 | |

### Archived no-template run (`eval_general_nochat/`, kept deliberately)
Off-protocol for the generative tasks; retained so both configurations exist.
| task | base | SR-Agent-Llama |
|---|---|---|
| mmlu acc | 68.00 | 67.63 |
| mmlu_pro exact_match | 41.86 | **43.29** |
| ifeval inst_level_loose | 61.99 | (never finished) |
| ifeval inst_level_strict | 58.15 | |
| ifeval prompt_level_loose | 48.98 | |
| ifeval prompt_level_strict | 43.99 | |

MMLU is the same in both blocks by construction — it is the no-template run in both.

## Scope of the missing-template bug
It was **confined to lm-eval**. AgentDojo and RL-Hammer build the Llama chat format themselves in
`llama_local_prompt.py` (`<|begin_of_text|>`, `<|start_header_id|>{role}<|end_header_id|>`, `<|eot_id|>`),
verified in the RL-Hammer sanity log. All AgentDojo rows in `13_iclr_results.md` and all six RL-Hammer
curves are unaffected.

## Job ledger (2026-09-14)
| jobid | task | note |
|---|---|---|
| 3054329 | srllama mmlu, no template | COMPLETED 67.63 — this is the MMLU number used |
| 3055603/3055604 | base/srllama mmlu_pro, no template | COMPLETED 41.86 / 43.29, archived |
| 3055286 | base ifeval, no template | COMPLETED 61.99 inst-loose, archived |
| 3057186 | base mmlu, template | COMPLETED 63.09 — evidence for the MMLU decision, not reported |
| 3057187/3057190/3057191 | mmlu + mmlu_pro, template | CANCELLED; mmlu_pro resubmitted as 3057328/3057329 |
| 3057188/3057189 | base ifeval / bbh, template | RUNNING |
| 3057192/3057193 | srllama ifeval / bbh, template | PENDING |
| 3057328/3057329 | base/srllama mmlu_pro, template | PENDING |

Queue note: fairshare is exhausted (EffectvUsage 1.0, Priority 1), so jobs start only when a slot frees.
Standing node denylist `c2-gpu-[004-006,010]` (see `09_cluster_washu.md`).
