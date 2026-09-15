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
  published. That mistake is what triggered this whole re-run. Confirmed 2026-09-15: with the template base IFEval
  is inst-loose 84.77 / inst-strict 81.06 / prompt-loose 78.19 / prompt-strict 73.01, and the published 79.1
  falls between inst-strict and prompt-loose — the template setting reproduces the public number, the
  no-template setting does not.
- MMLU is the exception. It is scored by **loglikelihood over " A"/" B"/" C"/" D"**, and lm-eval itself
  warns at `lm_eval/evaluator.py:465`:
  `Chat template formatting change affects loglikelihood and multiple-choice tasks.`
  Measured: base MMLU 68.00 without the template vs **63.09 with** it (-4.91). The template pushes the
  model toward a conversational answer instead of the bare option token, so the no-template number is the
  correct one. This matches Open LLM Leaderboard practice.
- MMLU-Pro is generative (5-shot, `custom-extract` of "The answer is (X)"), so it **does** take the
  template despite the name. It was briefly cancelled on 2026-09-14 by mistake and resubmitted.

**How to tell, from a results JSON alone, whether the template applied** (corrected 2026-09-14; the earlier
note here claimed there was no way and was wrong). The **top-level** `chat_template_sha` is the marker:
`null` when the template was off, and the sha `e10ca381…4b65` when it was on, alongside a top-level
`chat_template` holding the full Jinja source and `fewshot_as_multiturn: true`. The key that is *absent*
is `config.chat_template_sha` — looking for it inside `config` is what produced the wrong note.
Verified against the two base MMLU files, which differ only in this. `eval_general/report.py` classifies
every run by this marker, so the selection is automatic.

| task | scoring | few-shot | chat template |
|---|---|---|---|
| mmlu | loglikelihood MC | 0 | **no** |
| mmlu_pro (100/subject) | generate_until, custom-extract | 5 | yes |
| ifeval | generate_until, format compliance | 0 | yes |
| bbh_cot_fewshot | generate_until, CoT | 3 | yes |

`env/jobs/lmeval.sbatch` implements this table: `CHAT` defaults to 0 for `mmlu` and 1 for everything else,
and prints `chat_template=<0|1>` in its header line. Pass `CHAT=1` or `CHAT=0` to override. The request
cache is split by protocol (`<task>` vs `<task>_nochat`) so the two settings can never share a db file.
Before this branch existed the runner applied the template unconditionally, which would have silently
re-run MMLU off-protocol.

## Acceptance checks — run these on EVERY finished benchmark (user instruction, 2026-09-14)

    python eval_general/report.py

The script reads every results JSON under `eval_general/`, picks the canonical one per (model, task) by the
protocol table above, and prints both checks. It refuses to compare a base/SR pair whose item counts differ.

**Check 1 — is our harness comparable to the published one?** Our *base* Llama-3.1-8B-Instruct against the
base row of the Meta-SecAlign / ReasAlign papers. Close enough means their defended row can be cited directly
instead of re-running Meta-SecAlign on these benchmarks. The user's stated example of acceptable is MMLU
68 vs 72, so the script's default tolerance is 5 points (`--tol` to change it).

| task | published base | our base |
|---|---|---|
| MMLU | 72.0 | 68.00, off by 4.0, comparable |
| MMLU-Pro | 46.5 | pending |
| IFEval (mean of 4) | 79.1 | **79.26, off by 0.16** |
| BBH | 71.9 | **71.23, off by 0.67** |

**Check 1 is settled on three of four: our harness reproduces the published base row.** MMLU -4.00,
IFEval +0.16, BBH -0.67, all inside the 5-point tolerance. Only MMLU-Pro is outstanding, and it is the one
where a gap is expected anyway because we subsample.

**IFEval metric convention (settled 2026-09-15).** IFEval reports four sub-metrics and the published 79.1 is
their **mean**, not any one of them. Our four are prompt-strict 73.01, inst-strict 81.06, prompt-loose 78.19,
inst-loose 84.77; their mean is 79.26, which lands 0.16 from the published figure, while inst-loose alone is
5.67 high and the two-strict mean is 2.07 low. So the headline for check 1 is the mean, and check 2 compares
all four sub-metrics separately, because a mean of four correlated metrics has no honest standard error.

Caveat on MMLU-Pro for check 1 only: we run 100 items per subject, 1,400 of the full 12,032, so our MMLU-Pro
is a subsample and a few points of disagreement with the published 46.5 are expected from sampling alone.
Check 2 is unaffected as long as base and SR draw the same 1,400 items, which `report.py` verifies by
comparing item counts. Both runs must therefore carry `LIMIT=100`.

**Check 2 — did SR-Agent-Llama lose general ability?** SR-Agent-Llama against *our* base under one harness
config. The claim is parity, not a win. The script calls it parity when the gap is inside 2 combined standard
errors, which is the bar MMLU already clears at -0.37 against a 2 se of 1.06.

Check 2 is the result we report. Check 1 only decides whether we may also quote the papers' defended row.

## Results

### Final (the numbers to report)
| task | base | SR-Agent-Llama | delta |
|---|---|---|---|
| MMLU (0-shot, no template) | **68.00** | **67.63** | -0.37 |
| MMLU-Pro (5-shot, 100/subj, template) | pending 3057328 | pending 3057329 | |
| IFEval inst-loose (0-shot, template) | **84.77** | **84.77** | +0.00 |
| IFEval inst-strict | 81.06 | 81.89 | +0.84 |
| IFEval prompt-loose | 78.19 | 77.82 | -0.37 |
| IFEval prompt-strict | 73.01 | 74.31 | +1.29 |
| BBH CoT (3-shot, template) | **71.23** | running 3057193 | |

### Archived no-template run (`eval_general_nochat/`, kept deliberately)
Off-protocol for the generative tasks; retained so both configurations exist.
| task | base | SR-Agent-Llama |
|---|---|---|
| mmlu acc | 68.00 | 67.63 |
| mmlu_pro exact_match | 41.86 | **43.29** |
| ifeval mean of 4 | 53.28 | (never finished) |
| ifeval inst_level_loose | 61.99 | |
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
| 3057188 | base ifeval, template | COMPLETED — inst-loose 84.77 / inst-strict 81.06 / prompt-loose 78.19 / prompt-strict 73.01 |
| 3057189 | base bbh, template | COMPLETED — 71.23 exact_match,get-answer (stderr 0.51), 6:37 |
| 3057192 | srllama ifeval, template | COMPLETED — 84.77 / 81.89 / 77.82 / 74.31 |
| 3057193 | srllama bbh, template | RUNNING |
| 3057328/3057329 | base/srllama mmlu_pro, template | RUNNING (started 05:35/05:39) |

Queue note: fairshare is exhausted (EffectvUsage 1.0, Priority 1), so jobs start only when a slot frees.
Standing node denylist `c2-gpu-[004-006,010]` (see `09_cluster_washu.md`).

### Settings audit of the in-flight batch (2026-09-14 23:5x)
All six jobs verified against their spooled batch script and their submission command. Editing
`env/jobs/lmeval.sbatch` does NOT reach a submitted job — slurm spools a copy — so the `CHAT` branch added
today applies only to future submissions. All six in-flight tasks are generative and want the template
anyway, so the old unconditional behaviour is correct for them and nothing needs resubmitting.

| jobid | task | limit | partition | denylist | time |
|---|---|---|---|---|---|
| 3057188 | base ifeval | full 541 | general-gpu | yes | 8 h |
| 3057189 | base bbh | full 6511 | general-gpu | yes | 20 h |
| 3057192 | srllama ifeval | full 541 | general-gpu | yes | 8 h |
| 3057193 | srllama bbh | full 6511 | general-gpu | yes | 20 h |
| 3057328 | base mmlu_pro | LIMIT=100 → 1400 | general-gpu + **preempt** | added after the fact | 8 h |
| 3057329 | srllama mmlu_pro | LIMIT=100 → 1400 | general-gpu + **preempt** | added after the fact | 8 h |

Two deviations found in the MMLU-Pro pair, which was resubmitted with a plain `env/sb gpu` instead of the
`-p general-gpu --exclude=$(cat slurm_logs/.rlh_denylist)` the other four carry:
- **No node denylist.** Fixed in place with `scontrol update JobId=<id> ExcNodeList=…`, which a pending job
  accepts without losing its queue position. Worth knowing as the cheap repair for this mistake.
- **`general-preempt-gpu` included**, so these two can land on a preemptible A100 and be requeued. Left as
  is: `--use_cache --cache_requests true` makes a requeued run resume, and the extra partition is a
  scheduling advantage while fairshare is exhausted. It does mean these two may run slower than the H100 rows.

### The request cache is protocol-safe — proved by the MMLU-Pro restart (2026-09-15)
When `base mmlu_pro` restarted it showed **1262** requests to run while `srllama mmlu_pro` showed **1400**.
That is not an item-count mismatch and needs no fix. The progress bar counts cache *misses*: job 3057187
had run 138/1400 of the same templated configuration before being cancelled, and 1400 − 1262 = 138 exactly.
Both runs therefore cover all 1,400 items, and `report.py` reads item counts from `n-samples`, not the bar.

The stronger point: that same cache db also holds a **complete** 1,400-item no-template run (job 3055603).
If the cache key ignored the rendered prompt, the restart would have hit all 1,400 and run nothing. It hit
exactly the 138 templated entries. **This is direct proof that a template flip cannot pick up stale
responses**, which had previously only been inferred from IFEval getting zero hits.

Schedule risk to watch: `srllama bbh` has a 20 h limit and base BBH is measured at 7.08 s/item, i.e. ≈12.6 h
for 6511 items; the LoRA is slower, so the srllama run could approach the limit. `scontrol update TimeLimit`
can only *shorten* a job (`09_cluster_washu.md`), so the only remedy is cancel and resubmit, and the request
cache means a resubmitted run resumes rather than restarts.

## Reading of the IFEval result
Instruction-following is **unchanged** by SRFT: four metrics, largest deviation 1.29 points, and SR is
ahead on two of them (inst-strict +0.84, prompt-strict +1.29). This is stronger than the parity the
section set out to show, and is the cleanest general-capability evidence in the set, because IFEval is
scored by programmatic format compliance rather than by an accuracy proxy.

## Published-number sanity checks
Both completed base rows land where the public numbers for `Llama-3.1-8B-Instruct` sit, which is the
independent check that the template setting is right:
IFEval 81.06 inst-strict / 78.19 prompt-loose against a published 79.1; BBH 71.23 against a commonly
reported 70-72. The no-template run missed IFEval by 17 points.

## Account contention
`reasalign_repro_A/B` (3057398/3057399, started 2026-09-15 ~00:00, c2-gpu-001) run under the same
`li.hao` account but were submitted by someone else. They compete for the account's GPU allocation, so
the pending lm-eval jobs start later than the queue position alone would suggest. Do not cancel them.
