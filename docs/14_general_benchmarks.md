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

| task | published base | published Meta-SecAlign-8B | our base |
|---|---|---|---|
| MMLU | 72.0 | 71.7 | 68.00, off by 4.0, comparable |
| MMLU-Pro | 46.5 | 46.7 | **45.64 lm-eval strict / 47.50 official extraction** |
| IFEval (mean of 4) | 79.1 | 74.5 | **79.26, off by 0.16** |
| BBH | 71.9 | 70.9 | **71.23, off by 0.67** |

(Published rows = Table 3 of arXiv 2507.02735, Llama-3.1-8B-Instruct block, fetched 2026-09-15; the SecAlign-original
column there is 71.7 / 45.9 / 73.5 / 71.2.)

**Check 1 PASSES on all four: our harness reproduces the published base row.** MMLU -4.00, MMLU-Pro -0.86,
IFEval +0.16, BBH -0.67, every one inside the 5-point tolerance. MMLU-Pro landing this close is better than
expected given that we score 1,400 of 12,032 items. The Meta-SecAlign / ReasAlign defended row may therefore
be cited against ours rather than re-run.

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
| MMLU-Pro (5-shot, 100/subj, template) | **45.64** | **40.86** | **-4.79, outside 2 se = 3.67 — the one FAILURE** |
| IFEval inst-loose (0-shot, template) | **84.77** | **84.77** | +0.00 |
| IFEval inst-strict | 81.06 | 81.89 | +0.84 |
| IFEval prompt-loose | 78.19 | 77.82 | -0.37 |
| IFEval prompt-strict | 73.01 | 74.31 | +1.29 |
| BBH CoT (3-shot, template) | **71.23** | **64.26** | **-6.97, outside 2 se = 1.45 — the SECOND FAILURE** |

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
| 3057193 | srllama bbh, template | COMPLETED 10:52 — 64.26 exact_match,get-answer (stderr 0.52) |
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

## The MMLU-Pro drop — the one check-2 failure (2026-09-15)

SR-Agent-Llama scores **40.86 against the base's 45.64, a 4.79-point drop against a 2 se bar of 3.67.**
(BBH, finished later the same day, fails check 2 as well — see the next section. The two failures are the
two tasks scored by extracting an answer string from free-form CoT, which is the pattern that matters.)

**The drop is not spread across subjects — it is almost entirely math.**

| subject | base | SR | delta | | no-template base | no-template SR | delta |
|---|---|---|---|---|---|---|---|
| math | 55.0 | 26.0 | **-29.0** | | 45.0 | 41.0 | -4.0 |
| chemistry | 35.0 | 23.0 | -12.0 | | 30.0 | 38.0 | +8.0 |
| philosophy | 49.0 | 38.0 | -11.0 | | 41.0 | 38.0 | -3.0 |
| computer_science | 49.0 | 39.0 | -10.0 | | 45.0 | 49.0 | +4.0 |
| economics | 54.0 | 48.0 | -6.0 | | 37.0 | 45.0 | +8.0 |
| the other 9 | | | -4 to +4 | | | | |

SR is lower in 8 of 14 subjects, but math alone contributes about 2.1 of the 4.79-point mean drop.

**Why this looks like a generation-format failure rather than lost knowledge.** MMLU-Pro is scored by
`custom-extract`, a regex for "The answer is (X)" at the end of a chain of thought. A response that reasons
correctly but never emits that string scores zero. The decisive evidence is the no-template column: the
*same adapter weights* score **41.0** on math without the chat template and **26.0** with it, while the base
goes the other way, 45.0 → 55.0. The template makes the base much better at math and the SR model much
worse. Weights do not lose arithmetic because a template was applied; a generation *style* can stop matching
an extraction regex. The plausible mechanism is that the SR-Agent LoRA, trained on agentic trajectories in
exactly this chat format, falls into a reflective or tool-calling register under the template and either
runs past the generation cap or ends without the required answer string.

**Diagnostic submitted, 3058806 / 3058807**: `mmlu_pro_math` only, 5-shot, 100 items, both models, with the
new `LOG_SAMPLES=1` knob so every prompt, response and score is written out. That settles it directly —
count how many SR responses fail the regex versus how many extract an answer that is simply wrong. Results
go to `eval_general/<tag>/mmlu_pro_math/`, a separate directory that cannot disturb the canonical rows.

Until that returns, report MMLU-Pro as a drop and say the cause is under investigation. Do not describe it
as parity.

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

## The BBH drop — the second check-2 failure (2026-09-15)

`srllama bbh` (3057193, 10 h 52, all 6,511 items) finished at 11:46: **64.26 vs the base's 71.23, a
6.97-point drop against a 2 se bar of 1.45.** So two of the four benchmarks fail check 2 — and the two that
fail are exactly the two scored by **extracting an answer string out of free-form chain-of-thought**
(BBH `get-answer`, MMLU-Pro `custom-extract`), while the two that pass are the loglikelihood one (MMLU) and
the one scored by programmatic format compliance with no answer extraction at all (IFEval).

**The drop is concentrated in a handful of subtasks, as on MMLU-Pro.** SR minus base, 27 subtasks:

| subtask | base | SR | delta |
|---|---|---|---|
| tracking_shuffled_objects_seven_objects | 83.2 | 12.4 | **-70.8** |
| penguins_in_a_table | 82.2 | 51.4 | **-30.8** |
| reasoning_about_colored_objects | 70.0 | 48.4 | -21.6 |
| geometric_shapes | 54.8 | 36.8 | -18.0 |
| snarks | 72.5 | 55.1 | -17.4 |
| disambiguation_qa / date_understanding / tracking_shuffled_objects_five | | | -10.8 / -9.6 / -9.6 |
| 14 more | | | -8.8 to +0.0 |
| hyperbaton, logical_deduction_three, sports_understanding, multistep_arithmetic_two, temporal_sequences, navigate | | | +0.8 to +7.2 |

SR is lower in 20 of 27 subtasks, but the top five carry **-5.9 of the -6.97 mean**. A uniform loss of
reasoning ability would not look like this; `tracking_shuffled_objects_seven_objects` collapsing from 83.2 to
12.4 while its three-object version only moves 92.8 → 90.4 is a length/format signature, not a knowledge one —
the seven-object variant is the longest chain of thought in the set.

**The likely mechanism, and it is sharper than on MMLU-Pro: the stop strings.** The task's generation kwargs
(read from the results JSON) are

    max_gen_toks: 1024, until: ["</s>", "Q", "\n\n"], do_sample: false, temperature: 0.0

`"\n\n"` means **the response is cut at the first blank line**. A few-shot CoT answer that runs as one
paragraph and ends in "So the answer is (X)" survives; a model that opens with a reflective preamble, or
breaks its reasoning into paragraphs or a bulleted list, is truncated before it ever emits the answer string
and scores zero however correct the reasoning was. The SR-Agent LoRA is trained on agentic trajectories in
exactly this chat format, so a multi-paragraph / reflective register is precisely what it would have learned.
`"Q"` is also a bare-substring stop, so any capital Q in the output ends it.

This is the same class of explanation as the MMLU-Pro math finding and now has two independent instances.
**Diagnostic to run: `LOG_SAMPLES=1` on the worst subtasks** (`tracking_shuffled_objects_seven_objects`,
`penguins_in_a_table`) for both models, then count, over SR's failures, how many responses contain no answer
string at all (truncated / never emitted) versus how many extract an answer that is simply wrong. If the
first count dominates, the honest report is that the *extraction protocol* penalises the SR model's output
style, and the fix for the paper is to report a regex-tolerant or stop-string-relaxed variant alongside the
default one — not to claim parity on the default.

Until that diagnostic returns, report BBH as a drop under investigation, exactly as MMLU-Pro.

## Where the two check-2 failures come from — settled from the per-item samples (2026-09-15, afternoon)

**How the samples were obtained without a GPU.** Every finished run's responses sit in the request cache
(`eval_general/.cache/<tag>/<task>_rank0.db`, keyed on the rendered prompt + generation kwargs). Replaying the
identical configuration with `LOG_SAMPLES=1` therefore hits the cache 100 % and writes every prompt / response /
score without a single forward pass; the model still has to be instantiated, so the runner got a `DEVICE=cpu`
knob and the replay runs on a CPU node (`-p general-short --gres=gpu:0 --mem=80G`, 2–3 min, jobs 3060480–3060483,
`Cached requests: 6511, Requests remaining: 0`). Samples live in `eval_general/samples/<tag>/`; the replayed
aggregate reproduces the canonical numbers exactly (45.64 / 40.86, 71.23 / 64.26). The two GPU diagnostic jobs
3058806/3058807 were cancelled as superseded — they would have produced the same cached responses.

**The analyzer** `eval_general/analyze_samples.py --task bbh|mmlu_pro` classifies every item under lm-eval's own
STRICT filter into `correct` / `no_phrase` (never says "answer is" — truncated or ended without the sentence) /
`format_only` (names the target after "answer is" but the strict regex or exact-match missed it) / `wrong` (a
robust extractor finds a different answer — the only genuine failure), and re-scores both models with the ROBUST
extractor (last "answer is …", case/markdown/parenthesis-tolerant), applied identically to both models.

### BBH: the gap is truncation, not knowledge

| | base | SR-Agent |
|---|---|---|
| correct (strict) | 4638 (71.2 %) | 4184 (64.3 %) |
| no_phrase | 411 (6.3 %) | **827 (12.7 %)** |
| format_only | 71 | 117 |
| wrong | 1391 (21.4 %) | **1383 (21.2 %)** |

Decomposition of the −6.97 gap: **no_phrase −6.39, format_only −0.71, wrong +0.12.** SR-Agent gets *exactly as
many answers wrong as the base*; it just fails to emit the answer sentence twice as often. Where it fails is
diagnostic: on `tracking_shuffled_objects_seven_objects` 207 of 250 SR responses (base: 2) are ~140 characters
long, all end with "." and all look like

    Let's think step by step.
    (0) At the start: Alice: orange, Bob: yellow, Claire: brown, Dave: white, Eve: black, Fred: red, Gertrude: purple.

— cut after step (0). The stock BBH generation kwargs are `until: ["</s>", "Q", "\n\n"]`, so the harness stops
the model at the first blank line. SR-Agent writes a blank line between reasoning steps; the base does not. The
text after the blank line **was never generated**, which is why re-scoring cannot recover it: under the ROBUST
extractor SR moves 64.26 → 65.90 and the base 71.23 → 72.25, gap −6.34, essentially unchanged. (The three-object
version of the same task barely moves, 92.8 → 90.4, because a three-step answer fits before the first blank line.)

### MMLU-Pro: the gap is SR-Agent skipping the chain of thought — a style change, not an extraction failure

Here the extraction hypothesis was **wrong** and the samples say so: SR has *fewer* `no_phrase` items than the base
(135 vs 180) and *more* `wrong` (681 vs 573); the decomposition of −4.79 is no_phrase +3.21, format_only −0.29,
**wrong −7.71**, and the ROBUST extractor leaves the gap at −4.29. What differs is the *response style*:

| | base | SR-Agent |
|---|---|---|
| items answered directly, no chain of thought (response < 200 chars or starting "The answer is") | **31 / 1400 (2.2 %)** | **800 / 1400 (57.1 %)** |
| median response length, math | 1046 chars | **18 chars** ("The answer is (B).") |

Every subject's prompt literally says "Think step by step and then finish your answer with 'the answer is (X)'";
the base complies, SR-Agent answers in one line on 57 % of items. On subjects where working matters that costs
accuracy: on the 68 math items SR answered directly, the base (reasoning on the same items) scores 66 %, SR 25 %;
on biology, where recall suffices, the direct answers are fine (68 % vs 63 %). So the MMLU-Pro drop is real under
the default protocol, but it measures a **behavioural shift induced by SRFT** (the agentic SFT trajectories teach
terse assistant turns) rather than lost knowledge — and that is a claim to be tested, not asserted.

### The test: one symmetric protocol knob per benchmark (check 3 in `report.py`)

`eval_general/tasks/make_variants.py` generates two task groups from the stock lm-eval dirs (committed under
`eval_general/tasks/`, loaded by the runner via `--include_path`), each changing exactly one thing:

| variant | knob | what it tests |
|---|---|---|
| `bbh_cot_fewshot_relaxed` | `until: ["</s>", "\n\nQ:"]` (no bare `"\n\n"` / `"Q"` stop) | does SR reach the answer sentence when it is allowed to finish? |
| `mmlu_pro_cot` | assistant turn prefilled with "Let's think step by step." (lm-eval `gen_prefix`, rendered with `continue_final_message`) — the standard zero-shot-CoT trigger | does SR match the base once it actually reasons? |

Both models are re-run under each knob; the default rows remain the reported numbers and the variant rows are
shown next to them. **Re-inference is unavoidable for both**: the BBH text was never generated, and on MMLU-Pro
the answers that exist are one-liners. Predictions to check against: under the relaxed stop strings SR's BBH
should recover most of the 6.4-point no_phrase deficit while the base moves little; under the CoT prefill SR's
MMLU-Pro should approach the base's 45.6 — if it does not, the drop is a capability loss and gets reported as one.

### Pre-flight before the full re-inference (2026-09-15 12:14–12:26, smoke jobs 3060497 / 3060498)

User's rule: a full re-run costs 3–11 GPU-hours per job, so the knobs are verified on `LIMIT=2` smoke runs
(srllama, `LOG_SAMPLES=1`, `-p general-short`, output under `eval_general/smoke/`) with
`eval_general/verify_variant_smoke.py`, which reads the exact prompt and gen kwargs lm-eval logged. Both pass:

| variant | what the logged samples show |
|---|---|
| `bbh_cot_fewshot_relaxed` | `until` = `["</s>", "\n\nQ:"]` on all 54 requests; 52/54 reach "the answer is"; 6/54 responses contain a blank line (the old stop is gone); the two seven-object tracking items now run **8 steps** each (default run: cut after step 0) and both are correct |
| `mmlu_pro_cot` | all 28 prompts end with `<\|start_header_id\|>assistant<\|end_header_id\|>\n\nLet's think step by step.` and no `<\|eot_id\|>`; median response 907 chars; **1/28** direct answers (default: 57 %); the model continues with " \n\nStatement 1: …" i.e. it reasons after the prefill; smoke aggregate 46.4 on 28 items |

Full runs submitted 12:3x, identical settings to the canonical ones (`-p general-gpu`, denylist excluded,
request cache on; BBH 30 h limit since the responses are no longer cut short, MMLU-Pro 10 h, `LIMIT=100`):

| jobid | run | status |
|---|---|---|
| 3060600 | base bbh_cot_fewshot_relaxed | CANCELLED 12:5x — user: the base is not re-run |
| 3060601 | base mmlu_pro_cot | CANCELLED 12:5x — same |
| 3060602 | srllama bbh_cot_fewshot_relaxed | PENDING |
| 3060603 | srllama mmlu_pro_cot | PENDING |

**User decision (2026-09-15 12:5x): the base is NOT re-run under the variants.** Its default row already reproduces
the published Llama-3.1-8B-Instruct numbers (check 1), so the SR-variant rows are compared against the base's
DEFAULT rows (45.64 / 71.23); `report.py` check 3 marks such rows `variant*`. Caveat to keep in mind when reading
them: the knobs could in principle lift the base too (it has 411 `no_phrase` BBH items of its own), so an SR-variant
number at parity with the base default is a lower bound on the base's own variant score — state it that way.

Results land in `eval_general/srllama/<variant>/` and `report.py` check 3 prints them next to the default rows.

### Why SR-Agent skips the chain of thought on MMLU-Pro but not on BBH (user question, 2026-09-15 13:0x)

Checked against the training data and the exact prompts lm-eval logged, not guessed:

1. **SRFT's reflection is conditioned on the tool-agent context.** `LLaMA-Factory/data/toucan_32B_v3_base.json`:
   3,698 trajectories, ONE system prompt for all of them (the tool-use agent prompt with a `<tools>` block), and
   every one of the 6,847 final assistant turns and 15,492 function-call turns starts with `<think>…</think>`.
   Visible answers after the think block are not short (median 1,228 chars, 1 % under 200). In the general
   benchmarks the model never emits `<think>` — 0 of 1,400 MMLU-Pro, 0 of 6,511 BBH, 0 of 28 smoke responses —
   because none of these prompts carries the tool-agent system prompt. So "SRFT teaches thinking" holds *in the
   agentic regime it was trained in*; in a plain QA chat the reflection does not fire, and what governs the response
   is the generic chat prior, which the SFT shifted.
2. **What that prior does is imitate the in-context assistant-turn pattern, and lm-eval's MMLU-Pro rendering makes
   that pattern "say nothing".** With `--fewshot_as_multiturn` the stock MMLU-Pro config (`fewshot_config.doc_to_target: ""`)
   puts each worked exemplar — question + chain of thought + "The answer is (X)" — into the **user** turn and leaves the
   five **assistant** turns **empty** (verified on the logged prompt: 5 × `assistant len=0`). The base's RLHF prior
   follows the system instruction "Think step by step …" anyway (2 % one-liners); SR-Agent follows the demonstrated
   pattern (57 % one-liners). BBH is the control: its exemplar CoT sits in the **assistant** turns
   (`assistant len=369, "Let's think step by step.\n(0) At the start: …"`), and there SR-Agent reasons on every
   item — its only BBH problem is the blank line between steps hitting the `\n\n` stop.
3. It is not a length or capacity issue: SR-Agent's BBH responses run to 1,495 chars / 8 steps once the stop string is
   relaxed, and its MMLU-Pro-CoT smoke responses have a 907-char median after the prefill.

Implication for the write-up: the MMLU-Pro default row measures how strongly each model copies a degenerate
few-shot rendering, which is why the CoT-prefill variant is the informative one. An alternative, arguably cleaner
variant would fix the rendering itself (put the exemplar CoT in the assistant turns, as BBH does); not run for now.

### Paper framing for the general-capability block (user, 2026-09-15 15:xx)

The four benchmarks cover the three things a reviewer will ask about: **MMLU** = world knowledge, **IFEval** =
instruction following, **MMLU-Pro + BBH** = reasoning. Story to tell, conditional on check 3 landing near the base:

- MMLU and IFEval: parity under the default protocol, nothing to explain.
- MMLU-Pro: SR-Agent-Llama's *trigger* for visible reasoning moved — SRFT teaches `<think>` conditioned on the
  agent context, so in plain QA it has to be prompted once ("Let's think step by step." prefill); once triggered its
  score is the evidence that the reasoning ability itself is intact. Report the default row (40.86) AND the
  prefill row, side by side.
- BBH: an evaluation artifact, not a trigger issue — SR already reasons on every item; the stock `\n\n` stop string
  cuts its multi-paragraph answers before the answer sentence. Report the default row (64.26) AND the relaxed-stop
  row, side by side.
- State that the base was not re-run under the variants (check 1 already matches the published base row), so an
  SR-variant number at parity with the base default is a lower bound on the base's own variant score.

## MMLU-Pro-CoT result (job 3060603, 3 h 20, 1,400 items; samples replayed by 3063118) — 2026-09-15 18:0x

**Strict (lm-eval `custom-extract`): SR-Agent-Llama 38.86 vs base default 45.64 (−6.79)** — *lower* than the default
SR row (40.86). Per subject the knob did what it was meant to on the reasoning-heavy subjects and hurt the
recall-heavy ones: math 26 → **53** (base 55), chemistry 23 → 26, computer_science 39 → 43; biology 65 → **42**
(base 61), economics 48 → 41, psychology 56 → 48.

**Why: once it reasons, SR-Agent concludes in prose instead of the templated sentence.** Per-item classes
(`analyze_samples.py --task mmlu_pro_cot`, SR-CoT vs base default, same robust extractor for both):

| | base (default) | SR-Agent (CoT prefill) |
|---|---|---|
| correct (strict) | 639 (45.6 %) | 544 (38.9 %) |
| format_only — right answer stated, strict regex missed it | 29 | **99** |
| no_phrase — no option letter stated at all | 121 | **165** |
| **wrong** — commits to a different letter | **611** | **592** |

Decomposition of the −6.79: **format_only −5.00, no_phrase −3.14, wrong +1.36.** SR-Agent gets *fewer* answers
wrong than the base when it reasons; the whole deficit is how it ends the response: "…I will choose B as the
most representative example", "The correct definition is … which is option E", "…which is the definition of an
endergonic reaction" (right answer, no letter). 21.8 % of its CoT responses never say "answer is" (default SR
9.6 %, base 12.9 %); only 49 of those hit the 2,048-token cap.

**Under one symmetric robust extractor** (last "answer is (X)"; else the last conclusion phrase — "choose X",
"option X", "best answer is X", a final "X. …" line; case/markdown/parenthesis-tolerant; the same code path on the
base's responses, which also gain +2.07 from it):

| protocol | base | SR-Agent | gap | 2 se |
|---|---|---|---|---|
| default, strict (reported row) | 45.64 | 40.86 | −4.79 | 3.67 |
| CoT prefill, strict | 45.64 | 38.86 | −6.79 | 3.68 |
| default, robust | 47.71 | 43.86 | −3.86 | 3.76 |
| **CoT prefill, robust** | **47.71** | **45.93** | **−1.79** | **3.77 → PARITY** |

So MMLU-Pro needs **two** style corrections before it measures knowledge on SR-Agent: (1) a trigger to reason at
all (the prefill; math 26 → 53), and (2) an extractor that accepts a prose conclusion (the robust rule; +7.1 points
on the CoT run vs +2.1 on the base). With both, SR-Agent is inside 2 se of the base, and its count of genuinely
wrong answers is the lower of the two. The remaining 165 `no_phrase` items include conclusions that name the
right option in words only (e.g. "endergonic" for A); an option-text matcher would recover some of those for both
models — not done, the letter-based rule is easier to defend.

How to report: the default strict row stays; add the CoT-prefill row *with the robust extractor*, and state the
robust rule and that it is applied to the base's responses too. Do not report the strict CoT number alone — it
is the least informative of the four.

### "answer is (X)" is NOT the benchmark's requirement — it is lm-eval's (user question, 2026-09-15 18:3x)

Checked against the upstream file `TIGER-AI-Lab/MMLU-Pro/evaluate_from_local.py` (fetched 2026-09-15). The
OFFICIAL MMLU-Pro extraction is three-tier: (1) `answer is \(?([A-J])\)?`; (2) else `[aA]nswer:\s*([A-J])`;
(3) else `\b[A-J]\b(?!.*\b[A-J]\b)` — **the last standalone A–J letter anywhere in the response**. lm-eval 0.4.9's
`custom-extract` implements tier 1 only (tier 2 sits commented out in its yaml). So the harness we run is stricter
than the benchmark's own scorer, and the fix needs no re-inference and no home-made rule: re-score the saved
responses with the official code. `eval_general/score_mmlu_pro_official.py` does exactly that (same code path for
every run):

| run | lm-eval strict | **official extraction** | tier hits (1 / 2 / 3 / none) |
|---|---|---|---|
| base default | 45.64 | **47.50** | 1180 / 0 / 127 / 93 |
| SR-Agent default | 40.86 | **43.79** (−3.71, 2 se 3.76 → parity, marginal) | 1216 / 1 / 135 / 48 |
| SR-Agent CoT prefill | 38.86 | **45.79** (−1.71, 2 se 3.77 → **PARITY**) | 984 / 0 / 269 / 147 |

Per subject under the official extraction, SR-CoT vs base: math **58 vs 56**, computer_science 50 vs 50,
psychology 57 vs 57, law 32 vs 30, business 56 vs 53; biology 55 vs 67 and chemistry 34 vs 41 remain the two
subjects clearly below. Note the base's own official score (47.50) sits +1.0 from the published 46.5, closer than
the lm-eval number was, consistent with the published row having been produced by the official scorer.

**This supersedes the home-made robust extractor for the paper**: report MMLU-Pro with the benchmark's official
extraction (base 47.50 / SR default 43.79 / SR CoT-prefill 45.79), state that lm-eval generated the responses and
the official `evaluate_from_local.py` regexes scored them, and keep the lm-eval-strict numbers in the appendix
with the explanation above. The trigger story (default → prefill) still holds on math: 29 → 58.

### BBH has no official scorer (checked 2026-09-15 18:5x)
`suzgunmirac/BIG-Bench-Hard` ships only `bbh/` (data), `cot-prompts/` and `code-davinci-002-outputs/`; the paper
(Suzgun et al. 2022) extracts the text after "the answer is" and exact-matches it against the target. lm-eval's
`get-answer` filter is that convention with two quirks — case-sensitive "the answer is" and dropping the final
character (it assumes a trailing period). Plan for the BBH row: score the relaxed-stop run with lm-eval's own
filter first (that is what the published rows use); only if that still trails, show the case-/punctuation-tolerant
re-score (`analyze_samples.py --task bbh_relaxed`, same rule for the base) as the secondary number.

## Why our base MMLU is 68.0 against the published 72.0 — a different metric, not a weaker model (2026-09-15 20:xx)

User's concern: a 4-point lower base makes SR-Agent look worse next to Meta-SecAlign-8B. Settled by reading
Meta-SecAlign's released evaluation code (`facebookresearch/Meta_SecAlign`, `lm_eval_config/` + `test_lm_eval.py`,
now vendored verbatim under `eval_general/tasks/meta_secalign/` with a README):

- Their "MMLU" is **`meta_mmlu_0shot_instruct`: Meta's own Llama-3.1 recipe — 0-shot chain-of-thought, generative,
  answer extracted with `best answer is ([A-Z])`, 1,024 tokens**, prompts taken pre-rendered (Llama-3.1 chat format
  baked in) from the gated `meta-llama/Llama-3.1-8B-Instruct-evals` dataset, `apply_chat_template=False`. That is the
  protocol behind Meta's model-card number **73.0** ("MMLU (CoT), 0-shot"); Meta-SecAlign reproduces it as 72.0.
- Our "MMLU" is lm-eval's stock `mmlu`: **0-shot loglikelihood over the option letters, no CoT**, the Hendrycks /
  Open-LLM-Leaderboard convention. Meta's model card gives **69.4** for the closest loglikelihood-style setting
  (5-shot `macro_avg/acc`); our 68.00 is that kind of number. The two metrics differ by 3–4 points on the *same*
  weights — the gap is the metric, not the model.
- The other three rows match because there the protocols coincide: their `meta_mmlu_pro_instruct` is 5-shot CoT
  generative (ours too), `meta_bbh` is 3-shot CoT generative (ours too), `meta_ifeval` is IFEval with the four-metric
  mean computed in `test_lm_eval.py` (which also confirms the "mean of 4" reading of 79.1).
- Two useful side facts from their configs: (i) their BBH uses **`until: "\n\nQ: "`** — exactly the relaxed stop
  string we chose, not lm-eval's bare `"\n\n"`, so their BBH row never truncated at a blank line; (ii) their
  MMLU-Pro is the full 12,032-item set with the `best answer is` regex.

**The clean fix is to run Meta-SecAlign's own configs on our base and on SR-Agent-Llama** (`--include_path
eval_general/tasks/meta_secalign`, `CHAT=0`, `SHOTS=0`, which the runner now sets for any `meta_*` task). Then every
cell in the table is the same recipe as the published Meta-SecAlign-8B row and no "our harness vs theirs" caveat is
needed. Cost per model (H100, HF backend): MMLU 14,042 generative CoT items ≈ 4–6 h; MMLU-Pro 12,032 × 5-shot CoT
≈ 8–12 h; BBH 6,511 ≈ 4 h; IFEval 541 ≈ 1 h. Smoke first (`LIMIT=5`, general-short) to confirm the pre-rendered
prompts run correctly through the HF backend without a template and without a doubled BOS.

### Meta-protocol smoke passed; MMLU under Meta's recipe submitted for both models (2026-09-15 21:5x)
Smoke 3064704 (`meta_mmlu_0shot_instruct`, base, LIMIT=5, `LOG_SAMPLES=1`): the logged prompt is the pre-rendered
Llama-3.1 chat string verbatim (starts `<|start_header_id|>user<|end_header_id|>`, ends with the open assistant
header, no template applied, no BOS added), gen kwargs `until: []`, 1,024 tokens; all five base responses end with
"The best answer is X." and `strict-match` extracts the letter (4/5 correct). Full runs submitted: base **3064721** and
srllama **3064722** × `meta_mmlu_0shot_instruct` (14,042 items, 12 h limit, `-p general-gpu`, denylist). The other three Meta
configs (`meta_mmlu_pro_instruct`, `meta_bbh`, `meta_ifeval`) are ready to submit on the user's decision.
## BBH-relaxed result (job 3060602, 11 h 04, 6,511 items, lm-eval's own `get-answer` filter) — 2026-09-16 01:2x

**SR-Agent-Llama 68.68 ±0.51 vs base default 71.23 ±0.51: −2.55, 2 se 1.45 — still outside 2 se, but 64 % of the
default-protocol gap (−6.97) was the blank-line truncation.** Same generation as the default run except that the
harness no longer stops at `"\n\n"` (their published recipe, `meta_bbh`, uses the same `"\n\nQ: "` stop — see the
vendored configs), so subtasks that never hit a blank line are bit-identical between the two SR runs.

Where the truncation was (SR default → SR relaxed, base in brackets): tracking_shuffled_objects_seven **12.4 → 80.8**
(83.2), penguins_in_a_table **51.4 → 78.1** (82.2), snarks 55.1 → 69.1 (72.5), reasoning_about_colored_objects
48.4 → 55.6 (70.0), date_understanding 58.0 → 63.6 (67.6), movie_recommendation 57.2 → 61.2 (66.0).

What remains below the base after the fix: geometric_shapes 38.0 (54.8, −16.8), reasoning_about_colored_objects
55.6 (70.0, −14.4), disambiguation_qa 54.4 (64.4, −10.0), tracking_shuffled_objects_five 80.0 (89.6, −9.6); seven
subtasks are above the base (hyperbaton +7.2, logical_deduction_three +6.4, sports_understanding +3.6, …).
Per-item classification of the residual (extraction vs genuinely wrong) follows from the sample replay.

**Residual after the stop-string fix, per item** (`analyze_samples.py --task bbh_relaxed`, SR-relaxed vs base default,
samples replayed from cache): of the −2.55, **wrong −1.20, no_phrase −0.78, format_only −0.57**; under the symmetric
tolerant scorer 71.17 vs 73.17 (−2.00, still outside 2 se = 1.45). The genuine part sits in `geometric_shapes`
(SVG-path shape recognition: 38.0 vs 54.8, 145 wrong) and `tracking_shuffled_objects_five` (80.0 vs 89.6);
`reasoning_about_colored_objects` is pure formatting (57 format-only items; 78.4 vs 76.4 under the tolerant scorer).
**Conclusion for the paper: BBH is the one row with a small real residual (≈2 points), say so; do not claim parity.**

### BBH FINAL (user decision 2026-09-16 01:4x): tolerant extractor, base 73.17 vs SR-Agent 71.17
The reported BBH row is the **base default run vs the SR-Agent relaxed-stop run, both scored with the tolerant
extractor** of `analyze_samples.py` (last "answer is …" in the response, case-insensitive, markdown/parentheses/trailing
punctuation stripped, option letter compared; word/number targets compared after the same normalisation). One rule for
both models; the base itself gains +1.94 from it (126 of its answers are "The answer is (X)" with a capital T, which
lm-eval's case-sensitive regex misses). **73.17 ±0.55 vs 71.17 ±0.56, −2.00, 2 se 1.57 — a small real residual,
reported as such.** `report.py` now computes these two numbers from the samples and marks them `tolerant` in the
provenance; lm-eval-strict (71.23 / 68.68) and the truncated default (64.26) are kept in check 3 / the appendix.
