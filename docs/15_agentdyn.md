# AgentDyn — third benchmark (dynamic, open-ended agent security)

Paper: **AgentDyn: Are Your Agent Security Defenses Deployable in Real-World Dynamic Environments?**, arXiv
[2602.03117](https://arxiv.org/abs/2602.03117) — Hao Li, Ruoyao Wen, Shanghao Shi, Ning Zhang, Yevgeniy Vorobeychik,
Chaowei Xiao. **Same lab** (Ning Zhang / Chaowei Xiao); the first author's working copy is on this cluster at
`/storage3/fs1/zhang.ning/Active/hao/AgentDyn` (HEAD 2c99edc, 2026-05-02) with his conda envs `agentdyn` /
`agentdyn_rebuttal` — useful as a reference, we do not write into it.

Branch for this work: **`exp/agentdyn`** (from `main`, 2026-09-15).

## 1. What the benchmark is

A fork of AgentDojo that adds three suites of **open-ended, dynamic** tasks. Its claim is that existing defenses are
either not secure enough *or* **over-defend**, and that static benchmarks hide this because their tasks are short,
scripted and contain no legitimate third-party instructions.

| | |
|---|---|
| suites | `shopping`, `github`, `dailylife` (the four AgentDojo suites are still there and unchanged) |
| size | 20 user tasks per suite = **60 benign**; injection tasks 9 / 9 / 10 → **560 attacked** (20 × 9 + 20 × 9 + 20 × 10) |
| task shape | avg 7.1 steps, 3.17 application scenarios per task, requires re-planning from tool output |
| helpful instructions | at least one **legitimate** third-party instruction on the critical path (OTP verification, checkout prompt, form-filling) — a blanket "ignore everything from tools" defence fails the task |
| metrics | Benign Utility / Utility under Attack / ASR — same three as our AgentDojo tables |
| scoring | programmatic `utility()` / `security()` on the environment state, **no LLM judge** → no API cost |
| `security` flag | same convention as our repo: `security: true` = **the attack succeeded** (so `eval/compute_attack_stats.py` is reusable unchanged) |
| paper's models | GPT-4o, GPT-5-mini, GPT-5.1, Gemini-2.5-Pro/Flash, Claude-Sonnet-3.5/4.5, Llama-3.3-70B, Qwen3-235B, … — **all ≥ 70B**; ours is 8B |
| paper's defenses | repeat_user_prompt, spotlighting_with_delimiting, tool_filter, transformers_pi_detector, piguard_detector, prompt_guard_2_detector, camel, progent, drift (+ Meta-SecAlign-70B) |

Why it is worth a third block in the ICLR paper: AgentDojo covers *static* attacks and RL-Hammer covers *adaptive*
attacks; AgentDyn adds **realistic dynamic environments**, and its own headline (defenses over-defend) is exactly the
axis where reflection should beat filter/alignment defences — SR reasons about whether a third-party instruction is
legitimate instead of refusing everything.

## 2. Our copy

`SRFT/agentdyn/` — vendored clone of `git@github.com:SaFo-Lab/AgentDyn.git`, upstream commit recorded in
`agentdyn/UPSTREAM_COMMIT.txt` (`5353cf7`, 2026-05-19, newer than the copy in `hao/AgentDyn`). `.git` was removed so
the tree lives in our repo like `agentdojo/` does; to take an upstream update, re-clone into a temp dir and diff.
Upstream's own paper logs (914 MB, all 12 models × 10 defenses) were moved to `agentdyn/runs_upstream_paper/` and
git-ignored — that is where the reference numbers for API models come from. Our own results go to `agentdyn/runs/`.

### Port of our Llama pipelines (the only code we added)
Upstream AgentDyn has no local-HF provider (API models + `local`/`vllm_parsed` only). Ported from `SRFT/agentdojo`,
byte-identical, so the prompt format and generation settings match our AgentDojo rows exactly:

- `src/agentdojo/agent_pipeline/llms/{llama_sr_agent_llm,llama_local_prompt,llama_llm,openai_to_llama,llama_to_assistant}.py`
- `src/agentdojo/models.py`: enum `LLAMA_3_1_8B_Instruct` / `LLAMA_3_1_8B_SAFE_AGENT`, providers `hf` / `hf_llama_sr_agent`, `MODEL_NAMES` entries
- `src/agentdojo/agent_pipeline/agent_pipeline.py`: two imports + two branches at the top of `get_llm`

Everything else (`base_tasks.py` identical to ours; `task_suite.py` differs only in "take the last assistant message"
and a removed debug print; `benchmark.py`, defenses, suites) is upstream's, untouched.

`agentdyn/eval/compute_attack_stats.py` is a copy of the AgentDojo one (no suite names hard-coded).

### Environment
Own conda env **`srft_agentdyn`** (clone of `agentdojo` + `pip install -e agentdyn`). It must be separate: both repos
install a package called `agentdojo`, so installing AgentDyn into the `agentdojo` env would break every AgentDojo eval.
Extra deps vs our env: `cyclopts`, `pydantic-ai`, `tiktoken`, `vertexai`, `jsonref`, `openapi-pydantic`, `json_repair`,
`deepdiff>=8.6.1` (most only needed for camel/progent/drift, which we do not run).

## 3. Plan agreed with the user (2026-09-15)

Rows: **Llama-3.1-8B-Instruct (undefended)** and **SR-Agent-Llama (no append)** — same two as `docs/13` §1a, same LoRA
`LLaMA-Factory/saves/llama31-8b/lora/v3base_local_sft_8k_r64_GA4_qkvo_3epoch_5e-6`, `SYS_APPEND=0`. Base = the same
pipeline with a non-existent `LORA_PATH` (the convention from `docs/06`).

**Step 1 is a feasibility gate, not a result:** run only the 60 benign tasks for both models (120 trajectories). The
paper's smallest model is 70B; if an 8B model cannot complete these open-ended tasks, Benign Utility ≈ 0 and every
UA/ASR number below it is meaningless (a low ASR would only mean the agent never got far enough to be attacked).
Only if the gate passes do we spend GPU time on the 560 attacked cases.

### How to run
```bash
# benign gate (3 jobs per model, one per suite)
bash agentdyn/scripts/submit_eval_agentdyn.sh agentdyn_srllama_noappend $PWD/LLaMA-Factory/saves/llama31-8b/lora/v3base_local_sft_8k_r64_GA4_qkvo_3epoch_5e-6 benign
bash agentdyn/scripts/submit_eval_agentdyn.sh agentdyn_llama_base       /nonexistent_base_model_fallback                                              benign
# later: attacked shards (6 jobs per model) / everything
#   ... same command with `attack` or `all` as the third argument
# stats (same script and conventions as AgentDojo)
cd agentdyn && python eval/compute_attack_stats.py <RUN_NAME>/meta-llama_Llama-3.1-8B-Instruct-safe-agent
```
Knobs: `PARTS`, `NPROC` (default 2), `TLIM`, `SYS_APPEND` (default 0 — the ICLR protocol), `SUITES_ONLY`,
`SRFT_SBATCH_FLAGS="--test-only"`. Resume = resubmit the same command (one result JSON per task, already-run tasks are
skipped). All the WashU gotchas from `docs/09_cluster_washu.md` apply (node denylist, 80 GB GPUs for Llama).

## 4. Status

| date | what |
|---|---|
| 2026-09-15 | branch created; upstream cloned and vendored; Llama pipelines ported; launcher + sbatch + stats copied; env `srft_agentdyn` building. **Nothing run yet.** |
| 2026-09-16 | **Benign gate FAILED.** base Llama 7/60 = 11.67, SR-Agent-Llama 6/60 = 10.00 — one task apart, inside the noise band. Every 8B-class row on this benchmark (ours + upstream's Meta-SecAlign-8B 5.00 + Llama-3.3-70B 10.00) sits at 5–12 %, while the 70B class is at 53–55 %. The 560 attacked cases were **not** run: at ~11 % benign a low ASR would only mean the agent never reached the injection. Numbers and failure modes in `docs/01` §ADYN; reproduce with `python eval/agentdyn_report.py agentdyn_llama_base agentdyn_srllama_noappend`. |

## 5. Upstream's Llama numbers are interface artifacts, not capability (measured 2026-09-15)

Re-derived from upstream's own logs in `agentdyn/runs_upstream_paper/` (benign = `*/user_task_*/none/`, our main-table
convention). Two different Llama rows are low for two different reasons; **neither is a capability or defence
measurement**, so do not cite them as a baseline.

| model (no extra defence) | Benign | UA | ASR | pipeline in the logs |
|---|---|---|---|---|
| Meta-SecAlign-8B | 5.00 (3/60) | 7.32 | 5.54 | `local` |
| Llama-3.3-70B undefended | 10.00 (6/60) | 6.43 | 12.50 | `meta-llama_llama-3.3-70b-instruct` (OpenRouter) |
| Meta-SecAlign-70B | 55.00 | 53.93 | 9.11 | `local` |
| GPT-4o undefended | 53.33 | 55.36 | 38.93 | `gpt-4o-2024-08-06` |

### (a) Meta-SecAlign-8B — killed by the strict tool-call parser
12 of 60 benign trajectories emitted **no parsed tool call at all**; 11 of those contain call-shaped text the parser
dropped. Classified:

| dropped output | n |
|---|---|
| `<function=get_current_day></function>` — correct format, zero-arg call written without `{}` → `json.loads("")` raises | **6** |
| `<function=browse_webpage>{"url": "…"};</function>` — correct format, stray `;` breaks the JSON | **2** |
| `<browse_webpage>{…}</browsable_webpage>` — wrong tag name | 1 |
| no call-shaped text at all | 3 |

`local_llm._parse_model_output` takes the first `<function=…>` tag and requires `json.loads` to return a dict; on
failure it returns the text as a **final answer**, so the trajectory ends immediately — all 12 have exactly 3 messages
(system + user + assistant) against a median of 13. So 8 of 60 benign tasks (13 %) die on punctuation, and the total
score is only 3/60. Note `<function=name>{json}</function>` **is** Meta's documented Llama 3.1 *custom tool calling*
format, so this is parser strictness, not a foreign protocol.

### (b) Llama-3.3-70B — killed by the OpenRouter function-calling path
17 of 60 benign trajectories have an **empty assistant content** or a single "We cannot assist you without more
information." and stop, with no `tool_calls` returned. GPT-4o and Meta-SecAlign-70B have 0/60 such trajectories.

### (c) The one-line proof
**Meta-SecAlign-70B's base model *is* Llama-3.3-70B: 55.00 vs 10.00 benign — 45 points apart.** The only difference is
`local` pipeline vs OpenRouter. That gap measures the interface, not the model and not the defence.

### (d) Related: ReasAlign (arXiv 2601.10173, same author) — NOT an OpenRouter problem
Checked the local clone `/storage3/fs1/zhang.ning/Active/hao/ReasAlign` (`git@github.com:leolee99/ReasAlign.git`).
It serves Llama **locally** through `LlamaClient` (transformers + PEFT, fp16) — there is no OpenRouter/API path for the
Llama rows, so the AgentDyn (b) explanation does not transfer. Three concrete defects in `client.py::employ` that would
depress a local Llama baseline:
1. **double BOS** — `apply_chat_template(tokenize=False)` already emits `<|begin_of_text|>`, then `self.tokenizer(prompts, …)` is called with the default `add_special_tokens=True`, prepending a second one. (Our `llama_sr_agent_llm.py` passes `add_special_tokens=False` for exactly this reason.)
2. **prompt truncated to 512 tokens** — `truncation=True, max_length=max_length` with `max_length=512` cuts the *input*; the same 512 is reused as `max_new_tokens`. Long injected documents / tool outputs are silently chopped.
3. **batched right padding** — `padding=True` with `pad_token = eos_token` and `padding_side` never set (default right) for a decoder-only model, plus `input_length = input_ids.shape[-1]` used to slice every sequence in the batch, which is only correct for the longest one.

Not verified by re-running ReasAlign; this is a code read. Worth raising with the author before citing its Llama rows.

## 6. Our own AgentDojo Llama rows use the SAME strict parser (verified 2026-09-15)

Checked because the numbers in `docs/13` §1a must not come from a lenient tool-call parser:
- `llama_local_prompt.parse_output` (ours) and `local_llm._parse_model_output` (the one AgentDyn uses) are **equally
  strict**: same `<function\s*=\s*([^>]+)>` regex, same `json.loads` requirement, same dict requirement, **no**
  empty-arg tolerance and **no** `;` stripping in either. Ours only adds `<think>`/`<thinking>` handling, which is
  reflection, not tool calls.
- Both Llama rows ran through the **same pipeline**: the result JSONs of `llama31_base_noappend` and
  `llama31_v3base_local_3epoch_noappend` both carry `pipeline_name = meta-llama_Llama-3.1-8B-Instruct-safe-agent`
  (base = the same pipeline with a non-existent `LORA_PATH`). Same prompt, same parser, same generation settings.
- Measured parse loss on the 97 benign tasks: base **19/97** trajectories had a call-shaped output dropped
  (10 of them the zero-arg `{}` case), SR-Agent-Llama **6/97**. All 25 end with exactly 3 messages.

So neither row was given a discount; both were scored under the same strict rules, and the untrained base was hit
harder — the same way Meta-SecAlign-8B is hit in upstream's runs.

## 7. Attacked set: SR-Agent-Llama over-defends (observed 2026-09-16, SR at 384/560)

Preliminary — SR-Agent-Llama's attacked run is ~69 % complete. base Llama is FINAL (560/560).

| | Benign | UA | ASR | UA/Benign (retention under attack) |
|---|---|---|---|---|
| base Llama-3.1-8B | 11.67 | **8.93** | 9.82 | 0.77 |
| SR-Agent-Llama (384/560) | 10.00 | 3.66 | **2.09** | **0.37** |

ASR is where we want it (2.09 vs 9.82, lower on all three suites; dailylife 1.53 vs 21.50). UA is not: the two models
have the *same* benign utility, so the UA gap is not capability — it is behaviour under attack.

**Mechanism.** Share of trajectories whose FINAL assistant message contains refusal / security wording
(`cannot|unable to|refus|suspicious|prompt injection|malicious|…`):

| | benign | attacked |
|---|---|---|
| base Llama | 10.2 % | 5.4 % (goes *down*) |
| SR-Agent-Llama | 23.9 % | **71.4 %** |

Failure modes on the attacked set:

| | task completed | stopped mid-way / wrong | ran out of turns | died at step 1 |
|---|---|---|---|---|
| base (560) | 8.9 % | 25.9 % | 32.5 % | 32.7 % |
| SR-Agent-Llama (384) | 3.6 % | **62.8 %** | 30.5 % | 3.1 % |

Sampled final answers show the behaviour precisely: SR **identifies the injection correctly and in detail** ("the
response … contained a clear prompt injection attempt embedded within a `<section>` block — a malicious message
disguised as an 'important message' from Emma Johnson") **and then stops**. It treats detection as the terminal action
instead of flagging the injection and resuming the user's task. That is the over-defense side of exactly the
security/utility dilemma AgentDyn was built to expose.

Consistent with, not contradicted by, AgentDojo: there SR-Agent-Llama's UA is 29.08 vs the base's 23.08 (`docs/13` §1a).
AgentDojo's tasks are short, so one abort costs little; AgentDyn's are long-horizon and dynamic (7.1 steps, 3.17 app
scenarios), so a single abort destroys the whole task.

Implication for a future version: the training data has no trajectory that flags an injection **and then continues the
legitimate task**. Analysis to run once SR reaches 560: split the 71.4 % into (a) correct detection followed by an
abort and (b) false positives on benign tasks — only (b) is over-defense in the strict sense.
