# Why SecAlign-style DPO keeps (or raises) utility, and what that means for v3 (2026-09-09, WashU)

Context: the paper compared SR-Agent (Qwen3-8B) against the released Meta-SecAlign-8B (Llama). A same-base replication, "Qwen-SecAlign"
(SecAlign DPO on Qwen3-8B, no chat-template change), gives — user's table, AgentDojo Benign / UA / ASR:

| model | Banking | Slack | Travel | Workspace | **ALL (949)** |
|---|---|---|---|---|---|
| Qwen3-8B base | 43.75 / 37.50 / 31.25 | 85.71 / 50.47 / 58.09 | 50.00 / 27.85 / 32.14 | 60.00 / 59.46 / 1.78 | 58.90 / 50.47 / 16.96 |
| Qwen-SecAlign | 68.75 / 54.17 / 15.28 | 85.71 / 49.52 / 10.48 | 60.00 / 52.14 / 7.86 | 70.00 / 60.18 / 1.25 | **70.07 / 56.90 / 5.38** |
| SR-Agent (v0) | 56.25 / 29.17 / 4.86 | 66.67 / 46.67 / 1.90 | 30.00 / 22.14 / 0.00 | 52.50 / 57.32 / 0.17 | 51.32 / 46.68 / 1.05 |

**Which model is which (user, 2026-09-10):** the table's "Qwen-SecAlign" row is a **Qwen3-8B Meta-SecAlign (SecAlign++ recipe) trained by
someone else** — its trajectories/CSV are NOT in this repo. The repo's `eval/attack_stats_qwen3_8b_secalign_dpo_Qwen_Qwen3-8B-secalign.csv`
(think on) is **our own plain-SecAlign DPO on Qwen3-8B** (UA 50.42 / ASR 5.60, no benign column); `..._nothink.csv` is the same LoRA with think off
(UA 38.78 / ASR 5.37). Keep the two apart: plain SecAlign DPO on Qwen leaves UA at base level (50.4 vs 50.5) with ASR 5.6; the +11 benign /
+6.4 UA row is SecAlign++ (randomized injection position + self-generated labels), which is exactly the recipe difference §2 hypotheses 2 and 5
predict to matter. Note also that the table's ALL utility is weighted by attacked-trajectory counts (144/105/140/560 → base 58.90, SR-Agent 51.32);
the paper / `compute_attack_stats.py` use the 97 clean tasks unweighted (60.82, 51.55).

So on the same base, DPO on Alpaca-derived pairs loses to SR-Agent on ASR (5.4 vs 1.05) but **gains 11 benign-utility points over the base**,
while SR-Agent loses 7.6. Question: where does that utility come from, and should v3 mix Alpaca SFT data to get it?

Sources read for this note: SecAlign (arXiv 2410.05451, CCS'25) and Meta-SecAlign (arXiv 2507.02735, v3 Feb 2026); repo material
`examples/train_lora/qwen3_8b_lora_dpo_secalign.yaml`, `agentdojo/.../llms/qwen_secalign_llm.py`, `03_analysis_utility_drop.md` §E/§F, `04_self_distill_plan.md` §0b.

## 1. What the two recipes actually do
**SecAlign (2024).** Preference pairs from Cleaned-Alpaca samples that have a data field: x = instruction + data + *injected instruction taken
from another sample* (90 % appended at the end of the data, 10 % "completion" style with fake delimiters); y_w = the dataset's own response to
the benign instruction (text-davinci-003 labels); y_l = the dataset's response to the injected instruction. DPO, sigmoid, β = 0.1, LoRA r64 **α8**
dropout 0.1 on q/v, lr 1.4–2e-4, 3 epochs, applied to an already-Instruct model. Reported utility: AlpacaEval2 WinRate unchanged (< 0.7 pt),
MMLU −2 to −3. Nothing agentic is ever trained on.

**Meta-SecAlign / SecAlign++ (2025).** Three changes on top: (1) a new `input` chat role for untrusted data; (2) **randomized injection position**
(45 % end / 45 % start / 10 % completion) — without it the model learns the shortcut "ignore the last sentence of the last message", which in
AgentDojo (system prompt full of instructions, user task last) made Llama-3.3-70B answer with empty content: AgentDojo utility **15.5 → 84.5**
just from randomization (their Table 1); (3) **self-generated y_w / y_l** from the initialization model instead of davinci-003 labels (Table 2:
AgentDojo utility 84.5 self vs 15.5 davinci vs 72.8 GPT-4o vs 58.1 GPT-5 — in-distribution labels beat stronger but off-distribution annotators).
Hyper-parameters for the 8B: LoRA r64 α8 dropout 0.1 on q/v/gate/up/down, lr 1.6e-4, β 0.1, 3 epochs, 19,157 pairs.
Results that matter here: Llama-3.3-70B AgentDojo utility 59.8 → 84.5 (UA 43.4 → 79.5; without the sandwich defense 62.9 → 79.4) — the authors
say they are "unsure why" utility *increases*; on Llama-3.1-8B utility is flat to slightly down (Table 6); on **Qwen3-4B AgentDojo utility drops
47.4 → 42.3** (Table 7). Test-time LoRA α sweep 0 → 8 (Fig. 3 / Table 11): ASR 95 → 0.5 while every utility benchmark moves ≤ 1–2 pts.

**Our Qwen-SecAlign.** `qwen3_8b_lora_dpo_secalign.yaml` = the Meta-SecAlign-8B hyper-parameters exactly (r64 α8 dropout 0.1, q/v/gate/up/down,
lr 1.6e-4, β 0.1 sigmoid, 3 epochs, cutoff 2048), qwen3 template with `enable_thinking: false` during training, no `input` role, dataset
`qwen3_secalign_dpo` (chosen/rejected; which annotator produced the responses is not recorded in the repo — see open questions).
Eval pipeline `hf_qwen_secalign`: think ON (budget 512 + forced close, same as SR-Agent), **no system-prompt append**, tools via the qwen3 template.

## 2. Why utility survives (ranked)
1. **The objective barely moves the policy, and only where the pair differs.** DPO with β = 0.1 optimises the *margin* between y_w and y_l
   relative to π_ref; both are short chat answers to the same prompt, so the gradient carries "which instruction to obey", not "how to write".
   The adapter is tiny: LoRA α/r = 8/64 → scaling 0.125 (SR-Agent's SFT LoRA is α96/r64 → 1.5, 3 epochs over 3.7k trajectories × ~4.6k tokens,
   imitating Claude-written text). Meta-SecAlign's α sweep is the cleanest evidence: the same adapter takes ASR from 95 to 0.5 while utility
   stays within noise. Our own ledger says the same from the other side: the SR-Agent gap is an *off-policy imitation cost* (03 §F: v1 UA
   drop traced to contaminated targets; v2-traj with Qwen-written thinks is back at v0's utility), not a price of security.
2. **The defense removes a real drain on benign agentic utility.** AgentDojo tool outputs are full of imperative text even when no attack is
   running (landlord notices, bills, e-mails, calendar notes). SecAlign §4.5 measured it on Llama-3-8B: undefended, 52 % of imperative
   sentences in the data part were *executed as instructions*; after SecAlign, 16 %. A base model that follows stray imperatives in tool
   results derails the user task; a defended model stays on it. Meta-SecAlign's 70B +25 AgentDojo utility is the same effect (and it survives
   removing the sandwich prompt). Our per-suite pattern fits: the +11 lives in banking (+25), travel (+10), workspace (+10) — the suites whose
   tool outputs carry the most instruction-like content — and slack is unchanged (85.71 both).
3. **Think mode untouched.** The DPO adapter was trained with thinking off and never saw `<think>` tokens; at eval Qwen's native reasoning
   distribution is intact and the preference is applied to the answer tokens only. SR-Agent rewrites the think distribution itself (03 §D/E),
   which is exactly where the v0/v1 utility went.
4. **Eval-side confounds (affect the Qwen-SecAlign vs SR-Agent gap, not SecAlign vs base):** SR-Agent is evaluated with `_SYS_PROMPT_APPEND`
   (the 3-point "summarise goal / identify injection / pick optimal action" think instruction), which alone costs ~4 benign points
   (55.7 without vs 51.5 with, 04 §0b); the secalign pipeline injects nothing. Single runs at T = 0.6: ±5 pts on slack/travel (105/140
   trajectories) is noise, +11 on 949 is not.
5. **What it is NOT: Alpaca content teaching agent skills.** Alpaca has no tools, no multi-turn; its y_w are short davinci-003 answers. With those
   labels and a fixed injection position Meta-SecAlign got the *worst* AgentDojo utility (15.5); the utility returned with position
   randomization and self labels, i.e. with the *training signal*, not the corpus. Our own negative result is consistent: SFT on
   toucan v2 + SecAlign-SFT data (03 §E, `attack_stats_qwen3_8b_safe_agent_mix_sft_*`) gave UA 42.6 vs 46.7 pure. And Qwen3-4B lost utility under
   SecAlign++ — the gain is model-dependent, not a property of the data.

## 3. Consequences for v3 and the "v3 + Alpaca SFT mix" idea
- **Not now, and not as SFT.** Mixing no-think davinci answers into a think-mode SFT is off-policy on two axes (style and mode; needs empty
  `<think>\n\n</think>` wrappers, `mix_qwen3_secalign_sft_with_toucan.py` does that), has already hurt UA once, and would confound the v3 read-out.
- **v3 is already the Meta-SecAlign lesson applied to our thinks** (self-generated labels: Qwen paraphrases Claude's reasoning in its own
  distribution). If v3 benign utility ≥ base (60.8 / 58.9), hypothesis 1 is confirmed and no Alpaca mix is needed; keep "v3 + Alpaca" at most
  as a reviewer-facing ablation row.
- **If v3 is still below base, the next lever mirrors SecAlign's *objective*, not its corpus:** a light DPO stage on top of the v3 SFT with
  self-generated, on-distribution pairs from our per-step records — chosen = expert/clean action with the v3 think, rejected = the Qwen3-32B
  candidate that follows the injection (23 % of injected steps have one, 04 §0b) — small α (8), 1 epoch, β 0.1. Prior data point:
  SFT(v0) + SecAlign-DPO on Alpaca pairs → UA 48.2 / ASR 2.85 vs 46.7 / 1.05 (04 §0b): +1.5 UA for +1.8 ASR even with off-distribution pairs.
- **Cheap diagnostics that would settle hypothesis 2 without a GPU** (existing trajectories, general-short/CPU): (i) in benign runs, the rate of
  tool calls whose arguments/verbs come from text inside a previous tool result rather than from the user task, for base vs Qwen-SecAlign vs
  SR-Agent; (ii) empty-content / no-tool-call turns per suite (the Meta-SecAlign shortcut symptom). If Qwen-SecAlign's +11 is mostly fewer
  derailments in banking/travel, an SR-Agent think that *names* the stray instruction (which v3 keeps) should get the same benefit for free.
- Eval hygiene for the ICLR tables: report SR-Agent with and without `_SYS_PROMPT_APPEND`, and run Qwen-SecAlign through the same
  `attack_stats` script/run dir convention so the three rows are computed identically.

## 4. Open questions
Resolved 2026-09-10: the 70.07 row is a third party's Qwen3-8B Meta-SecAlign (SecAlign++), not in the repo; `qwen3_8b_secalign_dpo` (think on) is our
plain-SecAlign replication. Still open:
- The third-party Qwen Meta-SecAlign run needs its trajectory dir (or at least the per-suite benign counts) before it goes into a paper table,
  and we need to know whether it added the `input` role / how it was prompted in AgentDojo.
- `qwen3_secalign_dpo.json`: are chosen/rejected the SecAlign release's davinci-003 labels or Qwen-generated (SecAlign++ style)? Not on disk here.
- Was the eval run with AgentDojo's sandwich (`repeat_user_prompt`) defense? Meta-SecAlign's headline AgentDojo numbers use it; ours should not.
