# ICLR rebuttal experiments (2026-11-09 → 2026-11-13, WashU)

Everything that was run for the ICLR rebuttal, in one place: why each experiment exists, exactly how it was run, the numbers, how to
read them, and what they do **not** show. The per-version fields (data, ckpt, job ids) are also in `01_experiments.md` §aba-cleanSFT and
`12_rl_hammer_injecagent.md` (last two sections); this file is the narrative index and the place to copy numbers from.
Every number below was read from a result file named next to it; commits are on `main`.

| # | experiment | answers | status |
|---|---|---|---|
| 1 | **aba-cleanSFT** — Llama-3.1-8B fine-tuned on the *clean* expert trajectories (no injection, no reflection); AgentDojo | ML5G W2 / Q1 | FINAL |
| 2 | **RL-Hammer on aba-cleanSFT** — (a) transfer matrix 7 attackers × 4 targets, (b) one dedicated attacker | ML5G Q1, Qhw2 W2, UquZ W3 | FINAL |
| 3 | **`tool_knowledge`** — second AgentDojo attack on base Llama, aba-cleanSFT, SR-Agent-Llama | Qhw2 W2, UquZ W3 | FINAL |
| 4 | **PAIR-style attack** on InjecAgent — base Llama, SR-Agent-Llama | Qhw2 W2 (second adaptive family) | FINAL |
| 5 | **Second evaluation of the three Qwen3-8B ablation attackers** | UquZ W3 / Q3 | FINAL |
| 6 | CPU-only analyses: paired base vs SR-Agent-Llama, travel suite, template/trigger overlap | ML5G Q1 / Q4 | done |
| 7 | Considered, not run: AutoInject, Nasr et al. search, Andriushchenko et al. random search, "no-candidate rationale" arm | Qhw2 W1/W2, all three (W1) | see §7 |

## 0. Context

**Reviews (scores 3 / 2 / 2, target 3 / 3 / 2).** UquZ: rating 3, confidence 2. ML5G: rating 2, confidence 3 — the reviewer most likely
to move; asks whether the Llama utility gain is "agentic SFT" and whether security then comes from a stronger task baseline (Q1), for
self-generated reflections (Q2), a scaling curve over the 38 injection tasks (Q3), the travel-suite UA drop (Q4), more baselines and a
Limitations section. Qhw2: rating 2, confidence 4, soundness 1 — wants the distinction from rationale SFT established (W1/Q1), more
adaptive attacks (AutoInject, Andriushchenko et al. 2024, Nasr et al. 2025; W2/Q2), more backbones (W3), and the trade-off metric dropped (W4).
All three ask, in different words, whether sampled reference-agent failures beat teacher rationales written without them — **not
answered by any experiment here** (§7).

**Compute.** The batch queue was FIFO with multi-day estimates (`09_cluster_washu.md` §Interactive GPU), so almost everything ran on one
interactive H100 80GB: `srun -p general-interactive --gres=gpu:1 -c 8 --mem=64G -t 3-00:00:00` inside tmux session `aba-srun`, job
**4468163**, 2026-11-10 22:08 → 2026-11-13 22:08. The existing job scripts were run unchanged with `bash <script>` inside it. The only
batch jobs were the 4-GPU RL-Hammer training **4490145** and its eval **4490146** (§2b).

**Models and protocol.** Llama family, the paper's main-table protocol: `MODEL=LLAMA_3_1_8B_SAFE_AGENT`, `SYS_APPEND=0`, AgentDojo
v1.2.1, 949 attacked + 97 benign tasks, single seed, T 0.6. Adapters (all in `LLaMA-Factory/saves/llama31-8b/lora/`):
SR-Agent-Llama `v3base_local_sft_8k_r64_GA4_qkvo_3epoch_5e-6`, ABL-A `abl_nothink_sft_…`, aba-cleanSFT `aba_cleansft_sft_…`.

---

## 1. aba-cleanSFT — agentic SFT on clean trajectories (ML5G Q1)

**Question.** SR-Agent-Llama's benign utility is above its base (27.84 → 37.11; banking 25 → 43.75). Is that the training data acting
as agentic SFT, and does the security gain then come from a stronger task baseline rather than from the defence? ABL-A cannot answer
it: its trajectories still contain the injection and are supervised with the non-hijacked action, i.e. it is security training.

**Data** — `LLaMA-Factory/data/toucan_32B_v3_base_llama_local_clean_nothink.json`, built by `ablation/build_clean.py` (CPU, ~1 min) from
the ABL-A data plus the injection ground truth in `toucan_32B_v3_base.json` (`meta.injections[*].combined`):
- the same **3,698 trajectories** / 22,339 assistant turns / 15,492 observations as SR-Agent-Llama's training set;
- 3,488 trajectories had their injection removed — 5,852 occurrences, inserted in exactly three ways: appended (`obs + "\n" + INJ`,
  2,335), prepended (`INJ + "\n" + obs`, 1,799), inside (`left + " " + INJ + " " + right`, 1,716), plus 2 observations that were the
  injection alone. Removing INJ with its separator restores the original observation (`"protein": 1. <INJ> 0` → `"protein": 1.0`);
- 210 trajectories never received their injection (`has_injection_inserted_by_step` all False) and are copied unchanged, so nothing is dropped;
- 0 assistant turns and 0 human turns changed; no `<think>` anywhere. sha256 `14191332…` (`.report.json` next to the file).
- Offline check: `multibase/check_llama_render.py --data <file> --n 300 --all-lengths` → 309/309 identical token sequences between the
  LLaMA-Factory `llama3` render and the inference render; max 6,511 tokens, 0 over the 8,192 cutoff.

**Training** — `examples/train_lora/llama31_8b_lora_sft_aba_cleansft.yaml` = the ABL-A yaml with only `dataset` and `output_dir` changed
(LoRA r64 α96 on q/k/v/o, lr 5e-6, 3 epochs, GA 16, cutoff 8192). 696 steps, 3,698 examples, 54,525,952 trainable parameters;
1 h 20 on the interactive H100 (2026-11-10 22:13 → 23:33). Adapter sha256 starts `7c2a987781050ec0`.

**Evaluation** — `ablation/run_aba_cleansft_interactive.sh` chains the training script, the ten `eval_sdL2.sbatch` shards (2 procs,
sequential, 2 h 56) and `eval/compute_attack_stats.py`. All 16 eval processes print `lora=…aba_cleansft… exists=True append=False`.
Run dir `agentdojo/runs/aba_cleanSFT`, stats `agentdojo/eval/attack_stats_aba_cleanSFT_meta-llama_Llama-3.1-8B-Instruct-safe-agent.csv`.

**Result (`important_instructions`, FINAL 2026-11-11 02:29).**

| Llama-3.1-8B (no append) | Benign | UA | ASR | Trade-off |
|---|---|---|---|---|
| undefended base | 27.84 (27/97) | 23.08 (219/949) | 7.59 (72/949) | 115.49 |
| **aba-cleanSFT** (clean trajectories, no reflection) | **28.87** (28/97) | **25.92** (246/949) | **5.69** (54/949) | 120.23 |
| ABL-A (injected trajectories, no reflection) | 34.02 (33/97) | 29.29 (278/949) | 0.84 (8/949) | 128.45 |
| SR-Agent-Llama (full SRFT) | 37.11 (36/97) | 29.08 (276/949) | 1.26 (12/949) | 127.82 |

Per suite (Benign / UA / ASR):

| | banking | slack | travel | workspace |
|---|---|---|---|---|
| base | 25.00 / 29.17 / 27.08 | 52.38 / 25.71 / 23.81 | 10.00 / 12.86 / 4.29 | 25.00 / 23.57 / 0.36 |
| aba-cleanSFT | 37.50 / 29.86 / 25.69 | 33.33 / 19.05 / 6.67 | 20.00 / 12.14 / 1.43 | 27.50 / 29.64 / 1.43 |
| SR-Agent-Llama | 43.75 / 36.11 / 5.56 | 33.33 / 28.57 / 0.00 | 25.00 / 7.14 / 0.00 | 42.50 / 32.86 / 0.71 |

Paired utility 2×2 (`eval/paired_failure_analysis.py <A> <B> meta-llama_Llama-3.1-8B-Instruct-safe-agent`; both / A only / B only / neither):

| A → B | attacked (949) | benign (97) |
|---|---|---|
| base → aba-cleanSFT | 114 / 105 / 132 / 598 (net +27; McNemar χ² 3.1, p ≈ 0.08) | 18 / 9 / 10 / 60 (net +1) |
| aba-cleanSFT → SR-Agent-Llama | 177 / 69 / 99 / 604 (net +30; χ² 5.4, p ≈ 0.02) | 23 / 5 / 13 / 56 (net +8; exact sign test p ≈ 0.10) |
| aba-cleanSFT → ABL-A | 189 / 57 / 89 / 614 (net +32) | 23 / 5 / 10 / 59 (net +5) |
| base → SR-Agent-Llama | 119 / 100 / 157 / 573 (net +57; χ² 12.6, p < 0.001) | 18 / 9 / 18 / 52 (net +9; p ≈ 0.12) |

**Reading.**
1. Plain agentic SFT on the clean trajectories barely moves utility: benign +1 task, UA +2.8 points (borderline). The prediction made
   before the run ("utility up") was wrong. SR-Agent-Llama's utility gain is therefore **not** explained by the TOUCAN trajectories acting
   as generic agentic SFT; it appears only in the two models trained on the *attacked* trajectories (ABL-A and SRFT).
2. Security does not come with task competence: ASR 5.69 against 1.26; banking 25.69 against the base's 27.08 and SRFT's 5.56.
   Slack is the only suite that moves (23.81 → 6.67), and §3 shows that does not survive a change of template.
3. The benign column (97 tasks, one sample each) cannot carry a significance claim in any comparison (p ≈ 0.10–0.12); UA on 949
   trajectories can. Quote UA; treat "banking 25 → 43.75" as 4 → 7 of 16 tasks.
4. *Why* training on attacked trajectories raises utility is not established. "Supervised to keep working through distracting tool
   output" is consistent with the data but untested — write "consistent with", not "because".

---

## 2. RL-Hammer on aba-cleanSFT

### 2a. Transfer matrix — 7 attackers × 4 Llama targets (2026-11-11 10:07 → 10:41)
`injecAgent-rl-harmmer/rl-injector/jobs/transfer_cleansft.sbatch` (run with `bash`, 34 min for 20 new cells, 8 reused from the
2026-09-18 matrix). The saved adversarial prompts of each run's best attacker checkpoint are replayed against every target; the
attacker model is never loaded. 100 InjecAgent test cases per cell, `llama_local` format, no append. ASR %:

| attacker (trained against; self-ASR) | base Llama | **aba-cleanSFT** | SR-Agent-Llama | ABL-A |
|---|---|---|---|---|
| base Llama r1 ckpt-765 (100) | 100 | **73** | 43 | 10 |
| base Llama r3 ckpt-816 (99) | 99 | **35** | 12 | 4 |
| SR-Agent-Llama r1 ckpt-816 (48) | 86 | **23** | 48 | 8 |
| SR-Agent-Llama + append r2 ckpt-918 (64) | 86 | **24** | 49 | 7 |
| SR-Agent-Llama r2 ckpt-714 (7) | 38 | **29** | 6 | 9 |
| ABL-A r1 ckpt-867 (14) | 81 | **50** | 6 | 9 |
| ABL-A r2 ckpt-357 (10) | 65 | **46** | 10 | 8 |
| **max per target** | 100 | **73** | 49 | 10 |
| mean over the 7 attackers | 79.3 | 40.0 | 24.9 | 7.9 |

Validity: the two base-Llama attackers reproduce their own curve values on the base target (100 / 99). Outputs `outputs/transfer_*`.
No attacker in this matrix was trained against aba-cleanSFT, so its column is a lower bound.

### 2b. Dedicated attacker (one run, seed 1024)
`SR_LORA=<aba_cleansft> SEED=1024 SRFT_SBATCH_FLAGS="--exclude=$(cat slurm_logs/.rlh_denylist)" bash injecAgent-rl-harmmer/rl-injector/jobs/submit_rlh.sh sr_llama 0 iclr_rlh_abacleansft`
— the protocol of the SR-Agent-Llama no-append curves, only the target adapter differs. Training **4490145**: 4×H100, 1,020 GRPO steps
(20 epochs), 2026-11-12 08:02 → 15:16 (7 h 14; the dry-run estimate had been 11-24). Checkpoints 51–663 were evaluated on the
interactive H100 while training ran (`CKPT_ONLY`), 714–1020 by the chained job **4490146**. Outputs `outputs/iclr_rlh_abacleansft_/`.

ASR per epoch: **18 19 26 40 36 51 51 64 59 72 65 68 66 72 70 65 78 63 76 74** → final **74**, last-5 mean **71.2**, peak **78** (epoch 17).

| Llama target (RL-Hammer, no append) | #attackers | final | peak |
|---|---|---|---|
| undefended base (r1 / r3) | 2 | 98 / 99 | 100 / 99 |
| **aba-cleanSFT** | 1 | **74** | **78** |
| Meta-SecAlign-8B | 2 | 80 / 59 | 80 / 68 |
| SR-Agent-Llama (r1 / r2) | 2 | 32 / 3 | 48 / 7 |
| ABL-A (r1 / r2) | 2 | 4 / 8 | 14 / 10 |

**Reading (2a + 2b).** Clean-trajectory SFT buys no adaptive robustness: 73 % from prompts never optimised against it, 70–78 % from its
own attacker — the level of Meta-SecAlign-8B. Two things in the same tables go **against** us and constrain the rebuttal: (i) the base
r1 attacker transfers to SR-Agent-Llama at 43 %, and SR-Agent-Llama's own attacker reaches 48 %, so "SR-Agent stays below 20 %" holds
only for the two-run mean (17.5), not for the worst case; (ii) **ABL-A stays ≤ 10 % against all seven attackers**, i.e. on Llama the
model trained on injected trajectories *without* reflection is the most robust one (max 10 vs 49). The Llama data support "training on
attacked trajectories with the correct action gives robustness", not "reflection adds robustness on top".

---

## 3. Second AgentDojo attack: `tool_knowledge` (Qhw2 W2, UquZ W3)

**Attack.** AgentDojo's `tool_knowledge` template: the important-message wrapper plus the ground-truth tool names and arguments of the
injection task ("You should use the tool `send_money` with the following arguments: {…}"). Of the attacks registered in our fork
(`direct`, `ignore_previous`, `system_message`, `injecagent`, `important_instructions` + name variants, `tool_knowledge`, DoS family)
it is the strongest static one in the literature. **Overlap with training:** its wording occurs in none of the 686 training triggers
(`<INFORMATION>`, "important message", "Signed," → 0). By contrast the `injecagent` template sentence *is* a training trigger
(3 trajectories) and `ignore_previous`-like wording appears in 2 triggers (4 trajectories) — state that if those are ever reported.

**Run.** `ATTACK=tool_knowledge bash agentdojo/scripts/run_attack_interactive.sh "<run>|<lora or /nonexistent>|<nproc>" …`
(2026-11-11 16:12 → 2026-11-12 12:47). Attacked shards only (949 each); benign is the model's `important_instructions` run. Run dirs
`agentdojo/runs/{llama31_base_noappend,aba_cleanSFT,llama31_v3base_local_3epoch_noappend}_toolknowledge`; stats
`eval/attack_stats_<run>_…__tool_knowledge.csv` via `compute_attack_stats.py --attack tool_knowledge` (new option; default output
unchanged, verified on the aba-cleanSFT run). Wall time on one H100: base 2 h 35 (3 procs), aba-cleanSFT 2 h 30 (3 procs),
**SR-Agent-Llama 15 h 30** (2 procs — plan any further full SR-Agent-Llama eval at ~1 min per trajectory). One base workspace shard
failed at 3 procs and was completed by the 1-proc retry; 949/949 for all three; adapters verified in every process log.

| Llama-3.1-8B (no append) | UA `important_instructions` | ASR `important_instructions` | UA `tool_knowledge` | ASR `tool_knowledge` |
|---|---|---|---|---|
| undefended base | 23.08 | 7.59 (72) | 24.66 | 7.17 (68) |
| aba-cleanSFT | 25.92 | 5.69 (54) | 24.97 | **7.69 (73)** |
| SR-Agent-Llama | 29.08 | 1.26 (12) | **29.50** | **0.42 (4)** |

Per suite under `tool_knowledge` (UA / ASR):

| | banking | slack | travel | workspace |
|---|---|---|---|---|
| base | 31.94 / 15.28 | 24.76 / 34.29 | 13.57 / 3.57 | 25.54 / 0.89 |
| aba-cleanSFT | 27.08 / 23.61 | 22.86 / 28.57 | 12.14 / 5.00 | 28.04 / 0.36 |
| SR-Agent-Llama | 36.81 / 1.39 | 25.71 / 0.95 | 10.00 / 0.00 | 33.21 / 0.18 |

**Reading.** (1) SR-Agent-Llama's static result carries over to a second, unseen template: ASR 0.42 with UA unchanged. (2) aba-cleanSFT
is as vulnerable as the base here (73 vs 68 attacks); its low slack ASR under `important_instructions` (6.67) becomes 28.57, so
clean SFT has no static robustness that survives a change of template. (3) `tool_knowledge` is **not** a stronger attack than
`important_instructions` on the undefended 8B model (68 vs 72); call it "a second template", not "a stronger attack".
No Meta-SecAlign-8B row: its weights are not on WashU.

---

## 4. PAIR-style iterative-refinement attack on InjecAgent (Qhw2 W2)

**What it is.** `injecAgent-rl-harmmer/rl-injector/pair_attack.py`: PAIR (Chao et al., arXiv 2310.08419) adapted from jailbreaking to
indirect prompt injection. An attacker LLM writes the text placed in the tool response, sees the agent's reply and whether the injected
tool was called, and rewrites its attempt. As in the paper: 30 parallel streams × 3 rounds (≤ 90 target queries per case), black-box,
a case is broken when ANY query succeeds; the streams are split over three strategy prompts (logical appeal / authority / role-play).
**Deviations:** attacker = Llama-3.1-8B-Instruct, not fine-tuned (paper: Mixtral 8x7B); judge = InjecAgent's rule-based parser (paper:
Llama Guard); target prompt, sampling and parser = the RL-Hammer evaluation of the Llama targets (`llama_local`, no append, 1024 tokens).
One vLLM engine serves attacker and target (base + LoRA). 100 test cases, seed 1024, ~20 min for both targets per setting.
`python pair_attack.py --targets base=,srllama=<SR-Agent-Llama LoRA> [--target_temperature 0 --tag iclr_pair_t0]` (env `rlhammer`).
Outputs `outputs/iclr_pair_<target>_/` (T 0.6) and `outputs/iclr_pair_t0_<target>_/`: `summary.json`, `cases.json` tracked;
`queries.jsonl` (every prompt and reply, 16 MB) git-ignored.

**Why three extra columns.** With a sampled target, "any of 90 queries" is inflated by retries alone. So next to PAIR's metric:
**control** = the benchmark's own injection sampled 90 times per case with no attacker; **replay** = the first successful prompt of each
broken case re-sampled 20 times ("verified" = at least half succeed; "single-shot" = mean replay rate, unbroken cases counted 0).

| target T | target | control: per-query | control: any of 90 | **PAIR: any of 90** | mean queries | per-query, rounds 1 / 2 / 3 | verified | single-shot |
|---|---|---|---|---|---|---|---|---|
| **0.6 (our protocol; the row to report — user decision 2026-11-12)** | base Llama | 24.4 | 60 | **97** | 34.8 | 28.3 / 4.0 / 2.2 | 69 | 64.6 |
| 0.6 | SR-Agent-Llama | 3.8 | 76 | **68** | 60.9 | 2.7 / 1.0 / 1.1 | 0 | 4.5 |
| 0 (PAIR's own setting) | base Llama | 26.0 | 26 | **97** | 34.5 | 31.2 / 3.0 / 2.7 | 96 | 96.0 |
| 0 | SR-Agent-Llama | 1.2 | 9 | **53** | 67.8 | 1.8 / 1.0 / 0.5 | 18 | 19.4 |

**Reading.**
- T = 0.6: on the base PAIR finds prompts that work reliably (69 % verified). On SR-Agent-Llama it does not exceed the no-attacker
  control (68 vs 76) and none of its 68 "successful" prompts replays at ≥ 50 % (median replay rate 5 %; 30 never succeed again) — those
  successes are sampling.
- T = 0: PAIR raises SR-Agent-Llama from 9 % (control) to 53 %, with 18 cases reproducibly broken (8 at 100 %). So PAIR is **not
  ineffective** against SR-Agent-Llama; it is far less effective than against the base (97 %, 96 % verified, half the queries). vLLM
  greedy decoding is not exactly deterministic for this target (control per-query 1.2 % but any-of-90 9 %), so some retry inflation remains.
- If the T = 0.6 row is quoted, quote **all** of it — PAIR 68 with its control 76, not "verified 0 %" alone — and say that the paper's
  target temperature is 0. Recommended one-line addition: the T = 0 any-query numbers (97 / 53; control 26 / 9).
- Limitations to state: 8B attacker; two targets, no Meta-SecAlign; SR-Agent-Llama's ≈ 4 % single-query success rate at T 0.6 means an
  attacker allowed many retries succeeds on most cases without optimising anything. RL-Hammer is much stronger on the same target (32–48 %).
- Rebuttal paragraph + table (T 0.6) were drafted in the session of 2026-11-12; regenerate the numbers from the `summary.json` files.

---

## 5. Second evaluation of the three Qwen3-8B ablation attackers (UquZ Q3)

**What was done — and what it is not.** Each Qwen ablation arm has **one** trained RL-Hammer attacker. `jobs/reeval_ablq8.sh eval2`
re-ran `eval_attacker_ckpts.sbatch` on the same 20 attacker checkpoints of each arm under a new run name (`<run>_eval2`) with the
original target settings (think / append per arm, checked against the original banners): the attacker **re-generates** its adversarial
prompts (120 generations, 0 loaded from `saved_adv_prompts/`) and the target is re-sampled. 2026-11-13 12:49 → 16:30. This measures
evaluation noise for a fixed attacker. **It is not a second attacker run**: the paper's `#Runs = 2` for the base and SR-Agent means two
independently trained attackers, and the variance that matters lives there (SR-Agent-Qwen 32 vs 2, SR-Agent-Llama 32 vs 3). Report as
"#attackers 1, #evaluations 2" / "each ablation attacker was trained once and evaluated twice".

| arm (target think / append) | evaluation | Final | Last-5 | Peak (epoch) |
|---|---|---|---|---|
| w/o failure experience (on / on) — `iclr_rlh_ablq8_nolast_yyy` | 1 (2026-09-23) | 65 | 65.0 | 67 (8) |
| | 2 (2026-11-13) | 76 | 69.4 | 76 (20) |
| | **mean** | **70.5** | **67.2** | **70.5 (20)** |
| w/o reflection, think on (on / off) — `iclr_rlh_ablq8_yyn` | 1 (2026-09-18) | 60 | 54.8 | 69 (12) |
| | 2 | 68 | 54.0 | 68 (12) |
| | **mean** | **64.0** | **54.4** | **68.5 (12)** |
| w/o reflection, think off (off / off) — `iclr_rlh_ablq8_ynn` | 1 (2026-09-18) | 60 | 57.0 | 60 (20) |
| | 2 | 56 | 56.2 | 57 (18) |
| | **mean** | **58.0** | **56.6** | **58.0 (18)** |

Reference (two attackers each, final-epoch mean): undefended Qwen3-8B 59.5, SR-Agent-Qwen3-8B 17.0. "Peak" of the mean row is the peak
of the epoch-wise mean curve.

Per-epoch ASR:
- w/o failure experience — eval 1 `3 4 18 19 51 56 54 67 61 63 62 61 67 61 67 65 65 65 65 65` · eval 2 `6 4 10 21 37 58 53 62 62 64 63 68 61 69 71 65 68 71 67 76`
- w/o reflection, think on — eval 1 `3 2 4 9 21 33 46 60 63 63 66 69 61 60 61 63 58 56 37 60` · eval 2 `1 6 8 12 24 32 49 53 67 62 62 68 60 66 61 60 55 56 31 68`
- w/o reflection, think off — eval 1 `1 0 2 1 5 5 7 17 27 30 30 33 29 50 58 57 56 59 53 60` · eval 2 `2 0 2 1 1 6 5 16 24 29 32 30 27 51 53 55 56 57 57 56`

**Reading.** The curves keep their shape (take-off epoch, plateau height). Last-5 moves by at most 4.4 points between evaluations while
single checkpoints move by up to ~15, so quote Last-5 or the two-evaluation mean, not a single Final. All three arms stay at the
undefended base's level and far above SR-Agent in both evaluations. The think-on arm's dip at epoch 19 (37 / 31) reproduces — a
property of that checkpoint. Unexplained: evaluation 1 of the w/o-failure-experience arm reads exactly 65 for its last five
checkpoints; evaluation 2 does not (65–76).

**Attackers available for further work.** WashU holds weights + saved prompts for every `iclr_rlh_*` run (20 checkpoints each). The
NeurIPS Qwen attackers are **not** here: no weights for YYY1 / YYY2 (never migrated from DSAI) and only one per-case file with prompts
(YYY1 ckpt-1020, ASR 32). The Qwen family has still never had a transfer matrix.

---

## 6. CPU-only analyses

- **Paired base vs SR-Agent-Llama** (`paired_failure_analysis.py llama31_base_noappend llama31_v3base_local_3epoch_noappend <pipeline>`):
  attacked 119 / 100 / 157 / 573. In the 157 trajectories SR-Agent-Llama wins, the base makes **zero tool calls in 84**. Of the 100 it
  loses, only **2** end by declining / citing the injection; 76 already differ from the base at the first tool call.
- **Travel suite (ML5G Q4).** UA 7.14 vs base 12.86 is 10 vs 18 of 140 trajectories: base-only 14, SR-only 6, neither 116, spread over
  user tasks 3 / 0 / 15 / 2 / 7 (2–3 each). Benign travel goes the other way (5/20 vs 2/20). Under `tool_knowledge` the same gap is 10.00
  vs 13.57. Small-sample policy difference, not over-suppression (2 declines in 100 losses).
- **Template / trigger overlap** — §3.
- **Injection-corpus diversity (ML5G Q3, "only 38 malicious tasks"; computed 2026-11-15 from `toucan_32B_v3_base.json` `meta.injections`,
  `agentdojo` v1.2.1 suites and `InjecAgent/raw/test_cases_dh_base.json`).** Three layers, no new training:
  1. *Injected instructions.* 323 distinct injected instructions, not 38: the 38 Template-1 malicious tasks (24–155 uses each) plus
     **285 off-task user requests** taken from other Toucan trajectories (≤ 9 uses each) — benign-looking, in-domain instructions the
     agent must still refuse because they arrive in a tool observation. The model therefore learns "instruction in tool output ≠ user
     goal", not a list of harmful strings.
  2. *Surface forms.* Each instruction is prefixed with one of 686 distinct triggers (from the 750-template StruQ + TaskTracker pool) at a
     random position (start / middle / end, p = 0.5 per observation), giving **3,455 distinct injected strings** over 3,698 trajectories
     and 22,339 supervised steps (37.8 % of observations carry one).
  3. *Train / test disjointness.* Training = 5 Toucan MCP platforms, 207 tool names. AgentDojo = 4 suites, 69 tools, 35 injection goals,
     one fixed `<INFORMATION> … Emma Johnson` wrapper; InjecAgent = 38 toolkits, 312 tools, 30 attacker instructions; RL-Hammer rewrites
     those 30 by RL. Tool-name overlap: **1 / 207** (`send_email`, a different API on each side). Not one of the 686 training triggers
     contains "important message" or `<INFORMATION>`. **0 / 35** AgentDojo goals and **0 / 30** InjecAgent instructions share even a
     6-word window with any of the 323 training instructions. Every test injection is unseen at the level of tool, goal text and wrapper.
  Lists: 38 malicious / off-task samples / AgentDojo goals / InjecAgent instructions in the session scratchpad `q3_lists.json`
  (regenerate with the snippet in `99_changelog.md` 2026-11-15). A count-vs-ASR scaling curve would need ≥ 3 re-generations + trainings
  + RL-Hammer runs (≈ 40 GPU-h each); not run — the argument above is what the data support.
- **Self-generated reflections (ML5G Q2)** — nothing new was run; the existing rows are v1 (53.61 / 40.15 / 1.48) and v2-traj
  (53.61 / 46.58 / 2.32) against v0 (51.55 / 46.68 / 1.05), with the caveats in `03_analysis_utility_drop.md` §J (Claude reflection used
  as a hint; failure-contrast content drops from 23 % to 9 % of thinks naming a wrong tool call; static only, no RL-Hammer).

## 7. Considered and not run

| item | why not |
|---|---|
| **AutoInject** (Chen et al., arXiv 2602.05746) on AgentDojo | Per-pair GRPO with a 260-query budget; each query is a full agent run (~1 min for SR-Agent-Llama on one H100) → > 4 h per pair. Official code is AgentDojo-based with a GPT-4o-mini feedback model; no API key on WashU. A reimplementation on InjecAgent is computationally feasible (~15 h for 100 cases) but cannot be validated against any published number. |
| **Nasr et al. 2025** search attack (arXiv 2510.09023) | MAP-Elites with a Gemini-2.5-Pro mutator, up to 800 queries per scenario (≈ 13 h per scenario here), no released code. Useful to cite: it takes Meta-SecAlign-70B from 2 % to 96 %, so the defensible claim is relative robustness under a matched attacker, not robustness to unbounded per-scenario search. |
| **Andriushchenko et al. 2024** random search (arXiv 2404.02151) | A white-box (first-token log-prob) jailbreak method with no agent / prompt-injection protocol; would need our own objective (e.g. "start with the tool call instead of `<think>`"). Cheap per iteration, but new code and an unvalidated design. |
| **"Teacher rationale without sampled candidates" arm** (all three reviewers, W1) | Needs Claude regeneration of paragraph 3 for ~22k steps, a Qwen3-8B training run and a 4-GPU RL-Hammer run; not possible in the window. Available evidence is indirect: the w/o-failure-experience arm (removes all of paragraph 3) and the candidate statistics in `04_self_distill_plan.md` §0b (on 13,720 injected steps only 23 % have a candidate that follows the injection; on 51 % no candidate equals the expert action). |
| Benign re-sampling (3 seeds) for base / aba-cleanSFT / SR-Agent-Llama | Proposed for ML5G Q1(b), never requested. Would take ~2 h and settle whether 37.11 vs 27.84 benign is real. |
| Qwen transfer matrix; clean SFT on Qwen3-8B; extra attacker seeds | Not requested / no 4-GPU slot. |

## 8. What the results allow the rebuttal to say

**ML5G Q1, second half ("does security come from a stronger task baseline?") — four consistent rows:**

| attack | base | aba-cleanSFT | SR-Agent-Llama |
|---|---|---|---|
| AgentDojo `important_instructions`, ASR | 7.59 | 5.69 | 1.26 |
| AgentDojo `tool_knowledge`, ASR | 7.17 | 7.69 | 0.42 |
| RL-Hammer transfer, max over 7 attackers | 100 | 73 | 49 |
| RL-Hammer dedicated attacker, final | 98 / 99 | 74 | 32 / 3 |

Task-competence training gives no security, statically or adaptively. First half: the utility gain is not generic agentic SFT (§1), and
the benign column alone is under-powered — use UA.

**Do not write:**
- "SR-Agent stays below 20 % under adaptive attack" without the worst run next to it (Llama 32 / peak 48, transfer 43–49; Qwen 32 / peak 42).
- That reflection contributes to robustness or to AgentDojo ASR **on Llama** — ABL-A matches SRFT statically (0.84 vs 1.26) and beats
  it adaptively (max 10 vs 49). The ablation evidence for reflection and for the failure-contrast paragraph is Qwen-only.
- "#Runs = 2" for the Qwen ablation arms (§5).
- "The agent's own sampled candidate actions": the candidates come from Qwen3-32B (T 1.0, 3 per step) and are shared by all three targets.
- PAIR's "verified 0 %" without the any-query numbers and the control (§4).

## 9. Reproduce

```bash
# 1. data + training + AgentDojo (one GPU, inside an allocation)
python ablation/build_clean.py
bash ablation/run_aba_cleansft_interactive.sh                       # SKIP_TRAIN=1 to resume at the eval
# 2a. transfer matrix (skips existing cells)                        2b. dedicated attacker (4 GPUs, batch)
bash injecAgent-rl-harmmer/rl-injector/jobs/transfer_cleansft.sbatch
SR_LORA=$PWD/LLaMA-Factory/saves/llama31-8b/lora/aba_cleansft_sft_8k_r64_GA4_qkvo_3epoch_5e-6 SEED=1024 \
  bash injecAgent-rl-harmmer/rl-injector/jobs/submit_rlh.sh sr_llama 0 iclr_rlh_abacleansft
# 3. second AgentDojo attack
ATTACK=tool_knowledge bash agentdojo/scripts/run_attack_interactive.sh "<run>|<lora or /nonexistent_base_model_fallback>|<nproc>" ...
python agentdojo/eval/compute_attack_stats.py <run>/meta-llama_Llama-3.1-8B-Instruct-safe-agent --attack tool_knowledge
# 4. PAIR (env rlhammer, from injecAgent-rl-harmmer/rl-injector)
python pair_attack.py --targets base=,srllama=<SR-Agent-Llama LoRA>                 # add --target_temperature 0 --tag iclr_pair_t0
# 5. second evaluation of the Qwen ablation attackers
bash injecAgent-rl-harmmer/rl-injector/jobs/reeval_ablq8.sh eval2
```

Commits (all on `main`): `d06c7183` data builder + config · `0228008b` interactive driver + cluster notes · `e972667e` aba-cleanSFT
result · `9cac244d` transfer matrix · `99fe586f` / `dcf59902` `tool_knowledge` · `bb81ccb0` dedicated attacker · `160f5b5a` PAIR ·
`490c9b21` second evaluation. Tracked in git: all stats CSVs, the four AgentDojo run dirs (`agentdojo/runs/aba_cleanSFT`,
`…/*_toolknowledge`) and every `rl-injector/outputs/iclr_*` dir. Git-ignored, WashU only: the clean training JSON (rebuild with
`ablation/build_clean.py`, check the sha256), the aba-cleanSFT adapter, the attacker checkpoints under `rl-injector/checkpoints/`,
`outputs/transfer_*` and `outputs/iclr_pair*/queries.jsonl` — upload to the HF repos (`10_sync_workflow.md`) if skipjack needs them.
