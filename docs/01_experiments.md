# Experiment ledger (ICLR resubmission)

One row per version in the summary table; one section per version below with the same fixed fields.
Rules: add the row when the experiment is *planned*, fill numbers when the stats CSV exists, never overwrite a version — make a new one.
Metrics = AgentDojo v1.2.1, attack `important_instructions`, `eval/compute_attack_stats.py` (Benign Utility / Utility under Attack / ASR, %).

## Summary

| ver | one-line idea | think transcription scheme | data | ckpt (LoRA) | eval run dir | Benign ↑ | UA ↑ | ASR ↓ | status |
|---|---|---|---|---|---|---|---|---|---|
| v0 (paper) | Claude-written 3-paragraph reflections, SFT | **Claude** (off-policy, 3-paragraph template) | `LLaMA-Factory/data/toucan_32B_v2.json` | `saves/qwen3-8b/lora/toucan_32B_v2_sft_8k_r64_GA4_qkvo_3epoch_5e-6` | `agentdojo/runs/3_01_toucan_32B_v2_sft_8k_r64_GA4_qkvo_3epoch_5e-6` (+ old no-attack csv) | 51.55 | 46.68 | 1.05 | done (NeurIPS Table 1) |
| v0-base | Qwen3-8B, no fine-tune, think on | — (native Qwen think) | — | — | old server (csv only: `eval/attack_stats_Qwen_Qwen3-8B_baseline.csv`) | 60.82 | 50.47 | 16.97 (paper says 17.91) | done |
| v1 sdL2 | same data, think traces re-written by Qwen3-8B itself (rung L2 = Claude reflection as hint), everything else = v0 | **L2 reason-it-yourself**: Qwen3-8B thinks from the inference prefix, Claude reflection only as a hint, action must match expert (content degrades, 03 §J) | `LLaMA-Factory/data/toucan_32B_v2_sdL2.json` | `saves/qwen3-8b/lora/sdL2_sft_8k_r64_GA4_qkvo_3epoch_5e-6` | `agentdojo/runs/sdL2_3epoch` | 53.61 | 40.15 | 1.48 | done 2026-09-06 (trade-off 138.67 vs v0 145.63) |
| v1b sdL2-perstep | v1 data exploded per assistant step, think-free history, loss on target turn only (`mask_history: true`) → training context == inference context | L2 reason-it-yourself (same thinks as v1) | `LLaMA-Factory/data/toucan_32B_v2_sdL2_perstep.json` | `saves/qwen3-8b/lora/sdL2_perstep_mh_sft_8k_r64_GA4_qkvo_3epoch_5e-6` | `agentdojo/runs/sdL2_perstep_3epoch` | 34.02 | 36.78 | 0.63 | done 2026-09-07 (trade-off 136.15) |
| v2 sdL2fa-perstep | v1's Qwen thinks + EXPERT actions/answers everywhere (assembler bug fixed), per-step format like v1b, empty-reply steps excluded as targets but kept in context | L2 reason-it-yourself (same thinks as v1) | `LLaMA-Factory/data/toucan_32B_v2_sdL2fa_perstep.json` | `saves/qwen3-8b/lora/sdL2fa_perstep_mh_sft_8k_r64_GA4_qkvo_3epoch_5e-6` | `agentdojo/runs/sdL2fa_perstep_3epoch` | 47.42 | 43.41 | 1.05 | done 2026-09-08 (trade-off 142.36) |
| v2-e1 | v2's epoch-1 checkpoint (1,205 of 3,615 steps) — over-training test | L2 reason-it-yourself (same thinks as v1) | same as v2 | `…/sdL2fa_perstep_mh_sft_8k_r64_GA4_qkvo_3epoch_5e-6/checkpoint-1205` | `agentdojo/runs/sdL2fa_perstep_1epoch` | 47.42 | 41.20 | 1.37 | done 2026-09-08 23:xx (trade-off 139.83); 110 workspace traj re-run after L40S OOM |
| v2-trajectory | sdL2fa data (Qwen thinks + expert answers) trained with the exact v1/v0 recipe: multi-turn, `mask_history: false` — the missing cell of the 2×2 (think source × format) | L2 reason-it-yourself (same thinks as v1) | `LLaMA-Factory/data/toucan_32B_v2_sdL2fa.json` | `saves/qwen3-8b/lora/sdL2fa_traj_sft_8k_r64_GA4_qkvo_3epoch_5e-6` | `agentdojo/runs/sdL2fa_traj_3epoch` | 53.61 | 46.58 | 2.32 | done 2026-09-09 01:49 (trade-off 144.26 vs v0 145.63) |
| v0' v3base-traj | paper recipe (v0 yaml, multi-turn, mask_history false) on RECOVERED data: 3,185 mid-turn replies filled from per-step source records, injection ground truth in `meta` | Claude (as v0) | `LLaMA-Factory/data/toucan_32B_v3_base.json` | `saves/qwen3-8b/lora/v3base_traj_sft_8k_r64_GA4_qkvo_3epoch_5e-6` | `agentdojo/runs/v3base_traj_3epoch` | – | – | – | data DONE; training CANCELLED by user 2026-09-09 (skip to v3-paraphrase) |
| v3-paraphrase | Claude thinks paraphrased by Qwen3-8B in first person WITH the step context (thinking off), content-preserving filters (04_self_distill_plan v3 recipe); SOURCE = recovered `toucan_32B_v3_base` (decided 2026-09-09), trained with the `qwen3_8b_lora_sft_v3base_traj.yaml` recipe (paper yaml, dataset/output_dir only) | **Paraphrase**: Qwen3-8B rewrites the Claude reflection in first person with the step context (thinking off); content guaranteed by Claude + filters, tokens Qwen | `LLaMA-Factory/data/toucan_32B_v3_para.json` | `saves/qwen3-8b/lora/v3para_traj_sft_8k_r64_GA4_qkvo_3epoch_5e-6` | `agentdojo/runs/v3para_traj_3epoch` | – | – | – | data DONE 2026-09-09 04:33 (98.8 % paraphrased); training 347215 RUNNING on gl106 (L40S) since 15:18, ETA ≈ 23:00; eval-submitter 347216 chained |

Per-suite numbers for each version are in its section. Trade-off = UA + (100 − ASR).
"Think transcription scheme" = who produced the `<think>` targets and how; the two Qwen schemes differ in the TASK given to the 8B, see the table at the top of `04_self_distill_plan.md`.

---

## v0 — paper model (NeurIPS 2026 submission)
| field | value |
|---|---|
| idea | SRFT: inject attacks into TOUCAN expert trajectories, sample Qwen3-32B candidate actions, Claude Sonnet 4.6 writes 3-paragraph self-reflection, SFT Qwen3-8B on reflection+action |
| transcription / data recipe | `docs/02_paper_summary.md` §Four stages; templates Fig. 3/4 of the PDF |
| data | `LLaMA-Factory/data/toucan_32B_v2.json` — 3707 traj / 22,456 steps; think mean 292 words; 100 % Claude-written |
| training | `LLaMA-Factory/examples/train_lora/qwen3_8b_lora_sft_think.yaml` (dataset must be set back to toucan_32B_v2): LoRA r64 α96 q/k/v/o, lr 5e-6, bs1×GA4×4 GPU, 3 ep, cutoff 8192, mask_history false; 696 steps, 2 h 02 on 4×A100; final train_loss 0.978 |
| ckpt | `LLaMA-Factory/saves/qwen3-8b/lora/toucan_32B_v2_sft_8k_r64_GA4_qkvo_3epoch_5e-6` (61.3 M trainable params — paper's "0.17B" is wrong) |
| inference | HF two-phase generate, think budget 512 + forced transition, `_SYS_PROMPT_APPEND` on, temp 0.6 / top-p 0.95 / top-k 20 |
| eval | attacked: `agentdojo/runs/3_01_…/Qwen_Qwen3-8B-safe-agent` → `eval/attack_stats_3_01_toucan_32B_v2_sft_8k_r64_GA4_qkvo_3epoch_5e-6_Qwen_Qwen3-8B-safe-agent.csv`; benign: `eval/attack_stats_qwen3_8b_toucan_lora_think_sys_append_no_attack_Qwen_Qwen3-8B-safe-agent.csv` (run dir not migrated) |
| results | banking 56.25 / 29.17 / 4.86 · slack 66.67 / 46.67 / 1.90 · travel 30.00 / 22.14 / 0.00 · workspace 52.50 / 57.32 / 0.17 · **ALL 51.55 / 46.68 / 1.05** (trade-off 145.63) |
| diagnosis | `docs/03_analysis_utility_drop.md`: off-policy Claude CoT + unconditional template → step-0 hallucinated "tool response" (73 %), style split, UA decays with epochs; ≈4 benign points are the system-prompt append |

## v1 — sdL2: self-distilled think traces (Qwen3-8B rewrites, hint rung L2)
| field | value |
|---|---|
| hypothesis | keeping the reflection *content* but letting Qwen3-8B write it in its own distribution recovers utility without losing ASR |
| transcription recipe | `docs/04_self_distill_plan.md` (per-step prompts, hint ladder L0/L1/L2, filters, smoke + full-run results); code `self_distill/` (`common.py`, `generate_vllm.py`, `assemble_trajectories.py`) |
| generation | vLLM 0.9.2, Qwen/Qwen3-8B, rung **L2 only**, n=4, temp 0.6/top-p 0.95/top-k 20, ≤1024 tok; 8 shards × 1 GPU (L40S 41 min, H100 13 min, H200 10 min each); raw attempts `self_distill/runs/sdL2/shard*.jsonl`; reading sample `self_distill/runs/sdL2/sample_trajectories.md` |
| data | `LLaMA-Factory/data/toucan_32B_v2_sdL2.json` (registered `toucan_32B_v2_sdL2`) — same 3707 traj / 22,456 steps; 20,030 steps rewritten (89.2 %: step0 79.8 / clean 91.1 / injected 91.1), 2,426 keep Claude think (in 1,596 traj); Qwen think mean 171 words; stats `…_sdL2.stats.json` |
| known data caveats | injection labels heuristic (from Claude ¶2); no judge for final-answer steps; ~10 % Claude fallback steps |
| training | `examples/train_lora/qwen3_8b_lora_sft_sdL2.yaml` = v0 yaml with only dataset + output_dir changed (mask_history **false**); launcher `scripts/train_qwen3_8b_sdL2_sft.slurm`; job 313882, 2×H200, GA 8 (eff. batch 16), 696 steps, 84 min; loss 1.539 → 0.721 (final train_loss 0.804; v0 at same steps 0.861/0.978) |
| ckpt | `LLaMA-Factory/saves/qwen3-8b/lora/sdL2_sft_8k_r64_GA4_qkvo_3epoch_5e-6` (final = epoch 3; `checkpoint-232` / `-464` = epoch 1 / 2) |
| inference | identical to v0 (HF two-phase, SYS_APPEND=1, think on) |
| eval | `agentdojo/runs/sdL2_3epoch/Qwen_Qwen3-8B-safe-agent` (attacked + benign in one dir); jobs 315291/315292/315293/315368–315371, stats job 315372 → `eval/attack_stats_sdL2_3epoch_Qwen_Qwen3-8B-safe-agent.csv`; recipe `docs/06_eval_agentdojo.md` §4 |
| results | banking 62.50 / 33.33 / 4.86 (128.47) · slack 61.90 / 35.24 / 0.95 (134.29) · travel 40.00 / 23.57 / 0.00 (123.57) · workspace 52.50 / 46.96 / 1.07 (145.89) · **ALL 53.61 / 40.15 / 1.48** (trade-off 138.67). CSV `eval/attack_stats_sdL2_3epoch_Qwen_Qwen3-8B-safe-agent.csv`; eval jobs 315291/315292/315293/315736–315742, stats 315743; wall time 21:15 → 09:38 (queue-bound) |
| vs v0 | Benign **+2.06**, UA **−6.53**, ASR +0.43 (10 → 14 successful attacks / 949). Benign up on banking (+6.25) and travel (+10.0); UA down on slack (−11.4) and workspace (−10.4) |
| analysis | `docs/03_analysis_utility_drop.md` §F. Think channel fixed (step-0 hallucinated tool-response 73 % → 23 %, openers 99 % native, forced transitions 17 % → 12 %, benign +2). UA loss = **answer-channel contamination in sdL2 data** (implementation bug: assembler used Qwen's final-answer text instead of the expert's, contrary to the design; tool calls were correct): 5,961 expert answers replaced/filled and 22 % of them start with reflection text, 83–94 % mention injection (v0 data: 0 % / 0.5 %). At inference 68 % of final answers open reflection-style (v0 4 %); failures with injection talk 219 → 463 while silent wrong execution fell 262 → 88 |
| next / variants queued | **v2 sdL2fa** (proposed, data-only): same Qwen thinks + EXPERT final answers, drop originally-empty final steps, re-assemble from existing shard jsonl, train with v1 recipe; v1b (running): same data, per-step/`mask_history: true`; v1c: epoch-1/2 ckpts eval; later: L0-first ladder + clean replay, ground-truth injection labels once source runs are migrated |

## v1b — sdL2-perstep: inference-consistent contexts (per-step samples, think-free history)
| field | value |
|---|---|
| hypothesis | v1 still trains with Claude/Qwen thinks of *previous* turns in context (LLaMA-Factory multi-turn, mask_history=false) while inference shows history turns without think; removing this mismatch should help utility |
| caveat | uses the same sdL2 data as v1, i.e. inherits the contaminated final-answer targets (§F.2); expected to keep the UA deficit — it isolates the context-mismatch effect only |
| data | `self_distill/explode_per_step.py` → `LLaMA-Factory/data/toucan_32B_v2_sdL2_perstep.json` (registered `toucan_32B_v2_sdL2_perstep`): 22,002 samples (22,456 steps minus 454 empty-final steps); history assistant turns think-free; mean 1,753 tokens/sample |
| framework patch | `LLaMA-Factory/src/llamafactory/data/template.py` `encode_multiturn`: with env `LF_NO_HISTORY_EMPTY_COT=1` no empty `<think></think>` is inserted for history turns (default LLaMA-Factory behaviour differs from the official Qwen3 template / our inference). Verified render == inference render |
| training | `examples/train_lora/qwen3_8b_lora_sft_sdL2_perstep.yaml` = v1 yaml + dataset + `mask_history: true`; same LoRA/lr/epochs; 2 GPUs GA 8 (eff. batch 16) → ~4.1k steps |
| ckpt | `LLaMA-Factory/saves/qwen3-8b/lora/sdL2_perstep_mh_sft_8k_r64_GA4_qkvo_3epoch_5e-6` |
| eval | same recipe as v1 (`scripts/submit_eval_sdL2.sh sdL2_perstep_3epoch <ckpt>`), inference path unchanged |
| training result | job 315868, 2×H100 (gh127), 4,128 steps, 4 h 12 min, final train_loss 0.641 (last-turn-only loss, not comparable to v1's 0.804) |
| results | banking 37.50 / 26.39 / 1.39 · slack 28.57 / 30.48 / 0.00 · travel 25.00 / 26.43 / 0.71 · workspace 40.00 / 43.21 / 0.54 · **ALL 34.02 / 36.78 / 0.63** (trade-off 136.15). vs v1: Benign **−19.6**, UA −3.4, ASR −0.8 (6 successful attacks / 949, lowest so far). CSV `eval/attack_stats_sdL2_perstep_3epoch_Qwen_Qwen3-8B-safe-agent.csv`; jobs 318187–318196 ran 2026-09-06 20:44 → 09-07 04:09 (mix of A100/H100/L40S), stats 318197 |
| analysis | `docs/03_analysis_utility_drop.md` §G. Same contaminated answer targets as v1, but per-step training puts the loss on the target turn only and takes 4,128 optimizer steps (v1: 696), so the reflection-style answer is learned far harder: at inference 72 % of final answers open reflection-style, **97 % mention injection**, benign failures 60/64 are injection-talk. Tool use itself is near-perfect (silent wrong execution 12, v0 262). ⇒ v1→v1b is NOT a clean read of "context consistency": the format change interacts with the contamination and with 6× more steps |

## v2 — sdL2fa-perstep: expert answers restored, per-step training
| field | value |
|---|---|
| hypothesis | v1's UA drop came from Qwen-written final answers (§F.2). With the design rule restored (only the think is replaced; every tool call AND final answer is the expert's), UA should return to ≈ v0 (46.7) while keeping v1's think-channel gains and benign +2 |
| what it isolates | v1 → v1b = per-step/context-consistent format; v1b → v2 = answer fix. The 3,185 empty mid-conversation replies are excluded as targets (they were targets in v0/v1; present identically in all versions' contexts) |
| data | `self_distill/assemble_trajectories.py` (fixed: `body = rest` for all steps) → `toucan_32B_v2_sdL2fa.json` (same 20,030 rewritten thinks as v1; 0 action/answer differences vs expert, verified) → `explode_per_step.py` → `toucan_32B_v2_sdL2fa_perstep.json`: **19,271 samples** = 15,564 tool-call + 3,707 final-answer steps, 0 empty targets. Both registered in `dataset_info.json` |
| training | `examples/train_lora/qwen3_8b_lora_sft_sdL2fa_perstep.yaml` = v1b yaml with dataset/output changed; same launcher, 2 GPUs, GA 8, `LF_NO_HISTORY_EMPTY_COT=1`; job 318170 |
| ckpt | `LLaMA-Factory/saves/qwen3-8b/lora/sdL2fa_perstep_mh_sft_8k_r64_GA4_qkvo_3epoch_5e-6` |
| eval | `bash agentdojo/scripts/submit_eval_sdL2.sh sdL2fa_perstep_3epoch <ckpt>` after training |
| results (final) | Benign / UA / ASR: banking 37.50 / 29.86 / 2.08 · slack 47.62 / 32.38 / 2.86 · travel 35.00 / 35.71 / 0.00 · workspace 57.50 / 50.89 / 0.71 · **ALL 47.42 / 43.41 / 1.05** (trade-off 142.36). vs v0 51.55 / 46.68 / 1.05: benign −4.1, UA −3.3, ASR equal. vs v1 53.61 / 40.15 / 1.48: benign −6.2, UA +3.3. CSV `eval/attack_stats_sdL2fa_perstep_3epoch_Qwen_Qwen3-8B-safe-agent.csv` (computed locally; stats job 329608 was stuck on InvalidQOS after the account QOS changed) |
| analysis | `docs/03_analysis_utility_drop.md` §H. Answer channel fixed (0 % reflection-style, 9 % injection-talk), ASR back to 1.05. Remaining UA gap to v0 (−3.3) = tool-call looping (138 trajectories with ≥3 identical consecutive calls vs v0 59; 47 hit the 15-iteration cap with no answer; same loop rate on benign runs ⇒ not attack-triggered) + step-0 hallucination back to 51 % (per-step weighting of the 20 % Claude-fallback step-0 targets). Travel +13.6, slack −14.3, workspace −6.4 |
| training result | job 318170, 2×L40S (gl105), 3,615 steps, 6 h 15, train_loss 0.774; loss by epoch 0.684 / 0.602 / 0.648 (rises in epoch 3) |
| watch-out | shares v1b's per-step format and ~5× the optimizer steps of v0/v1; if UA is still low, evaluate `checkpoint-1205` (epoch 1) before blaming the format |
| decision rule (user, 2026-09-06) | if v2 UA is still ≈ 40 (far below Qwen baseline 50.5), the cause is not the missing-reply defect (v0 shared it) but the rewritten think content or format → then v3 = recover full multi-turn data from the old server |

## v2-e1 — v2 at epoch 1 (over-training test)
| field | value |
|---|---|
| hypothesis | v2's loops (14.5 % of trajectories) and step-0 regression (51 %) grow with the 5× optimizer steps of per-step training; epoch 1 (1,205 steps ≈ 1.7× v0's 696) should show fewer loops and higher UA if so |
| ckpt | `checkpoint-1205` inside the v2 save dir (tokenizer files copied in from the final dir so the eval loader can read it; adapter untouched) |
| eval | `submit_eval_sdL2.sh sdL2fa_perstep_1epoch <ckpt>` → `agentdojo/runs/sdL2fa_perstep_1epoch`, stats CSV `eval/attack_stats_sdL2fa_perstep_1epoch_Qwen_Qwen3-8B-safe-agent.csv` |
| results (final, 949) | Benign 47.42 · UA **41.20** · ASR 1.37 (trade-off 139.83). Partial-839 numbers were 44.34 / 1.31; the 110 re-run workspace trajectories scored far lower (see note below). CSV `eval/attack_stats_sdL2fa_perstep_1epoch_Qwen_Qwen3-8B-safe-agent.csv` |
| note | the 3 OOM'd workspace shards were resubmitted assuming resume; the result loader had a path bug (fixed 2026-09-09, see 06_eval), so they RE-SAMPLED 433 workspace trajectories: workspace UA 52.0 (first sample, 450) → 43.2 (second sample, 433) for the identical checkpoint. The final 41.20 is a valid single sample, but this ~9-point swing on one suite is the run-to-run noise floor for every number in this table |
| verdict | **Not over-training.** Epoch 1 vs epoch 3 on the same 839 trajectories: loop trajectories 14.3 % vs 14.2 %, hit-the-cap 27 vs 43, tool calls/traj 4.33 vs 4.39, step-0 fake tool response 50 % vs 51 %, UA 44.3 vs 44.9. Everything that separates v2 from v0 is already there after 1,205 steps. → the residual comes from the per-step data/format (Claude-fallback step-0 targets, no anti-loop signal), not from step count. Analysis §I |

## v2-trajectory — clean answers, multi-turn (v1 recipe on sdL2fa)
| field | value |
|---|---|
| question it answers | does the L2 'reason-it-yourself' transcription (Qwen tokens, degraded content — see 04 'Two transcription schemes') keep utility under the v0 recipe? Only the think differs from v0 (same data structure, same recipe, expert actions and answers everywhere). It does NOT test the paraphrase scheme (Qwen tokens + Claude content) — that is v3. |
| reading | UA ≈ 46+ and benign ≈ 53 ⇒ thinks are fine, all earlier losses were the answer bug (v1/v1b) and the per-step format (v1b/v2). UA still ≈ 43 ⇒ the L2-hinted thinks lack the decide-when-to-stop content; regenerate with a different hint. |
| data | `toucan_32B_v2_sdL2fa.json` (3707 multi-turn trajectories, 20,030 Qwen thinks, 2,426 Claude fallbacks, all actions/answers expert; 3,185 empty mid-turn replies kept as in v0) |
| training | `examples/train_lora/qwen3_8b_lora_sft_sdL2fa_traj.yaml` = v1 yaml with dataset/output_dir changed; launcher `scripts/train_qwen3_8b_sdL2_sft.slurm`, 2 GPUs, GA 8, partitions a100/h100/h200/l40s, no `LF_NO_HISTORY_EMPTY_COT` (identical to v1) |
| ckpt | `LLaMA-Factory/saves/qwen3-8b/lora/sdL2fa_traj_sft_8k_r64_GA4_qkvo_3epoch_5e-6` |
| eval | auto-submitted by a chained job (`afterok` on the training job) → `agentdojo/runs/sdL2fa_traj_3epoch`, stats CSV `eval/attack_stats_sdL2fa_traj_3epoch_…csv` |
| results (partial 922/949, 2026-09-09 02:00; slack/bank/benign still running) | UA **46.75** / ASR **1.95** (18 attacks); benign 53.19 on 94 of 116 tasks. Per suite UA/ASR: banking 35.6/6.7 (135) · slack 50.6/2.3 (87) · travel 31.4/0.7 · workspace 52.7/1.1. ⇒ with the multi-turn recipe the Qwen L2 thinks keep UA at v0 level (46.7); the cost of the shorter thinks shows up in ASR (1.05 → ~1.9) |

## v0' — v3base-traj: paper recipe on recovered data
| field | value |
|---|---|
| why | `toucan_32B_v2.json` lacks the assistant reply before every new user turn (3,185 of 6,892 `gpt` turns, 46 %, in 1,828 multi-turn trajectories): the model was trained to answer `<think>…</think>` + nothing there. Fix it once, rebase every comparison on it |
| source | per-step records pulled from DSAI 2026-09-09 → `agentdojo/qwen3-32b-bedrock-samples/<suite>/user_task_N/injection_task_M/assistant_step_NNN.json` (22,930 files, 355 MB; 22,456 used, the 474 extra are `windows_v1/*/template_1` runs not in the training set) |
| data | `data_recovery/rebuild_from_steps.py --strict --drop-raw-trajectories` → `LLaMA-Factory/data/toucan_32B_v3_base.json` (registered `toucan_32B_v3_base`): **3,698 trajectories / 22,339 assistant steps** (v0: 3707 / 22,456). All 22,456 source steps verified (think text, tool calls, non-empty answers identical); **3,185 filled, 0 empty left**; filled replies mean 192 words; the 9 hotel user_task_134 trajectories (117 steps) whose expert made 2 malformed `{"_raw": …}` calls were dropped (user decision 2026-09-09) → 0 unparsable tool-call targets. `meta.injections` (trigger/task/target message) + `meta.steps[i].has_injection_inserted_by_step` (13,790 of 22,456 steps have an injection in context) + candidate stats added. Report `toucan_32B_v3_base.json.report.json` |
| known residual gaps (unchanged vs v0) | 11,482 tool-call steps had expert text next to the call in the source, not restored — **irrelevant by design** (user 2026-09-09): AgentDojo's `ToolsExecutor` only acts on `tool_calls` (`tool_execution.py` L70-74); text beside a call never reaches the utility check. 348 trajectories exceed the 8,192 cutoff (truncated by LLaMA-Factory as in v0) |
| length | rendered with the qwen3 template: tokens/traj mean 4,644 (v0 4,352), 348 trajectories > 8,192 cutoff (v0 203); label tokens 12.43 M (v0 11.35 M, +9.5 %) |
| training | `examples/train_lora/qwen3_8b_lora_sft_v3base_traj.yaml` = paper yaml with only dataset/output_dir changed (verified by diff); launcher `scripts/train_qwen3_8b_sdL2_sft.slurm`, 2 GPUs GA 8 as v2-traj |
| ckpt | `LLaMA-Factory/saves/qwen3-8b/lora/v3base_traj_sft_8k_r64_GA4_qkvo_3epoch_5e-6` |
| eval | `submit_eval_sdL2.sh v3base_traj_3epoch <ckpt>` chained `afterok` on the training job → `agentdojo/runs/v3base_traj_3epoch` |
| results | pending |

## v3-paraphrase — context-grounded paraphrase of Claude thinks
| field | value |
|---|---|
| recipe | `docs/04_self_distill_plan.md` "v3 recipe" §1–§2 (+ §1b ground-truth action list, §4a pilots) |
| what changes vs v1/v2 | Qwen no longer reasons about a given action; it rewrites Claude's reflection in first person with all alternatives/consequences kept, seeing the inference prefix, the deduped 32B candidates and the expert action (tagged). Length ≈ Claude's, ≤ 480 tokens; content preserved by filters |
| source data | `LLaMA-Factory/data/toucan_32B_v3_base.json` (3,698 traj / 22,339 steps); injection labels from `meta.steps` ground truth |
| generation | vLLM 0.9.2, Qwen/Qwen3-8B, thinking OFF, n=2 T=0.6 (+ retry n=4 T=0.8), max 900 tok; 8 shards × 1 GPU: array **342969**; raw attempts `self_distill/runs/v3para/out_shard*.jsonl`; assembly **342970** (`assemble_paraphrase.py`) → `toucan_32B_v3_para.json` + `.stats.json` |
| pilots | 342667 (first prompt): 134/200 raw → 185/200 after filter fixes; 342911 (GT-annotated prompt): 174/200 raw → 190/200 (95 %) after relaxing the paragraph filter; review file `self_distill/pilot_para/review_samples.md` |
| full run result | shards 342969 (8 × 39 min on L40S): 20,268/22,339 accepted (90.7 %); rescue 343881 (47 min: re-judge with widened final-step regex + regenerate with a length note, n=6, T=0.9) recovered 1,808 of the 2,071 rejected; assembly 342970 → **`toucan_32B_v3_para.json`: 3,698 traj / 22,339 steps, 22,076 paraphrased (98.8 %), 263 Claude fallbacks (1.2 %: step-0 19, clean 24, injected 60, final 160) in 237 trajectories**. Paraphrase length 270 words / 356 tokens mean (Claude 292 words / 396 tokens), p90 433 tokens, 0.3 % > 480 (the fallbacks). Format check: 0 bad separators, 0 empty bodies; leak phrases 12/3000 sampled thinks; first person 99 %. Stats `toucan_32B_v3_para.stats.json` |
| training | `examples/train_lora/qwen3_8b_lora_sft_v3para_traj.yaml` (= paper yaml, dataset/output_dir only), launcher `scripts/train_qwen3_8b_sdL2_sft.slurm`. Job 342999 (2 GPUs GA 8, chained afterok 342970) was eligible from 04:33 but never scheduled in 8 h (2-GPU requests starve; 1-GPU jobs started within seconds all night) → cancelled 12:20 and resubmitted as 346239 → **347215** (1 GPU, `NPROC_PER_NODE=1`, GA 16 = same effective batch 16, 10 h, a100/h100/h200/l40s; mem lowered to 30 GB = 5 CPUs so it fits the free slots on busy l40s nodes) |
| ckpt | `LLaMA-Factory/saves/qwen3-8b/lora/v3para_traj_sft_8k_r64_GA4_qkvo_3epoch_5e-6` |
| eval | job **347216** (afterok 347215) runs `submit_eval_sdL2.sh v3para_traj_3epoch <ckpt>` → `agentdojo/runs/v3para_traj_3epoch`, CSV `eval/attack_stats_v3para_traj_3epoch_…csv`; 1 seed first |
| results | pending |
