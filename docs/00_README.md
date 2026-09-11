# SRFT docs index

Files are numbered in reading order: 01 ledger/results → 02–04 paper & analysis & method → 05–06 how to train/eval → 07–09 assets, env, cluster → 99 changelog.

Project: **SRFT (Self-Reflection Fine-Tuning)** — SR-Agent = Qwen3-8B + LoRA, defends tool-using agents
against prompt injection. Submitted to NeurIPS 2026 (paper id 27619), now being resubmitted to **ICLR**.
Migrated from the old server (DSAI, `/scratch/cxiao13/zixuan/SRFT`) to skipjack on 2026-09-04.

| File | What it holds |
|---|---|
| [01_experiments.md](01_experiments.md) | **Experiment ledger**: one row/section per version (idea, data, training, ckpt, eval dir, numbers) — sync point for the ICLR push |
| [02_paper_summary.md](02_paper_summary.md) | Method, setup, headline numbers, appendix details from the NeurIPS PDF |
| [07_assets.md](07_assets.md) | Paths of data, LoRA checkpoints, run trajectories; migration status |
| [06_eval_agentdojo.md](06_eval_agentdojo.md) | Verified commands: stats for a run dir, launching the benchmark |
| [08_environment.md](08_environment.md) | conda env `agentdojo`, versions |
| [09_cluster_skipjack.md](09_cluster_skipjack.md) | Skipjack (JHU) docs links, hardware/partitions, ssh config, queue gotchas |
| [09_cluster_washu.md](09_cluster_washu.md) | WashU RIS Compute2: account, partitions, storage, how to submit, queue reality, verified commands |
| [10_sync_workflow.md](10_sync_workflow.md) | Two-cluster workflow: `env/` layer, `env/sb` job submission, git branch rules, HF data/ckpt exchange, new-machine setup |
| [11_secalign_utility_analysis.md](11_secalign_utility_analysis.md) | why SecAlign/Meta-SecAlign DPO keeps utility on the same Qwen base (papers + our replication) and what it implies for v3 / an Alpaca mix |
| [03_analysis_utility_drop.md](03_analysis_utility_drop.md) | Why SR-Agent loses utility (data + inference evidence) and review of the ICLR self-rewrite plan |
| [04_self_distill_plan.md](04_self_distill_plan.md) | Concrete self-distillation recipe: per-step prompts, hint ladder, filters, training config |
| [05_training.md](05_training.md) | LLaMA-Factory env, paper recipe, verified launcher, data-format gotcha, run table |
| [99_changelog.md](99_changelog.md) | Dated log |

Machine layer: `SRFT/env/` (`local.sh` marker → `skipjack.sh`/`washu.sh`, `sb` submitter, `freeze/`). Code outside docs: `SRFT/data_recovery/` (README + script to fill the missing mid-turn replies from DSAI per-step records); `SRFT/multibase/` (other base models: Llama data converter + train/inference render check, 2026-09-11).

Root on skipjack: `/weka/scratch/jhu/cxiao13/zwang544/SRFT/` (also `~/scratch_cxiao13/zwang544/SRFT`); on WashU: `/storage3/fs1/zhang.ning/Active/hao/zixuan/SRFT` (`09_cluster_washu.md`).
Git: private GitHub repo `SRFT`, branch `main` (since 2026-09-09; the pre-NeurIPS repo was renamed `SRFT-nips-archive`).
