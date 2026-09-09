# SRFT workspace (ICLR resubmission of the NeurIPS 2026 paper "Self-Reflection Fine-Tuning")

**First thing every session: `cat env/local.sh`** → `SRFT_CLUSTER=skipjack` or `washu`. Then read `docs/09_cluster_<that>.md` for how jobs are
submitted there. The repo is shared by two clusters through GitHub (`main`) and HuggingFace (data/ckpts); rules in `docs/10_sync_workflow.md`.
Start with `git pull --rebase`; commit and push at the end of every working block (docs + ledger included).

Read `docs/00_README.md` first, then `docs/01_experiments.md` (the experiment ledger — every version's data / ckpt / eval / numbers). The `docs/` folder is the project's living notebook:
- `docs/00_README.md`                 — index
- `docs/01_experiments.md`            — EXPERIMENT LEDGER: one row/section per version (data, ckpt, eval dir, numbers) — the sync point
- `docs/02_paper_summary.md`          — key points and numbers of the submitted paper
- `docs/03_analysis_utility_drop.md`  — utility-drop diagnosis (v0) and v1 root cause (§F)
- `docs/04_self_distill_plan.md`      — self-distillation recipe, smoke + full L2 run results
- `docs/05_training.md`               — how to train the LoRA (env, launcher, format gotcha, run table)
- `docs/06_eval_agentdojo.md`         — how to compute stats for a run and how to launch AgentDojo evaluations
- `docs/07_assets.md`                 — where data / checkpoints / trajectories live, what was migrated, what is missing
- `docs/08_environment.md`            — conda envs, versions (freezes in `env/freeze/`)
- `docs/09_cluster_skipjack.md`       — Skipjack (JHU) docs links, partitions, quirks; `docs/09_cluster_washu.md` — WashU counterpart
- `docs/10_sync_workflow.md`          — two-cluster workflow: env layer, `env/sb` submission, git branch rules, HF data/ckpt exchange
- `docs/99_changelog.md`              — dated log of what was done

Rules for Claude:
- Cluster settings (conda path, HF cache, account, partitions, extra sbatch flags) come ONLY from `env/<cluster>.sh` via `env/select.sh`.
  Submit jobs with `bash env/sb gpu|cpu [flags] <script>`; never write absolute paths, accounts or partitions into a script or doc example.
- For any cluster/slurm/GPU/storage question, first read the matching `docs/09_cluster_*.md`; on skipjack the official docs at
  https://docs.arch.jhu.edu/en/latest/ can be fetched with WebFetch. Do not guess.
- Every new experiment version gets a row in `docs/01_experiments.md` summary + a section with the fixed fields when it is planned; fill in numbers when its stats CSV exists. Never overwrite a version.
- Whenever you verify a command, learn where a file lives, or run an experiment, append to the matching doc (and a dated line in `99_changelog.md`) in the same session, then commit.
- Data files (`LLaMA-Factory/data/toucan_*.json`) and LoRA dirs (`LLaMA-Factory/saves/`) are git-ignored: after producing one, upload it to the
  private HF repos (`10_sync_workflow.md` §HuggingFace) and write the HF path into the ledger.
- In this repo's AgentDojo result JSONs, `security: true` means the ATTACK SUCCEEDED (inverted vs upstream AgentDojo). Use `eval/compute_attack_stats.py`, do not hand-roll metrics.
- Do not modify `agentdojo/runs/3_01_*` or `LLaMA-Factory/saves/*` (paper artifacts).
