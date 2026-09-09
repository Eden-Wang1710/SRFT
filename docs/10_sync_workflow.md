# Two-cluster workflow: git (code, docs, results) + HuggingFace (data, checkpoints)

Decided 2026-09-09. Goal: work on skipjack (JHU) and the WashU cluster without copying files through a laptop.

## What lives where
| kind | where | notes |
|---|---|---|
| code, sbatch/yaml configs, docs, ledger, eval CSVs, AgentDojo trajectories (`agentdojo/runs/`), `self_distill/recovery`, `pilot_para`, `smoke` inputs | **GitHub** private repo `SRFT`, branch `main` | ~150 MB of text; `agentdojo/runs/` grows ≈ 20 MB / 1,081 files per evaluated version — revisit (move to HF) if the repo passes ~1 GB |
| training data `LLaMA-Factory/data/toucan_*.json` | **HF dataset repo** `<user>/srft-data` (private) | files up to 161 MB (> GitHub's 100 MB limit); `dataset_info.json` + `*.stats.json` + `*.report.json` stay in git |
| LoRA checkpoints `LLaMA-Factory/saves/qwen3-8b/lora/<run>/` | **HF model repo** `<user>/srft-ckpts` (private), one folder per run | upload only the top-level adapter (≈ 250 MB): `adapter_model.safetensors adapter_config.json trainer_state.json trainer_log.jsonl train_results.json training_loss.png README.md` + tokenizer files. NOT the `checkpoint-*/` dirs (718 MB of optimizer state each) unless an epoch checkpoint is itself an experiment (v2-e1 style) |
| `agentdojo/qwen3-32b-bedrock-samples/` (355 MB, 22,930 files) | skipjack only | needed only to rebuild `toucan_32B_v3_base.json`; rsync once if WashU ever needs it |
| `self_distill/runs/`, `agentdojo/logs*/`, `LLaMA-Factory/logs/`, `slurm_logs/` | local only (git-ignored) | raw generation attempts and slurm logs; reproducible or historical |
| conda envs, HF model cache (Qwen/Qwen3-8B 16 GB) | per machine | rebuild from `env/freeze/*.txt` (`08_environment.md`) |

## Machine layer (`env/`)
- `env/local.sh` — git-ignored, one line `SRFT_CLUSTER=skipjack|washu`, created once per machine (`env/local.sh.example`). **This is how
  Claude and every script know which cluster they are on; the user never has to say it.**
- `env/skipjack.sh`, `env/washu.sh` — committed; conda.sh path, HF cache, account, partitions, extra sbatch flags, `srft_sbatch` wrapper.
- `env/select.sh` — sourced by every job script; exports `SRFT_ROOT SRFT_CLUSTER CONDA_SH HF_HOME ... SRFT_SLURM_LOGS`.
- `env/sb gpu|cpu [sbatch flags] <script>` — the only way to submit; adds `-A/-p/extra/-o` for the current cluster. Flags after `gpu|cpu`
  override (e.g. `-p l40s --gres=gpu:1 -t 10:00:00 --export=ALL,SFT_CONFIG=...`).
- `env/freeze/{agentdojo,llamafactory,sd_gen}.txt` — pip freezes of the three envs (skipjack, 2026-09-09). `env/jobs/` — smoke-test sbatch files.
- Job scripts must not contain absolute paths, accounts or partitions. Pattern (already in all 10 scripts):
  ```bash
  SRFT_ROOT="${SRFT_ROOT:-$(d="${SLURM_SUBMIT_DIR:-$PWD}"; while [ ! -f "$d/env/select.sh" ] && [ "$d" != / ]; do d=$(dirname "$d"); done; echo "$d")}"
  source "$SRFT_ROOT/env/select.sh"
  source "$CONDA_SH" && conda activate <env>
  cd "$SRFT_ROOT/<subdir>"
  ```

## Running jobs (both clusters)
```bash
cd $SRFT_ROOT
bash env/sb gpu self_distill/shard.sbatch                                   # array of 8 × 1 GPU
bash env/sb gpu --gres=gpu:1 -c 5 --mem=30G -t 10:00:00 \
  --export=ALL,SFT_CONFIG=examples/train_lora/<cfg>.yaml,EXTRA_ARGS="gradient_accumulation_steps=16" \
  LLaMA-Factory/scripts/train_qwen3_8b_sdL2_sft.slurm                       # 1-GPU LoRA SFT (recipe in 05_training.md)
bash agentdojo/scripts/submit_eval_sdL2.sh <RUN_NAME> <abs LoRA dir>        # 10 GPU jobs + stats job (06_eval_agentdojo.md §4)
bash env/sb gpu --test-only <script>                                        # dry-run: scheduler accepts it, prints est. start
```
Logs: `slurm_logs/<jobname>_<jobid>.out`. Eval per-process logs: `agentdojo/logs/eval_parallel/<run>/...`.

## Git rules
- `main` is the only long-lived branch and must always be usable on both clusters. Start every session with `git pull --rebase`.
- One branch per experiment version (`exp/v3-paraphrase`, `exp/v4-...`), created from `main` on whichever cluster runs it. It holds the
  training yaml, `dataset_info.json` entry, the ledger row/section, and when done the `agentdojo/runs/<run>` trajectories + `eval/*.csv`.
  Merge into `main` (no squash needed) as soon as the numbers are in the ledger, then delete the branch.
- Doc/script fixes go straight to `main`.
- An experiment runs on ONE cluster end to end (data → train → eval); do not split a version across clusters.
- `docs/01_experiments.md`, `docs/99_changelog.md`, `docs/05_training.md` use `merge=union` (`.gitattributes`): concurrent appends from
  both clusters merge without conflicts. Still read the result after a merge — union keeps both sides' lines, it does not order them.
- Never commit: anything under `saves/`, `toucan_*.json`, `slurm_logs/`, `env/local.sh`. A pre-commit size guard is not installed; if
  `git push` is rejected for a >100 MB file, `git rm --cached` it and add it to `.gitignore`.
- Commit messages: `<area>: <what>` (e.g. `v3-para: eval numbers`, `docs: washu cluster notes`).

## HuggingFace exchange (both clusters can reach huggingface.co; `hf` CLI is in the `llamafactory` and `sd_gen` envs)
One-time per machine: `conda activate llamafactory && hf auth login` (write token).
```bash
# upload a dataset file (from the machine that built it)
hf upload <user>/srft-data LLaMA-Factory/data/toucan_32B_v3_para.json toucan_32B_v3_para.json --repo-type dataset
# download it on the other machine
hf download <user>/srft-data toucan_32B_v3_para.json --repo-type dataset --local-dir LLaMA-Factory/data
# upload a LoRA (top level only)
R=v3para_traj_sft_8k_r64_GA4_qkvo_3epoch_5e-6
hf upload <user>/srft-ckpts LLaMA-Factory/saves/qwen3-8b/lora/$R $R --exclude "checkpoint-*/*"
# download a LoRA
hf download <user>/srft-ckpts --include "$R/*" --local-dir LLaMA-Factory/saves/qwen3-8b/lora
```
Ledger convention: the `data` and `ckpt` fields of a version keep the local path AND, once uploaded, `HF: srft-data/<file>` / `HF: srft-ckpts/<run>`.
If `HF_HUB_OFFLINE=1` is exported (skipjack default, `env/skipjack.sh`), prefix the command with `HF_HUB_OFFLINE=0`.

## First-time setup on a new machine (WashU)
1. `git clone git@github.com:<user>/SRFT.git` into scratch (not home); `cp env/local.sh.example env/local.sh`, set `SRFT_CLUSTER=washu`.
2. Fill every `CHANGE_ME` in `env/washu.sh`; write `docs/09_cluster_washu.md`; commit both on `main`.
3. Rebuild conda envs from `env/freeze/*.txt` (recipe in `08_environment.md`; `agentdojo` and `LLaMA-Factory` are editable installs from this repo).
4. `hf auth login`; pull Qwen/Qwen3-8B into `HF_HOME` (`hf download Qwen/Qwen3-8B`), then set `HF_HUB_OFFLINE=1` in `env/washu.sh`.
5. Download what the next experiment needs from `srft-data` / `srft-ckpts` (at least `toucan_32B_v3_base.json` and the paper LoRA for a smoke eval).
6. Verify: `bash env/sb gpu --test-only self_distill/shard.sbatch`; run `env/jobs/gpu_smoke.sbatch`; run one eval shard
   (`PARTS=... bash agentdojo/scripts/submit_eval_sdL2.sh smoke_<date> <paper LoRA>` and cancel after the first shard finishes, or run
   `scripts/eval_parallel.py --suite banking --attack important_instructions --injection-tasks injection_task_0 --nproc 1 --logdir runs/smoke`).
7. Log what was verified in `09_cluster_washu.md` and `99_changelog.md`, push.
