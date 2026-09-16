# Meta-SecAlign's own lm-eval task configs (verbatim copy, 2026-09-15)

Source: https://github.com/facebookresearch/Meta_SecAlign `lm_eval_config/` + `test_lm_eval.py` (kept as
`test_lm_eval.py.orig` for reference). These are the configs that produced the utility rows in the Meta-SecAlign
paper (arXiv 2507.02735, Table 3). They are Meta's Llama-3.1 evaluation recipe: every prompt is taken pre-rendered
(Llama-3.1 chat format baked in) from the gated datasets `meta-llama/Llama-3.1-8B-Instruct-evals` /
`meta-llama/Llama-3.1-8B-evals`, run with `apply_chat_template=False`, `num_fewshot=0` (shots are inside the prompts).

| their task | protocol | published base / Meta-SecAlign-8B |
|---|---|---|
| `meta_mmlu_0shot_instruct` | MMLU **0-shot CoT, generative**, regex `best answer is ([A-Z])`, 1024 tok | 72.0 / 71.7 |
| `meta_mmlu_pro_instruct` | MMLU-Pro 5-shot CoT, full 12,032 items, regex `best answer is ([A-Z])`, 1024 tok | 46.5 / 46.7 |
| `meta_bbh` | BBH 3-shot CoT, `until: "\n\nQ: "` (NOT a bare blank line), regex `the answer is (.*?)\.`, 512 tok | 71.9 / 70.9 |
| `meta_ifeval` | IFEval, 1280 tok; the paper's number is the mean of the 4 sub-metrics (`test_lm_eval.py`) | 79.1 / 74.5 |

`ifeval/joined_ifeval.parquet` (their data file) is NOT copied — fetch it from the repo if `meta_ifeval` is run.
Why we keep these: our own MMLU row (lm-eval `mmlu`, 0-shot **loglikelihood**, no template) is a different
metric from theirs, which is why our base scores 68.00 against their 72.0 while the other three rows match.
Running these configs on our base and on SR-Agent-Llama makes the comparison to Meta-SecAlign-8B exact.
