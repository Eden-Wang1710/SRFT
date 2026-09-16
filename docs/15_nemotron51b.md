# Llama-3.1-Nemotron-51B as a larger Llama base — feasibility check (2026-09-16, skipjack, branch `exp/nemotron-51b`)

## Why this model
The ICLR story needs a Llama row above 8B, but **Meta ships nothing between 8B and 70B**: Llama-3.1 = 8B/70B/405B,
Llama-3.2 = 1B/3B text (11B/90B are vision), Llama-3.3 = 70B only, Llama-4 = MoE (Scout 109B-A17B / Maverick 400B-A17B).
Llama-2-13B is the only 13B and is unusable (2023, 4k context, no tool calling).

`nvidia/Llama-3_1-Nemotron-51B-Instruct` fills the gap: NAS-pruned + distilled **from Llama-3.1-70B**, plain instruct
(NOT a reasoning model — unlike `Llama-3_3-Nemotron-Super-49B`, which is post-trained for agentic tool calling and would
re-introduce the Qwen "overwrite the native think distribution" problem). Practical advantage: 96 GiB in bf16 fits on
**one H200 (141 GB) or one B200 (180 GB)**, so eval keeps the current 1-GPU-per-shard sharding instead of needing TP=2
like a 70B would.

Caveat for the paper: it is an NVIDIA derivative, not a Meta release, and there is no Meta-SecAlign counterpart at 51B,
so it buys a scale point but not a same-base head-to-head (that only exists at 70B — Meta-SecAlign reports AgentDojo
for Llama-3.3-70B only; see `docs/13` §1a notes).

## Architecture (verified from the downloaded config, 2026-09-16)
| property | value |
|---|---|
| `model_type` / `architectures` | `nemotron-nas` / `DeciLMForCausalLM` — custom code, **`trust_remote_code: true` is required** |
| layers / hidden / vocab | 80 / 8192 / 128256, `max_position_embeddings` 131072, rope `llama3` (factor 8.0), `pretraining_tp` 1 |
| total params | **51.50 B** → bf16 weights **96 GiB** |
| attention per layer | **54 of 80 layers have a real attention block** (44 with `n_heads_in_group` 8, 6 with 32, 3 with 64, 1 with 16); **18 layers replace attention with a single `linear_attn` projection**; **8 layers have no attention at all** |
| FFN per layer | present in all 80 (`ffn_mult` 5.25 ×44, 1.3125 ×20, 2.625 ×16) |
| attn implementations | `DECILM_ATTENTION_CLASSES` = `eager` / `flash_attention_2` / `sdpa` — all three LLaMA-Factory may select |
| support flags | `supports_gradient_checkpointing=True`, `_supports_sdpa=True`, `_supports_flash_attn_2=True`, `_no_split_modules=["DeciLMDecoderLayer"]` |
| weights on skipjack | `$HF_HOME/hub/models--nvidia--Llama-3_1-Nemotron-51B-Instruct`, snapshot `f4d9431910e0`, 22 safetensors shards, 96 GB |

## LoRA attaches — the question this doc was opened for
Verified without loading any weights: build the skeleton on a meta device (`accelerate.init_empty_weights`) and run
`peft.get_peft_model` with the paper's target list.

```
lora_target: q_proj,k_proj,v_proj,o_proj   (r64, alpha96)
-> 216 modules across 54 of 80 layers
-> trainable params: 175,947,776 || all params: 51,676,962,816 || trainable%: 0.3405
```
For comparison the 8B run trains 54.5 M (0.67 %). The 18 `linear_attn` layers are **deliberately not** LoRA targets, so
the recipe stays the paper's "q/k/v/o only" — PEFT simply finds no `q_proj` in those layers and skips them.

## The one real incompatibility: transformers version
`modeling_decilm.py` line 28 imports `NEED_SETUP_CACHE_CLASSES_MAPPING` from `transformers.generation.utils`. That
constant was removed after 4.5x, so **transformers 4.57.1 raises `ImportError` before the model can be built**.

- Its only use is one line inside `_prepare_generation_config` (`NEED_SETUP_CACHE_CLASSES_MAPPING["variable"] = VariableCache`),
  i.e. it matters for **generation**, not training. A shim would therefore let training run on 4.57.1.
- But a shim does **not** fix inference: in 4.57.1 the valid cache implementations are the fixed tuple
  `ALL_CACHE_IMPLEMENTATIONS` (`static`, `offloaded_static`, `sliding_window`, `hybrid`, …) with **no registration entry
  point** (`CACHE_IMPLEMENTATION_MAPPING` no longer exists), and the remote code hard-sets `cache_implementation="variable"`.
- **transformers 4.51.3 runs it with no patch at all** (config + model instantiation + PEFT all clean), satisfies
  LLaMA-Factory's `transformers>=4.49.0`, and is already the version the `agentdojo` eval env uses.
- LLaMA-Factory 0.9.4.dev0's only `>4.57.0` guards are conditional imports of `qwen3_omni_moe` (`model/patcher.py:46,54`,
  `model_utils/moe.py:32`) — irrelevant here, and properly guarded. All of `llamafactory.{model.patcher, model.loader,
  model.adapter, data.template, hparams.parser, train.sft.workflow}` import cleanly under 4.51.3.

⇒ **Decision: pin the version, do not patch the model.** New conda env `nemotron` = clone of `llamafactory` with
`transformers==4.51.3` (trl 0.9.6, datasets 4.0.0, peft 0.17.1, accelerate 1.11.0, torch 2.7.0+cu128).

## Tokenizer is byte-identical to Llama-3.1-8B-Instruct
`chat_template` sha1 `f15c056603d9` on both; same `bos`/`eos` (`<|begin_of_text|>` / `<|eot_id|>`), same 256
`added_tokens_decoder` ids, same `model_max_length` 131072. Therefore **the existing data and renderer are reused verbatim**:
`LLaMA-Factory/data/toucan_32B_v3_base_llama_local.json`, `template: llama3`, and
`agentdojo/.../llms/llama_local_prompt.py`. No new conversion, no new template.

## Files added / changed on this branch
- `LLaMA-Factory/examples/train_lora/nemotron51b_lora_sft_v3base_local.yaml` — the 8B yaml with only `model_name_or_path`,
  `output_dir` and `gradient_checkpointing: true` changed (the last is memory-only, math-neutral).
- `LLaMA-Factory/scripts/train_qwen3_8b_sdL2_sft.slurm` — `conda activate llamafactory` → `"${ENV:-llamafactory}"`
  so a run can select the `nemotron` env; backward compatible, every existing caller is unaffected.

## Memory / partition
Estimate for 1 GPU, batch 1, cutoff 8192, gradient checkpointing on: 96 GiB weights + ~2.5 GB LoRA optimizer state
+ ~11 GB checkpointed activations + ~8 GB logits/CE peak (vocab 128256) ≈ **118 GB** → comfortable on B200 (180 GB),
feasible but tight on H200 (141 GB). **`env/skipjack.sh` `GPU_PARTITIONS` does not include `b200`**; pass `-p b200,h200`
explicitly (`env/sb` appends user flags after its own `-p`, so the later one wins).

## Still to do on the eval side (not yet applied — known cost, not blockers)
1. `agentdojo/src/agentdojo/agent_pipeline/llms/llama_sr_agent_llm.py:82-83` — neither `AutoTokenizer.from_pretrained`
   nor `AutoModelForCausalLM.from_pretrained` passes `trust_remote_code=True`; required for `nemotron-nas`, harmless for
   Llama-3.1-8B. Everything else already generalises: the provider reads `LLAMA_SR_AGENT_BASE_MODEL` and uses
   `device_map="auto"`, so **no new provider class is needed**.
2. `agentdojo/src/agentdojo/models.py` — add a `ModelsEnum` entry + its `"hf_llama_sr_agent"` provider mapping.
3. `agentdojo/scripts/submit_eval_sdL2.sh` — add a `case "$MODEL"` branch (its default arm is `exit 2`).

## Verdict: YES — LLaMA-Factory can LoRA-tune this model (verified 2026-09-16)

The code path is **proved**, without waiting for a 96 GiB GPU slot. A tiny model with the *same* `nemotron-nas` remote code
and one layer of each variant (0 = normal attention, 1 = `replace_with_linear`, 2 = `no_op`, 3 = normal attention),
`hidden_size` 512, was built from the real `configuration_decilm.py`/`modeling_decilm.py` and trained through the real CLI:

```
python -m llamafactory.cli train examples/train_lora/nemotron51b_lora_sft_v3base_local.yaml \
  model_name_or_path=<tiny> bf16=false fp16=false max_steps=1 max_samples=8 cutoff_len=512 \
  save_strategy=steps save_steps=100000 gradient_checkpointing=false
-> {'loss': 11.8577, 'grad_norm': 0.9576982855796814, 'learning_rate': 0.0, 'epoch': 0.5}
-> train_runtime 0:00:01.78, adapter written
```
`grad_norm` finite and non-zero ⇒ gradients really flow through the adapters. The written `adapter_config.json` has
`target_modules ['o_proj','q_proj','v_proj','k_proj']`, `r 64`, `lora_alpha 96` — the paper recipe, produced by LF itself.
`adapter_model.safetensors` holds 16 tensors, 4 per target module, on **layers 0 and 3 only**: the `replace_with_linear`
and `no_op` layers are skipped cleanly rather than crashing or being mis-targeted. That is exactly the behaviour the
54/80-layer count predicts at full scale.

What the tiny run does **not** cover: memory and throughput at 51.5 B. That is the only open item.

### Real-scale smoke
Job **472145** — `-p h200 --gres=gpu:1 -c 6 --mem=48G -t 01:00:00`, `ENV=nemotron`, `NPROC_PER_NODE=1`,
`EXTRA_ARGS="max_steps=2 max_samples=64 save_strategy=steps save_steps=100000 logging_steps=1 output_dir=saves/_smoke_nemo51b"`.
Pass criteria: `trainable params: 175,947,776`, two finite-loss steps, no OOM. Result to be filled in here.

Two submission lessons already paid for (both now in `docs/09`): **b200 is closed to our account**
(`AllowAccounts=schmidt`) so `-p b200` parks forever at `Reason=PartitionConfig` however many cards are idle; and
`save_strategy=no` in `EXTRA_ARGS` is parsed as the boolean `False` and kills the job in argument parsing — it would have
failed *after* a multi-day queue wait.
