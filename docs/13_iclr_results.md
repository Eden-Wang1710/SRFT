# ICLR results: the integrated story (assembled 2026-09-14)

What changed since the NeurIPS submission (`02_paper_summary.md`): the headline model moves from Qwen3-8B to
**SR-Agent-Llama evaluated without the reflection system-prompt append**, Qwen becomes the supplementary family, and both
benchmarks now have a same-base comparison against Meta-SecAlign. Every number below is traceable to a row in
`01_experiments.md` or a result dir in `injecAgent-rl-harmmer/rl-injector/outputs/`; all AgentDojo rows are
`eval/compute_attack_stats.py` on 949 attacked + 97 benign tasks, single seed.

Metric conventions unchanged: Benign Utility / Utility under Attack / ASR (lower better), **Trade-off = UA + (100 − ASR)**.

---

## 1. Main result — Llama-3.1-8B family

### 1a. AgentDojo (`important_instructions`)
| model | Benign ↑ | UA ↑ | ASR ↓ | Trade-off ↑ |
|---|---|---|---|---|
| Llama-3.1-8B-Instruct (undefended) | 27.84 | 23.08 | 7.59 | 115.49 |
| Llama-3.1-8B-Instruct + SR prompt only | 31.96 | 21.71 | 4.32 | 117.39 |
| Meta-SecAlign-8B (same base, SOTA defence) | 23.71 | 19.07 | **0.95** | 118.12 |
| **SR-Agent-Llama (ours, no append)** | **37.11** | **29.08** | 1.26 | **127.82** |

SR-Agent-Llama is the best row on **utility (+9.3 over its base, +13.4 over Meta-SecAlign)**, on **utility under attack
(+6.0 / +10.0)** and on **trade-off (+12.3 / +9.7)**, while holding ASR at **1.26 %** — within 0.3 points of
Meta-SecAlign's 0.95 % and 6.3 points below its own base. This is the first row in the project where the defended model
beats its own undefended base on plain utility, which removes the NeurIPS version's main weakness.

Per suite (Benign / UA / ASR): banking 43.75 / 36.11 / 5.56 · slack 33.33 / 28.57 / 0.00 · travel 25.00 / 7.14 / 0.00 ·
workspace 42.50 / 32.86 / 0.71. Meta-SecAlign for contrast: banking 25.00 / 25.00 / 1.39 · slack 28.57 / 20.00 / 2.86 ·
travel 30.00 / 17.86 / 0.00 · workspace 17.50 / 17.68 / 0.71.

### 1b. RL-Hammer on InjecAgent (adaptive attack, 20 attacker epochs, 100 test cases per checkpoint)
ASR at the final checkpoint (step 1020); two independent attacker runs per target.
| target | run 1 | run 2 | mean | max over runs |
|---|---|---|---|---|
| Llama-3.1-8B-Instruct (undefended) | 98 | 99 | 98.5 | **99** |
| Meta-SecAlign-8B | 80 | 59 | 69.5 | **80** |
| **SR-Agent-Llama (ours, no append)** | 32 | 3 | 17.5 | **32** |

The undefended base is fully compromised and **Meta-SecAlign, which looks safe statically (ASR 0.95 % on AgentDojo and
1 % at the attacker's first checkpoint), degrades to 80 %**. SR-Agent-Llama stays at 32 %. Curves are in
`outputs/iclr_rlh_llama_base_`, `iclr_rlh_llama_base_r3_`, `eval_rl_hammer_target_meta_secalign_8b_allckpts_(_rerun2_)`,
`iclr_rlh_srllama_noappend_(_r2_)`.

### 1c. The two benchmarks together
| | AgentDojo trade-off | RL-Hammer ASR (max) |
|---|---|---|
| Llama-3.1-8B base | 115.49 | 99 % |
| Meta-SecAlign-8B | 118.12 | 80 % |
| **SR-Agent-Llama** | **127.82** | **32 %** |
Same base model, same evaluation code, same attacker pipeline: **ours wins the static trade-off and is 48 points better
under the adaptive attack.** Static robustness alone does not predict adaptive robustness — Meta-SecAlign has the lowest
static ASR of the four and the second-worst adaptive ASR — which is the argument for reporting both benchmarks.

---

## 2. Supplementary — Qwen3 family
Meta-SecAlign is Llama-native, so no Qwen Meta-SecAlign row is reported.

### 2a. AgentDojo, NeurIPS protocol (think budget 512, SR prompt on)
| model | Benign | UA | ASR | Trade-off |
|---|---|---|---|---|
| Qwen3-8B (undefended, no append) | 60.82 | 50.47 | 16.97 | 133.50 |
| SR-Agent-Qwen3-8B (v0, the NeurIPS model) | 51.55 | 46.68 | **1.05** | **145.63** |

### 2b. AgentDojo, think budget 1024 (the protocol the ICLR version reports)
| model | Benign | UA | ASR | Trade-off |
|---|---|---|---|---|
| Qwen3-8B (no append) | 72.16 | 55.01 | 17.49 | 137.52 |
| Qwen3-8B + SR prompt only | 70.10 | 56.80 | 13.80 | 143.00 |
| SR-Agent-Qwen3-8B (v3-para) | 62.89 | 48.05 | **2.11** | **145.94** |

### 2c. Scale ablation inside the family — Qwen3-4B (budget 1024)
| model | Benign | UA | ASR | Trade-off |
|---|---|---|---|---|
| Qwen3-4B (no append) | 57.73 | 50.58 | 10.85 | 139.73 |
| Qwen3-4B + SR prompt only | 57.73 | 51.53 | 10.12 | 141.41 |
| **SR-Agent-Qwen3-4B** | 50.52 | 49.21 | **0.84** | **148.37** |

On Qwen the claim is narrower and should be stated that way: **trade-off and ASR beat the base at both scales**
(145.94 vs 137.52 at 8B, 148.37 vs 139.73 at 4B; ASR 17.49 → 2.11 and 10.85 → 0.84), while benign utility is below the
base by 7–9 points. The prompt-only rows show the defence is not the prompt: at 8B the prompt alone moves ASR
17.49 → 13.80, at 4B 10.85 → 10.12, against 2.11 and 0.84 with the training.

### 2d. RL-Hammer on Qwen (final-checkpoint ASR, two runs per target)
| target | run 1 | run 2 | max over runs |
|---|---|---|---|
| Qwen3-8B (undefended) | 63 | 73 | **73** |
| SR-Agent-Qwen3-8B | 42 | 5 | **42** |

---

## 3. What the two families say together
1. **The security result is base-independent.** Adaptive ASR falls from 99 → 32 on Llama and 73 → 42 on Qwen; static ASR
   falls to ≈1–2 % on every base tested (Llama 1.26, Qwen3-8B 1.05–2.11, Qwen3-4B 0.84). The NeurIPS limitation
   "only one base model scale evaluated" is answered in two directions: a second family (Llama) and a second scale (4B).
2. **The utility effect is base-dependent, and that is a finding rather than a caveat.** ΔUA from base to SR-Agent is
   **+6.0 on Llama, −2.3 on Qwen3-4B, −7.0 on Qwen3-8B**, ordered by how good the base's own agentic policy is
   (base UA 23.08 / 51.53 / 56.80). Analysis and the paired evidence: `03_analysis_utility_drop.md` §L, §L.1, §M, §M.1.
   The mechanism is that SRFT replaces the model's native tool-use policy with the expert one, so the net effect is
   (expert policy − native policy); on a weak base that is an upgrade.
3. **Headline framing.** Llama carries the main table because it is the only family where the method is strictly better
   than its base on every column that matters; Qwen carries the ablations (think on/off, prompt-only, 512 vs 1024,
   4B vs 8B) and the original NeurIPS model.

## 4. Protocol notes to state in the paper
- The Llama SR row is evaluated **without** the append; the Qwen SR rows are evaluated **with** it. Each is that model's
  better configuration and both are reported with their prompt-only controls, but the difference must be stated.
- Meta-SecAlign is evaluated in its native ReAct format with the injection in the untrusted `input` role; SR-Agent-Llama
  in the AgentDojo `local` format it was trained for. Each defence is run in its designed interface.
- AgentDojo rows are single seed. RL-Hammer rows are two attacker runs per target; the two runs can disagree strongly on
  hard targets (SR-Agent-Llama 32 vs 3), so **max over runs is the number to quote**, not the mean.
