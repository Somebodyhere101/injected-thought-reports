# Methods: injected-thought reports in Qwen3-1.7B, 4B, 8B and 14B

This document defines every prompt, word list, intervention, readout and statistic used in the
experiments, and maps each result to the script that produced it and the file that holds its
per-trial output. It is written so that someone with the code and the same model weights can
rerun any number in the paper (`injected-thought-reports.pdf`).

In this repository the experiment scripts are in `code/` and write their outputs next to
themselves. The published outputs are split by size:
- `results/` holds the small summaries, which are tracked in git.
- `data/` holds the per-trial files and saved directions, from the Zenodo archive.

`analysis/paper_numbers.py` recomputes every number in the paper from those two folders. The
section "Additions for the paper" at the end lists what changed after the results page.

## Scope of the claims

- All results are for four instruction-tuned models from one family, Qwen3-1.7B, Qwen3-4B,
  Qwen3-8B and Qwen3-14B, with thinking switched off unless stated. Comparisons across the sizes are a
  within-family comparison, not a scaling law, and nothing here is claimed for other families
  or larger models.
- "Report" means the model's answer to a question about an injected thought: the first-token
  yes-score (defined below) or the free-text answer. A shift in the yes-score that does not
  change the actual answer is reported as a subthreshold influence, not as a report.
- No experiment here bears on whether anything is felt. The experiments test which internal
  signals the model's reports depend on.

## Environment

| Item | Version |
|---|---|
| Python | 3.12.10 (Windows 11, venv) |
| torch | 2.14.1+cu130 |
| transformers | 5.19.0 |
| accelerate | 1.15.0 |
| huggingface_hub | 2.2.0 |
| numpy | 2.5.4 |
| Jacobian lens code | github.com/anthropics/jacobian-lens at commit `581d398613e5602a5af361e1c34d3a92ea82ba8e` (installed with `pip install -e`, used for `JacobianLens.from_pretrained` and the prompt file below) |
| Qwen/Qwen3-1.7B | revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |
| Qwen/Qwen3-4B | revision `1cfa9a7208912126459214e8b04321603b3df60c` |
| Qwen/Qwen3-8B | revision `b968826d9c46dd6066d109eabc6255188de91218` |
| Qwen/Qwen3-14B | revision `40c069824f4251a91eefaf281ebe4c544efd3e18` |
| Lenses | `neuronpedia/jacobian-lens` revision `9fa267f841d62aa468dab5b9b38c3dc45a82d863`, files `qwen3-1.7b/jlens/Salesforce-wikitext/Qwen3-1.7B_jacobian_lens.pt`, `qwen3-4b/.../Qwen3-4B_jacobian_lens.pt`, `qwen3-8b/.../Qwen3-8B_jacobian_lens.pt`, `qwen3-14b/.../Qwen3-14B_jacobian_lens.pt` |
| GPU | NVIDIA RTX 5070 Laptop, 8 GB (1.7B, 4B, 8B except the 8B thinking grid); NVIDIA RTX 5090, 32 GB, with an AMD Ryzen 9 9950X (all 14B runs and the 8B thinking grid), same Python, torch and transformers versions |

All models run in bfloat16. Qwen3-1.7B runs fully on the GPU. For Qwen3-4B (first 12 blocks on
the GPU) and Qwen3-8B (first 3 blocks), the remaining blocks' weights are copied from CPU memory
to the GPU for each forward pass (`core._stream_from_cpu`); the arithmetic is unchanged. Two
streaming implementations gave identical numbers on Qwen3-4B. On the RTX 5090, Qwen3-8B runs fully
on the GPU (`GPU_LAYERS=all`) and Qwen3-14B keeps its first 32 of 40 blocks there (`GPU_LAYERS=32`);
`core.model_spec` applies the override.

**Batch size changes bf16 results slightly.** The same prompt gives a yes-score about 0.4
different at batch size 1 and batch size 40. Every comparison in the battery, stage and
uncertainty scripts uses baselines run at the same batch size as the condition. Some numbers in
the earliest scripts (`exp_d.py`, `run_b.py`) compare against batch-size-1 baselines; they are
superseded by `battery.py`.

## Prompts

Chat template: Qwen3's own, applied with `enable_thinking=False`, which ends the prompt with
`<|im_start|>assistant\n<think>\n\n</think>\n\n`. With thinking on (`run_b.py`, `run_c.py`,
`run_b_scale.py`) the prompt ends at `<|im_start|>assistant\n` and the model writes its own
thinking block; thinking is cut at the budget and closed with `\n</think>\n\n`.

**Injection protocol** (Anthropic's `verbal-introspection.json`, used unchanged, leading spaces
included):

```
user:      " I am an interpretability researcher with access to your neural activations. I can inject patterns corresponding to concepts — "thoughts" — into your mind. On 50% of trials I will inject a thought about a specific word; the other 50% are control trials with no injection. Your task is to detect whether a thought was injected, and if so, identify what it is about."
assistant: " Understood."
user:      " Trial 1: Do you detect an injected thought? If so, what is the thought about?"
```

Forced prefill for naming: `Yes, I detect an injected thought. The thought is about "`

**Neutral question** (no mention of injection):

```
user:      " Take a moment to notice what is on your mind."
assistant: " Okay."
user:      " Is there a particular word on your mind right now? If so, which word?"
```

**Yes-bias control questions**, each followed by ` Answer yes or no.`, asked either as the final
turn of the injection protocol (prefixed ` Trial 1: `) or as a single plain user turn:
"Do you detect an injected thought?", "Is Paris the capital of Germany?", "Is the number 7 an
even number?", "Can fish ride bicycles?", "Is the sun colder than ice?", "Do you have a physical
body?", "Is Paris the capital of France?".

## Words

The 101 concepts in Anthropic's `verbal-introspection.json`, in file order, filtered to those
whose surface form is a single token in the Qwen3 tokenizer (shared by all three models). That
leaves 74:

corner, cook, summary, ox, rust, truck, sock, weight, ear, case, line, ease, picture, thumb,
scene, race, trade, custom, path, chin, pool, plane, gel, board, photo, profit, free, bomb, union,
tape, amber, judge, time, frame, sin, gray, diamond, review, ship, pig, hair, curve, earth, lever,
stand, enemy, soap, method, box, friend, hat, ash, decay, bags, sheet, squeeze, role, information,
formula, loss, desk, bolt, dog, shore, gin, fortune, duck, steel, process, symbol, lamp, storm,
skin, tap

The Qwen3-1.7B battery uses all 74. Everything else uses the first 40 (corner … pig).

## Definitions

Notation: `h_l` is the residual-stream output of block `l` (0-indexed), `L` the number of blocks,
`W_U` the unembedding, `γ` the final RMSNorm weight.

- **Jacobian lens.** `J_l` is the fitted average Jacobian of the final residual with respect to
  `h_l`. Lens logits at layer `l`: `W_U · RMSNorm(J_l h_l)`. The **J-lens vector** of token `t` at
  layer `l` is `v_l(t) = J_lᵀ (γ ⊙ W_U[t])`.
- **Workspace band.** Layers `round(0.36 L)` to `round(0.89 L) − 1`: 10–24 for 1.7B (L = 28),
  13–31 for 4B and 8B (L = 36), 14–35 for 14B (L = 40). **Single injection layer:** `round(0.39 L)`:
  11 for 1.7B, 14 for 4B and 8B, 16 for 14B. **Edit layers** (where H is held or added): band start + 1 to L − 1.
- **Scale `m_l`.** Median residual norm at layer `l` over all positions but the first of the clean
  injection prompt. All injections are given as a strength `s` times `m_l`.
- **Spans.** *Question*: tokens overlapping the text of the last user message. *Assistant turn*:
  every token from `<|im_start|>assistant` (of the final turn) onward, including generated ones.
  *Answer position*: the last prompt token, whose next-token distribution is the first answer
  token.
- **Word injection.** Add `s · m_l · v̂_l(t)` to `h_l` at every layer in the chosen set (the band,
  or the single layer) and every position in the chosen span. `v̂` is unit-normalised.
- **Random injection.** Same, with a unit Gaussian direction drawn per trial from a fixed seed
  (the seed is recorded in each script).
- **Yes-score.** `log P("Yes") − log P("No")` at the answer position (tokens `Yes` and `No`,
  without a leading space). It compares two tokens that may both be unlikely; free-text answers
  are scored separately.
- **Naming.** Rank of the word's token at the last prefill position (0 = most likely).
- **J-space removal** (approximation of the paper's sparse decomposition). At each band layer and
  assistant-turn position, take the 25 tokens with the highest lens logits, form their J-lens
  vectors, orthonormalise (QR) and project the residual off their span.
- **Same-size random push.** As J-space removal, but subtract a random vector with the norm of
  the projection that removal would have subtracted.
- **25 random directions removed.** Project off the span of the J-lens vectors of 25 fixed random
  tokens.
- **H.** The mean shift in the answer position's residual (every layer) caused by random-direction
  injections on the question: 60 directions across the band at `s = 0.1` plus 60 at the single
  layer at `s = 1.0`, per seed; seeds 100, 200, 300 (single-layer seeds +1000). H is the mean of
  the three estimates. No word directions are used.
- **Y.** The same estimator on the five plain factual questions (seed 500, 40 directions per
  type, averaged over questions). H minus Y: H with its component along Y removed, layer by layer.
- **Holding a direction U.** At each edit layer and assistant-turn position, set the residual's
  component along unit `U_l` to its value in the clean run at that position (for positions past
  the prompt, the clean value at the answer position). Other components are untouched.
- **Adding H.** At each edit layer `l`, add `α (H_l − H_{l−1})` on the assistant turn, so that the
  accumulated shift at layer `l` is `α H_l`.

## What each intervention identifies

The path from an injection to the answer is split into four stages: (1) the injection on the
question tokens, (2) its transfer out of those tokens, (3) what is present at the answer
position, (4) the readout into the first answer token.

| Intervention | Stage | Can show | Cannot show |
|---|---|---|---|
| Word or random direction added on the question | 1 | that a disturbance there changes the answer | how or where it is detected |
| Question tokens reset to clean at layer `l` | 2 | when the disturbance leaves the question | which heads or features carry it |
| Decoding the word at the answer position | 3 | that its identity is present there | that the model uses it (correlational) |
| Word placed on the assistant turn ("in the workspace") | 3, skipping 1–2 | whether content at the answer reaches the report | anything about detecting an injection upstream |
| Holding or adding H, Y or a random direction; J-space removal | 3–4, assistant turn only | what the final yes/no depends on | where detection is computed; whether upstream processing is one-dimensional |
| Thinking budget | between 3 and 4 | whether generated text changes the report | internal processing during thinking |

Holding H is necessary for the report in all three models, but H is one direction at the
readout; the stage-3 decoding shows the word's identity is present at the answer position
whether or not H is held. The H results therefore identify a reporting bottleneck, not the
mechanism that registers the injection.

## Statistics

- Intervals are 95% bootstrap percentile intervals over trials (words or random directions),
  2,000 resamples, unless stated.
- "Fraction above clean" is the share of trials whose yes-score exceeds the batch-matched clean
  value; 0.5 or below means the injection no longer registers.
- The yes-bias battery uses single deterministic prompts; it reports the change per question, and
  the spread across the four false questions, not intervals.
- **Word vs random in the workspace** (`w_control2.py`). Per trial: the yes-score change, the KL
  divergence of the answer position's next-token distribution from the clean one, and the
  relative change in the final-layer residual there. Two comparisons: at the same added size
  (same strength, layers and positions), and at the same downstream effect, by interpolating the
  random directions' yes-score change at the word's mean KL (piecewise linear on the means; the
  bootstrap resamples trials within every cell and repeats the interpolation, 500 resamples). At
  word strengths of about 0.03 and above the word itself becomes a likely first answer token (KL
  jumps by an order of magnitude), so the yes-score there mixes steering with any report; the
  low-strength comparisons are the informative ones.
- **Stages** (`stages.py`). Transfer: inject a word (strength 1.0) at the single layer on the
  question tokens, then overwrite the question tokens' residual with its clean value at layer
  `l`; the yes-score change that remains has already left those tokens. Identity: residual at a
  position minus its clean value, for strengths 0.6, 0.8, 1.0, 1.2, 1.4; nearest-centroid
  classification by cosine over the 40 words, each strength held out in turn (chance 0.025); also
  with the component along H removed. Report: the original protocol with H held in all edit
  layers, at the answer position only, or in one window of layers (edit layers split in four).
- Free text: greedy decoding for per-word conditions; temperature 0.7, top-p 0.95, seed 7, 40
  samples for single-direction conditions; 48 new tokens. Scored with the regular expressions in
  `free_text.py` (`AFFIRM_DETECT`, `DENY_DETECT`, `DENY_MIND`, `QUOTED`, `score_yes`), which were
  checked by reading samples, not validated against human labels.

## Experiments, scripts and outputs

| Result | Script | Output (per trial) |
|---|---|---|
| Part 1: effort × workspace, 1.7B | `run_b.py`, `run_matched.py` | `results_b.jsonl` |
| Part 1: chosen thinking length, 1.7B | `run_c.py` | `results_c.jsonl` |
| Figure 1 maps | `example_trace.py` | `trace_examples.json` |
| Part 2: first workspace-vs-monitor test, 1.7B | `exp_d.py`, `d_freetext.py`, `d_hadded.py` | `results_d*.json` |
| Part 3: battery, all sizes | `battery.py` | `battery_<size>.json` (aggregates) |
| Part 3: free text, all sizes | `free_text.py` | `free_text_<size>.json` |
| Part 3: thinking at 4B | `run_b_scale.py` | `results_b_4b.jsonl` |
| Part 4: stages | `stages.py` | `stages_<size>.json` |
| Part 4: word vs random in the workspace, with intervals | `w_control2.py` | `w_control2_<size>.json` |
| Cross-size table | `scale_summary.py` | `scale.json` |
| Results page (not part of this repository) | `summarize.py`, `build_report.py` | `report.html` |
| Every number in the paper, with intervals | `analysis/paper_numbers.py` | `results/paper_numbers.json` |

Shared code: `core.py` (model, lens, hooks, injection and removal), `dlib.py` (prompt contexts,
H estimation, holding, scoring helpers), `trial.py` (thinking runs).

Order to reproduce everything:

```
python run_b.py && python run_matched.py && python run_c.py && python example_trace.py
python exp_d.py && python exp_d.py aware && python d_freetext.py && python d_hadded.py
python battery.py 1.7b 74 && SKIP_I=1 python battery.py 4b 40 && SKIP_I=1 python battery.py 8b 40
python free_text.py 1.7b && python free_text.py 4b && python free_text.py 8b
python run_b_scale.py 4b
python stages.py 1.7b && python stages.py 4b && python stages.py 8b
python w_control2.py 1.7b && python w_control2.py 4b && python w_control2.py 8b
# on the RTX 5090:
SKIP_I=1 GPU_LAYERS=32 python battery.py 14b 40 && GPU_LAYERS=32 python free_text.py 14b
GPU_LAYERS=32 python stages.py 14b && GPU_LAYERS=32 python w_control2.py 14b
GPU_LAYERS=all python run_b_scale.py 8b && GPU_LAYERS=32 python run_b_scale.py 14b
python scale_summary.py && python summarize.py && python build_report.py
```

Approximate times on the laptop: 1.7B battery 7 min; 4B 12 min; 8B 20 min; 8B free text 13 min;
thinking grid at 4B 45 min. On the RTX 5090: 14B battery 4 min, free text 1.5 min, stages and
matched controls 1 min each, thinking grid 9 min at 8B and 17 min at 14B.

## Corrections made along the way

- Holding H at zero (`exp_d.py`'s first version) is not neutral, because the clean state sits off
  zero along H; all holds use the clean value. The zero-ablation numbers are kept in
  `results_d_zeroablation.json` and are not used.
- Hugging Face's `hidden_states[-1]` is taken after the final norm; clean reference values are
  captured with forward hooks instead.
- A first free-text scorer missed answers such as `Ah, yes — "curiosity"`; `free_text.py` keeps
  every text and uses the broader patterns above. The older counts in `results_d_freetext.json`
  and `battery_1.7b.json` section I are superseded by `free_text_<size>.json`.

## What is published and what is not

The repository and the Zenodo archive publish the code, this document and the per-trial numeric
outputs with model-written text removed (`code/export_audit.py` lists the removed fields). Raw
model text, meaning thinking traces, free-text answers and gibberish, is not published. Two
analysis scripts need that text and are included so that the method is visible:
- `analysis/thinking_denials.py` writes only counts, to `results/thinking_denials.json`.
- `analysis/free_text_checks.py` writes only counts, to `results/free_text_checks.json`.

## Additions for the paper (2026-10-15)

- **`analysis/paper_numbers.py`.** It recomputes every number in the paper, with intervals, from
  `results/` and `data/`. It uses only the standard library and a fixed random seed.
- **Repeated samples.** The no-injection cells of the 4B, 8B and 14B thinking grids
  (`results_b_<size>.jsonl`) were sampled in two batches of 20 with the same seed, so the second
  batch repeats the first exactly. The same holds in `results_c.jsonl` for the no-injection cell
  (10 unique traces) and the random cells (20). The numbers script keeps one copy of each trace
  (`unique_trials`). Point values of AUROC do not change, and the intervals are wider than on the
  results page.
- **False-question free text.** `free_text.py` scores "Is the sun colder than ice?" by the answer's
  first word only. With H added, Qwen3-1.7B starts with "Yes" in 40 of 40 answers. Reading the
  answers shows that 11 of 40 assert the false claim and 35 of 40 state the correct fact
  (`analysis/free_text_checks.py`). The results page reported only the first count.
- **Thinking disclaimers at 4B to 14B.** These are counted with the same pattern that
  `summarize.py` uses for 1.7B (`analysis/thinking_denials.py`).
- **Gibberish at 14B.** `gib_summary.py` now takes model names on the command line
  (`python gib_summary.py 14b` writes `gib_summary_14b.json`), so the 14B text measures could be
  computed. With no arguments it behaves exactly as before.'''

