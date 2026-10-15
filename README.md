# Injected-thought reports in Qwen3

Code and data for **"The Injected Word Reaches the Answer but Not the Report: Injected-Thought Reports in Qwen3 Run Mainly Through a Content-Free Yes Direction"** (Sam Ramdan, October 2026).

- Paper: [`injected-thought-reports.pdf`](injected-thought-reports.pdf), also on Zenodo: `<ZENODO-DOI>`. The DOI is added once the record is published.
- Per-trial data: the Zenodo archive `code-and-data.zip`, under `<ZENODO-DOI>`.
- Model-written texts: the Zenodo archive `model-texts.zip`, same record. It holds the free-text answers, thinking traces and gibberish, screened before release (see its README).
- The paper is the third in a series that asks whether a hidden-state direction tracks the process it appears to track. The earlier papers are on the compute–retrieve axis: [code](https://github.com/Somebodyhere101/compute-retrieve), [version 2](https://doi.org/10.5281/zenodo.23374358), [A Compounding Fingerprint](https://doi.org/10.5281/zenodo.23318679).

## Findings

All results are for four instruction-tuned models from one family: Qwen3-1.7B, 4B, 8B and 14B. Thinking is off unless stated. Nothing here is claimed for other families or larger models, or about experience.

1. **The models detect that something was injected, not what.**
   - A word's Jacobian-lens direction injected on the question raises the first-token yes-score at every size, by +3.5 to +6.9 logits.
   - A random direction of the same added size also raises it.
   - When forced to name the thought, at most 3 words reach the top 10.
   - No free-text answer at any size affirms an injected thought.

   This replicates, in smaller Qwen3 models, the content-agnostic detection of [Lederman & Mahowald (2026)](https://arxiv.org/abs/2603.05414).
2. **One content-free direction carries most of the report, and it is a yes direction.**
   - H is estimated from random disturbances only, never from words.
   - Holding H at its clean value removes the report at 1.7B and 8B. The share of word trials above clean falls 0.97→0.00 and 1.00→0.00.
   - At 4B and 14B it removes the average effect, or about 85% of it at 14B. About half of the word trials stay above clean (1.00→0.47 at 4B, 0.93→0.55 at 14B), so smaller routes that bypass H cannot be excluded at those sizes.
   - Holding a random direction does not do this, and neither does holding a separately estimated yes direction Y.
   - Added alone, H raises plainly false questions toward "yes" by about as much as the introspection question.

   This gives a mechanism for the yes-bias reported behaviourally by [Godet (2025)](https://www.lesswrong.com/posts/kfgmHvxcTbav9gnxe/introspection-or-confusion) and [Hahami et al. (2026)](https://arxiv.org/abs/2512.12411).
3. **The word reaches the answer but not the report.**
   - The injection leaves the question tokens within a few layers.
   - A 40-way decoder identifies the injected word from the answer position with 0.75–1.00 accuracy in the last layer, whether or not H is held.
   - Decodability shows that the information is present, not that no downstream computation uses it. The behavioural fact is narrower: the reports almost never identify the word.
   - In 1.7B and 8B, holding H in the last quarter of layers alone erases the report.
4. **Content placed in the workspace is almost never reported.**
   - A word's J-lens direction on the assistant turn is named in one of 160 free-text answers: once at Qwen3-4B, where 4 of 40 answers affirm a thought.
   - In Qwen3-8B it moves the yes-score more than random directions matched on added size (+10.8 [10.5, 11.1]) or on output KL (+9.2 [7.5, 10.0]).
   - It moves the yes-score less than random directions matched on the change in the final residual (−5.6 [−6.1, −5.0]).
   - It never changes an answer.
5. **Chain of thought adds denial, not access.**
   - 256 tokens of thinking lower detection AUROC at every size (14B: 0.93 → 0.82).
   - Naming does not improve.
   - 90–97% of traces contain capability disclaimers.
6. **Deliberate gibberish (exploratory).**
   - Identical tokens under a gibberish vs a coherent instruction isolate a goal direction G.
   - Removing G shortens or ends the gibberish beyond random directions at the same output change.
   - Real words do not come back. In 1.7B only, the remaining text turns into an ordered alphabet.

[`NOVELTY.md`](NOVELTY.md) lists what is new relative to 2026 work on concept injection, and how the overlapping papers are cited.

## Repository layout

| Path | Contents |
|---|---|
| `code/` | Every experiment script, exactly as run (see "Rerun the experiments"). The shared code is `core.py` (model, lens, hooks, injection, J-space removal), `dlib.py` (prompt contexts, H estimation, holding, scoring) and `trial.py` (thinking runs). |
| `analysis/` | `paper_numbers.py` recomputes every number in the paper with intervals, using the standard library only. `thinking_denials.py` and `free_text_checks.py` derive counts from model text, which is not published. |
| `results/` | Small result summaries: battery, free-text counts, stages, Experiment D, gibberish summaries, `scale.json`, and `paper_numbers.json` (every number in the paper). |
| `data/` | Not in git. Per-trial outputs and saved directions from the Zenodo archive (see [`data/README.md`](data/README.md)). |
| `paper/` | LaTeX source, bibliography, `figures.py` (pgfplots, standard library only) and the figures. |
| `METHODS.md`, `GIB_METHODS.md` | Every prompt, word list, intervention, readout and statistic, and which script produced which output. |

## Environment

| Item | Version used |
|---|---|
| Python | 3.12.10 |
| torch | 2.14.1+cu130 |
| transformers | 5.19.0 |
| accelerate | 1.15.0 |
| huggingface_hub | 2.2.0 |
| numpy | 2.5.4 |
| wordfreq | 3.x (gibberish text measures only) |
| Models | `Qwen/Qwen3-1.7B` @ `70d244cc`, `Qwen/Qwen3-4B` @ `1cfa9a72`, `Qwen/Qwen3-8B` @ `b968826d`, `Qwen/Qwen3-14B` @ `40c06982` |
| GPUs | RTX 5070 Laptop 8 GB (1.7B, 4B, 8B); RTX 5090 32 GB (all 14B runs and the 8B thinking grid) |

All models run in bf16.
- On the 8 GB laptop, 4B and 8B stream most blocks from CPU memory for each forward pass (`core._stream_from_cpu`). The arithmetic is unchanged.
- On a larger GPU, set `GPU_LAYERS=all` to keep the whole model on the GPU, or set it to a number of blocks. The 14B runs used `GPU_LAYERS=32`.

### The Jacobian-lens dependency

The experiments use Anthropic's Jacobian lens code and Neuronpedia's published lens fits.

1. **Code.** Use [github.com/anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens) at commit `581d398613e5602a5af361e1c34d3a92ea82ba8e`. It provides the `jlens` package (`JacobianLens.from_pretrained`) and the protocol file `data/experiments/verbal-introspection.json`, which holds the injection prompt and the 101 concepts. `core.py` reads that file from `third_party/jacobian-lens/` next to `code/`:

   ```bash
   git clone https://github.com/anthropics/jacobian-lens third_party/jacobian-lens
   git -C third_party/jacobian-lens checkout 581d398613e5602a5af361e1c34d3a92ea82ba8e
   pip install -e third_party/jacobian-lens
   ```

2. **Lenses.** The lenses are downloaded automatically from the Hugging Face repository [`neuronpedia/jacobian-lens`](https://huggingface.co/neuronpedia/jacobian-lens). The paper used revision `9fa267f841d62aa468dab5b9b38c3dc45a82d863` and these files:
   - `qwen3-1.7b/jlens/Salesforce-wikitext/Qwen3-1.7B_jacobian_lens.pt`
   - `qwen3-4b/jlens/Salesforce-wikitext/Qwen3-4B_jacobian_lens.pt`
   - `qwen3-8b/jlens/Salesforce-wikitext/Qwen3-8B_jacobian_lens.pt`
   - `qwen3-14b/jlens/Salesforce-wikitext/Qwen3-14B_jacobian_lens.pt`

   The fits are on WikiText. See [neuronpedia.org/jlens](https://www.neuronpedia.org/jlens).

Then install the rest:

```bash
pip install -r requirements.txt
```

## Recompute the paper's numbers and figures (no GPU)

```bash
# 1. download code-and-data.zip from Zenodo and copy its data/ folder here
# 2. every number in the paper, with intervals -> results/paper_numbers.json
python analysis/paper_numbers.py
# 3. figures (needs pdflatex with pgfplots and standalone), then the paper
cd paper && python figures.py && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

`paper_numbers.py` uses a fixed seed, so its output is identical from run to run.

## Rerun the experiments

Run the scripts from `code/`. They write their outputs next to themselves. Times are for the RTX 5070 Laptop unless stated.

| Paper section | Command | Output |
|---|---|---|
| §7, 1.7B thinking grid (effort × workspace) | `python run_b.py && python run_matched.py` | `results_b.jsonl` |
| §7, chosen thinking length, 1.7B | `python run_c.py` | `results_c.jsonl` |
| Appendix C, Experiment D, 1.7B | `python exp_d.py && python exp_d.py aware && python d_freetext.py && python d_hadded.py` | `results_d*.json` |
| §3–§4, battery (detection, H, holds, yes-bias, phantom) | `python battery.py 1.7b 74`; `SKIP_I=1 python battery.py 4b 40`; `SKIP_I=1 python battery.py 8b 40` | `battery_<size>.json`, `dirs_<size>.pt` |
| §3–§4, free text | `python free_text.py <size>` (after the battery) | `free_text_<size>.json` (contains model text) |
| §5, stages | `python stages.py <size>` | `stages_<size>.json` |
| §6, workspace vs random with matched controls | `python w_control2.py <size>` | `w_control2_<size>.json` |
| §7, thinking grid at 4B, 8B, 14B | `python run_b_scale.py <size>` | `results_b_<size>.jsonl` |
| Cross-size summary | `python scale_summary.py` | `scale.json` |
| §8, deliberate gibberish | `python gib.py <size> && python gib_ext.py <size>`, then `python gib_summary.py` (1.7B–8B) and `python gib_summary.py 14b` | `gib_<size>.json`, `gib_ext_<size>.json`, `gib_summary*.json` |

On a 32 GB GPU, the 14B runs and the 8B thinking grid take these commands. Use `set GPU_LAYERS=32&&` on Windows.

```bash
SKIP_I=1 GPU_LAYERS=32 python battery.py 14b 40
GPU_LAYERS=32 python free_text.py 14b
GPU_LAYERS=32 python stages.py 14b
GPU_LAYERS=32 python w_control2.py 14b
GPU_LAYERS=all python run_b_scale.py 8b
GPU_LAYERS=32 python run_b_scale.py 14b
GPU_LAYERS=32 python gib.py 14b && GPU_LAYERS=32 python gib_ext.py 14b
```

Approximate times are 7 min for the 1.7B battery, 12 min for 4B and 20 min for 8B, and 45 min for the 4B thinking grid. On the RTX 5090, the 14B battery took 4 min and the 14B thinking grid 17 min.

After a rerun, recompute the numbers from your own outputs with `python analysis/paper_numbers.py code`. Folders given on the command line are searched first. `pilot.py` and `pilot2.py` are the calibration pilots that chose injection layers and strengths. `analyze.py` and `summarize.py` produced the 1.7B summaries for the results page. `export_audit.py` strips model text from the outputs before publication.

## What is published

The scripts, both methods documents, every numeric output and the directions used (H, Y and G as `.pt` tensors) are published here and in `code-and-data.zip`. The model-written texts are published separately in the Zenodo archive `model-texts.zip`: free-text answers, thinking traces and generated gibberish. Every text went through an automated screen, and a sample from every condition was read before release; the archive's README says exactly what was read. `analysis/build_texts.py` assembled the archive. `code/export_audit.py` strips the texts from the numeric outputs.

## Citation

```bibtex
@misc{ramdan2026injected,
  title  = {The Injected Word Reaches the Answer but Not the Report: Injected-Thought Reports in Qwen3 Run Mainly Through a Content-Free Yes Direction},
  author = {Ramdan, Sam},
  year   = {2026},
  publisher = {Zenodo},
  doi    = {<ZENODO-DOI>}
}
```

## Use of AI assistance

An AI system (Claude, Anthropic), working under the author's direction, designed the experiments, wrote the code, ran the experiments and analyses, and drafted the paper. The author directed the work and is responsible for its content.

## License

MIT (see [`LICENSE`](LICENSE)).
