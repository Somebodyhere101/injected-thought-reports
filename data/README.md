# data/

This folder is not tracked in git. Copy here the `data/` folder from the Zenodo archive
`code-and-data.zip` (DOI: `<ZENODO-DOI>`). `analysis/paper_numbers.py` reads it together with
`results/`.

| File | What it holds |
|---|---|
| `results_b.jsonl` | Qwen3-1.7B thinking grid: injection × budget (0/64/256) × assistant-turn treatment, one row per word |
| `results_b_4b.jsonl`, `results_b_8b.jsonl`, `results_b_14b.jsonl` | Thinking grids at 4B, 8B, 14B (no injection / word; thinking off / 256; untouched / J-space removed / same-size push) |
| `results_c.jsonl` | Qwen3-1.7B, thinking length chosen by the model (cap 1,024) |
| `w_control2_<size>.json` | Word vs random directions on the assistant turn: per-trial yes-score change, KL and final-residual change |
| `gib_<size>.json` | Deliberate gibberish: per-condition measures, identical-token control, causal tests, lexical suppression |
| `trace_examples.json` | J-lens rank maps of four injected words, used on the results page |
| `directions/dirs_<size>.pt` | H, Y (raw, per layer) and the workspace strength used by the battery |
| `directions/gib_G_<size>.pt` | The goal direction G per layer |

Model-written text (thinking traces, free-text answers, generated gibberish) has been removed from
every file (see `code/export_audit.py`). It is published separately as `model-texts.zip` in the
same Zenodo record.
