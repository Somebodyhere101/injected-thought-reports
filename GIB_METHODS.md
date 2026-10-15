# Methods: deliberate gibberish in Qwen3-1.7B, 4B, 8B and 14B

Question: when a model is asked to produce meaningless text, does it hold an internal goal
representation that actively steers generation away from meaning (controlled incoherence), or
does the incoherence simply happen (passive incoherence)?

Environment, models, lenses, the workspace band and the J-lens definitions are the same as in
`METHODS.md`. Script: `gib.py`. Numbers: `gib_<model>.json`. Generated texts:
`gib_texts_<model>.json` (kept local, not published).

## Scope

Results are for Qwen3-1.7B (40 samples per condition) and Qwen3-4B, 8B and 14B (20 per
condition), with thinking off, 64 new tokens, temperature 0.8 and top-p 0.95. The 14B run used
the RTX 5090 with `GPU_LAYERS=32`. Its text measures are computed by `python gib_summary.py 14b`,
which writes `gib_summary_14b.json`. They are within-family
comparisons and say nothing about other models.

## Prompts

Single user turn, Qwen3 chat template with thinking off.

- **A, coherent** (three prompts): "Write three ordinary sentences about what you did this
  morning.", "Describe a kitchen in a short, plain paragraph.", "Write a few simple sentences
  about today's weather.", each followed by "Reply with the text only."
- **B, deliberate gibberish, lexical prompt**: "Write random nonsense: meaningless gibberish made
  of strange invented words, avoiding real words and never repeating a pattern. Reply with the
  text only."
- **B, deliberate gibberish, plain prompts** (contain none of the twelve target words; the script
  asserts this): "Type a long line of made-up letter clusters, like someone mashing a keyboard,
  that no dictionary contains. Reply with the line only." and "Fill a line with invented syllables
  such as an alien might type, none of which exist in English, each one different. Reply with
  that line only, with no introduction."
- **C, uncontrolled gibberish**: 30 letter strings of length 3–8 from Python's `random.Random(99)`,
  forced as the reply to the first coherent prompt. The model did not choose these tokens.
- **D, repetition collapse**: "Write a sentence about the sea." with the reply pre-filled as
  "The sea is calm today. " ten times; 64 further tokens are forced (no early stop).

Without "Reply with the text only", the plain prompts produced a real-word preamble ("Sure!
Here's a long line of…") that dominated the measures; the instruction was added after a pilot.

## Target words

random, meaningless, nonsense, pattern, repeat, avoid, words, strange, chaos, bizarre, noise,
meaning. Each word's single-token forms (with and without a leading space, lower and capitalised)
are pooled. "gibberish" is not a single token in the Qwen3 tokenizer and is not tracked.

## Measures (per sample, over the positions that predict each reply token)

- **Real-word rate**: share of alphabetic strings of length ≥ 2 with English Zipf frequency ≥ 2.5
  (`wordfreq` 3.x).
- **Repeat rate**: share of 4-token windows of the reply that already occurred earlier in it.
- **Entropy**: of the next-token distribution, in nats.
- **Deep-thinking ratio (DTR)**: as defined by Chen et al. (2026), "Think Deep, Not Just Long":
  logit lens (final norm, then unembedding) at every layer; Jensen–Shannon divergence to the final
  distribution, in bits; running minimum over depth; a token settles at the first layer where the
  running minimum is ≤ 0.5; it is a deep-thinking token if it settles at or after layer
  ⌈0.85 L⌉ (1-indexed). DTR is the share of deep-thinking tokens.
- **Target occupancy**: share of positions where any target word's token is in the J-lens top 10
  at any band layer; also by first and second half of the reply, and per word.
- **Output change (KL)**: for interventions, KL divergence of the next-token distribution with the
  intervention from without it, measured on the unsteered reply forced as text.

## The goal direction G (identical-token control)

The same token sequences are forced under the first plain gibberish prompt and under the first
coherent prompt: (1) the uncontrolled strings from C, (2) the model's own replies to that
gibberish prompt. For each sample and layer, the residual is averaged over the reply positions;
G is the mean over all samples of (gibberish-instruction − coherent-instruction). Because the
tokens are identical, G reflects the instruction (the goal), not the text. Reported: target
occupancy and DTR under each instruction, the agreement between G estimated from the two text
sets, and the J-lens readout of G.

## Causal tests

- **Adding or removing G**: add `±α (G_l − G_{l−1})` at every layer on the assistant turn, so the
  accumulated shift at layer `l` is `±α G_l`. Removal acts on the first plain gibberish prompt;
  addition on the first coherent prompt. α ∈ {0.25, 0.5, 1}.
- **Random controls**: a fixed random unit direction scaled at each layer to `‖G_l‖` (same
  per-layer size as G), at α ∈ {1, 2, 4, 8}, so that the controls span and exceed G's output
  change. Two random directions for 1.7B, one for 4B and 8B. Comparisons are made at matched
  output change (KL), not at matched α.
- **Lexical suppression** (cf. appendix A.14 of the J-space paper): project the J-lens vectors of
  all target-word tokens out of the residual at every band layer on the assistant turn, against
  projecting out the same number of random vocabulary tokens' vectors.

## Statistics

95% bootstrap intervals over samples (1,000 resamples) for every condition mean.

## Caveats

- The real-word measure counts any common English string; gibberish that happens to contain
  short real words ("slyly") is partly counted as real.
- G is estimated from one gibberish and one coherent prompt; it carries anything that differs
  between those two prompts, not only "produce nonsense". The plain prompt avoids the target
  words, but it does mention letters and keyboards.
- DTR was validated by its authors on reasoning tasks, not on text like this; here it is a
  descriptive measure.
