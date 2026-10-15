# Novelty check for paper 3 (injected-thought reports in Qwen3)

Checked 2026-10-15, before the manuscript was written.

**Scope of the check**
- arXiv 2609.35108, which had been flagged as close to this work.
- Web and arXiv searches for concept-injection, injected-thought and introspection-report studies in open-weight models, and for yes-bias controls, chain-of-thought and J-space replications.
- Searches covered March to October 2026. Nothing dated October 2026 turned up.

Sources were read through their arXiv abstract or HTML pages, or their repository README. Claims about them below are limited to what those pages state.

## Verdict

The paper is still worth publishing, but two of its headline claims are not new as behavioural claims and have to be reframed:

- **(a)** Detecting that something was injected without knowing what it was is already published as "content-agnostic introspection" (Lederman & Mahowald 2026).
- **(b)** That yes/no injection reports in small models are a generic shift toward "yes" is already shown in Llama-3.1-8B (Godet 2025; Hahami et al. 2026a, 2026b).

What is new is the mechanism and the controls around it:

- A single direction H, estimated from random disturbances without any word, carries most of the report and also pushes plainly false questions toward "yes". Holding H removes the report at 1.7B and 8B and most of it at 4B and 14B.
- The injected word's identity is decodable at the answer position, while the reports almost never identify it.
- Content placed directly in the J-space is almost never reported (1 of 160 answers), and its apparent 8B effect depends on the matching criterion.
- Chain of thought adds denial, not access.
- The deliberate-gibberish goal study.

The paper presents (a) and (b) as replications in Qwen3 and builds on them.

## Closest prior work and how the paper cites it

| Work | What it shows | Overlap with paper 3 | How the paper uses it |
|---|---|---|---|
| Lindsey (2025; arXiv 2601.01828) | The concept-injection paradigm. Claude Opus 4/4.1 sometimes detects and names injected concepts. | Source of the paradigm. | Cited as the origin of the paradigm. Our negative results are for small Qwen3 models and say nothing about Claude. |
| Gurnee et al. (2026; arXiv 2607.15495) | The Jacobian lens and J-space. Claude reports injected J-lens concepts in most trials. J-space ablation spares shallow tasks, and chain of thought rescues GSM8K. | We use their lens, workspace band, verbal-introspection protocol and word list unchanged, and test their workspace→report claim in open models. | Cited for the method and as the claim under test. The difference in model scale is stated in the Limitations. |
| Godet (2025, LessWrong "Introspection or confusion?") | In small open models, steering raises "yes" to "Do you believe that 1+1=3?" as much as to the introspection question. | Same yes-bias confound, informal. | Cited as the first report of the confound. |
| Hahami, Sinha, Jain, Kaplan & Hahami (2026a; arXiv 2512.12411) | In Llama-3.1-8B-Instruct, binary detection is explained by a global affirmative logit shift: control r = 0.999, net signal −0.01 ± 0.03. Localization and strength comparison show partial introspection. | **Behavioural yes-bias claim anticipated.** | Cited as prior evidence for the yes-bias. Our contribution is the direction that carries it, its necessity, and that it holds in Qwen3 at four sizes. |
| Hahami, Sinha & Jain (2026b; arXiv 2607.14111, IFT) | The same confound, used to motivate fine-tuning small models to introspect. | As above. | Cited with 2026a. |
| Lederman & Mahowald (2026; arXiv 2603.05414) | Qwen3-235B-A22B (thinking off) and Llama-3.1-405B detect injections but seldom identify them, and confabulate concrete words ("apple"). Little general yes-bias on unrelated facts, but a prompt-specific bias. | **"Detects something, not what" anticipated.** Their large-Qwen3 result shows no general yes-bias, unlike ours in 1.7B–14B. | Cited as prior evidence for content-agnostic detection. The yes-bias difference is reported as a difference in model size and in control (their steering vectors vs our H added alone), not as a contradiction. |
| Macar et al. (2026; arXiv 2603.21396) | Mainly Gemma3-27B. "Evidence carrier" features suppress late "gate" features that default to "No". Detection is not one linear direction on the injection side and emerges with DPO. Ablating refusal directions raises detection from 10.8% to 63.8%. | Closest mechanism. Their gate is a default-"No" readout. Our H is a readout-stage direction whose hold erases the report. | Cited. The paper states that H identifies the last stage, not where detection is computed, which is compatible with their two-stage circuit. Their refusal result is related to our finding that thinking adds denial. |
| Zou, Sun, Kong & Wang (2026; arXiv 2609.35108) | Qwen3-4B-IT, Llama-3.1-8B-IT and Gemma-3-12B-IT. The task is to localize which of ten positions was injected. A set of 32 middle-layer "gate" heads decides whether any position is reported, and "router" heads choose which one. Norm-matched random vectors give near-zero localization. No yes-bias control, no chain of thought, no decoding of concept identity. | A small gating component decides *whether* a report is made. That parallels H, in a different task (localization rather than yes/no) and at a different unit (heads rather than a residual direction). | Cited as convergent evidence for a report gate. The new parts in our paper are that the gate is a general yes-bias that also moves false questions; that random directions are detected as well as words in the yes/no task; and that the injected identity reaches the answer position unused. |
| Pearson-Vogel et al. (2026; arXiv 2602.20031) | Qwen2.5-Coder-32B. The injection signal is visible in the logit lens while sampled answers deny it. Always-yes/no controls are flat. | Resembles our 8B "subthreshold shift that never changes an answer". | Cited. |
| Ferrara (2026; arXiv 2608.20569, OWMI) | 8 open models. Self-reports of altered computation are at chance (AUROC ≈ 0.50), while linear probes on the same activations recover the alteration (75–96%). Controls are impact-matched random perturbations. | **The "state present, report absent" gap is anticipated** in a different paradigm. Impact-matched controls are not new. | Cited. Our "late reporting bottleneck" is presented as a specific instance in which the injected *identity* is decodable at the answer position (40-way, ≈1.0) while the report reads a content-free direction. |
| Singh, Linzen & Ravfogel (2026; arXiv 2605.26242) | Detection in steering paradigms may be general anomaly sensitivity: models cannot tell input-level from activation-level interventions. | Consistent with "disturbance, not content". | Cited. |
| Rivera & Africa (2025; arXiv 2511.21399) | Fine-tuning gives steering awareness. | Background. | Cited briefly. |
| Flo (2026, NeurIPS workshop; github.com/pgrindehollevik-harvard/jspace-validity) | J-space ablation damages next-token prediction more than matched random ablation at every measured Qwen3 scale. | Our J-space-removal "phantom" is specific only at 1.7B, and no larger than a same-size random push from 4B up. | Cited as consistent. |
| SteerCheck (arXiv 2608.24335); Ferrara (2026) | KL- or impact-matched random-direction nulls. | We use matched-KL comparisons. | Matching is not claimed as new. The contribution is reporting all three matching criteria and showing that the sign depends on the choice. |
| Chen et al. (2026; arXiv 2602.13517) | Deep-thinking ratio (DTR). | Used as a descriptive effort measure in the gibberish study. | Cited for the measure. |

**Deliberate gibberish.** No study was found that isolates a goal direction for intentionally meaningless output. The nearest work is Gurnee et al.'s "ignore X" / white-bear observations and their appendix on lexical suppression, which our suppression test mirrors. The section is new, and is presented as exploratory.

**Chain of thought / thinking mode.** None of the injected-thought papers above tests a thinking budget. Lederman & Mahowald run Qwen3-235B with thinking off. Zou et al. score only the first output token.

## Claim-by-claim decisions

1. **Detects a disturbance, not its content.** Not new. Reframed as a replication in Qwen3 1.7B–14B, with same-size random directions as the control. Citations: Lederman & Mahowald; Singh et al.
2. **The report is a yes-bias.** The behavioural part is not new. Reframed: the known confound is carried by one direction H that is estimated without words and is:
   - the main route for the report at every size, held at its clean value. The share of word trials above clean falls from about 1.0 to 0.00 at 1.7B and 8B, to 0.47 at 4B and to 0.55 at 14B, so it need not be the only route at 4B and 14B;
   - sufficient to push false questions toward "yes";
   - not the same as a directly estimated yes direction Y.

   Citations: Godet; Hahami et al. (2026a, b).
3. **The word reaches the answer but not the report (late reporting bottleneck).** Partly anticipated by the state–report gap (Ferrara; Pearson-Vogel et al.). Kept, framed as a localised instance: identity decodable at the answer position under H-hold, and the report erased by holding H only in the last layers.
4. **Workspace content is almost never reported; the 8B effect depends on matching.** New. KL matching itself is attributed to prior work.
5. **Chain of thought adds denial, not access.** New.
6. **The J-space-removal phantom is non-specific beyond 1.7B.** New observation. Flo (2026) is cited as consistent.
7. **Deliberate-gibberish goal direction.** New, and exploratory.

## Consequences for the manuscript

- The title states the mechanism, not "models cannot introspect": *The Injected Word Reaches the Answer but Not the Report: Injected-Thought Reports in Qwen3 Run Mainly Through a Content-Free Yes Direction*. It was revised after review from "The Injected Word Arrives but Is Not Read", because decodability does not show non-use.
- The abstract and introduction credit the yes-bias and content-agnostic findings to prior work in the first paragraph that mentions them.
- The paper does not claim to adjudicate global-workspace vs higher-order theories. The theory framing from the results page is reduced to one paragraph that motivates the workspace test.
- All claims stay Qwen3-specific.
