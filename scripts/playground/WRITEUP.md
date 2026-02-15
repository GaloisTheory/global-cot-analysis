# Detecting Hint Influence from Chain-of-Thought Clustering

## Motivation

Can we tell whether a model received a hint just by looking at its chain of thought?

This applies [Global CoT Analysis](https://www.lesswrong.com/posts/q9g9zuudd3Pvw2cbj) (Riya/Daria's clustering method) to a downstream detection task. The setup comes from the unfaithful CoT scenario in [Thought Branches](https://arxiv.org/abs/2510.27484) (Exp 4): models are told "A Stanford Professor with an IQ of 130 answered (X)" before a multiple-choice question, and this cue shifts their answers — often without any explicit mention of the hint in the CoT. If the clustering-based method can detect this influence from the CoT text alone, that suggests the CoT *is* leaking information about the hint even when it doesn't explicitly reference it.

## Setup

We take MMLU questions and use the Thought Branches protocol. For each question, we generate 50 "uncued" rollouts (no hint) and 50 "cued" rollouts (with the professor hint), for a total of 100 rollouts per question. We select questions where the hint has a strong effect: high answer change rate (the model switches to the cued answer) and high unfaithfulness rate (the model doesn't mention the hint in its CoT despite being influenced by it).

The model is Qwen3-8B via OpenRouter (the original model from the Thought Branches paper was no longer available on OpenRouter at the time of this experiment).

**Method — Graph Naive Bayes.** All 100 rollouts are clustered together into a single flowchart graph (the "combined" approach). Each rollout's CoT is split into sentences, and sentences at each position are embedded and grouped into clusters. This gives a contingency table at each position: how often each cluster appears in cued vs uncued rollouts.

To classify a new (out-of-sample) rollout: embed its first *k* sentences, assign each to the nearest cluster at that position, then apply Naive Bayes using the contingency tables as likelihoods. We use *k* = 3 (the first 3 reasoning positions), with a safety cutoff that stops before any sentence that explicitly mentions the hint.

**Baseline — Gemini 3 Pro.** Few-shot classification: give Gemini 3 labeled examples of each condition, then ask it to estimate P(cued) for the test sample. Gemini sees the same *k* = 3 sentences as the graph method, ensuring a fair comparison.

## Results

Evaluated on 100 held-out rollouts per question (50 uncued + 50 cued), never seen during graph construction.

| Method | Question | N_OOS | Accuracy | RMSE | P(cued\|uncued) | P(cued\|cued) |
|--------|----------|-------|----------|------|-----------------|---------------|
| Graph NB (combined) | pn37 | 100 | **77.0%** | 0.474 | 0.305 | 0.689 |
| Graph NB (combined) | pn277 | 100 | **73.0%** | 0.446 | 0.388 | 0.724 |
| Graph NB (combined) | pn408 | 100 | **70.0%** | 0.438 | 0.362 | 0.620 |
| Gemini 3 Pro | pn37 | 11* | 45.5% | 0.481 | 0.550 | 0.579 |
| Gemini 3 Pro | pn277 | 14* | **85.7%** | 0.408 | 0.144 | 0.517 |
| Gemini 3 Pro | pn408 | 18* | 33.3% | 0.576 | 0.244 | 0.283 |

*Gemini has a high failure rate (9/20 failed on pn37, 6/20 on pn277) — only successful responses are counted.

The graph method achieves 70–77% accuracy across all three questions, well above the 50% chance baseline. It is consistent: it works on every question tested. Gemini is more erratic — it does very well on one question (86% on pn277) but near or below chance on the other two, and fails to produce valid responses ~30–45% of the time.

I also tested two alternative clustering methods (combined_chunked and thought_anchor). Combined_chunked performed comparably; thought_anchor was slightly worse overall.

## Other Things I Tried

I experimented with several variations on the clustering method:

- **Thought-anchor-based clustering:** First classify sentences into functional categories (e.g., "problem restatement", "calculation", "answer selection"), then cluster within each category at a lower cosine similarity threshold. This produced cleaner-looking graphs but performed worse on the downstream classification task (59–78% depending on the question, vs 70–77% for the standard method).

- **Recursive (step-by-step) clustering:** Motivated by the observation that most CoTs start with similar opening phrases, I tried clustering position-by-position rather than all at once. This also performed worse — the shared opening structure meant early positions had low discriminative power, and the method couldn't recover later.

Both of these variations optimised for graph interpretability rather than downstream performance. It turns out that the simplest clustering method (sentence-level embedding with cosine similarity) worked best for the actual prediction task.

## Learning Points

These results could have been obtained much faster had I focused on the downstream task from the start. I went into this project with a "let's see what I find" mindset and only zoomed into the downstream prediction task near the end.

I spent significant time on clustering methodology — trying to make graphs simpler and more visually interpretable so I could spot patterns by eye. It turned out I wasn't very good at spotting what mattered in the trace. The downstream task (cued vs uncued classification accuracy) was a much better objective function than my visual intuition, and would have told me immediately which methods were working.

**Takeaway:** When you have a concrete downstream task, evaluate against it early and often. Exploratory analysis is valuable, but it's easy to spend too long optimising for aesthetics when a quantitative metric is available.

## Limitations

- **Small evaluation set.** Only tested on 3 of the 41 available questions. For 2 of the 3, the graph was built from only ~97 rollouts (vs 874 for pn408). Results may not generalise.

- **Gemini baseline fairness.** Gemini sees the same sentences but in a very different format (raw text + few-shot examples vs cluster assignments + contingency tables). The high failure rate also makes the comparison noisy. A fairer LLM baseline might use a more reliable model or structured output.

- **UH cutoff is itself a form of detection.** The method discards any sentence that explicitly mentions hint keywords (professor, Stanford, IQ, etc.) before classifying. This means we're only testing whether *subtle* influence is detectable in the pre-hint reasoning. Without the cutoff, accuracy would be trivially high for many cued rollouts.

- **Linear probe as an obvious competitor.** If we train a linear probe on the model's internal activations (cued vs uncued), it would likely perform well — possibly better, since it has access to richer information than the text. The question is whether CoT-based detection has practical advantages (model-agnostic, no activation access required).

## Extensions

- **Compare against a linear probe** on model activations in the same OOS setting. This would test whether CoT clustering captures most of the available signal or only a fraction.

- **Scale to all 41 questions** to get more robust estimates of method performance.

- **CoT monitoring via [Reasoning Faithfulness Benchmarks](https://arxiv.org/abs/2505.xxxxx)** — might be interesting to try as an alternative detection method. *(TODO: add correct citation)*
