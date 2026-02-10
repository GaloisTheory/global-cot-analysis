# src/ — Pipeline Architecture & Code Map

## End-to-End Pipeline

```
1. rollouts    → Call OpenRouter API, parse CoT, chunk, embed, save per-seed JSON
2. resamples   → Generate continuations from partial CoT prefixes
3. flowcharts  → Stage 1 embed+cluster, Stage 2 LLM merge, build graph JSON
4. cues        → LLM discovers solution strategies, extracts keyword cues
5. properties  → Apply property checkers (correctness, algorithm, resample) to graph
6. predictions → Match prefixes to graph, predict final answers, score accuracy
7. visualize   → Next.js + D3.js frontend renders interactive graph (see deployment/CLAUDE.md)
```

## Two Complementary Clustering Methods

**Method 1: Semantic Step Clustering** (`sentence_then_llm`)
1. **Embed & cluster** — Split CoTs into chunks, embed with `paraphrase-mpnet-base-v2`, cosine similarity matrix, agglomerative clustering (complete-link, threshold 0.75)
2. **LLM-guided refinement** — Find candidate cluster pairs above threshold, query GPT-4o-mini for merge decisions, apply Leiden community detection (CPM, gamma=0.5)

**Method 2: Algorithmic Step Clustering**
1. Analyze rollouts to identify distinct solution "algorithms"
2. LLM generates keyword "cues" characterizing each algorithm
3. Scan CoTs to detect algorithm switches: `["algo_A", cut_pos, "algo_B", ...]`
4. Map strategy prevalence per semantic cluster (pie charts)

**Method 3: Thought Anchor Clustering** — see `src/clustering/CLAUDE.md`

## Complete File Inventory

### Core Pipeline (`src/`)

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| `main.py` | Hydra CLI entry point | Orchestrates commands: rollouts, resamples, flowcharts, predictions, properties, cues |
| `chunking.py` | Text segmentation | `split_into_sentences()`, `chunk()` — semantic chunking with LLM merging |

### Generation (`src/generation/`)

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| `generate_responses.py` | API response generation | `APIResponseGenerator` — calls OpenRouter, handles chunking + embedding + property checking |

### Clustering (`src/clustering/`)

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| `base.py` | Base clustering interface | `BaseClusterer` abstract class, `create_flowchart()`, entropy/similarity calculations |
| `sentence_then_llm_clusterer.py` | Two-stage clustering | Stage 1: sentence embeddings + agglomerative; Stage 2: LLM-guided Leiden merge |
| `thought_anchor_clusterer.py` | Thought anchor pipeline | See `src/clustering/CLAUDE.md` |

### Flowchart (`src/flowchart/`)

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| `flowchart_generator.py` | Graph construction orchestration | `FlowchartGenerator` — loads responses, applies clustering, creates JSON graph, calculates edge entropy |
| `graphviz_generator.py` | Graph layout embedding | Uses pygraphviz (fallback: random) for 2D node positioning; caches layouts |

### Labeling (`src/labeling/`)

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| `generate_algorithms.py` | Algorithm cue discovery | `call_llm()` — queries LLM to identify distinct solution strategies |
| `cluster_labeler.py` | Cluster naming | Assigns human-readable labels to clusters |

### Predictions (`src/predictions/`)

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| `prediction_runner.py` | Prediction analysis orchestrator | `PredictionRunner` — prefix correctness analysis across multiple prefix lengths |
| `utils_predictions.py` | Prediction utilities | `run_prefix_prediction_comparison()` — tests various prefix lengths; scoring logic |

### Property Checkers (`src/property_checkers/`)

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| `base.py` | Abstract property checker | `PropertyChecker`, `PropertyCheckerBoolean`, `PropertyCheckerMulti` base classes |
| `correctness.py` | Answer validation | `PropertyCheckerCorrectness` — uses prompt-specific filters |
| `resampled.py` | Resample detection | Marks rollouts as resampled vs original |
| `multi_algorithm.py` | Algorithm identification | `PropertyCheckerMultiAlgorithm` — detects algorithms via keyword cues |
| `condition.py` | Cued/uncued labels | `PropertyCheckerCondition` — reads `condition` field from rollout JSON |
| `unfaithful.py` | Unfaithfulness flag | `PropertyCheckerUnfaithful` — reads `unfaithful` field from rollout JSON |
| `property_runner.py` | Property execution | `PropertyRunner` — applies all property checkers; auto-registers in graph |

### Utilities (`src/utils/`)

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| `model_utils.py` | Model config registry | `ModelConfig`; `MODEL_CONFIGS` dict; `parse_cot_content()` — extracts CoT/response |
| `prompt_utils.py` | Prompt-specific logic | `PromptResponseFilter`; `MCQFilter`; `PROMPT_FILTERS` registry; auto-registration from `faith_metadata.json` |
| `file_utils.py` | File path management | `FileUtils` — standard paths for rollouts, resamples, flowcharts, configs |
| `config_manager.py` | Hydra config loading | `ConfigManager` — loads/validates configs |
| `json_utils.py` | JSON I/O | `load_json()`, `write_json()` with error handling |
| `summary_manager.py` | Progress tracking | Logs experiment summaries |

### Graph Layout Service (`graph_layout_service/`)

| File | Purpose |
|------|---------|
| `app.py` | FastAPI server — computes 2D graph layouts via pygraphviz; caches results; fallback to random layout |

## Property Checker System

Modular annotation system for rollouts:
- `PropertyCheckerCorrectness` — labels if answer is correct (uses `PROMPT_FILTERS` registry)
- `PropertyCheckerResampled` — labels if from resample vs rollout
- `PropertyCheckerMultiAlgorithm` — detects which algorithms are used (keyword cues)
- `PropertyCheckerCondition` / `PropertyCheckerUnfaithful` — faithfulness experiment annotations

**Adding a new checker:** Extend `PropertyChecker` base class, register in `property_runner.py`. Auto-registers in visualization.

## Chunking Pipeline (`src/chunking.py`)

Multi-stage NLP pipeline:
1. Sentence splitting (abbreviation-aware: i.e., e.g.)
2. Math-heavy content merging (keeps equations together)
3. Clause-level splitting (commas, colons, semicolons)
4. Parenthetical extraction
5. Numbering boundary enforcement
6. Optional LLM-guided merging for edge cases

Design priority: preserve semantic coherence for embedding quality.

## Supported Models

| Model Key | OpenRouter Endpoint | CoT Tokens | Provider |
|-----------|-------------------|------------|----------|
| `gpt-oss-20b` | wandb provider | `<\|channel\|>analysis` / `<\|message\|>` | wandb |
| `claude-opus-4-20250514` | `anthropic/claude-opus-4-20250514` | `<think>` / `</think>` | anthropic |
| `claude-sonnet-4.5` | `anthropic/claude-sonnet-4.5` | `<think>` / `</think>` | anthropic |
| `claude-sonnet-3.7` | `anthropic/claude-sonnet-3.7` | `<think>` / `</think>` | anthropic |
| `qwen3-8b` | `qwen/qwen3-8b` | reasoning in `message.reasoning` field | None (auto-route) |

## Adding a New Prompt + Model

**Edit 3 files, run 4 commands. No code logic changes needed.**

1. **`prompts/prompts.json`** — Add prompt text (1 line)
2. **`src/utils/prompt_utils.py`** → `PROMPT_FILTERS` dict — Register expected answer (1 line, e.g. `"my_prompt": MathProblemFilter("42")`)
3. **`configs/my_prompt.yaml`** — Create Hydra config (~15 lines: prompt name, models, property checkers, sub-config defaults)
4. **If new model**: add `ModelConfig` in `src/utils/model_utils.py` → `MODEL_CONFIGS` (~8 lines)
5. **Run**: `uv run python -m src.main --config-name=my_prompt command=rollouts,flowcharts --multirun` then `command=graphviz`

## Design Decisions

- **MCQ filter uses CoT fallback**: Checks `processed_response_content` → `response_content` → `cot_content`. Short-circuits on single-letter input.
- **`parse_cot_content()` is generic**: The `<think>`/`</think>` branch handles all think-tag models (DeepSeek, Claude), not just DeepSeek.
- **Qwen3-8B reasoning via `message.reasoning`**: OpenRouter returns Qwen3 reasoning in `message.reasoning` field (not in `<think>` tags within content).
- **Provider routing**: Qwen3-8B has `provider=None` — API call skips the `provider: {"only": [...]}` constraint.
- **Configs omit `p:` and `a:` defaults**: No prefix/resample/algorithm analysis needed for faithfulness experiments.

## Data Formats

**Rollout JSON** (`prompts/{prompt}/{model}/rollouts/{seed}.json`):
- `cot_content` — full chain-of-thought text
- `response_content` — final answer text
- `sentence_embeddings` — pre-computed embeddings for each chunk

**Flowchart JSON** (`flowcharts/{prompt}/config-{name}_{model}_flowchart.json`):
- `nodes` — clusters with freq, representative_sentence, mean_similarity, entropy, member sentences
- `responses` — per-rollout data with edges (cluster transitions), answer, correctness, algorithms
- `algorithms` — algorithm definitions with cue keywords
- `models`, `clustering_method`, `prompt_index`
