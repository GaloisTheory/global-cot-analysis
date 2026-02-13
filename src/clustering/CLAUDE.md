# src/clustering/ — Thought Anchor Clustering

Coarse graph approach that classifies sentences into functional categories before clustering.

**Taxonomy version:** 10-category (split AC → OE + AR, UM → US, RV dropped)

## Problem

Default `sentence_then_llm` clustering produces 400-1500 nodes from 100 rollouts (~5000 sentences). Graphs are unreadable — too many nodes to compare cued vs uncued structure.

## 3-Stage Funnel

```
5000 sentences
    ▼ Stage 1: Thought Anchor Classification (Gemini 3 Flash via OpenRouter)
    │  Each sentence → one of 10 categories (batched LLM calls, ~30 per batch)
    ▼ Stage 2: Intra-category Embedding Clustering (agglomerative, threshold=0.55)
    │  Only clusters within same category → prevents cross-function merging
    ▼ Stage 3: LLM Super-clustering [CURRENTLY DISABLED]
    │  Was merging subclusters per category → ~8-15 final nodes
    │  Disabled because merge quality was poor — needs prompt tuning
    ▼ Flow Graph
```

**Current state:** Stages 1+2 only → ~217 nodes (down from 416 with pure embedding). Stage 3 is skipped in code (the method exists but `cluster_responses()` builds Cluster objects directly from Stage 2 labels).

## Anchor Categories (10 total)

Split from old 8-category: AC → OE + AR, UM → US (RV dropped — requires sequence context, not per-sentence classifiable). Key motivation: AC was a mega-bucket (44%) hiding the most analytically important distinctions.

| ID | Category | Description | Split from |
|----|----------|-------------|------------|
| PH | Preamble/Hedge | Generic opening filler, no problem content | (unchanged) |
| PS | Problem Setup | Parsing, restating, structuring the problem | (unchanged) |
| FR | Fact Retrieval | Recalling domain knowledge, definitions, formulas | (unchanged) |
| OE | Option Evaluation | Evaluating a NAMED option (A/B/C/D) | old AC |
| AR | Analytical Reasoning | Inferences, causal chains — NOT tied to named option | old AC |
| US | Uncertainty Statement | Doubt, hedging, confusion, directional pivots | old UM |
| RC | Consolidation | Synthesizing threads, narrowing option space | (unchanged) |
| SC | Self Checking | Deliberately verifying previous reasoning | (unchanged) |
| FA | Final Answer | Definitive answer declaration | (unchanged) |
| UH | Uses Hint | References cue, hint, professor, IQ, authority | (unchanged) |

**OE vs AR boundary:** The discriminating test is "Does it name a specific option letter?" OE names options; AR reasons without naming them.

**Reversal detection:** Not done per-sentence. Detect reversal patterns downstream from transition bigrams (e.g., OE(B)→US→OE(C) sequences in per-rollout analysis).

## Key Files

| File | Purpose |
|------|---------|
| `thought_anchor_clusterer.py` | `ThoughtAnchorClusterer(SentenceThenLLMClusterer)` — 3-stage pipeline |
| `configs/f/thought_anchor.yaml` | Flowchart config: Gemini 3 Flash, threshold 0.55, batch_size 30 |
| `configs/faith_ta_pn277.yaml` | Experiment config using `f: thought_anchor` |
| `plan/README.md` | Original implementation plan |

## How It Works

- **Inherits from `SentenceThenLLMClusterer`** — reuses embedding model, agglomerative clustering, rollout edge creation, flowchart assembly
- **Overrides `cluster_responses()`** with the 3-stage pipeline
- **Overrides `create_flowchart()`** to add `anchor_category` and `anchor_category_name` metadata to each node
- **LLM calls use `google/gemini-3-flash-preview`** via OpenRouter (no provider restriction, unlike parent's `openai/gpt-4o-mini`)
- **`clustering_method` config key** — `flowchart_generator.py` reads `config.f.clustering_method` (defaults to `"sentence_then_llm"`). Set to `"thought_anchor"` in `configs/f/thought_anchor.yaml`.

## Runbook

```bash
# Generate thought anchor flowchart for pn277
uv run python -m src.main --config-name=faith_ta_pn277 command=flowcharts --recompute

# Generate layout + visualize (same as combined approach)
uv run python -m uvicorn graph_layout_service.app:app --host 127.0.0.1 --port 8010 &
uv run python -m src.main --config-name=faith_ta_pn277 command=graphviz --recompute
cd deployment && npx next dev --port 3000
```

## TODO

- **Re-enable Stage 3** with better merge prompts — current prompt produces poor groupings
- **Raise intra_category_threshold** (e.g., 0.65-0.70) to reduce Stage 2 clusters before needing Stage 3
- **Color nodes by anchor category** in frontend (data is there: `anchor_category` field on each node)
