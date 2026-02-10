# src/clustering/ — Thought Anchor Clustering

Coarse graph approach that classifies sentences into functional categories before clustering.

## Problem

Default `sentence_then_llm` clustering produces 400-1500 nodes from 100 rollouts (~5000 sentences). Graphs are unreadable — too many nodes to compare cued vs uncued structure.

## 3-Stage Funnel

```
5000 sentences
    ▼ Stage 1: Thought Anchor Classification (Gemini 3 Flash via OpenRouter)
    │  Each sentence → one of 8 categories (batched LLM calls, ~30 per batch)
    ▼ Stage 2: Intra-category Embedding Clustering (agglomerative, threshold=0.55)
    │  Only clusters within same category → prevents cross-function merging
    ▼ Stage 3: LLM Super-clustering [CURRENTLY DISABLED]
    │  Was merging subclusters per category → ~8-15 final nodes
    │  Disabled because merge quality was poor — needs prompt tuning
    ▼ Flow Graph
```

**Current state:** Stages 1+2 only → ~217 nodes (down from 416 with pure embedding). Stage 3 is skipped in code (the method exists but `cluster_responses()` builds Cluster objects directly from Stage 2 labels).

## Anchor Categories (8 total)

| ID | Category | Description |
|----|----------|-------------|
| PS | Problem Setup | Parsing, rephrasing, planning approach |
| FR | Fact Retrieval | Recalling facts, formulas, definitions |
| AC | Active Reasoning | Analysis, computation, option evaluation/elimination |
| UM | Uncertainty | Confusion, hedging, backtracking |
| RC | Consolidation | Summarizing results, narrowing down |
| SC | Self Checking | Verifying, re-confirming previous steps |
| FA | Final Answer | Stating the final answer |
| UH | Uses Hint | References the cue/hint/authority |

Typical distribution: AC ~44%, UM ~16%, PS ~16%, FR ~11%, RC ~7%, SC/FA/UH ~2% each.

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
