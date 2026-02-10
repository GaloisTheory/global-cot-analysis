# 3-Stage Coarse Clustering Pipeline

## Architecture: 3-Stage Funnel

```
5000 sentences
    │
    ▼ Stage 1: Thought Anchor Classification (Gemini 3 Flash)
    │  → Each sentence tagged with one of 8 categories
    │
    ▼ Stage 2: Intra-category Embedding Clustering (lower threshold)
    │  → ~3-8 subclusters per category = 30-80 total clusters
    │  → Constraint: only cluster within same anchor category
    │
    ▼ Stage 3: LLM Super-clustering on centroids (Gemini 3 Flash)
    │  → LLM merges subclusters within each category
    │  → Target: 1-3 nodes per category → ~8-15 final nodes
    │
    ▼ Flow Graph
```

## Thought Anchor Categories (8 total)

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

## Usage

```bash
# Run with thought anchor clustering
uv run python -m src.main --config-name=faith_ta_pn277 command=flowcharts --recompute

# Or use the original sentence clustering
uv run python -m src.main --config-name=faith_combined_pn277 command=flowcharts --recompute
```

## Files

- `src/clustering/thought_anchor_clusterer.py` — ThoughtAnchorClusterer (3-stage pipeline)
- `configs/f/thought_anchor.yaml` — Flowchart config for thought anchor method
- `configs/faith_ta_pn277.yaml` — Experiment config using thought anchor clustering
