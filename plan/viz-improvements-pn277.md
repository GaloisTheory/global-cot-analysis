# Visualization Improvements for pn=277 Faithfulness Graph

## Changes Made

### 1. Sentence-level clustering (`configs/f/combined.yaml`)
- `sentences_instead_of_chunks: false` → `true`
- Uses `sentences` field (65 per rollout, split on `.?!\n`) instead of `chunked_cot_content` (110 fragments from 7-stage clause-splitting pipeline)
- Expected: ~613 clusters → significantly fewer, with cleaner labels (complete reasoning steps)

### 2. Directed left-to-right layout (`graphviz_generator.py` + `app.py`)
- Engine: `sfdp` → `dot` (respects edge direction, topological ordering)
- Graph: `directed=False` → `directed=True`
- Added `rankdir="LR"` — START flows left-to-right toward answer nodes
- Added BFS rank grouping: nodes at same distance from START aligned vertically via `rank="same"` subgraphs

## Files Modified

| File | Change |
|------|--------|
| `configs/f/combined.yaml` | `sentences_instead_of_chunks: true` |
| `src/flowchart/graphviz_generator.py:111` | Engine `sfdp` → `dot` |
| `graph_layout_service/app.py:104` | `directed=True` |
| `graph_layout_service/app.py:157` | `rankdir="LR"` |
| `graph_layout_service/app.py:127-155` | BFS rank grouping |

## Verification

```bash
# 1. Delete cached flowchart + layout for pn277
rm -f flowcharts/faith_combined_pn277/config-faith_combined_pn277-combined_qwen3_8b_flowchart.json
rm -rf graph_layout_service/cache/

# 2. Regenerate flowchart
uv run python -m src.main --config-name=faith_combined_pn277 command=flowcharts

# 3. Start layout service + generate layout
uv run python -m uvicorn graph_layout_service.app:app --host 127.0.0.1 --port 8010 &
sleep 3
uv run python -m src.main --config-name=faith_combined_pn277 command=graphviz

# 4. Start frontend
cd deployment && npx next dev --port 3000
# Open localhost:3000 → select faith_combined_pn277 → color by "condition"
```
