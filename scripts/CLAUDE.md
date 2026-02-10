# scripts/ — Script Reference

## Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `generate_faith_configs.py` | One-time: reads `good_problems/` JSON → generates 82 prompts, metadata entries, and Hydra configs | `uv run python scripts/generate_faith_configs.py` |
| `build_combined_rollouts.py` | Merge 50 uncued + 50 cued rollouts into combined set for superset clustering | `uv run python scripts/build_combined_rollouts.py --pn {N}` |
| `run_all_faith.sh` | Batch runner: loops all 41 pn values × 2 conditions (rollouts or flowcharts) | `bash scripts/run_all_faith.sh [flowcharts]` |
| `verify_unfaithfulness.py` | Single-question verification (pn=499 only) | `uv run python scripts/verify_unfaithfulness.py` |
| `verify_unfaithfulness_all.py` | Multi-question verification + unfaithfulness keyword parser | `uv run python scripts/verify_unfaithfulness_all.py` |
| `compare_graphs.py` | Quantitative graph comparison: clusters, edges, entropy, answers | `uv run python scripts/compare_graphs.py` |
| `analyze_combined_graph.py` | Per-condition analysis: cluster occupancy, KL divergence, accuracy | `uv run python scripts/analyze_combined_graph.py` |

## Multi-Question Batch Workflow (Steps 8-11)

```bash
# Step 8: Generate configs (one-time, already done)
uv run python scripts/generate_faith_configs.py

# Step 9: Generate rollouts for all 41 questions (~4,100 API calls, ~$5-15)
#   Idempotent: skips seeds that already have saved rollouts
#   NOTE: has set -e, so stops on first failure
bash scripts/run_all_faith.sh

# Step 10: Verify unfaithfulness across all questions
uv run python scripts/verify_unfaithfulness_all.py

# Step 11: Build flowcharts for all questions
bash scripts/run_all_faith.sh flowcharts
```
