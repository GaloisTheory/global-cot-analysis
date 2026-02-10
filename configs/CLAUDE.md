# configs/ — Hydra Configuration System

Four composable config types combined into per-prompt experiment configs.

## Config Types

| Config | Location | Key Parameters |
|--------|----------|----------------|
| Response (`r/`) | `configs/r/default.yaml` | `num_seeds_rollouts: 50`, `num_seeds_prefixes: 10`, `max_workers: 250` |
| Flowchart (`f/`) | `configs/f/default.yaml` | `sentence_similarity_threshold: 0.75`, `llm_model: openai/gpt-4o-mini`, `gamma: 0.5`, `sentence_embedding_model: paraphrase-mpnet-base-v2` |
| Predictions (`p/`) | `configs/p/default.yaml` | `top_rollouts: 20`, `beta: 0`, `weigh: true`, `strict: false`, `sliding: true` |
| Algorithms (`a/`) | `configs/a/default.yaml` | `num_rollouts_to_study: 50` |

Special flowchart configs: `f/combined.yaml` (100 seeds), `f/thought_anchor.yaml` (Gemini 3 Flash, threshold 0.55).

## Creating a New Experiment Config

```yaml
# configs/my_experiment.yaml
defaults:
  - _self_
  - r: default
  - f: default      # or f: combined, f: thought_anchor

_name_: "my_experiment"
prompt: "my_prompt"
models: ["qwen3-8b"]
property_checkers: ["correctness"]
command: "flowcharts"
```

## Faithfulness Config Naming Convention

- `faith_uncued_pn{N}` — Uncued condition for problem number N
- `faith_cued_pn{N}` — Cued condition for problem number N
- `faith_combined_pn{N}` — Combined 100-rollout superset (uses `f: combined`)
- `faith_ta_pn{N}` — Thought anchor clustering (uses `f: thought_anchor`)

## Config Composition

The `defaults:` block composes sub-configs. Faithfulness experiments typically omit `p:` and `a:` since no prefix/resample/algorithm analysis is needed. The `_self_` entry means top-level keys override sub-config defaults.
