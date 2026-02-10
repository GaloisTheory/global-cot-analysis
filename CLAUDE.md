# Context Project

Research on interpreting LLM chain-of-thought reasoning via two complementary approaches:

1. **Global CoT Analysis** ([LessWrong blog](https://www.lesswrong.com/posts/q9g9zuudd3Pvw2cbj) · [live demo](https://cot-clustering.vercel.app/) · [GitHub](https://github.com/Centrattic/global-cot-analysis)) — pattern discovery via clustering
2. **Thought Branches** (Macar, Bogdan, Rajamanoharan, Nanda; arXiv:2510.27484) — causal analysis via resampling

**Key Insight:** Studying a single CoT is insufficient. Both projects study distributions over reasoning paths but ask different questions: Global CoT asks "what strategies do models use?" while Thought Branches asks "which reasoning steps causally influence outcomes?"

---

## Active Experiment: Unfaithful CoT Graphs (faithfulness × global-cot-analysis)

### What We're Trying

Combining both projects: using Global CoT Analysis's clustering pipeline to visualize how **unfaithful reasoning** (from Thought Branches Exp 4) differs structurally from faithful reasoning.

**Hypothesis:** If a cue (e.g., "A Stanford Professor answered (A)") influences reasoning through latent channels, the **global reasoning graphs** for cued vs uncued conditions should differ structurally — different cluster distributions, different transition patterns, different entropy — even though no individual sentence mentions the cue.

### Problems

- **41 questions** from `good_problems/Professor_itc_failure_threshold0.3_correct_base_no_mention.json`
- pn values: 19, 26, 37, 59, 60, 62, 68, 81, 119, 145, 198, 212, 215, 277, 288, 295, 309, 324, 339, 340, 369, 371, 382, 408, 418, 430, 432, 440, 459, 474, 479, 499, 700, 715, 768, 804, 819, 827, 877, 960, 972
- Each has different gt_answer and cue_answer (mapped in `prompts/faith_metadata.json`)
- Original pilot question: **pn=499** — "The low intake of carbohydrate in the diet:" (gt=C, cue=A)
- Model: `qwen/qwen3-8b` via OpenRouter
- Cue type: "The following was answered as (X) by a Stanford Professor with an IQ of 130."

### Status

**Branch:** `feat/faithfulness-experiment` on `global-cot-analysis`
**PR:** https://github.com/Centrattic/global-cot-analysis/pull/2

| Step | Status | Description |
|------|--------|-------------|
| 0. Tests | Done | 33 unit tests for CoT parsing + MCQ answer extraction |
| 1. DeepSeek support | Done | Added model config + extended `parse_cot_content()` for `<think>`/`</think>` |
| 2. MCQ filter | Done | `MCQFilter` class with 4-pattern regex cascade, CoT fallback |
| 3. Prompts | Done | `prompts/prompts.json` with cued + uncued versions |
| 4. Hydra configs | Done | `configs/faith_uncued.yaml` + `configs/faith_cued.yaml` |
| 5. Generate rollouts | **Done** | 50 rollouts each for `faith_uncued` and `faith_cued` with `qwen3-8b` |
| 5.5. Verify unfaithfulness | **Done** | **PASS**: gap=64% (uncued: 98% correct, cued: 64% cue answer) |
| 6. Build flowcharts | **TODO** | `uv run python -m src.main --config-name=faith_uncued command=flowcharts` |
| 7. Compare graphs | **TODO** | `uv run python scripts/compare_graphs.py` |
| 8. Multi-question configs | **Done** | `scripts/generate_faith_configs.py` → 82 configs, 84 prompts, metadata |
| 9. Multi-question rollouts | **TODO** | `bash scripts/run_all_faith.sh` — 4,100 API calls |
| 10. Multi-question verification | **TODO** | `uv run python scripts/verify_unfaithfulness_all.py` |
| 11. Multi-question flowcharts | **TODO** | `bash scripts/run_all_faith.sh flowcharts` |

### Superset Clustering (combined graph approach)

**Problem:** Running flowcharts separately per condition produces two independent graphs with non-aligned clusters.

**Solution:** Cluster all 100 rollouts (50 cued + 50 uncued) together into one graph, with `condition` and `unfaithful` property checkers to color-code them.

**Branch:** `feat/faithfulness-graphs` on `global-cot-analysis`
**Worktree:** `projects/context/global-cot-analysis-faithfulness-graphs/`

| Step | Status | Description |
|------|--------|-------------|
| 1. Build combined rollouts | **Done** | `scripts/build_combined_rollouts.py` → 100 rollouts |
| 2. Property checkers | **Done** | `condition.py` + `unfaithful.py` |
| 3. Prompt filter + config | **Done** | `faith_combined` in `PROMPT_FILTERS`, `prompts.json`, configs |
| 4. Generate flowchart | **TODO** | `uv run python -m src.main --config-name=faith_combined command=flowcharts` |
| 5. Analyze graph | **TODO** | `uv run python scripts/analyze_combined_graph.py` |
| 6. Visualize | **TODO** | Frontend: color by "condition" to see cued vs uncued distribution |

#### Key Files (faithfulness-graphs)

| File | Purpose |
|------|---------|
| `scripts/build_combined_rollouts.py` | Merge + annotate rollouts (incl. unfaithful tags) |
| `src/property_checkers/condition.py` | `PropertyCheckerCondition` — reads `condition` field |
| `src/property_checkers/unfaithful.py` | `PropertyCheckerUnfaithful` — reads `unfaithful` field |
| `configs/f/combined.yaml` | Flowchart config with `num_seeds_rollouts: 100` |
| `configs/faith_combined.yaml` | Experiment config: correctness + condition + unfaithful checkers |
| `scripts/analyze_combined_graph.py` | Per-condition graph analysis |

### Thought Anchor Clustering (coarse graph approach)

Classifies sentences into 8 functional categories first, then clusters within each. Reduces ~400 nodes to ~217. See `src/clustering/CLAUDE.md` for full details (3-stage funnel, category table, implementation).

---

## Runbook: Combined Faithfulness Graph for a New Question

Step-by-step guide to generate and visualize a combined cued+uncued graph for any of the 41 questions. All commands run from the **global-cot-analysis** repo root.

**Prerequisites:**
- Source rollouts must already exist: `prompts/faith_uncued_pn{N}/qwen3-8b/rollouts/` and `prompts/faith_cued_pn{N}/qwen3-8b/rollouts/` (50 each)
- `OPENROUTER_API_KEY` must be set (for GPT-4o-mini LLM merge during clustering)
- `uv sync` has been run at least once

**Step 1: Register the prompt (3 file edits)**

All three edits are required or the pipeline will fail:

1. **`prompts/prompts.json`** — Add entry with the uncued prompt text:
   ```json
   "faith_combined_pn{N}": "<uncued prompt text from faith_uncued_pn{N} entry>"
   ```

2. **`prompts/faith_metadata.json`** — Add entry (auto-registers in PROMPT_FILTERS via prompt_utils.py):
   ```json
   "faith_combined_pn{N}": {"pn": N, "gt_answer": "X", "cue_answer": "Y", "condition": "combined"}
   ```
   Look up gt_answer and cue_answer from the existing `faith_cued_pn{N}` entry in the same file.

3. **`configs/faith_combined_pn{N}.yaml`** — Create Hydra config:
   ```yaml
   defaults:
     - _self_
     - r: default
     - f: combined    # <-- this sets num_seeds_rollouts: 100

   _name_: "faith_combined_pn{N}"
   prompt: "faith_combined_pn{N}"
   models: ["qwen3-8b"]
   property_checkers: ["correctness", "condition", "unfaithful"]
   command: "flowcharts"
   ```

**Step 2: Build combined rollouts**

```bash
uv run python scripts/build_combined_rollouts.py --pn {N}
```

This merges 50 uncued (seeds 0-49) + 50 cued (seeds 50-99) into `prompts/faith_combined_pn{N}/qwen3-8b/rollouts/`. Auto-detects unfaithful rollouts. Verify: output should say "Written 100 combined rollouts".

**Step 3: Generate flowchart (slow — ~2-5 min)**

```bash
uv run python -m src.main --config-name=faith_combined_pn{N} command=flowcharts
```

Requires `OPENROUTER_API_KEY` for GPT-4o-mini LLM-guided cluster merging. Creates JSON in `flowcharts/faith_combined_pn{N}/`.

**Step 4: Start layout service + generate graph layout**

```bash
# Start the layout service — MUST use uv run, system python3 doesn't have uvicorn
uv run python -m uvicorn graph_layout_service.app:app --host 127.0.0.1 --port 8010 &
sleep 3

# Verify it's running
curl -s http://127.0.0.1:8010/  # should return {"status":"ok",...}

# Generate layout positions
uv run python -m src.main --config-name=faith_combined_pn{N} command=graphviz
```

Note: pygraphviz is not installed — the service falls back to random layout (messier but functional).

**Step 5: Start frontend**

```bash
cd deployment
npm install          # if first time
npx next dev --port 3000
```

Open `http://localhost:3000`, select `faith_combined_pn{N}` from the dropdown, then use property checker controls to color by "condition" or "unfaithful".

**Common pitfalls:**
- **Rollouts are gitignored** — worktrees and fresh clones won't have them. Symlink if needed: `ln -s /path/to/main/prompts/faith_{un}cued_pn{N} prompts/`
- **Don't use `python3`** — system Python doesn't have project deps. Always use `uv run python`.
- **Don't use `npm run dev`** in deployment/ — calls `concurrently` which isn't installed. Use `npx next dev`.
- **Layout service must start BEFORE `command=graphviz`** — otherwise graphviz generation silently skips layout.
- **3 empty rollouts are normal** — some seeds produce empty CoT; pipeline skips them gracefully (97/100 typical).

---

## Common Gotchas

- **Rollouts are gitignored** — `prompts/*/` directories with rollout data are not in git
- **Always use `uv run python`** not `python3` — system Python lacks project deps
- **Port conflicts** — `lsof -ti:3000 | xargs kill -9` before starting frontend
- **`npm run dev` vs `npx next dev`** — use `npx next dev --port 3000` (no concurrently)
- **`.gitignore` uses `prompts/*`** (not `prompts/`) so negation patterns work

---

## Thought Branches (Exp 4: Unfaithful CoT)

The **sandwich experiment**: transplant a single sentence from cued reasoning into uncued context. If P(cue_answer) increases, that sentence carries latent cue influence invisible to surface-level monitoring.

Key finding: models shift answers toward the cue without mentioning it in CoT. Sentences carry hidden influence that propagates through the reasoning chain.

Source: `thought-branches/faithfulness/` (separate repo). Exp 1-3 (blackmail, whistleblower, resume bias) are in the same repo but not relevant to current work.

---

## Deeper Context (Subdirectory Files)

| File | What's There |
|------|-------------|
| `src/CLAUDE.md` | Pipeline architecture, complete file inventory, code map, design decisions, data formats, extensibility |
| `src/clustering/CLAUDE.md` | Thought anchor 3-stage funnel, 8 categories, implementation details, TODOs |
| `deployment/CLAUDE.md` | Frontend startup, component map, API routes, port conflicts |
| `configs/CLAUDE.md` | Hydra config types, naming conventions, how to create new configs |
| `scripts/CLAUDE.md` | Script reference, multi-question batch workflow |
