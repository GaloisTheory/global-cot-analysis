# OOS Evaluation Guide — Multi-Flowchart Cued Prediction

Compares 3 graph clustering methods + Gemini baseline for predicting cued vs uncued CoT on held-out rollouts.

## What This Does

`cued_prediction_notebook_2.py` evaluates three graph NB classifiers:

1. **combined** — sentence_then_llm clustering on raw sentences (`f: combined`)
2. **combined_chunked** — sentence_then_llm on chunked content (`f: combined_chunked`)
3. **thought_anchor** — 3-stage functional classification (`f: thought_anchor`)

Plus a Gemini 3 Pro LLM baseline (few-shot, 20 samples).

---

## Results (pn37 & pn277)

### pn37 (gt=B, cue=D)

```
Method                              N_train N_OOS   Accuracy    RMSE P(cued|uncued) P(cued|cued)
----------------------------------------------------------------------------------------------
combined (~97 rollouts)                  97   100     77.0%   0.474          0.305        0.689
combined_chunked (~97 rollouts)          97   100     69.0%   0.482          0.300        0.636
thought_anchor (~97 rollouts)            97   100     78.0%   0.464          0.180        0.424
Gemini 3 Pro baseline                   ---    11     45.5%   0.481          0.550        0.579
```

### pn277 (gt=B, cue=D)

```
Method                              N_train N_OOS   Accuracy    RMSE P(cued|uncued) P(cued|cued)
----------------------------------------------------------------------------------------------
combined (~97 rollouts)                  97   100     73.0%   0.446          0.388        0.724
combined_chunked (~97 rollouts)          97   100     72.0%   0.450          0.283        0.699
thought_anchor (~97 rollouts)            97   100     69.0%   0.479          0.429        0.601
Gemini 3 Pro baseline                   ---    14     85.7%   0.408          0.144        0.517
```

### pn408 (reference, gt=C, cue=A)

```
Method                              N_train N_OOS   Accuracy    RMSE P(cued|uncued) P(cued|cued)
----------------------------------------------------------------------------------------------
combined (874 rollouts)                 874   100     70.0%   0.438          0.362        0.620
combined_chunked (96 rollouts)           96   100     65.0%   0.463          0.422        0.627
thought_anchor (96 rollouts)             96   100     59.0%   0.488          0.498        0.606
Gemini 3 Pro baseline                   ---    18     33.3%   0.576          0.244        0.283
```

**Notes:**
- Gemini has many FAILED responses (pn37: 9/20 failed, pn277: 6/20 failed)
- Graph NB methods evaluate all 100 OOS rollouts
- For pn37/pn277, all 3 flowcharts use ~97 training rollouts — comparison is purely clustering method

---

## Reproducing for a New Question (pn=N)

### Prerequisites
- Training rollouts exist: `prompts/faith_{uncued,cued}_pn{N}/qwen3-8b/rollouts/` (50 each)
- Combined training rollouts: `prompts/faith_combined_pn{N}/qwen3-8b/rollouts/` (100)
- `OPENROUTER_API_KEY` is set

### Step 1: Create OOS configs (2 files)

```yaml
# configs/faith_uncued_pn{N}_oos.yaml
defaults:
  - _self_
  - r: default
  - f: default
_name_: "faith_uncued_pn{N}_oos"
prompt: "faith_uncued_pn{N}_oos"
models: ["qwen3-8b"]
property_checkers: ["correctness"]
command: "rollouts"
```

Same pattern for `faith_cued_pn{N}_oos.yaml`.

### Step 2: Register OOS prompts + metadata

**`prompts/prompts.json`** — add 3 entries:
- `faith_uncued_pn{N}_oos` — copy text from `faith_uncued_pn{N}`
- `faith_cued_pn{N}_oos` — copy text from `faith_cued_pn{N}`
- `faith_combined_pn{N}_oos` — copy text from `faith_uncued_pn{N}`

**`prompts/faith_metadata.json`** — add 3 entries with correct gt/cue answers:
```json
"faith_uncued_pn{N}_oos": {"pn": N, "gt_answer": "X", "cue_answer": "Y", "condition": "uncued"},
"faith_cued_pn{N}_oos": {"pn": N, "gt_answer": "X", "cue_answer": "Y", "condition": "cued"},
"faith_combined_pn{N}_oos": {"pn": N, "gt_answer": "X", "cue_answer": "Y", "condition": "combined"}
```

### Step 3: Generate OOS rollouts

```bash
uv run python -m src.main --config-name=faith_uncued_pn{N}_oos command=rollouts
uv run python -m src.main --config-name=faith_cued_pn{N}_oos command=rollouts
```

Run sequentially to avoid rate limiting. Each produces 50 rollouts. Idempotent.

### Step 4: Build combined OOS rollouts

```bash
uv run python scripts/build_combined_rollouts.py --pn {N} --num-seeds 50 \
    --uncued-prompt faith_uncued_pn{N}_oos --cued-prompt faith_cued_pn{N}_oos \
    --output-prompt faith_combined_pn{N}_oos
```

Verify: 100 files in `prompts/faith_combined_pn{N}_oos/qwen3-8b/rollouts/`.

### Step 5: Create combined_chunked config (if missing)

```yaml
# configs/faith_combined_chunked_pn{N}.yaml
defaults:
  - _self_
  - r: default
  - f: combined_chunked
_name_: "faith_combined_chunked_pn{N}"
prompt: "faith_combined_pn{N}"
models: ["qwen3-8b"]
property_checkers: ["correctness", "condition", "unfaithful"]
command: "flowcharts"
```

### Step 6: Generate missing flowcharts

```bash
uv run python -m src.main --config-name=faith_combined_pn{N} command=flowcharts
uv run python -m src.main --config-name=faith_combined_chunked_pn{N} command=flowcharts
# TA (if missing): uv run python -m src.main --config-name=faith_ta_pn{N} command=flowcharts
```

Verify 3 JSONs in `flowcharts/faith_combined_pn{N}/`.

### Step 7: Run evaluation

Edit `cued_prediction_notebook_2.py` — update `FLOWCHART_CONFIGS` and `OOS_DIR`:

```python
FLOWCHART_CONFIGS = [
    {
        "label": "combined (~97 rollouts)",
        "path": REPO_ROOT / "flowcharts" / "faith_combined_pn{N}"
        / "config-faith_combined_pn{N}-combined_qwen3_8b_flowchart.json",
        "use_chunked": False,
    },
    {
        "label": "combined_chunked (~97 rollouts)",
        "path": REPO_ROOT / "flowcharts" / "faith_combined_pn{N}"
        / "config-faith_combined_chunked_pn{N}-combined_chunked_qwen3_8b_flowchart.json",
        "use_chunked": True,
    },
    {
        "label": "thought_anchor (~97 rollouts)",
        "path": REPO_ROOT / "flowcharts" / "faith_combined_pn{N}"
        / "config-faith_ta_pn{N}-thought_anchor_qwen3_8b_flowchart.json",
        "use_chunked": False,
    },
]

OOS_DIR = REPO_ROOT / "prompts" / "faith_combined_pn{N}_oos" / "qwen3-8b" / "rollouts"
```

Then run:
```bash
uv run python scripts/playground/cued_prediction_notebook_2.py
```

---

## *** CRITICAL: Graph NB vs Gemini Must Be Comparable ***

Both methods MUST see the same input sentences for a fair comparison.
If you change one, change the other. This has been fixed but could regress.

**What "comparable" means:**
- Graph NB uses `k = min(TOP_K, uh_cutoff, len(sents))` sentences (see `predict_nb()`)
- Gemini MUST also use exactly `k` sentences — enforced via `build_oos_cot_text(max_sents=TOP_K)`
- If you change `TOP_K`, both methods automatically adjust (they share the constant)
- The "Accuracy by k" sweep in Cell 7 only varies the graph method — Gemini is not re-run per k

**History:** Originally `build_oos_cot_text` passed the full CoT (up to UH cutoff, up to 3000 chars)
to Gemini while graph NB only used `TOP_K` positions. This made Gemini see far more text,
making the comparison unfair. Fixed by adding `max_sents=TOP_K` to `build_oos_cot_text`.

## Gemini Extraction: Prefill Approach

Gemini via OpenRouter does NOT support `response_format: {"type": "json_object"}` —
it gets ignored and the model returns free text. Structured output only works via
Gemini's native API, not OpenRouter's OpenAI-compatible layer.

**What works:** Assistant message prefill via OpenRouter multi-turn format.

```python
PREFILL = '{"probability_cued":'

messages = [
    {"role": "user", "content": prompt},
    {"role": "assistant", "content": PREFILL},   # Gemini continues from here
]
```

The model completes the JSON (e.g. ` 72}`) and `call_gemini()` concatenates
`PREFILL + completion` → `{"probability_cued": 72}` → parsed with `json.loads()`.

**Why this works:** OpenRouter translates the trailing assistant message into Gemini's
multi-turn format. Gemini Pro respects the prefill and continues from it.
(Gemini Flash is less reliable with prefills — stick with Pro.)

**Previous failures:**
1. `response_format: {"type": "json_object"}` — silently ignored by OpenRouter/Gemini
2. `max_tokens: 128` — too low for Gemini 3 Pro (a thinking model that uses ~100+ tokens
   on internal reasoning before outputting anything, causing `finish_reason: "length"`
   and empty/truncated content)
3. Regex extraction from free text — worked ~80% of the time but failed when model
   wrapped the answer in prose or used unexpected formatting

---

## Flowchart Naming Convention

```
flowcharts/faith_combined_pn{N}/
  config-faith_combined_pn{N}-combined_qwen3_8b_flowchart.json
  config-faith_combined_chunked_pn{N}-combined_chunked_qwen3_8b_flowchart.json
  config-faith_ta_pn{N}-thought_anchor_qwen3_8b_flowchart.json
```

Pattern: `config-{config_name}-{clustering_type}_{model}_flowchart.json`

---

## Cost & Time (per question)

| Step | API Calls | Cost | Time |
|------|-----------|------|------|
| OOS rollouts (2x50) | ~100 | ~$0.50 | ~5 min |
| Flowcharts (up to 3) | ~50 | ~$0.30 | ~10 min |
| Notebook eval | ~20 Gemini | ~$0.10 | ~5 min |
| **Total** | **~170** | **~$1** | **~20 min** |

---

## Checklist

- [ ] Create 2 OOS config YAMLs
- [ ] Add 3 entries to `prompts/prompts.json`
- [ ] Add 3 entries to `prompts/faith_metadata.json`
- [ ] Generate uncued + cued OOS rollouts (50 each)
- [ ] Build combined OOS rollouts (100 total)
- [ ] Create combined_chunked config (if missing)
- [ ] Generate any missing flowcharts
- [ ] Verify 3 flowchart JSONs exist
- [ ] Update notebook paths and run
