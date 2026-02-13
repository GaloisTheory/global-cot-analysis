"""Cued vs Uncued Prediction — Multi-Question Comparison

Loops over multiple questions x flowchart types, runs Graph NB + Gemini
baseline on shared OOS data, prints comparison tables, and generates plots.

Run as a VS-Code / PyCharm "percent" notebook or from the terminal:
    cd projects/global-cot-analysis
    uv run python scripts/playground/cued_prediction_simple.py
"""

# %% Cell 1 — Imports & Config
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.playground.playground_utils import EmbeddingCache
from scripts.playground.cued_prediction_utils import (
    FlowchartPredictor,
    compute_metrics,
    detect_uh_cutoff,
    evaluate_gemini_oos,
    load_oos_rollouts,
)

# --- Parameters ---
TOP_K = 3
LAPLACE_ALPHA = 1.0
N_FEW_SHOT = 3
MAX_COT_CHARS = 3000
N_GEMINI_OOS = 100
HINT_KEYWORDS = ["professor", "stanford", "iq", "hint", "cue", "authority"]

# --- Questions & flowcharts ---
FC_BASE = REPO_ROOT / "flowcharts"

QUESTIONS = [
    {
        "label": "pn37",
        "pn": 37,
        "cache_key": "pn37",
        "configs": [
            {"method": "combined", "path": FC_BASE / "faith_combined_pn37" / "config-faith_combined_pn37-combined_qwen3_8b_flowchart.json", "use_chunked": False},
            {"method": "combined_chunked", "path": FC_BASE / "faith_combined_pn37" / "config-faith_combined_chunked_pn37-combined_chunked_qwen3_8b_flowchart.json", "use_chunked": True},
            {"method": "thought_anchor", "path": FC_BASE / "faith_combined_pn37" / "config-faith_ta_pn37-thought_anchor_qwen3_8b_flowchart.json", "use_chunked": False},
        ],
    },
    {
        "label": "pn277",
        "pn": 277,
        "cache_key": "pn277",
        "configs": [
            {"method": "combined", "path": FC_BASE / "faith_combined_pn277" / "config-faith_combined_pn277-combined_qwen3_8b_flowchart.json", "use_chunked": False},
            {"method": "combined_chunked", "path": FC_BASE / "faith_combined_pn277" / "config-faith_combined_chunked_pn277-combined_chunked_qwen3_8b_flowchart.json", "use_chunked": True},
            {"method": "thought_anchor", "path": FC_BASE / "faith_combined_pn277" / "config-faith_ta_pn277-thought_anchor_qwen3_8b_flowchart.json", "use_chunked": False},
        ],
    },
    {
        "label": "pn408\n(100 rollouts)",
        "pn": 408,
        "cache_key": "pn408",
        "configs": [
            {"method": "combined", "path": FC_BASE / "faith_combined_pn408" / "config-faith_combined_pn408-combined_qwen3_8b_flowchart.json.bak100", "use_chunked": False},
            {"method": "combined_chunked", "path": FC_BASE / "faith_combined_pn408" / "config-faith_combined_chunked_pn408-combined_chunked_qwen3_8b_flowchart.json", "use_chunked": True},
            {"method": "thought_anchor", "path": FC_BASE / "faith_combined_pn408" / "config-faith_ta_pn408-thought_anchor_qwen3_8b_flowchart.json", "use_chunked": False},
        ],
    },
    {
        "label": "pn408\n(874 rollouts)",
        "pn": 408,
        "cache_key": "pn408",
        "configs": [
            {"method": "combined", "path": FC_BASE / "faith_combined_pn408" / "config-faith_combined_pn408-combined_qwen3_8b_flowchart.json", "use_chunked": False},
            {"method": "combined_chunked", "path": FC_BASE / "faith_combined_pn408" / "config-faith_combined_chunked_pn408-combined_chunked_qwen3_8b_flowchart.json", "use_chunked": True},
            {"method": "thought_anchor", "path": FC_BASE / "faith_combined_pn408" / "config-faith_ta_pn408-thought_anchor_qwen3_8b_flowchart.json", "use_chunked": False},
        ],
    },
]

PLOT_DIR = REPO_ROOT / "scripts" / "playground" / "plots"

print(f"Questions: {len(QUESTIONS)}")
print(f"TOP_K={TOP_K}, LAPLACE_ALPHA={LAPLACE_ALPHA}, N_GEMINI_OOS={N_GEMINI_OOS}")

# %% Cell 2 — Shared resources
cache_dir = REPO_ROOT / "scripts" / "playground" / ".embedding_cache_prediction"
emb_cache = EmbeddingCache(cache_dir=cache_dir)
gemini_cache_dir = REPO_ROOT / "scripts" / "playground" / ".gemini_cache"

# %% Cell 3 — Run all evaluations (Graph NB + Gemini per question)
all_results = {}  # {label: {method: metrics}}
oos_data_cache = {}  # {pn: (sentences, conditions, seeds, cutoffs)}

for q in QUESTIONS:
    label = q["label"]
    pn = q["pn"]
    print(f"\n{'='*60}")
    print(f"{label.replace(chr(10), ' ')}")
    print(f"{'='*60}")

    # Load OOS data (cached per pn — avoids reloading for pn408 variants)
    if pn not in oos_data_cache:
        oos_dir = REPO_ROOT / "prompts" / f"faith_combined_pn{pn}_oos" / "qwen3-8b" / "rollouts"
        oos_sents, oos_conds, oos_seeds, _ = load_oos_rollouts(oos_dir, "sentences")
        oos_cuts = [detect_uh_cutoff(s, HINT_KEYWORDS) for s in oos_sents]
        oos_data_cache[pn] = (oos_sents, oos_conds, oos_seeds, oos_cuts)
        print(f"OOS: {len(oos_sents)} rollouts ({dict(Counter(oos_conds))})")
    else:
        print(f"OOS: {len(oos_data_cache[pn][0])} rollouts (cached)")
    oos_sents, oos_conds, oos_seeds, oos_cuts = oos_data_cache[pn]

    q_results = {}

    # Graph NB
    for cfg in q["configs"]:
        pred = FlowchartPredictor(cfg["path"], emb_cache, HINT_KEYWORDS,
                                  laplace_alpha=LAPLACE_ALPHA, use_chunked=cfg["use_chunked"])
        results = pred.predict_batch(oos_sents, oos_conds, oos_cuts, top_k=TOP_K)
        m = compute_metrics(results)
        m["n_train"] = pred.n_train
        q_results[cfg["method"]] = m
        print(f"  {cfg['method']:<20s} n_train={pred.n_train:4d}  acc={m['accuracy']:.1%}  rmse={m['rmse']:.3f}")

    # Gemini baseline
    fs_pred = FlowchartPredictor(q["configs"][0]["path"], emb_cache, HINT_KEYWORDS,
                                 laplace_alpha=LAPLACE_ALPHA, use_chunked=False)
    gem_results = evaluate_gemini_oos(
        predictor=fs_pred, oos_sentences=oos_sents, oos_conditions=oos_conds,
        oos_uh_cutoffs=oos_cuts, n_gemini_oos=N_GEMINI_OOS, n_few_shot=N_FEW_SHOT,
        max_cot_chars=MAX_COT_CHARS, top_k=TOP_K, max_workers=10,
        cache_key=q["cache_key"], cache_dir=gemini_cache_dir, verbose=False,
    )
    gm = compute_metrics(gem_results)
    gm["n_train"] = None
    q_results["gemini_baseline"] = gm
    print(f"  {'gemini_baseline':<20s} {'':>13s}  acc={gm['accuracy']:.1%}  rmse={gm['rmse']:.3f}  (n={gm['n']})")

    all_results[label] = q_results

# %% Cell 4 — Summary Table
METHODS_ORDER = ["gemini_baseline", "thought_anchor", "combined_chunked", "combined"]
METHOD_DISPLAY = {
    "gemini_baseline": "Gemini 3 Pro",
    "thought_anchor": "Thought Anchor",
    "combined_chunked": "Combined Chunked",
    "combined": "Combined",
}

print(f"\n{'='*100}")
print("SUMMARY")
print(f"{'='*100}")

q_labels_flat = [q["label"].replace("\n", " ") for q in QUESTIONS]
print(f"\n{'Method':<22s}", end="")
for ql in q_labels_flat:
    print(f"  {ql:>20s}", end="")
print()
print("-" * (22 + 22 * len(QUESTIONS)))

for method in METHODS_ORDER:
    print(f"{METHOD_DISPLAY[method]:<22s}", end="")
    for q in QUESTIONS:
        m = all_results[q["label"]].get(method)
        if m and m["accuracy"] is not None:
            print(f"  {m['accuracy']:>8.1%} / {m['rmse']:.3f}", end="")
        else:
            print(f"  {'N/A':>20s}", end="")
    print()

# %% Cell 5 — Confusion Matrices
print("\n--- Confusion Matrices ---")
for q in QUESTIONS:
    ql = q["label"].replace("\n", " ")
    print(f"\n  {ql}:")
    for method in METHODS_ORDER:
        m = all_results[q["label"]].get(method)
        if not m or not m["confusion"]:
            continue
        c = m["confusion"]
        print(f"    {METHOD_DISPLAY[method]:<22s}  TP={c['tp']:2d} FP={c['fp']:2d} FN={c['fn']:2d} TN={c['tn']:2d}")

# %% Cell 6 — Accuracy-by-k Sweep (first question only, to keep output short)
print("\n=== Accuracy by k (pn37) ===\n")
oos_sents_37, oos_conds_37, _, oos_cuts_37 = oos_data_cache[37]
q37_configs = QUESTIONS[0]["configs"]

header = f"{'k':<4s}" + "".join(f"{'  ' + METHOD_DISPLAY[c['method']]:>22s}" for c in q37_configs)
print(header)
print("-" * (4 + 22 * len(q37_configs)))

for k in range(1, TOP_K + 2):
    row = f"{k:<4d}"
    for cfg in q37_configs:
        pred = FlowchartPredictor(cfg["path"], emb_cache, HINT_KEYWORDS,
                                  laplace_alpha=LAPLACE_ALPHA, use_chunked=cfg["use_chunked"])
        results = pred.predict_batch(oos_sents_37, oos_conds_37, oos_cuts_37, top_k=k)
        m = compute_metrics(results)
        row += f"{m['accuracy']:>21.1%} " if m["accuracy"] is not None else f"{'N/A':>22s}"
    print(row)

# %% Cell 7 — Plots
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Plot 3 main lines (pn37, pn277, pn408 w/ 100 rollouts).
# pn408 (874 rollouts) only differs at "combined", so show it as a single extra point.
PLOT_QUESTIONS = QUESTIONS[:3]  # pn37, pn277, pn408 (100 rollouts)
q_markers = ["o", "s", "D"]
q_colors = ["#2196F3", "#4CAF50", "#FF9800"]
x_labels = [METHOD_DISPLAY[m] for m in METHODS_ORDER]
x_pos = np.arange(len(METHODS_ORDER))

# 874-rollout data for the extra point
pn408_874 = all_results.get("pn408\n(874 rollouts)", {})
combined_idx = METHODS_ORDER.index("combined")

# --- Accuracy plot ---
fig, ax = plt.subplots(figsize=(9, 5.5))
for qi, q in enumerate(PLOT_QUESTIONS):
    ql = q["label"].replace("\n", " ")
    accs, valid_x = [], []
    for i, method in enumerate(METHODS_ORDER):
        m = all_results[q["label"]].get(method)
        if m and m["accuracy"] is not None:
            accs.append(m["accuracy"] * 100)
            valid_x.append(x_pos[i])
    ax.plot(valid_x, accs, marker=q_markers[qi], color=q_colors[qi],
            label=ql, markersize=10, linewidth=1.5, linestyle="--", alpha=0.8)

# Extra point: pn408 with 874 rollouts (combined only)
m874 = pn408_874.get("combined")
if m874 and m874["accuracy"] is not None:
    ax.plot(x_pos[combined_idx], m874["accuracy"] * 100, marker="^", color="#E91E63",
            markersize=12, linewidth=0, label="pn408 (874 rollouts)", zorder=5)

ax.axhline(y=50, color="gray", linestyle=":", alpha=0.5, label="chance")
ax.set_xticks(x_pos)
ax.set_xticklabels(x_labels, rotation=15, ha="right")
ax.set_ylabel("OOS Accuracy (%)")
ax.set_title("Cued vs Uncued Prediction — OOS Accuracy by Method")
ax.legend(loc="best", fontsize=9)
ax.set_ylim(20, 100)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(PLOT_DIR / "oos_accuracy_comparison.png", dpi=150)
print(f"\nSaved: {PLOT_DIR / 'oos_accuracy_comparison.png'}")

# --- RMSE plot ---
fig2, ax2 = plt.subplots(figsize=(9, 5.5))
for qi, q in enumerate(PLOT_QUESTIONS):
    ql = q["label"].replace("\n", " ")
    rmses, valid_x = [], []
    for i, method in enumerate(METHODS_ORDER):
        m = all_results[q["label"]].get(method)
        if m and m["rmse"] is not None:
            rmses.append(m["rmse"])
            valid_x.append(x_pos[i])
    ax2.plot(valid_x, rmses, marker=q_markers[qi], color=q_colors[qi],
             label=ql, markersize=10, linewidth=1.5, linestyle="--", alpha=0.8)

# Extra point: pn408 with 874 rollouts (combined only)
if m874 and m874["rmse"] is not None:
    ax2.plot(x_pos[combined_idx], m874["rmse"], marker="^", color="#E91E63",
             markersize=12, linewidth=0, label="pn408 (874 rollouts)", zorder=5)

ax2.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5, label="chance RMSE")
ax2.set_xticks(x_pos)
ax2.set_xticklabels(x_labels, rotation=15, ha="right")
ax2.set_ylabel("RMSE")
ax2.set_title("Cued vs Uncued Prediction — OOS RMSE by Method (lower is better)")
ax2.legend(loc="best", fontsize=9)
ax2.set_ylim(0.3, 0.75)
ax2.grid(axis="y", alpha=0.3)
fig2.tight_layout()
fig2.savefig(PLOT_DIR / "oos_rmse_comparison.png", dpi=150)
print(f"Saved: {PLOT_DIR / 'oos_rmse_comparison.png'}")

print("\nDone.")
# %%
