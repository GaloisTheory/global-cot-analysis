"""Run full Graph NB + Gemini evaluation for pn37, pn277, pn408 and generate plots."""

import json
import sys
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

# --- Config ---
HINT_KEYWORDS = ["professor", "stanford", "iq", "hint", "cue", "authority"]
TOP_K = 3
N_GEMINI_OOS = 100  # all 100 OOS samples
N_FEW_SHOT = 3
MAX_COT_CHARS = 3000

# Each entry: (display_label, pn_for_oos_dir, cache_key, flowchart_configs)
# pn408 appears twice: once with the 874-rollout combined, once with the 100-rollout version
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

RESULTS_PATH = REPO_ROOT / "scripts" / "playground" / "full_eval_results.json"
PLOT_DIR = REPO_ROOT / "scripts" / "playground" / "plots"

cache_dir = REPO_ROOT / "scripts" / "playground" / ".embedding_cache_prediction"
emb_cache = EmbeddingCache(cache_dir=cache_dir)
gemini_cache_dir = REPO_ROOT / "scripts" / "playground" / ".gemini_cache"

# --- Run evaluations ---
all_results = {}  # {label: {method: metrics}}
oos_data_cache = {}  # {pn: (sentences, conditions, seeds, cutoffs)} — avoid reloading

for q in QUESTIONS:
    label = q["label"]
    pn = q["pn"]
    print(f"\n{'='*60}")
    print(f"{label.replace(chr(10), ' ')}")
    print(f"{'='*60}")

    # Load OOS data (cached per pn)
    if pn not in oos_data_cache:
        oos_dir = REPO_ROOT / "prompts" / f"faith_combined_pn{pn}_oos" / "qwen3-8b" / "rollouts"
        oos_sentences, oos_conditions, oos_seeds, _ = load_oos_rollouts(oos_dir, "sentences")
        oos_uh_cutoffs = [detect_uh_cutoff(s, HINT_KEYWORDS) for s in oos_sentences]
        oos_data_cache[pn] = (oos_sentences, oos_conditions, oos_seeds, oos_uh_cutoffs)
    oos_sentences, oos_conditions, oos_seeds, oos_uh_cutoffs = oos_data_cache[pn]
    print(f"OOS: {len(oos_sentences)} rollouts")

    q_results = {}

    # Graph NB methods
    for cfg in q["configs"]:
        method = cfg["method"]
        print(f"\n  Graph NB: {method}")
        pred = FlowchartPredictor(cfg["path"], emb_cache, HINT_KEYWORDS, use_chunked=cfg["use_chunked"])
        results = pred.predict_batch(oos_sentences, oos_conditions, oos_uh_cutoffs, top_k=TOP_K)
        m = compute_metrics(results)
        m["n_train"] = pred.n_train
        q_results[method] = m
        print(f"    n_train={pred.n_train}, accuracy={m['accuracy']:.1%}, rmse={m['rmse']:.3f}")

    # Gemini baseline (use combined predictor for few-shot, shared cache per pn)
    print(f"\n  Gemini baseline (n={N_GEMINI_OOS})...")
    combined_cfg = q["configs"][0]
    pred_for_fs = FlowchartPredictor(combined_cfg["path"], emb_cache, HINT_KEYWORDS, use_chunked=False)
    gemini_results = evaluate_gemini_oos(
        predictor=pred_for_fs,
        oos_sentences=oos_sentences,
        oos_conditions=oos_conditions,
        oos_uh_cutoffs=oos_uh_cutoffs,
        n_gemini_oos=N_GEMINI_OOS,
        n_few_shot=N_FEW_SHOT,
        max_cot_chars=MAX_COT_CHARS,
        top_k=TOP_K,
        max_workers=10,
        cache_key=q["cache_key"],
        cache_dir=gemini_cache_dir,
        verbose=True,
    )
    gm = compute_metrics(gemini_results)
    gm["n_train"] = None
    q_results["gemini_baseline"] = gm
    print(f"    accuracy={gm['accuracy']:.1%}, rmse={gm['rmse']:.3f}, n={gm['n']}")

    all_results[label] = q_results

# --- Save results ---
serializable = {}
for label, methods in all_results.items():
    serializable[label] = {}
    for method, m in methods.items():
        serializable[label][method] = {
            k: (v if not isinstance(v, (np.floating, np.integer)) else float(v))
            for k, v in m.items()
        }

RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(RESULTS_PATH, "w") as f:
    json.dump(serializable, f, indent=2)
print(f"\nResults saved to {RESULTS_PATH}")

# --- Print summary table ---
q_labels = [q["label"].replace("\n", " ") for q in QUESTIONS]
methods_order = ["gemini_baseline", "thought_anchor", "combined_chunked", "combined"]
method_display = {
    "gemini_baseline": "Gemini 3 Pro",
    "thought_anchor": "Thought Anchor",
    "combined_chunked": "Combined Chunked",
    "combined": "Combined",
}

print(f"\n{'='*100}")
print("SUMMARY")
print(f"{'='*100}")

print(f"\n{'Method':<22s}", end="")
for ql in q_labels:
    print(f"  {ql:>12s} acc  rmse", end="")
print()
print("-" * (22 + 23 * len(q_labels)))
for method in methods_order:
    print(f"{method_display[method]:<22s}", end="")
    for q in QUESTIONS:
        m = all_results[q["label"]].get(method)
        if m and m["accuracy"] is not None:
            print(f"  {m['accuracy']:>12.1%} {m['rmse']:>5.3f}", end="")
        else:
            print(f"  {'N/A':>12s} {'N/A':>5s}", end="")
    print()

# --- Generate plots ---
PLOT_DIR.mkdir(parents=True, exist_ok=True)

q_markers = ["o", "s", "D", "^"]
q_colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]

x_labels = [method_display[m] for m in methods_order]
x_pos = np.arange(len(methods_order))

fig, ax = plt.subplots(figsize=(9, 5.5))
for qi, q in enumerate(QUESTIONS):
    ql = q["label"].replace("\n", " ")
    accs = []
    valid_x = []
    for i, method in enumerate(methods_order):
        m = all_results[q["label"]].get(method)
        if m and m["accuracy"] is not None:
            accs.append(m["accuracy"] * 100)
            valid_x.append(x_pos[i])
    ax.plot(valid_x, accs, marker=q_markers[qi], color=q_colors[qi],
            label=ql, markersize=10, linewidth=1.5, linestyle="--", alpha=0.8)

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

fig2, ax2 = plt.subplots(figsize=(9, 5.5))
for qi, q in enumerate(QUESTIONS):
    ql = q["label"].replace("\n", " ")
    rmses = []
    valid_x = []
    for i, method in enumerate(methods_order):
        m = all_results[q["label"]].get(method)
        if m and m["rmse"] is not None:
            rmses.append(m["rmse"])
            valid_x.append(x_pos[i])
    ax2.plot(valid_x, rmses, marker=q_markers[qi], color=q_colors[qi],
             label=ql, markersize=10, linewidth=1.5, linestyle="--", alpha=0.8)

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
