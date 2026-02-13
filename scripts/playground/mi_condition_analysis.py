"""MI(Cluster, Condition) Analysis — pn408 thought anchor flowchart.

Run as a VS-Code / PyCharm "percent" notebook or from the terminal:
    cd projects/global-cot-analysis
    uv run python scripts/playground/mi_condition_analysis.py
"""

# %% [markdown]
# # MI(Cluster, Condition) at Each Reasoning Position
# Quantify where cued vs uncued conditions diverge in the thought anchor graph for pn408.

# %% Cell 1 — Imports & config
import json
import sys
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.clustering.thought_anchor_clusterer import ANCHOR_CATEGORIES

# --- Pick a flowchart (uncomment ONE) ---
# pn408 full (thought anchor)
#FLOWCHART_PATH = REPO_ROOT / "flowcharts" / "faith_combined_pn408" / "config-faith_ta_pn408-thought_anchor_qwen3_8b_flowchart.json"
# pn408 s100 (thought anchor)
# FLOWCHART_PATH = REPO_ROOT / "flowcharts" / "faith_combined_pn408_s100" / "config-faith_ta_pn408_s100-thought_anchor_qwen3_8b_flowchart.json"
# pn408 full 1000 rollouts (combined)
# FLOWCHART_PATH = REPO_ROOT / "flowcharts" / "faith_combined_pn408" / "config-faith_combined_pn408-combined_qwen3_8b_flowchart.json"
# pn277 (thought anchor)
# FLOWCHART_PATH = REPO_ROOT / "flowcharts" / "faith_combined_pn277" / "config-faith_ta_pn277-thought_anchor_qwen3_8b_flowchart.json"
# pn37 (thought anchor)
# FLOWCHART_PATH = REPO_ROOT / "flowcharts" / "faith_combined_pn37" / "config-faith_ta_pn37-thought_anchor_qwen3_8b_flowchart.json"
# pn408 (combined chunked)
# FLOWCHART_PATH = REPO_ROOT / "flowcharts" / "faith_combined_pn408" / "config-faith_combined_chunked_pn408-combined_chunked_qwen3_8b_flowchart.json"
# pn339 (combined chunked)
# FLOWCHART_PATH = REPO_ROOT / "flowcharts" / "faith_combined_pn339" / "config-faith_combined_chunked_pn339-combined_chunked_qwen3_8b_flowchart.json"
# pn277 (combined)
# FLOWCHART_PATH = REPO_ROOT / "flowcharts" / "faith_combined_pn277" / "config-faith_combined_pn277-combined_qwen3_8b_flowchart.json"
MAX_POSITION = 30
MIN_ROLLOUTS = 10

print(f"REPO_ROOT: {REPO_ROOT}")
print(f"Flowchart: {FLOWCHART_PATH}")
print(f"MAX_POSITION={MAX_POSITION}, MIN_ROLLOUTS={MIN_ROLLOUTS}")
print(f"Anchor categories: {list(ANCHOR_CATEGORIES.keys())}")

# %% Cell 2 — Load flowchart + build lookups
with open(FLOWCHART_PATH, "r", encoding="utf-8") as f:
    flowchart = json.load(f)

# Flatten nodes: list-of-single-key-dicts → dict
nodes = {}
for node_dict in flowchart["nodes"]:
    for k, v in node_dict.items():
        nodes[k] = v

# cluster ID → anchor category code
cluster_to_anchor = {}
for cid, info in nodes.items():
    cluster_to_anchor[cid] = info.get("anchor_category", "??")

responses = flowchart["responses"]

# Condition breakdown
cond_counts = Counter()
for resp in responses.values():
    cond = resp.get("condition", "?")
    uf = bool(resp.get("unfaithful", False))
    if cond == "uncued":
        cond_counts["uncued"] += 1
    elif cond == "cued" and not uf:
        cond_counts["cued-faithful"] += 1
    elif cond == "cued" and uf:
        cond_counts["cued-unfaithful"] += 1
    else:
        cond_counts[f"other:{cond}"] += 1

print(f"\nLoaded {len(nodes)} cluster nodes, {len(responses)} responses")
print(f"Condition breakdown: {dict(cond_counts)}")

# %% Cell 3 — Extract cluster sequences + condition labels


def rollout_bucket(resp: dict) -> str:
    condition = resp.get("condition", "?")
    if condition in ("uncued", "cued"):
        return condition
    return f"other:{condition}"


def extract_cluster_sequence(edges: list) -> list:
    """Extract ordered cluster IDs from edge list, excluding START and response-* nodes."""
    seq = []
    for edge in edges:
        node_b = edge["node_b"]
        if node_b == "START" or node_b.startswith("response-"):
            continue
        seq.append(node_b)
    return seq


sequences = []
buckets = []
resp_keys = []

for rkey, resp in responses.items():
    seq = extract_cluster_sequence(resp["edges"])
    sequences.append(seq)
    buckets.append(rollout_bucket(resp))
    resp_keys.append(rkey)

seq_lens = [len(s) for s in sequences]
print(f"\nSequence lengths: min={min(seq_lens)}, median={sorted(seq_lens)[len(seq_lens)//2]}, max={max(seq_lens)}")
print(f"Bucket distribution: {Counter(buckets)}")

# %% Cell 4 — Build position-wise contingency tables
position_data = {}  # p -> {"ct": np.array, "clusters": list, "conditions": list, "n_rollouts": int}

BUCKET_ORDER = ["uncued", "cued"]

for p in range(MAX_POSITION + 1):
    pairs = []  # (cluster_id, bucket)
    for i, seq in enumerate(sequences):
        if len(seq) > p:
            pairs.append((seq[p], buckets[i]))

    if len(pairs) < MIN_ROLLOUTS:
        continue

    cluster_ids = sorted(set(c for c, _ in pairs))
    cid_to_idx = {c: i for i, c in enumerate(cluster_ids)}

    ct = np.zeros((len(cluster_ids), len(BUCKET_ORDER)), dtype=int)
    for cid, bucket in pairs:
        if bucket in BUCKET_ORDER:
            ct[cid_to_idx[cid], BUCKET_ORDER.index(bucket)] += 1

    position_data[p] = {
        "ct": ct,
        "clusters": cluster_ids,
        "conditions": BUCKET_ORDER,
        "n_rollouts": len(pairs),
    }

print(f"\nPositions with data: {len(position_data)} (of {MAX_POSITION + 1})")
for p in sorted(position_data):
    pd = position_data[p]
    print(f"  pos {p:2d}: {pd['n_rollouts']:3d} rollouts, {len(pd['clusters']):3d} clusters")

# %% Cell 5 — Compute MI, NMI, Cramér's V, permutation p-value

N_PERMUTATIONS = 5000  # permutation test iterations


def compute_mi(ct: np.ndarray) -> tuple:
    """Compute MI (bits), H_cluster, H_condition from a contingency table."""
    ct = ct.astype(float)
    N = ct.sum()
    if N == 0:
        return 0.0, 0.0, 0.0

    # Joint probabilities
    p_joint = ct / N
    # Marginals
    p_row = ct.sum(axis=1) / N  # cluster marginal
    p_col = ct.sum(axis=0) / N  # condition marginal

    # Entropies
    H_cluster = -np.sum(p_row[p_row > 0] * np.log2(p_row[p_row > 0]))
    H_condition = -np.sum(p_col[p_col > 0] * np.log2(p_col[p_col > 0]))
    H_joint = -np.sum(p_joint[p_joint > 0] * np.log2(p_joint[p_joint > 0]))

    MI = H_cluster + H_condition - H_joint
    return MI, H_cluster, H_condition


def compute_mi_scalar(ct: np.ndarray) -> float:
    """MI only (for permutation inner loop)."""
    ct = ct.astype(float)
    N = ct.sum()
    if N == 0:
        return 0.0
    p_joint = ct / N
    p_row = ct.sum(axis=1) / N
    p_col = ct.sum(axis=0) / N
    H_cluster = -np.sum(p_row[p_row > 0] * np.log2(p_row[p_row > 0]))
    H_condition = -np.sum(p_col[p_col > 0] * np.log2(p_col[p_col > 0]))
    H_joint = -np.sum(p_joint[p_joint > 0] * np.log2(p_joint[p_joint > 0]))
    return H_cluster + H_condition - H_joint


def compute_nmi(MI: float, H_X: float, H_Y: float) -> float:
    """NMI = MI / min(H_X, H_Y)."""
    denom = min(H_X, H_Y)
    if denom == 0:
        return 0.0
    return MI / denom


def compute_cramers_v(ct: np.ndarray) -> tuple:
    """Cramér's V and chi2 p-value."""
    # Remove all-zero rows/cols for chi2
    row_sums = ct.sum(axis=1)
    col_sums = ct.sum(axis=0)
    ct_clean = ct[row_sums > 0][:, col_sums > 0]

    if ct_clean.shape[0] < 2 or ct_clean.shape[1] < 2:
        return 0.0, 1.0, 0.0

    chi2, p, dof, _ = chi2_contingency(ct_clean, correction=False)
    N = ct_clean.sum()
    k = min(ct_clean.shape) - 1
    if k == 0 or N == 0:
        return 0.0, p, chi2
    V = np.sqrt(chi2 / (N * k))
    return V, p, chi2


def permutation_p_value(
    cluster_labels: np.ndarray,
    condition_labels: np.ndarray,
    n_clusters: int,
    n_conditions: int,
    observed_mi: float,
    n_perm: int = N_PERMUTATIONS,
    rng_seed: int = 42,
) -> float:
    """Permutation test: shuffle condition labels, recompute MI, get empirical p-value."""
    rng = np.random.default_rng(rng_seed)
    count_ge = 0
    for _ in range(n_perm):
        perm_cond = rng.permutation(condition_labels)
        ct_perm = np.zeros((n_clusters, n_conditions), dtype=int)
        for cl, co in zip(cluster_labels, perm_cond):
            ct_perm[cl, co] += 1
        mi_perm = compute_mi_scalar(ct_perm)
        if mi_perm >= observed_mi:
            count_ge += 1
    return (count_ge + 1) / (n_perm + 1)  # +1 for continuity correction


print(f"Computing MI + permutation p-values ({N_PERMUTATIONS} permutations per position)...")

results = []
for p in sorted(position_data):
    pd = position_data[p]
    ct = pd["ct"]

    MI, H_cluster, H_condition = compute_mi(ct)
    NMI = compute_nmi(MI, H_cluster, H_condition)
    V, p_val_chi2, chi2 = compute_cramers_v(ct)

    # Build flat label arrays for permutation test
    cluster_labels = []
    condition_labels = []
    n_clusters = ct.shape[0]
    n_conditions = ct.shape[1]
    for i in range(n_clusters):
        for j in range(n_conditions):
            cluster_labels.extend([i] * ct[i, j])
            condition_labels.extend([j] * ct[i, j])
    cluster_labels = np.array(cluster_labels)
    condition_labels = np.array(condition_labels)

    p_val_perm = permutation_p_value(
        cluster_labels, condition_labels, n_clusters, n_conditions, MI
    )

    results.append({
        "position": p,
        "n_rollouts": pd["n_rollouts"],
        "n_clusters": len(pd["clusters"]),
        "MI": MI,
        "H_cluster": H_cluster,
        "H_condition": H_condition,
        "NMI": NMI,
        "cramers_v": V,
        "p_value": p_val_perm,       # permutation p-value (primary)
        "p_value_chi2": p_val_chi2,   # chi2 p-value (for comparison)
        "chi2": chi2,
    })
    print(f"  pos {p:2d}: MI={MI:.4f}  p_perm={p_val_perm:.4f}  p_chi2={p_val_chi2:.4g}")

# Print summary table
print(f"\n{'pos':>3s} {'n_roll':>6s} {'n_clus':>6s} {'MI':>7s} {'H_clus':>7s} {'NMI':>6s} {'V':>6s} {'p_perm':>8s} {'p_chi2':>10s} {'sig':>4s}")
print("-" * 75)
for r in results:
    sig = "*" if r["p_value"] < 0.05 else ""
    print(
        f"{r['position']:3d} {r['n_rollouts']:6d} {r['n_clusters']:6d} "
        f"{r['MI']:7.4f} {r['H_cluster']:7.4f} {r['NMI']:6.4f} "
        f"{r['cramers_v']:6.4f} {r['p_value']:8.4f} {r['p_value_chi2']:10.4g} {sig:>4s}"
    )

# %% Cell 6 — MI vs Position line chart (primary visualization)
positions = [r["position"] for r in results]
mi_vals = [r["MI"] for r in results]
h_cluster_vals = [r["H_cluster"] for r in results]
nmi_vals = [r["NMI"] for r in results]
cramers_vals = [r["cramers_v"] for r in results]
p_vals = [r["p_value"] for r in results]
n_rollouts_vals = [r["n_rollouts"] for r in results]

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# Top: MI + H(cluster) for context
ax = axes[0]
ax.plot(positions, mi_vals, "o-", color="tab:blue", label="MI (bits)", markersize=4)
ax.plot(positions, h_cluster_vals, "--", color="tab:gray", alpha=0.6, label="H(cluster)")
ax.set_ylabel("Bits")
ax.set_title("MI(Cluster, Condition) vs Reasoning Position — pn408 [uncued vs cued]")
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)

# Annotate top-3 MI positions
top3_idx = np.argsort(mi_vals)[-3:][::-1]
for idx in top3_idx:
    ax.annotate(
        f"pos {positions[idx]}",
        xy=(positions[idx], mi_vals[idx]),
        xytext=(5, 10),
        textcoords="offset points",
        fontsize=8,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="tab:red", lw=0.8),
        color="tab:red",
    )

# Middle: NMI + Cramér's V
ax = axes[1]
ax.plot(positions, nmi_vals, "s-", color="tab:orange", label="NMI", markersize=4)
ax.plot(positions, cramers_vals, "^-", color="tab:green", label="Cramér's V", markersize=4)
ax.set_ylabel("Score [0,1]")
ax.set_ylim(-0.05, 1.05)
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)

# Bottom: -log10(p_value) — permutation (bars) vs chi2 (dots)
ax = axes[2]
p_vals_chi2 = [r["p_value_chi2"] for r in results]
neg_log_p = [-np.log10(max(p, 1e-300)) for p in p_vals]
neg_log_p_chi2 = [-np.log10(max(p, 1e-300)) for p in p_vals_chi2]
ax.bar(positions, neg_log_p, color="tab:purple", alpha=0.7, width=0.8, label="permutation")
ax.scatter(positions, neg_log_p_chi2, color="tab:gray", marker="x", s=30, zorder=5, label="chi² (unreliable)")
ax.axhline(-np.log10(0.05), color="red", linestyle="--", linewidth=1, label="p=0.05")
ax.set_ylabel("-log10(p)")
ax.set_xlabel("Reasoning position")
ax.legend(loc="upper right", fontsize=8)
ax.grid(True, alpha=0.3)

# Secondary x-axis annotation for n_rollouts
ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
ax2.set_xticks(positions[::2])
ax2.set_xticklabels([str(n_rollouts_vals[i]) for i in range(0, len(positions), 2)], fontsize=7)
ax2.set_xlabel("n rollouts", fontsize=8)
ax2.tick_params(axis="x", direction="in", pad=-14)

plt.tight_layout()
plt.show()

# %% Cell 7 — Contingency heatmaps at top-MI positions
top3_positions = [results[i]["position"] for i in np.argsort(mi_vals)[-3:][::-1]]
print(f"Top-3 MI positions: {top3_positions}")

fig, axes = plt.subplots(1, 3, figsize=(26, 12))

for ax_idx, p in enumerate(top3_positions):
    ax = axes[ax_idx]
    pd = position_data[p]
    ct = pd["ct"]
    cluster_ids = pd["clusters"]

    # Get row totals for sorting, cap at top 15
    row_totals = ct.sum(axis=1)
    top_rows = np.argsort(row_totals)[-15:][::-1]

    ct_top = ct[top_rows]
    labels = []
    for row_idx in top_rows:
        cid = cluster_ids[row_idx]
        anchor = cluster_to_anchor.get(cid, "??")
        rep = nodes.get(cid, {}).get("representative_sentence", "")[:40]
        labels.append(f"[{anchor}] {rep}")

    im = ax.imshow(ct_top, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(BUCKET_ORDER)))
    ax.set_xticklabels(BUCKET_ORDER, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)

    # Annotate counts
    for i in range(ct_top.shape[0]):
        for j in range(ct_top.shape[1]):
            if ct_top[i, j] > 0:
                ax.text(j, i, str(ct_top[i, j]), ha="center", va="center", fontsize=7)

    r = results[[r["position"] for r in results].index(p)]
    ax.set_title(
        f"Position {p}\nMI={r['MI']:.3f}  NMI={r['NMI']:.3f}  p_perm={r['p_value']:.3f}",
        fontsize=10,
    )

fig.suptitle("Contingency Heatmaps at Top-MI Positions", fontsize=13, y=1.02)
plt.tight_layout()
plt.show()

# %% Cell 8 — Anchor-category-level MI (coarser view)
anchor_results = []

for p in sorted(position_data):
    pd = position_data[p]
    cluster_ids = pd["clusters"]
    ct_cluster = pd["ct"]

    # Map cluster rows → anchor category rows
    anchor_ids = sorted(set(cluster_to_anchor.get(cid, "??") for cid in cluster_ids))
    aid_to_idx = {a: i for i, a in enumerate(anchor_ids)}

    ct_anchor = np.zeros((len(anchor_ids), len(BUCKET_ORDER)), dtype=int)
    for row_idx, cid in enumerate(cluster_ids):
        anchor = cluster_to_anchor.get(cid, "??")
        ct_anchor[aid_to_idx[anchor]] += ct_cluster[row_idx]

    MI_a, H_a, H_c = compute_mi(ct_anchor)
    NMI_a = compute_nmi(MI_a, H_a, H_c)
    V_a, p_val_a, _ = compute_cramers_v(ct_anchor)

    anchor_results.append({
        "position": p,
        "MI": MI_a,
        "NMI": NMI_a,
        "cramers_v": V_a,
        "p_value": p_val_a,
        "n_anchors": len(anchor_ids),
    })

# Plot: anchor-level MI vs cluster-level MI
fig, ax = plt.subplots(figsize=(18, 12))
a_pos = [r["position"] for r in anchor_results]
a_mi = [r["MI"] for r in anchor_results]
ax.plot(positions, mi_vals, "o-", color="tab:blue", label="Cluster-level MI", markersize=4)
ax.plot(a_pos, a_mi, "s-", color="tab:red", label="Anchor-category MI", markersize=4)
ax.set_xlabel("Reasoning position")
ax.set_ylabel("MI (bits)")
ax.set_title("Cluster-level vs Anchor-category MI — pn408")
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()

print(f"\n{'pos':>3s} {'n_anch':>6s} {'MI_anch':>8s} {'MI_clus':>8s} {'NMI_anch':>8s} {'V_anch':>7s} {'p_anch':>10s}")
print("-" * 60)
for ar, cr in zip(anchor_results, results):
    print(
        f"{ar['position']:3d} {ar['n_anchors']:6d} {ar['MI']:8.4f} {cr['MI']:8.4f} "
        f"{ar['NMI']:8.4f} {ar['cramers_v']:7.4f} {ar['p_value']:10.4g}"
    )

# %% Cell 9 — Per-anchor-category MI breakdown
# For each of the 8 categories, restrict to clusters of that type, compute MI at each position.
anchor_position_mi = {}  # anchor_code -> {position -> MI}

for anchor_code in ANCHOR_CATEGORIES:
    anchor_position_mi[anchor_code] = {}

    for p in sorted(position_data):
        pd = position_data[p]
        cluster_ids = pd["clusters"]
        ct_cluster = pd["ct"]

        # Filter to clusters of this anchor type
        matching_rows = [
            i for i, cid in enumerate(cluster_ids) if cluster_to_anchor.get(cid) == anchor_code
        ]

        if len(matching_rows) < 2:
            continue

        ct_sub = ct_cluster[matching_rows]
        # Need at least some variation
        if ct_sub.sum() < MIN_ROLLOUTS:
            continue

        MI_sub, _, _ = compute_mi(ct_sub)
        anchor_position_mi[anchor_code][p] = MI_sub

# Heatmap: x=position, y=anchor category, color=MI
anchor_codes = list(ANCHOR_CATEGORIES.keys())
all_positions = sorted(position_data.keys())

mi_matrix = np.full((len(anchor_codes), len(all_positions)), np.nan)
for i, ac in enumerate(anchor_codes):
    for j, p in enumerate(all_positions):
        if p in anchor_position_mi[ac]:
            mi_matrix[i, j] = anchor_position_mi[ac][p]

fig, ax = plt.subplots(figsize=(18, 12))
im = ax.imshow(mi_matrix, cmap="YlOrRd", aspect="auto", interpolation="nearest")
ax.set_xticks(range(len(all_positions)))
ax.set_xticklabels(all_positions, fontsize=8)
ax.set_yticks(range(len(anchor_codes)))
ax.set_yticklabels([f"{ac} ({ANCHOR_CATEGORIES[ac]})" for ac in anchor_codes], fontsize=9)
ax.set_xlabel("Reasoning position")
ax.set_title("Per-Anchor-Category MI(Cluster, Condition) — pn408")
fig.colorbar(im, ax=ax, label="MI (bits)", shrink=0.8)

# Annotate cells
for i in range(mi_matrix.shape[0]):
    for j in range(mi_matrix.shape[1]):
        val = mi_matrix[i, j]
        if not np.isnan(val) and val > 0.01:
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6)

plt.tight_layout()
plt.show()

# %% Cell 10 — Summary
print("\n" + "=" * 80)
print("SUMMARY: MI(Cluster, Condition) Analysis — pn408 [uncued vs cued, permutation test]")
print("=" * 80)

print(f"\n{'pos':>3s} {'n_roll':>6s} {'n_clus':>6s} {'MI':>7s} {'NMI':>6s} {'V':>6s} {'p_perm':>8s} {'p_chi2':>10s} {'sig':>4s}")
print("-" * 65)
for r in results:
    sig = "***" if r["p_value"] < 0.001 else "**" if r["p_value"] < 0.01 else "*" if r["p_value"] < 0.05 else ""
    print(
        f"{r['position']:3d} {r['n_rollouts']:6d} {r['n_clusters']:6d} "
        f"{r['MI']:7.4f} {r['NMI']:6.4f} {r['cramers_v']:6.4f} "
        f"{r['p_value']:8.4f} {r['p_value_chi2']:10.4g} {sig:>4s}"
    )

# Divergence onset
for r in results:
    if r["NMI"] > 0.1 or r["p_value"] < 0.05:
        print(f"\nDivergence onset: position {r['position']} (NMI={r['NMI']:.4f}, p={r['p_value']:.4g})")
        break
else:
    print("\nNo significant divergence detected at NMI>0.1 or p<0.05 threshold.")

# Peak MI
peak = max(results, key=lambda r: r["MI"])
print(f"Peak MI: position {peak['position']} (MI={peak['MI']:.4f}, NMI={peak['NMI']:.4f}, p={peak['p_value']:.4g})")

print("\nDone.")
