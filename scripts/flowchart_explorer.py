# %% [markdown]
# # Flowchart Distribution Charts
#
# Visualize cued vs uncued distribution across thought anchor categories,
# with faithful/unfaithful breakdown within cued rollouts.
#
# Run cells individually in VS Code (Ctrl+Shift+Enter).

# %% Cell 1: Load flowchart data
import json
from pathlib import Path
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np

# --- Configure ---
REPO_ROOT = Path(__file__).resolve().parents[1]
FLOWCHART_PATH = REPO_ROOT / "flowcharts/faith_combined_pn37/config-faith_ta_pn37-thought_anchor_qwen3_8b_flowchart.json"

with open(FLOWCHART_PATH) as f:
    data = json.load(f)

# Parse nodes
nodes = {}
for node_obj in data["nodes"]:
    for cid, ndata in node_obj.items():
        nodes[cid] = ndata

responses = data["responses"]

CATEGORIES = ["PS", "FR", "AC", "UM", "RC", "SC", "FA", "UH"]
CAT_NAMES = {
    "PS": "Problem\nSetup",
    "FR": "Fact\nRetrieval",
    "AC": "Active\nReasoning",
    "UM": "Uncertainty",
    "RC": "Consolid-\nation",
    "SC": "Self\nChecking",
    "FA": "Final\nAnswer",
    "UH": "Uses\nHint",
}

print(f"Loaded: {data['prompt_index']} | {len(nodes)} nodes | {len(responses)} responses")

# %% Cell 2: Build per-node visit counts by group (uncued / cued_faithful / cued_unfaithful)


def resp_group(resp):
    if resp.get("condition") == "uncued":
        return "uncued"
    return "cued_unfaithful" if resp.get("unfaithful", False) else "cued_faithful"


# Count unique rollout visits per node per group
node_group_visits = defaultdict(Counter)

for ridx, resp in responses.items():
    group = resp_group(resp)
    visited = set()
    for edge in resp["edges"]:
        for key in ("node_a", "node_b"):
            nid = edge[key]
            if nid.startswith("cluster-") and nid not in visited:
                visited.add(nid)
                node_group_visits[nid][group] += 1

# Aggregate by anchor category
cat_groups = {cat: Counter() for cat in CATEGORIES}
for nid, groups in node_group_visits.items():
    cat = nodes[nid].get("anchor_category", "??")
    if cat in cat_groups:
        for g, count in groups.items():
            cat_groups[cat][g] += count

# Count total rollouts per group
group_totals = Counter()
for resp in responses.values():
    group_totals[resp_group(resp)] += 1

n_uncued = group_totals["uncued"]
n_cued_f = group_totals["cued_faithful"]
n_cued_uf = group_totals["cued_unfaithful"]
n_cued = n_cued_f + n_cued_uf

print(f"Uncued: {n_uncued} | Cued faithful: {n_cued_f} | Cued unfaithful: {n_cued_uf}")

# %% Cell 3: Chart — Cued vs Uncued by Category (normalized per rollout)

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(CATEGORIES))
w = 0.35

uncued_vals = [cat_groups[c]["uncued"] / n_uncued for c in CATEGORIES]
cued_vals = [(cat_groups[c]["cued_faithful"] + cat_groups[c]["cued_unfaithful"]) / n_cued for c in CATEGORIES]

ax.bar(x - w / 2, uncued_vals, w, label=f"Uncued (n={n_uncued})", color="#4C72B0")
ax.bar(x + w / 2, cued_vals, w, label=f"Cued (n={n_cued})", color="#DD8452")

ax.set_ylabel("Mean node visits per rollout")
ax.set_title(f"Cued vs Uncued Category Distribution — {data['prompt_index']}")
ax.set_xticks(x)
ax.set_xticklabels([CAT_NAMES[c] for c in CATEGORIES], fontsize=9)
ax.legend()
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
plt.show()

# %% Cell 4: Chart — Faithful vs Unfaithful within Cued (normalized per rollout)

if n_cued_uf == 0:
    print("No unfaithful rollouts — skipping chart.")
else:
    fig, ax = plt.subplots(figsize=(10, 5))

    faith_vals = [cat_groups[c]["cued_faithful"] / n_cued_f for c in CATEGORIES]
    unfaith_vals = [cat_groups[c]["cued_unfaithful"] / n_cued_uf for c in CATEGORIES]

    ax.bar(x - w / 2, faith_vals, w, label=f"Cued Faithful (n={n_cued_f})", color="#55A868")
    ax.bar(x + w / 2, unfaith_vals, w, label=f"Cued Unfaithful (n={n_cued_uf})", color="#C44E52")

    ax.set_ylabel("Mean node visits per rollout")
    ax.set_title(f"Cued Rollouts: Faithful vs Unfaithful — {data['prompt_index']}")
    ax.set_xticks(x)
    ax.set_xticklabels([CAT_NAMES[c] for c in CATEGORIES], fontsize=9)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    plt.show()

# %% Cell 5: Chart — Three-way comparison (uncued, cued faithful, cued unfaithful)

if n_cued_uf == 0:
    print("No unfaithful rollouts — skipping three-way chart.")
else:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    w3 = 0.25

    uncued_norm = [cat_groups[c]["uncued"] / n_uncued for c in CATEGORIES]
    faith_norm = [cat_groups[c]["cued_faithful"] / n_cued_f for c in CATEGORIES]
    unfaith_norm = [cat_groups[c]["cued_unfaithful"] / n_cued_uf for c in CATEGORIES]

    ax.bar(x - w3, uncued_norm, w3, label=f"Uncued (n={n_uncued})", color="#4C72B0")
    ax.bar(x, faith_norm, w3, label=f"Cued Faithful (n={n_cued_f})", color="#55A868")
    ax.bar(x + w3, unfaith_norm, w3, label=f"Cued Unfaithful (n={n_cued_uf})", color="#C44E52")

    ax.set_ylabel("Mean node visits per rollout")
    ax.set_title(f"Three-Way Category Distribution — {data['prompt_index']}")
    ax.set_xticks(x)
    ax.set_xticklabels([CAT_NAMES[c] for c in CATEGORIES], fontsize=9)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    plt.show()

# %% Cell 6: Table — Normalized visit rates and deltas

print(f"\n{'Cat':<5} {'Name':<18} {'Uncued':>8} {'Cued F':>8} {'Cued UF':>8} │ {'Δ(C-U)':>8} {'Δ(UF-F)':>8}")
print("─" * 70)
for c in CATEGORIES:
    name = CAT_NAMES[c].replace("\n", " ")
    u = cat_groups[c]["uncued"] / n_uncued
    f = cat_groups[c]["cued_faithful"] / n_cued_f if n_cued_f else 0
    uf = cat_groups[c]["cued_unfaithful"] / n_cued_uf if n_cued_uf else 0
    cued_avg = (cat_groups[c]["cued_faithful"] + cat_groups[c]["cued_unfaithful"]) / n_cued
    delta_cu = cued_avg - u
    delta_uf_f = uf - f if n_cued_uf else 0
    print(f"{c:<5} {name:<18} {u:>8.1f} {f:>8.1f} {uf:>8.1f} │ {delta_cu:>+8.1f} {delta_uf_f:>+8.1f}")
