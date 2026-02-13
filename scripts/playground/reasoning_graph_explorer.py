"""Reasoning Graph Explorer — interactive playground.

Run as a VS-Code / PyCharm "percent" notebook or from the terminal:
    cd projects/global-cot-analysis
    uv run python scripts/playground/reasoning_graph_explorer.py
"""

# %% [markdown]
# # Reasoning Graph Explorer
# Recursive sentence-by-sentence reasoning graph construction.

# %% Cell 0 — Config
from pdb import post_mortem
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
# Add project root so imports like `src.*` resolve reliably.
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "playground"))

PN_NUMBER = 408
MODEL = "qwen3-8b"
THRESHOLD = 0.7
ENTROPY_THRESHOLD = 0.5

CACHE_DIR = REPO_ROOT / "cache" / f"reasoning_graph_pn{PN_NUMBER}"

from playground_utils import (
    load_rollouts,
    build_position_index,
    EmbeddingCache,
    cluster_sentences,
    classify_sentences_llm,
    compute_transition_matrix,
    compute_conditional_entropy,
    merge_position_sentences,
    show_clusters,
    show_anchor_distribution,
    ANCHOR_CATEGORIES,
    CLASSIFICATION_PROMPT,
)

print(f"Config: pn={PN_NUMBER}, model={MODEL}, threshold={THRESHOLD}")
print(f"Cache:  {CACHE_DIR}")

# %% Cell 0.5 — Show task prompt (and GT answer if available)
import json

prompts_path = REPO_ROOT / "prompts" / "prompts.json"
metadata_path = REPO_ROOT / "prompts" / "faith_metadata.json"

with open(prompts_path, "r", encoding="utf-8") as f:
    prompts = json.load(f)
with open(metadata_path, "r", encoding="utf-8") as f:
    metadata = json.load(f)

candidate_keys = [
    f"faith_combined_pn{PN_NUMBER}",
    f"faith_uncued_pn{PN_NUMBER}",
    f"faith_cued_pn{PN_NUMBER}",
]
prompt_key = next((k for k in candidate_keys if k in prompts), None)

if prompt_key is None:
    print(f"\nNo prompt found for pn={PN_NUMBER}")
else:
    prompt_text = prompts[prompt_key]
    gt_answer = metadata.get(prompt_key, {}).get("gt_answer")
    cue_answer = metadata.get(prompt_key, {}).get("cue_answer")
    condition = metadata.get(prompt_key, {}).get("condition")
    cued_key = f"faith_cued_pn{PN_NUMBER}"
    cued_prompt = prompts.get(cued_key)

    print(f"\n=== Task prompt ({prompt_key}) ===")
    print(prompt_text)
    if gt_answer is not None:
        print(f"\nGround-truth answer: ({gt_answer})")
    if cue_answer is not None:
        print(f"Cue answer: ({cue_answer})")
    if condition is not None:
        print(f"Condition: {condition}")
    if cued_prompt:
        cue_text = cued_prompt.splitlines()[0].strip()
        print(f"Cue text: {cue_text}")
        print(f"\n=== Cued prompt ({cued_key}) ===")
        print(cued_prompt)

# %% Cell 1 — Load data
rollouts = load_rollouts(PN_NUMBER, MODEL, REPO_ROOT)
pos_index = build_position_index(rollouts)
cache = EmbeddingCache(CACHE_DIR, model_name="all-MiniLM-L6-v2")

n_positions = max(pos_index.keys()) + 1
lens = [len(r["sentences"]) for r in rollouts]
print(f"Loaded {len(rollouts)} rollouts, max position={n_positions - 1}")
print(f"Sentence counts: min={min(lens)}, median={sorted(lens)[len(lens)//2]}, max={max(lens)}")

# %% Cell 2 — Explore a position
import os
from collections import Counter

current_pos = 1
pos_data = pos_index[current_pos]

sentences = [s for _, s in pos_data]
unique_counts = Counter(sentences).most_common()

# Sanity check: unfaithful rollouts should only appear in cued condition.
invalid_unfaithful = [
    ridx
    for ridx, r in enumerate(rollouts)
    if bool(r.get("unfaithful", False)) and r.get("condition") != "cued"
]
if invalid_unfaithful:
    print(
        f"WARNING: found {len(invalid_unfaithful)} non-cued rollouts with unfaithful=True "
        f"(examples: {invalid_unfaithful[:5]})"
    )
else:
    print("Sanity check passed: unfaithful=True appears only for cued rollouts.")


def rollout_bucket(rollout: dict) -> str:
    condition = rollout.get("condition", "?")
    if condition == "uncued":
        return "uncued"
    if condition == "cued":
        return "cued-unfaithful" if bool(rollout.get("unfaithful", False)) else "cued-faithful"
    return f"other:{condition}"


sentence_bucket_counts = {}
for ridx, sent in pos_data:
    bucket = rollout_bucket(rollouts[ridx])
    sentence_bucket_counts.setdefault(sent, Counter())[bucket] += 1

print(f"\n=== Position {current_pos}: {len(sentences)} rollouts ===")
print(f"Unique sentences: {len(unique_counts)}")
for sent, cnt in unique_counts[:8]:
    bucket_counts = sentence_bucket_counts[sent]
    uncued_n = bucket_counts.get("uncued", 0)
    cued_faithful_n = bucket_counts.get("cued-faithful", 0)
    cued_unfaithful_n = bucket_counts.get("cued-unfaithful", 0)
    print(
        f"  [{cnt:3d}x] (uncued={uncued_n:3d}, cued-faithful={cued_faithful_n:3d}, "
        f"cued-unfaithful={cued_unfaithful_n:3d}) {sent[:100]}"
    )

conditions = Counter(rollouts[ridx].get("condition", "?") for ridx, _ in pos_data)
print(f"\nConditions: {dict(conditions)}")

# %% Cell 3 — Embed & cluster current position
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

embeddings = cache.encode(sentences)
labels = cluster_sentences(embeddings, threshold=THRESHOLD)
n_clusters = len(set(labels))

print(f"\nClusters at position {current_pos}: {n_clusters} (threshold={THRESHOLD})")
show_clusters(sentences, labels)

# Similarity matrix heatmap
S = embeddings @ embeddings.T
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(S, cmap="viridis", vmin=0, vmax=1)
ax.set_title(f"Cosine similarity — position {current_pos}")
fig.colorbar(im, ax=ax)
fig.savefig(CACHE_DIR / f"sim_pos{current_pos}.png", dpi=100, bbox_inches="tight")
plt.close(fig)
print(f"Saved similarity heatmap to {CACHE_DIR / f'sim_pos{current_pos}.png'}")

# %% Cell 4 — Threshold sweep
thresholds = np.arange(0.5, 0.95, 0.05)
n_clusters_sweep = [len(set(cluster_sentences(embeddings, t))) for t in thresholds]

fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(thresholds, n_clusters_sweep, "o-")
ax.set_xlabel("Similarity threshold")
ax.set_ylabel("Number of clusters")
ax.set_title(f"Threshold sweep — position {current_pos}")
fig.savefig(CACHE_DIR / f"threshold_sweep_pos{current_pos}.png", dpi=100, bbox_inches="tight")
plt.close(fig)
print(f"Threshold sweep: {list(zip([f'{t:.2f}' for t in thresholds], n_clusters_sweep))}")

# %% Cell 5 — LLM classify current position
api_key = os.environ.get("OPENROUTER_API_KEY")
if api_key:
    classifications = classify_sentences_llm(sentences)
    dist = show_anchor_distribution(classifications)

    # Cross-tab: cluster × anchor
    from collections import defaultdict
    cross = defaultdict(Counter)
    for lab, cat in zip(labels, classifications):
        cross[lab][cat] += 1
    # Build cluster → sentences mapping for representative examples
    cluster_sents = defaultdict(list)
    for sent, lab in zip(sentences, labels):
        cluster_sents[lab].append(sent)

    print("\nCluster × Anchor cross-tab:")
    for cid in sorted(cross):
        rep = cluster_sents[cid][0][:100]
        print(f"  Cluster {cid}: {dict(cross[cid])}  e.g. \"{rep}\"")
else:
    classifications = None
    print("Skipping LLM classification (OPENROUTER_API_KEY not set)")

# %% Cell 6 — Optional: use existing ThoughtAnchorClusterer
# NOTE: ThoughtAnchorClusterer.__init__ requires a config dict and loads a full
# SentenceTransformer model (~420MB). Only enable if you want a full comparison.
USE_EXISTING_CLASSIFIER = True

if USE_EXISTING_CLASSIFIER and api_key:
    from src.clustering.thought_anchor_clusterer import ThoughtAnchorClusterer

    tac = ThoughtAnchorClusterer({
        "classification_model": "google/gemini-3-flash-preview",
        "classification_batch_size": 30,
        "intra_category_threshold": 0.5,
    })
    # Use the internal batch method for sentence-level classification
    existing_labels = tac._classify_batch(sentences, batch_idx=0)
    print("Existing classifier labels:", Counter(existing_labels))
    if classifications:
        agree = sum(a == b for a, b in zip(classifications, existing_labels))
        print(f"Agreement with standalone: {agree}/{len(classifications)} ({100*agree/len(classifications):.1f}%)")
else:
    print("Skipping existing classifier comparison")

# %% Cell 7 — Load next position & cluster
next_pos = current_pos + 1
if next_pos in pos_index:
    pos_data_next = pos_index[next_pos]
    sentences_next = [s for _, s in pos_data_next]
    embeddings_next = cache.encode(sentences_next)
    labels_next = cluster_sentences(embeddings_next, threshold=THRESHOLD)

    print(f"\n=== Position {next_pos}: {len(sentences_next)} rollouts, "
          f"{len(set(labels_next))} clusters ===")
    show_clusters(sentences_next, labels_next)

    S_next = embeddings_next @ embeddings_next.T
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(S_next, cmap="viridis", vmin=0, vmax=1)
    ax.set_title(f"Cosine similarity — position {next_pos}")
    fig.savefig(CACHE_DIR / f"sim_pos{next_pos}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
else:
    print(f"Position {next_pos} does not exist")

# %% Cell 8 — Transition matrix
if next_pos in pos_index:
    T = compute_transition_matrix(labels, labels_next, pos_data, pos_data_next)
    H, per_cluster_H = compute_conditional_entropy(T)

    print(f"\nTransition matrix (pos {current_pos} → {next_pos}):")
    print(T)
    print(f"\nConditional entropy H(C_{next_pos}|C_{current_pos}) = {H:.3f} bits")
    for cid, h in sorted(per_cluster_H.items()):
        print(f"  Cluster {cid}: H={h:.3f}")

    if H < ENTROPY_THRESHOLD:
        print(f"\n→ MERGE SUGGESTION: H={H:.3f} < {ENTROPY_THRESHOLD} — "
              f"positions {current_pos} and {next_pos} are near-deterministic")
    else:
        print(f"\n→ No merge: H={H:.3f} >= {ENTROPY_THRESHOLD}")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(T, cmap="Blues")
    ax.set_xlabel(f"Cluster at pos {next_pos}")
    ax.set_ylabel(f"Cluster at pos {current_pos}")
    ax.set_title(f"Transition counts (H={H:.2f})")
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            if T[i, j] > 0:
                ax.text(j, i, str(T[i, j]), ha="center", va="center", fontsize=8)
    fig.savefig(CACHE_DIR / f"transition_{current_pos}_{next_pos}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)

# %% Cell 9 — Merge positions
if next_pos in pos_index:
    merged_texts = merge_position_sentences(rollouts, [current_pos, next_pos])
    merged_sents = list(merged_texts.values())
    merged_embs = cache.encode(merged_sents)
    merged_labels = cluster_sentences(merged_embs, threshold=THRESHOLD)

    print(f"\n=== Merged positions [{current_pos}, {next_pos}]: "
          f"{len(merged_sents)} rollouts, {len(set(merged_labels))} clusters ===")
    show_clusters(merged_sents, merged_labels, max_examples=2)

    print(f"\nBefore merge: pos {current_pos}={len(set(labels))} clusters, "
          f"pos {next_pos}={len(set(labels_next))} clusters")
    print(f"After merge:  {len(set(merged_labels))} clusters")

# %% Cell 10 — Iterative stepper
from dataclasses import dataclass, field


@dataclass
class StepState:
    position: int = 0
    layers: list = field(default_factory=list)  # [(positions, labels, sentences)]

    def step(self, pos_index, rollouts, cache, threshold, entropy_threshold):
        """Advance one position. Returns (labels, merge_suggested)."""
        if self.position not in pos_index:
            print(f"Position {self.position} not available — done.")
            return None, False

        data = pos_index[self.position]
        sents = [s for _, s in data]
        embs = cache.encode(sents)
        labs = cluster_sentences(embs, threshold=threshold)
        n_cl = len(set(labs))

        merge = False
        if self.layers:
            prev_positions, prev_labels, prev_data = self.layers[-1]
            T = compute_transition_matrix(prev_labels, labs, prev_data, data)
            H, _ = compute_conditional_entropy(T)
            merge = H < entropy_threshold
            print(f"Pos {self.position}: {n_cl} clusters, H={H:.3f} "
                  f"{'→ MERGE' if merge else ''}")
        else:
            print(f"Pos {self.position}: {n_cl} clusters (first layer)")

        if merge:
            # Extend last layer's positions
            prev_positions.append(self.position)
        else:
            self.layers.append(([self.position], labs, data))

        self.position += 1
        return labs, merge


# Demo: step through first 5 positions
stepper = StepState()
print("\n=== Iterative stepper (first 5 positions) ===")
for _ in range(5):
    result = stepper.step(pos_index, rollouts, cache, THRESHOLD, ENTROPY_THRESHOLD)
    if result[0] is None:
        break

print(f"\nLayers after stepping: {len(stepper.layers)}")
for i, (positions, labs, _) in enumerate(stepper.layers):
    print(f"  Layer {i}: positions {positions}, {len(set(labs))} clusters")

# %% Cell 11 — Full auto run
print("\n=== Full auto run ===")
auto = StepState()
max_pos = max(pos_index.keys())

for _ in range(max_pos + 1):
    result, merge = auto.step(pos_index, rollouts, cache, THRESHOLD, ENTROPY_THRESHOLD)
    if result is None:
        break

print(f"\n--- Summary ---")
print(f"Positions scanned: {auto.position}")
print(f"Layers (reasoning steps): {len(auto.layers)}")
for i, (positions, labs, _) in enumerate(auto.layers):
    print(f"  Step {i}: positions {positions} → {len(set(labs))} clusters")
