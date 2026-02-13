"""Helpers for the reasoning graph explorer playground."""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import AgglomerativeClustering


# ---------------------------------------------------------------------------
# Re-exports from existing src (caller must have src on sys.path)
# ---------------------------------------------------------------------------
from src.clustering.thought_anchor_clusterer import (
    ANCHOR_CATEGORIES,
    CLASSIFICATION_PROMPT,
)
from src.utils.json_utils import load_json


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_rollouts(
    pn: int,
    model: str,
    repo_root: Path,
) -> List[Dict[str, Any]]:
    """Load all rollout JSONs for a (pn, model) pair.

    Returns a list of dicts sorted by seed.
    """
    rollout_dir = repo_root / "prompts" / f"faith_combined_pn{pn}" / model / "rollouts"
    if not rollout_dir.exists():
        raise FileNotFoundError(f"No rollouts at {rollout_dir}")

    rollouts = []
    for p in sorted(rollout_dir.glob("*.json"), key=lambda x: int(x.stem)):
        with open(p, "r", encoding="utf-8") as f:
            rollouts.append(json.load(f))
    return rollouts


def build_position_index(
    rollouts: List[Dict[str, Any]],
) -> Dict[int, List[Tuple[int, str]]]:
    """Build ``pos -> [(rollout_idx, sentence)]`` mapping.

    Only includes positions that have at least one sentence across rollouts.
    """
    index: Dict[int, List[Tuple[int, str]]] = defaultdict(list)
    for ridx, r in enumerate(rollouts):
        for pos, sent in enumerate(r["sentences"]):
            index[pos].append((ridx, sent))
    return dict(index)


# ---------------------------------------------------------------------------
# Embedding cache
# ---------------------------------------------------------------------------

class EmbeddingCache:
    """Lazy-loading embedding cache backed by ``.npz`` files."""

    def __init__(
        self,
        cache_dir: Path,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.cache_dir / "embeddings.npz"
        self.model_name = model_name
        self._model = None

        # sentence -> embedding (loaded lazily)
        self._store: Dict[str, np.ndarray] = {}
        if self.cache_path.exists():
            data = np.load(self.cache_path, allow_pickle=True)
            sentences = list(data["sentences"])
            embeddings = data["embeddings"]
            for s, e in zip(sentences, embeddings):
                self._store[s] = e

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, sentences: List[str]) -> np.ndarray:
        """Return normalized embeddings for *sentences*, using cache where possible."""
        missing = [s for s in sentences if s not in self._store]
        if missing:
            embs = self.model.encode(missing, normalize_embeddings=True, show_progress_bar=False)
            for s, e in zip(missing, embs):
                self._store[s] = e
            self.save()
        return np.array([self._store[s] for s in sentences])

    def save(self) -> None:
        all_sents = list(self._store.keys())
        all_embs = np.array(list(self._store.values()))
        np.savez(self.cache_path, sentences=np.array(all_sents, dtype=object), embeddings=all_embs)


# ---------------------------------------------------------------------------
# Clustering (extracted from SentenceThenLLMClusterer._cluster_by_agglomerative)
# ---------------------------------------------------------------------------

def cluster_sentences(
    embeddings: np.ndarray,
    threshold: float = 0.7,
) -> List[int]:
    """Agglomerative clustering on normalized embeddings.

    ``threshold`` is a *similarity* threshold (0-1). Pairs with cosine
    similarity >= threshold may end up in the same cluster.
    """
    if len(embeddings) < 2:
        return list(range(len(embeddings)))
    S = embeddings @ embeddings.T
    D = np.clip(1.0 - S, 0.0, 2.0).astype(np.float16)
    np.fill_diagonal(D, np.float16(0.0))
    h = np.float32(1.0 - threshold)
    model = AgglomerativeClustering(
        linkage="complete",
        metric="precomputed",
        distance_threshold=float(h),
        n_clusters=None,
        compute_full_tree="auto",
    )
    labels = model.fit_predict(D)
    return list(map(int, labels))


# ---------------------------------------------------------------------------
# LLM classification (standalone rewrite of ThoughtAnchorClusterer._classify_batch)
# ---------------------------------------------------------------------------

def classify_sentences_llm(
    sentences: List[str],
    prompt_template: str = CLASSIFICATION_PROMPT,
    api_key: Optional[str] = None,
    model: str = "google/gemini-3-flash-preview",
    batch_size: int = 30,
    base_url: str = "https://openrouter.ai/api/v1",
) -> List[str]:
    """Classify sentences into anchor categories via LLM.

    Returns a list of category codes (one per sentence).
    Raises ``RuntimeError`` if ``api_key`` is not provided and
    ``OPENROUTER_API_KEY`` is not set.
    """
    import requests

    api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set and no api_key provided")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    all_labels: List[str] = []
    for start in range(0, len(sentences), batch_size):
        batch = sentences[start : start + batch_size]
        numbered = "\n".join(f"{i+1}. {s}" for i, s in enumerate(batch))
        prompt = prompt_template.format(sentences=numbered)

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1024,
            "temperature": 0.1,
        }
        resp = requests.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()

        # Strip markdown code fences if present
        if "```" in content:
            lines = content.split("\n")
            json_lines, in_block = [], False
            for line in lines:
                if line.strip().startswith("```"):
                    in_block = not in_block
                    continue
                if in_block:
                    json_lines.append(line)
            content = "\n".join(json_lines)

        try:
            labels = json.loads(content)
        except json.JSONDecodeError:
            labels = ["AC"] * len(batch)

        # Pad / truncate
        if len(labels) < len(batch):
            labels.extend(["AC"] * (len(batch) - len(labels)))
        elif len(labels) > len(batch):
            labels = labels[: len(batch)]

        all_labels.extend(labels)
    return all_labels


# ---------------------------------------------------------------------------
# Transition matrix & entropy
# ---------------------------------------------------------------------------

def compute_transition_matrix(
    labels_a: List[int],
    labels_b: List[int],
    rollout_map_a: List[Tuple[int, str]],
    rollout_map_b: List[Tuple[int, str]],
) -> np.ndarray:
    """Count transitions per-rollout between cluster assignments at two positions.

    ``rollout_map_x`` is ``[(rollout_idx, sentence), ...]`` from ``build_position_index``.
    Returns a ``(max_a+1, max_b+1)`` count matrix.
    """
    idx_to_label_a = {ridx: lab for (ridx, _), lab in zip(rollout_map_a, labels_a)}
    idx_to_label_b = {ridx: lab for (ridx, _), lab in zip(rollout_map_b, labels_b)}

    n_a = max(labels_a) + 1
    n_b = max(labels_b) + 1
    T = np.zeros((n_a, n_b), dtype=int)

    shared = set(idx_to_label_a) & set(idx_to_label_b)
    for ridx in shared:
        T[idx_to_label_a[ridx], idx_to_label_b[ridx]] += 1
    return T


def compute_conditional_entropy(T: np.ndarray) -> Tuple[float, Dict[int, float]]:
    """H(C_{i+1} | C_i) with per-cluster breakdown.

    Returns ``(overall_H, {cluster_i: H_i})``.
    """
    row_sums = T.sum(axis=1)
    total = T.sum()
    if total == 0:
        return 0.0, {}

    per_cluster: Dict[int, float] = {}
    overall = 0.0
    for i, row_total in enumerate(row_sums):
        if row_total == 0:
            per_cluster[i] = 0.0
            continue
        probs = T[i] / row_total
        probs = probs[probs > 0]
        h_i = -float(np.sum(probs * np.log2(probs)))
        per_cluster[i] = h_i
        overall += (row_total / total) * h_i
    return overall, per_cluster


# ---------------------------------------------------------------------------
# Merge positions
# ---------------------------------------------------------------------------

def merge_position_sentences(
    rollouts: List[Dict[str, Any]],
    positions: List[int],
) -> Dict[int, str]:
    """Concatenate sentences at multiple positions per rollout.

    Returns ``{rollout_idx: merged_text}``. Rollouts that don't reach all
    positions are skipped.
    """
    merged: Dict[int, str] = {}
    for ridx, r in enumerate(rollouts):
        sents = r["sentences"]
        if all(p < len(sents) for p in positions):
            merged[ridx] = " ".join(sents[p] for p in positions)
    return merged


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def show_clusters(
    sentences: List[str],
    labels: List[int],
    max_examples: int = 3,
) -> None:
    """Print cluster sizes and representative examples."""
    by_cluster: Dict[int, List[str]] = defaultdict(list)
    for s, lab in zip(sentences, labels):
        by_cluster[lab].append(s)

    for cid in sorted(by_cluster):
        members = by_cluster[cid]
        print(f"\n--- Cluster {cid}  ({len(members)} sentences) ---")
        for s in members[:max_examples]:
            print(f"  • {s[:120]}")
        if len(members) > max_examples:
            print(f"  ... and {len(members) - max_examples} more")


def show_anchor_distribution(
    classifications: List[str],
) -> Dict[str, int]:
    """Print and return category frequency table."""
    counts = Counter(classifications)
    total = len(classifications)
    print(f"\nAnchor distribution ({total} sentences):")
    for code in sorted(ANCHOR_CATEGORIES):
        n = counts.get(code, 0)
        name = ANCHOR_CATEGORIES[code]
        bar = "█" * int(40 * n / total) if total else ""
        print(f"  {code} ({name:20s}): {n:4d}  {bar}")
    return dict(counts)
