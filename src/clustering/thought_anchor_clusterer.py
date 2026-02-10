"""
3-Stage Thought Anchor clustering pipeline.

Stage 1: LLM-based classification into 8 thought anchor categories
Stage 2: Intra-category embedding clustering (agglomerative)
Stage 3: LLM super-clustering to merge subclusters within each category

Produces ~8-15 final nodes from ~5000 sentences.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import json
import time
from dataclasses import dataclass, field
from collections import defaultdict
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from .sentence_then_llm_clusterer import SentenceThenLLMClusterer, Cluster


# Thought anchor category taxonomy
ANCHOR_CATEGORIES = {
    "PS": "Problem Setup",
    "FR": "Fact Retrieval",
    "AC": "Active Reasoning",
    "UM": "Uncertainty",
    "RC": "Consolidation",
    "SC": "Self Checking",
    "FA": "Final Answer",
    "UH": "Uses Hint",
}

CLASSIFICATION_PROMPT = """Classify each sentence into exactly one category from this taxonomy:
- PS (Problem Setup): Parsing, rephrasing the problem, planning approach
- FR (Fact Retrieval): Recalling facts, formulas, definitions
- AC (Active Reasoning): Analysis, computation, evaluating/eliminating options
- UM (Uncertainty): Confusion, hedging, backtracking, 'Wait...'
- RC (Consolidation): Summarizing intermediate results, narrowing down
- SC (Self Checking): Verifying, re-confirming previous steps
- FA (Final Answer): Stating the final answer
- UH (Uses Hint): References a cue, hint, or authority figure

Return ONLY a JSON array of category codes, one per sentence. Example: ["PS", "AC", "FR"]

Sentences:
{sentences}"""

SUPER_CLUSTER_PROMPT = """You are merging subclusters within the "{category_name}" category of chain-of-thought reasoning.

Below are {n_clusters} subclusters. Each has a representative sentence and size.
Decide which subclusters should be merged because they represent the same reasoning step.

Return a JSON array of arrays, where each inner array contains the subcluster indices (0-based) that should be merged together. Singletons that shouldn't merge with anything should appear as single-element arrays.

Example: [[0, 2], [1], [3, 4]] means merge subclusters 0+2, keep 1 alone, merge 3+4.

Target: reduce to at most {target} groups.

Subclusters:
{subclusters}"""


class ThoughtAnchorClusterer(SentenceThenLLMClusterer):
    """3-stage pipeline: LLM classification -> embedding clustering -> LLM super-clustering."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.classification_model = config.get(
            "classification_model", "google/gemini-3-flash-preview"
        )
        self.classification_batch_size = config.get("classification_batch_size", 30)
        self.intra_category_threshold = config.get("intra_category_threshold", 0.55)
        self.target_clusters_per_category = config.get("target_clusters_per_category", 2)

    def cluster_responses(
        self,
        responses: Dict[str, Any],
        prompt_index: str,
        model: str,
    ) -> Dict[str, int]:
        """3-stage thought anchor clustering pipeline."""

        print(f"Clustering {len(responses)} responses using thought-anchor pipeline")

        # Extract all sentences
        sentences = []
        seed_to_sentences = {}
        content_key = self._get_content_key()

        for seed, response_data in responses.items():
            content = response_data.get(content_key, None)
            if not content:
                content = response_data.get("cot_content", "")
            if content:
                if isinstance(content, list):
                    response_sentences = content
                else:
                    response_sentences = [content]
                sentences.extend(response_sentences)
                seed_to_sentences[seed] = response_sentences

        if not sentences:
            raise ValueError("No sentences found in responses")

        print(f"Extracted {len(sentences)} sentences from {len(seed_to_sentences)} rollouts")

        # Generate embeddings for all sentences (reused in Stage 2)
        print("Generating sentence embeddings...")
        embeddings = self.sentence_model.encode(sentences, normalize_embeddings=True)
        print(f"Generated embeddings with shape: {embeddings.shape}")

        # Stage 1: Classify sentences into thought anchor categories
        print("\n=== Stage 1: Thought Anchor Classification ===")
        classifications = self._classify_sentences_batch(sentences)
        self._print_classification_summary(classifications)

        # Stage 2: Intra-category embedding clustering
        print("\n=== Stage 2: Intra-Category Embedding Clustering ===")
        category_clusters, global_labels = self._cluster_within_categories(
            sentences, embeddings, classifications
        )
        n_subclusters = len(set(global_labels))
        print(f"Stage 2 produced {n_subclusters} subclusters across all categories")

        # Stage 3: LLM super-clustering (skipped for now)
        # Build Cluster objects directly from Stage 2 labels
        global_id_to_sentences: Dict[str, List[int]] = defaultdict(list)
        for sent_idx, gid in enumerate(global_labels):
            global_id_to_sentences[gid].append(sent_idx)

        merged_clusters = []
        for gid in sorted(global_id_to_sentences.keys()):
            sent_indices = global_id_to_sentences[gid]
            cluster = Cluster(
                sentences=[sentences[i] for i in sent_indices],
                id=str(gid),
                merged_from=[str(gid)],
            )
            merged_clusters.append(cluster)

        final_labels = global_labels
        n_final = len(merged_clusters)
        print(f"\nFinal result: {n_final} clusters from {len(sentences)} sentences")

        # Store state for edge creation (same interface as parent)
        self.sentence_clusters = final_labels
        self.seed_to_sentences = seed_to_sentences
        self.merged_clusters = merged_clusters

        # Create rollout edges
        print("\nCreating rollout edges...")
        rollout_edges = self._create_rollout_edges(responses, merged_clusters, prompt_index)
        self.rollout_edges = rollout_edges

        # Store classifications for node metadata
        self._sentence_classifications = classifications

        cluster_assignments = {seed: 0 for seed in responses.keys()}
        return cluster_assignments

    def create_flowchart(
        self,
        responses: Dict[str, Any],
        cluster_assignments: Dict[str, int],
        prompt_index: str,
        models: List[str],
        config_name: str,
        property_checker_names: List[str] = None,
        node_property_checker_names: List[str] = None,
    ) -> Dict[str, Any]:
        """Create flowchart with anchor_category metadata on each node."""
        flowchart = super().create_flowchart(
            responses,
            cluster_assignments,
            prompt_index,
            models,
            config_name,
            property_checker_names,
            node_property_checker_names,
        )

        flowchart["clustering_method"] = "thought_anchor"

        # Add anchor_category to each cluster node
        for node_obj in flowchart["nodes"]:
            cluster_key = list(node_obj.keys())[0]
            if not cluster_key.startswith("cluster-"):
                continue
            node_data = node_obj[cluster_key]

            # Determine dominant category from sentences in this cluster
            category_counts = defaultdict(int)
            for sent_data in node_data.get("sentences", []):
                text = sent_data["text"]
                # Look up classification
                if hasattr(self, "_sentence_classifications"):
                    for i, s in enumerate(self._all_sentences):
                        if s == text and i < len(self._sentence_classifications):
                            cat = self._sentence_classifications[i]
                            category_counts[cat] += sent_data.get("count", 1)
                            break

            if category_counts:
                dominant = max(category_counts, key=category_counts.get)
                node_data["anchor_category"] = dominant
                node_data["anchor_category_name"] = ANCHOR_CATEGORIES.get(dominant, dominant)

        return flowchart

    # ─── Stage 1: Thought Anchor Classification ─────────────────────────

    def _classify_sentences_batch(self, sentences: List[str]) -> List[str]:
        """Classify all sentences into thought anchor categories via batched LLM calls."""
        valid_codes = set(ANCHOR_CATEGORIES.keys())
        classifications = []
        batch_size = self.classification_batch_size
        batches = [
            sentences[i : i + batch_size] for i in range(0, len(sentences), batch_size)
        ]

        print(
            f"Classifying {len(sentences)} sentences in {len(batches)} batches "
            f"(batch_size={batch_size}, model={self.classification_model})"
        )

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx = {
                executor.submit(self._classify_batch, batch, batch_idx): batch_idx
                for batch_idx, batch in enumerate(batches)
            }

            results = {}
            for future in tqdm(
                as_completed(future_to_idx),
                total=len(batches),
                desc="Classifying sentences",
            ):
                batch_idx = future_to_idx[future]
                try:
                    batch_labels = future.result()
                    results[batch_idx] = batch_labels
                except Exception as e:
                    print(f"Batch {batch_idx} failed: {e}")
                    # Default to AC for failed batches
                    results[batch_idx] = ["AC"] * len(batches[batch_idx])

        # Reassemble in order
        for batch_idx in range(len(batches)):
            batch_labels = results.get(batch_idx, ["AC"] * len(batches[batch_idx]))
            # Validate labels
            for label in batch_labels:
                if label in valid_codes:
                    classifications.append(label)
                else:
                    classifications.append("AC")  # Default fallback

        # Store for flowchart metadata
        self._all_sentences = sentences

        assert len(classifications) == len(sentences), (
            f"Classification count mismatch: {len(classifications)} vs {len(sentences)}"
        )
        return classifications

    def _classify_batch(self, batch: List[str], batch_idx: int) -> List[str]:
        """Classify a single batch of sentences via LLM."""
        numbered = "\n".join(f"{i+1}. {s}" for i, s in enumerate(batch))
        prompt = CLASSIFICATION_PROMPT.format(sentences=numbered)

        with self._lock:
            time.sleep(self.request_delay)

        payload = {
            "model": self.classification_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1024,
            "temperature": 0.1,
        }

        import requests

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            data=json.dumps(payload),
            timeout=120,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"].strip()

        # Parse JSON array from response (handle markdown code blocks)
        if "```" in content:
            # Extract content between code fences
            lines = content.split("\n")
            json_lines = []
            in_block = False
            for line in lines:
                if line.strip().startswith("```"):
                    in_block = not in_block
                    continue
                if in_block:
                    json_lines.append(line)
            content = "\n".join(json_lines)

        labels = json.loads(content)

        # Pad or truncate to match batch size
        if len(labels) < len(batch):
            labels.extend(["AC"] * (len(batch) - len(labels)))
        elif len(labels) > len(batch):
            labels = labels[: len(batch)]

        return labels

    def _print_classification_summary(self, classifications: List[str]):
        """Print distribution of thought anchor categories."""
        from collections import Counter

        counts = Counter(classifications)
        total = len(classifications)
        print(f"\nThought Anchor Distribution ({total} sentences):")
        for code in ANCHOR_CATEGORIES:
            count = counts.get(code, 0)
            pct = 100 * count / total if total > 0 else 0
            bar = "#" * int(pct / 2)
            print(f"  {code} ({ANCHOR_CATEGORIES[code]:20s}): {count:5d} ({pct:5.1f}%) {bar}")

    # ─── Stage 2: Intra-Category Embedding Clustering ───────────────────

    def _cluster_within_categories(
        self,
        sentences: List[str],
        embeddings: np.ndarray,
        classifications: List[str],
    ) -> Tuple[Dict[str, List[Tuple[int, int]]], List[int]]:
        """Cluster sentences within each anchor category using agglomerative clustering.

        Returns:
            category_clusters: dict mapping category code -> list of (global_sentence_idx, local_cluster_id)
            global_labels: list of global cluster IDs (one per sentence)
        """
        # Group sentence indices by category
        category_indices: Dict[str, List[int]] = defaultdict(list)
        for i, cat in enumerate(classifications):
            category_indices[cat].append(i)

        global_labels = [0] * len(sentences)
        category_clusters: Dict[str, List[Tuple[int, int]]] = {}
        next_global_id = 0

        for cat_code in sorted(category_indices.keys()):
            indices = category_indices[cat_code]
            if not indices:
                continue

            cat_embeddings = embeddings[indices]

            if len(indices) == 1:
                # Single sentence -> single cluster
                local_labels = [0]
            else:
                local_labels = self._cluster_by_agglomerative(
                    cat_embeddings, self.intra_category_threshold
                )

            # Map local cluster IDs to global IDs
            local_to_global = {}
            pairs = []
            for local_idx, local_label in enumerate(local_labels):
                if local_label not in local_to_global:
                    local_to_global[local_label] = next_global_id
                    next_global_id += 1
                global_id = local_to_global[local_label]
                global_sentence_idx = indices[local_idx]
                global_labels[global_sentence_idx] = global_id
                pairs.append((global_sentence_idx, local_label))

            category_clusters[cat_code] = pairs
            n_local = len(set(local_labels))
            print(
                f"  {cat_code} ({ANCHOR_CATEGORIES.get(cat_code, '?'):20s}): "
                f"{len(indices):5d} sentences -> {n_local:3d} subclusters"
            )

        return category_clusters, global_labels

    # ─── Stage 3: LLM Super-Clustering ──────────────────────────────────

    def _super_cluster_categories(
        self,
        sentences: List[str],
        embeddings: np.ndarray,
        classifications: List[str],
        category_clusters: Dict[str, List[Tuple[int, int]]],
        global_labels: List[int],
    ) -> Tuple[List[Cluster], List[int]]:
        """LLM-based merging of subclusters within each category.

        Returns:
            merged_clusters: list of final Cluster objects
            final_labels: updated global label for each sentence
        """
        # Build subcluster objects grouped by category
        # global_id -> list of sentence indices
        global_id_to_sentences: Dict[int, List[int]] = defaultdict(list)
        for sent_idx, gid in enumerate(global_labels):
            global_id_to_sentences[gid].append(sent_idx)

        # Group global IDs by category
        category_to_global_ids: Dict[str, List[int]] = defaultdict(list)
        for cat_code, pairs in category_clusters.items():
            seen = set()
            for sent_idx, local_label in pairs:
                gid = global_labels[sent_idx]
                if gid not in seen:
                    category_to_global_ids[cat_code].append(gid)
                    seen.add(gid)

        # For each category, ask LLM which subclusters to merge
        merge_map: Dict[int, int] = {}  # old global_id -> new global_id
        final_clusters: List[Cluster] = []
        next_final_id = 0

        for cat_code in sorted(category_to_global_ids.keys()):
            gids = category_to_global_ids[cat_code]

            if len(gids) <= self.target_clusters_per_category:
                # Already at or below target, no merging needed
                for gid in gids:
                    merge_map[gid] = next_final_id
                    sent_indices = global_id_to_sentences[gid]
                    cluster_sentences = [sentences[i] for i in sent_indices]
                    cluster = Cluster(
                        sentences=cluster_sentences,
                        id=str(next_final_id),
                        merged_from=[str(gid)],
                    )
                    final_clusters.append(cluster)
                    next_final_id += 1
                print(
                    f"  {cat_code}: {len(gids)} subclusters <= target "
                    f"({self.target_clusters_per_category}), keeping as-is"
                )
                continue

            # Build representative info for each subcluster
            subcluster_info = []
            for gid in gids:
                sent_indices = global_id_to_sentences[gid]
                cluster_sentences = [sentences[i] for i in sent_indices]
                # Pick most central sentence as representative
                if len(sent_indices) > 1:
                    cluster_embs = embeddings[sent_indices]
                    centroid = cluster_embs.mean(axis=0)
                    centroid /= np.linalg.norm(centroid) + 1e-8
                    sims = cluster_embs @ centroid
                    rep_idx = sent_indices[int(np.argmax(sims))]
                else:
                    rep_idx = sent_indices[0]
                subcluster_info.append({
                    "gid": gid,
                    "size": len(sent_indices),
                    "representative": sentences[rep_idx],
                    "sentences": cluster_sentences,
                })

            # Call LLM for merge decisions
            merge_groups = self._call_llm_for_super_cluster(
                cat_code, subcluster_info
            )

            for group in merge_groups:
                merged_gids = [gids[i] for i in group if i < len(gids)]
                if not merged_gids:
                    continue

                all_sents = []
                merged_from = []
                for gid in merged_gids:
                    sent_indices = global_id_to_sentences[gid]
                    all_sents.extend(sentences[i] for i in sent_indices)
                    merged_from.append(str(gid))
                    merge_map[gid] = next_final_id

                cluster = Cluster(
                    sentences=all_sents,
                    id=str(next_final_id),
                    merged_from=merged_from,
                )
                final_clusters.append(cluster)
                next_final_id += 1

            # Handle any gids not covered by merge_groups
            covered = set()
            for group in merge_groups:
                for i in group:
                    if i < len(gids):
                        covered.add(gids[i])
            for gid in gids:
                if gid not in covered:
                    merge_map[gid] = next_final_id
                    sent_indices = global_id_to_sentences[gid]
                    cluster = Cluster(
                        sentences=[sentences[i] for i in sent_indices],
                        id=str(next_final_id),
                        merged_from=[str(gid)],
                    )
                    final_clusters.append(cluster)
                    next_final_id += 1

            print(
                f"  {cat_code}: {len(gids)} subclusters -> "
                f"{len([g for g in merge_groups if any(gids[i] in merge_map for i in g if i < len(gids))])} groups"
            )

        # Update global labels
        final_labels = []
        for sent_idx, old_gid in enumerate(global_labels):
            final_labels.append(merge_map.get(old_gid, old_gid))

        print(f"\nStage 3 result: {len(final_clusters)} final clusters")
        return final_clusters, final_labels

    def _call_llm_for_super_cluster(
        self,
        category_code: str,
        subcluster_info: List[Dict],
    ) -> List[List[int]]:
        """Ask LLM which subclusters within a category should be merged."""
        category_name = ANCHOR_CATEGORIES.get(category_code, category_code)

        # Format subcluster descriptions
        descriptions = []
        for i, info in enumerate(subcluster_info):
            desc = f"[{i}] (size={info['size']}): \"{info['representative']}\""
            # Add a few more examples if cluster is large
            if info["size"] > 3:
                extras = [s for s in info["sentences"][:4] if s != info["representative"]]
                if extras:
                    desc += "\n     Also: " + " | ".join(f'"{e}"' for e in extras[:3])
            descriptions.append(desc)

        prompt = SUPER_CLUSTER_PROMPT.format(
            category_name=category_name,
            n_clusters=len(subcluster_info),
            target=self.target_clusters_per_category,
            subclusters="\n".join(descriptions),
        )

        with self._lock:
            time.sleep(self.request_delay)

        payload = {
            "model": self.llm_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1024,
            "temperature": 0.1,
        }

        try:
            import requests

            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                data=json.dumps(payload),
                timeout=120,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"].strip()

            # Parse JSON (handle code fences)
            if "```" in content:
                lines = content.split("\n")
                json_lines = []
                in_block = False
                for line in lines:
                    if line.strip().startswith("```"):
                        in_block = not in_block
                        continue
                    if in_block:
                        json_lines.append(line)
                content = "\n".join(json_lines)

            merge_groups = json.loads(content)

            # Validate: must be list of lists of ints
            if not isinstance(merge_groups, list):
                raise ValueError(f"Expected list, got {type(merge_groups)}")
            validated = []
            for group in merge_groups:
                if isinstance(group, list) and all(isinstance(x, int) for x in group):
                    validated.append(group)
            return validated

        except Exception as e:
            print(f"Super-cluster LLM call failed for {category_code}: {e}")
            # Fallback: keep all subclusters separate
            return [[i] for i in range(len(subcluster_info))]
