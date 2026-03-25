#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Titan Retrieval Layer - Context Builder

Retrieves and ranks relevant context using:
1. Reciprocal Rank Fusion (RRF) for multi-signal ranking
2. Time-weighted confidence decay
3. Graph-aware expansion (SYNAPSE-inspired spreading activation)

Core principle: Context assembly only includes items where
confidence x recency > threshold
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

from .storage import SQLiteStore, VectorStore, KnowledgeGraph, Node

LOG = logging.getLogger("titan.retrieval")

# Time decay constants
DECAY_HALF_LIFE_DAYS = 7
DECAY_MINIMUM = 0.1

# RRF constant
RRF_K = 60

# Retrieval thresholds
MIN_EFFECTIVE_CONFIDENCE = 0.15
MAX_CONTEXT_LENGTH = 4000


@dataclass
class RetrievedItem:
    """An item retrieved from memory."""
    node_id: str
    label: str
    node_type: str
    content: str
    base_confidence: float
    effective_confidence: float
    recency_score: float
    vector_score: float = 0.0
    fts_score: float = 0.0
    graph_score: float = 0.0
    final_score: float = 0.0
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "label": self.label,
            "type": self.node_type,
            "content": self.content,
            "confidence": self.effective_confidence,
            "score": self.final_score,
        }


def calculate_time_decay(created_at: str, half_life_days: float = DECAY_HALF_LIFE_DAYS) -> float:
    """Calculate time decay factor using exponential decay: 2^(-t/half_life)."""
    try:
        created = datetime.fromisoformat(created_at)
        now = datetime.now()
        age_days = (now - created).total_seconds() / 86400
        decay = math.pow(2, -age_days / half_life_days)
        return max(decay, DECAY_MINIMUM)
    except Exception:
        return 1.0


def calculate_recency_score(created_at: str) -> float:
    """Calculate recency score (0 to 1). More recent = higher."""
    try:
        created = datetime.fromisoformat(created_at)
        now = datetime.now()
        age_days = (now - created).total_seconds() / 86400
        return math.exp(-age_days / 30)
    except Exception:
        return 0.5


def reciprocal_rank_fusion(rankings: List[List[Tuple[str, float]]],
                            k: int = RRF_K) -> Dict[str, float]:
    """
    Combine multiple rankings using Reciprocal Rank Fusion.
    RRF score = sum(1 / (k + rank)) for each ranking list
    """
    fused_scores = {}
    for ranking in rankings:
        for rank, (node_id, _) in enumerate(ranking, start=1):
            if node_id not in fused_scores:
                fused_scores[node_id] = 0.0
            fused_scores[node_id] += 1.0 / (k + rank)
    return fused_scores


class ContextBuilder:
    """
    Builds context from memory for queries.

    Uses multiple retrieval signals:
    1. Vector similarity (semantic meaning)
    2. FTS (keyword matching)
    3. Graph traversal (relationships)
    4. Recency (time-weighted)
    """

    def __init__(self, sqlite: SQLiteStore, vectors: VectorStore,
                 graph: KnowledgeGraph):
        self.sqlite = sqlite
        self.vectors = vectors
        self.graph = graph

    def retrieve(self, query: str, limit: int = 10,
                 min_confidence: float = MIN_EFFECTIVE_CONFIDENCE) -> List[RetrievedItem]:
        """
        Retrieve relevant items for a query.
        Uses raw similarity scores as primary signal, with confidence/recency
        as secondary tiebreakers. Entity-only nodes are filtered out.
        """
        if not query or not query.strip():
            return []

        # ── Dual-path retrieval: FTS-first + Vector-semantic ──────────
        #
        # Key insight: for factual queries with specific keywords ("What port
        # does auth use?"), FTS is far more precise than vector similarity.
        # For semantic queries ("Tell me about the infrastructure"), vectors win.
        #
        # Strategy: Run both, rank by FTS-rank (keyword precision) when available,
        # fall back to vector similarity for items FTS doesn't find.

        # 1. FTS search — keyword precision, ordered by BM25 rank
        fts_results = self.sqlite.search_fts(query, limit=limit * 3)
        # Build ordered FTS ranking: position 1 = best match
        fts_rank = {}
        if fts_results:
            for pos, (nid, _raw_rank) in enumerate(fts_results):
                fts_rank[nid] = pos  # lower = better

        # 2. Vector similarity search — semantic recall
        vector_results = self.vectors.search(query, limit=limit * 3)
        vec_lookup = {}
        if vector_results:
            for nid, sim in vector_results:
                vec_lookup[nid] = sim

        # 3. Merge candidates
        all_candidates = set(fts_rank.keys()) | set(vec_lookup.keys())
        if not all_candidates:
            return []

        # --- Neural Cortex ---
        _cortex = None
        try:
            from .neural_cortex import get_cortex
            _cortex = get_cortex()
        except Exception:
            pass

        # 4. RRF fusion: combine FTS and vector RANKS (not scores)
        #    The key fix: all scoring is rank-based. Neither FTS raw scores
        #    nor vector cosine similarities are used directly — only positions.
        #    This prevents any single item from dominating via score inflation.
        #    Confidence and recency are demoted to minor tiebreakers (10% each).
        items = []
        _last_rwl_weights = [0.8, 0.08, 0.08, 0.04]

        # Build rank lookups (position 0 = best)
        vec_ranked = sorted(vec_lookup.items(), key=lambda x: -x[1])
        vec_rank_map = {nid: pos for pos, (nid, _) in enumerate(vec_ranked)}

        rrf_k = 10  # Lower k = more weight to top positions

        for node_id in all_candidates:
            node = self.sqlite.get_node(node_id)
            if not node:
                continue

            # Skip entity-only nodes
            if node.type == "entity" and not node.metadata.get("full_text"):
                continue

            time_decay = calculate_time_decay(node.created_at)
            base_confidence = node.metadata.get("confidence", 0.5)
            effective_confidence = base_confidence * time_decay

            if effective_confidence < min_confidence:
                continue

            recency_score = calculate_recency_score(node.created_at)
            vector_score = vec_lookup.get(node_id, 0.0)

            # RRF: sum of 1/(k+rank) for each ranking list
            relevance = 0.0
            if node_id in vec_rank_map:
                relevance += 1.0 / (rrf_k + vec_rank_map[node_id] + 1)
            if node_id in fts_rank:
                relevance += 1.0 / (rrf_k + fts_rank[node_id] + 1)

            degree = self.sqlite.get_node_degree(node_id)
            graph_score = min(1.0, degree / 10.0)

            content = node.metadata.get("full_text", node.label)

            # Scoring: relevance is the primary signal.
            # Confidence/recency/graph are micro-tiebreakers (max 1% each).
            # This ensures the ranking is driven by actual query-item match,
            # not by metadata that is identical across all items.
            tiebreaker = (
                effective_confidence * 0.003 +
                recency_score * 0.003 +
                graph_score * 0.002
            )

            # Neural Cortex: only apply learned weights when MATURE (300+ steps).
            # During cold start, the RWL network produces near-random weights
            # that destroy the relevance ranking.
            _cortex_boost = 0.0
            if _cortex:
                try:
                    rwl_steps = getattr(_cortex, '_rwl_steps', 0)
                    if rwl_steps >= 300:
                        _rwl_f = {
                            "rrf": relevance, "conf": effective_confidence,
                            "recency": recency_score, "graph": graph_score,
                            "query_len": len(query.split()),
                            "n_results": len(all_candidates),
                            "valence": node.metadata.get("valence", 0.0),
                            "arousal": node.metadata.get("arousal", 0.5),
                        }
                        _weights = _cortex.get_retrieval_weights(_rwl_f)
                        _last_rwl_weights = _weights
                        # Cortex re-weights the tiebreaker portion only
                        tiebreaker = (
                            effective_confidence * _weights[1] * 0.05 +
                            recency_score * _weights[2] * 0.05 +
                            graph_score * _weights[3] * 0.05
                        )
                except Exception:
                    pass

            final_score = relevance + tiebreaker

            items.append(RetrievedItem(
                node_id=node_id, label=node.label,
                node_type=node.type, content=content,
                base_confidence=base_confidence,
                effective_confidence=effective_confidence,
                recency_score=recency_score,
                vector_score=vector_score,
                fts_score=1.0 / (rrf_k + fts_rank.get(node_id, 999) + 1),
                graph_score=graph_score,
                final_score=final_score,
                metadata=node.metadata
            ))

        items.sort(key=lambda x: x.final_score, reverse=True)
        result = items[:limit]

        # Neural Cortex: log retrieval access + co-retrieval + RWL feedback
        try:
            if _cortex and result:
                _ids = [it.node_id for it in result]
                _scores = [it.final_score for it in result]
                _cortex.log_access(_ids, query, _scores)
                _cortex.log_co_retrieval(_ids)

                try:
                    _avg_conf = sum(it.effective_confidence for it in result) / len(result)
                    _avg_score = sum(it.final_score for it in result) / len(result)
                    _reward = _avg_conf * 0.6 + _avg_score * 0.4
                    _rwl_fb_features = {
                        "rrf": result[0].vector_score,
                        "conf": result[0].effective_confidence,
                        "recency": result[0].recency_score,
                        "graph": result[0].graph_score,
                        "query_len": len(query.split()),
                        "n_results": len(result),
                        "valence": result[0].metadata.get("valence", 0.0),
                        "arousal": result[0].metadata.get("arousal", 0.5),
                    }
                    _cortex.record_rwl_feedback(_last_rwl_weights, _rwl_fb_features, _reward)
                except Exception:
                    pass
        except Exception:
            pass

        return result

    def expand_with_graph(self, items: List[RetrievedItem],
                           hops: int = 1) -> List[RetrievedItem]:
        """
        Expand retrieved items using SYNAPSE-inspired spreading activation.

        Instead of simple BFS, activation energy propagates through the graph
        weighted by edge confidence, with fan-effect dilution and lateral
        inhibition.
        """
        if not items:
            return items

        # Phase 1: Initialize activation from seed nodes
        activation: Dict[str, float] = {}
        seed_ids = set()
        for item in items:
            activation[item.node_id] = item.final_score
            seed_ids.add(item.node_id)

        # Phase 2: Propagation (T=3 iterations)
        T_ITERATIONS = 3
        TEMPORAL_DECAY_RATE = 0.01

        conn = self.sqlite._get_conn()

        for t in range(T_ITERATIONS):
            new_activation: Dict[str, float] = {}

            for node_id, energy in activation.items():
                if energy < 0.01:
                    continue

                edges = conn.execute(
                    "SELECT dst, confidence FROM edges WHERE src = ? AND confidence > 0.2",
                    (node_id,)
                ).fetchall()
                edges += conn.execute(
                    "SELECT src, confidence FROM edges WHERE dst = ? AND confidence > 0.2",
                    (node_id,)
                ).fetchall()

                if not edges:
                    continue

                fan_factor = 1.0 / max(len(edges), 1)

                for neighbor_id, edge_conf in edges:
                    propagated = energy * edge_conf * fan_factor

                    try:
                        neighbor_node = self.sqlite.get_node(neighbor_id)
                        if neighbor_node:
                            src_node = self.sqlite.get_node(node_id)
                            if src_node:
                                src_time = datetime.fromisoformat(src_node.created_at)
                                dst_time = datetime.fromisoformat(neighbor_node.created_at)
                                age_diff = abs((src_time - dst_time).total_seconds() / 86400)
                                propagated *= math.exp(-TEMPORAL_DECAY_RATE * age_diff)
                    except Exception:
                        pass

                    if neighbor_id in new_activation:
                        new_activation[neighbor_id] = max(new_activation[neighbor_id], propagated)
                    else:
                        new_activation[neighbor_id] = propagated

            for nid in activation:
                activation[nid] *= 0.7
            for nid, val in new_activation.items():
                if nid in activation:
                    activation[nid] = max(activation[nid], val)
                else:
                    activation[nid] = val

        # Phase 3: Lateral Inhibition (top-7 winner-take-all)
        INHIBITION_TOP_K = 7
        INHIBITION_BETA = 0.15

        sorted_nodes = sorted(activation.items(), key=lambda x: x[1], reverse=True)
        top_k_ids = set(nid for nid, _ in sorted_nodes[:INHIBITION_TOP_K])

        for nid in list(activation.keys()):
            if nid not in top_k_ids and nid not in seed_ids:
                activation[nid] *= (1.0 - INHIBITION_BETA)

        # Phase 4: Sigmoid gating
        SIGMOID_STEEPNESS = 5.0
        SIGMOID_THRESHOLD = 0.3

        def sigmoid_gate(x):
            return 1.0 / (1.0 + math.exp(-SIGMOID_STEEPNESS * (x - SIGMOID_THRESHOLD)))

        for nid in list(activation.keys()):
            activation[nid] = sigmoid_gate(activation[nid])

        # Phase 5: Build expanded items
        expanded_ids = set(item.node_id for item in items)
        expanded_items = list(items)
        activated_count = 0

        for node_id, act_score in sorted(activation.items(), key=lambda x: x[1], reverse=True):
            if node_id in expanded_ids:
                continue
            if act_score < 0.1:
                continue
            if activated_count >= hops * 5:
                break

            node = self.sqlite.get_node(node_id)
            if not node:
                continue

            time_decay = calculate_time_decay(node.created_at)
            base_confidence = node.metadata.get("confidence", 0.5)
            effective_confidence = base_confidence * time_decay

            if effective_confidence < MIN_EFFECTIVE_CONFIDENCE:
                continue

            parent_score = max((it.final_score for it in items), default=0.5)
            final = parent_score * act_score * 0.6

            expanded_items.append(RetrievedItem(
                node_id=node_id, label=node.label,
                node_type=node.type,
                content=node.metadata.get("full_text", node.label),
                base_confidence=base_confidence,
                effective_confidence=effective_confidence,
                recency_score=calculate_recency_score(node.created_at),
                graph_score=act_score, final_score=final,
                metadata=node.metadata
            ))

            expanded_ids.add(node_id)
            activated_count += 1

        return expanded_items

    def build_context_string(self, items: List[RetrievedItem],
                              max_length: int = MAX_CONTEXT_LENGTH) -> str:
        """Build a context string from retrieved items."""
        if not items:
            return ""

        lines = []
        current_length = 0

        for item in items:
            if item.effective_confidence >= 0.8:
                certainty = "certain"
            elif item.effective_confidence >= 0.6:
                certainty = "likely"
            elif item.effective_confidence >= 0.4:
                certainty = "possible"
            else:
                certainty = "uncertain"

            line = f"[{certainty}] {item.content}"

            if current_length + len(line) > max_length:
                break

            lines.append(line)
            current_length += len(line) + 1

        return "\n".join(lines)

    def get_related_claims(self, entity: str) -> List[dict]:
        """Get claims related to an entity."""
        conn = self.sqlite._get_conn()
        claims = []

        rows = conn.execute("""
            SELECT * FROM claims
            WHERE subject LIKE ? OR object LIKE ?
            ORDER BY created_at DESC
            LIMIT 20
        """, (f"%{entity}%", f"%{entity}%")).fetchall()

        for row in rows:
            claims.append({
                "subject": row["subject"],
                "predicate": row["predicate"],
                "object": row["object"],
                "confidence": row["confidence"],
                "origin": row["origin"],
                "created_at": row["created_at"]
            })

        return claims


class Retriever:
    """High-level retrieval interface with caching."""

    def __init__(self, sqlite: SQLiteStore, vectors: VectorStore,
                 graph: KnowledgeGraph):
        self.builder = ContextBuilder(sqlite, vectors, graph)
        self._cache: Dict[str, Tuple[List[RetrievedItem], float]] = {}
        self._cache_ttl = 60

    def retrieve(self, query: str, limit: int = 5,
                 expand_graph: bool = True) -> List[RetrievedItem]:
        """Retrieve relevant context for a query."""
        cache_key = f"{query}:{limit}:{expand_graph}"
        if cache_key in self._cache:
            items, timestamp = self._cache[cache_key]
            if datetime.now().timestamp() - timestamp < self._cache_ttl:
                return items

        items = self.builder.retrieve(query, limit=limit)

        if expand_graph and items:
            items = self.builder.expand_with_graph(items, hops=1)
            items.sort(key=lambda x: x.final_score, reverse=True)
            items = items[:limit]

        self._cache[cache_key] = (items, datetime.now().timestamp())
        return items

    def get_context_string(self, query: str, limit: int = 5) -> str:
        """Get formatted context string for a query."""
        items = self.retrieve(query, limit=limit)
        return self.builder.build_context_string(items)

    def clear_cache(self):
        """Clear the retrieval cache."""
        self._cache.clear()


_retriever: Optional[Retriever] = None


def get_retriever(sqlite: SQLiteStore, vectors: VectorStore,
                  graph: KnowledgeGraph) -> Retriever:
    """Get or create retriever instance."""
    global _retriever
    if _retriever is None:
        _retriever = Retriever(sqlite, vectors, graph)
    return _retriever
