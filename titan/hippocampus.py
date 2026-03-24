"""
Hippocampus -- Neural Retrieval Enhancement Layer for Titan
===========================================================
Bio-inspired associative memory that transforms Titan from a database
into something closer to remembering.

Architecture (4 neural modules, ~226K params, <20ms on CPU):
  QUM  -- Query Understanding Module (Entorhinal Cortex)
  AMM  -- Associative Memory Module (CA3 Hopfield Network)
  CDM  -- Correlation Discovery Module (CA1 Cross-Attention)
  RCM  -- Response Composer Module (Prefrontal Output)

Training only during consolidation phases.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

LOG = logging.getLogger("titan.hippocampus")


# ====================================================================
#  Constants
# ====================================================================
MAX_CLUSTERS = 64
MAX_CANDIDATES = 50
TOP_K_RESULTS = 10
EMBEDDING_DIM = 384
CLUSTER_META_DIM = 16
INPUT_DIM = EMBEDDING_DIM + CLUSTER_META_DIM  # 400

# Cold start phases: (step_threshold, fallback_weight)
_COLD_START_PHASES = [
    (50,  1.00),
    (150, 0.50),
    (300, 0.25),
]
_MATURE_FALLBACK = 0.05

# Training
_LR = 5e-4
_WEIGHT_DECAY = 5e-5
_GRAD_CLIP = 1.0
_BUFFER_MIN_TRAIN = 30

# QUM input dimensions
_QUM_TEXT_FEATURES = 12
_QUM_CONTEXT_FEATURES = 12
_QUM_EPQ_FEATURES = 4
_QUM_TEMPORAL_FEATURES = 4
_QUM_SESSION_FEATURES = 6
QUM_INPUT_DIM = (_QUM_TEXT_FEATURES + _QUM_CONTEXT_FEATURES +
                 _QUM_EPQ_FEATURES + _QUM_TEMPORAL_FEATURES +
                 _QUM_SESSION_FEATURES)  # 38

# Temporal words (English + German)
_TEMPORAL_WORDS = frozenset({
    "gestern", "vorgestern", "letzte", "letzten", "letzter", "letztes",
    "woche", "monat", "jahr", "heute", "morgen", "kuerzlich", "neulich",
    "damals", "frueher", "yesterday", "last", "week", "month", "ago",
    "recently", "remember", "erinnerst", "weisst", "noch",
})

# Emotion words
_EMOTION_WORDS = frozenset({
    "freude", "traurig", "wuetend", "angst", "liebe", "hass", "gluecklich",
    "trauer", "schmerz", "lust", "happy", "sad", "angry", "fear", "love",
    "hate", "pain", "joy", "excited", "nervous", "proud", "ashamed",
    "aufgeregt", "stolz", "einsam", "lonely",
})


# ====================================================================
#  DB Schema (training signals)
# ====================================================================
_SCHEMA = """
CREATE TABLE IF NOT EXISTS retrieval_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
    query_hash TEXT NOT NULL,
    query_features TEXT,
    query_text TEXT,
    result_node_ids TEXT,
    result_scores TEXT,
    phase TEXT,
    latency_ms REAL
);
CREATE TABLE IF NOT EXISTS outcome_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
    query_hash TEXT NOT NULL,
    node_id TEXT NOT NULL,
    was_useful INTEGER NOT NULL DEFAULT 0,
    reward REAL DEFAULT 0.0
);
CREATE INDEX IF NOT EXISTS idx_retrieval_ts ON retrieval_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_outcome_ts ON outcome_log(timestamp);
"""

_MIGRATIONS = [
    "ALTER TABLE retrieval_log ADD COLUMN query_text TEXT",
]


# ====================================================================
#  Neural Network Modules
# ====================================================================

if HAS_TORCH:

    class _QUM_Features(nn.Module):
        """Query Understanding -- text/context features branch. ~7.7K params."""
        def __init__(self):
            super().__init__()
            self.trunk = nn.Sequential(
                nn.Linear(QUM_INPUT_DIM, 64), nn.GELU(), nn.LayerNorm(64),
                nn.Linear(64, 48), nn.GELU(), nn.LayerNorm(48),
            )
            self.intent_head = nn.Sequential(nn.Linear(48, 16), nn.Tanh())
            self.temporal_head = nn.Sequential(nn.Linear(48, 2), nn.Sigmoid())
            self.emotional_head = nn.Sequential(nn.Linear(48, 4), nn.Tanh())
            self.expansion_head = nn.Sequential(nn.Linear(48, 16), nn.Tanh())

        def forward(self, x):
            h = self.trunk(x)
            return {
                "intent": self.intent_head(h),
                "temporal": self.temporal_head(h),
                "emotional": self.emotional_head(h),
                "expansion": self.expansion_head(h),
            }

    class _QUM_Expander(nn.Module):
        """Query expansion -- blends embedding with intent. ~50.6K params."""
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(EMBEDDING_DIM + 16, 64), nn.GELU(),
                nn.Linear(64, EMBEDDING_DIM),
            )

        def forward(self, embedding, intent):
            x = torch.cat([embedding, intent], dim=-1)
            return self.net(x)

    class _AMM(nn.Module):
        """Associative Memory -- Modern Hopfield + temporal binding. ~89K params."""
        def __init__(self):
            super().__init__()
            self.shared_proj = nn.Sequential(nn.Linear(INPUT_DIM, 128), nn.GELU())
            self.key_head = nn.Linear(128, 64)
            self.value_head = nn.Linear(128, 64)
            self.query_head = nn.Linear(128, 64)
            self.temporal_binder = nn.Sequential(
                nn.Linear(68, 64), nn.GELU(), nn.Linear(64, 64),
            )
            self.hebbian_gate = nn.Sequential(
                nn.Linear(128, 32), nn.GELU(), nn.Linear(32, 1), nn.Sigmoid(),
            )
            self._beta = 1.0 / math.sqrt(64)

        def forward(self, query_proj, cluster_keys, cluster_values, temporal_features):
            attn_logits = self._beta * query_proj @ cluster_keys.T
            attn_weights = F.softmax(attn_logits, dim=-1)
            retrieved = attn_weights @ cluster_values
            if temporal_features.dim() == 2:
                weighted_temporal = attn_weights @ temporal_features
            else:
                weighted_temporal = temporal_features
            bound = self.temporal_binder(
                torch.cat([retrieved, weighted_temporal], dim=-1)
            )
            return {"attn_weights": attn_weights, "retrieved": retrieved, "bound": bound}

        def project_query(self, query_input):
            h = self.shared_proj(query_input)
            return self.query_head(h)

        def project_clusters(self, cluster_inputs):
            h = self.shared_proj(cluster_inputs)
            return self.key_head(h), self.value_head(h)

        def predict_coactivation(self, query_proj, candidate_proj):
            x = torch.cat([query_proj, candidate_proj], dim=-1)
            return self.hebbian_gate(x)

    class _CDM(nn.Module):
        """Correlation Discovery -- cross-attention re-ranker. ~53K params."""
        def __init__(self):
            super().__init__()
            self.candidate_enc = nn.Sequential(
                nn.Linear(INPUT_DIM, 64), nn.GELU(), nn.LayerNorm(64),
            )
            self.cross_q = nn.Linear(64, 48)
            self.cross_k = nn.Linear(64, 48)
            self.cross_v = nn.Linear(64, 48)
            self.correlation_net = nn.Sequential(
                nn.Linear(104, 64), nn.GELU(), nn.LayerNorm(64),
                nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 4),
            )
            self.final_score = nn.Sequential(nn.Linear(4, 1), nn.Sigmoid())
            self.causal_chain = nn.Sequential(
                nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid(),
            )
            self.emotional_res = nn.Sequential(
                nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 1), nn.Sigmoid(),
            )

        def forward(self, query_enc, candidate_encs, extra_features, emotional_pairs):
            B, N, _ = candidate_encs.shape
            q = self.cross_q(query_enc).unsqueeze(1)
            k = self.cross_k(candidate_encs)
            v = self.cross_v(candidate_encs)
            attn = F.softmax(q @ k.transpose(-2, -1) / math.sqrt(48), dim=-1)
            attended = (attn @ v).expand(-1, N, -1)
            q_expand = query_enc.unsqueeze(1).expand(-1, N, -1)
            q_48 = self.cross_q(q_expand)
            corr_input = torch.cat([attended, q_48, extra_features], dim=-1)
            sub_scores = self.correlation_net(corr_input)
            final = self.final_score(sub_scores).squeeze(-1)
            causal_input = torch.cat([q_expand, candidate_encs], dim=-1)
            causal = self.causal_chain(causal_input).squeeze(-1)
            emo_res = self.emotional_res(emotional_pairs).squeeze(-1)
            combined = 0.5 * final + 0.25 * causal + 0.25 * emo_res
            return {
                "final_scores": combined, "sub_scores": sub_scores,
                "causal": causal, "emotional": emo_res,
            }

    class _RCM(nn.Module):
        """Response Composer -- token-efficient formatting. ~26K params."""
        def __init__(self):
            super().__init__()
            self.compress_gate = nn.Sequential(
                nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 3),
            )
            self.calibrator = nn.Sequential(
                nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 1), nn.Sigmoid(),
            )
            self.orderer = nn.Sequential(
                nn.Linear(32, 16), nn.GELU(), nn.Linear(16, 1), nn.Sigmoid(),
            )
            self.dedup = nn.Sequential(
                nn.Linear(EMBEDDING_DIM * 2, 32), nn.GELU(),
                nn.Linear(32, 1), nn.Sigmoid(),
            )

        def score_compression(self, features):
            return F.softmax(self.compress_gate(features), dim=-1)

        def calibrate_confidence(self, features):
            return self.calibrator(features).squeeze(-1)

        def predict_order(self, pair_features):
            return self.orderer(pair_features).squeeze(-1)

        def detect_duplicates(self, emb_pairs):
            return self.dedup(emb_pairs).squeeze(-1)

    class HippocampusNet(nn.Module):
        """Combined network -- all 4 modules. ~226K params."""
        def __init__(self):
            super().__init__()
            self.qum_features = _QUM_Features()
            self.qum_expander = _QUM_Expander()
            self.amm = _AMM()
            self.cdm = _CDM()
            self.rcm = _RCM()

        def param_count(self):
            return sum(p.numel() for p in self.parameters())


# ====================================================================
#  Memory Cluster Manager
# ====================================================================

@dataclass
class MemoryCluster:
    cluster_id: int
    centroid: np.ndarray
    node_ids: List[str]
    avg_valence: float = 0.0
    avg_arousal: float = 0.0
    temporal_center: float = 0.0
    temporal_spread: float = 0.0
    access_frequency: float = 0.0
    size: int = 0


# ====================================================================
#  Feature Extraction (pure Python, <0.5ms)
# ====================================================================

def extract_qum_features(query: str, context: Optional[Dict] = None) -> np.ndarray:
    """Extract 38-dim feature vector for QUM."""
    ctx = context or {}
    words = query.split()
    n_words = len(words)
    unique_words = set(w.lower() for w in words)
    chars = len(query)

    question_marks = query.count("?")
    entity_count = sum(1 for w in words if w[0:1].isupper() and len(w) > 1)
    temporal_count = sum(1 for w in words if w.lower() in _TEMPORAL_WORDS)
    emotion_count = sum(1 for w in words if w.lower() in _EMOTION_WORDS)
    specificity = min(1.0, entity_count / max(n_words, 1) * 3)
    negation_count = sum(1 for w in words if w.lower() in ("nicht", "kein", "keine", "never", "not", "no"))
    pronoun_count = sum(1 for w in words if w.lower() in ("ich", "du", "er", "sie", "wir", "i", "you", "we", "he", "she"))

    text_feats = [
        min(n_words / 30.0, 1.0),
        min(chars / 200.0, 1.0),
        min(question_marks / 3.0, 1.0),
        min(entity_count / 5.0, 1.0),
        min(temporal_count / 3.0, 1.0),
        specificity,
        1.0 - specificity,
        min(emotion_count / 3.0, 1.0),
        min(negation_count / 2.0, 1.0),
        pronoun_count / max(n_words, 1),
        len(unique_words) / max(n_words, 1),
        sum(len(w) for w in words) / max(n_words, 1) / 10.0,
    ]

    # Context features (12) -- last 3 messages
    recent_msgs = ctx.get("recent_messages", [])
    ctx_feats = []
    for i in range(3):
        if i < len(recent_msgs):
            msg = recent_msgs[i]
            msg_text = msg if isinstance(msg, str) else str(msg)
            msg_words = msg_text.split()
            overlap = len(set(w.lower() for w in msg_words) & unique_words) / max(len(unique_words), 1)
            ctx_feats.extend([
                min(len(msg_words) / 30.0, 1.0),
                1.0 if "?" in msg_text else 0.0,
                overlap,
                sum(1 for w in msg_words if w.lower() in _EMOTION_WORDS) / max(len(msg_words), 1),
            ])
        else:
            ctx_feats.extend([0.0, 0.0, 0.0, 0.0])

    # E-PQ/emotional state (4) -- optional, defaults to neutral
    epq_feats = [
        ctx.get("valence", 0.0),
        ctx.get("arousal", 0.5),
        ctx.get("autonomy", 0.5),
        ctx.get("empathy", 0.5),
    ]

    # Temporal features (4)
    now = time.localtime()
    hour = now.tm_hour + now.tm_min / 60.0
    wday = now.tm_wday
    temporal_feats = [
        math.sin(2 * math.pi * hour / 24),
        math.cos(2 * math.pi * hour / 24),
        math.sin(2 * math.pi * wday / 7),
        math.cos(2 * math.pi * wday / 7),
    ]

    # Session features (6) -- optional
    session_feats = [
        min(ctx.get("messages_in_session", 0) / 30.0, 1.0),
        min(ctx.get("avg_response_length", 0) / 500.0, 1.0),
        ctx.get("topic_stability", 0.5),
        min(ctx.get("time_since_last_query", 60) / 300.0, 1.0),
        ctx.get("retrieval_success_rate", 0.5),
        {"chat": 0.0, "idle": 0.5, "consolidation": 1.0}.get(
            ctx.get("cognitive_mode", "chat"), 0.0
        ),
    ]

    features = text_feats + ctx_feats + epq_feats + temporal_feats + session_feats
    return np.array(features, dtype=np.float32)


def _extract_node_meta(node: Dict, query_valence: float = 0.0,
                       query_arousal: float = 0.5) -> np.ndarray:
    """Extract 16-dim metadata features for a node."""
    meta = node.get("metadata", {}) if isinstance(node.get("metadata"), dict) else {}
    conf = meta.get("confidence", 0.5)
    created = node.get("created_at", "")
    age_days = 0.0
    if created:
        try:
            from datetime import datetime
            dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
            age_days = ((datetime.now(dt.tzinfo) - dt) if dt.tzinfo else (datetime.now() - dt)).total_seconds() / 86400
        except Exception:
            pass

    valence = meta.get("valence", 0.0)
    arousal = meta.get("arousal", 0.5)

    return np.array([
        conf,
        min(age_days / 30.0, 1.0),
        valence,
        arousal,
        min(meta.get("access_count", 0) / 10.0, 1.0),
        min(node.get("degree", 0) / 10.0, 1.0),
        0.0,  # cluster_id_norm
        0.0,  # graph_distance
        math.exp(-age_days / 30.0),  # temporal_proximity
        1.0 - min(abs(valence - query_valence) + abs(arousal - query_arousal), 2.0) / 2.0,
        0.0,  # causal_chain_length
        0.0,  # co_retrieval_score
        math.exp(-age_days / 7.0),   # recency
        meta.get("utility", 0.5),
        meta.get("surprise", 0.0),
        min(len(meta.get("full_text", "")) / 500.0, 1.0),
    ], dtype=np.float32)


# ====================================================================
#  Hippocampus Orchestrator
# ====================================================================

class Hippocampus:
    """Neural retrieval enhancement layer for Titan."""

    def __init__(self, data_dir: Path, titan_db_path: Path,
                 embed_fn=None):
        """
        Args:
            data_dir: Directory for hippocampus data (models, DB)
            titan_db_path: Path to the main titan.db
            embed_fn: Callable(text) -> np.ndarray for embedding text.
                     If None, uses SentenceTransformer directly.
        """
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required for Hippocampus")

        self._data_dir = data_dir
        self._titan_db_path = titan_db_path
        self._embed_fn = embed_fn
        self._model_dir = data_dir / "models"
        self._model_path = self._model_dir / "titan_hippocampus.pt"
        self._db_path = data_dir / "hippocampus.db"
        self._vectors_path = data_dir / "titan_vectors.npz"
        self._vector_ids_path = data_dir / "titan_vector_ids.json"

        self._lock = threading.RLock()
        self._net = HippocampusNet()
        self._net.eval()

        self._cdm_optimizer = optim.AdamW(self._net.cdm.parameters(), lr=_LR,
                                          weight_decay=_WEIGHT_DECAY)
        self._rcm_optimizer = optim.AdamW(self._net.rcm.parameters(), lr=_LR,
                                          weight_decay=_WEIGHT_DECAY)

        self._total_steps = 0
        self._training_steps = 0

        # Cluster cache
        self._clusters: List[MemoryCluster] = []
        self._cluster_keys: Optional[torch.Tensor] = None
        self._cluster_values: Optional[torch.Tensor] = None
        self._cluster_temporal: Optional[torch.Tensor] = None

        # Vector cache
        self._vectors: Optional[np.ndarray] = None
        self._vector_ids: Optional[List[str]] = None
        self._vector_norms: Optional[np.ndarray] = None

        # Node metadata cache
        self._node_meta_cache: Dict[str, Dict] = {}
        self._cache_age = 0.0

        # Embedding LRU cache
        self._emb_cache: Dict[str, np.ndarray] = {}
        self._emb_cache_order: List[str] = []
        self._emb_cache_max = 128

        # Fast node_id -> vector index mapping
        self._vector_id_to_idx: Dict[str, int] = {}

        # Initialize
        data_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._load_vectors()
        self._load_checkpoint()
        self._maybe_build_clusters()

        LOG.info("Hippocampus initialized: %d params, phase=%s, vectors=%d",
                 self._net.param_count(), self._get_phase_name(),
                 len(self._vector_ids) if self._vector_ids else 0)

    # -- DB --

    def _init_db(self):
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._db_path), timeout=5)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.executescript(_SCHEMA)
        for sql in _MIGRATIONS:
            try:
                conn.execute(sql)
            except sqlite3.OperationalError:
                pass
        conn.commit()
        conn.close()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), timeout=3)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    # -- Vector Cache --

    def _load_vectors(self):
        try:
            if self._vectors_path.exists() and self._vector_ids_path.exists():
                data = np.load(str(self._vectors_path))
                self._vectors = data["vectors"]
                with open(self._vector_ids_path, "r") as f:
                    self._vector_ids = json.load(f)
                if len(self._vector_ids) != len(self._vectors):
                    min_len = min(len(self._vectors), len(self._vector_ids))
                    self._vectors = self._vectors[:min_len]
                    self._vector_ids = self._vector_ids[:min_len]
                norms = np.linalg.norm(self._vectors, axis=1, keepdims=True)
                norms[norms == 0] = 1
                self._vector_norms = self._vectors / norms
                self._vector_id_to_idx = {nid: i for i, nid in enumerate(self._vector_ids)}
            else:
                self._vectors = np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
                self._vector_ids = []
                self._vector_norms = np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
                self._vector_id_to_idx = {}
        except Exception as e:
            LOG.error("Failed to load vectors: %s", e)
            self._vectors = np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
            self._vector_ids = []
            self._vector_norms = np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
            self._vector_id_to_idx = {}

    def reload_vectors(self):
        with self._lock:
            self._load_vectors()
            self._maybe_build_clusters()

    # -- ANN Search --

    def _ann_search(self, query_vec, top_k=30, _norms=None, _ids=None):
        norms = _norms if _norms is not None else self._vector_norms
        ids = _ids if _ids is not None else self._vector_ids
        if norms is None or len(norms) == 0:
            return []
        q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
        sims = norms @ q_norm
        k = min(top_k, len(sims))
        top_idx = np.argpartition(sims, -k)[-k:]
        top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]
        return [(ids[i], float(sims[i])) for i in top_idx]

    # -- Embedding Cache --

    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        key = hashlib.md5(text.encode()).hexdigest()
        if key in self._emb_cache:
            return self._emb_cache[key]
        try:
            if self._embed_fn is not None:
                emb = self._embed_fn(text)
            else:
                from sentence_transformers import SentenceTransformer
                if not hasattr(self, '_st_model'):
                    self._st_model = SentenceTransformer("all-MiniLM-L6-v2")
                emb = self._st_model.encode(text, convert_to_numpy=True)
            if emb is not None:
                if len(self._emb_cache) >= self._emb_cache_max:
                    oldest = self._emb_cache_order.pop(0)
                    self._emb_cache.pop(oldest, None)
                self._emb_cache[key] = emb
                self._emb_cache_order.append(key)
                return emb
        except Exception:
            pass
        return None

    # -- Cluster Management --

    def _maybe_build_clusters(self):
        if self._vectors is None or len(self._vectors) < 4:
            self._clusters = []
            return
        n = len(self._vectors)
        k = min(MAX_CLUSTERS, max(2, n // 5))
        try:
            centroids, labels = self._simple_kmeans(self._vectors, k, max_iter=20)
            clusters = []
            for ci in range(k):
                mask = labels == ci
                if not mask.any():
                    continue
                ids = [self._vector_ids[i] for i in range(n) if mask[i]]
                clusters.append(MemoryCluster(
                    cluster_id=ci, centroid=centroids[ci],
                    node_ids=ids, size=len(ids),
                ))
            self._clusters = clusters
            self._precompute_cluster_projections()
        except Exception as e:
            LOG.warning("Cluster build failed: %s", e)
            self._clusters = []

    @staticmethod
    def _simple_kmeans(data, k, max_iter=20, chunk_size=2000):
        n = len(data)
        idx = np.random.choice(n, size=min(k, n), replace=False)
        centroids = data[idx].copy()
        labels = np.zeros(n, dtype=np.int32)
        for _ in range(max_iter):
            if n <= chunk_size:
                dists = np.linalg.norm(data[:, None, :] - centroids[None, :, :], axis=2)
                new_labels = dists.argmin(axis=1)
            else:
                new_labels = np.zeros(n, dtype=np.int32)
                for start in range(0, n, chunk_size):
                    end = min(start + chunk_size, n)
                    chunk_dists = np.linalg.norm(
                        data[start:end, None, :] - centroids[None, :, :], axis=2
                    )
                    new_labels[start:end] = chunk_dists.argmin(axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for ci in range(k):
                mask = labels == ci
                if mask.any():
                    centroids[ci] = data[mask].mean(axis=0)
        return centroids, labels

    @torch.no_grad()
    def _precompute_cluster_projections(self):
        if not self._clusters:
            self._cluster_keys = None
            self._cluster_values = None
            self._cluster_temporal = None
            return

        cluster_inputs = []
        temporal_features = []
        for c in self._clusters:
            meta = np.zeros(CLUSTER_META_DIM, dtype=np.float32)
            meta[0] = 0.5
            meta[1] = c.temporal_center / 30.0 if c.temporal_center else 0.0
            meta[2] = c.avg_valence
            meta[3] = c.avg_arousal
            meta[4] = min(c.access_frequency / 10.0, 1.0)
            meta[5] = min(c.size / 20.0, 1.0)
            combined = np.concatenate([c.centroid, meta])
            cluster_inputs.append(combined)
            temporal_features.append([
                meta[1], c.temporal_spread / 30.0 if c.temporal_spread else 0.0,
                min(c.access_frequency / 10.0, 1.0),
                1.0 if c.temporal_center and c.temporal_center < 3 else 0.0,
            ])

        t_inputs = torch.tensor(np.array(cluster_inputs), dtype=torch.float32)
        keys, values = self._net.amm.project_clusters(t_inputs)
        self._cluster_keys = keys.detach()
        self._cluster_values = values.detach()
        self._cluster_temporal = torch.tensor(
            np.array(temporal_features), dtype=torch.float32
        )

    # -- Cold Start --

    def _get_phase_name(self):
        for i, (threshold, _) in enumerate(_COLD_START_PHASES):
            if self._total_steps < threshold:
                return ["bootstrap", "blending", "maturing"][i]
        return "mature"

    def _get_fallback_weight(self):
        for threshold, weight in _COLD_START_PHASES:
            if self._total_steps < threshold:
                prev_threshold = 0
                for pt, _ in _COLD_START_PHASES:
                    if pt == threshold:
                        break
                    prev_threshold = pt
                progress = (self._total_steps - prev_threshold) / max(threshold - prev_threshold, 1)
                rate = progress * progress * (3.0 - 2.0 * progress)
                idx = _COLD_START_PHASES.index((threshold, weight))
                prev_weight = _COLD_START_PHASES[max(0, idx - 1)][1] \
                    if idx > 0 else 1.0
                return prev_weight + rate * (weight - prev_weight)
        return _MATURE_FALLBACK

    @property
    def is_mature(self):
        return self._total_steps >= _COLD_START_PHASES[-1][0]

    def is_ready(self):
        return self._vectors is not None and len(self._vector_ids) > 0

    # ====================================================================
    #  MAIN RETRIEVAL -- The Hot Path (<20ms target)
    # ====================================================================

    @torch.no_grad()
    def retrieve(self, query: str, limit: int = 10,
                 context: Optional[Dict] = None) -> Optional[List[Dict]]:
        t0 = time.monotonic()
        ctx = context or {}

        if not self.is_ready():
            return None

        with self._lock:
            vectors = self._vectors
            vector_norms = self._vector_norms
            vid_to_idx = self._vector_id_to_idx
            clusters = self._clusters
            cluster_keys = self._cluster_keys
            cluster_values = self._cluster_values
            cluster_temporal = self._cluster_temporal

        if not self._node_meta_cache:
            self.refresh_node_cache()
        elif (time.monotonic() - self._cache_age > 60):
            threading.Thread(
                target=self.refresh_node_cache, daemon=True, name="hippo-cache"
            ).start()

        # Step 1: QUM
        qum_feats = extract_qum_features(query, ctx)
        qum_tensor = torch.tensor(qum_feats, dtype=torch.float32).unsqueeze(0)
        qum_out = self._net.qum_features(qum_tensor)
        intent = qum_out["intent"]
        temporal = qum_out["temporal"]
        emotional = qum_out["emotional"]

        # Step 2: Embedding + ANN
        query_emb = self._get_embedding(query)
        if query_emb is None:
            return None

        emb_tensor = torch.tensor(query_emb, dtype=torch.float32).unsqueeze(0)
        expanded_emb = self._net.qum_expander(emb_tensor, intent)
        expanded_np = expanded_emb.squeeze(0).numpy()

        ann_results = self._ann_search(expanded_np, top_k=30,
                                        _norms=vector_norms,
                                        _ids=self._vector_ids)
        if not ann_results:
            return None

        # Step 3: AMM Hopfield
        candidate_ids = set()
        candidate_scores: Dict[str, float] = {}

        for nid, score in ann_results:
            candidate_ids.add(nid)
            candidate_scores[nid] = score

        if cluster_keys is not None and len(clusters) > 0:
            query_meta = np.zeros(CLUSTER_META_DIM, dtype=np.float32)
            query_meta[2] = ctx.get("valence", 0.0)
            query_meta[3] = ctx.get("arousal", 0.5)
            query_input = np.concatenate([expanded_np, query_meta])
            query_input_t = torch.tensor(query_input, dtype=torch.float32).unsqueeze(0)
            query_proj = self._net.amm.project_query(query_input_t)

            amm_out = self._net.amm(query_proj, cluster_keys, cluster_values, cluster_temporal)
            attn = amm_out["attn_weights"].squeeze(0).numpy()

            top_cluster_idx = np.argsort(attn)[-3:]
            for ci in top_cluster_idx:
                if ci < len(clusters) and attn[ci] > 0.05:
                    for nid in clusters[ci].node_ids[:10]:
                        if nid not in candidate_ids:
                            candidate_ids.add(nid)
                            candidate_scores[nid] = attn[ci] * 0.5

            if len(candidate_ids) > 1 and query_proj is not None:
                for nid in list(candidate_ids)[:20]:
                    c_idx = vid_to_idx.get(nid)
                    if c_idx is None:
                        continue
                    c_emb = vectors[c_idx]
                    c_meta = np.zeros(CLUSTER_META_DIM, dtype=np.float32)
                    c_input = torch.tensor(
                        np.concatenate([c_emb, c_meta]), dtype=torch.float32
                    ).unsqueeze(0)
                    c_proj = self._net.amm.query_head(
                        self._net.amm.shared_proj(c_input)
                    )
                    coact = self._net.amm.predict_coactivation(query_proj, c_proj).item()
                    if coact > 0.6:
                        candidate_scores[nid] = candidate_scores.get(nid, 0) * (1.0 + 0.3 * coact)

        if len(candidate_ids) > MAX_CANDIDATES:
            sorted_cands = sorted(candidate_scores.items(), key=lambda x: -x[1])
            candidate_ids = set(nid for nid, _ in sorted_cands[:MAX_CANDIDATES])

        # Step 4: CDM Re-Ranking
        cand_list = list(candidate_ids)
        if not cand_list:
            return None

        cand_embs, cand_metas, cand_extra, cand_emo = [], [], [], []
        q_val = emotional[0, 0].item()
        q_aro = emotional[0, 1].item()

        for nid in cand_list:
            c_idx = vid_to_idx.get(nid)
            emb = vectors[c_idx] if c_idx is not None else np.zeros(EMBEDDING_DIM, dtype=np.float32)

            node = self._node_meta_cache.get(nid, {})
            meta = _extract_node_meta(node, q_val, q_aro)

            cand_embs.append(np.concatenate([emb, meta]))
            cand_metas.append(meta)

            ann_score = candidate_scores.get(nid, 0.0)
            cand_extra.append([
                meta[8], meta[9], 0.0, meta[11], meta[13], meta[14],
                ann_score, meta[12],
            ])
            cand_emo.append([
                q_val, q_aro, meta[2], meta[3],
                abs(q_val - meta[2]), abs(q_aro - meta[3]),
                meta[9], abs(meta[2]) + abs(meta[3] - 0.5),
            ])

        N = len(cand_list)
        cand_input = torch.tensor(np.array(cand_embs), dtype=torch.float32)
        cand_enc = self._net.cdm.candidate_enc(cand_input)

        query_meta = np.zeros(CLUSTER_META_DIM, dtype=np.float32)
        query_meta[2] = q_val
        query_meta[3] = q_aro
        query_full = np.concatenate([expanded_np, query_meta])
        query_enc = self._net.cdm.candidate_enc(
            torch.tensor(query_full, dtype=torch.float32).unsqueeze(0)
        )

        extra_t = torch.tensor(np.array(cand_extra), dtype=torch.float32).unsqueeze(0)
        emo_t = torch.tensor(np.array(cand_emo), dtype=torch.float32).unsqueeze(0)

        cdm_out = self._net.cdm(query_enc, cand_enc.unsqueeze(0), extra_t, emo_t)
        final_scores = cdm_out["final_scores"].squeeze(0).numpy()

        recency_w = temporal[0, 0].item()
        for i, nid in enumerate(cand_list):
            recency = cand_metas[i][12]
            final_scores[i] = final_scores[i] * (0.7 + 0.3 * recency * recency_w)

        # Step 5: RCM Response Composition
        for i, nid in enumerate(cand_list):
            node = self._node_meta_cache.get(nid, {})
            ntype = node.get("type", "")
            meta = node.get("metadata", {}) if isinstance(node.get("metadata"), dict) else {}
            has_content = bool(meta.get("full_text") or node.get("label", ""))
            if ntype == "entity" and not meta.get("full_text"):
                final_scores[i] *= 0.1
            elif not has_content:
                final_scores[i] *= 0.05

        top_k = min(limit, N)
        top_idx = np.argsort(final_scores)[-top_k:][::-1]

        # Dedup
        dedup_scores = {}
        if len(top_idx) > 1:
            for ri in range(len(top_idx)):
                for rj in range(ri + 1, len(top_idx)):
                    ii, jj = top_idx[ri], top_idx[rj]
                    idx_a = vid_to_idx.get(cand_list[ii])
                    idx_b = vid_to_idx.get(cand_list[jj])
                    if idx_a is not None and idx_b is not None:
                        pair = torch.tensor(
                            np.concatenate([vectors[idx_a], vectors[idx_b]]),
                            dtype=torch.float32,
                        ).unsqueeze(0)
                        dup_prob = self._net.rcm.detect_duplicates(pair).item()
                        if dup_prob > 0.7:
                            dedup_scores[jj] = max(dedup_scores.get(jj, 0), dup_prob)

        results = []
        for rank, i in enumerate(top_idx):
            if i in dedup_scores and rank >= 3:
                continue
            nid = cand_list[i]
            score = float(final_scores[i])
            node = self._node_meta_cache.get(nid, {})
            meta = node.get("metadata", {}) if isinstance(node.get("metadata"), dict) else {}
            content = meta.get("full_text", "") or node.get("label", "") or nid

            if content.startswith("[") and "]" in content:
                content = content[content.index("]") + 1:].strip()
            if content.startswith("Question:") and "Answer:" in content:
                content = content[content.index("Answer:") + 7:].strip()

            cal_feats = torch.tensor([[
                meta.get("confidence", 0.5), score,
                cand_metas[i][12], cand_metas[i][5], cand_metas[i][4],
                0.5, abs(cand_metas[i][2]),
                1.0 - self._get_fallback_weight(),
            ]], dtype=torch.float32)
            cal_conf = self._net.rcm.calibrate_confidence(cal_feats).item()

            comp_feats = torch.tensor([[
                score, cal_conf, cand_metas[i][12], abs(cand_metas[i][2]),
                0.0, rank / max(top_k, 1), 0.0, float(qum_feats[5]),
                0.0, 1.0 if node.get("type") == "claim" else 0.0,
                cand_metas[i][15], cand_metas[i][14],
                cdm_out["causal"].squeeze(0)[i].item(),
                cdm_out["emotional"].squeeze(0)[i].item(),
                dedup_scores.get(i, 0.0),
                1.0 - rank / max(top_k, 1),
            ]], dtype=torch.float32)
            comp_probs = self._net.rcm.score_compression(comp_feats).squeeze(0)
            comp_decision = comp_probs.argmax().item()
            if comp_decision == 2 and rank < 3:
                comp_decision = 1

            if comp_decision == 0:
                max_len = 150
            elif comp_decision == 1:
                max_len = 80
            else:
                continue

            if len(content) > max_len:
                content = content[:max_len].rsplit(" ", 1)[0] + "..."

            if cal_conf >= 0.8:
                certainty = "certain"
            elif cal_conf >= 0.6:
                certainty = "likely"
            elif cal_conf >= 0.4:
                certainty = "possible"
            else:
                certainty = "faint"

            age_days = cand_metas[i][1] * 30
            if age_days < 1:
                age_str = "today"
            elif age_days < 7:
                age_str = f"{int(age_days)}d"
            elif age_days < 30:
                age_str = f"{int(age_days/7)}w"
            else:
                age_str = f"{int(age_days/30)}m"

            results.append({
                "node_id": nid, "label": node.get("label", nid),
                "type": node.get("type", "memory"), "content": content,
                "confidence": cal_conf, "score": score,
                "certainty": certainty, "age": age_str,
                "phase": self._get_phase_name(),
            })

        elapsed_ms = (time.monotonic() - t0) * 1000
        self._total_steps += 1
        self._log_retrieval(query, results, elapsed_ms)

        return results if results else None

    def build_context_string(self, items: List[Dict], max_length: int = 2000) -> str:
        if not items:
            return ""
        lines = []
        total = 0
        for item in items:
            line = f"({item.get('certainty', 'likely')}) {item['content']} [{item.get('age', '?')}]"
            if total + len(line) > max_length:
                break
            lines.append(line)
            total += len(line)
        if not lines:
            return ""
        return "[MEMORY]\n" + "\n".join(lines) + "\n[/MEMORY]"

    # -- Logging --

    def _log_retrieval(self, query, results, latency_ms):
        try:
            qhash = hashlib.md5(query.encode()).hexdigest()[:12]
            node_ids = [r["node_id"] for r in results]
            scores = [r["score"] for r in results]
            conn = self._get_conn()
            conn.execute(
                "INSERT INTO retrieval_log (query_hash, query_text, result_node_ids, result_scores, phase, latency_ms) VALUES (?,?,?,?,?,?)",
                (qhash, query[:500], json.dumps(node_ids), json.dumps(scores),
                 self._get_phase_name(), latency_ms)
            )
            conn.commit()
            conn.close()
        except Exception:
            pass

    def record_outcome(self, query, node_id, was_useful, reward=0.0):
        try:
            qhash = hashlib.md5(query.encode()).hexdigest()[:12]
            conn = self._get_conn()
            conn.execute(
                "INSERT INTO outcome_log (query_hash, node_id, was_useful, reward) VALUES (?,?,?,?)",
                (qhash, node_id, 1 if was_useful else 0, reward)
            )
            conn.commit()
            conn.close()
        except Exception:
            pass

    # -- Node Metadata Cache --

    def refresh_node_cache(self):
        try:
            conn = sqlite3.connect(str(self._titan_db_path), timeout=3)
            rows = conn.execute(
                "SELECT id, type, label, created_at, protected, metadata FROM nodes"
            ).fetchall()
            cache = {}
            for row in rows:
                nid, ntype, label, created, protected, meta_str = row
                meta = json.loads(meta_str) if meta_str else {}
                cache[nid] = {
                    "id": nid, "type": ntype, "label": label,
                    "created_at": created, "protected": protected,
                    "metadata": meta, "degree": 0,
                }
            # Add degree info
            try:
                for src, dst in conn.execute("SELECT src, dst FROM edges"):
                    if src in cache:
                        cache[src]["degree"] = cache[src].get("degree", 0) + 1
                    if dst in cache:
                        cache[dst]["degree"] = cache[dst].get("degree", 0) + 1
            except Exception:
                pass
            conn.close()

            self._node_meta_cache = cache
            self._cache_age = time.monotonic()
        except Exception as e:
            LOG.debug("Node cache refresh failed: %s", e)

    # ====================================================================
    #  Training (called during consolidation)
    # ====================================================================

    def train_cycle(self) -> Dict:
        t0 = time.monotonic()
        stats = {}

        with self._lock:
            self.reload_vectors()
            self.refresh_node_cache()
            self._maybe_build_clusters()
            stats["clusters"] = len(self._clusters)

            stats["cdm"] = self._train_cdm()
            stats["rcm"] = self._train_rcm()

            self._save_checkpoint()

        stats["elapsed_ms"] = (time.monotonic() - t0) * 1000
        stats["phase"] = self._get_phase_name()
        stats["total_steps"] = self._total_steps
        stats["params"] = self._net.param_count()
        LOG.info("Hippocampus train_cycle: %s", stats)
        return stats

    def _train_cdm(self) -> Dict:
        try:
            conn = self._get_conn()
            rows = conn.execute(
                "SELECT o.query_hash, o.node_id, o.was_useful, o.reward, r.query_text "
                "FROM outcome_log o "
                "LEFT JOIN retrieval_log r ON o.query_hash = r.query_hash "
                "ORDER BY o.timestamp DESC LIMIT 200"
            ).fetchall()
            conn.close()

            if len(rows) < _BUFFER_MIN_TRAIN:
                return {"skipped": True, "reason": f"only {len(rows)} outcomes"}

            self._net.train()
            optimizer = self._cdm_optimizer
            total_loss = 0.0
            n_batches = 0
            _query_enc_cache: Dict[str, torch.Tensor] = {}

            for qhash, node_id, was_useful, reward, query_text in rows:
                idx = self._vector_id_to_idx.get(node_id)
                if idx is None:
                    continue
                emb = self._vectors[idx]
                meta = _extract_node_meta(
                    self._node_meta_cache.get(node_id, {}), 0.0, 0.5
                )
                cand_input = torch.tensor(
                    np.concatenate([emb, meta]), dtype=torch.float32
                ).unsqueeze(0)
                cand_enc = self._net.cdm.candidate_enc(cand_input)

                if qhash in _query_enc_cache:
                    query_enc = _query_enc_cache[qhash]
                elif query_text:
                    q_emb = self._get_embedding(query_text)
                    if q_emb is not None:
                        q_meta = np.zeros(CLUSTER_META_DIM, dtype=np.float32)
                        q_full = np.concatenate([q_emb, q_meta])
                        query_enc = self._net.cdm.candidate_enc(
                            torch.tensor(q_full, dtype=torch.float32).unsqueeze(0)
                        ).detach()
                        _query_enc_cache[qhash] = query_enc
                    else:
                        query_enc = torch.randn(1, 64) * 0.1
                else:
                    query_enc = torch.randn(1, 64) * 0.1

                extra = torch.zeros(1, 1, 8)
                emo = torch.zeros(1, 1, 8)

                out = self._net.cdm(query_enc, cand_enc.unsqueeze(0), extra, emo)
                pred = out["final_scores"].squeeze()
                target = torch.tensor(float(was_useful), dtype=torch.float32)
                loss = F.binary_cross_entropy(pred, target)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._net.cdm.parameters(), _GRAD_CLIP)
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            self._net.eval()
            self._training_steps += n_batches
            return {
                "batches": n_batches,
                "avg_loss": total_loss / max(n_batches, 1),
                "improved": n_batches > 0,
            }
        except Exception as e:
            self._net.eval()
            return {"error": str(e)}

    def _train_rcm(self) -> Dict:
        try:
            if self._vectors is None or len(self._vectors) < 10:
                return {"skipped": True}

            self._net.train()
            optimizer = self._rcm_optimizer
            total_loss = 0.0
            n = 0

            indices = np.random.choice(len(self._vectors), size=min(50, len(self._vectors)), replace=False)
            for i in range(len(indices)):
                for j in range(i + 1, min(i + 5, len(indices))):
                    a, b = indices[i], indices[j]
                    sim = float(np.dot(self._vectors[a], self._vectors[b]) /
                                (np.linalg.norm(self._vectors[a]) * np.linalg.norm(self._vectors[b]) + 1e-8))
                    is_dup = 1.0 if sim > 0.85 else 0.0

                    pair = torch.tensor(
                        np.concatenate([self._vectors[a], self._vectors[b]]),
                        dtype=torch.float32,
                    ).unsqueeze(0)
                    pred = self._net.rcm.detect_duplicates(pair)
                    target = torch.tensor([is_dup])
                    loss = F.binary_cross_entropy(pred, target)

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self._net.rcm.parameters(), _GRAD_CLIP)
                    optimizer.step()

                    total_loss += loss.item()
                    n += 1

            self._net.eval()
            return {"batches": n, "avg_loss": total_loss / max(n, 1)}
        except Exception as e:
            self._net.eval()
            return {"error": str(e)}

    # ====================================================================
    #  Checkpoint Save/Load
    # ====================================================================

    def _save_checkpoint(self):
        self._model_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "net_state": self._net.state_dict(),
            "cdm_optimizer_state": self._cdm_optimizer.state_dict(),
            "rcm_optimizer_state": self._rcm_optimizer.state_dict(),
            "total_steps": self._total_steps,
            "training_steps": self._training_steps,
        }
        tmp = self._model_path.with_suffix(".tmp")
        torch.save(checkpoint, str(tmp))
        tmp.rename(self._model_path)

    def _load_checkpoint(self):
        if not self._model_path.exists():
            return
        try:
            ckpt = torch.load(str(self._model_path), weights_only=False, map_location="cpu")
            self._net.load_state_dict(ckpt["net_state"])
            if "cdm_optimizer_state" in ckpt:
                self._cdm_optimizer.load_state_dict(ckpt["cdm_optimizer_state"])
            if "rcm_optimizer_state" in ckpt:
                self._rcm_optimizer.load_state_dict(ckpt["rcm_optimizer_state"])
            self._total_steps = ckpt.get("total_steps", 0)
            self._training_steps = ckpt.get("training_steps", 0)
            LOG.info("Hippocampus checkpoint loaded: %d steps", self._total_steps)
        except Exception as e:
            LOG.warning("Failed to load hippocampus checkpoint: %s", e)

    # -- Stats --

    def get_stats(self) -> Dict:
        return {
            "params": self._net.param_count(),
            "total_steps": self._total_steps,
            "training_steps": self._training_steps,
            "phase": self._get_phase_name(),
            "fallback_weight": round(self._get_fallback_weight(), 3),
            "clusters": len(self._clusters),
            "vectors": len(self._vector_ids) if self._vector_ids else 0,
            "emb_cache_size": len(self._emb_cache),
            "node_cache_size": len(self._node_meta_cache),
        }


# ====================================================================
#  Singleton
# ====================================================================
_instance: Optional[Hippocampus] = None
_instance_lock = threading.Lock()


def get_hippocampus() -> Optional[Hippocampus]:
    """Returns the singleton or None if not initialized."""
    return _instance


def init_hippocampus(data_dir: Path, titan_db_path: Path,
                     embed_fn=None) -> Hippocampus:
    """Initialize the hippocampus singleton."""
    global _instance
    with _instance_lock:
        if _instance is None:
            _instance = Hippocampus(data_dir, titan_db_path, embed_fn)
    return _instance


def reset_hippocampus():
    """Reset the singleton (for testing)."""
    global _instance
    _instance = None
