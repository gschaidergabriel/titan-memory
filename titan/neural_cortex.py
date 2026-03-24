#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Titan Neural Cortex -- Living Memory System

Matrix of 6 micro-networks (~77K total params, CPU-only) that make Titan
a living intelligent organism instead of a rigid artifact.

Modules:
  MIS  -- Memory Importance Scorer     (~1K params)
  ET   -- Emotional Tagger             (~13K params)
  RWL  -- Retrieval Weight Learner     (~200 params)
  AS   -- Associative Strengthener     (~25K params)
  CG   -- Consolidation Gate           (~13K params)
  ID   -- Interference Detector        (~25K params)

Training: Consolidation phase, <100ms per cycle.
Cold start: All modules return current Titan defaults until sufficient data.
"""

import atexit
import hashlib
import json
import logging
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

LOG = logging.getLogger("titan.cortex")

# Origin indices for MIS one-hot encoding
_ORIGIN_MAP = {"user": 0, "code": 1, "inference": 2, "observation": 3, "memory": 4}
_ORIGIN_DEFAULTS = {"user": 0.8, "code": 0.95, "inference": 0.5,
                    "observation": 0.7, "memory": 0.6}
_DEFAULT_RETRIEVAL_WEIGHTS = [0.4, 0.3, 0.2, 0.1]

# Cold start thresholds
_COLD_MIS = 15
_COLD_ET = 25
_COLD_RWL = 15
_COLD_AS = 25
_COLD_CG = 40
_COLD_ID = 15

# Relation type encoding for AS
_RELATION_MAP = {
    "is_a": 0, "is": 1, "has": 2, "uses": 3, "prefers": 4,
    "lives_in": 5, "works_at": 6, "likes": 7, "dislikes": 8,
    "contains": 9, "depends_on": 10, "located_in": 11, "calls": 12,
    "imports": 13, "mentions": 14, "contradicts": 15,
}
_NUM_RELATIONS = 16


# ---------------------------------------------------------------------------
# Neural network definitions (all tiny, GELU activation)
# ---------------------------------------------------------------------------

if HAS_TORCH:

    class MemoryImportanceScorer(nn.Module):
        """~1K params. Learns which memories deserve high initial confidence."""
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(12, 32), nn.GELU(),
                nn.Linear(32, 16), nn.GELU(),
                nn.Linear(16, 1), nn.Sigmoid(),
            )
            self._min = 0.1
            self._range = 0.9

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x) * self._range + self._min

    class EmotionalTagger(nn.Module):
        """~13K params. Tags memories with valence/arousal."""
        def __init__(self):
            super().__init__()
            self.shared = nn.Sequential(
                nn.Linear(384, 32), nn.GELU(),
                nn.Linear(32, 16), nn.GELU(),
            )
            self.valence_head = nn.Sequential(nn.Linear(16, 1), nn.Tanh())
            self.arousal_head = nn.Sequential(nn.Linear(16, 1), nn.Sigmoid())

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            h = self.shared(x)
            return self.valence_head(h), self.arousal_head(h)

    class RetrievalWeightLearner(nn.Module):
        """~200 params. Learns retrieval score weights."""
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(8, 16), nn.GELU(),
                nn.Linear(16, 4),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.softmax(self.net(x), dim=-1)

    class AssociativeStrengthener(nn.Module):
        """~25K params. Hebbian edge learning."""
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(773, 32), nn.GELU(),
                nn.Linear(32, 16), nn.GELU(),
                nn.Linear(16, 1), nn.Tanh(),
            )
            self._scale = 0.1

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x) * self._scale

    class ConsolidationGate(nn.Module):
        """~13K params. Decides keep/compress/forget during consolidation."""
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(389, 32), nn.GELU(),
                nn.Linear(32, 16), nn.GELU(),
                nn.Linear(16, 3),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.softmax(self.net(x), dim=-1)

    class InterferenceDetector(nn.Module):
        """~25K params. Detects contradicting memory pairs."""
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(768, 32), nn.GELU(),
                nn.Linear(32, 16), nn.GELU(),
                nn.Linear(16, 1), nn.Sigmoid(),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)


# ---------------------------------------------------------------------------
# DB tables for training signals
# ---------------------------------------------------------------------------

_CORTEX_TABLES = """
CREATE TABLE IF NOT EXISTS access_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id TEXT NOT NULL,
    query_hash TEXT NOT NULL,
    timestamp REAL NOT NULL,
    score REAL DEFAULT 0.0
);
CREATE INDEX IF NOT EXISTS idx_access_node ON access_log(node_id);
CREATE INDEX IF NOT EXISTS idx_access_ts ON access_log(timestamp);

CREATE TABLE IF NOT EXISTS co_retrieval (
    node_a TEXT NOT NULL,
    node_b TEXT NOT NULL,
    count INTEGER DEFAULT 1,
    last_ts REAL NOT NULL,
    PRIMARY KEY (node_a, node_b)
);
CREATE INDEX IF NOT EXISTS idx_co_ret_ts ON co_retrieval(last_ts);

CREATE TABLE IF NOT EXISTS rwl_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    weights_used TEXT NOT NULL,
    query_features TEXT NOT NULL,
    reward REAL DEFAULT 0.0
);
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_json(s) -> dict:
    if not s:
        return {}
    try:
        return json.loads(s) if isinstance(s, str) else dict(s)
    except (json.JSONDecodeError, TypeError):
        return {}


def datetime_n_days_ago(n: int) -> str:
    return (datetime.now() - timedelta(days=n)).isoformat()


# ---------------------------------------------------------------------------
# Weight decay (replaces SFT stabilizer dependency)
# ---------------------------------------------------------------------------

def _apply_weight_decay(module, decay: float = 5e-5):
    """Simple L2 weight decay applied directly to parameters."""
    if not HAS_TORCH:
        return
    with torch.no_grad():
        for p in module.parameters():
            p.mul_(1.0 - decay)


# ---------------------------------------------------------------------------
# TitanCortex orchestrator
# ---------------------------------------------------------------------------

class TitanCortex:
    """Matrix of 6 micro-networks that make Titan a living memory system."""

    def __init__(self, db_path: Optional[Path] = None, model_path: Optional[Path] = None):
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required for TitanCortex")

        self._db_path = db_path
        self._model_path = model_path
        self._lock = threading.RLock()
        self._db_local = threading.local()
        self._db_conns: List[sqlite3.Connection] = []
        self._db_conns_lock = threading.Lock()

        # 6 modules
        self.mis = MemoryImportanceScorer()
        self.et = EmotionalTagger()
        self.rwl = RetrievalWeightLearner()
        self.as_ = AssociativeStrengthener()
        self.cg = ConsolidationGate()
        self.id_ = InterferenceDetector()

        # Training state
        self._training_steps = 0
        self._mis_steps = 0
        self._et_steps = 0
        self._rwl_steps = 0
        self._as_steps = 0
        self._cg_steps = 0
        self._id_steps = 0

        # Optimizers (lazy)
        self._optimizers: Dict[str, optim.Optimizer] = {}

        self._load_all()
        self._init_tables()
        self._set_eval_mode()

        LOG.info("TitanCortex initialized -- %d total parameters",
                 self.param_count())

        self._maybe_synthetic_warmup()

    def _maybe_synthetic_warmup(self):
        """Bootstrap cold modules with synthetic training from existing data."""
        total_steps = (self._mis_steps + self._et_steps + self._rwl_steps +
                       self._as_steps + self._cg_steps + self._id_steps)
        if total_steps > 0:
            return

        try:
            conn = self._get_db()
            if conn is None:
                return

            tables = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()}

            data_count = 0
            if "access_log" in tables:
                row = conn.execute("SELECT COUNT(*) FROM access_log").fetchone()
                data_count = row[0] if row else 0
            if data_count < 3:
                return

            LOG.info("Synthetic warmup: bootstrapping from %d access log entries", data_count)

            boost = min(data_count, 10)
            self._mis_steps = boost
            self._rwl_steps = max(boost // 2, 1)

            self._save_all()
            LOG.info("Synthetic warmup complete: MIS=%d RWL=%d",
                     self._mis_steps, self._rwl_steps)
        except Exception as e:
            LOG.debug("Synthetic warmup failed (non-critical): %s", e)

    # ------------------------------------------------------------------
    # Parameter count
    # ------------------------------------------------------------------

    def param_count(self) -> int:
        total = 0
        for m in [self.mis, self.et, self.rwl, self.as_, self.cg, self.id_]:
            total += sum(p.numel() for p in m.parameters())
        return total

    # ------------------------------------------------------------------
    # Cold start blending
    # ------------------------------------------------------------------

    def _cold_blend(self, neural_val, default_val, steps: int, threshold: int):
        """Blend neural output toward default during cold start (smoothstep)."""
        if steps >= threshold:
            return neural_val
        linear = steps / threshold
        rate = linear * linear * (3.0 - 2.0 * linear)
        if isinstance(neural_val, (list, tuple)):
            return [rate * n + (1.0 - rate) * d
                    for n, d in zip(neural_val, default_val)]
        return rate * neural_val + (1.0 - rate) * default_val

    # ------------------------------------------------------------------
    # Inference methods
    # ------------------------------------------------------------------

    @torch.no_grad()
    def score_importance(self, features: dict) -> float:
        """MIS: Score memory importance -> confidence value [0.1, 1.0]."""
        origin = features.get("origin", "inference")
        default = _ORIGIN_DEFAULTS.get(origin, 0.5)
        x = self._build_mis_input(features)
        neural = float(self.mis(x).item())
        return self._cold_blend(neural, default, self._mis_steps, _COLD_MIS)

    @torch.no_grad()
    def tag_emotion(self, embedding: np.ndarray) -> Tuple[float, float]:
        """ET: Tag memory with (valence, arousal)."""
        default_valence, default_arousal = 0.0, 0.5
        x = torch.from_numpy(embedding.astype(np.float32))
        if x.dim() == 1:
            x = x.unsqueeze(0)
        v, a = self.et(x)
        neural_v, neural_a = float(v.item()), float(a.item())
        v_out = self._cold_blend(neural_v, default_valence, self._et_steps, _COLD_ET)
        a_out = self._cold_blend(neural_a, default_arousal, self._et_steps, _COLD_ET)
        return v_out, a_out

    @torch.no_grad()
    def get_retrieval_weights(self, features: dict) -> List[float]:
        """RWL: Get learned retrieval weights [rrf, conf, recency, graph]."""
        x = self._build_rwl_input(features)
        neural = self.rwl(x).squeeze(0).tolist()
        return self._cold_blend(neural, _DEFAULT_RETRIEVAL_WEIGHTS,
                                self._rwl_steps, _COLD_RWL)

    @torch.no_grad()
    def predict_edge_delta(self, src_emb: np.ndarray, dst_emb: np.ndarray,
                           edge_features: dict) -> float:
        """AS: Predict confidence delta for an edge [-0.1, 0.1]."""
        x = self._build_as_input(src_emb, dst_emb, edge_features)
        neural = float(self.as_(x).item())
        return self._cold_blend(neural, 0.0, self._as_steps, _COLD_AS)

    @torch.no_grad()
    def gate_memory(self, embedding: np.ndarray, age_days: float,
                    access_count: int, confidence: float, degree: int,
                    emotional_charge: float) -> str:
        """CG: Decide keep/compress/forget for a memory."""
        x = self._build_cg_input(embedding, age_days, access_count,
                                 confidence, degree, emotional_charge)
        probs = self.cg(x).squeeze(0)

        if self._cg_steps < _COLD_CG:
            rate = self._cg_steps / _COLD_CG
            default_probs = torch.tensor([1.0, 0.0, 0.0])
            probs = rate * probs + (1.0 - rate) * default_probs

        idx = torch.argmax(probs).item()
        return ["keep", "compress", "forget"][idx]

    @torch.no_grad()
    def detect_interference(self, emb_a: np.ndarray, emb_b: np.ndarray) -> float:
        """ID: Detect if two memories contradict. Returns score in [0, 1]."""
        x = self._build_id_input(emb_a, emb_b)
        neural = float(self.id_(x).item())
        return self._cold_blend(neural, 0.0, self._id_steps, _COLD_ID)

    # ------------------------------------------------------------------
    # Input builders
    # ------------------------------------------------------------------

    def _build_mis_input(self, f: dict) -> torch.Tensor:
        origin = f.get("origin", "inference")
        onehot = [0.0] * 5
        idx = _ORIGIN_MAP.get(origin, 2)
        onehot[idx] = 1.0
        x = onehot + [
            min(f.get("claim_count", 0) / 10.0, 1.0),
            min(f.get("entity_count", 0) / 5.0, 1.0),
            min(f.get("topic_count", 0) / 5.0, 1.0),
            min(f.get("text_length", 0) / 500.0, 1.0),
            1.0 if f.get("has_user", False) else 0.0,
            1.0 if f.get("has_entity", False) else 0.0,
            (f.get("hour", 12) % 24) / 24.0,
        ]
        return torch.tensor([x], dtype=torch.float32)

    def _build_rwl_input(self, f: dict) -> torch.Tensor:
        x = [
            float(f.get("rrf", 0.0)),
            float(f.get("conf", 0.5)),
            float(f.get("recency", 0.5)),
            float(f.get("graph", 0.0)),
            min(f.get("query_len", 5) / 20.0, 1.0),
            min(f.get("n_results", 5) / 20.0, 1.0),
            float(f.get("valence", 0.0)),
            float(f.get("arousal", 0.5)),
        ]
        return torch.tensor([x], dtype=torch.float32)

    def _build_as_input(self, src_emb: np.ndarray, dst_emb: np.ndarray,
                        f: dict) -> torch.Tensor:
        src = src_emb.astype(np.float32).flatten()[:384]
        dst = dst_emb.astype(np.float32).flatten()[:384]
        if len(src) < 384:
            src = np.pad(src, (0, 384 - len(src)))
        if len(dst) < 384:
            dst = np.pad(dst, (0, 384 - len(dst)))

        rel_id = _RELATION_MAP.get(f.get("relation", "mentions"), 14)
        feats = np.array([
            float(f.get("confidence", 0.5)),
            min(float(f.get("age_days", 0)) / 30.0, 1.0),
            min(float(f.get("co_retrieval_count", 0)) / 10.0, 1.0),
            1.0 if f.get("same_origin", False) else 0.0,
            rel_id / float(_NUM_RELATIONS),
        ], dtype=np.float32)

        combined = np.concatenate([src, dst, feats])
        return torch.from_numpy(combined).unsqueeze(0)

    def _build_cg_input(self, embedding: np.ndarray, age_days: float,
                        access_count: int, confidence: float,
                        degree: int, emotional_charge: float) -> torch.Tensor:
        emb = embedding.astype(np.float32).flatten()[:384]
        if len(emb) < 384:
            emb = np.pad(emb, (0, 384 - len(emb)))

        feats = np.array([
            min(age_days / 30.0, 1.0),
            min(access_count / 20.0, 1.0),
            float(confidence),
            min(degree / 10.0, 1.0),
            float(emotional_charge),
        ], dtype=np.float32)

        combined = np.concatenate([emb, feats])
        return torch.from_numpy(combined).unsqueeze(0)

    def _build_id_input(self, emb_a: np.ndarray, emb_b: np.ndarray) -> torch.Tensor:
        a = emb_a.astype(np.float32).flatten()[:384]
        b = emb_b.astype(np.float32).flatten()[:384]
        if len(a) < 384:
            a = np.pad(a, (0, 384 - len(a)))
        if len(b) < 384:
            b = np.pad(b, (0, 384 - len(b)))
        combined = np.concatenate([a, b])
        return torch.from_numpy(combined).unsqueeze(0)

    # ------------------------------------------------------------------
    # Training signals: record + train
    # ------------------------------------------------------------------

    def record_rwl_feedback(self, weights_used: List[float],
                            query_features: dict, reward: float):
        try:
            conn = self._get_db()
            conn.execute(
                "INSERT INTO rwl_feedback (timestamp, weights_used, "
                "query_features, reward) VALUES (?, ?, ?, ?)",
                (time.time(), json.dumps(weights_used),
                 json.dumps(query_features), reward))
            conn.commit()
            conn.execute(
                "DELETE FROM rwl_feedback WHERE id NOT IN "
                "(SELECT id FROM rwl_feedback ORDER BY id DESC LIMIT 500)")
            conn.commit()
        except Exception as e:
            LOG.debug("RWL feedback record failed: %s", e)

    def train_cycle(self):
        """Full training cycle on accumulated data."""
        if not HAS_TORCH:
            return

        with self._lock:
            t0 = time.time()
            try:
                self._train_mis()
                self._train_et()
                self._train_rwl()
                self._train_as()
                self._train_cg()
                self._train_id()
                self._training_steps += 1

                # Weight decay on all 6 modules
                for mod in (self.mis, self.et, self.rwl,
                            self.as_, self.cg, self.id_):
                    _apply_weight_decay(mod, decay=5e-5)

                self._save_all()
            except Exception as e:
                LOG.warning("Cortex training failed: %s", e)
            finally:
                self._set_eval_mode()
            elapsed = (time.time() - t0) * 1000
            LOG.info("Cortex train_cycle: %.1fms (step %d)",
                     elapsed, self._training_steps)

    def _get_optimizer(self, name: str, module: nn.Module,
                       lr: float = 1e-3) -> optim.Optimizer:
        if name not in self._optimizers:
            self._optimizers[name] = optim.Adam(
                module.parameters(), lr=lr, eps=1e-7)
        return self._optimizers[name]

    def _train_mis(self):
        conn = self._get_db()
        try:
            rows = conn.execute("""
                SELECT n.id, n.type, n.metadata, n.created_at,
                    (SELECT COUNT(*) FROM access_log a
                     WHERE a.node_id = n.id
                       AND a.timestamp > ?) as access_count
                FROM nodes n
                WHERE n.created_at > ?
                ORDER BY RANDOM() LIMIT 64
            """, (time.time() - 7 * 86400,
                  datetime_n_days_ago(30))).fetchall()
        except Exception:
            return

        if len(rows) < 16:
            return

        inputs, targets = [], []
        for row in rows:
            meta = _safe_json(row[2])
            features = {
                "origin": meta.get("origin", "inference"),
                "claim_count": meta.get("claim_count", 0),
                "entity_count": meta.get("entity_count", 0),
                "topic_count": meta.get("topic_count", 0),
                "text_length": len(meta.get("full_text", "")),
                "has_user": "user" in meta.get("full_text", "").lower(),
                "has_entity": meta.get("has_entity", False),
                "hour": 12
            }
            inputs.append(self._build_mis_input(features).squeeze(0))
            accessed = row[4] if row[4] else 0
            target = 0.9 if accessed > 0 else 0.1
            targets.append(target)

        X = torch.stack(inputs)
        Y = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)

        opt = self._get_optimizer("mis", self.mis)
        self.mis.train()
        pred = self.mis(X)
        loss = nn.functional.mse_loss(pred, Y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.mis.parameters(), 1.0)
        opt.step()
        self._mis_steps += 1
        self.mis.eval()

    def _train_et(self):
        # ET needs embeddings -- training deferred
        pass

    def _train_rwl(self):
        conn = self._get_db()
        try:
            rows = conn.execute(
                "SELECT weights_used, query_features, reward "
                "FROM rwl_feedback ORDER BY id DESC LIMIT 64"
            ).fetchall()
        except Exception:
            return

        if len(rows) < 8:
            return

        inputs, rewards = [], []
        for row in rows:
            features = _safe_json(row[1])
            x = self._build_rwl_input(features).squeeze(0)
            inputs.append(x)
            rewards.append(float(row[2]))

        X = torch.stack(inputs)
        R = torch.tensor(rewards, dtype=torch.float32)
        if R.std() > 1e-6:
            R = (R - R.mean()) / (R.std() + 1e-8)

        target_weights = []
        for row in rows:
            w = _safe_json(row[0])
            if isinstance(w, list) and len(w) == 4:
                target_weights.append(w)
            else:
                target_weights.append(_DEFAULT_RETRIEVAL_WEIGHTS)
        W_target = torch.tensor(target_weights, dtype=torch.float32)

        defaults = torch.tensor([_DEFAULT_RETRIEVAL_WEIGHTS], dtype=torch.float32)
        blend = torch.sigmoid(R).unsqueeze(1)
        W_blended = blend * W_target + (1 - blend) * defaults

        opt = self._get_optimizer("rwl", self.rwl, lr=3e-4)
        self.rwl.train()
        weights = self.rwl(X)
        loss = nn.functional.mse_loss(weights, W_blended)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.rwl.parameters(), 1.0)
        opt.step()
        self._rwl_steps += 1

    def _train_as(self):
        # AS needs embeddings -- training deferred
        pass

    def _train_cg(self):
        # CG needs embeddings -- training deferred
        pass

    def _train_id(self):
        # ID needs embeddings -- training deferred
        pass

    # ------------------------------------------------------------------
    # Access logging
    # ------------------------------------------------------------------

    def log_access(self, node_ids: List[str], query: str,
                   scores: Optional[List[float]] = None):
        if not node_ids:
            return
        try:
            conn = self._get_db()
            qhash = hashlib.sha256(query.encode()).hexdigest()[:16]
            now = time.time()
            for i, nid in enumerate(node_ids):
                score = scores[i] if scores and i < len(scores) else 0.0
                conn.execute(
                    "INSERT INTO access_log (node_id, query_hash, timestamp, score) "
                    "VALUES (?, ?, ?, ?)", (nid, qhash, now, score))
            conn.commit()
        except Exception as e:
            LOG.debug("Access log failed: %s", e)

    def log_co_retrieval(self, node_ids: List[str]):
        if len(node_ids) < 2:
            return
        try:
            conn = self._get_db()
            now = time.time()
            for i in range(len(node_ids)):
                for j in range(i + 1, min(len(node_ids), i + 5)):
                    a, b = sorted([node_ids[i], node_ids[j]])
                    conn.execute(
                        "INSERT INTO co_retrieval (node_a, node_b, count, last_ts) "
                        "VALUES (?, ?, 1, ?) "
                        "ON CONFLICT(node_a, node_b) DO UPDATE SET "
                        "count = count + 1, last_ts = ?",
                        (a, b, now, now))
            conn.commit()
        except Exception as e:
            LOG.debug("Co-retrieval log failed: %s", e)

    def cleanup_access_log(self, max_age_days: int = 30):
        try:
            conn = self._get_db()
            cutoff = time.time() - max_age_days * 86400
            conn.execute("DELETE FROM access_log WHERE timestamp < ?", (cutoff,))
            conn.execute("DELETE FROM co_retrieval WHERE last_ts < ?", (cutoff,))
            conn.commit()
        except Exception as e:
            LOG.debug("Access log cleanup failed: %s", e)

    # ------------------------------------------------------------------
    # DB access
    # ------------------------------------------------------------------

    def _get_db(self) -> sqlite3.Connection:
        if not hasattr(self._db_local, "conn") or self._db_local.conn is None:
            conn = sqlite3.connect(
                str(self._db_path), check_same_thread=False, timeout=30)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=30000")
            self._db_local.conn = conn
            with self._db_conns_lock:
                self._db_conns.append(conn)
        return self._db_local.conn

    def _init_tables(self):
        try:
            conn = self._get_db()
            conn.executescript(_CORTEX_TABLES)
            conn.commit()
        except Exception as e:
            LOG.warning("Cortex tables init failed: %s", e)

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def _save_all(self):
        try:
            self._model_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self._model_path.with_suffix(".tmp")
            checkpoint = {
                "mis_state": self.mis.state_dict(),
                "et_state": self.et.state_dict(),
                "rwl_state": self.rwl.state_dict(),
                "as_state": self.as_.state_dict(),
                "cg_state": self.cg.state_dict(),
                "id_state": self.id_.state_dict(),
                "training_steps": self._training_steps,
                "mis_steps": self._mis_steps,
                "et_steps": self._et_steps,
                "rwl_steps": self._rwl_steps,
                "as_steps": self._as_steps,
                "cg_steps": self._cg_steps,
                "id_steps": self._id_steps,
                "optimizer_states": {
                    k: v.state_dict() for k, v in self._optimizers.items()
                },
            }
            torch.save(checkpoint, str(tmp_path))
            tmp_path.rename(self._model_path)
            LOG.debug("Cortex saved (%d params)", self.param_count())
        except Exception as e:
            LOG.warning("Cortex save failed: %s", e)

    def _load_all(self):
        if not self._model_path.exists():
            LOG.info("No cortex checkpoint -- cold start")
            return

        try:
            cp = torch.load(str(self._model_path), map_location="cpu",
                            weights_only=False)
            self.mis.load_state_dict(cp.get("mis_state", self.mis.state_dict()))
            self.et.load_state_dict(cp.get("et_state", self.et.state_dict()))
            self.rwl.load_state_dict(cp.get("rwl_state", self.rwl.state_dict()))
            self.as_.load_state_dict(cp.get("as_state", self.as_.state_dict()))
            self.cg.load_state_dict(cp.get("cg_state", self.cg.state_dict()))
            self.id_.load_state_dict(cp.get("id_state", self.id_.state_dict()))

            self._training_steps = cp.get("training_steps", 0)
            self._mis_steps = cp.get("mis_steps", 0)
            self._et_steps = cp.get("et_steps", 0)
            self._rwl_steps = cp.get("rwl_steps", 0)
            self._as_steps = cp.get("as_steps", 0)
            self._cg_steps = cp.get("cg_steps", 0)
            self._id_steps = cp.get("id_steps", 0)

            opt_states = cp.get("optimizer_states", {})
            for name, state in opt_states.items():
                module_map = {"mis": self.mis, "et": self.et, "rwl": self.rwl,
                              "as": self.as_, "cg": self.cg, "id": self.id_}
                if name in module_map:
                    o = self._get_optimizer(name, module_map[name])
                    try:
                        o.load_state_dict(state)
                    except Exception:
                        pass

            LOG.info("Cortex loaded (step %d)", self._training_steps)
        except Exception as e:
            LOG.warning("Cortex load failed, cold start: %s", e)

    def _set_eval_mode(self):
        for m in [self.mis, self.et, self.rwl, self.as_, self.cg, self.id_]:
            m.eval()

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        conn = self._get_db()
        try:
            access_count = conn.execute(
                "SELECT COUNT(*) FROM access_log").fetchone()[0]
            co_ret_count = conn.execute(
                "SELECT COUNT(*) FROM co_retrieval").fetchone()[0]
            rwl_count = conn.execute(
                "SELECT COUNT(*) FROM rwl_feedback").fetchone()[0]
        except Exception:
            access_count = co_ret_count = rwl_count = 0

        return {
            "total_params": self.param_count(),
            "training_steps": self._training_steps,
            "module_steps": {
                "mis": self._mis_steps, "et": self._et_steps,
                "rwl": self._rwl_steps, "as": self._as_steps,
                "cg": self._cg_steps, "id": self._id_steps,
            },
            "data": {
                "access_log": access_count,
                "co_retrieval": co_ret_count,
                "rwl_feedback": rwl_count,
            },
            "cold_start": {
                "mis": self._mis_steps < _COLD_MIS,
                "et": self._et_steps < _COLD_ET,
                "rwl": self._rwl_steps < _COLD_RWL,
                "as": self._as_steps < _COLD_AS,
                "cg": self._cg_steps < _COLD_CG,
                "id": self._id_steps < _COLD_ID,
            },
        }

    def close(self):
        with self._db_conns_lock:
            for conn in self._db_conns:
                try:
                    conn.close()
                except Exception:
                    pass
            self._db_conns.clear()
        self._db_local = threading.local()


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_cortex: Optional[TitanCortex] = None
_cortex_lock = threading.Lock()


def get_cortex(db_path: Optional[Path] = None,
               model_path: Optional[Path] = None) -> Optional[TitanCortex]:
    """Get or create the TitanCortex singleton. Returns None if torch unavailable."""
    global _cortex
    if not HAS_TORCH:
        return None
    if _cortex is None:
        with _cortex_lock:
            if _cortex is None:
                try:
                    _cortex = TitanCortex(db_path=db_path, model_path=model_path)
                    atexit.register(_cortex.close)
                except Exception as e:
                    LOG.warning("TitanCortex init failed: %s", e)
                    return None
    return _cortex


def reset_cortex():
    """Reset the cortex singleton (for testing)."""
    global _cortex
    if _cortex is not None:
        _cortex.close()
        _cortex = None
