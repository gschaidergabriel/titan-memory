#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Titan Maintenance Layer - Pruning & Aging

Handles:
1. Time-based confidence decay
2. Orphan node pruning
3. Low-confidence item removal
4. Memory compaction
5. Permanent memory graduation (NeuroCI-inspired)

Pruning rules (from spec):
- degree=0 (orphan nodes)
- protected=false
- confidence<0.2
- age>7 days
"""

import json
import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from .storage import SQLiteStore, VectorStore, KnowledgeGraph, Node

LOG = logging.getLogger("titan.maintenance")


# Pruning thresholds
PRUNE_MIN_CONFIDENCE = 0.2
PRUNE_MIN_AGE_DAYS = 7
DECAY_HALF_LIFE_DAYS = 7

# Compaction settings
MAX_NODES_SOFT = 10000
MAX_NODES_HARD = 50000


@dataclass
class PruneStats:
    """Statistics from a pruning run."""
    nodes_checked: int = 0
    nodes_pruned: int = 0
    nodes_graduated: int = 0
    edges_pruned: int = 0
    vectors_removed: int = 0
    confidence_updated: int = 0
    duration_ms: int = 0

    def to_dict(self) -> dict:
        return {
            "nodes_checked": self.nodes_checked,
            "nodes_pruned": self.nodes_pruned,
            "nodes_graduated": self.nodes_graduated,
            "edges_pruned": self.edges_pruned,
            "vectors_removed": self.vectors_removed,
            "confidence_updated": self.confidence_updated,
            "duration_ms": self.duration_ms
        }


class MaintenanceEngine:
    """
    Memory maintenance engine.

    Runs periodic maintenance tasks:
    - Decay confidence over time
    - Prune orphan nodes
    - Remove low-confidence items
    - Compact storage
    - Graduate high-value memories to permanent protection
    """

    def __init__(self, sqlite: SQLiteStore, vectors: VectorStore,
                 graph: KnowledgeGraph):
        self.sqlite = sqlite
        self.vectors = vectors
        self.graph = graph
        self._last_maintenance = 0
        self._maintenance_interval = 3600  # 1 hour

    def should_run_maintenance(self) -> bool:
        now = time.time()
        return (now - self._last_maintenance) > self._maintenance_interval

    def run_maintenance(self, force: bool = False) -> PruneStats:
        """
        Run full maintenance cycle.

        Steps:
        1. Unprotect expired nodes
        2. Update confidence decay
        3. Graduate high-value memories
        4. Neural Cortex access log cleanup
        5. Find and prune candidates
        6. Save vectors
        """
        if not force and not self.should_run_maintenance():
            return PruneStats()

        start_time = time.time()
        stats = PruneStats()

        LOG.info("Starting maintenance cycle")

        # 0. Unprotect nodes past their 24h protection window
        self._unprotect_expired_nodes()

        # 1. Update confidence decay
        stats.confidence_updated = self._decay_confidence()

        # 1b. Neural Cortex: access log cleanup
        try:
            from .neural_cortex import get_cortex
            _cortex = get_cortex()
            if _cortex:
                _cortex.cleanup_access_log(max_age_days=30)
        except Exception:
            pass

        # 1c. Permanent Memory Graduation (NeuroCI-inspired)
        stats.nodes_graduated = 0
        try:
            conn = self.sqlite._get_conn()
            graduated = conn.execute("""
                SELECT n.id, n.metadata FROM nodes n
                WHERE n.protected = 0
                AND n.created_at < datetime('now', '-3 days')
                AND n.id IN (
                    SELECT node_id FROM access_log
                    WHERE timestamp > ?
                    GROUP BY node_id
                    HAVING COUNT(*) >= 3
                )
            """, (time.time() - 14 * 86400,)).fetchall()

            for row in graduated:
                node_id = row[0]
                try:
                    meta = json.loads(row[1]) if row[1] else {}
                    utility = meta.get("utility", meta.get("confidence", 0.5))
                    if utility >= 0.6:
                        conn.execute(
                            "UPDATE nodes SET protected = 1 WHERE id = ?",
                            (node_id,)
                        )
                        meta["permanent"] = True
                        meta["graduated_at"] = datetime.now().isoformat()
                        conn.execute(
                            "UPDATE nodes SET metadata = ? WHERE id = ?",
                            (json.dumps(meta), node_id)
                        )
                        stats.nodes_graduated += 1
                except Exception:
                    continue
            if stats.nodes_graduated:
                conn.commit()
                LOG.info("Permanent memory: %d nodes graduated", stats.nodes_graduated)
        except Exception as e:
            LOG.debug("Permanent memory graduation skipped: %s", e)

        # 2. Find and prune candidates
        candidates = self._find_prune_candidates()
        stats.nodes_checked = len(candidates)

        for node_id, reason in candidates:
            if self._prune_node(node_id):
                stats.nodes_pruned += 1
                stats.vectors_removed += 1

        # 3. Save vectors
        self.vectors.save()

        stats.duration_ms = int((time.time() - start_time) * 1000)
        self._last_maintenance = time.time()

        LOG.info(f"Maintenance complete: {stats.nodes_pruned} nodes pruned "
                 f"in {stats.duration_ms}ms")

        return stats

    def _unprotect_expired_nodes(self):
        """Remove protection from nodes past their 24h unprotect_after window."""
        conn = self.sqlite._get_conn()
        now = datetime.now().isoformat()
        rows = conn.execute("""
            SELECT id, metadata FROM nodes WHERE protected = 1
        """).fetchall()
        unprotected = 0
        for row in rows:
            try:
                meta = json.loads(row["metadata"] or "{}")
                unprotect_after = meta.get("unprotect_after")
                if unprotect_after and unprotect_after < now:
                    conn.execute("UPDATE nodes SET protected = 0 WHERE id = ?", (row["id"],))
                    unprotected += 1
            except Exception:
                pass
        if unprotected:
            conn.commit()
            LOG.debug(f"Unprotected {unprotected} expired nodes")

    def _decay_confidence(self) -> int:
        """Apply time-based confidence decay to all nodes."""
        conn = self.sqlite._get_conn()
        updated = 0
        now = datetime.now()

        try:
            conn.execute("BEGIN IMMEDIATE")
        except Exception as e:
            LOG.warning(f"Decay skipped: could not acquire write lock: {e}")
            return 0

        try:
            rows = conn.execute("""
                SELECT id, created_at, metadata FROM nodes
                WHERE protected = 0
            """).fetchall()

            for row in rows:
                try:
                    node_id = row["id"]
                    created_at = row["created_at"]
                    metadata = row["metadata"]

                    if not metadata:
                        continue

                    meta = json.loads(metadata)
                    base_confidence = meta.get("confidence", 0.5)

                    created = datetime.fromisoformat(created_at)
                    age_days = (now - created).total_seconds() / 86400

                    decay_factor = math.pow(2, -age_days / DECAY_HALF_LIFE_DAYS)
                    new_confidence = base_confidence * decay_factor

                    if abs(new_confidence - base_confidence) > 0.05:
                        meta["effective_confidence"] = new_confidence
                        conn.execute("""
                            UPDATE nodes SET metadata = ? WHERE id = ?
                        """, (json.dumps(meta), node_id))
                        updated += 1

                except Exception as e:
                    if "locked" in str(e) or "database" in str(e).lower():
                        LOG.warning(f"Decay aborted after {updated} updates: {e}")
                        conn.rollback()
                        return updated
                    LOG.debug(f"Skipping node {row['id']}: {e}")

            conn.commit()
            return updated

        except Exception as e:
            LOG.warning(f"Decay failed, rolling back: {e}")
            try:
                conn.rollback()
            except Exception:
                pass
            return updated

    def _find_prune_candidates(self) -> List[Tuple[str, str]]:
        """Find nodes that are candidates for pruning."""
        conn = self.sqlite._get_conn()
        candidates = []
        now = datetime.now()
        cutoff = now - timedelta(days=PRUNE_MIN_AGE_DAYS)

        rows = conn.execute("""
            SELECT id, created_at, metadata FROM nodes
            WHERE protected = 0
            AND created_at < ?
        """, (cutoff.isoformat(),)).fetchall()

        for row in rows:
            node_id = row["id"]
            reason = None

            degree = self.sqlite.get_node_degree(node_id)
            if degree == 0:
                reason = "orphan"

            if not reason:
                try:
                    meta = json.loads(row["metadata"] or "{}")
                    effective_conf = meta.get("effective_confidence",
                                              meta.get("confidence", 0.5))
                    if effective_conf < PRUNE_MIN_CONFIDENCE:
                        reason = "low_confidence"
                except Exception:
                    pass

            if reason:
                candidates.append((node_id, reason))

        return candidates

    def _prune_node(self, node_id: str) -> bool:
        """Prune a single node."""
        try:
            node = self.sqlite.get_node(node_id)
            if not node or node.protected:
                return False

            self.vectors.remove(node_id)
            return self.sqlite.delete_node(node_id)

        except Exception as e:
            LOG.error(f"Failed to prune node {node_id}: {e}")
            return False

    def protect_node(self, node_id: str) -> bool:
        """Mark a node as protected from pruning."""
        return self.sqlite.set_protected(node_id, True)

    def unprotect_node(self, node_id: str) -> bool:
        """Remove protection from a node."""
        return self.sqlite.set_protected(node_id, False)

    def force_forget(self, node_id: str) -> bool:
        """Force forget a node (even if protected). Use with caution!"""
        try:
            self.sqlite.set_protected(node_id, False)
            self.vectors.remove(node_id)

            conn = self.sqlite._get_conn()
            conn.execute("DELETE FROM edges WHERE src = ? OR dst = ?",
                         (node_id, node_id))
            conn.execute("DELETE FROM memory_fts WHERE node_id = ?",
                         (node_id,))
            conn.execute("DELETE FROM nodes WHERE id = ?", (node_id,))
            conn.commit()

            return True
        except Exception as e:
            LOG.error(f"Failed to force forget {node_id}: {e}")
            return False

    def compact_if_needed(self) -> PruneStats:
        """Run aggressive compaction if over soft limit."""
        stats = self.sqlite.get_stats()
        node_count = stats["nodes"]

        if node_count < MAX_NODES_SOFT:
            return PruneStats()

        LOG.warning(f"Node count {node_count} exceeds soft limit, running compaction")
        return self._aggressive_prune(node_count)

    def _aggressive_prune(self, current_count: int) -> PruneStats:
        """Aggressive pruning when over limits."""
        stats = PruneStats()
        conn = self.sqlite._get_conn()

        target = MAX_NODES_SOFT * 0.8
        to_remove = int(current_count - target)

        if to_remove <= 0:
            return stats

        rows = conn.execute("""
            SELECT id, created_at, metadata FROM nodes
            WHERE protected = 0
            ORDER BY created_at ASC
            LIMIT ?
        """, (to_remove * 2,)).fetchall()

        pruned = 0
        for row in rows:
            if pruned >= to_remove:
                break

            node_id = row["id"]
            degree = self.sqlite.get_node_degree(node_id)

            if degree <= 2:
                if self._prune_node(node_id):
                    pruned += 1

        stats.nodes_pruned = pruned
        self.vectors.save()

        return stats

    def get_maintenance_status(self) -> dict:
        """Get current maintenance status."""
        stats = self.sqlite.get_stats()
        return {
            "nodes": stats["nodes"],
            "edges": stats["edges"],
            "vectors": self.vectors.get_stats()["vectors"],
            "last_maintenance": self._last_maintenance,
            "next_maintenance": self._last_maintenance + self._maintenance_interval,
            "soft_limit": MAX_NODES_SOFT,
            "hard_limit": MAX_NODES_HARD,
        }


_engine: Optional[MaintenanceEngine] = None


def get_maintenance_engine(sqlite: SQLiteStore, vectors: VectorStore,
                            graph: KnowledgeGraph) -> MaintenanceEngine:
    """Get or create maintenance engine."""
    global _engine
    if _engine is None:
        _engine = MaintenanceEngine(sqlite, vectors, graph)
    return _engine
