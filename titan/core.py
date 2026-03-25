#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Titan Core - E-CPMM v5.1 Orchestrator

The main coordinator for the Titan Conversational Memory system.

Leitmotiv: Context is not text. Context is a time-weighted, uncertain
graph structure that is observed through text.

Key principles:
- Claims, not facts (epistemological humility)
- Time-weighted confidence decay
- Counter-hypotheses for inferences
- Controlled forgetting
- 10 neural micro-networks for living memory
"""

import atexit
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from .storage import (
    SQLiteStore, VectorStore, KnowledgeGraph, TriHybridStorage,
    Node, Edge, Claim, _ensure_dir
)
from .ingestion import Architect, ExtractionResult
from .retrieval import Retriever, RetrievedItem, ContextBuilder
from .maintenance import MaintenanceEngine, PruneStats

LOG = logging.getLogger("titan")


@dataclass
class TitanConfig:
    """Configuration for Titan memory system."""
    # Storage
    data_dir: str = "~/.titan/data"
    vector_model: str = "all-MiniLM-L6-v2"

    # Retrieval
    default_limit: int = 5
    min_confidence: float = 0.15
    expand_graph: bool = False

    # Maintenance
    auto_maintenance: bool = True
    maintenance_interval: int = 3600  # seconds

    # Consolidation
    auto_consolidation: bool = False
    consolidation_interval_hours: float = 6.0

    # Memory limits
    max_context_length: int = 4000

    def resolved_data_dir(self) -> Path:
        """Resolve ~ and return absolute Path."""
        return Path(self.data_dir).expanduser().resolve()


class Titan:
    """
    Titan - Neural-Enhanced Conversational Memory System.

    Main interface for:
    - Ingesting text into memory
    - Retrieving relevant context (traditional + neural)
    - Managing memory lifecycle
    - Running consolidation (neural training)

    Architecture:
    - Tri-hybrid storage: SQLite + Vector Store + Knowledge Graph
    - 6 Neural Cortex micro-networks (~77K params)
    - 4 Hippocampus neural modules (~226K params)
    - Consolidation engine for periodic training
    """

    def __init__(self, config: TitanConfig = None):
        self.config = config or TitanConfig()
        self._lock = threading.Lock()
        self._data_dir = self.config.resolved_data_dir()
        _ensure_dir(self._data_dir)

        db_path = self._data_dir / "titan.db"
        model_dir = self._data_dir / "models"
        _ensure_dir(model_dir)

        # Initialize storage layers
        self.sqlite = SQLiteStore(db_path)
        self.vectors = VectorStore(self._data_dir, self.config.vector_model)
        self.graph = KnowledgeGraph(self.sqlite)

        # Initialize processing layers
        self.architect = Architect(self.sqlite, self.vectors, self.graph)
        self.retriever = Retriever(self.sqlite, self.vectors, self.graph)
        self.maintenance = MaintenanceEngine(
            self.sqlite, self.vectors, self.graph
        )

        # Initialize Neural Cortex
        self._cortex = None
        try:
            from .neural_cortex import get_cortex, reset_cortex
            reset_cortex()  # Clean singleton state
            self._cortex = get_cortex(
                db_path=db_path,
                model_path=model_dir / "titan_cortex.pt"
            )
        except Exception as e:
            LOG.debug("Neural Cortex not available: %s", e)

        # Initialize Hippocampus
        self._hippocampus = None
        try:
            from .hippocampus import init_hippocampus, reset_hippocampus
            reset_hippocampus()
            self._hippocampus = init_hippocampus(
                data_dir=self._data_dir,
                titan_db_path=db_path,
                embed_fn=self.vectors.embed
            )
        except Exception as e:
            LOG.debug("Hippocampus not available: %s", e)

        # Initialize Consolidation Engine
        self._consolidation = None
        from .consolidation import ConsolidationEngine
        self._consolidation = ConsolidationEngine(
            self, interval_hours=self.config.consolidation_interval_hours
        )

        # Background maintenance thread
        self._maintenance_thread = None
        self._running = False

        if self.config.auto_maintenance:
            self._start_maintenance_thread()

        if self.config.auto_consolidation:
            self._consolidation.start_background()

        atexit.register(self._shutdown)
        LOG.info("Titan initialized (data_dir: %s)", self._data_dir)

    def _start_maintenance_thread(self):
        if self._maintenance_thread is not None:
            return
        self._running = True
        self._maintenance_thread = threading.Thread(
            target=self._maintenance_loop, daemon=True, name="TitanMaintenance"
        )
        self._maintenance_thread.start()

    def _maintenance_loop(self):
        while self._running:
            try:
                if self.maintenance.should_run_maintenance():
                    self.maintenance.run_maintenance()
            except Exception as e:
                LOG.error(f"Maintenance error: {e}")
            time.sleep(60)

    def _shutdown(self):
        self._running = False
        if self._consolidation:
            self._consolidation.stop()
        try:
            self.vectors.save()
            LOG.info("Titan shutdown complete")
        except Exception:
            pass

    # =========================================================================
    # Public API
    # =========================================================================

    def ingest(self, text: str, origin: str = "user",
               confidence: float = None) -> dict:
        """
        Ingest text into memory.

        Args:
            text: The text to ingest
            origin: Source of the text (user, code, inference, observation)
            confidence: Optional override for confidence level

        Returns:
            Dictionary with ingestion results
        """
        with self._lock:
            result = self.architect.ingest(text, origin, confidence)

        # Notify hippocampus to reload vectors
        if self._hippocampus:
            try:
                self._hippocampus.reload_vectors()
            except Exception:
                pass

        return {
            "success": True,
            "event_id": result.event_id,
            "claims": len(result.claims),
            "entities": result.entities,
            "topics": result.topics
        }

    def retrieve(self, query: str, limit: int = None, context: dict = None) -> List[dict]:
        """
        Retrieve relevant context for a query.
        Tries Hippocampus (neural) first, falls back to traditional retrieval.

        Args:
            query: The search query
            limit: Maximum number of results
            context: Optional context dict (valence, arousal, recent_messages, etc.)

        Returns:
            List of retrieved items as dictionaries
        """
        limit = limit or self.config.default_limit

        # Try Hippocampus neural retrieval only when MATURE (300+ training steps).
        # During cold start, the Hippocampus produces worse rankings than the
        # traditional RRF path. Let it train before trusting it.
        if self._hippocampus:
            try:
                steps = getattr(self._hippocampus, '_total_steps', 0)
                if self._hippocampus.is_ready() and steps >= 300:
                    result = self._hippocampus.retrieve(query, limit=limit, context=context or {})
                    if result:
                        return result
            except Exception as e:
                LOG.debug("Hippocampus retrieval failed: %s", e)

        # Traditional retrieval: FTS5 + Vector + RRF fusion
        items = self.retriever.retrieve(
            query, limit=limit, expand_graph=self.config.expand_graph
        )
        return [item.to_dict() for item in items]

    def get_context_string(self, query: str, limit: int = None, context: dict = None) -> str:
        """
        Get formatted context string for a query.
        Uses Hippocampus token-efficient format when available.
        """
        limit = limit or self.config.default_limit

        if self._hippocampus:
            try:
                steps = getattr(self._hippocampus, '_total_steps', 0)
                if self._hippocampus.is_ready() and steps >= 300:
                    items = self._hippocampus.retrieve(query, limit=limit, context=context or {})
                    if items:
                        return self._hippocampus.build_context_string(items)
            except Exception:
                pass

        return self.retriever.get_context_string(query, limit)

    def forget(self, node_id: str) -> bool:
        """Manually forget a node (if not protected)."""
        with self._lock:
            return self.maintenance._prune_node(node_id)

    def protect(self, node_id: str) -> bool:
        """Protect a node from automatic pruning."""
        return self.maintenance.protect_node(node_id)

    def unprotect(self, node_id: str) -> bool:
        """Remove protection from a node."""
        return self.maintenance.unprotect_node(node_id)

    def get_node(self, node_id: str) -> Optional[dict]:
        """Get a node by ID."""
        node = self.sqlite.get_node(node_id)
        return node.to_dict() if node else None

    def get_related(self, node_id: str, relation: str = None) -> List[dict]:
        """Get nodes related to a given node."""
        return self.graph.get_related(node_id, relation)

    def get_claims(self, entity: str) -> List[dict]:
        """Get claims related to an entity."""
        return self.retriever.builder.get_related_claims(entity)

    def add_counter_hypothesis(self, claim_id: str, counter_text: str) -> bool:
        """Add a counter-hypothesis to an existing claim."""
        return self.architect.add_counter_hypothesis(claim_id, counter_text)

    def run_maintenance(self) -> dict:
        """Manually trigger maintenance."""
        stats = self.maintenance.run_maintenance(force=True)
        return stats.to_dict()

    def consolidate(self) -> dict:
        """
        Manually trigger a consolidation cycle.

        This trains all neural components:
        - Neural Cortex (6 micro-networks)
        - Hippocampus (4 neural modules)
        - Runs maintenance
        """
        if self._consolidation:
            return self._consolidation.run_consolidation()
        return {"error": "Consolidation engine not available"}

    def get_stats(self) -> dict:
        """Get memory statistics."""
        sqlite_stats = self.sqlite.get_stats()
        vector_stats = self.vectors.get_stats()
        maintenance_status = self.maintenance.get_maintenance_status()

        stats = {
            "nodes": sqlite_stats["nodes"],
            "edges": sqlite_stats["edges"],
            "events": sqlite_stats["events"],
            "claims": sqlite_stats["claims"],
            "vectors": vector_stats["vectors"],
            "vector_model": vector_stats["model"],
            "last_maintenance": maintenance_status["last_maintenance"],
            "data_dir": str(self._data_dir),
        }

        if self._cortex:
            try:
                stats["neural_cortex"] = self._cortex.get_stats()
            except Exception:
                pass

        if self._hippocampus:
            try:
                stats["hippocampus"] = self._hippocampus.get_stats()
            except Exception:
                pass

        if self._consolidation:
            stats["consolidation"] = self._consolidation.get_stats()

        return stats

    def health_check(self) -> dict:
        """Check system health."""
        try:
            self.sqlite.get_stats()
            sqlite_ok = True
        except Exception:
            sqlite_ok = False

        try:
            _ = self.vectors._get_model()
            vectors_ok = True
        except Exception:
            vectors_ok = False

        return {
            "healthy": sqlite_ok and vectors_ok,
            "sqlite": "ok" if sqlite_ok else "error",
            "vectors": "ok" if vectors_ok else "error",
            "neural_cortex": "ok" if self._cortex else "unavailable",
            "hippocampus": "ok" if self._hippocampus else "unavailable",
            "data_dir": str(self._data_dir),
        }


# =========================================================================
# Singleton Access
# =========================================================================

_titan: Optional[Titan] = None


def get_titan(config: TitanConfig = None) -> Titan:
    """Get or create the Titan singleton."""
    global _titan
    if _titan is None:
        _titan = Titan(config)
    return _titan


def reset_titan():
    """Reset the Titan singleton (for testing)."""
    global _titan
    if _titan is not None:
        _titan._shutdown()
        _titan = None
    # Also reset sub-singletons
    from .neural_cortex import reset_cortex
    from .hippocampus import reset_hippocampus
    from .ingestion import get_architect
    from .retrieval import get_retriever
    from .maintenance import get_maintenance_engine
    reset_cortex()
    reset_hippocampus()
    # Reset module-level singletons
    import titan.ingestion as _ing
    import titan.retrieval as _ret
    import titan.maintenance as _maint
    _ing._architect = None
    _ret._retriever = None
    _maint._engine = None


# =========================================================================
# Convenience Functions
# =========================================================================

def remember(text: str, origin: str = "user", confidence: float = 0.8) -> dict:
    """Ingest text into memory."""
    return get_titan().ingest(text, origin=origin, confidence=confidence)


def recall(query: str, limit: int = 5) -> List[dict]:
    """Retrieve relevant context for a query."""
    return get_titan().retrieve(query, limit=limit)


def get_context(query: str) -> str:
    """Get assembled context string for a query."""
    return get_titan().get_context_string(query)


def forget(node_id: str) -> bool:
    """Manually forget a node (if not protected)."""
    return get_titan().forget(node_id)


def protect(node_id: str) -> bool:
    """Protect a node from pruning."""
    return get_titan().protect(node_id)
