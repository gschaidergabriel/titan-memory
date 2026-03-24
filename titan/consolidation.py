#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Titan Consolidation Engine

Self-contained training cycle for Titan's neural components.

When integrated into a larger system, consolidation can be triggered externally
(e.g., during idle phases or a dream/sleep cycle). In standalone mode, this
runs automatically on a configurable schedule via a background thread.

Consolidation cycle:
1. Neural Cortex train_cycle() -- trains MIS, ET, RWL, AS, CG, ID
2. Hippocampus train_cycle() -- trains QUM, AMM, CDM, RCM
3. Run maintenance (decay, pruning, graduation)
"""

import logging
import threading
import time
from typing import Any, Dict, Optional

LOG = logging.getLogger("titan.consolidation")


class ConsolidationEngine:
    """Self-contained training cycle for Titan's neural components."""

    def __init__(self, titan: Any, interval_hours: float = 6):
        """
        Args:
            titan: Titan instance
            interval_hours: How often to run consolidation (default: every 6 hours)
        """
        self._titan = titan
        self._interval_s = interval_hours * 3600
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._last_consolidation = 0.0
        self._consolidation_count = 0

    def run_consolidation(self) -> Dict:
        """
        Full consolidation cycle:
        1. Neural Cortex train_cycle()
        2. Hippocampus train_cycle()
        3. Run maintenance (decay, pruning, graduation)

        Returns:
            Dictionary with consolidation stats
        """
        t0 = time.time()
        stats = {
            "cortex": {},
            "hippocampus": {},
            "maintenance": {},
        }

        # 1. Neural Cortex training
        try:
            from .neural_cortex import get_cortex
            cortex = get_cortex()
            if cortex:
                cortex.train_cycle()
                stats["cortex"] = cortex.get_stats()
                LOG.info("Cortex training complete")
        except Exception as e:
            LOG.warning("Cortex training failed: %s", e)
            stats["cortex"] = {"error": str(e)}

        # 2. Hippocampus training
        try:
            from .hippocampus import get_hippocampus
            hippo = get_hippocampus()
            if hippo:
                stats["hippocampus"] = hippo.train_cycle()
                LOG.info("Hippocampus training complete")
        except Exception as e:
            LOG.warning("Hippocampus training failed: %s", e)
            stats["hippocampus"] = {"error": str(e)}

        # 3. Maintenance
        try:
            maint_stats = self._titan.maintenance.run_maintenance(force=True)
            stats["maintenance"] = maint_stats.to_dict()
            LOG.info("Maintenance complete")
        except Exception as e:
            LOG.warning("Maintenance failed: %s", e)
            stats["maintenance"] = {"error": str(e)}

        # 4. Save vectors
        try:
            self._titan.vectors.save()
        except Exception:
            pass

        elapsed = time.time() - t0
        stats["elapsed_s"] = round(elapsed, 2)
        stats["consolidation_count"] = self._consolidation_count + 1
        self._last_consolidation = time.time()
        self._consolidation_count += 1

        LOG.info("Consolidation cycle #%d complete in %.1fs",
                 self._consolidation_count, elapsed)

        return stats

    def start_background(self):
        """Start background consolidation thread."""
        if self._thread is not None and self._thread.is_alive():
            LOG.warning("Consolidation thread already running")
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._background_loop,
            daemon=True,
            name="TitanConsolidation"
        )
        self._thread.start()
        LOG.info("Background consolidation started (interval: %.1fh)",
                 self._interval_s / 3600)

    def stop(self):
        """Stop background consolidation."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        LOG.info("Background consolidation stopped")

    def _background_loop(self):
        """Background loop that runs consolidation periodically."""
        while self._running:
            try:
                now = time.time()
                if now - self._last_consolidation >= self._interval_s:
                    self.run_consolidation()
            except Exception as e:
                LOG.error("Consolidation error: %s", e)

            # Sleep in small increments to allow clean shutdown
            for _ in range(60):
                if not self._running:
                    return
                time.sleep(1)

    @property
    def is_running(self) -> bool:
        return self._running and self._thread is not None and self._thread.is_alive()

    def get_stats(self) -> Dict:
        return {
            "running": self.is_running,
            "interval_hours": self._interval_s / 3600,
            "last_consolidation": self._last_consolidation,
            "consolidation_count": self._consolidation_count,
        }
