#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Titan Ingestion Pipeline - The Architect

Extracts CLAIMS (not facts!) from text.

Epistemological distinction:
- A fact is objectively true
- A claim is something asserted to be true
- All ingested content is treated as claims with confidence levels
"""

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from .storage import (
    SQLiteStore, VectorStore, KnowledgeGraph,
    Node, Edge, Claim,
)

LOG = logging.getLogger("titan.ingestion")


# Confidence levels by origin
ORIGIN_CONFIDENCE = {
    "user": 0.8,        # User stated explicitly
    "code": 0.95,       # From code analysis
    "inference": 0.5,   # AI inference
    "observation": 0.7, # Observed behavior
    "memory": 0.6,      # From memory recall
}


@dataclass
class ExtractedClaim:
    """A claim extracted from text."""
    subject: str
    predicate: str
    obj: str  # 'object' is reserved
    confidence: float
    source_text: str = ""

    def to_triple(self) -> Tuple[str, str, str]:
        return (self.subject, self.predicate, self.obj)

    def claim_id(self) -> str:
        return hashlib.sha256(
            f"{self.subject}:{self.predicate}:{self.obj}".encode()
        ).hexdigest()[:16]


@dataclass
class ExtractionResult:
    """Result of claim extraction."""
    claims: List[ExtractedClaim] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    raw_text: str = ""
    event_id: str = ""


class ClaimExtractor:
    """
    Extracts claims from text using pattern matching.

    Uses heuristic patterns rather than LLM to maintain
    local, deterministic operation.
    """

    # Relation patterns (subject, relation, object)
    RELATION_PATTERNS = [
        (r"(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+is\s+(?:a|an|the)\s+(.+?)(?:\.|,|$)", "is_a"),
        (r"(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+is\s+(.+?)(?:\.|,|$)", "is"),
        (r"(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+has\s+(?:a|an|the)\s+(.+?)(?:\.|,|$)", "has"),
        (r"(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+uses?\s+(.+?)(?:\.|,|$)", "uses"),
        (r"(?:I|[A-Z][a-z]+)\s+prefer(?:s)?\s+(.+?)(?:\s+over\s+(.+?))?(?:\.|,|$)", "prefers"),
        (r"(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+lives?\s+in\s+(.+?)(?:\.|,|$)", "lives_in"),
        (r"(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+works?\s+(?:at|for)\s+(.+?)(?:\.|,|$)", "works_at"),
        (r"(?:I|[A-Z][a-z]+)\s+(?:like|love|enjoy)s?\s+(.+?)(?:\.|,|$)", "likes"),
        (r"(?:I|[A-Z][a-z]+)\s+(?:hate|dislike)s?\s+(.+?)(?:\.|,|$)", "dislikes"),
        (r"(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+contains?\s+(.+?)(?:\.|,|$)", "contains"),
        (r"(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+depends?\s+on\s+(.+?)(?:\.|,|$)", "depends_on"),
        (r"(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+is\s+located\s+(?:in|at)\s+(.+?)(?:\.|,|$)", "located_in"),
        (r"(\b[a-z_]+(?:\.[a-z_]+)*)\s+(?:calls?|invokes?)\s+(\b[a-z_]+(?:\.[a-z_]+)*)(?:\.|,|$)", "calls"),
        (r"(\b[a-z_]+(?:\.[a-z_]+)*)\s+(?:imports?|requires?)\s+(\b[a-z_]+(?:\.[a-z_]+)*)(?:\.|,|$)", "imports"),
    ]

    # Entity patterns
    ENTITY_PATTERNS = [
        r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b",
        r"\b([a-z_]+\.[a-z_]+(?:\.[a-z_]+)*)\b",
        r"(/[a-z0-9_\-/]+(?:\.[a-z]+)?)",
        r"(https?://[^\s]+)",
    ]

    # Topic indicators
    TOPIC_INDICATORS = [
        "about", "regarding", "concerning", "topic", "subject",
        "related to", "discussing", "focus on"
    ]

    def __init__(self):
        self._compiled_relations = [
            (re.compile(p, re.IGNORECASE), rel)
            for p, rel in self.RELATION_PATTERNS
        ]
        self._compiled_entities = [
            re.compile(p) for p in self.ENTITY_PATTERNS
        ]

    def extract_claims(self, text: str, origin: str = "user") -> List[ExtractedClaim]:
        """Extract claims from text."""
        claims = []
        base_confidence = ORIGIN_CONFIDENCE.get(origin, 0.5)

        sentences = re.split(r'[.!?]+', text)

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 5:
                continue

            for pattern, relation in self._compiled_relations:
                matches = pattern.findall(sentence)
                for match in matches:
                    if isinstance(match, tuple) and len(match) >= 2:
                        subject = self._normalize_entity(match[0])
                        obj = self._normalize_entity(match[1])

                        if subject and obj and subject != obj:
                            claims.append(ExtractedClaim(
                                subject=subject,
                                predicate=relation,
                                obj=obj,
                                confidence=base_confidence,
                                source_text=sentence
                            ))

        return claims

    def extract_entities(self, text: str) -> List[str]:
        """Extract named entities from text."""
        entities = set()

        for pattern in self._compiled_entities:
            matches = pattern.findall(text)
            for match in matches:
                entity = self._normalize_entity(match)
                if entity and len(entity) >= 3:
                    entities.add(entity)

        return list(entities)

    def extract_topics(self, text: str) -> List[str]:
        """Extract topics from text."""
        topics = []
        text_lower = text.lower()

        for indicator in self.TOPIC_INDICATORS:
            if indicator in text_lower:
                idx = text_lower.find(indicator) + len(indicator)
                remaining = text[idx:idx + 50].strip()
                if remaining:
                    words = remaining.split()[:3]
                    topic = " ".join(words).strip(".,!?")
                    if topic and len(topic) >= 3:
                        topics.append(topic)

        return topics

    def _normalize_entity(self, entity: str) -> str:
        """Normalize an entity name."""
        if not entity:
            return ""
        entity = entity.strip().strip(".,!?:;\"'")
        if len(entity) < 2:
            return ""
        common_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been"}
        if entity.lower() in common_words:
            return ""
        return entity


class Architect:
    """
    The Architect - Main ingestion pipeline.

    Processes text -> extracts claims -> stores in Titan.
    """

    def __init__(self, sqlite: SQLiteStore, vectors: VectorStore,
                 graph: KnowledgeGraph):
        self.sqlite = sqlite
        self.vectors = vectors
        self.graph = graph
        self.extractor = ClaimExtractor()

    def ingest(self, text: str, origin: str = "user",
               confidence: float = None) -> ExtractionResult:
        """
        Ingest text into memory.
        Returns extraction result with claims, entities, topics.
        """
        if not text or not text.strip():
            return ExtractionResult()

        # Generate event ID
        event_id = hashlib.sha256(
            f"{text}:{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]

        base_confidence = confidence or ORIGIN_CONFIDENCE.get(origin, 0.5)

        # Store raw event
        self.sqlite.add_event(event_id, text, origin)

        # Extract claims
        claims = self.extractor.extract_claims(text, origin)

        # Extract entities
        entities = self.extractor.extract_entities(text)

        # Extract topics
        topics = self.extractor.extract_topics(text)

        # Store claims
        now = datetime.now().isoformat()
        for claim in claims:
            self.sqlite.add_claim(Claim(
                subject=claim.subject,
                predicate=claim.predicate,
                object=claim.obj,
                confidence=claim.confidence,
                origin=origin,
                timestamp=now,
                source_event_id=event_id
            ))

            self.graph.add_relation(
                subject=claim.subject,
                predicate=claim.predicate,
                obj=claim.obj,
                confidence=claim.confidence,
                origin=origin
            )

        # Store entities as nodes
        for entity in entities:
            node_id = hashlib.sha256(entity.encode()).hexdigest()[:16]
            if not self.sqlite.get_node(node_id):
                self.sqlite.add_node(Node(
                    id=node_id, type="entity", label=entity,
                    created_at=now, protected=False,
                    metadata={"origin": origin}
                ))
                self.vectors.add(node_id, entity)

        # --- Neural Cortex: importance scoring + emotional tagging ---
        _mis_confidence = base_confidence
        _valence, _arousal = 0.0, 0.5
        chunk_id = f"chunk_{event_id}"
        _cortex = None
        try:
            from .neural_cortex import get_cortex
            _cortex = get_cortex()
            if _cortex:
                _mis_f = {
                    "origin": origin,
                    "claim_count": len(claims),
                    "entity_count": len(entities),
                    "topic_count": len(topics),
                    "text_length": len(text),
                    "has_user": "user" in text.lower(),
                    "has_entity": bool(entities),
                    "hour": datetime.now().hour,
                }
                _mis_confidence = _cortex.score_importance(_mis_f)

                # ET: emotional tagging
                _emb = self.vectors.embed(text)
                _valence, _arousal = _cortex.tag_emotion(_emb)
        except Exception:
            pass

        # --- CraniMem-inspired Composite Utility Score ---
        # Fuses 3 signals: Importance (MIS) + Surprise + Emotion
        _importance = _mis_confidence
        _surprise = 0.5  # Default: neutral surprise (no external NAc dependency)
        _emotion = max(abs(_valence), _arousal) if _valence != 0.0 else 0.5

        # Composite: (Importance + Surprise + Emotion) / 3
        _utility = (_importance + _surprise + _emotion) / 3.0
        _utility = max(0.1, min(1.0, _utility))

        # Store memory chunk (protected for 24h to survive first maintenance cycle)
        self.sqlite.add_node(Node(
            id=chunk_id,
            type="memory",
            label=text[:100],
            created_at=now,
            protected=True,
            metadata={
                "origin": origin,
                "confidence": max(_mis_confidence, 0.8),
                "utility": round(_utility, 3),
                "full_text": text,
                "event_id": event_id,
                "unprotect_after": (datetime.now() + timedelta(hours=24)).isoformat(),
                "valence": round(_valence, 3),
                "arousal": round(_arousal, 3),
                "surprise": round(_surprise, 3),
                "claim_count": len(claims),
                "entity_count": len(entities),
                "topic_count": len(topics),
            }
        ))
        self.sqlite.index_for_fts(chunk_id, text, {"origin": origin})
        self.vectors.add(chunk_id, text)

        # --- Neural Cortex: interference detection ---
        try:
            if _cortex and claims:
                _check_claim_interference(
                    _cortex, self.sqlite, self.vectors, claims, chunk_id)
        except Exception:
            pass

        # Link entities to memory chunk
        for entity in entities:
            entity_id = hashlib.sha256(entity.encode()).hexdigest()[:16]
            if self.sqlite.get_node(entity_id) and self.sqlite.get_node(chunk_id):
                self.graph.add_relation(
                    subject=chunk_id,
                    predicate="mentions",
                    obj=entity_id,
                    confidence=base_confidence,
                    origin=origin
                )

        LOG.debug(f"Ingested text: {len(claims)} claims, {len(entities)} entities")

        return ExtractionResult(
            claims=claims,
            entities=entities,
            topics=topics,
            raw_text=text,
            event_id=event_id
        )

    def ingest_code(self, code: str, filepath: str) -> ExtractionResult:
        """Ingest code with higher confidence."""
        text = f"Code at {filepath}: {code[:500]}"
        result = self.ingest(text, origin="code", confidence=0.95)

        code_id = hashlib.sha256(filepath.encode()).hexdigest()[:16]
        now = datetime.now().isoformat()

        self.sqlite.add_node(Node(
            id=code_id, type="code", label=filepath,
            created_at=now, protected=True,
            metadata={
                "filepath": filepath,
                "content_hash": hashlib.sha256(code.encode()).hexdigest()[:16]
            }
        ))

        return result

    def add_counter_hypothesis(self, claim_id: str, counter_text: str,
                                origin: str = "inference") -> bool:
        """Add a counter-hypothesis to an existing claim."""
        counter_id = f"counter_{claim_id}_{hashlib.sha256(counter_text.encode()).hexdigest()[:8]}"
        now = datetime.now().isoformat()

        self.sqlite.add_node(Node(
            id=counter_id, type="counter_hypothesis",
            label=counter_text[:100], created_at=now,
            protected=False,
            metadata={"full_text": counter_text, "origin": origin}
        ))

        self.graph.add_relation(
            subject=counter_id, predicate="contradicts",
            obj=claim_id, confidence=0.5, origin=origin
        )

        return True


def _check_claim_interference(cortex, sqlite, vectors, claims, chunk_id):
    """Check new claims against existing ones for contradictions."""
    try:
        for claim in claims[:3]:
            claim_text = f"{claim.subject} {claim.predicate} {claim.obj}"
            claim_emb = vectors.embed(claim_text)
            similar = vectors.search(claim_text, limit=5)
            for node_id, sim_score in similar:
                if sim_score < 0.3:
                    continue
                if node_id == chunk_id:
                    continue
                node = sqlite.get_node(node_id)
                if not node:
                    continue
                existing_emb = vectors.embed(node.label)
                interference = cortex.detect_interference(claim_emb, existing_emb)
                if interference > 0.7:
                    sqlite.add_edge(Edge(
                        src=chunk_id, dst=node_id,
                        relation="contradicts",
                        confidence=interference,
                        origin="neural_cortex",
                        created_at=datetime.now().isoformat()
                    ))
                    break
    except Exception:
        pass


# Singleton architect instance
_architect: Optional[Architect] = None


def get_architect(sqlite: SQLiteStore, vectors: VectorStore,
                  graph: KnowledgeGraph) -> Architect:
    """Get or create architect instance."""
    global _architect
    if _architect is None:
        _architect = Architect(sqlite, vectors, graph)
    return _architect
