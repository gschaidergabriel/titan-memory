#!/usr/bin/env python3
"""
Basic usage example for Titan Memory.

Demonstrates the core API: ingest, retrieve, get_context_string, protect, forget.
"""

from titan import Titan, TitanConfig


def main():
    # Initialize with a custom data directory
    config = TitanConfig(data_dir="./my_agent_memory")
    memory = Titan(config)

    # ── Ingest memories ──────────────────────────────────────────
    print("Ingesting memories...")

    r1 = memory.ingest(
        "The user's name is Alice and she likes Python programming.",
        origin="user"
    )
    print(f"  Ingested: {r1['claims']} claims, entities: {r1['entities']}")

    r2 = memory.ingest(
        "Alice works at a startup in Berlin.",
        origin="user"
    )
    print(f"  Ingested: {r2['claims']} claims, entities: {r2['entities']}")

    r3 = memory.ingest(
        "Alice prefers dark mode and uses VS Code as her editor.",
        origin="observation"
    )
    print(f"  Ingested: {r3['claims']} claims, entities: {r3['entities']}")

    # ── Retrieve memories ────────────────────────────────────────
    print("\nRetrieving memories about Alice...")
    results = memory.retrieve("What do we know about Alice?")
    for i, item in enumerate(results):
        print(f"  [{i+1}] (score={item.get('score', 0):.3f}) {item.get('content', '')[:80]}")

    # ── Get formatted context string ─────────────────────────────
    print("\nFormatted context string:")
    context = memory.get_context_string("Tell me about Alice")
    print(context)

    # ── Get claims about an entity ───────────────────────────────
    print("\nClaims about 'Alice':")
    claims = memory.get_claims("Alice")
    for claim in claims:
        print(f"  {claim['subject']} --[{claim['predicate']}]--> {claim['object']} "
              f"(conf={claim['confidence']:.2f})")

    # ── Show statistics ──────────────────────────────────────────
    print("\nMemory statistics:")
    stats = memory.get_stats()
    print(f"  Nodes:   {stats['nodes']}")
    print(f"  Edges:   {stats['edges']}")
    print(f"  Events:  {stats['events']}")
    print(f"  Claims:  {stats['claims']}")
    print(f"  Vectors: {stats['vectors']}")
    print(f"  Model:   {stats['vector_model']}")

    # ── Health check ─────────────────────────────────────────────
    print("\nHealth check:")
    health = memory.health_check()
    for k, v in health.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
