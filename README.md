# WE INTRODUCE TITAN THE Neural-Enhanced Conversational Memory System

**Titan** is a standalone, open-source Python library that gives any AI agent a living, neural-enhanced long-term memory. It replaces the brittle "dump everything into a vector DB" approach with something closer to how biological memory actually works.

> *"Context is not text. Context is a time-weighted, uncertain graph structure that is observed through text."*

## Why Titan?

Most agent memory systems treat memory as a retrieval problem. Titan treats it as a **living system** with 10 neural micro-networks that learn, forget, consolidate, and adapt.

**What makes Titan different:**
- **Claims, not facts.** Every piece of information is a *claim* with a confidence level and origin. Nothing is treated as ground truth.
- **Time-weighted decay.** Confidence halves every 7 days. Old memories fade unless they prove useful.
- **Controlled forgetting.** Orphan nodes, low-confidence items, and stale data are pruned automatically. High-value memories graduate to permanent protection.
- **10 neural micro-networks** (~303K params total, CPU-only) that learn from actual usage patterns.
- **Tri-hybrid storage:** SQLite (deterministic facts) + Vector Store (semantic search) + Knowledge Graph (explicit relations).
- **Bio-inspired retrieval** with Hopfield associative memory, spreading activation, and cross-attention re-ranking.

## Architecture

```
                        ┌─────────────────────────────────────────────────────────┐
                        │                    TITAN MEMORY                         │
                        │                                                         │
    text ──────────────►│  INGESTION PIPELINE (The Architect)                     │
                        │  ├── Claim Extraction (regex, no LLM)                   │
                        │  ├── Entity Recognition                                 │
                        │  ├── Neural Cortex: MIS (importance), ET (emotion)      │
                        │  ├── CraniMem Composite Utility Score                   │
                        │  └── Interference Detection (ID network)                │
                        │                                                         │
                        │  TRI-HYBRID STORAGE                                     │
                        │  ├── SQLite + FTS5 (The Ledger)                        │
                        │  ├── Vector Store: all-MiniLM-L6-v2 (Semantic Field)   │
                        │  └── Knowledge Graph (The Structure)                    │
                        │                                                         │
    query ─────────────►│  RETRIEVAL PIPELINE                                     │
                        │  ├── Traditional: RRF + Graph Expansion (SYNAPSE)       │
                        │  │   └── Neural Cortex: RWL (learned weights)          │
                        │  └── Hippocampus (Neural):                              │
                        │      ├── QUM: Query Understanding (38-dim features)     │
                        │      ├── AMM: Hopfield Pattern Completion               │
                        │      ├── CDM: Cross-Attention Re-Ranking               │
                        │      └── RCM: Response Composition + Dedup             │
                        │                                                         │
                        │  CONSOLIDATION (Background)                             │
                        │  ├── Neural Cortex train_cycle (6 networks)            │
                        │  ├── Hippocampus train_cycle (4 networks)              │
                        │  └── Maintenance: decay, prune, graduate               │
                        └─────────────────────────────────────────────────────────┘
```

## Neural Components

### Neural Cortex (~77K params)

| Module | Params | Purpose |
|--------|--------|---------|
| **MIS** — Memory Importance Scorer | ~1K | Learns which memories deserve high initial confidence |
| **ET** — Emotional Tagger | ~13K | Tags memories with valence (positive/negative) and arousal |
| **RWL** — Retrieval Weight Learner | ~200 | Learns optimal weights for RRF, confidence, recency, graph signals |
| **AS** — Associative Strengthener | ~25K | Hebbian learning for edge confidence updates |
| **CG** — Consolidation Gate | ~13K | Decides keep/compress/forget during consolidation |
| **ID** — Interference Detector | ~25K | Detects contradicting memory pairs |

### Hippocampus (~226K params)

| Module | Params | Purpose |
|--------|--------|---------|
| **QUM** — Query Understanding | ~58K | Extracts intent, temporal, emotional, and expansion signals |
| **AMM** — Associative Memory | ~89K | Modern Hopfield network with temporal binding and Hebbian gates |
| **CDM** — Correlation Discovery | ~53K | Cross-attention re-ranker with causal chain detection |
| **RCM** — Response Composer | ~26K | Token-efficient formatting, confidence calibration, deduplication |

## Feature Comparison

| Feature | Titan | Mem0 | Letta | Graphiti | Zep |
|---------|-------|------|-------|----------|-----|
| Tri-hybrid storage (SQL + Vector + Graph) | Yes | No | Partial | Graph only | No |
| Neural importance scoring | Yes (10 networks) | No | No | No | No |
| Epistemological uncertainty (claims, not facts) | Yes | No | No | No | No |
| Time-weighted confidence decay | Yes | No | No | No | No |
| Controlled forgetting with graduation | Yes | Manual | No | No | Yes |
| Bio-inspired retrieval (Hopfield, spreading activation) | Yes | No | No | No | No |
| Counter-hypotheses | Yes | No | No | No | No |
| Self-training consolidation | Yes | No | No | No | No |
| Fully local, no API keys | Yes | No | Partial | No | No |
| CPU-only neural inference (<20ms) | Yes | N/A | N/A | N/A | N/A |

## Installation

```bash
pip install titan-memory
```

Or install from source:

```bash
git clone https://github.com/gschaidergabriel/titan-memory.git
cd titan-memory
pip install -e .
```

**Requirements:** Python 3.10+, PyTorch 2.0+, sentence-transformers 2.2+

## Quick Start

```python
from titan import Titan, TitanConfig

# Initialize with custom data directory
config = TitanConfig(data_dir="./my_agent_memory")
memory = Titan(config)

# Ingest memories
memory.ingest("The user's name is Alice and she likes Python.", origin="user")
memory.ingest("Alice works at a startup in Berlin.", origin="user")

# Retrieve
results = memory.retrieve("What do we know about Alice?")
for item in results:
    print(f"[{item.get('confidence', 0):.2f}] {item['content']}")

# Get formatted context for LLM injection
context = memory.get_context_string("Tell me about Alice")
print(context)

# Stats
print(memory.get_stats())
```

### Convenience Functions

```python
from titan import remember, recall, get_context, forget, protect

# Uses a global singleton (data in ~/.titan/data/)
remember("The user prefers dark mode.")
results = recall("user preferences")
context = get_context("What are the user's preferences?")
```

## API Reference

### `TitanConfig`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_dir` | `str` | `~/.titan/data` | Directory for all Titan data (DB, vectors, models) |
| `vector_model` | `str` | `all-MiniLM-L6-v2` | SentenceTransformer model for embeddings |
| `default_limit` | `int` | `5` | Default number of results to return |
| `min_confidence` | `float` | `0.15` | Minimum effective confidence for retrieval |
| `expand_graph` | `bool` | `True` | Enable graph expansion in retrieval |
| `auto_maintenance` | `bool` | `True` | Run maintenance automatically (hourly) |
| `auto_consolidation` | `bool` | `False` | Run neural training automatically |
| `consolidation_interval_hours` | `float` | `6.0` | Hours between consolidation cycles |
| `max_context_length` | `int` | `4000` | Maximum context string length |

### `Titan`

| Method | Returns | Description |
|--------|---------|-------------|
| `ingest(text, origin, confidence)` | `dict` | Ingest text into memory. Returns claims, entities, topics. |
| `retrieve(query, limit, context)` | `list[dict]` | Retrieve relevant memories. Uses Hippocampus if available. |
| `get_context_string(query, limit)` | `str` | Get LLM-ready formatted context string. |
| `forget(node_id)` | `bool` | Manually forget a memory (if not protected). |
| `protect(node_id)` | `bool` | Protect a memory from automatic pruning. |
| `unprotect(node_id)` | `bool` | Remove protection from a memory. |
| `consolidate()` | `dict` | Manually trigger neural training cycle. |
| `run_maintenance()` | `dict` | Manually trigger maintenance (decay, prune, graduate). |
| `get_stats()` | `dict` | Get memory statistics. |
| `health_check()` | `dict` | Check system health. |
| `get_claims(entity)` | `list[dict]` | Get claims related to an entity. |
| `get_related(node_id, relation)` | `list[dict]` | Get nodes related to a given node. |
| `add_counter_hypothesis(claim_id, text)` | `bool` | Add counter-hypothesis to a claim. |

### Origin Types

| Origin | Default Confidence | Description |
|--------|--------------------|-------------|
| `user` | 0.8 | User stated explicitly |
| `code` | 0.95 | From code analysis |
| `inference` | 0.5 | AI inference |
| `observation` | 0.7 | Observed behavior |
| `memory` | 0.6 | From memory recall |

## How the Neural Components Work

### Cold Start

All neural networks start in **cold start** mode with smoothstep blending:

1. **Bootstrap phase** (0-50 steps): Fully rule-based defaults. Neural outputs ignored.
2. **Blending phase** (50-150 steps): 50/50 blend between neural and rule-based.
3. **Maturing phase** (150-300 steps): 75% neural, 25% rule-based.
4. **Mature phase** (300+ steps): 95% neural.

This means Titan works perfectly from step zero -- it just gets better over time.

### Consolidation

When `consolidate()` is called (or runs automatically):

1. **Neural Cortex** trains on accumulated data:
   - MIS learns which memories get retrieved (supervised on access logs)
   - RWL learns optimal retrieval weights (REINFORCE-style from feedback)
   - ET, AS, CG, ID training deferred until sufficient embedding data

2. **Hippocampus** trains on outcome signals:
   - CDM trains on which results were useful (binary cross-entropy)
   - RCM trains on deduplication detection (cosine similarity labels)

3. **Maintenance** runs: decay confidence, prune orphans, graduate high-value memories.

### Retrieval Pipeline

1. **Traditional path**: FTS5 keyword search + Vector cosine similarity -> RRF fusion -> Neural Cortex RWL weighting -> SYNAPSE spreading activation
2. **Hippocampus path** (when ready): QUM query understanding -> ANN + Hopfield cluster expansion -> CDM cross-attention re-ranking -> RCM response composition with deduplication

The Hippocampus path is tried first; if unavailable (cold start, no vectors), falls back to traditional.

## Performance

| Operation | Latency | Notes |
|-----------|---------|-------|
| Ingest | ~5-15ms | Depends on text length and claim extraction |
| Retrieve (traditional) | ~2-8ms | FTS5 + vector search + RRF |
| Retrieve (hippocampus) | ~5-20ms | Target: <20ms on CPU |
| Consolidation cycle | ~50-200ms | Neural training + maintenance |
| Neural Cortex inference | <1ms | All 6 networks combined |

All neural inference runs on CPU. No GPU required.

## Benchmarks

Head-to-head comparison against Mem0 v1.0.7 on 200 conversational memory items, 15 labeled queries, CPU-only (AMD Ryzen 9 7940HS), same embedding model (`all-MiniLM-L6-v2`). Graphiti and Letta could not be benchmarked due to hard infrastructure requirements (Neo4j server, Letta server).

| Metric | Titan | Mem0 (local Qwen-3B) |
|--------|-------|----------------------|
| **Ingest latency** | **51.5 ms/item** | 11,837 ms/item (230x slower) |
| Retrieval P50 | 20.4 ms | **10.5 ms** |
| **Precision@5** | 0.400 | **0.867** |
| **MRR** | 0.244 | **0.800** |
| RAM delta | **9.4 MB** | 93.4 MB |
| External dependencies | **None** | ChromaDB + LLM API |

**The tradeoff:** Titan uses deterministic regex extraction (fast, zero dependencies) while Mem0 uses full LLM inference for extraction (slow, better quality). For always-on local agents with continuous memory ingestion, Titan's 230x speed advantage and zero-dependency deployment are decisive. For infrequent memory storage with cloud API access, Mem0's extraction quality is superior.

See the full **[Benchmark Paper](docs/BENCHMARK.md)** for methodology, analysis, and detailed discussion.

## Data Directory Structure

```
~/.titan/data/
├── titan.db              # SQLite: nodes, edges, events, claims, FTS5, training signals
├── titan_vectors.npz     # Compressed vector embeddings
├── titan_vector_ids.json # Node ID to vector index mapping
├── hippocampus.db        # Hippocampus training logs
└── models/
    ├── titan_cortex.pt   # Neural Cortex checkpoint (6 networks)
    └── titan_hippocampus.pt  # Hippocampus checkpoint (4 networks)
```

## Framework Integrations

Titan works with any LLM framework. Two universal patterns:

1. **System prompt augmentation** -- inject `memory.get_context_string(query)` into the system message before each LLM call. Works with every model.
2. **Tool-based** -- give the LLM `memory_store` / `memory_recall` tools and let it decide when to use them. Requires tool-calling support.

| Framework | Pattern | Example | Guide |
|-----------|---------|---------|-------|
| **Claude API** (Anthropic SDK) | `tool_use` loop | [`examples/claude_api.py`](examples/claude_api.py) | [Guide](docs/INTEGRATIONS.md#claude-api-anthropic-sdk) |
| **Ollama + Python** | System prompt + `[STORE:]` | [`examples/ollama_chat.py`](examples/ollama_chat.py) | [Guide](docs/INTEGRATIONS.md#ollama--python) |
| **OpenAI-Compatible** (vLLM, LocalAI, Groq, etc.) | System prompt + tools | [`examples/openai_compatible.py`](examples/openai_compatible.py) | [Guide](docs/INTEGRATIONS.md#openai-compatible-apis) |
| **LangChain** | BaseMemory / LCEL / AgentExecutor | [`examples/langchain_memory.py`](examples/langchain_memory.py) | [Guide](docs/INTEGRATIONS.md#langchain) |
| **LlamaIndex** | MemoryBlock + Retriever | -- | [Guide](docs/INTEGRATIONS.md#llamaindex) |
| **CrewAI** | `@tool` decorator | -- | [Guide](docs/INTEGRATIONS.md#crewai) |
| **AutoGen v0.4** | Memory protocol subclass | -- | [Guide](docs/INTEGRATIONS.md#autogen-v04) |
| **Pydantic AI** | deps injection + `@agent.tool` | -- | [Guide](docs/INTEGRATIONS.md#pydantic-ai) |
| **Open Interpreter** | `custom_instructions` | -- | [Guide](docs/INTEGRATIONS.md#open-interpreter) |

See the full **[Integration Guide](docs/INTEGRATIONS.md)** for complete code, gotchas, and install instructions.

## License

MIT License. See [LICENSE](LICENSE).
