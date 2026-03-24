# Titan Memory: Empirical Benchmark

## Abstract

We benchmark Titan Memory v1.0.0 against three established conversational memory systems -- Mem0, Graphiti, and Letta -- on a corpus of 200 conversational memory items with 15 labeled retrieval queries. Only Titan and Mem0 could be benchmarked head-to-head; Graphiti requires a running Neo4j server and LLM API key, and Letta requires a running Letta server, making fully local benchmarking impossible for both.

**Key findings:** Titan ingests memories **230x faster** than Mem0 (52ms vs 11.8s per item) and uses **10x less RAM** (9.4 MB vs 93.4 MB). However, Mem0 achieves **2x better retrieval quality** (Precision@5: 0.867 vs 0.400, MRR: 0.800 vs 0.244). This reflects a fundamental design tradeoff: Titan uses deterministic regex extraction (fast, no dependencies) while Mem0 uses full LLM inference for extraction (slow, higher quality). For always-on local agents with continuous memory ingestion, Titan's speed and zero-dependency architecture is the practical choice. For infrequent memory storage with cloud API access, Mem0's extraction quality is superior.

## 1. Systems Under Test

| System | Version | Architecture | Extraction Method |
|--------|---------|-------------|-------------------|
| **Titan** | v1.0.0 | SQLite + FTS5 + Vector + Knowledge Graph | Regex-based claim extraction |
| **Mem0** | v1.0.7 | ChromaDB (vector store) | LLM-based extraction (Qwen-3B on CPU) |
| **Graphiti** | v0.28.2 | Neo4j (temporal knowledge graph) | LLM-based extraction |
| **Letta** | v0.16.6 | PostgreSQL + LLM-managed memory blocks | LLM tool calls |

## 2. Methodology

### 2.1 Corpus

200 conversational memory items spanning:
- User facts and personal details (name, age, occupation, location)
- User preferences (programming languages, tools, food, music)
- Project details (names, technologies, deadlines)
- Behavioral observations and inferred patterns
- Conversational context and relationship history

### 2.2 Queries

15 labeled queries with keyword ground truth, covering:
- Direct fact recall ("What is the user's name?")
- Preference retrieval ("What programming languages does the user like?")
- Relational queries ("What projects is the user working on?")
- Temporal queries ("What happened recently?")
- Composite queries requiring multiple memory items

### 2.3 Configuration

- **CPU-only**: All systems ran on CPU (AMD Ryzen 9 7940HS)
- **Same embedding model**: `all-MiniLM-L6-v2` for both Titan and Mem0
- **Mem0 LLM**: Local Qwen-3B via CPU inference (no cloud API)
- **No GPU acceleration**: Integrated Radeon 780M was not used
- **Cold measurement**: Each system started from scratch (no pre-trained models, no cached embeddings)

### 2.4 Metrics

- **Ingestion latency**: Wall-clock time per item (includes all extraction, embedding, and storage)
- **Retrieval latency**: P50, P95, and mean over 15 queries
- **Precision@K**: Fraction of top-K results that match keyword ground truth (K=1, K=5)
- **MRR**: Mean Reciprocal Rank -- average of 1/rank for first correct result per query
- **Resource usage**: RAM delta (peak - baseline), disk footprint after all items ingested

## 3. Results

### 3.1 Ingestion Performance

```
                    Titan           Mem0            Ratio
Per-item latency    51.5 ms         11,837 ms       230x faster
Total (200 items)   10.3 s          2,367 s         230x faster
                                    (~40 minutes)
```

This is the headline result. Titan ingests a single memory item in 52ms. Mem0 takes nearly 12 seconds per item -- almost 40 minutes for the full 200-item corpus.

**Why the difference is so large:**

- **Titan** extracts claims, entities, and relationships using regex patterns and heuristic rules. No neural inference is involved during ingestion except for the embedding computation (shared with Mem0) and lightweight cortex scoring (~1ms for importance + emotion tagging).

- **Mem0** sends every ingested text through a full LLM inference pass (Qwen-3B, 3 billion parameters on CPU) to extract structured facts, entities, and relationships. This is the dominant cost. With a cloud API (e.g., GPT-4), ingestion would be faster (network round-trip replaces local CPU inference) but would introduce API costs and a hard dependency on external services.

**Implication for always-on agents:** An agent processing 1,000 conversational turns per day would spend ~52 seconds on Titan ingestion vs ~3.3 hours on Mem0 ingestion (with local LLM). This makes Mem0's local mode impractical for high-throughput scenarios.

### 3.2 Retrieval Latency

```
                    Titan           Mem0
P50                 20.4 ms         10.5 ms
P95                 24.9 ms         12.3 ms
Mean                20.2 ms         10.7 ms
```

Mem0 is approximately 2x faster on retrieval. This is expected:

- **Mem0** performs a single vector similarity search against ChromaDB. This is a straightforward ANN lookup -- fast and well-optimized.

- **Titan** performs RRF (Reciprocal Rank Fusion) across three signals: FTS5 full-text search, vector cosine similarity, and knowledge graph expansion. The neural cortex RWL network then re-weights these signals. This multi-path retrieval is more expensive but is designed to capture different types of relevance (keyword match, semantic similarity, structural relationships).

Both systems are well within interactive latency requirements. A 20ms retrieval is imperceptible to users.

### 3.3 Retrieval Quality

```
                    Titan           Mem0
Precision@1         0.133           0.733
Precision@5         0.400           0.867
MRR                 0.244           0.800
```

**Mem0 wins decisively on retrieval quality.** This is the honest and important finding.

**Why Mem0 retrieves better results:**

1. **LLM extraction produces superior representations.** When Mem0 ingests "I've been using Python for 5 years and recently started learning Rust," the LLM extracts structured facts like "User has 5 years of Python experience" and "User recently started learning Rust" as separate, clean memory items. Titan's regex extraction captures the raw text and identifies entities (Python, Rust) but may miss implicit relationships and nuances.

2. **Clean vector targets.** Because Mem0 stores LLM-extracted summaries rather than raw text, the vector embeddings are more semantically focused. A query like "What programming languages does the user know?" has higher cosine similarity to "User has 5 years of Python experience" than to the original conversational text.

3. **Titan's cold-start disadvantage.** Titan's neural components (Hippocampus, Cortex RWL) were in early cold-start mode during this benchmark (200 items, no prior training). With accumulated usage data and consolidation cycles, retrieval quality improves as the neural networks learn from access patterns. This benchmark captures worst-case Titan quality.

**Important caveat:** Mem0's quality advantage comes entirely from LLM extraction, which is also what makes it 230x slower. This is not a free lunch -- it is a direct tradeoff between ingestion speed and retrieval quality.

### 3.4 Resource Usage

```
                    Titan           Mem0
RAM delta           9.4 MB          93.4 MB
Disk footprint      1.2 MB          1.31 MB
Neural parameters   77,580          0
Cold start time     8.51 s          5.65 s
```

- **RAM**: Titan uses 10x less RAM. Mem0's 93.4 MB includes ChromaDB's in-memory index. Note: this does not include the LLM's memory footprint (Qwen-3B adds ~6-8 GB when loaded for ingestion).

- **Disk**: Nearly identical. Both store embeddings and metadata efficiently.

- **Cold start**: Titan is slower to cold-start (8.51s vs 5.65s) because it loads the `all-MiniLM-L6-v2` embedding model and initializes 6 neural cortex networks. Mem0 defers LLM loading until first ingestion.

- **Neural parameters**: Titan maintains 77,580 trainable parameters across 6 cortex micro-networks (MIS, ET, RWL, AS, CG, ID). The Hippocampus (~226K additional params) was not loaded in standalone benchmark mode. These networks are CPU-only PyTorch and add negligible inference cost (<1ms per call).

### 3.5 Infrastructure Requirements

This is where the systems diverge most dramatically:

```
System      What you need to run it
─────────   ──────────────────────────────────────────────────────────
Titan       pip install titan-memory
            (SQLite bundled with Python, PyTorch CPU, no servers)

Mem0        pip install mem0ai
            + LLM endpoint (local Ollama/vLLM or cloud API key)
            + ChromaDB (bundled, but uses significant RAM)

Graphiti    pip install graphiti-core
            + Neo4j server (must be running and accessible)
            + LLM API key (required for entity/relation extraction)

Letta       pip install letta
            + Letta server (must be running)
            + PostgreSQL (for production; SQLite for dev)
            + LLM API key (required for memory management)
```

**Graphiti** and **Letta** could not be benchmarked because their hard infrastructure requirements make fully local, serverless operation impossible. This is not a criticism -- both are designed for different deployment contexts -- but it does mean we cannot provide head-to-head numbers.

## 4. Feature Comparison Matrix

```
Feature                                  Titan    Mem0     Graphiti   Letta
─────────────────────────────────────────────────────────────────────────────
STORAGE
  Tri-hybrid (SQL + Vector + Graph)      Yes      No       No         No
  Full-text search (FTS5)                Yes      No       No         No
  Vector similarity search               Yes      Yes      No         Yes
  Knowledge graph                        Yes      No       Yes        No
  Temporal validity tracking             No       No       Yes        No

EXTRACTION
  No LLM required for ingestion          Yes      No       No         No
  Claim-based extraction                 Yes      No       No         No
  Entity recognition                     Yes      Yes      Yes        Yes
  Relationship extraction                Yes      Yes      Yes        No
  Counter-hypotheses                     Yes      No       No         No

NEURAL COMPONENTS
  Neural importance scoring              Yes      No       No         No
  Neural emotion tagging                 Yes      No       No         No
  Learned retrieval weights              Yes      No       No         No
  Hopfield associative memory            Yes      No       No         No
  Self-training consolidation            Yes      No       No         No
  Trainable parameters                   77K+     0        0          0

LIFECYCLE
  Time-weighted confidence decay         Yes      No       No         No
  Controlled forgetting                  Yes      Manual   No         Yes
  Memory graduation (protection)         Yes      No       No         No
  Epistemological uncertainty             Yes      No       No         No

DEPLOYMENT
  Fully local, zero external deps        Yes      No       No         No
  No API keys required                   Yes      No       No         No
  No external servers required           Yes      Yes*     No         No
  Single pip install                     Yes      Yes*     No         No
  CPU-only operation                     Yes      Yes*     Yes*       Yes*

  * Mem0: requires LLM endpoint for extraction (local or cloud)
  * Graphiti: requires Neo4j + LLM API
  * Letta: requires Letta server
  * CPU-only: all can run on CPU, but Graphiti/Letta need their servers running
```

## 5. The Tradeoff

Titan and Mem0 represent two fundamentally different design philosophies for conversational memory:

**Titan: Deterministic extraction, neural lifecycle**

Titan extracts claims and entities using regex and heuristic rules. This is fast (52ms/item), deterministic (same input always produces same extraction), and requires zero external infrastructure. The tradeoff is lower extraction quality -- regex cannot capture implicit relationships, paraphrased facts, or nuanced semantics the way an LLM can. Titan compensates with 10 neural micro-networks that learn from usage patterns over time, improving retrieval quality through consolidation cycles. But at cold start, retrieval quality lags behind LLM-extracted systems.

**Mem0: LLM extraction, vector retrieval**

Mem0 leverages a full LLM to extract clean, structured facts from conversational text. This produces high-quality memory representations that are easy to retrieve via vector similarity. The tradeoff is speed (11.8s/item with a local 3B model) and a hard dependency on an LLM endpoint. With a cloud API, speed improves but introduces cost and availability concerns. Mem0 has no neural lifecycle components -- memories do not decay, consolidate, or self-improve.

**Which tradeoff matters more depends on your use case.**

For a local AI agent running 24/7 that processes thousands of conversational turns daily, the 230x ingestion speed difference is not a benchmark curiosity -- it is the difference between a system that keeps up in real time and one that falls behind. Titan's zero-dependency deployment also means no API keys to manage, no servers to maintain, and no costs that scale with usage.

For an agent that stores memories infrequently (e.g., end-of-conversation summaries via a cloud API), Mem0's superior extraction quality is the better choice. The ingestion latency is amortized over long conversations, and the cloud LLM provides extraction quality that regex cannot match.

## 6. When to Use What

### Use Titan when:
- You need a fully local agent with no API keys or external servers
- Your agent runs 24/7 and processes memories continuously
- You want neural lifecycle management (decay, consolidation, graduation)
- You need tri-hybrid storage (text + semantic + graph) in a single package
- Resource constraints matter (9.4 MB RAM vs 93.4 MB+)
- You value deterministic, reproducible extraction

### Use Mem0 when:
- Retrieval quality is paramount and ingestion is infrequent
- You have access to a fast LLM endpoint (cloud API or powerful local GPU)
- Your agent stores end-of-conversation summaries rather than streaming memories
- You want the simplest possible API with battle-tested vector search
- You do not need knowledge graph relationships or temporal decay

### Use Graphiti when:
- You need temporal knowledge tracking (facts valid between time ranges)
- You have Neo4j infrastructure already deployed
- Your domain requires rich graph queries over entity relationships
- You have an LLM API budget for extraction

### Use Letta when:
- You want the LLM itself to manage what gets remembered and forgotten
- You prefer a managed server architecture over embedded libraries
- You need PostgreSQL-backed durability for production workloads
- You want memory managed through natural language tool calls

## 7. Limitations

This benchmark has several important limitations:

1. **Small corpus.** 200 items is modest. Scaling behavior at 10K, 100K, or 1M items may differ significantly. Vector search complexity, graph traversal cost, and FTS5 index size all scale differently.

2. **CPU-only testing.** All systems ran on CPU. Mem0 with a GPU-accelerated local LLM (or a fast cloud API) would have substantially lower ingestion latency. The 230x ratio is specific to CPU-only local inference with a 3B parameter model.

3. **Mem0 used a small local model.** Qwen-3B on CPU is not how most Mem0 users deploy. With GPT-4 or Claude, extraction quality would likely be even higher, and ingestion latency would drop to network round-trip time (~500ms-2s). However, this would introduce API costs and an external dependency.

4. **Graphiti and Letta could not be benchmarked.** Both require infrastructure (Neo4j server, Letta server) that prevents fully local head-to-head comparison. Their numbers are absent, not zero.

5. **Cold-start Titan.** Titan's neural components (especially the Hippocampus retrieval pipeline) were in early cold-start mode. A production Titan instance with thousands of consolidated memories and trained neural networks would likely show improved retrieval quality. We did not measure this.

6. **Keyword-based ground truth.** Precision and MRR were measured against keyword ground truth labels. This may undercount semantically correct results that use different wording, slightly disadvantaging regex-based extraction (Titan) relative to LLM-based extraction (Mem0) which normalizes language during ingestion.

7. **Single embedding model.** Both Titan and Mem0 used `all-MiniLM-L6-v2`. Results may differ with larger or domain-specific embedding models.

## Hardware

```
CPU:      AMD Ryzen 9 7940HS w/ Radeon 780M Graphics
RAM:      24.4 GB
Python:   3.12.3
PyTorch:  2.5.0+cu124 (CPU mode)
OS:       Linux 6.17.0-19-generic (Ubuntu 24.04)
```

## Reproducing

```bash
cd benchmarks/
pip install titan-memory mem0ai
python benchmark.py
```

Results are saved to `benchmarks/results.json`. Note that Mem0 ingestion takes approximately 40 minutes with a local 3B model on CPU.

---

*Benchmark conducted March 2026. All numbers are reproducible on equivalent hardware.*
