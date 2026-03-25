#!/usr/bin/env python3
"""Extended Titan Benchmark: nDCG, Recall@K, Forgetting, Contradiction, Concurrent, Memory Growth"""
import gc, json, math, os, random, shutil, statistics, sys, tempfile, threading, time
from pathlib import Path

sys.path.insert(0, '/home/ai-core-node/titan-memory')
random.seed(42)

# ── Data Generation ─────────────────────────────────────────
PEOPLE = ["Alice","Bob","Carlos","Diana","Elena","Frank","Grace","Hiro",
          "Isabel","James","Kenji","Lena","Marco","Nina","Oscar","Petra",
          "Quinn","Rosa","Stefan","Tanya","Umar","Vera","Wei","Xena","Yuki","Zara"]
COMPANIES = ["Google","TechStart","DataFlow","CloudNine","NeuralWorks","Stripe",
             "Vercel","Supabase","Prisma","Railway"]
CITIES = ["Berlin","Tokyo","Austin","London","Toronto","Lisbon","Seoul","Amsterdam"]
TECHS = ["Python","Rust","Go","TypeScript","React","PostgreSQL","MongoDB","Redis",
         "Kafka","Kubernetes","Docker","Terraform","Prometheus","Grafana","FastAPI"]
TOPICS = ["security","performance","scalability","testing","deployment","CI/CD","architecture"]

templates = [
    lambda: f"{random.choice(PEOPLE)} works as a {random.choice(['senior engineer','tech lead','architect','data scientist','DevOps engineer','product manager'])} at {random.choice(COMPANIES)} in {random.choice(CITIES)}.",
    lambda: f"{random.choice(PEOPLE)} mentioned that {random.choice(TOPICS)} is their top priority for Q{random.randint(1,4)}.",
    lambda: f"The {random.choice(['auth','payment','notification','search','analytics','gateway'])} service runs on port {random.randint(3000,9999)}.",
    lambda: f"We migrated from {random.choice(TECHS)} to {random.choice(TECHS)} because of {random.choice(['performance issues','licensing costs','team preference'])}.",
    lambda: f"{random.choice(PEOPLE)} prefers {random.choice(TECHS)} over {random.choice(TECHS)} for {random.choice(['backend','frontend','infrastructure'])}.",
    lambda: f"{random.choice(PEOPLE)} lives in {random.choice(CITIES)} and works remotely for {random.choice(COMPANIES)}.",
    lambda: f"Our {random.choice(TECHS)} cluster uses {random.randint(2,50)} pods across {random.randint(1,3)} AWS regions.",
    lambda: f"{random.choice(PEOPLE)} built a {random.choice(['monitoring dashboard','deployment pipeline','testing framework','CLI tool'])} that improved {random.choice(['build time','coverage','error rate'])} by {random.randint(10,60)}%.",
    lambda: f"We use {random.choice(TECHS)} for {random.choice(TOPICS)} in {random.choice(['production','staging'])}.",
    lambda: f"The {random.choice(['observability','infrastructure','CI/CD'])} stack costs {random.choice([500,1000,2000,5000])} USD per month.",
]
data_1200 = [random.choice(templates)() for _ in range(1200)]

# Queries with multi-keyword ground truth (for Recall: ANY keyword = relevant)
QUERIES = [
    ("Where does Alice work?", ["alice"]),
    ("What port does the auth service use?", ["auth", "port"]),
    ("Who lives in Berlin?", ["berlin"]),
    ("Who prefers Python?", ["python", "prefer"]),
    ("What monitoring tools?", ["prometheus", "grafana", "monitoring"]),
    ("What about security?", ["security"]),
    ("What runs on Kubernetes?", ["kubernetes", "cluster", "pods"]),
    ("Who built a custom tool?", ["built", "tool", "dashboard", "pipeline"]),
    ("What costs money?", ["cost", "usd", "month"]),
    ("What migration happened?", ["migrat"]),
    ("Who is a tech lead?", ["tech lead"]),
    ("Who works at Google?", ["google"]),
    ("Who works remotely?", ["remote"]),
    ("What do we use for deployment?", ["deployment", "ci", "cd"]),
    ("Who works in Tokyo?", ["tokyo"]),
]

def rss_mb():
    try:
        with open(f'/proc/{os.getpid()}/status') as f:
            for l in f:
                if l.startswith('VmRSS:'): return int(l.split()[1])/1024
    except: return 0

def disk_mb(p):
    t=0
    for f in Path(p).rglob('*'):
        if f.is_file(): t+=f.stat().st_size
    return t/1e6

def fresh_titan(data_dir):
    for mod in list(sys.modules.keys()):
        if 'titan' in mod: del sys.modules[mod]
    from titan import Titan, TitanConfig
    from titan.core import reset_titan
    reset_titan()
    return Titan(TitanConfig(data_dir=data_dir))

def is_relevant(text, keywords):
    t = text.lower()
    return any(k.lower() in t for k in keywords)

# Count how many items in the corpus are relevant per query
def count_relevant(items_texts, keywords):
    return sum(1 for t in items_texts if is_relevant(t, keywords))

print("="*65)
print("  TITAN EXTENDED BENCHMARK")
print("="*65)

ALL = {}

# ══════════════════════════════════════════════════════════════
# 1. RECALL@K (completeness — how many relevant items found?)
# ══════════════════════════════════════════════════════════════
print("\n── 1. RECALL@K ──")
d = tempfile.mkdtemp()
t = fresh_titan(os.path.join(d, 'recall'))
for text in data_1200[:500]: t.ingest(text, origin="user")

recall_at = {1:[], 3:[], 5:[], 10:[]}
for q, kws in QUERIES:
    items10 = t.retrieve(q, limit=10)
    texts10 = [str(i.get('content',i.get('label',''))) for i in items10]
    # Count total relevant in corpus
    all_texts = [data_1200[i] for i in range(500)]
    total_rel = count_relevant(all_texts, kws)
    if total_rel == 0: continue
    for k in [1,3,5,10]:
        found = sum(1 for txt in texts10[:k] if is_relevant(txt, kws))
        recall_at[k].append(found / min(total_rel, k))

for k in [1,3,5,10]:
    v = statistics.mean(recall_at[k]) if recall_at[k] else 0
    ALL[f"recall_at_{k}"] = round(v, 3)
    print(f"  Recall@{k:<3} = {v:.3f}")

from titan.core import reset_titan; reset_titan()
shutil.rmtree(d, ignore_errors=True); gc.collect()

# ══════════════════════════════════════════════════════════════
# 2. nDCG@5 (ranking quality — are the best results on top?)
# ══════════════════════════════════════════════════════════════
print("\n── 2. nDCG@5 ──")
d = tempfile.mkdtemp()
t = fresh_titan(os.path.join(d, 'ndcg'))
for text in data_1200[:500]: t.ingest(text, origin="user")

ndcg_scores = []
for q, kws in QUERIES:
    items = t.retrieve(q, limit=5)
    # Binary relevance: 1 if matches keywords, 0 otherwise
    rels = [1.0 if is_relevant(str(i.get('content',i.get('label',''))), kws) else 0.0 for i in items]
    # DCG
    dcg = sum(r / math.log2(i+2) for i, r in enumerate(rels))
    # Ideal DCG (all relevant items first)
    ideal = sorted(rels, reverse=True)
    idcg = sum(r / math.log2(i+2) for i, r in enumerate(ideal))
    ndcg = dcg / idcg if idcg > 0 else 0.0
    ndcg_scores.append(ndcg)

avg_ndcg = statistics.mean(ndcg_scores)
ALL["ndcg_at_5"] = round(avg_ndcg, 3)
print(f"  nDCG@5 = {avg_ndcg:.3f}")

reset_titan(); shutil.rmtree(d, ignore_errors=True); gc.collect()

# ══════════════════════════════════════════════════════════════
# 3. MEMORY GROWTH (RAM + Disk vs item count)
# ══════════════════════════════════════════════════════════════
print("\n── 3. MEMORY GROWTH ──")
d = tempfile.mkdtemp()
t = fresh_titan(os.path.join(d, 'growth'))
growth = {}
rss0 = rss_mb()
for i, text in enumerate(data_1200[:1000]):
    t.ingest(text, origin="user")
    if (i+1) in [50, 100, 200, 500, 1000]:
        rss_now = rss_mb()
        disk = disk_mb(os.path.join(d, 'growth'))
        growth[i+1] = {"ram_mb": round(rss_now - rss0, 1), "disk_mb": round(disk, 2)}
        print(f"  {i+1:>5} items: RAM +{growth[i+1]['ram_mb']}MB, Disk {growth[i+1]['disk_mb']}MB")
ALL["memory_growth"] = growth
reset_titan(); shutil.rmtree(d, ignore_errors=True); gc.collect()

# ══════════════════════════════════════════════════════════════
# 4. WRITE THROUGHPUT (sustained ingest rate)
# ══════════════════════════════════════════════════════════════
print("\n── 4. WRITE THROUGHPUT ──")
d = tempfile.mkdtemp()
t = fresh_titan(os.path.join(d, 'write'))
# Warmup
t.ingest("warmup", origin="user"); t.retrieve("warmup")

batch_times = []
for batch_start in range(0, 500, 50):
    t0 = time.monotonic()
    for text in data_1200[batch_start:batch_start+50]:
        t.ingest(text, origin="user")
    batch_times.append((time.monotonic()-t0)/50*1000)

ALL["write_throughput"] = {
    "avg_ms_per_item": round(statistics.mean(batch_times), 1),
    "p50_ms": round(sorted(batch_times)[len(batch_times)//2], 1),
    "p95_ms": round(sorted(batch_times)[int(len(batch_times)*0.95)], 1),
    "items_per_sec": round(1000/statistics.mean(batch_times), 1),
}
print(f"  Avg: {ALL['write_throughput']['avg_ms_per_item']}ms/item")
print(f"  P50: {ALL['write_throughput']['p50_ms']}ms  P95: {ALL['write_throughput']['p95_ms']}ms")
print(f"  Throughput: {ALL['write_throughput']['items_per_sec']} items/sec")
reset_titan(); shutil.rmtree(d, ignore_errors=True); gc.collect()

# ══════════════════════════════════════════════════════════════
# 5. CONCURRENT ACCESS (agent reads while ingesting)
# ══════════════════════════════════════════════════════════════
print("\n── 5. CONCURRENT ACCESS ──")
d = tempfile.mkdtemp()
t = fresh_titan(os.path.join(d, 'conc'))
for text in data_1200[:200]: t.ingest(text, origin="user")

write_errors, read_errors = [], []
write_times, read_times = [], []

def writer():
    for i in range(100):
        try:
            t0 = time.monotonic()
            t.ingest(f"Concurrent write #{i}: {random.choice(data_1200)}", origin="observation")
            write_times.append((time.monotonic()-t0)*1000)
        except Exception as e:
            write_errors.append(str(e))

def reader():
    for q, _ in QUERIES * 3:
        try:
            t0 = time.monotonic()
            t.retrieve(q, limit=3)
            read_times.append((time.monotonic()-t0)*1000)
        except Exception as e:
            read_errors.append(str(e))

threads = [threading.Thread(target=writer),
           threading.Thread(target=reader),
           threading.Thread(target=reader)]
t0_all = time.monotonic()
for th in threads: th.start()
for th in threads: th.join()
total_ms = (time.monotonic()-t0_all)*1000

ALL["concurrent"] = {
    "total_ms": round(total_ms),
    "write_errors": len(write_errors),
    "read_errors": len(read_errors),
    "write_ops": len(write_times),
    "read_ops": len(read_times),
    "write_avg_ms": round(statistics.mean(write_times), 1) if write_times else 0,
    "read_avg_ms": round(statistics.mean(read_times), 1) if read_times else 0,
}
print(f"  1 writer (100 ops) + 2 readers ({len(read_times)} ops)")
print(f"  Total: {round(total_ms)}ms")
print(f"  Write errors: {len(write_errors)}, Read errors: {len(read_errors)}")
print(f"  Write avg: {ALL['concurrent']['write_avg_ms']}ms, Read avg: {ALL['concurrent']['read_avg_ms']}ms")
reset_titan(); shutil.rmtree(d, ignore_errors=True); gc.collect()

# ══════════════════════════════════════════════════════════════
# 6. CONTRADICTION BENCHMARK (10 contradiction pairs)
# ══════════════════════════════════════════════════════════════
print("\n── 6. CONTRADICTION BENCHMARK ──")
contradictions = [
    ("Alice works at Google as a senior engineer.",
     "Alice just left Google and joined Stripe as CTO.",
     "Where does Alice work?", "stripe"),
    ("The auth service runs on port 8080.",
     "We moved the auth service to port 9090 after the migration.",
     "What port does auth use?", "9090"),
    ("Bob lives in Tokyo and works for TechStart.",
     "Bob relocated from Tokyo to Berlin last month.",
     "Where does Bob live?", "berlin"),
    ("Our primary database is MySQL.",
     "We completed the migration from MySQL to PostgreSQL.",
     "What database do we use?", "postgresql"),
    ("The project deadline is June 2026.",
     "The deadline was pushed back from June to September 2026.",
     "What is the project deadline?", "september"),
    ("We use Jenkins for CI/CD.",
     "We replaced Jenkins with GitHub Actions for all pipelines.",
     "What CI system do we use?", "github"),
    ("Carlos is the tech lead for the search team.",
     "Carlos moved to the infrastructure team, Diana now leads search.",
     "Who leads the search team?", "diana"),
    ("The API rate limit is 100 requests per minute.",
     "We increased the API rate limit to 500 requests per minute.",
     "What is the API rate limit?", "500"),
    ("We use Datadog for monitoring.",
     "We switched from Datadog to Prometheus and Grafana to cut costs.",
     "What monitoring tools do we use?", "prometheus"),
    ("The staging environment runs on 2 nodes.",
     "We scaled staging from 2 to 8 nodes after the load test failure.",
     "How many staging nodes?", "8"),
]

d = tempfile.mkdtemp()
t = fresh_titan(os.path.join(d, 'contra'))
# Add some background noise
for text in data_1200[:100]: t.ingest(text, origin="user")

hits = 0
for old, new, query, expected in contradictions:
    t.ingest(old, origin="user")
    t.ingest(new, origin="user")
    results = t.retrieve(query, limit=3)
    found = any(expected.lower() in str(r.get('content','')).lower() for r in results)
    if found: hits += 1
    status = "PASS" if found else "FAIL"
    top = str(results[0].get('content',''))[:60] if results else "none"
    print(f"  {status}  {query:35} → {top}")

ALL["contradiction"] = {"total": len(contradictions), "passed": hits,
                         "rate": round(hits/len(contradictions), 2)}
print(f"  Score: {hits}/{len(contradictions)} ({hits/len(contradictions)*100:.0f}%)")
reset_titan(); shutil.rmtree(d, ignore_errors=True); gc.collect()

# ══════════════════════════════════════════════════════════════
# 7. FORGETTING EVALUATION (does maintenance prune correctly?)
# ══════════════════════════════════════════════════════════════
print("\n── 7. FORGETTING EVALUATION ──")
d = tempfile.mkdtemp()
t = fresh_titan(os.path.join(d, 'forget'))

# Ingest 100 items
for text in data_1200[:100]: t.ingest(text, origin="user")
stats_before = t.get_stats()

# Run maintenance
maint_result = t.run_maintenance()
stats_after = t.get_stats()

# Manually forget a node
nodes_conn = t.sqlite._get_conn()
unprotected = nodes_conn.execute(
    "SELECT id FROM nodes WHERE protected = 0 AND type = 'memory' LIMIT 1"
).fetchone()
forget_ok = False
if unprotected:
    forget_ok = t.forget(unprotected[0])

# Protect a node
protected_node = nodes_conn.execute(
    "SELECT id FROM nodes WHERE type = 'memory' LIMIT 1"
).fetchone()
protect_ok = False
if protected_node:
    protect_ok = t.protect(protected_node[0])
    # Try to forget protected node — should fail
    forget_protected = t.forget(protected_node[0])
    protect_works = not forget_protected
else:
    protect_works = False

ALL["forgetting"] = {
    "nodes_before_maintenance": stats_before["nodes"],
    "nodes_after_maintenance": stats_after["nodes"],
    "manual_forget_works": forget_ok,
    "protection_prevents_forget": protect_works,
    "maintenance_result": maint_result,
}
print(f"  Nodes before maintenance: {stats_before['nodes']}")
print(f"  Nodes after maintenance: {stats_after['nodes']}")
print(f"  Manual forget: {'PASS' if forget_ok else 'FAIL'}")
print(f"  Protection blocks forget: {'PASS' if protect_works else 'FAIL'}")
reset_titan(); shutil.rmtree(d, ignore_errors=True); gc.collect()

# ══════════════════════════════════════════════════════════════
# 8. LONG CONVERSATION SIMULATION
# ══════════════════════════════════════════════════════════════
print("\n── 8. LONG CONVERSATION (200 turns) ──")
d = tempfile.mkdtemp()
t = fresh_titan(os.path.join(d, 'conv'))

# Simulate 200 conversation turns
conv_data = [
    "My name is Alex and I'm building a weather app.",
    "I'm using Python with FastAPI for the backend.",
    "The frontend is React with TypeScript.",
    "I deployed it on Railway with PostgreSQL.",
    "I prefer dark mode for all my tools.",
    "My cofounder Sarah handles the design work.",
    "We're based in Copenhagen.",
    "Our launch date is April 15, 2026.",
    "We use Stripe for payments.",
    "The API runs on port 8443 with HTTPS.",
]
# Pad to 200 turns with varied repetitions and new info
for i in range(200):
    text = random.choice(conv_data) if i < 10 else f"Turn {i}: {random.choice(data_1200[:50])}"
    t.ingest(text, origin="user" if i % 3 == 0 else "observation")

# Query early facts after 200 turns
conv_queries = [
    ("What is my name?", ["alex"]),
    ("What am I building?", ["weather"]),
    ("What backend framework?", ["fastapi"]),
    ("Where are we based?", ["copenhagen"]),
    ("What is the launch date?", ["april"]),
    ("Who handles design?", ["sarah"]),
    ("What port does the API use?", ["8443"]),
    ("What payment system?", ["stripe"]),
]

h1,h5,mrr=0,0,0
for q, kws in conv_queries:
    items = t.retrieve(q, limit=5)
    for rank, item in enumerate(items):
        txt = str(item.get('content',item.get('label',''))).lower()
        if any(k.lower() in txt for k in kws):
            if rank==0: h1+=1
            h5+=1; mrr+=1/(rank+1); break

nq = len(conv_queries)
ALL["long_conversation"] = {
    "turns": 200,
    "early_fact_queries": nq,
    "precision_at_1": round(h1/nq, 3),
    "precision_at_5": round(h5/nq, 3),
    "mrr": round(mrr/nq, 3),
}
print(f"  200 turns, querying facts from first 10 turns:")
print(f"  P@1={h1/nq:.3f}  P@5={h5/nq:.3f}  MRR={mrr/nq:.3f}")
reset_titan(); shutil.rmtree(d, ignore_errors=True)

# ══════════════════════════════════════════════════════════════
# SAVE + SUMMARY
# ══════════════════════════════════════════════════════════════
with open('/home/ai-core-node/titan-memory/benchmarks/extended_results.json', 'w') as f:
    json.dump(ALL, f, indent=2, default=str)

print(f"\n{'='*65}")
print("  SUMMARY")
print("="*65)
print(f"  Recall@1:    {ALL['recall_at_1']}")
print(f"  Recall@5:    {ALL['recall_at_5']}")
print(f"  Recall@10:   {ALL['recall_at_10']}")
print(f"  nDCG@5:      {ALL['ndcg_at_5']}")
print(f"  Write:       {ALL['write_throughput']['avg_ms_per_item']}ms/item ({ALL['write_throughput']['items_per_sec']} items/sec)")
print(f"  Concurrent:  {ALL['concurrent']['write_errors']+ALL['concurrent']['read_errors']} errors in {ALL['concurrent']['total_ms']}ms")
print(f"  Contradict:  {ALL['contradiction']['passed']}/{ALL['contradiction']['total']} ({ALL['contradiction']['rate']*100:.0f}%)")
print(f"  Forget:      manual={'OK' if ALL['forgetting']['manual_forget_works'] else 'FAIL'}, protect={'OK' if ALL['forgetting']['protection_prevents_forget'] else 'FAIL'}")
print(f"  Long Conv:   P@5={ALL['long_conversation']['precision_at_5']} after {ALL['long_conversation']['turns']} turns")
mg = ALL['memory_growth']
print(f"  Growth:      50→{mg.get(50,{}).get('ram_mb','?')}MB, 1000→{mg.get(1000,{}).get('ram_mb','?')}MB RAM")
print()
