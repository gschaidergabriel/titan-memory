# L.C.M.E. Quickstart: From Zero to Memory in 15 Minutes

Step-by-step guide to get L.C.M.E. running with a local Qwen-3B model on your machine. No cloud APIs, no GPU required.

## What You Will Have at the End

- A local Qwen-3B model running via Ollama
- L.C.M.E. giving your agent long-term memory
- A working chat loop where the agent remembers everything across sessions

---

## Step 1: Install Ollama

Ollama runs LLMs locally with a single command.

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**macOS:**
```bash
brew install ollama
```

**Windows:** Download from [ollama.com/download](https://ollama.com/download)

Verify it works:
```bash
ollama --version
```

---

## Step 2: Get a Hugging Face Token

You need this to download the abliterated (uncensored) Qwen model.

1. Go to [huggingface.co](https://huggingface.co) and create a free account
2. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. Click **New token**, name it anything, set permission to **Read**
4. Copy the token (starts with `hf_...`)

Set it in your terminal:
```bash
export HF_TOKEN=hf_your_token_here
```

> [!NOTE]
> If you use the standard (non-abliterated) Qwen model, you do not need a HF token. Skip to the `ollama pull` command in Option A.

---

## Step 3: Download Qwen 2.5 3B

**Option A — Standard Qwen (no HF token needed, easiest):**
```bash
ollama pull qwen2.5:3b
```
Done. Model name for later: `qwen2.5:3b`

**Option B — Abliterated Qwen (uncensored, needs HF token):**

First, install the Hugging Face CLI:
```bash
pip install huggingface-hub
```

Download the GGUF file (~1.9 GB):
```bash
huggingface-cli download bartowski/Qwen2.5-3B-Instruct-abliterated-GGUF \
  Qwen2.5-3B-Instruct-abliterated-Q4_K_M.gguf \
  --local-dir ./models \
  --token $HF_TOKEN
```

Create an Ollama model from it:
```bash
cat > Modelfile << 'EOF'
FROM ./models/Qwen2.5-3B-Instruct-abliterated-Q4_K_M.gguf
TEMPLATE """{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""
PARAMETER stop "<|im_end|>"
PARAMETER temperature 0.7
PARAMETER num_ctx 4096
EOF

ollama create qwen3b-abl -f Modelfile
```

Test it:
```bash
ollama run qwen3b-abl "Hello, who are you?"
```

Model name for later: `qwen3b-abl`

> [!TIP]
> Q4_K_M uses ~3-4 GB RAM at runtime. On a 16 GB machine you will have plenty of room left for L.C.M.E.

---

## Step 4: Install L.C.M.E.

```bash
pip install lcme
```

With HNSW support (recommended if you expect 500+ memories):
```bash
pip install lcme[hnsw]
```

Or from source:
```bash
git clone https://github.com/gschaidergabriel/lcme.git
cd lcme
pip install -e .
```

Verify:
```bash
python3 -c "from lcme import LCME; print('L.C.M.E. ready')"
```

First run downloads the embedding model (`all-MiniLM-L6-v2`, ~80 MB). This happens once.

---

## Step 5: Connect L.C.M.E. to Qwen

Create a file called `agent.py`:

```python
import re
import ollama
from lcme import LCME, LCMEConfig

# Initialize memory (persists in ./memory/ across restarts)
memory = LCME(LCMEConfig(data_dir="./memory"))

# Your model name from Step 3
MODEL = "qwen3b-abl"   # or "qwen2.5:3b" if you used Option A


def chat(user_message, history):
    # Retrieve relevant memories
    memory_block = memory.get_context_string(user_message)

    # System prompt with memory context
    system = (
        "You are a helpful assistant with long-term memory.\n"
        "You remember everything the user tells you across conversations.\n\n"
        + memory_block + "\n\n"
        "If the user tells you something worth remembering, include "
        "[STORE: <what to remember>] in your response."
    )

    # Call Qwen
    messages = [{"role": "system", "content": system}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    response = ollama.chat(model=MODEL, messages=messages)
    reply = response["message"]["content"]

    # Parse [STORE: ...] from response
    for text in re.findall(r'\[STORE:\s*(.+?)\]', reply):
        memory.ingest(text, origin="inference")

    # Store user message
    memory.ingest("User said: " + user_message, origin="user")

    # Update history
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": reply})
    return reply


def main():
    print("L.C.M.E. + " + MODEL)
    print("Memory: ./memory/  |  'quit' to exit  |  'stats' for memory info")
    print("-" * 60)

    history = []

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "stats":
            s = memory.get_stats()
            print("  Nodes:", s["nodes"], "| Edges:", s["edges"], "| Vectors:", s["vectors"])
            continue

        reply = chat(user_input, history)
        print("\nAssistant:", reply)


if __name__ == "__main__":
    main()
```

---

## Step 6: Run It

```bash
python3 agent.py
```

Example session:

```
You: My name is Alex and I'm building a weather app in Rust.
Assistant: Nice to meet you, Alex! [STORE: User is Alex, building a weather app in Rust]

You: What am I working on?
Assistant: You're building a weather app in Rust, Alex!

You: stats
  Nodes: 8 | Edges: 3 | Vectors: 4

You: quit
```

Close the script, restart it, and ask again:

```
You: What's my name?
Assistant: Your name is Alex! You're working on a weather app in Rust.
```

The memory persists in `./memory/`. It survives restarts, reboots, and updates.

---

## Step 7: What Happens Under the Hood

When you type a message:

1. **L.C.M.E. retrieves** relevant memories for your query (12ms, keyword + semantic + graph fusion)
2. The memories are injected into the **system prompt** so Qwen sees them as context
3. Qwen responds, and if it includes `[STORE: ...]` directives, those get ingested
4. Your original message is always stored as a `user` origin memory
5. Next time you ask something related, the memories come back

When you restart:

- SQLite database, vector embeddings, and knowledge graph are all on disk
- L.C.M.E. loads them instantly on startup
- No retraining, no reindexing, no cold start penalty

---

## Step 8: Customize

**Change the model** — edit the `MODEL` variable in `agent.py`:
```python
MODEL = "llama3.1:8b"      # Larger, smarter, needs ~6 GB RAM
MODEL = "phi3:3.8b"         # Microsoft, good for reasoning
MODEL = "gemma2:2b"         # Google, smallest, ~2 GB RAM
MODEL = "mistral:7b"        # Mistral, good all-rounder
```

**Change memory location:**
```python
memory = LCME(LCMEConfig(data_dir="/path/to/your/memory"))
```

**Enable automatic consolidation** (trains the neural networks periodically):
```python
memory = LCME(LCMEConfig(
    data_dir="./memory",
    auto_consolidation=True,
    consolidation_interval_hours=6.0,
))
```

**Use the convenience API** (global singleton, data in `~/.lcme/data/`):
```python
from lcme import remember, recall

remember("The API runs on port 8080.")
print(recall("What port?"))
```

---

## Troubleshooting

**"ollama: command not found"**
Ollama is not installed or not in PATH. Restart your terminal after installing.

**"Error: model not found"**
Run `ollama list` to see available models. Make sure the model name in `agent.py` matches.

**"No module named lcme"**
Run `pip install lcme` again. Make sure you're using the same Python that runs `agent.py`.

**First run is slow (~8 seconds)**
That's the embedding model loading (one-time). After that, ingest is ~100ms and retrieval is ~12ms.

**Memory not persisting**
Check that `./memory/` directory exists and contains `lcme.db`. If you're running from a different directory, the path is relative to where you run `python3 agent.py`.

---

## What's Next

- Read the [full API docs](../README.md#api) for all methods
- See [integration examples](INTEGRATIONS.md) for LangChain, LlamaIndex, llama-server
- Run `memory.get_stats()` to see how many memories you have
- Run `memory.consolidate()` manually to train the neural networks from your usage data