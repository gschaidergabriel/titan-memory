# Titan Memory -- Framework Integration Guide

Titan is framework-agnostic. It is a Python library with a simple API. This guide shows how to wire it into every major AI agent framework.

## Table of Contents

- [Universal Pattern](#universal-pattern)
- [Claude API (Anthropic SDK)](#claude-api-anthropic-sdk)
- [Ollama + Python](#ollama--python)
- [OpenAI-Compatible APIs](#openai-compatible-apis)
- [LangChain](#langchain)
- [LlamaIndex](#llamaindex)
- [CrewAI](#crewai)
- [AutoGen v0.4](#autogen-v04)
- [Pydantic AI](#pydantic-ai)
- [Open Interpreter](#open-interpreter)
- [Summary Table](#summary-table)

---

## Universal Pattern

There are two ways to give an LLM access to Titan memory. Most integrations use both.

### Pattern A: System Prompt Augmentation (Passive Recall)

Before each LLM call, retrieve relevant memories and inject them into the system prompt. The LLM sees the context but does not control when retrieval happens.

```python
from titan import Titan, TitanConfig

memory = Titan(TitanConfig(data_dir="./memory"))

# Before each LLM call:
context = memory.get_context_string(user_message)
system_prompt = f"You are a helpful assistant.\n\n{context}"
# Pass system_prompt to your LLM
```

**Pros:** Works with every model, no tool-calling support needed, simple.
**Cons:** LLM cannot decide when to store or recall; you control the timing.

### Pattern B: Tool-Based (Active Store/Recall)

Give the LLM tools to store and retrieve memories. The LLM decides when to use them.

```python
# Define tools: memory_store, memory_recall, memory_forget
# LLM calls them via its tool-calling protocol
# You execute them and return results
```

**Pros:** LLM decides what is worth remembering, can search on demand.
**Cons:** Requires tool-calling support, adds latency from tool round-trips.

### Best Practice: Both

Use system prompt augmentation for passive context on every turn, plus tools for when the LLM wants to actively store or search. This is the pattern used in most examples below.

---

## Claude API (Anthropic SDK)

**Install:**
```bash
pip install anthropic titan-memory
```

**How it works:** Claude's `tool_use` protocol lets you define memory tools in the `tools` parameter of `messages.create()`. Claude responds with `tool_use` blocks when it wants to call a tool. You execute the tool locally, send back a `tool_result` block, and loop until Claude responds with `stop_reason="end_turn"`.

Memory injection happens in two places:
- System prompt augmentation: inject `[MEMORY]` block before each turn (passive)
- Tool definitions: Claude calls `memory_store` / `memory_recall` / `memory_forget` (active)

**Full example:** [`examples/claude_api.py`](../examples/claude_api.py)

```python
import anthropic
from titan import Titan, TitanConfig

client = anthropic.Anthropic()  # Uses ANTHROPIC_API_KEY env var
memory = Titan(TitanConfig(data_dir="./claude_memory"))

# Define Titan memory tools for Claude
TITAN_TOOLS = [
    {
        "name": "memory_store",
        "description": (
            "Store important information in long-term memory. Use when the user "
            "shares facts, preferences, project details, or decisions worth "
            "remembering across conversations."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The information to remember. Be specific and self-contained."
                },
                "origin": {
                    "type": "string",
                    "enum": ["user", "inference", "observation"],
                    "description": "Source: 'user' for stated facts, 'inference' for deductions."
                }
            },
            "required": ["text"]
        }
    },
    {
        "name": "memory_recall",
        "description": (
            "Search long-term memory for relevant information. Use when you need "
            "to remember something from a previous conversation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search for. Use natural language, not keywords."
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results (default 5).",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "memory_forget",
        "description": (
            "Remove a specific memory. Use when the user asks you to forget "
            "something or information is confirmed wrong."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "node_id": {
                    "type": "string",
                    "description": "The ID of the memory node to forget."
                }
            },
            "required": ["node_id"]
        }
    }
]

def execute_tool(name: str, input_data: dict) -> str:
    if name == "memory_store":
        result = memory.ingest(
            input_data["text"],
            origin=input_data.get("origin", "user")
        )
        return f"Stored. Event {result['event_id']}: {result['claims']} claims, entities: {result['entities']}"
    elif name == "memory_recall":
        results = memory.retrieve(input_data["query"], limit=input_data.get("limit", 5))
        if not results:
            return "No relevant memories found."
        lines = []
        for r in results:
            lines.append(
                f"[{r.get('timestamp', '?')}] (conf={r.get('confidence', '?')}, "
                f"id={r.get('node_id', '?')}) {r.get('text', r.get('content', ''))}"
            )
        return "\n".join(lines)
    elif name == "memory_forget":
        ok = memory.forget(input_data["node_id"])
        return f"{'Forgotten' if ok else 'Failed (protected or not found)'}."
    return f"Unknown tool: {name}"

def chat(user_message: str, conversation: list = None) -> str:
    if conversation is None:
        conversation = []

    # Passive memory: inject into system prompt
    memory_context = memory.get_context_string(user_message)
    system_prompt = (
        "You are a helpful assistant with long-term memory.\n"
        "Proactively use memory_store when the user shares important facts.\n"
        "Use memory_recall when you need to check what you know.\n\n"
        f"{memory_context}"
    )

    conversation.append({"role": "user", "content": user_message})

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system=system_prompt,
            tools=TITAN_TOOLS,
            messages=conversation
        )

        conversation.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            return "\n".join(
                block.text for block in response.content if block.type == "text"
            )

        # Process tool calls
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result_str = execute_tool(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_str
                })

        conversation.append({"role": "user", "content": tool_results})
```

**Gotchas:**
- `tool_result` must reference the exact `tool_use_id` from Claude's response.
- You must include the full assistant message (with `tool_use` blocks) when sending `tool_result` back, or Claude loses context.
- `max_tokens` applies to Claude's response, not the full conversation.
- `stop_reason` is `"tool_use"` when Claude wants to call a tool and is NOT done.

---

## Ollama + Python

**Install:**
```bash
pip install ollama titan-memory
```

**How it works:** You build the messages list manually and call `ollama.chat()`. No built-in agent loop -- you implement it yourself. Ollama supports the `tools` parameter (OpenAI format) for models that support function calling (llama3, qwen2.5, mistral, etc.).

### Pattern A: System Prompt Augmentation (Simple, Reliable)

Works with every model. No tool-calling support needed.

**Full example:** [`examples/ollama_chat.py`](../examples/ollama_chat.py)

```python
import re
import ollama
from titan import Titan, TitanConfig

titan = Titan(TitanConfig(data_dir="./ollama_memory"))

def chat_with_memory(user_message: str, messages: list = None) -> str:
    if messages is None:
        messages = []

    memory_block = titan.get_context_string(user_message)

    system_msg = (
        "You are a helpful assistant with long-term memory.\n\n"
        f"{memory_block}\n\n"
        "If the user tells you something worth remembering, say "
        "'[STORE: <what to remember>]' in your response."
    )

    full_messages = [{"role": "system", "content": system_msg}]
    full_messages.extend(messages)
    full_messages.append({"role": "user", "content": user_message})

    response = ollama.chat(model="llama3.1:8b", messages=full_messages)
    reply = response["message"]["content"]

    # Parse [STORE: ...] directives from response
    stores = re.findall(r'\[STORE:\s*(.+?)\]', reply)
    for text in stores:
        titan.ingest(text, origin="inference")

    # Always store the conversation turn
    titan.ingest(f"User: {user_message}", origin="user")

    messages.append({"role": "user", "content": user_message})
    messages.append({"role": "assistant", "content": reply})
    return reply
```

### Pattern B: Tool-Based (Requires Tool-Capable Model)

```python
import ollama
from titan import Titan, TitanConfig

titan = Titan(TitanConfig(data_dir="./ollama_memory"))

def chat_with_tools(user_message: str) -> str:
    memory_block = titan.get_context_string(user_message)

    messages = [
        {"role": "system", "content": f"You have long-term memory.\n\n{memory_block}"},
        {"role": "user", "content": user_message},
    ]

    tools = [
        {
            "type": "function",
            "function": {
                "name": "memory_store",
                "description": "Store information in long-term memory",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "What to remember"},
                    },
                    "required": ["text"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "memory_recall",
                "description": "Search long-term memory",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "What to search for"},
                    },
                    "required": ["query"]
                }
            }
        }
    ]

    while True:
        response = ollama.chat(model="qwen2.5:7b", messages=messages, tools=tools)
        msg = response["message"]
        messages.append(msg)

        if not msg.get("tool_calls"):
            return msg["content"]

        for tc in msg["tool_calls"]:
            fn_name = tc["function"]["name"]
            fn_args = tc["function"]["arguments"]

            if fn_name == "memory_store":
                result = titan.ingest(fn_args["text"], origin="user")
                result_str = f"Stored: {result['claims']} claims"
            elif fn_name == "memory_recall":
                result_str = titan.get_context_string(fn_args["query"])
            else:
                result_str = f"Unknown tool: {fn_name}"

            messages.append({"role": "tool", "content": result_str})
```

**Gotchas:**
- Ollama runs models locally -- the model must support tool calling for Pattern B.
- Context window is model-dependent (4K-128K). System message + memory context eats into it.
- No conversation persistence -- you manage the `messages` list.
- Ollama's tool calling is less reliable than Claude/GPT for complex tool schemas.
- `ollama.chat()` is synchronous; use `ollama.AsyncClient` for async.

---

## OpenAI-Compatible APIs

**Install:**
```bash
pip install openai titan-memory
```

**How it works:** Uses the OpenAI Python SDK pointed at any compatible endpoint (vLLM, LocalAI, LiteLLM, Groq, Together, OpenAI itself). Tool calls are returned in `response.choices[0].message.tool_calls`. You execute tools and send results back as `tool` role messages.

**Full example:** [`examples/openai_compatible.py`](../examples/openai_compatible.py)

```python
import json
from openai import OpenAI
from titan import Titan, TitanConfig

titan = Titan(TitanConfig(data_dir="./oai_memory"))

# Point at any OpenAI-compatible endpoint
client = OpenAI(
    base_url="http://localhost:8000/v1",  # vLLM, LocalAI, etc.
    api_key="not-needed-for-local",
)
# For actual OpenAI: client = OpenAI()  (uses OPENAI_API_KEY)
# For Groq: client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_KEY)
# For Together: client = OpenAI(base_url="https://api.together.xyz/v1", api_key=TOGETHER_KEY)

tools = [
    {
        "type": "function",
        "function": {
            "name": "memory_store",
            "description": "Store information in long-term memory for future reference.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "What to remember"},
                    "origin": {
                        "type": "string",
                        "enum": ["user", "inference", "observation"],
                        "description": "Source type"
                    }
                },
                "required": ["text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "memory_recall",
            "description": "Search long-term memory for relevant stored information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer", "description": "Max results", "default": 5}
                },
                "required": ["query"]
            }
        }
    }
]

def execute_tool_call(tool_call) -> str:
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)

    if name == "memory_store":
        result = titan.ingest(args["text"], origin=args.get("origin", "user"))
        return f"Stored: {result['claims']} claims, {result['entities']} entities"
    elif name == "memory_recall":
        results = titan.retrieve(args["query"], limit=args.get("limit", 5))
        if not results:
            return "No memories found."
        return "\n".join(
            f"[conf={r.get('confidence', '?')}] {r.get('text', r.get('content', ''))}"
            for r in results
        )
    return f"Unknown tool: {name}"

def chat(user_message: str, messages: list = None) -> str:
    if messages is None:
        messages = []

    memory_context = titan.get_context_string(user_message)
    system = {
        "role": "system",
        "content": f"You are a helpful assistant with long-term memory.\n\n{memory_context}"
    }

    messages.append({"role": "user", "content": user_message})

    while True:
        response = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct",  # Or whatever is served
            messages=[system] + messages,
            tools=tools,
            tool_choice="auto",
        )

        msg = response.choices[0].message
        messages.append(msg.model_dump())

        if not msg.tool_calls:
            titan.ingest(f"User: {user_message}", origin="user")
            return msg.content

        for tc in msg.tool_calls:
            result = execute_tool_call(tc)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })
```

**Gotchas:**
- Not all OpenAI-compatible servers support the `tools` parameter (vLLM does, some do not). Fall back to system prompt augmentation if tools are unsupported.
- Tool calling quality varies dramatically by model. For local models, system prompt augmentation is often more reliable.
- Response format is identical to OpenAI, but some servers have quirks with `tool_call_id` handling.
- `tool_choice="auto"` lets the model decide; `"required"` forces a tool call.

---

## LangChain

**Install:**
```bash
pip install langchain langchain-anthropic titan-memory
```

**How it works:** LangChain calls `memory.load_memory_variables()` before each LLM call to inject context, and `memory.save_context()` after each LLM call to store the turn. For agent patterns, tools are called in a loop until the agent produces a final answer.

Three integration options are shown below. Use whichever fits your LangChain usage pattern.

**Full example:** [`examples/langchain_memory.py`](../examples/langchain_memory.py)

### Option A: BaseMemory Subclass (Legacy ConversationChain)

```python
from typing import Any, Dict, List
from langchain_core.memory import BaseMemory
from langchain_core.prompts import PromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain.chains import ConversationChain
from titan import Titan, TitanConfig

class TitanMemory(BaseMemory):
    """LangChain memory backed by Titan."""

    titan: Any = None
    memory_key: str = "titan_context"
    input_key: str = "input"
    human_prefix: str = "Human"
    ai_prefix: str = "AI"

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, data_dir: str = "./lc_memory", **kwargs):
        super().__init__(**kwargs)
        self.titan = Titan(TitanConfig(data_dir=data_dir))

    @property
    def memory_variables(self) -> List[str]:
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """Called BEFORE the LLM. Retrieves relevant context."""
        query = inputs.get(self.input_key, "")
        context = self.titan.get_context_string(query) if query else ""
        return {self.memory_key: context}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Called AFTER the LLM. Stores the conversation turn."""
        user_input = inputs.get(self.input_key, "")
        ai_output = outputs.get("response", outputs.get("output", ""))
        if user_input:
            self.titan.ingest(f"{self.human_prefix}: {user_input}", origin="user")
        if ai_output:
            self.titan.ingest(f"{self.ai_prefix}: {ai_output}", origin="observation")

    def clear(self) -> None:
        self.titan.run_maintenance()

# Usage
memory = TitanMemory(data_dir="./lc_memory")
llm = ChatAnthropic(model="claude-sonnet-4-20250514")

template = """You are a helpful assistant with long-term memory.

{titan_context}

Current conversation:
Human: {input}
AI:"""
prompt = PromptTemplate(input_variables=["input", "titan_context"], template=template)
chain = ConversationChain(llm=llm, memory=memory, prompt=prompt)
result = chain.invoke({"input": "My dog's name is Luna"})
print(result["response"])
```

### Option B: Modern LCEL Pipe

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_anthropic import ChatAnthropic
from titan import Titan, TitanConfig

titan = Titan(TitanConfig(data_dir="./lc_memory"))
llm = ChatAnthropic(model="claude-sonnet-4-20250514")

def inject_memory(input_dict):
    query = input_dict["input"]
    context = titan.get_context_string(query)
    return {**input_dict, "memory": context}

def store_and_return(response):
    titan.ingest(response.content, origin="observation")
    return response

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant.\n\n{memory}"),
    ("human", "{input}")
])

chain = RunnableLambda(inject_memory) | prompt | llm | RunnableLambda(store_and_return)
result = chain.invoke({"input": "My dog's name is Luna"})
print(result.content)
```

### Option C: AgentExecutor with Titan Tools

```python
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_anthropic import ChatAnthropic
from langchain.agents import create_tool_calling_agent, AgentExecutor
from titan import Titan, TitanConfig

titan = Titan(TitanConfig(data_dir="./lc_memory"))

@tool
def memory_store(text: str, origin: str = "user") -> str:
    """Store information in long-term memory."""
    result = titan.ingest(text, origin=origin)
    return f"Stored: {result['claims']} claims, {result['entities']} entities"

@tool
def memory_recall(query: str, limit: int = 5) -> str:
    """Search long-term memory for relevant information."""
    results = titan.retrieve(query, limit=limit)
    if not results:
        return "No memories found."
    return "\n".join(
        f"[{r.get('confidence', '?')}] {r.get('text', r.get('content', ''))}"
        for r in results
    )

llm = ChatAnthropic(model="claude-sonnet-4-20250514")
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant with long-term memory tools."),
    MessagesPlaceholder("chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])
agent = create_tool_calling_agent(llm, [memory_store, memory_recall], prompt)
executor = AgentExecutor(agent=agent, tools=[memory_store, memory_recall])
result = executor.invoke({"input": "Remember: my server runs on port 8101"})
print(result["output"])
```

**Gotchas:**
- LangChain v0.2+ deprecated `ConversationChain` in favor of LCEL pipes. Option B is the modern approach.
- `memory_variables` must match your prompt template variable names exactly.
- `save_context` is called AFTER the LLM response, not during.
- The memory object is shared across chain invocations (stateful).

---

## LlamaIndex

**Install:**
```bash
pip install llama-index llama-index-llms-anthropic titan-memory
```

**How it works:** LlamaIndex agents use a `Memory` object with `MemoryBlock` components. `aget()` is called each turn to inject long-term context into the system prompt. `aput()` is called when short-term buffer messages are flushed. A custom `BaseRetriever` can also be used in RAG pipelines.

```python
import asyncio
from typing import List, Optional
from llama_index.core.memory import Memory, BaseMemoryBlock
from llama_index.core.llms import ChatMessage
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, TextNode, QueryBundle
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import FunctionCallingAgent
from llama_index.llms.anthropic import Anthropic
from titan import Titan, TitanConfig

titan = Titan(TitanConfig(data_dir="./li_memory"))

# Custom MemoryBlock: long-term memory for the agent
class TitanMemoryBlock(BaseMemoryBlock):
    name: str = "titan_memory"
    _titan: Titan = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, titan_instance: Titan, **kwargs):
        super().__init__(**kwargs)
        self._titan = titan_instance

    async def aget(self, messages: Optional[List[ChatMessage]] = None, **kwargs) -> str:
        """Called each turn. Returns context prepended to system prompt."""
        if not messages:
            return ""
        query = ""
        for msg in reversed(messages):
            if msg.role == "user":
                query = msg.content
                break
        if not query:
            return ""
        return self._titan.get_context_string(query)

    async def aput(self, messages: List[ChatMessage]) -> None:
        """Called when messages are flushed from short-term buffer."""
        for msg in messages:
            origin = "user" if msg.role == "user" else "observation"
            self._titan.ingest(msg.content, origin=origin)

    async def atruncate(self) -> str:
        return ""  # Titan handles its own pruning

# Custom Retriever: for RAG pipelines
class TitanRetriever(BaseRetriever):
    _titan: Titan = None
    _limit: int = 5

    def __init__(self, titan_instance: Titan, limit: int = 5, **kwargs):
        super().__init__(**kwargs)
        self._titan = titan_instance
        self._limit = limit

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        results = self._titan.retrieve(query_bundle.query_str, limit=self._limit)
        nodes = []
        for r in results:
            text = r.get("text", r.get("content", ""))
            node = TextNode(
                text=text,
                metadata={
                    "node_id": r.get("node_id", ""),
                    "timestamp": r.get("timestamp", ""),
                    "origin": r.get("origin", ""),
                }
            )
            score = r.get("confidence", r.get("score", 0.5))
            nodes.append(NodeWithScore(node=node, score=score))
        return nodes

# Agent tools
def memory_store_tool(text: str, origin: str = "user") -> str:
    """Store information in long-term memory."""
    result = titan.ingest(text, origin=origin)
    return f"Stored: {result['claims']} claims"

def memory_recall_tool(query: str) -> str:
    """Search long-term memory."""
    return titan.get_context_string(query)

tools = [
    FunctionTool.from_defaults(fn=memory_store_tool),
    FunctionTool.from_defaults(fn=memory_recall_tool),
]

# Build agent
titan_block = TitanMemoryBlock(titan_instance=titan)
agent_memory = Memory.from_defaults(
    memory_blocks=[titan_block],
    session_id="user_session_1",
    token_flush_size=3000,
)

llm = Anthropic(model="claude-sonnet-4-20250514")
agent = FunctionCallingAgent.from_tools(
    tools=tools,
    llm=llm,
    memory=agent_memory,
    system_prompt="You are a helpful assistant with long-term memory.",
)

response = agent.chat("My project uses port 8101 for the reasoning model.")
print(response)
```

**Gotchas:**
- `MemoryBlock.aget()` must return a string (prepended to the message list).
- `MemoryBlock.aput()` receives flushed messages from the short-term buffer, not every message.
- LlamaIndex uses async by default; sync wrappers exist but add overhead.
- Short-term memory is a FIFO queue; old messages get flushed to long-term via `aput()`.
- `token_flush_size` controls when short-term messages are flushed to long-term.

---

## CrewAI

**Install:**
```bash
pip install crewai titan-memory
```

**How it works:** CrewAI agents use a ReAct loop (Thought -> Action -> Observation) to process tasks. Memory is integrated via tools -- the agent decides when to store and recall. CrewAI has its own built-in memory system (ChromaDB), but using Titan directly via tools is cleaner and avoids the ChromaDB dependency.

```python
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from titan import Titan, TitanConfig

titan = Titan(TitanConfig(data_dir="./crew_memory"))

@tool("Store Memory")
def store_memory(text: str) -> str:
    """Store important information in long-term Titan memory.
    Use when you learn new facts, decisions, or user preferences."""
    result = titan.ingest(text, origin="observation")
    return f"Stored: {result['claims']} claims, entities: {result['entities']}"

@tool("Recall Memory")
def recall_memory(query: str) -> str:
    """Search Titan long-term memory for relevant information.
    Use when you need context from previous conversations or stored knowledge."""
    return titan.get_context_string(query)

@tool("Memory Stats")
def memory_stats() -> str:
    """Get memory system statistics."""
    stats = titan.get_stats()
    return f"Nodes: {stats['nodes']}, Edges: {stats['edges']}, Vectors: {stats['vectors']}"

researcher = Agent(
    role="Research Analyst",
    goal="Find and remember important information",
    backstory="You are a meticulous researcher who stores findings in memory.",
    tools=[store_memory, recall_memory, memory_stats],
    llm="claude-sonnet-4-20250514",
    verbose=True
)

writer = Agent(
    role="Technical Writer",
    goal="Write reports using stored knowledge",
    backstory="You access memory to find relevant context for writing.",
    tools=[recall_memory],
    llm="claude-sonnet-4-20250514",
    verbose=True
)

research_task = Task(
    description="Research the user's project and store key findings. Project: {project_name}",
    expected_output="Summary of findings stored in memory",
    agent=researcher,
)

writing_task = Task(
    description="Write a status report using recalled memories about: {project_name}",
    expected_output="A concise status report",
    agent=writer,
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    process=Process.sequential,
    verbose=True,
)

result = crew.kickoff(inputs={"project_name": "My AI Project"})
```

**Gotchas:**
- Setting `memory=True` on the Crew enables CrewAI's built-in ChromaDB memory ON TOP of whatever you do with tools. Use `memory=False` (default) when using Titan directly via tools to avoid confusion.
- ChromaDB is a hard dependency when `memory=True`.
- There is no official custom memory backend API in CrewAI (as of early 2026). The recommended approach is tools.
- Entity memory and short-term memory both use ChromaDB by default.

---

## AutoGen v0.4

**Install:**
```bash
pip install autogen-agentchat autogen-ext titan-memory
```

**How it works:** AutoGen v0.4 uses a `Memory` protocol. `update_context()` is called before each LLM call to inject relevant memories into the model's message list. `add()` stores content after each turn. `query()` searches memory on demand.

```python
import asyncio
from typing import Any
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_core.memory import Memory, MemoryContent, MemoryQueryResult
from autogen_core.model_context import ChatCompletionContext
from autogen_ext.models.anthropic import AnthropicChatCompletionClient
from titan import Titan, TitanConfig

class TitanAutoGenMemory(Memory):
    """AutoGen v0.4 Memory backed by Titan."""

    def __init__(self, data_dir: str = "./ag_memory"):
        self.titan = Titan(TitanConfig(data_dir=data_dir))
        self.name = "titan_memory"

    async def update_context(self, model_context: ChatCompletionContext) -> None:
        """Called before each LLM call. Injects Titan context."""
        messages = await model_context.get_messages()
        query = ""
        for msg in reversed(messages):
            if hasattr(msg, 'content') and msg.role == "user":
                query = msg.content if isinstance(msg.content, str) else str(msg.content)
                break
        if not query:
            return
        context_str = self.titan.get_context_string(query)
        if context_str and context_str.strip():
            from autogen_core.models import SystemMessage
            await model_context.add_message(
                SystemMessage(content=f"Relevant memories:\n{context_str}")
            )

    async def add(self, content: MemoryContent, cancellation_token: Any = None) -> None:
        """Store content in Titan."""
        text = content.content if isinstance(content.content, str) else str(content.content)
        source = content.source if hasattr(content, 'source') else "observation"
        self.titan.ingest(text, origin=source)

    async def query(
        self, query: str | MemoryContent, cancellation_token: Any = None
    ) -> MemoryQueryResult:
        """Query Titan memory."""
        query_str = query if isinstance(query, str) else str(query.content)
        results = self.titan.retrieve(query_str)
        memory_contents = []
        for r in results:
            memory_contents.append(MemoryContent(
                content=r.get("text", r.get("content", "")),
                source="titan",
                metadata={
                    "confidence": r.get("confidence", 0),
                    "node_id": r.get("node_id", ""),
                }
            ))
        return MemoryQueryResult(results=memory_contents)

    async def clear(self) -> None:
        self.titan.run_maintenance()

    async def close(self) -> None:
        pass

# Usage
async def main():
    memory = TitanAutoGenMemory(data_dir="./ag_memory")

    async def remember_fact(text: str, origin: str = "user") -> str:
        """Store a fact in long-term memory."""
        result = memory.titan.ingest(text, origin=origin)
        return f"Stored: {result['claims']} claims"

    async def recall_facts(query: str) -> str:
        """Search long-term memory."""
        return memory.titan.get_context_string(query)

    model_client = AnthropicChatCompletionClient(model="claude-sonnet-4-20250514")

    agent = AssistantAgent(
        name="titan_agent",
        model_client=model_client,
        memory=[memory],
        tools=[remember_fact, recall_facts],
        system_message="You are a helpful assistant with long-term memory.",
    )

    stream = agent.run_stream(task="My server runs the RLM on port 8101. Remember that.")
    await Console(stream)

asyncio.run(main())
```

**Gotchas:**
- AutoGen v0.4 is a complete rewrite from v0.2 -- the APIs are different.
- `update_context()` mutates the context IN PLACE (no return value).
- All memory methods are async.
- `MemoryContent` is the standard wrapper for content going into memory.

---

## Pydantic AI

**Install:**
```bash
pip install pydantic-ai titan-memory
```

**How it works:** Pydantic AI uses typed dependency injection. You define a deps dataclass carrying the Titan instance, then register tools that receive `RunContext[DepsType]` as their first argument. A `@agent.system_prompt` decorator dynamically injects memory context before each run.

```python
import asyncio
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
from titan import Titan, TitanConfig

@dataclass
class TitanDeps:
    """Carries Titan instance into tool functions."""
    titan: Titan
    user_id: str = "default"

agent = Agent(
    "anthropic:claude-sonnet-4-20250514",
    deps_type=TitanDeps,
    system_prompt=(
        "You are a helpful assistant with long-term memory. "
        "Use memory_store to save important information and "
        "memory_recall to search for relevant context."
    ),
)

@agent.system_prompt
async def inject_memory_context(ctx: RunContext[TitanDeps]) -> str:
    """Called before each LLM call. Returns text prepended to system prompt."""
    return ctx.deps.titan.get_context_string("recent context")

@agent.tool
async def memory_store(
    ctx: RunContext[TitanDeps], text: str, origin: str = "user"
) -> str:
    """Store information in long-term memory. Use for facts, preferences, decisions."""
    result = ctx.deps.titan.ingest(text, origin=origin)
    return f"Stored: {result['claims']} claims, entities: {result['entities']}"

@agent.tool
async def memory_recall(
    ctx: RunContext[TitanDeps], query: str, limit: int = 5
) -> str:
    """Search long-term memory for relevant information."""
    results = ctx.deps.titan.retrieve(query, limit=limit)
    if not results:
        return "No relevant memories found."
    lines = []
    for r in results:
        lines.append(
            f"[{r.get('timestamp', '?')}] (conf={r.get('confidence', '?')}) "
            f"{r.get('text', r.get('content', ''))}"
        )
    return "\n".join(lines)

async def main():
    titan = Titan(TitanConfig(data_dir="./pai_memory"))
    deps = TitanDeps(titan=titan, user_id="alice")

    result = await agent.run(
        "My project is a local AI assistant with long-term memory capabilities.",
        deps=deps,
    )
    print(result.data)

    # Second call -- memory persists
    result = await agent.run(
        "What project am I working on?",
        deps=deps,
        message_history=result.all_messages(),
    )
    print(result.data)

asyncio.run(main())
```

**Gotchas:**
- Dependencies are immutable during a run -- set them before `agent.run()`.
- Tools receive `RunContext[DepsType]` as their first argument; other arguments become the LLM's tool schema.
- System prompt functions are called EACH run, not once at agent creation.
- No built-in conversation history -- manage it via the `message_history` parameter.

---

## Open Interpreter

**Install:**
```bash
pip install open-interpreter titan-memory
```

**How it works:** Open Interpreter has no formal plugin system. Memory injection happens via `interpreter.custom_instructions`, which is appended to the system message every turn. The LLM accesses Titan by generating Python code that calls the Titan API -- Open Interpreter executes it.

```python
from interpreter import interpreter
from titan import Titan, TitanConfig

memory = Titan(TitanConfig(data_dir="./oi_memory"))

def chat_with_memory(user_message: str) -> str:
    memory_block = memory.get_context_string(user_message)

    interpreter.custom_instructions = f"""
You have access to long-term memory via the `titan` Python package.
It is already installed. Use it when relevant.

Current memory context:
{memory_block}

To store new information:
    from titan import remember
    remember("The user prefers dark mode.", origin="user")

To search memory:
    from titan import recall
    results = recall("user preferences")
    for r in results:
        print(r['text'], r['confidence'])

To get a formatted context block:
    from titan import get_context
    context = get_context("current project status")
"""
    interpreter.llm.model = "claude-sonnet-4-20250514"
    interpreter.auto_run = True  # Skip code approval prompts

    response = interpreter.chat(user_message)

    # Auto-ingest the conversation turn
    memory.ingest(
        f"User: {user_message}\nAssistant: {response[-1].get('content', '') if response else ''}",
        origin="observation"
    )
    return response

# Usage
chat_with_memory("Remember that my server runs on port 8101")
chat_with_memory("What port does my server run on?")
```

**Gotchas:**
- No `tool_use` / function calling -- memory access goes through code generation and execution.
- `interpreter.messages` persists in-process but is NOT saved across restarts.
- `custom_instructions` is a string, not a structured hook. It is re-rendered each turn.
- The LLM must be told to use the Titan API in its system prompt for it to actually call it.
- `auto_run=True` skips code approval prompts -- use with caution.

---

## Summary Table

| Framework | Integration Pattern | Memory Hook | Difficulty |
|-----------|-------------------|-------------|------------|
| **Claude API** | `tool_use` (store/recall/forget) | `tools` param + system prompt | Easy |
| **Ollama** | System prompt + tools | Manual message building | Easy |
| **OpenAI-compat** | System prompt + tools | Manual message building | Easy |
| **LangChain** | BaseMemory subclass + tools | `load_memory_variables`/`save_context` | Medium |
| **LlamaIndex** | MemoryBlock + Retriever | `aget`/`aput` + `_retrieve` | Medium |
| **CrewAI** | `@tool` decorator | Tools (no custom storage API) | Easy |
| **AutoGen v0.4** | Memory protocol subclass | `update_context`/`add`/`query` | Medium |
| **Pydantic AI** | deps + `@agent.tool` | `RunContext` injection | Easy |
| **Open Interpreter** | `custom_instructions` + codegen | System prompt + Python exec | Easy |

**Recommended starting points:**

1. **Claude API** -- Deepest integration. Claude decides when to store and recall.
2. **Pydantic AI** -- Cleanest developer experience. Type-safe, Titan in deps.
3. **OpenAI-compatible** -- Widest reach. Works with any provider.
4. **LangChain AgentExecutor** -- Largest ecosystem. Tool-based approach.
