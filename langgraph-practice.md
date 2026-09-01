# LangGraph Practice — Built as a Chain, Not a List

Continues from `langchain-practice.md`. Every snippet actually ran, in `D:\nvidia\.venv-langchain` (`langgraph==1.2.9`, `langchain==1.3.14`), against the same Azure OpenAI deployment (`gpt-4.1-mini`) as the rest of this project.

LangGraph is the piece LangChain 1.x now points to for anything stateful — agents, memory, human-in-the-loop. See the deprecation warning documented at the bottom of `langchain-practice.md`.

Each cluster builds on the one before it. First the code, then why it matters, then a self-check to confirm it stuck.

---

> 🔗 **Hands-on reps:** [Code Drills 10 — StateGraph Basics](/topic/code-drills-langgraph-agents#cluster-1-stategraph-basics)

## Cluster 1 — State and Nodes: The Two Ideas Everything Else Builds On

> **TL;DR**
> - A `StateGraph` is nodes (plain functions) plus a shared `State` (a `TypedDict`). Each node returns a *partial* dict, and LangGraph shallow-merges it into the running state by key.
> - Default merge behavior is **overwrite** — fine for a single "answer" key, but it silently deletes conversation history if you use it for chat messages.
> - `Annotated[list, add_messages]` swaps that overwrite for **append** — the one line that turns "each node's return replaces the field" into "each node's return adds to it," which is the shape every chat agent actually needs.

### The minimal graph
The smallest possible graph is one node, wired to a start and an end:

```python
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_openai import AzureChatOpenAI
import os

llm = AzureChatOpenAI(
    azure_deployment="gpt-4.1-mini",
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"].strip(),
    api_key=os.environ["AZURE_OPENAI_KEY"].strip(),
    api_version="2024-06-01", temperature=0,
)

class State(TypedDict):
    question: str
    answer: str

def answer_node(state: State) -> dict:
    resp = llm.invoke(state["question"])
    return {"answer": resp.content}     # return a PARTIAL dict -- LangGraph merges it into state

graph = StateGraph(State)
graph.add_node("answer", answer_node)
graph.set_entry_point("answer")
graph.add_edge("answer", END)
app = graph.compile()

result = app.invoke({"question": "What is a KV cache, one sentence?"})
print(result)   # {'question': '...', 'answer': '...'} -- input keys survive alongside the node's output
```
Two terms here are doing real work. A `TypedDict` is just a plain dict with declared key names and types. It serves as the schema for the shared state. "Shallow-merged" means merged key by key, at the top level: keys a node returns get written in, and every other key survives untouched.

The key thing to notice: a node returns a *partial* dict, not the full state. LangGraph merges that partial dict into the running state, key by key — it doesn't swap the whole state out. That's the entire mechanism that lets a 10-node graph work: each node only "owns" the keys it actually computes. No node needs to know about, or re-thread, the rest of the state.

### Making a field append instead of overwrite
That default merge behavior — plain overwrite — turns into a real bug the moment a graph needs to accumulate conversation history, instead of just producing one answer:

```python
from typing import Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage

class ChatState(TypedDict):
    messages: Annotated[list, add_messages]     # the Annotated reducer is the whole trick

def chat_node(state: ChatState) -> dict:
    resp = llm.invoke(state["messages"])
    return {"messages": [resp]}      # returning ONE new message, not the full list

g = StateGraph(ChatState)
g.add_node("chat", chat_node)
g.set_entry_point("chat")
g.add_edge("chat", END)
app = g.compile()

r1 = app.invoke({"messages": [HumanMessage("My name is Gowtham.")]})
print(len(r1["messages"]))                                   # 2: human + AI
r2 = app.invoke({"messages": r1["messages"] + [HumanMessage("What's my name?")]})
print(len(r2["messages"]), r2["messages"][-1].content)        # 4, and it remembers "Gowtham"
```
Notice it's `Annotated[list, add_messages]`, not just `list`. Without a reducer, the default merge behavior is plain overwrite. A node returning `{"messages": [new_msg]}` would *replace* the whole history with a single message — silently deleting everything before it.

`add_messages` is a specific reducer function. It appends instead of overwriting. It also de-duplicates messages by message ID, and converts plain dicts or tuples into proper message objects.

This exact pattern is the standard way every LangGraph chat agent accumulates conversation history. Forgetting the annotation is the single most common "why did my agent forget everything" bug.

The two merge behaviors side by side:

```
  Default (overwrite):                      With add_messages (append):

  state = {answer: "A"}                     state = {messages: [Human, AI]}
  node returns {answer: "B"}                node returns {messages: [AI-2]}
        │                                          │
        ▼                                          ▼
  state = {answer: "B"}                     state = {messages: [Human, AI, AI-2]}
  (A is GONE)                               (nothing lost -- AI-2 appended)
```

<details>
<summary><strong>Self-check — answer before revealing</strong></summary>

1. What does a node function actually return — the full state, or something smaller?
2. Without `Annotated[list, add_messages]`, what happens to conversation history when a second node runs and returns `{"messages": [new_msg]}`?
3. What does `add_messages` do beyond simple appending?
4. Why does a 10-node graph benefit from each node only returning the keys it computed, rather than the entire state?
5. If a chat agent "forgets" everything after the second turn, what's the first thing to check in its state schema?

**Answers**
1. A *partial* dict — just the keys that node actually computed. LangGraph shallow-merges it into the running state by key; it doesn't need the rest of the state re-supplied.
2. It gets wiped — the default merge behavior is overwrite, so the new one-message list replaces the entire existing history instead of adding to it, with no error or warning.
3. It also de-duplicates messages by message ID and converts plain dicts/tuples into proper message objects — not just appending, but keeping the list well-formed.
4. It means no node needs to know about or re-thread keys it doesn't care about — each node's contract is just "the keys I own," which scales much better than every node needing full awareness of a growing state shape.
5. Whether the `messages` field (or whatever accumulates history) is declared as `Annotated[list, add_messages]` — a plain `list` with no reducer is the single most common cause of that bug.
</details>

> **Recap**
> Nodes return partial dicts; LangGraph merges them into state by key. Default merge is overwrite, which is fine for a single computed value but silently destroys accumulating data like chat history. `Annotated[list, add_messages]` switches that field to append-with-dedup — the one-line difference between a graph that answers once and an agent that remembers.

---

> 🔗 **Hands-on reps:** [Code Drills 10 — Conditional Edges & Control Flow](/topic/code-drills-langgraph-agents#cluster-2-conditional-edges-control-flow)

## Cluster 2 — Control Flow: Conditional Edges and Runaway Recursion

> **TL;DR**
> - `add_conditional_edges` is the diamond decision box in an otherwise boxes-and-arrows graph — a router function reads state and returns a *label*, and a separate dict maps that label to the actual next node.
> - A loop isn't a special construct — it's just a node whose router sometimes points back to that same node.
> - If a router never returns its exit label, the graph doesn't hang silently — it raises a catchable `GraphRecursionError` once it hits `recursion_limit` (default 25).

### Branching, and building a loop from it
Every `StateGraph` is boxes and arrows. The boxes are nodes — plain functions. The arrows are edges — "go here next." `add_conditional_edges` is the one arrow that's actually a diamond. It reads state, then picks which arrow to follow from a dict of labeled options, instead of always going the same direction. Here's a loop, built from exactly that:

```python
class LoopState(TypedDict):
    n: int
    total: int

def add_one(state: LoopState) -> dict:
    return {"n": state["n"] + 1, "total": state["total"] + state["n"]}

def should_continue(state: LoopState) -> str:
    return "loop" if state["n"] < 5 else "done"          # the ROUTER: returns a label, not a node name directly

g = StateGraph(LoopState)
g.add_node("add_one", add_one)
g.set_entry_point("add_one")
g.add_conditional_edges("add_one", should_continue, {"loop": "add_one", "done": END})
app = g.compile()

print(app.invoke({"n": 0, "total": 0}))   # {'n': 5, 'total': 10} -- looped 5 times before hitting END
```
The router function returns a label string. A separate dict maps that label to a node name. Keeping those two things apart matters: "which branch to take" (the router's logic) stays decoupled from "which node that branch points to" (the mapping dict). That means the same router function can be reused across graphs with differently-named nodes. It also keeps the graph's edges readable as a plain mapping, instead of buried inside conditional logic.

This is also literally how a cycle gets built. `add_one` pointing back to itself, via `"loop": "add_one"`, is a loop. Nothing special about it — just an edge like any other.

Draw the graph the code actually builds, not the code itself:

```
                    ┌─────────────┐
         ┌─────────▶│   add_one   │────┐
         │          └─────────────┘    │
         │  "loop"         │           │  should_continue(state)
         │           n < 5 │           │  reads state, returns
         └─────────────────┘           │  a LABEL ("loop"/"done"),
                                        │  not a node name
                                        ▼
                             n >= 5 ──▶ END
                             "done"
```
Once you can sketch a graph this way on paper, before writing any code, "why is my agent looping forever" turns into a simple exercise: trace the arrows, and find the one that never reaches END. You don't need to re-read Python control flow to find the bug.

### When the router never lets go
That same diamond-shaped router reappears here with one bug: it never returns `"done"`.

```python
def should_continue_forever(state: LoopState) -> str:
    return "loop"     # bug: never returns "done" -- this WILL hit the limit

g2 = StateGraph(LoopState)
g2.add_node("add_one", add_one)
g2.set_entry_point("add_one")
g2.add_conditional_edges("add_one", should_continue_forever, {"loop": "add_one", "done": END})
app2 = g2.compile()

try:
    app2.invoke({"n": 0, "total": 0}, config={"recursion_limit": 5})
except Exception as e:
    print(type(e).__name__, ":", e)
# GraphRecursionError : Recursion limit of 5 reached without hitting a stop condition.
```
The arrow structure here is identical to the working loop above — `add_one` still points back to itself. So the graph doesn't hang silently. It raises a catchable `GraphRecursionError` once it hits the configured limit.

This matters more than it looks. The default `recursion_limit` is 25. That's fine for a short, deterministic loop like the one above. But an LLM-driven agent loop — tool call, result, maybe another tool call, and so on — can legitimately need more steps. It can also spiral, if the model keeps re-requesting a failing tool.

`GraphRecursionError` is a real, catchable exception. It's not a silent infinite hang. But that only helps if something in the calling code actually catches and handles it — otherwise the whole request just fails with a 500.

<details>
<summary><strong>Self-check — answer before revealing</strong></summary>

1. What does a router function passed to `add_conditional_edges` return — a node name, or something else?
2. Why is a loop not a "special construct" in LangGraph?
3. What's the default `recursion_limit`, and what exception fires when it's hit?
4. If `should_continue_forever` never returns `"done"`, does the graph hang forever?
5. Why does decoupling the router's label from the node-name mapping make the router function more reusable?

**Answers**
1. A label string (like `"loop"` or `"done"`) — a separate dict passed to `add_conditional_edges` maps that label to the actual node name to go to next.
2. It's just a node whose router sometimes points back to that same node — `"loop": "add_one"` is an ordinary edge in the mapping dict, not a distinct language feature.
3. The default is 25; hitting it raises a catchable `GraphRecursionError`, not a silent infinite hang.
4. No — it raises `GraphRecursionError` once `recursion_limit` is reached, as long as the calling code catches it rather than letting the request fail with an uncaught exception.
5. Because the router's logic ("which branch to take") never references actual node names — the same router function can be reused across different graphs where nodes happen to be named differently, since the label-to-node mapping lives separately.
</details>

> **Recap**
> `add_conditional_edges` is the diamond in an otherwise linear boxes-and-arrows graph: a router returns a label, a dict maps that label to a node. A loop is just a router pointing back at its own node — nothing special. If the router never returns its exit label, the graph raises a catchable `GraphRecursionError` at `recursion_limit` (default 25) rather than hanging silently, so tracing the arrows on paper is the fastest way to find a missing exit condition.

---

> 🔗 **Hands-on reps:** [Code Drills 10 — Memory & Checkpointing](/topic/code-drills-langgraph-agents#cluster-4-memory-checkpointing)

## Cluster 3 — Memory: Persisting State Across Separate Calls

> **TL;DR**
> - A single `.invoke()` already accumulates messages within itself (Cluster 1) — a `checkpointer=` at compile time plus a stable `thread_id` in `config` is what persists that state *across* separate `.invoke()` calls.
> - Different `thread_id`s get completely isolated histories — same mechanism LangChain 1.x's deprecation warning points at (`thread_id` is the LangGraph equivalent of `RunnableWithMessageHistory`'s `session_id`).
> - `MemorySaver` is in-process only. Restart the Python process and every thread's history is gone, silently — `SqliteSaver`/`PostgresSaver` are the real production equivalents, same interface.

### Persisting state across calls, not just within one
This reuses the `ChatState` graph from Cluster 1. Compiling it with a checkpointer is what turns "remembers within one `.invoke()`" into "remembers across many separate calls":

```python
from langgraph.checkpoint.memory import MemorySaver

app = g.compile(checkpointer=MemorySaver())    # reusing the ChatState graph from Cluster 1

cfg = {"configurable": {"thread_id": "conv-1"}}
app.invoke({"messages": [HumanMessage("My name is Gowtham.")]}, config=cfg)
r = app.invoke({"messages": [HumanMessage("What's my name?")]}, config=cfg)
print(r["messages"][-1].content)          # "Your name is Gowtham." -- remembered via thread_id

cfg2 = {"configurable": {"thread_id": "conv-2"}}          # DIFFERENT thread_id
r2 = app.invoke({"messages": [HumanMessage("What's my name?")]}, config=cfg2)
print(r2["messages"][-1].content)          # has no idea -- separate thread, separate history
```
This is the exact mechanism LangChain 1.x's deprecation warning points to: `RunnableWithMessageHistory is deprecated. Use LangGraph's built-in persistence instead.` `thread_id` here is LangGraph's equivalent of `RunnableWithMessageHistory`'s `session_id`.

The pitfall that matters most in practice: `MemorySaver` is in-process memory only. Restart the Python process, and every thread's history is gone — with no error. The next call just starts a fresh conversation, silently.

`SqliteSaver` and `PostgresSaver` are the production equivalents. Same interface, real persistence. `MemorySaver` is for local dev and testing only — the same way `InMemoryVectorStore` (from the LangChain doc) is.

The checkpointing flow, end to end:

```
  compile(checkpointer=MemorySaver())
              │
  ┌───────────┴────────────────────────────────────────┐
  │                                                      │
  ▼                                                      ▼
thread_id="conv-1"                                thread_id="conv-2"
  │                                                      │
  invoke: "My name is Gowtham."                          │
  │  saved to checkpoint store, keyed by thread_id        │
  ▼                                                      ▼
  invoke: "What's my name?"                        invoke: "What's my name?"
  │  loads conv-1's saved history first                    │  loads conv-2's history (EMPTY)
  ▼                                                      ▼
  "Your name is Gowtham."                           "I don't know your name"
  (remembered)                                       (isolated -- never saw conv-1)

  ── process restart ──▶  MemorySaver's dict is gone. Both thread_ids start from empty, silently.
```

<details>
<summary><strong>Self-check — answer before revealing</strong></summary>

1. What two things does a graph need before it can remember across separate `.invoke()` calls?
2. Why does `thread_id="conv-2"` get a completely fresh history instead of seeing `"conv-1"`'s conversation?
3. What LangChain 1.x deprecation warning does this mechanism directly answer?
4. What happens to `MemorySaver`'s saved histories when the Python process restarts, and does it raise an error?
5. What's the production-grade replacement for `MemorySaver`, and how much code changes when swapping it in?

**Answers**
1. A `checkpointer=` (e.g. `MemorySaver()`) passed at `.compile()` time, and a stable `thread_id` passed in `config={"configurable": {"thread_id": ...}}` on every call.
2. Each `thread_id` is a separate key in the checkpoint store — the checkpointer looks up saved state by that key, so a different `thread_id` simply has nothing saved under it yet.
3. `RunnableWithMessageHistory is deprecated. Use LangGraph's built-in persistence instead` — `thread_id` here is the direct equivalent of that class's `session_id`.
4. Every thread's history is gone, with zero error signal — the next call just starts a fresh conversation silently, since `MemorySaver` is just an in-process Python dict.
5. `SqliteSaver` or a Postgres-backed checkpointer — swapping it in requires no other code changes, since they share the same `BaseCheckpointSaver` interface as `MemorySaver`.
</details>

> **Recap**
> `checkpointer=MemorySaver()` at compile time plus a stable `thread_id` in `config` is what persists a graph's state across separate `.invoke()` calls — different `thread_id`s stay fully isolated. It's the direct LangGraph equivalent of `RunnableWithMessageHistory`'s `session_id`, but `MemorySaver` itself is dev-only: a process restart silently wipes every thread's history, which is exactly what `SqliteSaver`/`PostgresSaver` fix in production.

---

> 🔗 **Hands-on reps:** [Code Drills 10 — Tool-Calling Agents](/topic/code-drills-langgraph-agents#cluster-3-tool-calling-agents)

## Cluster 4 — Building an Agent, and Watching It Think Step by Step

> **TL;DR**
> - `create_agent(llm, tools=[...])` gives you the same request → execute → respond loop hand-rolled manually in `langchain-practice.md`, without writing the loop yourself.
> - Watch for the name: `langgraph.prebuilt.create_react_agent` is deprecated as of LangGraph v1.0 — the current import is `from langchain.agents import create_agent`.
> - Under the hood, `create_agent` compiles to the exact conditional-edge loop from Cluster 2 — request a tool, route to a tool-execution node, repeat until the model stops requesting tools.
> - `.stream(stream_mode="updates")` surfaces each of those hidden internal steps individually, instead of waiting for the final answer.

### Building the agent without hand-rolling the loop
`langchain-practice.md` walked through the tool-calling loop by hand, once. `create_agent` gives you that same behavior, pre-built:

```python
from langchain.agents import create_agent      # NOT langgraph.prebuilt.create_react_agent -- see note below
from langchain_core.tools import tool

@tool
def exam_day_countdown(target_date: str) -> str:
    """Given an ISO date (YYYY-MM-DD), return how many days remain until it."""
    from datetime import date
    d = date.fromisoformat(target_date) - date.today()
    return f"{d.days} days remaining"

agent = create_agent(llm, tools=[exam_day_countdown])
result = agent.invoke({"messages": [("human", "How many days until 2026-07-13?")]})
print(result["messages"][-1].content)
```
This is a real, current gotcha, verified the hard way. `langgraph.prebuilt.create_react_agent` — the function name in essentially every LangGraph tutorial and blog post up to this point — still runs. But it prints a warning: `LangGraphDeprecatedSinceV10: create_react_agent has been moved to langchain.agents. Please update your import to from langchain.agents import create_agent. Deprecated in LangGraph V1.0 to be removed in V2.0.`

The prebuilt ReAct-agent constructor moved from the `langgraph` package to the `langchain` package as of LangGraph v1.0. `create_agent` — new name, new package — is the non-deprecated path. If a snippet imports `create_react_agent` from `langgraph.prebuilt`, it's targeting pre-1.0 LangGraph.

Under the hood, `create_agent` compiles to exactly the kind of conditional-edge loop built by hand in Cluster 2:

```
   ┌──────────────────────────────────────────┐
   │                                            │
   ▼                                            │
 [ model node ]  ──requested a tool?──▶ [ tool-execution node ] ──┘
   │       yes: route to tool node               (runs the tool,
   │       no: route to END                        appends result)
   ▼
  END (final answer)
```
Request a tool. Route back to a tool-execution node. Repeat until the model stops requesting tools. It's the same shape as the manual `while`-style loop in `langchain-practice.md` — just wrapped behind one `create_agent(...)` call.

### Watching the hidden steps fire
`create_agent` hides that internal graph structure by default. `.invoke()` only returns the final answer. To see each intermediate step, swap in `.stream()`, with the right mode:

```python
class PipelineState(TypedDict):
    question: str
    draft: str
    answer: str

def draft_node(state: PipelineState) -> dict:
    return {"draft": llm.invoke(state["question"]).content}

def polish_node(state: PipelineState) -> dict:
    return {"answer": state["draft"].upper()[:50]}

g3 = StateGraph(PipelineState)
g3.add_node("draft", draft_node)
g3.add_node("polish", polish_node)
g3.set_entry_point("draft")
g3.add_edge("draft", "polish")
g3.add_edge("polish", END)
app3 = g3.compile()

for step in app3.stream({"question": "What is attention, one sentence?"}, stream_mode="updates"):
    print(step.keys(), list(step.values())[0])
# dict_keys(['draft'])  {'draft': '...'}
# dict_keys(['polish']) {'answer': '...'}
```
A closer look at `stream_mode="updates"`. LangGraph's `.stream()` supports several modes. `"values"` sends the full accumulated state, after every node. `"updates"` sends just what each node returned, keyed by node name — that's the one used above. `"messages"` gives token-level streaming from inside a node, for chat UIs.

Picking the wrong mode is a common confusion. `"values"` re-sends the *entire* state on every step — verbose, but simple. `"updates"` shows exactly what changed — better for a progress UI that says "now running node X." Neither one gives you token-by-token text on its own; that needs `"messages"` mode, or a per-node `.stream()` call on the underlying LLM.

Run the same `create_agent` above with `.stream(..., stream_mode="updates")` instead of `.invoke()`, and it shows exactly which tool-execution step is running at any moment. That's useful for the same reason a progress UI is useful anywhere else in this hub.

<details>
<summary><strong>Self-check — answer before revealing</strong></summary>

1. What's the current, non-deprecated way to import the prebuilt ReAct-agent constructor, and what package did it move from?
2. What does `create_agent` compile to internally, structurally?
3. What are the three `stream_mode` options, and which one shows only what changed per step?
4. If a snippet online imports `create_react_agent` from `langgraph.prebuilt`, what does that tell you about its vintage?
5. Why doesn't `stream_mode="updates"` alone give you token-by-token text?

**Answers**
1. `from langchain.agents import create_agent` — it moved from the `langgraph` package (`langgraph.prebuilt.create_react_agent`, now deprecated) to the `langchain` package as of LangGraph v1.0.
2. The same conditional-edge loop built by hand in Cluster 2 — a model node that requests a tool, routing to a tool-execution node, looping back until the model stops requesting tools.
3. `"values"` (full accumulated state after every node), `"updates"` (just what changed, keyed by node name), and `"messages"` (token-level streaming from inside a node). `"updates"` is the one that shows only what changed.
4. It's targeting pre-1.0 LangGraph — that import path predates the v1.0 move to `langchain.agents`, even though it still runs (with a deprecation warning) as of `1.2.9`.
5. `"updates"` shows what each *node* returned as a whole, not intermediate tokens from inside the LLM call itself — token-by-token text needs `"messages"` mode or a per-node `.stream()` call on the underlying LLM.
</details>

> **Recap**
> `create_agent(llm, tools=[...])` is the manual tool-calling loop, pre-built — internally it's the same conditional-edge request → execute → respond loop from Cluster 2. Import it from `langchain.agents`, not `langgraph.prebuilt` (deprecated since v1.0). `.stream(stream_mode="updates")` surfaces each hidden internal step instead of only the final answer, which is the difference between "updates" (what changed) and "values" (the whole state, resent every time).

---

## Where People Trip Up (in Detail)

- **Chat agent "forgetting" everything after the second turn?** Covered above, but worth restating as the single highest-frequency LangGraph bug: `messages: list` (no reducer) means every node's return overwrites the whole list. There's no exception, no warning — the agent just appears to have amnesia. Check the state schema's reducer before anything else.

- **Following a tutorial that imports `create_react_agent`?** Both the package and the name changed in v1.0, verified above. Any tutorial, blog post, or cached knowledge referencing `from langgraph.prebuilt import create_react_agent` predates LangGraph 1.0. It still works (with a `LangGraphDeprecatedSinceV10` warning) as of `1.2.9`, but is marked for removal in 2.0 — new code should use `from langchain.agents import create_agent`.

- **Assuming `MemorySaver` survives a restart?** It's an in-process Python dict under the hood — a process restart, a redeploy, or even just running two separate Python processes (e.g., a dev server with auto-reload) loses every thread's history with zero error signal. For anything that needs to survive a restart, `SqliteSaver` (same `checkpointer=` interface, one extra import) or a Postgres-backed checkpointer is the real fix — swapping `MemorySaver()` for `SqliteSaver.from_conn_string(...)` requires no other code changes because they share the same `BaseCheckpointSaver` interface.

- **Every call starting a brand-new conversation despite a checkpointer being attached?** Even with a checkpointer attached, if `config={"configurable": {"thread_id": ...}}` is omitted (or a new random ID is generated per call instead of a stable per-user/per-session one), the graph has nothing to look up — it behaves exactly as if no checkpointer were configured at all, silently, with no error.

- **Hit the recursion limit and thinking "just raise the number"?** An LLM-driven conditional edge (route based on whether the model wants to call another tool) can loop far more than a human would predict once tool failures or ambiguous model outputs are in the mix. The default of 25 is a reasonable starting point, but the fix for "hit the limit" is almost never just raising it — a graph that needs 200 steps to answer a question usually has a routing bug (a condition that should route to `END` more often than it does), and raising the limit just delays the same failure into a longer, more expensive one.

- **Got a `KeyError` far from where the actual bug is?** A node returning a key not declared in the `TypedDict` state schema is *not* caught by Python at runtime (`TypedDict` provides no runtime validation — it's a static-analysis-only type hint) — the extra key is silently added to the state dict and just exists, unused by anything expecting it, which reads as "it worked" until something downstream reads `state["typo_key"]` and gets a `KeyError` far from the actual bug. Reach for a real schema-validation layer (Pydantic `BaseModel` as the state type, which LangGraph also supports) if this class of bug shows up often — it trades a small amount of ceremony for actual runtime errors at the point of the mistake.

- **Graph slower than expected even though nodes should run concurrently?** Sync `.invoke()` inside an async graph node blocks the whole graph's event loop — same underlying issue as the LangChain doc's async pitfall. LangGraph graphs can run nodes concurrently (e.g., via `Send` for map-style fan-out, or independent branches), and a single node doing a blocking synchronous call inside an otherwise-async graph serializes work that should have been parallel. Use `ainvoke`/async node functions consistently within a graph that's driven via `.ainvoke()`/`.astream()`.

- **Copying LangGraph code from a search result?** Version churn is worse here than in plain LangChain. Between the `langgraph==0.0.x`/`0.1.x` era and `1.2.9` (installed here), the prebuilt agent constructor moved packages, the recommended memory pattern solidified around checkpointers, and various graph-construction helpers were renamed. Given this project's global environment doesn't even have `langgraph` installed at all yet (checked directly: `ModuleNotFoundError` before this session's `.venv-langchain` was created), there's no legacy-version baggage here — but *any* LangGraph code copied from search results should be treated as version-suspect until checked against the actually-installed version (`pip show langgraph`), the same discipline documented for LangChain itself.

---

## Practice Q&A (Self-Test)

**Q1. When a node function returns `{"answer": resp.content}`, what happens to the rest of the graph's state? Why does this matter for a 10-node graph?**
A: Nothing happens to it — it survives untouched. The node returns a partial dict, and LangGraph merges it into the running state, key by key, instead of swapping the whole state out. That's what lets each node in a large graph "own" just the keys it actually computes. No node needs to know about, or re-thread, the rest of the state.

**Q2. What does `Annotated[list, add_messages]` actually do? What's the most common "my agent forgot everything" bug tied to it?**
A: Without a reducer, the default merge is a plain overwrite. A node returning `{"messages": [new_msg]}` would replace the whole history with just that one message, silently deleting everything before it. `add_messages` is a reducer that appends instead — it also de-duplicates by message ID and converts plain dicts or tuples into message objects. Forgetting the `Annotated[list, add_messages]` annotation is the most common cause of an agent that looks like it has amnesia after the second turn.

**Q3. In `add_conditional_edges("add_one", should_continue, {"loop": "add_one", "done": END})`, what does the router function return? Why keep that decoupled from the node-name mapping?**
A: The router (`should_continue`) returns a label string, not a node name. A separate dict maps that label to the real node name. Keeping those apart means the same router logic can be reused across graphs where nodes have different names. It's also how a cycle gets built at all — a node pointing back to itself via the mapping, like `"loop": "add_one"`, is just an ordinary entry in that dict.

**Q4. What is the default `recursion_limit`? What exception fires when it's hit? Why is "just raise the number" usually the wrong fix?**
A: The default is 25. Hitting it raises a real, catchable `GraphRecursionError` — not a silent infinite hang. A graph that needs 200 steps to answer a question usually has a routing bug: a condition that should route to `END` more often than it does. Raising the limit doesn't fix that. It just delays the same failure into a longer, more expensive one.

**Q5. Why is `MemorySaver` not real persistence? What's the production alternative?**
A: `MemorySaver` is just an in-process Python dict. A process restart, a redeploy, or even running two separate Python processes loses every thread's history — with zero error signal. `SqliteSaver`, or a Postgres-backed checkpointer, is the production fix. Swapping one in needs no other code changes, since they share the same `BaseCheckpointSaver` interface as `MemorySaver`.

**Q6. What happens if `config={"configurable": {"thread_id": ...}}` is left out of a call to a graph compiled with a checkpointer?**
A: The graph has nothing to look up for that call. It behaves exactly as if no checkpointer were configured at all — silently, with no error. Every call becomes a brand-new conversation, even though a checkpointer is attached.

**Q7. What warning does `langgraph.prebuilt.create_react_agent` print, as of `langgraph==1.2.9`? What should new code import instead?**
A: `LangGraphDeprecatedSinceV10: create_react_agent has been moved to langchain.agents. Please update your import to from langchain.agents import create_agent. Deprecated in LangGraph V1.0 to be removed in V2.0.` It still runs. New code should use `from langchain.agents import create_agent` instead.

**Q8. What are the three `stream_mode` options for `.stream()`? What does each one show?**
A: `"values"` re-sends the full accumulated state after every node — verbose, but simple. `"updates"` shows just what each node returned, keyed by node name — better for a progress UI. `"messages"` gives token-level streaming from inside a node, for chat UIs. Neither `"values"` nor `"updates"` gives you token-by-token text on its own; only `"messages"` mode does.

**Q9. If a node returns a key that isn't declared in the `TypedDict` state schema, does LangGraph or Python catch it at runtime? What's the fix if this bug keeps showing up?**
A: No. `TypedDict` gives you no runtime validation — it's a static-analysis-only type hint. The extra key gets silently added to the state dict and just sits there, unused, until something downstream reads it and gets a `KeyError` far from the actual bug. The fix is using a Pydantic `BaseModel` as the state type instead, which LangGraph also supports. It costs a bit more ceremony, but it turns that mistake into a real runtime error, at the actual point of the mistake.

**Q10. How does `thread_id` in LangGraph relate to `session_id` in LangChain's `RunnableWithMessageHistory`? What deprecation warning connects the two?**
A: `thread_id` is LangGraph's direct equivalent of `session_id`. It's the exact mechanism LangChain 1.x's deprecation warning points to: `RunnableWithMessageHistory is deprecated. Use LangGraph's built-in persistence instead.` A `checkpointer=MemorySaver()` passed at compile time, plus a stable `thread_id` in config, is what gives a graph memory across separate `.invoke()` calls.


---

## Video-Sourced Practice MCQs

A practice set for LangGraph, built the same way as this hub's NCA-GENL community bank: topics checked against a real YouTube LangGraph-interview-prep video, then written up as fully original multiple-choice questions here. The clusters above already cover state/nodes/edges basics, conditional-edge loops, and cross-call memory -- this set goes further into checkpointer/thread_id mechanics, time travel, human-in-the-loop pausing, the multi-agent supervisor pattern, tool-node auto-recovery, fan-out/fan-in parallelism, subgraphs, and configurable run-scoped state.

<script type="application/json" class="topic-quiz-data" data-title="LangGraph Practice">
[
  {
    "d": "Persistence & Threads",
    "q": "When you compile a graph with a checkpointer (e.g. `MemorySaver` or a SQLite saver) and pass a `thread_id` on each call, what does the `thread_id` actually identify?",
    "o": [
      "A unique name for one specific conversation/run — it's how the checkpointer knows which saved state belongs to which user or session, so two different threads never see or overwrite each other's history",
      "The name of the Python thread (in the concurrency sense) the graph happens to execute on",
      "A unique name for a single NODE inside the graph, unrelated to which conversation is running",
      "A required setting that controls how many parallel branches the graph is allowed to run at once"
    ],
    "a": [
      0
    ],
    "e": "A checkpointer can hold many separate conversations' worth of saved state at once — `thread_id` is the key that keeps them apart, the same way a save-file name tells you whose saved game you're loading. Without a consistent `thread_id`, you'd have no way to resume THE SAME conversation later rather than accidentally landing in someone else's. It has nothing to do with OS/Python-level threading (a coincidental name overlap, not the same concept) — LangGraph's `thread_id` is purely an application-level conversation identifier. It doesn't identify a node either — nodes are identified by their own names in the graph definition. And it doesn't configure parallelism — that's controlled by how you structure edges (fan-out), not by the thread ID."
  },
  {
    "d": "Time Travel",
    "q": "LangGraph's \"time travel\" feature lets you look at old versions of the graph's state and even resume execution from one. Concretely, how do you actually rewind to a past point?",
    "o": [
      "Manually re-type the entire conversation history into a brand new state object from scratch every time",
      "Set the graph's `recursion_limit` to a negative number to force it backward through its own execution history",
      "Grab the ID of a specific PAST checkpoint (the checkpointer already snapshotted the state after every step) and tell the graph to resume execution starting from that exact checkpoint",
      "Delete the current thread entirely and start an unrelated new one — there is no way to actually return to a specific past point"
    ],
    "a": [
      2
    ],
    "e": "Because the checkpointer already snapshots state after every single step (that's what persistence/checkpointing IS), rewinding is just a matter of referencing an earlier checkpoint's ID and telling the graph to resume from there — no manual reconstruction needed, since the exact historical state is already saved. Manually retyping history (option 2) defeats the entire purpose of having a checkpointer do this automatically. Deleting the thread (option 3) throws away the very history you'd need to rewind through, the opposite of time travel. And `recursion_limit` is an unrelated safety cap on how many steps/loops a run is allowed to take — it has no 'rewind' behavior, negative or otherwise."
  },
  {
    "d": "Human-in-the-Loop (HITL)",
    "q": "You want the agent to PAUSE before a specific risky node (e.g. one that sends an email) and wait for a human to approve or adjust it before continuing. How do you set this up, and what can the human actually do while it's paused?",
    "o": [
      "There is no way to inspect or change anything during a pause — HITL only supports a plain yes/no approval, with no ability to edit the actual state",
      "Interrupting a node permanently stops that thread's execution forever — there's no way to resume after a HITL pause",
      "Configure the graph to `interrupt` before (or after) that specific node; while paused, a human can inspect the state (`get_state`) and even manually edit it (`update_state`) — e.g. fixing a wrong argument or adding a hint — before the run resumes",
      "You must rewrite the entire graph from scratch to add a pause point after the fact; it cannot be configured on an existing graph"
    ],
    "a": [
      2
    ],
    "e": "HITL in LangGraph is built on the same checkpointing/state machinery as everything else: you configure an `interrupt` before or after the chosen node, which pauses execution and snapshots the state exactly as it stood — a human can then call `get_state` to see what the agent is 'thinking' and `update_state` to directly modify it (correcting an error, injecting a hint, overriding a decision) before letting the run continue from that point. It's a real editing capability, not just an approve/reject toggle. Interrupts are a configuration you set when compiling the graph (or per-call), not something requiring a full rewrite of the graph's structure. And pausing is explicitly resumable — that's the entire point of combining interrupts with a checkpointer; it isn't a permanent stop."
  },
  {
    "d": "Multi-Agent Patterns",
    "q": "In a multi-agent LangGraph system with a \"supervisor\" node coordinating several specialist agent nodes (researcher, writer, etc.), what is the supervisor node's specific job?",
    "o": [
      "It only runs once, at the very start of the graph, and has no further role after the first specialist agent begins",
      "It replaces the state entirely at every step, discarding whatever the previous specialist agent produced",
      "It directly performs all the actual work itself, while the specialist nodes just wait idly for instructions and never execute anything",
      "It looks at the work completed so far (the shared state) and decides which specialist agent node should run NEXT — effectively routing control flow between the other agents based on progress"
    ],
    "a": [
      3
    ],
    "e": "The supervisor acts like a coordinator/dispatcher: it inspects the current shared state (what's been done, what's still needed) and decides which specialist node gets control next — essentially playing the role of a conditional-edge router, but implemented as its own reasoning node rather than a fixed rule. It doesn't do the specialist work itself (that's precisely what the researcher/writer/etc. nodes are for) — its job is routing, not execution. It also isn't meant to blow away prior work — the whole point of shared state in a multi-agent graph is that each specialist's output persists and builds on what came before. And it's not a one-time-only node — it's typically re-invoked repeatedly, after each specialist finishes, to decide the NEXT step, not just the first one."
  },
  {
    "d": "Tool Reliability",
    "q": "LangGraph provides a `ToolNode` that runs whatever tool calls the LLM requested. If the underlying tool call throws an ERROR, what's the typical way to handle it so the agent can recover on its own?",
    "o": [
      "Immediately terminate the entire graph run the instant any tool throws any error, with no possibility of recovery",
      "Send the error message back to the LLM as an observation in the state — the LLM can then read what went wrong and attempt to correct its own tool call on the next step, rather than the whole run simply crashing",
      "Silently discard the error and pretend the tool call succeeded with an empty result, so the agent never learns anything went wrong",
      "Automatically rewrite the tool's own source code to prevent that specific error from ever recurring"
    ],
    "a": [
      1
    ],
    "e": "Feeding the actual error text back into the state as an observation gives the LLM the same kind of feedback a human developer would use to debug — it can see exactly what went wrong (a bad argument type, a missing field) and try a corrected tool call on its next turn, turning a hard failure into a recoverable one. Silently swallowing the error (option 2) hides real problems and lets the agent proceed on false information. Immediately terminating on any error (option 3) throws away this recovery mechanism entirely — it's the less resilient, non-self-correcting approach this pattern specifically improves on. And there's no mechanism (in LangGraph or otherwise) for an LLM tool call to autonomously rewrite the tool's own source code — that's well outside what this error-handling pattern does."
  },
  {
    "d": "Parallel Execution",
    "q": "LangGraph can run independent steps concurrently — described as \"fan-out\" for launching them and \"fan-in\" for what happens afterward. What does \"fan-in\" specifically refer to?",
    "o": [
      "A required manual step where a human explicitly merges the results, since LangGraph cannot combine parallel outputs automatically",
      "The point where those parallel tasks all finish and their individual results get COMBINED back into a single, unified state update",
      "The initial moment when a single task is split out into multiple parallel branches (this describes fan-out, not fan-in)",
      "A safety mechanism that cancels all parallel tasks the instant any single one of them fails"
    ],
    "a": [
      1
    ],
    "e": "Fan-out is launching multiple independent branches to run concurrently (e.g. three unrelated tool calls that don't depend on each other's output); fan-in is the complementary step where the graph waits for all of them to complete and merges their separate results back into one coherent state update before continuing — the two terms describe opposite ends of the same parallel-execution pattern, so mixing them up (option 2 describes fan-out while claiming to define fan-in) is a common confusion. There's no described cancel-on-first-failure safety mechanism baked into fan-in itself — that would be a separate error-handling policy layered on top, not what the term means. And combining results is exactly what the graph's reducer/state-merging machinery handles automatically (the same mechanism from Cluster 1's state-merging discussion) — it does not require manual human merging."
  },
  {
    "d": "Composition",
    "q": "LangGraph supports putting an entire compiled graph inside another graph as if it were a single node — described as building \"a small specific graph\" and using it \"as a single node inside a much bigger system.\" What is this pattern (subgraphs) actually useful for?",
    "o": [
      "It replaces the need for individual nodes entirely — once you use a subgraph, you can no longer add plain function-based nodes to that graph",
      "It's purely a visual/diagramming convenience with no effect on how the graph actually executes at runtime",
      "It only works if the inner graph and outer graph share the exact same state schema, with no ability to define independent internal state",
      "Encapsulating a self-contained, reusable piece of multi-step logic (its own internal state/nodes/edges) behind a single node's worth of interface, so a larger graph can use it without needing to know its internal structure"
    ],
    "a": [
      3
    ],
    "e": "A subgraph lets you build and test a self-contained unit of multi-step behavior (its own nodes, edges, even its own internal state) once, then drop it into a larger system as a single black-box node — the outer graph doesn't need to know or care about the subgraph's internal wiring, just its inputs and outputs, exactly like calling a well-tested function without re-reading its implementation. This is a genuine runtime execution structure, not merely a diagramming convenience — the inner graph actually runs as part of the outer graph's execution. Subgraphs are commonly used specifically BECAUSE they can define their own internal state shape, decoupled from the outer graph's state, rather than being forced to share one schema. And regular nodes remain completely usable alongside subgraph-nodes in the same larger graph — using one doesn't disable the other."
  },
  {
    "d": "Run-Scoped Configuration",
    "q": "LangGraph distinguishes the graph's core STATE (data that flows through and gets updated by nodes) from \"configurable\" values like a user ID or preferred language. Why keep these separate rather than just putting everything into the state?",
    "o": [
      "Configurable values are settings that stay constant for the ENTIRE run and aren't part of the actual step-by-step data/logic nodes operate on — keeping them out of state avoids cluttering every node's data-processing logic with values that never actually change during that run",
      "Configurable values are recalculated fresh by every single node, unlike state which persists — the opposite of how it actually works",
      "Only the state can be checkpointed and persisted; configurable values are always lost the moment a run pauses or restarts",
      "There's no real distinction — configurable values and state are simply two different names for the exact same underlying mechanism"
    ],
    "a": [
      0
    ],
    "e": "State is meant to change as nodes do work — messages get appended, intermediate results get stored, and so on. A setting like 'user_id' or 'preferred language' isn't something any node computes or updates — it's fixed context for the whole run, so treating it as configuration (rather than yet another state field every node has to thread through unchanged) keeps the actual data-flow logic focused on what's genuinely dynamic. They are NOT the same mechanism — that's the entire reason LangGraph exposes them as separate concepts rather than collapsing them into one. Configurable values are the opposite of 'recalculated by every node' — they're static for the run's duration, which is exactly the property that makes them configuration rather than state. And configuration values are available throughout a run including across pauses within that run, not something that's lost the moment a checkpoint pause happens."
  }
]
</script>
<div class="topic-quiz-mount"></div>
