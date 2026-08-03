# Code Drills — Tier 3: LangGraph — StateGraph, Tool-Calling Agents, Memory

Continues `code-drills-finetuning-peft.md` and closes the Code Drills tier: everything before this file makes a model smarter (fine-tuning) or better-informed (RAG); this file makes it *act* — call tools, loop, remember across turns. Terser companion to `langgraph-practice.md`'s narrative chains (same underlying examples, more reps); `practice-langgraph` in the hub picks up from here into deeper multi-agent patterns. Verified in `.venv-llm-rag` (langgraph installed alongside langchain 1.3.14) against the project's real Azure `gpt-4.1-mini` deployment — every graph actually compiled and ran, no mocked LLM calls.

---

## Cluster 1 — StateGraph Basics

> 🔗 **Theory:** [LangGraph Practice — State and Nodes](/topic/practice-langgraph#cluster-1-state-and-nodes-the-two-ideas-everything-else-builds-on)

**1. Define the state a graph will pass between nodes.**
```python
from typing import TypedDict

class State(TypedDict):
    question: str
    answer: str
```

**2. Write a node — a plain function that reads state and returns a PARTIAL update.**
```python
import os
from langchain_openai import AzureChatOpenAI

llm = AzureChatOpenAI(
    azure_deployment="gpt-4.1-mini",
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"].strip(),
    api_key=os.environ["AZURE_OPENAI_KEY"].strip(),
    api_version="2024-06-01", temperature=0,
)

def answer_node(state: State) -> dict:
    resp = llm.invoke(state["question"])
    return {"answer": resp.content}    # a PARTIAL dict — merged into state, not a full replacement
```

**3. Build and compile a minimal one-node graph.**
```python
from langgraph.graph import StateGraph, END

graph = StateGraph(State)
graph.add_node("answer", answer_node)
graph.set_entry_point("answer")
graph.add_edge("answer", END)
app = graph.compile()
```

**4. Run the graph and see how the node's partial return merges into the full state.**
```python
result = app.invoke({"question": "What is a KV cache, in one sentence?"})
result.keys()    # dict_keys(['question', 'answer']) — the INPUT key survives alongside the node's OUTPUT key
```

**5. Chain multiple nodes with linear edges.**
```python
class DraftState(TypedDict):
    topic: str
    draft: str
    polished: str

def draft_node(state: DraftState) -> dict:
    resp = llm.invoke(f"Write one rough sentence about: {state['topic']}")
    return {"draft": resp.content}

def polish_node(state: DraftState) -> dict:
    resp = llm.invoke(f"Polish this sentence, keep it to one sentence: {state['draft']}")
    return {"polished": resp.content}

g = StateGraph(DraftState)
g.add_node("draft", draft_node)
g.add_node("polish", polish_node)
g.set_entry_point("draft")
g.add_edge("draft", "polish")     # draft's OUTPUT state flows into polish's INPUT
g.add_edge("polish", END)
app2 = g.compile()
result2 = app2.invoke({"topic": "gradient descent"})
```

**6. Make a state field APPEND instead of overwrite — the shape every chat agent needs.**
```python
from typing import Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage

class ChatState(TypedDict):
    messages: Annotated[list, add_messages]    # the Annotated reducer is the whole trick

def chat_node(state: ChatState) -> dict:
    resp = llm.invoke(state["messages"])
    return {"messages": [resp]}    # returning ONE new message — add_messages appends it, doesn't replace the list

g3 = StateGraph(ChatState)
g3.add_node("chat", chat_node)
g3.set_entry_point("chat")
g3.add_edge("chat", END)
chat_app = g3.compile()
r1 = chat_app.invoke({"messages": [HumanMessage("My name is Sam.")]})
len(r1["messages"])    # 2: the human message + the AI reply
# WITHOUT the Annotated[list, add_messages] reducer, this same return would REPLACE the whole history
# with a single message — the single most common "why did my agent forget everything" bug
```

**7. Visualize the compiled graph's structure.**
```python
print(app2.get_graph().draw_mermaid())    # a Mermaid diagram string — paste into any Mermaid renderer,
                                             # or render directly if your environment supports it
```

**8. Stream intermediate state after EACH node, not just the final result.**
```python
for step in app2.stream({"topic": "backpropagation"}):
    print(step)    # {'draft': {...}} then separately {'polish': {...}} — one dict per node as it completes
# useful for showing a UI "thinking..." progress indicator, or debugging which node produced what
```

---

## Cluster 2 — Conditional Edges & Control Flow

> 🔗 **Theory:** [LangGraph Practice — Control Flow](/topic/practice-langgraph#cluster-2-control-flow-conditional-edges-and-runaway-recursion)

**9. Route to a different node based on the current state.**
```python
class RouteState(TypedDict):
    question: str
    category: str
    answer: str

def classify_node(state: RouteState) -> dict:
    q = state["question"].lower()
    category = "math" if any(c.isdigit() for c in q) else "general"
    return {"category": category}

def math_node(state: RouteState) -> dict:
    return {"answer": f"[math path] {llm.invoke(state['question']).content}"}

def general_node(state: RouteState) -> dict:
    return {"answer": f"[general path] {llm.invoke(state['question']).content}"}

def route(state: RouteState) -> str:              # a router: reads state, returns the NEXT NODE'S NAME
    return "math" if state["category"] == "math" else "general"

g4 = StateGraph(RouteState)
g4.add_node("classify", classify_node)
g4.add_node("math", math_node)
g4.add_node("general", general_node)
g4.set_entry_point("classify")
g4.add_conditional_edges("classify", route, {"math": "math", "general": "general"})
g4.add_edge("math", END)
g4.add_edge("general", END)
router_app = g4.compile()
```

**10. Confirm the router actually sends different inputs down different paths.**
```python
r1 = router_app.invoke({"question": "What is 5 + 7?", "category": "", "answer": ""})
r2 = router_app.invoke({"question": "What is a transformer?", "category": "", "answer": ""})
r1["answer"].startswith("[math path]"), r2["answer"].startswith("[general path]")   # (True, True)
```

**11. Build a loop — an edge back to an earlier node, not just forward progress.**
```python
class RetryState(TypedDict):
    attempts: int
    done: bool

def try_node(state: RetryState) -> dict:
    attempts = state["attempts"] + 1
    return {"attempts": attempts, "done": attempts >= 3}    # "succeeds" on the 3rd attempt

def should_continue(state: RetryState) -> str:
    return "end" if state["done"] else "retry"

g5 = StateGraph(RetryState)
g5.add_node("try", try_node)
g5.set_entry_point("try")
g5.add_conditional_edges("try", should_continue, {"retry": "try", "end": END})   # "retry" points BACK to "try"
retry_app = g5.compile()
result = retry_app.invoke({"attempts": 0, "done": False})
result["attempts"]    # 3
```

**12. Guard against a loop that never terminates — LangGraph's built-in circuit breaker.**
```python
from langgraph.errors import GraphRecursionError

try:
    retry_app.invoke({"attempts": 0, "done": False}, {"recursion_limit": 2})   # cap steps at 2
except GraphRecursionError as e:
    print("hit recursion limit:", e)
# a bug where should_continue() never returns "end" would otherwise loop FOREVER — recursion_limit
# is the safety net, and a real LangGraph agent should always have one set deliberately, not left at the default
```

**13. Branch into more than two paths from one conditional edge.**
```python
def route_three_way(state: RouteState) -> str:
    q = state["question"].lower()
    if "code" in q: return "code"
    if any(c.isdigit() for c in q): return "math"
    return "general"

# add_conditional_edges takes a dict mapping EVERY possible router return value to a real node name —
# g.add_conditional_edges("classify", route_three_way, {"code": "code_node", "math": "math", "general": "general"})
```

**14. Know what `START`/`END` actually are — not ordinary nodes, sentinel markers.**
```python
from langgraph.graph import START, END
# graph.add_edge(START, "answer")  is equivalent to  graph.set_entry_point("answer")
# graph.add_edge("answer", END)     marks a node as a terminal point — the graph stops there
```

---

## Cluster 3 — Tool-Calling Agents

> 🔗 **Theory:** [LangChain Practice — Giving the Model Tools](/topic/practice-langchain#cluster-3-giving-the-model-tools-and-actually-running-them)

**15. Define a tool the LLM can choose to call.**
```python
from langchain_core.tools import tool

@tool
def get_word_length(word: str) -> int:
    """Return the number of characters in a word."""     # the docstring IS the tool description the LLM sees
    return len(word)
```

**16. Bind tools to an LLM so it knows they exist and can request them.**
```python
llm_with_tools = llm.bind_tools([get_word_length])
```

**17. See the LLM REQUEST a tool call — it does not execute anything itself.**
```python
response = llm_with_tools.invoke("How many letters are in the word 'bureaucracy'?")
response.tool_calls
# [{'name': 'get_word_length', 'args': {'word': 'bureaucracy'}, 'id': '...'}]
# the LLM only produces the REQUEST (name + args) — nothing has actually run yet, response.content is
# typically empty here. Executing the function is a separate step, entirely outside the LLM.
```

**18. Actually execute a requested tool call and feed the result back.**
```python
from langchain_core.messages import ToolMessage

tool_call = response.tool_calls[0]
result = get_word_length.invoke(tool_call["args"])       # actually runs the Python function: 11
tool_message = ToolMessage(content=str(result), tool_call_id=tool_call["id"])
# tool_call_id MUST match — it's how the model correlates this result back to the specific call it made,
# especially when multiple tools were called in the same turn (drill #21)
```

**19. Build a `ToolNode` — LangGraph's built-in node that executes tool calls automatically.**
```python
from langgraph.prebuilt import ToolNode

tool_node = ToolNode([get_word_length])
# saves hand-writing drill #18's execute-and-wrap-in-ToolMessage logic yourself for every tool
```

**20. Route between "call a tool" and "just respond" based on whether the LLM asked for one.**
```python
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

def agent_node(state: AgentState) -> dict:
    resp = llm_with_tools.invoke(state["messages"])
    return {"messages": [resp]}

def has_tool_calls(state: AgentState) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else "end"

g6 = StateGraph(AgentState)
g6.add_node("agent", agent_node)
g6.add_node("tools", tool_node)
g6.set_entry_point("agent")
g6.add_conditional_edges("agent", has_tool_calls, {"tools": "tools", "end": END})
g6.add_edge("tools", "agent")     # after running tools, go BACK to the agent so it can use the result
agent_app = g6.compile()
```

**21. Run the full ReAct-style loop: reason -> act (tool) -> observe -> reason again -> final answer.**
```python
result = agent_app.invoke({"messages": [HumanMessage("How many letters are in 'bureaucracy'?")]})
result["messages"][-1].content    # the model's FINAL answer, after seeing the tool's result
len(result["messages"])            # typically 4: human question, AI tool-call request, tool result, AI final answer
```

**22. Handle a turn where the LLM requests MULTIPLE tool calls at once.**
```python
@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b

llm_multi = llm.bind_tools([add, multiply])
resp = llm_multi.invoke("What is 3 + 4, and separately what is 6 * 7?")
len(resp.tool_calls)    # can be 2 — a single LLM turn can request several independent tool calls at once;
                          # ToolNode (drill #19) executes all of them and returns one ToolMessage per call
```

---

## Cluster 4 — Memory & Checkpointing

> 🔗 **Theory:** [LangGraph Practice — Memory](/topic/practice-langgraph#cluster-3-memory-persisting-state-across-separate-calls)

**23. Add a checkpointer so state persists ACROSS separate `.invoke()` calls.**
```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
persistent_app = g3.compile(checkpointer=checkpointer)   # reusing ChatState/chat_node from drill #6
```

**24. Use a `thread_id` to keep separate conversations from bleeding into each other.**
```python
config = {"configurable": {"thread_id": "user-42"}}
r1 = persistent_app.invoke({"messages": [HumanMessage("My name is Sam.")]}, config)
r2 = persistent_app.invoke({"messages": [HumanMessage("What's my name?")]}, config)
r2["messages"][-1].content    # correctly recalls "Sam" — the checkpointer restored thread "user-42"'s
                                 # history BEFORE this call ran, without you re-passing r1's messages manually
```

**25. Confirm a DIFFERENT `thread_id` gets a clean slate, not shared history.**
```python
config_other = {"configurable": {"thread_id": "user-99"}}
r3 = persistent_app.invoke({"messages": [HumanMessage("What's my name?")]}, config_other)
# the model has no idea — "user-99" never saw the "My name is Sam" message; threads are fully isolated
```

**26. Inspect the saved state history for a thread — useful for debugging or building a "conversation log" UI.**
```python
history = list(persistent_app.get_state_history(config))
len(history)    # one snapshot per graph step that ran under this thread_id, most recent first
```

---

## Cluster 5 — Judgment Calls

**27. Know when a plain LCEL chain (`code-drills-rag-langchain.md`) is enough, and when you actually need LangGraph.**
```python
# plain LCEL chain (prompt | llm | parser): fine for a FIXED, linear pipeline — same steps, every time,
#   no branching, no memory needed beyond what you pass in manually
# LangGraph: reach for it the moment you need any of — conditional branching (Cluster 2), a loop/retry
#   (drill #11), tool-calling with the model deciding IF a tool is needed (Cluster 3), or state that
#   persists across separate calls without manual re-threading (Cluster 4). Building a chain that keeps
#   needing "if" statements between LCEL steps is usually the signal it's time to switch.
```

**28. Recognize the most common LangGraph bug, tying Clusters 1 and 2 together.**
```python
# symptom: agent seems to "forget" earlier messages, or a loop runs forever until GraphRecursionError
# cause #1 (drill #6): a list-valued state field declared as plain `list` instead of
#   `Annotated[list, add_messages]` — every node's return SILENTLY overwrites history instead of appending
# cause #2 (drill #12): a conditional edge's router function has a bug where it never returns the "end"
#   branch — always set an explicit recursion_limit so this fails loudly and fast instead of hanging
```

---

**This closes the Code Drills tier's LLM-systems half.** Full path: `code-drills-basics.md` → `code-drills-data-structures.md` → `code-drills-oop-intermediate.md` → `code-drills-numpy-pandas.md` → `code-drills-classical-ml.md` → `code-drills-deep-learning.md` → `code-drills-llm-huggingface.md` → `code-drills-rag-langchain.md` → `code-drills-finetuning-peft.md` → this file. For deeper LLM-systems material past this point, `core-technical-depth.md`, `rag-deeper.md`, `prompt-engineering-deeper.md`, and `practice-langgraph`'s own narrative docs pick up exactly where these reps leave off.
