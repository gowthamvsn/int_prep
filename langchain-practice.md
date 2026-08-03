# LangChain Practice — Built as a Chain, Not a List

Same format as the rest of this hub: **question → code → why it matters**. Every snippet was actually executed in an isolated venv (`D:\nvidia\.venv-langchain`) against the same Azure OpenAI deployment (`gpt-4.1-mini`) this project already uses in `server.py` — not just written and assumed correct. Installed versions: `langchain==1.3.14`, `langchain-core==1.4.9`, `langchain-openai==1.3.5`, `langgraph==1.2.9`. **Note the gap:** this machine's *global* Python env has `langchain==0.3.7` (older) — LangChain's 0.3→1.x jump changed real behavior (see the memory cluster below), so code copied from a 0.3.x tutorial can silently do the wrong thing on 1.x, and vice versa. Always check `pip show langchain` before trusting a snippet from memory or an old blog post. Each cluster is one continuous thread — every question inherits the answer before it, closing with a worked summary example.

---

## Cluster 1 — Calling a Model and Composing It With LCEL

### 1. What's the actual minimal way to call a chat model?
```python
import os
from langchain_openai import AzureChatOpenAI

# langchain_openai looks for AZURE_OPENAI_API_KEY by default -- this project's .env uses
# AZURE_OPENAI_KEY (matching server.py), so pass credentials explicitly rather than relying
# on the auto-detected env var names.
llm = AzureChatOpenAI(
    azure_deployment="gpt-4.1-mini",                              # matches server.py's AZURE_OPENAI_DEPLOYMENT
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"].strip(),    # .strip(): the .env value has a leading space
    api_key=os.environ["AZURE_OPENAI_KEY"].strip(),
    api_version="2024-06-01",
    temperature=0,
)
response = llm.invoke("Name one advantage of LoRA over full fine-tuning, in one sentence.")
print(response.content)          # AIMessage.content -- the string; response itself carries usage metadata too
```
`AzureChatOpenAI` and not `ChatOpenAI`: `langchain_openai` ships two separate classes because Azure's auth/routing (endpoint + deployment name, not just a model string) is genuinely different from OpenAI's API — using the wrong one is the single most common "why won't this connect" issue for anyone coming from OpenAI's own docs. **A second, verified-the-hard-way gotcha:** `AzureChatOpenAI` raises `openai.OpenAIError: Missing credentials` if it can't find `AZURE_OPENAI_API_KEY` in the environment — it does NOT know about this project's differently-named `AZURE_OPENAI_KEY` var, so the fix is passing `api_key=`/`azure_endpoint=` explicitly rather than renaming this project's `.env` to match LangChain's expected names.

### 2. Given a bare `llm.invoke()` call works, how do you attach a reusable PROMPT template to it (LCEL)?
```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a terse NVIDIA NCA-GENL exam tutor. Answer in <=2 sentences."),
    ("human", "{question}"),
])
chain = prompt | llm          # the `|` pipe is LCEL: compose Runnables like Unix pipes
result = chain.invoke({"question": "What does the temperature parameter control?"})
print(result.content)
```
The `|` operator, specifically: every LCEL component (`prompt`, `llm`, output parsers, retrievers) implements the same `Runnable` interface (`.invoke`/`.batch`/`.stream`/`.ainvoke`), so `|` just composes them into a pipeline — that uniformity is *why* streaming, batching, and async all fall out for free on any chain you build this way, instead of needing separate code paths for each.

**Visual + memory hook — read `|` exactly like a Unix shell pipe, because it's the same idea, not just similar-looking syntax:**
```
{"question": "..."}  ──▶  [ prompt ]  ──▶  [ llm ]  ──▶  [ output parser ]  ──▶  result
   plain dict              ChatPromptValue    AIMessage        str/dict

Unix, for comparison:
  cat file.txt  ──▶  [ grep "x" ]  ──▶  [ sort ]  ──▶  [ uniq ]  ──▶  output
```
**Remember it as `ls | grep | sort`, just for LLM components instead of shell commands** — every box is something that takes ONE typed input and produces ONE typed output, and `|` only works because every box (`Runnable`) speaks the identical `.invoke()` interface, the same way every Unix pipe stage reads stdin and writes stdout regardless of what the program actually does inside. That's also the answer to "why do streaming/batching/async all just work on any chain" — you're not getting three separate features, you're getting one property (uniform interface) paying off three ways at once, exactly like every Unix pipeline automatically supporting `| head` or backgrounding with `&` without each command needing to implement that itself.

### 3. Given the chain returns free text, how do you get STRUCTURED output back instead?
```python
from pydantic import BaseModel, Field

class ExamAnswer(BaseModel):
    answer: str = Field(description="the direct answer, one sentence")
    confidence: float = Field(description="0-1, how sure the model is")

structured_llm = llm.with_structured_output(ExamAnswer)
out = structured_llm.invoke("What does top_p sampling control?")
print(out.answer, out.confidence)          # out is an ExamAnswer instance, not a string -- no manual JSON parsing
```
`with_structured_output` over asking for JSON in the prompt and parsing it yourself: it uses the provider's native tool-calling/JSON-mode under the hood, so the model is constrained at generation time rather than hoping it obeys a text instruction — the older `PydanticOutputParser` (prompt-based, parse-and-hope) still exists and still fails occasionally on malformed JSON; the tool-calling route practically doesn't.

### 4. Given any chain built from `|` speaks the same `Runnable` interface (question 2), how do you swap `.invoke()` for STREAMING without rewriting anything?
```python
for chunk in chain.stream({"question": "List 3 GenAI eval metrics, briefly."}):
    print(chunk.content, end="", flush=True)     # each chunk is a partial AIMessageChunk
```
Why it matters for a UI like this project's `doc_template.html` panel: `.stream()` and `.invoke()` are the *same* chain — you don't rewrite anything to add streaming, you just call a different Runnable method, exactly the payoff question 2's visual predicted. This project's own `/api/ask` currently uses a plain blocking `requests.post` to Azure, not this — swapping in `.stream()` here would be the natural way to make the "Thinking…" panel fill in token-by-token instead of waiting for the whole answer.

### Summary example
A tutoring chain built as `prompt | llm` (question 2) can return either free text via `.invoke()` or token-by-token via `.stream()` (question 4) with ZERO code changes to the chain itself, because both methods are just different faces of the same `Runnable` interface — and if the answer needs to be machine-parseable instead of prose, swapping to `.with_structured_output(ExamAnswer)` (question 3) gets a validated Pydantic object back instead of a string, still built on the exact same `AzureChatOpenAI` client from question 1.

---

## Cluster 2 — Running Steps in Parallel and Surviving Failures

### 1. Given two independent chain calls both need to run, how do you run them in PARALLEL instead of one after another?
```python
from langchain_core.runnables import RunnableParallel

summary_prompt = ChatPromptTemplate.from_template("Summarize in 5 words: {question}")
parallel = RunnableParallel(answer=chain, summary=summary_prompt | llm)
out = parallel.invoke({"question": "What is RLHF?"})
print(out["answer"].content, "|", out["summary"].content)   # both branches ran concurrently, not sequentially
```
Why this beats two separate `.invoke()` calls: `RunnableParallel` fires both branches concurrently (via threads for sync `.invoke`, real async for `.ainvoke`) and returns a single dict keyed by branch name — for two independent LLM calls this roughly halves wall-clock latency versus awaiting them one after another.

### 2. Given a chain now runs reliably when the API is healthy, how do you handle it when a call fails TRANSIENTLY (rate limit, timeout) versus when the whole deployment is DOWN?
```python
fallback_llm = AzureChatOpenAI(
    azure_deployment="gpt-4.1-mini",
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"].strip(),
    api_key=os.environ["AZURE_OPENAI_KEY"].strip(),
    api_version="2024-06-01", temperature=0.7,
)
robust_llm = llm.with_retry(stop_after_attempt=3).with_fallbacks([fallback_llm])
print(robust_llm.invoke("Say OK").content)
```
Why both, and in this order: `with_retry` handles *transient* failures (rate limits, timeouts) by retrying the *same* model with backoff; `with_fallbacks` handles the case where retries are exhausted or the model itself is down, by trying a *different* Runnable entirely. Skipping `with_retry` and going straight to fallback wastes a working model's capacity on a blip; skipping fallback leaves you with no answer at all when a deployment has a real outage.

### Summary example
A production endpoint calling two independent prompts (a full answer and a 5-word summary) wraps both in `RunnableParallel` (question 1) to halve latency, and wraps the whole thing in `.with_retry().with_fallbacks([...])` (question 2) so a transient rate-limit gets retried on the same deployment first, and only a genuine outage falls through to a backup deployment — resilience and speed addressed as two separate, stackable concerns rather than one bundled fix.

---

> 🔗 **Hands-on reps:** [Code Drills 10 — Tool-Calling Agents](/topic/code-drills-langgraph-agents#cluster-3-tool-calling-agents)

## Cluster 3 — Giving the Model Tools, and Actually Running Them

### 1. How do you give the model a callable tool it can choose to invoke?
```python
from langchain_core.tools import tool

@tool
def exam_day_countdown(target_date: str) -> str:
    """Given an ISO date (YYYY-MM-DD), return how many days remain until it."""
    from datetime import date
    d = date.fromisoformat(target_date) - date.today()
    return f"{d.days} days remaining"

llm_with_tools = llm.bind_tools([exam_day_countdown])
resp = llm_with_tools.invoke("How many days until 2026-07-13?")
print(resp.tool_calls)     # [{'name': 'exam_day_countdown', 'args': {'target_date': '2026-07-13'}, 'id': '...'}]
```
Why the docstring on `exam_day_countdown` isn't optional: `@tool` turns the docstring into the tool's *description* in the schema sent to the model — the model decides whether/how to call the tool based on that text alone. A vague or missing docstring is the #1 reason a model silently never calls a tool it technically has access to, or calls it with wrong argument types.

### 2. Given the model just REQUESTED a tool call (question 1's `tool_calls` list), does LangChain execute it automatically?
```python
from langchain_core.messages import HumanMessage

messages = [HumanMessage("How many days until 2026-07-13?")]
ai_msg = llm_with_tools.invoke(messages)
messages.append(ai_msg)
for call in ai_msg.tool_calls:
    result = exam_day_countdown.invoke(call["args"])          # run the actual Python function
    messages.append({"role": "tool", "content": result, "tool_call_id": call["id"]})
final = llm_with_tools.invoke(messages)                        # model sees the tool's result, answers in prose
print(final.content)
```
No — `bind_tools` only gets you the model's *request* to call a tool; LangChain does not execute your function for you. This manual loop (request, execute, feed result back, repeat) is worth seeing once because every agent framework, including LangGraph's `create_react_agent` (`langgraph-practice.md`), is this exact loop wrapped in a driver. Understanding it is what makes agent bugs debuggable instead of magic.

### Summary example
Asking "how many days until 2026-07-13" triggers `bind_tools` (question 1) to produce a `tool_calls` request rather than an answer; the manual loop (question 2) then actually runs `exam_day_countdown`, appends its result as a `"role": "tool"` message, and calls the model a SECOND time so it can turn the raw number into prose — three separate model-adjacent steps (request, execute, re-invoke) that a framework like LangGraph would hide behind one driver call, but which are worth tracing by hand at least once.

---

> 🔗 **Hands-on reps:** [Code Drills 8 — Wiring Retrieval Into LangChain](/topic/code-drills-rag-langchain#cluster-3-wiring-retrieval-into-langchain)

## Cluster 4 — Building RAG: From Raw Text to a Grounded Answer

### 1. How do you build a minimal retrieval store end to end, starting from a raw document?
```python
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.embeddings import DeterministicFakeEmbedding   # stand-in: swap for AzureOpenAIEmbeddings if you have an embeddings deployment
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

docs = [Document(page_content=open("numpy-practice.md", encoding="utf-8").read())]
chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)

embeddings = DeterministicFakeEmbedding(size=384)
store = InMemoryVectorStore.from_documents(chunks, embeddings)
retriever = store.as_retriever(search_kwargs={"k": 3})

hits = retriever.invoke("how do you reshape an array without copying")
print(len(hits), hits[0].page_content[:80])
```
`chunk_overlap` matters as much as `chunk_size`: a hard cut at exactly `chunk_size` characters can slice a sentence (or a code block) in half between two chunks, so the answer to a question ends up split across chunks with neither one scoring highly enough to retrieve — `chunk_overlap` duplicates a slice of text at each boundary specifically so a concept that straddles a cut still appears whole in at least one chunk. (`DeterministicFakeEmbedding` here stands in for a real embedding model purely so this snippet runs with zero cost/API dependency — swap in `AzureOpenAIEmbeddings(azure_deployment=...)` for real semantic search; this project's `.env` currently only has a chat deployment configured, not an embeddings one.)

### 2. Given a retriever that returns relevant chunks, how do you actually wire it into a FULL chain that retrieves, stuffs context into a prompt, and generates a grounded answer?
```python
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

rag_prompt = ChatPromptTemplate.from_template(
    "Answer using ONLY the context below. If the answer isn't in it, say so.\n\nContext:\n{context}\n\nQuestion: {question}"
)

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)
print(rag_chain.invoke("how do you reshape an array without copying data?"))
```
`RunnablePassthrough()` here: the dict on the left is itself a `Runnable` (a `RunnableParallel` shorthand, from Cluster 2) — it needs to produce *both* `context` (via the retriever) *and* `question` (the original string, untouched) for the prompt template. `RunnablePassthrough` is the identity function as a Runnable: "whatever came into this chain, put it here unchanged." Forgetting it is why a first attempt at this pattern often throws a `KeyError` on `question` — the dict only had `context` in it.

### Summary example
A question like "how do you reshape an array without copying data" flows through the full chain built across both questions: the retriever (question 1) finds the 3 most relevant chunks from `numpy-practice.md` (using the overlap discipline that keeps split concepts intact), `RunnablePassthrough` (question 2) carries the original question through UNCHANGED alongside those chunks so both land in the `rag_prompt` template together, and the model is instructed to answer only from that context — the same "chunk → embed → retrieve → augment → generate" pipeline from `core-technical-depth.md`'s RAG section, just expressed as one LCEL chain instead of six separate manual steps.

---

## Cluster 5 — Memory Across Turns, and Seeing What a Chain Actually Did

### 1. How do you keep a running conversation (memory) across multiple turns?
```python
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

store = {}
def get_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a study assistant."),
    ("placeholder", "{history}"),
    ("human", "{question}"),
])
chain_with_history = RunnableWithMessageHistory(
    chat_prompt | llm, get_history,
    input_messages_key="question", history_messages_key="history",
)
cfg = {"configurable": {"session_id": "user-1"}}
chain_with_history.invoke({"question": "My name is Gowtham."}, config=cfg)
r = chain_with_history.invoke({"question": "What's my name?"}, config=cfg)
print(r.content)      # answers "Gowtham" -- second call sees the first turn via get_history
```
`session_id` lives in `config`, not the input dict: separating "what the user is asking" (input) from "which conversation this belongs to" (config) is what lets the *same* chain object safely serve many concurrent users/sessions without them bleeding into each other. **Verified live, and worth flagging loudly:** running this exact snippet on `langchain-core==1.4.9` prints `LangChainDeprecationWarning: RunnableWithMessageHistory is deprecated. Use LangGraph's built-in persistence instead.` — it still works, but LangChain 1.x has explicitly moved the "recommended" way to do memory out of `langchain` entirely and into LangGraph's checkpointer (`MemorySaver`/`SqliteSaver` + `thread_id` — see `langgraph-practice.md`). This is the single biggest structural change between the 0.3.x this machine's global env has and the 1.x this venv has: memory/persistence is no longer LangChain's job.

### 2. Given a chain now has memory across turns, how do you see what it's actually DOING at each step, for debugging?
```python
from langchain_core.tracers import ConsoleCallbackHandler

chain.invoke(
    {"question": "What is quantization? One sentence."},
    config={"callbacks": [ConsoleCallbackHandler()]},
)   # prints every intermediate Runnable's input/output to stdout, colored by step
```
**Verified the hard way — `set_verbose(True)` (the commonly-cited approach) produced NO output at all** against a plain `prompt | llm` LCEL chain in this venv; it's a holdover from the legacy `Chain` class hierarchy and doesn't hook into LCEL's `Runnable` execution. `ConsoleCallbackHandler` passed via `config={"callbacks": [...]}` is what actually works for LCEL — note it's passed through the SAME `config=` dict that `session_id` lives in (question 1), since both are per-invocation metadata, not part of the actual input. This is the free, zero-setup version of full LangSmith tracing (`os.environ["LANGCHAIN_TRACING_V2"]="true"` + `LANGCHAIN_API_KEY`, for a persistent shareable trace UI instead of stdout). If a debugging snippet you found online uses `set_verbose`/`langchain.debug=True` and produces nothing, this is why.

### Summary example
A multi-turn chain built with `RunnableWithMessageHistory` (question 1) correctly remembers "My name is Gowtham" across two calls scoped to `session_id="user-1"` in `config` — and when a THIRD user's answer looks wrong, `ConsoleCallbackHandler` passed through that same `config` dict (question 2) reveals exactly which intermediate Runnable produced the bad output, without needing `set_verbose` (which silently does nothing on LCEL chains) or a full LangSmith setup just to debug one call.

---

## Common issues & pitfalls (in detail)

**Import paths break across versions — this is the #1 friction point.** Pre-0.1, everything lived under `langchain.*` (`langchain.chat_models.ChatOpenAI`, `langchain.llms.OpenAI`). As of 0.1+, provider integrations moved to separate packages: `langchain_openai.ChatOpenAI`/`AzureChatOpenAI`, `langchain_community` for community-maintained integrations, `langchain_core` for the base abstractions (`Runnable`, message types, prompts). A huge fraction of "ImportError" questions online are someone following a pre-0.1 tutorial against a 0.3.x install. Fix: always check which package a class actually lives in for the installed version (`pip show langchain langchain-core langchain-openai`), not by tutorial vintage.

**Silent context-window truncation.** If you stuff too much retrieved context (or too long a chat history) into a prompt, some providers truncate silently or return a degraded answer rather than a clear error — LangChain does not enforce token budgets for you by default. `.get_num_tokens()` (on the model) or a `tiktoken`-based counter before sending is the only reliable guard; retrievers should cap `k` and text splitters should cap `chunk_size` with the model's real context window in mind, not an arbitrary number.

**Retriever returning technically-matched-but-useless chunks.** Cosine similarity on embeddings finds *lexically or semantically similar* text, not necessarily the text that actually answers the question — a chunk can score high because it shares vocabulary while the real answer sits in a neighboring, lower-scoring chunk. This is why `chunk_overlap` (above) and `k` (how many chunks to pull) both need tuning per-corpus, and why production RAG systems often add a reranking step (cross-encoder) after the initial vector search rather than trusting top-k blindly.

**Unbounded memory growth.** `InMemoryChatMessageHistory`/`ConversationBufferMemory`-style memory keeps every message forever by default — in a long-running session this means every subsequent call re-sends the *entire* history, which quietly grows both latency and API cost per turn, and eventually blows the context window outright. `ConversationSummaryMemory`/`trim_messages` (cap by token count, keep-last-N) are the standard fix; know which one a codebase is actually using before assuming "memory" is free.

**Agents that never terminate, or terminate for the wrong reason.** A tool-calling loop (manual, or via an agent executor) only stops when the model stops requesting tools — if a tool's output is confusing to the model, or a tool call keeps failing, the model can keep re-calling it indefinitely. Every agent driver needs an explicit `max_iterations`/`recursion_limit`; treating that as "just a safety net that won't fire" is wrong — it fires more often than expected once tools can fail or return ambiguous results.

**`temperature=0` is not the same as deterministic.** Even at temperature 0, most hosted APIs are not bit-for-bit reproducible across calls (different hardware paths, provider-side batching, minor floating-point nondeterminism) — a test suite that asserts exact string equality on LLM output will flake. Assert on structure (does it parse as valid JSON, does it contain an expected substring/tool call) instead of exact text.

**Blocking calls inside an async context.** `.invoke()` is synchronous; calling it inside an `async def` (e.g., inside a FastAPI/Flask-async route, or inside another chain's async execution) blocks the entire event loop for the duration of the network call. Use `.ainvoke()`/`.astream()` (every Runnable has async variants) anywhere the surrounding code is already async — mixing sync `.invoke()` into an async app is a common cause of a server that becomes unresponsive to all other requests during an LLM call.

**Prompt injection via retrieved documents, not just user input.** In a RAG system, the "context" fed to the model comes from documents you didn't necessarily vet at generation time — if those documents are ever user-uploaded or scraped from the web, they can contain text instructing the model to ignore its system prompt. Treating retrieved context as trusted just because it came from your own vector store (rather than "the user typed it") is a real, underappreciated attack surface — the fix is the same as any prompt injection defense: clear system/user role separation, and never let retrieved content carry instructions the model treats as higher-privilege than the system prompt.

**Deprecated chain classes still importable, still wrong to use.** `LLMChain`, `ConversationChain`, `SimpleSequentialChain` still exist in 0.3.x for backward compatibility but are explicitly legacy — they predate LCEL, don't compose with `|`, and don't get streaming/batching/async for free. If a snippet (or an older Stack Overflow answer) uses `LLMChain(llm=llm, prompt=prompt)`, the direct LCEL equivalent is `prompt | llm` — functionally similar, but only the LCEL form gets everything covered in the streaming/parallel/fallback snippets above.

**`langchain-community` is being sunset — verified via its own import warning.** `import langchain_community` on the versions installed here (`langchain-community==0.4.2`) prints `DeprecationWarning: langchain-community is being sunset and is no longer actively maintained` at import time. Community-maintained integrations (many loaders, some vectorstores) are migrating to standalone packages (e.g. `langchain-chroma`, `langchain-postgres`) — if a project pins `langchain-community` and hasn't touched it in a while, check whether the specific integration it uses has a dedicated package now before adding new code against the community one.

**Memory/persistence moved out of LangChain itself in 1.x.** As shown above, `RunnableWithMessageHistory` now emits a deprecation warning pointing at LangGraph's checkpointer. Anything built fresh against 1.x should default to a LangGraph-based agent/graph with a checkpointer for multi-turn state, rather than layering `RunnableWithMessageHistory` onto a plain LCEL chain — the plain-chain approach still works today but is explicitly the deprecated path.

**Vectorstore persistence is easy to get wrong.** `InMemoryVectorStore` (used above) and a default-configured `Chroma`/`FAISS` instance both live in process memory unless you explicitly persist to disk — restart the process and the index is gone, silently, with no error; the next query just runs against an empty store. If a RAG feature "stopped finding anything" after a deploy or restart, an unpersisted vectorstore is the first thing to check.

---

## Practice Q&A (Self-Test)

**Q1. Why does this project use `AzureChatOpenAI` instead of `ChatOpenAI`, and what specific error did the file document when the wrong environment variable name was used for credentials?**
A: `AzureChatOpenAI` exists as a separate class because Azure's auth/routing (endpoint + deployment name, not just a model string) is genuinely different from OpenAI's own API. Verified in the file: `AzureChatOpenAI` raises `openai.OpenAIError: Missing credentials` if it can't find `AZURE_OPENAI_API_KEY` in the environment — it doesn't know about this project's differently-named `AZURE_OPENAI_KEY`, so credentials must be passed explicitly via `api_key=`/`azure_endpoint=`.

**Q2. What makes the `|` (pipe) operator in LCEL work, and what capabilities "fall out for free" as a result?**
A: Every LCEL component (`prompt`, `llm`, output parsers, retrievers) implements the same `Runnable` interface (`.invoke`/`.batch`/`.stream`/`.ainvoke`), so `|` just composes them into a pipeline. That uniformity is why streaming, batching, and async work on any chain built this way, without separate code paths for each.

**Q3. Why does `llm.with_structured_output(ExamAnswer)` work more reliably than asking for JSON in the prompt and parsing it yourself?**
A: `with_structured_output` uses the provider's native tool-calling/JSON-mode under the hood, constraining the model at generation time rather than hoping it obeys a text instruction. The older `PydanticOutputParser` approach (prompt-based, parse-and-hope) still exists and still occasionally fails on malformed JSON.

**Q4. According to the file, what is `RunnableWithMessageHistory`'s current status in LangChain 1.x, and what should new multi-turn code use instead?**
A: Running `RunnableWithMessageHistory` on `langchain-core==1.4.9` prints `LangChainDeprecationWarning: RunnableWithMessageHistory is deprecated. Use LangGraph's built-in persistence instead.` It still works, but new code should default to LangGraph's checkpointer (`MemorySaver`/`SqliteSaver` + `thread_id`) rather than layering this onto a plain LCEL chain.

**Q5. Why did `set_verbose(True)` fail to show any output when debugging a plain `prompt | llm` LCEL chain, and what actually worked?**
A: `set_verbose(True)` is a holdover from the legacy `Chain` class hierarchy and doesn't hook into LCEL's `Runnable` execution, so it produced no output at all. `ConsoleCallbackHandler` passed via `config={"callbacks": [...]}` is what actually works for LCEL, printing every intermediate Runnable's input/output.

**Q6. Why is `RunnablePassthrough()` needed in the RAG chain's input dict (`{"context": retriever | format_docs, "question": RunnablePassthrough()}`)?**
A: The dict is itself a Runnable (a `RunnableParallel` shorthand) that must produce both `context` (via the retriever) and `question` (the original string, untouched) for the prompt template. `RunnablePassthrough` is the identity function as a Runnable; forgetting it is why a first attempt at this pattern often throws a `KeyError` on `question`.

**Q7. Why does `bind_tools` alone not execute a tool call, and what loop does the manual tool-execution example in the file demonstrate?**
A: `bind_tools` only gets you the model's request to call a tool — LangChain does not execute the function for you. The file's manual loop (invoke, append AI message, run the tool function for each `tool_calls` entry, append the tool result, invoke again) is the exact request → execute → feed-result-back → repeat pattern that every agent framework, including LangGraph's agent constructor, wraps in a driver.

**Q8. What does the file say happens if you stuff too much retrieved context or chat history into a prompt, and what's the guard against it?**
A: Some providers truncate silently or return a degraded answer rather than a clear error, since LangChain does not enforce token budgets for you by default. The guard is checking `.get_num_tokens()` or a `tiktoken`-based counter before sending, and capping retriever `k` and text-splitter `chunk_size` with the model's real context window in mind.

**Q9. Why can `temperature=0` still produce non-identical outputs across calls, and what should a test suite assert instead of exact string equality?**
A: Even at temperature 0, most hosted APIs are not bit-for-bit reproducible across calls due to different hardware paths, provider-side batching, and minor floating-point nondeterminism. A test suite should assert on structure (valid JSON parse, an expected substring, or a tool call) rather than exact text, or it will flake.

**Q10. What does the file say about calling `.invoke()` inside an async context, and what's the fix?**
A: `.invoke()` is synchronous, so calling it inside an `async def` (e.g., inside a FastAPI/Flask-async route) blocks the entire event loop for the duration of the network call. The fix is to use `.ainvoke()`/`.astream()` — every Runnable has async variants — anywhere the surrounding code is already async.


---

## Video-Sourced Practice MCQs (Set 2)

A second practice set for LangChain, built the same way as this hub's NCA-GENL community bank: topics checked against a real YouTube AI-engineer-interview-prep video (part of a series also covering LangGraph, CrewAI, and banking-domain RAG), then written up as fully original multiple-choice questions here. These cover ground the clusters above don't touch -- LangGraph's graph-with-conditional-edges structure versus a linear LCEL chain, CrewAI's multi-agent researcher/critic/writer pattern, the RAG-vs-fine-tuning decision framework, and two concrete LLM cost-management levers (model tiering and semantic caching).

<script type="application/json" class="topic-quiz-data" data-title="LangChain Practice (Set 2)">
[
  {
    "d": "LangGraph vs. LangChain",
    "q": "A basic LangChain chain built with LCEL (`prompt | llm | parser`) executes as a fixed, linear sequence. What's the core structural difference LangGraph introduces that a plain linear chain cannot express?",
    "o": [
      "LangGraph is just a faster execution engine for the exact same linear sequence, with no structural difference",
      "LangGraph only works with a single LLM call and cannot orchestrate multiple steps at all",
      "LangGraph removes the need for prompts entirely, replacing them with hardcoded rules",
      "A stateful GRAPH of nodes connected by conditional edges — allowing branching, retries, and LOOPS based on a step's output, none of which a fixed linear sequence can represent"
    ],
    "a": [
      3
    ],
    "e": "A linear chain has one path: step 1 always leads to step 2, always leads to step 3. LangGraph instead models the workflow as nodes and CONDITIONAL edges, so the graph can inspect a node's output and decide to loop back (e.g. 're-retrieve with a different query if quality is low'), branch to a different node, or retry — control flow a straight-line chain has no mechanism to express at all. It's a structural difference in what can be BUILT, not merely a performance optimization on identical logic. LangGraph still uses prompts and LLM calls at its nodes — it doesn't replace them. And it's specifically designed for orchestrating MULTIPLE steps/nodes, the opposite of being limited to one call."
  },
  {
    "d": "LangGraph vs. LangChain",
    "q": "Concretely: \"retrieve a document, evaluate its quality, and if quality is low, re-retrieve with a different query before generating an answer\" is given as an example of something LangGraph can express but a basic LCEL chain cannot. WHY specifically can't a linear chain express this?",
    "o": [
      "Linear chains cannot call a retriever at all, so retrieval itself is the blocking issue, unrelated to the conditional retry",
      "The quality-evaluation step itself is impossible for a linear chain to run in any form",
      "The re-retrieval step is CONDITIONAL and potentially repeats (a loop) — a linear chain's steps run in one fixed forward order exactly once each, with no built-in mechanism to jump backward to an earlier step based on a later step's result",
      "A linear chain actually CAN express this exact logic — the two frameworks are functionally interchangeable for this use case"
    ],
    "a": [
      2
    ],
    "e": "The key missing piece is the CONDITIONAL LOOP: 'if quality is low, go back and re-retrieve' requires the workflow to jump backward to an earlier step depending on a later step's output — something a linear chain's fixed forward-only sequence has no way to represent, since each step in a chain runs exactly once, in order, with no branching decision point. It is NOT that the two frameworks are interchangeable here — this is precisely the capability gap LangGraph exists to fill. A basic chain can absolutely call a retriever (retrieval itself isn't the blocker) and can absolutely run a quality-evaluation step (that's just another LLM call or function) — what it structurally cannot do is loop BACK to retry based on that evaluation's result."
  },
  {
    "d": "LangGraph vs. LangChain",
    "q": "LangGraph is also described as maintaining PERSISTENT STATE across an entire graph's execution. Why does this matter specifically for multi-turn conversational or long-running workflows?",
    "o": [
      "Persistent state is a performance-only feature that speeds up token generation, unrelated to correctness",
      "Without persistent state, each node/step would need to be re-given all prior context manually and consistently; persistent state lets information (conversation history, intermediate results, retry counts, etc.) automatically carry forward across every node in the graph without you having to re-thread it through each call by hand",
      "Every LLM call already remembers all previous conversations automatically by default, making persistent state redundant",
      "Persistent state only matters for single-turn, one-shot queries and has no relevance to multi-turn conversations"
    ],
    "a": [
      1
    ],
    "e": "In a multi-step or multi-turn workflow, later steps (or later turns) often need to know what happened earlier — the conversation so far, an intermediate retrieval result, how many retries have already happened. Persistent state means the graph tracks and carries this forward automatically as execution moves between nodes, rather than you having to manually pass and re-pass that context into every single call. It's the OPPOSITE of only mattering for single-turn queries — a one-shot query has nothing to persist across turns in the first place, so this feature is specifically valuable for exactly the multi-turn/long-running case. It's a correctness/architecture feature, not a token-generation speed optimization. And LLM calls are stateless by default — an API call has no memory of previous calls unless the calling application explicitly re-supplies that context, which is exactly the manual burden persistent state removes."
  },
  {
    "d": "Multi-Agent Systems (CrewAI)",
    "q": "A CrewAI pipeline described here uses three agents: a researcher (retrieves documents), a critic (evaluates those documents for relevance and flags GAPS), and a writer (synthesizes the final response). What specific failure mode does the critic agent's role guard against, that a single-agent RAG pipeline lacks?",
    "o": [
      "The critic agent's only job is to check spelling and grammar in the final written response",
      "Confidently-wrong answers going out ungapped — without a dedicated evaluation step, a single-agent pipeline may synthesize a fluent, confident-sounding answer even when the retrieved documents are actually insufficient or irrelevant, with no automatic mechanism flagging that gap before the answer reaches the user",
      "Nothing — a single-agent RAG pipeline can already automatically detect its own irrelevant retrievals with equal reliability, making the critic role redundant",
      "The critic agent replaces the researcher agent entirely, making retrieval unnecessary"
    ],
    "a": [
      1
    ],
    "e": "A single LLM call that both retrieves and answers has no built-in checkpoint forcing it to notice 'these documents don't actually support a confident answer' — it can and often will generate fluent, plausible-sounding text regardless of whether the underlying evidence justifies it. Explicitly separating out a critic agent whose ONLY job is to evaluate relevance and flag gaps creates a structural checkpoint that catches this before the writer ever synthesizes a final response — which is exactly why this pipeline is described as reducing confident-wrong-answer outcomes on complex multi-document queries compared to a single-agent approach with no such check. The critic's role is about EVIDENCE quality, not proofreading the final prose. It doesn't replace retrieval — the researcher agent still does that; the critic evaluates what was retrieved. And a single-agent pipeline lacking this dedicated step is precisely the gap being described, not an equally-reliable alternative."
  },
  {
    "d": "RAG vs. Fine-Tuning",
    "q": "Given a scenario where the underlying knowledge (new regulations, new prices, new documents) changes FREQUENTLY, and you need every answer to cite its source for auditability — which approach is the better default, and why?",
    "o": [
      "Neither RAG nor fine-tuning can provide source citations under any circumstances — that requires a separate, unrelated system",
      "Both approaches are exactly equivalent for this scenario, with no meaningful tradeoff between them",
      "Fine-tuning — permanently baking frequently-changing facts into the model's weights is the more efficient choice when facts change often",
      "RAG — it retrieves current external knowledge at query time (so updates just mean re-indexing documents, not retraining) and naturally supports citing the specific retrieved source, unlike a fine-tuned model's baked-in weights"
    ],
    "a": [
      3
    ],
    "e": "RAG separates 'what the model knows how to do' from 'what facts it currently has access to' — updating the knowledge base is just re-indexing new documents, no retraining required, which is exactly suited to frequently-changing facts. It also naturally supports citation, since the answer is generated from specific retrieved chunks you can point back to. Fine-tuning instead bakes facts directly into the weights via an expensive training run — doing that repeatedly every time a regulation or price changes would mean constant, costly retraining, the opposite of efficient for this scenario. The two approaches are NOT equivalent here — this exact frequently-changing/auditability combination is the textbook case favoring RAG specifically. And RAG's retrieved-then-cited source documents ARE a natural, built-in citation mechanism — it doesn't require some unrelated separate system to bolt on."
  },
  {
    "d": "RAG vs. Fine-Tuning",
    "q": "Now the opposite scenario: you need the model to consistently adopt a very specific OUTPUT FORMAT or writing style, the domain vocabulary is fixed and won't change, and retrieval-step latency is unacceptable for your use case. Which approach fits better here, and why?",
    "o": [
      "Fine-tuning — it permanently adapts the model's behavior/style with no runtime retrieval step (so no added latency), which is exactly suited to a fixed vocabulary and a consistent format requirement that doesn't need to change per-query",
      "RAG — retrieving relevant examples at query time is always faster than any fine-tuned model's direct generation, regardless of the retrieval step's own latency",
      "Both approaches are equally fast at inference time, so latency is never a relevant factor in this choice",
      "Fine-tuning cannot influence a model's output format or writing style at all — only prompting can do that"
    ],
    "a": [
      0
    ],
    "e": "Fine-tuning directly adjusts the model's weights toward a target behavior (a specific format, a consistent style, fixed domain vocabulary) so that behavior becomes the model's default without needing to retrieve and stuff supporting examples into the prompt on every single call — which also means no retrieval-step latency at inference time, exactly matching a latency-critical requirement. Claiming RAG is 'always faster' ignores that RAG's retrieval step is itself an added latency cost that fine-tuning specifically avoids — that's precisely why latency-critical, format-stable use cases lean the other way. Fine-tuning very much CAN and does directly influence output format and style — that's one of its primary practical uses, not something exclusive to prompting. And latency absolutely is a relevant, explicitly-named factor in this decision, not an irrelevant one."
  },
  {
    "d": "LLM Cost Management",
    "q": "\"Model tiering\" — routing simple queries to a cheaper/smaller model and reserving an expensive, more capable model only for complex reasoning — is described as cutting LLM costs by roughly 60-80% on its own. Why does tiering alone capture such a large share of the savings?",
    "o": [
      "Because the expensive model is always slower, and tiering's savings come entirely from time saved, not money saved",
      "Because cheap models are strictly more accurate than expensive models on every task, so tiering both saves money AND improves quality with no tradeoff",
      "Because in most real query volumes, a large majority of incoming queries are actually SIMPLE, and each one sent to the expensive model was paying premium-model pricing for a task the cheap model could have handled just as well — so tiering eliminates that overpayment on the bulk of traffic, not just a small fraction of it",
      "Model tiering only saves money if you also implement semantic caching at the same time — it has zero effect by itself"
    ],
    "a": [
      2
    ],
    "e": "The reasoning behind tiering's outsized impact is a volume argument: if a large share of production traffic is genuinely simple (classification, short lookups, basic Q&A) that a cheaper model handles just as well, then EVERY one of those queries was previously paying full premium-model pricing for no accuracy benefit — redirecting that bulk of traffic to a cheaper model captures most of the available savings without touching the smaller share of genuinely complex queries that still need the expensive model. It's a cost argument, not fundamentally a latency-savings mechanism (though tiering may incidentally be faster too, that's not the stated 60-80% cost mechanism). Tiering is described as delivering savings ON ITS OWN — semantic caching is mentioned as an ADDITIONAL, separate lever, not a prerequisite for tiering to work at all. And the tradeoff is real: cheap models are cheaper because they're generally less capable on hard tasks, which is exactly why complex reasoning queries are deliberately still routed to the expensive tier rather than everything going cheap."
  },
  {
    "d": "LLM Cost Management",
    "q": "Semantic caching (e.g. with a tool like GPTCache) is listed alongside model tiering as an LLM cost-reduction technique. How does semantic caching differ from a simple exact-string cache (\"if this EXACT prompt was seen before, reuse the stored response\")?",
    "o": [
      "Semantic caching matches queries by MEANING/similarity, not exact text — so two differently-worded queries asking essentially the same thing can both hit the same cached response, catching far more repeat traffic than exact-string matching ever could",
      "Semantic caching and exact-string caching are the same mechanism, just implemented with different libraries — there's no functional difference in what gets cached or reused",
      "Semantic caching requires re-training the LLM itself every time a new query comes in, making it slower and more expensive than exact-string caching",
      "Semantic caching only works for a single, specific query and cannot generalize to any other phrasing whatsoever, making it strictly narrower than exact-string caching"
    ],
    "a": [
      0
    ],
    "e": "An exact-string cache only helps if the identical text was asked before, character for character — a real-world weakness, since users phrase the 'same' question many different ways ('What's your refund policy?' vs. 'How do refunds work?'). Semantic caching instead compares queries by MEANING (typically via embedding similarity), so semantically equivalent-but-differently-worded queries can still hit a cached response, which is exactly why it catches meaningfully more repeat traffic in practice than exact-string matching and is called out as a real cost lever. It is NOT the same mechanism as exact-string caching — the whole point is that it generalizes across phrasing where exact-match caching fails. It's actually BROADER than exact-string caching (catching more cases), not narrower. And it doesn't retrain the LLM at all — it's a caching layer sitting in front of the model, entirely separate from the model's own weights or training process."
  }
]
</script>
<div class="topic-quiz-mount"></div>
