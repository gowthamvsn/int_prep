# The AI Engineer Playbook — Tool-Named Answers, Not Category Names

Every question below is answered the way a hiring manager actually wants to hear it in a room — with the exact library, the exact flag, the exact model name, and where it sits in the stack. Not "we used LangChain and a vector database," but *which* vector database, *why that one*, and what you'd literally type to run it. Compiled from six source posts (a hiring-manager checklist, five interview-question category sweeps, a 7-technique optimization cheat-sheet, a 10-step roadmap, a 10-paper reading list, and a second Head-of-AI/ML question sweep with its comment section) screenshotted to WhatsApp on 2026-08-21.

> 🔗 **Where this sits relative to the rest of the hub:** this doc is the interview-recitation layer — the tool name and the one-line "why." For the implementation underneath each answer, see [Core Technical Depth](/topic/core-technical), [RAG, Deeper](/topic/rag-deeper), [Prompt Engineering, Deeper](/topic/prompt-engineering-deeper), [LLM Landscape](/topic/llm-landscape), [MLOps Practice](/topic/mlops-practice), and [Production ML Practice](/topic/production-ml).

---

## 01 · What a Hiring Manager Actually Wants to Know

A hiring manager's rant, not a quiz — the point is that "we used LangChain and a vector DB" isn't an answer. Below is what each question is really probing, and the concrete stack that backs it up.

### 1. Can you design a system end-to-end — from ingestion to serving? What are the bottlenecks?

Walk it stage by stage and name the actual tool at each one, not a category:

- **Ingestion** — files/events land via **Airflow** or **Dagster** for batch, **Kafka** if it's continuous. Large PDFs and OCR-heavy scans are where this stage chokes first.
- **Chunk + embed** — LangChain's `RecursiveCharacterTextSplitter` or LlamaIndex's `SemanticSplitterNodeParser` for chunking; **OpenAI text-embedding-3-small** (~$0.02/M tokens) or a self-hosted **BGE-large-en-v1.5** via `sentence-transformers` for embeddings. Embedding the whole corpus is the first real cost line.
- **Storage** — **pgvector** if already on Postgres, **Qdrant** or **Weaviate** for hybrid search and payload filtering out of the box, **Pinecone** for zero ops (at a price).
- **Retrieval** — HNSW index, pull top 20–50 candidates, then a reranker (**Cohere Rerank** or a local cross-encoder) cuts that to 5–8 before it ever reaches the prompt.
- **Generation** — **vLLM** if self-hosted, straight API calls to GPT-4o/Claude if not. This is where 70–90% of both latency and dollar cost live — first place to profile, not last.
- **Serving** — FastAPI's `StreamingResponse` behind **Envoy** or **Nginx**.
- **Logging** — traces via **OpenTelemetry** into **Langfuse** or **Arize Phoenix**, so a bad answer gets replayed end-to-end instead of guessed at.

### 2. How would you estimate costs? How would you reduce them?

`(input + output tokens per request) × price per token × requests/day`. At GPT-4o-class pricing (roughly $2.50/M input, $10/M output), a RAG app pushing 2k context tokens and getting 500 back at 50k requests/day lands around $300–400/day in generation alone, before embeddings. Track it live in **Langfuse** or **Helicone** — both attach a real dollar figure to every logged request instead of estimating after the fact. To cut it: cache repeats in **Redis** or with **GPTCache** (a semantic cache built for exactly this — matches near-duplicate queries, not just exact ones), route easy queries to a cheaper model with **LiteLLM**'s router config, and trim context — most RAG prompts carry more retrieved text than the model ever actually uses.

### 3. How would you reduce latency? What's a good latency-vs-quality tradeoff?

Time goes to three places: retrieval (tens of ms with a decent HNSW index), reranking (adds 50–150ms if calling Cohere's API over the network — worth it, but know it's there), and generation, which dwarfs both. Serving-side, **vLLM** with `--tensor-parallel-size` set correctly and PagedAttention doing its job is table stakes; **TensorRT-LLM** squeezes more out on NVIDIA-only hardware if you're willing to compile a per-model engine. Perceived-latency-side, streaming via SSE is non-negotiable — total time doesn't change, but the user sees the first token in ~200ms instead of a blank screen for 3 seconds. On the actual tradeoff: quantizing to INT4 with **AWQ** typically costs 1–3% on eval benchmarks for a 2–3x speed gain — that's the default trade unless it's a legal or medical use case.

### 4. Do you really need self-hosted LLMs? When?

Self-host when data can't leave your network (HIPAA, on-prem finance), or when running enough sustained volume that GPU-hours beat per-token API pricing — run the actual math for your traffic, don't assume. Stack: **vLLM** or Hugging Face's **TGI** behind **KServe** or **Ray Serve** on Kubernetes, on A100s or H100s. Otherwise default to the API directly — OpenAI/Anthropic/Gemini SDKs, or **LiteLLM** for one interface across all three so switching providers doesn't mean rewriting every call site.

### 5. How would you fine-tune on user behavior? Which framework? What about model serving?

Pull accept/reject/edit signals out of the logs, build preference triples for DPO or plain (input, output) pairs for SFT. For training: **Hugging Face TRL + PEFT** for LoRA/QLoRA — full fine-tune of anything above ~7B params is rarely worth it — or **Axolotl** to drive it from a YAML config instead of writing the training loop by hand. Runs get tracked in **Weights & Biases**. Serving: **vLLM**'s `--enable-lora` flag, which hot-swaps adapters on one base model instead of hosting a full separate model per fine-tune — that detail is what actually keeps this affordable.

### 6. How would you construct the dataset? What about the loss function? What about MLOps?

Standard cross-entropy on next-token prediction for SFT; **DPO** loss for training on preference pairs instead of gold labels — no separate reward model needed, unlike full RLHF/PPO. Data gets versioned with **DVC** so a training run is reproducible against the exact snapshot it used. Experiments tracked in **MLflow** or W&B. The retraining pipeline itself lives as an **Airflow** or **Prefect** DAG, so it's a scheduled job, not something run by hand from a laptop.

### 7. Which database would you use, and why? Vector DB? SQL? NoSQL?

**pgvector** if already running Postgres — one fewer service, and since v0.5 it has a real HNSW index, so "pgvector is slow" isn't an excuse anymore. **Qdrant** or **Weaviate** for hybrid dense+sparse search and metadata filtering as first-class features. **Pinecone** for fully managed and no ops burden. **Chroma** only for local dev/prototyping — `pip install`, zero ops, runs in-process — but not the production store at real QPS; it's the "start here, graduate later" option, not a permanent choice. For the structured side: **Postgres** for anything transactional, **Redis** for session state/caching, **MongoDB** or **DynamoDB** only if the schema is genuinely document-shaped and joins aren't needed.

### 8. What metrics would you track? How?

System metrics — p50/p95/p99 latency, error rate, GPU utilization via the **DCGM exporter** — go to **Prometheus + Grafana**. LLM-specific quality metrics — retrieval precision, faithfulness, cost per request, token counts — go to an LLM observability tool: **Langfuse** (open source, self-hostable, default choice), **Arize Phoenix** if free and self-hosting isn't needed, **LangSmith** if already deep in the LangChain ecosystem.

### 9. What about system monitoring? How would you debug failure cases?

Same tools as above, because their real value is the trace view — click into one bad response and see the exact retrieved chunks, the exact prompt that was sent, token-by-token generation, and per-stage latency, so debugging is "read the trace," not "try to reproduce it on a laptop." **OpenTelemetry** underneath for vendor-neutral tracing that isn't locked into one tool's SDK.

### 10. What about the feedback loop? How would you track and evaluate it?

Thumbs up/down stored in Postgres next to the request ID — nothing fancier needed at first. Periodically pull the negative-signal rows into an eval set and run it through **RAGAS** to see which metric actually moved: retrieval, faithfulness, or relevance. Once volume's high enough, that same data becomes fine-tuning data (see Q5).

### 11. How would you make the system more deterministic?

`temperature=0` plus a fixed `seed` parameter gets you most of the way. For real structural determinism: OpenAI's `response_format={"type":"json_schema","strict":true}` or Anthropic's forced tool-use constrains the shape of the output; for open models, the **Outlines** or **Guidance** libraries do grammar-constrained decoding — the model literally cannot emit a token that breaks the schema. Beyond that: anything with real business consequences (pricing, eligibility, math) moves out of the model entirely and into plain code the model calls as a tool.

### 12. How would you replace embedding models and backfill embeddings without downtime?

**Qdrant** and **Pinecone** both handle this cleanly via collection aliases / namespaces: build the new collection under a new name, backfill it with a **Celery** or **Airflow** job running in the background, validate retrieval quality against an eval set, then flip the alias to point at the new collection. The swap is atomic, so there's no window where traffic sees a half-migrated index; the old collection stays queryable until confidence is high, then gets dropped.

```
live traffic ──► alias "prod" ──► [ collection_v1 (model A) ]   ← serving

background job ──► [ collection_v2 (model B) ] ← backfilling, not aliased yet

  once backfill is done + validated against eval set:
      alias "prod" repointed ──► [ collection_v2 (model B) ]   ← atomic cutover
      [ collection_v1 ]  ← kept briefly, then dropped
```

### 13. What are the fallback mechanisms?

Retry with exponential backoff — the `tenacity` library in Python rather than hand-rolling it. Provider/model fallback through **LiteLLM**, which has fallback routing built in: primary model fails or rate-limits, it automatically retries on a secondary. Circuit breaking so a failing dependency stops getting hammered — **Istio** if already a service-mesh environment, a simple in-process breaker otherwise. And one hard rule baked into the prompt: if the required context isn't there, say so — never let a fallback path silently degrade into a hallucination.

> **His 5 favorite questions — testing fundamentals, not tool fluency**
> 1. **Solve it without LLMs or vector DBs.** A regex plus a lookup table, or plain **Elasticsearch** keyword search, solves 80% of "search my docs" problems more cheaply and more reliably than a full RAG stack.
> 2. **Solve it with classical IR, rules, or heuristics.** **BM25** — Elasticsearch/OpenSearch's default scoring, or the `rank_bm25` Python package for a quick prototype — is the baseline your fancy dense retriever is supposed to beat. If it doesn't, you don't need the fancy retriever.
> 3. **How would you make the system more deterministic?** Same answer as Q11 above.
> 4. **Explain tokenization and embeddings from scratch.** Text splits into subword pieces via a byte-pair-encoding tokenizer — `tiktoken` for OpenAI models, `SentencePiece` for Llama/Mistral-family models. Each token maps to an integer ID; the model turns each ID into a vector (an embedding) positioned so words with similar meaning land near each other in that space.
> 5. **What actually happens during fine-tuning?** AdamW optimizer, a warmup-then-cosine-decay learning-rate schedule, base weights frozen while a LoRA adapter gets added on the attention projection matrices — that's literally what `target_modules=["q_proj","v_proj"]` in a PEFT `LoraConfig` is specifying, not magic.

---

## 02 · LLM Infrastructure & Inference

The plumbing between "prompt in" and "tokens out." Almost everything here lives inside the serving engine — the skill is picking the right engine and the right flag, not implementing this from scratch.

### 1. What is KV Cache, and why does it matter?

Every new token needs to attend to every token generated before it. Without a cache, the model recomputes attention over the *entire* sequence at every single step. The KV cache stores each past token's key/value tensors so only the new token gets computed fresh — it's the single biggest reason generation is fast at all, but it grows with sequence length × batch size and is the main thing eating GPU memory.

```
token 1 ─► compute K,V ─► cache [K1,V1]
token 2 ─► compute K,V ─► cache [K1,V1, K2,V2]     (past K,V reused, not recomputed)
token 3 ─► compute K,V ─► cache [K1,V1, K2,V2, K3,V3]
```

**Where it lives:** managed automatically inside the serving engine — **vLLM** or Hugging Face's **TGI**. The knobs actually touched: vLLM's `--kv-cache-dtype fp8` to roughly halve its memory footprint, and `--gpu-memory-utilization` to cap how much VRAM it's allowed to claim.

### 2. Continuous batching vs. dynamic batching

Dynamic (static) batching waits to fill a batch, runs it, and nobody's response returns until the slowest sequence in that batch finishes. Continuous batching swaps sequences in and out of the active batch token by token — the instant one finishes, its GPU slot goes straight to a new request, no waiting on the batch to drain.

**Where it lives:** this is **vLLM**'s headline feature (also in TGI). Automatic once on vLLM — the relevant flag is `--max-num-seqs`, controlling how many sequences it tries to keep in flight at once.

### 3. Tensor parallelism vs. pipeline parallelism

Tensor parallelism splits the math *inside* one layer across GPUs — each GPU computes part of the same matrix multiply and they constantly sync, so this needs a fast interconnect (NVLink, not plain PCIe). Pipeline parallelism splits *different layers* across GPUs, assembly-line style.

```
Tensor parallelism (split ONE layer):        Pipeline parallelism (split LAYERS):
  [ GPU 0: half the matrix ] ┐                input ─►[GPU0: layers 1-10]─►[GPU1: layers 11-20]─►output
                              ├─► combine
  [ GPU 1: half the matrix ] ┘
```

**Where it lives:** training-time: **Megatron-LM** (NVIDIA) or **DeepSpeed** (Microsoft) implement both. Inference-time with vLLM: `--tensor-parallel-size 4` splits across 4 GPUs on one node; `--pipeline-parallel-size` splits across nodes.

### 4. What is speculative decoding?

A small draft model quickly guesses several next tokens; the big target model verifies them all in one parallel forward pass instead of generating one at a time. Correct guesses are free speed; the first wrong one falls back to normal generation from that point. Since the target model always verifies, output quality is identical to running it alone — typically 2–3x faster.

**Where it lives:** **vLLM** supports it natively via `--speculative-model <small-model-path>`; **TensorRT-LLM** has draft-target speculative decoding built into its runtime too.

### 5. What is quantization, and how does it improve inference?

Weights normally sit as 16-bit floats. Quantization compresses them to INT8 or INT4 — smaller in memory, faster to multiply — at a small accuracy cost.

**Where it lives:** **AutoGPTQ** for GPTQ, **AutoAWQ** for AWQ (AWQ tends to hold accuracy better at INT4 in practice), **bitsandbytes** for the 4-bit NF4 quantization used in QLoRA fine-tuning, **llama.cpp**'s GGUF format for running on CPU or Apple Silicon.

### 6. How do vLLM and TensorRT-LLM improve serving performance?

**vLLM**'s core trick is PagedAttention — it manages the KV cache like an OS pages memory, in fixed-size blocks, avoiding the fragmentation from one big contiguous allocation per request. **TensorRT-LLM** takes a different approach: it fuses operations into hand-tuned kernels and compiles the model into an optimized engine ahead of time, squeezing out hardware-specific speed on NVIDIA silicon that a general framework leaves on the table.

**How you actually run them:** vLLM: `pip install vllm`, then `vllm serve <model>` gives an OpenAI-compatible endpoint in one command — why it's most teams' default. TensorRT-LLM: compile the model into a TensorRT engine offline, then serve it through **NVIDIA Triton Inference Server** — more setup, faster on NVIDIA hardware specifically.

### 7. What causes GPU memory fragmentation?

Every request's KV cache is a different length, and allocating one contiguous block per request leaves gaps as requests of varying size finish and free memory at different times — like disk fragmentation. Eventually there's plenty of *total* free memory but no single block big enough for the next request.

**Where you'd see/fix it:** watch it with `nvidia-smi` directly or the **DCGM exporter** feeding Grafana. The fix, PagedAttention, isn't something configured manually — it's free by serving through **vLLM** instead of a naive Hugging Face `generate()` loop.

### 8. How would you scale LLM inference for millions of users?

- **Horizontal scaling** — replicas on **Kubernetes**, autoscaled with **KEDA** on queue depth (CPU-based autoscaling is meaningless for a GPU-bound workload).
- **Serving orchestration** — **Ray Serve** or **Triton** managing the model-serving layer itself.
- **Load balancing** — **Envoy** or **Nginx** in front.
- **Tiered routing** — **LiteLLM** or a small custom router sending easy queries to a cheap model.
- **Caching** — **Redis** or **GPTCache** in front of everything for repeat queries.

### 9. How would you choose between open-source and closed-source LLMs?

Open-source — **Llama 3.x**, **Mistral**, **Qwen2.5** — gives full control: self-host, fine-tune, keep data in-house, cheaper at real scale, served through vLLM/TGI. Closed — **GPT-4o/GPT-5**, **Claude**, **Gemini** — gives best-in-class quality with zero infra burden and the fastest path to shipping, at ongoing per-token cost. Decision hinges on traffic volume, data sensitivity, and how differentiated the model's behavior actually needs to be. (See [LLM Landscape](/topic/llm-landscape) for the full 25-model map.)

### 10. How would you optimize inference cost without sacrificing quality?

Stack the levers instead of picking one: **AWQ/GPTQ** quantization, **vLLM** continuous batching so GPUs aren't idling, **LiteLLM** difficulty-based routing, **Redis/GPTCache** for repeats, shorter prompts, and speculative decoding for extra free speed. Full toolbox in [section 07](#07-7-llm-optimization-techniques) below.

---

## 03 · RAG & Vector Databases

Getting an LLM to answer from your data instead of only what it memorized in training.

```
query ─► embed query ─┬─► dense (vector) search  ─┐
                       └─► sparse (BM25) search    ─┴─► merge ─► rerank ─► top-k chunks ─► prompt ─► LLM ─► answer
```

### 1. Why is chunking important in RAG systems?

Limited context window plus embeddings work best when a chunk holds one coherent idea. Too big and irrelevant filler dilutes the match; too small and the context needed to actually answer gets lost.

**Tools:** LangChain's `RecursiveCharacterTextSplitter` for a fast default, LlamaIndex's `SemanticSplitterNodeParser` for boundaries that follow meaning instead of a fixed character count, **Unstructured.io** to parse messy PDFs/HTML into clean text before any splitter runs.

### 2. How do you choose chunk size for documents?

No universal number. `chunk_size=512, chunk_overlap=50` is the LangChain-default starting point to tune from — not a rule — validated against a small retrieval eval set built from real documents and real questions, not a blog post's recommendation.

### 3. What's the difference between sparse and dense retrieval?

Sparse matches exact words — cheap, great for names/codes/rare terms. Dense matches meaning via embeddings — finds a relevant chunk even when the query is phrased completely differently. Strongest systems run both (see Q6).

**Tools:** Sparse: **Elasticsearch**/**OpenSearch**'s built-in BM25, or `rank_bm25` for a lightweight prototype. Dense: **OpenAI text-embedding-3**, **Cohere embed-v3**, or **BGE-large-en-v1.5** for local.

### 4. What causes poor retrieval quality in RAG pipelines?

Bad chunking, a mismatched embedding model between docs and query, stale/duplicate index data, top-k too small, missing metadata filters, or plain vocabulary mismatch. Don't guess which one — measure it.

**Tools:** **RAGAS**'s `context_precision`/`context_recall` metrics, or **Arize Phoenix**'s retrieval trace view, to find out which of the five it actually is before touching anything.

### 5. What are rerankers, and why are they used?

Vector search is fast but approximate — it pulls a wide candidate net (say top 50) on a cheap similarity score. A reranker scores each candidate against the query far more precisely, one pair at a time — too slow over a whole index, cheap over a 50-item shortlist.

**Tools:** **Cohere's Rerank API** for the easy path; `BAAI/bge-reranker-large` or `cross-encoder/ms-marco-MiniLM-L-6-v2` via `sentence-transformers` to run in-house.

### 6. How would you build a hybrid retrieval pipeline?

- Run BM25 and dense search in parallel on the same query.
- Merge and de-duplicate — usually via **Reciprocal Rank Fusion (RRF)**.
- Apply metadata filters (date, source, permissions).
- Rerank the merged candidates.
- Pass the final top-k into the prompt.

**Tools:** **Qdrant** and **Weaviate** both do dense+sparse fusion natively — Weaviate exposes it as `hybrid` search with an `alpha` param controlling the blend. Elasticsearch/OpenSearch can do it too via their kNN plugin plus standard BM25.

### 7. How would you reduce hallucinations in a RAG system?

Ground the prompt tightly, permit "I don't know," ask for citations per claim. No tool fixes a bad retriever — improving retrieval quality is usually the bigger lever than anything in the prompt.

**Tools:** **RAGAS**'s `faithfulness` score to actually measure whether it's working, rather than eyeballing outputs.

### 8. How would you evaluate a RAG pipeline end-to-end?

Retrieval metrics (did the pipeline fetch the right chunks — precision/recall/MRR) and generation metrics (given those chunks, is the answer faithful and relevant) — measured separately so it's clear which half is broken.

**Tools:** **RAGAS** end to end, **TruLens** for a more UI-driven eval loop, **LangSmith** evals if already on LangChain.

### 9. How would you optimize vector database search latency?

Approximate nearest-neighbor indexing instead of brute force, lower embedding dimensionality where quality allows, shard as the index grows, cache frequent queries, pre-filter by metadata before the vector search runs.

**Tools:** **HNSW** is the default index type in Qdrant, Weaviate, Pinecone, and pgvector (since 0.5) — if a vector DB is still on IVF or brute force by default, that's the first thing to change.

### 10. How would you handle document updates without rebuilding the entire index?

Incremental upsert/delete by ID — nothing else needs to be touched. Keep a document-ID → chunk-ID mapping to delete a document's old chunks before inserting the re-chunked version. Full reindex is only needed when swapping the embedding model itself (see Q12 in section 01).

**Tools:** Qdrant and Pinecone both expose a plain `upsert(id, vector, payload)` call. In pgvector it's just an `UPDATE` on the row.

---

## 04 · Prompt Engineering

The cheapest lever on model behavior — squeeze this before reaching for fine-tuning.

### 1. What makes an effective system prompt?

Write it the way you'd write an API contract, not a vibe: the role, the goal, explicit constraints on what it must never do, the exact output format, and defined behavior for the unclear/missing-information case. Vague system prompt, inconsistent model — every time.

### 2. Zero-shot vs. one-shot vs. few-shot prompting

Zero-shot: just ask. One-shot: one worked example of the input→output pattern. Few-shot: several examples so the model pattern-matches both content and format. More examples buys consistency, costs context tokens on every single request.

### 3. What is Chain-of-Thought prompting, and when should it be used?

"Think step by step" before the final answer — gives the model room to work through multi-step reasoning, math, or logic instead of jumping straight to a (often wrong) conclusion. Skip it for simple lookups or classification, where it just burns tokens for nothing.

### 4. What is ReAct prompting?

Reasoning + Acting: the model writes a thought, takes an action (usually a tool call), observes the result, thinks again, repeats. This think → act → observe loop is the pattern underneath essentially every tool-using agent.

```
Thought: I need the current stock price to answer this.
Action: call get_price("NVDA")
Observation: 187.42
Thought: I now have what I need.
Answer: NVDA is currently trading at $187.42.
```

### 5. How would you reduce hallucinations using prompt engineering?

Explicitly permit "I don't know," require citations, constrain answers to given context only, lower temperature, break big ambiguous asks into smaller verifiable steps.

### 6. What is prompt injection, and why is it dangerous?

Untrusted text the model reads — a webpage, a document, a user message — carries hidden instructions that hijack it into ignoring its system prompt. Dangerous because a compromised model can leak data, bypass its own safety rules, or, worst case for an agent with real tool access, take unintended actions on the attacker's behalf.

### 7. How would you defend against prompt injection attacks?

- Treat external content as **data, never instructions** — wrap/tag it explicitly.
- System prompt outranks user/tool content by explicit rule.
- Validate and sandbox whatever tools return before it goes back to the model.
- Add an output/guardrail filter checking the model's final action before execution.
- **Least privilege** — never give an agent more tool access than the task needs.

**Tools:** **NeMo Guardrails** (NVIDIA) or **Lakera Guard** for a dedicated guardrail layer instead of hand-rolled detection; **Rebuff** is a lighter open-source option built specifically for injection detection.

### 8. How would you structure prompts for tool calling?

Describe each tool like a function signature — name, one-line purpose, typed parameters — and require calls in a strict, parseable schema.

**Tools:** OpenAI and Anthropic both expose native function/tool-calling in their API — pass a JSON schema, they return structured `tool_use` blocks. Prefer that over hand-rolled prompt-based tool calling whenever the model supports it.

### 9. What are structured outputs and JSON mode?

Constrain generation itself so it always conforms to a schema, instead of hoping free text happens to be parseable.

**Tools:** OpenAI's `response_format={"type":"json_schema","strict":true}`, Anthropic's forced tool-use trick (define one tool matching the schema, force the model to call it), or the `instructor` library, which patches either client so you get back a validated Pydantic object directly — removes the whole "the JSON almost parsed" class of bug.

### 10. When would you choose prompt engineering over fine-tuning?

Start with prompting — cheap, no training data, iterate in minutes. Move to fine-tuning once prompting plateaus: very specific consistent output is needed at real scale, cost needs to shrink by baking behavior into a smaller model, or there's proprietary knowledge too large to stuff into every prompt.

---

## 05 · MLOps & Deployment

Where a model stops being a notebook experiment and becomes something other people depend on staying up.

### 1. How would you deploy an ML model into production?

Versioned container artifact, pushed through CI, deployed with gradual rollout and an instant rollback path — not something built by hand each time.

**Tools:** **Docker** image via **GitHub Actions**/GitLab CI, deployed through **KServe** or **Seldon Core** on Kubernetes — both give canary/shadow traffic splitting out of the box instead of building it with load balancer rules.

### 2. What is model drift, and how do you detect it?

The real world changes, so the input-output relationship the model learned stops matching it. Data drift = inputs shift; concept drift = the underlying relationship shifts.

**Tools:** **Evidently AI** or **whylogs**/WhyLabs for both data-drift and prediction-drift dashboards — purpose-built for this, saves writing your own KS-test/PSI code.

### 3. What's the difference between CI/CD and CT pipelines in ML?

CI/CD ships the serving code. CT retrains the model itself.

**Tools:** CI/CD: GitHub Actions/Jenkins as usual. CT: an **Airflow** or **Kubeflow Pipelines** DAG — pull data → retrain → eval → register in **MLflow**'s model registry → promote only if it beats the current champion.

### 4. What metrics would you monitor in production?

System health, prediction distribution, and a business/quality signal tied to the actual task — a model can look perfectly healthy on latency while quietly getting worse at its job.

**Tools:** **Prometheus/Grafana** for system metrics, **Evidently** for drift, quality signal logged into the same **MLflow** run for correlation.

### 5. What causes training-serving skew?

Features at inference time don't match what training saw — usually because training used a batch pipeline and serving uses a separate, hand-written real-time path that's quietly drifted from it.

**Tools:** Fixed by a feature store (see Q9) — one definition, both paths.

### 6. How would you perform canary deployments for ML models?

Route a small % of live traffic to the new version, compare metrics side by side, ramp up only if it's at least as good.

**Tools:** **Argo Rollouts** or **Flagger** for Kubernetes-native traffic shifting; **KServe** and **Seldon** both support canary rollout as a first-class config, not something hand-built.

### 7. How would you automate model retraining?

- A schedule or an **Evidently** drift alert fires a webhook.
- An **Airflow**/**Kubeflow** DAG pulls fresh labeled data.
- Model retrains, gets evaluated against a held-out set and the current production model.
- Promoted in the **MLflow** registry only if it clears a quality bar.

### 8. How would you monitor data quality in production?

Validate against expected schema/ranges on the way in, then watch for distribution shift over time — two different checks, two different failure modes.

**Tools:** **Great Expectations** for the "is this data even valid" gate, **Evidently** or **whylogs** for the "is it silently drifting" watch.

### 9. How would you manage feature stores across training and inference?

One feature definition, served two ways — batch for training, low-latency lookup for real-time inference — so both paths agree by construction instead of by discipline.

**Tools:** **Feast** is the standard open-source choice — same feature definitions, with a **Redis**- or DynamoDB-backed online store for the real-time side.

### 10. How would you design rollback and disaster recovery for ML systems?

Every deployed version immutable and addressable, rollback a single operation, critical services replicated across zones, and the recovery process actually rehearsed — a DR plan nobody's tested is just a document.

**Tools:** **MLflow**'s model registry gives versioned artifacts with stage transitions (staging → production → archived) — rollback is "promote the previous version," a registry operation, not a redeploy from scratch.

---

## 06 · Python & Backend

The engineering underneath the model — whether your AI feature survives real traffic or falls over at 50 concurrent users.

### 1. How would you build a scalable FastAPI backend for AI inference?

Async endpoints so I/O waits don't block the server, model calls pushed onto a queue so one slow request can't starve everyone, load balancer for horizontal scaling, lightweight and heavy endpoints scaled separately.

**Tools:** **Uvicorn** workers behind **Gunicorn** for process management, **Nginx** or **Envoy** in front, model calls queued through **Redis + RQ** or **Celery** so the web tier stays thin.

### 2. Synchronous vs. asynchronous APIs

Sync blocks the thread until an operation finishes. Async lets it switch to other work while waiting on I/O — critical for AI APIs, where most wall-clock time is spent waiting on the model or the database, not computing. In FastAPI, an `async def` endpoint runs on the event loop; a plain `def` endpoint runs in a threadpool automatically — know which one is being written.

### 3. How would you handle concurrent inference requests?

Queue instead of firing straight at the GPU, batch compatible requests, async I/O throughout the web layer, explicit concurrency limits so load degrades gracefully instead of everything slowing down together.

**Tools:** **vLLM** handles batching automatically if that's the serving layer; on the FastAPI side, an `asyncio.Semaphore` or a Redis-backed counter caps in-flight requests instead of trusting the OS to sort it out under load.

### 4. What causes memory bottlenecks in Python AI systems?

Large model weights resident in RAM/VRAM, duplicated data loads, the GIL limiting true parallel CPU-bound work, unbounded caches/queues, and memory not freed between requests because something still holds a reference.

**Tools:** Profile with `py-spy` or `memory_profiler` before guessing. `torch.cuda.empty_cache()` is a common band-aid people reach for that usually masks a real leak rather than fixing it.

### 5. How would you optimize API throughput?

Batch, async, cache, keep payloads small, pool connections — but profile first. A model quantized to fix a bottleneck that was actually a blocking DB call is a real, common mistake.

**Tools:** `py-spy` or `cProfile` to find the real bottleneck before touching the model.

### 6. How would you implement request batching for GPU inference?

Collect requests within a short window, run them through the model as one GPU call, split results back out per caller.

**Tools:** **vLLM**/**TGI** do this at the serving layer — don't hand-roll batching in the FastAPI app unless there's a specific reason the serving engine can't do it.

### 7. How would you design rate limiting for AI APIs?

Per-key/per-user tracking, limits on both requests/minute *and* tokens/minute — LLM cost scales with tokens, so one huge request can do more damage than a hundred small ones — with a clear 429 and retry-after.

**Tools:** `slowapi` (FastAPI's rate-limiting middleware) or `fastapi-limiter`, both backed by **Redis** since the counter needs to be shared across replicas, not kept per-process.

### 8. How would you stream LLM responses token by token?

Push each token to the client the instant it's generated instead of waiting for the full response. Total time is unchanged; perceived speed is transformed.

```
without streaming:  client waits ░░░░░░░░░░░░░░░░░░ ─► gets full answer at once
with streaming:     client sees   T o k e n s   a p p e a r i n g   a s   t h e y ' r e   m a d e
```

**Tools:** FastAPI's `StreamingResponse`, or `sse-starlette` for proper Server-Sent Events semantics with reconnect support instead of a raw chunked stream.

### 9. How would you handle background tasks in FastAPI?

Run it after the response is already back with the user — logging, notifications, kicking off async jobs.

**Tools:** FastAPI's built-in `BackgroundTasks` for anything under a few seconds; **Celery** (with Redis or RabbitMQ as the broker) once it's a real job — retries, scheduling, distributed workers.

### 10. How would you secure an AI API in production?

- Authenticate every request — API keys/OAuth via **fastapi-users** or **Auth0**.
- Rate-limit per Q7, Redis-backed.
- Validate/sanitize all inputs, including anything reaching the LLM — blunts prompt injection too.
- Never expose internal errors/stack traces to the client.
- TLS termination at the load balancer (Envoy/Nginx), secrets pulled from **AWS Secrets Manager** or **HashiCorp Vault** — never in an env file anywhere near the repo.

---

## 07 · 7 LLM Optimization Techniques

These stack — a well-optimized serving stack runs several at once, not one. KV caching, continuous batching, speculative decoding, and tensor/pipeline parallelism are explained in full in [section 02](#02-llm-infrastructure-inference); gathered here with the two new ones as one toolbox.

| # | Technique | What it does | Tools |
|---|---|---|---|
| 1 | **Quantization** | Fewer bits per weight (INT8/INT4) — smaller model, faster math, small accuracy cost. | AutoGPTQ, AutoAWQ, bitsandbytes, llama.cpp/GGUF |
| 2 | **Knowledge Distillation** | Train a small student model to mimic a larger teacher's output distribution — most of the capability, a fraction of the size. | Hugging Face `transformers` + a custom KL-divergence loss against the teacher's logits; DistilBERT-style recipes are the reference implementation |
| 3 | **KV Caching** | Reuse past tokens' key/value tensors instead of recomputing them every step. | vLLM / TGI, automatic — `--kv-cache-dtype fp8` |
| 4 | **Continuous Batching** | Swap requests in/out of the active batch as they arrive/finish instead of waiting for a fixed batch to fill. | vLLM (`--max-num-seqs`), TGI |
| 5 | **Speculative Decoding** | A draft model proposes tokens fast; the target model verifies them all in one pass. | vLLM (`--speculative-model`), TensorRT-LLM |
| 6 | **Tensor Parallelism** | Split one layer's math across GPUs working together on the same layer. | Megatron-LM, DeepSpeed, vLLM `--tensor-parallel-size` |
| 7 | **Pipeline Parallelism** | Split different layers across GPUs, assembly-line style. | Megatron-LM, DeepSpeed, vLLM `--pipeline-parallel-size` |

**If the problem is… → reach for…**

| Symptom | Fix |
|---|---|
| Model doesn't fit on one GPU | Tensor parallelism (Megatron-LM/DeepSpeed), pipeline parallelism |
| Each token takes too long | KV caching (built into vLLM), speculative decoding (`--speculative-model`) |
| GPU idles between requests | Continuous batching — switch to vLLM/TGI if not already on one |
| Model itself too big/expensive | Quantization (AutoAWQ/AutoGPTQ), knowledge distillation |

---

## 08 · The 10-Step Roadmap

The order these topics actually build on each other.

1. **Python & SQL** — the tools everything else is written with. SQL matters because almost all real data lives in Postgres/MySQL and you'll be writing window functions constantly, not toy SELECTs. *(Python fundamentals, OOP, data structures, APIs, SQL joins/CTEs/window fns, pandas & numpy — see [SQL Practice](/topic/sql-practice), [NumPy Practice](/topic/practice-numpy), [Pandas Practice](/topic/practice-pandas).)*
2. **Machine Learning** — the fundamentals underneath every LLM: data prep, feature engineering, fitting a model, and actually evaluating whether it's good, hands-on in scikit-learn. *(See [scikit-learn Practice](/topic/practice-sklearn), [Classical ML Models Practice](/topic/practice-ml-models).)*
3. **Deep Learning & Transformers** — how neural nets learn, why attention replaced RNNs, how text becomes tokens and tokens become embeddings, hands-on via Hugging Face's `transformers` library. *(See [Deep Learning Practice](/topic/practice-deep-learning), [PyTorch Deep Dive](/topic/practice-pytorch-deep).)*
4. **LLM Development** — using these models day to day: the OpenAI/Anthropic/Gemini SDKs directly, Ollama for running models locally, vLLM basics for serving, and the parameters (temperature, top-p, context window, function calling) that actually change behavior. *(See [LLM Landscape](/topic/llm-landscape).)*
5. **AI Frameworks** — the orchestration layer wiring models, tools, and memory together: LangChain/LangGraph for building the flow, LlamaIndex if RAG is the focus, MCP for standardizing tool connections. *(See [LangChain Practice](/topic/practice-langchain), [LangGraph Practice](/topic/practice-langgraph).)*
6. **RAG** — grounding the model in your own data: pgvector/Qdrant/Weaviate, chunking, hybrid search, reranking, RAGAS evaluation — full pipeline in [section 03](#03-rag-vector-databases) above, deeper still in [RAG, Deeper](/topic/rag-deeper).
7. **AI Agents** — giving the model autonomy: LangGraph for stateful multi-step agents, guardrails (NeMo Guardrails/Lakera) to keep it from going off the rails. *(See [LangGraph Practice](/topic/practice-langgraph), [Code Drills 10 — LangGraph](/topic/code-drills-langgraph-agents).)*
8. **Production AI** — the engineering that decides whether this survives real traffic: FastAPI, Docker/K8s, Redis, vLLM, Prometheus/Grafana — full detail in [section 06](#06-python-backend) and [section 02](#02-llm-infrastructure-inference) above.
9. **AI System Design** — putting it all together on a whiteboard, naming the actual tools at each stage — where interviewers really differentiate candidates. *(See [System Design Prep](/topic/system-design).)*
10. **Interview Preparation** — turning all of the above into answers you can say out loud, with the tool names attached, without notes.

> "Most people stop after learning LangChain. That's exactly where the learning begins. Companies don't hire you because you know a framework — they hire you because you know *when* to use it, *why* you chose it, and *how* you'll make it work in production." That's what naming the actual tool at every step above is for — it's the difference between "I'd use a vector database" and "I'd use pgvector because we're already on Postgres and the HNSW index since 0.5 is fast enough that a dedicated service isn't worth the ops overhead."

---

## 09 · 10 Papers Worth Reading

A reading-list post makes an argument through curation, not just through captions — three of these ten are quietly making the same point (see the callout at the end). What follows is what each paper is actually claiming and the mechanism behind it, not just a one-line caption.

### 1. On the Theoretical Limitations of Embedding-Based Retrieval

A single dense embedding is a fixed-dimension vector, and "relevance for this query" is really "return this exact subset of documents." The proof works like the classic rank/communication-complexity arguments used to show what a matrix of a given rank can and can't represent: the number of distinct query→relevant-subset mappings a *d*-dimensional embedding space can realize is bounded by a function of *d*, and that bound grows far too slowly to cover every possible relevance pattern over a real-sized corpus. Concretely — for a fixed embedding dimension, there exist query/document combinations where *no* embedding, however well-trained, can rank the right documents into the top-k, because the required decision boundary literally isn't expressible as a dot-product threshold in that many dimensions.

> Not "train the embedding model better" — a hard mathematical ceiling. This is the formal argument for why the hybrid search pipeline in [section 03](#03-rag-vector-databases) isn't a nice-to-have: no dense retriever alone can be complete for arbitrarily complex relevance, no matter how big it is.

### 2. Agentic Context Engineering

Argues for treating what you feed an agent as a living playbook that gets updated turn over turn, not a fresh one-shot summary. It names two specific failure modes: **brevity bias** — a context-compression step optimized for conciseness strips out exactly the caveats and edge-case handling that matter most, because they read as "detail" rather than "signal"; and **context collapse** — repeatedly summarizing a summary in a long agent loop loses a bit of fidelity each pass, and errors compound until the context converges to something confidently wrong.

> This is the real mechanism behind "my agent worked great in the demo and fell apart after 20 turns." Fix implied: append to context like a changelog, don't regenerate a fresh summary every loop.

### 3. Towards a Science of Scaling Agent Systems

Tests 260 multi-agent configurations against single-agent baselines. Spread: +80.8% to −70.0% relative to baseline. The swing factor was whether the coordination topology — how agents split work and hand off — matched the task's actual structure. Parallelizable, independent subtasks benefit from more agents; tasks with tight sequential dependencies get *worse* with more agents, because handoff overhead and error compounding across agents dominates.

> "Add more agents" is a regression more often than an upgrade unless the task is genuinely decomposable. The design decision is coordination topology, not agent count — relevant directly to the AI Agents step in [section 08](#08-the-10-step-roadmap).

### 4. Agent Learning via Early Experience

Sits between imitation learning (mimic expert demonstrations, no exploration) and full RL (learn from a reward signal via costly trial and error). The agent takes its own actions in the environment and learns from the actual consequences those actions produced — not a hand-designed reward function, just "what happened after I did that" as the training signal.

> Relevant for improving an agent past what behavior-cloning off human trajectories gives you, without standing up a full RLHF pipeline and a reward model.

### 5. Recursive Language Models — *MIT*

Instead of cramming a huge document into the context window — which degrades well before the nominal token limit (the "lost in the middle" effect) — treat the long input as an environment the model can recursively operate over: it issues sub-queries against portions of the document, gets results back, and recurses, closer to a model-driven divide-and-conquer than a fixed chunking scheme. Reported to handle inputs two orders of magnitude past the stated context limit.

> A structurally different answer to "handle a huge document" than RAG's chunk-and-retrieve — the model navigates adaptively instead of a uniform pre-chunk. Worth it specifically when a document's relevant structure doesn't chunk cleanly.

### 6. Small Language Models are the Future of Agentic AI — *NVIDIA Research*

Profiles what most agent tool-calls actually look like in production — parsing a tool's output into a schema, choosing between a small fixed set of next actions, formatting a response — and argues these are narrow, repetitive, low-entropy decisions, not open-ended reasoning. A well-tuned small model (roughly 1–3B params) matches a frontier model's accuracy on these specific calls, because the task doesn't need frontier-scale general reasoning — you're paying GPT-4o-class pricing for what's structurally closer to a classification problem.

> Directly actionable: audit an agent pipeline's tool-calls by narrowness, route the boring 80% (routing, formatting, simple classification) to a small fine-tuned or distilled model, and reserve frontier-model cost for the genuinely hard reasoning steps. Same tiered-routing lever from [section 02, Q10](#02-llm-infrastructure-inference), now with a research argument for why it holds.

### 7. The AI Productivity Index

Built from real professional tasks pulled from banking, consulting, law, and medicine — not academic QA puzzles. The best-performing frontier model still cleared only 67.0%.

> A sharp counter to "AI is about to replace knowledge work" — on tasks that actually look like real professional work, the best model available still fails roughly a third of the time. Good number to have on hand when someone overclaims in an interview.

### 8. HallusionBench

Separates two failure modes in vision-language models that usually get lumped together as "hallucination": **language hallucination** (the model answers from its language-prior/training data regardless of what's actually in the image) versus **visual illusion** (it genuinely misperceives the image content). It measures this with paired questions over subtly modified images, scored on question-pair accuracy — credit only if the model gets *both* the original and the modified version right, a far harsher bar than plain per-question accuracy. Best model: 31.42% on that stricter metric.

> If evaluating a multimodal system, the per-question accuracy numbers quoted elsewhere are almost certainly inflated relative to what this stricter pairwise metric would show — the methodology matters as much as the headline number.

### 9. GLiNER2

The original GLiNER reframes named-entity recognition as matching text spans against natural-language descriptions of entity types, using a small bidirectional encoder (BERT-scale, not generative) — so it does zero-shot NER for entity types it's never seen a labeled example of, purely from a description. GLiNER2 extends the same small-encoder architecture to also handle text classification and structured field extraction, unifying three tasks people often solve by prompting a full LLM, in one sub-100M-parameter model.

> Same thesis as paper 6: for NER/classification/extraction specifically, reaching for GPT-4o is overkill — a purpose-built small encoder is faster, cheaper, and often more consistent for exactly this shape of task.

### 10. Document Summarization with Conformal Importance Guarantees

Applies **conformal prediction** — a statistical technique that wraps around any existing model and, using a calibration set, produces a provable coverage guarantee (e.g., "with 95% confidence, X holds") without touching the model's internals or retraining it — to summarization specifically. The result is a calibrated, provable guarantee about what fraction of "important" content survives into the summary, as a wrapper around whatever summarizer is already running.

> Summarization eval is normally vibes-based (a ROUGE score, or "looks fine to me"). This gives an actual statistical guarantee that could go in a compliance document — relevant for anything summarization-adjacent in legal, medical, or finance.

> **Read 1, 5 & 6 together.** Paper 1 says dense embeddings have a hard mathematical ceiling on what they can retrieve. Paper 6 says most of what an agent actually needs a "big" model for is narrow enough that a small model suffices. Paper 5 says even the context-window limit itself is softer than assumed if the model recurses instead of relying on retrieval. Put together: a lot of RAG/agent stacks reach for *embedding model + vector DB + frontier model* as the default architecture, when the actual bottleneck in each layer often has a cheaper or more reliable fix than "bigger embedding model, bigger LLM."

---

## 10 · More From a Head of AI/ML

A second interview-question sweep from a different poster, framed explicitly "from a Head of AI/ML perspective." Most of it lands on ground already covered in sections 01–08 (tokenization, embeddings, chunking, quantization, hosted-vs-open-source). What's below is only what's genuinely new: three concrete comparisons, an agent-memory pattern, a worked debugging example, and — the most useful part — four comments from senior engineers naming what actually separates a strong answer from a rehearsed one.

### What's the role of positional encoding in attention?

Attention on its own has no notion of order — the same set of tokens in any order produces identical dot-product computations, since attention is just weighted sums over pairwise similarities. Positional encoding injects order into each token's representation before attention runs. Two flavors worth naming precisely: the original Transformer's **absolute sinusoidal encoding** (fixed sin/cos functions of position, added to the token embedding) versus **RoPE** (rotary position embeddings — used in Llama, Mistral, Qwen, GPT-NeoX-family models), which instead rotates the query/key vectors by an angle proportional to position, so the dot product between two tokens' Q and K naturally encodes their *relative* distance rather than absolute position. That relative-distance property is exactly why RoPE-based models extend to sequence lengths longer than training more gracefully than sinusoidal ones — and why techniques like NTK-aware scaling and YaRN exist specifically to stretch RoPE further for long-context fine-tunes.

### LoRA vs. QLoRA vs. full fine-tune — what's the actual tradeoff?

| Approach | What's trained | Memory reality | When to pick it |
|---|---|---|---|
| **Full fine-tune** | Every weight | Highest — Adam optimizer state alone is ~4x model size (params + grads + 2 momentum terms) in fp32; needs multi-GPU or DeepSpeed ZeRO past ~7B | Large proprietary dataset, real compute budget, and LoRA has empirically plateaued below target quality |
| **LoRA** | Small injected low-rank adapter matrices (typically `q_proj`/`v_proj`, sometimes all linear layers) — ~0.1–1% of total params | Low — no optimizer state needed for frozen base weights | Default choice for domain adaptation, style, or behavior tuning — gets you 90%+ of full fine-tune's gain |
| **QLoRA** | Same LoRA adapters, but the frozen base is also quantized to 4-bit (NF4 via `bitsandbytes`) during training, with double quantization + paged optimizers | Lowest — fine-tune a 65–70B model on one 48GB GPU where full fine-tune or fp16 LoRA wouldn't fit | Same use case as LoRA, but the base model is too big for the GPU otherwise |

### How do you design system prompts that stay robust across different users?

The trap is validating a system prompt against the 5 examples tested and having it break on phrasing nobody anticipated. Concretely: enumerate the personas/intents actually expected — not just the happy path — give it explicit fallback instructions for out-of-scope asks, and, the part people skip, build a real eval set of adversarial and edge-case inputs and regression-test the prompt against it in CI every time it changes, the same way code gets regression-tested.

**Tools:** **promptfoo** is built specifically for this — YAML-defined test cases run against a prompt on every change, so a "robustness fix" for one user doesn't silently break behavior for another.

### How do you track, version, and backfill changing context?

Treat context as a first-class versioned artifact. Log the exact context sent with every request — not just the user's message, but the retrieved chunks, the system prompt version, any injected memory — tagged with a content hash or version ID. When the underlying context changes (say, a company policy doc gets updated), it's precisely knowable which past requests were served under the old version, and whether to backfill/reprocess them.

**Where it lives:** usually the same observability tool already logging requests — **Langfuse**/**Arize Phoenix** — with a prompt/context version field added to every trace, not a separate system.

### How do you build and maintain agent memory?

Split it in two. **Short-term/working memory** — the current task's state — usually just lives in context or a scratchpad and dies with the session. **Long-term memory** — facts that should persist across sessions — gets stored as embeddings in the same vector DB as the RAG index (or a separate namespace), retrieved the same way: top-k relevant memories pulled in each turn rather than the whole history replayed.

**Tools:** **LangGraph**'s memory checkpointing if already on that framework, or **Mem0** — a library purpose-built for exactly this "write important facts, retrieve top-k relevant ones per turn" pattern — to avoid hand-rolling it.

### How do you evaluate retrieval quality — precision@k, reranking, citation?

**Precision@k**: of the top-k chunks retrieved, what fraction are actually relevant — needs a labeled relevance judgment (human or LLM-judge) to compute. Beyond that, a **citation-based eval** checks something stricter: does each claim in the generated answer trace back to a *specific* retrieved chunk, versus merely being "consistent with" the general topic. That catches subtler hallucination than a faithfulness score alone, because a model can produce a plausible claim that happens to align with the topic without any single chunk actually supporting it.

### CI/CD for LLM workflows — what's actually different from traditional ML?

Traditional ML CI/CD is deterministic — same model version plus same input equals same output, so exact-match regression tests work. LLM workflows aren't: even at `temperature=0`, a provider-side model update can silently change behavior, and "correctness" often isn't a single right answer. So LLM CI/CD needs eval-based gates instead of assert-equal tests — run a fixed eval set through an LLM-as-judge or a rubric scorer on every prompt/model change, and gate the merge on a score threshold, not an exact match.

**Tools:** **promptfoo** or a custom eval harness wired into **GitHub Actions**, scoring against the same eval set every time.

### Walk me through a debugging session for incorrect LLM outputs.

Concretely, not generally — this is the order to actually run it in:

- Pull the trace for that exact request by ID from **Langfuse** (or whatever's logging).
- Check what got **retrieved**. If the right chunk isn't in the top-k, it's a retrieval bug — stop here, go check embedding/chunking, not the prompt.
- If the right chunk *was* retrieved but the answer still ignored it — check whether it got **truncated or buried mid-prompt**; models attend worse to the middle of a long context, so reorder or shorten before touching anything else.
- If the chunk was present, intact, and well-placed — check the model's **raw output before post-processing**. A surprising fraction of "wrong answers" are actually a downstream parsing bug mangling a correct response.
- Only once retrieval, prompt placement, and parsing are all ruled out should the model itself be suspected — and even then the fix is usually a prompt clarification, not an immediate jump to fine-tuning.

> **What actually separates strong candidates — straight from the comments**
> - **Sathish Kumar Subramani · Senior Engineering Manager, Generative AI:** "Add a question on evaluation design: how would you know the system actually improved after changing the model, prompt, retrieval strategy, or context? That's what separates experimentation from disciplined engineering."
> - **Jaswindder Kummar · Engineering Director, Cloud/Platform:** "Add failure analysis across the entire AI stack. When an output is wrong, can the candidate determine whether the problem came from retrieval, context, orchestration, the model, or the evaluation layer itself?" (This is exactly the debugging walkthrough above.)
> - **Prashant Varshney · Senior AI Engineer & Architect:** "The strongest interview questions force candidates to reason about tradeoffs rather than recite definitions. Production AI rarely gives you a clean choice between quality, latency, cost, and reliability."
> - **Dewank Mahajan · AI & Analytics:** "The strongest question here might be 'can you solve this without an LLM?' Good AI engineering includes knowing when deterministic software is the better architecture." (Same as the hiring manager's favorite question in [section 01](#01-what-a-hiring-manager-actually-wants-to-know).)

---

*Source: six posts screenshotted to WhatsApp on 2026-08-21 — a hiring-manager checklist, five interview-question categories (LLM Infra, RAG, Prompt Engineering, MLOps, Backend), a 7-technique optimization cheat-sheet, a 10-step roadmap, a 10-paper reading list, and a second Head-of-AI/ML question sweep with its comment section. Paper links in the original posts were tracking redirects and aren't reproduced here — search each title directly.*
