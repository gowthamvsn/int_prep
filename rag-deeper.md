# RAG, Deeper — Advanced Retrieval, Evaluation, and Knowledge Graphs (GraphRAG)

`core-technical-depth.md` and the NCA-GENL guide cover basic RAG: chunk → embed → store → retrieve → stuff into the prompt. That basic version works fine on a demo. It falls apart in production. Retrieval returns the wrong chunks. Answers look plausible but aren't actually grounded in anything. And nobody can tell you why it failed. This doc closes that gap.

### Why does "just embed and retrieve" stop working in practice?
Because a single dense-embedding similarity search is a blunt instrument. Quick recap of those three words. An *embedding* turns a piece of text into a list of numbers, positioned so similar meanings land near each other. *Dense* just means these are learned vectors, not sparse keyword counts. A *similarity search* returns the k stored chunks whose numbers sit closest to the query's numbers — the "top-k." (Full mechanics in `nca-genl` and `core-technical-depth.md`.)

Here's where it's blunt. It's good at "this text is topically similar." It's bad at exact terms — product codes, names, acronyms. It's bad at multi-part questions. And it has no idea whether the top-k chunks it returned are actually enough to answer the question. Every technique below fixes one specific failure of this naive version.

**Visual + memory hook — every technique below slots into ONE of five stations on the same assembly line; memorize the line, not eight separate techniques:**
```
 query          REWRITE          RETRIEVE            RE-RANK           COMPRESS      GENERATE
  │            (query rewrite,   (hybrid search:      (cross-encoder,   (trim each     │
  │             HyDE — fix a     BM25 + dense —       50 candidates     chunk to just   │
  │             bad SEARCH       fix a MISSING        down to real      the relevant    │
  │             query)           MATCH)               top 5)            sentences)     │
  └──────────────────▶─────────────────▶─────────────────▶─────────────────▶───────────▶  answer
                                                                                    (faithfulness
                                                                                     checked here)
```
**Remember it as an assembly line, not a grab-bag of tricks.** Every failure mode in this doc maps to exactly one station. A bad search query is a REWRITE problem. A missed keyword is a RETRIEVE problem. A mediocre top-5 is a RE-RANK problem. Wasted tokens on irrelevant text is a COMPRESS problem. A hallucinated claim is a GENERATE problem. When a RAG system misbehaves, walk the line station-by-station — check which stage's output first looks wrong. That finds the fix far faster than guessing which of eight techniques to reach for.

> 🔗 **Hands-on reps:** [Code Drills 8 — Where Simple RAG Breaks](/topic/code-drills-rag-langchain#cluster-4-where-simple-rag-breaks-and-the-direct-fixes)

## Cluster 1 — Walking the Assembly Line, Station by Station

> **TL;DR**
> - Naive RAG breaks in specific, fixable ways — every technique below patches exactly one station on a five-stage line: **rewrite → retrieve → re-rank → compress → generate**.
> - **Query rewriting / HyDE** fix a bad search query before retrieval even runs.
> - **Hybrid search** (dense + BM25) catches exact terms plain embeddings miss; a **cross-encoder re-ranker** takes a slower, sharper second look at a fast first pass.
> - **Contextual compression** trims each retrieved chunk down to just the relevant sentences; **multi-hop RAG** loops the whole line when one question needs facts chained across several documents.
> - **RAGAS, ARES, and G-Eval** grade the pipeline three different ways — prompted judge, trained judge with error bars, and a judge that writes its own rubric.
> - None of this is free: accuracy, latency, and cost trade against each other, and a huge context window doesn't make retrieval unnecessary.

### Fixing the query before retrieval even runs
People ask questions the way they'd ask a person. Documents aren't written that way. So the first station on the line is REWRITE.

**Query rewriting** asks an LLM to rephrase the user's question into a better search query, before anything gets retrieved. It can expand abbreviations, split a compound question into sub-questions, or add likely synonyms.

**HyDE (Hypothetical Document Embeddings)** takes a different angle on the same problem. Instead of embedding the question, you ask an LLM to write a *hypothetical answer* to it first — even if that answer might be wrong — and embed that instead. A generated answer tends to be written in the same style and vocabulary as the real documents. So it often matches better than the bare question does.

### Hybrid search: catching what dense embeddings miss
Once a well-formed query reaches the RETRIEVE station, dense embeddings still have a blind spot. They're great at meaning. They're bad at exact tokens. A query for "error code E402" can retrieve text that's semantically similar but never actually contains "E402."

**Hybrid search** fixes this by running two searches at once. One is a classic keyword search — BM25, a frequency-weighted version of TF-IDF. Both BM25 and TF-IDF are pure text-matching scores: they reward words that appear often in one document but rarely across the whole corpus, no embeddings involved. The other is the usual dense vector search. Hybrid search combines the two ranked lists, commonly with **Reciprocal Rank Fusion**: `score = sum(1 / (k + rank_in_each_list))`. You get semantic recall and exact-match precision at the same time, instead of picking one.

### Re-ranking: a second, slower, more accurate look
Initial retrieval — dense or hybrid — is optimized to be fast across millions of chunks. That speed costs accuracy.

A **bi-encoder** embeds the query and every document independently. Scoring is just a dot product. It's cheap. But the query and a document never actually look at each other.

A **cross-encoder** re-ranker is slower but far more accurate. It feeds the query and one candidate document in together. They attend to each other. It outputs a single relevance score.

The standard pattern is a cascade. Retrieve 50 candidates fast with a bi-encoder. Re-rank them down to a real top 5 with a cross-encoder. Fast-and-broad, then slow-and-narrow — the same two-stage shape shows up in classic search engines too, not just RAG.

### Compression: trimming what actually reaches the prompt
Even a genuinely relevant top-5 can waste tokens. A whole chunk might be relevant only because of one sentence buried in the middle of it.

**Contextual compression** fixes that. It runs each retrieved chunk through a cheap LLM call that pulls out just the sentences relevant to the query, before the chunk ever reaches the final prompt. That trims token usage. It also cuts the chance the model gets distracted by irrelevant nearby text.

### Multi-hop RAG: looping the whole line
Everything above assumes one retrieval pass is enough. It isn't always.

Take "what's the revenue of the company that acquired the startup founded by X?" That needs three separate lookups: who founded the startup, who acquired it, and that acquirer's revenue. No single chunk has the whole answer.

**Multi-hop RAG** handles this by running retrieval in a loop. Retrieve. Let the LLM decide what's still missing. Retrieve again with a refined query. Repeat until there's enough to answer. This is retrieval as an *agentic* loop, not a single lookup — see `practice-langgraph.md` for the orchestration mechanics.

```
        ┌─────────────────────────────────────────────┐
        │                                               │
        ▼                                               │
   retrieve (full line: rewrite→retrieve→               │
             re-rank→compress)                           │
        │                                               │
        ▼                                               │
   "do I have enough to answer yet?"  ── no, missing X ──┘
        │
       yes
        │
        ▼
     generate final answer
```
Each hop is a full trip down the five-station line, not a shortcut. That's what makes multi-hop expensive. It's also what makes it work on chained, multi-fact questions that plain single-pass retrieval can't touch.

### Measuring it: RAGAS's four scores
Once the full line produces a final answer at GENERATE, a gut feel isn't enough to know whether it worked. The standard metric set, popularized by the **RAGAS** framework, scores each station separately:
- **Context precision** — of the chunks retrieved, how many were actually relevant? (Scores RETRIEVE/RE-RANK.)
- **Context recall** — of the chunks that *were* relevant somewhere in the corpus, how many did retrieval actually find? (Scores RETRIEVE.)
- **Faithfulness** — does the generated answer only use claims the retrieved context supports, or did the model hallucinate something that isn't there? (Scores GENERATE.)
- **Answer relevance** — does the generated answer actually address the question asked? A faithful-but-off-topic answer still fails this one. That's a different failure than faithfulness.

Splitting evaluation this way tells you where on the line to fix the pipeline. Bad context precision means improve retrieval or re-ranking. Bad faithfulness means the generation prompt needs stricter grounding instructions — not a retrieval fix at all.

### ARES: trained judges plus a confidence interval
RAGAS's scoring loop is basically this: write a careful prompt, hand the judge LLM the question, context, and answer, and parse a score back out. That's fast, and it needs zero labeled data. But the number you get is a single point estimate, from a judge whose own biases you never measured. If it reads 0.81 this week and 0.78 next week, you can't say whether the pipeline actually got worse, or the judge just wobbled.

**ARES (Automated RAG Evaluation System)** attacks that weakness two ways.

First, **trained judges instead of prompted ones**. Rather than prompting a large model per example, ARES fine-tunes small, cheap classifier-style LLM judges — one per dimension: context relevance, answer faithfulness, answer relevance. Scoring a large eval set now costs a fraction of a GPT-4-class judge call per row.

Second, and the more distinguishing idea, **a statistical correction step** called prediction-powered inference (PPI). ARES holds out a small set of human-annotated examples. It uses them to measure how the cheap judge systematically deviates from human judgment. Then it combines "many machine labels" with "few human labels" into a **confidence interval** around the estimated score, instead of a bare number. You end up able to say "context relevance is 0.78 ± 0.04" — a claim you can defend — instead of "context relevance is 0.78," a number that just moved.

Two honest caveats, worth checking against the ARES paper before quoting it as fact. In the published method, the judges are trained largely on synthetically generated query/answer data derived from your own corpus, not on human annotations — human annotations are reserved mainly for the PPI calibration step. And the judge backbone is a small fine-tuned language model, not a frontier LLM. The safe summary: **ARES trades RAGAS's zero-setup convenience for trained cheap judges plus human-calibrated error bars.**

### G-Eval: writing the rubric on the fly
RAGAS and ARES are both locked to a fixed menu of RAG-specific dimensions. When you need to grade a criterion neither of them ships with, **G-Eval** is the answer: describe the criterion in a sentence and let a strong LLM build the rubric for you.

Two mechanics make it more than "ask GPT-4 to rate this 1-5."

First, **chain-of-thought-generated evaluation steps**. You give the model a short task definition and the criterion — say, "rate coherence 1-5." G-Eval has the LLM write out the evaluation steps itself first, an auto-generated scoring form specific to that criterion, and only then applies that form to each example. The rubric gets generated once, not improvised per example. That's what keeps grading consistent.

Second, **probability-weighted scores**. Instead of taking the single discrete integer the judge emits at face value, G-Eval reads the model's token probabilities over the candidate score tokens and computes the expected value (`score = Σ p(s) · s`). LLM judges cluster hard on round answers — ask for a 1-5 score and you mostly get 3s and 5s back. Raw discrete scores produce huge ties and poor correlation with human rankings. The weighted score is continuous. It breaks those ties, and in the paper it correlates better with human judgment.

One caveat worth flagging rather than papering over: the weighting step needs access to token logprobs, so it only works against APIs that expose them. Against a model that doesn't, you get G-Eval's auto-generated rubric but not its continuous scoring — losing the more interesting half. G-Eval also isn't RAG-specific at all. It's a general LLM-as-judge recipe that happens to work fine on RAG outputs too.

### Which one do you actually reach for
| Framework | What it optimizes for | Reach for it when |
|---|---|---|
| **RAGAS** | Fast, reference-free scoring of the four standard RAG dimensions — no labeled data, no training step | The default first move: you need a per-station read on a pipeline today and have zero annotations |
| **ARES** | Cheap trained judges plus a human-calibrated confidence interval on the estimated score | You have (or can afford) a small human-labeled set and need to *defend* "retrieval got better," not just watch a number move |
| **G-Eval** | Flexible grading against any criterion you can describe in a sentence, at finer score granularity | The thing you care about isn't one of the standard RAG dimensions — tone, safety, does-it-cite-a-source — or you're grading non-RAG generation entirely |

All three are LLM-as-judge under the hood. They just differ in what they do to the judge. RAGAS **prompts** it. ARES **trains** it and puts error bars on it. G-Eval **writes its own rubric** and reads its hesitation — the token probabilities — instead of just its final answer. And all three inherit every LLM-as-judge bias: position, verbosity, self-preference. So all three still need a human spot-check before you treat the number as ground truth.

### The accuracy/latency/cost triangle
Every station on the line can be made more accurate. So why not just max out accuracy everywhere and call it done? Because latency, cost, and relevancy behave like a fixed budget split three ways — not three independent dials. Pushing hard on one corner tends to cost you one of the other two. It's worth naming this explicitly as its own tradeoff triangle in an interview, the same way the CAP theorem names a fixed tradeoff in distributed systems.

| Lever | Helps | Costs |
|---|---|---|
| **Caching** (exact-match via an AI gateway, or semantic caching where the query doesn't have to match exactly) | Latency, cost (a cache hit skips retrieval and generation entirely) | Relevancy, if semantic caching serves a "close enough" cached answer to a query that actually needed a fresh one |
| **Smaller model for simple sub-tasks** (e.g. a cheap model for a summarization step, reserving the frontier model for the reasoning step) | Latency, cost | Relevancy/quality on whichever step got downgraded |
| **Shrinking embedding dimensions** | Latency, cost (smaller vectors, faster ANN search) | Relevancy (less representational capacity per vector) |
| **Cross-encoder re-ranker instead of an LLM-as-judge re-check** | Cost and latency, *for the same relevancy gain* — the one lever here that isn't a straight tradeoff, since a small re-ranker model gets most of an LLM-judge's relevancy benefit at a fraction of the price | — |
| **Raising reasoning effort** (adaptive thinking, a "think step by step" instruction) | Relevancy/accuracy on genuinely hard queries | Latency and cost, directly — more tokens, more time |

The interview-ready version: don't claim you can improve accuracy, latency, and cost all at once with no tradeoff. Name which corner you're spending down to buy the other two. `Designing an LLM Inference System at Scale` (`system-design-prep.md`) applies this same discipline to the compute/memory/latency tradeoffs on the serving side.

### Doesn't a huge context window just replace RAG?
Context windows now reach 1M+ tokens. Some models could fit your whole knowledge base directly in the prompt. Doesn't that just replace RAG?

No — not for the workloads RAG is actually built for. Three reasons, and each one still holds even with a huge window.

1. **Cost and latency scale with input tokens.** Re-sending a million tokens on every single query, when the answer only needed three paragraphs of it, means paying for and waiting on 999,997 tokens of pure overhead — every time, forever. RAG's retrieval step exists specifically to avoid that repeated cost.
2. **"Fits in the window" isn't the same as "the model reliably uses all of it."** The well-documented **"lost in the middle"** effect shows retrieval-from-context accuracy dropping for facts buried in the middle of a very long prompt, even when the tokens are technically present. A bigger window doesn't guarantee the model actually finds the one fact that matters.
3. **A long-context approach still has a hard ceiling, and no update story.** A knowledge base that grows past the window size, or changes hourly, needs a retrieval mechanism regardless of how large the window is. RAG's index can grow and be re-embedded incrementally, without ever touching the prompt budget.

The honest framing for an interview: a bigger context window shrinks the number of cases where RAG is the only option. It doesn't eliminate the reasons RAG exists. Cost-per-query, precision on buried facts, and a knowledge base that outlives any fixed window are all still real at 1M tokens.

<details>
<summary><strong>Self-check — answer before revealing</strong></summary>

1. A user searches for an exact product SKU and gets back semantically related but wrong products. Which station is broken, and what fixes it?
2. Why re-rank with a cross-encoder instead of just using a cross-encoder for the entire retrieval step?
3. RAGAS reports high faithfulness but low answer relevance. What does that combination actually tell you?
4. You've been handed ~200 human-labeled RAG examples and asked to prove a retrieval change actually helped. Why reach for ARES over RAGAS here?
5. Why does G-Eval bother reading token probabilities instead of just using the 1-5 score the judge printed?
6. A colleague claims a 1M-token context window makes RAG obsolete. What's the one-sentence rebuttal?

**Answers**
1. The RETRIEVE station — dense embeddings are weak on exact tokens like SKUs. Hybrid search (BM25 alongside the dense search) fixes it.
2. A cross-encoder scores query+document pairs jointly, which is accurate but needs a full forward pass per pair — too slow against millions of documents. The two-stage cascade (fast bi-encoder down to ~50, then cross-encoder down to 5) gets both speed and accuracy.
3. The model is grounded (not hallucinating) but not actually answering the question asked — a generation-prompt problem, not a retrieval problem.
4. RAGAS gives a point score from a prompted judge whose own error you never measured. ARES's PPI step uses the labeled set to calibrate that error into a defensible confidence interval — without labels, ARES has nothing to calibrate against, so RAGAS is the right default.
5. LLM judges cluster on round numbers (mostly 3s and 5s on a 1-5 scale), which ties badly and correlates poorly with human rankings. Probability-weighting the score tokens produces a continuous score that breaks those ties.
6. A bigger window shrinks the number of cases where RAG is the only option, but it doesn't eliminate why RAG exists — repeated-query cost, "lost in the middle" accuracy loss on buried facts, and a knowledge base that outgrows any fixed window are all still real.
</details>

> **Recap**
> Naive RAG fails in specific, nameable ways, and every technique here patches one station on a five-stage line: rewrite the query (query rewriting/HyDE), retrieve broadly (hybrid search), re-rank sharply (cross-encoder), compress what's kept, and loop the whole thing for multi-hop questions. RAGAS/ARES/G-Eval grade those stations independently so you know which one to fix. None of it is free — accuracy, latency, and cost trade off against each other, and a big context window narrows RAG's necessity without removing it.

### Summary example
Walk a query for "error code E402" down the line. HyDE rewrites it into a hypothetical answer. Hybrid search (BM25 + dense) retrieves candidates, so the exact code isn't missed. A cross-encoder re-ranks 50 candidates down to 5 genuinely relevant ones. Compression trims those 5 down to just the relevant sentences before they hit the prompt. A single-hop question like this needs only one trip down the line. A harder question — "what's the revenue of the company that acquired the startup E402 belonged to" — would loop the whole line multiple times instead.

Now run RAGAS on the output. High context precision but low faithfulness points straight at the GENERATE station, not back at retrieval. That tells you exactly which station to revisit, instead of guessing among all five.

Say the fix needs to be proven, not eyeballed: "did tightening the grounding prompt really raise faithfulness, or did the judge just wobble?" That's the point to spend a few hundred human annotations and switch to ARES for a confidence interval instead of a point score.

And if what you actually need graded is "did the answer cite the maintenance procedure it used" — not one of the four standard dimensions — write that criterion in a sentence and let G-Eval generate the rubric for it. Same assembly line. Three different sharpnesses of measuring tape.

---

## Cluster 2 — Knowledge Graphs and GraphRAG: A Different Kind of Retrieval Entirely

> **TL;DR**
> - A vector DB answers "what text is similar to this." A **knowledge graph** answers "what's exactly, explicitly connected to this" — a different kind of lookup entirely.
> - Building one means **entity extraction → relation extraction → entity resolution → store** (Neo4j/Cypher, or `networkx` at small scale).
> - **GraphRAG** beats plain vector RAG on multi-hop questions and "give me the whole picture" queries — at the cost of real upfront engineering (extraction is noisy, entity resolution is genuinely hard).
> - **Node2vec** gets you embedding-style similarity search that still respects graph structure, when you want both.

### What a knowledge graph answers that a vector DB can't
A knowledge graph stores facts as **(entity) → [relationship] → (entity)** triples, instead of paragraphs of text: `(NVIDIA) -[acquired]-> (Mellanox)`, `(Mellanox) -[specializes_in]-> (networking)`. Entities are nodes. Relationships are labeled edges. A vector database answers "what text is similar to this." A graph answers "what is directly, explicitly connected to this." That's a fundamentally different, exact kind of lookup than similarity search.

### Building one from raw text
Getting from unstructured text to that structure takes four steps.

**Entity extraction (NER)** pulls out named things — people, orgs, products.

**Relation extraction** figures out how pairs of entities relate: "acquired," "works at," "caused." Often this means prompting an LLM with the sentence and asking it to output a structured triple.

**Entity resolution** merges "NVIDIA," "Nvidia Corp," and "NVDA" into one canonical node, so the graph doesn't fragment into duplicates. This step is usually the hardest of the four — harder than extraction itself.

Finally, you **store it**. A graph database like **Neo4j** works well (query language Cypher, e.g. `MATCH (a)-[:ACQUIRED]->(b) WHERE a.name = "NVIDIA" RETURN b`). At smaller scale, a `networkx` graph in Python is enough.

### When GraphRAG actually beats plain vector RAG
GraphRAG retrieves by **traversing relationships** in the graph, instead of — or alongside — similarity search. It shines exactly where plain RAG struggles.

One case: multi-hop questions, like "who are the competitors of companies my company has partnered with?" This is the same multi-hop gap Cluster 1 covers, just solved by graph traversal instead of an agentic retrieval loop.

The other case: questions that need the whole picture, not just the top-k most similar snippets. "Summarize everything connected to Project X" is the clearest example. Vector search would hand back the 5 most similar chunks. A graph traversal returns everything actually linked to Project X, however that happens to be phrased in the source text.

The tradeoff is real upfront engineering work — extraction is noisy, and entity resolution is genuinely hard. It's worth it for structured, relationship-heavy domains. It's often overkill for a simple FAQ bot.

### Getting graph structure and similarity search at once
Sometimes you want both: the semantic-similarity benefits of embeddings, and the structural information a graph carries. **Node2vec** (and similar methods) learns a vector for each node, so that nodes "close" in the graph — reachable via short random walks — end up close in vector space too. That gives you similarity search that respects graph structure. It also gives you a node's graph position as a feature you can feed into a downstream ML model.

```
   raw text  ──▶  entity extraction (NER)  ──▶  relation extraction
                                                       │
                                                       ▼
                                             entity resolution
                                        (merge "NVIDIA"/"Nvidia Corp"/"NVDA")
                                                       │
                                                       ▼
                                        store: Neo4j (Cypher) / networkx
                                                       │
                                        ┌──────────────┴──────────────┐
                                        ▼                              ▼
                              graph traversal query           Node2vec embeddings
                          ("everything connected to X")     (similarity search that
                                                              respects graph structure)
```

<details>
<summary><strong>Self-check — answer before revealing</strong></summary>

1. What's the fundamental difference between what a vector DB answers and what a knowledge graph answers?
2. Which of the four graph-building steps tends to be the hardest, and why?
3. "Summarize everything connected to Project X" — why does a graph traversal handle this better than top-k vector search?
4. When is GraphRAG overkill rather than a win?
5. What does Node2vec actually give you that raw graph traversal doesn't?

**Answers**
1. A vector DB answers "what's semantically similar to this text." A knowledge graph answers "what's explicitly, exactly connected to this entity" — similarity vs. exact structural relationship.
2. Entity resolution — merging "NVIDIA," "Nvidia Corp," and "NVDA" into one node. Extraction just has to spot that something is an entity; resolution has to recognize different mentions are the *same* entity, and getting it wrong silently fragments the graph.
3. Vector search returns the top-k most similar chunks, which caps out at a fixed handful of snippets. A graph traversal returns everything actually linked to Project X, regardless of how each connected fact happens to be phrased.
4. On structured, relationship-heavy domains it earns its keep; on something like a simple FAQ bot, the upfront extraction and entity-resolution engineering cost isn't worth it.
5. A vector representation of each node that respects graph structure (nodes close in the graph end up close in vector space) — similarity search that's still graph-aware, or a feature you can feed into a downstream model.
</details>

> **Recap**
> Knowledge graphs store exact (entity)→[relationship]→(entity) triples, built via extraction → relation extraction → entity resolution → storage. GraphRAG traverses those relationships instead of doing similarity search, which wins on multi-hop and "whole picture" questions but costs real engineering to build and maintain well. Node2vec bridges the two worlds when you want graph-aware similarity search too.

### Summary example
"Summarize everything connected to Project X" is exactly the query GraphRAG handles that plain vector RAG can't. Entity extraction, relation extraction, and entity resolution turn raw project documents into a triple store. A graph traversal from the "Project X" node then returns every directly-connected fact, regardless of how it's phrased. If a downstream model also needs a similarity-searchable representation of each entity's graph position, Node2vec produces exactly that — combining the graph's exactness with a vector's searchability.

## Practice Q&A (Self-Test)

### A user searches for an exact product SKU and the RAG system returns semantically related but wrong products. What's the fix?
Add hybrid search: a keyword/BM25 component running alongside the dense embedding search. Exact codes and identifiers are exactly what dense embeddings are weakest at. Exact-match keyword search is exactly what BM25 is strongest at.

### Why re-rank with a cross-encoder instead of just using a cross-encoder for the entire retrieval step?
A cross-encoder scores query+document pairs jointly. That's far more accurate, but it needs a full forward pass per pair — too slow to run against millions of documents. The two-stage pattern fixes this: fast bi-encoder retrieval down to ~50 candidates, then accurate cross-encoder re-ranking down to the final top 5. You get both speed and accuracy.

### RAGAS reports high faithfulness but low answer relevance. What does that combination actually tell you?
The model is generating answers fully grounded in the retrieved context — it's not hallucinating. But the answer isn't addressing the actual question asked. That points to a generation-prompt problem: the model needs clearer instructions to use the context to answer the specific question. It's not a retrieval or grounding problem.

### You've been given ~200 human-labeled RAG examples and asked to prove a retrieval change actually helped. Why is that the moment to reach for ARES over RAGAS?
RAGAS returns a single point score from a prompted LLM judge whose own error you never measured. A move from 0.78 to 0.81 could easily just be judge noise. ARES uses trained lightweight judges over the full set, plus prediction-powered inference against the small human-labeled set. That produces a confidence interval instead of a bare number — which is what turns "it went up" into a defensible claim. The labeled set is the price of admission. With zero annotations, ARES's calibration step has nothing to calibrate against, so RAGAS is the right default.

### Why does G-Eval bother reading token probabilities instead of just using the 1-5 score the LLM judge printed?
LLM judges cluster on a few round values. Ask for a 1-5 score across a hundred examples and you get mostly 3s and 5s back. Half the examples tie, and the ranking barely correlates with human judgment. Weighting the candidate score tokens by their probabilities, then taking the expected value, yields a continuous score that breaks those ties. The practical catch: it needs an API that exposes logprobs. Against a model that doesn't, you keep G-Eval's auto-generated rubric but lose its finer-grained scoring.

### When would GraphRAG clearly beat plain vector-similarity RAG?
When the question needs traversing explicit relationships across multiple hops — "who works at companies that partnered with my company's competitors," for example. Or when it needs a complete, structured view of everything connected to an entity. In both cases, "most similar text chunks" isn't the same as "everything actually and exactly related."

### Why is entity resolution often the hardest part of building a knowledge graph, not extraction itself?
Extraction just needs to spot that "NVIDIA," "Nvidia Corp," and "NVDA" are entities. The hard part is recognizing they're the same entity, so the graph doesn't fragment into duplicate, disconnected nodes that each only capture part of the real information. Bad entity resolution silently breaks graph traversal, even when extraction itself looks accurate.

---

## Cluster 3 — Data Quality Before Anything Gets Embedded

> **TL;DR**
> - Everything in Clusters 1-2 assumes clean input data. It usually isn't — and both failure modes below are silent: nothing errors, the pipeline just quietly degrades.
> - **Near-duplicate documents** crowd out genuinely different information in your top-k results — catch them with exact hashing, MinHash/Jaccard, or embedding similarity, cheapest first.
> - **Missing cross-document normalization** (mismatched units, currencies, reporting periods across documents) makes retrieval and generation both "succeed" while quietly returning the wrong number.

Everything in Clusters 1-2 assumes the source data going in was already clean. It usually isn't. Both failure modes below happen silently — nothing errors, the pipeline just quietly degrades.

### Near-duplicates: catching them before they crowd out real information
Corporate knowledge bases are full of the same document copied with a different date, a different filename, or one paragraph edited. If both copies get embedded and indexed, a single relevant fact now takes up two — or ten — of your top-k retrieval slots. That crowds out genuinely different information.

Three ways to catch it, cheapest first.

**Exact-match hashing** hashes the full document text. Identical hashes mean exact duplicates. It's trivial to catch, but it catches nothing else.

**MinHash / Jaccard similarity** breaks the text into small overlapping word sets, called "shingles." It estimates the overlap between two documents' shingle sets, and discards anything above a similarity threshold — say, over 90% overlap. This catches near-duplicates exact hashing misses, like a copy with a changed timestamp, without needing embeddings at all.

**Embedding cosine similarity** embeds every document and discards anything whose nearest neighbor exceeds a similarity threshold. It's more expensive, since it requires running the embedding model over everything first. But it catches paraphrased near-duplicates that share few exact words.

### Cross-document normalization: the failure that doesn't look like a data problem
Duplicates handled, there's a second silent failure. It doesn't even look like a data-quality issue at first.

Say one financial document states amounts in raw dollars, and another states them "in thousands" — a `5` on page 2 there means $5,000, not $5. If both get chunked and embedded independently, a query comparing the two will silently compare the wrong magnitude. Nothing errors. Retrieval and generation both "succeed." The number is just wrong.

Two real fixes. One: run a pass over each full document first, to extract document-wide metadata — units, currency, reporting period — and attach it to every chunk from that document before embedding. Two: generate a per-chunk summary with a separate model call, so a chunk saying "35" gets summarized as "$35,000," with the unit resolved before it's what gets embedded and retrieved.

Either way, the fix has to happen before chunking is finalized. You can't recover document-wide context from an isolated chunk after the fact.

```
  raw documents
       │
       ▼
  dedup pass (cheapest → most expensive)
    1. exact-hash match        → catches literal copies
    2. MinHash / Jaccard       → catches "changed the date" copies
    3. embedding similarity    → catches paraphrased rewrites
       │
       ▼
  normalization pass (per full document, BEFORE chunking)
    extract units / currency / reporting period
    → attach as metadata to every chunk from that doc
       │
       ▼
  chunk → embed → index          (Clusters 1-2 pick up from here)
```

<details>
<summary><strong>Self-check — answer before revealing</strong></summary>

1. Why does a near-duplicate document actually hurt retrieval quality, rather than just being harmless clutter?
2. Rank the three dedup methods from cheapest to most expensive, and name one thing each one catches that the cheaper method(s) miss.
3. What's the actual failure mode when two documents use different units (raw dollars vs. "in thousands") and neither gets normalized?
4. Why does the normalization fix have to happen *before* chunking, not after?
5. A pipeline shows no errors, yet a financial-comparison query returns a wildly wrong number. What's the first data-quality culprit you'd check?

**Answers**
1. It takes up top-k retrieval slots that genuinely different information could have used — a duplicate fact crowds out other relevant content instead of just sitting there unused.
2. Exact-hash (cheapest, catches literal copies only) → MinHash/Jaccard (catches near-duplicates like a changed timestamp, no embeddings needed) → embedding cosine similarity (most expensive, catches paraphrased rewrites that share few exact words).
3. A query comparing the two documents silently compares incompatible magnitudes — nothing errors, retrieval and generation both "succeed," and the answer is confidently wrong.
4. An isolated chunk has no way to recover document-wide context (like which unit convention the whole document uses) once it's separated from the rest of the document — the metadata has to be captured while the full document is still in view.
5. Missing cross-document normalization — mismatched units/currency/reporting period across the source documents being compared.
</details>

> **Recap**
> Clean data isn't a given — near-duplicates crowd out real information in top-k results (catch with hashing → MinHash → embedding similarity, cheapest first), and missing cross-document normalization silently compares incompatible units or periods. Both failures are invisible in the pipeline's error logs; they only show up as confidently wrong answers.

### Summary example
A legal-document RAG system for a law firm ingests thousands of internal memos. Exact-hash dedup catches the literal copy-pasted templates. MinHash catches the ones where someone changed the client name and date but kept 95% of the boilerplate. Embedding-similarity dedup catches a memo that was substantially rewritten but says the same thing.

Separately: a contracts corpus has some documents stating penalty clauses in "per day" terms, others in "per business day." That distinction needs to be extracted as metadata and attached to every chunk. Otherwise a query comparing two contracts' penalty terms silently compares incompatible units, and returns a confidently wrong answer — the exact same failure shape as the "$35 vs $35,000" example above.

---

## Video-Sourced Practice MCQs

A practice set on operational/production RAG engineering, sourced from a real YouTube RAG-interview-prep video. Deliberately NOT re-covering this file's existing advanced-retrieval material (hybrid search, re-ranking, RAGAS/ARES/G-Eval, GraphRAG) -- instead focused on the practical engineering side: why RAG is needed at all, diagnosing which pipeline STAGE causes RAG-specific hallucination, enterprise-scale design concerns (access control, metadata filtering), the production challenges list (data freshness tradeoffs, caching vs. parallel retrieval), and the correct end-to-end pipeline order. All wording is original.

<script type="application/json" class="topic-quiz-data" data-title="RAG, Deeper">
[
  {
    "d": "Why RAG Exists",
    "q": "An LLM already has broad general knowledge from training. What are the two core reasons a company would still need RAG on top of it, rather than just querying the model directly?",
    "o": [
      "The model already has full access to any private company document by default; RAG's only purpose is to make responses shorter",
      "The model's training data has a cutoff (so it doesn't know recent events/updates), and it has no access to your PRIVATE, internal data (like a company's own insurance/policy documents) that were never part of its training data at all",
      "RAG exists purely to make the model's responses format as valid JSON — it has no relationship to knowledge freshness or private data access",
      "RAG's only purpose is to reduce the number of tokens the model uses per response, with no connection to what the model does or doesn't know"
    ],
    "a": [
      1
    ],
    "e": "The two textbook reasons are exactly these: a training cutoff means genuinely recent facts (this week's news, a price that changed yesterday) simply aren't in the model's weights at all, and a base model has zero access to an organization's own private/internal documents unless something explicitly supplies them — RAG's entire mechanism (retrieving external context and feeding it into the prompt) directly addresses both gaps. RAG has no special relationship to output FORMAT (that's what structured output/JSON mode handles, a separate concern) — conflating the two misattributes what RAG solves. The model does NOT have automatic access to private documents by default — that's precisely the gap RAG fills, not something already true beforehand. And while retrieval can indirectly affect prompt length, 'reducing token count' isn't RAG's purpose — sometimes RAG-augmented prompts are actually LONGER than a bare question, since they include retrieved context."
  },
  {
    "d": "Diagnosing RAG-Specific Hallucination",
    "q": "A RAG system is retrieving documents but still hallucinating. Named causes include poor retrieval quality, irrelevant chunks reaching the prompt, weak/underspecified prompts, and missing context. What do these four causes have in common that makes them specifically RAG PIPELINE problems, not just generic LLM issues?",
    "o": [
      "These causes only ever occur if the underlying LLM itself was poorly trained, and have nothing to do with anything the RAG pipeline itself does",
      "These four causes are actually unrelated to RAG specifically and would occur identically even in a plain LLM call with no retrieval step involved at all",
      "All four causes are solved by exactly the same single fix (using a bigger LLM), regardless of which pipeline stage is actually at fault",
      "Each one describes a failure at a SPECIFIC STAGE of the retrieval-to-generation pipeline (the retrieval step itself, the chunk selection, the prompt assembly, the context completeness) — meaning the fix is pipeline engineering at that stage, not just \"the model is hallucinating\" in the abstract"
    ],
    "a": [
      3
    ],
    "e": "What ties these four together is that each maps to a distinct, fixable STAGE of the RAG pipeline specifically: poor retrieval is a retrieval-quality problem, irrelevant chunks reaching the prompt is a chunk-selection/ranking problem, weak prompts is a prompt-assembly problem, and missing context is a completeness-of-retrieved-material problem — which is exactly why diagnosing WHICH stage failed matters more than just labeling the output 'a hallucination.' These are specifically RAG-PIPELINE failure modes precisely because they wouldn't exist without a retrieval step to go wrong in the first place (a plain LLM call has no retrieval stage to fail at) — so claiming they're unrelated to RAG gets the causal relationship backwards. A single 'use a bigger model' fix doesn't address any of these — a bigger model fed the same irrelevant chunks or the same weak prompt will still likely hallucinate, because the problem is upstream of the model's own capability. And these are pipeline-engineering issues, not a claim about the base LLM's training quality — the same well-trained LLM would behave differently given better retrieval, chunking, and prompting."
  },
  {
    "d": "Enterprise-Scale RAG Design",
    "q": "Designing a RAG system for enterprise scale is described as needing more than the basic embed-then-retrieve pipeline: multi-stage retrieval, metadata filtering, caching, monitoring, and access control (e.g. multi-tenant/role-based). What's the common thread behind adding ACCESS CONTROL specifically to this list?",
    "o": [
      "At enterprise scale, different users/tenants often should NOT be able to retrieve the same documents (e.g. one customer's private data shouldn't leak into another customer's answers) — access control enforces WHO can retrieve WHAT, a concern that doesn't exist in a single-user proof-of-concept",
      "Access control exists only to reduce LLM API costs and has no actual bearing on document-level security or data isolation between users or tenants",
      "In an enterprise deployment, every single user is always given access to every document with no restriction, making access control a purely theoretical, unused feature",
      "Access control's only function is to make the vector database run faster, with no actual relationship to who can see which documents"
    ],
    "a": [
      0
    ],
    "e": "A single-user proof-of-concept has no concept of 'documents that belong to a different user/tenant' — but a real multi-tenant enterprise system needs to guarantee, structurally, that retrieval never crosses that boundary (customer A's RAG answers must never surface customer B's confidential documents), which is exactly the isolation problem role-based/multi-tenant access control is designed to solve at the retrieval layer. It has nothing to do with vector database raw SPEED — that's a separate performance concern (indexing/scaling), not what access control addresses. Claiming everyone gets access to everything with no restriction describes the OPPOSITE of why enterprise deployments need this feature — unrestricted access is precisely the risk access control exists to prevent. And it isn't a cost-reduction mechanism either — its purpose is data isolation/security, a different concern from API cost optimization (which caching and model routing address instead)."
  },
  {
    "d": "Production RAG Challenges",
    "q": "Named production challenges for a live RAG system include high latency, vector DB scaling as data grows, security (e.g. prompt injection), cost, and DATA FRESHNESS. Why is data freshness specifically described as a tradeoff, not just \"refresh as often as possible\"?",
    "o": [
      "Refreshing embeddings on every possible change is compute-/cost-intensive, so the actual right cadence depends on how fast the underlying source data genuinely changes for THAT application — refreshing more than the data's real rate of change wastes resources without improving answer quality",
      "Data freshness only matters for RAG systems that don't use a vector database at all, and is irrelevant to any system that does use one",
      "There is no real tradeoff at all — data should always be refreshed as frequently as technically possible with zero downside to doing so at maximum frequency",
      "Refreshing the vector database more frequently always makes answers LESS accurate, so the correct strategy is to never refresh embeddings after the initial build"
    ],
    "a": [
      0
    ],
    "e": "The stated reasoning is explicit: refreshing on every possible change is CPU-intensive, so you have to understand your specific application's actual pace of change to pick a sane cadence — refreshing a slowly-changing knowledge base every hour wastes compute for no accuracy benefit, while refreshing a fast-changing one too rarely serves stale answers, which is exactly why it's framed as a genuine cost-vs-freshness tradeoff rather than a free 'more is always better' dial. Claiming zero downside to maximum-frequency refreshing ignores the explicitly stated compute/cost cost of doing so. Claiming more frequent refresh makes answers WORSE has the relationship backwards — stale data (too INFREQUENT refresh) is what causes outdated, wrong answers, not frequent refresh. And data freshness is a property of whatever knowledge store backs retrieval — it applies directly to vector-database-backed RAG (the exact case being discussed), not something that only matters in some alternate architecture without one."
  },
  {
    "d": "RAG Pipeline Order",
    "q": "Which of these correctly orders the standard RAG pipeline stages, from raw source documents to a final generated answer?",
    "o": [
      "Chunk the documents → generate the final answer directly from the raw chunks with no embedding or vector similarity search step involved at all",
      "Ingest documents → chunk them → generate embeddings → store in a vector database → embed the user's query → retrieve nearest matches → augment the prompt with retrieved context → generate the answer",
      "Generate the answer first → then retrieve supporting documents afterward to justify whatever was already generated → then chunk and embed those documents as a final step",
      "Embed the user's query → generate the final answer directly from that embedding alone, with no document retrieval or prompt augmentation step involved at all"
    ],
    "a": [
      1
    ],
    "e": "This is the standard pipeline order because each stage depends on the one before it: you can't chunk before ingesting, can't embed before chunking (embeddings are computed per-chunk), can't retrieve before both the corpus AND the query are embedded into the same vector space, and can't meaningfully augment a prompt before you've actually retrieved something to put into it — generation has to come last since it consumes everything the earlier stages produced. Generating the answer FIRST and retrieving justification afterward (option 2) inverts the entire causal structure of RAG — the retrieved context is supposed to INFORM the answer, not be fetched after the fact to rationalize one already produced. Skipping retrieval entirely after embedding the query (option 3) removes the actual 'retrieval' from retrieval-augmented generation — that's just an embedding computation with no augmentation happening at all. And skipping the embedding/vector-search step entirely (option 4) removes the mechanism that finds RELEVANT chunks in the first place — without it you'd have no principled way to select which chunks are actually relevant to the query."
  },
  {
    "d": "Chunking & Embedding Quality",
    "q": "The video lists \"low quality embeddings\" as a cause of RAG hallucination, tied specifically to how chunking was done. What's the actual mechanism connecting POOR chunking decisions to bad retrieval quality?",
    "o": [
      "Chunk size and boundaries have literally zero effect on embedding quality — embeddings are computed identically regardless of how text was chunked beforehand",
      "Chunking quality only matters for keyword-based traditional search and has no bearing whatsoever on semantic/embedding-based retrieval",
      "A chunk that's poorly sized or poorly bounded can end up mixing unrelated content together (or splitting one coherent idea across two separate chunks), so its embedding represents a muddled mix of concepts rather than one clean, retrievable idea — making it less likely to be correctly matched against a genuinely relevant query",
      "Poor chunking can only ever affect how FAST retrieval runs, with no possible effect on which documents actually get retrieved or how relevant they are"
    ],
    "a": [
      2
    ],
    "e": "An embedding is a numerical summary of WHATEVER text is inside that chunk — if a chunk poorly mixes two unrelated topics (bad chunk boundaries) or arbitrarily cuts a single coherent explanation in half, the resulting embedding is a muddled average that doesn't cleanly represent either concept, making it a worse match for a query that's actually about just one of them. Chunking absolutely affects the CONTENT going into the embedding computation, so claiming it has zero effect ignores that embeddings are computed directly FROM the chunked text. It's not merely a speed concern either — it directly affects retrieval RELEVANCE (which documents get matched and how well), not just how quickly retrieval executes. And chunking quality matters for BOTH keyword and semantic retrieval — semantic/embedding-based search is, if anything, MORE sensitive to poor chunking, since the entire embedding is derived from exactly the text within chunk boundaries."
  },
  {
    "d": "Cost & Latency at Scale",
    "q": "For a high-traffic production RAG system, \"caching\" and \"parallel retrieval\" are both named as infrastructure-level performance levers. What specifically does CACHING address here that parallel retrieval does not?",
    "o": [
      "Parallel retrieval is only usable when caching is completely disabled — the two techniques are mutually exclusive and cannot be combined in the same system",
      "Caching only works for write operations (updating the vector database), and has no relationship to read-side query performance whatsoever",
      "Caching avoids redoing the SAME (or highly similar) retrieval+generation work for repeated/similar queries by reusing a stored result, whereas parallel retrieval instead speeds up a SINGLE query's own multiple independent lookups by running them concurrently",
      "Caching and parallel retrieval are two different names for the exact identical optimization, with no distinction in what problem each one solves"
    ],
    "a": [
      2
    ],
    "e": "These two levers target different sources of wasted work: caching specifically avoids RE-doing retrieval and generation for queries that are the same or similar to ones already answered (why pay the full pipeline cost twice for near-identical questions?), while parallel retrieval speeds up the independent LOOKUPS within a SINGLE query that don't depend on each other's results (running several searches concurrently instead of one-by-one). They solve genuinely different bottlenecks, so treating them as the same optimization misses why both are separately worth having. Caching is fundamentally a READ-side optimization for repeated query patterns, not a write-side mechanism for updating the vector store — that's a different concern (data freshness, covered separately). And the two are explicitly complementary, not mutually exclusive — a well-optimized production system typically uses both caching (for repeat queries) AND parallel retrieval (for each individual query's own independent lookups) at the same time."
  },
  {
    "d": "Metadata Filtering",
    "q": "Enterprise RAG design mentions applying \"metadata filtration\" before a document reaches the LLM as context. How does metadata filtering differ from the vector-similarity search that finds semantically relevant chunks?",
    "o": [
      "Metadata filtering and vector-similarity search are exactly the same operation, just using different underlying code paths with an identical effect on the result set",
      "Metadata filtering makes vector-similarity search completely unnecessary and fully replaces it in any production RAG pipeline",
      "Metadata filtering can only be applied AFTER the LLM has already generated its final answer, making it a post-hoc correction rather than a retrieval-time filter",
      "Metadata filtering narrows the candidate set using STRUCTURED fields (e.g. date, department, document type, access level) BEFORE or alongside similarity ranking — cutting out documents that are semantically similar-sounding but structurally wrong (e.g. the right topic but the wrong date range or wrong department) in a way pure similarity search can't detect on its own"
    ],
    "a": [
      3
    ],
    "e": "Vector similarity search finds chunks whose MEANING resembles the query — but it has no inherent concept of structured constraints like 'only from Q3 2025' or 'only HR-department documents' unless that information happens to be embedded in the text itself. Metadata filtering applies those structured constraints directly (as a database-style filter), narrowing the candidate pool by facts a pure embedding comparison can't reliably enforce — catching cases where a chunk is semantically ON-topic but structurally WRONG (right subject, wrong date/department/permission level). The two are complementary mechanisms operating on different signal types (structured fields vs. semantic meaning), not identical operations. Metadata filtering happens at RETRIEVAL time, narrowing what even gets considered for similarity ranking or is applied alongside it — not as a correction applied after the LLM has already produced an answer. And it doesn't replace similarity search entirely — the two are typically combined, with metadata filtering narrowing the pool and similarity search then ranking within it."
  }
]
</script>
<div class="topic-quiz-mount"></div>
