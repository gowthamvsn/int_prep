# Code Drills — Tier 3: RAG & LangChain — Chunking to a Full Grounded-Answer Pipeline

Continues `code-drills-llm-huggingface.md`, using its embedding drills as the building block for real retrieval. Terser, quiz-style companion to `langchain-practice.md`'s narrative chains (which build the same RAG pipeline as one continuous worked example) and `core-technical-depth.md`'s RAG-internals section; `rag-deeper.md` picks up past this file into hybrid search, re-ranking, and GraphRAG. Verified in `.venv-llm-rag` (langchain 1.3.14, langchain-openai 1.4.1, faiss 1.14.3, sentence-transformers 5.6.1) — Clusters 1-2 use real downloaded models (all-MiniLM-L6-v2) and a real FAISS index; Clusters 3-4 make real calls to this project's Azure `gpt-4.1-mini` deployment, the same one `server.py`/`langchain-practice.md` already use. `.env` has no embeddings deployment configured, which is exactly why these drills embed locally instead of via Azure — a real, common setup, not a workaround invented for this file.

---

## Cluster 1 — Chunking

**1. Split a long document into overlapping chunks.**
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text = open("numpy-practice.md", encoding="utf-8").read()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_text(text)
len(chunks), len(chunks[0])    # many chunks, each <= 500 characters
```

**2. Understand why `chunk_overlap` exists — what breaks without it.**
```python
# a hard cut at exactly chunk_size can slice a sentence, or a code block, in half BETWEEN two chunks.
# a question whose answer straddles that cut then has NEITHER chunk score highly enough to be retrieved —
# chunk_overlap duplicates a slice of text at each boundary so the full concept survives in at least one chunk
splitter_no_overlap = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
splitter_with_overlap = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# with_overlap produces MORE chunks (some content duplicated across the 50-char overlap zone) — the
# storage/retrieval cost tradeoff you're paying for not losing straddled answers
```

**3. See why `RecursiveCharacterTextSplitter` tries multiple separators, not just a raw character count.**
```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200, chunk_overlap=20,
    separators=["\n\n", "\n", ". ", " ", ""],   # tries these IN ORDER, falls back only if a split still too big
)
# "recursive" here means: try splitting on paragraph breaks first; if a resulting piece is still over
# chunk_size, split THAT on single newlines; then sentences; then words; only raw characters as a last resort.
# this is why it tends to keep whole paragraphs/sentences together far better than a naive text[i:i+500] loop.
```

**4. Split pre-structured `Document` objects (carries metadata like source filename) instead of raw strings.**
```python
from langchain_core.documents import Document

docs = [Document(page_content=text, metadata={"source": "numpy-practice.md"})]
chunks = splitter.split_documents(docs)
chunks[0].metadata    # {'source': 'numpy-practice.md'} — every chunk inherits the parent doc's metadata
# metadata survives all the way to retrieval — it's how a RAG answer can cite WHICH file a fact came from
```

**5. Choose a chunk size with the tradeoff explicit, not just picking a number.**
```python
# small chunks (e.g. 200 chars): more precise retrieval (less irrelevant text per chunk), but a concept
#   that needs several sentences of context to make sense can get retrieved without that context
# large chunks (e.g. 2000 chars): more self-contained context per chunk, but dilutes the embedding —
#   a chunk covering 5 different topics embeds as a blurry average of all 5, hurting retrieval precision
# 300-800 characters (or ~100-300 tokens) is a common practical starting range — tune against real queries
```

---

## Cluster 2 — Embeddings & a Real Vector Store

> 🔗 **Theory:** [Core Technical Depth — RAG and Vector Databases](/topic/core-technical#rag-and-vector-databases)

**6. Embed a batch of chunks with a real local model (no API cost, no rate limit).**
```python
from sentence_transformers import SentenceTransformer
import numpy as np

embedder = SentenceTransformer("all-MiniLM-L6-v2")
chunk_vecs = embedder.encode(chunks[:5] if isinstance(chunks[0], str) else [c.page_content for c in chunks[:5]])
chunk_vecs.shape    # (5, 384) — one 384-dim vector per chunk
```

**7. Build a real FAISS index and search it.**
```python
import faiss

dim = chunk_vecs.shape[1]
index = faiss.IndexFlatL2(dim)          # exact search, L2 (Euclidean) distance — the simplest FAISS index
index.add(chunk_vecs.astype("float32"))   # FAISS requires float32, not numpy's default float64

query_vec = embedder.encode(["how do you reshape an array"]).astype("float32")
distances, indices = index.search(query_vec, k=3)     # k=3 nearest neighbors
indices[0]    # array of the 3 closest chunk indices, ranked nearest first
```

**8. Use cosine similarity (via normalized inner product) instead of raw L2 distance — often preferred for text.**
```python
faiss.normalize_L2(chunk_vecs)            # normalizes IN PLACE — vectors become unit length
index_cos = faiss.IndexFlatIP(dim)         # IP = inner product; on NORMALIZED vectors, this IS cosine similarity
index_cos.add(chunk_vecs.astype("float32"))
query_vec_norm = embedder.encode(["how do you reshape an array"]).astype("float32")
faiss.normalize_L2(query_vec_norm)
sims, indices = index_cos.search(query_vec_norm, k=3)   # sims are now cosine similarities, higher = closer
```

**9. Save and reload a FAISS index — avoid re-embedding the whole corpus every run.**
```python
faiss.write_index(index, "chunks.index")
loaded_index = faiss.read_index("chunks.index")
```

**10. Wrap raw retrieval into a reusable function — the shape every retriever underneath a framework has.**
```python
def retrieve(query, index, embedder, chunks, k=3):
    q_vec = embedder.encode([query]).astype("float32")
    distances, idx = index.search(q_vec, k)
    return [chunks[i] for i in idx[0]]

results = retrieve("how do you reshape an array without copying", index, embedder,
                    [c.page_content if hasattr(c, "page_content") else c for c in chunks])
```

---

## Cluster 3 — Wiring Retrieval Into LangChain

> 🔗 **Theory:** [LangChain Practice — Building RAG](/topic/practice-langchain#cluster-4-building-rag-from-raw-text-to-a-grounded-answer)

**11. Wrap the local embedder in LangChain's `Embeddings` interface, so it plugs into LangChain's vector stores.**
```python
from langchain_core.embeddings import Embeddings

class LocalEmbeddings(Embeddings):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):            # LangChain calls this for a BATCH of documents
        return self.model.encode(texts).tolist()   # LangChain expects plain lists, not numpy arrays

    def embed_query(self, text):                  # LangChain calls this for a SINGLE query string
        return self.model.encode([text])[0].tolist()

embeddings = LocalEmbeddings()
# this is the entire contract LangChain needs from ANY embedding model — implement these two methods
# and AzureOpenAIEmbeddings, HuggingFaceEmbeddings, and this local wrapper are all interchangeable
```

**12. Build a LangChain vector store from documents, using the local embeddings wrapper.**
```python
from langchain_community.vectorstores import FAISS as LC_FAISS
# langchain_community is being sunset in favor of standalone integration packages (e.g. a dedicated
# langchain-faiss package) — still works as of this writing, but check for a newer import path if this
# raises a deprecation-heavy warning by the time you're reading this

docs = [Document(page_content=c) for c in chunks[:20]]   # keep it small for a fast local demo
store = LC_FAISS.from_documents(docs, embeddings)
```

**13. Turn a vector store into a `Retriever` — a `Runnable`, so it composes with `|` like everything else in LangChain.**
```python
retriever = store.as_retriever(search_kwargs={"k": 3})
hits = retriever.invoke("how do you reshape an array without copying")
len(hits), hits[0].page_content[:80]
```

**14. Build the full RAG chain — retrieve, stuff context into a prompt, generate a grounded answer.**
```python
import os
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

llm = AzureChatOpenAI(
    azure_deployment="gpt-4.1-mini",
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"].strip(),
    api_key=os.environ["AZURE_OPENAI_KEY"].strip(),
    api_version="2024-06-01",
    temperature=0,
)

rag_prompt = ChatPromptTemplate.from_template(
    "Answer using ONLY the context below. If the answer isn't in it, say so.\n\n"
    "Context:\n{context}\n\nQuestion: {question}"
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
# RunnablePassthrough(): the left-hand dict needs BOTH context (via the retriever) and question
# (the original string, untouched) reaching the prompt — RunnablePassthrough is the identity function
# as a Runnable. Forgetting it is why a first attempt throws KeyError on "question".
```

**15. Ask a question the context genuinely doesn't answer — see the grounding instruction actually matter.**
```python
print(rag_chain.invoke("what is the capital of France?"))
# with a numpy-only corpus, a well-grounded chain says something like "the context doesn't contain this" —
# an UN-grounded chain (no "answer using ONLY the context" instruction) would just answer from its own
# training knowledge instead, silently ignoring your retrieved context. This is the whole point of RAG's
# prompt instruction — retrieval alone doesn't force the model to actually USE what was retrieved.
```

**16. Return the source chunks alongside the answer — for citation, not just a bare string.**
```python
from langchain_core.runnables import RunnableParallel

rag_chain_with_sources = RunnableParallel(
    answer=(
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt | llm | StrOutputParser()
    ),
    sources=retriever,
)
result = rag_chain_with_sources.invoke("how do you reshape an array without copying data?")
result["answer"]     # the generated text
result["sources"]     # the actual Document chunks used — lets a UI show "based on: numpy-practice.md"
```

---

## Cluster 4 — Where Simple RAG Breaks (and the direct fixes)

> 🔗 **Theory:** [RAG, Deeper — Walking the Assembly Line](/topic/rag-deeper#cluster-1-walking-the-assembly-line-station-by-station)

**17. See retrieval fail on a query that means the right thing but shares no vocabulary with the chunk.**
```python
# if the corpus says "gradient descent minimizes the loss function" and the query is
# "how does the model get better during training", pure keyword search (drill against
# code-drills-basics.md's string methods) would find NOTHING — zero shared words.
# Embedding-based retrieval (this whole file) is specifically the fix: it matches MEANING, not tokens —
# see code-drills-llm-huggingface.md drill #36 for the same gap demonstrated directly on embeddings.
```

**18. See retrieval fail the other way — a chunk that's lexically similar but semantically wrong.**
```python
# "Python's GIL prevents true multithreading" vs. query "does Python support multiple threads" —
# these can retrieve well since they share real conceptual overlap, but a corpus with BOTH
# "Python supports threading" (true, with caveats) and unrelated general "threading" content
# can retrieve the wrong chunk if embeddings alone can't distinguish domain-specific nuance.
# real fix at scale: hybrid search (embedding + keyword/BM25 combined) and re-ranking — rag-deeper.md.
```

**19. Know the difference between "not enough chunks retrieved" and "wrong chunks retrieved," because the fix differs.**
```python
# increase k (e.g. from 3 to 8) if the answer is scattered across more source chunks than you're pulling in
retriever_more = store.as_retriever(search_kwargs={"k": 8})
# but MORE chunks isn't free: more (and more irrelevant) context can dilute the model's attention and
# increase cost/latency — if the problem is WRONG chunks (not too few), increasing k just adds more noise
```

**20. Recognize a hallucination that happens despite retrieval actually working correctly.**
```python
# retrieval can return the RIGHT chunks and the model can still fabricate a detail that isn't literally
# in them — e.g. inventing a specific version number or exact parameter value not stated in the context.
# grounding the prompt ("answer using ONLY the context") reduces this but does not eliminate it —
# this is exactly why RAG evaluation frameworks (RAGAS, ARES — rag-deeper.md) score "faithfulness"
# (is the answer actually supported by the retrieved context) as a SEPARATE metric from "relevance"
# (did retrieval find the right chunks) — a system can succeed at one and fail at the other independently.
```

---

**Next in the Code Drills tier:** `code-drills-finetuning-peft.md` (LoRA/QLoRA, PEFT, and the Trainer API — how a base model like the one in `code-drills-llm-huggingface.md` actually gets adapted to a new task).
