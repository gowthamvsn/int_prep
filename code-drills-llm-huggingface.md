# Code Drills — Tier 3: LLMs — Tokenization, HuggingFace, Decoding, Embeddings

Continues `code-drills-deep-learning.md`. Where that file ended at "here's how a neural net trains," this one starts at "here's how an LLM actually turns text into numbers, predicts the next token, and turns numbers back into text" — the mechanical layer underneath everything in `llm-landscape.md`, `nca-genl`'s transformer teardown, and `core-technical-depth.md`. `module-cheatsheet.md`'s LLM section has the same calls as a flat lookup. Verified in a dedicated `.venv-llm-rag` (transformers 5.14.1, sentence-transformers 5.6.1, torch 2.13.0+cpu) against real downloaded models (gpt2, distilgpt2, all-MiniLM-L6-v2) — not mocked.

---

## Cluster 1 — Tokenization

> 🔗 **Theory:** [LLM Landscape](/topic/llm-landscape) — the model map behind whichever tokenizer/model you load below

**1. Load a tokenizer for a specific model.**
```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
```

**2. Tokenize text into the integer ids a model actually consumes.**
```python
text = "Machine learning is fun"
ids = tokenizer(text)["input_ids"]     # e.g. [37573, 4673, 318, 1257] — 4 tokens, not 4 characters or 4 words
```

**3. Decode ids back into readable text.**
```python
tokenizer.decode(ids)    # 'Machine learning is fun' — round-trips exactly for a well-formed sequence
```

**4. See how tokenization is subword, not whole-word — the reason vocab size can stay small.**
```python
tokenizer.tokenize("unhappiness")     # ['un', 'h', 'appiness'] — split into pieces the tokenizer actually
                                        # learned from training data frequency, NOT into "real" morphemes
                                        # (un/happy/ness) — BPE optimizes for compression, not linguistics
tokenizer.tokenize("the")              # ['the'] — a common word stays as one token
# this is exactly why a ~50k-token vocabulary can represent effectively unlimited words
```

**5. Know the special tokens a tokenizer reserves.**
```python
tokenizer.bos_token, tokenizer.eos_token, tokenizer.pad_token   # ('<|endoftext|>', '<|endoftext|>', None)
# gpt2 reuses the SAME token for both start and end of text, and defines NO pad token at all —
# batching breaks without one, so the standard workaround is:
tokenizer.pad_token = tokenizer.eos_token
```

**6. Batch-tokenize multiple sequences of different lengths, padded to the same width.**
```python
batch = tokenizer(["hi", "machine learning is fun"], padding=True, return_tensors="pt")
batch["input_ids"].shape    # (2, 4) — "hi" (1 real token) is padded out to match the 4-token sequence
```

**7. Read the attention mask that comes with a padded batch.**
```python
batch["attention_mask"]
# tensor([[1, 0, 0, 0],     <- "hi": 1 real token, then 3 padding slots (gpt2 pads on the RIGHT by default)
#         [1, 1, 1, 1]])     <- "machine learning is fun": all 4 real, no padding needed
# 1 = "attend to this token", 0 = "ignore, this is just padding" — without this mask, the model would
# treat padding as real content and get confused by it. Check tokenizer.padding_side if unsure which
# side a given model pads on — it varies by model family (decoder-only models often pad LEFT for generation).
```

**8. Truncate sequences that exceed a model's context window.**
```python
tokenizer("a very long document " * 1000, truncation=True, max_length=512)["input_ids"]
# cuts to exactly 512 tokens instead of raising an error or silently overflowing the model's limit
```

**9. Count tokens without running the model — the real-world use case is cost/context-budget estimation.**
```python
n_tokens = len(tokenizer.encode("How much does this prompt cost in tokens?"))
# API pricing (OpenAI, Anthropic, etc.) bills by token count, not character or word count — this is
# literally how you'd estimate a prompt's cost before sending it
```

**10. Apply a chat template — turn a list of role/content messages into the exact prompt string a chat model expects.**
```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"},
]
# plain gpt2 was never trained on a chat format, so its tokenizer ships with NO chat_template and
# apply_chat_template() raises ValueError on it — every actual chat/instruct model (Llama-Instruct,
# Qwen-Chat, etc.) DOES define one, in its own format (ChatML, Llama's [INST]...[/INST], etc.):
tokenizer.chat_template = (
    "{% for message in messages %}{{ message['role'] + ': ' + message['content'] + '\n' }}{% endfor %}"
    "{% if add_generation_prompt %}{{ 'assistant: ' }}{% endif %}"
)     # a minimal template, just to show the mechanism — real models' templates are pre-defined for you
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
# 'system: You are a helpful assistant.\nuser: What is 2+2?\nassistant: '
# the whole point: you never hand-format role tags yourself, and swapping models swaps the format for free
```

---

## Cluster 2 — Loading & Running a Model

**11. Load a causal (autoregressive, text-generating) language model.**
```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("gpt2")
```

**12. Run a forward pass and inspect the raw output shape.**
```python
import torch
inputs = tokenizer("The capital of France is", return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
outputs.logits.shape    # (1, seq_len, vocab_size) — one score PER VOCAB TOKEN, at EVERY position
```

**13. Turn the last position's logits into next-token probabilities.**
```python
import torch.nn.functional as F
next_token_logits = outputs.logits[0, -1, :]      # scores for what comes after the LAST input token
probs = F.softmax(next_token_logits, dim=-1)
top_id = probs.argmax()
tokenizer.decode(top_id)    # the single most likely next token — this IS the core LLM operation, repeated
```

**14. Generate text the simplest way — greedy decoding (always pick the top token).**
```python
inputs = tokenizer("The capital of France is", return_tensors="pt")
output_ids = model.generate(**inputs, max_new_tokens=10, do_sample=False)
tokenizer.decode(output_ids[0], skip_special_tokens=True)
```

**15. Generate with sampling instead — introduces randomness for more varied output.**
```python
output_ids = model.generate(**inputs, max_new_tokens=20, do_sample=True, temperature=0.8, top_p=0.9)
tokenizer.decode(output_ids[0], skip_special_tokens=True)
# do_sample=True is required — temperature/top_p/top_k are silently ignored under greedy decoding
```

**16. Generate with beam search — explore several candidate continuations, keep the best.**
```python
output_ids = model.generate(**inputs, max_new_tokens=10, num_beams=5, early_stopping=True)
# tracks the 5 highest-probability SEQUENCES (not just next tokens) in parallel — costs ~5x the compute
# of greedy, generally produces more globally coherent (but less creative/varied) text
```

**17. Fix the "pad token not set" warning that comes up constantly with gpt2-family models.**
```python
model.generate(**inputs, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id)
# without this, batched generation with variable-length outputs has no defined padding behavior
```

**18. Move a model to a lower-memory dtype (matters far more on real GPU-hosted models than gpt2).**
```python
model_fp16 = AutoModelForCausalLM.from_pretrained("gpt2", dtype=torch.float16)
# halves memory footprint vs. the default float32 — the first lever before reaching for quantization.
# (older code/tutorials use the `torch_dtype=` kwarg name for this same argument — still works, just deprecated)
```

**19. Count a model's total and trainable parameters — same drill as MLPs, just bigger numbers.**
```python
total_params = sum(p.numel() for p in model.parameters())
total_params    # gpt2 (small): 124,439,808 — matches the "124M" in "GPT-2 small"
```

**20. Skip steps 11-14 entirely with the one-line `pipeline()` shortcut.**
```python
from transformers import pipeline
generator = pipeline("text-generation", model="gpt2")
generator("Once upon a time", max_new_tokens=15, do_sample=False)
# pipeline() bundles tokenizer + model + generate() + decode() into one call — great for quick iteration,
# drills #1-19 are what it's doing internally, useful to know once you need finer control
```

---

## Cluster 3 — Decoding Strategies, Compared Directly

**21. See greedy vs. sampled output diverge on the exact same prompt.**
```python
inputs = tokenizer("My favorite food is", return_tensors="pt")
greedy = model.generate(**inputs, max_new_tokens=15, do_sample=False)
sampled = model.generate(**inputs, max_new_tokens=15, do_sample=True, temperature=1.0)
# greedy is DETERMINISTIC — same prompt always gives the same output; sampled varies run to run
```

**22. Tune temperature — the "how random" knob.**
```python
model.generate(**inputs, max_new_tokens=15, do_sample=True, temperature=0.2)   # near-greedy, safe, repetitive
model.generate(**inputs, max_new_tokens=15, do_sample=True, temperature=1.5)   # wild, more incoherent
# temperature divides the logits before softmax: low temp sharpens the distribution toward the top
# choice, high temp flattens it toward uniform — 0.7-1.0 is the common practical range
```

**23. Restrict sampling to only the top-k most likely tokens.**
```python
model.generate(**inputs, max_new_tokens=15, do_sample=True, top_k=50)
# only the 50 highest-probability tokens are even considered at each step — cuts off the long, low-quality tail
```

**24. Restrict sampling with nucleus (top-p) filtering instead — an adaptive alternative to top-k.**
```python
model.generate(**inputs, max_new_tokens=15, do_sample=True, top_p=0.9)
# keeps the SMALLEST set of tokens whose cumulative probability reaches 0.9 — adapts per step,
# unlike top_k's fixed count (a confident distribution might keep just 3 tokens; a flat one might keep 200)
```

**25. Reduce repetitive loops — a known failure mode of small/undertrained LMs.**
```python
model.generate(**inputs, max_new_tokens=30, do_sample=False, repetition_penalty=1.3, no_repeat_ngram_size=2)
# repetition_penalty: down-weights tokens already generated | no_repeat_ngram_size=2: hard-bans repeating
# any 2-token sequence that's already appeared
```

**26. Know the difference between `max_new_tokens` and `max_length`.**
```python
model.generate(**inputs, max_new_tokens=20)   # 20 tokens ADDED on top of the input prompt length
model.generate(**inputs, max_length=20)         # 20 tokens TOTAL, including the input prompt — easy to mix up
```

**27. Understand what actually stops generation.**
```python
# generation stops at whichever comes FIRST: max_new_tokens/max_length reached, OR the model emits
# its eos_token_id. A model that never learned to emit eos (or has it suppressed) will just keep going
# until the length limit — a real, commonly-hit failure mode worth recognizing, not just a theoretical one
```

**28. Generate for a whole batch of prompts in one call, not a Python loop.**
```python
prompts = ["The weather today is", "My favorite hobby is"]
inputs = tokenizer(prompts, return_tensors="pt", padding=True)
outputs = model.generate(**inputs, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id)
[tokenizer.decode(o, skip_special_tokens=True) for o in outputs]   # one output string per input prompt
```

---

## Cluster 4 — Embeddings

**29. Load a sentence-embedding model (different job than a generative LM — turns text into a fixed vector).**
```python
from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer("all-MiniLM-L6-v2")
```

**30. Encode text into a vector.**
```python
vec = embedder.encode("Machine learning is fun")
vec.shape    # (384,) — a fixed-size vector regardless of input length, unlike token-level LM output
```

**31. Compute cosine similarity between two embeddings — the core operation behind semantic search.**
```python
import numpy as np
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

v1 = embedder.encode("I love machine learning")
v2 = embedder.encode("I enjoy studying AI")
v3 = embedder.encode("The weather is sunny today")
cosine_sim(v1, v2)    # 0.54 — clearly related, despite sharing almost no words in common
cosine_sim(v1, v3)    # -0.01 — essentially unrelated, near zero
```

**32. Build a tiny semantic search over a small corpus.**
```python
corpus = ["The cat sat on the mat", "Dogs are loyal pets", "Machine learning uses data to learn patterns"]
corpus_vecs = embedder.encode(corpus)
query_vec = embedder.encode("What is AI?")
sims = [cosine_sim(query_vec, cv) for cv in corpus_vecs]
best_match = corpus[int(np.argmax(sims))]    # picks the ML sentence — nearest in MEANING, not shared keywords
```

**33. Encode many sentences efficiently in one batched call, not a loop.**
```python
sentences = ["sentence one", "sentence two", "sentence three"]
vecs = embedder.encode(sentences, batch_size=32, show_progress_bar=False)
vecs.shape    # (3, 384) — one row per sentence, computed together for GPU/CPU efficiency
```

**34. Know why embedding dimensionality is a fixed model property, not something you choose per call.**
```python
embedder.get_embedding_dimension()    # 384 for this model — baked in by how it was trained
# (older sentence-transformers versions call this get_sentence_embedding_dimension — same thing, renamed)
# a different embedding model (e.g. a 1536-dim OpenAI embedding) is NOT comparable/mixable with this one —
# always embed queries and corpus with the SAME model
```

**35. Normalize embeddings so a plain dot product IS cosine similarity — the trick vector DBs rely on.**
```python
v1n = v1 / np.linalg.norm(v1)
v2n = v2 / np.linalg.norm(v2)
np.dot(v1n, v2n)    # identical to cosine_sim(v1, v2) — because normalization removes the magnitude term
# most vector databases (FAISS, etc.) offer a normalized inner-product index specifically for this reason
```

**36. See semantic search catch what keyword matching misses.**
```python
doc = "The feline rested on the rug"     # zero words in common with the query below
query = "cat sitting on a carpet"
cosine_sim(embedder.encode(doc), embedder.encode(query))    # still high — meaning matches, vocabulary doesn't
# this exact gap is why RAG pipelines use embeddings instead of a keyword/regex search over documents
```

---

**Next in the Code Drills tier:** `code-drills-rag-langchain.md` (chunking, vector stores, and a full RAG pipeline built from these embedding primitives).
