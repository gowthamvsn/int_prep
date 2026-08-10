# LLM Landscape — Open Source vs. Closed Source Models

A reference, not a tutorial: what's actually out there, who trained it, whether you can download the weights, and what it's actually for. 25 models across closed-source, open-weight/open-source, and domain-specific encoder models — the kind of breadth an interviewer expects you to have opinions about, not just recognize by name.

**Three categories, not two** — the "open vs. closed" framing hides a real middle category that matters:
- **Closed-source**: weights never released; access only via a paid API (GPT-4, Claude, Gemini).
- **Open-weight**: the trained weights are downloadable and runnable locally, but the license may restrict commercial use, and the training data/code usually isn't released (Llama, most of the models below). This is NOT the same as "open source" in the traditional software sense — you get the finished artifact, not the recipe.
- **Fully open**: weights, training code, and often the training data itself are all released under a permissive license (BLOOM, most BERT-family models, StarCoder). Genuinely reproducible.

**Visual + memory hook — before memorizing 25 names, memorize the 3 questions that actually pick one:**
```
                    Can data leave your infrastructure?
                    /                              \
                  NO                               YES
                   │                                 │
         Open-weight, self-hosted          Is it worth paying per-token
         (Llama/Mistral/Mixtral)           for less ops burden?
                   │                          /            \
        Need it to fit on modest GPU?       YES             NO
              /        \                     │               │
           YES          NO              Closed API      Open-weight anyway,
     smaller open-      bigger open-    (GPT-4/Claude/    just self-hosted
     weight model       weight model    Gemini)           (same left branch)
     (Mistral 7B)       (Llama 3 70B,
                         Mixtral)
```
**Remember it as "data residency first, budget-vs-ops second"** — almost every real "which LLM" decision collapses to those two questions before any benchmark score matters: if the data legally/contractually can't leave your infrastructure, the entire closed-API column is eliminated regardless of quality, and only then does "pay per token for zero ops" vs. "self-host for control and unit economics at volume" become the deciding question. Naming the right question an interviewer is testing beats naming the right model — the specific model names below go stale within a year; this decision shape doesn't.

---

> 🔗 **Hands-on reps:** [Code Drills 7 — LLMs: Tokenization, HuggingFace, Decoding, Embeddings](/topic/code-drills-llm-huggingface)

## Closed-source / proprietary (API-only)

| Model | Organization | Purpose / notes |
|---|---|---|
| **GPT-4 / GPT-4o** | OpenAI | General-purpose multimodal (text/image/audio) flagship; most widely deployed via ChatGPT and the OpenAI API; strong reasoning and coding. |
| **Claude (3/4 family)** | Anthropic | General-purpose, built around Constitutional AI alignment; very large context windows; strong at long-document reasoning, coding, and agentic tool use. |
| **Gemini (1.5/2.0/2.5)** | Google DeepMind | Natively multimodal from the ground up (text/image/audio/video in one model); deep integration with Google Search/Workspace; very long context windows. |
| **Bard** | Google | Not a current model — the deprecated *product name* for Google's chatbot (originally LaMDA-based, later PaLM-2-based). Rebranded to **Gemini** in Feb 2024; calling the current product "Bard" is a dated/incorrect usage worth knowing, not repeating. |
| **Grok** | xAI | General-purpose with real-time access to X (Twitter) data, positioned around current-events awareness and a less-filtered response style. |
| **Command R / Command R+** | Cohere | Enterprise-focused, purpose-built for RAG and tool-calling workflows rather than general chat — optimized retrieval-augmented generation performance. |
| **Amazon Titan** | AWS | Foundation models (text generation + embeddings) offered through Bedrock, built for enterprise pipelines already inside AWS. |
| **Mistral Large / Medium** | Mistral AI | Mistral's flagship *closed* API models — distinct from the same company's open-weight smaller models below; competitive reasoning at a lower price point than GPT-4-class models. |

## Open-weight / open-source, general-purpose

| Model | Organization | Purpose / notes |
|---|---|---|
| **Llama 2 / Llama 3** | Meta | Open-weight (custom license, not fully OSI-approved), the most widely used base for fine-tuning and research; strong performance-per-parameter, huge downstream ecosystem. |
| **Mistral 7B / Mixtral 8x7B** | Mistral AI | Apache 2.0, fully permissive. Mixtral is a sparse **mixture-of-experts** — 47B total parameters but only ~13B active per token, so it runs at roughly 13B-model inference cost with much higher quality. |
| **Falcon** | Technology Innovation Institute (UAE) | Open-weight, trained on the large curated **RefinedWeb** web-text dataset; competitive with Llama at its release. |
| **BLOOM** | BigScience (Hugging Face-led collaboration) | Fully open (weights + training details), explicitly built for **multilingual** coverage — 46 natural languages and 13 programming languages — and research transparency over raw benchmark performance. |
| **GPT-J / GPT-NeoX** | EleutherAI | Open GPT-3-architecture replications that predate Llama; enabled open LLM research before any major lab released comparable open weights. |
| **Gemma** | Google DeepMind | Open-weight lightweight models distilled from the same research as Gemini, sized to run on consumer hardware. |
| **Phi (Phi-2 / Phi-3)** | Microsoft | Small open-weight models trained on curated "textbook-quality" synthetic + filtered data — punches well above its parameter count on reasoning benchmarks. |
| **Qwen** | Alibaba | Open-weight, notably strong multilingual performance (especially Chinese) and coding ability. |
| **DeepSeek (V2/V3/R1)** | DeepSeek AI | Open-weight, notable for strong reasoning/coding performance at a fraction of the usual training cost, using a large sparse MoE architecture; the R1 line specifically popularized transparent chain-of-thought reasoning traces. |
| **Vicuna** | LMSYS (UC Berkeley et al.) | Open — Llama fine-tuned on user-shared ChatGPT conversations; one of the first open models to approach GPT-3.5-level chat quality, and became an early standard chat-model benchmark. |
| **Alpaca** | Stanford | Open — Llama instruction-tuned on ~52K self-generated instruction examples; the proof-of-concept that cheap instruction-tuning (not just scale) unlocks a lot of chat capability. |
| **StableLM** | Stability AI | Open-weight general-purpose models from the company behind Stable Diffusion. |

## Encoder-only & domain-specific models

| Model | Organization | Purpose / notes |
|---|---|---|
| **BERT** | Google | Open. **Bidirectional encoder-only** — not generative, not typically called an "LLM" in the modern decoder-only sense. Foundational for classification/NER/QA via fine-tuning, not text generation. |
| **RoBERTa** | Meta | Open — BERT's pretraining recipe run longer, on more data, with tuned hyperparameters (no next-sentence-prediction task); consistently outperforms BERT on the same downstream tasks. |
| **DistilBERT** | Hugging Face | Open — a distilled/compressed BERT: ~40% smaller, ~60% faster, retains ~97% of BERT's performance. The standard choice when latency/cost matters more than the last few points of accuracy. |
| **T5** | Google | Open — frames *every* NLP task (classification, translation, summarization, QA) as text-to-text, unifying pretraining and fine-tuning under one objective. |
| **BioBERT** | Korea University / Clova AI | Open — BERT **continued-pretrained** on PubMed abstracts + PMC full-text articles; for biomedical NER, relation extraction, and QA. |
| **PubMedBERT** | Microsoft | Open — pretrained **from scratch** on PubMed text (not continued from general-domain BERT); shown to outperform continued-pretraining approaches like BioBERT on biomedical benchmarks specifically because it never wastes capacity on general-domain vocabulary. |
| **ClinicalBERT** | Multiple groups (e.g. Alsentzer et al., MIT) | Open — pretrained on real clinical notes (e.g., MIMIC-III), for EHR-based tasks like readmission risk and clinical NLP — directly the model family behind tasks like the Hospital Readmission project. |
| **Code Llama / StarCoder** | Meta / BigCode | Open-weight, code-specialized. Code Llama is Llama fine-tuned on code; StarCoder is trained from scratch on permissively-licensed public code — both for code completion/generation rather than general chat. |

---

## How "7B" and "13B" are actually counted — stage by stage

"7 billion parameters" isn't a marketing round number — it's the literal sum of every weight matrix in the model. Worked below on **Llama 2 7B's real published architecture** (`hidden_size=4096`, `n_layers=32`, `intermediate_size=11008`, `vocab_size=32000`), computed stage by stage, not asserted.

### The formula, per stage

<div class="formula">
Token embedding:        vocab × hidden
Per layer — attention:  4 × hidden²                    (Q, K, V, O projections)
Per layer — FFN:        3 × hidden × intermediate       (SwiGLU: gate, up, down projections)
Per layer — norms:      2 × hidden                      (RMSNorm scale vectors — negligible)
Final norm:              hidden                          (negligible)
Output head (LM head):  vocab × hidden                  (NOT tied to input embedding in Llama)
</div>

### Stage-by-stage, real numbers (Llama 2 7B)

| Stage | Formula (this model's numbers) | Parameters |
|---|---|---|
| Token embedding | 32,000 × 4,096 | 131,072,000 |
| Attention, **one** layer | 4 × 4,096² | 67,108,864 |
| FFN (SwiGLU), **one** layer | 3 × 4,096 × 11,008 | 135,266,304 |
| Norms, **one** layer | 2 × 4,096 | 8,192 |
| **One layer, total** | attention + FFN + norms | 202,383,360 |
| **All 32 layers** | 32 × 202,383,360 | 6,476,267,520 |
| Final norm | 4,096 | 4,096 |
| Output head (LM head) | 32,000 × 4,096 | 131,072,000 |
| **TOTAL** | embedding + all layers + final norm + head | **6,738,415,616 ≈ 6.74B** |

6,738,415,616 rounds to the **"7B"** in the model's name — not a coincidence, the literal sum. And it's not evenly spread: **FFN layers hold 64.2% of all parameters, attention holds 31.9%, embeddings only 3.9%** — a fact worth knowing cold, since most people's mental model overweights attention's share just because it's the more-discussed mechanism.

### Seeing it — four diagrams, same real numbers as the table above

The table above proves the count; these four show where every one of those numbers physically lives, zoomed in one level at a time — the full stack, then one layer, then what's actually inside a "head," then what "intermediate size" means. Hover any box for a one-line definition.

<figure class="fig" data-llmviz="stack" id="llmviz-stack"></figure>
<figure class="fig" data-llmviz="layer" id="llmviz-layer"></figure>
<figure class="fig" data-llmviz="heads" id="llmviz-heads"></figure>
<figure class="fig" data-llmviz="ffn" id="llmviz-ffn"></figure>

### Not a one-off — the same formula, a second real model

Applying the identical formula to **Llama 2 13B's** real published config (`hidden=5120`, `n_layers=40`, `intermediate=13824`) gives **13,015,864,320 ≈ 13.016B** — matching its name almost exactly, computed independently, confirming the formula rather than being tuned to fit one example.

<div class="callout"><span class="tag">Honest limit of this formula</span>Applying it naively to Llama 2 70B's config overcounts (≈78.4B vs. the real ≈68.9B) — 70B uses <strong>grouped-query attention (GQA)</strong>, where multiple query heads share fewer key/value heads, shrinking the K/V projections below the naive 4×hidden² assumption. The formula above is exactly right for standard multi-head attention (7B/13B); a real GQA model needs one more term (fewer K/V parameters, same Q/O). Getting the boundary of your own formula wrong on a bigger model, and saying so, beats quietly presenting a fabricated-looking "70B" number as if it were exact.</div>

### Parameters vs. training tokens — why 7B isn't trained on just any amount of data

The question "how many tokens does it take to get a 7B-parameter model" has a real, quantitative answer via the **Chinchilla scaling law**: compute-optimal training uses roughly **D ≈ 20 × N** tokens for N parameters.

| | Real number |
|---|---|
| Llama 2 7B parameters (computed above) | 6,738,415,616 |
| Chinchilla-optimal token count (20 × N) | 134,768,312,320 (≈134.8B tokens) |
| **Actual** published Llama 2 7B training tokens | **2,000,000,000,000 (2T tokens)** |
| Actual tokens-per-parameter ratio | 296.8 |
| How far beyond Chinchilla-optimal | **≈14.8×** |

Llama 2 7B was trained on roughly **15× more tokens than Chinchilla training-compute-optimality would suggest** — a real, deliberate choice, not an oversight. Chinchilla's D≈20N minimizes *training* compute for a target loss; it says nothing about *inference* cost. Meta's own stated rationale: a smaller model trained far past the Chinchilla point can reach the same quality as a larger Chinchilla-optimal model while costing much less to *serve* at inference time, for every query, forever — and inference cost, not training cost, dominates a deployed model's total lifetime cost. This is exactly why model families now publish both a parameter count **and** a token count — one number alone doesn't tell you whether a model was trained for training-efficiency or serving-efficiency.

<div class="callout"><span class="tag">Why architects pick these specific dimensions</span><code>hidden_size</code>, <code>n_layers</code>, and <code>intermediate_size</code> aren't picked independently — model families keep a roughly consistent width-to-depth ratio (wider models get proportionally more layers) so that compute stays balanced between attention and FFN work as the model scales, and <code>n_heads</code> is sized so each attention head's dimension (<code>hidden/n_heads</code>) lands in a consistent range (commonly 64–128). The target parameter count (and the compute budget it implies) is decided first; the specific <code>(hidden, layers, intermediate)</code> triple is then chosen to hit that count at a balanced aspect ratio — "7B" is a design target the architecture is solved for, not a byproduct.</div>

---

## How do you actually know if one of these 25 models is "good" — a chain from raw output to a leaderboard number

### 1. Once a model is trained, what's the very first, crudest way to know if it's any good?
Run it on questions with a known right answer and check how many it gets correct — the same idea as any classical ML accuracy check, just applied to a language model's generated text instead of a classifier's predicted label.

### 2. Doing that by hand doesn't scale to comparing 25 models — what replaces it?
A **benchmark**: a large, fixed, standardized set of questions with known answers, run identically across every model, so the resulting score is actually comparable model-to-model instead of "whatever questions I happened to think of."

### 3. What does MMLU actually test, and how is its score computed?
**MMLU (Massive Multitask Language Understanding)** — ~16,000 multiple-choice questions across 57 subjects (law, medicine, history, math, and more). The score is just accuracy: `correct answers / total questions`, e.g. 86% — but the "multitask" part matters: a model can't specialize its way to a high score, since scoring well requires broad competence across genuinely unrelated subjects at once.

### 4. Multiple-choice is one format. What does a benchmark testing PLAIN COMMON SENSE look like instead?
**HellaSwag** — given a real-world situation's beginning, pick which of several possible endings is the most plausible continuation. The wrong options are deliberately generated to be superficially plausible (grammatically fine, topically related) so a model has to actually reason about physical/social plausibility, not just pattern-match surface fluency.

### 5. Both of those are still multiple-choice. What does a benchmark testing whether a model can actually WRITE correct code look like?
**HumanEval** — 164 hand-written programming problems, each with a function signature, a docstring, and hidden unit tests. The model has to generate a complete, runnable function body; scoring runs the generated code against the hidden tests and reports **pass@k** (the probability that at least one of `k` sampled generations passes all tests) — a genuinely different scoring shape from "did it pick option A, B, C, or D," because there's no multiple-choice guessing floor at all.

### 6. Why do model cards report FIVE OR SIX different benchmark scores instead of just one overall number?
Because each benchmark probes a different failure mode — a model can genuinely excel at broad factual recall (high MMLU) while being mediocre at generating correct code (lower HumanEval), or vice versa. **GLUE/SuperGLUE** (a suite of classic NLP tasks — sentiment, entailment, similarity) plays the same role for encoder-style models that MMLU/HellaSwag/HumanEval play for generative LLMs. Reporting one blended number would hide exactly the tradeoff a real deployment decision needs to see.

### 7. Do these standardized benchmarks have limitations, the same way BLEU/ROUGE do for translation/summarization (covered in `nca-genl`)?
Yes — the biggest one is **contamination**: if a benchmark's actual questions leaked into a model's pretraining data (increasingly likely for older, widely-scraped benchmarks), the model can score well by having memorized answers rather than by reasoning, inflating the number without reflecting real capability. This is exactly why newer benchmarks are released continuously and older ones are treated with growing skepticism over time — the same "don't trust a single automatic metric forever" caution already established for ROUGE/BLEU in `nca-genl` and `rag-deeper.md`'s RAGAS section.

### Summary example
Comparing two candidate models for a coding-assistant product: Model A scores MMLU 82% / HumanEval 71%; Model B scores MMLU 79% / HumanEval 58%. A single blended "average benchmark score" might even call these close — but for THIS specific product, HumanEval is the metric that actually matters, and Model A's 13-point lead there is the real signal, not the 3-point MMLU gap. Choosing based on the wrong benchmark for the actual use case is a more common mistake than choosing based on no benchmark at all.

---

## Does a model work equally well in every language — multilinguality, chained from tokenization

### 1. `nca-genl` already covers tokenization (BPE/WordPiece/SentencePiece) turning text into integers. Does that process work the same way for every language?
Not equally well. A tokenizer's vocabulary is learned from its training data's actual text — if that training data is mostly English, common English words become single tokens while text in other languages gets chopped into many more, smaller sub-word pieces just to represent the same amount of meaning.

### 2. Why does that tokenization difference actually matter, beyond being a curiosity?
Because cost and context budget are both measured in TOKENS, not words or characters. If the same sentence takes 15 tokens in English but 40 tokens in Hindi on the identical model, that non-English text costs roughly 2.7× more to process and eats through the context window 2.7× faster — a real, measurable tax on non-English usage baked in at the tokenizer level, before the model even starts reasoning.

### 3. Given that unequal tokenization, does the model's actual REASONING quality also differ by language, or just the cost?
Both, typically. Pretraining data is usually dominated by English (and a handful of other high-resource languages), so a model has simply seen vastly more examples of English reasoning patterns — quality on genuinely low-resource languages (with little training text available anywhere) tends to lag behind high-resource languages even after accounting for the tokenization tax, because there's less signal to have learned from in the first place.

### 4. Is fine-tuning a whole new model from scratch the only fix for a specific target language?
No — the same **LoRA/PEFT** approach from `core-technical-depth.md` applies directly: freeze the pretrained base (which already carries broad, transferable knowledge from its dominant languages) and train a small adapter specifically on target-language data, the same freeze-then-adapt idea used for domain adaptation, just applied to language coverage instead of subject-matter coverage.

### 5. How would you actually verify a model works acceptably in a target language, rather than assuming?
Run the same benchmark methodology as above (question 2-6), but on a benchmark translated into or natively written in the target language — a model's English MMLU score tells you nothing reliable about its performance on the same subjects asked in Vietnamese or Swahili, which is exactly why multilingual-specific benchmarks (e.g., a translated MMLU variant) exist as their own category, not an assumed extension of the English number.

### Summary example
Deploying a customer-support LLM for a market where most users write in Tagalog. The English benchmark scores in this doc's model tables say nothing directly useful here — the real due diligence is checking token cost specifically for Tagalog (likely several times the English rate), sourcing or running a Tagalog-language benchmark rather than assuming the English MMLU score transfers, and treating a LoRA adapter fine-tuned on Tagalog support transcripts as the practical fix if base performance falls short, rather than assuming a bigger English-dominant model will simply "figure it out."

---

## How the API itself evolved — five real generations, string-in/string-out to agents that spawn agents

A mental model, not an official version number any vendor publishes — but every stage below is a real, dated, documented shift in how you actually talk to these models, and knowing which generation a given API surface belongs to explains *why* it's shaped the way it is.

### Generation 1 (2020–2022): one string in, one string out
GPT-3's original Completions API (2020) took a single flat text string and returned its continuation — no roles, no turns, no concept of "system" vs. "user." Few-shot prompting in this era meant literally concatenating examples into that one string. The structuring trick that survives from this era, still real and still recommended today: **Anthropic's XML-tag convention** (`<thinking>`, `<document>`, `<answer>`) — a way to fake sections inside a single flat prompt, which Claude models are specifically trained to pay attention to. It's a workaround from the single-string era that outlived the era itself because it still works.

### Generation 2 (from March 2023): the string gets structured
OpenAI's Chat Completions API launched March 1, 2023 alongside `gpt-3.5-turbo`, replacing the freeform string with `messages: [{role: system/user/assistant, content}]` — a **dedicated system slot** for behavioral instructions ("do this, never do that"), separated from the actual conversation turns. Function/tool calling landed three months later (June 13, 2023), adding a *third* structural piece — a `tools` array the model could invoke instead of only ever emitting prose. Anthropic's Messages API mirrors this same three-way split (`system`, `messages`, `tools`). This is the generational shift that matters most in practice: instructions, conversation, and capabilities became three separate, independently-cacheable inputs instead of one string you hoped the model parsed correctly.

### Generation 3: trained on what the previous generation actually said
Once a generation is deployed via API, its real production interactions become training signal for the *next* one — RLHF reward models built from logged preference data, and Anthropic's Constitutional AI/RLAIF approach specifically has a model critique outputs (its own or a prior model's) as the training signal, rather than relying only on human-labeled preferences. Exact recipes are proprietary and vendor-specific; the general shape — each generation partly trained on evidence of how the last one actually got used and where it fell short — is real and publicly discussed, not an implementation detail worth overclaiming precision on.

### Generation 4: one call isn't enough — decompose into a dependency graph
A single LLM call hits a ceiling on tasks with unpredictable structure. The real, documented fix (Anthropic's own "Building Effective Agents" engineering post) is decomposing a task across *multiple* LLM calls with real dependencies between them — the **orchestrator-workers** pattern specifically: a central call dynamically breaks the task into subtasks nobody could have hardcoded in advance, delegates each to a worker call, and synthesizes the results. Prompt chaining (sequential, fixed steps) and parallelization (fan-out, then merge) are the simpler siblings of the same idea. This is the generation where "the LLM" stopped being one call and became a small distributed system.

### Generation 5: trained to reason, checks its own work, and delegates
The current generation: models with a real, separate **thinking** mode (`thinking: {type: "adaptive"}` in Claude's API, o1-style reasoning models elsewhere) that reason before answering, with an `effort` dial controlling how much. Two real, documented behavioral shifts ride along with this: models now **verify their own output without being told to** (current-generation Claude does this by default — prompt guidance for it has flipped from "please double-check your work" to "stop double-checking, you're doing it too much"), and they **delegate to subagents more readily** — spawning a sub-call to handle an independent chunk of work rather than doing everything in one linear trace, the same pattern Anthropic's Managed Agents platform formalizes as a declared `multiagent` coordinator with a roster of agents it can call.

**Why this framework is worth having:** each generation didn't replace the previous one's problem, it moved it. Gen 2 solved "where do instructions live," Gen 4 solved "how do you do more than one call's worth of work," Gen 5 is solving "how do you know the multi-call answer is actually right" — and that last question is exactly what `Designing an Explainable, Debuggable AI Agent System` (`system-design-deep-drills.md`, Drill 5) and the real agent failures in `real-world-incidents.md` Part 3B are about.

---

## Practice Q&A (Self-Test)

**Q1. What's the actual difference between "open source," "open-weight," and "closed-source" for an LLM — and which category does Llama fall into?**
A: Closed-source means the weights are never released, only API access. Open-weight means the trained weights are downloadable and runnable, but the license may restrict use and the training code/data usually isn't released. Fully open means weights, training code, and often data are all released under a permissive license. Llama is **open-weight**, not fully open source — Meta's license has commercial-use restrictions and the training data/code aren't published.

**Q2. Why is BERT not typically called an "LLM" in the modern sense, despite being hugely influential?**
A: BERT is a **bidirectional encoder-only** model — it's trained to understand text (via masked-token prediction) and produce representations for classification/NER/QA via fine-tuning, not to generate text autoregressively. Modern "LLM" usually implies a decoder-only, generative, next-token-prediction model like GPT or Llama.

**Q3. What's the key architectural difference between Mixtral and a dense model like Llama, and why does it matter for inference cost?**
A: Mixtral is a sparse **mixture-of-experts** model — 47B total parameters, but a router activates only ~13B of them per token. A dense model of the same total size would use all its parameters on every token. This means Mixtral gets much higher quality than a dense 13B model while costing roughly the same to run at inference time.

**Q4. What happened to "Bard," and what should you actually call Google's current model?**
A: Bard was the product name for Google's chatbot, originally built on LaMDA and later upgraded to PaLM 2. It was rebranded to **Gemini** in February 2024 — "Bard" no longer refers to an active product, and using the name in a current context is out of date.

**Q5. Why does PubMedBERT outperform BioBERT on biomedical benchmarks despite both being biomedical BERT variants?**
A: BioBERT starts from general-domain BERT and continues pretraining on biomedical text, so part of its vocabulary/capacity is still shaped by general-domain text. PubMedBERT is pretrained **from scratch** entirely on PubMed text, so its whole vocabulary and representation space is biomedical-specific from the start — which the PubMedBERT paper showed outperforms the continued-pretraining approach on biomedical tasks.

**Q6. What's the actual size/speed/performance tradeoff DistilBERT makes, and when would you pick it?**
A: DistilBERT is roughly 40% smaller and 60% faster than BERT while retaining about 97% of its performance. Pick it when inference latency or serving cost matters more than the last few points of accuracy — a real production constraint, not just a toy benchmark distinction.

**Q7. Why might you pick Cohere's Command R over GPT-4 for a specific application?**
A: Command R is purpose-built and optimized specifically for **RAG and tool-calling workflows**, rather than being a general-purpose chat model that also does RAG reasonably well. For a retrieval-heavy enterprise application, that specialization can mean better grounding/citation behavior and lower cost than a general flagship model.

**Q8. What's the difference between Alpaca's and Vicuna's approach to instruction-tuning a base Llama model?**
A: Alpaca fine-tunes Llama on ~52K instruction examples that were themselves generated by an LLM (self-instruct style) — proving cheap synthetic instruction data could unlock real chat capability. Vicuna fine-tunes Llama on real user-shared ChatGPT conversations, which produced noticeably higher chat quality and became an early standard benchmark for open chat models.

**Q9. For a task involving real clinical notes / EHR data (like a readmission-risk model), which model family from this list is the natural fit, and why?**
A: ClinicalBERT — it's pretrained directly on real clinical notes (e.g., MIMIC-III), so it already has a representation space shaped by clinical language, abbreviations, and structure, unlike general BERT or even biomedical-literature models like BioBERT/PubMedBERT which are trained on published papers, not bedside notes.

**Q10. What's DeepSeek notable for relative to other open-weight models on this list?**
A: Strong reasoning and coding performance achieved at a fraction of the usual training cost, using a large sparse mixture-of-experts architecture — and the R1 line specifically popularized releasing transparent chain-of-thought reasoning traces alongside the final answer, rather than hiding the reasoning process.

**Q11. In Llama 2 7B's real parameter breakdown, which stage holds the most parameters — attention or FFN — and by how much?**
A: FFN, by a wide margin: 64.2% of the total vs. attention's 31.9%. Per layer, FFN is 3 × hidden × intermediate = 135,266,304 params vs. attention's 4 × hidden² = 67,108,864 — roughly double — because the FFN's intermediate dimension (11,008) is much wider than the model's hidden dimension (4,096).

**Q12. Why does the naive parameter-counting formula (4×hidden² for attention) overcount Llama 2 70B but land almost exactly right for 7B and 13B?**
A: 70B uses grouped-query attention (GQA), where multiple query heads share a smaller number of key/value heads — shrinking the K/V projection matrices below the standard 4×hidden² assumption. 7B and 13B use standard multi-head attention (every head gets its own K/V), so the naive formula applies exactly — computed as 6,738,415,616 and 13,015,864,320 respectively, both matching their published names almost precisely.

**Q13. Llama 2 7B was trained on 2 trillion tokens — is that more or less than Chinchilla's compute-optimal recommendation, and why would a lab choose that deliberately?**
A: Far more — Chinchilla-optimal for a 6.74B-parameter model is roughly 134.8B tokens (20×N), so 2T tokens is about 14.8× beyond that point. Chinchilla's D≈20N minimizes training compute for a target loss, but says nothing about inference cost — training a smaller model on far more tokens than training-optimal can reach the same quality while being far cheaper to serve at inference time for every future query, which dominates a deployed model's real lifetime cost.

**Q14. Model A scores 3 points higher than Model B on MMLU. Is Model A the better choice for a coding-assistant product?**
A: Not necessarily — MMLU tests broad multitask factual/reasoning knowledge across 57 academic subjects, which isn't the same skill as writing correct, runnable code. For a coding assistant specifically, HumanEval (pass@k on hidden unit tests) is the more directly relevant benchmark; a model with a lower MMLU but a meaningfully higher HumanEval score is very plausibly the better pick for this specific product, since the benchmark has to match the actual deployment task, not just be "the more well-known number."

**Q15. A model scores unexpectedly high on an older, widely-used benchmark. What's a real reason to be skeptical of that score before trusting it?**
A: Contamination — if that benchmark's actual questions leaked into the model's pretraining data (increasingly likely for older benchmarks scraped widely across the internet), the model can score well by having memorized answers rather than by genuinely reasoning, inflating the number without reflecting real capability. This is the same "don't trust one automatic metric forever" caution already covered for BLEU/ROUGE in `nca-genl` and RAGAS in `rag-deeper.md`.

**Q16. The same sentence takes 15 tokens in English and 40 tokens in Hindi on an identical model. What are the two separate, compounding consequences of that gap, beyond "it's just less efficient"?**
A: Cost and context budget are both measured in tokens, so the Hindi text costs roughly 2.7× more to process per equivalent amount of meaning, AND it eats through the model's fixed context window 2.7× faster — meaning a Hindi conversation hits the context limit (and the associated KV-cache memory cost from `nca-genl`) sooner than an equivalent English conversation would, on the exact same model.

<script>
(function(){
"use strict";
const esc=s=>String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
const txt=(x,y,cls,s,anch)=>`<text x="${x}" y="${y}" class="${cls}"${anch?` text-anchor="${anch}"`:""}>${s}</text>`;
const DEFS='<defs><marker id="llmah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="var(--muted)"/></marker></defs>';
const box=(x,y,w,h,cls,title,sub,sub2)=>`<rect class="${cls}" x="${x}" y="${y}" width="${w}" height="${h}" rx="6"/>`
  +(title?txt(x+w/2,y+h/2+(sub?-4:4),"vlab",title,"middle"):"")
  +(sub?txt(x+w/2,y+h/2+10,"vsm",sub,"middle"):"")
  +(sub2?txt(x+w/2,y+h/2+22,"vsm",sub2,"middle"):"");
const hit=(bx,by,bw,bh,tip,inner)=>`<g class="llmhit" tabindex="0" aria-label="${esc(tip)}" data-tip="${esc(tip)}"><rect class="hitbg" x="${bx}" y="${by}" width="${bw}" height="${bh}" rx="6" fill="transparent" stroke="transparent" stroke-width="2"/>${inner}</g>`;
const varr=(x,y1,y2)=>`<line x1="${x}" y1="${y1}" x2="${x}" y2="${y2}" stroke="var(--muted)" stroke-width="1.3" marker-end="url(#llmah)"/>`;
const harr=(x1,x2,y)=>`<line x1="${x1}" y1="${y}" x2="${x2}" y2="${y}" stroke="var(--muted)" stroke-width="1.3" marker-end="url(#llmah)"/>`;
const LLMVIZ={};

LLMVIZ.stack={t:"The full stack — how a 7B model is built from repeated pieces",
  hint:"Llama 2 7B: 32 identical layers between an embedding and an output head",w:900,h:520,
  cap:"Every layer has the IDENTICAL shape — takes a [*, 4096] vector, returns a [*, 4096] vector — which is exactly why they can stack: layer 2 doesn't know or care whether its input came from the embedding or from layer 1. Depth (32 layers) and width (4096 hidden dim) are independent dials; this model chose both.",
  svg(){
    let s="";
    const cx=450;
    s+=hit(160,10,340,36,"The input to the whole model: a sequence of token IDs, e.g. [1,450,6635,338] for 'The capital is'.",
      box(160,10,340,36,"chip","Input token ids","[batch, seq_len]"));
    s+=varr(cx,46,72);
    s+=hit(160,74,340,58,"Token embedding: a lookup table, one 4096-dim row per vocabulary word. 32,000 × 4,096 = 131,072,000 parameters.",
      box(160,74,340,58,"bigbox","Token Embedding","[32,000 × 4,096]","131,072,000 params"));
    s+=varr(cx,132,158);
    s+='<rect x="176" y="168" width="308" height="90" rx="8" class="cell" opacity="0.5"/>';
    s+='<rect x="168" y="176" width="308" height="90" rx="8" class="cell" opacity="0.75"/>';
    s+=hit(160,184,340,90,"One Transformer Layer, repeated 32 times with SEPARATE weights each time (not the same weights reused) — 202,383,360 parameters per layer, 6,476,267,520 across all 32.",
      box(160,184,340,90,"bigbox","Transformer Layer","× 32 (separate weights each)","202,383,360 params / layer"));
    s+=txt(520,229,"vacc","← see “Inside one layer” below");
    s+=varr(cx,274,300);
    s+=hit(160,302,340,40,"Final RMSNorm before the output head — a single 4,096-element scale vector, 4,096 parameters. Negligible in size, necessary for stable outputs.",
      box(160,302,340,40,"cell","Final Norm","4,096 params"));
    s+=varr(cx,342,368);
    s+=hit(160,370,340,58,"The output/LM head projects the final 4096-dim vector back up to a score for every one of the 32,000 vocabulary words. NOT the same matrix as the input embedding in Llama (untied). 32,000 × 4,096 = 131,072,000 parameters.",
      box(160,370,340,58,"bigbox","Output Head (LM head)","[4,096 × 32,000]","131,072,000 params"));
    s+=varr(cx,428,454);
    s+=hit(160,456,340,40,"The final output: a probability distribution over all 32,000 possible next tokens.",
      box(160,456,340,40,"chip","Output probabilities","over 32,000 vocab tokens"));
    return s;}};

LLMVIZ.layer={t:"Inside ONE transformer layer — attention, then FFN, same shape in and out",
  hint:"202,383,360 params in this one layer: 67.1M attention + 135.3M FFN + 8K norms",w:980,h:280,
  cap:"Two sub-blocks per layer, always in this order: attention first (mixes information ACROSS token positions), then FFN (transforms EACH position independently, no mixing across tokens). Both sub-blocks preserve the 4096 dimension and both get a residual (skip) connection, which is why 32 of these can stack without the signal degrading.",
  svg(){
    let s="";
    s+=hit(10,110,110,50,"The input to this layer: a [*, 4096] vector — either the embedding output, or the previous layer's output.",
      box(10,110,110,50,"chip","input","[4096]"));
    s+=harr(120,190,135);
    s+=hit(190,90,260,90,"Multi-Head Attention: Q, K, V, O projections. 32 heads of 128 dims each (see the next diagram). 4 × 4096² = 67,108,864 parameters.",
      box(190,90,260,90,"bigbox","Multi-Head Attention","32 heads × 128 dim","67,108,864 params"));
    s+=hit(460,112,36,46,"Residual (skip) connection: the attention block's output is ADDED to its own input, not used to replace it — this is what lets gradients flow cleanly through 32 stacked layers.",
      '<circle cx="478" cy="135" r="16" class="cellhot"/>'+txt(478,140,"vacc","+","middle"));
    s+=harr(514,584,135);
    s+=hit(584,90,260,90,"Feed-Forward Network (SwiGLU): expands 4096 → 11,008 → back to 4096. 3 × 4096 × 11,008 = 135,266,304 parameters — see the diagram below for what “intermediate size” means here.",
      box(584,90,260,90,"bigbox","FFN (SwiGLU)","intermediate = 11,008","135,266,304 params"));
    s+=hit(854,112,36,46,"Second residual connection, same purpose as the first — FFN's output is added to its input, not a replacement.",
      '<circle cx="872" cy="135" r="16" class="cellhot"/>'+txt(872,140,"vacc","+","middle"));
    s+=harr(908,960,135);
    s+=txt(960,135,"vsm","→ [4096]","start");
    s+=txt(490,20,"vacc","one layer total: 202,383,360 params (67,108,864 + 135,266,304 + 8,192 norms)","middle");
    s+=txt(490,260,"vsm","note: input shape [4096] === output shape [4096] — this is WHY 32 of these can stack","middle");
    return s;}};

LLMVIZ.heads={t:"What a “head” actually is — one 4096-dim vector split into 32 parallel 128-dim attentions",
  hint:"head_dim = hidden / n_heads = 4,096 / 32 = 128, exactly",w:980,h:400,
  cap:"“32 heads” doesn't mean 32 separate attention mechanisms bolted together — it means the SAME 4096-dim Q/K/V vectors get reshaped into 32 chunks of 128 dims each, so 32 independent (small, cheap) attention computations happen in parallel, each free to focus on a different kind of relationship between tokens, before being concatenated back to 4096 and mixed by the output projection.",
  svg(){
    let s="";
    s+=hit(370,10,240,40,"The layer's Q, K, and V projections each produce a 4096-dim vector per token — before splitting into heads.",
      box(370,10,240,40,"bigbox","Q / K / V vector","[4096] per token"));
    s+=varr(490,52,76);
    s+=txt(490,92,"vacc","reshape: [4096] → [32 heads × 128 dims]","middle");
    const heads=[0,1,2,3,"…",30,31];
    const hx=[40,160,280,400,520,640,760];
    heads.forEach((h,i)=>{
      const x=hx[i];
      if(h==="…"){s+=txt(x+34,150,"vsm","…","middle");return;}
      s+=hit(x,110,68,60,"Head "+h+": a 128-dim slice of Q/K/V. Computes its own attention — softmax(Q_h·K_hᵀ/√128)·V_h — completely independently of every other head.",
        box(x,110,68,60,"cell","head "+h,"128 dim"));
    });
    s+=txt(490,196,"vsm","each head independently: softmax(Q_h · K_hᵀ / √128) · V_h","middle");
    s+=hit(370,220,240,50,"All 32 heads' 128-dim outputs are concatenated back into one 4096-dim vector — 32 × 128 = 4096, exactly, no information lost or padded.",
      box(370,220,240,50,"bigbox","Concat","32 × 128 = [4096]"));
    s+=varr(490,272,296);
    s+=hit(370,298,240,50,"The output projection O mixes the concatenated heads back together — this is the 4th of the 4 matrices (Q,K,V,O) in the 4×4096² attention parameter count.",
      box(370,298,240,50,"bigbox","Output proj. (O)","[4096 × 4096]"));
    s+=txt(490,375,"vsm","more heads = more, narrower “views” on the same data — not more total capacity (still 4096 dims total, however it's split)","middle");
    return s;}};

LLMVIZ.ffn={t:"What “intermediate size” actually is — the FFN's expand-then-compress",
  hint:"11,008 / 4,096 ≈ 2.688× — deliberately narrower than the classic 4×, because SwiGLU uses 3 matrices instead of 2",w:980,h:360,
  cap:"“Intermediate size” is the FFN's internal working width — wider than the model's hidden size everywhere else, because the FFN is where per-token processing capacity actually lives. SwiGLU's 3-matrix design uses a smaller multiplier (2.688× here, not the classic 4×) specifically to keep the total FFN parameter budget comparable to a plain 2-matrix GELU FFN at 4× — verified: 8/3 × 4096 ≈ 10,923, and the real value, 11,008, is that same target rounded up to a multiple of 256 for hardware efficiency.",
  svg(){
    let s="";
    s+=hit(20,140,110,50,"The FFN's input: a [4096] vector, one per token, processed independently of every other token's position.",
      box(20,140,110,50,"chip","input","[4096]"));
    s+=harr(130,195,165);
    s+=hit(195,60,190,70,"gate_proj: 4096 → 11,008. One of two matrices that expand the vector into the FFN's wide “intermediate” working space.",
      box(195,60,190,70,"bigbox","gate_proj","[4096 × 11,008]"));
    s+=hit(195,230,190,70,"up_proj: 4096 → 11,008, the second expansion matrix, computed in parallel with gate_proj.",
      box(195,230,190,70,"bigbox","up_proj","[4096 × 11,008]"));
    s+=harr(385,455,95);
    s+=harr(385,455,265);
    s+=hit(455,145,140,70,"SwiGLU activation: silu(gate) ⊙ up — an elementwise gating multiply in the 11,008-wide space. This nonlinearity is WHY the FFN can learn things a single linear layer never could.",
      box(455,145,140,70,"cellhot","SwiGLU","silu(gate) ⊙ up","[11,008]"));
    s+=harr(595,660,180);
    s+=hit(660,145,190,70,"down_proj: 11,008 → 4096, compressing back down to the model's hidden size so the output can be added back via the residual connection.",
      box(660,145,190,70,"bigbox","down_proj","[11,008 × 4096]"));
    s+=harr(850,920,180);
    s+=txt(920,180,"vsm","→ [4096]","start");
    s+=txt(490,20,"vacc","3 matrices × 4096 × 11,008 = 135,266,304 params — the FFN's full cost","middle");
    s+=txt(490,340,"vsm","classic 2-matrix GELU FFN uses 4× hidden; SwiGLU's 3 matrices use ≈2.688× to land at the SAME total param budget","middle");
    return s;}};

const llmvtip=document.createElement("div");llmvtip.id="llmvtip";document.body.appendChild(llmvtip);
document.querySelectorAll("figure[data-llmviz]").forEach(f=>{
  const d=LLMVIZ[f.dataset.llmviz];if(!d)return;
  f.innerHTML=`<div class="vhead"><span class="vtitle">${d.t}</span><span class="vhint">${d.hint||""}</span></div>`
    +`<svg viewBox="0 0 ${d.w} ${d.h}" width="${d.w}" style="max-width:100%;height:auto;display:block" role="img" aria-label="${esc(d.t)}">${DEFS}${d.svg()}</svg>`
    +`<figcaption>${d.cap}</figcaption>`;
});
document.addEventListener("mousemove",e=>{
  const h=e.target.closest?e.target.closest("g.llmhit"):null;
  if(h&&h.dataset.tip){
    llmvtip.textContent=h.dataset.tip;llmvtip.style.display="block";
    const r=llmvtip.getBoundingClientRect();
    let x=e.clientX+14,y=e.clientY+18;
    if(x+r.width>innerWidth-8)x=innerWidth-r.width-8;
    if(y+r.height>innerHeight-8)y=e.clientY-r.height-10;
    llmvtip.style.left=x+"px";llmvtip.style.top=y+"px";
  }else{llmvtip.style.display="none";}
});
})();
</script>
