# My Projects — Architecture, Code Structure & Module Breakdown

Every fact in this doc is pulled from what's already stated across `system-design-prep.md`, `core-technical-depth.md`, `live-coding-prep.md`, and `behavioral-partnership-star-stories.md` — nothing here is invented to make a diagram look complete. Where a real architecture implies a piece that isn't explicitly confirmed anywhere else in this hub, it's marked **`[FILL IN/CONFIRM]`** — the same honesty convention `behavioral-partnership-star-stories.md` already uses for gaps only your real memory can close. Three zoom levels per project, deepest first for the two with the most established detail:

- **Level 1 — High-level architecture**: the boxes-and-arrows shape, what an interviewer sees on a whiteboard.
- **Level 2 — Code-level breakdown**: what services/layers actually exist.
- **Level 3 — Module-level breakdown**: what's inside each layer.

---

# FinSight — Multi-Agent Wealth Management Platform

3 LLMs across 7 agents, an Isolation Forest fraud layer, OTP-gated transactions, deployed on Azure Kubernetes Service (AKS) with CI/CD, holding sub-1-second real-time portfolio sync latency.

## Level 1 — High-level architecture
```
                          ┌─────────────────────────────────────┐
  User action        ┌───▶│  SYNCHRONOUS PATH (must fit <1s)     │───▶ updated UI
  (trade/portfolio    │    │  Fraud check (Isolation Forest)      │     (fast, blocking)
  update request) ────┤    │  Portfolio math                      │
                       │    │  OTP-gated transaction confirmation  │
                       │    └─────────────────────────────────────┘
                       │
                       └───▶┌─────────────────────────────────────┐
                             │  ASYNCHRONOUS PATH (completes after  │───▶ UI updates again
                             │  the sync response already returned) │     when ready, clearly
                             │  3-agent debate:                     │     communicated as such
                             │   Portfolio agent → proposes trade   │
                             │   Market agent → market context      │
                             │   Critic agent → finds flaws/risk    │
                             │  (+ 4 more of the 7 total agents,    │
                             │   [FILL IN/CONFIRM] specific roles)  │
                             └─────────────────────────────────────┘
                                          deployed on AKS + CI/CD
```
**The one design decision that shapes this entire diagram:** everything on the critical path had to be fast enough to not need an LLM — which is *why* fraud detection is Isolation Forest (a classical, millisecond-speed model) instead of an LLM call, and why the "does an extra debate round actually change the recommendation" question was answered per-request, not defaulted to always-on. The sync/async split itself — not any individual agent — is the actual architectural achievement here, and it's the answer to "how did you hit sub-1-second latency with 3 LLMs in the loop" in one sentence.

## Level 2 — Code-level breakdown
```
┌──────────────────────────────────────────────────────────────────┐
│  API / orchestration layer                                        │
│  [FILL IN/CONFIRM: specific framework — FastAPI is confirmed for   │
│   NaviDoc below, not explicitly stated for FinSight]                │
├──────────────────────────────────────────────────────────────────┤
│  Agent layer (LangGraph-shaped orchestration — see                 │
│  `practice-langgraph.md` for the general pattern this maps onto)   │
│    • Portfolio agent   — proposes allocation/trade                 │
│    • Market agent      — current market-condition context          │
│    • Critic agent      — finds flaws/unstated risk                 │
│    • [FILL IN/CONFIRM] — 4 more agents, roles not specified         │
│                           elsewhere in this hub                     │
├──────────────────────────────────────────────────────────────────┤
│  Classical-ML services (deliberately NOT LLM calls)                 │
│    • Isolation Forest — fraud/anomaly detection, on critical path   │
│    • Random Forest — ticker-scoring model, 87% accuracy,            │
│      trained across 800+ tickers                                    │
├──────────────────────────────────────────────────────────────────┤
│  Security layer                                                    │
│    • OTP-gated transaction confirmation                            │
├──────────────────────────────────────────────────────────────────┤
│  Deployment: Azure Kubernetes Service + CI/CD pipeline               │
└──────────────────────────────────────────────────────────────────┘
```

## Level 3 — Module-level breakdown
- **Portfolio agent module** — consumes the Random Forest ticker-scoring model's output (feature/constraint set predicting upside across 800+ tickers) and turns it into a concrete allocation/trade proposal.
- **Market agent module** — pulls current market-condition context that could argue against or refine the Portfolio agent's proposal; the two agents deliberately hold different framings of the same decision (is the allocation sound vs. is the timing/market context sound) rather than one agent trying to hold both.
- **Critic agent module** — prompted specifically to find flaws/unstated risk in what the other two produced, structurally a critique/debate loop (same shape as the generic drafter-critic pattern in `prompt-engineering-deeper.md`'s Reflexion section, just with 3 named roles instead of 2 generic ones).
- **Fraud detection module** — Isolation Forest, runs synchronously, pre-transaction; chosen specifically because LLM-call latency couldn't fit inside the sync budget.
- **Ticker-scoring module** — Random Forest, 87% accuracy, the feature-engineering work here (deciding what predicted upside) is the same "formalize a real-world problem into decision variables and an objective" skill named in `core-technical-depth.md`'s optimization section.
- **OTP/auth module** — gates transaction confirmation, sits after the sync path's fraud+math checks resolve.
- **Latency router** — the actual sync-vs-async decision logic; this is the piece doing the real architectural work in the whole system.
- **`[FILL IN/CONFIRM]`** — the remaining 4 agents' specific roles, the frontend stack, the portfolio-data storage layer, and the specific LLM(s) behind the 3 agents are not established elsewhere in this hub. Worth confirming from memory before stating any of these as fact in an interview.

---

# NaviDoc — Multimodal Clinical RAG Backend

FastAPI + PyTorch + PostgreSQL + MongoDB. Medical image analysis combined with RAG-based retrieval over clinical documents, 35% ROUGE/BLEU, presented at the Texas Health Informatics Alliance Conference (2025) and Texas Medical Center (2026).

## Level 1 — High-level architecture
```
  Clinical documents ──▶ [ chunk along natural SECTION      ──▶ PostgreSQL (structured,
  (guidelines, EHR-       boundaries, not blind token count,     access-controlled data)
  adjacent)                with overlap so no fact straddles         +
                            a boundary unfilled ]                MongoDB (document-shaped
                                       │                          clinical content)
                                       ▼
                          transformer ENCODER (bidirectional)
                          embeds chunks for retrieval
                                       │
  Clinician question ──▶ [ retrieve top-k, filtered by       ──▶ separate causal LLM
                            access permissions BEFORE            generates grounded answer
                            the similarity search runs,          WITH citation back to the
                            not after ]                          specific source passage
                                       │
                                       ▼
                          FastAPI serving layer ──▶ answer + citation, defensible
                                                     under direct clinician questioning
```
**Why two different attention architectures, not one model doing both jobs:** the encoder's job (bidirectional — "what does this chunk mean, for search") and the causal LLM's job (unidirectional — "generate the next token of a grounded answer") are genuinely different tasks, the same encoder-vs-decoder distinction covered in `core-technical-depth.md`'s transformer section. Trying to force one architecture to do both would fight against what each attention pattern is actually good at.

## Level 2 — Code-level breakdown
```
┌──────────────────────────────────────────────────────────────────┐
│  FastAPI serving layer                                             │
├──────────────────────────────────────────────────────────────────┤
│  PyTorch model layer                                                │
│    • Transformer encoder — embeds chunks (retrieval side)           │
│    • Causal LLM — generates grounded, cited answers                 │
│      [FILL IN/CONFIRM: specific models — other docs on this hub     │
│       note NaviDoc runs pretrained models via API/RAG, not          │
│       fine-tuned weights]                                            │
│    • Medical image analysis module (multimodal input)               │
│      [FILL IN/CONFIRM: specific architecture — the transfer-        │
│       learning pattern confirmed elsewhere (ResNet18/MobileNetV2    │
│       on the separate Alzheimer's/Pneumonia projects below) may     │
│       or may not be the same approach used here]                    │
├──────────────────────────────────────────────────────────────────┤
│  Storage layer — split BY DATA SHAPE, not arbitrarily                │
│    • PostgreSQL — structured, access-controlled data                 │
│    • MongoDB — document-shaped clinical content                      │
├──────────────────────────────────────────────────────────────────┤
│  Ingestion / chunking module                                         │
│    • Section-boundary-aware chunker (not fixed-token)                │
│    • Overlap logic (no fact straddling a boundary is lost)           │
├──────────────────────────────────────────────────────────────────┤
│  Access-control layer — permission filtering applied AT/BEFORE       │
│  retrieval, not as a post-hoc check (see `system-design-prep.md`'s   │
│  RAG-system design section for why this ordering is a hard           │
│  requirement, not a nice-to-have, in a clinical/enterprise context)  │
├──────────────────────────────────────────────────────────────────┤
│  Evaluation: 35% ROUGE/BLEU (word/phrase-overlap metric — see        │
│  `rag-deeper.md`'s RAGAS section for the richer faithfulness/        │
│  relevance metrics this doc's own UNT research work moved toward)    │
└──────────────────────────────────────────────────────────────────┘
```

## Level 3 — Module-level breakdown
- **Chunking module** — splits along the document's own natural section boundaries (a clinical guideline's precondition-then-instruction structure), with overlap specifically so a fact that straddles a boundary still lands fully inside at least one chunk; a naive fixed-token chunker was explicitly ruled out because splitting a precondition from the instruction that depends on it isn't a minor bug in a clinical context, it's a wrong-answer-with-confidence risk.
- **Embedding module (encoder)** — bidirectional transformer, turns each chunk into a retrieval-ready vector.
- **Generation module (causal LLM)** — produces the answer, prompted to only claim what the retrieved context supports and to cite which passage backs each claim.
- **Storage split module** — the PostgreSQL/MongoDB divide follows the data's actual shape and sensitivity: structured/access-controlled data in Postgres, document-shaped clinical content in Mongo — the permission-filtering logic downstream follows directly from how data was already partitioned here, not bolted on later.
- **Access-control module** — filters retrieval candidates by the requesting user's permissions *during* the similarity search, not after (retrieving an unauthorized chunk at all is treated as a security bug, not a quality one).
- **Image analysis module** — `[FILL IN/CONFIRM]`, multimodal input handling not detailed elsewhere in this hub beyond "medical image analysis."
- **Citation-mapping module** — maps a generated claim back to the actual source document/passage so a clinician can verify it directly, which is what made the system defensible under direct questioning at the Texas Health Informatics Alliance Conference and Texas Medical Center, not just accurate on an automated metric.

---

# Healthcare AI Side Projects — Lighter Treatment (Level 1 Only)

Real, named, quantified — but without the module-level detail established for FinSight/NaviDoc above. Presented at architecture level only, honestly, rather than inventing structure underneath.

### QuitBuddy — Teen Smoking-Cessation Platform
Built with a Johns Hopkins faculty collaborator. Live-avatar/voice interaction, RAG + prompt engineering (no fine-tuning), 80%+ faithfulness validated by external LLM-as-judge evaluation.
```
Teen user ──▶ live-avatar/voice interface ──▶ pretrained LLM (API) + RAG + prompt engineering
                                                          │
                                              external LLM-as-judge evaluation
                                              (checks: did this stay on-message,
                                               grounded, within domain boundaries)
                                              ──▶ 80%+ faithfulness score
```
The evaluation step is the architecturally important part here, not just a QA afterthought — talking to teenagers about substance use means an off-domain or hallucinated response is a safety concern, not just a quality miss, which is why faithfulness was explicitly measured rather than assumed.

### Clinical Assistant Chatbot
A second, more focused RAG system: vector database, contextual Q&A over healthcare documents, 35% ROUGE/BLEU, Docker-containerized for reproducible deployment.
```
Healthcare docs ──▶ vector DB ──▶ RAG retrieval ──▶ contextual Q&A ──▶ 35% ROUGE/BLEU
                                                                        (Docker-containerized)
```

### Mental Health Wellness Chatbot
`[FILL IN/CONFIRM]` — grouped elsewhere in this hub with NaviDoc/FinSight/QuitBuddy as running on pretrained models via API/RAG rather than fine-tuned weights; no further architectural detail established.

### Pneumonia Detection (MobileNetV2) & Alzheimer's MRI Staging (ResNet18)
Two separate computer-vision projects, same underlying transfer-learning mechanics — directly the same freeze-then-adapt idea as LoRA (`core-technical-depth.md`), just applied to vision instead of language:
```
Medical image ──▶ [ FROZEN pretrained backbone         ──▶ small trained        ──▶ prediction
                     (MobileNetV2 or ResNet18) ]            classification head       (pneumonia
                     pretrained features kept intact,       (only this trains,        present? /
                     low learning rate specifically         low LR to avoid           MRI stage)
                     to avoid destroying them                overfitting a small
                                                               medical dataset)
```
The shared lesson across both: never trust a model's reported accuracy without checking it on the actual target population — a habit named explicitly in `core-technical-depth.md` as carrying over into how a quantization accuracy tradeoff would be validated on real domain data, not a generic benchmark.

---

# Data Infrastructure & Reliability Engineering (Bosch / Cognizant–CapitalOne / Wipro)

Not a single product, but a real, richly-quantified body of infrastructure work — five years of hands-on database administration across Oracle, MongoDB, PostgreSQL, MySQL, MSSQL, and Redis. Worth its own architecture-level treatment given how specific the numbers are.

## Level 1 — High-level architecture (the monitoring/reliability shape across all of it)
```
  70 enterprise clients, up to 5TB scale (Bosch mobility cloud platform)
          │
          ▼
  [ Replication · network security policy enforcement · data migrations ·
    automated housekeeping ]  ──▶  sustained 99.999% availability
          │
          ├──▶ INCIDENT: ransomware-locked MongoDB instance
          │     Fix: mount locally, full backup + restore ──▶ zero data loss
          │
          ├──▶ INCIDENT: MongoDB split-brain, 6-node replica set (Cognizant/CapitalOne)
          │     Fix: evict + resync the stuck secondary, primary stayed available
          │     throughout ──▶ resolved within 30 minutes
          │
          └──▶ 120 production/dev/QA servers monitored (Cognizant/CapitalOne)
                Python + shell scripts, vectorized aggregation against recent
                history (not per-server loops) ──▶ anomaly detection in 35 seconds
```
**The one idea underneath all four boxes:** trust a faster, cheaper signal over waiting for a slow, expensive one — input/replication/node-state drift caught early, versus waiting for a customer complaint or an outage. This is explicitly named in `system-design-prep.md` as the same instinct behind input-drift-vs-outcome-drift monitoring for an ML system — the DB-ops experience and the ML-monitoring framework are the same underlying pattern, just applied to different failure signals.

## Level 2 — Code/pipeline-level breakdown
```
┌──────────────────────────────────────────────────────────────────┐
│  Azure-based ETL pipeline (built for the database team)             │
│    • Query turnaround: 6 minutes ──▶ 1.8 seconds                    │
│    • Fed the sales classification/clustering work that drove an     │
│      8% increase in annual revenue margins                           │
│    • Mechanism: single-pass, window-function-based SQL (LAG/LEAD    │
│      inside a CTE) replacing multiple round-trip queries +           │
│      application-side loops — see `sql-practice.md`'s window-        │
│      function section for the general pattern this is an instance   │
│      of                                                               │
├──────────────────────────────────────────────────────────────────┤
│  Monitoring scripts (Cognizant/CapitalOne, 120 servers)              │
│    • Python + shell                                                  │
│    • Vectorized aggregation/lookups against recent history           │
│      (`merge_asof`-shaped — compare "this event" to "most recent     │
│      relevant prior state," per entity, without per-entity loops)    │
│    • 35-second anomaly detection, replacing a fully manual review    │
├──────────────────────────────────────────────────────────────────┤
│  Incident response (ransomware + split-brain)                        │
│    • Local mount + backup/restore (ransomware, zero data loss)       │
│    • Evict + resync stuck secondary (split-brain, primary stayed     │
│      up throughout, resolved in 30 minutes)                          │
└──────────────────────────────────────────────────────────────────┘
```

## Level 3 — Module-level breakdown
- **ETL pipeline module** — extract/load/transform steps rebuilt around window functions computed once, in-database, instead of pulled back into application code and looped over; this is the literal real-world version of the `LAG`/`LEAD`-in-a-CTE pattern taught in `sql-practice.md` and `leetcode-sql.md`.
- **Monitoring/anomaly-detection module** — per-entity (per-server) baseline comparison, vectorized rather than looped, the same shape as `merge_asof`-style "compare to most recent relevant prior state" — explicitly named elsewhere in this hub as the identical pattern applicable to comparing a locomotive's current sensor reading against its own recent history.
- **Ransomware recovery module (procedure, not code)** — diagnose → mount locked volume locally → full backup → restore, chosen deliberately over faster-looking options (paying, restoring from a stale backup, immediate failover) that risked data loss.
- **Split-brain recovery module (procedure, not code)** — diagnose which node holds a divergent cluster-state view → evict it → resynchronize → verify primary availability was never interrupted.
- **Feature pipeline → business outcome link** — the ETL pipeline isn't a separate accomplishment from the 8% margin increase, it's the prerequisite for it: no amount of good problem formulation or modeling matters if the underlying data pipeline can't support the iteration speed the analysis needs, a point made directly in `problem-formulation-framework.md`.

---

## How to use this doc in an interview
Say the confirmed facts as facts. For anything marked `[FILL IN/CONFIRM]`, either fill it in from real memory before the interview, or — per this hub's own repeated advice in `system-design-prep.md` and `live-coding-prep.md` — say plainly that the specific detail isn't something you're citing precisely, rather than improvising false familiarity under a follow-up question. An interviewer probing a Level 3 module detail that turns out to be invented does more damage than an honest "I'd want to double-check that specific number before stating it as fact."
