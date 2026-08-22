# System Design Prep

System design rounds test whether you can reason about a whole system's lifecycle under real constraints — not whether you can name the fanciest architecture. The consistent pattern across every topic below: state requirements and constraints before drawing boxes, and always close the loop back to monitoring/feedback rather than stopping at "deploy the model."

**For real, dated, sourced incidents and wins** (Chevrolet's $1 chatbot, Cloudflare's 27-minute regex outage, Zillow Offers' $500M writedown, vLLM's 24x throughput, LoRA's 10,000x parameter reduction, and more) that make these frameworks concrete with actual company names and numbers, see `real-world-incidents.md` — built specifically to answer "what actually happened, not a hypothetical."

## Reading this from the hiring manager's seat
The specific thing I'd be listening for here, given your background: every production system on your résumé — FinSight on AKS, NaviDoc on FastAPI/PostgreSQL/MongoDB, the Bosch mobility platform — is cloud-native, built on infrastructure you fully controlled. A freight railroad's ML systems very often can't assume that. Sensor telemetry off a locomotive or a wayside detector may travel over intermittent field connectivity before it ever reaches a cloud region; the system of record for a maintenance work order is frequently a decades-old enterprise system that was never designed with an API in mind; and anything touching safety-relevant decisions may need to satisfy a certification or change-control process that a typical cloud deployment pipeline doesn't have to think about. My honest concern reading your résumé is: *has this candidate only ever designed for a world where the data shows up clean, on time, and through infrastructure they own end to end?*

The way to close that gap in this round isn't to pretend you've done legacy/OT integration before — you haven't, and I'd rather hear that stated plainly than hear you improvise false familiarity. What I actually want to hear is that you *ask* about it: when a prompt says "design a system to predict X from sensor data," the senior move is asking where that sensor data physically originates, how reliably and how fast it reaches anywhere you could compute on it, and what system currently holds the ground-truth labels you'd train against — before you draw a single box. Your database-operations background is a real asset here if you use it right: five years of owning production data infrastructure (including recovering from a ransomware incident and a split-brain replica set) means you instinctively know that "the data pipeline is reliable" is an assumption to verify, not a given — bring that instinct to bear explicitly on the field-data-reliability question, rather than assuming a rail environment behaves like an Azure-hosted API.

---

## ML System Design Framework: Feature Pipeline → Training → Serving → Monitoring → Feedback Loop

> **TL;DR**
> - A production ML system isn't "a model" — it's a loop: feature pipeline → training → serving → monitoring → feedback, and back again.
> - The pipeline's #1 failure mode is **training-serving skew**: a feature computed slightly differently at training time than at serving time.
> - Monitoring has to watch three separate things — is the input weird, are predictions weird, are outcomes actually getting worse — because a system can fail on any one of these while looking fine on the others.
> - Feedback loops can quietly poison themselves: if you only get ground truth on the cases the model flagged, its blind spots never get corrected.

### Plain-English explanation
A production ML system is not "a model" — it's a pipeline that turns raw data into a served prediction reliably, repeatedly, and in a way that tells you when it's wrong. The five stages form a loop, not a line: monitoring feeds back into retraining, and feedback from real outcomes feeds back into the training data itself.

**Visual + memory hook — draw this exact loop on the whiteboard before drawing anything else, in any ML system design round:**
```
┌─▶ FEATURE PIPELINE ──▶ TRAINING ──▶ SERVING ──▶ MONITORING ──┐
│   (raw data → clean,      (versioned    (API,       (input/         │
│    joined, transformed     data/code/    batch or    prediction/    │
│    features — must match   model)        real-time)  outcome drift) │
│    training↔serving        │                              │        │
│    exactly, or it's                                       │        │
│    training-serving skew)                                 ▼        │
│                                                    FEEDBACK LOOP    │
└─────────────── real outcomes fold back ◀── (did the prediction  ◀──┘
                  into future training         actually help?)
                  data
```
**Remember it as a circle, not a pipeline diagram with an end** — the single most common structural mistake in this round is presenting these five stages as a line that stops at "serving," when the whole point of the framework is that monitoring and feedback close the loop back to the start. If you only remember one thing under interview pressure, remember to draw the arrow from the last box back to the first one — everything in "Where people trip up" below is really a version of "someone treated this loop as a line and a stage silently rotted."

### Walking the loop, stage by stage

Start with **requirements**, before you draw a single box: what's the latency need (real-time vs. batch), how much volume, how stale can the input data be, and what does an error actually cost. This is just the Problem Formulation framework, pointed at infrastructure instead of model architecture.

Once requirements are set, raw data — sensor streams, transactional systems, logs — has to become a **feature pipeline**: cleaned, joined, and transformed into features. The decision that matters most here is **online/offline consistency** — features computed at training time (often batch, out of a data warehouse) have to match what's computed at serving time (often real-time, out of a streaming or low-latency store) bit-for-bit, or you get **training-serving skew**. A feature store (Feast, say) exists specifically to solve this, by centralizing the feature definitions so both paths compute the same thing.

With consistent features in hand, **training** needs to stay traceable: versioned datasets, versioned code, versioned model artifacts, so any prediction you serve can be traced back to exactly what produced it — plus offline evaluation against held-out data before anything ships.

That versioned, evaluated model then has to reach real traffic — **serving** — packaged behind an API: a batch job if predictions aren't time-sensitive, a low-latency service if they are. Decide up front whether it needs a shadow/champion-challenger phase before it fully replaces whatever's running today (shadow = the new model runs on live traffic but its outputs are only logged, never acted on; champion-challenger = the current model keeps serving while the candidate is scored on the same traffic until it earns the switch).

Once it's serving real traffic, you need to know when it's quietly getting worse — that's **monitoring**: input drift (do incoming features look statistically different from training data), prediction drift (is the output distribution shifting), and outcome/label drift (once ground truth shows up, is accuracy actually degrading) — more on all three in the dedicated monitoring topic below.

And finally, what monitoring finds has to feed back into the start of the loop — the **feedback loop**: did the predicted failure actually happen, did the recommended action get taken and what happened after. Capturing that and folding it back into future training data is what separates a one-time deployment from a system that actually improves over time. It's also where **feedback loops can turn into self-reinforcing bias** if you're not careful — a fraud model that only ever sees labels for transactions it flagged never learns from the ones it missed.

### Summary example
A locomotive-failure system walks the full loop: requirements settle on daily batch scoring; the feature pipeline centralizes rolling-average sensor features in one definition shared by training and serving; training versions the resulting model against a held-out time-based split; serving writes daily risk scores to a table maintenance planners already use; monitoring tracks input drift on sensor distributions and outcome drift on actual failures; and the feedback loop captures outcomes for both flagged and unflagged units, specifically so the model's blind spots get corrected in the next retraining pass rather than quietly compounding.

### Where people trip up
- **Model looks great offline, falls apart in production?** That's usually training-serving skew — a feature computed slightly differently in the training pipeline (batch, with different data freshness/joins) than in the serving pipeline (real-time). Always ask "is this the exact same code path computing this feature in both places" before trusting an offline metric to predict production behavior.
- **Monitoring dashboard only checks "is the API up"?** That's treating monitoring as an ops concern instead of a modeling one. A model can be perfectly "up" — 200 status codes, low latency — while silently degrading in prediction quality as the input distribution shifts. Uptime monitoring and model-quality monitoring are different systems, and you need both.
- **Feedback loop making the model worse over time instead of better?** The loop is probably only capturing outcomes for the subset of cases the model already acts on. A model that only surfaces certain locomotives for inspection never gets ground truth on the ones it didn't flag — its blind spots never get corrected, and they compound.

<details>
<summary><strong>Self-check — answer before revealing</strong></summary>

1. Why is this framework described as a loop instead of a pipeline that ends at "serving"?
2. What's the specific term for a model looking great offline but degrading in production, and what usually causes it?
3. Name two things monitoring should track that "is the API up" completely misses.
4. Why can a feedback loop make a model worse over time instead of better?
5. What's the one design decision in the feature pipeline stage that matters most, and why?

**Answers**
1. Because monitoring and the feedback loop fold real-world outcomes back into future training data — treating it as a line that stops at serving is the single most common structural mistake in this round.
2. Training-serving skew — a feature computed slightly differently between the training pipeline (often batch) and the serving pipeline (often real-time).
3. Prediction drift (is the output distribution shifting) and outcome/label drift (is actual accuracy degrading once ground truth arrives) — a model can be fully "up" while both silently get worse.
4. If it only captures outcomes for cases the model already acts on, it never gets ground truth on what it missed — its blind spots go uncorrected and can compound.
5. Online/offline consistency — features at training time and serving time have to match bit-for-bit, or you get training-serving skew; a feature store exists specifically to enforce that.
</details>

> **Recap**
> Five stages, one loop: feature pipeline, training, serving, monitoring, feedback — closing back to the start, not stopping at serving. The single biggest failure mode is training-serving skew (a feature computed differently at train vs. serve time); the single biggest monitoring gap is watching uptime instead of prediction/outcome drift; and a feedback loop that only learns from cases the model already acted on will quietly get worse, not better.

### Where this framework comes from real work, not just theory
This five-stage loop is the same shape as two things I've actually built. At Bosch, the classification and clustering models I developed with business stakeholders — the ones that drove an 8% increase in annual revenue margins — didn't start as a modeling exercise; they started with an Azure-based ETL pipeline I engineered for the database team, because the feature pipeline (extract, load, query, transform) had to be reliable and fast (I got query turnaround from 6 minutes down to 1.8 seconds) *before* any model built on top of it could be trusted. And separately, I owned end-to-end database operations for 70 enterprise clients on Bosch's mobility cloud platform at up to 5TB scale — which is the monitoring/feedback-loop end of this framework in its purest form: sustaining 99.999% availability isn't possible without knowing the difference between "the system is up" and "the system is behaving correctly," which is exactly the input-drift-vs-outcome-drift distinction this framework insists on.

### Likely interview question + model answer
**Question:** "Design the end-to-end ML system for predicting locomotive component failures, from raw sensor data to a maintenance crew acting on it."

**Model answer (spoken flow):** "I'd start by pinning down requirements before drawing any boxes: how fresh does a prediction need to be — if sensor data streams continuously, do we need near-real-time scoring, or is a daily batch run against yesterday's data enough for maintenance planning purposes? I'd assume daily batch is workable here unless told otherwise, since maintenance scheduling isn't typically a sub-second decision.

For the feature pipeline, raw telemetry — temperature, vibration, pressure trends — gets aggregated into features like rolling averages and rate-of-change over the last N days, joined with maintenance history and component age. The one thing I'd insist on architecturally is that these feature computations live in one shared definition, ideally a feature store, so the exact same logic that ran over historical data during training also runs at serving time — I've seen training-serving skew silently tank production performance when those two paths drift apart even slightly.

Training happens offline on a versioned historical dataset with versioned code, producing a versioned model artifact, evaluated against a held-out time-based split — not a random split, since sensor data is temporal and a random split would leak future information into training. 

For serving, since this is a daily batch use case, I'd run scoring as a scheduled batch job rather than a synchronous API, writing risk scores to a table maintenance planners already consult — minimizing new tooling a crew has to learn.

Monitoring is where I'd spend real design effort, not an afterthought: input drift on the sensor feature distributions, since sensor calibration or fleet composition can shift over time; prediction drift, watching whether the fraction of units flagged high-risk suddenly jumps or drops; and outcome drift once we get ground truth — did flagged components actually fail, did unflagged ones fail unexpectedly (that second number matters as much as the first, because it's exactly where the feedback loop risk lives).

And that's the last piece — the feedback loop. I'd make sure we're capturing outcomes for units the model didn't flag as high-risk too, not just the ones it did, specifically so the model's blind spots get corrected in retraining rather than reinforced, since a model that only ever learns from the cases it already acted on can quietly get worse at exactly the failures it's missing."

---

## Designing a Real-Time Prediction System (Delay / Anomaly Prediction)

> **TL;DR**
> - Real-time adds one hard constraint batch systems don't have: a prediction must land inside a bounded time window, and that number shapes every other decision.
> - Data has to stream in (Kafka-style), and "features from the last 10 minutes" has to be computed on the fly in a windowed layer that still matches the offline training definition exactly.
> - Model complexity is a dial you trade against the latency budget, not a fixed choice.
> - The bottleneck is almost always feature computation, not model inference — and the system needs an explicit fallback for when a dependency isn't there in time.

### Plain-English explanation
A real-time system adds a hard constraint the batch framework above doesn't have: a prediction has to be ready within a bounded time window of new data arriving, which changes almost every architectural decision — feature computation, model complexity, and infrastructure all have to fit inside that latency budget.

**Visual + memory hook — the path a single event takes, and where it can blow the budget:**
```
  event arrives (GPS ping, sensor reading)
          │
          ▼
  STREAMING INGESTION (Kafka or similar)  ← no batch loading, ever
          │
          ▼
  ONLINE FEATURE COMPUTATION               ← windowed aggregation,
  ("avg speed, last 10 min")                 e.g. must match offline
          │                                   training definition
          ▼                                   bit-for-bit
  MODEL (sized to fit the latency budget)  ← heavier model = slower;
          │                                   complexity is a dial,
          ▼                                   not a fixed choice
  SERVING (warm in-memory server + cache)
          │
          ▼
  prediction, within budget ──── OR ────▶ dependency times out
                                              │
                                              ▼
                                     explicit FALLBACK
                                (cached last-known-good, safe
                                 default — never a silent crash)

  meanwhile: a fast rolling-baseline layer corrects the model's
  predictions between full retrains, so the system doesn't wait
  on a slow monthly retrain to track a fast-moving world
```

### From the latency number to what happens when a dependency fails

Everything starts with **nailing the latency budget first** — "real-time" means very different things depending on the use case: a few seconds is fine for a dispatcher-facing alert, but an automated control-loop decision might need a few hundred milliseconds, and those two numbers lead to genuinely different architectures.

Once that budget's set, data has to actually arrive fast enough to hit it — which rules out batch loading. **Streaming ingestion** means events (GPS pings, sensor readings, signal data) arrive continuously through a message queue or stream, Kafka or similar, rather than getting loaded in a scheduled job.

Given events stream in continuously, a feature like "average speed over the last 10 minutes" can't come from a batch job — it needs **online feature computation**: a streaming aggregation layer (windowed stream processing) that can serve that value with low latency. Think of it as the real-time twin of the batch feature pipeline, and it has to match the offline training definition exactly, for the same skew reasons as the framework above.

With low-latency features available, ask whether the model itself fits inside the same budget. It's a real question — **model choice under a latency budget** means a heavier model (a large ensemble, a big neural net) may simply not fit the time you've got. Model complexity is a design variable you trade against latency, not something you pick in isolation.

Once you've settled on a latency-appropriate model, the **serving infrastructure** around it needs to match: a low-latency model server (in-memory, no cold starts) behind an API, usually with a caching layer for repeat or similar queries, and horizontal scaling for load.

Even a system that serves within budget today has to handle the world changing faster than its normal retrain cadence — **concept drift under real-time constraints**. A real-time system can't wait for a slow monthly retrain if the operating environment (seasonal weather shifting delay patterns, say) moves faster than that. The fix is either more frequent retraining, or an architecture where a simple, frequently-updated component — a rolling baseline — corrects a more complex, slower-to-retrain model's predictions.

And last: every piece above assumes the model and its feature dependencies are actually available — so the design needs explicit **fallback behavior** for when they're not. What happens when a feature dependency times out at request time? A real-time system needs a tested fallback — a simpler rule, a cached last-known-good prediction, or a safe default — rather than failing open or closed by accident.

### Summary example
A delay-prediction system pins its latency budget at "a few seconds," consumes GPS/sensor events via Kafka, computes a windowed "average speed, last 10 minutes" feature in a streaming layer matching the offline definition exactly, picks a model light enough to fit that budget, serves it from an always-warm in-memory server with caching, and updates a fast rolling-baseline correction layer more often than the full model retrains to track seasonal drift — with an explicit fallback serving a cached last-known-good prediction if the streaming feature layer itself goes down, rather than crashing exactly when conditions are already unusual.

### Where I've actually operated under a real-time latency budget
FinSight, the multi-agent wealth-management platform I built, holds real-time portfolio sync under 1 second of frontend latency while running 3 LLMs across 7 agents, deployed on Azure Kubernetes Service with CI/CD — which forced exactly the tradeoffs this section describes. The naive design (call every agent fully, sequentially, on every update) does not fit a sub-1-second budget once you have a Portfolio agent, a Market agent, a Critic agent, fraud detection (Isolation Forest — deliberately a fast classical model, not an LLM call, specifically because fraud checks sit on the critical path and couldn't absorb LLM latency), and OTP-gated transaction confirmation all potentially in the loop. The real design work was deciding what had to happen synchronously in that 1-second window versus what could happen asynchronously after the user-facing update already returned — the fraud check and portfolio math had to be fast and synchronous; the fuller multi-agent debate reasoning could complete slightly after the initial sync without the user perceiving lag, as long as the UI communicated that clearly rather than silently blocking.

### Where people trip up
- **A "real-time" system's end-to-end latency blows the budget — where's it actually going?** Almost always feature computation, not model inference. Inference for most non-giant models is milliseconds; the bottleneck is fetching, joining, and aggregating the features that feed it, especially if that means a database round-trip per request.
- **Online and offline feature values disagree, even slightly?** The streaming aggregation window definition doesn't exactly match the batch training definition — "last 10 minutes" computed with slightly different boundary handling, say. This is training-serving skew's real-time-specific form, and it's worth naming as a risk before anyone asks.
- **No defined fallback, and a dependency just failed?** Failure modes weren't designed for — only the happy path was. A real-time system that silently returns no prediction (or crashes) when an upstream feature service times out will surprise operators exactly when they need the system most: during unusual conditions, which is also usually when infrastructure is under the most stress.

<details>
<summary><strong>Self-check — answer before revealing</strong></summary>

1. Why does the latency budget have to be pinned down before any other architecture decision?
2. When a real-time system's actual latency blows its budget, which step is usually the culprit — and why not the model itself?
3. What's the real-time-specific form of training-serving skew, and give a concrete example of how it happens.
4. Why can't a real-time system just wait for its normal monthly retrain to handle concept drift?
5. What should happen when an upstream feature dependency times out mid-request?

**Answers**
1. Because it constrains feature computation, model complexity, and serving infrastructure all at once — a few-second budget and a few-hundred-millisecond budget lead to genuinely different architectures, so every downstream choice depends on knowing which one you're building for.
2. Feature computation — fetching, joining, and aggregating the inputs, especially anything requiring a database round-trip per request. Model inference for most non-giant models is milliseconds, so it's rarely the bottleneck.
3. The streaming aggregation window doesn't exactly match the offline batch definition — e.g., "average speed, last 10 minutes" computed with slightly different boundary handling between the real-time and training pipelines.
4. Because the operating environment (e.g., seasonal weather affecting delay patterns) can shift faster than a monthly retrain cadence — the fix is either retraining more often or layering a fast, frequently-updated correction on top of the slower model.
5. An explicit, tested fallback fires — a simpler rule, a cached last-known-good prediction, or a safe default — rather than the system failing open or closed by accident.
</details>

> **Recap**
> Real-time design starts and ends with the latency budget: it dictates streaming ingestion, windowed online feature computation that must match training exactly, a model sized to fit the time available, and serving infrastructure with no cold starts. The bottleneck is almost always features, not inference. Concept drift needs a fast correction layer between retrains, and every dependency needs an explicit fallback rather than a silent failure.

---

## Designing a RAG System for Internal Documents

> **TL;DR**
> - This is the chunk/embed/retrieve/generate pipeline (see `core-technical-depth.md`), plus the enterprise infrastructure around it: ingestion, access control, freshness, citation.
> - The one requirement that's a security bug, not a quality bug: retrieval must filter by the requesting user's permissions *during* the search, not after.
> - Citations that map back to real, permission-checked documents are what make an answer verifiable instead of just plausible.
> - A standing eval against known question/document pairs is what tells you the system still works once the corpus hits 10,000+ documents — it isn't optional infrastructure.

### Plain-English explanation
This is a specific application of the chunking/embedding/retrieval/generation pipeline (see `core-technical-depth.md`) to a system-design context: the emphasis here is on the surrounding infrastructure decisions — document ingestion, access control, freshness, and citation — not just the retrieval algorithm itself.

**Visual + memory hook — permission metadata travels with a chunk from ingestion all the way to the answer:**
```
  DOCUMENTS (PDFs, wikis, tickets)
          │
          ▼
  INGESTION: chunk → embed → upsert         ← permissions attached
  + attach metadata (source, permissions,      HERE, at the chunk
    last-updated) to every chunk                level, not later
          │
          ▼
  VECTOR INDEX  (chunk + embedding + permission metadata, together)
          │
          ▼
  user query ──▶ RETRIEVAL, filtered by requesting        ← filter
                 user's permissions DURING the search        BEFORE/
                 (never after)                                DURING,
          │                                                   never
          ▼                                                   after
  GROUNDED GENERATION: answer only from retrieved
  context, cite which chunk backs each claim
          │
          ▼
  citations mapped back to real documents ──▶ user can verify
```

### From what the documents even are, to proving it still works at scale

Before any pipeline gets built, three questions about the documents themselves need answers — and one of them is a security concern, not a quality one. What document types are you dealing with (PDFs, wikis, ticketing systems)? How often do they change — does the index need to update within minutes of an edit, or is nightly fine? And, critically for an enterprise setting: **do different users have different access permissions to different documents**? The retrieval layer has to respect that — retrieving a chunk the requesting user isn't authorized to see is a security bug, not a quality bug.

Once permissions are a known requirement, they need to get attached to a chunk as it's first ingested. The **ingestion pipeline** — a scheduled or event-triggered job — parses new or changed documents, chunks them, computes embeddings, and upserts into the vector index, with document-level metadata (source, permissions, last-updated) attached to every chunk so it can be filtered at query time.

With that metadata attached to every chunk, the next question is *when* it actually gets enforced. **Retrieval with access control** means the query-time step has to filter candidates by the requesting user's permissions *before or during* the similarity search — never after. Filtering after the fact still means an unauthorized chunk got matched, and potentially logged or exposed, before anyone checked whether it should have been.

Given retrieval now returns only authorized chunks, the final answer needs to stay verifiable rather than merely plausible — that's **grounded generation with citations**. The prompt instructs the model to answer only from retrieved context and to cite which source chunk supports each claim; the serving layer maps those citations back to actual document links so a user can verify the answer. That's what makes the system trustworthy enough to actually rely on for decisions.

Once citations point back to real documents, ask how stale those documents can get before the index needs to catch up — that's **freshness**. Decide the acceptable staleness window and design ingestion frequency around it: real-time re-indexing on every document edit is a much bigger engineering lift than nightly batch re-indexing, and most internal-document use cases don't actually need sub-hour freshness once you ask.

And with everything above in place, the last piece is proving it still works once the corpus has grown to 10,000+ documents — the **evaluation loop**. A held-out set of real questions with known correct source documents, tracked over time, catches retrieval-quality regressions as the corpus grows and changes. This isn't optional infrastructure — it's how you actually know the system still works after the 10,000th document gets added.

### Summary example
An internal-documents RAG system for a company with role-based access requirements attaches permission metadata to every chunk at ingestion, enforces that metadata as a hard filter during retrieval rather than after, and generates answers with citations mapping back to real, permission-checked documents — refreshed on a nightly cadence since sub-hour freshness turned out not to be needed once asked, with a standing eval of known question/answer pairs confirming retrieval quality hasn't degraded as the corpus has grown past 10,000 documents.

### Where I've actually designed this system: NaviDoc
NaviDoc, the clinical RAG backend I built (FastAPI, PyTorch, PostgreSQL, MongoDB), sits in a domain — clinical/EHR-adjacent documents — where the access-control and grounded-citation requirements in this section aren't abstract, they're the entire reason the system had to be built "safety-first" rather than bolted on afterward. Clinical documents carry the same access-sensitivity BNSF's internal documents would (not every employee should retrieve every internal document, exactly as not every system should surface every patient-adjacent document to every requester), which is why I split structured/access-controlled data into PostgreSQL and the more document-shaped clinical content into MongoDB rather than putting everything in one undifferentiated store — the retrieval layer's permission filtering follows directly from how the data was already partitioned by sensitivity and structure at the storage layer, not bolted on as a query-time afterthought. I also didn't treat 35% ROUGE/BLEU as the finish line: presenting this work to actual clinicians at the Texas Health Informatics Alliance Conference and Texas Medical Center meant the grounded-citation requirement (which source passage backs which claim) had to hold up under direct questioning from domain experts, not just look good in an automated metric.

### Where people trip up
- **RAG system leaks content a user shouldn't see?** Permission filtering was probably applied after retrieval instead of as a hard constraint on the search itself. The fix is encoding access-control metadata directly in the vector index and filtering at query time, not as a post-hoc check on already-retrieved results.
- **Answer quality quietly degrading as the document corpus grows?** Nobody's re-running a retrieval-quality eval against a fixed question set over time. A system that worked well at 1,000 documents can degrade at 100,000 as more near-duplicate or contradictory chunks compete for the same top-k slots — and you won't notice without a standing eval.
- **Users don't trust the system despite it testing as "accurate"?** It probably doesn't show its sources. Citations that map back to the actual document aren't a nice-to-have UI feature — they're what lets a user (or an interviewer) verify the system rather than take it on faith, and their absence is one of the most common reasons an otherwise-good RAG system doesn't get adopted.

<details>
<summary><strong>Self-check — answer before revealing</strong></summary>

1. Why is retrieving an unauthorized chunk a security bug rather than a quality bug?
2. At what point in the pipeline does permission metadata get attached to a chunk, and why does that timing matter?
3. Why is filtering by permissions *after* retrieval not good enough, even if the filtered results never reach the user?
4. What actually makes a RAG answer "verifiable" rather than just plausible-sounding?
5. Why does a standing evaluation loop matter more as the document corpus grows, not less?

**Answers**
1. Because an unauthorized user seeing even part of a document they're not cleared for is an access-control failure, independent of whether the answer itself was well-written or accurate.
2. At ingestion — permissions get attached as chunk-level metadata (alongside source and last-updated) so they're available to filter on at query time, rather than something bolted on after the fact.
3. Because the unauthorized chunk was still matched by the search and potentially logged or exposed in the process — the harm happens at match time, not just at display time.
4. Citations that map each claim back to a real, permission-checked source document, so a user can go check the claim against the actual document rather than trusting the model's word for it.
5. Because near-duplicate or contradictory chunks increasingly compete for the same top-k slots as the corpus grows — a setup that worked well at 1,000 documents can silently degrade at 100,000, and only a standing eval against a fixed question set would catch that.
</details>

> **Recap**
> This is the chunk/embed/retrieve/generate pipeline from `core-technical-depth.md`, wrapped in enterprise requirements: permission metadata attached at ingestion, enforced as a hard filter during retrieval (never after), citations that make answers verifiable, a deliberate freshness window, and a standing eval so quality regressions get caught as the corpus grows past 10,000 documents.

---

## Designing an LLM Inference System at Scale

> **TL;DR**
> - Every lever here solves one of three problems: memory, compute, or scheduling — a strong answer walks through all three, not just the one you know best.
> - **Quantize first** (INT4 gets a 70B model's weights from 140GB to ~35GB) — it's the cheapest win, before touching infrastructure at all.
> - If the model still doesn't fit, **tensor parallelism** splits it across GPUs — a genuinely different tool from data-parallel training (DDP).
> - **Continuous batching** keeps the GPU busy across concurrent users; **PagedAttention** stops the KV cache (usually the real memory bottleneck) from fragmenting; autoscale on **queue depth**, not GPU utilization; cache **semantically**, not just exact-match; and push non-urgent work onto spot instances.

### Plain-English explanation
"How would you design the inference system?" for a large model is one of the most consistently asked GenAI system-design prompts right now — precisely because a model that answers correctly is worthless in production if it can't serve real traffic within a latency and cost budget. This tests something different from the RAG design above: not "does the answer make sense" but "can this actually run, for real concurrent users, without the GPU bill or the latency exploding." Every lever below is really solving one of three problems — memory, compute, or scheduling — and a strong answer walks through all three rather than going deep on just the one you know best.

**Visual + memory hook — the levers in the order you'd actually reach for them:**
```
  1. QUANTIZE            70B @ FP16 (140 GB) ──▶ 70B @ INT4 (~35 GB)
     (cheapest win first)      won't fit on 1 GPU      fits on 1 GPU
          │
          ▼
  2. Still doesn't fit?  TENSOR PARALLELISM — split weight matrices
     (memory problem)     across GPUs on a fast interconnect (NVLink)
          │               (different tool from DDP — DDP splits the
          ▼                BATCH, not the model itself)
  3. CONTINUOUS BATCHING — swap sequences in/out every decoding step
     (compute problem)     so the GPU never idles between tokens
          │
          ▼
  4. PAGED ATTENTION — KV cache stored in small fixed-size blocks,
     (memory problem,      allocated just-in-time, shareable across
      the real bottleneck) requests with a common prefix
          │
          ▼
  5. AUTOSCALE on QUEUE DEPTH, not raw GPU utilization
     (scheduling problem)   + a warm minimum pool (cold GPU starts
          │                   take minutes, not seconds)
          ▼
  6. SEMANTIC CACHE — embed the query, serve a cached answer on a
     (cost problem)          similarity hit, zero extra GPU cost
          │
          ▼
  7. SPOT INSTANCES for anything async/non-urgent — never for a
     (cost problem)          synchronous request a user is waiting on
```

### Memory first, then compute, then scheduling, then cost

Before touching infrastructure at all, the single cheapest win is to **quantize the model**. A 70B-parameter model at FP16 (2 bytes/parameter) needs `70B × 2 bytes ≈ 140 GB` just for the weights — already more than a single 80GB H100 has, before any KV cache or activation memory. Quantizing to INT4 (0.5 bytes/parameter) cuts that to `70B × 0.5 ≈ 35 GB`, fitting on one GPU with real room left for a KV cache. `core-technical-depth.md` already covers the PTQ/GPTQ/bitsandbytes mechanics and the quality tradeoff in detail — the point here is that quantization is step one of a serving design, not an afterthought bolted on later.

If the model still doesn't fit on one GPU even after quantizing, you split it with **tensor parallelism** — a genuinely different tool from the data parallelism (DDP) covered in `pytorch-deep-dive.md`. DDP puts a full model copy on every GPU and splits the *batch* across them, syncing gradients after each step — it doesn't help when a single model copy doesn't fit in memory to begin with. Tensor parallelism instead splits individual weight matrices *within* a layer across GPUs (each GPU holds a slice of every attention head), so every forward pass needs fast GPU-to-GPU communication at every layer — exactly why tensor-parallel serving needs GPUs on a fast interconnect (NVLink) in one node, not just "any 4 GPUs somewhere." vLLM and TensorRT-LLM implement this, so it's configured, not hand-rolled.

Once the model fits and runs, the next problem is serving many concurrent requests without the GPU sitting idle between tokens — solved by **batching requests continuously, not statically**. Processing one request at a time leaves the GPU idle most of the time waiting on token-by-token generation; batching a fixed group and waiting for the whole group to finish is better, but lets one slow request block every short one behind it. **Continuous (in-flight) batching** — vLLM's default — swaps a finished sequence out and a new request in at every single decoding step, so the batch composition changes token-by-token and the GPU never idles waiting on the slowest request in a fixed group. This is the same batching-queue mechanism visualized in `bnsf-technical-visual.html`'s Triton diagram, taken one step further: dynamic batching groups requests that arrive close together, continuous batching keeps reshuffling the batch mid-generation.

Continuous batching keeps many sequences in flight at once, which introduces its own memory cost — the **KV cache**, which you have to manage deliberately, since it's usually the real memory bottleneck, not the weights. `nca-genl`'s KV-cache section already works out the memory math (roughly 512 KB/token for a 7B-class model, scaling to tens of GB at realistic batch sizes). The problem PagedAttention (vLLM) solves: naively pre-allocating each sequence's *maximum possible* KV cache upfront wastes huge amounts of memory on sequences that end up shorter (internal fragmentation), and variable-length sequences fragment what's left into unusable gaps (external fragmentation). PagedAttention borrows the OS's own answer to this exact problem — virtual memory paging — storing the KV cache in small fixed-size non-contiguous blocks with a lookup table from logical position to physical block, so memory is allocated just-in-time and can even be *shared* across requests with a common prefix, like an identical system prompt.

With one fleet of GPUs now serving efficiently, the question becomes when to add more — **put a queue in front of the fleet and autoscale on queue depth, not raw GPU utilization**. GPU utilization can read "fine" (pegged at 100%) while requests are still backing up behind it — queue depth is the more direct signal of whether the fleet is actually keeping up with arrival rate. The GPU-specific catch: a cold GPU instance can take minutes to spin up (loading tens of GB of weights onto the device), unlike a stateless web service that starts in seconds. That's why a purely reactive "scale up once the queue grows" policy is often too slow, and production systems instead keep a warm minimum pool sized to a traffic floor and scale reactively on top of it, rather than scaling from zero.

Given the fleet now scales appropriately, you can avoid recomputing an answer for a question that's already been asked — even if not word-for-word — by **caching semantically, not just exactly**. An exact-match cache (identical string in, identical string out) misses the very common case of two users asking functionally the same question with different wording. Semantic caching embeds the incoming query (the same embedding step as `rag-deeper.md`'s retrieval pipeline) and checks similarity against recently-cached query embeddings — a hit above some similarity threshold serves the cached answer at zero additional GPU cost. The real engineering tension is the threshold itself: too loose and a wrong answer gets served to a subtly different question; too tight and the cache rarely fires — a tuned, monitored parameter, not a one-time setting.

And last: since every request that reaches the fleet now runs efficiently, ask whether there's a cheaper place to run the ones that aren't time-sensitive — **route non-critical work to spot/preemptible instances, and keep on-demand capacity for anything synchronous and user-facing.** Spot GPU capacity is typically 60–70% cheaper than on-demand, in exchange for the provider being able to reclaim it on short notice (commonly ~2 minutes' warning). That tradeoff is exactly why it belongs under async/batch workloads (nightly re-scoring, non-urgent document processing) that can tolerate a request being requeued, and never under a synchronous request a real user is actively waiting on mid-response.

### Summary example
Serving a 70B model at scale walks all seven levers in order: INT4 quantization gets the weights to ~35GB, which fits on ONE GPU — so tensor parallelism turns out not to be needed here, unlike a larger model that genuinely wouldn't fit. Continuous batching keeps that one GPU busy across many concurrent users, PagedAttention keeps their KV caches packed without fragmentation, a queue-depth-based autoscaler with a warm minimum pool adds capacity before a cold-start delay becomes visible to users, semantic caching serves repeat-ish questions for free, and any overnight non-urgent re-scoring work runs on spot instances instead of the on-demand fleet reserved for real users.

### If you haven't personally operated an LLM-serving stack at this scale
Say that plainly rather than improvising false familiarity — same advice as this doc opens with. What actually demonstrates seniority here isn't having personally run a 4×H100 vLLM cluster, it's reasoning from memory math and known tools to a coherent design under follow-up questions: deriving the FP16→INT4 numbers above from first principles, knowing DDP and tensor parallelism solve different problems (and saying which one a given bottleneck actually calls for), and being able to explain PagedAttention as a memory-fragmentation fix in OS-paging terms — that holds up better under questioning than a memorized list of tool names without the reasoning behind them.

### Where people trip up
- **Tensor parallelism proposed when the real problem is throughput, not memory?** That's the wrong parallelism strategy — it adds real cross-GPU communication overhead on every forward pass. If the model already fits on one GPU and the goal is serving more concurrent users, continuous batching and/or more independent replicas is the right lever, not splitting a model that didn't need splitting.
- **Quantization proposed without mentioning the quality tradeoff?** Worth noticing — 4-bit quantization is near-identical quality for many tasks but not free, and a strong answer names the eval step that confirms task-specific quality holds up, not just the memory-savings number.
- **KV cache left out of the memory budget entirely?** The design looks fine on paper and then OOMs in production. At realistic batch sizes and context lengths the KV cache can exceed the model weights themselves (the exact numbers `nca-genl` works out) — a memory plan that only accounts for model weights is incomplete.

<details>
<summary><strong>Self-check — answer before revealing</strong></summary>

1. A 70B model needs 140GB at FP16. What does INT4 quantization get that down to, and why does that number matter for GPU fit?
2. Why is tensor parallelism a different tool from DDP, even though both involve multiple GPUs?
3. What specifically does continuous (in-flight) batching do that static batching doesn't?
4. Why does PagedAttention borrow the concept of OS virtual memory paging specifically?
5. Why autoscale on queue depth instead of GPU utilization?
6. What's the real engineering tension in setting a semantic cache's similarity threshold?

**Answers**
1. ~35GB (`70B × 0.5 bytes`) — that fits on a single 80GB GPU with real room left over for the KV cache, versus 140GB, which doesn't fit at all.
2. DDP puts a full model copy on every GPU and splits the batch across them; tensor parallelism splits individual weight matrices within a layer across GPUs because a single copy doesn't fit in memory to begin with — they solve different problems (throughput vs. memory) and need different infrastructure (tensor parallelism needs a fast interconnect like NVLink).
3. It swaps a finished sequence out and a new request in at every single decoding step, so the GPU never idles waiting on the slowest request in a fixed group — static batching waits for the whole group to finish before starting the next one.
4. Because naive pre-allocation of each sequence's maximum possible KV cache wastes memory on sequences that end up shorter (internal fragmentation) and fragments what's left across variable-length sequences (external fragmentation) — paging allocates small fixed-size blocks just-in-time, the same fix an OS uses for the identical class of problem.
5. GPU utilization can read "pegged at 100%, fine" while requests are still backing up behind it — queue depth is the more direct signal of whether the fleet is actually keeping up with arrival rate.
6. Too loose and a wrong answer gets served to a subtly different question; too tight and the cache rarely fires — it's a tuned, monitored parameter, not a one-time setting.
</details>

> **Recap**
> Solve memory first (quantize, then tensor-parallel if it still doesn't fit), then compute (continuous batching to keep the GPU busy, PagedAttention to keep the KV cache from fragmenting), then scheduling (autoscale on queue depth with a warm pool), then cost (semantic caching, spot instances for non-urgent work). The KV cache — not the weights — is usually the real memory bottleneck once serving starts.

---

## Designing Production Model Monitoring

> **TL;DR**
> - Three kinds of drift can each fail independently — a system can look fine on two and be quietly broken on the third.
> - **Input drift** (what comes in) is fastest to detect, **prediction drift** (what comes out) is next, **outcome/label drift** (is accuracy actually degrading) is slowest but most direct — you don't get to pick just one.
> - A global average can hide severe degradation in one segment (a fleet, a region), so monitoring has to slice, not just aggregate.
> - An alert with no defined response — retrain, page someone, roll back — is just noise.

### Plain-English explanation
Monitoring an ML system means watching for three distinct kinds of drift that can each fail independently: **input drift** (the data coming in looks different from what the model was trained on), **prediction drift** (the model's output distribution is shifting), and **outcome/label drift** (once ground truth eventually arrives, the model's actual accuracy is degrading). A system can fail on any one of these while looking fine on the others.

**Visual + memory hook — three signals, ordered fast to slow, feeding one response plan:**
```
  INPUT DRIFT              PREDICTION DRIFT           OUTCOME/LABEL DRIFT
  (fastest — catches       (next-fastest — watches     (slowest, but most
   problems before          what comes OUT)             direct — waits on
   outcomes exist)                                       real ground truth)
       │                          │                            │
       ▼                          ▼                            ▼
  incoming feature          output distribution         actual accuracy
  distribution vs.          shift vs. training           (recall/precision/
  training distribution     baseline                     calibration) drop
       │                          │                            │
       └──────────────┬───────────┴──────────────┬─────────────┘
                       ▼                          │
              SLICE BY SEGMENT                    │   ← a global average
              (fleet, region, class)               │     can hide severe
                       │                            │     degradation in
                       ▼                            │     one subgroup
                  ALERT fires ◀──────────────────────┘
                       │
                       ▼
         defined RESPONSE PLAN (not silence):
         auto-retrain? page on-call? roll back?
                       │
                       ▼
         RETRAINING — scheduled, or triggered by
         a signal trusted enough to act on
```

### From the fastest signal to the slowest, then what to do about any of them

Of the three kinds of drift, **input drift** is the one you can detect fastest — before you even know if a prediction was right or wrong. Compare the statistical distribution of incoming feature values against the training distribution, per feature, using a distance or divergence metric (Population Stability Index, KL divergence, or a simple KS-test for continuous features), and alert when a feature's distribution shifts beyond a threshold. This catches problems *before* they show up in outcomes, which matters when ground truth is delayed — "did this component actually fail" might not be known for weeks.

Input drift watches what goes in; the next-fastest signal watches what comes out — **prediction drift**. Track the distribution of the model's outputs over time (what fraction of predictions are "high risk" this week vs. the training baseline). A sudden jump can mean either a genuine environmental shift or a broken upstream feature silently feeding the model garbage.

Both of those are indirect signals. The slowest but most direct signal, once it finally arrives, is **outcome/label drift**: once true outcomes are available, track the performance metrics chosen during problem formulation (recall, precision, calibration) over time on a rolling window, and alert on degradation past a defined threshold. It's the most direct signal you have — and also the slowest to arrive.

All three drift types can be computed as one global number, but that hides something important — which is why you also need **segment monitoring**. Drift and degradation can hide inside an aggregate average while being severe in one subgroup (a specific fleet, region, or locomotive class); monitoring should slice by the segments that actually matter operationally, not just report one global number.

Once all four signals above can fire an alert, what turns a firing alert into something useful rather than noise is an **alerting and response plan**. Define thresholds *before* deployment, not reactively after a problem, and define what happens when an alert fires — automatic retraining, a page to an on-call data scientist, a rollback to a previous model version. An alert with no defined response is just noise.

And given that response plan exists, the last question is whether retraining itself happens on a fixed schedule or reactively when these signals fire — that's the **retraining trigger** decision. Scheduled (monthly, say, regardless of drift) is simpler and more predictable operationally; triggered is more responsive, but it needs the monitoring signal to actually be reliable enough to trust as a trigger.

### Summary example
A locomotive failure-prediction system layers all six pieces: input drift on sensor feature distributions catches a sensor recalibration within hours; prediction drift catches a sudden jump in flagged units the same day; outcome drift confirms real accuracy degradation weeks later once failures are actually observed; segment monitoring reveals the degradation is concentrated in one older fleet, invisible in the global number; a pre-defined response plan pages an on-call data scientist rather than silently logging the alert; and because the input-drift signal is trusted enough to act on, retraining is triggered reactively rather than waiting for the next scheduled monthly run.

### Where I've actually operated production monitoring at this level of stakes
This is the part of my background most directly relevant to this section: at Bosch, I owned end-to-end database operations for 70 enterprise clients on the mobility cloud platform — replication, network security policy enforcement, data migrations for onboarding, automated housekeeping — across MongoDB, PostgreSQL, MySQL, MSSQL, and Redis at up to 5TB scale, sustaining 99.999% availability. Two incidents from that work map directly onto "monitoring catches the problem before it becomes a crisis" versus "here's what happens when it doesn't get caught early enough": I recovered a **ransomware-locked MongoDB instance** by mounting it locally and performing a full backup and restore, preserving a production client's complete dataset with zero data loss — which only worked because the situation was caught and correctly diagnosed fast, not because ransomware recovery is ever "routine." And earlier, at Cognizant on CapitalOne's banking infrastructure, I resolved a **MongoDB split-brain incident on a 6-node replica set within 30 minutes** by safely evicting the stuck secondary and resynchronizing it, keeping the primary available throughout — that's the database-infrastructure version of exactly this section's core point: the failure was caught and diagnosed (a specific node had a divergent view of cluster state) before it became a customer-facing outage, which is only possible because monitoring at the node/replication level existed and was trusted enough to act on quickly. The pattern I'd bring to an ML monitoring system is the same instinct: don't wait for the slow, expensive signal (a customer complaint, an outcome-drift number weeks later) when a faster, cheaper signal (input drift, replication lag, a node's divergent state) would have caught it first.

### Where people trip up
- **Dashboard shows everything green while users report the model is clearly wrong?** Monitoring probably only covers aggregate metrics, not the segment the complaints are coming from. A model degraded specifically on a rare locomotive class can look fine in an overall accuracy number dominated by the common classes.
- **Input drift alerts firing constantly with no corresponding real problem?** The threshold was probably set arbitrarily instead of calibrated against how much natural variation the training data already had. Some features naturally have seasonal or operational variance — the threshold needs to be set relative to that baseline noise, not a fixed generic cutoff.
- **Outcome drift is the first signal anyone notices a problem?** Input and prediction drift monitoring weren't in place. Outcome drift is real but slow — you're waiting for ground truth — and by the time it's visible, the model may have been making degraded decisions for weeks; input/prediction drift monitoring exists specifically to catch problems earlier, before the slow, expensive signal arrives.

<details>
<summary><strong>Self-check — answer before revealing</strong></summary>

1. Rank the three drift types from fastest to slowest to detect, and explain why outcome drift is slowest.
2. Why can a system pass every drift check on the global number and still be badly broken?
3. What's wrong with an alert that has no defined response plan attached to it?
4. When is a scheduled retraining cadence preferable to a signal-triggered one, and vice versa?
5. Give a concrete example of prediction drift that isn't actually a real environmental shift.

**Answers**
1. Input drift, then prediction drift, then outcome/label drift. Outcome drift is slowest because it requires waiting for real-world ground truth to arrive — sometimes weeks, as in "did this component actually fail."
2. Because degradation can be severe in one subgroup (a fleet, a region, a rare class) while being invisible in an aggregate average dominated by the common cases — monitoring has to slice by the segments that matter operationally.
3. It's just noise — nobody knows whether it should trigger automatic retraining, a page to an on-call data scientist, or a rollback, so in practice nothing happens when it fires.
4. Scheduled is simpler and more predictable operationally, and is the safer default when you don't yet trust the monitoring signal enough to act on it directly; triggered is more responsive, but only makes sense once the triggering signal has proven reliable.
5. A sudden jump in the fraction of "high risk" predictions caused by a broken upstream feature silently feeding the model garbage — the output distribution shifted, but nothing about the real world actually changed.
</details>

> **Recap**
> Three drift signals, fastest to slowest: input (what comes in), prediction (what comes out), outcome (is accuracy actually degrading). Slice every one of them by operational segment, not just a global average, or you'll miss degradation hiding inside an aggregate. Every alert needs a pre-defined response, and retraining is either scheduled or signal-triggered — triggered only once the signal is trustworthy enough to act on.

---

## Designing a Search + LLM Product (Perplexity/You.com-style)

> **TL;DR**
> - This is RAG pointed at the live web instead of a curated corpus — access control disappears, and a source-quality problem takes its place.
> - Search is a decision, not a default: a router sorts queries into no-search / search-required / ambiguous, biased toward searching, because a skipped search produces a confident, unciteable hallucination — the worst failure this product has.
> - Citations have to attach to a specific **claim span**, verified post-hoc against the source passage — not just a list of links at the bottom.
> - Stream the answer and resolve citations behind it, because the metric that matters is time-to-first-token, not total pipeline time; cache with a TTL tied to how volatile the query actually is.

### Plain-English explanation
This is RAG pointed at the live web instead of a curated corpus — which inverts almost every assumption the internal-documents design above relied on: the corpus is unbounded, unvetted, self-contradictory, and changing underneath you, and the user compares your latency against a plain search-results page rather than against a chatbot.

**Visual + memory hook — the pipeline, with the stream starting long before the pipeline finishes:**
```
  query ──▶ ROUTER: no-search / search-required / ambiguous
                     (biased toward searching — a missed search is
                      the worst failure; ambiguous → search)
                        │
                        ▼
            RETRIEVE (search API or own crawler)
            fetch top N pages in parallel, strip boilerplate
                        │
                        ▼
            RE-RANK passages with a cross-encoder
            (web ranks for clicks, not for grounding quality)
                        │
                        ▼
   ┌────────────────────┴────────────────────┐
   ▼                                          ▼
GENERATION starts streaming              next batch of pages
immediately, inline citation markers     fetched + re-ranked
render optimistically                    IN PARALLEL
   │
   ▼
each completed SENTENCE → async entailment check
("does passage k actually support this span?")
   │
   ├─ supported → marker settles solid, gains source favicon
   └─ unsupported → marker visibly retracted
   │
   ▼
CONFLICTING SOURCES? → explicit policy (recency / authority /
consensus / surface the disagreement) — never silently pick one
   │
   ▼
CACHE — two layers, TTL set by query volatility:
  retrieval result (expensive, reusable across rewordings)
  answer text (cheap, tied to exact wording + volatility class)
```

### From deciding whether to search at all, to serving a repeat query for free

Before any retrieval happens, you have to decide whether the query even needs the web — **search is a decision, not a default**. A lightweight classifier (or the model's own tool-call decision) sorts queries into three buckets: **no-search-needed** (arithmetic, translation, "rewrite this paragraph," stable world knowledge), **search-required** (anything time-sensitive, entity-specific, numeric, or otherwise verifiable — "who won last night," "current price of X," "what changed in version 3.2"), and **ambiguous**. The two error types here are wildly asymmetric: a **false negative** (skipping search when it was needed) produces a confident, unciteable hallucination — the worst failure this product has — while a **false positive** (searching when it wasn't needed) just costs a search-API call and a few hundred milliseconds, and the answer is usually still correct. So the threshold should be deliberately biased toward searching, and the ambiguous bucket should route to search, not away from it.

Once a query is routed as search-required, what does retrieval actually feed the model — and how does that differ from the curated corpus in the internal-documents design above? **The grounding context is live web content, and that difference is the whole section.** In `Designing a RAG System for Internal Documents`, the corpus was static, curated, and access-controlled — you owned every document, chunked and embedded it yourself, and the hard requirement was permission filtering, because retrieving a chunk the user isn't authorized to see was a security bug. Here, none of that holds: everything retrieved is public, so **the access-control problem disappears entirely — and is replaced by a source-quality problem that didn't exist there at all.** You can't pre-embed the web, so retrieval is a two-stage fetch at query time: either call a commercial search API (Bing/Brave/Google CSE — fast to build, no crawl infrastructure, but you inherit that engine's ranking and pay per query) or run your own crawler and index (expensive and slow to build, but you control ranking, freshness, and per-domain quality priors, which is the actual moat in this product). Either way, you fetch the top N results, strip boilerplate (nav, ads, cookie banners), chunk, and **re-rank the passages against the query with a cross-encoder** (`rag-deeper.md`'s two-stage retrieval pattern) — because web search ranks for *clicks*, not for *being good grounding context*, and the #1 SEO result is often a content-farm listicle with less substance than the #7 primary source.

With re-ranked web passages in the prompt as grounding context, how does a specific claim in the answer get tied to a specific source, rather than a list of links at the bottom? That's **span-level attribution — the unit is a claim, not a document.** Three mechanisms, usually layered: (a) **generation-time citation** — number the passages in the prompt and instruct the model to emit an inline marker after every sentence it draws from one; cheap, but the model can and does mis-cite, attaching `[2]` to a sentence passage 2 doesn't actually support. (b) **Post-hoc verification** — after generation, split the answer into sentence-level claim spans (with character offsets into the answer text, so the UI can highlight exactly which words a citation covers) and run an **NLI/entailment model or cross-encoder** asking "does passage *k* entail this span?", dropping or flagging any span nothing entails. (c) **Hybrid**, which is what you'd actually ship: generate with inline markers, then verify each marker with the entailment check, silently correcting a marker that points at the wrong passage and visibly demoting a span nothing supports. The reason to insist on character-offset spans rather than document-level footnotes isn't UI polish — spans are the only unit you can *measure*: attribution precision (what fraction of cited spans are actually entailed by the source they cite) is a real number only if a citation has a defined scope.

Given post-hoc entailment can confirm a span against one passage, what happens when two retrieved passages entail contradictory claims? **Conflicting sources need an explicit, stated resolution policy — silently picking one is the failure mode.** Detection is the easy half: the same entailment machinery flags it when passage A entails a span and passage B entails its negation, and cheap heuristics catch most of the rest (two different numbers or dates asserted for the same entity+attribute). The resolution policy is a product decision, and a strong answer names which one it's choosing and why: **recency preference** works for facts that genuinely change (prices, standings, "current CEO") but depends on publication timestamps, which are notoriously unreliable and frequently re-stamped on the web; **source-authority priors** (a curated or learned per-domain quality score — primary source > established outlet > aggregator > content farm) is the most robust and also the most work; **consensus** (n-of-m agreement) is tempting and dangerous, because web content is heavily syndicated and copy-pasted, so without near-duplicate detection by content hash you will count one wire story republished five times as five independent confirmations. The fourth option is the honest one: for a genuinely contested fact, **surface the disagreement** ("Source A reports X as of March; Source B reports Y") rather than resolving it — the other three policies all manufacture confidence the evidence doesn't support.

Classification, retrieval, re-ranking, generation, and entailment verification all sit between the keystroke and the answer — so how does that fit a latency budget users measure against a plain search-results page? **Streaming changes what the budget is even measuring — optimize time-to-first-token, not total time.** Priced serially, the pipeline is brutal: classification ~50 ms, search API 200-500 ms, fetching and boilerplate-stripping N pages 300-800 ms (fetch in parallel with a hard per-page timeout and drop the stragglers — never let the slowest page in the batch gate the answer), cross-encoder re-ranking 50-150 ms, then multi-paragraph generation measured in seconds, then entailment verification on top of that. Run end-to-end and serially, that's 5-10 seconds before the user sees a pixel, against a competitor that paints results in under a second. The fix is structural, not micro-optimization: **start generating as soon as the first re-ranked passages land, stream tokens immediately, and resolve citations behind the stream** — render inline markers optimistically as the model emits them, run the entailment check asynchronously per completed sentence, and let each marker settle a beat later (turning solid, gaining a source favicon, or being visibly retracted). Overlap the rest too: fetch and re-rank the next batch of pages while the first batch is already generating. The user perceives median time-to-first-token; total time only becomes visible if the stream stalls mid-answer.

Given the answer streams while citations resolve behind it, how do you avoid paying that whole pipeline cost again for a query someone already asked? **Cache in two layers, with a TTL derived from the query's volatility — not one global number.** The semantic-cache mechanism is the same one described for LLM inference above: embed the incoming query, serve a cached result on a similarity hit. What's new here is a **freshness dimension the static-corpus case never needed**: "boiling point of water" can be cached indefinitely, "who won last night" should have a TTL of minutes or not be cached at all — and the router above can emit a volatility class alongside its search/no-search decision. Cache the two halves separately: the **retrieval result** (search results plus fetched, stripped passages) is the expensive, rate-limited, per-query-billed part and is often still valid for a differently-worded question whose *answer text* isn't reusable, so it deserves its own longer-lived cache below the answer cache. And **stale-while-revalidate** — serve the cached answer instantly, kick off a background refresh so the next user gets fresh — is right for high-volume head queries and wrong for anything where staleness is a correctness bug rather than a mild annoyance (prices, medical or safety information, anything a user will act on).

And since every layer above can fail independently, how do you actually know the product is good? **Evaluate the layers separately, because a single end-to-end score can't tell you which one broke.** A held-out query set with known-correct answers *and* known-good source documents lets you score: routing accuracy (did search fire when it should have), retrieval quality (did a passage that supports the correct answer make it into the top-k after re-ranking), **attribution precision and recall** (what fraction of cited spans are genuinely entailed, and what fraction of factual spans carry any citation at all — the second one catches the model quietly asserting things with no source), and answer correctness. `rag-deeper.md`'s RAGAS-style per-station scoring is the same idea, and the diagnostic payoff is identical: bad context precision means fix retrieval, bad faithfulness means fix the generation prompt or the verifier, not retrieval. The eval discipline this product needs that a static-corpus RAG doesn't: **the eval set decays**, because the web moves — an answer that was correct when the set was written can be genuinely wrong six months later, so eval queries need periodic re-validation, and a drop in the score is as likely to mean "the world changed" as "the system regressed."

### Summary example
A user types "what's the latest on the Fed's rate decision." The router classifies it as search-required and high-volatility; retrieval hits a search API, fetches and strips the top 10 pages in parallel with a 600 ms per-page timeout, and cross-encoder re-ranks the passages so a primary-source statement outranks an SEO-optimized summary. Generation begins as soon as the top passages land and streams immediately, emitting inline markers the UI renders optimistically. Behind the stream, each completed sentence goes to an entailment check; one sentence citing a stale blog post fails and its marker is retracted rather than left standing. Two sources disagree on the effective date, so instead of silently preferring the more recent timestamp, the answer surfaces both attributions explicitly. The whole result is cached with a short TTL because the volatility class says "hours, not weeks," while the fetched passages are cached longer under their own key — and the standing eval set confirms attribution precision hasn't slipped, with the caveat that a failing eval query gets re-validated against the current web before anyone calls it a regression.

### Where I've actually built the citation half of this, and where I honestly haven't
The **grounded-citation** half of this design is real work I've done, and it's worth saying precisely which half. NaviDoc's citation-mapping module maps a generated claim back to the specific source passage that supports it — not a list of documents at the bottom — and that requirement existed because the system had to be defensible under direct questioning from clinicians at the Texas Health Informatics Alliance Conference and Texas Medical Center, where a plausible-but-unsupported claim isn't a quality miss. Separately, my UNT research assistantship is specifically LLM hallucination mitigation using RAG to ground responses against scientific literature, holding 20-second end-to-end retrieval from complex medical documents — which is exactly the retrieval-latency-versus-grounding-rigor tension the streaming discussion above is about, just with a much more forgiving budget than a consumer search product's. What I have **not** built is the live-web half: a curated clinical corpus and scientific literature are vetted, stable, and internally consistent in a way the open web simply isn't, so the source-quality priors, conflict-resolution policy, and volatility-based caching described above are design reasoning on my part, not operating experience — and I'd say that plainly rather than imply otherwise.

### Where people trip up
- **Product cites sources that don't actually support the sentence they're attached to?** Citations were probably produced by the generating model rather than verified against the passage — the model is optimizing for a plausible-looking marker, not a true one. A post-hoc entailment check per claim span is the fix, and it's also what makes attribution precision measurable at all.
- **Answer states one side of a genuinely disputed fact with full confidence?** The pipeline resolved a source conflict silently — usually by taking whichever passage ranked first or was timestamped latest. The fix is an explicit conflict policy with "surface the disagreement" as a real option, not a fallback nobody implemented.
- **p50 latency looks acceptable on a dashboard but users still call the product slow?** The budget was probably measured as total end-to-end time while the user actually experiences time-to-first-token. Stream generation before verification finishes and resolve citations behind the stream, rather than trying to shave milliseconds off a serial pipeline.
- **Cache makes the product fast but occasionally wrong about current events?** One global TTL was probably applied across every query type. Volatility is a property of the query, and the classifier that already decides search-or-not is the natural place to emit it.
- **Evaluation only scores answer quality and never attribution?** The eval was probably inherited from a static-corpus RAG setup where sources weren't the product. Here the citation *is* the product, so attribution precision and recall belong in the eval set as first-class metrics.

<details>
<summary><strong>Self-check — answer before revealing</strong></summary>

1. Why should the routing threshold be biased toward searching rather than balanced between the two error types?
2. What problem does the internal-documents RAG design have that this one doesn't — and what problem replaces it?
3. Why does span-level, character-offset attribution matter more than document-level footnotes?
4. Name the four ways this design can handle two sources that contradict each other, and which one is "the honest one."
5. What's the structural fix for a 5-10 second serial pipeline, and which latency metric does it actually improve?
6. Why does the eval set for this product need periodic re-validation in a way a static-corpus RAG eval set doesn't?

**Answers**
1. Because the two error types are wildly asymmetric — skipping search when it was needed produces a confident, unciteable hallucination (the worst failure this product has), while searching unnecessarily just costs a bit of latency and money and the answer is usually still correct anyway.
2. The internal-documents design has a permission-filtering / access-control requirement that disappears here, since everything on the open web is already public. It's replaced by a source-quality problem: the web is unvetted, self-contradictory, heavily syndicated, and ranked for clicks rather than for good grounding.
3. Because spans are the only unit you can actually measure — attribution precision (what fraction of cited spans are genuinely entailed by their source) is only a real number if each citation has a defined scope, which a document-level footnote doesn't give you.
4. Recency preference, source-authority priors, consensus (n-of-m agreement), and surfacing the disagreement explicitly instead of resolving it. Surfacing the disagreement is the honest one — the other three all manufacture confidence the evidence doesn't actually support.
5. Start generating as soon as the first re-ranked passages land, stream tokens immediately, and resolve citations asynchronously behind the stream. It improves time-to-first-token, not total pipeline time — total time only becomes visible again if the stream stalls mid-answer.
6. Because the web moves — an answer that was correct when the eval set was written can be genuinely wrong six months later, so a drop in score can mean "the world changed" just as easily as "the system regressed," and only periodic re-validation can tell those apart.
</details>

> **Recap**
> Search is a routed decision, biased toward searching, because a missed search is the worst failure. Retrieval hits the live web instead of a curated corpus, trading the access-control problem for a source-quality one, so passages get cross-encoder re-ranked before they ground anything. Citations attach to character-offset claim spans, verified post-hoc by an entailment check, with an explicit policy for when sources disagree. Stream the answer and resolve citations behind it to win on time-to-first-token, cache with a volatility-aware TTL, and evaluate every layer separately since one end-to-end score can't tell you which one broke.

---

## Designing an Autonomous Web-Browsing Agent

> **TL;DR**
> - A browsing agent is a model looped against a live, adversarial, stateful environment it can permanently change — the hard problems are perception, constraining the action space, and proving it actually finished.
> - Give it a **hybrid view** (accessibility tree elements numbered onto a screenshot) so it acts on stable, auditable handles instead of raw pixels.
> - The safety guarantee against irreversible actions (purchases, deletions) has to live in the **tool layer** — a missing tool, not a prompt telling it to behave — because page content is untrusted input and can talk to the agent directly.
> - `done()` is self-report, not evaluation: grade the environment state, and watch for **environment drift** (the web changing under you) as the reason a healthy agent's success rate can fall with no code change.

### Plain-English explanation
A browsing agent is a model in a loop with a live, adversarial, stateful environment it can permanently change — so the hard design problems aren't prompting, they're how faithfully the agent perceives a page, how tightly the action space is constrained against irreversible mistakes, and how you prove it actually finished the task rather than merely stopped.

**Visual + memory hook — one turn of the loop, and where the guardrails sit:**
```
  ┌─▶ OBSERVE page (hybrid: accessibility tree, numbered
  │            onto a screenshot — stable, auditable handles)
  │             │
  │             ▼
  │   REASON (what's the next step toward the goal?)
  │             │
  │             ▼
  │   ACT — but only from a small, deliberately incomplete
  │         action space: reversible actions (click, scroll,
  │         navigate, read) are freely available; irreversible
  │         ones (submit, purchase, delete) require a tool
  │         that doesn't exist, or a human confirmation gate
  │             │
  │             ▼
  │   OBSERVE the result ──▶ stuck? (repetition / no state
  │             │              change / cycling) → escalate:
  └─────────────┘              hint → ask_human → abort w/ report
                │
                ▼
        done() / ask_human / abort
                │
                ▼
   GRADE THE ENVIRONMENT, not the agent's self-report:
   programmatic validator → LLM-judge (checked against humans)
   → stratified human review
```

### From one turn of the loop, to constraining what it can do, to proving it actually finished

Before any question about models or pages, it helps to name the actual runtime structure: a browsing agent runs **a ReAct-style loop** — observe page state, reason, act, observe the result, repeat, until a stop condition fires. (`prompt-engineering-deeper.md` and `langgraph-practice.md` cover ReAct as a prompting/orchestration pattern; this section is what it takes to point one at a real browser.) Each iteration appends `(observation, thought, action, result)` to a trajectory. Two consequences fall out of that structure immediately and shape everything downstream. First, **the state lives in the browser, not the model** — cookies, session, scroll position, open modals, and the DOM are all real state the model cannot see except through whatever you serialize into its context, which makes this a *partially observable* environment where two different real states can produce identical observations. Second, **the trajectory grows**: a 40-step run with full page dumps blows past any context window, so older observations get truncated or summarized — which means the agent can genuinely forget what it already tried, and re-try it, unless you keep an explicit compacted memory of attempted actions and their outcomes separate from the raw trajectory.

Since each turn opens with an "observe the page" step, what does the agent actually see — and what's the tradeoff between the options? **Rendered screenshot plus a vision model, versus the parsed DOM / accessibility tree — and in practice you want both.** A **screenshot** gives the agent what a human sees: canvas-rendered apps, charts, images, actual visual layout, whether an element is genuinely visible or covered by an overlay, and custom widgets built out of non-semantic `div`s. Its costs are real: a full-page screenshot at legible resolution runs on the order of a thousand-plus image tokens *per step*, which at 30 steps dominates your cost; grounding a click to exact pixel coordinates is error-prone; and small or low-contrast text is destroyed by downscaling. The **DOM / accessibility tree** is cheap, precise, and — the underrated part — gives you **stable element handles**, so an action is "click element 47" rather than "click at (812, 431)." Its costs: a modern page's DOM is enormous and mostly noise, so you must filter aggressively to interactive-and-visible elements; anything conveyed only visually is invisible to it; and heavy JS apps frequently produce accessibility trees that are simply wrong. The practical design is **hybrid, in the set-of-marks style**: extract the accessibility tree, filter to interactive elements, draw numbered boxes over them on the screenshot, and let the model reason visually while *acting* on a numbered handle. The decisive argument for handles over coordinates isn't accuracy alone — **a handle is auditable**: you can log exactly which element was clicked, replay the trajectory deterministically, and diff two runs, none of which is possible with a pixel pair.

Given the agent acts on numbered element handles rather than raw pixels, what's the action space — and which actions must not be freely available? **The action space is small and deliberately incomplete, and the safety property lives in the tool layer, not the prompt.** A reasonable set: `click(id)`, `type(id, text)`, `scroll(direction)`, `navigate(url)`, `extract_text(region)`, `go_back()`, `wait()`, `ask_human(question)`, `done(result)`. The design question is which actions are *withheld*, and the organizing principle is **reversibility, not action name**: reading, scrolling, and navigating are reversible; anything that mutates external state — submitting a form, sending a message, completing a purchase, deleting something, granting an OAuth consent — is not. Defense in depth, in order of how much you should trust each layer:
- **Capability allowlist at the tool layer.** If the agent shouldn't be able to purchase, it should not be given a tool that can. This is not the same as telling it not to. **A prompt is a request; a missing tool is a guarantee** — and on the open web that distinction is load-bearing, because **page content is untrusted input**: a page can contain text addressed to the agent instructing it to do something (prompt injection), so any safety property that exists only in the system prompt is defeatable by the very environment the agent is being asked to read.
- **Environment sandboxing.** A fresh, isolated browser profile with no logged-in sessions, no saved payment methods, and a domain allowlist. The agent can't buy what it has no credentials for — that's a structural guarantee rather than a behavioral one.
- **Human confirmation gate on every irreversible action**, and the gate has to show the *rendered consequence* — what exactly will be submitted, to whom, for how much — not "approve action 12," which trains the reviewer to click yes.
- **Hard budgets**: max steps, max navigations, wall-clock timeout, and a spend cap defaulting to zero.
- **Read-only mode as the default for evaluation runs**, so measuring the agent never risks acting through it.

The answer to avoid in this round is "the agent should be careful" or "we'd prompt it not to" — that isn't a design, and an interviewer asking this question is specifically checking whether you put the guarantee somewhere the model can't talk its way past.

With irreversible actions gated and every action spending from a step budget, how do you detect the agent is stuck in a loop rather than making progress? **Layer cheap structural checks under one expensive semantic one, and treat budget exhaustion as the backstop, not the detector.** In increasing cost: **exact repetition** — hash `(URL, action, target element)` and stop when the same action fires on the same page state N times, which catches the single most common failure (repeatedly clicking a control that does nothing). **State non-progress** — hash the normalized accessibility tree; if the observation is unchanged after K actions, the agent is acting into a void. **Cyclic navigation** — the URL/state sequence revisits an already-seen state, the A→B→A→B pattern. **Semantic non-progress** — a critic pass or LLM-judge over the last M steps asking whether the trajectory has moved closer to the goal; more expensive, but it's the only one that catches an agent doing *different* useless things every step, which none of the hash checks will ever fire on. What happens *on* detection matters as much as detecting it: **escalate rather than die**. Inject a hint and let it try a different strategy; failing that, call `ask_human` with the specific blocker and the trajectory so far; failing that, abort with a structured failure report naming where it got stuck. **Aborting with an explanation is a successful outcome for the system even though the task failed** — silently looping until the wall-clock timeout is not, because it burns the budget and produces nothing anyone can debug.

Once the agent terminates via `done()`, `ask_human`, or a stuck-detection abort, how do you know the task was actually completed correctly? **"It stopped" isn't success, and `done()` is self-reported — grade the environment, not the agent's opinion of itself.** Three tiers: **programmatic state-based validators** wherever possible — assert the final environment state directly (the booking exists, the downloaded file hashes correctly, the extracted value matches an answer key). This is exactly how browsing benchmarks work: WebArena-style suites pair each task with a programmatic validator, and WebVoyager/Mind2Web-style sets pair tasks with reference answers or trajectories. Report **task success rate sliced by task category**, never just the aggregate — an agent at 80% on search-and-read and 5% on multi-step form filling averages to a number that describes neither. **LLM-judge over the trajectory plus final screenshot** for tasks with no programmatic check ("find the cheapest flight under these constraints") — scalable and cheap, but you must **measure the judge's agreement with human raters on a sample before trusting its number**, or you've swapped an unmeasured metric for a differently-unmeasured one. **Human review sampling** — a stratified sample of trajectories reviewed every release, deliberately over-weighted toward runs the judge scored as ambiguous and runs that requested a gated action, because that's where the failure modes you haven't named yet are hiding. And measure what success rate can't see: steps-to-completion (efficiency), cost per *completed* task (not per run — a cheap agent that fails is not cheap), gated-action request rate, and **harmful-action-attempt rate**: how often the agent tried something a guardrail blocked. That number should be zero **by measurement, never by assumption** — if you aren't logging blocked attempts, you don't know whether your guardrails are load-bearing or decorative.

Given a measured task success rate now exists, what makes that number degrade with nobody shipping a change to the agent? **Environment drift — the web is the agent's runtime, and it changes underneath you.** A site redesigns and every element handle the agent learned to expect is wrong; an A/B test serves a different layout to half your runs; a cookie-consent modal appears in one region and not another; bot detection starts firing CAPTCHAs it didn't fire last month. This is the browsing-agent form of the **data drift** in `production-ml-practice.md` — nothing about the model or the prompt changed, the world did — and it's precisely why the monitoring section above insists on segment-level rather than aggregate signals. Concretely: run the benchmark suite **continuously on a schedule, not only at release**; alert on **per-site and per-task-category success-rate drops** rather than the global average, which will absorb a single site breaking completely; and maintain a **pinned, archived snapshot environment** alongside the live one, so when the number falls you can tell "the agent regressed" apart from "the site changed" — without that control, every drop turns into an argument nobody can settle.

### Summary example
An agent tasked with "find the current warranty terms for part number X on the manufacturer's site and summarize them" runs the ReAct loop with a hybrid observation: accessibility tree filtered to interactive elements, numbered onto a screenshot. It navigates, types the part number into a search box, and clicks through — all reversible actions from an allowlist that simply does not include form submission or purchase, in a sandboxed profile with no saved credentials. On step 9 it clicks a "Details" control three times with an unchanged observation hash; repetition detection fires, a strategy hint is injected, and when that also fails it calls `ask_human` with the specific blocker rather than burning the remaining 30 steps. On a successful run, grading doesn't take `done()` at face value: a programmatic check confirms the extracted warranty duration matches the answer key, the run is logged with steps-taken and zero blocked-action attempts, and it lands in the stratified sample for human review. Six weeks later the aggregate success rate is flat but the per-site slice for this manufacturer drops from 90% to 40% — the pinned snapshot environment still passes, which immediately identifies a site redesign rather than an agent regression.

### Where the irreversible-action gate in this design is something I've actually built
I have not built an autonomous web-browsing agent, and I'd say so directly rather than stretch a multi-agent project into one. But **the core safety idea above — a human confirmation gate standing between an agentic system and an irreversible real-world action — is a design decision I've actually shipped**: FinSight's transaction path is OTP-gated, meaning the agents can propose and reason about a trade all they want, but the irreversible step requires an out-of-band human confirmation that no amount of model output can satisfy on its own. That's the same structural principle as withholding the `purchase` tool from a browsing agent — the guarantee lives outside the model, in a layer the model cannot argue past — and it sat alongside a deliberate reversibility distinction in the same system, where the fast, reversible work (portfolio math, the Isolation Forest fraud check) ran freely on the synchronous path while the consequential step was gated. What I haven't done is the perception and stuck-detection work described above; FinSight's agents act on structured internal APIs, not on an adversarial rendered webpage, which is a genuinely easier problem and I wouldn't claim otherwise.

### Where people trip up
- **Agent confidently reports success on a task it didn't complete?** Success was measured by the agent's own `done()` call. Self-report is not evaluation — a programmatic state validator, or a judge whose agreement with human raters has actually been measured, is the fix.
- **Agent occasionally takes a destructive action despite a system prompt forbidding it?** The prohibition lived in the prompt instead of the tool layer. Page content is untrusted input and can address the agent directly, so the fix is removing the capability, not strengthening the instruction.
- **Agent burns 40 steps and times out on a task a human finishes in 4?** The only stop condition is the budget backstop, with no non-progress detection underneath it. Hash the observation and the `(URL, action, element)` tuple and stop on repetition, then escalate to `ask_human` rather than looping.
- **Aggregate success rate looks healthy while users report constant failures?** The metric isn't sliced by site and task category. One completely broken site or one hard task type disappears into a global average, exactly as the monitoring section's segment argument predicts.
- **Screenshot-only agent misclicks constantly?** Actions were grounded in pixel coordinates rather than element handles. Set-of-marks numbering over accessibility-tree elements fixes the accuracy problem and, just as importantly, makes trajectories auditable and replayable.

<details>
<summary><strong>Self-check — answer before revealing</strong></summary>

1. Why is a browsing agent's environment described as "partially observable," and what design consequence does that have?
2. Why is a numbered element handle better grounding for an action than raw pixel coordinates, beyond just accuracy?
3. Why must the guarantee against an irreversible action live in the tool layer instead of the system prompt?
4. Name the four escalating tiers of stuck-loop detection, from cheapest to most expensive.
5. Why isn't `done()` treated as sufficient evidence a task succeeded, and what replaces it?
6. What's "environment drift" in this context, and why does it require per-site monitoring rather than an aggregate success rate?

**Answers**
1. Because cookies, session, scroll position, open modals, and the DOM are all real browser state the model can't see except through whatever gets serialized into its context — two genuinely different states can produce identical observations, so the agent needs an explicit compacted memory of what it already tried.
2. A handle is auditable: you can log exactly which element was clicked, replay the trajectory deterministically, and diff two runs — none of which is possible with a pixel pair.
3. Because page content is untrusted input — a page can contain text addressed to the agent instructing it to do something (prompt injection), so any safety property that exists only in the prompt is defeatable by the very environment the agent reads. A missing tool is a guarantee; a prompt is only a request.
4. Exact repetition (same action on the same page state), state non-progress (observation unchanged after K actions), cyclic navigation (A→B→A→B), and semantic non-progress (an LLM-judge asking whether the trajectory is actually moving toward the goal).
5. `done()` is self-reported by the agent, not verified. It's replaced by programmatic state-based validators where possible, an LLM-judge (checked against human agreement) where there's no programmatic check, and stratified human review sampling weighted toward ambiguous or gated-action runs.
6. Environment drift is the web itself changing underneath the agent — a site redesign, an A/B test, a new CAPTCHA — with no change to the model or prompt. It needs per-site and per-task-category monitoring because a single broken site disappears into a healthy-looking global average.
</details>

> **Recap**
> A browsing agent is a ReAct loop pointed at a live, adversarial environment: observe (hybrid screenshot + accessibility-tree handles) → reason → act (from a small, reversibility-gated action space) → repeat until a stop condition. The safety guarantee against irreversible actions lives in the tool layer, never the prompt. Stuck-loop detection escalates from cheap structural checks to an expensive semantic one, and escalates to a human rather than dying silently. Success is graded against the environment, not the agent's self-report, and a healthy success rate can still fall from environment drift — the web changing under the agent with no code change on your end.

---

## Designing an Evaluation Framework for a Customer-Support Chatbot Replacing Human Agents

> **TL;DR**
> - The business is buying whether the user's problem got solved, not whether the response text is fluent — so **resolution rate** (not BLEU/ROUGE/thumbs-up) is the north star, defined with a re-contact clause so "conversation ended" can't quietly count as "problem solved."
> - Escalation should be a layered decision (retrieval confidence, generation confidence, hard policy rules, explicit "talk to a person") — not "the model's confidence" alone — calibrated per intent, not as one global threshold.
> - Offline eval (replay historical tickets, LLM-judge checked against humans) is a shipping gate; only an online A/B test against a human-only baseline, with resolution rate as primary and everything else as a guardrail, measures the real thing.
> - The bot only gets ground truth from cases it escalates — so its confident wrong answers are invisible unless you deliberately sample non-escalated conversations for review too.

### Plain-English explanation
Every convenient metric here measures the bot's *output* — fluency, faithfulness, response quality — while the thing the business is actually buying is whether the user's problem got solved, which is a business outcome rather than an NLG score. The design tension is that a bot can max out every output metric while resolving nothing, and the metric that catches that is also the one that takes the longest to observe.

**Visual + memory hook — the decision that gates every contact, and the eval layers around it:**
```
  incoming contact
        │
        ▼
  ESCALATION DECISION (layered, in order of usefulness):
    retrieval confidence (before generation, strongest signal)
    → generation confidence (self-consistency, verifier pass)
    → hard policy table (refunds, legal, safety — always human)
    → explicit "talk to a person" (honored immediately, no negotiating)
        │
   ┌────┴────┐
   ▼         ▼
 bot        human
 answers    handles
   │
   ▼
 RESOLVED?  (no human touch + no re-contact in 7 days
             + no new ticket on the same problem)

  ── before launch ──          ── after launch ──
  OFFLINE EVAL                 ONLINE A/B TEST
  replay historical tickets    vs. permanent human-only holdout
  LLM-judge vs. rubric,        resolution rate = primary metric
  checked against humans       CSAT/escalation/re-contact = guardrails
        │                            │
        └──────────┬─────────────────┘
                    ▼
       POST-LAUNCH DRIFT WATCH (fastest → slowest):
       input clustering → retrieval-score drift → prediction
       drift → outcome/re-contact drift → standing canary set

  feedback loop: escalated cases → free labeled data
                 (but the bot's CONFIDENT wrong answers generate
                  nothing unless non-escalated convos are sampled too)
```

### From defining "good," through escalation and offline/online eval, to catching it going quietly bad

Before choosing any metric, it's worth pinning down what "good" actually means for a bot that's replacing human agents. **Resolution rate is the north star, and it has to be defined operationally or it will be gamed by accident.** Not BLEU, not ROUGE, not a thumbs-up on the individual response — those measure whether text was well-formed, and a well-formed wrong answer scores beautifully. Define *resolved* with all three clauses: the user's issue closed **without a human touching it**, **without the user re-contacting about the same issue within a window** (7 days is a common choice), and **without a new ticket or phone call on the same underlying problem**. That re-contact clause is what stops "the conversation ended" from silently counting as "the problem was solved." Two more framing points worth stating unprompted: resolution rate is only meaningful **against the human-agent baseline** — human agents don't resolve 100% either, so the target is "at or above the human baseline on the intent mix the bot actually handles," not an absolute number pulled from the air — and it must be paired from day one with counter-metrics (escalation rate, CSAT, re-contact rate, handle time, cost per contact), for reasons the online-test discussion below makes concrete.

Resolution requires the bot *not* to hand off, but a confidently wrong answer is far worse than a handoff — so how does the bot decide it's out of its depth? **Confidence-based escalation, built from several signals — "the model's confidence" alone is the weak answer.** The signals, roughly in order of usefulness:
- **Retrieval-side, available *before* generation** — the top-k retrieval score and the margin between top-1 and top-2, plus whether anything cleared a relevance floor at all. If the knowledge base contains nothing relevant, escalate regardless of how fluent the draft answer is. This is usually the single strongest signal, and its timing is the point: it fires before you've spent a generation call or shown the user anything.
- **Generation-side** — raw sequence log-probability is poorly calibrated and shouldn't carry the decision alone; **self-consistency** (sample *n* answers, measure agreement) is meaningfully better at *n*× the inference cost; an explicit **verifier pass** (does the retrieved context entail the drafted answer) is the same entailment idea as the search product's span verification, applied pre-send instead of post-hoc.
- **Policy-side, which is not a confidence question at all** — a hard routing table where certain intents always reach a human regardless of how confident the model is: refunds above a threshold, account closure, anything legal, safety-related, or complaint-shaped, and detected anger or distress. Conflating "the model is unsure" with "a human must handle this by policy" is a design mistake; they're separate gates with separate owners.
- **Conversation-side** — turn count without progress, the user rephrasing the same question, and an explicit "let me talk to a person," which must be honored **immediately and unconditionally**; a bot that negotiates with that request is the fastest way to destroy CSAT that exists.

**Calibrating the threshold** is the real work: on a labeled set, plot resolution-rate-given-not-escalated against escalation rate and pick the operating point — the same precision/recall tradeoff in operational clothing. Both degenerate ends are instructive. Threshold too low and the bot escalates everything: it scores perfectly on every hallucination metric, delivers **zero value, and is actually net-negative** because it has inserted latency into every single contact. Threshold too high and it answers hard cases confidently and wrongly, which is the expensive failure — it costs the resolution, the customer's trust, *and* usually a second contact. And the threshold is not one global number: **calibrate per intent**, because being wrong about store hours and being wrong about a refund policy differ by orders of magnitude in cost, and re-calibrate on a schedule, since both the knowledge base and the incoming question mix drift.

That escalation threshold has to be calibrated on a labeled set — but where does that set come from before the bot has taken any live traffic? **Offline eval on a held-out set of real past tickets with known correct resolutions.** Sample historical tickets **stratified by intent**, deliberately over-sampling the rare and hard intents that random sampling underweights exactly where accuracy matters most; take the human agent's actual recorded resolution as the reference; replay the opening user message through the bot. **Grade with an LLM-judge against a rubric** — did the bot reach the same *substantive* resolution as the human, not the same wording — which is precisely why ROUGE/BLEU is the wrong instrument here: two correct resolutions to the same ticket can share almost no vocabulary, and two answers that share most of their vocabulary can differ on the one clause that matters. Then **validate the judge against human raters on a sample and report the agreement rate (or Cohen's kappa) as a first-class number**, because an uncalibrated judge is a confident random number generator wearing a metric's clothing. Name the structural limits yourself rather than waiting to be asked: (a) it's a **replay**, so it's effectively single-turn — real support is multi-turn and the user reacts to what the bot said, which no replay simulates; (b) historical tickets carry **selection bias** — they're the contacts that became tickets, and the recorded resolution is the one a particular human found, not necessarily the best one; (c) the **knowledge base has moved** since those tickets closed, so some answers graded "wrong" are actually right under current policy. Offline eval is a gate for shipping. It is not a substitute for what comes next.

Offline eval is a shipping gate that can't simulate a real multi-turn user — so what does the **online test** have to measure, and why can't it be one number? **An A/B test against a human-agent-only baseline, randomized at the contact level, with a primary metric and pre-registered guardrails — because every single metric here is gameable alone.** Measure resolution rate, CSAT, escalation rate, re-contact rate, and handle time **as a set**, and commit before the test that all of them must hold:
- **Escalation rate alone**: a bot that escalates 100% of contacts is flawless on every safety and hallucination metric and worth less than nothing, since it added a step to every interaction. This is the cleanest illustration of why a single metric can't own this decision.
- **Resolution rate alone**: a bot that closes conversations assertively, or subtly discourages follow-up questions, can post a strong resolution rate alongside collapsing CSAT and a re-contact spike that only shows up a week later.
- **CSAT alone**: agreeable, apologetic non-answers rate well in the moment, and CSAT carries severe response bias — mostly the delighted and the furious respond at all.
- **Handle time alone**: wrong answers are very fast.

So the design is one **primary metric (resolution rate)** with the rest as **guardrails that must not regress**, powered for enough sample to detect a meaningful move in the primary rather than a move in whichever guardrail happens to be noisiest. One more thing the "replacing human agents" framing makes non-optional: **keep a human-only holdout running permanently**, not just for the duration of the test — the moment the last human-handled slice disappears, you lose both your baseline for every future comparison and your ability to fall back when something breaks.

Given the online test runs against a permanently-maintained human-only holdout, how do you catch the bot degrading *after* launch, when a product change makes its knowledge base stale? **Layer the drift signals fastest-first, exactly as `production-ml-practice.md` and the monitoring section above prescribe — this specific failure is invisible to the fast ones alone.** The failure shape: nobody changed the model, nobody changed the prompt, and the bot's outputs look completely normal — the product shipped a new returns policy, the knowledge-base article wasn't updated, and the bot now answers the *old* policy fluently and confidently. That's **concept drift** in `production-ml-practice.md`'s exact sense: the input→correct-output relationship changed while the inputs still look statistically ordinary, which is precisely why input-distribution monitoring alone cannot catch it. The layered answer:
- **Input drift** (fastest): cluster incoming messages and alert on a new or rapidly growing cluster — "why was I charged this fee" appearing as a new intent is often the first observable trace of a product change.
- **Retrieval-signal drift** (nearly as fast, and the most specific to this system): a rise in low-retrieval-score conversations, or a rise in escalation rate, is frequently the **earliest** indication that the knowledge base no longer covers what people are asking — and it arrives before any outcome data exists.
- **Prediction drift**: a shift in which intents the bot resolves and which answer templates it reaches for.
- **Outcome drift** (slowest, most direct): resolution rate and re-contact rate on a rolling window, **sliced by intent** — the aggregate will happily hide one badly broken intent, which is the same segment-monitoring point the production-monitoring section above makes with locomotive classes.
- **A standing canary question set**: a fixed list of questions with known-correct *current* answers, re-run on a schedule and failing loudly when an answer goes stale — the support-bot version of the retrieval-quality eval loop in the internal-documents RAG design.

And the durable fix is process, not telemetry: **couple knowledge-base updates to the product release checklist**, because reliably detecting a staleness you could have prevented is still worse than not creating it.

With all five layers above now measured, what feedback actually makes the bot better — and what's the trap in that loop? **Every escalated conversation is a free labeled training example, and that's exactly what makes the loop dangerous.** When the bot hands off, the human agent's resolution becomes a reference answer for a case the bot couldn't handle — the highest-value supervision the system produces, generated at no extra cost. The trap is the **self-reinforcing feedback loop** the framework section at the top of this file names: you only ever get human ground truth for the cases the bot **escalated**. The cases it handled confidently and wrongly generate no correction at all — they generate a quietly unhappy customer and, at best, a re-contact somebody has to go looking for. So two deliberate counter-measures: **sample a fixed fraction of non-escalated conversations for human review**, which is the direct analogue of capturing outcomes for the locomotives the model *didn't* flag, and treat **re-contact within the window as a cheap automatic negative label** on a conversation the bot believed it had resolved. Without both, the bot's blind spots are structurally invisible to its own training data and compound over time.

### Summary example
A support bot launches against a resolution-rate definition that requires no human touch and no re-contact within 7 days. Escalation is driven primarily by retrieval margin, with a hard policy table routing refunds and complaints to humans regardless of confidence, and the threshold calibrated per intent on a stratified replay of 2,000 historical tickets graded by an LLM-judge whose agreement with human raters was measured at the outset. The online test randomizes at the contact level against a human-only arm, with resolution rate primary and CSAT, escalation rate, and re-contact rate as pre-registered guardrails — which is what catches an early candidate that posted a strong resolution rate while re-contacts rose 4 points. Two months post-launch a returns-policy change ships without a knowledge-base update: input clustering surfaces a new intent within a day, retrieval scores for that cluster crater, and the canary question set fails on the returns question — all well before outcome drift on resolution rate would have surfaced it weeks later. The correction lands in training data from the escalated cases, while a standing 2% sample of *non*-escalated conversations catches the ones the bot answered confidently and wrongly, which the escalation-only loop would never have shown anyone.

### Where I've actually built the LLM-judge half of this, and the metric gap it taught me
Two real pieces of my own work sit directly on this section, and the honest framing is that they cover the **output-quality** half and demonstrate exactly the gap the section opens with. QuitBuddy — the teen smoking-cessation platform I built with a Johns Hopkins faculty collaborator — was validated at **80%+ faithfulness by external LLM-as-judge evaluation**, and that evaluation was architectural rather than a QA afterthought: talking to teenagers about substance use means an off-domain or hallucinated response is a safety issue, so faithfulness had to be measured rather than assumed. That's the offline-eval mechanism above, built and shipped. And NaviDoc reported **35% ROUGE/BLEU** — a word-overlap metric — which is precisely the kind of number this section argues is the *wrong* instrument for a resolution question, and I'd rather name that myself than have it pointed out: an overlap score tells you the answer looked like the reference, not that the user's problem got solved. What neither project gave me is the **online** half — an A/B test against a human-agent baseline with resolution rate and CSAT as paired guardrails is design reasoning on my part, not something I've run.

### Where people trip up
- **Bot scores well on response-quality metrics while the support team sees no reduction in workload?** The metrics measure generated text rather than resolved issues. Resolution rate with a re-contact clause is the fix, and it has to be defined before launch, since retrofitting it means re-instrumenting the ticketing system.
- **Bot looks impressively safe because it almost never hallucinates?** It probably escalates nearly everything, and escalation rate was never treated as a metric that could fail. A 100%-escalation bot is perfectly safe and net-negative — escalation rate belongs in the guardrail set specifically so "safe" can't be achieved by delivering nothing.
- **Offline eval scores look strong and live performance doesn't match?** A replay of the first user message can't simulate a multi-turn conversation where the user reacts to what the bot said. Offline eval is a shipping gate — only a contact-level online test against a human baseline measures the thing being bought.
- **LLM-judge's grades trusted without ever being checked against human raters?** Judge calibration got skipped as an obvious step. Report judge-to-human agreement as a first-class number, or the whole offline eval rests on an unmeasured assumption.
- **Bot silently starts answering an outdated policy?** Knowledge-base staleness is concept drift, and the inputs still look completely normal. Retrieval-score and escalation-rate monitoring plus a standing canary question set catch it early, and coupling knowledge-base updates to the product release checklist prevents most of it outright.
- **Bot's training data only ever improves it on cases it already escalates?** Ground truth is only collected where a human intervened. Sample non-escalated conversations for review and treat re-contact as an automatic negative label, or its confident mistakes stay structurally invisible.

<details>
<summary><strong>Self-check — answer before revealing</strong></summary>

1. Why does resolution rate need a re-contact clause, and not just "did the human not have to touch it"?
2. Rank the escalation signals from strongest/earliest to weakest, and explain why retrieval-side confidence usually comes first.
3. What happens at each degenerate end of the escalation threshold — set too low, and set too high?
4. Name two structural limits of offline replay eval that you should name before an interviewer asks.
5. Why can't a single primary metric (resolution rate alone) safely drive the online A/B test?
6. Why is a returns-policy update that isn't reflected in the knowledge base a form of concept drift rather than input drift?
7. What's the trap in using escalated conversations as training data, and what two counter-measures fix it?

**Answers**
1. Without it, "the conversation ended" would silently count as "the problem was solved" — a bot that closes conversations assertively or discourages follow-up could look resolved while the user has to come back later, which the re-contact window is specifically designed to catch.
2. Retrieval-side signals (top-k score, margin, relevance floor) first, since they're available before generation is even spent and are usually the strongest signal; then generation-side (self-consistency, a verifier pass); then policy-side hard rules (not a confidence question at all); then conversation-side signals like an explicit request for a human, which must be honored unconditionally regardless of any confidence score.
3. Too low: the bot escalates nearly everything, scores perfectly on hallucination metrics, but delivers zero value and is net-negative because it added latency to every contact. Too high: it answers hard cases confidently and wrongly, costing the resolution, the customer's trust, and usually a second contact.
4. It's effectively single-turn (a replay can't simulate a multi-turn user reacting to what the bot said); historical tickets carry selection bias (they're the contacts that became tickets, graded against one human's resolution, not necessarily the best one); and the knowledge base has moved since those tickets closed, so some "wrong" answers are actually right under current policy.
5. Because every metric here is gameable alone — a bot can post a strong resolution rate while CSAT collapses or re-contacts spike a week later, so resolution rate needs CSAT, escalation rate, re-contact rate, and handle time held as guardrails that must not regress.
6. Because the input→correct-output relationship changed (the same kind of question now has a different correct answer) while the inputs themselves still look statistically ordinary — nothing about the incoming messages looks anomalous, so input-distribution monitoring alone has nothing to fire on.
7. The bot only gets human ground truth for cases it escalated, so its confident wrong answers generate no correction and stay structurally invisible. The fixes: sample a fixed fraction of non-escalated conversations for human review, and treat re-contact within the resolution window as a cheap automatic negative label.
</details>

> **Recap**
> Resolution rate — defined with a re-contact clause and measured against the human baseline — is the north star, not output-quality metrics. Escalation is a layered, per-intent-calibrated decision, not a single confidence number. Offline eval (replay + judge, checked against humans) gates shipping; only an online A/B test against a permanent human-only holdout, with resolution rate primary and everything else a guardrail, measures the real thing. Post-launch, layer drift signals fastest-first to catch knowledge-base staleness before outcome drift would. And deliberately sample non-escalated conversations, or the bot's confident mistakes never make it into its own training data.

---

## Designing a Fine-Tuning Pipeline for a 70B Model on a Small Proprietary Dataset Under a Tight Budget

> **TL;DR**
> - Full fine-tuning at 70B is ruled out mostly by memory and overfitting, not dollar cost: ~1.12TB of parameter/optimizer state, and 70B trainable parameters against a few thousand examples memorizes instantly.
> - QLoRA fits comfortably on **one** 80GB GPU (~40-45GB total): 35GB for the 4-bit frozen base, under 0.3GB for adapters, the rest activations.
> - The small dataset — not the memory budget — is what should keep rank and adapted-module count low; more capacity than the data can support just memorizes.
> - $500 buys roughly 10-15 full QLoRA runs (~$30-55 each); the equivalent full-fine-tune session would cost $100-250 *per session* on a 16-GPU reservation you can't even rent by the hour.

### Plain-English explanation
What makes this a design question rather than a `Trainer` call is that the two constraints happen to point the same direction for different reasons: the budget forces you to touch as few parameters as possible, and the small dataset means you *should* touch as few as possible regardless of what you could afford. The interesting reasoning is where they disagree — and in knowing that the cost argument against full fine-tuning is the weaker of the two.

**Visual + memory hook — the decision funnel from "rule out full fine-tuning" to "serve the adapter":**
```
  FULL FINE-TUNING at 70B — ruled out
  ~1.12 TB state (weights+grads+Adam moments) → needs ≥16×80GB
  + 70B params vs. ~1.2M supervised tokens = instant memorization
          │
          ▼
  QLoRA on ONE 80GB GPU  (~40-45 GB total)
  ┌─────────────────────────────────────────┐
  │  35 GB   4-bit NF4 frozen base           │
  │ <0.3 GB  adapters + Adam moments (bf16)  │  ← negligible next
  │  few GB  activations (grad checkpoint)   │    to the frozen base
  └─────────────────────────────────────────┘
          │
          ▼
  RANK kept LOW (r=8-16, attention only) — the small
  DATASET is the binding constraint, not the memory budget
          │
          ▼
  SPLIT by entity (not row) → augment inputs only, verified
  outputs fixed → ablate against REAL held-out data
          │
          ▼
  TRAIN: eval every 25-50 steps, early-stop on validation,
  higher LR than full FT, general-capability check before/after
          │
          ▼
  ~$30-55/run, 10-15 runs fit a $500 budget
          │
          ▼
  SERVE: adapter merged or separate — either way, test the
  eval on the EXACT serving precision (4-bit vs. bf16)
```

### From ruling out the obvious approach, through sizing the adapter to the data, to serving what you trained

Before choosing any technique, it's worth asking why full fine-tuning is ruled out at 70B — and whether it's really the dollar cost that rules it out. **Start with the memory arithmetic, then be honest that memory and overfitting rule it out more decisively than money does.** Full fine-tuning a 70B model with Adam in mixed precision needs, per the standard accounting: bf16 weights `70B × 2 = 140 GB`, bf16 gradients `140 GB`, fp32 master weights `280 GB`, and Adam's two fp32 moment buffers at `280 GB` each — **roughly 1.12 TB of parameter and optimizer state before a single activation**. At 80 GB per GPU that's 14 GPUs of pure state, so ~16×80 GB across two interconnected nodes with FSDP/ZeRO-3 is the realistic floor. Four reasons it's actually ruled out, and the order matters:
1. **You can't rent 16 interconnected GPUs the way you rent one** — a 2-node NVLink/InfiniBand allocation is a reservation with minimum commitments and queue time, not a by-the-minute credit-card purchase.
2. **The budget has to cover a sweep**, not a run — 10-30 training runs across learning rate, epochs, and data variants, not one heroic attempt.
3. **Each run emits a ~140 GB checkpoint**, so storage and transfer become their own line item at sweep scale.
4. **It's statistically wrong for this data size** — 70B trainable parameters against a few thousand examples memorizes the training set almost immediately and degrades the base model's general capability (catastrophic forgetting), so it would be the wrong choice even if the money and hardware were free.

The mechanics of the alternative — LoRA's frozen-`W`-plus-`A·B` detour, the `alpha/r` scaling, QLoRA's NF4 base with double quantization and paged optimizers — are already worked out in `core-technical-depth.md`'s **Model Fine-Tuning: LoRA and QLoRA** section and in the NCA-GENL guide; this section applies them to this scenario rather than re-deriving them.

With full fine-tuning ruled out on hardware, iteration count, and overfitting grounds, what does the QLoRA memory budget actually look like on one GPU? **Work it out concretely — it fits on a single 80 GB card with real headroom.** The 4-bit NF4 frozen base is `70B × 0.5 bytes = 35 GB`, the same arithmetic the LLM-inference section runs for serving. The adapters, on a Llama-70B-shaped model (80 layers, `d_model` 8192, grouped-query attention so `v_proj` is 8192→1024) with **r=8 on `q_proj` and `v_proj` only**: `q_proj` contributes `A(8192×8) + B(8×8192) = 131,072` parameters, `v_proj` contributes `A(8192×8) + B(8×1024) = 73,728`, so ≈205k per layer × 80 layers ≈ **16.4M trainable parameters — about 0.023% of 70B**. In bf16 that's ~33 MB of adapter weights; add Adam's fp32 moments and an fp32 master copy and you're still under ~0.3 GB total. **The trainable side is genuinely negligible next to the frozen base** — which is the whole point, and it means the remaining budget is all activations. With gradient checkpointing, micro-batch 1, and sequences of 1024-2048 tokens, that's single-digit GB. **Total ≈ 40-45 GB on one 80 GB A100 or H100**, leaving headroom for longer sequences or a slightly larger micro-batch. Flag this as sized, not measured: activation memory swings with sequence length, batch size, and implementation details, so it's the number you'd verify with one short smoke run before booking hours. QLoRA's **paged optimizers** exist precisely to absorb the transient spike from an unusually long sequence rather than OOM-ing mid-run.

Adapters cost almost nothing in memory — so why not crank the rank and adapt every linear layer? This is where the small dataset changes the answer, and it's the actual design tradeoff: **the opposite of what the memory budget alone would let you do.** Count the supervision: 2,000 examples at ~600 tokens each is ≈**1.2M supervised tokens**. Against that, the r=8 attention-only configuration above has ~16.4M trainable parameters — already roughly **13 trainable parameters per token of supervision**, which is heavily over-parameterized. Push to r=64 across all linear modules (`q,k,v,o,gate,up,down`) and you land in the high hundreds of millions of trainable parameters — *hundreds* of parameters per supervised token. That isn't more capacity, that's a memorizer with a validation curve that turns upward inside the first epoch. So the small-data prescription, which is genuinely distinct from the generic LoRA guidance in `core-technical-depth.md`:
- **Keep rank low** — r=8 or 16, not 64+. And remember `alpha/r` scales the adapter's contribution, so changing `r` alone changes the effective update magnitude: a rank sweep is really an `(r, alpha)` sweep, with `alpha = 2r` a reasonable anchor.
- **Adapt few modules** — attention projections only (`q,v`, or `q,k,v,o` if that underfits), not all-linear. The MLP `gate/up/down` matrices are where most of a transformer's parameters live, so including them is the single biggest jump in trainable count you can make.
- **Regularize** — `lora_dropout` around 0.05-0.1 and weight decay on the adapters.
- **Name the direction of the tradeoff explicitly**, because this is what the interviewer is listening for: if the task is **narrow domain adaptation** (learn our terminology, our formatting, our ticket taxonomy), low rank on attention only is usually sufficient — that's exactly what `core-technical-depth.md`'s pitfall about r=4 being adequate for narrow adaptation but too constrained for broad behavioral change is saying, read from the other end. If the goal is broad **behavioral** change, you need more rank — but you also need far more data than "a small proprietary dataset," so on this brief the answer is low rank, and **if low rank underfits, the correct next move is more data, not more rank.**

Given the design is deliberately low-capacity to match a small dataset, how do you get more out of the few examples you have without inventing facts? **Split before you augment, augment only in the safe direction, and ablate the augmentation against real data.**
- **Split first, by the right unit.** With a few thousand examples, a validation set of a few hundred is a painful but non-negotiable cost — and it must be split by the natural entity (customer, document, ticket thread), **not by row**, or near-duplicates leak across the boundary and your validation loss quietly lies to you. Hold out a small **test** set you touch exactly once, because with a validation set this small you will absolutely overfit the *validation* set through hyperparameter selection if it's the only thing you ever look at.
- **Augment in the safe direction.** Use a stronger LLM to paraphrase the **inputs** — rephrase the user's question, vary formatting, verbosity, and typos — while keeping the human-verified **output** fixed. That teaches invariance without inventing anything. Generating brand-new `(input, output)` pairs wholesale is the dangerous direction: you're asking the stronger model for knowledge of *your proprietary domain*, which by construction it doesn't have, so you will manufacture confident domain errors and then train on them. If you generate pairs anyway, they need human verification before entering the training set and a flag so you can ablate them out.
- **Ablate.** Train with and without the synthetic data and compare on the **real** held-out set. Augmentation that improves synthetic validation and not real validation is a self-licking result.
- **Run the prompted baseline first.** On a few hundred to a few thousand examples, few-shot prompting or RAG over the same corpus very often beats fine-tuning outright — and saying so is a legitimate, senior answer in this round. If the fine-tune can't beat a well-constructed prompted baseline on the held-out set, the whole pipeline was the wrong tool and the budget is better spent elsewhere.
- **Quality over quantity at this scale** — a few hundred carefully curated, consistently formatted examples reliably beat a few thousand noisy ones, because at this ratio the model has enough capacity to faithfully learn your inconsistencies.

With a small train/val/test split and augmentation ablated against real data, how do you actually run training so it stops before overfitting? **Evaluate often, early-stop on the metric you care about, and check what you might have broken.** With 2,000 examples at an effective batch of ~8, one epoch is only ~250 steps — so evaluate every 25-50 steps, not once per epoch, or you'll blow past the minimum entirely. **Early-stop on validation with patience, and keep the best checkpoint by validation metric, not the last one** (a surprisingly common bug: the loop early-stops correctly and then saves the final state anyway). Expect **2-4 epochs**; on a set this small, validation loss frequently bottoms out inside epoch 2 and climbs afterward. Learning rate: LoRA wants and tolerates a **higher LR than full fine-tuning** — 1e-4 to 2e-4 is the usual range against full fine-tuning's ~1e-5 — with a short warmup and cosine or linear decay. Two checks beyond loss: **run the actual task metric** (exact match, or a rubric-graded LLM-judge over a fixed set of held-out prompts) alongside validation loss, because loss can improve while the behavior you care about doesn't; and **run a small general-capability check before and after** — a handful of general benchmark prompts — because a fine-tune that nails the domain and degrades everything else is a regression, and **catastrophic forgetting does not show up in domain validation loss by construction**. One practical luxury worth naming: each checkpoint is a ~30-50 MB adapter, so keeping every experiment's artifact costs essentially nothing, which is not true of full fine-tuning's 140 GB checkpoints.

Given each run is a single 80 GB GPU with early stopping, does this actually fit a $500 budget — and how does that compare to the full fine-tune ruled out above? **Yes, with real margin — and here is the arithmetic, which is an order-of-magnitude estimate rather than a quote, since GPU rental rates move constantly and effective throughput is implementation-dependent.**

*Per-run compute:*
- **Tokens**: 2,000 examples × ~600 tokens ≈ 1.2M tokens per epoch; 3 epochs ≈ **3.6M tokens**.
- **Training FLOPs**: the standard `6 × N × tokens` estimate gives `6 × 70e9 × 3.6e6 ≈ 1.5e18` FLOPs. Gradient checkpointing recomputes the forward pass, so add ~33% → **≈ 2e18 FLOPs**. One nuance worth saying out loud, because it's the arithmetic people get wrong here: `6N` counts the **full model's** forward and backward, and that's correct even though only the adapters receive updates — the frozen 70B base is still fully evaluated and back-propagated *through* on every step. **QLoRA saves memory, not compute.**
- **Effective throughput**: an 80 GB A100 peaks around 312 TFLOP/s in bf16, but QLoRA on a single GPU with micro-batch 1, gradient checkpointing, and 4-bit dequantization in the matmul path realistically achieves a low fraction of peak — assume ~10-15% model FLOPs utilization, i.e. **≈ 4e13 FLOP/s**.
- **Wall clock**: `2e18 / 4e13 ≈ 5e4 seconds ≈ 14 hours`. Call it **10-20 hours** once data loading and periodic eval passes are counted.

*Per-run dollars:* a single 80 GB A100 rents for roughly **$1.50-2.50/hour on secondary GPU clouds** and roughly **$3-4/hour on hyperscaler on-demand** (spot is cheaper still, and a frequently-checkpointed fine-tune tolerates preemption well). At $2-4/hour × ~14 hours ≈ **$30-55 per full run**.

*Budget fit:* **$500 buys roughly 10-15 full runs** — and in practice more experiments than that, because most of a hyperparameter sweep over `(r, alpha, LR)` is decidable on a **1-epoch or even partial-epoch run** at a fraction of the cost. Realistically that's a proper sweep, two or three full runs, and a rerun after a data fix, with margin left. Verdict: comfortably achievable.

*The full fine-tuning comparison, honestly:* the accounting above requires ≥16×80 GB. The **compute per token is identical** — the same `6N` — so on only 3.6M tokens the pure FLOPs are small, and 16 GPUs would chew through them in well under an hour of theoretical compute. **That's the trap in this comparison and you should name it rather than let an interviewer find it.** What actually breaks the budget is that you rent the entire 16-GPU, 2-node allocation for the entire session at roughly $2.50-4/GPU-hour → **$40-64/hour**, and a realistic session — streaming and sharding 140 GB of weights, debugging an FSDP config, absorbing one OOM retry — is hours, not the ~20 minutes of theoretical compute. One session lands around **$100-250**; a 10-run sweep is **$1,000-2,500**, several times the budget, before 140 GB × 10 of checkpoint storage and before the practical reality that a 2-node interconnected reservation isn't an on-demand purchase in the first place. So the bottom line has two parts: **QLoRA fits the budget with margin, full fine-tuning doesn't — and the overfitting argument above would rule full fine-tuning out even if the money were there.**

And given a ~40 MB adapter is the deliverable, how does it actually get served — and what's the subtle mismatch between how it was trained and how it's deployed? **Two deployment options, and one train/serve consistency question that's easy to miss.** Per `core-technical-depth.md`, you either **keep the adapter separate** — hot-swappable per task or per customer, and vLLM-style servers can hold one 35 GB base in memory while serving many tenants' adapters against it, which is the entire economic argument for this approach at organization scale — or you **merge `W + B·A`** into a dense matrix, which removes any added inference latency by paving the detour into the highway. The subtle part: **QLoRA trained the adapter against a 4-bit quantized frozen base.** Merging it into a bf16 base means the weights the adapter sits on at serving time are *not* the weights it was fit against. Usually that's fine; sometimes it's measurably not. Either way it's an assumption to **test, not assume** — so either serve on the same 4-bit base you trained against (which keeps train and serve numerically consistent and connects straight to the quantize-first step of `Designing an LLM Inference System at Scale`), or merge into bf16 and **re-run the held-out eval on the exact serving configuration** before shipping. This is the fine-tuning-specific instance of the training-serving skew idea this file opens with: the thing that silently changed between training and serving isn't a feature definition, it's the base model's numerical precision.

### Summary example
A team needs a 70B model to speak their internal maintenance-log vocabulary, with ~2,000 curated examples and $500. Full fine-tuning is ruled out in the first minute on the 1.12 TB state calculation and, more decisively, on the fact that 70B trainable parameters against 1.2M supervised tokens memorizes instantly. QLoRA on a single 80 GB card sizes to ~40-45 GB — 35 GB of NF4 base, under 0.3 GB of adapters and optimizer state, the rest activations. Rank is set deliberately low at r=8 on `q_proj`/`v_proj` only, ~16.4M trainable parameters, because this is narrow domain adaptation rather than broad behavioral change. The data is split by ticket thread rather than by row, augmented only by paraphrasing inputs around verified outputs, and ablated against the real held-out set — after a few-shot prompted baseline is measured first, so there's something to beat. Training evaluates every 40 steps, early-stops in epoch 2, keeps the best-by-validation checkpoint, and runs a general-capability spot check to confirm nothing else broke. Each run costs roughly $30-55 for ~14 GPU-hours, so the budget funds a partial-epoch sweep plus several full runs with room to spare. The 33 MB adapter ships served against the same 4-bit base it trained on, with the held-out eval re-run on that exact serving configuration to confirm the number held.

### If you haven't personally fine-tuned a 70B model, say so — and say what you have done that's the same instinct
My production LLM work — NaviDoc, FinSight, QuitBuddy, the Mental Health Wellness Chatbot — runs on **pretrained models via API and RAG, not fine-tuned weights**, and `core-technical-depth.md` already states that plainly rather than dressing it up; each of those was a deliberate tradeoff (QuitBuddy needed an 80%+ faithfulness bar on a narrow, sensitive domain quickly, and RAG plus careful prompt engineering got there without the fine-tuning infrastructure and labeling cost). What I *do* have hands-on is the **same instinct one abstraction level down**: the Pneumonia Detection (MobileNetV2) and Alzheimer's MRI staging (ResNet18) projects both froze a pretrained backbone, trained only a small task-specific head, and used a deliberately low learning rate **specifically to avoid overfitting a small medical dataset** — which is this section's rank-and-module argument, applied to vision instead of language. That's a real, defensible bridge into this design: freeze the pretrained capability, add minimal trainable capacity, size that capacity to how little data you actually have. What I'd be careful not to claim is the operational half — I haven't personally run a multi-day QLoRA sweep or measured MFU on an A100, so the arithmetic above is reasoning from published throughput and rental rates, not a bill I've paid.

### Where people trip up
- **LoRA fine-tune on a few thousand examples shows a beautiful training loss and a validation loss that started rising in epoch 1?** Rank and module coverage were probably set for a large-dataset regime. Drop to r=8-16 on attention projections only, re-tune `alpha` alongside `r`, and early-stop on validation rather than running a fixed epoch count.
- **Validation loss looks great but production performance doesn't match?** The split was probably random over rows, and near-duplicate examples leaked across it. Split by the natural entity (ticket thread, document, customer), which matters far more at a few thousand examples than at a million.
- **Synthetic augmentation improves validation but not the real held-out set?** The generating model invented domain facts it doesn't have. Augment *inputs* around human-verified outputs, flag synthetic rows so they can be ablated, and always compare on real data.
- **Model nails the target domain and gets noticeably worse at everything else?** No general-capability check ran alongside the domain eval. Catastrophic forgetting cannot show up in domain validation loss, so it needs its own before/after check on general prompts.
- **Merged, deployed model scores worse than the adapter did in evaluation?** It was probably merged into a full-precision base rather than the 4-bit base it was trained against. Re-run the held-out eval on the exact serving configuration, or serve on the same quantized base you trained on.
- **Budget evaporates after three runs?** Every run was probably a full multi-epoch run. Most of a sweep over rank, alpha, and learning rate is decidable on partial-epoch runs, and reserving full runs for the two or three configurations that survive the sweep is what turns $500 into a real experiment plan rather than three expensive guesses.

<details>
<summary><strong>Self-check — answer before revealing</strong></summary>

1. What are the four reasons full fine-tuning is ruled out at 70B, and which one would still apply even with unlimited money and hardware?
2. On a Llama-70B-shaped model, roughly how many trainable parameters does r=8 LoRA on `q_proj`/`v_proj` across 80 layers cost, and what fraction of the 70B base is that?
3. Why should the small dataset — not the memory budget — set the rank and module coverage?
4. Why must the train/val/test split happen by entity (ticket thread, customer) rather than by row?
5. Why does QLoRA save memory but not compute, relative to full fine-tuning, on the same number of tokens?
6. What's the subtle mismatch between how a QLoRA adapter was trained and how it might get deployed, and how do you catch it?

**Answers**
1. (1) You can't rent a 16-GPU interconnected allocation the way you rent one GPU, (2) the budget has to cover a 10-30 run sweep, not one run, (3) each run emits a ~140GB checkpoint, and (4) 70B trainable parameters against a few thousand examples memorizes almost instantly. The fourth — statistical overfitting — would rule it out even with unlimited money and hardware.
2. About 16.4M trainable parameters (≈205k per layer × 80 layers), roughly 0.023% of the 70B base.
3. Because 2,000 examples at ~600 tokens is only ~1.2M supervised tokens, and r=8 attention-only already gives ~13 trainable parameters per token of supervision — going higher doesn't add usable capacity, it turns the adapter into a memorizer whose validation loss climbs inside the first epoch.
4. Because splitting by row lets near-duplicate examples leak across the train/validation boundary, which makes validation loss quietly lie about how well the model generalizes to genuinely new cases.
5. Because the `6 × N × tokens` FLOPs estimate counts the full model's forward and backward pass regardless of which parameters receive updates — the frozen 70B base is still fully evaluated and backpropagated through on every step. QLoRA reduces what has to be stored (memory), not how much has to be computed.
6. The adapter was trained against a 4-bit quantized frozen base, so merging it into a full-precision (bf16) base for deployment means it's sitting on weights it was never fit against. The fix is to either serve on the same 4-bit base you trained on, or re-run the held-out eval on the exact serving configuration before shipping — never assume the merge is safe.
</details>

> **Recap**
> Full fine-tuning at 70B is ruled out mainly by ~1.12TB of state and instant overfitting on a small dataset, not raw dollar cost. QLoRA fits one 80GB GPU with real headroom (~40-45GB total), and it's the small dataset — not the generous memory budget — that should keep rank and adapted-module count low. Split by entity, augment only the input side around verified outputs, and ablate against real data. $500 comfortably funds a real hyperparameter sweep plus several full runs. And whatever precision you trained the adapter against, test — don't assume — that the same precision holds at serving time.

---

## Practice Q&A (Self-Test)

**Q1. What are the five stages of the ML system design framework, and why does the file describe them as a loop rather than a line?**
A: Feature pipeline, training, serving, monitoring, and feedback loop. It's a loop because monitoring feeds back into retraining, and real-world outcomes captured through the feedback loop feed back into the training data itself — the system is designed to keep improving, not just to ship once.

**Q2. What is training-serving skew, and what's the standard architectural fix?**
A: It's when a feature is computed slightly differently in the training pipeline (often batch, from a warehouse) than in the serving pipeline (often real-time), causing a model that looks good offline to perform poorly in production. A feature store (e.g., Feast) fixes this by centralizing feature definitions so both paths compute the same feature identically.

**Q3. What are the three kinds of drift in production model monitoring, and how do they differ in how quickly they surface a problem?**
A: Input drift (incoming feature distributions differ from training data), prediction drift (the model's output distribution shifts), and outcome/label drift (accuracy actually degrades once ground truth arrives). Input and prediction drift can be caught before a problem shows up in outcomes, while outcome drift is the most direct signal but also the slowest, since it requires waiting for real-world ground truth.

**Q4. How can a feedback loop become a self-reinforcing bias, using the file's locomotive inspection example?**
A: If the model only surfaces certain locomotives for inspection, it only ever gets ground-truth outcomes on the ones it flagged — it never learns from the failures it missed on unflagged units. That means its blind spots never get corrected in retraining and can compound over time instead of improving.

**Q5. In the real-time prediction system framework, what's the first design step, and why does it change the architecture more than other decisions?**
A: Nailing the latency budget first — "real-time" can mean a few seconds (a dispatcher-facing alert) or a few hundred milliseconds (an automated control-loop decision), and those lead to very different architectures. The latency budget constrains feature computation, model complexity, and serving infrastructure all at once, so it has to be pinned down before any of those choices are made.

**Q6. In the RAG-for-internal-documents design, why must permission filtering happen during or before retrieval rather than after?**
A: Filtering after retrieval still means an unauthorized chunk was matched and potentially logged or exposed — the vector index itself needs access-control metadata attached to every chunk so the search can be constrained by the requesting user's permissions at query time. The file frames retrieving a chunk a user isn't authorized to see as a security bug, not a quality bug.

**Q7. How did FinSight handle a sub-1-second latency budget across multiple LLM agents, and why was fraud detection built with a classical model instead of an LLM?**
A: FinSight held real-time portfolio sync under 1 second while running 3 LLMs across 7 agents by deciding what had to happen synchronously (fraud check, portfolio math) versus what could complete asynchronously after the user-facing update already returned (the fuller multi-agent debate reasoning). Fraud detection used Isolation Forest, a fast classical model, specifically because it sits on the critical path and couldn't absorb LLM inference latency.

**Q8. A 70B-parameter model needs to fit and serve well on a single 80GB GPU. Quantizing to INT4 gets the weights to ~35GB — is tensor parallelism still needed?**
A: Not for memory reasons anymore — 35GB fits comfortably on one 80GB GPU alongside a real KV cache. Tensor parallelism solves a *memory-doesn't-fit* problem specifically; once quantization has already solved that, reaching for tensor parallelism anyway adds real cross-GPU communication overhead for no benefit. If more throughput is still needed at that point, continuous batching and/or additional independent replicas are the correct next levers, not splitting a model that now fits on one device.

**Q9. Why does PagedAttention specifically borrow the concept of OS virtual memory paging, rather than just allocating a smaller fixed KV cache buffer per sequence?**
A: A smaller fixed buffer just shifts the problem — some sequences would overflow it (truncating context) while others waste most of it (a sequence that ends early still held memory sized for the worst case), which is exactly the internal-fragmentation problem uncapped pre-allocation has, just at a different size. Paging fixes the actual structural issue: allocating memory in small fixed-size blocks *just-in-time* as a sequence actually grows, with a lookup table mapping logical to physical blocks — the same trick that lets an OS give every process the illusion of contiguous memory without pre-committing to its maximum possible size upfront.

**Q10. What two real Bosch/Cognizant incidents does the file use to illustrate the monitoring philosophy of catching problems before they become a crisis?**
A: Recovering a ransomware-locked MongoDB instance by mounting it locally and performing a full backup and restore with zero data loss, and resolving a MongoDB split-brain incident on a 6-node replica set within 30 minutes by evicting and resynchronizing the stuck secondary while keeping the primary available. Both worked because the problem was caught and correctly diagnosed fast, which the file ties to trusting a faster, cheaper signal over waiting for a slow, expensive one.

**Q11. Why does the model answer for the locomotive failure system recommend a time-based split rather than a random split for offline evaluation?**
A: Because sensor data is temporal, and a random split would leak future information into training — evaluating on a held-out time-based split more honestly simulates how the model will actually be used, predicting forward from what's known so far.

**Q12. According to the "hiring manager's seat" framing, what gap does the candidate's all-cloud-native background (FinSight, NaviDoc, Bosch) raise for this round, and what's the recommended way to close it?**
A: The concern is whether the candidate has only ever designed for a world where data shows up clean, on time, through infrastructure they fully control — unlike a railroad, where sensor telemetry may cross intermittent field connectivity and systems of record may be decades-old with no API. The recommended move isn't to fake prior legacy/OT experience, but to explicitly ask where the sensor data physically originates and how reliably it reaches compute, before drawing any architecture boxes.

**Q13. A search + LLM product is RAG over live web results. Which requirement from the internal-documents RAG design disappears entirely, and what replaces it?**
A: Access control disappears — everything retrieved from the open web is already public, so the permission-filtering requirement that made retrieving an unauthorized chunk a *security* bug in the internal-documents design simply doesn't exist. What replaces it is a source-quality problem that the curated-corpus design never had: the web is unvetted, self-contradictory, heavily syndicated, and ranked for clicks rather than for being good grounding context — which is why the pipeline needs cross-encoder re-ranking of fetched passages, per-domain authority priors, and an explicit policy for conflicting sources.

**Q14. Retrieval, generation, and citation verification together take 5-10 seconds, but users expect near-instant results. What's the structural fix, and what metric does it actually improve?**
A: Stream the answer as soon as the first re-ranked passages land and resolve citations *behind* the stream — render inline markers optimistically as the model emits them, run the entailment check asynchronously per completed sentence, and let each marker settle a beat later (turning solid, or being visibly retracted). That doesn't reduce total end-to-end time; it changes which number the user experiences, from total time to **time-to-first-token**. Total time only becomes visible again if the stream stalls mid-answer.

**Q15. Why must a browsing agent's constraint against destructive actions live in the tool layer rather than the system prompt?**
A: Because page content is untrusted input. A webpage can contain text addressed to the agent instructing it to do something (prompt injection), so any safety property that exists only in the system prompt is defeatable by the very environment the agent is being asked to read. A prompt is a request; a missing tool is a guarantee — if the agent shouldn't be able to purchase, it shouldn't be given a tool that can, backed by a sandboxed browser profile with no saved credentials and a human confirmation gate showing the rendered consequence of any irreversible action.

**Q16. An autonomous browsing agent's `done()` call reports success. Why isn't that evaluation, and what three tiers replace it?**
A: `done()` is self-report, and "the loop stopped" is not the same as "the task was completed correctly." The three tiers are: programmatic state-based validators wherever possible (assert the final environment state against an answer key, the WebArena-style approach), an LLM-judge over the trajectory and final screenshot for tasks with no programmatic check — but only after measuring the judge's agreement with human raters — and stratified human review sampling weighted toward ambiguous runs and runs that requested a gated action. Success rate must also be sliced by site and task category, since 80% on search-and-read averaged with 5% on multi-step forms describes neither.

**Q17. Why can't a support chatbot's evaluation use response quality (BLEU/ROUGE or a per-response thumbs-up) as its primary metric?**
A: Those measure whether the text was well-formed, and a well-formed wrong answer scores beautifully. The business is buying whether the user's issue actually got resolved, so the north star is resolution rate defined operationally: closed without a human touching it, without re-contact about the same issue within a window (e.g. 7 days), and without a new ticket on the same underlying problem. Two correct resolutions to the same ticket can also share almost no vocabulary, which is exactly what word-overlap metrics can't see.

**Q18. A support bot escalates almost everything to humans and therefore never hallucinates. Why is that a failure rather than a safe outcome, and what metric design catches it?**
A: A 100%-escalation bot is flawless on every hallucination and safety metric while delivering zero value — and it's actually net-negative, since it inserted latency into every contact. The fix is a guardrail-metric design: resolution rate as the single primary metric, with escalation rate, CSAT, re-contact rate, and handle time as pre-registered guardrails that must all hold, because every one of them is gameable alone (resolution rate by closing conversations assertively, CSAT by agreeable non-answers, handle time by being fast and wrong).

**Q19. A product change makes the support bot's knowledge base stale. Why won't input-distribution monitoring alone catch it, and what's the earliest reliable signal?**
A: It's concept drift in `production-ml-practice.md`'s sense — the input→correct-output relationship changed while the inputs still look statistically ordinary and the bot's outputs stay fluent, so input monitoring has nothing anomalous to fire on. The earliest reliable signals are retrieval-side: a rise in low-retrieval-score conversations or a rise in escalation rate indicates the knowledge base no longer covers what people are asking, and both arrive before any outcome data exists. A standing canary question set with known-correct current answers, plus coupling knowledge-base updates to the product release checklist, is the durable fix.

**Q20. On a $500 budget with ~2,000 proprietary examples, why is full fine-tuning of a 70B model ruled out — and is the raw compute cost really the reason?**
A: No, and that's the point worth making before an interviewer finds it. The compute per token is the same `6 × N × tokens` for both approaches, so on only ~3.6M tokens the pure FLOPs are cheap either way. Full fine-tuning is ruled out because Adam in mixed precision needs ~1.12 TB of parameter and optimizer state (140 GB bf16 weights + 140 GB grads + 280 GB fp32 master + 2×280 GB Adam moments), i.e. ≥16×80 GB across two interconnected nodes — which is a reservation, not an on-demand purchase; because the budget must fund a 10-30 run sweep, not one run, and each session on a 16-GPU allocation at ~$40-64/hour lands around $100-250; because each run emits a ~140 GB checkpoint; and most decisively because 70B trainable parameters against ~1.2M supervised tokens memorizes immediately. QLoRA on one 80 GB card is ~40-45 GB of memory and roughly 14 GPU-hours (~2e18 FLOPs at ~10-15% MFU), i.e. ~$30-55 per run at $2-4/hour — order-of-magnitude estimates, not quotes — so $500 funds 10-15 full runs plus a cheaper partial-epoch sweep.

**Q21. With only ~2,000 examples, why choose a *lower* LoRA rank and fewer adapted modules, when the memory budget could easily afford more?**
A: Because memory isn't the binding constraint — supervision is. 2,000 examples at ~600 tokens is ~1.2M supervised tokens; r=8 on `q_proj`/`v_proj` across a 70B model is already ~16.4M trainable parameters, about 13 trainable parameters per supervised token. Going to r=64 across all linear modules (including the MLP `gate/up/down` matrices, where most parameters live) pushes that into hundreds of parameters per token, which memorizes rather than generalizes. Also, `alpha/r` scales the adapter's contribution, so a rank sweep is really an `(r, alpha)` sweep — and if low rank underfits, the correct move is more data, not more rank.

**Q22. QLoRA trained an adapter against a 4-bit base. What's the risk in merging it into a bf16 base for serving, and how does that connect to a concept from the top of this file?**
A: The adapter was fit against quantized base weights, so merging into full precision means it's now sitting on weights it was never trained against — usually fine, sometimes measurably not, and either way an assumption to test rather than assume. It's the fine-tuning-specific form of **training-serving skew**: the thing that silently changed between training and serving isn't a feature definition, it's the base model's numerical precision. The fix is either serving on the same 4-bit base you trained on (which also lines up with the quantize-first step of the LLM-inference-at-scale design) or merging into bf16 and re-running the held-out eval on the exact serving configuration.
