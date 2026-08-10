# System Design — Deep Interrogation Drills

`system-design-prep.md` teaches the frameworks. `real-world-incidents.md` shows what actually broke in the real world. This file is neither — it's **what the round actually feels like**: an interviewer hands you one scenario with a real number in it, and then doesn't stop. Every answer gets a follow-up. Every number gets "why that number, specifically." Every design gets pushed to 10x scale to see what breaks first. This is a transcript of that pressure, worked through with real arithmetic — read it once to learn the moves, then close the file and run the next one cold, out loud, against a clock.

---

## Drill 1: RAG at 10,000 Concurrent Users — Accuracy AND Latency

**Interviewer:** Your RAG system needs to serve 10,000 concurrent users. Design it for both accuracy and latency.

**Candidate:** First I need to convert "10,000 concurrent users" into an actual load number, because it isn't a QPS figure by itself. If the average user sends a query roughly every 20 seconds while active (typical for a chat-style product — read the answer, think, ask again), Little's Law gives me `QPS ≈ concurrent_users / avg_seconds_between_requests = 10,000 / 20 = 500 QPS`. I'd confirm that 20-second assumption with product data rather than guess, but I want a number to design against rather than stay abstract.

**Interviewer:** Fine, 500 QPS. What's your latency budget?

**Candidate:** I'd split it into two numbers, not one, because they're perceived differently: time-to-first-token (how fast the answer starts appearing) and total time (how fast it finishes). For a chat surface, I'd target **p95 time-to-first-token under ~1 second** and let total time run a few seconds as long as it's streaming — this is the same point `Designing a Search + LLM Product` makes: users measure the first number, not the second, as long as the stream doesn't stall.

**Interviewer:** Break down that 1-second budget across your actual pipeline stages.

**Candidate:** Roughly: query embedding (~10-30ms, batched on GPU), ANN vector search (~5-50ms depending on index size and `ef_search`), optional cross-encoder reranking (~50-200ms for a top-20-50 candidate set), then the first LLM token. If I want first-token under a second and the LLM call itself needs several hundred milliseconds just to start generating, retrieval and reranking together need to fit in roughly **150-300ms**, combined — that's the number that actually constrains my architecture choices, not the 1-second headline.

**Interviewer:** At 500 QPS, what breaks first — the vector index or the reranker?

**Candidate:** The reranker, almost certainly, and by a wide margin. A single-node ANN index (HNSW, in-memory) can typically serve thousands of read QPS if it fits in RAM — 500 QPS of read-only nearest-neighbor lookups usually isn't the bottleneck. But if I rerank the top 20-50 candidates per query with a cross-encoder, that's `500 × 20 = 10,000` to `500 × 50 = 25,000` transformer forward passes *per second*. That's real GPU compute, and it's the first thing I'd expect to saturate.

**Interviewer:** So what do you actually do about that?

**Candidate:** Several levers, in the order I'd reach for them: shrink the reranked candidate set (top-20 instead of top-50, if recall holds — measured, not assumed); dynamically batch reranking requests across concurrent users into one GPU forward pass, the same continuous-batching idea from `Designing an LLM Inference System at Scale`, just applied to the cross-encoder instead of the generator; horizontally scale reranker replicas with autoscaling on queue depth, not raw GPU utilization, for the same reason that section argues against utilization-based scaling; and semantic-cache rerank results for repeated or near-duplicate queries.

**Interviewer:** You mentioned caching. How much does that actually save you at 10,000 concurrent users — give me a real answer, not "it helps."

**Candidate:** Honestly, it depends entirely on query repetition rate, and I wouldn't assume a number — I'd measure it. A narrow-domain FAQ-style support bot might see 30-40% of queries be near-duplicates of something asked in the last hour, which makes caching a major lever. An open-ended research assistant might see under 5% repetition, in which case caching barely moves the needle and I shouldn't design around it. This is a "check before you build" item, not a "everyone knows caching helps" assumption.

**Interviewer:** Now the accuracy half. How do you know your retrieval accuracy at 10,000 concurrent users under load is the same as what you measured in your offline eval with one request at a time?

**Candidate:** This is the actual trap in this question, and it's easy to miss: under load, teams often silently trade accuracy for speed — lowering `ef_search` on the ANN index, shrinking the reranked candidate set, or skipping reranking entirely once queues back up — and then nobody re-measures whether recall@k actually held at the degraded settings. The fix is running the exact same standing retrieval-quality eval from `Designing a RAG System for Internal Documents` **under simulated load**, not just under quiet lab conditions, and alerting if live-measured recall/precision drops as load-shedding kicks in. A system can be fast and wrong at scale while looking identical to fast-and-right in a demo.

**Interviewer:** So when you're over capacity, what's your actual policy — slower answers, worse answers, or dropped requests? Pick one and defend it.

**Candidate:** I'd make this an explicit, ordered policy rather than let it happen by accident: first lever is horizontal autoscaling with a warm minimum pool (never scale from zero, per the inference-at-scale section's cold-start point); second lever is queueing with a bounded max wait and an explicit backpressure signal (429) to the client rather than silently degrading quality; third lever, only if I must degrade gracefully, is a **named, monitored** step-down (e.g., `ef_search` reduction) with an accuracy-floor alarm attached to it. The failure mode I'm avoiding is accuracy quietly becoming the invisible pressure-relief valve nobody's watching.

**Interviewer:** 10,000 users becomes 100,000 next quarter. What's the first thing that stops scaling linearly?

**Candidate:** The vector index. A single-node in-memory index eventually doesn't fit in RAM and can't absorb more QPS on one machine, so I'd need to shard it — partition by namespace or hash, fan out the query to every shard, merge top-k results — which adds real fan-out latency and coordination complexity that didn't exist at 10,000 users. The LLM generation tier scales more predictably (roughly linearly in GPU count) but becomes the dominant *cost* line, since it's the most compute-expensive stage per request by far.

**Interviewer:** Last one. Accuracy and latency are in real tension here. Name the actual dial, and tell me who should be turning it.

**Candidate:** The literal dials are `ef_search` and reranked-candidate-count — turning them down buys latency and costs recall, measurably. But which setting is "correct" isn't an engineering call in isolation — it's the same error-cost calibration from `problem-formulation-framework.md`: what does a slightly-worse retrieved answer cost the business versus what does a slower response cost in abandoned sessions. I'd bring the measured tradeoff curve to product, not decide it unilaterally.

---

## Drill 2: Continuous Training on Continuous Input

**Interviewer:** Your system receives a continuous stream of new labeled data. How do you continuously train on it?

**Candidate:** First I need to pin down what "continuously" actually means architecturally, because three very different things hide behind that word: true online learning (update weights after every single example), continuous micro-batch retraining (retrain on a rolling window every N minutes/hours), or continuous adapter fine-tuning (small LoRA-style updates on a frozen base, on a recurring cadence). I'd ask which is intended — but my default recommendation is the second or third, not the first, and I'd say why unprompted.

**Interviewer:** Why not true online, per-example SGD? It sounds like the most "continuous" answer.

**Candidate:** Two reasons. Statistically, the gradient from one example is extremely noisy relative to a model of any real size — there's no averaging to smooth out a bad update, so one mislabeled or adversarial example can meaningfully move the weights with nothing to catch it. And operationally, there's no natural batch boundary at which to *evaluate before committing* — the update ships the instant it's computed, with no gate. That's structurally the same failure as Microsoft's Tay in `real-world-incidents.md`: a system that updated live on unfiltered input got steered somewhere bad within hours, precisely because nothing stood between "new input arrived" and "the model changed because of it."

**Interviewer:** Fine — micro-batch retraining, then. How often, and on how much data, concretely?

**Candidate:** Sized for a stable gradient estimate, not for "as often as possible." Say 50,000 new labeled examples arrive per day. Rather than retraining on every trickle, I'd accumulate a window — every 2-4 hours, or every 5,000-10,000 new examples, whichever comes first — large enough that an update reflects a real shift in the data rather than noise in a tiny batch. And critically, every window is mixed with a sampled slice of historical data, not trained on the new window alone.

**Interviewer:** Why replay old data? Isn't the entire point of continuous training to adapt to what's new right now?

**Candidate:** Because training only on the newest window is exactly how you get catastrophic forgetting — the model overfits to whatever's most recent (today's traffic mix, today's slang, today's adversarial probing pattern) and quietly loses general behavior that simply isn't represented in the last few hours of data. Same mechanism as the catastrophic-forgetting pitfall in `Designing a Fine-Tuning Pipeline for a 70B Model`, just recurring instead of one-shot. Concretely: maintain a replay buffer — a stratified sample across time and segments — and mix it into every training window at a tuned ratio, something like 70% new data / 30% replay as a starting point, validated against a general-capability eval rather than picked arbitrarily.

**Interviewer:** How do you know a given update didn't make things worse, before it's serving real traffic?

**Candidate:** It never auto-deploys straight to production. Every update goes through the same shipping gate a normal release would: score it against a standing held-out eval set — one that was never in the training window or the replay buffer — plus a general-capability check, and only promote the new checkpoint if it doesn't regress past a defined threshold. If it does, the update is discarded, not patched, and the previous checkpoint keeps serving. It's a continuous canary gate, not a one-time launch gate.

**Interviewer:** The world is changing continuously too, which is the whole premise here. How do you know that held-out eval set itself hasn't gone stale?

**Candidate:** It can, and this is the exact eval-set-decay point from `Designing a Search + LLM Product` — a question that was correctly answered by the eval set six months ago can be wrong now if the underlying task or world shifted underneath it. So the eval set needs its own re-validation cadence, separate and slower than the training cadence, with a human check before a failing eval query gets counted as a real regression rather than the world having moved.

**Interviewer:** What about the labels themselves? A continuous input stream rarely arrives with instant, clean ground truth.

**Candidate:** Right, and I'd name that lag explicitly rather than assume it away — if the true label arrives a day after the prediction (did the recommendation actually convert, did the flagged transaction actually turn out fraudulent), the training window has to be bounded by the **confirmed-label** window, not the raw-input-arrival window. Where a fast, noisy proxy label exists (click-through as a stand-in for eventual conversion), I'd use it for a fast signal but weight it lower than the slower, clean label rather than blending the two as if they were equally trustworthy.

**Interviewer:** Roughly, what does retraining every 2-4 hours cost compared to a normal weekly batch retrain? Put a number on it.

**Candidate:** Using the QLoRA arithmetic from the fine-tuning-pipeline scenario as a reference point — roughly $30-55 and ~14 GPU-hours per full adapter training run — retraining every 4 hours instead of weekly is about **42x more frequent** (168 hours / 4 hours). Cost has to come down close to proportionally or the budget explodes, which is the real argument for continuous updates being small adapter fine-tunes on a frozen base rather than continuous full retrains: the frozen base means each update is cheap enough to actually run 42x more often without 42x the spend.

**Interviewer:** Last one. Six months of continuous updates in — is the model actually better than where it started, or has it quietly drifted somewhere you wouldn't have chosen on purpose?

**Candidate:** This is where the feedback-loop self-reinforcement warning from `Designing Production Model Monitoring` matters most, because it compounds continuously instead of just once: if some of the "new" labels are downstream outcomes the model's own decisions influenced, training on them closes a loop that can reinforce the model's existing blind spots rather than correct them. The fix is the same one named there, applied on a rolling basis — deliberately sample and label cases *outside* what the model is currently confident about, not just whatever flows back in naturally — plus tracking a long-horizon general-capability metric across the full six months, not just each window's own eval, so a slow drift away from the original goal is visible even though no single update ever looked like a regression on its own.

---

## Drill 3: Real-Time Fraud Detection — 5,000 Transactions/Second, Sub-50ms Decision Budget

**Interviewer:** You need to approve or decline 5,000 transactions a second, each within 50 milliseconds. Design it.

**Candidate:** 50ms is the number that eliminates most of the usual toolkit immediately — that's not enough time for an LLM call, and it's tight even for a moderately sized neural net once you count network hops. I'd default to a **fast classical model** on the hot path — gradient-boosted trees or a small model like Isolation Forest — precisely because it's a deterministic, low-latency forward pass, not a generative call. This is the same real decision FinSight made in `system-design-prep.md`: fraud detection ran on Isolation Forest specifically because it sat on the critical path and couldn't absorb LLM-scale latency.

**Interviewer:** 50ms, broken down — where does it actually go?

**Candidate:** Roughly: feature lookup (recent transaction history, account velocity — often the biggest unknown, since it may mean a database round-trip), feature computation (a few milliseconds if precomputed/cached, much more if computed live), model inference (sub-millisecond to a few ms for a small tree ensemble), and network/serialization overhead on both sides. If feature lookup requires a live database join, that's very likely the majority of the 50ms budget, not the model itself — which is exactly the pattern the real-time-prediction framework in `system-design-prep.md` names: the bottleneck in a "real-time" system is almost always feature retrieval, not model inference.

**Interviewer:** So how do you make feature lookup fast enough?

**Candidate:** Precompute and cache what can be precomputed — rolling aggregates like "transactions in the last hour" maintained continuously in a low-latency store (Redis or similar) rather than queried fresh per request — so the hot path reads a small number of already-computed values instead of joining raw history live. This is the online/offline feature-consistency point from the ML-system-design framework again: whatever's precomputed offline-adjacent has to match, exactly, what a training-time feature pipeline would have computed, or I get training-serving skew on top of everything else.

**Interviewer:** At 5,000 TPS, what happens when your feature store is briefly unavailable?

**Candidate:** That has to have a defined fallback, not an accident — per the real-time-prediction framework's point on this exact failure: either a cached last-known-good feature value, a conservative default (e.g., treat missing velocity features as higher-risk, not lower-risk, since failing open on a fraud check is the worse direction to fail), or in the extreme, a much simpler rule-based fallback that can run with no feature store at all. What I wouldn't accept is the system silently approving every transaction because a dependency timed out — that's fail-open on the exact system where fail-open is most dangerous.

**Interviewer:** Your model's accuracy looks great offline. Three weeks after launch, fraud losses are creeping up even though nothing was redeployed. What's happening?

**Candidate:** Concept drift, almost certainly — fraud patterns actively adapt to whatever the current model blocks, so the input-to-correct-output relationship is shifting even though the model and code haven't changed. I'd expect input-distribution drift on transaction features to be the earliest signal, well before the outcome metric (confirmed fraud losses, which lag by days-to-weeks while a case gets investigated) moves — same fastest-to-slowest drift layering as `Designing Production Model Monitoring`. And I'd check it's not a feedback-loop problem specifically: if I only ever get confirmed-fraud labels on transactions the model already flagged, I have zero ground truth on the fraud my own model is currently missing, and that blind spot compounds every week it isn't corrected.

**Interviewer:** What would you change if this needs to go from 5,000 TPS to 50,000?

**Candidate:** The feature store and the model-serving tier both need to scale roughly horizontally with load — more replicas behind a load balancer, cache sharded rather than single-node — but the number I'd actually watch is p99, not average latency: at 10x the traffic, a queueing tail that was invisible at 5,000 TPS can start blowing the 50ms budget for a small but real fraction of transactions, and "average latency is fine" is exactly the kind of aggregate number that hides a real problem, the same segment-monitoring point that shows up everywhere else in this hub.

---

## Drill 4: Operating a Production RAG/LLM Pipeline — Pipelines, Integrity, Hosting, Hallucination, and the Daily Job

This one's shaped differently from the first three — it's not one number pushed to its limit, it's the full width of "you own this system now, not just the design doc." Six movements: building the pipeline, moving data between its stages with integrity intact, hosting it highly-available, stopping it from hallucinating or half-answering, knowing when to intervene, and shipping a new feature without anyone noticing. `production-ml-practice.md` and `mlops-practice.md` go deeper on rollout mechanics and the pipeline loop respectively — this drill is where they get pulled together into one operational answer.

### Movement 1 — Building the pipeline and moving data between stages

**Interviewer:** Walk me through the actual pipeline, stage by stage — not the model, the plumbing around it.

**Candidate:** For a RAG system specifically: ingestion (new/changed documents land, from a CMS, S3 bucket, or ticketing system) → parsing/chunking → embedding → upsert into the vector index → serving (retrieve → rerank → generate) → logging → monitoring → feedback capture. I'd orchestrate the ingestion-through-upsert half as a DAG (Airflow, Dagster, or a simpler queue-based worker if the volume doesn't justify a scheduler) — the point of a DAG here isn't sophistication, it's that each stage has an explicit dependency and can be retried independently without re-running everything upstream of it.

**Interviewer:** How does data actually move from the chunking stage to the embedding stage — same process, a queue, a database write?

**Candidate:** I'd put a durable handoff between any two stages that can fail independently and don't need to be synchronous — a message queue (Kafka, SQS) or a staging table with a status column (`pending → embedded → indexed`), not a direct in-memory function call. The reason: if the embedding stage is down for 10 minutes, chunked documents from ingestion shouldn't be lost or silently skipped — they queue up and get processed once the embedding service is back. A direct synchronous call couples the two stages' uptime together for no reason.

**Interviewer:** What stops a document from silently getting processed twice, or not at all, if a worker crashes mid-stage?

**Candidate:** Idempotency and an explicit status field, not "hope the crash doesn't happen at a bad moment." Each document gets a stable ID; a stage claims a document (status → `processing`, with a claim timestamp), and only marks it `done` after the write to the next stage succeeds — if a worker crashes mid-claim, a watchdog re-queues anything stuck in `processing` past a timeout. And the embedding upsert itself should be idempotent (upsert by document-chunk ID, not insert), so a retried write after a crash doesn't create a duplicate chunk in the index.

### Movement 2 — Data integrity, in simple terms

**Interviewer:** How do you know the data moving through this pipeline is actually correct, not just "didn't error out"?

**Candidate:** "Didn't error" and "correct" are different checks, and I'd gate on both. At each stage boundary: a schema/shape check (does this chunk have the fields I expect — text, source ID, permission metadata — before it's allowed into the next stage), a sanity check on volume (did today's ingestion job process roughly the expected number of documents, or did it silently process 3% of them because a source connector broke), and a content check where it's cheap (a chunk with zero characters, or one that's 99% whitespace, is a parsing bug, not valid data). None of these require ML — they're the same boring data-engineering discipline that catches most real pipeline failures before they ever reach the model.

**Interviewer:** Give me a concrete number — how would you actually alert on "ingestion silently broke"?

**Candidate:** Track daily ingested-document count as a time series and alert on a percentage deviation from a rolling baseline — e.g., if the 7-day trailing average is ~2,000 documents/day and today's run processed under 500 (a 75%+ drop), that's a page, not a log line, because a silent 75% drop is far more dangerous than a hard failure: a hard failure is visible immediately, a silent partial failure looks like a normal, boring day.

### Movement 3 — Hosting and high availability

**Interviewer:** How do you actually host this so it survives a bad day?

**Candidate:** Every stateless tier (retrieval API, reranker, generation gateway) runs as multiple replicas behind a load balancer, spread across at least two availability zones, so a single AZ failure doesn't take the whole system down — standard horizontal redundancy. The stateful tier (the vector index, if it's not already a managed service) needs its own replication story — either a managed vector DB with built-in replication, or a self-hosted index with a hot standby, because "just restart it" doesn't work for something holding an in-memory index that takes minutes to rebuild.

**Interviewer:** What's your actual failover story if the primary vector-index replica goes down mid-traffic?

**Candidate:** Health checks on each replica feeding the load balancer's routing decision, so traffic stops going to an unhealthy node within seconds, not after a human notices. And a defined degraded mode for the gap: if all retrieval replicas are briefly unavailable, the system should fail toward "no context, refuse to answer or say so" rather than "generate an ungrounded answer as if retrieval had succeeded" — silently dropping the grounding step is a correctness failure disguised as an availability one.

**Interviewer:** What's your actual uptime target, and what does that number mean in practice?

**Candidate:** I'd want that stated as an explicit SLO, not left implicit — say 99.9%, which is about 43 minutes of allowed downtime a month. That number should directly drive the redundancy decisions above: single-AZ hosting can't hit 99.9% no matter how good the code is, because an AZ-level outage alone burns the whole monthly budget in one event.

### Movement 4 — Stopping hallucination and partial responses

**Interviewer:** How do you actually stop this thing from hallucinating?

**Candidate:** Layered, not one trick. First, **grounding**: the system prompt instructs the model to answer only from retrieved context and to explicitly say it doesn't know rather than fill the gap — this alone doesn't guarantee anything, since the model can still ignore the instruction. So second, **verification**: a post-hoc entailment check (does the retrieved context actually support the generated claim, the same span-verification idea from `Designing a Search + LLM Product`) before the answer ships, catching cases where the model asserted something the context doesn't support. Third, **measurement**: faithfulness/groundedness is tracked as its own metric on a standing eval set — separate from "does the answer sound good" — because you can't fix a rate you're not measuring.

**Interviewer:** And a partial or cut-off response — how do you handle that?

**Candidate:** Detect it structurally, don't just hope the stream completes. If the response is meant to be structured (JSON, a numbered list with a known expected shape), validate the shape after the stream ends and reject/retry if it's malformed or truncated. If it's free text, a hard generation timeout with a documented behavior on trip — either retry once with a tighter token budget, or return what streamed so far plus an explicit "response was cut short" signal to the client, never silently presenting a truncated answer as if it were complete. The client-side contract matters as much as the server-side generation: the UI needs to be able to tell a genuinely finished answer from one that got cut off, which means the API needs to say so explicitly rather than the client guessing from "the stream stopped."

**Interviewer:** What actually causes a partial response in production, most commonly?

**Candidate:** Usually one of: the generation hit `max_tokens` before finishing (a token-budget-sizing problem, fixable by estimating the needed length upfront and either warning or auto-continuing), a downstream timeout fired mid-stream (a load or infra problem, not a model problem), or the client disconnected and the server didn't detect it, wasting GPU compute generating tokens nobody receives — worth explicitly canceling generation on client disconnect, both for cost and for correctness telemetry not misattributing that as a real completed response.

### Movement 5 — What to actually watch, and when it means "intervene now"

**Interviewer:** Give me the actual short list of metrics you'd watch, and what number on each one means "page someone" versus "note it and move on."

**Candidate:** Layered fastest-to-slowest, per the monitoring framework in `system-design-prep.md`: **p99 latency** against the SLO (page if p99 breaches budget for more than a few consecutive minutes, since a brief spike is often just a GC pause or a cold replica, but a sustained breach means real capacity trouble); **error rate** (page above some absolute threshold, e.g. >1% of requests erroring, tuned to the product's tolerance); **retrieval-score drift** — a rising share of queries with no result clearing the relevance floor (this is often the earliest sign something upstream broke, before any outcome data exists — flag for investigation, not necessarily an immediate page); **faithfulness/groundedness score** on the standing eval, run continuously not just at release (a sustained drop here is a page, because it means the system is actively giving unsupported answers to real users); and **escalation/refusal rate** as a guardrail metric (both directions are bad — a spike often means the knowledge base went stale, a drop toward zero can mean the safety gate stopped firing, which is worse and easier to miss).

**Interviewer:** Which of those is the one you'd actually wake up for at 3am?

**Candidate:** Sustained faithfulness drop or a p99/error-rate SLO breach — those two mean the system is actively serving users something wrong or not serving them at all, right now. Retrieval-score drift and escalation-rate movement are real signals but usually tell you something's *heading* toward a problem, which is a next-morning investigation, not a page — the distinction between "this is currently hurting a user" and "this predicts a future problem" is what actually separates a page from a ticket.

### Movement 6 — The daily job, and shipping a new feature without users noticing

**Interviewer:** Forget an incident — what does the boring, uneventful daily routine on this system actually look like?

**Candidate:** A short standing checklist, most days confirming nothing's wrong rather than fixing something: review the overnight dashboard (latency, error rate, cost, faithfulness score, escalation rate — the metrics from Movement 5, glanced at, not re-derived); check the ingestion volume against baseline (Movement 2's silent-failure check); spot-check a small sample of flagged/escalated conversations from the last 24 hours (the non-escalated-sampling discipline from the support-chatbot eval design — catching confident-wrong answers nobody else would surface); and confirm the standing canary/eval-question set still passes. Most days this is 15 minutes that confirms the system is healthy — the value isn't in what it usually finds, it's that skipping it is exactly how a slow problem (a stale knowledge base, a creeping cost trend) goes three weeks before anyone notices instead of one day.

**Interviewer:** Now ship a new feature — say, a new retrieval source — into this system without disrupting current users. Walk me through it.

**Candidate:** Staged, each stage gated on the previous one holding, per the canary framework in `production-ml-practice.md`: first, **shadow mode** — the new retrieval source runs alongside the existing pipeline for every real request, its results are logged and scored, but never shown to a user; this tells me if it retrieves well against real traffic with zero user-facing risk. Second, a small **canary** — route a small slice of real traffic (5%) to actually use the new source's output, with the same guardrail metrics from Movement 5 watched specifically on that slice, compared against the 95% control. Third, gradually widen the percentage only as each step holds — 5% → 25% → 100% — with an automatic or fast-manual rollback trigger if faithfulness, latency, or error rate on the canary slice regresses past a threshold at any step. The feature flag controlling the percentage is the actual mechanism — it means "roll back" is flipping a config value, not a deploy, which is what makes each step genuinely low-risk and fast to undo.

**Interviewer:** What's the one thing that undermines this whole staged rollout if you get it wrong?

**Candidate:** Picking guardrail metrics for the canary slice that are too slow or too aggregate to actually catch a regression before you've widened past it — if the metric you're watching only moves after outcome data trickles in over days, a same-day ramp from 5% to 100% will have already fully shipped before the signal arrives. The canary is only as good as the fastest metric layered under it, which is exactly why Movement 5's fastest-to-slowest signal stack matters here too, not just for steady-state monitoring.

---

## Drill 5: Designing an Explainable, Debuggable AI Agent System

### Plain-English primer, with one real example all the way through

**The business problem, in one sentence.** A company builds an AI agent to do real work — approve refunds, answer account questions, route support tickets — and every so often it does something wrong, and nobody can say *why*, which is a business risk (angry customers, compliance exposure, money out the door) long before it's an engineering annoyance.

**A concrete story.** Say you run a refunds agent for an online store. Policy: refunds are only allowed within 30 days of purchase. One day, the agent approves a refund for an order placed **47 days ago** — a clear policy violation. Nobody told it to break the rule. The code is fine. The servers are fine. So what happened?

It turns out that somewhere in the system, a fact got written for VIP customers — something like *"VIP customers get extended refund windows"* — meant for a different, narrower case. That fact got pulled into this customer's conversation because it looked relevant, and the model treated it as permission to override the 30-day rule. Nothing crashed. Nothing errored. The system just quietly did the wrong thing, confidently, in writing, to a real customer.

**Why normal logging doesn't catch this.** A normal log tells you "refund request → refund approved." It does not tell you "approved *because* the model was shown a VIP fact that didn't apply here." The information the log is missing isn't a bug in the code — it's a fact about *what the AI was shown right before it decided*. That's a fundamentally different thing to record than a stack trace or an error code, and most logging systems were never built to capture it.

**The engineering fix, in plain words.** Instead of building one big block of text ("the prompt") and hoping it's right, you label every single fact, instruction, and tool the AI can see with three simple things:
1. **Where it goes** — is this a background instruction, part of the conversation, or a tool the AI can use?
2. **When it's allowed to show up** — always, only under some condition, only after a certain step, or only when the AI specifically asks for it?
3. **Whether it's safe to reuse/cache**, so you're not paying full price to re-send the same stable instructions on every single message.

Do that, and every answer the AI gives can be traced backward like a paper trail: *this answer* came from *this exact text the model saw*, which came from *these specific facts*, each of which fired because of *this specific rule*, each of which was written by *this specific person or system* at some point in the past. That backward trail is the entire fix — it turns "we have no idea why it did that" into "here is the exact fact that caused it."

**Proving it, not just suspecting it.** Finding the suspicious VIP fact isn't proof it caused the bad refund — correlation isn't causation, same as in any other kind of debugging. The real proof: **remove that one fact, run the exact same request again a few times, and see if the answer changes.** If "APPROVED" flips to "DECLINED" every time you remove it, and stays "APPROVED" when you remove some other, unrelated fact instead, you've *proven* which fact was the cause — not just eyeballed a suspect.

**The tool-picking version of the same problem, with a simpler analogy.** Imagine a call-center rep's screen has ten buttons that all sound almost the same — "Issue Refund," "Process Return," "Cancel Order," "Reverse Charge" — and once in a while the rep clicks the wrong one because the labels are too similar. Two fixes, and AI agents need both: (1) **before you ever go live**, review the button labels and flag any pair that's too similar to reliably tell apart — this is a one-time design review, done with the tool descriptions rather than a live rep; (2) **while it's running**, keep a light-touch check that flags every time the "click" was a close call between two similar-sounding buttons, even if the "right" one was ultimately picked, so you can catch confusion before it becomes a mistake.

**Business logic vs. engineering design — keeping the two straight:**

| | Business logic (the "what should happen") | Engineering design (the "how do we know / how do we build it") |
|---|---|---|
| Example | "Refunds require the order to be under 30 days old" | The typed fact/rule model that lets you trace *which* text told the AI otherwise |
| Who owns it | Product, policy, compliance | Engineering |
| What breaks it | The rule itself is wrong, outdated, or was never written down clearly | The rule was right, but the wrong text reached the model, or reached it at the wrong time |
| This drill's focus | Not really this — a wrong *policy* is a product problem | Entirely this — building the system so a wrong *outcome* is always explainable and provable, regardless of whether the policy or the plumbing was at fault |

The engineering system doesn't decide what the refund policy should be — it makes sure that whatever the policy is, when the AI breaks it, someone can find out exactly why within minutes instead of never.

### What his blog posts teach, in plain language

Beyond the code, Sanjay Krishna Anbalagan has written a series of Medium posts arguing for this way of thinking. A few worth knowing, explained simply (some are member-only/paywalled past the opening, so this is what's confirmed available plus the stated thesis of each):

- **["Your Logs Are No Longer for You"](https://medium.com/codetodeploy/your-logs-are-no-longer-for-you-d12720dea6aa)** — His analogy: doctors used to write quick shorthand notes for themselves ("pt stable, cont mgmt"). That was fine when the same doctor read their own notes later. Once patient care became a "handoff sport" — different doctors, shift changes, referrals — sloppy shorthand became dangerous, and medicine had to switch to structured, standardized charts anyone could pick up cold. His claim: software logging is having the exact same handoff moment right now, except the new reader isn't a different doctor — it's an AI model that "never attended your standups" and has zero unwritten context about your system. If your logs only make sense to the engineer who wrote them, they're now failing their most important reader. The subtitle sums up the shift: logging goes from being a "cost center" (insurance you hope to never need) to a "product capability" (something the AI actively depends on to work correctly).

- **["The Flowchart Pattern: Making Backend Code Self-Explainable for AI"](https://medium.com/data-science-collective/the-flowchart-pattern-making-backend-code-self-explainable-for-ai-a508d779345c)** — The direct ancestor of `footprintjs`. His argument: traditional backend code keeps its logic private and only gets investigated by a human when something breaks. But once an AI is expected to explain a decision (a loan denial, a fraud flag, a support routing choice) *as part of the normal answer*, "what happened inside?" stops being a rare debugging question and becomes a routine product requirement. His fix is to write backend logic as an explicit flowchart of steps that record their own reads, writes, and branch decisions as they run — so the explanation is generated *from the actual execution*, not invented after the fact by an LLM guessing from scraps of log text.

- **["Act, Answer, Recall: The Three Modes of an Agentic Web App"](https://medium.com/codetodeploy/act-answer-recall-the-three-modes-of-an-agentic-web-app-a6c232e6a91f)** — His point: teams build "an AI agent" as if it's one thing, but it's secretly doing three different jobs that each need their own safety rules: **Act** ("book this for me" — needs transactional safety, since a mistake here does something real and possibly irreversible), **Answer** ("what's in my account?" — needs to be grounded in real, current data, not a plausible-sounding guess), and **Recall** ("why did you just do that?" — needs an honest explanation of the actual execution, not a fabricated-sounding justification). Because the user types all three into the same chat box, teams often build one code path for all three and inherit the worst failure mode of each — an "Act" that fails as loosely as an "Answer" is how you get an agent that books the wrong flight *and* can't explain why.

- **["Everyone Shows What MCP Does — But Nobody Tells You What It Abstracts"](https://medium.com/data-science-collective/everyone-shows-what-mcp-does-but-nobody-tells-you-what-it-abstracts-91432a79e416)** — Relevant to the tool-selection half of this drill: his argument is that most explanations of the Model Context Protocol show *what* it does (a standard way to plug tools into an AI) without naming what it's actually hiding from you underneath — the same context-injection and tool-exposure machinery this whole drill is about, just wrapped in a protocol so you don't have to build it yourself.

**The one-sentence version of everything above:** when an AI system has to explain itself, "why did it say that?" needs the same kind of rigor engineers already give "why did it crash?" — a trace you can walk backward, a way to prove the cause instead of guessing at it, and a clear line between "the rule was wrong" (a business call) and "the rule was applied to the wrong situation" (an engineering bug).

---

**Interviewer:** Your agent gave a customer a wrong answer — a refund it shouldn't have approved. Your logs show the request, the response, and a clean 200. Where do you even start?

**Candidate:** That's actually the core failure this question is testing for, and I'd name it before proposing a fix: classical logging records what the *code* did, never what the *context* did. The code path was correct, infrastructure was healthy, and the answer still came out wrong because something in the prompt — a stale fact, a mis-scoped instruction, a poisoned retrieval result — steered the model somewhere it shouldn't have gone. That's a third error class alongside bugs and outages: a **contextual error**, and it needs its own instrumentation, not more `print` statements around the LLM call.

**Interviewer:** So what does that instrumentation actually look like — what do you record, concretely?

**Candidate:** I'd make every piece of context injection typed and traceable rather than just concatenated into a prompt string. Concretely, one primitive: `injection = slot × trigger × cache`. Three slots, fixed by the LLM API surface — `system`, `messages`, `tools`. Four triggers describing *when* something fires — `always` (steering, static facts), `rule` (a runtime predicate), `on-tool-return`, and `llm-activated` (the model explicitly requests it, e.g. calling a `read_skill()`-style function). Every fact, instruction, or skill in the system declares its slot and trigger up front, so at runtime you know not just what was sent to the model but *why* it was sent.

**Interviewer:** Why go to the trouble of a typed model instead of just logging the full prompt on every call? Isn't that the same information?

**Candidate:** It has the same *content* but not the same *structure*, and the structure is what makes it debuggable rather than just archived. A logged prompt string tells you what the model saw once; it doesn't tell you which rule let a given fact in, whether that fact is stale, or whether removing it would have changed the answer. With typed injections, a wrong answer can be walked backward as an actual causal chain: the answer read from a specific LLM call, that call's prompt was assembled from specific injections, each injection fired because of a specific trigger, and each one originated from a specific fact or rule definition. That's a graph you can traverse, not a blob you have to re-read and guess at.

**Interviewer:** Walking it backward sounds like it just relocates the guessing — how do you actually prove a specific piece of context *caused* the wrong answer, rather than just correlating with it?

**Candidate:** Ablation, not inspection. Once you've ranked suspects by influence, you remove the top suspect from the context, re-run the exact same request with the *same seed* multiple times, and count how often the answer flips. If removing one poisoned fact flips "APPROVED" to "DECLINED" in 3 out of 3 reruns, that's a causal proof, not a hunch — the same logic as an A/B test, just applied to one request's context instead of a population of users. If the answer doesn't flip, that suspect wasn't the cause no matter how suspicious it looked in the trace.

**Interviewer:** All of this tracing sits on the hot path of every LLM call. What does it cost you in latency?

**Candidate:** It shouldn't cost anything measurable if it's designed as an observability system and not inline logic. The pattern I'd use: each stage in the pipeline emits trace events onto the call stack as a side effect, but a separate dispatcher delivers those events to listeners on the next idle tick — one beat behind, never blocking the request that produced them. That's the same principle as async logging or a message queue decoupling a write from the request that triggered it; the only discipline required is that nothing in the request path ever *waits* on the trace being recorded.

**Interviewer:** Now the other half of "why did it do that" — tool selection. If the agent picks the wrong tool, how do you even find out that happened, let alone why?

**Candidate:** Two separate checks, one at design time and one at runtime, because "wrong tool" has two different causes. Design-time: lint the tool catalog itself — embed every tool's description and compute pairwise similarity, flagging any pair of tools whose descriptions are too close together to reliably disambiguate, plus anti-patterns like a description that says *what* a tool does but never *when* to call it. A lot of wrong-tool-selection bugs are really tool-description bugs that were never caught before shipping. Runtime: score the model's actual tool choice against that same embedding geometry on every call — flag a narrow margin between the chosen tool and the runner-up as a near-tie worth reviewing, separately from an outright wrong pick.

**Interviewer:** Say the static lint passes — the descriptions are fine — but you're still burning tokens on 40 tools most turns don't need. What's the fix, and does it cost you anything?

**Candidate:** Demand-driven exposure instead of a static always-loaded catalog: tools attach to `llm-activated` triggers — the model has to explicitly unlock a skill (something like calling `read_skill('refunds')`) before that skill's tools even enter the catalog for the next turn. Turn 1 might expose one general tool; turn 4, once the model has scoped into "this is a refund request," exposes five refund-specific ones. The real tension this creates is with prompt caching: caching is a prefix match, and if the tool list is part of that prefix and it changes every turn, you'd expect to invalidate the cache constantly. The mitigation is to place cache markers per injection based on how stable its trigger is — `always` content is the most cache-friendly and sits earliest in the prefix, `llm-activated` content is the least stable and sits latest — so the frequently-changing tool set only invalidates the small suffix of the prompt, not the whole thing.

**Interviewer:** You mentioned this needs to survive compliance review too, not just debugging. What does that actually require that the tracing above doesn't already give you?

**Candidate:** Tamper-evidence, which is a different property from traceability. The trace I described proves *why* a decision happened; a regulator or auditor needs proof that the trace itself wasn't edited after the fact. That's a hash chain over the typed events — each record's hash includes the previous record's hash, so altering any historical entry breaks every hash after it, detectable by recomputing the chain and checking it matches. I'd be precise about the guarantee, though: hash-chained is **tamper-evident**, not **tamper-proof** — it tells you a breach happened, it doesn't prevent someone with write access from breaking the chain and claiming corruption. For real non-repudiation you need both ends of the chain anchored somewhere outside your own control — a separate write-once store or an external signed log — the same reason a payment ledger keeps an external reconciliation record and doesn't just trust its own database.

**Interviewer:** Last one. This entire design is a debugging and audit layer. What does it *not* solve, and what's still on you?

**Candidate:** Tracing tells you why an answer happened; it never tells you whether the answer was *right*. Those are separate systems, and conflating them is the trap — I could have perfect causal tracing on a badly-calibrated escalation threshold and still ship confidently-wrong answers all day, fully explained and still wrong. This is why the design has to sit alongside, not instead of, the standing eval framework from `Designing an Evaluation Framework for a Customer-Support Chatbot`: the eval set is what tells you the system is correct; the tracing is what tells you, once something's already gone wrong, exactly which piece of context did it. Skipping the eval layer because "we have great observability now" is the mistake this whole design invites if you don't name the boundary explicitly.

*Grounded in a real open-source implementation of this pattern — `agentfootprint`, built on `footprintjs` by Sanjay Krishna Anbalagan (MIT-licensed, github.com/footprintjs) — which implements the injection primitive, the dynamic-recomposition loop, ablation-based root-causing, and the hash-chained audit export described above.*

---

## Drill 6: Designing a Dead-Link Detection and Cleanup System for AEO (Answer Engine Optimization)

**Interviewer:** Your company's content gets cited by AI answer engines — ChatGPT, Perplexity, Google's AI Overviews — pulling from your site to answer user questions. Someone on the SEO team says a chunk of your citations have started pointing at dead pages. Why does that matter enough to build a system around, and what are you actually building?

**Candidate:** It matters because AEO (Answer Engine Optimization — SEO's successor for a world where the primary "search result" is a generated answer, not a ranked list of blue links) depends on trust signals the crawler can check cheaply, and a broken link is one of the cheapest ones to check. If an answer engine's crawler hits a 404 on a page you're citing from, or a page you link out to as a source, that's a concrete, unambiguous signal that the content is stale or unmaintained — and it can get your page demoted or dropped from the citation pool entirely, independent of how good the actual content is. The system to build is a recurring crawl of your own site's link graph — every internal link and every outbound citation — that classifies each one as alive, dead, or flaky, and routes the dead ones to whoever owns the fix, before an external crawler finds them first.

**Interviewer:** Walk me through the actual pipeline. What's step one?

**Candidate:** Step one is building the link graph itself — crawl your own site (or read it from your CMS/sitemap if one exists) and extract every `<a href>`, recording source page, target URL, and whether the target is internal or external. That graph is the unit everything else operates on; you can't check links you haven't enumerated.

**Interviewer:** And then you just HTTP-request every URL and see what comes back?

**Candidate:** That's the naive version, and it breaks in two specific ways at scale. First, volume — a site with tens of thousands of pages times several links each is not something you check serially; you need a worker pool doing concurrent requests, rate-limited **per target domain**, not globally, because hammering one external site with hundreds of parallel requests gets you rate-limited or blocked, which shows up as false dead-link positives that are actually you being throttled. Second, transient failures — a site that's down for 30 seconds during your crawl window is not the same as a site that's actually gone, and treating a single timeout as "dead" generates noisy, wrong tickets. The fix for the second problem is a status model with more than two states: `200`-class is alive, a `404`/`410` is a strong dead signal, and anything else (timeout, 5xx, connection refused) is `flaky` until it fails the *same* check on a retry with backoff across a longer window — hours, not seconds — before being promoted to dead.

**Interviewer:** Say you've correctly identified 4,000 dead links across the site. You can't fix all of them today. How do you decide what to act on first?

**Candidate:** Prioritize by impact, not by count. Two signals matter most: how much traffic or citation volume the *source* page carries (a dead outbound link on your highest-traffic article matters more than one on a page nobody reads), and whether the dead link is internal or external — an internal dead link is a broken user journey on your own site and is unambiguously your bug to fix; an external dead link is a citation to someone else's page that went away, and the fix there is usually "find a replacement source or remove the citation," not "fix the other site." Rank the backlog by source-page traffic × link type, and that ordering is what goes to whoever triages it.

**Interviewer:** Can any of this be auto-fixed, or does everything need a human?

**Candidate:** Some of it, carefully. If an internal link is dead because the target page was renamed or moved and you have redirect records, that's a safe auto-fix — rewrite the link to the new URL, no judgment call involved. If an external citation is dead, auto-fixing is much riskier: you could auto-replace it with a search for a similar page, but silently swapping in a different source changes what your content is claiming to cite, which is a content-accuracy decision, not a plumbing one. That case should be flagged for a human to pick the replacement or decide to just remove the citation, not resolved automatically.

**Interviewer:** How do you make sure this doesn't become a one-time cleanup that quietly rots again in six months?

**Candidate:** Run it as a recurring job, not a one-off script — daily or weekly depending on site size, re-checking the full link graph and diffing against the last run so you only alert on *new* dead links rather than re-reporting the same backlog every time. That diff is also what feeds a simple health metric (percentage of links currently alive, trend over time) that can sit on a dashboard the content team actually looks at, which is what turns this from an engineering side project into something that stays maintained.

---

## How to run these yourself

Reading the transcripts above teaches the moves. It does not substitute for doing this cold. The actual drill: pick a scenario not written here — a video recommendation system at 1 million daily active users, an on-call alerting pipeline that can't miss a real incident, an ad-ranking system that has to stay fair across demographics — write down one concrete number (concurrency, TPS, latency budget, data volume), and interrogate your own design with the same pattern used above: convert the headline number into a load number, break the latency budget into pipeline stages, find what breaks first, push to 10x, and ask "how do I know accuracy holds, not just latency" before you're satisfied. Better yet, get someone else to ask the follow-ups — the entire value of this format is that you don't get to pick which question comes next.

## Common pitfalls in this format
- **If you answer the opening question and stop, you've answered a different, easier question than the one being asked** — "design a RAG system" and "design a RAG system that holds up at 10,000 concurrent users, provably" are different rounds, and only the second one is what's actually being tested here.
- **If you give a number without showing how you got it, it reads as guessed** — "500 QPS" is a strong answer; "10,000 concurrent users" restated as if that were already a load number is not. Show the conversion (Little's Law, or an explicit stated assumption), every time.
- **If every follow-up answer is about latency and none are about how you'd know accuracy held, you're leaving half the question unanswered** — the two named scenarios in this file ("...for accuracy and latency", "...how do you handle this and that") are both explicitly two-sided, and interviewers ask the accuracy half specifically because most candidates default straight to the latency half and stop.
- **If you can't name what breaks first at 10x scale, you've designed for the number you were given and nothing past it** — every drill above ends by pushing past the stated scale specifically because that's where a memorized-sounding answer runs out and real reasoning has to take over.
