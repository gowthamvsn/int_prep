# System Design — Deep Interrogation Drills

`system-design-prep.md` teaches the frameworks. `real-world-incidents.md` shows what actually broke in the real world, with real incidents. This file is different. It's what the round actually feels like.

An interviewer hands you one scenario, with one real number in it. Then they don't stop. Every answer gets a follow-up. Every number gets a "why that number, specifically." Every design gets pushed to 10x scale, to see what breaks first.

This is a transcript of that pressure, worked through with real arithmetic. Read it once to learn the moves. Then close the file and run the next one cold, out loud, against a clock.

---

## Drill 1: RAG at 10,000 Concurrent Users — Accuracy AND Latency

**Interviewer:** Your RAG system needs to serve 10,000 concurrent users. Design it for both accuracy and latency.

**Candidate:** First I need to turn "10,000 concurrent users" into an actual load number. By itself, it isn't a QPS figure.

Say the average user sends a query every 20 seconds while active — typical for a chat product: read the answer, think, ask again. Little's Law gives me the QPS:

```
QPS ≈ concurrent_users / avg_seconds_between_requests = 10,000 / 20 = 500 QPS
```

I'd confirm that 20-second assumption with real product data, not guess it. But I want a number to design against, not stay abstract.

**Interviewer:** Fine, 500 QPS. What's your latency budget?

**Candidate:** I'd split this into two numbers, not one. Users perceive them differently.

1. **Time-to-first-token** — how fast the answer starts appearing.
2. **Total time** — how fast it finishes.

For a chat surface, I'd target **p95 time-to-first-token under about 1 second**. Total time can run a few seconds, as long as it's streaming.

`Designing a Search + LLM Product` makes this same point: users measure the first number, not the second, as long as the stream doesn't stall.

**Interviewer:** Break down that 1-second budget across your actual pipeline stages.

**Candidate:** Roughly:
- Query embedding — ~10-30ms, batched on GPU.
- ANN vector search — ~5-50ms, depending on index size and `ef_search`.
- Optional cross-encoder reranking — ~50-200ms, for a top-20-50 candidate set.
- Then the first LLM token.

If I want first-token under a second, and the LLM call itself needs several hundred milliseconds just to start generating, retrieval and reranking together need to fit in roughly **150-300ms**, combined. That's the number that actually constrains my architecture — not the 1-second headline.

**Interviewer:** At 500 QPS, what breaks first — the vector index or the reranker?

**Candidate:** The reranker, almost certainly, and by a wide margin.

A single-node ANN index — HNSW, in-memory — can typically serve thousands of read QPS if it fits in RAM. 500 QPS of read-only nearest-neighbor lookups usually isn't the bottleneck.

But say I rerank the top 20-50 candidates per query with a cross-encoder:
```
500 × 20 = 10,000 transformer forward passes/sec   (top-20)
500 × 50 = 25,000 transformer forward passes/sec   (top-50)
```
That's real GPU compute. It's the first thing I'd expect to saturate.

**Interviewer:** So what do you actually do about that?

**Candidate:** Several levers, in the order I'd reach for them:

1. Shrink the reranked candidate set — top-20 instead of top-50, if recall holds. Measured, not assumed.
2. Dynamically batch reranking requests across concurrent users into one GPU forward pass. Same continuous-batching idea from `Designing an LLM Inference System at Scale`, just applied to the cross-encoder instead of the generator.
3. Horizontally scale reranker replicas, autoscaling on queue depth rather than raw GPU utilization — for the same reason that section argues against utilization-based scaling.
4. Semantic-cache rerank results for repeated or near-duplicate queries.

**Interviewer:** You mentioned caching. How much does that actually save you at 10,000 concurrent users — give me a real answer, not "it helps."

**Candidate:** Honestly, it depends entirely on query repetition rate. I wouldn't assume a number — I'd measure it.

A narrow-domain FAQ-style support bot might see 30-40% of queries be near-duplicates of something asked in the last hour. Caching is a major lever there.

An open-ended research assistant might see under 5% repetition. Caching barely moves the needle there, and I shouldn't design around it.

This is a "check before you build" item, not a "everyone knows caching helps" assumption.

**Interviewer:** Now the accuracy half. How do you know your retrieval accuracy at 10,000 concurrent users under load is the same as what you measured in your offline eval with one request at a time?

**Candidate:** This is the actual trap in this question, and it's easy to miss.

Under load, teams often silently trade accuracy for speed: lowering `ef_search` on the ANN index, shrinking the reranked candidate set, or skipping reranking entirely once queues back up. Then nobody re-measures whether recall@k actually held at the degraded settings.

The fix: run the exact same standing retrieval-quality eval from `Designing a RAG System for Internal Documents`, but **under simulated load**, not just quiet lab conditions. Alert if live-measured recall or precision drops as load-shedding kicks in.

A system can be fast and wrong at scale, while looking identical to fast-and-right in a demo.

**Interviewer:** So when you're over capacity, what's your actual policy — slower answers, worse answers, or dropped requests? Pick one and defend it.

**Candidate:** I'd make this an explicit, ordered policy, not something that happens by accident.

1. Horizontal autoscaling with a warm minimum pool. Never scale from zero — that's the inference-at-scale section's cold-start point.
2. Queueing with a bounded max wait, and an explicit backpressure signal (429) to the client, instead of silently degrading quality.
3. Only if I must degrade gracefully — a **named, monitored** step-down (like an `ef_search` reduction), with an accuracy-floor alarm attached.

The failure mode I'm avoiding: accuracy quietly becoming the invisible pressure-relief valve nobody's watching.

**Interviewer:** 10,000 users becomes 100,000 next quarter. What's the first thing that stops scaling linearly?

**Candidate:** The vector index.

A single-node in-memory index eventually doesn't fit in RAM, and can't absorb more QPS on one machine. I'd need to shard it: partition by namespace or hash, fan the query out to every shard, merge the top-k results. That adds real fan-out latency and coordination complexity that didn't exist at 10,000 users.

The LLM generation tier scales more predictably — roughly linearly in GPU count. But it becomes the dominant *cost* line, since it's by far the most compute-expensive stage per request.

**Interviewer:** Last one. Accuracy and latency are in real tension here. Name the actual dial, and tell me who should be turning it.

**Candidate:** The literal dials are `ef_search` and reranked-candidate-count. Turning them down buys latency and costs recall, measurably.

But which setting is "correct" isn't purely an engineering call. It's the same error-cost calibration from `problem-formulation-framework.md`: what does a slightly-worse retrieved answer cost the business, versus what does a slower response cost in abandoned sessions?

I'd bring the measured tradeoff curve to product, not decide it unilaterally.

---

## Drill 2: Continuous Training on Continuous Input

**Interviewer:** Your system receives a continuous stream of new labeled data. How do you continuously train on it?

**Candidate:** First I need to pin down what "continuously" actually means architecturally. Three very different things hide behind that word:

1. **True online learning** — update weights after every single example.
2. **Continuous micro-batch retraining** — retrain on a rolling window every N minutes or hours.
3. **Continuous adapter fine-tuning** — small LoRA-style updates on a frozen base, on a recurring cadence.

I'd ask which is intended. But my default recommendation is the second or third, not the first — and I'd say why unprompted.

**Interviewer:** Why not true online, per-example SGD? It sounds like the most "continuous" answer.

**Candidate:** Two reasons.

Statistically, the gradient from one example is extremely noisy relative to a model of any real size. There's no averaging to smooth out a bad update. One mislabeled or adversarial example can meaningfully move the weights, with nothing to catch it.

Operationally, there's no natural batch boundary at which to *evaluate before committing*. The update ships the instant it's computed, with no gate.

That's structurally the same failure as Microsoft's Tay in `real-world-incidents.md`: a system that updated live on unfiltered input got steered somewhere bad within hours. Nothing stood between "new input arrived" and "the model changed because of it."

**Interviewer:** Fine — micro-batch retraining, then. How often, and on how much data, concretely?

**Candidate:** Sized for a stable gradient estimate, not for "as often as possible."

Say 50,000 new labeled examples arrive per day. Instead of retraining on every trickle, I'd accumulate a window — every 2-4 hours, or every 5,000-10,000 new examples, whichever comes first. Large enough that an update reflects a real shift in the data, not noise in a tiny batch.

And critically: every window gets mixed with a sampled slice of historical data. Never trained on the new window alone.

**Interviewer:** Why replay old data? Isn't the entire point of continuous training to adapt to what's new right now?

**Candidate:** Because training only on the newest window is exactly how you get catastrophic forgetting.

The model overfits to whatever's most recent — today's traffic mix, today's slang, today's adversarial probing pattern — and quietly loses general behavior that just isn't represented in the last few hours of data. Same mechanism as the catastrophic-forgetting pitfall in `Designing a Fine-Tuning Pipeline for a 70B Model`, just recurring instead of one-shot.

Concretely: maintain a replay buffer — a stratified sample across time and segments — and mix it into every training window at a tuned ratio. Something like 70% new data / 30% replay as a starting point, validated against a general-capability eval rather than picked arbitrarily.

**Interviewer:** How do you know a given update didn't make things worse, before it's serving real traffic?

**Candidate:** It never auto-deploys straight to production.

Every update goes through the same shipping gate a normal release would: score it against a standing held-out eval set — one that was never in the training window or the replay buffer — plus a general-capability check. Only promote the new checkpoint if it doesn't regress past a defined threshold.

If it does, the update is discarded, not patched. The previous checkpoint keeps serving. It's a continuous canary gate, not a one-time launch gate.

**Interviewer:** The world is changing continuously too, which is the whole premise here. How do you know that held-out eval set itself hasn't gone stale?

**Candidate:** It can, and this is the exact eval-set-decay point from `Designing a Search + LLM Product`.

A question that was correctly answered by the eval set six months ago can be wrong now, if the underlying task or world shifted underneath it.

So the eval set needs its own re-validation cadence — separate and slower than the training cadence — with a human check before a failing eval query gets counted as a real regression, rather than the world having moved.

**Interviewer:** What about the labels themselves? A continuous input stream rarely arrives with instant, clean ground truth.

**Candidate:** Right, and I'd name that lag explicitly rather than assume it away.

If the true label arrives a day after the prediction — did the recommendation actually convert, did the flagged transaction actually turn out fraudulent — the training window has to be bounded by the **confirmed-label** window, not the raw-input-arrival window.

Where a fast, noisy proxy label exists — click-through as a stand-in for eventual conversion — I'd use it for a fast signal, but weight it lower than the slower, clean label. Not blend the two as if they were equally trustworthy.

**Interviewer:** Roughly, what does retraining every 2-4 hours cost compared to a normal weekly batch retrain? Put a number on it.

**Candidate:** Using the QLoRA arithmetic from the fine-tuning-pipeline scenario as a reference point — roughly $30-55 and ~14 GPU-hours per full adapter training run:

```
168 hours (one week) / 4 hours = 42x more frequent
```

Retraining every 4 hours instead of weekly is about **42x more frequent**. Cost has to come down close to proportionally, or the budget explodes.

That's the real argument for continuous updates being small adapter fine-tunes on a frozen base, rather than continuous full retrains: the frozen base makes each update cheap enough to actually run 42x more often, without 42x the spend.

**Interviewer:** Last one. Six months of continuous updates in — is the model actually better than where it started, or has it quietly drifted somewhere you wouldn't have chosen on purpose?

**Candidate:** This is where the feedback-loop self-reinforcement warning from `Designing Production Model Monitoring` matters most. It compounds continuously instead of just once.

If some of the "new" labels are downstream outcomes the model's own decisions influenced, training on them closes a loop. That loop can reinforce the model's existing blind spots, instead of correcting them.

The fix, applied on a rolling basis:
1. Deliberately sample and label cases *outside* what the model is currently confident about — not just whatever flows back in naturally.
2. Track a long-horizon general-capability metric across the full six months, not just each window's own eval.

That second piece is what makes a slow drift away from the original goal visible, even though no single update ever looked like a regression on its own.

---

## Drill 3: Real-Time Fraud Detection — 5,000 Transactions/Second, Sub-50ms Decision Budget

**Interviewer:** You need to approve or decline 5,000 transactions a second, each within 50 milliseconds. Design it.

**Candidate:** 50ms is the number that eliminates most of the usual toolkit immediately.

That's not enough time for an LLM call. It's tight even for a moderately sized neural net, once you count network hops.

I'd default to a **fast classical model** on the hot path — gradient-boosted trees, or a small model like Isolation Forest. It's a deterministic, low-latency forward pass, not a generative call.

This is the same real decision FinSight made in `system-design-prep.md`: fraud detection ran on Isolation Forest specifically because it sat on the critical path and couldn't absorb LLM-scale latency.

**Interviewer:** 50ms, broken down — where does it actually go?

**Candidate:** Roughly:
- Feature lookup — recent transaction history, account velocity. Often the biggest unknown, since it may mean a database round-trip.
- Feature computation — a few milliseconds if precomputed or cached, much more if computed live.
- Model inference — sub-millisecond to a few ms for a small tree ensemble.
- Network and serialization overhead, on both sides.

If feature lookup requires a live database join, that's very likely the majority of the 50ms budget — not the model itself. That's exactly the pattern the real-time-prediction framework in `system-design-prep.md` names: the bottleneck in a "real-time" system is almost always feature retrieval, not model inference.

**Interviewer:** So how do you make feature lookup fast enough?

**Candidate:** Precompute and cache what can be precomputed. Rolling aggregates like "transactions in the last hour" get maintained continuously in a low-latency store — Redis or similar — rather than queried fresh per request. The hot path reads a small number of already-computed values, instead of joining raw history live.

This is the online/offline feature-consistency point from the ML-system-design framework again: whatever's precomputed offline-adjacent has to match, exactly, what a training-time feature pipeline would have computed. Otherwise I get training-serving skew on top of everything else.

**Interviewer:** At 5,000 TPS, what happens when your feature store is briefly unavailable?

**Candidate:** That has to have a defined fallback, not an accident. Per the real-time-prediction framework's point on this exact failure, the options are:

1. A cached last-known-good feature value.
2. A conservative default — treat missing velocity features as higher-risk, not lower-risk, since failing open on a fraud check is the worse direction to fail.
3. In the extreme, a much simpler rule-based fallback that can run with no feature store at all.

What I wouldn't accept: the system silently approving every transaction because a dependency timed out. That's fail-open on the exact system where fail-open is most dangerous.

**Interviewer:** Your model's accuracy looks great offline. Three weeks after launch, fraud losses are creeping up even though nothing was redeployed. What's happening?

**Candidate:** Concept drift, almost certainly. Fraud patterns actively adapt to whatever the current model blocks. The input-to-correct-output relationship shifts, even though the model and code haven't changed.

I'd expect input-distribution drift on transaction features to be the earliest signal — well before the outcome metric moves. Confirmed fraud losses lag by days to weeks while a case gets investigated. Same fastest-to-slowest drift layering as `Designing Production Model Monitoring`.

And I'd check it's not a feedback-loop problem specifically: if I only ever get confirmed-fraud labels on transactions the model already flagged, I have zero ground truth on the fraud my own model is currently missing. That blind spot compounds every week it isn't corrected.

**Interviewer:** What would you change if this needs to go from 5,000 TPS to 50,000?

**Candidate:** The feature store and the model-serving tier both need to scale roughly horizontally with load — more replicas behind a load balancer, cache sharded rather than single-node.

But the number I'd actually watch is p99, not average latency. At 10x the traffic, a queueing tail that was invisible at 5,000 TPS can start blowing the 50ms budget for a small but real fraction of transactions. "Average latency is fine" is exactly the kind of aggregate number that hides a real problem — the same segment-monitoring point that shows up everywhere else in this hub.

---

## Drill 4: Operating a Production RAG/LLM Pipeline — Pipelines, Integrity, Hosting, Hallucination, and the Daily Job

This one's shaped differently from the first three. It's not one number pushed to its limit. It's the full width of "you own this system now, not just the design doc."

Six movements: building the pipeline, moving data between its stages with integrity intact, hosting it highly-available, stopping it from hallucinating or half-answering, knowing when to intervene, and shipping a new feature without anyone noticing. `production-ml-practice.md` and `mlops-practice.md` go deeper on rollout mechanics and the pipeline loop, respectively. This drill is where they get pulled together into one operational answer.

### Movement 1 — Building the pipeline and moving data between stages

**Interviewer:** Walk me through the actual pipeline, stage by stage — not the model, the plumbing around it.

**Candidate:** For a RAG system specifically:

```
ingestion → parsing/chunking → embedding → upsert into vector index
   → serving (retrieve → rerank → generate) → logging → monitoring → feedback capture
```

New or changed documents land from a CMS, an S3 bucket, or a ticketing system.

I'd orchestrate the ingestion-through-upsert half as a DAG — Airflow, Dagster, or a simpler queue-based worker if the volume doesn't justify a scheduler. The point of a DAG here isn't sophistication. It's that each stage has an explicit dependency and can be retried independently, without re-running everything upstream of it.

**Interviewer:** How does data actually move from the chunking stage to the embedding stage — same process, a queue, a database write?

**Candidate:** I'd put a durable handoff between any two stages that can fail independently and don't need to be synchronous. A message queue — Kafka, SQS — or a staging table with a status column (`pending → embedded → indexed`). Not a direct in-memory function call.

The reason: if the embedding stage is down for 10 minutes, chunked documents from ingestion shouldn't be lost or silently skipped. They queue up and get processed once the embedding service is back. A direct synchronous call couples the two stages' uptime together, for no reason.

**Interviewer:** What stops a document from silently getting processed twice, or not at all, if a worker crashes mid-stage?

**Candidate:** Idempotency and an explicit status field. Not "hope the crash doesn't happen at a bad moment."

1. Each document gets a stable ID.
2. A stage claims a document — status goes to `processing`, with a claim timestamp.
3. The stage only marks it `done` after the write to the next stage succeeds.
4. If a worker crashes mid-claim, a watchdog re-queues anything stuck in `processing` past a timeout.

And the embedding upsert itself should be idempotent — upsert by document-chunk ID, not insert — so a retried write after a crash doesn't create a duplicate chunk in the index.

### Movement 2 — Data integrity, in simple terms

**Interviewer:** How do you know the data moving through this pipeline is actually correct, not just "didn't error out"?

**Candidate:** "Didn't error" and "correct" are different checks. I'd gate on both.

At each stage boundary:
1. A schema/shape check — does this chunk have the fields I expect (text, source ID, permission metadata) before it's allowed into the next stage.
2. A sanity check on volume — did today's ingestion job process roughly the expected number of documents, or did it silently process 3% of them because a source connector broke.
3. A content check where it's cheap — a chunk with zero characters, or one that's 99% whitespace, is a parsing bug, not valid data.

None of these require ML. They're the same boring data-engineering discipline that catches most real pipeline failures before they ever reach the model.

**Interviewer:** Give me a concrete number — how would you actually alert on "ingestion silently broke"?

**Candidate:** Track daily ingested-document count as a time series. Alert on a percentage deviation from a rolling baseline.

Example: if the 7-day trailing average is ~2,000 documents/day, and today's run processed under 500 — a 75%+ drop — that's a page, not a log line.

A silent 75% drop is far more dangerous than a hard failure. A hard failure is visible immediately. A silent partial failure looks like a normal, boring day.

### Movement 3 — Hosting and high availability

**Interviewer:** How do you actually host this so it survives a bad day?

**Candidate:** Every stateless tier — retrieval API, reranker, generation gateway — runs as multiple replicas behind a load balancer, spread across at least two availability zones. So a single AZ failure doesn't take the whole system down. Standard horizontal redundancy.

The stateful tier — the vector index, if it's not already a managed service — needs its own replication story. Either a managed vector DB with built-in replication, or a self-hosted index with a hot standby. "Just restart it" doesn't work for something holding an in-memory index that takes minutes to rebuild.

**Interviewer:** What's your actual failover story if the primary vector-index replica goes down mid-traffic?

**Candidate:** Health checks on each replica feed the load balancer's routing decision. Traffic stops going to an unhealthy node within seconds, not after a human notices.

And there's a defined degraded mode for the gap: if all retrieval replicas are briefly unavailable, the system should fail toward "no context, refuse to answer or say so" — not "generate an ungrounded answer as if retrieval had succeeded." Silently dropping the grounding step is a correctness failure disguised as an availability one.

**Interviewer:** What's your actual uptime target, and what does that number mean in practice?

**Candidate:** I'd want that stated as an explicit SLO, not left implicit. Say 99.9% — that's about 43 minutes of allowed downtime a month.

That number should directly drive the redundancy decisions above. Single-AZ hosting can't hit 99.9%, no matter how good the code is — an AZ-level outage alone burns the whole monthly budget in one event.

### Movement 4 — Stopping hallucination and partial responses

**Interviewer:** How do you actually stop this thing from hallucinating?

**Candidate:** Layered, not one trick.

1. **Grounding.** The system prompt instructs the model to answer only from retrieved context, and to explicitly say it doesn't know rather than fill the gap. This alone doesn't guarantee anything — the model can still ignore the instruction.
2. **Verification.** A post-hoc entailment check: does the retrieved context actually support the generated claim, the same span-verification idea from `Designing a Search + LLM Product`. This catches cases where the model asserted something the context doesn't support.
3. **Measurement.** Faithfulness and groundedness get tracked as their own metric, on a standing eval set — separate from "does the answer sound good." You can't fix a rate you're not measuring.

**Interviewer:** And a partial or cut-off response — how do you handle that?

**Candidate:** Detect it structurally. Don't just hope the stream completes.

If the response is meant to be structured — JSON, a numbered list with a known expected shape — validate the shape after the stream ends, and reject or retry if it's malformed or truncated.

If it's free text, use a hard generation timeout with a documented behavior on trip. Either retry once with a tighter token budget, or return what streamed so far plus an explicit "response was cut short" signal to the client. Never silently present a truncated answer as if it were complete.

The client-side contract matters as much as the server-side generation: the UI needs to be able to tell a genuinely finished answer from one that got cut off. That means the API needs to say so explicitly, rather than the client guessing from "the stream stopped."

**Interviewer:** What actually causes a partial response in production, most commonly?

**Candidate:** Usually one of three things:

1. Generation hit `max_tokens` before finishing. A token-budget-sizing problem, fixable by estimating the needed length upfront and either warning or auto-continuing.
2. A downstream timeout fired mid-stream. A load or infra problem, not a model problem.
3. The client disconnected, and the server didn't detect it. This wastes GPU compute generating tokens nobody receives — worth explicitly canceling generation on client disconnect, both for cost and so correctness telemetry doesn't misattribute that as a real completed response.

### Movement 5 — What to actually watch, and when it means "intervene now"

**Interviewer:** Give me the actual short list of metrics you'd watch, and what number on each one means "page someone" versus "note it and move on."

**Candidate:** Layered fastest-to-slowest, per the monitoring framework in `system-design-prep.md`:

1. **p99 latency**, against the SLO. Page if p99 breaches budget for more than a few consecutive minutes. A brief spike is often just a GC pause or a cold replica. A sustained breach means real capacity trouble.
2. **Error rate.** Page above some absolute threshold — e.g. >1% of requests erroring — tuned to the product's tolerance.
3. **Retrieval-score drift** — a rising share of queries with no result clearing the relevance floor. Often the earliest sign something upstream broke, before any outcome data exists. Flag for investigation, not necessarily an immediate page.
4. **Faithfulness/groundedness score**, on the standing eval, run continuously — not just at release. A sustained drop here is a page, because it means the system is actively giving unsupported answers to real users.
5. **Escalation/refusal rate**, as a guardrail metric. Both directions are bad. A spike often means the knowledge base went stale. A drop toward zero can mean the safety gate stopped firing — worse, and easier to miss.

**Interviewer:** Which of those is the one you'd actually wake up for at 3am?

**Candidate:** Sustained faithfulness drop, or a p99/error-rate SLO breach. Those two mean the system is actively serving users something wrong, or not serving them at all, right now.

Retrieval-score drift and escalation-rate movement are real signals. But they usually tell you something's *heading* toward a problem — a next-morning investigation, not a page. That's the real distinction: "this is currently hurting a user" versus "this predicts a future problem" is what actually separates a page from a ticket.

### Movement 6 — The daily job, and shipping a new feature without anyone noticing

**Interviewer:** Forget an incident — what does the boring, uneventful daily routine on this system actually look like?

**Candidate:** A short standing checklist. Most days, it confirms nothing's wrong rather than fixing something:

1. Review the overnight dashboard — latency, error rate, cost, faithfulness score, escalation rate. The metrics from Movement 5, glanced at, not re-derived.
2. Check the ingestion volume against baseline — Movement 2's silent-failure check.
3. Spot-check a small sample of flagged/escalated conversations from the last 24 hours. The non-escalated-sampling discipline from the support-chatbot eval design — catching confident-wrong answers nobody else would surface.
4. Confirm the standing canary/eval-question set still passes.

Most days this is 15 minutes that confirms the system is healthy. The value isn't in what it usually finds — it's that skipping it is exactly how a slow problem (a stale knowledge base, a creeping cost trend) goes three weeks before anyone notices, instead of one day.

**Interviewer:** Now ship a new feature — say, a new retrieval source — into this system without disrupting current users. Walk me through it.

**Candidate:** Staged, each stage gated on the previous one holding. Per the canary framework in `production-ml-practice.md`:

1. **Shadow mode.** The new retrieval source runs alongside the existing pipeline for every real request. Its results get logged and scored, but never shown to a user. This tells me if it retrieves well against real traffic, with zero user-facing risk.
2. **A small canary.** Route a small slice of real traffic — 5% — to actually use the new source's output. Watch the same guardrail metrics from Movement 5, specifically on that slice, compared against the 95% control.
3. **Widen gradually**, only as each step holds — 5% → 25% → 100%. Automatic or fast-manual rollback if faithfulness, latency, or error rate on the canary slice regresses past a threshold at any step.

The feature flag controlling the percentage is the actual mechanism. "Roll back" means flipping a config value, not a deploy — which is what makes each step genuinely low-risk and fast to undo.

**Interviewer:** What's the one thing that undermines this whole staged rollout if you get it wrong?

**Candidate:** Picking guardrail metrics for the canary slice that are too slow or too aggregate to actually catch a regression before you've widened past it.

If the metric you're watching only moves after outcome data trickles in over days, a same-day ramp from 5% to 100% will have already fully shipped before the signal arrives.

The canary is only as good as the fastest metric layered under it. That's exactly why Movement 5's fastest-to-slowest signal stack matters here too, not just for steady-state monitoring.

---

## Drill 5: Designing an Explainable, Debuggable AI Agent System

### Plain-English primer, with one real example all the way through

**The business problem, in one sentence.** A company builds an AI agent to do real work — approve refunds, answer account questions, route support tickets. Every so often it does something wrong, and nobody can say why. That's a business risk — angry customers, compliance exposure, money out the door — long before it's an engineering annoyance.

**A concrete story.** Say you run a refunds agent for an online store. Policy: refunds are only allowed within 30 days of purchase.

One day, the agent approves a refund for an order placed **47 days ago**. A clear policy violation. Nobody told it to break the rule. The code is fine. The servers are fine. So what happened?

Somewhere in the system, a fact got written for VIP customers — something like *"VIP customers get extended refund windows"* — meant for a different, narrower case. That fact got pulled into this customer's conversation because it looked relevant. The model treated it as permission to override the 30-day rule.

Nothing crashed. Nothing errored. The system just quietly did the wrong thing, confidently, in writing, to a real customer.

**Why normal logging doesn't catch this.** A normal log tells you "refund request → refund approved." It does not tell you "approved *because* the model was shown a VIP fact that didn't apply here."

The missing information isn't a bug in the code. It's a fact about *what the AI was shown right before it decided*. That's a fundamentally different thing to record than a stack trace or an error code. Most logging systems were never built to capture it.

**The engineering fix, in plain words.** Instead of building one big block of text (the "prompt") and hoping it's right, you label every single fact, instruction, and tool the AI can see with three simple things:
1. **Where it goes** — is this a background instruction, part of the conversation, or a tool the AI can use?
2. **When it's allowed to show up** — always, only under some condition, only after a certain step, or only when the AI specifically asks for it?
3. **Whether it's safe to reuse or cache**, so you're not paying full price to re-send the same stable instructions on every single message.

Do that, and every answer the AI gives can be traced backward, like a paper trail:
- *This answer* came from *this exact text the model saw*.
- That text came from *these specific facts*.
- Each fact fired because of *this specific rule*.
- Each rule was written by *this specific person or system*, at some point in the past.

That backward trail is the entire fix. It turns "we have no idea why it did that" into "here is the exact fact that caused it."

**Proving it, not just suspecting it.** Finding the suspicious VIP fact isn't proof it caused the bad refund. Correlation isn't causation — same as in any other kind of debugging.

The real proof: **remove that one fact, run the exact same request again a few times, and see if the answer changes.**

If "APPROVED" flips to "DECLINED" every time you remove it, and stays "APPROVED" when you remove some other, unrelated fact instead, you've *proven* which fact was the cause. Not just eyeballed a suspect.

**The tool-picking version of the same problem, with a simpler analogy.** Imagine a call-center rep's screen has ten buttons that all sound almost the same: "Issue Refund," "Process Return," "Cancel Order," "Reverse Charge." Once in a while the rep clicks the wrong one, because the labels are too similar.

AI agents need the same two fixes:
1. **Before you ever go live** — review the button labels, and flag any pair that's too similar to reliably tell apart. A one-time design review, done with the tool descriptions rather than a live rep.
2. **While it's running** — keep a light-touch check that flags every time the "click" was a close call between two similar-sounding buttons, even if the "right" one was ultimately picked. That way you catch confusion before it becomes a mistake.

**Business logic vs. engineering design — keeping the two straight:**

| | Business logic (the "what should happen") | Engineering design (the "how do we know / how do we build it") |
|---|---|---|
| Example | "Refunds require the order to be under 30 days old" | The typed fact/rule model that lets you trace *which* text told the AI otherwise |
| Who owns it | Product, policy, compliance | Engineering |
| What breaks it | The rule itself is wrong, outdated, or was never written down clearly | The rule was right, but the wrong text reached the model, or reached it at the wrong time |
| This drill's focus | Not really this — a wrong *policy* is a product problem | Entirely this — building the system so a wrong *outcome* is always explainable and provable, regardless of whether the policy or the plumbing was at fault |

The engineering system doesn't decide what the refund policy should be. It makes sure that whatever the policy is, when the AI breaks it, someone can find out exactly why within minutes instead of never.

### What his blog posts teach, in plain language

Beyond the code, Sanjay Krishna Anbalagan has written a series of Medium posts arguing for this way of thinking. A few worth knowing, explained simply (some are member-only/paywalled past the opening, so this is what's confirmed available plus the stated thesis of each):

- **["Your Logs Are No Longer for You"](https://medium.com/codetodeploy/your-logs-are-no-longer-for-you-d12720dea6aa)** — His analogy: doctors used to write quick shorthand notes for themselves ("pt stable, cont mgmt"). That was fine when the same doctor read their own notes later.

  Once patient care became a "handoff sport" — different doctors, shift changes, referrals — sloppy shorthand became dangerous. Medicine had to switch to structured, standardized charts anyone could pick up cold.

  His claim: software logging is having that same handoff moment right now. Except the new reader isn't a different doctor. It's an AI model that "never attended your standups" and has zero unwritten context about your system. If your logs only make sense to the engineer who wrote them, they're now failing their most important reader.

  The subtitle sums up the shift: logging goes from being a "cost center" (insurance you hope to never need) to a "product capability" (something the AI actively depends on to work correctly).

- **["The Flowchart Pattern: Making Backend Code Self-Explainable for AI"](https://medium.com/data-science-collective/the-flowchart-pattern-making-backend-code-self-explainable-for-ai-a508d779345c)** — The direct ancestor of `footprintjs`.

  His argument: traditional backend code keeps its logic private, and only gets investigated by a human when something breaks. But once an AI is expected to explain a decision — a loan denial, a fraud flag, a support routing choice — *as part of the normal answer*, "what happened inside?" stops being a rare debugging question. It becomes a routine product requirement.

  His fix: write backend logic as an explicit flowchart of steps that record their own reads, writes, and branch decisions as they run. That way the explanation gets generated *from the actual execution* — not invented after the fact by an LLM guessing from scraps of log text.

- **["Act, Answer, Recall: The Three Modes of an Agentic Web App"](https://medium.com/codetodeploy/act-answer-recall-the-three-modes-of-an-agentic-web-app-a6c232e6a91f)** — His point: teams build "an AI agent" as if it's one thing. It's secretly doing three different jobs, and each needs its own safety rules.

  1. **Act** — "book this for me." Needs transactional safety, since a mistake here does something real and possibly irreversible.
  2. **Answer** — "what's in my account?" Needs to be grounded in real, current data, not a plausible-sounding guess.
  3. **Recall** — "why did you just do that?" Needs an honest explanation of the actual execution, not a fabricated-sounding justification.

  Because the user types all three into the same chat box, teams often build one code path for all three, and inherit the worst failure mode of each. An "Act" that fails as loosely as an "Answer" is how you get an agent that books the wrong flight *and* can't explain why.

- **["Everyone Shows What MCP Does — But Nobody Tells You What It Abstracts"](https://medium.com/data-science-collective/everyone-shows-what-mcp-does-but-nobody-tells-you-what-it-abstracts-91432a79e416)** — Relevant to the tool-selection half of this drill. His argument: most explanations of the Model Context Protocol show *what* it does — a standard way to plug tools into an AI — without naming what it's actually hiding underneath. That hidden machinery is the same context-injection and tool-exposure machinery this whole drill is about, just wrapped in a protocol so you don't have to build it yourself.

**The one-sentence version of everything above:** when an AI system has to explain itself, "why did it say that?" needs the same kind of rigor engineers already give "why did it crash?" — a trace you can walk backward, a way to prove the cause instead of guessing at it, and a clear line between "the rule was wrong" (a business call) and "the rule was applied to the wrong situation" (an engineering bug).

---

**Interviewer:** Your agent gave a customer a wrong answer — a refund it shouldn't have approved. Your logs show the request, the response, and a clean 200. Where do you even start?

**Candidate:** That's actually the core failure this question is testing for. I'd name it before proposing a fix.

Classical logging records what the *code* did. It never records what the *context* did. Here, the code path was correct. Infrastructure was healthy. The answer still came out wrong, because something in the prompt — a stale fact, a mis-scoped instruction, a poisoned retrieval result — steered the model somewhere it shouldn't have gone.

That's a third error class, alongside bugs and outages: a **contextual error**. It needs its own instrumentation, not more `print` statements around the LLM call.

**Interviewer:** So what does that instrumentation actually look like — what do you record, concretely?

**Candidate:** I'd make every piece of context injection typed and traceable, instead of just concatenated into a prompt string.

Concretely, one primitive: `injection = slot × trigger × cache`.

- **Three slots**, fixed by the LLM API surface — `system`, `messages`, `tools`.
- **Four triggers**, describing *when* something fires:
  - `always` — steering, static facts.
  - `rule` — a runtime predicate.
  - `on-tool-return`.
  - `llm-activated` — the model explicitly requests it, e.g. calling a `read_skill()`-style function.

Every fact, instruction, or skill in the system declares its slot and trigger up front. At runtime you know not just what was sent to the model, but *why* it was sent.

**Interviewer:** Why go to the trouble of a typed model instead of just logging the full prompt on every call? Isn't that the same information?

**Candidate:** It has the same *content*, but not the same *structure* — and the structure is what makes it debuggable rather than just archived.

A logged prompt string tells you what the model saw once. It doesn't tell you which rule let a given fact in, whether that fact is stale, or whether removing it would have changed the answer.

With typed injections, a wrong answer can be walked backward as an actual causal chain: the answer read from a specific LLM call, that call's prompt was assembled from specific injections, each injection fired because of a specific trigger, and each one originated from a specific fact or rule definition. That's a graph you can traverse, not a blob you have to re-read and guess at.

**Interviewer:** Walking it backward sounds like it just relocates the guessing — how do you actually prove a specific piece of context *caused* the wrong answer, rather than just correlating with it?

**Candidate:** Ablation, not inspection.

Once you've ranked suspects by influence, you remove the top suspect from the context, re-run the exact same request with the *same seed* multiple times, and count how often the answer flips.

If removing one poisoned fact flips "APPROVED" to "DECLINED" in 3 out of 3 reruns, that's a causal proof, not a hunch. Same logic as an A/B test, just applied to one request's context instead of a population of users.

If the answer doesn't flip, that suspect wasn't the cause — no matter how suspicious it looked in the trace.

**Interviewer:** All of this tracing sits on the hot path of every LLM call. What does it cost you in latency?

**Candidate:** It shouldn't cost anything measurable, if it's designed as an observability system and not inline logic.

The pattern I'd use: each stage in the pipeline emits trace events onto the call stack as a side effect. A separate dispatcher delivers those events to listeners on the next idle tick — one beat behind, never blocking the request that produced them.

That's the same principle as async logging, or a message queue decoupling a write from the request that triggered it. The only discipline required: nothing in the request path ever *waits* on the trace being recorded.

**Interviewer:** Now the other half of "why did it do that" — tool selection. If the agent picks the wrong tool, how do you even find out that happened, let alone why?

**Candidate:** Two separate checks, one at design time and one at runtime, because "wrong tool" has two different causes.

**Design-time:** lint the tool catalog itself. Embed every tool's description and compute pairwise similarity, flagging any pair of tools whose descriptions are too close together to reliably disambiguate. Also flag anti-patterns, like a description that says *what* a tool does but never *when* to call it. A lot of wrong-tool-selection bugs are really tool-description bugs that were never caught before shipping.

**Runtime:** score the model's actual tool choice against that same embedding geometry, on every call. Flag a narrow margin between the chosen tool and the runner-up as a near-tie worth reviewing, separately from an outright wrong pick.

**Interviewer:** Say the static lint passes — the descriptions are fine — but you're still burning tokens on 40 tools most turns don't need. What's the fix, and does it cost you anything?

**Candidate:** Demand-driven exposure, instead of a static always-loaded catalog.

Tools attach to `llm-activated` triggers. The model has to explicitly unlock a skill — something like calling `read_skill('refunds')` — before that skill's tools even enter the catalog for the next turn. Turn 1 might expose one general tool. Turn 4, once the model has scoped into "this is a refund request," exposes five refund-specific ones.

The real tension this creates is with prompt caching. Caching is a prefix match, and if the tool list is part of that prefix and it changes every turn, you'd expect to invalidate the cache constantly.

The mitigation: place cache markers per injection, based on how stable its trigger is. `always` content is the most cache-friendly and sits earliest in the prefix. `llm-activated` content is the least stable and sits latest. So the frequently-changing tool set only invalidates the small suffix of the prompt, not the whole thing.

**Interviewer:** You mentioned this needs to survive compliance review too, not just debugging. What does that actually require that the tracing above doesn't already give you?

**Candidate:** Tamper-evidence — a different property from traceability.

The trace I described proves *why* a decision happened. A regulator or auditor needs proof that the trace itself wasn't edited after the fact. That's a hash chain over the typed events: each record's hash includes the previous record's hash, so altering any historical entry breaks every hash after it, detectable by recomputing the chain and checking it matches.

I'd be precise about the guarantee, though: hash-chained is **tamper-evident**, not **tamper-proof**. It tells you a breach happened. It doesn't prevent someone with write access from breaking the chain and claiming corruption.

For real non-repudiation, you need both ends of the chain anchored somewhere outside your own control — a separate write-once store, or an external signed log. Same reason a payment ledger keeps an external reconciliation record, and doesn't just trust its own database.

**Interviewer:** Last one. This entire design is a debugging and audit layer. What does it *not* solve, and what's still on you?

**Candidate:** Tracing tells you why an answer happened. It never tells you whether the answer was *right*.

Those are separate systems, and conflating them is the trap. I could have perfect causal tracing on a badly-calibrated escalation threshold, and still ship confidently-wrong answers all day — fully explained, and still wrong.

This is why the design has to sit alongside, not instead of, the standing eval framework from `Designing an Evaluation Framework for a Customer-Support Chatbot`. The eval set tells you the system is correct. The tracing tells you, once something's already gone wrong, exactly which piece of context did it.

Skipping the eval layer because "we have great observability now" is the mistake this whole design invites, if you don't name the boundary explicitly.

*Grounded in a real open-source implementation of this pattern — `agentfootprint`, built on `footprintjs` by Sanjay Krishna Anbalagan (MIT-licensed, github.com/footprintjs) — which implements the injection primitive, the dynamic-recomposition loop, ablation-based root-causing, and the hash-chained audit export described above.*

---

## Drill 6: Designing a Dead-Link Detection and Cleanup System for AEO (Answer Engine Optimization)

**Interviewer:** Your company's content gets cited by AI answer engines — ChatGPT, Perplexity, Google's AI Overviews — pulling from your site to answer user questions. Someone on the SEO team says a chunk of your citations have started pointing at dead pages. Why does that matter enough to build a system around, and what are you actually building?

**Candidate:** It matters because AEO — Answer Engine Optimization, SEO's successor for a world where the primary "search result" is a generated answer, not a ranked list of blue links — depends on trust signals the crawler can check cheaply. A broken link is one of the cheapest ones to check.

If an answer engine's crawler hits a 404 on a page you're citing from, or a page you link out to as a source, that's a concrete, unambiguous signal that the content is stale or unmaintained. It can get your page demoted or dropped from the citation pool entirely, independent of how good the actual content is.

The system to build: a recurring crawl of your own site's link graph — every internal link and every outbound citation — that classifies each one as alive, dead, or flaky, and routes the dead ones to whoever owns the fix, before an external crawler finds them first.

**Interviewer:** Walk me through the actual pipeline. What's step one?

**Candidate:** Step one is building the link graph itself.

Crawl your own site — or read it from your CMS/sitemap if one exists — and extract every `<a href>`. Record the source page, the target URL, and whether the target is internal or external.

That graph is the unit everything else operates on. You can't check links you haven't enumerated.

**Interviewer:** And then you just HTTP-request every URL and see what comes back?

**Candidate:** That's the naive version, and it breaks in two specific ways at scale.

**First, volume.** A site with tens of thousands of pages, times several links each, isn't something you check serially. You need a worker pool doing concurrent requests, rate-limited **per target domain**, not globally — hammering one external site with hundreds of parallel requests gets you rate-limited or blocked, which shows up as false dead-link positives that are actually you being throttled.

**Second, transient failures.** A site that's down for 30 seconds during your crawl window isn't the same as a site that's actually gone. Treating a single timeout as "dead" generates noisy, wrong tickets.

The fix for the second problem is a status model with more than two states:
- `200`-class is alive.
- A `404`/`410` is a strong dead signal.
- Anything else — timeout, 5xx, connection refused — is `flaky` until it fails the *same* check on a retry with backoff across a longer window (hours, not seconds), before being promoted to dead.

**Interviewer:** Say you've correctly identified 4,000 dead links across the site. You can't fix all of them today. How do you decide what to act on first?

**Candidate:** Prioritize by impact, not by count. Two signals matter most:

1. How much traffic or citation volume the *source* page carries — a dead outbound link on your highest-traffic article matters more than one on a page nobody reads.
2. Whether the dead link is internal or external. An internal dead link is a broken user journey on your own site — unambiguously your bug to fix. An external dead link is a citation to someone else's page that went away — the fix there is usually "find a replacement source or remove the citation," not "fix the other site."

Rank the backlog by source-page traffic × link type. That ordering is what goes to whoever triages it.

**Interviewer:** Can any of this be auto-fixed, or does everything need a human?

**Candidate:** Some of it, carefully.

If an internal link is dead because the target page was renamed or moved, and you have redirect records, that's a safe auto-fix — rewrite the link to the new URL, no judgment call involved.

If an external citation is dead, auto-fixing is much riskier. You could auto-replace it with a search for a similar page, but silently swapping in a different source changes what your content is claiming to cite. That's a content-accuracy decision, not a plumbing one. That case should be flagged for a human to pick the replacement, or decide to just remove the citation, not resolved automatically.

**Interviewer:** How do you make sure this doesn't become a one-time cleanup that quietly rots again in six months?

**Candidate:** Run it as a recurring job, not a one-off script. Daily or weekly, depending on site size.

Re-check the full link graph each time and diff against the last run, so you only alert on *new* dead links rather than re-reporting the same backlog every time.

That diff also feeds a simple health metric — percentage of links currently alive, trend over time — that can sit on a dashboard the content team actually looks at. That's what turns this from an engineering side project into something that stays maintained.

---

## How to run these yourself

Reading the transcripts above teaches the moves. It doesn't substitute for doing this cold.

The actual drill: pick a scenario not written here. A video recommendation system at 1 million daily active users. An on-call alerting pipeline that can't miss a real incident. An ad-ranking system that has to stay fair across demographics.

Write down one concrete number — concurrency, TPS, latency budget, data volume. Then interrogate your own design with the same pattern used above:
1. Convert the headline number into a load number.
2. Break the latency budget into pipeline stages.
3. Find what breaks first.
4. Push to 10x.
5. Ask "how do I know accuracy holds, not just latency," before you're satisfied.

Better yet, get someone else to ask the follow-ups. The entire value of this format is that you don't get to pick which question comes next.

## Common pitfalls in this format
- **If you answer the opening question and stop, you've answered a different, easier question than the one being asked.** "Design a RAG system" and "design a RAG system that holds up at 10,000 concurrent users, provably" are different rounds. Only the second one is what's actually being tested here.
- **If you give a number without showing how you got it, it reads as guessed.** "500 QPS" is a strong answer. "10,000 concurrent users," restated as if that were already a load number, is not. Show the conversion — Little's Law, or an explicit stated assumption — every time.
- **If every follow-up answer is about latency and none are about how you'd know accuracy held, you're leaving half the question unanswered.** The two named scenarios in this file ("...for accuracy and latency," "...how do you handle this and that") are both explicitly two-sided. Interviewers ask the accuracy half specifically because most candidates default straight to the latency half and stop.
- **If you can't name what breaks first at 10x scale, you've designed for the number you were given and nothing past it.** Every drill above ends by pushing past the stated scale, specifically because that's where a memorized-sounding answer runs out and real reasoning has to take over.
