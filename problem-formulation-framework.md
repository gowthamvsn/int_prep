# Problem Formulation Framework

## 0. Reading this from the hiring manager's seat

Picture me as the one hiring for this Sr/Staff Data Scientist seat at BNSF. Here's what I'd actually be listening for in this round. And here's the specific concern your résumé raises — the one you need to answer before I'd raise my hand for you.

**What I'm listening for.** Not whether you know the objective/target/metrics/constraints framework. Every candidate who's prepped knows to ask clarifying questions — that part's table stakes.

What I actually care about: when you get an ambiguous prompt, where does your first instinct come from? Is it grounded in the real physical and operational reality of a railroad? Or is it grounded in generic SaaS/enterprise-software instincts?

"Reduce unplanned locomotive downtime" is not the same kind of problem as "reduce customer churn." A locomotive is a $2-3M physical asset. It takes days to get to a shop. Sensor data can be noisy, or missing for stretches. A wrong call has safety implications — not just a lost subscription. I want to hear that register show up in how you formulate the problem, not just as a caveat you tack on at the end.

**The specific gap your résumé raises, and how you need to close it out loud.** Your background is strong on GenAI/RAG systems — NaviDoc, FinSight, QuitBuddy — and on cloud database reliability at Bosch and Capital One. Genuinely impressive work. But neither one is heavy-industry, physical-asset, safety-critical work.

Here's my honest internal question reading your résumé: does this person default to "let's build a RAG system," or "let's fine-tune an LLM," just because that's their recent pattern? Even when the actual problem calls for something much simpler — a classical model, a rule, a basic statistical control chart, a solver?

The strongest possible answer in this round explicitly rules out an LLM or GenAI approach where it isn't the right tool, and says why. If every example you reach for is LLM-shaped, I'll read that as a hammer looking for nails, not as judgment.

Your sales-classification and clustering work at Bosch is better evidence to lead with here than your GenAI portfolio. It's plain classical ML solving a plain business problem. That's exactly what shows you don't reach for the trendiest tool by default.

## 1. Plain-English explanation

Every business prompt you'll get in this interview is deliberately underspecified. "Reduce delays." "Predict failures." "Improve on-time performance." None of these tell you what to build.

The interviewer isn't testing whether you know an algorithm. They're testing something different: can you turn a vague sentence into something a data scientist could actually go build, without someone else translating it for you first?

That translation has four pieces. They come in a fixed order, because each one constrains the next:

1. **Objective** — what business outcome are we actually trying to move, in one sentence? And who cares if it moves?
2. **Target definition** — what specific, measurable, labelable thing will the model predict or optimize, such that improving it plausibly improves the objective?
3. **Metrics** — how do we know, numerically, whether the model is good offline, and whether it's working online?
4. **Constraints** — what real-world limits (latency, interpretability, data availability, fairness, cost, regulatory) rule out otherwise-valid solutions?

**Visual + memory hook — a funnel, not a checklist, because each stage genuinely narrows what the next stage is allowed to be:**
```
   OBJECTIVE           "reduce unplanned locomotive downtime"
       │                (wide — a whole business goal)
       ▼
   TARGET             "predict: will THIS locomotive fail in
   DEFINITION          the next 14 days?" (one specific,
       │                labelable, buildable thing)
       ▼
   METRICS            precision/recall on that specific
       │               prediction, offline AND online
       ▼
   CONSTRAINTS        must run on a 3-day maintenance lead
                       time, explainable to a mechanic,
                       can't use data that isn't reliably
                       available in the field
                       ▼
                  (only NOW is "which algorithm" even
                   a well-posed question)
```
Remember it as a funnel that only narrows, never widens. You can't pick a metric before the target is defined — what would you even be measuring? You can't meaningfully discuss constraints before you know the target and the metric they're constraining.

A candidate who starts at the bottom — "I'd use XGBoost" — is starting outside the funnel entirely. There's nothing yet for XGBoost to be the answer *to*. The skill being tested is moving down this funnel in order, and out loud. Not eventually landing on a reasonable-sounding algorithm.

## 2. Step-by-step mechanics

**Step 1 — Restate the prompt as a question, then interrogate it.**
Take "reduce unplanned locomotive downtime." Ask: downtime measured how — hours, incidents, dollars? For which fleet? Over what horizon? Compared to what baseline?

Don't propose a solution yet. This step alone should take 2-3 minutes of a live interview, and it should be mostly you asking the interviewer questions — not talking at them.

**Step 2 — Write the objective as a single measurable sentence.**
"Reduce unplanned locomotive downtime hours per active unit by X% over the next 12 months, measured against the trailing-12-month baseline."

Notice what that sentence contains: a metric, a population, and a time horizon. If the interviewer hasn't given you these, ask for them. Or state the assumption you're making out loud, and move on. Never silently assume.

**Step 3 — Decompose the objective into a predictable, actionable target.**
"Downtime" itself isn't predictable at the moment you need to act. By the time a locomotive is down, it's too late.

So back up the causal chain. Downtime is caused by component failures. Component failures are often preceded by sensor anomalies — temperature, vibration, pressure trends — in some lead-time window before the failure happens.

The target becomes: "probability that component C fails within the next N days, given sensor readings up to today."

This is the single most important move in the whole framework. The target is almost never the objective's literal wording. It's the earliest reliably-observable proxy upstream of it that you can still act on in time.

**Step 4 — Interrogate the target for label quality and leakage.**
- Where do labels come from? Work orders, failure codes, manual inspection logs? Are they complete, or only recorded when someone remembers to log them?
- Is there a lag between the event and when it's recorded — one that would leak future information into training features?
- Is the target class imbalanced (failures are rare)? Does that change your metric choice, in Step 5, before it changes your modeling choice?

**Step 5 — Choose offline and online metrics, and explicitly state the gap between them.**
Offline: something like PR-AUC, or recall at a fixed precision. In plain terms: *recall* is "of the real failures, what fraction did we catch." *Precision* is "of the alarms we raised, what fraction were real." PR-AUC summarizes that tradeoff across every threshold at once (defined properly in `sklearn-practice.md` Cluster 3). For rare events, these are the honest metrics to use. Accuracy is a trap here — covered in Core Technical Depth's pitfalls.

Online/business: dollars saved, downtime-hours avoided, or "% of failures caught with at least N days of lead time."

State plainly that a model can win on the offline metric and still fail the business metric. If the lead time on true positives is too short for maintenance crews to act, for instance, none of it matters. This is exactly the ML-metric-vs-business-metric gap covered in the Service Impact file. Naming it out loud here is what separates a senior answer from a mid-level one.

**Step 6 — Enumerate constraints before proposing a solution.**
- **Latency/operational** — does a prediction need to reach a yard within minutes (real-time), or is a nightly batch fine?
- **Interpretability** — will a maintenance crew act on a black-box score, or do they need a reason code?
- **Data availability** — do you actually have the sensor telemetry at the fidelity and frequency needed, today? Or is that a 6-month data-engineering project in disguise?
- **Cost of errors is asymmetric** — a false negative (a missed failure) can cause a derailment. A false positive costs an unnecessary inspection. State this asymmetry explicitly. It will directly justify your precision/recall tradeoff later.

**Step 7 — Only now, propose an approach.** Frame it as a hypothesis to be validated, not a decision already made: "Given all of the above, I'd start with a gradient-boosted survival model or classifier on the lead-time-window target. I'd validate it offline with recall-at-precision, then run a shadow or A/B test measuring actual downtime-hours-avoided before rollout."

Unpacking a few terms in that sentence: a *survival model* predicts time-until-event, instead of a plain yes/no label. *Gradient-boosted* is the sequential, error-correcting tree ensemble covered in `ml-models-practice.md` Cluster 3. A *shadow test* runs the new model on live data while its outputs are logged but never acted on. An *A/B test* lets a random slice of real decisions actually use it, and compares outcomes against the rest.

## 2b. Where I've actually run this exact process

This framework isn't something I'm reciting from a textbook. It's close to the literal sequence I went through at Bosch, before the sales classification and clustering work that ended up driving an 8% increase in annual revenue margins.

The initial ask from business stakeholders wasn't "build a classification model." It was closer to "help us understand where we're leaving margin on the table" — exactly the same kind of ambiguity as "reduce unplanned locomotive downtime."

I had to work backward from that vague objective to an actual, predictable target: which customer, product, and pricing patterns were associated with better versus worse margin outcomes. That had to happen before any modeling choice mattered.

Then I had to separate two different metrics. The offline modeling metric — how well the clustering and classification separated meaningfully different segments. The business metric — actual margin movement, measured over a full quarter, not a backtest. I needed that separation before I'd trust the result enough to call it an 8% improvement, rather than an artifact of a good-looking offline number.

One more thing had to happen first, before any of that. I'd already had to build the Azure-based ETL pipeline that fed clean, fast, reliable data into the analysis. That's its own version of the "constraints" step in this framework: no amount of good problem formulation matters if the data pipeline underneath it takes 6 minutes per query and can't support the iteration speed the actual analysis needs.

## 3. Worked mini-example (a "runnable" artifact for this topic)

There's no code to run for a framework, so the runnable artifact here is a worksheet instead. Fill it in, literally, during the interview — write it on the whiteboard or shared doc verbatim. It structures your spoken answer for you.

```
BUSINESS PROMPT (verbatim):  "________________________________"

OBJECTIVE (metric + population + horizon):
  Move [metric] for [population] by [amount] over [timeframe].

TARGET (what the model actually predicts, and why it's upstream of the objective):
  P(event) within [lead time], defined from [label source].

LABEL QUALITY CHECKS:
  - Source: ____________        Lag/leakage risk: ____________
  - Class balance: ____________  Missingness: ____________

METRICS
  Offline: ____________ (why this, not accuracy/AUC-ROC)
  Online/business: ____________
  Named gap between them: ____________

CONSTRAINTS
  Latency: ____   Interpretability: ____   Data availability: ____
  Error asymmetry: FN costs ____ vs FP costs ____

PROPOSED APPROACH (stated as a hypothesis):
  ____________________________________________
```

Filling this in out loud, in order, for whatever prompt you're given — that's the actual deliverable for this round.

## 4. Common mistakes/pitfalls

**You jump straight to naming a model or algorithm in the first 30 seconds.** That means you're pattern-matching to a "which ML technique" question, instead of the actual "can you scope a problem" question being asked. This round is deliberately not about algorithm choice. Interviewers are trained to notice candidates who skip straight there.

**Your "target" and "objective" turn out to be the same sentence with different words.** That means you didn't actually decompose the causal chain. Saying the target is "predict downtime," when downtime itself isn't observable early enough to act on, is the classic version of this mistake. Always ask: by the time I know this, can anyone still do anything about it?

**You propose accuracy as your headline offline metric for a rare-event problem.** That means you haven't stated the base rate out loud. A 1%-failure-rate problem gets 99% accuracy just by predicting "never fails." That's a dead giveaway you didn't check the class balance before picking a metric.

## 5. Likely interview question + model spoken answer

**Question:** "BNSF wants to use AI to reduce unplanned locomotive downtime. Walk me through how you'd approach this."

**Model answer (spoken flow):**

"Before I propose anything, I want to pin down what 'reduce downtime' actually means. That phrase alone doesn't tell me what to build.

So first, I'd ask: is downtime measured in hours, incidents, or dollars? For which fleet? Compared to what baseline — say, the trailing twelve months? Let's say the answer is hours of unplanned downtime per active unit, and the goal is a meaningful year-over-year reduction.

Here's why I don't jump to modeling yet. 'Downtime' itself isn't something I can act on — by the time a locomotive is actually down, it's too late to prevent it. So I traced the causal chain backward. Downtime is caused by component failures. Failures are often preceded by measurable drift in sensor signals — temperature, vibration, pressure — over some lead-time window before the failure actually happens.

That gave me my real target. Not 'predict downtime.' Instead: 'predict the probability that a given component fails within the next N days, given its recent sensor history.' I picked N based on how much lead time maintenance crews actually need to act. If the model fires the day before failure, it's useless operationally, even if it's accurate.

Before I'd trust that target, I'd check where the failure labels actually come from — work orders, or failure codes. If those are only logged inconsistently, or logged after a delay that leaks into my features, the model will look good offline and fail in production. I'd also expect failures to be rare, so I'd flag upfront that accuracy is the wrong offline metric here. I'd use something like recall at a fixed precision, or PR-AUC, because a naive 'never fails' predictor would already score 99%+ accuracy and tell us nothing useful.

Then I separated offline metrics from online ones, on purpose. Offline, I'm optimizing something like recall at precision. Online, what actually matters to the business is downtime-hours avoided, and the false-positive cost of unnecessary inspections. I'd call out that gap explicitly. A model can improve the offline metric and still fail commercially — if, say, its true positives fire with too little lead time for a crew to react. That's exactly the kind of gap I'd want to catch in a shadow deployment before a full rollout, not after.

Last, I'd list constraints before proposing a model. Does this need to run in real time as telemetry streams in, or is a nightly batch job fine? Does a maintenance crew need an interpretable reason code, or just a risk score? And critically — is the error cost asymmetric? A missed failure risks a much more serious safety incident than an unnecessary inspection does. That asymmetry should directly shape where I set the decision threshold.

Only at that point would I actually propose an approach. Something like a gradient-boosted or survival model on the lead-time-window target, validated offline on recall-at-precision, then rolled out through a shadow test that measures actual downtime-hours avoided — not just the offline metric — before it ever changes a real maintenance schedule.

This isn't a process I'm describing in the abstract, either. It's the same sequence I actually went through at Bosch, before the sales analytics work that ended up moving annual revenue margins by 8%. The initial ask was just as underspecified as 'reduce downtime.' Getting from that vague sentence to a model stakeholders actually trusted took exactly this discipline: pin down the objective, find the earliest actionable target upstream of it, separate the offline metric from the business one, and only then propose a modeling approach."

---

## Practice Q&A (Self-Test)

**Q1. What are the four components of the problem formulation framework, and why does the order in which you address them matter?**
A: Objective, target definition, metrics, and constraints, in that fixed order. Each piece constrains the next — you can't pick a sensible target until the objective is pinned down, and you can't pick metrics until the target is defined. A candidate who jumps straight to naming a model has skipped the part that actually proves they understood the business.

**Q2. Why is "reduce unplanned locomotive downtime" not directly usable as a model target?**
A: Downtime itself isn't observable early enough to act on. By the time a locomotive is actually down, it's too late to prevent it. You have to trace the causal chain backward to the earliest reliably-observable proxy you can still act on — sensor-drift patterns that precede a component failure by some lead-time window, for instance.

**Q3. What is described as the single most important move in the whole framework?**
A: Decomposing the objective into a predictable, actionable target — Step 3. The target is almost never the objective's literal wording. It's the earliest reliably-observable proxy upstream of the objective that you can still act on in time.

**Q4. Why is accuracy a trap as the offline metric for the locomotive failure model, and what should be used instead?**
A: Failures are rare, so a naive "never fails" predictor would already score around 99% accuracy while catching nothing. Accuracy hides the base rate. Use recall at a fixed precision, or PR-AUC, instead — those actually reflect performance on the rare positive class.

**Q5. What is the "named gap" the framework insists you state explicitly between offline and online metrics?**
A: A model can win on the offline metric — recall at precision, say — and still fail the business metric — downtime-hours avoided — if its true positives fire with too little lead time for a maintenance crew to act. Naming that gap out loud, instead of assuming a good offline number implies business impact, is what the framework says separates a senior answer from a mid-level one.

**Q6. Why is the cost of errors called out as asymmetric in this problem, and how should that shape the model?**
A: A false negative — a missed failure — can cause a derailment, a safety incident. A false positive only costs an unnecessary inspection. That asymmetry should directly justify where the decision threshold gets set, favoring catching more true failures even at the cost of more false alarms.

**Q7. What real project from the candidate's background is offered as evidence this framework isn't just theoretical, and what was the initial ask?**
A: The Bosch sales classification and clustering work that drove an 8% increase in annual revenue margins. The initial stakeholder ask was as vague as "help us understand where we're leaving margin on the table." The candidate had to work backward from that to an actual predictable target before any modeling choice mattered.

**Q8. What four questions should you ask when interrogating a target for label quality and leakage?**
A: Where do the labels actually come from — work orders, failure codes, inspection logs — and are they complete? Is there a lag between the event and when it's recorded that could leak future information into features? And is the target class imbalanced in a way that should change the metric choice before it changes the modeling choice?

**Q9. What's the classic pitfall when a candidate names a specific model or algorithm in the first 30 seconds of this kind of question?**
A: It signals they're pattern-matching to a "which ML technique" question, instead of the actual "can you scope a problem" question being tested. Interviewers are trained to notice candidates who skip straight to a model choice, without first restating the prompt, defining the target, and naming metrics and constraints.

**Q10. According to the "hiring manager's seat" framing, what specific concern does the candidate's résumé raise for this round, and what's the strongest way to address it?**
A: The concern is whether the candidate defaults to "build a RAG system" or "fine-tune an LLM" out of habit from their recent GenAI portfolio, even when a problem calls for something simpler, like a classical model or a rule. The strongest answer explicitly rules out an LLM or GenAI approach where it isn't the right tool, and leads with the classical ML work — like the Bosch sales classification and clustering project — rather than the GenAI portfolio.
