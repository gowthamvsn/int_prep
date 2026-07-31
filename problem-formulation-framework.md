# Problem Formulation Framework

## 0. Reading this from the hiring manager's seat
If I were the one hiring for this Sr/Staff Data Scientist seat at BNSF, here's what I'd actually be listening for in this specific round, and here's the specific concern your background raises that you need to answer before I'd raise my hand for you.

**What I'm listening for**: not whether you know the objective/target/metrics/constraints framework — every candidate who's prepped knows to ask clarifying questions. I'm listening for whether your *first instinct*, when given an ambiguous prompt, is grounded in the actual physical and operational reality of a railroad, or whether it's grounded in generic SaaS/enterprise-software instincts. "Reduce unplanned locomotive downtime" is not the same kind of problem as "reduce customer churn" — a locomotive is a $2-3M physical asset that takes days to get to a shop, sensor data can be noisy or missing for stretches, and a wrong call has safety implications, not just a lost subscription. I want to hear that register in how you formulate the problem, not just in a caveat you tack on at the end.

**The specific gap your résumé raises, and how you need to close it out loud**: your background is strong on GenAI/RAG systems (NaviDoc, FinSight, QuitBuddy) and cloud database reliability (Bosch, CapitalOne) — genuinely impressive, but neither is heavy-industry, physical-asset, safety-critical work. My honest internal question reading your résumé would be: *does this person default to "let's build a RAG system" or "let's fine-tune an LLM" because that's their recent pattern, even when the actual problem calls for something much simpler — a classical model, a rule, a basic statistical control chart, or a solver?* The strongest possible answer in this round is one where you explicitly rule out an LLM/GenAI approach for a problem where it's not the right tool, and say why. If every example you reach for is LLM-shaped, I will read that as a hammer looking for nails, not as judgment. Your actual sales-classification/clustering work at Bosch — plain classical ML solving a plain business problem — is the better evidence to lead with here, not your GenAI portfolio, precisely because it shows you don't reach for the trendiest tool by default.

## 1. Plain-English explanation

Every business prompt you'll get in this interview ("reduce delays," "predict failures," "improve on-time performance") is deliberately underspecified. The interviewer is not testing whether you know an algorithm — they're testing whether you can turn a vague sentence into something a data scientist could actually go build, without someone else having to translate it for you first. That translation has four pieces, in a fixed order, because each one constrains the next:

1. **Objective** — what business outcome are we actually trying to move, in one sentence, and who cares if it moves?
2. **Target definition** — what specific, measurable, labelable thing will the model predict or optimize, such that improving it plausibly improves the objective?
3. **Metrics** — how do we know, numerically, whether the model is good (offline) and whether it's working (online)?
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
**Remember it as a funnel that only narrows, never widens** — you cannot pick a metric before the target is defined (what would you even be measuring?), and you cannot meaningfully discuss constraints before you know the target and metric they're constraining. A candidate who starts at the bottom ("I'd use XGBoost") is starting outside the funnel entirely — there's nothing yet for XGBoost to be the answer *to*. The skill being tested is that you move down this funnel *in order* and *out loud*, not that you eventually reach a reasonable-sounding algorithm.

## 2. Step-by-step mechanics

**Step 1 — Restate the prompt as a question, then interrogate it.**
Take "reduce unplanned locomotive downtime" and ask: downtime measured how (hours? incidents? dollars?), for which fleet, over what horizon, compared to what baseline? Do not propose a solution yet. This step alone should take 2–3 minutes of a live interview and should be mostly you asking the interviewer questions, not talking at them.

**Step 2 — Write the objective as a single measurable sentence.**
"Reduce unplanned locomotive downtime hours per active unit by X% over the next 12 months, measured against the trailing-12-month baseline." Notice this sentence contains a metric, a population, and a time horizon. If the interviewer hasn't given you these, ask, or explicitly state the assumption you're making and move on — never silently assume.

**Step 3 — Decompose the objective into a predictable, actionable target.**
"Downtime" itself isn't predictable at the moment you need to act — by the time a locomotive is down, it's too late. So you back up the causal chain: downtime is caused by component failures; component failures are often preceded by sensor anomalies (temperature, vibration, pressure trends) in some lead-time window. The *target* becomes: "probability that component C fails within the next N days, given sensor readings up to today." This is the single most important move in the whole framework — the target is almost never the objective's literal wording, it's the earliest reliably-observable proxy upstream of it that you can still act on in time.

**Step 4 — Interrogate the target for label quality and leakage.**
- Where do labels come from? (Work orders? Failure codes? Manual inspection logs?) Are they complete, or only recorded when someone remembers to log them?
- Is there a lag between the event and when it's recorded that would leak future information into training features?
- Is the target class imbalanced (failures are rare), and does that change your metric choice (Step 5) before it changes your modeling choice?

**Step 5 — Choose offline and online metrics, and explicitly state the gap between them.**
Offline: something like PR-AUC or recall at a fixed precision (rare-event classification — accuracy is a trap here, covered in Core Technical Depth pitfalls). Online/business: dollars saved, downtime-hours avoided, or "% of failures caught with ≥N days lead time." State plainly that a model can win on the offline metric and still fail the business metric if, say, the lead time on true positives is too short for maintenance crews to act — this is exactly the ML-metric-vs-business-metric gap covered in the Service Impact file, and naming it here is what separates senior from mid-level answers.

**Step 6 — Enumerate constraints before proposing a solution.**
- **Latency/operational**: does a prediction need to reach a yard within minutes (real-time) or is a nightly batch fine?
- **Interpretability**: will a maintenance crew act on a black-box score, or do they need a reason code?
- **Data availability**: do you actually have the sensor telemetry at the fidelity/frequency needed, today, or is that a 6-month data-engineering project in disguise?
- **Cost of errors is asymmetric**: a false negative (missed failure) can cause a derailment; a false positive costs an unnecessary inspection. State this asymmetry explicitly — it will directly justify your precision/recall tradeoff later.

**Step 7 — Only now, propose an approach**, and frame it as a hypothesis to be validated, not a decision: "Given the above, I'd start with a gradient-boosted survival model or classifier on the lead-time-window target, validate offline with recall-at-precision, then run a shadow/A-B test measuring actual downtime-hours-avoided before rollout."

## 2b. Where I've actually run this exact process
This framework isn't something I'm reciting from a textbook — it's close to the literal sequence I went through at Bosch before the sales classification/clustering work that ended up driving an 8% increase in annual revenue margins. The initial ask from business stakeholders wasn't "build a classification model" — it was closer to "help us understand where we're leaving margin on the table," which has exactly the same ambiguity as "reduce unplanned locomotive downtime." I had to work backward from that vague objective to an actual predictable target (which customer/product/pricing patterns were associated with better vs. worse margin outcomes) before any modeling choice mattered, and I had to separate the offline modeling metric (how well the clustering/classification separated meaningfully different segments) from the business metric (actual margin movement, measured over a full quarter, not a backtest) before I'd have trusted the result enough to call it an 8% improvement rather than an artifact of a good-looking offline number. Separately, before I could build any of that, I'd already had to engineer the Azure-based ETL pipeline that fed clean, fast, reliable data into the analysis in the first place — which is its own version of the "constraints" step in this framework: no amount of good problem formulation matters if the data pipeline underneath it takes 6 minutes per query and can't support the iteration speed the actual analysis needs.

## 3. Worked mini-example (a "runnable" artifact for this topic)

There's no code to execute for a framework, so the runnable artifact here is a worksheet you can literally fill in during the interview — write it on the whiteboard/shared doc verbatim, it structures your spoken answer:

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

Filling this in out loud, in order, for whatever prompt you're given, *is* the deliverable for this round.

## 4. Common mistakes/pitfalls

- **If you jump straight to naming a model or algorithm in the first 30 seconds, it's because you're pattern-matching to a "which ML technique" question instead of a "can you scope a problem" question** — this round is deliberately not about algorithm choice, and interviewers are trained to notice candidates who skip straight there.
- **If your "target" and "objective" turn out to be the same sentence with different words, it's because you didn't actually decompose the causal chain** — e.g., saying the target is "predict downtime" when downtime itself isn't observable early enough to act on. Always ask "by the time I know this, can anyone still do anything about it?"
- **If you propose accuracy as your headline offline metric for a rare-event problem, it's because you haven't stated the base rate out loud** — a 1%-failure-rate problem gets 99% accuracy by predicting "never fails," which is a dead giveaway you didn't check the class balance before picking a metric.

## 5. Likely interview question + model spoken answer

**Question:** "BNSF wants to use AI to reduce unplanned locomotive downtime. Walk me through how you'd approach this."

**Model answer (spoken flow):**

"Before I propose anything, I want to pin down what 'reduce downtime' actually means, because that phrase alone doesn't tell me what to build. So first I'd ask: is downtime measured in hours, incidents, or dollars, for which fleet, and compared to what baseline — say, the trailing twelve months? Let's say the answer is hours of unplanned downtime per active unit, and the goal is a meaningful year-over-year reduction.

The reason I don't jump to modeling here is that 'downtime' itself isn't something I can act on — by the time a locomotive is actually down, it's too late to prevent it. So I traced the causal chain backward: downtime is caused by component failures, and failures are often preceded by measurable drift in sensor signals — temperature, vibration, pressure — over some lead-time window before the failure actually happens. That gave me my actual target: not 'predict downtime,' but 'predict the probability that a given component fails within the next N days, given its recent sensor history.' I picked N based on how much lead time maintenance crews actually need to act — if the model fires the day before failure, it's useless operationally even if it's accurate.

Before I trusted that target, I'd check where the failure labels actually come from — work orders or failure codes — because if those are only logged inconsistently, or logged after a delay that leaks into my features, the model will look good offline and fail in production. I'd also expect failures to be rare, so I flagged upfront that accuracy is the wrong offline metric here; I'd use something like recall at a fixed precision, or PR-AUC, because a naive 'never fails' predictor would already score 99%+ accuracy and tell us nothing.

Then I separated offline from online metrics on purpose: offline, I'm optimizing something like recall at precision. Online, what actually matters to the business is downtime-hours avoided and the false-positive cost of unnecessary inspections. I called out that gap explicitly, because a model can improve the offline metric and still fail commercially if, say, its true positives fire with too little lead time for a crew to react — that's exactly the kind of gap I'd want to catch in a shadow deployment before a full rollout, not after.

Last, I listed constraints before proposing a model: does this need to run in real time as telemetry streams in, or is a nightly batch job fine; does a maintenance crew need an interpretable reason code or just a risk score; and — critically — is the error cost asymmetric, because a missed failure risks a much more serious safety incident than an unnecessary inspection does, and that asymmetry should directly shape where I set the decision threshold.

Only at that point would I actually propose an approach — something like a gradient-boosted or survival model on the lead-time-window target, validated offline on recall-at-precision, and then rolled out through a shadow test that measures actual downtime-hours avoided, not just the offline metric, before it ever changes a real maintenance schedule.

This isn't a process I'm describing in the abstract, either — it's the same sequence I actually went through at Bosch before the sales analytics work that ended up moving annual revenue margins by 8%. The initial ask was just as underspecified as 'reduce downtime,' and getting from that vague sentence to a model stakeholders actually trusted required exactly this discipline: pin down the objective, find the earliest actionable target upstream of it, separate the offline metric from the business one, and only then propose a modeling approach."

---

## Practice Q&A (Self-Test)

**Q1. What are the four components of the problem formulation framework, and why does the order in which you address them matter?**
A: Objective, target definition, metrics, and constraints, in that fixed order. Each piece constrains the next — you can't pick a sensible target until the objective is pinned down, and you can't pick metrics until the target is defined — so candidates who jump straight to a model name have skipped the part that proves they understood the business.

**Q2. Why is "reduce unplanned locomotive downtime" not directly usable as a model target?**
A: Downtime itself isn't observable early enough to act on — by the time a locomotive is actually down, it's too late to prevent it. You have to trace the causal chain backward to the earliest reliably-observable proxy you can still act on, such as sensor-drift patterns that precede a component failure by some lead-time window.

**Q3. What is described as the single most important move in the whole framework?**
A: Decomposing the objective into a predictable, actionable target — Step 3. The target is almost never the objective's literal wording; it's the earliest reliably-observable proxy upstream of the objective that you can still act on in time.

**Q4. Why is accuracy a trap as the offline metric for the locomotive failure model, and what should be used instead?**
A: Failures are rare, so a naive "never fails" predictor would already score around 99% accuracy while catching nothing — accuracy hides the base rate. The framework recommends recall at a fixed precision or PR-AUC instead, since those metrics actually reflect performance on the rare positive class.

**Q5. What is the "named gap" the framework insists you state explicitly between offline and online metrics?**
A: A model can win on the offline metric (e.g., recall at precision) and still fail the business metric (downtime-hours avoided) if, say, true positives fire with too little lead time for a maintenance crew to act. Naming that gap out loud — rather than assuming a good offline number implies business impact — is what the framework says separates senior from mid-level answers.

**Q6. Why is the cost of errors called out as asymmetric in this problem, and how should that shape the model?**
A: A false negative (a missed failure) can cause a derailment — a safety incident — while a false positive only costs an unnecessary inspection. That asymmetry should directly justify where the decision threshold is set, favoring catching more true failures even at the cost of more false alarms.

**Q7. What real project from the candidate's background is offered as evidence this framework isn't just theoretical, and what was the initial ask?**
A: The Bosch sales classification/clustering work that drove an 8% increase in annual revenue margins. The initial stakeholder ask was as vague as "help us understand where we're leaving margin on the table," and the candidate had to work backward from that to an actual predictable target before any modeling choice mattered.

**Q8. What four questions should you ask when interrogating a target for label quality and leakage?**
A: Where do the labels actually come from (work orders, failure codes, inspection logs) and are they complete; is there a lag between the event and when it's recorded that could leak future information into features; and is the target class imbalanced in a way that should change the metric choice before the modeling choice.

**Q9. What's the classic pitfall when a candidate names a specific model or algorithm in the first 30 seconds of this kind of question?**
A: It signals they're pattern-matching to a "which ML technique" question instead of the actual "can you scope a problem" question being tested. Interviewers are trained to notice candidates who skip straight to a model choice without first restating the prompt, defining the target, and naming metrics and constraints.

**Q10. According to the "hiring manager's seat" framing, what specific concern does the candidate's résumé raise for this round, and what's the strongest way to address it?**
A: The concern is whether the candidate defaults to "build a RAG system" or "fine-tune an LLM" out of habit from their recent GenAI portfolio, even when a problem calls for something simpler like a classical model or a rule. The strongest answer explicitly rules out an LLM/GenAI approach where it isn't the right tool and leads with the classical ML work (like the Bosch sales classification/clustering project) rather than the GenAI portfolio.
