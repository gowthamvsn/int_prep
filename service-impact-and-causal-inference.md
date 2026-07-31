# Service Impact and Causal Inference

## Reading this from the hiring manager's seat
The metrics on your résumé — 16% team efficiency, 8% revenue margin, 6-minute-to-1.8-second query turnaround — are real and quantified, which already puts you ahead of most candidates who wave their hands at impact. But they're enterprise-IT and automotive-mobility metrics. My honest question in this round would be: *does this person know how to find and speak the specific operational KPI language of a railroad, or will they keep translating everything back into generic "efficiency" and "revenue" terms that don't map onto how this business actually measures itself?*

A railroad running under precision-scheduled-railroading discipline (which BNSF does) doesn't primarily talk in "revenue margin" for an operations initiative — it talks in **car velocity** (how fast a railcar moves door to door), **terminal dwell time** (how long a car sits idle in a yard), **cars-on-line** (how much of the network's rolling stock is tied up at once, which is capital efficiency in physical form), **locomotive and crew productivity**, and **safety incident rate**, with dollar impact usually derived *from* those operational numbers rather than stated as the headline. The strongest possible answer in this round translates your actual, real experience (the 8% margin story, the 16% efficiency story) into "here's the analogous operational lever I'd look for at BNSF, and here's the specific metric I'd expect it to move" — rather than restating your Bosch numbers as-is and hoping the interviewer draws the connection themselves. Do that translation explicitly, out loud, every time.

---

## Tying ML Model Performance to Business Metrics

### Plain-English explanation
An ML metric (AUC, F1, RMSE) measures how good the model is at the prediction task in isolation. A business metric (dollars saved, downtime-hours avoided, on-time performance) measures whether the *business* is actually better off. These are related but not the same thing, and the gap between them is exactly where "the model looked great in the demo but didn't move any real numbers" stories come from.

### Built as a chain: from a model score to a defensible dollar number

### 1. Before any conversion can happen, what's the first thing you have to write down explicitly, connecting a model's metric to an actual business result?
**Make the causal chain explicit**: model metric → decision it drives → action taken → business outcome. Every link in that chain can break independently of the others. A model can have excellent recall and still fail to improve the business outcome if, say, the action taken on a true positive doesn't actually prevent the bad outcome (e.g., flagging a failure 1 day before it happens when the maintenance lead time needed is 3 days).

### 2. Given that chain has multiple links, how do you find out WHICH link is actually breaking, instead of just re-optimizing the model metric?
**Quantify the conversion at each link**, not just the first one: of the failures the model correctly flags, what fraction actually get acted on in time? Of those acted on, what fraction of the bad outcome is actually prevented? This turns "improve recall" into a testable chain of assumptions instead of a single number to optimize.

### 3. Given each link's conversion rate is now measured, how do you turn that into an actual DOLLAR number that picks a decision threshold?
**State the ML-metric-to-business-metric conversion rate explicitly** as its own number — e.g., "we estimate $X of downtime avoided per correctly-flagged failure, net of the cost of unnecessary inspections per false positive" — and use that to pick the actual operating point (decision threshold) on the model, rather than picking the threshold that maximizes F1 in a vacuum.

### 4. Given a dollar-value conversion exists, why can the BUSINESS-optimal threshold and the ML-metric-optimal threshold (like F1's peak) genuinely disagree?
**Recognize when the two metrics can point in different directions**: a threshold change that improves recall (catches more real failures) usually also increases false positives (more unnecessary inspections); the business-optimal threshold is wherever the *marginal* dollar value of one more catch equals the *marginal* cost of one more false alarm — not wherever a generic ML metric like F1 happens to peak, since F1 weighs precision and recall equally regardless of their actual dollar costs, which are almost never equal in practice.

### 5. Given a business-optimal threshold is now computed on paper, what's left before trusting it in production?
**Validate the conversion with a controlled rollout** (shadow deployment, A/B test, or phased rollout) before assuming the offline-metric-to-business-metric relationship holds — the whole point of question 3's estimate is that it's an assumption until measured in production.

### Summary example
A failure-prediction model's causal chain (question 1) is traced link by link (question 2): 90% of flagged failures get acted on, and of those, 80% of the bad outcome is actually prevented — giving a measured $15,000-per-catch value net of $800-per-false-alarm cost (question 3). Sweeping thresholds against that number, rather than against F1, lands on a DIFFERENT threshold than F1 would pick (question 4), because F1 assumes those two costs are equal when they clearly aren't here — and that threshold only becomes trustworthy once a phased rollout (question 5) confirms the assumed $15,000/$800 conversion actually holds against real production outcomes, not just the offline estimate.

### Runnable code
```python
import numpy as np

def expected_business_value(y_true, y_scores, threshold, value_per_tp, cost_per_fp):
    """Turn a model's scores into an expected dollar value at a given decision threshold,
    instead of stopping at a threshold-free metric like AUC."""
    y_pred = (y_scores >= threshold).astype(int)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tp * value_per_tp - fp * cost_per_fp

np.random.seed(0)
y_true = np.random.binomial(1, 0.05, size=2000)          # 5% base failure rate
y_scores = np.clip(y_true * 0.6 + np.random.normal(0, 0.25, 2000), 0, 1)  # noisy but informative score

# Sweep thresholds and find the one that maximizes BUSINESS value, not F1
best_threshold, best_value = None, -np.inf
for t in np.arange(0.05, 0.95, 0.05):
    value = expected_business_value(y_true, y_scores, t, value_per_tp=15000, cost_per_fp=800)
    if value > best_value:
        best_value, best_threshold = value, t

print(f"business-optimal threshold: {best_threshold:.2f}, expected value: ${best_value:,.0f}")
```

### Common pitfalls
- **If a model shows a strong offline metric improvement but the business sees no change, it's because a link further down the causal chain is broken** — e.g., the predictions are accurate but arrive too late for anyone to act on them, or the people meant to act on them don't trust/use the output. Always trace the full chain, not just the model's own metric.
- **If a threshold is chosen by maximizing F1 (or accuracy), it's because the cost of a false positive and a false negative were implicitly assumed equal** — F1 weighs precision and recall symmetrically; real business costs almost never are (a missed safety-critical failure and an unnecessary inspection do not cost the same), so the threshold should be chosen from an explicit cost/value model, not a metric that ignores those costs.
- **If a pilot showed great business results but the full rollout didn't, it's often because the pilot wasn't representative** — a pilot run on the easiest segment, with the most engaged users, or during an unusually favorable period, will overstate the effect; validate on a representative sample and be explicit about what population the pilot covered.

---

## Causal Inference and DAGs

### Plain-English explanation
Correlation tells you two things move together; it does not tell you one *causes* the other, or whether some third factor causes both. Causal inference is the discipline of reasoning about which relationships are actually causal, using **Directed Acyclic Graphs (DAGs)** to make your assumptions about the causal structure explicit — because the right adjustment (what to control for) depends entirely on that structure, and getting it wrong in either direction (adjusting for the wrong thing) can *introduce* bias rather than remove it.

### Built as a chain: from drawing assumptions to a checkable adjustment rule

### 1. Before any adjustment decision can be made, what's the very first thing causal inference asks you to write down explicitly?
**Draw the DAG**: nodes are variables, directed edges are assumed causal relationships (arrow from cause to effect). This is a statement of your assumptions, not something the data proves — different plausible DAGs can fit the same data, which is exactly why stating your assumptions explicitly (and defending them) is the actual skill being tested.

### 2. Given a DAG exists, what's the first TYPE of variable in it that you must identify and adjust for, or your estimate is biased?
**Identify confounders**: a variable that causes *both* the treatment and the outcome (e.g., "locomotive age" causes both "how often it gets the new maintenance model's attention" and "likelihood of failure"). A confounder creates a spurious association between treatment and outcome even with zero true causal effect, and **must be adjusted for** (via stratification, regression adjustment, matching, or propensity scores) or the estimate is biased.

### 3. Given confounders must be adjusted for, is EVERY variable sitting between treatment and outcome a confounder — what's the variable that looks similar but must NOT be adjusted for?
**Identify mediators**: a variable that sits *on the causal path* between treatment and outcome (e.g., "model deployment" → "faster issue detection" → "less downtime" — faster detection is a mediator, not a confounder). **Do not adjust for mediators** if you want the total effect of the treatment — controlling for a mediator blocks part of the very effect you're trying to measure, silently shrinking your estimate toward zero.

### 4. Given mediators must NOT be adjusted for, what's the third variable type — one that looks like it should help but actively fabricates a fake relationship if you adjust for it?
**Identify colliders**: a variable that is *caused by both* treatment and outcome (or by two variables you're studying). **Do not adjust for colliders** — conditioning on a collider *creates* a spurious association between its two causes where none existed, a mistake serious enough to have its own name (collider bias / "Berkson's paradox").

### 5. Given all three variable types (confounder/mediator/collider) are now distinguishable, how do you turn that into one MECHANICAL rule for which variables belong in your model?
**Use the DAG to decide the correct adjustment set**: the backdoor criterion (adjust for variables that block every non-causal "backdoor path" from treatment to outcome, without adjusting for mediators or colliders) tells you exactly which variables belong in your regression/matching model — this is a mechanical, checkable procedure once the DAG is drawn, not a judgment call at that point.

**Visual + memory hook — the arrows' direction is the entire rule; get the shape right and the adjust/don't-adjust decision falls out automatically:**
```
CONFOUNDER                MEDIATOR                   COLLIDER
(adjust FOR it)            (do NOT adjust)            (do NOT adjust)

   Age                  Deployment                 Treatment  Outcome
   /    \                    │                          \      /
  ▼      ▼                   ▼                            ▼  ▼
Treatment Outcome      Faster detection              Selected-into-study
                              │                        (both point INTO it)
   arrows FAN OUT             ▼
   from the confounder   Less downtime
   to both T and O              │
                                ▼
                            Outcome
                        arrows form a CHAIN
                        T → mediator → O
```
**Remember it as "which way do the arrows point relative to the variable":** a confounder has two arrows pointing OUT of it (into both treatment and outcome) — it's a common upstream cause, so it must be adjusted for or it fakes a relationship that isn't there. A mediator sits IN LINE, arrows passing through it (treatment → mediator → outcome) — adjusting for it blocks part of the real effect you're trying to measure. A collider has two arrows pointing INTO it (caused by both the variables you care about) — adjusting for it manufactures a fake association between its causes out of nothing. Same action (put a variable in your regression), three completely different consequences, and the arrow direction is the only thing that tells you which one you're looking at.

### 6. Given the backdoor criterion tells you exactly what to adjust for on OBSERVATIONAL data, is adjustment alone ever as strong as an experiment?
No — **prefer a randomized or quasi-experimental design when possible** (A/B test, difference-in-differences, instrumental variables, regression discontinuity). These designs make the causal claim far more defensible than adjustment on observational data alone, because randomization (or a credible natural experiment) breaks the link between treatment assignment and unobserved confounders by construction, which no amount of adjusting for *observed* confounders (questions 2-5) can guarantee — you can only adjust for what you thought to measure.

### Summary example
A locomotive-age DAG (question 1) reveals age as a confounder (question 2) that must be adjusted for, "faster detection" as a mediator (question 3) that must NOT be adjusted for, and "selected into the maintenance study" as a potential collider (question 4) also left alone — the backdoor criterion (question 5) turns those three judgment calls into one mechanical adjustment set. But even with that correct set, an observational estimate stays weaker than a staggered rollout would have been (question 6), because adjustment can only correct for confounders you thought to measure, while randomization breaks the confounding link even against ones you didn't.

### Runnable code (Simpson's paradox: confounding reverses the apparent effect)
```python
import numpy as np
import pandas as pd

np.random.seed(42)
n = 4000

# Confounder: locomotive_age (older units both get the new monitoring system MORE, and fail more anyway)
locomotive_age = np.random.choice(["old", "new"], size=n, p=[0.5, 0.5])

# Treatment: whether this unit got the new predictive-maintenance system (older units disproportionately got it first)
p_treated = np.where(locomotive_age == "old", 0.8, 0.2)
treated = np.random.binomial(1, p_treated)

# True causal effect: treatment REDUCES failure probability by 5 percentage points, but age also drives failure risk
base_failure_rate = np.where(locomotive_age == "old", 0.30, 0.08)
failure_prob = np.clip(base_failure_rate - 0.05 * treated, 0, 1)
failed = np.random.binomial(1, failure_prob)

df = pd.DataFrame({"age": locomotive_age, "treated": treated, "failed": failed})

naive = df.groupby("treated")["failed"].mean()
print("NAIVE (confounded) comparison:\n", naive, "\n")
# Naive result can show treated units failing MORE than untreated -- backwards from the true effect,
# because 'old' locomotives are both more likely to be treated AND more likely to fail regardless.

adjusted = df.groupby(["age", "treated"])["failed"].mean().unstack()
print("ADJUSTED (stratified by confounder) comparison:\n", adjusted)
# Within each age stratum, treated units show LOWER failure rates -- the true, correctly-signed effect.
```

### Where this is real work, not a hypothetical
At Bosch, I partnered directly with business stakeholders to analyze sales data and develop classification and clustering models that were reported as directly driving an 8% increase in annual revenue margins. I'd be honest in an interview about exactly what "directly driving" can and can't claim: the models identified segments and patterns (which customer/product clusters had margin-improvement potential, which classification signals predicted a good pricing or targeting decision) that stakeholders then acted on — but the 8% is a business outcome measured after the fact, and the intellectually honest version of that story includes being able to explain what would have made me *more* confident it was the model's doing and not, say, a concurrent pricing change or a seasonally favorable market. That's exactly the causal-inference discipline in the second half of this file: I'd want to know whether the model's recommendations were rolled out to some segments before others (giving a natural comparison group), or whether other changes were happening in the same quarter that could also explain part of the lift. Being able to hold both things at once — genuine pride in a real, reported business result, and genuine rigor about what that number can and can't prove on its own — is exactly the judgment this line of interview questioning is testing for.

### Common mistakes/pitfalls
- **If a naive comparison shows the treatment "making things worse" while a stratified/adjusted comparison shows the opposite, it's Simpson's paradox from an unadjusted confounder** — exactly the pattern in the code above: locomotive age drives both treatment assignment and failure risk, so the raw comparison is dominated by the confounder, not the treatment effect. Always ask "what else could cause both the treatment and the outcome" before trusting a raw correlation.
- **If adjusting for a variable makes your causal estimate *worse* (moves further from a trusted experimental benchmark, or starts showing an implausible sign), it's often because that variable is a mediator or a collider, not a confounder** — controlling for a mediator removes part of the real effect; controlling for a collider fabricates a fake one. The fix isn't "add more controls," it's going back to the DAG and checking whether that variable actually causes both treatment and outcome (confounder — adjust for it) or sits on the causal path / is caused by both (mediator/collider — don't).
- **If a model's deployment correlates with an improved business metric and that's presented as proof the model caused the improvement, it's because no counterfactual was established** — plenty of other things change at the same time as a model launch (seasonality, other initiatives, general trend). The credible version of this claim needs either a randomized/staggered rollout, a comparison group that didn't get the model, or a pre-registered analysis plan — not just "the metric went up after we shipped."

### Likely interview question + model answer
**Question:** "Downtime dropped 12% in the quarter after you deployed the predictive maintenance model. How do you know the model caused that, and not something else?"

**Model answer (spoken flow):** "Honestly, a before-and-after comparison alone doesn't establish that — a lot of things can move at the same time as a model launch: seasonal maintenance patterns, other process changes, or just a temporarily favorable quarter. So before I claimed causation, I'd want to know how the rollout actually happened. If we did a staggered or phased rollout — some depots or fleets got the model first, others later — that gives me a natural comparison group, and I'd look at whether the depots that got it first improved sooner and by a similar margin, versus the ones that didn't yet have it. That's a much stronger causal claim than a single before/after number, because the untreated group acts as a counterfactual for what would have happened anyway.

I'd also explicitly think through what could confound that comparison — for instance, if the model was deployed first on newer locomotive fleets, and newer fleets have naturally lower failure rates regardless, a naive comparison would make the model look better than it actually is, for reasons that have nothing to do with the model. So I'd stratify or adjust for fleet age and any other factor that plausibly affects both which units got the model first and their baseline downtime risk — but I'd be careful not to over-adjust: if 'faster issue detection' is part of the causal pathway from the model to reduced downtime, I don't want to control that away, since that's the effect I'm trying to measure, not a confounder to remove.

Ideally, if the rollout wasn't naturally staggered, I'd push for holding out a genuine control group for at least one comparison cycle specifically so we can measure this cleanly, because the cost of not knowing whether the model actually caused the improvement is that we might scale an intervention that isn't actually working, or worse, deprioritize something that was.

I'd bring this same discipline from real experience, not just theory — at Bosch, models I built with business stakeholders were reported as driving an 8% increase in annual revenue margins, and I made a point of being precise, internally, about what that number could and couldn't prove on its own versus what would need a cleaner comparison to fully back up. It's easy to let a good headline number go unexamined when it's flattering; the more valuable habit is asking the causal question of your own result before someone else has to ask it for you."

---

## Practice Q&A (Self-Test)

**Q1. What's the difference between an ML metric and a business metric, and why can a model improve one without moving the other?**
A: An ML metric (AUC, F1, RMSE) measures how good the model is at the prediction task in isolation, while a business metric (dollars saved, downtime-hours avoided) measures whether the business is actually better off. The gap between them opens whenever a link in the causal chain — model metric → decision → action → outcome — breaks, such as a correctly flagged failure arriving too late for anyone to act on.

**Q2. In the expected_business_value code example, why does the threshold that maximizes business value differ from the threshold that maximizes F1?**
A: F1 weighs precision and recall symmetrically regardless of their actual dollar costs, but real business costs are almost never equal — the file's example values a true positive at $15,000 and a false positive at $800. The business-optimal threshold is wherever the marginal dollar value of one more catch equals the marginal cost of one more false alarm, not wherever F1 happens to peak.

**Q3. What is a confounder, and why must it be adjusted for? Give the example used in the file.**
A: A confounder is a variable that causes both the treatment and the outcome — the file's example is locomotive age, which causes both whether a unit got the new predictive-maintenance system first and its baseline likelihood of failure. Left unadjusted, it creates a spurious association between treatment and outcome even if the true causal effect is different (or reversed), so it must be adjusted for via stratification, regression, matching, or propensity scores.

**Q4. What is a mediator, and why should you not adjust for it if you want the total effect of a treatment?**
A: A mediator sits on the causal path between treatment and outcome — the file's example is "model deployment" → "faster issue detection" → "less downtime," where faster detection is the mediator. Controlling for a mediator blocks part of the very effect you're trying to measure, silently shrinking your estimate toward zero.

**Q5. What is a collider, and what happens if you mistakenly condition on it?**
A: A collider is a variable caused by both the treatment and the outcome (or by two variables under study). Conditioning on a collider creates a spurious association between its two causes where none existed — a mistake serious enough to have its own name, collider bias or Berkson's paradox.

**Q6. Explain the Simpson's paradox example in the file: what did the naive comparison show versus the stratified comparison, and why?**
A: The naive (unadjusted) comparison could show treated locomotives failing more than untreated ones — backwards from the true effect — because older locomotives were both more likely to receive the treatment (80% vs 20%) and more likely to fail regardless (30% vs 8% base rate). Once stratified by age, treated units showed the correctly-signed lower failure rate, since the true causal effect built into the simulation was a 5-percentage-point reduction from treatment.

**Q7. What operational railroad metrics does the file say BNSF actually talks in, rather than generic "revenue margin"?**
A: Car velocity, terminal dwell time, cars-on-line, locomotive and crew productivity, and safety incident rate — with dollar impact usually derived from those operational numbers rather than stated as the headline, consistent with precision-scheduled-railroading discipline.

**Q8. What base rate and value/cost parameters are used in the expected_business_value example code?**
A: A 5% base failure rate (y_true drawn from a binomial with p=0.05), a value of $15,000 per true positive, and a cost of $800 per false positive, with thresholds swept from 0.05 to 0.95 in steps of 0.05 to find the business-optimal operating point.

**Q9. In the model answer to "downtime dropped 12% — how do you know the model caused it," how does a staggered rollout help establish causality?**
A: If some depots or fleets got the model before others, the later-rollout group acts as a natural comparison group / counterfactual — you'd check whether the early-rollout depots improved sooner and by a similar margin. That's a much stronger causal claim than a single before/after number, though you'd still need to adjust for anything (like fleet age) that affected both which units got the model first and their baseline risk, without over-adjusting for mediators like faster issue detection.

**Q10. How does the candidate describe the honest caveat about the Bosch 8% revenue margin result?**
A: The models identified segments and patterns that stakeholders then acted on, but the 8% is a business outcome measured after the fact — the candidate says they'd want to know whether recommendations rolled out to some segments before others (a natural comparison group) or whether other changes, like a concurrent pricing change or a favorable market, happened in the same quarter that could also explain part of the lift.
