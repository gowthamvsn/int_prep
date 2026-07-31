# Behavioral / Partnership STAR Stories

**How this file is built**: these four stories are anchored in your actual background from `base_cv.py` (Bosch, the CapitalOne/Cognizant incident work, the UNT research assistantship, the student org you founded) — real companies, real projects, real quantified outcomes. But your CV (correctly) only records *what happened*, not *how the conversation went* or *exactly what someone objected to* — so anywhere you see `[FILL IN: ...]`, that's a gap only your real memory can close. Read each story, then replace every bracket with what actually happened before you say this out loud in an interview. An interviewer at Sr/Staff level will ask a follow-up ("what did they say when you pushed back," "what would you do differently") that only a real memory survives — a polished-but-invented answer falls apart under that probing.

### Reading this from the hiring manager's seat
Here's what I'd specifically be probing in this round, given your background: your real partnership experience — Bosch stakeholders, a Johns Hopkins faculty collaborator on QuitBuddy, clinicians at a health-informatics conference — has mostly been with people who are either technically fluent themselves or already bought into the value of AI/data work. My honest concern reading your résumé for a BNSF Sr/Staff seat: *can this person hold their ground and communicate effectively with a skeptical, operations-first stakeholder — a dispatcher, a mechanical engineer, a safety officer — who has seen "the new software" over-promise before, works in a culture that is deliberately conservative and cost-disciplined (BNSF is a Berkshire Hathaway company, and that ownership shows up as a real preference for proven, unglamorous wins over flashy pilots), and does not care that your RAG system hit 35% ROUGE/BLEU or that your model runs on Kubernetes?*

The adjustment I'd want to hear you name explicitly, without being asked: influence in this environment often means translating a technical result into "how many fewer inspection-hours does this cost you" or "how much sooner does this catch a real problem," stated in the stakeholder's own operational terms, and being comfortable with a slower, more evidence-demanding buy-in process than a fast-moving GenAI project or an academic research collaboration usually requires. If your stories only show you persuading people who were already predisposed to agree with an AI-forward pitch, that's the gap I'd push on — so where you can, lean on the parts of your background that *did* involve skeptical, non-technical, operationally-minded stakeholders (the Bosch business stakeholders you partnered with on sales analytics, who cared about margin, not methodology) rather than only the more AI-native audiences.

### Plain-English explanation of what this round tests
Behavioral rounds at the Sr/Staff level test **judgment under ambiguity and low authority**, not just "did something go well once." The interviewer wants evidence you can influence people who don't report to you, hold a technical position under pressure, operate when requirements are genuinely unclear, and — critically — talk about a real failure without over-blaming yourself (poor calibration) or blaming everyone else (no ownership).

### Built as a chain: four stories, each covering a DIFFERENT axis the round tests

### 1. If a candidate could only demonstrate ONE thing in this round, what's the most foundational judgment call — getting something to happen with no positional power to force it?
**Story 1 — Leading Without Formal Authority.** This tests whether you can get a cross-functional outcome to happen through credibility and making it easy for others to say yes, not escalation — the Bosch GenAI workflow-automation story below, told through the *influence* angle, not just the *build* angle.

### 2. Given you can get people to say yes to something new (question 1), what happens when someone in power actively disagrees with your technical judgment — can you hold your ground without becoming a pushover OR combative?
**Story 2 — Disagreeing With a Stakeholder.** This is a genuinely different axis from Story 1: influence-without-authority is about *proposing*, this is about *defending a position under pressure* once it's being actively pushed back on — the ransomware-recovery story below, where the pressure was almost certainly toward a faster, riskier option.

### 3. Given you can both propose (question 1) and defend (question 2) a position, what happens when there's no clear position to defend yet — when the requirements themselves are genuinely undefined?
**Story 3 — Handling Ambiguous Requirements.** A third distinct axis: this isn't about convincing anyone of anything, it's about not freezing and not silently guessing when given a direction instead of a spec — the UNT research-assistantship story below, where "work on hallucination mitigation for healthcare" was a direction, not a spec.

### 4. Given questions 1-3 all show you succeeding at something (persuading, defending, clarifying), what does the round need to see that NONE of those three can show — genuine failure, handled with real ownership?
**Story 4 — A Real Failure and Lessons Learned.** This is the one axis a résumé can't hand you pre-written, because a résumé documents outcomes, not the version of a project that didn't work the first time — calibration and ownership can only be tested against something that actually went wrong, which is why this story has to be mined from real memory rather than assembled from a CV line the way Stories 1-3 were.

### Summary example
A candidate who only has stories 1-3 ready has proven they can propose, defend, and clarify — but an interviewer probing for judgment under ambiguity will notice the gap immediately if pushed on "tell me about a time something didn't work": Story 4 is what closes that gap, and it's the one story on this page that cannot be pre-written, because inventing a failure and presenting it as history would mean fabricating the same professional record Stories 1-3 draw their credibility from.

### STAR mechanics, as a reminder
- **Situation** — enough context to understand the stakes, in 2-3 sentences.
- **Task** — your specific responsibility or goal, one sentence.
- **Action** — the bulk of the answer. Specific decisions and *why* ("I did X because Y"), not a list of activities.
- **Result** — a concrete, ideally quantified outcome, plus (for the failure story) what you changed afterward.

**Visual + memory hook — STAR isn't four equal quarters, it's one short setup and one long payoff:**
```
Situation  ██░░░░░░░░░░░░░░░░░░  ~15%   (2-3 sentences, just enough stakes)
Task       █░░░░░░░░░░░░░░░░░░░  ~10%   (one sentence — what was YOUR job)
Action     ████████████████░░░░  ~55%   (the actual answer — decisions AND why)
Result     ██████░░░░░░░░░░░░░░  ~20%   (a real number, plus what changed after)
```
**Remember it as a wide-then-narrow hourglass, not four equal boxes** — the single most common way candidates burn their limited time badly is treating each STAR letter as roughly equal airtime, which means the Situation eats 40% of the answer setting up context nobody asked for while Action (the only part that's actually evidence of *your* judgment) gets rushed. If you're ever unsure whether to cut something while telling a story out loud, cut from Situation first, never from Action — Situation only needs to establish stakes, Action is the entire point of the round.

---

## Story 1: Leading Without Formal Authority

### What this is testing
Whether you can get a cross-functional outcome to happen with no positional power to force it — through credibility and making it easy for others to say yes, not escalation.

### Real anchor (from your CV)
Bosch, Feb 2021–Dec 2024: *"Led development of a GenAI-based workflow automation system using LLMs, reducing manual project coordination and improving team efficiency by 16%."* You were an individual-contributor Data Scientist, not a manager of the teams whose workflow this changed — which is exactly what makes this a "leading without authority" story, if you tell the *influence* part, not just the *build* part.

### STAR skeleton with real facts + fill-in-the-blanks
**Situation**: "At Bosch, project coordination across `[FILL IN: which teams/functions — engineering, program management, mobility cloud ops?]` was eating a lot of manual effort — `[FILL IN: what the actual pain was, e.g. status-chasing across teams, manual handoffs, a specific recurring bottleneck]`. I didn't manage those teams; I was proposing a system that would change how they worked day to day."

**Task**: "My goal was to get a GenAI-based workflow automation system actually adopted, which meant convincing people who didn't report to me that changing their process was worth it — not just building something technically sound."

**Action** (this is the part to make real — the template shape is: *what you built, who you had to convince, what specifically made them say yes*): "I built `[FILL IN: what the system actually automated — was it LLM-driven status summarization, ticket triage, meeting-note-to-action-item extraction?]`. Before asking anyone to change their workflow, I `[FILL IN: what you did to make the ask land — did you pilot it quietly with one willing team first? show a before/after time comparison? get one respected engineer to vouch for it?]`, because `[FILL IN: your actual reasoning — e.g. "I'd seen top-down tool mandates get quietly ignored, so I wanted proof from a real team before asking others"]`."

**Result**: "That translated into a 16% improvement in team efficiency `[FILL IN: efficiency measured how — hours saved per sprint, faster handoffs, fewer status meetings?]`, and `[FILL IN: what happened to adoption after the initial win — did other teams ask to use it, did it become a standard tool?]`."

### Alternate real anchor, if this one doesn't fit the interview's flow
Founding a student organization at UNT from a 3-member team to 100+ members (Golden Eagle Award, Best New Student Organization) is a cleaner, less-technical "leading without authority" story if the interviewer already has plenty of your technical leadership examples from elsewhere in the loop — worth having both ready, since a 5-hour loop increases the odds you're asked this more than once by different interviewers.

### Common pitfalls in this story type
- **If your answer is "I built it well and people used it," it's because you're describing execution, not influence** — the interviewer is listening for the moment you had to persuade someone who could have said no.
- **If you can't answer "what if the first team had said no to piloting it," it's because you haven't thought through what actually made the ask compelling** — have that ready.

---

## Story 2: Disagreeing With a Stakeholder

### What this is testing
Whether you can hold a technically-grounded position under pressure without being a pushover or needlessly combative.

### Real anchor (from your CV)
Bosch: *"Recovered a ransomware-locked MongoDB instance by mounting it locally and performing a full backup and restore, preserving a production client's complete dataset with zero data loss."* Ransomware incidents almost always come with pressure to take the fastest-looking option (pay the ransom, restore from a stale backup and accept some data loss, fail over immediately) rather than the safer, slower, more deliberate one — which makes this a strong real candidate for "held a technical position under pressure," if that tension actually happened.

### STAR skeleton with real facts + fill-in-the-blanks
**Situation**: "A production MongoDB instance for one of our enterprise clients on Bosch's mobility cloud platform got hit by ransomware — the data was locked, and `[FILL IN: who was in the room / on the incident call putting pressure on a specific resolution — the client, my manager, an incident commander — and what they initially wanted to do]`."

**Task**: "My job was to actually recover the client's data with zero loss, which meant `[FILL IN: what the disagreement actually was — did someone want to restore from an older backup and accept some data loss to move faster? consider paying? fail over to a secondary that might also be compromised?]`."

**Action**: "Instead of `[FILL IN: the faster/riskier option that was on the table]`, I proposed mounting the locked instance locally and performing a full backup and restore — I pushed for this because `[FILL IN: your actual reasoning, e.g. "a stale backup meant losing N hours/days of production transactions for an enterprise client, and I believed the locked instance was still recoverable without paying anything or accepting that loss"]`. I `[FILL IN: how you actually made the case under pressure — did you time-box your approach so if it failed there was still a fallback? show early evidence it was working?]`."

**Result**: "We preserved the client's complete dataset with zero data loss, `[FILL IN: how long it took, and — for the STAR to land as a partnership story, not just a technical win — what the person who initially wanted the other approach said afterward, or how that changed how incidents got handled going forward]`."

### Common pitfalls in this story type
- **If your story is "I was right and they eventually agreed," with no acknowledgment of their pressure, it's because you're telling a story about being correct, not about partnership** — name why the other option looked reasonable to them in the moment (usually: speed, or fear of a worse outcome), not just that you overruled it.
- **If you can't name anything that was actually uncertain or risky about your approach in the moment, it's because the story's been smoothed into something too clean to be real** — recovering a ransomware-locked instance without paying isn't guaranteed to work; say what would have happened if the local mount/restore hadn't worked.

---

## Story 3: Handling Ambiguous Requirements

### What this is testing
Whether you default to asking the clarifying questions that actually matter and stating assumptions explicitly (the Problem Formulation framework, applied to a real research setting), versus freezing or silently guessing.

### Real anchor (from your CV)
UNT Graduate Research Assistant, Aug 2025–present: *"Conducting applied research under the Health Informatics Program Director on LLM hallucination mitigation for healthcare — using RAG to ground model responses against scientific literature, achieving 20-second end-to-end retrieval from complex medical documents."* An applied-research assistantship under a program director is a textbook source of genuine ambiguity — "work on hallucination mitigation for healthcare" is a direction, not a spec.

### STAR skeleton with real facts + fill-in-the-blanks
**Situation**: "When I started this research assistantship, the direction was 'work on LLM hallucination mitigation for healthcare using RAG' — `[FILL IN: what was actually left open at the start — which clinical domain/document set, what counted as success, whether this was aimed at a specific publication venue or dataset]`."

**Task**: "Before I could actually start building, I needed to turn that into something specific enough to evaluate — `[FILL IN: what the eventual scope became, e.g. grounding against a specific corpus of scientific literature, targeting a specific claim-verification or Q&A benchmark]`."

**Action**: "I asked `[FILL IN: what you actually asked your advisor/program director early on — what's the target document set, what's an acceptable retrieval latency, how will 'reduced hallucination' actually get measured]`, because `[FILL IN: why those specific questions — e.g. "the evaluation metric would completely change what counted as a good architecture, so I didn't want to build the retrieval pipeline before that was settled"]`. Where it was still ambiguous after that — `[FILL IN: e.g. which specific faithfulness metric to report, or how deep in the corpus to search]` — I made a reasonable default choice and stated it explicitly rather than waiting, specifically `[FILL IN: what default you picked and why]`."

**Result**: "That produced a system with 20-second end-to-end retrieval from complex medical documents, `[FILL IN: and what came out of that — a working benchmark, a paper draft, a direction for the next phase]`, built against a scope that was actually agreed rather than assumed."

### Common pitfalls in this story type
- **If your story is "I asked a lot of clarifying questions," full stop, it's because you're describing a habit, not judgment** — show that you distinguished which ambiguities actually changed the architecture (worth blocking on) from which you could default through.
- **If you never mention stating an assumption explicitly, it's because the story risks sounding like you either interrogated your advisor endlessly or guessed silently** — the credible middle ground is what's being tested.

---

## Story 4: A Real Failure and Lessons Learned

### What this is testing
Calibration and ownership: a real failure, told honestly, with a durable change afterward — not a humble-brag ("I care too much") and not blame-shifting.

### Why this one can't be pre-written for you
Your CV, like every résumé, documents outcomes, not the version of each project that didn't work the first time — so unlike the three stories above, I don't have a real failure to anchor this one in. Making one up and presenting it as your history would be fabricating your professional record, which isn't something to risk carrying into an actual interview. What I *can* do is point at the real places in your own timeline most likely to hold a genuine failure story, so you're mining your own memory instead of starting from a blank page.

### Real candidate moments worth mining (all directly from your CV)
- **The Azure ETL pipeline rebuild** (6-minute query turnaround down to 1.8 seconds): a rewrite this dramatic almost never lands perfectly on the first attempt — was there an early version that broke something, silently returned wrong results, or had to be rolled back before the final version worked?
- **Five years of 99.999%/99.99% uptime across MongoDB/PostgreSQL/MySQL/MSSQL/Redis at Bosch, and Oracle at Cognizant/Wipro before that**: that number is an average — there's very likely at least one specific incident inside those years where something you did (a migration, a config change, a judgment call under on-call pressure) caused an outage or a close call, distinct from the ransomware story above, which you already handled well.
- **The sales classification/clustering work that drove the 8% revenue margin increase**: initial modeling attempts that don't hit the mark before the one that does are extremely common — was there a model or feature approach you were confident in that didn't actually hold up when validated with the business stakeholders?
- **FinSight's 3-agent debate architecture or the 87%-accuracy Random Forest model**: multi-agent and financial-prediction systems are notorious for an earlier version that looked good in backtesting but failed in a way that taught you something specific about evaluation.

### STAR skeleton to fill in once you've picked the real one
**Situation**: "`[FILL IN: what you built/shipped, and the context]`."

**Task**: "`[FILL IN: what you were actually responsible for]`."

**Action**: "The mistake was `[FILL IN: your specific decision or gap — not a passive mechanism, your actual choice]`. Once I found it, I `[FILL IN: what you did first — communicated it, stopped the damage, then root-caused]`."

**Result**: "`[FILL IN: how it was resolved]`, and the lasting change was `[FILL IN: a standing change to how you work now, not just a one-time fix]`."

### Common pitfalls in this story type
- **If your "failure" is a success story wearing a costume ("I worked too hard"), it's because you're protecting yourself instead of answering the question** — interviewers have heard every version of the fake-failure answer, and it costs more credibility than an honest, moderate real one.
- **If the story ends at "I fixed it" with no durable change afterward, it's because you're describing incident response, not a lesson learned.**
- **If you spend the story explaining why it wasn't really your fault, it's because you're optimizing for not looking bad instead of demonstrating ownership** — name your own specific decision that led to the gap, not just the mechanism that happened "to" you.

---

## Practice Q&A (Self-Test)

**Q1. In Story 1 (Leading Without Authority), what's the real CV anchor, and what specific quantified result is tied to it?**
A: Leading development of a GenAI-based workflow automation system at Bosch that reduced manual project coordination and improved team efficiency by 16%. The story only works as a "leading without authority" answer if it's told through the influence angle — you were an individual-contributor Data Scientist, not a manager of the teams whose workflow changed.

**Q2. What alternate real anchor does the file suggest for Story 1, and what recognitions did it earn?**
A: Founding a student organization at UNT that grew from a 3-member team to 100+ members, which earned the Golden Eagle Award and Best New Student Organization. It's offered as a cleaner, less-technical option worth having ready in case the "leading without authority" question comes up again later in a 5-hour loop.

**Q3. In Story 2 (Disagreeing With a Stakeholder), what's the real anchor, and what result does the CV record?**
A: Recovering a ransomware-locked MongoDB instance at Bosch by mounting it locally and performing a full backup and restore. The recorded result is preserving a production client's complete dataset with zero data loss.

**Q4. What's the key pitfall to avoid when telling Story 2, according to the file?**
A: Telling it as "I was right and they eventually agreed" without acknowledging why the other, riskier/faster option looked reasonable to others in the moment (usually speed, or fear of a worse outcome) turns it into a story about being correct rather than about partnership. You should also be able to say what was actually uncertain about your approach — recovering a ransomware-locked instance without paying wasn't guaranteed to work.

**Q5. In Story 3 (Ambiguous Requirements), what's the real anchor, and what specific quantified outcome does the CV record?**
A: The UNT Graduate Research Assistantship on LLM hallucination mitigation for healthcare, using RAG to ground model responses against scientific literature. The CV records achieving 20-second end-to-end retrieval from complex medical documents.

**Q6. What is Story 3 specifically testing, per the file's "What this is testing" section?**
A: Whether you default to asking the clarifying questions that actually matter and stating assumptions explicitly — the Problem Formulation framework applied to a real research setting — versus freezing or silently guessing when a direction like "work on hallucination mitigation for healthcare" is given without a spec.

**Q7. Why can't Story 4 (Real Failure) be pre-written, and what four real candidate moments does the file suggest mining for it?**
A: A résumé documents outcomes, not the versions of a project that didn't work the first time, and inventing a failure would mean fabricating professional history. The four candidate moments are: the Azure ETL pipeline rebuild (6 minutes to 1.8 seconds), the years of 99.999%/99.99% uptime across multiple databases (which likely hides at least one incident), the sales classification/clustering work behind the 8% margin increase (possible early modeling attempts that didn't hold up), and FinSight's 3-agent debate architecture or 87%-accuracy Random Forest model (possible backtest-vs-production gap).

**Q8. What's the "success story wearing a costume" pitfall the file warns about for the failure story?**
A: Presenting a failure like "I worked too hard" is really protecting yourself instead of answering the question, since interviewers have heard every version of the fake-failure answer — it costs more credibility than an honest, moderate real one, and the story must also include a durable change afterward, not just "I fixed it."

**Q9. What are the four STAR components, and what specifically should the Action section contain according to the file's reminder?**
A: Situation (2-3 sentences of context/stakes), Task (your specific responsibility, one sentence), Action (the bulk of the answer), and Result (a concrete, ideally quantified outcome, plus for the failure story what you changed afterward). The file specifies Action should contain specific decisions and *why* — "I did X because Y" — not a list of activities.

**Q10. According to the "hiring manager's seat" section, what gap in the candidate's partnership experience is this round probing, and which real experience does the file suggest leaning on?**
A: The concern is whether the candidate can hold their ground with a skeptical, operations-first stakeholder in a deliberately conservative, cost-disciplined culture (BNSF, a Berkshire Hathaway company), since the candidate's real partnership experience has mostly been with technically fluent or already-bought-in people (Bosch stakeholders, a Johns Hopkins collaborator, clinicians). The file suggests leaning on the Bosch business stakeholders from the sales analytics work, who cared about margin, not methodology, as the closest real example of a skeptical, non-technical, operationally-minded audience.
