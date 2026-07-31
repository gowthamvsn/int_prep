# Live Coding Prep

Round 2's coding sub-round tests fluency, not cleverness — can you write correct, idiomatic code under time pressure and narrate it. For every topic below: explain out loud what you're about to do *before* you type, write the straightforward version first, then optimize only if asked.

**Visual + memory hook — the same 3-step order, every single problem, no exceptions:**
```
1. NARRATE           2. BRUTE FORCE            3. OPTIMIZE
   "I'm going to        write the                 only now, and only
   use a hashmap        obviously-correct          if asked — trade
   to do this in        version first,             something (space
   one pass, here's      even if O(n²)              for time, usually)
   why" — BEFORE          — a working slow           for a real, named
   typing a line           answer beats a             reason, stated
                           broken fast one             out loud
```
**Remember it as "say it, ship the ugly version, then earn the clever one"** — in that exact order, every time, no matter how well you already know the optimal solution. Jumping straight to the clever O(n) trick without narrating first, or without a working brute-force fallback, is the single most common way a candidate who *actually knows the answer* still reads as weaker than one who visibly works through the steps — this round is scoring the process as much as the destination.

## Reading this from the hiring manager's seat
Here's the specific thing I'd be probing for in this round, given your résumé: your last several years of hands-on building have leaned heavily on orchestrating LLM APIs and pretrained models (FinSight, NaviDoc, QuitBuddy) rather than writing classical algorithms or hand-rolled model code day to day. That's a completely legitimate way to build real systems — but it raises a fair question for a Sr/Staff-level bar: *can this person actually write correct, fluent code from first principles under time pressure, or have they gotten used to a model doing the hard part while they write orchestration/glue code around it?*

Your SQL and Pandas fundamentals are the least risky part of this for you — five years of hands-on database administration (Bosch, Cognizant, Wipro) means you've almost certainly written more raw SQL under production pressure than most "pure ML" candidates ever will, and that's worth saying plainly rather than underselling. Where I'd actually be watching closely is the classical-algorithm and from-scratch-modeling sections below (Dijkstra, precision/recall by hand, the transformer built with raw matrix multiplies) — not because BNSF needs you to reimplement PyTorch, but because writing it once, correctly, live, is the fastest signal of whether you understand what's actually happening inside the systems you've been orchestrating, versus having a surface-level fluency with API calls. Practice the from-scratch sections out loud, not just read them — the gap between "I understand this" and "I can produce this correctly in fifteen minutes while someone watches" is exactly what this round is designed to expose.

---

## SQL Window Functions

### Plain-English explanation
A normal aggregate (`GROUP BY`) collapses many rows into one. A **window function** computes a value across a set of related rows ("the window") but keeps every row — you get the per-row detail *and* a value computed relative to its group. Think "rank each sensor reading within its locomotive" vs. "give me one row per locomotive."

### Built as a chain: from the shared syntax to why you can't filter on it directly

### 1. Every window function shares one piece of syntax — what is it, and what do its two parts each control?
Every window function uses `OVER (PARTITION BY ... ORDER BY ...)`. `PARTITION BY` defines the group (like `GROUP BY` but non-collapsing); `ORDER BY` defines the row sequence the function walks.

### 2. Given rows are now grouped and ordered, what's the simplest thing you can compute per row within that window — a plain rank?
`ROW_NUMBER()` — assigns 1, 2, 3... with no ties, even if values are equal (tie-breaking is arbitrary unless your `ORDER BY` is unique).

### 3. Given `ROW_NUMBER()` breaks ties arbitrarily, what changes if you actually want tied rows to SHARE a rank?
`RANK()` — same value gets the same rank, but leaves a gap afterward (1, 2, 2, 4). `DENSE_RANK()` — same value gets the same rank, no gap (1, 2, 2, 3).

### 4. Given all three ranking functions look at ONE row's position, how do you instead pull a value from a NEIGHBORING row in the same partition, without a self-join?
`LAG(col, n)` / `LEAD(col, n)` — read a value from `n` rows before/after the current row *within the partition*, without a self-join. This is how you compute deltas, period-over-period change, or "time since last event."

### 5. Given all four functions above compute a value per row, why can't you filter directly on that value in the SAME query's `WHERE` clause?
Execution order matters: window functions run *after* `WHERE`/`GROUP BY` but *before* `ORDER BY`/`LIMIT` in the logical query plan — which is why you can't filter on a window function directly in `WHERE` (use a subquery or `QUALIFY`, if your dialect supports it).

### Runnable code (PostgreSQL dialect)
```sql
CREATE TABLE sensor_readings (
  locomotive_id  INT,
  reading_time   TIMESTAMP,
  temp_c         NUMERIC
);

INSERT INTO sensor_readings VALUES
  (1, '2026-01-01 00:00', 85.0),
  (1, '2026-01-01 01:00', 87.5),
  (1, '2026-01-01 02:00', 91.0),
  (2, '2026-01-01 00:00', 78.0),
  (2, '2026-01-01 01:00', 78.0),
  (2, '2026-01-01 02:00', 95.0);

-- Rank each reading within its locomotive by temperature, hottest first
SELECT
  locomotive_id, reading_time, temp_c,
  ROW_NUMBER() OVER (PARTITION BY locomotive_id ORDER BY temp_c DESC) AS row_num,
  RANK()       OVER (PARTITION BY locomotive_id ORDER BY temp_c DESC) AS rnk,
  DENSE_RANK() OVER (PARTITION BY locomotive_id ORDER BY temp_c DESC) AS dense_rnk
FROM sensor_readings;

-- Hour-over-hour temperature delta per locomotive (this is the interview-favorite pattern)
SELECT
  locomotive_id, reading_time, temp_c,
  temp_c - LAG(temp_c) OVER (PARTITION BY locomotive_id ORDER BY reading_time) AS delta_temp
FROM sensor_readings;

-- "Latest reading per locomotive" without a GROUP BY + subquery join
SELECT * FROM (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY locomotive_id ORDER BY reading_time DESC) AS rn
  FROM sensor_readings
) t
WHERE rn = 1;
```

### Common pitfalls
- **If your "top-N per group" query returns more or fewer rows than N, it's because you used `RANK()` instead of `ROW_NUMBER()`** and there were ties — `RANK()` will happily return 4 rows for "top 3" if two rows tie for 2nd place. Use `ROW_NUMBER()` when you need exactly N.
- **If a query errors with "window functions are not allowed in WHERE," it's because you tried to filter on a window function in the same `SELECT` that defines it** — window functions evaluate after `WHERE`. Wrap the query in a subquery/CTE and filter on the outer layer instead.
- **If `LAG()` returns unexpected values across group boundaries, it's because you forgot `PARTITION BY`** — without it, `LAG()` walks the *entire* result set in `ORDER BY` sequence, so the last row of locomotive 1 will look back at the last row of locomotive 2 if you didn't partition.

### Likely interview question + model answer
**Question:** "Given a table of sensor readings per locomotive over time, write a query that flags any reading where the temperature jumped more than 15°C from the previous reading for that locomotive."

**Model answer (spoken flow):** "I need the previous reading's temperature available on the same row as the current one, for the same locomotive, in time order — that's exactly what `LAG` is for, so I don't need a self-join. I'd partition by `locomotive_id` so `LAG` never crosses between locomotives, and order by `reading_time` so 'previous' means chronologically previous. Then I compute the current temp minus the lagged temp as `delta_temp`, and wrap that in a CTE so I can filter on it afterward — I can't filter on the window function directly in the same `SELECT`'s `WHERE` clause, because window functions evaluate after `WHERE` runs. So: CTE computes `delta_temp` using `temp_c - LAG(temp_c) OVER (PARTITION BY locomotive_id ORDER BY reading_time)`, then the outer query does `WHERE ABS(delta_temp) > 15`. I used `ABS` because the prompt said 'jumped,' which I'd clarify — do we care about spikes up, drops down, or both — before assuming.

This is close to actual work I've done, not just a textbook pattern — at Bosch I engineered the Azure-based ETL pipeline for the database team, and the reason query turnaround dropped from 6 minutes to 1.8 seconds wasn't a bigger compute tier, it was replacing what used to be multiple round-trip queries and application-side loops with exactly this kind of single-pass, window-function-based SQL: computing per-partition comparisons (latest-vs-previous, current-vs-running-baseline) directly in one query instead of pulling rows back to Python and looping over them. The lesson that generalizes: any time someone's about to write application code to compare 'this row' to 'the previous row for this same entity,' that's almost always a `LAG`/`LEAD` sitting inside a CTE, computed once by the database instead of round-tripped."

---

## Python / Pandas Data Manipulation

### Plain-English explanation
Pandas problems in interviews test whether you reach for **vectorized operations** (fast, idiomatic) instead of Python `for` loops over rows (slow, and a signal you don't know the library). The core skill is recognizing which of a small set of primitives — `groupby`, `merge`, `pivot_table`, rolling windows — matches the shape of the transformation being asked for.

### Built as a chain: from collapsing rows to reshaping the whole table

### 1. What's the base operation every other pandas primitive here builds on, and what's its non-collapsing sibling?
**`groupby(...).agg(...)`** collapses rows by key, like SQL `GROUP BY`. `groupby(...).transform(...)` does the same computation but *broadcasts the result back to every row* — use `transform` when you need a per-row column like "this row's value minus its group's mean," not a summary table.

### 2. Given `groupby` handles one table, how do you combine TWO tables when their keys don't line up exactly (e.g., timestamps)?
**`merge`** — `how='inner'/'left'/'outer'` behaves like SQL joins. `pd.merge_asof` is the one interviewers love for time-series/sensor data: it joins on the *nearest* prior key instead of an exact match — essential when event timestamps never line up exactly with sensor timestamps.

### 3. Given a merged, ordered table now exists, how do you compute a MOVING statistic over it rather than one fixed group-level number?
**`rolling(window=n)`** computes a moving statistic (mean, std) over the last `n` rows — must be sorted first, and grouped rolling needs `.groupby(...).rolling(...)`.

### 4. Given `groupby`/`merge`/`rolling` all operate on the table's existing shape, how do you change the SHAPE itself — long rows into wide columns?
**`pivot_table`** reshapes long data to wide (one row per key, one column per category) with an aggregation function for collisions — it's `groupby` + reshape in one call.

### 5. Given all four primitives above are vectorized, what's the one anti-pattern that undoes all of that speed?
Rule of thumb for speed: if you write `df.iterrows()`, stop — there is almost always a vectorized equivalent, and `iterrows()` is 10–100x slower on non-trivial data.

### Runnable code
```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    "locomotive_id": [1, 1, 1, 2, 2, 2],
    "reading_time": pd.to_datetime([
        "2026-01-01 00:00", "2026-01-01 01:00", "2026-01-01 02:00",
        "2026-01-01 00:00", "2026-01-01 01:00", "2026-01-01 02:00",
    ]),
    "temp_c": [85.0, 87.5, 91.0, 78.0, 78.0, 95.0],
})

# groupby + transform: how far above THIS locomotive's own average is each reading?
df["temp_above_own_mean"] = df["temp_c"] - df.groupby("locomotive_id")["temp_c"].transform("mean")

# rolling per-group mean (2-reading moving average per locomotive)
df = df.sort_values(["locomotive_id", "reading_time"])
df["rolling_mean_2"] = (
    df.groupby("locomotive_id")["temp_c"].rolling(2, min_periods=1).mean().reset_index(level=0, drop=True)
)

# merge_asof: join maintenance events to the nearest PRIOR sensor reading (classic time-series join)
events = pd.DataFrame({
    "locomotive_id": [1, 2],
    "event_time": pd.to_datetime(["2026-01-01 01:30", "2026-01-01 01:15"]),
    "event_type": ["inspection", "brake_check"],
})
merged = pd.merge_asof(
    events.sort_values("event_time"),
    df.sort_values("reading_time"),
    left_on="event_time", right_on="reading_time",
    by="locomotive_id", direction="backward",
)
print(df)
print(merged)
```

### Common pitfalls
- **If `merge_asof` throws "left keys must be sorted," it's because you forgot to sort both frames by the join column first** — unlike `merge`, `merge_asof` requires sorted input and will not sort it for you.
- **If a `groupby().mean()` silently drops a column you expected, it's because that column is non-numeric** (e.g., a string or object dtype) and older pandas versions silently excluded it from numeric aggregation — always check `df.dtypes` before trusting an aggregate's shape, or pass `numeric_only=True` explicitly to make the behavior deterministic.
- **If your rolling average looks identical across groups, it's because you called `.rolling()` on the whole frame instead of `.groupby(...).rolling(...)`** — a plain `df["col"].rolling(n).mean()` slides across group boundaries, quietly mixing locomotive 1's last reading into locomotive 2's first window.

### Likely interview question + model answer
**Question:** "You have a table of sensor readings and a separate table of maintenance events. For each event, find the most recent sensor reading before it happened, for the same locomotive."

**Model answer (spoken flow):** "This is a join, but not an exact-match join — event timestamps and sensor timestamps won't line up to the second, so a regular `merge` on time would return nothing. What I actually want is 'the nearest reading at or before this event, per locomotive,' which is precisely what `merge_asof` does with `direction='backward'`. Before calling it, I sorted both frames by their time columns, because `merge_asof` assumes sorted input and gives wrong results silently if it isn't — that's a mistake I've made before, so I check it every time. I passed `by='locomotive_id'` so the match only looks within the same locomotive, not globally across the whole fleet, and `left_on`/`right_on` since the two frames use different column names for time. The result is one row per event, carrying along whatever reading was in effect at that moment — which is exactly the shape I'd need downstream if, say, I wanted to check whether temperature was already anomalous right before a maintenance event was logged.

I'd bring the same instinct here that I used at Cognizant, where I built Python and shell monitoring scripts covering 120 production, dev, and QA servers for CapitalOne's banking workloads, replacing what had been a fully manual review process — those scripts had to detect and flag infrastructure anomalies in 35 seconds, which meant the detection logic had to be vectorized aggregation and lookups against recent history, not per-server loops re-querying state one at a time. The pattern is identical: 'compare this event to the most recent relevant prior state, per entity, fast' is the same shape whether the entity is a database server's CPU/memory baseline or a locomotive's sensor history, and `merge_asof` (or the equivalent vectorized lookup) is what keeps that fast at scale instead of degrading linearly with the number of entities you're monitoring."

---

## Classic Algorithms

### Dijkstra's Shortest Path

#### Plain-English explanation
Dijkstra finds the shortest path from one node to all others in a graph with non-negative edge weights, by always expanding the closest unvisited node next — a greedy strategy that works specifically *because* weights can't be negative (a negative edge could later "beat" a path you already finalized, which Dijkstra can't undo).

#### Built as a chain: from initial distances to a finished shortest-path table

#### 1. Before any exploration happens, what's the starting state for every node's distance?
Initialize distance to the source as 0, all others as infinity.

#### 2. Given every node starts at infinity except the source, what data structure decides WHICH node to explore next?
Use a min-priority-queue keyed by current best-known distance.

#### 3. Given the queue always pops the closest known node, what do you do with a popped node that's already been finalized?
Pop the closest unvisited node. If it's already been finalized with a shorter distance, skip it (this handles stale queue entries cheaply instead of decrease-key).

#### 4. Given a genuinely new closest node is popped, what actual UPDATE happens to its neighbors?
**Relax** every outgoing edge: if `dist[current] + edge_weight < dist[neighbor]`, update it and push the neighbor back onto the queue.

#### 5. Given relaxation keeps pushing updated neighbors back onto the queue, when does the whole process actually stop, and what do you have at the end?
Repeat until the queue is empty. Every node's final popped distance is its shortest path.

#### Runnable code
```python
import heapq

def dijkstra(graph: dict[str, list[tuple[str, float]]], source: str) -> dict[str, float]:
    """graph: {node: [(neighbor, weight), ...]}"""
    dist = {node: float("inf") for node in graph}
    dist[source] = 0.0
    pq = [(0.0, source)]
    visited = set()

    while pq:
        d, node = heapq.heappop(pq)
        if node in visited:
            continue
        visited.add(node)
        for neighbor, weight in graph.get(node, []):
            new_dist = d + weight
            if new_dist < dist[neighbor]:
                dist[neighbor] = new_dist
                heapq.heappush(pq, (new_dist, neighbor))
    return dist

graph = {
    "yard_A": [("yard_B", 4), ("yard_C", 1)],
    "yard_B": [("yard_D", 1)],
    "yard_C": [("yard_B", 2), ("yard_D", 5)],
    "yard_D": [],
}
print(dijkstra(graph, "yard_A"))  # {'yard_A': 0.0, 'yard_B': 3.0, 'yard_C': 1.0, 'yard_D': 4.0}
```

#### Common pitfalls
- **If Dijkstra gives a wrong (too-short) answer, it's because a negative edge weight exists somewhere** — Dijkstra's greedy finalization assumes a node's shortest distance can never improve after it's popped, which negative edges violate. Use Bellman-Ford instead.
- **If your implementation times out on a large graph, it's because you used a plain list and linear-scanned for the minimum** instead of a heap — that turns an O((V+E) log V) algorithm into O(V²).
- **If you get stale/duplicate work, it's because you didn't skip already-visited nodes popped from the queue** — without the `if node in visited: continue` check, you'll re-relax edges from a node multiple times (still correct, just wasteful — but in an interview, skipping this line looks like you don't understand why it's there).

---

### Precision, Recall, F1 From Scratch

#### Plain-English explanation
These three numbers all come from the same 2×2 confusion matrix and answer three different questions: **precision** — "of everything I flagged, how much was actually right?"; **recall** — "of everything that was actually right, how much did I catch?"; **F1** — the harmonic mean of the two, which penalizes a model that's lopsided (great precision, terrible recall, or vice versa) more than a plain average would.

#### Built as a chain: from the raw confusion counts to one balanced score

#### 1. Before any of the three metrics can be computed, what's the first thing every prediction has to be bucketed into?
Compare each prediction to its true label and bucket into TP (predicted positive, actually positive), FP (predicted positive, actually negative), FN (predicted negative, actually positive), TN (predicted negative, actually negative).

#### 2. Given those four buckets exist, how do you compute precision — "of everything I flagged, how much was right"?
`precision = TP / (TP + FP)` — denominator is *everything you predicted positive*.

#### 3. Given precision uses TP and FP, how does recall use the SAME TP but a different denominator?
`recall = TP / (TP + FN)` — denominator is *everything that was actually positive*.

#### 4. Given precision and recall can disagree sharply, how do you combine them into ONE number without just averaging?
`F1 = 2 * precision * recall / (precision + recall)` — harmonic mean; using the arithmetic mean instead would let a model with 100% precision / 1% recall score ~50%, which overstates a nearly-useless model.

#### Runnable code
```python
def precision_recall_f1(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}

y_true = [1, 0, 1, 1, 0, 1, 0, 0]
y_pred = [1, 0, 0, 1, 1, 1, 0, 0]
print(precision_recall_f1(y_true, y_pred))  # {'precision': 0.75, 'recall': 0.75, 'f1': 0.75}
```

#### Common pitfalls
- **If precision and recall are both 0 and you get a `ZeroDivisionError`, it's because your model predicted the positive class zero times** (all-negative predictor) — always guard the denominators, and treat that case as precision=0 (undefined-but-conventionally-zero), not a crash.
- **If someone asks for "the F1 score" on a multi-class problem and you compute a single global number without asking which averaging they mean, it's because you forgot F1 doesn't generalize to multi-class on its own** — you need to pick `macro` (unweighted mean across classes — treats rare classes equally), `micro` (aggregate all TP/FP/FN globally first — dominated by frequent classes), or `weighted` (weighted by class support), and these can disagree substantially on imbalanced data.
- **If a model looks great on accuracy but the interviewer keeps pushing on precision/recall, it's because the dataset is imbalanced** — on a 95%-negative dataset, predicting "always negative" gets 95% accuracy and 0% recall; naming this tradeoff unprompted is what signals seniority.

### Where I've actually had to make this tradeoff
Two of my own projects sit right on this exam-favorite scenario. The **Pneumonia Detection classifier** (MobileNetV2 transfer learning, 95% AUC, with Grad-CAM and SHAP for interpretability) is a binary medical classifier where a false negative — telling a patient with pneumonia they're clear — is categorically worse than a false positive that triggers a follow-up chest X-ray review; I cared about recall on the positive (disease) class specifically, and used AUC plus Grad-CAM/SHAP together precisely because a single threshold-free number like AUC doesn't tell you where you've set the operating point, and clinical interpretability tools like Grad-CAM let a radiologist sanity-check *why* the model flagged what it flagged, not just *that* it flagged something. Similarly, the **Hospital Readmission model** (XGBoost, best ROC-AUC among evaluated models, predicting 30-day readmission risk for diabetic patients) has the same asymmetry — missing a patient who will be readmitted costs more (in both health and cost terms) than an unnecessary follow-up call for one who won't. In both cases the real interview-relevant lesson wasn't "compute precision and recall," it was deciding *which* class's recall to optimize for and being able to justify that choice by the actual cost of each error type, not by whichever metric happened to look best.

---

## PyTorch: Fine-Tuning a Language Model (BERT)

### Plain-English explanation
BERT was pretrained on masked-language-modeling over huge unlabeled text, so it already "understands" language structure. Fine-tuning means bolting a small task-specific head (e.g., a linear classifier) onto its output and continuing training — at a much smaller learning rate — on your small labeled dataset, so the pretrained knowledge is nudged, not overwritten.

### Built as a chain: from loading pretrained weights to a stable fine-tuning run

### 1. Before any training happens, what has to be loaded, and what's added on top of it?
Load a pretrained tokenizer and model checkpoint matched to your task (`AutoModelForSequenceClassification` for classification — it adds the head automatically, initialized randomly).

### 2. Given a model and tokenizer both exist, how does raw text actually become something the model can consume as a batch?
Tokenize text into input IDs + attention masks, padded/truncated to a fixed max length.

### 3. Given a tokenized batch, what does a forward pass through the now-headed model actually produce?
Feed batches through the model; it returns logits over your num_labels classes.

### 4. Given logits exist, how do you turn them into an actual weight update, and why does the optimizer's learning rate need to be so much smaller than training from scratch?
Compute cross-entropy loss against your labels, backprop, step an optimizer (**AdamW**, typically LR ~2e-5 — much smaller than training from scratch, because you don't want to destroy the pretrained weights).

### 5. Given a small LR is already protecting the pretrained weights, what ADDITIONAL scheduling is standard on top of it?
A linear or cosine LR **decay schedule with warmup** is standard — a big LR spike on frozen-then-unfrozen pretrained weights is a common source of instability.

### 6. Given warmup already smooths the LR, is there an even MORE conservative option for a tiny labeled dataset?
Optionally freeze the base encoder and only train the head first (faster, less prone to catastrophic forgetting on tiny datasets), then unfreeze for a final low-LR pass.

### Runnable code
```python
# pip install torch transformers
import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

MODEL_NAME = "bert-base-uncased"  # downloads weights on first run
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)

texts = [
    "The locomotive passed inspection with no issues.",
    "Critical brake failure reported on unit 4471.",
    "Routine maintenance completed on schedule.",
    "Emergency derailment risk flagged by sensor alert.",
]
labels = [0, 1, 0, 1]  # 0 = routine, 1 = critical

encodings = tokenizer(texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
labels_t = torch.tensor(labels)

optimizer = AdamW(model.parameters(), lr=2e-5)
num_steps = 3
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_steps)

model.train()
for epoch in range(num_steps):
    optimizer.zero_grad()
    outputs = model(
        input_ids=encodings["input_ids"].to(device),
        attention_mask=encodings["attention_mask"].to(device),
        labels=labels_t.to(device),
    )
    loss = outputs.loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    print(f"epoch {epoch}: loss={loss.item():.4f}")
```

### An honest note on where my experience actually sits
I haven't personally fine-tuned a BERT-family model end to end in production — my real-world LLM systems (NaviDoc, FinSight, QuitBuddy, the Mental Health Wellness Chatbot) are built on pretrained models used via API and RAG, which was the right call for each of those given time and data constraints. Where I *have* done exactly this shape of transfer learning — freeze a pretrained backbone, train a small task-specific head on limited labeled data, at a low learning rate to avoid destroying pretrained features — is the vision side: ResNet18 for Alzheimer's MRI staging and MobileNetV2 for Pneumonia Detection. Same underlying mechanics as BERT fine-tuning (freeze-then-adapt, small LR, watch for the model forgetting pretrained structure if the LR is too aggressive), different modality. I'd be direct about that distinction in an interview rather than overstate hands-on BERT fine-tuning specifically.

### Common pitfalls
- **If fine-tuning "forgets" everything and outputs garbage, it's because the learning rate was too high** (a from-scratch LR like 1e-3 will blow the pretrained weights apart in a few steps) — always start around 1e-5 to 5e-5 for full fine-tuning.
- **If loss is `nan` after a few steps, it's because gradients exploded**, often from a too-large LR combined with no gradient clipping — `clip_grad_norm_` is cheap insurance, not optional polish.
- **If validation accuracy is excellent but the model fails badly on real production text, it's because the labeled fine-tuning set is tiny and non-representative** — BERT fine-tuning is extremely sample-efficient compared to training from scratch, but a few hundred examples from one narrow distribution (e.g., only clean-formatted incident reports) still won't generalize to messy real-world input.

---

## PyTorch: A Transformer Built From Scratch

### Plain-English explanation
"From scratch" means implementing self-attention, multi-head attention, the feed-forward block, residual connections, and layer norm yourself with raw matrix multiplies — not calling `nn.Transformer` or `nn.MultiheadAttention`. Interviewers ask this to check you understand what's *inside* the black box, not just that you can call an API.

### Built as a chain: from raw input to vocabulary logits, one matrix multiply at a time

### 1. Before attention can compute anything, what three projections does every token need?
Project the input into Query, Key, Value with three learned linear layers.

### 2. Given Q/K/V exist for the full model width, how do you split them so MULTIPLE heads can attend independently?
Reshape into `num_heads` parallel subspaces so each head attends independently.

### 3. Given each head has its own Q/K/V slice, what's the actual attention computation per head, and why divide by `sqrt(d_k)`?
Compute `softmax(QKᵀ / sqrt(d_k))·V` per head — divide by `sqrt(d_k)` to keep the dot products from saturating softmax (this is *the* fact every interviewer expects you to justify unprompted).

### 4. Given plain attention lets every token see every other token, what has to change if this is a DECODER block instead of an encoder?
Apply a causal mask (set future positions to `-inf` before softmax) if this is a decoder.

### 5. Given each head now produces its own masked, weighted output, how do the heads recombine into one block's output?
Concatenate heads, project back to model width, add the residual, apply LayerNorm.

### 6. Given attention's output is now normalized, what's the SECOND sub-layer every transformer block also needs?
Feed through a two-layer FFN with a non-linearity (GELU/ReLU), add another residual, another LayerNorm.

### 7. Given one full block (attention + FFN) is built, how does that become an actual language model?
Stack N of these blocks; a final linear "LM head" maps to vocabulary logits.

### Runnable code
```python
import torch
import torch.nn as nn
import math

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        B, T, D = x.shape
        q = self.w_q(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)  # [B, h, T, d_k]
        k = self.w_k(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)  # [B, h, T, T]
        if causal:
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        out = weights @ v  # [B, h, T, d_k]
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.w_o(out)

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, num_heads)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # pre-norm residual
        x = x + self.ffn(self.ln2(x))    # pre-norm residual
        return x

class TinyGPT(nn.Module):
    def __init__(self, vocab_size: int, d_model=128, num_heads=4, d_ff=512, num_layers=4, max_len=64):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))

# smoke test: random tokens, one training step, verify loss decreases
vocab_size = 50
model = TinyGPT(vocab_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
x = torch.randint(0, vocab_size, (8, 16))
targets = torch.randint(0, vocab_size, (8, 16))

for step in range(5):
    logits = model(x)  # [B, T, vocab_size]
    loss = nn.functional.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"step {step}: loss={loss.item():.4f}")
```

### Where this connects to my current research
Building this from raw matrix multiplies, rather than just calling `nn.MultiheadAttention`, is directly relevant to what I'm doing right now at UNT: my research on LLM hallucination mitigation for healthcare requires actually reasoning about *why* a model attends to (or fails to attend to) specific retrieved passages, not just treating the retrieval-then-generation pipeline as two black boxes glued together. Understanding exactly how Q/K/V, the causal mask, and the softmax normalization interact is what lets me debug "the retrieved passage clearly contains the answer, so why did the model still hallucinate" as an attention/context problem rather than shrugging and re-prompting until it works.

### Common pitfalls
- **If you forget to divide by `sqrt(d_k)`, the model still "runs" but trains badly or not at all** — large dot products push softmax into a near-one-hot regime with vanishing gradients almost everywhere else; this is a silent failure, not a crash, which is why interviewers specifically probe whether you know *why* the scaling term exists.
- **If a decoder attends to future tokens, it's because the causal mask was applied *after* softmax instead of before** (or omitted) — masking with `-inf` must happen on the raw scores, before softmax, so those positions get exactly zero probability; masking the softmax *output* directly breaks the row-sums-to-1 property.
- **If your loss doesn't move at all across steps, it's because the learning rate is too small for random-init weights, or you're not calling `optimizer.zero_grad()`** before `backward()` — gradients accumulate by default in PyTorch, so omitting `zero_grad()` silently sums gradients across steps.

---

## PyTorch: CNN Image Classification

### Plain-English explanation
A CNN learns spatial filters (edges, textures, then shapes) via convolution, using far fewer parameters than a fully-connected network would need for the same image size, because the same small filter is reused ("shared weights") across every spatial location.

### Built as a chain: from local patterns to class logits

### 1. What's the basic repeating building block a CNN stacks to extract spatial features?
Stack `Conv2d → activation → pooling` blocks; each conv layer detects local patterns, pooling downsamples and adds translation tolerance.

### 2. Given multiple conv/pool blocks are stacked, how do the feature maps' shape typically change from the first block to the last?
Channel depth typically increases while spatial size decreases through the network (fewer, larger-area, more-abstract feature maps).

### 3. Given the final feature maps are small and deep (question 2), how do they turn into actual class predictions?
Flatten the final feature maps and feed through one or more fully-connected layers to the class logits.

### 4. Given logits now exist, how do you actually train this end to end, and what do you watch for while doing it?
Train with cross-entropy loss + an optimizer (Adam is a common default), tracking both train and validation accuracy to catch overfitting.

### Runnable code
```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 32x32 -> 16x16
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 16x16 -> 8x8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128), nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# Synthetic stand-in for CIFAR-10-shaped data (swap in torchvision.datasets.CIFAR10 for the real thing)
images = torch.randn(64, 3, 32, 32)
labels = torch.randint(0, 10, (64,))
loader = DataLoader(TensorDataset(images, labels), batch_size=16, shuffle=True)

model = SimpleCNN()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

model.train()
for epoch in range(3):
    total_loss = 0.0
    for xb, yb in loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"epoch {epoch}: avg_loss={total_loss/len(loader):.4f}")
```

### Where I've actually built this
This toy CNN is deliberately trained from scratch to show the mechanics, but my real image-classification work uses **transfer learning** rather than training from random init, because medical-imaging datasets are almost never large enough to train a good CNN from zero: for Alzheimer's disease-stage classification from brain MRI scans, I used a **ResNet18 backbone pretrained on ImageNet**, in PyTorch, fine-tuning it on the MRI data rather than training convolutional filters from scratch — and got 98% classification accuracy, which a from-scratch CNN on a comparatively small medical dataset would be very unlikely to reach without overfitting badly first. The practical difference from the code above: instead of `nn.Sequential` with fresh `Conv2d` layers, you load `torchvision.models.resnet18(weights="IMAGENET1K_V1")`, replace the final fully-connected layer to output your number of classes, and typically freeze the early convolutional blocks (which have already learned general edge/texture/shape detectors that transfer across domains) while fine-tuning the later layers and the new head at a low learning rate — the same freeze-then-adapt logic as the LoRA/PEFT discussion in `core-technical-depth.md`, just applied to vision instead of language.

### Common pitfalls
- **If you get a shape-mismatch error in the first `Linear` layer, it's because you hard-coded the flattened size** without recomputing it after changing conv/pool layers — `32 * 8 * 8` is only correct for this exact architecture on 32x32 input; changing strides/padding/pool counts changes it, and the fix is either recomputing by hand or using `nn.AdaptiveAvgPool2d(1)` to make the classifier size-agnostic.
- **If validation accuracy plateaus far below training accuracy, it's because the model is overfitting the small/synthetic dataset** — the fix is data augmentation, dropout, weight decay, or simply more data, not a bigger model.
- **If training loss doesn't drop at all across epochs, it's because the input wasn't normalized** — raw 0–255 pixel values (or in this synthetic example, un-normalized random floats) put the network far outside the range its weight initialization assumes; always normalize inputs to roughly zero mean/unit variance.

---

## Keras: Text Classification with Embeddings + LSTM

### Plain-English explanation
An `Embedding` layer maps each integer word ID to a dense trainable vector; an `LSTM` reads that sequence of vectors one step at a time, carrying a hidden state forward so word order and long-range dependencies matter for the final prediction — unlike a bag-of-words model, which ignores order entirely.

### Built as a chain: from raw text to a trained classifier

### 1. Before anything can be batched, what shape does raw text need to be forced into?
Tokenize text to integer sequences and pad/truncate to a fixed length (batches need uniform shape).

### 2. Given uniformly-shaped integer sequences exist, how does the `Embedding` layer turn them into something meaningful?
`Embedding(vocab_size, embedding_dim)` turns `(batch, seq_len)` integers into `(batch, seq_len, embedding_dim)` vectors.

### 3. Given a sequence of embedding vectors now exists, how does `LSTM` turn that sequence into a single representation for classification?
`LSTM(units)` consumes the sequence and returns either the final hidden state (for classification, the common case) or the full sequence (if stacking more recurrent layers).

### 4. Given the LSTM's final hidden state exists, how does it become an actual prediction?
A `Dense` layer with sigmoid (binary) or softmax (multi-class) produces the prediction.

### 5. Given a prediction now comes out the other end, what has to match up for training to actually work?
Compile with an appropriate loss (`binary_crossentropy` / `sparse_categorical_crossentropy`) and train with `model.fit`.

### Runnable code
```python
# pip install tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

vocab_size = 10000
max_len = 200

(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=vocab_size)
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_len)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_len)

model = keras.Sequential([
    layers.Embedding(vocab_size, 64, input_length=max_len),
    layers.LSTM(64),
    layers.Dense(32, activation="relu"),
    layers.Dense(1, activation="sigmoid"),
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()
model.fit(x_train, y_train, batch_size=64, epochs=2, validation_split=0.2)

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"test accuracy: {test_acc:.4f}")
```

### An honest note on where my real NLP work has gone
My production text/NLP systems — the Mental Health Wellness Chatbot's crisis detection, QuitBuddy's on-domain conversation handling — use a pretrained LLM (Google Gemini 2.0 Flash, in both cases) with careful prompt engineering and guardrails, rather than a from-scratch embedding+LSTM classifier trained on labeled data. That was the right call for those specific problems: crisis-language detection and staying within strict domain boundaries needed the broad language understanding a pretrained LLM already has, and I didn't have (or need) a large labeled dataset to train a competitive LSTM classifier from scratch. Where the embedding+LSTM pattern in this section *would* be the right tool is a narrower, higher-volume classification task where an LLM API call per request is too slow or too expensive at scale, and a labeled dataset already exists — I'd make that build-vs-call-an-API decision the same way I did for RAG vs. fine-tuning in `core-technical-depth.md`: based on latency/cost budget and data availability, not by default.

### Common pitfalls
- **If accuracy stays near 50% (chance level) for binary classification, it's because the final activation/loss pair is mismatched** — sigmoid output requires `binary_crossentropy`; softmax output for 2 classes requires `categorical_crossentropy` (or `sparse_categorical_crossentropy` for integer labels) — mixing them silently trains the wrong thing.
- **If sequences longer than `max_len` seem to lose important information, it's because truncation cuts from the front by default** — for reviews where the verdict is often in the last sentence, truncating from the end (`truncating='pre'` vs `'post'`) can matter more than people expect; always check which end padding/truncation happens on.
- **If training is extremely slow, it's because a plain LSTM processes the sequence step-by-step and can't parallelize across time** the way a Transformer can — for long sequences, this is exactly the RNN-vs-Transformer tradeoff, and it's a legitimate answer to say you'd consider a Transformer encoder if sequence length or training-time constraints demanded it.

---

## Keras: CNN Image Classification

### Plain-English explanation
Same convolutional principle as the PyTorch CNN above, expressed in Keras's more declarative `Sequential` API — stack conv/pool blocks, flatten, dense to logits.

### Step-by-step mechanics
Identical to the PyTorch version's mechanics section — the only difference here is API surface, not concept.

### Runnable code
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = x_train[..., None]  # add channel dim: (N, 28, 28) -> (N, 28, 28, 1)
x_test = x_test[..., None]

model = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(16, 3, activation="relu"), layers.MaxPooling2D(2),
    layers.Conv2D(32, 3, activation="relu"), layers.MaxPooling2D(2),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax"),
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=128, epochs=3, validation_split=0.1)

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"test accuracy: {test_acc:.4f}")
```

### Where I've actually built this: the Pneumonia Detection classifier
This exact stack — Keras/TensorFlow, `Conv2D`/pooling blocks, a small dense head — is close to what I actually used for the Pneumonia Detection project, with one key difference that mattered a lot in practice: I used **MobileNetV2 transfer learning** rather than training convolutional layers from scratch, since chest X-ray datasets of usable size are limited and a from-scratch CNN would need far more data to reach a clinically-useful accuracy without overfitting. That got to 95% AUC — but the number I actually cared about defending, given it's a medical classifier, was interpretability, not just accuracy: I added **Grad-CAM** (which highlights the image regions that most influenced the model's prediction, as a heatmap overlaid on the X-ray) and **SHAP** (which attributes the prediction to specific input features) specifically so a clinician reviewing a flagged case could see *why* the model said "pneumonia," not just trust a bare probability. That's a pattern I'd bring to any BNSF computer-vision use case too — e.g., a machine-vision defect-detection system flagging a railcar wheel — where "the model is 95% confident" is a much weaker operational answer than "the model is 95% confident, and here's the region of the image driving that."

### Common pitfalls
- **If `model.fit` errors on input shape, it's because you forgot the channel dimension** — Keras `Conv2D` expects `(batch, height, width, channels)`, and grayscale data loaded as `(N, 28, 28)` needs an explicit `[..., None]` (or `np.expand_dims`) to become `(N, 28, 28, 1)`.
- **If you forget to normalize pixel values (dividing by 255), training still runs but converges much slower or gets stuck** — same root cause as the PyTorch pitfall above: inputs far outside the range the initialization/optimizer defaults assume.
- **If you use `categorical_crossentropy` with integer labels (not one-hot), it errors or silently misbehaves** — `sparse_categorical_crossentropy` is the one that accepts plain integer class labels directly; that mismatch is one of the most common Keras beginner errors and worth naming proactively if asked "why sparse."

---

## Practice Q&A (Self-Test)

**Q1. What's the difference between ROW_NUMBER(), RANK(), and DENSE_RANK() when two rows tie on the ORDER BY column, and why does that difference matter for a "top-N per group" query?**
A: ROW_NUMBER() breaks ties arbitrarily and always assigns 1, 2, 3... with no repeats; RANK() gives tied rows the same rank but leaves a gap afterward (1, 2, 2, 4); DENSE_RANK() gives tied rows the same rank with no gap (1, 2, 2, 3). If you need exactly N rows for "top N per group" and use RANK() instead of ROW_NUMBER(), a tie can cause the query to return more than N rows.

**Q2. Why can't you filter directly on a window function in the same SELECT's WHERE clause, and what's the fix?**
A: Window functions execute after WHERE/GROUP BY but before ORDER BY/LIMIT in the logical query plan, so the WHERE clause can't see a window function's result yet. The fix is to compute the window function in a CTE or subquery, then filter on it in the outer query (or use QUALIFY if the dialect supports it).

**Q3. Walk through how you'd write a query to flag temperature jumps of more than 15°C between consecutive readings for the same locomotive.**
A: Use LAG(temp_c) OVER (PARTITION BY locomotive_id ORDER BY reading_time) inside a CTE to pull the previous reading onto the same row without a self-join, compute delta_temp as the current minus the lagged value, then filter the outer query on ABS(delta_temp) > 15. Partitioning by locomotive_id is essential — without it, LAG would look back across locomotive boundaries and produce nonsensical deltas.

**Q4. What's the difference between groupby().agg() and groupby().transform() in pandas, and when would you reach for transform?**
A: agg() collapses rows by key into a summary table, like SQL GROUP BY; transform() performs the same computation but broadcasts the result back to every original row. You'd use transform() when you need a per-row column, such as "this row's value minus its group's mean," rather than a one-row-per-group summary.

**Q5. Why does merge_asof require sorted input, and what does direction='backward' do?**
A: Unlike a regular merge, merge_asof assumes both frames are already sorted by the join column and will silently produce wrong results (or throw "left keys must be sorted") if they aren't — it doesn't sort for you. direction='backward' matches each row to the nearest prior key rather than an exact match, which is what makes it useful for joining an event to the most recent sensor reading before it.

**Q6. Why does Dijkstra's algorithm break with negative edge weights, and what should you use instead?**
A: Dijkstra's greedy strategy finalizes a node's shortest distance as soon as it's popped from the priority queue, assuming that distance can never improve later — a negative edge could still "beat" an already-finalized path, which Dijkstra has no mechanism to undo. Bellman-Ford should be used instead when negative weights are possible.

**Q7. Why is F1 the harmonic mean of precision and recall rather than their arithmetic mean?**
A: The harmonic mean penalizes a lopsided model (e.g., near-perfect precision but terrible recall) far more than an arithmetic mean would. The file notes a model with 100% precision and 1% recall would score roughly 50% under an arithmetic mean, which overstates a nearly-useless model — the harmonic mean pulls that score down much closer to zero.

**Q8. In the from-scratch transformer, why do you divide QK^T by sqrt(d_k) before the softmax, and what happens if you forget?**
A: Without scaling, large dot products push softmax into a near-one-hot regime with vanishing gradients almost everywhere else — dividing by sqrt(d_k) keeps the dot products in a range where softmax gradients stay meaningful. If you forget it, the model still runs without error but trains badly or not at all, which is a silent failure rather than a crash.

**Q9. What accuracy/AUC did the Pneumonia Detection classifier reach, what architecture did it use, and why did the candidate add Grad-CAM and SHAP on top of it?**
A: It used MobileNetV2 transfer learning and reached 95% AUC. Grad-CAM and SHAP were added because a false negative (telling a patient with pneumonia they're clear) is categorically worse than a false positive, and a single threshold-free number like AUC doesn't show where the operating point is set or let a radiologist sanity-check *why* the model flagged a case, which clinical interpretability requires.

**Q10. How does the candidate's actual Alzheimer's MRI classification work differ from the toy from-scratch CNN shown in this file, and what accuracy did it achieve?**
A: Instead of training fresh Conv2d layers from random initialization like the toy CNN, the real project used a ResNet18 backbone pretrained on ImageNet, fine-tuned on the MRI data with the final layer replaced and early conv blocks typically frozen. This transfer-learning approach reached 98% classification accuracy, which the file notes a from-scratch CNN on a comparatively small medical dataset would be unlikely to reach without overfitting first.
