# Data Engineering Practice — Built as a Chain, Not a List

`sql-practice.md` covers how to write a query. This doc covers the layer above that: how data gets arranged into tables worth querying in the first place, and how it keeps arriving correctly, on schedule, without someone manually re-running things by hand. Same chain format as the other docs — each cluster builds on the one before it, and ends with one worked example.

---

## Cluster 1 — OLTP vs. OLAP: Why This Is a Separate Discipline at All

### 1. Doesn't the production database already have all the data?

It does. But that database is built for one specific job.

It's optimized for **OLTP** — Online Transaction Processing. Fast, safe writes, one order at a time, one signup at a time, one row at a time. The data is **normalized**, meaning it's split into many small linked tables so nothing gets stored twice. That's what keeps writes safe and consistent.

**OLAP** — Online Analytical Processing — asks a completely different kind of question. Something like "total sales by region, by month, for the last two years." Answering that means scanning and aggregating millions of rows at once, not writing one.

Running that kind of query against the live production database is risky in two different ways. First, it can slow down or lock the system real customers are using right now. Second, the normalized shape itself is wrong for the job — getting one clean "sales by region" number can mean joining across six or more tables.

### 2. What does "the wrong shape" actually mean?

Normalization is exactly right for OLTP. Update a customer's address once, in one place, and every order referencing them sees the update instantly.

For OLAP, that same design is a liability. Every analytical query has to re-join those same small tables back together, over and over, at scale.

A **data warehouse** exists to solve this. It holds a second copy of the data — denormalized on purpose, shaped for fast aggregation instead of safe single-row writes.

Same data, opposite shape, opposite job:
```
OLTP (production app DB)              OLAP (data warehouse)
  many small normalized tables          fewer, wider, denormalized tables
  optimized for: one row at a time      optimized for: millions of rows at once
  "update this customer's address"      "total revenue by region, last 2 years"
  locks matter, safety matters          scan speed matters, history matters
```

OLTP answers "what is true right now, for this one row." OLAP answers "what happened, in aggregate, across everything." Different questions need different-shaped tables — the whole reason a warehouse exists as a separate copy, instead of just querying the app's database directly.

### Summary example

A retail app's production database normalizes orders, customers, and products into separate linked tables. That's correct — it's what makes processing one checkout safe.

Now ask "what was our best-selling product category last quarter, by region" against that same schema. That needs joining four or more tables and scanning the whole order history, competing with real checkouts for the database's resources.

That's the actual reason this kind of question gets asked against a warehouse's denormalized copy instead — a copy built and refreshed specifically for this kind of aggregation.

---

## Cluster 2 — Star Schema: Fact and Dimension Tables

### 1. If a warehouse is denormalized on purpose, what does that shape actually look like?

The standard shape is a **star schema**: one central **fact table**, surrounded by several **dimension tables**.

```sql
-- fact table: one row per event, mostly foreign keys + numbers
CREATE TABLE fact_orders (
  order_id     BIGINT,
  customer_key INT REFERENCES dim_customer(customer_key),
  product_key  INT REFERENCES dim_product(product_key),
  date_key     INT REFERENCES dim_date(date_key),
  quantity     INT,
  revenue      NUMERIC
);

-- dimension table: descriptive attributes about ONE thing
CREATE TABLE dim_customer (
  customer_key INT PRIMARY KEY,
  customer_id  BIGINT,        -- the original ID from the source system
  name         TEXT,
  region       TEXT,
  signup_date  DATE
);
```

### 2. How do you tell which table is which, just by looking at it?

A **fact table** holds events or measurements. It's mostly foreign keys pointing outward, plus a handful of numeric measures — quantity, revenue, duration. It's also the biggest, fastest-growing table in the warehouse: one row per event.

A **dimension table** holds descriptive context about one entity — customer, product, date, store. It answers "who, what, when, where." Not "how much."

The fact table sits in the middle. Dimensions radiate out like a star:
```
              dim_customer
                    │
dim_product ── fact_orders ── dim_date
                    │
               dim_store
```

Think of the fact table as the verb — an order happened. The dimension tables are the nouns describing it: who, what, when, where. Every analytical question really just means "aggregate the fact table's numbers, sliced by one or more dimension's attributes."

### 3. A star schema still repeats some data in its dimensions. Is there a fully normalized version?

Yes — a **snowflake schema**. It normalizes the dimensions further. `dim_product`, for example, could split into `dim_product` plus `dim_category`. Less redundancy, more joins.

Most warehouses default to star over snowflake anyway. Query simplicity and read speed matter more here than storage savings — the opposite priority from the OLTP side in Cluster 1.

### Summary example

"Revenue by region, by month" against the schema above is one join chain, not a many-table maze.

Join `fact_orders` to `dim_customer` for region, and to `dim_date` for month. Then aggregate: `SUM(revenue) GROUP BY dim_customer.region, dim_date.month`. That's the same `GROUP BY` / `HAVING` mechanics from `sql-practice.md` Cluster 3, running against a schema that was shaped in advance to make this exact kind of question cheap to ask.

---

## Cluster 3 — Slowly Changing Dimensions: When the "Who/What" Itself Changes

### 1. A customer moves from Texas to California. What happens to their past orders?

This is a classic data-engineering interview question, because the obvious answer is usually wrong.

The obvious move is to just `UPDATE` the customer's region in place. But that silently rewrites history. Every past order now appears to have come from California — even the ones placed while the customer still lived in Texas.

Whether that's a problem depends entirely on what the business actually wants to ask later. That's exactly what the three SCD (slowly changing dimension) types below are for.

### 2. What are the three standard ways to handle it?

- **Type 1 — overwrite.** Just update the row. Simple, but history is gone. Fine when the old value genuinely doesn't matter, like correcting a typo in a name.
- **Type 2 — new row, keep history.** Insert a new dimension row for the customer with the updated `region`, and close the old row's validity window.
- **Type 3 — new column for the previous value.** Add a `previous_region` column. Limited — it only remembers exactly one prior state.

```sql
-- Type 2: the standard "real" answer in an interview
customer_key | customer_id | region     | effective_date | end_date   | is_current
    101      |     42      | Texas      | 2024-01-01     | 2025-06-14 | FALSE
    205      |     42      | California | 2025-06-15     | NULL       | TRUE
```

### 3. Walking through what Type 2 actually does, step by step

Say customer 42 moves on 2025-06-15.

1. Before the move, one dimension row exists: `customer_key 101`, `customer_id 42`, `region = Texas`, `is_current = TRUE`.
2. The move happens. Instead of editing that row, a new one is inserted: `customer_key 205`, `customer_id 42`, `region = California`, `effective_date = 2025-06-15`, `is_current = TRUE`.
3. The old row is closed out, not deleted. `customer_key 101` gets `end_date = 2025-06-15`, and `is_current` flips to `FALSE`. It stays in the table.
4. Every fact row for an order placed before the move still points at `customer_key 101`. Those rows are never touched. They still say Texas — correctly.
5. Every new order placed after the move gets `customer_key 205` when it's loaded. It says California — also correctly.

Type 2 doesn't edit the past. It adds a new present. Old fact rows keep pointing at the old dimension row, so historical reports stay historically accurate, and new fact rows automatically pick up the new one.

### 4. Why does the fact table join on `customer_key` instead of the original `customer_id`?

`customer_id` (42) identifies the same person across every version of them. `customer_key` (101, then 205) identifies which *version* of that person was true at the time a given fact row happened.

That surrogate key, `customer_key`, is the entire mechanism that makes Type 2 work. Without it, there'd be no way to tell which address was current when a given order was placed.

### Summary example

A subscription business needs "signups by region, as of the time of signup" to stay accurate even after customers relocate. That's a hard requirement, and it's a hard requirement for Type 2 specifically.

Here's why Type 1 fails at it: it would retroactively move every past signup to the customer's current region, quietly corrupting a metric the business actually depends on.

The fix costs one extra column pair (`effective_date` / `end_date`) and one extra row per change. In exchange, historical reports stay true to what was actually known at the time.

---

## Cluster 4 — ETL vs. ELT

### 1. Data needs to move from a source system into the warehouse. What's the traditional way?

**ETL — Extract, Transform, Load.** Pull data from the source. Transform, clean, and reshape it in a separate processing step, outside the warehouse. Then load the already-transformed result in.

This was the default back when warehouse compute was expensive and limited. You transformed elsewhere specifically to avoid burning warehouse cycles on it.

### 2. What changed, and what's the modern default?

**ELT — Extract, Load, Transform.** Load the raw data into the warehouse first, untransformed. Then run the transformation as SQL — or a tool like dbt — inside the warehouse, using its own cheap, elastic compute.

Cloud warehouses (Snowflake, BigQuery, Redshift) made warehouse compute cheap and scalable. That flipped the economics. Transforming in-warehouse is now usually simpler to build and easier to debug than a separate transformation layer.

Where the T happens is the whole difference:
```
ETL:  source ──▶ [ TRANSFORM (separate compute) ] ──▶ load ──▶ warehouse
ELT:  source ──▶ load (raw)  ──▶ warehouse ──▶ [ TRANSFORM (warehouse's own compute) ]
```

ETL cleans the data before it's let into the house. ELT lets it in raw, and cleans it at the kitchen table instead. The second approach only makes sense once the kitchen — warehouse compute — got cheap enough not to mind the mess arriving.

### 3. Does ELT throw away the raw data once it's transformed?

No. Keeping it is one of ELT's real advantages.

The raw load usually stays as-is, in a "raw" or "staging" layer. The transformed result gets materialized separately, often as views or tables built with **dbt**.

So if a transformation turns out to be wrong, you re-run it against the untouched raw data. You don't have to re-extract from the source system all over again.

### Summary example

A source system's API is slow, rate-limited, and only available a few hours a day. (This is the same shape of constraint the Vela-style import-pipeline design leans on.)

Under ETL: a transformation bug found a week later means re-pulling from that same slow, rate-limited source to fix it.

Under ELT: the raw extract from a week ago is still sitting untouched in the warehouse's staging layer. Fixing the bug just means re-running the transformation SQL against data already in hand — no second trip to the fragile source system required.

---

## Cluster 5 — Batch vs. Streaming

### 1. Data lands in the warehouse via ETL or ELT. How often does that actually run?

**Batch processing** — data is collected and processed in chunks on a schedule. Hourly, nightly, whatever fits. Not the instant it's created.

This is the default, and for most analytical questions it's the right default. A "sales by region" report doesn't need to reflect an order placed four seconds ago.

### 2. When is batch not good enough, and what replaces it?

**Streaming.** Each event gets processed individually, as it arrives, often within milliseconds.

The tradeoff is real complexity and cost. A streaming system — a Kafka-style event stream, or a stream-processing engine — has to stay running continuously. It has to handle out-of-order and duplicate events, and manage state across a never-ending flow of data. A nightly batch job doesn't have any of these problems, because it processes one bounded chunk and then it's done.

### 3. How do you actually decide which one a problem needs?

Ask what staleness actually costs.

A nightly sales dashboard being 18 hours behind costs nothing real. Fraud detection being even 30 seconds behind can mean the fraudulent transaction already went through — that gap is the entire decision.

**Micro-batching** — processing every few seconds or minutes instead of every few hours — is a common middle ground. Use it when true event-by-event streaming's complexity isn't justified, but nightly batch is genuinely too slow.

### Summary example

A retailer needs both a monthly executive revenue dashboard and real-time fraud detection on card transactions. These are not the same engineering problem wearing different clothes.

The dashboard is a textbook nightly batch job — simple, cheap, and a day of latency is invisible to a monthly number.

Fraud detection needs streaming. The entire value of catching fraud collapses if detection lags behind the transaction. Same cost-of-staleness question as above, two different, both-correct answers for the same company.

---

## Cluster 6 — Orchestration: Pipelines as DAGs

### 1. A real warehouse has dozens of interdependent jobs. How do you keep that from turning into chaos?

Model the whole pipeline as a **DAG** — a Directed Acyclic Graph of tasks. An edge means "this task must finish before that one starts."

"Acyclic" means no loops. Task A can never depend, directly or indirectly, on a task that depends on A. That guarantees the whole pipeline has a valid finishing order, and can't get stuck waiting on itself.

```
extract_orders ──┐
                  ├──▶ transform_fact_orders ──▶ build_revenue_dashboard
extract_customers┘                          ↑
                                    transform_dim_customer
```

A DAG is just "what has to finish before what." The two raw extracts above can run in parallel — nothing connects them. Both have to finish before the transform step that needs both of them can start. The dashboard step waits on the transform step, not on the raw extracts directly.

The orchestrator's whole job is reading this graph and running each task the moment its upstream dependencies are done — in parallel wherever the graph allows it, not just running everything in one fixed sequence every time.

### 2. What happens when one task in the middle of the DAG fails?

Everything downstream of that task should get skipped. There's no point building a dashboard from a transform that never ran.

Independent branches of the DAG — the ones with no dependency on the failed task — should keep running. It's the same instinct as isolating one bad record instead of failing an entire 10,000-record import, just applied at the task level instead of the record level.

A failed task typically retries a bounded number of times, with backoff, before the whole run gets marked failed and someone gets paged.

### 3. A bug shipped three weeks ago. Thirty days of data need reprocessing. How do you do that without re-running everything by hand?

This is a **backfill**. It only works cleanly if every task is **idempotent** — running it twice for the same date gives you the same result as running it once, not duplicated data.

Design every task to `DELETE` + `INSERT`, or `UPSERT`, for its specific date or partition, instead of blindly `APPEND`. Do that, and a backfill becomes "re-run this task for these 30 dates" — safely, instead of a special one-off procedure someone has to reason through from scratch.

### Summary example

A transformation bug has been silently under-counting revenue for two weeks. Because every transform task in the DAG is idempotent — it upserts by date, never blind-appends — the fix is simple: correct the transformation logic, then re-trigger the DAG for each of the last 14 dates. Each run cleanly replaces that date's previously-wrong output instead of stacking a second, duplicate copy of two weeks of revenue on top of the first.

---

## Cluster 7 — Data Quality: What Actually Breaks Pipelines Silently

### 1. A DAG can fail loudly — a task errors out, someone gets paged. What's actually more dangerous?

A pipeline that **succeeds** while producing wrong data.

No task errored. No alert fired. The dashboard just quietly shows the wrong number until a human happens to notice, often weeks later. This is why data-quality checks need to be a first-class step in the DAG, not something a human eyeballs occasionally.

### 2. What are the concrete failure modes worth checking for automatically?

- **Schema drift** — a source system adds, removes, renames, or retypes a column with zero warning. A pipeline reading it either breaks loudly (best case) or silently drops or mangles the field (worst case).
- **Volume anomalies** — today's row count is 40% of yesterday's, meaning an upstream extract silently failed partway through. Or it's 300% of normal, meaning an unexpected duplicate load happened.
- **Null spikes** — a field that's normally 99% populated suddenly comes in 40% NULL. That means something broke upstream, not that customers suddenly stopped providing that field.
- **Referential integrity gaps** — a fact row references a dimension key that doesn't exist yet. Often a **late-arriving dimension**: the order event arrived before the customer record describing it did.

### 3. What's the actual fix for that last one — a fact row arriving before its dimension?

Two options, and neither is free.

Hold the fact row in a staging area until its dimension exists. That adds latency, but guarantees correctness. Or insert a placeholder "unknown" dimension row immediately, and backfill the real attributes once the dimension catches up. That adds no latency, but downstream numbers are briefly incomplete.

Which one is right depends on whether the consuming report can tolerate a few "unknown customer" rows for a few minutes — the same cost-of-staleness question as Cluster 5.

### Summary example

A source system silently renames a `country` column to `country_code` during an unannounced migration.

A pipeline with no schema-drift check just keeps running, no error raised. The column the transform expects is simply gone, so every downstream `region` field silently fills with NULL. The revenue-by-region dashboard quietly goes wrong for however long it takes someone to notice.

A schema check — comparing today's source columns against yesterday's, run as its own DAG task before the transform step — catches this the same day instead of three weeks later.

---

## Cluster 8 — Partitioning and Incremental Loads

### 1. A fact table has 5 billion rows and keeps growing. What keeps queries fast?

**Partitioning.** Physically split the table into smaller chunks, almost always by date. A query for "last week's orders" then only scans last week's partition, instead of all 5 billion rows.

This is the warehouse-scale version of the same instinct behind `sql-practice.md` Cluster 8's indexing: give the engine a way to skip the rows it doesn't need, instead of scanning everything.

```sql
CREATE TABLE fact_orders (
  order_id BIGINT, customer_key INT, revenue NUMERIC, order_date DATE
) PARTITION BY RANGE (order_date);
```

### 2. The table is partitioned by date. How does tonight's job avoid reprocessing all 5 billion historical rows?

**Incremental load.** Only extract and process rows newer than the last successful run's watermark: `updated_at >= last_watermark`. It's the exact same watermark pattern used for incremental sync in an API-import pipeline.

A **full reload** — reprocessing everything from scratch — gets reserved for backfills (Cluster 6), or for catching updates and deletes that an incremental, append-only load would otherwise miss entirely.

Partitioning decides what a *query* scans. Incremental loading decides what a *job* processes:
```
partitioning:         query for "last week" touches ONLY last week's partition
incremental loading:  tonight's job processes ONLY rows changed since last night's watermark
                       (both are the same idea — "only touch what actually changed" —
                        applied to reads vs. applied to writes)
```

### 3. What risk does incremental loading introduce that a full reload never has?

Missing **late-arriving or backdated data**.

A row whose `updated_at` timestamp doesn't reflect when it actually needs to be picked up — a batch correction applied to old records, or a source system with clock skew — can silently fall outside the watermark window. It never gets picked up by any run.

This is exactly why a full reload still runs periodically, even when incremental is the everyday default. Same incremental-vs-full-sync tradeoff, for the same reason, as an API import pipeline's own deletion-detection full-sync pass.

### 4. Walking through why a backdated correction slips through, step by step

Say `fact_orders` is partitioned by `order_date`, and loaded incrementally every night on `updated_at >= last_watermark`.

1. A customer-service agent corrects an order's revenue field. The order itself was placed 45 days ago.
2. The correction happens today, so the row's `updated_at` gets set to today's timestamp.
3. Tonight's incremental job filters on `updated_at >= last_watermark`. Today's timestamp passes that filter, so the row does get picked up — nothing is missed here.
4. But the row's `order_date` is still 45 days old. It lands in the 45-day-old partition, not today's.
5. A downstream report that only queries "today's partition" would miss this correction entirely. A report that queries the full table would still catch it.

That gap between step 4 and step 5 is a concrete reason a monthly full-reload safety net still exists, even with incremental loads running flawlessly every night.

---

## Practice Q&A (Self-Test)

### A dimension table's `UPDATE` in place vs. inserting a new row with an `effective_date` — which SCD type is which, and which one preserves history?
`UPDATE` in place is Type 1 (overwrite, history lost). Inserting a new row with `effective_date`/`end_date` is Type 2 (preserves full history, since old fact rows still point at the old, unchanged dimension row via its surrogate key).

### Why does ELT generally beat ETL once you're running on a cloud warehouse, when both eventually apply the same transformation?
Cloud warehouse compute is cheap and elastic, so running the transform inside the warehouse (ELT) is simpler to build and debug than maintaining a separate transformation layer (ETL). ELT also keeps the untransformed raw data on hand in the warehouse, so a transformation bug gets fixed by re-running SQL against data already there, not by re-extracting from the source system again.

### A DAG task fails halfway through a nightly run. What should happen to the tasks downstream of it, and what should happen to unrelated parallel branches?
Everything downstream of the failed task should be skipped, since it would be building on data that never got produced. Unrelated parallel branches of the DAG, with no dependency on the failed task, should keep running — a DAG's whole value is knowing which parts of a pipeline actually depend on which other parts, instead of treating one failure as a reason to stop everything.

### A pipeline task appends new rows every run instead of upserting by key. What breaks the first time it needs to be backfilled?
It isn't idempotent — re-running it for a date it already processed adds a second, duplicate copy of that date's data instead of cleanly replacing it. Backfills only work safely when every task upserts or replaces by its partition key instead of blindly appending.

### An incremental load filters on `updated_at >= last_watermark` every night. What's the one category of data change this can silently miss, and what's the standard safety net?
A backdated correction whose `updated_at` doesn't reflect when it logically needs to be picked up, or any change from a source with clock skew, can fall outside the watermark window and never get loaded. The standard safety net is a periodic full reload, run specifically to catch what the day-to-day incremental process structurally can't.

---

## How this connects to the rest of the hub

Everything above is the layer that has to exist correctly before `system-design-prep.md`'s ML framework — feature pipeline, training, serving, monitoring — has anything to train on. It's also the layer that has to exist before `sql-practice.md`'s queries have a well-shaped table to run against.

If a feature pipeline looks broken, check it against Cluster 7's failure modes first. Schema drift and late-arriving dimensions cause far more "the model's inputs look wrong" incidents than the model itself ever does.
