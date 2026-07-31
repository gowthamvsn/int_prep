# SQL Practice — Built as a Chain, Not a List

Almost every real data-science job starts with "the data is in a database, go get it." Pandas can't help you until the data is already in Python — SQL is how it gets there. Each cluster below is one continuous thread — every question inherits the answer before it, closing with a worked summary example.

---

## Cluster 1 — The Query Itself: Structure and Evaluation Order

### 1. What's the actual minimal query, and what order does it run in?
```sql
SELECT column_a, column_b
FROM table_name
WHERE column_a > 100
ORDER BY column_b DESC
LIMIT 10;
```
Read it in THIS order (not top-to-bottom — this is the order the database actually evaluates it in): `FROM` (get the table) → `WHERE` (filter rows) → `SELECT` (pick columns) → `ORDER BY` (sort) → `LIMIT` (cut down). Knowing this evaluation order is what explains most "why can't I do that" moments in every cluster below.

**Visual + memory hook — the order SQL actually RUNS in:**
```
FROM  ──▶  WHERE  ──▶  GROUP BY  ──▶  HAVING  ──▶  SELECT  ──▶  ORDER BY  ──▶  LIMIT
 get the    filter       collapse       filter      pick        sort          cut
 table(s)   raw rows     into groups    GROUPS      columns/    the result    down
                                                     aliases
```
**Remember it as "the table exists before you can filter it, and groups exist before you can filter THEM"** — that single ordering fact resolves most confusion in this doc before it even comes up.

### Summary example
`SELECT customer_name FROM customers WHERE age > 30 ORDER BY customer_name LIMIT 5` runs FROM (get customers) → WHERE (keep age>30) → SELECT (project just the name) → ORDER BY (alphabetize) → LIMIT (first 5) — never top-to-bottom as written, which is exactly why a column alias defined in `SELECT` can't be referenced in that same query's `WHERE` clause (WHERE runs before SELECT even exists yet).

---

## Cluster 2 — Combining Tables

### 1. How do you combine rows from two tables at all?
```sql
SELECT o.order_id, c.customer_name, o.total
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id;
```
Plain `JOIN` means **INNER JOIN** — only rows that match in BOTH tables.

### 2. What if you want every row from one table, even the ones with no match?
- **LEFT JOIN** — every row from the left table, matched columns from the right filled with `NULL` if there's no match. This is the one you actually want 80% of the time in analysis ("show me every customer, even ones with zero orders").
- **RIGHT JOIN** — the mirror image; rarely used because you can just swap table order and use LEFT JOIN instead.
- **FULL OUTER JOIN** — everything from both sides, `NULL` wherever there's no match on either side.

**Visual + memory hook — the four joins as overlapping circles (A = left table, B = right table):**
```
INNER JOIN                LEFT JOIN                 FULL OUTER JOIN
  ______   ______           ______   ______           ______   ______
 /      \ /      \         /######\ /      \         /######\ /######\
|   A    X    B   |       |   A  ##X##  B   |       |   A  ##X##  B   |
 \______/ \______/         \######/ \______/         \######/ \######/
  keep ONLY the           keep ALL of A, plus       keep EVERYTHING,
  overlap (X)             the overlap (X)           NULL where no match

RIGHT JOIN is just LEFT JOIN's picture with the shading flipped to the other circle.
```
**Remember it as:** the shaded region is what survives. If you can redraw these three circles from memory, you already know which join to reach for.

### 3. Given that INNER only keeps the overlap, what's the concrete trap that catches people?
An INNER JOIN silently drops rows with no match — that's the un-shaded crescent in the LEFT JOIN picture above, gone entirely with no warning. If a report's numbers look too low, check whether an INNER JOIN quietly threw away exactly the rows you needed.

### Summary example
"Show revenue per customer, including customers who've never ordered" REQUIRES `LEFT JOIN customers TO orders` (every customer survives, `NULL`/0 for non-orderers) — using `JOIN` (inner) instead would silently drop every zero-order customer from the report, understating the true customer count with no error or warning anywhere in the output.

---

## Cluster 3 — Aggregating: GROUP BY and HAVING

### 1. How do you compute one summary row per group?
```sql
SELECT customer_id, COUNT(*) AS num_orders, SUM(total) AS lifetime_value
FROM orders
GROUP BY customer_id
HAVING COUNT(*) > 5;
```

### 2. Given Cluster 1's evaluation order (WHERE before GROUP BY, GROUP BY before HAVING), why can't you filter on `COUNT(*)` in `WHERE`?
`WHERE` filters rows BEFORE grouping happens; `HAVING` filters GROUPS after aggregating. `WHERE COUNT(*) > 5` is a syntax error — `COUNT(*)` doesn't exist yet at the point `WHERE` runs, exactly per the evaluation-order diagram in Cluster 1.

### Summary example
"Customers with more than 5 orders and total spend over $10,000": `WHERE` can't help here since neither condition exists until after grouping — the full query is `GROUP BY customer_id HAVING COUNT(*) > 5 AND SUM(total) > 10000`, both conditions correctly living in `HAVING` because both depend on the aggregated group, not the raw pre-grouped rows.

---

## Cluster 4 — Window Functions

### 1. `GROUP BY` collapses rows into one per group. What if you want the aggregate WITHOUT losing the individual rows?
```sql
SELECT
  employee_id,
  department,
  salary,
  AVG(salary) OVER (PARTITION BY department) AS dept_avg_salary,
  RANK() OVER (PARTITION BY department ORDER BY salary DESC) AS salary_rank
FROM employees;
```
A **window function** computes an aggregate WITHOUT collapsing rows — every employee keeps their own row, with their department's average salary and rank attached as extra columns. `PARTITION BY` is "group by, but don't collapse"; `OVER (...)` is what makes it a window function instead of a plain aggregate.

**Visual + memory hook — GROUP BY collapses, OVER keeps every row:**
```
GROUP BY department              OVER (PARTITION BY department)
  (collapses to 1 row/dept)        (keeps every row, attaches the group stat)

department | avg_salary          employee | department | salary | dept_avg
  Sales    |   62000               Alice  |   Sales    | 58000  |  62000
  Eng      |   85000               Bob    |   Sales    | 66000  |  62000
                                    Carol  |   Eng      | 85000  |  85000
```
**Remember it as:** `GROUP BY` answers "one row per group," `OVER (PARTITION BY ...)` answers "every original row, PLUS its group's stat riding along" — the same underlying aggregate math, a fundamentally different output shape.

### 2. Beyond `AVG`/`RANK`, what other window functions come up constantly?
- `ROW_NUMBER()` — 1,2,3,... no ties, ever.
- `RANK()` — ties share a rank, next rank SKIPS (1,2,2,4).
- `DENSE_RANK()` — ties share a rank, next rank does NOT skip (1,2,2,3).
- `LAG()`/`LEAD()` — previous/next row's value within the partition — the standard way to compute "change from last month" in SQL.
- `SUM(...) OVER (ORDER BY date)` — a running total.

### Summary example
Ranking employees by salary within their own department, keeping every employee visible (not collapsed): `RANK() OVER (PARTITION BY department ORDER BY salary DESC)` gives each employee a department-relative rank in their own row — two employees tied for 2nd place both get rank 2, and the next employee gets rank 4 (RANK skips), or rank 3 if you'd used `DENSE_RANK` instead.

---

## Cluster 5 — CTEs (Common Table Expressions)

### 1. Queries can get deeply nested — how do you keep a multi-step query readable?
```sql
WITH high_value_customers AS (
  SELECT customer_id, SUM(total) AS lifetime_value
  FROM orders
  GROUP BY customer_id
  HAVING SUM(total) > 10000
)
SELECT c.customer_name, h.lifetime_value
FROM high_value_customers h
JOIN customers c ON c.customer_id = h.customer_id;
```
A CTE (`WITH ... AS (...)`) is a named, temporary result you can query like a table — it exists only for this one query. Same result as nesting the subquery inline, but readable TOP-TO-BOTTOM instead of inside-out, and you can reference the same CTE more than once, or chain several (`WITH a AS (...), b AS (...) SELECT ...`).

### Summary example
Finding which high-value customers (from Cluster 3's aggregation logic) also placed an order in the last 30 days requires combining an aggregation with a join — nesting that as a raw subquery reads inside-out and gets hard to follow past two steps; naming it `high_value_customers` via a CTE, then joining that named result to a second CTE for "recent orders," keeps each logical step readable on its own line, top to bottom.

---

## Cluster 6 — Finding and Removing Duplicates

### 1. How do you find duplicate rows by a specific key?
```sql
SELECT customer_id, email, COUNT(*)
FROM customers
GROUP BY customer_id, email
HAVING COUNT(*) > 1;
```
The same `GROUP BY`+`HAVING` pattern from Cluster 3, just applied to finding duplicates instead of summarizing.

### 2. Once found, how do you actually DELETE the duplicates, keeping only one?
```sql
-- keep only the first occurrence per key
DELETE FROM customers
WHERE ctid NOT IN (
  SELECT MIN(ctid) FROM customers GROUP BY email
);
```
(`ctid` is Postgres-specific row identity; other engines use `ROW_NUMBER() OVER (PARTITION BY email ORDER BY id) AS rn` in a CTE — combining Clusters 4 and 5 — then `DELETE ... WHERE rn > 1`.)

### Summary example
Cleaning up accidentally-duplicated customer signups sharing the same email: first `GROUP BY email HAVING COUNT(*) > 1` confirms how many duplicates exist and which emails are affected, THEN the `DELETE ... WHERE ctid NOT IN (SELECT MIN(ctid) ...)` actually removes all but the earliest row per email — running the DELETE without first confirming the count is exactly the kind of step worth doing in that order, not skipping straight to deletion.

---

## Cluster 7 — Handling NULLs Correctly

### 1. Why does `WHERE column = NULL` return zero rows, even for rows that genuinely have NULL in that column?
`NULL` means "unknown," and unknown compared to ANYTHING (via `=`, `!=`, `<`, `>`) evaluates to unknown, not true — so no row ever passes a `= NULL` or `!= NULL` comparison, regardless of the data.

### 2. Given that, how do you actually test for NULL-ness?
```sql
WHERE column IS NULL
WHERE column IS NOT NULL
```

### 3. Do aggregates like `COUNT`/`SUM`/`AVG` error out on NULLs, or silently do something else?
They silently SKIP NULLs rather than erroring — which can quietly bias an average if you expected NULLs to count as zero (a NULL discount doesn't average in as a $0 discount, it's simply excluded from the average entirely, shifting the result).

### Summary example
Computing average discount given, where some orders have no discount recorded (`NULL`, not `0`): `AVG(discount)` silently excludes those NULL rows from the average entirely rather than treating them as 0 — if the real intent was "average discount across ALL orders, counting no-discount as 0," the correct query is `AVG(COALESCE(discount, 0))`, not a bare `AVG(discount)`.

---

## Cluster 8 — Performance: Indexes and Query Plans

### 1. What actually makes a query slow?
Usually a **full table scan** — the database checking every single row one by one because it has no faster way to find the matching ones.

### 2. What's an index, and how does it fix that?
An index is a sorted side-structure (usually a B-tree) that lets the database jump straight to matching rows instead of scanning the whole table.

**Visual + memory hook — a full scan checks every page; an index jumps straight there, like a book's index vs. reading cover to cover:**
```
FULL TABLE SCAN                    INDEX SCAN (B-tree)
check row 1... no                       ┌─────┐
check row 2... no                       │ root │
check row 3... no                       └──┬──┘
check row 4... MATCH                    ┌──┴──┐
check row 5... no                    ┌──┤     ├──┐
  ... (every row, one by one)        │  │     │  │
                                    leaf leaf leaf leaf  ──▶ jump straight to the
                                                              matching leaf, done
```
**Remember it as:** an index is a book's index — you don't read every page to find "chapter 7," you jump straight there. Rule of thumb: index columns you filter (`WHERE`) or join on (`ON`) frequently, on large tables — but indexes speed up READS while slowing down WRITES (every `INSERT`/`UPDATE` has to update the index too), which is exactly why you don't index every column by default.

### 3. Given that a slow query might be doing a full scan, how do you actually CONFIRM that instead of guessing?
```sql
EXPLAIN ANALYZE SELECT * FROM orders WHERE customer_id = 42;
```
`EXPLAIN` (or `EXPLAIN ANALYZE` in Postgres) shows the query PLAN — whether it's using an index scan or a full table (sequential) scan. That's the first thing to check when a query is "mysteriously slow," rather than guessing at the cause.

### Summary example
A `WHERE customer_id = 42` query on a 10-million-row `orders` table takes 8 seconds. Running `EXPLAIN ANALYZE` on it reveals a "Seq Scan" (full table scan) in the plan — adding `CREATE INDEX ON orders(customer_id)` and re-running `EXPLAIN ANALYZE` now shows an "Index Scan," and the same query drops to milliseconds, confirmed by the plan itself rather than just "it feels faster."

---

## Cluster 9 — Connecting SQL to Pandas

### 1. Given everything above runs inside the database, how do you actually pull a query's result into Python?
```python
import pandas as pd
from sqlalchemy import create_engine

engine = create_engine("postgresql://user:pass@host:5432/dbname")
df = pd.read_sql("SELECT * FROM orders WHERE total > 100", engine)
```

### 2. Given that you COULD pull a whole table and filter/aggregate it in pandas instead — should you?
No — push filtering, joining, and aggregation into the SQL query itself rather than pulling the whole table into pandas and filtering there. The database is almost always faster at this than pandas, and it means far less data crossing the network, the exact same principle already covered in `python-utilities-practice.md`'s File I/O cluster for reading large files in bounded chunks rather than all at once.

### Summary example
Needing regional sales totals from a 50-million-row `orders` table: doing `GROUP BY region, SUM(sales)` in the SQL query itself and pulling back only the aggregated handful of rows via `pd.read_sql` is dramatically better than pulling all 50 million rows into pandas and running `df.groupby("region")["sales"].sum()` there — same final answer, but one approach transfers a few kilobytes over the network and the other transfers gigabytes for no benefit.

---

## Practice Q&A (Self-Test)

### `WHERE` vs `HAVING` — which one can reference `COUNT(*)`?
`HAVING`. `WHERE` runs before grouping/aggregation happens, so aggregate results like `COUNT(*)` don't exist yet at that point — only `HAVING`, which runs after grouping, can filter on them.

### A LEFT JOIN between orders and customers returns more rows than you expected. What's the most likely cause?
A one-to-many relationship on the join key (e.g. a customer with 3 orders produces 3 joined rows, not 1) — the classic "join fan-out." Check the join key's cardinality before trusting a joined row count.

### `RANK()` vs `DENSE_RANK()` — two employees tie for 2nd place. What does each give the employee in 4th?
`RANK()` gives them rank 4 (ranks 1, 2, 2, 4 — it skips the number matching the tie count). `DENSE_RANK()` gives them rank 3 (ranks 1, 2, 2, 3 — no gaps).

### Why does `SELECT * FROM t WHERE some_col != NULL` return zero rows even when `some_col` has NULLs in it?
Any comparison against `NULL` (`=`, `!=`, `<`, `>`) evaluates to unknown, not true — so no row ever passes. You need `IS NOT NULL` to test for nullness specifically.

### You're about to pull a 50-million-row table into pandas just to run `df.groupby("region")["sales"].sum()`. What's the better move?
Do the `GROUP BY region SUM(sales)` in SQL and pull back only the aggregated result (a handful of rows) — the database engine is built for this and you avoid transferring 50 million rows over the network for no reason.

---

## Video-Sourced Practice MCQs

A second practice set for SQL Practice, built the same way as this hub's NCA-GENL community bank: topics checked against a real YouTube interview-prep video for this subject, then written up as original multiple-choice questions here (the source video mostly asked these as open-ended questions -- the wrong-answer options and their explanations below are original, written to match this hub's "explain every option" convention, not copied from the video). Click an answer, check it, and use "ask about this question" for anything that needs more explanation.

<script type="application/json" class="topic-quiz-data" data-title="SQL Practice">
[
  {
    "d": "Basic Queries",
    "q": "What does `SELECT * FROM employees` return?",
    "o": [
      "Only the first row of the employees table",
      "All columns for every row in the employees table",
      "An error, since * is not valid SQL syntax",
      "A count of how many rows exist in employees"
    ],
    "a": [
      1
    ],
    "e": "Only the first row would need an explicit `LIMIT 1` (or `TOP 1`) clause -- plain SELECT * has no such restriction. A row count is what `SELECT COUNT(*)` returns, a completely different query using an aggregate function. `*` is valid, standard SQL syntax specifically meaning \"all columns\" -- it's not an error. `*` after SELECT is the wildcard meaning \"every column\"; combined with `FROM employees` and no WHERE clause, it returns every column for every row in the table -- if you only need specific fields, you'd name them explicitly instead (`SELECT name, salary FROM employees`)."
  },
  {
    "d": "Basic Queries",
    "q": "What does `SELECT COUNT(DISTINCT country) FROM employees` return?",
    "o": [
      "The total number of rows in the employees table, regardless of duplicates",
      "The alphabetically first country value in the table",
      "The number of DIFFERENT (unique) values that appear in the country column",
      "An error, since COUNT and DISTINCT can't be combined"
    ],
    "a": [
      2
    ],
    "e": "Total row count regardless of duplicates is plain `COUNT(*)` or `COUNT(country)` without DISTINCT -- adding DISTINCT specifically changes the behavior to ignore repeats. Returning one alphabetically-first value describes `MIN(country)`, an unrelated aggregate function. COUNT and DISTINCT combine perfectly validly and commonly -- this is standard, widely-used SQL syntax, not an error. `DISTINCT country` first collapses the country column down to only its unique values (so if 'USA' appears 40 times, it counts once), and `COUNT(...)` around that then tallies how many unique values remain -- e.g. if employees come from exactly 4 different countries, this returns 4, regardless of how many total employees there are."
  },
  {
    "d": "Basic Queries",
    "q": "What happens if you run `UPDATE employees SET country = 'India'` with NO WHERE clause?",
    "o": [
      "EVERY row in the table gets its country column set to 'India' -- the WHERE clause is what normally limits which rows are affected",
      "Only the first row is updated, as a safety default",
      "The query fails with a syntax error",
      "Nothing happens; SQL requires a WHERE clause on every UPDATE"
    ],
    "a": [
      0
    ],
    "e": "SQL does NOT require a WHERE clause -- it's optional, which is exactly what makes omitting it dangerous rather than impossible. There's no \"first row only\" safety default in standard SQL -- that would actually be safer, but it's not how UPDATE behaves; skipping it is a common, costly real-world mistake for that reason. It's syntactically completely valid to omit WHERE -- the query runs without error, which is precisely the risk. The WHERE clause is what restricts an UPDATE to specific rows matching a condition; without it, the SET clause applies to every single row in the table -- so `UPDATE employees SET country = 'India'` with no WHERE would overwrite EVERY employee's country field."
  },
  {
    "d": "Basic Queries",
    "q": "In `SELECT * FROM employees WHERE name LIKE '%a'`, what does the pattern `'%a'` match?",
    "o": [
      "Names that contain the letter 'a' anywhere, but must be exactly 2 characters long",
      "Names that start with the letter 'a'",
      "Names that contain no letter 'a' at all",
      "Names that END with the letter 'a', regardless of how many characters come before it"
    ],
    "a": [
      3
    ],
    "e": "Starting with 'a' would be the pattern `'a%'` -- percent AFTER the letter matches anything following it, the reverse of what's given here. A fixed 2-character length would need the underscore wildcard (e.g. `'_a'` for exactly one character then 'a') -- percent (%) explicitly means zero-OR-MORE characters, not a fixed count. \"Contains no 'a' at all\" is the opposite of what LIKE '%a' searches for. `%` is SQL's wildcard for \"any sequence of zero or more characters,\" so placing it BEFORE the 'a' means \"anything (or nothing), followed by a\" -- i.e., match any name that ends in the letter 'a', regardless of length."
  },
  {
    "d": "DDL & Constraints",
    "q": "What is the key difference between DELETE, TRUNCATE, and DROP when applied to a table?",
    "o": [
      "DROP only removes rows, never the table itself",
      "DELETE removes rows (optionally filtered by WHERE) but keeps the table structure; TRUNCATE removes ALL rows but keeps the table structure; DROP removes the table AND its structure entirely",
      "They are three different names for the exact same operation",
      "TRUNCATE is used to rename a table, not remove data"
    ],
    "a": [
      1
    ],
    "e": "They are absolutely not interchangeable -- using DROP when you meant DELETE destroys the table definition itself, not just its data, which is unrecoverable without a backup. DROP is the most destructive of the three -- it removes the table AND its structure/definition, not just rows, which is the opposite of \"only removes rows.\" TRUNCATE has nothing to do with renaming -- that would be `ALTER TABLE ... RENAME`, an unrelated DDL command. The real distinction: DELETE removes specific rows (filterable with WHERE) while leaving the table itself intact; TRUNCATE is a faster way to remove ALL rows at once (like DELETE with no WHERE) while still leaving the table structure standing; DROP goes further and removes the table's structure entirely -- columns, constraints, indexes, everything -- not just its data."
  },
  {
    "d": "DDL & Constraints",
    "q": "What must be true of a PRIMARY KEY column's values?",
    "o": [
      "There is no restriction; PRIMARY KEY is just a label with no enforcement",
      "They must be unique, but NULL values are allowed since NULL isn't a real value",
      "They must be unique across all rows, AND cannot contain NULL values",
      "They can repeat, as long as no more than two rows share the same value"
    ],
    "a": [
      2
    ],
    "e": "NULL is explicitly disallowed in a primary key, precisely because NULL represents 'unknown,' which would defeat the entire purpose of uniquely identifying a row -- you can't guarantee two NULLs aren't secretly the 'same' unknown value. Allowing up to two repeats describes something like a non-unique index, not a primary key, which must enforce strict one-to-one uniqueness. PRIMARY KEY is actively enforced by the database engine, not just a naming label -- inserting a duplicate or NULL primary-key value raises a constraint violation error. A primary key's whole job is to uniquely identify every row, so the database enforces two things simultaneously: every value must be unique across the table, and NULL is never allowed, since an unknown value can't reliably guarantee uniqueness."
  },
  {
    "d": "Aggregation & Sorting",
    "q": "Why can't you filter on an aggregate function's result (like `COUNT(*) > 5`) using a WHERE clause, and what do you use instead?",
    "o": [
      "WHERE and HAVING are fully interchangeable in every context",
      "HAVING is only used for sorting results, not filtering them",
      "You actually can use WHERE for this; HAVING doesn't exist in standard SQL",
      "WHERE filters individual rows BEFORE grouping/aggregation happens, so it has no aggregate result yet to compare against; HAVING filters AFTER aggregation, once group results (like COUNT per group) actually exist"
    ],
    "a": [
      3
    ],
    "e": "HAVING is a real, standard SQL clause specifically designed for this -- the claim that it doesn't exist is simply wrong. HAVING filters rows/groups, it does not sort them -- sorting is ORDER BY's job entirely, a separate clause. WHERE and HAVING are NOT interchangeable -- swapping WHERE for a per-group aggregate condition causes a SQL error, since the aggregate value doesn't exist at the row-filtering stage WHERE operates on. The actual sequence: WHERE filters individual raw rows before any grouping occurs, so at that stage there IS no aggregate value like a per-group COUNT yet to filter on; HAVING runs after GROUP BY has produced per-group aggregate results, so it can filter groups based on those computed values (e.g. `GROUP BY department HAVING COUNT(*) > 5`)."
  },
  {
    "d": "Aggregation & Sorting",
    "q": "A query nests: an inner query selects the TOP 3 salaries in ascending order, and the outer query takes the FIRST row of that result. What salary does this return?",
    "o": [
      "The third-highest salary in the table",
      "The lowest salary in the table",
      "This is invalid SQL and would produce an error",
      "The highest salary in the table"
    ],
    "a": [
      0
    ],
    "e": "The highest salary would need the inner query sorted DESCENDING with just the top 1 row -- ascending order changes which end of the ranking you land on. The lowest overall salary would be the case if the inner query selected ALL rows ascending and you took the first -- but this inner query is capped at just the top 3, which changes the outcome. Nested subqueries like this are completely valid, standard SQL -- there's no error here. Walking through it: sorting ascending and taking the top 3 gives you the THREE smallest values among what were originally the largest salaries (i.e., the 3rd, 2nd, and 1st highest, now re-ordered smallest-to-largest within that trio); taking the FIRST row of that inner result (the smallest of those three) lands you on the third-highest salary overall -- a classic Nth-highest-value pattern."
  }
]
</script>
<div class="topic-quiz-mount"></div>
