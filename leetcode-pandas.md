# LeetCode Pandas — 35 Real Interview Problems, Answers Included

LeetCode's Pandas category is the most directly DS-relevant part of the whole site — it's testing the exact library you'll use on the job, not general algorithms. These 35 mirror the real technique categories that category actually covers: creation/inspection, selection/filtering, column manipulation, missing data, dedup, type conversion, groupby, merge/concat, reshape (pivot/melt), string/datetime ops, and method chaining. Terse — problem, solution, one-line why. Builds directly on `pandas-practice.md`.

## DataFrame Basics — Creation, Inspection, Selection

**1. Create a DataFrame from a list of lists** with given column names.
```python
df = pd.DataFrame(data, columns=["student_id", "age"])
```

**2. Get the shape of a DataFrame** as `(rows, columns)`.
```python
df.shape
```

**3. Select specific columns**, in a given order.
```python
df[["name", "age"]]
```

**4. Select rows matching a condition** — students older than 20.
```python
df[df["age"] > 20]
```

**5. Select rows matching multiple conditions** — age between 18 and 25 inclusive, gender is "M".
```python
df[(df["age"] >= 18) & (df["age"] <= 25) & (df["gender"] == "M")]
```
*Technique: parentheses around each condition are required — `&`/`|` bind tighter than comparison operators in Python, so `df.age >= 18 & ...` without parens raises an error or silently does the wrong thing.*

**6. Select rows by label vs. by position** — the two very different tools.
```python
df.loc[2]        # by index LABEL (could be 2, could be a string, whatever the index actually is)
df.iloc[2]        # by POSITION (always the 3rd row, regardless of what the index labels are)
```
*Technique: mixing these up is the single most common pandas bug — `.loc` and `.iloc` only agree when the index happens to be the default `0..n-1` RangeIndex.*

## Creating, Modifying, Renaming, Dropping Columns

**7. Create a new column** from existing ones — bonus is double the salary.
```python
df["bonus"] = df["salary"] * 2
```

**8. Create a column with conditional logic** — "Senior" if age >= 40 else "Junior".
```python
df["level"] = np.where(df["age"] >= 40, "Senior", "Junior")
```
*Technique: `np.where` for a simple binary condition — for 3+ branches, `pd.cut` (ranges) or `.apply(lambda...)` (arbitrary logic) instead.*

**9. Rename columns** to a new naming scheme.
```python
df = df.rename(columns={"id": "student_id", "first": "first_name"})
```

**10. Drop columns** that aren't needed.
```python
df = df.drop(columns=["extra_col", "another_col"])
```

**11. Change a column's data type** — a numeric-looking string column to actual int.
```python
df["age"] = df["age"].astype(int)
```

**12. Round a numeric column** to 2 decimal places.
```python
df["salary"] = df["salary"].round(2)
```

## Missing Data & Duplicates

**13. Drop rows with any missing value** in a specific column.
```python
df = df.dropna(subset=["email"])
```

**14. Fill missing values** with a default.
```python
df["quantity"] = df["quantity"].fillna(0)
```

**15. Fill missing values with the column's mean** (a common imputation pattern).
```python
df["score"] = df["score"].fillna(df["score"].mean())
```

**16. Drop exact duplicate rows**, keeping the first occurrence.
```python
df = df.drop_duplicates(keep="first")
```

**17. Drop duplicates based on a subset of columns** — one row per `customer_id`, keep the most recent.
```python
df = df.sort_values("order_date", ascending=False).drop_duplicates(subset=["customer_id"], keep="first")
```
*Technique: sort first, then dedup on a subset — `drop_duplicates` alone has no concept of "most recent" without an explicit sort establishing the order first.*

## GroupBy & Aggregation

**18. Count rows per group.**
```python
df.groupby("department").size()
```

**19. Multiple aggregations at once** per group — mean and max salary per department.
```python
df.groupby("department")["salary"].agg(["mean", "max"])
```

**20. Different aggregations for different columns.**
```python
df.groupby("department").agg(avg_salary=("salary", "mean"), headcount=("employee_id", "count"))
```
*Technique: named aggregation (`new_col=("source_col", "func")`) avoids the awkward multi-level column names a plain `.agg({...})` dict produces.*

**21. Filter groups by an aggregate condition** — departments with more than 5 employees (the pandas equivalent of SQL's `HAVING`).
```python
df.groupby("department").filter(lambda g: len(g) > 5)
```

**22. Find the row with the max value per group** — highest-paid employee per department.
```python
df.loc[df.groupby("department")["salary"].idxmax()]
```
*Technique: `idxmax()` returns the *index label* of the max row per group, then `.loc[...]` pulls the full rows — a common two-step pattern people reach for `apply` first, when this is faster and simpler.*

**23. Rank within groups** — rank each employee's salary within their own department, ties sharing a rank.
```python
df["salary_rank"] = df.groupby("department")["salary"].rank(method="dense", ascending=False)
```
*Technique: `.rank(method="dense")` is pandas' direct equivalent of SQL's `DENSE_RANK() OVER (PARTITION BY ...)`.*

## Merge, Concat, Reshape

**24. Inner join two DataFrames** on a shared key.
```python
pd.merge(orders, customers, on="customer_id", how="inner")
```

**25. Left join, keeping every row from the left table.**
```python
pd.merge(customers, orders, on="customer_id", how="left")
```
*Technique: exactly the same "unmatched left rows get NaN, not dropped" behavior as SQL's LEFT JOIN — see `sql-practice.md`.*

**26. Stack two DataFrames with the same columns** on top of each other.
```python
pd.concat([df_jan, df_feb], ignore_index=True)
```
*Technique: `ignore_index=True` — without it, both DataFrames keep their original 0..n indices, so the combined result has duplicate index labels.*

**27. Pivot long data into wide** — one row per student, one column per subject, values are scores.
```python
df.pivot(index="student", columns="subject", values="score")
```

**28. Melt wide data into long** — the exact inverse of #27.
```python
df.melt(id_vars=["student"], var_name="subject", value_name="score")
```

**29. Pivot table with an aggregation** (plain `.pivot` fails if there are duplicate index/column combinations; `.pivot_table` handles it by aggregating).
```python
df.pivot_table(index="region", columns="quarter", values="revenue", aggfunc="sum")
```

## String, Datetime & Method Chaining

**30. Filter rows where a string column matches a pattern.**
```python
df[df["email"].str.contains(r"^[\w.-]+@leetcode\.com$", regex=True, na=False)]
```
*Technique: `na=False` matters — without it, any row with a missing email raises/propagates NaN through the boolean mask instead of just evaluating to "no match."*

**31. Extract the year from a date column.**
```python
df["order_date"] = pd.to_datetime(df["order_date"])
df["year"] = df["order_date"].dt.year
```

**32. Compute days between two date columns.**
```python
df["days_to_ship"] = (df["ship_date"] - df["order_date"]).dt.days
```

**33. Split a string column into multiple columns.**
```python
df[["first_name", "last_name"]] = df["full_name"].str.split(" ", n=1, expand=True)
```

**34. Method-chain a full transformation** in one readable pipeline — filter, add a column, sort, select — instead of reassigning `df` at every step.
```python
result = (
    df[df["quantity"] > 0]
    .assign(total=lambda x: x["quantity"] * x["price"])
    .sort_values("total", ascending=False)
    [["product_id", "total"]]
)
```
*Technique: `.assign()` instead of `df["total"] = ...` specifically so the new column can be created mid-chain without breaking out to a separate statement — the difference between one readable pipeline and five reassignment lines.*

**Visual + memory hook — the exact same pipe shape as LCEL in `langchain-practice.md`, just DataFrame stations instead of LLM stations:**
```
df  ──▶  [ filter rows ]  ──▶  [ .assign(new col) ]  ──▶  [ sort_values ]  ──▶  [ select cols ]  ──▶  result
          quantity>0            total = qty*price          by total desc         [id, total]
```
**Remember it as the pandas dialect of `grep | sort | uniq`:** each station takes a DataFrame in and hands a DataFrame to the next station — that's *why* it can all sit inside one parenthesized expression with no intermediate variable, the same structural reason LCEL's `|` chains work. The one wrinkle worth flagging: unlike a true Unix pipe or LCEL's `Runnable`, not every pandas method returns something chainable by default (a few mutate in place) — `.assign()` and the bracket/`.pipe()` forms exist specifically to keep every station in the chain returning a fresh DataFrame so the pipe never breaks.

**35. Find products with stock below a reorder threshold, one row per product, most understocked first.**
```python
result = (
    inventory[inventory["stock_quantity"] < inventory["reorder_level"]]
    .sort_values("stock_quantity")
    [["product_id", "stock_quantity"]]
)
```

## Practice Q&A (Self-Test)

### `df.loc[2]` and `df.iloc[2]` sometimes return the exact same row and sometimes return completely different rows. Why?
`.loc` selects by index *label* and `.iloc` selects by integer *position* — they only coincide when the DataFrame's index happens to be the default `0, 1, 2, ...` RangeIndex in original order. After any `sort_values`, `filter`, or `drop_duplicates` that doesn't reset the index, the labels no longer match positions, and `.loc[2]` means "the row labeled 2" (wherever it now sits) while `.iloc[2]` still means "the 3rd row, whatever its label is."

### Problem #17 sorts before calling `drop_duplicates`. What would go wrong if you dropped duplicates first and sorted after?
`drop_duplicates(subset=["customer_id"], keep="first")` keeps whichever row for each customer happens to already be first *in the DataFrame's current row order* — without sorting by date first, "first" has no relationship to "most recent," so you'd get an arbitrary row per customer (whatever order the data originally arrived in) instead of specifically the most recent one.

### Why use `np.where` for problem #8 instead of `df["level"] = "Senior" if df["age"] >= 40 else "Junior"`?
Plain Python `if/else` expects a single boolean, but `df["age"] >= 40` produces a whole Series of booleans (one per row) — Python can't collapse a multi-value boolean Series into one true/false for an `if` statement, and raises an ambiguity error. `np.where(condition, if_true, if_false)` is vectorized: it evaluates the condition and picks a value element-by-element across the whole column at once.

### In problem #22, why call `.loc[groupby(...).idxmax()]` instead of `.groupby("department").max()`?
`.groupby("department")["salary"].max()` only returns the maximum *salary value* per department, losing every other column (employee name, id, etc.) from that row. `idxmax()` instead returns the *index label* of the row holding that maximum per group, so `.loc[...]` can then pull back the complete original row — name, id, everything — not just the number that happened to be the max.
