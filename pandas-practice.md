# Pandas Practice — Built as a Chain, Not a List

`import pandas as pd, numpy as np` assumed throughout. Each cluster is one continuous thread — every question inherits directly from the answer before it, closing with a worked summary example. Every snippet has been run and verified against data of the right shape.

---

## Cluster 1 — Creating and Selecting Data

### 1. How do you create a DataFrame from scratch, with a specific column order?
```python
df = pd.DataFrame({
    "unit": ["4471", "9002", "1187"],
    "wear_pct": [42, 18, 55],
    "depot": ["KC", "LA", "KC"],
})   # dict insertion order = column order in modern Python/pandas — no separate 'columns=' needed
```

### 2. Given a DataFrame, how do you select one specific cell — by its LABEL or by its raw POSITION?
```python
df.loc[0, "wear_pct"]     # LABEL-based: row label 0, column name "wear_pct"
df.iloc[0, 1]              # POSITION-based: first row, second column, by integer position only
```
After filtering or sorting, row LABELS often aren't `0,1,2,...` anymore — `.iloc[0]` always means "the first row physically present," `.loc[0]` means "the row labeled 0," which may not exist or may not be first. Mixing them up is one of the most common real pandas bugs.

**Visual + memory hook — a library shelf (label) vs. "the 3rd book from the left" (position):**
```
Original df:           After df = df.sort_values("wear_pct"):
  label 0: wear=42       label 2: wear=18   ← now PHYSICALLY first
  label 1: wear=18       label 0: wear=42
  label 2: wear=55       label 1: wear=55   ← now PHYSICALLY last

  df.loc[0]  → still finds the row LABELED 0 (wear=42), wherever it now sits
  df.iloc[0] → always the row now sitting FIRST (wear=18) — label is irrelevant
```
**Remember it as:** `.loc` follows the name tag wherever it moves; `.iloc` only cares about physical shelf position, name tags be damned — a fresh `df.reset_index(drop=True)` after sorting is what makes the two agree again.

### 3. Given that, how do you actually filter rows by a condition?
```python
df[df["wear_pct"] > 40]                       # boolean mask, same idea as NumPy
df.query("wear_pct > 40 and depot == 'KC'")    # readable for complex conditions; column names as bare words
```
`.query()` avoids repeating `df["col"]` over and over on long chained conditions, and is often faster on large DataFrames since it can use a faster expression evaluator (numexpr) under the hood.

### 4. Combining multiple conditions inside the bracket form — why does plain `and`/`or` fail?
```python
df[(df["wear_pct"] > 40) & (df["depot"] == "KC")]   # MUST use & and |, not 'and'/'or'
```
Python's `and`/`or` short-circuit on a SINGLE boolean; a pandas boolean Series isn't one boolean, it's element-wise across every row. You need the bitwise `&`/`|` operators, and each condition MUST be wrapped in parentheses because `&` binds tighter than `>` — this is the exact same "not a single boolean" reasoning already covered for NumPy boolean masks in `numpy-practice.md`, just with pandas' extra parenthesization requirement layered on top.

### Summary example
A dataset of 500 units gets filtered and sorted by wear before selecting the worst offender: `df_sorted = df[df["wear_pct"] > 40].sort_values("wear_pct", ascending=False)`. `df_sorted.iloc[0]` correctly gets the single worst row regardless of what label it originally had; `df_sorted.loc[0]` would instead try to find whatever row was ORIGINALLY labeled 0 — which might not even be in the filtered set anymore, or might not be the worst offender at all.

---

## Cluster 2 — GroupBy: Aggregating and Per-Row Group Stats

### 1. How do you compute one summary value per group?
```python
df.groupby("depot")["wear_pct"].mean()                     # one aggregate column
df.groupby("depot").agg(avg_wear=("wear_pct", "mean"),      # named aggregation: clear output column names
                          max_wear=("wear_pct", "max"))
```
The plain `.agg({"wear_pct": ["mean","max"]})` form produces confusing multi-index columns; named aggregation (`agg(name=(col, func))`) gives flat, readable column names directly.

**Visual + memory hook — every groupby, no matter how complex the aggregation, is always the same 3-step shape:**
```
SPLIT                    APPLY                    COMBINE
depot=A: [40,45]    ──▶  mean → 42.5         ──▶  depot | avg_wear
depot=B: [30]       ──▶  mean → 30.0         ──▶    A   |  42.5
depot=C: [50,55,60] ──▶  mean → 55.0         ──▶    B   |  30.0
                                                      C   |  55.0
```
**Remember it as "Split-Apply-Combine"** — the actual technical name for this pattern (from Hadley Wickham's original paper it's named after).

### 2. That COMBINE step collapses to one row per group. What if you need the group's stat attached to EVERY original row instead?
```python
df["dev_from_depot_mean"] = df["wear_pct"] - df.groupby("depot")["wear_pct"].transform("mean")
```
`.agg` draws the right-hand table above (one row per group); `.transform` broadcasts that same result back onto every original row instead of collapsing — essential when you need a per-row column (like "how far is THIS unit from its depot's average"), not a summary table.

### 3. What if the thing you want per group isn't a simple aggregate at all — say, the top 2 rows by wear, per depot?
```python
def top2(group):
    return group.nlargest(2, "wear_pct")
df.groupby("depot", group_keys=False).apply(top2)    # top 2 highest-wear rows PER depot
```
This is still Split-Apply-Combine — APPLY is just a custom function returning multiple rows instead of one number. `group_keys=False` matters here specifically: without it, the groupby key gets added as an extra index level on the result, which isn't wanted when the function already returns full, complete rows.

### 4. After a groupby+aggregate, the group column often ends up as the INDEX, not a normal column — how do you fix that?
```python
df = df.rename(columns={"wear_pct": "wear_percent"})
df = df.groupby("depot")["wear_percent"].mean().reset_index()   # turn the group-by index back into a column
```
`reset_index()` converts the grouping column back to a normal column, which most downstream code (merges, plotting, `to_csv`) expects rather than an index.

### Summary example
Flagging which units are unusually worn FOR THEIR DEPOT (not worn overall): `df["dev_from_depot_mean"] = df["wear_pct"] - df.groupby("depot")["wear_pct"].transform("mean")` gives a per-unit column, then `df[df["dev_from_depot_mean"] > 10]` finds units that are 10+ points worse than their OWN depot's average — a genuinely different, more useful flag than a global `wear_pct > 40` threshold would give, since it accounts for depots that just run hotter/dirtier on average.

---

## Cluster 3 — Combining DataFrames

### 1. How do you combine two DataFrames on a shared key, and how do the join types differ?
```python
pd.merge(df1, df2, on="unit", how="left")   # keep every row of df1; unmatched df2 columns become NaN
pd.merge(df1, df2, on="unit", how="inner")   # keep only rows with a match in BOTH
pd.merge(df1, df2, on="unit", how="outer")   # keep every row from BOTH, NaN where unmatched
```
`inner` silently drops rows that don't match — if `df2` is missing a unit due to a data issue, an inner join hides that unit from your entire downstream analysis with no warning. Default to `left` when `df1` is your "source of truth" and you want to KNOW about missing matches, not silently lose them.

**Visual + memory hook — identical picture to SQL's joins (`sql-practice.md`), pandas is just a different syntax for the same operation:**
```
how="inner"                how="left"                 how="outer"
  ______   ______           ______   ______           ______   ______
 /      \ /      \         /######\ /      \         /######\ /######\
| df1    X   df2  |       | df1  ##X##  df2 |       | df1  ##X## df2  |
 \______/ \______/         \######/ \______/         \######/ \######/
  only the overlap        all of df1 + overlap       everything, NaN
```
**Remember it as:** `how="left"` shades df1 entirely — "every row of df1 survives, no matter what." If a row count shrinks after a merge you expected to be safe, `how` is the first thing to check against this picture.

### 2. What if the join key has a DIFFERENT name in each table?
```python
pd.merge(df1, df2, left_on="unit_id", right_on="unit", how="left")
```

### 3. A merge can silently multiply rows if the key isn't actually unique on one side — how do you catch that before it corrupts your analysis?
```python
pd.merge(df1, df2, on="unit", how="left", validate="one_to_one")   # raises MergeError if not truly 1:1
```
If `df2` accidentally has duplicate `unit` values, a plain merge silently multiplies rows in the output (a many-to-one blowup) with no error — `validate` catches this AT merge time instead of you discovering row counts don't match hours later, deep into an already-corrupted analysis.

### Summary example
Joining a `units` table (1 row per unit) with a `maintenance_events` table (potentially MANY rows per unit): `pd.merge(units, events, on="unit", how="left", validate="one_to_many")` is the honest declaration of intent — if `events` unexpectedly had duplicate rows that would make it "many-to-many" instead, `validate` raises immediately rather than silently producing a row-count blowup that looks like real data until someone notices the totals don't add up.

---

## Cluster 4 — Reshaping Between Wide and Long

### 1. How do you reshape long data (one row per depot-quarter) into a wide pivot table (one row per depot, one column per quarter)?
```python
df.pivot_table(index="depot", columns="quarter", values="wear_pct", aggfunc="mean")
```
If `(depot, quarter)` isn't unique in the source data, `pivot_table` needs to know HOW to combine duplicates into one cell — plain `pivot()` (no `_table`) requires uniqueness and errors otherwise; `pivot_table` always requires an aggregation function for exactly this reason.

### 2. How do you go the other direction — wide back to long?
```python
df.melt(id_vars=["depot"], value_vars=["Q1", "Q2", "Q3"], var_name="quarter", value_name="wear_pct")
```
The exact inverse of `pivot_table` above.

**Visual + memory hook — the shape actually flipping:**
```
LONG (one row per depot-quarter)      WIDE (one row per depot)
 depot  quarter  wear_pct              depot   Q1   Q2   Q3
  KC      Q1       40          pivot    KC     40   45   42
  KC      Q2       45         ──────▶
  KC      Q3       42         ◀──────
  LA      Q1       30          melt     LA     30   35   NaN
  LA      Q2       35
```
**Remember it as:** pivot makes the table WIDER and SHORTER (values move from rows into new columns); melt makes it TALLER and NARROWER (column headers move back into a single "quarter" column) — the two are exact inverses, and you can always sanity-check a pivot by melting it back and confirming you land on the original long data.

### Summary example
Wear percentage logged long-format across 3 quarters for 2 depots. `pivot_table(index="depot", columns="quarter", values="wear_pct", aggfunc="mean")` gives a 2×3 wide table ready for a heatmap; `melt`-ing that wide table back with the same `id_vars`/`value_vars` recovers the original long rows exactly — confirming the reshape didn't silently drop or duplicate anything.

---

## Cluster 5 — Applying Functions: Row-wise, Element-wise, and When to Avoid Both

### 1. How do you apply a custom function to every value in a column vs. across a whole row?
```python
df["wear_category"] = df["wear_pct"].apply(lambda x: "high" if x > 40 else "low")   # element-wise on a Series
df["combo"] = df.apply(lambda row: f"{row['unit']}-{row['depot']}", axis=1)          # row-wise across a DataFrame
```
`df.apply` defaults to `axis=0` (apply down each column); `axis=1` applies ACROSS each row, giving your lambda a full row (Series) to read multiple columns from — forgetting `axis=1` when you need row-wise logic spanning columns is a frequent bug.

### 2. Given that `.apply` works, why avoid it when a vectorized option exists?
```python
# slow: recomputes a Python-level function call per row
df["wear_category"] = df["wear_pct"].apply(lambda x: "high" if x > 40 else "low")
# fast: vectorized C-level comparison, no per-row Python call
df["wear_category"] = np.where(df["wear_pct"] > 40, "high", "low")
```
`.apply` still loops in Python under the hood, one function call per row. On a few hundred rows the difference is invisible; on millions of rows, `np.where`/vectorized string methods can be an order of magnitude faster — the exact same vectorized-vs-Python-loop tradeoff already covered for boolean masking in `numpy-practice.md`.

### Summary example
Categorizing 2 million wear readings as high/low: the `.apply(lambda x: ...)` version takes several seconds of pure Python-loop overhead; `np.where(df["wear_pct"] > 40, "high", "low")` computes the identical result via one vectorized C-level pass — the same logic, the same output, an order-of-magnitude difference in wall-clock time purely from avoiding the per-row Python function call.

---

## Cluster 6 — Handling Missing and Duplicate Data

### 1. Before doing anything else, how do you even check how much data is missing?
```python
df.isna().sum()                      # count of NaNs per column — always check this before modeling
```

### 2. Given that count, how do you actually handle the missing values — drop, or fill?
```python
df.dropna(subset=["wear_pct"])        # drop rows missing THIS column specifically, not any column
df["wear_pct"].fillna(df["wear_pct"].median())   # median is robust to outliers; mean is not
df.fillna(method="ffill")             # forward-fill: carry the last valid value forward (time series)
```
Without `subset`, `dropna()` drops a row if ANY column has a NaN, which can silently discard far more data than intended — always scope it to the columns that actually matter for the current step.

### 3. Separately from missing values, how do you find and handle DUPLICATE rows?
```python
df.duplicated(subset=["unit"]).sum()          # count duplicates by a key, not the whole row
df.drop_duplicates(subset=["unit"], keep="last")   # keep is 'first' by default — 'last' keeps most recent
```

### Summary example
A dataset has 12 missing `wear_pct` values and 5 duplicate `unit` entries (likely re-submitted readings). Checking `df.isna().sum()` first reveals the 12 gaps; `df["wear_pct"].fillna(median)` fills them robustly (median resists the influence of a few extreme outlier readings, unlike mean); `df.drop_duplicates(subset=["unit"], keep="last")` then keeps only each unit's most recent reading — running duplicate-removal BEFORE the fillna would have let a stale duplicate's value influence the median calculation, so the order here matters.

---

## Cluster 7 — Working with Dates and Time Series

### 1. How do you convert a raw string column into an actual datetime type?
```python
df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
df["month"] = df["date"].dt.month
df["day_of_week"] = df["date"].dt.day_name()
```
Without `format=`, pandas guesses the date format per-value — slow on large data, and it can silently misparse ambiguous dates (e.g., `01/02/2026` as Jan 2 vs. Feb 1). `errors="coerce"` turns unparseable values into `NaT` instead of crashing the whole conversion.

### 2. Given real dates, how do you compute a value relative to N periods ago, per entity?
```python
df = df.sort_values("date")
df["wear_change"] = df.groupby("unit")["wear_pct"].diff()        # difference from the PREVIOUS row per group
df["wear_pct_lag1"] = df.groupby("unit")["wear_pct"].shift(1)     # the previous row's raw value per group
```
Without `.groupby("unit")`, `diff()`/`shift()` would compute differences ACROSS different units' boundaries, mixing unrelated time series together — the exact same "forgot to partition" bug already covered for SQL window functions in `sql-practice.md`.

### 3. Given lagged values work per-group, how do you compute a ROLLING statistic (e.g., a 3-reading moving average) the same way?
```python
df["rolling_mean_3"] = df.groupby("unit")["wear_pct"].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)
```
Without `min_periods=1`, the first 2 rows of each group (which don't have 3 prior values yet) become `NaN` — `min_periods=1` lets the window compute with however many values are actually available, avoiding unnecessary NaNs right at the start of each group's history.

### Summary example
Tracking wear trend per locomotive over time: after `pd.to_datetime` and sorting, `groupby("unit")["wear_pct"].diff()` flags any single-reading jump, while `groupby("unit")["wear_pct"].rolling(3, min_periods=1).mean()` smooths out single-reading noise into a 3-reading trend — the diff catches a sudden spike, the rolling mean shows the underlying trajectory, and both are computed per-unit so one locomotive's history never bleeds into another's.

---

## Cluster 8 — Memory, Dtypes, and Safe Mutation

### 1. How do you check which columns are numeric vs. text?
```python
df.select_dtypes(include="number")     # all numeric columns
df.select_dtypes(include="object")      # all string/object columns
```

### 2. Given a large DataFrame, how do you actually check and reduce its memory footprint?
```python
df.memory_usage(deep=True).sum() / 1e6      # actual MB, deep=True includes string object overhead
df["depot"] = df["depot"].astype("category")   # repeated strings -> compact integer codes internally
df["wear_pct"] = pd.to_numeric(df["wear_pct"], downcast="integer")   # int64 -> smallest sufficient int type
```
A column like "depot" with a handful of repeated string values stored as `object` duplicates the FULL string in memory for every row; `category` stores each unique value once and uses small integer codes per row — often a 10x+ memory reduction, and it also speeds up groupby on that column.

### 3. Given a filtered subset of a DataFrame, why does editing it sometimes raise a confusing warning?
```python
# risky: is filtered a view or a copy? pandas isn't always sure, and warns
filtered = df[df["wear_pct"] > 40]
filtered["flag"] = True     # may or may not affect df — the warning exists because this is ambiguous

# safe: explicit copy states your intent
filtered = df[df["wear_pct"] > 40].copy()
filtered["flag"] = True     # unambiguously independent now
```
`SettingWithCopyWarning` isn't decorative — it's pandas telling you it can't guarantee whether `filtered` is a view into `df` or a throwaway copy, so it can't guarantee whether your edit lands on the original data too. `.copy()` removes the ambiguity by stating your intent explicitly — the exact same view-vs-copy distinction already covered for NumPy arrays in `numpy-practice.md`, just triggered here by a filter instead of a reshape.

**Visual + memory hook — the same "one box, two labels" ambiguity as NumPy's view/copy, now with a genuine "pandas isn't sure which one this is" in the middle:**
```
filtered = df[mask]              filtered = df[mask].copy()
                                  
  df ──┐                          df ──▶ [ original data ]
       ├──?──▶ [ maybe shared,     
  filtered ┘    maybe not — even   filtered ──▶ [ independent copy ]
               pandas can't        
               promise which ]     filtered["flag"]=True → df is UNTOUCHED, guaranteed
  filtered["flag"]=True → 
  MIGHT silently touch df too
```

### Summary example
Flagging high-wear units for a report, without corrupting the source data: `filtered = df[df["wear_pct"] > 40].copy()` then `filtered["flag"] = True` — the explicit `.copy()` guarantees `df` itself stays completely untouched, versus skipping it and getting a `SettingWithCopyWarning` that's really pandas saying "I don't know if I just silently edited your original dataset too."

---

## Cluster 9 — String Operations, Binning, and Reading Large Files

### 1. How do you apply multiple string operations to a column at once?
```python
df["depot"] = df["depot"].str.strip().str.upper()      # .str accessor: vectorized string ops, chainable
df["has_kc"] = df["depot"].str.contains("KC", na=False)  # na=False avoids NaN making the whole mask unusable
```
`.str.contains` on a NaN value returns NaN, not True/False — leaving `na` at its default can produce a boolean mask that isn't actually all True/False, breaking downstream filtering. Setting `na=False` explicitly avoids that ambiguity.

### 2. Given a cleaned numeric column, how do you bin it into meaningful categories?
```python
df["wear_bucket"] = pd.cut(df["wear_pct"], bins=[0, 25, 50, 75, 100], labels=["low", "med", "high", "critical"])
```
`cut` uses YOUR bin edges (meaningful thresholds, e.g. regulatory cutoffs); `pd.qcut` instead splits into equal-sized groups by count regardless of what the values actually mean — use `cut` when the boundaries carry real-world meaning, `qcut` when you just want quartiles/deciles with no inherent threshold.

### 3. What if the file itself is too big to load into memory in one shot?
```python
for chunk in pd.read_csv("big_file.csv", chunksize=100_000):
    process(chunk)     # each chunk is a normal DataFrame of up to 100,000 rows
```
A file bigger than available RAM crashes a plain `read_csv`; chunked reading processes it in bounded-memory pieces — the standard fix before reaching for something heavier like Dask or Spark.

### Summary example
Processing a 50 GB CSV of raw sensor strings that won't fit in memory: reading it via `chunksize=100_000` keeps memory bounded; within each chunk, `.str.strip().str.upper()` cleans a messy depot-code column, `pd.cut` buckets wear percentages into regulatory-meaningful categories, and each cleaned chunk gets written out or aggregated incrementally — the same per-chunk cleaning logic that would work fine on a small in-memory DataFrame, just looped over pieces instead of loaded all at once.

---

## Practice Q&A (Self-Test)

**Q1. What's the difference between `df.loc[0, "wear_pct"]` and `df.iloc[0, 1]`, and why can they silently point to different rows after filtering?**
A: `.loc` is label-based — `df.loc[0]` means "the row labeled 0." `.iloc` is position-based — `df.iloc[0]` always means "the first row physically present," regardless of its label. After filtering or sorting, row labels are often no longer 0,1,2,... so `loc[0]` may not exist or may not be the first row anymore.

**Q2. Why does `df[df["wear_pct"] > 40 and df["depot"] == "KC"]` fail, and what's the fix?**
A: Python's `and`/`or` short-circuit on a single boolean, but a pandas boolean Series is element-wise, not a single boolean. The fix is `df[(df["wear_pct"] > 40) & (df["depot"] == "KC")]`, using the bitwise `&`/`|` operators with each condition parenthesized (since `&` binds tighter than `>`).

**Q3. Given `df["dev_from_depot_mean"] = df["wear_pct"] - df.groupby("depot")["wear_pct"].transform("mean")`, why use `.transform` instead of `.agg` here?**
A: `.agg`/plain aggregation collapses each group to one summary row, but `.transform` broadcasts the group's aggregate result back to every original row, preserving the DataFrame's row count. This is required here because the result is a per-row column (each row's deviation from its own group's mean), not a summary table.

**Q4. Why can `pd.merge(df1, df2, on="unit", how="inner")` silently produce misleading downstream analysis?**
A: An inner join keeps only rows with a match in both tables, so any `unit` present in `df1` but missing from `df2` (e.g., due to a data issue) gets silently dropped with no warning. That missing unit disappears from the entire downstream analysis, which is why defaulting to `how="left"` when df1 is the source of truth is the safer default.

**Q5. What does passing `validate="one_to_one"` to `pd.merge` protect against?**
A: It protects against silent row duplication: if `df2` unexpectedly has duplicate `unit` values, a plain merge would multiply rows in the output (a many-to-one blowup) with no error. `validate` raises a `MergeError` immediately instead of letting you discover mismatched row counts hours later.

**Q6. In `df.apply(lambda row: f"{row['unit']}-{row['depot']}", axis=1)`, what does `axis=1` control, and what happens if you forget it?**
A: `axis=1` makes the lambda receive a full row (a Series spanning multiple columns) so it can combine values from `row['unit']` and `row['depot']`. `df.apply` defaults to `axis=0` (applying down each column), so forgetting `axis=1` when row-wise logic across columns is needed is a frequent bug.

**Q7. Why prefer `np.where(df["wear_pct"] > 40, "high", "low")` over `df["wear_pct"].apply(lambda x: ...)` for the same result?**
A: `.apply` still loops in Python under the hood, calling the lambda once per row; `np.where` is vectorized at the C level. On a few hundred rows the difference is invisible, but on millions of rows the vectorized version can be an order of magnitude faster.

**Q8. What does `df.dropna(subset=["wear_pct"])` do differently from a bare `df.dropna()`?**
A: A bare `df.dropna()` drops a row if ANY column has a NaN, which can silently discard far more data than intended. `subset=["wear_pct"]` scopes the drop to only rows missing that specific column, keeping rows that have NaNs elsewhere but valid data in the column that actually matters.

**Q9. Why does converting `df["depot"]` to `astype("category")` reduce memory, and what's the added benefit?**
A: An `object` column stores the full string in memory for every repeated occurrence; `category` stores each unique value once and uses compact integer codes per row instead, often yielding a 10x+ reduction for repetitive string columns like a handful of depot names. It also speeds up groupby operations on that column.

**Q10. Why does `filtered = df[df["wear_pct"] > 40]; filtered["flag"] = True` trigger a `SettingWithCopyWarning`, and what removes it?**
A: pandas can't always guarantee whether `filtered` is a view into `df` or an independent copy, so it warns because the assignment's effect on the original `df` is ambiguous. Explicitly calling `.copy()` when creating `filtered` states your intent and removes the ambiguity, guaranteeing the edit is independent of `df`.

---

## Video-Sourced Practice MCQs

A second practice set for Pandas Practice, built the same way as this hub's NCA-GENL community bank: topics checked against a real YouTube interview-prep video for this subject, then written up as original multiple-choice questions here (the source video mostly asked these as open-ended questions -- the wrong-answer options and their explanations below are original, written to match this hub's "explain every option" convention, not copied from the video). Click an answer, check it, and use "ask about this question" for anything that needs more explanation.

<script type="application/json" class="topic-quiz-data" data-title="Pandas Practice">
[
  {
    "d": "Core Structures",
    "q": "What is pandas, and what is it built on top of?",
    "o": [
      "A machine learning library for training classifiers on tabular data",
      "A SQL database engine embedded in Python",
      "An open-source data manipulation and analysis library, built on top of NumPy, providing the Series and DataFrame data structures",
      "A standalone plotting library unrelated to NumPy"
    ],
    "a": [
      2
    ],
    "e": "It's not a plotting library -- that's Matplotlib/Seaborn's role, though pandas objects can call plotting methods that delegate to them. It isn't a database engine -- pandas holds data in memory as Series/DataFrame objects, not as a queryable on-disk database (though it can read/write to one). It isn't a training library either -- pandas prepares and manipulates data that a library like scikit-learn then trains on, but pandas itself has no model-fitting logic. Pandas is specifically a data manipulation and analysis library built on NumPy arrays under the hood, adding labeled indexing, heterogeneous columns, and tabular operations that raw NumPy doesn't provide."
  },
  {
    "d": "Core Structures",
    "q": "What is the key structural difference between a pandas Series and a pandas DataFrame?",
    "o": [
      "A Series is a one-dimensional labeled array; a DataFrame is a two-dimensional table of rows and columns (conceptually, a collection of aligned Series)",
      "There is no real difference -- they're interchangeable names for the same object",
      "A Series can only hold numbers; a DataFrame can hold any type",
      "A DataFrame is one-dimensional; a Series is two-dimensional"
    ],
    "a": [
      0
    ],
    "e": "A Series can hold any single data type per element (strings, floats, ints, even Python objects) -- the type restriction claim is wrong, and doesn't distinguish it from DataFrame anyway. The dimensionality claim is exactly backwards -- DataFrame is 2D, Series is 1D, not the other way around. They are absolutely not interchangeable -- code that expects rows-and-columns tabular access will break on a Series, and vice versa for column-selection syntax. The real distinction: a Series is a single labeled 1D array (like one column with an index), while a DataFrame is a 2D table -- and conceptually, each column of a DataFrame IS a Series, all sharing the same row index."
  },
  {
    "d": "Core Structures",
    "q": "What does \"reindexing\" a pandas DataFrame actually do?",
    "o": [
      "It conforms the DataFrame to a NEW index, reordering existing data to match and optionally filling in values for any new index labels that didn't exist before",
      "It permanently deletes all rows not in the current index",
      "It resets the index to 0, 1, 2, ... regardless of the data",
      "It renames the columns to sequential integers"
    ],
    "a": [
      0
    ],
    "e": "Reindexing doesn't delete data outright -- rows whose labels aren't in the new index are dropped from the VIEW/result, but this is a controlled realignment, not a blanket deletion operation. Renaming columns to integers describes something else entirely (like resetting column labels), not reindexing, which operates on the index (rows) by default. Unconditionally resetting to 0,1,2,... is what `.reset_index()` does -- a different, simpler operation that ignores any custom index you might want to reindex TO. Reindexing's actual job is conforming a DataFrame or Series to match a given new index -- reordering existing rows to align with it, and introducing NaN (or a specified fill value) for any label in the new index that wasn't present in the original data."
  },
  {
    "d": "Data Types & Grouping",
    "q": "What defines \"categorical\" data in pandas, and what values can a categorical column actually hold?",
    "o": [
      "Categorical data must be numeric and continuous",
      "Data limited to a fixed, repeating set of possible values (like country or gender codes); a categorical column's entries are restricted to its defined categories or NaN",
      "Any column containing text strings is automatically categorical",
      "Categorical columns can hold any arbitrary value, with no restriction"
    ],
    "a": [
      1
    ],
    "e": "Plain text/string columns are NOT automatically categorical in pandas -- they default to the generic 'object' dtype unless you explicitly convert them with `.astype('category')`; treating any string column as categorical would misrepresent, say, a column of unique free-text comments. Categorical data is the opposite of continuous numeric data -- it represents discrete, repeating labels (like 'US', 'UK', 'IN' for country), not measured numeric quantities. Unrestricted arbitrary values describes a normal object column, not a categorical one -- the defining trait of category dtype is exactly that it enforces a fixed set of allowed labels. Correctly: categorical data is data that takes on a limited, fixed number of possible repeating values, and pandas' category dtype enforces that every entry is one of those defined categories (or missing, as NaN)."
  },
  {
    "d": "Data Types & Grouping",
    "q": "What is the difference between `series.copy()` (with `deep=True`, the default) and simply doing `new_series = series` (assignment)?",
    "o": [
      "They behave identically -- both create a fully independent copy",
      "`.copy()` only works on DataFrames, never on Series",
      "Assignment (`new_series = series`) creates a new, independent object automatically; .copy() is redundant",
      "`new_series = series` just creates another reference to the SAME underlying object (changes to one affect the other); `.copy(deep=True)` creates a genuinely independent object that can be modified without affecting the original"
    ],
    "a": [
      3
    ],
    "e": "They do NOT behave identically -- this is precisely the bug that trips people up: plain assignment doesn't copy anything, it just gives two names to the one object. The claim that assignment auto-copies is backwards -- Python variable assignment for objects like Series binds a new name to the SAME object in memory, it never implicitly duplicates data. `.copy()` is a real Series method, not DataFrame-only -- both classes support it since they share this underlying need. The real behavior: `new_series = series` makes `new_series` just another label pointing at the identical underlying object, so mutating one appears to mutate the other; `.copy(deep=True)` explicitly allocates new, independent memory so the copy can be changed freely without side-effects on the original."
  },
  {
    "d": "Data Types & Grouping",
    "q": "What does pandas' `.groupby()` fundamentally do?",
    "o": [
      "It deletes duplicate rows from the DataFrame",
      "It splits the data into groups based on some criterion, so you can then apply an aggregation or transformation to each group and combine the results (split-apply-combine)",
      "It merges two DataFrames together based on a shared key",
      "It sorts the DataFrame by a column, nothing more"
    ],
    "a": [
      1
    ],
    "e": "Sorting alone is `.sort_values()`'s job -- groupby CAN produce sorted-looking output as a side effect, but sorting isn't its purpose. Deleting duplicates is `.drop_duplicates()`, an unrelated operation. Merging two separate DataFrames on a key is `.merge()` or `.join()`, a totally different operation involving two datasets, not one. `.groupby()` implements the classic split-apply-combine pattern: split the DataFrame into groups sharing a common value (e.g. all rows for 'country'=='US'), apply a function like sum/mean/count to each group independently, then combine the per-group results back into one output -- the core mechanism behind virtually all data aggregation in pandas."
  },
  {
    "d": "Core Structures",
    "q": "What is the key difference between `df.loc[...]` and `df.iloc[...]` for selecting rows/columns in a DataFrame?",
    "o": [
      "loc is for DataFrames only; iloc is for Series only",
      "loc only works on columns; iloc only works on rows",
      "loc and iloc are exactly the same; the names are just historical",
      "loc selects by LABEL (the index/column names themselves); iloc selects by INTEGER POSITION (0-based, like a Python list), regardless of what the labels actually are"
    ],
    "a": [
      3
    ],
    "e": "They are not interchangeable -- swapping them silently changes which rows/columns you get whenever the DataFrame's labels aren't simply 0,1,2,... in order (e.g. after filtering or reindexing), which is a very common real bug. Neither is restricted to only-columns or only-rows -- both support selecting along both axes with `[row_selector, column_selector]` syntax. Neither is Series-only or DataFrame-only -- both objects support both accessors. The real distinction: `.loc` indexes by the actual label (e.g. `df.loc['row_name']` or `df.loc[:, 'col_name']`), while `.iloc` indexes purely by integer position (e.g. `df.iloc[0]` always means the first row physically present, no matter what its label says) -- mixing these up after a filter/sort operation is one of the most common pandas bugs."
  },
  {
    "d": "Core Structures",
    "q": "Pandas' DataFrame and Series are described as being built using NumPy arrays internally. What does this dependency actually buy pandas?",
    "o": [
      "It means every pandas operation is automatically GPU-accelerated",
      "It means pandas can only store numeric data, never strings or dates",
      "pandas inherits NumPy's fast, vectorized, contiguous-memory array operations for the actual numeric computation under each column, rather than reimplementing low-level array math itself",
      "Nothing -- it's purely historical and could be removed with no effect"
    ],
    "a": [
      2
    ],
    "e": "It's not purely historical cruft -- removing NumPy as pandas' computational backend would mean reimplementing vectorized array math from scratch, losing the performance pandas is known for. Pandas absolutely does store non-numeric data (strings, datetimes, categoricals) -- these use different underlying storage than a pure NumPy numeric array, but the point stands that pandas' NUMERIC columns lean directly on NumPy arrays for speed. Nothing here is automatically GPU-accelerated -- plain pandas runs on CPU; GPU acceleration requires a separate library (like RAPIDS cuDF, which deliberately mirrors the pandas API precisely so it can be swapped in). The real benefit: for numeric columns, pandas stores the data as NumPy arrays and reuses NumPy's fast vectorized operations, instead of writing its own slower Python-level math."
  }
]
</script>
<div class="topic-quiz-mount"></div>
