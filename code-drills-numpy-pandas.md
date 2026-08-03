# Code Drills — Tier 1: NumPy & Pandas

Continues `code-drills-oop-intermediate.md`. Terser, quiz-style companion to the full narrative deep dives in `numpy-practice.md` and `pandas-practice.md` — use this file for rapid-fire practice/self-test, use those two for the "why" behind vectorization, broadcasting rules, and worked chains. All snippets verified against the installed numpy 2.2.6 / pandas 2.2.2.

---

## Cluster 1 — NumPy

> 🔗 **Theory:** [NumPy Practice](/topic/practice-numpy#cluster-1-creating-and-shaping-arrays)

**1. Create an array from a Python list.**
```python
import numpy as np
arr = np.array([1, 2, 3, 4, 5])
arr.shape, arr.dtype    # ((5,), dtype('int64'))
```

**2. Generate evenly spaced values two ways — by step vs. by count.**
```python
np.arange(0, 10, 2)        # array([0, 2, 4, 6, 8])       — step of 2, stop exclusive
np.linspace(0, 1, 5)         # array([0., 0.25, 0.5, 0.75, 1.]) — 5 points, stop INCLUSIVE
```

**3. Create arrays filled with a constant.**
```python
np.zeros((2, 3))     # 2x3 array of 0.0
np.ones((2, 3))      # 2x3 array of 1.0
np.full((2, 3), 7)    # 2x3 array of 7
```

**4. Reshape an array without changing its data.**
```python
arr = np.arange(6)
arr.reshape(2, 3)     # [[0, 1, 2], [3, 4, 5]] — same 6 values, new (2, 3) layout
arr.reshape(2, -1)     # -1 means "infer this dimension" -> also (2, 3) here
```

**5. Index and slice a 2D array.**
```python
m = np.array([[1, 2, 3], [4, 5, 6]])
m[0, 1]      # 2       — row 0, col 1
m[:, 1]       # [2, 5]  — every row, column 1
m[1, :]       # [4, 5, 6] — row 1, every column
```

**6. Select elements with a boolean mask.**
```python
arr = np.array([1, -2, 3, -4, 5])
arr[arr > 0]        # array([1, 3, 5]) — the condition itself produces a boolean array used as a filter
arr[arr > 0] = 0     # can assign through a mask too -> [0, -2, 0, -4, 0]
```

**7. Do elementwise math without a Python loop (vectorization).**
```python
a = np.array([1, 2, 3])
b = np.array([10, 20, 30])
a + b       # [11, 22, 33] — elementwise, no explicit loop
a * 2        # [2, 4, 6]    — scalar broadcasts across every element
```

**8. Understand broadcasting — combining arrays of different shapes.**
```python
m = np.array([[1, 2, 3], [4, 5, 6]])   # shape (2, 3)
row = np.array([10, 20, 30])            # shape (3,)
m + row     # [[11, 22, 33], [14, 25, 36]] — row is "stretched" across both rows of m, no copy made
```

**9. Aggregate along a specific axis.**
```python
m = np.array([[1, 2, 3], [4, 5, 6]])
m.sum()          # 21    — all elements
m.sum(axis=0)     # [5, 7, 9]   — sum DOWN each column (collapses rows)
m.sum(axis=1)     # [6, 15]      — sum ACROSS each row (collapses columns)
```

**10. Multiply matrices (not elementwise).**
```python
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
a @ b            # matrix multiplication: [[19, 22], [43, 50]]
a * b            # elementwise instead: [[5, 12], [21, 32]] — a common mix-up, `@` != `*`
```

**11. Transpose a matrix.**
```python
m = np.array([[1, 2, 3], [4, 5, 6]])
m.T      # shape (3, 2): [[1, 4], [2, 5], [3, 6]]
```

**12. Find the index of the max/min value.**
```python
arr = np.array([3, 7, 1, 9, 4])
arr.argmax()    # 3 — index of the 9
arr.argmin()    # 2 — index of the 1
```

**13. Sort an array, and get the indices that WOULD sort it.**
```python
arr = np.array([3, 1, 4, 1, 5])
np.sort(arr)      # [1, 1, 3, 4, 5]    — sorted values
np.argsort(arr)    # [1, 3, 0, 2, 4]    — indices that would produce that sorted order
```

**14. Combine arrays — stack vs. concatenate.**
```python
a, b = np.array([1, 2]), np.array([3, 4])
np.concatenate([a, b])     # [1, 2, 3, 4]           — joins along existing axis
np.stack([a, b])            # [[1, 2], [3, 4]]        — adds a NEW axis
np.vstack([a, b])           # [[1, 2], [3, 4]]        — stack as rows
np.hstack([a, b])           # [1, 2, 3, 4]             — stack side by side
```

**15. Generate reproducible random numbers.**
```python
rng = np.random.default_rng(seed=42)   # modern API — prefer this over the legacy np.random.seed()
rng.random(3)                            # 3 floats in [0, 1)
rng.integers(0, 10, size=5)               # 5 random ints in [0, 10)
```

**16. Select values conditionally with `np.where`.**
```python
arr = np.array([1, -2, 3, -4, 5])
np.where(arr > 0, arr, 0)    # [1, 0, 3, 0, 5] — vectorized ternary: (condition, if-true, if-false)
```

**17. Get unique values and their counts.**
```python
arr = np.array([1, 2, 2, 3, 3, 3])
np.unique(arr)                          # [1, 2, 3]
np.unique(arr, return_counts=True)       # (array([1, 2, 3]), array([1, 2, 3])) — values, then counts
```

**18. Check/cast an array's dtype.**
```python
arr = np.array([1, 2, 3])
arr.dtype              # dtype('int64')
arr.astype(np.float32)  # same values, cast to float32 — important before feeding into most ML frameworks
```

**19. Compute a vector norm (magnitude).**
```python
v = np.array([3, 4])
np.linalg.norm(v)    # 5.0 — Euclidean (L2) norm: sqrt(3**2 + 4**2)
```

**20. Flatten a multi-dimensional array into 1D.**
```python
m = np.array([[1, 2], [3, 4]])
m.flatten()      # [1, 2, 3, 4] — always returns a copy
m.ravel()         # [1, 2, 3, 4] — same result, but a VIEW when possible (faster, shares memory)
```

---

## Cluster 2 — Pandas

> 🔗 **Theory:** [Pandas Practice](/topic/practice-pandas#cluster-1-creating-and-selecting-data)

**21. Build a DataFrame from a dict of columns.**
```python
import pandas as pd
df = pd.DataFrame({
    "name": ["Sam", "Ana", "Lee"],
    "score": [91, 88, 95],
})
df.shape    # (3, 2)
```

**22. Read a CSV into a DataFrame.**
```python
df = pd.read_csv("people.csv")
```

**23. Get a quick overview of a DataFrame.**
```python
df.head()        # first 5 rows
df.info()         # column dtypes, non-null counts, memory usage
df.describe()     # count/mean/std/min/quartiles/max for numeric columns
```

**24. Select one column vs. multiple columns.**
```python
df["score"]              # Series — a single column
df[["name", "score"]]    # DataFrame — a subset of columns (note the double brackets)
```

**25. Select rows by label (`loc`) vs. by position (`iloc`).**
```python
df.loc[0]           # row with INDEX LABEL 0
df.iloc[0]           # row at POSITION 0 — differ if the index has been reordered/filtered
df.loc[0, "name"]    # a single cell, by label
df.iloc[0:2]         # first two rows by position (slice is exclusive on stop, like Python lists here)
```

**26. Filter rows with a boolean condition.**
```python
df[df["score"] > 90]                       # rows where score > 90
df[(df["score"] > 90) & (df["name"] != "Lee")]   # combine conditions with & / | — NOT `and`/`or`, and each clause needs parens
```

**27. Create a new column derived from existing ones.**
```python
df["passed"] = df["score"] >= 90        # vectorized — no loop, no .apply() needed for simple comparisons
df["score_pct"] = df["score"] / 100
```

**28. Handle missing values — detect, fill, or drop.**
```python
df.isna().sum()                 # count of missing values per column
df["score"].fillna(0)           # replace NaN with 0
df.dropna()                      # drop any row containing at least one NaN
df["score"].fillna(df["score"].mean())   # a common real pattern: impute with the column mean
```

**29. Group rows and aggregate.**
```python
df.groupby("dept")["score"].mean()             # average score per department
df.groupby("dept").agg({"score": ["mean", "max"], "name": "count"})   # multiple aggregations at once
```

**30. Sort a DataFrame by one or more columns.**
```python
df.sort_values("score")                          # ascending
df.sort_values("score", ascending=False)          # descending
df.sort_values(["dept", "score"], ascending=[True, False])   # multi-column: dept A-Z, score high-to-low within each
```

**31. Merge (join) two DataFrames on a shared key.**
```python
scores = pd.DataFrame({"id": [1, 2], "score": [91, 88]})
names = pd.DataFrame({"id": [1, 2], "name": ["Sam", "Ana"]})
pd.merge(scores, names, on="id")           # inner join by default — keeps only matching ids
pd.merge(scores, names, on="id", how="left")   # left/right/outer available too, same as SQL joins
```

**32. Apply a custom function across a column.**
```python
df["grade"] = df["score"].apply(lambda s: "A" if s >= 90 else "B")
# prefer vectorized ops (drill #27) when possible — .apply() is a Python-level loop under the hood, slower
```

**33. Count occurrences of each unique value in a column.**
```python
df["dept"].value_counts()    # e.g. Engineering  2 \n Sales  1 — sorted descending by count
```

**34. Build a pivot table (spreadsheet-style cross-tab).**
```python
df.pivot_table(values="score", index="dept", columns="passed", aggfunc="mean")
# rows = dept, columns = passed (True/False), cells = average score for that combination
```

**35. Rename columns.**
```python
df.rename(columns={"score": "test_score"})    # returns a NEW DataFrame by default (inplace=False)
```

**36. Drop columns or rows.**
```python
df.drop(columns=["score_pct"])       # drop a column
df.drop(index=[0, 1])                 # drop rows by index label
```

**37. Convert a column's dtype.**
```python
df["score"] = df["score"].astype(float)
df["id"] = df["id"].astype(str)        # e.g. turn a numeric ID into a string before merging on it
```

**38. Parse a column of date strings into real datetimes.**
```python
df["date"] = pd.to_datetime(df["date"])       # now supports .dt.year, .dt.month, date arithmetic, etc.
df["month"] = df["date"].dt.month
```

**39. Convert a DataFrame to a list of dicts (or a plain dict) — the bridge back to `json.dumps`.**
```python
df.to_dict(orient="records")    # [{'name': 'Sam', 'score': 91}, {'name': 'Ana', 'score': 88}, ...]
# feed this straight into json.dumps(...) from code-drills-data-structures.md drill #32
```

**40. Save a DataFrame to CSV or JSON.**
```python
df.to_csv("out.csv", index=False)      # index=False: don't write the row-number index as its own column
df.to_json("out.json", orient="records")
```

---

**Next in the Code Drills tier:** `code-drills-classical-ml.md` (sklearn train/eval workflow, including RandomForest).
