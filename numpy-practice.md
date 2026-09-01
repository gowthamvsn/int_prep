# NumPy Practice — Built as a Chain, Not a List

Every snippet here runs on its own. `import numpy as np` is assumed throughout.

Each cluster builds on the one before it — one continuous thread, not a random list. Every cluster ends with one worked example that ties the pieces together.

Run each snippet yourself. Don't just read it. Muscle memory is the actual point.

---

> 🔗 **Hands-on reps:** [Code Drills 4 — NumPy](/topic/code-drills-numpy-pandas#cluster-1-numpy)

## Cluster 1 — Creating and Shaping Arrays

### 1. Making an array of zeros, ones, or a constant
```python
import numpy as np
a = np.zeros((3, 4))          # shape=(3,4): a tuple, not two args — np.zeros(3,4) is a common typo error
b = np.ones((2, 2), dtype=np.int32)   # dtype controls memory (int32=4 bytes) vs default float64 (8 bytes)
c = np.full((2, 3), 7)        # fill_value is positional — the array is entirely 7s
```
`np.zeros` takes one argument: `shape`. That shape must already be a tuple, like `(3, 4)`.

Write `np.zeros(3, 4)` instead, and NumPy reads `4` as a second positional argument — `dtype` — not as part of the shape. That's exactly why the typo errors, or silently does the wrong thing.

`dtype` matters for more than just correctness, too. The default is float64, which uses 8 bytes per number. `float32` or `int32` use half that, 4 bytes. If you don't need the extra precision, you're just doubling your memory use for nothing. At real array or DataFrame scale, that's a real cost, not pedantry.

### 2. Generating evenly spaced numbers instead of a fixed shape
```python
np.arange(0, 10, 2)      # [0,2,4,6,8] — stop is EXCLUSIVE, like Python range
np.linspace(0, 10, 5)    # [0., 2.5, 5., 7.5, 10.] — 5 points INCLUSIVE of both ends
```
`arange`'s step size can drift when you use floats, because of rounding. `linspace`'s `num` argument — how many points you want, not a step size — is exact.

Prefer `linspace` whenever you need a guaranteed endpoint, or an exact count of points.

### 3. Changing an array's shape without touching its data
```python
a = np.arange(12)
b = a.reshape(3, 4)     # view, not copy — modifying b changes a
c = a.reshape(3, -1)    # -1 means "infer this dimension" — here, infers 4
```
`-1` means: figure out this dimension for me. Use it whenever you don't want to hardcode a dimension that depends on the array's size.

That matters when the size varies — for example, the last batch of a dataset is often smaller than the rest.

### 4. Confirming reshape gives you a view, not a copy — and why it matters
```python
a = np.arange(12)
b = a.reshape(3, 4)
print(b.base is a)      # True = view (shares memory); None = copy
```
A view shares memory with the original array. Mutate the view, and you silently mutate the original too. That's a classic source of bugs — you think you copied the data, but you didn't. When you genuinely need independence:
```python
b = a.copy()             # always a real copy, independent memory
```

**Visual + memory hook — a view is a second LABEL on the same box; a copy is a second BOX:**
```
VIEW (b = a.reshape(...))          COPY (b = a.copy())
                                    
  a ──┐                             a ──▶ [ 0 1 2 3 4 5 ... ]   (original memory)
      ├──▶ [ 0 1 2 3 4 5 ... ]      
  b ──┘     (ONE shared box,        b ──▶ [ 0 1 2 3 4 5 ... ]   (independent memory)
             two labels)            
  b[0]=99 → a[0] is ALSO 99         b[0]=99 → a[0] stays 0
```
**Remember it as:** `a` and `b` are just two name-tags stuck on the same box until `.copy()` actually builds a second box — `b.base is a` is literally asking "do these two tags point at the same box."

### Summary example
Say you load a flat sensor log — 12 readings — and reshape it into a 3×4 grid (3 sensors × 4 timestamps).

1. `b = a.reshape(3, 4)`.
2. Check `b.base is a` — it's `True`. So `b` is a view, not a copy.
3. Set `b[0, 0] = 999`. This silently corrupts the original flat log too.

If you needed the two to diverge, you'd have called `a.copy()` first.

---

## Cluster 2 — Selecting and Filtering

### 1. Pulling out just the positive values
```python
a = np.array([1, -2, 3, -4, 5])
mask = a > 0              # array([ True, False,  True, False,  True])
positives = a[mask]       # array([1, 3, 5])
a[a < 0] = 0              # in-place conditional replacement — clip negatives to 0
```
This is called **boolean masking**. It runs as a fast, vectorized C-level loop.

The alternative — a Python `for` loop with an `if` inside it — is 10 to 100 times slower on large arrays.

### 2. Doing the same thing without a separate mask variable
```python
np.where(a > 0, a, 0)     # ternary-style: where condition, take a, else take 0
```
`np.where`'s three arguments — condition, value-if-true, value-if-false — all broadcast together. That means the same call can do more than a simple clip.

`np.where(a > 0, a, -a)`, for example, computes absolute value in one line.

### 3. Getting the indices where a condition is true, not the values
```python
np.argwhere(a > 0).flatten()     # array([0, 2, 4]) — the indices, not the values
np.nonzero(a > 0)                 # tuple of arrays, one per dimension (same info, different shape)
```

### Summary example
Say you have sensor readings `[1, -2, 3, -4, 5]`, and you want to flag the faulty (negative) ones.

1. `mask = a < 0` gives `[False, True, False, True, False]`.
2. `np.argwhere(mask).flatten()` gives `[1, 3]` — those are the sensor indices to physically go inspect.
3. `a[~mask]` gives `[1, 3, 5]` — the clean data, safe to keep aggregating from.

---

## Cluster 3 — Broadcasting

### 1. Combining a smaller array with a bigger one, without a loop
```python
a = np.ones((3, 4))
row = np.array([1, 2, 3, 4])       # shape (4,)
result = a + row                    # row is broadcast across all 3 rows automatically
```
NumPy has one rule for this, and it's worth learning as a small checklist:

1. Line up the two shapes from the RIGHT — like lining up decimal points in two numbers.
2. Compare each pair of aligned dimensions, one pair at a time.
3. They're compatible if they're equal, or if one of them is 1 (or missing entirely) — a `1` stretches to match whatever's on the other side.
4. If any pair fails step 3, broadcasting fails.

Walk through `(3,4) + (4,)`: aligned from the right, that's `4` against `4`. They match. It works.

Now walk through `(3,4) + (3,)`: aligned from the right, that's `4` against `3`. Neither is 1, and they aren't equal. It fails.

### 2. Broadcasting DOWN rows instead of ACROSS columns
```python
col = np.array([1, 2, 3]).reshape(-1, 1)   # shape (3,1), NOT (3,) — the reshape is the whole trick
result = a + col      # now broadcasts down each row
```
A 1-D array shaped `(3,)` broadcasts as a ROW. To broadcast as a COLUMN instead, you need shape `(3,1)` — explicitly.

That one distinction — `(3,)` vs `(3,1)` — causes a large fraction of real broadcasting bugs.

**Visual + memory hook — line up shapes on the RIGHT, like lining up decimal points, and any size-1 (or missing) slot stretches to fit:**
```
  a:     (3, 4)                a:      (3, 4)               a:     (3, 4)
  row:      (4,)   ✓ aligns    col:   (3, 1)   ✓ aligns      bad:      (3,)   ✗
          ↑ matches                    ↑ stretches                    ↑ doesn't match
        4 lines up with 4          the "1" stretches to 4          3 doesn't line up with 4
        row broadcasts ACROSS     col broadcasts DOWN            no reshape = no broadcast
```
**Remember it as:** broadcasting is decimal-point alignment — write both shapes flush-right, and every column either matches exactly, or one side is a `1` that stretches, or it fails. `(3,)` failing against `(3,4)` isn't NumPy being difficult, it's `3` landing under the wrong column (under the `4`, not the `3`) once you align from the right.

### Summary example
Say you have a `(3,4)` grid of sensor readings — 3 units, 4 timestamps — and you need two separate adjustments.

1. Subtract a per-TIMESTAMP calibration offset. Its shape is `(4,)`, so it broadcasts across all 3 rows automatically.
2. Subtract a per-UNIT baseline. Its shape needs to be `(3,1)` — you have to call `.reshape(-1,1)` explicitly, so it broadcasts down instead of across.
3. Do both in one line: `a - calibration_offset - unit_baseline.reshape(-1,1)`.

Each correction stretches along a different axis, which is exactly why the two shapes need to be different.

---

## Cluster 4 — Aggregating Across an Axis

### 1. Summing per-column vs. per-row on a 2D array
```python
a = np.array([[1, 2, 3], [4, 5, 6]])
a.sum(axis=0)   # array([5, 7, 9])  — sum DOWN each column (collapses rows, axis 0)
a.sum(axis=1)   # array([ 6, 15])   — sum ACROSS each row (collapses columns, axis 1)
```
`axis=0` collapses the FIRST dimension — the rows. What's left is one entry per column.

This is the single most common NumPy/pandas confusion. Say it out loud: "axis=0 walks down."

**Visual + memory hook — the axis number is which direction the arrow travels, not which thing survives:**
```
        col0 col1 col2
row0  [   1    2    3  ]     axis=0 ↓↓↓  (walks DOWN through rows)
row1  [   4    5    6  ]              ──▶ result: [5, 7, 9]  (one per COLUMN)

        axis=1 ──▶──▶──▶  (walks ACROSS each row)
                           ──▶ result: [6, 15]  (one per ROW)
```
**Remember it as:** the axis number tells you which direction the arrow WALKS, and the result always has one entry per thing that arrow passed THROUGH multiple of — `axis=0`'s arrow walks down through multiple rows, landing on one number per column.

### 2. Keeping the collapsed dimension around, for a later broadcast
```python
a.sum(axis=1, keepdims=True)   # shape (2,1) instead of (2,) — stays broadcastable against a
```
Without `keepdims`, `a - a.mean(axis=1)` breaks — the shapes don't match. With `keepdims=True`, the result stays 2-D, and it broadcasts cleanly.

That's the exact pattern for "subtract the row mean from every row." Same broadcasting rules from Cluster 3, just fed by an aggregate instead of a hand-built array.

### 3. Getting the INDEX of the max, not the value
```python
a = np.array([3, 7, 1, 9, 4])
a.argmax()          # 3 — the INDEX of the max value (9), not the value itself
```
It's easy to confuse `argmax()` (the index) with `max()` (the value) — the names look similar, and it's a common mistake to expect one but get the other.

### 4. Getting the full sort order, not just one index
```python
a = np.array([30, 10, 20])
np.sort(a)          # array([10, 20, 30]) — sorted values
np.argsort(a)        # array([1, 2, 0])   — indices that WOULD sort a
```
`argsort` lets you sort one array by the order of ANOTHER. For example: `scores[np.argsort(scores)[::-1]]` sorts descending. Or sort a list of names by a parallel score array that doesn't already share an order with them.

### 5. Getting just the top-k, without a full sort
```python
a = np.array([5, 1, 9, 3, 7, 2])
k = 3
idx = np.argpartition(a, -k)[-k:]     # O(n) partition, not O(n log n) full sort
top_k_sorted = idx[np.argsort(a[idx])][::-1]
```
`argpartition` only guarantees one thing: the top-k values end up in the last k positions. It doesn't sort them among themselves.

You then sort just those k elements — cheap, since k is small. Fully sorting a huge array just to read off the top 5 wastes real time at scale.

### Summary example
Say you're grading 6 exam scores, `[5, 1, 9, 3, 7, 2]` (out of 10), and you need the top 3.

1. `np.argpartition(a, -3)[-3:]` cheaply isolates indices `[4, 2, ...]` — the top 3, but unordered.
2. Sort just those 3 elements: `[9, 7, 5]`.

You never had to fully sort all 6 scores. That matters once "6" becomes 6 million.

---

## Cluster 5 — Combining Arrays

### 1. Combining two 1-D arrays into a 2D array, vs. one long array
```python
a = np.array([1, 2, 3]); b = np.array([4, 5, 6])
np.vstack([a, b])     # stacks as new rows -> shape (2,3)
np.hstack([a, b])     # concatenates end-to-end -> shape (6,)
```

### 2. Combining them along a genuinely NEW axis
```python
np.stack([a, b], axis=1)  # NEW axis inserted at position 1 -> shape (3,2), interleaved
```
`vstack`/`hstack` combine along an EXISTING axis. `stack` creates a brand-new axis entirely.

Reach for `stack` when you want each array to stay a distinct "slice" — stacking several grayscale images into one 3-D array, for example, where "which image" needs to be its own dimension, not merged into the pixel data.

**Visual + memory hook — vstack/hstack squeeze into an EXISTING axis, stack builds a NEW shelf:**
```
a = [1,2,3]   b = [4,5,6]

vstack: [[1,2,3],        hstack: [1,2,3,4,5,6]        stack(axis=1): [[1,4],
         [4,5,6]]                (6,) — one long row              [2,5],
         (2,3) — new ROW                                          [3,6]]
                                                                    (3,2) — a NEW
                                                                     axis threading
                                                                     a with b
```
**Remember it as:** vstack/hstack answer "which existing direction do these glue onto" (a new row, or a longer row) — stack answers "I want a brand new dimension whose job is just to say WHICH original array this slice came from," which is exactly what you want for a batch of images where "which image" must stay its own countable axis.

### Summary example
Say you have 3 grayscale image arrays, each shaped `(28,28)`, and you want to combine them into one batch.

1. `np.stack([img1, img2, img3], axis=0)` gives shape `(3,28,28)` — a genuinely new "which image" axis.
2. Compare that to `np.hstack` or `np.vstack`: both would instead try to merge the images along an existing pixel dimension, producing a distorted shape you didn't want.

---

## Cluster 6 — Linear Algebra Operations

### 1. Is `*` the same as real matrix multiplication?
```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
A @ B            # real matrix multiplication (dot product of rows·columns)
A * B            # element-wise: A[i,j]*B[i,j], same shape required
```
No.

That's dangerous specifically because both produce a same-shaped array. Use `*` when you meant `@`, and you don't get an error — you get a structurally valid, silently wrong answer.

Always sanity-check the actual output values, not just the shape, whenever this distinction matters.

### 2. Solving a linear system `Ax = b` for `x`
```python
A = np.array([[3, 1], [1, 2]])
b_vec = np.array([9, 8])
x = np.linalg.solve(A, b_vec)     # exact solve via LU decomposition, NOT np.linalg.inv(A) @ b
```
`np.linalg.inv(A) @ b` is a textbook anti-pattern. Computing the explicit inverse is slower, and numerically less stable, than solving directly.

Use `solve` every time you have a concrete `b`. Only reach for the explicit inverse when you have a genuine symbolic need for it.

### 3. Finding the directions a matrix doesn't rotate — eigenvectors
```python
vals, vecs = np.linalg.eig(A)     # vals: eigenvalues; vecs: columns are the eigenvectors
```
`vecs[:, i]` — a COLUMN — is the eigenvector for `vals[i]`. Indexing a row by mistake is a common bug here, since it's easy to forget eigenvectors are stored column-wise.

(See `math-foundations-refresher.md` for what an eigenvector actually represents geometrically, and how PCA uses this exact computation.)

### Summary example
Say you need to solve `3x + y = 9` and `x + 2y = 8` directly.

1. `np.linalg.solve([[3,1],[1,2]], [9,8])` gives `x=2, y=3`.
2. Check it: `3(2)+3=9` ✓ and `2+2(3)=8` ✓.

NumPy computes this via LU decomposition internally — never via an explicit, slower, less stable matrix inverse.

---

## Cluster 7 — Randomness and Reproducibility

### 1. Generating random numbers that stay reproducible across runs
```python
rng = np.random.default_rng(seed=42)   # modern API — NOT np.random.seed(42) (legacy, global, less safe)
rng.random(5)              # 5 uniform floats in [0, 1)
rng.integers(0, 10, size=5)  # 5 random ints in [0, 10) -- high is EXCLUSIVE by default
rng.normal(loc=0, scale=1, size=5)   # 5 samples from N(0,1)
```
`np.random.seed` mutates NumPy's legacy GLOBAL random state. That state is shared across your entire program — unrelated code elsewhere can silently affect your "reproducible" randomness.

`default_rng` returns an isolated `Generator` instance instead. It's been the modern, recommended API since NumPy 1.17.

### 2. Shuffling a feature matrix `X` and its labels `y` together, without desynchronizing them
```python
X = np.arange(20).reshape(10, 2)    # 10 samples, 2 features -- stand-in for a real feature matrix
y = np.arange(10)                    # 10 labels, aligned with X's rows
rng = np.random.default_rng(0)
idx = rng.permutation(10)     # a shuffled array of 0..9 — apply the SAME idx to X and y to keep them aligned
X_shuffled = X[idx]
y_shuffled = y[idx]
```
Call `rng.permutation` twice — once for `X`, once for `y` — and you get two DIFFERENT random orderings. That silently desynchronizes every row of `X` from its true label.

The only safe pattern: generate one shuffled index array, and apply that same array to both.

### Summary example
Say you're shuffling a 10-sample dataset before a train/test split.

1. `idx = rng.permutation(10)` might give `[3,7,0,9,...]`.
2. Apply that exact same `idx` to both `X[idx]` and `y[idx]`. Sample 3's features stay paired with sample 3's true label.

Call `rng.permutation(10)` a second time for `y` instead, and you'd almost certainly get a different order — silently mislabeling every row.

---

## Cluster 8 — Cleaning and Practical Numeric Operations

### 1. How one missing value (NaN) poisons an aggregate like `.mean()`
```python
a = np.array([1.0, np.nan, 3.0])
a.mean()          # nan — ANY nan poisons a normal aggregate
```
A single `NaN`, anywhere in the array, makes the ENTIRE aggregate `NaN`. It doesn't get skipped. It silently spreads.

### 2. Computing an aggregate that actually ignores NaNs
```python
np.nanmean(a)      # 2.0 — ignores nan values
np.isnan(a)        # array([False,  True, False]) — boolean mask of where NaNs are
```
A `NaN` that silently propagates through a pipeline — into a loss function, say — can make an entire training run produce `NaN` outputs, with no obvious error pointing at why.

Check for NaNs explicitly, with `np.isnan`. Don't assume an aggregate will handle them gracefully on its own.

### 3. Keeping numbers inside a safe range — before a `log()` call, say
```python
a = np.array([-5, 0, 5, 10, 15])
np.clip(a, 0, 10)    # array([ 0,  0,  5, 10, 10]) — a_min, a_max, both inclusive
```
This is exactly how gradient clipping works in practice: `np.clip(grad, -1, 1)`.

It's also how safe-log operations work — clip a probability away from exactly 0 before calling `log()`, so you avoid `log(0) = -inf`.

Same clipping idea as the log-curve section in `math-foundations-refresher.md`, just applied defensively, ahead of a numerically dangerous input.

### 4. Smoothing a noisy 1-D signal without pandas — a rolling average
```python
a = np.array([1, 2, 3, 4, 5, 6])
window = 3
kernel = np.ones(window) / window
moving_avg = np.convolve(a, kernel, mode="valid")   # array([2., 3., 4., 5.])
```
`mode="full"` pads with zeros at the edges. That distorts the average near the boundaries.

`mode="valid"` only returns positions where the window fully overlaps real data. That's the right choice whenever the moving average needs to be trustworthy — not just the right length.

### Summary example
Say you're cleaning a sensor stream `[1.0, NaN, 3.0, 108.0, 4.0]` — one missing reading, one obvious outlier spike.

1. `np.isnan(a)` flags index 1 as the missing value.
2. `np.clip(a, 0, 10)` caps the 108.0 spike down to 10 (assuming 0–10 is the sensor's valid range).
3. `np.nanmean(clipped)` computes a trustworthy average, immune to both problems at once.

That order matters: clip the impossible values first, then average around whatever's still missing.

---

## Practice Q&A (Self-Test)

**Q1. Given `a = np.array([[1, 2, 3], [4, 5, 6]])`, what does `a.sum(axis=0)` return, and why not `[6, 15]`?**
A: `array([5, 7, 9])`. `axis=0` collapses the first dimension (rows), producing one sum per column — "axis=0 walks down." `[6, 15]` is what `axis=1` gives, since that collapses columns and sums across each row instead.

**Q2. Why does `np.zeros(3,4)` fail while `np.zeros((3,4))` works?**
A: `np.zeros` takes a single `shape` argument, which must be a tuple like `(3,4)`. Calling `np.zeros(3,4)` passes `4` as a second positional argument (`dtype`) instead of part of the shape, so it either errors or does the wrong thing — a common typo.

**Q3. What's the difference between `a.reshape(3,4)` and `a.copy().reshape(3,4)` in terms of mutation risk?**
A: `a.reshape(3,4)` returns a view when possible — `b.base is a` is `True`, so modifying `b` also modifies `a`. `a.copy()` first creates independent memory, so any later reshape or mutation on the copy leaves the original array untouched.

**Q4. Why does `np.ones((3,4)) + np.array([1,2,3])` raise a broadcasting error, but adding `np.array([1,2,3,4])` works?**
A: NumPy aligns shapes from the right; `(3,4)` and `(4,)` align because the trailing dimension `4` matches `4`, so the row vector broadcasts across all 3 rows. `(3,4)` and `(3,)` don't align on the right (4 vs 3), so it fails unless the `(3,)` array is reshaped to `(3,1)` to broadcast down columns instead.

**Q5. What does `a.argmax()` return for `a = np.array([3, 7, 1, 9, 4])`, and how is it different from `a.max()`?**
A: `a.argmax()` returns `3`, the INDEX of the largest value. `a.max()` would return `9`, the value itself. Confusing the two — expecting an index but getting a value or vice versa — is a common mistake.

**Q6. Why use `np.argpartition` instead of `np.sort` to get the top-k largest values from a large array?**
A: `np.argpartition` only guarantees the top-k end up in the last k positions (unordered among themselves) in O(n) time, versus a full sort's O(n log n). You then only need to sort those k elements, which is cheap — fully sorting a huge array just to read off the top 5 wastes time at scale.

**Q7. What's the practical difference between `A @ B` and `A * B` for two 2x2 arrays, and why is mixing them up dangerous?**
A: `A @ B` computes real matrix multiplication (row·column dot products); `A * B` computes element-wise multiplication, `A[i,j]*B[i,j]`. Both produce a same-shaped array, so using `*` when you meant `@` doesn't raise an error — it silently produces a wrong answer that looks structurally valid.

**Q8. Why use `np.random.default_rng(seed=42)` over `np.random.seed(42)`?**
A: `np.random.seed` mutates NumPy's legacy global random state, which is shared across the entire program — unrelated code elsewhere can silently affect your "reproducible" randomness. `default_rng` returns an isolated `Generator` instance and is the modern recommended API since NumPy 1.17.

**Q9. If you need to shuffle a feature matrix `X` and labels `y` together, why is calling `rng.permutation(10)` twice (once for each) wrong?**
A: Two separate calls to `rng.permutation` produce two different random orderings, which desynchronizes each row of `X` from its corresponding label in `y`. The correct approach is to generate one shuffled index array and apply that same array to index both `X` and `y`.

**Q10. What does `np.nanmean(np.array([1.0, np.nan, 3.0]))` return, and why is that different from calling `.mean()` on the same array?**
A: `np.nanmean` returns `2.0` because it ignores NaN values when averaging. Plain `.mean()` on the same array returns `nan`, since any NaN "poisons" a normal aggregate — this is why explicitly checking for NaNs (e.g., with `np.isnan`) rather than assuming aggregates handle them gracefully matters.

---

## Video-Sourced Practice MCQs

This is a second practice set, built the same way as this hub's NCA-GENL community bank. The topics are checked against a real YouTube interview-prep video for this subject. The questions themselves are written up fresh here as original multiple-choice questions.

The source video mostly asked these as open-ended questions. The wrong-answer options and their explanations below are original — written to match this hub's "explain every option" convention, not copied from the video.

Click an answer, check it, and use "ask about this question" for anything that needs more explanation.

<script type="application/json" class="topic-quiz-data" data-title="NumPy Practice">
[
  {
    "d": "Arrays & Dtypes",
    "q": "What is NumPy, fundamentally?",
    "o": [
      "A plotting library for visualizing arrays",
      "A machine learning framework for training neural networks",
      "An open-source library providing fast, homogeneous n-dimensional array objects and vectorized operations for Python",
      "A database system for storing numerical tables"
    ],
    "a": [
      2
    ],
    "e": "A plotting library describes Matplotlib, not NumPy -- NumPy has no visualization tools of its own. A database system is wrong: NumPy arrays live in memory for computation, not on-disk storage/querying. A machine learning framework describes something like PyTorch or scikit-learn, which are built ON TOP OF NumPy, not NumPy itself. NumPy's actual identity is the ndarray -- a fixed-type, contiguous-memory array -- plus the vectorized math operations that run on it, which is exactly why libraries like pandas, scikit-learn, and PyTorch all use it as their numerical foundation."
  },
  {
    "d": "Arrays & Dtypes",
    "q": "Why is a NumPy array operation like element-wise addition typically much faster than the equivalent loop over a Python list?",
    "o": [
      "NumPy arrays store one fixed data type in contiguous memory, so the operation runs in a single compiled C loop instead of per-element Python type-checking",
      "NumPy pre-loads the entire result into a lookup table so no computation happens",
      "NumPy arrays are always stored on the GPU, list are not",
      "Python lists are immutable, which slows down every read"
    ],
    "a": [
      0
    ],
    "e": "NumPy arrays are not automatically on the GPU -- that requires a separate library (like CuPy or a GPU-aware framework); plain NumPy runs on CPU. There's no precomputed lookup table -- the math genuinely executes, just efficiently. Python lists are mutable, not immutable, so that's not the mechanism either. The real reason: a Python list can hold mixed types, so Python must check each element's type and dispatch the right operation one at a time (interpreter overhead per element). A NumPy array enforces one dtype for the whole array, so the addition can be compiled once and run as a tight C loop over contiguous memory -- no per-element bookkeeping."
  },
  {
    "d": "Arrays & Dtypes",
    "q": "For a NumPy array `arr` created from `[[1,2,3],[4,5,6]]`, what does `arr.shape` return, and what does that tell you?",
    "o": [
      "'int64' -- the data type of the elements",
      "(2, 3) -- 2 rows and 3 columns",
      "(3, 2) -- 3 rows and 2 columns",
      "(6,) -- the total number of elements, flattened"
    ],
    "a": [
      1
    ],
    "e": "(6,) would be the shape AFTER flattening with .flatten() or .ravel() -- this array hasn't been flattened, so that's not what .shape reports. (3, 2) reverses rows and columns -- a common mix-up, but .shape always lists axis-0 size first (rows), then axis-1 (columns). A dtype string is what `.dtype` returns, a completely different attribute answering a different question ('what kind of values' vs. 'what shape'). `.shape` returns a tuple of each axis's length in order, so a 2-row, 3-column array correctly reports (2, 3) -- the same pattern extends to 3D+ arrays as (depth, rows, cols)."
  },
  {
    "d": "Arrays & Dtypes",
    "q": "What does `arr.dtype` tell you about a NumPy array, and why does it matter?",
    "o": [
      "The single data type (e.g. int64, float64) that every element in the array shares",
      "The number of elements in the array",
      "The array's shape (dimensions)",
      "Whether the array is sorted"
    ],
    "a": [
      0
    ],
    "e": "Shape is `.shape`'s job, not `.dtype`'s -- these are two separate, commonly-confused attributes. Element count is `.size`, again a different attribute entirely. Sortedness isn't tracked by any single attribute -- you'd have to check the values directly (e.g. compare against `np.sort(arr)`). `.dtype` reports the single, uniform data type NumPy is storing every element as (e.g. an array of whole numbers shows int64, one with decimals shows float64) -- this uniformity is exactly what makes the fast, vectorized C-loop execution from the previous question possible in the first place."
  },
  {
    "d": "Indexing & Search",
    "q": "What does `np.bincount(arr)` require of its input, and what does it return?",
    "o": [
      "arr must be sorted first, or the function raises an error",
      "arr must contain non-negative integers; it returns how many times each integer value occurs",
      "arr can contain any real numbers, including negatives; it returns the sum of each unique value",
      "arr can be any shape; it returns the array's dtype"
    ],
    "a": [
      1
    ],
    "e": "Negative numbers are explicitly NOT allowed -- bincount can't build a count array for a negative index, since array indices start at 0. bincount doesn't require pre-sorting -- it works by using each value AS an index into a count array, regardless of input order. It has nothing to do with reporting dtype -- that's `.dtype`'s job, an unrelated attribute. `np.bincount` specifically requires non-negative integers (or booleans) and returns an array where each position i holds the count of how many times the integer i appeared in the input -- a fast way to tally frequencies of small non-negative integers."
  },
  {
    "d": "Indexing & Search",
    "q": "You want to check whether a NumPy array is genuinely empty (holds zero actual values). Why is `arr.size == 0` the reliable check, while `len(arr) == 0` can mislead you?",
    "o": [
      "size only works on 1D arrays, so len() is the only option for 2D+",
      "len() and .size always return the same value, so it doesn't matter which you use",
      "len() checks the dtype, not the element count",
      "len() only looks at the length of the FIRST axis, so an array with shape (1, 0) has len()==1 (misleadingly 'non-empty') even though it holds zero total elements; .size correctly reports the total element count across all axes"
    ],
    "a": [
      3
    ],
    "e": "They do NOT always agree -- that's the entire point of the question, and the case where they diverge is exactly the trap this question is testing. `.size` works fine on any number of dimensions -- it's actually MORE general than the len()-only claim, not less. `len()` doesn't touch dtype at all -- it counts entries along one axis, unrelated to data type. The real mechanism: an array shaped (1, 0) has one row but zero columns, so `len(arr)` (which only measures axis 0) reports 1, while `arr.size` (which multiplies every axis's length: 1 x 0) correctly reports 0 -- making `.size` the trustworthy check for genuine emptiness across any shape."
  },
  {
    "d": "Indexing & Search",
    "q": "What does `np.nonzero(arr > 3)` return, given a NumPy array `arr`?",
    "o": [
      "A single count of how many elements are greater than 3",
      "The actual values in arr that are greater than 3",
      "The row and column indices (as separate arrays) of every position where the condition arr > 3 is True",
      "A new array with all elements greater than 3 replaced by 1"
    ],
    "a": [
      2
    ],
    "e": "A count is what `(arr > 3).sum()` would give you -- np.nonzero doesn't collapse to a single number. The actual values are what `arr[arr > 3]` (boolean indexing) returns -- a different, commonly-paired operation, but not what nonzero itself does. Replacing values with 1 would need an explicit assignment like `arr[arr > 3] = 1`, not a query function. `arr > 3` first produces a same-shaped boolean array (True where the condition holds); `np.nonzero` then returns the coordinates -- one array of row indices and one array of column indices -- of every True (i.e. nonzero, since True==1) position, which is exactly how you locate WHERE a condition holds, not just what the values are."
  },
  {
    "d": "Indexing & Search",
    "q": "You need to delete column index 1 from a 2D NumPy array and later insert a new column back at that same position. Which two functions do that?",
    "o": [
      "np.delete(arr, 1, axis=1) and np.insert(arr, 1, new_col, axis=1)",
      "np.pop(arr, 1) and np.push(arr, new_col, 1)",
      "np.drop and np.add, both with axis=0",
      "np.remove and np.append"
    ],
    "a": [
      0
    ],
    "e": "np.remove and np.pop/np.push don't exist as NumPy functions at all -- these are Python-list-method names that don't carry over to NumPy's array API. np.drop and np.add aren't the right pairing either -- np.add is element-wise addition, unrelated to inserting a whole column. The real pair is `np.delete(arr, index, axis)` to remove a slice along a given axis, and `np.insert(arr, index, values, axis)` to add one back at a specific position -- and critically, `axis=1` means \"operate on columns,\" while `axis=0` would mean rows, so getting the axis right is what actually determines whether you're deleting/inserting a column or a row."
  },
  {
    "d": "Indexing & Search",
    "q": "When creating a 3D NumPy array by nesting Python lists (e.g. `np.array([[[1,2],[3,4]], [[5,6],[7,8]]])`), what does each level of nesting represent?",
    "o": [
      "NumPy automatically flattens all nesting into 1D regardless of structure",
      "The innermost list represents the depth axis, and the outermost represents individual elements",
      "Only the outermost list matters; inner lists are ignored",
      "The outermost list is the depth (number of 2D layers/matrices); each inner list-of-lists is one 2D matrix, with the innermost lists as that matrix's rows"
    ],
    "a": [
      3
    ],
    "e": "Inner lists are absolutely not ignored -- they define the actual row/column structure of each layer; dropping them would lose all the real data. NumPy does NOT auto-flatten nested lists into 1D -- `np.array()` preserves the nesting structure as real dimensions, which is the whole reason nested lists are the standard way to build multi-dimensional arrays. The innermost-vs-outermost roles are reversed in that option -- depth is the OUTER grouping, not the inner one. Correctly: the outermost list's length becomes the first axis (how many 2D layers/matrices exist), and within each layer, the list-of-lists structure defines that layer's own rows and columns -- so a 3D array's shape reads as (depth, rows, cols)."
  }
]
</script>
<div class="topic-quiz-mount"></div>
