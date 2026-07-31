# LeetCode Stats/Probability & "Implement It From Scratch" — 25 DS-Specific Problems

The category generic LeetCode lists skip entirely, but DS interviews love: randomized-algorithm design problems (LeetCode does have real ones — Shuffle an Array, Random Pick with Weight, etc.) plus the "implement `train_test_split`/k-means/linear regression from scratch, no sklearn" style questions that show up specifically because you're interviewing for a DS role, not a general SWE one. Ties together `practice-stats.md`, `math-foundations-refresher.md`, and `practice-numpy.md`.

## Randomized Algorithm Design (real LeetCode problems)

**1. Shuffle an Array** — uniformly random permutation, in place.
```python
import random
def shuffle(nums):
    arr = nums[:]
    for i in range(len(arr) - 1, 0, -1):
        j = random.randint(0, i)
        arr[i], arr[j] = arr[j], arr[i]
    return arr
```
*Technique: Fisher-Yates — swap each element with a uniformly random earlier-or-equal one, walking backward. Shuffling forward with `random.randint(0, len-1)` for every index is a common bug (Sattolo-adjacent) that produces a NON-uniform distribution over permutations.*

**2. Random Pick with Weight** — pick index i with probability proportional to `weights[i]`.
```python
import random, bisect
class WeightedPicker:
    def __init__(self, weights):
        self.prefix = []
        total = 0
        for w in weights:
            total += w
            self.prefix.append(total)
        self.total = total
    def pick_index(self):
        target = random.uniform(0, self.total)
        return bisect.bisect_left(self.prefix, target)
```
*Technique: build a cumulative-sum (prefix) array once, then a single random draw + binary search picks the right "bucket" — O(log n) per pick after O(n) setup, instead of re-scanning weights every time.*

**3. Random Pick Index** — uniformly pick a random index among all occurrences of `target` in an array, without extra memory for storing all matches.
```python
import random
def pick(nums, target):
    result, count = None, 0
    for i, n in enumerate(nums):
        if n == target:
            count += 1
            if random.randint(1, count) == count:
                result = i
    return result
```
*Technique: reservoir sampling of size 1 — replace the current answer with probability `1/count` each time a new match is seen, which (non-obviously, but provably by induction) gives every match exactly equal `1/total_matches` probability without ever knowing the total count in advance.*

**Visual + memory hook — the replacement probability shrinks exactly enough to keep every past match fair:**
```
match #1 seen:  keep it with P = 1/1 = 100%           [always the answer, nothing to compare to yet]
match #2 seen:  replace with P = 1/2 = 50%            [match #1 survives with 50% chance too]
match #3 seen:  replace with P = 1/3 = 33%            [match #1 now survives 1/2 × 2/3 = 33% — still fair]
match #4 seen:  replace with P = 1/4 = 25%            [every match so far still sits at exactly 1/4]
```
**Remember it as a chair that gets less likely to change hands as the line gets longer** — the FIRST match always starts out holding the answer, but every later match gets a fair, shrinking shot at taking it (`1/count` at the moment it arrives), and the chain-multiplication of survival probabilities works out so that by the time you've seen all `n` matches, every single one of them — including the very first — has exactly a `1/n` chance of being the final answer. That's the whole trick: you never need to know `n` in advance, because each step's replacement probability is only ever computed from the count *so far*.

**4. Linked List Random Node** — uniformly random node value from a linked list of unknown length, O(1) space.
```python
import random
def get_random(head):
    result, node, count = None, head, 0
    while node:
        count += 1
        if random.randint(1, count) == count:
            result = node.val
        node = node.next
    return result
```
*Technique: the exact same reservoir-sampling idea as #3, applied to a singly-linked structure where you can't just call `len()` or index randomly.*

**5. Reservoir Sampling, general k** — uniformly sample k items from a stream of unknown total length.
```python
import random
def reservoir_sample(stream, k):
    reservoir = []
    for i, item in enumerate(stream):
        if i < k:
            reservoir.append(item)
        else:
            j = random.randint(0, i)
            if j < k:
                reservoir[j] = item
    return reservoir
```
*Technique: generalizes #3/#4 from k=1 — fill the reservoir with the first k items, then for every later item, replace a uniformly random slot with shrinking probability, keeping every item's final inclusion probability exactly `k/n`.*

**6. Implement `rand10()` using only `rand7()`**
```python
import random
def rand7(): return random.randint(1, 7)

def rand10():
    while True:
        row, col = rand7(), rand7()
        num = (row - 1) * 7 + col     # uniform over 1..49
        if num <= 40:
            return (num - 1) % 10 + 1
```
*Technique: rejection sampling — combine two rand7() calls into a uniform 1..49, throw away the 9 values (41-49) that don't divide evenly into groups of 10, so what survives stays perfectly uniform.*

**7. Insert Delete GetRandom O(1)** — a data structure supporting O(1) insert, remove, and uniformly-random-element access.
```python
import random
class RandomizedSet:
    def __init__(self):
        self.vals = []
        self.idx = {}
    def insert(self, val):
        if val in self.idx: return False
        self.idx[val] = len(self.vals)
        self.vals.append(val)
        return True
    def remove(self, val):
        if val not in self.idx: return False
        i, last = self.idx[val], self.vals[-1]
        self.vals[i] = last
        self.idx[last] = i
        self.vals.pop()
        del self.idx[val]
        return True
    def get_random(self):
        return random.choice(self.vals)
```
*Technique: removal swaps the target with the LAST element before popping — a plain `list.remove(val)` is O(n) because it has to shift everything after it; swap-then-pop keeps removal O(1) at the cost of not preserving order, which this problem doesn't require.*

## "Implement It From Scratch" (no sklearn/scipy)

**8. Implement `train_test_split`**
```python
import numpy as np
def train_test_split_scratch(X, y, test_size=0.2, random_state=None):
    rng = np.random.default_rng(random_state)
    n = len(X)
    idx = rng.permutation(n)
    split = int(n * (1 - test_size))
    train_idx, test_idx = idx[:split], idx[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
```

**9. Implement k-fold cross-validation splitting**
```python
import numpy as np
def k_fold_split(n, k=5, random_state=None):
    rng = np.random.default_rng(random_state)
    idx = rng.permutation(n)
    folds = np.array_split(idx, k)
    for i in range(k):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        yield train_idx, val_idx
```

**10. Implement simple linear regression** (closed-form normal equation, no gradient descent).
```python
import numpy as np
def fit_linear_regression(X, y):
    X_b = np.hstack([np.ones((len(X), 1)), X])       # add bias/intercept column
    theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y     # normal equation
    return theta   # theta[0] = intercept, theta[1:] = coefficients
```
*Technique: the normal equation `θ = (XᵀX)⁻¹Xᵀy` is the closed-form minimizer of squared error — exact and fast for small/medium feature counts, but `XᵀX` becomes expensive and numerically unstable to invert as the feature count grows, which is exactly why gradient descent (#11) is preferred at scale.*

**11. Implement linear regression via gradient descent** (ties directly to the by-hand gradient-descent step in `ds-fundamentals`).
```python
import numpy as np
def fit_linear_regression_gd(X, y, lr=0.01, epochs=1000):
    n, d = X.shape
    w, b = np.zeros(d), 0.0
    for _ in range(epochs):
        y_pred = X @ w + b
        error = y_pred - y
        dw = (2 / n) * (X.T @ error)
        db = (2 / n) * np.sum(error)
        w -= lr * dw
        b -= lr * db
    return w, b
```

**12. Implement k-means clustering**
```python
import numpy as np
def kmeans(X, k, epochs=100, random_state=0):
    rng = np.random.default_rng(random_state)
    centroids = X[rng.choice(len(X), k, replace=False)]
    for _ in range(epochs):
        dists = np.linalg.norm(X[:, None] - centroids[None, :], axis=2)
        labels = dists.argmin(axis=1)
        new_centroids = np.array([X[labels == i].mean(axis=0) if (labels == i).any()
                                   else centroids[i] for i in range(k)])
        if np.allclose(new_centroids, centroids): break
        centroids = new_centroids
    return labels, centroids
```
*Technique: alternate between "assign each point to its nearest centroid" and "recompute each centroid as the mean of its assigned points" until the centroids stop moving — this is literally the Expectation-Maximization pattern at its simplest.*

**13. Implement k-nearest neighbors (classification)**
```python
import numpy as np
from collections import Counter
def knn_predict(X_train, y_train, x_query, k=5):
    dists = np.linalg.norm(X_train - x_query, axis=1)
    nearest_idx = np.argsort(dists)[:k]
    nearest_labels = y_train[nearest_idx]
    return Counter(nearest_labels).most_common(1)[0][0]
```

**14. Implement a decision stump (best single split by Gini impurity)**
```python
import numpy as np
def gini(labels):
    _, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    return 1 - np.sum(p ** 2)

def best_split(X_col, y):
    best_gain, best_thresh = -1, None
    parent_gini = gini(y)
    for thresh in np.unique(X_col):
        left_mask = X_col <= thresh
        if left_mask.all() or (~left_mask).all(): continue
        left_gini, right_gini = gini(y[left_mask]), gini(y[~left_mask])
        n = len(y)
        weighted = (left_mask.sum() / n) * left_gini + ((~left_mask).sum() / n) * right_gini
        gain = parent_gini - weighted
        if gain > best_gain:
            best_gain, best_thresh = gain, thresh
    return best_thresh, best_gain
```
*Technique: try every observed value as a candidate threshold, score each by how much it reduces weighted Gini impurity versus the unsplit parent — this exact loop, run recursively on each resulting side, is what actually builds a decision tree.*

**15. Implement a simple Naive Bayes classifier** (Gaussian features).
```python
import numpy as np
def fit_gaussian_nb(X, y):
    classes = np.unique(y)
    params = {}
    for c in classes:
        X_c = X[y == c]
        params[c] = {"mean": X_c.mean(axis=0), "var": X_c.var(axis=0), "prior": len(X_c) / len(X)}
    return params

def predict_gaussian_nb(params, x):
    def log_gaussian(x, mean, var):
        return -0.5 * np.sum(np.log(2 * np.pi * var) + (x - mean) ** 2 / var)
    scores = {c: np.log(p["prior"]) + log_gaussian(x, p["mean"], p["var"]) for c, p in params.items()}
    return max(scores, key=scores.get)
```
*Technique: works in LOG space throughout (log-prior + log-likelihood, then argmax) instead of multiplying raw probabilities — the same numerical-stability reasoning as `math-foundations-refresher.md`'s log-likelihood section: many small probabilities multiplied together underflow to 0, but their logs sum safely.*

## Statistics & Sampling From Scratch

**16. Compute Pearson correlation from scratch** (no scipy).
```python
import numpy as np
def pearson_corr(x, y):
    x, y = np.array(x), np.array(y)
    x_c, y_c = x - x.mean(), y - y.mean()
    return np.sum(x_c * y_c) / (np.sqrt(np.sum(x_c**2)) * np.sqrt(np.sum(y_c**2)))
```

**17. Compute precision, recall, and F1 from scratch** (no sklearn).
```python
def precision_recall_f1(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return precision, recall, f1
```

**18. Implement bootstrap resampling** — estimate a statistic's sampling distribution.
```python
import numpy as np
def bootstrap_estimate(data, stat_fn=np.mean, n_iterations=1000, random_state=0):
    rng = np.random.default_rng(random_state)
    data = np.array(data)
    estimates = [stat_fn(rng.choice(data, size=len(data), replace=True)) for _ in range(n_iterations)]
    return np.percentile(estimates, [2.5, 97.5]), np.mean(estimates)
```
*Technique: resampling WITH replacement, same size as the original data, repeated many times — the spread of the resulting statistic across iterations approximates its true sampling distribution without needing a closed-form formula for its standard error.*

**19. Implement a two-proportion z-test** (the actual math behind an A/B test significance check).
```python
import math
def two_proportion_z_test(successes_a, n_a, successes_b, n_b):
    p_a, p_b = successes_a / n_a, successes_b / n_b
    p_pool = (successes_a + successes_b) / (n_a + n_b)
    se = math.sqrt(p_pool * (1 - p_pool) * (1/n_a + 1/n_b))
    z = (p_a - p_b) / se
    return z
```
*Technique: the pooled proportion `p_pool` (combining both groups) is used specifically for the standard-error calculation UNDER THE NULL HYPOTHESIS of no difference — using each group's own separate proportion instead would be testing a subtly different, less standard null.*

**20. Implement inverse transform sampling** — generate samples from an arbitrary discrete distribution using only `random()`.
```python
import random, bisect
def sample_from_distribution(values, probs):
    cumulative = []
    total = 0
    for p in probs:
        total += p
        cumulative.append(total)
    r = random.random()
    idx = bisect.bisect_left(cumulative, r)
    return values[idx]
```
*Technique: the same cumulative-sum + binary-search shape as #2 (Random Pick with Weight) — this is a general recipe, not two unrelated tricks: build a CDF, draw uniformly, invert it via binary search.*

**21. Monte Carlo estimate of pi**
```python
import random
def estimate_pi(n_samples=1_000_000):
    inside = 0
    for _ in range(n_samples):
        x, y = random.uniform(-1, 1), random.uniform(-1, 1)
        if x*x + y*y <= 1: inside += 1
    return 4 * inside / n_samples
```
*Technique: the ratio of points landing inside the unit circle to points in the enclosing square converges to the ratio of their areas (π/4) — a simple, honest illustration of what "Monte Carlo estimation" means: approximate an answer via random sampling when the exact calculation is hard or unknown.*

**22. Implement Welford's online algorithm** — running mean and variance in a single pass, numerically stable.
```python
def welford_update(count, mean, m2, new_value):
    count += 1
    delta = new_value - mean
    mean += delta / count
    delta2 = new_value - mean
    m2 += delta * delta2
    variance = m2 / count if count > 1 else 0
    return count, mean, m2, variance
```
*Technique: naively accumulating `sum(x)` and `sum(x**2)` separately and computing variance as `E[x²] - E[x]²` at the end is numerically unstable for large values (subtracting two large nearly-equal numbers loses precision) — Welford's incremental update avoids that subtraction entirely.*

**23. Implement min-max and z-score normalization from scratch**
```python
import numpy as np
def min_max_normalize(x):
    return (x - x.min()) / (x.max() - x.min())

def z_score_normalize(x):
    return (x - x.mean()) / x.std()
```

**24. Implement weighted sampling WITHOUT replacement**
```python
import random
def weighted_sample_without_replacement(items, weights, k):
    items, weights = list(items), list(weights)
    result = []
    for _ in range(k):
        total = sum(weights)
        r = random.uniform(0, total)
        upto = 0
        for i, w in enumerate(weights):
            upto += w
            if upto >= r:
                result.append(items.pop(i))
                weights.pop(i)
                break
    return result
```
*Technique: re-normalizing the remaining weights after every draw (implicitly, by recomputing `total` from what's left) is what makes this "without replacement" — the naive shortcut of just drawing k times independently with replacement can pick the same item twice, which this problem specifically must avoid.*

**25. Compute a histogram (binning) from raw values, no numpy/pandas built-ins**
```python
def compute_histogram(values, num_bins):
    lo, hi = min(values), max(values)
    width = (hi - lo) / num_bins
    counts = [0] * num_bins
    for v in values:
        idx = min(int((v - lo) / width), num_bins - 1)   # clamp the max value into the last bin
        counts[idx] += 1
    return counts, [lo + i * width for i in range(num_bins + 1)]
```
*Technique: the `min(..., num_bins - 1)` clamp matters — without it, the single maximum value in the dataset computes to exactly `num_bins` (one past the last valid index) and either crashes or silently creates a phantom extra bin.*

## Practice Q&A (Self-Test)

### Problems #3, #4, and #5 all use "reservoir sampling." What's the actual guarantee it provides, and why is it needed at all instead of just collecting everything into a list and calling `random.choice`?
Reservoir sampling guarantees every item seen so far has exactly equal probability of being the current answer, updated online, in O(1) space regardless of how many items have streamed by — `random.choice` requires the full list already materialized in memory, which defeats the purpose when the stream is huge or of genuinely unknown length (as in #4's linked list, where you can't call `len()` without a full traversal anyway).

### #10 (normal equation) and #11 (gradient descent) both fit a linear regression. When would you actually prefer the normal equation, and when does it break down?
The normal equation gives an exact closed-form answer in one shot and is preferable for small-to-moderate numbers of features. It breaks down as feature count grows large: computing `(XᵀX)⁻¹` is roughly O(d³) in the number of features and can become numerically unstable (or the matrix can be singular/non-invertible) with highly correlated features — gradient descent scales better to high-dimensional or huge datasets and is the only practical option once you're not just doing plain linear regression (e.g. anything with a non-closed-form loss, like logistic regression or a neural net).

### #15 (Naive Bayes) computes everything in log space. If you skipped that and just multiplied raw Gaussian densities directly, what specifically would go wrong on a real dataset with many features?
Multiplying many probability densities together (one per feature) produces a number that shrinks toward zero extremely fast as feature count grows — with enough features, the product underflows to exactly 0.0 in floating point, making every class's score indistinguishable (all zero) even though the classes are genuinely different. Summing logs instead keeps every intermediate value in a numerically safe range, and `argmax` over log-scores gives the identical answer as `argmax` over the raw products would have, without ever risking underflow.

### #19 (two-proportion z-test) computes a POOLED proportion for the standard error instead of using each group's own separate observed proportion. Why?
The test is specifically evaluating the null hypothesis that both groups share the SAME true underlying proportion — under that null, the best estimate of that shared proportion is the pooled one (combining both groups' successes and totals), so the standard error should be computed consistent with that assumption. Using each group's own separate proportion for the standard error would be implicitly assuming they're different before the test has even run, which is a subtly different (and non-standard) test.
