# scikit-learn Practice — Built as a Chain, Not a List

Most snippets on this page assume `X, y` already exist.

- `X` is the **feature matrix** — one row per example, one column per measured attribute. Think of it as a spreadsheet of inputs.
- `y` is the **target** — the one column you're trying to predict, one value per row.

Each cluster below is one continuous thread. Every question builds on the answer before it, and each cluster ends with a worked summary example.

**If you're new to ML, five terms carry most of this page's weight — defined here once.** (Deeper, diagrammed treatments live in `ds-fundamentals` and `math-foundations-refresher.md`.)

- **Model / fitting.** A model is a formula with adjustable numbers inside. *Fitting* (training) means letting the algorithm tune those numbers until the formula predicts `y` from `X` as well as it can, on the data it was given.
- **Train/test split.** You hide part of the data — the test set — from the model during fitting. Later, you use it to measure how the model does on data it's never seen. A score on already-seen data flatters the model. The score on held-out data is the honest one.
- **Overfitting.** The model memorized the training data's quirks and noise instead of learning the general pattern. It looks great on seen data and falls apart on new data. (*Underfitting* is the opposite failure — a model too simple to capture the real pattern, mediocre everywhere.)
- **Hyperparameter.** A setting *you* choose before training — tree depth, regularization strength. That's different from the parameters the model learns on its own during fitting. "Tuning" means systematically trying settings and keeping whichever works best.
- **Data leakage.** Any way information from the test set (or from the future) sneaks into training or preprocessing. It inflates your measured score without making the model any better — worse than no score at all, because it's a *convincing wrong* number. Most of Cluster 1 exists to prevent leakage structurally, not just by discipline.

---

## Cluster 1 — Splitting and Preprocessing Without Leaking

### 1. How do you split data into train and test sets correctly?

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

`stratify=y` forces both the train split and the test split to keep the same class proportions as the full dataset. Without it, a random split can produce a test set with a very different class balance than train — especially likely on imbalanced data.

`random_state` fixes the randomness. Without it, every run of this code produces a different split, and your results aren't reproducible — not by you next week, and not by anyone reviewing your work.

### 2. How do you standardize features without leaking test information into it?

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit_transform: learns mean/std FROM train, then applies
X_test_scaled = scaler.transform(X_test)          # transform only: reuses train's mean/std, does NOT refit
```

**Standardizing** rescales each column to mean 0 and standard deviation 1. That puts a column measured in thousands (income) and a column measured in single digits (age) on equal footing. Anything distance-based or gradient-based needs that, or it ends up treating features unfairly.

Here's the leakage risk, step by step:

1. If you call `fit_transform` on the test set, the scaler relearns its mean and standard deviation from test data.
2. That lets information about the test set's distribution leak into preprocessing.
3. Your reported performance ends up inflated — higher than what the model would actually do on truly unseen data.

The rule that avoids all of this: never call `fit_transform` on test data. Only `transform`.

### 3. How do you encode categorical columns, and what happens if test has a category train never saw?

```python
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
X_train_enc = enc.fit_transform(X_train[["depot"]])
X_test_enc = enc.transform(X_test[["depot"]])     # any category not seen in train is handled, not crashed
```

**One-hot encoding** turns a text category into numbers a model can use. `depot = "Dallas"` becomes a set of 0/1 columns, one per depot seen in training, with a 1 in the matching column. Models do math on numbers, not strings, so this step is necessary.

Without `handle_unknown="ignore"`, a category that shows up in test or production but was never seen during training raises an error at inference time. With `"ignore"`, that category gets encoded as all-zeros instead, and the pipeline keeps running. Still worth monitoring for — just not a hard crash.

### 4. Both the scaler and the encoder need the same fit-on-train-only discipline. How do you enforce that structurally?

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000)),
])
pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)
```

A `Pipeline` guarantees that the scaler is fit only on whatever data `.fit()` is called with. Inside cross-validation — splitting the training data into several "folds" and holding each one out in turn to test on, which Cluster 2 covers properly — this means each fold's scaler is refit on just that fold's own training portion. That's the only way to avoid leakage during CV.

Manually scaling the data before CV, then passing the pre-scaled data in, is a very common leakage bug. A `Pipeline` closes that door for you.

**Visual + memory hook — a Pipeline is a sealed pipe: data goes in one end, nothing skips a station:**
```
X_train ──▶ [ StandardScaler ] ──▶ [ LogisticRegression ] ──▶ predictions
              .fit_transform()         .fit()
              learns mean/std          learns weights
              from THIS data           from THIS data
              only                     only

Inside cross-validation, this whole sealed pipe gets rebuilt fresh, per fold:
  Fold 1: [scaler learns fold 1's train stats] → [model learns on fold 1's train]
  Fold 2: [scaler learns fold 2's train stats] → [model learns on fold 2's train]
  (never: one scaler fit on everything, reused across folds — that's the leak)
```

**Remember it as:** a `Pipeline` isn't convenience syntax, it's a seal against "peeking" — every step's `.fit()` only ever sees exactly the rows the outer `.fit()` call was given.

### 5. A Pipeline chains steps in sequence. What if numeric and categorical columns need different transformers applied in parallel?

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

preprocess = ColumnTransformer([
    ("num", StandardScaler(), ["wear_pct", "age_days"]),
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["depot"]),
])
```

Numeric and categorical columns need fundamentally different treatment. Applying `StandardScaler` to a categorical column — or one-hot encoding to a continuous one — is a common beginner error. `ColumnTransformer` prevents it structurally, by naming exactly which columns get which transformer.

This `preprocess` object then slots into a `Pipeline` as its first step, the same way `StandardScaler` did on its own above.

### Summary example

A dataset has both `wear_pct` (numeric) and `depot` (categorical).

1. `ColumnTransformer` routes each column to its correct transformer — `StandardScaler` for `wear_pct`, `OneHotEncoder` for `depot`.
2. That `ColumnTransformer` gets wrapped inside a `Pipeline`, alongside the classifier.
3. The whole thing gets `.fit()`-ed only on `X_train`.

One structural choice, building the Pipeline this way, closes every leakage risk from the questions above — instead of leaving several separate manual disciplines to remember.

---

> 🔗 **Hands-on reps:** [Code Drills 5 — Pipelines, Cross-Validation & Hyperparameter Search](/topic/code-drills-classical-ml#cluster-2-pipelines-cross-validation-hyperparameter-search)

## Cluster 2 — Cross-Validation and Hyperparameter Tuning

### 1. A single train/test split gives one score. How do you get a more reliable estimate?

```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="roc_auc")
print(scores.mean(), scores.std())    # report BOTH — std tells you how stable the estimate is
```

`cv=5` splits the training data into 5 folds. The model trains 5 separate times, each time holding a different fold out as a mini test set. That gives you 5 scores instead of 1.

Two models can have the same mean CV score and still not be equally trustworthy. If one has a much higher standard deviation across folds, its estimate is less stable — and a single mean number hides that completely. Always report both.

(`scoring="roc_auc"` picks AUC as the score here — a classification metric that, unlike accuracy, isn't flattered by imbalanced classes. Cluster 3 unpacks it.)

### 2. With imbalanced classes, does plain k-fold CV risk the same class-balance problem `stratify` fixed in Cluster 1?

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1")
```

Yes — `StratifiedKFold` fixes it the same way `stratify=y` did back in Cluster 1.

`shuffle=True` matters for a separate reason. Without it, `StratifiedKFold` splits the data in its original order. If the data happens to be sorted or grouped in any way — common with real-world exports — the folds can end up systematically different from each other. Shuffling, with a fixed seed for reproducibility, avoids that.

### 3. With a reliable CV estimate in hand, how do you actually search for the best hyperparameters?

```python
from sklearn.model_selection import GridSearchCV
param_grid = {"clf__C": [0.01, 0.1, 1, 10], "clf__penalty": ["l1", "l2"]}
grid = GridSearchCV(pipe, param_grid, cv=5, scoring="roc_auc", n_jobs=-1)
grid.fit(X_train, y_train)
print(grid.best_params_, grid.best_score_)
```

The two hyperparameters being searched here are both **regularization** controls. Regularization is a deliberate penalty on model complexity — it discourages extreme learned coefficients, so the model can't contort itself around training noise.

- `C` is the *inverse* strength of that penalty. Smaller `C` means stronger restraint.
- `penalty` picks the flavor — L1 vs. L2. (Worked out with real numbers in `math-foundations-refresher.md`.)

When tuning a `Pipeline`, parameter names must follow the pattern `<step_name>__<param_name>`. That's how `GridSearchCV` knows which pipeline step each hyperparameter belongs to. A plain `"C"` instead of `"clf__C"` raises an error.

`n_jobs=-1` uses all available CPU cores in parallel. Grid search over many combinations is embarrassingly parallel — leaving this at the default of 1 core can make tuning take many times longer than it needs to.

### 4. Grid search tries every combination. What if there are too many hyperparameters for that to be practical?

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform
param_dist = {"clf__C": loguniform(1e-3, 1e2)}
search = RandomizedSearchCV(pipe, param_dist, n_iter=20, cv=5, scoring="roc_auc", random_state=42, n_jobs=-1)
search.fit(X_train, y_train)
```

Grid search's cost grows multiplicatively with every hyperparameter you add. Randomized search instead samples a fixed budget (`n_iter`), no matter how many hyperparameters or values are in play. Empirically, it finds near-optimal regions almost as well as exhaustive search, for a fraction of the compute.

Why `loguniform` for `C` specifically: regularization strength spans orders of magnitude, from 0.001 to 100. Sampling uniformly on a linear scale wastes most of your samples in one narrow range. Log-uniform sampling spreads them evenly across every order of magnitude instead.

### Summary example

Tuning a logistic regression's `C` and `penalty`.

1. With only 2 hyperparameters and a handful of values each, `GridSearchCV` covers 8 total combinations — cheap enough to run exhaustively.
2. Scale that up to 5 hyperparameters with 10 values each, and you get 100,000 combinations.
3. At that point, `RandomizedSearchCV` becomes the only practical choice: a fixed `n_iter=50` budget, with `loguniform` sampling for anything spanning orders of magnitude, trades a small amount of thoroughness for a massive compute saving.

---

## Cluster 3 — Evaluating Classification Properly

### 1. Beyond a single accuracy number, how do you evaluate a classifier properly?

```python
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
y_pred = pipe.predict(X_test)
y_proba = pipe.predict_proba(X_test)[:, 1]     # probability of the POSITIVE class (column 1), not the prediction
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(roc_auc_score(y_test, y_proba))           # AUC needs probabilities/scores, NOT hard 0/1 predictions
```

What each of those three calls actually tells you:

- `classification_report` breaks out **precision** and **recall** per class. Precision asks: of everything the model flagged as positive, what fraction really was? Recall asks: of everything truly positive, what fraction did the model catch? Those are two different questions, and a single accuracy number blends them together and hides the answer to both.
- `confusion_matrix` is the raw 2×2 count table underneath: true positives, false positives, true negatives, false negatives.
- **ROC-AUC** summarizes how well the model *ranks* positives above negatives, across every possible decision threshold at once. 1.0 is perfect ranking. 0.5 is a coin flip.

One mechanical trap: `predict_proba` returns a column per class. For binary classification, column 0 is P(negative) and column 1 is P(positive). Pass the wrong column to `roc_auc_score`, and it silently computes AUC for the wrong class.

### 2. Before trusting any of those numbers as "good," what's the cheapest sanity check?

```python
from sklearn.dummy import DummyClassifier
baseline = DummyClassifier(strategy="most_frequent")   # always predicts the majority class
baseline.fit(X_train, y_train)
print(baseline.score(X_test, y_test))    # THIS is the number your real model has to beat
```

Is the model even beating doing nothing? On a dataset that's 95% one class, `DummyClassifier` scores 95% accuracy doing nothing intelligent at all. Establish this number first. It's what stops you from being impressed by a good-looking accuracy that a trivial baseline already matches.

### 3. A model beats the baseline. How do you quickly check if it's overfitting?

```python
print("train:", pipe.score(X_train, y_train))
print("test:", pipe.score(X_test, y_test))
```

Compare the two numbers. A train score much higher than the test score is the single fastest overfitting diagnostic you have — check this gap before touching any hyperparameter.

The gap tells you which problem you have, and the two need very different fixes:

- Both scores low, small gap: a **bias** problem. The model is too simple to capture the real pattern (underfitting).
- Train score high, test score much lower: a **variance** problem. The model latched onto training noise (overfitting).

(The full bias-variance story, with diagrams, lives in `ds-fundamentals`.)

**Visual + memory hook — the three possible gap shapes, and what each one means:**
```
UNDERFIT                  GOOD FIT                  OVERFIT
train: 62%                 train: 91%                train: 99%
test:  60%                 test:  89%                test:  71%
  small gap,                 small gap,                 HUGE gap —
  both LOW                   both HIGH                  memorized train,
  → bias problem              → this is what              generalizes badly
  (model too simple)            you want                  → variance problem
                                                            (model too complex,
                                                             or needs more data)
```

### Summary example

A model scores 94% test accuracy. Impressive, until you check `DummyClassifier(strategy="most_frequent")` and it also scores 94%, because the classes are that imbalanced — the "impressive" model added zero real signal.

Checking train (96%) against test (94%) shows the gap is small. So it's not overfitting. The real problem is that accuracy was the wrong metric to be impressed by in the first place — `classification_report`'s precision/recall breakdown per class matters far more here than the single accuracy number.

---

> 🔗 **Hands-on reps:** [Code Drills 5 — Imbalance, Persistence & Preprocessing](/topic/code-drills-classical-ml#cluster-3-imbalance-persistence-preprocessing)

## Cluster 4 — Handling Class Imbalance

### 1. With imbalanced classes, what's the simplest fix, directly in the model itself?

```python
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(class_weight="balanced")    # auto-reweights inversely proportional to class frequency
```

`class_weight="balanced"` changes the loss function's penalty — the loss function is the running "how wrong am I" score that training minimizes — so that misclassifying a rare-class example hurts more than misclassifying a common one. It doesn't fabricate any synthetic data. That makes it the simpler, leakage-free first move, usually worth trying before anything more involved.

### 2. If reweighting the loss isn't enough, how do you generate more minority-class examples?

```python
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
```

SMOTE synthesizes new minority-class points by interpolating between real ones. It draws each new fake example somewhere along the straight line between two genuine minority-class neighbors, so the additions stay plausible instead of random.

The one rule that must never be broken: SMOTE must be fit only on the training split, never before splitting. Apply it before the split, and synthetic points derived from what becomes test data can leak into training — plus the test set's own synthetic points wouldn't represent real unseen data at all. `imblearn`'s own `Pipeline` (not sklearn's) exists specifically to keep SMOTE properly scoped inside cross-validation folds — the same leakage discipline as Cluster 1's `StandardScaler`.

### Summary example

A fraud dataset with a 2% positive class.

1. Try `class_weight="balanced"` first — it's cheap and leakage-free. If it alone gets acceptable recall, SMOTE isn't even needed.
2. If it doesn't, apply SMOTE — but only inside `imblearn.pipeline.Pipeline`, so it stays correctly scoped to just the training fold.
3. Applying SMOTE to the full dataset before splitting would leak synthetic points derived from test-set fraud cases into training, silently inflating the test score in a way that wouldn't hold up in production.

---

## Cluster 5 — Feature Selection and Importance

### 1. With many candidate features, how do you automatically select the most predictive ones?

```python
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(score_func=f_classif, k=10)
X_train_sel = selector.fit_transform(X_train, y_train)
selected_cols = X_train.columns[selector.get_support()]    # get_support() -> boolean mask of kept columns
```

`f_classif` (the ANOVA F-value — the same F-statistic idea from `stats-scipy-practice.md`'s ANOVA cluster) assumes a roughly linear relationship between each feature and the target. `mutual_info_classif` captures non-linear relationships too, but costs more to compute. Pick based on whether you suspect a non-linear feature-target relationship.

### 2. Instead of selecting features before training, how do you find out which features a trained tree-based model actually relied on?

```python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
importances = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
```

`max_depth` matters here specifically. Unconstrained trees (`max_depth=None`) grow until every leaf is pure. That tends to overfit, and it also inflates the apparent importance of high-cardinality features — columns with many distinct values, like a ZIP code or customer ID — that can split data very finely just by chance. Capping depth is one of the simplest regularization levers for tree ensembles, the same bias-variance dial as `ml-models-practice.md`'s dartboard visual.

### Summary example

30 candidate features, most of them noise.

1. `SelectKBest(f_classif, k=10)` narrows to the 10 most linearly predictive features, before training anything.
2. Training a `RandomForestClassifier` on all 30 features and reading `feature_importances_` instead shows which features the model actually leaned on, after training.
3. The two approaches can disagree on a feature with a real but non-linear relationship to the target — `f_classif` underrates it, while a tree-based importance correctly surfaces it.

---

## Cluster 6 — Evaluating Regression

### 1. Classification has accuracy, precision, and recall. What's the equivalent for a regression model?

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
y_pred = reg.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)          # same units as target, robust to outliers
rmse = mean_squared_error(y_test, y_pred, squared=False)   # penalizes large errors more than MAE
r2 = r2_score(y_test, y_pred)                       # fraction of variance explained, 1.0=perfect
```

### 2. Both MAE and RMSE measure error. Why report both instead of just picking one?

Because RMSE squares errors before averaging, and MAE doesn't. If RMSE comes out much larger than MAE, that gap itself is diagnostic — it means a small number of predictions have large errors dragging RMSE up. MAE alone would mask that entirely.

### Summary example

A model's MAE is $500, but its RMSE is $3,200. That large gap immediately signals a handful of predictions with huge errors — RMSE's squaring punishes them disproportionately — rather than uniformly mediocre predictions across the board. It points an investigation toward specific outlier cases, not the model's general calibration.

---

> 🔗 **Hands-on reps:** [Code Drills 5 — Unsupervised Learning & Diagnosing Over/Underfitting](/topic/code-drills-classical-ml#cluster-4-unsupervised-learning-diagnosing-overunderfitting)

## Cluster 7 — Unsupervised: Clustering and Dimensionality Reduction

### 1. With no labels at all, how do you group similar data points together?

```python
from sklearn.cluster import KMeans
inertias = []
for k in range(1, 10):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    inertias.append(km.inertia_)     # sum of squared distances to nearest centroid
# plot inertias vs k, look for the "elbow" where adding clusters stops helping much
```

How K-means actually works, one step at a time:

1. Place `k` **centroids** — imaginary center points — at random.
2. Assign every row to its nearest centroid.
3. Move each centroid to the middle of the rows now assigned to it.
4. Repeat steps 2 and 3 until nothing moves. The final groups are your clusters.

Because the starting placement is random, the result can get stuck in a bad **local optimum** — a clustering that no small adjustment improves, even though a genuinely better overall clustering exists. `n_init=10` runs the whole process 10 times from different random starts and keeps the best (lowest-inertia) run. Set it too low, and you risk a genuinely worse clustering purely from bad luck.

### 2. With too many features to even visualize, how do you reduce dimensionality first?

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X_train_scaled)   # ALWAYS scale first -- PCA is sensitive to feature scale
print(pca.explained_variance_ratio_)        # fraction of total variance each component captures
```

**Dimensionality reduction** compresses many columns into a few new synthetic ones that preserve as much of the data's spread as possible — so 20 features can become 2 you can actually plot.

PCA is the standard linear way to do it. It finds directions of maximum variance — the eigenvector machinery from `math-foundations-refresher.md`. But scale first: a feature measured in larger raw units (income in dollars vs. age in years) will dominate the principal components purely from its scale, not because it's actually more informative, unless everything is standardized first. Same "always scale first" discipline as Cluster 1's `StandardScaler`, just feeding PCA this time instead of a classifier.

### Summary example

Clustering customers on 20 raw features, including both "annual income" (tens of thousands) and "age" (tens).

1. Without scaling first, K-means would effectively cluster almost entirely on income, since its raw numeric range dwarfs age's.
2. Running `StandardScaler` before `KMeans` (or before `PCA`, for a 2D visualization of the same clusters) puts every feature on equal footing first.
3. The resulting clusters then reflect genuine multi-feature similarity, not just one feature's arbitrary unit of measurement.

---

## Cluster 8 — Persisting a Trained Model

### 1. After all the above, how do you save a fitted pipeline so it doesn't need retraining every time?

```python
import joblib
joblib.dump(pipe, "model.joblib")
loaded_pipe = joblib.load("model.joblib")
```

This is **serialization** — saving a live Python object to a file, byte for byte, so a different process, later, can reload it exactly as it was, fitted state included. `joblib` over plain `pickle`: it's more efficient specifically for objects containing large NumPy arrays, like a fitted model's learned weights — which is exactly what most scikit-learn estimators are. That's why it's the library's own recommended serialization tool.

### Summary example

A tuned `Pipeline` (scaler and classifier, from Cluster 1) that took 20 minutes of `GridSearchCV` to produce.

1. Save it once, with `joblib.dump`.
2. Load it in a separate serving script, with `joblib.load`.
3. The entire fitted pipeline, preprocessing included, comes back in milliseconds — instead of re-running the full training and tuning process.

---

## Practice Q&A (Self-Test)

**Q1. In `train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)`, what does each of `stratify` and `random_state` guard against?**
A: `stratify=y` forces both the train and test splits to keep the same class proportions as the full dataset. That prevents a random split from producing a test set with a very different class balance — especially risky on imbalanced data. `random_state` fixes the randomness, so the split is reproducible across runs and reviewers.

**Q2. Why is `scaler.fit_transform(X_test)` a data leakage bug, and what should you call instead?**
A: `fit_transform` would relearn the mean and standard deviation from the test set itself. That lets information about the test distribution leak into preprocessing, and it inflates reported performance above what the model would actually see on truly unseen data. The correct call is `scaler.transform(X_test)` — it reuses the mean/std already learned from `X_train`, without refitting.

**Q3. What does `OneHotEncoder(handle_unknown="ignore")` do when `X_test` contains a category never seen in `X_train`, and what's the alternative behavior without it?**
A: With `handle_unknown="ignore"`, an unseen category gets encoded as all-zeros and the pipeline keeps running — worth monitoring, but not a crash. Without it, encountering an unseen category at inference or test time raises an error and halts the pipeline.

**Q4. Why does wrapping `StandardScaler` and `LogisticRegression` in a `Pipeline` matter beyond convenience, specifically during cross-validation?**
A: A `Pipeline` guarantees the scaler is fit only on whatever data `.fit()` is called with. Inside cross-validation, that means each fold's scaler is refit on just that fold's own training portion. Manually scaling the full dataset before running CV, then passing the pre-scaled data in, is a common leakage bug this structurally prevents.

**Q5. What problem does `ColumnTransformer` solve when you have both numeric and categorical columns?**
A: It lets you apply different transformers to different named columns in one object — `StandardScaler` to `["wear_pct", "age_days"]` and `OneHotEncoder` to `["depot"]`. This structurally prevents a common beginner error: applying scaling to categorical columns, or one-hot encoding to continuous ones.

**Q6. Why does the file recommend printing both `scores.mean()` and `scores.std()` after `cross_val_score`?**
A: Two models can have the same mean CV score but very different standard deviations across folds. High variance means the estimate itself is less trustworthy — something the mean alone hides. Reporting both gives a fuller picture of reliability, not just central tendency.

**Q7. In `GridSearchCV` with `param_grid = {"clf__C": [...], "clf__penalty": [...]}`, why is the `clf__` prefix required instead of just `"C"`?**
A: When tuning a `Pipeline`, parameter names must follow `<step_name>__<param_name>`, so `GridSearchCV` knows which pipeline step — here, the `"clf"` step — each hyperparameter belongs to. A plain `"C"` instead of `"clf__C"` raises an error, because it doesn't map to any pipeline step's parameter.

**Q8. Why does the file use `loguniform(1e-3, 1e2)` for sampling `C` in `RandomizedSearchCV` instead of a plain uniform range?**
A: Regularization strength like `C` naturally spans orders of magnitude, from 0.001 to 100. Sampling uniformly on a linear scale would waste most samples in one narrow range. Log-uniform sampling spreads samples evenly across every order of magnitude, matching how `C`'s effect actually varies.

**Q9. When computing `roc_auc_score(y_test, y_proba)`, why is `y_proba = pipe.predict_proba(X_test)[:, 1]` specifically indexed with `[:, 1]` rather than `[:, 0]`?**
A: `predict_proba` returns one probability column per class — for binary classification, column 0 is P(negative) and column 1 is P(positive). `roc_auc_score` needs probabilities for the positive class, so passing the wrong column silently computes AUC for the wrong class.

**Q10. Why does the file suggest `class_weight="balanced"` as a first move for imbalanced classification, before trying SMOTE?**
A: `class_weight="balanced"` reweights the loss function's penalty for misclassifying the minority class, instead of fabricating synthetic data points, making it a simpler, leakage-free first option. SMOTE, by contrast, must be fit only on the training split — never before the train/test split — to avoid synthetic points leaking into or contaminating the test set.

---

## Video-Sourced Practice MCQs (Set 2)

A second scikit-learn practice set, built the same way as this hub's NCA-GENL community bank. Topics were checked against a real YouTube scikit-learn-interview-prep video, then written up here as fully original multiple-choice questions — every option and explanation below is original, not copied from the video.

These questions focus on angles the clusters above don't already drill in depth: clustering algorithm choice (KMeans vs. DBSCAN) and evaluation, core hyperparameter meanings, how regularization actually works, the fit/transform API discipline, KNN's prediction mechanics, why CV beats training accuracy for model selection, and what SVM's margin maximization is actually doing.

<script type="application/json" class="topic-quiz-data" data-title="scikit-learn Practice (Set 2)">
[
  {
    "d": "Clustering Algorithm Choice",
    "q": "KMeans and DBSCAN both do unsupervised clustering, but need fundamentally different inputs from you. What's the key practical difference in how you configure each?",
    "o": [
      "KMeans requires you to specify the number of clusters `k` upfront; DBSCAN instead takes a neighborhood-distance/density parameter and DISCOVERS the number of clusters itself, while also flagging outliers as noise",
      "DBSCAN requires you to specify `k`; KMeans instead auto-detects the number of clusters",
      "Both require exactly the same inputs — `k`, the number of clusters — with no other configuration difference",
      "Neither algorithm requires any configuration — both fully auto-detect every parameter with zero input"
    ],
    "a": [
      0
    ],
    "e": "KMeans partitions data into exactly `k` clusters, so you must commit to that number before fitting — a real limitation when you don't know how many natural groups exist. DBSCAN instead groups points based on local density (a distance threshold and a minimum-points-per-neighborhood parameter) and lets the number of clusters emerge from the data, with the added benefit of explicitly labeling sparse points as noise/outliers rather than forcing them into a cluster. The second and third options have the actual requirement backwards or flattened into false equivalence, and the fourth ignores that both algorithms need real configuration to run meaningfully."
  },
  {
    "d": "Clustering Evaluation",
    "q": "After running KMeans without ground-truth labels, the silhouette score is used to judge cluster quality. What does a silhouette score close to 1 actually indicate, versus a score close to -1?",
    "o": [
      "A negative silhouette score is a computation error and always signals a bug, never a valid result",
      "Close to 1 means the model achieved 100% classification accuracy; close to -1 means 0% accuracy — it's the same concept as accuracy applied to clustering",
      "Close to 1: points are well-matched to their own cluster and far from neighboring clusters (good separation); close to -1: points are likely assigned to the WRONG cluster, closer to a neighboring cluster than their own",
      "The silhouette score only measures how many clusters were found, not how good the assignments are"
    ],
    "a": [
      2
    ],
    "e": "Silhouette score is computed per point from two distances — how close it is to points in its OWN cluster (cohesion) versus the nearest OTHER cluster (separation) — and averaged. A score near +1 means points sit comfortably inside a tight, well-separated cluster; a score near -1 means a point is actually closer to a different cluster than the one it got assigned to, a sign the clustering itself is likely bad in that region. It has nothing to do with classification 'accuracy' (there are no ground-truth labels being compared here at all), it doesn't just count clusters, and negative values are a legitimate, meaningful, non-error outcome that specifically flags likely misassignment."
  },
  {
    "d": "Random Forest Hyperparameters",
    "q": "`RandomForestClassifier(n_estimators=100)` is a common line to see in interviews. What does `n_estimators=100` actually configure?",
    "o": [
      "The number of features considered at each split point in every tree",
      "The total number of training samples the model is allowed to see",
      "The maximum depth allowed for each individual tree in the forest",
      "The number of individual decision trees in the ensemble — 100 separate trees are trained and their predictions combined (majority vote for classification)"
    ],
    "a": [
      3
    ],
    "e": "`n_estimators` is literally the tree count — Random Forest's core idea is training many independent trees (each typically on a bootstrap-resampled subset of the data) and combining their predictions, since averaging over many imperfect, decorrelated trees reduces the overfitting any single deep tree would suffer. Tree depth is a separate parameter (`max_depth`), the number of features considered per split is yet another separate parameter (`max_features`), and `n_estimators` has no relationship to how many training rows the model uses — that's controlled by your train/test split, not this hyperparameter."
  },
  {
    "d": "Regularization",
    "q": "Both `LinearRegression` and `Ridge` fit a linear model, but `Ridge` adds one specific thing `LinearRegression` doesn't. What is it, and what problem does it solve?",
    "o": [
      "Ridge uses a completely different algorithm (gradient boosting) internally, unrelated to linear regression's least-squares fitting",
      "Ridge adds an L2 penalty on the size of the coefficients to the loss being minimized, shrinking large weights toward zero to reduce overfitting/variance — `LinearRegression` has no such penalty and can produce arbitrarily large coefficients on noisy or correlated features",
      "Ridge removes outliers from the training data automatically before fitting; `LinearRegression` does not",
      "There's no meaningful difference — `Ridge` is just an alias for `LinearRegression` with a different name"
    ],
    "a": [
      1
    ],
    "e": "Ridge regression minimizes the usual squared-error loss PLUS a penalty term proportional to the sum of squared coefficients (L2 regularization), which discourages any single coefficient from growing huge — especially valuable when features are correlated (multicollinearity), where plain `LinearRegression` can produce wildly large, unstable coefficients that fit the training noise rather than the true signal. It is still fundamentally the same least-squares linear model under the hood, not a different algorithm like boosting. It does nothing about outlier removal — that's a separate preprocessing concern entirely. And the two are meaningfully different estimators, not aliases — fitting the same data with each generally gives different coefficients."
  },
  {
    "d": "Preprocessing API",
    "q": "On a `StandardScaler`, there's `fit()`, `transform()`, and `fit_transform()`. Given a train/test split, which sequence is correct to avoid leaking test-set statistics into scaling?",
    "o": [
      "`scaler.fit(X_test)` first to learn the real-world distribution, then `transform()` both sets using test statistics",
      "`scaler.fit_transform(X_train)` to learn the mean/std from training data AND scale it in one step, then `scaler.transform(X_test)` (transform ONLY, reusing the training-learned statistics, never re-fitting on test)",
      "Call `fit()` once on the combined train+test data, then `transform()` each set separately",
      "`scaler.fit_transform(X_train)` then `scaler.fit_transform(X_test)` — fit fresh on each set independently for consistency"
    ],
    "a": [
      1
    ],
    "e": "`fit()` computes the mean and standard deviation from whatever data you give it; `transform()` applies an already-learned mean/std to scale data. The correct discipline is: learn the scaling statistics ONLY from training data (`fit_transform(X_train)`), then apply those SAME learned statistics to the test set (`transform(X_test)` — note: no `fit` here). Calling `fit_transform` on the test set separately (option 2) would compute test-specific statistics, meaning your test set no longer reflects genuinely unseen data scaled the way production data would be. Fitting on test data at all (option 3) leaks test-set information backward into how you preprocess. And fitting on the combined data (option 4) still leaks test statistics into the scaling that touches training data — the fix must use train-only statistics for both fit and every subsequent transform."
  },
  {
    "d": "K-Nearest Neighbors",
    "q": "For `KNeighborsClassifier(n_neighbors=5)`, how does the model actually decide the predicted class for a new point?",
    "o": [
      "It clusters the training data into 5 groups first, then assigns the new point to whichever group's centroid is closest",
      "It picks the single closest training point and copies its label, ignoring the other 4 neighbors entirely regardless of `n_neighbors`",
      "It finds the 5 training points closest (by distance) to the new point, and predicts the MAJORITY class among those 5 neighbors",
      "It trains 5 separate models and averages their predicted probabilities together"
    ],
    "a": [
      2
    ],
    "e": "KNN has no real 'training' phase beyond storing the data — at prediction time, it computes the distance from the new point to every training point, takes the `n_neighbors` closest ones, and votes: whichever class is most common among those neighbors becomes the prediction. It isn't an ensemble of 5 separate trained models (that description fits something like bagging, not KNN). `n_neighbors=5` specifically means all 5 neighbors participate in the vote — using only the single nearest point would be `n_neighbors=1`, a meaningfully different (and much noisier) model. And it doesn't pre-cluster the data at all — KMeans does that, KNN is a different, purely distance-based, per-query algorithm."
  },
  {
    "d": "Model Selection Discipline",
    "q": "`GridSearchCV(model, param_grid, cv=5)` searches hyperparameters. Why does it use 5-FOLD cross-validation internally rather than just checking each hyperparameter combination's accuracy on the training set directly?",
    "o": [
      "Training-set accuracy alone would just reward whichever hyperparameters let the model memorize the training data hardest (e.g. an unconstrained tree depth); cross-validation instead estimates how each combination performs on data it wasn't fit on, which is what you actually care about",
      "GridSearchCV requires cv=5 specifically as a hard-coded, unchangeable minimum — no other value is technically valid",
      "5-fold CV is only used to make the search run faster — it has no effect on which hyperparameters get selected",
      "Training-set accuracy and cross-validated accuracy always produce the identical ranking of hyperparameters, so the choice doesn't actually matter"
    ],
    "a": [
      0
    ],
    "e": "If you picked hyperparameters purely by which combination scores highest ON the training data, you'd systematically favor whichever setting overfits hardest (e.g. no depth limit on a tree lets it perfectly memorize training rows) — a model that then generalizes terribly. Cross-validation instead repeatedly holds out a fold as a stand-in for unseen data and averages performance across folds, which is a genuinely different (and much more honest) signal than training accuracy, so the ranking of hyperparameter combinations really can and does change between the two approaches. CV also isn't primarily a speed optimization — it's typically SLOWER than a single fit since it fits the model `cv` times per combination. And `cv` is a normal integer parameter you can set to any reasonable value (3, 5, 10, etc.), not a fixed requirement."
  },
  {
    "d": "Support Vector Machines",
    "q": "An SVM classifier is described as finding the boundary that \"maximizes the margin\" between classes. What does \"maximizing the margin\" actually mean, and why does it help generalization?",
    "o": [
      "It means maximizing the total number of support vectors used, since more support vectors always means a better-fit boundary",
      "It means choosing the boundary that passes through as many training points as possible, maximizing how many points touch the line exactly",
      "It refers to maximizing training accuracy directly — margin and accuracy are two names for the same underlying quantity",
      "It means choosing the decision boundary that is as far as possible from the NEAREST points of either class (the support vectors) — a wide margin leaves more room for small errors/noise in new data before a point crosses to the wrong side"
    ],
    "a": [
      3
    ],
    "e": "The margin is the distance between the decision boundary and the closest training points of each class (the support vectors that actually define it) — maximizing that gap means the boundary sits as far as possible from ambiguous, borderline cases, so a slightly noisy or slightly shifted new point is less likely to end up on the wrong side of the line than it would with a boundary that hugged the data tightly. It has nothing to do with passing through training points (that would describe overfitting, the opposite goal) or with maximizing the support-vector COUNT (fewer, well-chosen support vectors defining a wide margin is the actual goal, not more of them). And margin is a geometric property of the decision boundary, a genuinely different quantity from training accuracy — a model can have high training accuracy with a razor-thin, fragile margin."
  }
]
</script>
<div class="topic-quiz-mount"></div>
