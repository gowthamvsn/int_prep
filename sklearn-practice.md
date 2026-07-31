# scikit-learn Practice — Built as a Chain, Not a List

Most snippets assume `X, y` already exist as a feature matrix and target array/Series. Each cluster is one continuous thread — every question inherits the answer before it, closing with a worked summary example.

---

## Cluster 1 — Splitting and Preprocessing Without Leaking

### 1. How do you split data into train/test sets correctly?
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```
Without `stratify=y`, a random split can (especially on imbalanced data) produce a test set with a very different class balance than train — `stratify` forces both splits to preserve the same class proportions as the full dataset. Without `random_state`, every run produces a different split, making results non-reproducible between you and anyone reviewing your work.

### 2. Given a clean split, how do you standardize features WITHOUT leaking test information into it?
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit_transform: learns mean/std FROM train, then applies
X_test_scaled = scaler.transform(X_test)          # transform only: reuses train's mean/std, does NOT refit
```
Fitting the scaler on test data lets information about the test set's distribution leak into preprocessing — a real, common form of data leakage that inflates reported performance versus what the model would actually do on truly unseen data. Never call `fit_transform` on test data — only `transform`.

### 3. Given that numeric scaling must not touch test data, what about CATEGORICAL columns — how do you encode them, and what happens if test has a category train never saw?
```python
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
X_train_enc = enc.fit_transform(X_train[["depot"]])
X_test_enc = enc.transform(X_test[["depot"]])     # any category not seen in train is handled, not crashed
```
Without `handle_unknown="ignore"`, a category appearing in test/production but never seen during training raises an error at inference time — `"ignore"` instead encodes it as all-zeros, letting the pipeline keep running (worth monitoring for, but not a hard crash).

### 4. Both scaler and encoder above need the SAME fit-on-train-only discipline. How do you enforce that structurally, not just by remembering it every time?
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
A `Pipeline` guarantees the scaler is fit ONLY on whatever data `.fit()` is called with — inside cross-validation, this means each fold's scaler is refit on just that fold's training portion, which is the only way to avoid leakage during CV. Manually scaling before CV and passing the pre-scaled data in is a very common, very real leakage bug.

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

### 5. Given that a Pipeline chains steps in SEQUENCE, what if numeric and categorical columns need genuinely DIFFERENT transformers applied in PARALLEL?
```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

preprocess = ColumnTransformer([
    ("num", StandardScaler(), ["wear_pct", "age_days"]),
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["depot"]),
])
```
Numeric and categorical columns need fundamentally different treatment — applying `StandardScaler` to a categorical column (or one-hot encoding to a continuous one) is a common beginner error `ColumnTransformer` structurally prevents by naming exactly which columns get which transformer. This `preprocess` object then slots into a `Pipeline` (question 4) as its first step, exactly like `StandardScaler` did.

### Summary example
A dataset with both `wear_pct` (numeric) and `depot` (categorical): `ColumnTransformer` routes each column type to its correct transformer, wrapped inside a `Pipeline` alongside the classifier — the WHOLE thing then gets `.fit()`-ed only on `X_train`, so every leakage risk from questions 2-5 is closed by one structural choice rather than four separate manual disciplines to remember.

---

## Cluster 2 — Cross-Validation and Hyperparameter Tuning

### 1. A single train/test split gives one score. How do you get a more reliable estimate?
```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="roc_auc")
print(scores.mean(), scores.std())    # report BOTH — std tells you how stable the estimate is
```
Two models with the same mean CV score but very different standard deviations are NOT equally trustworthy — high variance across folds means the estimate itself is unstable, which a single mean number hides.

### 2. Given imbalanced classes, does plain k-fold CV risk the same class-balance problem `stratify` fixed in Cluster 1?
```python
from sklearn.model_selection import StratifiedKFold, cross_val_score
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1")
```
Yes — and `shuffle=True` matters for a related but distinct reason: without it, `StratifiedKFold` splits the data in its ORIGINAL order, so if the data happens to be sorted or grouped in any way (common with real-world exports), folds can end up systematically different. Shuffling (with a fixed seed for reproducibility) avoids that.

### 3. Given a reliable CV estimate, how do you actually search for the best hyperparameters?
```python
from sklearn.model_selection import GridSearchCV
param_grid = {"clf__C": [0.01, 0.1, 1, 10], "clf__penalty": ["l1", "l2"]}
grid = GridSearchCV(pipe, param_grid, cv=5, scoring="roc_auc", n_jobs=-1)
grid.fit(X_train, y_train)
print(grid.best_params_, grid.best_score_)
```
When tuning a `Pipeline`, parameter names must be `<step_name>__<param_name>` so `GridSearchCV` knows which pipeline step each hyperparameter belongs to — a plain `"C"` instead of `"clf__C"` raises an error. `n_jobs=-1` uses all available CPU cores in parallel — grid search over many combinations is embarrassingly parallel, and leaving this at the default (1) can make tuning take many times longer than necessary.

### 4. Grid search tries EVERY combination — what if there are too many hyperparameters for that to be practical?
```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform
param_dist = {"clf__C": loguniform(1e-3, 1e2)}
search = RandomizedSearchCV(pipe, param_dist, n_iter=20, cv=5, scoring="roc_auc", random_state=42, n_jobs=-1)
search.fit(X_train, y_train)
```
Grid search cost grows multiplicatively with every hyperparameter added; randomized search samples a fixed budget (`n_iter`) regardless of how many hyperparameters/values are in play, and empirically finds near-optimal regions almost as well as exhaustive search for a fraction of the compute. `loguniform` for `C` specifically: regularization strength spans orders of magnitude (0.001 to 100) — sampling uniformly on a linear scale wastes most samples in one narrow range; log-uniform spreads them evenly across orders of magnitude.

### Summary example
Tuning a logistic regression's `C` and `penalty`: with only 2 hyperparameters and a handful of values each, `GridSearchCV` (8 total combinations) is cheap enough to run exhaustively. Scaling up to 5 hyperparameters with 10 values each would be 100,000 combinations — at that point `RandomizedSearchCV` with a fixed `n_iter=50` budget and `loguniform` sampling for anything spanning orders of magnitude becomes the only practical choice, trading a small amount of thoroughness for a massive compute saving.

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
`predict_proba` returns a column per class; for binary classification, column 0 is P(negative), column 1 is P(positive) — passing the wrong column to `roc_auc_score` silently computes AUC for the wrong class.

### 2. Before trusting ANY of those numbers as "good," what's the cheapest sanity check — is the model even beating doing nothing?
```python
from sklearn.dummy import DummyClassifier
baseline = DummyClassifier(strategy="most_frequent")   # always predicts the majority class
baseline.fit(X_train, y_train)
print(baseline.score(X_test, y_test))    # THIS is the number your real model has to beat
```
On a 95%-majority-class dataset, `DummyClassifier` scores 95% accuracy doing NOTHING intelligent at all — establishing this number first is what stops you from being impressed by a "good-looking" accuracy that a trivial baseline already achieves.

### 3. Given a model that beats the baseline on test data, how do you quickly check if it's actually overfitting?
```python
print("train:", pipe.score(X_train, y_train))
print("test:", pipe.score(X_test, y_test))
```
Train score much higher than test score is the single fastest overfitting diagnostic available — before touching any hyperparameter, always look at this gap first, since it directly tells you whether you have a variance problem (overfitting) or a bias problem (both scores low = underfitting), needing very different fixes.

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
A model scores 94% test accuracy — impressive, until `DummyClassifier(strategy="most_frequent")` also scores 94% because the classes are that imbalanced, revealing the "impressive" model added zero real signal. Checking train (96%) vs. test (94%) separately shows the gap is small, so it's not overfitting — the real problem is that accuracy was the wrong metric to be impressed by at all, which is exactly why `classification_report`'s precision/recall breakdown per class matters more than the single accuracy number here.

---

## Cluster 4 — Handling Class Imbalance

### 1. Given imbalanced classes (Cluster 3's exact scenario), what's the simplest fix, directly in the model itself?
```python
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(class_weight="balanced")    # auto-reweights inversely proportional to class frequency
```
`class_weight="balanced"` is often the simpler, leakage-free FIRST move — it changes the loss function's penalty for misclassifying the minority class rather than fabricating synthetic data, and is usually the right first thing to try before anything more involved.

### 2. If reweighting the loss isn't enough, how do you actually generate more minority-class examples — and what's the one rule that must never be broken doing it?
```python
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
```
SMOTE synthesizes new minority-class points by interpolating between real ones — if applied BEFORE the train/test split, synthetic points derived from what becomes test data can leak into training, and the test set's synthetic points don't represent real unseen data at all. SMOTE must be fit only on the training split, never before splitting — `imblearn`'s own `Pipeline` (not sklearn's) exists specifically to keep SMOTE properly scoped inside cross-validation folds, the exact same leakage discipline as Cluster 1's `StandardScaler`.

### Summary example
A fraud dataset with 2% positive class: trying `class_weight="balanced"` first is cheap and leakage-free — if that alone gets acceptable recall, SMOTE isn't even needed. If it doesn't, SMOTE (correctly scoped to only the training fold, via `imblearn.pipeline.Pipeline`) generates synthetic fraud examples — but applying SMOTE to the FULL dataset before splitting would leak synthetic points derived from test-set fraud cases into training, silently inflating the test score in a way that wouldn't hold up in production.

---

## Cluster 5 — Feature Selection and Importance

### 1. With many candidate features, how do you automatically select the most predictive ones?
```python
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(score_func=f_classif, k=10)
X_train_sel = selector.fit_transform(X_train, y_train)
selected_cols = X_train.columns[selector.get_support()]    # get_support() -> boolean mask of kept columns
```
`f_classif` (ANOVA F-value — the same F-statistic idea from `stats-scipy-practice.md`'s ANOVA cluster) assumes a roughly linear relationship between each feature and the target; `mutual_info_classif` captures non-linear relationships too but is more expensive to compute — pick based on whether you suspect non-linear feature-target relationships.

### 2. Instead of selecting features BEFORE training, how do you find out which features a trained tree-based model actually relied on?
```python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
importances = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
```
`max_depth` matters here specifically: unconstrained trees (`max_depth=None`) grow until every leaf is pure, which tends to overfit AND inflates the apparent importance of high-cardinality features that can split data very finely just by chance — capping depth is one of the simplest regularization levers for tree ensembles, directly the same bias-variance dial from `ml-models-practice.md`'s dartboard visual.

### Summary example
30 candidate features, most of them noise: `SelectKBest(f_classif, k=10)` narrows to the 10 most linearly predictive BEFORE training anything, while training a `RandomForestClassifier` on all 30 and reading `feature_importances_` instead reveals which features the model actually leaned on AFTER training — the two approaches can disagree on a feature with a real but non-linear relationship to the target, which `f_classif` would underrate and a tree-based importance would correctly surface.

---

## Cluster 6 — Evaluating Regression

### 1. Classification has accuracy/precision/recall — what's the equivalent for a regression model?
```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
y_pred = reg.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)          # same units as target, robust to outliers
rmse = mean_squared_error(y_test, y_pred, squared=False)   # penalizes large errors more than MAE
r2 = r2_score(y_test, y_pred)                       # fraction of variance explained, 1.0=perfect
```

### 2. Given both MAE and RMSE measure error, why report BOTH instead of just picking one?
If RMSE is much larger than MAE, that gap itself is diagnostic — it means a small number of predictions have LARGE errors dragging RMSE up (since RMSE squares errors before averaging, which MAE doesn't), which MAE alone would mask entirely.

### Summary example
A model's MAE is $500 but its RMSE is $3,200 — that large gap immediately signals a handful of predictions with huge errors (RMSE's squaring punishes them disproportionately) rather than uniformly mediocre predictions across the board, pointing an investigation toward specific outlier cases rather than the model's general calibration.

---

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
K-means' result depends on random initial centroid placement and can get stuck in a bad local optimum — `n_init=10` runs it 10 times from different random starts and keeps the best (lowest inertia) run; setting it too low risks a genuinely worse clustering purely from bad luck.

### 2. With too many features to even visualize, how do you reduce dimensionality first?
```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X_train_scaled)   # ALWAYS scale first -- PCA is sensitive to feature scale
print(pca.explained_variance_ratio_)        # fraction of total variance each component captures
```
PCA finds directions of maximum variance (the eigenvector machinery from `math-foundations-refresher.md`) — a feature measured in larger raw units (income in dollars vs. age in years) will dominate the principal components purely from its SCALE, not because it's actually more informative, unless everything is standardized first — the exact same "always scale first" discipline as Cluster 1's `StandardScaler`, just feeding PCA instead of a classifier.

### Summary example
Clustering customers on 20 raw features including both "annual income" (tens of thousands) and "age" (tens): without scaling first, K-means would effectively cluster almost entirely on income, since its raw numeric range dwarfs age's — `StandardScaler` before `KMeans` (or before `PCA` for a 2D visualization of the same clusters) puts every feature on equal footing first, so the resulting clusters reflect genuine multi-feature similarity, not one feature's arbitrary unit of measurement.

---

## Cluster 8 — Persisting a Trained Model

### 1. After all the above, how do you save a fitted pipeline so it doesn't need retraining every time?
```python
import joblib
joblib.dump(pipe, "model.joblib")
loaded_pipe = joblib.load("model.joblib")
```
`joblib` over plain `pickle`: it's more efficient specifically for objects containing large NumPy arrays (like a fitted model's learned weights), which is exactly what most scikit-learn estimators are — it's the library's own recommended serialization tool for this reason.

### Summary example
A tuned `Pipeline` (scaler + classifier, from Cluster 1) that took 20 minutes of `GridSearchCV` to produce gets saved once with `joblib.dump`, then loaded in a separate serving script with `joblib.load` — the entire fitted pipeline, preprocessing included, restored in milliseconds instead of re-running the full training and tuning process.

---

## Practice Q&A (Self-Test)

**Q1. In `train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)`, what does each of `stratify` and `random_state` guard against?**
A: `stratify=y` forces both the train and test splits to preserve the same class proportions as the full dataset, preventing a random split from producing a test set with a very different class balance (especially risky on imbalanced data). `random_state` fixes the randomness so the split is reproducible across runs and reviewers.

**Q2. Why is `scaler.fit_transform(X_test)` a data leakage bug, and what should you call instead?**
A: `fit_transform` would relearn the mean/std from the test set itself, letting information about the test distribution leak into preprocessing and inflating reported performance versus what the model would actually see on truly unseen data. The correct call is `scaler.transform(X_test)`, which reuses the mean/std already learned from `X_train` without refitting.

**Q3. What does `OneHotEncoder(handle_unknown="ignore")` do when `X_test` contains a category never seen in `X_train`, and what's the alternative behavior without it?**
A: With `handle_unknown="ignore"`, an unseen category is encoded as all-zeros and the pipeline keeps running (worth monitoring, but not a crash). Without it, encountering an unseen category at inference/test time raises an error, halting the pipeline.

**Q4. Why does wrapping `StandardScaler` and `LogisticRegression` in a `Pipeline` matter beyond convenience, specifically during cross-validation?**
A: A `Pipeline` guarantees the scaler is fit only on whatever data `.fit()` is called with, so inside cross-validation each fold's scaler is refit on just that fold's training portion. Manually scaling the full dataset before running CV and passing the pre-scaled data in is a common leakage bug this structurally prevents.

**Q5. What problem does `ColumnTransformer` solve when you have both numeric and categorical columns?**
A: It lets you apply different transformers to different named columns in one object — e.g., `StandardScaler` to `["wear_pct", "age_days"]` and `OneHotEncoder` to `["depot"]`. This structurally prevents the common beginner error of applying scaling to categorical columns or one-hot encoding to continuous ones.

**Q6. Why does the file recommend printing both `scores.mean()` and `scores.std()` after `cross_val_score`?**
A: Two models can have the same mean CV score but very different standard deviations across folds — high variance means the estimate itself is unstable and less trustworthy, which the mean alone hides. Reporting both gives a fuller picture of reliability, not just central tendency.

**Q7. In `GridSearchCV` with `param_grid = {"clf__C": [...], "clf__penalty": [...]}`, why is the `clf__` prefix required instead of just `"C"`?**
A: When tuning a `Pipeline`, parameter names must follow `<step_name>__<param_name>` so `GridSearchCV` knows which pipeline step (here, the `"clf"` step) each hyperparameter belongs to. Passing a plain `"C"` instead of `"clf__C"` raises an error because it doesn't map to any pipeline step's parameter.

**Q8. Why does the file use `loguniform(1e-3, 1e2)` for sampling `C` in `RandomizedSearchCV` instead of a plain uniform range?**
A: Regularization strength like `C` naturally spans orders of magnitude (0.001 to 100); sampling uniformly on a linear scale would waste most samples in one narrow range. Log-uniform sampling spreads samples evenly across orders of magnitude, matching how `C`'s effect actually varies.

**Q9. When computing `roc_auc_score(y_test, y_proba)`, why is `y_proba = pipe.predict_proba(X_test)[:, 1]` specifically indexed with `[:, 1]` rather than `[:, 0]`?**
A: `predict_proba` returns one probability column per class — for binary classification, column 0 is P(negative) and column 1 is P(positive). `roc_auc_score` needs probabilities/scores for the positive class, so passing the wrong column silently computes AUC for the wrong class.

**Q10. Why does the file suggest `class_weight="balanced"` as a first move for imbalanced classification, before trying SMOTE?**
A: `class_weight="balanced"` reweights the loss function's penalty for misclassifying the minority class rather than fabricating synthetic data points, making it a simpler, leakage-free first option. SMOTE, by contrast, must be fit only on the training split (never before the train/test split) to avoid synthetic points leaking into or contaminating the test set.


---

## Video-Sourced Practice MCQs (Set 2)

A second practice set for scikit-learn, built the same way as this hub's NCA-GENL community bank: topics checked against a real YouTube scikit-learn-interview-prep video, then written up as fully original multiple-choice questions here (every option and explanation below is original, not copied from the video). These focus on angles the clusters above don't already drill in depth -- clustering algorithm choice (KMeans vs. DBSCAN) and evaluation, core hyperparameter meanings, regularization's actual mechanism, the fit/transform API discipline, KNN's prediction mechanics, why CV beats training accuracy for model selection, and what SVM's margin maximization is actually doing.

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
