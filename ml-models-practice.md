# Classical ML Models Practice

This doc assumes `X_train, X_test, y_train, y_test` already exist. See `sklearn-practice.md` for how that split gets made.

New to terms like fitting, overfitting, hyperparameters, or leakage? `sklearn-practice.md` has a short primer at the top that defines all five. This doc builds on those definitions instead of repeating them.

Each cluster below is one continuous thread. Every question builds on the answer before it. Each cluster ends with one worked example, using real numbers, start to finish.

---

## Cluster 1 — Linear Models

### 1. How do you fit a plain linear regression, and what do the coefficients mean?
```python
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)
print(dict(zip(X_train.columns, reg.coef_)))    # one coefficient per feature
print("intercept:", reg.intercept_)
```
Linear regression fits a straight-line formula:
```
prediction = intercept + coef1·feature1 + coef2·feature2 + ...
```
Each **coefficient** is the learned multiplier on one feature. Read it as "for every extra unit of this feature, add this much to the prediction." The **intercept** is just the prediction when every feature is zero.

Here's a reading trap worth knowing about. A coefficient's size reflects two things at once: the feature's real effect, and its raw scale. A feature measured in the thousands (like income) will get a tiny coefficient even if it matters a lot. A feature measured in single digits (like years of experience) gets an inflated-looking coefficient just from being on a smaller scale. If you want to compare coefficients as "importance," standardize the features first — otherwise you're comparing apples to oranges.

### 2. Raw coefficients can overfit to noise. How do you add regularization, and how do you choose L1 vs. L2?
```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet
ridge = Ridge(alpha=1.0)    # L2: shrinks all coefficients toward zero, rarely to exactly zero
lasso = Lasso(alpha=0.1)    # L1: can shrink coefficients to EXACTLY zero -- performs feature selection
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)   # mix of both
```
**Regularization** means adding a penalty on coefficient size to whatever quantity the model is minimizing during fitting. Every large coefficient now has to justify itself with real predictive payoff. That's what stops the model from contorting itself around training noise.

The L1-vs-L2 distinction, worked out with real numbers, lives in `math-foundations-refresher.md`'s norms section. The short version:
- **Lasso (L1)** can shrink a coefficient all the way to exactly zero. That's real feature selection — if you suspect only a handful of features actually matter, Lasso will tell you which ones by zeroing out the rest.
- **Ridge (L2)** shrinks every coefficient toward zero, but rarely reaches exactly zero. Use it as the default when you believe most features contribute a little (especially with multicollinearity) and you don't want to throw any of them out entirely.

`alpha` is the regularization strength. Higher alpha means more shrinkage, which means a simpler model. Tune it with cross-validation — don't guess it.

### 3. The target is a classification, not a number. How does a coefficient even apply, and how do you read it?
```python
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(X_train, y_train)
odds_ratios = np.exp(clf.coef_[0])    # exponentiate the log-odds coefficient to get an interpretable odds ratio
```
Two quick definitions first. **Odds** are probability-of-happening divided by probability-of-not-happening — a 75% probability is odds of 3-to-1, written just as the number 3. **Log-odds** are the natural logarithm of that number. That's the scale logistic regression actually works in internally, because taking a log turns "multiply the odds" into "add to the score" (see `math-foundations-refresher.md`'s calculus section for why `eˣ` and `ln` pair up like this).

So a logistic regression's coefficients are in log-odds units, not probability units. A coefficient of 0.69 does not mean "69% more likely." Here's what it actually means, step by step:

1. Take the coefficient: `0.69`.
2. Exponentiate it: `e^0.69 ≈ 2.0`.
3. Read that 2.0 as an odds ratio: a one-unit increase in that feature roughly doubles the odds.

Reporting the raw coefficient instead of the odds ratio is one of the most common ways people misread logistic regression output.

### Summary example
A logistic regression predicting equipment failure has a coefficient of 0.69 on `age_days`. Reported raw, that number looks unremarkable. Run it through `np.exp(0.69) ≈ 2.0` and the real story shows up: each additional standardized unit of age roughly doubles the odds of failure. That's the number worth putting in front of a stakeholder — not the raw log-odds coefficient.

---

> 🔗 **Hands-on reps:** [Code Drills 5 — Train & Evaluate a RandomForest](/topic/code-drills-classical-ml#cluster-1-train-evaluate-a-randomforest)

## Cluster 2 — Decision Trees

### 1. How do you build and visualize a single decision tree?
```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, random_state=42)
tree.fit(X_train, y_train)
plot_tree(tree, feature_names=X_train.columns, filled=True, fontsize=8)
```
A **decision tree** is a learned flowchart of yes/no questions on feature values: "is `wear_pct` > 40? → is `age_days` > 300? → predict FAIL." To make a prediction, you walk a new row down the flowchart until it lands on a **leaf** — an end node — and answer with whatever the training rows in that leaf mostly were.

Two settings control how big the flowchart is allowed to get:
- `max_depth` caps how many questions deep it can go.
- `min_samples_leaf` stops a leaf from being created based on just 1-2 data points, which is almost certainly noise, not signal.

Both are regularization levers against overfitting. And they matter together, not separately — a tree can still overfit badly even at a modest `max_depth` if `min_samples_leaf` is left unconstrained, because it can grow tiny, unreliable leaves instead of going deeper.

### 2. How does the tree actually decide where to split?
```python
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion="gini")   # or criterion="entropy"
```
At each candidate split, the tree measures how "impure" — how mixed-class — each resulting group would be. Gini impurity and entropy are two different mathematical measures of that mixedness. The tree just picks whichever split reduces impurity the most.

In practice they usually agree on the best split. Gini is slightly cheaper to compute, since it skips the logarithm entropy needs, which is why it's scikit-learn's default.

### Summary example
A tree considers splitting on `wear_pct > 40`. Before the split, the node is a 50/50 mix of pass and fail — Gini impurity near its maximum. After the split, one branch is 90% fail and the other is 90% pass. Both branches are now far purer.

That impurity reduction is exactly what the tree scores when comparing this split against every other candidate threshold, on every other feature, at every node.

---

## Cluster 3 — Ensembles: Bagging vs. Boosting

### 1. A single tree overfits easily. How do you combine many trees to do better, and why does that even help?
```python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=300, max_features="sqrt", random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
```
An **ensemble** is many models voting together. The idea: individual models' mistakes partly cancel out, the same way averaging many noisy measurements gets you closer to the truth than trusting just one.

A Random Forest is the "bagging" kind of ensemble. Each tree trains on a **bootstrap sample** — rows drawn at random *with replacement*, so each tree sees a slightly different version of the data.

Here's the part people forget: each tree is also only allowed to consider a random subset of features at every split (`max_features="sqrt"`, meaning the square root of the total feature count, by convention). That forces the trees to differ from each other even when trained on the same data.

Why this matters: without that feature-level randomness, all the trees would tend to make similar splits on the same dominant features. The ensemble wouldn't reduce variance much, because the trees would be too similar to each other to average out any real error. The feature subsampling is what actually decorrelates the trees, and that decorrelation is what makes averaging them powerful.

### 2. A Random Forest averages independent trees. What's the alternative — trees that learn from each other in sequence?
```python
import xgboost as xgb
model = xgb.XGBClassifier(
    n_estimators=200, learning_rate=0.05, max_depth=4,
    subsample=0.8, colsample_bytree=0.8, random_state=42,
)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
```
That alternative is **boosting**. A Random Forest builds many independent trees and averages them, which reduces variance. Boosting builds trees sequentially instead — each new tree is trained specifically to correct the previous ensemble's errors, which reduces bias.

That sequential correction is why boosting typically needs a low `learning_rate` (how much each new tree's correction counts) and more trees than a forest would use. You're trading training time for often-higher accuracy. It's also why boosting is more prone to overfitting if left unregularized — `subsample`, `colsample_bytree`, and `max_depth` all exist to fight that.

Here are the two shapes side by side: parallel voters vs. an error-correcting relay.
```
BAGGING (Random Forest)                    BOOSTING (XGBoost / LightGBM)

 data ──┬──▶ tree 1 ──┐                    data ──▶ tree 1   "here's my best guess"
 (each  ├──▶ tree 2 ──┤                               │ errors
 tree:  ├──▶ tree 3 ──┼──▶ AVERAGE                    ▼
 random ├──▶  ...     ──┤    the votes      residuals ──▶ tree 2   "I'll fix what 1 got wrong"
 rows + └──▶ tree 300 ─┘                              │ remaining errors
 random                                               ▼
 features)                                 residuals ──▶ tree 3   "I'll fix what 1+2 still miss"
                                                      │  ... × n_estimators
 trees never see each other;                          ▼
 independence IS the strength              final = tree1 + lr·tree2 + lr·tree3 + …
 (errors cancel in the average)            each tree only makes sense given the ones before
```

### 3. Boosting needs both `learning_rate` and `n_estimators`. Why must they be tuned together, not separately?
```python
# a lower learning_rate needs MORE estimators to reach the same fit -- they trade off directly
model_a = xgb.XGBClassifier(n_estimators=100, learning_rate=0.3)   # fast, coarse corrections
model_b = xgb.XGBClassifier(n_estimators=1000, learning_rate=0.01)  # slow, fine corrections
```
A high learning rate with few trees converges fast, but it can overshoot, oscillate, and generalize worse. A low learning rate with many trees trains slower, but each correction is gentler and it typically generalizes better.

This is the single most important hyperparameter pair to tune together in any gradient boosting model. Tuning either one alone misses how strongly they interact.

### 4. Instead of guessing the right `n_estimators`, how do you let the model stop itself at the right point?
```python
model = xgb.XGBClassifier(n_estimators=1000, learning_rate=0.05, early_stopping_rounds=20)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
print("best iteration:", model.best_iteration)
```
`early_stopping_rounds` stops training automatically once the validation score hasn't improved for 20 consecutive rounds. It directly implements the "stop at the validation-loss minimum" idea from the loss-curve diagnostic (`ds-fundamentals`), using real held-out performance instead of a fixed, guessed tree count.

### 5. Is there a third variant beyond bagging (Random Forest) and boosting (XGBoost) worth knowing?
```python
import lightgbm as lgb
model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=42)
model.fit(X_train, y_train)
```
LightGBM. It's still boosting — same sequential-correction idea as question 2 — but it grows trees differently. XGBoost's default grows **level-wise**, splitting every leaf at the current depth before going deeper. LightGBM grows **leaf-wise**, always splitting whichever single leaf reduces loss the most next.

Leaf-wise growth can reach a better fit with fewer splits. The tradeoff: `num_leaves` needs capping directly, not just depth, because a leaf-wise tree can grow very unevenly deep down one branch and overfit a small subset of the data.

### Summary example
The same dataset, fit three ways.

1. `RandomForestClassifier` averages 300 independent, feature-subsampled trees. It reduces variance, parallelizes well, and is less sensitive to tuning.
2. `XGBClassifier` with `early_stopping_rounds=20` builds trees sequentially, each one correcting the prior ensemble's errors. It reduces bias, needs `learning_rate` and `n_estimators` tuned together, and often edges out the forest in raw accuracy.
3. `LGBMClassifier` does the same sequential correction, but leaf-wise. It's often faster on large data — at the cost of needing `num_leaves` watched closely so one branch doesn't run away.

---

## Cluster 4 — Trusting Feature Importance

### 1. How do you read which features a boosted tree model actually relied on?
```python
import pandas as pd
importance = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
```
Worth knowing before you trust this list: the default importance (`"gain"`-based by column — check `importance_type`) can overweight high-cardinality features. Those are columns with many distinct values, like a ZIP code or anything ID-like. They tend to get used for a lot of splits simply because they offer more possible split points to choose from, not because they're actually more predictive.

### 2. Built-in tree importance can mislead. How do you get a more trustworthy, model-agnostic version?
```python
from sklearn.inspection import permutation_importance
result = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
importance = pd.Series(result.importances_mean, index=X_test.columns).sort_values(ascending=False)
```
**Permutation importance** measures something more direct: how much a model's actual test performance drops when one feature's values get randomly shuffled, breaking its relationship with the target. If shuffling a feature barely hurts performance, it wasn't doing much real work.

This works identically for any model type, not just trees, and it isn't biased by a feature's cardinality or by how a specific algorithm happens to count "usage."

### Summary example
An XGBoost model's built-in importance ranks a high-cardinality `customer_id`-adjacent feature as the top predictor. Permutation importance on that same model tells a different story: shuffling that feature barely hurts test performance at all. The built-in ranking turns out to be an artifact of split-count, not real predictive value.

---

## Cluster 5 — Support Vector Machines: the Concepts, Before the Real Numbers

### 1. How do you fit an SVM, and why does the kernel choice matter so much?
```python
from sklearn.svm import SVC
svm_linear = SVC(kernel="linear", C=1.0)
svm_rbf = SVC(kernel="rbf", C=1.0, gamma="scale")
```
A linear kernel can only separate classes with a straight line, or a flat plane in higher dimensions. The RBF (radial basis function) kernel does something cleverer: it implicitly projects the data into a much higher-dimensional space, where a straight-line separator in *that* space corresponds to a curved boundary back in the original space.

Use linear when you believe the true boundary is roughly straight — it's also faster and more interpretable. Use RBF when you suspect a genuinely non-linear boundary and you have enough data to estimate one reliably.

### 2. RBF projects into a higher-dimensional space. What do the two knobs controlling that projection, `C` and `gamma`, actually do?
```python
SVC(kernel="rbf", C=100, gamma=0.01)   # high C: less tolerance for misclassified training points (can overfit)
SVC(kernel="rbf", C=0.1, gamma=10)      # low C: more tolerant of margin violations (can underfit); high gamma: very local/wiggly boundary
```
They control different things, and both matter together:
- `C` trades margin width against training-error tolerance. A high `C` means a narrow margin, few training errors allowed, and real risk of overfitting.
- `gamma` controls how far a single training point's influence reaches. A high `gamma` means a very local, wiggly decision boundary — also prone to overfitting.

A grid search over both jointly, not one at a time, is standard practice, because their effects interact.

Here's a memory hook for both of them: the classic dartboard. "High C" and "high gamma" are really just two different names for the same underlying bias/variance knob that every model in this doc has, in a different costume each time.
```
                    LOW VARIANCE            HIGH VARIANCE
                 (predictions consistent)  (predictions scattered)

  LOW BIAS         ..                          .    .
  (near bullseye)  .:.    ← ideal              .  .   .
                    ..                       .    .  .   .
                  tight cluster           scattered, but centered
                  ON the bullseye          around the bullseye

  HIGH BIAS          .  .                       .        .
  (systematically   .::.                    .       .
   off-target)       . .                        .    .      .
                  tight cluster,             scattered AND
                  but off-center             off-center — worst case
```
Remember it this way: underfitting is the move from top-left toward bottom-left — predictions get consistent, but consistently wrong. That's `gamma` too smooth, `C` too low, a tree too shallow, or `k` in kNN too large. Overfitting is the move toward the right side — predictions get scattered, chasing individual training points instead of the real pattern. That's `C` too high, `gamma` too high, a tree too deep, or `k` too small.

Every regularization knob in this entire doc — `max_depth`, `min_samples_leaf`, `num_leaves`, `C`, `gamma`, `k` — is turning this same dial between the four quadrants, just wearing a different name per algorithm. Once this picture is automatic, a new hyperparameter you've never seen before becomes an easy question: which direction does turning this move me on the dartboard? Not a fact you need to look up.

### Summary example
An RBF SVM with `C=100, gamma=10` scores 99% train accuracy but only 71% test accuracy. That's squarely the overfitting corner of the dartboard — both knobs cranked toward "chase every training point."

Back off to `C=1, gamma="scale"` (sklearn's data-informed default), and train accuracy drops to 93%, but test accuracy climbs to 90%. Smaller train/test gap, genuinely more useful model. That's the exact tradeoff the dartboard visual is describing.

The next section goes deeper on this same SVM and PCA material. Nothing sketched this time — every number is computed for real on actual fitted models and a real dataset, so it's verifiable, not just illustrative.

---

## SVM & PCA — Detailed, Pictorial

Everything below is computed, not sketched. A real `sklearn.svm.SVC` fitted on a small hand-checkable dataset. A real `np.linalg.eigh` eigendecomposition. A real pipeline on the actual UCI Wine dataset. Hover any box in a diagram for a one-line definition.

### SVM, step by step — what "maximum margin" actually means

<div class="formula">
Decision boundary:   w · x + b = 0
Margin width:         2 / ||w||
Classify a new point: sign(w · x + b)
</div>

A Support Vector Machine doesn't just find *a* line that separates two classes. It finds the line with the **widest possible margin** to the nearest point of either class. Those nearest points are the **support vectors** — the only training points that actually determine where the boundary sits. Every other point could be deleted and the boundary wouldn't move at all.

Here it is fitted for real, on 6 points: class +1 at (3,3), (4,3), (3,4); class −1 at (0,0), (1,0), (0,1). Fit with `SVC(kernel="linear", C=1000)`:

<figure class="fig" data-mlviz="svmmargin" id="mlviz-svmmargin"></figure>

Only 3 of the 6 points became support vectors: (1,0), (0,1), and (3,3). The other three — (0,0), (4,3), (3,4) — sit strictly farther from the boundary and contribute nothing to defining it. The real fitted boundary is `0.4001·x1 + 0.3999·x2 − 1.3999 = 0`, which is very close to `x1+x2=3.5` — matching the geometric midpoint intuition. The real margin width is **3.536**, computed as `2/||w||`, not estimated by eye.

### Why the kernel choice is the single biggest SVM decision

<figure class="fig" data-mlviz="svmkernel" id="mlviz-svmkernel"></figure>

Here's a real, dramatic difference on a real dataset: 200 points arranged as two concentric circles, 30% held out as test. A **linear** kernel can only cut with a straight line, so on genuinely circular data it does barely better than a coin flip: **53.3% test accuracy**. The **RBF kernel** implicitly lifts the data into a higher-dimensional space, where a flat cut *there* corresponds to a curved boundary *here*: **100% test accuracy**. Same data, same train/test split — only the kernel changed.

### What `C` actually controls — margin width vs. tolerance for violations

<figure class="fig" data-mlviz="svmc" id="mlviz-svmc"></figure>

`C` is a real, measurable trade-off, not a vague "regularization knob." Here it's fit at three real values on the same overlapping dataset. As C climbs from 0.01 to 1.0 to 1000:
- The margin width shrinks: 2.695 → 0.957 → 0.855.
- The number of support vectors shrinks: 118 → 31 → 27, out of 140 training points.

A small `C` tolerates more points inside or across the margin — wide margin, many support vectors, more regularized. A large `C` insists on getting almost every training point right — narrow margin, few support vectors, more prone to overfitting on noisy data.

<div class="callout"><span class="tag">Honest note</span>In this particular dataset, test accuracy stayed flat at 93.3% across all three C values — the classes are separated enough that C's effect on generalization didn't show up here. What did change, measurably, is exactly what C is defined to control: margin width and support-vector count. Don't let a flat accuracy number hide that C is still doing real, verifiable work.</div>

---

### PCA, step by step — finding the direction of maximum variance

<div class="formula">
1. Center the data:        X_c = X − mean(X)
2. Covariance matrix:       C = (1/(n−1)) · X_cᵀ · X_c
3. Eigendecompose:          C = V · Λ · Vᵀ    (V = eigenvectors, Λ = eigenvalues)
4. Project onto top-k eigenvectors (sorted by eigenvalue, descending)
</div>

PCA doesn't shrink data randomly. It finds the exact direction the data varies the most along — the top eigenvector of the covariance matrix — and that direction becomes the new first axis.

Two words worth pinning down if they're unfamiliar: the covariance matrix is a table of how strongly each pair of features moves together, and an eigenvector is a direction that matrix stretches without rotating. Both are worked through with small real numbers in `math-foundations-refresher.md`. Below, everything is computed on 10 real points, using `np.cov` and `np.linalg.eigh` — not estimated:

<figure class="fig" data-mlviz="pcaeig" id="mlviz-pcaeig"></figure>

The real covariance matrix here is `[[0.617, 0.615], [0.615, 0.717]]`. That large off-diagonal value, 0.615, is why the two features are so correlated — points move up-and-right together. The real eigenvalues are **1.284** (PC1) and **0.049** (PC2). PC1 alone captures **96.3%** of all the variance in the data; PC2 only adds 3.7%.

Projecting all 10 points onto PC1 alone — throwing away the second dimension entirely — gives a mean squared reconstruction error of just **0.044**. PC2 barely mattered to begin with, so dropping it costs almost nothing.

<figure class="fig" data-mlviz="pcascree" id="mlviz-pcascree"></figure>

### Combining PCA + SVM — a real pipeline, real dataset

<figure class="fig" data-mlviz="pcasvm" id="mlviz-pcasvm"></figure>

The real UCI Wine dataset: 178 samples, 13 real chemical-composition features, 3 wine cultivars.

1. An RBF-kernel SVM fit on all 13 scaled features gets **98.15% test accuracy**.
2. Compress down to just the top 2 principal components first — which capture only **54.9%** of the total variance.
3. Fit the exact same SVM on those 2 numbers instead of 13. It gets **96.30% test accuracy**.

That's barely worse, using 2 numbers instead of 13, and it needed fewer support vectors to do it — 33 versus 57.

This is the real, practical reason PCA and SVM get paired in practice. PCA buys you visualization (you can actually plot 2 dimensions) and speed, at a small, measurable, honestly-reported accuracy cost. It's a real trade you can quantify before making it, not a free lunch.

<div class="callout"><span class="tag">Where this shows up</span>SVM's max-margin idea generalizes to margin-based deep learning losses, like hinge loss and triplet loss. PCA's "find the direction of max variance" idea is the same math behind whitening, some anomaly-detection methods, and — at a much larger scale — the intuition behind why embedding spaces can be compressed without losing most of their useful structure.</div>

## Cluster 6 — K-Nearest Neighbors and Naive Bayes

### 1. How do you fit K-Nearest Neighbors, and why does feature scaling matter more here than for the tree models above?
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
knn_pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
knn_pipe.fit(X_train, y_train)
```
KNN classifies a point by finding its nearest neighbors, using raw Euclidean (or similar) distance. A feature measured in the thousands, like mileage, will completely dominate that distance calculation over a feature measured in single digits, like number of inspections — regardless of which one is actually more predictive.

Trees don't have this problem. They split one feature at a time and never compute cross-feature distance. For KNN, scaling isn't a minor tuning detail — it's a correctness requirement.

### 2. Scaling is required. How do you choose `k` (n_neighbors) itself — the same bias-variance dial as `C`, `gamma`, and `max_depth` above?
```python
from sklearn.model_selection import cross_val_score
for k in [3, 5, 7, 9, 15, 25]:
    pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
    scores = cross_val_score(pipe, X_train, y_train, cv=5)
    print(k, scores.mean())
```
A very small `k`, like 1, fits training data almost perfectly, but it's extremely sensitive to noise and outliers. That's high variance — the scattered-right side of Cluster 5's dartboard. A very large `k` smooths over real local structure and drifts toward just predicting the overall majority class. That's high bias — the top side of the dartboard.

Cross-validation should pick `k`, not intuition.

### 3. Moving away from distance-based methods entirely — how does Naive Bayes classify, and what's actually "naive" about it?
```python
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, y_train)
```
It assumes every feature is conditionally independent of every other feature, given the class. In real data that's almost never exactly true — features usually correlate with each other some amount.

Even so, the classifier often works well in practice, violated assumption and all. Here's why: it only needs the relative ranking of class probabilities to be right, not the exact probability values, and errors from the independence assumption often partially cancel out.

### Summary example
Take 20 features, several of them correlated with each other — directly violating Naive Bayes' independence assumption. `GaussianNB` still ranks the true class highest in most cases. Its probability estimates get distorted by the correlation, but the relative order of classes often survives that distortion anyway.

That's a genuinely different failure mode than KNN. KNN needs every feature properly scaled first, or the distance calculation itself becomes meaningless — no independence assumption involved at all.

## Cluster 7 — Combining Multiple Models

### 1. Every model above is trained separately. How do you combine several different model types into one stronger ensemble?
```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

stack = StackingClassifier(
    estimators=[("rf", RandomForestClassifier(n_estimators=200, random_state=42)),
                ("svm", SVC(probability=True, random_state=42))],
    final_estimator=LogisticRegression(),
    cv=5,
)
stack.fit(X_train, y_train)
```
The `final_estimator` — a "meta-model" — learns how much to trust each base model's predictions, potentially per region of feature space, instead of averaging them blindly. It can learn, for instance, that the SVM is more reliable on one subset of cases and the Random Forest on another.

`cv=5` matters specifically here. The base models predict via cross-validation internally, so the meta-model trains on genuinely out-of-fold predictions — not predictions the base models have already memorized. Training on in-sample base predictions would badly overstate the stack's real performance.

### 2. Stacking trains a whole extra model just to combine predictions. Is there a cheaper way?
```python
from sklearn.ensemble import VotingClassifier
voter = VotingClassifier(
    estimators=[("rf", RandomForestClassifier(random_state=42)), ("lr", LogisticRegression())],
    voting="soft",   # average predicted PROBABILITIES, not just majority-vote the hard predictions
)
voter.fit(X_train, y_train)
```
Hard voting only sees each model's final class prediction, throwing away its confidence. Soft voting averages the actual predicted probabilities instead, so a model that's 90% confident correctly influences the outcome more than one that's only 51% confident.

As long as the underlying models produce well-calibrated probabilities, soft voting typically edges out hard voting — at a fraction of stacking's complexity and training cost.

### Summary example
Combining a Random Forest and an SVM, two ways.

`VotingClassifier(voting="soft")` is a one-line, no-extra-training way to average their probability outputs. That's a reasonable default.

`StackingClassifier` instead trains a `LogisticRegression` meta-model on each base model's out-of-fold predictions. It can outperform soft voting specifically when the two base models are reliable in genuinely different regions of the feature space — the meta-model learns that pattern instead of averaging blindly everywhere.

## Cluster 8 — The Cheapest Sanity Check of All

### 1. Before reaching for anything above — trees, SVMs, ensembles — how do you quickly check if a simple model is already good enough?
```python
baseline_linear = LogisticRegression().fit(X_train, y_train)
baseline_tree = RandomForestClassifier(random_state=42).fit(X_train, y_train)
print("linear:", baseline_linear.score(X_test, y_test))
print("tree ensemble:", baseline_tree.score(X_test, y_test))
```
Just fit both and compare. If a plain linear model already gets close to a tuned tree ensemble's performance, that's valuable information — it suggests the true relationships in the data are mostly linear and additive, favoring the simpler, faster, more interpretable model for production, instead of defaulting to "the fancier model must be better" without checking.

This is the model-family version of `sklearn-practice.md`'s `DummyClassifier` baseline. Cheap, fast, and run first — before investing in anything more complex.

### Summary example
A Random Forest takes 40 minutes to tune and scores 91% test accuracy. A plain `LogisticRegression`, fit in under a second with no tuning, scores 89% on the same data.

That 2-point gap may not be worth the forest's added complexity, slower inference, and reduced interpretability in production. Running both cheaply up front is what makes that tradeoff visible, before committing to the more complex option by default.

---

## Practice Q&A (Self-Test)

**Q1. Why do linear regression coefficients need scaled inputs before you can compare their magnitudes as "importance"?**
A: A coefficient's size reflects both the feature's real effect and its raw scale — a feature measured in thousands gets a tiny coefficient even if it matters a lot, while one measured in single digits gets an inflated-looking coefficient. Standardizing features first removes the scale confound so magnitudes become comparable.

**Q2. What's the key practical difference between `Ridge(alpha=1.0)` and `Lasso(alpha=0.1)`, and when would you pick Lasso?**
A: Ridge (L2) shrinks all coefficients toward zero but rarely to exactly zero; Lasso (L1) can shrink coefficients to exactly zero, effectively performing feature selection. Pick Lasso when you suspect only a handful of features actually matter and want the model to identify which ones.

**Q3. If `clf.coef_[0]` for a feature is 0.69 in a fitted `LogisticRegression`, what does `np.exp(0.69) ≈ 2.0` actually mean?**
A: Logistic regression coefficients are in log-odds units, not probability units, so 0.69 does not mean "69% more likely." Exponentiating gives the odds ratio — here, the odds of the positive class roughly double (multiply by ~2.0) for a one-unit increase in that feature.

**Q4. Why can a `DecisionTreeClassifier(max_depth=3)` still overfit if `min_samples_leaf` is left unconstrained?**
A: `max_depth` only caps how many splits deep the tree can go, but `min_samples_leaf` prevents a leaf from being created based on just 1-2 data points, which is almost certainly noise rather than signal. Without constraining `min_samples_leaf`, a tree can still create tiny, unreliable leaves and overfit badly even at a modest depth.

**Q5. In `RandomForestClassifier(max_features="sqrt")`, what does `max_features` actually do, and why is it more important than just "building many trees"?**
A: `max_features="sqrt"` restricts each tree to only consider a random subset of features (sqrt of the total, by convention) at every split, forcing the trees to differ from each other even on the same data. Without this feature-level randomness, all trees would tend to make similar splits on the same dominant features, and the ensemble wouldn't reduce variance much — this decorrelation is what makes averaging the trees powerful.

**Q6. What is the fundamental difference between how a Random Forest and a gradient boosting model (like XGBoost) build their trees?**
A: A Random Forest builds many independent trees in parallel and averages them, which reduces variance. Gradient boosting builds trees sequentially, where each new tree is trained specifically to correct the previous ensemble's errors, which reduces bias — this is why boosting typically needs a low `learning_rate` and more trees, and is more prone to overfitting without regularization like `subsample`/`colsample_bytree`/`max_depth`.

**Q7. Why must `learning_rate` and `n_estimators` be tuned together in XGBoost rather than independently?**
A: A lower `learning_rate` needs more estimators to reach the same fit, since they trade off directly — a high learning rate with few trees converges fast but can overshoot and generalize worse, while a low learning rate with many trees is slower but each correction is gentler and typically generalizes better. The file calls this the single most important hyperparameter pair to tune jointly.

**Q8. What does `early_stopping_rounds=20` do during `model.fit(X_train, y_train, eval_set=[(X_test, y_test)])`, and why is it better than guessing `n_estimators`?**
A: It stops training automatically once the validation score hasn't improved for 20 consecutive rounds, rather than requiring you to guess the right fixed tree count. This directly implements "stop at the validation-loss minimum" using real held-out performance instead of an arbitrary `n_estimators` value.

**Q9. Why does K-Nearest Neighbors break "more than most models" without feature scaling?**
A: KNN classifies a point based on raw Euclidean (or similar) distance to its neighbors — a feature measured in the thousands (e.g., mileage) would completely dominate the distance calculation over a feature measured in single digits (e.g., inspection count), regardless of actual predictive value. For KNN this is called a correctness requirement, not a minor tuning detail, unlike for tree-based models.

**Q10. In a `StackingClassifier` with `cv=5`, why does the meta-model (`final_estimator`) need out-of-fold predictions from the base models rather than their in-sample predictions?**
A: Training the meta-model on predictions the base models already memorized (in-sample) would badly overstate the stack's real performance. `cv=5` makes the base models predict via cross-validation internally so the meta-model learns from genuinely out-of-fold predictions, which is what lets it learn to trust different base models in different regions of feature space.

<script>
(function(){
"use strict";
const esc=s=>String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
const txt=(x,y,cls,s,anch)=>`<text x="${x}" y="${y}" class="${cls}"${anch?` text-anchor="${anch}"`:""}>${s}</text>`;
const DEFS='<defs><marker id="mlah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="var(--muted)"/></marker></defs>';
const hit=(bx,by,bw,bh,tip,inner)=>`<g class="mlhit" tabindex="0" aria-label="${esc(tip)}" data-tip="${esc(tip)}"><rect class="hitbg" x="${bx}" y="${by}" width="${bw}" height="${bh}" rx="6" fill="transparent" stroke="transparent" stroke-width="2"/>${inner}</g>`;
const box=(x,y,w,h,cls,title,sub,sub2)=>`<rect class="${cls}" x="${x}" y="${y}" width="${w}" height="${h}" rx="6"/>`
  +(title?txt(x+w/2,y+h/2+(sub?-4:4),"vlab",title,"middle"):"")
  +(sub?txt(x+w/2,y+h/2+10,"vsm",sub,"middle"):"")
  +(sub2?txt(x+w/2,y+h/2+22,"vsm",sub2,"middle"):"");
const MLVIZ={};

MLVIZ.svmmargin={t:"Maximum margin — the real fitted hyperplane",
  hint:"real SVC(kernel='linear', C=1000) fit; 3 of 6 points became support vectors",w:900,h:420,
  cap:"Fitted boundary: 0.4001·x1 + 0.3999·x2 − 1.3999 = 0 (essentially x1+x2=3.5). Margin width = 2/‖w‖ = 3.536, computed, not estimated. Only (1,0), (0,1), and (3,3) are support vectors — deleting any OTHER point wouldn't move the boundary at all.",
  svg(){
    const ox=70,oy=380,scale=64;
    const mapX=v=>ox+v*scale, mapY=v=>oy-v*scale;
    let s='';
    for(let g=0;g<=5;g++){
      s+=`<line x1="${mapX(g)}" y1="${oy}" x2="${mapX(g)}" y2="20" stroke="var(--line)" stroke-width="0.6"/>`;
      s+=`<line x1="${ox}" y1="${mapY(g)}" x2="${ox+5*scale+20}" y2="${mapY(g)}" stroke="var(--line)" stroke-width="0.6"/>`;
      s+=txt(mapX(g),oy+16,"vsm",g,"middle");
    }
    s+=`<line x1="${ox}" y1="${oy}" x2="${ox+5*scale+20}" y2="${oy}" stroke="var(--muted)" stroke-width="1.2" marker-end="url(#mlah)"/>`;
    s+=`<line x1="${ox}" y1="${oy}" x2="${ox}" y2="10" stroke="var(--muted)" stroke-width="1.2" marker-end="url(#mlah)"/>`;
    s+=txt(ox+5*scale+26,oy+4,"vsm","x1");s+=txt(ox-8,16,"vsm","x2","end");
    // decision boundary line: x2=(1.3999-0.4001*x1)/0.3999
    const bx1=-0.4, bx2=5.3;
    const by1=(1.3999-0.4001*bx1)/0.3999, by2=(1.3999-0.4001*bx2)/0.3999;
    s+=hit(mapX(bx1)-4,mapY(Math.max(by1,by2))-4,mapX(bx2)-mapX(bx1)+8,mapY(Math.min(by1,by2))-mapY(Math.max(by1,by2))+8,
      "Decision boundary: 0.4001·x1+0.3999·x2−1.3999=0 — the real fitted separating line, equidistant from the nearest point of each class.",
      `<line x1="${mapX(bx1)}" y1="${mapY(by1)}" x2="${mapX(bx2)}" y2="${mapY(by2)}" stroke="var(--accent-hi)" stroke-width="2"/>`);
    // margin lines (offset along unit normal [0.7072,0.7054] by margin/2=1.768)
    const nx=0.7072,ny=0.7054,off=1.768;
    [1,-1].forEach(sign=>{
      const dx=sign*off*nx, dy=sign*off*ny;
      s+=`<line x1="${mapX(bx1+dx)}" y1="${mapY(by1+dy)}" x2="${mapX(bx2+dx)}" y2="${mapY(by2+dy)}" stroke="var(--muted)" stroke-width="1" stroke-dasharray="5 4"/>`;
    });
    const pos=[[3,3],[4,3],[3,4]], neg=[[0,0],[1,0],[0,1]];
    const svs=new Set(["1,0","0,1","3,3"]);
    pos.concat(neg).forEach(([px,py])=>{
      const key=px+","+py, isSV=svs.has(key), cls=pos.some(p=>p[0]===px&&p[1]===py)?"cellhot":"cell";
      const tip=isSV?`Support vector at (${px},${py}) — one of the 3 points that actually define the boundary and margin.`
                     :`(${px},${py}) — NOT a support vector. It could be deleted and the boundary/margin wouldn't change at all.`;
      s+=hit(mapX(px)-10,mapY(py)-10,20,20,tip,
        `<circle cx="${mapX(px)}" cy="${mapY(py)}" r="${isSV?9:6}" class="${cls}" stroke-width="${isSV?2.4:1.1}"/>`
        +(isSV?`<circle cx="${mapX(px)}" cy="${mapY(py)}" r="13" fill="none" stroke="var(--accent)" stroke-width="1.2" stroke-dasharray="2 2"/>`:""));
    });
    s+=txt(mapX(4.3),mapY(4.6),"vacc","margin width = 3.536","middle");
    return s;}};

MLVIZ.svmkernel={t:"Why kernel choice is the single biggest SVM decision",
  hint:"real SVC on 200-point concentric circles, 30% held-out test",w:900,h:340,
  cap:"Same data, same train/test split, only the kernel changes. Linear can only cut with a straight line — on genuinely circular data that's barely better than a coin flip. RBF implicitly lifts the data into a space where a flat cut corresponds to a curved boundary here.",
  svg(){
    let s="";
    function panel(px,title,titleCls,acc,desc){
      let inner=txt(px+190,30,titleCls,title,"middle");
      inner+=`<circle cx="${px+190}" cy="150" r="110" fill="none" stroke="var(--line)" stroke-width="1"/>`;
      inner+=`<circle cx="${px+190}" cy="150" r="45" class="cellhot"/>`;
      inner+=`<circle cx="${px+190}" cy="150" r="110" fill="none" stroke="var(--muted)" stroke-width="1" stroke-dasharray="3 3"/>`;
      inner+=txt(px+190,155,"vsm","inner class","middle");
      inner+=txt(px+190,270,"vsm","outer class","middle");
      inner+=txt(px+190,300,"vacc",`test accuracy: ${acc}`,"middle");
      return hit(px,10,380,310,desc,inner);}
    s+=panel(10,"Linear kernel — a straight cut","vbad","53.3%",
      "Linear kernel on concentric circles: 53.3% test accuracy — barely better than a coin flip on 2 classes, because a straight line fundamentally cannot separate an inner circle from an outer ring.");
    s+=panel(410,"RBF kernel — a curved cut","vok","100.0%",
      "RBF kernel, same data, same split: 100.0% test accuracy. RBF implicitly projects into a higher-dimensional space where the circle becomes linearly separable, then maps that flat cut back to a curved boundary here.");
    return s;}};

MLVIZ.svmc={t:"What C actually controls — margin width vs. tolerance for violations",
  hint:"real SVC(kernel='linear') fits at 3 real C values, same overlapping dataset",w:900,h:320,
  cap:"As C climbs, margin shrinks and fewer points are needed to define it. Test accuracy stayed flat (93.3%) on this dataset — a reminder that C's REAL, guaranteed effect is on margin width and support-vector count, not always on accuracy.",
  svg(){
    let s="";
    function panel(px,title,titleCls,margin,nsv,desc){
      const maxH=180,h=(margin/2.9)*maxH,by=260-h;
      let inner=txt(px+130,30,titleCls,title,"middle");
      inner+=`<rect x="${px+70}" y="${by}" width="120" height="${h}" rx="4" class="cellhot"/>`;
      inner+=txt(px+130,by-10,"vacc",`margin = ${margin}`,"middle");
      inner+=txt(px+130,278,"vsm",`${nsv} support vectors`,"middle");
      return hit(px,10,260,290,desc,inner);}
    s+=panel(10,"C = 0.01 (tolerant)","vok",2.695,"118 / 140",
      "C=0.01: wide margin (2.695), tolerates many points inside/across it — 118 of 140 training points became support vectors. Heavily regularized.");
    s+=panel(320,"C = 1.0","vlab",0.957,"31 / 140",
      "C=1.0: margin narrows to 0.957, only 31 of 140 points are support vectors — a middle ground.");
    s+=panel(630,"C = 1000 (strict)","vbad",0.855,"27 / 140",
      "C=1000: narrowest margin (0.855), fewest support vectors (27 of 140) — the model insists on getting almost every training point right, at real risk of overfitting to noise.");
    return s;}};

MLVIZ.pcaeig={t:"Covariance → eigenvectors — finding the direction of maximum variance",
  hint:"real np.cov + np.linalg.eigh on 10 points; eigenvalues 1.284 / 0.049",w:900,h:420,
  cap:"Real covariance matrix [[0.617,0.615],[0.615,0.717]] — the large off-diagonal (0.615) is why the cloud is a diagonal streak, not a circle. PC1 (long arrow) points along that streak; PC2 (short arrow) is perpendicular, capturing what little variance PC1 misses.",
  svg(){
    const pts=[[2.5,2.4],[0.5,0.7],[2.2,2.9],[1.9,2.2],[3.1,3.0],[2.3,2.7],[2.0,1.6],[1.0,1.1],[1.5,1.6],[1.1,0.9]];
    const mean=[1.81,1.91];
    const ox=60,oy=380,scale=90;
    const mapX=v=>ox+v*scale, mapY=v=>oy-v*scale;
    let s="";
    for(let g=0;g<=3.5;g+=0.5){
      s+=`<line x1="${mapX(g)}" y1="${oy}" x2="${mapX(g)}" y2="20" stroke="var(--line)" stroke-width="0.5"/>`;
    }
    s+=`<line x1="${ox}" y1="${oy}" x2="${ox+3.5*scale}" y2="${oy}" stroke="var(--muted)" stroke-width="1.1" marker-end="url(#mlah)"/>`;
    s+=`<line x1="${ox}" y1="${oy}" x2="${ox}" y2="10" stroke="var(--muted)" stroke-width="1.1" marker-end="url(#mlah)"/>`;
    pts.forEach(([x,y])=>{
      s+=hit(mapX(x)-7,mapY(y)-7,14,14,`Point (${x},${y})`,`<circle cx="${mapX(x)}" cy="${mapY(y)}" r="5" class="cell"/>`);
    });
    s+=hit(mapX(mean[0])-6,mapY(mean[1])-6,12,12,`Mean = (${mean[0]},${mean[1]}) — every point is centered around this before eigendecomposition.`,
      `<circle cx="${mapX(mean[0])}" cy="${mapY(mean[1])}" r="5" class="cellblk"/>`);
    // PC1 direction [0.6779,0.7352], eigenvalue 1.284 -> length scaled by sqrt(eigenvalue)
    const pc1=[0.6779,0.7352], pc2=[-0.7352,0.6779];
    const len1=Math.sqrt(1.284)*1.6, len2=Math.sqrt(0.049)*1.6;
    s+=hit(mapX(mean[0]-pc1[0]*len1)-6,mapY(mean[1]+pc1[1]*len1)-6,
      Math.abs(mapX(mean[0]+pc1[0]*len1)-mapX(mean[0]-pc1[0]*len1))+12,
      Math.abs(mapY(mean[1]-pc1[1]*len1)-mapY(mean[1]+pc1[1]*len1))+12,
      "PC1: eigenvalue=1.284, direction=[0.678,0.735] — captures 96.3% of total variance. This is the SINGLE direction the data varies along the most.",
      `<line x1="${mapX(mean[0]-pc1[0]*len1)}" y1="${mapY(mean[1]-pc1[1]*len1)}" x2="${mapX(mean[0]+pc1[0]*len1)}" y2="${mapY(mean[1]+pc1[1]*len1)}" stroke="var(--accent-hi)" stroke-width="2.4" marker-end="url(#mlah)"/>`);
    s+=hit(mapX(mean[0]-pc2[0]*len2)-6,mapY(mean[1]+pc2[1]*len2)-6,
      Math.abs(mapX(mean[0]+pc2[0]*len2)-mapX(mean[0]-pc2[0]*len2))+12,
      Math.abs(mapY(mean[1]-pc2[1]*len2)-mapY(mean[1]+pc2[1]*len2))+12,
      "PC2: eigenvalue=0.049, direction=[−0.735,0.678] — perpendicular to PC1, captures only 3.7% of total variance.",
      `<line x1="${mapX(mean[0]-pc2[0]*len2)}" y1="${mapY(mean[1]-pc2[1]*len2)}" x2="${mapX(mean[0]+pc2[0]*len2)}" y2="${mapY(mean[1]+pc2[1]*len2)}" stroke="var(--muted)" stroke-width="1.6" marker-end="url(#mlah)"/>`);
    s+=txt(mapX(0.3),40,"vsm","x2");s+=txt(mapX(3.3),oy+16,"vsm","x1");
    return s;}};

MLVIZ.pcascree={t:"Variance explained — the scree plot",
  hint:"real eigenvalues 1.284 and 0.049, normalized to percent of total",w:600,h:260,
  cap:"PC1 alone captures 96.3% of all variance; PC2 adds only 3.7% more. Dropping PC2 entirely (projecting onto PC1 only) gives a mean squared reconstruction error of just 0.044 — the lost dimension barely mattered.",
  svg(){
    let s="";
    const bars=[["PC1",96.3,"cellhot"],["PC2",3.7,"cell"]];
    const base=220,maxH=160;
    s+=`<line x1="40" y1="${base}" x2="560" y2="${base}" stroke="var(--muted)" stroke-width="1"/>`;
    bars.forEach((b,i)=>{
      const x=100+i*220,h=(b[1]/100)*maxH,by=base-h;
      s+=hit(x,by-20,140,h+40,`${b[0]}: ${b[1]}% of total variance.`,
        `<rect x="${x}" y="${by}" width="140" height="${h}" rx="4" class="${b[2]}"/>`
        +txt(x+70,by-8,"vacc",b[1]+"%","middle")+txt(x+70,base+18,"vlab",b[0],"middle"));
    });
    s+=txt(300,30,"vsm","reconstruction error using PC1 only: 0.044 (very small)","middle");
    return s;}};

MLVIZ.pcasvm={t:"A real pipeline — 13 features → PCA(2) → SVM",
  hint:"real UCI Wine dataset, 178 samples, 3 classes; real sklearn Pipeline",w:900,h:280,
  cap:"PCA(2) keeps only 54.9% of the original variance, yet the SVM trained on just those 2 numbers reaches 96.30% test accuracy vs. 98.15% using all 13 features — a small, honestly-measured cost for a 6.5x smaller input and fewer support vectors (33 vs 57).",
  svg(){
    let s="";
    s+=hit(10,90,150,90,"Real UCI Wine dataset: 178 samples, 13 real chemical-composition features, 3 cultivars.",
      box(10,90,150,90,"bigbox","Wine dataset","178 samples","13 features"));
    s+=`<line x1="160" y1="135" x2="220" y2="135" stroke="var(--muted)" stroke-width="1.3" marker-end="url(#mlah)"/>`;
    s+=hit(220,20,180,90,"SVM (RBF) fit directly on all 13 scaled features: 98.15% test accuracy, 57 support vectors.",
      box(220,20,180,90,"bigbox","SVM on all 13 features","test acc: 98.15%","57 support vectors"));
    s+=`<line x1="160" y1="135" x2="220" y2="220" stroke="var(--muted)" stroke-width="1.3" marker-end="url(#mlah)"/>`;
    s+=hit(220,175,180,90,"PCA reduces 13 features to 2 principal components, keeping only 54.9% of the total variance (35.7%+19.2%).",
      box(220,175,180,90,"cellhot","PCA → 2 components","54.9% variance kept","(35.7% + 19.2%)"));
    s+=`<line x1="400" y1="220" x2="460" y2="220" stroke="var(--muted)" stroke-width="1.3" marker-end="url(#mlah)"/>`;
    s+=hit(460,175,180,90,"Same SVM (RBF), now on just the 2 PCA features: 96.30% test accuracy, 33 support vectors — barely worse than using all 13 features.",
      box(460,175,180,90,"bigbox","SVM on PCA(2)","test acc: 96.30%","33 support vectors"));
    s+=txt(450,30,"vsm","full-feature path (top) vs. PCA-reduced path (bottom) — same SVM, same data split","middle");
    return s;}};

const mlvtip=document.createElement("div");mlvtip.id="mlvtip";document.body.appendChild(mlvtip);
document.querySelectorAll("figure[data-mlviz]").forEach(f=>{
  const d=MLVIZ[f.dataset.mlviz];if(!d)return;
  f.innerHTML=`<div class="vhead"><span class="vtitle">${d.t}</span><span class="vhint">${d.hint||""}</span></div>`
    +`<svg viewBox="0 0 ${d.w} ${d.h}" width="${d.w}" style="max-width:100%;height:auto;display:block" role="img" aria-label="${esc(d.t)}">${DEFS}${d.svg()}</svg>`
    +`<figcaption>${d.cap}</figcaption>`;
});
document.addEventListener("mousemove",e=>{
  const h=e.target.closest?e.target.closest("g.mlhit"):null;
  if(h&&h.dataset.tip){
    mlvtip.textContent=h.dataset.tip;mlvtip.style.display="block";
    const r=mlvtip.getBoundingClientRect();
    let x=e.clientX+14,y=e.clientY+18;
    if(x+r.width>innerWidth-8)x=innerWidth-r.width-8;
    if(y+r.height>innerHeight-8)y=e.clientY-r.height-10;
    mlvtip.style.left=x+"px";mlvtip.style.top=y+"px";
  }else{mlvtip.style.display="none";}
});
})();
</script>

---

## Video-Sourced Practice MCQs

A second practice set for Classical ML Models Practice, built the same way as this hub's NCA-GENL community bank. The topics were checked against a real YouTube interview-prep video for this subject, then written up here as original multiple-choice questions. The source video mostly asked these as open-ended questions — the wrong-answer options and their explanations below are original, written to match this hub's "explain every option" convention, not copied from the video. Click an answer, check it, and use "ask about this question" for anything that needs more explanation.

<script type="application/json" class="topic-quiz-data" data-title="Classical ML Models Practice">
[
  {
    "d": "Learning Paradigms",
    "q": "What is the main motivation for using machine learning techniques at all?",
    "o": [
      "To reduce the cost of storing data",
      "To automate decision-making processes based on patterns learned from data",
      "To make decisions without needing any data",
      "To manually control every step of data processing"
    ],
    "a": [
      1
    ],
    "e": "\"Without any data\" is a contradiction -- machine learning is fundamentally a data-driven approach; there's no ML technique that operates with zero data input. Manual control of every processing step is the OPPOSITE of what ML is for -- ML specifically replaces hand-coded, manually-controlled rules with learned patterns. Reducing storage cost isn't a goal of the modeling technique itself -- that's a data-infrastructure concern, unrelated to why you'd choose ML over a rule-based system. The core motivation is automation: instead of a human hand-writing every decision rule, ML lets a model learn the decision-making logic directly from data patterns, then apply that learned logic to new, unseen cases automatically."
  },
  {
    "d": "Learning Paradigms",
    "q": "A model is trained using input data that is explicitly paired with correct answers (labels). Which learning paradigm is this?",
    "o": [
      "Semi-random learning (not a real term, included as a distractor)",
      "Supervised learning",
      "Unsupervised learning",
      "Reinforcement learning"
    ],
    "a": [
      1
    ],
    "e": "Reinforcement learning doesn't use labeled input-output pairs at all -- it learns from reward/penalty signals received through interacting with an environment, a fundamentally different feedback structure. Unsupervised learning specifically means there are NO labels provided -- the model finds structure/patterns on its own, the opposite of this scenario. \"Semi-random learning\" isn't a real machine learning paradigm -- it's a made-up distractor to test whether you're recognizing real terminology versus a plausible-sounding fake. Training on data paired with correct answers is the textbook definition of supervised learning -- the model learns a mapping from inputs to the known correct outputs it was shown during training."
  },
  {
    "d": "Learning Paradigms",
    "q": "When would you reach for UNSUPERVISED learning specifically?",
    "o": [
      "When you need the model to learn from trial-and-error rewards",
      "When you have labeled data and a clear prediction target",
      "When you want to find patterns, groupings, or structure in data that has NO labels",
      "When you have no data available at all"
    ],
    "a": [
      2
    ],
    "e": "No data at all rules out any ML approach, supervised or unsupervised alike -- you always need some data to learn from. Labeled data with a clear prediction target is precisely the supervised learning scenario, not unsupervised -- having labels is what unsupervised learning explicitly lacks. Trial-and-error rewards describes reinforcement learning's feedback mechanism, a third, distinct paradigm. Unsupervised learning is specifically for situations with unlabeled data, where the goal is discovering hidden structure -- like grouping similar data points together (clustering) or finding lower-dimensional patterns -- without ever being told what the 'correct' grouping or output should be."
  },
  {
    "d": "Learning Paradigms",
    "q": "Reinforcement learning is most naturally described as solving which kind of problem?",
    "o": [
      "None of these -- RL doesn't solve any well-defined problem type",
      "Straightforward classification of independent, unrelated data points",
      "Simple time-series forecasting with no decision component",
      "Sequential decision-making, where one decision affects the state the next decision is made from, guided by rewards and penalties"
    ],
    "a": [
      3
    ],
    "e": "Classification of independent points describes standard supervised learning -- each prediction there doesn't influence the input to the NEXT prediction, unlike RL where actions change the environment's state. Time-series forecasting can overlap with supervised or even unsupervised techniques, but it lacks RL's core ingredient: an AGENT taking ACTIONS that change what happens next, evaluated via reward/penalty. Dismissing RL as solving no defined problem is simply inaccurate -- it has a well-established formal framework (agent, environment, state, action, reward). RL's defining shape is sequential decision-making: an agent takes an action, that action changes the environment's state, which affects the NEXT decision, and the agent learns a policy that maximizes cumulative reward over that whole sequence -- fundamentally different from single, independent predictions."
  },
  {
    "d": "Model Behavior",
    "q": "What is \"overfitting\" in machine learning?",
    "o": [
      "A model that performs well on training data but poorly on new, unseen (test/validation) data",
      "A model that performs poorly on both training and test data",
      "A model that uses too few parameters to capture the data's patterns",
      "A model that trains extremely quickly"
    ],
    "a": [
      0
    ],
    "e": "Poor performance on BOTH training and test data describes underfitting, not overfitting -- an underfit model hasn't even learned the training patterns well, let alone generalized them. Training speed is unrelated to overfitting entirely -- a model can overfit slowly or underfit quickly; speed doesn't indicate which failure mode is happening. Too few parameters is again underfitting's territory -- an overly simple model lacks the capacity to fit the data at all, whereas overfitting usually involves too MUCH capacity relative to the data available. Overfitting specifically means the model has essentially memorized the training set's quirks and noise rather than learning generalizable patterns -- so it looks great on training data but its performance drops noticeably on data it hasn't seen before."
  },
  {
    "d": "Model Behavior",
    "q": "Which technique is specifically used to improve a model's ability to generalize (perform well on new, unseen data)?",
    "o": [
      "Feature deletion with no selection criteria",
      "Reducing the size of the training set",
      "Regularization (e.g. L1/L2 penalties)",
      "Deliberately overfitting the model further"
    ],
    "a": [
      2
    ],
    "e": "Deliberately overfitting further is the exact opposite of what improves generalization -- it would make the training-vs-test performance gap worse, not better. Randomly deleting features with no principled selection criteria could remove genuinely useful signal just as easily as noise -- it's not a reliable generalization technique on its own. Shrinking the training set generally HURTS generalization -- less data typically means the model has fewer examples to learn robust, generalizable patterns from, the opposite of the goal. Regularization techniques (like L1/Lasso or L2/Ridge penalties, or dropout in neural networks) work by discouraging the model from fitting noise -- penalizing overly large or overly specific weight patterns -- which directly improves how well the model generalizes to new data."
  },
  {
    "d": "Algorithm Selection",
    "q": "Of K-means clustering, linear regression, DBSCAN, and PCA, which is the one example of a SUPERVISED learning algorithm?",
    "o": [
      "Principal Component Analysis (PCA)",
      "K-means clustering",
      "DBSCAN",
      "Linear regression"
    ],
    "a": [
      3
    ],
    "e": "K-means clustering is unsupervised -- it groups data points into clusters using only the data's structure, with no labeled 'correct group' provided. DBSCAN is also unsupervised -- it's a density-based clustering algorithm, grouping points by how densely packed they are, again with no labels involved. PCA is unsupervised too -- it's a dimensionality-reduction technique that finds directions of maximum variance in the data itself, not a labeled target. Linear regression is the one supervised algorithm in this list: it's trained on labeled (input, correct-output) pairs to learn a mapping that predicts a continuous target value -- the defining trait of supervised learning that the other three lack."
  },
  {
    "d": "Algorithm Selection",
    "q": "Which technique would you use specifically for dimensionality reduction?",
    "o": [
      "Principal Component Analysis (PCA)",
      "Naive Bayes",
      "Logistic regression",
      "Random forest"
    ],
    "a": [
      0
    ],
    "e": "Naive Bayes is a classification algorithm based on Bayes' theorem with an independence assumption between features -- it predicts a class label, it doesn't reduce feature dimensions. Logistic regression is also a classification technique (despite the word 'regression' in its name) -- it predicts a class probability, unrelated to reducing the number of features. Random forest is an ensemble of decision trees used for classification or regression -- again, a predictive model, not a dimension-reduction tool. PCA is specifically designed for dimensionality reduction: it projects high-dimensional data onto a smaller number of new axes (principal components) that capture the most variance, which is exactly the tool for this job among the four options."
  }
]
</script>
<div class="topic-quiz-mount"></div>
