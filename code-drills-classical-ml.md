# Code Drills — Tier 2: Classical ML (scikit-learn)

Continues `code-drills-numpy-pandas.md`. This is the direct answer to "how do you train and evaluate a RandomForest" — plus everything around it that makes a model real: splitting, scaling, tuning, evaluating, saving. Terser companion to `ml-models-practice.md`'s deeper narrative; `module-cheatsheet.md` has the same calls as a flat lookup table. All snippets verified against installed scikit-learn 1.4.2.

---

## Cluster 1 — Train & Evaluate a RandomForest

> 🔗 **Theory:** [Classical ML Models Practice — Ensembles](/topic/practice-ml-models#cluster-3-ensembles-bagging-vs-boosting)

**1. Load a built-in toy dataset to practice on.**
```python
from sklearn.datasets import load_iris
data = load_iris()
X, y = data.data, data.target    # X: (150, 4) features, y: (150,) class labels 0/1/2
```

**2. Split data into train and test sets.**
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42     # random_state fixes the shuffle -> reproducible split
)
```

**3. Train a RandomForestClassifier — the core ask, in three lines.**
```python
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)     # fit() does all the training — builds 200 decision trees on bootstrapped samples
```

**4. Make predictions on unseen data.**
```python
y_pred = clf.predict(X_test)
```

**5. Evaluate accuracy — the simplest metric.**
```python
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)    # fraction of predictions that exactly matched y_test
```

**6. Get precision/recall/F1 per class in one call.**
```python
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
# precision: of predicted-positive, how many were right | recall: of actual-positive, how many were caught
```

**7. Build a confusion matrix to see exactly what's being confused with what.**
```python
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)
# rows = actual class, columns = predicted class; off-diagonal cells are the mistakes
```

**8. Inspect which features the RandomForest relied on most.**
```python
clf.feature_importances_    # array of scores, one per input feature, summing to 1.0
dict(zip(data.feature_names, clf.feature_importances_))   # paired with human-readable names
```

**9. Train a RandomForestRegressor for a continuous target, and evaluate it differently.**
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

reg = RandomForestRegressor(n_estimators=200, random_state=42)
reg.fit(X_train, y_train)              # y_train here is continuous, not class labels
preds = reg.predict(X_test)
mean_squared_error(y_test, preds)       # lower is better, in squared target units
r2_score(y_test, preds)                  # 1.0 = perfect, 0.0 = no better than predicting the mean
```

## Cluster 2 — Pipelines, Cross-Validation & Hyperparameter Search

> 🔗 **Theory:** [scikit-learn Practice — Cross-Validation and Hyperparameter Tuning](/topic/practice-sklearn#cluster-2-cross-validation-and-hyperparameter-tuning)

**10. Standardize features before a model that's scale-sensitive (RandomForest itself doesn't need this — logistic regression, SVM, and KNN do).**
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit_transform on TRAIN: learns mean/std AND applies them
X_test_scaled = scaler.transform(X_test)          # transform only on TEST: reuses train's mean/std, no re-fitting
```

**11. Chain preprocessing and a model into one `Pipeline` object.**
```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000)),
])
pipe.fit(X_train, y_train)     # scaling and fitting happen together, in the right order, no leakage risk
pipe.predict(X_test)             # test data flows through the SAME fitted scaler automatically
```

**12. Get a more reliable accuracy estimate with k-fold cross-validation.**
```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(RandomForestClassifier(random_state=42), X, y, cv=5)
scores.mean(), scores.std()    # average +/- spread across 5 different train/test splits, not just one
```

**13. Tune RandomForest hyperparameters with an exhaustive grid search.**
```python
from sklearn.model_selection import GridSearchCV
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 5, 10],
}
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
grid.fit(X_train, y_train)
grid.best_params_    # the combination that scored highest across the 5 folds
grid.best_score_      # that combination's cross-validated score
```

**14. Tune with a random sample of the grid instead — faster when the grid is large.**
```python
from sklearn.model_selection import RandomizedSearchCV
param_dist = {"n_estimators": range(50, 500), "max_depth": [None, 5, 10, 20]}
search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42), param_dist, n_iter=20, cv=5, random_state=42
)
search.fit(X_train, y_train)   # only tries 20 random combinations, not every single one
```

**15. Train a simple linear baseline before reaching for anything fancier.**
```python
from sklearn.linear_model import LogisticRegression
baseline = LogisticRegression(max_iter=1000)
baseline.fit(X_train, y_train)
baseline.score(X_test, y_test)   # .score() on a classifier = accuracy, shortcut for accuracy_score
```

**16. Compare several models in a loop instead of copy-pasting the same 3 lines.**
```python
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

models = {
    "logreg": LogisticRegression(max_iter=1000),
    "rf": RandomForestClassifier(random_state=42),
    "svm": SVC(),
    "knn": KNeighborsClassifier(),
}
for name, model in models.items():
    model.fit(X_train, y_train)
    print(name, accuracy_score(y_test, model.predict(X_test)))
```

## Cluster 3 — Imbalance, Persistence & Preprocessing

> 🔗 **Theory:** [scikit-learn Practice — Handling Class Imbalance](/topic/practice-sklearn#cluster-4-handling-class-imbalance)

**17. Handle imbalanced classes without resampling — reweight the loss instead.**
```python
clf = RandomForestClassifier(class_weight="balanced", random_state=42)
# "balanced": automatically weights classes inversely proportional to their frequency —
# the minority class's mistakes count for more during training
```

**18. Save a trained model to disk, and load it back later.**
```python
import joblib
joblib.dump(clf, "model.joblib")
loaded = joblib.load("model.joblib")
loaded.predict(X_test)    # works identically to the original `clf` — no retraining needed
```

**19. Get predicted probabilities, not just the final class.**
```python
clf.predict_proba(X_test)    # shape (n_samples, n_classes) — each row sums to 1.0
clf.predict_proba(X_test)[:, 1]   # probability of the "positive" class, in binary classification
```

**20. Compute ROC-AUC — a threshold-independent way to score a binary classifier.**
```python
from sklearn.metrics import roc_auc_score
proba = clf.predict_proba(X_test)[:, 1]
roc_auc_score(y_test, proba)   # 1.0 = perfect ranking, 0.5 = no better than random guessing
```

**21. One-hot encode a categorical feature before feeding it to a model.**
```python
from sklearn.preprocessing import OneHotEncoder
import numpy as np
cats = np.array([["red"], ["blue"], ["red"], ["green"]])
OneHotEncoder(sparse_output=False).fit_transform(cats)
# one binary column per category — models can't use "red"/"blue" as text directly
```

**22. Fill missing values before training (most models can't handle NaN).**
```python
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="mean")     # or "median", "most_frequent", "constant"
X_filled = imputer.fit_transform(X_train)
```

**23. Apply DIFFERENT preprocessing to numeric vs. categorical columns in one step.**
```python
from sklearn.compose import ColumnTransformer
numeric_cols, cat_cols = ["age", "income"], ["city"]
preprocess = ColumnTransformer([
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(), cat_cols),
])
full_pipe = Pipeline([("prep", preprocess), ("clf", RandomForestClassifier())])
```

## Cluster 4 — Unsupervised Learning & Diagnosing Over/Underfitting

> 🔗 **Theory:** [scikit-learn Practice — Unsupervised: Clustering and Dimensionality Reduction](/topic/practice-sklearn#cluster-7-unsupervised-clustering-and-dimensionality-reduction)

**24. Cluster unlabeled data with K-means.**
```python
from sklearn.cluster import KMeans
km = KMeans(n_clusters=3, n_init="auto", random_state=42)
labels = km.fit_predict(X)    # no y at all — unsupervised, groups similar points together
```

**25. Reduce dimensionality with PCA (e.g. for visualization).**
```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)     # compress 4 features down to 2, keeping the most variance
pca.explained_variance_ratio_    # how much of the original variance each of the 2 components captured
```

**26. Spot overfitting by comparing train score to test score.**
```python
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)
train_acc = clf.score(X_train, y_train)
test_acc = clf.score(X_test, y_test)
# train_acc near 1.0 but test_acc much lower -> overfitting: memorized train, didn't generalize
```

**27. See a hyperparameter's effect directly by sweeping it.**
```python
for depth in [2, 5, 10, None]:
    clf = RandomForestClassifier(max_depth=depth, random_state=42)
    clf.fit(X_train, y_train)
    print(depth, clf.score(X_train, y_train), clf.score(X_test, y_test))
# shallow max_depth -> underfits (both scores low); max_depth=None -> trees grow until pure, risk of overfitting
```

**28. Know RandomForest's main tuning knobs, at a glance.**
```python
RandomForestClassifier(
    n_estimators=200,     # more trees = more stable, diminishing returns past a few hundred, slower to train
    max_depth=10,          # caps how deep each tree grows — the main overfitting control
    min_samples_leaf=1,     # raise this (e.g. 5) to force simpler, less overfit trees
    max_features="sqrt",    # how many features each split considers — adds randomness, decorrelates trees
    class_weight=None,       # set "balanced" for imbalanced classes (drill #17)
    random_state=42,         # fixes randomness for reproducibility
)
```

---

**Next in the Code Drills tier:** `code-drills-deep-learning.md` (PyTorch tensors, training loops, CNNs, and LSTM hyperparameter tuning).
