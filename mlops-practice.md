# MLOps Practice — Experiment Tracking, Versioning, and Model Registries

Everything else on this hub is about building a good model. MLOps is about not losing track of what you built, being able to reproduce it, and handing it off cleanly to production. This is the part that quietly separates "I trained a model once" from "I can tell you exactly which model is in production, what data trained it, and how to roll it back."

**Visual + memory hook — every tool in this doc is a labeled stop on the same loop, not a separate topic to memorize:**
```
   ┌──▶ TRACK (MLflow/W&B) ──▶ REGISTER (model registry) ──▶ DEPLOY
   │      log params/                promote to                 │
   │      metrics/artifacts          Staging→Production          │
   │      per run                                                ▼
   │                                                         MONITOR
   │                                                    (production-ml-practice.md:
   │                                                     drift, latency, rollback)
   │                                                                │
   └───────────────── RETRAIN on new/versioned data ◀───────────────┘
                       (DVC tracks which data version
                        trained which registered model)
```
**Remember it as a loop with exactly one entry and one exit label per stage** — track, register, deploy, monitor, retrain, back to track. Every section below this point is really just answering "what tool sits at this one stop on the loop, and what does it log/version/promote there." When a new MLOps tool name comes up that isn't in this doc, the fastest way to place it is asking which single stop on this loop it belongs to, rather than treating it as an unrelated new concept.

## Built as a chain: walking the loop above, one stop at a time

### 1. Why do you need experiment TRACKING at all — isn't a spreadsheet enough?
After the twentieth training run (different learning rate, different data version, different architecture tweak), "which run was the good one, and what exact settings produced it?" becomes genuinely unanswerable from memory or a spreadsheet you forgot to update. Experiment tracking tools log this automatically, every run, with zero discipline required from you in the moment.

### 2. Given tracking is needed, what does the open-source standard tool for it actually log, concretely?
**MLflow**:
```python
import mlflow

with mlflow.start_run():
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("model_type", "resnet18")
    mlflow.log_metric("val_accuracy", 0.943)
    mlflow.log_metric("val_loss", 0.187)
    mlflow.pytorch.log_model(model, "model")
```

Every run logs **parameters** (the settings you chose), **metrics** (the results you got, can log per-epoch too), and **artifacts** (the actual model file, plots, anything else worth keeping). MLflow's UI then lets you sort/filter/compare hundreds of runs side by side — "show me every run with `val_accuracy > 0.93`, sorted by `learning_rate`."

### 3. Given MLflow logs the same three things every run, when would you reach for Weights & Biases instead?
**Weights & Biases (W&B)** — the same idea, hosted and more visual:
```python
import wandb

wandb.init(project="fraud-detection", config={"lr": 0.001, "batch_size": 32})
for epoch in range(epochs):
    wandb.log({"train_loss": train_loss, "val_accuracy": val_acc})
```
Functionally similar to MLflow (params/metrics/artifacts, comparison dashboards), with real-time live-updating charts during training and easier team-wide sharing since it's cloud-hosted by default. The two aren't mutually exclusive — MLflow for anything staying entirely in-house, W&B where a hosted dashboard and easy collaboration matter more.

### 4. Given hundreds of tracked runs now exist (questions 2-3), how do you move from "the best run so far" to "the model that's actually LIVE," at the REGISTER stop?
A **model registry** is a versioned, centralized store of trained models with a lifecycle status attached: `Staging` → `Production` → `Archived`. Instead of a model being "whatever file someone last copied to the server," each version is tracked, and promoting a new model to `Production` (or rolling back to a previous version) is a deliberate, auditable action:

```python
mlflow.register_model("runs:/<run_id>/model", "fraud-detector")
# later, promote a specific version
client.transition_model_version_stage("fraud-detector", version=3, stage="Production")
```

The core question a registry answers on demand: "what model is live right now, what version, trained on what data, and can we get back to the previous one in one step?" Without one, that question requires archaeology.

### 5. Given a registry now tracks WHICH model is live (question 4), how do you also track WHICH DATA trained it — since a 10 GB dataset can't just live in Git?
Git is built for text diffs; a 2 GB model checkpoint or a 10 GB training dataset doesn't diff meaningfully and will bloat a git repo into unusability. **DVC (Data Version Control)** solves this the same conceptual way Git LFS does: the actual large file lives in external storage (S3, GCS, a shared drive), and DVC commits a small pointer file to Git instead.

```bash
dvc add data/training_set.csv       # stores the real file externally, creates a small .dvc pointer
git add data/training_set.csv.dvc   # git tracks only the lightweight pointer
git commit -m "add v2 training data"
dvc push                            # uploads the actual data to remote storage
```

Now `git log` on the `.dvc` file gives you a real history of *which data version* trained which model — the same reproducibility Git gives code, extended to the large files Git can't handle directly, and exactly the "RETRAIN on new/versioned data" arrow feeding back into TRACK in the loop diagram above.

### 6. Given tracked runs, a registry, and versioned data all now exist, what has to happen automatically at the DEPLOY stop before a new model is trusted?
**CI/CD for ML** — regular CI/CD asks "does the code still work?" (tests pass, it builds). ML CI/CD has to additionally ask "does the *model* still work?" — a code change can pass every unit test and still silently degrade model quality. A reasonable ML pipeline, triggered on a PR or a merge:
1. Standard checks — lint, unit tests, type checks (nothing ML-specific yet).
2. **Data validation** — schema check on any new/changed data (right columns, right types, no unexpected nulls) before it's allowed to feed a training job.
3. **Retrain or evaluate** on a fixed validation set, and **compare metrics against the current production model** — fail the pipeline if quality regresses past a threshold, not just if code errors out.
4. **Register** the new model version (question 4) only if it clears that bar.
5. **Deploy to staging first**, run smoke tests against real-shaped requests, then promote to production — covered in depth in `production-ml-practice.md`.

The throughline: in ML, "the build passed" and "the model is still good" are two different questions, and a pipeline that only checks the first one will happily ship a quietly worse model.

### 7. Given every stop on the loop (tracking, registry, versioning, CI/CD) is now in place, what's the actual MINIMUM checklist to reproduce a result from three months ago?
To actually reproduce a past result, you need all of: the exact code version (git commit hash), the exact data version (question 5's DVC hash or equivalent), the exact hyperparameters (question 2-3's MLflow/W&B log), the random seed, and the library/framework versions (`requirements.txt` pinned, not just `torch` unpinned). Missing any one of these turns "reproduce last month's result" into a guessing game — the checklist is really just naming every stop on the loop above and confirming each one left a durable, checkable record.

### Summary example
A fraud model's journey around the full loop: MLflow (question 2) logs 50 tuning runs, the best one gets registered and promoted to `Production` (question 4), trained on a DVC-tracked data version (question 5) that a CI/CD pipeline validated and compared against the prior production model before allowing the promotion (question 6) — and three months later, reproducing that exact result needs the git commit, the DVC hash, the logged hyperparameters, the seed, and pinned library versions all together (question 7), because any single missing piece from this loop turns "reproduce it" into a guessing game.

## Practice Q&A (Self-Test)

### What's the actual difference between an experiment-tracking tool (MLflow/W&B) and a model registry?
Experiment tracking logs every run's parameters/metrics/artifacts so you can compare and find the best one. A model registry is the next step after you've picked a winner — it version-controls that specific model with a lifecycle status (Staging/Production/Archived) so there's always a clear, auditable answer to "what's live right now."

### Why can't you just `git add` a 5 GB model checkpoint the same way you'd add a Python file?
Git is optimized for line-by-line text diffing; large binary files don't diff meaningfully and bloat the repo's history permanently (every clone re-downloads every version ever committed). DVC (or Git LFS) keeps the actual large file in external storage and commits only a small pointer file to Git, keeping the repo itself lightweight while still versioning the data/model.

### An ML CI/CD pipeline passes all its tests and merges, but the newly retrained model is quietly 8% less accurate than the one it replaced. What pipeline step was missing?
A step that evaluates the new model against a fixed validation set and compares its metrics to the current production model's, failing the pipeline on regression — standard code tests (lint, unit tests, build) only check that the code runs, not that the model it produces is still good.

### You need to reproduce a model result from three months ago. You have the code's git commit hash and the hyperparameters from MLflow — is that enough?
Not necessarily — you also need the exact data version used (via DVC or equivalent), the random seed, and pinned library versions. Any one of those drifting (data was updated since, a library version bumped) can silently change the result even with identical code and hyperparameters.
