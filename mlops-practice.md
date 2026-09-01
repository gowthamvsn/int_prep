# MLOps Practice — Experiment Tracking, Versioning, and Model Registries

Everything else on this hub is about building a good model. MLOps is about something different: not losing track of what you built, being able to reproduce it, and handing it off cleanly to production.

There's a real gap between "I trained a model once" and "I can tell you exactly which model is in production, what data trained it, and how to roll it back." This doc is about closing that gap.

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
Remember it as a loop with one entry and one exit label per stage: track, register, deploy, monitor, retrain, back to track.

Every section below is really just answering one question: what tool sits at this one stop on the loop, and what does it log, version, or promote there? When a new MLOps tool name comes up that isn't in this doc, the fastest way to place it is asking which single stop on this loop it belongs to. That's faster than treating it as a brand-new concept.

## Walking the loop, one stop at a time

### Why do you need experiment tracking at all? Isn't a spreadsheet enough?

Say you've run twenty training experiments. A different learning rate here. A different data version there. A small architecture tweak on top.

Now try to answer one question: which run was the good one, and what exact settings produced it?

From memory, you can't. From a spreadsheet you forgot to update half the time, you also can't.

Experiment tracking tools solve this by logging every run automatically. You don't need any discipline in the moment. The tool just does it, every time.

### What does the standard tracking tool actually log?

**MLflow** is the open-source standard. Here's what one run looks like:

```python
import mlflow

with mlflow.start_run():
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("model_type", "resnet18")
    mlflow.log_metric("val_accuracy", 0.943)
    mlflow.log_metric("val_loss", 0.187)
    mlflow.pytorch.log_model(model, "model")
```

Every run logs three kinds of things:
- **Parameters** — the settings you chose, like learning rate and model type.
- **Metrics** — the results you got. You can log these per-epoch too, not just once at the end.
- **Artifacts** — the actual model file, plus plots or anything else worth keeping.

MLflow's UI then lets you sort and filter hundreds of runs side by side. You can ask something like "show me every run with `val_accuracy > 0.93`, sorted by `learning_rate`" and get an answer in seconds instead of digging through logs.

### When would you reach for Weights & Biases instead of MLflow?

**Weights & Biases (W&B)** does the same basic job — logging params, metrics, and artifacts — but it's hosted, and more visual.

```python
import wandb

wandb.init(project="fraud-detection", config={"lr": 0.001, "batch_size": 32})
for epoch in range(epochs):
    wandb.log({"train_loss": train_loss, "val_accuracy": val_acc})
```

The comparison dashboards work about the same way as MLflow's. What you get on top: real-time charts that update live while training runs, and easier sharing across a team, since it's cloud-hosted by default.

The two tools aren't mutually exclusive. Use MLflow for anything that should stay entirely in-house. Use W&B where a hosted dashboard and easy collaboration matter more than keeping everything internal.

### Once you have hundreds of tracked runs, how do you go from "the best run so far" to "the model that's actually live"?

A **model registry** is a versioned, centralized store of trained models. Each model version carries a lifecycle status: `Staging` → `Production` → `Archived`.

Without a registry, a model in production is just "whatever file someone last copied to the server." With one, every version is tracked, and promoting a new model — or rolling back to an old one — is a deliberate, auditable action.

```python
mlflow.register_model("runs:/<run_id>/model", "fraud-detector")
# later, promote a specific version
client.transition_model_version_stage("fraud-detector", version=3, stage="Production")
```

A registry exists to answer one question on demand: what model is live right now, what version is it, what data trained it, and can we get back to the previous one in one step? Without a registry, answering that question means digging through old commits and Slack messages. With one, it's a lookup.

### How do you also track which DATA trained a model — since a 10 GB dataset can't just live in Git?

A registry (above) tracks which *model* is live. But a model is only as reproducible as the data that trained it, and Git wasn't built for that data.

Git is built for text diffs. A 2 GB model checkpoint or a 10 GB training dataset doesn't diff meaningfully — committing it just bloats the repo into something unusable.

**DVC (Data Version Control)** solves this the same way Git LFS does. The actual large file lives in external storage — S3, GCS, a shared drive. DVC commits a small pointer file to Git instead of the file itself.

```bash
dvc add data/training_set.csv       # stores the real file externally, creates a small .dvc pointer
git add data/training_set.csv.dvc   # git tracks only the lightweight pointer
git commit -m "add v2 training data"
dvc push                            # uploads the actual data to remote storage
```

Now `git log` on that `.dvc` file gives you a real history of which data version trained which model. That's the same reproducibility Git already gives your code, just extended to the large files Git can't handle directly. It's also the exact arrow in the loop diagram above labeled "retrain on new/versioned data" — it's what feeds back into tracking.

### What has to happen automatically before a new model is trusted enough to deploy?

Tracked runs, a registry, and versioned data are all in place now. One thing is still missing: an automated check that a new model is actually good enough before it ships.

That's **CI/CD for ML**. CI/CD (continuous integration / continuous deployment) is the automated pipeline that tests and ships every code change without a human running each step by hand.

Regular CI/CD asks one question: does the code still work? Tests pass, it builds, done.

ML CI/CD has to ask a second question on top of that: does the *model* still work? A code change can pass every unit test and still quietly make the model worse. A reasonable ML pipeline, triggered on a pull request or a merge, looks like this:

1. Standard checks — lint, unit tests, type checks. Nothing ML-specific yet.
2. **Data validation** — a schema check on any new or changed data. Right columns, right types, no unexpected nulls. This runs before that data is allowed to feed a training job.
3. **Retrain or evaluate** on a fixed validation set, then **compare the new metrics against the current production model's metrics**. Fail the pipeline if quality regresses past a threshold — not just if the code throws an error.
4. **Register** the new model version, but only if it clears that bar.
5. **Deploy to staging first.** Run smoke tests against real-shaped requests, then promote to production. `production-ml-practice.md` covers this rollout step in depth.

The throughline: in ML, "the build passed" and "the model is still good" are two different questions. A pipeline that only checks the first one will happily ship a quietly worse model.

### With tracking, a registry, versioning, and CI/CD all in place — what's the actual minimum checklist to reproduce a result from three months ago?

You need all five of these together:
1. The exact code version — the git commit hash.
2. The exact data version — the DVC hash from above, or an equivalent.
3. The exact hyperparameters — logged by MLflow or W&B.
4. The random seed.
5. The library and framework versions — a pinned `requirements.txt`, not just `torch` left unpinned.

Miss any one of these, and "reproduce last month's result" turns into a guessing game. The checklist above is really just naming every stop on the loop from the top of this doc, and confirming each one left a durable, checkable record.

### Summary example

Here's a fraud model's full trip around the loop, step by step.

1. MLflow logs 50 tuning runs, with params, metrics, and artifacts for each one.
2. The best run gets registered, then promoted to `Production`.
3. It was trained on a DVC-tracked data version, so there's a real record of which data produced it.
4. A CI/CD pipeline validated that data and compared the new model against the prior production model before allowing the promotion.
5. Three months later, someone needs to reproduce that exact result. That takes five things together: the git commit, the DVC hash, the logged hyperparameters, the seed, and the pinned library versions.

Any single missing piece from that list turns "reproduce it" into a guessing game.

---

## Practice Q&A (Self-Test)

**Q1. What's the actual difference between an experiment-tracking tool (MLflow/W&B) and a model registry?**
A: Experiment tracking logs every run's parameters, metrics, and artifacts, so you can compare runs and find the best one. A model registry is the next step after you've picked a winner. It version-controls that specific model with a lifecycle status — Staging, Production, Archived — so there's always a clear, auditable answer to "what's live right now."

**Q2. Why can't you just `git add` a 5 GB model checkpoint the way you'd add a Python file?**
A: Git is built for line-by-line text diffing. Large binary files don't diff meaningfully, and adding them bloats the repo's history permanently — every future clone re-downloads every version that was ever committed. DVC (or Git LFS) keeps the actual large file in external storage and commits only a small pointer file to Git. The repo stays lightweight, and the data or model still gets versioned.

**Q3. An ML CI/CD pipeline passes all its tests and merges. The newly retrained model is quietly 8% less accurate than the one it replaced. What pipeline step was missing?**
A: A step that evaluates the new model against a fixed validation set, compares its metrics to the current production model's, and fails the pipeline on regression. Standard code tests — lint, unit tests, build — only check that the code runs. They say nothing about whether the model it produces is still good.

**Q4. You need to reproduce a model result from three months ago. You have the code's git commit hash and the hyperparameters from MLflow. Is that enough?**
A: Not necessarily. You also need the exact data version used, via DVC or an equivalent, plus the random seed and the pinned library versions. Any one of those can drift silently — data gets updated, a library gets bumped — and change the result even with identical code and hyperparameters.
