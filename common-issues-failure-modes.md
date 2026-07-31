# Common Issues & Failure Modes — What Breaks, and Why

A reference for the "wait, why is this happening" moments — organized by where in the stack they show up: classical ML, deep learning, LLM/RAG systems, and production. Every entry follows the same shape: the symptom, the actual cause, and the fix. In simple language, cross-linked to the docs that go deeper on the mechanism.

**Visual + memory hook — triage by WHEN the symptom shows up, before reading any entry below:**
```
                    Something's wrong. WHEN did you notice?

  During training              Offline eval looks         In production,
  (loss is NaN, or             great, deployment           nothing crashed,
  not moving at all)           looks bad                   just... worse
        │                            │                            │
        ▼                            ▼                            ▼
  DEEP LEARNING              CLASSICAL ML / DATA            PRODUCTION section
  section (NaN loss,         section (data leakage,         (drift, training/
  vanishing gradients,       train/serving skew)            serving skew, cost
  DDP/OOM)                                                  blowup, forgetting)
```
**Remember it as "when, not what," for the first triage step** — the fastest way into this doc under real pressure isn't scanning every entry for a matching symptom, it's asking WHEN the problem became visible (during training / at eval time / after shipping), which immediately cuts the search to one of the four sections below. Only after that narrows it down does the specific symptom wording start to matter.

## Classical ML & Data Issues

**Symptom: model does suspiciously well on test data, badly in the real world.**
Cause: **data leakage** — information from outside the training data (often, accidentally, from the future or from the target itself) snuck into the features. Common concrete versions: normalizing/scaling using statistics computed on the *whole* dataset before splitting (so test-set information leaks into training-set scaling); a feature that's actually a proxy for the target (e.g. "account_closed_date" as a feature when predicting churn); joining in a table that only has rows for the outcome you're trying to predict. Fix: always split *before* any fitting step (scalers, imputers, encoders — see `sklearn.pipeline.Pipeline` in `sql-practice.md`'s sibling docs), and audit any feature that correlates suspiciously well with the target.

**Symptom: 95% accuracy, but the model is useless.**
Cause: **class imbalance** — if 95% of transactions are legitimate, "always predict legitimate" already scores 95% accuracy while catching zero fraud. Fix: look at precision/recall/F1 per class, not overall accuracy (see `ds-fundamentals`'s confusion-matrix material); consider resampling (SMOTE) or class weights.

**Symptom: a model that performed great in testing predicts garbage on a specific subgroup in production.**
Cause: the training data underrepresented that subgroup, so the model never really learned it — this is a *fairness/bias* issue as much as an accuracy one (see the bias-taxonomy material in the NCA-GENL guide's Trustworthy AI section). Fix: check per-subgroup performance explicitly, don't rely on an aggregate metric to reveal a subgroup-specific gap.

**Symptom: cross-validation score looks great, single holdout test looks much worse.**
Cause: the CV folds weren't independent of some grouping structure in the data (e.g. multiple rows from the same customer end up split across train and test folds within a fold, letting the model partially "memorize" that customer). Fix: use grouped cross-validation (`GroupKFold`) when rows aren't truly independent.

## Deep Learning Issues

**Symptom: loss becomes `NaN` partway through training.**
Cause, in rough order of likelihood: learning rate too high (the classic first thing to check — see the gradient-descent worked example in `ds-fundamentals`), an unstable operation (e.g. `log(0)` from a probability that hit exactly 0, or division by a variance that hit exactly 0), or exploding gradients in a deep/recurrent network. Fix: lower the learning rate, add gradient clipping (`clip_grad_norm_` — see `module-cheatsheet.md`), add a small epsilon inside anything doing a log or division, check for `inf`/`NaN` in the data itself.

**Symptom: training loss keeps dropping, validation loss stops improving or gets worse.**
Cause: **overfitting** — the model is memorizing training-specific noise rather than learning the general pattern. Fix: more data, regularization (L1/L2, dropout — see `math-foundations-refresher.md` for why L1 vs L2 behave differently), early stopping, or a smaller/simpler model.

**Symptom: loss barely moves at all, from the very first epoch.**
Cause: **vanishing gradients** in a deep network (each layer's gradient gets multiplied by a small number, and it compounds — `0.25²⁰` is effectively zero, as worked out in the NCA-GENL guide), a learning rate that's far too low, or all weights initialized to the same value (breaking symmetry never happens, every neuron in a layer learns the identical thing). Fix: residual connections, better initialization, batch/layer normalization, a higher learning rate with a warmup schedule.

**Symptom: works fine on CPU or a single GPU, breaks (or silently gives different results) once you add more GPUs.**
Cause: a common **DDP (Distributed Data Parallel)** trap — forgetting to use a `DistributedSampler` on the `DataLoader`, so every GPU sees the *same* data instead of a disjoint shard of it, silently wasting compute and inflating the effective batch size without you intending it. Fix: always pair DDP with a `DistributedSampler`, and call `sampler.set_epoch(epoch)` each epoch so shuffling actually differs across epochs.

**Symptom: GPU out-of-memory (OOM) error, sometimes only after several epochs of running fine.**
Cause: usually one of — a batch size too large for available VRAM, gradients/activations accumulating because you forgot `optimizer.zero_grad()` or ran a forward pass without `torch.no_grad()` during evaluation, or a growing Python-side list accidentally holding onto GPU tensors (e.g. appending `loss` instead of `loss.item()` to a logging list, which keeps that computation graph alive). Fix: reduce batch size or use gradient accumulation, double check `.zero_grad()`/`.no_grad()` usage, and always detach/`.item()` anything you're just logging.

## LLM & RAG Issues

**Symptom: the model states something confidently that's simply false.**
Cause: **hallucination** — the model is a next-token predictor; it will produce a fluent, plausible-sounding continuation whether or not the underlying fact is true, especially for rare/specific/recent facts it saw little of during training. Fix: ground answers in retrieved, verifiable context (RAG — see `rag-deeper.md`), instruct the model explicitly to say "I don't know" when the context doesn't support an answer, and check faithfulness (does the answer only use claims present in the retrieved context) as its own metric, separate from whether the answer merely "sounds right."

**Symptom: a RAG system gives a wrong or incomplete answer even though the right document is in the corpus.**
Cause: almost always a **retrieval** failure, not a generation failure — the right chunk wasn't in the top-k results (wrong chunk size, a purely dense embedding search missing an exact keyword match, or a multi-hop question needing information from two different chunks at once). Fix: work through `rag-deeper.md`'s techniques (hybrid search, re-ranking, multi-hop) in order of how the failure actually looks, and evaluate retrieval and generation as separate metrics (RAGAS-style) so you know which one to fix.

**Symptom: the conversation gets long and the model "forgets" earlier context or starts ignoring the system prompt.**
Cause: **context window limits** — once the conversation plus retrieved context exceeds the model's context length, something has to be truncated (usually the oldest messages), and even within the window, models pay less reliable attention to information buried in the middle of a very long context than to the beginning or end ("lost in the middle"). Fix: summarize/compress older turns instead of keeping full history verbatim, keep the system prompt and critical instructions positioned where they get the most attention (start and end), and trim retrieved context to what's actually relevant (contextual compression, `rag-deeper.md`).

**Symptom: API costs blow up unexpectedly.**
Cause: an uncapped conversation history growing every turn (each turn re-sends the *entire* history, so cost grows roughly quadratically over a long conversation), overly large `max_tokens`, or an agent loop that keeps calling tools/re-prompting far more times than expected on an edge case. Fix: cap and summarize history length, set sane `max_tokens`, add hard iteration limits to any agentic loop, and log token usage per call during development so a runaway pattern is visible before it's expensive.

**Symptom: fine-tuning "worked" (loss went down) but the model got worse at everything it used to do well.**
Cause: **catastrophic forgetting** — full fine-tuning on a narrow dataset can overwrite general capabilities the base model had. Fix: this is a large part of why parameter-efficient fine-tuning (LoRA/QLoRA — see `core-technical-depth.md`) is preferred in practice: freezing the base weights and only training a small adapter naturally limits how much general capability can be overwritten.

## Practice Q&A (Self-Test)

### A model scores 96% accuracy in testing but performs noticeably worse once deployed, on data that looks similar to the training set. What should you check first?
Data leakage — specifically, whether any preprocessing step (scaling, imputing, encoding) was fit on the full dataset *before* the train/test split, letting test-set information leak into training. This is the single most common cause of an inflated offline score that doesn't hold up.

### Training loss goes to `NaN` at epoch 3. What are the first three things to check, in order?
Learning rate (lower it first — the highest-probability cause), then look for an unstable operation like `log(0)` or division by a near-zero value in the loss/metric computation, then check whether gradients are exploding (add gradient clipping) — especially likely in deep or recurrent architectures.

### Multi-GPU training via DDP finishes faster than single-GPU but the final model is no better, as if it only saw the same amount of data as one GPU would. What's the likely bug?
Missing `DistributedSampler` on the `DataLoader` — without it, every GPU processes the *same* full dataset rather than a disjoint shard, so you get more compute thrown at identical data rather than the intended larger effective dataset coverage per epoch.

### A RAG chatbot gives a wrong answer, and you've confirmed the correct document IS in the vector store. Is this a generation problem or a retrieval problem, and how do you confirm which?
Almost certainly retrieval — check what chunks were actually returned for that query; if the correct chunk wasn't in the top-k, that's a retrieval failure (fix via hybrid search/re-ranking/query rewriting), not something prompting the generation step differently would fix. Measuring context precision/recall separately from faithfulness (RAGAS-style, `rag-deeper.md`) is how you confirm this instead of guessing.

### Full fine-tuning of an LLM on a narrow customer-support dataset makes it much better at support tickets but noticeably worse at general reasoning it used to handle fine. What's this called, and what's the standard mitigation?
Catastrophic forgetting — updating all the weights on a narrow dataset can overwrite general capabilities learned during pretraining. The standard mitigation is parameter-efficient fine-tuning (LoRA/QLoRA): freezing the base weights and training only a small adapter naturally bounds how much general capability gets overwritten.
