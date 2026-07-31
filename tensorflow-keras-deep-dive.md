# TensorFlow / Keras Deep Dive — Built as a Chain, Not a List

Same format as `pytorch-deep-dive.md`, the TensorFlow/Keras side. **Verification note, stated plainly:** this environment's global Python has a broken TensorFlow install (TF 2.15 vs. NumPy 2 — a real, pre-existing conflict, not something introduced this session). Every snippet below was actually run and verified, but in a separate, isolated virtual environment (`.venv-tf`, TensorFlow 2.10.1 — the last version with native Windows GPU support) created specifically so this verification never touched the global environment. `import tensorflow as tf` and `from tensorflow import keras` assumed throughout. Each cluster is one continuous thread — every question inherits the answer before it, closing with a worked summary example.

---

## Cluster 1 — Functional API: When Sequential Genuinely Can't Express the Model

### 1. How do you build the same model with the Functional API instead of Sequential, and why would you need to?
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

inputs = keras.Input(shape=(10,))
x = layers.Dense(32, activation="relu")(inputs)
x = layers.Dense(16, activation="relu")(x)
outputs = layers.Dense(2, activation="softmax")(x)
model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
```
`Sequential` only supports one input, one output, and a single linear chain of layers — the moment you need multiple inputs, multiple outputs, a skip/residual connection, or any layer reused in more than one place, `Sequential` genuinely cannot express it. The Functional API represents the model as an explicit graph of tensors, which can express all of those.

### 2. Given that Functional can express more than one input, how do you build a model with TWO inputs and one output — a real use case, not a toy?
```python
tabular_input = keras.Input(shape=(10,), name="tabular")
image_input = keras.Input(shape=(32, 32, 3), name="image")

tab_branch = layers.Dense(16, activation="relu")(tabular_input)
img_branch = layers.Conv2D(16, 3, activation="relu")(image_input)
img_branch = layers.GlobalAveragePooling2D()(img_branch)

merged = layers.concatenate([tab_branch, img_branch])
output = layers.Dense(1, activation="sigmoid")(merged)
model = keras.Model(inputs=[tabular_input, image_input], outputs=output)
```
`layers.concatenate` (not `layers.Add`) is the right merge here: `concatenate` joins two different-meaning feature vectors side by side, preserving both independently for later layers to combine as they learn to; `Add` requires matching shapes and forces an element-wise sum, which only makes sense when both branches represent the SAME kind of quantity (like a residual connection), not two unrelated feature spaces like tabular data and an image embedding.

### Summary example
A fraud model that combines structured transaction fields with a scanned receipt image needs exactly this shape: two `keras.Input` branches (question 2), each processed by layers suited to its own data type, merged with `concatenate` rather than `Add` because a transaction amount and an image embedding aren't the same kind of quantity — a model `Sequential` (question 1) could never express, since it only ever accepts one input tensor.

---

## Cluster 2 — Custom Training Loops: What `.fit()` Was Hiding

### 1. How do you write a custom training loop with `GradientTape`, for when `.fit()` isn't flexible enough?
```python
import numpy as np

model = keras.Sequential([layers.Dense(16, activation="relu"), layers.Dense(2)])
optimizer = keras.optimizers.Adam(learning_rate=1e-3)
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

X = np.random.randn(32, 10).astype("float32")
y = np.random.randint(0, 2, 32)

with tf.GradientTape() as tape:
    logits = model(X, training=True)         # training=True matters -- see the next question
    loss = loss_fn(y, logits)
grads = tape.gradient(loss, model.trainable_weights)
optimizer.apply_gradients(zip(grads, model.trainable_weights))
print(float(loss))
```
`GradientTape` records only what happens inside its `with` block: it watches every trainable-variable operation while open and builds a computation graph for differentiation on the fly — anything computed outside the block (or on a tensor not marked as watched) simply won't have a gradient, mirroring PyTorch's dynamic-graph model rather than TF1's static-graph approach.

**Visual + memory hook — an actual cassette tape, recording only while it's rolling:**
```
                    ┌───────────────────────────────┐
                    │   with tf.GradientTape() as tape:   ◀── tape starts ROLLING
                    │       logits = model(X)         │      records: every op touching
                    │       loss = loss_fn(y, logits)  │      a trainable variable
                    └───────────────────────────────┘
                                    │  tape.gradient(loss, weights) ◀── tape STOPS,
                                    ▼                                   plays back in
                              gradients, one per weight                 reverse (backprop)

  loss = loss_fn(y, model(X_extra))   ◀── computed OUTSIDE the `with` block:
                                          the tape was already stopped — no recording,
                                          no gradient, silently
```
**Remember it as a tape recorder, literally** — indentation inside the `with` block is the record light being on; anything that happens once you've dedented back out happened after the recorder was switched off, and asking `tape.gradient()` for something it never recorded returns `None` rather than an error, which is exactly why a custom training loop can silently fail to train a layer if part of the forward pass accidentally sits outside the block.

### 2. Given the tape recorded `model(X, training=True)`, what does that `training=` flag actually control, and why does it matter outside `.fit()`/`.predict()`?
```python
# during custom training:
logits = model(X, training=True)     # Dropout active, BatchNorm uses CURRENT batch statistics

# during custom evaluation/inference:
logits = model(X, training=False)     # Dropout off, BatchNorm uses stored RUNNING statistics
```
This is the direct Keras analogue of PyTorch's `model.train()`/`model.eval()`: `.fit()` and `.predict()` set this flag for you automatically; the moment you call the model directly (inside a custom `GradientTape` loop, or for a manual inference call), YOU are responsible for passing the right value — forgetting `training=False` at inference time is the same class of bug as forgetting `model.eval()` in PyTorch, with the same consequence (wrong BatchNorm statistics, unwanted Dropout noise).

### Summary example
A custom `GradientTape` loop that trains but evaluates suspiciously worse than `.fit()` would on the same data usually has one bug: the eval call still says `training=True` (question 2) — Dropout is still randomly zeroing activations and BatchNorm is still using per-batch statistics instead of the stored running ones, degrading eval numbers for a reason that has nothing to do with the tape's recording (question 1) at all.

---

## Cluster 3 — Custom Layers and Models: Subclassing When Built-Ins Run Out

### 1. How do you build a custom layer (subclassing), not just stacking built-in ones?
```python
class WeightedSum(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def build(self, input_shape):
        # build() runs ONCE, the first time the layer sees real input, when shapes are actually known
        self.w = self.add_weight(shape=(input_shape[-1],), initializer="ones", trainable=True, name="w")
    def call(self, inputs):
        return inputs * self.w      # element-wise learned weighting

layer = WeightedSum()
out = layer(tf.random.normal((4, 10)))
print(out.shape, layer.w.shape)
```
Weight creation goes in `build()`, not `__init__`: at `__init__` time, the layer doesn't yet know the shape of its actual input (you might not have connected it to anything yet) — `build()` is called automatically the first time real data flows through, when `input_shape` is genuinely known, which is what lets the SAME layer class adapt to different input sizes without hardcoding a shape in the constructor.

### 2. Given a custom layer handles one non-standard OPERATION, how do you subclass `keras.Model` itself for non-standard forward LOGIC (control flow, not just a new op)?
```python
class TwoTowerModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.tower_a = keras.Sequential([layers.Dense(16, activation="relu")])
        self.tower_b = keras.Sequential([layers.Dense(16, activation="relu")])
        self.head = layers.Dense(1, activation="sigmoid")
    def call(self, inputs):
        a, b = inputs
        combined = tf.concat([self.tower_a(a), self.tower_b(b)], axis=-1)
        return self.head(combined)

model = TwoTowerModel()
out = model([tf.random.normal((4, 10)), tf.random.normal((4, 8))])
print(out.shape)
```
Subclassing is the right tool when the forward pass has real CONTROL FLOW (a conditional branch, a loop over a variable number of inputs, recursive calls) that the Functional API's static-graph-of-layers can't express — for a fixed, static architecture like this simple example, Functional (Cluster 1) would actually be the more common choice; subclassing earns its complexity when the logic genuinely needs it.

### Summary example
A recommendation model that routes each input through a DIFFERENT tower depending on a runtime flag (real conditional control flow) can't be expressed as a static Functional graph at all — it needs `keras.Model` subclassing (question 2), the same escalation from custom-operation (question 1's `WeightedSum`, one learned multiply) to custom-control-flow (an `if` inside `call()`) that separates "needs a new layer" from "needs a new model class."

---

## Cluster 4 — Custom Losses and Metrics, With Configuration

### 1. How do you write a custom loss as a CLASS (not just a function) when it needs configuration, like focal loss's `gamma`?
```python
class FocalLoss(keras.losses.Loss):
    def __init__(self, gamma=2.0, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
    def call(self, y_true, y_pred):
        bce = keras.losses.binary_crossentropy(y_true, y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        return bce * tf.pow(1 - p_t, self.gamma)     # down-weights already-easy, well-classified examples

model.compile(optimizer="adam", loss=FocalLoss(gamma=2.0))
```
Standard cross-entropy weighs every example equally regardless of how confidently correct the model already is on it; focal loss's `(1-p_t)^gamma` term shrinks the loss contribution from examples the model already classifies confidently and correctly, effectively focusing training signal on the hard/misclassified examples — genuinely useful on severely imbalanced data where easy negatives would otherwise dominate the gradient.

### 2. Given a loss needs its own state (`gamma`), a METRIC needs running state too (precision/recall accumulated across batches) — how do you write a custom metric class?
```python
class F1Score(keras.metrics.Metric):
    def __init__(self, name="f1", **kwargs):
        super().__init__(name=name, **kwargs)
        self.precision = keras.metrics.Precision()
        self.recall = keras.metrics.Recall()
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)
    def result(self):
        p, r = self.precision.result(), self.recall.result()
        return 2 * p * r / (p + r + keras.backend.epsilon())    # epsilon avoids divide-by-zero
    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[F1Score()])
```
`keras.backend.epsilon()` in the denominator, specifically: early in training (or on a bad batch), precision and recall can both legitimately be exactly 0, making `p + r` zero — without the small epsilon guard, this raises a division error or produces `nan` that can silently propagate into logged metrics and confuse monitoring. Note the three-method shape (`update_state`/`result`/`reset_state`) is what lets a metric accumulate correctly ACROSS batches within an epoch, unlike a loss which is computed fresh per batch.

### Summary example
Training on severely imbalanced fraud data: `FocalLoss` (question 1) reshapes the GRADIENT signal so easy negatives stop dominating, while a custom `F1Score` metric (question 2) reshapes what gets REPORTED, since plain accuracy on imbalanced data is misleading regardless of what loss was used to train — the two customizations solve different problems (what the optimizer optimizes vs. what you actually read to judge the model) and are routinely used together.

---

## Cluster 5 — Efficient Data Pipelines

### 1. How do you build an efficient input pipeline with `tf.data` instead of feeding raw NumPy arrays directly?
```python
X = np.random.randn(1000, 10).astype("float32")
y = np.random.randint(0, 2, 1000)

dataset = tf.data.Dataset.from_tensor_slices((X, y))
dataset = dataset.shuffle(buffer_size=1000).batch(32).prefetch(tf.data.AUTOTUNE)

for xb, yb in dataset.take(1):
    print(xb.shape, yb.shape)
```
`.prefetch(tf.data.AUTOTUNE)` matters specifically: without it, the GPU sits idle while the CPU prepares the next batch, and vice versa — `prefetch` overlaps data preparation for batch N+1 with the model's computation on batch N, and `AUTOTUNE` lets TensorFlow pick the actual optimal buffer size at runtime rather than you guessing a fixed number.

### 2. Given a `tf.data` pipeline feeds the model, how do you add on-the-fly data augmentation as part of the model itself, rather than a separate preprocessing step outside the pipeline?
```python
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

model = keras.Sequential([
    keras.Input(shape=(32, 32, 3)),
    data_augmentation,          # active only during training -- automatically inert at inference
    layers.Conv2D(16, 3, activation="relu"),
    layers.GlobalAveragePooling2D(),
    layers.Dense(10, activation="softmax"),
])
```
Augmentation layers built INTO the model (vs. a separate offline preprocessing script) apply ONLY during training (the same `training=True`/`False` flag from Cluster 2) and pass through unchanged during inference — no separate code path needed, no risk of accidentally applying random augmentation to real inference input, and the augmentation runs on GPU as part of the model graph instead of the CPU, which is often faster.

### Summary example
A vision pipeline that both loads efficiently AND augments correctly needs both pieces: `tf.data` with `prefetch(AUTOTUNE)` (question 1) keeps the GPU fed instead of idle, and augmentation layers inside the model (question 2) reuse the exact same `training=` flag from Cluster 2's `GradientTape` discussion to guarantee augmentation never leaks into a real inference call.

---

## Cluster 6 — Callbacks: Automating What the PyTorch Loop Did by Hand

### 1. How do you stop training automatically when validation loss stops improving?
```python
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True,
)
# model.fit(X, y, validation_split=0.2, epochs=100, callbacks=[early_stop])
```
`restore_best_weights=True` is the detail worth not skipping: without it, training stops at the RIGHT time but the model's weights are left at whatever the LAST epoch produced (already past the best point, by definition of "patience" epochs) — this flag rolls the weights back to the actual best-performing checkpoint before returning, which is almost always what you actually want.

### 2. Given early stopping ends training at the right point, how do you also save a checkpoint DURING training, only when the model actually improves?
```python
checkpoint = keras.callbacks.ModelCheckpoint(
    filepath="best_model.keras", monitor="val_loss", save_best_only=True,
)
```
`save_best_only=True` matters over saving every epoch: saving every epoch either wastes disk space or forces you to manually track which file was best — this flag does the "only checkpoint on improvement" logic that had to be hand-rolled manually in `pytorch-deep-dive.md`'s early-stopping example, built directly into the framework here.

### 3. Given training eventually stalls, is stopping the only option — how do you have Keras try a smaller learning rate FIRST, before giving up?
```python
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6,
)
```
This is a genuinely different tool from `EarlyStopping`, not a duplicate: `EarlyStopping` gives up on training entirely once it stalls; `ReduceLROnPlateau` instead tries a smaller learning rate first (a stalled loss often means the current LR is too large to make further progress near a minimum) — commonly used TOGETHER, with `ReduceLROnPlateau`'s patience set lower so it tries recovering before `EarlyStopping`'s patience gives up entirely.

### Summary example
A realistic callback stack combines all three, tuned in the order they should fire: `ReduceLROnPlateau` (question 3) with `patience=3` tries a smaller LR first at the first sign of a stall, `ModelCheckpoint` (question 2) saves every genuine improvement along the way regardless of which LR produced it, and `EarlyStopping` (question 1) with `patience=5` and `restore_best_weights=True` only gives up — and rolls back to the true best epoch — if even the reduced LR fails to help within 2 further epochs.

---

## Cluster 7 — Mixed Precision and Transfer Learning

### 1. How do you enable mixed-precision training in Keras — the TF-side equivalent of `torch.autocast` from `pytorch-deep-dive.md`?
```python
keras.mixed_precision.set_global_policy("mixed_float16")
# every new model built after this line uses float16 for most compute, float32 for numerically sensitive ops
model = keras.Sequential([layers.Dense(64, activation="relu"), layers.Dense(2, activation="softmax")])
print(model.dtype_policy)
```
This needs to be set as a GLOBAL POLICY before model creation, unlike PyTorch's per-op `autocast` context: Keras bakes the precision policy into each layer at construction time — layers built before calling `set_global_policy` keep their original (usually float32) policy, so this must run before you build the model, not wrapped around just the training step like PyTorch's `autocast` context manager.

### 2. Separately from precision, how do you do transfer learning with a pretrained model in Keras?
```python
base_model = keras.applications.MobileNetV2(input_shape=(96, 96, 3), include_top=False, weights="imagenet")
base_model.trainable = False        # freeze the pretrained backbone entirely

model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(10, activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
```
`include_top=False` matters: the pretrained model's original final classification layers were trained for ImageNet's specific 1000 classes — `include_top=False` strips those off, keeping only the convolutional feature-extraction backbone, so you can attach your own classification head sized for YOUR number of classes.

### 3. Given the backbone trained frozen in question 2, how do you unfreeze part of it for fine-tuning, and why change the learning rate when you do?
```python
base_model.trainable = True
for layer in base_model.layers[:-20]:      # keep all but the last 20 layers frozen
    layer.trainable = False
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5), loss="sparse_categorical_crossentropy")
```
Unfreeze only the LAST layers, and use a much lower learning rate this time: early layers in a pretrained CNN learn generic features (edges, textures) that transfer well across almost any image task; later layers learn more task-specific, higher-level features — unfreezing just the later layers lets the model adapt those specifically to your task while preserving the generic early features. The much lower LR (1e-5 vs. a typical 1e-3) protects the pretrained weights from being destroyed by a large gradient step now that they're unfrozen and being updated.

### Summary example
A resource-constrained image classifier trains in two phases that combine questions 2 and 3, optionally sped up by question 1: first with `base_model.trainable = False` and a normal LR to train just the new head cheaply, then with the last 20 layers unfrozen and `lr=1e-5` to gently adapt task-specific features — both phases can run under `mixed_precision.set_global_policy("mixed_float16")` set once at the very start, since it must precede model construction entirely.

---

## Cluster 8 — Sequence Models and Learning-Rate Schedules

### 1. How do you build an LSTM classifier in Keras — the direct equivalent of the PyTorch version in `deep-learning-practice.md`?
```python
model = keras.Sequential([
    layers.Embedding(input_dim=1000, output_dim=32, mask_zero=True),   # mask_zero: Keras' padding_idx equivalent
    layers.LSTM(64),
    layers.Dense(2, activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
```
`mask_zero=True` is the direct Keras analogue of PyTorch's `padding_idx=0`: it tells every downstream layer (the LSTM specifically) to skip position 0 (the padding token) entirely rather than process it as real content — same purpose as `padding_idx`, implemented as an automatically-propagated mask instead of a per-layer flag.

### 2. Given the LSTM trains with `optimizer="adam"` at a fixed rate, how do you use a learning-rate SCHEDULE instead — and why pass the schedule object rather than updating a plain float manually?
```python
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2, decay_steps=1000, decay_rate=0.9,
)
optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
```
The schedule is evaluated fresh at every single training STEP (not just every epoch), giving smooth, fine-grained decay — and it's saved/restored correctly alongside the optimizer's other state if you checkpoint mid-training, which a manually-updated plain float wouldn't be.

### Summary example
An LSTM text classifier (question 1) trained on variable-length, padded sequences needs `mask_zero=True` so padding never pollutes the recurrent state, and pairing it with an `ExponentialDecay` schedule (question 2) instead of `ReduceLROnPlateau` (Cluster 6) is a deliberate choice for smooth, predictable decay on a known step budget rather than reactive, validation-driven decay.

---

## Cluster 9 — Logging and Saving: What Survives After Training Ends

### 1. How do you log training for visualization in TensorBoard?
```python
tensorboard_cb = keras.callbacks.TensorBoard(log_dir="./logs", histogram_freq=1)
# model.fit(X, y, epochs=10, callbacks=[tensorboard_cb])
# then run in a terminal: tensorboard --logdir ./logs
```
`histogram_freq=1` is worth knowing about beyond just the default scalar logging: this setting logs weight and activation HISTOGRAMS every epoch — genuinely useful for spotting a layer whose weights are collapsing toward zero or exploding, a diagnostic scalar loss curves alone won't show you directly.

### 2. Given TensorBoard logs training as it happens, how do you save the trained model afterward — and when does saving weights-only matter more than saving the full model?
```python
model.save("full_model.keras")                    # architecture + weights + optimizer state, one file
reloaded = keras.models.load_model("full_model.keras")

model.save_weights("weights_only.weights.h5")       # JUST the learned numbers, no architecture
# model2 = build_same_architecture_function()
# model2.load_weights("weights_only.weights.h5")
```
Weights-only saving is sometimes the more robust choice, mirroring PyTorch's `state_dict` guidance in `pytorch-deep-dive.md`: saving the full model bundles in the exact class/config at save time — if you refactor your model-building code later, reloading the full saved model can break; saving just the weights and re-running your (version-controlled) model-building code to reconstruct the architecture is more robust to that kind of drift, at the cost of needing to keep the building code around.

### Summary example
A model trained with `histogram_freq=1` logging (question 1) reveals a layer's weights collapsing toward zero mid-training — a real bug worth fixing before shipping. Once fixed, saving the corrected model as weights-only (question 2) rather than a full `.keras` file is the safer long-term choice if the model-building code is still actively evolving, since a full-model reload would break the moment that building code changes shape.

---

## Practice Q&A (Self-Test)

**Q1. Why can't the two-input `tabular_input`/`image_input` model be built with `Sequential`, and what does the Functional API do differently?**
A: `Sequential` only supports one input, one output, and a single linear chain of layers. The Functional API represents the model as an explicit graph of tensors, which can express multiple inputs/outputs, skip connections, or any layer reused in more than one place — exactly what the two-tower example needs.

**Q2. Why is `layers.concatenate` used to merge the tabular and image branches rather than `layers.Add`?**
A: `concatenate` joins two different-meaning feature vectors side by side, preserving both independently for later layers to combine. `Add` requires matching shapes and forces an element-wise sum, which only makes sense when both branches represent the same kind of quantity (like a residual connection) — not two unrelated feature spaces like tabular data and an image embedding.

**Q3. In the custom `GradientTape` training loop, what exactly does the tape record, and what happens to a computation done outside the `with tf.GradientTape() as tape:` block?**
A: The tape watches every trainable-variable operation that happens while it's open and builds a computation graph for differentiation on the fly. Anything computed outside the block (or on a tensor not being watched) simply won't have a gradient — this mirrors PyTorch's dynamic-graph model rather than TF1's static-graph approach.

**Q4. Why must you pass `training=True`/`training=False` explicitly when calling a model directly inside a custom training loop, but not when using `.fit()`/`.predict()`?**
A: `.fit()` and `.predict()` set this flag for you automatically, but calling the model directly makes you responsible for it. Forgetting `training=False` at inference time is the same class of bug as forgetting `model.eval()` in PyTorch, with the same consequence: wrong BatchNorm statistics and unwanted Dropout noise.

**Q5. In the custom `WeightedSum` layer, why does weight creation happen in `build()` rather than in `__init__`?**
A: At `__init__` time the layer doesn't yet know the shape of its actual input. `build()` is called automatically the first time real data flows through, when `input_shape` is genuinely known — this is what lets the same layer class adapt to different input sizes without hardcoding a shape in the constructor.

**Q6. When would you choose to subclass `keras.Model` (like `TwoTowerModel`) instead of using the Functional API?**
A: Subclassing is the right tool when the forward pass has real control flow — a conditional branch, a loop over a variable number of inputs, recursive calls — that the Functional API's static graph of layers can't express. For a fixed, static architecture, Functional would actually be the more common, simpler choice.

**Q7. In the `FocalLoss` custom loss class, what does the `(1 - p_t) ** gamma` term do, and why is it useful on imbalanced data?**
A: It shrinks the loss contribution from examples the model already classifies confidently and correctly, focusing training signal on the hard/misclassified examples. Standard cross-entropy weighs every example equally regardless of confidence, so on severely imbalanced data, easy negatives would otherwise dominate the gradient.

**Q8. In the MobileNetV2 transfer-learning example, why set `include_top=False`, and why does the later fine-tuning step use a much lower learning rate (1e-5 vs. a typical 1e-3)?**
A: `include_top=False` strips off the pretrained model's ImageNet-specific classification layers, keeping only the convolutional feature-extraction backbone so a custom head can be attached. The much lower learning rate when unfreezing the last 20 layers protects the pretrained weights from being destroyed by a large gradient step now that they're being updated.

**Q9. Why must `keras.mixed_precision.set_global_policy("mixed_float16")` be called before building the model, unlike PyTorch's `autocast` context manager which wraps just the training step?**
A: Keras bakes the precision policy into each layer at construction time. Layers built before calling `set_global_policy` keep their original (usually float32) policy, so the call must happen before the model is built rather than around the training step.

**Q10. Why does the custom `F1Score` metric add `keras.backend.epsilon()` in the denominator of `2 * p * r / (p + r + epsilon)`?**
A: Early in training, or on a bad batch, precision and recall can both legitimately be exactly 0, making `p + r` zero. Without the small epsilon guard, this raises a division error or produces `nan` that can silently propagate into logged metrics and confuse monitoring.


---

## Video-Sourced Practice MCQs

A practice set for TensorFlow/Keras, built the same way as this hub's NCA-GENL community bank: topics checked against a real YouTube TensorFlow-interview-prep video, then written up as fully original multiple-choice questions here. This video was foundational/beginner level (tensors, ops, computational graphs, the 5-step Keras workflow, deployment targets) -- genuinely complementary to the intermediate/advanced clusters above (Functional API, custom training loops, custom layers/losses, tf.data, callbacks, mixed precision, sequence models) rather than overlapping them.

<script type="application/json" class="topic-quiz-data" data-title="TensorFlow/Keras Deep Dive">
[
  {
    "d": "Tensor Basics",
    "q": "TensorFlow's fundamental data structure is the tensor. How do a scalar, a vector, and a matrix relate to each other in terms of tensor RANK (dimensionality)?",
    "o": [
      "A scalar is a 0-dimensional tensor (a single value), a vector is 1-dimensional (a list of numbers), and a matrix is 2-dimensional (rows and columns) — each step up adds one more axis",
      "A scalar, vector, and matrix are all exactly the same rank (2-dimensional); the terms only differ by which programming language you're using",
      "A matrix is always 1-dimensional, identical in rank to a vector, with no meaningful distinction between the two",
      "A vector is 0-dimensional and a scalar is 1-dimensional — the naming in the first option has scalar and vector swapped"
    ],
    "a": [
      0
    ],
    "e": "Rank counts how many independent AXES/indices you need to address a single element: a scalar needs zero indices (it's just one number), a vector needs one index (position in a list), and a matrix needs two (row and column) — each is exactly one dimension higher than the last, which is precisely why they're called 0-D, 1-D, and 2-D tensors respectively. They are NOT all the same rank — that collapses a meaningful, load-bearing distinction (higher-rank tensors, like 3-D+ ones for images or batches, build on exactly this same pattern). The scalar/vector ranks in option 3 are simply swapped from their correct values. And a matrix is 2-dimensional, not 1-dimensional like a vector — needing to look up a row AND a column (two indices) is exactly what distinguishes a matrix from a vector's single index."
  },
  {
    "d": "Tensors: Constant vs. Variable",
    "q": "The transcript shows creating tensors with `tf.constant`. A separate, equally fundamental TensorFlow object is `tf.Variable`. What's the key difference that makes `Variable` necessary for training a model, when `constant` already exists?",
    "o": [
      "`tf.constant` is used exclusively for GPU computation, and `tf.Variable` is used exclusively for CPU computation, with no other distinction between them",
      "A `tf.constant` tensor's value is fixed/immutable once created; a `tf.Variable` is MUTABLE and specifically designed to be updated in place — which is exactly what a model's trainable weights need, since training repeatedly changes those values via gradient updates",
      "`tf.Variable` can only ever hold a single scalar value, while `tf.constant` is the only one of the two that can hold multi-dimensional data",
      "`tf.constant` and `tf.Variable` are simply two different names for the exact same object with no functional difference between them"
    ],
    "a": [
      1
    ],
    "e": "Model weights need to change throughout training — every gradient step updates them — which requires a MUTABLE container; `tf.constant` is deliberately immutable (its value can never change after creation), making it the wrong tool for anything that needs updating, while `tf.Variable` is specifically built to support in-place updates, which is exactly why trainable parameters are always represented as Variables, not constants. They are NOT interchangeable names for the same thing — the mutability difference is the entire reason both exist as separate concepts. Both objects can hold multi-dimensional tensor data of any shape — the distinction is mutability, not dimensionality restriction. And neither object is tied exclusively to a specific device (CPU vs. GPU) — that's controlled by separate device-placement mechanisms, unrelated to the constant/Variable distinction."
  },
  {
    "d": "Tensor Operations",
    "q": "\"Broadcasting\" is described as automatically handling operations between tensors of DIFFERENT shapes. What does broadcasting actually let you do?",
    "o": [
      "Broadcasting is a deployment-time feature only, used for distributing a trained model across many servers, unrelated to tensor arithmetic",
      "Broadcasting only applies to tensors that already have IDENTICAL shapes, and does nothing at all for tensors with genuinely different shapes",
      "Broadcasting converts any two tensors into having the exact same NUMBER OF DIMENSIONS as each other, but has no effect on being able to combine their actual values in an operation",
      "Perform an element-wise operation (like addition) between tensors of different (but compatible) shapes — e.g. adding a single value or a smaller tensor to every row of a larger one — without manually reshaping or duplicating data yourself first"
    ],
    "a": [
      3
    ],
    "e": "Broadcasting is specifically the mechanism that lets you combine tensors of DIFFERENT (compatible) shapes in an element-wise operation without manually duplicating the smaller one to match — e.g. adding a single bias value to every element of a large matrix, where TensorFlow implicitly 'stretches' the smaller tensor's shape to match, so you don't write that repetition yourself. It's not just about matching dimension COUNT with no effect on the actual values — the entire point is enabling a real arithmetic operation to succeed and produce a sensibly-combined result. It's the OPPOSITE of only applying to already-identical shapes — broadcasting exists precisely FOR the different-shape case; identical-shape tensors never needed broadcasting to combine in the first place. And it has nothing to do with deploying a model across servers — that's an unrelated serving/infrastructure concept, not a tensor arithmetic mechanism."
  },
  {
    "d": "Computational Graphs",
    "q": "TensorFlow represents computation as a graph, with operations as NODES and tensors flowing between them as EDGES. What key capability does this explicit graph structure enable, beyond just running the math?",
    "o": [
      "The graph structure exists purely for visual documentation purposes in a UI and has zero effect on how computation or gradients actually happen",
      "Automatic differentiation — because every operation and its inputs/outputs are explicitly recorded in the graph, TensorFlow can trace back through it to compute gradients automatically for backpropagation, rather than you deriving and coding those derivatives by hand",
      "Representing computation as a graph makes it IMPOSSIBLE to run any operation on a GPU or TPU, restricting all graph-based computation to CPU only",
      "A computational graph can only ever contain a single operation total, making \"graph\" a misleading term with no actual multi-step structure involved"
    ],
    "a": [
      1
    ],
    "e": "Because every operation (node) and the tensors flowing into/out of it (edges) are explicitly recorded, TensorFlow has a complete map of exactly how the final output depends on every earlier value — which is precisely what it needs to automatically apply the chain rule backward through that graph and compute gradients for every parameter, without a human deriving and coding each derivative by hand. This is a real computational capability with functional consequences (autodiff, optimization, parallel execution, device placement), not merely a passive visualization with no effect on execution. It's also the opposite of restricting computation to CPU only — the graph structure is exactly what enables flexible DEVICE PLACEMENT across CPU, GPU, or TPU, since the framework can see the whole computation and assign pieces to whichever device fits. And a real computational graph represents an entire chain of MANY connected operations (that's the whole reason it's called a graph rather than a single node) — restricting it to one operation total would defeat its purpose."
  },
  {
    "d": "Reduction Operations",
    "q": "TensorFlow provides \"reduction\" operations like `tf.reduce_sum` and `tf.reduce_mean`. What does a reduction operation actually do to a tensor, as distinct from an element-wise operation?",
    "o": [
      "Reduction operations and element-wise operations are exactly the same thing, just under two different naming conventions with no distinction in output shape",
      "`tf.reduce_sum` and `tf.reduce_mean` can only be applied to a tensor's FIRST dimension, and are structurally incapable of operating on any other axis",
      "It AGGREGATES values across one or more dimensions, collapsing that dimension down (e.g. summing all values in a row reduces a row of numbers down to one number) — unlike an element-wise op, which keeps the original shape and just transforms each value independently",
      "A reduction operation produces an output tensor that is always LARGER in every dimension than its input, the opposite of what \"reduce\" implies"
    ],
    "a": [
      2
    ],
    "e": "A reduction operation collapses one or more axes by aggregating the values along them — summing (or averaging, etc.) across a dimension replaces that entire dimension's worth of values with a single number per remaining index, which is exactly why the OUTPUT has fewer dimensions (or a smaller size along the reduced axis) than the input; an element-wise operation, by contrast, produces an output with the SAME shape as the input, just with each individual value independently transformed (like squaring every element). Claiming the output is always LARGER directly contradicts what 'reduce' means — reduction operations shrink the relevant dimension, they don't grow it. Reduction and element-wise ops are genuinely different operation categories with different output shapes, not just two names for one behavior. And reduction operations can target ANY specified axis (via an `axis` argument), not just the first dimension — that flexibility is a core, commonly-used feature of these functions."
  },
  {
    "d": "The Keras Training Workflow",
    "q": "The standard 5-step Keras workflow is: prepare data → define model architecture → COMPILE (optimizer, loss, metrics) → fit (train) → evaluate. What does the `compile` step specifically set up, and why can't you skip straight from defining the model to calling `.fit()`?",
    "o": [
      "Compile configures HOW training will actually happen — which optimizer updates the weights, which loss function measures error, and which metrics to track — none of which exist yet right after just defining the model's layers; `.fit()` needs all three already specified to know what to actually optimize and measure",
      "`.fit()` can be called directly after defining the model with no compile step at all, since Keras assumes reasonable defaults for absolutely everything with zero configuration required",
      "Compile is the step that actually loads and preprocesses the raw training data into memory, a data-handling step unrelated to the optimizer or loss function",
      "Compile's only function is to check the model's code for Python syntax errors, with no relationship to training itself"
    ],
    "a": [
      0
    ],
    "e": "Defining a model's LAYERS only specifies its architecture/structure — it says nothing yet about HOW to train it: what algorithm updates the weights (the optimizer, e.g. Adam), what quantity to minimize (the loss function, appropriate to the problem type), and what to report on along the way (metrics, e.g. accuracy). `compile()` is exactly the step that locks in those three training-specific choices, which `.fit()` then needs already configured before it can run a single training step — skipping straight to `.fit()` with none of that specified leaves the training process with no defined objective or update rule. Compile has nothing to do with syntax-checking the model's code — that's a basic Python-level concern unrelated to what compile configures. `.fit()` cannot simply run with 'reasonable defaults for everything' in place of an explicit compile step — the optimizer/loss/metrics genuinely need to be specified (there's no silent default optimizer/loss Keras substitutes on your behalf). And data loading/preprocessing is a separate, EARLIER step in the workflow (data preparation) — compile deals with the model's training configuration, not with getting data into memory."
  },
  {
    "d": "Deployment Targets",
    "q": "TensorFlow offers several deployment paths: TensorFlow Serving, TensorFlow Lite, and TensorFlow.js. How do these three differ in their intended deployment TARGET?",
    "o": [
      "TensorFlow.js can only run inside a native mobile app and has no ability to run inside an actual web browser at all",
      "All three are exactly the same underlying runtime with only a different marketing name attached to each, deployable interchangeably with zero difference in target environment",
      "TensorFlow Serving is for high-performance model inference in SERVER environments; TensorFlow Lite is optimized for MOBILE/embedded devices with reduced model size; TensorFlow.js runs models directly in a WEB BROWSER or Node.js — three genuinely different runtime environments",
      "TensorFlow Lite is meant for large-scale SERVER deployments, while TensorFlow Serving is specifically optimized for small mobile devices — the two roles as commonly described, just swapped"
    ],
    "a": [
      2
    ],
    "e": "Each tool targets a genuinely different deployment environment with different constraints: TF Serving is built for production SERVER-side inference at scale (high throughput, versioned model serving), TF Lite specifically shrinks and optimizes models to fit MOBILE and embedded-device resource constraints (limited memory/compute), and TF.js is built to run a model directly inside a BROWSER or Node.js process (no server round-trip needed for inference) — picking the right one depends entirely on where the model actually needs to run. They are not interchangeable, same-runtime tools with different names — each has distinct engineering tradeoffs suited to its specific target environment. The roles in option 3 are simply swapped — TF Serving is the SERVER-scale tool and TF Lite is the MOBILE/embedded one, not the reverse. And TF.js's signature use case is explicitly running IN A BROWSER (or Node.js) — that's its primary distinguishing feature, not something it's incapable of."
  }
]
</script>
<div class="topic-quiz-mount"></div>
