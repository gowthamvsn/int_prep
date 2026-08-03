# Deep Learning Practice — Built as a Chain, Not a List

Primarily **PyTorch** (`import torch, torch.nn as nn`) — every snippet here was actually run and verified in this session, which is why the framework choice isn't arbitrary: this environment's TensorFlow install is broken (a real NumPy2/ml_dtypes conflict), so the one Keras equivalent noted at the end is syntax-correct standard API but unverified here. Each cluster is one continuous thread — every question inherits the answer before it, closing with a worked summary example.

---

> 🔗 **Hands-on reps:** [Code Drills 6 — Building & Training a Model](/topic/code-drills-deep-learning#cluster-2-building-training-a-model)

## Cluster 1 — Building and Training a Basic Network

### 1. How do you define a basic feedforward network?
```python
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_features, hidden, out_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_features),
        )
    def forward(self, x):
        return self.net(x)

model = MLP(in_features=10, hidden=32, out_features=2)
x = torch.randn(4, 10)     # batch of 4, 10 features each
out = model(x)
print(out.shape)           # torch.Size([4, 2])
```
`super().__init__()` is not optional: `nn.Module`'s constructor sets up the internal bookkeeping (parameter registration, submodule tracking) that lets `.parameters()`, `.to(device)`, and `.state_dict()` all work correctly — skip it and layers you assign won't be tracked as trainable parameters at all, a silent, confusing bug.

### 2. Given a defined model, how do you write the training loop, and why does the order of these 5 lines matter?
```python
import torch.optim as optim

optimizer = optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(5):
    for xb, yb in [(torch.randn(8,10), torch.randint(0,2,(8,)))]:  # stand-in for a real DataLoader
        optimizer.zero_grad()      # 1. clear old gradients FIRST
        logits = model(xb)          # 2. forward pass
        loss = criterion(logits, yb)  # 3. compute loss
        loss.backward()             # 4. backprop -- computes gradients
        optimizer.step()            # 5. apply the update using those gradients
```
PyTorch accumulates (ADDS to) gradients by default rather than overwriting them — deliberate, since it's what makes gradient accumulation across micro-batches possible — but it means forgetting `zero_grad()` silently sums gradients across iterations, corrupting every update after the first. That's why it must come first, every single iteration.

### 3. That loop used a hand-rolled stand-in for real data — how do you build a proper `Dataset`/`DataLoader` instead?
```python
from torch.utils.data import Dataset, DataLoader

class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

import numpy as np
ds = TabularDataset(np.random.randn(100, 10), np.random.randint(0, 2, 100))
loader = DataLoader(ds, batch_size=16, shuffle=True)
for xb, yb in loader:
    pass   # xb: [16, 10], yb: [16]
```
`shuffle=True` matters specifically for TRAINING: without it, the model sees data in the same fixed order every epoch, which can let it learn spurious patterns tied to that ordering (especially if the data is sorted by label or time). Keep `shuffle=False` for validation/test loaders instead, so results stay reproducible and comparable run to run.

### Summary example
A full minimal training step: `TabularDataset` wraps raw NumPy arrays, `DataLoader(ds, batch_size=16, shuffle=True)` yields shuffled batches, and each batch runs through the exact 5-line sequence from question 2 — `zero_grad()` → forward → loss → `backward()` → `step()` — with the training loop's `shuffle=True` specifically preventing the model from learning any accidental pattern in how the original data happened to be ordered.

---

> 🔗 **Hands-on reps:** [Code Drills 6 — CNNs](/topic/code-drills-deep-learning#cluster-3-cnns)

## Cluster 2 — Convolutional Networks

### 1. Before building a CNN, how do you compute a `Conv2d` layer's OUTPUT SHAPE, without guessing?
```python
conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
x = torch.randn(1, 3, 32, 32)     # [batch, channels, height, width]
out = conv(x)
print(out.shape)   # torch.Size([1, 16, 32, 32]) -- same H,W because padding=1 with kernel=3, stride=1
```
The formula worth memorizing, not guessing: `output_size = floor((input_size + 2*padding - kernel_size) / stride) + 1`. With `padding = (kernel_size-1)/2` and `stride=1` (as above), output size exactly equals input size — this "same padding" pattern is worth recognizing on sight.

### 2. Given predictable shapes per layer, how do you stack several into a full CNN?
```python
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)    # collapses ANY spatial size to 1x1 -- shape-agnostic
        self.fc = nn.Linear(32, num_classes)
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)

model = SimpleCNN()
out = model(torch.randn(2, 3, 64, 64))    # works for 64x64 input...
out2 = model(torch.randn(2, 3, 128, 128))  # ...AND 128x128, without changing the Linear layer
```
`AdaptiveAvgPool2d(1)` instead of `Flatten()` + a hardcoded `Linear` size: a plain `Flatten()` bakes in a specific spatial size (e.g., `32*8*8`), which breaks the moment input resolution changes — `AdaptiveAvgPool2d(1)` always collapses to a fixed 1×1-per-channel output regardless of input size, making the classifier head genuinely resolution-independent.

### Summary example
The same `SimpleCNN` instance runs on both a 64×64 and a 128×128 input with no code change and no error — the `Conv2d`/`BatchNorm2d`/`ReLU`/`MaxPool2d` stack (question 2) processes whatever spatial size arrives using the predictable shape math (question 1), and `AdaptiveAvgPool2d(1)` collapses whatever the resulting spatial size is down to a fixed 1×1-per-channel vector before the `Linear` head — the one piece of the network that DOES require a fixed input size, protected from ever seeing a variable one.

---

## Cluster 3 — Train vs. Eval Mode: BatchNorm and Dropout

### 1. Why does running inference on a single sample sometimes break, specifically involving BatchNorm?
```python
model.train()   # BatchNorm uses CURRENT BATCH statistics during training
# ... training loop ...
model.eval()    # BatchNorm switches to using its stored RUNNING statistics at inference
with torch.no_grad():
    preds = model(torch.randn(1, 3, 64, 64))   # a batch of 1 -- would break BatchNorm's per-batch stats in train mode
```
In train mode, BatchNorm normalizes using the CURRENT batch's mean/variance — with a batch size of 1 (a single real-time prediction), that "variance" is degenerate/meaningless. `model.eval()` switches BatchNorm to use accumulated RUNNING statistics from training instead, which is what makes single-sample inference work correctly at all.

### 2. Given that `.eval()` changes BatchNorm's behavior, does it affect anything else — like Dropout?
```python
class MLPWithDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 32)
        self.dropout = nn.Dropout(p=0.3)     # p = probability of ZEROING each unit, during training only
        self.fc2 = nn.Linear(32, 2)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)
```
Yes — the same switch. Randomly zeroing units is a regularization technique forcing the network not to rely too heavily on any single unit; at inference you want the FULL, deterministic network making the prediction, not a randomly-degraded version of it. `model.eval()` automatically disables dropout too (scaling activations appropriately instead) — one method call, two different layer types both correctly switching behavior.

### Summary example
Forgetting `model.eval()` before a single-image inference call is a real, common bug with two simultaneous symptoms from one root cause: BatchNorm silently computes nonsense statistics from a batch of size 1, AND Dropout is still randomly zeroing units that should all be active — both fixed by the exact same one-line call, which is exactly why `model.eval()` (paired with `torch.no_grad()`) is the standard, non-optional first line of any inference function.

---

> 🔗 **Hands-on reps:** [Code Drills 6 — RNN/LSTM, and Tuning Them](/topic/code-drills-deep-learning#cluster-4-rnn-lstm-and-tuning-them-to-work-better)

## Cluster 4 — Recurrent Networks: RNN → LSTM → GRU

### 1. How do you build a basic RNN, and what IS the "hidden state" it produces?
```python
rnn = nn.RNN(input_size=8, hidden_size=16, batch_first=True)
x = torch.randn(4, 5, 8)     # [batch=4, seq_len=5, input_size=8]
output, hidden = rnn(x)
print(output.shape)   # [4, 5, 16] -- the hidden state at EVERY time step
print(hidden.shape)    # [1, 4, 16] -- just the FINAL hidden state (1 = num_layers*num_directions)
```
`batch_first=True` is worth always setting explicitly: PyTorch's RNN family defaults to `(seq_len, batch, features)` ordering, the OPPOSITE of almost every other PyTorch API's `(batch, ...)` convention — setting it avoids a very common shape-mismatch bug when RNN output feeds into a layer expecting batch-first tensors.

### 2. Why do plain RNNs specifically struggle with LONG sequences?
Conceptually: at each step, the hidden state gets multiplied by a weight matrix and squashed by `tanh`. Repeated over many time steps, this is EXACTLY the vanishing-gradient chain-rule problem from `math-foundations-refresher.md`'s calculus section — an RNN unrolled over T time steps is architecturally identical to a T-layer deep network for backprop purposes, so the same `0.25^T`-style vanishing-gradient math applies, just with T = sequence length instead of T = network depth.

### 3. Given that vanishing gradients are the problem, how does LSTM fix it — what do its TWO states actually represent?
```python
lstm = nn.LSTM(input_size=8, hidden_size=16, batch_first=True)
x = torch.randn(4, 5, 8)
output, (h_n, c_n) = lstm(x)
print(output.shape)   # [4, 5, 16] -- hidden state at every step
print(h_n.shape)        # [1, 4, 16] -- final hidden state ("what to output now")
print(c_n.shape)        # [1, 4, 16] -- final cell state ("what to remember long-term")
```
The **cell state** (`c_n`) is designed to flow across time steps with only minor, gated modifications — structurally similar to a residual connection's "untouched path," but across TIME instead of across LAYERS — while the **hidden state** (`h_n`) is what actually gets used for output at each step. Three learned gates (forget, input, output) control what gets written to/read from the cell state, letting gradients survive far more time steps than question 2's plain RNN.

**Visual + memory hook — the cell state as a conveyor belt running along the top, gates as three valves:**
```
c(t-1) ──────[×]──────────[+]──────────────▶ c(t)     ← the conveyor belt: mostly
                ▲            ▲                          just flows through, untouched
              forget       input                        unless a gate says otherwise
              gate         gate  ┐
                ▲            ▲   │ new candidate
                │            │   │ info to maybe
           x(t)+h(t-1) ──────┴───┘ add
                │
              output
              gate ──▶ h(t)   ← this step's actual output,
                                 a FILTERED read of the belt
```
**Remember it as a factory conveyor belt with three valves:** the forget gate scrubs old info off the belt, the input gate drops new material on, the output gate decides how much of the belt to reveal as this step's answer — the belt itself never gets squashed through an activation function on the main path, exactly why it survives far more time steps than a plain RNN's hidden state.

### 4. LSTM has 3 gates and 2 states. Is there a SIMPLER gated architecture that trades some of that away for speed?
```python
gru = nn.GRU(input_size=8, hidden_size=16, batch_first=True)
output, h_n = gru(x)    # only ONE state (no separate cell state) -- simpler than LSTM
```
**GRU** merges LSTM's forget/input gates into a single "update gate" and has no separate cell state — fewer parameters, faster to train, and in practice often performs comparably to LSTM on many tasks. LSTM is still the safer default for longer sequences or when compute isn't the constraint; GRU is a reasonable first thing to try when training speed or model size matters more.

### Summary example
A 500-token sequence fed to a plain RNN (question 1) suffers vanishing gradients by the time backprop reaches token 1 (question 2's `0.25^500`-style math). The same sequence through an LSTM (question 3) survives because the cell state's conveyor belt carries information across those 500 steps with only gated, minor modifications instead of a fresh squashing multiplication at every single step. A GRU (question 4) gets most of that same benefit with a simpler, faster-to-train architecture — the right first choice when it's not yet clear the extra LSTM machinery is needed.

---

## Cluster 5 — Sequence Classification and Variable-Length Batches

### 1. How do you build a full sequence classifier (e.g., text sentiment) combining an embedding with an LSTM?
```python
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
    def forward(self, x):
        emb = self.embedding(x)            # [batch, seq_len] -> [batch, seq_len, embed_dim]
        _, (h_n, _) = self.lstm(emb)
        return self.fc(h_n[-1])             # use the FINAL layer's hidden state for classification

model = LSTMClassifier(vocab_size=1000, embed_dim=32, hidden_dim=64, num_classes=2)
tokens = torch.randint(1, 1000, (4, 20))    # batch of 4 sequences, length 20
print(model(tokens).shape)                   # [4, 2]
```
`padding_idx=0` on the Embedding layer matters: token ID 0 (the padding token used to make variable-length sequences a uniform batch shape) has its embedding gradient always zeroed — otherwise the model wastes capacity learning a "meaning" for a token that's purely a structural placeholder, and that learned padding embedding can leak noise into the sequence representation.

### 2. Given that batches need PADDING to be uniform-shaped, how do you stop the LSTM from wasting compute (and corrupting output) processing that padding as if it were real content?
```python
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

embedding = nn.Embedding(1000, 32, padding_idx=0)
lstm = nn.LSTM(32, 64, batch_first=True)
tokens = torch.randint(1, 1000, (4, 20))     # 4 sequences, padded to length 20
emb = embedding(tokens)                        # [4, 20, 32]

lengths = torch.tensor([20, 15, 20, 8])   # real length of each of the 4 sequences before padding
packed = pack_padded_sequence(emb, lengths, batch_first=True, enforce_sorted=False)
packed_out, (h_n, c_n) = lstm(packed)
output, _ = pad_packed_sequence(packed_out, batch_first=True)
```
Without `pack_padded_sequence`, the LSTM processes the PADDING tokens too, as if they were real sequence content — wasting compute and, more importantly, corrupting the final hidden state with signal from meaningless padding steps. Packing tells the RNN exactly where each real sequence ends, so it stops updating that sequence's hidden state at the right point. `enforce_sorted=False` means you don't have to manually sort the batch by length first.

### Summary example
A batch of 4 reviews, padded to length 20 but really 20/15/20/8 tokens long: without packing, the 8-token review's hidden state gets contaminated by 12 steps of pure padding-embedding noise before classification. `pack_padded_sequence` with the real `lengths` tensor tells the LSTM to stop updating that sequence's state at step 8 exactly, so `h_n` for that review reflects only its real 8 tokens — a materially different (and correct) final hidden state feeding into the classifier head.

---

## Cluster 6 — Loss Functions, Gradient Clipping, and Learning Rate Scheduling

> Everything in this cluster is API mechanics — how to call the right loss/scheduler correctly. For the actual arithmetic underneath (forward pass, loss, backprop, and weight update computed by hand with real numbers), see the **Neural Net Numericals** topic.

### 1. How do you choose and correctly configure a loss function for classification?
```python
# multi-class (mutually exclusive classes): CrossEntropyLoss expects RAW LOGITS, not softmax output
criterion_multi = nn.CrossEntropyLoss()
logits = torch.randn(4, 3)             # [batch, num_classes] -- 3 classes
targets = torch.tensor([0, 2, 1, 1])    # class INDICES, not one-hot
loss = criterion_multi(logits, targets)

# binary: BCEWithLogitsLoss also expects raw logits, NOT sigmoid output
criterion_binary = nn.BCEWithLogitsLoss()
logits_binary = torch.randn(4, 1)
targets_binary = torch.randint(0, 2, (4, 1)).float()   # must be float, not long, for BCE
```
"Expects raw logits, not probabilities" is worth memorizing precisely: these loss functions internally combine the activation (softmax/sigmoid) and the loss computation in one numerically-stable operation, specifically to avoid the precision problems of computing something like `log(softmax(x))` as two separate steps — the same log-sum-exp reasoning from `math-foundations-refresher.md`. Passing already-activated values in double-applies the normalization and silently trains a mis-scaled model that still runs without error.

### 2. Given a computed loss, how do you protect against EXPLODING gradients specifically (the opposite failure from Cluster 4's vanishing problem)?
```python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```
This goes exactly between `backward()` and `step()`: gradients must already be computed before they can be clipped, and clipping must happen before the optimizer actually uses them. `max_norm=1.0` rescales the WHOLE gradient vector if its norm exceeds 1.0, preserving direction while capping magnitude — cheap insurance, especially for RNN/LSTM training (Cluster 4) where exploding gradients are a known real risk.

### 3. Beyond protecting against exploding gradients, how do you make the learning rate itself adapt over training instead of staying fixed?
```python
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
for epoch in range(50):
    # ... train one epoch ...
    scheduler.step()    # called once per EPOCH (not per batch) for this scheduler
```
A larger LR early in training moves quickly through the initial, coarse part of the loss landscape; a smaller LR later allows fine-grained convergence near a good minimum without overshooting it — `CosineAnnealingLR` smoothly decays the LR following a cosine curve over `T_max` epochs, a common, effective default schedule shape.

### Summary example
Training an LSTM (Cluster 4) on long sequences: `CrossEntropyLoss` on raw logits computes the loss correctly and numerically stably; `clip_grad_norm_` sits between `backward()` and `step()` specifically because LSTMs remain exploding-gradient-prone even with gated cell states; `CosineAnnealingLR` gradually reduces the learning rate so early-training's large, fast steps give way to late-training's careful fine-tuning — three independent safeguards, each solving a different specific failure mode, commonly all used together on the same training run.

---

## Cluster 7 — Saving, Devices, Transfer Learning, and Early Stopping

### 1. How do you save and reload a trained model correctly?
```python
torch.save(model.state_dict(), "model.pt")

model2 = MLP(in_features=10, hidden=32, out_features=2)   # must recreate the SAME architecture first
model2.load_state_dict(torch.load("model.pt"))
model2.eval()
```
Save `state_dict()` (just the learned weights), not `torch.save(model, ...)` (the whole object) — saving the whole object pickles the exact class definition/code alongside the weights, which breaks if you refactor the model class even slightly later, or try to load it in a different codebase. Saving just the weights and re-instantiating the architecture in code is the more robust, portable, officially-recommended pattern.

### 2. Given a saved model, how do you move it (and data) onto a GPU when available, without breaking on machines that don't have one?
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
xb = xb.to(device)
yb = yb.to(device)
```
Hardcoding `"cuda"` breaks the exact moment the code runs on a machine without a GPU (a laptop, a different server, a CI pipeline) — checking `torch.cuda.is_available()` first is the standard, portable way to write device-agnostic code that "just works" in both environments without a code change.

### 3. For transfer learning specifically, how do you freeze a pretrained backbone and train only a new head?
```python
model = SimpleCNN(num_classes=10)
for param in model.features.parameters():
    param.requires_grad = False       # frozen: no gradient computed, no update ever applied

optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
```
Passing ALL parameters to the optimizer (including frozen ones) doesn't break anything functionally — frozen params have no gradient so they'd never actually move — but `filter(lambda p: p.requires_grad, ...)` still avoids the optimizer tracking momentum/variance state for parameters that will never update, and makes the intent explicit in the code.

### 4. Given a training loop that could overfit past its best point, how do you implement early stopping manually?
```python
best_val_loss = float("inf")
patience, patience_counter = 5, 0

for epoch in range(100):
    train_loss = 0.0  # ... compute real train_loss ...
    val_loss = 0.0     # ... compute real val_loss ...
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "best_model.pt")   # always keep the BEST checkpoint, not the last
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"early stopping at epoch {epoch}")
            break
```
Save the checkpoint INSIDE the "improved" branch, not once at the end of training: by the time patience runs out, the model has already been overfitting for `patience` epochs past its best point. Saving only when validation loss actually improves guarantees you keep the genuinely best-performing checkpoint (using `state_dict()`, question 1's saving pattern), not whatever the weights happened to be when the loop finally exited.

### Summary example
Fine-tuning a pretrained `SimpleCNN` on a new, smaller dataset: freeze `model.features` (question 3), train only the new head on `device` (question 2), running early stopping (question 4) that checkpoints via `state_dict()` (question 1) every time validation loss improves — four independent techniques from four different questions, routinely combined in exactly this combination for a realistic transfer-learning task.

### Keras equivalent for the CNN above (standard API, unverified in this session's broken-TF environment)
```python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Input(shape=(64, 64, 3)),
    layers.Conv2D(16, 3, padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2),
    layers.GlobalAveragePooling2D(),   # Keras' name for the same "adaptive pool" idea used above
    layers.Dense(10, activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
```
`GlobalAveragePooling2D` here is the direct Keras analogue of Cluster 2's `AdaptiveAvgPool2d(1)` — same purpose, collapsing any spatial size to one value per channel so the `Dense` head doesn't hardcode an input-resolution-dependent size.

---

## Practice Q&A (Self-Test)

**Q1. In the `MLP` example, what would go wrong if you forgot `super().__init__()` in the constructor?**
A: `nn.Module`'s constructor sets up parameter registration and submodule tracking. Without it, layers you assign in `__init__` won't be tracked as trainable parameters at all, so `.parameters()`, `.to(device)`, and `.state_dict()` silently fail to include them — a confusing bug with no error message.

**Q2. Why must `optimizer.zero_grad()` be called before `loss.backward()` on every iteration of the training loop, and what breaks if you forget it?**
A: PyTorch accumulates (adds to) gradients by default rather than overwriting them, which is what enables gradient accumulation across micro-batches. Forgetting `zero_grad()` means gradients from the previous iteration silently sum with the new ones, corrupting every update after the first.

**Q3. Why is `shuffle=True` used for the training `DataLoader` but `shuffle=False` for validation/test loaders?**
A: Without shuffling, the model sees data in the same fixed order every epoch, which can let it learn spurious patterns tied to that ordering (especially if data is sorted by label or time). Training benefits from shuffling for that reason, while validation/test loaders keep `shuffle=False` so results are reproducible and comparable run to run.

**Q4. Given `nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)` applied to a `[1,3,32,32]` input, what is the output shape, and what formula gets you there without guessing?**
A: `output_size = floor((input_size + 2*padding - kernel_size) / stride) + 1` = `floor((32+2-3)/1)+1 = 32`, so the output is `[1, 16, 32, 32]`. This is the "same padding" pattern: `padding = (kernel_size-1)/2` with `stride=1` keeps H and W unchanged.

**Q5. In `SimpleCNN`, why does the model use `AdaptiveAvgPool2d(1)` instead of `Flatten()` followed by a hardcoded `Linear` input size?**
A: `Flatten()` bakes in a specific spatial size (e.g. `32*8*8`), which breaks the moment input resolution changes. `AdaptiveAvgPool2d(1)` always collapses to a fixed 1x1-per-channel output regardless of input size, making the classifier head resolution-independent — the model works on both 64x64 and 128x128 inputs without modification.

**Q6. Why does skipping `model.eval()` before running inference on a single sample (batch size 1) cause a real, common bug specifically involving BatchNorm?**
A: In train mode, BatchNorm normalizes using the current batch's mean/variance, and with a batch of 1 that "variance" is degenerate/meaningless. `model.eval()` switches BatchNorm (and Dropout) to use accumulated running statistics from training instead, which is what makes single-sample inference work correctly.

**Q7. What is the difference between an LSTM's hidden state (`h_n`) and cell state (`c_n`), and why does LSTM have two states where a plain RNN has only one?**
A: The cell state is designed to flow across time steps with only minor, gated modifications — structurally like a residual connection's untouched path, but across time instead of layers — while the hidden state is what's actually used for output at each step. Three learned gates (forget, input, output) control what's written to/read from the cell state, letting gradients survive many more time steps than a plain RNN's repeated tanh multiplications allow.

**Q8. In `LSTMClassifier`, why is `padding_idx=0` passed to `nn.Embedding`, and what happens if you omit it?**
A: `padding_idx=0` zeroes the gradient for the padding token's embedding, since token ID 0 is purely a structural placeholder used to make variable-length sequences a uniform batch shape. Without it, the model wastes capacity learning a "meaning" for padding, and that learned embedding can leak noise into the sequence representation.

**Q9. Why do `CrossEntropyLoss` and `BCEWithLogitsLoss` expect raw logits rather than already-softmaxed/sigmoided probabilities, and what's the practical risk of getting this wrong?**
A: These loss functions internally combine the activation and the loss computation in one numerically-stable operation, specifically to avoid the precision problems of computing something like `log(softmax(x))` as two separate steps. Passing already-activated values in double-applies the normalization and silently trains a mis-scaled model that still runs without throwing an error.

**Q10. In the manual early-stopping loop, why is the checkpoint saved inside the `if val_loss < best_val_loss` branch rather than once at the end of training?**
A: By the time patience runs out, the model has already been overfitting for `patience` epochs past its best point. Saving only when validation loss actually improves guarantees the checkpoint kept on disk is the genuinely best-performing one, not whatever the weights happened to be when the loop finally exited.

---

## Video-Sourced Practice MCQs

A second practice set for Deep Learning Practice, built the same way as this hub's NCA-GENL community bank: topics checked against a real YouTube interview-prep video for this subject, then written up as original multiple-choice questions here (the source video mostly asked these as open-ended questions -- the wrong-answer options and their explanations below are original, written to match this hub's "explain every option" convention, not copied from the video). Click an answer, check it, and use "ask about this question" for anything that needs more explanation.

<script type="application/json" class="topic-quiz-data" data-title="Deep Learning Practice">
[
  {
    "d": "Fundamentals",
    "q": "Why do neural networks need activation functions at all -- what would happen without them?",
    "o": [
      "Activation functions only affect training speed, never what the network can represent",
      "Without any nonlinear activation function, stacking any number of layers would still collapse mathematically into one simple linear function, unable to learn complex, non-linear patterns",
      "Activation functions are only needed in the very last layer, never in hidden layers",
      "Nothing would change; activation functions are purely optional stylistic choices"
    ],
    "a": [
      1
    ],
    "e": "They're not optional stylistic flourishes -- removing them changes what the network is mathematically capable of representing, not just its 'style.' It's not just a speed issue -- this is a REPRESENTATIONAL limitation: a network of purely linear layers, no matter how deep, is mathematically equivalent to one single linear layer, which can't capture curves, thresholds, or complex decision boundaries. Hidden layers need nonlinearity just as much as -- often more than -- the output layer, since it's the hidden layers' repeated nonlinear transformations that let deep networks build up complex representations layer by layer. The core reason: without a nonlinear activation function (like ReLU, sigmoid, or tanh) between layers, composing many linear layers together is mathematically identical to just one linear layer -- nonlinearity is what actually gives depth its power to model complex, real-world patterns."
  },
  {
    "d": "Fundamentals",
    "q": "What does the backpropagation algorithm actually compute, and in what direction does it move through the network?",
    "o": [
      "It only updates the very first layer's weights, ignoring the rest",
      "It computes how much each weight contributed to the total error, propagating that gradient information BACKWARD from the output layer toward the input layer",
      "It randomly reinitializes all weights, moving in no particular direction",
      "It computes the model's final predictions, moving forward from input to output"
    ],
    "a": [
      1
    ],
    "e": "Computing predictions by moving forward through the network describes the FORWARD pass -- a separate, earlier step that happens before backpropagation, not backpropagation itself. Random reinitialization describes weight initialization, a one-time setup step before training begins -- utterly different from the repeated, directed gradient computation backprop performs every training step. Updating only the first layer would make deep networks untrainable beyond one layer -- backprop's entire value is that it correctly updates EVERY layer's weights, all the way through the network. Backpropagation's real job: after the forward pass produces a prediction and the loss (error) is computed, backprop uses the chain rule to work BACKWARD from the output layer toward the input layer, calculating exactly how much each individual weight contributed to that error -- which is the gradient each weight then gets updated with."
  },
  {
    "d": "CNN",
    "q": "In a Convolutional Neural Network, what is the distinct job of a convolutional layer versus a pooling layer?",
    "o": [
      "They do the exact same thing; the two names are interchangeable",
      "The pooling layer detects features; the convolutional layer only resizes images",
      "Both layers exist purely to increase the number of parameters in the model",
      "The convolutional layer slides filters over the input to detect features (edges, textures) and produce feature maps; the pooling layer then downsamples those feature maps (e.g. via max pooling) to reduce spatial size and add translation invariance"
    ],
    "a": [
      3
    ],
    "e": "They are not interchangeable -- a network built ONLY from convolutional layers with no pooling would have a very different (and much larger) computational footprint, since pooling's whole purpose is to shrink the feature maps down. The roles given in that option are swapped -- feature DETECTION is the convolutional layer's job (via learned filters/kernels), while resizing/downsampling is what pooling does, not the other way around. Neither layer's purpose is to inflate parameter count -- pooling in particular has FEWER (often zero learnable) parameters, and exists specifically to reduce computation, the opposite of \"increasing parameters.\" The real division of labor: convolutional layers apply learned filters across the input to detect specific local patterns, producing feature maps that highlight where those patterns occur; pooling layers then downsample those feature maps (e.g., max pooling keeps only the strongest activation in each region), cutting computational cost and making the network less sensitive to small shifts in exactly where a feature appears."
  },
  {
    "d": "RNN/LSTM/GRU",
    "q": "What is the single defining architectural feature that lets a Recurrent Neural Network (RNN) process sequential data (like text or time series) at all?",
    "o": [
      "RNNs use convolutional filters to scan across the sequence",
      "RNNs process the entire sequence in one single parallel pass, like a transformer",
      "RNNs maintain an internal hidden state that carries information from previous time steps forward, so each step's output depends on both the current input AND that accumulated history",
      "RNNs require every input sequence to be exactly the same fixed length"
    ],
    "a": [
      2
    ],
    "e": "Convolutional filters are a CNN's tool for spatial pattern detection in grid-like data (images) -- RNNs use an entirely different recurrent mechanism suited to sequences, not convolution. Processing the whole sequence in one parallel pass describes a transformer's self-attention approach, which was actually developed partly BECAUSE RNNs can't do this -- RNNs are inherently sequential, one step at a time, not parallel. Requiring a fixed sequence length is false -- one of RNNs' real strengths is handling VARIABLE-length sequences, since the same recurrent step just repeats as many times as the sequence requires. The defining trait: an RNN carries a hidden state forward from one time step to the next, so when processing token/step N, the network has access to accumulated information from steps 1 through N-1 as well as the current input -- this internal memory is exactly what makes sequential, context-aware processing possible."
  },
  {
    "d": "RNN/LSTM/GRU",
    "q": "What problem do LSTMs (Long Short-Term Memory networks) solve relative to a vanilla (basic) RNN, and how?",
    "o": [
      "LSTMs can only process fixed-length sequences, unlike vanilla RNNs",
      "LSTMs make training slower on purpose, to force better generalization",
      "LSTMs solve the vanishing-gradient problem over long sequences by using gates (forget, input, output) and a separate cell state to control what long-term information is kept, updated, or discarded",
      "LSTMs remove the need for a hidden state entirely"
    ],
    "a": [
      2
    ],
    "e": "Slower training isn't an intentional design goal, and isn't really what distinguishes LSTMs -- their added gating logic does add some computation, but that's a side effect, not the point. LSTMs still use a hidden state -- they ADD a separate cell state alongside it, they don't remove the hidden state concept. If anything, LSTMs are typically praised for handling variable AND longer sequences better than vanilla RNNs, not for imposing a fixed-length restriction -- that claim has it backwards. The real fix: vanilla RNNs struggle to retain information over many time steps because gradients shrink toward zero as they're propagated back through a long sequence (the vanishing gradient problem); LSTMs address this with three gates (forget, input, output) plus a dedicated cell state that acts as a more stable long-term memory channel, letting relevant information persist across many more steps than a vanilla RNN can manage."
  },
  {
    "d": "Transformers",
    "q": "In the transformer attention formula `softmax(Q @ K^T / sqrt(d_k)) @ V`, what do Q, K, and V represent, and what is that formula computing?",
    "o": [
      "Q (query) represents what a position is looking for, K (key) represents what each position offers, and V (value) is the actual content -- the formula computes attention scores from Q-K similarity, then uses those scores to compute a weighted sum of the V vectors",
      "Q, K, and V all represent the exact same vector, just renamed three times for clarity",
      "This formula only applies to image data, never to text",
      "Q, K, V are three unrelated random matrices with no specific meaning"
    ],
    "a": [
      0
    ],
    "e": "They aren't random or meaningless -- Q, K, and V are learned projections of the input embeddings, each serving a specific, named role in the attention mechanism, not arbitrary matrices. They are NOT the same vector renamed -- Q, K, and V are produced by three separate learned weight matrices applied to the input, precisely so they can capture different aspects (what's being asked for vs. what's on offer vs. the actual content to retrieve). This formula is the foundation of the original transformer architecture built for TEXT (machine translation, in the original paper) -- it has since been adapted to images too (e.g. Vision Transformers), but text/sequence modeling is its original and primary use case, not an exclusion. The real mechanics: Q (query) asks 'what am I looking for,' K (key) answers 'what do I contain,' their dot product (scaled by sqrt(d_k) to stabilize softmax) produces attention scores measuring relevance, and softmax turns those into weights that combine the V (value) vectors into a context-aware output for each position."
  },
  {
    "d": "Fundamentals",
    "q": "What problem do residual (skip) connections, as used in ResNets, solve, and how?",
    "o": [
      "They allow gradients to flow directly to earlier layers via a shortcut path, which addresses the vanishing-gradient problem and makes it practical to train networks with very many (e.g. 100+) layers",
      "They make the network use less memory by skipping layers entirely during inference",
      "They eliminate the need for activation functions anywhere in the network",
      "They are only useful for reducing a model's file size on disk"
    ],
    "a": [
      0
    ],
    "e": "Skip connections don't skip layers during actual inference computation -- the layers still run; the 'skip' is an ADDITIONAL shortcut path for the gradient/signal, added alongside the normal layer path, not a way to avoid computing layers. They don't remove the need for activation functions -- those are still applied within the normal path; residual connections address a different problem (gradient flow across depth), not nonlinearity. Model file size isn't the target either -- residual connections can even add a small number of parameters/connections, and their benefit is about trainability at depth, not storage efficiency. The actual mechanism: a residual connection adds a direct shortcut path around one or more layers, so the gradient during backpropagation has an unobstructed route back to earlier layers instead of only passing through (and potentially shrinking across) every intermediate layer -- this is specifically what made training networks with hundreds of layers practically feasible, where vanilla deep networks would otherwise suffer severe vanishing gradients."
  }
]
</script>
<div class="topic-quiz-mount"></div>
