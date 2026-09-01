# Deep Learning Practice

This doc uses PyTorch (`import torch, torch.nn as nn`) for nearly everything. Every snippet here was actually run and checked in this session, so you know it works as written.

There's one exception. This environment's TensorFlow install is broken — a real conflict between NumPy 2 and `ml_dtypes`. So the one Keras example near the end is syntax-correct, standard API, but it was never actually run here.

Each cluster builds on the one before it. Every cluster ends with a worked summary example that ties the pieces together.

**Six words this doc uses constantly.** Defined once here, in plain English. (The arithmetic behind them, worked by hand with real numbers, lives in the Neural Net Numericals topic. The geometry lives in `ds-fundamentals`.)

- **Tensor** — PyTorch's array type. A grid of numbers with a shape, like `[4, 10]` (4 rows, 10 columns). Inputs, weights, outputs — everything in PyTorch is a tensor.
- **Layer & activation** — a layer is one learned formula. `nn.Linear(10, 32)` takes 10 numbers in and produces 32 numbers out. An activation, like ReLU, sits between layers and adds a nonlinear "squish" — ReLU just zeroes out any negative number. Without activations, stacking layers doesn't actually help: the whole stack collapses mathematically into one single linear formula. The nonlinearity is what makes depth worth anything.
- **Loss** — one number that scores how wrong the model's current predictions are. Training is just repeatedly nudging the weights to make this number smaller.
- **Gradient / backpropagation** — the gradient tells you, for each weight, which direction to move it and how strongly, to reduce the loss. Backpropagation is the algorithm that computes the gradient for every weight, by walking the chain rule backward from the loss.
- **Batch & epoch** — a batch is the handful of examples processed together in one update step. It's the `4` in a `[4, 10]` input. An epoch is one full pass through the entire training set.
- **Learning rate (`lr`)** — how big each weight-nudge is. Too large, and training overshoots and diverges. Too small, and training crawls.

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
`super().__init__()` is not optional. `nn.Module`'s constructor sets up internal bookkeeping — parameter registration, submodule tracking — and that bookkeeping is what lets `.parameters()`, `.to(device)`, and `.state_dict()` all work correctly. Skip it, and every layer you assign won't be tracked as trainable at all. It's a silent, confusing bug: no error, just a model that never learns.

### 2. How do you write the training loop, and why does the order of these 5 lines matter?
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
PyTorch adds new gradients to whatever's already there, instead of overwriting them. That's deliberate — it's what makes gradient accumulation across micro-batches possible. But it also means: forget `zero_grad()`, and gradients from the last iteration silently sum with the new ones, corrupting every update after the first. That's why it has to come first, every single iteration.

### 3. That loop used a hand-rolled stand-in for real data. How do you build a proper `Dataset`/`DataLoader` instead?
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
`shuffle=True` matters specifically for training. Without it, the model sees data in the same fixed order every single epoch. If the data happens to be sorted by label or by time, the model can learn spurious patterns tied to that ordering, instead of the real signal. Keep `shuffle=False` for validation and test loaders — that way results stay reproducible and comparable, run to run.

### Summary example
A full minimal training step, start to finish. `TabularDataset` wraps raw NumPy arrays. `DataLoader(ds, batch_size=16, shuffle=True)` yields shuffled batches. Each batch runs through the exact 5-line sequence from question 2: `zero_grad()` → forward → loss → `backward()` → `step()`. The training loop's `shuffle=True` specifically stops the model from learning any accidental pattern in how the original data happened to be ordered.

---

> 🔗 **Hands-on reps:** [Code Drills 6 — CNNs](/topic/code-drills-deep-learning#cluster-3-cnns)

## Cluster 2 — Convolutional Networks

### 1. How do you compute a `Conv2d` layer's output shape, without guessing?
```python
conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
x = torch.randn(1, 3, 32, 32)     # [batch, channels, height, width]
out = conv(x)
print(out.shape)   # torch.Size([1, 16, 32, 32]) -- same H,W because padding=1 with kernel=3, stride=1
```
Here's what a convolution actually does, in one sentence: it slides a small window — the **kernel**, 3×3 here — across the image, and at each position it computes a weighted sum. Each of the 16 `out_channels` is its own separate learned window, producing its own "feature map." One might light up on vertical edges. Another might light up on a texture. Nothing here is hand-designed — all 16 windows are learned.

`stride` is how far the window jumps at each step. `padding` adds a border of zeros around the image, so the window can center itself on edge pixels too.

The output-shape formula is worth memorizing, not guessing:
```
output_size = floor((input_size + 2*padding - kernel_size) / stride) + 1
```
With `padding = (kernel_size-1)/2` and `stride=1`, as in the example above, the output size comes out exactly equal to the input size. This "same padding" pattern is worth recognizing on sight.

### 2. How do you stack several conv layers into a full CNN?
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
Why `AdaptiveAvgPool2d(1)` instead of `Flatten()` followed by a hardcoded `Linear` size? A plain `Flatten()` bakes in one specific spatial size — something like `32*8*8` — and that breaks the moment your input resolution changes. `AdaptiveAvgPool2d(1)` always collapses down to a fixed 1×1-per-channel output, no matter what spatial size comes in. That makes the classifier head genuinely resolution-independent.

### Summary example
The same `SimpleCNN` instance runs on both a 64×64 and a 128×128 input, no code change, no error. Here's how: the `Conv2d`/`BatchNorm2d`/`ReLU`/`MaxPool2d` stack from question 2 processes whatever spatial size arrives, using the predictable shape math from question 1. Then `AdaptiveAvgPool2d(1)` collapses whatever spatial size comes out of that stack down to a fixed 1×1-per-channel vector, before it ever reaches the `Linear` head. The `Linear` head is the one piece of this network that does require a fixed input size — and it's fully protected from ever seeing a variable one.

---

## Cluster 3 — Train vs. Eval Mode: BatchNorm and Dropout

### 1. Why does running inference on a single sample sometimes break, specifically with BatchNorm?
```python
model.train()   # BatchNorm uses CURRENT BATCH statistics during training
# ... training loop ...
model.eval()    # BatchNorm switches to using its stored RUNNING statistics at inference
with torch.no_grad():
    preds = model(torch.randn(1, 3, 64, 64))   # a batch of 1 -- would break BatchNorm's per-batch stats in train mode
```
First, what BatchNorm is actually for. It re-centers and rescales each layer's outputs, using that batch's mean and variance, so the numbers flowing between layers stay in a stable range. That stabilizes training, and speeds it up noticeably.

Here's the catch. In train mode, BatchNorm normalizes using the *current batch's* mean and variance. Now picture a batch size of 1 — a single real-time prediction. A "variance" computed from one number is meaningless.

That's exactly what `model.eval()` fixes. It switches BatchNorm to use the running statistics it accumulated during training, instead of the current batch's. That's what makes single-sample inference actually work.

### 2. Does `.eval()` also change Dropout's behavior?
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
Yes — the same switch handles it. Dropout randomly zeroes out units during training, on purpose. It's a regularization trick that stops the network from leaning too hard on any single unit. But at inference time, you want the full, deterministic network making the prediction — not a randomly-degraded version of it. `model.eval()` disables dropout too, automatically, scaling the remaining activations to compensate. One method call, two different layer types, both correctly switching behavior.

### Summary example
Forgetting `model.eval()` before a single-image inference call is a real, common bug, and it has two symptoms from one root cause. BatchNorm silently computes nonsense statistics from a batch of size 1. At the same time, Dropout is still randomly zeroing units that should all be active. Both problems get fixed by the exact same one-line call. That's why `model.eval()`, paired with `torch.no_grad()`, is the standard first line of any inference function — not optional, not a nice-to-have.

---

> 🔗 **Hands-on reps:** [Code Drills 6 — RNN/LSTM, and Tuning Them](/topic/code-drills-deep-learning#cluster-4-rnn-lstm-and-tuning-them-to-work-better)

## Cluster 4 — Recurrent Networks: RNN → LSTM → GRU

### 1. How do you build a basic RNN, and what actually is the "hidden state" it produces?
```python
rnn = nn.RNN(input_size=8, hidden_size=16, batch_first=True)
x = torch.randn(4, 5, 8)     # [batch=4, seq_len=5, input_size=8]
output, hidden = rnn(x)
print(output.shape)   # [4, 5, 16] -- the hidden state at EVERY time step
print(hidden.shape)    # [1, 4, 16] -- just the FINAL hidden state (1 = num_layers*num_directions)
```
The **hidden state** is the RNN's running summary of everything it has read so far. It's a 16-number vector — the `hidden_size` — that gets updated after each element of the sequence. It does double duty: it's both the network's memory and its per-step output.

`output` hands you that summary as it stood at *every single step*. `hidden` is just the last one.

One thing worth setting explicitly on the API: `batch_first=True`. PyTorch's RNN family defaults to `(seq_len, batch, features)` ordering — the opposite of almost every other PyTorch API's `(batch, ...)` convention. Setting `batch_first=True` avoids a very common shape-mismatch bug, the kind that shows up when RNN output feeds into a layer that expects batch-first tensors.

### 2. Why do plain RNNs specifically struggle with long sequences?
At each step, the hidden state gets multiplied by a weight matrix, then squashed by `tanh`. Repeat that over many time steps, and you get exactly the vanishing-gradient problem from `math-foundations-refresher.md`'s calculus section.

An RNN unrolled over `T` time steps is architecturally identical to a `T`-layer deep network, as far as backprop is concerned. So the same `0.25^T`-style vanishing-gradient math applies — just with `T` = sequence length, instead of `T` = network depth.

### 3. How does LSTM fix that — and what do its two states actually represent?
```python
lstm = nn.LSTM(input_size=8, hidden_size=16, batch_first=True)
x = torch.randn(4, 5, 8)
output, (h_n, c_n) = lstm(x)
print(output.shape)   # [4, 5, 16] -- hidden state at every step
print(h_n.shape)        # [1, 4, 16] -- final hidden state ("what to output now")
print(c_n.shape)        # [1, 4, 16] -- final cell state ("what to remember long-term")
```
The **cell state** (`c_n`) is built to flow across time steps with only minor, gated modifications. Structurally, that's similar to a residual connection's untouched path — except this one runs across *time*, instead of across *layers*.

The **hidden state** (`h_n`) is what actually gets used for output at each step.

Three learned gates — forget, input, output — control what gets written to the cell state, and what gets read from it. That gating is what lets gradients survive far more time steps than question 2's plain RNN can manage.

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
Remember it this way: a factory conveyor belt with three valves. The forget gate scrubs old info off the belt. The input gate drops new material on. The output gate decides how much of the belt to reveal as this step's answer. The belt itself never gets squashed through an activation function on its main path — and that's exactly why it survives far more time steps than a plain RNN's hidden state.

### 4. LSTM has 3 gates and 2 states. Is there a simpler gated architecture that trades some of that away for speed?
```python
gru = nn.GRU(input_size=8, hidden_size=16, batch_first=True)
output, h_n = gru(x)    # only ONE state (no separate cell state) -- simpler than LSTM
```
**GRU** merges LSTM's forget and input gates into one "update gate," and drops the separate cell state entirely. That means fewer parameters, and faster training. In practice, it often performs comparably to LSTM on many tasks.

LSTM is still the safer default for longer sequences, or whenever compute isn't the constraint. GRU is a reasonable first thing to try when training speed or model size matters more.

### Summary example
A 500-token sequence fed to a plain RNN, from question 1, suffers vanishing gradients by the time backprop reaches token 1 — question 2's `0.25^500`-style math, playing out for real. The same sequence through an LSTM, from question 3, survives. Why: the cell state's conveyor belt carries information across those 500 steps with only gated, minor modifications, instead of a fresh squashing multiplication at every single step. A GRU, from question 4, gets most of that same benefit, with a simpler and faster-to-train architecture. It's a reasonable first choice, before you know for sure the extra LSTM machinery is needed.

---

## Cluster 5 — Sequence Classification and Variable-Length Batches

### 1. How do you build a full sequence classifier — say, text sentiment — combining an embedding with an LSTM?
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
An **Embedding layer** is a learned lookup table. Each token ID — a word's integer index in the vocabulary — maps to a vector of `embed_dim` numbers. During training, the network adjusts those vectors so that tokens used in similar ways end up with similar vectors. That's the whole trick behind turning integer word IDs into something a network can actually do math on.

Given that, `padding_idx=0` on the Embedding layer matters. Token ID 0 is the padding token, used to make variable-length sequences a uniform batch shape. `padding_idx=0` keeps its embedding gradient at zero, always. Without it, the model wastes capacity learning a "meaning" for a token that's purely a structural placeholder — and that learned padding embedding can leak noise into the sequence representation.

### 2. Batches need padding to be uniform-shaped. How do you stop the LSTM from wasting compute — and corrupting output — on that padding?
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
Without `pack_padded_sequence`, the LSTM processes the padding tokens too, as if they were real sequence content. That wastes compute — and worse, it corrupts the final hidden state with signal from meaningless padding steps.

Packing tells the RNN exactly where each real sequence ends, so it stops updating that sequence's hidden state at the right point. `enforce_sorted=False` just means you don't have to manually sort the batch by length first.

### Summary example
A batch of 4 reviews, padded to length 20, but really 20, 15, 20, and 8 tokens long. Without packing, the 8-token review's hidden state gets contaminated by 12 steps of pure padding-embedding noise before it ever reaches classification. `pack_padded_sequence`, given the real `lengths` tensor, tells the LSTM to stop updating that sequence's state at step 8 exactly. So `h_n` for that review reflects only its real 8 tokens — a materially different, and correct, final hidden state feeding into the classifier head.

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
**Logits** are the raw, unbounded scores a model's last layer produces, before softmax or sigmoid turns them into probabilities. They can be any real number, positive or negative — bigger just means "more confident in this class."

"Expects raw logits, not probabilities" is worth memorizing precisely. These loss functions internally combine the activation step (softmax or sigmoid) and the loss computation into one numerically-stable operation. That's specifically to avoid the precision problems of computing something like `log(softmax(x))` as two separate steps — the same log-sum-exp reasoning from `math-foundations-refresher.md`. Pass in already-activated values instead, and you double-apply the normalization. The model still runs, without error. It just trains mis-scaled, silently.

### 2. How do you protect against exploding gradients — the opposite failure from Cluster 4's vanishing problem?
```python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```
This goes exactly between `backward()` and `step()`. Gradients have to be computed before they can be clipped, and clipping has to happen before the optimizer actually uses them.

`max_norm=1.0` rescales the whole gradient vector if its norm exceeds 1.0, preserving its direction while capping its magnitude. It's cheap insurance — especially for RNN/LSTM training from Cluster 4, where exploding gradients are a known real risk.

### 3. Beyond clipping, how do you make the learning rate itself adapt over training, instead of staying fixed?
```python
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
for epoch in range(50):
    # ... train one epoch ...
    scheduler.step()    # called once per EPOCH (not per batch) for this scheduler
```
Early in training, a larger learning rate moves you quickly through the coarse part of the loss landscape. Later, a smaller learning rate lets you converge carefully near a good minimum, without overshooting it. `CosineAnnealingLR` smoothly decays the learning rate along a cosine curve over `T_max` epochs — a common, effective default shape for that schedule.

### Summary example
Training an LSTM, from Cluster 4, on long sequences pulls all three techniques together. `CrossEntropyLoss` on raw logits computes the loss correctly and numerically stably. `clip_grad_norm_` sits between `backward()` and `step()`, specifically because LSTMs stay exploding-gradient-prone even with gated cell states. `CosineAnnealingLR` gradually reduces the learning rate, so early training's large, fast steps give way to late training's careful fine-tuning. Three independent safeguards, each solving a different specific failure mode — and commonly all used together, on the same training run.

---

## Cluster 7 — Saving, Devices, Transfer Learning, and Early Stopping

### 1. How do you save and reload a trained model correctly?
```python
torch.save(model.state_dict(), "model.pt")

model2 = MLP(in_features=10, hidden=32, out_features=2)   # must recreate the SAME architecture first
model2.load_state_dict(torch.load("model.pt"))
model2.eval()
```
Save `state_dict()` — just the learned weights — not `torch.save(model, ...)`, which saves the whole object. Saving the whole object pickles the exact class definition and code alongside the weights. That breaks the moment you refactor the model class even slightly, or try to load it in a different codebase. Saving just the weights, and re-instantiating the architecture from code, is the more robust, portable, officially-recommended pattern.

### 2. How do you move a model — and its data — onto a GPU when one's available, without breaking on a machine that doesn't have one?
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
xb = xb.to(device)
yb = yb.to(device)
```
Hardcoding `"cuda"` breaks the instant the code runs on a machine without a GPU — a laptop, a different server, a CI pipeline. Checking `torch.cuda.is_available()` first is the standard, portable way to write device-agnostic code. It just works, in both environments, with no code change.

### 3. For transfer learning specifically, how do you freeze a pretrained backbone and train only a new head?
**Transfer learning** means reusing a network already trained on a big dataset, and retraining only its last part on your own, smaller one. The early layers' learned features — edges, textures, shapes — transfer across tasks. So your small dataset only has to teach the final classification step. The **backbone** is that reused, feature-extracting bulk. The **head** is the small new output layer you actually train.
```python
model = SimpleCNN(num_classes=10)
for param in model.features.parameters():
    param.requires_grad = False       # frozen: no gradient computed, no update ever applied

optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
```
Passing every parameter to the optimizer, including the frozen ones, wouldn't actually break anything — frozen params have no gradient, so they'd never move anyway. But `filter(lambda p: p.requires_grad, ...)` still avoids one thing: the optimizer tracking momentum and variance state for parameters that will never update. It also makes the intent explicit right there in the code.

### 4. How do you implement early stopping manually, so training doesn't run past a model's best point?
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
Save the checkpoint inside the "improved" branch, not once at the end of training. Here's why: by the time patience runs out, the model has already been overfitting for `patience` epochs past its actual best point. Saving only when validation loss actually improves guarantees you keep the genuinely best-performing checkpoint — using `state_dict()`, question 1's saving pattern — instead of whatever the weights happened to be when the loop finally exited.

### Summary example
Fine-tuning a pretrained `SimpleCNN` on a new, smaller dataset, step by step: freeze `model.features` (question 3), train only the new head on `device` (question 2), and run early stopping (question 4), which checkpoints via `state_dict()` (question 1) every time validation loss improves. Four independent techniques, from four different questions — routinely combined in exactly this way for a realistic transfer-learning task.

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
`GlobalAveragePooling2D` here is the direct Keras analogue of Cluster 2's `AdaptiveAvgPool2d(1)`. Same purpose: collapse any spatial size down to one value per channel, so the `Dense` head doesn't hardcode a size that depends on input resolution.

---

## Practice Q&A (Self-Test)

**Q1. In the `MLP` example, what would go wrong if you forgot `super().__init__()` in the constructor?**
A: `nn.Module`'s constructor sets up parameter registration and submodule tracking. Skip it, and layers you assign in `__init__` won't be tracked as trainable parameters at all. `.parameters()`, `.to(device)`, and `.state_dict()` all silently fail to include them — a confusing bug, with no error message.

**Q2. Why must `optimizer.zero_grad()` be called before `loss.backward()` on every iteration of the training loop, and what breaks if you forget it?**
A: PyTorch adds new gradients to whatever's already there, by default, rather than overwriting them. That's what enables gradient accumulation across micro-batches. Forget `zero_grad()`, and gradients from the previous iteration silently sum with the new ones — corrupting every update after the first.

**Q3. Why is `shuffle=True` used for the training `DataLoader` but `shuffle=False` for validation/test loaders?**
A: Without shuffling, the model sees data in the same fixed order every epoch. That can let it learn spurious patterns tied to that ordering, especially if the data is sorted by label or time. Training benefits from shuffling for that reason. Validation and test loaders keep `shuffle=False` instead, so results stay reproducible and comparable, run to run.

**Q4. Given `nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)` applied to a `[1,3,32,32]` input, what is the output shape, and what formula gets you there without guessing?**
A: `output_size = floor((input_size + 2*padding - kernel_size) / stride) + 1` = `floor((32+2-3)/1)+1 = 32`. So the output is `[1, 16, 32, 32]`. This is the "same padding" pattern: `padding = (kernel_size-1)/2` with `stride=1` keeps H and W unchanged.

**Q5. In `SimpleCNN`, why does the model use `AdaptiveAvgPool2d(1)` instead of `Flatten()` followed by a hardcoded `Linear` input size?**
A: `Flatten()` bakes in one specific spatial size, like `32*8*8`. That breaks the moment input resolution changes. `AdaptiveAvgPool2d(1)` always collapses to a fixed 1x1-per-channel output, no matter what size comes in — which makes the classifier head resolution-independent. That's why the model works on both 64x64 and 128x128 inputs, with no modification.

**Q6. Why does skipping `model.eval()` before running inference on a single sample (batch size 1) cause a real, common bug specifically involving BatchNorm?**
A: In train mode, BatchNorm normalizes using the current batch's mean and variance. With a batch of 1, that "variance" is meaningless. `model.eval()` switches BatchNorm — and Dropout — to use accumulated running statistics from training instead. That's what makes single-sample inference actually work.

**Q7. What is the difference between an LSTM's hidden state (`h_n`) and cell state (`c_n`), and why does LSTM have two states where a plain RNN has only one?**
A: The cell state flows across time steps with only minor, gated modifications — structurally like a residual connection's untouched path, but across time instead of layers. The hidden state is what's actually used for output at each step. Three learned gates — forget, input, output — control what's written to and read from the cell state, letting gradients survive many more time steps than a plain RNN's repeated tanh multiplications allow.

**Q8. In `LSTMClassifier`, why is `padding_idx=0` passed to `nn.Embedding`, and what happens if you omit it?**
A: `padding_idx=0` zeroes the gradient for the padding token's embedding, since token ID 0 is purely a structural placeholder, used to make variable-length sequences a uniform batch shape. Without it, the model wastes capacity learning a "meaning" for padding, and that learned embedding can leak noise into the sequence representation.

**Q9. Why do `CrossEntropyLoss` and `BCEWithLogitsLoss` expect raw logits rather than already-softmaxed or sigmoided probabilities, and what's the practical risk of getting this wrong?**
A: These loss functions combine the activation step and the loss computation internally, in one numerically-stable operation. That specifically avoids the precision problems of computing something like `log(softmax(x))` as two separate steps. Pass in already-activated values instead, and you double-apply the normalization — the model still runs, no error, but it trains mis-scaled, silently.

**Q10. In the manual early-stopping loop, why is the checkpoint saved inside the `if val_loss < best_val_loss` branch rather than once at the end of training?**
A: By the time patience runs out, the model has already been overfitting for `patience` epochs past its best point. Saving only when validation loss actually improves guarantees the checkpoint kept on disk is the genuinely best-performing one — not whatever the weights happened to be when the loop finally exited.

---

## Video-Sourced Practice MCQs

A second practice set. These questions were checked against a real YouTube interview-prep video for this subject, then written up here as original multiple-choice questions — the source video mostly asked them as open-ended questions, so the wrong-answer options and their explanations below are original, written to match this hub's "explain every option" convention, not copied from the video. Click an answer, check it, and use "ask about this question" for anything that needs more explanation.

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
