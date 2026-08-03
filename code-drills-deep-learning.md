# Code Drills — Tier 2: PyTorch — Tensors, Training Loops, CNNs, LSTM Tuning

Continues `code-drills-classical-ml.md` and closes the loop the user asked for directly: train/eval a CNN, and tune an LSTM's hyperparameters to make it train more efficiently. Terser companion to `pytorch-deep-dive.md` and `deep-learning-practice.md` (read those for the conceptual "why" — backprop math, vanishing gradients, etc.); `module-cheatsheet.md` has the same calls as a flat lookup. Drill #15 of `code-drills-oop-intermediate.md` is the OOP prerequisite for everything here — every model below is just that `__call__`-wraps-`forward` pattern with real layers inside. All snippets verified against installed torch 2.7.0+cu118 (CPU).

---

## Cluster 1 — Tensors & Autograd

> 🔗 **Theory:** [PyTorch Deep Dive — Autograd Internals](/topic/practice-pytorch-deep#cluster-1-autograd-internals-custom-functions-and-hooks)

**1. Create a tensor from a Python list or a NumPy array.**
```python
import torch
t1 = torch.tensor([1.0, 2.0, 3.0])
t2 = torch.from_numpy(np.array([1.0, 2.0, 3.0]))    # shares memory with the numpy array — mutating one affects the other
```

**2. Inspect a tensor's shape, dtype, and device.**
```python
t = torch.zeros(2, 3)
t.shape       # torch.Size([2, 3])
t.dtype        # torch.float32 — PyTorch's default float type
t.device        # device(type='cpu') — where the data actually lives
```

**3. Do elementwise math and matrix multiplication.**
```python
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])
a + b          # elementwise: [5., 7., 9.]
a @ b           # dot product (1D @ 1D): 32.0
m = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
m @ m           # matrix multiplication, same `@` as NumPy
```

**4. Reshape a tensor without changing its data.**
```python
t = torch.arange(6)
t.view(2, 3)       # [[0,1,2],[3,4,5]] — requires the underlying memory to be contiguous
t.reshape(2, 3)     # same result, but works even on non-contiguous tensors (copies if it must)
```

**5. Write device-agnostic code that runs on GPU if available, CPU otherwise.**
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)      # move the MODEL's parameters
x = x.to(device)               # move the DATA — both must be on the SAME device or ops will raise an error
```

**6. Track gradients through a computation with `requires_grad`, then call `.backward()`.**
```python
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2 + 2 * x            # y = x^2 + 2x, dy/dx = 2x + 2
y.backward()                    # computes the gradient and stores it on x
x.grad                          # tensor(8.) — 2*3 + 2 = 8, matches calculus exactly
```

**7. Read a gradient after backprop, on a real parameter.**
```python
w = torch.tensor([1.0, 2.0], requires_grad=True)
loss = (w ** 2).sum()
loss.backward()
w.grad    # tensor([2., 4.]) — d(loss)/dw = 2w
```

**8. Disable gradient tracking for inference — saves memory, since no backward pass is coming.**
```python
model.eval()
with torch.no_grad():
    preds = model(x)     # no computation graph built -> faster, less memory, but gradients unavailable
```

**9. Detach a tensor from the computation graph without disabling grad globally.**
```python
y = x ** 2
y_detached = y.detach()    # a new tensor, same data, but NOT connected to x's graph anymore
# useful when you want a value (e.g. for logging/plotting) without it affecting backprop
```

**10. Convert between a tensor and a NumPy array.**
```python
t = torch.tensor([1.0, 2.0, 3.0])
arr = t.numpy()                        # tensor -> numpy (must be on CPU, and not require grad)
t2 = torch.from_numpy(arr)             # numpy -> tensor
t3 = torch.tensor([1.0], requires_grad=True)
t3.detach().numpy()                     # detach() first if requires_grad=True, or .numpy() raises
```

**11. Know the difference between an in-place op and a regular one.**
```python
a = torch.tensor([1.0, 2.0])
a.add(1)          # returns a NEW tensor [2., 3.] — `a` itself is unchanged
a.add_(1)          # the trailing underscore means IN-PLACE — `a` itself becomes [2., 3.]
# in-place ops can break autograd if the modified tensor is needed for a gradient computation — use with care
```

**12. Broadcast tensors of different shapes, same rule as NumPy.**
```python
m = torch.ones(2, 3)
row = torch.tensor([1.0, 2.0, 3.0])
m + row    # row (shape (3,)) is broadcast across both rows of m (shape (2, 3))
```

---

## Cluster 2 — Building & Training a Model

> 🔗 **Theory:** [Deep Learning Practice — Building and Training a Basic Network](/topic/practice-deep-learning#cluster-1-building-and-training-a-basic-network)

**13. Define a small MLP as an `nn.Module` — the exact pattern from the OOP bridge drill, with real layers.**
```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_features, hidden, out_features):
        super().__init__()               # ALWAYS call this first — registers the layers below with PyTorch
        self.fc1 = nn.Linear(in_features, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, out_features)

    def forward(self, x):                 # PyTorch's nn.Module.__call__ invokes this for you — never call .forward() directly
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)

model = MLP(in_features=10, hidden=32, out_features=2)
```

**14. Build the same MLP faster with `nn.Sequential`, when there's no branching logic.**
```python
model = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 2),
)
# equivalent to drill #13, but you lose the ability to easily add conditional/branching forward logic
```

**15. Define a loss function and an optimizer.**
```python
criterion = nn.CrossEntropyLoss()               # for multi-class classification, expects raw logits (no softmax needed)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)   # .parameters() hands the optimizer every learnable weight
```

**16. Write the full training loop — the four-line core every PyTorch model repeats.**
```python
for epoch in range(10):
    optimizer.zero_grad()          # 1. clear old gradients — PyTorch ACCUMULATES grads by default
    outputs = model(X_train)        # 2. forward pass
    loss = criterion(outputs, y_train)
    loss.backward()                  # 3. backward pass — computes gradients for every parameter
    optimizer.step()                  # 4. apply the update — moves weights in the direction that reduces loss
```

**17. Wrap raw tensors in a `Dataset` + `DataLoader` for batching and shuffling.**
```python
from torch.utils.data import TensorDataset, DataLoader
dataset = TensorDataset(X_train, y_train)
loader = DataLoader(dataset, batch_size=32, shuffle=True)   # shuffle=True: reshuffled every epoch, reduces overfitting to order
```

**18. Write a custom `Dataset` for data that isn't already one big tensor (e.g. loaded per-file).**
```python
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y

    def __len__(self):                  # DataLoader calls this to know how many samples exist
        return len(self.X)

    def __getitem__(self, idx):          # DataLoader calls this to fetch ONE sample at a time
        return self.X[idx], self.y[idx]
```

**19. Loop over a `DataLoader` inside the training loop (the realistic version of drill #16).**
```python
for epoch in range(10):
    for X_batch, y_batch in loader:      # one mini-batch at a time, instead of the whole dataset at once
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
```

**20. Switch between train and eval mode correctly (matters for Dropout/BatchNorm).**
```python
model.train()      # Dropout active, BatchNorm uses batch statistics — for the training loop
model.eval()         # Dropout disabled, BatchNorm uses running statistics — for validation/inference
# forgetting eval() during validation silently makes your val metrics noisier/worse than they should be
```

**21. Save and load a model's weights.**
```python
torch.save(model.state_dict(), "model.pt")     # state_dict: an ordered dict of layer name -> weight tensor

model2 = MLP(in_features=10, hidden=32, out_features=2)   # must rebuild the SAME architecture first
model2.load_state_dict(torch.load("model.pt"))
model2.eval()
```

**22. Evaluate accuracy on a validation/test set.**
```python
model.eval()
correct = total = 0
with torch.no_grad():
    for X_batch, y_batch in val_loader:
        preds = model(X_batch).argmax(dim=1)    # pick the highest-scoring class per sample
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)
accuracy = correct / total
```

**23. Implement manual early stopping — stop training once validation loss stops improving.**
```python
best_val_loss, patience, bad_epochs = float("inf"), 3, 0
for epoch in range(100):
    train_one_epoch(model, loader, optimizer, criterion)
    val_loss = evaluate(model, val_loader, criterion)
    if val_loss < best_val_loss:
        best_val_loss, bad_epochs = val_loss, 0
        torch.save(model.state_dict(), "best.pt")   # only checkpoint when it actually improves
    else:
        bad_epochs += 1
        if bad_epochs >= patience:
            break     # stop — further training is very likely just overfitting from here
```

**24. Compute the total number of trainable parameters in a model.**
```python
sum(p.numel() for p in model.parameters() if p.requires_grad)
```

---

## Cluster 3 — CNNs

> 🔗 **Theory:** [Deep Learning Practice — Convolutional Networks](/topic/practice-deep-learning#cluster-2-convolutional-networks)

**25. Define a single Conv2d layer and understand its arguments.**
```python
conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
# in_channels=3: RGB input | out_channels=64: number of learned filters/feature maps produced
# kernel_size=3: each filter looks at a 3x3 patch | padding=1: keeps spatial size unchanged for a 3x3 kernel
```

**26. Compute the output spatial size after a conv layer.**
```python
# output_size = floor((input_size + 2*padding - kernel_size) / stride) + 1
# a 32x32 image through Conv2d(kernel_size=3, padding=1, stride=1):
# floor((32 + 2*1 - 3) / 1) + 1 = 32  -> same size in, same size out (that's what padding=1 buys you here)
```

**27. Downsample with max pooling.**
```python
pool = nn.MaxPool2d(kernel_size=2, stride=2)    # takes the max of each non-overlapping 2x2 patch
x = torch.randn(1, 64, 32, 32)                    # (batch, channels, height, width)
pool(x).shape                                       # torch.Size([1, 64, 16, 16]) — halves height and width
```

**28. Build a minimal CNN: conv -> relu -> pool -> flatten -> fully connected.**
```python
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(16 * 16 * 16, num_classes)   # 32x32 input -> pooled once -> 16x16 spatial

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))    # (batch, 16, 16, 16)
        x = x.view(x.size(0), -1)                      # flatten everything except the batch dimension
        return self.fc(x)
```

**29. Understand why `.view(x.size(0), -1)` is needed before the first `Linear` layer.**
```python
# nn.Linear expects a 2D input: (batch, features). Conv/pool output is 4D: (batch, channels, H, W).
# x.size(0) keeps the batch dimension untouched; -1 flattens channels*H*W into one long feature vector.
```

**30. Fine-tune a pretrained CNN backbone instead of training from scratch (transfer learning).**
```python
import torchvision.models as models
backbone = models.resnet18(weights="IMAGENET1K_V1")    # pretrained on 1.2M ImageNet images
backbone.fc = nn.Linear(backbone.fc.in_features, 10)     # replace the final layer for YOUR number of classes
# only the new fc layer starts randomly initialized — everything before it already knows general image features
```

**31. Augment training images to reduce overfitting.**
```python
from torchvision import transforms
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),                 # converts PIL image -> tensor, scales pixels to [0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),   # ImageNet stats
])
```

**32. Load an image dataset from a folder structure (`class_name/image.jpg`).**
```python
from torchvision.datasets import ImageFolder
dataset = ImageFolder("data/train", transform=train_transform)   # folder name becomes the class label automatically
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

**33. Add batch normalization to stabilize/speed up CNN training.**
```python
nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, padding=1),
    nn.BatchNorm2d(16),     # normalizes each channel's activations across the batch — argument matches out_channels
    nn.ReLU(),
)
```

**34. Add dropout to a CNN's fully connected head to fight overfitting.**
```python
self.head = nn.Sequential(
    nn.Linear(512, 128),
    nn.ReLU(),
    nn.Dropout(p=0.5),        # randomly zeroes 50% of activations during training only (inactive in eval())
    nn.Linear(128, 10),
)
```

---

## Cluster 4 — RNN / LSTM, and Tuning Them to Work Better

> 🔗 **Theory:** [Deep Learning Practice — RNN → LSTM → GRU](/topic/practice-deep-learning#cluster-4-recurrent-networks-rnn-lstm-gru)

**35. Define an LSTM layer and understand its constructor arguments.**
```python
lstm = nn.LSTM(input_size=100, hidden_size=128, num_layers=2, batch_first=True)
# input_size: dimensionality of EACH element in the sequence (e.g. embedding size)
# hidden_size: size of the internal memory/state — the model's "capacity" per layer
# num_layers=2: stacks two LSTMs, second layer's input is the first layer's output sequence
# batch_first=True: input/output shaped (batch, seq_len, features) instead of (seq_len, batch, features)
```

**36. Know the exact input/output shapes an LSTM expects and returns.**
```python
x = torch.randn(32, 10, 100)     # (batch=32, seq_len=10, input_size=100)
output, (h_n, c_n) = lstm(x)
output.shape    # (32, 10, 128) — hidden state at EVERY timestep, last layer only
h_n.shape        # (2, 32, 128)  — FINAL hidden state, one per layer (num_layers=2 here)
c_n.shape        # (2, 32, 128)  — FINAL cell state, same shape as h_n
```

**37. Build an LSTM-based sequence classifier — use the LAST timestep's output for a single prediction.**
```python
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        last_hidden = output[:, -1, :]     # take the LAST timestep -> (batch, hidden_size)
        return self.fc(last_hidden)
```

**38. Understand what the hidden state (`h`) and cell state (`c`) actually represent.**
```python
# h (hidden state): the "working memory" passed to the next timestep AND used for output at each step
# c (cell state):   the "long-term memory" conveyor belt — gates control what's added/removed from it
# this h/c pair (vs. RNN's single hidden state) is exactly what lets LSTMs retain information over
# long sequences without the vanishing-gradient collapse a vanilla RNN suffers from
```

**39. Handle variable-length sequences efficiently with packing (skip wasted computation on padding).**
```python
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

sequences = [torch.randn(5, 10), torch.randn(3, 10), torch.randn(8, 10)]   # different lengths
lengths = [5, 3, 8]
padded = pad_sequence(sequences, batch_first=True)          # pad shorter ones with zeros to match the longest
packed = pack_padded_sequence(padded, lengths, batch_first=True, enforce_sorted=False)
output, (h_n, c_n) = lstm2(packed)                             # LSTM skips the padded positions entirely
output, _ = pad_packed_sequence(output, batch_first=True)      # unpack back to a regular padded tensor
```

**40. Use a bidirectional LSTM to read the sequence both forward and backward.**
```python
bi_lstm = nn.LSTM(input_size=100, hidden_size=128, batch_first=True, bidirectional=True)
x = torch.randn(32, 10, 100)
output, (h_n, c_n) = bi_lstm(x)
output.shape    # (32, 10, 256) — 128*2, forward and backward hidden states concatenated at every timestep
# useful when the FULL sequence is available upfront (e.g. text classification) — not for live/streaming data
```

**41. Tune `hidden_size` — the model's per-layer capacity.**
```python
# too small (e.g. 8): underfits — not enough capacity to represent the pattern, train AND val loss stay high
# too large (e.g. 1024) on a small dataset: overfits fast, and each step gets much slower
# practical approach: start around 64-256, watch the train/val gap (code-drills-classical-ml.md drill #26's
# overfitting check applies identically here), increase only if train loss itself is still too high
```

**42. Tune `num_layers`, and use `dropout` to prevent stacked layers from overfitting.**
```python
lstm = nn.LSTM(input_size=100, hidden_size=128, num_layers=3, dropout=0.3, batch_first=True)
# dropout here applies BETWEEN stacked LSTM layers only (not after the last one) — silently does
# NOTHING if num_layers=1, a common gotcha. More layers = more capacity but harder/slower to train;
# most sequence tasks rarely need more than 2-3.
```

**43. Diagnose a learning-rate that's too high — the #1 cause of a training run that "just doesn't work."**
```python
# symptoms of LR too high: loss oscillates wildly, jumps to NaN/inf, or explodes upward instead of decreasing
# symptoms of LR too low: loss decreases, but barely — flat-looking curve, painfully slow convergence
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)   # 1e-3 is a reasonable Adam starting point for RNNs/LSTMs
# if loss goes NaN: lower lr by 10x (e.g. 1e-4) FIRST, before touching architecture — this is the highest-leverage knob
```

**44. Apply gradient clipping — the standard fix for LSTM's exploding-gradient tendency.**
```python
for X_batch, y_batch in loader:
    optimizer.zero_grad()
    output = model(X_batch)
    loss = criterion(output, y_batch)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)   # caps the gradient's overall size
    optimizer.step()      # RNN/LSTM gradients can spike sharply on long sequences — clipping keeps steps stable
```

**45. Tune `batch_size` — the tradeoff between training stability, speed, and generalization.**
```python
# small batch (e.g. 8-32): noisier gradient estimates (can help escape sharp minima), slower per epoch (wall-clock)
# large batch (e.g. 256+): smoother/faster per-step, but can generalize slightly worse and needs more memory
# for LSTMs specifically: larger batches also mean more padding waste if sequence lengths vary a lot (see drill #39)
```

**46. Decay the learning rate when validation loss plateaus, instead of leaving it fixed.**
```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
for epoch in range(50):
    train_one_epoch(model, loader, optimizer, criterion)
    val_loss = evaluate(model, val_loader, criterion)
    scheduler.step(val_loss)     # if val_loss hasn't improved for 2 epochs, multiply lr by 0.5
```

**47. Understand sequence length's practical effect on LSTM training (truncated BPTT).**
```python
# backprop through an LSTM unrolls one step per timestep — a 500-step sequence means 500 chained
# gradient computations, which is slow AND still prone to vanishing gradients despite LSTM's gating.
# practical fix: chunk long sequences into fixed windows (e.g. 50-100 steps) and carry hidden state across
# chunks with .detach() so gradients don't backprop through the ENTIRE history each step:
h = h.detach()   # keeps the state's VALUE, cuts the graph — the standard truncated-BPTT pattern
```

**48. Put it together — a small hyperparameter sweep over the knobs that matter most.**
```python
best_score, best_config = -1, None
for hidden_size in [64, 128, 256]:
    for num_layers in [1, 2]:
        for lr in [1e-3, 1e-4]:
            model = LSTMClassifier(input_size=100, hidden_size=hidden_size, num_classes=2)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            # ... train for a few epochs, then evaluate on validation data ...
            val_acc = evaluate(model, val_loader)   # placeholder — reuse drill #22's pattern
            if val_acc > best_score:
                best_score, best_config = val_acc, (hidden_size, num_layers, lr)
# this IS what GridSearchCV (code-drills-classical-ml.md drill #13) does for you automatically for sklearn
# models — PyTorch has no built-in equivalent, so sweeps like this are usually hand-rolled or done via
# a library like Optuna/Ray Tune once the search space grows past a few knobs.
```

---

**This closes the Code Drills tier — from `x = 5` to tuning an LSTM.** Full path: `code-drills-basics.md` → `code-drills-data-structures.md` → `code-drills-oop-intermediate.md` → `code-drills-numpy-pandas.md` → `code-drills-classical-ml.md` → this file. For the conceptual depth behind any of this (the transformer math, why Adam beats SGD, backprop derivations), the existing hub tiers pick up exactly where this leaves off — see `pytorch-deep-dive.md`, `deep-learning-practice.md`, and `math-foundations-refresher.md`.
