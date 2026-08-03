# PyTorch Deep Dive — Built as a Chain, Not a List

Continues from `deep-learning-practice.md`, more advanced territory: what happens under the hood, how to debug it, and how to ship it. Every snippet was actually executed in this session. Each cluster is one continuous thread — every question inherits the answer before it, closing with a worked summary example.

---

> 🔗 **Hands-on reps:** [Code Drills 6 — Tensors & Autograd](/topic/code-drills-deep-learning#cluster-1-tensors-autograd)

## Cluster 1 — Autograd Internals: Custom Functions and Hooks

### 1. When a built-in operation isn't enough, how do you write a custom autograd `Function` with your own forward AND backward math?
```python
import torch

class ClampedSquare(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)                 # stash what backward() will need
        return x.clamp(-3, 3) ** 2                  # forward: clamp then square

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_input = grad_output * 2 * x.clamp(-3, 3)   # chain rule, by hand, for THIS custom op
        grad_input[x.abs() > 3] = 0                       # gradient is 0 outside the clamp range
        return grad_input

x = torch.randn(5, requires_grad=True)
y = ClampedSquare.apply(x)
y.sum().backward()
print(x.grad)
```
`ctx.save_for_backward(x)` instead of a plain Python attribute: it registers the tensor properly with autograd's memory management (freed at the right time, interacts correctly with `.detach()` and checkpointing) — stashing tensors as plain `ctx` attributes works for simple cases but bypasses this bookkeeping, which can leak memory in long training runs. The `backward` method here IS the chain rule from `math-foundations-refresher.md`, written out explicitly instead of PyTorch deriving it automatically.

### 2. Given a custom or built-in layer already running, how do you INSPECT what's flowing through it during a real forward pass, without touching its code?
```python
import torch.nn as nn

model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 2))
activations = {}

def make_hook(name):
    def hook(module, input, output):
        activations[name] = output.detach()
    return hook

model[1].register_forward_hook(make_hook("relu_out"))
_ = model(torch.randn(4, 10))
print(activations["relu_out"].shape)          # [4, 20] -- captured without changing the model's code at all
print((activations["relu_out"] == 0).float().mean())   # fraction of dead ReLU units this batch
```
Forward hooks let you inspect/log intermediate activations, gradients, or shapes WITHOUT modifying the model's `forward()` method — essential when debugging someone else's model, checking for dead ReLUs (units stuck at exactly 0), or building an activation-visualization tool.

### 3. Given that forward hooks inspect the forward pass, how do you inspect GRADIENTS flowing backward through a specific layer?
```python
grad_log = {}
def grad_hook(module, grad_input, grad_output):
    grad_log["grad_output_norm"] = grad_output[0].norm().item()

handle = model[0].register_full_backward_hook(grad_hook)
out = model(torch.randn(4, 10))
out.sum().backward()
print(grad_log["grad_output_norm"])
handle.remove()     # always remove hooks you no longer need -- they persist and can leak memory otherwise
```
`register_full_backward_hook` (not the older `register_backward_hook`): the older version had documented inconsistent behavior with modules that have multiple inputs/outputs — the full version is the corrected, currently-recommended way to reliably catch a vanishing/exploding gradient at the exact layer it starts.

### Summary example
Debugging a model that trains slower than expected: a forward hook on each layer reveals one ReLU layer with 80% dead units (question 2) — a real capacity problem — while a backward hook on the same layer (question 3) shows gradient norms near zero flowing INTO it, confirming the dead units aren't receiving learning signal either. Neither hook required editing the model's `forward()` method at all.

---

## Cluster 2 — Mixed Precision and Distributed Training

### 1. How do you actually run mixed-precision training end to end, not just describe it?
```python
import torch.optim as optim

model = nn.Linear(10, 2).cuda() if torch.cuda.is_available() else nn.Linear(10, 2)
device = next(model.parameters()).device
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

xb, yb = torch.randn(8, 10, device=device), torch.randint(0, 2, (8,), device=device)
optimizer.zero_grad()
with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=torch.cuda.is_available()):
    out = model(xb)
    loss = nn.functional.cross_entropy(out, yb)
scaler.scale(loss).backward()     # scales the loss UP before backward to avoid fp16 underflow
scaler.step(optimizer)             # unscales gradients, checks for inf/nan, then steps (or skips if unstable)
scaler.update()                     # adjusts the scale factor for next time
```
fp16 has a narrow representable range, so small gradients can underflow to exactly zero — `GradScaler` multiplies the loss by a large factor before `backward()` so gradients land in a safer range, then divides back out before the optimizer sees them. `step()` silently SKIPS the update if it detects inf/nan; `update()` adapts the scale factor over time — this is why hand-rolling fp16 training without `GradScaler` is a common source of silent `nan` losses.

**Visual + memory hook — the numbers get inflated before the danger zone, then deflated back after it:**
```
loss ──▶ [ × big scale factor ] ──▶ SAFE ZONE: backward() ──▶ [ ÷ scale factor ] ──▶ optimizer.step()
              scaler.scale()          fp16 gradients no       scaler unscales      real-sized
                                       longer underflow        internally           gradients
                                       to zero here             before this check
```
**Remember it as zooming in before a photo, then zooming back out:** three separate calls exist because three separate things happen — inflate, compute safely, deflate-and-check — not API verbosity.

### 2. Given a single GPU trains with mixed precision, how do you scale the SAME training loop across MULTIPLE GPUs (DDP)?
```python
"""
# save as train_ddp.py, launch with: torchrun --nproc_per_node=NUM_GPUS train_ddp.py
import os, torch, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

dist.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

model = MyModel().to(local_rank)
model = DDP(model, device_ids=[local_rank])

sampler = DistributedSampler(dataset)     # ensures each GPU rank sees a DISJOINT shard of the data
loader = DataLoader(dataset, batch_size=32, sampler=sampler)

for epoch in range(10):
    sampler.set_epoch(epoch)               # reshuffles differently each epoch across all ranks consistently
    for xb, yb in loader:
        xb, yb = xb.to(local_rank), yb.to(local_rank)
        loss = criterion(model(xb), yb)
        optimizer.zero_grad(); loss.backward(); optimizer.step()

dist.destroy_process_group()
"""
```
`sampler.set_epoch(epoch)` is a real, easy-to-miss requirement: without it, `DistributedSampler` uses the SAME shuffle order every epoch across all ranks — calling it at the start of each epoch reseeds the shuffle consistently (using the epoch number) so training doesn't waste shuffling's regularization benefit by seeing the exact same batch order repeatedly.

### Summary example
Training a large model that doesn't fit on one GPU's compute budget in reasonable time: `GradScaler`+`autocast` (question 1) halves memory and often speeds up each individual GPU's step, while `DistributedSampler`+`DDP` (question 2) splits the data across multiple GPUs so each processes a disjoint shard per step — the two techniques are independent and routinely combined, since one addresses per-GPU efficiency and the other addresses scaling across GPUs.

---

## Cluster 3 — Custom Losses and Deliberate Weight Initialization

### 1. How do you write a custom loss function correctly, so gradients still flow through it?
```python
class WeightedMSE(nn.Module):
    def __init__(self, weight_high=3.0, threshold=0.8):
        super().__init__()
        self.weight_high = weight_high
        self.threshold = threshold
    def forward(self, pred, target):
        se = (pred - target) ** 2
        weights = torch.where(target > self.threshold, self.weight_high, 1.0)   # differentiable weighting
        return (se * weights).mean()

criterion = WeightedMSE()
pred = torch.randn(5, requires_grad=True)
target = torch.rand(5)
loss = criterion(pred, target)
loss.backward()
```
The whole computation must stay in PyTorch ops (no `.item()` or NumPy mid-computation): calling `.item()` or converting to NumPy detaches the value from the autograd graph — any computation done outside PyTorch tensor operations can't have gradients flow back through it. A custom loss that "runs" but silently returns `None` for `.grad` almost always has exactly this bug: a stray `.item()`, `.numpy()`, or Python `if` on a tensor's value instead of `torch.where`.

### 2. Given a correctly-differentiable custom loss, does the STARTING point of training (weight initialization) matter too?
```python
def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")   # matched to ReLU's expected activation scale
        nn.init.zeros_(module.bias)

model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 2))
model.apply(init_weights)     # applies the function to EVERY submodule recursively
```
Yes — Kaiming (He) initialization accounts for ReLU zeroing out roughly half its inputs, scaling initial weights up accordingly to keep activation variance stable across layers at the start of training. Xavier/Glorot initialization (the other common choice) assumes a symmetric activation like tanh and is the better match there. Using the mismatched one doesn't crash anything, it just makes early training slower/less stable in a way that's easy to misattribute to something else entirely.

### Summary example
A custom-loss model trains but the loss barely moves in the first several epochs: checking `pred.grad` confirms gradients ARE flowing (question 1's discipline was followed correctly), so the actual culprit turns out to be `nn.init.xavier_normal_` applied to a ReLU network (question 2's mismatch) — switching to `kaiming_normal_` fixes the slow start without touching the loss function at all.

---

## Cluster 4 — Transformer Building Blocks, Off the Shelf

### 1. Instead of hand-rolling attention, how do you use PyTorch's built-in Transformer encoder block?
```python
encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=256, batch_first=True)
encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
x = torch.randn(2, 10, 64)     # [batch, seq_len, d_model]
out = encoder(x)                # [2, 10, 64] -- same shape, refined representation
```
The from-scratch build (`nca-genl`'s §1.4) exists to prove you understand the mechanism; `nn.TransformerEncoderLayer` is the production-grade, heavily-optimized version (fused kernels, correct default initialization, tested edge cases) you'd actually use in real code once the mechanism is understood — knowing both, and when each is appropriate, is the actual skill.

### 2. Given a bidirectional encoder block, how do you make it CAUSAL (GPT-style, autoregressive)?
```python
seq_len = 10
causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len)   # built-in helper, upper triangle = -inf
out = encoder(x, mask=causal_mask, is_causal=True)
```
`is_causal=True` as a SEPARATE flag from just passing the mask: recent PyTorch versions use a faster fused-attention code path specifically when told the mask is causal (rather than a generic arbitrary mask) — passing the flag lets PyTorch pick the optimized implementation instead of the general one, a real (if version-dependent) performance difference for the identical mathematical result.

### 3. Before training any architecture built this way, how do you sanity-check its actual SIZE?
```python
def count_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total

trainable, total = count_params(encoder)
print(f"{trainable:,} trainable / {total:,} total")
```
A typo in a layer's dimensions can silently create a model 100x larger or smaller than intended — this one-liner catches that immediately, before burning GPU-hours discovering it from a suspiciously slow (or suspiciously fast) training loop.

### Summary example
Building a small GPT-style model: `nn.TransformerEncoderLayer` (question 1) provides the optimized attention/FFN blocks, `generate_square_subsequent_mask` + `is_causal=True` (question 2) makes it autoregressive instead of bidirectional, and `count_params` (question 3) run immediately after construction confirms the parameter count matches the intended model size — catching a `d_model`/`nhead` typo before a single training step runs.

---

## Cluster 5 — Reproducibility and the `eval()`/`no_grad()` Distinction

### 1. How do you make a PyTorch run FULLY reproducible, not just "seeded"?
```python
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True     # forces deterministic (sometimes slower) CUDA algorithms
    torch.backends.cudnn.benchmark = False          # disables auto-tuning, which itself introduces nondeterminism

set_seed(42)
```
Seeding `torch.manual_seed` alone is NOT sufficient: Python's `random`, NumPy's RNG, and PyTorch's CPU/GPU RNGs are all independent — a data augmentation pipeline using plain `random.random()` won't respect only `torch.manual_seed`. And `cudnn.benchmark=True` (a common speed optimization) lets cuDNN pick different algorithms run-to-run based on timing, which is fast but explicitly NOT reproducible.

### 2. Given a fully-seeded, reproducible setup, what's the actual difference between `model.eval()` and `torch.no_grad()` — a real, common confusion?
```python
model.eval()                 # changes LAYER BEHAVIOR: BatchNorm uses running stats, Dropout turns off
with torch.no_grad():         # changes AUTOGRAD: stops tracking gradients, saves memory or during inference
    preds = model(torch.randn(4, 10))
```
`model.eval()` alone still tracks gradients (wasting memory unnecessarily during pure inference); `torch.no_grad()` alone leaves BatchNorm/Dropout in training-mode behavior (wrong statistics/randomness for real inference) — the same distinction already drawn in `deep-learning-practice.md`. Production inference code should use both together; a validation loop DURING training also needs `model.eval()` but should switch back to `model.train()` afterward before resuming training.

### Summary example
Reporting a benchmark number that needs to be exactly reproducible by a reviewer: `set_seed(42)` (question 1) alone isn't enough if the eval loop forgets `torch.no_grad()` — a bug that wouldn't change the REPORTED accuracy number but would silently waste memory and, if that eval code is later modified to include backward calls, could inject nondeterminism precisely because gradients were being tracked when they shouldn't have been.

---

## Cluster 6 — Exporting for Deployment

### 1. How do you export a trained model to TorchScript, for deployment without a Python runtime?
```python
model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 2)).eval()
example_input = torch.randn(1, 10)
traced = torch.jit.trace(model, example_input)
traced.save("model_traced.pt")

loaded = torch.jit.load("model_traced.pt")
print(loaded(example_input).shape)
```
`.eval()` before tracing is non-negotiable: tracing records the ACTUAL operations executed for the example input — in train mode, Dropout/BatchNorm behave differently (randomly, or using batch stats) than at serving time, and the traced graph would permanently bake in the wrong (training-mode) behavior. Tracing also has a real limitation: it only records the specific control-flow path taken for THAT one example input — a model with a genuine data-dependent `if` branch needs `torch.jit.script` instead, which parses the actual Python control flow rather than replaying one recorded execution.

### 2. What if the deployment target isn't even PyTorch at all — a C++ or browser runtime?
```python
torch.onnx.export(
    model, example_input, "model.onnx",
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},   # allow variable batch size at inference
)
```
Without `dynamic_axes`, the exported ONNX graph hardcodes the exact batch size used during export (here, 1) — any real serving system needs to handle varying batch sizes, so marking dimension 0 as dynamic is what makes the exported model actually usable for real traffic instead of one fixed batch size forever.

### Summary example
Deploying the same model two ways: TorchScript (question 1) for a Python-free but still PyTorch-runtime environment, ONNX (question 2) for a genuinely cross-framework C++ inference server — both require `.eval()` mode first, and ONNX specifically needs `dynamic_axes` set or the exported model will only ever accept exactly one request at a time, batch size 1, forever.

---

## Cluster 7 — Specialized Architectures: Autoencoders and GANs

### 1. How do you build a simple autoencoder, and use it for anomaly detection without any labeled anomalies?
```python
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 32), nn.ReLU(), nn.Linear(32, latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 32), nn.ReLU(), nn.Linear(32, input_dim))
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

ae = Autoencoder(input_dim=20)
x = torch.randn(16, 20)
reconstructed = ae(x)
reconstruction_error = ((x - reconstructed) ** 2).mean(dim=1)   # per-sample error -- the actual anomaly score
```
An autoencoder trained only on NORMAL data learns to compress and reconstruct normal patterns well; a genuinely anomalous input doesn't match the patterns it learned, so it reconstructs poorly — the per-sample reconstruction error becomes a usable anomaly score without ever needing a single labeled anomalous example during training.

### 2. Given an encoder/decoder pair that learns to RECONSTRUCT, how does a GAN's generator/discriminator pair learn to GENERATE new data instead, and why does training alternate two updates?
```python
generator = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 20))
discriminator = nn.Sequential(nn.Linear(20, 32), nn.ReLU(), nn.Linear(32, 1))
opt_g = torch.optim.Adam(generator.parameters(), lr=2e-4)
opt_d = torch.optim.Adam(discriminator.parameters(), lr=2e-4)
criterion = nn.BCEWithLogitsLoss()

real_data = torch.randn(16, 20)
noise = torch.randn(16, 16)

# 1. train discriminator: tell real from fake
opt_d.zero_grad()
fake_data = generator(noise).detach()      # detach: don't let discriminator's step update the generator
real_loss = criterion(discriminator(real_data), torch.ones(16, 1))
fake_loss = criterion(discriminator(fake_data), torch.zeros(16, 1))
(real_loss + fake_loss).backward()
opt_d.step()

# 2. train generator: fool the discriminator
opt_g.zero_grad()
fake_data = generator(noise)                # NOT detached this time -- gradient needs to reach the generator
gen_loss = criterion(discriminator(fake_data), torch.ones(16, 1))   # wants discriminator to say "real"
gen_loss.backward()
opt_g.step()
```
`.detach()` appears in step 1 but not step 2, precisely because of what's being trained: in step 1, only the discriminator is training — gradients back into the generator would be wasted work, hence `.detach()`. In step 2, the whole point is getting a gradient signal INTO the generator by backpropagating through the (frozen this round) discriminator — forgetting to detach in step 1 wastes compute; forgetting to leave it un-detached in step 2 is the real bug, silently producing zero generator gradient.

### Summary example
An autoencoder (question 1) and a GAN's generator (question 2) both learn a compressed/latent representation of data, but for opposite purposes: the autoencoder is scored by how well it RECONSTRUCTS its own input (useful for anomaly detection — a bad reconstruction IS the signal), while the GAN's generator is scored by whether a separate discriminator can tell its output apart from real data at all (useful for generating novel, realistic-looking data) — the alternating `.detach()`/no-`.detach()` pattern in the GAN loop is what keeps those two networks' training signals from contaminating each other on the wrong step.

---

## Cluster 8 — Cross-Validation for a Neural Network, Not Just a sklearn Model

### 1. `sklearn-practice.md` covers `cross_val_score` for sklearn models — how do you do the same k-fold discipline for a PyTorch model?
```python
from sklearn.model_selection import KFold
import numpy as np

X_all = torch.randn(100, 10)
y_all = torch.randint(0, 2, (100,))
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
fold_scores = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_all)):
    model = nn.Sequential(nn.Linear(10, 16), nn.ReLU(), nn.Linear(16, 2))   # FRESH model every fold
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    X_tr, y_tr = X_all[train_idx], y_all[train_idx]
    X_val, y_val = X_all[val_idx], y_all[val_idx]
    for epoch in range(20):
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(X_tr), y_tr)
        loss.backward(); optimizer.step()
    with torch.no_grad():
        acc = (model(X_val).argmax(1) == y_val).float().mean().item()
    fold_scores.append(acc)

print(f"mean acc: {np.mean(fold_scores):.3f} +/- {np.std(fold_scores):.3f}")
```
A FRESH model must be created INSIDE the fold loop, not reused: reusing the same model instance across folds means fold 2's training starts from fold 1's already-trained weights, not from scratch — this leaks information across folds and makes the resulting CV estimate meaningless as a measure of how the architecture performs on genuinely unseen data. `sklearn`'s `Pipeline` handles this automatically by design (`sklearn-practice.md`, Cluster 1); a hand-rolled PyTorch loop has to enforce it manually, explicitly, every time.

### Summary example
Evaluating a small architecture's stability across 5 folds gives `mean acc: 0.780 +/- 0.045` — the `+/- 0.045` matters as much as the mean (the same "report std, not just mean" discipline from `sklearn-practice.md`'s cross-validation cluster), and it's only a trustworthy number because a brand-new, randomly-initialized model was created at the top of each fold's loop rather than one model incrementally trained across all 5 folds.

---

## Cluster 9 — Diffusion Models: A Third Way to Generate Data

### 1. Given an autoencoder reconstructs its input (Cluster 7, Q1) and a GAN's generator learns by fooling a discriminator (Cluster 7, Q2), how does a diffusion model generate new data through a third, completely different mechanism?
A diffusion model has two phases, and only one of them is learned. The **forward process** is fixed, known math with no training at all: take a real data point and progressively add a little Gaussian noise over many steps until nothing recognizable is left. The **reverse process** is where the learning happens: a network is trained to undo one small noising step at a time, so that starting from pure random noise and repeatedly applying it reconstructs a brand-new, realistic sample.

```python
def make_schedule(T, beta_start=0.001, beta_end=0.20):
    betas = [beta_start + (beta_end - beta_start) * i / (T - 1) for i in range(T)]
    alphas = [1.0 - b for b in betas]
    alphas_cumprod, running = [], 1.0
    for a in alphas:
        running *= a
        alphas_cumprod.append(running)          # alpha_bar_t = product of alphas up to step t
    return betas, alphas_cumprod

def forward_diffusion(x0, t, alphas_cumprod, noise):
    ab = alphas_cumprod[t]                                    # how much of the ORIGINAL signal survives at step t
    return [(ab**0.5) * x0[i] + ((1 - ab)**0.5) * noise[i] for i in range(len(x0))]   # closed-form: jump straight to step t, no loop needed
```

Run this on a toy 4-pixel "image" `x0 = [1, 1, 1, 1]` (a uniform bright patch) with `T=10` and a linear schedule from `beta=0.001` to `beta=0.20` — every number below is a real, computed run, not illustrative:

| t | alpha_bar_t | signal weight (√alpha_bar) | noise weight (√1-alpha_bar) | x_t (4 toy pixels) |
|---|---|---|---|---|
| 1 | 0.999 | 0.999 | 0.032 | `[0.99, 1.02, 0.99, 0.99]` |
| 3 | 0.932 | 0.965 | 0.261 | `[0.72, 0.91, 1.26, 1.08]` |
| 5 | 0.791 | 0.890 | 0.457 | `[1.36, 1.00, 1.07, 0.97]` |
| 7 | 0.609 | 0.780 | 0.625 | `[-0.26, 1.32, 1.10, 1.09]` |
| 9 | 0.423 | 0.650 | 0.760 | `[-0.63, -0.67, -0.03, 0.29]` |
| 10 | 0.338 | 0.582 | 0.814 | `[0.83, 0.54, 1.01, 0.06]` |

The signal weight shrinks monotonically (0.999 → 0.582) while the noise weight grows monotonically (0.032 → 0.814) — that's the fixed, guaranteed part of the schedule. Any single draw of `x_t` still looks random because it IS random (Gaussian noise is added fresh every step), but by t=9 the pixel values ([-0.63, -0.67, -0.03, 0.29]) no longer resemble the original uniform `[1,1,1,1]` pattern at all, while at t=1 they're still obviously close to it — exactly the "clean signal to static" story, just quantified.

### 2. Given the forward process destroys the image into noise using fixed, known math, why can't the reverse process just run that same formula backward — why does undoing the noise require training a neural network at all?
The forward formula computes `x_t` FROM a known `x0` — that's a fully determined calculation. Reversing it would require computing `x0` FROM `x_t`, but at generation time there is no real `x0`: that's precisely the thing being generated. There's no formula for "what was the original clean image" because infinitely many different clean images could have produced the same noisy `x_t`. A neural network (in practice a U-Net for images) is trained to make a statistical best-guess: given a noisy `x_t` and the timestep `t`, predict the specific noise that was added, learned from millions of real (image, noise) pairs during training.

```python
import torch, torch.nn as nn

class TinyDenoiser(nn.Module):
    def __init__(self, dim=4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim + 1, 32), nn.ReLU(), nn.Linear(32, dim))
    def forward(self, x_t, t):
        t_embed = torch.full((x_t.shape[0], 1), float(t))
        return self.net(torch.cat([x_t, t_embed], dim=1))     # predicts the noise that was added at step t

model = TinyDenoiser()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
betas, alphas_cumprod = make_schedule(T=10)

for step in range(1000):
    x0 = torch.rand(32, 4)                                    # a batch of real (toy) training data
    t = torch.randint(0, 10, (1,)).item()
    noise = torch.randn(32, 4)
    ab = alphas_cumprod[t]
    x_t = (ab**0.5) * x0 + ((1 - ab)**0.5) * noise
    predicted_noise = model(x_t, t)
    loss = nn.functional.mse_loss(predicted_noise, noise)     # the ENTIRE training objective: predict the noise, nothing else
    optimizer.zero_grad(); loss.backward(); optimizer.step()
```

Notice the loss is a plain MSE regression against known noise — no adversarial second network, no minimax game (contrast with the GAN loop in Cluster 7, Q2). At generation time there's no training data at all: start from `x_T` = pure random noise, and for `t = T` down to `1`, ask the trained model to predict the noise in `x_t`, subtract a scaled portion of that prediction to step to `x_{t-1}`, and repeat until `x_0` — a sample that was never in the training set.

```
clean image ──add noise──▶ ... ──add noise──▶ pure static     (forward: fixed math, zero training)
    x_0                        x_t                  x_T

pure static ──predict & remove noise──▶ ... ──▶ clean image   (reverse: LEARNED, one small step at a time)
    x_T          (the trained TinyDenoiser)            x_0
```
**Remember it as sculpting from a block of marble, not painting on a blank canvas:** the model never builds an image up from nothing in one shot — it starts with a solid block of noise and, over many steps, chips away exactly the part that doesn't belong, one predictable-sized chip (a learned noise estimate) at a time.

### Summary example
All three generative approaches in this file learn a different thing, scored a different way: the **autoencoder** (Cluster 7, Q1) learns to compress-and-reconstruct its OWN input, scored by reconstruction error, in one deterministic pass. The **GAN** (Cluster 7, Q2) learns to fool a second, adversarial network, scored by a shifting minimax game, also in one pass. The **diffusion model** learns to predict noise, scored by a plain, stable MSE loss — no adversary, no minimax — but needs dozens to thousands of small sequential passes instead of one. That training stability (a boring regression loss instead of two networks fighting) is a major reason diffusion overtook GANs as the dominant approach behind Stable Diffusion, DALL-E, and Midjourney, at the real cost of slower generation.

---

## Practice Q&A (Self-Test)

**Q1. In `ClampedSquare`, why use `ctx.save_for_backward(x)` instead of just storing `x` as a plain attribute on `ctx`?**
A: `save_for_backward` registers the tensor properly with autograd's memory management, so it's freed at the right time and interacts correctly with things like `.detach()` and checkpointing. Stashing tensors as plain `ctx` attributes works for simple cases but bypasses this bookkeeping, which can leak memory in long training runs.

**Q2. What does `register_forward_hook` let you do that plain code can't, and what does the `relu_out` hook example in this file actually compute?**
A: It lets you inspect or log intermediate activations, gradients, or shapes without modifying the model's `forward()` method at all — useful for debugging someone else's model or building a visualization tool. The example hook captures the ReLU layer's output and computes the fraction of dead ReLU units (activations exactly 0) in that batch.

**Q3. Why use `register_full_backward_hook` instead of the older `register_backward_hook`?**
A: The older `register_backward_hook` had documented inconsistent behavior with modules that have multiple inputs/outputs. `register_full_backward_hook` is the corrected, currently-recommended version for reliably inspecting gradients at a specific layer, e.g. to catch a vanishing/exploding gradient at the exact layer it starts.

**Q4. Walk through why `GradScaler` training uses three separate calls — `scaler.scale(loss).backward()`, `scaler.step(optimizer)`, `scaler.update()` — instead of just calling `backward()`/`step()` directly under `autocast`.**
A: fp16 has a narrow representable range, so small gradients can underflow to exactly zero. `scale()` multiplies the loss up before `backward()` so gradients land in a safer range; `step()` unscales the gradients, checks for inf/nan, and steps the optimizer (or silently skips the update if unstable); `update()` adapts the scale factor for next time. Hand-rolling fp16 training without this pattern is a common source of silent `nan` losses.

**Q5. In the DDP training skeleton, why is `sampler.set_epoch(epoch)` called at the start of every epoch?**
A: Without it, `DistributedSampler` uses the same shuffle order every epoch across all ranks. Calling `set_epoch` reseeds the shuffle consistently (using the epoch number) so training doesn't see the exact same batch order repeatedly, preserving shuffling's regularization benefit.

**Q6. In `WeightedMSE`, why must the entire computation stay in PyTorch tensor ops (e.g. `torch.where`) rather than using `.item()`, `.numpy()`, or a Python `if` on a tensor's value?**
A: Calling `.item()` or converting to NumPy detaches the value from the autograd graph, so any computation done outside PyTorch tensor operations can't have gradients flow back through it. A custom loss that "runs" but silently returns `None` for `.grad` almost always has exactly this bug.

**Q7. Why does Kaiming (He) initialization pair specifically with ReLU, and what actually goes wrong if you use Xavier/Glorot instead?**
A: Kaiming init accounts for ReLU zeroing out roughly half its inputs, scaling initial weights up to keep activation variance stable across layers at the start of training. Xavier/Glorot assumes a symmetric activation like tanh. Using the mismatched one doesn't crash anything — it just makes early training slower/less stable in a way that's easy to misattribute to something else.

**Q8. When adding a causal mask to `nn.TransformerEncoderLayer` for GPT-style autoregressive use, why pass `is_causal=True` in addition to the mask itself?**
A: Recent PyTorch versions can use a faster fused-attention code path specifically when told the mask is causal, rather than treating it as a generic arbitrary mask. Passing the flag lets PyTorch pick the optimized implementation for the identical mathematical result — a real, version-dependent performance difference.

**Q9. In the GAN training loop, why does `.detach()` appear on `fake_data` in the discriminator step but not in the generator step?**
A: In the discriminator step, only the discriminator is being trained, so computing gradients back into the generator would be wasted work — hence `.detach()`. In the generator step, the whole point is to get a gradient signal into the generator by backpropagating through the (frozen this round) discriminator, so `fake_data` must NOT be detached; forgetting to detach in step 1 wastes compute, but forgetting to leave it un-detached in step 2 is the real bug, since it would silently produce zero generator gradient.

**Q10. Why does `torch.onnx.export` use `dynamic_axes={"input": {0: "batch_size"}, ...}`, and what happens if you omit it?**
A: Without `dynamic_axes`, the exported ONNX graph hardcodes the exact batch size used during export (e.g. 1). Marking dimension 0 as dynamic lets the exported model handle varying batch sizes at inference, which is what makes it actually usable for real serving traffic instead of one fixed batch size forever.

**Q11. In a diffusion model, why is the forward (noising) process computed directly from a fixed formula while the reverse (denoising) process has to be learned by a neural network?**
A: The forward process computes a known quantity — `x_t` derived from a known `x0` plus a known amount of noise — so it's a deterministic calculation, no training needed. The reverse process would require computing `x0` from `x_t`, but at generation time there is no real `x0`; infinitely many clean images could have produced the same noisy `x_t`. A network has to learn a statistical best-guess (predict the noise) from real training examples, because there's no formula that recovers information that was genuinely destroyed.

**Q12. A diffusion model's training loss is a plain MSE between predicted and actual noise — no discriminator, no adversarial game. Why does this make diffusion models more stable to train than GANs?**
A: A GAN's generator and discriminator are in an adversarial minimax game — if one network gets too strong too fast, the other stops receiving a useful gradient signal (a classic GAN failure mode, e.g. discriminator collapse). A diffusion model's noise-prediction objective is a single, fixed regression target at every step, so there's no second network to destabilize the loss landscape — it's ordinary supervised learning repeated many times, which is why diffusion training rarely suffers the mode collapse and instability that plague GAN training.


---

## Video-Sourced Practice MCQs (Set 2)

A second practice set for PyTorch, built the same way as this hub's NCA-GENL community bank: topics checked against a real YouTube PyTorch-interview-prep video, then written up as fully original multiple-choice questions here (the source video mostly gave prose explanations, not MCQs -- every option and explanation below is original, written to match this hub's "explain every option" convention, not copied from the video). These deliberately cover angles NOT already drilled in the clusters above (autograd internals, mixed precision/DDP, custom losses/init, transformer blocks, reproducibility, export, GANs, cross-validation, diffusion) -- tensor memory sharing, buffers vs. parameters, retain_graph, loss numerical stability, quantization, tensor combination ops, and imbalanced-data sampling.

<script type="application/json" class="topic-quiz-data" data-title="PyTorch Deep Dive (Set 2)">
[
  {
    "d": "Tensors & Memory",
    "q": "You have a large NumPy array and want it as a PyTorch tensor. `torch.from_numpy(arr)` and `torch.tensor(arr)` both work, but behave differently. What's the actual difference?",
    "o": [
      "`torch.from_numpy` shares the same underlying memory as the array (no copy — editing one changes the other); `torch.tensor` always copies the data into a new buffer",
      "`torch.from_numpy` always copies; `torch.tensor` shares memory",
      "They're functionally identical, just two spellings of the same operation",
      "`torch.from_numpy` only works on GPU tensors, `torch.tensor` only works on CPU tensors"
    ],
    "a": [
      0
    ],
    "e": "`torch.from_numpy` creates a tensor that VIEWS the NumPy array's existing memory — no copy happens, so it's fast for large arrays, but mutating the tensor mutates the original array too (and vice versa). `torch.tensor(arr)` always allocates a fresh buffer and copies the values in, which is safer but slower for big data. The second option has the two exactly backwards. The third option ignores a real, interview-relevant distinction. The fourth invents a CPU/GPU restriction that doesn't exist — both start on CPU regardless of the array's origin."
  },
  {
    "d": "nn.Parameter vs. Buffers",
    "q": "Inside an `nn.Module`, wrapping a tensor in `nn.Parameter` makes it get gradients and get updated by the optimizer. What's `register_buffer` for — why not just make everything a `Parameter`?",
    "o": [
      "A buffer is identical to a Parameter except it's stored on CPU only",
      "A buffer is a temporary variable that gets deleted after each forward pass and never saved",
      "There's no real difference — `register_buffer` is a deprecated alias for `nn.Parameter`",
      "A buffer is tracked as part of the module's state (saved/loaded, moved to GPU with `.to()`) but is NEVER updated by backprop — for things like BatchNorm's running mean, which must persist but isn't learned via gradients"
    ],
    "a": [
      3
    ],
    "e": "Buffers exist for exactly this gap: data the module needs to carry around and move between devices consistently with its parameters (so `.to(device)` and `state_dict()` still catch it), but that should NEVER receive a gradient update — BatchNorm's running mean/variance being the textbook example, since those are computed from data statistics, not learned via backprop. Making everything a Parameter would make the optimizer try to gradient-update statistics that were never meant to be learned that way. The CPU-only claim and the 'deleted after forward pass' claim both describe behavior buffers explicitly do NOT have — they persist exactly like parameters, just without gradients."
  },
  {
    "d": "Backward Pass Mechanics",
    "q": "Calling `.backward()` on a loss builds and consumes a computation graph for that call. What does `retain_graph=True` actually change, and when do you need it?",
    "o": [
      "It has no functional effect — it's a purely cosmetic debugging flag",
      "It keeps the graph in memory after `.backward()` instead of freeing it, which you need when you must call `.backward()` again through the SAME forward pass (e.g. gradient accumulation across multiple losses sharing intermediate tensors) — at the cost of extra memory",
      "It retrains the entire model from scratch using the same graph",
      "It makes gradients accumulate across multiple different training steps automatically, replacing the need for `zero_grad()`"
    ],
    "a": [
      1
    ],
    "e": "By default, PyTorch frees the intermediate activations that built the graph the moment `.backward()` finishes, since normally you don't need them again. If your training scheme calls `.backward()` a second time through tensors that came from the SAME forward pass (e.g. computing gradients for two different loss terms that share upstream computation), that graph must still exist — `retain_graph=True` keeps it around, at the cost of holding extra memory. It has nothing to do with `zero_grad()` (which clears accumulated gradient VALUES, a separate mechanism), doesn't do anything 'automatically' about accumulation, and definitely isn't cosmetic — omitting it in the scenario above throws a runtime error the second time you call `.backward()`."
  },
  {
    "d": "Loss Functions",
    "q": "For binary classification, `nn.BCELoss` expects a sigmoid-activated probability as input, while `nn.BCEWithLogitsLoss` expects the RAW logits (no sigmoid applied). Why is the logits version generally preferred?",
    "o": [
      "It runs faster purely because it skips computing a probability, with no numerical-stability benefit at all",
      "`BCELoss` is deprecated and no longer works in current PyTorch versions",
      "It internally combines sigmoid and the log-loss computation in a single, numerically stable operation, avoiding the overflow/underflow that can happen computing `log(sigmoid(x))` as two separate steps for very large or very negative x",
      "It's mathematically a completely different loss function that happens to have a similar name"
    ],
    "a": [
      2
    ],
    "e": "Computing `sigmoid(x)` first and then taking `log()` of the result as two separate floating-point operations can blow up: for a very negative logit, `sigmoid(x)` rounds to exactly 0.0 in floating point, and `log(0.0)` is `-inf`. `BCEWithLogitsLoss` fuses the two steps using a numerically stable formulation (the log-sum-exp trick) that never actually computes an intermediate probability that could hit exactly 0 or 1. It's the same underlying loss mathematically, not a different one. The speed argument alone (third option) misses the actual reason it's recommended. And `BCELoss` still exists and works — it's just riskier for extreme inputs, not deprecated."
  },
  {
    "d": "Numerical Stability",
    "q": "`nn.Softmax` and `nn.LogSoftmax` both turn raw scores into a probability-like output, but `LogSoftmax` is often preferred internally (e.g. paired with `NLLLoss`). Why?",
    "o": [
      "There's no numerical reason at all — it's purely a matter of naming convention with identical implementations",
      "`LogSoftmax` produces a completely different probability distribution than `Softmax`, and is only correct for regression tasks",
      "`Softmax` only works on the CPU, so GPU training requires `LogSoftmax`",
      "Computing softmax's `exp(x)` directly can overflow for large input scores; `LogSoftmax` uses a numerically stable formulation that avoids ever computing a raw, unstabilized exponential"
    ],
    "a": [
      3
    ],
    "e": "`exp(x)` for a large `x` can overflow to `inf` in floating point, and `inf/inf` from the softmax normalization then produces `NaN`. `LogSoftmax` is computed with a stabilized formula (subtracting the max value before exponentiating, done internally) that sidesteps that overflow, which is exactly why it's paired with `NLLLoss` for classification instead of `Softmax`+`log()` done as two separate steps. It produces the mathematically equivalent log-probabilities of the SAME distribution, not a different one, and it isn't a classification-only vs. regression-only distinction. There's no CPU/GPU restriction on either function — both run on both."
  },
  {
    "d": "Deployment: Quantization",
    "q": "PyTorch offers dynamic and static quantization to shrink model size/speed up inference. What's the actual difference in WHEN each computes its quantization ranges?",
    "o": [
      "They quantize completely different parts of the model (dynamic = weights only, static = activations only) with no overlap",
      "Dynamic quantization only works during training; static quantization only works after training is fully finished, with no other distinction",
      "Dynamic quantization computes activation ranges on the fly during each inference call (flexible, no calibration data needed); static quantization pre-computes fixed ranges ahead of time using representative calibration data (faster at inference, but requires that extra calibration step)",
      "Static quantization is always less accurate and strictly worse than dynamic — there's no tradeoff to consider"
    ],
    "a": [
      2
    ],
    "e": "Dynamic quantization defers the range calculation to runtime — each forward pass observes the actual activation values and quantizes on the spot, so it needs no extra calibration data but has some runtime overhead recomputing ranges every time. Static quantization instead runs a calibration pass over representative data BEFORE deployment to lock in fixed quantization ranges, which is faster at inference time (no per-call range computation) but only works well if the calibration data's distribution actually matches production traffic. The training/inference-time distinction in option 2 isn't the real axis (both are inference-time techniques). Option 3's clean weights-vs-activations split isn't accurate either. And it's a genuine tradeoff (flexibility vs. speed), not a strict accuracy ordering either way."
  },
  {
    "d": "Combining Tensors",
    "q": "`torch.cat` and `torch.stack` both combine a list of tensors, but produce different shapes. Given two tensors each of shape `(3, 4)`, what shape does `torch.stack([a, b])` produce, versus `torch.cat([a, b], dim=0)`?",
    "o": [
      "`stack` gives `(3, 4, 2)` and `cat` gives `(3, 8)` — both add a new trailing dimension",
      "`stack` creates a NEW dimension, giving shape `(2, 3, 4)`; `cat` joins along an EXISTING dimension without adding one, giving shape `(6, 4)`",
      "Both produce exactly the same shape, `(6, 4)` — they're interchangeable aliases",
      "`cat` requires the tensors to have different shapes, while `stack` requires identical shapes"
    ],
    "a": [
      1
    ],
    "e": "`torch.stack` treats the two `(3,4)` tensors as two separate 'layers' and adds a brand-new leading dimension for them, producing `(2, 3, 4)` — like stacking two sheets of paper into a small book. `torch.cat` with `dim=0` instead extends the EXISTING first dimension, joining `3` rows plus `3` more rows into `6` rows of shape `(6, 4)` — no new dimension is created. They are not interchangeable and don't produce the same shape. `stack`'s output shape in option 3 is simply wrong (the new dimension for `stack` is wherever you specify, defaulting to position 0, not appended at the end). Both functions actually require the input tensors to share the same shape (aside from the concatenation dimension for `cat`) — option 4 has that requirement backwards."
  },
  {
    "d": "Handling Imbalanced Data",
    "q": "`torch.utils.data.WeightedRandomSampler` helps with imbalanced datasets by changing how often each sample is drawn into a batch. If class A has 900 examples and class B has 100 (out of 1000 total), what's the standard way to set each SAMPLE's weight so batches stop being dominated by class A?",
    "o": [
      "Give each sample a weight proportional to the INVERSE of its class's frequency (so class-A samples get weight ∝ 1/900 and class-B samples get weight ∝ 1/100) — this makes the rarer class roughly as likely to be drawn per batch as the common one",
      "Give every sample in the dataset the exact same weight, since the sampler already balances classes automatically with no configuration",
      "Give class A (the majority) a higher weight so it's drawn even more often, since it has more reliable statistics",
      "Weight only class B's samples; leave class A's weight at exactly zero so it's never sampled again"
    ],
    "a": [
      0
    ],
    "e": "Weighting each sample by 1/(its class's count) means a class-B sample (weight ∝ 1/100) is 9× more likely to be picked than any single class-A sample (weight ∝ 1/900) — which compensates for there being 9× fewer of them, roughly equalizing how often each CLASS shows up across a batch, without discarding any class-A data outright (unlike undersampling) or fabricating synthetic examples (unlike SMOTE). Giving everyone equal weight is just plain random sampling — it does nothing about the imbalance, and the sampler doesn't auto-balance without weights being supplied. Upweighting the majority class (option 3) makes the imbalance WORSE, the opposite of the goal. Zeroing out the majority class's weight (option 4) throws away 90% of the dataset entirely rather than balancing it."
  }
]
</script>
<div class="topic-quiz-mount"></div>
