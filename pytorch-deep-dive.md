# PyTorch Deep Dive

This picks up where `deep-learning-practice.md` left off. Here you go one level deeper: what's actually happening under the hood, how to debug it when it breaks, and how to ship it.

Every snippet on this page actually ran, in a real session. Each cluster builds on the one before it, and ends with one small worked example that ties the pieces together.

---

> 🔗 **Hands-on reps:** [Code Drills 6 — Tensors & Autograd](/topic/code-drills-deep-learning#cluster-1-tensors-autograd)

## Cluster 1 — Autograd Internals: Custom Functions and Hooks

### 1. How do you write a custom autograd `Function`, with your own forward and backward math?
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
**Autograd** is PyTorch's automatic-gradient machine. Every tensor operation you run gets recorded into a graph. Call `loss.backward()`, and PyTorch walks that graph in reverse, applying the chain rule at each step. That's how gradients show up "for free" — you never derive any calculus by hand.

Writing a custom `Function` steps outside that automation, for one operation. You supply the forward math yourself. You supply its derivative yourself too.

One detail matters: `ctx.save_for_backward(x)`, not a plain Python attribute. This registers the tensor properly with autograd's memory management — it gets freed at the right time, and works correctly with `.detach()` and gradient checkpointing. Stashing a tensor as a plain `ctx` attribute works for small examples, but skips that bookkeeping. In a long training run, that can quietly leak memory.

The `backward` method here is the chain rule from `math-foundations-refresher.md`, written out by hand. PyTorch usually does this step for you automatically. This time, you're doing it yourself, on purpose, so you can see it happen.

### 2. How do you inspect what's flowing through a layer during a real forward pass, without touching its code?
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
A forward hook lets you look at what's flowing through a layer, without changing the model's code at all. Register a hook function, and PyTorch calls it every time that layer runs — you log the activation, its shape, whatever you need.

This matters most when you're debugging someone else's model, and editing `forward()` isn't an option, or is too risky. It's also how you'd check for dead ReLUs — units stuck at exactly 0 — or build a tool that visualizes activations.

### 3. How do you inspect gradients flowing backward through a specific layer?
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
Forward hooks watch data going forward. To watch gradients going backward, use `register_full_backward_hook`.

Use the "full" version, not the older `register_backward_hook`. The older one had documented, inconsistent behavior on modules with multiple inputs or outputs. The full version is the corrected, currently-recommended way to do this — and it's how you'd reliably catch a vanishing or exploding gradient at the exact layer where it starts.

### Summary example
Say a model trains slower than expected. Put a forward hook on each layer. One ReLU layer shows 80% dead units — a real capacity problem. Put a backward hook on that same layer. Gradient norms flowing into it are near zero too, confirming those dead units aren't getting any learning signal either. Neither hook touched the model's `forward()` method at all.

---

## Cluster 2 — Mixed Precision and Distributed Training

### 1. How do you run mixed-precision training, end to end?
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
Start with the vocabulary. **fp16** means 16-bit floating-point numbers. That's half the memory of the usual 32-bit kind, and much faster on modern GPUs. The tradeoff: fp16 can only represent a much narrower range of values.

**Mixed precision** means doing most of the math in fp16, but keeping the numerically fragile parts — weight updates, the loss itself — in fp32. You get most of the speed, with little of the risk.

The risk that's left: fp16's narrow range means small gradients can underflow all the way to exactly zero. `GradScaler` handles this in three steps.
1. `scaler.scale(loss).backward()` — multiplies the loss by a large factor before running backward, so the gradients land in a safer range.
2. `scaler.step(optimizer)` — divides that factor back out, checks for `inf`/`nan`, then steps the optimizer. If it finds `inf`/`nan`, it silently skips the update instead.
3. `scaler.update()` — adjusts the scale factor for next time.

Skip `GradScaler` and hand-roll fp16 training yourself, and this is exactly the kind of bug you get: a loss that silently turns into `nan`, with no obvious cause.

Picture it as inflating the numbers before the danger zone, then deflating them back after:
```
loss ──▶ [ × big scale factor ] ──▶ SAFE ZONE: backward() ──▶ [ ÷ scale factor ] ──▶ optimizer.step()
              scaler.scale()          fp16 gradients no       scaler unscales      real-sized
                                       longer underflow        internally           gradients
                                       to zero here             before this check
```
Think of it like zooming in before a photo, then zooming back out. Three separate calls exist because three separate things happen: inflate, compute safely, deflate-and-check. That's not API verbosity — each call is doing real work.

### 2. How do you scale that same training loop across multiple GPUs (DDP)?
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
Here's what DDP (DistributedDataParallel) actually does, mechanically.
1. Each GPU runs its own complete copy of the model, as its own separate process. Each copy is called a **rank**.
2. Each rank trains on its own slice of the data — a **shard** — and no two ranks see the same shard.
3. After every backward pass, all the ranks average their gradients together, before anyone steps the optimizer.
4. That averaging keeps every copy of the model bit-identical, while the group collectively sees N times the data per step (N = number of GPUs).

One easy-to-miss requirement falls out of this: `sampler.set_epoch(epoch)`. Skip it, and `DistributedSampler` uses the exact same shuffle order every single epoch, on every rank. Calling it at the top of each epoch reseeds the shuffle using the epoch number, so every rank reshuffles consistently but differently each time. Without it, training quietly loses shuffling's regularization benefit — it keeps seeing the same batch order over and over.

### Summary example
Say a model is too big to train on one GPU's compute budget in a reasonable amount of time. `GradScaler` + `autocast` (question 1) roughly halves memory and often speeds up each individual GPU's step. `DistributedSampler` + `DDP` (question 2) splits the data across multiple GPUs, so each processes a different shard per step. These two techniques solve different problems — one makes each GPU's step more efficient, the other scales you across more GPUs — so they're independent, and routinely used together.

---

## Cluster 3 — Custom Losses and Deliberate Weight Initialization

### 1. How do you write a custom loss function that still lets gradients flow through it?
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
The rule: keep the entire computation in PyTorch tensor operations. No `.item()`, no NumPy, mid-computation.

Calling `.item()` or converting to NumPy pulls the value out of the autograd graph. Once that happens, nothing computed from it can send a gradient back through. A custom loss that "runs" fine, but silently gives you `None` for `.grad`, almost always has exactly this bug hiding somewhere: a stray `.item()`, a `.numpy()` call, or a plain Python `if` on a tensor's value instead of `torch.where`.

### 2. Does the starting point of training — weight initialization — matter too?
```python
def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")   # matched to ReLU's expected activation scale
        nn.init.zeros_(module.bias)

model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 2))
model.apply(init_weights)     # applies the function to EVERY submodule recursively
```
Yes. Where you start matters, separately from whether the loss itself is correct.

Kaiming (He) initialization is built for ReLU specifically. ReLU zeros out roughly half its inputs, so Kaiming scales the initial weights up to compensate — keeping activation variance stable across layers at the very start of training.

Xavier/Glorot initialization is the other common choice, and it assumes a symmetric activation like tanh instead.

Use the mismatched one, and nothing crashes. Training just gets slower and less stable in the early going — in a way that's easy to blame on something else entirely.

### Summary example
Say a model with a custom loss trains, but the loss barely budges for the first several epochs. Check `pred.grad` — gradients ARE flowing, so the loss function (question 1's discipline was followed correctly) is fine. The real culprit turns out to be `nn.init.xavier_normal_`, applied to a network full of ReLUs (question 2's mismatch). Switch to `kaiming_normal_`, and the slow start fixes itself. The loss function never needed to change at all.

---

## Cluster 4 — Transformer Building Blocks, Off the Shelf

### 1. How do you use PyTorch's built-in Transformer encoder block, instead of hand-rolling attention?
```python
encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=256, batch_first=True)
encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
x = torch.randn(2, 10, 64)     # [batch, seq_len, d_model]
out = encoder(x)                # [2, 10, 64] -- same shape, refined representation
```
Building attention from scratch (`nca-genl`'s §1.4) exists to prove you actually understand the mechanism. `nn.TransformerEncoderLayer` is the version you'd use for real: fused kernels, correct default initialization, edge cases already tested. Once you understand the mechanism, this is what you reach for. Knowing both — and knowing when each one is the right call — is the actual skill.

### 2. How do you make it causal (GPT-style, autoregressive)?
```python
seq_len = 10
causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len)   # built-in helper, upper triangle = -inf
out = encoder(x, mask=causal_mask, is_causal=True)
```
Two words are being toggled here.
- **Bidirectional** means every position can attend to the whole sequence — past and future both. That's BERT-style, and it's right for *understanding* a complete piece of text you already have.
- **Causal** (also called autoregressive) means each position can only see what came before it. That's GPT-style, and it's required for *generating* text left to right — at generation time, the future words don't exist yet.

The mask enforces this by blanking out the upper triangle of the attention grid.

One API detail worth knowing: `is_causal=True` is passed as a separate flag, on top of the mask itself. Recent PyTorch versions use a faster, fused-attention code path specifically when you tell it the mask is causal, instead of treating it as a generic arbitrary mask. The math comes out identical either way — the flag is just what unlocks the faster implementation.

### 3. How do you sanity-check the model's actual size, before training?
```python
def count_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total

trainable, total = count_params(encoder)
print(f"{trainable:,} trainable / {total:,} total")
```
A typo in a layer's dimensions can silently build a model 100x larger, or smaller, than you intended. Run this the moment you build the model. It catches the mistake immediately — instead of you discovering it hours later, from a training loop that feels suspiciously slow or suspiciously fast.

### Summary example
Say you're building a small GPT-style model. `nn.TransformerEncoderLayer` (question 1) gives you the optimized attention and feed-forward blocks. `generate_square_subsequent_mask` plus `is_causal=True` (question 2) makes it autoregressive instead of bidirectional. Run `count_params` (question 3) right after construction, and confirm the parameter count matches what you expected. That catches a `d_model` or `nhead` typo before a single training step runs — not after.

---

## Cluster 5 — Reproducibility and the `eval()`/`no_grad()` Distinction

### 1. How do you make a PyTorch run fully reproducible, not just "seeded"?
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
Setting `torch.manual_seed` alone is not enough.

Python's `random`, NumPy's RNG, and PyTorch's own CPU and GPU RNGs are all independent of each other. A data augmentation pipeline using plain `random.random()` won't respect `torch.manual_seed` at all — it needs its own seed.

There's a second trap: `cudnn.benchmark=True` is a common speed optimization, and it lets cuDNN pick different algorithms from run to run, based on timing. That's fast. It's also explicitly not reproducible — turn it off if you need the same result every time.

### 2. What's the actual difference between `model.eval()` and `torch.no_grad()` — a real, common mix-up?
```python
model.eval()                 # changes LAYER BEHAVIOR: BatchNorm uses running stats, Dropout turns off
with torch.no_grad():         # changes AUTOGRAD: stops tracking gradients, saves memory or during inference
    preds = model(torch.randn(4, 10))
```
These two do genuinely different things.

`model.eval()` alone still tracks gradients. During pure inference, that wastes memory for no reason.

`torch.no_grad()` alone leaves BatchNorm and Dropout in training-mode behavior — wrong running statistics, wrong randomness, for real inference. This is the same distinction already drawn in `deep-learning-practice.md`.

Production inference code needs both, together. A validation loop that runs during training also needs `model.eval()` — but remember to call `model.train()` again afterward, before training resumes.

### Summary example
Say you need to report a benchmark number a reviewer can exactly reproduce. `set_seed(42)` (question 1) alone isn't enough if the eval loop forgets `torch.no_grad()` (question 2). That particular bug wouldn't change the accuracy number you report. But it would silently waste memory — and if that eval code later gets a backward call added to it, it could inject real nondeterminism, precisely because gradients were being tracked when they shouldn't have been.

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
Call `.eval()` before tracing. This is not optional.

Tracing records the actual operations run for one example input. In train mode, Dropout and BatchNorm behave differently — randomly, or using batch statistics — than they do at serving time. Trace in train mode, and the traced graph permanently bakes in that wrong, training-mode behavior.

Tracing has a real limitation too: it only records the one control-flow path taken for that specific example input. If your model has a genuine data-dependent `if` branch, tracing can silently miss the other branch entirely. For that, use `torch.jit.script` instead — it parses the actual Python control flow, rather than replaying one recorded execution.

### 2. What if the deployment target isn't PyTorch at all — a C++ or browser runtime?
```python
torch.onnx.export(
    model, example_input, "model.onnx",
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},   # allow variable batch size at inference
)
```
**ONNX** is a framework-neutral file format for trained models. Export once, and any runtime that speaks ONNX — a C++ server, a browser, a mobile app — can run it, with no PyTorch installed at all.

One export detail matters more than it looks like it should: `dynamic_axes`. Skip it, and the exported ONNX graph hardcodes the exact batch size used during export — here, 1. Any real serving system needs to handle varying batch sizes. Marking dimension 0 as dynamic is what makes the exported model usable for real traffic, instead of accepting exactly one request at a time, forever.

### Summary example
Say you're deploying the same model two different ways. TorchScript (question 1), for an environment that's Python-free but still runs PyTorch. ONNX (question 2), for a genuinely cross-framework C++ inference server. Both need `.eval()` mode first. ONNX specifically also needs `dynamic_axes` set — skip it, and the exported model only ever accepts exactly one request at a time, batch size 1, forever.

---

## Cluster 7 — Specialized Architectures: Autoencoders and GANs

### 1. How do you build a simple autoencoder, and use it for anomaly detection with zero labeled anomalies?
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
An **autoencoder** is a network trained to do one thing: reproduce its own input, after squeezing it through a deliberately narrow middle layer. That narrow middle is called the **latent** vector — here, 8 numbers standing in for the original 20.

The squeeze is the whole point. To reconstruct well through a bottleneck that narrow, the network can't just copy the input. It's forced to learn the data's real, underlying patterns instead.

Train it only on normal data, and it gets good at compressing and reconstructing normal patterns. Feed it a genuinely anomalous input, and that input doesn't match what it learned — so it reconstructs poorly. The per-sample reconstruction error becomes a usable anomaly score, and you never needed a single labeled anomalous example to get there.

### 2. How does a GAN's generator/discriminator pair learn to generate new data, and why does training alternate two updates?
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
Notice `.detach()` shows up in step 1 but not step 2. That's not arbitrary — it follows directly from what's being trained at each step.

In step 1, only the discriminator is training. A gradient flowing back into the generator here would be wasted work, so `.detach()` cuts it off.

In step 2, the whole point is the opposite: get a gradient signal INTO the generator, by backpropagating through the discriminator (frozen for this round). `fake_data` must stay attached to the graph here.

Forgetting to detach in step 1 just wastes some compute. Forgetting to leave it attached in step 2 is the real bug — it silently produces zero gradient for the generator, and the generator never learns.

### Summary example
An autoencoder (question 1) and a GAN's generator (question 2) both learn a compressed, latent representation of data — but for opposite purposes. The autoencoder is scored by how well it reconstructs its own input. A bad reconstruction IS the anomaly signal. The GAN's generator is scored by whether a separate discriminator can tell its output apart from real data at all — useful for generating new, realistic-looking data instead. The alternating `.detach()` / no-`.detach()` pattern in the GAN loop is what keeps these two networks' training signals from bleeding into each other on the wrong step.

---

## Cluster 8 — Cross-Validation for a Neural Network, Not Just a sklearn Model

### 1. How do you do the same k-fold discipline for a PyTorch model that `sklearn-practice.md`'s `cross_val_score` does for sklearn models?
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
One rule matters more than any other here: build a fresh model inside the fold loop. Don't reuse the same model instance across folds.

Reuse it, and fold 2's training starts from fold 1's already-trained weights, not from scratch. That leaks information across folds. The resulting CV estimate stops meaning anything as a measure of how the architecture performs on genuinely unseen data.

`sklearn`'s `Pipeline` handles this automatically, by design (`sklearn-practice.md`, Cluster 1). A hand-rolled PyTorch loop has no such safety net — you have to enforce it yourself, explicitly, every single fold.

### Summary example
Evaluating a small architecture across 5 folds gives `mean acc: 0.780 +/- 0.045`. That `+/- 0.045` matters as much as the mean does — the same "report the spread, not just the average" discipline from `sklearn-practice.md`'s cross-validation cluster. And it's only a trustworthy number because a brand-new, randomly-initialized model got created at the top of each fold's loop — not one model trained incrementally across all 5 folds.

---

## Cluster 9 — Diffusion Models: A Third Way to Generate Data

### 1. How does a diffusion model generate new data, through a third, completely different mechanism?

An autoencoder reconstructs its own input (Cluster 7, Q1). A GAN's generator learns by fooling a discriminator (Cluster 7, Q2). A diffusion model generates new data through a third, completely different mechanism.

A diffusion model has two phases. Only one of them gets trained.

The **forward process** is fixed, known math. No training involved at all. Take a real data point. Add a little Gaussian noise. Repeat, over many steps, until nothing recognizable is left.

The **reverse process** is where the learning happens. A network is trained to undo one small noising step at a time. Start from pure random noise, apply that trained undo-step repeatedly, and you end up with a brand-new, realistic sample.

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

Here's a real, computed run — not just an illustration. Take a toy 4-pixel "image," `x0 = [1, 1, 1, 1]` (a uniform bright patch). Use `T=10` steps, with a linear schedule from `beta=0.001` to `beta=0.20`.

| t | alpha_bar_t | signal weight (√alpha_bar) | noise weight (√1-alpha_bar) | x_t (4 toy pixels) |
|---|---|---|---|---|
| 1 | 0.999 | 0.999 | 0.032 | `[0.99, 1.02, 0.99, 0.99]` |
| 3 | 0.932 | 0.965 | 0.261 | `[0.72, 0.91, 1.26, 1.08]` |
| 5 | 0.791 | 0.890 | 0.457 | `[1.36, 1.00, 1.07, 0.97]` |
| 7 | 0.609 | 0.780 | 0.625 | `[-0.26, 1.32, 1.10, 1.09]` |
| 9 | 0.423 | 0.650 | 0.760 | `[-0.63, -0.67, -0.03, 0.29]` |
| 10 | 0.338 | 0.582 | 0.814 | `[0.83, 0.54, 1.01, 0.06]` |

Two things move in this table, and both are guaranteed by the schedule, not random. The signal weight shrinks steadily, from 0.999 down to 0.582. The noise weight grows steadily, from 0.032 up to 0.814.

Any single draw of `x_t` still looks random, because it genuinely is random — fresh Gaussian noise gets added at every step. But look at the trend. By `t=9`, the pixel values (`[-0.63, -0.67, -0.03, 0.29]`) no longer resemble the original uniform `[1,1,1,1]` pattern at all. At `t=1`, they're still obviously close to it. That's the "clean signal turns into static" story — just with real numbers behind it.

### 2. Why can't the reverse process just run the forward formula backward — why does undoing the noise require training a neural network at all?

The forward formula computes `x_t` from a known `x0`. That's a fully determined calculation — plug in the numbers, get an answer.

Running it backward would mean computing `x0` from `x_t`. But at generation time, there's no real `x0` to find. That's the exact thing you're trying to generate. There's no formula for "what was the original clean image," because infinitely many different clean images could all have produced this same noisy `x_t`.

So instead, a neural network — in practice, usually a U-Net for images — is trained to make a statistical best guess. Given a noisy `x_t` and the timestep `t`, predict the specific noise that got added. It learns this from millions of real (image, noise) pairs during training.

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

Look closely at that loss: it's a plain MSE regression against known noise. No adversarial second network. No minimax game — a real contrast with the GAN loop in Cluster 7, Q2.

Generation itself works like this, and there's no training data involved at all:
1. Start from `x_T`, which is pure random noise.
2. For `t = T` down to `1`: ask the trained model to predict the noise in `x_t`.
3. Subtract a scaled portion of that prediction, to step from `x_t` to `x_{t-1}`.
4. Repeat until you reach `x_0` — a brand-new sample that was never in the training set.

```
clean image ──add noise──▶ ... ──add noise──▶ pure static     (forward: fixed math, zero training)
    x_0                        x_t                  x_T

pure static ──predict & remove noise──▶ ... ──▶ clean image   (reverse: LEARNED, one small step at a time)
    x_T          (the trained TinyDenoiser)            x_0
```
Think of it as sculpting from a block of marble, not painting on a blank canvas. The model never builds an image up from nothing, in one shot. It starts with a solid block of noise, and over many steps, chips away exactly the part that doesn't belong — one predictable-sized chip, a learned noise estimate, at a time.

### Summary example
All three generative approaches on this page learn a different thing, scored a different way.
- The **autoencoder** (Cluster 7, Q1) learns to compress and reconstruct its own input, scored by reconstruction error, in one deterministic pass.
- The **GAN** (Cluster 7, Q2) learns to fool a second, adversarial network, scored by a shifting minimax game, also in one pass.
- The **diffusion model** learns to predict noise, scored by a plain, stable MSE loss — no adversary, no minimax — but needs dozens to thousands of small sequential passes instead of one.

That training stability — a boring regression loss, instead of two networks actively fighting each other — is a big part of why diffusion overtook GANs as the dominant approach behind Stable Diffusion, DALL-E, and Midjourney. The real cost: slower generation.

---

## Practice Q&A (Self-Test)

**Q1. In `ClampedSquare`, why use `ctx.save_for_backward(x)` instead of just storing `x` as a plain attribute on `ctx`?**
A: `save_for_backward` registers the tensor properly with autograd's memory management. It gets freed at the right time, and works correctly with things like `.detach()` and checkpointing. Stashing a tensor as a plain `ctx` attribute works for simple cases, but skips that bookkeeping — and can leak memory in a long training run.

**Q2. What does `register_forward_hook` let you do that plain code can't, and what does the `relu_out` hook example in this file actually compute?**
A: It lets you inspect or log intermediate activations, gradients, or shapes without touching the model's `forward()` method at all — useful for debugging someone else's model, or building a visualization tool. The example hook captures the ReLU layer's output, and computes the fraction of dead ReLU units (activations at exactly 0) in that batch.

**Q3. Why use `register_full_backward_hook` instead of the older `register_backward_hook`?**
A: The older `register_backward_hook` had documented, inconsistent behavior on modules with multiple inputs or outputs. `register_full_backward_hook` is the corrected, currently-recommended version for reliably inspecting gradients at a specific layer — for example, to catch a vanishing or exploding gradient at the exact layer where it starts.

**Q4. Walk through why `GradScaler` training uses three separate calls — `scaler.scale(loss).backward()`, `scaler.step(optimizer)`, `scaler.update()` — instead of just calling `backward()`/`step()` directly under `autocast`.**
A: fp16 has a narrow representable range, so small gradients can underflow to exactly zero. `scale()` multiplies the loss up before `backward()`, so gradients land in a safer range. `step()` unscales the gradients, checks for `inf`/`nan`, and steps the optimizer — or silently skips the update if it's unstable. `update()` adapts the scale factor for next time. Hand-roll fp16 training without this pattern, and a common result is a silent `nan` loss.

**Q5. In the DDP training skeleton, why is `sampler.set_epoch(epoch)` called at the start of every epoch?**
A: Without it, `DistributedSampler` uses the same shuffle order every epoch, across every rank. Calling `set_epoch` reseeds the shuffle using the epoch number, so training doesn't see the exact same batch order over and over — preserving shuffling's regularization benefit.

**Q6. In `WeightedMSE`, why must the entire computation stay in PyTorch tensor ops (e.g. `torch.where`) rather than using `.item()`, `.numpy()`, or a Python `if` on a tensor's value?**
A: Calling `.item()` or converting to NumPy detaches the value from the autograd graph, so any computation done outside PyTorch tensor operations can't send a gradient back through it. A custom loss that "runs" but silently returns `None` for `.grad` almost always has exactly this bug hiding somewhere.

**Q7. Why does Kaiming (He) initialization pair specifically with ReLU, and what actually goes wrong if you use Xavier/Glorot instead?**
A: Kaiming init accounts for ReLU zeroing out roughly half its inputs, and scales the initial weights up to keep activation variance stable at the start of training. Xavier/Glorot assumes a symmetric activation, like tanh. Use the mismatched one, and nothing crashes — it just makes early training slower and less stable, in a way that's easy to blame on something else.

**Q8. When adding a causal mask to `nn.TransformerEncoderLayer` for GPT-style autoregressive use, why pass `is_causal=True` in addition to the mask itself?**
A: Recent PyTorch versions can use a faster, fused-attention code path specifically when told the mask is causal, instead of treating it as a generic arbitrary mask. Passing the flag lets PyTorch pick that faster implementation, for the exact same mathematical result — a real, if version-dependent, performance difference.

**Q9. In the GAN training loop, why does `.detach()` appear on `fake_data` in the discriminator step but not in the generator step?**
A: In the discriminator step, only the discriminator is training, so a gradient flowing back into the generator would be wasted work — hence `.detach()`. In the generator step, the whole point is getting a gradient signal into the generator, by backpropagating through the (frozen, this round) discriminator, so `fake_data` must NOT be detached there. Forgetting to detach in step 1 wastes some compute. Forgetting to leave it attached in step 2 is the real bug — it silently produces zero gradient for the generator.

**Q10. Why does `torch.onnx.export` use `dynamic_axes={"input": {0: "batch_size"}, ...}`, and what happens if you omit it?**
A: Without `dynamic_axes`, the exported ONNX graph hardcodes the exact batch size used during export (e.g. 1). Marking dimension 0 as dynamic lets the exported model handle varying batch sizes at inference, which is what makes it usable for real serving traffic, instead of accepting one fixed batch size forever.

**Q11. In a diffusion model, why is the forward (noising) process computed directly from a fixed formula while the reverse (denoising) process has to be learned by a neural network?**
A: The forward process computes a known quantity — `x_t`, derived from a known `x0` plus a known amount of noise — so it's a deterministic calculation, no training needed. The reverse process would require computing `x0` from `x_t`, but at generation time there's no real `x0`. Infinitely many clean images could have produced the same noisy `x_t`. A network has to learn a statistical best guess — predict the noise — from real training examples, because there's no formula that can recover information that was genuinely destroyed.

**Q12. A diffusion model's training loss is a plain MSE between predicted and actual noise — no discriminator, no adversarial game. Why does this make diffusion models more stable to train than GANs?**
A: A GAN's generator and discriminator are locked in an adversarial minimax game. If one network gets too strong too fast, the other stops getting a useful gradient signal — a classic GAN failure mode, like discriminator collapse. A diffusion model's noise-prediction objective is a single, fixed regression target at every step. There's no second network to destabilize the loss landscape. It's ordinary supervised learning, repeated many times — which is why diffusion training rarely suffers the mode collapse and instability that plague GAN training.

---

## Video-Sourced Practice MCQs (Set 2)

A second practice set for PyTorch, built the same way as this hub's NCA-GENL community bank. Topics were checked against a real YouTube PyTorch-interview-prep video, then written up here as fully original multiple-choice questions — the source video mostly gave prose explanations, not MCQs, so every option and explanation below is original, written to match this hub's "explain every option" convention, not copied from the video.

These questions cover ground the clusters above don't touch: tensor memory sharing, buffers vs. parameters, `retain_graph`, loss numerical stability, quantization, tensor combination ops, and imbalanced-data sampling.

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
