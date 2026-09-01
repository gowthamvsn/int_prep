# Neural Net Numericals — Forward Pass, Backward Pass, Loss, Weight Updates, Shapes

Every other doc in this hub explains these ideas in words and code. This one is different. It's pure arithmetic.

You'll get real numbers for weights, biases, and inputs. You compute the forward pass by hand. Then the loss. Then you backpropagate the gradient by hand, and produce the updated weight. This is exactly the kind of "walk me through the numbers" question interviewers ask, when they want to know you understand backprop as more than a `.backward()` call.

Every number below was computed and checked with NumPy. None of it is hand-guessed. First comes a full worked toy network, step by step. Then a 20-question quiz that drills the same skills: forward pass, loss, gradients, weight updates, epoch/iteration counting, and shape arithmetic.

## The toy network used throughout

A minimal 2-input → 2-hidden(sigmoid) → 1-output(sigmoid) network. Small enough to trace every number by hand, big enough to actually need backprop — not just a single weight:

```
x  = [0.5, 0.8]                      # 2 inputs
W1 = [[0.1, 0.4],                    # hidden unit 1's weights, from [x1, x2]
      [0.3, 0.2]]                    # hidden unit 2's weights, from [x1, x2]
b1 = [0.1, -0.2]                     # one bias per hidden unit
W2 = [0.5, -0.3]                     # output weights, from [hidden1, hidden2]
b2 = 0.2
y_true = 1                           # binary target
lr = 0.1                             # learning rate for the SGD step
```

### Forward pass

**Hidden layer.** Each hidden unit computes a pre-activation `z` first. Take the dot product of its weight row and the input. Then add its bias:

```
z1_1 = (0.1)(0.5) + (0.4)(0.8) + 0.1  = 0.05 + 0.32 + 0.1  = 0.47
z1_2 = (0.3)(0.5) + (0.2)(0.8) + (-0.2) = 0.15 + 0.16 - 0.2 = 0.11
```

Now apply sigmoid, `σ(z) = 1/(1+e^-z)`, to turn each `z` into an activation:

```
a1 = [σ(0.47), σ(0.11)] = [0.6154, 0.5275]
```

**Output layer.** Same pattern as before. Dot product of `W2` with the hidden activations, then add `b2`:

```
z2 = (0.5)(0.6154) + (-0.3)(0.5275) + 0.2 = 0.3077 - 0.1582 + 0.2 = 0.3494
a2 = σ(0.3494) = 0.5865        <- this is the network's prediction
```

### Loss

Two common loss choices. Both use the same `a2 = 0.5865` and `y = 1`:

```
MSE loss = 0.5*(y - a2)^2 = 0.5*(0.4135)^2 = 0.0855
BCE loss = -[y*log(a2) + (1-y)*log(1-a2)] = -log(0.5865) = 0.5336   (second term drops out since y=1)
```

### Backward pass (using MSE, the classic hand-worked case)

Backprop is just the chain rule. Apply it one layer at a time, working backward.

**Output layer's error term.** This measures how much the loss wants `z2` to change:

```
δ2 = dL/da2 * da2/dz2
   = -(y - a2) * a2*(1-a2)              <- -(y-a2) is MSE's derivative; a2*(1-a2) is sigmoid's derivative
   = -(1 - 0.5865) * (0.5865 * 0.4135)
   = -0.4135 * 0.2425
   = -0.1003
```

**Gradients for the output weights.** Each one is just `δ2` times whichever hidden activation feeds that weight:

```
dL/dW2 = δ2 * a1 = -0.1003 * [0.6154, 0.5275] = [-0.0617, -0.0529]
dL/db2 = δ2 = -0.1003
```

**Backprop into the hidden layer.** Push `δ2` back through `W2`. Then multiply by the hidden layer's own sigmoid derivative:

```
δ1 = (W2 * δ2) ⊙ a1*(1-a1)
   = ([0.5,-0.3] * -0.1003) ⊙ [0.2367, 0.2496]
   = [-0.0501, 0.0301] ⊙ [0.2367, 0.2496]
   = [-0.0119, 0.0075]
```

**Gradients for the input weights.** Each `δ1` element times whichever input feeds it:

```
dL/dW1 = outer(δ1, x) = [[-0.0119*0.5, -0.0119*0.8],
                          [ 0.0075*0.5,  0.0075*0.8]]
       = [[-0.0059, -0.0095],
          [ 0.0037,  0.0060]]
dL/db1 = δ1 = [-0.0119, 0.0075]
```

### Weight update (one SGD step, `lr = 0.1`)

`w_new = w_old - lr * gradient`. Every weight and bias in the network follows this exact same rule:

```
W2_new = [0.5, -0.3]      - 0.1*[-0.0617, -0.0529]  = [0.5062, -0.2947]
b2_new = 0.2              - 0.1*(-0.1003)            = 0.2100
W1_new = [[0.1, 0.4],     - 0.1*[[-0.0059,-0.0095],  = [[0.1006, 0.4009],
          [0.3, 0.2]]            [ 0.0037, 0.0060]]     [0.2996, 0.1994]]
b1_new = [0.1, -0.2]      - 0.1*[-0.0119, 0.0075]    = [0.1012, -0.2008]
```

Look at the signs. Every gradient that was negative belongs to a weight that needed to grow. Every gradient that was positive belongs to a weight that needed to shrink. That sign is exactly what "gradient descent" descends along.

### Second worked example: softmax + categorical cross-entropy

The sigmoid+MSE/BCE case above works for binary output — yes or no. For multi-class output, you use softmax plus categorical cross-entropy instead. Combine those two and you get an unusually clean gradient:

```
logits = [2.0, 1.0, 0.1]      true class = index 0
softmax: probs = e^logit / sum(e^logits) = [0.6590, 0.2424, 0.0986]
CCE loss = -log(probs[true_class]) = -log(0.6590) = 0.4170

dL/dlogits = probs - onehot(true_class) = [0.6590-1, 0.2424-0, 0.0986-0] = [-0.3410, 0.2424, 0.0986]
```

That `probs - onehot` result is clean. You don't even need to spell out the chain rule. This is exactly why softmax and cross-entropy almost always get used together, instead of mixed with other losses.

### ReLU and "dead neurons"

```
x = [-2.0, 3.0, 0.5, -0.1]   w = [1.5, -0.5, 2.0, 1.0]   b = 0.3
z = x·w + b = -3.0 + (-1.5) + 1.0 + (-0.1) + 0.3 = -3.3
ReLU(z) = max(0, -3.3) = 0
```

A negative pre-activation means ReLU outputs exactly `0`. ReLU's gradient is also `0` everywhere `z<0`. So this neuron passes **zero** gradient backward here. It contributes nothing to any upstream weight update. That's the "dying ReLU" failure mode, in one number.

### Epochs, iterations, and batch size

```
dataset size = 2000 samples,  batch size = 40
iterations per epoch = dataset_size / batch_size = 2000/40 = 50
training for 15 epochs -> total weight-update steps = 50 * 15 = 750
```

One **iteration** is one gradient step on one batch. One **epoch** is enough iterations to see every sample once. Batch size links the two. They are not interchangeable.

### Shape arithmetic

```
Conv2d output size = floor((in - k + 2p)/s) + 1

32x32 input, kernel 5, stride 1, pad 0  -> (32-5+0)/1 + 1 = 28    (shrinks — "valid" convolution)
32x32 input, kernel 3, stride 1, pad 1  -> (32-3+2)/1 + 1 = 32    (unchanged — "same" convolution)
28x28 input, kernel 3, stride 2, pad 1  -> (28-3+2)/2 + 1 = 14    (halved — strided downsampling)

Flattening a (channels=16, H=7, W=7) feature map before a Linear layer:
  flat length = 16*7*7 = 784
```

---

## Transformer forward & backward pass — hand-computed, toy scale

Same idea as the network above. This time it's applied to the mechanism that actually runs GPT, Llama, and Claude: attention.

Everything here stays toy-scale, so every number is checkable by hand. Vocab is `{the:0, cat:1, sat:2, on:3, mat:4}`. Model width is `d_model=4`. One attention head, `d_k=2`. FFN hidden size 2.

A real production model runs this exact same computation. It just uses bigger numbers: `d_model=4096+`, 32+ heads, 32 to 100 stacked layers. Bigger matrices. Nothing conceptually new. Every number below was verified with NumPy/PyTorch, not hand-guessed.

### Forward pass: "the cat sat" → predict "on"

**Step 0 — tokenize.** `"the cat sat"` → `[0, 1, 2]`

**Step 1 — embed + positional.** Embedding matrix `E` (5 rows, 4 columns, learned) works as a lookup table. A learned positional vector `p_i` gets added to each token, so the model knows the order:

```
E[the]=[0.2, 0.0, 0.0, 0.0]   E[cat]=[0.0, 1.0, 0.8, 0.0]   E[sat]=[0.0, 0.0, 1.0, 1.0]
p1=[0.0, 0.0, 0.1, 0.0]       p2=[0.0, 0.1, 0.0, 0.0]       p3=[0.1, 0.0, 0.0, 0.1]

x1 = E[the]+p1 = [0.2, 0.0, 0.1, 0.0]
x2 = E[cat]+p2 = [0.0, 1.1, 0.8, 0.0]
x3 = E[sat]+p3 = [0.1, 0.0, 1.0, 1.1]
```

**Step 2 — project: every token gets a query, key, value.** Three learned 4×2 matrices multiply every `x`. It's an ordinary matmul. Here's `W_Q` as an example:

```
W_Q = [[0,0],[0,1],[1,0],[0,0]]
q3 = x3 . W_Q = [0.1*0+0.0*0+1.0*1+1.1*0, 0.1*0+0.0*1+1.0*0+1.1*0] = [1.0, 0.0]

        q            k            v
the:  [0.1, 0.0]   [0.0, 0.2]   [0.0, 0.1]
cat:  [0.8, 1.1]   [1.1, 0.0]   [1.1, 0.8]
sat:  [1.0, 0.0]   [0.0, 0.1]   [0.0, 2.1]
```

**Step 3 — score: `Q·K^T`, causal mask, scale, softmax.** Every query dots every key. This is a decoder, so every future position (above the diagonal) gets set to `-inf` before softmax. That way no token can peek ahead:

```
            the    cat    sat
the  ->  [ 0.00    -inf   -inf ]
cat  ->  [ 0.22    0.88   -inf ]
sat  ->  [ 0.00    1.10   0.00 ]
```

Follow the "sat" row — that's the token predicting the next word. Scale by `sqrt(d_k) = sqrt(2)`, then apply softmax:

```
scaled  = [0, 1.10/1.41, 0] = [0, 0.78, 0]
softmax = [e^0, e^0.78, e^0] / sum = [1, 2.18, 1] / 4.18 = [0.24, 0.52, 0.24]
```

From learned weights alone, the mechanism decided the verb should attend 52% to its subject, "cat."

**Step 4 — mix: weighted average of values, then residual.**

```
z3 = 0.24*v_the + 0.52*v_cat + 0.24*v_sat
   = 0.24*[0.0,0.1] + 0.52*[1.1,0.8] + 0.24*[0.0,2.1] = [0.57, 0.94]
```

`W_O` (2×4) maps this back to model width: `[0.57, 0.94, 0.0, 0.0]`. The residual connection then adds the input back. This means the block *refines* the token instead of replacing it:

```
h3 = x3 + attn = [0.1, 0.0, 1.0, 1.1] + [0.57, 0.94, 0.0, 0.0] = [0.67, 0.94, 1.00, 1.10]
```

**Step 5 — FFN (where ReLU earns its keep).**

```
a = ReLU(h3 . W1)  with W1 columns [0,1,0,1] and [1,0,-1,0]:
  h3.col1 = 0.94+1.10 = 2.04      h3.col2 = 0.67-1.00 = -0.33
  a = ReLU([2.04, -0.33]) = [2.04, 0]     <- non-linearity just gated a signal off

ffn = a . W2 = 2.04*[0.1, 0.2, 0.0, 0.3] = [0.20, 0.41, 0.00, 0.61]
y3  = h3 + ffn = [0.87, 1.35, 1.00, 1.71]     (second residual)
```

**Step 6 — predict: logits over the vocab, softmax.** The LM head is a learned 4×5 matrix:

```
logits = [ the: 0.2   cat: 0.8   sat: 1.0   on: 3.1   mat: 1.9 ]
e^logit = [ 1.22       2.23       2.72       22.2      6.69 ]   (sum 35.1)
P       = [ 3.5%       6.4%       7.8%       63.3%     19.1% ]
```

Greedy decoding picks "on." The whole model, start to finish: lookup, project, score, mask, softmax, mix, FFN, logits, softmax.

**Step 7 — close both loops.** Say the training text continued "...on." The loss at this position is `cross_entropy = -ln(0.633) ≈ 0.46`. Backprop then pushes every matrix used — `E, W_Q/K/V/O, W1, W2, LM head` — to make that 0.633 bigger next time.

The masked score matrix means all three positions' predictions get computed in *one parallel pass*. One sentence gives you three training signals at once. That parallelism is why transformers beat RNNs.

One more thing, for generation: the `k` and `v` vectors for "the," "cat," and "sat" don't change on the next step. Storing them instead of recomputing them is exactly what a **KV cache** does.

*(Honesty note: this walkthrough skips LayerNorm around each sublayer, and skips multi-head attention. Both would change the numbers. Neither changes the story.)*

### Backward pass: the same sentence, every gradient, until "on" gets more likely

The `§Second worked example` above showed the softmax-plus-cross-entropy identity on a 3-class toy. This is the same identity. Here it gets propagated through *every matrix* in the transformer above.

**8a — the loss gradient.** When softmax feeds cross-entropy, `dL/dlogits = P - onehot(target)`. Every product and log-rule term cancels out:

```
P = [0.035, 0.064, 0.078, 0.633, 0.191]      onehot(on) = [0, 0, 0, 1, 0]
dL/dlogits = [+0.035, +0.064, +0.078, -0.367, +0.191]
```

Every *wrong* word gets a positive gradient, which pushes it down. The *correct* word, "on," gets the only negative entry, which pushes it up. Sanity check: the five numbers sum to about 0, since `sum(P) = sum(onehot) = 1`.

**8b — the LM head: a weight gradient is an outer product.** For any linear layer `z = y.W`, the rule is `dL/dW[:,j] = y * (dL/dlogits)_j`. Use the minimal-norm `W_out` consistent with the logits above: `W_out[:,on] = [0.415, 0.644, 0.477, 0.815]`.

```
dL/dW_out[:,on] = y3 * (-0.367) = [0.87,1.35,1.00,1.71] * (-0.367) = [-0.319, -0.494, -0.367, -0.626]

SGD update, lr=0.1:  dW_out[:,on] = -0.1 * above = [+0.032, +0.049, +0.037, +0.063]
```

Now recompute the forward pass with only the LM head updated. New logits: `[0.177, 0.759, 0.950, 3.338, 1.776]`. That gives **P(on) = 70.4%**, up from 63.3%. Loss falls from **0.46 to 0.35**. One gradient step, one layer, and the model is measurably more confident in the right answer. This isn't a metaphor. It's the actual arithmetic of training.

**8c — through the residual and FFN: the gradient highway, and a dead neuron.** The `+` in `y3 = h3 + FFN(h3)` sends the incoming gradient down *both* branches, unchanged. `dL/dy3 ≈ [-0.086, -0.133, -0.098, -0.168]` passes straight to `h3` through the residual. It also flows separately backward through the FFN. Recall `a = ReLU([2.04, -0.33]) = [2.04, 0]`. ReLU's derivative is 1 where the input was positive, and exactly 0 where it was negative:

```
dL/da1 = (W2 row1).dL/dy3 = -0.085   x  ReLU'(2.04)=1   -> passes through unchanged
dL/da2 = (anything)                 x  ReLU'(-0.33)=0   -> exactly zero, no matter what W2's row 2 contains
```

The second hidden neuron was clamped to 0 on the way forward. So no gradient reaches it, and none reaches the `W1` column that produced it either. A neuron that didn't fire doesn't learn on this step. That's "dead ReLU," in one real number — not an abstraction.

**8d — into attention: the softmax jacobian.** Only the first two entries of `dL/dh3` continue into attention, since `W_O` only wrote into `h3`'s first two slots: `dL/dz3 = [-0.086, -0.218]`. Recall `z3 = 0.24*v_the + 0.52*v_cat + 0.24*v_sat`. Two kinds of gradient fall out of this: one per attention weight, one per value. The weight-gradients then pass through softmax's own jacobian before they reach the raw scores: `dL/ds_i = weight_i * (dL/dweight_i - sum_j weight_j * dL/dweight_j)`.

```
sum_j weight_j * dL/dweight_j = 0.24(-0.022)+0.52(-0.268)+0.24(-0.458) = -0.254
dL/ds_the = 0.24*(-0.022-(-0.254)) = +0.056
dL/ds_cat = 0.52*(-0.268-(-0.254)) = -0.007
dL/ds_sat = 0.24*(-0.458-(-0.254)) = -0.049      (sum ~ 0.000 -- always true, a free correctness check)
```

A positive gradient on "the" means gradient descent will *lower* that attention score. Negative gradients on "cat" and "sat" mean both get *reinforced*. "Sat" — attending to itself — gets reinforced even more than "cat" here, because `v_sat=[0,2.1]` happens to carry a strong signal in exactly the direction the LM head rewards. Real gradients don't always match tidy intuition. This is what backprop actually decided.

**8e — into Q, K, V, and the embedding table.** Same recipe — local derivative times learning rate — propagates one level further. `W_V` gets its biggest update in the rows matching `x_cat`, since that's where the reinforced attention mass now points. `W_K` updates most in the rows matching `x_sat`, since "sat" now attends more to itself. `W_Q` updates only in the rows belonging to `x3="sat"`, the only token whose query this is.

Every delta here is small — `lr=0.1` on just one sentence. Real pretraining takes a tiny, noisy step like this from every sentence in a batch, repeated millions of times. Their average is what sculpts `W_Q` into "ask subject-shaped questions" and `W_V` into "carry the content worth copying." Nobody hand-designed those patterns. Training found them.

### The optimizer step: plain SGD vs. Adam, on the identical gradient

Take the single weight `W_out[1,on]`, gradient `g = -0.319` from 8b. Adam's defaults are `beta1=0.9, beta2=0.999`:

```
SGD:  dw = -lr*g = -0.1*(-0.319) = +0.032

Adam, step 1 (m0=v0=0):
  m1 = 0.9*0 + 0.1*g = -0.0319          v1 = 0.999*0 + 0.001*g^2 = 0.000102
  bias-corrected:  m_hat = m1/(1-0.9^1) = g        v_hat = v1/(1-0.999^1) = g^2
  dw = -lr * m_hat/(sqrt(v_hat)+eps) = -0.1 * g/|g| = -0.1*(-1) = +0.100
```

At step 1, Adam's update always collapses to exactly `+/-lr`. It only knows the gradient's *sign* so far, because `m_hat/sqrt(v_hat) = g/|g|` whenever the running averages are freshly seeded. Here that's a step **3x larger** than plain SGD.

But a parameter with a tiny gradient would get the exact same `+/-0.1` step. That's exactly why Adam needs bias-correction, and typically a learning-rate **warmup** too: its confident, uniform early steps can overshoot, before `v` has built up enough history to tell large gradients from small ones.

### From random init to a converged next-token predictor

Same toy architecture. But this time, every matrix starts as random noise instead of the hand-picked values above. It's trained on causal-LM next-token prediction over "the cat sat on": given "the," predict "cat"; given "the cat," predict "sat"; given "the cat sat," predict "on" — all three done in one masked parallel pass. Here's a real, seeded, verified training run:

```
step   loss    P(cat|the)   P(sat|the cat)   P(on|the cat sat)
0      1.52    21.2%        21.4%            23.4%     <- random init ~ uniform over 5 words, as expected
10     0.48    58.7%        56.1%            72.6%
30     0.02    97.4%        96.2%            99.2%
60     0.006   99.3%        99.0%            99.8%
100    0.003   99.7%        99.5%            99.9%
```

At step 0, every prediction sits near 20%. That's expected — with 5 possible words, a random model should guess close to uniformly. Checking for that is the first thing to verify about any from-scratch model: if random weights don't produce a roughly uniform distribution, initialization is broken before training even starts.

By step 30, the model has essentially memorized its one training sentence. That's expected at this tiny scale. It's also exactly why real pretraining needs trillions of tokens across millions of diverse documents, instead of one repeated sentence. A model that only ever sees "the cat sat on" learns to output "on" after that exact prefix, no matter what — that's memorization, not language understanding.

The mechanism itself — forward, loss, backward, gradient step, repeat — is identical at every scale. Only the data's size and diversity change.

### Where the parameters actually live — GPT-2 small, real dimensions

Same forward/backward mechanism. This time, counted at a real model's scale. GPT-2 small uses `d_model=768` and `h=12` heads, so each head works in `d_k = 768/12 = 64` dimensions.

```
Attention W_Q,W_K,W_V,W_O:  4 * (768*768)         ~ 2.4M params / block
FFN (4x expansion, 768->3072->768):                ~ 4.7M params / block
x 12 blocks:                                       ~ 85M
Token embeddings:  50,257 vocab * 768               ~ 38.6M
TOTAL                                               ~ 124M  <- exactly GPT-2 small
```

Two facts worth remembering here.

First: **roughly two-thirds of a transformer's parameters live in the feed-forward layers, not attention.** That's why Mixture-of-Experts swaps in multiple FFNs, not multiple attentions.

Second: self-attention costs `O(n^2 * d)` in sequence length `n`, since every token scores every other token. That's why long context windows are expensive, why KV-cache size grows the way it does, and why FlashAttention and sparse/linear-attention variants exist.

Residuals and LayerNorm aren't optional extras either. `x + Sublayer(x)` gives every gradient an untouched identity path back through 30 to 100 stacked layers — the same vanishing-gradient fix as the sigmoid-network case earlier in this doc, just built into the architecture. LayerNorm normalizes each *token's* own feature vector. BatchNorm can't do this job here, since it needs batch statistics that are meaningless across variable-length padded sequences.

---

## Numerical Practice Quiz (20 questions)

Every question below reuses the exact numbers computed above. Nothing here was eyeballed — each one was checked with NumPy before being written into an option.

<script type="application/json" class="topic-quiz-data" data-title="Neural Net Numericals (Forward/Backward Pass, Loss, Weight Updates, Shapes)">
[
  {
    "d": "Forward Pass",
    "q": "Using this toy network — x=[0.5, 0.8], hidden unit 1's row of W1 is [0.1, 0.4] with bias b1_1=0.1 — what is the pre-activation z for hidden unit 1?",
    "o": [
      "0.32",
      "0.47",
      "0.37",
      "0.86"
    ],
    "a": [
      1
    ],
    "e": "z = w·x + b = (0.1)(0.5) + (0.4)(0.8) + 0.1 = 0.05 + 0.32 + 0.1 = 0.47. 0.37 is what you get computing the dot product correctly but forgetting to add the bias. 0.86 comes from adding every number in sight (weights + inputs + bias) instead of multiplying weights by inputs first. 0.32 only uses the second weight-input product (0.4×0.8) and drops both the first term and the bias entirely."
  },
  {
    "d": "Forward Pass",
    "q": "Given z=0.47 for hidden unit 1 (sigmoid activation), what is a1 for that unit?",
    "o": [
      "0.47",
      "0.3846",
      "0.6154",
      "0.5"
    ],
    "a": [
      2
    ],
    "e": "sigmoid(0.47) = 1/(1+e^-0.47) ≈ 0.6154. Answering 0.47 skips applying the activation function entirely — that's just the raw pre-activation. 0.3846 is 1−0.6154, i.e. computing 1−sigmoid(z) (or sigmoid(−z)) instead of sigmoid(z). 0.5 wrongly applies the 'sigmoid of exactly 0 is 0.5' shortcut to a nonzero input."
  },
  {
    "d": "Forward Pass",
    "q": "Hidden activations are a1=[0.6154, 0.5275]. Output weights are W2=[0.5, -0.3], b2=0.2. What is z2, the output layer's pre-activation?",
    "o": [
      "0.1494",
      "0.4022",
      "0.6660",
      "0.3494"
    ],
    "a": [
      3
    ],
    "e": "z2 = (0.5)(0.6154) + (−0.3)(0.5275) + 0.2 = 0.3077 − 0.1582 + 0.2 = 0.3494. 0.6660 comes from flipping the sign on the second weight (treating −0.3 as +0.3), turning a subtraction into an addition. 0.1494 computes the dot product correctly but forgets to add the bias. 0.4022 mistakenly uses the pre-activations z1=[0.47, 0.11] instead of the actual activations a1, skipping the hidden layer's sigmoid."
  },
  {
    "d": "Forward Pass",
    "q": "z2 = 0.3494. What is a2, the network's final prediction (sigmoid output)?",
    "o": [
      "0.4135",
      "0.3494",
      "0.5",
      "0.5865"
    ],
    "a": [
      3
    ],
    "e": "sigmoid(0.3494) ≈ 0.5865. Answering 0.3494 again skips the activation function. 0.4135 is 1−0.5865 — the complement, as if computing sigmoid(−z2). 0.5 assumes any small pre-activation rounds to exactly one-half, ignoring the actual input value."
  },
  {
    "d": "Loss Calculation",
    "q": "True label y=1, prediction a2=0.5865. Using MSE loss L = 0.5(y−ŷ)², what is L?",
    "o": [
      "0.0855",
      "0.4135",
      "0.2068",
      "0.1710"
    ],
    "a": [
      0
    ],
    "e": "L = 0.5×(1−0.5865)² = 0.5×0.4135² = 0.5×0.1710 = 0.0855. 0.4135 is just the raw error (y−ŷ), with no squaring or halving at all. 0.1710 squares the error correctly but forgets the ½ factor. 0.2068 applies the ½ factor to the error but forgets to square it first (0.5×0.4135 instead of 0.5×0.4135²)."
  },
  {
    "d": "Loss Calculation",
    "q": "Same prediction a2=0.5865, true label y=1. Using binary cross-entropy L = −[y·log(ŷ) + (1−y)·log(1−ŷ)], what is L?",
    "o": [
      "0.4135",
      "0.5336",
      "1.0",
      "0.8831"
    ],
    "a": [
      1
    ],
    "e": "Since y=1, the second term vanishes and L = −log(0.5865) ≈ 0.5336. 0.8831 is −log(1−0.5865) — the loss you'd get if y were 0 instead of 1, penalizing the wrong class. 0.4135 is just the raw error again, not a log-loss at all. 1.0 isn't derived from the formula — it's a rounded guess that ignores the actual prediction value."
  },
  {
    "d": "Loss Calculation",
    "q": "For a 3-class softmax output with logits [2.0, 1.0, 0.1] and true class = index 0, softmax gives probs ≈ [0.6590, 0.2424, 0.0986]. What is the categorical cross-entropy loss?",
    "o": [
      "1.4170",
      "1.0986",
      "0.6590",
      "0.4170"
    ],
    "a": [
      3
    ],
    "e": "Categorical cross-entropy only looks at the true class's probability: L = −log(0.6590) ≈ 0.4170. 1.4170 is −log(0.2424) — the loss you'd get by mistakenly treating class index 1 as the true class. 1.0986 = log(3), the loss for a uniform/untrained 3-way guess, not this actual prediction. 0.6590 is just the raw probability itself, with no log or negative sign ever applied — not a loss at all."
  },
  {
    "d": "Backward Pass / Gradients",
    "q": "For MSE loss and a sigmoid output, the output error term is δ2 = −(y−a2)×a2(1−a2). Using a2=0.5865, y=1, what is δ2?",
    "o": [
      "-0.1003",
      "-0.4135",
      "0.1003",
      "-0.2425"
    ],
    "a": [
      0
    ],
    "e": "δ2 = −(1−0.5865)×(0.5865×0.4135) = −0.4135×0.2425 ≈ −0.1003. +0.1003 is what happens if you drop the leading minus sign from MSE's derivative, using (y−a2) instead of −(y−a2). −0.4135 skips multiplying by the sigmoid derivative entirely — that's just the raw output error. −0.2425 reports the sigmoid-derivative factor alone, without ever multiplying it by the output error."
  },
  {
    "d": "Backward Pass / Gradients",
    "q": "δ2 = −0.1003 and hidden activations a1=[0.6154, 0.5275]. The gradient for the output weights is dL/dW2 = δ2 × a1. What is dL/dW2 for the FIRST output weight (connecting hidden unit 1 to the output)?",
    "o": [
      "-0.0617",
      "-0.1003",
      "-0.0529",
      "0.0617"
    ],
    "a": [
      0
    ],
    "e": "dL/dW2[0] = δ2 × a1[0] = (−0.1003)×0.6154 ≈ −0.0617. −0.1003 forgets to multiply by the hidden activation at all, just reusing δ2 directly. +0.0617 flips the sign, as if δ2 had come out positive. −0.0529 is actually the gradient for the SECOND output weight (δ2×a1[1] = −0.1003×0.5275) — a mix-up between which hidden unit's activation pairs with which weight."
  },
  {
    "d": "Backward Pass / Gradients",
    "q": "Backpropagating into the hidden layer: δ1 = (W2 × δ2) elementwise-multiplied by σ'(a1). For hidden unit 1: W2[0]=0.5, δ2=−0.1003, and σ'(a1[0])=a1[0](1−a1[0])=0.6154×0.3846≈0.2367. What is δ1 for hidden unit 1?",
    "o": [
      "-0.1003",
      "-0.0501",
      "-0.0119",
      "0.0119"
    ],
    "a": [
      2
    ],
    "e": "δ1[0] = (W2[0]×δ2)×σ'(a1[0]) = (0.5×−0.1003)×0.2367 = (−0.0501)×0.2367 ≈ −0.0119. −0.0501 stops one step early — it's W2×δ2 alone, before multiplying by the hidden layer's own sigmoid derivative. −0.1003 skips backpropagating through W2 entirely, just reusing the output error term as-is. +0.0119 has the right magnitude but the wrong sign."
  },
  {
    "d": "Backward Pass / Gradients",
    "q": "δ1[0]=−0.0119 (hidden unit 1's error term) and input x=[0.5, 0.8]. The gradient for W1's first weight (hidden unit 1, from input 1) is dL/dW1[0][0] = δ1[0] × x[0]. What is it?",
    "o": [
      "-0.0119",
      "0.0059",
      "-0.0095",
      "-0.0059"
    ],
    "a": [
      3
    ],
    "e": "dL/dW1[0][0] = δ1[0]×x[0] = (−0.0119)×0.5 ≈ −0.0059. −0.0119 forgets to multiply by the input at all. −0.0095 uses x[1]=0.8 instead of x[0]=0.5 — pairing the error term with the wrong input feature. +0.0059 has the right magnitude but flips the sign."
  },
  {
    "d": "Backward Pass / Gradients",
    "q": "For this network (input → hidden(sigmoid) → output(sigmoid) → MSE loss), which chain-rule product correctly gives dL/dW1 for a hidden-layer weight?",
    "o": [
      "dL/da2 × da2/dz2 × dz2/da1 × da1/dz1 × dz1/dW1",
      "dz1/dW1 × dL/da2",
      "dL/da2 × dz2/da1 × da1/dz1",
      "da2/dz2 × dz1/dW1"
    ],
    "a": [
      0
    ],
    "e": "Backprop into an earlier layer must chain through EVERY intermediate step between the loss and that weight: loss→output-activation, output-activation→output-preactivation, output-preactivation→hidden-activation, hidden-activation→hidden-preactivation, hidden-preactivation→the weight itself. Option 2 skips da2/dz2, breaking the derivative. Option 3 jumps straight from the weight to the loss's derivative wrt the final activation, skipping every layer in between. Option 4 multiplies two unrelated partial derivatives that were never adjacent links in the actual chain."
  },
  {
    "d": "Weight Updates",
    "q": "Generic SGD update rule: w_new = w_old − lr×gradient. Given w=0.5, gradient=0.2, lr=0.1, what is w_new?",
    "o": [
      "0.52",
      "0.48",
      "-1.50",
      "0.30"
    ],
    "a": [
      1
    ],
    "e": "w_new = 0.5 − (0.1)(0.2) = 0.5 − 0.02 = 0.48. 0.52 comes from ADDING the scaled gradient instead of subtracting it — moving uphill instead of downhill. 0.30 subtracts the RAW gradient without ever scaling it by the learning rate, a step 10× too large. −1.50 treats the learning rate as if it were 1/lr, scaling the gradient by 10 instead of by 0.1."
  },
  {
    "d": "Weight Updates",
    "q": "Using the actual computed gradient dL/dW2[0] = −0.0617 and lr=0.1, with current weight W2[0]=0.5, what is the updated weight after one SGD step?",
    "o": [
      "0.5062",
      "0.5006",
      "0.4938",
      "0.5617"
    ],
    "a": [
      0
    ],
    "e": "w_new = 0.5 − (0.1)(−0.0617) = 0.5 + 0.00617 ≈ 0.5062 — subtracting a NEGATIVE gradient increases the weight. 0.4938 mishandles that sign, subtracting the gradient's magnitude as if it were positive. 0.5617 skips scaling by the learning rate, adding the raw gradient value directly (a step 10× too big). 0.5006 applies a learning rate of 0.01 instead of the actual 0.1 — a full order of magnitude too small a step."
  },
  {
    "d": "Weight Updates",
    "q": "If the learning rate is set far too high (e.g. lr=5 instead of 0.1) for this same gradient step, what is the most likely practical consequence during training?",
    "o": [
      "Weight shapes change to match the new learning rate",
      "Loss oscillates wildly or diverges (overshoots the minimum) instead of decreasing smoothly",
      "Training converges faster with no downside",
      "The gradient itself becomes zero"
    ],
    "a": [
      1
    ],
    "e": "A learning rate that's too large takes update steps far bigger than the local curvature of the loss supports, so each step can overshoot the minimum and land on an even worse point — the classic 'diverging loss' failure mode, not faster convergence. The gradient's VALUE doesn't depend on the learning rate at all (they're only multiplied together at the update step, so the gradient computation itself is unaffected). And the learning rate is a scalar multiplying every element of a gradient tensor — it changes magnitude, never a tensor's shape."
  },
  {
    "d": "Epochs, Iterations & Batches",
    "q": "Training set has 2000 samples, batch size = 40. How many iterations (weight-update steps) make up ONE epoch?",
    "o": [
      "2040",
      "2000",
      "40",
      "50"
    ],
    "a": [
      3
    ],
    "e": "iterations per epoch = dataset_size / batch_size = 2000/40 = 50. Answering 2000 confuses the TOTAL sample count with the number of update steps — you take one gradient step per BATCH, not per sample. Answering 40 reports the batch size itself, not how many batches fit in the dataset. Answering 2040 comes from adding the two numbers instead of dividing them."
  },
  {
    "d": "Epochs, Iterations & Batches",
    "q": "Same setup (2000 samples, batch size 40, so 50 iterations/epoch). Training runs for 15 epochs. How many total weight-update steps happen across the whole run?",
    "o": [
      "15",
      "600",
      "750",
      "50"
    ],
    "a": [
      2
    ],
    "e": "Total iterations = iterations-per-epoch × epochs = 50×15 = 750. 600 comes from multiplying batch size by epoch count instead (40×15) — the wrong two numbers entirely. 50 stops after computing iterations for a SINGLE epoch and forgets to multiply by how many epochs actually run. 15 just restates the epoch count, treating 'epoch' and 'iteration' as the same unit when an epoch is a full pass over the data made of many iterations."
  },
  {
    "d": "Shapes",
    "q": "A 32×32 input image goes through a Conv2d layer with kernel size 5, stride 1, no padding. Using out = floor((in − k + 2p)/s) + 1, what is the output spatial size?",
    "o": [
      "30",
      "32",
      "28",
      "27"
    ],
    "a": [
      2
    ],
    "e": "out = (32−5+0)/1 + 1 = 27+1 = 28. Answering 32 assumes 'same' padding, but this layer explicitly uses padding=0 (valid convolution), which always shrinks the spatial size when kernel>1. Answering 27 computes (in−k)/s correctly but forgets the trailing +1 the formula requires. Answering 30 is what you'd get by mistakenly plugging in kernel size 3 instead of the actual 5."
  },
  {
    "d": "Shapes",
    "q": "Same formula. For a 32×32 input, kernel 3, stride 1, padding 1 ('same' padding), what is the output size?",
    "o": [
      "16",
      "34",
      "32",
      "30"
    ],
    "a": [
      2
    ],
    "e": "out = (32−3+2×1)/1 + 1 = 31+1 = 32 — this is exactly why padding=1 with a 3×3 kernel is called 'same' padding, it preserves spatial size. 30 is what you'd get with padding=0 instead of 1. 34 comes from treating p=1 as if it contributed 2p=4 (as though p were 2). 16 confuses padding with STRIDE — that's the output from halving via stride=2, a completely different mechanism."
  },
  {
    "d": "Shapes",
    "q": "A feature map coming out of the conv/pool stack has shape (channels=16, height=7, width=7) for each sample. Before feeding it into a fully-connected layer, you flatten it. What is the flattened per-sample vector length?",
    "o": [
      "112",
      "784",
      "30",
      "49"
    ],
    "a": [
      1
    ],
    "e": "Flattening multiplies every dimension together: 16×7×7 = 784. 49 only accounts for the spatial dimensions (7×7), leaving out the channel dimension entirely — as if there were just one channel. 112 multiplies channels by height (16×7) but drops the width dimension. 30 adds the three numbers together instead of multiplying them, which isn't how flattening a tensor works at all."
  }
]
</script>
<div class="topic-quiz-mount"></div>
