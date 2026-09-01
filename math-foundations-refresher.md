# Math Foundations Refresher

Every other doc in this hub uses math without explaining it. A vector here. A dot product there. "The probability of..." somewhere else. This doc is where all of that gets explained simply, one small piece at a time.

No fancy notation. Every formula is written out in plain symbols you can read at a glance.

---

## Part 1 — Linear Algebra

This is the math neural networks are built from.

### 1. What is a vector?

A vector is just a list of numbers. That's it.

`[3, 4]` is a vector. Think of it as an arrow: start at zero, go 3 steps right, then 4 steps up.

**How long is that arrow?** Use this formula:
```
length = sqrt(3² + 4²) = sqrt(9 + 16) = sqrt(25) = 5
```

Here's a picture:
```
  y
  4 |        • (3,4)
    |      ⟋
    |    ⟋   length = √(3²+4²) = 5
    |  ⟋
  0 +──────────── x
    0        3
```

In machine learning, a vector usually means one of two things:
- **A row of features.** Example: `[age=34, income=52000]`.
- **An embedding.** A list of numbers (often 768 or more) that represents the *meaning* of a word or sentence, not a physical direction. The "arrow" idea still works — it's just an arrow in a space with way more than 2 dimensions, so you can't draw it on paper.

### 2. What is a matrix?

A vector is one list of numbers. A **matrix** is a grid of numbers — rows and columns.

A matrix with 2 rows and 3 columns is called a `[2×3]` matrix.

**Multiplying two matrices** only works if the middle numbers match. `[2×3] @ [3×4]` works, because the 3s match. The result is `[2×4]` — the middle numbers cancel out, and you're left with the outer shape.

Here's what actually happens, step by step, with small numbers:
```
A = [[1, 2],       B = [[5, 6],
     [3, 4]]            [7, 8]]

A @ B = [[1*5+2*7, 1*6+2*8],   = [[19, 22],
         [3*5+4*7, 3*6+4*8]]      [43, 50]]
```

Each number in the answer comes from multiplying matching positions in a row and a column, then adding them up. That's called a **dot product** — more on that in question 3.

**Why this matters:** every `nn.Linear` layer in a neural network is exactly this. The formula is `output = input @ weight_matrix + bias`. When you see "project Q, K, V" in a transformer, that's just this same multiply, done at a bigger scale.

### 3. What is a dot product?

You already saw one above. Here it is by itself:
```
[1, 2, 3] · [4, 5, 6] = 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
```

Multiply matching positions. Add them up. That's a dot product.

**What does the number mean?** A dot product measures how similar two vectors are.
- A large number means the two vectors point in a similar direction, and are both large.
- A number near zero means they're unrelated (perpendicular).
- A negative number means they point in opposite directions.

**Why this shows up in attention and embeddings:**
- Attention scores are computed as `Q · K` — literally asking "how much does this query match this key?"
- Comparing two embeddings for similarity is a dot product too. (Cosine similarity is the same thing, just after scaling both vectors down to length 1 first.)

### 4. Why do attention formulas write `Kᵀ`?

The little `ᵀ` means **transpose** — flip rows into columns.
```
[[1,2,3]]ᵀ = [[1],[2],[3]]
```

**Why bother?** Because matrix multiplication needs shapes to match, and without the flip, they don't.

In attention, `Q` and `K` both start out shaped `[seq_len × d_k]`. You can't multiply two matrices with the same shape like that directly. So you flip `K` to `[d_k × seq_len]`. Now the multiply works:
```
[seq_len × d_k] @ [d_k × seq_len] = [seq_len × seq_len]
```

That result is a full grid: every token's score against every other token. Exactly what attention needs.

### 5. What is an eigenvector, and why does PCA care?

Most of the time, when a matrix multiplies a vector, it rotates the vector *and* changes its length.

But for some special vectors, the matrix only stretches them — it doesn't rotate them at all. Those are called **eigenvectors**.
```
A @ v = λ * v
```
Here, `v` is the eigenvector, and `λ` (a single number) is how much it gets stretched.

**Why PCA cares:** PCA looks at your dataset's covariance matrix and finds its eigenvectors. The eigenvector with the biggest stretch factor (`λ`) points in the direction your data varies the most. That direction is called "principal component 1."

You don't need to compute any of this by hand — `sklearn.decomposition.PCA` does it for you. But you should be able to say, in one sentence: *"PCA finds the directions of maximum variance by computing eigenvectors of the covariance matrix."*

### 6. L1 vs. L2 — two ways to measure a vector's size

Question 1 measured a vector's length the normal way — straight-line distance. That's called the **L2 norm**:
```
L2 norm = sqrt(sum of x²)
```

There's another way to measure size, called the **L1 norm**. Instead of a straight line, imagine walking a city grid (like a taxicab, only moving along streets, never diagonally):
```
L1 norm = sum of |x|
```

**Why this matters for regularization:**
- **L1 regularization (lasso)** uses the L1 norm as a penalty. Every unit of weight costs the same amount, no matter how big or small it already is. So it's "cheapest" to push unimportant weights all the way down to exactly zero — which is exactly what L1 does.
- **L2 regularization (ridge)** uses the L2 norm. Because it squares the weights, big weights get penalized a lot, but a weight that's already close to zero barely gets penalized at all. So L2 shrinks weights toward zero, but rarely makes them exactly zero.

### Summary example

Here's how all six pieces above fit into one real computation: a transformer scoring one query token against every other token.

1. The query is a vector (question 1) — 64 numbers instead of 2, but same idea.
2. It gets compared to every key vector using a dot product (question 3).
3. That comparison is done for every key at once using matrix multiplication: `Q @ Kᵀ` (questions 2 and 4).
4. The transpose in that formula is what makes the shapes line up.

The whole attention score grid is just the small 2×2 example from question 2, scaled up, using the similarity idea from question 3, made possible by the transpose from question 4.

---

## Part 2 — Probability & Statistics Foundations

### 1. Probability vs. likelihood — what's the difference?

These two words get mixed up constantly. Here's the actual difference:

- **Probability** asks: *"Given a fixed model, how likely is this data?"*
- **Likelihood** asks the reverse: *"Given this fixed data, how well does this model explain it?"*

Same math, different thing held constant.

This matters because most ML models are fit using **maximum likelihood estimation** — searching for the model parameters that make the data you actually observed look as probable as possible.

### 2. Bayes' theorem — flipping a conditional probability

`P(A|B)` means "the probability of A, given that B already happened."

Sometimes you know `P(B|A)` but you actually want `P(A|B)` — the reverse. Bayes' theorem is the formula that flips it:
```
P(A|B) = P(B|A) * P(A) / P(B)
```

**A worked example, step by step.**

A disease test is 99% accurate. The disease affects 1% of people. Someone tests positive. What's the actual chance they have the disease?

Step 1 — write down what you know:
```
P(disease) = 0.01
P(positive | disease) = 0.99         (true positive rate)
P(positive | no disease) = 0.01      (false positive rate)
```

Step 2 — find the overall chance of testing positive, from any cause:
```
P(positive) = 0.99 × 0.01 + 0.01 × 0.99 = 0.0099 + 0.0099 = 0.0198
```

Step 3 — plug into Bayes' theorem:
```
P(disease | positive) = 0.99 × 0.01 / 0.0198 = 0.5
```

**The answer is 50%. Not 99%.**

Why? The disease is rare. So even though the test is accurate, the huge number of healthy people still produces almost as many false positives (0.0099) as the tiny number of sick people produces true positives (0.0099). This is the single most common trap in applied statistics — and it's exactly why "the model is 99% accurate" means nothing on its own. You also need to know how rare the thing you're detecting is.

### 3. The chain rule of probability — what happens with three or more variables?

Bayes' theorem above handles two variables. What if you have three, chained together?

There's a different rule for that, called the **chain rule of probability**. (Not the same thing as the calculus chain rule later in this doc — they just happen to share a name.)

It breaks one hard joint probability into a chain of easier, smaller questions:
```
P(A, B, C) = P(A) · P(B|A) · P(C|A, B)
```

This is always exactly true. It's not a shortcut or an approximation.

**A worked example.** Draw 3 cards from a deck, without putting them back. What's the chance all three are hearts?

Break it into three easier questions:
```
P(1st card is a heart) = 13/52
P(2nd card is a heart, given the 1st was) = 12/51    (one fewer heart, one fewer card left)
P(3rd card is a heart, given the first two were) = 11/50
```

Multiply the three answers together:
```
P(all 3 hearts) = 13/52 × 12/51 × 11/50 ≈ 0.0129
```

Each step only has to account for what already happened. You never have to solve the whole problem at once. That's the entire point of the chain rule.

**Why this matters for AI.** A language model writes a sentence one word at a time. To do that, it needs the probability of the whole sentence. That's the exact same problem as the cards, just with words instead of hearts:
```
P(x₁, x₂, ..., xₙ) = P(x₁) · P(x₂|x₁) · P(x₃|x₁,x₂) · ... · P(xₙ|x₁,...,xₙ₋₁)
```

Each piece on the right is the model's softmax output at one position — its guess for the next word, given every word so far (see question 4 below for what softmax actually outputs). Multiply all those guesses together, and you get the model's confidence in the whole sentence.

This is why these models are called **autoregressive**. They guess one word, then use that guess to help guess the next one. Same rule as the cards — just applied to language.

**One line to remember this by:** the chain rule lets you turn "the probability of this whole complicated thing" into "the probability of each small piece, one at a time." Cards, sentences, anything in sequence — same rule underneath.

### 4. Distributions — the shapes a probability can take

Bayes' theorem needs a starting probability, `P(A)`. What shape can that number actually take?

A **distribution** describes how likely each possible value is. There are two families:

- **Discrete** — a fixed list of possible outcomes, each with its own probability.
- **Continuous** — any real number in a range, described by a curve instead of a list.

**Discrete distributions:**

- **Bernoulli(p)** — one yes/no coin flip, with probability `p` of "yes." This is the building block behind binary classification. It's also exactly how a neural net's **dropout** works — each neuron independently survives with probability `p`.
- **Binomial(n, p)** — how many "yes" results you got, out of `n` independent Bernoulli(p) flips. If a whole layer makes `n` independent dropout decisions, the number of surviving neurons is Binomial(n, p).
- **Categorical(p₁...pₖ)** — like Bernoulli, but with more than 2 possible outcomes, each with its own probability. **This is exactly what a softmax layer outputs.** Predicting the next word means sampling from a categorical distribution over the whole vocabulary.
- **Poisson(λ)** — how many independent events happen in a fixed window of time, given an average rate `λ`. Example: how many support tickets come in per hour.

**Continuous distributions:**

- **Uniform(a, b)** — every value between `a` and `b` is equally likely. Used, for example, as a common weight-initialization range — before training starts, there's "no information yet," so every starting value is equally plausible.
- **Normal / Gaussian (μ, σ²)** — the familiar bell curve. Most natural measurements cluster around a mean `μ`, with a spread `σ`. (Question 7 below explains *why* this shape shows up so often — it's called the Central Limit Theorem.) This distribution is used for weight initialization (Xavier/He) and for the noise added at each step of a diffusion model.

Here's what a few of these actually look like, with real numbers:
```
Bernoulli(p=0.3)          Binomial(n=10,p=0.5)              Poisson(λ=3)
P(0)=0.70  P(1)=0.30       peak at k=5: 0.246                peak at k=2,3: 0.224
  ▇▇▇▇▇▇▇   ▇▇▇             ▁▁▂▄▆█▆▄▂▁▁  (k=0 to k=10)        ▂▅██▆▄▂▁  (k=0 to k=7)
```

One more useful fact: when a model does **greedy decoding**, it just always picks the single most likely outcome from its categorical distribution (called the "mode"). When it does temperature or top-k or top-p sampling instead, it draws a random sample from that same distribution, just reshaped first.

### 5. Mean, variance, and standard deviation — computing them by hand

Take this data: `[2, 4, 4, 4, 5, 5, 7, 9]`.

**Step 1 — the mean.** Add everything up, divide by the count:
```
mean = 40 / 8 = 5
```

**Step 2 — the variance.** For each number, find how far it is from the mean, square that distance, then average all the squared distances:
```
variance = (9+1+1+1+0+0+4+16) / 8 = 32 / 8 = 4
```

**Step 3 — the standard deviation.** Just the square root of the variance:
```
std = sqrt(4) = 2
```

**Why report std instead of variance?** Std is measured in the same units as your original data. Variance isn't — it's squared, so the units get squared too. That's why std, not variance, is the number usually reported and shown as error bars on a chart.

### 6. What is a p-value, and what does it NOT mean?

Say you're comparing two groups, and you want to know: is the difference between them *real*, or just random noise?

A **p-value** answers this: *"If there were actually no real difference at all, how likely would I be to see a gap this big, just by chance?"*

Example: `p = 0.03` means: "if nothing real were going on, you'd still see a gap this big about 3% of the time, purely by chance."

**Here is what it does NOT mean**, and this trips people up constantly:
- It does **not** mean "there's a 97% chance the effect is real."
- It does **not** mean "there's a 3% chance the null hypothesis is true."

A p-value is a statement about how surprising your *data* is, assuming nothing real is happening. It is not a probability attached to whether your hypothesis is true. This exact mix-up shows up in interviews all the time.

### 7. Why does p-value math work even when your data isn't a bell curve?

The math behind p-values assumes your data's average follows a normal (bell-curve) shape. But real data is often messy and skewed. So why does it still work?

The answer is the **Central Limit Theorem**: no matter what shape your original data is in, the *average* of many independent samples tends to look like a normal distribution, as your sample size grows.

This is why A/B tests can safely use normal-distribution statistics (like a t-test) on things like conversion rate — even though a single user's "did they convert" (0 or 1) isn't remotely bell-shaped on its own. Average enough of them together, and the average itself becomes approximately normal anyway.

### 8. Correlation vs. causation — does one cause the other?

Two variables moving together (correlated) does **not** mean one causes the other.

Classic example: ice cream sales and drowning deaths rise and fall together. Neither causes the other. Both are actually caused by a third thing — hot weather.

This gap between "they move together" and "one causes the other" is a big enough problem that it gets its own topic in this hub: `service-impact-and-causal-inference.md`. Correlation is easy to compute. Proving causation needs either a real randomized experiment, or a causal-inference method that approximates one.

### 9. Bayesian statistics — treating belief as something that updates

Everything above (p-values, the disease-test example) gives you one fixed number, computed once. Is there a different way to think about probability?

Yes. **Bayesian statistics** treats your belief about something unknown as a full *distribution* — not one fixed answer — and updates that distribution every time new evidence shows up.

**A worked example — figuring out if a coin is biased, step by step.**

Start with a belief before seeing any data at all:
```
prior: Beta(1, 1)   (flat — every possible bias, 0% to 100%, is equally plausible)
```

Flip the coin 10 times. You get 7 heads, 3 tails. Update your belief:
```
posterior = Beta(1+7, 1+3) = Beta(8, 4)
posterior mean = 8 / (8+4) = 0.667
```
You now lean toward "biased toward heads," but you're still pretty uncertain.

Flip it 90 more times. Combined with the first 10, that's 62 heads and 38 tails total. Update again:
```
posterior = Beta(1+62, 1+38) = Beta(63, 39)
posterior mean = 63 / (63+39) = 0.618
```
Your estimate is now narrower and more confident.

Here's what that looks like — your belief physically narrowing as evidence piles up:
```
Prior (flat)         After 10 flips           After 100 flips
▁▁▁▁▁▁▁▁▁▁▁▁▁         ▁▂▄▆█▆▄▂▁                    ▁▂▅███▅▂▁
0    0.5    1        0   0.67    1               0  0.62   1
```

**A simple way to remember this:** yesterday's posterior becomes today's prior. Each round's updated belief is the starting point for the next round. (One technical reason this example uses a Beta distribution specifically: it's the "conjugate prior" for yes/no data, which just means the update has this clean formula — `Beta(a,b)` becomes `Beta(a+successes, b+failures)` — instead of needing to be solved with harder math.)

### 10. Frequentist vs. Bayesian — what's the actual practical difference?

| | Frequentist (questions 1–8 above) | Bayesian (question 9) |
|---|---|---|
| What "probability" means | How often something happens, over many repeats | How strongly you believe something, which can change with evidence |
| What you get as an answer | A p-value, or a single confidence interval | A full distribution of belief |
| Do you use prior knowledge? | No, not formally | Yes — and you have to justify it |
| Can you say "95% chance the true value is in this range"? | **No** — that's a common misreading (same trap as question 6's p-value one) | **Yes** — this is exactly what a Bayesian credible interval means |

### Summary example

An A/B test gives a frequentist p-value of 0.04. That means: "reject the null hypothesis" — a real, useful conclusion. But it does **not** mean "96% chance the new version is better." That's not what a p-value says.

Now run the same comparison the Bayesian way instead (question 9's approach). You'd get a full posterior distribution over "how much better is the new version." From that, a statement like *"there's an 80% chance the new version improves conversion by at least 2%"* becomes something you can actually, correctly say. For a lot of real business questions, that's a more useful answer than the frequentist one.

---

## Part 3 — Calculus

Every time you read "the model updates its weights" anywhere in this hub, calculus is what's actually happening underneath.

### 1. What is a derivative, actually?

A derivative is the **slope of a curve at one exact point** — how fast `y` changes for a tiny nudge in `x`, right at that spot.
```
f'(x) = lim(h→0) [ f(x+h) − f(x) ] / h
```

**A worked example, step by step.** Take `f(x) = x²`. What's the slope at `x = 3`? Try shrinking step sizes (`h`) and watch what happens:
```
h = 1:      (4² − 3²) / 1     = (16 − 9)  / 1     = 7
h = 0.1:    (3.1² − 3²) / 0.1 = (9.61 − 9) / 0.1   = 6.1
h = 0.01:   (3.01² − 3²)/0.01 = (9.0601−9) / 0.01  = 6.01
h = 0.001:  ...                                     = 6.001
```

As `h` gets smaller, the answer keeps getting closer to exactly **6**. That matches the shortcut formula for this kind of problem (`d/dx x² = 2x`, so `2×3=6`) — but notice we found it just by shrinking `h`, without needing to already know the shortcut. That shrinking process — not the shortcut formula — is what a derivative actually *is*.

Here's a picture of what's happening: a straight line just touching the curve at one point.
```
 y
16 |                                    * <- actual curve, f(4) = 16
15 |                               x      <- tangent line's prediction at x=4: f(3) + slope*1 = 9 + 6 = 15
   |                          ⟋
 9 |                     *              <- f(3) = 9, tangent touches here, slope = 6
   |                ⟋
   +------+------+------+------+----
          2      3      4      5    x
```

Notice the tangent line predicts `15` at `x=4`, but the real curve is at `16`. Close, but not exact — because a straight line is only a good approximation *near* the point you drew it from. That's exactly what the shrinking-`h` table above was showing you.

### 2. What other derivative rules come up constantly?

| Rule | Formula | Where it shows up |
|---|---|---|
| Power rule | `d/dx xⁿ = n·xⁿ⁻¹` | almost everywhere |
| Constant | `d/dx c = 0` | a bias term contributes 0 to weight gradients |
| Sum rule | `d/dx [f+g] = f' + g'` | total loss = sum of per-example losses, so its gradient = sum of per-example gradients |
| Exponential | `d/dx eˣ = eˣ` | `eˣ` is the *only* function that's its own derivative |
| Natural log | `d/dx ln(x) = 1/x` | why cross-entropy's gradient has a clean `1/p` term |
| Chain rule | `d/dx f(g(x)) = f'(g(x)) · g'(x)` | this one **is** backprop — see next question |

### 3. Is "backprop" really the same thing as the calculus chain rule?

Yes. Not similar to it. The literal same operation.

**A worked example.** Take `h(x) = (3x+1)²` at `x = 2`. Treat it as one function wrapped inside another: an outer function `f(u) = u²`, wrapped around an inner function `g(x) = 3x+1`.

Step by step:
```
g(2) = 3(2)+1 = 7
f'(u) = 2u  ->  f'(g(2)) = 2×7 = 14
g'(x) = 3
h'(2) = f'(g(2)) × g'(x) = 14 × 3 = 42
```

Check this a different way, by expanding `h(x)` directly: `h(x) = 9x²+6x+1`, so `h'(x) = 18x+6`. At `x=2`: `36+6 = 42`. Same answer.

That multiplication — the outer function's derivative times the inner function's derivative — is exactly what backpropagation computes. It does this one layer at a time, working backward from the loss to the very first layer's weights.

This is also exactly why the **vanishing gradient problem** happens. If you multiply together 20 numbers that are each smaller than 1, the result shrinks toward zero: `0.25²⁰ ≈ 0.00000024`. That's the same 2-step multiplication above, just repeated 20 times instead of 2.

For this same idea worked through a full toy neural network — real numbers, every layer, the full forward pass and backward pass — see the **Neural Net Numericals** topic.

### 4. A real model has millions of weights, not just one `x`. How does this generalize?

A **partial derivative** is a derivative taken with respect to just one variable, while treating every other variable as fixed for the moment.

The **gradient** is the list of *all* those partial derivatives, one per weight, stacked together into a vector. That vector points in the direction where the loss increases the fastest.

That's exactly why gradient descent moves in the *opposite* direction of the gradient — you want to go where the loss gets smaller, not bigger.

### 5. Why does `eˣ` show up everywhere in ML?

Here's what the curve `eˣ` actually looks like:
```
x        eˣ
-2       0.14
-1       0.37
 0       1.00
 1       2.72
 2       7.39
 3      20.09
```

For negative `x`, it flattens out toward zero, but never quite reaches it. At `x=0` it's exactly 1. For positive `x`, it grows explosively.

**Why this matters for turning raw numbers into probabilities:** a model's raw output (called a logit) can be any number, including negative ones. But a probability can never be negative. `eˣ` is guaranteed to be positive, no matter how negative `x` is — which makes it the perfect building block for softmax:
```
softmax(x)ᵢ = e^xᵢ / Σⱼ e^xⱼ
```

Combined with the fact that `eˣ` is its own derivative (question 2), this makes it both mathematically convenient and exactly the right shape for the job.

### 6. Why does ML always pair `eˣ` with its opposite, `ln(x)`?

Here's what `ln(x)` (natural log) looks like:
```
x        ln(x)
0.1      -2.30
0.5      -0.69
1         0.00
2         0.69
5         1.61
10        2.30
```

As `x` approaches 0, this plunges steeply toward negative infinity. At `x=1`, it crosses exactly 0. After that, it keeps climbing, but more and more slowly. `ln(x)` isn't defined at all for `x` at or below 0.

`eˣ` and `ln(x)` are exact opposites of each other (`ln(eˣ) = x`). That's why you'll see a "log-sum-exp" step in softmax and cross-entropy code — it's how you move safely between the two.

**Two concrete reasons logs matter, beyond just being an inverse:**

1. **They turn multiplication into addition.** `log(a×b) = log(a) + log(b)`. A whole dataset's likelihood is a *product* of many small per-example probabilities — multiply enough small numbers together and the result underflows to 0 on real computer hardware. Taking the log turns that fragile product into a stable sum instead.
2. **They punish confident wrongness much more than confident correctness.** This is exactly cross-entropy's `−log(p)` shape. Predicting the true class with 78.56% confidence costs `0.24` in loss. Predicting it with only 3.91% confidence costs `3.24` — over 13 times worse. That's the steep plunge of the log curve near zero, doing its job.

### Summary example

Here's how every piece of Part 3 fits into one real forward-and-backward pass:

1. A transformer's forward pass ends by computing a loss, using cross-entropy's `−log(p)` (question 6).
2. That `p` came from softmax's `eˣ/Σeˣ` (question 5), applied to the model's raw output numbers.
3. Backprop (question 3) then computes the gradient of that loss with respect to *every single weight*, one layer at a time, using the chain rule.
4. The partial-derivative and gradient machinery (question 4) is what lets this work across millions of weights at once, not just one.

And if the network is 20 layers deep, with small gradients at each layer, that same repeated chain-rule multiplication is exactly what shrinks the earliest layer's gradient down toward that vanishing `0.25²⁰` number from question 3. It's not a separate problem — it's the same multiplication, just repeated more times.

---

## Practice Q&A (Self-Test)

### Why does `Q @ Kᵀ` need the transpose? What would happen without it?

`Q` is shaped `[seq_len × d_k]`. `K` is also shaped `[seq_len × d_k]` — the same shape. Multiplying two matrices requires the inner numbers to match, and here they don't (`d_k` doesn't match `seq_len`). So `Q @ K` as written is invalid.

Flipping `K` to `[d_k × seq_len]` fixes this. Now the multiply works: `[seq_len × d_k] @ [d_k × seq_len] = [seq_len × seq_len]`. That gives you exactly the grid attention needs — every token scored against every other token.

### A test is 99.9% accurate. The condition it detects affects 1 in 10,000 people. Is a positive result more likely to be real, or a false alarm?

A false alarm, by a wide margin. This is the same Bayes' theorem trap from the worked disease example above. Because the condition is so rare, even a tiny false-positive rate applied to the huge healthy population produces more false alarms than true detections applied to the tiny group who actually have it. This is exactly why rare-event detectors — fraud, rare diseases, rare mechanical failures — need a very low false-positive rate specifically, not just a high overall accuracy number.

### Why does L1 regularization zero out weights, while L2 only shrinks them?

L1's penalty grows in a straight line (`sum of |w|`). So shrinking any weight — big or small — saves the same amount, which makes it worth pushing unimportant weights all the way down to exactly zero.

L2's penalty grows as a square (`sum of w²`). It punishes large weights hard, but once a weight is already small, the "savings" from shrinking it further are tiny. So L2 smooths weights toward zero, but almost never reaches exactly zero.

### You get p = 0.04 on an A/B test. Is it correct to say "there's a 96% chance the new version is better"?

No. `p = 0.04` means: if the new version actually had no real effect at all, you'd still see a difference this big 4% of the time, just by chance. It says nothing directly about the probability that the new version is truly better. Getting that second statement would require a Bayesian approach with a prior belief, not a plain p-value.

### Why can A/B tests use normal-distribution statistics, even when the thing being measured (did this user convert — 0 or 1) isn't normal at all?

The Central Limit Theorem. The *distribution of the sample mean* — the average conversion rate across thousands of users — approaches a normal shape as your sample size grows, even though each individual data point is a lopsided 0-or-1 outcome. The test only needs the averaged statistic to be roughly normal. It doesn't need the individual data points to be.

### You update a Beta(1,1) prior after 10 coin flips. Then you flip 90 more times. Should you throw out the first update and start over with all 100 flips, or keep updating from where you left off?

Either way gives you the exact same final answer. Updating `Beta(1,1)` once with all 100 flips (62 heads, 38 tails) gives `Beta(63, 39)`. Updating first with 10 flips (getting `Beta(8,4)`), then updating that result with the next 90 flips, also gives `Beta(63, 39)`. Same number either way. This is exactly what "yesterday's posterior is today's prior" means — it doesn't matter whether evidence arrives all at once or in small batches over time.

### Why is it wrong to say "95% chance the true value is in this range" about a normal confidence interval, but correct for a Bayesian credible interval?

A frequentist confidence interval comes from a *procedure*. If you repeated that procedure many times, 95% of the resulting intervals would contain the true value. The 95% describes how reliable the procedure is across many repeats — it's not a probability about this one specific interval you're looking at right now.

A Bayesian credible interval is different. It directly describes a range of your belief distribution. So saying "there's a 95% chance the true value is in this range" is a completely accurate description of what it means.

### Why does shrinking `h` toward 0 give you the exact derivative, instead of just an approximation forever?

Each smaller `h` measures the curve over a shorter, straighter-looking stretch, so it gets a little more accurate each time. As `h` shrinks all the way to 0, that stretch shrinks down to a single point, and the error shrinks down to exactly 0. The limit isn't an estimate of the derivative — it *is* the derivative. The worked table earlier (`h=1 → 7`, `h=0.1 → 6.1`, `h=0.01 → 6.01`, converging to 6) is that shrinking, shown with real numbers.

### Why is the chain rule specifically the operation behind backpropagation, not just something "related to" it?

Backprop needs to know how the loss changes when an early layer's weights change. But the loss doesn't touch those weights directly — it only reaches them *through* every layer in between. That's exactly the "function inside a function" shape the chain rule is built for: the outer function's derivative, times the inner function's derivative, layer after layer, multiplying all the way back to the start. It's not similar to backprop. Backprop is the chain rule, applied repeatedly. That's also exactly why 20 layers of small gradients multiply down into a vanishing gradient, instead of just adding up.

### Cross-entropy loss uses `−log(p)`. Why the negative sign, and why use log at all instead of something simpler like `1 − p`?

`log(p)` comes out negative for any valid probability (since `p` is at most 1). Adding the negative sign flips it positive — which matches what a loss is supposed to be: a positive number that gets bigger the worse the model does.

Using `log` specifically (instead of a simple `1−p`) creates a steep penalty near zero. Predicting the true class with only 3.91% confidence should be punished far more harshly than proportionally worse than 78.56% confidence — and the log curve's plunge toward negative infinity near zero delivers exactly that kind of severe, non-linear penalty.

### Softmax uses `eˣ` to turn logits into probabilities. Why not just divide each raw logit by their total instead?

Raw logits can be negative, and they can add up to something close to zero. Dividing directly by that sum can produce negative "probabilities," or wildly unstable numbers — not a valid probability distribution at all.

`eˣ` is guaranteed to be positive, for any input, no matter how negative. So exponentiating first guarantees every number going into the final division is a valid positive number. Only after that does dividing by the total actually produce numbers that are positive and add up to 1 — which is what a real probability distribution requires.
