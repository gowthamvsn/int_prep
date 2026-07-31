# Math Foundations Refresher — Built as a Chain, Not a List

Every other doc on this hub uses this math without re-explaining it — a vector here, a dot product there, "the probability of..." somewhere else, "backprop" as if the chain rule needs no introduction. Each Part below is one continuous thread — every question inherits the answer before it, closing with a worked summary example. No raw LaTeX anywhere — every formula is plain, readable symbols in a code block.

## Part 1 — Linear Algebra (the language neural nets are written in)

### 1. What is a vector, really?
A list of numbers with a direction and length. `[3, 4]` is a vector — in 2D, "go 3 right, 4 up." Its **length (magnitude)** is `sqrt(3² + 4²) = sqrt(25) = 5`.

**Visual + memory hook — an arrow, not just a list of numbers:**
```
  y
  4 |        • (3,4)
    |      ⟋
    |    ⟋   length = √(3²+4²) = 5
    |  ⟋
  0 +──────────── x
    0        3
```
In ML, a vector is usually a row of features (`[age=34, income=52000, ...]`) or an embedding (`[0.12, -0.44, 0.03, ...]` — 768 or more numbers representing meaning, not physical direction — the arrow picture still holds, just in a space with far more than 2 dimensions to draw).

### 2. Given a vector as one list of numbers, what is a MATRIX, and what does multiplying two of them actually compute?
A matrix is a grid of numbers — rows × columns. A `[2×3]` matrix has 2 rows and 3 columns. Matrix multiplication `A @ B` only works if `A`'s columns match `B`'s rows (`[2×3] @ [3×4] = [2×4]`) — the inner numbers must agree, and they "cancel out," leaving the outer shape. Concretely, each output cell is a **dot product** of a row from `A` and a column from `B`:
```
A = [[1, 2],       B = [[5, 6],
     [3, 4]]            [7, 8]]

A @ B = [[1*5+2*7, 1*6+2*8],   = [[19, 22],
         [3*5+4*7, 3*6+4*8]]      [43, 50]]
```
Every `nn.Linear` layer in a neural net is exactly this: `output = input @ weight_matrix + bias`. When the transformer docs say "project Q, K, V," that's a `[seq_len × d_model] @ [d_model × d_k]` matrix multiply, nothing more exotic than the 2×2 example above at a bigger scale.

### 3. Each output cell above was called a "dot product" — what IS that, and why does it show up in attention and cosine similarity specifically?
`[1, 2, 3] · [4, 5, 6] = 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32`. Multiply matching positions, add them up. A dot product is a **similarity measure**: large when two vectors point in a similar direction with large magnitude, near zero when they're perpendicular (unrelated), negative when they point opposite directions. That's exactly why attention scores are `Q · K` (how much does this query "match" this key) and why embedding similarity is a dot product (cosine similarity is just a dot product after both vectors are scaled to length 1).

### 4. Attention formulas keep writing `Kᵀ` — given that matrix multiply needs shapes to agree, why the transpose specifically?
Flip rows into columns: `[[1,2,3]]ᵀ = [[1],[2],[3]]`. It shows up constantly because matrix multiplication needs shapes to agree — `Q @ Kᵀ` in attention turns `K` sideways so its rows (one per token) become columns, making `[seq×d_k] @ [d_k×seq] = [seq×seq]` (a full "every token vs. every token" score grid) actually valid.

### 5. Given that a matrix transforms vectors, are there directions it DOESN'T rotate, only stretches — and why does PCA care?
For a matrix `A`, an eigenvector `v` is a special direction `A` doesn't rotate, only stretches: `A @ v = λ * v`, where `λ` (the eigenvalue) is how much it stretches. PCA computes the eigenvectors of a dataset's covariance matrix; the eigenvector with the largest eigenvalue is the direction the data varies the most along — "principal component 1." You don't need to compute these by hand (`sklearn.decomposition.PCA` does it), but "PCA finds the directions of maximum variance via eigenvectors of the covariance matrix" is the one-sentence version worth having ready.

### 6. Given a vector's magnitude (question 1) measures its size the "straight-line" way — is there another way to measure size, and why would it matter for regularization?
**L2 norm** (Euclidean, the same magnitude from question 1) is `sqrt(sum(x²))` — straight-line distance. **L1 norm** is `sum(|x|)` — a taxicab/grid distance. This exact distinction is why L1 regularization (lasso) zeroes out weights entirely (treats every unit of weight as equally costly, so it's cheapest to drop unimportant ones to exactly 0) while L2 regularization (ridge) only shrinks weights toward zero without eliminating them (squaring makes large weights disproportionately expensive to keep, but never makes zero "free" the way L1 does).

### Summary example
A transformer computing attention for one query token: the query vector `q` (question 1's "arrow," now 64-dimensional) gets dot-producted (question 3) against every key vector via `Q @ Kᵀ` (questions 2 and 4) to produce a score per token — the entire attention score matrix is nothing more than the 2×2 toy multiply from question 2, scaled up, using the similarity logic from question 3, made shape-valid by the transpose from question 4.

## Part 2 — Probability & Statistics Foundations

### 1. What's the difference between probability and likelihood — the two words everything below builds on?
**Probability** asks: given a fixed model, how likely is this data? **Likelihood** asks: given this fixed data, how well does this model explain it? Same formula, different things held constant — "maximum likelihood estimation" (how most ML models are actually fit) means searching over model parameters to find the ones that make the observed data most probable.

### 2. Given a conditional probability, how do you FLIP it around (Bayes' theorem), with real numbers?
`P(A|B)` = probability of A given B already happened.
```
P(A|B) = P(B|A) * P(A) / P(B)
```
**Worked example** — a disease test that's 99% accurate, disease affects 1% of people. Someone tests positive — what's the actual chance they have the disease?
```
P(disease) = 0.01
P(positive | disease) = 0.99         (true positive rate)
P(positive | no disease) = 0.01      (false positive rate)
P(positive) = 0.99*0.01 + 0.01*0.99 = 0.0099 + 0.0099 = 0.0198

P(disease | positive) = 0.99 * 0.01 / 0.0198 = 0.5
```
Only **50%**, not 99% — the disease is rare, so the false positives from the huge healthy population (0.0099) roughly equal the true positives from the tiny sick population (0.0099). The single most common intuition trap in applied stats, and exactly why "the model is 99% accurate" is meaningless without knowing the class balance.

### 3. Bayes' theorem needs a `P(A)` to start from — what are the actual SHAPES that starting probability can take (distributions)?
A distribution describes how likely each possible value is. Two families matter: **discrete** (a fixed list of possible outcomes, each with its own probability) and **continuous** (any real number in a range, described by a density curve).

**Discrete:**

- **Bernoulli(p)** — a single yes/no coin flip with probability `p` of "yes." The building block of binary classification — and of a neural net's **dropout mask**, where each neuron independently survives with probability `p`.
- **Binomial(n,p)** — the *count* of "yes" results across `n` independent Bernoulli(p) trials. Stack `n` independent dropout decisions in one layer and the number of neurons that survive is Binomial(n,p).
- **Categorical(p₁...pₖ)** — Bernoulli generalized past 2 outcomes to `k` outcomes, each with its own probability. **This is exactly what a softmax layer outputs** — next-token prediction is sampling from a categorical distribution over the whole vocabulary.
- **Poisson(λ)** — counts of independent events in a fixed window, given average rate `λ` (e.g. "how many support tickets per hour," or requests/sec hitting an inference endpoint).

**Continuous:**

- **Uniform(a,b)** — every value in `[a,b]` is equally likely (e.g. a common weight-initialization range — "no information yet" before any bias is learned).
- **Normal (Gaussian(μ,σ²))** — the bell curve; most natural measurements, and many aggregated ML quantities, cluster around a mean `μ` with spread `σ` (see question 6 below for *why* — the Central Limit Theorem). Used for Xavier/He weight initialization and the noise injected at each step of a diffusion model's forward process.

**Worked shapes, with real computed probabilities (not illustrative):**
```
Bernoulli(p=0.3)          Binomial(n=10,p=0.5)              Poisson(λ=3)
P(0)=0.70  P(1)=0.30       peak at k=5: 0.246                peak at k=2,3: 0.224
  ▇▇▇▇▇▇▇   ▇▇▇             ▁▁▂▄▆█▆▄▂▁▁  (k=0 to k=10)        ▂▅██▆▄▂▁  (k=0 to k=7)
```
Greedy decoding always takes a categorical distribution's **mode** (its single highest-probability outcome); temperature/top-k/top-p sampling instead draw a random sample from it — same categorical distribution, just reshaped first.

### 4. Given a batch of real data, how do you actually compute its center and spread (mean, variance, std)?
Data: `[2, 4, 4, 4, 5, 5, 7, 9]`. Mean = `40 / 8 = 5`. Variance = average squared distance from the mean = `(9+1+1+1+0+0+4+16)/8 = 32/8 = 4`. Standard deviation = `sqrt(4) = 2`. Std is in the same units as the original data (variance isn't, since it's squared) — that's why std, not variance, is the one usually reported and plotted as error bars.

### 5. Given a mean and spread for two groups, how do you test whether they're GENUINELY different (a p-value), and what does that number NOT mean?
The probability of seeing a result at least this extreme *if the null hypothesis were true* (no real effect). p=0.03 means: "if nothing real were going on, you'd still see data this extreme 3% of the time by pure chance." It is **not** "the probability the null hypothesis is true" — a very common misreading. (`stats-scipy-practice.md` builds this exact idea into a full t-test cascade, with one-tailed/two-tailed, effect size, and power analysis chained on top.)

### 6. That p-value math assumes something about the shape of your data's average — why does it still work even when the raw data ISN'T normally distributed?
The **Central Limit Theorem**: the average of many independent samples tends toward a normal distribution, regardless of the shape of the original data's distribution, as sample size grows. This is why A/B tests can use normal-distribution statistics (like a t-test) on metrics (conversion rate, revenue per user) that individually don't look normal at all — averaging enough of them makes the sampling distribution of the mean approximately normal anyway.

### 7. Two variables move together (correlated) — does that mean one CAUSES the other?
No. Classic counter-example: ice cream sales and drowning deaths correlate strongly — both are caused by a third variable (hot weather), neither causes the other. This exact gap is why `service-impact-and-causal-inference.md` exists as its own topic — correlation is Statistics 101, proving causation needs a randomized experiment or a causal-inference method that approximates one.

### 8. Everything above (p-values, the disease-test Bayes calculation) is a single snapshot — plug numbers in, get one answer. Is there a fundamentally different way to treat probability itself?
Yes — **Bayesian statistics** treats a belief about an unknown quantity as a full DISTRIBUTION that gets updated every time new evidence arrives, rather than a single fixed answer computed once.

**Worked example — estimating a coin's true bias, updated as evidence arrives:**
```
Start with a PRIOR belief: "I have no strong reason to think this coin is biased"
  → prior: Beta(1, 1)  (flat — every bias from 0% to 100% equally plausible)

Flip it 10 times: 7 heads, 3 tails
  → POSTERIOR (updated belief) = Beta(1+7, 1+3) = Beta(8, 4)
  → posterior mean = 8/(8+4) = 0.667 — leans toward "biased toward heads," but still uncertain

Flip it 90 MORE times: 55 heads, 35 tails (100 flips total: 62 heads, 38 tails)
  → posterior = Beta(1+62, 1+38) = Beta(63, 39)
  → posterior mean = 63/(63+39) = 0.618 — narrower, more confident estimate
```
**Visual — the belief distribution physically narrows and shifts as evidence piles on:**
```
Prior (flat)         After 10 flips           After 100 flips
▁▁▁▁▁▁▁▁▁▁▁▁▁         ▁▂▄▆█▆▄▂▁                    ▁▂▅███▅▂▁
0    0.5    1        0   0.67    1               0  0.62   1
```
**Remember it as "yesterday's posterior is today's prior"** — each round's updated belief becomes the starting point for the next round's update. This is also *why* a Beta distribution specifically is used here: it's the "conjugate prior" for yes/no data, meaning the update has this simple closed form (`Beta(a,b)` → `Beta(a+successes, b+failures)`) instead of needing to be solved numerically.

### 9. Given both paradigms now exist side by side, what's the actual practical difference, not just philosophy?
| | Frequentist (questions 1-7 above) | Bayesian (question 8) |
|---|---|---|
| What a "probability" means | Long-run frequency over many repeated experiments | A degree of belief, which can update with evidence |
| Typical output | A p-value, a single confidence interval | A full posterior distribution |
| Prior knowledge | Not formally used | Explicitly incorporated — and must be justified |
| "95% chance the true value is in this range" | **Not** what a frequentist CI means (the same misreading as question 5's p-value trap) | This IS a valid statement about a Bayesian **credible interval** |

### Summary example
An A/B test's frequentist p-value (question 5) says "p=0.04, reject the null" — a real, useful answer, but NOT "96% chance the new version is better." Running the same comparison as a Bayesian update instead (question 8's machinery) would produce a full posterior distribution over "how much better is the new version," from which a statement like "there's an 80% chance the new version improves conversion by at least 2%" becomes a literally valid statement — a genuinely different, and for many business questions more directly useful, answer than the frequentist version.

## Part 3 — Calculus (what's actually running underneath "the model learns")

Every "the model updates its weights" sentence anywhere on this hub is calculus happening.

### 1. What is a derivative, actually — not just the symbolic rule?
A derivative is the **slope of the curve at one exact point** — how fast `y` changes for a tiny nudge in `x` right there.
```
f'(x) = lim(h→0) [ f(x+h) − f(x) ] / h
```
**Worked example** — `f(x) = x²`, slope at `x=3`, computed with shrinking step sizes `h`:
```
h = 1:      (4² − 3²) / 1     = (16 − 9)  / 1     = 7
h = 0.1:    (3.1² − 3²) / 0.1 = (9.61 − 9) / 0.1   = 6.1
h = 0.01:   (3.01² − 3²)/0.01 = (9.0601−9) / 0.01  = 6.01
h = 0.001:  ...                                     = 6.001
```
As `h` shrinks toward 0, the slope converges to exactly **6** — matching the power rule (`d/dx x² = 2x`, `2×3=6`) without ever needing to state the rule as a given. That convergence, not the symbolic rule, is what a derivative *is*.

**Visual — the tangent line touching the curve at one point:**
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
The tangent (slope 6) predicts `15` at `x=4`; the real curve is at `16` — close, not exact, because a straight line is only a LOCAL approximation, exactly what the shrinking-`h` table demonstrates.

### 2. Beyond the power rule that recovered "2x" above, what other derivative rules get used constantly?
| Rule | Formula | Where it shows up |
|---|---|---|
| Power rule | `d/dx xⁿ = n·xⁿ⁻¹` | almost everywhere |
| Constant | `d/dx c = 0` | a bias term contributes 0 to weight gradients |
| Sum rule | `d/dx [f+g] = f' + g'` | loss = sum of per-example losses, so gradient = sum of per-example gradients |
| **Exponential** | `d/dx eˣ = eˣ` | `eˣ` is the *only* function that's its own derivative |
| **Natural log** | `d/dx ln(x) = 1/x` | why cross-entropy's gradient has a clean `1/p` term |
| Chain rule | `d/dx f(g(x)) = f'(g(x)) · g'(x)` | **this one is backprop** — see next |

### 3. That last rule (chain rule) is called "backprop" elsewhere on this hub — is that a loose analogy, or literally the same operation?
Literally the same operation. `h(x) = (3x+1)²` at `x=2` — treat it as outer `f(u)=u²` wrapped around inner `g(x)=3x+1`:
```
g(2) = 3(2)+1 = 7
f'(u) = 2u  ->  f'(g(2)) = 2×7 = 14
g'(x) = 3
h'(2) = f'(g(2)) × g'(x) = 14 × 3 = 42
```
Check by expanding directly: `h(x)=9x²+6x+1`, `h'(x)=18x+6`, at `x=2`: `36+6=42`. Matches exactly. That multiplication — derivative of the outer times derivative of the inner — is precisely what backpropagation computes, one layer at a time, from the loss back to the first layer's weights. And the vanishing-gradient problem (`ds-fundamentals`, `common-issues-failure-modes.md`) is a direct CONSEQUENCE of this same multiplication: many numbers smaller than 1 multiplied together shrink toward zero (`0.25²⁰ ≈ 2.4×10⁻⁷`) — the exact same 2-step multiplication above, just repeated 20 times instead of 2.

For this same chain rule carried out over a full toy neural net — real weight matrices, a real forward pass, loss, backprop into every layer, and the resulting weight update, all with actual numbers instead of one variable — see the **Neural Net Numericals** topic.

### 4. A real loss function depends on millions of weights, not just one `x` — how does "the derivative" even generalize to that?
A **partial derivative** is a derivative taken with respect to ONE variable while holding every other fixed. The **gradient** is the vector of ALL those partial derivatives together, pointing in the direction of steepest INCREASE — which is exactly why gradient descent moves in the OPPOSITE direction (`ds-fundamentals`'s hand-worked gradient-descent step is this section applied at ML scale).

### 5. The exponential rule above (`d/dx eˣ = eˣ`) made `e` sound special — what does the exponential CURVE actually look like, and why is it everywhere in ML?
```
x        eˣ
-2       0.14
-1       0.37
 0       1.00
 1       2.72
 2       7.39
 3      20.09
```
Flattens toward (never reaching) 0 for negative `x`, equals exactly 1 at `x=0`, then grows explosively for positive `x`. Why `exp` specifically for turning raw logits (any real number, including negative) into probabilities: **`eˣ` is positive for every possible input**, no matter how negative — exactly softmax's requirement (`softmax(x)ᵢ = e^xᵢ / Σⱼ e^xⱼ`, and a probability can never legally be negative). Combined with being its own derivative, it's both mathematically convenient and the cleanest fit for the always-positive requirement.

### 6. Given `eˣ`'s shape, what does its exact INVERSE (the log rule from question 2) look like, and why does ML pair the two constantly?
```
x        ln(x)
0.1      -2.30
0.5      -0.69
1         0.00
2         0.69
5         1.61
10        2.30
```
Steep and plunging toward `-∞` as `x→0⁺`, crosses `0` at `x=1`, climbs ever more slowly after — `ln(x)` is undefined for `x≤0` entirely. `eˣ` and `ln(x)` are exact inverses (`ln(eˣ)=x`), which is why a "log-sum-exp" step shows up constantly in softmax/cross-entropy implementations, moving between the two safely. Two concrete reasons logs matter beyond being an inverse: they turn multiplication into addition (`log(a×b)=log(a)+log(b)` — a whole dataset's likelihood is a PRODUCT of per-example probabilities that underflows to 0 on real hardware; the log turns it into a numerically-stable SUM), and they punish confident wrongness harshly while confident correctness stays cheap — exactly cross-entropy's `−log(p)` shape (predicting the true class at 78.56% confidence costs `0.24`; at 3.91% confidence costs `3.24` — over 13× harsher, because of `log`'s steep plunge near 0).

### Summary example
A transformer's forward pass ends at a loss value computed via cross-entropy's `−log(p)` (question 6), where `p` came from softmax's `eˣ/Σeˣ` (question 5) applied to raw logits. Backprop (question 3) then computes the gradient of that loss with respect to every weight via repeated chain-rule multiplication, layer by layer, using the partial-derivative/gradient machinery from question 4 to handle millions of weights at once — and if the network is 20 layers deep with small per-layer gradients, that same chain-rule multiplication is exactly what shrinks the earliest layer's gradient toward the vanishing `0.25²⁰` number from question 3, not a separate phenomenon.

## Practice Q&A (Self-Test)

### Why does `Q @ Kᵀ` need the transpose — what would happen without it?
`Q` is `[seq_len × d_k]` and `K` is also `[seq_len × d_k]`. Matrix multiplication requires the inner dimensions to match, so `Q @ K` (both `[seq_len × d_k]`) is invalid — the `d_k` from `Q`'s columns doesn't line up with `K`'s rows (`seq_len`). Transposing `K` to `[d_k × seq_len]` makes the shapes `[seq_len × d_k] @ [d_k × seq_len] = [seq_len × seq_len]`, valid and exactly the "every token scored against every token" grid attention needs.

### A test is 99.9% accurate and the condition affects 1 in 10,000 people. Roughly, is a positive result more likely to be a true positive or a false positive?
False positive, by a wide margin — apply Bayes' theorem the same way as the worked disease example above. With a condition this rare, the 0.1% false-positive rate applied to the huge healthy population produces more false alarms than true detections applied to the tiny affected population. This is why rare-event detectors (fraud, rare disease, rare mechanical failure) need very low false-positive rates specifically, not just high overall accuracy.

### Why does L1 regularization zero out weights while L2 only shrinks them?
L1's penalty (`sum(|w|)`) grows linearly, so shrinking any weight — big or small — saves the same amount of penalty per unit, making it worth pushing unimportant weights all the way to exactly 0. L2's penalty (`sum(w²)`) grows quadratically, so it aggressively shrinks large weights but the "reward" for the last little bit near zero is tiny — it shrinks weights smoothly but essentially never reaches exactly 0.

### You get p = 0.04 on an A/B test. Is it correct to say "there's a 96% chance the new version is actually better"?
No. p = 0.04 means: if the new version had *no real effect*, you'd still see a difference this large 4% of the time by chance alone. It says nothing directly about the probability that the new version is truly better — that would require a Bayesian framing with a prior, not a plain p-value.

### Why can A/B tests use normal-distribution statistics even when the underlying metric (e.g. "did this user convert," 0 or 1) isn't normally distributed at all?
The Central Limit Theorem: the *distribution of the sample mean* (average conversion rate across thousands of users) approaches a normal distribution as sample size grows, even though each individual observation is a lopsided 0/1 Bernoulli outcome. The test doesn't need individual data points to be normal — only the averaged statistic being compared.

### After flipping a coin 10 times and updating a Beta(1,1) prior, you flip it 90 more times. Do you throw away the first update and recompute from scratch using all 100 flips, or update again from where you left off?
Either approach gives the mathematically identical final posterior — `Beta(1,1)` updated once with all 100 flips (62 heads, 38 tails) gives `Beta(63, 39)`, exactly the same as updating first with 10 flips to get `Beta(8,4)` and then updating that result with the next 90 flips (`Beta(8+55, 4+35) = Beta(63, 39)`). This is exactly what "yesterday's posterior is today's prior" means in practice — Bayesian updating doesn't care whether evidence arrives in one batch or many small sequential ones.

### Why is it wrong to say "there's a 95% chance the true value is in this range" about a plain frequentist confidence interval, but correct to say about a Bayesian credible interval?
A frequentist confidence interval is built from a procedure that, if repeated over many hypothetical experiments, would contain the true (fixed, non-random) parameter 95% of the time — the 95% describes the reliability of the *procedure* across repeats, not a probability about this one specific interval. A Bayesian credible interval instead directly describes a range of the posterior *distribution* over plausible values, so "95% chance the true value is in this range" is a literally accurate description of what a credible interval represents.

### Why does shrinking `h` toward 0 in `[f(x+h) − f(x)] / h` give you the derivative, rather than just being an approximation forever?
Each smaller `h` gives a slightly more accurate slope estimate because it measures the curve over a shorter, straighter-looking stretch. As `h → 0`, that stretch shrinks to a single point and the approximation error shrinks to exactly 0 — the limit is the derivative, not an estimate of it. The worked table (`h=1 → 7`, `h=0.1 → 6.1`, `h=0.01 → 6.01`, converging to 6) is that shrinking happening with real numbers.

### Why is the chain rule specifically the mathematical operation behind backpropagation, rather than just "related to" it?
Backprop computes how the loss changes with respect to an early layer's weights, but the loss only touches those weights *indirectly* — through every layer in between. That's exactly the "function of a function" shape the chain rule is built for: derivative of the outer computation times derivative of the inner one, layer after layer, multiplying all the way back. It's not an analogy — backprop is the chain rule applied repeatedly, which is also why 20 layers of small (<1) local gradients multiply down to a vanishing gradient rather than just adding up.

### Cross-entropy loss uses `−log(p)`. Why the negative sign, and why log at all instead of just using `1 − p` as the loss?
`log(p)` is negative for any valid probability (`p ≤ 1`), so negating it gives a positive loss, matching the "loss should be a positive number that's bigger when the model is worse" requirement. Using `log` specifically (instead of a linear `1−p`) is what creates the steep-near-zero shape: predicting the true class with 3.91% confidence should be punished far more than proportionally harder than 78.56% confidence, and the log curve's plunge toward `−∞` as its input approaches 0 delivers exactly that non-linear, increasingly severe penalty.

### Softmax turns raw logits (which can be negative) into probabilities using `eˣ`. Why not just normalize the raw logits directly (divide each by their sum) instead of exponentiating first?
Raw logits can be negative or sum to a value close to zero, so dividing them directly by their sum can produce negative "probabilities" or wildly unstable results — not valid probabilities at all. `eˣ` is guaranteed positive for *any* real input, no matter how negative, so exponentiating first guarantees every value going into the normalization is a valid positive number, and only then does dividing by the sum produce numbers that are actually positive and sum to 1, as a real probability distribution must.
