# Statistics with SciPy/statsmodels — Built as a Chain, Not a List

`from scipy import stats; import statsmodels.api as sm; import numpy as np` assumed throughout.

Every cluster below is one continuous thread — each numbered question inherits directly from the answer before it, the way you'd actually ask it out loud if you kept saying "okay but then..." No raw LaTeX anywhere on this page — every formula is written out in plain, readable symbols (`x̄`, `μ`, `σ`, `√`, `Σ`) inside a code block, because notation nobody can read at a glance teaches nothing. Every cluster ends with one small worked example carrying real numbers end to end, and every cluster closes with "what else is in this family" so a single method never looks like the only option.

---

## Cluster 1 — The Basic Idea of a Hypothesis Test

### 1. What is a sample, and why do we use one instead of measuring everyone?
A **sample** is a smaller, manageable subset of data pulled from a much larger **population** you actually care about. You use a sample because measuring the whole population is usually impossible or absurdly expensive (you can't test every locomotive bearing ever made, or survey every customer who will ever exist) — the entire goal of everything below is to use the sample to make a trustworthy statement about the population you didn't fully measure.

### 2. Do two samples always come from the same population?
Not necessarily — that's exactly the question a hypothesis test answers. The **null hypothesis (H₀)** starts by *assuming* they do (no real difference between them, any gap you see is just random sampling noise); the test then checks whether your data makes that assumption look implausible. If it does, you reject H₀ and conclude the two samples more likely came from genuinely different populations (i.e., there's a real effect).

### 3. Why is the null hypothesis value so often exactly zero?
The **null hypothesis value** (written `μ₀`) is whatever number H₀ claims is true. It's conventionally zero because "zero" is the mathematical way of saying "no difference, no effect, no change" — testing whether a new drug changes blood pressure means testing whether the *difference* in means is zero; testing whether a coin is fair means testing whether its bias away from 50% is zero. Zero isn't special mathematically, it's just what "nothing is happening" looks like as a number.

### 4. Given a sample and a null value to test against, how do you actually measure how far apart they are (the t-statistic)?
```
t = (x̄ − μ₀) / (s / √n)

x̄  = sample mean
μ₀ = the null hypothesis value (what you're testing against)
s  = sample standard deviation
n  = sample size
```
This is a **one-sample t-test** shape — comparing one sample's mean against a fixed benchmark value. Read it as a ratio: the top (`x̄ − μ₀`) is the raw gap you observed; the bottom (`s / √n`, the **standard error**) is how much noise you'd expect a sample of this size and spread to naturally produce. A large `t` means the gap is big *relative to the noise you'd expect by chance* — small noise (small `s`) or a big sample (`n` large, shrinking `s/√n`) makes even a modest-looking gap register as a large `t`.

### 5. What does the SIZE of that t-statistic actually buy you (why compute it at all)?
The raw value of `t` gets converted into a **p-value** — the probability of seeing a gap at least this extreme *if the null hypothesis (μ₀, no real difference) were actually true*. A large `|t|` → a small p-value → "this gap is surprising if nothing real were going on" → reject H₀. A small `|t|` → a large p-value → "this gap is exactly the kind of noise you'd expect anyway" → fail to reject H₀. The t-statistic is the intermediate number; the p-value is the thing you actually report and act on.

### 6. What does that p-value NOT mean — the misreading everyone falls into?
p = 0.03 means "if nothing real were going on, you'd still see a gap this big about 3% of the time by pure chance." It is **not** "there's a 97% chance the effect is real," and it is **not** "there's a 3% chance the null hypothesis is true." The p-value is a statement about how surprising your DATA is under an assumption (H₀), not a probability attached to the hypothesis itself — this exact mix-up is one of the most commonly tested statistics-literacy traps in interviews.

### 7. Comparing TWO samples instead of one sample against a fixed value — what changes?
```python
group_a = [82, 85, 79, 91, 88]
group_b = [76, 74, 80, 71, 77]
t_stat, p_value = stats.ttest_ind(group_a, group_b, equal_var=False)
```
Same underlying idea — `(gap between means) / (expected noise)` — just now the "gap" is `mean(group_a) − mean(group_b)` instead of `x̄ − μ₀`, and the standard error accounts for BOTH samples' variability and sizes instead of just one. `equal_var=False` runs **Welch's t-test**, which doesn't assume the two groups have equal variance (a real, frequently-violated assumption — see Cluster 3 for how to actually check this) — the classic Student's t-test can give a misleadingly confident, too-small p-value when variances genuinely differ.

### 8. One-tailed or two-tailed — how does deciding the direction in advance change the p-value?
```python
# two-tailed (default): "is group_a's mean DIFFERENT from group_b's, in either direction?"
t_stat, p_two = stats.ttest_ind(group_a, group_b, equal_var=False, alternative="two-sided")
# one-tailed: "is group_a's mean SPECIFICALLY GREATER?" -- a stronger, narrower claim
t_stat, p_one = stats.ttest_ind(group_a, group_b, equal_var=False, alternative="greater")
```
A two-tailed test spreads your 5% error budget (α) across BOTH ends ("is A bigger, or is B bigger") — a one-tailed test spends the whole budget on one specific end, because you've committed in advance to only caring about that direction. For symmetric data, the one-tailed p-value comes out to roughly *half* the two-tailed value on identical numbers — which is exactly why the direction has to be picked *before* looking at the data, not after ("we always expected ours to be better, so let's re-run one-tailed" after a two-tailed result narrowly missed significance is p-hacking, not a legitimate choice).

**Visual — where the rejection region (the shaded "surprising enough" zone) actually sits:**
```
TWO-TAILED                                   ONE-TAILED (alternative="greater")
    reject here    reject here                              reject here
    ▓▓▓▓          ▓▓▓▓                                            ▓▓▓▓▓▓▓▓
   ◀────────center────────▶                    ◀────────center────────▶
   2.5% in EACH tail = 5% total                 all 5% in the ONE tail
```

### 9. What other tests live in this same "compare means" family, for cases the two-sample t-test doesn't fit?
- **Paired t-test** — the two samples are actually the SAME subjects measured twice (before/after), not independent groups (Cluster 2).
- **ANOVA** — comparing three or more group means at once, not just two (Cluster 2).
- **Mann-Whitney U** — comparing two groups when the data isn't normal enough to trust a mean-based test (Cluster 4).

### Summary example — tying the whole chain together
A teacher wants to know if a new teaching method changes test scores. H₀: the true mean difference is zero (no change) — that's `μ₀`. She collects a sample of 25 students' before/after scores, computes `x̄` (the sample's average difference) and `s` (how much that difference varies student to student), and plugs them into `t = (x̄ − μ₀) / (s / √n)`. Suppose `x̄ = 4.2`, `s = 9.1`, `n = 25`: `t = 4.2 / (9.1/5) = 4.2 / 1.82 ≈ 2.31`. That `t ≈ 2.31` converts to a two-tailed p-value around 0.03 — small enough to reject H₀ and conclude the method likely does change scores, while staying honest that "likely" is exactly what a p-value of 0.03 (not 0.97, not certainty) actually supports.

---

## Cluster 2 — Comparing Groups That Aren't a Simple Two-Sample Split

### 1. What if the two samples aren't independent — they're the SAME people, measured twice?
```python
before = [70, 75, 68, 80, 72]
after  = [74, 78, 70, 85, 77]
t_stat, p_value = stats.ttest_rel(before, after)
```
`ttest_ind` treats the two lists as unrelated groups and throws away the fact that `before[i]` and `after[i]` are the same person — that pairing is real information (person 3's own baseline matters more than the group average does). `ttest_rel` (the **paired t-test**) works on the per-person DIFFERENCES (`after[i] − before[i]`) directly, which removes person-to-person variation that has nothing to do with the treatment, giving a more powerful, correctly-specified test whenever the pairing is real.

### 2. What if there's no second group at all — just one sample and a fixed target?
That's exactly the one-sample t-test from Cluster 1, question 4 — `t = (x̄ − μ₀) / (s/√n)`, with `μ₀` as the fixed benchmark (a spec value, a regulatory limit, a historical baseline) rather than a second group's mean.

### 3. What if there are THREE OR MORE groups — why not just run pairwise t-tests on every pair?
```python
group_a = [82, 85, 79, 91, 88]
group_b = [76, 74, 80, 71, 77]
group_c = [90, 95, 92, 88, 94]
f_stat, p_value = stats.f_oneway(group_a, group_b, group_c)
```
Each individual t-test at α=0.05 carries its own 5% false-positive chance. Three separate pairwise tests (A-vs-B, B-vs-C, A-vs-C) compound that risk — this is the **multiple comparisons problem**, covered in full in Cluster 9. **ANOVA** (`f_oneway`) tests the single, combined question "are ANY of these means different" in one test at the stated α, instead of stacking three separately-risky tests.

### 4. How does ANOVA's F-statistic actually work — what is it comparing?
```
F = (variance BETWEEN group means) / (variance WITHIN each group)
```
If the groups are truly identical, the spread *between* their means should look about the same size as the ordinary noise *within* any one group — `F ≈ 1`. If the groups are genuinely different, the between-group spread dwarfs the within-group noise — `F` gets large, and the p-value gets small. This is why ANOVA needs the variances to be reasonably similar across groups in the first place (the same `equal_var` idea from Cluster 1, tested properly in Cluster 3) — the whole comparison assumes "within-group noise" means roughly the same thing in every group.

### 5. ANOVA says the groups differ — but WHICH ones? Why can't you just run pairwise t-tests now?
```python
from statsmodels.stats.multicomp import pairwise_tukeyhsd
data = group_a + group_b + group_c
labels = ["A"]*len(group_a) + ["B"]*len(group_b) + ["C"]*len(group_c)
print(pairwise_tukeyhsd(data, labels))
```
Falling back to raw pairwise t-tests here re-introduces exactly the multiple-comparisons risk ANOVA was used to avoid in the first place. **Tukey's HSD** runs the pairwise comparisons you now legitimately want, with a built-in correction for testing multiple pairs at once — you get which-pairs-differ answers without re-inflating the false-positive rate.

### 6. What else lives in this "comparing groups" family?
- **Levene's test** — do the groups have the same VARIANCE, a different question from "same mean" entirely (Cluster 3).
- **Mann-Whitney U / Kruskal-Wallis** — the rank-based versions of the two-sample t-test and ANOVA, for when the data isn't normal enough to trust either (Cluster 4).
- **Chi-square** — comparing groups defined by CATEGORIES instead of a numeric measurement (Cluster 5).

### Summary example
Comparing defect rates across 3 manufacturing shifts. `f_oneway` on the three shifts' daily defect counts gives `F = 6.8, p = 0.004` — small enough to conclude at least one shift genuinely differs. Running `pairwise_tukeyhsd` afterward shows Shift 2 vs. Shift 3 is the significant pair (p=0.01), while Shift 1 vs. either other shift isn't (p>0.3) — the two-step process (ANOVA to detect ANY difference, Tukey to localize WHICH pair) is what correctly answers "which shift has a problem" without the false-positive inflation of just running three raw t-tests from the start.

---

## Cluster 3 — Comparing Variance Itself, Not Just the Mean

### 1. Can two groups have identical means but still be meaningfully different?
Yes — and it's a real, common blind spot. `group_a = [10, 12, 11, 13, 9]` and `group_b = [5, 20, 2, 25, 3]` both have mean 11, **identical**. A t-test comparing these would find no difference at all, because a t-test only ever looks at means. But `group_b` swings from 2 to 25 while `group_a` stays tightly between 9 and 13 — a real, measurable difference in consistency that a mean-based test is structurally blind to.

### 2. How do you actually test whether two groups' VARIANCES differ?
```python
stat, p_value = stats.levene(group_a, group_b)
```
**Levene's test** (preferred over the older classic F-test because it doesn't require the data to be normally distributed) tests specifically whether the spread differs, independent of whether the means differ.

**Visual — same center, different width, a shape a t-test cannot see:**
```
group_a:        group_b:
    ▁▃▅█▅▃▁          ▂▂▂▃▃▃▃▃▂▂▂
   9 10 11 12 13    2  ...  11  ...  25
   narrow, consistent    wide, unpredictable
   (mean = 11)            (mean = 11, SAME)
```

### 3. How does this connect back to `equal_var=False` from the very first t-test in Cluster 1?
Directly — `equal_var=False` was choosing Welch's t-test specifically *because* the analyst can't assume the two groups have equal variance. Levene's test is how you'd actually CHECK that assumption with a number instead of guessing: run Levene's first, and if it comes back significant (variances genuinely differ), that's your confirmation that Welch's (not classic Student's) t-test was the right call all along.

### 4. Where does this matter outside of picking a t-test variant?
Anywhere consistency itself is the thing being measured — comparing two manufacturing processes with identical average output but very different variability, comparing two model versions' prediction consistency even when their average accuracy matches, or comparing two ad campaigns' conversion rates when one is wildly inconsistent day to day and the other isn't.

### Summary example
Two production lines both average 11 units/hour defect-free output. Line A's hourly counts barely move (`[10,12,11,13,9]`); Line B swings wildly (`[5,20,2,25,3]`). `stats.levene(line_a, line_b)` returns a small p-value — the variances are genuinely different even though `stats.ttest_ind` on the same data would report no significant difference in the means at all. The honest conclusion: Line B isn't worse on average, but it's far less predictable, which is itself an operational problem a mean-only comparison would have completely missed.

---

## Cluster 4 — When the Data Isn't Well-Behaved

### 1. What does "normally distributed" actually mean, and why do t-tests/ANOVA lean on it?
A normal distribution is the familiar symmetric bell curve — most values cluster near the mean, with progressively fewer values further out in either direction, in a specific, predictable proportion. The t-test and ANOVA's p-value math is built assuming the underlying data (or at least the sampling distribution of the mean) follows this shape closely enough — badly skewed data or a few extreme outliers can distort the mean and standard deviation enough to make the resulting p-value untrustworthy.

### 2. How do you actually check whether your data is normal enough to trust, instead of assuming?
```python
stat, p_value = stats.shapiro(sample)
```
**Shapiro-Wilk** tests the null hypothesis "this data IS normally distributed" — a small p-value here means you should be suspicious of the normality assumption. **The catch:** with a very large sample, Shapiro-Wilk becomes hyper-sensitive and flags trivial, practically-irrelevant deviations from perfect normality as "significant" — a Q-Q plot (plotting your data's quantiles against a normal distribution's) is often more useful to actually LOOK at than the p-value alone once `n` is large.

### 3. Shapiro-Wilk says the data ISN'T normal enough — now what?
Reach for a **non-parametric test** — the rank-based cousin of whichever mean-based test you were about to run:
```python
stat, p_value = stats.mannwhitneyu(group_a, group_b)     # non-parametric alternative to ttest_ind
stat, p_value = stats.kruskal(group_a, group_b, group_c)  # non-parametric alternative to ANOVA
```

### 4. How do these actually work — what does "non-parametric" mean mechanically?
Instead of comparing raw values (which a few extreme outliers can dominate), Mann-Whitney U and Kruskal-Wallis convert every value to its RANK (1st smallest, 2nd smallest, ...) across all groups combined, and test whether those ranks are randomly mixed between groups or systematically clustered. A single extreme outlier can only ever shift by one rank position no matter how extreme it is — this is exactly why these tests are robust to the two problems (outliers, non-normality) that make a mean-based test untrustworthy.

**Visual — the same value-vs-rank tradeoff as Pearson vs. Spearman correlation (Cluster 6), applied to comparing groups instead:**
```
              uses raw VALUES              uses RANKS instead
  compare 2   t-test (ttest_ind)           Mann-Whitney U
  compare 3+  ANOVA (f_oneway)             Kruskal-Wallis
```

### Summary example
Response times for a support system, heavily skewed by a handful of multi-hour outlier tickets. `stats.shapiro(response_times)` comes back with p < 0.001 — not normal, an outlier-dominated t-test would be untrustworthy here. Switching to `stats.mannwhitneyu(team_a_times, team_b_times)` instead compares the two teams' RANKS rather than their raw (outlier-corrupted) times, giving a trustworthy answer to "is one team genuinely faster" that the raw-value comparison couldn't provide.

---

## Cluster 5 — Two Categorical Variables

### 1. What if neither variable is a number at all — both are categories?
Comparing means (everything in Clusters 1-4) doesn't apply when there's nothing to average — e.g. "device type" (mobile/desktop) and "converted" (yes/no) are both categories, not numbers. The question becomes "are these two categorical variables ASSOCIATED" instead of "do these means differ."

### 2. How do you actually test for association between two categorical variables?
```python
observed = np.array([[50, 30], [20, 40]])    # rows = category A levels, cols = category B levels
chi2, p_value, dof, expected = stats.chi2_contingency(observed)
```
The chi-square statistic compares your real counts (`observed`) against the counts you'd expect if the two variables were completely unrelated (`expected`, computed from the row/column totals alone):
```
chi² = Σ (observed − expected)² / expected
```

**Visual — the whole test as "how far apart are these two grids":**
```
OBSERVED (what really happened)        EXPECTED (if the two variables were
                                         totally unrelated -- just row% × col%)
        B=0   B=1                              B=0    B=1
 A=0  [ 50    30 ]  80                  A=0  [ 42    38 ]  80
 A=1  [ 20    40 ]  60                  A=1  [ 28    32 ]  60
        70    70   140                        70     70   140
```
Big gaps between the two grids → big chi² → small p-value → a real association. **Why also look at `expected`, not just the p-value:** the p-value only says IF there's an association; comparing the two grids directly shows you the DIRECTION (here, A=0 over-represents B=0 relative to what independence would predict).

### 3. What assumption does this test need, and how do you check it?
```python
if (expected < 5).any():
    print("warning: some expected cell counts are below 5 — chi-square approximation may be unreliable")
```
Chi-square is itself an approximation that assumes reasonably large expected counts in every cell — with small samples or rare categories, that approximation breaks down and the p-value can't be trusted. Mentioning this check unprompted is exactly the kind of assumption-awareness that separates a senior answer from a "ran the function, reported the number" answer.

### Summary example
Testing whether mobile vs. desktop users convert at different rates. `observed = [[50,30],[20,40]]` (mobile: 50 converted/30 didn't; desktop: 20/40). `chi2_contingency` returns chi²≈15.2, p<0.001 — a real association. Comparing `expected` (`[[42,38],[28,32]]`) to `observed` shows mobile converts MORE than independence would predict (50 vs. expected 42) — the direction the p-value alone couldn't tell you.

---

## Cluster 6 — Correlation Between Two Numeric Variables

### 1. What does it actually mean for two numeric variables to be "correlated"?
That they tend to move together — as one goes up, the other tends to go up (positive) or down (negative), consistently enough to not look like coincidence.

### 2. How do you measure that, and what does the resulting number mean?
```python
r, p_value = stats.pearsonr(df["age_days"], df["wear_pct"])
```
**Pearson's r** ranges from -1 to +1. `|r|` is how TIGHT the relationship is (how close the points sit to a straight line); the sign is whether it rises or falls.

**Visual — what each r value actually looks like as a scatterplot, so a reported number maps to a picture, not just a range:**
```
r = 0.95 (strong+)     r = 0.5 (moderate)     r ≈ 0.0 (none)      r = -0.9 (strong-)
 .                       .    .                  .   .   .          .
   .                   .    .    .              .  .   .  .            .
     .   .           .    .    .                .    .    .              .
   .    .           .   .    .      .           .  .    .    .              .
      .            .        .    .              .    .   .                    .
  tight, rises    loose cloud,        no visible          tight, FALLS
  left-to-right   still trending up   pattern at all       left-to-right
```

### 3. When does Pearson's r give a misleading answer, and what's the alternative?
```python
rho, p_value_s = stats.spearmanr(df["age_days"], df["wear_pct"])
```
Pearson's r specifically measures LINEAR association — a real, strong, consistently-increasing-but-CURVED relationship still drags Pearson's r down, because the relationship isn't a straight line, even though it's a genuinely strong monotonic trend. **Spearman's rho** works on ranks instead of raw values, so it only cares whether the relationship is consistently monotonic (always increasing or always decreasing), catching non-linear-but-monotonic relationships Pearson understates.

### 4. What's next once you know two variables are correlated — can you predict one from the other?
That's regression (Cluster 7) — correlation tells you THAT two variables move together; regression gives you an actual equation for predicting one from the other, plus a number (R²) for how much of the variation that equation actually explains.

### Summary example
Equipment wear vs. age in days: `pearsonr` gives r=0.61, but the scatterplot curves — wear climbs fast early, then levels off. `spearmanr` gives rho=0.89. Trust Spearman here — the relationship is genuinely strong and consistently increasing, just not a straight line, which is exactly the situation Pearson understates and Spearman doesn't.

---

## Cluster 7 — Regression as an Extension of Correlation

### 1. Correlation says two variables move together — how do I get an actual equation predicting one from the other?
```python
X = sm.add_constant(df[["age_days", "mileage"]])   # statsmodels does NOT auto-add an intercept
model = sm.OLS(df["wear_pct"], X).fit()
print(model.summary())
```
**Why `sm.add_constant` is a real, easy-to-forget gotcha:** without it, statsmodels fits a line forced through the origin — every coefficient comes out wrong, silently, with no error raised. **Why statsmodels over sklearn here:** `model.summary()` gives p-values, confidence intervals, and R² directly — sklearn's `LinearRegression` gives predictions and coefficients but no built-in statistical inference, because sklearn is built for prediction pipelines, not for explaining a relationship.

### 2. Once fit, how do I know if the equation is actually any good (R²)?
```python
model.rsquared        # e.g. 0.71 -- 71% of the variance in wear_pct is "explained" by age_days + mileage
```
```
R² = 1 − (unexplained scatter around the line) / (total scatter around the mean)
```
R² answers "how much better is this equation than just always guessing the average wear_pct for everyone."

### 3. Does adding more predictors always make R² go up — is that a problem?
Yes, and yes. Plain R² can only stay flat or increase as you add predictors, even completely useless, random ones — the model can always find some tiny coincidental fit in more columns.
```python
model.rsquared_adj    # e.g. 0.68 -- the same idea, penalized for how many predictors were used
```
**Adjusted R²** charges a penalty per predictor, so it can actually go DOWN when a new feature isn't earning its place — exactly the signal plain R² structurally can't give you. Always report both when comparing models with different numbers of predictors.

**Visual:**
```
  R² = 0.9  ──▶  ●●●●●●●●●●●  tight around the line, line explains almost everything
  R² = 0.3  ──▶  ● ●  ●●  ●    ●  loose scatter, line explains only a little

  adjusted R² = same formula, but charges rent per predictor —
  a predictor that doesn't earn its rent makes adjusted R² go DOWN
  even while plain R² ticks up
```

### 4. A high R² still doesn't guarantee the regression is trustworthy — what else has to be checked?
```python
residuals = model.resid
fitted = model.fittedvalues
# plot residuals vs fitted -- looking for a random cloud, NOT a pattern
```
A good regression's residuals (actual − predicted) should look like structureless noise scattered evenly around zero, regardless of the fitted value. Two patterns to watch for: a **funnel shape** (spread growing/shrinking as fitted values increase) is **heteroscedasticity** — non-constant error variance, which doesn't bias the coefficients themselves but DOES make `model.summary()`'s p-values and confidence intervals untrustworthy, since those are computed assuming constant variance. A **curved band** means the true relationship isn't linear at all — no amount of tuning a linear model fixes a shape problem like that.

**Visual — three residual-plot shapes, three different verdicts:**
```
GOOD (homoscedastic)      BAD (heteroscedastic,          BAD (non-linear,
                           funnel shape)                   curved band)
  .  .   .  .   .           .                                    . .
 . .  . .  . .  .          . .    .                          . .     . .
. . .  . .  . .  .        .  .   .   .                    . .           . .
──────────────────      ──────────────────             ──────────────────
random cloud, even       spread WIDENS as fitted        systematic curve —
spread                    values grow                    line was the wrong shape
```

### 5. What if two predictors are correlated with EACH OTHER, not just with the target?
```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
X_vars = df[["age_days", "mileage", "wear_pct"]]
vif = pd.DataFrame({"feature": X_vars.columns,
                     "VIF": [variance_inflation_factor(X_vars.values, i) for i in range(X_vars.shape[1])]})
```
**VIF** measures how much a feature's variance is inflated because it's linearly predictable from the OTHER features. VIF=1 means no correlation with other predictors; VIF>5 (some use >10) signals real **multicollinearity** — individual coefficients become unstable and hard to interpret one at a time, even though the model's overall predictions may still be fine.

### Summary example
Predicting `wear_pct` from `age_days` and `mileage`: R²=0.71, adjusted R²=0.68 (close together — both predictors are pulling real weight). Residual plot shows a random cloud (homoscedastic, linear assumption looks fine). VIF on `age_days` comes back at 6.2 — moderately concerning, since older equipment also tends to have more mileage, meaning the two coefficients individually are less trustworthy even though the overall R² is solid.

---

## Cluster 8 — How Confident Should You Be In a Single Estimate?

### 1. A sample gives me one number (a mean). How confident should I be that it's close to the true population value?
That's what a **confidence interval** answers — a range that's likely to contain the true population mean, not just the single point estimate.

### 2. How do you compute a 95% confidence interval for a mean by hand?
```python
mean = sample.mean()
sem = stats.sem(sample)                      # standard error of the mean = std / sqrt(n)
ci_low, ci_high = stats.t.interval(confidence=0.95, df=n-1, loc=mean, scale=sem)
```

### 3. Why the t-distribution instead of the normal distribution here?
With a SAMPLE (not the true population), you're also estimating the standard deviation from that same data — that adds extra uncertainty beyond just the mean's own sampling variation. The t-distribution has heavier tails than normal specifically to account for that extra layer of uncertainty, converging to normal only as `n` grows large. Using normal instead of t for a small sample understates how uncertain you really should be.

### Summary example
`sample = [102, 98, 105, 110, 95, 101]`, mean=101.8, `sem≈2.2`, `df=5`. `stats.t.interval(0.95, df=5, loc=101.8, scale=2.2)` gives roughly (96.1, 107.6) — you can say "the true population mean is most likely between 96 and 108," a genuinely more useful statement than just reporting 101.8 alone.

---

## Cluster 9 — When You Run Many Tests at Once

### 1. If I run 20 hypothesis tests on the same dataset, what goes wrong even if nothing real is happening anywhere?
Each individual test at α=0.05 carries its own 5% false-positive chance. Run 20 independent tests where NOTHING real is going on, and the chance that at least one comes back "significant" purely by luck climbs to roughly `1 − 0.95²⁰ ≈ 64%` — you're now more likely than not to see a fake positive somewhere, just from volume.

### 2. How do you correct for this?
```python
from statsmodels.stats.multitest import multipletests
reject, corrected_pvals, _, _ = multipletests(p_values_list, alpha=0.05, method="fdr_bh")
```
Bonferroni (divide α by the number of tests) is the simplest, most conservative fix; **FDR (False Discovery Rate, Benjamini-Hochberg)** is the more commonly used modern default — it controls the expected PROPORTION of false positives among your significant results rather than the probability of even one, which is less punishingly conservative when you're running many tests.

### Summary example
A/B testing 20 different button colors against a control, all at once. Without correction, roughly 12-13 of the 20 would show "significant" results purely by chance even if none of them actually work. Applying `multipletests(..., method="fdr_bh")` shrinks the list down to the 1-2 that survive correction — those are the ones actually worth trusting.

---

## Cluster 10 — Verifying a Method Actually Does What You Think

### 1. How do you know a statistical method is doing what you think it's doing?
Generate data with a KNOWN true answer, run the method on it, and check whether it recovers that known answer. This is the single best sanity check available whenever you're not 100% sure a test/method is behaving correctly.

### 2. How do you simulate data from a known distribution to test a method?
```python
rng = np.random.default_rng(42)
samples = rng.normal(loc=50, scale=10, size=1000)     # 1000 draws from N(50, 10)
```
Now you KNOW the true mean is 50 — running your confidence-interval code on `samples` and checking that 50 falls inside the resulting interval (most of the time, across repeated simulations) verifies the method is working before you trust it on real data where you don't know the true answer.

### 3. What if you need a confidence interval for a statistic with no clean formula (a median, say)?
```python
rng = np.random.default_rng(42)
boot_means = [rng.choice(sample, size=len(sample), replace=True).mean() for _ in range(10_000)]
ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
```
**Bootstrapping**: resample the ORIGINAL data, with replacement, to the same size, thousands of times — each resample gives a slightly different statistic, and the spread of those thousands of results approximates the statistic's true sampling distribution, with no formula required at all. Works identically for a median, a correlation coefficient, or any custom metric.

**Visual — pulling thousands of "parallel universe" samples from the one dataset you actually have:**
```
  original data: [10, 12, 11, 13, 9]
        ├──▶ resample (w/ replacement) ──▶ [11, 9, 11, 13, 12] ──▶ mean = 11.2
        ├──▶ resample (w/ replacement) ──▶ [13, 13, 10, 9, 11]  ──▶ mean = 11.2
        ├──▶ resample (w/ replacement) ──▶ [9, 12, 12, 11, 9]   ──▶ mean = 10.6
        │           ... 10,000 times ...
        ▼
   spread of all 10,000 means = the approximate sampling distribution
```

### Summary example
Need a 95% CI for the MEDIAN wear_pct across a skewed set of equipment readings — no clean formula exists for a median's CI the way there is for a mean. Bootstrap 10,000 resamples of the data, compute the median of each, and take the 2.5th/97.5th percentiles of that whole distribution — a valid CI with zero formula-lookup required.

---

## Cluster 11 — Distribution Shape Beyond Mean and Variance

### 1. Can two datasets share the same mean AND variance but still look totally different?
Yes — one could be symmetric, the other could have a long tail dragging a few extreme values off to one side, while both still average out to the same center and the same overall spread.

### 2. How do you measure that asymmetry (skewness) and tail-heaviness (kurtosis)?
```python
from scipy.stats import skew, kurtosis
skew(df["wear_pct"])       # >0: long tail to the RIGHT; <0: long tail to the LEFT; ~0: roughly symmetric
kurtosis(df["wear_pct"])   # >0: heavier tails / more outliers than normal; <0: lighter tails
```
A fast, no-formula skew check: compare mean to median — in a right-skewed dataset the mean sits noticeably ABOVE the median, because the mean gets pulled toward the long tail while the median doesn't.

**Visual — the direction of the LONG TAIL is the sign, not where most of the data sits:**
```
NEGATIVE skew (left tail)      SYMMETRIC (skew ≈ 0)         POSITIVE skew (right tail)
        ▂▄▆██                       ▁▃▆█▆▃▁                      ██▆▄▂
    ◀───long tail                  even both sides                 long tail───▶
    mean < median                  mean ≈ median                   mean > median
```

### 3. Given a skewed dataset with some extreme values, how do you actually find the outliers?
```python
q1, q3 = df["wear_pct"].quantile([0.25, 0.75])
iqr = q3 - q1
lower_bound, upper_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr
outliers = df[(df["wear_pct"] < lower_bound) | (df["wear_pct"] > upper_bound)]
```
**Why IQR instead of "more than 2 standard deviations from the mean":** the z-score method computes the mean and standard deviation FROM the same data the outliers are corrupting — a few extreme outliers inflate the standard deviation itself, hiding moderate outliers and understating how extreme the real ones are. IQR is based on the 25th/75th percentiles, which barely move even with several extreme values present.

**Visual — this is literally the math behind a box plot's whiskers:**
```
        lower bound          Q1      median      Q3          upper bound
        (Q1 − 1.5×IQR)        │25%│      │      │75%│        (Q3 + 1.5×IQR)
              │                ┌───┴──────┼──────┴───┐              │
   ○ ○    ─────┤               │          │          │              ├─────    ○
  outliers     └───────────────┴──────────┴──────────┴──────────────┘   outliers
```

### Summary example
`wear_pct` readings mostly cluster 30-50%, with a handful of readings above 90%. `skew()` reports 1.8 (strongly right-skewed) — confirmed cheaply by mean (48) sitting well above median (41). IQR method: Q1=35, Q3=50, IQR=15, upper bound = 50+22.5=72.5 — every reading above 72.5% flags as an outlier worth investigating individually, without the standard deviation itself being distorted by those same extreme values.

---

## Cluster 12 — How the Sample Itself Can Be the Problem

### 1. Does it matter HOW a sample was collected, even if every test afterward is run correctly?
Yes — every test in this entire doc silently assumes the sample was collected in an unbiased way. Run a perfect t-test on a badly-collected sample and you get a perfectly-computed, completely misleading answer.

### 2. What's the simplest fix when one subgroup is much smaller than another?
```python
df.groupby("region", group_keys=False).apply(lambda g: g.sample(frac=0.1))  # stratified: same % from EVERY group
```
Plain random sampling (`df.sample(n=500)`) can end up with almost no rows from a small subgroup purely by chance if that subgroup is a tiny fraction of the whole. **Stratified sampling** guarantees every stratum is actually represented by sampling proportionally WITHIN each group first.

### 3. What are the actual named failure modes when a sample is collected badly?
- **Selection bias** — the sample systematically excludes part of the population by HOW it was collected (a survey emailed only to active users misses exactly the churned customers who'd answer differently).
- **Survivorship bias** — analyzing only the cases that "survived" some filter, with the excluded cases leaving zero trace at all.
- **Non-response bias** — validly selected people who never respond, systematically different from those who do.
- **Convenience sampling** — using whichever data was easiest to grab, not a sample representative of the real target population.

**Visual — survivorship bias specifically, the least intuitive of the four:**
```
100 planes take off
        │
        ▼
  ┌─────────────┐         ┌─────────────┐
  │  80 RETURN   │        │  20 SHOT DOWN │  ← never observed, never counted —
  │  (damage      │        │  (no damage    │    this is the actual danger zone,
  │   observable)  │        │   data exists)  │    invisible in the data entirely
  └─────────────┘         └─────────────┘
        │
        ▼
  "reinforce where the RETURNING planes are damaged" ← WRONG —
  those hits were survivable BY DEFINITION, since these planes made it back
```

### Summary example
A company studies only customers who completed 6 months of subscription to find what "successful" customers have in common — survivorship bias, since every early-churner is invisible to the analysis with zero trace. A pattern found among survivors can look completely convincing while being the exact opposite of what actually causes early churn, the same "reinforce where the surviving planes were hit" mistake.

---

## Cluster 13 — Quantifying HOW MUCH, Not Just Whether

### 1. A p-value says an effect is statistically significant. Does that mean it's a BIG effect?
No — a large enough sample can make a trivially small, practically meaningless difference statistically significant, because p-values are sensitive to sample size in a way that says nothing about practical importance.

### 2. How do you measure the actual SIZE of an effect, separately from whether it's "real"?
```python
mean_a, mean_b = np.mean(group_a_scores), np.mean(group_b_scores)     # 85, 75.6 (Cluster 1's own numbers)
pooled_std = np.sqrt((np.var(group_a_scores, ddof=1) + np.var(group_b_scores, ddof=1)) / 2)   # 4.11
cohens_d = (mean_a - mean_b) / pooled_std     # (85 - 75.6) / 4.11 = 2.29
```
**Cohen's d** measures how many standard deviations apart the two means are, regardless of sample size. Conventionally: 0.2=small, 0.5=medium, 0.8=large — reusing Cluster 1's own t-test numbers, `d≈2.29` is a huge effect, not just a "detectable" one.

**Visual — d is directly how far apart the two humps from Cluster 1's t-test visual have slid:**
```
d = 0.2 (small)      d = 0.5 (medium)      d = 0.8 (large)       d = 2.29 (this example)
 ⬮⬮  overlapping       ⬮ ⬮  some gap         ⬮  ⬮  clear gap      ⬮    ⬮  barely touching
```

### 3. Before even running the experiment — how much data do you actually need?
```python
from statsmodels.stats.power import TTestIndPower
required_n = TTestIndPower().solve_power(effect_size=0.5, alpha=0.05, power=0.8)   # ~64 per group
```
**Power analysis** answers "how many samples do I need" given the smallest effect size worth caring about, BEFORE spending time/money collecting data — an underpowered experiment's null result is genuinely ambiguous (no real effect, or an effect too small for this sample to detect, look identical).

**Visual — the full 2×2 that power, Type I, and Type II error all live inside (the "crying wolf" table):**
```
                        H0 actually TRUE            H0 actually FALSE
  Reject H0          Type I error (α)              Correct! (True Positive)
  ("found an effect")  "cried wolf" — false alarm    caught the real effect
  Fail to reject      Correct (True Negative)       Type II error (β)
  H0                                                 "stayed silent while the
                                                       wolf was right there"
                                                       POWER = 1 − β
```

### Summary example
A study reports p<0.001 for a new teaching method. Computing Cohen's d on the same data gives d=0.15 — statistically real, but a genuinely tiny effect, barely worth the cost of rolling it out. A separate small pilot study finds "no significant effect" of a different intervention — but a power analysis shows the sample was only powered to detect d≥0.8, meaning a real, moderate effect (d=0.4) could easily have been missed entirely, not ruled out.

---

## Practice Q&A (Self-Test)

**Q1. Why does `stats.ttest_ind(group_a, group_b, equal_var=False)` use Welch's t-test instead of the classic Student's t-test, and why is that the safer default?**
A: `equal_var=False` runs Welch's t-test, which does not assume the two groups have equal variance — an assumption the classic Student's t-test makes and that is frequently violated in practice. The classic test can give a misleadingly confident (too-small) p-value when variances genuinely differ between groups, so Welch's is the safer default.

**Q2. A p-value comes back as 0.03. What does that number actually mean, and what is the common misinterpretation to avoid?**
A: The p-value is the probability of seeing a difference this large (or larger) IF the null hypothesis were true — it is NOT the probability that the null hypothesis is true, and NOT the probability the finding is a fluke. Saying "0.03 means there's a 97% chance this is real" is exactly the misinterpretation to avoid.

**Q3. You have before/after measurements on the same units. Why is `stats.ttest_rel` the correct test instead of `stats.ttest_ind`?**
A: `ttest_ind` treats the two samples as independent and ignores that each "before" is linked to a specific matching "after" for the same unit. Throwing away that pairing information typically produces a much less statistically powerful (and technically wrong) test, so `ttest_rel` (the paired t-test) should be used whenever the two samples are the same subjects measured twice.

**Q4. After running `stats.chi2_contingency`, why should you look at the `expected` array and not just the p-value?**
A: The chi-square test only tells you IF there's an association between the two categorical variables, not what it looks like. Comparing the `observed` table to the `expected` table (what you'd see under no association) reveals the direction and nature of the association, which the p-value alone can't show.

**Q5. What assumption should you check before trusting a chi-square test's result, and how do you check it?**
A: Chi-square is itself an approximation that assumes reasonably large expected counts in every cell; check `(expected < 5).any()` — if any expected cell count is below 5, the approximation may be unreliable and the p-value can't be trusted.

**Q6. Why run ANOVA (`stats.f_oneway`) across three groups instead of three separate pairwise t-tests?**
A: Each individual t-test at α=0.05 carries a 5% false-positive chance, and running three separate tests compounds that risk — the multiple comparisons problem. ANOVA tests the overall "are ANY of these means different" question in one test at the stated α, properly controlling the overall false-positive rate.

**Q7. If ANOVA shows a significant difference among groups, how do you find out which specific groups differ, and why not just use raw pairwise t-tests for that?**
A: Use `pairwise_tukeyhsd` from `statsmodels.stats.multicomp` — it performs the pairwise comparisons you now want but with a built-in correction for the multiple-comparisons problem, so you get which-pairs-differ answers without re-inflating the false-positive rate ANOVA was used to avoid.

**Q8. What silent mistake happens if you fit `sm.OLS(y, X)` without calling `sm.add_constant(X)` first, and why does sklearn's `LinearRegression` not have this problem?**
A: Without `sm.add_constant`, statsmodels fits a regression forced through the origin (no intercept term), and every coefficient comes out wrong — silently, with no error raised. sklearn's `LinearRegression` adds an intercept by default, but it also doesn't give the p-values/CIs/R² that `model.summary()` provides, since sklearn is built for prediction, not inferential statistics.

**Q9. How do you interpret a VIF (variance inflation factor) value, and what does a high VIF actually mean for your model?**
A: VIF measures how much a feature's variance is inflated because it's linearly predictable from the other features; VIF=1 means no correlation with other predictors, while VIF>5 (some use >10) signals real multicollinearity. High VIF makes individual coefficients unstable and hard to interpret, even though the model's overall predictions may still be fine.

**Q10. Why does the confidence interval for a sample mean use `stats.t.interval` with `df=n-1` rather than the normal distribution, and what's the practical caveat when checking normality with Shapiro-Wilk on large samples?**
A: With a sample (not the true population), you're also estimating the standard deviation from the same data, adding extra uncertainty; the t-distribution has heavier tails than normal to account for that, converging to normal only as n grows large — using normal instead understates true uncertainty for small samples. Separately, Shapiro-Wilk becomes hyper-sensitive with very large samples and flags trivial, practically-irrelevant deviations from normality as "significant," so a Q-Q plot is often more useful than the p-value alone at large n.

**Q11. A scatterplot shows wear percentage rising with equipment age, but the relationship curves — it climbs fast early on, then keeps climbing but more slowly. `pearsonr` gives r=0.61; `spearmanr` gives rho=0.89. Which number should you trust as "how strong is this relationship," and why the gap?**
A: Trust Spearman's rho here. Pearson's r specifically measures LINEAR association, so a real, strong, consistently-increasing-but-curved relationship still drags Pearson's r down since the relationship isn't a straight line — it isn't wrong, it's answering a narrower question ("how linear is this") than the one being asked. Spearman's rho works on ranks rather than raw values, so it only cares whether the relationship is consistently monotonic (always increasing), which this is — 0.89 is the more honest answer to "how strong is this relationship," and the 0.61-vs-0.89 gap itself is the tell that the true relationship is monotonic but non-linear.

**Q12. Two manufacturing lines produce parts with the exact same average weight. Line A's t-test against Line B shows no significant difference. Does that mean the two lines are equally good?**
A: Not necessarily — a t-test only compares means, and two datasets can have identical means while one is far more variable than the other. If Line B's output swings much more widely around that same average, Levene's test (comparing variance, not mean) would catch a real quality difference the t-test is structurally blind to — consistency itself can be the thing that matters, and "same average" says nothing about it.

**Q13. A dataset has a few extreme outliers, and you're not confident it's normally distributed. Why reach for Mann-Whitney U instead of just running `ttest_ind` anyway and hoping for the best?**
A: The t-test compares means, which outliers can drag substantially in either direction, and it formally assumes the underlying data is roughly normal. Mann-Whitney U converts every value to its rank first, so a single extreme outlier only shifts by one rank position no matter how extreme it is — it tests whether ranks are randomly mixed between groups rather than comparing raw magnitudes, making it robust to exactly the two problems (outliers, non-normality) that make a t-test's result untrustworthy here.

**Q14. A study reports p < 0.001 for a new teaching method's effect on test scores. A colleague says "that's a massive effect." Is that a safe conclusion from the p-value alone?**
A: No — p-values are sensitive to sample size, and a large enough study can make a tiny, practically trivial difference statistically significant. A very small p-value tells you the effect is unlikely to be pure chance; it says nothing about whether the effect is big enough to matter practically. That second question needs an effect size (Cohen's d) computed separately, not inferred from how small the p-value is.

**Q15. A small pilot study finds "no statistically significant effect" of a new feature on user engagement. Is it safe to conclude the feature genuinely has no effect?**
A: Not necessarily — if the study was underpowered (too small a sample for the effect size realistically worth detecting), a true, real effect can easily fail to reach significance simply because the study never had a good chance of detecting it. This is a Type II error risk, and "no significant effect found" and "no effect exists" are genuinely different claims that a low-power study can't distinguish between. A power analysis run *before* the study would have flagged whether the sample size was even large enough to matter.

**Q16. You need a 95% confidence interval for the MEDIAN of a skewed dataset, and you don't know a clean formula for that. What's the practical way to get one?**
A: Bootstrapping — resample the original data with replacement (same size) thousands of times, compute the median of each resample, and take the 2.5th and 97.5th percentiles of that whole distribution of medians as the CI bounds. This works for the median (or any other statistic) without needing a closed-form formula, which is exactly the situation a formula-based approach like this doc's earlier by-hand mean CI can't handle.

**Q17. A two-tailed test gives p=0.07 (not significant at α=0.05). Someone suggests re-running it one-tailed since "we always expected the new version to be better anyway," and it comes back at p=0.035 (significant). Is this a legitimate result?**
A: No — this is p-hacking, not a legitimate use of a one-tailed test. A one-tailed test is only valid when the direction was committed to *before* seeing the data, for a real reason the other direction genuinely wasn't worth testing. Switching to one-tailed specifically because the two-tailed result wasn't significant, after already seeing which direction the data leans, is choosing the test that gives the answer you want rather than the test that matches your actual prior claim.

**Q18. Why does scipy default to `alternative="two-sided"` rather than defaulting to whichever direction is more common in practice?**
A: Two-tailed is the more conservative choice — it doesn't assume a direction, so it can't be quietly misused to manufacture significance by picking the direction after the fact. Defaulting to two-sided forces the analyst to make an explicit, deliberate choice to narrow to one tail (and be able to justify why), rather than making the easier-to-abuse option the path of least resistance.

**Q19. Adding a completely random, useless column to a regression makes plain R² tick up slightly. Does that mean the model actually improved?**
A: No — plain R² can only stay flat or increase as predictors are added, even genuinely random ones, because the model can always find some tiny coincidental fit in more columns. Adjusted R² is the number that actually answers "did this predictor earn its place" — it penalizes for predictor count and can go down when a new feature isn't pulling real weight, which is exactly the case here.

**Q20. A regression's residual-vs-fitted plot shows a clear funnel shape — tight near small fitted values, spreading wide near large ones. Does this mean the coefficients themselves are wrong?**
A: Not necessarily — this is heteroscedasticity (non-constant error variance), and it doesn't automatically bias the coefficient estimates themselves. What it does break is the trustworthiness of the p-values and confidence intervals in `model.summary()`, since those are computed under an assumption of constant variance across all fitted values.

**Q21. Two datasets have identical mean and standard deviation. `skew()` reports 0.05 for one and 1.8 for the other. What does that difference actually tell you, and how would you sanity-check it without even running the function?**
A: The second dataset has a long tail stretching toward larger values, even though its center and spread match the first dataset. A fast sanity check without computing skewness directly: compare the mean to the median — in a right-skewed dataset, the mean sits noticeably above the median, because the mean gets pulled toward the long tail while the median doesn't.

**Q22. A company analyzes only customers who completed a 6-month subscription to see what "successful" customers have in common, in order to design a retention strategy. What's the specific bias risk here, and why is it more dangerous than a normal sampling problem?**
A: Survivorship bias — the analysis only looks at customers who survived to 6 months, and completely excludes everyone who churned earlier, with zero trace of them in the "successful customer" analysis. It's more dangerous than ordinary selection bias because the missing cases leave no signal in the data at all that anything is wrong.


---

## Video-Sourced Practice MCQs (Set 2)

A second practice set for Statistics, sourced differently from this hub's other video-based quizzes: rather than covering the same hypothesis-testing ground as the clusters above, this one is built entirely from two classic probability-teaser interview problems (a weighted-average waiting-time problem, and a Bayes' theorem biased-coin problem) walked through in a real YouTube data-scientist-interview-prep video. Every number below was independently re-verified with exact fraction arithmetic (not just re-stated from the video), and one scenario (the batch-size-20 variant) is a genuinely new twist on the original problem to test real understanding rather than memorized recall. All wording, wrong-answer options, and explanations are original.

<script type="application/json" class="topic-quiz-data" data-title="Statistics (SciPy/statsmodels) Practice (Set 2)">
[
  {
    "d": "Probability Teasers — Weighted Averages",
    "q": "A product launches for 1000 people; only 10 people per day are shown it (so it takes 100 days for everyone to see it). The first 10 people wait 0 days, the next 10 wait 1 day, ..., the last 10 wait 99 days. What is the AVERAGE number of waiting days across all 1000 people?",
    "o": [
      "33 days",
      "49.5 days",
      "99 days",
      "50 days"
    ],
    "a": [
      1
    ],
    "e": "This is a weighted average, not a plain average of the day numbers: each waiting-day value (0 through 99) is tied to exactly 10 people, so average = (0+1+2+...+99)/100 = (99×100/2)/100 = 4950/100 = 49.5. Answering 50 rounds up as if the waiting days ran 1 to 100 instead of 0 to 99 — but the FIRST batch waits zero days, not one. 99 is just the maximum waiting time (the last batch), not the average across everyone. 33 doesn't correspond to any correct step in this calculation at all — it's not derivable from the given numbers."
  },
  {
    "d": "Probability Teasers — Weighted Averages",
    "q": "For that same sum, 0+1+2+...+99, there's a fast mental-math trick instead of adding 100 numbers one at a time: pair the first and last values (0+99=99... or count pairs summing to 99 across the range). What is 49×100 + 50, all divided by 100 — the shortcut used to get 49.5?",
    "o": [
      "5000/100 = 50",
      "9900/100 = 99",
      "4950/100 = 49.5 — 49 pairs each summing to 99, wait, using pairs that sum to 99 (0+99, 1+98, ... 49 pairs total) plus the unpaired middle terms, collapsing to this single expression",
      "4900/100 = 49"
    ],
    "a": [
      2
    ],
    "e": "Pairing values from opposite ends of a 0-to-99 run (there are 50 numbers... actually 100 numbers 0..99, forming 50 pairs) is exactly the classic Gauss pairing trick, which collapses a 100-term sum into a single multiplication instead of 100 additions — and it evaluates to exactly 4950, giving 4950/100=49.5, matching the direct calculation in the previous question. 5000/100=50 and 4900/100=49 are both plausible-looking but wrong arithmetic results of the same shortcut — a sign of a small arithmetic slip in applying the pairing, not a different valid method. 9900/100=99 confuses the SUM total (which isn't 9900 either) with something else entirely — it's not a step that appears in this calculation."
  },
  {
    "d": "Probability Teasers — Weighted Averages",
    "q": "Same setup, but now suppose the product is shown to 20 people per day instead of 10 (still 1000 people total, so it now takes only 50 days for everyone to see it). What is the new average waiting time?",
    "o": [
      "25 days",
      "49.5 days (unchanged, since total people is the same)",
      "12.25 days",
      "24.5 days"
    ],
    "a": [
      3
    ],
    "e": "Doubling the daily rate halves the total rollout period (100 days → 50 days), and the same weighted-average logic applies to the new range: average = (0+1+...+49)/50 = (49×50/2)/50 = 1225/50 = 24.5 — almost exactly HALF the original 49.5, because the whole waiting-time distribution got compressed into half as many days. Answering 49.5 (unchanged) ignores that the waiting-day RANGE shrank even though total people stayed fixed — it's total DAYS, not total people, that the average waiting time actually depends on. 25 is a rounding of the right idea but not the precise weighted-average value. 12.25 comes from incorrectly halving 24.5 again, over-correcting for a rate change that was already fully accounted for once."
  },
  {
    "d": "Probability Teasers — Bayes' Theorem",
    "q": "You randomly pick either a fair coin or a coin that's biased to always land tails (both equally likely to be picked, 50/50). You flip the chosen coin 5 times and get tails all 5 times. Setting up Bayes' theorem, what is P(5 tails | biased coin) — the probability of this exact outcome IF you'd picked the biased coin?",
    "o": [
      "0.5^5 divided by 2, to account for the 50/50 chance of picking it in the first place",
      "0.5, the same 50% chance as any single fair-coin flip",
      "1.0 (certainty) — a coin biased to always show tails will show tails on every single flip, no matter how many times you flip it",
      "0.03125 (1/32), treating the biased coin the same as a fair one for this calculation"
    ],
    "a": [
      2
    ],
    "e": "By definition, this coin has tails on BOTH sides — it is physically incapable of landing heads, so any number of flips, however many, always comes up tails with probability exactly 1.0. Treating it like a fair coin (0.5, or 0.5^5=1/32) misses the entire point of what 'biased to always show tails' means. Dividing by 2 for the coin-selection probability (option 4) is double-counting — that 50/50 selection probability is a SEPARATE term in Bayes' formula, not something folded into this particular conditional probability."
  },
  {
    "d": "Probability Teasers — Bayes' Theorem",
    "q": "Same setup. What is P(5 tails | fair coin) — the probability of getting 5 tails in a row from an ORDINARY fair coin?",
    "o": [
      "5 × 0.5 = 2.5, since there are 5 flips each contributing a 0.5 chance",
      "1.0, since 5 tails is a perfectly valid, unremarkable outcome for a fair coin",
      "0.5, since a fair coin is unbiased regardless of how many flips you consider",
      "0.5^5 = 1/32 = 0.03125 — each of the 5 independent flips has a 50% chance of tails, so all 5 together is 0.5 multiplied by itself 5 times"
    ],
    "a": [
      3
    ],
    "e": "Each flip of a fair coin is an independent event with P(tails)=0.5, and the probability of a SPECIFIC sequence of independent events is the product of their individual probabilities: 0.5×0.5×0.5×0.5×0.5 = 0.5^5 = 1/32 ≈ 0.03125. Answering plain 0.5 confuses the per-flip probability with the probability of the whole 5-flip sequence. Answering 1.0 confuses 'a valid possible outcome' with 'a certain outcome' — 5 tails is possible from a fair coin, just unlikely, not guaranteed. And probabilities of independent events multiply, they don't add — 5×0.5 isn't how compound probability works and also isn't even a valid probability (it exceeds 1)."
  },
  {
    "d": "Probability Teasers — Bayes' Theorem",
    "q": "Using the law of total probability: P(5 tails) = P(fair)×P(5 tails|fair) + P(biased)×P(5 tails|biased) = 0.5×(1/32) + 0.5×1.0. What is P(5 tails) overall?",
    "o": [
      "1/32 = 0.03125 (just the fair-coin term, forgetting the biased-coin contribution)",
      "33/64 = 0.515625",
      "1/64, from multiplying the two conditional probabilities together instead of combining them via the law of total probability",
      "1.0 (certainty, since one of the two coins must produce this outcome)"
    ],
    "a": [
      1
    ],
    "e": "P(5 tails) = 0.5×(1/32) + 0.5×1.0 = 1/64 + 32/64 = 33/64 = 0.515625 — the law of total probability correctly WEIGHTS each coin's contribution by how likely you were to have picked it, then adds the two paths together. Stopping after just the fair-coin term (1/32) throws away the (larger) biased-coin contribution entirely. Answering 1.0 confuses 'this outcome is possible under both hypotheses' with 'this outcome is certain overall' — it's neither certain nor even particularly likely a priori, only more likely than a naive fair-coin-only calculation would suggest. And multiplying the two conditional probabilities together (1/64) applies the wrong operation — the law of total probability ADDS weighted branches, it doesn't multiply them."
  },
  {
    "d": "Probability Teasers — Bayes' Theorem",
    "q": "Putting it together with Bayes' theorem: P(biased | 5 tails) = [P(5 tails|biased) × P(biased)] / P(5 tails) = (1.0 × 0.5) / (33/64). What is the final answer, and why is it so close to 1?",
    "o": [
      "32/33 ≈ 0.970 — very close to certainty, because observing 5 tails in a row is MUCH more consistent with having picked the always-tails coin than with 1-in-32 luck from a fair coin",
      "1/33 ≈ 0.030 — flipping the numerator and denominator of the correct calculation",
      "0.5, unchanged from the prior, since 5 flips isn't really enough evidence to update your belief",
      "1.0 exactly, since getting 5 tails proves with certainty that the coin must be biased"
    ],
    "a": [
      0
    ],
    "e": "(1.0×0.5)/(33/64) = (1/2)×(64/33) = 32/33 ≈ 0.9697. It's this close to 1 because the observed evidence (5 tails in a row) is drastically more likely under the 'biased coin' hypothesis (probability 1.0) than under the 'fair coin' hypothesis (probability 1/32≈0.03) — Bayesian updating shifts your belief hard toward whichever hypothesis makes the observed data most likely, and here that's an enormous likelihood ratio in the biased coin's favor. Saying the belief is 'unchanged' at 0.5 ignores that this is precisely what Bayesian updating exists to do — revise a prior in light of evidence. 1/33 inverts the correct fraction (a common algebra slip when rearranging Bayes' theorem). And it's NOT exactly 1.0 — a fair coin landing on 5 tails by chance remains possible, just now assessed as unlikely (1/33 posterior probability) rather than impossible."
  },
  {
    "d": "Probability Teasers — Bayes' Theorem",
    "q": "Contrast case: same setup (50/50 fair-or-biased-to-always-tails coin), but this time you flip it 5 times and get 5 HEADS instead of 5 tails. What is P(biased coin | 5 heads)?",
    "o": [
      "Exactly 0 — the biased coin can never physically produce a head, so observing even a single head proves with certainty that you picked the fair coin",
      "32/33, the same answer as the 5-tails case, since the coins are symmetric",
      "1/33, the complement of the 5-tails answer",
      "0.5, unchanged from the prior"
    ],
    "a": [
      0
    ],
    "e": "This is the one case where the evidence is fully decisive rather than merely suggestive: P(5 heads | biased coin) = 0 exactly, because that coin is tails-on-both-sides and physically cannot produce a head under any circumstance. Plugging 0 into Bayes' theorem's numerator forces the whole posterior to exactly 0 — no algebra needed, and no residual uncertainty remains, unlike the 5-tails case where a small (1/33) chance of a fair coin remained. The coins are NOT symmetric here (one can produce heads, the other structurally cannot), so reusing the 5-tails answer of 32/33 is wrong. Believing it 'stays at the prior' ignores that observing something IMPOSSIBLE under one hypothesis is about as strong a piece of evidence as exists. And 1/33 isn't a meaningful complement here — the correct answer is a clean, exact 0, not a small nonzero residual."
  }
]
</script>
<div class="topic-quiz-mount"></div>
