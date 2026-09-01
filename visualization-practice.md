# Data Visualization Practice — Built as a Chain, Not a List

`import matplotlib.pyplot as plt, seaborn as sns` is assumed throughout.

Each cluster is one continuous thread. Every question builds on the answer before it, and each cluster closes with a worked summary example.

---

## Cluster 1 — Which Chart, and How to Actually Build the Figure

### 1. Which chart do you even reach for?
**Visual + memory hook — one question decides almost every chart choice: how many variables, and what type?**
```
 1 variable                2 variables                    3+ variables
 ───────────               ─────────────                  ──────────────
 numeric:                  both numeric:                  add COLOR for a
   histogram / KDE           scatter (+ trend line)        3rd numeric or
 categorical:               1 numeric + 1 categorical:     categorical variable
   bar chart                 box/violin plot (compare      on top of any
                              distributions across          2-variable chart
                              groups)                       above ("small
                            both categorical:                multiples" —
                              heatmap of counts, or          one panel per
                              grouped/stacked bar            category — is
                            one numeric OVER TIME:           the other option)
                              line chart, never bars
```
**Remember it as "count the variables, then name their types."**

A line chart on a categorical x-axis is wrong — it implies a connection between categories that doesn't actually exist. A scatterplot needs two genuinely numeric axes.

The moment you need a 3rd variable, reach for color or small multiples first. Both stay flat and easy to read accurately.

### 2. Building a figure with more than one panel
```python
fig, axes = plt.subplots(1, 2, figsize=(10, 4))   # 1 row, 2 columns; axes is an ARRAY of Axes objects
axes[0].plot([1, 2, 3], [4, 5, 6])
axes[1].scatter([1, 2, 3], [6, 5, 4])
fig.tight_layout()      # prevents overlapping titles/labels between subplots
plt.show()
```
The object-oriented API — explicit `fig`/`ax` objects — is what you need the moment you have more than one subplot, or need to modify a specific axes later.

The pyplot "state machine" API (`plt.plot`, `plt.xlabel`, ...) implicitly acts on "whichever axes was last active." That gets confusing and error-prone once you have multiple subplots.

### 3. Making a chart actually readable — labels
```python
fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_xlabel("Training step")
ax.set_ylabel("Loss")
ax.set_title("Loss vs. training step")
ax.legend(["run A"])     # or pass label="run A" to plot() and just call ax.legend()
```
A chart with no axis labels or units is often unusable to anyone but the person who just made it. Five minutes later, even they might not remember what it shows.

In an interview, or in a real deliverable, an unlabeled chart reads as unfinished work — not a stylistic choice.

### 4. Saving a chart without losing anything at the edges
```python
fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
fig.savefig("chart.png", dpi=150, bbox_inches="tight")
```
Without `bbox_inches="tight"`, saved figures often clip axis labels or legends that sit outside the default bounding box.

This one argument fixes the single most common complaint: "my saved chart cut off the label."

### Summary example
Say you're building a two-panel comparison for a report.

1. `fig, axes = plt.subplots(1, 2, figsize=(10,4))`.
2. Label each panel with `set_xlabel`/`set_ylabel`/`set_title`, so it stands alone even if it gets screenshotted out of context.
3. Save it: `fig.savefig("comparison.png", dpi=150, bbox_inches="tight")`.

That last argument keeps the legend on the right panel from getting silently clipped off in the exported file.

---

## Cluster 2 — Showing a Single Distribution

### 1. Plotting a histogram — and what goes wrong with a careless bin count
```python
fig, ax = plt.subplots()
ax.hist(data, bins=30, edgecolor="white")
```
Too few bins — 5, say — can hide a bimodal distribution, by averaging two peaks into one bar. Too many — 500, say — makes the histogram noisy and hard to read.

There's no universally correct number. Always try a couple of values. `bins="auto"` is a reasonable default, based on the Freedman-Diaconis rule — but don't accept the first result uncritically either way.

### 2. Showing the distribution's SHAPE as a smooth curve instead
```python
sns.kdeplot(data=df, x="wear_pct", hue="depot", fill=True, common_norm=False)
```
`common_norm=False` matters here. Leave it `True`, and multiple groups' KDEs get normalized so their AREAS sum to one shared total — that distorts the comparison if the group sizes differ a lot.

Set it `False`, and each group's density gets normalized independently. Now you're comparing SHAPE, not shape-weighted-by-sample-count.

### 3. Comparing a distribution ACROSS categories, not just looking at one
```python
sns.boxplot(data=df, x="depot", y="wear_pct")       # shows median, IQR, and outliers per category
sns.violinplot(data=df, x="depot", y="wear_pct")     # ALSO shows the full shape of the distribution
```
A boxplot summarizes a distribution with 5 numbers — min, Q1, median, Q3, max-ish. That's the same IQR machinery as `stats-scipy-practice.md`'s outlier-detection cluster. Two distributions with the same quartiles can look identical as boxplots, even if one is bimodal and the other isn't.

A violin plot's width shows the actual density shape instead — the same KDE idea from question 2, just laid sideways, per category. It catches bimodality a boxplot would hide.

The tradeoff: violin plots get harder to read with small sample sizes, because the density estimate itself gets noisy.

### Summary example
Say you're looking at wear percentage across 3 depots. One of them secretly mixes two distinct equipment generations together — a bimodal distribution.

1. A boxplot shows all 3 depots as similarly-shaped single boxes. It completely hides the bimodality.
2. Switch to `sns.violinplot` on the same data, and the two-humped shape in that one depot shows up immediately.

Same underlying data. A materially different amount of information visible.

---

## Cluster 3 — Relationships Between Variables

### 1. Showing the relationship between two continuous variables, plus a third
```python
sns.scatterplot(data=df, x="age_days", y="wear_pct", hue="depot", size="mileage")
```
Encode a categorical variable as color (`hue`), and a continuous one as point size (`size`), and you get 4 variables into one 2-D scatterplot: x, y, category, and magnitude. Genuinely useful.

Don't stack more than 2-3 encodings, though — past that, the chart becomes unreadable. That's a real limit, not a stylistic preference. It's the same "3+ variables" branch from Cluster 1's decision tree.

### 2. Showing the OVERALL trend, not just the raw cloud of points
```python
sns.regplot(data=df, x="age_days", y="wear_pct", scatter_kws={"alpha": 0.4}, line_kws={"color": "red"})
```
`alpha` — transparency — on the scatter points matters specifically when you have many overlapping points. A solid scatter hides how DENSE a region actually is.

Semi-transparent points let overlapping regions visually darken instead. That shows density directly, on the same plot, without needing a separate density chart.

### 3. Seeing EVERY pairwise relationship among many numeric columns at once
```python
corr = df.select_dtypes("number").corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1)
```
Without `center=0` and fixed `vmin=-1, vmax=1`, the color scale auto-ranges to the DATA's actual min and max.

Say a correlation heatmap has no strong negative correlations. Auto-ranging would then render its weak positive correlations in the same "cool" color you'd expect for a negative one.

Fixing the scale to correlation's TRUE possible range — -1 to 1, with 0 as the neutral midpoint — makes color comparisons consistent and honest. That holds even across different heatmaps compared side by side.

**Visual + memory hook — auto-ranged color vs. fixed-range color, same underlying numbers:**
```
Auto-ranged (data min=-0.1, max=0.6)      Fixed (-1 to 1, center=0)
  -0.1 ████ (rendered as "coolest")        -0.1 ▓▓░░ (rendered as barely negative)
   0.3 ▓▓▓▓ (rendered as "medium")          0.3 ▓▓▓▓ (rendered as mild positive)
   0.6 ░░░░ (rendered as "warmest")          0.6 ████ (rendered as strong positive)

  a WEAK negative correlation (-0.1) looks     the SAME -0.1 now correctly reads as
  as extreme as this dataset's coolest color   "barely different from zero"
```
**Remember it as:** an auto-ranged color scale answers "what's coolest/warmest IN THIS DATASET," not "what's actually strong or weak" — fixing the range to correlation's true bounds (-1 to 1) is what makes the color itself mean something absolute, comparable across any two heatmaps you ever build.

### Summary example
Say you're comparing feature correlations across two different model datasets, side by side.

1. With auto-ranged color scales, dataset A's strongest correlation (0.4) and dataset B's strongest correlation (0.9) could both render as the same "hottest" red. That visually implies equally strong relationships — which is wrong.
2. Fix both heatmaps to `vmin=-1, vmax=1, center=0` instead. Now the color itself is an honest, comparable signal: A's 0.4 renders visibly cooler than B's 0.9, exactly reflecting the real difference in strength.

---

## Cluster 4 — Time Series Specifically

### 1. Plotting a time series properly
```python
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df["date"], df["wear_pct"])
ax.tick_params(axis="x", rotation=45)     # avoids overlapping/unreadable date labels
fig.tight_layout()
```
Bars imply discrete, unrelated categories. A line implies a continuous underlying process connecting each point — which is correct, for most time series.

Using bars for genuinely continuous time series data is a common chart-type mismatch. It subtly misleads the reader. Same "count the variables, name their types" logic from Cluster 1, just applied to the time axis specifically.

### 2. Showing MULTIPLE entities' time series on the same axes
```python
fig, ax = plt.subplots(figsize=(10, 4))
for unit, group in df.groupby("unit"):
    ax.plot(group["date"], group["wear_pct"], label=unit, marker="o", markersize=3)
ax.legend(title="Unit")
```
Group first — the same `groupby` split from `pandas-practice.md` — then plot one line per group, with a legend.

That's what keeps multiple entities visually distinguishable. Skip it, and you get an unreadable tangle of unlabeled lines instead.

### Summary example
Say you're plotting wear trends for 5 locomotives on one chart.

1. `groupby("unit")` splits the data by locomotive.
2. Each unit gets its own labeled line, via the loop.
3. The legend makes it possible to trace any single locomotive's trajectory.

Skip the groupby, and a single `ax.plot(df["date"], df["wear_pct"])` would instead draw one tangled, meaningless zigzag — jumping between different units' unrelated values at each date.

---

## Cluster 5 — Small Multiples and Consistent Styling

### 1. One panel PER category, side by side, instead of overlaying groups on one chart
```python
g = sns.FacetGrid(df, col="depot", col_wrap=3, height=3)
g.map_dataframe(sns.histplot, x="wear_pct")
g.set_titles("{col_name}")
```
`FacetGrid` beats a manual loop of subplots for one reason: it automatically handles consistent axis SCALES across panels, so they're honestly comparable at a glance. It also handles legend and title placement for you.

Worth learning specifically because "same chart, once per category, on a shared scale" is one of the most common real EDA needs. It's exactly what Cluster 1's "3+ variables" branch pointed at, as the alternative to over-stacking color encodings.

### 2. Keeping every chart in a notebook looking like one coherent piece of work
```python
sns.set_theme(style="whitegrid", palette="deep")
```
Set this once, at the top of a notebook. Every chart in the report then looks like one coherent piece of work, instead of a patchwork of default-matplotlib and default-seaborn charts pasted together inconsistently.

### 3. Making sure the color choices are actually readable by everyone
```python
sns.set_palette("colorblind")     # seaborn ships a palette specifically designed to be distinguishable
```
Roughly 1 in 12 men have some form of color vision deficiency. A red/green categorical distinction — a very common default choice — is exactly the pairing most likely to be indistinguishable to them.

The `"colorblind"` palette avoids problematic pairings by design.

### Summary example
Say you're building a report with 6 different EDA charts.

1. Set `sns.set_theme(...)` and `sns.set_palette("colorblind")` once, at the very top.
2. Every chart after that — a `FacetGrid` panel-per-depot histogram, a correlation heatmap, a time series plot — shares the same visual language automatically.

No need to repeat style arguments on every single plotting call, and every chart stays distinguishable to a colorblind reviewer.

---

## Cluster 6 — Honest and Precisely Annotated Charts

### 1. The classic way a bar chart can accidentally — or deliberately — mislead
```python
fig, ax = plt.subplots()
ax.bar(["A", "B"], [98, 99])
ax.set_ylim(0, 100)     # start bars at a TRUE zero baseline, not wherever the data happens to start
```
A bar chart with a y-axis starting at 95, instead of 0, can make a 1-point difference look like a 20x difference visually.

Bar charts specifically encode value as HEIGHT from a baseline. That baseline MUST be zero, or the chart isn't honest.

**Visual + memory hook — same two numbers, two very different visual stories:**
```
Truncated axis (starts at 95)       Honest axis (starts at 0)
   B █████████ 99                     B █
   A ████████  98                     A █           (98 vs 99 — barely different)
   ──────────────                     ──────────────
   95    97    99                     0    50    100
   looks like a HUGE gap              looks like what it actually is:
                                        nearly identical
```
**Remember it as:** this rule is specific to BARS, because bars claim "height = value, measured from zero" as their whole visual language — a line chart zooming into a narrow y-range is a different, usually legitimate choice, since a line's slope (not its height from zero) is what it's actually communicating.

### 2. Pointing at one SPECIFIC point on a chart, without cluttering it
```python
fig, ax = plt.subplots()
ax.plot(x, y)
ax.annotate("anomaly here", xy=(peak_x, peak_y), xytext=(peak_x, peak_y + 5),
            arrowprops=dict(arrowstyle="->"))
```
`xy` is the point being pointed AT — the real data location. `xytext` is where the TEXT label itself gets drawn.

Separating the two lets you place the label somewhere legible, away from a busy part of the chart, while an arrow still points precisely at the actual data point.

### 3. When the data spans several orders of magnitude — does a normal linear axis still work?
```python
fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_yscale("log")
```
No. Data spanning several orders of magnitude — company revenues from thousands to billions, say — is unreadable on a linear axis. The small values compress into an indistinguishable line near zero.

Log scale fixes that. It's also the right choice for visualizing multiplicative or percentage change, because on a log scale, equal visual DISTANCE means equal RATIO — not equal absolute difference.

That's exactly right for a training loss curve. An early drop from 4.0 to 2.0, and a later drop from 0.4 to 0.2, are the SAME proportional improvement. They should look visually equivalent, and on a log scale, they do.

### Summary example
Say you're presenting a "99% vs 98% accuracy" comparison honestly.

1. A bar chart MUST use `set_ylim(0, 100)`. Anything else visually lies about the magnitude of a 1-point difference.

Separately, say you're plotting a training loss curve that drops from 4.0 to 0.04 over training.

2. That needs `set_yscale("log")` specifically. On a linear scale, the early dramatic drop swallows all the visual detail of what happens later. On a log scale, both the early drop and the later fine-tuning improvements stay visible on the same axis.

---

## Practice Q&A (Self-Test)

**Q1. Why use `fig, axes = plt.subplots(...)` (the object-oriented API) instead of calling `plt.plot()` directly, especially once you have more than one subplot?**
A: The pyplot "state machine" API implicitly acts on whichever axes was last active, which gets confusing and error-prone the moment you have multiple subplots. The object-oriented API with explicit `fig`/`ax` objects lets you address and modify a specific axes directly and unambiguously.

**Q2. When saving a figure with `fig.savefig(...)`, why include `bbox_inches="tight"`?**
A: Without it, saved figures often clip axis labels or legends that sit outside the default bounding box. This one argument fixes the single most common "my saved chart cut off the label" complaint.

**Q3. Why isn't histogram bin count a "just pick a number" decision — what specifically goes wrong at the extremes?**
A: Too few bins (e.g. 5) can hide a bimodal distribution by averaging two peaks into one bar; too many (e.g. 500) makes the histogram noisy and hard to read. There's no universally correct number — try a couple of values, or use `bins="auto"` as a reasonable Freedman-Diaconis-rule-based default.

**Q4. When would you choose a violin plot over a boxplot for comparing distributions across categories, and what's the tradeoff?**
A: A boxplot summarizes with roughly 5 numbers (min/Q1/median/Q3/max-ish) and can make a bimodal distribution look identical to a unimodal one with the same quartiles. A violin plot's width shows the actual density shape, catching bimodality a boxplot would hide — at the cost of being harder to read with small sample sizes, where density estimates get noisy.

**Q5. In `sns.scatterplot(data=df, x="age_days", y="wear_pct", hue="depot", size="mileage")`, how many variables are being encoded at once, and why shouldn't you keep adding more encodings?**
A: Four variables are encoded: x, y, a categorical variable via color (`hue`), and a continuous magnitude via point size (`size`). Stacking more than 2-3 encodings onto one chart is a real readability limit, not a stylistic preference — the chart becomes unreadable beyond that.

**Q6. Why does the correlation heatmap example set `center=0` with fixed `vmin=-1, vmax=1` instead of letting the color scale auto-range to the data?**
A: Auto-ranging to the data's actual min/max means a heatmap with no strong negative correlations would render weak positive correlations in the same "cool" color as if they were negative. Fixing the scale to the true possible range of correlation (-1 to 1) with 0 as the neutral midpoint makes color comparisons consistent and honest, including across different heatmaps.

**Q7. Why is a line chart the correct choice for time series data rather than a bar chart?**
A: Bars imply discrete, unrelated categories, while a line implies a continuous underlying process connecting each point, which is correct for most time series. Using bars for genuinely continuous time series data is a common chart-type mismatch that subtly misleads the reader about what the data represents.

**Q8. In `ax.bar(["A", "B"], [98, 99])`, why is `ax.set_ylim(0, 100)` essential, and why doesn't the same zero-baseline rule apply to line charts?**
A: Bar charts encode value as height from a baseline, so a y-axis starting anywhere other than zero (e.g. 95) can make a 1-point difference look like a 20x difference. This rule is specific to bars — a line chart zooming into a range doesn't encode value as height from zero, so it's a different, usually legitimate choice.

**Q9. In `ax.annotate("anomaly here", xy=(peak_x, peak_y), xytext=(peak_x, peak_y + 5), ...)`, what's the difference between `xy` and `xytext`, and why does separating them matter?**
A: `xy` is the point being pointed AT — the actual data location — while `xytext` is where the text label itself is drawn. Separating them lets you place the label somewhere legible, away from a busy part of the chart, while the arrow still points precisely at the real data point.

**Q10. Why would you call `sns.set_palette("colorblind")` rather than relying on seaborn's default categorical colors?**
A: Roughly 1 in 12 men have some form of color vision deficiency, and a default red/green categorical distinction is exactly the pairing most likely to be indistinguishable to them. The `"colorblind"` palette is specifically designed to avoid problematic color pairings.
