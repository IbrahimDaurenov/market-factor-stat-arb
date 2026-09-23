# `signals.py`

This module converts PCA residual spreads into trading signals and manages persistent long and short positions.

The final strategy uses the pipeline

```math
e_t
\rightarrow
s_t
\rightarrow
z_t
\rightarrow
p_t
```

where:

- $e_t$ is the PCA residual;
- $s_t$ is the cumulative residual spread;
- $z_t$ is its standardized value;
- $p_t$ is the resulting position state.

---

## Residual Spread

The previous module constructs the residual spread

```math
s_{t,i}
=
\sum_{j=0}^{H-1}
e_{t-j,i}
```

for every stock $i$.

The final strategy uses

```math
H=40
```

days.

A large positive value means that the stock has recently outperformed its PCA-implied component.

A large negative value means that it has underperformed.

The strategy tests whether these relative deviations mean-revert.

---

# Rolling Z-Score

To determine whether the current spread is unusual, it is standardized relative to its own recent history.

For stock $i$, define the historical mean

```math
\mu_{t,i}
=
\frac{1}{W}
\sum_{j=1}^{W}
s_{t-j,i}
```

and historical standard deviation

```math
\sigma_{t,i}
=
\sqrt{
\frac{1}{W-1}
\sum_{j=1}^{W}
(s_{t-j,i}-\mu_{t,i})^2
}
```

Then

```math
z_{t,i}
=
\frac{
s_{t,i}-\mu_{t,i}
}{
\sigma_{t,i}
}
```

The final model uses

```math
W=252
```

trading days.

---

## Why `shift(1)` Is Important

The implementation begins with

```python
past_spreads = spreads.shift(1)
```

This means the rolling mean and standard deviation for day $t$ use only

```math
s_{t-W},
\ldots,
s_{t-1}
```

and do not include $s_t$ itself.

Therefore the chronology is

```math
\text{past spreads}
\rightarrow
(\mu_t,\sigma_t)
\rightarrow
s_t
\rightarrow
z_t
```

This avoids using information from the current observation to define its own historical distribution.

---

# Z-Score Trading Rule

The final strategy uses

```math
z_{\mathrm{entry}}=2.5
```

If

```math
z_{t,i}<-2.5
```

the stock is a long candidate.

If

```math
z_{t,i}>2.5
```

the stock is a short candidate.

The intuition is mean reversion:

```math
\text{large negative deviation}
\rightarrow
\text{expect upward correction}
```

and

```math
\text{large positive deviation}
\rightarrow
\text{expect downward correction}
```

This is a trading hypothesis, not a mathematical consequence of PCA.

PCA only defines the residual.

The mean-reversion assumption is tested by the backtest.

---

# Position State

For every stock,

```math
p_{t,i}
\in
\{-1,0,+1\}
```

where

```math
+1=\text{long}
```

```math
-1=\text{short}
```

```math
0=\text{flat}
```

Positions persist across days rather than being rebuilt from zero every day.

---

# Z-Score Exit Rule

For a long position, the strategy exits when the spread returns to the center:

```math
z_{t,i}\geq0
```

For a short position, it exits when

```math
z_{t,i}\leq0
```

So the basic logic is

```math
-2.5
\rightarrow
\text{long entry}
\rightarrow
0
\rightarrow
\text{exit}
```

or

```math
+2.5
\rightarrow
\text{short entry}
\rightarrow
0
\rightarrow
\text{exit}
```

---

# Maximum Holding Period

A position is also closed if it remains open for too long.

If

```math
h_{t,i}
```

is the number of days the position has been held, then the position exits when

```math
h_{t,i}
\geq
H_{\max}
```

The final strategy uses

```math
H_{\max}=40
```

days.

This prevents a failed mean-reversion trade from remaining open indefinitely.

---

# Maximum Number of Positions

The strategy limits the number of simultaneous active positions.

Let

```math
A_t
=
\sum_i
\mathbf{1}
(p_{t,i}\neq0)
```

Then

```math
A_t
\leq
A_{\max}
```

The final model uses

```math
A_{\max}=10
```

---

# Ranking Simultaneous Signals

More than 10 stocks can sometimes satisfy the entry rule on the same day.

For the z-score strategy, signal strength is

```math
|z_{t,i}|
```

Candidates are sorted from largest to smallest absolute z-score.

Therefore the most extreme residual deviations are selected first.

---

# Exit Before Entry

`update_positions()` first processes existing positions.

Only after exits are handled does it search for new entries.

This order matters because closing positions can free capacity for new trades.

The sequence is

```math
\text{existing positions}
\rightarrow
\text{exits}
\rightarrow
\text{available slots}
\rightarrow
\text{new entries}
```

---

## No Same-Day Re-entry

If a stock exits on day $t$, it is stored in `exited_today`.

That stock cannot immediately reopen on the same day.

This avoids the sequence

```math
\text{exit}
\rightarrow
\text{immediate re-entry}
```

from one observation.

---

# Empirical Quantile Alternative

A second signal definition was tested without using z-scores.

Instead of standardizing the spread, the historical empirical distribution is used directly.

For each stock, calculate

```math
q_{low,t}
```

```math
q_{50,t}
```

and

```math
q_{high,t}
```

from the previous $W$ observations.

The implementation uses

```math
q_{low}=0.025
```

and

```math
q_{high}=0.975
```

so the entry thresholds correspond to the historical 2.5% and 97.5% tails.

---

## Quantile Entry Rule

Long entry:

```math
s_{t,i}<q_{low,t,i}
```

Short entry:

```math
s_{t,i}>q_{high,t,i}
```

---

## Quantile Exit Rule

Long positions exit when

```math
s_{t,i}\geq q_{50,t,i}
```

Short positions exit when

```math
s_{t,i}\leq q_{50,t,i}
```

Thus the historical median acts as the center of the distribution.

---

## Quantile Signal Strength

For simultaneous candidates, the code defines

```math
\text{strength}
=
\frac{
|s_t-q_{50}|
}{
q_{high}-q_{low}
}
```

The denominator normalizes the deviation by the historical width of the distribution.

This allows signals from different stocks to be compared on approximately the same scale.

---

## Quantiles vs Z-Scores

The empirical-quantile version was tested as a robustness experiment.

In the final backtest it produced:

- more active positions;
- higher turnover;
- higher transaction costs;
- lower Sharpe ratio.

Therefore the final model uses the z-score signal.

---

# AR(1) Diagnostic

An AR(1) model was also tested as an optional mean-reversion filter.

Importantly, the implementation does **not** fit AR(1) directly to the daily residual $e_t$.

It first creates the cumulative residual level

```math
S_t
=
\sum_{\tau\leq t}e_\tau
```

and then models

```math
S_t
=
\alpha
+
\phi S_{t-1}
+
\epsilon_t
```

---

## Estimating Phi

For the regression

```math
Y=\alpha+\phi X+\epsilon
```

the OLS slope is

```math
\phi
=
\frac{
\operatorname{Cov}(X,Y)
}{
\operatorname{Var}(X)
}
```

Here,

```math
X=S_{t-1}
```

and

```math
Y=S_t
```

so the rolling estimator is

```math
\hat\phi
=
\frac{
\operatorname{Cov}(S_{t-1},S_t)
}{
\operatorname{Var}(S_{t-1})
}
```

The intercept is

```math
\hat\alpha
=
\bar S_t
-
\hat\phi\bar S_{t-1}
```

---

## AR(1) Mean Reversion

For

```math
|\phi|<1
```

the AR(1) process is stationary.

Ignoring the intercept for intuition,

```math
E[S_{t+h}\mid S_t]
\approx
\phi^hS_t
```

When

```math
0<\phi<1
```

the effect of a deviation decays over time.

---

# Half-Life

Define half-life as the number of periods required for the expected deviation to fall to one half of its original size.

We require

```math
\phi^h=\frac{1}{2}
```

Taking logarithms,

```math
h\log(\phi)
=
\log(1/2)
```

so

```math
\boxed{
h_{1/2}
=
\frac{
\log(1/2)
}{
\log(\phi)
}
}
```

This is only meaningful in the implementation when

```math
0<\phi<1
```

---

## Why the AR(1) Filter Was Not Used

In the data, estimated $\phi$ was usually close to one.

There is also an important structural reason for this.

Since

```math
S_t
=
S_{t-1}+e_t
```

a cumulative sum of approximately zero-mean residuals naturally behaves similarly to a random walk.

A random walk corresponds approximately to

```math
\phi=1
```

Therefore finding $\phi$ close to one for the cumulative residual level is not strong evidence against or for short-term residual mean reversion.

Adding the AR(1) filter also reduced backtest performance.

For this reason, the final specification uses

```python
use_ar1_filter = False
```

The AR(1) model remains in the project as a diagnostic and a failed experiment.

---

# Optional AR Filter in Position Selection

If `phi_today` is supplied, an entry is accepted only when

```math
\phi_{\min}
<
\phi_t
<
\phi_{\max}
```

If no AR parameters are supplied,

```python
phi_today = None
```

the AR filter is skipped completely.

This is the configuration used by the final strategy.

---

# Final Signal Specification

The final strategy therefore uses:

```math
H=40
```

for residual accumulation,

```math
W=252
```

for z-score estimation,

```math
z_{\mathrm{entry}}=2.5
```

for entries,

```math
H_{\max}=40
```

for maximum holding time, and

```math
A_{\max}=10
```

for the number of simultaneous positions.

The final signal pipeline is

```math
e_t
\rightarrow
s_t
\rightarrow
z_t
\rightarrow
\text{entry/exit rules}
\rightarrow
p_t
```

AR(1) and empirical quantiles remain as alternative experiments rather than components of the final trading model.